#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text (help(module))
"""
Contamination indication from the handset records.

Question: did the 50 W jamming transmitted from Test Area 1 reach the recording
position at Test Area 3?

Inputs. The Android GnssLogger .json files; the transmission plan and the test
catalogue through the timeline script (its parser, its catalogue loader and its power rule
effective_power(), so the schedule has a single source of truth); the timeline script's
session inventory (ground_truth/session_ground_truth.csv) and time-base table
(ground_truth/time_base_hypotheses.csv, which names the plan's time zone); and, for
verification only, each session's OBD-II CSV, read solely to mark when the
recording vehicle was moving (the exclusion rule allows the odometer for
labelling and verification, never as a feature). It opens no .sbf, .ubx or .bin
file. Helpers shared with the receiver-status script come from codes/common/helpers.py.

DIRECTION FIXED IN ADVANCE. The Android automatic gain control field reports higher
gain, or lower input power, as a more positive number. Strong in-band interference
therefore drives level_db MORE NEGATIVE. A shift in the opposite direction is
evidence against the hypothesis, not for it, and is never converted to an
unsigned magnitude.

PRIMARY EVIDENCE. Test 1.8.2 is a pyramid: fifteen five-minute steps, jamming for
three minutes of each, with bands added one at a time and then removed. Of the
eight bands it cycles through, the handsets observe four, and the schedule
predicts a specific nesting for them. That prediction is stated before the data
is examined and is parsed from the transmission plan's own comment text.

LIVE READINGS. A gain-control reading is "not live" (frozen) when its band has no
raw satellite measurement in that epoch AND its value has stayed unchanged for at
least 30 s. Every test uses live readings; the same tests on
all readings are written as a sensitivity table.

Outputs
  Data/processed/ground_truth/handset_agc_timeseries.csv
  Data/processed/ground_truth/handset_epoch_summary.csv
  Data/processed/ground_truth/handset_band_structure.csv
  Data/processed/ground_truth/handset_level_comparison.csv
  Data/processed/ground_truth/handset_within_capture_comparison.csv
  Data/processed/ground_truth/handset_transition_timing.csv
  Data/processed/ground_truth/handset_positive_control.csv
  Data/processed/ground_truth/handset_control_phases.csv        (descriptive only)
  Data/processed/ground_truth/handset_control_onset.csv         (descriptive only)
  Data/processed/ground_truth/handset_sensitivity_all_readings.csv
  Data/processed/ground_truth/pyramid_prediction.csv
  Data/processed/ground_truth/pyramid_observed.csv
  Data/processed/ground_truth/session_distance_to_sender.csv
  Data/processed/ground_truth/free_space_ratio.csv
  Graph/distance_to_transmitter.png
  Graph/agc_level_timeseries.png
  Graph/level_distributions.png
  Graph/satellites_and_cn0.png
  Graph/pyramid_predicted_vs_observed.png
  Graph/positive_control.png
"""

from __future__ import annotations  # future import: hints kept as text, so "int | None" works on older Pythons

import contextlib  # standard-library context managers: redirect_stdout silences the sensitivity run again of the tests
import importlib.util  # standard-library tools for loading a .py file by its path (used to load the timeline script)
import io  # standard-library in-memory text stream, the place the silenced printout goes
import json  # standard-library JSON reader, used to parse the handset GnssLogger files
import math  # standard-library maths: log10 for the free-space ratio in decibels
import os  # standard-library operating-system access: sets matplotlib's cache folder
import random  # standard-library random numbers; only seeded here, nothing draws from it
import re  # standard-library regular expressions, used on file names
import sys  # standard-library system access: sys.modules for the loaded module
from datetime import datetime, timedelta, timezone  # date-time classes: instants, durations, the UTC zone
from pathlib import Path  # Path objects for building and reading file paths

import numpy as np  # third-party numerical library under the short name np: arrays, medians, percentiles
import pandas as pd  # third-party table library under the short name pd: DataFrames, merges, CSV output

# Path: codes/.mplconfig, the folder matplotlib keeps its configuration and font cache in, inside the project folder
MPLCONFIG = Path(__file__).resolve().parent / ".mplconfig"
MPLCONFIG.mkdir(exist_ok=True)  # creates the folder if missing; exist_ok=True means no error when it exists
os.environ["MPLCONFIGDIR"] = str(MPLCONFIG)  # os.environ: tells matplotlib, imported next, to use that folder
import matplotlib  # third-party plotting library, imported first so its back end can be chosen
matplotlib.use("Agg")  # picks the "Agg" back end: draws straight to image files, no screen window needed
import matplotlib.dates as mdates  # matplotlib's date helpers: datetime <-> plot number, time-axis ticks
import matplotlib.pyplot as plt  # plotting interface under the short name plt; must come after use("Agg")
from matplotlib.patches import Patch  # coloured-box shape used to build custom legend entries

# helpers shared with the receiver-status script: one copy of each, in codes/common/helpers.py next to this script
from helpers import (G50, GBL, GHI, GUNK, break_gaps, cliffs_delta, die, haversine_m, iqr,
                            # continuation of the from-import: further names loaded from the helpers module (schedule, pyramid, saving, shading, tagging, time axis)
                            load_schedule, parse_pyramid, save_fig, shade_windows, tag_times, time_axis)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

SEED = 20260914  # constant: the project's fixed seed; nothing in this script is random since cliffs_delta is exact
random.seed(SEED)  # seeds Python's random generator (not used by this script, fixed on principle)
np.random.seed(SEED)  # seeds NumPy's global generator (likewise not used, fixed on principle)

# the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "Data"  # the '/' operator on a Path joins path parts: the project folder/Data
DAY = DATA / "Day1_14092026"  # the raw capture-day folder; the handset folders sit inside it
PROCESSED = DATA / "processed"  # where this script's tables are written, and the timeline script's tables are read
GRAPH = PROJECT / "Graph"  # where figures 03 to 08 are written
TIMELINE_SCRIPT = PROJECT / "codes" / "1_ground_truth" / "01_timeline.py"  # the timeline script, whose parser and rules are reused

TZ_UTC = timezone.utc  # the UTC time-zone object, the internal time base
TEST_DAY = "2026-09-14"  # the capture date as text, used in printed messages

# Appendix A, Table 1: survey point SENDER, marked "FFI Jammer".
# The only named jamming transmitter in the survey table, on high ground
# (ellipsoidal height 397.28 m) at the northern end of the range.
TX_SENDER = (69.2788429, 16.0060623, 397.28)  # tuple: latitude, longitude (degrees), height (m) of the jammer site
TX_NAME = "SENDER (FFI Jammer), Appendix A Table 1"  # label used in printed text (the Figure 3 title now carries its own fixed wording)
# Appendix A, description of Test Area 3: the motorcade start point. A named landmark only: the centre of the
# plausibility radius below, never "the recording position" (that is measured per session from the handset fixes).
STAVE = (69.212187, 15.858559)  # tuple: latitude, longitude of the Stave landmark

# float constant (watts); the old trailing comment gives its meaning
HIGH_POWER_W = 1.0          # "above 1 W"
# float constant (watts): a Test Area 1 test at or above 40 W joins the "50 W" group
FIFTY_W_MIN = 40.0          # what counts as the 50 W group
# float constant (km): fixes beyond this distance from the Stave landmark are marked implausible
PLAUSIBLE_RADIUS_KM = 25.0  # a fix further than this from Stave is not believable
# float constant (km/h): consecutive fixes implying a faster speed are marked implausible
MAX_SPEED_KMH = 200.0       # a 1 s step implying more than this is not believable
# float constant (m): threshold on the path length summed over a 60 s window within one capture
STATIONARY_M = 150.0        # movement below this over a minute counts as stationary
FROZEN_RUN_S = 30.0  # seconds: an unchanged value lasting at least this long, with no measurement, is not live
MAX_GAP_S = 5.0  # seconds: plotted lines break where consecutive 1 Hz readings are further apart than this
SPOOF_RANGES_M = (30, 50, 100, 200)  # spoofer ranges in metres for the free-space comparison

# Frequencies the handsets report gain control on, and the plan band names that
# correspond to them. The plan cycles through eight bands; these are the four
# the handsets can see. B1I's centre is 1561.098 MHz.
BAND_LABEL = {  # dict literal: band centre frequency in hertz -> long label for tables
    1176450000: "L5 / E5a / B2a (1176.45 MHz)",  # dict entry: the L5 / E5a / B2a centre frequency and its long label
    # dict entry: B1I's centre frequency in hertz -> its long label; 1561.098 MHz is the true centre
    1561098000: "B1I (1561.098 MHz)",
    1575420000: "L1 / E1 / B1C (1575.42 MHz)",  # dict entry: the L1 / E1 / B1C centre frequency and its long label
    1602000000: "G1 GLONASS (1602 MHz)",  # dict entry: the GLONASS G1 centre frequency and its long label
}
BAND_SHORT = {1176450000: "L5", 1561098000: "B1I",  # dict: frequency in hertz -> short band name
              1575420000: "L1", 1602000000: "G1"}  # dict entry: short names of the L1 and GLONASS G1 bands
PLAN_BAND_TO_HZ = {"L1": 1575420000, "G1": 1602000000,  # dict: band name as written in the plan -> frequency
                   # dict entries of the same lookup: plan band names L5 and B1I -> their centre frequencies in hertz
                   "L5": 1176450000, "B1I": 1561098000}
# GLONASS is frequency-division: the Samsung resolves individual FDMA channels.
# They are one physical band and are collapsed onto the G1 centre frequency.
GLONASS_LO, GLONASS_HI = 1597000000, 1606000000  # tuple assignment: the frequency range treated as GLONASS G1
BAND_ORDER = [1575420000, 1602000000, 1176450000, 1561098000]  # list: panel order L1, G1, L5, B1I in the figures
Y_AGC = "AGC level\n(dB, device scale)"  # y-axis label for gain control: plain words, not the column name

# Okabe-Ito
C_ORANGE, C_SKY, C_GREEN = "#E69F00", "#56B4E9", "#009E73"  # tuple assignment of three hex colour strings
C_BLUE, C_VERMILLION, C_PURPLE = "#0072B2", "#D55E00", "#CC79A7"  # three more palette colours
C_GREY, C_BLACK = "#999999", "#000000"  # grey for grids and frozen readings, black for edges

NOTES_OUT: list[str] = []  # annotated empty list: findings worth noting, printed at the end
ANOMALIES: list[str] = []  # annotated empty list: things that look wrong, printed at the end


def note(msg: str) -> None:  # function: records a finding for the end-of-run notes
    NOTES_OUT.append(msg)  # list.append adds the message


def flag(msg: str) -> None:  # function: records an anomaly rather than smoothing it over
    ANOMALIES.append(msg)  # list.append adds the message


# --------------------------------------------------------------------------
# 1. Schedule, from the timeline script's parser
# --------------------------------------------------------------------------

def load_section1():  # function: loads the timeline script as a module so its plan parser can be reused, not copied
    # a module "spec" says where to load from; needed as the name 01_... starts with a digit, so no plain import
    spec = importlib.util.spec_from_file_location("timeline", TIMELINE_SCRIPT)
    mod = importlib.util.module_from_spec(spec)  # creates an empty module object from that spec
    sys.modules["timeline"] = mod  # registers it under the name "section1", as a normal import would
    spec.loader.exec_module(mod)  # runs the timeline script's top-level code (constants, functions); its main() is not run
    return mod  # hands back the loaded module


# --------------------------------------------------------------------------
# 3. Handset parsing
# --------------------------------------------------------------------------

def band_key(hz) -> int | None:  # function: a reported frequency -> one of the four band centres, or None
    """Collapse a reported carrier frequency onto one of the four physical bands."""
    hz = int(round(float(hz)))  # float() accepts text or numbers; round() then int() gives whole hertz
    if GLONASS_LO <= hz <= GLONASS_HI:  # if: a chained comparison, inside the GLONASS FDMA range
        return 1602000000  # every GLONASS channel is mapped to the G1 centre
    for k in BAND_LABEL:  # for loop over the four band centres (iterating a dict gives its keys)
        if abs(hz - k) <= 2_000_000:  # if: within 2 MHz of that centre (underscores are digit separators)
            return k  # that band
    return None  # no band matched


# function: reads every handset file; returns band rows, epoch rows, per-file facts and collapse statistics
def parse_handsets() -> tuple[pd.DataFrame, pd.DataFrame, list[dict], dict]:
    # generator filtered by a regex: .json files one folder down whose folder is NOT a session folder, i.e. phones
    files = sorted(p for p in DAY.glob("*/*.json")
                   if not re.match(r"^\d{8}_\d{6}_T_", p.parent.name))  # generator clause: drops JSON files inside capture-session folders, leaving the phone logs
    if not files:  # if: none found (an empty list is falsy)
        die("no handset .json files found")  # stop

    from collections import Counter, defaultdict  # local import: Counter counts keys; defaultdict makes missing keys
    multi_value: Counter = Counter()  # counts, per (device, band), the epochs whose collapsed readings disagreed
    multi_total: Counter = Counter()  # counts, per (device, band), all collapsed epochs
    multi_spread: dict = defaultdict(list)  # per (device, band), the list of spreads when they disagreed
    band_rows, epoch_rows, meta = [], [], []  # tuple assignment: three empty lists for the outputs
    for f in files:  # for loop over the handset files
        doc = json.loads(f.read_text(encoding="utf-8"))  # parses the whole JSON file into dicts and lists
        dev = doc["data"]["devices"][0]  # the first (only) device description
        model = dev.get("model", "")  # the handset model, e.g. "Pixel 6"
        start = datetime.fromisoformat(doc["startTime"].replace("Z", "+00:00"))  # capture start; "Z" means UTC
        end = datetime.fromisoformat(doc["endTime"].replace("Z", "+00:00"))  # capture end

        nm = re.match(r"^(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_", f.name)  # date and time from the file name
        fname_ts = datetime.strptime(f"{nm[1]} {nm[2]}", "%Y-%m-%d %H-%M-%S")  # nm[1] is group 1; parsed, no zone
        # if: the file name and the stated start differ by a second or more (zone stripped so they can subtract)
        if abs((start.replace(tzinfo=None) - fname_ts).total_seconds()) >= 1.0:
            die(f"{f.name}: filename timestamp and startTime disagree")  # stop: the time base would be in doubt
        if not doc["startTime"].endswith("Z"):  # if: the start time lacks the explicit UTC marker
            die(f"{f.name}: startTime does not carry the explicit Z that marks UTC")  # stop

        ms = doc["data"]["measurements"]  # the list of one-second epochs
        if len(ms) != int(doc["events"]):  # if: the header's event count differs from what the file holds
            # records the mismatch
            flag(f"{f.name}: header declares {doc['events']} events but the file holds "
                 f"{len(ms)} measurements")  # f-string continued: how many measurements the file actually holds

        prev_t = None  # the previous epoch's time, for the order check
        collapsed_total = kept_total = 0  # chained assignment: raw gain-control entries, and rows kept after collapse
        for e in ms:  # for loop over the epochs of this file
            t = datetime.fromisoformat(e["measurementDateTime"].replace("Z", "+00:00"))  # the epoch time in UTC
            if prev_t is not None and t < prev_t:  # if: time went backwards
                flag(f"{f.name}: epoch timestamps are not monotonic at {t.isoformat()}")  # recorded
            prev_t = t  # remembers this time for the next epoch

            sats = e["gnssEvent"].get("measurements") or []  # the per-satellite list; "or []" guards against None
            cn0 = [s["cn0DbHz"] for s in sats if s.get("cn0DbHz") is not None]  # every satellite's C/N0 (dB-Hz)
            per_band_sats: dict[int, list[float]] = {}  # annotated empty dict: band -> C/N0 values in that band
            for s in sats:  # for loop over the satellites
                k = band_key(s.get("carrierFrequencyHz", 0))  # which band this satellite signal is in
                if k is not None and s.get("cn0DbHz") is not None:  # if: a known band and a C/N0 value
                    per_band_sats.setdefault(k, []).append(s["cn0DbHz"])  # setdefault makes the list, then append

            loc = (e.get("gnssLocations") or [{}])[0]  # first location entry; [{}] gives an empty dict if absent
            lat_top, lon_top = e.get("lat"), e.get("lon")  # the top-level position fields
            lat_gl, lon_gl = loc.get("latitude"), loc.get("longitude")  # the location-list position fields
            # conditional expression: distance between the two positions, or None if any coordinate is missing
            sep = (haversine_m(lat_top, lon_top, lat_gl, lon_gl)
                   if None not in (lat_top, lon_top, lat_gl, lon_gl) else None)  # conditional expression: no separation when either of the two positions is missing

            epoch_rows.append({  # appends one row per epoch: identity, time, satellites, C/N0 and position fields
                "device_folder": f.parent.name, "device_model": model,  # dict entry: the phone's folder and model, which identify the device
                "logger_session_id": doc["sessionId"], "source_file": f.name,  # dict entry: the logger's session number and the file the epoch came from
                "timestamp_utc": t,  # dict entry: the epoch's time in UTC
                "n_satellites_epoch": len(sats),  # dict entry: how many satellites the phone tracked in this epoch
                # median = the middle value once sorted, so one very strong or weak satellite cannot drag it; else NaN
                "median_cn0_dbhz": float(np.median(cn0)) if cn0 else np.nan,
                "lat_top_level": lat_top, "lon_top_level": lon_top,  # dict entry: latitude and longitude from the log's top-level position fields
                "lat_gnsslocations": lat_gl, "lon_gnsslocations": lon_gl,  # dict entry: latitude and longitude from the log's separate location records
                "position_fields_separation_m": sep,  # dict entry: distance between those two positions in metres
                "position_accuracy_m": e.get("accuracy"),  # dict entry: the accuracy radius the phone reports for its fix
                "location_provider": e.get("provider"),  # dict entry: which location provider produced the fix
            })

            # The gain-control list repeats one physical reading per constellation using that frequency.
            # Collapse to one row per band; where the merged readings disagree, the disagreements are counted
            # below and the median is kept.
            agcs = e["gnssEvent"].get("automaticGainControls") or []  # the raw gain-control list; "or []" guards None
            groups: dict[int, list[dict]] = {}  # annotated empty dict: band -> the gain-control entries for it
            for a in agcs:  # for loop over the raw gain-control entries
                # dict.get with a fallback: the rounded frequency if present, else the carrier frequency
                k = band_key(a.get("roundedFrequencyHz", a.get("carrierFrequencyHz")))
                if k is None:  # if: not one of the four bands
                    continue  # skip it
                groups.setdefault(k, []).append(a)  # files the entry under its band
            for k, g in groups.items():  # for loop over (band, entries) pairs
                vals = [x["levelDb"] for x in g]  # list comprehension: the gain-control readings in dB for this band
                jm = {bool(x.get("jammed")) for x in g}  # set comprehension: the distinct values of the jammed flag
                spread = float(max(vals) - min(vals))  # how far the readings being merged disagree, in dB
                # The Pixel reports one physical reading per frequency and repeats
                # it for each constellation. The Samsung runs separate gain-control
                # chains per constellation on the same frequency, so its collapsed
                # groups can legitimately disagree. Counted, not flagged per epoch.
                # Counter keyed by a tuple: adds 1 if the readings disagreed (spread above rounding noise), else 0
                multi_value[(model, BAND_SHORT[k])] += (1 if spread > 1e-9 else 0)
                multi_total[(model, BAND_SHORT[k])] += 1  # counts every collapsed epoch for that device and band
                if spread > 1e-9:  # if: they disagreed
                    multi_spread[(model, BAND_SHORT[k])].append(spread)  # keeps the size of the disagreement
                collapsed_total += len(g)  # raw entries consumed
                kept_total += 1  # one row kept
                band_rows.append({  # appends one row per epoch per band: the collapsed reading and its context
                    "device_folder": f.parent.name, "device_model": model,  # dict entry: the phone's folder and model, which identify the device
                    "logger_session_id": doc["sessionId"],  # dict entry: the logger's session number
                    "timestamp_utc": t, "band_hz": k, "band": BAND_SHORT[k],  # dict entry: epoch time, band frequency and short band name
                    "level_db": float(np.median(vals)),  # the median of the merged readings is the band's value
                    "level_db_spread_db": round(spread, 4),  # dict entry: spread of the merged gain readings in dB, rounded to four decimals
                    "jammed": bool(any(jm)),  # any(): True if any merged entry carried the jammed flag
                    "n_agc_entries_collapsed": len(g),  # dict entry: how many raw gain entries were merged into this row
                    # set comprehension of constellation type codes, sorted and joined with ";" into one cell
                    "constellations_sharing_band": ";".join(
                        str(c) for c in sorted({x["constellationType"] for x in g})),  # generator clause: the distinct constellation codes that share the band, as text
                    "n_satellites_epoch": len(sats),  # dict entry: satellites tracked in the whole epoch
                    "n_satellites_band": len(per_band_sats.get(k, [])),  # satellites tracked in this band
                    # conditional expression: median C/N0 of this band's satellites, or NaN if there are none
                    "median_cn0_dbhz_band": (float(np.median(per_band_sats[k]))
                                             if per_band_sats.get(k) else np.nan),  # conditional expression: no C/N0 median when the phone tracks no satellite in this band
                })

        # appends one summary per file: identity, capture window, epoch count and collapse counts
        meta.append({"file": f.name, "device_folder": f.parent.name, "model": model,
                     "session_id": doc["sessionId"], "start_utc": start, "end_utc": end,  # dict entry: the logger's session number and the capture's start and end
                     "n_epochs": len(ms), "agc_entries": collapsed_total,  # dict entry: epochs in the file and the raw gain entries seen
                     "agc_rows_kept": kept_total})  # dict entry: gain rows kept after merging duplicates

    # dict of the collapse statistics; the dict comprehension turns each spread list into its median
    collapse = {"n_with_spread": dict(multi_value), "n_total": dict(multi_total),
                "median_spread": {k: float(np.median(v)) for k, v in multi_spread.items()}}  # dict entry: the median spread of the merged readings, per device and band
    return pd.DataFrame(band_rows), pd.DataFrame(epoch_rows), meta, collapse  # the two tables become DataFrames


# --------------------------------------------------------------------------
# 4. Live readings, position and per-session distance
# --------------------------------------------------------------------------

# function: adds value_run_s, frozen and frozen_reason to the band rows.
# Mechanism found in the raw files (checks 2-4): long runs of one unchanged value occur only in epochs whose band
# has no raw satellite measurement; but a band with no measurement can still report a changing (live) value, as the
# Pixel's L1, G1 and B1I do under our own attack. So a reading is frozen only when BOTH hold: no measurement in its
# band (n_satellites_band == 0) AND a value unchanged for at least FROZEN_RUN_S seconds. Losing every satellite is
# never on its own a reason to drop a reading.
def flag_frozen(bands: pd.DataFrame) -> pd.DataFrame:
    key = ["device_folder", "logger_session_id", "band_hz"]  # one series per phone, capture and band
    b = bands.sort_values(key + ["timestamp_utc"]).copy()  # sorted so consecutive rows are consecutive seconds
    prev_v = b.groupby(key)["level_db"].shift()  # the previous reading's value in the same series (NaN at a start)
    gap = b.groupby(key)["timestamp_utc"].diff().dt.total_seconds()  # seconds since the previous reading
    # boolean: a new run starts where the value changes or the series has a gap over MAX_GAP_S (NaN compares True)
    new_run = (b["level_db"] != prev_v) | (gap > MAX_GAP_S)
    b["_run"] = new_run.cumsum()  # running count of run starts = a run identifier
    # transform: for every row, its run's span, last minus first timestamp, in seconds
    b["value_run_s"] = b.groupby("_run")["timestamp_utc"].transform(lambda x: (x.max() - x.min()).total_seconds())
    no_meas = b["n_satellites_band"] == 0  # boolean: no raw measurement in this band in this epoch
    long_run = b["value_run_s"] >= FROZEN_RUN_S  # boolean: the value has not changed for 30 s or more
    b["frozen"] = no_meas & long_run  # "&": both conditions must hold
    # conditional text per row via np.where: the reason, or "" for a live reading
    b["frozen_reason"] = np.where(b["frozen"], "no measurement in band and value unchanged >= 30 s", "")
    b["_flag_run_only"] = long_run & ~no_meas  # diagnostic: long run but satellites present (kept)
    b["_flag_nomeas_only"] = no_meas & ~long_run  # diagnostic: no satellites but value changing (kept)
    return b.drop(columns="_run").sort_index()  # back in the original row order, without the helper column


def analyse_position(ep: pd.DataFrame) -> pd.DataFrame:  # function: distances, steps, speeds and plausibility
    ep = ep.sort_values(["device_folder", "timestamp_utc"]).copy()  # sorted by phone then time; .copy() avoids aliasing
    # list comprehension over (lat, lon) pairs: distance to the jammer site in km, NaN where the fix is missing
    ep["dist_to_tx_km"] = [
        haversine_m(la, lo, TX_SENDER[0], TX_SENDER[1]) / 1000.0  # generator clause: distance from the fix to the transmitter in kilometres
        if pd.notna(la) and pd.notna(lo) else np.nan  # conditional expression: the distance is NaN when the fix has no latitude or longitude
        for la, lo in zip(ep["lat_gnsslocations"], ep["lon_gnsslocations"])]  # generator clause: runs over the latitude and longitude of every epoch
    # the same, as distance to the Stave landmark, used only for the 25 km plausibility test
    ep["dist_to_stave_km"] = [
        haversine_m(la, lo, STAVE[0], STAVE[1]) / 1000.0  # generator clause: distance from the fix to the Stave landmark in kilometres
        if pd.notna(la) and pd.notna(lo) else np.nan  # conditional expression: the distance is NaN when the fix has no latitude or longitude
        for la, lo in zip(ep["lat_gnsslocations"], ep["lon_gnsslocations"])]  # generator clause: runs over the latitude and longitude of every epoch
    ep["step_m"] = np.nan  # new column, filled per capture below
    ep["implied_speed_kmh"] = np.nan  # likewise
    # for loop per phone AND capture: the first fix of a capture has no step from the previous capture
    for _, g in ep.groupby(["device_folder", "logger_session_id"], sort=False):
        la, lo = g["lat_gnsslocations"].to_numpy(), g["lon_gnsslocations"].to_numpy()  # coordinates as arrays
        tt = g["timestamp_utc"].to_numpy()  # times
        s, v = [np.nan], [np.nan]  # the first epoch has no previous one, so its step and speed are NaN
        for i in range(1, len(g)):  # for loop over every later epoch
            if np.isnan(la[i]) or np.isnan(la[i - 1]):  # if: either fix is missing
                # three statements on one line separated by ";": record NaN step and speed, then skip ahead
                s.append(np.nan); v.append(np.nan); continue
            d = haversine_m(la[i - 1], lo[i - 1], la[i], lo[i])  # metres moved since the previous epoch
            dt = (pd.Timestamp(tt[i]) - pd.Timestamp(tt[i - 1])).total_seconds()  # seconds elapsed
            s.append(d)  # records the step
            v.append(d / dt * 3.6 if dt > 0 else np.nan)  # m/s times 3.6 = km/h; NaN if no time passed
        ep.loc[g.index, "step_m"] = s  # .loc by the group's row labels: writes this capture's steps
        ep.loc[g.index, "implied_speed_kmh"] = v  # and speeds
    # boolean column: within 25 km of the Stave landmark AND implied speed under 200 km/h; "&" combines element-wise
    ep["position_plausible"] = (
        (ep["dist_to_stave_km"] < PLAUSIBLE_RADIUS_KM)  # assigned value: first condition, the fix lies within the plausible radius around Stave
        & (ep["implied_speed_kmh"].fillna(0) < MAX_SPEED_KMH))  # fillna(0): unknown speed passes; NaN distance fails
    return ep  # the epoch table with the new columns


# function: the time-based stationary flag: path length summed over the
# preceding 60 s, within one phone and one capture, so a window never spans two captures; at least 10 readings
def stationary_flags(ep: pd.DataFrame) -> pd.Series:
    out = pd.Series(False, index=ep.index)  # default: not stationary
    for _, g in ep.groupby(["device_folder", "logger_session_id"], sort=False):  # per phone and capture
        g = g.sort_values("timestamp_utc")  # time order
        # rolling ("60s") on a time index: sums the steps in the 60 s ending at each reading; min_periods=10 readings
        roll = g.set_index("timestamp_utc")["step_m"].fillna(0).rolling("60s", min_periods=10).sum()
        out.loc[g.index] = (roll.to_numpy() < STATIONARY_M)  # True where under 150 m was travelled; NaN -> False
    out[~ep["position_plausible"]] = False  # implausible fixes are never stationary
    return out  # one boolean per epoch, aligned with ep's index


# function: when the recording vehicle was moving in each session, from the OBD-II SPEED signal. Verification only:
# the exclusion rule allows the odometer for labelling and verification, never as a feature. Returns
# {test code: (first moving instant, last moving instant) in UTC, or None if the vehicle never moved}.
def obd_moving_intervals(sessions: pd.DataFrame) -> dict:
    out = {}  # empty dict of intervals
    for _, r in sessions.iterrows():  # for loop over our sessions
        files = sorted((DAY / r["session_folder"]).glob("*OBD-II*.csv"))  # the session's OBD-II CSV
        if len(files) != 1:  # if: none or several
            die(f"{r['test_code']}: expected one OBD-II CSV, found {len(files)}")  # stop
        o = pd.read_csv(files[0])  # the polling log
        sp = o[(o["signal"] == "SPEED") & (o["valid_value"].astype(str) == "True")]  # valid speed replies only
        mv = sp[pd.to_numeric(sp["value"], errors="coerce") > 0]  # replies reporting any movement
        if mv.empty:  # if: the vehicle never moved
            out[r["test_code"]] = None  # stationary for the whole capture
            continue  # next session
        t = pd.to_datetime(mv["system_time_iso8601"], utc=True)  # ISO times with offset -> UTC timestamps
        out[r["test_code"]] = (t.min(), t.max())  # first and last moving instant
    return out  # the dict


# function: each session's distance to SENDER from the handset fixes actually recorded in it (method decided
# beforehand). A session whose vehicle moved is split into its moving part and its parked part. Within a
# part: the median of plausible fixes in the capture margins (outside the scheduled window, where the position is not
# spoofed); only where a part has no margin fixes, the median over the whole part (method column says which).
def session_distances(ep: pd.DataFrame, sessions: pd.DataFrame, moving: dict) -> pd.DataFrame:
    rows = []  # empty list of distance rows
    for _, r in sessions.iterrows():  # for loop over our sessions
        cs = pd.Timestamp(r["capture_start_utc"], tz="UTC")  # capture start
        ce = pd.Timestamp(r["capture_end_utc"], tz="UTC")  # capture end
        ss = pd.Timestamp(r["scheduled_start_utc"], tz="UTC")  # scheduled start
        se = pd.Timestamp(r["scheduled_end_utc"], tz="UTC")  # scheduled end
        mv = moving[r["test_code"]]  # the moving interval, or None
        # list of (part name, start, end): one part if the vehicle never moved, else moving then parked
        parts = ([("whole capture", cs, ce)] if mv is None
                 # else branch of the conditional expression: a moving and a parked part, split one microsecond after the last movement
                 else [("moving", cs, mv[1]), ("parked", mv[1] + pd.Timedelta(microseconds=1), ce)])
        for part, p0, p1 in parts:  # for loop over the parts
            # plausible handset fixes in this part, both phones together (one vehicle, one position)
            f = ep[(ep.timestamp_utc >= p0) & (ep.timestamp_utc <= p1) & ep.position_plausible
                   # continuation of the boolean mask: '&' also requires a distance to SENDER (fixes without a position are dropped)
                   & ep.dist_to_tx_km.notna()]
            if part == "moving" and not f.empty:  # if: a fix falls while the vehicle moved, the method would not hold
                die(f"{r['test_code']}: {len(f)} handset fixes fall inside the moving interval")  # stop (assertion)
            mg = f[(f.timestamp_utc < ss) | (f.timestamp_utc >= se)]  # fixes in the margins (outside the window)
            use, method = (mg, "margins") if len(mg) else (f, "whole part (no margin fixes)")  # tuple assignment
            if f.empty:  # if: the phones logged nothing in this part
                method = "no handset fix"  # nothing to measure
            rows.append({  # appends the row for ground_truth/session_distance_to_sender.csv
                # dict entries (one row of ground_truth/session_distance_to_sender.csv): the session code and which part of it this row covers
                "test_code": r["test_code"], "part": part,
                # dict entries: the part's start and end as text; strftime formats a timestamp by the given pattern
                "part_start_utc": p0.strftime("%Y-%m-%d %H:%M:%S"), "part_end_utc": p1.strftime("%Y-%m-%d %H:%M:%S"),
                # dict entries: how many plausible fixes fall in the part, and how many of those lie in the capture margins
                "n_fixes": len(f), "n_margin_fixes": len(mg),
                # conditional expressions: the first and last fix, or "" when there is none
                "first_fix_utc": f.timestamp_utc.min().strftime("%H:%M:%S") if len(f) else "",
                # dict entry: time of the last fix as HH:MM:SS; the conditional expression writes '' when there is no fix
                "last_fix_utc": f.timestamp_utc.max().strftime("%H:%M:%S") if len(f) else "",
                # dict entry: which fixes the distance came from ('margins', 'whole part ...' or 'no handset fix')
                "method": method,
                # the chosen distance: median of the chosen fixes, else NaN (an empty CSV cell)
                "dist_to_sender_km": round(float(use.dist_to_tx_km.median()), 3) if len(use) else np.nan,
                # dict entry: 5th percentile of the chosen distances (quantile(0.05)), rounded to metres; NaN when there are none
                "dist_p5_km": round(float(use.dist_to_tx_km.quantile(0.05)), 3) if len(use) else np.nan,
                # dict entry: 95th percentile of the chosen distances; with p5 it shows how tightly the fixes agree
                "dist_p95_km": round(float(use.dist_to_tx_km.quantile(0.95)), 3) if len(use) else np.nan,
                # both values: margins median and whole-part median
                "dist_margins_km": round(float(mg.dist_to_tx_km.median()), 3) if len(mg) else np.nan,
                # dict entry: median over every plausible fix in the part, kept beside the margins value as decided
                "dist_whole_part_km": round(float(f.dist_to_tx_km.median()), 3) if len(f) else np.nan,
                # dict entry: median latitude of the chosen fixes, 7 decimals (about 1 cm), used to compare sites
                "median_lat": round(float(use.lat_gnsslocations.median()), 7) if len(use) else np.nan,
                # dict entry: median longitude of the chosen fixes, 7 decimals
                "median_lon": round(float(use.lon_gnsslocations.median()), 7) if len(use) else np.nan,
                # dict entry: median distance of the chosen fixes from the Stave landmark, to show which site this is
                "dist_to_stave_landmark_km": round(float(use.dist_to_stave_km.median()), 3) if len(use) else np.nan,
            })
    return pd.DataFrame(rows)  # the per-session distance table


# --------------------------------------------------------------------------
# 5. The tests (each returns its table; main() runs them on live readings, then silently on all readings)
# --------------------------------------------------------------------------

# function: the central comparison, level by exposure group per device and band, all epochs then stationary only
def comparison_test(bands: pd.DataFrame) -> pd.DataFrame:
    print("  Direction fixed in advance: in-band jamming drives level_db MORE NEGATIVE.")  # the fixed direction
    print("  A positive difference is evidence AGAINST the hypothesis.")  # and what the opposite sign means
    print(f"  The group '{GUNK}' is excluded from every comparison.")  # decision
    comp_rows = []  # empty list of comparison rows
    for restrict in (False, True):  # for loop: first all epochs, then stationary epochs only
        lab = "stationary epochs only" if restrict else "all epochs"  # conditional expression: the label
        print(f"\n  --- {lab} ---")  # a sub-heading in the printout
        for dev in sorted(bands["device_model"].unique()):  # for loop over phones (never pooled)
            print(f"\n  {dev}")  # the phone name
            # table header
            print(f"      {'band':<5} {'group':<34} {'n':>6} {'median':>8} {'IQR':>7} "
                  f"{'vs baseline':>12} {'Cliff d':>8}")  # f-string continued: header cells for the difference to baseline and Cliff's delta
            for hz in sorted(bands[bands.device_model == dev]["band_hz"].unique()):  # for loop over its bands
                sub = bands[(bands.device_model == dev) & (bands.band_hz == hz)]  # this phone and band
                if restrict:  # if: the stationary pass
                    sub = sub[sub.stationary.fillna(False)]  # keeps stationary rows; missing flags count as False
                base = sub.loc[sub.exposure_group == GBL, "level_db"].dropna().to_numpy()  # the baseline readings
                for g in (G50, GHI, GBL):  # for loop over the three compared groups (GUNK is never compared)
                    v = sub.loc[sub.exposure_group == g, "level_db"].dropna().to_numpy()  # this group's readings
                    if len(v) == 0:  # if: the group is empty
                        continue  # skip it
                    # median difference, group minus baseline; negative = lower gain = the predicted jamming sign
                    diff = (np.median(v) - np.median(base)) if len(base) else np.nan
                    # Cliff's delta of group against baseline; negative = group tends lower (predicted sign)
                    dl = cliffs_delta(v, base) if (len(base) and g != GBL) else np.nan
                    # the row; conditional expressions print "n/a" and "-" where a value is undefined
                    print(f"      {BAND_SHORT[hz]:<5} {g:<34} {len(v):>6} {np.median(v):>8.2f} {iqr(v):>7.2f} "
                          # '+' joins the difference column: printed with its sign (:+12.2f), or 'n/a' when there is no baseline
                          + (f"{diff:>+12.2f}" if not np.isnan(diff) else f"{'n/a':>12}")
                          # '+' joins Cliff's delta (3 decimals), or '-' for the baseline row, where no delta is defined
                          + (f" {dl:>8.3f}" if not np.isnan(dl) else f" {'-':>8}"))
                    comp_rows.append({  # appends the comparison row for ground_truth/handset_level_comparison.csv
                        "restriction": lab, "device_model": dev,  # dict entry: which restriction the row belongs to, and the phone model
                        "band": BAND_SHORT[hz], "band_hz": hz, "exposure_group": g,  # dict entry: short band name, frequency and exposure group of the row
                        "n": len(v), "median_level_db": round(float(np.median(v)), 3),  # dict entry: number of gain readings and their median level
                        "iqr_level_db": round(iqr(v), 3),  # dict entry: interquartile range of the gain level
                        "p5": round(float(np.percentile(v, 5)), 3),  # dict entry: 5th percentile of the gain level
                        "p95": round(float(np.percentile(v, 95)), 3),  # dict entry: 95th percentile of the gain level
                        # conditional expressions: rounded values, or None (an empty CSV cell) when undefined
                        "median_minus_baseline_db": (round(float(diff), 3) if not np.isnan(diff) else None),
                        # dict entry: Cliff's delta rounded to 4 places, or None (an empty CSV cell) for the baseline row
                        "cliffs_delta_vs_baseline": (round(dl, 4) if not np.isnan(dl) else None),
                    })
    return pd.DataFrame(comp_rows)  # the comparison table


# function: the within-capture contrast (50 W against baseline inside one capture, so position is held fixed)
def within_capture_test(ep: pd.DataFrame, bands: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    # table header
    print(f"\n      {'device':<12} {'session':>7} {'n 50 W':>8} {'n base':>8} {'within-capture contrast?':>26}")
    within_ok = []  # empty list of (phone, capture) pairs holding enough of both groups
    for dev in sorted(ep["device_model"].unique()):  # for loop over phones
        for sid in sorted(ep[ep.device_model == dev]["logger_session_id"].unique()):  # and their captures
            s = ep[(ep.device_model == dev) & (ep.logger_session_id == sid)]  # this capture's epochs
            n50 = int((s.exposure_group == G50).sum())  # epochs in the 50 W group
            nbl = int((s.exposure_group == GBL).sum())  # epochs in the baseline group
            ok = n50 >= 100 and nbl >= 100  # boolean: at least 100 seconds of each
            print(f"      {dev:<12} {sid:>7} {n50:>8} {nbl:>8} {'yes' if ok else 'no':>26}")  # counts and yes/no
            if ok:  # if: this capture supports a within-capture contrast
                within_ok.append((dev, sid))  # keeps it
    print(f"\n  Captures supporting a within-capture contrast: {len(within_ok)}.")  # how many
    # table header
    print(f"\n      {'device':<12} {'sess':>5} {'band':<5} {'n 50 W':>7} {'n base':>7} "
          f"{'med 50 W':>9} {'med base':>9} {'50W-base':>9} {'Cliff d':>8}")  # f-string continued: header cells for the two medians, their difference and Cliff's delta
    within_rows = []  # empty list of within-capture rows
    for dev, sid in within_ok:  # for loop unpacking each qualifying (phone, capture)
        for hz in sorted(bands[bands.device_model == dev]["band_hz"].unique()):  # and each band
            # this capture and band only, so where the handset lay is held fixed
            s = bands[(bands.device_model == dev) & (bands.logger_session_id == sid) & (bands.band_hz == hz)]
            a = s.loc[s.exposure_group == G50, "level_db"].dropna().to_numpy()  # 50 W readings
            b = s.loc[s.exposure_group == GBL, "level_db"].dropna().to_numpy()  # baseline readings
            if len(a) < 100 or len(b) < 100:  # if: fewer than 100 of either
                continue  # skip this band
            diff = float(np.median(a) - np.median(b))  # 50 W median minus baseline median; negative = predicted
            dl = cliffs_delta(a, b)  # negative = 50 W readings tend lower than baseline, the predicted sign
            print(f"      {dev:<12} {sid:>5} {BAND_SHORT[hz]:<5} {len(a):>7} {len(b):>7} "  # the row
                  f"{np.median(a):>9.2f} {np.median(b):>9.2f} {diff:>+9.2f} {dl:>8.3f}")  # f-string continued: the two medians, their difference and Cliff's delta for this row
            # appends the row for ground_truth/handset_within_capture_comparison.csv
            within_rows.append({"restriction": "within one capture", "device_model": dev,
                                "logger_session_id": sid, "band": BAND_SHORT[hz],  # dict entry: the logger's session number and the short band name
                                "band_hz": hz, "n_50w": len(a), "n_baseline": len(b),  # dict entry: band frequency and readings during the 50 W test and the baseline
                                "median_50w_db": round(float(np.median(a)), 3),  # dict entry: median gain level during the 50 W test
                                "median_baseline_db": round(float(np.median(b)), 3),  # dict entry: median gain level in the baseline minutes
                                "median_minus_baseline_db": round(diff, 3),  # dict entry: 50 W median minus baseline median; a negative value is the predicted drop
                                "cliffs_delta_vs_baseline": round(dl, 4),  # dict entry: Cliff's delta of the 50 W readings against the baseline
                                "direction_matches_prediction": bool(diff < 0)})  # True if 50 W reads lower
    # list of the CSV's columns, so an empty table still writes a header
    cols = ["restriction", "device_model", "logger_session_id", "band", "band_hz", "n_50w", "n_baseline",
            "median_50w_db", "median_baseline_db", "median_minus_baseline_db", "cliffs_delta_vs_baseline",
            "direction_matches_prediction"]
    wdf = pd.DataFrame(within_rows, columns=cols)  # the within-capture table (header only when empty)
    if not wdf.empty:  # if: at least one contrast was possible
        nm = int(wdf["direction_matches_prediction"].sum())  # how many moved in the predicted direction
        print(f"\n  Within a single capture, {nm} of {len(wdf)} device-band contrasts move")  # reports it
        print("  in the direction fixed in advance (jamming drives level_db down).")  # the next line of the same printed text
    return wdf, within_ok  # the table and the qualifying captures


# function: transition timing, the largest change of the smoothed level near each scheduled 50 W edge
def timing_test(bands: pd.DataFrame, plan, meta: list) -> pd.DataFrame:
    trans_rows = []  # empty list of timing rows
    # list comprehension: (folder, model, session, start, end) for each handset capture
    caps = [(m["device_folder"], m["model"], m["session_id"],
             pd.Timestamp(m["start_utc"]), pd.Timestamp(m["end_utc"])) for m in meta]  # tuple element: capture start and end as time stamps
    edges = []  # empty list of (test code, "on"/"off", instant) switching edges
    for p in plan:  # for loop over the schedule
        if p["power_w"] is None or p["power_w"] < FIFTY_W_MIN or p["test_area"] != 1:  # if: not a 50 W TA1 test
            continue  # skip it
        edges.append((p["test_code"], "on", pd.Timestamp(p["start_utc"])))  # its switch-on instant
        edges.append((p["test_code"], "off", pd.Timestamp(p["end_utc"])))  # its switch-off instant
    win = pd.Timedelta(minutes=5)  # the search half-window: 5 minutes either side of an edge
    for dev_folder, model, sid, cs, ce in caps:  # for loop unpacking each capture
        for code, kind, t in edges:  # nested loop over the edges
            if not (cs + win <= t <= ce - win):  # if: the edge +/- 5 min does not fit inside the capture
                continue  # only edges wholly inside a capture
            for hz in sorted(bands[bands.device_model == model]["band_hz"].unique()):  # for loop over bands
                # mask of five conditions: this phone, capture and band, within +/- 5 min of the edge; time-sorted
                sub = bands[(bands.device_model == model) & (bands.logger_session_id == sid)
                            # continuation of the boolean mask: '&' adds this band, and readings from 5 min before the edge onwards
                            & (bands.band_hz == hz) & (bands.timestamp_utc >= t - win)
                            & (bands.timestamp_utc <= t + win)].sort_values("timestamp_utc")  # element-wise condition: readings up to 5 minutes after the edge, then sorted by time
                if len(sub) < 120:  # if: fewer than 2 minutes of readings
                    continue  # skip
                # rolling median over 15 readings centred on each one (at least 5), smoothing second-to-second noise
                v = sub["level_db"].rolling(15, center=True, min_periods=5).median()
                tt = sub["timestamp_utc"].to_numpy()  # the times as an array
                d = v.diff().to_numpy()  # change of the smoothed trace from one reading to the next
                # index of the largest drop for a switch-on, or the largest rise for a switch-off (NaNs ignored)
                k = int(np.nanargmin(d)) if kind == "on" else int(np.nanargmax(d))
                best = pd.Timestamp(tt[k])  # when that largest change happened
                trans_rows.append({  # appends the row: scheduled against observed, offset and step size
                    "device_model": model, "logger_session_id": sid,  # dict entry: the phone model and its logger session
                    "test_code": code, "edge": kind, "band": BAND_SHORT[hz],  # dict entry: test code, switch-on or switch-off edge, and the band
                    "scheduled_utc": t.strftime("%Y-%m-%d %H:%M:%S"),  # dict entry: when the plan says the edge happens
                    "observed_utc": best.strftime("%Y-%m-%d %H:%M:%S"),  # dict entry: when the phone's gain shows its largest change near the edge
                    "offset_s": round((best - t).total_seconds(), 1),  # observed minus scheduled, seconds
                    # the size of that largest change in dB (conditional expression picks drop or rise)
                    "step_size_db": round(float(np.nanmin(d) if kind == "on" else np.nanmax(d)), 3),
                })
    tdf = pd.DataFrame(trans_rows)  # the timing table
    if tdf.empty:  # if: no edge could be tested
        print("  No scheduled 50 W edge falls wholly inside a handset capture with")  # says so
        print("  enough margin to test. The timing argument is unavailable.")  # the next line of the same printed text
        return tdf  # the empty table
    # count of tests; ngroups counts the distinct (test, edge) combinations
    print(f"  {len(tdf)} band-edge tests across {tdf.groupby(['test_code', 'edge']).ngroups} distinct scheduled edges.")
    # table header
    print(f"\n      {'device':<12} {'test':<7} {'edge':<4} {'band':<4} "
          # continuation of the header f-string: the column titles, each padded to a fixed width
          f"{'scheduled':<10} {'observed':<10} {'offset s':>9} {'step dB':>9}")
    for _, r in tdf.iterrows():  # for loop over the timing rows; [11:] keeps only HH:MM:SS
        # f-string print of one timing row; [11:] keeps only HH:MM:SS of the two times
        print(f"      {r['device_model']:<12} {r['test_code']:<7} {r['edge']:<4} {r['band']:<4} "
              # continuation of the f-string: scheduled and observed times ([11:] = HH:MM:SS) and the offset in seconds
              f"{r['scheduled_utc'][11:]:<10} {r['observed_utc'][11:]:<10} {r['offset_s']:>9.0f} "
              # continuation of the f-string: the size of the largest change in dB, 2 decimals
              f"{r['step_size_db']:>9.2f}")
    near = tdf[tdf.offset_s.abs() <= 30]  # tests whose largest change fell within 30 s of schedule
    w = win.total_seconds()  # the half-window in seconds (300)
    print(f"\n  Largest change within 30 s of the scheduled instant: "  # how many landed near the instant
          # continuation of the f-string: how many of all timing tests fell within 30 s, and that share in per cent
          f"{len(near)} of {len(tdf)} ({100 * len(near) / len(tdf):.0f} %).")
    print(f"  Median absolute offset: {tdf.offset_s.abs().median():.0f} s.")  # typical distance from schedule
    # uniform on -w..+w: mean |offset| = w/2 (150 s); chance of |offset| <= 30 s = 60 / 2w = 30/w (10 %)
    print(f"  Under chance (offset uniform over +/-{w / 60:.0f} min) the expected absolute offset is "
          # continuation of the f-string: the chance expectations, half the window (150 s) and 30/300 = 10 %
          f"{w / 2:.0f} s and about {100 * 30 / w:.0f} % of cases fall within 30 s.")
    if len(near) / len(tdf) <= (30 / w) * 1.5:  # if: the observed rate is at most 1.5 times the chance rate
        print("  The observed rate is not above chance: the transitions do not line up with the schedule.")  # says so
    return tdf  # the timing table


# function: the pyramid test - within-step contrast, nesting and band selectivity inside the 1.8.2 window
def pyramid_test(bands: pd.DataFrame, pyr: dict) -> tuple[pd.DataFrame, list]:
    p0 = pd.Timestamp(pyr["entry"]["start_utc"])  # the pyramid's scheduled start
    p1 = pd.Timestamp(pyr["entry"]["end_utc"])  # and end
    # band rows inside the window (end exclusive); .copy() so new columns can be added safely
    inside = bands[(bands.timestamp_utc >= p0) & (bands.timestamp_utc < p1)].copy()
    print(f"  Handset coverage of the 1.8.2 window ({p0:%H:%M}-{p1:%H:%M} UTC): "  # coverage
          # continuation of the f-string: band rows and distinct epochs (nunique) inside the pyramid window
          f"{len(inside)} band-rows from {inside['timestamp_utc'].nunique()} epochs.")
    if inside.empty:  # if: the handsets did not record during the pyramid
        print("  No handset data inside the pyramid window; this test is unavailable.")  # says so
        return pd.DataFrame(), []  # nothing to report
    # seconds since the pyramid start for every row, as one array
    sec = (inside["timestamp_utc"] - p0).dt.total_seconds().to_numpy()
    inside["pyramid_step"] = (sec // (pyr["step_min"] * 60)).astype(int) + 1  # "//": whole steps; step number 1-15
    # True in the first jam_min minutes of each step: the jamming minutes
    inside["in_jamming_minutes"] = (sec % (pyr["step_min"] * 60)) < pyr["jam_min"] * 60
    pyr_rows = []  # empty list of pyramid result rows
    print(f"\n  Within-step contrast: level_db during the {pyr['jam_min']}-minute jamming portion "  # a sub-heading in the printout
          # continuation of the f-string: the break length in minutes, taken from the parsed pyramid
          f"against the {pyr['brk_min']}-minute break, per band.")
    # table header
    print(f"\n      {'device':<12} {'band':<4} {'predicted steps':<16} {'n jam':>7} {'n brk':>7} "
          # continuation of the header f-string: column titles for the medians, their difference and Cliff's delta
          f"{'med jam':>9} {'med brk':>9} {'jam-brk':>9} {'Cliff d':>8}")
    for dev in sorted(inside["device_model"].unique()):  # for loop over phones
        for hz in sorted(inside[inside.device_model == dev]["band_hz"].unique()):  # and bands
            pr = pyr["predicted"][hz]  # the steps in which this band should be jammed
            # this phone and band, in its predicted steps only; between() is inclusive of both ends
            s = inside[(inside.device_model == dev) & (inside.band_hz == hz)
                       # continuation of the boolean mask: '&' keeps only the steps in which this band is predicted to be jammed
                       & inside.pyramid_step.between(min(pr), max(pr))]
            a = s.loc[s.in_jamming_minutes, "level_db"].dropna().to_numpy()  # readings in jamming portions
            b = s.loc[~s.in_jamming_minutes, "level_db"].dropna().to_numpy()  # readings in break portions
            if len(a) < 30 or len(b) < 30:  # if: fewer than 30 of either
                continue  # skip
            diff = float(np.median(a) - np.median(b))  # jamming minus break median; negative = predicted
            dl = cliffs_delta(a, b)  # negative = jamming-portion readings tend lower, the predicted sign
            # the row; f'{min(pr)}-{max(pr)}' is an f-string nested inside the outer f-string
            print(f"      {dev:<12} {BAND_SHORT[hz]:<4} {f'{min(pr)}-{max(pr)}':<16} {len(a):>7} {len(b):>7} "
                  f"{np.median(a):>9.2f} {np.median(b):>9.2f} {diff:>+9.2f} {dl:>8.3f}")  # f-string continued: the two medians, their difference and Cliff's delta for this row
            pyr_rows.append({  # appends the row for ground_truth/pyramid_observed.csv
                "device_model": dev, "band": BAND_SHORT[hz], "band_hz": hz,  # dict entry: phone model, short band name and frequency
                # dict entries (one row of ground_truth/pyramid_observed.csv): the first and last step in which the band should be jammed
                "predicted_step_first": min(pr), "predicted_step_last": max(pr),
                # dict entries: readings in the jamming minutes and in the break minutes
                "n_jamming": len(a), "n_break": len(b),
                # dict entry: median level in the jamming minutes, rounded to 3 decimals
                "median_jamming_db": round(float(np.median(a)), 3),
                # dict entry: median level in the break minutes
                "median_break_db": round(float(np.median(b)), 3),
                # dict entry: jamming minus break median; negative is the predicted direction
                "jamming_minus_break_db": round(diff, 3),
                # dict entry: Cliff's delta, jamming against break, rounded to 4 places
                "cliffs_delta": round(dl, 4),
                "direction_matches_prediction": bool(diff < 0),  # True if lower while jamming
            })
    print("\n  Nesting: are the bands disturbed only in the steps predicted?")  # a sub-heading in the printout
    # table header
    print(f"\n      {'device':<12} {'band':<4} {'pred steps':<11} {'med in-pred':>12} {'med out-pred':>13} {'in-out':>9}")
    for dev in sorted(inside["device_model"].unique()):  # for loop over phones
        for hz in sorted(inside[inside.device_model == dev]["band_hz"].unique()):  # and bands
            pr = pyr["predicted"][hz]  # this band's predicted steps
            # jamming portions only, this phone and band
            s = inside[(inside.device_model == dev) & (inside.band_hz == hz) & inside.in_jamming_minutes]
            a = s.loc[s.pyramid_step.between(min(pr), max(pr)), "level_db"].dropna()  # in predicted steps
            b = s.loc[~s.pyramid_step.between(min(pr), max(pr)), "level_db"].dropna()  # outside them
            if len(a) < 30 or len(b) < 30:  # if: too few of either
                # names this band; the conditional expression says which side is short
                why = (f"({BAND_SHORT[hz]} is predicted in steps {min(pr)}-{max(pr)}, leaving under 30 "
                       # continuation of the conditional expression: the explanation names the band, or says the in-prediction side is short
                       f"out-of-prediction readings)" if len(b) < 30 else "(too few in-prediction readings)")
                print(f"      {dev:<12} {BAND_SHORT[hz]:<4} {f'{min(pr)}-{max(pr)}':<11} {why}")  # the line
                continue  # next band
            # medians inside and outside the predicted steps, and their difference
            print(f"      {dev:<12} {BAND_SHORT[hz]:<4} {f'{min(pr)}-{max(pr)}':<11} {a.median():>12.2f} "
                  # continuation of the f-string: median outside the predicted steps, and in-minus-out with its sign
                  f"{b.median():>13.2f} {a.median() - b.median():>+9.2f}")
    # A pyramid adds bands one at a time, so a real pyramid response must be band-selective.
    print("\n  Band selectivity. A pyramid response must be band-selective; a")  # a sub-heading in the printout
    print("  deflection that moves every band at once cannot be one.")  # the next line of the same printed text
    # table header (the last column measures our CAPTURES, not our tests)
    print(f"\n      {'device':<12} {'deflection epochs':>18} {'bands moving together':>24} "
          # continuation of the header f-string: the last column title, right-aligned in 28 characters
          f"{'inside one of our captures':>28}")
    sel = []  # list of (fraction of bands together, fraction inside our captures) per phone
    for dev in sorted(inside["device_model"].unique()):  # for loop over phones
        s = inside[inside.device_model == dev]  # this phone's rows in the window
        # pivot_table reshapes to one row per time and one column per band (median if a cell has several)
        piv = s.pivot_table(index="timestamp_utc", columns="band", values="level_db", aggfunc="median")
        if piv.shape[1] < 2 or piv.empty:  # if: fewer than two bands, selectivity cannot be judged
            continue  # skip
        # per band: (reading - median) / IQR; apply() runs the lambda on each column; "or 1.0" replaces an IQR of 0
        z = (piv - piv.median()) / piv.apply(lambda c: iqr(c.dropna().to_numpy()) or 1.0)
        deep = z < -3.0  # boolean table: readings more than 3 IQRs below the band's median (a deep dip)
        any_deep = deep.any(axis=1)  # per time: True if any band dips deeply
        if not any_deep.any():  # if: no deep dip at all
            print(f"      {dev:<12} {'0':>18} {'-':>24} {'-':>28}")  # a row of zeros and dashes
            continue  # next phone
        # at dip times, the average fraction of bands dipping together (1.0 = every band at once)
        frac_all = float(deep[any_deep].sum(axis=1).mean() / piv.shape[1])
        ours = inside[inside.timestamp_utc.isin(piv.index[any_deep])]  # all band rows at the dip times
        frac_ours = float((ours.inside_our_capture.fillna("") != "").mean())  # share inside one of our captures
        sel.append((frac_all, frac_ours))  # keeps the two fractions
        print(f"      {dev:<12} {int(any_deep.sum()):>18} {100 * frac_all:>23.0f}% {100 * frac_ours:>27.0f}%")  # row
    # the interpretation is printed only when the computed fractions support it (an interpretive reading
    # of the table, with 0.5 as the line between "most" and "not most")
    if sel and all(fa > 0.5 and fo > 0.5 for fa, fo in sel):  # if: on every phone most bands move together, mostly
        print("\n      Interpretation: on every phone the deep deflections move most bands at once and mostly")  # text
        print("      fall inside our own spoofing captures, which fits a nearby broadband source and not a")  # the next line of the same printed text
        print("      band-stepping pyramid.")  # the next line of the same printed text
        # records the finding with its numbers
        note("The deep deflections seen during the 1.8.2 window move most bands at once ("
             # '+' joins, into the note text, each phone's share of bands dipping together (a generator inside str.join)
             + ", ".join(f"{100 * fa:.0f} %" for fa, _ in sel) + " of bands on average) and fall mostly inside "
             # continuation: the share of dips inside our captures, joined the same way, then the note's conclusion
             "our own spoofing captures (" + ", ".join(f"{100 * fo:.0f} %" for _, fo in sel) + "). A pyramid "
             "that adds bands one at a time cannot produce that, so these are not a response to it.")
    else:  # else: the fractions do not support that reading
        print("\n      The computed fractions do not support reading the deflections as all-band events "  # says so
              "inside our captures.")
    return pd.DataFrame(pyr_rows), sel  # the within-step table and the selectivity fractions


# function: the positive control, level inside each of our scheduled windows against the capture margins
def control_test(bands: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    # table header
    print(f"\n      {'device':<12} {'test':<7} {'band':<4} {'n in':>6} {'n out':>6} "
          f"{'med in':>8} {'med out':>8} {'in-out':>8} {'Cliff d':>8}")  # f-string continued: header cells for the two medians, their difference and Cliff's delta
    ctrl_rows = []  # empty list of control rows
    for _, sr in sessions.iterrows():  # for loop over our six sessions
        ss = pd.Timestamp(sr["scheduled_start_utc"], tz="UTC")  # scheduled start
        se = pd.Timestamp(sr["scheduled_end_utc"], tz="UTC")  # scheduled end
        cs = pd.Timestamp(sr["capture_start_utc"], tz="UTC")  # capture start
        ce = pd.Timestamp(sr["capture_end_utc"], tz="UTC")  # capture end
        for dev in sorted(bands["device_model"].unique()):  # for loop over phones
            for hz in sorted(bands[bands.device_model == dev]["band_hz"].unique()):  # and bands
                # this phone and band during the capture
                s = bands[(bands.device_model == dev) & (bands.band_hz == hz)
                          & (bands.timestamp_utc >= cs) & (bands.timestamp_utc <= ce)]  # element-wise condition: readings inside the capture window only
                # readings inside the scheduled window (start inclusive, end exclusive)
                a = s.loc[(s.timestamp_utc >= ss) & (s.timestamp_utc < se), "level_db"].dropna().to_numpy()
                # readings in the margins: before the start OR ("|") from the end onwards
                b = s.loc[(s.timestamp_utc < ss) | (s.timestamp_utc >= se), "level_db"].dropna().to_numpy()
                if len(a) < 60 or len(b) < 60:  # if: under a minute of either
                    continue  # skip
                diff = float(np.median(a) - np.median(b))  # inside minus margins; negative = gain turned down
                dl = cliffs_delta(a, b)  # negative = inside readings tend lower; -1.00 = no overlap at all
                print(f"      {dev:<12} {sr['test_code']:<7} {BAND_SHORT[hz]:<4} {len(a):>6} {len(b):>6} "  # row
                      # continuation of the f-string: the two medians, their signed difference and Cliff's delta
                      f"{np.median(a):>8.2f} {np.median(b):>8.2f} {diff:>+8.2f} {dl:>8.3f}")
                ctrl_rows.append({  # appends the row for ground_truth/handset_positive_control.csv
                    "device_model": dev, "our_test_code": sr["test_code"],  # dict entry: the phone model and the captured test it is checked against
                    "band": BAND_SHORT[hz], "band_hz": hz,  # dict entry: short band name and frequency
                    "n_inside_scheduled": len(a), "n_in_margins": len(b),  # dict entry: readings inside the scheduled test and in the margins around it
                    "median_inside_db": round(float(np.median(a)), 3),  # dict entry: median gain level inside the scheduled test
                    "median_margins_db": round(float(np.median(b)), 3),  # dict entry: median gain level in the margins
                    "inside_minus_margins_db": round(diff, 3),  # dict entry: inside median minus margins median; a negative value means the gain dropped
                    "cliffs_delta": round(dl, 4),  # dict entry: Cliff's delta of the inside readings against the margins
                    # the pass rule: at least 3 dB lower inside AND Cliff's delta below -0.3 (a clear, consistent drop)
                    "deflection_detected": bool(diff < -3.0 and dl < -0.3)})
    return pd.DataFrame(ctrl_rows)  # the control table


# function: runs all five tests on one set of band rows; returns a dict of their tables
def run_tests(bands: pd.DataFrame, ep: pd.DataFrame, plan, meta: list, pyr: dict, sessions: pd.DataFrame) -> dict:
    print("\n" + "=" * 78)  # a line of repeated characters that separates sections of the printout
    print("[6] CENTRAL COMPARISON - level_db by exposure group, per device per band")  # the section heading of the printout
    print("=" * 78)  # a line of repeated characters that separates sections of the printout
    cdf = comparison_test(bands)  # the central comparison
    print("\n" + "-" * 78)  # a line of repeated characters that separates sections of the printout
    print("  CONFOUND CHECK: is the contrast within a capture or between captures?")  # the section heading of the printout
    print("-" * 78)  # a line of repeated characters that separates sections of the printout
    wdf, within_ok = within_capture_test(ep, bands)  # the within-capture contrast
    print("\n" + "=" * 78)  # a line of repeated characters that separates sections of the printout
    print("[7] TRANSITION TIMING - do the steps fall where the schedule says?")  # the section heading of the printout
    print("=" * 78)  # a line of repeated characters that separates sections of the printout
    tdf = timing_test(bands, plan, meta)  # the transition timing
    print("\n" + "=" * 78)  # a line of repeated characters that separates sections of the printout
    print("[8] THE PYRAMID TEST - primary evidence")  # the section heading of the printout
    print("=" * 78)  # a line of repeated characters that separates sections of the printout
    pv, sel = pyramid_test(bands, pyr)  # the pyramid test
    print("\n" + "=" * 78)  # a line of repeated characters that separates sections of the printout
    print("[9] POSITIVE CONTROL - does this instrument detect interference at all?")  # the section heading of the printout
    print("=" * 78)  # a line of repeated characters that separates sections of the printout
    ctrl = control_test(bands, sessions)  # the positive control
    # dict literal: every table under a short name
    return {"comp": cdf, "within": wdf, "within_ok": within_ok, "timing": tdf, "pyramid": pv,
            # continuation of the returned dict: the band-selectivity fractions and the control table
            "selectivity": sel, "control": ctrl}


# function: the verdict logic, unchanged in; returns (verdict label, the counts it rests on)
def verdict_of(t: dict) -> tuple[str, dict]:
    pv, tdf, ctrl = t["pyramid"], t["timing"], t["control"]  # tuple unpacking: the three tables used
    n_match = int(pv["direction_matches_prediction"].sum()) if not pv.empty else 0  # pyramid: predicted direction
    n_tests = len(pv)  # pyramid contrasts computed
    big = pv[pv.cliffs_delta.abs() > 0.30] if not pv.empty else pv  # |delta| above 0.30, EITHER sign (rule as before)
    near_n = len(tdf[tdf.offset_s.abs() <= 30]) if not tdf.empty else 0  # timing tests within 30 s
    control_passes = bool(not ctrl.empty and ctrl.deflection_detected.any())  # at least one control contrast passes
    # decision rule: at least 4 contrasts, at least 75 % (and at least 3) in the predicted direction, 2 or more big
    strong_pyr = n_tests >= 4 and n_match >= max(3, int(0.75 * n_tests)) and len(big) >= 2
    strong_time = (not tdf.empty) and near_n >= max(2, int(0.5 * len(tdf)))  # at least half (and 2) near schedule
    if strong_pyr and strong_time:  # if: both the pyramid and the timing support contamination
        v = "EVIDENCE THAT THE JAMMING REACHED THE RECORDING POSITION"  # verdict label
    elif strong_pyr or strong_time:  # elif: only one of them does
        v = "INSUFFICIENT EVIDENCE, leaning towards contamination"  # verdict label
    elif n_tests == 0:  # elif: no pyramid data at all
        v = "INSUFFICIENT EVIDENCE"  # verdict label
    elif control_passes:  # elif: both tests negative and the control passed
        v = "EVIDENCE THAT THE 50 W JAMMING DID NOT REACH THE RECORDING POSITION"  # verdict label
    else:  # else: tests negative but the control failed too
        v = "INSUFFICIENT EVIDENCE THAT THE JAMMING REACHED THE RECORDING POSITION"  # verdict label
    # the counts, so the report can show what the verdict rests on
    return v, {"n_match": n_match, "n_tests": n_tests, "n_big_either": len(big),
               # dict entry of the returned counts: big effects in the predicted direction ('&' of two boolean Series, summed)
               "n_big_predicted": int(((pv.cliffs_delta.abs() > 0.30) & pv.direction_matches_prediction).sum())
               # else branch of the conditional expression: 0 when there is no pyramid table
               if not pv.empty else 0,
               # dict entries of the returned counts: timing tests within 30 s, all timing tests, and whether the control passed
               "near_n": near_n, "n_timing": len(tdf), "control_passes": control_passes,
               # dict entries of the returned counts: the two booleans the verdict rule rests on
               "strong_pyr": strong_pyr, "strong_time": strong_time}


# --------------------------------------------------------------------------
# 6. Descriptive control tables. DESCRIPTIVE ONLY: they never feed a verdict or a pass rule. The phase
# split comes from the catalogue; the onset/recovery rule was chosen AFTER the data had been examined, so it is not
# fixed in advance and must never be described as such. Its sensitivity is reported alongside.
# --------------------------------------------------------------------------

# function: per-phase deflections of every session's scheduled window, against that capture's margins
def control_phases(bands: pd.DataFrame, sessions: pd.DataFrame, phases: pd.DataFrame) -> pd.DataFrame:
    # the group's "first N minutes" of simultaneous spoofing and jamming, from the timeline script's phase table
    grp = phases[(phases.source_field == "group_description")
                 # continuation of the boolean mask: '&' keeps the group's 'simultaneous spoofing and jamming' phase rows
                 & (phases.phase_name == "simultaneous spoofing and jamming")]
    rows = []  # empty list of phase rows
    for _, sr in sessions.iterrows():  # for loop over our sessions
        ss = pd.Timestamp(sr["scheduled_start_utc"], tz="UTC")  # scheduled start
        se = pd.Timestamp(sr["scheduled_end_utc"], tz="UTC")  # scheduled end
        cs = pd.Timestamp(sr["capture_start_utc"], tz="UTC")  # capture start
        ce = pd.Timestamp(sr["capture_end_utc"], tz="UTC")  # capture end
        # boolean: the test overrides the group default with "no initial jamming" (the timeline script's phase table)
        override = bool(((phases.test_code == sr["test_code"])
                         # continuation: '&' requires the override phase; .any() asks whether such a row exists for this test
                         & (phases.phase_name == "no initial jamming (test-level override)")).any())
        first = grp.loc[grp.test_code == sr["test_code"], "offset_end_min"]  # this test's initial-phase length
        if override or first.empty:  # if: no initial jamming phase, the window is one phase
            parts = [("whole window", ss, se)]  # one phase
        else:  # else: the group default applies
            mid = ss + pd.Timedelta(minutes=float(first.iloc[0]))  # end of the initial phase (5 min)
            parts = [("initial jamming and spoofing", ss, mid), ("spoofing only", mid, se)]  # two phases
        for dev in sorted(bands["device_model"].unique()):  # for loop over phones
            for hz in sorted(bands[bands.device_model == dev]["band_hz"].unique()):  # and bands
                s = bands[(bands.device_model == dev) & (bands.band_hz == hz)  # this phone and band in the capture
                          & (bands.timestamp_utc >= cs) & (bands.timestamp_utc <= ce)]  # element-wise condition: readings inside the capture window only
                b = s.loc[(s.timestamp_utc < ss) | (s.timestamp_utc >= se), "level_db"].dropna().to_numpy()  # margins
                for name, p0, p1 in parts:  # for loop over the phases
                    # boolean mask on time: this phase's readings, as a NumPy array without NaN
                    a = s.loc[(s.timestamp_utc >= p0) & (s.timestamp_utc < p1), "level_db"].dropna().to_numpy()
                    rows.append({  # appends one descriptive row (computed whenever both sides hold readings)
                        # dict entries (one row of ground_truth/handset_control_phases.csv): the session, phone and band
                        "our_test_code": sr["test_code"], "device_model": dev, "band": BAND_SHORT[hz],
                        # dict entries: which phase, and its start as HH:MM:SS
                        "phase": name, "phase_start_utc": p0.strftime("%H:%M:%S"),
                        # dict entries: the phase end, and how many readings lie in the phase and in the margins
                        "phase_end_utc": p1.strftime("%H:%M:%S"), "n_phase": len(a), "n_margins": len(b),
                        # conditional expressions: medians and contrasts, or NaN when a side is empty
                        "median_phase_db": round(float(np.median(a)), 3) if len(a) else np.nan,
                        # dict entry: median of the margin readings, or NaN when there are none
                        "median_margins_db": round(float(np.median(b)), 3) if len(b) else np.nan,
                        # dict entry: phase median minus margin median (negative = gain turned down), 3 decimals
                        "phase_minus_margins_db": (round(float(np.median(a) - np.median(b)), 3)
                                                   # else branch: NaN when either side has no readings
                                                   if len(a) and len(b) else np.nan),
                        # dict entry: Cliff's delta, phase against margins; descriptive only
                        "cliffs_delta": round(cliffs_delta(a, b), 4) if len(a) and len(b) else np.nan,
                    })
    return pd.DataFrame(rows)  # the descriptive phase table


# function: onset and recovery of each session's deflection, for one threshold (dB) and hold time (s).
# Rule (chosen after the data were examined; descriptive only): smooth with a 15 s centred median; onset = the first
# instant from 2 min before the scheduled start at which the smoothed level stays more than thr_db below the margin
# median for at least hold_s; recovery = the first later instant at which it stays above that line for hold_s.
def onset_recovery(bands: pd.DataFrame, sessions: pd.DataFrame, thr_db: float, hold_s: float) -> list:
    rows = []  # empty list of rows
    for _, sr in sessions.iterrows():  # for loop over our sessions
        ss = pd.Timestamp(sr["scheduled_start_utc"], tz="UTC")  # scheduled start
        se = pd.Timestamp(sr["scheduled_end_utc"], tz="UTC")  # scheduled end
        cs = pd.Timestamp(sr["capture_start_utc"], tz="UTC")  # capture start
        ce = pd.Timestamp(sr["capture_end_utc"], tz="UTC")  # capture end
        for dev in sorted(bands["device_model"].unique()):  # for loop over phones
            for hz in sorted(bands[bands.device_model == dev]["band_hz"].unique()):  # and bands
                # boolean mask with three conditions (phone, band, from capture start) and sort: the time-ordered readings
                s = bands[(bands.device_model == dev) & (bands.band_hz == hz) & (bands.timestamp_utc >= cs)
                          & (bands.timestamp_utc <= ce)].sort_values("timestamp_utc")  # time-ordered readings
                mg = s.loc[(s.timestamp_utc < ss) | (s.timestamp_utc >= se), "level_db"]  # margin readings
                if len(s) < 60 or len(mg) < 60:  # if: under a minute of readings or of margin
                    continue  # skip
                line = float(mg.median()) - thr_db  # the level a reading must stay below to count as deflected
                # 15 s centred time-based rolling median on a time index (center=True centres the window)
                sm = s.set_index("timestamp_utc")["level_db"].rolling("15s", center=True, min_periods=3).median()
                below = (sm < line).to_numpy()  # boolean per reading: deflected
                tt = sm.index  # the times
                onset = recovery = None  # not found yet

                def held(i, state):  # nested function: True if 'below == state' holds from tt[i] for hold_s
                    j = i  # walks forward from i
                    while j < len(tt) and below[j] == state:  # while loop: same state continues
                        if (tt[j] - tt[i]).total_seconds() >= hold_s:  # if: it has lasted long enough
                            return True  # held
                        j += 1  # next reading
                    return False  # broke off too early

                start = int(np.searchsorted(tt, ss - pd.Timedelta(minutes=2)))  # first reading from 2 min before
                for i in range(start, len(tt)):  # for loop forward in time
                    if onset is None and below[i] and held(i, True):  # if: a sustained deflection starts here
                        onset = tt[i]  # onset found
                    elif onset is not None and not below[i] and held(i, False):  # elif: a sustained return
                        recovery = tt[i]  # recovery found
                        break  # done
                rows.append({  # appends the row
                    # dict entries (one row of ground_truth/handset_control_onset.csv): the session, phone and band
                    "our_test_code": sr["test_code"], "device_model": dev, "band": BAND_SHORT[hz],
                    # dict entries: the threshold (dB) and hold time (s) this row was computed with
                    "threshold_db": thr_db, "hold_s": hold_s,
                    "primary_setting": bool(thr_db == 3.0 and hold_s == 30.0),  # the setting quoted, 3 dB / 30 s
                    # dict entry: median of the margin readings, the level the threshold is measured from
                    "margin_median_db": round(float(mg.median()), 3),
                    # conditional expressions: times as text, or "" when not found
                    "onset_utc": onset.strftime("%H:%M:%S") if onset is not None else "",
                    # dict entry: the recovery time as HH:MM:SS, or '' when no sustained return was found
                    "recovery_utc": recovery.strftime("%H:%M:%S") if recovery is not None else "",
                    # dict entry: onset minus the scheduled start in seconds (round to 0.1 s)
                    "onset_minus_scheduled_start_s": (round((onset - ss).total_seconds(), 1)
                                                      # else branch: NaN when no onset was found
                                                      if onset is not None else np.nan),
                    # dict entry: recovery minus the scheduled end in seconds
                    "recovery_minus_scheduled_end_s": (round((recovery - se).total_seconds(), 1)
                                                       # else branch: NaN when no recovery was found
                                                       if recovery is not None else np.nan),
                })
    return rows  # list of rows for ground_truth/handset_control_onset.csv


# --------------------------------------------------------------------------
# 7. Figures (UTC time axes; lines break at gaps over MAX_GAP_S; frozen readings drawn in light grey)
# --------------------------------------------------------------------------

def plot_series(ax, g: pd.DataFrame, col: str, color, lw: float, label=None) -> None:  # function: one gap-aware line
    x, y = break_gaps(g["timestamp_utc"], g[col], MAX_GAP_S)  # NaN inserted at gaps, so they are not bridged
    ax.plot(x, y, lw=lw, color=color, label=label, zorder=3)  # the line


# function: one band's gain-control trace with live readings in colour and frozen readings in light grey
def plot_agc(ax, g: pd.DataFrame, color, lw: float, label=None) -> None:
    g = g.sort_values("timestamp_utc")  # time order
    live = g[~g.frozen]  # live readings
    plot_series(ax, live, "level_db", color, lw, label)  # the live trace
    fr = g[g.frozen]  # frozen readings
    if not fr.empty:  # if: any
        # small grey dots: visible as a flat stripe, clearly not part of the live trace
        ax.scatter(mdates.date2num(fr["timestamp_utc"]), fr["level_db"], s=1.2, color=C_GREY, zorder=2)


def figure_levels(bands: pd.DataFrame, plan, path: Path) -> None:  # function: Figure 4, gain control against time
    devs = sorted(bands["device_model"].unique())  # the two phone models, sorted: one column each
    t0 = bands["timestamp_utc"].min() - pd.Timedelta(minutes=5)  # axis start: 5 minutes before the first reading
    t1 = bands["timestamp_utc"].max() + pd.Timedelta(minutes=5)  # axis end: 5 minutes after the last
    # grid of axes, 4 rows by 2 columns, sharing the x-axis; squeeze=False keeps a 2-D array even for one column
    fig, axes = plt.subplots(len(BAND_ORDER), len(devs), figsize=(15, 11), sharex=True, squeeze=False)
    for ci, dev in enumerate(devs):  # for loop over columns (phones)
        for ri, hz in enumerate(BAND_ORDER):  # nested loop over rows (bands)
            ax = axes[ri][ci]  # the panel for this phone and band
            sub = bands[(bands.device_model == dev) & (bands.band_hz == hz)]  # boolean mask: this phone and band
            shade_windows(ax, plan, t0, t1, FIFTY_W_MIN)  # shades the 50 W windows behind the trace
            if sub.empty:  # if: this phone does not report this band
                # writes a note in the middle of the panel (transAxes: coordinates 0-1 within the axes)
                ax.text(0.5, 0.5, "band not reported", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9, color=C_GREY)  # keyword arguments of text(): the note is centred in the panel in small grey type
            else:  # else: there are readings
                for _, g in sub.groupby("logger_session_id"):  # one trace per capture
                    plot_agc(ax, g, C_BLUE, 0.5)  # live in blue, frozen in grey, gaps not bridged
            time_axis(ax, t0, t1)  # shared time-axis formatting
            if ci == 0:  # if: the left-hand column
                ax.set_ylabel(f"{BAND_SHORT[hz]}\n{Y_AGC}", fontsize=9)  # band name and plain quantity
            if ri == 0:  # if: the top row
                ax.set_title(dev, fontsize=11)  # the phone model as the column title
            ax.tick_params(labelsize=8)  # smaller tick labels
    for ax in axes[-1]:  # for loop over the bottom row of panels
        ax.set_xlabel("UTC on 14 September 2026")  # x-axis title
    fig.legend(handles=[  # one legend for the whole figure, built from three proxy artists
        Patch(facecolor=C_VERMILLION, alpha=0.35, label="Scheduled 50 W jamming at Test Area 1"),  # shading
        plt.Line2D([], [], color=C_BLUE, lw=1.2, label="Handset gain control, live readings"),  # the trace
        # proxy for the frozen readings
        plt.Line2D([], [], color=C_GREY, marker="o", ls="none", ms=3,
                   # keyword argument label: the legend text for the not-live readings
                   label="Not live: no measurement in the band and value unchanged >= 30 s")],
        # keyword arguments: legend centred below the panels, three entries in one row, no frame
        loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=3, frameon=False, fontsize=9.5)
    fig.suptitle("Handset automatic gain control against time, by device and band",  # the figure title
                 fontsize=12.5, y=0.995)  # keyword arguments of suptitle(): the title's font size and its height near the top edge
    fig.tight_layout(rect=(0, 0.02, 1, 0.985))  # packs the panels inside the given box, leaving room for legend
    save_fig(fig, path)  # writes the PNG without time stamps


def figure_distributions(bands: pd.DataFrame, path: Path) -> None:  # function: Figure 5, box plots by exposure group
    devs = sorted(bands["device_model"].unique())  # the two phone models
    groups = [G50, GHI, GBL]  # the three compared groups (the excluded unknown-power group is not drawn)
    cols = {G50: C_VERMILLION, GHI: C_ORANGE, GBL: C_SKY}  # dict: group -> box colour
    short = {G50: "50 W\nTA1", GHI: "other\n>1 W", GBL: "baseline"}  # dict: group -> short tick label
    fig, axes = plt.subplots(len(BAND_ORDER), len(devs), figsize=(12.5, 11), squeeze=False)  # 4 x 2 panels
    for ci, dev in enumerate(devs):  # for loop over phones
        for ri, hz in enumerate(BAND_ORDER):  # nested loop over bands
            ax = axes[ri][ci]  # this panel
            sub = bands[(bands.device_model == dev) & (bands.band_hz == hz)]  # this phone and band (live readings)
            data, labs, cs, ns = [], [], [], []  # the data arrays, labels, colours and counts for the boxes
            for g in groups:  # for loop over the exposure groups
                v = sub.loc[sub.exposure_group == g, "level_db"].dropna().to_numpy()  # readings in this group
                if len(v) >= 10:  # if: at least 10 readings, enough for a box
                    data.append(v); labs.append(short[g])  # two statements: the data and its short label
                    cs.append(cols[g]); ns.append(len(v))  # two statements: its colour and its count
            if not data:  # if: no group had enough readings
                ax.text(0.5, 0.5, "insufficient data", transform=ax.transAxes,  # a note in the middle of the panel
                        ha="center", va="center", fontsize=9, color=C_GREY)  # keyword arguments of text(): the note is centred in the panel in small grey type
                ax.set_xticks([])  # no x ticks
            else:  # else: draw the boxes
                # box plot: box = IQR, line = median, whiskers up to 1.5 IQR out; fillable boxes; outlier dots hidden
                bp = ax.boxplot(data, patch_artist=True, showfliers=False,
                                widths=0.55, medianprops=dict(color=C_BLACK, lw=1.6))  # keyword arguments of boxplot(): box width, and a thick black line for the median
                for patch, c in zip(bp["boxes"], cs):  # zip pairs each box with its colour
                    patch.set_facecolor(c); patch.set_alpha(0.75)  # two statements: fill colour and transparency
                    patch.set_edgecolor(C_BLACK)  # black outline
                ax.set_xticks(range(1, len(labs) + 1))  # boxes sit at x = 1, 2, 3
                # tick labels "group\nn=count" built by a list comprehension over zipped labels and counts
                ax.set_xticklabels([f"{l}\nn={n}" for l, n in zip(labs, ns)], fontsize=7.5)
            ax.grid(axis="y", color=C_GREY, alpha=0.25, lw=0.5)  # faint horizontal grid
            if ci == 0:  # if: left column
                ax.set_ylabel(f"{BAND_SHORT[hz]}\n{Y_AGC}", fontsize=9)  # y label in plain words
            if ri == 0:  # if: top row
                ax.set_title(dev, fontsize=11)  # column title
            ax.tick_params(labelsize=8)  # smaller tick labels
    # two-line figure title
    fig.suptitle("Gain control by exposure group (live readings), per device and band\n"
                 # second line of the title, then keyword arguments for its font size and height
                 "(the two devices use incompatible scales and are never pooled)", fontsize=12, y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.975))  # packs the panels below the title
    save_fig(fig, path)  # writes the PNG


def figure_sats_cn0(ep: pd.DataFrame, plan, path: Path) -> None:  # function: Figure 6, satellites and C/N0
    devs = sorted(ep["device_model"].unique())  # the two phone models
    t0 = ep["timestamp_utc"].min() - pd.Timedelta(minutes=5)  # axis start
    t1 = ep["timestamp_utc"].max() + pd.Timedelta(minutes=5)  # axis end
    fig, axes = plt.subplots(2, len(devs), figsize=(15, 6.5), sharex=True, squeeze=False)  # 2 rows x 2 columns
    for ci, dev in enumerate(devs):  # for loop over phones
        sub = ep[ep.device_model == dev]  # this phone's epochs
        # nested loop over two (column name, label) pairs with enumerate: row 0 satellites, row 1 median C/N0
        for ri, (col, lab) in enumerate([("n_satellites_epoch", "satellites tracked"),
                                         ("median_cn0_dbhz", "median C/N0 (dB-Hz)")]):  # list element: the second row, the median C/N0 with its axis label
            ax = axes[ri][ci]  # this panel
            shade_windows(ax, plan, t0, t1, FIFTY_W_MIN)  # 50 W windows behind
            for _, g in sub.groupby("logger_session_id"):  # one line per capture
                plot_series(ax, g.sort_values("timestamp_utc"), col, C_GREEN if ri else C_BLUE, 0.5)  # gap-aware
            time_axis(ax, t0, t1)  # time-axis formatting
            if ci == 0:  # if: left column
                ax.set_ylabel(lab, fontsize=9)  # y label
            if ri == 0:  # if: top row
                ax.set_title(dev, fontsize=11)  # column title
            ax.tick_params(labelsize=8)  # smaller tick labels
    for ax in axes[-1]:  # bottom row
        ax.set_xlabel("UTC on 14 September 2026")  # x-axis title
    # one legend entry, for the shaded windows
    fig.legend(handles=[Patch(facecolor=C_VERMILLION, alpha=0.35, label="Scheduled 50 W jamming at Test Area 1")],
               # keyword arguments: legend centred below the panels, one column, no frame
               loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=1, frameon=False, fontsize=10)
    fig.suptitle("Satellites tracked and median carrier-to-noise density against time",  # figure title
                 fontsize=12.5, y=0.995)  # keyword arguments of suptitle(): the title's font size and its height near the top edge
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))  # packs the panels
    save_fig(fig, path)  # writes the PNG


def figure_pyramid(pyr: dict, bands: pd.DataFrame, path: Path) -> None:  # function: Figure 7, predicted vs observed
    devs = sorted(bands["device_model"].unique())  # the two phone models
    t0 = pd.Timestamp(pyr["entry"]["start_utc"]) - pd.Timedelta(minutes=3)  # 3 minutes before the pyramid
    t1 = pd.Timestamp(pyr["entry"]["end_utc"]) + pd.Timedelta(minutes=3)  # 3 minutes after it
    # one column of 1 + 2 panels sharing x; gridspec_kw sets their relative heights (list + list concatenates)
    fig, axes = plt.subplots(1 + len(devs), 1, figsize=(14, 4.2 + 2.6 * len(devs)), sharex=True,
                             gridspec_kw={"height_ratios": [1.5] + [1.4] * len(devs)})  # keyword argument gridspec_kw=: the top panel slightly taller than the device panels
    ax = axes[0]  # the top panel: the prediction
    for ri, hz in enumerate(BAND_ORDER):  # for loop over bands with their row index
        y = len(BAND_ORDER) - 1 - ri  # row position, reversed so L1 is at the top
        for st in pyr["steps"]:  # for loop over the 15 steps
            if st["step"] not in pyr["predicted"][hz]:  # if: the band is not jammed in this step
                continue  # no bar
            s, e = pd.Timestamp(st["jam_start_utc"]), pd.Timestamp(st["jam_end_utc"])  # the jamming minutes
            # a hatched bar over the jamming portion in this band's row
            ax.barh(y, mdates.date2num(e) - mdates.date2num(s), left=mdates.date2num(s),
                    # continuation of the barh call: bar height, vermillion fill with black edge and // hatching, drawn in front
                    height=0.62, color=C_VERMILLION, edgecolor=C_BLACK, lw=0.6, hatch="//", zorder=3)
    for st in pyr["steps"]:  # for loop over the steps again, for step markers
        # a thin vertical line at each step start
        ax.axvline(mdates.date2num(pd.Timestamp(st["step_start_utc"])), color=C_GREY, lw=0.5, alpha=0.7, zorder=1)
        # the step number printed at the middle of the step, above the top row
        ax.text(mdates.date2num(pd.Timestamp(st["step_start_utc"]) + timedelta(minutes=pyr["step_min"] / 2)),
                # continuation of the text call: y position above the top row, the step number as text, small grey font
                len(BAND_ORDER) - 0.35, str(st["step"]), fontsize=7, ha="center", color=C_GREY)
    ax.set_yticks(range(len(BAND_ORDER)))  # four y ticks
    ax.set_yticklabels([BAND_SHORT[hz] for hz in reversed(BAND_ORDER)], fontsize=9)  # labels bottom-up: B1I ... L1
    ax.set_ylim(-0.6, len(BAND_ORDER) - 0.05)  # vertical limits
    # panel title stating the parsed structure (a plain string joined to two f-strings)
    ax.set_title("Predicted: the pyramid as scheduled in the transmission plan "
                 f"({pyr['n_steps']} steps, {pyr['jam_min']} min jamming and "  # f-string continued: the number of steps and the jamming minutes of each
                 f"{pyr['brk_min']} min break each)", fontsize=10.5)  # f-string continued: the break minutes of each step, closing the title
    ax.tick_params(labelsize=8)  # smaller tick labels
    styles = {1575420000: C_BLUE, 1602000000: C_ORANGE, 1176450000: C_GREEN, 1561098000: C_PURPLE}  # band colours
    for di, dev in enumerate(devs):  # for loop over phones: one observed panel each
        ax = axes[1 + di]  # the panel below the prediction
        # boolean mask: this phone's live readings within the plotted window
        sub = bands[(bands.device_model == dev) & (bands.timestamp_utc >= t0) & (bands.timestamp_utc <= t1)]
        for st in pyr["steps"]:  # step markers, as in the top panel
            # axvline: a thin vertical line at each step start, behind the traces
            ax.axvline(mdates.date2num(pd.Timestamp(st["step_start_utc"])), color=C_GREY, lw=0.5, alpha=0.7, zorder=1)
        for hz in BAND_ORDER:  # for loop over the bands
            s2 = sub[sub.band_hz == hz].sort_values("timestamp_utc")  # this band's readings in time order
            if s2.empty:  # if: none
                continue  # skip
            v = s2["level_db"].to_numpy()  # the readings as an array
            # robust scaling: minus the median, divided by the IQR (or 1 if the IQR is ~0), so both phones share an axis
            s2 = s2.assign(z=(v - np.median(v)) / (iqr(v) if iqr(v) > 1e-9 else 1.0))
            plot_series(ax, s2, "z", styles[hz], 0.6, BAND_SHORT[hz])  # the scaled trace, gaps not bridged
        ax.axhline(0, color=C_GREY, lw=0.6)  # a horizontal line at 0, i.e. at the median
        ax.set_ylabel("AGC level,\nmedian-centred,\nIQR-scaled", fontsize=8.5)  # y label in plain words
        ax.set_title(f"Observed: {dev} (live readings)", fontsize=10.5)  # panel title
        ax.legend(loc="upper right", fontsize=7.5, ncol=4, frameon=False)  # legend of the four bands
        ax.tick_params(labelsize=8)  # smaller tick labels
        ax.grid(axis="y", color=C_GREY, alpha=0.2, lw=0.4)  # faint horizontal grid
    for a in axes:  # for loop over all three panels
        time_axis(a, t0, t1)  # the same time-axis formatting
    axes[-1].set_xlabel("UTC on 14 September 2026")  # x label on the bottom panel only
    fig.suptitle("Test 1.8.2 pyramid: predicted step structure against observed gain control",  # figure title
                 fontsize=12.5, y=0.997)  # keyword arguments of suptitle(): the title's font size and its height near the top edge
    fig.tight_layout(rect=(0, 0, 1, 0.98))  # packs the panels
    save_fig(fig, path)  # writes the PNG


def figure_distance(ep: pd.DataFrame, plan, path: Path) -> None:  # function: Figure 3, distance to the jammer site
    devs = sorted(ep["device_model"].unique())  # the two phone models
    t0 = ep["timestamp_utc"].min() - pd.Timedelta(minutes=5)  # axis start
    t1 = ep["timestamp_utc"].max() + pd.Timedelta(minutes=5)  # axis end
    # one panel per phone, stacked, sharing the x-axis
    fig, axes = plt.subplots(len(devs), 1, figsize=(14, 3.1 * len(devs)), sharex=True, squeeze=False)
    for di, dev in enumerate(devs):  # for loop over phones
        ax = axes[di][0]  # this phone's panel
        sub = ep[ep.device_model == dev]  # its epochs
        shade_windows(ax, plan, t0, t1, FIFTY_W_MIN)  # 50 W windows behind
        ok = sub[sub.position_plausible]  # boolean mask: plausible fixes
        bad = sub[~sub.position_plausible & sub["dist_to_tx_km"].notna()]  # "~" negates: rejected but present fixes
        for _, g in ok.groupby("logger_session_id"):  # one line per capture
            plot_series(ax, g.sort_values("timestamp_utc"), "dist_to_tx_km", C_BLUE, 0.8)  # gaps not bridged
        if not bad.empty:  # if: there are rejected fixes
            # crosses for them; np.clip draws values beyond 30 km at 30 km so a 244 km excursion stays on the axis
            ax.scatter(mdates.date2num(bad["timestamp_utc"]), np.clip(bad["dist_to_tx_km"], 0, 30),
                       s=9, color=C_VERMILLION, marker="x", zorder=4,  # keyword arguments of scatter(): small vermillion crosses drawn above the other points
                       label="rejected fix (beyond 25 km of Stave or above 200 km/h), drawn at most 30 km")  # the label of the rejected fixes
            ax.legend(loc="upper right", fontsize=8, frameon=False)  # legend for the crosses
        ax.set_ylabel("distance to\ntransmitter (km)", fontsize=9)  # y label
        ax.set_title(dev, fontsize=11)  # panel title
        ax.set_ylim(0, 30)  # y range 0-30 km
        time_axis(ax, t0, t1)  # time-axis formatting
        ax.grid(axis="y", color=C_GREY, alpha=0.25, lw=0.5)  # faint horizontal grid
        ax.tick_params(labelsize=8)  # smaller tick labels
    axes[-1][0].set_xlabel("UTC on 14 September 2026")  # x label on the bottom panel
    # figure legend for the shaded windows
    fig.legend(handles=[Patch(facecolor=C_VERMILLION, alpha=0.35, label="Scheduled 50 W jamming at Test Area 1")],
               # keyword arguments: legend centred below the panels, one column, no frame
               loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=1, frameon=False, fontsize=10)
    fig.suptitle("Reported handset distance to the Test Area 1 transmitter site\n"  # two-line title
                 "(survey point SENDER, marked FFI Jammer)", fontsize=12, y=0.997)  # second title line: a fixed string; the Appendix A reference is dropped
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))  # packs the panels
    save_fig(fig, path)  # writes the PNG


# function: Figure 8, the positive control for one of our tests; dist_km is that session's distance to SENDER
def figure_control(bands: pd.DataFrame, sessions: pd.DataFrame, plan, test_code: str,
                   # continuation of the parameter list: the session's distance to SENDER in km, and the output path
                   dist_km: float, path: Path) -> None:
    sr = sessions[sessions.test_code == test_code].iloc[0]  # the inventory row of that test
    cs = pd.Timestamp(sr["capture_start_utc"], tz="UTC")  # capture start as a UTC timestamp
    ce = pd.Timestamp(sr["capture_end_utc"], tz="UTC")  # capture end
    ss = pd.Timestamp(sr["scheduled_start_utc"], tz="UTC")  # scheduled start
    se = pd.Timestamp(sr["scheduled_end_utc"], tz="UTC")  # scheduled end
    t0, t1 = cs - pd.Timedelta(minutes=2), ce + pd.Timedelta(minutes=2)  # plotted range: the capture +/- 2 min
    devs = sorted(bands["device_model"].unique())  # the two phone models
    fig, axes = plt.subplots(len(devs), 1, figsize=(13, 3.5 * len(devs)), sharex=True, squeeze=False)  # one per phone
    cols = {1575420000: C_BLUE, 1602000000: C_ORANGE, 1176450000: C_GREEN, 1561098000: C_PURPLE}  # band colours
    for di, dev in enumerate(devs):  # for loop over phones
        ax = axes[di][0]  # this phone's panel
        ax.axvspan(mdates.date2num(ss), mdates.date2num(se), color=C_GREEN, alpha=0.16, zorder=0, lw=0)  # our window
        shade_windows(ax, plan, t0, t1, FIFTY_W_MIN)  # shades any 50 W windows in range
        # boolean mask: this phone, within the plotted range
        sub = bands[(bands.device_model == dev) & (bands.timestamp_utc >= t0) & (bands.timestamp_utc <= t1)]
        for hz in sorted(sub["band_hz"].unique()):  # for loop over the bands present
            plot_agc(ax, sub[sub.band_hz == hz], cols[hz], 0.7, BAND_SHORT[hz])  # live in colour, frozen grey
        ax.axvline(mdates.date2num(ss), color=C_GREEN, lw=1.4, ls="--", zorder=4)  # dashed line at scheduled start
        ax.axvline(mdates.date2num(se), color=C_GREEN, lw=1.4, ls="--", zorder=4)  # dashed line at scheduled end
        ax.set_ylabel(Y_AGC, fontsize=9)  # y label in plain words
        ax.set_title(dev, fontsize=11)  # panel title
        ax.legend(loc="lower left", fontsize=8, ncol=4, frameon=False)  # band legend
        time_axis(ax, t0, t1)  # time-axis formatting
        ax.grid(axis="y", color=C_GREY, alpha=0.25, lw=0.5)  # faint horizontal grid
        ax.tick_params(labelsize=8)  # smaller tick labels
    axes[-1][0].set_xlabel("UTC on 14 September 2026")  # x label
    fig.legend(handles=[  # figure legend with three proxies
        Patch(facecolor=C_GREEN, alpha=0.35, label=f"Scheduled window of our own test {test_code}"),  # our window
        Patch(facecolor=C_VERMILLION, alpha=0.35,  # the 50 W windows, with this session's measured distance
              # keyword argument label: the 50 W legend entry, with this session's measured distance to one decimal
              label=f"Scheduled 50 W jamming at Test Area 1, {dist_km:.1f} km from this session's position"),
        # proxy for the frozen readings
        plt.Line2D([], [], color=C_GREY, marker="o", ls="none", ms=3,
                   # keyword argument label: the legend text for the not-live readings
                   label="Not live: no measurement in the band and value unchanged >= 30 s")],
        # keyword arguments: legend centred below the panels, one column, no frame
        loc="lower center", bbox_to_anchor=(0.5, -0.07), ncol=1, frameon=False, fontsize=9)
    fig.suptitle("Positive control: the handsets register a local milliwatt transmitter\n"  # two-line figure title
                 # second line of the title, then keyword arguments for its font size and height
                 "and register nothing from a distant 50 W one", fontsize=12.5, y=0.998)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))  # packs the panels
    save_fig(fig, path)  # writes the PNG


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:  # function: runs the whole section in order
    PROCESSED.mkdir(parents=True, exist_ok=True)  # creates Data/processed if absent; exist_ok: no error if present
    GRAPH.mkdir(parents=True, exist_ok=True)  # creates Graph/ likewise

    print("=" * 78)  # string repetition: a rule of 78 "=" characters
    print("CONTAMINATION INDICATION FROM THE HANDSET RECORDS")  # report title
    print("=" * 78)  # closing rule

    s1 = load_section1()  # the timeline script loaded as a module
    plan, _ = load_schedule(s1, PROCESSED)  # the schedule, with the timeline script's power rule and the timeline script's plan time zone
    sessions = pd.read_csv(PROCESSED / "ground_truth" / "session_ground_truth.csv")  # the timeline script's session inventory
    phases = pd.read_csv(PROCESSED / "ground_truth" / "test_phases.csv")  # the timeline script's phase table (for the descriptive control-phase table)
    print(f"\n[1] Schedule loaded from the timeline script's parser: {len(plan)} entries on "  # entry and session counts
          f"{TEST_DAY}, {len(sessions)} capture sessions.")  # f-string continued: the test day and the number of capture sessions
    unk = [p["test_code"] for p in plan if p["power_w"] is None]  # entries whose power no document states
    print(f"      Entries of unknown power (excluded group): {', '.join(unk) if unk else 'none'}")  # listed

    # ---- the prediction, stated before the data is examined -----------------
    pyr = parse_pyramid(plan, flag)  # the 1.8.2 structure and prediction, before any handset file is opened
    # dict comprehension: the prediction re-keyed from plan band name to the handset band's frequency
    pyr["predicted"] = {hz: pyr["predicted_by_band"].get(nm, []) for nm, hz in PLAN_BAND_TO_HZ.items()}
    print("\n[2] PREDICTION FIXED IN ADVANCE, parsed from the transmission plan's own comment for test 1.8.2")  # the section heading of the printout
    print(f"      Structure recovered: {pyr['n_steps']} steps of {pyr['step_min']} min, "  # the recovered structure
          # continuation of the f-string: jamming and break minutes and how many band names were parsed
          f"{pyr['jam_min']} min jamming then {pyr['brk_min']} min break, from {pyr['n_tokens']} band tokens.")
    print(f"      {pyr['n_steps']} x {pyr['step_min']} min = {pyr['n_steps'] * pyr['step_min']} min, against a "  # check
          # continuation: the scheduled window in minutes (end minus start, in seconds, / 60), to check the arithmetic
          f"scheduled window of {(pyr['entry']['end_utc'] - pyr['entry']['start_utc']).total_seconds() / 60:.0f} min.")
    print(f"      Scheduled {pyr['entry']['start_utc']:%H:%M}-{pyr['entry']['end_utc']:%H:%M} UTC "  # the window
          f"({pyr['entry']['start_hhmm']}-{pyr['entry']['end_hhmm']} Norwegian local).")  # f-string continued: the same window as the plan prints it in Norwegian local time
    # how many bands the pyramid cycles through, and how many the handsets observe (computed)
    print(f"      Of the {len(pyr['predicted_by_band'])} bands the pyramid cycles through, the handsets observe "
          # continuation of the f-string: how many of those bands the handsets observe (4)
          f"{len(PLAN_BAND_TO_HZ)}.")
    print("      Predicted steps in which each is disturbed:")  # a sub-heading in the printout
    for hz in BAND_ORDER:  # for loop over L1, G1, L5, B1I
        a = pyr["predicted"][hz]  # the predicted steps for this band
        # band, label, first-last step and how many of 15 (">4" right-aligns, "<28" left-aligns in 28 characters)
        print(f"          {BAND_SHORT[hz]:>4} {BAND_LABEL[hz]:<28} steps {min(a)}-{max(a)}  ({len(a)} of {pyr['n_steps']})")
    print("      Direction, fixed in advance: in-band interference drives level_db MORE NEGATIVE.")  # the direction
    # DataFrame from a list comprehension, one dict per step ("**{...}" adds predicted_<band> True/False); to CSV
    pd.DataFrame([{"step": r["step"],
                   "step_start_utc": r["step_start_utc"].strftime("%Y-%m-%d %H:%M:%S"),  # dict entry: start of the step as text
                   "jam_end_utc": r["jam_end_utc"].strftime("%Y-%m-%d %H:%M:%S"),  # dict entry: end of the step's jamming minutes as text
                   "step_end_utc": r["step_end_utc"].strftime("%Y-%m-%d %H:%M:%S"),  # dict entry: end of the step, including its break, as text
                   "bands_scheduled": r["bands"],  # dict entry: the bands the plan schedules for the step
                   # dict unpacking (**) into the row: predicted_<band> True/False for each handset band, from a dict comprehension
                   **{f"predicted_{BAND_SHORT[hz]}": (r["step"] in pyr["predicted"][hz]) for hz in BAND_ORDER}}
                  # the outer comprehension's loop (one dict per step), then the table is written to ground_truth/pyramid_prediction.csv
                  for r in pyr["steps"]]).to_csv(PROCESSED / "ground_truth" / "pyramid_prediction.csv", index=False, lineterminator="\n")

    # ---- parse handsets -----------------------------------------------------
    bands, ep, meta, collapse = parse_handsets()  # band rows, epoch rows, per-file facts, collapse statistics
    print(f"\n[3] Handsets parsed: {len(meta)} captures, {len(ep)} epochs, {len(bands)} band-rows after "  # counts
          # continuation of the f-string: the raw gain-control entries summed over every file (a generator in sum())
          f"collapsing {sum(m['agc_entries'] for m in meta)} raw gain-control entries.")
    for m in meta:  # for loop over the handset files
        # one line per file: model, session, window, epochs and entries -> rows (":5d" = integer 5 wide)
        print(f"      {m['model']:<12} session {m['session_id']:>2}  {m['start_utc']:%H:%M:%S}-{m['end_utc']:%H:%M:%S} UTC  "
              # continuation of the f-string: epochs, raw entries and the rows kept after collapsing
              f"{m['n_epochs']:5d} epochs  {m['agc_entries']:6d} -> {m['agc_rows_kept']:6d} rows")
    print("\n      How the two devices report gain control. The Pixel emits one reading")  # fixed text
    print("      per radio frequency and repeats it for each constellation using that")  # the next line of the same printed text
    print("      frequency; the Samsung runs a separate chain per constellation on the")  # the next line of the same printed text
    print("      same frequency. Groups collapsed to one row per band, and how often the")  # the next line of the same printed text
    print("      collapsed readings disagreed:")  # the next line of the same printed text
    for key in sorted(collapse["n_total"]):  # for loop over (device, band) keys, sorted
        n, tot = collapse["n_with_spread"].get(key, 0), collapse["n_total"][key]  # disagreeing and total epochs
        msp = collapse["median_spread"].get(key)  # median disagreement in dB, or None
        # the counts and percentage; the conditional expression adds the median spread only when there is one
        print(f"          {key[0]:<12} {key[1]:<4} {n:6d} of {tot:6d} ({100 * n / tot:5.1f} %) disagreed"
              + (f", median spread {msp:.2f} dB" if msp else ""))  # string concatenation: adds the median spread only when there is one
        if n / tot > 0.5:  # if: the readings disagreed in more than half the epochs (separate chains)
            # records that the median across chains is used for that device and band
            note(f"On the {key[0]}, the {key[1]} band carries more than one constellation-specific gain-control "
                 # continuation of the note: the share of epochs with several chains and the median spread in dB
                 f"chain in {100 * n / tot:.0f} % of epochs (median spread {collapse['median_spread'].get(key, 0):.2f} "
                 "dB). The per-band value used is the median across those chains.")

    # ---- live readings --------------------------------------
    bands = flag_frozen(bands)  # adds value_run_s, frozen and frozen_reason
    print("\n[3b] LIVE READINGS: a reading is not live when its band has no raw measurement in that epoch AND")  # a line of repeated characters that separates sections of the printout
    print("      its value has been unchanged for at least 30 s (mechanism checked in the raw files).")  # the next line of the same printed text
    c12 = sessions[sessions.test_code == "2.6.12"].iloc[0]  # our stationary-spoofer session, the control window
    w0 = pd.Timestamp(c12["scheduled_start_utc"], tz="UTC")  # its scheduled start (15:50 UTC)
    w1 = pd.Timestamp(c12["scheduled_end_utc"], tz="UTC")  # and end (16:05 UTC)
    inw = (bands.timestamp_utc >= w0) & (bands.timestamp_utc < w1)  # boolean: inside the 2.6.12 control window
    no_meas = bands.n_satellites_band == 0  # boolean: no measurement in the band
    long_run = bands.value_run_s >= FROZEN_RUN_S  # boolean: value unchanged >= 30 s
    # the four counts asked for, overall and inside the control window
    for lab, m in (("unchanged >= 30 s alone (kept)", long_run & ~no_meas),
                   # tuple: label and boolean mask for readings with no measurement but a changing value (kept)
                   ("no measurement alone (kept: the value still changes)", no_meas & ~long_run),
                   # tuple: label and boolean mask for readings meeting both conditions (removed as not live)
                   ("both = not live (removed)", long_run & no_meas)):
        print(f"      {lab:<55} {int(m.sum()):6d}   in the 2.6.12 window {int((m & inw).sum()):5d}")  # one line
    print("      Not live, per device and band:")  # a sub-heading in the printout
    for (dev, bd), g in bands.groupby(["device_model", "band"]):  # for loop over (phone, band) groups
        if g.frozen.any():  # if: any reading in this group is frozen
            print(f"          {dev:<12} {bd:<4} {int(g.frozen.sum()):5d} of {len(g):6d}")  # count of all

    # ---- tag ----------------------------------------------------------------
    tags = tag_times(ep["timestamp_utc"], plan, sessions, HIGH_POWER_W, FIFTY_W_MIN)  # what was scheduled when
    ep = ep.merge(tags, on="timestamp_utc", how="left")  # left join: each epoch gains its tag columns
    bands = bands.merge(tags, on="timestamp_utc", how="left")  # the same for the band rows
    n_unk_ep = int((ep.exposure_group == GUNK).sum())  # phone-seconds in the excluded unknown-power group
    span_unk = ep.loc[ep.exposure_group == GUNK, "timestamp_utc"]  # their times
    print(f"\n      Entries of unknown power: {n_unk_ep} phone-seconds fall in '{GUNK}'"  # how many, and when
          # '+' joins the time span of the excluded seconds when there are any (a conditional expression), else ''
          + (f" ({span_unk.min():%H:%M:%S}-{span_unk.max():%H:%M:%S} UTC)" if n_unk_ep else "")
          + "; they are excluded from every group comparison.")

    # ---- POSITION, before any level comparison ------------------------------
    ep = analyse_position(ep)  # distances, steps (within each capture), speeds and plausibility
    print("\n" + "=" * 78)  # a line of repeated characters that separates sections of the printout
    print("[4] POSITION CHECK (done before the level comparisons)")  # the section heading of the printout
    print("=" * 78)  # a line of repeated characters that separates sections of the printout
    sep = ep["position_fields_separation_m"].dropna()  # separations between the two position fields, gaps dropped
    print("\n  The two position fields. 'lat'/'lon' against gnssLocations[0]:")  # a sub-heading in the printout
    print(f"      n={len(sep)}, median separation {sep.median():.3f} m, maximum {sep.max():.3f} m")  # the numbers
    if sep.max() < 1.0:  # if: the two fields never differ by as much as a metre
        # fixed text, printed only when the two position fields never differ
        print("      These are NOT two independent estimates: the top-level lat/lon is a copy of gnssLocations[0].")
        # records the finding
        note(f"The top-level lat/lon and gnssLocations[0] are identical in all {len(sep)} epochs (maximum "
             # continuation of the note: the largest separation (0.000 m) and what it means
             f"separation {sep.max():.3f} m), so they are one estimate reported twice, not two estimates.")
        # records that the brief's expectation was wrong, quoting the largest accuracy radius
        flag("It was expected beforehand that the two position fields would differ by roughly 2.5 km in at "
             "least one capture. They do not differ at all. The kilometre-scale quantities in these files are the "
             # continuation of the anomaly text: the largest accuracy radius in metres
             f"'accuracy' radius (up to {ep['position_accuracy_m'].max():.0f} m) and genuine movement of the "
             "reported position over time, including a spoofed excursion.")
    print(f"      Provider for every fix: {', '.join(sorted(ep['location_provider'].dropna().unique()))}")  # providers
    print(f"      Accuracy radius: median {ep['position_accuracy_m'].median():.1f} m, "  # accuracy radius
          f"maximum {ep['position_accuracy_m'].max():.0f} m")  # f-string continued: the largest accuracy radius in metres
    nimp = int((~ep["position_plausible"]).sum())  # "~" negates; summing booleans counts implausible fixes
    print(f"\n  Rejected fixes: {nimp} of {len(ep)} ({100 * nimp / len(ep):.2f} %) lie more than "  # the count
          # continuation of the f-string: the two rejection thresholds, 25 km and 200 km/h
          f"{PLAUSIBLE_RADIUS_KM:.0f} km from the Stave landmark or imply more than {MAX_SPEED_KMH:.0f} km/h.")
    far = ep.nlargest(3, "dist_to_stave_km")[["device_model", "timestamp_utc", "dist_to_stave_km",  # 3 farthest
                                              "inside_our_capture"]]
    for _, r in far.iterrows():  # for loop over those three
        # f-string print: phone, time and distance; the conditional expression adds the capture when inside one
        print(f"      {r['device_model']:<12} {r['timestamp_utc']:%H:%M:%S} UTC  {r['dist_to_stave_km']:9.1f} km "
              # continuation: '+' adds the capture name only when the fix falls inside one of our captures
              "from Stave" + (f"  (inside our capture of {r['inside_our_capture']})" if r["inside_our_capture"] else ""))

    if far["dist_to_stave_km"].max() > 100:  # if: the largest excursion exceeds 100 km (the spoofed position)
        # records it as the spoofed position rather than a fault
        note(f"The largest excursion in the reported position is {far['dist_to_stave_km'].max():.0f} km, which is "
             "the spoofed position rather than a receiver fault, and it falls inside a scheduled spoofing test. It is "
             "excluded from the distance analysis and is itself a further confirmation of the timeline script's time base.")
    print("\n  Distance to the transmitter by exposure group (plausible fixes only):")  # a sub-heading in the printout
    dist_rows = []  # empty list of per-group distance summaries
    for dev in sorted(ep["device_model"].unique()):  # for loop over phones
        for g in (G50, GHI, GBL):  # nested loop over the compared groups
            grp_all = ep[(ep.device_model == dev) & (ep.exposure_group == g)]  # this phone's epochs in group g
            v = grp_all[grp_all.position_plausible]["dist_to_tx_km"].dropna()  # plausible distances only
            if v.empty:  # if: nothing to summarise
                # says whether the group itself is empty or only lacks plausible fixes
                print(f"      {dev:<12} {g:<34} " + ("group is empty - no handset epoch falls in it"
                                                     # else branch of the conditional expression: the group exists but none of its fixes is plausible
                                                     if grp_all.empty else "no plausible fixes in this group"))
                continue  # next group
            print(f"      {dev:<12} {g:<34} n={len(v):5d}  median {v.median():5.2f} km  "  # count, median and range
                  f"range {v.min():5.2f}-{v.max():5.2f} km")  # f-string continued: the smallest and largest distance in kilometres
            dist_rows.append({"device_model": dev, "median_km": float(v.median())})  # keeps the median
    dd = pd.DataFrame(dist_rows)  # the distance summaries as a table
    spread = dd.groupby("device_model")["median_km"].agg(lambda x: x.max() - x.min())  # per phone: max - min median
    for dev, v in spread.items():  # for loop over (phone, spread) pairs
        print(f"      spread of group medians, {dev}: {v:.2f} km")  # the spread in km
    if (spread > 1.0).any():  # if: on some phone the groups sit at materially different ranges from the jammer
        # fixed text: the group ranges differ by more than 1 km on some phone
        print("      The groups DIFFER materially in distance, so the comparison is also run on stationary epochs.")
        # records why the stationary restriction is used
        note("Group median distance to the transmitter differs by more than one kilometre within a device, so a "
             "level difference between groups could partly reflect a change in range rather than the jammer being "
             "on. The central comparison is therefore repeated on stationary intervals only.")
    else:  # else: all within 1 km
        print("      The groups do NOT differ materially in distance (all within 1 km).")  # says so

    # ---- per-session distance to SENDER --------------
    moving = obd_moving_intervals(sessions)  # when each session's vehicle moved (OBD-II, verification only)
    dist = session_distances(ep, sessions, moving)  # the distance table
    dist.to_csv(PROCESSED / "ground_truth" / "session_distance_to_sender.csv", index=False, lineterminator="\n")  # written to CSV
    print(f"\n  Distance from each session's recording position to {TX_NAME}, from the handset fixes:")  # the section heading of the printout
    for _, r in dist.iterrows():  # for loop over the rows
        # f-string print of one distance row; the conditional expression prints the distance or only the method
        print(f"      {r['test_code']:<7} {r['part']:<14} fixes {r['n_fixes']:5d} (margins {r['n_margin_fixes']:5d})  "
              # '+' joins either the distance, the method and the Stave-landmark distance ...
              + (f"{r['dist_to_sender_km']:6.2f} km  [{r['method']}]  Stave landmark {r['dist_to_stave_landmark_km']:.2f} km"
                 # ... or, when the part has no fix (the 2.6.5 moving part), only the method
                 if pd.notna(r["dist_to_sender_km"]) else f"[{r['method']}]"))
    for code, mv in moving.items():  # for loop over the moving intervals
        if mv is not None:  # if: that vehicle moved
            print(f"      {code}: vehicle moving {mv[0]:%H:%M:%S}-{mv[1]:%H:%M:%S} UTC (OBD-II SPEED > 0)")  # the interval
    d265 = dist[(dist.test_code == "2.6.5") & (dist.part == "parked")]  # 2.6.5's parked part
    d262 = dist[dist.test_code == "2.6.2"]  # the 2.6.2 site
    if len(d265) and len(d262) and pd.notna(d265.iloc[0].median_lat):  # if: both positions exist
        # distance between the two median positions, to test "parked part = the 2.6.2 site"
        sep_m = haversine_m(d265.iloc[0].median_lat, d265.iloc[0].median_lon, d262.iloc[0].median_lat,
                            # continuation of the haversine_m call: the 2.6.2 median longitude, the last argument
                            d262.iloc[0].median_lon)
        print(f"      2.6.5 parked position to the 2.6.2 position: {sep_m:.0f} m")  # the separation
        # note(): records, for the report, how long 2.6.5 moved and how close its parked position is to 2.6.2's
        note(f"In 2.6.5 the vehicle moved until {moving['2.6.5'][1]:%H:%M:%S} UTC and then stood {sep_m:.0f} m from "
             "the 2.6.2 position; the handsets only began logging after it had parked.")  # records it

    # stationary intervals: path length over the preceding 60 s within one capture, below 150 m
    ep["stationary"] = stationary_flags(ep)  # the new column
    # left join of the stationary flag onto the band rows
    bands = bands.merge(ep[["device_folder", "timestamp_utc", "dist_to_tx_km", "dist_to_stave_km",
                            "position_plausible", "implied_speed_kmh", "stationary"]],
                        on=["device_folder", "timestamp_utc"], how="left")  # keyword arguments of merge(): matched on device and epoch time; band rows without a match are kept
    # f-string print: stationary epochs, their count and share (the mean of a boolean column)
    print(f"\n  Stationary epochs (under {STATIONARY_M:.0f} m of path in the preceding 60 s, within one capture): "
          f"{int(ep['stationary'].sum())} of {len(ep)} ({100 * ep['stationary'].mean():.1f} %)")  # count and share

    # ---- band structure and quantisation ------------------------------------
    print("\n" + "=" * 78)  # a line of repeated characters that separates sections of the printout
    print("[5] BAND STRUCTURE AND SCALE, PER DEVICE (all readings)")  # the section heading of the printout
    print("=" * 78)  # a line of repeated characters that separates sections of the printout
    bs_rows = []  # empty list: one summary row per phone and band
    for dev in sorted(bands["device_model"].unique()):  # for loop over phones
        sub = bands[bands.device_model == dev]  # this phone's band rows
        for hz in sorted(sub["band_hz"].unique()):  # for loop over this phone's bands
            sb = sub[sub.band_hz == hz]  # this band
            v = sb["level_db"].to_numpy()  # all readings in this band, as an array
            j = int(sb["jammed"].sum())  # how many carried the jammed flag
            # method chain: round to 6 places, count how often each value occurs, keep the 10 most frequent
            top = sb["level_db"].round(6).value_counts().head(10)
            bs_rows.append({  # appends the summary row written to ground_truth/handset_band_structure.csv
                "device_model": dev, "band": BAND_SHORT[hz], "band_hz": hz,  # dict entry: phone model, short band name and frequency
                "band_label": BAND_LABEL[hz], "n_readings": len(v),  # dict entry: the band's long label and the number of readings
                "n_not_live": int(sb["frozen"].sum()),  # readings flagged not live
                "level_min": round(float(v.min()), 3),  # dict entry: lowest gain level of the band
                "level_p5": round(float(np.percentile(v, 5)), 3),  # dict entry: 5th percentile of the gain level
                "level_median": round(float(np.median(v)), 3),  # dict entry: median gain level
                "level_p95": round(float(np.percentile(v, 95)), 3),  # dict entry: 95th percentile of the gain level
                "level_max": round(float(v.max()), 3),  # dict entry: highest gain level of the band
                "level_iqr": round(iqr(v), 3),  # dict entry: interquartile range of the gain level
                "n_distinct_values": int(len(np.unique(v))),  # np.unique: the distinct values; len counts them
                "distinct_fraction": round(len(np.unique(v)) / len(v), 4),  # distinct values per reading
                "top10_values": ";".join(f"{k}:{c}" for k, c in top.items()),  # "value:count" pairs in one cell
                "jammed_count": j,  # dict entry: readings the phone flagged as jammed
                "jammed_fraction": round(j / len(v), 4),  # dict entry: share of readings flagged as jammed
                # joins every row's constellation list, splits it again, de-duplicates with set, sorts and rejoins
                "constellations": ";".join(sorted(set(";".join(sb["constellations_sharing_band"]).split(";")))),
                # mean number of raw entries merged into each row for this band
                "mean_agc_entries_collapsed": round(float(sb["n_agc_entries_collapsed"].mean()), 3),
            })
    bsdf = pd.DataFrame(bs_rows)  # the band-structure table
    for _, r in bsdf.iterrows():  # for loop over the summaries, printed as a table
        # f-string print of one band summary: count, median, IQR, distinct values and not-live count
        print(f"      {r['device_model']:<12} {r['band']:<4} n {r['n_readings']:6d}  median {r['level_median']:8.2f}  "
              # continuation of the f-string: IQR, distinct values and the not-live count of this band
              f"IQR {r['level_iqr']:6.2f}  distinct {r['n_distinct_values']:5d}  not live {r['n_not_live']:5d}")
    bsdf.to_csv(PROCESSED / "ground_truth" / "handset_band_structure.csv", index=False, lineterminator="\n")  # written to CSV
    meds = bsdf.groupby("device_model")["level_median"].median()  # per phone, the median of its band medians
    print(f"\n  Scale difference between devices: median level across bands "  # the gap between the two scales
          # continuation: each phone's median level ('against' between them) and the gap between the two
          f"{' against '.join(f'{v:.2f} ({k})' for k, v in meds.items())}, a gap of {meds.max() - meds.min():.2f} dB.")
    print("  The two devices are on incompatible scales and are never pooled.")  # fixed text
    nmin = int(bsdf["n_distinct_values"].min())  # the smallest number of distinct values in any band
    if nmin >= 50:  # if: every band takes at least 50 distinct values
        # f-string print: the smallest number of distinct values in any band (computed, not 'hundreds')
        print(f"  QUANTISATION: every band takes at least {nmin} distinct values, so a shift in medians cannot be "
              "an artefact of a quantisation step.")  # the computed minimum, not "hundreds"
        # note(): records that no band is coarsely quantised, with the computed minimum
        note(f"No band on either device is coarsely quantised; every band takes at least {nmin} distinct "
             "values, so differences in median cannot be attributed to a quantisation boundary.")  # records it
    else:  # else: some band is coarse
        for _, r in bsdf[bsdf.n_distinct_values < 50].iterrows():  # for loop over the coarse bands
            # flag(): records each coarsely quantised band as an anomaly instead of ignoring it
            flag(f"{r['device_model']} {r['band']} takes only {r['n_distinct_values']} distinct values; a median "
                 "shift there may reflect quantisation rather than gain")  # records each as an anomaly

    # ---- the tests: on live readings, then (silently) on all readings ---------
    live = bands[~bands.frozen]  # live readings only
    T = run_tests(live, ep, plan, meta, pyr, sessions)  # the five tests, printed
    n_notes, n_anom = len(NOTES_OUT), len(ANOMALIES)  # how many notes exist before the silent run again
    with contextlib.redirect_stdout(io.StringIO()):  # "with": the printout of the run again goes to a discarded buffer
        T_all = run_tests(bands, ep, plan, meta, pyr, sessions)  # the same tests on all readings (sensitivity)
    del NOTES_OUT[n_notes:], ANOMALIES[n_anom:]  # slice deletion: drops notes the silent run again repeated
    cdf, wdf, tdf, ctrl = T["comp"], T["within"], T["timing"], T["control"]  # tuple unpacking: the live tables
    pv = T["pyramid"]  # the live pyramid table
    hits = ctrl[ctrl.deflection_detected] if not ctrl.empty else ctrl  # the control contrasts meeting the pass rule
    # f-string print: how many control contrasts pass the -3 dB and delta < -0.30 rule, of all of them
    print(f"\n  Positive control: deflections of more than 3 dB in the expected direction with an effect size beyond "
          f"0.30: {len(hits)} of {len(ctrl)} device-test-band contrasts.")  # the summary line
    for _, h in hits.iterrows():  # for loop over the passing contrasts
        print(f"      {h['device_model']:<12} test {h['our_test_code']:<7} {h['band']:<4} "  # each one
              # continuation of the f-string: the contrast's level difference and Cliff's delta, both with their sign
              f"{h['inside_minus_margins_db']:+.1f} dB (Cliff d {h['cliffs_delta']:+.2f})")
    if not hits.empty:  # if: the control passed
        worst = hits.loc[hits.inside_minus_margins_db.idxmin()]  # idxmin: the row with the deepest drop
        print(f"  The control PASSES. The clearest case is test {worst['our_test_code']} on the "  # the clearest case
              # continuation of the f-string: the phone and band of the clearest case
              f"{worst['device_model']}, where the {worst['band']} gain control falls by "
              # continuation: the size of its drop in whole dB (abs() removes the minus sign for the prose)
              f"{abs(worst['inside_minus_margins_db']):.0f} dB inside the scheduled window.")
        # the sessions whose catalogue name says the motorcade was stationary (computed)
        still = list(sessions.loc[sessions.test_name.str.contains("stationary", case=False), "test_code"])
        # f-string print: the sessions whose catalogue name says 'stationary', found with str.contains
        print(f"  Sessions whose catalogue name says the motorcade was stationary: {', '.join(still) or 'none'} "
              f"(of {len(sessions)}); a stationary spoofer stays close for the whole window.")  # computed statement
        if (hits.band == "L5").any():  # if: any passing contrast is on L5, a band not spoofed
            # records the L5 corroboration of the group-level continuous-jamming statement
            note("The positive control deflects on L5 as well as on the spoofed bands. L5 is not spoofed in group "
                 "2.6, but the catalogue's group-level description states continuous jamming on L2, L5, E5b and E6 "
                 "throughout the group, so a deflection there is what the documentation predicts.")
        # note(): records the control result with the deepest drop, computed from the passing contrasts
        note(f"Positive control: the handsets register our own roughly 1 mW spoofing with a deflection of up to "
             # continuation of the note: the deepest drop in whole dB, then why it matters
             f"{abs(hits.inside_minus_margins_db.min()):.0f} dB, confined to the scheduled test window, which gives "
             "weight to their failure to register the 50 W transmissions.")  # records the control result
    else:  # else: nothing passed
        print("  The control FAILS: the handsets do not register our own spoofing either.")  # says so

    # instrument resolution: second-to-second variation against the group differences (computed)
    print("\n  Instrument resolution (live readings): median |change| per second against the largest group difference")
    res = []  # list of (device, band, median change, largest difference)
    for dev in sorted(live["device_model"].unique()):  # for loop over phones
        for hz in sorted(live[live.device_model == dev]["band_hz"].unique()):  # and their bands
            # boolean mask and sort: this phone's and band's live readings, by capture then time
            g = live[(live.device_model == dev) & (live.band_hz == hz)].sort_values(["logger_session_id", "timestamp_utc"])
            ch = float(g.groupby("logger_session_id")["level_db"].diff().abs().median())  # typical one-second jump
            # largest absolute group-minus-baseline difference for this phone and band, from the comparison table
            dmax = float(cdf[(cdf.device_model == dev) & (cdf.band_hz == hz)]["median_minus_baseline_db"].abs().max())
            res.append((dev, BAND_SHORT[hz], ch, dmax))  # keeps them
            # f-string print: the typical one-second change beside the largest group difference
            print(f"      {dev:<12} {BAND_SHORT[hz]:<5} {ch:>8.3f} dB/s   largest group difference {dmax:6.2f} dB")
    for dev in sorted({r[0] for r in res}):  # for loop over phones, for one computed note each
        rr = [r for r in res if r[0] == dev]  # that phone's rows
        # records the resolution limit with the computed values (no typed number)
        note(f"On the {dev} the gain-control reading moves by a median {min(r[2] for r in rr):.2f} to "
             # continuation of the note: the largest one-second change across the phone's bands
             f"{max(r[2] for r in rr):.2f} dB from one second to the next across its bands, against a largest "
             # continuation of the note: the largest group difference for that phone, for comparison
             f"exposure-group difference of {max(r[3] for r in rr):.2f} dB.")
    strict = ep[ep.no_scheduled_test_anywhere]  # epochs when nothing of any power was scheduled anywhere
    # f-string print: epochs with nothing scheduled anywhere, of all epochs, and the share
    print(f"\n  Stricter baseline (no scheduled test of ANY power anywhere on the range): {len(strict)} epochs of "
          f"{len(ep)} ({100 * len(strict) / len(ep):.1f} %).")  # how many there are, and their share
    if len(strict) < 60:  # if: under a minute's worth, too few for a baseline
        # note(): records why that stricter baseline was not used (too few epochs)
        note(f"Only {len(strict)} of {len(ep)} handset epochs fall at a moment when no test of any power was "
             "scheduled anywhere on the range, too few for a separate baseline; the baseline used is 'nothing "
             "above 1 W', which still contains the roughly 1 mW spoofing under study.")  # records why
    if ep[ep.exposure_group == GHI].empty:  # if: no epoch fell in the middle group
        mid = sorted(p["test_code"] for p in plan  # the tests that would form it (above 1 W, not 50 W at TA1)
                     # filter clause of the comprehension: keeps tests of known power above 1 W ...
                     if p["power_w"] is not None and p["power_w"] > HIGH_POWER_W
                     # ... that are NOT a 50 W Test Area 1 test (those form the 50 W group instead)
                     and not (p["power_w"] >= FIFTY_W_MIN and p["test_area"] == 1))
        # records it, naming the tests from the schedule rather than typing them
        note(f"The '{GHI}' group contains no handset epochs: the tests that would form it ({', '.join(mid)}) all "
             "transmit inside a 50 W Test Area 1 window, so those seconds already belong to the 50 W group.")

    # ---- control descriptives --------------------------------------
    ph = control_phases(live, sessions, phases)  # per-phase deflections, descriptive only
    ph.to_csv(PROCESSED / "ground_truth" / "handset_control_phases.csv", index=False, lineterminator="\n")  # written to CSV
    ons = []  # rows of the onset/recovery sensitivity grid
    for thr in (2.0, 3.0, 5.0):  # thresholds in dB
        for hold in (15.0, 30.0, 60.0):  # hold times in seconds
            ons += onset_recovery(live, sessions, thr, hold)  # "+=" extends the list with this setting's rows
    on = pd.DataFrame(ons)  # the grid as a table
    on.to_csv(PROCESSED / "ground_truth" / "handset_control_onset.csv", index=False, lineterminator="\n")  # written to CSV
    # fixed text: where the descriptive control tables went, and that they never feed a verdict
    print("\n  Control phases and onset/recovery written (descriptive only; the onset rule was chosen after the")
    print("  data were examined and never feeds a verdict).")  # the next line of the same printed text

    # ---- satellites and CN0 ---------------------------------------------------
    print("\n" + "=" * 78)  # a line of repeated characters that separates sections of the printout
    print("[10] SATELLITES TRACKED AND CARRIER-TO-NOISE DENSITY")  # the section heading of the printout
    print("=" * 78)  # a line of repeated characters that separates sections of the printout
    for dev in sorted(ep["device_model"].unique()):  # for loop over phones
        sub = ep[ep.device_model == dev]  # this phone's epochs
        bs = sub[sub.exposure_group == GBL]  # its baseline epochs
        for g in (G50, GHI, GBL):  # for loop over the compared groups
            v = sub[sub.exposure_group == g]  # epochs in this group
            if v.empty:  # if: none
                continue  # skip
            ds = v["n_satellites_epoch"].median() - bs["n_satellites_epoch"].median()  # median satellites vs base
            dc = v["median_cn0_dbhz"].median() - bs["median_cn0_dbhz"].median()  # median C/N0 vs base
            print(f"      {dev:<12} {g:<34} {len(v):>6} sats {ds:>+6.1f}  C/N0 {dc:>+7.2f} vs baseline")  # the row
    for dev, sid in T["within_ok"]:  # within each capture holding both groups, which holds position fixed
        s = ep[(ep.device_model == dev) & (ep.logger_session_id == sid)]  # that capture's epochs
        a, b = s[s.exposure_group == G50], s[s.exposure_group == GBL]  # tuple assignment: 50 W and baseline epochs
        # f-string print: within one capture, the 50 W minus baseline difference in satellites and in C/N0
        print(f"      within {dev:<12} {sid:>3}: sats {a['n_satellites_epoch'].median() - b['n_satellites_epoch'].median():+.1f}"
              f"  C/N0 {a['median_cn0_dbhz'].median() - b['median_cn0_dbhz'].median():+.2f}")  # differences

    # ---- jammed flag (all readings: a device output, described as reported) --
    print("\n" + "=" * 78)  # a line of repeated characters that separates sections of the printout
    print("[11] THE 'jammed' FLAG - secondary evidence only")  # the section heading of the printout
    print("=" * 78)  # a line of repeated characters that separates sections of the printout
    tot = bands.groupby("device_model")["jammed"].agg(["sum", "count"])  # per phone: flags set, and rows in total
    print("  Device totals: " + "; ".join(  # one line joining a generator of strings
        # the generator inside str.join: 'phone flags of rows (per cent)' for each phone row of the totals table
        f"{k} {int(r['sum'])} of {int(r['count'])} ({100 * r['sum'] / r['count']:.2f} %)" for k, r in tot.iterrows()))
    rate = (tot["sum"] / tot["count"]).replace(0, np.nan)  # each phone's flag rate (0 -> NaN to avoid dividing by 0)
    if rate.notna().sum() == 2:  # if: both phones set the flag at least once
        # the computed ratio of the two rates (replaces the typed "orders of magnitude")
        print(f"  The two devices' flag rates differ by a factor of {rate.max() / rate.min():.0f}; the flag is a "
              "device-specific threshold, so it can indicate WHEN, not HOW MUCH.")
    for dev in sorted(bands["device_model"].unique()):  # for loop over phones
        for hz in sorted(bands[bands.device_model == dev]["band_hz"].unique()):  # for loop over bands
            for g in (G50, GHI, GBL):  # nested loop over groups
                # boolean mask with three conditions: this phone, band and exposure group (all readings)
                s = bands[(bands.device_model == dev) & (bands.band_hz == hz) & (bands.exposure_group == g)]
                if s.empty:  # if: none
                    continue  # skip
                j = int(s["jammed"].sum())  # rows with the jammed flag set
                # f-string print: readings, flagged readings and their percentage
                print(f"      {dev:<12} {BAND_SHORT[hz]:<5} {g:<34} {len(s):>7} {j:>8} {100 * j / len(s):>6.1f}%")

    # ---- the power-ratio finding, per session ---------------
    print("\n" + "=" * 78)  # a line of repeated characters that separates sections of the printout
    print("[12] RELATIVE POWER AT THE ANTENNA - free space, per session distance")  # the section heading of the printout
    print("=" * 78)  # a line of repeated characters that separates sections of the printout
    fs_rows = []  # rows of ground_truth/free_space_ratio.csv
    for _, r in dist[dist.dist_to_sender_km.notna()].iterrows():  # for loop over parts with a measured distance
        for d_spoof_m in SPOOF_RANGES_M:  # for loop over the spoofer ranges in metres
            # free space: received power ~ transmitted / range^2; ratio = (50 W / d_jam^2) / (1 mW / d_spoof^2)
            pr = (50.0 / (r["dist_to_sender_km"] * 1000) ** 2) / (1e-3 / d_spoof_m ** 2)
            fs_rows.append({"test_code": r["test_code"], "part": r["part"],  # appends the row
                            # dict entries (one row of ground_truth/free_space_ratio.csv): the session's distance and the assumed spoofer range
                            "dist_to_sender_km": r["dist_to_sender_km"], "spoofer_range_m": d_spoof_m,
                            # dict entry: the free-space power ratio, jammer over spoofer, 4 decimals
                            "ratio_jammer_over_spoofer": round(pr, 4),
                            "ratio_db": round(10 * math.log10(pr), 2)})  # + means the jammer arrives stronger
    fs = pd.DataFrame(fs_rows)  # the table
    fs.to_csv(PROCESSED / "ground_truth" / "free_space_ratio.csv", index=False, lineterminator="\n")  # written to CSV
    # pivot_table: reshapes the long table to one row per session part and one column per spoofer range
    piv = fs.pivot_table(index=["test_code", "part", "dist_to_sender_km"], columns="spoofer_range_m",
                         values="ratio_db", sort=False)  # one row per session part, one column per range
    print("      jammer minus spoofer at the antenna, dB, by spoofer range (m):")  # a sub-heading in the printout
    print("      " + piv.to_string().replace("\n", "\n      "))  # the table, indented
    dmin, dmax_ = dist.dist_to_sender_km.min(), dist.dist_to_sender_km.max()  # the measured distance range
    # note(): records the free-space finding with the computed distance range and dB range
    note(f"Under free-space propagation the 50 W jammer at {dmin:.1f}-{dmax_:.1f} km (per session) and the 1 mW "
         # continuation of the note: the smallest jammer-minus-spoofer value in dB
         f"spoofer at tens of metres deliver comparable power at the antenna ({fs.ratio_db.min():+.1f} to "
         # continuation of the note: the largest value and the ranges it covers, then the terrain argument
         f"{fs.ratio_db.max():+.1f} dB over {SPOOF_RANGES_M[0]}-{SPOOF_RANGES_M[-1]} m), and terrain attenuates "
         "the distant transmitter far more than the near one.")  # records the finding with computed values

    # ---- figures --------------------------------------------------------------
    written = []  # the figures actually written
    figure_distance(ep, plan, GRAPH / "distance_to_transmitter.png")  # Figure 3
    written.append("distance_to_transmitter.png")  # recorded
    figure_levels(bands, plan, GRAPH / "agc_level_timeseries.png")  # Figure 4 (live and not-live readings)
    written.append("agc_level_timeseries.png")  # recorded
    figure_distributions(live[live.exposure_group != GUNK], GRAPH / "level_distributions.png")  # Figure 5
    written.append("level_distributions.png")  # recorded
    figure_sats_cn0(ep, plan, GRAPH / "satellites_and_cn0.png")  # Figure 6
    written.append("satellites_and_cn0.png")  # recorded
    figure_pyramid(pyr, live, GRAPH / "pyramid_predicted_vs_observed.png")  # Figure 7 (live readings)
    written.append("pyramid_predicted_vs_observed.png")  # recorded
    verdict, vc = verdict_of(T)  # the verdict on live readings, and its counts
    verdict_all, vc_all = verdict_of(T_all)  # the same on all readings (sensitivity)
    if vc["control_passes"]:  # if: the control passed, Figure 8 is drawn
        # the test with the deepest single drop: min per test, then idxmin picks that test's code
        best_test = ctrl[ctrl.deflection_detected].groupby("our_test_code")["inside_minus_margins_db"].min().idxmin()
        dbt = dist[dist.test_code == best_test].dist_to_sender_km.dropna()  # that session's measured distance
        # figure_control(): Figure 8 for the clearest test, with that session's last measured distance (the parked part if split)
        figure_control(bands, sessions, plan, best_test, float(dbt.iloc[-1]),
                       GRAPH / "positive_control.png")  # Figure 8 (last part = parked, if split)
        written.append("positive_control.png")  # recorded
    print(f"\n[13] Figures written to Graph/ at 300 dpi: {', '.join(written)}.")  # the list actually written

    # ---- write tables ---------------------------------------------------------
    cdf.to_csv(PROCESSED / "ground_truth" / "handset_level_comparison.csv", index=False, lineterminator="\n")  # live comparison
    wdf.to_csv(PROCESSED / "ground_truth" / "handset_within_capture_comparison.csv", index=False, lineterminator="\n")  # always
    tdf.to_csv(PROCESSED / "ground_truth" / "handset_transition_timing.csv", index=False, lineterminator="\n")  # live timing
    pv.to_csv(PROCESSED / "ground_truth" / "pyramid_observed.csv", index=False, lineterminator="\n")  # live pyramid
    ctrl.to_csv(PROCESSED / "ground_truth" / "handset_positive_control.csv", index=False, lineterminator="\n")  # live control
    sens = []  # the sensitivity table: every test, on live and on all readings
    for setname, TT in (("live readings", T), ("all readings", T_all)):  # for loop over the two reading sets
        # nested loop over the five tests: (name, key into the result dict)
        for test, key in (("group comparison", "comp"), ("within capture", "within"), ("transition timing", "timing"),
                          ("pyramid", "pyramid"), ("positive control", "control")):  # and the five tests
            if not TT[key].empty:  # if: that test produced rows
                sens.append(TT[key].assign(reading_set=setname, test=test))  # tagged copy of its table
    pd.concat(sens, ignore_index=True).to_csv(PROCESSED / "ground_truth" / "handset_sensitivity_all_readings.csv",  # one long table
                                              # continuation of to_csv: no index column, "\n" line endings for byte-identical runs again
                                              index=False, lineterminator="\n")
    # list of column names: fixes the columns and their order in ground_truth/handset_agc_timeseries.csv
    bcols = ["device_folder", "device_model", "logger_session_id", "timestamp_utc",
             "band", "band_hz", "level_db", "level_db_spread_db", "jammed",
             "n_agc_entries_collapsed", "constellations_sharing_band", "n_satellites_epoch", "n_satellites_band",
             "median_cn0_dbhz_band", "value_run_s", "frozen", "frozen_reason",
             "exposure_group", "fifty_w_test", "active_tests", "unknown_power_tests",
             "max_concurrent_power_w", "max_power_test_code", "max_power_test_area",
             "max_power_source", "no_scheduled_test_anywhere", "inside_our_capture",
             "dist_to_tx_km", "dist_to_stave_km", "position_plausible", "stationary"]
    b_out = bands[bcols].copy()  # those columns only, as an independent copy
    # .dt.strftime formats each timestamp as text with microseconds, so the CSV text is fixed
    b_out["timestamp_utc"] = b_out["timestamp_utc"].dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    for c in ("level_db", "dist_to_tx_km", "dist_to_stave_km", "median_cn0_dbhz_band", "value_run_s"):  # floats
        b_out[c] = b_out[c].round(4)  # fixed rounding to 4 places, so runs again write identical text
    b_out.sort_values(["device_folder", "timestamp_utc", "band_hz"]).to_csv(  # sorted, then written
        PROCESSED / "ground_truth" / "handset_agc_timeseries.csv", index=False, lineterminator="\n")  # argument of to_csv(): the output path, without the index and with Unix line ends
    # column list and order for ground_truth/handset_epoch_summary.csv
    ecols = ["device_folder", "device_model", "logger_session_id", "source_file",
             "timestamp_utc", "n_satellites_epoch", "median_cn0_dbhz",
             "lat_top_level", "lon_top_level", "lat_gnsslocations", "lon_gnsslocations",
             "position_fields_separation_m", "position_accuracy_m", "location_provider",
             "dist_to_tx_km", "dist_to_stave_km", "step_m", "implied_speed_kmh",
             "position_plausible", "stationary", "exposure_group", "active_tests", "unknown_power_tests",
             "max_concurrent_power_w", "no_scheduled_test_anywhere", "inside_our_capture"]
    e_out = ep[ecols].copy()  # those columns only, as a copy
    e_out["timestamp_utc"] = e_out["timestamp_utc"].dt.strftime("%Y-%m-%d %H:%M:%S.%f")  # timestamps as fixed text
    for c in ("median_cn0_dbhz", "dist_to_tx_km", "dist_to_stave_km", "step_m",  # for loop over the float columns
              "implied_speed_kmh", "position_fields_separation_m"):  # tuple element: the last two float columns rounded before writing
        e_out[c] = e_out[c].round(4)  # fixed rounding to 4 places
    e_out.sort_values(["device_folder", "timestamp_utc"]).to_csv(  # sorted, then written
        PROCESSED / "ground_truth" / "handset_epoch_summary.csv", index=False, lineterminator="\n")  # argument of to_csv(): the output path, without the index and with Unix line ends

    # ---- verdict ---------------------------------------------------------------
    print("\n" + "=" * 78)  # a line of repeated characters that separates sections of the printout
    print("PLAIN-LANGUAGE REPORT AND VERDICT")  # the section heading of the printout
    print("=" * 78)  # a line of repeated characters that separates sections of the printout
    # f-string print: the question, with the computed range of per-session distances
    print(f"\n  The question: did the 50 W jamming transmitted from Test Area 1, {dmin:.1f}-{dmax_:.1f} km away "
          "depending on the session, reach the recording position?")  # the question with computed distances
    # f-string print: pyramid contrasts in the predicted direction, of all
    print(f"\n  Pyramid within-step contrast: {vc['n_match']} of {vc['n_tests']} device-band tests moved in the "
          "predicted direction (level lower while jamming than during the break).")  # the pyramid count
    if vc["n_tests"]:  # if: any pyramid contrast exists (0 is falsy)
        # both directions reported, and which way the large effects point
        print(f"  Effect size above 0.30: {vc['n_big_either']} contrasts in either direction, "
              # continuation of the f-string: how many large effects point the predicted way
              f"{vc['n_big_predicted']} of them in the predicted direction.")
    if vc["n_timing"]:  # if: timing tests exist
        # f-string print: timing tests within 30 s, of all
        print(f"\n  Transition timing: {vc['near_n']} of {vc['n_timing']} band-edge tests placed the largest change "
              "within 30 s of the scheduled instant.")  # the timing count
    if vc["control_passes"]:  # if: the control passed
        best = abs(ctrl.inside_minus_margins_db.min())  # the deepest drop, as a positive number of dB
        # f-string print: the positive control's deepest drop, computed
        print(f"\n  Positive control: the same handsets register our own roughly 1 mW spoofing with a deflection "
              f"of up to {best:.0f} dB, confined to the scheduled window.")  # summary
    print("\n  VERDICT (live readings): " + verdict)  # the verdict label
    print("  VERDICT (all readings, sensitivity): " + verdict_all)  # and the sensitivity verdict
    if verdict == "EVIDENCE THAT THE 50 W JAMMING DID NOT REACH THE RECORDING POSITION":  # the branch this data reaches
        # f-string print: the verdict's reasoning, with the deepest drop, the power ratio and distances computed
        print(f"  Three independent tests come out negative, and the same instrument registers a deflection of up to "
              # continuation: the deepest control drop in whole dB
              f"{abs(ctrl.inside_minus_margins_db.min()):.0f} dB from a transmitter radiating about one milliwatt, "
              # continuation: the power ratio 50 W / 1 mW printed with a thousands separator (:,.0f)
              f"only inside that transmitter's scheduled window. A source radiating {50 / 1e-3:,.0f} times more power "
              f"from {dmin:.1f}-{dmax_:.1f} km produced nothing distinguishable from noise.")  # computed values
        print("  CONFIDENCE: moderate. A consumer handset inside a vehicle is an indication, not a verdict.")  # caveat
    # fixed text: where the question is settled (the receiver-status script)
    print("\n  WHAT WOULD SETTLE IT: the Septentrio and u-blox receivers' own interference monitoring, examined in "
          "the receiver-status script.")  # next step

    if NOTES_OUT:  # if: any notes were recorded
        print("\n" + "-" * 78)  # a line of repeated characters that separates sections of the printout
        print(f"NOTES ({len(NOTES_OUT)})")  # heading with the count
        print("-" * 78)  # a line of repeated characters that separates sections of the printout
        for n in NOTES_OUT:  # for loop over the notes, in the order raised
            print(f"  - {n}")  # printed as bullets
    print("\n" + "-" * 78)  # a line of repeated characters that separates sections of the printout
    print(f"ANOMALIES AND THINGS THAT LOOK WRONG ({len(ANOMALIES)})")  # heading with the count
    print("-" * 78)  # a line of repeated characters that separates sections of the printout
    if not ANOMALIES:  # if: none
        print("  none")  # says so
    for a in ANOMALIES:  # for loop over the anomalies
        print(f"  - {a}")  # printed as bullets
    print("\n" + "=" * 78)  # a line of repeated characters that separates sections of the printout
    print("Outputs written:")  # the section heading of the printout
    # "+" concatenates the sorted CSV names of this script and the figures actually written
    for f in sorted(PROCESSED.glob("handset_*.csv")) + sorted(PROCESSED.glob("pyramid_*.csv")) + [
            # continuation of the list: the two new distance tables, printed after the others
            PROCESSED / "ground_truth" / "session_distance_to_sender.csv", PROCESSED / "ground_truth" / "free_space_ratio.csv"]:
        print(f"  {f.relative_to(PROJECT)}")  # path relative to the project folder
    for f in written:  # the figures
        print(f"  Graph/{f}")  # each figure's path
    print(f"\nRandom seed fixed at {SEED}.")  # seed note
    print("=" * 78)  # closing rule


if __name__ == "__main__":  # if: true only when the file is run directly, not when imported
    main()  # runs the section
