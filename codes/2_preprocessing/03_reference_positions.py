#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
Reference positions for the three recording sites, from the clean stationary intervals.

This file holds two parts. The first builds the clean intervals. The second estimates the positions: single-point
least squares per epoch at 1 Hz from GPS C1C and Galileo E1 (C1C on the Septentrio, C1X on the u-blox) pseudoranges, orbits and clocks
from the IGS merged broadcast file only (I/NAV for Galileo; the conversion script's functions), one clock per constellation, a 10 degree mask,
Klobuchar from the IGS header's GPS parameters, Saastamoinen with a standard atmosphere, sin^2(elevation) weights, and iterative
rejection of residuals over 30 m (counted). BeiDou B1I (C2I) is not in the position: with the GPS+Galileo site position fixed, its
residuals over the same clean epochs are a test of the BeiDou orbit code written here (GEO satellites apart from the rest).

Reads   Data/processed/ground_truth/margin_clean_minutes.csv     the receiver-status script's clean margins, "ALL combined" rows (minutes, not times)
        Data/processed/ground_truth/session_ground_truth.csv      capture and scheduled windows of the six sessions
        Data/processed/ground_truth/receiver_status_timeseries.csv  gains and u-blox noise, used only to VERIFY that the intervals are quiet
Writes  Data/processed/preprocessing/reference_clean_intervals.csv  site, test_code, receiver, start_utc, end_utc, minutes, rule_applied
        Data/processed/checks/reference_interval_reconciliation.csv  each margin against the receiver-status script's surviving minutes
        Data/processed/preprocessing/reference_position_epochs.csv         every 1 Hz solution (ECEF, geodetic, satellites used, rejected)
        Data/processed/preprocessing/reference_positions.csv               median position per session x receiver and per site, with scatter
        Data/processed/preprocessing/reference_position_checks.csv         checks 1-4 and the BeiDou test in long form (check, group, quantity, value)
        Data/processed/checks/reference_bds_residuals.csv              BeiDou C2I residuals per satellite (median and 95th percentile)
        Graph/reference_positions.png               the figure: horizontal scatter of the 1 Hz solutions per site
        Data/processed/checks/reference_heights.png                    check plot: ellipsoidal height per session and receiver

Interval rule: start from each session's capture margin, the minutes
before and after the scheduled window. Then, in this order:
  2.6.5   excluded entirely: the drives from 12:27:22 to 12:34:57 are excluded, and the pre-test margin lies inside them. The post-test margin
          is attack-affected to the end of capture (ATTACK_EXTENDED_END in the conversion script); the X5 gain
          onset 12:50:59 was the cut for the X5 as a motion witness and not a cleanliness boundary, because gain lags deception.
  2.6.10  spoofed tracking lasted to 14:43:59, so the post-test margin starts at 14:44:00.
  2.6.12  the margin before 15:45:12 is possibly contaminated and is excluded; the mosaic-X5 tracked nothing from 15:50:10 to
          16:07:04, so its post-test margin starts at 16:07:04; u-blox 01 after its clock release carries an offset common
          to all satellites, so its post-test margin is excluded.
  2.6.11  has no margin (the scheduled window covers the whole capture): nothing.
  every edge that borders an excluded or an attack interval then gets a 30 s guard (the reference-position rule only, label guards
  stay with the label script). Edges at the start or end of the capture get none.
A site or receiver with under two minutes is reported, never padded.

Verification and ground-truth evidence only: nothing here is a feature (exclusion rule).
"""

from __future__ import annotations  # future import: type hints kept as text (a convention across the project's scripts)

import os  # operating-system access: MPLCONFIGDIR is set before matplotlib loads
from datetime import datetime, timezone  # date-time classes, for the epoch times of the RINEX files
import random  # standard-library random; seeded below although nothing here draws from it
import sys  # standard-library system access: sys.path and sys.exit
from pathlib import Path  # Path objects for building file paths

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
os.environ["MPLCONFIGDIR"] = str(PROJECT / "codes" / ".mplconfig")  # matplotlib's cache inside the project folder

import numpy as np  # NumPy: arrays of times and readings
import pandas as pd  # pandas: reads the CSVs and handles timestamps

sys.path.insert(0, str(PROJECT / "codes" / "common"))  # puts codes/ first on the import path so the helpers module is found
from helpers import exit_if_failed, GPS_EPOCH, LEAP_SECONDS, decision_times  # decision_times reads Data/processed/ground_truth/decision_times.csv; the GPS epoch and the 18 s between GPS time and UTC; exit_if_failed ends the run on a failed check

SEED = 20260914  # integer constant: the project's fixed random seed
random.seed(SEED)  # seeds Python's generator (nothing here draws from it)
np.random.seed(SEED)  # seeds NumPy's generator likewise

PROC = PROJECT / "Data" / "processed"  # where the CSVs are read and written
CHECK_DIR = PROJECT / "Data" / "processed" / "checks"  # where the reconciliation table is written

DATE = "2026-09-14"  # the one recording date of all six sessions
DT = decision_times(PROC)  # dict: decision instants from ground_truth/decision_times.csv (the decision-time script), so none is typed here
GUARD_S = 30.0  # float constant: the settling guard in seconds at every edge next to an excluded or an attack interval
MIN_MINUTES = 2.0  # float constant: a site or receiver with fewer clean minutes than this is reported as weak
# dict: site name -> its sessions, from the handset script's measured positions; 2.6.5 stands at the 2.6.2 site after 12:34:57
SITES = {"Stave": ["2.6.12"], "2.6.2 site": ["2.6.2", "2.6.5"], "south-west site": ["2.6.10", "2.6.4", "2.6.11"]}
# dict: stream -> name; the mosaic-H is its main antenna only, the mosaic-G5 and u-blox 02 are not primary sources
RECEIVERS = {"SBF_02": "mosaic-X5", "SBF_03": "mosaic-H main antenna", "UBX_01": "u-blox 01"}
# dict: the status table's receiver tag for each stream, used only for the quietness check
STATUS_RX = {"SBF_02": "Septentrio_02", "SBF_03": "Septentrio_03", "UBX_01": "u-blox_01"}

MARGIN_CONTAMINATION_END = pd.Timestamp(f"{DATE} {DT['margin_contamination_end']}", tz="UTC")  # 15:45:12, read from ground_truth/decision_times.csv; 2.6.12: before this instant the margin is possibly contaminated
SPOOF_END_2610 = pd.Timestamp(f"{DATE} 14:44:00", tz="UTC")  # 2.6.10: spoofed tracking lasted to 14:43:59, so clean data start here; the per-second end is later, 14:44:04 pooled (X5 and mosaic-H 14:44:03, u-blox 01 14:44:04; ground_truth/spoof_end_2610.csv); this minute bound is kept
G30_NORMAL = pd.Timestamp(f"{DATE} {DT['G30_normal']}", tz="UTC")  # 13:02:41, read from ground_truth/decision_times.csv; 2.6.2: u-blox 01 reacquires the true G30 here (the satellite diagnostics script); before it G30 was an invalid measurement, consistent with the counterfeit G30 of 2.6.5 still being held ('false code lock' withdrawn)
G08_LAST = pd.Timestamp(f"{DATE} {DT['G08_last']}", tz="UTC")  # 14:44:43, read from ground_truth/decision_times.csv; 2.6.10: last second of the 3 s in which both Septentrio receivers carry G08 with a -30 m code error (diagnostic)
X5_BACK = pd.Timestamp(f"{DATE} 16:07:04", tz="UTC")  # 2.6.12: the mosaic-X5 tracks again from here (it tracked nothing 15:50:10 to 16:07:04)

CHECKS: list = []  # list of check rows, written by the second part
FAILED: list = []  # list: names of the self-checks that failed; the script ends the run with status 1 when it is not empty


def check(name: str, expected, found, ok: bool) -> None:  # function: records one self-check and prints it
    CHECKS.append({"check": name, "expected": str(expected), "found": str(found), "ok": bool(ok)})  # one row
    if ok is not None and not bool(ok):  # if: a pass/fail check (None marks a reported number) that did not pass
        FAILED.append(name)  # its name kept for the exit at the end of the run
    print(f"  [{'ok' if ok else 'FAIL'}] {name}: expected {expected}, found {found}")  # printed for the log


def utc(hms: str) -> pd.Timestamp:  # function: "HH:MM:SS" on the recording date -> a UTC timestamp
    return pd.Timestamp(f"{DATE} {hms}", tz="UTC")  # the date string and the time, tagged UTC


def stamp(t: pd.Timestamp) -> str:  # function: a timestamp -> "YYYY-MM-DD HH:MM:SS" text, byte-stable
    return t.strftime("%Y-%m-%d %H:%M:%S")  # whole seconds: every edge in this file is a whole second


def load_margins() -> tuple:  # function: the sessions and their combined margins as plain dicts
    gt = pd.read_csv(PROC / "ground_truth" / "session_ground_truth.csv")  # capture and scheduled windows
    sess = {}  # test code -> its four instants
    for _, r in gt.iterrows():  # for loop over the six sessions
        # dict literal: the session's four instants (capture start and end, scheduled start and end), each parsed to a UTC timestamp
        sess[r.test_code] = {"cap_a": pd.Timestamp(r.capture_start_utc, tz="UTC"), "cap_b": pd.Timestamp(r.capture_end_utc, tz="UTC"),
                             "sch_a": pd.Timestamp(r.scheduled_start_utc, tz="UTC"), "sch_b": pd.Timestamp(r.scheduled_end_utc, tz="UTC")}  # instants
    mc = pd.read_csv(PROC / "ground_truth" / "margin_clean_minutes.csv")  # the receiver-status script's clean margins
    mc = mc[mc.indicator.str.startswith("ALL combined")].reset_index(drop=True)  # the all-indicator rows only
    margins = []  # one dict per margin
    for _, r in mc.iterrows():  # for loop over the combined rows
        a_txt, b_txt = r.margin.split("-")  # "12:27:21-12:30:00" -> two clock texts
        a, b = utc(a_txt), utc(b_txt)  # the margin's start and end
        s = sess[r.test_code]  # the session's instants
        # margins.append adds a dict: the margin's session, edges, whether it precedes the window, and the receiver-status script's minutes
        margins.append({"test": r.test_code, "a": a, "b": b, "pre": a < s["sch_a"], "margin_min": float(r.margin_min),
                        "surviving_min": float(r.surviving_min)})  # pre = before the scheduled window
    return sess, margins  # sessions and margins


def margin_interval(m: dict, rx: str, sess: dict) -> dict:  # function: one margin's interval for one receiver, with the rules applied
    a, b = m["a"], m["b"]  # start with the whole margin
    # edge kinds: a pre-test margin starts at capture start (no guard) and ends at the attack onset; a post-test margin the reverse
    ka, kb = ("capture start", "attack onset") if m["pre"] else ("attack end", "capture end")
    notes = [f"margin {a:%H:%M:%S}-{b:%H:%M:%S} ({m['surviving_min']:.3f} min survive every indicator in the receiver-status script)"]  # rule trail
    why = None  # the reason the margin is excluded entirely, if any
    if m["test"] == "2.6.5":  # if: the session that moved
        if m["pre"]:  # if: its pre-test margin
            # reason text: why 2.6.5's pre-test margin goes (decision the 2.6.5 drive exclusion)
            why = "the 2.6.5 drive exclusion: 2.6.5 is excluded from 12:27:22 to 12:34:57 and its pre-test margin lies inside; 2.6.5 contributes to no reference position"
        else:  # else: its post-test margin
            # reason text (parenthesised string): why 2.6.5's post-test margin goes, as decided
            why = ("2.6.5 post-test margin excluded entirely: the session is attack-affected to the end of capture (ATTACK_EXTENDED_END); the X5 gain onset 12:50:59 was the cut for the X5 as a motion witness, not a cleanliness "
                   "boundary, because gain lags deception; 2.6.5 contributes to no reference position")
    if m["test"] == "2.6.12" and b <= MARGIN_CONTAMINATION_END:  # if: the margin before 15:45:12
        # reason text: why 2.6.12's margin before 15:45:12 goes
        why = "the possibly contaminated 2.6.12 margin: the margin before 15:45:12 is possibly contaminated by an unexplained step in the gains"
    if m["test"] == "2.6.12" and m["pre"] and a >= MARGIN_CONTAMINATION_END:  # if: the pre-test margin from 15:45:12
        a, ka = MARGIN_CONTAMINATION_END, "the possibly contaminated 2.6.12 margin exclusion"  # its start borders the excluded stretch
        notes.append("the possibly contaminated 2.6.12 margin excludes the time before 15:45:12")  # recorded
    if m["test"] == "2.6.10" and not m["pre"]:  # if: 2.6.10's post-test margin
        a, ka = max(a, SPOOF_END_2610), "the end of spoofed tracking in 2.6.10 exclusion"  # clean data start after the spoofed tracking ended at 14:43:59
        notes.append("the end of spoofed tracking in 2.6.10: spoofed tracking lasted to 14:43:59")  # recorded
    if m["test"] == "2.6.10" and not m["pre"] and rx in ("SBF_02", "SBF_03"):  # if: 2.6.10's post-test margin on a Septentrio receiver
        a, ka = max(a, G08_LAST + pd.Timedelta(seconds=1)), "G08 reacquisition transient"  # clean data start after G08's first three seconds
        notes.append("both Septentrio receivers carry G08 with a -30 m code error in 14:44:41-14:44:43, their first seconds after reacquiring it")  # recorded
    if m["test"] == "2.6.2" and m["pre"] and rx == "UBX_01":  # if: 2.6.2's pre-test margin on u-blox 01
        # reason text (parenthesised f-string): why u-blox 01's 2.6.2 pre-test margin goes, the invalid G30 it still held from 2.6.5 (the satellite diagnostics script)
        why = (f"u-blox 01 held an invalid G30 measurement (21-23 dB-Hz, no carrier, code error -560 m, in place from the first logged second, consistent with the counterfeit G30 that all three receivers tracked at the end of 2.6.5) until it reacquired the true "
               f"signal at {G30_NORMAL:%H:%M:%S}; excluded until then plus 30 s, which covers the whole margin (the satellite diagnostics script)")  # reason
    if m["test"] == "2.6.12" and not m["pre"]:  # if: 2.6.12's post-test margin
        if rx == "SBF_02":  # if: the mosaic-X5
            a, ka = max(a, X5_BACK), "X5 not tracking before 16:07:04"  # it tracked nothing until then
            notes.append("mosaic-X5 tracked nothing until 16:07:04")  # recorded
        if rx == "UBX_01":  # if: u-blox 01
            # reason text: why u-blox 01's post-test margin goes
            why = "u-blox 01's clock release: after its clock release (first true-time epoch 16:05:15) u-blox 01 carries an offset common to all satellites; its post-test margin is excluded"
    out = {"test": m["test"], "rx": rx, "margin": m, "notes": notes, "excluded": why, "raw_a": a, "raw_b": b}  # the result so far
    if why is not None:  # if: excluded entirely
        out.update(a=None, b=None, minutes=0.0)  # nothing kept
        return out  # done
    if ka not in ("capture start", "capture end"):  # if: the start borders an excluded or an attack interval
        a = a + pd.Timedelta(seconds=GUARD_S)  # the guard moves it inward
        notes.append(f"30 s guard at the start ({ka})")  # recorded
    if kb not in ("capture start", "capture end"):  # if: the end borders an attack interval
        b = b - pd.Timedelta(seconds=GUARD_S)  # the guard moves it inward
        notes.append(f"30 s guard at the end ({kb})")  # recorded
    if b <= a:  # if: nothing is left after the guards
        out.update(excluded="shorter than the guards after: " + "; ".join(notes[1:]), a=None, b=None, minutes=0.0)  # nothing kept, with the rules that cut it
        return out  # done
    out.update(a=a, b=b, minutes=(b - a).total_seconds() / 60.0)  # the kept interval and its length
    return out  # done


def build_intervals(sess: dict, margins: list) -> list:  # function: every (margin, receiver) result, plus 2.6.11's missing margin
    res = []  # list of result dicts
    for site, tests in SITES.items():  # for loop over the sites
        for t in tests:  # for loop over their sessions
            ms = [m for m in margins if m["test"] == t]  # the session's margins
            for rx in RECEIVERS:  # for loop over the three receivers
                if not ms:  # if: the session has no margin (2.6.11)
                    s = sess[t]  # its instants
                    # res.append adds a dict for 2.6.11, which has no margin, so that the absence is on record for each receiver
                    res.append({"site": site, "test": t, "rx": rx, "margin": None, "notes": [], "a": None, "b": None, "minutes": 0.0,
                                "excluded": "no margin: the scheduled window covers the whole capture", "raw_a": s["cap_a"], "raw_b": s["cap_a"]})  # empty
                    continue  # next receiver
                for m in ms:  # for loop over the margins
                    r = margin_interval(m, rx, sess)  # apply the rules
                    r["site"] = site  # remember the site
                    res.append(r)  # keep
    return res  # all results


def interval_rows(res: list) -> pd.DataFrame:  # function: the output table of preprocessing/reference_clean_intervals.csv
    rows = []  # one dict per result
    for r in res:  # for loop over the results
        if r["excluded"]:  # if: nothing kept from this margin
            rule = "EXCLUDED: " + r["excluded"]  # the rule, said plainly
            start = end = stamp(r["raw_a"])  # a zero-length row at the margin's start, so the exclusion is on record
        else:  # else: an interval is kept
            rule = "; ".join(r["notes"])  # the rule trail
            start, end = stamp(r["a"]), stamp(r["b"])  # its edges
        # rows.append adds a dict: one row of preprocessing/reference_clean_intervals.csv, with the rule trail in rule_applied
        rows.append({"site": r["site"], "test_code": r["test"], "receiver": r["rx"], "start_utc": start, "end_utc": end,
                     "minutes": round(r["minutes"], 3), "rule_applied": rule})  # the output row
    return pd.DataFrame(rows)  # the table


def reconcile(res: list) -> pd.DataFrame:  # function: each margin against the receiver-status script's surviving minutes (X5 view; the rules differ by receiver only at 2.6.12)
    rows = []  # one dict per margin
    seen = set()  # (test, margin start) already listed
    for r in res:  # for loop over the results
        m = r["margin"]  # the margin dict, or None for 2.6.11
        if m is None or r["rx"] != "SBF_02" or (m["test"], m["a"]) in seen:  # if: no margin, not the X5 view, or already listed
            continue  # skip
        seen.add((m["test"], m["a"]))  # now listed
        # minutes left after the rule cut and before the guards: none if the margin is excluded entirely
        raw = 0.0 if r["excluded"] else (r["raw_b"] - r["raw_a"]).total_seconds() / 60.0
        # rows.append adds a dict: one margin against the receiver-status script's surviving minutes, the minutes after the rules and after the guards
        rows.append({"test_code": m["test"], "margin": f"{m['a']:%H:%M:%S}-{m['b']:%H:%M:%S}", "margin_min": m["margin_min"],
                     # dict entry: minutes the interference-status step kept, and minutes left after the rule, before the guards
                     "section3_surviving_min": m["surviving_min"], "after_rule_min_before_guards": round(raw, 3),
                     "kept_min_x5_after_guards": round(r["minutes"], 3), "x5_outcome": r["excluded"] or "kept"})  # one row
    return pd.DataFrame(rows)  # the table


def quiet_check(res: list, margins: list) -> list:  # function: counts readings in the kept intervals that the receiver-status script's criteria call disturbed
    cols = ["test_code", "receiver", "timestamp_utc", "band", "agc_gain_db", "noise_per_ms", "cw_suppression", "antenna_id"]  # needed columns
    st = pd.read_csv(PROC / "ground_truth" / "receiver_status_timeseries.csv", low_memory=False, usecols=cols)  # the status readings
    st["t"] = pd.to_datetime(st.timestamp_utc, utc=True)  # reading times as UTC timestamps
    ind = ([(rx_, b_, "agc_gain_db") for rx_ in ("Septentrio_02", "Septentrio_03") for b_ in ("GPSL1/E1", "GLOL1", "B1")]  # six gains
           + [("u-blox_01", "L1", "noise_per_ms"), ("u-blox_01", "L1", "cw_suppression")])  # and two u-blox fields
    refs = {}  # indicator -> (reference, IQR) as the receiver-status script defines them: median of per-session margin medians, pooled IQR
    for rx_, b_, f_ in ind:  # for loop over the indicators
        g = st[(st.receiver == rx_) & (st.band == b_) & st[f_].notna()]  # this indicator's readings
        g = g[(g.antenna_id.fillna(0) == 0) | (rx_ != "Septentrio_03")]  # the mosaic-H main antenna only
        meds, pool = [], []  # per-margin medians and pooled values
        for tc in sorted({m["test"] for m in margins}):  # for loop over sessions with margins
            sel = pd.concat([g[(g.test_code == tc) & (g.t >= m["a"]) & (g.t < m["b"])][f_] for m in margins if m["test"] == tc])  # its margin readings
            if len(sel):  # if: any
                meds.append(sel.median())  # that session's margin median
                pool.append(sel)  # and its values
        allv = pd.concat(pool)  # every margin reading
        iqr = float(allv.quantile(0.75) - allv.quantile(0.25))  # pooled interquartile range
        refs[(rx_, b_, f_)] = (float(np.median(meds)), iqr if iqr > 0 else 1.0)  # reference and IQR (0 becomes 1, as in the receiver-status script)
    out = []  # one dict per kept interval
    for r in res:  # for loop over the results
        if r["excluded"]:  # if: nothing kept
            continue  # skip
        n = bad = 0  # readings and disturbed readings in this interval
        for (rx_, b_, f_), (ref, iqr) in refs.items():  # for loop over the indicators
            g = st[(st.receiver == rx_) & (st.band == b_) & (st.t >= r["a"]) & (st.t < r["b"]) & st[f_].notna()]  # readings in the interval
            if rx_ == "Septentrio_03":  # if: the mosaic-H
                g = g[g.antenna_id.fillna(0) == 0]  # main antenna only
            n += len(g)  # counted
            bad += int((g[f_] < ref - 3).sum()) if f_ == "agc_gain_db" else int((g[f_] > ref + 3 * iqr).sum())  # the receiver-status script's two rules
        out.append({"site": r["site"], "test": r["test"], "rx": r["rx"], "n": n, "bad": bad})  # the interval's counts
    return out  # the counts


# ======================================================================================================================
# the positions
# ======================================================================================================================
C_LIGHT = 299792458.0  # float constant: speed of light in m/s
WGS_A, WGS_F = 6378137.0, 1 / 298.257223563  # WGS84 semi-major axis (m) and flattening
WGS_E2 = WGS_F * (2 - WGS_F)  # first eccentricity squared
ELEV_MASK_DEG = 10.0  # float constant: satellites below this elevation are left out
REJECT_M = 30.0  # float constant: a post-fit residual larger than this is rejected, the largest first
HUMIDITY = 0.7  # float constant: relative humidity of the standard atmosphere (the value RTKLIB uses)
GPS_UNIX = 315964800.0  # float constant: the GPS epoch (6 January 1980) as unix seconds, to turn GPS time into UTC seconds
BDT_MINUS_GPST = 14.0  # float constant: BeiDou time runs 14 s behind GPS time
BDT_EPOCH_GPS_S = 1356 * 604800 + BDT_MINUS_GPST  # float constant: BDT week 0 (1 January 2006) in GPS seconds since 6 January 1980
MU_BDS, OMEGA_E_BDS = 3.986004418e14, 7.292115e-5  # BeiDou's gravitational parameter and Earth rotation rate (BDS-SIS-ICD, CGCS2000)
F_BDS = -2 * np.sqrt(MU_BDS) / C_LIGHT ** 2  # relativistic clock constant for BeiDou, s/m^0.5
BDS_GEO = set(range(1, 6)) | set(range(59, 64))  # float-free constant: PRNs of the BeiDou GEO satellites (C01-C05 and C59-C63)
# dict: stream family -> RINEX observation code per constellation (Septentrio C1C for Galileo, u-blox C1X); BeiDou B1I is C2I on both
OBS_CODES = {"SBF": {"G": "C1C", "E": "C1C", "C": "C2I"}, "UBX": {"G": "C1C", "E": "C1X", "C": "C2I"}}
BDS_OK_MEDIAN_M = 3.0  # float constant: BeiDou residuals within this median and twice this 95th percentile count as 'a few metres'
BDS_OK_P95_M = 6.0  # float constant: the 95th percentile limit of that statement
TMP_NOTE = "decompressed in memory"  # string constant: the RINEX is decompressed in memory, no temporary file is written
OBS_FILES = {"2.6.12": {"SBF_02": "SBF_02", "SBF_03": "SBF_03", "UBX_01": "UBX_01_part01"}}  # 2.6.12's u-blox 01 uses piece 1 (before the clock jumps)
RX_NAME = {**RECEIVERS, "median of medians": "median of medians"}  # names for printing, including the site rows
SITE_OF = {t: s for s, ts in SITES.items() for t in ts}  # dict comprehension: session -> its site
DAY_DIR = PROJECT / "Data" / "Day1_14092026"  # the raw recordings (read only), for the X5's own PVT
COLORS = {"SBF_02": "#0072B2", "SBF_03": "#E69F00", "UBX_01": "#009E73"}  # Okabe-Ito colours of the three receivers
MARKERS = {"2.6.12": "o", "2.6.2": "s", "2.6.10": "^", "2.6.4": "v"}  # marker per session for the session medians


def load_module(name: str, filename: str):  # function: loads codes/<filename> as a module (names start with a digit)
    import importlib.util  # standard-library tools for loading a Python file by its path
    spec = importlib.util.spec_from_file_location(name, PROJECT / "codes" / filename)  # where to load from
    mod = importlib.util.module_from_spec(spec)  # an empty module object from that spec
    sys.modules[name] = mod  # registers it under name, as a normal import would
    spec.loader.exec_module(mod)  # runs the file's top-level code (its main() is guarded and not run)
    return mod  # hands back the loaded module


def ecef_to_llh(x: float, y: float, z: float) -> tuple:  # function: ECEF metres -> geodetic latitude, longitude (degrees), ellipsoidal height (m)
    lon = np.arctan2(y, x)  # longitude from the x and y components
    p = np.hypot(x, y)  # distance from the rotation axis
    lat = np.arctan2(z, p * (1 - WGS_E2))  # first latitude guess
    for _ in range(6):  # six iterations are far below a millimetre
        n = WGS_A / np.sqrt(1 - WGS_E2 * np.sin(lat) ** 2)  # prime-vertical radius at this latitude
        h = p / np.cos(lat) - n  # height from the radius
        lat = np.arctan2(z, p * (1 - WGS_E2 * n / (n + h)))  # improved latitude
    return float(np.degrees(lat)), float(np.degrees(lon)), float(h)  # degrees, degrees, metres


def llh_to_ecef(lat_deg: float, lon_deg: float, h: float) -> np.ndarray:  # function: geodetic -> ECEF metres
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)  # radians
    n = WGS_A / np.sqrt(1 - WGS_E2 * np.sin(lat) ** 2)  # prime-vertical radius
    return np.array([(n + h) * np.cos(lat) * np.cos(lon), (n + h) * np.cos(lat) * np.sin(lon), (n * (1 - WGS_E2) + h) * np.sin(lat)])  # x, y, z


def enu_axes(lat_deg: float, lon_deg: float) -> np.ndarray:  # function: rows are the east, north and up unit vectors at a geodetic point
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)  # radians
    return np.array([[-np.sin(lon), np.cos(lon), 0.0],  # east
                     [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],  # north
                     [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]])  # up


def saastamoinen(h: float, lat_deg: float, el: np.ndarray) -> np.ndarray:  # function: tropospheric delay in metres, standard atmosphere (RTKLIB form)
    hh = min(max(h, 0.0), 10000.0)  # height limited to 0-10 km for the atmosphere formula
    pres = 1013.25 * (1 - 2.2557e-5 * hh) ** 5.2568  # standard pressure in hPa
    temp = 15.0 - 6.5e-3 * hh + 273.16  # standard temperature in K
    e = 6.108 * HUMIDITY * np.exp((17.15 * temp - 4684.0) / (temp - 38.45))  # water vapour pressure in hPa
    z = np.pi / 2 - el  # zenith angle
    lat = np.radians(lat_deg)  # latitude in radians
    trph = 0.0022768 * pres / (1 - 0.00266 * np.cos(2 * lat) - 0.00028 * hh / 1000) / np.cos(z)  # hydrostatic part
    trpw = 0.002277 * (1255 / temp + 0.05) * e / np.cos(z)  # wet part
    return trph + trpw  # total slant delay, metres


def klobuchar(t_gps_s: float, lat_deg: float, lon_deg: float, el: np.ndarray, az: np.ndarray, alpha: list, beta: list) -> np.ndarray:  # function: L1 ionospheric delay in metres (IS-GPS-200)
    phi_u, lam_u = lat_deg / 180.0, lon_deg / 180.0  # receiver latitude and longitude in semicircles
    e = el / np.pi  # elevation in semicircles
    psi = 0.0137 / (e + 0.11) - 0.022  # earth-centred angle to the ionospheric point, semicircles
    phi_i = np.clip(phi_u + psi * np.cos(az), -0.416, 0.416)  # latitude of the ionospheric point
    lam_i = lam_u + psi * np.sin(az) / np.cos(phi_i * np.pi)  # longitude of the ionospheric point
    phi_m = phi_i + 0.064 * np.cos((lam_i - 1.617) * np.pi)  # geomagnetic latitude
    t = np.mod(4.32e4 * lam_i + t_gps_s, 86400.0)  # local time at the ionospheric point
    f = 1.0 + 16.0 * (0.53 - e) ** 3  # slant factor
    per = np.maximum(sum(b * phi_m ** k for k, b in enumerate(beta)), 72000.0)  # period of the cosine
    amp = np.maximum(sum(a * phi_m ** k for k, a in enumerate(alpha)), 0.0)  # amplitude of the cosine
    x = 2 * np.pi * (t - 50400.0) / per  # phase
    day = np.where(np.abs(x) < 1.57, 5e-9 + amp * (1 - x ** 2 / 2 + x ** 4 / 24), 5e-9)  # delay in seconds: cosine by day, 5 ns by night
    return C_LIGHT * f * day  # delay in metres


def igs_klobuchar(path: Path) -> tuple:  # function: the GPS alpha and beta parameters from the IGS file's header
    import gzip  # standard-library gzip reader
    a = b = None  # not found yet
    with gzip.open(path, "rt", errors="replace") as fh:  # opens the compressed IGS file as text
        for line in fh:  # for loop over its lines
            if line[60:].startswith("END OF HEADER"):  # if: the header ends
                break  # stop reading
            if line[60:].startswith("IONOSPHERIC CORR") and line[:4] in ("GPSA", "GPSB"):  # if: the GPS alpha or beta record (Galileo's has three coefficients, BeiDou's differ)
                vals = [float(v.replace("D", "E")) for v in (line[5:17], line[17:29], line[29:41], line[41:53])]  # its four coefficients
                if line[:4] == "GPSA":  # if: GPS alpha
                    a = vals  # kept
                else:  # else: GPS beta
                    b = vals  # kept
    if a is None or b is None:  # if: the header lacks them
        sys.exit("STOP: the IGS header holds no GPSA/GPSB ionospheric parameters")  # halt
    return a, b  # alpha and beta


def bds_records(S04, path: Path, day) -> dict:  # function: BeiDou broadcast records from the IGS file, per satellite, as arrays (health 0, within a day)
    fl = lambda s: float(s.replace("D", "E")) if s.strip() else 0.0  # lambda: a Fortran D-exponent field as a float
    keys = ["af0", "af1", "af2", "crs", "dn", "m0", "cuc", "e", "cus", "sqa", "toe", "cic", "om0", "cis", "i0", "crc", "w", "omd", "idot", "toe_abs", "toc_abs", "tgd"]  # fields kept
    lines = S04.nav_text_lines(path)  # the file's lines (a .gz file is decompressed first)
    k = next(i for i, l in enumerate(lines) if l[60:].startswith("END OF HEADER")) + 1  # first data line
    recs, cur = [], None  # records and the record being read
    for l in lines[k:]:  # for loop over the data lines
        if l[:1].isalpha():  # if: a record starts
            cur = [l]  # a new record
            recs.append(cur)  # kept
        elif cur is not None and l.strip():  # elif: a continuation line
            cur.append(l)  # appended
    out = {}  # satellite -> list of records
    skipped = 0  # BeiDou records that are short or fail to parse (counted, and the run stops if any)
    for r in recs:  # for loop over the records
        sv = r[0][:3]  # the satellite name
        if sv[0] != "C":  # if: not BeiDou
            continue  # skipped
        if len(r) < 8:  # if: a BeiDou record shorter than its eight lines
            skipped += 1  # counted instead of skipped silently
            continue  # not usable
        try:  # try: a malformed record raises
            toc = S04.datetime(int(r[0][4:8]), int(r[0][9:11]), int(r[0][12:14]), int(r[0][15:17]), int(r[0][18:20]), int(r[0][21:23]), tzinfo=S04.TZ_UTC)  # BDT calendar time of clock
            v0 = [fl(r[0][23 + 19 * j:42 + 19 * j]) for j in range(3)]  # af0, af1, af2
            o = [[fl(x[4 + 19 * j:23 + 19 * j]) for j in range(4)] for x in r[1:8]]  # broadcast orbit lines 1-7
        except (ValueError, IndexError):  # except: a malformed record
            skipped += 1  # counted instead of skipped silently
            continue  # next record
        if abs((toc.date() - day).days) > S04.EPHEMERIS_MAX_AGE_DAYS or o[5][1] != 0:  # if: misdated or unhealthy (SatH1)
            continue  # not used
        toe_abs = o[4][2] * 604800 + o[2][0] + BDT_EPOCH_GPS_S  # toe as GPS seconds: BDT week and second of week, shifted to the GPS scale
        toc_abs = (toc - S04.GPS_EPOCH).total_seconds() + BDT_MINUS_GPST  # time of clock as GPS seconds (BDT calendar plus 14 s)
        # setdefault(sv, []).append adds this record's 22 numbers (clock, orbit, reference times, TGD1) to the satellite's list; the list is sorted into arrays below
        out.setdefault(sv, []).append([v0[0], v0[1], v0[2], o[0][1], o[0][2], o[0][3], o[1][0], o[1][1], o[1][2], o[1][3], o[2][0], o[2][1], o[2][2], o[2][3],
                                       o[3][0], o[3][1], o[3][2], o[3][3], o[4][0], toe_abs, toc_abs, o[5][2]])  # record; TGD1 is the third field of line 6
    if skipped:  # if: any BeiDou record was short or failed to parse
        sys.exit(f"STOP: {skipped} BeiDou broadcast records short or unparsed in {path}")  # stops: a dropped record would change which ephemeris a satellite uses
    return {sv: {k_: np.array(sorted(v, key=lambda x: x[19]))[:, j] for j, k_ in enumerate(keys)} for sv, v in out.items()}  # arrays sorted by toe


def bds_xyz(p: dict, t: np.ndarray, geo: np.ndarray) -> tuple:  # function: BeiDou satellite ECEF positions (m) and clock offsets (s), BDS-SIS-ICD; GEO rotated
    A = p["sqa"] ** 2  # semi-major axis
    tk = t - p["toe_abs"]  # time from the reference epoch
    M = p["m0"] + (np.sqrt(MU_BDS / A ** 3) + p["dn"]) * tk  # mean anomaly
    E = M.copy()  # eccentric anomaly by fixed-point iteration
    for _ in range(10):  # ten iterations
        E = M + p["e"] * np.sin(E)  # Kepler's equation
    nu = np.arctan2(np.sqrt(1 - p["e"] ** 2) * np.sin(E), np.cos(E) - p["e"])  # true anomaly
    ph = nu + p["w"]  # argument of latitude
    s2, c2 = np.sin(2 * ph), np.cos(2 * ph)  # harmonic arguments
    u = ph + p["cus"] * s2 + p["cuc"] * c2  # corrected argument of latitude
    r = A * (1 - p["e"] * np.cos(E)) + p["crs"] * s2 + p["crc"] * c2  # corrected radius
    i = p["i0"] + p["cis"] * s2 + p["cic"] * c2 + p["idot"] * tk  # corrected inclination
    x, y = r * np.cos(u), r * np.sin(u)  # position in the orbital plane
    # node longitude: MEO and IGSO use the Earth-fixed form, GEO the inertial form (no Earth rotation term in tk)
    Om = np.where(geo, p["om0"] + p["omd"] * tk - OMEGA_E_BDS * p["toe"], p["om0"] + (p["omd"] - OMEGA_E_BDS) * tk - OMEGA_E_BDS * p["toe"])
    Xg = x * np.cos(Om) - y * np.cos(i) * np.sin(Om)  # first coordinate
    Yg = x * np.sin(Om) + y * np.cos(i) * np.cos(Om)  # second coordinate
    Zg = y * np.sin(i)  # third coordinate
    a5 = np.radians(-5.0)  # the GEO tilt of minus five degrees
    th = OMEGA_E_BDS * tk  # the GEO Earth-rotation angle
    Xr = Xg * np.cos(th) + Yg * np.cos(a5) * np.sin(th) + Zg * np.sin(a5) * np.sin(th)  # GEO: rotated X
    Yr = -Xg * np.sin(th) + Yg * np.cos(a5) * np.cos(th) + Zg * np.sin(a5) * np.cos(th)  # GEO: rotated Y
    Zr = -Yg * np.sin(a5) + Zg * np.cos(a5)  # GEO: rotated Z
    X, Y, Z = np.where(geo, Xr, Xg), np.where(geo, Yr, Yg), np.where(geo, Zr, Zg)  # GEO or MEO/IGSO coordinates
    dt_ = t - p["toc_abs"]  # time from the clock reference
    dts = p["af0"] + p["af1"] * dt_ + p["af2"] * dt_ ** 2 + F_BDS * p["e"] * p["sqa"] * np.sin(E) - p["tgd"]  # clock with relativity and TGD1
    return np.stack([X, Y, Z], axis=1), dts  # positions and clock offsets


def read_obs(path: Path, spans: list, want: dict) -> pd.DataFrame:  # function: pseudoranges inside the spans, first epoch of each UTC second
    import hatanaka  # the compression package (Hatanaka plus gzip), as the conversion script uses it
    txt = hatanaka.decompress(path)  # plain RINEX in memory (bytes)
    lines = (txt.decode("utf-8", errors="replace") if isinstance(txt, bytes) else txt).split("\n")  # its lines
    codes, cur, k = {}, None, 0  # observation codes per constellation; current constellation; header end
    for k, l in enumerate(lines):  # for loop over the header lines
        lab = l[60:].strip()  # the header label
        if lab == "SYS / # / OBS TYPES":  # if: an observation-types record
            if l[0] != " ":  # if: a new constellation
                cur = l[0]  # its letter
                codes[cur] = l[7:60].split()  # its first codes
            else:  # else: a continuation line
                codes[cur] += l[7:60].split()  # more codes
        elif lab == "END OF HEADER":  # elif: the header ends
            break  # stop
    cols = {s: codes[s].index(c) for s, c in want.items() if s in codes and c in codes[s]}  # column of the wanted code per constellation
    lo = [a.timestamp() for a, _ in spans]  # span starts in unix seconds
    hi = [b.timestamp() for _, b in spans]  # span ends in unix seconds
    rows, keep, last = [], False, None  # kept observations; whether the current epoch is kept; last second kept
    for l in lines[k + 1:]:  # for loop over the data lines
        if l[:1] == ">":  # if: an epoch line
            t_gps = (datetime(int(l[2:6]), int(l[7:9]), int(l[10:12]), int(l[13:15]), int(l[16:18]), tzinfo=timezone.utc) - GPS_EPOCH).total_seconds() + float(l[18:29])  # GPS seconds
            t_unix = t_gps - LEAP_SECONDS + GPS_UNIX  # UTC as unix seconds
            sec = int(np.floor(t_unix))  # its UTC second
            keep = sec != last and any(a <= t_unix < b for a, b in zip(lo, hi))  # the first epoch of a second inside a span
            if keep:  # if: kept
                last = sec  # this second is now taken
        elif keep and l[:1] in cols:  # elif: a satellite line of a kept epoch for a constellation we use
            i = cols[l[0]]  # the column of its wanted code
            f = l[3 + 16 * i:3 + 16 * i + 14]  # that observation's 14-character field
            if f.strip():  # if: observed
                rows.append((sec, t_gps, l[:3], float(f)))  # second, GPS time, satellite, pseudorange
    return pd.DataFrame(rows, columns=["sec", "t_gps", "sv", "pr"])  # the table



class Orbits:  # class: satellite positions and clocks at transmission time for GPS, Galileo (the conversion script's functions) and BeiDou (the code above)
    def __init__(self, S04, day):  # constructor: loads the IGS records once
        self.S04 = S04  # the conversion script as a module
        self.keph = S04.ephemeris_kepler([S04.IGS_BRDC], day, inav_only=True)  # GPS and Galileo records, I/NAV only, off-date and unhealthy ones dropped
        self.bds = bds_records(S04, S04.IGS_BRDC, day)  # BeiDou records, built here

    def state(self, svs: list, t_rx: float, pr: np.ndarray) -> tuple:  # method: positions (n x 3), clock offsets (s) and a usable mask, at transmission time
        n = len(svs)  # number of satellites
        pos, dts, ok = np.zeros((n, 3)), np.zeros(n), np.zeros(n, bool)  # results
        for j, sv in enumerate(svs):  # for loop over the satellites
            src = self.bds if sv[0] == "C" else self.keph  # BeiDou or GPS/Galileo records
            if sv not in src:  # if: the IGS file has no record (C43, C45, I06 and the like)
                continue  # not used
            rec = src[sv]  # that satellite's records as arrays
            k = int(np.abs(t_rx - rec["toe_abs"]).argmin())  # the nearest record in time
            if abs(t_rx - rec["toe_abs"][k]) > self.S04.EPH_MAX_AGE_S:  # if: outside the 4 h fit interval
                continue  # not used
            p = {key: np.array([v[k]]) for key, v in rec.items()}  # the chosen record as one-element arrays
            tt = np.array([t_rx - pr[j] / C_LIGHT])  # transmission time, first guess
            if sv[0] == "C":  # if: BeiDou
                geo = np.array([int(sv[1:]) in BDS_GEO])  # whether it is a GEO satellite
                x1, d1 = bds_xyz(p, tt, geo)  # position and clock at the first guess
                x2, d2 = bds_xyz(p, tt - d1, geo)  # corrected by the satellite clock
            else:  # else: GPS or Galileo
                mu = self.S04.MU_GPS if sv[0] == "G" else self.S04.MU_GAL  # the system's constant
                x1, d1 = self.S04.kepler_xyz(p, tt, mu)  # position and clock at the first guess
                x2, d2 = self.S04.kepler_xyz(p, tt - d1, mu)  # corrected by the satellite clock
            pos[j], dts[j], ok[j] = x2[0], d2[0], bool(np.isfinite(x2).all())  # stored
        return pos, dts, ok  # the arrays


def geometry(x: np.ndarray, pos: np.ndarray) -> tuple:  # function: Sagnac-corrected range, unit vectors, elevation and azimuth from a receiver position
    th = 7.2921151467e-5 * np.linalg.norm(pos - x, axis=1) / C_LIGHT  # Earth rotation during the signal's travel
    sx = pos[:, 0] * np.cos(th) + pos[:, 1] * np.sin(th)  # satellite x rotated into the receive-time frame
    sy = -pos[:, 0] * np.sin(th) + pos[:, 1] * np.cos(th)  # satellite y likewise
    s = np.stack([sx, sy, pos[:, 2]], axis=1)  # rotated satellite positions
    d = s - x  # line of sight from the receiver
    rho = np.linalg.norm(d, axis=1)  # geometric range
    lat, lon, _ = ecef_to_llh(*x)  # receiver geodetic position
    enu = d @ enu_axes(lat, lon).T  # line of sight in east, north, up
    el = np.arcsin(np.clip(enu[:, 2] / rho, -1, 1))  # elevation
    az = np.arctan2(enu[:, 0], enu[:, 1])  # azimuth from north
    return rho, -d / rho[:, None], el, az  # range, unit vectors from the satellite to the receiver, elevation, azimuth


def corrections(x: np.ndarray, t_gps: float, svs: list, el: np.ndarray, az: np.ndarray, kl: tuple) -> np.ndarray:  # function: troposphere plus ionosphere (L1 scaled to B1I for BeiDou), metres
    lat, lon, h = ecef_to_llh(*x)  # receiver geodetic position
    trop = saastamoinen(h, lat, np.maximum(el, np.radians(5.0)))  # troposphere, elevation floored at 5 degrees against blow-up
    iono = klobuchar(t_gps, lat, lon, np.maximum(el, np.radians(5.0)), az, kl[0], kl[1])  # Klobuchar delay at L1
    scale = np.array([(1575.42 / 1561.098) ** 2 if sv[0] == "C" else 1.0 for sv in svs])  # B1I is lower in frequency than L1: its delay is larger
    return trop + iono * scale  # total path delay


def solve_epoch(svs: list, pr: np.ndarray, pos: np.ndarray, dts: np.ndarray, t_gps: float, x0: np.ndarray, kl: tuple) -> dict | None:  # function: one epoch's weighted least squares with one clock per constellation and iterative rejection
    sysl = np.array([sv[0] for sv in svs])  # constellation letter per satellite
    use = np.ones(len(svs), bool)  # satellites still in the solution
    n_rej = 0  # rejected so far
    rej = []  # indices of the rejected satellites
    x = x0.copy()  # the position, refined
    while True:  # loop: solve, then reject the worst residual if it exceeds the limit
        for _ in range(8):  # up to eight Gauss-Newton iterations
            idx = np.flatnonzero(use)  # satellites in use
            rho, e, el, az = geometry(x, pos[idx])  # geometry from the current position
            m = el >= np.radians(ELEV_MASK_DEG)  # the elevation mask
            idx, rho, e, el, az = idx[m], rho[m], e[m], el[m], az[m]  # only those above the mask
            systems = [s for s in "GE" if (sysl[idx] == s).sum() >= 2]  # a constellation counts only with two satellites or more
            keepm = np.isin(sysl[idx], systems)  # drop satellites of a constellation with one satellite
            idx, rho, e, el, az = idx[keepm], rho[keepm], e[keepm], el[keepm], az[keepm]  # only those
            if len(idx) < 3 + len(systems) + 1:  # if: no redundancy left
                return None  # no solution this epoch
            z = pr[idx] - rho + C_LIGHT * dts[idx] - corrections(x, t_gps, [svs[i] for i in idx], el, az, kl)  # observed minus modelled, clock still in
            H = np.zeros((len(idx), 3 + len(systems)))  # design matrix
            H[:, :3] = e  # position columns: unit vectors from the satellite to the receiver
            for c, s in enumerate(systems):  # for loop over the constellations
                H[:, 3 + c] = (sysl[idx] == s)  # one clock column per constellation
            w = np.sin(el) ** 2  # elevation weights
            A = H.T @ (H * w[:, None])  # weighted normal matrix
            th = np.linalg.solve(A, H.T @ (w * z))  # position correction and clocks
            x = x + th[:3]  # updated position
            if np.linalg.norm(th[:3]) < 1e-4:  # if: converged to 0.1 mm
                break  # stop iterating
        r = z - H @ th  # post-fit residuals
        if np.abs(r).max() > REJECT_M:  # if: any residual beyond the limit
            Hw = H * np.sqrt(w)[:, None]  # design matrix scaled by the square root of the weights
            lev = np.einsum("ij,jk,ik->i", Hw, np.linalg.inv(A), Hw)  # leverage of each observation (diagonal of the hat matrix)
            # the observation whose deletion residual |r| / (1 - leverage) is largest goes: with raw residuals one bad pseudorange of high leverage pulls the others and a good one can be dropped
            rej.append(int(idx[(np.abs(r) / np.maximum(1 - lev, 1e-6)).argmax()]))  # remember which satellite goes
            use[rej[-1]] = False  # drop the worst one
            n_rej += 1  # counted
            continue  # solve again
        return {"x": x, "idx": idx, "res": r, "el": el, "n_rej": n_rej, "rej": rej, "systems": systems, "n": len(idx), "w": w}  # the epoch's solution


def solve_session(orb: Orbits, obs: pd.DataFrame, x0: np.ndarray, kl: tuple) -> tuple:  # function: all epochs of one session and receiver -> epoch table and residual table
    ep_rows, res_rows = [], []  # outputs
    for sec, g in obs.groupby("sec", sort=True):  # for loop over the UTC seconds
        g = g[g.sv.str[0].isin(["G", "E"])]  # GPS and Galileo only; BeiDou is the test, not the position
        t_gps = float(g.t_gps.iloc[0])  # the epoch's GPS time
        svs = list(g.sv)  # satellites
        pos, dts, ok = orb.state(svs, t_gps, g.pr.to_numpy())  # their positions and clocks
        if ok.sum() < 5:  # if: too few usable satellites
            continue  # no solution
        sel = np.flatnonzero(ok)  # usable ones
        sol = solve_epoch([svs[i] for i in sel], g.pr.to_numpy()[sel], pos[sel], dts[sel], t_gps, x0, kl)  # the solution
        if sol is None:  # if: none
            continue  # next epoch
        lat, lon, h = ecef_to_llh(*sol["x"])  # geodetic
        # ep_rows.append adds a dict: one 1 Hz solution (ECEF, geodetic, satellites used per constellation, rejected observations, rms residual)
        ep_rows.append({"sec": int(sec), "x": sol["x"][0], "y": sol["x"][1], "z": sol["x"][2], "lat": lat, "lon": lon, "h": h,
                        # dict entry: GPS and Galileo satellites that entered this epoch's position solution
                        "n_gps": int(sum(svs[sel[i]][0] == "G" for i in sol["idx"])), "n_gal": int(sum(svs[sel[i]][0] == "E" for i in sol["idx"])),
                        "n_rejected": sol["n_rej"], "rejected_sv": ";".join(svs[sel[i]] for i in sol["rej"]), "rms_m": float(np.sqrt(np.mean(sol["res"] ** 2)))})  # one row per epoch
        for i, r_, e_ in zip(sol["idx"], sol["res"], sol["el"]):  # for loop over the satellites in the solution
            res_rows.append((int(sec), svs[sel[i]], float(r_), float(np.degrees(e_))))  # second, satellite, residual, elevation
    return pd.DataFrame(ep_rows), pd.DataFrame(res_rows, columns=["sec", "sv", "res_m", "el_deg"])  # the two tables


def bds_test(orb: Orbits, obs: pd.DataFrame, x_site: np.ndarray, kl: tuple) -> pd.DataFrame:  # function: BeiDou C2I residuals with the position fixed, one BeiDou clock per epoch
    out = []  # (second, satellite, residual, elevation)
    for sec, g in obs[obs.sv.str[0] == "C"].groupby("sec", sort=True):  # for loop over the seconds, BeiDou observations only
        svs = list(g.sv)  # satellites
        t_gps = float(g.t_gps.iloc[0])  # GPS time of the epoch
        pos, dts, ok = orb.state(svs, t_gps, g.pr.to_numpy())  # positions and clocks from the BeiDou code
        if ok.sum() < 4:  # if: too few
            continue  # no epoch
        sel = np.flatnonzero(ok)  # usable satellites
        rho, e, el, az = geometry(x_site, pos[sel])  # geometry from the fixed position
        m = el >= np.radians(ELEV_MASK_DEG)  # the elevation mask
        if m.sum() < 4:  # if: too few above the mask
            continue  # no epoch
        sel, rho, el, az = sel[m], rho[m], el[m], az[m]  # only those above the mask
        z = g.pr.to_numpy()[sel] - rho + C_LIGHT * dts[sel] - corrections(x_site, t_gps, [svs[i] for i in sel], el, az, kl)  # observed minus modelled
        w = np.sin(el) ** 2  # elevation weights
        b = float(np.sum(w * z) / np.sum(w))  # the one BeiDou clock of the epoch, weighted mean
        for i, r_, e_ in zip(sel, z - b, el):  # for loop over the satellites
            out.append((int(sec), svs[i], float(r_), float(np.degrees(e_))))  # one row
    return pd.DataFrame(out, columns=["sec", "sv", "res_m", "el_deg"])  # the table


def read_pvt(S03, folder: Path, spans: list) -> pd.DataFrame:  # function: the mosaic-X5's own PVTGeodetic solution inside the spans (computed solutions only)
    import struct  # standard-library binary decoding
    from helpers import gps_to_utc  # receiver time stamp -> UTC
    f = next(folder.glob("*_SBF_02.sbf"))  # the X5 file
    integ = S03.Integrity()  # the checked reader's integrity container
    rows = []  # decoded blocks
    for blk, rev, tow, wnc, p, end in S03.sbf_blocks(f, integ, {4007}):  # CRC-verified PVTGeodetic blocks
        t = gps_to_utc(tow, wnc)  # receiver time -> UTC
        if t is None or len(p) < 26:  # if: no valid time or a short payload
            continue  # skipped
        if not any(a <= t < b for a, b in spans):  # if: outside the clean intervals
            continue  # skipped
        mode, err = p[0] & 0x0F, p[1]  # solution type and error code
        lat, lon, h = struct.unpack_from("<ddd", p, 2)  # latitude and longitude in radians, ellipsoidal height in metres
        if mode in (0, 3) or err != 0 or lat < -1e10 or lon < -1e10 or h < -1e10:  # if: no computed solution (or a fixed location, never a measured position)
            continue  # skipped
        rows.append((np.degrees(lat), np.degrees(lon), h))  # kept
    return pd.DataFrame(rows, columns=["lat", "lon", "h"])  # the table


def stats_row(df: pd.DataFrame, x_med: np.ndarray) -> dict:  # function: scatter of epoch solutions about their median, horizontal and vertical 95th percentiles
    lat, lon, h = ecef_to_llh(*x_med)  # geodetic position of the median
    d = np.stack([df.x - x_med[0], df.y - x_med[1], df.z - x_med[2]], axis=1) @ enu_axes(lat, lon).T  # offsets in east, north, up
    return {"h_p95_m": float(np.percentile(np.hypot(d[:, 0], d[:, 1]), 95)), "v_p95_m": float(np.percentile(np.abs(d[:, 2]), 95))}  # p95 values


def obs_path(folder_name: str, test: str, rx: str) -> Path:  # function: the RINEX observation file of a session and stream
    stem = OBS_FILES.get(test, {}).get(rx, rx)  # the stream's file part (2.6.12's u-blox 01 is piece 1)
    return PROJECT / "Data" / "rinex" / folder_name / f"{folder_name.split('_T_')[0]}_{stem}.crx.gz"  # e.g. 20260914_155653_SBF_03.crx.gz


def diff_stats(a: pd.DataFrame, b: pd.DataFrame, ref: np.ndarray) -> dict | None:  # function: per-epoch difference of two receivers' solutions on common seconds, horizontal and vertical
    m = a.merge(b, on="sec", suffixes=("_a", "_b"))  # the seconds both have
    if len(m) == 0:  # if: none in common
        return None  # nothing to compare
    lat, lon, _ = ecef_to_llh(*ref)  # the site's geodetic position, for the east-north-up axes
    d = np.stack([m.x_a - m.x_b, m.y_a - m.y_b, m.z_a - m.z_b], axis=1) @ enu_axes(lat, lon).T  # per-epoch difference in east, north, up
    hz, vt = np.hypot(d[:, 0], d[:, 1]), d[:, 2]  # horizontal distance and signed vertical difference
    # return a dict literal: the number of common seconds and the median and 95th percentile of the horizontal and vertical per-epoch differences
    return {"n_common": len(m), "h_median_m": float(np.median(hz)), "h_p95_m": float(np.percentile(hz, 95)),
            "v_median_abs_m": float(np.median(np.abs(vt))), "v_p95_abs_m": float(np.percentile(np.abs(vt), 95)), "v_median_signed_m": float(np.median(vt))}  # the statistics


def solve_subset(orb: Orbits, g: pd.DataFrame, sats: list, x0: np.ndarray, kl: tuple) -> np.ndarray | None:  # function: one epoch solved from a given set of satellites only
    g = g[g.sv.isin(sats)]  # the observations of those satellites
    svs = list(g.sv)  # their names
    pos, dts, ok = orb.state(svs, float(g.t_gps.iloc[0]), g.pr.to_numpy())  # positions and clocks
    sel = np.flatnonzero(ok)  # usable ones
    if len(sel) < 5:  # if: too few
        return None  # no solution
    sol = solve_epoch([svs[i] for i in sel], g.pr.to_numpy()[sel], pos[sel], dts[sel], float(g.t_gps.iloc[0]), x0, kl)  # the solution
    return None if sol is None else sol["x"]  # its position


def common_sat_check(orb: Orbits, obs_all: dict, rs: pd.DataFrame, ep: pd.DataFrame, site_x: dict, kl: tuple, add) -> None:  # function: check 1 again with the same satellites in both receivers of a pair, per epoch
    used = {k: set(g.sv) for k, g in rs.groupby(["test_code", "receiver", "sec"])}  # satellites in each receiver's solution per second
    print("check 1b: the same pairs with identical satellite sets per epoch (each receiver re-solved from the satellites both used)")  # heading
    for test in ["2.6.12", "2.6.2", "2.6.4"]:  # for loop over sessions where two or more receivers have clean epochs
        for a_, b_ in (("SBF_02", "SBF_03"), ("SBF_02", "UBX_01"), ("SBF_03", "UBX_01")):  # the receiver pairs
            if (test, a_) not in obs_all or (test, b_) not in obs_all:  # if: a receiver has no clean interval here
                continue  # skip
            ea, eb = ep[(ep.test_code == test) & (ep.receiver == a_)], ep[(ep.test_code == test) & (ep.receiver == b_)]  # the two receivers' solved epochs
            if len(ea) == 0 or len(eb) == 0:  # if: none
                continue  # skip
            xa0 = np.array([ea.x.median(), ea.y.median(), ea.z.median()])  # starting position for the re-solves
            gA, gB = dict(tuple(obs_all[(test, a_)].groupby("sec"))), dict(tuple(obs_all[(test, b_)].groupby("sec")))  # observations per second
            rows = []  # (period, east, north, up) of the per-epoch difference
            site = ea.site.iloc[0]  # the site
            lat, lon, _ = ecef_to_llh(*site_x[site])  # its geodetic position for the axes
            for sec in sorted(set(ea.sec) & set(eb.sec)):  # for loop over the seconds both receivers solved
                sats = sorted(used.get((test, a_, sec), set()) & used.get((test, b_, sec), set()))  # the satellites both used
                if len(sats) < 5:  # if: too few in common
                    continue  # skip
                xa, xb = solve_subset(orb, gA[sec], sats, xa0, kl), solve_subset(orb, gB[sec], sats, xa0, kl)  # each receiver from the common set
                if xa is None or xb is None:  # if: either failed
                    continue  # skip
                d = (xa - xb) @ enu_axes(lat, lon).T  # the difference in east, north, up
                rows.append((ea[ea.sec == sec].period.iloc[0], float(np.hypot(d[0], d[1])), float(d[2])))  # period, horizontal, vertical
            df = pd.DataFrame(rows, columns=["period", "hz", "vt"])  # the differences
            for per, q in df.groupby("period"):  # for loop over the two margins
                grp = f"{test} {per} {RECEIVERS[a_]} minus {RECEIVERS[b_]}"  # the group label
                # dict literal: the statistics of one pair, one period, on identical satellite sets (common seconds, horizontal and vertical median and 95th percentile, signed vertical median)
                st = {"n_common": len(q), "h_median_m": float(q.hz.median()), "h_p95_m": float(q.hz.quantile(0.95)), "v_median_abs_m": float(q.vt.abs().median()),
                      "v_p95_abs_m": float(q.vt.abs().quantile(0.95)), "v_median_signed_m": float(q.vt.median())}  # the statistics
                for k_, v_ in st.items():  # for loop over them
                    add("1b same antenna, common satellites", grp, k_, v_)  # recorded
                print(f"  {grp:62s} n={st['n_common']:4d} horizontal median {st['h_median_m']:.2f} p95 {st['h_p95_m']:.2f} m; vertical |median| {st['v_median_abs_m']:.2f} p95 {st['v_p95_abs_m']:.2f} m (signed median {st['v_median_signed_m']:+.2f})")  # printed


def write_rounded(df: pd.DataFrame, path: Path) -> None:  # function: writes a table with latitude and longitude to nine decimals (about a tenth of a millimetre) and the other numbers to four
    out = df.copy()  # a copy, so the caller's table keeps its precision
    for c in out.columns:  # for loop over the columns
        if pd.api.types.is_float_dtype(out[c]):  # if: a floating-point column
            out[c] = out[c].round(9 if c in ("lat", "lon", "lat_deg", "lon_deg") else 4)  # nine decimals for degrees, four for metres and the rest
    out.to_csv(path, index=False)  # written; the floats print with their shortest exact form, which is stable between runs


def main_positions(res: list, tbl: pd.DataFrame) -> None:  # function: the positions and their checks
    S03 = load_module("receiver_status_for_reference_positions", "1_ground_truth/03_receiver_status.py")  # the receiver-status script's script: the checked SBF reader (for the X5's own PVT)
    S04 = load_module("convert_rinex_for_reference_positions", "2_preprocessing/01_convert_rinex.py")  # the conversion script's script: orbit functions and the IGS record reader
    import datetime as dtm  # standard-library dates, for the day of the recording
    import hashlib  # standard-library hashing, for the IGS file's checksum
    sha = hashlib.sha256(S04.IGS_BRDC.read_bytes()).hexdigest()  # the IGS file's SHA-256
    check("the IGS broadcast file matches its logged SHA-256 (1)", S04.IGS_SHA256[:12], sha[:12], sha == S04.IGS_SHA256)  # the same file as every earlier section
    orb = Orbits(S04, dtm.date(2026, 9, 14))  # satellite orbits and clocks from the IGS file only
    kl = igs_klobuchar(S04.IGS_BRDC)  # the GPS Klobuchar parameters from its header
    print(f"IGS records: {len(orb.keph)} GPS+Galileo satellites, {len(orb.bds)} BeiDou satellites; Klobuchar alpha {kl[0]}")  # basic counts
    gt = pd.read_csv(PROC / "ground_truth" / "session_ground_truth.csv").set_index("test_code")  # session folders and windows
    dist = pd.read_csv(PROC / "ground_truth" / "session_distance_to_sender.csv").dropna(subset=["median_lat"]).drop_duplicates("test_code", keep="last").set_index("test_code")  # starting positions only
    kept = tbl[tbl.minutes > 0]  # the intervals actually kept
    ep_all, res_all, obs_all = [], [], {}  # epoch tables, residual tables and the observation tables per session and receiver
    for (test, rx), g in kept.groupby(["test_code", "receiver"], sort=False):  # for loop over session x receiver with kept intervals
        spans = [(pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC")) for a, b in zip(g.start_utc, g.end_utc)]  # its kept intervals
        folder = gt.loc[test, "session_folder"]  # the session folder name
        obs = read_obs(obs_path(folder, test, rx), spans, OBS_CODES[rx[:3]])  # the 1 Hz pseudoranges inside the intervals
        obs_all[(test, rx)] = obs  # kept for the BeiDou test
        x0 = llh_to_ecef(float(dist.loc[test, "median_lat"]), float(dist.loc[test, "median_lon"]), 0.0)  # only a starting point for the iteration
        ep, rs = solve_session(orb, obs, x0, kl)  # the 1 Hz solutions and their residuals
        ep["test_code"], ep["receiver"] = test, rx  # labels
        rs["test_code"], rs["receiver"] = test, rx  # labels
        ep["site"] = g.site.iloc[0]  # the site
        ep["period"] = np.where(ep.sec < pd.Timestamp(gt.loc[test, "scheduled_start_utc"], tz="UTC").timestamp(), "pre-test", "post-test")  # which margin
        print(f"  {test} {RECEIVERS[rx]:24s} {len(obs.sec.unique()):5d} seconds read, {len(ep):5d} solved, {int(ep.n_rejected.sum())} observations rejected")  # progress line
        ep_all.append(ep)  # kept
        res_all.append(rs)  # kept
    ep = pd.concat(ep_all, ignore_index=True)  # every 1 Hz solution
    rs = pd.concat(res_all, ignore_index=True)  # every residual
    # ---- medians per session x receiver and per site ----
    pos_rows, med = [], {}  # output rows; (test, rx) -> median ECEF
    for (test, rx), g in ep.groupby(["test_code", "receiver"], sort=False):  # for loop over session x receiver
        xm = np.array([g.x.median(), g.y.median(), g.z.median()])  # the median ECEF position
        med[(test, rx)] = xm  # kept
        lat, lon, h = ecef_to_llh(*xm)  # its geodetic form
        # pos_rows.append adds a dict: one session x receiver median position with its scatter and counts (the row of preprocessing/reference_positions.csv)
        pos_rows.append({"scope": "session x receiver", "site": g.site.iloc[0], "test_code": test, "receiver": rx, "n_epochs": len(g),
                         # dict entry: rejected observations and the median position in ECEF and geodetic coordinates
                         "n_obs_rejected": int(g.n_rejected.sum()), "x_m": xm[0], "y_m": xm[1], "z_m": xm[2], "lat_deg": lat, "lon_deg": lon, "h_m": h,
                         **stats_row(g, xm), "median_gps": float(g.n_gps.median()), "median_gal": float(g.n_gal.median())})  # the row
    site_x = {}  # site -> combined ECEF estimate
    for site, tests in SITES.items():  # for loop over the sites
        keys = [k for k in med if k[0] in tests]  # its session x receiver medians
        xs = np.array([med[k] for k in keys])  # their positions
        xc = np.median(xs, axis=0)  # the combined estimate: median of the medians
        site_x[site] = xc  # kept
        lat, lon, h = ecef_to_llh(*xc)  # geodetic form
        d = (xs - xc) @ enu_axes(lat, lon).T  # each median's offset from the estimate, east north up
        hd = np.hypot(d[:, 0], d[:, 1])  # horizontal offsets
        # pos_rows.append adds a dict: the combined site estimate (median of the session x receiver medians) and the spread between those medians
        pos_rows.append({"scope": "site", "site": site, "test_code": "+".join(sorted({k[0] for k in keys})), "receiver": "median of medians", "n_epochs": int(sum((ep.test_code == k[0]) & (ep.receiver == k[1]) for k in keys).sum()),
                         # dict entry: the combined site position in ECEF and geodetic coordinates; no rejections at this level
                         "n_obs_rejected": 0, "x_m": xc[0], "y_m": xc[1], "z_m": xc[2], "lat_deg": lat, "lon_deg": lon, "h_m": h,
                         # dict entry: per-epoch scatter and satellite counts do not apply to a site row, so they stay empty
                         "h_p95_m": np.nan, "v_p95_m": np.nan, "median_gps": np.nan, "median_gal": np.nan,
                         # dict entry: largest horizontal and vertical distance of a session median from the site estimate
                         "spread_h_max_m": float(hd.max()), "spread_v_max_m": float(np.abs(d[:, 2]).max()),
                         "spread_h_pairwise_max_m": float(max(np.hypot(*((a_ - b_) @ enu_axes(lat, lon).T)[:2]) for a_ in xs for b_ in xs))})  # spread between the session x receiver medians
    pos = pd.DataFrame(pos_rows)  # the positions table
    write_rounded(pos, PROC / "preprocessing" / "reference_positions.csv")  # written, latitude and longitude to nine decimals, everything else to four
    write_rounded(ep.drop(columns=["period"]), PROC / "preprocessing" / "reference_position_epochs.csv")  # written
    for r_ in pos_rows:  # for loop over the rows, printed
        print(f"  {r_['scope'][:7]:7s} {r_['site']:16s} {r_['test_code']:18s} {RX_NAME.get(r_['receiver'], r_['receiver']):24s} n={r_['n_epochs']:5d} lat {r_['lat_deg']:.7f} lon {r_['lon_deg']:.7f} h {r_['h_m']:7.2f}  p95 h {r_['h_p95_m']:.2f} v {r_['v_p95_m']:.2f}" + (f"  spread of the medians: horizontal max {r_['spread_h_max_m']:.2f}, pairwise {r_['spread_h_pairwise_max_m']:.2f}, vertical max {r_['spread_v_max_m']:.2f} m" if r_['scope'] == 'site' else ""))  # one line per row
    chk = []  # long-form check rows: check, group, quantity, value
    add = lambda c, g_, q, v: chk.append({"check": c, "group": g_, "quantity": q, "value": v})  # lambda: appends one number to the check table
    # ---- check 1: same antenna, per-epoch differences on common seconds ----
    print("check 1: receivers within one session, per-epoch differences on common seconds")  # heading
    for test in ["2.6.12", "2.6.2", "2.6.10", "2.6.4"]:  # for loop over the sessions with solutions
        for a_, b_ in (("SBF_02", "SBF_03"), ("SBF_02", "UBX_01"), ("SBF_03", "UBX_01")):  # the three receiver pairs
            for per in ("pre-test", "post-test"):  # for loop over the two margins
                ea = ep[(ep.test_code == test) & (ep.receiver == a_) & (ep.period == per)]  # first receiver's solutions
                eb = ep[(ep.test_code == test) & (ep.receiver == b_) & (ep.period == per)]  # second receiver's solutions
                st = diff_stats(ea, eb, site_x[ea.site.iloc[0]] if len(ea) else np.zeros(3)) if len(ea) and len(eb) else None  # the statistics, if any common seconds
                if st is None:  # if: nothing in common
                    continue  # nothing to report
                grp = f"{test} {per} {RECEIVERS[a_]} minus {RECEIVERS[b_]}"  # the group label
                for q, v in st.items():  # for loop over the statistics
                    add("1 same antenna", grp, q, v)  # recorded
                print(f"  {grp:62s} n={st['n_common']:4d} horizontal median {st['h_median_m']:.2f} p95 {st['h_p95_m']:.2f} m; vertical |median| {st['v_median_abs_m']:.2f} p95 {st['v_p95_abs_m']:.2f} m (signed median {st['v_median_signed_m']:+.2f})")  # printed
    common_sat_check(orb, obs_all, rs, ep, site_x, kl, add)  # check 1 again on identical satellite sets
    # ---- satellites with a persistent offset in the kept intervals ----
    sat_off = rs.groupby(["test_code", "receiver", "sv"]).res_m.agg(["median", "size"]).reset_index()  # median signed residual per satellite
    flagged = sat_off[(sat_off["size"] >= 30) & (sat_off["median"].abs() > 5.0)]  # satellites with at least 30 epochs and a median beyond 5 m
    for _, r_ in flagged.iterrows():  # for loop over them
        add("satellite offsets", f"{r_.test_code} {RECEIVERS[r_.receiver]} {r_.sv}", "median_residual_m", float(r_["median"]))  # recorded
    print(f"satellites with a median residual beyond 5 m over at least 30 epochs: {len(flagged)}" + ("" if len(flagged) == 0 else "  " + ", ".join(f"{a} {b} {c} {d:+.1f} m" for a, b, c, d in zip(flagged.test_code, flagged.receiver, flagged.sv, flagged["median"]))))  # printed
    check("no satellite in a kept interval has a median residual beyond 5 m", 0, len(flagged), len(flagged) == 0)  # persistent offsets (counterfeit or fault) are excluded before this point
    # ---- check 2: same site ----
    print("check 2: the south-west site, 2.6.10 against 2.6.4; and within-session stability (pre-test against post-test)")  # heading
    def sess_med(test, rx, per=None):  # nested function: a session x receiver median position, optionally for one margin
        g = ep[(ep.test_code == test) & (ep.receiver == rx)]  # that session and receiver
        g = g if per is None else g[g.period == per]  # one margin if asked
        return (np.array([g.x.median(), g.y.median(), g.z.median()]), len(g)) if len(g) else (None, 0)  # position and epoch count
    ref_sw = site_x["south-west site"]  # the site's estimate, for the axes
    lat_sw, lon_sw, _ = ecef_to_llh(*ref_sw)  # its geodetic position
    sw_flag = False  # whether any receiver differs by more than 3 m horizontally
    for rx in RECEIVERS:  # for loop over the receivers
        x1, n1 = sess_med("2.6.10", rx)  # 2.6.10's median and epoch count
        x2, n2 = sess_med("2.6.4", rx)  # 2.6.4's
        if x1 is None or x2 is None:  # if: one is missing
            continue  # nothing to compare
        d = (x1 - x2) @ enu_axes(lat_sw, lon_sw).T  # the difference in east, north, up
        hz = float(np.hypot(d[0], d[1]))  # horizontal difference
        sw_flag = sw_flag or hz > 3.0  # beyond 3 m horizontally?
        add("2 same site", f"south-west 2.6.10 minus 2.6.4 {RECEIVERS[rx]}", "h_diff_m", hz)  # recorded
        add("2 same site", f"south-west 2.6.10 minus 2.6.4 {RECEIVERS[rx]}", "v_diff_m", float(d[2]))  # recorded
        add("2 same site", f"south-west 2.6.10 minus 2.6.4 {RECEIVERS[rx]}", "n_epochs_2.6.10", n1)  # the weak session's epoch count
        add("2 same site", f"south-west 2.6.10 minus 2.6.4 {RECEIVERS[rx]}", "n_epochs_2.6.4", n2)  # the other session's
        print(f"  {RECEIVERS[rx]:24s} horizontal {hz:.2f} m, vertical {d[2]:+.2f} m   (2.6.10: {n1} epochs, WEAK; 2.6.4: {n2} epochs)")  # printed
    add("2 same site", "south-west", "any receiver differs by more than 3 m horizontally", float(sw_flag))  # the decision flag
    for test in ["2.6.12", "2.6.2"]:  # for loop over the sessions with both margins
        for rx in RECEIVERS:  # for loop over the receivers
            xa, na = sess_med(test, rx, "pre-test")  # pre-test median
            xb, nb = sess_med(test, rx, "post-test")  # post-test median
            if xa is None or xb is None:  # if: one margin missing
                continue  # nothing to compare
            lat0, lon0, _ = ecef_to_llh(*xa)  # axes at the pre-test position
            d = (xa - xb) @ enu_axes(lat0, lon0).T  # difference in east, north, up
            add("2b within session", f"{test} {RECEIVERS[rx]} pre-test minus post-test", "h_diff_m", float(np.hypot(d[0], d[1])))  # recorded
            add("2b within session", f"{test} {RECEIVERS[rx]} pre-test minus post-test", "v_diff_m", float(d[2]))  # recorded
            print(f"  {test} {RECEIVERS[rx]:24s} pre-test minus post-test: horizontal {np.hypot(d[0], d[1]):.2f} m, vertical {d[2]:+.2f} m ({na} and {nb} epochs)")  # printed
    # ---- check 3: the receiver's own PVT ----
    print("check 3: mosaic-X5 PVTGeodetic median over the same clean intervals against this estimate")  # heading
    pvt_med = {}  # test -> (lat, lon, h, n)
    for test in ["2.6.12", "2.6.2", "2.6.10", "2.6.4"]:  # for loop over the sessions
        g = kept[(kept.test_code == test) & (kept.receiver == "SBF_02")]  # the X5's kept intervals
        spans = [(pd.Timestamp(a, tz="UTC").to_pydatetime(), pd.Timestamp(b, tz="UTC").to_pydatetime()) for a, b in zip(g.start_utc, g.end_utc)]  # as datetimes
        pv = read_pvt(S03, DAY_DIR / gt.loc[test, "session_folder"], spans)  # the X5's own solutions there
        if len(pv) == 0:  # if: none
            add("3 own PVT", test, "n_pvt", 0)  # recorded
            print(f"  {test}: no computed PVT solution inside the clean intervals (n=0)")  # printed
            continue  # next session
        xp = llh_to_ecef(float(pv.lat.median()), float(pv.lon.median()), float(pv.h.median()))  # the PVT median as ECEF
        pvt_med[test] = (float(pv.lat.median()), float(pv.lon.median()), float(pv.h.median()), len(pv))  # kept for the figure
        xo = med[(test, "SBF_02")]  # our X5 estimate
        lat0, lon0, _ = ecef_to_llh(*xo)  # axes
        d = (xp - xo) @ enu_axes(lat0, lon0).T  # PVT minus ours
        add("3 own PVT", test, "n_pvt", len(pv))  # recorded
        add("3 own PVT", test, "h_diff_m", float(np.hypot(d[0], d[1])))  # recorded
        add("3 own PVT", test, "v_diff_m", float(d[2]))  # recorded
        print(f"  {test}: n={len(pv)} PVT solutions; PVT minus ours: horizontal {np.hypot(d[0], d[1]):.2f} m, vertical {d[2]:+.2f} m")  # printed
    # ---- check 6: the handsets of the handset script, an independent instrument ----
    print("check 6: site estimates against the handset medians of the handset script (independent instrument)")  # heading
    hs = pd.read_csv(PROC / "ground_truth" / "session_distance_to_sender.csv").dropna(subset=["median_lat"])  # the handset script's handset medians per session part
    for site, xc in site_x.items():  # for loop over the sites
        lat0, lon0, _ = ecef_to_llh(*xc)  # the estimate's geodetic position
        for _, r_ in hs[hs.test_code.isin(SITES[site]) & (hs.part != "moving")].iterrows():  # the handset medians of the site's sessions
            d = (llh_to_ecef(r_.median_lat, r_.median_lon, 0.0) - llh_to_ecef(lat0, lon0, 0.0)) @ enu_axes(lat0, lon0).T  # offset on the ground
            hz = float(np.hypot(d[0], d[1]))  # horizontal difference
            add("6 handsets", f"{site} {r_.test_code}", "h_diff_m", hz)  # recorded
            print(f"  {site:16s} {r_.test_code:7s} handset median is {hz:.2f} m from the estimate")  # printed
    # ---- check 7: the 2.6.5 stops against the reference positions ----
    print("check 7: the two 2.6.5 stops (mosaic-X5 PVT medians of the motion script) against the site estimates")  # heading
    vs = pd.read_csv(PROC / "preprocessing" / "vehicle_motion_stops.csv")  # the motion script's stop positions
    for lab, row, site in (("stop 2 (Stave)", vs[(vs.scope == "OBD stop") & (vs.segment == 2)].iloc[0], "Stave"), ("stop 4 (final)", vs[(vs.scope == "OBD stop") & (vs.segment == 4)].iloc[0], "2.6.2 site")):  # the two stops
        xc = site_x[site]  # the site estimate
        lat0, lon0, _ = ecef_to_llh(*xc)  # its geodetic position
        d = (llh_to_ecef(row.median_lat, row.median_lon, 0.0) - llh_to_ecef(lat0, lon0, 0.0)) @ enu_axes(lat0, lon0).T  # offset on the ground
        add("7 2.6.5 stops", f"{lab} against {site}", "h_diff_m", float(np.hypot(d[0], d[1])))  # recorded
        print(f"  2.6.5 {lab:15s} is {np.hypot(d[0], d[1]):.2f} m from the {site} estimate")  # printed
    # ---- check 4: residuals per constellation ----
    print("check 4: post-fit residuals of the GPS+Galileo solutions, per constellation (absolute values)")  # heading
    rs["abs_res"] = rs.res_m.abs()  # absolute residuals
    for letter, name in (("G", "GPS"), ("E", "Galileo")):  # for loop over the constellations
        a = rs[rs.sv.str[0] == letter]["abs_res"]  # its absolute residuals
        add("4 residuals", name, "median_abs_m", float(a.median()))  # recorded
        add("4 residuals", name, "p95_abs_m", float(a.quantile(0.95)))  # recorded
        add("4 residuals", name, "n", len(a))  # recorded
        print(f"  {name:8s} n={len(a):6d} median {a.median():.2f} m, 95th percentile {a.quantile(0.95):.2f} m")  # printed
    for rx in RECEIVERS:  # for loop over the receivers
        a = rs[rs.receiver == rx]["abs_res"]  # that receiver's absolute residuals
        add("4 residuals", RECEIVERS[rx], "median_abs_m", float(a.median()))  # recorded
        add("4 residuals", RECEIVERS[rx], "p95_abs_m", float(a.quantile(0.95)))  # recorded
        print(f"  {RECEIVERS[rx]:24s} median {a.median():.2f} m, 95th percentile {a.quantile(0.95):.2f} m")  # printed
    rej = int(ep.n_rejected.sum())  # all rejected observations
    add("4 residuals", "all", "observations rejected (over 30 m)", rej)  # recorded
    add("4 residuals", "all", "epochs solved", len(ep))  # recorded
    print(f"  rejected {rej} observations in {len(ep)} solved epochs")  # printed
    # ---- BeiDou orbit test ----
    print("BeiDou C2I residuals with the GPS+Galileo site position fixed (one BeiDou clock per epoch)")  # heading
    brs = []  # per session x receiver residual tables
    for (test, rx), obs in obs_all.items():  # for loop over the observation tables
        site = SITE_OF[test]  # the test's site
        b = bds_test(orb, obs, site_x[site], kl)  # BeiDou residuals there
        b["test_code"], b["receiver"] = test, rx  # labels
        brs.append(b)  # kept
    brs = pd.concat(brs, ignore_index=True)  # all of them
    brs["abs_res"] = brs.res_m.abs()  # absolute residuals
    brs["geo"] = brs.sv.map(lambda s: int(s[1:]) in BDS_GEO)  # GEO satellites flagged
    # groupby('sv').agg: per BeiDou satellite, the number of residuals, their median and 95th percentile of the absolute value, median elevation and GEO flag
    per_sv = brs.groupby("sv").agg(n=("abs_res", "size"), median_abs_m=("abs_res", "median"), p95_abs_m=("abs_res", lambda v: float(np.percentile(v, 95))),
                                   median_el_deg=("el_deg", "median"), geo=("geo", "first")).reset_index()  # per satellite
    per_sv.round(3).to_csv(CHECK_DIR / "reference_bds_residuals.csv", index=False)  # written
    for flag, name in ((False, "MEO/IGSO"), (True, "GEO")):  # for loop over the two classes
        a = brs[brs.geo == flag]["abs_res"]  # that class's absolute residuals
        if len(a) == 0:  # if: none observed
            add("BeiDou test", name, "n", 0)  # recorded: the class was never observed
            print(f"  {name:9s} no observations above the {ELEV_MASK_DEG:.0f} degree mask at this latitude: not tested")  # printed
            continue  # nothing more to report
        add("BeiDou test", name, "n", len(a))  # recorded
        add("BeiDou test", name, "median_abs_m", float(a.median()))  # recorded
        add("BeiDou test", name, "p95_abs_m", float(a.quantile(0.95)))  # recorded
        print(f"  {name:9s} n={len(a):6d} median {a.median():.2f} m, 95th percentile {a.quantile(0.95):.2f} m")  # printed
    print(per_sv.round(2).to_string(index=False))  # the per-satellite table
    meo = brs[~brs.geo]["abs_res"]  # MEO and IGSO residuals
    ok_bds = float(meo.median()) <= BDS_OK_MEDIAN_M and float(meo.quantile(0.95)) <= BDS_OK_P95_M  # within a few metres?
    check("BeiDou MEO/IGSO C2I residuals within a few metres (median and 95th percentile)", f"<= {BDS_OK_MEDIAN_M} and {BDS_OK_P95_M} m", f"{meo.median():.2f} and {meo.quantile(0.95):.2f} m", ok_bds)  # the orbit code is validated for MEO/IGSO
    odd = per_sv[per_sv.p95_abs_m > BDS_OK_P95_M].sv.tolist()  # satellites beyond the limit individually
    print(f"  satellites with a 95th percentile above {BDS_OK_P95_M} m: {odd if odd else 'none'}")  # printed
    pd.DataFrame(chk).to_csv(PROC / "preprocessing" / "reference_position_checks.csv", index=False, float_format="%.4f")  # written
    plot_reference_positions(ep, pos, site_x, pvt_med)  # the figure
    heights_plot(ep)  # the check plot
    check("every solved epoch used at least five satellites", ">= 5", int((ep.n_gps + ep.n_gal).min()), int((ep.n_gps + ep.n_gal).min()) >= 5)  # basic sanity of the geometry
    check("every site has an estimate", 3, len(site_x), len(site_x) == 3)  # the three sites
    write_rounded(ep.drop(columns=["period"]), PROC / "preprocessing" / "reference_position_epochs.csv")  # written again after the checks (unchanged)


def plot_reference_positions(ep: pd.DataFrame, pos: pd.DataFrame, site_x: dict, pvt_med: dict) -> None:  # function: the figure, horizontal scatter per site
    import matplotlib  # the plotting library
    matplotlib.use("Agg")  # file-only backend
    import matplotlib.pyplot as plt  # pyplot
    from helpers import save_fig  # byte-stable saving
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.9))  # one panel per site
    for ax, (site, xc) in zip(axes, site_x.items()):  # for loop over the sites
        lat, lon, h = ecef_to_llh(*xc)  # the site estimate's geodetic position
        A = enu_axes(lat, lon)  # its east, north, up axes
        allx = ep[ep.site == site]  # every solution at this site
        off = np.einsum("ij,kj->ik", np.atleast_2d(np.stack([allx.x - xc[0], allx.y - xc[1], allx.z - xc[2]], axis=1)), A)  # their offsets from the estimate (einsum: the same matrix product without the BLAS routine that warned spuriously)
        rad = np.hypot(off[:, 0], off[:, 1])  # horizontal distance of each from the estimate
        lim = max(2.0, 1.5 * float(np.percentile(rad, 98)))  # frame half-width: 1.5 times the 98th percentile, at least 2 m
        n_out = int((np.maximum(np.abs(off[:, 0]), np.abs(off[:, 1])) > lim).sum())  # solutions outside the frame
        for rx, col in COLORS.items():  # for loop over the receivers
            g = ep[(ep.site == site) & (ep.receiver == rx)]  # that receiver's solutions at this site
            if len(g) == 0:  # if: none
                continue  # skip
            d = np.einsum("ij,kj->ik", np.atleast_2d(np.stack([g.x - xc[0], g.y - xc[1], g.z - xc[2]], axis=1)), A)  # offsets from the site estimate (einsum: the same matrix product without the BLAS routine that warned spuriously)
            ax.scatter(d[:, 0], d[:, 1], s=4, color=col, alpha=0.35, linewidths=0, label=RECEIVERS[rx])  # the 1 Hz solutions
            for test in sorted(g.test_code.unique()):  # for loop over the sessions at the site
                gg = g[g.test_code == test]  # one session
                dm = np.einsum("ij,kj->ik", np.atleast_2d(np.array([gg.x.median() - xc[0], gg.y.median() - xc[1], gg.z.median() - xc[2]])), A)[0]  # its median offset (einsum: the same matrix product without the BLAS routine that warned spuriously)
                ax.scatter(dm[0], dm[1], s=55, marker=MARKERS.get(test, "o"), color=col, edgecolors="black", linewidths=0.9, zorder=3)  # the session median
        for test in SITES[site]:  # for loop over the site's sessions
            if test in pvt_med:  # if: the X5's own PVT median exists
                lp, lo, hp, _ = pvt_med[test]  # its position
                dp = np.einsum("ij,kj->ik", np.atleast_2d((llh_to_ecef(lp, lo, hp) - xc)), A)[0]  # offset from the site estimate (einsum: the same matrix product without the BLAS routine that warned spuriously)
                ax.scatter(dp[0], dp[1], s=70, marker="D", color="#D55E00", edgecolors="black", linewidths=0.9, zorder=4)  # the X5 PVT median
        ax.plot(0, 0, "k+", ms=10, zorder=5)  # the site estimate itself
        ax.set_xlim(-lim, lim)  # the frame, east
        ax.set_ylim(-lim, lim)  # the frame, north
        ax.set_aspect("equal")  # equal metre scales
        if n_out:  # if: some solutions fall outside the frame
            ax.text(0.03, 0.03, f"{n_out} solutions outside the frame\n(largest {float(rad.max()):.0f} m)", transform=ax.transAxes, fontsize=7, va="bottom")  # say so on the panel
        ax.grid(color="#999999", alpha=0.3, lw=0.5)  # faint grid
        ax.set_xlabel("east of the site estimate (m)")  # axis label
        ax.set_title(site, fontsize=9)  # panel title
    axes[0].set_ylabel("north of the site estimate (m)")  # axis label
    axes[0].legend(fontsize=7, loc="upper left", markerscale=3)  # legend of receivers
    fig.tight_layout()  # spacing
    save_fig(fig, PROJECT / "Graph" / "reference_positions.png")  # written, byte-stable


def heights_plot(ep: pd.DataFrame) -> None:  # function: check plot, ellipsoidal height per session and receiver
    import matplotlib  # the plotting library
    matplotlib.use("Agg")  # file-only backend
    import matplotlib.pyplot as plt  # pyplot
    from helpers import save_fig  # byte-stable saving
    fig, ax = plt.subplots(figsize=(8.0, 3.6))  # one panel
    x = 0  # running x position
    ticks, labels = [], []  # axis labels
    for test in ["2.6.12", "2.6.2", "2.6.10", "2.6.4"]:  # for loop over the sessions
        for rx, col in COLORS.items():  # for loop over the receivers
            g = ep[(ep.test_code == test) & (ep.receiver == rx)]  # the session and receiver
            if len(g) == 0:  # if: none
                continue  # skip
            lo, md, hi = np.percentile(g.h, [5, 50, 95])  # height percentiles
            ax.errorbar(x, md, yerr=[[md - lo], [hi - md]], fmt="o", color=col, capsize=3)  # median with the 5-95 range
            x += 1  # next slot
        ticks.append(x - 1.5)  # the session's label position
        labels.append(test)  # its label
        x += 1  # a gap between sessions
    ax.set_xticks(ticks)  # session labels
    ax.set_xticklabels(labels)  # their text
    ax.set_ylabel("ellipsoidal height (m), median and 5-95 %")  # axis label
    ax.grid(axis="y", color="#999999", alpha=0.3, lw=0.5)  # faint grid
    fig.tight_layout()  # spacing
    save_fig(fig, CHECK_DIR / "reference_heights.png")  # written


def main_intervals() -> tuple:  # function: the interval part of this script; returns the results and the table
    sess, margins = load_margins()  # sessions and combined margins
    print(f"{len(margins)} combined margins read for {len({m['test'] for m in margins})} sessions with a margin")  # basic count
    res = build_intervals(sess, margins)  # apply the rules to every margin and receiver
    tbl = interval_rows(res)  # the output table
    tbl.to_csv(PROC / "preprocessing" / "reference_clean_intervals.csv", index=False)  # written
    rec = reconcile(res)  # the reconciliation with the receiver-status script
    rec.to_csv(CHECK_DIR / "reference_interval_reconciliation.csv", index=False)  # written
    print("reconciliation with the receiver-status script (X5 view):")  # heading
    print(rec.to_string(index=False))  # the table
    check("every margin of the receiver-status script is listed", len(margins), len(rec), len(margins) == len(rec))  # nothing lost
    # self-check: where no rule cuts a margin, its length equals the receiver-status script's margin minutes (rounded to a millisecond-level tolerance)
    plain = rec[rec.x5_outcome != "the possibly contaminated 2.6.12 margin: the margin before 15:45:12 is possibly contaminated by an unexplained step in the gains"]  # all but the possibly contaminated 2.6.12 margin
    ok = bool((np.abs(plain.margin_min - (plain.margin.map(lambda s: (utc(s.split('-')[1]) - utc(s.split('-')[0])).total_seconds() / 60.0))) < 0.01).all())  # lengths agree
    check("margin lengths equal the receiver-status script's margin_min (within 0.01 min)", "all", "all" if ok else "some differ", ok)  # the margin table is consistent
    print("kept minutes after rules and guards, by site and receiver:")  # heading
    k = tbl[tbl.minutes > 0].groupby(["site", "receiver"]).minutes.sum().round(3)  # minutes per site and receiver
    for site in SITES:  # for loop over the sites
        for rx in RECEIVERS:  # for loop over the receivers
            mins = float(k.get((site, rx), 0.0))  # the total, 0 if none
            weak = " WEAK (under two minutes)" if mins < MIN_MINUTES else ""  # reported, never padded
            print(f"  {site:16s} {RECEIVERS[rx]:24s} {mins:7.3f} min{weak}")  # one line per site and receiver
    print("kept minutes by session and receiver:")  # heading
    kt = tbl[tbl.minutes > 0].groupby(["test_code", "receiver"]).minutes.sum().round(3)  # minutes per session and receiver
    for t in ["2.6.12", "2.6.2", "2.6.5", "2.6.10", "2.6.4", "2.6.11"]:  # for loop over the sessions
        # one line per session; a session under two minutes on a receiver is marked weak and a session with none is marked excluded
        print("  " + t + "  " + "  ".join(f"{RECEIVERS[rx]}: {float(kt.get((t, rx), 0.0)):.3f}" + (" WEAK" if 0 < float(kt.get((t, rx), 0.0)) < MIN_MINUTES else (" none" if float(kt.get((t, rx), 0.0)) == 0 else "")) for rx in RECEIVERS))
    q = quiet_check(res, margins)  # readings in the kept intervals that the receiver-status script's criteria call disturbed
    nbad = sum(x["bad"] for x in q)  # all of them
    print(f"quietness check: {sum(x['n'] for x in q):,} status readings in {len(q)} kept intervals, {nbad} disturbed")  # the count
    check("no kept interval contains a reading the receiver-status script's criteria call disturbed", 0, nbad, nbad == 0)  # the intervals are quiet
    return res, tbl  # handed to the position part


def main() -> None:  # function: the reference positions, intervals then positions
    res, tbl = main_intervals()  # the clean intervals
    main_positions(res, tbl)  # the positions and their checks
    pd.DataFrame(CHECKS).to_csv(CHECK_DIR / "reference_position_selfchecks.csv", index=False)  # every self-check of both parts
    print(f"checks: {sum(c['ok'] for c in CHECKS)} of {len(CHECKS)} ok")  # summary line


if __name__ == "__main__":  # if: true only when the file is run directly
    main()  # runs everything
    exit_if_failed(FAILED, "2_preprocessing/03_reference_positions.py")  # the helpers module: status 1 when any self-check failed, so a chained run stops
