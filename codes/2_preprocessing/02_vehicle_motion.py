#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
Vehicle motion: when did the recording vehicle move, and when did it stand still?

Reads   Data/Day1_14092026/<session>/*_OBD-II_CSV_01.csv   the vehicle's own speed (SPEED rows), all six sessions
        Data/Day1_14092026/<2.6.5>/*_SBF_02.sbf            mosaic-X5 PVTGeodetic velocity and position (second witness)
        Data/processed/ground_truth/session_distance_to_sender.csv       the three recording sites as the handset script measured them
Writes  Data/processed/preprocessing/vehicle_motion.csv                   one row per moving or stationary segment, every session
        Data/processed/preprocessing/vehicle_motion_obd_sessions.csv      one row per session: SPEED rows, gaps, segments (the motion table)
        Data/processed/preprocessing/vehicle_motion_x5_params.csv         the X5 witness's settings and measured quantities (noise floor, threshold ...)
        Data/processed/preprocessing/vehicle_motion_x5_check.csv          each OBD stop and start beside the receiver's own
        Data/processed/preprocessing/vehicle_motion_x5_events.csv         each X5-only movement while the wheels read zero, classified
        Data/processed/preprocessing/vehicle_motion_stops.csv             where each 2.6.5 stationary segment was (X5 median position)
        Graph/vehicle_motion_265.png               the figure: OBD speed and X5 speed, 2.6.5
        Data/processed/checks/obd_speed_all_sessions.png         check plot: OBD speed of all six sessions on one axis
        Data/processed/checks/motion_checks.csv                  every self-check and its outcome

The mosaic-X5 is used as a witness only before its own front-end gain onset in 2.6.5 (12:50:59, the receiver-status script): from then
on the spoofer's passage disturbs its solution (no PVT at all from 12:51:01.6 to 12:51:59), so events there are
classified "inside the spoofer passage" and not judged as motion.

Two speed numbers are used for the mosaic-X5 and they are different things. The NOISE FLOOR (99th percentile of its
horizontal speed over the long stationary stretch 12:35:30 to the X5 gain onset, about 0.05 km/h) says how much the receiver's
velocity wanders at rest. The COMPARISON THRESHOLD (1 km/h) is what an X5 sample must exceed to count as motion when
its stops and starts are matched to the OBD's: the OBD reports whole km/h, so an OBD zero means "below about 1 km/h".

Verification and labelling evidence only: nothing here is a feature (exclusion rule). The OBD-II odometer and the
receiver's velocity are used to say WHEN the vehicle moved; the SBG IMU is not used (OBD plus receiver
velocity replace it, because the IMU would add a third, binary decode and no independent stop or start time).
"""

from __future__ import annotations  # future import: type hints kept as text (a convention across the project's scripts)

import csv  # standard-library CSV reader: the independent count of speed rows uses it instead of pandas
import importlib.util  # standard-library tools for loading a Python file by its path as a module
import os  # operating-system access: MPLCONFIGDIR is set before matplotlib loads
import random  # standard-library random; seeded below although nothing here draws from it
import struct  # standard-library binary decoding: little-endian floats out of a PVTGeodetic payload
import sys  # standard-library system access: sys.modules registers a loaded module, sys.exit stops the script
from pathlib import Path  # Path objects for building file paths

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
os.environ["MPLCONFIGDIR"] = str(PROJECT / "codes" / ".mplconfig")  # matplotlib's cache inside the project folder

import numpy as np  # NumPy: arrays of times and speeds
import pandas as pd  # pandas: reads the CSVs and formats timestamps
import matplotlib  # the plotting library; imported first so its backend can be set before pyplot loads
matplotlib.use("Agg")  # file-only backend: no window is opened
import matplotlib.dates as mdates  # date helpers: datetime -> plot number
import matplotlib.pyplot as plt  # pyplot: the plotting interface

sys.path.insert(0, str(PROJECT / "codes" / "common"))  # puts codes/ first on the import path so the helpers module is found
from helpers import exit_if_failed, C_GREY, gps_to_utc, haversine_m, host_clock, host_times, save_fig, time_axis  # shared tools; exit_if_failed ends the run on a failed check

SEED = 20260914  # integer constant: the project's fixed random seed
random.seed(SEED)  # seeds Python's generator (nothing here draws from it)
np.random.seed(SEED)  # seeds NumPy's generator likewise

DAY = PROJECT / "Data" / "Day1_14092026"  # the raw data folder (read only)
PROC = PROJECT / "Data" / "processed"  # where the CSVs are written
GRAPH = PROJECT / "Graph"  # the figure folder
CHECK_DIR = PROJECT / "Data" / "processed" / "checks"  # the check outputs

MIN_STOP_S = 10.0  # float constant: a stationary segment needs at least 10 s of consecutive zero speed
TOL_S = 5.0  # float constant: an OBD stop or start and the receiver's must agree within 5 s
CMP_KMH = 1.0  # float constant: the X5 comparison threshold, the OBD's resolution (whole km/h)
EVENT_FACTOR = 5.0  # float constant: an X5-only event is X5 speed above this many noise floors while the wheels read zero
EVENT_MERGE_S = 2.0  # float constant: X5 samples above the event level closer together than this belong to one event
EVENT_MIN_SAMPLES = 2  # integer constant: an event needs at least this many X5 samples above the event level
EVENT_PAD_S = 1.0  # float constant: the integration and position windows reach this far beyond an event's first and last sample
POS_WINDOW_S = 10.0  # float constant: median position taken over this long before and after an event
POS_MIN_FIXES = 50  # integer constant: fewest position fixes (of about 100) for a before or after median to count
CREEP_MIN_M = 0.5  # float constant: both displacements must exceed this to call an event real creep
CREEP_AGREE = 0.5  # float constant: the two displacement vectors agree if they differ by less than this share of the larger
T265 = "2.6.5"  # the one session in which the vehicle moved
X5_READ_END = pd.Timestamp("2026-09-14 12:52:00", tz="UTC")  # the X5 is read to this instant; events after the onset are only classified
# the mosaic-X5's front-end gain onset in 2.6.5 (the receiver-status script, ground_truth/ta3_schedule_confirmation.csv): the spoofer's passage begins here
_ta3 = pd.read_csv(PROJECT / "Data" / "processed" / "ground_truth" / "ta3_schedule_confirmation.csv")  # the receiver-status script's onset and recovery table
_row = _ta3[(_ta3.test_code == "2.6.5") & (_ta3.receiver == "Septentrio_02")].iloc[0]  # the X5 row of 2.6.5
ONSET = pd.Timestamp("2026-09-14 " + str(_row.onset_utc), tz="UTC")  # 12:50:59: the X5 is used only before this instant
RECOVERY = pd.Timestamp("2026-09-14 " + str(_row.recovery_utc), tz="UTC")  # 12:52:21: the X5 gain recovery, for the record
LONG_STILL = (pd.Timestamp("2026-09-14 12:35:30", tz="UTC"), ONSET)  # noise stretch: 12:35:30 to the X5 gain onset
MOVING_SAMPLES_265 = 1351  # integer constant: moving samples in 2.6.5 as counted beforehand by hand
SITE_RADIUS_M = 100.0  # float constant: within this distance of a known site counts as being at that site
EARTH_R = 6371008.8  # float constant (m): the mean Earth radius, the same value the helpers module.haversine_m uses


def load_module(name: str, filename: str):  # function: loads codes/<filename> as a module (names start with a digit)
    spec = importlib.util.spec_from_file_location(name, PROJECT / "codes" / filename)  # where to load from
    mod = importlib.util.module_from_spec(spec)  # an empty module object from that spec
    sys.modules[name] = mod  # registers it under name, as a normal import would
    spec.loader.exec_module(mod)  # runs the file's top-level code (its main() is guarded and not run)
    return mod  # hands back the loaded module


S02 = load_module("handset_check", "1_ground_truth/02_handset_check.py")  # the handset script's script: the Stave landmark constant
S03 = load_module("receiver_status", "1_ground_truth/03_receiver_status.py")  # the receiver-status script's script: the checked SBF block reader

CHECKS: list = []  # list of (check, expected, found, ok) rows, written at the end
FAILED: list = []  # list: names of the self-checks that failed; the script ends the run with status 1 when it is not empty


def check(name: str, expected, found, ok: bool) -> None:  # function: records one self-check and prints it
    CHECKS.append({"check": name, "expected": str(expected), "found": str(found), "ok": bool(ok)})  # one row
    if ok is not None and not bool(ok):  # if: a pass/fail check (None marks a reported number) that did not pass
        FAILED.append(name)  # its name kept for the exit at the end of the run
    print(f"  [{'ok' if ok else 'FAIL'}] {name}: expected {expected}, found {found}")  # printed for the log


def stamp(ns) -> str:  # function: integer nanoseconds since 1970 -> "YYYY-MM-DD HH:MM:SS.ffffff" UTC text
    return pd.Timestamp(int(ns), tz="UTC").strftime("%Y-%m-%d %H:%M:%S.%f")  # microsecond text, byte-stable


def clock(ns) -> str:  # function: integer nanoseconds since 1970 -> "HH:MM:SS.f" UTC text, for the log
    return stamp(ns)[11:21]  # the time of day with a tenth of a second


def sessions() -> list:  # function: (test_code, folder) for the six sessions, in time order
    out = []  # empty list of results
    for folder in sorted(DAY.glob("*_T_*")):  # for loop over the session folders; names start with the start time
        out.append((folder.name.split("_T_")[1], folder))  # test code is the text after "_T_"
    return out  # six tuples


def load_speed(folder: Path) -> pd.DataFrame:  # function: every SPEED row of a session's OBD file, UTC time in ns
    f = next(folder.glob("*_OBD-II_CSV_01.csv"))  # the session's OBD-II file (exactly one)
    d = pd.read_csv(f, usecols=["signal", "value", "valid_value", "unix_time_ns"])  # only the four columns needed
    d = d[d.signal == "SPEED"].copy()  # boolean mask: keeps the SPEED rows, drops RPM and the rest
    d["t_ns"] = d.unix_time_ns.astype(np.int64)  # UTC nanoseconds since 1970 (the specification: never system_time_iso8601)
    d["valid"] = d.valid_value.astype(str) == "True"  # the logger's own validity flag as a real boolean
    d["kmh"] = pd.to_numeric(d.value, errors="coerce")  # speed in km/h; an unreadable value becomes NaN
    d.loc[~d.valid, "kmh"] = np.nan  # an invalid reading carries no speed: it is missing, not zero
    return d.sort_values("t_ns").reset_index(drop=True)[["t_ns", "valid", "kmh"]]  # time order, three columns


def run_bounds(flag: np.ndarray):  # function: start and end index of each run of equal values in a boolean array
    change = np.r_[True, flag[1:] != flag[:-1]]  # True wherever the value differs from the previous sample
    starts = np.flatnonzero(change)  # index where each run begins
    ends = np.r_[starts[1:], len(flag)] - 1  # index where each run ends (last sample of the run)
    return starts, ends  # two arrays of equal length


def stationary_state(t_ns: np.ndarray, still: np.ndarray) -> np.ndarray:  # function: True where a >= 10 s stop holds
    state = np.zeros(len(still), dtype=bool)  # starts all moving; a stop is switched on below
    starts, ends = run_bounds(still)  # the runs of consecutive still samples (and of moving samples)
    for s, e in zip(starts, ends):  # for loop over every run
        # a run counts as a stop only if it is still AND lasts at least MIN_STOP_S from its first to its last sample
        if still[s] and (t_ns[e] - t_ns[s]) / 1e9 >= MIN_STOP_S:
            state[s:e + 1] = True  # slice assignment: marks the whole run stationary
    return state  # shorter zero runs stay inside the moving state


def base_segments(t_ns: np.ndarray, stationary: np.ndarray) -> list:  # function: [{t0, t1, state, source}] tiling a series
    starts, ends = run_bounds(stationary)  # runs of equal state
    out = []  # one dict per segment
    for k, (s, e) in enumerate(zip(starts, ends)):  # for loop with a counter over the runs
        t1 = t_ns[ends[k] + 1] if k + 1 < len(starts) else t_ns[e]  # ends where the next starts (last: its last sample)
        # list.append adds a dict literal: one segment (start, end, state, source) per run of equal state
        out.append({"t0": int(t_ns[s]), "t1": int(t1), "state": "stationary" if stationary[s] else "moving",
                    "source": "OBD", "split_m": np.nan})  # a segment starts at its first sample
    return out  # list of dicts, in time order


def transitions(segs: list) -> list:  # function: (kind, t_ns) for every stop and start between consecutive segments
    out = []  # empty list of transitions
    for a, b in zip(segs[:-1], segs[1:]):  # zip of the list with itself shifted by one: each neighbouring pair
        if a["state"] != b["state"]:  # if: the state changes (a split inside a stop is not a transition)
            out.append(("stop" if b["state"] == "stationary" else "start", b["t0"]))  # moving->still is a stop
    return out  # list of tuples


def seg_stats(seg: dict, last: bool, v_t: np.ndarray, v_kmh: np.ndarray, inv_t: np.ndarray) -> dict:  # function: one segment's numbers
    hi = (v_t <= seg["t1"]) if last else (v_t < seg["t1"])  # the final segment includes its closing sample
    sel = (v_t >= seg["t0"]) & hi  # boolean mask: the valid speed samples inside the segment
    ih = (inv_t <= seg["t1"]) if last else (inv_t < seg["t1"])  # the same rule for the invalid rows
    n_inv = int(((inv_t >= seg["t0"]) & ih).sum())  # invalid SPEED rows inside the segment
    win = v_t[(v_t >= seg["t0"]) & (v_t <= seg["t1"])]  # samples from the start to the next segment's first sample
    gap = float(np.max(np.diff(win)) / 1e9) if len(win) > 1 else 0.0  # longest silence between valid samples, s
    # return a dict literal: the five numbers one segment contributes to the table and the checks
    return {"max_speed_kmh": float(np.nanmax(v_kmh[sel])), "n_samples": int(sel.sum()), "n_invalid": n_inv,
            "n_moving": int((v_kmh[sel] > 0).sum()), "obd_gap_s": round(gap, 3)}  # numbers for the table and checks


def read_x5(folder: Path) -> pd.DataFrame:  # function: mosaic-X5 PVTGeodetic position and velocity to 12:52:00, receiver time
    f = next(folder.glob("*_SBF_02.sbf"))  # the mosaic-X5 file of the session
    integ = S03.Integrity()  # the reader's integrity container (its CRC counts are not needed here)
    rows = []  # one tuple per block
    # generator from the receiver-status script's checked reader: yields only the PVTGeodetic blocks (4007), CRC-verified
    for blk, rev, tow, wnc, p, end in S03.sbf_blocks(f, integ, {4007}):
        t = gps_to_utc(tow, wnc)  # receiver time stamp -> UTC, or None for a Do-Not-Use time
        if t is None or len(p) < 38:  # if: no valid time, or a payload too short to reach Vn and Ve
            continue  # skipped
        mode, err = p[0] & 0x0F, p[1]  # PVT solution type (bits 0-3) and error code
        lat, lon = struct.unpack_from("<dd", p, 2)  # "<dd": two little-endian 8-byte floats, radians, bytes 2-17
        vn, ve = struct.unpack_from("<ff", p, 30)  # "<ff": two little-endian 4-byte floats (m/s) at bytes 30-37: Vn, Ve
        pos_ok = mode not in (0, 3) and err == 0 and lat > -1e10 and lon > -1e10  # computed solution with a position
        vel_ok = pos_ok and vn > -1e10 and ve > -1e10  # Septentrio Do-Not-Use is -2e10: those values are missing
        rows.append((pd.Timestamp(t).value, end, mode, err, pos_ok, vel_ok,  # time in ns, end offset, status flags
                     np.degrees(lat) if pos_ok else np.nan, np.degrees(lon) if pos_ok else np.nan,  # degrees or missing
                     float(vn) if vel_ok else np.nan, float(ve) if vel_ok else np.nan,  # velocity components, m/s
                     float(np.hypot(vn, ve)) if vel_ok else np.nan))  # horizontal speed in m/s or missing
    x = pd.DataFrame(rows, columns=["t_ns", "end", "mode", "err", "pos_ok", "vel_ok", "lat", "lon", "vn", "ve", "hspeed_ms"])  # table
    x = x[x.t_ns <= X5_READ_END.value].sort_values("t_ns").reset_index(drop=True)  # pre-spoofing only, time order
    return x  # the X5 PVT rows to 12:52:00 (the caller cuts them at the X5 gain onset where required)


def enu_m(lat0: float, lon0: float, lat, lon):  # function: local east and north offsets in metres from a reference point
    north = np.radians(np.asarray(lat) - lat0) * EARTH_R  # north offset: latitude difference in radians times the radius
    east = np.radians(np.asarray(lon) - lon0) * EARTH_R * np.cos(np.radians(lat0))  # east offset, shrunk by cos(latitude)
    return east, north  # two arrays (or scalars) in metres


def independent_counts(folder: Path) -> tuple:  # function: (all SPEED rows, valid ones, valid ones above zero) by the csv module
    n_all = n_valid = n_move = 0  # three running counts
    with open(next(folder.glob("*_OBD-II_CSV_01.csv")), newline="") as fh:  # opens the OBD file as plain text
        for r in csv.DictReader(fh):  # for loop: one dict per CSV row, no pandas involved
            if r["signal"] != "SPEED":  # if: some other signal (RPM and so on)
                continue  # not counted
            n_all += 1  # every SPEED row
            if r["valid_value"] == "True":  # if: the logger flagged the reading valid
                n_valid += 1  # a valid reading
                if float(r["value"]) > 0:  # if: the vehicle was moving
                    n_move += 1  # a moving reading
    return n_all, n_valid, n_move  # the independent counts


def find_events(xv: pd.DataFrame, x: pd.DataFrame, still_spans: list, floor_kmh: float, tr_times: list) -> list:  # function: X5-only movements while OBD reads zero
    lvl = EVENT_FACTOR * floor_kmh  # the event level in km/h: five noise floors
    t = xv.t_ns.to_numpy(np.int64)  # X5 sample times with a velocity
    kmh = xv.hspeed_ms.to_numpy(float) * 3.6  # their horizontal speed in km/h
    inside = np.zeros(len(t), dtype=bool)  # True where an X5 sample falls inside an OBD-stationary span
    for a, b in still_spans:  # for loop over the OBD stationary spans
        inside |= (t >= a) & (t <= b)  # in-place OR: marks the samples within this span
    idx = np.flatnonzero(inside & (kmh > lvl))  # indices of samples above the event level while the wheels read zero
    groups, cur = [], []  # finished groups, and the group being built
    for i in idx:  # for loop over those samples in time order
        if cur and (t[i] - t[cur[-1]]) / 1e9 > EVENT_MERGE_S:  # if: a gap longer than the merge time
            groups.append(cur)  # closes the current group
            cur = []  # starts a new one
        cur.append(i)  # adds the sample to the current group
    if cur:  # if: a group is still open at the end
        groups.append(cur)  # closes it
    events = []  # classified events
    for g in groups:  # for loop over the groups
        if len(g) < EVENT_MIN_SAMPLES:  # if: a single stray sample
            continue  # not an event
        ta, tb = int(t[g[0]]), int(t[g[-1]])  # first and last sample of the event
        w0, w1 = ta - int(EVENT_PAD_S * 1e9), tb + int(EVENT_PAD_S * 1e9)  # the padded window
        win = xv[(xv.t_ns >= w0) & (xv.t_ns <= w1)]  # X5 velocity samples across the event
        tt = win.t_ns.to_numpy(np.float64) / 1e9  # their times in seconds
        # trapezoid rule by hand: sum of (average of neighbouring velocities) x (time step) = displacement in metres
        ie = float(np.sum(0.5 * (win.ve.to_numpy()[1:] + win.ve.to_numpy()[:-1]) * np.diff(tt)))  # east, m
        inn = float(np.sum(0.5 * (win.vn.to_numpy()[1:] + win.vn.to_numpy()[:-1]) * np.diff(tt)))  # north, m
        pb = x[x.pos_ok & (x.t_ns >= w0 - int(POS_WINDOW_S * 1e9)) & (x.t_ns < w0)]  # fixes in the 10 s before the window
        pa = x[x.pos_ok & (x.t_ns > w1) & (x.t_ns <= w1 + int(POS_WINDOW_S * 1e9))]  # fixes in the 10 s after the window
        # dict literal: the event's identity and numbers so far (times in ns, counts, peak, integrated displacement)
        row = {"start_utc": stamp(ta), "end_utc": stamp(tb), "t0": ta, "t1": tb, "duration_s": round((tb - ta) / 1e9, 2),
               # dict entry: sample count, peak X5 speed and the integrated east displacement of the event
               "n_samples": len(g), "max_x5_kmh": round(float(kmh[g].max()), 3), "int_east_m": round(ie, 3),
               "int_north_m": round(inn, 3), "int_m": round(float(np.hypot(ie, inn)), 3),  # dict entry: integrated north displacement and the total integrated displacement of the event
               "n_fix_before": len(pb), "n_fix_after": len(pa),  # dict entry: how many position fixes lie in the windows before and after the event
               # True if an OBD stop or start lies within the event's windows: then the before/after positions include the drive
               "adjacent_to_obd_transition": any(w0 - POS_WINDOW_S * 1e9 <= tt <= w1 + POS_WINDOW_S * 1e9 for tt in tr_times)}  # flag
        if ta >= ONSET.value:  # if: the event begins after the X5 gain onset, inside the spoofer's passage
            row.update({"pos_east_m": np.nan, "pos_north_m": np.nan, "pos_m": np.nan})  # the receiver's motion is not judged here
            row["class"] = "inside the spoofer passage"  # not motion, not noise: the X5 solution itself is disturbed
        elif len(pb) >= POS_MIN_FIXES and len(pa) >= POS_MIN_FIXES:  # elif: enough fixes on both sides for a median
            la0, lo0 = float(pb.lat.median()), float(pb.lon.median())  # median position before
            e1, n1 = enu_m(la0, lo0, float(pa.lat.median()), float(pa.lon.median()))  # median position after, relative to before
            # dict.update merges the position-change keys into the event row: metres east, north and total between the two median positions
            row.update({"pos_east_m": round(float(e1), 3), "pos_north_m": round(float(n1), 3),
                        "pos_m": round(float(np.hypot(e1, n1)), 3)})  # the position change, metres east and north
            diff = float(np.hypot(ie - e1, inn - n1))  # the size of the disagreement between the two vectors
            big = max(row["int_m"], row["pos_m"])  # the larger of the two displacements
            creep = row["int_m"] > CREEP_MIN_M and row["pos_m"] > CREEP_MIN_M and diff <= CREEP_AGREE * big  # the rule
            row["class"] = "real creep below OBD resolution" if creep else "receiver velocity noise"  # classification
        else:  # else: too few position fixes to measure the displacement
            row.update({"pos_east_m": np.nan, "pos_north_m": np.nan, "pos_m": np.nan})  # no position change available
            row["class"] = "receiver velocity noise"  # cannot be shown to be creep, so the segment stays whole
        events.append(row)  # saved
    return events  # list of dicts


def main() -> None:  # function: the whole analysis of this script
    sess = sessions()  # the six sessions
    prm = []  # list of (name, value, unit, description): the X5 witness's settings and measured numbers, written at the end
    check("six sessions found", 6, len(sess), len(sess) == 6)  # the data folder holds what the project expects
    obd = {}  # per test code: valid-speed table, invalid times, base segments
    print("OBD segmentation")  # heading in the log
    for code, folder in sess:  # for loop over the six sessions
        d = load_speed(folder)  # all SPEED rows
        v = d[d.valid].reset_index(drop=True)  # the valid ones: the speed series
        t, kmh = v.t_ns.to_numpy(np.int64), v.kmh.to_numpy(float)  # arrays of time (ns) and speed (km/h)
        segs = base_segments(t, stationary_state(t, kmh == 0))  # OBD-defined segments (>= 10 s of zeros = stop)
        inv_t = d.t_ns[~d.valid].to_numpy(np.int64)  # times of the invalid SPEED rows
        obd[code] = {"d": d, "v": v, "t": t, "kmh": kmh, "inv": inv_t, "segs": segs}  # kept for later steps
        stats = [seg_stats(s, s is segs[-1], t, kmh, inv_t) for s in segs]  # list comprehension: numbers per segment
        # self-check: the durations add up to the span from the first to the last valid speed sample
        total = sum(s["t1"] - s["t0"] for s in segs) / 1e9
        # arithmetic on nanosecond integers: the record's span in seconds, for the tiling check
        span = (t[-1] - t[0]) / 1e9
        # check() records and prints one self-check: the segment durations must add up to the span to a microsecond
        check(f"{code}: segment durations sum to the speed-record span (s)", f"{span:.3f}", f"{total:.3f}", abs(total - span) < 1e-6)
        # self-check: the moving samples in the segments equal an independent csv-module count of rows with speed > 0
        n_seg = sum(s["n_moving"] for s in stats)
        n_allraw, n_validraw, n_raw = independent_counts(folder)  # counts from the raw file
        # self-check: the moving count from the segments against the independent csv-module count
        check(f"{code}: samples with speed > 0 (segments vs raw rows)", n_raw, n_seg, n_seg == n_raw)
        # self-check: pandas and the csv module must agree on the number of SPEED rows and valid SPEED rows
        check(f"{code}: SPEED rows and valid SPEED rows (pandas vs raw csv)", f"{n_allraw} and {n_validraw}", f"{len(d)} and {len(v)}",
              (len(d), len(v)) == (n_allraw, n_validraw))  # the two ways of reading the file agree
        # f-string log line per session: segments, maximum speed, median interval, longest gap and invalid rows
        print(f"  {code}: {len(segs)} OBD segment(s), max speed {np.nanmax(kmh):.0f} km/h, median dt {np.median(np.diff(t)) / 1e9:.4f} s, "
              f"max gap {np.max(np.diff(t)) / 1e9:.3f} s, invalid {int((~d.valid).sum())}")  # one line per session
    # comprehension inside sum(): adds the moving samples of every 2.6.5 segment, to compare with the plan count of 1,351
    s5 = sum(s["n_moving"] for s in [seg_stats(q, q is obd[T265]["segs"][-1], obd[T265]["t"], obd[T265]["kmh"], obd[T265]["inv"])
                                     for q in obd[T265]["segs"]])  # 2.6.5 moving samples counted from the segments
    check(f"{T265}: moving samples as counted beforehand", MOVING_SAMPLES_265, s5, s5 == MOVING_SAMPLES_265)  # the count of 1,351 made beforehand
    still_codes = [c for c, _ in sess if c != T265]  # the sessions expected to be entirely stationary
    # all() over a generator: true only if each of the five sessions is one stationary segment with maximum speed 0
    ok_still = all(len(obd[c]["segs"]) == 1 and obd[c]["segs"][0]["state"] == "stationary" and np.nanmax(obd[c]["kmh"]) == 0
                   for c in still_codes)  # one stationary segment, maximum speed 0
    # self-check: how many of the five sessions came out as one segment
    check("five sessions entirely stationary", "5 sessions, one stationary segment each", sum(len(obd[c]["segs"]) == 1 for c in still_codes), ok_still)

    segs5 = obd[T265]["segs"]  # the 2.6.5 OBD segments
    v5 = obd[T265]["v"]  # the 2.6.5 valid-speed table
    print("2.6.5 OBD segments (UTC):")  # heading in the log
    for i, s in enumerate(segs5):  # for loop over them
        print(f"  {i + 1} {s['state']:10s} {stamp(s['t0'])[11:]} -> {stamp(s['t1'])[11:]}  {(s['t1'] - s['t0']) / 1e9:8.2f} s")  # row
    print(f"  last moving sample {stamp(v5.t_ns[v5.kmh > 0].iloc[-1])[11:]}")  # the instant quoted in the plan (12:34:56.95)
    prm.append(("obd_last_moving_utc", stamp(v5.t_ns[v5.kmh > 0].iloc[-1])[11:22], "UTC", "time of the last OBD sample with speed above zero in 2.6.5"))  # for the report

    print(f"mosaic-X5 second witness (2.6.5, before the X5 gain onset {ONSET:%H:%M:%S}; read to 12:52:00)")  # heading in the log
    folder5 = dict(sess)[T265]  # the 2.6.5 session folder
    x_all = read_x5(folder5)  # X5 PVT rows to 12:52:00
    x = x_all[x_all.t_ns < ONSET.value].reset_index(drop=True)  # the rows before the X5 gain onset: the only ones used as a witness
    xv = x[x.vel_ok].reset_index(drop=True)  # the rows with a usable velocity
    xv_all = x_all[x_all.vel_ok].reset_index(drop=True)  # the same to 12:52:00, for classifying events inside the spoofer passage
    # list += list appends the X5 witness's settings (counts, interval, onset, recovery) to the parameter table
    prm += [("x5_pvt_blocks_before_onset", len(x), "blocks", "mosaic-X5 PVTGeodetic blocks before the X5 gain onset"),
            ("x5_pvt_blocks_with_velocity", len(xv), "blocks", "of those, blocks with a usable velocity"),  # list element: X5 blocks that carry a usable velocity
            ("x5_median_interval_s", round(float(np.median(np.diff(x.t_ns))) / 1e9, 3), "s", "median interval between PVT blocks"),  # list element: median spacing of the X5 PVT blocks in seconds
            # list element: X5 gain onset in 2.6.5, after which the X5 is no longer used as a witness
            ("x5_onset_utc", f"{ONSET:%H:%M:%S}", "UTC", "X5 front-end gain onset in 2.6.5 (the receiver-status script): end of X5 use"),
            ("x5_recovery_utc", f"{RECOVERY:%H:%M:%S}", "UTC", "X5 gain recovery in 2.6.5 (the receiver-status script)")]  # list append by +=
    # f-string log line: PVT blocks before the onset, how many carry a velocity, their interval and solution types
    print(f"  {len(x)} PVTGeodetic blocks before the onset, {len(xv)} with a velocity, median interval {np.median(np.diff(x.t_ns)) / 1e9:.3f} s, "
          f"mode types {x['mode'].value_counts().to_dict()}")  # basic counts
    check("X5 PVT blocks without a velocity (share)", "small", f"{1 - len(xv) / len(x):.4f}", len(xv) / len(x) > 0.9)  # most rows usable
    # self-check: the X5 receiver time is on the laptop clock's scale (host time of each block minus receiver time)
    ht = host_times(folder5 / "host_timing" / (next(folder5.glob("*_SBF_02.sbf")).name + ".hosttime.csv"),
                    x.end.to_numpy(np.int64), host_clock(folder5))  # host UTC of each PVT block
    off = (ht.asi8 - x.t_ns.to_numpy()) / 1e9  # host minus receiver time, seconds
    # self-check: the X5's receiver time must sit within a second of the laptop clock (delivery latency, not a clock error)
    check("X5 host time minus receiver time: median and 99th percentile (s)", "small and positive",
          f"{np.median(off):.3f} and {np.percentile(off, 99):.3f}", abs(np.median(off)) < 1.0)  # latency, not a clock error
    # list += list appends the host-minus-receiver latency numbers to the parameter table
    prm += [("x5_host_minus_receiver_median_s", round(float(np.median(off)), 3), "s", "host time minus receiver time, median"),
            ("x5_host_minus_receiver_p99_s", round(float(np.percentile(off, 99)), 3), "s", "host time minus receiver time, 99th percentile")]  # time-base check
    # the X5 noise floor: 99th percentile of horizontal speed over the long stationary stretch
    win = xv[(xv.t_ns >= LONG_STILL[0].value) & (xv.t_ns <= LONG_STILL[1].value)]  # 12:35:30-12:51:30 samples
    floor_ms = float(np.percentile(win.hspeed_ms, 99))  # the 99th percentile, m/s
    floor_kmh = floor_ms * 3.6  # the same in km/h
    # list += list appends the noise floor, its sample count, the comparison threshold and the event level to the parameter table
    prm += [("x5_noise_floor_kmh", round(floor_kmh, 3), "km/h", "99th percentile of X5 horizontal speed, 12:35:30 to the X5 gain onset"),
            ("x5_noise_floor_samples", len(win), "samples", "X5 samples behind the noise floor"),  # list element: how many X5 samples the noise floor is computed from
            # list element: the X5 speed above which a sample counts as motion when matched to OBD
            ("x5_comparison_threshold_kmh", CMP_KMH, "km/h", "X5 speed above which a sample counts as motion when matched to OBD (OBD resolution)"),
            ("x5_event_level_kmh", round(EVENT_FACTOR * floor_kmh, 3), "km/h", "level above which X5 samples form an event while OBD reads zero (5 noise floors)")]  # two numbers
    print(f"  X5 NOISE FLOOR ({LONG_STILL[0]:%H:%M:%S}-{LONG_STILL[1]:%H:%M:%S}, {len(win)} samples): 99th percentile {floor_ms:.4f} m/s = {floor_kmh:.3f} km/h")  # state it
    print(f"  X5 COMPARISON THRESHOLD: {CMP_KMH:.1f} km/h (the OBD reports whole km/h)")  # and the other number
    check("noise floor lies below the comparison threshold (km/h)", f"< {CMP_KMH}", f"{floor_kmh:.3f}", floor_kmh < CMP_KMH)  # sensible pair
    obd_in = v5[(v5.t_ns >= LONG_STILL[0].value) & (v5.t_ns <= LONG_STILL[1].value)]  # OBD samples in the window
    check("OBD speed over the X5 noise stretch: maximum (km/h)", 0, float(obd_in.kmh.max()), float(obd_in.kmh.max()) == 0)  # all zero
    tx, hx = xv.t_ns.to_numpy(np.int64), xv.hspeed_ms.to_numpy(float) * 3.6  # X5 time and horizontal speed in km/h
    segs_x = base_segments(tx, stationary_state(tx, hx <= CMP_KMH))  # the X5's own stops under the same 10 s rule at 1 km/h
    tr_o, tr_x = transitions(segs5), transitions(segs_x)  # stops and starts of the two witnesses
    # what the noise floor itself would give as the threshold: the first X5 stop under the same 10 s rule, against the OBD's first stop
    tr_f = transitions(base_segments(tx, stationary_state(tx, hx <= floor_kmh)))  # X5 transitions with the floor as threshold
    first_stop_f = next(t for k, t in tr_f if k == "stop")  # the first stop it finds
    # list += list appends the floor-as-threshold result (how many transitions, how late the first stop) to the parameter table
    prm += [("x5_floor_threshold_transitions", len(tr_f), "transitions", "X5 stops and starts found when the noise floor itself is the threshold"),
            ("x5_floor_threshold_first_stop_late_s", round((first_stop_f - tr_o[0][1]) / 1e9, 1), "s", "its first stop minus the OBD's first stop")]  # the first attempt, kept
    print(f"  OBD transitions {[(k, clock(t)) for k, t in tr_o]}")  # printed for the log
    print(f"  X5  transitions {[(k, clock(t)) for k, t in tr_x]}")  # printed for the log
    cmp_rows, used = [], set()  # comparison table; indices of X5 transitions matched to an OBD one
    for kind, t_o in tr_o:  # for loop over the OBD stops and starts
        cand = [(abs(t_x - t_o), j) for j, (k, t_x) in enumerate(tr_x) if k == kind and j not in used]  # same kind, unmatched
        dist, j = min(cand) if cand else (None, None)  # the nearest of them, or none
        if j is not None:  # if: one was found
            used.add(j)  # it is now taken
        t_x = tr_x[j][1] if j is not None else None  # its time
        # dict literal: one row of the comparison table, an OBD stop or start beside the nearest X5 one
        cmp_rows.append({"kind": kind, "obd_utc": stamp(t_o), "x5_utc": stamp(t_x) if t_x else "",
                         "x5_minus_obd_s": round((t_x - t_o) / 1e9, 3) if t_x else np.nan,  # dict entry: X5 transition time minus OBD transition time in seconds, empty when unmatched
                         "within_5s": bool(t_x) and dist / 1e9 <= TOL_S, "note": "matched"})  # one row per OBD transition
    extra = [(k, t_x) for j, (k, t_x) in enumerate(tr_x) if j not in used]  # X5 transitions with no OBD partner
    for k, t_x in extra:  # for loop over them
        # dict literal: an X5 transition with no OBD partner, listed as 'X5 only' and not counted as a failure
        cmp_rows.append({"kind": k, "obd_utc": "", "x5_utc": stamp(t_x), "x5_minus_obd_s": np.nan, "within_5s": np.nan,
                         "note": "X5 only (see events)"})  # listed, not a failure
    pd.DataFrame(cmp_rows).to_csv(PROC / "preprocessing" / "vehicle_motion_x5_check.csv", index=False)  # saved
    for r in cmp_rows:  # for loop over the matched rows
        if r["note"] == "matched":  # if: an OBD transition
            # self-check per OBD transition: its nearest X5 partner of the same kind must lie within the 5 s tolerance
            check(f"{r['kind']} at {r['obd_utc'][11:19]}: nearest X5 {r['kind']} within {TOL_S:.0f} s of OBD", f"|diff| <= {TOL_S}",
                  r["x5_minus_obd_s"], bool(r["within_5s"]))  # each OBD stop and start
    if not all(r["within_5s"] for r in cmp_rows if r["note"] == "matched"):  # if: any OBD stop or start has no X5 partner in 5 s
        pd.DataFrame(CHECKS).to_csv(CHECK_DIR / "motion_checks.csv", index=False)  # keep the checks for inspection
        sys.exit("STOP: an OBD stop or start has no X5 partner within 5 s; see preprocessing/vehicle_motion_x5_check.csv")  # halt

    # ---- X5-only events: movement the receiver sees while the wheels read zero --------------------------------
    still_spans = [(s["t0"], min(s["t1"], X5_READ_END.value)) for s in segs5 if s["state"] == "stationary"]  # OBD stops, to 12:52:00
    events = find_events(xv_all, x_all, still_spans, floor_kmh, [t_o for _, t_o in tr_o])  # detect and classify, spoofer passage included
    print(f"X5-only events while the OBD reads zero (event level {EVENT_FACTOR:.0f} x floor = {EVENT_FACTOR * floor_kmh:.2f} km/h): {len(events)}")  # count
    for i, e in enumerate(events):  # for loop over the events
        e["event"] = i + 1  # numbering from 1
        # f-string log line: the event's window, peak, the two displacement numbers, the adjacency flag and its class
        print(f"  E{e['event']} {e['start_utc'][11:21]}-{e['end_utc'][11:21]} n={e['n_samples']} max {e['max_x5_kmh']} km/h: "
              # f-string continued: integrated displacement with its east and north parts, then the position change
              f"integral {e['int_m']} m (E {e['int_east_m']}, N {e['int_north_m']}), position {e['pos_m']} m "
              # f-string continued: east and north position change, fixes before and after, and the adjacency flag
              f"(E {e['pos_east_m']}, N {e['pos_north_m']}), fixes {e['n_fix_before']}/{e['n_fix_after']}, next to an OBD stop/start: "
              f"{e['adjacent_to_obd_transition']} -> {e['class']}")  # both numbers and the adjacency flag
    ev = pd.DataFrame(events).drop(columns=["t0", "t1"]) if events else pd.DataFrame()  # table without the integer times
    ev.to_csv(PROC / "preprocessing" / "vehicle_motion_x5_events.csv", index=False)  # saved
    # self-check: every X5-only transition at 1 km/h lies inside a detected event (within the merge time)
    for k, t_x in extra:  # for loop over the unmatched X5 transitions
        hit = any(e["t0"] - 2e9 <= t_x <= e["t1"] + 2e9 for e in events)  # inside an event's span, 2 s either side
        check(f"X5-only {k} at {clock(t_x)} falls inside a detected event", "yes", "yes" if hit else "no", hit)  # each one
    # note: runs of X5 blocks without a computed position (PVT mode 0) up to 12:52:00, with the receiver's own error code
    nopvt = x_all[~x_all.pos_ok]  # blocks with no PVT solution
    runs = (nopvt.t_ns.diff() / 1e9 > 1.0).cumsum()  # a new run number wherever the gap exceeds one second
    # str.join over a generator: one phrase per run of X5 blocks without a position, with the receiver's own error codes
    desc = "; ".join(f"{clock(q.t_ns.iloc[0])} to {clock(q.t_ns.iloc[-1])} ({len(q)} blocks, error codes {sorted(q.err.unique().tolist())})"
                     for _, q in nopvt.groupby(runs))  # one phrase per run
    # list += list appends the PVT-loss numbers (blocks, first and last time, delay after the onset) to the parameter table
    prm += [("x5_no_pvt_blocks", len(nopvt), "blocks", "X5 blocks without a PVT solution between the X5 gain onset and 12:52:00"),
            # list element: delay from the X5 gain onset to the first block without a position solution
            ("x5_no_pvt_first_after_onset_s", round((nopvt.t_ns.iloc[0] - ONSET.value) / 1e9, 1), "s", "first such block, seconds after the onset"),
            ("x5_no_pvt_first_utc", clock(nopvt.t_ns.iloc[0])[:8], "UTC", "first such block"),  # list element: clock time of the first block without a position solution
            ("x5_no_pvt_last_utc", clock(nopvt.t_ns.iloc[-1])[:8], "UTC", "last such block")]  # PVT outage numbers
    # self-check (a note): the PVT loss is listed with its first block's delay after the X5 gain onset
    check("X5 blocks without a PVT solution up to 12:52:00 (data-quality note: the spoofer's passage)", "listed",
          f"{len(nopvt)}: {desc}; the first is {(nopvt.t_ns.iloc[0] - ONSET.value) / 1e9:.1f} s after the X5 gain onset {ONSET:%H:%M:%S}", True)  # a note
    creeps = sorted([e for e in events if e["class"].startswith("real creep")], key=lambda e: e["t0"])  # creep events in time order
    print(f"  real creep: {len(creeps)}; receiver velocity noise: {len(events) - len(creeps)}")  # tally

    # ---- final segments: OBD tiles, with stops split at each real creep ---------------------------------------
    final = {}  # test code -> segment list with stats
    for code, _ in sess:  # for loop over the sessions
        segs = [dict(s) for s in obd[code]["segs"]]  # copies of the OBD segments
        if code == T265:  # if: the one session with X5 events
            for e in creeps:  # for loop over the creep events
                for k, s in enumerate(segs):  # for loop over the segments
                    if s["state"] == "stationary" and s["t0"] < e["t0"] < s["t1"]:  # if: the event starts inside this stop
                        segs[k:k + 1] = [{**s, "t1": e["t0"]},  # the part before the event keeps the segment's start
                                         # list element: the stationary part from the event on, with the displacement that caused the split
                                         {"t0": e["t0"], "t1": s["t1"], "state": "stationary", "source": "X5",
                                          "split_m": e["pos_m"]}]  # the part from the event on, a new segment
                        break  # one segment per event
        o = obd[code]  # this session's arrays
        for s in segs:  # for loop over the final segments
            s.update(seg_stats(s, s is segs[-1], o["t"], o["kmh"], o["inv"]))  # adds the numbers
        final[code] = segs  # kept
    out_rows = []  # rows of preprocessing/vehicle_motion.csv
    for code, _ in sess:  # for loop over the sessions
        for k, s in enumerate(final[code]):  # for loop over the final segments
            # dict literal: one row of preprocessing/vehicle_motion.csv, a segment with its source and its longest OBD silence
            out_rows.append({"test_code": code, "segment": k + 1, "state": s["state"], "start_utc": stamp(s["t0"]),
                             "end_utc": stamp(s["t1"]), "duration_s": round((s["t1"] - s["t0"]) / 1e9, 3),  # dict entry: segment end time and its length in seconds
                             # dict entry: peak speed, sample count and invalid SPEED rows of the segment
                             "max_speed_kmh": s["max_speed_kmh"], "n_samples": s["n_samples"], "n_invalid": s["n_invalid"],
                             "source": s["source"], "obd_gap_s": s["obd_gap_s"],  # dict entry: which source set the segment (OBD or X5) and the longest OBD silence inside it
                             "split_displacement_m": s["split_m"]})  # one row per segment
    pd.DataFrame(out_rows).to_csv(PROC / "preprocessing" / "vehicle_motion.csv", index=False)  # the main output table
    ses_rows = []  # one row per session for the motion table
    for code, folder in sess:  # for loop over the sessions
        o = obd[code]  # this session's arrays
        n_allraw, n_validraw, n_moveraw = independent_counts(folder)  # raw counts again, from the csv module
        # dict literal: one row of the per-session table (SPEED rows, gaps, segments, moving and stationary seconds)
        ses_rows.append({"test_code": code, "speed_rows": n_allraw, "valid_rows": n_validraw, "invalid_rows": n_allraw - n_validraw,
                         "max_speed_kmh": float(np.nanmax(o["kmh"])), "moving_samples": n_moveraw,  # dict entry: highest valid OBD speed and the number of moving samples in the session
                         "median_interval_s": round(float(np.median(np.diff(o["t"]))) / 1e9, 4),  # dict entry: median spacing between OBD SPEED rows in seconds
                         "max_gap_s": round(float(np.max(np.diff(o["t"]))) / 1e9, 3), "n_segments": len(final[code]),  # dict entry: longest OBD silence in seconds and the number of motion segments
                         "moving_s": round(sum(s["t1"] - s["t0"] for s in final[code] if s["state"] == "moving") / 1e9, 1),  # dict entry: total seconds the vehicle spent moving in the session
                         "stationary_s": round(sum(s["t1"] - s["t0"] for s in final[code] if s["state"] == "stationary") / 1e9, 1)})  # row
    pd.DataFrame(ses_rows).to_csv(PROC / "preprocessing" / "vehicle_motion_obd_sessions.csv", index=False)  # the per-session table
    for code, _ in sess:  # for loop over the sessions: the self-checks again on the final table
        # arithmetic on nanosecond integers: the total length of the final segments in seconds
        tot = sum(s["t1"] - s["t0"] for s in final[code]) / 1e9
        # the same session's record span in seconds, from its first to its last valid sample
        span = (obd[code]["t"][-1] - obd[code]["t"][0]) / 1e9
        check(f"{code}: final segments (after any X5 split) sum to the span (s)", f"{span:.3f}", f"{tot:.3f}", abs(tot - span) < 1e-6)  # tile
        # self-check: no valid SPEED sample is lost when the segments are cut or split
        check(f"{code}: samples in the final segments equal the valid SPEED rows", len(obd[code]["v"]),
              sum(s["n_samples"] for s in final[code]), len(obd[code]["v"]) == sum(s["n_samples"] for s in final[code]))  # nothing lost
    print("2.6.5 final segments (UTC):")  # heading in the log
    for i, s in enumerate(final[T265]):  # for loop over them
        print(f"  {i + 1} {s['state']:10s} {s['source']:3s} {stamp(s['t0'])[11:]} -> {stamp(s['t1'])[11:]}  {(s['t1'] - s['t0']) / 1e9:8.2f} s  gap {s['obd_gap_s']} s  invalid {s['n_invalid']}")  # row

    # ---- agreement of the two speeds while moving, and the OBD data-quality notes ----------------------------
    mv = v5[(v5.kmh > 0) & (v5.t_ns < ONSET.value)]  # OBD samples with motion before the X5 gain onset
    best = min(((float(np.median(np.abs(np.interp(mv.t_ns + lag * 1e9, tx, hx) - mv.kmh))), lag)  # median |difference|
                for lag in np.arange(-5, 5.01, 0.1)))  # tried for lags from -5 s to +5 s in 0.1 s steps
    zero_lag = float(np.median(np.abs(np.interp(mv.t_ns, tx, hx) - mv.kmh)))  # the same at zero lag
    # list += list appends the OBD/X5 speed-agreement numbers (zero-lag difference, best lag, difference there)
    prm += [("x5_obd_median_abs_diff_kmh", round(zero_lag, 2), "km/h", "median |X5 - OBD| speed while moving, zero lag"),
            ("x5_obd_best_lag_s", round(float(best[1]), 1), "s", "lag at which that difference is smallest"),  # list element: time lag at which the X5 and OBD speeds agree best
            ("x5_obd_best_lag_diff_kmh", round(best[0], 2), "km/h", "median |X5 - OBD| at the best lag")]  # agreement numbers
    check("moving speeds: median |X5 - OBD| at zero lag (km/h)", "a few km/h at most", f"{zero_lag:.2f} (best lag {best[1]:+.1f} s gives {best[0]:.2f})", zero_lag < 5)  # sizes
    for code, _ in sess:  # for loop over the sessions: list every OBD silence longer than one second
        t, d = obd[code]["t"], obd[code]["d"]  # valid times; every SPEED row
        for k in np.flatnonzero(np.diff(t) / 1e9 > 1.0):  # indices where the next valid sample is more than 1 s later
            a, b = int(t[k]), int(t[k + 1])  # the silence's first and last valid sample
            n_in = int(((d.t_ns > a) & (d.t_ns < b) & (~d.valid)).sum())  # invalid rows inside the silence
            # f-string: the silence's start, end, length, invalid rows and the speed on both sides, for the checks file
            found = f"{clock(a)} to {clock(b)} ({(b - a) / 1e9:.1f} s), {n_in} invalid rows, speed {obd[code]['kmh'][k]:.0f} before and {obd[code]['kmh'][k + 1]:.0f} after"
            if code == T265:  # if: the 2.6.5 outage: say what the X5 saw during it
                g = xv[(xv.t_ns > a) & (xv.t_ns < b)]  # X5 samples inside it
                state = "stationary" if g.hspeed_ms.max() * 3.6 <= CMP_KMH else "moving"  # at the 1 km/h comparison threshold
                # f-string appended with +=: for the 2.6.5 outage, what the X5 saw (state, median, maximum and when)
                found += (f"; X5 during it: {state}, median {g.hspeed_ms.median() * 3.6:.2f} km/h, maximum "
                          f"{g.hspeed_ms.max() * 3.6:.2f} km/h at {clock(g.t_ns.iloc[int(np.argmax(g.hspeed_ms.to_numpy()))])}")  # X5 view
            # list += list appends the silence's start, end, length and invalid-row count to the parameter table
            prm += [(f"obd_gap_{code}_start_utc", clock(a)[:10], "UTC", f"start of an OBD silence over 1 s in {code}"),
                    (f"obd_gap_{code}_end_utc", clock(b)[:10], "UTC", "end of that silence"),  # list element: clock time at which this OBD silence ends
                    (f"obd_gap_{code}_s", round((b - a) / 1e9, 1), "s", "its length"),  # list element: length of the OBD silence in seconds
                    (f"obd_gap_{code}_invalid_rows", n_in, "rows", "invalid SPEED rows inside it")]  # the gap's numbers for the report
            if code == T265:  # if: the 2.6.5 outage: keep what the X5 saw during it
                # list += list appends the X5's median and maximum speed during the 2.6.5 outage
                prm += [("obd_gap_2.6.5_x5_median_kmh", round(float(g.hspeed_ms.median()) * 3.6, 2), "km/h", "X5 horizontal speed during it, median"),
                        ("obd_gap_2.6.5_x5_max_kmh", round(float(g.hspeed_ms.max()) * 3.6, 2), "km/h", "X5 horizontal speed during it, maximum")]  # X5 view
            check(f"{code}: OBD silence over 1 s (data-quality note)", "listed", found, True)  # a note, not a flag on labels
    inv5 = obd[T265]["inv"]  # invalid SPEED times in 2.6.5
    first_inv = int(inv5.min())  # the first invalid row
    # self-check (a note): the invalid rows begin a few seconds after the final stop, so the outage is tied to it
    check("2.6.5 invalid SPEED rows relative to the final stop (data-quality note)", "listed",
          f"{len(inv5)} rows from {clock(first_inv)}, {(first_inv - tr_o[-1][1]) / 1e9:.1f} s after the stop at {clock(tr_o[-1][1])}", True)  # clustering

    # ---- where was the vehicle? ----------------------------------------------------------------------------------
    print("2.6.5 stationary segments: where was the vehicle?")  # heading in the log
    sd = pd.read_csv(PROC / "ground_truth" / "session_distance_to_sender.csv")  # the handset script's measured site table
    pick = lambda c: sd[sd.test_code == c].iloc[0]  # lambda: the first the handset script row of a session
    sw = sd[sd.test_code.isin(["2.6.10", "2.6.4", "2.6.11"])]  # the three sessions at the south-west site
    sites = {"Stave (2.6.12 site)": (pick("2.6.12").median_lat, pick("2.6.12").median_lon),  # Stave as the 2.6.12 median
             "2.6.2 site": (pick("2.6.2").median_lat, pick("2.6.2").median_lon),  # the 2.6.2 site
             "south-west site (2.6.10/2.6.4/2.6.11)": (float(sw.median_lat.median()), float(sw.median_lon.median())),  # SW
             "Stave landmark (the handset script's constant)": tuple(S02.STAVE)}  # the landmark constant used in the handset script
    spans = [("OBD stop", i + 1, s["t0"], s["t1"]) for i, s in enumerate(segs5) if s["state"] == "stationary"]  # whole OBD stops
    if creeps:  # if: some stop was split, also give each final stationary segment
        spans += [("final segment", i + 1, s["t0"], s["t1"]) for i, s in enumerate(final[T265]) if s["state"] == "stationary"]  # parts
    stop_rows = []  # one row per span
    for scope, num, a, b in spans:  # for loop over the spans
        hi = min(b, ONSET.value)  # segment end, cut at the X5 gain onset (the X5 is used before the spoofer's passage only)
        pp = x[(x.pos_ok) & (x.t_ns >= a) & (x.t_ns <= hi)]  # X5 positions inside the span
        la, lo = float(pp.lat.median()), float(pp.lon.median())  # the median position
        e_, n_ = enu_m(la, lo, pp.lat.to_numpy(), pp.lon.to_numpy())  # each fix's offset from the median, metres
        spread = float(np.percentile(np.hypot(e_, n_), 95))  # 95th percentile scatter around the median, m
        # dict literal: one row of preprocessing/vehicle_motion_stops.csv (scope, times, fix count, median position and scatter so far)
        row = {"scope": scope, "segment": num, "start_utc": stamp(a), "end_used_utc": stamp(hi), "n_pvt": len(pp),
               "median_lat": round(la, 7), "median_lon": round(lo, 7), "p95_scatter_m": round(spread, 1)}  # base columns
        best_site, best_d = None, 1e12  # nearest known recording site so far
        for nm, (sa, sb) in sites.items():  # for loop over the known sites
            dm = float(haversine_m(la, lo, sa, sb))  # distance in metres from the median position
            row["dist_m_" + nm.split(" (")[0].replace(" ", "_")] = round(dm, 1)  # one distance column per site
            if "landmark" not in nm and dm < best_d:  # the landmark is a point, not a recording site
                best_site, best_d = nm, dm  # keeps the nearest recording site
        row["nearest_recording_site"] = best_site if best_d <= SITE_RADIUS_M else "none within 100 m"  # verdict
        row["nearest_site_distance_m"] = round(best_d, 1)  # and its distance
        stop_rows.append(row)  # saved
        # f-string log line: the stop's span, fix count, median position, scatter and nearest recording site
        print(f"  {scope} {num} ({row['start_utc'][11:19]}-{row['end_used_utc'][11:19]}): {len(pp)} fixes, "
              f"{la:.6f}, {lo:.6f}, p95 scatter {spread:.1f} m -> {row['nearest_recording_site']} ({best_d:.0f} m)")  # report
        print("      " + "; ".join(f"{k} {v:.0f} m" for k, v in row.items() if str(k).startswith("dist_m_")))  # all distances
    pd.DataFrame(stop_rows).to_csv(PROC / "preprocessing" / "vehicle_motion_stops.csv", index=False)  # saved

    plot_vehicle_motion_265(v5, final[T265], xv, creeps)  # the figure
    check_plot(obd)  # the one-panel check plot
    pd.DataFrame(CHECKS).to_csv(CHECK_DIR / "motion_checks.csv", index=False)  # every check, for the check
    pd.DataFrame(prm, columns=["name", "value", "unit", "description"]).to_csv(PROC / "preprocessing" / "vehicle_motion_x5_params.csv", index=False)  # settings
    print(f"checks: {sum(c['ok'] for c in CHECKS)} of {len(CHECKS)} ok")  # summary line


def plot_vehicle_motion_265(v5: pd.DataFrame, segs: list, xv: pd.DataFrame, creeps: list) -> None:  # function: the figure, OBD and X5 speed of 2.6.5
    fig, ax = plt.subplots(figsize=(7.2, 3.8))  # one panel
    for s in segs:  # for loop over the segments
        if s["state"] == "stationary":  # if: a stop
            # axvspan shades a vertical band between two times: marks each stationary segment grey on the figure
            ax.axvspan(mdates.date2num(pd.Timestamp(s["t0"], tz="UTC")), mdates.date2num(pd.Timestamp(s["t1"], tz="UTC")),
                       color=C_GREY, alpha=0.25, lw=0)  # shaded band for the stationary segment
    # plot draws a line: the OBD speed against UTC, in blue
    ax.plot(mdates.date2num(pd.to_datetime(v5.t_ns, unit="ns", utc=True)), v5.kmh, color="#0072B2", lw=1.0,
            label="OBD-II speed")  # the vehicle's own speed
    # plot draws a line: the X5 horizontal speed (m/s times 3.6 = km/h) in orange, before its gain onset
    ax.plot(mdates.date2num(pd.to_datetime(xv.t_ns, unit="ns", utc=True)), xv.hspeed_ms * 3.6, color="#D55E00", lw=0.8,
            alpha=0.85, label="mosaic-X5 horizontal speed")  # the receiver's own velocity (before its gain onset)
    last = pd.Timestamp(segs[-1]["t0"], tz="UTC")  # the instant the vehicle came to rest for good: the final segment's first sample
    ax.axvline(mdates.date2num(last), color="black", lw=0.9, ls="--")  # vertical line at 12:34:56
    ax.axvline(mdates.date2num(ONSET), color="#CC79A7", lw=0.9, ls=":")  # vertical line at the X5 gain onset, 12:50:59
    for e in creeps:  # for loop over the real-creep events: a thin line where a stop was split
        ax.axvline(mdates.date2num(pd.Timestamp(e["t0"], tz="UTC")), color="#009E73", lw=0.8, ls="-.")  # split marker
    ax.text(mdates.date2num(last) + 10 / 86400, 46, f"{last.round('s'):%H:%M:%S}", fontsize=7, ha="left")  # label, a few seconds right of its line
    ax.text(mdates.date2num(ONSET) - 10 / 86400, 46, f"{ONSET:%H:%M:%S}\n(X5 gain onset,\nend of X5 use)", fontsize=7, ha="right")  # label
    ax.set_ylabel("speed (km/h)")  # axis label
    ax.set_xlabel("UTC (OBD-II on the laptop clock, mosaic-X5 on receiver time)")  # axis label
    ax.set_ylim(-2, 56)  # range with a little margin
    # time_axis formats the x axis as UTC HH:MM between two instants (12:26 to 12:55)
    time_axis(ax, pd.Timestamp("2026-09-14 12:26:00", tz="UTC").to_pydatetime(),
              pd.Timestamp("2026-09-14 12:55:00", tz="UTC").to_pydatetime())  # UTC axis, 12:26-12:55
    ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 0.82))  # legend below the labels
    fig.tight_layout()  # spacing
    save_fig(fig, GRAPH / "vehicle_motion_265.png")  # written, byte-stable


def check_plot(obd: dict) -> None:  # function: OBD speed of all six sessions on one axis (check, not a thesis figure)
    fig, ax = plt.subplots(figsize=(9.0, 3.4))  # one panel
    for i, (code, o) in enumerate(obd.items()):  # for loop over the six sessions
        # plot draws one coloured line per session, labelled with its maximum speed
        ax.plot(mdates.date2num(pd.to_datetime(o["t"], unit="ns", utc=True)), o["kmh"], lw=1.0, color=f"C{i}",
                label=f"{code} (max {np.nanmax(o['kmh']):.0f} km/h)")  # one coloured line per session
    ax.set_ylabel("OBD-II speed (km/h)")  # axis label
    ax.set_xlabel("UTC")  # axis label
    # time_axis formats the whole-afternoon axis (12:20 to 16:15 UTC) for the check plot
    time_axis(ax, pd.Timestamp("2026-09-14 12:20:00", tz="UTC").to_pydatetime(),
              pd.Timestamp("2026-09-14 16:15:00", tz="UTC").to_pydatetime())  # whole afternoon
    ax.legend(fontsize=7, ncol=3, loc="upper right")  # legend
    fig.tight_layout()  # spacing
    save_fig(fig, CHECK_DIR / "obd_speed_all_sessions.png")  # written


if __name__ == "__main__":  # if: true only when the file is run directly
    main()  # runs everything
    exit_if_failed(FAILED, "2_preprocessing/02_vehicle_motion.py")  # the helpers module: status 1 when any self-check failed, so a chained run stops
