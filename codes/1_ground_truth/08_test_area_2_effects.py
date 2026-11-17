#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
Final-label rule L2: does a concurrent test at Test Area 2, or a Test Area 1 sweep, show in our receivers' gain?

Rule:
  * Tests that decide: every concurrent test at Test Area 2 that lists a feature band and overlaps a capture (ground_truth/concurrent_transmissions.csv:
    1.1.12, 1.1.13, 1.1.21, 1.1.23, 1.1.26, 1.1.27, 1.1.29), and the two Test Area 1 sweeps 1.3.1 and 1.3.12. The other Test Area 1 tests
    that overlap a capture (1.2.6, 1.8.2) are measured and reported only.
  * At each on edge (concurrent start) and off edge (concurrent end) whose 30 s on both sides lie inside the capture, per receiver and per
    feature band (GPSL1/E1, GLOL1, B1 on the two Septentrio receivers, mosaic-H main antenna; the L1 block on u-blox 01), the gain step
    is the median of the per-second gain over [e, e + 30 s) minus its median over [e - 30 s, e). A test has an effect on a receiver when
    any such step reaches the label script's gain-evidence threshold, |step| >= 5 dB (400 counts on the u-blox), at any edge, in the direction the
    edge implies: a fall at a switch-on, a rise at a switch-off (post hoc: it removes the false attribution of the
    motorcade's 12:36 fall in 2.6.5 to 1.1.21's switch-off and changes no label, the direction-blind result is reported beside).
    A test has an effect at the antenna when it has one on any receiver (pooled, as the labels pool evidence).
  * Effect: the label script (final label set) makes the clean and attack seconds the test overlaps, [start, end) inside the capture, uncertain on all
    three receivers (Option 1, label_per_receiver_only: only on the receivers with their own effect), with the labels' 30 s guard.
  * C/N0 (GPS median per second from the observables script's observables table) is stepped the same way and reported beside the gain, after the effect is
    decided; it moves no label. 1.1.27 in 2.6.11 is reported whatever the rule gives.

Reads   Data/processed/ground_truth/concurrent_transmissions.csv, ground_truth/session_ground_truth.csv, ground_truth/receiver_status_timeseries.csv (gain), labels/labels_1hz.csv
        (overlap by registered label, report), features/observables_1hz.csv.gz (GPS C/N0, report only, read after the decision)
Writes  Data/processed/ground_truth/ta2_test_effects.csv (one row per session, test, receiver and edge), Data/processed/checks/ta2_checks.csv
"""

from __future__ import annotations  # future import: type hints kept as text (the project's convention)

import sys  # system access: sys.path, so the helpers module is found
from pathlib import Path  # Path objects for building file paths

import numpy as np  # NumPy: medians and NaN
import pandas as pd  # pandas: tables

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
sys.path.insert(0, str(PROJECT / "codes" / "common"))  # list method insert: puts codes/ first on the import path so the helpers module is found
from helpers import exit_if_failed, script_constant  # status 1 on a failed check; a constant of another script read as text

PROC = PROJECT / "Data" / "processed"  # the processed-data folder
CHECK_DIR = PROJECT / "Data" / "processed" / "checks"  # where the check table goes
DATE = "2026-09-14"  # the recording date (UTC)
RXS = ["SBF_02", "SBF_03", "UBX_01"]  # list: the three receivers on the shared antenna
RX = {"SBF_02": ("Septentrio_02", "agc_gain_db"), "SBF_03": ("Septentrio_03", "agc_gain_db"), "UBX_01": ("u-blox_01", "agc_cnt")}  # dict: stream -> (status tag, gain field)
FEATURE = {"SBF_02": ["GPSL1/E1", "GLOL1", "B1"], "SBF_03": ["GPSL1/E1", "GLOL1", "B1"], "UBX_01": ["L1"]}  # dict: the feature bands of each receiver's status
GAIN_DROP = script_constant(PROJECT / "codes" / "3_labels" / "01_labels.py", "GAIN_DROP")  # dict: the label script's gain-evidence thresholds, 5 dB and 400 counts (read as text)
FE = set(script_constant(PROJECT / "codes" / "3_labels" / "01_labels.py", "FEATURE_IN"))  # set: the band names that make a concurrent test a feature-band test (as in the label script)
SWEEPS = ("1.3.1", "1.3.12")  # tuple: the two Test Area 1 sweeps that decide
SIDE_S = 30  # integer constant: seconds on each side of an edge
LABELS = ["clean", "attack_jam_spoof", "attack_spoof", "uncertain", "excluded"]  # list: the five labels, for the overlap counts
CHECKS: list = []  # list: one dict per check
FAILED: list = []  # list: names of the checks that failed


def T(text: str) -> int:  # function: "YYYY-MM-DD HH:MM:SS" (or "HH:MM:SS" on the recording date) -> whole unix seconds
    return int(pd.Timestamp(text if len(text) > 8 else f"{DATE} {text}", tz="UTC").timestamp())  # conditional expression: a time alone takes the date


def hms(sec) -> str:  # function: unix seconds -> "HH:MM:SS"
    return pd.Timestamp(int(sec), unit="s", tz="UTC").strftime("%H:%M:%S")  # formatted


def check(name: str, ok, detail: str) -> None:  # function: records one check and prints it (ok None marks a reported number)
    CHECKS.append({"check": name, "result": "REPORT" if ok is None else ("PASS" if ok else "FAIL"), "detail": detail})  # nested conditional expression picks the word
    if ok is not None and not bool(ok):  # if: a pass/fail check that did not pass
        FAILED.append(name)  # kept for the exit status
    print(f"  [{'REPORT' if ok is None else ('PASS' if ok else 'FAIL')}] {name}: {detail}")  # printed for the run log


def step(s: pd.Series, e: int) -> float:  # function: median of a per-second series over [e, e + 30) minus its median over [e - 30, e); NaN when a side is empty
    after, before = s[(s.index >= e) & (s.index < e + SIDE_S)], s[(s.index >= e - SIDE_S) & (s.index < e)]  # boolean selections: the two sides
    return float(after.median() - before.median()) if len(after) and len(before) else float("nan")  # conditional expression: the step, or NaN


def tests() -> pd.DataFrame:  # function: the concurrent tests that decide and those reported only
    ct = pd.read_csv(PROC / "ground_truth" / "concurrent_transmissions.csv", dtype={"our_test_code": str})  # every concurrent test x our session
    fb = ct.bands.apply(lambda b: bool(FE & set(str(b).split(";"))))  # lambda in apply: does the test list a feature band
    ct["role"] = np.where((ct.concurrent_test_area == 2) & fb | ct.concurrent_test_code.isin(SWEEPS), "decides", "reported only")  # np.where: Test Area 2 feature-band tests and the sweeps decide
    return ct.sort_values(["our_test_code", "concurrent_start_utc", "concurrent_test_code"]).reset_index(drop=True)  # a fixed order


def main() -> None:  # function: steps, effects, overlaps, then C/N0, then the table
    gt = pd.read_csv(PROC / "ground_truth" / "session_ground_truth.csv").set_index("test_code")  # capture times
    st = pd.read_csv(PROC / "ground_truth" / "receiver_status_timeseries.csv", usecols=["test_code", "receiver", "timestamp_utc", "band", "agc_gain_db", "agc_cnt"], low_memory=False)  # gains only
    st = st[~st.band.str.contains("Aux", na=False)].copy()  # main antenna only
    st["sec"] = pd.to_datetime(st.timestamp_utc, utc=True).astype("int64") // 10 ** 9  # whole seconds (floor)
    lab = pd.read_csv(PROC / "labels" / "labels_1hz.csv", usecols=["test_code", "receiver", "sec", "label"], dtype={"test_code": str})  # registered labels (overlap counts)
    ct = tests()  # the tests
    rows = []  # one row per session, test, receiver and edge
    for r in ct.itertuples():  # for loop over every test x session
        t = r.our_test_code  # our session
        cs, ce = T(gt.at[t, "capture_start_utc"]), T(gt.at[t, "capture_end_utc"])  # capture, both ends inclusive
        a, b = T(r.concurrent_start_utc), T(r.concurrent_end_utc)  # the test's window, half-open [a, b) as the label script's column
        for rx in RXS:  # for loop over the receivers
            tag, fld = RX[rx]  # its status tag and gain field
            y = st[(st.receiver == tag) & (st.test_code == t)]  # its readings in the session
            x = lab[(lab.test_code == t) & (lab.receiver == rx) & (lab.sec >= a) & (lab.sec < b)]  # registered labels of the overlapped seconds
            ov = {f"overlap_{l}_s": int((x.label == l).sum()) for l in LABELS}  # dict comprehension: overlapped seconds by registered label
            for edge, e in (("on", a), ("off", b)):  # for loop over the two edges
                inside = cs <= e - SIDE_S and e + SIDE_S - 1 <= ce  # both 30 s sides inside the capture
                row = {"our_test": t, "concurrent_test": r.concurrent_test_code, "test_area": int(r.concurrent_test_area), "role": r.role, "power_w": r.power_w,  # identity
                       "start_utc": hms(a), "end_utc": hms(b), "receiver": rx, "edge": edge, "edge_utc": hms(e), "edge_inside_capture": bool(inside), **ov}  # dict literal: identity and overlap
                steps = {}  # band -> gain step at this edge
                for bd in FEATURE[rx]:  # for loop over the receiver's feature bands
                    s = y[y.band == bd].groupby("sec")[fld].median()  # the band's gain per second (median of the readings in each second)
                    steps[bd] = step(s, e) if inside else float("nan")  # conditional expression: the step, or NaN for an edge outside the capture
                    row[f"gain_step_{bd}"] = round(steps[bd], 3) if np.isfinite(steps[bd]) else np.nan  # one column per band
                fin = {k: v for k, v in steps.items() if np.isfinite(v)}  # dict comprehension: the bands with a step
                big = max(fin, key=lambda k: abs(fin[k])) if fin else ""  # max with a key: the band with the largest |step|
                row["max_abs_gain_step"] = round(abs(fin[big]), 3) if fin else np.nan  # its size
                row["band_of_max"], row["direction"] = big, ("" if not fin else ("down" if fin[big] < 0 else "up"))  # tuple assignment: the band and the sign
                row["threshold"] = GAIN_DROP[fld]  # 5 dB or 400 counts
                row["effect_edge_direction_blind"] = bool(fin) and abs(fin[big]) >= GAIN_DROP[fld]  # the first wording of the rule (reported only)
                row["effect_edge"] = any(abs(v) >= GAIN_DROP[fld] and (v < 0 if edge == "on" else v > 0) for v in fin.values())  # generator in any(): a large enough step in the edge's direction on any feature band
                rows.append(row)  # kept
    out = pd.DataFrame(rows)  # the table
    key = ["our_test", "concurrent_test"]  # one test in one session
    out["effect_receiver"] = out.groupby(key + ["receiver"]).effect_edge.transform("any")  # transform("any"): the test has an effect on this receiver at either edge
    out["effect_antenna"] = out.groupby(key).effect_edge.transform("any") & (out.role == "decides")  # on any receiver, for the tests that decide
    out["seconds_moved_direct"] = np.where(out.effect_antenna & (out.edge == "on"), out.overlap_clean_s + out.overlap_attack_jam_spoof_s + out.overlap_attack_spoof_s, 0)  # clean and attack seconds made uncertain (counted once, on the on-edge row; the guard comes in the label script)
    # ---- C/N0, read only now that every effect is fixed (report only)
    ob = pd.read_csv(PROC / "features" / "observables_1hz.csv.gz", usecols=["test_code", "receiver", "sec", "sys", "cn0"], dtype={"test_code": str})  # the observables script's per-satellite table
    cn = ob[ob.sys == "G"].groupby(["test_code", "receiver", "sec"]).cn0.median()  # GPS C/N0 median per second
    vals = []  # the C/N0 step of each row
    for r in out.itertuples():  # for loop over the rows
        s = cn.xs((r.our_test, r.receiver), level=["test_code", "receiver"]) if (r.our_test, r.receiver) in cn.index.droplevel("sec") else pd.Series(dtype=float)  # conditional expression: the pair's series
        vals.append(round(step(s, T(r.edge_utc)), 3) if r.edge_inside_capture else np.nan)  # conditional expression: the step, or NaN outside the capture
    out["cn0_step_dbhz"] = vals  # reported beside the gain
    out.to_csv(PROC / "ground_truth" / "ta2_test_effects.csv", index=False, lineterminator="\n")  # written
    dec = out[out.role == "decides"]  # the deciding rows
    check("every deciding test and session has rows for the three receivers and both edges", bool((dec.groupby(key).size() == 6).all()), f"{dec.groupby(key).ngroups} test x session pairs")  # completeness
    eff = dec[dec.effect_antenna].drop_duplicates(key)  # tests with an effect at the antenna
    check("tests with an effect at the antenna", None, "; ".join(f"{r.our_test} {r.concurrent_test}" for r in eff.itertuples()) or "none")  # reported
    x = out[(out.our_test == "2.6.11") & (out.concurrent_test == "1.1.27")]  # 1.1.27 in 2.6.11, reported whatever the rule gives
    check("1.1.27 in 2.6.11 (disclosed)", None, "; ".join(f"{r.receiver} {r.edge} {r.edge_utc}: gain {r.max_abs_gain_step} {r.direction} ({r.band_of_max}), C/N0 {r.cn0_step_dbhz}" for r in x.itertuples()))  # reported
    pd.DataFrame(CHECKS).to_csv(CHECK_DIR / "ta2_checks.csv", index=False, lineterminator="\n")  # the checks
    print(out[["our_test", "concurrent_test", "role", "receiver", "edge", "edge_utc", "edge_inside_capture", "max_abs_gain_step", "band_of_max", "direction", "effect_edge", "effect_antenna", "cn0_step_dbhz"]].to_string(index=False))  # for the run log


if __name__ == "__main__":  # if: true only when the file is run directly
    main()  # runs L2
    exit_if_failed(FAILED, "1_ground_truth/08_test_area_2_effects.py")  # the helpers module: status 1 when any check failed
