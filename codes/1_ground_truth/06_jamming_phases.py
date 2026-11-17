#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
Final-label rule L1: the opening-jamming phase of 2.6.5, 2.6.12, 2.6.4 and 2.6.11, per receiver, read from
each receiver's own front-end gain, and at the antenna as the union over the receivers whose gain reads it (as the labels pool evidence).

Rule:
  * Step events: the gain-pattern script's boundary rule (as applied to the labels): the group's mean gain per second over the receiver's span in the session,
    interpolated over gaps of up to 6 s, a centred 5 s rolling median (at least 2 values), the change from t-3 s to t+3 s; a change of
    at least 3 dB (u-blox: 240 counts, 400 counts per 5 dB as the receiver-status script's entry drops) is a step; a new event needs more than 20 s since
    the previous step second. The feature-band group is the gain-pattern script's: GPSL1/E1 and B1 on the Septentrio receivers (mosaic-H main antenna)
    and the L1 block on u-blox 01 (GLOL1, first included, is withdrawn: the gain-pattern script left it out because G1 shows almost no gain
    response, and with it the X5's up event moved 2 s and the mosaic-H lost its 2.6.5 down step). The non-feature bands are GPSL2, L5/E5a/B2a and E5b/B2b (no E6 band on these receivers) and the u-blox
    L2 block, each band on its own.
  * A candidate phase is a feature-group down step followed by the first feature-group up step after it. It is accepted when
      (a) the down step lies no earlier than the scheduled jamming start minus 3 s and no later than the scheduled jamming end plus
          JAM_TAIL_S (120 s); the 3 s is the rule's half-width (an event is the first second whose t-3..t+3 change reaches the
          threshold, so it can precede the step by up to 3 s);
      (b) no non-feature band has an up event within 3 s (the rule's half-width) of the feature up event: the phase ends at the second
          the feature bands recover while the non-feature bands stay depressed (a whole-interval reading, under which a partial non-feature recovery inside the phase rejected 2.6.5, is not used);
      (c) the up step comes no more than the scheduled phase (5 min) plus JAM_TAIL_S after the down step.
    When the capture starts inside the scheduled phase (2.6.11), the capture start stands for the down step it could not see.
    The earliest accepted candidate is the receiver's phase, [down step, up step - 1] in whole seconds (the labels' convention).
  * At the antenna: the union of the per-receiver phases; where no receiver reads one, the schedule (first five minutes) is kept.
  * Gate R0 (implementation): on the mosaic-X5 with the gain-pattern script's own two groups (L1 group GPSL1/E1 + B1; continuous GPSL2, E5b/B2b,
    L5/E5a/B2a) the events equal ground_truth/phase_boundaries.csv row for row. Gate G1: the X5's 2.6.5 phase lies within 1 s of
    12:36:00 and 12:40:00 at both edges. If R0 fails, nothing is written and the run stops. If G1 fails, the antenna
    phase falls back to group_overrides' 12:36:00-12:40:00 in 2.6.5 and to the schedule elsewhere, and the run continues.
  * The withdrawn three-band group's phase (GPSL1/E1, GLOL1, B1) is reported beside the ruled one; it decides nothing.
Only the receivers' status gains are read: no residual, C/N0 or other observable.

Reads   Data/processed/ground_truth/receiver_status_timeseries.csv, ground_truth/session_ground_truth.csv, ground_truth/test_phases.csv, ground_truth/phase_boundaries.csv (gate R0);
        codes/1_ground_truth/04_gain_and_position_evidence.py (JAM_TAIL_S, read as text)
Writes  Data/processed/ground_truth/jam_phase_boundaries.csv (per receiver and at the antenna), Data/processed/checks/jam_events.csv (every event),
        Data/processed/checks/jam_checks.csv (the gates and checks)
"""

from __future__ import annotations  # future import: type hints kept as text (the project's convention)

import sys  # system access: sys.path, so the helpers module is found
from pathlib import Path  # Path objects for building file paths

import numpy as np  # NumPy: the per-second grid
import pandas as pd  # pandas: tables and rolling medians

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
sys.path.insert(0, str(PROJECT / "codes" / "common"))  # list method insert: puts codes/ first on the import path so the helpers module is found
from helpers import exit_if_failed, script_constant  # status 1 on a failed check; a constant of another script read as text

PROC = PROJECT / "Data" / "processed"  # the processed-data folder
CHECK_DIR = PROJECT / "Data" / "processed" / "checks"  # where the check tables go
DATE = "2026-09-14"  # the recording date (UTC)
SESSIONS = ["2.6.5", "2.6.12", "2.6.4", "2.6.11"]  # list: the sessions with a scheduled opening jamming phase (2.6.2 and 2.6.10 exempt themselves)
# dict: stream -> (status-table tag, gain field, step threshold): 3 dB on the Septentrio gains, 240 counts on the u-blox (400 counts per 5 dB)
RX = {"SBF_02": ("Septentrio_02", "agc_gain_db", 3.0), "SBF_03": ("Septentrio_03", "agc_gain_db", 3.0), "UBX_01": ("u-blox_01", "agc_cnt", 3.0 * 400.0 / 5.0)}
FEATURE = {"SBF_02": ["GPSL1/E1", "B1"], "SBF_03": ["GPSL1/E1", "B1"], "UBX_01": ["L1"]}  # dict: the feature-band group of each receiver, the gain-pattern script's two bands
WITHDRAWN = {"SBF_02": ["GPSL1/E1", "GLOL1", "B1"], "SBF_03": ["GPSL1/E1", "GLOL1", "B1"], "UBX_01": ["L1"]}  # dict: the earlier three-band group, withdrawn (reported only)
NONFEATURE = {"SBF_02": ["GPSL2", "L5/E5a/B2a", "E5b/B2b"], "SBF_03": ["GPSL2", "L5/E5a/B2a", "E5b/B2b"], "UBX_01": ["L2"]}  # dict: the non-feature bands, each judged alone
L1_05H, CONT_05H = ["GPSL1/E1", "B1"], ["GPSL2", "E5b/B2b", "L5/E5a/B2a"]  # tuple assignment: the gain-pattern script's own two groups (gate R0 and the reported two-band phase)
INTERP_S, MED_S, HALF_S, GAP_S = 6, 5, 3, 20  # the gain-pattern script's rule: interpolation limit, centred median length, half-width of the change, seconds between events
JAM_TAIL_S = float(script_constant(PROJECT / "codes" / "1_ground_truth" / "04_gain_and_position_evidence.py", "JAM_TAIL_S"))  # 120 s: the lag after the scheduled jamming end during which the gain can still show it
GATE, GATE_TOL_S = ("12:36:00", "12:40:00"), 1  # tuple assignment: the X5's 2.6.5 phase from group_overrides and the tolerance of the decision
CHECKS: list = []  # list: one dict per check, written at the end
FAILED: list = []  # list: names of the checks that failed


def T(hms: str) -> int:  # function: "HH:MM:SS" on the recording date -> whole unix seconds
    return int(pd.Timestamp(f"{DATE} {hms}", tz="UTC").timestamp())  # parse as UTC, seconds since 1970


def hms(sec) -> str:  # function: unix seconds -> "HH:MM:SS" text ("" for a missing value)
    return "" if sec is None or (isinstance(sec, float) and np.isnan(sec)) else pd.Timestamp(int(sec), unit="s", tz="UTC").strftime("%H:%M:%S")  # conditional expression: empty for none


def check(name: str, ok, detail: str) -> None:  # function: records one check and prints it (ok None marks a reported number)
    CHECKS.append({"check": name, "result": "REPORT" if ok is None else ("PASS" if ok else "FAIL"), "detail": detail})  # nested conditional expression picks the word
    if ok is not None and not bool(ok):  # if: a pass/fail check that did not pass
        FAILED.append(name)  # kept for the exit status
    print(f"  [{'REPORT' if ok is None else ('PASS' if ok else 'FAIL')}] {name}: {detail}")  # printed for the run log


def load_status() -> pd.DataFrame:  # function: the gain columns of the status table, main antenna only, with whole seconds
    st = pd.read_csv(PROC / "ground_truth" / "receiver_status_timeseries.csv", usecols=["test_code", "receiver", "timestamp_utc", "band", "agc_gain_db", "agc_cnt"], low_memory=False)  # gains only
    st = st[st.test_code.isin(SESSIONS) & ~st.band.str.contains("Aux", na=False)].copy()  # the four sessions; the mosaic-H's Aux1 antenna left out
    st["sec"] = pd.to_datetime(st.timestamp_utc, utc=True).astype("int64") // 10 ** 9  # nanoseconds since 1970 // 1e9: whole seconds (floor), as the gain-pattern script
    return st  # the table


def step_events(y: pd.DataFrame, bands: list, field: str, thr: float, a: int, b: int) -> list:  # function: the gain-pattern script's boundary rule on the mean of some bands, as (second, change) events
    g = y[y.band.isin(bands)].groupby("sec")[field].mean()  # the bands' mean gain per second
    s = g.reindex(np.arange(a, b + 1)).interpolate(limit=INTERP_S).rolling(MED_S, center=True, min_periods=2).median()  # method chain: full grid, gaps to 6 s filled, centred 5 s median
    d = s.shift(-HALF_S) - s.shift(HALF_S)  # change from t-3 s to t+3 s (shift(-3) is the value 3 s later)
    big = d[d.abs() >= thr]  # boolean selection: the step seconds
    out, last = [], -999  # events; the previous step second
    for k, v in big.items():  # for loop over the step seconds in time order
        if k - last > GAP_S:  # if: more than 20 s after the previous step second, a new event
            out.append((int(k), float(v)))  # event: (second, signed change)
        last = k  # every step second resets the 20 s clock (the gain-pattern script's rule)
    return out  # the events


def gate_inputs(st: pd.DataFrame) -> bool:  # function: gate R0, the implementation reproduces the gain-pattern script's committed X5 events row for row
    ref = pd.read_csv(PROC / "ground_truth" / "phase_boundaries.csv")  # the gain-pattern script's table
    mine = []  # this script's events, in the gain-pattern script's row order
    for t in ["2.6.5", "2.6.4", "2.6.11", "2.6.12"]:  # for loop over the gain-pattern script's sessions, in its order
        y = st[(st.receiver == "Septentrio_02") & (st.test_code == t)]  # the X5 in the session
        a, b = int(y.sec.min()), int(y.sec.max())  # the span of its status data (every band)
        for name, bands in (("L1 group (L1/E1, B1)", L1_05H), ("continuous (L2, L5, E5b)", CONT_05H)):  # for loop over the gain-pattern script's two groups
            mine += [(t, name, hms(k), round(v, 1), "down" if v < 0 else "up") for k, v in step_events(y, bands, "agc_gain_db", 3.0, a, b)]  # list comprehension: the gain-pattern script's columns
    theirs = list(zip(ref.test_code, ref.group, ref.event_utc, ref.change_db, ref.direction))  # zip: the gain-pattern script's rows as tuples
    ok = mine == theirs  # equal lists: same events, values and order
    check("R0 the gain-pattern script's rule reproduces ground_truth/phase_boundaries.csv on the X5 (both groups, four sessions)", ok, f"{len(mine)} events against {len(theirs)} committed")  # the gate
    return ok  # passed or not


def windows() -> dict:  # function: per session, capture start, scheduled start and the scheduled opening-phase length in seconds
    gt = pd.read_csv(PROC / "ground_truth" / "session_ground_truth.csv").set_index("test_code")  # sessions
    ph = pd.read_csv(PROC / "ground_truth" / "test_phases.csv")  # the catalogue's phases
    W = {}  # dict: session -> times
    for t in SESSIONS:  # for loop over the four sessions
        r = ph[(ph.test_code == t) & (ph.phase_name == "simultaneous spoofing and jamming")].iloc[0]  # the opening phase row (as the label script's jam_phases)
        W[t] = {"cs": T(gt.at[t, "capture_start_utc"][11:]), "ss": T(gt.at[t, "scheduled_start_utc"][11:]), "len": int(r.offset_end_min * 60)}  # capture start, scheduled start, 300 s
    return W  # the windows


def phase(t: str, rx: str, ev: list, nonev: dict, w: dict) -> tuple:  # function: the receiver's opening phase under (a)-(c), or None, and the rejected candidates
    ss, jend = w["ss"], w["ss"] + w["len"]  # scheduled start; scheduled end of the opening phase (12:35:00 in 2.6.5)
    downs = [(k, v) for k, v in ev if v < 0 and ss - HALF_S <= k <= jend + JAM_TAIL_S]  # list comprehension: down steps allowed by (a)
    cands = sorted(downs)  # candidates in time order
    if ss < w["cs"] <= jend:  # if: the capture starts inside the scheduled phase (2.6.11): its start stands for the unseen down step
        cands = [(w["cs"], float("nan"))] + cands  # list concatenation: the capture start first
    rejected = []  # why each earlier candidate failed
    for a, va in cands:  # for loop over the candidates in time order
        ups = [(k, v) for k, v in ev if v > 0 and k > a]  # list comprehension: feature-group up steps after it
        if not ups:  # if: none
            rejected.append(f"{hms(a)}: no up step")  # recorded
            continue  # next candidate
        b, vb = ups[0]  # the first up step after the down step
        if b - a > w["len"] + JAM_TAIL_S:  # if: (c) longer than the scheduled phase plus the tail
            rejected.append(f"{hms(a)}-{hms(b)}: (c) {b - a} s > {w['len'] + JAM_TAIL_S:.0f} s")  # recorded
            continue  # next candidate
        nf = [f"{band} {hms(k)}" for band, e in nonev.items() for k, v in e if v > 0 and abs(k - b) <= HALF_S]  # list comprehension: non-feature up events within 3 s of the feature up event
        if nf:  # if: (b) a non-feature band recovers too
            rejected.append(f"{hms(a)}-{hms(b)}: (b) {', '.join(nf)}")  # recorded
            continue  # next candidate
        return {"start": int(a), "end": int(b) - 1, "down_utc": hms(a), "down_change": va, "up_utc": hms(b), "up_change": vb}, rejected  # the phase [down, up - 1]
    return None, rejected  # no candidate accepted


def main() -> None:  # function: events, phases, union, gates, outputs
    st = load_status()  # gains
    print("== gate R0: the gain-pattern script's rule on the X5 ==")  # header
    r0 = gate_inputs(st)  # the implementation gate
    W = windows()  # the windows
    rows, evrows = [], []  # per-receiver rows; every event
    for t in SESSIONS:  # for loop over the sessions
        for rx, (tag, fld, thr) in RX.items():  # for loop over the receivers
            y = st[(st.receiver == tag) & (st.test_code == t)]  # its main-antenna readings in the session
            a, b = int(y.sec.min()), int(y.sec.max())  # its span (every band of the receiver)
            ev = step_events(y, FEATURE[rx], fld, thr, a, b)  # the ruled feature group's events
            ev5 = step_events(y, WITHDRAWN[rx], fld, thr, a, b) if rx != "UBX_01" else ev  # conditional expression: the withdrawn three-band group, reported (the u-blox has one L1 block)
            nonev = {bd: step_events(y, [bd], fld, thr, a, b) for bd in NONFEATURE[rx]}  # dict comprehension: each non-feature band's events
            evrows += [{"test_code": t, "receiver": rx, "group": g, "event_utc": hms(k), "change": round(v, 1), "direction": "down" if v < 0 else "up"}  # list comprehension: event rows
                       for g, e in [("feature (the gain-pattern script two-band)", ev), ("feature (three-band, withdrawn)", ev5)] + list(nonev.items()) for k, v in e]  # every group and band
            p, rej = phase(t, rx, ev, nonev, W[t])  # the ruled phase
            p5, _ = phase(t, rx, ev5, nonev, W[t])  # the withdrawn three-band group's phase under the same (a)-(c) (reported only)
            row = {"test_code": t, "receiver": rx, "basis": "gain step events (L1)" if p else "none read", "start_utc": hms(p["start"]) if p else "", "end_utc": hms(p["end"]) if p else "",  # conditional expressions: empty when none
                   "start_sec": p["start"] if p else np.nan, "end_sec": p["end"] if p else np.nan, "down_event_utc": p["down_utc"] if p else "", "down_change": p["down_change"] if p else np.nan,  # the edges
                   "up_event_utc": p["up_utc"] if p else "", "up_change": p["up_change"] if p else np.nan, "rejected_candidates": "; ".join(rej),  # the up edge and the rejections
                   "start_three_band_withdrawn_utc": hms(p5["start"]) if p5 else "", "end_three_band_withdrawn_utc": hms(p5["end"]) if p5 else ""}  # dict literal: the withdrawn group's phase beside
            rows.append(row)  # kept
        got = [r for r in rows if r["test_code"] == t and r["basis"] != "none read"]  # list comprehension: the receivers that read a phase
        sched = (W[t]["ss"], W[t]["ss"] + W[t]["len"] - 1)  # the schedule's phase, inclusive (the label script's jam_phases)
        if got:  # if: any receiver reads one: the union at the antenna
            iv = sorted((int(r["start_sec"]), int(r["end_sec"])) for r in got)  # generator in sorted(): the intervals by start
            un = [list(iv[0])]  # the union, built left to right
            for a, b in iv[1:]:  # for loop over the others
                if a <= un[-1][1] + 1:  # if: it overlaps or touches the last union piece
                    un[-1][1] = max(un[-1][1], b)  # extends it
                else:  # else: disjoint
                    un.append([a, b])  # a new piece
            basis = "union of " + "+".join(r["receiver"] for r in got)  # string: who read it
        else:  # else: nobody reads one
            un, basis = [list(sched)], "schedule (fallback)"  # the schedule
        for a, b in un:  # for loop over the antenna pieces
            rows.append({"test_code": t, "receiver": "antenna", "basis": basis, "start_utc": hms(a), "end_utc": hms(b), "start_sec": a, "end_sec": b})  # one antenna row
    out = pd.DataFrame(rows)  # the table
    out["n_seconds"] = out.end_sec - out.start_sec + 1  # inclusive length
    out["sched_start_utc"] = out.test_code.map(lambda t: hms(W[t]["ss"]))  # lambda in map: the scheduled start beside
    out["sched_end_utc"] = out.test_code.map(lambda t: hms(W[t]["ss"] + W[t]["len"] - 1))  # the schedule's last second of the phase
    out["start_minus_sched_s"] = out.start_sec - out.test_code.map(lambda t: W[t]["ss"])  # offset of the start from the schedule
    out["end_minus_sched_s"] = out.end_sec - out.test_code.map(lambda t: W[t]["ss"] + W[t]["len"] - 1)  # offset of the end
    pd.DataFrame(evrows).to_csv(CHECK_DIR / "jam_events.csv", index=False, lineterminator="\n")  # every event, written whatever the gates give
    print(out.drop(columns=["start_sec", "end_sec"]).to_string(index=False))  # the phases, for the run log
    x5 = out[(out.test_code == "2.6.5") & (out.receiver == "SBF_02")].iloc[0]  # the X5's 2.6.5 row
    g1 = x5.basis != "none read" and abs(int(x5.start_sec) - T(GATE[0])) <= GATE_TOL_S and abs(int(x5.end_sec) - T(GATE[1])) <= GATE_TOL_S  # within 1 s at both edges
    if g1:  # if: the gate passes
        check("G1 the X5's 2.6.5 phase within 1 s of 12:36:00-12:40:00", g1, f"{x5.start_utc}-{x5.end_utc} ({x5.basis})")  # the gate
    else:  # else: the decision: no pause; the registered phase in 2.6.5, the schedule elsewhere
        CHECKS.append({"check": "G1 the X5's 2.6.5 phase within 1 s of 12:36:00-12:40:00", "result": "FAIL: fallback applied", "detail": f"{x5.start_utc}-{x5.end_utc} ({x5.basis})"})  # recorded, not a stop
        ov = pd.read_csv(PROC / "ground_truth" / "group_overrides.csv")  # the registered observed phase of 2.6.5
        o = ov[(ov.test_code == "2.6.5") & (ov.group == "jamming")].iloc[0]  # 12:36:00-12:40:00 (the same on the three receivers)
        fb = [{"test_code": t, "receiver": "antenna", "basis": "fallback: group_overrides" if t == "2.6.5" else "fallback: schedule",  # one antenna row per session
               "start_sec": int(o.start_sec) if t == "2.6.5" else W[t]["ss"], "end_sec": int(o.end_sec) if t == "2.6.5" else W[t]["ss"] + W[t]["len"] - 1} for t in SESSIONS]  # list comprehension: the fallback phases
        out = pd.concat([out[out.receiver != "antenna"], pd.DataFrame(fb).assign(start_utc=lambda x: x.start_sec.map(hms), end_utc=lambda x: x.end_sec.map(hms))], ignore_index=True)  # per-receiver rows kept as measured, antenna rows replaced
        out["n_seconds"] = out.end_sec - out.start_sec + 1  # lengths again
        out["start_minus_sched_s"] = out.start_sec - out.test_code.map(lambda t: W[t]["ss"])  # offsets again
        out["end_minus_sched_s"] = out.end_sec - out.test_code.map(lambda t: W[t]["ss"] + W[t]["len"] - 1)  # and of the end
        print("  [FAIL] G1: fallback applied")  # printed
    ant = out[out.receiver == "antenna"]  # the antenna rows
    check("one antenna phase or more per session", set(ant.test_code) == set(SESSIONS), f"{len(ant)} antenna rows")  # every session has one
    pd.DataFrame(CHECKS).to_csv(CHECK_DIR / "jam_checks.csv", index=False, lineterminator="\n")  # the checks
    if not r0:  # if: the implementation gate failed (G1 has its fallback)
        raise SystemExit("STOP: gate R0 failed (jam_checks.csv); ground_truth/jam_phase_boundaries.csv is not written and no final label is built")  # stop before any label
    out.to_csv(PROC / "ground_truth" / "jam_phase_boundaries.csv", index=False, lineterminator="\n")  # written


if __name__ == "__main__":  # if: true only when the file is run directly
    main()  # runs L1
    exit_if_failed(FAILED, "1_ground_truth/06_jamming_phases.py")  # the helpers module: status 1 when any check failed
