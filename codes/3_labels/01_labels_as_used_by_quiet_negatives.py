#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
The label script as it stood when the quiet-negative study (3_labels/02_quiet_negatives.py) was made, kept unchanged so that study
can run. The study rebuilds the label script's build() in memory with three switches (the guard at the G30 and G08 exclusions and the
Stave stop relabel) by replacing three lines of its source text; those lines are written here with the clock times as literals
(13:02:41, 14:44:41-14:44:43, 12:27:22-12:34:57). The current label script (3_labels/01_labels.py) reads the same instants from
Data/processed/ground_truth/decision_times.csv and computes the same labels, but its source text no longer carries those lines, so
the study loads this version instead. Nothing runs when this file is run; it is only loaded by 02_quiet_negatives.py. The labels the
thesis uses are written by 3_labels/01_labels.py.
"""

from __future__ import annotations  # future import: type hints kept as text (the project's convention)

import os  # operating-system access: MPLCONFIGDIR is set before matplotlib loads
import sys  # system access: sys.path and sys.modules
import importlib.util  # standard-library tools to load a Python file whose name starts with a digit
from pathlib import Path  # Path objects for building file paths

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
os.environ["MPLCONFIGDIR"] = str(PROJECT / "codes" / ".mplconfig")  # keeps matplotlib's cache inside the project folder

import numpy as np  # NumPy: boolean arrays over seconds
import pandas as pd  # pandas: tables, CSV reading and writing
import matplotlib  # the plotting library itself, to choose a backend
matplotlib.use("Agg")  # function call: a file-only backend, so no window opens and runs again are identical
import matplotlib.pyplot as plt  # pyplot: figures and axes
import matplotlib.dates as mdates  # date handling on the time axis
from matplotlib.patches import Patch  # coloured boxes for a hand-made legend

sys.path.insert(0, str(PROJECT / "codes" / "common"))  # list method insert: puts codes/ first on the import path so the helpers module is found
from helpers import save_fig  # the project's byte-stable figure saver

SEED = 20260914  # integer constant: the project's fixed seed (nothing here draws a random number)
np.random.seed(SEED)  # seeds NumPy's generator for form's sake, as every script does
PROC = PROJECT / "Data" / "processed"  # the '/' operator on a Path joins parts: the processed-data folder
GRAPH = PROJECT / "Graph"  # where the figures go
CHECK_DIR = PROJECT / "Data" / "processed" / "checks"  # where the self-checks go
DATE = "2026-09-14"  # the one recording date
SESSIONS = ["2.6.5", "2.6.2", "2.6.10", "2.6.4", "2.6.11", "2.6.12"]  # list: the six sessions in time order
LETTER = dict(zip(SESSIONS, "abcdef"))  # dict built by zip: session -> the letter of its the figure panel file
RXS = ["SBF_02", "SBF_03", "UBX_01"]  # list: the three receivers on the shared antenna
NAME = {"SBF_02": "mosaic-X5", "SBF_03": "mosaic-H main", "UBX_01": "u-blox 01"}  # dict: stream -> readable name
STATUS_RX = {"SBF_02": "Septentrio_02", "SBF_03": "Septentrio_03", "UBX_01": "u-blox_01"}  # dict: stream -> status-table tag
GAIN_KEY = {"SBF_02": ("GPSL1/E1", "agc_gain_db"), "SBF_03": ("GPSL1/E1", "agc_gain_db"), "UBX_01": ("L1", "agc_cnt")}  # dict: the L1 gain (band, field) of each receiver, main antenna only
GAIN_DROP = {"agc_gain_db": 5.0, "agc_cnt": 400.0}  # dict: the receiver-status script's entry drop: 5 dB (Septentrio) or 400 counts (u-blox)
GUARD_S = 30  # integer constant: the label guard of 34a, seconds
MERGE_S = 30  # integer constant: evidence intervals closer than this many seconds are merged
SHORT_S = 30  # integer constant: a gain run shorter than this is a "short dip", and a gap without observations shorter than this sets no guard
POS_M = 50.0  # float constant: own-position departure distance, metres
HOLD_S = 10.0  # float constant: a departure must last this many seconds
POS_GAP_S = 5.0  # float constant: fixes further apart than this break a departure run (the evidence script's convention)
FEATURE_IN = {"L1", "E1", "B1", "B1I", "B1C", "G1"}  # set: band names that make a concurrent test a feature-band test (as in 05g)
MARGIN_CONTAMINATION_START = "15:45:12"  # 2.6.12: no clean second before this time (equal to the receiver-status script's computed split, checked below)
POST_265_END = "12:54:24"  # 2.6.5: the fixed uncertain interval of runs from the end of the pooled attack interval to here
LABELS = ["clean", "attack_jam_spoof", "attack_spoof", "uncertain", "excluded"]  # list: the five labels, in reporting order
COLOR = {"clean": "#009E73", "attack_jam_spoof": "#7A0019", "attack_spoof": "#D55E00", "uncertain": "#BBBBBB", "excluded": "#333333"}  # dict: label -> colour
RXCOL = {"SBF_02": "#0072B2", "SBF_03": "#E69F00", "UBX_01": "#009E73"}  # dict: receiver -> line colour (Okabe-Ito, as the position script)
CHECKS: list = []  # list: one dict per self-check, written at the end


def load_module(name: str, path: Path):  # function: loads a .py file as a module (the scripts' names start with a digit)
    spec = importlib.util.spec_from_file_location(name, path)  # where the module comes from
    mod = importlib.util.module_from_spec(spec)  # an empty module object built from that spec
    sys.modules[name] = mod  # registered, so the module's own imports resolve
    spec.loader.exec_module(mod)  # runs the file's top level; its main() is behind a __name__ guard and does not run
    return mod  # the loaded module


def T(hms: str) -> int:  # function: "HH:MM:SS(.fff)" on the recording date -> whole unix seconds, floored
    return int(np.floor(pd.Timestamp(f"{DATE} {hms}", tz="UTC").timestamp()))  # parse, seconds since 1970, floor to the second


def hms(sec) -> str:  # function: unix seconds -> "HH:MM:SS" text
    return pd.Timestamp(int(sec), unit="s", tz="UTC").strftime("%H:%M:%S")  # integer seconds formatted as clock time


def check(name: str, ok: bool, detail: str) -> None:  # function: records one self-check and prints it
    CHECKS.append({"check": name, "result": "PASS" if ok else "FAIL", "detail": detail})  # conditional expression picks the word
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")  # printed for the run log


S03 = load_module("receiver_status_for_labels", PROJECT / "codes" / "1_ground_truth" / "03_receiver_status.py")  # the receiver-status script, unchanged: margin rule and disturbed runs
D05 = load_module("gain_evidence_for_labels", PROJECT / "codes" / "1_ground_truth" / "04_gain_and_position_evidence.py")  # the evidence script, unchanged: the receivers' own positions and site geometry


# ---------------------------------------------------------------- inputs
def load_inputs() -> dict:  # function: reads every input table once
    gt = pd.read_csv(PROC / "ground_truth" / "session_ground_truth.csv")  # the timeline script's sessions: capture and scheduled windows
    gt["_o"] = pd.to_datetime(gt.capture_start_utc)  # helper column: capture start as datetime (03 sorts its sessions by it)
    gt = gt.sort_values("_o").reset_index(drop=True)  # capture order, renumbered, exactly as the receiver-status script's main() builds its sessions
    ep = pd.read_csv(PROC / "features" / "observables_epochs_1hz.csv.gz", usecols=["test_code", "receiver", "sec", "has_observations"])  # the grid: four columns only
    tl = pd.read_csv(PROC / "ground_truth" / "early_warning_timeline.csv", usecols=["test_code", "receiver", "gain_onset_section3", "gain_recovery_section3"])  # the receiver-status gain onset and recovery only
    eai = pd.read_csv(PROC / "ground_truth" / "effective_attack_interval.csv")  # the receiver-status interval table: quiet references and run counts
    # list over two lines: the status columns this script reads (gain, noise, CW suppression, flags and identity)
    cols = ["test_code", "receiver", "receiver_family", "timestamp_utc", "band", "agc_gain_db", "blanking_pct", "agc_cnt", "cw_suppression", "noise_per_ms",
            "rf_flag_notauthentic", "rf_flag_nma_fail", "jamming_state", "exposure_group", "inside_our_capture"]  # list: the status columns used
    st = pd.read_csv(PROC / "ground_truth" / "receiver_status_timeseries.csv", usecols=cols, low_memory=False)  # the receivers' status (gain, flags)
    st["timestamp_utc"] = pd.to_datetime(st.timestamp_utc, utc=True)  # text -> timezone-aware timestamps, as 03 held them
    sp = pd.read_csv(PROC / "ground_truth" / "spoof_2612_timeline.csv")  # the receiver-status script's 2.6.12 event table (own clock of u-blox 01)
    ov = pd.read_csv(PROC / "ground_truth" / "group_overrides.csv")  # the observed jamming phase of 2.6.5
    ct = pd.read_csv(PROC / "ground_truth" / "concurrent_transmissions.csv")  # concurrent tests at the other test areas
    ph = pd.read_csv(PROC / "ground_truth" / "test_phases.csv")  # the catalogue's phase structure per test
    return {"gt": gt, "ep": ep, "tl": tl, "eai": eai, "st": st, "sp": sp, "ov": ov, "ct": ct, "ph": ph}  # dict of tables


def windows(gt: pd.DataFrame) -> dict:  # function: per session, capture, scheduled window and scheduled jamming phase in whole seconds
    W = {}  # dict: session -> its times
    for r in gt.itertuples():  # for loop over sessions (itertuples gives one named tuple per row)
        W[r.test_code] = {"cs": T(r.capture_start_utc[11:]), "ce": T(r.capture_end_utc[11:]),  # capture start and end (both inclusive)
                          "ss": T(r.scheduled_start_utc[11:]), "se": T(r.scheduled_end_utc[11:])}  # scheduled window (both inclusive, as Tables 32 and 35)
    return W  # the windows


def jam_phases(ph: pd.DataFrame, ov: pd.DataFrame, W: dict) -> dict:  # function: per session, the jamming-plus-spoofing phase used for the phase name
    J = {}  # dict: session -> list of (start, end) inclusive, in whole seconds
    for t in SESSIONS:  # for loop over sessions
        r = ph[(ph.test_code == t) & (ph.phase_name == "simultaneous spoofing and jamming")].iloc[0]  # the catalogue's opening phase row
        sched = [] if bool(r.overridden_by_test) else [(W[t]["ss"], W[t]["ss"] + int(r.offset_end_min * 60) - 1)]  # conditional expression: none where the test exempts itself (2.6.2, 2.6.10), else the first five minutes
        J[t] = {"scheduled": sched, "used": sched, "basis": "schedule (fallback)" if sched else "none (test has no opening jamming)"}  # default: the schedule
    o = ov[(ov.test_code == "2.6.5") & (ov.group == "jamming")].iloc[0]  # the observed phase of 2.6.5 (same for the three receivers)
    J["2.6.5"].update(used=[(int(o.start_sec), int(o.end_sec))], basis="observed per-band gain (ground_truth/group_overrides.csv)")  # 12:36:00-12:40:00, end inclusive as the effects script reads it
    J["2.6.12"].update(basis="observed per-band gain agrees with the schedule")  # 15:50:00-15:54:59: gain and schedule agree within 5 s
    return J  # the phases


# ---------------------------------------------------------------- the receiver-status script's indicator rule, per point
def status_long(st: pd.DataFrame) -> pd.DataFrame:  # function: the long table of the receiver-status script's main() rebuilt from the status CSV it wrote
    ids = ["test_code", "receiver", "receiver_family", "timestamp_utc", "band", "exposure_group", "inside_our_capture"]  # list: identity columns, as 03
    sept = st[st.receiver_family == "Septentrio"]  # the Septentrio rows
    ubx = st[st.receiver_family == "u-blox"]  # the u-blox rows
    a = sept.melt(id_vars=ids, value_vars=list(S03.SEPT_TEST_FIELDS), var_name="field", value_name="value")  # melt: wide -> long, the receiver-status script's Septentrio test fields
    b = ubx.melt(id_vars=ids, value_vars=list(S03.UBX_TEST_FIELDS), var_name="field", value_name="value")  # the same for the receiver-status script's u-blox test fields
    long = pd.concat([a, b], ignore_index=True)  # stacked, index renumbered, as 03
    long["value"] = pd.to_numeric(long["value"], errors="coerce")  # numbers; anything else -> NaN, as 03
    return long[long.value.notna()]  # missing values dropped, as 03


def indicator_mask(long: pd.DataFrame, sessions: pd.DataFrame, split: tuple) -> tuple:  # function: the receiver-status script's margin_clean with its per-point mask kept
    # This repeats margin_clean's references, grid and rules line for line and keeps bad_any, which margin_clean discards;
    # the check in main() proves that its minutes equal ground_truth/margin_clean_minutes.csv and a fresh S03.margin_clean() call.
    ind = ([(STATUS_RX["SBF_02"], b, "agc_gain_db", "gain") for b in ("GPSL1/E1", "GLOL1", "B1")]  # list comprehension: mosaic-X5 gains
           + [(STATUS_RX["SBF_03"], b, "agc_gain_db", "gain") for b in ("GPSL1/E1", "GLOL1", "B1")]  # mosaic-H main-antenna gains
           + [("u-blox_01", "L1", "noise_per_ms", "iqr"), ("u-blox_01", "L1", "cw_suppression", "iqr")])  # the u-blox indicators in the combination
    ser, refs = {}, {}  # indicator -> its series; indicator -> (reference, IQR)
    for rx, band, fld, rule in ind:  # for loop over the combined indicators
        g = long[(long.receiver == rx) & (long.band == band) & (long.field == fld)].sort_values("timestamp_utc")  # this indicator's readings in time order
        ser[(rx, band, fld)] = g  # kept
        meds, pooled = [], []  # per-session margin medians, pooled margin values
        for _, sr in sessions.iterrows():  # for loop over sessions
            parts = [g[(g.timestamp_utc >= a) & (g.timestamp_utc < b)] for a, b in S03.margin_parts(sr) if b > a]  # list comprehension: readings in each non-empty margin
            m = pd.concat(parts) if parts else g.iloc[0:0]  # the session's margin readings
            if len(m):  # if: any
                meds.append(m.value.median()); pooled.append(m.value)  # two statements: the median and the values
        allm = pd.concat(pooled) if pooled else pd.Series(dtype=float)  # every margin reading of this indicator
        q = float(allm.quantile(0.75) - allm.quantile(0.25)) if len(allm) else 0.0  # pooled interquartile range
        refs[(rx, band, fld)] = (float(np.median(meds)), q if q > 0 else 1.0)  # (reference, IQR; 0 -> 1), as 03
    parts_out = []  # list of dicts: one per session and margin part, with the grid and its mask
    for _, sr in sessions.iterrows():  # for loop over sessions
        parts = S03.margin_parts(sr)  # its two margins
        if sr["test_code"] == split[0] and parts[0][0] < split[1] < parts[0][1]:  # if: the pre-test margin is cut at the possibly contaminated 2.6.12 margin time
            parts = [(parts[0][0], split[1]), (split[1], parts[0][1]), parts[1]]  # the pre-test margin in two parts, as 03
        for a, b in parts:  # for loop over its margins
            if b <= a:  # if: this margin is empty
                continue  # skip
            grid = pd.date_range(a, b, freq="100ms", inclusive="left")  # the 0.1 s grid over this margin
            bad_any = np.zeros(len(grid), bool)  # contaminated by any combined indicator
            cov_any = np.zeros(len(grid), bool)  # covered by any combined indicator
            for rx, band, fld, rule in ind:  # for loop over indicators
                ref, q = refs[(rx, band, fld)]  # its reference and IQR
                g = ser[(rx, band, fld)]  # its series
                gg = g[(g.timestamp_utc >= a - pd.Timedelta(seconds=60)) & (g.timestamp_utc < b)].set_index("timestamp_utc").value  # readings from 60 s before the margin
                gg = gg[~gg.index.duplicated()]  # one reading per time
                v = (gg.reindex(grid, method="ffill", tolerance=pd.Timedelta(seconds=60)).to_numpy() if len(gg) else np.full(len(grid), np.nan))  # each point takes the last reading, at most 60 s old
                cov = ~np.isnan(v)  # boolean: a reading covers this point
                bad = (v < ref - 3.0) if rule == "gain" else (v > ref + 3 * q)  # conditional expression: the indicator's contamination rule
                bad_any |= bad & cov; cov_any |= cov  # two statements: "|=" accumulates the logical or
            parts_out.append({"test_code": sr["test_code"], "a": a, "b": b, "grid": grid, "bad": bad_any, "cov": cov_any})  # kept
    return parts_out, refs  # the parts and the references


def indicator_seconds(parts: list) -> dict:  # function: per session, the set of margin seconds that pass every indicator and those that fail
    out = {}  # dict: session -> {"pass": set, "fail": set, "empty": set}
    for p in parts:  # for loop over margin parts
        d = out.setdefault(p["test_code"], {"pass": set(), "fail": set()})  # dict method setdefault: the session's entry, created on first use
        sec = (p["grid"].asi8 // 1_000_000_000).astype(np.int64)  # nanoseconds since 1970 -> whole seconds (integer division)
        bad_sec = set(sec[p["bad"]].tolist())  # seconds with at least one contaminated point
        for s in set(sec.tolist()):  # for loop over the seconds of this part
            (d["fail"] if s in bad_sec else d["pass"]).add(s)  # conditional expression picks the set; the second is filed
    for d in out.values():  # for loop over sessions
        d["pass"] -= d["fail"]  # a second split across two parts fails if either part fails
    return out  # the per-session sets


# ---------------------------------------------------------------- independent evidence
def gain_series(long: pd.DataFrame, test: str, rx: str) -> pd.Series:  # function: one receiver's L1 gain in one session, indexed by time, as 03 builds it
    band, fld = GAIN_KEY[rx]  # tuple unpacking: its band and field
    g = long[(long.test_code == test) & (long.receiver == STATUS_RX[rx]) & (long.band == band) & (long.field == fld)]  # boolean mask: those readings
    return g.set_index("timestamp_utc")["value"].sort_index()  # a time-indexed series


def gain_evidence(d: dict, long: pd.DataFrame, W: dict, eai: pd.DataFrame) -> tuple:  # function: every gain interval per session and receiver, and the run checks
    ev, rep = [], []  # evidence rows; the run-count reproduction rows
    for r in d["tl"].itertuples():  # for loop over the 18 session x receiver rows of the timeline
        if isinstance(r.gain_onset_section3, str) and isinstance(r.gain_recovery_section3, str):  # if: the receiver-status script found an onset and a recovery
            ev.append({"test_code": r.test_code, "receiver": r.receiver, "start": T(r.gain_onset_section3), "end": T(r.gain_recovery_section3), "source": "gain_onset_recovery"})  # floored, recovery second included
    for t in SESSIONS:  # for loop over sessions
        w = W[t]  # its windows
        cs, ce = pd.Timestamp(w["cs"], unit="s", tz="UTC"), pd.Timestamp(w["ce"], unit="s", tz="UTC")  # capture start and end as timestamps
        lo, hi = pd.Timestamp(max(w["cs"], w["ss"]), unit="s", tz="UTC"), pd.Timestamp(min(w["ce"], w["se"]), unit="s", tz="UTC")  # captured part of the window, as 03
        for rx in RXS:  # for loop over receivers
            band, fld = GAIN_KEY[rx]  # its band and field
            ref = float(eai[(eai.test_code == t) & (eai.receiver == STATUS_RX[rx]) & (eai.field == fld)].quiet_reference.iloc[0])  # the receiver-status script's quiet reference
            s = gain_series(long, t, rx)  # its gain
            whole = s[(s.index >= cs) & (s.index <= ce)]  # the whole capture, as 03
            ins = s[(s.index >= lo) & (s.index < hi)]  # inside the captured window, as 03
            wr = S03.disturbed_runs(whole, ref, GAIN_DROP[fld], -1)  # the receiver-status script's disturbed runs over the whole capture (falls only)
            ir = S03.disturbed_runs(ins, ref, GAIN_DROP[fld], -1) if len(ins) >= 30 else []  # and inside the window, for the reproduction check
            e = eai[(eai.test_code == t) & (eai.receiver == STATUS_RX[rx]) & (eai.field == fld)].iloc[0]  # the receiver-status script's row
            lw = max([(pd.Timestamp(b) - pd.Timestamp(a)).total_seconds() / 60 for a, b in wr], default=0.0)  # longest whole-capture run, minutes
            # dict literal over two lines: our run counts and longest run beside the receiver-status script's, for the reproduction check
            rep.append({"test_code": t, "receiver": rx, "runs_whole": len(wr), "runs_whole_s3": int(e.n_passages_whole_capture), "runs_window": len(ir), "runs_window_s3": int(e.n_disturbed_runs),
                        "longest_whole_min": round(lw, 2), "longest_whole_min_s3": float(e.longest_passage_whole_capture_min)})  # the reproduction row
            for a, b in wr:  # for loop over the whole-capture runs
                ev.append({"test_code": t, "receiver": rx, "start": int(pd.Timestamp(a).timestamp() // 1), "end": int(pd.Timestamp(b).timestamp() // 1), "source": "gain_disturbed_run"})  # floored to whole seconds
            S03.MIN_SPAN_S = 0  # the receiver-status script's shortest kept run set to zero, for the short-dip search only
            try:  # try/finally: the module constant is restored even if the call fails
                sr_ = S03.disturbed_runs(whole, ref, GAIN_DROP[fld], -1)  # every run, however short
            finally:  # always runs
                S03.MIN_SPAN_S = SHORT_S  # restored to the receiver-status script's 30 s
            for a, b in sr_:  # for loop over all runs
                a_, b_ = int(pd.Timestamp(a).timestamp() // 1), int(pd.Timestamp(b).timestamp() // 1)  # whole seconds
                if (b_ - a_) < SHORT_S and a_ >= w["ss"] and b_ <= w["se"]:  # if: shorter than 30 s and inside the scheduled window
                    ev.append({"test_code": t, "receiver": rx, "start": a_, "end": b_, "source": "gain_short_dip"})  # counts as gain evidence
    o = d["ov"][d["ov"].group == "jamming"]  # the observed jamming phase rows of 2.6.5 (one per receiver)
    for r in o.itertuples():  # for loop over them
        ev.append({"test_code": r.test_code, "receiver": r.receiver, "start": int(r.start_sec), "end": int(r.end_sec), "source": "observed_jamming_phase"})  # gain evidence
    check("MIN_SPAN_S restored", S03.MIN_SPAN_S == 30, f"S03.MIN_SPAN_S = {S03.MIN_SPAN_S} after the short-dip search")  # the constant is back
    # Option B (decided 6 October 2026): a disturbed run counts only where that receiver has no onset-recovery interval
    # overlapping it, so onset and recovery set the edges wherever the receiver-status script timed them, and the runs add only the passages the held
    # 30 s onset could not time (the X5 in 2.6.2, the X5's opening-phase dip in 2.6.11)
    orr = [e for e in ev if e["source"] == "gain_onset_recovery"]  # list comprehension: the onset-recovery intervals
    keep = []  # the evidence kept
    for e in ev:  # for loop over every gain interval
        hit = any(o["test_code"] == e["test_code"] and o["receiver"] == e["receiver"] and o["start"] <= e["end"] and o["end"] >= e["start"] for o in orr)  # generator inside any(): an own onset-recovery overlaps it
        if e["source"] != "gain_disturbed_run" or not hit:  # if: not a run, or a run that the receiver-status script's onset did not time
            keep.append(e)  # kept
    return keep, pd.DataFrame(rep)  # evidence rows and the reproduction table


def departure_runs(sec: np.ndarray, dh: np.ndarray) -> list:  # function: runs of fixes beyond 50 m lasting at least 10 s, as (first, last) whole seconds
    runs, i, n = [], 0, len(sec)  # result list, position, count
    while i < n:  # while loop over the fixes in time order
        if dh[i] > POS_M:  # if: a fix beyond 50 m starts a run
            j = i  # the run's last fix so far
            while j + 1 < n and dh[j + 1] > POS_M and sec[j + 1] - sec[j] <= POS_GAP_S:  # extends while the next fix is far and close in time
                j += 1  # one fix on
            if sec[j] - sec[i] >= HOLD_S:  # if: the run lasted at least 10 s
                runs.append((int(sec[i]), int(sec[j])))  # kept: first fix to last fix beyond 50 m
            i = j + 1  # continues after the run
        else:  # else: a near fix
            i += 1  # next fix
    return runs  # the departures


def position_evidence(gt: pd.DataFrame, W: dict) -> tuple:  # function: own-position departures of the X5 and u-blox 01, and the distance series for the figure
    ev, dist = [], {}  # evidence rows; (session, receiver) -> distance table
    g2 = gt.set_index("test_code")  # sessions indexed by test code, as own_positions expects
    for t in SESSIONS:  # for loop over sessions
        site = D05.site_xyz(D05.M.SITE_OF[t])  # the session's site estimate, ECEF metres
        for rx in ("SBF_02", "UBX_01"):  # for loop over the receivers that output a position (the mosaic-H outputs none)
            own = D05.own_positions(g2, t, rx)  # first valid own fix of each second (u-blox on host time)
            a0, b0 = D05.capture_spans(g2, t, rx)[0]  # the evidence script's usable span: in 2.6.5 it starts at 12:34:57, when the vehicle stands at the site; end exclusive
            own = own[(own.sec >= a0) & (own.sec < b0)].reset_index(drop=True)  # inside it: a moving vehicle's distance from the site is not a departure
            dh = D05.horiz_from_site(own.lat.to_numpy(), own.lon.to_numpy(), own.h.to_numpy(), site) if len(own) else np.array([])  # horizontal distance from the site
            dist[(t, rx)] = pd.DataFrame({"sec": own.sec.to_numpy(), "dh": dh})  # kept for the figure
            for a, b in departure_runs(own.sec.to_numpy(), dh):  # for loop over departures
                ev.append({"test_code": t, "receiver": rx, "start": a, "end": b, "source": "own_position"})  # evidence row
    return ev, dist  # evidence and distances


def time_evidence(sp: pd.DataFrame) -> list:  # function: u-blox 01's own measurement clock in 2.6.12, first offset state to the measurement-clock release
    u = sp[sp.receiver == "u-blox_01"].set_index("event").host_time_utc  # u-blox 01's events by name
    a = T(u["measurement-engine deception"])  # the first offset state (15:51:17)
    b = T(u["measurement-engine release"])  # the release instant floored (16:04:57.997 -> 16:04:57), the last second still offset; the navigation-time lag is the receiver's memory and is not used
    return [{"test_code": "2.6.12", "receiver": "UBX_01", "start": a, "end": b, "source": "own_time"}]  # one evidence row


def merge(iv: list) -> list:  # function: merges (start, end) intervals whose gap is shorter than MERGE_S seconds
    out = []  # merged list
    for a, b in sorted(iv):  # for loop over intervals sorted by start
        if out and a - out[-1][1] - 1 < MERGE_S:  # if: fewer than 30 uncovered seconds lie between this and the last merged one
            out[-1][1] = max(out[-1][1], b)  # extends it
        else:  # else: a separate interval
            out.append([a, b])  # starts a new one
    return [tuple(x) for x in out]  # list comprehension: tuples


# ---------------------------------------------------------------- labelling
def build(t: str, rx: str, ev: pd.DataFrame, grid: pd.DataFrame, W: dict, J: dict, ind: dict, own_rx_only: bool, guard_all_noobs: bool = False, use_54c: bool = True, guard_nogrid: bool = False) -> pd.DataFrame:  # function: the labels of one session and receiver
    w = W[t]  # its windows
    sec = np.arange(w["cs"], w["ce"] + 1, dtype=np.int64)  # every capture second, inclusive
    n = len(sec)  # how many
    e = ev[(ev.test_code == t) & ((ev.receiver == rx) if own_rx_only else True)]  # conditional expression: this receiver's evidence (Option 1) or all three (pooled)
    pooled = merge(list(zip(e.start, e.end)))  # the attack intervals
    idx = lambda a, b: slice(max(a, w["cs"]) - w["cs"], min(b, w["ce"]) - w["cs"] + 1)  # lambda: (start, end) inclusive -> slice of the second array, clipped to the capture
    g = grid[(grid.test_code == t) & (grid.receiver == rx)].set_index("sec").has_observations  # this receiver's grid
    hasobs = g.reindex(sec).fillna(0).to_numpy().astype(bool)  # reindex: capture seconds missing from the grid count as without observations
    excl = np.full(n, "", dtype=object)  # exclusion reason per second ("" = not excluded)
    noobs = ~hasobs  # boolean array: no observations
    excl[noobs] = "no_observations"  # filed first; the specific reasons below overwrite it
    if t == "2.6.5":  # if: the session with the two drives
        excl[idx(T("12:27:22"), T("12:34:57"))] = "vehicle_motion"  # the drives and the stop at Stave
    if t == "2.6.2" and rx == "UBX_01":  # if: u-blox 01 in 2.6.2
        excl[idx(w["cs"], T("13:02:41") + GUARD_S)] = "invalid_G30"  # its invalid G30 to 13:02:41, plus 30 s
    if t == "2.6.10" and rx in ("SBF_02", "SBF_03"):  # if: a Septentrio receiver in 2.6.10
        excl[idx(T("14:44:41"), T("14:44:43") + GUARD_S)] = "G08_transient"  # the 3 s G08 transient, plus 30 s
    if t == "2.6.12" and rx == "UBX_01":  # if: u-blox 01 in 2.6.12
        excl[idx(T("16:04:58"), w["ce"])] = "ubx_clock_release"  # after its clock release (from 16:04:58 inclusive)
    att = np.zeros(n, bool)  # boolean array: inside a pooled attack interval
    for a, b in pooled:  # for loop over the attack intervals
        att[idx(a, b)] = True  # marked
    inwin = (sec >= w["ss"]) & (sec <= w["se"])  # boolean array: inside the scheduled window
    jam = np.zeros(n, bool)  # boolean array: inside the jamming-plus-spoofing phase used for the phase name
    for a, b in J[t]["used"]:  # for loop over that phase (none in 2.6.2 and 2.6.10)
        jam[idx(a, b)] = True  # marked
    unc = inwin.copy()  # uncertain: every in-window second to begin with (attack takes precedence later)
    for a, b in pooled:  # for loop over attack intervals: the seconds between the window and an interval that extends beyond it
        if b > w["se"]:  # if: the interval ends after the window
            unc[idx(w["se"] + 1, a - 1)] = True  # from the window end to the interval start (empty when the interval starts inside the window)
        if a < w["ss"]:  # if: the interval starts before the window
            unc[idx(b + 1, w["ss"] - 1)] = True  # from its end to the window start (empty when it reaches into the window)
    post = np.zeros(n, bool)  # boolean array: the 2.6.5 fixed uncertain interval
    if t == "2.6.5" and use_54c and pooled:  # if: 2.6.5 and the rule is on
        post[idx(max(b for a, b in pooled) + 1, T(POST_265_END))] = True  # from the end of the pooled attack interval to 12:54:24 (counterfeit G30, part 0 (b))
    p = ind.get(t, {"pass": set()})["pass"]  # the seconds passing every receiver-status indicator
    passes = np.array([s in p for s in sec])  # list comprehension inside np.array: per second, pass or not
    f_ = ind.get(t, {"fail": set()})["fail"]  # the seconds failing an indicator
    nogrid = np.array([(s not in p) and (s not in f_) for s in sec]) & ~inwin  # outside the window but without any 0.1 s point: the receiver-status script's grid ends at the whole-second capture end, so the last capture second is never examined
    cand = (excl == "") & ~att & ~unc & ~post & passes  # candidate clean: not excluded, not attack, not uncertain, passing the indicators
    if t == "2.6.12":  # if: the Stave session
        cand &= sec >= T(MARGIN_CONTAMINATION_START)  # nothing before 15:45:12
    trig = ~cand  # boolean array: seconds that set the 30 s guard
    if not guard_all_noobs:  # if: the adopted rule (no guard around short gaps without observations)
        runs = np.flatnonzero(np.diff(np.r_[0, (excl == "no_observations").astype(int), 0]))  # np.r_ pads with zeros; diff marks run edges; flatnonzero lists them
        for a, b in zip(runs[0::2], runs[1::2]):  # for loop over runs: slicing [0::2] takes starts, [1::2] ends (exclusive)
            if b - a < SHORT_S:  # if: a gap shorter than 30 s
                trig[a:b] = False  # it sets no guard
    if not guard_nogrid:  # if: the rule used (an unexamined second is not evidence of contamination)
        trig &= ~(nogrid & (excl == "") & ~att & ~unc & ~post)  # it stays uncertain but sets no guard
    near = np.convolve(trig.astype(int), np.ones(2 * GUARD_S + 1, int), mode="same") > 0  # convolution with a 61-wide box: true within 30 s of a trigger
    clean = cand & ~near  # clean: candidates more than 30 s from every trigger
    lab = np.full(n, "uncertain", dtype=object)  # every second starts as uncertain
    lab[clean] = "clean"  # clean seconds
    lab[att] = np.where(jam[att], "attack_jam_spoof", "attack_spoof")  # np.where picks the attack label by phase
    lab[excl != ""] = "excluded"  # exclusion overrides everything
    phase = np.where(sec < w["ss"], "pre_test", np.where(sec > w["se"], "post_test", np.where(jam, "jam_spoof", "spoof")))  # nested np.where: the phase name of every second
    phase = np.where(att & ~inwin, np.where(jam, "jam_spoof", "spoof"), phase)  # attack seconds outside the window take the attack phase (2.6.5 passage, 2.6.10 tail)
    # pd.DataFrame from a dict of arrays: the per-second table, with two helper columns (dropped before writing) and the attack intervals
    return pd.DataFrame({"sec": sec, "label": lab, "phase": phase, "exclusion_reason": np.where(lab == "excluded", excl, ""), "in_scheduled_window": inwin, "indicator_pass": passes,
                         "_noobs": noobs, "_cand": cand, "_nogrid": nogrid & (excl == "") & ~att & ~unc & ~post}), pooled  # the per-second table (two helper columns dropped later) and the attack intervals


def concurrent(ct: pd.DataFrame, t: str, sec: np.ndarray) -> np.ndarray:  # function: per second, the concurrent feature-band tests on air (a column only)
    out = np.full(len(sec), "", dtype=object)  # empty text per second
    c = ct[ct.our_test_code == t]  # this session's concurrent tests
    c = c[c.bands.apply(lambda b: bool(FEATURE_IN & set(str(b).split(";"))))]  # those that list a feature band (as 05g)
    for r in c.sort_values("concurrent_test_code").itertuples():  # for loop over them, in a fixed order
        on = (sec >= T(r.concurrent_start_utc[11:])) & (sec < T(r.concurrent_end_utc[11:]))  # half-open window, as 05g
        out[on] = [x + (";" if x else "") + r.concurrent_test_code for x in out[on]]  # list comprehension: appends the id, separated by ";"
    return out  # the column


def flags(st: pd.DataFrame, t: str, rx: str, sec: np.ndarray) -> np.ndarray:  # function: per second, whether the receiver's own flag was set (a column only)
    f = D05.status_flags(st, t, rx)  # the evidence script's flag series: RFStatus not-authentic or NMA fail (Septentrio), jamming state >= 2 (u-blox)
    m = f.groupby("sec").on.max()  # per second: set if any reading in the second is set
    return m.reindex(sec).map({True: "1", False: "0"}).fillna("").to_numpy()  # "1", "0", or "" where there is no reading


def cover(ev: pd.DataFrame, t: str, sec: np.ndarray, rx=None, sources=None) -> np.ndarray:  # function: per second, whether any matching evidence interval covers it
    e = ev[ev.test_code == t]  # this session's evidence
    if rx is not None:  # if: one receiver asked for
        e = e[e.receiver == rx]  # its evidence only
    if sources is not None:  # if: some sources asked for
        e = e[e.source.isin(sources)]  # those only
    m = np.zeros(len(sec), bool)  # nothing covered yet
    for a, b in zip(e.start, e.end):  # for loop over intervals
        m |= (sec >= a) & (sec <= b)  # "|=": adds the interval
    return m  # the mask


def who(ev: pd.DataFrame, t: str, sec: np.ndarray, col: str) -> np.ndarray:  # function: per second, the receivers (col="receiver") or sources (col="source") whose evidence covers it
    e = ev[ev.test_code == t]  # this session's evidence
    out = np.full(len(sec), "", dtype=object)  # empty text per second
    for v in sorted(e[col].unique()):  # for loop over the distinct receivers or sources, sorted for a stable order
        m = cover(e[e[col] == v], t, sec)  # where this one covers
        out[m] = [x + (";" if x else "") + v for x in out[m]]  # list comprehension: appends its name
    return out  # the column


GAIN_SOURCES = ["gain_onset_recovery", "gain_disturbed_run", "gain_short_dip", "observed_jamming_phase"]  # list: the evidence sources that are the gain


def label_all(d: dict, ev: pd.DataFrame, W: dict, J: dict, ind: dict, **kw) -> tuple:  # function: the labels of every session and receiver, pooled and Option 1
    rows, pooled_iv = [], []  # per-second tables; pooled intervals for the evidence file
    for t in SESSIONS:  # for loop over sessions
        for rx in RXS:  # for loop over receivers
            L, pooled = build(t, rx, ev, d["ep"], W, J, ind, own_rx_only=False, **kw)  # the pooled rule
            L1, own = build(t, rx, ev, d["ep"], W, J, ind, own_rx_only=True, **kw)  # Option 1: this receiver's evidence only
            L.insert(0, "receiver", rx); L.insert(0, "test_code", t)  # two statements: identity columns first
            L["label_per_receiver_only"] = L1.label.to_numpy()  # Option 1's label beside the pooled one
            rows.append(L)  # kept
            if rx == RXS[0]:  # if: the first receiver (the pooled intervals are the same for all three)
                pooled_iv += [{"test_code": t, "receiver": "pooled", "start": a, "end": b, "source": "pooled_interval"} for a, b in pooled]  # list comprehension: one row each
    return pd.concat(rows, ignore_index=True), pooled_iv  # one table


def edge_sources(ev: pd.DataFrame, piv: list) -> list:  # function: for each pooled interval, which evidence sets its start and its end, and whether a disturbed run does
    for r in piv:  # for loop over pooled intervals (dicts, changed in place)
        e = ev[(ev.test_code == r["test_code"]) & (ev.start >= r["start"]) & (ev.end <= r["end"])]  # the evidence inside it
        s_ = e[e.start == r["start"]]  # rows that start where the interval starts
        e_ = e[e.end == r["end"]]  # rows that end where it ends
        r["start_source"] = ";".join(sorted(f"{x.receiver}:{x.source}" for x in s_.itertuples()))  # generator inside join: receiver:source, sorted
        r["end_source"] = ";".join(sorted(f"{x.receiver}:{x.source}" for x in e_.itertuples()))  # the same for the end
        fl = [k for k, v in (("start", s_), ("end", e_)) if (v.source == "gain_disturbed_run").any()]  # list comprehension: edges set by a run
        r["edge_from_disturbed_run"] = ";".join(fl)  # "", "start", "end" or "start;end"
    return piv  # the same list


def raw_edge_check(long: pd.DataFrame, ev: pd.DataFrame, piv: list, eai: pd.DataFrame) -> pd.DataFrame:  # function: every pooled edge set by a disturbed run, against the raw gain trace
    rows, panels = [], []  # table rows; data for the check figure
    for r in piv:  # for loop over pooled intervals
        for edge in r["edge_from_disturbed_run"].split(";") if r["edge_from_disturbed_run"] else []:  # conditional expression: the run-set edges, or none
            t_edge = r[edge]  # the edge second
            src = r[f"{edge}_source"].split(";")  # the evidence rows that set it
            for rx, s in (x.split(":") for x in src):  # generator: receiver and source of each
                if s != "gain_disturbed_run":  # if: not the run (an onset could share the second)
                    continue  # skip
                _, fld = GAIN_KEY[rx]  # its gain field
                ref = float(eai[(eai.test_code == r["test_code"]) & (eai.receiver == STATUS_RX[rx]) & (eai.field == fld)].quiet_reference.iloc[0])  # quiet reference
                drop = GAIN_DROP[fld]  # entry drop
                run = ev[(ev.test_code == r["test_code"]) & (ev.receiver == rx) & (ev.source == s) & ((ev.start == t_edge) | (ev.end == t_edge))].iloc[0]  # the run itself
                g = gain_series(long, r["test_code"], rx)  # raw gain, every sample
                gs = pd.Series(g.to_numpy(), index=(g.index.asi8 / 1e9))  # same values indexed by unix seconds (float)
                win = gs[(gs.index >= run.start - 60) & (gs.index < run.end + 61)]  # from 60 s before the run to 60 s after it
                ent, ext = win[win < ref - drop], win[win < ref - 0.4 * drop]  # samples beyond the entry and beyond the exit level
                if edge == "start":  # if: a start edge
                    raw_ent, raw_ext = (ent.index.min() if len(ent) else np.nan), (ext.index.min() if len(ext) else np.nan)  # first samples beyond each level
                else:  # else: an end edge
                    raw_ent, raw_ext = (ent.index.max() if len(ent) else np.nan), (ext.index.max() if len(ext) else np.nan)  # last samples beyond each level
                at = gs[(gs.index >= t_edge) & (gs.index < t_edge + 1)]  # raw samples inside the edge second
                # dict literal over five lines: one row per run-set edge, with the raw first/last samples beyond the entry and exit levels
                rows.append({"test_code": r["test_code"], "pooled_interval": f"{hms(r['start'])}-{hms(r['end'])}", "edge": edge, "receiver": rx, "run": f"{hms(run.start)}-{hms(run.end)}",
                             # dict entry: edge time, the quiet gain reference and the entry and exit levels derived from it
                             "edge_utc": hms(t_edge), "quiet_reference": ref, "entry_level": ref - drop, "exit_level": ref - 0.4 * drop,
                             # dict entry: first raw sample beyond the entry level and its distance in seconds from the edge
                             "raw_beyond_entry_utc": hms(raw_ent) if np.isfinite(raw_ent) else "", "edge_minus_raw_entry_s": round(t_edge - np.floor(raw_ent), 1) if np.isfinite(raw_ent) else np.nan,
                             # dict entry: last raw sample beyond the exit level and its distance in seconds from the edge
                             "raw_beyond_exit_utc": hms(raw_ext) if np.isfinite(raw_ext) else "", "edge_minus_raw_exit_s": round(t_edge - np.floor(raw_ext), 1) if np.isfinite(raw_ext) else np.nan,
                             "raw_gain_at_edge": float(at.median()) if len(at) else np.nan})  # dict literal over several lines: one row per run-set edge
                panels.append((rows[-1], gs[(gs.index >= t_edge - 120) & (gs.index <= t_edge + 120)]))  # the trace around the edge, for the figure
    if panels:  # if: there is anything to draw
        fig, axs = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.2), squeeze=False)  # one panel per edge
        for ax, (row, tr) in zip(axs[0], panels):  # zip pairs each panel with its edge
            ax.plot(pd.to_datetime(tr.index, unit="s"), tr.to_numpy(), lw=0.8, color=RXCOL[row["receiver"]], label="raw gain")  # the raw trace
            for y, ls, lb in ((row["quiet_reference"], "-", "quiet reference"), (row["entry_level"], "--", "entry (5 dB / 400)"), (row["exit_level"], ":", "exit (2 dB / 160)")):  # for loop over the three levels
                ax.axhline(y, color="k", lw=0.6, ls=ls, label=lb)  # a level line
            ax.axvline(pd.Timestamp(T(row["edge_utc"]), unit="s"), color="r", lw=0.8, label=f"run {row['edge']} {row['edge_utc']}")  # the run edge
            ax.set_title(f"{row['test_code']} {NAME[row['receiver']]}: run {row['edge']}", fontsize=8)  # panel title
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S")); ax.tick_params(labelsize=6)  # two statements: clock ticks, small labels
            ax.legend(fontsize=5)  # legend
        save_fig(fig, CHECK_DIR / "label_run_edges.png")  # byte-stable save
    return pd.DataFrame(rows)  # the table


def runs_table(lab: pd.DataFrame) -> pd.DataFrame:  # function: one row per contiguous run of the same label, phase and exclusion reason
    k = lab.label + "|" + lab.phase + "|" + lab.exclusion_reason  # string concatenation: the run key of every second
    key = lab.test_code + "|" + lab.receiver  # session and receiver key
    new = (k != k.shift()) | (key != key.shift()) | (lab.sec.diff() != 1)  # a run starts where the key, the session or the second sequence changes
    rid = new.cumsum()  # cumulative sum: a run number per second
    g = lab.groupby(rid)  # grouped by run
    # pd.DataFrame from a dict of per-run first/last values: the run table
    out = pd.DataFrame({"test_code": g.test_code.first(), "receiver": g.receiver.first(), "label": g.label.first(), "phase": g.phase.first(), "exclusion_reason": g.exclusion_reason.first(),
                        "start_utc": g.sec.first().map(hms), "end_utc": g.sec.last().map(hms), "n_seconds": g.size()})  # the run rows
    return out.reset_index(drop=True)  # renumbered


def counts(lab: pd.DataFrame, col: str) -> pd.DataFrame:  # function: seconds per label x session x receiver, all five labels always present
    c = lab.groupby(["test_code", "receiver", col]).size().unstack(fill_value=0)  # cross-tabulation: rows session x receiver, columns labels
    c = c.reindex(columns=LABELS, fill_value=0)  # every label as a column, in reporting order
    c = c.reindex(pd.MultiIndex.from_product([SESSIONS, RXS], names=["test_code", "receiver"]))  # session order, then receiver order
    c["total"] = c[LABELS].sum(axis=1)  # row total
    return c.reset_index()  # flat table


# ---------------------------------------------------------------- figure
def plot_labels(t: str, lab: pd.DataFrame, long: pd.DataFrame, dist: dict, W: dict, eai: pd.DataFrame) -> None:  # function: the figure panel for one session
    w = W[t]  # its windows
    x0, x1 = pd.Timestamp(w["cs"], unit="s"), pd.Timestamp(w["ce"] + 1, unit="s")  # the capture as naive UTC timestamps for the axis
    fig, axs = plt.subplots(3, 1, figsize=(10, 6.2), sharex=True, gridspec_kw={"height_ratios": [1.5, 1.4, 1.1]})  # three stacked panels sharing the time axis
    ax = axs[0]  # the label strips
    rows = [(rx, "label", NAME[rx]) for rx in RXS] + [(rx, "label_per_receiver_only", f"{NAME[rx]}, own evidence only") for rx in RXS]  # list: (receiver, column, row title), pooled rows first
    for i, (rx, col, title) in enumerate(rows):  # enumerate gives the row number with each row
        y = len(rows) - 1 - i  # rows drawn top-down
        L = lab[(lab.test_code == t) & (lab.receiver == rx)]  # this receiver's seconds
        k = L[col]  # its labels
        new = (k != k.shift()).to_numpy()  # where a run starts
        st_ = np.flatnonzero(new)  # run starts (positions)
        en = np.r_[st_[1:], len(k)]  # run ends (exclusive): the next start, or the end
        for a, b in zip(st_, en):  # for loop over runs
            s0 = pd.Timestamp(int(L.sec.iloc[a]), unit="s")  # run start time
            ax.broken_barh([(mdates.date2num(s0), (b - a) / 86400.0)], (y - 0.4 + (0.15 if col != "label" else 0), 0.8 - (0.3 if col != "label" else 0)), color=COLOR[k.iloc[a]], lw=0)  # one bar; Option 1 rows thinner
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[2] for r in rows][::-1], fontsize=7)  # two statements: row titles, bottom-up
    ax.axvspan(pd.Timestamp(w["ss"], unit="s"), pd.Timestamp(w["se"] + 1, unit="s"), color="k", alpha=0.06, lw=0)  # scheduled window, lightly shaded
    ax.legend(handles=[Patch(color=COLOR[l], label=l) for l in LABELS], fontsize=7, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)  # list comprehension: one legend box per label
    ax.set_title(f"Session labels, panel ({LETTER[t]}). Session {t}: labels per second", fontsize=9, pad=22)  # title, raised above the legend row
    ax2 = axs[1]  # the gains
    ax2b = ax2.twinx()  # a second y axis on the right for the u-blox counts
    for rx in RXS:  # for loop over receivers
        s = gain_series(long, t, rx)  # its L1 gain
        s = s[(s.index >= pd.Timestamp(w["cs"], unit="s", tz="UTC")) & (s.index <= pd.Timestamp(w["ce"] + 1, unit="s", tz="UTC"))]  # inside the capture
        tgt = ax2b if rx == "UBX_01" else ax2  # conditional expression: counts on the right axis, decibels on the left
        tgt.plot(s.index.tz_localize(None), s.to_numpy(), lw=0.8, color=RXCOL[rx], label=f"{NAME[rx]} L1 gain")  # the trace (timezone removed for the axis)
    ax2.set_ylabel("gain (dB), X5 and mosaic-H", fontsize=7); ax2b.set_ylabel("u-blox 01 AGC count", fontsize=7)  # two statements: axis titles
    ax2.axvspan(pd.Timestamp(w["ss"], unit="s"), pd.Timestamp(w["se"] + 1, unit="s"), color="k", alpha=0.06, lw=0)  # scheduled window
    h1, l1 = ax2.get_legend_handles_labels(); h2, l2 = ax2b.get_legend_handles_labels()  # two statements: legend entries of both axes
    ax2.legend(h1 + h2, l1 + l2, fontsize=6, loc="lower left")  # one legend for both
    ax3 = axs[2]  # own-position distance
    for rx in ("SBF_02", "UBX_01"):  # for loop over the receivers with a position
        dd = dist[(t, rx)]  # its distances
        ax3.plot(pd.to_datetime(dd.sec, unit="s"), dd.dh, ".", ms=1.5, color=RXCOL[rx], label=f"{NAME[rx]} own position ({len(dd)} fixes)")  # one dot per fix
    ax3.axhline(POS_M, color="k", lw=0.6, ls="--", label="50 m")  # the departure distance
    ax3.set_yscale("log"); ax3.set_ylim(0.1, 2e4)  # two statements: logarithmic distance axis from 0.1 m to 20 km
    ax3.set_ylabel("distance from site (m)", fontsize=7)  # axis title
    ax3.legend(fontsize=6, loc="upper left")  # legend
    ax3.axvspan(pd.Timestamp(w["ss"], unit="s"), pd.Timestamp(w["se"] + 1, unit="s"), color="k", alpha=0.06, lw=0)  # scheduled window
    ax3.set_xlim(x0, x1)  # the capture
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))  # clock-time ticks
    ax3.set_xlabel("UTC, 14 September 2026 (u-blox on host time)", fontsize=8)  # x title
    save_fig(fig, GRAPH / f"figure23{LETTER[t]}_labels_{t}.png")  # byte-stable save


# ---------------------------------------------------------------- main
def main() -> None:  # function: builds the labels, checks them, writes them
    d = load_inputs()  # every input
    W = windows(d["gt"])  # windows per session
    J = jam_phases(d["ph"], d["ov"], W)  # jamming-phase boundaries per session
    long = status_long(d["st"])  # the receiver-status script's long table, rebuilt
    print("== the receiver-status all-indicator rule, reproduced ==")  # header
    split = ("2.6.12", pd.Timestamp(f"{DATE} {MARGIN_CONTAMINATION_START}", tz="UTC"))  # the receiver-status script's possibly contaminated 2.6.12 margin split
    parts, refs = indicator_mask(long, d["gt"], split)  # the wrapped rule, with its per-point mask
    mc = pd.read_csv(PROC / "ground_truth" / "margin_clean_minutes.csv")  # the receiver-status script's published table
    allc = mc[mc.indicator.str.startswith("ALL")].reset_index(drop=True)  # its "ALL combined" rows
    # list comprehension inside pd.DataFrame over two lines: my margin, covered and contaminated minutes for every margin part
    mine = pd.DataFrame([{"test_code": p["test_code"], "margin": f"{p['a']:%H:%M:%S}-{p['b']:%H:%M:%S}", "margin_min": round(len(p["grid"]) / 600, 3),
                          "covered_min": round(p["cov"].sum() / 600, 3), "contaminated_min": round(p["bad"].sum() / 600, 3)} for p in parts])  # list comprehension: my minutes per part
    cmp_ = allc.merge(mine, on=["test_code", "margin"], suffixes=("", "_mine"), how="outer")  # outer join: rows on both sides
    dmax = max(float((cmp_[c] - cmp_[c + "_mine"]).abs().max()) for c in ("margin_min", "covered_min", "contaminated_min"))  # largest difference over the three columns
    check("indicator mask = ground_truth/margin_clean_minutes.csv ALL rows", len(cmp_) == len(allc) == len(mine) and dmax <= 0.0005, f"{len(mine)} parts, {len(allc)} published rows, largest difference {dmax:.4f} min")  # equal to 0.001 min
    fresh = S03.margin_clean(long, d["gt"], split=split)  # a fresh call to the receiver-status script's function, unchanged
    pub = mc[mc.test_code != "all"].reset_index(drop=True)  # the published table without the receiver-status script's extra single-indicator row
    fz = fresh.reset_index(drop=True)  # renumbered
    same = (list(fz.columns) == list(pub.columns)) and len(fz) == len(pub) and all((fz[c].astype(str) == pub[c].astype(str)).all() if fz[c].dtype == object else np.allclose(fz[c].astype(float), pub[c].astype(float), atol=1e-9) for c in fz.columns)  # generator inside all(): every column equal
    check("fresh S03.margin_clean() = ground_truth/margin_clean_minutes.csv", bool(same), f"{len(fz)} rows x {len(fz.columns)} columns compared")  # equal
    for p in parts:  # for loop over parts: my minutes against the fresh call's ALL row and the published row
        mg = f"{p['a']:%H:%M:%S}-{p['b']:%H:%M:%S}"  # the margin's text, as the receiver-status script writes it
        fr = fz[(fz.test_code == p["test_code"]) & (fz.margin == mg) & fz.indicator.str.startswith("ALL")]  # the fresh call's row
        pr = allc[(allc.test_code == p["test_code"]) & (allc.margin == mg)]  # the published row
        print(f"    {p['test_code']:7s} {mg}: contaminated {p['bad'].sum() / 600:.3f} min (fresh 03: {float(fr.contaminated_min.iloc[0]):.3f}; published: {float(pr.contaminated_min.iloc[0]):.3f})")  # one line per part
    ind = indicator_seconds(parts)  # per session, the passing seconds
    print("== independent evidence ==")  # header
    gev, rep = gain_evidence(d, long, W, d["eai"])  # gain evidence and the run reproduction
    okr = bool((rep.runs_whole == rep.runs_whole_s3).all() and (rep.runs_window == rep.runs_window_s3).all() and ((rep.longest_whole_min - rep.longest_whole_min_s3).abs() <= 0.005).all())  # every count and length equal
    check("disturbed runs = ground_truth/effective_attack_interval.csv", okr, f"{len(rep)} session x receiver rows; whole-capture runs {int(rep.runs_whole.sum())} (S3 {int(rep.runs_whole_s3.sum())}), in-window {int(rep.runs_window.sum())} (S3 {int(rep.runs_window_s3.sum())})")  # reproduced
    pev, dist = position_evidence(d["gt"], W)  # own-position departures
    tev = time_evidence(d["sp"])  # own-time evidence
    ev = pd.DataFrame(gev + pev + tev).sort_values(["test_code", "receiver", "start", "source"]).reset_index(drop=True)  # every evidence interval
    ev["start_utc"], ev["end_utc"] = ev.start.map(hms), ev.end.map(hms)  # tuple assignment: readable times
    for r in ev.itertuples():  # for loop over the evidence, printed for the report
        print(f"    {r.test_code:7s} {r.receiver:7s} {r.source:24s} {r.start_utc}-{r.end_utc} ({r.end - r.start + 1} s)")  # one line each
    lab, pooled_iv = label_all(d, ev, W, J, ind)  # the labels (adopted rule)
    pooled_iv = edge_sources(ev, pooled_iv)  # which evidence sets each pooled edge
    print("== pooled attack intervals ==")  # header
    for r in pooled_iv:  # for loop over the pooled intervals
        print(f"    {r['test_code']:7s} {hms(r['start'])}-{hms(r['end'])} ({r['end'] - r['start'] + 1} s)  start {r['start_source']}  end {r['end_source']}")  # one line each
    re_ = raw_edge_check(long, ev, pooled_iv, d["eai"])  # run-set edges against the raw gain
    re_.to_csv(CHECK_DIR / "label_run_edges.csv", index=False, lineterminator="\n")  # written
    print("== pooled edges set by a disturbed run, against the raw gain trace ==")  # header
    print(re_.to_string(index=False))  # the table
    # ---- the per-second columns that do not decide the label
    parts_ = []  # the finished per-session, per-receiver tables
    for (t, rx), L in lab.groupby(["test_code", "receiver"], sort=False):  # for loop over session x receiver groups, in table order
        L = L.copy()  # a copy, so columns can be added
        s = L.sec.to_numpy()  # its seconds
        sch = np.where((s >= W[t]["ss"]) & (s <= W[t]["se"]), "spoof", "")  # scheduled phase: spoofing inside the window ...
        for a, b in J[t]["scheduled"]:  # ... and the scheduled opening jamming where the test has one
            sch = np.where((s >= a) & (s <= b), "jam_spoof", sch)  # overwritten
        L["scheduled_phase"] = sch  # column
        L["utc"] = [hms(x) for x in s]  # list comprehension: clock text
        L["gain_effect_this_rx"] = cover(ev, t, s, rx=rx, sources=GAIN_SOURCES)  # this receiver's own gain evidence
        L["gain_effect_pooled"] = cover(ev, t, s, sources=GAIN_SOURCES)  # any receiver's gain evidence
        L["own_pos_departure_x5"] = cover(ev, t, s, rx="SBF_02", sources=["own_position"])  # the X5 own-position departure
        L["own_pos_departure_ubx"] = cover(ev, t, s, rx="UBX_01", sources=["own_position"])  # u-blox 01's departure
        L["own_time_offset_ubx"] = cover(ev, t, s, rx="UBX_01", sources=["own_time"])  # u-blox 01's measurement clock offset
        L["evidence_receivers"] = who(ev, t, s, "receiver")  # which receivers' evidence covers the second
        src = who(ev, t, s, "source")  # which sources
        att = L.label.str.startswith("attack").to_numpy()  # attack seconds
        L["evidence_sources"] = np.where(att & (src == ""), "merged_gap", src)  # attack seconds covered only by the 30 s merge are named
        L["status_flag_set"] = flags(d["st"], t, rx, s)  # the receiver's own flag (column only)
        L["concurrent_feature_band_test"] = concurrent(d["ct"], t, s)  # concurrent feature-band tests (column only)
        parts_.append(L)  # kept
    lab = pd.concat(parts_, ignore_index=True)  # one table
    # list over two lines: the published column order of labels/labels_1hz.csv
    cols = ["test_code", "receiver", "sec", "utc", "label", "phase", "in_scheduled_window", "scheduled_phase", "gain_effect_this_rx", "gain_effect_pooled", "own_pos_departure_x5", "own_pos_departure_ubx",
            "own_time_offset_ubx", "evidence_receivers", "evidence_sources", "status_flag_set", "concurrent_feature_band_test", "label_per_receiver_only", "exclusion_reason", "indicator_pass"]  # list: the column order
    for c in ("in_scheduled_window", "gain_effect_this_rx", "gain_effect_pooled", "own_pos_departure_x5", "own_pos_departure_ubx", "own_time_offset_ubx", "indicator_pass"):  # for loop over the yes/no columns
        lab[c] = lab[c].astype(int)  # True/False -> 1/0 for a compact CSV
    out = lab[cols]  # the published columns
    out.to_csv(PROC / "labels" / "labels_1hz.csv", index=False, lineterminator="\n")  # written
    iv = runs_table(out)  # contiguous runs
    iv["edge_from_disturbed_run"] = ""  # empty unless a run edge is a pooled edge set by a disturbed run
    for r in pooled_iv:  # for loop over pooled intervals with a run-set edge
        for edge in (r["edge_from_disturbed_run"].split(";") if r["edge_from_disturbed_run"] else []):  # for loop over those edges
            col = "start_utc" if edge == "start" else "end_utc"  # conditional expression: which end of a run to match
            m = (iv.test_code == r["test_code"]) & iv.label.str.startswith("attack") & (iv[col] == hms(r[edge]))  # attack runs whose edge is this pooled edge
            iv.loc[m, "edge_from_disturbed_run"] = [x + (";" if x else "") + edge for x in iv.loc[m, "edge_from_disturbed_run"]]  # list comprehension: appended
    iv.to_csv(PROC / "labels" / "labels_intervals.csv", index=False, lineterminator="\n")  # written
    c2, c1 = counts(out, "label"), counts(out, "label_per_receiver_only")  # tuple assignment: the pooled counts and Option 1's
    cnt = pd.concat([c2.assign(rule="pooled (Option 2)"), c1.assign(rule="per receiver (Option 1)")], ignore_index=True)  # stacked, with the rule named
    cnt = cnt[["rule", "test_code", "receiver"] + LABELS + ["total"]]  # column order
    cnt.to_csv(PROC / "labels" / "labels_counts.csv", index=False, lineterminator="\n")  # written
    evo = pd.concat([ev, pd.DataFrame(pooled_iv).assign(start_utc=lambda x: x.start.map(hms), end_utc=lambda x: x.end.map(hms))], ignore_index=True)  # evidence plus pooled intervals
    evo[["test_code", "receiver", "source", "start_utc", "end_utc", "start", "end", "start_source", "end_source", "edge_from_disturbed_run"]].fillna("").to_csv(PROC / "labels" / "labels_evidence.csv", index=False, lineterminator="\n")  # written
    # ---- self-checks
    print("== self-checks ==")  # header
    key = out.test_code + "|" + out.receiver + "|" + out.sec.astype(str)  # one key per row
    check("every grid second labelled exactly once", key.is_unique and out.label.isin(LABELS).all() and out.label_per_receiver_only.isin(LABELS).all(), f"{len(out)} rows, {key.nunique()} distinct keys, labels all in the five")  # unique and valid
    ek = set((d["ep"].test_code + "|" + d["ep"].receiver + "|" + d["ep"].sec.astype(str)).tolist())  # the epoch grid's keys
    extra = out[~key.isin(ek)]  # rows not in the epoch grid
    # check over two lines: every epoch second present and every added second excluded for a stated reason
    check("grid = epoch seconds + no-observation seconds", ek <= set(key.tolist()) and (extra.label == "excluded").all() and extra.exclusion_reason.isin(["no_observations", "vehicle_motion"]).all(),
          f"{len(ek)} epoch seconds all present; {len(extra)} added capture seconds, all excluded ({', '.join(f'{k} {v}' for k, v in extra.exclusion_reason.value_counts().sort_index().items())})")  # every epoch second present, every added one excluded
    rc = iv.groupby(["test_code", "receiver", "label"]).n_seconds.sum().unstack(fill_value=0).reindex(columns=LABELS, fill_value=0)  # independent count from the run table
    rc = rc.reindex(pd.MultiIndex.from_product([SESSIONS, RXS], names=["test_code", "receiver"])).fillna(0).astype(int)  # same order as the counts
    check("counts = recount from labels/labels_intervals.csv", bool((rc.to_numpy() == c2.set_index(["test_code", "receiver"])[LABELS].to_numpy()).all()), f"{int(rc.to_numpy().sum())} seconds recounted from {len(iv)} runs")  # equal
    bad_in = int(((out.label == "clean") & (out.in_scheduled_window == 1)).sum())  # clean seconds inside a window
    near_bad = 0  # clean seconds within 30 s of a guarding second
    for (t, rx), L in lab.groupby(["test_code", "receiver"]):  # for loop over session x receiver
        lb = L.label.to_numpy()  # labels
        # the guard's triggers: every second that is not a clean candidate (attack, uncertain by rule, excluded, failing an indicator);
        # seconds removed by the guard itself are not triggers, otherwise the guard would cascade through the whole margin
        trig = ~L._cand.to_numpy() & ~L._nogrid.to_numpy()  # boolean NOT of the candidate flag; the unexamined capture-end second sets no guard
        r_ = np.flatnonzero(np.diff(np.r_[0, (L.exclusion_reason.to_numpy() == "no_observations").astype(int), 0]))  # ... except short gaps without observations
        for a, b in zip(r_[0::2], r_[1::2]):  # for loop over those gaps
            if b - a < SHORT_S:  # if: shorter than 30 s
                trig[a:b] = False  # not a guard trigger
        cl = np.flatnonzero(lb == "clean")  # positions of clean seconds
        tg = np.flatnonzero(trig)  # positions of triggers
        if len(cl) and len(tg):  # if: both exist
            j = np.searchsorted(tg, cl)  # for each clean second, where it would sit among the triggers
            dl = np.minimum(np.abs(cl - tg[np.clip(j - 1, 0, len(tg) - 1)]), np.abs(tg[np.clip(j, 0, len(tg) - 1)] - cl))  # distance to the nearest trigger on either side
            near_bad += int((dl <= GUARD_S).sum())  # clean seconds 30 s or closer
    check("no clean second in a window or within 30 s of a non-clean second", bad_in == 0 and near_bad == 0, f"in window {bad_in}, within 30 s {near_bad} (short no-observation gaps exempt)")  # zero both
    sub = all(set(out[(out.test_code == t) & (out.label == "clean")].sec) <= ind.get(t, {"pass": set()})["pass"] for t in SESSIONS)  # generator inside all(): clean set within the survivors, every session
    check("clean seconds subset of the receiver-status script's all-indicator survivors", sub, "per session and receiver")  # subset
    p262 = [r for r in pooled_iv if r["test_code"] == "2.6.2"]  # 2.6.2's pooled intervals
    check("2.6.2 pooled interval 13:04:53-13:06:48 (Option B)", len(p262) == 1 and hms(p262[0]["start"]) == "13:04:53" and hms(p262[0]["end"]) == "13:06:48", ", ".join(f"{hms(r['start'])}-{hms(r['end'])}" for r in p262))  # as expected
    x5o1 = int(out[(out.test_code == "2.6.2") & (out.receiver == "SBF_02")].label_per_receiver_only.str.startswith("attack").sum())  # X5 attack seconds under Option 1
    check("2.6.2 X5 has 88 attack seconds under Option 1 (Option B)", x5o1 == 88, f"{x5o1} attack seconds (its own disturbed run 13:05:21-13:06:48)")  # the revised expectation
    p2610 = [r for r in pooled_iv if r["test_code"] == "2.6.10"]  # 2.6.10's pooled intervals
    e2610 = max(r["end"] for r in p2610) if p2610 else None  # its end
    check("2.6.10 pooled end from independent evidence", e2610 is not None and hms(e2610) == "14:44:06", f"{hms(e2610)}; residual-based the end of spoofed tracking in 2.6.10 end 14:43:59, difference {e2610 - T('14:43:59')} s")  # 14:44:06
    pd.DataFrame(CHECKS).to_csv(CHECK_DIR / "label_labels_checks.csv", index=False, lineterminator="\n")  # the checks written
    # ---- variants discussed at the plan step (counts only, for the report)
    var = [("adopted rule", {}), ("guard around every gap without observations", {"guard_all_noobs": True}), ("without the 2.6.5 post-test rule", {"use_54c": False}), ("unexamined capture-end second sets the guard", {"guard_nogrid": True})]  # list: (name, keyword arguments)
    vrows = []  # rows of the variants table
    for name, kw in var:  # for loop over variants
        vl, _ = label_all(d, ev, W, J, ind, **kw) if kw else (lab, None)  # conditional expression: rebuild for a variant, reuse for the adopted rule
        for (t, rx), L in vl.groupby(["test_code", "receiver"], sort=False):  # for loop over session x receiver
            vrows.append({"variant": name, "test_code": t, "receiver": rx, **{l: int((L.label == l).sum()) for l in LABELS}})  # dict unpacking with a dict comprehension: the counts
    ev_nav = ev.copy()  # the evidence with the navigation-time end, for comparison
    ev_nav.loc[ev_nav.source == "own_time", "end"] = T(d["sp"][(d["sp"].receiver == "u-blox_01") & (d["sp"].event == "navigation time release")].host_time_utc.iloc[0])  # .loc: own time to the navigation release, floored likewise
    vl, pv = label_all(d, ev_nav, W, J, ind)  # labels with that end
    for (t, rx), L in vl.groupby(["test_code", "receiver"], sort=False):  # for loop over session x receiver
        vrows.append({"variant": "own time to the navigation-time release", "test_code": t, "receiver": rx, **{l: int((L.label == l).sum()) for l in LABELS}})  # the counts
    vdf = pd.DataFrame(vrows)  # the variants table
    vdf.to_csv(CHECK_DIR / "label_labels_variants.csv", index=False, lineterminator="\n")  # written
    print("    2.6.12 pooled end with the navigation-time release: " + ", ".join(f"{hms(r['start'])}-{hms(r['end'])}" for r in pv if r["test_code"] == "2.6.12"))  # for the report
    # ---- figure
    for t in SESSIONS:  # for loop over sessions
        plot_labels(t, lab, long, dist, W, d["eai"])  # the figure panel
    print("== counts (pooled, Option 2) ==")  # header
    print(c2.to_string(index=False))  # the counts
    print("== counts (per receiver, Option 1) ==")  # header
    print(c1.to_string(index=False))  # the counts
    fails = [c for c in CHECKS if c["result"] != "PASS"]  # list comprehension: failed checks
    print(f"== {len(CHECKS) - len(fails)} of {len(CHECKS)} checks pass ==")  # summary line


