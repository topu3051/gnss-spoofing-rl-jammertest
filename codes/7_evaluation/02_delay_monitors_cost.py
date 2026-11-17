#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
RQ4: the detectors against the conventional monitors at matched clean-window shares, alarm delay against the gain depression each
receiver registered, and the computational cost.

  1 Reference monitors (the evidence script's definitions): the gain fall (median gain over the 90 s that ended 30 s ago minus the gain now), alarm when it
    reaches the threshold in each of the last 10 seconds (the evidence script's gain_alarm_runs at persistence 10, compared exactly so that a point chosen
    one float step above a value holds); and the receiver's own position's horizontal distance from the site, as an exact causal
    transcription of the evidence script's first_hold_time (a run of fixes beyond the threshold continues across gaps of up to 5 s and alarms from the
    first fix at which it has lasted 10 s). The mosaic-H reports no position, so it has no position monitor, and every comparison says so.
    Deployable points like the logistic regression's: the lowest threshold whose share of the training sessions' clean stream seconds in
    alarm is at most the share (folds by leave-one-session-out; 2.6.12 from the five position sessions). Every result carries its coverage
    and two clean shares (all clean seconds; those where the monitor is defined), the realised share on the evaluated session with the
    'point not held' mark above twice the target, and the gain monitor its circularity flag: all 9 pooled onsets were set by
    a gain quantity; the sensitivity drops the intervals set by that receiver's own monitored quantity (gain: 3, 7 and 4 of 9 left).
  2 Both onset marks on every interval: in alarm on the last decided second at most 60 s before the onset, or
    within the 30 s refractory period of the last stopping-mode event: no detection delay; kept, marked, left out of medians and of rho,
    with an all-included sensitivity. The agent experiment's within-receiver intervals get their marks by running its point runs again
    (frozen agents and committed predictions, gated bitwise against heldout_latency.csv, its verdicts unchanged); 2.6.12's come from the
    transfer script's tables.
  3 Delay against registered strength: every interval at the share 0.05 with a counted delay, against the interval's median depression of
    that receiver's main-antenna L1 gain below its quiet level (36 dB, 39 dB, 5,265 counts, checked). Spearman's rho per receiver and
    detector is the primary result, with a 95 % bootstrap interval over intervals (10,000 resamples, seed 20260914); pooled over receivers
    with u-blox counts converted at about 100 counts per dB is secondary.
  4 Cost: parameters, depth, multiply-accumulate operations per decision and median time per decision over 10,000 decisions on one thread
    (asserted) with the model loaded, for the logistic regression, the forest (n_jobs 1, asserted), CUSUM, k of n and both DQN arms (the
    agent timed at lambda 100, seed 20260914: the network is the same for every lambda); P's fitted transform and the feature script's window
    statistics (one window-feature call per window) timed separately; the feature script's per-second aggregation from the RINEX is not timed.
    Against the 1 s decision interval and the receivers' 100 ms, 50 ms and 1 s output intervals. cost_timing.csv, cost_summary.csv and
    the figure depend on wall clock and are excluded from the byte comparison (--no-cost rebuilds the summary and the figure from the
    timing file).
2.6.12's gain and own positions are read only after the transfer script's guard accepts the freeze; the reads are counted in reads.csv.
It imports the transfer script (which loads the evaluation, training and environment scripts), the evidence script and the window-feature
script for their unchanged functions.
Writes  Data/processed/rq4/*; Graph/monitor_curves.png, delay_depression.png, cost.png
"""
from __future__ import annotations  # future import: type hints kept as text (the project's convention)

import argparse  # standard library: the command line
import importlib.util  # standard library: loads the transfer script, the evidence script and the window-feature script (their names start with a digit)
import os  # standard library: the thread environment variables, set before numpy's libraries start
import platform  # standard library: the machine and system for the cost table
import subprocess  # standard library: the processor name (sysctl)
import sys  # standard library: module registry and exit status
import time  # standard library: perf_counter_ns for the timing
from collections import deque  # standard library: the k-of-n window of past log-odds
from pathlib import Path  # standard library: paths

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):  # for loop over the thread variables of the numerical libraries
    os.environ.setdefault(_v, "1")  # one thread each (set before numpy loads), so every timing is single-threaded

import numpy as np  # numpy: arrays
import pandas as pd  # pandas: tables

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/


def _load(name: str, file: str):  # function: loads a script by path as a module (its main() is not run)
    spec = importlib.util.spec_from_file_location(name, PROJECT / "codes" / file)  # where it is
    mod = importlib.util.module_from_spec(spec)  # an empty module object
    sys.modules[name] = mod  # registered
    spec.loader.exec_module(mod)  # its top level
    return mod  # the module


X = _load("transfer_for_delay_monitors", "7_evaluation/01_transfer.py")  # the transfer script: the guard, the Part B points, the onset marks and the evaluation, training, environment and adopted-pipeline scripts through it
B12, T, E, A, M = X.B12, X.T, X.E, X.A, X.M  # the evaluation, training, environment, adopted-pipeline and regression scripts
sys.path.insert(0, str(PROJECT / "codes" / "common"))  # the evidence script's own_positions imports the helpers module from codes/
D = _load("gain_evidence_for_delay_monitors", "1_ground_truth/04_gain_and_position_evidence.py")  # the evidence script, unchanged: gain_series, gain_drop, own_positions, horiz_from_site, site_xyz, capture_spans, first_hold_time

RQ4_DIR = E.PROC / "rq4"  # this section's folder
GRAPH = E.GRAPH  # figures
SEALED, RXS = X.SEALED, X.RXS  # "2.6.12", the three receivers
SESS5 = list(X.SESSIONS)  # list: the five position sessions in the screen script's order (a list, so it can be extended)
ALL6 = SESS5 + [SEALED]  # list concatenation: the five folds and 2.6.12
NAME = E.NAME  # stream -> name
SHARES = T.SHARES  # (0.02, 0.05, 0.10)
PERSIST = 10  # the evidence script's persistence of the gain monitor (gain_alarm_runs at 10)
HOLD_S, GAP_S = float(D.HOLD_POS_S), 5.0  # tuple assignment: the evidence script's hold of the position monitor (10 s) and first_hold_time's gap that ends a run (5 s)
SEED, N_BOOT = T.SEED, 10_000  # tuple assignment: the bootstrap seed and resamples
COUNTS_PER_DB = 100.0  # the evidence script's conversion of u-blox AGC counts to dB (46 dB against 4,563 counts at the 2.6.12 onset), for the pooled rho only
MONITORS = ("gain", "position")  # tuple: the two reference monitors
BAND = {"SBF_02": ("GPSL1/E1", "agc_gain_db"), "SBF_03": ("GPSL1/E1", "agc_gain_db"), "UBX_01": ("L1", "agc_cnt")}  # dict: receiver -> (band, field) of the gain the evidence script reads, selected explicitly
QUIET = {"SBF_02": 36.0, "SBF_03": 39.0, "UBX_01": 5265.0}  # dict: the quiet levels stated beforehand, checked against the receiver-status script's table
GRID = {"gain": {"SBF_02": np.round(np.arange(0, 30.5, 0.5), 2), "SBF_03": np.round(np.arange(0, 30.5, 0.5), 2), "UBX_01": np.arange(0, 3050, 50.0)}, "position": np.r_[0.0, np.logspace(0, 4, 41)]}  # dict: the thresholds swept for the curves (dB, counts, m)
FIXED = {"gain": {"SBF_02": (1.0, 2.0, 3.0, 5.0), "SBF_03": (1.0, 2.0, 3.0, 5.0), "UBX_01": (100.0, 200.0, 300.0, 500.0)}, "position": (50.0,)}  # dict: the evidence script's fixed settings, run as extra curve points
DNAME = {"lr": "logistic regression", "rf": "forest", "cusum": "CUSUM", "kofn": "k of n", "agent_primary": "agent, primary", "agent_surrogate": "agent, surrogate", "gain": "gain-fall monitor", "position": "position monitor"}  # dict: detector -> name
CHECKS, READS = [], []  # this script's checks; its reads of 2.6.12's label-side data


def check(name: str, ok: bool, detail: str) -> None:  # function: records one check and prints it
    CHECKS.append({"check": name, "pass": bool(ok), "detail": detail})  # appended
    print(("PASS " if ok else "FAIL ") + name + " | " + str(detail)[:160], flush=True)  # printed


# ----------------------------------------------------------------------------------------------------------------------
# the monitors' statistics (the evidence script's quantities, the held statistic H is the largest threshold at which the monitor alarms at a second)
# ----------------------------------------------------------------------------------------------------------------------
def persistent_min(sec: np.ndarray, x: np.ndarray, p: int) -> pd.Series:  # function: the gain monitor's held statistic: the smallest value over the last p consecutive seconds (NaN unless all p exist), so H >= c exactly when the evidence script's gain_alarm_runs has a run of p seconds at or above c ending at or covering this second from its p-th second
    s = pd.Series(x, index=sec).sort_index()  # by second
    s = s.reindex(range(int(s.index.min()), int(s.index.max()) + 1))  # a full one-second grid (missing seconds NaN)
    return s.rolling(p, min_periods=p).min()  # rolling minimum over p seconds; NaN if any is missing


def hold_stat(t: np.ndarray, sec: np.ndarray, d: np.ndarray, hold: float = HOLD_S, gap: float = GAP_S) -> pd.Series:  # function: the position monitor's held statistic, an exact causal transcription of the evidence script's first_hold_time: at fix i, H_i >= c exactly when the run of fixes at or above c that contains i (a gap over 5 s ends a run) began at least `hold` seconds before fix i
    n = len(t)  # the fixes
    H = np.full(n, np.nan)  # undefined until a run can have lasted `hold` seconds
    seg = np.cumsum(np.r_[True, np.diff(t) > gap])  # running count: the gap-free stretch of each fix (a gap over 5 s ends a run)
    j = 0  # m0*: the latest fix at least `hold` seconds before fix i (it only moves forward, since t increases)
    for i in range(n):  # for loop over the fixes in time order
        while j + 1 <= i and t[i] - t[j + 1] >= hold:  # while: a later fix is still at least `hold` seconds back
            j += 1  # move to it
        if t[i] - t[j] >= hold and seg[j] == seg[i]:  # if: such a fix exists and no gap over 5 s lies between it and fix i
            H[i] = d[j:i + 1].min()  # the run from fix j to fix i holds for any threshold up to its smallest distance
    return pd.Series(H, index=sec)  # by whole second (the evidence script keeps the first fix of each second)


def statistics(st: pd.DataFrame, gt: pd.DataFrame, rx: str, t: str) -> dict:  # function: the two monitors' held statistics of one receiver and session, by second
    g = D.gain_drop(D.gain_series(st, t, rx))  # 05d: the gain fall (dB or counts, the evidence script selects the band: GPSL1/E1 or L1)
    out = {"gain": persistent_min(g.index.to_numpy(np.int64), g.to_numpy(float), PERSIST), "fixes": 0}  # held 10 s; the fix count, filled below
    if rx == "SBF_03":  # the mosaic-H reports no position
        out["position"] = None  # no position monitor
        return out  # done
    own = D.own_positions(gt, t, rx)  # 05d: the first valid own fix of each second (u-blox on host time)
    a0, b0 = D.capture_spans(gt, t, rx)[0]  # the evidence script's usable span (2.6.5 from 12:34:57)
    own = own[(own.sec >= a0) & (own.sec < b0)].reset_index(drop=True)  # inside it
    out["fixes"] = len(own)  # counted (2.6.12's are reported as reads)
    if not len(own):  # if: no fix at all
        out["position"] = pd.Series(dtype=float)  # nothing defined
        return out  # done
    dh = D.horiz_from_site(own.lat.to_numpy(), own.lon.to_numpy(), own.h.to_numpy(), D.site_xyz(D.M.SITE_OF[t]))  # horizontal distance from the session's site
    out["position"] = hold_stat(own.t.to_numpy(float), own.sec.to_numpy(np.int64), dh)  # the evidence script's hold, transcribed
    return out  # the statistics


def grids() -> dict:  # function: the shared stream grid (windowed seconds, labels, decidability) of every receiver and session, 2.6.12 from the transfer script's held-out grid
    steps = T.read_steps()  # the environment script's step table (position sessions)
    out = {}  # (receiver, session) -> grid
    for rx in RXS:  # for loop over the receivers
        for t in SESS5:  # for loop over the position sessions
            out[(rx, t)] = T.session_seconds(steps, rx, t).assign(decidable=lambda d: d.valid)  # the windowed seconds (decidable for a sequential detector = valid)
    g12 = pd.read_csv(X.U2612 / "grid.csv.gz", dtype={"session": str})  # the transfer script's 2.6.12 grid
    for rx in RXS:  # for loop over the receivers
        out[(rx, SEALED)] = g12[g12.receiver == rx].sort_values("sec").reset_index(drop=True)  # its grid
    return out  # the grids


def marks(sec: np.ndarray, dec: np.ndarray, stop: np.ndarray, decided: np.ndarray, held: str) -> list:  # function: both onset marks at every pooled interval of a session for one run
    ev = E.events(sec, np.asarray(stop) > 0)  # the stopping-mode events (the environment script's refractory rule)
    rows = []  # one row per interval
    for a, b in E.SC.pooled_intervals(held):  # for loop over the pooled intervals
        s = X.onset_state(sec, np.asarray(dec), np.asarray(decided, bool), int(a))  # in alarm, not in alarm or not decidable on the last decided second at most 60 s back
        r, g = X.refractory_at(sec, ev, int(a))  # tuple assignment: refractory at onset, and the last event's age
        rows.append({"onset": int(a), "onset_state": s["state"], "onset_ref_age_s": s["ref_age_s"], "refractory_at_onset": bool(r), "last_event_age_s": g, "no_detection_delay": s["state"] == "in alarm" or bool(r)})  # the row
    return rows  # the rows


# ----------------------------------------------------------------------------------------------------------------------
# 1. the monitors through the environment script's stream, at points chosen like every other detector's
# ----------------------------------------------------------------------------------------------------------------------
def monitors(st: pd.DataFrame, gt: pd.DataFrame, G: dict) -> tuple:  # function: statistics on the grids, points, runs, curves, coverage
    stat, series = {}, []  # (monitor, receiver, session) -> held statistic on the grid; the per-second table
    for rx in RXS:  # for loop over the receivers
        for t in ALL6:  # for loop over the sessions (2.6.12 after the guard, which main() ran first)
            s = statistics(st, gt, rx, t)  # the statistics
            if t == SEALED:  # 2.6.12: the label-side read is counted
                READS.append({"what": f"{NAME[rx]} own position fixes in 2.6.12 (raw file, after the guard)", "rows": int(s["fixes"])})  # recorded
            g = G[(rx, t)]  # the grid
            sec = g.sec.to_numpy(np.int64)  # its seconds
            row = {"receiver": rx, "session": t, "sec": sec, "label": g.label.to_numpy(object)}  # dict literal: the series row's keys
            for mon in MONITORS:  # for loop over the monitors
                v = s[mon].reindex(sec).to_numpy(float) if s[mon] is not None and len(s[mon]) else np.full(len(sec), np.nan)  # the held statistic on the grid (NaN: undefined)
                stat[(mon, rx, t)] = v  # kept
                row[mon] = v  # in the table
            series.append(pd.DataFrame(row))  # the session's rows
    S_ = pd.concat(series, ignore_index=True)  # every grid second with both statistics
    pts, runs, lat, curve, cov = [], [], [], [], []  # the outputs
    for mon in MONITORS:  # for loop over the monitors
        for rx in RXS:  # for loop over the receivers
            if mon == "position" and rx == "SBF_03":  # the mosaic-H: no position monitor
                continue  # nothing to run
            for held in ALL6:  # for loop over the evaluated sessions: the five folds and 2.6.12
                train = [s for s in SESS5 if s != held]  # the training sessions (all five for 2.6.12)
                clean_v = np.concatenate([stat[(mon, rx, s)][G[(rx, s)].label.to_numpy() == "clean"] for s in train])  # the held statistic at the training sessions' clean grid seconds (NaN where undefined)
                n_all = len(clean_v)  # every clean grid second (the denominator every detector is chosen on)
                n_def = int(np.isfinite(clean_v).sum())  # those where the monitor is defined
                g = G[(rx, held)]  # the evaluated session's grid
                sec, lab = g.sec.to_numpy(np.int64), g.label.to_numpy(object)  # tuple assignment: arrays
                v = stat[(mon, rx, held)]  # its statistic
                cl, dfn = lab == "clean", np.isfinite(v)  # tuple assignment: clean seconds; defined seconds
                # dict literal: the session's coverage
                cov.append({"monitor": mon, "receiver": rx, "session": held, "grid_seconds": len(sec), "defined_seconds": int(dfn.sum()), "clean_seconds": int(cl.sum()), "defined_clean_seconds": int((cl & dfn).sum()), "attack_seconds": int(np.isin(lab, A.ATTACK).sum()), "defined_attack_seconds": int((np.isin(lab, A.ATTACK) & dfn).sum())})  # coverage of clean and attack seconds
                for sh in SHARES:  # for loop over the shares
                    thr = T.lowest_threshold(clean_v, n_all, sh)  # the lowest threshold with at most the share of ALL clean training seconds in alarm (the training script's rule)
                    tr_all = float((np.nan_to_num(clean_v, nan=-np.inf) >= thr).sum() / n_all) if n_all else np.nan  # its training share over all clean seconds
                    tr_def = float((clean_v[np.isfinite(clean_v)] >= thr).sum() / n_def) if n_def else np.nan  # and over the defined ones
                    pts.append({"monitor": mon, "receiver": rx, "held_out": held, "share": sh, "threshold": thr, "train_clean": n_all, "train_clean_defined": n_def, "train_share_all": tr_all, "train_share_defined": tr_def})  # the point
                    dec = (np.nan_to_num(v, nan=-np.inf) >= thr).astype(int)  # alarm where the held statistic reaches the threshold (silent where undefined; exact comparison, so the point holds on the training seconds)
                    fd, fs = B12.both(g, dec, dec, held)  # through the environment script's stream (stopping mode applies the refractory rule)
                    base = {"monitor": mon, "receiver": rx, "outer": held, "share": sh, "param": thr}  # the run's identity
                    real = float((dec.astype(bool) & cl).sum() / cl.sum()) if cl.sum() else np.nan  # the realised decision-mode share on the evaluated session (for the 'point not held' mark)
                    # dict unpacking over two lines: the run with the second clean share and the realised share against the target
                    runs.append({**base, **B12.run_row(fd, fs), "dec_clean_defined": int((cl & dfn).sum()), "dec_clean_share_defined": float((dec.astype(bool) & cl).sum() / (cl & dfn).sum()) if (cl & dfn).sum() else np.nan,
                                 "realised_share": real, "point_not_held": bool(real > 2 * sh) if np.isfinite(real) else False})  # dict literal continued: the realised share and its mark
                    mk = {m_["onset"]: m_ for m_ in marks(sec, dec, dec, dfn, held)}  # dict comprehension: each interval's marks (decided seconds: those where the statistic is defined)
                    lat += [{**base, **x, **{k: v_ for k, v_ in mk[x["onset"]].items() if k != "onset"}} for x in fs["latency"]]  # list comprehension: its intervals with their marks
                for prm in sorted(set(GRID[mon][rx] if mon == "gain" else GRID[mon]) | set(FIXED[mon][rx] if mon == "gain" else FIXED[mon])):  # for loop over the curve's thresholds and the evidence script's fixed settings (set union)
                    dec = (np.nan_to_num(v, nan=-np.inf) >= prm).astype(int)  # the decisions
                    fd, fs = B12.both(g, dec, dec, held)  # through the stream
                    curve.append({"monitor": mon, "receiver": rx, "outer": held, "param": float(prm), "fixed_05d": prm in (FIXED[mon][rx] if mon == "gain" else FIXED[mon]), "clean": fd["clean"], "clean_in_alarm": fd["clean_in_alarm"], "clean_defined": int((cl & dfn).sum()), "intervals": fs["intervals"], "delays": ";".join(str(x["delay_s"]) for x in fs["latency"] if x["detected"])})  # the curve row
    return S_, pd.DataFrame(pts), pd.DataFrame(runs), pd.DataFrame(lat), pd.DataFrame(curve), pd.DataFrame(cov)  # every table


def curves(C: pd.DataFrame) -> pd.DataFrame:  # function: the monitors' operating curves pooled over the five position folds (2.6.12 separately)
    rows = []  # one row per monitor, receiver, scope and threshold
    for scope, d0 in (("position sessions", C[C.outer != SEALED]), ("2.6.12", C[C.outer == SEALED])):  # for loop over the two scopes
        for (mon, rx, prm), d in d0.groupby(["monitor", "receiver", "param"], sort=True):  # for loop over the curve points
            dl = [float(x) for s in d.delays.fillna("") for x in str(s).split(";") if x not in ("", "nan")]  # every detected interval's delay
            # dict literal over two lines: the point
            rows.append({"scope": scope, "monitor": mon, "receiver": rx, "param": prm, "fixed_05d": bool(d.fixed_05d.iloc[0]), "clean_share": d.clean_in_alarm.sum() / d.clean.sum() if d.clean.sum() else np.nan, "clean_share_defined": d.clean_in_alarm.sum() / d.clean_defined.sum() if d.clean_defined.sum() else np.nan,
                         "median_delay_s": float(np.median(dl)) if dl else np.nan, "miss_share": 1 - len(dl) / d.intervals.sum() if d.intervals.sum() else np.nan, "intervals": int(d.intervals.sum())})  # dict literal continued
    return pd.DataFrame(rows)  # the table


# ----------------------------------------------------------------------------------------------------------------------
# 2. the detectors' intervals with both marks: the agent experiment's point runs repeated on the position folds (gated), 2.6.12 from the transfer script's tables
# ----------------------------------------------------------------------------------------------------------------------
def agent_marks() -> pd.DataFrame:  # function: every the agent experiment point run repeated on its held-out session (frozen agents, committed predictions): per interval its delay and both marks; gated against heldout_latency.csv
    pts = B12.rt(B12.EVAL / "deployable_points.csv")  # the committed points
    sel = B12.rt(B12.GRID / "selection.csv")  # the frozen budgets
    LATr = B12.rt(B12.EVAL / "heldout_latency.csv")  # the agent experiment's registered per-interval delays
    steps = T.read_steps()  # the environment script's step table
    preds = {m: B12.rt(E.PROC / f"{'logistic_regression' if m == 'lr' else 'random_forest'}" / f"{'logistic_regression' if m == 'lr' else 'random_forest'}_predictions_primary.csv.gz") for m in ("lr", "rf")}  # dict comprehension: the committed outer probabilities
    rows, bad = [], []  # the interval rows; gate failures
    for rx in RXS:  # for loop over the receivers
        for outer in SESS5:  # for loop over the folds
            sd = T.session_seconds(steps, rx, outer)  # the held-out session's windowed seconds
            pred = {m: p[(p.receiver == rx) & (p.held_out == outer)].set_index("sec").p for m, p in preds.items()}  # dict comprehension: the probabilities by second
            mat = B12.heldout_material(rx, outer)  # the fold's material over the held-out rows
            _, ok, Z = B12.grid_rows(mat, sd)  # the grid's validity and log-odds
            sec = sd.sec.to_numpy(np.int64)  # the seconds
            P_ = pts[(pts.receiver == rx) & (pts.outer == outer)]  # the fold's points
            jobs = []  # (identity, decisions, stopping input)
            for r in P_[P_.detector.isin(B12.BASE)].itertuples():  # for loop over the baselines' points
                n, k = (int(r.n), int(r.k)) if r.detector == "kofn" else (0, 0)  # k-of-n's setting
                d_, s_ = B12.baseline_decisions(r.detector, float(r.param), n, k, sd, pred, ok, Z)  # the decisions (the evaluation script's, unchanged)
                jobs.append(({"detector": r.detector, "share": float(r.share), "lambda": np.nan, "seed": -1}, d_, s_))  # kept
            done = set()  # (arm, lambda) already run (two shares can share a lambda)
            for r in P_[P_.detector.str.startswith("agent")].itertuples():  # for loop over the agent's points
                if str(r.no_point) == "True" or (r.detector, float(r.param)) in done:  # no point, or this lambda is already run
                    continue  # next
                done.add((r.detector, float(r.param)))  # remembered
                arm, lam = r.detector.split("_", 1)[1], float(r.param)  # tuple assignment: the arm and the lambda
                b = int(sel[(sel.receiver == rx) & (sel.outer == outer) & (sel.arm == arm) & (sel["lambda"] == lam)].selected_budget.iloc[0])  # its frozen budget
                for seed in T.OUTER_SEEDS:  # for loop over the five seeds
                    spec = {"stage": "outer", "receiver": rx, "outer": outer, "inner": "", "arm": arm, "lambda": lam, "seed": seed, "budget": b}  # the frozen agent
                    model, _ = T.load_q(spec, B12.MODELS, mat)  # its weights
                    d_, s_ = T.agent_decisions(model, E.SpoofingDetectionEnv(mat, arm, lam), sd)  # decision and stopping mode
                    jobs.append(({"detector": r.detector, "share": np.nan, "lambda": lam, "seed": seed}, d_, s_))  # kept
            for ident, d_, s_ in jobs:  # for loop over the runs
                fd, fs = B12.both(sd, d_, s_, outer)  # through the environment script's stream
                q = LATr[(LATr.receiver == rx) & (LATr.outer == outer) & (LATr.detector == ident["detector"]) & ((LATr.share == ident["share"]) if ident["seed"] < 0 else ((LATr["lambda"] == ident["lambda"]) & (LATr.seed == ident["seed"])))].sort_values("onset")  # the agent experiment's registered intervals of this run
                mine = pd.DataFrame(fs["latency"]).sort_values("onset")  # the repeat's intervals
                same = len(q) == len(mine) and np.array_equal(q.onset.to_numpy(), mine.onset.to_numpy()) and np.array_equal(q.detected.astype(bool).to_numpy(), mine.detected.astype(bool).to_numpy()) and np.array_equal(q.delay_s.to_numpy(float), mine.delay_s.to_numpy(float), equal_nan=True)  # bitwise equal, NaN = NaN
                if not same:  # if: the repeat differs
                    bad.append(f"{rx} {outer} {ident}")  # recorded
                decided = np.ones(len(sec), bool) if ident["detector"] in X.OWN_ALL else ok  # the detector's own decided seconds
                mk = {m_["onset"]: m_ for m_ in marks(sec, d_, s_, decided, outer)}  # dict comprehension: each interval's marks
                rows += [{"receiver": rx, "outer": outer, **ident, **x, **{k: v for k, v in mk[x["onset"]].items() if k != "onset"}} for x in fs["latency"]]  # list comprehension: the intervals with their marks
        print(f"agent point runs repeated, {rx}: {len(rows)} interval rows", flush=True)  # progress
    check("the agent experiment's point runs repeated on their held-out sessions reproduce heldout_latency.csv bitwise (every interval of every run)", not bad, f"{len(rows)} intervals; mismatches {bad[:3]}")  # recorded
    return pd.DataFrame(rows)  # the table


def summarise(d: pd.DataFrame) -> dict:  # function: one detector's interval over its runs (agents: the seeds): detected, the counted delay (no mark) and the all-included delay, and the marks
    det, cnt = d[d.detected.astype(bool)], d[d.detected.astype(bool) & ~d.no_detection_delay.astype(bool)]  # tuple assignment: the detecting runs; those whose delay counts
    # dict literal over two lines: the summary
    return {"runs": len(d), "detecting_runs": len(det), "detected": bool(len(det)), "counted_runs": len(cnt), "counted": bool(len(cnt)), "delay_s": float(cnt.delay_s.median()) if len(cnt) else np.nan, "delay_all_s": float(det.delay_s.median()) if len(det) else np.nan,
            "in_alarm_runs": int((d.onset_state == "in alarm").sum()), "refractory_runs": int(d.refractory_at_onset.astype(bool).sum()), "marked_runs": int(d.no_detection_delay.astype(bool).sum())}  # dict literal continued: the marks


def detector_intervals(MK12: pd.DataFrame) -> pd.DataFrame:  # function: one row per detector, receiver, share and interval at the points (agents: seeds summarised), both scopes
    rows = []  # the rows
    p12 = B12.rt(B12.EVAL / "deployable_points.csv")  # the agent experiment's points
    on13 = X.rt(X.U2612 / "onset_states.csv")  # the transfer script's 2.6.12 intervals with both marks (its recomputation)
    p13 = X.rt(X.UNSEEN / "points.csv").assign(outer=SEALED)  # Part B's points
    for scope, tab, pts in (("position sessions", MK12, p12), ("2.6.12", on13, p13)):  # for loop over the two scopes
        for r in pts.itertuples():  # for loop over the points
            if r.detector.startswith("agent"):  # an agent
                if str(r.no_point) == "True":  # no lambda meets the share
                    continue  # no runs
                d = tab[(tab.receiver == r.receiver) & (tab.outer == r.outer) & (tab.detector == r.detector) & (tab["lambda"] == r.param)]  # its seeds' intervals
            else:  # a baseline
                d = tab[(tab.receiver == r.receiver) & (tab.outer == r.outer) & (tab.detector == r.detector) & (tab.share == r.share)]  # its intervals
            for onset, g in d.groupby("onset", sort=True):  # for loop over the intervals
                rows.append({"scope": scope, "detector": r.detector, "receiver": r.receiver, "session": r.outer, "share": r.share, "onset": int(onset), "end": int(g.end.iloc[0]), **summarise(g)})  # the row
    return pd.DataFrame(rows)  # the table


def monitor_intervals(LATm: pd.DataFrame) -> pd.DataFrame:  # function: the monitors' intervals in the same form
    rows = []  # the rows
    for (mon, rx, held, sh, onset), g in LATm.groupby(["monitor", "receiver", "outer", "share", "onset"], sort=True):  # for loop over the monitor intervals (one run each)
        rows.append({"scope": "2.6.12" if held == SEALED else "position sessions", "detector": mon, "receiver": rx, "session": held, "share": sh, "onset": int(onset), "end": int(g.end.iloc[0]), **summarise(g)})  # the row
    return pd.DataFrame(rows)  # the table


def own_onsets() -> dict:  # function: (monitor or 'gain' for the detectors, receiver) -> {(session, onset)} whose onset that receiver's own monitored quantity set
    on = pd.read_csv(E.PROC / "transfer" / "unseen_attack_onset_sources.csv", dtype={"test_code": str})  # the stated quiet levels
    # dict comprehension over two lines: own gain and own position per receiver
    return {**{("gain", rx): {(r.test_code, int(r.start)) for r in on.itertuples() if bool(getattr(r, f"onset_own_gain_{rx}"))} for rx in RXS},
            **{("position", rx): {(r.test_code, int(r.start)) for r in on.itertuples() if bool(getattr(r, f"onset_own_position_{rx}"))} for rx in RXS}}  # dict comprehension continued: own position per receiver


def comparison(IV: pd.DataFrame, runs_m: pd.DataFrame, cov: pd.DataFrame) -> pd.DataFrame:  # function: every detector and monitor per receiver, scope and share: counted and all-included delay, misses, marks, two clean shares, the realised share, coverage, and the decision-5 sensitivity
    own = own_onsets()  # the circular intervals
    steps = T.read_steps()  # the environment script's steps (decidable clean seconds of the sequential detectors)
    g12 = pd.read_csv(X.U2612 / "grid.csv.gz")  # the 2.6.12 grid
    agent_runs = B12.rt(B12.EVAL / "point_runs.csv")  # the agent experiment's runs at the points
    unseen_runs = X.rt(X.U2612 / "point_runs.csv")  # Part B's runs at the points
    rows = []  # one row per receiver, scope, share and detector
    for (scope, rx, sh, det), d in IV.groupby(["scope", "receiver", "share", "detector"], sort=True):  # for loop over the cells
        circ = own[("position" if det == "position" else "gain", rx)]  # the sensitivity drops onsets set by the monitored quantity (the gain for the gain monitor and, for comparison, for the detectors)
        ds = d[~d.apply(lambda r: (r.session, int(r.onset)) in circ, axis=1)]  # lambda: the intervals not set by that quantity
        if det in MONITORS:  # a monitor: its own runs
            m = runs_m[(runs_m.monitor == det) & (runs_m.receiver == rx) & (runs_m.share == sh) & ((runs_m.outer == SEALED) == (scope == "2.6.12"))]  # its runs
            cin, call, cdef = m.dec_clean_in_alarm.sum(), m.dec_clean.sum(), m.dec_clean_defined.sum()  # clean in alarm, all clean, defined clean
            c = cov[(cov.monitor == det) & (cov.receiver == rx) & ((cov.session == SEALED) == (scope == "2.6.12"))]  # its coverage
            cov_clean = c.defined_clean_seconds.sum() / c.clean_seconds.sum() if c.clean_seconds.sum() else np.nan  # share of clean seconds defined
            cov_att = c.defined_attack_seconds.sum() / c.attack_seconds.sum() if c.attack_seconds.sum() else np.nan  # share of attack seconds defined
        else:  # a detector: its runs at the point, and its decidable clean seconds
            R_ = agent_runs if scope == "position sessions" else unseen_runs  # the runs of the scope
            m = R_[(R_.receiver == rx) & (R_.share == sh) & (R_.detector == det)]  # its runs (agents: every seed)
            cin, call = m.dec_clean_in_alarm.sum(), m.dec_clean.sum()  # clean in alarm, all clean
            seq = det not in X.OWN_ALL  # a sequential detector is silent where its state is undefined; the regression and the forest decide at every window
            if scope == "position sessions":  # the five folds
                cdef = sum(int(((steps.receiver == rx) & (steps.session == o) & steps.window & (steps.label == "clean") & (steps.valid if seq else True)).sum()) for o in m.outer)  # generator in sum(): defined clean seconds per run
            else:  # 2.6.12
                gg = g12[(g12.receiver == rx) & (g12.label == "clean")]  # its clean grid seconds
                cdef = len(m) * int((gg.decidable if seq else gg.label.notna()).sum())  # defined clean seconds over the runs
            cov_clean, cov_att = np.nan, np.nan  # coverage reported for the monitors
        real = float(cin / call) if call else np.nan  # the realised (pooled) decision-mode share of the cell
        cnt, det_ = d[d.counted.astype(bool)], d[d.detected.astype(bool)]  # tuple assignment: intervals with a counted delay; detected intervals
        # dict literal over six lines: the row
        rows.append({"scope": scope, "receiver": rx, "share": sh, "detector": det, "circular_gain": det == "gain", "intervals": len(d), "detected": int(d.detected.sum()), "counted": int(d.counted.sum()), "pooled_median_delay_s": float(cnt.delay_s.median()) if len(cnt) else np.nan,
                     "pooled_median_delay_all_s": float(det_.delay_all_s.median()) if len(det_) else np.nan, "miss_share": float(1 - d.detected.mean()) if len(d) else np.nan, "intervals_in_alarm": int((d.in_alarm_runs > 0).sum()), "intervals_refractory": int((d.refractory_runs > 0).sum()), "intervals_marked_all_runs": int((d.marked_runs == d.runs).sum()),  # dict literal continued: delays, misses and marks
                     "clean_share_all": real, "clean_share_defined": float(cin / cdef) if cdef else np.nan, "point_not_held": bool(real > 2 * sh) if np.isfinite(real) else False, "coverage_clean": cov_clean, "coverage_attack": cov_att,  # dict literal continued: shares and coverage
                     "sens_quantity": "own position" if det == "position" else "own gain", "sens_intervals": len(ds), "sens_median_delay_s": float(ds[ds.counted.astype(bool)].delay_s.median()) if ds.counted.astype(bool).any() else np.nan, "sens_miss_share": float(1 - ds.detected.mean()) if len(ds) else np.nan,  # dict literal continued: the decision-5 sensitivity
                     "note": "no position monitor: the mosaic-H reports no position" if (rx == "SBF_03" and det == "gain") else ""})  # dict literal continued
    for scope in ("position sessions", "2.6.12"):  # the mosaic-H's missing monitor, stated as rows of its own
        for sh in SHARES:  # for loop over the shares
            rows.append({"scope": scope, "receiver": "SBF_03", "share": sh, "detector": "position", "circular_gain": False, "intervals": 0, "detected": 0, "counted": 0, "note": "no position monitor: the mosaic-H reports no position in any session"})  # the row
    return pd.DataFrame(rows)  # the table


# ----------------------------------------------------------------------------------------------------------------------
# 3. delay against the gain depression the receiver registered
# ----------------------------------------------------------------------------------------------------------------------
def depression(IV: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:  # function: each interval's median gain depression below the receiver's quiet level, the band selected explicitly
    eai = pd.read_csv(E.PROC / "ground_truth" / "effective_attack_interval.csv", dtype={"test_code": str})  # the receiver-status script's quiet references
    rows = []  # one row per receiver, session and interval
    for (rx, t, a, b), _ in IV.groupby(["receiver", "session", "onset", "end"], sort=True):  # for loop over the intervals
        band, field = BAND[rx]  # tuple assignment: the band and field the evidence script reads for this receiver
        qr = eai[(eai.receiver == D.RX_STATUS[rx]) & (eai.test_code == t) & (eai.band == band) & (eai.field == field)]  # its quiet level, by band and field
        q = float(qr.quiet_reference.iloc[0])  # the value
        g = D.gain_series(st, t, rx)  # 05d: the gain on a one-second grid (the same band)
        w = g[(g.index >= a) & (g.index <= b)].dropna()  # the interval's readings
        dep = float(np.median(q - w.to_numpy())) if len(w) else np.nan  # the median depression (dB, or counts on the u-blox)
        rows.append({"receiver": rx, "session": t, "onset": int(a), "end": int(b), "band": band, "quiet_level": q, "quiet_rows": len(qr), "readings": len(w), "depression": dep, "unit": "counts" if rx == "UBX_01" else "dB", "depression_db": dep / COUNTS_PER_DB if rx == "UBX_01" else dep})  # the row (dB for the pooled rho)
    out = pd.DataFrame(rows)  # the table
    check("the quiet levels are 36 dB, 39 dB and 5,265 counts in every session, one row each (ground_truth/effective_attack_interval.csv, band and field explicit)", all((g.quiet_level == QUIET[rx]).all() and (g.quiet_rows == 1).all() for rx, g in out.groupby("receiver")), str(out.groupby("receiver").quiet_level.unique().to_dict()))  # generator in all(): recorded
    return out  # the table


def spearman_boot(x: np.ndarray, y: np.ndarray, seed: int = SEED, n_boot: int = N_BOOT) -> tuple:  # function: Spearman's rho with a 95 % percentile bootstrap interval over the pairs; (rho, low, high, n, finite resamples)
    from scipy.stats import spearmanr  # rank correlation
    x, y = np.asarray(x, float), np.asarray(y, float)  # arrays
    n = len(x)  # pairs
    if n < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:  # if: too few pairs or a constant variable
        return np.nan, np.nan, np.nan, n, 0  # undefined
    rho = float(spearmanr(x, y).statistic)  # the point estimate
    rng = np.random.default_rng(seed)  # the seeded generator
    idx = rng.integers(0, n, size=(n_boot, n))  # every resample's pairs
    bs = []  # the resampled rhos
    for k in range(n_boot):  # for loop over the resamples
        xi, yi = x[idx[k]], y[idx[k]]  # the resample
        if np.unique(xi).size > 1 and np.unique(yi).size > 1:  # if: both vary (rho defined)
            bs.append(spearmanr(xi, yi).statistic)  # its rho
    bs = np.asarray(bs, float)  # array
    return rho, float(np.percentile(bs, 2.5)) if len(bs) else np.nan, float(np.percentile(bs, 97.5)) if len(bs) else np.nan, n, len(bs)  # rho, interval, pairs, defined resamples


def strength(IV: pd.DataFrame, DEP: pd.DataFrame) -> tuple:  # function: rho per receiver and detector (primary) and pooled (secondary), on counted delays, all-included delays, and without the circular intervals
    own = own_onsets()  # the circular intervals
    d = IV[IV.share == 0.05].merge(DEP, on=["receiver", "session", "onset", "end"], how="left", validate="many_to_one")  # every interval at 0.05 with its depression
    d["circular"] = d.apply(lambda r: (r.session, int(r.onset)) in own[("position" if r.detector == "position" else "gain", r.receiver)], axis=1)  # lambda: set by the receiver's own monitored quantity
    rows = []  # one row per subset, level, receiver and detector
    for subset, dd, flag, col in (("counted delays (primary; marked delays left out)", d, "counted", "delay_s"), ("all-included sensitivity (marked delays kept)", d, "detected", "delay_all_s"), ("counted delays without own-quantity onsets", d[~d.circular], "counted", "delay_s")):  # tuple of (subset, pairs, which intervals, which delay)
        for (rx, det), g in dd.groupby(["receiver", "detector"], sort=True):  # per receiver (primary)
            g = g[g[flag].astype(bool) & g.depression.notna()]  # the intervals of the subset with a depression
            rho, lo, hi, n, nb = spearman_boot(g.depression.to_numpy(), g[col].to_numpy())  # rho and its interval
            rows.append({"subset": subset, "level": "per receiver (primary)", "receiver": rx, "detector": det, "pairs": n, "rho": rho, "ci_low": lo, "ci_high": hi, "defined_resamples": nb, "unit": "counts" if rx == "UBX_01" else "dB"})  # the row
        for det, g in dd.groupby("detector", sort=True):  # pooled over the receivers (secondary)
            g = g[g[flag].astype(bool) & g.depression_db.notna()]  # the intervals of the subset
            rho, lo, hi, n, nb = spearman_boot(g.depression_db.to_numpy(), g[col].to_numpy())  # rho on dB (u-blox counts / 100)
            rows.append({"subset": subset, "level": "pooled over receivers (secondary; u-blox at about 100 counts per dB)", "receiver": "all", "detector": det, "pairs": n, "rho": rho, "ci_low": lo, "ci_high": hi, "defined_resamples": nb, "unit": "dB"})  # the row
    return d, pd.DataFrame(rows)  # the pairs and the rhos


# ----------------------------------------------------------------------------------------------------------------------
# 4. cost
# ----------------------------------------------------------------------------------------------------------------------
def hardware() -> dict:  # function: the machine the timings ran on
    import sklearn  # versions
    import torch  # versions
    cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()  # the processor
    mem = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip()  # the memory in bytes
    ncpu = subprocess.run(["sysctl", "-n", "hw.ncpu"], capture_output=True, text=True).stdout.strip()  # logical processors
    # dict literal over two lines: the record
    return {"processor": cpu, "logical_cpus": ncpu, "memory_gb": round(int(mem) / 2 ** 30, 1) if mem.isdigit() else "", "system": f"{platform.system()} {platform.release()} ({platform.machine()})", "python": platform.python_version(),
            "numpy": np.__version__, "scikit-learn": sklearn.__version__, "torch": torch.__version__, "stable-baselines3": T.sb3.__version__, "torch_threads": torch.get_num_threads(), "thread_env": ",".join(f"{v}={os.environ.get(v)}" for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"))}  # dict literal continued


def timed(fn, n: int) -> np.ndarray:  # function: n calls of fn(i), each timed alone with perf_counter_ns; returns microseconds
    out = np.empty(n)  # the times
    for i in range(n):  # for loop over the calls
        t0 = time.perf_counter_ns()  # start
        fn(i)  # the call
        out[i] = (time.perf_counter_ns() - t0) / 1000.0  # microseconds
    return out  # the times


def base_rows() -> pd.DataFrame:  # function: the feature script's per-second base table of the five position sessions, read in chunks so that no 2.6.12 row is kept
    parts = []  # the kept chunks
    for ch in pd.read_csv(E.PROC / "features" / "features_base_1hz.csv.gz", dtype={"test_code": str}, chunksize=20_000):  # for loop over chunks of the file
        parts.append(ch[ch.test_code != SEALED])  # 2.6.12's rows dropped as they are read
    return pd.concat(parts, ignore_index=True)  # the position sessions' rows


def cost(n_dec: int = 10_000) -> tuple:  # function: the static cost (parameters, depth, MACs) and the timings, per receiver, of the five-session P and Part B's agents
    import torch  # PyTorch: the agents' networks
    S07 = _load("window_features_for_delay_monitors", "4_features/03_window_features.py")  # 07, unchanged: win_stats
    base = base_rows()  # the feature script's base table, position sessions only
    A.use_labels("final")  # the scoring module points at the final label set
    wc10 = A.window_dict()  # the feature script's window dictionary at w = 10
    cells = A.d1_cells(A.load_cells(A.FINAL, A.W, True), wc10)  # the position windows
    sel = X.rt(X.UNSEEN / "selection.csv")  # Part B's budgets
    fits = X.rt(X.UNSEEN / "fits.csv")  # the frozen digests
    static, tim = [], []  # the two tables
    check("torch decides on one thread", torch.get_num_threads() == 1, str(torch.get_num_threads()))  # recorded (12 sets it on import)
    for rx in RXS:  # for loop over the receivers
        idx, Xw = cells[rx]  # its windows
        rec, _ = X.fit_p(idx, Xw, wc10, SESS5, {}, {"run": "R14cost", "receiver": rx})  # the five-session P
        fp, fz = X.fingerprint(rec), fits[(fits.receiver == rx) & (fits.fit == "five sessions")].iloc[0]  # tuple assignment: its digests and the frozen ones
        check(f"{rx}: the timed five-session P is the frozen one (transform, regression, forest digests)", all(fp[k] == fz[k] for k in fp), fp["transform_sha256"][:12])  # recorded
        lr, rf = rec["model"]["lr"], rec["model"]["rf"]  # the two models
        check(f"{rx}: the forest predicts with n_jobs 1", rf.n_jobs == 1, str(rf.n_jobs))  # recorded (10 sets it after the fit)
        Zall = A.transform(rec, Xw, wc10)  # every window's 36 columns (for the model timings)
        zall = A.decision(lr, Zall)  # every window's log-odds
        rows_i = np.arange(n_dec) % len(Zall)  # the windows the decisions use, in order (cycling)
        mat = X.material_b(rx, X.OUTER_B)  # Part B's material (the agents' dimensions)
        agents = {}  # arm -> (model, observation of each decision)
        for arm in ("primary", "surrogate"):  # for loop over the arms
            b = int(sel[(sel.receiver == rx) & (sel.arm == arm) & (sel["lambda"] == 100.0)].selected_budget.iloc[0])  # the frozen budget at lambda 100
            spec = {"stage": "outer", "receiver": rx, "outer": X.OUTER_B, "inner": "", "arm": arm, "lambda": 100.0, "seed": SEED, "budget": b}  # the agent timed: lambda 100, seed 20260914 (the network is the same at every lambda)
            mdl, _ = T.load_q(spec, X.MODELS_B, mat)  # its frozen weights
            if arm == "primary":  # the primary arm: its state rows
                O = mat["primary"]  # 108 values per decision
            else:  # the surrogate arm: its 30 log-odds with S_t, compressed, as the training script's agent_decisions builds them
                O = np.vstack([T.signed_log(np.column_stack([mat["surrogate"][r_], S_])) for _, r_, S_ in (T.surrogate_runs(mat, s_) for s_ in SESS5)])  # nested generators: every session's rows with its S_t
            agents[arm] = (mdl, O.astype(np.float32))  # kept
        base_rx = base[base.receiver == rx].sort_values(["test_code", "sec"])  # its base rows
        bcols = [c for c in base_rx.columns if c not in ("test_code", "receiver", "sec")]  # the base columns
        Bm = base_rx[bcols].to_numpy(float)  # as a matrix
        nodes = int(sum(e.tree_.node_count for e in rf.estimators_))  # the forest's nodes
        depth = [e.get_depth() for e in rf.estimators_]  # each tree's depth
        path = rf.decision_path(Zall[rows_i[:2000]])[0]  # node indicator of 2,000 decisions
        comps = float(np.asarray(path.sum(axis=1)).ravel().mean() - len(rf.estimators_))  # mean nodes visited minus one leaf per tree: comparisons per decision
        net = lambda m: [l for l in m.q_net.q_net if isinstance(l, torch.nn.Linear)]  # lambda: the Q-network's linear layers
        for arm in ("primary", "surrogate"):  # the two arms' static figures
            L_ = net(agents[arm][0])  # its layers
            static.append({"receiver": rx, "detector": f"agent_{arm}", "parameters": int(sum(p.numel() for p in agents[arm][0].q_net.parameters())), "depth": f"{len(L_)} linear layers ({'-'.join(str(l.in_features) for l in L_)}-{L_[-1].out_features})", "macs_per_decision": int(sum(l.in_features * l.out_features for l in L_)) + (len(rec["sel"]) if arm == "surrogate" else 0), "comparisons_per_decision": 1, "note": "online network only (the target network is not used to decide); timed at lambda 100, seed 20260914" + ("; plus the regression's 36 MACs for z_t" if arm == "surrogate" else "")})  # the row
        # list of rows over four lines: the baselines' static figures
        static += [{"receiver": rx, "detector": "lr", "parameters": int(lr.coef_.size + lr.intercept_.size), "depth": "1 linear layer", "macs_per_decision": int(lr.coef_.size), "comparisons_per_decision": 1, "note": "decision value and a threshold"},
                   {"receiver": rx, "detector": "rf", "parameters": nodes, "depth": f"max {max(depth)}, mean {np.mean(depth):.1f} over {len(depth)} trees", "macs_per_decision": 0, "comparisons_per_decision": round(comps, 1), "note": "parameters = nodes (each a split or a leaf); comparisons = mean nodes visited less the leaves, 2,000 decisions"},  # list continued: the forest
                   {"receiver": rx, "detector": "cusum", "parameters": int(lr.coef_.size + lr.intercept_.size) + 1, "depth": "the regression and one recursion", "macs_per_decision": int(lr.coef_.size) + 1, "comparisons_per_decision": 2, "note": "z_t from the regression, S_t = max(0, S_(t-1) + z_t), S_t >= h"},  # list continued: the cumulative sum
                   {"receiver": rx, "detector": "kofn", "parameters": int(lr.coef_.size + lr.intercept_.size) + 3, "depth": "the regression and a count", "macs_per_decision": int(lr.coef_.size), "comparisons_per_decision": 10, "note": "z_t from the regression; at most 10 values sorted and compared (n of 10, the largest n)"}]  # list continued
        Xrow = [Xw.iloc[[i]] for i in rows_i[:1000]]  # one-row frames: the transform's input (1,000 built before the clock and cycled, to spare memory)
        Zs = np.ascontiguousarray(Zall[rows_i])  # the transformed rows
        jobs = {"transform (P's fitted steps, one window)": lambda i: A.transform(rec, Xrow[i % 1000], wc10),  # dict of timed calls: P's transform
                "window statistics (one 07 win_stats call on the 10 base rows of a window)": lambda i: S07.win_stats(Bm[(i % (len(Bm) - 10)):(i % (len(Bm) - 10)) + 10], 10),  # the feature script's statistics, one call per window (07 also forms the 9 incomplete windows of the call, which it drops)
                "lr": lambda i: lr.decision_function(Zs[i:i + 1]),  # the regression's decision value
                "rf": lambda i: rf.predict_proba(Zs[i:i + 1]),  # the forest's probability
                "cusum (recursion only; plus lr)": None, "kofn (window only; plus lr)": None,  # filled below (they keep state)
                "agent_primary": lambda i: agents["primary"][0].q_net(torch.as_tensor(agents["primary"][1][i % len(agents["primary"][1])][None])),  # the online network's forward pass
                "agent_surrogate": lambda i: agents["surrogate"][0].q_net(torch.as_tensor(agents["surrogate"][1][i % len(agents["surrogate"][1])][None]))}  # the same for the surrogate arm
        state = {"S": 0.0, "w": deque(maxlen=10)}  # CUSUM's S and k of n's window
        h = 5.0  # an illustrative threshold (its value does not change the cost)

        def cus(i):  # nested function: one CUSUM update
            state["S"] = max(0.0, state["S"] + float(zall[rows_i[i]]))  # Page's recursion
            return state["S"] >= h  # the decision

        def kon(i):  # nested function: one k-of-n update
            state["w"].append(float(zall[rows_i[i]]))  # the newest log-odds
            return np.sort(np.fromiter(state["w"], float))[-4] >= h if len(state["w"]) >= 4 else False  # the 4th largest of the last 10 against h

        jobs["cusum (recursion only; plus lr)"], jobs["kofn (window only; plus lr)"] = cus, kon  # tuple assignment: filled in
        with torch.no_grad():  # no gradient bookkeeping during the forward passes
            for name, fn in jobs.items():  # for loop over the timed calls
                for _ in range(50):  # a short warm-up, untimed
                    fn(0)  # one call
                us = timed(fn, n_dec)  # the 10,000 timed decisions
                tim.append({"receiver": rx, "item": name, "decisions": n_dec, "median_us": float(np.median(us)), "p25_us": float(np.percentile(us, 25)), "p75_us": float(np.percentile(us, 75)), "p99_us": float(np.percentile(us, 99))})  # the row
        print(f"cost {rx} done", flush=True)  # progress
    return pd.DataFrame(static), pd.DataFrame(tim)  # both tables


def cost_summary(tim: pd.DataFrame) -> pd.DataFrame:  # function: per receiver and detector, the median time per decision with and without the transform, against the four intervals
    rows = []  # one row per receiver and detector
    for rx, d in tim.groupby("receiver", sort=True):  # for loop over the receivers
        t = d.set_index("item").median_us  # the medians
        tr, ws = t["transform (P's fitted steps, one window)"], t["window statistics (one 07 win_stats call on the 10 base rows of a window)"]  # the transform and the window statistics
        parts = {"lr": t["lr"], "rf": t["rf"], "cusum": t["lr"] + t["cusum (recursion only; plus lr)"], "kofn": t["lr"] + t["kofn (window only; plus lr)"], "agent_primary": t["agent_primary"], "agent_surrogate": t["lr"] + t["agent_surrogate"]}  # dict: the model's own time per decision
        for det, m in parts.items():  # for loop over the detectors
            full = m + tr + ws  # model, transform and window statistics (the feature script's per-second aggregation from the RINEX is not timed)
            # dict literal over two lines: the row
            rows.append({"receiver": rx, "detector": det, "model_us": m, "transform_us": tr, "window_stats_us": ws, "total_us": full, "share_of_1s": full / 1e6, "share_of_receiver_interval": full / {"SBF_02": 1e5, "SBF_03": 5e4, "UBX_01": 1e6}[rx],
                         "receiver_interval_ms": {"SBF_02": 100, "SBF_03": 50, "UBX_01": 1000}[rx], "fits_receiver_interval": full < {"SBF_02": 1e5, "SBF_03": 5e4, "UBX_01": 1e6}[rx]})  # dict literal continued
    return pd.DataFrame(rows)  # the table


# ----------------------------------------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------------------------------------
def figures(CV: pd.DataFrame, PAIRS: pd.DataFrame, RHO: pd.DataFrame, CS: pd.DataFrame, cov: pd.DataFrame) -> None:  # function: the figures
    import matplotlib.pyplot as plt  # pyplot (09 chose the file-only backend)
    STY = {**B12.STYLE, "gain": ("#000000", "X", "gain-fall monitor (circular: 9 of 9 onsets set by gain)"), "position": ("#777777", "*", "position monitor")}  # dict: every detector's colour, marker and label
    wc = B12.rt(B12.EVAL / "curves.csv")  # the agent experiment's within-receiver curves
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)  # delay and misses per receiver
    for j, rx in enumerate(RXS):  # for loop over the receivers
        for det in B12.DET:  # the detectors' within-receiver curves, faint
            d = wc[(wc.receiver == rx) & (wc.detector == det)]  # its points
            if det == "kofn":  # one k-of-n setting: the most frequent at the points
                nk = B12.rt(B12.EVAL / "deployable_points.csv").query("receiver == @rx and detector == 'kofn'")[["n", "k"]].value_counts().idxmax()  # its (n, k)
                d = d[(d.n == nk[0]) & (d.k == nk[1])]  # that curve
            d = d.sort_values("clean_share")  # along the share axis
            for i, y in enumerate(("median_delay_s", "miss_share")):  # for loop over the rows
                axes[i, j].plot(d.clean_share, d[y], color=STY[det][0], lw=0.9, alpha=0.6, label=STY[det][2] if (i == 0 and j == 0) else None)  # the curve
        for mon in MONITORS:  # the monitors' curves, bold
            d = CV[(CV.scope == "position sessions") & (CV.monitor == mon) & (CV.receiver == rx)].sort_values("clean_share")  # its points
            for i, y in enumerate(("median_delay_s", "miss_share")):  # for loop over the rows
                if len(d):  # if: the monitor exists on this receiver
                    axes[i, j].plot(d.clean_share, d[y], color=STY[mon][0], lw=2.2, label=STY[mon][2] if (i == 0 and j == 0) else None)  # the curve over all clean seconds
                    axes[i, j].plot(d.clean_share_defined, d[y], color=STY[mon][0], lw=1.0, ls="--", label=f"{STY[mon][2].split(' (')[0]}, share over defined clean seconds" if (i == 0 and j == 0) else None)  # the same over defined clean seconds
                    f = d[d.fixed_05d]  # the evidence script's fixed settings
                    axes[i, j].scatter(f.clean_share, f[y], marker=STY[mon][1], s=40, color=STY[mon][0], zorder=4)  # marked
        if rx == "SBF_03":  # the mosaic-H
            axes[0, j].text(0.98, 0.95, "no position monitor:\nthe mosaic-H reports no position", transform=axes[0, j].transAxes, ha="right", va="top", fontsize=7)  # stated on the panel
        for i in range(2):  # for loop over the rows
            for s in SHARES:  # the registered shares
                axes[i, j].axvline(s, color="#888888", lw=0.8, ls=":")  # a dotted line
            axes[i, j].set_xscale("symlog", linthresh=0.01)  # low shares spread out
            axes[i, j].set_xlim(0, 1)  # every share
            axes[i, j].grid(alpha=0.3, lw=0.5)  # recessive grid
        axes[0, j].set_yscale("symlog", linthresh=10)  # signed log
        c = cov[(cov.receiver == rx) & (cov.session != SEALED)]  # coverage on the position sessions
        cg, cp = c[c.monitor == "gain"], c[c.monitor == "position"]  # tuple assignment: the two monitors' coverage
        axes[0, j].set_title(f"{NAME[rx]}: gain defined on {cg.defined_clean_seconds.sum()}/{cg.clean_seconds.sum()} clean s" + (f", position on {cp.defined_clean_seconds.sum()}/{cp.clean_seconds.sum()}" if len(cp) else ""), fontsize=8, loc="left")  # title with the coverage
        axes[1, j].set_xlabel("pooled clean-window alarm share (decision mode)", fontsize=8)  # axis
    axes[0, 0].set_ylabel("median delay (s), stopping mode", fontsize=8)  # axis
    axes[1, 0].set_ylabel("share of intervals missed", fontsize=8)  # axis
    h_, l_ = axes[0, 0].get_legend_handles_labels()  # legend entries
    fig.legend(h_, l_, loc="lower center", ncol=3, fontsize=7, frameon=False)  # one legend
    fig.tight_layout(rect=(0, 0.11, 1, 1))  # layout
    M.save_fig(fig, GRAPH / "monitor_curves.png")  # byte-stable save
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.2))  # one panel per receiver
    rr = RHO[RHO.subset.str.startswith("counted delays (primary") & RHO.level.str.startswith("per receiver")]  # the primary rhos
    for j, rx in enumerate(RXS):  # for loop over the receivers
        d = PAIRS[(PAIRS.receiver == rx) & PAIRS.detected.astype(bool)]  # its detected intervals at 0.05
        for q, det in enumerate(B12.DET + MONITORS):  # for loop over the detectors and monitors
            g = d[d.detector == det]  # its intervals
            if g.empty:  # nothing detected
                continue  # next
            r = rr[(rr.receiver == rx) & (rr.detector == det)]  # its rho
            lab = f"{DNAME[det]}: rho {r.rho.iloc[0]:+.2f} [{r.ci_low.iloc[0]:+.2f}, {r.ci_high.iloc[0]:+.2f}], n {int(r.pairs.iloc[0])}" if len(r) and pd.notna(r.rho.iloc[0]) else f"{DNAME[det]}: rho undefined, n {int(g.counted.astype(bool).sum())}"  # its legend text (counted pairs)
            jit = (q - 3.5) * 0.15 * (1 if rx != "UBX_01" else COUNTS_PER_DB)  # a small horizontal offset so overlapping points stay visible
            y_ = np.where(g.counted.astype(bool), g.delay_s, g.delay_all_s)  # the counted delay, or the all-included one where every run is marked
            axes[j].scatter(g.depression + jit, np.clip(y_, 0.5, None), marker=STY[det][1], s=40, facecolor=np.where(g.counted.astype(bool), STY[det][0], "white").tolist(), edgecolor=STY[det][0], label=lab)  # one point per interval; hollow: marked (in alarm or refractory at onset), left out of rho
        axes[j].set_yscale("log")  # delays span orders of magnitude
        axes[j].set_xlabel("median gain depression in the interval (" + ("AGC counts" if rx == "UBX_01" else "dB") + " below the quiet level)", fontsize=8)  # axis
        axes[j].set_title(NAME[rx], fontsize=9, loc="left")  # title
        axes[j].grid(alpha=0.3, lw=0.5)  # recessive grid
        axes[j].legend(fontsize=5.5, frameon=False, loc="upper right")  # its legend with the rhos
    axes[0].set_ylabel("delay at the 0.05 point (s; 0 drawn at 0.5)", fontsize=8)  # axis
    fig.text(0.5, 0.01, "filled: counted delay (in rho); hollow: in alarm or refractory at onset (no detection delay; left out of rho, kept in the all-included sensitivity)", ha="center", fontsize=7)  # the marker rule
    fig.tight_layout(rect=(0, 0.04, 1, 1))  # layout
    M.save_fig(fig, GRAPH / "delay_depression.png")  # byte-stable save
    fig, ax = plt.subplots(figsize=(10, 4.5))  # one panel
    dets = list(B12.DET)  # the six detectors
    for k, rx in enumerate(RXS):  # for loop over the receivers
        d = CS[CS.receiver == rx].set_index("detector").reindex(dets)  # its rows
        x = np.arange(len(dets)) + (k - 1) * 0.25  # dodged positions
        ax.bar(x, d.model_us, width=0.24, color=["#0072B2", "#D55E00", "#009E73"][k], label=f"{NAME[rx]}: model")  # the model's time
        ax.bar(x, d.transform_us + d.window_stats_us, width=0.24, bottom=d.model_us, color=["#0072B2", "#D55E00", "#009E73"][k], alpha=0.35, label=f"{NAME[rx]}: transform and window statistics")  # the rest
    for ms, lab_, ls in ((1000, "1 s (decision interval; u-blox 01 output)", "-"), (100, "100 ms (mosaic-X5 output)", "--"), (50, "50 ms (mosaic-H output)", ":")):  # for loop over the intervals
        ax.axhline(ms * 1000, color="#000000", lw=0.9, ls=ls, label=lab_)  # the interval
    ax.set_yscale("log")  # orders of magnitude
    ax.set_xticks(range(len(dets)), [DNAME[d] for d in dets], fontsize=8)  # list comprehension: names
    ax.set_ylabel("median time per decision (µs, one thread)", fontsize=8)  # axis
    ax.legend(fontsize=6.5, frameon=False, ncol=3, loc="upper left")  # legend
    ax.grid(alpha=0.3, lw=0.5, axis="y")  # recessive grid
    fig.tight_layout()  # layout
    M.save_fig(fig, GRAPH / "cost.png")  # byte-stable save


# ----------------------------------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------------------------------
def main() -> None:  # function: the RQ4 study in full
    ap = argparse.ArgumentParser()  # the parser
    ap.add_argument("--no-cost", action="store_true")  # skip the timing and rebuild the summary and the figure from the written timing file
    a = ap.parse_args()  # parsed
    t0 = time.time()  # clock (printed only)
    RQ4_DIR.mkdir(parents=True, exist_ok=True)  # the folder
    proof = X.guard()  # 2.6.12's gain and positions are read only if the freeze holds
    E.write_csv(proof, RQ4_DIR / "guard.csv")  # the proof
    gt = pd.read_csv(E.PROC / "ground_truth" / "session_ground_truth.csv", dtype={"test_code": str}).set_index("test_code")  # the timeline script's sessions, as the evidence script expects
    st = pd.read_csv(E.PROC / "ground_truth" / "receiver_status_timeseries.csv", dtype={"test_code": str}, usecols=["test_code", "receiver", "timestamp_utc", "band", "agc_gain_db", "agc_cnt"])  # the gains (label-side evidence; read after the guard)
    for rx in RXS:  # for loop over the receivers: 2.6.12's gain rows read, counted
        READS.append({"what": f"{NAME[rx]} gain readings in 2.6.12 ({BAND[rx][0]}, ground_truth/receiver_status_timeseries.csv, after the guard)", "rows": int(((st.test_code == SEALED) & (st.receiver == D.RX_STATUS[rx]) & (st.band == BAND[rx][0])).sum())})  # recorded
    G = grids()  # the shared stream grids
    S_, PTS, RUNS, LATm, C, COV = monitors(st, gt, G)  # 1: the monitors
    CV = curves(C)  # their curves
    MK12 = agent_marks()  # 2: the agent experiment's point runs repeated with both marks (gated)
    IV = pd.concat([detector_intervals(MK12), monitor_intervals(LATm)], ignore_index=True)  # every detector's and monitor's intervals at the points
    CMP = comparison(IV, RUNS, COV)  # the comparison table
    own = own_onsets()  # the circular intervals
    kept = {rx: 9 - len(own[("gain", rx)]) for rx in RXS}  # dict comprehension: intervals not set by each receiver's own gain, of the 9 pooled onsets
    check("the decision: the gain monitor's sensitivity keeps 3, 7 and 4 of 9 intervals (mosaic-X5, mosaic-H, u-blox 01)", [kept[r] for r in RXS] == [3, 7, 4], str(kept))  # recorded
    DEP = depression(IV, st)  # 3: each interval's depression
    PAIRS, RHO = strength(IV, DEP)  # rho
    for name, df in (("monitor_points.csv", PTS), ("monitor_runs.csv", RUNS), ("monitor_latency.csv", LATm), ("monitor_curves.csv", CV), ("monitor_coverage.csv", COV), ("agent_marks.csv", MK12),  # tuple of (file, table)
                     ("intervals.csv", IV), ("comparison.csv", CMP), ("depression.csv", DEP), ("strength_pairs.csv", PAIRS), ("spearman.csv", RHO), ("reads.csv", pd.DataFrame(READS))):  # tuple continued
        E.write_csv(df, RQ4_DIR / name)  # written
    E.write_gz(S_, RQ4_DIR / "monitor_series.csv.gz")  # the held statistics on every grid second
    E.write_gz(C, RQ4_DIR / "monitor_curve_runs.csv.gz")  # the curve runs
    if not a.no_cost:  # 4: the cost
        ST, TI = cost()  # static figures and timings
        E.write_csv(ST, RQ4_DIR / "cost_static.csv")  # deterministic
        E.write_csv(TI, RQ4_DIR / "cost_timing.csv")  # wall clock: non-deterministic, excluded from the byte comparison
        E.write_csv(pd.DataFrame([hardware()]), RQ4_DIR / "cost_hardware.csv")  # the machine
    TI = pd.read_csv(RQ4_DIR / "cost_timing.csv")  # the timings (this run's or the last run's)
    CS = cost_summary(TI)  # against the intervals
    E.write_csv(CS, RQ4_DIR / "cost_summary.csv")  # derived from the timings: non-deterministic too
    figures(CV, PAIRS, RHO, CS, COV)  # the figures to 68 (68 from the timings)
    ck = pd.DataFrame(CHECKS)  # the checks
    E.write_csv(ck, RQ4_DIR / ("checks_nocost.csv" if a.no_cost else "checks.csv"))  # written (the cost checks exist only when the cost ran)
    print(CMP[CMP.share == 0.05].to_string(index=False))  # printed
    print(RHO[RHO.level.str.startswith("per receiver")].to_string(index=False))  # printed
    print(f"RQ4: {time.time() - t0:.0f} s")  # wall time
    if not ck["pass"].all():  # if: a check failed
        sys.exit(1)  # exit status 1


if __name__ == "__main__":  # guard: runs main() only when the file is executed
    main()  # runs main
