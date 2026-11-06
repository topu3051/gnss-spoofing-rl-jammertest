#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
Behaviour of each observable in the attack groups against a clean baseline. Exploration only: the groups are
PROVISIONAL, nothing here is a label or a feature, and there are NO p-values (the seconds are autocorrelated). The bootstrap intervals
resample blocks of 30 s and of 60 s inside one session and receiver, so they describe variation within the session only, not variation
between the six events.

Groups (per session and receiver)
  relabelling (ground_truth/group_overrides.csv, written by the gain-pattern script): 2.6.5's observed jamming phase 12:36:00-12:40:00 (gain pattern on all three receivers) is the
                     jamming group, and the three scheduled-jamming seconds before it are 'scheduled-no-effect'
  clean              the position script clean intervals; where a session has fewer than MIN_CLEAN_S clean seconds the same receiver's clean seconds
                     of the other sessions are pooled instead ("pooled baseline"); per-satellite quantities are then compared
                     within 10 degree elevation bins
  jamming, spoofing  the provisional phases of the observables script (scheduled)
  effect             the receiver-status script's gain onset to gain recovery of the same receiver (independent evidence; no residual is used)
  uncertain   spoofing-phase seconds before the gain onset that show a residual anomaly (a satellite beyond 30 m for 3 s):
                     counted and reported, never assigned to a group

Clock: whole-millisecond jumps (multiples of 299,792.458 m) are found against the running median drift, removed before the drift is
formed, and counted. Every 2.6.12 u-blox 01 model-based quantity is marked "host-time oracle".

Reads   Data/processed/features/observables_1hz.csv.gz, features/observables_epochs_1hz.csv.gz, ground_truth/ta3_schedule_confirmation.csv, features/observables_missingness.csv
Writes  Data/processed/features/observable_effects.csv           one row per session, receiver, comparison, level and observable
        Data/processed/features/observable_group_stats.csv       median, IQR and missing fraction per group
        Data/processed/features/observable_leakage_flags.csv     per observable: missingness- and jamming-driven cells
        Data/processed/features/observable_cn0_behaviour.csv     the C/N0 question per session and receiver
        Data/processed/features/observable_glonass_report.csv    GLONASS observables per spoofing phase
        Data/processed/features/observable_uncertain.csv         the uncertain seconds
        Data/processed/features/observable_clock_jumps.csv       the millisecond jumps removed
        Data/processed/features/observable_effects_checks.csv    the self-checks
        Graph/observable_effects_heatmap.png, Graph/cn0_through_windows.png
"""

from __future__ import annotations  # future import: type hints kept as text (a convention across the project's scripts)

import importlib.util  # standard-library module loader, to load the evidence script whose name starts with a digit
import multiprocessing as mp  # standard-library process pool, to run the bootstrap cells in parallel
import os  # operating-system access: MPLCONFIGDIR is set before matplotlib loads
import random  # standard-library random; seeded below
import sys  # standard-library system access: sys.path
from pathlib import Path  # Path objects for building file paths

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
os.environ["MPLCONFIGDIR"] = str(PROJECT / "codes" / ".mplconfig")  # matplotlib's cache inside the project folder

import numpy as np  # NumPy: arrays and medians
import pandas as pd  # pandas: tables
from scipy.stats import rankdata  # midranks, for Cliff's delta with ties

sys.path.insert(0, str(PROJECT / "codes" / "common"))  # puts codes/ first on the import path so the helpers module is found
from helpers import exit_if_failed, iqr, save_fig  # exit on a failed check; IQR of the finite values (this file's copy moved there); byte-stable figure saving

SEED = 20260914  # integer constant: the project's fixed random seed
random.seed(SEED)  # seeds Python's generator
np.random.seed(SEED)  # seeds NumPy's generator

PROC = PROJECT / "Data" / "processed"  # where the CSVs are read and written
GRAPH = PROJECT / "Graph"  # where the figures go

spec = importlib.util.spec_from_file_location("gain_evidence_for_observable_effects", PROJECT / "codes" / "1_ground_truth" / "04_gain_and_position_evidence.py")  # describes where the evidence script lives
D = importlib.util.module_from_spec(spec)  # creates an empty module object for it
sys.modules["gain_evidence_for_observable_effects"] = D  # registers it so its own imports resolve
spec.loader.exec_module(D)  # runs the evidence script's top level (definitions only)
T = D.T  # clock text -> unix seconds on the recording date
SESSIONS, RXS = D.SESSIONS, D.RXS  # the six sessions in time order and the three receivers
RXNAME = {"SBF_02": "X5", "SBF_03": "mosaic-H", "UBX_01": "u-blox 01"}  # short names for tables and figures
STATUS_RX = {"Septentrio_02": "SBF_02", "Septentrio_03": "SBF_03", "u-blox_01": "UBX_01"}  # the receiver-status script's receiver tags -> streams (Septentrio_01 is the mosaic-G5, not used)

MIN_CLEAN_S = 100  # integer constant: fewer clean seconds than this and the baseline is pooled from the other sessions
MIN_BIN = 20  # integer constant: an elevation bin enters a standardised delta only with this many values on both sides
BLOCKS = (30, 60)  # tuple: bootstrap block lengths in seconds
N_BOOT = 200  # integer constant: bootstrap replicates per block length
CMS = 299792.458  # float constant: the range light travels in one millisecond, metres
JUMP_TOL_M = 5.0  # float constant: a clock step counts as a whole-millisecond jump within this many metres of a multiple
JUMP_WIN = 61  # integer constant: window (seconds) of the running median drift
EFFECT_THR = 0.33  # float constant: |delta| from which a difference is called at least medium (Romano's conventional bound)
MISS_GAP = 0.25  # float constant: a missing fraction this much larger in the attack group marks a missingness-driven difference
UNC_M, UNC_RUN = D.ANOM_M, D.ANOM_RUN  # the anomaly rule of 05d: a residual beyond 30 m for 3 consecutive seconds

SAT_Q = ["cn0", "dop_res_hz", "pr_res_m", "abs_dop_res_hz", "abs_pr_res_m", "dC_m", "d2C_m", "abs_d2C_m", "cmc_rate_m_s", "abs_cmc_rate_m_s", "lock_loss"]  # list: per-satellite quantities of the main set (GPS, Galileo, BeiDou MEO/IGSO above 10 degrees)
GLO_Q = ["cn0", "dC_m", "d2C_m", "abs_d2C_m", "cmc_rate_m_s", "abs_cmc_rate_m_s", "lock_loss"]  # list: per-satellite quantities of GLONASS (no elevation, no orbit)
# list literal over two lines: the per-second quantities compared in the effects table
EP_Q = ["n_tracked", "n_lost", "n_new", "d_n_tracked", "n_lock_loss", "pdop", "hdop", "clk_drift_m_s", "isb_m", "h_off_m", "cn0_median", "cn0_std",
        "cn0_maxabs", "dopres_median", "dopres_std", "dopres_maxabs", "prres_median", "prres_maxabs", "dC_median", "dC_maxabs", "d2C_median", "d2C_maxabs",
        "cmc_median", "cmc_maxabs", "r_cn0_median", "r_cn0_std", "r_dC_maxabs", "r_d2C_maxabs", "r_cmc_maxabs"]  # list: per-second quantities
MODEL_Q = {"dop_res_hz", "pr_res_m", "abs_dop_res_hz", "abs_pr_res_m", "pdop", "hdop", "clk_drift_m_s", "isb_m", "h_off_m", "dopres_median", "dopres_std", "dopres_maxabs", "prres_median", "prres_maxabs"}  # set: the quantities that need the orbit and the site, hence the host-time oracle in 2.6.12 u-blox 01
ORACLE = ("2.6.12", "UBX_01")  # tuple: the session and receiver whose time is the laptop clock
# list literal over two lines: the per-second quantities drawn in the figure
HEAT_EP = ["n_tracked", "n_lost", "d_n_tracked", "pdop", "hdop", "clk_drift_m_s", "isb_m", "h_off_m", "cn0_median", "cn0_std", "dopres_std", "dopres_maxabs", "prres_maxabs",
           "dC_maxabs", "d2C_maxabs", "cmc_maxabs", "r_cn0_median", "r_d2C_maxabs"]  # list: the per-second quantities drawn in the figure

CHECKS: list = []  # list: the self-checks
FAILED: list = []  # list: names of the self-checks that failed; the script ends the run with status 1 when it is not empty
G: dict = {}  # dict: the big tables, set before the pool starts so the forked workers share them


def check(item: str, quantity: str, value, ok: bool | None = None) -> None:  # function: records one self-check and prints it
    CHECKS.append({"item": item, "quantity": quantity, "value": value, "pass": "" if ok is None else bool(ok)})  # append a dict row to the list
    if ok is not None and not bool(ok):  # if: a pass/fail check (None marks a reported number) that did not pass
        FAILED.append(f"{item}: {quantity}")  # its name kept for the exit at the end of the run
    print(f"  [{'ok' if ok else ('FAIL' if ok is False else '--')}] {item}: {quantity} = {value}")  # progress line


def cliff(x: np.ndarray, y: np.ndarray) -> float:  # function: Cliff's delta, P(x > y) - P(x < y), from midranks
    n1, n2 = len(x), len(y)  # the two sample sizes
    if n1 == 0 or n2 == 0:  # if: one side is empty
        return np.nan  # no delta
    r = rankdata(np.concatenate([x, y]))  # midranks of the pooled sample
    u = r[:n1].sum() - n1 * (n1 + 1) / 2.0  # Mann-Whitney U of the first sample
    return float(2.0 * u / (n1 * n2) - 1.0)  # delta from U: 2U/(n1 n2) - 1


def binned(xa: np.ndarray, ba: np.ndarray, xb: np.ndarray, bb: np.ndarray) -> float:  # function: Cliff's delta standardised over 10 degree elevation bins, weighted by the attack group's count in each bin
    num = den = 0.0  # weighted sum and total weight
    for b in range(10, 90, 10):  # for loop over the bins 10-20 ... 80-90
        a, c = xa[ba == b], xb[bb == b]  # the values of each side in this bin
        if len(a) >= MIN_BIN and len(c) >= MIN_BIN:  # if: enough on both sides
            num += len(a) * cliff(a, c)  # that bin's delta times its weight
            den += len(a)  # total weight
    return num / den if den else np.nan  # the standardised delta, or NaN when no bin qualifies


def med(v: np.ndarray) -> float:  # function: median of the finite values
    v = v[np.isfinite(v)]  # finite values only
    return float(np.median(v)) if len(v) else np.nan  # NaN for an empty set


def clock_fix(g: pd.DataFrame) -> tuple:  # function: one session-receiver's clock drift with whole-millisecond jumps removed; returns the corrected drift, the jump multiples and the unmatched large steps
    c, s = g.clk_gps_m.to_numpy(float), g.sec.to_numpy()  # clock bias (m) and seconds, in time order
    d = np.full(len(g), np.nan)  # first differences, NaN where undefined
    pair = (np.diff(s) == 1) & ~np.isnan(c[1:]) & ~np.isnan(c[:-1])  # consecutive solved seconds
    d[1:][pair] = np.diff(c)[pair]  # fancy-indexed assignment: the step into each second that has a solved predecessor
    base = pd.Series(d).rolling(JUMP_WIN, center=True, min_periods=10).median().to_numpy()  # the running median drift (the receiver's steady frequency offset)
    r = d - base  # the step beyond the steady drift
    k = np.round(r / CMS)  # nearest whole number of milliseconds
    jump = (k != 0) & (np.abs(r - k * CMS) <= JUMP_TOL_M)  # a jump: at least one millisecond and within a few metres of a multiple
    miss = (np.abs(r) >= CMS / 2) & ~jump  # steps of at least half a millisecond that are not within the tolerance of a multiple
    return d - np.where(jump, k * CMS, 0.0), np.where(jump, k, 0.0), np.where(miss, r - k * CMS, 0.0)  # drift with the jumps taken out, the jump multiples, and the offsets of the unmatched large steps


def block_ids(secs: np.ndarray, length: int) -> np.ndarray:  # function: block number of each (sorted, unique) second: consecutive runs cut every `length` seconds
    brk = np.r_[True, np.diff(secs) != 1]  # True at the first second of every run
    run = np.cumsum(brk) - 1  # run number
    pos = secs - secs[brk][run]  # seconds since the start of the run
    return np.unique(run * 100000 + pos // length, return_inverse=True)[1]  # a consecutive block number from the (run, position) pair


def load_inputs() -> tuple:  # function: reads the observables script's tables, adds the effect interval, the clock fix and the derived columns
    sat = pd.read_csv(PROC / "features" / "observables_1hz.csv.gz", low_memory=False)  # per-satellite table
    ep = pd.read_csv(PROC / "features" / "observables_epochs_1hz.csv.gz", low_memory=False)  # per-epoch table
    ta3 = pd.read_csv(PROC / "ground_truth" / "ta3_schedule_confirmation.csv")  # the receiver-status script's gain onset and recovery
    ta3["receiver"] = ta3.receiver.map(STATUS_RX)  # stream names
    ta3 = ta3.dropna(subset=["receiver"])  # drops the mosaic-G5 rows
    eff = {}  # (session, receiver) -> (first second, last second) of the effect interval
    for r in ta3.itertuples():  # for loop over the rows
        if isinstance(r.onset_utc, str) and isinstance(r.recovery_utc, str):  # if: both instants exist
            eff[(r.test_code, r.receiver)] = (int(np.floor(T(r.onset_utc))), int(np.floor(T(r.recovery_utc))))  # whole seconds
    lo = np.array([eff.get((t, r), (1, 0))[0] for t, r in zip(ep.test_code, ep.receiver)])  # interval start per epoch row (1 > 0 means empty)
    hi = np.array([eff.get((t, r), (1, 0))[1] for t, r in zip(ep.test_code, ep.receiver)])  # interval end per epoch row
    ep["effect"] = (ep.sec >= lo) & (ep.sec <= hi) & (ep.group_provisional != "clean")  # inside the gain onset-recovery interval
    ep = ep.sort_values(["test_code", "receiver", "sec"]).reset_index(drop=True)  # time order within each session and receiver
    ep["group_first"] = ep.group_provisional  # the group as the observables script wrote it, kept for the reconciliation check
    ov_path = PROC / "ground_truth" / "group_overrides.csv"  # the diagnosed external interference (seconds taken out of the attack groups)
    if ov_path.exists():  # if: the file exists
        for r in pd.read_csv(ov_path).itertuples():  # for loop over its rows
            m_e = (ep.test_code == r.test_code) & (ep.receiver == r.receiver) & (ep.sec >= r.start_sec) & (ep.sec <= r.end_sec) & ep.group_provisional.isin(["spoofing", "jamming"])  # epoch rows to relabel
            ep.loc[m_e, "group_provisional"] = r.group  # the new provisional group
            m_s = (sat.test_code == r.test_code) & (sat.receiver == r.receiver) & (sat.sec >= r.start_sec) & (sat.sec <= r.end_sec) & sat.group_provisional.isin(["spoofing", "jamming"])  # satellite rows to relabel
            sat.loc[m_s, "group_provisional"] = r.group  # the same label
    ep["clk_drift_m_s"] = np.nan  # filled below
    ep["clk_jump_ms"] = 0.0  # filled below
    ep["clk_unmatched_m"] = 0.0  # filled below
    for (t, r), g in ep.groupby(["test_code", "receiver"]):  # for loop over the 18 session-receiver pairs
        dc, kj, um = clock_fix(g)  # drift without jumps, the jumps and the unmatched large steps
        ep.loc[g.index, "clk_drift_m_s"], ep.loc[g.index, "clk_jump_ms"], ep.loc[g.index, "clk_unmatched_m"] = dc, kj, um  # stored on the pair's rows
    ep["isb_m"] = ep.clk_gal_m - ep.clk_gps_m  # Galileo minus GPS clock: the inter-system bias, free of common clock steps
    ep["oracle"] = (ep.test_code == ORACLE[0]) & (ep.receiver == ORACLE[1])  # the rows computed on the laptop clock
    sat = sat.merge(ep[["test_code", "receiver", "sec", "effect"]], on=["test_code", "receiver", "sec"], how="left")  # the effect flag on every satellite row
    sat["effect"] = sat.effect.fillna(False).astype(bool)  # plain boolean
    return sat, ep, eff  # the tables and the intervals


def anomaly_seconds(sat: pd.DataFrame) -> pd.DataFrame:  # function: seconds that belong to a run of 3 consecutive seconds with one satellite's code residual beyond 30 m (the evidence script rule), per session and receiver
    s = sat[(sat.pr_res_m.abs() > UNC_M) & (sat.known_invalid == 0)][["test_code", "receiver", "sv", "sec"]].sort_values(["test_code", "receiver", "sv", "sec"])  # rows beyond the limit
    out = []  # seconds in qualifying runs
    for (t, r, v), g in s.groupby(["test_code", "receiver", "sv"]):  # for loop over every satellite track
        sec = g.sec.to_numpy()  # its flagged seconds
        brk = np.r_[True, np.diff(sec) != 1]  # run starts
        rid = np.cumsum(brk) - 1  # run number
        ln = np.bincount(rid)  # run lengths
        keep = ln[rid] >= UNC_RUN  # seconds inside runs long enough
        out += [(t, r, int(x)) for x in sec[keep]]  # list comprehension: the qualifying seconds
    return pd.DataFrame(out, columns=["test_code", "receiver", "sec"]).drop_duplicates()  # one row per anomalous second


def uncertain_table(ep: pd.DataFrame, eff: dict, an: pd.DataFrame) -> pd.DataFrame:  # function: Table of the uncertain seconds per session and receiver
    rows = []  # output rows
    for t in SESSIONS:  # for loop over the sessions
        for r in RXS:  # for loop over the receivers
            e = ep[(ep.test_code == t) & (ep.receiver == r)]  # the pair's seconds
            sp = e[e.group_provisional == "spoofing"]  # its scheduled spoofing seconds
            a = set(an[(an.test_code == t) & (an.receiver == r)].sec)  # anomalous seconds
            on = eff.get((t, r), (np.nan, np.nan))[0]  # gain onset (NaN if none)
            pre = sp[sp.sec < on] if not np.isnan(on) else sp  # spoofing-phase seconds before the onset (all of them where no onset exists)
            unc = [x for x in pre.sec if x in a]  # of those, the anomalous ones
            row = {"test_code": t, "receiver": r, "spoofing_seconds": len(sp), "gain_onset_utc": "" if np.isnan(on) else pd.to_datetime(on, unit="s", utc=True).strftime("%H:%M:%S")}  # dict literal: identifiers and the onset
            row["seconds_before_onset"] = len(pre)  # spoofing-phase seconds before the onset
            row["uncertain_seconds"] = len(unc)  # anomalous seconds among them
            row["first_anomaly_utc"] = pd.to_datetime(min(a & set(sp.sec)), unit="s", utc=True).strftime("%H:%M:%S") if a & set(sp.sec) else ""  # the first anomalous second of the spoofing phase
            row["anomalous_seconds_in_spoofing_phase"] = len(a & set(sp.sec))  # all anomalous seconds of the phase
            row["no_gain_onset"] = bool(np.isnan(on))  # no onset: the interval is undefined
            rows.append(row)  # list append: keeps the finished row
    return pd.DataFrame(rows)  # the table


def fresh(df: pd.DataFrame, secs: np.ndarray) -> np.ndarray:  # function: boolean mask of the rows whose second is in secs
    return df.sec.isin(secs).to_numpy()  # membership test


def work(item: tuple) -> list:  # function: all quantities of one session, receiver and comparison, with the point delta and the block-bootstrap intervals
    test, rx, comp, ci = item  # unpack the cell and its index (the seed depends on it)
    ep, sm, sg = G["ep"][rx], G["sm"][rx], G["sg"][rx]  # this receiver's epoch table, main-set and GLONASS satellite tables
    own = (ep.test_code == test) & (ep.group_provisional == "clean")  # the session's own clean seconds
    pooled = int(own.sum()) < MIN_CLEAN_S  # fewer than MIN_CLEAN_S: pool the other sessions' clean seconds
    base_mask = ((ep.test_code != test) & (ep.group_provisional == "clean")) if pooled else own  # the baseline seconds
    att_mask = (ep.test_code == test) & {"spoofing": ep.group_provisional == "spoofing", "jamming": ep.group_provisional == "jamming", "effect": ep.effect}[comp]  # the attack group's seconds
    eb, ea = ep[base_mask.to_numpy()], ep[att_mask.to_numpy()]  # the two groups' epoch rows
    lev = {"epoch": (ea, eb, EP_Q, False)}  # level -> (attack rows, baseline rows, quantities, elevation-binned)
    lev["sat_main"] = (sm[fresh(sm, ea.sec.to_numpy())], sm[fresh(sm, eb.sec.to_numpy())], SAT_Q, True)  # the main satellite set
    lev["sat_glonass"] = (sg[fresh(sg, ea.sec.to_numpy())], sg[fresh(sg, eb.sec.to_numpy())], GLO_Q, False)  # GLONASS
    out, rowmap = [], {}  # result rows; row index by (level, quantity)
    for name, (a, b, qs, bn) in lev.items():  # for loop over the three levels
        for q in qs:  # for loop over the quantities
            xa, xb = a[q].to_numpy(float), b[q].to_numpy(float)  # the values of each side
            fa, fb = np.isfinite(xa), np.isfinite(xb)  # which values exist
            # dict literal over three lines: identifiers, counts, missing fractions, median and IQR of one quantity in one cell
            row = {"test_code": test, "receiver": rx, "comparison": comp, "baseline": "pooled baseline" if pooled else "own clean", "level": name, "quantity": q,
                   # dict entry: whether host time replaces receiver time here, and the attack and baseline sample counts
                   "host_time_oracle": bool((test, rx) == ORACLE and q in MODEL_Q), "n_att": int(fa.sum()), "n_base": int(fb.sum()),
                   # dict entry: share of attack and of baseline seconds in which the quantity is missing
                   "miss_att": float(1 - fa.mean()) if len(xa) else np.nan, "miss_base": float(1 - fb.mean()) if len(xb) else np.nan,
                   "med_att": med(xa), "iqr_att": iqr(xa), "med_base": med(xb), "iqr_base": iqr(xb)}  # dict literal: identifiers, counts, missing fractions, median and IQR
            if bn:  # if: elevation-binned level
                row["delta"] = binned(xa[fa], a["bin"].to_numpy()[fa], xb[fb], b["bin"].to_numpy()[fb])  # standardised delta
                row["delta_unbinned"] = cliff(xa[fa], xb[fb])  # for comparison: the plain delta
            else:  # else: unbinned level
                row["delta"] = cliff(xa[fa], xb[fb])  # plain delta
                row["delta_unbinned"] = row["delta"]  # the same
            rowmap[(name, q)] = len(out)  # remember where the row sits
            out.append(row)  # list append: keeps the row
    if len(ea) == 0 or len(eb) == 0:  # if: one group has no seconds at all
        return out  # no bootstrap
    for L in BLOCKS:  # for loop over the block lengths
        rng = np.random.default_rng([SEED, ci, L])  # a generator that depends only on the cell and the block length
        info = {}  # level -> arrays for fast resampling
        for side, (secs_a) in (("a", np.sort(ea.sec.unique())), ("b", np.sort(eb.sec.unique()))):  # for loop over attack and baseline seconds
            bid = block_ids(secs_a, L)  # block number of each second
            for name, (a, b, qs, bn) in lev.items():  # for loop over the levels
                d = a if side == "a" else b  # this side's rows
                rb = bid[np.searchsorted(secs_a, d.sec.to_numpy())]  # block number of each row
                order = np.argsort(rb, kind="stable")  # rows sorted by block
                offs = np.r_[0, np.cumsum(np.bincount(rb, minlength=bid.max() + 1))]  # block boundaries in that order
                info[(name, side)] = (order, offs, bid.max() + 1)  # kept
        draws = {k: [] for k in rowmap}  # (level, quantity) -> bootstrap deltas
        for _ in range(N_BOOT):  # for loop over replicates
            pick = {s: rng.integers(0, info[("epoch", s)][2], info[("epoch", s)][2]) for s in "ab"}  # blocks drawn with replacement, same draw for all levels
            for name, (a, b, qs, bn) in lev.items():  # for loop over the levels
                ia = np.concatenate([info[(name, "a")][0][info[(name, "a")][1][k]:info[(name, "a")][1][k + 1]] for k in pick["a"]])  # attack rows of the drawn blocks
                ib = np.concatenate([info[(name, "b")][0][info[(name, "b")][1][k]:info[(name, "b")][1][k + 1]] for k in pick["b"]])  # baseline rows of the drawn blocks
                for q in qs:  # for loop over the quantities
                    xa, xb = a[q].to_numpy(float)[ia], b[q].to_numpy(float)[ib]  # resampled values
                    fa, fb = np.isfinite(xa), np.isfinite(xb)  # which exist
                    if bn:  # if: elevation-binned
                        draws[(name, q)].append(binned(xa[fa], a["bin"].to_numpy()[ia][fa], xb[fb], b["bin"].to_numpy()[ib][fb]))  # standardised delta
                    else:  # else: plain
                        draws[(name, q)].append(cliff(xa[fa], xb[fb]))  # plain delta
        for k, i in rowmap.items():  # for loop over the rows
            v = np.array(draws[k], float)  # the replicates
            ok = np.isfinite(v)  # which are defined
            lo, hi = (np.percentile(v[ok], 2.5), np.percentile(v[ok], 97.5)) if ok.mean() >= 0.8 else (np.nan, np.nan)  # 95 % percentile interval, needs 80 % defined
            out[i][f"ci{L}_lo"], out[i][f"ci{L}_hi"] = float(lo), float(hi)  # stored
    for r in out:  # for loop over the rows
        s30 = 1 if r.get("ci30_lo", np.nan) > 0 else (-1 if r.get("ci30_hi", np.nan) < 0 else 0)  # interval status at 30 s: above zero, below zero, or containing zero
        s60 = 1 if r.get("ci60_lo", np.nan) > 0 else (-1 if r.get("ci60_hi", np.nan) < 0 else 0)  # the same at 60 s
        r["ci_status_30"], r["ci_status_60"], r["ci_unstable"] = s30, s60, bool(s30 != s60)  # status and the flag
    return out  # the rows


def group_stats(ep: pd.DataFrame, sm: dict, sg: dict) -> pd.DataFrame:  # function: median, IQR and missing fraction per session, receiver, quantity and group
    rows = []  # output rows
    for t in SESSIONS:  # for loop over the sessions
        for r in RXS:  # for loop over the receivers
            e = ep[(ep.test_code == t) & (ep.receiver == r)]  # the pair's epochs
            own = (e.group_provisional == "clean").sum() >= MIN_CLEAN_S  # whether the session has its own baseline
            groups = {"clean": e[e.group_provisional == "clean"], "jamming": e[e.group_provisional == "jamming"], "spoofing": e[e.group_provisional == "spoofing"], "effect": e[e.effect]}  # the groups
            for extra in sorted(set(e.group_provisional) - {"clean", "jamming", "spoofing", "other"}):  # for loop over any group a relabelling added (for example scheduled-no-effect)
                groups[extra] = e[e.group_provisional == extra]  # reported as its own group
            if not own:  # if: no own baseline
                groups["pooled_clean"] = ep[(ep.receiver == r) & (ep.test_code != t) & (ep.group_provisional == "clean")]  # the pooled baseline
            for gname, ge in groups.items():  # for loop over the groups
                secs = ge.sec.to_numpy()  # their seconds
                for lev, tab, qs in (("epoch", ge, EP_Q), ("sat_main", sm[r][fresh(sm[r], secs)], SAT_Q), ("sat_glonass", sg[r][fresh(sg[r], secs)], GLO_Q)):  # for loop over the levels
                    for q in qs:  # for loop over the quantities
                        x = tab[q].to_numpy(float)  # the values
                        # append a dict over two lines: one group-statistics row (median, IQR, missing fraction)
                        rows.append({"test_code": t, "receiver": r, "group": gname, "level": lev, "quantity": q, "host_time_oracle": bool((t, r) == ORACLE and q in MODEL_Q),
                                     # dict entry: seconds in the group, finite values, and the share of seconds with no value
                                     "seconds": len(ge), "n_values": int(np.isfinite(x).sum()), "missing_fraction": float(1 - np.isfinite(x).mean()) if len(x) else np.nan,
                                     "median": med(x), "iqr": iqr(x)})  # one row
    return pd.DataFrame(rows)  # the table


def cn0_behaviour(ep: pd.DataFrame, sm: dict, sg: dict, eff_rows: pd.DataFrame) -> pd.DataFrame:  # function: the C/N0 question per session, receiver and attack group
    rows = []  # output rows
    for t in SESSIONS:  # for loop over the sessions
        for r in RXS:  # for loop over the receivers
            e = ep[(ep.receiver == r)]  # all of the receiver's epochs
            own = e[(e.test_code == t) & (e.group_provisional == "clean")]  # the session's own clean seconds
            pooled = len(own) < MIN_CLEAN_S  # whether to pool
            base = e[(e.test_code != t) & (e.group_provisional == "clean")] if pooled else own  # baseline seconds
            for comp in ("spoofing", "effect"):  # for loop over the two attack groups
                att = e[(e.test_code == t) & ((e.group_provisional == "spoofing") if comp == "spoofing" else e.effect)]  # attack seconds
                if len(att) == 0:  # if: none
                    continue  # skip
                a, b = sm[r][fresh(sm[r], att.sec.to_numpy())], sm[r][fresh(sm[r], base.sec.to_numpy())]  # main-set satellite rows
                diffs, ws = [], []  # per-bin median differences and weights
                for bn in range(10, 90, 10):  # for loop over the elevation bins
                    xa, xb = a.cn0[a.bin == bn].dropna().to_numpy(), b.cn0[b.bin == bn].dropna().to_numpy()  # C/N0 in the bin
                    if len(xa) >= MIN_BIN and len(xb) >= MIN_BIN:  # if: enough on both sides
                        diffs.append(np.median(xa) - np.median(xb))  # difference of medians in the bin
                        ws.append(len(xa))  # weight
                matched = float(np.average(diffs, weights=ws)) if ws else np.nan  # elevation-matched median difference (dB)
                ga, gb = sg[r][fresh(sg[r], att.sec.to_numpy())].cn0.dropna().to_numpy(), sg[r][fresh(sg[r], base.sec.to_numpy())].cn0.dropna().to_numpy()  # GLONASS C/N0
                share50 = lambda x: float((x > 50).mean()) if len(x) else np.nan  # share of values above 50 dB-Hz
                er = eff_rows[(eff_rows.test_code == t) & (eff_rows.receiver == r) & (eff_rows.comparison == comp) & (eff_rows.level == "sat_main") & (eff_rows.quantity == "cn0")]  # the effects row of C/N0
                row = {"test_code": t, "receiver": r, "comparison": comp, "baseline": "pooled baseline" if pooled else "own clean"}  # dict literal: identifiers
                row["seconds"], row["seconds_tracked_share"], row["base_seconds_tracked_share"] = len(att), float((att.n_tracked > 0).mean()), float((base.n_tracked > 0).mean())  # coverage of the groups
                row["median_n_tracked"], row["base_median_n_tracked"] = float(att.n_tracked.median()), float(base.n_tracked.median())  # satellites tracked
                row["cn0_median"], row["base_cn0_median"] = med(a.cn0.to_numpy(float)), med(b.cn0.to_numpy(float))  # C/N0 levels
                row["cn0_matched_diff_db"] = matched  # elevation-matched change
                row["cn0_delta"] = float(er.delta.iloc[0]) if len(er) else np.nan  # Cliff's delta of the standardised comparison
                row["share_above_50"], row["base_share_above_50"] = share50(a.cn0.dropna().to_numpy()), share50(b.cn0.dropna().to_numpy())  # share of values above 50 dB-Hz
                row["glonass_cn0_diff_db"] = med(ga) - med(gb) if len(ga) and len(gb) else np.nan  # GLONASS median difference (unmatched)
                vanish = (row["seconds_tracked_share"] < 0.5) or (row["median_n_tracked"] <= 0.5 * row["base_median_n_tracked"])  # satellites lost for most of the group
                if vanish:  # if: lost
                    label = "vanish (satellites lost)"  # label
                elif np.isnan(matched):  # elif: no matched comparison possible
                    label = "not assessable"  # label
                else:  # else: compare the level
                    label = "rise" if matched >= 1.0 else ("fall" if matched <= -1.0 else "stay")  # one decibel threshold
                row["behaviour"] = label  # kept
                row["rise_above_50_dbhz"] = bool((row["share_above_50"] - row["base_share_above_50"]) >= 0.10) if not np.isnan(row["share_above_50"]) else False  # ten points more of the values above 50 dB-Hz
                rows.append(row)  # list append: keeps the finished row
    return pd.DataFrame(rows)  # the table


def glonass_report(eff: pd.DataFrame, cn: pd.DataFrame) -> pd.DataFrame:  # function: GLONASS observables per spoofing phase and effect interval, beside the main-set C/N0 change
    g = eff[(eff.level == "sat_glonass") & eff.comparison.isin(["spoofing", "effect"])].copy()  # the GLONASS rows
    ep_n = eff[(eff.level == "epoch") & (eff.quantity == "n_tracked") & eff.comparison.isin(["spoofing", "effect"])]  # the satellites-tracked rows (all constellations)
    g["changed"] = (g.delta.abs() >= EFFECT_THR) & (((g.ci30_lo > 0) & (g.ci60_lo > 0)) | ((g.ci30_hi < 0) & (g.ci60_hi < 0)))  # medium effect and an interval clear of zero at both block lengths
    out = g[["test_code", "receiver", "comparison", "baseline", "quantity", "n_att", "n_base", "miss_att", "med_att", "med_base", "delta", "ci30_lo", "ci30_hi", "ci60_lo", "ci60_hi", "changed"]].copy()  # the columns
    m = cn.set_index(["test_code", "receiver", "comparison"])  # the C/N0 behaviour table
    out["main_set_cn0_behaviour"] = [m.behaviour.get((t, r, c), "") for t, r, c in zip(out.test_code, out.receiver, out.comparison)]  # the main set's C/N0 behaviour beside it
    out["n_tracked_delta"] = [float(ep_n[(ep_n.test_code == t) & (ep_n.receiver == r) & (ep_n.comparison == c)].delta.iloc[0]) if len(ep_n[(ep_n.test_code == t) & (ep_n.receiver == r) & (ep_n.comparison == c)]) else np.nan for t, r, c in zip(out.test_code, out.receiver, out.comparison)]  # the change in the number of satellites tracked
    return out  # the table


def leakage(eff: pd.DataFrame) -> pd.DataFrame:  # function: per observable, the cells where the difference rests on missingness or on the jamming phase
    jm = eff[eff.comparison == "jamming"].set_index(["test_code", "receiver", "level", "quantity"])  # jamming-phase rows
    rows = []  # output rows
    for (lev, q), grp in eff[eff.comparison == "spoofing"].groupby(["level", "quantity"]):  # for loop over observables
        n_eff = n_miss = n_jam = n_assess = n_unst = n_pos = 0  # counters
        for r in grp.itertuples():  # for loop over the cells
            if not np.isfinite(r.delta) or abs(r.delta) < EFFECT_THR:  # if: no medium effect
                continue  # not counted
            n_eff += 1  # a medium or larger effect
            n_pos += int(r.delta > 0)  # positive direction
            n_unst += int(bool(r.ci_unstable))  # interval status differs between 30 s and 60 s
            n_miss += int((r.miss_att - r.miss_base) > MISS_GAP)  # missingness-driven: the quantity is absent in far more of the attack seconds
            k = (r.test_code, r.receiver, lev, q)  # the cell's key
            if k in jm.index:  # if: the session has a jamming phase for this receiver
                j = jm.loc[k]  # its jamming row
                if np.isfinite(j.delta):  # if: defined
                    n_assess += 1  # the jamming comparison could be made
                    n_jam += int(abs(j.delta) >= EFFECT_THR and abs(j.delta) >= 0.8 * abs(r.delta))  # jamming-driven: jamming alone produces nearly as large an effect
        # append a dict over two lines: one leakage row per observable
        rows.append({"level": lev, "quantity": q, "cells": len(grp), "cells_with_medium_effect": n_eff, "positive_direction": n_pos, "missingness_driven": n_miss,
                     "jamming_assessable": n_assess, "jamming_driven": n_jam, "ci_unstable": n_unst, "median_delta": float(np.nanmedian(grp.delta)), "host_time_oracle_cells": int(grp.host_time_oracle.sum())})  # one row
    out = pd.DataFrame(rows)  # the table
    out["leakage_candidate"] = (out.missingness_driven >= 0.5 * out.cells_with_medium_effect) & (out.cells_with_medium_effect > 0) | (out.jamming_driven >= 0.5 * out.jamming_assessable.clip(lower=1)) & (out.jamming_driven > 0)  # at least half of the effects are missingness- or jamming-driven
    return out  # the table


def heatmap(eff: pd.DataFrame) -> None:  # function: the figure, the Cliff's delta of spoofing against clean, observables by session-receiver
    import matplotlib  # the plotting library
    matplotlib.use("Agg")  # file-only backend
    import matplotlib.pyplot as plt  # pyplot
    rows = [("sat_main", q) for q in SAT_Q] + [("sat_glonass", q) for q in GLO_Q] + [("epoch", q) for q in HEAT_EP]  # list: the drawn observables in order
    cols = [(t, r) for t in SESSIONS for r in RXS]  # list: the 18 session-receiver columns
    fig, axs = plt.subplots(1, 2, figsize=(21, 11), sharey=True)  # two panels side by side
    for ax, comp, ttl in zip(axs, ("spoofing", "effect"), ("scheduled spoofing phase against clean", "effect interval (gain onset to recovery) against clean")):  # for loop over the panels
        z = np.full((len(rows), len(cols)), np.nan)  # the delta matrix
        pool = [False] * len(cols)  # which columns use the pooled baseline
        ora = np.zeros((len(rows), len(cols)), bool)  # which cells are host-time oracle
        for j, (t, r) in enumerate(cols):  # for loop over the columns
            for i, (lev, q) in enumerate(rows):  # for loop over the rows
                m = eff[(eff.test_code == t) & (eff.receiver == r) & (eff.comparison == comp) & (eff.level == lev) & (eff.quantity == q)]  # the cell's row
                if len(m):  # if: it exists
                    z[i, j] = m.delta.iloc[0]  # the delta
                    pool[j] = m.baseline.iloc[0] == "pooled baseline"  # baseline type
                    ora[i, j] = bool(m.host_time_oracle.iloc[0])  # oracle flag
        im = ax.imshow(np.ma.masked_invalid(z), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")  # the heat map; masked cells stay blank
        ax.set_facecolor("#dddddd")  # grey for undefined cells
        for i in range(len(rows)):  # for loop over rows
            for j in range(len(cols)):  # for loop over columns
                if np.isfinite(z[i, j]):  # if: a value
                    ax.text(j, i, f"{z[i, j]:.1f}" + ("o" if ora[i, j] else ""), ha="center", va="center", fontsize=5.5, color="white" if abs(z[i, j]) > 0.6 else "black")  # the value; o marks the host-time oracle
        ax.set_xticks(range(len(cols)))  # one tick per column
        ax.set_xticklabels([f"{t} {RXNAME[r]}" + (" (pooled)" if pool[j] else "") for j, (t, r) in enumerate(cols)], rotation=90, fontsize=7)  # column labels with the baseline marker
        ax.set_yticks(range(len(rows)))  # one tick per row
        ax.set_yticklabels([("GLONASS " if lev == "sat_glonass" else ("per-second " if lev == "epoch" else "per-satellite ")) + q for lev, q in rows], fontsize=7)  # row labels
        ax.axhline(len(SAT_Q) - 0.5, color="k", lw=0.8)  # separator after the main-set satellite rows
        ax.axhline(len(SAT_Q) + len(GLO_Q) - 0.5, color="k", lw=0.8)  # separator after the GLONASS rows
        ax.set_title(f"Cliff's delta, {ttl}", fontsize=9)  # panel title
    fig.colorbar(im, ax=axs, shrink=0.6, label="Cliff's delta (attack minus clean; positive: attack values larger)")  # colour bar
    fig.suptitle("PROVISIONAL groups. 'pooled': baseline from the other sessions' clean seconds, per-satellite rows standardised over 10 degree elevation bins. 'o': computed on the laptop clock (2.6.12 u-blox 01). Grey: undefined.", fontsize=8)  # figure title
    save_fig(fig, GRAPH / "observable_effects_heatmap.png")  # saved


def cn0_figure(ep: pd.DataFrame, sat: pd.DataFrame, eff: dict) -> None:  # function: the figure, C/N0 of the tracked satellites through each window, phases shaded
    import matplotlib  # the plotting library
    matplotlib.use("Agg")  # file-only backend
    import matplotlib.pyplot as plt  # pyplot
    fig, axs = plt.subplots(6, 3, figsize=(17, 19), squeeze=False)  # one panel per session and receiver
    colors = {"clean": "#2ca02c", "jamming": "#ff9f1c", "spoofing": "#d62728", "scheduled-no-effect": "#999999"}  # phase colours
    for i, t in enumerate(SESSIONS):  # for loop over sessions (rows)
        for j, r in enumerate(RXS):  # for loop over receivers (columns)
            ax = axs[i, j]  # this panel
            s = sat[(sat.test_code == t) & (sat.receiver == r) & sat.cn0.notna()]  # C/N0 of all tracked satellites
            q = s.groupby("sec").cn0.quantile([0.25, 0.5, 0.75]).unstack()  # per-second quartiles
            e = ep[(ep.test_code == t) & (ep.receiver == r)]  # the pair's epochs
            t0 = e.sec.min()  # start of the span
            q = q.reindex(np.arange(e.sec.min(), e.sec.max() + 1))  # every second of the span, NaN where nothing is tracked, so that lines break at gaps instead of crossing them
            x = (q.index.to_numpy() - t0) / 60.0  # minutes from the start
            ax.fill_between(x, q[0.25], q[0.75], color="#9ecae1", lw=0)  # interquartile band
            ax.plot(x, q[0.5], color="#08519c", lw=0.7)  # median
            for gname, col in colors.items():  # for loop over the phases
                m = (e.group_provisional == gname).to_numpy()  # their seconds
                if m.any():  # if: the phase exists
                    sec = e.sec.to_numpy()[m]  # seconds
                    brk = np.r_[True, np.diff(sec) != 1]  # run starts
                    st, en = sec[brk], sec[np.r_[brk[1:], True]]  # run starts and run ends (the last second before the next break)
                    for a, b in zip(st, en):  # for loop over the runs
                        ax.axvspan((a - t0) / 60.0, (b + 1 - t0) / 60.0, ymin=0.0, ymax=0.06, color=col, lw=0)  # a coloured strip at the bottom
            if (t, r) in eff:  # if: a gain interval exists
                ax.axvline((eff[(t, r)][0] - t0) / 60.0, color="k", ls="--", lw=0.8)  # gain onset
                ax.axvline((eff[(t, r)][1] - t0) / 60.0, color="k", ls=":", lw=0.8)  # gain recovery
            ax2 = ax.twinx()  # second axis for the satellite count
            ax2.plot((e.sec.to_numpy() - t0) / 60.0, e.n_tracked.to_numpy(), color="#888888", lw=0.6)  # satellites tracked
            ax2.set_ylim(0, 60)  # fixed range
            ax2.tick_params(labelsize=6, colors="#666666")  # small grey ticks
            ax.set_ylim(0, 60)  # C/N0 range
            ax.tick_params(labelsize=6)  # small ticks
            ax.set_title(f"{t} {RXNAME[r]}", fontsize=8)  # panel title
            if j == 0:  # if: first column
                ax.set_ylabel("C/N0 (dB-Hz)", fontsize=7)  # axis label
            if i == 5:  # if: last row
                ax.set_xlabel("minutes from the start of the span", fontsize=7)  # axis label
    fig.suptitle("C/N0 of every tracked L1/G1/E1/B1 satellite, median and interquartile band (blue), satellites tracked (grey, right axis). Strip: green clean, orange jamming, red spoofing, grey scheduled with no effect visible (PROVISIONAL). Dashed/dotted: gain onset/recovery of the receiver-status script.", fontsize=8)  # figure title
    fig.tight_layout(rect=(0, 0, 1, 0.98))  # leaves room for the title
    save_fig(fig, GRAPH / "cn0_through_windows.png")  # saved


def main() -> None:  # function: runs the whole script
    sat, ep, eff = load_inputs()  # the observables script's tables with the effect interval and the clock fix
    sat["bin"] = np.minimum(np.floor(sat.el_deg / 10.0) * 10, 80)  # elevation bin start (10 ... 80) for the model-based set
    for c in ("pr_res_m", "dop_res_hz", "d2C_m", "cmc_rate_m_s"):  # for loop over the signed quantities whose spread matters more than their sign
        sat["abs_" + c] = sat[c].abs()  # absolute value: a growing spread shows in it, where the signed values stay centred on zero
    main_set = sat.sys.isin(["G", "E", "C"]) & (sat.el_deg >= 10) & (sat.geo == 0) & (sat.known_invalid == 0)  # GPS, Galileo, BeiDou MEO/IGSO above 10 degrees, valid measurements
    glo_set = (sat.sys == "R") & (sat.known_invalid == 0)  # GLONASS
    G["ep"] = {r: ep[ep.receiver == r].reset_index(drop=True) for r in RXS}  # epoch tables per receiver
    G["sm"] = {r: sat[main_set & (sat.receiver == r)].reset_index(drop=True) for r in RXS}  # main-set satellite tables per receiver
    G["sg"] = {r: sat[glo_set & (sat.receiver == r)].reset_index(drop=True) for r in RXS}  # GLONASS tables per receiver
    # ---- clock jumps ----
    jr = []  # clock-jump rows
    for (t, r), g in ep.groupby(["test_code", "receiver"]):  # for loop over the pairs
        k = g.clk_jump_ms[g.clk_jump_ms != 0]  # the jumps found
        raw = g.clk_gps_m.diff().where(g.sec.diff() == 1)  # the raw one-second steps
        # append a dict over three lines: one clock-jump row per session and receiver
        jr.append({"test_code": t, "receiver": r, "solved_seconds": int(g.clk_gps_m.notna().sum()), "jumps_removed": len(k), "total_ms": float(k.abs().sum()),
                   # dict entry: removed jump sizes, steps left unmatched and their sizes, and the median clock drift
                   "jump_multiples": ";".join(str(int(x)) for x in k.to_numpy()), "unmatched_large_steps": int((g.clk_unmatched_m != 0).sum()), "unmatched_offsets_m": ";".join(f"{x:.1f}" for x in g.clk_unmatched_m[g.clk_unmatched_m != 0]), "median_drift_m_s": float(g.clk_drift_m_s.median()),
                   # dict entry: spread of the clock drift and the 99th percentile of the raw clock step
                   "iqr_drift_m_s": float(g.clk_drift_m_s.quantile(.75) - g.clk_drift_m_s.quantile(.25)), "raw_p99_abs_step_m": float(raw.abs().quantile(.99)),
                   "corrected_p99_abs_step_m": float(g.clk_drift_m_s.abs().quantile(.99))})  # one row
    jdf = pd.DataFrame(jr)  # the table
    jdf.round(4).to_csv(PROC / "features" / "observable_clock_jumps.csv", index=False)  # written
    from helpers import write_facts  # the facts-table writer, imported here so no earlier line moves
    fx = []  # list of 4-tuples: each unmatched large clock step, which the jump table holds only as its distance from a multiple
    for (t, r), g in ep.groupby(["test_code", "receiver"]):  # for loop over the pairs, as above
        raw = g.clk_gps_m.diff().where(g.sec.diff() == 1)  # the raw one-second steps of the GPS clock, m (NaN without a solved predecessor)
        for n, (u, x) in enumerate(zip(g.utc[g.clk_unmatched_m != 0], raw[g.clk_unmatched_m != 0]), start=1):  # enumerate from 1 over the unmatched steps: their clock text and raw step
            fx += [(f"unmatched_step_{t}_{r}_{n}_utc", u, "UTC", "second whose one-second GPS clock step is at least half a millisecond and not within JUMP_TOL_M of a whole millisecond"),  # tuple: its second
                   (f"unmatched_step_{t}_{r}_{n}_ms", float(x) / CMS, "ms", "that raw one-second step of the GPS clock bias divided by CMS (signed)")]  # list +=: the step's second and size
    write_facts(__file__, fx)  # written as 06_observable_effects_facts.csv
    print(jdf[["test_code", "receiver", "solved_seconds", "jumps_removed", "total_ms", "unmatched_large_steps", "unmatched_offsets_m", "median_drift_m_s", "iqr_drift_m_s"]].to_string())  # for the report
    # ---- uncertain seconds ----
    an = anomaly_seconds(sat)  # seconds in anomaly runs
    unc = uncertain_table(ep, eff, an)  # the table
    unc.to_csv(PROC / "features" / "observable_uncertain.csv", index=False)  # written
    print(unc.to_string())  # for the report
    # ---- the effects ----
    items = [(t, r, c, i) for i, (t, r, c) in enumerate((t, r, c) for t in SESSIONS for r in RXS for c in ("spoofing", "effect", "jamming"))]  # list: every cell with its index
    ctx = mp.get_context("fork")  # fork: the workers inherit the tables without copying them
    with ctx.Pool(8) as pool:  # a pool of eight workers
        res = pool.map(work, items, chunksize=1)  # the cells in order (the result order is fixed)
    eff_rows = pd.DataFrame([r for cell in res for r in cell])  # all rows in one table
    eff_rows["ci_unstable"] = eff_rows.ci_unstable.fillna(False).astype(bool)  # cells without a bootstrap are not flagged
    ev = eff_rows.copy()  # a copy for rounding
    ev.round(4).to_csv(PROC / "features" / "observable_effects.csv", index=False)  # written
    # ---- tables built from the effects ----
    gs = group_stats(ep, G["sm"], G["sg"])  # medians and IQRs per group
    gs.round(4).to_csv(PROC / "features" / "observable_group_stats.csv", index=False)  # written
    lk = leakage(eff_rows)  # the leakage flags
    lk.round(4).to_csv(PROC / "features" / "observable_leakage_flags.csv", index=False)  # written
    cn = cn0_behaviour(ep, G["sm"], G["sg"], eff_rows)  # the C/N0 question
    cn.round(4).to_csv(PROC / "features" / "observable_cn0_behaviour.csv", index=False)  # written
    gl = glonass_report(eff_rows, cn)  # the GLONASS report
    gl.round(4).to_csv(PROC / "features" / "observable_glonass_report.csv", index=False)  # written
    bef = PROJECT / "Data" / "processed" / "features" / "observable_effects_before.csv"  # the effects as they stood before the 2.6.5 relabelling of the gain-pattern script
    if bef.exists():  # if: the snapshot exists
        o = pd.read_csv(bef)  # the earlier result
        now = eff_rows.assign(delta=eff_rows.delta.round(4), n_att=eff_rows.n_att)  # this run's rows with delta rounded as the earlier file was written
        k = ["test_code", "receiver", "comparison", "level", "quantity"]  # the key of a row
        m = o[o.test_code == "2.6.5"].merge(now[now.test_code == "2.6.5"], on=k, suffixes=("_before", "_after"))  # 2.6.5 rows before and after
        m["difference"] = m.delta_after - m.delta_before  # the change in Cliff's delta
        m[k + ["n_att_before", "n_att_after", "delta_before", "delta_after", "difference"]].round(4).to_csv(PROC / "ground_truth" / "before_after_265.csv", index=False)  # written
        o2 = o[o.test_code != "2.6.5"].merge(now[now.test_code != "2.6.5"], on=k, suffixes=("_before", "_after"))  # the other sessions
        check("relabelling", "rows of other sessions whose delta changed (must be 0), 2.6.5 rows compared, largest |change| in 2.6.5", f"{int(((o2.delta_before - o2.delta_after).abs() > 1e-9).sum())}, {len(m)}, {m.difference.abs().max():.3f}", int(((o2.delta_before - o2.delta_after).abs() > 1e-9).sum()) == 0)  # nothing outside 2.6.5 may move
    heatmap(eff_rows)  # the figure
    cn0_figure(ep, sat, eff)  # the figure
    selfchecks(ep, sat, eff_rows, gs, jdf)  # the checks
    pd.DataFrame(CHECKS).to_csv(PROC / "features" / "observable_effects_checks.csv", index=False)  # written
    print(f"effects rows: {len(eff_rows)}")  # final line


def selfchecks(ep: pd.DataFrame, sat: pd.DataFrame, eff: pd.DataFrame, gs: pd.DataFrame, jdf: pd.DataFrame) -> None:  # function: the checks that need the finished tables
    print("self-checks")  # heading
    rng = np.random.default_rng(SEED)  # a generator for the random test samples
    worst = 0.0  # largest disagreement between the rank formula and a brute-force pair count
    for _ in range(200):  # for loop over random small samples with ties
        x, y = rng.integers(0, 6, rng.integers(3, 25)).astype(float), rng.integers(0, 6, rng.integers(3, 25)).astype(float)  # integer data, so ties are common
        brute = (np.sum(x[:, None] > y[None, :]) - np.sum(x[:, None] < y[None, :])) / (len(x) * len(y))  # P(x>y) - P(x<y) counted over all pairs
        worst = max(worst, abs(brute - cliff(x, y)))  # the disagreement
    check("Cliff's delta", "rank formula against brute-force pair counting on 200 random samples with ties: largest difference", f"{worst:.2e}", worst < 1e-12)  # must agree
    e = ep[(ep.test_code == "2.6.4") & (ep.receiver == "SBF_02")]  # one cell for the median / IQR spot check
    r = gs[(gs.test_code == "2.6.4") & (gs.receiver == "SBF_02") & (gs.group == "spoofing") & (gs.level == "epoch") & (gs.quantity == "cn0_median")].iloc[0]  # the group-stats row
    x = e[e.group_provisional == "spoofing"].cn0_median.dropna()  # the direct computation from the observables table
    check("group statistics", "2.6.4 X5 spoofing cn0_median: table median / IQR against a direct computation", f"{r['median']:.3f}/{r.iqr:.3f} vs {x.median():.3f}/{x.quantile(.75) - x.quantile(.25):.3f}", abs(r["median"] - x.median()) < 1e-3 and abs(r.iqr - (x.quantile(.75) - x.quantile(.25))) < 1e-3)  # must agree
    miss = pd.read_csv(PROC / "features" / "observables_missingness.csv")  # the observables script's missingness table, an independent count
    tot = miss.groupby(["test_code", "receiver", "group"]).seconds.sum()  # its seconds per group
    mine = ep.groupby(["test_code", "receiver", "group_first"]).size()  # this script's seconds per group, as the observables script labelled them (before the external-interference relabelling)
    bad = int((tot.values != mine.reindex(tot.index.set_names(mine.index.names)).values).sum()) if len(tot) == len(mine) else -1  # groups whose seconds differ
    check("denominators", "seconds per session, receiver and group agree with the observables script's missingness table: groups differing", bad, bad == 0)  # must agree
    s = eff[(eff.level == "sat_main") & (eff.quantity == "cn0") & (eff.comparison == "spoofing")]  # the C/N0 rows
    check("elevation standardisation", "per-satellite C/N0, spoofing: largest |standardised delta - plain delta| over cells (the elevation confounding the binning removes)", f"{(s.delta - s.delta_unbinned).abs().max():.3f}", None)  # for reading
    ok = eff[np.isfinite(eff.ci30_lo)]  # rows with an interval
    inside = ((ok.delta >= ok.ci30_lo - 0.05) & (ok.delta <= ok.ci30_hi + 0.05)).mean()  # share of point estimates inside their 30 s interval (with a small tolerance)
    check("bootstrap", "share of point deltas inside their 30 s interval (+/-0.05), and rows with intervals", f"{inside:.3f}, {len(ok)}", inside > 0.9)  # the point estimate should lie in its own interval
    check("clock", "whole-millisecond jumps removed per receiver (all sessions)", "; ".join(f"{r}: {int(g.jumps_removed.sum())}" for r, g in jdf.groupby("receiver")), None)  # for reading
    check("clock", "corrected 99th percentile of |one-second step| never above the raw one", str(bool((jdf.corrected_p99_abs_step_m <= jdf.raw_p99_abs_step_m + 1e-6).all())), bool((jdf.corrected_p99_abs_step_m <= jdf.raw_p99_abs_step_m + 1e-6).all()))  # removing jumps cannot make the steps larger
    ov = int(((ep.group_provisional == "clean") & ep.effect).sum())  # clean seconds inside an effect interval
    check("groups", "clean seconds inside an effect interval", ov, ov == 0)  # none by construction


if __name__ == "__main__":  # guard: runs main() only when the file is executed, not when imported
    main()  # entry point
    exit_if_failed(FAILED, "4_features/06_observable_effects.py")  # the helpers module: status 1 when any self-check failed, so a chained run stops
