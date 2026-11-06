#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
Part 2 of the feature build: the base quantities at 1 Hz, one row per second of every receiver and session on the label grid;
part 3: the windows of 1, 2, 3, 5 and 10 s built from them (part 1 is the residual engine, 02_residuals.py).

Three blocks, the column lists identical for the three receivers (RQ3):
  orbit-free   from the RINEX observations alone (the L1-band codes), no elevation mask: per constellation (G, R, E, C) the median,
               largest absolute value and standard deviation over satellites of C/N0, raw Doppler, Doppler rate, the single and second time
               difference of the pseudorange, the drift-corrected time difference (code rate plus Doppler range rate) and the
               code-minus-carrier rate; tracked count, its change, lock-loss count; over all satellites tracked, new, lost, the C/N0 median
               and spread, and the common whole-millisecond step of the time differences. This block alone is the no-residual set.
  self_        from the residual script's self-solved variant (main set): per constellation the code residual (G, E, C), the de-biased residual (all
               four) and the Doppler residual (all four) with the same three statistics; counts of residuals, screened jumps and samples
               outside the baseline band or gate; PDOP; GLONASS, Galileo and BeiDou clock minus GPS clock; solved, pos_held, rejections.
  known_       the same from the residual script's known-antenna variant (reference position), without solved, pos_held and rejections.

A fourth block, geometry (added to the features of Research_Plan_Topu.pdf, section 2.5: the raw Doppler, Doppler-rate and single-difference
statistics and PDOP, whose level and spread across satellites are set by the sky at that hour and the receiver clock drift. Their window
change statistics (standard deviation, skewness, kurtosis) are in the main sets; their other window statistics enter only a "+ geometry"
ablation.

part 3: a window of w seconds ending at second t (t-w+1..t) speaks for t; it must lie in one capture and hold no excluded second;
an uncertain decision second keeps the window, unscored. Statistics as Pervysheva et al. (2025) Table 4 prints them (population
std, skewness and kurtosis in the 1/N form, SoS Diff = last^2 - first^2 by telescoping, TK the mean of x_i^2 - x_(i-1) x_(i+1)); SoS is
rescaled to w samples when some are missing (a stated departure). Lengths 1, 2 and 3 omit the statistics that are identical or undefined
there. Each window length and receiver gives an index file (keys, decision label, scoring mask: the only window file with a label) and
four feature files (orbitfree, geometry, self, known); features_window_columns.csv says which column belongs to which feature set.

Every feature uses the receiver's own time: a time difference exists only when the previous epoch of the satellite lies 1 s
(+-0.1 s) earlier on the receiver's clock. Host time only places the 2.6.12 u-blox 01 rows on the grid (the evidence script's index). Seconds outside the
span the engine reads (2.6.5 before 12:34:57) are empty. Nothing is fitted to data here.

The data dictionary features_base_lineage.csv names every column's sources; codes/tests/test_features.py fails if any barred source or name
reaches a feature column, and runs build_base() again with every barred file made unreadable.

Reads   Data/rinex/<session>/*.crx.gz, Data/processed/features/{residuals,epochs}_{self,known}.csv.gz,
        Data/processed/{session_ground_truth,glonass_channels,rinex_ubx01_2612_pieces}.csv, labels/labels_1hz.csv (the three key columns for the
        grid, the label column only in the part 2 report, never in build_base), the 2.6.12 u-blox 01 binary, host-time sidecar and capture
        clock table (grid placement only, through the evidence script), features/observables_1hz.csv.gz (reconciliation checks only)
Writes  Data/processed/features/features_base_1hz.csv.gz, Data/processed/features/features_base_lineage.csv,
        Data/processed/features/features_window_columns.csv, Data/processed/features/w{1,2,3,5,10}/{receiver}_{index,orbitfree,geometry,self,known}.csv.gz
        (window folders in .gitignore), Data/processed/checks/windows_window_files.csv,
        Data/processed/checks/base_checks.csv, base_numbers.csv, base_missing.csv, base_top5.csv, Graph/base_quantities.png
"""

from __future__ import annotations  # future import: type hints kept as text (a convention across the project's scripts)

import importlib.util  # standard-library module loader, for modules whose names start with a digit
import os  # operating-system access: MPLCONFIGDIR is set before matplotlib loads
import random  # standard-library random; seeded below
import sys  # standard-library system access: sys.path and sys.modules
import time  # standard-library clock, for the run-time line
import warnings  # standard-library warnings, to silence NumPy's notice about windows with no sample (they are NaN by design)
from pathlib import Path  # Path objects for building file paths

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
os.environ["MPLCONFIGDIR"] = str(PROJECT / "codes" / ".mplconfig")  # matplotlib's cache inside the project folder

import numpy as np  # NumPy: arrays, medians, linear algebra
import pandas as pd  # pandas: tables
import matplotlib  # plotting library
matplotlib.use("Agg")  # file-only backend, no window
import matplotlib.pyplot as plt  # the pyplot interface
import matplotlib.dates as mdates  # date tick locators and formatters

sys.path.insert(0, str(PROJECT / "codes" / "common"))  # puts codes/ first on the import path so the helpers module is found
from helpers import exit_if_failed, save_fig, cliffs_delta  # byte-stable figure saving; Cliff's delta (report only); exit_if_failed ends the run on a failed check

SEED = 20260914  # integer constant: the project's fixed random seed
random.seed(SEED)  # seeds Python's generator
np.random.seed(SEED)  # seeds NumPy's generator (the reconciliation sample below draws from its own seeded generator)

PROC = PROJECT / "Data" / "processed"  # where the input CSVs are
OUT = PROC / "features"  # where this section's tables go
CHECK_DIR = PROJECT / "Data" / "processed" / "checks"  # where the check tables go
GRAPH = PROJECT / "Graph"  # where the figures go


def load(name: str, filename: str):  # function: loads codes/<filename> as a module (names start with a digit)
    spec = importlib.util.spec_from_file_location(name, PROJECT / "codes" / filename)  # where to load from
    mod = importlib.util.module_from_spec(spec)  # an empty module object
    sys.modules[name] = mod  # registered, so the module's own imports resolve
    spec.loader.exec_module(mod)  # runs the file's top level (its main() is guarded and not run)
    return mod  # the module


A = load("residuals_for_window_features", "4_features/02_residuals.py")  # 07a: the engine's constants and, through it, the observables script, the evidence script and the position script
E, D, M = A.E, A.D, A.M  # the observables script (RINEX reader), the evidence script (spans, 2.6.12 u-blox host-time index), the position script (constants)

KEYS = ["test_code", "receiver", "sec"]  # the three key columns of the label grid (the only label-table columns build_base reads)
GT_COLS = ["test_code", "session_folder", "capture_start_utc", "capture_end_utc"]  # the only session-table columns build_base reads (the schedule is a label source and stays out)
SYS = "GREC"  # the four constellations, in a fixed order
MASK = A.MASK  # 10 degree elevation mask (residual block only, the orbit-free block has none)
MS_M, JUMP_MIN_M, JUMP_TOL_M, JUMP_KMAX = A.MS_M, A.JUMP_MIN_M, A.JUMP_TOL_M, A.JUMP_KMAX  # the millisecond-jump rule of the residual script, reused for the common step
DT_TOL = 0.1  # a time difference needs the previous epoch 1 s +- 0.1 s earlier on the receiver's clock
DOP_REJ_M_S = 5.0  # self-variant Doppler fit: a post-fit range-rate residual beyond 5 m/s (about 26 Hz) is rejected, largest deletion residual first
FREE_Q = ["cn0", "dop", "dop_rate", "dC", "d2C", "dC_dc", "cmc_rate"]  # list: the orbit-free per-satellite quantities
STATS = ["median", "maxabs", "std"]  # list: the three per-constellation statistics
RXS, SESSIONS = A.RXS, A.SESSIONS  # the three receivers and the six sessions

CHECKS: list = []  # list: the checks, written to base_checks.csv
FAILED: list = []  # list: names of the self-checks that failed; the script ends the run with status 1 when it is not empty
NUM: dict = {}  # dict: numbers the summary quotes
LINEAGE: dict = {}  # dict: column -> (block, unit, sources, definition), written as the data dictionary

SRC = {"code": "RINEX:C1C/C1X/C2I", "carrier": "RINEX:L1C/L1X/L2I", "doppler": "RINEX:D1C/D1X/D2I", "cn0": "RINEX:S1C/S1X/S2I", "lli": "RINEX:LLI(carrier)",  # dict: source tokens (the L1-band observation codes)
       "chan": "RINEX:header GLONASS SLOT/FRQ", "rxtime": "receiver time (differences only)", "igs": "IGS broadcast (causal)", "atmo": "Klobuchar+Saastamoinen",  # header table, own clock, orbits, atmosphere
       "self": "self-solved position", "ref": "reference position"}  # the two positions
UNIT = {"cn0": "dB-Hz", "dop": "Hz", "dop_rate": "Hz/s", "dC": "m/s", "d2C": "m/s^2", "dC_dc": "m/s", "cmc_rate": "m/s", "res": "m", "res_db": "m", "dop_res": "Hz"}  # dict: units of the per-satellite quantities
FREE_SRC = {"cn0": ["cn0"], "dop": ["doppler"], "dop_rate": ["doppler", "rxtime"], "dC": ["code", "rxtime"], "d2C": ["code", "rxtime"], "dC_dc": ["code", "doppler", "rxtime", "chan"], "cmc_rate": ["code", "carrier", "rxtime", "chan"]}  # dict: sources of each orbit-free quantity
GEOM_Q = {"dop": "its level and its spread across satellites are set by each satellite's line-of-sight velocity (the sky at that hour) plus the receiver clock drift",  # dict: the orbit-free quantities moved to the geometry block, with the reason
          "dop_rate": "its level and spread are set by each satellite's line-of-sight acceleration (the sky at that hour) plus the drift rate",  # Doppler rate
          "dC": "its level and spread are the range rates (the sky at that hour) plus the receiver clock drift, which differs by session"}  # raw single difference
GEOM_PDOP = "PDOP is fixed by where the satellites stand in the sky at that hour (and by which are tracked)"  # string: the reason for PDOP
FREE_DEF = {"cn0": "carrier-to-noise density", "dop": "raw Doppler (RINEX sign)", "dop_rate": "Doppler minus the previous epoch's", "dC": "pseudorange minus the previous epoch's (clock drift still in)",  # dict: one-line definitions
            # dict entry: plain definitions of the second code difference and the drift-corrected code difference
            "d2C": "second difference of the pseudorange over three consecutive epochs", "dC_dc": "dC plus wavelength times the mean Doppler of the two epochs: receiver clock drift and satellite motion cancel",
            "cmc_rate": "first difference of pseudorange minus wavelength times carrier phase"}  # continued


def check(name: str, expected: str, found, ok: bool | None) -> None:  # function: records one check and prints it
    CHECKS.append({"check": name, "expected": expected, "found": found, "ok": "" if ok is None else bool(ok)})  # append a dict row
    if ok is not None and not bool(ok):  # if: a pass/fail check (None marks a reported number) that did not pass
        FAILED.append(name)  # its name kept for the exit at the end of the run
    print(f"  [{'ok' if ok else ('FAIL' if ok is False else '--')}] {name}: expected {expected}; found {found}")  # progress line


def lin(col: str, block: str, unit: str, sources: list, definition: str) -> None:  # function: records one column's lineage for the data dictionary
    LINEAGE[col] = (block, unit, ";".join(SRC[s] for s in sources), definition)  # generator expression: the source tokens joined


def wavelengths() -> dict:  # function: carrier wavelength (m) per satellite name prefix: G, E, C by constellation, GLONASS by slot (the observables script's table from the RINEX headers)
    lam = {"G": E.LAM["G"], "E": E.LAM["E"], "C": E.LAM["C"]}  # GPS L1, Galileo E1, BeiDou B1I
    ch = pd.read_csv(PROC / "ground_truth" / "glonass_channels.csv")  # slot -> channel, union of the RINEX headers
    for s, k in zip(ch.slot, ch.channel):  # for loop over the slots
        lam[s] = A.C / (E.GLO_F0 + k * E.GLO_DF)  # that slot's G1 wavelength
    return lam  # the dict


def read_full(gt: pd.DataFrame, test: str, rx: str) -> tuple:  # function: every 1 Hz observation row of a pair (any of code, carrier, Doppler, C/N0), receiver time kept; and the span read
    if (test, rx) == ("2.6.12", "UBX_01"):  # if: the u-blox whose time the attack moved
        orig = D.read_full  # the evidence script's own reader, remembered
        D.read_full = lambda path, spans, r: E.read_all(path, spans, r).assign(t_rx=lambda d: d.t_gps)  # lambda: the observables script's reader, receiver time copied before the evidence script re-times the rows
        try:  # try: the original reader always comes back
            obs = D.ubx_2612_obs(gt)  # rows placed on the grid by host time; t_rx keeps the receiver's time
        finally:  # finally: runs even after an error
            D.read_full = orig  # restored
    else:  # else: every other pair
        obs = E.read_all(M.obs_path(gt.loc[test, "session_folder"], test, rx), D.capture_spans(gt, test, rx), rx)  # rows in the capture span (2.6.5 from the site start)
        obs["t_rx"] = obs.t_gps  # the receiver's time is the epoch time
    D.LINES.clear()  # frees the decompressed RINEX
    obs = obs[obs.sv.str[0].isin(list(SYS))]  # the four constellations
    return obs.sort_values(["sv", "sec"]).reset_index(drop=True)[["sec", "t_rx", "sv", "pr", "cp", "dop", "cn0", "lli_cp"]], D.capture_spans(gt, test, rx)[0]  # satellite-major order; the span (start, end+1)


def per_satellite(o: pd.DataFrame, lam: dict) -> pd.DataFrame:  # function: the orbit-free per-satellite quantities, differences on the receiver's clock, and the common millisecond step
    o = o.copy()  # a copy
    o["sys"] = o.sv.str[0]  # constellation letter
    o["lam"] = [lam.get(s, np.nan) if s[0] == "R" else lam[s[0]] for s in o.sv]  # list comprehension: wavelength per row (GLONASS per slot; NaN if a slot had no channel)
    g = o.groupby("sv")  # one group per satellite, rows in time order
    t1, t2 = g.t_rx.shift(1), g.t_rx.shift(2)  # receiver times of the satellite's previous two rows
    c1 = ((o.t_rx - t1) - 1.0).abs() <= DT_TOL  # the previous row is 1 s earlier on the receiver's clock
    c2 = c1 & (((t1 - t2) - 1.0).abs() <= DT_TOL)  # and the one before that too
    pr1, pr2, d1, cp1 = g.pr.shift(1), g.pr.shift(2), g.dop.shift(1), g.cp.shift(1)  # previous values of the satellite
    o["dop_rate"] = np.where(c1, o.dop - d1, np.nan)  # Doppler rate, Hz/s
    o["dC"] = np.where(c1, o.pr - pr1, np.nan)  # single difference of the pseudorange, m/s
    o["d2C"] = np.where(c2, o.pr - 2.0 * pr1 + pr2, np.nan)  # second difference, m/s^2
    o["dC_dc"] = np.where(c1, o.dC + o.lam * (o.dop + d1) / 2.0, np.nan)  # drift-corrected: range rate from the code minus range rate from the Doppler (RINEX sign: D > 0 when the range shortens)
    o["cmc_rate"] = np.where(c1, (o.pr - o.lam * o.cp) - (pr1 - o.lam * cp1), np.nan)  # code-minus-carrier rate, m/s
    o["lock_loss"] = (o.lli_cp % 2 == 1).astype(int)  # RINEX loss-of-lock bit 0 on the carrier
    med = o.groupby("sec").dC_dc.median()  # the per-second median of the drift-corrected difference over all satellites
    k = np.rint(med / MS_M)  # nearest whole number of milliseconds
    step = (med.abs() >= JUMP_MIN_M) & ((med - k * MS_M).abs() <= JUMP_TOL_M) & (k.abs() >= 1) & (k.abs() <= JUMP_KMAX)  # a receiver clock step of whole milliseconds common to all satellites (the residual script's jump rule)
    st = o.sec.isin(step[step].index)  # rows in a step second
    stp = g.sec.shift(1).isin(step[step].index) & c1  # rows whose previous epoch is a step second (their second difference spans the step)
    o.loc[st, ["dC", "dC_dc", "cmc_rate"]] = np.nan  # the step is not a measurement: these differences are dropped, and counted below
    o.loc[st | stp, "d2C"] = np.nan  # the second difference on both sides of the step
    o.attrs["step_secs"] = sorted(int(s) for s in step[step].index)  # generator expression: the step seconds, kept for the count
    return o  # the table


def agg(o: pd.DataFrame, col: str, by: list) -> pd.DataFrame:  # function: median, largest absolute value and standard deviation (two or more satellites) of one quantity per group
    x = o[by + [col]].dropna()  # rows where the quantity exists
    gb = x.groupby(by)[col]  # grouped
    out = pd.DataFrame({"median": gb.median(), "maxabs": x.assign(a=x[col].abs()).groupby(by).a.max(), "std": gb.std(ddof=1)})  # the three statistics (pandas std needs two values, else NaN)
    return out  # indexed by the group


def free_block(o: pd.DataFrame, grid: np.ndarray) -> pd.DataFrame:  # function: the orbit-free per-second columns of one pair on its grid seconds (inside the span)
    sec = pd.Series(grid)  # the grid seconds
    cols = {"sec": grid}  # dict: column name -> values, assembled once at the end (faster than adding columns one by one)
    for q in FREE_Q:  # for loop over the quantities
        a = agg(o, q, ["sec", "sys"])  # per second and constellation
        for c in SYS:  # for loop over the constellations
            for s in STATS:  # for loop over the statistics
                cols[f"{c}_{q}_{s}"] = sec.map(a[s].xs(c, level="sys")).to_numpy() if c in a.index.get_level_values("sys") else np.full(len(grid), np.nan)  # conditional expression: that constellation's values, or empty
    n = o.groupby(["sec", "sys"]).size().unstack(fill_value=0).reindex(columns=list(SYS), fill_value=0)  # satellites tracked per second and constellation
    ll = o.groupby(["sec", "sys"]).lock_loss.sum().unstack(fill_value=0).reindex(columns=list(SYS), fill_value=0)  # lock-loss flags per second and constellation
    for c in SYS:  # for loop over the constellations
        nt = sec.map(n[c]).fillna(0)  # tracked count (0 when nothing tracked)
        cols[f"{c}_n_tracked"], cols[f"{c}_d_n_tracked"] = nt.to_numpy(), nt.diff().to_numpy()  # the count and its change from the previous grid second
        cols[f"{c}_n_lock_loss"] = sec.map(ll[c]).fillna(0).to_numpy()  # carrier lock-loss flags
    tot = pd.Series(sum(cols[f"{c}_n_tracked"] for c in SYS))  # generator expression: all four together
    cols["n_tracked"], cols["d_n_tracked"] = tot.to_numpy(), tot.diff().to_numpy()  # total and its change
    sets = o.groupby("sec").sv.apply(set)  # set of satellites tracked each second
    prev = [sets.get(s - 1, set()) for s in grid]  # list comprehension: the previous grid second's set (empty when nothing)
    cur = [sets.get(s, set()) for s in grid]  # this second's set
    cols["n_new"] = np.array([len(a - b) for a, b in zip(cur, prev)], dtype=float)  # list comprehension: satellites not tracked the second before
    cols["n_lost"] = np.array([len(b - a) for a, b in zip(cur, prev)], dtype=float)  # satellites tracked the second before and not now
    cols["n_new"][:1], cols["n_lost"][:1] = np.nan, np.nan  # the first second of the span has no previous second
    a = o.dropna(subset=["cn0"]).groupby("sec").cn0  # C/N0 over all satellites per second
    cols["all_cn0_median"], cols["all_cn0_std"] = sec.map(a.median()).to_numpy(), sec.map(a.std(ddof=1)).to_numpy()  # median and spread across the tracked set
    cols["ms_step"] = sec.isin(o.attrs["step_secs"]).astype(int).to_numpy()  # 1 in a second of a common whole-millisecond step of the time differences
    return pd.DataFrame(cols)  # the block


def free_lineage() -> None:  # function: the orbit-free block's data-dictionary rows
    for q in FREE_Q:  # for loop over the quantities
        for c in SYS:  # for loop over the constellations
            for s in STATS:  # for loop over the statistics
                lin(f"{c}_{q}_{s}", "geometry" if q in GEOM_Q else "orbit-free", UNIT[q], FREE_SRC[q], f"{s} over the {c} satellites of the second: {FREE_DEF[q]}" + (f" [geometry block: {GEOM_Q[q]}]" if q in GEOM_Q else ""))  # one row; conditional expressions put the geometry quantities in their own block
    for c in SYS:  # for loop over the constellations
        lin(f"{c}_n_tracked", "orbit-free", "count", ["code", "carrier", "doppler", "cn0"], f"{c} satellites with any decision-14 observation")  # tracked count
        lin(f"{c}_d_n_tracked", "orbit-free", "count", ["code", "carrier", "doppler", "cn0"], "change from the previous grid second")  # its change
        lin(f"{c}_n_lock_loss", "orbit-free", "count", ["lli"], f"{c} satellites with the carrier loss-of-lock bit set")  # lock losses
    lin("n_tracked", "orbit-free", "count", ["code", "carrier", "doppler", "cn0"], "satellites tracked, all four constellations")  # total
    lin("d_n_tracked", "orbit-free", "count", ["code", "carrier", "doppler", "cn0"], "change from the previous grid second")  # its change
    lin("n_new", "orbit-free", "count", ["code", "carrier", "doppler", "cn0"], "satellites tracked now and not in the previous grid second")  # new
    lin("n_lost", "orbit-free", "count", ["code", "carrier", "doppler", "cn0"], "satellites tracked in the previous grid second and not now")  # lost
    lin("all_cn0_median", "orbit-free", "dB-Hz", ["cn0"], "median C/N0 over all tracked satellites")  # median C/N0
    lin("all_cn0_std", "orbit-free", "dB-Hz", ["cn0"], "standard deviation of C/N0 over all tracked satellites")  # C/N0 spread
    lin("ms_step", "orbit-free", "0/1", ["code", "doppler", "rxtime", "chan"], "the per-second median of dC_dc lies within 1 km of a whole number (1-20) of milliseconds: a common receiver clock step")  # the step flag


def doppler_res(r: pd.DataFrame, variant: str) -> np.ndarray:  # function: Doppler residual (Hz) per row: known, one drift per second; self, a per-second fit of receiver velocity and drift
    y = -r.lam.to_numpy() * r.dop.to_numpy() - r.rr_m_s.to_numpy()  # measured range rate minus the modelled one at rest, m/s: velocity and drift terms remain
    out = np.full(len(r), np.nan)  # results
    ok = (r.reason.fillna("") == "").to_numpy() & np.isfinite(y)  # rows with a code residual (orbit, above the mask, no jump, clock) and a Doppler
    gec = ok & (r.sys != "R").to_numpy()  # GPS, Galileo, BeiDou rows: they set the drift (and the velocity), as in the position
    ix = r.groupby("sec").indices  # dict: second -> row positions
    for sec in sorted(ix):  # for loop over the seconds
        j = ix[sec]  # its rows
        jg, jo = j[gec[j]], j[ok[j]]  # rows that fit, rows that get a residual
        if variant == "known":  # if: known antenna: the receiver is at rest at the reference
            if len(jg) >= 4:  # if: four or more, as for the clock
                out[jo] = -(y[jo] - np.median(y[jg])) / r.lam.to_numpy()[jo]  # residual in Hz after the median drift (the observables script's definition)
            continue  # next second
        use = np.ones(len(jg), bool)  # rows still in the fit
        while use.sum() >= 5:  # loop: four unknowns plus one for redundancy
            k = jg[use]  # rows in the fit
            H = np.column_stack([-r[["ex", "ey", "ez"]].to_numpy()[k], np.ones(len(k))])  # partials: minus line of sight (velocity), 1 (drift)
            w = np.sin(np.radians(r.el_deg.to_numpy()[k])) ** 2  # elevation weights
            Nm = H.T @ (H * w[:, None])  # weighted normal matrix
            try:  # try: a degenerate geometry makes it singular
                th = np.linalg.solve(Nm, H.T @ (w * y[k]))  # velocity (3) and drift (1), m/s; used here only, never written
            except np.linalg.LinAlgError:  # except: singular
                break  # no fit this second
            rr_ = y[k] - H @ th  # post-fit residuals, m/s
            if np.abs(rr_).max() > DOP_REJ_M_S:  # if: one beyond 5 m/s
                lev = np.einsum("ij,jk,ik->i", H * np.sqrt(w)[:, None], np.linalg.inv(Nm), H * np.sqrt(w)[:, None])  # einsum: leverage of each row
                use[np.flatnonzero(use)[(np.abs(rr_) / np.maximum(1 - lev, 1e-6)).argmax()]] = False  # the largest deletion residual goes
                continue  # fit again
            Ho = np.column_stack([-r[["ex", "ey", "ez"]].to_numpy()[jo], np.ones(len(jo))])  # partials of every row that gets a residual
            out[jo] = -(y[jo] - Ho @ th) / r.lam.to_numpy()[jo]  # residual in Hz about the fitted velocity and drift
            break  # done with this second
    return out  # one value per row


def variant_block(v: str, r: pd.DataFrame, ep: pd.DataFrame, grid: np.ndarray, o: pd.DataFrame, lam: dict) -> pd.DataFrame:  # function: the residual block of one variant and pair on its grid seconds
    r = r.merge(o[["sec", "sv", "dop"]], on=["sec", "sv"], how="left").sort_values(["sec", "sv"]).reset_index(drop=True)  # the engine rows with their Doppler
    r["sys"] = r.sv.str[0]  # constellation letter
    r["lam"] = [lam.get(s, np.nan) if s[0] == "R" else lam[s[0]] for s in r.sv]  # list comprehension: wavelength per row
    r["dop_res"] = doppler_res(r, v)  # Doppler residual, Hz
    p = f"{v}_"  # column prefix
    sec = pd.Series(grid)  # the grid seconds
    cols = {"sec": grid}  # dict: column name -> values, assembled once at the end
    for q, systems in (("res", "GEC"), ("res_db", SYS), ("dop_res", SYS)):  # for loop over the quantities and the constellations each covers
        col_src = "res_m" if q == "res" else ("res_db_m" if q == "res_db" else "dop_res")  # nested conditional: the engine column
        a = agg(r.rename(columns={col_src: q}), q, ["sec", "sys"])  # per second and constellation
        for c in systems:  # for loop over the constellations
            for s in STATS:  # for loop over the statistics
                cols[f"{p}{c}_{q}_{s}"] = sec.map(a[s].xs(c, level="sys")).to_numpy() if c in a.index.get_level_values("sys") else np.full(len(grid), np.nan)  # conditional expression: values or empty
    cnt = r.assign(has=r.res_m.notna(), jump=r.jump_k != 0, outb=r.base_state.isin(["out", "gate"])).groupby(["sec", "sys"])[["has", "jump", "outb"]].sum()  # counts per second and constellation
    for c in SYS:  # for loop over the constellations
        for k, name in (("has", "n_res"), ("jump", "n_jump"), ("outb", "n_outband")):  # for loop over the three counts
            s_ = cnt[k].xs(c, level="sys") if c in cnt.index.get_level_values("sys") else pd.Series(dtype=float)  # conditional expression: that constellation's counts
            cols[f"{p}{c}_{name}"] = sec.map(s_).fillna(0).to_numpy()  # 0 where nothing was counted
    pd_ = {}  # second -> PDOP
    geo = r[(r.sys != "R") & r.ex.notna() & (r.el_deg >= MASK)]  # GPS, Galileo, BeiDou rows with a line of sight above the mask (before any rejection)
    for sc, g in geo.groupby("sec"):  # for loop over the seconds
        sy = sorted(set(g.sys))  # constellations present
        H = np.column_stack([g[["ex", "ey", "ez"]].to_numpy()] + [(g.sys == s).to_numpy().astype(float) for s in sy])  # list comprehension: line of sight plus one clock column per constellation
        if len(g) < H.shape[1]:  # if: fewer satellites than unknowns
            continue  # no DOP
        try:  # try: singular geometry
            pd_[sc] = float(np.sqrt(np.trace(np.linalg.inv(H.T @ H)[:3, :3])))  # PDOP: frame-free, so no position is needed to orient it
        except np.linalg.LinAlgError:  # except: singular
            pass  # no DOP this second
    cols[f"{p}pdop"] = sec.map(pd.Series(pd_, dtype=float)).to_numpy()  # PDOP per second
    clk = r.dropna(subset=["clk_m"]).groupby(["sec", "sys"]).clk_m.first().unstack()  # the median clock of each constellation per second, metres
    for c in "REC":  # for loop over GLONASS, Galileo, BeiDou
        cols[f"{p}isb_{c}G"] = sec.map(clk[c] - clk["G"]).to_numpy() if (c in clk and "G" in clk) else np.full(len(grid), np.nan)  # clock difference to GPS: an inter-system offset, never the clock itself
    if v == "self":  # if: the self-solved variant
        epi = ep.set_index("sec")  # the check by second
        for col in ("solved", "pos_held", "n_rej"):  # for loop over the solver columns kept as features
            cols[f"{p}{col}"] = sec.map(epi[col]).fillna(0).to_numpy()  # 0 in a second without any pseudorange
    e = pd.DataFrame(cols)  # the block
    return e  # the block


def variant_lineage(v: str) -> None:  # function: the residual block's data-dictionary rows
    p, pos = f"{v}_", ("self" if v == "self" else "ref")  # prefix and position source
    base = ["code", "igs", "atmo", "rxtime", pos]  # sources of every residual quantity
    for q, systems in (("res", "GEC"), ("res_db", SYS), ("dop_res", SYS)):  # for loop over the quantities
        for c in systems:  # for loop over the constellations
            for s in STATS:  # for loop over the statistics
                lin(f"{p}{c}_{q}_{s}", v, UNIT[q], base + (["doppler", "chan"] if q == "dop_res" else (["chan"] if c == "R" else [])), f"{s} over the {c} satellites: " + {"res": "code residual about the median clock", "res_db": "residual minus the satellite's own causal baseline", "dop_res": "measured minus modelled Doppler after the drift (known) or the velocity-and-drift fit (self)"}[q])  # one row (dict lookup by quantity)
    for c in SYS:  # for loop over the constellations
        lin(f"{p}{c}_n_res", v, "count", base, f"{c} satellites with a residual")  # residual count
        lin(f"{p}{c}_n_jump", v, "count", base, f"{c} satellites screened as whole-millisecond jumps")  # jump count
        lin(f"{p}{c}_n_outband", v, "count", base, f"{c} samples outside the baseline band or gate")  # out-of-band count
    lin(f"{p}pdop", "geometry", "-", ["code", "igs", pos], f"position dilution of precision of the G/E/C satellites above the mask [geometry block: {GEOM_PDOP}]")  # PDOP, in the geometry block
    for c in "REC":  # for loop over the three offsets
        lin(f"{p}isb_{c}G", v, "m", base, f"median clock of {c} minus that of GPS")  # inter-system offset
    if v == "self":  # if: self
        lin("self_solved", v, "0/1", base, "the epoch's position was solved")  # solved
        lin("self_pos_held", v, "0/1", base, "the position of a solution at most 30 s old was held")  # held
        lin("self_n_rej", v, "count", base, "observations rejected by the 30 m deletion rule")  # rejections


def build_base(gt: pd.DataFrame) -> tuple:  # function: the whole base table; returns it and the per-satellite orbit-free rows (for the checks)
    lam = wavelengths()  # carrier wavelengths
    grid_all = pd.read_csv(PROC / "labels" / "labels_1hz.csv", usecols=KEYS)  # the label grid: the three key columns only
    res = {v: pd.read_csv(OUT / f"residuals_{v}.csv.gz", keep_default_na=False, na_values=[""]) for v in A.VARIANTS}  # dict comprehension: the residual script's residual tables (reason "" kept as empty text)
    eps = {v: pd.read_csv(OUT / f"epochs_{v}.csv.gz") for v in A.VARIANTS}  # the residual script's check tables
    parts, persat = [], []  # per pair blocks; per-satellite rows
    for test in SESSIONS:  # for loop over the sessions
        for rx in RXS:  # for loop over the receivers
            t0 = time.time()  # the clock
            o, (a, b) = read_full(gt, test, rx)  # observations and the span read
            o = per_satellite(o, lam)  # orbit-free per-satellite quantities
            g = grid_all[(grid_all.test_code == test) & (grid_all.receiver == rx)]  # the pair's grid seconds
            inside = g.sec[(g.sec >= a) & (g.sec < b)].to_numpy()  # those inside the span read
            blk = free_block(o, inside)  # orbit-free block
            for v in A.VARIANTS:  # for loop over the variants
                rv = res[v][(res[v].test_code == test) & (res[v].receiver == rx)]  # the engine's rows of the pair
                ev = eps[v][(eps[v].test_code == test) & (eps[v].receiver == rx)]  # its check rows
                blk = blk.merge(variant_block(v, rv.drop(columns=["test_code", "receiver"]), ev, inside, o, lam), on="sec", how="left")  # residual block attached
            full = g[KEYS].merge(blk, on="sec", how="left")  # every grid second; outside the span every column stays empty
            parts.append(full)  # kept
            persat.append(o.assign(test_code=test, receiver=rx))  # per-satellite rows for the checks
            print(f"  {test} {rx}: {len(o)} satellite-seconds, {len(inside)} of {len(g)} grid seconds inside the span, {time.time() - t0:.0f} s")  # progress line
    base = pd.concat(parts, ignore_index=True)  # the table
    free_lineage()  # data dictionary: orbit-free block
    for v in A.VARIANTS:  # for loop over the variants
        variant_lineage(v)  # data dictionary: residual block
    cols = KEYS + list(LINEAGE)  # column order: keys, then the dictionary's order
    if set(base.columns) != set(cols):  # if: a column without a dictionary row, or the reverse
        raise SystemExit(f"STOP: columns and lineage differ: {sorted(set(base.columns) ^ set(cols))}")  # every feature must have its sources stated
    return base[cols], pd.concat(persat, ignore_index=True)  # the table in a fixed column order; per-satellite rows


def write_base(base: pd.DataFrame) -> None:  # function: the base table and its data dictionary, byte-stable
    OUT.mkdir(parents=True, exist_ok=True)  # the folder
    E.write_gz(base.round(4), OUT / "features_base_1hz.csv.gz")  # four decimals for every float (0.1 mm, 0.1 mHz)
    pd.DataFrame([{"column": k, "block": b, "unit": u, "sources": s, "definition": d} for k, (b, u, s, d) in LINEAGE.items()]).to_csv(OUT / "features_base_lineage.csv", index=False)  # list comprehension: the dictionary


# ---------------------------------------------------------------- part 2 checks and report (labels used here only)

def reconcile(base: pd.DataFrame, persat: pd.DataFrame) -> None:  # function: the orbit-free quantities against the observables script's table, the aggregates against an independent count
    ob = pd.read_csv(PROC / "features" / "observables_1hz.csv.gz", usecols=["test_code", "receiver", "sec", "sv", "cn0", "dop_hz", "dC_m", "d2C_m", "cmc_rate_m_s", "dop_res_hz", "lock_loss"])  # the observables script's per-satellite table
    m = persat.merge(ob, on=["test_code", "receiver", "sec", "sv"], how="outer", indicator=True)  # every row of either
    only = m._merge.value_counts().to_dict()  # rows in one only
    eq = lambda a, b: int((~(((a - b).abs() <= 1e-3) | (a.isna() & b.isna()))).sum())  # lambda: rows differing by more than 1e-3 (the observables script stores four decimals), NaN equal to NaN
    both = m[m._merge == "both"]  # rows in both
    d_cn0, d_dop, d_ll = eq(both.cn0_x, both.cn0_y), eq(both.dop, both.dop_hz), int((both.lock_loss_x != both.lock_loss_y).sum())  # raw quantities must be equal
    def_dc = both.dC.notna() & both.dC_m.notna()  # rows where both define the difference
    d_dc, d_d2c, d_cmc = eq(both.dC[def_dc], both.dC_m[def_dc]), eq(both.d2C[both.d2C.notna() & both.d2C_m.notna()], both.d2C_m[both.d2C.notna() & both.d2C_m.notna()]), eq(both.cmc_rate[both.cmc_rate.notna() & both.cmc_rate_m_s.notna()], both.cmc_rate_m_s[both.cmc_rate.notna() & both.cmc_rate_m_s.notna()])  # equal where both define them
    only_here, only_05e = int((both.dC.notna() & both.dC_m.isna()).sum()), int((both.dC.isna() & both.dC_m.notna()).sum())  # where the receiver-time rule and the observables script's grid rule disagree on existence
    NUM.update({"recon_rows": len(both), "recon_only": str(only), "recon_dC_only_here": only_here, "recon_dC_only_05e": only_05e})  # kept
    # ---- the receiver-time gaps behind the differences the observables script kept and this table's receiver-time rule drops, for 7.x's prose ----
    from helpers import write_facts  # import inside the function: the facts-table writer
    k = ["test_code", "receiver", "sv"]  # list: one satellite's rows of one session and receiver
    ps = persat.sort_values(k + ["sec"])[k + ["sec", "t_rx"]].copy()  # the satellite rows in time order with their receiver times
    ps["gap"] = ps.t_rx - ps.groupby(k).t_rx.shift(1)  # groupby + shift: receiver time since the satellite's previous kept row, s
    o5 = both.loc[both.dC.isna() & both.dC_m.notna(), k + ["sec"]].merge(ps, on=k + ["sec"], how="left")  # the rows the observables script defines and this table does not, with their gaps
    rt = o5[((o5.gap - 1.0).abs() > DT_TOL) | o5.gap.isna()]  # dropped by the receiver-time rule (previous row not 1 s +- DT_TOL earlier); the others fall to the millisecond-step rule
    sb = rt[rt.receiver != "UBX_01"].gap.round(2)  # on the Septentrio receivers, to 0.01 s (the mosaic-H runs at 20 Hz, so its gaps are multiples of 0.05 s)
    vc = sb.value_counts().sort_index()  # value_counts: rows per gap value, in gap order
    rows = [("only05e_rows_receiver_time_septentrio", int(len(sb)), "satellite-seconds", "rows the observables script defines and the feature script's receiver-time rule drops, on SBF_02 and SBF_03"),  # tuple: the Septentrio rows
            ("only05e_rows_receiver_time_ublox", int((rt.receiver == "UBX_01").sum()), "satellite-seconds", "the same on UBX_01 (its own time jumped in 2.6.12)"),  # the u-blox rows
            ("only05e_rows_step_rule", int(len(o5) - len(rt)), "satellite-seconds", "rows the observables script defines that 07 drops at a common whole-millisecond clock step, not by the receiver-time rule"),  # the step rows
            ("only05e_gaps_septentrio", ";".join(f"{g:g}" for g in vc.index), "s", "distinct receiver-time intervals to the satellite's previous kept row, to 0.01 s, over the Septentrio receiver-time rows")]  # list of 4-tuples
    rows += [(f"only05e_gap_{g:g}_rows", int(n), "satellite-seconds", f"Septentrio receiver-time rows whose interval is {g:g} s") for g, n in vc.items()]  # list comprehension: rows per gap
    write_facts(__file__, rows)  # written as 03_window_features_facts.csv
    check("Orbit-free per-satellite rows and raw values against the observables script (features/observables_1hz.csv.gz)", "same rows; C/N0, Doppler and lock loss equal; differences equal where both define them", f"{len(both)} rows in both, {only}; differing C/N0 {d_cn0}, Doppler {d_dop}, lock loss {d_ll}; dC {d_dc}, d2C {d_d2c}, cmc rate {d_cmc}; dC defined here only {only_here}, in the observables script only {only_05e} (the receiver-time rule against the observables script's grid rule, and the common ms step)", bool(only.get("left_only", 0) == 0 and only.get("right_only", 0) == 0 and d_cn0 == 0 and d_dop == 0 and d_ll == 0 and d_dc == 0 and d_d2c == 0 and d_cmc == 0))  # the test
    ep = pd.read_csv(PROC / "features" / "observables_epochs_1hz.csv.gz", usecols=["test_code", "receiver", "sec", "n_G", "n_R", "n_E", "n_C"])  # the observables script's per-second counts
    x = base[KEYS + [f"{c}_n_tracked" for c in SYS]].merge(ep, on=KEYS, how="inner")  # seconds in both
    bad = sum(int((x[f"{c}_n_tracked"] != x[f"n_{c}"]).sum()) for c in SYS)  # generator expression: count disagreements
    check("Tracked counts per constellation against the observables script's epoch table", "equal on every common second", f"{len(x)} seconds, {bad} disagreements", bad == 0)  # the test
    rng = np.random.default_rng(20260914)  # a seeded generator for the sample
    smp = base[base.n_tracked > 0].iloc[rng.choice(int((base.n_tracked > 0).sum()), 600, replace=False)]  # 600 seconds with observations
    n_bad = 0  # disagreements
    for r in smp.itertuples():  # for loop over the sample, recomputed with plain Python
        rows = persat[(persat.test_code == r.test_code) & (persat.receiver == r.receiver) & (persat.sec == r.sec)]  # that second's satellites
        for c in SYS:  # for loop over the constellations
            v = sorted(float(z) for z in rows[(rows.sys == c)].cn0 if z == z)  # generator expression: C/N0 values (z == z is False for NaN)
            med = (v[(len(v) - 1) // 2] + v[len(v) // 2]) / 2 if v else np.nan  # conditional expression: the median by hand
            n_bad += int(not ((np.isnan(med) and np.isnan(getattr(r, f"{c}_cn0_median"))) or abs(round(med, 4) - getattr(r, f"{c}_cn0_median")) < 1e-3))  # agree to 1e-3 dB
    check("Aggregates recomputed by hand (C/N0 median per constellation) on 600 random seconds", "equal", f"{n_bad} disagreements of {600 * 4}", n_bad == 0)  # the test


def magnitudes(base: pd.DataFrame) -> None:  # function: units and physical magnitudes on clean seconds, reported against plausible ranges
    lab = pd.read_csv(PROC / "labels" / "labels_1hz.csv", usecols=KEYS + ["label"])  # labels (report only)
    c = base.merge(lab, on=KEYS)  # attached
    c = c[c.label == "clean"]  # clean seconds
    rows = []  # one row per quantity
    for col in ["G_cn0_median", "G_dop_maxabs", "G_dop_rate_maxabs", "G_dC_median", "G_d2C_maxabs", "G_dC_dc_maxabs", "G_cmc_rate_maxabs", "known_G_dop_res_maxabs", "self_G_dop_res_maxabs", "self_pdop", "self_isb_EG", "self_isb_CG", "self_isb_RG"]:  # for loop over representative columns
        a = c[col].dropna()  # values
        rows.append(f"{col} {a.quantile(0.01):.3g}..{a.median():.3g}..{a.quantile(0.99):.3g}")  # 1st, 50th and 99th percentile
    check("Magnitudes on clean seconds (1st..median..99th percentile)", "C/N0 20-55 dB-Hz; Doppler a few kHz; dC hundreds of m/s (receiver drift); dC_dc and Doppler residual near 0", "; ".join(rows), None)  # reported


def missing_report(base: pd.DataFrame) -> None:  # function: rows, columns and missing share per column per receiver, by label (report)
    lab = pd.read_csv(PROC / "labels" / "labels_1hz.csv", usecols=KEYS + ["label"])  # labels (report only)
    b = base.merge(lab, on=KEYS)  # attached
    feat = [c for c in base.columns if c not in KEYS]  # list comprehension: the feature columns
    rows = []  # long table
    for rx in RXS:  # for loop over the receivers
        x = b[b.receiver == rx]  # its rows
        nonex = x[x.label != "excluded"]  # rows that can enter a window
        miss_all, miss_ne = x[feat].isna().mean(), nonex[feat].isna().mean()  # missing share over all rows and over non-excluded rows
        by = {lb: x[x.label == lb][feat].isna().mean() for lb in ("clean", "attack_jam_spoof", "attack_spoof", "uncertain")}  # dict comprehension: per label
        for col in feat:  # for loop over the columns
            rows.append({"receiver": rx, "column": col, "block": LINEAGE[col][0], "missing_all": miss_all[col], "missing_not_excluded": miss_ne[col], **{f"missing_{lb}": s[col] for lb, s in by.items()}})  # dict unpacking: one row
        NUM[f"{rx}_rows"], NUM[f"{rx}_cols"] = len(x), len(feat)  # sizes
        for blk in ("orbit-free", "geometry", "self", "known"):  # for loop over the blocks
            cs = [c for c in feat if LINEAGE[c][0] == blk]  # list comprehension: the block's columns
            NUM[f"{rx}_{blk}_cols"] = len(cs)  # its width
            NUM[f"{rx}_{blk}_miss_median_nonexcl"] = float(miss_ne[cs].median())  # median missing share of its columns, non-excluded rows
            for lb, s in by.items():  # for loop over labels
                NUM[f"{rx}_{blk}_miss_median_{lb}"] = float(s[cs].median())  # median missing share by label
    pd.DataFrame(rows).round(4).to_csv(CHECK_DIR / "base_missing.csv", index=False)  # written


def top5(base: pd.DataFrame) -> None:  # function: report only: the five main-set columns that differ most between clean and attack_spoof seconds, per receiver (nothing is selected on it)
    lab = pd.read_csv(PROC / "labels" / "labels_1hz.csv", usecols=KEYS + ["label"])  # labels (report only)
    b = base.merge(lab, on=KEYS)  # attached
    main = [c for c in base.columns if c not in KEYS and LINEAGE[c][0] in ("orbit-free", "self")]  # list comprehension: the main set (orbit-free plus self)
    out = []  # rows
    for rx in RXS:  # for loop over the receivers
        cl, at = b[(b.receiver == rx) & (b.label == "clean")], b[(b.receiver == rx) & (b.label == "attack_spoof")]  # the two groups
        rows = []  # per column
        for col in main:  # for loop over the columns
            a, c = at[col].dropna(), cl[col].dropna()  # non-missing values
            if len(a) < 30 or len(c) < 30 or a.nunique() < 2 and c.nunique() < 2:  # if: too few values, or constant in both
                continue  # not reported
            rows.append({"receiver": rx, "column": col, "cliffs_delta": cliffs_delta(a, c), "n_clean": len(c), "n_attack_spoof": len(a), "miss_clean": cl[col].isna().mean(), "miss_attack_spoof": at[col].isna().mean(), "clean_q25": c.quantile(0.25), "clean_median": c.median(), "clean_q75": c.quantile(0.75), "attack_q25": a.quantile(0.25), "attack_median": a.median(), "attack_q75": a.quantile(0.75), "sessions_attack": ",".join(sorted(at.loc[a.index].test_code.unique()))})  # dict literal: the comparison
        r = pd.DataFrame(rows)  # the table
        r["abs_delta"] = r.cliffs_delta.abs()  # magnitude
        out.append(r.sort_values(["abs_delta", "column"], ascending=[False, True]).head(5))  # the five largest
    pd.concat(out, ignore_index=True).round(4).to_csv(CHECK_DIR / "base_top5.csv", index=False)  # written


def plot_base_quantities(base: pd.DataFrame) -> None:  # function: the figure, base quantities through 2.6.2 on the three receivers, labels shaded for context
    lab = pd.read_csv(PROC / "labels" / "labels_1hz.csv", usecols=KEYS + ["label"])  # labels (shading only)
    b = base.merge(lab, on=KEYS)  # attached
    b = b[b.test_code == "2.6.2"]  # the session with clean margins on both sides of its attack
    fig, ax = plt.subplots(5, 1, figsize=(12, 12), sharex=True)  # five panels, one time axis
    panels = [("G_cn0_median", "GPS C/N0, median (dB-Hz)", "linear"), ("G_dC_dc_std", "GPS drift-corrected dC, std (m/s)", "log"), ("self_G_res_maxabs", "self: GPS |residual|, largest (m)", "log"), ("known_G_res_maxabs", "known: GPS |residual|, largest (m)", "log"), ("n_tracked", "satellites tracked", "linear")]  # list: (column, label, scale)
    for k, (col, ylab, sc) in enumerate(panels):  # for loop over the panels
        for rx in RXS:  # for loop over the receivers
            x = b[b.receiver == rx]  # its rows
            ax[k].plot(pd.to_datetime(x.sec, unit="s"), x[col], lw=0.8, color=M.COLORS[rx], label=A.RXN[rx])  # the series
        ax[k].set_yscale(sc)  # axis scale
        ax[k].set_ylabel(ylab, fontsize=8)  # axis label
        x = b[b.receiver == "SBF_02"]  # the X5's labels for the shading (the pooled attack interval is the same on all three)
        for lb, colr in (("attack_spoof", "#D55E00"), ("clean", "#009E73")):  # for loop over the two shaded labels
            s = x[x.label == lb].sec.to_numpy()  # its seconds
            if len(s):  # if: any
                br = np.flatnonzero(np.diff(s) > 1)  # breaks between runs
                for i0, i1 in zip(np.r_[0, br + 1], np.r_[br, len(s) - 1]):  # for loop over the runs
                    ax[k].axvspan(pd.to_datetime(s[i0], unit="s"), pd.to_datetime(s[i1] + 1, unit="s"), color=colr, alpha=0.12, lw=0)  # shaded run
    ax[0].legend(fontsize=8, loc="lower left")  # legend
    ax[0].set_title("2.6.2: base quantities per second (green: clean seconds of the X5; orange: the pooled attack interval)", fontsize=10)  # title
    ax[-1].xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))  # ticks
    ax[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))  # UTC clock
    fig.tight_layout()  # layout
    save_fig(fig, GRAPH / "base_quantities.png")  # byte-stable save


# ---------------------------------------------------------------- part 3: windows 

WINDOWS = [1, 2, 3, 5, 10]  # list: window lengths in seconds
ALL10 = ["mean", "std", "median", "min", "max", "skew", "kurt", "sos", "sosdiff", "tk"]  # list: the ten window statistics, as Pervysheva et al. (2025) Table 4 prints them
STATS_W = {1: ["mean", "sos"], 2: ["mean", "min", "max", "std", "sos", "sosdiff"], 3: ["mean", "median", "min", "max", "std", "sos", "sosdiff", "tk"], 5: ALL10, 10: ALL10}  # dict: statistics written per window length; the others are identical or undefined there
CHANGE = {"std", "skew", "kurt"}  # set: statistics that do not move when a constant is added to the window, i.e. measures of change and shape, not level
PARTS = ["orbitfree", "geometry", "self", "known"]  # list: the four window files per window length and receiver, by the base column's block
PART_OF = {"orbit-free": "orbitfree", "geometry": "geometry", "self": "self", "known": "known"}  # dict: base block -> window file part
WIN_DIR = lambda w: OUT / f"w{w}"  # lambda: the folder of one window length (in .gitignore; hashes logged in checks_log)


def win_stats(X: np.ndarray, w: int) -> dict:  # function: the window statistics of every column for windows ending at each row; X is (rows, columns) of one pair in time order
    n, k = X.shape  # rows and columns
    P = np.vstack([np.full((w - 1, k), np.nan), X])  # w-1 empty rows in front, so every row has a full-length window (incomplete windows are dropped later)
    V = np.lib.stride_tricks.sliding_window_view(P, w, axis=0)  # view (rows, columns, w): the window ending at each row, oldest sample first; nothing after the row is in it
    m = ~np.isnan(V)  # which samples exist
    cnt = m.sum(axis=2)  # samples present per window and column
    out = {}  # statistic -> (rows, columns)
    with np.errstate(all="ignore"), warnings.catch_warnings():  # context managers: empty windows give NaN without floating-point or runtime warnings
        warnings.simplefilter("ignore", category=RuntimeWarning)  # the all-NaN median notice is expected for empty windows
        mu = np.where(cnt >= 1, np.nansum(V, axis=2) / np.maximum(cnt, 1), np.nan)  # mean of the samples present
        dev = V - mu[:, :, None]  # deviations from the window mean
        var = np.where(cnt >= 2, np.nansum(dev ** 2, axis=2) / np.maximum(cnt, 1), np.nan)  # population variance (1/N, as printed), needs two samples
        sd = np.sqrt(var)  # standard deviation
        out["mean"], out["std"] = mu, sd  # level and spread
        if "median" in STATS_W[w]:  # if: the median is written at this length
            out["median"] = np.where(cnt >= 1, np.nanmedian(np.where(m, V, np.nan), axis=2), np.nan)  # median of the samples present
        out["min"] = np.where(cnt >= 1, np.nanmin(np.where(m, V, np.inf), axis=2), np.nan)  # smallest
        out["max"] = np.where(cnt >= 1, np.nanmax(np.where(m, V, -np.inf), axis=2), np.nan)  # largest
        ok3 = (cnt >= 4) & (sd > 0)  # skewness and kurtosis need four samples and a spread
        z = dev / sd[:, :, None]  # standardised deviations
        out["skew"] = np.where(ok3, np.nansum(z ** 3, axis=2) / np.maximum(cnt, 1), np.nan)  # skewness, 1/N form (as printed)
        out["kurt"] = np.where(ok3, np.nansum(z ** 4, axis=2) / np.maximum(cnt, 1), np.nan)  # kurtosis, 1/N form, not excess (as printed)
        out["sos"] = np.where(cnt >= 1, w * np.nansum(V ** 2, axis=2) / np.maximum(cnt, 1), np.nan)  # sum of squares, rescaled to w samples when some are missing (a stated departure)
        first = np.argmax(m, axis=2)  # position of the first sample present
        last = w - 1 - np.argmax(m[:, :, ::-1], axis=2)  # position of the last sample present
        xf = np.take_along_axis(V, first[:, :, None], axis=2)[:, :, 0]  # the first sample present
        xl = np.take_along_axis(V, last[:, :, None], axis=2)[:, :, 0]  # the last sample present
        out["sosdiff"] = np.where(cnt >= 2, xl ** 2 - xf ** 2, np.nan)  # sum of x_i^2 - x_(i-1)^2 telescopes to last^2 - first^2 (as printed)
        if w >= 3:  # if: a window long enough for the Teager-Kaiser operator
            psi = V[:, :, 1:-1] ** 2 - V[:, :, :-2] * V[:, :, 2:]  # x_i^2 - x_(i-1) x_(i+1) at the interior samples (NaN if any of the three is missing)
            np_ = (~np.isnan(psi)).sum(axis=2)  # interior triples present
            out["tk"] = np.where(np_ >= 1, np.nansum(psi, axis=2) / np.maximum(np_, 1), np.nan)  # their mean
    return {st: out[st] for st in STATS_W[w]}  # dict comprehension: only the statistics written at this length


def window_columns() -> pd.DataFrame:  # function: the window data dictionary: every window column, its base column, statistic, file part and feature-set membership
    rows = []  # one row per window column
    for w in WINDOWS:  # for loop over the window lengths
        for base, (blk, unit, src, _) in LINEAGE.items():  # for loop over the base columns
            for st in STATS_W[w]:  # for loop over the statistics written at this length
                geo = blk == "geometry"  # a geometry base column
                chg = st in CHANGE  # a change statistic
                origin = "known" if base.startswith("known_") else ("self" if base.startswith("self_") else "orbit-free")  # nested conditional: which source block the base column comes from
                core = (not geo) or chg  # in the main sets: every statistic of a non-geometry column, and the change statistics of a geometry column
                rows.append({"w": w, "column": f"{base}__{st}", "base_column": base, "stat": st, "part": PART_OF[blk], "origin": origin, "unit": unit, "sources": src,  # dict literal: the column
                             "no_residual": int(core and origin == "orbit-free"), "main": int(core and origin in ("orbit-free", "self")), "known_antenna": int(core and origin in ("orbit-free", "known")),  # the three feature sets
                             "geometry_ablation": int(geo and not chg)})  # added only in the "+ geometry" ablations
    return pd.DataFrame(rows)  # the dictionary


def window_rows(lab: pd.DataFrame, w: int) -> pd.DataFrame:  # function: the windows kept: per pair, complete, inside one capture, without an excluded second; labelled by the last second
    out = []  # per pair
    for (test, rx), g in lab.groupby(["test_code", "receiver"], sort=False):  # for loop over the pairs, in the grid's order
        ex = (g.label == "excluded").to_numpy().astype(int)  # 1 for an excluded second
        c = np.convolve(ex, np.ones(w, dtype=int))[:len(ex)]  # excluded seconds in the window ending at each second (convolution with w ones)
        keep = (np.arange(len(ex)) >= w - 1) & (c == 0)  # complete and without an excluded second
        x = g[keep][KEYS + ["label"]].copy()  # the kept windows: the decision second's keys and label
        x["scored"] = x.label.isin(["clean", "attack_jam_spoof", "attack_spoof"]).astype(int)  # the scoring module's scoring mask: uncertain windows stay in the series, unscored
        x["pos"] = np.flatnonzero(keep)  # row position of the decision second within the pair (to pick the statistics)
        out.append(x)  # kept
    return pd.concat(out, ignore_index=True)  # all pairs


def build_windows(base: pd.DataFrame, lab: pd.DataFrame, w: int) -> tuple:  # function: one window length: the index (keys, label, scored) and the four feature parts per receiver
    wc = window_columns()  # the dictionary
    wc = wc[wc.w == w]  # this length
    idx = window_rows(lab, w)  # kept windows
    feat = list(LINEAGE)  # base feature columns, in dictionary order
    parts = {rx: {pt: [] for pt in PARTS} for rx in RXS}  # nested dict comprehension: receiver -> part -> list of blocks
    for (test, rx), g in base.groupby(["test_code", "receiver"], sort=False):  # for loop over the pairs (base rows are in grid order)
        st = win_stats(g[feat].to_numpy(dtype=float), w)  # every statistic of every base column
        pos = idx[(idx.test_code == test) & (idx.receiver == rx)].pos.to_numpy()  # decision rows kept
        cols = {}  # window column -> values at the kept rows
        for stat, arr in st.items():  # for loop over the statistics
            for j, b in enumerate(feat):  # for loop over the base columns
                cols[f"{b}__{stat}"] = arr[pos, j]  # the statistic of that column at the kept windows
        blk = pd.DataFrame(cols)  # the pair's window columns
        keys = g.iloc[pos][KEYS].reset_index(drop=True)  # their keys
        for pt in PARTS:  # for loop over the parts
            cs = list(wc[wc.part == pt].column)  # that part's columns, in dictionary order
            parts[rx][pt].append(pd.concat([keys, blk[cs]], axis=1))  # keys plus the part's columns
    out = {rx: {pt: pd.concat(v, ignore_index=True) for pt, v in d.items()} for rx, d in parts.items()}  # nested dict comprehension: concatenated per receiver and part
    return idx.drop(columns="pos"), out  # the index and the parts


def write_windows(w: int, idx: pd.DataFrame, parts: dict) -> list:  # function: writes one window length; returns (file, rows, columns) for the log
    d = WIN_DIR(w)  # the folder
    d.mkdir(parents=True, exist_ok=True)  # created
    files = []  # written files
    for rx in RXS:  # for loop over the receivers
        f = d / f"{rx}_index.csv.gz"  # the index: keys, the decision second's label and the scoring mask (the only window file with a label)
        E.write_gz(idx[idx.receiver == rx], f)  # written
        files.append((f, int((idx.receiver == rx).sum()), idx.shape[1]))  # logged
        for pt in PARTS:  # for loop over the parts
            f = d / f"{rx}_{pt}.csv.gz"  # one part
            E.write_gz(parts[rx][pt].round(4), f)  # four decimals, as the base table
            files.append((f, len(parts[rx][pt]), parts[rx][pt].shape[1]))  # logged
    return files  # the list


def window_checks(base: pd.DataFrame, lab: pd.DataFrame, built: dict) -> None:  # function: part 3 self-checks: counts against an independent count and the expected table, no excluded second inside, statistics recomputed by hand
    first_counts = {1: {"SBF_02": 5867, "SBF_03": 6542, "UBX_01": 6244}, 2: {"SBF_02": 5859, "SBF_03": 6521, "UBX_01": 6218}, 3: {"SBF_02": 5851, "SBF_03": 6502, "UBX_01": 6197},  # dict: the expected windows per receiver, fixed before the windows were built
           5: {"SBF_02": 5835, "SBF_03": 6470, "UBX_01": 6159}, 10: {"SBF_02": 5795, "SBF_03": 6406, "UBX_01": 6080}}  # continued
    bad_n, bad_ex, bad_lab = [], 0, 0  # count mismatches; windows holding an excluded second; label mismatches
    off_first_counts = []  # counts that differ from the expected table, which holds the registered labels' counts: reported, not a failure
    for w, (idx, parts) in built.items():  # for loop over the window lengths
        for rx in RXS:  # for loop over the receivers
            n_idx = int((idx.receiver == rx).sum())  # windows in the index
            n_ind = 0  # an independent count: every second whose w-second window lies in one capture and holds no excluded second
            for (t, r), g in lab[lab.receiver == rx].groupby(["test_code", "receiver"]):  # for loop over the pairs
                lb = list(g.label)  # labels in time order
                n_ind += sum(1 for i in range(w - 1, len(lb)) if "excluded" not in lb[i - w + 1:i + 1])  # generator expression: plain-Python count
            ns = {pt: len(parts[rx][pt]) for pt in PARTS}  # rows per part
            if not (n_idx == n_ind and all(v == n_idx for v in ns.values())):  # if: the index, the independent count or a part disagree (the check)
                bad_n.append(f"w{w} {rx}: index {n_idx}, independent {n_ind}, parts {ns}")  # recorded
            if n_idx != first_counts[w][rx]:  # if: the count differs from the expected table (expected after any relabel, so a record only)
                off_first_counts.append(f"w{w} {rx}: {n_idx} against the expected table {first_counts[w][rx]}")  # recorded
        m = idx.merge(lab, on=KEYS, suffixes=("", "_lab"))  # the index against the labels
        bad_lab += int((m.label != m.label_lab).sum())  # the label is the decision second's
        ls = lab.set_index(KEYS).label  # label lookup
        smp = idx.iloc[np.linspace(0, len(idx) - 1, 400).astype(int)]  # 400 evenly spread windows
        for r in smp.itertuples():  # for loop over the sample
            bad_ex += sum(ls.get((r.test_code, r.receiver, r.sec - k), "missing") in ("excluded", "missing") for k in range(w))  # generator expression: every second of the window exists and is not excluded
    tot = {w: len(built[w][0]) for w in built}  # dict comprehension: windows per length
    NUM["windows_total"] = str(tot)  # kept
    check("Window counts: index against an independent plain-Python count, every part the same length", "all equal", f"totals {tot}; mismatches {bad_n or 'none'}", not bad_n)  # the test (the expected table no longer part of it)
    check("Window counts against the expected table (the registered labels' counts; a record, not a pass/fail check)", "reported", f"differences {off_first_counts or 'none'}", None)  # reported only
    check("Each window labelled by its decision (last) second; no window holds an excluded or missing second (400-window sample per length)", "0 and 0", f"label mismatches {bad_lab}; excluded or missing seconds inside {bad_ex}", bad_lab == 0 and bad_ex == 0)  # the test
    feat = list(LINEAGE)  # base columns
    rng = np.random.default_rng(20260914)  # seeded generator for the sample
    n_bad, n_cmp = 0, 0  # disagreements; comparisons
    for w in (2, 3, 5, 10):  # for loop over the lengths with several statistics
        idx, parts = built[w]  # this length
        allp = {rx: pd.concat([parts[rx][pt].drop(columns=KEYS) for pt in PARTS], axis=1).assign(**{k: parts[rx]["orbitfree"][k] for k in KEYS}) for rx in RXS}  # dict comprehension: every window column of a receiver, with the keys
        for rx in RXS:  # for loop over the receivers
            A_ = allp[rx]  # its windows
            for i in rng.choice(len(A_), 40, replace=False):  # for loop over 40 random windows
                r = A_.iloc[i]  # the window row
                g = base[(base.test_code == r.test_code) & (base.receiver == rx) & (base.sec > r.sec - w) & (base.sec <= r.sec)]  # its w base seconds
                for b in rng.choice(feat, 6, replace=False):  # for loop over 6 random base columns
                    xx = [float(v) if v == v else None for v in g[b]]  # list comprehension: the window's samples in time order, None where missing (v == v is False for NaN)
                    x = [v for v in xx if v is not None]  # the samples present
                    if len(x) < 4:  # if: too few for every statistic
                        continue  # skipped
                    mu = sum(x) / len(x)  # mean by hand
                    sd = (sum((v - mu) ** 2 for v in x) / len(x)) ** 0.5  # population standard deviation by hand
                    xs = sorted(x)  # sorted, for the median
                    hand = {"mean": mu, "std": sd, "min": xs[0], "max": xs[-1], "median": (xs[(len(xs) - 1) // 2] + xs[len(xs) // 2]) / 2, "sos": w * sum(v * v for v in x) / len(x), "sosdiff": x[-1] ** 2 - x[0] ** 2}  # dict literal: statistics by hand
                    tri = [xx[i] ** 2 - xx[i - 1] * xx[i + 1] for i in range(1, len(xx) - 1) if None not in (xx[i - 1], xx[i], xx[i + 1])]  # list comprehension: the Teager-Kaiser operator at interior samples with both neighbours
                    if tri:  # if: at least one triple
                        hand["tk"] = sum(tri) / len(tri)  # its mean by hand
                    if sd > 0:  # if: a spread exists
                        hand["skew"] = sum(((v - mu) / sd) ** 3 for v in x) / len(x)  # skewness by hand
                        hand["kurt"] = sum(((v - mu) / sd) ** 4 for v in x) / len(x)  # kurtosis by hand
                    for st, hv in hand.items():  # for loop over the statistics
                        col = f"{b}__{st}"  # the window column
                        if col in A_.columns:  # if: written at this length
                            ev = float(r[col])  # the published value (four decimals)
                            n_cmp += 1  # one comparison
                            n_bad += int(not (abs(ev - round(hv, 4)) <= max(1e-3, 1e-7 * abs(hv))))  # agree to the rounding (relative for large sums of squares)
    NUM["window_hand_comparisons"] = n_cmp  # kept
    check("Window statistics recomputed by hand (all ten: mean, std, median, min, max, skew, kurt, SoS, SoS Diff, TK) on 40 windows x 6 columns per receiver and length", "equal to the four-decimal rounding", f"{n_bad} disagreements of {n_cmp}", n_bad == 0)  # the test


def main_registered() -> None:  # function: the registered set: part 2 and part 3 from start to finish
    t_start = time.time()  # the clock
    n_p, bad = A.protected_ok()  # protected files at the start
    check("Protected files at start", "all unchanged", f"{n_p} compared, {len(bad)} changed {bad}", not bad)  # the check
    gt = pd.read_csv(PROC / "ground_truth" / "session_ground_truth.csv", usecols=GT_COLS).set_index("test_code")  # session folders and capture times, nothing else
    base, persat = build_base(gt)  # the base table
    write_base(base)  # written
    grid = pd.read_csv(PROC / "labels" / "labels_1hz.csv", usecols=KEYS)  # the label grid
    per = base.groupby("receiver").size().to_dict()  # rows per receiver
    check("Rows: the base table against labels/labels_1hz.csv", "22,356 rows, 7,452 per receiver, same keys", f"{len(base)} rows {per}; keys equal: {base[KEYS].reset_index(drop=True).equals(grid.reset_index(drop=True))}", bool(len(base) == len(grid) and base[KEYS].reset_index(drop=True).equals(grid.reset_index(drop=True))))  # the test
    NUM["ms_step_seconds"] = "; ".join(f"{t} {r}: {int(n)}" for (t, r), n in base.groupby(["test_code", "receiver"]).ms_step.sum().items() if n > 0) or "none"  # generator expression: common millisecond steps per pair
    check("Common whole-millisecond steps of the time differences", "reported", NUM["ms_step_seconds"], None)  # reported
    reconcile(base, persat)  # against the observables script
    magnitudes(base)  # magnitudes
    missing_report(base)  # missing shares
    top5(base)  # report-only comparison, main set without the geometry block
    plot_base_quantities(base)  # the figure
    lab = pd.read_csv(PROC / "labels" / "labels_1hz.csv", usecols=KEYS + ["label"])  # labels: decide which windows are kept and give each its decision second's label (index files only)
    wc = window_columns()  # the window dictionary
    wc.to_csv(OUT / "features_window_columns.csv", index=False)  # written
    built, log = {}, []  # window tables per length; files written
    for w in WINDOWS:  # for loop over the window lengths
        t0 = time.time()  # the clock
        built[w] = build_windows(base, lab, w)  # the windows
        log += write_windows(w, *built[w])  # written
        print(f"  windows w = {w}: {len(built[w][0])} rows, {int((wc.w == w).sum())} columns, {time.time() - t0:.0f} s")  # progress line
    window_checks(base, lab, built)  # part 3 checks
    pd.DataFrame([{"file": str(f.relative_to(PROJECT)), "rows": n, "columns": c, "bytes": f.stat().st_size, "sha256": A.sha(f)} for f, n, c in log]).to_csv(CHECK_DIR / "windows_window_files.csv", index=False)  # list comprehension: every window file's size, rows and hash
    n_p, bad = A.protected_ok()  # protected files at the end
    check("Protected files at end", "all unchanged", f"{n_p} compared, {len(bad)} changed {bad}", not bad)  # the check
    pd.DataFrame(CHECKS).to_csv(CHECK_DIR / "base_checks.csv", index=False)  # the checks
    pd.DataFrame([{"key": k, "value": v} for k, v in NUM.items()]).to_csv(CHECK_DIR / "base_numbers.csv", index=False)  # list comprehension: the numbers
    print(f"done in {time.time() - t_start:.0f} s")  # run time


# ---------------------------------------------------------------- (rule L5): the final window set
OUTB = PROC / "features_final"  # the final set's window folder ; the registered folders under features/ are never written by it
STOP_SECS: set = set()  # set: the Stave stop's seconds (filled by main_final from labels/labels_1hz_final.csv; read by same_cells)


def write_gz_6sf(df: pd.DataFrame, path: Path) -> None:  # function: the observables script's gzip writer with every float written at 6 significant figures ("%.6g")
    import gzip  # import inside the function: gzip with a fixed header, kept local so no earlier line of this file moves
    with open(path, "wb") as fh, gzip.GzipFile(filename="", mode="wb", fileobj=fh, mtime=0, compresslevel=6) as gz:  # with-block over two context managers: no name or time in the header (byte-stable)
        gz.write(df.to_csv(index=False, lineterminator="\n", float_format="%.6g").encode("utf-8"))  # float_format="%.6g": six significant figures; integers and text as they are


def stave_fill(base: pd.DataFrame, gt: pd.DataFrame, stop: dict) -> int:  # function: the orbit-free block of the Stave stop's seconds, filled in place; returns the cells filled
    committed = dict(D.SITE_START)  # the evidence script's constant as committed
    if committed != {"2.6.5": "12:34:57"}:  # if: not the constant the committed base was built with
        raise SystemExit(f"STOP: the evidence script SITE_START is {committed}")  # stop
    first = min(min(s) for s in stop.values())  # generator in min(): the stop's first second over the receivers
    D.SITE_START = {"2.6.5": pd.Timestamp(first, unit="s", tz="UTC").strftime("%H:%M:%S")}  # in-memory override: 2.6.5 read from the end of the first drive (stave_base's method)
    lam, n = wavelengths(), 0  # tuple assignment: carrier wavelengths; cells filled so far
    try:  # try/finally: the constant always comes back
        for rx in RXS:  # for loop over the receivers
            o, (a, b) = read_full(gt, "2.6.5", rx)  # observations from the stop's start (the feature script's reader, the evidence script's span)
            o = per_satellite(o, lam)  # the feature script's per-satellite quantities
            g = base[(base.test_code == "2.6.5") & (base.receiver == rx)]  # the pair's grid rows
            blk = free_block(o, g.sec[(g.sec >= a) & (g.sec < b)].to_numpy())  # the feature script's orbit-free block over the whole extended span (its differences run across grid rows)
            blk = blk[blk.sec.isin(stop[rx])].reset_index(drop=True)  # only the stop's seconds are kept
            rows = g.index[g.sec.isin(stop[rx]).to_numpy()]  # their rows in the base table
            cols = [c for c in blk.columns if c != "sec"]  # list comprehension: the orbit-free and geometry-block columns free_block makes
            if not (base.loc[rows, "sec"].to_numpy() == blk.sec.to_numpy()).all() or len(rows) != len(stop[rx]):  # if: the rows do not line up second for second
                raise SystemExit(f"STOP: Stave rows of {rx} do not line up")  # stop
            if base.loc[rows, [c for c in base.columns if c not in KEYS]].notna().any().any():  # if: a stop row already holds a value (it lay outside the committed span, so none may)
                raise SystemExit(f"STOP: a Stave row of {rx} was not empty")  # stop
            base.loc[rows, cols] = blk[cols].to_numpy()  # filled; the self and known blocks stay missing (no reference position, self-solver not run there)
            n += int(blk[cols].notna().sum().sum())  # cells holding a value
    finally:  # always
        D.SITE_START = committed  # restored
    return n  # the count


def same_cells(w: int, rx: str, pt: str) -> tuple:  # function: one final part against the registered part, every cell of the registered rows: (cells, cells that disagree, keys and columns equal, new rows, new rows inside the Stave stop)
    f = pd.read_csv(OUTB / f"w{w}" / f"{rx}_{pt}.csv.gz", float_precision="round_trip")  # the final file as written (round_trip: exact parsing)
    r = pd.read_csv(WIN_DIR(w) / f"{rx}_{pt}.csv.gz", float_precision="round_trip")  # the registered file
    inr = f[KEYS].merge(r[KEYS].assign(_in=1), on=KEYS, how="left")._in.notna().to_numpy()  # final rows that the registered file has (the Stave windows are new)
    new = f[~inr]  # the new rows
    n_new, n_new_stop = len(new), int(((new.test_code == "2.6.5") & new.sec.isin(STOP_SECS)).sum())  # tuple assignment: how many, and how many end inside the Stave stop
    f = f[inr].reset_index(drop=True)  # the registered rows only, in the final file's order
    if list(f.columns) != list(r.columns) or not f[KEYS].equals(r[KEYS]):  # if: the files do not line up
        return 0, -1, False, n_new, n_new_stop  # nothing compared
    F, R = f.drop(columns=KEYS).to_numpy(dtype=float), r.drop(columns=KEYS).to_numpy(dtype=float)  # tuple assignment: the values
    with np.errstate(divide="ignore", invalid="ignore"):  # context manager: log10(0) gives -inf without a warning (replaced just below)
        e = np.where(F != 0, np.floor(np.log10(np.abs(F))), 0.0)  # the decimal exponent of each final value
    tol = 0.5e-4 + 0.5 * 10.0 ** (e - 5) + 1e-12 * np.abs(F)  # half a unit of the 4th decimal plus half a unit of the 6th significant figure (plus float noise): two roundings of one number
    ok = (np.isnan(F) & np.isnan(R)) | (np.abs(F - R) <= tol)  # equal as roundings, or both missing
    return int(F.size), int((~ok).sum()), True, n_new, n_new_stop  # counts


def main_final() -> None:  # function: the final windows: the feature script's windows on labels/labels_1hz_final.csv with the training label in the index, at 6 significant figures, into features_final/
    t_start = time.time()  # the clock
    n_p, bad = A.protected_ok()  # protected files at the start
    check("Protected files at start", "all unchanged", f"{n_p} compared, {len(bad)} changed {bad}", not bad)  # the check
    gt = pd.read_csv(PROC / "ground_truth" / "session_ground_truth.csv", usecols=GT_COLS).set_index("test_code")  # session folders and capture times
    base, _ = build_base(gt)  # the base table, unrounded in memory (as the registered run computes its windows)
    import gzip  # import inside the function: to read the committed base table as text
    committed = gzip.decompress((OUT / "features_base_1hz.csv.gz").read_bytes()).decode("utf-8")  # the committed base table's text
    check("Base table rebuilt = committed features_base_1hz.csv.gz at its 4 decimals (text)", "equal", "", base.round(4).to_csv(index=False, lineterminator="\n") == committed)  # the base did not move
    fr = pd.read_csv(PROC / "labels" / "labels_1hz_final.csv", usecols=KEYS + ["final_rule"], dtype={"final_rule": str}, keep_default_na=False)  # the rule behind every changed second of the final set
    st = fr[(fr.test_code == "2.6.5") & fr.final_rule.str.contains("the Stave stop relabel", regex=False)]  # the Stave stop's seconds, as the label script's final run moved them
    stop = {rx: set(st[st.receiver == rx].sec.astype(int)) for rx in RXS}  # dict comprehension: receiver -> its stop seconds
    STOP_SECS.update(*stop.values())  # set.update with unpacking: every stop second, for same_cells
    n_fill = stave_fill(base, gt, stop)  # the orbit-free block of the stop filled in place
    check("Stave stop: orbit-free block filled from the feature script's own functions, 2.6.5 read from the stop's start; self and known blocks left missing", "reported", f"{', '.join(f'{rx} {len(s)} s' for rx, s in stop.items())}; {n_fill} cells filled", None)  # reported
    lab = pd.read_csv(PROC / "labels" / "labels_1hz_final.csv", usecols=KEYS + ["label"])  # the final labels (scoring)
    tr = pd.read_csv(PROC / "labels" / "train_labels_1hz.csv", usecols=KEYS + ["train_label"])  # the strict training labels
    check("Keys: final labels and training labels = the base table's grid", "equal", "", bool(lab[KEYS].equals(base[KEYS].reset_index(drop=True)) and tr[KEYS].equals(lab[KEYS])))  # same grid, same order
    built, log = {}, []  # window tables per length; files written
    for w in WINDOWS:  # for loop over the window lengths
        idx, parts = build_windows(base, lab, w)  # the feature script's windows on the final labels
        idx = idx.merge(tr, on=KEYS, how="left", validate="one_to_one")  # the decision second's training label beside its scoring label
        built[w] = (idx, parts)  # kept for the checks
        d = OUTB / f"w{w}"  # the folder
        d.mkdir(parents=True, exist_ok=True)  # created
        for rx in RXS:  # for loop over the receivers
            f = d / f"{rx}_index.csv.gz"  # the index: keys, final label, scoring mask, training label
            write_gz_6sf(idx[idx.receiver == rx], f)  # written (no float column in it)
            log.append((f, int((idx.receiver == rx).sum()), idx.shape[1]))  # logged
            for pt in PARTS:  # for loop over the parts
                f = d / f"{rx}_{pt}.csv.gz"  # one part
                write_gz_6sf(parts[rx][pt], f)  # six significant figures, from the unrounded statistics
                log.append((f, len(parts[rx][pt]), parts[rx][pt].shape[1]))  # logged
        print(f"  final windows w = {w}: {len(idx)} rows")  # progress line
    window_checks(base, lab, built)  # the feature script's part 3 checks on the final set
    tl = all(bool((built[w][0].merge(tr, on=KEYS, suffixes=("", "_t")).pipe(lambda m: m.train_label == m.train_label_t)).all()) for w in WINDOWS)  # generator in all(): each window's training label is its decision second's
    check("Index files: the training label of each window is its decision second's (labels/train_labels_1hz.csv)", "true", "", tl)  # the check
    n_cells, n_bad, aligned = 0, 0, True  # every cell of every part against the registered file
    n_rows_new, n_rows_new_stop = 0, 0  # rows the registered files do not have, and those ending in the Stave stop
    for w in WINDOWS:  # for loop over the lengths
        for rx in RXS:  # for loop over the receivers
            for pt in PARTS:  # for loop over the parts
                c, b, ok, nn, ns = same_cells(w, rx, pt)  # one file pair (and its new rows)
                n_cells, n_bad, aligned = n_cells + c, n_bad + max(b, 0), aligned and ok  # tuple assignment: running totals
                n_rows_new, n_rows_new_stop = n_rows_new + nn, n_rows_new_stop + ns  # tuple assignment: running totals of the new rows
    check("Every feature cell of the final files equals the registered one to its 4 decimals (two roundings of one number)", "0 disagreements, files aligned", f"{n_cells} cells, {n_bad} disagree, aligned {aligned}", n_bad == 0 and aligned)  # the L5 proof
    check("rows beyond the registered files are the Stave windows only (2.6.5, ending inside the stop)", "all", f"{n_rows_new} new rows over the four parts and five lengths; {n_rows_new_stop} inside the stop", n_rows_new == n_rows_new_stop and n_rows_new > 0)  # the decision
    pd.DataFrame([{"file": str(f.relative_to(PROJECT)), "rows": n, "columns": c, "bytes": f.stat().st_size, "sha256": A.sha(f)} for f, n, c in log]).to_csv(OUTB / "windows_window_files_final.csv", index=False)  # list comprehension: every final window file's size, rows and hash
    n_p, bad = A.protected_ok()  # protected files at the end
    check("Protected files at end", "all unchanged", f"{n_p} compared, {len(bad)} changed {bad}", not bad)  # the check
    pd.DataFrame(CHECKS).to_csv(CHECK_DIR / "windows_checks_final.csv", index=False)  # the final run's checks
    print(f"done in {time.time() - t_start:.0f} s")  # run time


def main(*argv: str) -> None:  # function: runs the label set the command line names (the registered set by default; *argv collects the arguments)
    if list(argv) in ([], ['--labels', 'registered', '--precision', '4dp']):  # if: the registered set, the default, unchanged
        main_registered()  # the registered run
    elif list(argv) == ['--labels', 'final', '--precision', '6sf']:  # elif: the final set
        main_final()  # the final run, under new names
    else:  # anything else
        raise SystemExit('usage: 4_features/03_window_features.py [--labels registered --precision 4dp | --labels final --precision 6sf]')  # stop with the usage


if __name__ == "__main__":  # if: true only when the file is run directly
    main(*sys.argv[1:])  # the set named on the command line (main_registered() by default)
    exit_if_failed(FAILED, "4_features/03_window_features.py")  # the helpers module: status 1 when any self-check failed, so a chained run stops
