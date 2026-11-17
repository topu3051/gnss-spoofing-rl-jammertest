#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
The pre-registered checks of the logistic regression baseline that need their own runs
(C1, C2, C3), and the demonstration behind the warning filter of the regression script. C4a, C4b, C5 and C6 run inside
5_baselines/01_logistic_regression.py; this script reads its published tables and must be run after it.

C1 Sealed: 2.6.12 is never needed. The primary, C5 and their uncompressed sensitivities are rebuilt in memory through a reader that (i) deletes every 2.6.12 row
   before the regression script's own reader sees the table and (ii) moves every 2.6.12 feature value by 10^6 (missing values to 10^6) and flips its
   labels; both rebuilds must equal the published tables byte for byte. Controls: the same corruption, and a gentler shift of
   2.6.4's feature values by 1 with labels kept, applied to 2.6.4 must leave the compressed primary and C5 able to finish and
   change their output, so the test can see a change; a stop of the uncompressed sensitivity under either control is reported. The 09 source must read tables through its one reader only.
C2 Fold hygiene (compressed primary pipeline): for every receiver and fold, the held-out session's feature values moved by 10^6 (missing to 10^6) leave every
   fitted object (dropped columns, imputation medians, scaler, raw IQR, Pearson-kept columns, MI selection, coefficients,
   intercept) bitwise unchanged; control: moving one training session changes them. The training rows of every fit hold no
   held-out or 2.6.12 window.
C3 Trivial policies: never-alarm and always-alarm through the regression script's metric code against counts made by hand from the window index:
   recall 0 and 1 (overall and per class), clean windows in alarm 0 and all, false-alarm episodes 0 and the hand count of alarm
   runs touching a clean window, precision of always-alarm attack / (attack + clean) where the scoring module reports it, every interval missed
   under never-alarm, and under always-alarm detected at the first window inside it.
N  The three "... encountered in matmul" messages appear on ordinary finite unit-scale products in this environment, and every
   published fit's decision values agree with the elementwise recomputation within 1e-9.

Reads   the regression script's published tables in Data/processed/logistic_regression_registered/, and what the regression script reads
Writes  Data/processed/checks/logistic_regression_tests.csv
"""

from __future__ import annotations  # future import: type hints kept as text (a convention across the project's scripts)

import gzip  # standard-library gzip, to read the published compressed tables
import importlib.util  # standard-library module loader, for modules whose names start with a digit
import sys  # standard-library system access: sys.modules and the exit status
import warnings  # standard-library warnings: the demonstration records them
from pathlib import Path  # Path objects for building file paths

import numpy as np  # NumPy: arrays
import pandas as pd  # pandas: tables

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/


def load(name: str, filename: str):  # function: loads codes/<filename> as a module (names start with a digit)
    spec = importlib.util.spec_from_file_location(name, PROJECT / "codes" / filename)  # where to load from
    mod = importlib.util.module_from_spec(spec)  # an empty module object
    sys.modules[name] = mod  # registered, so the module's own imports resolve
    spec.loader.exec_module(mod)  # runs the file's top level (its main() is guarded and not run)
    return mod  # the module


M = load("logistic_regression_for_tests", "5_baselines/01_logistic_regression.py")  # the regression script: the pipeline, its reader and its metric code
OUT, CHECK_DIR = M.OUT, M.CHECK_DIR  # its output and check folders
RESULTS: list = []  # list: one row per test, written to logistic_regression_tests.csv


def record(test: str, expected: str, found, ok: bool) -> None:  # function: records one test and prints it
    RESULTS.append({"test": test, "expected": expected, "found": found, "ok": bool(ok)})  # append a dict row
    print(f"  [{'ok' if ok else 'FAIL'}] {test}: expected {expected}; found {found}")  # progress line


# ---------------------------------------------------------------- C1 sealed

CONFIGS = (("primary", "primary"), ("geometry", "c5_geometry"), ("primary_uncompressed", "primary_uncompressed"), ("geometry_uncompressed", "c5_geometry_uncompressed"))  # tuple of pairs: configuration -> file suffix, as the regression script's main writes them
TABLES = ("metrics", "latency", "pooled", "coefficients", "selection")  # the plain tables written per configuration


def files_text(outs: dict) -> dict:  # function: the text of every file the regression script's main writes from the four configurations, built as main builds it (a stopped configuration's files are absent)
    out = {}  # file name -> text
    for cfg, suffix in CONFIGS:  # for loop over the configurations
        if cfg not in outs:  # if: this configuration stopped
            continue  # its files stay absent
        out[f"logistic_regression_predictions_{suffix}.csv.gz"] = outs[cfg]["predictions"].to_csv(index=False, lineterminator="\n")  # held-out probabilities (compared decompressed)
        for k in TABLES:  # for loop over the plain tables
            out[f"logistic_regression_{k}_{suffix}.csv"] = outs[cfg][k].to_csv(index=False, lineterminator="\n")  # the table
    for k, f in (("diag_extreme", "logistic_regression_diag_extreme_columns.csv"), ("diag_lbfgs", "logistic_regression_diag_lbfgs_vs_newton_cg.csv")) if "primary" in outs and "primary_uncompressed" in outs else ():  # for loop over the two diagnostics (only when both primaries completed)
        out[f] = pd.concat([outs["primary"][k], outs["primary_uncompressed"][k]], ignore_index=True).to_csv(index=False, lineterminator="\n")  # both pipelines, as main concatenates them
    return out  # texts


def published_text(names) -> dict:  # function: the published files as text
    out = {}  # result
    for f in names:  # for loop over the file names
        b = (OUT / f).read_bytes()  # the file's bytes
        out[f] = (gzip.decompress(b) if f.endswith(".gz") else b).decode("utf-8")  # conditional expression: decompressed when gzipped
    return out  # text per file


def make_reader(mode: str, session: str, seen: list):  # function: a reader that deletes or corrupts one session's rows before the regression script's reader sees them
    def reader(path, **kw):  # nested function: the replacement for pd.read_csv
        df = pd.read_csv(path, **kw)  # the real read
        seen.append(Path(path).name)  # remembered, to show which tables passed through
        if "test_code" not in df.columns:  # if: a table without sessions
            return df  # unchanged
        m = (df.test_code == session).to_numpy()  # that session's rows
        if mode == "drop":  # if: deletion
            return df[~m].reset_index(drop=True)  # rows removed
        shift = 1.0 if mode == "shift" else 1e6  # conditional expression: the gentle control moves values by 1, the corruption by 10^6
        num = [c for c in df.columns if c not in ("test_code", "receiver", "sec", "label", "scored") and pd.api.types.is_numeric_dtype(df[c])]  # list comprehension: feature columns
        if num:  # if: a feature table
            df[num] = df[num].astype(float)  # every feature column as float, so moved values fit the column type
            v = df.loc[m, num].to_numpy(dtype=float)  # the session's values
            df.loc[m, num] = np.where(np.isnan(v), v, v + shift) if mode == "shift" else np.where(np.isnan(v), 1e6, v + shift)  # conditional expression: shifted by 1 (gaps kept), or moved by 10^6 with gaps set to 10^6
        if "label" in df.columns and mode != "shift":  # if: an index table (the gentle control keeps the labels)
            flip = {"clean": "attack_spoof", "attack_spoof": "clean", "attack_jam_spoof": "clean", "uncertain": "attack_spoof"}  # dict: each label to a different one
            df.loc[m, "label"] = df.loc[m, "label"].map(flip)  # labels flipped
            df.loc[m, "scored"] = df.loc[m, "label"].isin(M.SCORED).astype(int)  # the scoring mask follows the new labels
        return df  # the altered table
    return reader  # the function


def guarded(mode: str, session: str) -> tuple:  # function: the four configurations (primary, C5, both uncompressed) rebuilt in memory through an altering reader, each allowed to stop on its own
    seen: list = []  # tables read
    stopped: dict = {}  # configuration -> reason it stopped
    outs: dict = {}  # configuration -> its tables
    M.READER = make_reader(mode, session, seen)  # the regression script's reader replaced
    try:  # try/finally: the real reader is always restored
        wc, fl = M.load_meta()  # dictionary and lists
        cells = {rx: M.load_cell(M.W, rx) for rx in M.RXS}  # dict comprehension: every receiver's windows
        for c, _ in CONFIGS:  # for loop over the configurations
            try:  # try: one configuration
                outs[c] = M.run_config(wc, fl, c, cells=cells)  # its tables
            except (SystemExit, Exception) as e:  # except: it stopped or failed
                stopped[c] = str(e)[:100]  # the reason
    finally:  # always
        M.READER = pd.read_csv  # restored
        M.FIT_DIAG.clear()  # this rebuild's fit diagnostics are not the published ones
    return files_text(outs), stopped, seen  # texts, stops and the tables read


def c1() -> None:  # function: C1, the sealed-session test
    names = [f"logistic_regression_predictions_{x}.csv.gz" for _, x in CONFIGS] + [f"logistic_regression_{k}_{x}.csv" for _, x in CONFIGS for k in TABLES] + ["logistic_regression_diag_extreme_columns.csv", "logistic_regression_diag_lbfgs_vs_newton_cg.csv"]  # list comprehensions: the 26 files
    pub = published_text(names)  # the published texts
    comp = [f for f in names if "uncompressed" not in f and not f.startswith("logistic_regression_diag_")]  # the compressed primary's and C5's own files
    for mode in ("drop", "corrupt"):  # for loop over the two alterations of 2.6.12
        out, stopped, seen = guarded(mode, M.SEALED)  # rebuilt
        same = not stopped and all(out.get(f) == pub[f] for f in names)  # generator expression: every configuration completed and every file is equal byte for byte
        record(f"C1 sealed ({mode} every 2.6.12 row before the regression script's reader): primary, C5 and both uncompressed rebuilt", f"all {len(names)} files identical", f"identical {same}; stopped {stopped}; tables read {len(seen)} ({', '.join(sorted(set(seen)))})", same)  # the test
    for k, (mode, what) in enumerate((("corrupt", "moved by 10^6, labels flipped"), ("shift", "shifted by +1, labels kept"))):  # for loop over the two controls on 2.6.4
        out, stopped, _ = guarded(mode, "2.6.4")  # rebuilt with 2.6.4 altered
        done = "primary" not in stopped and "geometry" not in stopped  # the compressed primary and C5 completed
        diff = sum(out.get(f) != pub[f] for f in comp) if done else 0  # files of the compressed configurations that differ
        record(f"C1 control {k + 1}: 2.6.4 feature values {what}: the compressed primary and C5 complete and their output changes (a stop of the uncompressed sensitivity is reported, not counted)", "completed; changed", f"completed {done}; compressed files differing {diff} of {len(comp)}; stopped configurations {stopped}", done and diff > 0)  # the test
    src = (PROJECT / "codes" / "5_baselines" / "01_logistic_regression.py").read_text()  # the regression script's source
    n_read = src.count("read_csv")  # occurrences of read_csv
    record("C1 static: the regression script reads tables only through READER (one occurrence of read_csv, the READER default); the scoring module's pooled_intervals reads labels/labels_evidence.csv for the held-out session only", "1", n_read, n_read == 1)  # the test


# ---------------------------------------------------------------- C2 fold hygiene

def fitted(V: pd.DataFrame, y: np.ndarray, tr: np.ndarray, te: np.ndarray, tag: dict) -> tuple:  # function: every fitted object of one fold
    prep = M.prepare(V, tr)  # dropped columns, medians, scaler, pruning
    res = M.fit_fold(prep, y, tr, te, tag=tag)  # selection and model
    return (prep["dropped"], prep["med"], prep["scaler"][0], prep["scaler"][1], prep["iqr_raw"], prep["kept"], res["sel"], res["coef"], np.array([res["intercept"]]))  # tuple: the fitted objects in a fixed order


def moved(X: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:  # function: a copy with the given rows moved by 10^6 (missing values to 10^6)
    Y = X.astype(float)  # a float copy, so moved values fit every column
    v = Y.iloc[rows].to_numpy(dtype=float)  # the rows' values
    Y.iloc[rows] = np.where(np.isnan(v), 1e6, v + 1e6)  # moved
    return Y  # the copy


def equal(a: tuple, b: tuple) -> bool:  # function: two tuples of fitted objects bitwise equal
    return all((x == z) if isinstance(x, list) else np.array_equal(np.asarray(x), np.asarray(z), equal_nan=True) for x, z in zip(a, b))  # generator expression: lists compared as lists, arrays element by element


def c2() -> None:  # function: C2, fold hygiene
    wc, fl = M.load_meta()  # dictionary and lists
    same_all, ctl_all, rows_ok, n = True, True, True, 0  # results
    for rx in M.RXS:  # for loop over the receivers
        idx, X = M.load_cell(M.W, rx)  # windows (2.6.12 dropped)
        y = M.target(idx)  # target
        tc = idx.test_code.to_numpy()  # sessions
        for held, train in M.S.folds():  # for loop over the folds
            tr, te = M.fold_rows(idx, held, train)  # rows
            rows_ok &= not (set(tc[tr]) & {held, M.SEALED}) and len(tr) == int((np.isin(tc, train) & np.isin(idx.label.to_numpy(), M.SCORED)).sum())  # "&=": training rows hold no held-out or 2.6.12 window and are exactly the scored training windows
            cand = M.cand_list(fl, M.W, rx, held)  # the fold's candidate list
            tag = {"config": "c2", "mi_seed": M.SEED, "receiver": rx, "held_out": held, "perm": -1}  # identity for the fit diagnostics
            base = fitted(M.with_indicators(X, cand, wc, M.W, M.BLOCKS), y, tr, te, tag)  # fitted on the true values
            pert = fitted(M.with_indicators(moved(X, np.flatnonzero(tc == held)), cand, wc, M.W, M.BLOCKS), y, tr, te, tag)  # fitted with the held-out values moved (its indicators recomputed from the moved values)
            same_all &= equal(base, pert)  # "&=": all fitted objects unchanged
            if held == "2.6.2":  # if: one control per receiver
                ctl = fitted(M.with_indicators(moved(X, np.flatnonzero(tc == train[0])), cand, wc, M.W, M.BLOCKS), y, tr, te, tag)  # a training session moved
                ctl_all &= not equal(base, ctl)  # "&=": the control must change something
            n += 1  # folds tested
    M.FIT_DIAG.clear()  # these fits are not published
    record("C2 fold hygiene: held-out values moved by 10^6 leave every fitted object bitwise unchanged (dropped columns, medians, scaler, IQR, Pearson-kept, MI-selected, coefficients, intercept)", f"{n} of 15 unchanged", f"unchanged {same_all} ({n} folds)", same_all and n == 15)  # the test
    record("C2 control: moving one training session changes the fitted objects (one fold per receiver)", "changed in 3 of 3", f"changed {ctl_all}", ctl_all)  # the test
    record("C2 rows: every fit's training rows are the scored windows of the training sessions, none held out or from 2.6.12", "true", rows_ok, rows_ok)  # the test


# ---------------------------------------------------------------- C3 trivial policies

def hand_runs(sec: np.ndarray) -> list:  # function: maximal runs of consecutive seconds, by hand (a new run wherever the step is not 1 s)
    cut = np.flatnonzero(np.diff(sec) != 1) + 1  # positions where a new run starts
    return np.split(np.arange(len(sec)), cut)  # index arrays, one per run


def c3() -> None:  # function: C3, never-alarm and always-alarm against hand counts
    ok_all, n, notes = True, 0, []  # results
    for rx in M.RXS:  # for loop over the receivers
        idx, _ = M.load_cell(M.W, rx)  # the index (features unused)
        for held, train in M.S.folds():  # for loop over the folds
            _, te = M.fold_rows(idx, held, train)  # held-out rows in time order
            sec, lab = idx.sec.to_numpy(np.int64)[te], idx.label.to_numpy(object)[te]  # seconds and labels
            n_att, n_cln = int(np.isin(lab, M.ATTACK).sum()), int((lab == "clean").sum())  # hand counts
            ivs = M.SC.pooled_intervals(held)  # the pooled intervals
            for policy, val in (("never", 0.0), ("always", 1.0)):  # for loop over the two trivial policies
                row, lat = M.evaluate_fold(idx, te, np.full(len(te), val), held)  # the regression script's metric code
                exp_rec = 0.0 if policy == "never" else 1.0  # expected recall
                runs = hand_runs(sec)  # alarm runs under always-alarm (every window alarms)
                exp_ep = 0 if policy == "never" else sum(bool((lab[r] == "clean").any()) for r in runs)  # generator expression: runs that touch a clean window
                ok = row["recall"] == exp_rec  # overall recall
                ok &= all(np.isnan(row[k]) or row[k] == exp_rec for k in ("recall_jam_spoof", "recall_spoof"))  # "&=": per-class recall where the class exists
                ok &= row["clean_in_alarm"] == (0 if policy == "never" else n_cln)  # clean windows in alarm
                ok &= (row["fa_episodes"] is None) if n_cln == 0 else (row["fa_episodes"] == exp_ep)  # conditional expression: episodes, or no figure without clean windows
                if row["precision_reported"] and policy == "always":  # if: the scoring module reports precision here
                    ok &= abs(row["precision"] - n_att / (n_att + n_cln)) < 1e-12  # precision of always-alarm by hand
                prev = None  # end of the previous interval
                for L, (a, b) in zip(lat, ivs):  # for loop over the intervals
                    inside = sec[(sec >= a) & (sec <= b)]  # windows inside, by hand
                    if policy == "never":  # if: never-alarm
                        ok &= (not L["detected"]) and np.isnan(L["latency_s"])  # missed, latency undefined
                    else:  # else: always-alarm
                        ok &= L["detected"] and L["latency_s"] == int(inside.min() - a)  # detected at the first window inside
                        early = sum(1 for r in runs if sec[r[-1]] < a and (prev is None or sec[r[0]] > prev))  # generator expression: runs ending before the onset and starting after the previous interval, by hand
                        ok &= L["early_alarm_runs"] == early  # early alarm runs
                    prev = b  # remembered
                if not ok:  # if: a mismatch
                    notes.append(f"{rx} {held} {policy}")  # named
                ok_all &= bool(ok)  # "&=": all agree
                n += 1  # cases checked
    record("C3 trivial policies: never-alarm and always-alarm reproduce the hand counts (recall per class, clean windows in alarm, episodes, precision, latency, early alarm runs)", "30 of 30 cases", f"{n} cases, all agree {ok_all} {notes}", ok_all and n == 30)  # the test


# ---------------------------------------------------------------- N: the warning filter

def numeric() -> None:  # function: the environment's spurious matmul warnings, and the published decision-value agreement
    rng = np.random.default_rng(M.SEED)  # fixed seed
    A, w = rng.normal(size=(3000, 36)), rng.normal(size=36)  # unit-scale finite data
    with warnings.catch_warnings(record=True) as caught:  # with: records the warnings
        warnings.simplefilter("always")  # all recorded
        r = A @ w  # an ordinary matrix product
    msgs = sorted({str(x.message) for x in caught})  # set comprehension: the distinct messages
    record("N1 the matmul messages appear on an ordinary finite unit-scale product in this environment (NumPy 2.2.6, Accelerate)", "the three filtered messages, finite result", f"{msgs}; finite {bool(np.isfinite(r).all())}", set(msgs) <= set(M.MATMUL_MSGS) and bool(np.isfinite(r).all()))  # the test (an empty set also passes: then nothing needed filtering)
    fd = pd.read_csv(OUT / "logistic_regression_fit_diagnostics.csv")  # the published fit diagnostics
    mx = float(fd.decision_max_rel_diff.max())  # largest disagreement
    record("N2 every published fit: sklearn's decision values against the elementwise recomputation", f"<= {M.DEC_TOL}", f"{len(fd)} fits, largest {mx:.3g}", mx <= M.DEC_TOL)  # the test


def main() -> None:  # function: runs every test and exits with status 1 on any failure
    c3()  # cheapest first
    numeric()  # N
    c2()  # C2
    c1()  # C1 last (three rebuilds)
    pd.DataFrame(RESULTS).to_csv(CHECK_DIR / "logistic_regression_tests.csv", index=False, lineterminator="\n")  # the result table
    bad = [r["test"] for r in RESULTS if not r["ok"]]  # list comprehension: failures
    print(f"{len(RESULTS) - len(bad)} of {len(RESULTS)} tests pass")  # summary line
    if bad:  # if: any failure
        sys.exit(1)  # exit status 1


if __name__ == "__main__":  # guard: runs main() only when the file is executed, not when imported
    main()  # entry point
