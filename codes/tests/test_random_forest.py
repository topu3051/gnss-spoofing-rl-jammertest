#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
The pre-registered checks of the random forest that need their own runs (C1, C2, C3)
and the tests of the importance code (I1, I2). C4a, C4b, C5 and C6 run inside 5_baselines/02_random_forest.py; this script reads its
published tables and must be run after it.

C1 Sealed: 2.6.12 is never needed. The primary, C5 and the primary's importance (the forest script's core(): 13 files) are rebuilt in memory
   through a reader that (i) deletes every 2.6.12 row before the regression script's reader sees the table and (ii) moves every 2.6.12 feature value
   by 10^6 (missing values to 10^6) and flips its labels (the regression tests' make_reader); both rebuilds must equal the published files byte for
   byte. Controls: the same corruption, and a shift of 2.6.4's feature values by 1 with labels kept, applied to 2.6.4 must change
   the output, so the test can see a change. Static: the forest script reads no table except through the regression script's reader (no read_csv in its source).
C2 Fold hygiene: for every receiver and fold, the held-out session's feature values moved by 10^6 (missing to 10^6) leave every
   fitted object (dropped columns, imputation medians, scaler, raw IQR, Pearson-kept columns, MI selection, and every array of
   every one of the 500 trees) bitwise unchanged; control: moving one training session changes them (one fold per receiver).
C3 Trivial policies: never-alarm and always-alarm through the shared metric code (the regression tests' test of the regression script's evaluate_fold, which 10 uses
   unchanged) against counts made by hand: 30 of 30 cases.
I1 Permutation equivalence: permuting a family's raw values among the scored held-out windows before imputation, then applying
   the fold's fitted steps (median imputation, robust scaling, compression) and the selection, equals bitwise the permutation of
   the selected columns that the forest script's importance code performs (every family of one fold per receiver, three repeats).
I2 Importance machinery: a constant predictor gives a drop of exactly 0 in every metric of every group; a predictor that reads one
   selected column gives a non-zero drop only to that column's family (and to that column and its secondary group).

Reads   the forest script's published tables in Data/processed/random_forest_registered/, and what the forest script reads
Writes  Data/processed/checks/random_forest_tests.csv
"""

from __future__ import annotations  # future import: type hints kept as text (a convention across the project's scripts)

import gzip  # standard-library gzip, to read the published compressed tables
import importlib.util  # standard-library module loader, for modules whose names start with a digit
import sys  # standard-library system access: sys.modules and the exit status
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


R = load("random_forest_for_tests", "5_baselines/02_random_forest.py")  # the forest script: the forest pipeline and its importance code (with its own copy of 09 as R.M)
B = load("logistic_regression_tests_for_forest_tests", "tests/test_logistic_regression.py")  # 09b: its altering reader, its moved-copy helper and its trivial-policy test
M, S = R.M, R.S  # the 09 and the screen script instances that 10 uses (the reader is replaced on this 09)
RESULTS: list = []  # list: one row per test, written to random_forest_tests.csv


def record(test: str, expected: str, found, ok: bool) -> None:  # function: records one test and prints it
    RESULTS.append({"test": test, "expected": expected, "found": found, "ok": bool(ok)})  # append a dict row
    print(f"  [{'ok' if ok else 'FAIL'}] {test}: expected {expected}; found {found}")  # progress line


# ---------------------------------------------------------------- C1 sealed

def published_text(names) -> dict:  # function: the published files as text
    out = {}  # result
    for f in names:  # for loop over the file names
        b = (R.OUT / f).read_bytes()  # the file's bytes
        out[f] = (gzip.decompress(b) if f.endswith(".gz") else b).decode("utf-8")  # conditional expression: decompressed when gzipped
    return out  # text per file


def guarded(mode: str, session: str) -> tuple:  # function: the forest script's core rebuilt in memory through an altering reader
    seen: list = []  # tables read
    M.READER = B.make_reader(mode, session, seen)  # the regression script's reader (the one 10 uses) replaced by the regression tests' altering reader
    try:  # try/finally: the real reader is always restored
        wc, fl = M.load_meta()  # dictionary and lists
        cells = {rx: M.load_cell(M.W, rx) for rx in M.RXS}  # dict comprehension: every receiver's windows
        try:  # try: the rebuild may stop under a corruption
            co = R.core(cells, wc, fl)  # primary, C5 and the primary's importance
            texts, stopped = {k: v.to_csv(index=False, lineterminator="\n") for k, v in co["files"].items()}, ""  # dict comprehension: the text each file would have
        except (SystemExit, Exception) as e:  # except: it stopped or failed
            texts, stopped = {}, str(e)[:120]  # nothing, and the reason
    finally:  # always
        M.READER = pd.read_csv  # restored
        M.FIT_DIAG.clear()  # nothing of the regression script's diagnostics is kept
    return texts, stopped, seen  # texts, stop reason and the tables read


def c1() -> None:  # function: C1, the sealed-session test
    names = None  # the file names, taken from the first rebuild
    for mode in ("drop", "corrupt"):  # for loop over the two alterations of 2.6.12
        out, stopped, seen = guarded(mode, M.SEALED)  # rebuilt
        names = names or sorted(out)  # conditional "or": the files core() writes
        pub = published_text(names)  # the published texts
        same = not stopped and len(out) == 13 and all(out[f] == pub[f] for f in names)  # generator expression: completed and every file equal byte for byte
        record(f"C1 sealed ({mode} every 2.6.12 row before the regression script's reader): primary, C5 and the primary's importance rebuilt", "all 13 files identical", f"identical {same}; stopped '{stopped}'; tables read {len(seen)} ({', '.join(sorted(set(seen)))})", same)  # the test
    pub = published_text(names)  # the published texts
    for k, (mode, what) in enumerate((("corrupt", "moved by 10^6, labels flipped"), ("shift", "shifted by +1, labels kept"))):  # for loop over the two controls on 2.6.4
        out, stopped, _ = guarded(mode, "2.6.4")  # rebuilt with 2.6.4 altered
        diff = sum(out[f] != pub[f] for f in names) if out else 0  # conditional expression: files that differ
        record(f"C1 control {k + 1}: 2.6.4 feature values {what}: the rebuild completes and its output changes", "completed; changed", f"completed {not stopped}; files differing {diff} of {len(names)}; stopped '{stopped}'", (not stopped) and diff > 0)  # the test
    src = (PROJECT / "codes" / "5_baselines" / "02_random_forest.py").read_text()  # the forest script's source
    record("C1 static: the forest script reads tables only through the regression script's reader (no read_csv in its source); the scoring module's pooled_intervals reads labels/labels_evidence.csv for the held-out session only", "0", src.count("read_csv"), src.count("read_csv") == 0)  # the test


# ---------------------------------------------------------------- C2 fold hygiene

def fitted(V: pd.DataFrame, y: np.ndarray, tr: np.ndarray) -> tuple:  # function: every fitted object of one fold, the forest's trees included
    prep = M.prepare(V, tr)  # dropped columns, medians, scaler, pruning
    sel = M.mi_top(prep, y, tr, R.SEED)  # MI selection
    m = R.fit_forest(prep["Z"][tr][:, sel], y[tr], R.SEED)  # the forest
    trees = [getattr(e.tree_, f) for e in m.estimators_ for f in ("feature", "threshold", "value", "children_left", "children_right", "n_node_samples")]  # list comprehension: every array of every tree
    return (prep["dropped"], prep["med"], prep["scaler"][0], prep["scaler"][1], prep["iqr_raw"], prep["kept"], sel, *trees)  # tuple with unpacking: the fitted objects in a fixed order


def c2() -> None:  # function: C2, fold hygiene
    wc, fl = M.load_meta()  # dictionary and lists
    same_all, ctl_all, rows_ok, n = True, True, True, 0  # results
    for rx in M.RXS:  # for loop over the receivers
        idx, X = M.load_cell(M.W, rx)  # windows (2.6.12 dropped)
        y = M.target(idx)  # target
        tc = idx.test_code.to_numpy()  # sessions
        for held, train in S.folds():  # for loop over the folds
            tr, te = M.fold_rows(idx, held, train)  # rows
            rows_ok &= not (set(tc[tr]) & {held, M.SEALED}) and len(tr) == int((np.isin(tc, train) & np.isin(idx.label.to_numpy(), M.SCORED)).sum())  # "&=": training rows hold no held-out or 2.6.12 window and are exactly the scored training windows
            base = fitted(R.candidates(X, wc, fl, rx, held, "primary"), y, tr)  # fitted on the true values
            pert = fitted(R.candidates(B.moved(X, np.flatnonzero(tc == held)), wc, fl, rx, held, "primary"), y, tr)  # fitted with the held-out values moved (its indicator recomputed from the moved values)
            same_all &= B.equal(base, pert)  # "&=": every fitted object unchanged
            if held == "2.6.2":  # if: one control per receiver
                ctl = fitted(R.candidates(B.moved(X, np.flatnonzero(tc == train[0])), wc, fl, rx, held, "primary"), y, tr)  # a training session moved
                ctl_all &= not B.equal(base, ctl)  # "&=": the control must change something
            n += 1  # folds tested
    record("C2 fold hygiene: held-out values moved by 10^6 leave every fitted object bitwise unchanged (dropped columns, medians, scaler, IQR, Pearson-kept, MI-selected, every array of all 500 trees)", "15 of 15 unchanged", f"unchanged {same_all} ({n} folds)", same_all and n == 15)  # the test
    record("C2 control: moving one training session changes the fitted objects (one fold per receiver)", "changed in 3 of 3", f"changed {ctl_all}", ctl_all)  # the test
    record("C2 rows: every fit's training rows are the scored windows of the training sessions, none held out or from 2.6.12", "true", rows_ok, rows_ok)  # the test


# ---------------------------------------------------------------- C3 trivial policies

def c3() -> None:  # function: C3 through the regression tests' test of the shared metric code
    B.RESULTS.clear()  # the regression tests' own result list emptied
    B.c3()  # never-alarm and always-alarm against hand counts (the regression script's evaluate_fold, as 10 calls it)
    r = B.RESULTS[-1]  # its one result row
    record("C3 trivial policies (the regression tests' hand counts through the shared evaluate_fold): " + r["test"].split(": ", 1)[-1], r["expected"], r["found"], r["ok"])  # the test


# ---------------------------------------------------------------- I1 and I2: the importance code

def transform(prep: dict, V: pd.DataFrame, rows: np.ndarray, sel: list) -> np.ndarray:  # function: the fold's fitted column-wise steps applied to raw rows, by an independent route
    A = V.iloc[rows][prep["names"]].to_numpy(dtype=float)  # the columns that had a training value, in prep's order
    A = np.where(np.isnan(A), prep["med"], A)  # median imputation with the training medians
    Z = S.apply_scaler(A, prep["scaler"])  # robust scaling with the training parameters
    Z = np.sign(Z) * np.log1p(np.abs(Z))  # the signed-log compression
    return Z[:, sel]  # the selected columns


def i1_i2() -> None:  # function: the permutation-equivalence test and the machinery test
    wc, fl = M.load_meta()  # dictionary and lists
    fam, xs = R.name_maps(wc)  # the family rule
    ok1, n1, ok2c, ok2s, n2 = True, 0, True, True, 0  # results
    for rx, held in (("SBF_02", "2.6.4"), ("SBF_03", "2.6.2"), ("UBX_01", "2.6.10")):  # for loop over one fold per receiver
        idx, X = M.load_cell(M.W, rx)  # windows
        y = M.target(idx)  # target
        train = [s for s in M.SESSIONS if s != held]  # training sessions
        tr, te = M.fold_rows(idx, held, train)  # rows
        V = R.candidates(X, wc, fl, rx, held, "primary")  # candidates and indicator
        prep = M.prepare(V, tr)  # fitted steps
        sel = M.mi_top(prep, y, tr, R.SEED)  # selection
        names = [prep["names"][j] for j in sel]  # selected names
        lab = idx.label.to_numpy(object)[te]  # held-out labels
        rows = te[np.isin(lab, M.SCORED)]  # scored held-out rows
        Zs = prep["Z"][rows][:, sel]  # the forest script's matrix of the scored held-out windows
        ok1 &= bool(np.array_equal(transform(prep, V, rows, sel), Zs))  # "&=": the independent route reproduces it unpermuted
        for r in range(3):  # for loop over three repeats
            pm = np.random.default_rng(R.IMP_SEED0 + r).permutation(len(rows))  # the forest script's permutation of repeat r
            for f in R.FAMILIES:  # for loop over the families
                cols = [j for j, c in enumerate(names) if fam[c] == f]  # the family's selected positions
                if not cols:  # if: not selected
                    continue  # nothing to permute
                Vp = V.copy()  # a raw copy
                fcols = [c for c in V.columns if c in prep["names"] and fam.get(c) == f]  # every raw column of the family that the fitted steps carry
                Vp.loc[Vp.index[rows], fcols] = V.iloc[rows[pm]][fcols].to_numpy()  # raw values (missing values with them) permuted among the scored held-out rows
                route_a = transform(prep, Vp, rows, sel)  # permute first, then impute, scale, compress, select
                route_b = Zs.copy()  # the forest script's route
                route_b[:, cols] = Zs[pm][:, cols]  # permute the selected columns after the fitted steps
                ok1 &= bool(np.array_equal(route_a, route_b))  # "&=": bitwise equal
                n1 += 1  # comparisons
        labs = lab[np.isin(lab, M.SCORED)]  # scored labels
        groups = {f: [j for j, c in enumerate(names) if fam[c] == f] for f in R.FAMILIES}  # dict comprehension: family groups
        s, _ = R.importance(lambda Z: np.full(len(Z), 0.7), Zs, labs, groups, {"t": "const"}, False)  # a constant predictor
        d = pd.DataFrame(s)  # its rows
        ok2c &= bool((d[d.status == "selected"].drop_mean.fillna(0) == 0).all() and (d[d.status == "selected"].drop_sd.fillna(0) == 0).all())  # "&=": every drop exactly 0
        j0 = 0  # the first selected column (largest MI)
        s, _ = R.importance(lambda Z: 1.0 / (1.0 + np.exp(-Z[:, j0])), Zs, labs, groups, {"t": "one"}, False)  # a predictor that reads only column j0
        d = pd.DataFrame(s)  # its rows
        moved = set(d[(d.status == "selected") & (d.drop_mean.fillna(0) != 0)].group)  # groups with a non-zero mean drop
        nonzero_any = set(d[(d.status == "selected") & ((d.drop_mean.fillna(0) != 0) | (d.drop_sd.fillna(0) != 0))].group)  # groups with any non-zero drop
        ok2s &= nonzero_any <= {fam[names[j0]]} and fam[names[j0]] in moved  # "&=": only that column's family moves, and it does
        n2 += 1  # folds tested
    record("I1 permutation equivalence: raw values permuted before imputation, then the fitted steps, equal the forest script's permutation of the selected columns bitwise (and the unpermuted route reproduces the forest script's matrix)", "all equal", f"equal {ok1} ({n1} family-repeat comparisons in 3 folds)", ok1 and n1 > 0)  # the test
    record("I2 machinery: a constant predictor gives a drop of exactly 0 in every metric of every family", "0 everywhere (3 folds)", f"{ok2c}", ok2c)  # the test
    record("I2 machinery: a predictor reading one selected column moves only that column's family", "only that family (3 folds)", f"{ok2s} ({n2} folds)", ok2s)  # the test


def main() -> None:  # function: runs every test and exits with status 1 on any failure
    c3()  # cheapest first
    i1_i2()  # importance tests
    c2()  # C2
    c1()  # C1 last (four rebuilds)
    M.write_csv(pd.DataFrame(RESULTS), R.CHECK_DIR / "random_forest_tests.csv")  # the result table
    bad = [r["test"] for r in RESULTS if not r["ok"]]  # list comprehension: failures
    print(f"{len(RESULTS) - len(bad)} of {len(RESULTS)} tests pass")  # summary line
    if bad:  # if: any failure
        sys.exit(1)  # exit status 1


if __name__ == "__main__":  # guard: runs main() only when the file is executed, not when imported
    main()  # entry point
