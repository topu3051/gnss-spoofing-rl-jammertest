#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
The adopted pipeline P of the baselines, composed from the main-chain scripts only.

P is the adopted configuration: the final labels (labels/labels_1hz_final.csv) with the strict training
negatives (the training-label script's labels/train_labels_1hz.csv, carried in the features_b index), the orbit-free candidates at 10 s, the screen script's jamming screen
refitted on each fold's training sessions under the training label, the regression script's indicator, imputation, robust scaling, signed-log compression,
Pearson pruning and MI top 36, the logistic regression's logistic regression and the random forest's forest, per receiver, leave-one-session-out over the five position
sessions. 2.6.12 is sealed: the regression script's reader drops it at read and the gates assert it absent.

Moved here unchanged (the MOVED section below), with their comments: the functions of the pipeline library (04_pipeline_library.py, loaded as choices_lib) and of the final-labels library
(05_final_labels_library.py, loaded as final_lib) that P runs, found by following the p command of 06_run_baselines.py. Their syntax
trees equal those of the library functions; this file imports nothing from the tests. final_lib's
functions called choices_lib's as L.<name>; both live here now, so L is bound to this module and those calls are unchanged.
New here (the NEW section): nested_keep(), P's nested scheme with the inner logistic regressions kept and their decision values, and the
two inner fits the screen cannot serve fitted with the screen not applied; build(); transform()
and score(), which turn ANY window of a receiver (scored or not) into the 36 selected columns of an outer fold and score it; the gates.

Gates (all bitwise; any failure stops with exit status 1), written to Data/processed/rl_environment/adopted_pipeline_gates.csv:
  G0  outer predictions = logistic_regression/logistic_regression_predictions_primary.csv.gz and random_forest/random_forest_predictions_primary.csv.gz
  G0n inner out-of-fold predictions = baselines_controls/adopted_nested_predictions.csv.gz on every row it holds; selections = adopted_nested_selection.csv
  G0s outer kept and selected columns = baselines_controls/adopted_selection_kept_columns.csv, random_forest and logistic_regression selection files
  G0w transform() of the held-out and training windows = the fit's own matrix, and its scores = the committed probabilities
Reads   Data/processed/features_final/w10/*, features/features_window_columns.csv, labels/labels_evidence_final.csv (through the scoring module), the committed files above
Writes  (when run) Data/processed/rl_environment/adopted_pipeline_gates.csv and adopted_pipeline_inner_fits.csv
"""
from __future__ import annotations  # future import: type hints kept as text (the project's convention)

import ast  # standard library: this file's own syntax tree, for the no-check-import check
import contextlib  # standard library: redirect_stdout, used by quiet() to swallow printed progress
import importlib.util  # standard library: loads the forest script (its name starts with a digit)
import io  # standard library: in-memory text for quiet()
import sys  # standard library: module registry and exit status
import time  # standard library: wall-clock time of the run (printed only, never written)
import warnings  # standard library: the warnings of a decision-value product are caught and inspected, as 09 does
from pathlib import Path  # standard library: paths

import numpy as np  # numpy: arrays
import pandas as pd  # pandas: tables
from scipy.special import expit  # the logistic function, to check that probability = expit(decision value) bitwise

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
CODES = PROJECT / "codes"  # the project scripts
PROC = PROJECT / "Data" / "processed"  # processed tables
FINAL = PROC / "features_final"  # the final window files
ENV_DIR = PROC / "rl_environment"  # this section's tables


def load_script(name: str, filename: str):  # function: loads codes/<filename> as a module, unchanged (its main() stays behind its guard)
    spec = importlib.util.spec_from_file_location(name, CODES / filename)  # where to load from
    mod = importlib.util.module_from_spec(spec)  # an empty module object
    sys.modules[name] = mod  # registered so the module's own imports resolve
    spec.loader.exec_module(mod)  # runs its top level (constants and functions)
    return mod  # the module


R = load_script("random_forest_for_adopted_pipeline", "5_baselines/02_random_forest.py")  # 10, which loads 09 as its M; the regression script loads the screen script and the scoring module
M, S, SC = R.M, R.S, R.SC  # 09, the screen script and the scoring module exactly as 10 uses them
L = sys.modules[__name__]  # this module: final_lib's L.<name> calls of choices_lib's functions resolve here, unchanged

SEED = M.SEED  # 20260914, the project's seed
RXS, SESSIONS, KEYS = M.RXS, M.SESSIONS, M.KEYS  # receivers, the five folds in the screen script's order, key columns
SEALED, W = M.SEALED, M.W  # "2.6.12" and the window length 10
ATTACK, SCORED = M.ATTACK, M.SCORED  # attack labels and scored labels
BLOCKS = R.BLOCKS  # ("orbitfree",): the orbit-free primary's one missing-indicator block (the random forest's orbit-free set)
S11_INTERVAL = ("2.6.11", 2)  # choices_lib's constant, read by evaluate() only when a run drops that interval; P does not (P_CFG s11 False)
IND_COLS = ["rx_is_SBF_03", "rx_is_UBX_01"]  # choices_lib's receiver-indicator columns, used by fit_unit() only when rx_ind is True; P's is False
FAIL_COLS = ["run_id", "model", "held_out", "inner_held_out", "receivers", "reason"]  # columns of the failed-fit table
P_CFG = {"run_id": "P", "pooled": False, "rx_ind": False, "cand": "refit", "s11": False, "models": ("lr", "rf")}  # final_lib's configuration of the adopted primary

# ======================================================================================================================
# MOVED: the library functions (choices_lib, final_lib) that P runs, copied unchanged
# ======================================================================================================================


def window_dict() -> pd.DataFrame:  # function: the feature script's committed window dictionary at w = 10
    wc = M.read_table(PROC / "features" / "features_window_columns.csv")  # read through the regression script's reader
    return wc[wc.w == W].reset_index(drop=True)  # the w = 10 rows, in the feature script's order


def d1_columns(wc: pd.DataFrame) -> list:  # function: the orbit-free primary's candidate set (no_residual), in the feature script's column order
    return list(wc[wc.no_residual == 1].column)  # 778 columns at w = 10


def combine(cells: dict, rxs: list) -> tuple:  # function: the windows of one or several receivers stacked in RXS order, with each receiver's row offset
    idx = pd.concat([cells[r][0] for r in rxs], ignore_index=True)  # stacked index
    X = pd.concat([cells[r][1] for r in rxs], ignore_index=True)  # stacked values
    off, o = {}, 0  # receiver -> first row; running offset
    for r in rxs:  # in order
        off[r] = o  # its first row
        o += len(cells[r][0])  # next offset
    return idx, X, off  # stacked frames and offsets


def train_rows(idx: pd.DataFrame, train: list) -> np.ndarray:  # function: scored training rows by the training label (the regression script's fold_rows with tlabel), with its leakage checks
    tc = idx.test_code.to_numpy()  # sessions
    if SEALED in set(tc) or SEALED in train:  # sealed check
        raise SystemExit("STOP: 2.6.12 in a fit")  # stop
    return np.flatnonzero(np.isin(tc, train) & np.isin(idx.tlabel.to_numpy(), SCORED))  # ascending rows


def eval_rows(idx: pd.DataFrame, held: str) -> np.ndarray:  # function: every row of the held-out session in time order (the regression script's fold_rows)
    te = np.flatnonzero(idx.test_code.to_numpy() == held)  # its rows
    return te[np.argsort(idx.sec.to_numpy()[te], kind="stable")]  # sorted by time, stable


def screened(idx: pd.DataFrame, X: pd.DataFrame, train: list, cols: list, label_col: str = "tlabel") -> list:  # function: the screen script's screen refitted on the training sessions' windows, floor rule; the orbit-free columns it keeps
    sc = S.screen_cell(idx.assign(label=idx[label_col]), X[cols], train)  # the screen script's screen with the chosen labels in the "label" column
    flag = dict(zip(sc.column, sc.reason.fillna("")))  # column -> reasons (the screen script's "reason" = the floor rule, the screened list of the feature build)
    return [c for c in cols if not flag.get(c)]  # list comprehension: unflagged columns, in the feature script's order (the screen script's lists_and_summary)


def unfittable(idx: pd.DataFrame, train: list, cand: list) -> str:  # function: why a training set cannot be fitted ("" when it can): no clean or no attack window, or a screen that keeps no column
    trl = idx.tlabel.to_numpy()[train_rows(idx, train)]  # training labels
    if not (trl == "clean").any():  # no negative class
        return "training set has no clean window"  # reason
    if not np.isin(trl, ATTACK).any():  # no positive class
        return "training set has no attack window"  # reason
    if not cand:  # the screen script flags every column (fewer than 30 clean windows make every column not assessable)
        return "the screen keeps no column (fewer than 30 training windows of a class)"  # reason
    return ""  # fittable


def fit_unit(idx: pd.DataFrame, X: pd.DataFrame, wc: pd.DataFrame, train: list, evals: dict, cand: list, rx_ind: bool, models: tuple, y=None, keep: bool = False, tag: dict | None = None) -> dict:  # function: one training set, both models, probabilities on each evaluation row set
    tr = train_rows(idx, train)  # training rows
    for k, te in evals.items():  # evaluation sets
        if np.intersect1d(tr, te).size or set(idx.test_code.to_numpy()[te]) & set(train):  # leakage checks (the regression script's)
            raise SystemExit(f"STOP: group leakage in {k}")  # stop
    if y is None:  # true labels
        y = np.isin(idx.tlabel.to_numpy(), ATTACK).astype(int)  # 1 for attack windows (the regression script's target, on the training label)
    V = M.with_indicators(X, cand, wc, W, BLOCKS)  # the regression script: candidates plus the orbit-free missing indicator
    prep = M.prepare(V, tr)  # the regression script: drop, impute, scale, compress, Pearson-prune on the training rows
    sel = M.mi_top(prep, y, tr, SEED)  # the regression script: MI top 36 with the project seed
    Z = prep["Z"][:, sel]  # the selected columns
    names = [prep["names"][j] for j in sel]  # their names
    if rx_ind:  # pooled training with the receiver indicator: forced in after MI, unscaled 0/1
        Z = np.column_stack([Z] + [(idx.receiver.to_numpy() == c.replace("rx_is_", "")).astype(float) for c in IND_COLS])  # two 0/1 columns
        names = names + IND_COLS  # their names
    out = {"tr": tr, "y": y, "prep": prep, "sel": sel, "names": names, "n_cand": V.shape[1], "Z": Z, "p": {}, "fail": {}}  # the fit's record
    for mdl in models:  # "lr" and/or "rf"
        if mdl == "lr":  # the logistic regression's logistic regression
            try:  # try: 09 stops on any warning; here a stop is recorded as a failed fit
                m, _ = M.fit_lr(Z[tr], y[tr], M.C_PRIMARY, tag, strict=prep["compressed"])  # the regression script's fit with every warning rule
                pred = lambda Ze, m=m: M.predict(m, Ze)[0]  # lambda with a default argument: the regression script's checked prediction of this model
                out["fail"]["lr"] = ""  # no failure
            except SystemExit as e:  # the regression script's stop
                pred, m = None, None  # no model
                out["fail"]["lr"] = str(e)  # the reason
            M.FIT_DIAG.clear()  # the regression script's diagnostics list is not needed here
        else:  # the random forest's forest
            m = R.fit_forest(Z[tr], y[tr], SEED)  # the forest script's forest (fitted with 8 jobs, predicting serially)
            pred = lambda Ze, m=m: R.proba(m, Ze)  # lambda: the forest script's probability of attack
            out["fail"]["rf"] = ""  # no failure
        out["p"][mdl] = {k: (pred(Z[te]) if pred else np.full(len(te), np.nan)) for k, te in evals.items()}  # dict comprehension: probabilities per evaluation set
        if keep:  # kept for importance and C4b
            out[f"model_{mdl}"] = m  # the model
    return out  # record


def evaluate(idx_rx: pd.DataFrame, te: np.ndarray, p: np.ndarray, held: str, ident: dict, drop_s11: bool, thr: float = 0.5) -> tuple:  # function: the regression script's evaluate_fold at a threshold; metric row, latency rows, prediction frame
    old = M.THR  # the regression script's threshold constant
    M.THR = thr  # in-memory override (the inner-fold threshold only, 0.5 otherwise)
    try:  # try: the constant always comes back
        row, lat = M.evaluate_fold(idx_rx, te, p, held)  # the regression script's fold evaluation (the scoring module's policy, false alarms and latency)
    finally:  # finally
        M.THR = old  # restored
    if drop_s11:  # S11: the 1.1.27 interval is not an attack interval of that run
        lat = [r for r in lat if not (held == S11_INTERVAL[0] and r["interval"] == S11_INTERVAL[1])]  # list comprehension: its row dropped
    row.update(ident)  # identifiers
    lat = [{**r, **ident} for r in lat]  # list comprehension: identifiers on every latency row
    pr = pd.DataFrame({**ident, "held_out": held, "sec": idx_rx.sec.to_numpy()[te], "label": idx_rx.label.to_numpy()[te], "p": p, "alarm": p >= thr})  # prediction frame
    return row, lat, pr  # all three


def outer(cells: dict, wc: pd.DataFrame, cfg: dict, keep_cells: list = ()) -> tuple:  # function: the outer leave-one-session-out run of one configuration, both models
    cols = d1_columns(wc)  # orbit-free columns
    fl = M.read_table(PROC / "features" / "feature_lists.csv.gz") if cfg["cand"] == "registered" else None  # registered lists (the registered labels gate only)
    rows, lats, prs, sel, kept, fails = [], [], [], [], {}, []  # outputs
    for held, train in S.folds():  # the screen script's five folds
        groups = [RXS] if cfg["pooled"] else [[r] for r in RXS]  # one pooled fit, or one fit per receiver
        for rxs in groups:  # fits
            idx, X, off = combine(cells, rxs)  # training and evaluation frame
            if cfg["cand"] == "registered":  # the random forest's registered no_residual screened list
                cand = M.cand_list(fl, W, rxs[0], held, "no_residual")  # the regression script's reader of feature_lists
            else:  # refitted
                cand = screened(idx, X, train, cols)  # the screen script's screen on the training sessions
            ev = {r: eval_rows(idx, held)[np.isin(eval_rows(idx, held), np.arange(off[r], off[r] + len(cells[r][0])))] for r in rxs}  # dict comprehension: each receiver's held-out rows in the stacked frame
            why = unfittable(idx, train, cand)  # can it be fitted?
            if why:  # not fittable: recorded for both models, not scored
                fails += [{"run_id": cfg["run_id"], "model": mdl, "held_out": held, "inner_held_out": "", "receivers": "+".join(rxs), "reason": why} for mdl in cfg["models"]]  # list comprehension: failure rows
                continue  # next fit
            keep = any((r, held) in keep_cells for r in rxs)  # kept for C4b and importance
            f = fit_unit(idx, X, wc, train, ev, cand, cfg["rx_ind"], cfg["models"], keep=keep, tag={"run": cfg["run_id"], "held_out": held, "receivers": "+".join(rxs)})  # the fit
            for mdl in cfg["models"]:  # models
                if f["fail"][mdl]:  # the fit failed (the regression script's stop rule): recorded, not scored
                    fails.append({"run_id": cfg["run_id"], "model": mdl, "held_out": held, "inner_held_out": "", "receivers": "+".join(rxs), "reason": f["fail"][mdl]})  # failure row
                    continue  # next model
                for r in rxs:  # receivers
                    idx_r = cells[r][0]  # its own index
                    te_local = ev[r] - off[r]  # its rows in its own index
                    ident = {"run_id": cfg["run_id"], "model": mdl, "receiver": r}  # identifiers
                    row, lat, pr = evaluate(idx_r, te_local, f["p"][mdl][r], held, ident, cfg["s11"])  # the regression script's evaluation
                    trl = idx.tlabel.to_numpy()[f["tr"]]  # training labels of the fit
                    row.update({"train_rows": len(f["tr"]), "train_clean": int((trl == "clean").sum()), "train_attack": int(np.isin(trl, ATTACK).sum())})  # training counts
                    rows.append(row)  # kept
                    lats += lat  # kept
                    prs.append(pr)  # kept
            sel.append({"run_id": cfg["run_id"], "held_out": held, "receivers": "+".join(rxs), "candidates": f["n_cand"], "kept_after_pearson": len(f["prep"]["kept"]), "selected_columns": ";".join(f["names"])})  # selection row
            if keep:  # kept
                kept[(tuple(rxs), held)] = {**f, "idx": idx, "X": X, "off": off, "ev": ev, "cand": cand, "train": train}  # the fit with its frames
    return pd.DataFrame(rows), pd.DataFrame(lats), pd.concat(prs, ignore_index=True), pd.DataFrame(sel), kept, pd.DataFrame(fails, columns=FAIL_COLS)  # outputs


def nested(cells: dict, wc: pd.DataFrame, cfg: dict) -> tuple:  # function: nested scores: inside each outer fold's four training sessions, inner leave-one-session-out with the screen refitted
    cols = d1_columns(wc)  # orbit-free columns
    rows, lats, prs, sel, fails = [], [], [], [], []  # outputs
    for held, train in S.folds():  # outer folds; the outer held-out session is never read here
        for h in train:  # inner held-out session
            itrain = [t for t in train if t != h]  # the three inner training sessions
            groups = [RXS] if cfg["pooled"] else [[r] for r in RXS]  # fits
            for rxs in groups:  # fits
                idx, X, off = combine(cells, rxs)  # frames
                idx_in = idx[idx.test_code.isin(train)].reset_index(drop=True)  # the outer training sessions only: the outer held-out session is dropped before anything is fitted
                X_in = X[idx.test_code.isin(train).to_numpy()].reset_index(drop=True)  # its values
                off_in = {r: int(np.flatnonzero(idx_in.receiver.to_numpy() == r)[0]) for r in rxs}  # dict comprehension: each receiver's first row in the reduced frame
                cand = screened(idx_in, X_in, itrain, cols)  # the screen refitted on the three inner training sessions
                why = unfittable(idx_in, itrain, cand)  # can it be fitted?
                if why:  # not fittable (for example no clean window in the three sessions): recorded, not scored
                    fails += [{"run_id": cfg["run_id"], "model": mdl, "held_out": held, "inner_held_out": h, "receivers": "+".join(rxs), "reason": why} for mdl in cfg["models"]]  # list comprehension: failure rows
                    continue  # next fit
                te_all = eval_rows(idx_in, h)  # the inner held-out rows
                ev = {r: te_all[idx_in.receiver.to_numpy()[te_all] == r] for r in rxs}  # dict comprehension: per receiver
                f = fit_unit(idx_in, X_in, wc, itrain, ev, cand, cfg["rx_ind"], cfg["models"], tag={"run": cfg["run_id"], "outer": held, "inner": h})  # the inner fit
                for mdl in cfg["models"]:  # models
                    if f["fail"][mdl]:  # failed fit: recorded, not scored
                        fails.append({"run_id": cfg["run_id"], "model": mdl, "held_out": held, "inner_held_out": h, "receivers": "+".join(rxs), "reason": f["fail"][mdl]})  # failure row
                        continue  # next model
                    for r in rxs:  # receivers
                        idx_r = idx_in[idx_in.receiver == r].reset_index(drop=True)  # the receiver's reduced index
                        te_local = ev[r] - off_in[r]  # its rows there
                        ident = {"run_id": cfg["run_id"], "model": mdl, "receiver": r, "outer_held_out": held}  # identifiers
                        row, lat, pr = evaluate(idx_r, te_local, f["p"][mdl][r], h, ident, cfg["s11"])  # the regression script's evaluation of the inner fold
                        lab = idx_r.label.to_numpy()[te_local]  # inner held-out labels
                        row["inner_flags"] = ";".join(x for x, ok in (("no clean windows", (lab == "clean").any()), ("no attack windows", np.isin(lab, ATTACK).any())) if not ok)  # what this inner fold cannot carry
                        rows.append(row)  # kept
                        lats += lat  # kept
                        prs.append(pr)  # kept
                sel.append({"run_id": cfg["run_id"], "outer_held_out": held, "inner_held_out": h, "receivers": "+".join(rxs), "candidates": f["n_cand"], "selected_columns": ";".join(f["names"])})  # selection row
    return pd.DataFrame(rows), pd.DataFrame(lats), pd.concat(prs, ignore_index=True), pd.DataFrame(sel), pd.DataFrame(fails, columns=FAIL_COLS)  # outputs


def quiet(fn, *a, **k):  # function: calls fn with its printed progress swallowed (the run log keeps final_run's own lines)
    with contextlib.redirect_stdout(io.StringIO()):  # context manager: stdout into a discarded buffer
        return fn(*a, **k)  # the call's result


def use_labels(label_set: str) -> None:  # function: points the scoring module (through choices_lib) at the registered or the final label set
    SC.use_label_set(label_set)  # the scoring module's files: labels_1hz(_b), labels_evidence(_b), scoring_fold_policy(_b)


def load_cells(root: Path, w: int, final: bool, parts: tuple = ("orbitfree", "geometry", "self", "known")) -> dict:  # function: receiver -> (index with label and tlabel, window values), 2.6.12 dropped at read
    out = {}  # receiver -> (index, values)
    for rx in RXS:  # for loop over the receivers
        idx = M.read_table(root / f"w{w}" / f"{rx}_index.csv.gz")  # the regression script's reader: 2.6.12 dropped
        X = idx[KEYS]  # keys, to align the parts
        for pt in parts:  # for loop over the parts
            X = X.merge(M.read_table(root / f"w{w}" / f"{rx}_{pt}.csv.gz"), on=KEYS, how="left", validate="one_to_one")  # merged one to one
        if final:  # if: the final set, which carries the training label (strict)
            bad = ((idx.train_label == "attack") != idx.label.isin(ATTACK)) | ((idx.train_label == "none") & idx.label.isin(SCORED))  # an attack or scored window whose training label disagrees
            if bad.any():  # if: any
                raise SystemExit(f"STOP: {int(bad.sum())} {rx} windows with an inconsistent training label")  # stop
            tl = np.where(idx.train_label == "clean", "clean", idx.label)  # choices_lib's tlabel: clean for the quiet negatives, else the scoring label
        else:  # the registered set: training label = scoring label
            tl = idx.label.to_numpy()  # unchanged
        out[rx] = (idx[KEYS + ["label"]].assign(tlabel=tl).reset_index(drop=True), X.drop(columns=KEYS).reset_index(drop=True))  # index with tlabel, and values
    return out  # cells


def d1_cells(cells: dict, wc10: pd.DataFrame) -> dict:  # function: the cells restricted to the orbit-free columns (what choices_lib.outer reads)
    cols = L.d1_columns(wc10)  # 778 columns
    return {rx: (idx, X[cols]) for rx, (idx, X) in cells.items()}  # dict comprehension


def run_choices(cells: dict, wc10: pd.DataFrame, run_id: str, keep_all: bool) -> dict:  # function: choices_lib.outer() on orbit-free cells, both models; kept fits in outer_v's record form
    cfg = {**P_CFG, "run_id": run_id}  # the configuration under this run's name
    keep_cells = [(rx, h) for rx in RXS for h, _ in S.folds()] if keep_all else []  # list comprehension: every cell when importance or the controls need the fits
    m, lat, pr, sel, kept, fails = quiet(L.outer, cells, wc10, cfg, keep_cells)  # choices_lib's outer leave-one-session-out run, printing nothing
    recs = {}  # (receiver, fold) -> record
    for (rxs, held), kp in kept.items():  # for loop over the kept fits (one receiver each)
        rx = rxs[0]  # the receiver
        recs[(rx, held)] = {"idx": kp["idx"], "X": kp["X"], "V": M.with_indicators(kp["X"], kp["cand"], wc10, W, L.BLOCKS), "tr": kp["tr"], "te": kp["ev"][rx], "y": kp["y"],  # dict literal: the record (V rebuilt by the regression script's rule, deterministic)
                            "prep": kp["prep"], "sel": kp["sel"], "names": kp["names"], "cand": kp["cand"], "train": kp["train"], "model": {"lr": kp.get("model_lr"), "rf": kp.get("model_rf")}, "p": {k: v[rx] for k, v in kp["p"].items()}}  # the rest
    return {"metrics": m, "latency": lat, "predictions": pr, "selection": sel, "fails": fails.rename(columns={"inner_held_out": "_x"}).drop(columns="_x"), "kept": recs}  # dict literal: outputs


# ======================================================================================================================
# NEW: the inner models kept, the fitted transform of any window, and the gates
# ======================================================================================================================
NO_COLUMN = "the screen keeps no column (fewer than 30 training windows of a class)"  # unfittable()'s reason when the screen script's screen flags every column
LR_DIR, RF_DIR, FINAL_DIR = PROC / "logistic_regression", PROC / "random_forest", PROC / "baselines_controls"  # tuple assignment: the committed folders the gates compare against


def decision(m, Z: np.ndarray) -> np.ndarray:  # function: the logistic regression's decision values, with the regression script's rule for the three spurious Accelerate messages
    with warnings.catch_warnings(record=True) as caught:  # with: records every warning of the product
        warnings.simplefilter("always")  # all recorded, none hidden
        d = m.decision_function(Z)  # sklearn's linear score
    for wn in caught:  # for loop over the warnings
        if not (issubclass(wn.category, RuntimeWarning) and str(wn.message) in M.MATMUL_MSGS):  # if: anything but the three spurious matmul messages
            raise SystemExit(f"STOP: unexpected warning in a decision value: {wn.message}")  # stop
    if not np.isfinite(d).all():  # if: a value is not finite
        raise SystemExit("STOP: a decision value is not finite")  # stop
    return d  # the values (the gates also check expit(d) against the regression script's checked probability, bitwise)


def nested_keep(cells: dict, wc: pd.DataFrame, cfg: dict) -> dict:  # function: nested() with the inner logistic regressions kept and their decision values; an inner fit the screen cannot serve is fitted with the screen not applied
    cols = d1_columns(wc)  # orbit-free columns
    rows, lats, prs, sel, fails, fits = [], [], [], [], [], []  # outputs, as nested() plus the fit records
    for held, train in S.folds():  # outer folds; the outer held-out session is never read here
        for h in train:  # inner held-out session
            itrain = [t for t in train if t != h]  # the three inner training sessions
            for rxs in ([RXS] if cfg["pooled"] else [[r] for r in RXS]):  # fits (one per receiver for P)
                idx, X, off = combine(cells, rxs)  # frames
                keep_rows = idx.test_code.isin(train).to_numpy()  # the outer training sessions only: the outer held-out session is dropped before anything is fitted
                idx_in, X_in = idx[keep_rows].reset_index(drop=True), X[keep_rows].reset_index(drop=True)  # tuple assignment: their index and values
                off_in = {r: int(np.flatnonzero(idx_in.receiver.to_numpy() == r)[0]) for r in rxs}  # dict comprehension: each receiver's first row in the reduced frame
                cand = screened(idx_in, X_in, itrain, cols)  # the screen script's screen refitted on the three inner training sessions
                screen_applied = True  # the screen serves this fit unless it keeps no column
                why = unfittable(idx_in, itrain, cand)  # can it be fitted?
                if why == NO_COLUMN:  # if: the screen keeps no column (no opening-jamming windows in the three inner sessions)
                    cand, screen_applied = list(cols), False  # tuple assignment: every orbit-free candidate, screen not applied
                    why = unfittable(idx_in, itrain, cand)  # checked again
                if why:  # still not fittable: recorded, not scored
                    fails += [{"run_id": cfg["run_id"], "model": mdl, "held_out": held, "inner_held_out": h, "receivers": "+".join(rxs), "reason": why} for mdl in cfg["models"]]  # list comprehension: failure rows
                    continue  # next fit
                te_all = eval_rows(idx_in, h)  # the inner held-out rows, every window of that session in time order
                ev = {r: te_all[idx_in.receiver.to_numpy()[te_all] == r] for r in rxs}  # dict comprehension: per receiver
                f = fit_unit(idx_in, X_in, wc, itrain, ev, cand, cfg["rx_ind"], cfg["models"], keep=True, tag={"run": cfg["run_id"], "outer": held, "inner": h})  # the inner fit, models kept
                for mdl in cfg["models"]:  # models
                    if f["fail"][mdl]:  # failed fit: recorded, not scored
                        fails.append({"run_id": cfg["run_id"], "model": mdl, "held_out": held, "inner_held_out": h, "receivers": "+".join(rxs), "reason": f["fail"][mdl]})  # failure row
                        continue  # next model
                    for r in rxs:  # receivers
                        idx_r = idx_in[idx_in.receiver == r].reset_index(drop=True)  # the receiver's reduced index
                        te_local = ev[r] - off_in[r]  # its rows there
                        ident = {"run_id": cfg["run_id"], "model": mdl, "receiver": r, "outer_held_out": held}  # identifiers
                        row, lat, pr = evaluate(idx_r, te_local, f["p"][mdl][r], h, ident, cfg["s11"])  # the regression script's evaluation of the inner fold
                        lab = idx_r.label.to_numpy()[te_local]  # inner held-out labels
                        row["inner_flags"] = ";".join(x for x, ok in (("no clean windows", (lab == "clean").any()), ("no attack windows", np.isin(lab, ATTACK).any())) if not ok)  # what this inner fold cannot carry
                        dec = decision(f["model_lr"], f["Z"][ev[r]]) if mdl == "lr" else np.full(len(te_local), np.nan)  # conditional expression: the logistic regression's decision value (the surrogate's z_t), NaN for the forest
                        pr = pr.assign(dec=dec, screen_applied=screen_applied)  # beside every prediction
                        rows.append(row)  # kept
                        lats += lat  # kept
                        prs.append(pr)  # kept
                sel.append({"run_id": cfg["run_id"], "outer_held_out": held, "inner_held_out": h, "receivers": "+".join(rxs), "candidates": f["n_cand"], "selected_columns": ";".join(f["names"])})  # selection row, as nested() writes it
                # dict literal over two lines: the record (the forest and the matrices are not kept: memory); "transform" (added for the agent experiment) holds the inner fit's own
                # fitted preparation, candidates and selection, the three things transform() reads, so any window can be put through the inner fit's transform
                fits.append({"receiver": "+".join(rxs), "outer_held_out": held, "inner_held_out": h, "screen_applied": screen_applied, "candidates": f["n_cand"], "train_rows": len(f["tr"]), "selected_columns": ";".join(f["names"]), "model_lr": f.get("model_lr"),
                             "transform": {"prep": {k: f["prep"][k] for k in ("names", "med", "scaler", "compressed")}, "cand": cand, "sel": f["sel"]}})  # dict comprehension inside a dict literal: prep's four fitted parts (not its matrix Z), the candidate list, the selected positions
    return {"metrics": pd.DataFrame(rows), "latency": pd.DataFrame(lats), "predictions": pd.concat(prs, ignore_index=True), "selection": pd.DataFrame(sel), "fails": pd.DataFrame(fails, columns=FAIL_COLS), "fits": fits}  # dict literal: outputs


def build() -> dict:  # function: P from the final windows: cells, the outer fits (every record kept) and the nested inner fits
    use_labels("final")  # the scoring module points at the final label set, as final_run's labels_in_use() does
    wc10 = window_dict()  # the feature script's window dictionary at w = 10
    cells = d1_cells(load_cells(FINAL, W, True), wc10)  # the final windows with the training label, orbit-free columns (the p command of 06_run_baselines.py)
    for rx, (idx, _) in cells.items():  # for loop over the receivers
        if (idx.test_code == SEALED).any():  # if: a 2.6.12 window survived the reader (cannot)
            raise SystemExit(f"STOP: 2.6.12 in {rx}'s cells")  # stop
    outer = run_choices(cells, wc10, "P", True)  # P's outer leave-one-session-out run, every fit kept
    if len(outer["fails"]):  # if: an outer fit failed (none did in the adopted models on the final labels)
        raise SystemExit(f"STOP: {len(outer['fails'])} failed outer fits")  # stop
    nested = quiet(nested_keep, cells, wc10, P_CFG)  # the inner fits, printing nothing
    return {"wc10": wc10, "cells": cells, "outer": outer, "nested": nested}  # dict literal: everything P holds


def transform(rec: dict, X: pd.DataFrame, wc10: pd.DataFrame) -> np.ndarray:  # function: ANY windows of the receiver (orbit-free columns, scored or not) -> the outer fold's 36 selected columns, by the fitted steps of the regression script's prepare()
    prep = rec["prep"]  # the fold's fitted preparation
    V = M.with_indicators(X, rec["cand"], wc10, W, BLOCKS)  # the regression script: the fold's candidates plus the orbit-free missing-indicator (a per-window value)
    A = V[prep["names"]].to_numpy(dtype=float)  # the columns that had a training value, in prepare()'s order
    A = np.where(np.isnan(A), prep["med"], A)  # every gap filled with its training median
    Z = S.apply_scaler(A, prep["scaler"])  # 07c: centred and scaled with the training median and IQR
    if prep["compressed"]:  # if: the compressed pipeline (P's)
        Z = np.sign(Z) * np.log1p(np.abs(Z))  # signed-log compression, as prepare() applies it
    # np.ascontiguousarray: Z[:, sel] comes back in Fortran (column) order; the fit scored rows of Z[:, sel][te], which are in C (row) order, and
    # Apple Accelerate's matrix product sums in a different order for the two layouts, so the C copy is what reproduces the fit's probabilities bitwise
    return np.ascontiguousarray(Z[:, rec["sel"]])  # the 36 selected columns, in the selection's order, C order


def score(rec: dict, Z: np.ndarray) -> dict:  # function: the outer models' scores of transformed windows
    m = rec["model"]["lr"]  # the outer logistic regression
    p_lr, _ = M.predict(m, Z)  # the regression script's checked probability
    return {"dec": decision(m, Z), "p_lr": p_lr, "p_rf": R.proba(rec["model"]["rf"], Z)}  # dict literal: decision value, both probabilities


def read_round_trip(path: Path) -> pd.DataFrame:  # function: a committed CSV with every float parsed to the double that was written
    return pd.read_csv(path, float_precision="round_trip", dtype={"test_code": str, "held_out": str, "outer_held_out": str, "inner_held_out": str})  # session codes kept as text


def same_rows(new: pd.DataFrame, old: pd.DataFrame, keys: list, cols: list) -> tuple:  # function: (every committed row matched, values equal bitwise on cols) by keys
    m = old.merge(new, on=keys, how="left", suffixes=("_c", ""), validate="one_to_one", indicator=True)  # committed rows beside the new ones
    matched = bool((m._merge == "both").all())  # every committed row found
    eq = matched and all((m[c + "_c"].to_numpy() == m[c].to_numpy()).all() for c in cols)  # generator in all(): exact equality, column by column
    return matched, bool(eq)  # both answers


def gates(B: dict) -> pd.DataFrame:  # function: G0, G0n, G0s, G0w and the extra checks; one row each
    rows = []  # results
    chk = lambda gate, name, ok, detail: rows.append({"gate": gate, "check": name, "pass": bool(ok), "detail": detail})  # lambda: records one check
    pr = B["outer"]["predictions"]  # P's outer predictions, both models
    keys = ["run_id", "model", "receiver", "held_out", "sec"]  # their keys
    for mdl, f in (("lr", LR_DIR / "logistic_regression_predictions_primary.csv.gz"), ("rf", RF_DIR / "random_forest_predictions_primary.csv.gz")):  # for loop over the two committed files
        c = read_round_trip(f)  # the committed predictions
        n = pr[pr.model == mdl]  # the new ones
        matched, eq = same_rows(n, c, keys, ["label", "p", "alarm"])  # bitwise on label, p and alarm
        chk("G0", f"outer {mdl} predictions = {f.name}", matched and eq and len(n) == len(c), f"{len(n)} new, {len(c)} committed")  # recorded
    chk("G0", "outer predictions, both models together", len(pr) == 2 * len(read_round_trip(LR_DIR / "logistic_regression_predictions_primary.csv.gz")), f"{len(pr)} rows")  # 32,014 expected
    npr = B["nested"]["predictions"]  # the inner out-of-fold predictions
    c = read_round_trip(FINAL_DIR / "adopted_nested_predictions.csv.gz")  # the committed ones
    nk = ["run_id", "model", "receiver", "outer_held_out", "held_out", "sec"]  # their keys
    matched, eq = same_rows(npr, c, nk, ["label", "p", "alarm"])  # bitwise
    extra = npr.merge(c[nk], on=nk, how="left", indicator=True)  # new rows beside the committed keys
    extra = extra[extra._merge == "left_only"]  # rows the committed file does not hold
    chk("G0n", "inner predictions = adopted_nested_predictions.csv.gz on every row it holds", matched and eq, f"{len(c)} committed rows, all matched and equal: {matched and eq}")  # recorded
    chk("G0n", "rows the committed file lacks come only from fits with the screen not applied", (~extra.screen_applied.astype(bool)).all(), f"{len(extra)} new rows; inner folds {sorted(set(zip(extra.receiver, extra.outer_held_out, extra.held_out)))}")  # recorded
    lr = npr[npr.model == "lr"]  # the logistic regression's inner rows
    chk("G0n", "inner probability = expit(decision value) bitwise", np.array_equal(expit(lr.dec.to_numpy()), lr.p.to_numpy()), f"{len(lr)} rows")  # the surrogate's z_t is the score behind p
    sel = B["nested"]["selection"]  # the inner selections
    fits = pd.DataFrame(B["nested"]["fits"]).drop(columns="model_lr")  # the inner fit records
    s_old = read_round_trip(FINAL_DIR / "adopted_nested_selection.csv")  # the committed inner selections
    scr = sel.merge(fits[["receiver", "outer_held_out", "inner_held_out", "screen_applied"]].rename(columns={"receiver": "receivers"}), on=["receivers", "outer_held_out", "inner_held_out"])  # selection rows with their screen flag
    m_, e_ = same_rows(scr[scr.screen_applied].drop(columns="screen_applied"), s_old, ["run_id", "outer_held_out", "inner_held_out", "receivers"], ["candidates", "selected_columns"])  # equal on the screened fits
    chk("G0n", "inner selections = adopted_nested_selection.csv", m_ and e_ and int(scr.screen_applied.sum()) == len(s_old), f"{len(s_old)} committed, {int(scr.screen_applied.sum())} screened fits, {int((~scr.screen_applied).sum())} with the screen not applied")  # recorded
    so = B["outer"]["selection"]  # the outer selections (choices_lib.outer's table)
    rf_tab = read_round_trip(RF_DIR / "random_forest_selection_primary.csv")  # the random forest's committed table
    m_, e_ = same_rows(so, rf_tab, ["run_id", "held_out", "receivers"], ["candidates", "kept_after_pearson", "selected_columns"])  # equal
    chk("G0s", "outer selections = random_forest/random_forest_selection_primary.csv", m_ and e_ and len(so) == len(rf_tab), f"{len(rf_tab)} fits")  # recorded
    lr_tab = read_round_trip(LR_DIR / "logistic_regression_selection_primary.csv")  # the logistic regression's committed table
    m_, e_ = same_rows(so, lr_tab, ["held_out", "receivers"], ["selected_columns"])  # the selected columns
    chk("G0s", "outer selections = logistic_regression/logistic_regression_selection_primary.csv", m_ and e_ and len(so) == len(lr_tab), f"{len(lr_tab)} fits")  # recorded
    kc = read_round_trip(FINAL_DIR / "adopted_selection_kept_columns.csv")  # P's Pearson-kept and selected columns
    kept = pd.DataFrame([{"held_out": h, "receivers": rx, "kept_columns": ";".join(rec["prep"]["names"][j] for j in rec["prep"]["kept"]), "selected_columns": ";".join(rec["names"])} for (rx, h), rec in B["outer"]["kept"].items()])  # list comprehension: each record's kept and selected columns
    m_, e_ = same_rows(kept, kc, ["held_out", "receivers"], ["kept_columns", "selected_columns"])  # equal
    chk("G0s", "outer kept and selected columns = adopted_selection_kept_columns.csv", m_ and e_ and len(kept) == len(kc), f"{len(kc)} fits")  # recorded
    n_te, n_tr, bad = 0, 0, []  # windows checked; failures
    for (rx, h), rec in B["outer"]["kept"].items():  # for loop over the 15 outer fits
        X = rec["X"]  # the receiver's windows (orbit-free columns), every session
        te = rec["te"]  # the held-out session's rows in time order
        Zt = transform(rec, X.iloc[te], B["wc10"])  # ANY-window transform of the held-out windows
        sc = score(rec, Zt)  # their scores
        ok_te = np.array_equal(Zt, rec["prep"]["Z"][te][:, rec["sel"]]) and np.array_equal(sc["p_lr"], rec["p"]["lr"]) and np.array_equal(sc["p_rf"], rec["p"]["rf"]) and np.array_equal(expit(sc["dec"]), sc["p_lr"])  # matrix and scores bitwise
        trs = np.flatnonzero(rec["idx"].test_code.isin(rec["train"]).to_numpy())  # every window of the training sessions, scored or not
        ok_tr = np.array_equal(transform(rec, X.iloc[trs], B["wc10"]), rec["prep"]["Z"][trs][:, rec["sel"]])  # matrix bitwise
        n_te, n_tr = n_te + len(te), n_tr + len(trs)  # counted
        if not (ok_te and ok_tr):  # if: either differs
            bad.append(f"{rx} {h}")  # recorded
    chk("G0w", "transform() and score() reproduce the fit's matrix and the committed probabilities", not bad, f"{n_te} held-out and {n_tr} training windows in 15 fits; failing: {bad}")  # recorded
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))  # this file's syntax tree
    mods = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names] + [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]  # list comprehensions: every imported module
    scripts = sorted({n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.endswith(".py") and n.value != ".py"})  # set comprehension: every script file named (load_script loads them from codes/)
    chk("extra", "no import from codes/tests/", not any("tests" in x for x in mods) and all((CODES / x).is_file() for x in scripts), f"imports {len(mods)}; scripts loaded from codes/: {scripts}")  # recorded
    chk("extra", "2.6.12 absent from every table built", all(SEALED not in set(t.astype(str)) for t in (pr.held_out, npr.held_out, npr.outer_held_out)) and all(not (idx.test_code == SEALED).any() for idx, _ in B["cells"].values()), "cells, outer and inner predictions")  # recorded
    return pd.DataFrame(rows)  # the gate table


def main() -> None:  # function: builds P and runs the gates
    t0 = time.time()  # clock (printed only)
    B = build()  # P
    g = gates(B)  # the gates
    ENV_DIR.mkdir(parents=True, exist_ok=True)  # the environment folder
    g.to_csv(ENV_DIR / "adopted_pipeline_gates.csv", index=False, lineterminator="\n")  # the gate table
    pd.DataFrame(B["nested"]["fits"]).drop(columns=["model_lr", "transform"]).to_csv(ENV_DIR / "adopted_pipeline_inner_fits.csv", index=False, lineterminator="\n")  # the inner fits, with the screen flag (list in drop(): the model and the agent experiment transform record are objects, not table columns)
    print(g.to_string(index=False))  # printed for the report
    print(f"adopted pipeline: {int(g['pass'].sum())}/{len(g)} checks pass; {time.time() - t0:.0f} s")  # summary
    if not g["pass"].all():  # if: any gate failed
        sys.exit(1)  # exit status 1, so a failure cannot be missed


if __name__ == "__main__":  # guard: runs main() only when the file is executed, not when imported
    main()  # runs main
