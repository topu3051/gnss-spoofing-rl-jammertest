#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
Follow-up of data check 4: the per-receiver label sensitivity of the two supervised baselines.

The labels of Section 4.6 pool the attack evidence of three receivers that, the supervisor reports, each had their own antenna.
8_data_checks/03_antenna_dependence.py found that 23.3 % (mosaic-X5) and 28.9 % (mosaic-H) of the attack seconds rest on other receivers'
evidence alone, so the decision rule fixed before the check sends the labels to this sensitivity. Pooled labels stay primary.

What runs (a wrapper: 5_baselines/01_logistic_regression.py and 5_baselines/02_random_forest.py are imported through the library final_lib, never edited):
  a  per-receiver labels (own evidence and own indicators; quiet negatives recomputed the same way) from
     Data/processed/data_checks/labels_per_receiver_1hz.csv, written by 8_data_checks/03_antenna_dependence.py; the final window files are relabelled at
     their decision second (the window set cannot move: the excluded seconds are the same under both rules, checked there);
  b the adopted primary P (choices_lib.outer, both models, five leave-one-session-out folds, MI seed 20260914, forest seed 20260914,
     threshold 0.5) and its nested run (the adopted-pipeline script's nested_keep, which the agent experiment's points read) on the per-receiver labels; the operating points 0.02, 0.05 and 0.10 by the training script's rule (the
     lowest threshold within the share on the nested out-of-fold clean windows); C4b-refit (100 within-session permutations,
     seeds 20260914 + 1000 + k) with the session-control verdicts, the grouped importance and the forest script's classification, which answer RQ1;
     2.6.12 is dropped at read by the regression script's reader and never enters a fit, threshold, selection or importance;
  c cross-scoring without retraining: the published pooled models (logistic_regression and random_forest predictions, the agent experiment operating points) on the
     per-receiver labels, and the per-receiver models on the pooled labels;
  d one table per receiver: label set x training labels x model x operating point -> per-class precision, recall, F1 and the
     realised clean share, pooled over folds (Table the forest script's definition) and as the interquartile mean over folds (the agent experiment's), beside the
     published pooled numbers read from their files; RQ1 (classification) beside the published one;
  e the decision rule fixed before any result: RQ1 verdict change for any receiver, LR and RF swapping rank at 0.05 (by the
     interquartile-mean F1 of the agent experiment's decision mode), or F1 at 0.05 moving by more than 0.10 -> a limitation in 6.x.
P9 (agent against the best baseline) is not formed under the per-receiver labels: it needs the agents' operating points chosen anew
on inner streams under those labels, i.e. new agent runs, which the decision rule excludes; the published P9 stays as it is.

Reproduction gates before any per-receiver number: P run again on the pooled cells reproduces logistic_regression/random_forest predictions bitwise;
the adopted-pipeline script's nested scheme on the pooled cells reproduces the agent experiment's nested predictions bitwise and the threshold rule on them reproduces the agent experiment's
lr/rf points; the scorer reproduces logistic_regression/random_forest pooled tables at 0.5,
the agent experiment's per-fold decision-mode runs and the agent experiment's interquartile means.

Usage   python3 codes/8_data_checks/04_label_sensitivity.py [--root DIR] [--smoke]
        --root DIR writes under DIR/Data/processed/data_checks/label_sensitivity and DIR/Graph (the second pass of the byte-identity check)
        --smoke    2 permutations per C4b cell instead of 100: a test of the code paths, never reported
Reads   Data/processed/data_checks/labels_per_receiver_1hz.csv, features_final/w10 (through final_lib), logistic_regression and random_forest predictions and pooled tables,
        random_forest classification and importance, rl_agents/training/nested_predictions.csv.gz, baselines_controls/r3_verdicts.csv,
        rl_agents/evaluation/deployable_points.csv, point_runs.csv, aggregate.csv, verdicts.csv
Writes  Data/processed/data_checks/label_sensitivity/*  (listed in write order at the end of main) and Graph/v1_label_sensitivity_f1.png
"""

from __future__ import annotations  # future import: type hints kept as text (the project's convention)

import importlib.util  # standard-library module loader: final_lib is loaded by its path
import os  # operating-system access: MPLCONFIGDIR is set before matplotlib loads
import sys  # standard-library system access: the command line, sys.path and sys.modules
import time  # standard-library clock, for the run-time line
from pathlib import Path  # Path objects for building file paths

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
os.environ["MPLCONFIGDIR"] = str(PROJECT / "codes" / ".mplconfig")  # matplotlib's cache inside the project folder

import numpy as np  # NumPy: arrays
import pandas as pd  # pandas: tables
import matplotlib  # the plotting library itself, to choose a backend
matplotlib.use("Agg")  # file-only backend: no window, identical files on run again
import matplotlib.pyplot as plt  # pyplot: figures and axes

sys.path.insert(0, str(PROJECT / "codes" / "common"))  # codes/ first on the import path, so the helpers module is found
from helpers import exit_if_failed, save_fig  # status 1 on a failed check; the project's byte-stable figure saver

ARGS = sys.argv[1:]  # the command line after the script name
ROOT = Path(ARGS[ARGS.index("--root") + 1]).resolve() if "--root" in ARGS else PROJECT  # conditional expression: the output root (pass 2) or the project folder
PROC = PROJECT / "Data" / "processed"  # inputs are always read from the project
OUT = ROOT / "Data" / "processed" / "data_checks" / "label_sensitivity"  # this run's outputs
GRAPH = ROOT / "Graph"  # the figure folder
FIGURE = "label_sensitivity_f1"  # the figure's file stem
SHARES = (0.02, 0.05, 0.1)  # tuple: the registered shares of clean windows in alarm
POINTS = ["0.5"] + [str(s) for s in SHARES]  # list: the operating points as text (threshold 0.5, then the three shares)
MODELS = ("lr", "rf")  # tuple: logistic regression and random forest
N_PERM = 2 if "--smoke" in ARGS else 100  # conditional expression: permutations per C4b cell, 100 as the adopted run (E1); 2 in a code-path test (--smoke, never reported)
CHECKS: list = []  # list: one dict per self-check
FAILED: list = []  # list: names of the checks that failed

_spec = importlib.util.spec_from_file_location("final_lib", PROJECT / "codes" / "5_baselines" / "05_final_labels_library.py")  # the final-label run's library, by its path
B = importlib.util.module_from_spec(_spec)  # an empty module object
sys.modules["final_lib"] = B  # registered under the name its own code expects
_spec.loader.exec_module(B)  # its top level: loads choices_lib, which loads the window-feature, screen, scoring, regression and forest scripts unchanged
L, M, R, S, SC = B.L, B.M, B.R, B.S, B.SC  # choices_lib and the regression, forest, screen and scoring modules as the library holds them
_spec11 = importlib.util.spec_from_file_location("adopted_pipeline_for_label_sensitivity", PROJECT / "codes" / "5_baselines" / "07_adopted_pipeline.py")  # 11a: P's nested scheme as the agent experiment's points used it
A11 = importlib.util.module_from_spec(_spec11)  # an empty module object
sys.modules["adopted_pipeline_for_label_sensitivity"] = A11  # registered
_spec11.loader.exec_module(A11)  # its top level (definitions; its gates run only from its main())
RXS, KEYS, ATTACK, SCORED = B.RXS, B.KEYS, B.ATTACK, B.SCORED  # receivers, key columns, attack labels, scored labels
CELLS_C4B = R.c4b_cells()  # the forest script's C4b cells: 2.6.2 and 2.6.4 of every receiver, 2.6.10 of u-blox 01
LABEL_SETS = {"pooled": ("label_pooled", "train_label_pooled"), "per_receiver": ("label_per_receiver", "train_label_per_receiver")}  # dict: label set -> its scoring and training columns


def check(name: str, ok, detail: str) -> None:  # function: records one self-check and prints it (ok None marks a reported number)
    CHECKS.append({"check": name, "result": "REPORT" if ok is None else ("PASS" if ok else "FAIL"), "detail": detail})  # nested conditional expression picks the word
    if ok is not None and not bool(ok):  # if: a pass/fail check that did not pass
        FAILED.append(name)  # kept for the exit status
    print(f"  [{CHECKS[-1]['result']}] {name}: {detail}", flush=True)  # printed for the run log, at once


def lowest_threshold(v: np.ndarray, n: int, s: float) -> float:  # function: the training script's rule (12_agent_training.lowest_threshold, copied without change): the lowest t with #(v >= t) <= floor(s n)
    v = np.sort(v[np.isfinite(v)])[::-1]  # the values, descending
    k = int(np.floor(s * n))  # at most k of the n clean windows may be in alarm
    if not len(v):  # no value at all
        return np.nan  # undefined
    if k >= len(v):  # every value may alarm
        return float(v[-1])  # the smallest value
    return float(np.nextafter(v[k], np.inf))  # just above the (k+1)-th largest value: at most k values reach it


def iqm(x: np.ndarray) -> float:  # function: the evaluation script's interquartile mean over the defined values (the mean after cutting floor(n/4) from each end)
    x = np.sort(np.asarray(x, float)[np.isfinite(np.asarray(x, float))])  # the defined values, sorted
    if not len(x):  # if: none defined
        return np.nan  # undefined
    c = int(np.floor(0.25 * len(x)))  # how many to cut from each end
    return float(x[c:len(x) - c].mean())  # the middle values' mean


def relabel(cells: dict, lab: pd.DataFrame, set_name: str) -> dict:  # function: the window cells with the decision second's label and training label of a label set
    lc, tc = LABEL_SETS[set_name]  # tuple unpacking: the set's scoring and training columns
    out = {}  # receiver -> (index, values)
    for rx, (idx, X) in cells.items():  # for loop over the receivers' cells
        m = idx[KEYS].merge(lab[KEYS + [lc, tc]], on=KEYS, how="left", validate="one_to_one")  # each window's decision second, one to one
        if m[lc].isna().any():  # if: a window without a label row
            raise SystemExit(f"STOP: {rx} windows without a label row")  # stop
        new_lab = m[lc].to_numpy(object)  # the scoring label
        tl = np.where(m[tc].to_numpy(object) == "clean", "clean", new_lab)  # final_lib's tlabel: clean for the quiet negatives, else the scoring label
        bad = ((m[tc] == "attack").to_numpy() != np.isin(new_lab, ATTACK)) | ((m[tc] == "none").to_numpy() & np.isin(new_lab, SCORED))  # final_lib.load_cells' consistency rule
        if bad.any():  # if: any window breaks it
            raise SystemExit(f"STOP: {rx}: {int(bad.sum())} windows with an inconsistent training label")  # stop
        out[rx] = (idx[KEYS].assign(label=new_lab, tlabel=tl).reset_index(drop=True), X)  # the relabelled index beside the unchanged values
    return out  # the cells


def score_fold_rows(pred: pd.DataFrame, labmap: pd.DataFrame, thr: dict) -> tuple:  # function: the regression script's evaluate_fold on every (receiver, fold) of one model's predictions, labels from labmap, thresholds from thr
    rows, lats, prs = [], [], []  # metric rows, latency rows, prediction frames
    p_ = pred.merge(labmap, on=["receiver", "held_out", "sec"], how="left", validate="one_to_one")  # each prediction's label in the scored set
    for (rx, held), g in p_.groupby(["receiver", "held_out"], sort=False):  # for loop over the folds in the predictions' order
        g = g.sort_values("sec", kind="stable").reset_index(drop=True)  # time order, as the scoring module needs
        idx = g[["sec"]].assign(label=g.lab.to_numpy(object))  # an index-like frame: seconds and labels
        te = np.arange(len(g))  # every row is held out
        old = M.THR  # the regression script's threshold constant
        M.THR = float(thr[(rx, held)])  # in-memory override (choices_lib's inner-fold threshold does the same); restored below
        try:  # try/finally: the constant always comes back
            row, lat = M.evaluate_fold(idx, te, g.p.to_numpy(float), held)  # the regression script's fold evaluation under the scoring module's policy
        finally:  # always
            M.THR = old  # restored
        row.update(receiver=rx, threshold=float(thr[(rx, held)]))  # identifiers
        rows.append(row)  # kept
        lats += [{**r, "receiver": rx} for r in lat]  # list comprehension with dict unpacking: latency rows (pooled intervals; not interpreted here)
        prs.append(pd.DataFrame({"receiver": rx, "held_out": held, "sec": g.sec, "label": g.lab, "p": g.p, "alarm": g.p.to_numpy(float) >= float(thr[(rx, held)])}))  # the scored frame
    return pd.DataFrame(rows), pd.DataFrame(lats), pd.concat(prs, ignore_index=True)  # three tables


def summarise(m: pd.DataFrame, lt: pd.DataFrame, pr: pd.DataFrame) -> pd.DataFrame:  # function: per receiver, the regression script's pooled figures plus the clean class and the interquartile means over folds
    base = M.pooled(m, lt, pr)  # the regression script's pooled table (Table the forest script's definition)
    rows = []  # one dict per receiver
    for rx in base.receiver:  # for loop over the receivers
        mm, pp = m[m.receiver == rx], pr[pr.receiver == rx]  # its folds and windows
        tp, fn, fp, tn = (int(mm[c].sum()) for c in ("tp", "fn", "fp", "tn"))  # generator unpacked: the summed cells over all five folds
        cp = tn / (tn + fn) if tn + fn else np.nan  # clean precision: windows without alarm that are clean
        cr = tn / (tn + fp) if tn + fp else np.nan  # clean recall: clean windows without alarm
        sc = pp[pp.label.isin(SCORED)]  # scored windows
        att = sc.label.isin(ATTACK).to_numpy()  # attack among them
        auc = float(B.roc_auc_score(att, sc.p)) if att.any() and (~att).any() else np.nan  # AUROC over the pooled predictions (as choices_lib.summarise)
        cl = pp[pp.label == "clean"]  # clean windows
        # dict literal: the pooled figures, the clean class, the realised clean share and the interquartile means over the folds (the agent experiment's decision mode)
        rows.append({**base[base.receiver == rx].iloc[0][["recall_pooled", "recall_jam_spoof_pooled", "recall_spoof_pooled", "precision_pooled", "f1_pooled", "precision_folds"]].to_dict(),
                     # the clean class's F1 by the scoring module's count rule: NaN only without clean windows (tn + fp = 0), 0 when tn = 0, else the harmonic mean
                     "receiver": rx, "clean_precision_pooled": cp, "clean_recall_pooled": cr, "clean_f1_pooled": np.nan if tn + fp == 0 else (0.0 if tn == 0 else 2 * cp * cr / (cp + cr)),  # continued: more fields of the row
                     "clean_in_alarm_share_pooled": float(cl.alarm.mean()) if len(cl) else np.nan, "clean_windows": len(cl), "attack_windows": int(pp.label.isin(ATTACK).sum()),  # continued: more fields of the row
                     "auroc_pooled_predictions": auc, "recall_iqm": iqm(mm.recall), "precision_iqm": iqm(mm.precision), "f1_iqm": iqm(mm.f1), "clean_share_iqm": iqm(mm.clean_in_alarm_share)})  # the row
    return pd.DataFrame(rows)  # the table


def write_csv(df: pd.DataFrame, name: str) -> None:  # function: one table into the run's output folder with fixed line endings
    B.write_csv(df, OUT / name)  # final_lib's writer (folder created; byte-stable)


def write_gz(df: pd.DataFrame, name: str) -> None:  # function: one gzip table with no name or time in its header
    B.write_gz(df, OUT / name)  # final_lib's writer


def thresholds_from_nested(nested: pd.DataFrame) -> pd.DataFrame:  # function: the training script's deployable() points of the two baselines from nested out-of-fold predictions
    rows = []  # one dict per point
    for rx in RXS:  # for loop over the receivers
        for outer, train in S.folds():  # for loop over the outer folds and their training sessions (the screen script's folds)
            for mdl in MODELS:  # for loop over the two models
                q = nested[(nested.model == mdl) & (nested.receiver == rx) & (nested.outer_held_out == outer) & nested.held_out.isin(train) & (nested.label == "clean")]  # its inner clean out-of-fold windows
                for s in SHARES:  # for loop over the shares
                    t = lowest_threshold(q.p.to_numpy(), len(q), s)  # the training script's rule
                    rows.append({"receiver": rx, "outer": outer, "detector": mdl, "share": s, "param": t, "inner_clean": len(q), "inner_clean_share": float((q.p >= t).mean()) if len(q) else np.nan})  # the point
    return pd.DataFrame(rows)  # the table


def thr_map(points: pd.DataFrame, mdl: str, point: str) -> dict:  # function: (receiver, fold) -> threshold of one model at one operating point
    if point == "0.5":  # if: the a-priori threshold
        return {(rx, h): 0.5 for rx in RXS for h, _ in S.folds()}  # dict comprehension: 0.5 everywhere
    p = points[(points.detector == mdl) & np.isclose(points.share, float(point))]  # the model's points at that share
    return {(r.receiver, r.outer): float(r.param) for r in p.itertuples()}  # dict comprehension over the points


def published_tables() -> dict:  # function: every published number this run sits beside, read from its file
    rd = lambda p: B.read_exact(PROC / p)  # lambda: final_lib's exact reader on a processed file
    return {"pred": {"lr": rd("logistic_regression/logistic_regression_predictions_primary.csv.gz"), "rf": rd("random_forest/random_forest_predictions_primary.csv.gz")},  # dict literal: the published held-out predictions
            "pooled": {"lr": rd("logistic_regression/logistic_regression_pooled_primary.csv"), "rf": rd("random_forest/random_forest_pooled_primary.csv")},  # the published pooled tables (Table 10)
            "points": rd("rl_agents/evaluation/deployable_points.csv"), "point_runs": rd("rl_agents/evaluation/point_runs.csv"), "aggregate": rd("rl_agents/evaluation/aggregate.csv"),  # the agent experiment's points, runs and IQMs
            "verdicts": rd("rl_agents/evaluation/verdicts.csv"), "nested": rd("rl_agents/training/nested_predictions.csv.gz"), "r3": rd("baselines_controls/r3_verdicts.csv"),  # P9, the adopted-pipeline script's nested predictions (the agent experiment's points read these), the session control
            "class": {"rf": rd("random_forest/random_forest_c4b_classification.csv"), "lr": rd("random_forest/random_forest_c4b_classification_logreg.csv")}}  # the published RQ1 classification


def classify_run(P: dict, wc10: pd.DataFrame, tag: str) -> tuple:  # function: final_run.cmd_c4b's steps on a run's kept fits (C4b-refit, importance, classification); returns (b4, verdicts, classes, importance)
    b4 = B.c4b_refit(P["kept"], wc10, MODELS, CELLS_C4B, f"C4BR_{tag}", range(N_PERM))  # every within-session permutation, both models
    ta = {(r.model, r.receiver, r.held_out): r.auroc for r in P["metrics"].itertuples()}  # dict comprehension: the held-out AUROC of each fold
    v = B.cell_verdicts(b4, ta, CELLS_C4B, f"C4BR_{tag}")  # the session control per cell and model
    fam, xs = R.name_maps(wc10)  # the forest script's column -> family and secondary-group maps
    order = [(rx, h) for rx in RXS for h, _ in S.folds()]  # the forest script's cell order
    classes, imps = {}, {}  # model -> classification; model -> (true importance, median-permutation importance)
    for mdl, cfg_ in (("rf", "primary"), ("lr", "logreg_no_residual")):  # the two models under final_run's configuration names
        ip, _, _ = B.importance(P["kept"], wc10, mdl, cfg_, order)  # the forest script's importance of the true model in every cell
        med, ib = {}, []  # median permutations; their importance rows
        for rx, held in CELLS_C4B:  # for loop over the C4b cells
            vv = v[(v.model == mdl) & (v.receiver == rx) & (v.held_out == held)].verdict.iloc[0]  # the cell's session-control verdict
            if vv == "not assessable":  # if: no median model
                continue  # not classified (final_run's rule)
            k = B.median_perm(b4, mdl, rx, held)  # the median permutation
            med[(rx, held)] = k  # kept
            kp = P["kept"][(rx, held)]  # the true fit's record
            f = B.c4b_prep(kp, wc10, k)  # the permuted preparation again (deterministic)
            m_, p_, fail = B.c4b_model(kp, f, mdl, f"C4BR_{tag}", held, rx, k)  # the median permuted model refitted
            lab = kp["idx"].label.to_numpy()[kp["te"]]  # held-out labels
            a_ref = float(b4[(b4.model == mdl) & (b4.receiver == rx) & (b4.held_out == held) & (b4.perm == k)].auroc.iloc[0])  # its AUROC in the permutation table
            check(f"{tag} median permuted {mdl} {rx} {held} reproduced", (not fail) and B.scores(p_, lab)["auroc"] == a_ref, f"permutation {k}, AUROC {a_ref:.4f}")  # final_run's gate
            kpm = {"te": kp["te"], "idx": kp["idx"], "prep": f["prep"], "V": f["V"]}  # the record importance_cell reads, with the permuted preparation
            s_, _, _ = B.quiet(R.importance_cell, B.predictor(mdl, m_), kpm, f["names"], f["sel"], fam, xs, {"config": "c4b_refit_median", "model": B.MODEL_NAME[mdl], "receiver": rx, "held_out": held})  # the forest script's importance of the median permuted model
            ib += s_  # kept
        ibd = pd.DataFrame(ib)  # the median models' importance
        classes[mdl] = B.classification(ip, ibd, ta, b4, med, mdl, CELLS_C4B)  # the forest script's classification with the selection check
        imps[mdl] = (ip, ibd)  # kept for writing
    return b4, v, classes, imps  # the four


def plot_label_sensitivity(tab: pd.DataFrame) -> None:  # function: the figure, F1 at the three operating points, pooled against per-receiver labels, per receiver and model
    names = {"SBF_02": "mosaic-X5", "SBF_03": "mosaic-H", "UBX_01": "u-blox 01"}  # dict: stream -> readable name
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), sharey=True)  # a 2 x 3 grid: models in rows, receivers in columns
    x = np.arange(len(SHARES))  # one position per share
    style = [("pooled", "pooled", "#0072B2", "o", "pooled labels, pooled models (published)"), ("per_receiver", "per_receiver", "#D55E00", "s", "per-receiver labels, per-receiver models"),  # list of (labels, models, colour, marker, legend text)
             ("per_receiver", "pooled", "#999999", "^", "per-receiver labels, pooled models (no retraining)")]  # list continued
    for i, mdl in enumerate(MODELS):  # for loop over the models (rows)
        for j, rx in enumerate(RXS):  # for loop over the receivers (columns)
            ax = axes[i, j]  # this panel
            for ls, ms, col, mk, lab in style:  # for loop over the three series
                d = tab[(tab.receiver == rx) & (tab.model == mdl) & (tab.label_set == ls) & (tab.models_trained_on == ms) & (tab.point != "0.5")].sort_values("point")  # the three shares
                ax.plot(x, d.f1_iqm.to_numpy(float), marker=mk, color=col, label=lab)  # interquartile-mean F1 (folds 2.6.2 and 2.6.4) against the share
            ax.set_xticks(x, [f"{s:g}" for s in SHARES])  # the shares as tick labels
            ax.set_title(f"{names[rx]}, {'logistic regression' if mdl == 'lr' else 'random forest'}", fontsize=9)  # conditional expression: the panel's title
            ax.set_ylim(0, 1.02)  # F1 runs from 0 to 1
            ax.grid(alpha=0.3)  # light grid
            if j == 0:  # if: the left column
                ax.set_ylabel("F1 (IQM over the folds 2.6.2 and 2.6.4)")  # the axis name
            if i == 1:  # if: the bottom row
                ax.set_xlabel("share of clean windows in alarm (operating point)")  # the axis name
    axes[0, 0].legend(fontsize=7, loc="lower right")  # one legend
    fig.tight_layout()  # spacing
    GRAPH.mkdir(parents=True, exist_ok=True)  # the figure folder (needed under a --root of the second pass)
    save_fig(fig, GRAPH / f"{FIGURE}.png")  # the project's byte-stable saver (it closes the figure)


def main() -> None:  # function: the label sensitivity from start to finish
    t0 = time.time()  # the clock
    B.use_labels("final")  # the scoring module points at the final label set (fold policy; its pooled intervals only feed latency, not read here)
    wc10 = L.window_dict()  # the feature script's window dictionary at w = 10
    lab = pd.read_csv(PROC / "data_checks" / "labels_per_receiver_1hz.csv", dtype={"test_code": str}, keep_default_na=False)  # both label sets per second
    pub = published_tables()  # the published numbers
    # ---- a: cells
    cells_p = B.load_cells(B.FINAL, 10, True)  # P's cells: final windows, scoring and training labels (2.6.12 dropped at read)
    same = relabel(cells_p, lab, "pooled")  # the same cells relabelled from the data_checks file's pooled columns
    check("relabel(pooled) = P's own cells (labels and training labels, every window)", all(same[rx][0].equals(cells_p[rx][0]) for rx in RXS), f"{sum(len(c[0]) for c in cells_p.values())} windows")  # the relabeller is exact
    cells_r = relabel(cells_p, lab, "per_receiver")  # the per-receiver cells
    cnt = []  # window counts per label set
    for name, cl in (("pooled", cells_p), ("per_receiver", cells_r)):  # for loop over the two sets
        for rx, (idx, _) in cl.items():  # for loop over the receivers
            c = idx.groupby(["test_code"]).label.value_counts().unstack(fill_value=0)  # windows per session and label
            t = idx.groupby(["test_code"]).tlabel.value_counts().unstack(fill_value=0).add_prefix("train_")  # windows per session and training label
            cnt.append(c.join(t).reset_index().assign(label_set=name, receiver=rx))  # kept
    write_csv(pd.concat(cnt, ignore_index=True).fillna(0), "window_counts.csv")  # written
    # ---- reproduction gates on the pooled set
    Pp = B.run_choices(B.d1_cells(cells_p, wc10), wc10, "P", False)  # P's outer run, run again on the pooled cells
    for mdl in MODELS:  # for loop over the models
        a = Pp["predictions"][Pp["predictions"].model == mdl]  # the run again's predictions
        x = a[["receiver", "held_out", "sec", "label", "p"]].merge(pub["pred"][mdl][["receiver", "held_out", "sec", "label", "p"]], on=["receiver", "held_out", "sec"], suffixes=("", "_pub"), validate="one_to_one")  # aligned with the published file
        check(f"P run again on the pooled cells = published {mdl} predictions (labels and probabilities bitwise)", len(x) == len(a) == len(pub["pred"][mdl]) and bool((x.label == x.label_pub).all()) and bool(np.array_equal(x.p.to_numpy(), x.p_pub.to_numpy())), f"{len(x)} windows")  # the gate
    tp_ = thresholds_from_nested(pub["nested"])  # the training script's rule on P's committed nested predictions
    pp_ = pub["points"][pub["points"].detector.isin(MODELS)][["receiver", "outer", "detector", "share", "param"]]  # the agent experiment's committed lr and rf points
    mm = tp_.merge(pp_, on=["receiver", "outer", "detector", "share"], suffixes=("", "_pub"), validate="one_to_one")  # aligned
    check("threshold rule on the agent experiment's committed nested predictions = deployable_points (lr, rf), every point", len(mm) == len(pp_) == len(tp_) and bool(np.array_equal(mm.param.to_numpy(), mm.param_pub.to_numpy(), equal_nan=True)), f"{len(mm)} points")  # the gate
    Np = A11.quiet(A11.nested_keep, B.d1_cells(cells_p, wc10), wc10, A11.P_CFG)  # the adopted-pipeline script's nested scheme run again on the pooled cells
    kn = ["model", "receiver", "outer_held_out", "held_out", "sec"]  # the keys of a nested prediction
    xn = Np["predictions"][kn + ["label", "p"]].merge(pub["nested"][kn + ["label", "p"]], on=kn, suffixes=("", "_pub"), validate="one_to_one")  # aligned with the agent experiment's committed file
    check("the adopted-pipeline script's nested_keep run again on the pooled cells = nested_predictions.csv.gz (labels and probabilities bitwise)", len(xn) == len(pub["nested"]) == len(Np["predictions"]) and bool((xn.label == xn.label_pub).all()) and bool(np.array_equal(xn.p.to_numpy(), xn.p_pub.to_numpy())), f"{len(xn)} inner predictions")  # the gate
    # ---- b: P on the per-receiver labels, nested, points
    Pr = B.run_choices(B.d1_cells(cells_r, wc10), wc10, "PR", True)  # outer run, every fit kept for the importance and the controls
    check("per-receiver P: failed fits", Pr["fails"].empty, f"{len(Pr['fails'])} failed fits")  # no fit may fail
    for mdl in MODELS:  # for loop over the models
        write_gz(Pr["predictions"][Pr["predictions"].model == mdl], f"pr_predictions_{mdl}.csv.gz")  # held-out probabilities
    write_csv(Pr["metrics"], "pr_fold_metrics.csv")  # per-fold figures at 0.5 (choices_lib's rows)
    write_csv(Pr["selection"], "pr_selection.csv")  # the columns each fit selected
    Nr = A11.quiet(A11.nested_keep, B.d1_cells(cells_r, wc10), wc10, {**A11.P_CFG, "run_id": "PR"})  # the adopted-pipeline script's nested run (inner leave-one-session-out, the decision's unscreened fits) on the per-receiver labels
    write_gz(Nr["predictions"].drop(columns=["dec"]), "pr_nested_predictions.csv.gz")  # inner out-of-fold predictions (the decision value column left out)
    write_csv(Nr["fails"], "pr_nested_fit_failures.csv")  # unfittable inner folds
    pts_r = thresholds_from_nested(Nr["predictions"])  # the per-receiver operating points
    write_csv(pts_r, "pr_points.csv")  # written
    # ---- c and d: scoring, every label set x training labels x model x point
    labmaps = {name: lab[lab.test_code != "2.6.12"].rename(columns={"test_code": "held_out", col: "lab"})[["receiver", "held_out", "sec", "lab"]] for name, (col, _) in LABEL_SETS.items()}  # dict comprehension: per label set, (receiver, fold, second) -> label
    preds = {"pooled": pub["pred"], "per_receiver": {m: Pr["predictions"][Pr["predictions"].model == m].reset_index(drop=True) for m in MODELS}}  # dict: models trained on -> model -> held-out predictions
    points = {"pooled": pub["points"], "per_receiver": pts_r}  # dict: models trained on -> their operating points
    rows, folds = [], []  # the summary rows; the per-fold rows
    for ls in LABEL_SETS:  # for loop over the label sets the predictions are scored on
        for tr in LABEL_SETS:  # for loop over the label sets the models were trained on
            for mdl in MODELS:  # for loop over the models
                for pt in POINTS:  # for loop over the operating points
                    pr_ = preds[tr][mdl][["receiver", "held_out", "sec", "p"]]  # the held-out probabilities
                    m_, lt_, sc_ = score_fold_rows(pr_, labmaps[ls], thr_map(points[tr], mdl, pt))  # the regression script's fold evaluation
                    ident = {"label_set": ls, "models_trained_on": tr, "model": mdl, "point": pt}  # dict literal: the row's identity
                    folds.append(m_[["receiver", "held_out", "threshold", "recall", "recall_jam_spoof", "recall_spoof", "precision", "f1", "clean", "clean_in_alarm", "clean_in_alarm_share", "tp", "fn", "fp", "tn"]].assign(**ident))  # dict unpacking into assign: the fold rows
                    rows.append(summarise(m_, lt_, sc_).assign(**ident))  # the per-receiver summary
    F = pd.concat(folds, ignore_index=True)  # every fold row
    T = pd.concat(rows, ignore_index=True)  # every summary row
    write_csv(F, "score_folds.csv")  # written
    # ---- gates on the scorer: published pooled numbers reproduced from the published predictions
    base = T[(T.label_set == "pooled") & (T.models_trained_on == "pooled")]  # the published configuration rescored
    for mdl in MODELS:  # for loop over the models
        pb = pub["pooled"][mdl].set_index("receiver")  # the published Table 10 rows
        x = base[(base.model == mdl) & (base.point == "0.5")].set_index("receiver")  # rescored at 0.5
        cols = ["recall_pooled", "recall_jam_spoof_pooled", "recall_spoof_pooled", "precision_pooled", "f1_pooled", "clean_in_alarm_share_pooled", "auroc_pooled_predictions"]  # the compared columns
        ok = all(np.isclose(float(x.loc[rx, c]), float(pb.loc[rx, c]), rtol=0, atol=1e-12, equal_nan=True) for rx in RXS for c in cols)  # generator in all(): every figure equal to 1e-12
        check(f"scorer at 0.5 = published {mdl} pooled table ({'logistic_regression' if mdl == 'lr' else 'random_forest'}), seven figures x three receivers", ok, "")  # the gate
    prn = pub["point_runs"][pub["point_runs"].detector.isin(MODELS)]  # the agent experiment's per-fold decision-mode runs of lr and rf
    fb = F[(F.label_set == "pooled") & (F.models_trained_on == "pooled") & (F.point != "0.5")].assign(share=lambda d: d.point.astype(float))  # assign with a lambda: the rescored folds at the shares
    xm = fb.merge(prn, left_on=["receiver", "held_out", "model", "share"], right_on=["receiver", "outer", "detector", "share"], validate="one_to_one")  # aligned with the agent experiment's runs
    ok = len(xm) == len(prn) and all(np.allclose(xm[a].astype(float), xm[b].astype(float), rtol=0, atol=1e-12, equal_nan=True) for a, b in (("recall", "dec_recall"), ("precision", "dec_precision"), ("f1", "dec_f1"), ("clean_in_alarm_share", "dec_clean_share")))  # generator in all(): four figures equal
    check("scorer at the shares = point_runs (lr, rf; recall, precision, F1, clean share per fold)", ok, f"{len(xm)} runs")  # the gate
    ag = pub["aggregate"][pub["aggregate"].detector.isin(MODELS)]  # the agent experiment's interquartile means of lr and rf
    names = {"recall_iqm": "recall", "precision_iqm": "precision", "f1_iqm": "F1", "clean_share_iqm": "clean-window alarm share"}  # dict: this table's column -> the agent experiment's metric name
    bt = base[base.point != "0.5"].assign(share=lambda d: d.point.astype(float))  # rescored IQMs at the shares
    # all() over a generator: every the agent experiment IQM of lr and rf equals the rescored one to 1e-12
    ok = all(np.isclose(float(bt[(bt.receiver == r.receiver) & (bt.model == r.detector) & np.isclose(bt.share, r.share)][c].iloc[0]), float(r.iqm), rtol=0, atol=1e-12, equal_nan=True)
             for c, nm in names.items() for r in ag[ag.metric == nm].itertuples())  # generator in all(): every IQM equal (expression continued from the line above)
    check("scorer IQMs = aggregate (lr, rf; recall, precision, F1, clean share)", ok, f"{len(ag[ag.metric.isin(names.values())])} values")  # the gate
    if FAILED:  # if: a reproduction gate failed
        pd.DataFrame(CHECKS).to_csv(OUT / "checks.csv", index=False, lineterminator="\n")  # the checks so far
        raise SystemExit("STOP: a reproduction gate failed; no per-receiver result is reported")  # stop
    # ---- published numbers beside (read from their files): Table 10 at 0.5, the agent experiment's IQMs at the shares
    pubcols = []  # one dict per receiver, model and point
    for mdl in MODELS:  # for loop over the models
        for rx in RXS:  # for loop over the receivers
            pb = pub["pooled"][mdl][pub["pooled"][mdl].receiver == rx].iloc[0]  # Table the forest script's row
            pubcols.append({"receiver": rx, "model": mdl, "point": "0.5", "published_recall": pb.recall_pooled, "published_f1": pb.f1_pooled, "published_clean_share": pb.clean_in_alarm_share_pooled, "published_source": f"{'logistic_regression' if mdl == 'lr' else 'random_forest'}/{'logistic_regression' if mdl == 'lr' else 'random_forest'}_pooled_primary.csv (pooled over folds)"})  # the 0.5 row
            for s in SHARES:  # for loop over the shares
                a = ag[(ag.receiver == rx) & (ag.detector == mdl) & np.isclose(ag.share, s)].set_index("metric").iqm  # the agent experiment's IQMs of the cell
                pubcols.append({"receiver": rx, "model": mdl, "point": str(s), "published_recall": a["recall"], "published_f1": a["F1"], "published_clean_share": a["clean-window alarm share"], "published_source": "rl_agents/evaluation/aggregate.csv (IQM over folds)"})  # the share row
    T = T.merge(pd.DataFrame(pubcols), on=["receiver", "model", "point"], how="left")  # the published numbers beside every row of the same receiver, model and point
    T = T[["receiver", "label_set", "models_trained_on", "model", "point"] + [c for c in T.columns if c not in ("receiver", "label_set", "models_trained_on", "model", "point")]]  # identity columns first
    rank = {"SBF_02": 0, "SBF_03": 1, "UBX_01": 2, "pooled": 0, "per_receiver": 1, "lr": 0, "rf": 1, "0.5": 0, "0.02": 1, "0.05": 2, "0.1": 3}  # dict: the reading order of every identity value
    T = T.sort_values(["receiver", "label_set", "models_trained_on", "model", "point"], key=lambda c: c.map(rank), kind="stable").reset_index(drop=True)  # sort_values with a key: a fixed, readable order
    write_csv(T, "score_table.csv")  # every receiver in one file
    for rx in RXS:  # for loop over the receivers: one table each, as the specification asks
        write_csv(T[T.receiver == rx], f"score_table_{rx}.csv")  # written
    # ---- RQ1 on the per-receiver labels
    b4, v, classes, imps = classify_run(Pr, wc10, "PR")  # C4b-refit, the session control, importance and classification
    write_csv(b4, "pr_c4b_within_session.csv")  # every permutation
    write_csv(v, "pr_r3_verdicts.csv")  # the session control
    for mdl in MODELS:  # for loop over the models
        write_csv(classes[mdl], f"pr_classification_{mdl}.csv")  # the classification
        write_csv(imps[mdl][0], f"pr_importance_{mdl}.csv")  # the true model's importance
        write_csv(imps[mdl][1], f"pr_importance_c4b_{mdl}.csv")  # the median permuted models' importance
    cmp_rows = []  # one dict per model, cell and family
    for mdl in MODELS:  # for loop over the models
        a = pub["class"][mdl][["receiver", "held_out", "family", "class"]].rename(columns={"class": "class_pooled"})  # the published classification
        b = classes[mdl][["receiver", "held_out", "family", "class"]].rename(columns={"class": "class_per_receiver"})  # the per-receiver one
        cmp_rows.append(a.merge(b, on=["receiver", "held_out", "family"], how="outer").assign(model=mdl))  # side by side
    C = pd.concat(cmp_rows, ignore_index=True)  # one table
    C["changed"] = C.class_pooled != C.class_per_receiver  # whether the family's class moved
    write_csv(C, "rq1_classification_comparison.csv")  # written
    r3 = pub["r3"][["model", "receiver", "held_out", "verdict"]].merge(v[["model", "receiver", "held_out", "verdict"]], on=["model", "receiver", "held_out"], suffixes=("_pooled", "_per_receiver"), how="outer")  # the session control side by side
    write_csv(r3, "rq1_r3_comparison.csv")  # written
    vrows = []  # RQ1 verdict per receiver and model: the families that are attack evidence, per cell
    for (mdl, rx), g in C.groupby(["model", "receiver"], sort=False):  # for loop over the models and receivers
        ev_p = sorted(f"{h}:{f}" for h, f in zip(g.held_out[g.class_pooled == "attack evidence"], g.family[g.class_pooled == "attack evidence"]))  # sorted "fold:family" texts, pooled
        ev_r = sorted(f"{h}:{f}" for h, f in zip(g.held_out[g.class_per_receiver == "attack evidence"], g.family[g.class_per_receiver == "attack evidence"]))  # the same, per receiver
        fam_p = sorted(set(g.family[g.class_pooled == "attack evidence"]))  # the families that are evidence in any cell, pooled
        fam_r = sorted(set(g.family[g.class_per_receiver == "attack evidence"]))  # the same, per receiver
        # vrows.append adds a dict: the RQ1 verdict of one model and receiver under both label sets
        vrows.append({"model": mdl, "receiver": rx, "evidence_families_pooled": ";".join(fam_p), "evidence_families_per_receiver": ";".join(fam_r), "families_changed": fam_p != fam_r,
                      "evidence_cells_pooled": ";".join(ev_p), "evidence_cells_per_receiver": ";".join(ev_r), "cells_changed": ev_p != ev_r})  # the row
    V = pd.DataFrame(vrows)  # the RQ1 verdicts
    write_csv(V, "rq1_verdicts.csv")  # written
    # ---- e: the decision rule
    d5 = T[(T.point == "0.05") & (T.models_trained_on == T.label_set)]  # each label set with its own models (pooled published; per-receiver retrained)
    dr = []  # one dict per receiver
    for rx in RXS:  # for loop over the receivers
        g = d5[d5.receiver == rx].set_index(["label_set", "model"]).f1_iqm  # F1 (IQM) at 0.05
        rank_p = "lr>rf" if g[("pooled", "lr")] > g[("pooled", "rf")] else ("rf>lr" if g[("pooled", "lr")] < g[("pooled", "rf")] else "tie")  # nested conditional expression: the pooled ranking
        rank_r = "lr>rf" if g[("per_receiver", "lr")] > g[("per_receiver", "rf")] else ("rf>lr" if g[("per_receiver", "lr")] < g[("per_receiver", "rf")] else "tie")  # the per-receiver ranking
        dlr, drf = g[("per_receiver", "lr")] - g[("pooled", "lr")], g[("per_receiver", "rf")] - g[("pooled", "rf")]  # tuple assignment: F1 change per model
        rq1 = bool(V[V.receiver == rx].families_changed.any())  # any model's evidence families changed
        # dr.append adds a dict: the decision rule's quantities and verdicts for one receiver
        dr.append({"receiver": rx, "f1_lr_pooled": g[("pooled", "lr")], "f1_lr_per_receiver": g[("per_receiver", "lr")], "f1_rf_pooled": g[("pooled", "rf")], "f1_rf_per_receiver": g[("per_receiver", "rf")],
                   "delta_f1_lr": dlr, "delta_f1_rf": drf, "rank_pooled": rank_p, "rank_per_receiver": rank_r, "rank_swapped": rank_p != rank_r, "f1_moved_over_0.10": bool(abs(dlr) > 0.10 or abs(drf) > 0.10),  # continued: more fields of the row
                   "rq1_families_changed": rq1, "limitation_branch": bool(rank_p != rank_r or abs(dlr) > 0.10 or abs(drf) > 0.10 or rq1)})  # the row
    D = pd.DataFrame(dr)  # the decision table
    write_csv(D, "decision_rule.csv")  # written
    branch = "LIMITATION: a criterion is met; reported in 6.x with its numbers" if D.limitation_branch.any() else "ROBUSTNESS ONLY: no criterion met; reported beside the primary"  # conditional expression: the branch
    check("decision rule (fixed before any result)", None, branch + " | " + "; ".join(f"{r.receiver}: rank {r.rank_pooled}->{r.rank_per_receiver}, dF1 lr {r.delta_f1_lr:+.3f} rf {r.delta_f1_rf:+.3f}, RQ1 changed {r.rq1_families_changed}" for r in D.itertuples()))  # reported
    write_csv(pub["verdicts"].assign(note="published P9 (pooled labels); not formed under per-receiver labels: the agents' points would need new inner agent runs, which the decision rule excludes"), "p9_published.csv")  # P9 beside, as published
    plot_label_sensitivity(T)  # the figure
    pd.DataFrame(CHECKS).to_csv(OUT / "checks.csv", index=False, lineterminator="\n")  # the checks
    print(D.to_string(index=False))  # the decision table for the run log
    print(f"done in {time.time() - t0:.0f} s; failed checks: {len(FAILED)}")  # run time


if __name__ == "__main__":  # if: true only when the file is run directly
    main()  # runs the sensitivity
    exit_if_failed(FAILED, "8_data_checks/04_label_sensitivity.py")  # the helpers module: status 1 when any check failed
