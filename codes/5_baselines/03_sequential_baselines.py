#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
Variants and sensitivities of the logistic regression baseline (reported, never selected).

Every variant is one change from the primary of 5_baselines/01_logistic_regression.py and runs the pipeline (columns with no training value
dropped -> median imputation -> robust scaling -> sign(z) log1p(|z|) -> Pearson pruning -> MI top 36 -> newton-cg, tol 1e-6,
max_iter 100,000, L2, class_weight balanced); any warning other than the three Accelerate matmul messages, and any failed
gradient check (max |gradient| per unit sample weight > 1e-4), stops the run. Same folds, per receiver;
2.6.12 sealed (the regression script's reader).
  feature sets  no_residual (orbit-free set; orbit-free indicator only); known_antenna (known-antenna set; orbit-free indicator
                and a known-block indicator built by the same rule); no_circularity (main screened list without the circularity
                candidates and without the self-block indicator); shift_invariant (std, skew and kurt of the main screened list
                only, no indicator); main_plus_geometry (main screened list plus the geometry block's level statistics of the
                orbit-free and self origins); screen_no_floor (the no-floor screened list); unscreened (the whole main set)
  windows       w = 1, 2, 3, 5 (main screened list of that length)
  regulariser   C = 0.01, 0.1, 10; no_selection (every screened column and both indicators: no Pearson pruning, no MI)
  alarm rule    persist_3of5: an alarm at a window when at least 3 of the windows in the 5 s ending at it have p >= 0.5
                (causal; the primary's probabilities, so AUROC is the primary's)
C4b (within-session permutation, the regression script's seeds and count) is added to shift_invariant and no_circularity (report only).
Report-only diagnosis of the 2.6.4 spoofing-phase miss on the mosaic-X5 and mosaic-H: for the top 10 features by
|coefficient| of that fold's primary model, the percentile of every missed attack_spoof window in the training clean (2.6.2) and
training attack distributions, and whether it lies closer to the clean or to the attack training median; the primary model is
reproduced, not refitted, and is checked to equal the published coefficients exactly.

Rule for naming a variant (fixed before any variant was computed): per receiver, a variant moves beyond the primary's across-fold
spread when its pooled recall, its median AUROC or its pooled share of clean windows in alarm differs from the primary's by more
than the standard deviation (ddof 0) of the primary's per-fold values of that quantity; per-fold differences are in the long table.

Reads   what the regression script reads; Data/processed/logistic_regression_registered/logistic_regression_predictions_primary.csv.gz, logistic_regression_coefficients_primary.csv, logistic_regression_c4b_within_session.csv
Writes  Data/processed/logistic_regression_registered/logistic_regression_variants_*.csv(.gz), Data/processed/checks/logistic_regression_variants_checks.csv, Graph/logistic_regression_*.png
"""

from __future__ import annotations  # future import: type hints kept as text (a convention across the project's scripts)

import gzip  # standard-library gzip, to read the published compressed predictions
import importlib.util  # standard-library module loader, for modules whose names start with a digit
import io  # standard-library in-memory text streams
import sys  # standard-library system access: sys.modules and the exit status
import time  # standard-library clock, for the run-time line
from pathlib import Path  # Path objects for building file paths

import numpy as np  # NumPy: arrays
import pandas as pd  # pandas: tables

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
spec = importlib.util.spec_from_file_location("logistic_regression_for_sequential", PROJECT / "codes" / "5_baselines" / "01_logistic_regression.py")  # where 09 is
M = importlib.util.module_from_spec(spec)  # an empty module object
sys.modules["logistic_regression_for_sequential"] = M  # registered, so its own imports resolve
spec.loader.exec_module(M)  # the regression script's top level (its main() is guarded and not run); it also sets MPLCONFIGDIR and the Agg backend
import matplotlib.pyplot as plt  # the pyplot interface (after 09 chose the file-only backend)

S, SEED, OUT, GRAPH, CHECK_DIR = M.S, M.SEED, M.OUT, M.GRAPH, M.CHECK_DIR  # the screen script, the seed and the folders, from the regression script
RXS, RXN, SESSIONS = M.RXS, M.RXN, M.SESSIONS  # receivers, their names, the five folds
CHANGE = {"std", "skew", "kurt"}  # the shift-invariant statistics
PERSIST_K, PERSIST_N = 3, 5  # the persistence rule: 3 of the windows in the 5 s ending at a window
KNOWN_IND = "ind_missing_known"  # the known-block indicator of the known-antenna variant
CHECKS: list = []  # list: the checks, written to logistic_regression_variants_checks.csv


def check(name: str, expected: str, found, ok: bool | None) -> None:  # function: records one check and prints it
    CHECKS.append({"check": name, "expected": expected, "found": found, "ok": "" if ok is None else bool(ok)})  # append a dict row
    print(f"  [{'ok' if ok else ('FAIL' if ok is False else '--')}] {name}: expected {expected}; found {found}")  # progress line


# ---------------------------------------------------------------- the variants

def variants(wc: pd.DataFrame, fl: pd.DataFrame) -> list:  # function: every variant as (name, family, w, candidate function, blocks, C, selection, known indicator)
    circ = {w: S.groups_of(wc, w)["circularity"] for w in S.WINDOWS}  # dict comprehension: the circularity candidates' window columns per length
    stat = dict(zip(wc.column + "@" + wc.w.astype(str), wc.stat))  # dict: "column@w" -> statistic
    geo = {w: list(wc[(wc.w == w) & (wc.geometry_ablation == 1) & (wc.origin != "known")].column) for w in S.WINDOWS}  # dict comprehension: geometry level statistics of the main variant per length
    main = lambda w: (lambda rx, h: M.cand_list(fl, w, rx, h))  # nested lambda: the main screened list at length w
    B = M.BLOCKS  # the primary's indicator blocks
    return [  # list literal: one tuple per variant, in the order of the specification
        ("primary", "primary", 10, main(10), B, 1.0, "mi", False),  # the primary, run again by this script (checked against the regression script)
        ("no_residual", "feature set", 10, lambda rx, h: M.cand_list(fl, 10, rx, h, "no_residual"), ("orbitfree",), 1.0, "mi", False),  # the orbit-free set
        ("known_antenna", "feature set", 10, lambda rx, h: M.cand_list(fl, 10, rx, h, "known_antenna"), ("orbitfree",), 1.0, "mi", True),  # the known-antenna set
        ("no_circularity", "feature set", 10, lambda rx, h: [c for c in M.cand_list(fl, 10, rx, h) if c not in circ[10]], ("orbitfree",), 1.0, "mi", False),  # without the circularity candidates and the self indicator
        ("shift_invariant", "feature set", 10, lambda rx, h: [c for c in M.cand_list(fl, 10, rx, h) if stat[f"{c}@10"] in CHANGE], (), 1.0, "mi", False),  # std, skew, kurt only
        ("main_plus_geometry", "feature set", 10, lambda rx, h: M.cand_list(fl, 10, rx, h) + geo[10], B, 1.0, "mi", False),  # plus the geometry level statistics
        ("screen_no_floor", "feature set", 10, lambda rx, h: M.cand_list(fl, 10, rx, h, "main", "screened_no_floor"), B, 1.0, "mi", False),  # the no-floor screen
        ("unscreened", "feature set", 10, lambda rx, h: M.cand_list(fl, 10, rx, h, "main", "main"), B, 1.0, "mi", False),  # the whole main set
        ("w1", "window", 1, main(1), B, 1.0, "mi", False),  # w = 1
        ("w2", "window", 2, main(2), B, 1.0, "mi", False),  # w = 2
        ("w3", "window", 3, main(3), B, 1.0, "mi", False),  # w = 3
        ("w5", "window", 5, main(5), B, 1.0, "mi", False),  # w = 5
        ("C0.01", "regulariser", 10, main(10), B, 0.01, "mi", False),  # C = 0.01
        ("C0.1", "regulariser", 10, main(10), B, 0.1, "mi", False),  # C = 0.1
        ("C10", "regulariser", 10, main(10), B, 10.0, "mi", False),  # C = 10
        ("no_selection", "regulariser", 10, main(10), B, 1.0, "none", False),  # no pruning, no MI
    ]


def candidates(X: pd.DataFrame, cand: list, wc: pd.DataFrame, w: int, blocks: tuple, known: bool) -> pd.DataFrame:  # function: the regression script's candidates and indicators, plus the known-block indicator for the known-antenna variant
    V = M.with_indicators(X, cand, wc, w, blocks)  # the regression script's rule for the named blocks
    if known:  # if: the known-antenna variant
        x = wc[wc.w == w].set_index("column")  # dictionary rows
        cols = [c for c in cand if x.at[c, "part"] == "known" and x.at[c, "stat"] not in M.UNDEFINED_STATS]  # list comprehension: the known block's columns other than skewness and kurtosis
        V[KNOWN_IND] = X[cols].isna().any(axis=1).astype(float).to_numpy()  # 1 when any is missing (the regression script's rule)
    return V  # candidates and indicators


def run_variant(v: tuple, wc: pd.DataFrame, cells: dict, keep: bool = False) -> dict:  # function: one variant over every receiver and fold
    name, family, w, cand_fn, blocks, C, sel, known = v  # tuple unpacking
    preds, metrics, lat, selrep, keepd = [], [], [], [], {}  # outputs
    for rx in RXS:  # for loop over the receivers
        idx, X = cells[(w, rx)]  # the windows of this length (2.6.12 dropped by the regression script's reader)
        y = M.target(idx)  # target
        for held, train in S.folds():  # for loop over the folds
            tr, te = M.fold_rows(idx, held, train)  # rows
            V = candidates(X, cand_fn(rx, held), wc, w, blocks, known)  # candidates and indicators
            prep = M.prepare(V, tr)  # the regression script's label-free steps, compressed
            if sel == "none":  # if: no feature selection
                prep["kept"] = list(range(len(prep["names"])))  # every column, Pearson pruning set aside
            res = M.fit_fold(prep, y, tr, te, C=C, select=(sel == "mi"), tag={"config": f"variant_{name}", "mi_seed": SEED, "receiver": rx, "held_out": held, "perm": -1})  # selection, fit (strict rule), held-out probabilities
            row, lr = M.evaluate_fold(idx, te, res["p"], held)  # the scoring module's figures
            row.update({"variant": name, "family": family, "receiver": rx, "w": w, "C": C, "n_columns_fitted": len(res["sel"]), "n_iter": res["n_iter"], "max_abs_grad_per_weight": res["grad"]})  # identity and fit
            metrics.append(row)  # kept
            lat += [{**r, "variant": name, "receiver": rx} for r in lr]  # list comprehension with dict unpacking: latency rows
            preds.append(pd.DataFrame({"variant": name, "receiver": rx, "held_out": held, "sec": idx.sec.to_numpy()[te], "label": idx.label.to_numpy()[te], "p": res["p"], "alarm": res["p"] >= M.THR}))  # dict literal: held-out probabilities
            selrep.append({"variant": name, "receiver": rx, "held_out": held, "candidates": V.shape[1], "fitted": len(res["sel"]), "selected_columns": ";".join(res["names"])})  # dict literal: what was fitted
            if keep:  # if: the caller needs the prepared fold and the fit
                keepd[(rx, held)] = (prep, tr, te, y, idx, res)  # tuple: kept
    out = {"predictions": pd.concat(preds, ignore_index=True), "metrics": pd.DataFrame(metrics), "latency": pd.DataFrame(lat), "selection": pd.DataFrame(selrep), "keep": keepd}  # dict literal: the tables
    out["pooled"] = M.pooled(out["metrics"], out["latency"], out["predictions"]).assign(variant=name)  # the regression script's pooled figures per receiver
    print(f"  {name}: done")  # progress line
    return out  # everything


def k_of_n(sec: np.ndarray, raw: np.ndarray, k: int = PERSIST_K, n: int = PERSIST_N) -> np.ndarray:  # function: alarm where at least k of the windows in the n seconds ending at a window alarm (causal)
    cs = np.concatenate([[0], np.cumsum(raw.astype(int))])  # cumulative alarm count, with a leading zero
    lo = np.searchsorted(sec, sec - (n - 1), side="left")  # first window inside the n-second span ending at each window
    return (cs[np.arange(1, len(sec) + 1)] - cs[lo]) >= k  # alarms in the span, at least k


def persistence(prim: dict) -> dict:  # function: the primary's probabilities under the 3-of-5 rule
    preds, metrics, lat = [], [], []  # outputs
    pm = prim["metrics"].set_index(["receiver", "held_out"])  # the primary's rows
    for (rx, held), d in prim["predictions"].groupby(["receiver", "held_out"], sort=False):  # for loop over folds in the primary's order
        idx = d[["sec", "label"]].reset_index(drop=True)  # held-out windows in time order (as 09 evaluated them)
        al = k_of_n(d.sec.to_numpy(np.int64), d.alarm.to_numpy())  # the persistence alarms
        row, lr = M.evaluate_fold(idx, np.arange(len(idx)), al.astype(float), held)  # the scoring module's figures on the 0/1 alarms
        for k in ("auroc", "average_precision", "p_equal_threshold"):  # for loop over the threshold-free figures
            row[k] = pm.at[(rx, held), k]  # the primary's (persistence does not change the probabilities)
        row.update({"variant": "persist_3of5", "family": "alarm rule", "receiver": rx, "w": 10, "C": 1.0})  # identity
        metrics.append(row)  # kept
        lat += [{**r, "variant": "persist_3of5", "receiver": rx} for r in lr]  # list comprehension: latency rows
        preds.append(d.assign(variant="persist_3of5", alarm=al))  # alarms replaced
    out = {"predictions": pd.concat(preds, ignore_index=True), "metrics": pd.DataFrame(metrics), "latency": pd.DataFrame(lat)}  # dict literal: the tables
    out["pooled"] = M.pooled(out["metrics"], out["latency"], out["predictions"]).assign(variant="persist_3of5")  # pooled
    return out  # everything


# ---------------------------------------------------------------- the variant table and the spread rule

def variant_table(allm: pd.DataFrame, allp: pd.DataFrame, c4b: pd.DataFrame) -> pd.DataFrame:  # function: one row per variant and receiver, with the primary's spread rule applied
    rows = []  # results
    for rx in RXS:  # for loop over the receivers
        pm = allm[(allm.variant == "primary") & (allm.receiver == rx)]  # the primary's folds
        sd = {"recall": float(pm.recall.std(ddof=0)), "auroc": float(pm.auroc.dropna().std(ddof=0)), "clean": float(pm.clean_in_alarm_share.dropna().std(ddof=0))}  # dict literal: the primary's across-fold standard deviations
        pp = allp[(allp.variant == "primary") & (allp.receiver == rx)].iloc[0]  # the primary's pooled row
        for v, g in allp[allp.receiver == rx].groupby("variant", sort=True, observed=True):  # for loop over the variants in run order (the categorical's order)
            r = g.iloc[0]  # its pooled row
            cs = r.clean_in_alarm / r.clean_windows  # pooled share of clean windows in alarm
            d = {"recall": r.recall_pooled - pp.recall_pooled, "auroc": r.auroc_median - pp.auroc_median, "clean": cs - pp.clean_in_alarm / pp.clean_windows}  # dict literal: differences from the primary
            moved = [k for k in d if np.isfinite(d[k]) and abs(d[k]) > sd[k]] if v != "primary" else []  # list comprehension: quantities beyond the spread
            fam = allm[(allm.variant == v)].family.iloc[0]  # the family
            cb = c4b[(c4b.variant == v) & (c4b.receiver == rx)] if len(c4b) else c4b  # conditional expression: its C4b rows
            row = {"variant": v, "family": fam, "receiver": rx, "recall_pooled": r.recall_pooled, "recall_median": r.recall_median_over_folds, "recall_min": r.recall_min, "recall_max": r.recall_max,  # dict literal: recall
                   "recall_jam_spoof": r.recall_jam_spoof_pooled, "recall_spoof": r.recall_spoof_pooled, "precision_pooled": r.precision_pooled, "f1_pooled": r.f1_pooled,  # per class, precision and F1 (2 folds)
                   "clean_in_alarm_share": cs, "fa_episodes_count_folds": r.fa_episodes_count_folds, "fa_rate_per_h_2_6_2": r.fa_rate_per_h, "auroc_median": r.auroc_median, "ap_median": r.ap_median,  # false alarms and threshold-free scores
                   "intervals_detected": r.intervals_detected, "latency_median_s": r.latency_median_s, "accuracy_pooled": r.accuracy_pooled,  # latency and accuracy
                   "c4b_auroc_median_2_6_2": float(cb[cb.held_out == "2.6.2"].auroc.median()) if len(cb) else np.nan, "c4b_auroc_median_2_6_4": float(cb[cb.held_out == "2.6.4"].auroc.median()) if len(cb) else np.nan,  # C4b where it exists
                   "d_recall": d["recall"], "d_auroc": d["auroc"], "d_clean_share": d["clean"], "sd_recall_primary": sd["recall"], "sd_auroc_primary": sd["auroc"], "sd_clean_share_primary": sd["clean"],  # differences and the primary's spreads
                   "moved_beyond_spread": ";".join(moved)}  # the quantities that moved
            rows.append(row)  # kept
    return pd.DataFrame(rows)  # the table


# ---------------------------------------------------------------- C4b on two variants

def c4b_variant(name: str, keepd: dict) -> pd.DataFrame:  # function: the regression script's C4b (same seeds, same count) on a variant's prepared folds
    n0 = len(M.FIT_DIAG)  # fit diagnostics before
    preps = {k: v[:5] for k, v in keepd.items()}  # dict comprehension: (prep, tr, te, y, idx) as the regression script's c4b expects
    out = M.c4b(preps).assign(variant=name)  # the regression script's C4b
    for dg in M.FIT_DIAG[n0:]:  # for loop over its fit diagnostics
        dg["config"] = f"c4b_{name}"  # relabelled with the variant
    return out  # rows


# ---------------------------------------------------------------- the 2.6.4 diagnosis (report only)

def diagnosis_264(keepd: dict, wc: pd.DataFrame) -> pd.DataFrame:  # function: where the missed 2.6.4 spoofing windows lie against the training clean and attack distributions, top 10 features
    rows = []  # results
    for rx in ("SBF_02", "SBF_03"):  # for loop over the mosaic-X5 and the mosaic-H
        prep, tr, te, y, idx, res = keepd[(rx, "2.6.4")]  # the primary fold, reproduced
        lab_tr = idx.label.to_numpy()[tr]  # training labels
        lab_te, p = idx.label.to_numpy()[te], res["p"]  # held-out labels and probabilities
        missed = te[(lab_te == "attack_spoof") & (p < M.THR)]  # missed attack_spoof windows (rows)
        top = np.argsort(-np.abs(res["coef"]), kind="stable")[:10]  # the 10 largest |coefficients|
        Zc, Za = prep["Z"][tr[lab_tr == "clean"]], prep["Z"][tr[np.isin(lab_tr, M.ATTACK)]]  # training clean (2.6.2) and training attack rows, all columns
        cols = [res["sel"][j] for j in top]  # their positions in prep
        Zm = prep["Z"][missed][:, cols]  # missed windows on those columns
        cm, am = np.median(Zc[:, cols], axis=0), np.median(Za[:, cols], axis=0)  # training clean and attack medians
        closer_joint = float((np.linalg.norm(Zm - cm, axis=1) < np.linalg.norm(Zm - am, axis=1)).mean())  # share of missed windows closer to the clean median in the 10 columns jointly
        for r, j in enumerate(top):  # for loop over the 10 features
            c = res["sel"][j]  # column position
            x = prep["Z"][missed, c]  # missed windows' values (scaled and compressed, the model's input)
            pc = np.array([(Zc[:, c] <= v).mean() * 100 for v in x])  # list comprehension: percentile in the training clean distribution
            pa = np.array([(Za[:, c] <= v).mean() * 100 for v in x])  # percentile in the training attack distribution
            # dict literal: one feature's summary over the missed windows
            rows.append({"receiver": rx, "rank_abs_coef": r + 1, "column": res["names"][j], "coef": float(res["coef"][j]), "missed_windows": len(missed),
                         "missed_median_value": float(np.median(x)), "train_clean_median": float(np.median(Zc[:, c])), "train_attack_median": float(np.median(Za[:, c])),  # medians on the model's scale
                         "pct_in_clean_median": float(np.median(pc)), "pct_in_attack_median": float(np.median(pa)),  # median percentiles
                         "share_closer_to_clean": float((np.abs(x - np.median(Zc[:, c])) < np.abs(x - np.median(Za[:, c]))).mean()),  # per feature
                         "contribution_median": float(np.median(res["coef"][j] * x)), "share_closer_to_clean_joint_top10": closer_joint})  # coefficient x value, and the joint share
    return pd.DataFrame(rows)  # the table


# ---------------------------------------------------------------- figures

def family_figure(allm: pd.DataFrame, names: list, path: Path, title: str, episodes: bool = False) -> None:  # function: recall and clean windows in alarm per fold against the variant, one column per receiver
    nr = 3 if episodes else 2  # rows: recall, clean share, and episodes for the alarm rule
    fig, ax = plt.subplots(nr, 3, figsize=(16, 3.6 * nr), sharey="row")  # panels
    x = np.arange(len(SESSIONS))  # fold positions
    cols = plt.get_cmap("tab10")  # a qualitative palette
    off = np.linspace(-0.3, 0.3, len(names))  # horizontal offsets so markers do not overlap
    for k, rx in enumerate(RXS):  # for loop over the receivers
        for i, v in enumerate(names):  # for loop over the variants of this family
            d = allm[(allm.variant == v) & (allm.receiver == rx)].set_index("held_out").reindex(SESSIONS)  # its folds in order
            mk = "o" if v == "primary" else "s"  # the primary as a circle
            ax[0, k].plot(x + off[i], d.recall, mk, color=cols(i), label=v)  # recall
            ax[1, k].plot(x + off[i], d.clean_in_alarm_share, mk, color=cols(i), label=v)  # share of clean windows in alarm
            if episodes:  # if: the alarm-rule figure
                ax[2, k].plot(x + off[i], d.fa_episodes.astype(float), mk, color=cols(i), label=v)  # false-alarm episodes
        for r in range(nr):  # for loop over the rows
            ax[r, k].set_xticks(x)  # fold ticks
            ax[r, k].set_xticklabels(SESSIONS)  # fold names
            ax[r, k].set_xlim(-0.6, len(SESSIONS) - 0.4)  # every fold shown, also where a row has no value
        ax[0, k].set_ylim(-0.03, 1.03)  # recall range
        ax[1, k].set_ylim(-0.03, 1.03)  # share range
        ax[0, k].set_title(RXN[rx], fontsize=10)  # column title
        ax[nr - 1, k].set_xlabel("held-out session")  # axis label
    ax[0, 0].set_ylabel("recall")  # axis label
    ax[1, 0].set_ylabel("share of clean windows in alarm")  # axis label
    if episodes:  # if: the third row
        ax[2, 0].set_ylabel("false-alarm episodes")  # axis label
    ax[0, 0].legend(fontsize=7, ncol=2, loc="lower left")  # legend
    fig.suptitle(title, fontsize=11)  # figure title
    fig.tight_layout()  # layout
    M.save_fig(fig, path)  # byte-stable save


def plot_264_diagnosis(dx: pd.DataFrame) -> None:  # function: the figure, the 2.6.4 diagnosis: percentile of the missed spoofing windows in the training clean and attack distributions
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5), sharex=True)  # one panel per receiver
    for a, rx in zip(ax, ("SBF_02", "SBF_03")):  # for loop over the two receivers
        d = dx[dx.receiver == rx].sort_values("rank_abs_coef")  # its 10 features
        yy = np.arange(len(d))[::-1]  # rows from the top
        a.barh(yy + 0.18, d.pct_in_clean_median, 0.36, color="#009E73", label="percentile in training clean (2.6.2)")  # clean percentile
        a.barh(yy - 0.18, d.pct_in_attack_median, 0.36, color="#D55E00", label="percentile in training attack")  # attack percentile
        a.set_yticks(yy)  # rows
        a.set_yticklabels([f"{c} ({v:+.2f})" for c, v in zip(d.column, d.coef)], fontsize=7)  # list comprehension: feature and coefficient
        a.axvline(50, color="k", lw=0.7)  # the median
        a.set_xlim(0, 100)  # percentile range
        a.set_xlabel("median percentile of the missed attack_spoof windows")  # axis label
        a.set_title(f"{RXN[rx]}, held out 2.6.4: {int(d.missed_windows.iloc[0])} missed windows; closer to clean jointly: {d.share_closer_to_clean_joint_top10.iloc[0]:.2f}", fontsize=9)  # panel title
    ax[0].legend(fontsize=7, loc="lower right")  # legend
    fig.tight_layout()  # layout
    M.save_fig(fig, GRAPH / "logistic_regression_264_diagnosis.png")  # byte-stable save


# ---------------------------------------------------------------- main

def main() -> None:  # function: the variants from start to finish
    t0 = time.time()  # the clock
    n_p, bad = M.protected_ok()  # protected files at the start
    check("Protected files at start", "87 unchanged", f"{n_p} compared, {len(bad)} changed {bad}", n_p == 87 and not bad)  # the check
    wc, fl = M.load_meta()  # dictionary and lists
    V = variants(wc, fl)  # the variants
    cells = {(w, rx): M.load_cell(w, rx) for w in (10,) for rx in RXS}  # dict comprehension: the w = 10 windows (other lengths loaded when needed)
    outs = {}  # variant -> tables
    for v in V:  # for loop over the variants
        w = v[2]  # its window length
        if (w, RXS[0]) not in cells:  # if: this length is not loaded yet
            cells = {k: c for k, c in cells.items() if k[0] == 10}  # dict comprehension: keep only w = 10 (memory)
            cells.update({(w, rx): M.load_cell(w, rx) for rx in RXS})  # dict comprehension: this length's windows
        outs[v[0]] = run_variant(v, wc, cells, keep=v[0] in ("primary", "shift_invariant", "no_circularity"))  # the variant
    sealed = all(not (c[0].test_code == M.SEALED).any() for c in cells.values())  # generator expression: no 2.6.12 window loaded
    check("Sealed: no 2.6.12 row in any loaded window table", "true", sealed, sealed)  # the check
    pub = pd.read_csv(io.StringIO(gzip.decompress((OUT / "logistic_regression_predictions_primary.csv.gz").read_bytes()).decode()))  # the regression script's published probabilities
    rt = lambda df: pd.read_csv(io.StringIO(df.to_csv(index=False, lineterminator="\n")))  # lambda: a table through the same CSV write and read as the published one (pandas' CSV floats are not round-trip exact, so both sides take the same trip)
    mine = rt(outs["primary"]["predictions"])  # this script's run again, through the CSV trip
    same_p = bool(np.array_equal(mine.p.to_numpy(), pub.p.to_numpy()) and (mine.sec.to_numpy() == pub.sec.to_numpy()).all())  # bitwise equal, same windows
    check("This script's primary equals the regression script's primary (held-out probabilities bitwise after the same CSV write and read, same windows)", "true", same_p, same_p)  # the check that the variant runner is the regression script's pipeline
    co = pd.read_csv(OUT / "logistic_regression_coefficients_primary.csv")  # the published coefficients
    same_c = all(np.array_equal(np.sort(rt(pd.DataFrame({"coef": outs["primary"]["keep"][(rx, "2.6.4")][5]["coef"]})).coef.to_numpy()), np.sort(co[(co.receiver == rx) & (co.held_out == "2.6.4")].coef.to_numpy())) for rx in ("SBF_02", "SBF_03"))  # generator expression: the 2.6.4 models reproduce exactly (after the same CSV trip)
    check("2.6.4 diagnosis uses the published models (coefficients equal exactly after the same CSV trip, X5 and mosaic-H)", "true", same_c, same_c)  # the check (no refit)
    per = persistence(outs["primary"])  # the alarm-rule variant
    print("  C4b on shift_invariant and no_circularity ...")  # progress line
    c4b = pd.concat([pd.read_csv(OUT / "logistic_regression_c4b_within_session.csv").assign(variant="primary"), c4b_variant("shift_invariant", outs["shift_invariant"]["keep"]), c4b_variant("no_circularity", outs["no_circularity"]["keep"])], ignore_index=True)  # the primary's C4b and the two variants'
    order = [v[0] for v in V] + ["persist_3of5"]  # run order
    allm = pd.concat([outs[k]["metrics"] for k in order[:-1]] + [per["metrics"]], ignore_index=True)  # every fold row
    allp = pd.concat([outs[k]["pooled"] for k in order[:-1]] + [per["pooled"]], ignore_index=True)  # every pooled row
    allm["variant"] = pd.Categorical(allm.variant, order, ordered=True)  # categorical: keeps run order in grouping
    allp["variant"] = pd.Categorical(allp.variant, order, ordered=True)  # same for the pooled rows
    vt = variant_table(allm, allp, c4b)  # the variant table
    dx = diagnosis_264(outs["primary"]["keep"], wc)  # the 2.6.4 diagnosis
    fd = pd.DataFrame(M.FIT_DIAG)  # every counted fit of this script
    check("Every variant fit: no warning (any would have stopped the script) and max |gradient| per unit weight <= 1e-4", "all", f"{len(fd)} fits; largest {fd.max_abs_grad_per_weight.max():.3g}; iterations {int(fd.n_iter.min())}-{int(fd.n_iter.max())}; line-search {int(fd.linesearch_warnings.sum())}, convergence {int(fd.convergence_warnings.sum())}", bool((fd.max_abs_grad_per_weight <= M.GRAD_TOL).all() and fd.linesearch_warnings.sum() == 0 and fd.convergence_warnings.sum() == 0 and fd.compressed.all()))  # the check
    check("Fits", "16 variants x 15 + C4b 2 x 300 = 840", len(fd), len(fd) == 16 * 15 + 600)  # the count
    lcols = ["variant", "family", "receiver", "held_out"]  # identity columns first
    M.write_csv(allm[lcols + [c for c in allm.columns if c not in lcols]], OUT / "logistic_regression_variants_metrics_long.csv")  # every fold row
    M.write_csv(pd.concat([outs[k]["latency"] for k in order[:-1]] + [per["latency"]], ignore_index=True), OUT / "logistic_regression_variants_latency_long.csv")  # every latency row
    M.write_gz(pd.concat([outs[k]["predictions"] for k in order[:-1]] + [per["predictions"]], ignore_index=True), OUT / "logistic_regression_variants_predictions.csv.gz")  # every held-out probability
    M.write_csv(pd.concat([outs[k]["selection"] for k in order[:-1]], ignore_index=True), OUT / "logistic_regression_variants_selection.csv")  # what each fold fitted
    M.write_csv(vt, OUT / "logistic_regression_variants.csv")  # the variant table
    M.write_csv(c4b, OUT / "logistic_regression_variants_c4b.csv")  # C4b, primary and two variants
    M.write_csv(dx, OUT / "logistic_regression_variants_264_diagnosis.csv")  # the 2.6.4 diagnosis
    M.write_csv(fd, OUT / "logistic_regression_variants_fit_diagnostics.csv")  # fit diagnostics
    fs = ["primary", "no_residual", "known_antenna", "no_circularity", "shift_invariant", "main_plus_geometry", "screen_no_floor", "unscreened"]  # the feature-set family
    family_figure(allm, fs, GRAPH / "logistic_regression_variants_feature_sets.png", "Feature-set variants (one change from the primary each)")  # the figure
    family_figure(allm, ["w1", "w2", "w3", "w5", "primary"], GRAPH / "logistic_regression_variants_windows.png", "Window length (primary: w = 10 s)")  # the figure
    family_figure(allm, ["C0.01", "C0.1", "primary", "C10", "no_selection"], GRAPH / "logistic_regression_variants_regulariser.png", "Regularisation strength and selection (primary: C = 1, Pearson + MI top 36)")  # the figure
    family_figure(allm, ["primary", "persist_3of5"], GRAPH / "logistic_regression_variants_persistence.png", "Alarm rule: every window at p >= 0.5 (primary) against 3 of 5 windows", episodes=True)  # the figure
    plot_264_diagnosis(dx)  # the figure
    n_p, bad = M.protected_ok()  # protected files at the end
    check("Protected files at end", "87 unchanged", f"{n_p} compared, {len(bad)} changed {bad}", n_p == 87 and not bad)  # the check
    M.write_csv(pd.DataFrame(CHECKS), CHECK_DIR / "logistic_regression_variants_checks.csv")  # checks
    print(f"done in {time.time() - t0:.0f} s; failed checks: {sum(c['ok'] is False for c in CHECKS)}")  # run time and failures
    if any(c["ok"] is False for c in CHECKS):  # if: a check failed
        sys.exit(1)  # exit status 1


if __name__ == "__main__":  # guard: runs main() only when the file is executed, not when imported
    main()  # entry point
