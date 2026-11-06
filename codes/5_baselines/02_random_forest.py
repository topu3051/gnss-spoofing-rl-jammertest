#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
The random forest, its controls and grouped permutation importance (RQ1), under
leave-one-session-out folds only.

Primary configuration (pre-registered before any fit, in three parts (a) to (c)):
  per receiver (mosaic-X5, mosaic-H main, u-blox 01), five folds over 2.6.5, 2.6.2, 2.6.10, 2.6.4, 2.6.11; 2.6.12 sealed (the regression script's reader);
  target attack (attack_jam_spoof and attack_spoof) against clean; uncertain and excluded never trained on or scored;
  w = 10 s; (a) the orbit-free block alone (the logistic regression's no_residual set: the no_residual screened list with the floor for that fold) and
  the orbit-free missing-indicator; (b) the logistic regression's preprocessing unchanged (columns with no training value dropped, median imputation,
  robust scaling, sign(z) log1p(|z|), Pearson pruning at 0.89, MI top 36 with seed 20260914), so the forest sees the columns the logistic regression's
  no_residual logistic regression saw (checked: 15 of 15 selections equal); (c)
  RandomForestClassifier(n_estimators 500, max_features "sqrt", min_samples_leaf 5, class_weight "balanced_subsample",
  bootstrap True, criterion "gini", random_state 20260914), fitted with n_jobs 8 and predicting with n_jobs 1 (the
  fitted trees do not depend on n_jobs, but a parallel predict_proba sums the trees in completion order and is not bitwise
  reproducible); threshold 0.5 fixed a priori, alarm when p >= 0.5, no persistence; forest seeds 20260914-20260918 for dispersion,
  the first reported.

Grouped permutation importance (the RQ1 instrument): on the scored held-out windows of each fold only, with the
fold's fitted pipeline, 30 repeats; repeat r draws one row permutation (seed 20260914 + 2000 + r) used for every group. Columns are
permuted together by observable family, defined by the name rule family_of() below (committed before any importance existed):
cn0, code_second_difference, code_doppler_consistency, carrier_phase (code-minus-carrier rate and lock losses), tracking_count,
clock_step, doppler (Doppler and Doppler rate change statistics), code_rate (single-difference change statistics) and the
missing_indicator. A secondary, report-only grouping by cross-satellite statistic (level, extreme, dispersion, count/flag/offset)
and per-column importance are given beside it. Permuting a group's selected columns after the fitted column-wise steps equals
permuting its raw values before imputation (missing values move with their window, the forest tests test I1 proves it bitwise). A family with no
selected column is reported as "not selected". Drops (baseline minus permuted) in AUROC where the fold holds both classes, AUROC per
attack class where the fold holds clean windows and that class, recall at 0.5 overall and per class, and the share of clean windows
in alarm (a negative drop is a rise in false alarms). Two report-only diagnostics: the share of scored held-out
windows carrying an imputed value in a selected column, per label and per family; and every importance recomputed on the scored
held-out windows with no imputed selected value ("no_imputed" subset) beside the full one.
C4b comparison (fixed before any importance existed): per C4b cell (held-out 2.6.2 and 2.6.4; 2.6.10 for u-blox 01)
the median C4b model (the 50th of 100 by held-out AUROC, ties by permutation number) gets the same grouped importance. A family is
"used" when its true-label mean AUROC drop is >= 0.01 and its 5th percentile over the repeats is > 0; a used family is a "session
carrier" when the C4b model's mean drop is >= 0.5 x the true-label drop, otherwise "attack evidence"; 0.25 and 0.75 are reported as
sensitivities; a fold whose primary held-out AUROC is below 0.5 is "not interpretable"; a family with no selected column in the
primary is "not selected".
Controls here: C4a labels permuted across the training windows, 100 seeds (20260914 + k), held-out 2.6.2 and 2.6.4, median held-out
AUROC must lie in [0.4, 0.6] in all six cells; C4b labels permuted inside each training session, 100 seeds (20260914 + 1000 + k),
report only; C5 the 370-column geometry block alone, no indicator; C6 forest seeds and MI seeds 20260914-20260918 (the jamming screen's SEED
restored after every MI run). C1-C3 and the importance tests are in tests/test_random_forest.py. The no_residual logistic regression of the logistic regression is refitted
on the same prepared folds (checked against its published probabilities) for the comparison and for its grouped importance
 (report only).

Reads   Data/processed/features/w10/{receiver}_*.csv.gz (2.6.12 dropped at read), features_window_columns.csv, feature_lists.csv.gz,
        labels/labels_evidence.csv (through the scoring module), Data/processed/logistic_regression_registered/logistic_regression_variants_selection.csv, logistic_regression_variants_predictions.csv.gz,
        logistic_regression_variants_metrics_long.csv, logistic_regression_variants.csv, Data/processed/checks/expected_hashes_random_forest.txt, Data/processed/checks/random_forest_family_rule.csv
Writes  Data/processed/random_forest_registered/random_forest_*.csv(.gz), Data/processed/checks/random_forest_checks.csv, random_forest_numbers.csv, Graph/random_forest_*.png;
        with --rule-only: Data/processed/checks/random_forest_family_rule.csv and nothing else
"""

from __future__ import annotations  # future import: type hints kept as text (a convention across the project's scripts)

import hashlib  # standard-library hashing, for the protected-file check
import importlib.util  # standard-library module loader, for modules whose names start with a digit
import re  # standard-library regular expressions, for the family name rule
import sys  # standard-library system access: sys.modules, sys.argv and the exit status
import time  # standard-library clock, for the run-time line
from pathlib import Path  # Path objects for building file paths

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
spec = importlib.util.spec_from_file_location("logistic_regression_for_forest", PROJECT / "codes" / "5_baselines" / "01_logistic_regression.py")  # where 09 is
M = importlib.util.module_from_spec(spec)  # an empty module object
sys.modules["logistic_regression_for_forest"] = M  # registered, so its own imports resolve
spec.loader.exec_module(M)  # the regression script's top level (its main() is guarded and not run); it sets MPLCONFIGDIR, the Agg backend and codes/ on sys.path

import numpy as np  # NumPy: arrays
import pandas as pd  # pandas: tables
import matplotlib.pyplot as plt  # the pyplot interface (after 09 chose the file-only backend)
from sklearn.ensemble import RandomForestClassifier  # the model: a bagged ensemble of decision trees
from sklearn.metrics import roc_auc_score  # threshold-free score of the held-out probabilities
from helpers import save_fig  # byte-stable figure saving (codes/common was put on sys.path by the regression script)

S, SC, SEED = M.S, M.SC, M.SEED  # the screen script, the scoring module and the project seed, from the regression script
PROC = PROJECT / "Data" / "processed"  # processed tables
OUT = PROC / "random_forest_registered"  # this section's tables
CHECK_DIR = PROJECT / "Data" / "processed" / "checks"  # check tables
GRAPH = PROJECT / "Graph"  # figures
RXS, RXN, SESSIONS = M.RXS, M.RXN, M.SESSIONS  # receivers, their names, the five folds
ATTACK, SCORED, THR, W = M.ATTACK, M.SCORED, M.THR, M.W  # attack labels, scored labels, threshold 0.5, window length 10 s
SET, BLOCKS = "no_residual", ("orbitfree",)  # (a): the orbit-free set and its one missing-indicator
# dict literal: the pre-registered forest settings other than the seed and n_jobs
RF_PARAMS = {"n_estimators": 500, "max_features": "sqrt", "min_samples_leaf": 5, "class_weight": "balanced_subsample", "bootstrap": True, "criterion": "gini"}
FIT_JOBS, PRED_JOBS = 8, 1  # fit in parallel (trees identical for any n_jobs), predict serially (fixed summation order)
FOREST_SEEDS = [SEED + k for k in range(5)]  # list comprehension: the five forest seeds of C6, the reported one first
MI_SEEDS = M.MI_SEEDS  # the five MI seeds of C6 (the regression script's list, the primary's first)
N_PERM_C4A, C4A_FOLDS, C4A_BAND = 100, ["2.6.2", "2.6.4"], (0.4, 0.6)  # C4a: 100 permutations on the two folds with both classes, pass band
N_PERM_C4B = 100  # C4b: 100 within-session permutations per cell
C4B_MEDIAN_RANK = 50  # the median C4b model: the 50th of 100 by held-out AUROC (ties by permutation number)
N_REP, IMP_SEED0 = 30, SEED + 2000  # importance: 30 repeats, repeat r permuted with seed 20260914 + 2000 + r
USED_MIN, RATIO, RATIO_SENS = 0.01, 0.5, (0.25, 0.75)  # the family classification: "used" threshold, carrier ratio, and its two sensitivities
# list literal: the primary's families in reporting order
FAMILIES = ["cn0", "code_second_difference", "code_doppler_consistency", "carrier_phase", "tracking_count", "clock_step", "doppler", "code_rate", "missing_indicator"]
XSTATS = ["level", "extreme", "dispersion", "count_flag_offset", "indicator"]  # list literal: the secondary grouping by cross-satellite statistic
IND_FAMILY = {"ind_missing_orbitfree": "missing_indicator", "ind_missing_self": "self:missing_indicator", "ind_missing_known": "known:missing_indicator"}  # dict: indicator column -> family
# list literal: the metrics whose drop is reported; "auroc_jam_spoof" and "auroc_spoof" are clean against that attack class
METRICS = ["auroc", "auroc_jam_spoof", "auroc_spoof", "recall", "recall_jam_spoof", "recall_spoof", "clean_alarm_share"]

CHECKS: list = []  # list: the checks, written to random_forest_checks.csv
NUM: dict = {}  # dict: numbers the summary quotes


def check(name: str, expected: str, found, ok: bool | None) -> None:  # function: records one check and prints it
    CHECKS.append({"check": name, "expected": expected, "found": found, "ok": "" if ok is None else bool(ok)})  # append a dict row
    print(f"  [{'ok' if ok else ('FAIL' if ok is False else '--')}] {name}: expected {expected}; found {found}")  # progress line


def sha(path: Path) -> str:  # function: SHA-256 of a file's bytes
    return hashlib.sha256(path.read_bytes()).hexdigest()  # hex digest


def protected_ok() -> tuple:  # function: the 139 protected files (the logistic regression's 87, every the logistic regression output, the figures, 09/09b/09c) against their hashes
    rows = [l.split(maxsplit=1) for l in (CHECK_DIR / "expected_hashes_random_forest.txt").read_text().splitlines() if l.strip()]  # list comprehension: (hash, path) pairs
    bad = [p for h, p in rows if sha(PROJECT / p) != h]  # list comprehension: files whose hash changed
    return len(rows), bad  # how many were compared, and which differ


# ---------------------------------------------------------------- the family rule (committed before any importance)

def family_of(column: str, base: str | None) -> str:  # function: the observable family of one window column, from its base column's name
    if column in IND_FAMILY:  # if: a missing-indicator (it has no dictionary row)
        return IND_FAMILY[column]  # its family
    m = re.match(r"^(self_|known_)?([GREC]_|all_)?(.*)$", base)  # regular expression: optional block prefix, optional constellation prefix, the quantity
    blk, q = (m.group(1) or "").rstrip("_"), m.group(3)  # the block ("self", "known" or "") and the quantity with its cross-satellite statistic
    if blk:  # if: a residual block (variants V1 and V2)
        if q.startswith("res"):  # if: code residual or de-biased code residual
            return f"{blk}:code_residual"  # family
        if q.startswith("dop_res"):  # if: Doppler residual
            return f"{blk}:doppler_residual"  # family
        if q in ("n_res", "n_jump", "n_outband"):  # if: residual, jump and out-of-band counts
            return f"{blk}:residual_counts"  # family
        if q.startswith("isb_"):  # if: an inter-system clock offset
            return f"{blk}:intersystem_offset"  # family
        if q in ("solved", "pos_held", "n_rej"):  # if: the self-solver's state
            return f"{blk}:solver_state"  # family
        if q == "pdop":  # if: PDOP (geometry change statistics only)
            return f"{blk}:pdop"  # family
        return f"{blk}:UNPLACED"  # a name the rule cannot place (checked to be absent)
    if q.startswith("cn0_"):  # if: C/N0 per constellation or over all satellites
        return "cn0"  # family
    if q.startswith("d2C_"):  # if: second difference of the pseudorange
        return "code_second_difference"  # family
    if q.startswith("dC_dc_"):  # if: drift-corrected single difference (code rate plus Doppler range rate); tested before "dC_"
        return "code_doppler_consistency"  # family
    if q.startswith("cmc_rate_") or q == "n_lock_loss":  # if: code-minus-carrier rate or carrier loss-of-lock count
        return "carrier_phase"  # family
    if q in ("n_tracked", "d_n_tracked", "n_new", "n_lost"):  # if: a tracking count or its change
        return "tracking_count"  # family
    if q == "ms_step":  # if: the common millisecond receiver clock step flag
        return "clock_step"  # family
    if q.startswith("dop_"):  # if: raw Doppler or Doppler rate (geometry change statistics only)
        return "doppler"  # family
    if q.startswith("dC_"):  # if: raw single difference of the pseudorange, the code rate (geometry change statistics only)
        return "code_rate"  # family
    return "UNPLACED"  # a name the rule cannot place (checked to be absent)


def xstat_of(column: str, base: str | None) -> str:  # function: the secondary grouping by cross-satellite statistic (report only)
    if column in IND_FAMILY:  # if: a missing-indicator
        return "indicator"  # group
    if base.endswith("_median"):  # if: the median over satellites
        return "level"  # group
    if base.endswith("_maxabs"):  # if: the largest absolute value over satellites
        return "extreme"  # group
    if base.endswith("_std"):  # if: the spread over satellites
        return "dispersion"  # group
    return "count_flag_offset"  # counts, the clock-step flag and inter-system offsets


def family_rule_table(wc: pd.DataFrame) -> pd.DataFrame:  # function: the rule applied to every dictionary column at every window length, plus the indicators
    d = wc[["w", "column", "base_column", "part", "origin", "stat", "no_residual", "main", "known_antenna"]].copy()  # the dictionary's naming columns
    d["family"] = [family_of(c, b) for c, b in zip(d.column, d.base_column)]  # list comprehension: family per column
    d["xstat"] = [xstat_of(c, b) for c, b in zip(d.column, d.base_column)]  # list comprehension: secondary group per column
    ind = pd.DataFrame([{"w": w, "column": c, "base_column": "", "part": "indicator", "origin": "", "stat": "", "no_residual": int(c == "ind_missing_orbitfree"), "main": int(c != "ind_missing_known"), "known_antenna": int(c != "ind_missing_self"), "family": f, "xstat": "indicator"} for w in S.WINDOWS for c, f in IND_FAMILY.items()])  # list comprehension: the three indicators per length
    return pd.concat([d, ind], ignore_index=True).sort_values(["w", "part", "column"], kind="stable").reset_index(drop=True)  # one table in a fixed order


def name_maps(wc: pd.DataFrame) -> tuple:  # function: column -> family and column -> secondary group at w = 10
    x = wc[wc.w == W]  # this length
    fam = {c: family_of(c, b) for c, b in zip(x.column, x.base_column)}  # dict comprehension: families
    xs = {c: xstat_of(c, b) for c, b in zip(x.column, x.base_column)}  # dict comprehension: secondary groups
    for c, f in IND_FAMILY.items():  # for loop over the indicators
        fam[c], xs[c] = f, "indicator"  # added
    return fam, xs  # both


# ---------------------------------------------------------------- the forest

def fit_forest(Zt: np.ndarray, yt: np.ndarray, seed: int) -> RandomForestClassifier:  # function: the pre-registered forest fitted in parallel, then set to predict serially
    m = RandomForestClassifier(**RF_PARAMS, random_state=seed, n_jobs=FIT_JOBS).fit(Zt, yt)  # dict unpacking into keyword arguments: fitted on the training rows
    if list(m.classes_) != [0, 1]:  # if: the two classes are not both present in this order (cannot happen with scored training rows; asserted)
        raise SystemExit("STOP: forest classes are not [0, 1]")  # stop
    m.set_params(n_jobs=PRED_JOBS)  # serial prediction: trees summed in index order, so probabilities are bitwise reproducible
    return m  # the fitted forest


def proba(m: RandomForestClassifier, Z: np.ndarray) -> np.ndarray:  # function: probability of attack for each row
    return m.predict_proba(Z)[:, 1]  # column 1: class 1 (attack)


def forest_shape(m: RandomForestClassifier) -> dict:  # function: tree depth and size of a fitted forest (deterministic, written)
    d = np.array([e.tree_.max_depth for e in m.estimators_])  # list comprehension: depth of every tree
    lv = np.array([e.tree_.n_leaves for e in m.estimators_])  # list comprehension: leaves of every tree
    return {"tree_depth_median": float(np.median(d)), "tree_depth_max": int(d.max()), "tree_leaves_median": float(np.median(lv)), "tree_leaves_max": int(lv.max())}  # dict literal: summary


def candidates(X: pd.DataFrame, wc: pd.DataFrame, fl: pd.DataFrame, rx: str, held: str, config: str) -> pd.DataFrame:  # function: a configuration's candidate columns and indicators
    if config == "geometry":  # if: C5, the geometry block of the main variant alone, no indicator
        return M.with_indicators(X, M.geometry_block(wc, W), wc, W, ())  # the 370 columns
    return M.with_indicators(X, M.cand_list(fl, W, rx, held, SET), wc, W, BLOCKS)  # (a): the no_residual screened list with the floor, and the orbit-free indicator


def run_config(cells: dict, wc: pd.DataFrame, fl: pd.DataFrame, config: str, mi_seed: int = SEED, forest_seed: int = SEED, keep: bool = False) -> dict:  # function: one configuration over every receiver and fold
    preds, metrics, lat, selrep, kept = [], [], [], [], {}  # outputs
    for rx in RXS:  # for loop over the receivers
        idx, X = cells[rx]  # the windows (2.6.12 dropped by the regression script's reader)
        y = M.target(idx)  # the target
        for held, train in S.folds():  # for loop over the folds
            tr, te = M.fold_rows(idx, held, train)  # scored training rows and every held-out row in time order
            V = candidates(X, wc, fl, rx, held, config)  # candidates and indicator
            prep = M.prepare(V, tr)  # (b): the regression script's label-free steps on the training rows (drop, impute, scale, compress, prune)
            sel = M.mi_top(prep, y, tr, mi_seed)  # the screen script's MI top 36 of the kept columns (the jamming screen's SEED restored inside)
            Z = prep["Z"][:, sel]  # the selected columns
            m = fit_forest(Z[tr], y[tr], forest_seed)  # the forest on the scored training rows
            p = proba(m, Z[te])  # held-out probabilities
            row, lr = M.evaluate_fold(idx, te, p, held)  # the scoring module's figures through the regression script's metric code
            trl = idx.label.to_numpy()[tr]  # training labels
            cs = sorted(set(idx.test_code.to_numpy()[tr][trl == "clean"]))  # sessions that supply the clean training windows
            # dict update: identity and training material, as the logistic regression's metric table
            row.update({"receiver": rx, "config": config, "mi_seed": mi_seed, "forest_seed": forest_seed, "train_clean": int((trl == "clean").sum()), "train_clean_sessions": "+".join(cs), "train_clean_independent": round(float((trl == "clean").sum()) / W, 1),
                        "train_attack_jam_spoof": int((trl == "attack_jam_spoof").sum()), "train_attack_spoof": int((trl == "attack_spoof").sum())})  # training attack windows
            metrics.append(row)  # kept
            for r in lr:  # for loop over the latency rows
                r.update({"receiver": rx, "config": config, "mi_seed": mi_seed, "forest_seed": forest_seed})  # identity
                lat.append(r)  # kept
            preds.append(pd.DataFrame({"receiver": rx, "config": config, "mi_seed": mi_seed, "forest_seed": forest_seed, "held_out": held, "sec": idx.sec.to_numpy()[te], "label": idx.label.to_numpy()[te], "p": p, "alarm": p >= THR}))  # dict literal: held-out probabilities
            names = [prep["names"][j] for j in sel]  # list comprehension: the selected column names in MI order
            ind_kept = M.IND["orbitfree"] in prep["names"] and prep["names"].index(M.IND["orbitfree"]) in prep["kept"]  # chained "and": the indicator survived Pearson pruning (False for C5, which has none)
            # dict literal: what was fitted in this fold
            selrep.append({"receiver": rx, "config": config, "mi_seed": mi_seed, "forest_seed": forest_seed, "held_out": held, "candidates": V.shape[1], "dropped_no_training_value": len(prep["dropped"]), "kept_after_pearson": len(prep["kept"]), "selected": len(sel),
                           "ind_orbitfree_kept": bool(ind_kept), "ind_orbitfree_selected": M.IND["orbitfree"] in names, "p_equal_threshold": int((p == THR).sum()), **forest_shape(m),  # dict unpacking: the forest's tree sizes
                           "train_rows": len(tr), "heldout_rows": len(te), "selected_columns": ";".join(names)})  # rows behind the fit and the prediction, and the selected columns
            if keep:  # if: the caller reuses the prepared fold (importance, C4a, C4b, LR)
                kept[(rx, held)] = {"V": V, "prep": prep, "tr": tr, "te": te, "y": y, "idx": idx, "sel": sel, "names": names, "model": m, "p": p}  # dict literal: kept
    first = ["receiver", "config", "mi_seed", "forest_seed", "held_out"]  # identity columns first
    out = {"predictions": pd.concat(preds, ignore_index=True), "metrics": pd.DataFrame(metrics), "latency": pd.DataFrame(lat), "selection": pd.DataFrame(selrep)}  # dict literal: the tables
    for k in ("metrics", "latency"):  # for loop over the two tables with appended identity columns
        out[k] = out[k][first + [c for c in out[k].columns if c not in first]]  # list comprehension: reordered
    out["pooled"] = M.pooled(out["metrics"], out["latency"], out["predictions"]).assign(config=config, mi_seed=mi_seed, forest_seed=forest_seed)  # the regression script's pooled figures per receiver
    out["keep"] = kept  # the prepared folds (not written)
    return out  # everything


# ---------------------------------------------------------------- grouped permutation importance

def metrics_of(p: np.ndarray, lab: np.ndarray) -> dict:  # function: the importance metrics of one set of probabilities on scored windows
    att, cln, al = np.isin(lab, ATTACK), lab == "clean", p >= THR  # attack windows, clean windows, alarms
    out = {k: np.nan for k in METRICS}  # dict comprehension: every metric undefined until computed
    if att.any() and cln.any():  # if: both classes present
        out["auroc"] = float(roc_auc_score(att, p))  # AUROC, attack against clean
    for cls in ATTACK:  # for loop over the two attack classes
        c = lab == cls  # that class
        if c.any() and cln.any():  # if: clean and that class present
            m = c | cln  # the two groups
            out[f"auroc_{cls.replace('attack_', '')}"] = float(roc_auc_score(c[m], p[m]))  # AUROC of clean against that class
        if c.any():  # if: the class is present
            out[f"recall_{cls.replace('attack_', '')}"] = float(al[c].mean())  # recall of that class at 0.5
    if att.any():  # if: attack windows present
        out["recall"] = float(al[att].mean())  # recall at 0.5
    if cln.any():  # if: clean windows present
        out["clean_alarm_share"] = float(al[cln].mean())  # share of clean windows in alarm
    return out  # the metrics


def importance(predict, Zs: np.ndarray, lab: np.ndarray, groups: dict, ident: dict, keep_repeats: bool) -> tuple:  # function: drops of every metric when each group's columns are permuted together
    n = len(lab)  # windows
    base = metrics_of(predict(Zs), lab) if n else {k: np.nan for k in METRICS}  # conditional expression: the unpermuted metrics, undefined when the subset is empty
    perms = [np.random.default_rng(IMP_SEED0 + r).permutation(n) for r in range(N_REP)]  # list comprehension: one row permutation per repeat, shared by every group
    summ, reps = [], []  # summary rows and repeat rows
    for g, cols in groups.items():  # for loop over the groups in their fixed order
        if not cols or not n:  # if: no selected column in this group, or no window in this subset
            for k in METRICS:  # for loop over the metrics
                summ.append({**ident, "group": g, "status": "not selected" if not cols else "no windows", "n_selected_cols": len(cols), "n_windows": n, "metric": k, "base": base[k], "drop_mean": np.nan, "drop_sd": np.nan, "drop_p5": np.nan, "drop_p95": np.nan, "n_repeats": 0})  # dict unpacking: an empty row
            continue  # next group
        big = np.tile(Zs, (N_REP, 1))  # the window matrix repeated once per repeat, stacked
        for r, pm in enumerate(perms):  # for loop over the repeats
            big[r * n:(r + 1) * n, cols] = Zs[pm][:, cols]  # this repeat's block: the group's columns taken from the permuted rows
        pb = predict(big).reshape(N_REP, n)  # one prediction call for all repeats, back to one row per repeat
        D = np.array([[base[k] - metrics_of(pb[r], lab)[k] for k in METRICS] for r in range(N_REP)])  # nested list comprehension: drop per repeat and metric
        for j, k in enumerate(METRICS):  # for loop over the metrics
            v = D[:, j]  # its 30 drops
            ok = np.isfinite(v)  # defined repeats
            summ.append({**ident, "group": g, "status": "selected", "n_selected_cols": len(cols), "n_windows": n, "metric": k, "base": base[k],  # dict unpacking: identity and baseline
                         "drop_mean": float(v[ok].mean()) if ok.any() else np.nan, "drop_sd": float(v[ok].std()) if ok.any() else np.nan,  # mean and standard deviation (1/N) over the repeats
                         "drop_p5": float(np.percentile(v[ok], 5)) if ok.any() else np.nan, "drop_p95": float(np.percentile(v[ok], 95)) if ok.any() else np.nan, "n_repeats": int(ok.sum())})  # 5th and 95th percentiles
        if keep_repeats:  # if: the repeats are written (family grouping only)
            reps += [{**ident, "group": g, "repeat": r, **{f"drop_{k}": D[r, j] for j, k in enumerate(METRICS)}} for r in range(N_REP)]  # list comprehension with dict unpacking: one row per repeat
    return summ, reps  # both


def importance_cell(predict, kp: dict, names: list, sel: list, fam: dict, xs: dict, ident: dict) -> tuple:  # function: every grouping and both subsets for one fitted fold
    te, idx = kp["te"], kp["idx"]  # held-out rows and the index
    lab_te = idx.label.to_numpy(object)[te]  # held-out labels in time order
    sc = np.isin(lab_te, SCORED)  # scored held-out windows (uncertain never scored, never permuted in)
    Zs = kp["prep"]["Z"][te][:, sel][sc]  # the selected columns of the scored held-out windows, after the fitted steps
    labs = lab_te[sc]  # their labels
    imp = kp["V"].iloc[te[sc]][names].isna().to_numpy()  # which selected values were missing before imputation (windows x columns)
    no_imp = ~imp.any(axis=1)  # windows with no imputed selected value
    unplaced = [c for c in names if fam[c] not in FAMILIES]  # list comprehension: selected columns outside the primary's families (none for the primary)
    fams = FAMILIES + sorted({fam[c] for c in unplaced})  # every primary family, plus any other family a model selected (the variants V1 and V2)
    groups = {"family": {f: [j for j, c in enumerate(names) if fam[c] == f] for f in fams},  # nested comprehension: selected-column positions per family
              "xstat": {g: [j for j, c in enumerate(names) if xs[c] == g] for g in XSTATS},  # per secondary group
              "column": {c: [j] for j, c in enumerate(names)}}  # per column (secondary)
    summ, reps = [], []  # outputs
    for subset, mask in (("all", np.ones(len(labs), bool)), ("no_imputed", no_imp)):  # for loop over the full scored set and the no-imputed diagnostic
        for gname, gr in groups.items():  # for loop over the groupings
            s, r = importance(predict, Zs[mask], labs[mask], gr, {**ident, "grouping": gname, "subset": subset}, keep_repeats=(gname == "family"))  # dict unpacking: identity extended
            summ += s  # kept
            reps += r  # kept
    shares = []  # imputed-share rows
    for lbl in ("clean", "attack_jam_spoof", "attack_spoof"):  # for loop over the scored labels
        mm = labs == lbl  # windows of this label
        if not mm.any():  # if: the label is not held out
            continue  # next
        shares.append({**ident, "group": "any selected column", "label": lbl, "n_windows": int(mm.sum()), "share_imputed": float(imp[mm].any(axis=1).mean())})  # dict unpacking: any selected column imputed
        for f in fams:  # for loop over the families
            cols = groups["family"][f]  # its selected columns
            if cols:  # if: the family has selected columns
                shares.append({**ident, "group": f, "label": lbl, "n_windows": int(mm.sum()), "share_imputed": float(imp[mm][:, cols].any(axis=1).mean())})  # dict unpacking: a family column imputed
    return summ, reps, shares  # summary, repeats and imputed shares


def importance_config(kept: dict, fam: dict, xs: dict, config: str, model: str, predict_of) -> tuple:  # function: importance of every cell of a configuration
    summ, reps, shares = [], [], []  # outputs
    for (rx, held), kp in kept.items():  # for loop over the kept folds in their run order
        s, r, sh = importance_cell(predict_of(kp), kp, kp["names"], kp["sel"], fam, xs, {"config": config, "model": model, "receiver": rx, "held_out": held})  # one cell
        summ += s  # kept
        reps += r  # kept
        shares += sh  # kept
    return pd.DataFrame(summ), pd.DataFrame(reps), pd.DataFrame(shares)  # three tables


def core(cells: dict, wc: pd.DataFrame, fl: pd.DataFrame) -> dict:  # function: the primary, C5 and the primary's importance (the forest tests' C1 rebuilds exactly this)
    fam, xs = name_maps(wc)  # the family rule and the secondary grouping
    prim = run_config(cells, wc, fl, "primary", keep=True)  # the primary
    geo = run_config(cells, wc, fl, "geometry")  # C5
    summ, reps, shares = importance_config(prim["keep"], fam, xs, "primary", "forest", lambda kp: (lambda Z: proba(kp["model"], Z)))  # nested lambda: the primary's grouped importance
    files = {}  # file name -> table
    for suffix, o in (("primary", prim), ("c5_geometry", geo)):  # for loop over the two configurations
        files[f"random_forest_predictions_{suffix}.csv.gz"] = o["predictions"]  # held-out probabilities
        for k in ("metrics", "latency", "pooled", "selection"):  # for loop over the plain tables
            files[f"random_forest_{k}_{suffix}.csv"] = o[k]  # the table
    files["random_forest_importance_primary.csv"] = summ  # grouped, secondary and per-column importance, both subsets
    files["random_forest_importance_repeats_primary.csv.gz"] = reps  # every repeat of the family grouping
    files["random_forest_imputed_share.csv"] = shares  # the imputation diagnostic
    return {"files": files, "prim": prim, "geo": geo, "fam": fam, "xs": xs}  # dict literal: everything


def write_files(files: dict) -> None:  # function: writes each table, gzip for .gz names
    for name, df in files.items():  # for loop over the tables
        (M.write_gz if name.endswith(".gz") else M.write_csv)(df, OUT / name)  # conditional expression picks the regression script's writer; no time stamp in either


# ---------------------------------------------------------------- C4a, C4b, C6 and the logistic regression

def c4a(kept: dict) -> pd.DataFrame:  # function: labels permuted across all scored training windows, 100 seeds per cell, held-out AUROC with the true labels
    rows = []  # results
    for rx in RXS:  # for loop over the receivers
        for held in C4A_FOLDS:  # for loop over 2.6.2 and 2.6.4
            kp = kept[(rx, held)]  # the primary's prepared fold (its steps use no label)
            prep, tr, te, y, idx = kp["prep"], kp["tr"], kp["te"], kp["y"], kp["idx"]  # its parts
            lab = idx.label.to_numpy()[te]  # held-out labels
            sc = np.isin(lab, SCORED)  # scored held-out windows
            for k in range(N_PERM_C4A):  # for loop over the permutations
                yp = y.copy()  # the labels
                yp[tr] = np.random.default_rng(SEED + k).permutation(y[tr])  # shuffled across all scored training windows (MI and the forest see these)
                sel = M.mi_top(prep, yp, tr, SEED)  # MI on the permuted labels
                m = fit_forest(prep["Z"][tr][:, sel], yp[tr], SEED)  # the forest on the permuted labels
                p = proba(m, prep["Z"][te][:, sel])  # held-out probabilities
                rows.append({"receiver": rx, "held_out": held, "perm": k, "seed": SEED + k, "auroc": float(roc_auc_score(np.isin(lab[sc], ATTACK), p[sc])), **forest_shape(m)})  # dict unpacking: AUROC against the true held-out labels
            print(f"  C4a {rx} {held}: median AUROC {np.median([r['auroc'] for r in rows if r['receiver'] == rx and r['held_out'] == held]):.3f}")  # progress line
    return pd.DataFrame(rows)  # the distribution


def c4b_cells() -> list:  # function: the C4b cells (held-out 2.6.2 and 2.6.4; 2.6.10 for u-blox 01, which has clean windows there)
    return [(rx, h) for rx in RXS for h in ("2.6.2", "2.6.4") + (("2.6.10",) if rx == "UBX_01" else ())]  # list comprehension: (receiver, fold) pairs


def c4b_permuted(kp: dict, k: int) -> np.ndarray:  # function: the labels of permutation k, shuffled inside each training session
    tc, y, tr = kp["idx"].test_code.to_numpy(), kp["y"], kp["tr"]  # sessions, labels, training rows
    rng = np.random.default_rng(SEED + 1000 + k)  # a fixed seed per permutation, distinct from C4a's
    yp = y.copy()  # the labels
    for s in sorted(set(tc[tr])):  # for loop over the training sessions in a fixed order
        r_ = tr[tc[tr] == s]  # that session's scored training rows
        yp[r_] = rng.permutation(y[r_])  # shuffled inside the session: its attack share is kept
    return yp  # permuted labels


def c4b_fit(kp: dict, k: int) -> dict:  # function: one C4b fit: MI and forest on the within-session permuted labels
    yp = c4b_permuted(kp, k)  # permuted labels
    prep, tr, te = kp["prep"], kp["tr"], kp["te"]  # the prepared fold
    sel = M.mi_top(prep, yp, tr, SEED)  # MI on the permuted labels
    m = fit_forest(prep["Z"][tr][:, sel], yp[tr], SEED)  # the forest
    return {"sel": sel, "names": [prep["names"][j] for j in sel], "model": m, "p": proba(m, prep["Z"][te][:, sel])}  # dict literal: the fit and its held-out probabilities


def c4b(kept: dict) -> pd.DataFrame:  # function: labels permuted inside each training session, 100 seeds per cell; report only
    rows = []  # results
    for rx, held in c4b_cells():  # for loop over the C4b cells
        kp = kept[(rx, held)]  # the prepared fold
        tc, y, tr = kp["idx"].test_code.to_numpy(), kp["y"], kp["tr"]  # sessions, labels, training rows
        lab = kp["idx"].label.to_numpy()[kp["te"]]  # held-out labels
        sc, att, cln = np.isin(lab, SCORED), np.isin(lab, ATTACK), lab == "clean"  # held-out masks
        mixed = sorted(s for s in set(tc[tr]) if len(set(y[tr][tc[tr] == s])) == 2)  # sorted generator: training sessions holding both classes (the only ones a permutation changes)
        for k in range(N_PERM_C4B):  # for loop over the permutations
            f = c4b_fit(kp, k)  # the fit
            al = f["p"] >= THR  # alarms
            # dict literal: identity and scores
            rows.append({"receiver": rx, "held_out": held, "perm": k, "seed": SEED + 1000 + k, "permuted_sessions": "+".join(mixed), "auroc": float(roc_auc_score(att[sc], f["p"][sc])),
                         "recall": float(al[att].mean()), "clean_in_alarm_share": float(al[cln].mean()), **forest_shape(f["model"])})  # recall, false-alarm share, tree sizes
        print(f"  C4b {rx} {held}: permuted sessions {mixed}; median AUROC {np.median([r['auroc'] for r in rows if r['receiver'] == rx and r['held_out'] == held]):.3f}")  # progress line
    return pd.DataFrame(rows)  # the results


def c4b_median_perm(b4: pd.DataFrame, rx: str, held: str) -> int:  # function: the permutation number of the median C4b model of one cell
    d = b4[(b4.receiver == rx) & (b4.held_out == held)].sort_values(["auroc", "perm"], kind="stable")  # sorted by held-out AUROC, ties by permutation number
    return int(d.perm.iloc[C4B_MEDIAN_RANK - 1])  # the 50th


def classify(true_mean: float, true_p5: float, c4b_mean: float, prim_auroc: float, selected: bool, ratio: float) -> str:  # function: the classification of one family in one cell
    if prim_auroc < 0.5:  # if: the primary's held-out AUROC is below chance
        return "not interpretable"  # the fold is not classified
    if not selected:  # if: no selected column of the family in the primary
        return "not selected"  # reported as such
    if not (true_mean >= USED_MIN and true_p5 > 0):  # if: the family does not move AUROC reliably
        return "not used"  # not classified further
    return "session carrier" if c4b_mean >= ratio * true_mean else "attack evidence"  # conditional expression: the within-session-permuted model gets as much from it, or not


def classification(ip: pd.DataFrame, ib: pd.DataFrame, prim_metrics: pd.DataFrame, b4: pd.DataFrame, med: dict) -> pd.DataFrame:  # function: the C4b comparison table per cell and family
    rows = []  # results
    for rx, held in c4b_cells():  # for loop over the C4b cells
        pa = float(prim_metrics[(prim_metrics.receiver == rx) & (prim_metrics.held_out == held)].auroc.iloc[0])  # the primary's held-out AUROC
        ba = float(b4[(b4.receiver == rx) & (b4.held_out == held) & (b4.perm == med[(rx, held)])].auroc.iloc[0])  # the median C4b model's held-out AUROC
        q = lambda d: d[(d.receiver == rx) & (d.held_out == held) & (d.grouping == "family") & (d.subset == "all") & (d.metric == "auroc")].set_index("group")  # lambda: one cell's family AUROC rows
        t, b = q(ip), q(ib)  # primary and C4b rows
        for f in FAMILIES:  # for loop over the families
            sel_t, sel_b = t.at[f, "status"] == "selected", b.at[f, "status"] == "selected"  # selected in each model
            tm, tp5 = (float(t.at[f, "drop_mean"]), float(t.at[f, "drop_p5"])) if sel_t else (np.nan, np.nan)  # conditional expression: true-label drop and its 5th percentile
            bm = float(b.at[f, "drop_mean"]) if sel_b else 0.0  # conditional expression: a family the C4b model did not select gives it nothing
            r = {"receiver": rx, "held_out": held, "family": f, "primary_auroc": pa, "c4b_median_perm": med[(rx, held)], "c4b_auroc": ba, "selected_primary": sel_t, "selected_c4b": sel_b, "true_drop_mean": tm, "true_drop_p5": tp5, "c4b_drop_mean": bm, "ratio": bm / tm if sel_t and tm > 0 else np.nan}  # dict literal: the inputs
            r["class"] = classify(tm, tp5, bm, pa, sel_t, RATIO)  # the decision's classification
            for rr in RATIO_SENS:  # for loop over the two sensitivities
                r[f"class_ratio_{rr}"] = classify(tm, tp5, bm, pa, sel_t, rr)  # the classification at that ratio
            rows.append(r)  # kept
    return pd.DataFrame(rows)  # the table


def c6(cells: dict, wc: pd.DataFrame, fl: pd.DataFrame, prim: dict) -> tuple:  # function: forest-seed and MI-seed dispersion over 20260914-20260918
    runs = [prim]  # the primary is the first seed of both
    for s in FOREST_SEEDS[1:]:  # for loop over the other forest seeds (MI seed fixed)
        runs.append(run_config(cells, wc, fl, "primary", mi_seed=SEED, forest_seed=s))  # run again with that forest seed
    for s in MI_SEEDS[1:]:  # for loop over the other MI seeds (forest seed fixed)
        runs.append(run_config(cells, wc, fl, "primary", mi_seed=s, forest_seed=SEED))  # run again with that MI seed
        if S.SEED != SEED:  # if: the screen script's seed was not restored
            raise SystemExit("STOP: the jamming screen's SEED not restored after a C6 run")  # stop
    met = pd.concat([r["metrics"] for r in runs], ignore_index=True)  # every run's metrics
    sel = pd.concat([r["selection"] for r in runs], ignore_index=True)  # every run's selection
    base = sel[(sel.mi_seed == SEED) & (sel.forest_seed == SEED)].set_index(["receiver", "held_out"]).selected_columns  # the primary's selected sets
    sel["jaccard_with_primary"] = [len(set(a.split(";")) & set(base[(r, h)].split(";"))) / len(set(a.split(";")) | set(base[(r, h)].split(";"))) for a, r, h in zip(sel.selected_columns, sel.receiver, sel.held_out)]  # list comprehension: overlap with the primary's 36
    met["varied"] = np.where(met.mi_seed != SEED, "mi_seed", np.where(met.forest_seed != SEED, "forest_seed", "primary"))  # nested conditional: which seed a run varies
    per = met[["receiver", "held_out", "mi_seed", "forest_seed", "varied", "recall", "auroc", "clean_in_alarm_share"]].merge(sel[["receiver", "held_out", "mi_seed", "forest_seed", "jaccard_with_primary"]], on=["receiver", "held_out", "mi_seed", "forest_seed"])  # per run
    disp = []  # dispersion rows
    for v in ("forest_seed", "mi_seed"):  # for loop over the two seeds varied
        d = per[per.varied.isin(["primary", v])]  # the five runs of that seed (the primary included)
        g = d.groupby(["receiver", "held_out"]).agg(recall_min=("recall", "min"), recall_max=("recall", "max"), auroc_min=("auroc", "min"), auroc_max=("auroc", "max"), clean_alarm_min=("clean_in_alarm_share", "min"), clean_alarm_max=("clean_in_alarm_share", "max"), jaccard_min=("jaccard_with_primary", "min")).reset_index()  # named aggregation: range over the five seeds
        disp.append(g.assign(seed_varied=v))  # kept with its label
    return per, pd.concat(disp, ignore_index=True)  # per-run rows and the dispersion table


def logreg(kept: dict) -> tuple:  # function: the logistic regression's no_residual logistic regression refitted on the same prepared folds and selections (the regression script's fit_lr and predict)
    fits, rows = {}, []  # fitted models and the comparison against the published probabilities
    # the logistic regression's published held-out probabilities (no session column, so no 2.6.12 row can exist), read through the regression script's READER with keyword
    # arguments: float_precision "round_trip" parses each written number back to the identical float (pandas' default parser is off
    # by one unit in the last place on 8,879 of these 15,191 values, which the exact check caught)
    pub = M.READER(M.OUT / "logistic_regression_variants_predictions.csv.gz", dtype={"held_out": str}, float_precision="round_trip")
    pub = pub[pub.variant == "no_residual"]  # the no_residual variant
    for (rx, held), kp in kept.items():  # for loop over the folds
        Z = kp["prep"]["Z"][:, kp["sel"]]  # the selected columns
        m, _ = M.fit_lr(Z[kp["tr"]], kp["y"][kp["tr"]], M.C_PRIMARY, {"config": "forest_logreg", "mi_seed": SEED, "receiver": rx, "held_out": held, "perm": -1}, strict=True)  # the regression script's fit with its gradient check; any warning stops
        p, _ = M.predict(m, Z[kp["te"]])  # held-out probabilities, checked elementwise by the regression script
        q = pub[(pub.receiver == rx) & (pub.held_out == held)]  # the published windows of this fold
        same_sec = bool(np.array_equal(q.sec.to_numpy(), kp["idx"].sec.to_numpy()[kp["te"]]))  # the same windows in the same order
        rows.append({"receiver": rx, "held_out": held, "same_windows": same_sec, "max_abs_p_diff": float(np.max(np.abs(q.p.to_numpy() - p))) if same_sec else np.nan})  # dict literal: agreement with the logistic regression
        fits[(rx, held)] = m  # kept
    M.FIT_DIAG.clear()  # the regression script's fit-diagnostic list is not written by this script
    return fits, pd.DataFrame(rows)  # models and the agreement table


def forest_vs_logreg(prim: dict) -> tuple:  # function: the forest's per-fold and pooled figures beside the logistic regression's no_residual logistic regression
    lr = M.read_table(M.OUT / "logistic_regression_variants_metrics_long.csv")  # the logistic regression's per-fold metrics
    lr = lr[lr.variant == "no_residual"]  # the no_residual variant
    cols = ["recall", "recall_jam_spoof", "recall_spoof", "precision", "f1", "clean_in_alarm", "clean", "clean_in_alarm_share", "fa_figure", "fa_episodes", "fa_rate_per_h", "auroc", "average_precision"]  # the compared columns
    per = prim["metrics"][["receiver", "held_out"] + cols].merge(lr[["receiver", "held_out"] + cols], on=["receiver", "held_out"], suffixes=("_forest", "_logreg"), validate="one_to_one")  # one row per fold, both models
    lv = M.read_table(M.OUT / "logistic_regression_variants.csv")  # the logistic regression's pooled figures per variant
    lv = lv[lv.variant == "no_residual"].set_index("receiver")  # the no_residual rows
    pf = prim["pooled"].set_index("receiver")  # the forest's pooled rows
    pooled_rows = []  # pooled comparison
    for rx in RXS:  # for loop over the receivers
        # dict literal: the pooled figures of both models (the folds behind each as in the logistic regression: recall 5, precision/F1 2, AUROC 2 or 3)
        pooled_rows.append({"receiver": rx, "recall_forest": pf.at[rx, "recall_pooled"], "recall_logreg": lv.at[rx, "recall_pooled"], "recall_jam_spoof_forest": pf.at[rx, "recall_jam_spoof_pooled"], "recall_jam_spoof_logreg": lv.at[rx, "recall_jam_spoof"],
                            "recall_spoof_forest": pf.at[rx, "recall_spoof_pooled"], "recall_spoof_logreg": lv.at[rx, "recall_spoof"], "f1_forest": pf.at[rx, "f1_pooled"], "f1_logreg": lv.at[rx, "f1_pooled"],  # per class and F1
                            "clean_share_forest": pf.at[rx, "clean_in_alarm"] / pf.at[rx, "clean_windows"], "clean_share_logreg": lv.at[rx, "clean_in_alarm_share"], "auroc_median_forest": pf.at[rx, "auroc_median"], "auroc_median_logreg": lv.at[rx, "auroc_median"]})  # clean share and median AUROC
    return per, pd.DataFrame(pooled_rows)  # both tables


# ---------------------------------------------------------------- figures

COL = M.COL  # label colours of the logistic regression (colour-blind safe)
C_FOREST, C_LR, C_C5, C_C4B = "#0072B2", "#56B4E9", "#E69F00", "#999999"  # model colours (Okabe-Ito); markers differ as well, so colour is never the only cue


def plot_forest_heldout_probability(pr: pd.DataFrame) -> None:  # function: the figure, held-out probability per window of the forest, per receiver and fold, coloured by label
    fig, ax = plt.subplots(3, 5, figsize=(20, 9), sharey=True)  # 3 receivers x 5 folds
    for i, rx in enumerate(RXS):  # for loop over the receivers
        for j, held in enumerate(SESSIONS):  # for loop over the folds
            a = ax[i, j]  # the panel
            d = pr[(pr.receiver == rx) & (pr.held_out == held)]  # its windows
            t = (d.sec - d.sec.min()) / 60.0  # minutes since the first held-out window
            for lb in ("uncertain", "clean", "attack_jam_spoof", "attack_spoof"):  # for loop over labels, uncertain drawn first
                m = (d.label == lb).to_numpy()  # its windows
                a.plot(t[m], d.p[m], ".", ms=1.5, color=COL[lb], label=lb if (i, j) == (0, 0) else None)  # points
            for on, en in SC.pooled_intervals(held):  # for loop over the pooled attack intervals
                a.axvspan((on - d.sec.min()) / 60.0, (en - d.sec.min()) / 60.0, color="#FFE8CC", zorder=0)  # shaded
            a.axhline(THR, color="k", lw=0.7)  # the threshold
            a.set_ylim(-0.03, 1.03)  # probability range
            a.set_title(f"{RXN[rx]}, held out {held}", fontsize=9)  # panel title
            if i == 2:  # if: bottom row
                a.set_xlabel("minutes since the first held-out window")  # axis label
        ax[i, 0].set_ylabel("forest p(attack)")  # axis label
    ax[0, 0].legend(fontsize=7, markerscale=6, loc="center left")  # legend
    fig.tight_layout()  # layout
    save_fig(fig, GRAPH / "random_forest_heldout_probability.png")  # byte-stable save


def plot_forest_controls(prim: pd.DataFrame, lr: pd.DataFrame, geo: pd.DataFrame, b4: pd.DataFrame, a4: pd.DataFrame) -> None:  # function: the figure, the forest beside the logistic regression, C5, C4b and C4a
    fig, ax = plt.subplots(2, 3, figsize=(16, 8), sharey="row")  # recall row and AUROC row, one column per receiver
    x = np.arange(len(SESSIONS))  # fold positions
    for k, rx in enumerate(RXS):  # for loop over the receivers
        p_ = prim[prim.receiver == rx].set_index("held_out").loc[SESSIONS]  # forest by fold
        l_ = lr[lr.receiver == rx].set_index("held_out").loc[SESSIONS]  # logistic regression by fold
        g_ = geo[geo.receiver == rx].set_index("held_out").loc[SESSIONS]  # C5 by fold
        bm = b4[b4.receiver == rx].groupby("held_out").median(numeric_only=True).reindex(SESSIONS)  # C4b medians by fold (missing where not run)
        for dx, d, c, mk, lb in ((-0.2, p_, C_FOREST, "o", "forest (primary)"), (-0.07, l_, C_LR, "D", "logistic regression, same set"), (0.07, g_, C_C5, "s", "C5 geometry only")):  # for loop over the three models
            ax[0, k].plot(x + dx, d.recall, mk, color=c, label=lb)  # recall (filled)
            ax[0, k].plot(x + dx, d.clean_in_alarm_share, mk, mfc="none", color=c, label=f"{lb}: clean windows in alarm")  # false-alarm share (open)
            ax[1, k].plot(x + dx, d.auroc, mk, color=c, label=lb)  # AUROC
        ax[0, k].plot(x + 0.2, bm.recall, "^", color=C_C4B, label="C4b within-session permutation (median of 100)")  # C4b recall
        ax[1, k].plot(x + 0.2, bm.auroc, "^", color=C_C4B, label="C4b median")  # C4b AUROC
        for j, held in enumerate(SESSIONS):  # for loop over the folds
            v = a4[(a4.receiver == rx) & (a4.held_out == held)].auroc  # C4a distribution
            if len(v):  # if: a C4a fold
                ax[1, k].plot([j + 0.32] * 2, np.percentile(v, [5, 95]), color="k", lw=1)  # 5th-95th percentile
                ax[1, k].plot(j + 0.32, np.median(v), "_", color="k", ms=10, label="C4a median, 5-95 % (100 permutations)" if j == 1 else None)  # median
        ax[1, k].axhspan(*C4A_BAND, color="#EEEEEE", zorder=0)  # the C4a pass band
        for r in (0, 1):  # for loop over the rows
            ax[r, k].set_xticks(x)  # fold ticks
            ax[r, k].set_xticklabels(SESSIONS)  # fold names
            ax[r, k].set_xlim(-0.6, len(SESSIONS) - 0.4)  # every fold shown in both rows
            ax[r, k].set_ylim(-0.03, 1.03)  # range
        ax[0, k].set_title(f"{RXN[rx]}", fontsize=10)  # column title
        ax[1, k].set_xlabel("held-out session")  # axis label
    ax[0, 0].set_ylabel("recall (filled) / clean windows in alarm (open)")  # axis label
    ax[1, 0].set_ylabel("held-out AUROC (where both classes exist)")  # axis label
    ax[0, 0].legend(fontsize=7, loc="lower left")  # legend
    ax[1, 0].legend(fontsize=7, loc="lower left")  # legend
    fig.tight_layout()  # layout
    save_fig(fig, GRAPH / "random_forest_controls.png")  # byte-stable save


def plot_importance_within_session(cl: pd.DataFrame, ip: pd.DataFrame) -> None:  # function: the figure, grouped importance (AUROC drop) with the true labels beside the median C4b model, per C4b cell
    cols = ["2.6.2", "2.6.4", "2.6.10"]  # the C4b folds
    fig, ax = plt.subplots(3, 3, figsize=(16, 11), sharex=True)  # receivers x folds
    yy = np.arange(len(FAMILIES))[::-1]  # family positions, first at the top
    for i, rx in enumerate(RXS):  # for loop over the receivers
        for j, held in enumerate(cols):  # for loop over the folds
            a = ax[i, j]  # the panel
            d = cl[(cl.receiver == rx) & (cl.held_out == held)].set_index("family")  # the classification rows of this cell
            if d.empty:  # if: no C4b cell (2.6.10 on the Septentrio receivers has no clean window)
                a.text(0.5, 0.5, "no clean windows held out", ha="center", va="center", transform=a.transAxes, fontsize=9, color="#555555")  # note
                a.set_yticks([])  # no family axis
                continue  # next panel
            q = ip[(ip.receiver == rx) & (ip.held_out == held) & (ip.grouping == "family") & (ip.subset == "all") & (ip.metric == "auroc")].set_index("group")  # the primary's AUROC rows (for the percentiles)
            for f, yv in zip(FAMILIES, yy):  # for loop over the families
                if d.at[f, "selected_primary"]:  # if: the family is selected in the primary
                    a.plot([q.at[f, "drop_p5"], q.at[f, "drop_p95"]], [yv + 0.12] * 2, color=C_FOREST, lw=1)  # 5th-95th percentile over the repeats
                    a.plot(d.at[f, "true_drop_mean"], yv + 0.12, "o", color=C_FOREST, label="true labels (mean, 5-95 %)" if f == FAMILIES[0] else None)  # true-label mean drop
                a.plot(d.at[f, "c4b_drop_mean"], yv - 0.12, "^", color=C_C4B, label="median C4b model" if f == FAMILIES[0] else None)  # C4b mean drop (0 where not selected)
                a.text(1.01, yv, d.at[f, "class"], transform=a.get_yaxis_transform(), fontsize=6.5, va="center", color="#333333")  # the classification, right of the panel
            a.axvline(0, color="k", lw=0.6)  # zero drop
            a.axvline(USED_MIN, color="#BBBBBB", lw=0.6, ls="--")  # the "used" threshold
            a.set_yticks(yy)  # family ticks
            a.set_yticklabels(FAMILIES if j == 0 else [""] * len(FAMILIES), fontsize=7)  # conditional expression: names on the first column only
            a.set_title(f"{RXN[rx]}, held out {held}: AUROC forest {d.primary_auroc.iloc[0]:.2f}, C4b {d.c4b_auroc.iloc[0]:.2f}", fontsize=8)  # panel title with both held-out AUROCs
            if i == 2:  # if: bottom row
                a.set_xlabel("mean drop in held-out AUROC when the family is permuted")  # axis label
    ax[0, 0].legend(fontsize=7, loc="lower right")  # legend
    fig.tight_layout()  # layout
    save_fig(fig, GRAPH / "random_forest_importance_c4b.png")  # byte-stable save


def plot_importance_per_class(ip: pd.DataFrame) -> None:  # function: the figure, recall drop per family and attack class in every fold (per-class consistency)
    fam_all = ip[(ip.grouping == "family") & (ip.subset == "all")]  # every family row (columns only where the class is held out)
    combos = [(h, c) for h in SESSIONS for c in ("recall_jam_spoof", "recall_spoof") if fam_all[(fam_all.held_out == h) & (fam_all.metric == c)].base.notna().any()]  # list comprehension with a filter: fold x class pairs that exist (no jamming phase in 2.6.2 or 2.6.10)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))  # one heatmap per receiver
    vmax = 0.0  # common colour scale
    mats = []  # one matrix per receiver
    for rx in RXS:  # for loop over the receivers
        q = ip[(ip.receiver == rx) & (ip.grouping == "family") & (ip.subset == "all")]  # its family rows
        Mx = np.full((len(FAMILIES), len(combos)), np.nan)  # families x combos, empty where not defined
        for c_, (h, met) in enumerate(combos):  # for loop over the columns
            r = q[(q.held_out == h) & (q.metric == met)].set_index("group")  # this fold's rows of that metric
            for f_, f in enumerate(FAMILIES):  # for loop over the families
                Mx[f_, c_] = r.at[f, "drop_mean"] if f in r.index else np.nan  # conditional expression: the mean drop (NaN where the class is absent or the family not selected)
        mats.append(Mx)  # kept
        vmax = max(vmax, np.nanmax(np.abs(Mx)) if np.isfinite(Mx).any() else 0.0)  # the largest magnitude so far
    vmax = max(vmax, 0.05)  # at least a visible scale
    for k, (rx, Mx) in enumerate(zip(RXS, mats)):  # for loop over the receivers and their matrices
        a = ax[k]  # the panel
        im = a.imshow(Mx, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")  # diverging colours with a grey midpoint at 0
        for f_ in range(len(FAMILIES)):  # for loop over the rows
            for c_ in range(len(combos)):  # for loop over the columns
                v = Mx[f_, c_]  # the value
                a.text(c_, f_, "-" if not np.isfinite(v) else f"{v:.2f}", ha="center", va="center", fontsize=6, color="#222222")  # conditional expression: the value, or a dash where undefined
        a.set_xticks(range(len(combos)))  # column ticks
        a.set_xticklabels([f"{h} {'jam+spoof' if c == 'recall_jam_spoof' else 'spoof'}" for h, c in combos], fontsize=7, rotation=90)  # list comprehension: fold and class, set vertical so neighbouring labels cannot collide
        a.set_yticks(range(len(FAMILIES)))  # row ticks
        a.set_yticklabels(FAMILIES if k == 0 else [""] * len(FAMILIES), fontsize=7)  # conditional expression: names on the first panel only
        a.set_title(RXN[rx], fontsize=10)  # panel title
    fig.colorbar(im, ax=list(ax), shrink=0.8, label="mean drop in recall at 0.5 (negative: recall rises)")  # one shared colour bar
    save_fig(fig, GRAPH / "random_forest_importance_per_class.png")  # byte-stable save


def plot_forest_permutation(a4: pd.DataFrame) -> None:  # function: the figure, the C4a AUROC distribution of every cell with the pass band
    fig, ax = plt.subplots(figsize=(10, 4.5))  # one panel
    cells = [(rx, h) for rx in RXS for h in C4A_FOLDS]  # list comprehension: the six cells
    for i, (rx, h) in enumerate(cells):  # for loop over the cells
        v = np.sort(a4[(a4.receiver == rx) & (a4.held_out == h)].auroc.to_numpy())  # its 100 AUROCs, sorted
        ax.plot(i + np.linspace(-0.3, 0.3, len(v)), v, ".", ms=3, color=C_FOREST)  # spread along x in rank order (a strip plot)
        ax.plot([i - 0.35, i + 0.35], [np.median(v)] * 2, color="#D55E00", lw=2)  # the median
    ax.axhspan(*C4A_BAND, color="#EEEEEE", zorder=0)  # the pass band
    ax.axhline(0.5, color="k", lw=0.7)  # chance
    ax.set_xticks(range(len(cells)))  # cell ticks
    ax.set_xticklabels([f"{RXN[r]}\nheld out {h}" for r, h in cells], fontsize=8)  # list comprehension: cell names
    ax.set_ylabel("held-out AUROC of the forest, labels permuted in training")  # axis label
    ax.set_ylim(0, 1)  # range
    fig.tight_layout()  # layout
    save_fig(fig, GRAPH / "random_forest_permutation_c4a.png")  # byte-stable save


def plot_imputation_diagnostic(ip: pd.DataFrame) -> None:  # function: the figure, family recall drop on all scored windows against the windows with no imputed selected value
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)  # one panel per receiver
    for k, rx in enumerate(RXS):  # for loop over the receivers
        q = ip[(ip.receiver == rx) & (ip.grouping == "family") & (ip.metric == "recall") & (ip.status == "selected")]  # selected families' recall rows
        w = q.pivot_table(index=["held_out", "group"], columns="subset", values="drop_mean").dropna()  # one row per fold and family, both subsets
        a = ax[k]  # the panel
        a.plot(w["all"], w["no_imputed"], "o", ms=4, color=C_FOREST, alpha=0.8)  # one point per fold and family
        lim = [-0.05, max(0.1, float(np.nanmax(np.abs(w.to_numpy()))) * 1.05) if len(w) else 0.1]  # axis range covering every point
        a.plot(lim, lim, color="#999999", lw=0.8)  # the identity line
        a.set_title(RXN[rx], fontsize=10)  # panel title
        a.set_xlabel("recall drop, all scored held-out windows")  # axis label
    ax[0].set_ylabel("recall drop, windows with no imputed selected value")  # axis label
    fig.tight_layout()  # layout
    save_fig(fig, GRAPH / "random_forest_imputation_diagnostic.png")  # byte-stable save


# ---------------------------------------------------------------- main

def main() -> None:  # function: the forest from start to finish
    t0 = time.time()  # the clock
    wc, fl = M.load_meta()  # dictionary and lists (through the regression script's reader)
    rule = family_rule_table(wc)  # the family rule applied to every column
    rule_text = rule.to_csv(index=False, lineterminator="\n")  # its text
    if "--rule-only" in sys.argv:  # if: only the rule table is wanted (written and committed before any importance)
        (CHECK_DIR / "random_forest_family_rule.csv").write_text(rule_text)  # written
        print(f"family rule written: {len(rule)} rows; unplaced {int(rule.family.str.contains('UNPLACED').sum())}")  # progress line
        return  # nothing else
    OUT.mkdir(exist_ok=True)  # the output folder
    n_p, bad = protected_ok()  # protected files at the start
    check("Protected files at start (expected_hashes_random_forest.txt)", "139 unchanged", f"{n_p} compared, {len(bad)} changed {bad}", n_p == 139 and not bad)  # the check
    check("Family rule: the committed random_forest_family_rule.csv equals the rule applied now; no column unplaced", "identical; 0 unplaced", f"identical {(CHECK_DIR / "random_forest_family_rule.csv").read_text() == rule_text}; unplaced {int(rule.family.str.contains('UNPLACED').sum())}", (CHECK_DIR / "random_forest_family_rule.csv").read_text() == rule_text and not rule.family.str.contains("UNPLACED").any())  # the check
    (CHECK_DIR / "random_forest_family_rule.csv").write_text(rule_text)  # written in the full run too, after the check above compared it with the committed file
    fo = S.folds()  # the screen script's folds
    check("Folds: five leave-one-session-out folds; 2.6.12 in no training list", "true", str(fo), all(h not in t and M.SEALED not in t and len(t) == 4 for h, t in fo))  # generator expression: the check
    cells = {rx: M.load_cell(W, rx) for rx in RXS}  # dict comprehension: every receiver's windows, 2.6.12 dropped
    check("Sealed: no 2.6.12 row in any loaded window table", "0", sum(int((c[0].test_code == M.SEALED).sum()) for c in cells.values()), all(not (c[0].test_code == M.SEALED).any() for c in cells.values()))  # generator expressions: the check
    co = core(cells, wc, fl)  # the primary, C5 and the primary's importance
    prim, geo, fam, xs = co["prim"], co["geo"], co["fam"], co["xs"]  # its parts
    m = prim["metrics"]  # primary metrics
    ind = {rx: pd.crosstab(c[0].test_code, c[0].label) for rx, c in cells.items()}  # dict comprehension: windows per session and label, from the index files
    ok_cnt = all(int(ind[r.receiver].loc[[t for t in SESSIONS if t != r.held_out], "clean"].sum()) == r.train_clean and int(ind[r.receiver].at[r.held_out, "clean"]) == r.clean and int(ind[r.receiver].loc[r.held_out, list(ATTACK)].sum()) == r.attack_jam_spoof + r.attack_spoof for r in m.itertuples())  # generator expression: every fold's counts agree
    check("Window counts per fold equal an independent crosstab of the index files (and the logistic regression Table 60)", "all equal", f"{len(m)} folds", ok_cnt)  # the check
    lr_selection = M.read_table(M.OUT / "logistic_regression_variants_selection.csv")  # the logistic regression's selections
    lr_selection = lr_selection[lr_selection.variant == "no_residual"].set_index(["receiver", "held_out"]).selected_columns  # the no_residual logistic regression's 36 columns per fold
    eq = [(r.receiver, r.held_out) for r in prim["selection"].itertuples() if r.selected_columns == lr_selection[(r.receiver, r.held_out)]]  # list comprehension: folds whose selection equals the logistic regression's
    check("(b): the forest's selected columns equal the logistic regression's no_residual selection, column for column and in order", "15 of 15", f"{len(eq)} of {len(prim['selection'])}", len(eq) == 15)  # the check
    sel_fams = sorted({fam[c] for s in prim["selection"].selected_columns for c in s.split(";")})  # set comprehension: every family that appears among the selections
    check("Every selected column of the primary falls in one of the nine primary families", "all inside", sel_fams, set(sel_fams) <= set(FAMILIES))  # the check
    check("Held-out forest probabilities exactly equal to 0.5 (alarm at p >= 0.5)", "reported", int(prim["selection"].p_equal_threshold.sum()), None)  # reported
    ip = co["files"]["random_forest_importance_primary.csv"]  # the primary's importance
    base = ip[(ip.grouping == "family") & (ip.subset == "all") & (ip.group == FAMILIES[0])].pivot_table(index=["receiver", "held_out"], columns="metric", values="base")  # baselines inside the importance code
    mm = m.set_index(["receiver", "held_out"])  # the metric table by fold
    d_rec = float(np.nanmax(np.abs(base["recall"] - mm.loc[base.index, "recall"])))  # largest recall disagreement
    d_auc = float(np.nanmax(np.abs(base["auroc"] - mm.loc[base.index, "auroc"])))  # largest AUROC disagreement (folds with both classes)
    check("Importance baselines equal the regression script's metric code (recall in 15 folds, AUROC in the folds with both classes)", "0", f"recall {d_rec:.2g}, AUROC {d_auc:.2g}", d_rec == 0 and d_auc == 0)  # the check
    kp = prim["keep"][("SBF_02", "2.6.5")]  # one fold, the largest training set
    m1 = RandomForestClassifier(**RF_PARAMS, random_state=SEED, n_jobs=1).fit(kp["prep"]["Z"][kp["tr"]][:, kp["sel"]], kp["y"][kp["tr"]])  # the same forest fitted serially
    same_trees = all(np.array_equal(getattr(a.tree_, f), getattr(b.tree_, f)) for a, b in zip(m1.estimators_, kp["model"].estimators_) for f in ("feature", "threshold", "value", "children_left", "children_right"))  # generator expression: every tree array equal
    p_again = proba(kp["model"], kp["prep"]["Z"][kp["te"]][:, kp["sel"]])  # the held-out prediction repeated
    check("the forest fitted with n_jobs 8 equals the serial fit tree for tree; serial predictions repeat bitwise (X5, held-out 2.6.5)", "true, true", f"{same_trees}, {bool(np.array_equal(p_again, kp['p']))}", same_trees and bool(np.array_equal(p_again, kp["p"])))  # the check
    print("  logistic regression (the logistic regression no_residual) refit and importance ...")  # progress line
    lr_fits, lr_agree = logreg(prim["keep"])  # refitted
    check("Logistic regression refit equals the logistic regression's published no_residual probabilities (same windows, largest |difference|)", "15 of 15 same windows; 0", f"{int(lr_agree.same_windows.sum())} of 15; {lr_agree.max_abs_p_diff.max():.3g}", bool(lr_agree.same_windows.all()) and float(lr_agree.max_abs_p_diff.max()) == 0.0)  # the check
    lr_s, lr_r = [], []  # the logistic regression's importance rows
    for (rx, held), kp_ in prim["keep"].items():  # for loop over the folds
        s_, r_, _ = importance_cell((lambda mdl: (lambda Z: M.predict(mdl, Z)[0]))(lr_fits[(rx, held)]), kp_, kp_["names"], kp_["sel"], fam, xs, {"config": "logreg_no_residual", "model": "logreg", "receiver": rx, "held_out": held})  # immediately-applied lambda binds this fold's model
        lr_s += s_  # kept
        lr_r += r_  # kept
    M.FIT_DIAG.clear()  # nothing of the regression script's diagnostics is written here
    print("  C4a ...")  # progress line
    a4 = c4a(prim["keep"])  # C4a
    med = a4.groupby(["receiver", "held_out"]).auroc.median()  # median per cell
    ok4 = bool(((med >= C4A_BAND[0]) & (med <= C4A_BAND[1])).all() and len(med) == 6)  # all six medians inside the band
    check("C4a permutation control: median held-out AUROC within 0.5 +/- 0.1 in all six cells (100 permutations each)", "all six in [0.4, 0.6]", "; ".join(f"{RXN[r]} {h}: {v:.3f}" for (r, h), v in med.items()), ok4)  # the check
    print("  C4b ...")  # progress line
    b4 = c4b(prim["keep"])  # C4b
    medp = {c: c4b_median_perm(b4, *c) for c in c4b_cells()}  # dict comprehension: the median permutation of every cell
    ib_s, ib_r = [], []  # the median C4b models' importance rows
    for (rx, held), k in medp.items():  # for loop over the C4b cells
        kp_ = prim["keep"][(rx, held)]  # the prepared fold
        f_ = c4b_fit(kp_, k)  # the median model refitted (deterministic)
        p_ok = float(b4[(b4.receiver == rx) & (b4.held_out == held) & (b4.perm == k)].auroc.iloc[0])  # its AUROC in the C4b table
        lab = kp_["idx"].label.to_numpy()[kp_["te"]]  # held-out labels
        sc = np.isin(lab, SCORED)  # scored windows
        if float(roc_auc_score(np.isin(lab[sc], ATTACK), f_["p"][sc])) != p_ok:  # if: the refit does not reproduce the table (cannot happen; asserted)
            raise SystemExit(f"STOP: C4b median model of {rx} {held} not reproduced")  # stop
        s_, r_, _ = importance_cell((lambda mdl: (lambda Z: proba(mdl, Z)))(f_["model"]), kp_, f_["names"], f_["sel"], fam, xs, {"config": "c4b_median", "model": "forest", "receiver": rx, "held_out": held})  # immediately-applied lambda binds this model
        ib_s += s_  # kept
        ib_r += r_  # kept
    ib = pd.DataFrame(ib_s)  # the C4b importance table
    cl = classification(ip, ib, m, b4, medp)  # the decision's classification
    print("  C6 ...")  # progress line
    c6_runs, c6_disp = c6(cells, wc, fl, prim)  # C6
    check("C6: the jamming screen's SEED restored after every MI run", str(SEED), S.SEED, S.SEED == SEED)  # the check
    per_cmp, pooled_cmp = forest_vs_logreg(prim)  # forest beside the logistic regression
    files = dict(co["files"])  # the core tables
    files.update({"random_forest_importance_c4b.csv": ib, "random_forest_importance_repeats_c4b.csv.gz": pd.DataFrame(ib_r), "random_forest_importance_logreg.csv": pd.DataFrame(lr_s), "random_forest_importance_repeats_logreg.csv.gz": pd.DataFrame(lr_r),  # dict update: the other importance tables
                  "random_forest_c4b_classification.csv": cl, "random_forest_c4a_permutation.csv": a4, "random_forest_c4b_within_session.csv": b4, "random_forest_c6_seeds.csv": c6_runs, "random_forest_c6_dispersion.csv": c6_disp,  # controls and the classification
                  "random_forest_forest_vs_logreg.csv": per_cmp, "random_forest_forest_vs_logreg_pooled.csv": pooled_cmp, "random_forest_logreg_refit_agreement.csv": lr_agree})  # the comparison tables
    write_files(files)  # every table written
    plot_forest_heldout_probability(prim["predictions"])  # the figure
    lrm = M.read_table(M.OUT / "logistic_regression_variants_metrics_long.csv")  # the logistic regression's per-fold metrics
    plot_forest_controls(m, lrm[lrm.variant == "no_residual"], geo["metrics"], b4, a4)  # the figure
    plot_importance_within_session(cl, ip)  # the figure
    plot_importance_per_class(ip)  # the figure
    plot_forest_permutation(a4)  # the figure
    plot_imputation_diagnostic(ip)  # the figure
    for rx in RXS:  # for loop over the receivers: numbers the summary quotes
        p_ = prim["pooled"][prim["pooled"].receiver == rx].iloc[0]  # pooled row
        NUM[f"{rx}_recall_pooled"] = round(float(p_.recall_pooled), 3)  # pooled recall
        NUM[f"{rx}_auroc_median"] = round(float(p_.auroc_median), 3)  # median AUROC
    NUM["classes"] = cl["class"].value_counts().to_dict().__repr__()  # the classification counts
    n_p, bad = protected_ok()  # protected files at the end
    check("Protected files at end", "139 unchanged", f"{n_p} compared, {len(bad)} changed {bad}", n_p == 139 and not bad)  # the check
    M.write_csv(pd.DataFrame(CHECKS), CHECK_DIR / "random_forest_checks.csv")  # checks
    M.write_csv(pd.DataFrame([{"key": k, "value": v} for k, v in NUM.items()]), CHECK_DIR / "random_forest_numbers.csv")  # list comprehension: numbers
    print(f"done in {time.time() - t0:.0f} s; failed checks: {sum(c['ok'] is False for c in CHECKS)}")  # run time and failures
    if any(c["ok"] is False for c in CHECKS):  # if: a check failed
        sys.exit(1)  # exit status 1, so a failure cannot be missed


if __name__ == "__main__":  # guard: runs main() only when the file is executed, not when imported
    main()  # entry point
