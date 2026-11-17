#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
Part 4 of the feature build: the jamming screen, the fold-index functions, the session-fingerprint probe and
the fold table. Nothing is fitted on the whole data set: every quantity that is fitted takes the training sessions of one
leave-one-session-out fold: train on four of 2.6.5, 2.6.2, 2.6.10, 2.6.4, 2.6.11; 2.6.12 never in training).

Jamming screen (rules fixed before the screen was applied): per window length x receiver x fold, on the training sessions' windows
only, every window column gets S_j = |2 AUROC - 1| for attack_jam_spoof against clean and S_s for attack_spoof against clean (equal to
|Cliff's delta|, non-missing values). Reasons: "not_assessable" (fewer than 30 non-missing windows in a group), "jamming" (S_j > S_s and S_j >= 0.147, the lower bound
of a small effect for |Cliff's delta| (Romano et al. 2006), the rule without the floor is kept as a sensitivity: reason_no_floor, lists "screened_no_floor", rule "no_floor"),
"missingness" (|missing share jam - clean| exceeds |spoof - clean| by 0.10 or more). The screened list drops all three.
Two named column groups are reported with and without: the circularity candidates (residual-block availability
self_{G,R,E,C}_n_res, self_solved, self_pos_held, which reflect the same physical event as the receivers' own-fix label evidence) and
the geometry change statistics (window std, skewness and kurtosis of the geometry block).

Session-fingerprint probe (report only, selects nothing): per receiver at w = 10, a depth-4 decision tree (fixed
seed, missing values handled by the tree) predicts the session from the geometry block, the main set and the orbit-free (no-residual)
set, scored by 5-fold grouped cross-validation over contiguous 300 s blocks with a 10 s purge around every held-out block.

The fold-index functions (robust scaler, Pearson pruning at 0.89, mutual-information ranking of the top 36, Pervysheva et al. 2025's
settings) are written here and NOT applied in this task, except inside the leakage tests of tests/test_features.py.

Reads   Data/processed/features/w{1,2,3,5,10}/*, features_window_columns.csv, features_base_1hz.csv.gz, labels/labels_1hz.csv (training
        groups, fold table, coupling report), labels/scoring_fold_policy.csv
Writes  Data/processed/features/screen.csv.gz (every flag, per cell and column), feature_lists.csv.gz (main and screened
        lists per cell and set), Data/processed/checks/screen_screen_summary.csv, screen_probe.csv, screen_fold_table.csv,
        screen_coupling.csv, screen_checks.csv, Graph/jamming_screen.png, Graph/session_probe.png
"""

from __future__ import annotations  # future import: type hints kept as text (a convention across the project's scripts)

import importlib.util  # standard-library module loader, for modules whose names start with a digit
import os  # operating-system access: MPLCONFIGDIR is set before matplotlib loads
import random  # standard-library random; seeded below
import sys  # standard-library system access: sys.path and sys.modules
import time  # standard-library clock, for the run-time line
from pathlib import Path  # Path objects for building file paths

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
os.environ["MPLCONFIGDIR"] = str(PROJECT / "codes" / ".mplconfig")  # matplotlib's cache inside the project folder

import numpy as np  # NumPy: arrays
import pandas as pd  # pandas: tables
import matplotlib  # plotting library
matplotlib.use("Agg")  # file-only backend, no window
import matplotlib.pyplot as plt  # the pyplot interface
from sklearn.feature_selection import mutual_info_classif  # mutual information between each feature and the label (fold-index function only)
from sklearn.model_selection import GroupKFold  # cross-validation that keeps every group (time block) on one side
from sklearn.tree import DecisionTreeClassifier  # a shallow decision tree, which accepts missing values (probe only)

sys.path.insert(0, str(PROJECT / "codes" / "common"))  # puts codes/ first on the import path so the helpers module is found
from helpers import exit_if_failed, save_fig, cliffs_delta  # byte-stable figure saving; Cliff's delta = 2 AUROC - 1; exit_if_failed ends the run on a failed check

SEED = 20260914  # integer constant: the project's fixed random seed
random.seed(SEED)  # seeds Python's generator
np.random.seed(SEED)  # seeds NumPy's generator

PROC = PROJECT / "Data" / "processed"  # processed tables
OUT = PROC / "features"  # this section's tables
CHECK_DIR = PROJECT / "Data" / "processed" / "checks"  # check tables
GRAPH = PROJECT / "Graph"  # figures

KEYS = ["test_code", "receiver", "sec"]  # key columns
RXS = ["SBF_02", "SBF_03", "UBX_01"]  # the three receivers
RXN = {"SBF_02": "mosaic-X5", "SBF_03": "mosaic-H main", "UBX_01": "u-blox 01"}  # names for figures and tables
POSITION_SESSIONS = ["2.6.5", "2.6.2", "2.6.10", "2.6.4", "2.6.11"]  # list: the five position-spoofing sessions, the leave-one-session-out folds
TRANSFER = "2.6.12"  # the attack-type transfer session, never in training
WINDOWS = [1, 2, 3, 5, 10]  # window lengths
PARTS = ["orbitfree", "geometry", "self", "known"]  # window file parts
MIN_N = 30  # a group needs 30 non-missing windows for a separation to be assessed
MISS_DIFF = 0.10  # missingness reason: the jamming group's missing-share gap exceeds the spoofing group's by this much
# the floor fixed before the screen ran: S_jam must reach 0.147, the lower bound of a small effect for |Cliff's delta| (Romano et al. 2006);
# the thresholds were checked against the R effsize package documentation, which cites Romano, Kromrey, Coraggio and Skowronek (2006)
JAM_FLOOR = 0.147  # float constant: the jamming reason needs S_jam >= 0.147 as well as S_jam > S_spoof
CIRC_BASE = ["self_G_n_res", "self_R_n_res", "self_E_n_res", "self_C_n_res", "self_solved", "self_pos_held"]  # list: the circularity candidates' base columns
CHANGE = {"std", "skew", "kurt"}  # window statistics that measure change, not level
PEARSON_T, MI_K = 0.89, 36  # Pervysheva et al. (2025): drop one of a pair with |r| >= 0.89; keep the top 36 by mutual information
BLOCK_S, PURGE_S, PROBE_DEPTH = 300, 10, 4  # probe: 300 s time blocks, 10 s purge, tree depth 4

CHECKS: list = []  # list: the checks, written to screen_checks.csv
FAILED: list = []  # list: names of the self-checks that failed; the script ends the run with status 1 when it is not empty
NUM: dict = {}  # dict: numbers the summary quotes
SET = {"root": OUT, "train_label": False}  # dict: where the windows are read from and whether the screen takes the training label (registered: features/, scoring label)


def check(name: str, expected: str, found, ok: bool | None) -> None:  # function: records one check and prints it
    CHECKS.append({"check": name, "expected": expected, "found": found, "ok": "" if ok is None else bool(ok)})  # append a dict row
    if ok is not None and not bool(ok):  # if: a pass/fail check (None marks a reported number) that did not pass
        FAILED.append(name)  # its name kept for the exit at the end of the run
    print(f"  [{'ok' if ok else ('FAIL' if ok is False else '--')}] {name}: expected {expected}; found {found}")  # progress line


def load(name: str, filename: str):  # function: loads codes/<filename> as a module (names start with a digit)
    spec = importlib.util.spec_from_file_location(name, PROJECT / "codes" / filename)  # where to load from
    mod = importlib.util.module_from_spec(spec)  # an empty module object
    sys.modules[name] = mod  # registered, so the module's own imports resolve
    spec.loader.exec_module(mod)  # runs the file's top level (its main() is guarded and not run)
    return mod  # the module


# ---------------------------------------------------------------- folds and the fold-index functions (written, not applied here)

def folds() -> list:  # function: the leave-one-session-out folds: (held-out session, training sessions)
    return [(s, [t for t in POSITION_SESSIONS if t != s]) for s in POSITION_SESSIONS]  # list comprehension: each position session held out once; 2.6.12 is in no training list


def fit_robust_scaler(X: np.ndarray, train: np.ndarray) -> tuple:  # function: median and interquartile range per column from the training rows only
    T = X[train]  # training rows only
    med = np.nanmedian(T, axis=0)  # medians
    iqr = np.nanpercentile(T, 75, axis=0) - np.nanpercentile(T, 25, axis=0)  # interquartile ranges
    return med, np.where(iqr > 0, iqr, 1.0)  # a zero range is replaced by 1 so a constant column is only centred


def apply_scaler(X: np.ndarray, params: tuple) -> np.ndarray:  # function: centres and scales every row with training-fold parameters
    return (X - params[0]) / params[1]  # broadcasting over rows


def pearson_prune(X: np.ndarray, train: np.ndarray, thr: float = PEARSON_T) -> list:  # function: column indices kept after dropping the later column of every pair with |r| >= thr on the training rows
    T = pd.DataFrame(X[train])  # training rows as a table (pandas correlation skips missing values pairwise)
    r = T.corr().abs().to_numpy()  # absolute Pearson correlations
    keep = []  # kept columns
    for j in range(X.shape[1]):  # for loop over the columns in dictionary order
        if all(not (r[j, k] >= thr) for k in keep):  # generator expression: not too correlated with any kept column (NaN correlations do not prune)
            keep.append(j)  # kept
    return keep  # indices


def mi_rank(X: np.ndarray, y: np.ndarray, train: np.ndarray, k: int = MI_K) -> list:  # function: the top-k columns by mutual information with the label, on the training rows (missing values filled with the training median)
    T = X[train]  # training rows
    med = np.nanmedian(T, axis=0)  # training medians
    T = np.where(np.isnan(T), np.where(np.isnan(med), 0.0, med), T)  # filled
    mi = mutual_info_classif(T, y[train], random_state=SEED)  # mutual information, fixed seed
    return list(np.argsort(-mi, kind="stable")[:k])  # the k largest, ties in column order


# ---------------------------------------------------------------- data

def load_windows(w: int, rx: str) -> tuple:  # function: one window length and receiver: the index (keys, label, scored) and all window columns
    d = SET["root"] / f"w{w}"  # the folder (features/ for the registered set, features_final/ for the final set)
    idx = pd.read_csv(d / f"{rx}_index.csv.gz")  # keys, label, scoring mask
    if SET["train_label"]:  # if: the final set: the screen's groups follow the strict training label, as the design-choice study's STRICT run refitted its screen
        idx["label"] = np.where(idx.train_label == "clean", "clean", np.where(idx.train_label == "attack", idx.label, "none"))  # nested np.where: clean, the scoring attack name, or no group
    parts = [pd.read_csv(d / f"{rx}_{pt}.csv.gz") for pt in PARTS]  # list comprehension: the four feature parts
    X = parts[0]  # start from the first part
    for p in parts[1:]:  # for loop over the others
        X = X.merge(p, on=KEYS, how="inner", validate="one_to_one")  # joined on the keys (one row per window in each)
    X = idx[KEYS].merge(X, on=KEYS, how="left", validate="one_to_one")  # in the index's order
    return idx.reset_index(drop=True), X.drop(columns=KEYS).reset_index(drop=True)  # index and features, aligned row by row


def groups_of(wc: pd.DataFrame, w: int) -> dict:  # function: named column groups at one window length
    x = wc[wc.w == w]  # this length
    return {"circularity": set(x[x.base_column.isin(CIRC_BASE)].column),  # the circularity candidates' window columns
            "geometry_change": set(x[(x.part == "geometry") & x.stat.isin(CHANGE)].column)}  # the geometry change statistics


# ---------------------------------------------------------------- the jamming screen

def screen_cell(idx: pd.DataFrame, X: pd.DataFrame, train_sessions: list) -> pd.DataFrame:  # function: the screen of one cell, from the training sessions' windows only
    tr = idx.test_code.isin(train_sessions).to_numpy()  # windows of the training sessions
    if set(idx.test_code[tr]) - set(train_sessions):  # if: a window from another session slipped in (group leakage)
        raise SystemExit("STOP: a held-out window in the training rows")  # never allowed
    lab = idx.label.to_numpy()  # labels
    c, j, s = tr & (lab == "clean"), tr & (lab == "attack_jam_spoof"), tr & (lab == "attack_spoof")  # the three groups, training sessions only
    rows = []  # one row per column
    V = X.to_numpy(dtype=float)  # the values
    for k, col in enumerate(X.columns):  # for loop over the window columns
        vc, vj, vs = V[c, k], V[j, k], V[s, k]  # the column in each group
        nc, nj, ns = int(np.isfinite(vc).sum()), int(np.isfinite(vj).sum()), int(np.isfinite(vs).sum())  # non-missing counts
        mc, mj, ms = 1 - nc / max(c.sum(), 1), 1 - nj / max(j.sum(), 1), 1 - ns / max(s.sum(), 1)  # missing shares
        sj = abs(cliffs_delta(vj, vc)) if nj and nc else np.nan  # conditional expression: jamming separation |2 AUROC - 1|
        ss = abs(cliffs_delta(vs, vc)) if ns and nc else np.nan  # spoofing separation
        reasons, reasons_nf = [], []  # why the column is flagged, under the floor rule and under the no-floor rule (sensitivity)
        if min(nc, nj, ns) < MIN_N:  # if: too few values to judge
            reasons.append("not_assessable")  # flagged
            reasons_nf.append("not_assessable")  # and under the no-floor rule
        else:  # else: both separations exist
            if sj > ss and sj >= JAM_FLOOR:  # if: it separates jamming from clean more than spoofing, and at least by a small effect
                reasons.append("jamming")  # flagged
            if sj > ss:  # if: the no-floor rule of part 4 (kept as a sensitivity)
                reasons_nf.append("jamming")  # flagged
        if abs(mj - mc) - abs(ms - mc) >= MISS_DIFF:  # if: its missingness follows jamming more than spoofing
            reasons.append("missingness")  # flagged
            reasons_nf.append("missingness")  # flagged under both rules
        rows.append((col, nc, nj, ns, mc, mj, ms, sj, ss, ";".join(reasons), ";".join(reasons_nf)))  # one row
    return pd.DataFrame(rows, columns=["column", "n_clean", "n_jam", "n_spoof", "miss_clean", "miss_jam", "miss_spoof", "S_jam", "S_spoof", "reason", "reason_no_floor"])  # the cell


def run_screen(wc: pd.DataFrame) -> pd.DataFrame:  # function: every cell (window length x receiver x fold)
    out = []  # cells
    for w in WINDOWS:  # for loop over the window lengths
        for rx in RXS:  # for loop over the receivers
            t0 = time.time()  # the clock
            idx, X = load_windows(w, rx)  # its windows
            for held, train in folds():  # for loop over the folds
                cell = screen_cell(idx, X, train)  # the screen on the training sessions
                cell.insert(0, "held_out", held)  # fold
                cell.insert(0, "receiver", rx)  # receiver
                cell.insert(0, "w", w)  # window length
                out.append(cell)  # kept
            print(f"  screen w{w} {rx}: {X.shape[1]} columns, {time.time() - t0:.0f} s")  # progress line
    return pd.concat(out, ignore_index=True)  # all cells


def lists_and_summary(sc: pd.DataFrame, wc: pd.DataFrame) -> tuple:  # function: the main and screened lists per cell and set, and the counts by reason, set and named group
    lists, summ = [], []  # outputs
    sets = {"no_residual": "no_residual", "main": "main", "known_antenna": "known_antenna"}  # dict: list name -> dictionary column
    for (w, rx, held), cell in sc.groupby(["w", "receiver", "held_out"], sort=False):  # for loop over the cells
        g = groups_of(wc, w)  # named groups
        x = wc[wc.w == w]  # dictionary at this length
        for name, col in sets.items():  # for loop over the sets
            cols = list(x[x[col] == 1].column)  # the set's columns in dictionary order
            lists.append({"w": w, "receiver": rx, "held_out": held, "set": name, "list": "main", "n": len(cols), "columns": ";".join(cols)})  # the unscreened list
            for rule, rcol, lname in (("floor", "reason", "screened"), ("no_floor", "reason_no_floor", "screened_no_floor")):  # for loop over the two rules: the adopted floor and the no-floor sensitivity
                flag = dict(zip(cell.column, cell[rcol].fillna("")))  # column -> reasons under this rule
                scr = [c for c in cols if not flag.get(c)]  # list comprehension: the screened list (no reason)
                lists.append({"w": w, "receiver": rx, "held_out": held, "set": name, "list": lname, "n": len(scr), "columns": ";".join(scr)})  # the screened list
                rs = [flag.get(c, "") for c in cols]  # list comprehension: reasons of the set's columns
                row = {"w": w, "receiver": rx, "held_out": held, "set": name, "rule": rule, "n_columns": len(cols), "n_screened_list": len(scr)}  # dict literal: the summary row
                for reason in ("jamming", "not_assessable", "missingness"):  # for loop over the reasons
                    row[f"n_{reason}"] = sum(reason in r for r in rs)  # generator expression: columns with that reason (a column can carry two)
                for gname, gcols in g.items():  # for loop over the named groups
                    inset = [c for c in cols if c in gcols]  # list comprehension: the group's columns in this set
                    row[f"n_{gname}"] = len(inset)  # how many
                    row[f"n_{gname}_flagged"] = sum(bool(flag.get(c)) for c in inset)  # how many the screen flags
                    row[f"n_screened_without_{gname}"] = len([c for c in scr if c not in gcols])  # screened list without the group
                row["n_screened_without_both"] = len([c for c in scr if c not in g["circularity"] and c not in g["geometry_change"]])  # list comprehension: without both groups
                summ.append(row)  # kept
    return pd.DataFrame(lists), pd.DataFrame(summ)  # the two tables


# ---------------------------------------------------------------- session-fingerprint probe (report only)

def probe(wc: pd.DataFrame) -> pd.DataFrame:  # function: how well a shallow tree tells the sessions apart from each feature set, per receiver at w = 10
    w = 10  # the window length named in the plan
    x = wc[wc.w == w]  # dictionary
    g = groups_of(wc, w)  # named groups
    geo = list(x[(x.part == "geometry") & x.origin.isin(["orbit-free", "self"])].column)  # (i) the geometry block of the main variant: every statistic
    main = list(x[x.main == 1].column)  # (ii) the main set
    sets = {"geometry block": geo, "main set": main, "orbit-free (no-residual) set": list(x[x.no_residual == 1].column),  # (iii) and the variants
            "main without circularity candidates": [c for c in main if c not in g["circularity"]],  # main minus circularity candidates
            "main without geometry change statistics": [c for c in main if c not in g["geometry_change"]],  # main minus geometry change statistics
            "main without both": [c for c in main if c not in g["circularity"] and c not in g["geometry_change"]],  # main minus both
            "geometry change statistics only": sorted(g["geometry_change"] & set(main))}  # just the change statistics
    rows = []  # results
    for rx in RXS:  # for loop over the receivers
        idx, X = load_windows(w, rx)  # its windows (none hold an excluded second)
        y = idx.test_code.to_numpy()  # the class: the session
        first = idx.groupby("test_code").sec.transform("min")  # each session's first decision second
        blk = (idx.test_code + "_" + ((idx.sec - first) // BLOCK_S).astype(str)).to_numpy()  # contiguous 300 s block id within each session
        maj = pd.Series(y).value_counts(normalize=True).iloc[0]  # majority-class rate
        for name, cols in sets.items():  # for loop over the sets
            A = X[cols].to_numpy(dtype=float)  # the set's columns
            pred = np.empty(len(y), dtype=object)  # out-of-fold predictions
            for tr, te in GroupKFold(n_splits=5).split(A, y, blk):  # for loop over the grouped folds (deterministic, no shuffling)
                keep = np.ones(len(tr), bool)  # training rows kept after the purge
                sec, trs = idx.sec.to_numpy(), idx.sec.to_numpy()[tr]  # decision seconds: all, and the training ones
                for s in sorted(set(y[te])):  # for loop over the sessions in the held-out blocks
                    ts = np.sort(sec[te][y[te] == s])  # held-out decision seconds of that session
                    m = y[tr] == s  # training rows of the same session
                    p = np.searchsorted(ts, trs[m])  # where each training second would sit among the held-out ones
                    lo, hi = ts[np.clip(p - 1, 0, len(ts) - 1)], ts[np.clip(p, 0, len(ts) - 1)]  # the held-out seconds just before and just after it
                    near = np.minimum(np.abs(trs[m] - lo), np.abs(trs[m] - hi))  # distance to the nearest held-out second
                    keep[np.flatnonzero(m)[near <= PURGE_S]] = False  # purge: within 10 s of a held-out window of the same session
                clf = DecisionTreeClassifier(max_depth=PROBE_DEPTH, random_state=SEED)  # the shallow tree
                clf.fit(A[tr][keep], y[tr][keep])  # fitted on the purged training blocks
                pred[te] = clf.predict(A[te])  # predicted session of the held-out blocks
            acc = float((pred == y).mean())  # accuracy
            bal = float(np.mean([(pred[y == s] == s).mean() for s in np.unique(y)]))  # balanced accuracy: mean recall over sessions
            rows.append({"receiver": rx, "feature_set": name, "n_columns": len(cols), "n_windows": len(y), "n_sessions": len(np.unique(y)), "accuracy": acc, "balanced_accuracy": bal, "majority_rate": maj, "chance_balanced": 1 / len(np.unique(y))})  # dict literal: the result
    out = pd.DataFrame(rows)  # the table
    for rx in RXS:  # for loop over the receivers
        o = out[out.receiver == rx].set_index("feature_set")  # its rows
        chance, geo = o.loc["geometry block", "chance_balanced"], o.loc["geometry block", "balanced_accuracy"]  # chance and the geometry block's balanced accuracy
        out.loc[out.receiver == rx, "relative_to_geometry"] = (out.loc[out.receiver == rx, "balanced_accuracy"] - chance) / (geo - chance)  # (set - chance) / (geometry - chance): 1 means the set identifies sessions as well as geometry does
    return out  # results


# ---------------------------------------------------------------- fold table and the circularity coupling report

def fold_table(screen_summary: pd.DataFrame) -> pd.DataFrame:  # function: per fold and receiver: labelled seconds, residual coverage by label, unsolved clean seconds, the scoring module policy
    base = pd.read_csv(OUT / "features_base_1hz.csv.gz", usecols=KEYS + [f"{v}_{c}_n_res" for v in ("self", "known") for c in "GREC"] + ["self_solved"])  # residual counts and the solver flag
    lab = pd.read_csv(PROC / "labels" / "labels_1hz.csv", usecols=KEYS + ["label"])  # labels
    pol = pd.read_csv(PROC / "labels" / "scoring_fold_policy.csv")  # the scoring module's fold policy
    b = base.merge(lab, on=KEYS)  # attached
    b["self_any"] = b[[f"self_{c}_n_res" for c in "GREC"]].fillna(0).sum(axis=1) > 0  # at least one self residual
    b["known_any"] = b[[f"known_{c}_n_res" for c in "GREC"]].fillna(0).sum(axis=1) > 0  # at least one known residual
    rows = []  # results
    for held in POSITION_SESSIONS + [TRANSFER]:  # for loop over the folds and the transfer session
        for rx in RXS:  # for loop over the receivers
            x = b[(b.test_code == held) & (b.receiver == rx)]  # that session and receiver
            r = {"held_out": held, "role": "transfer (never trained)" if held == TRANSFER else "leave-one-session-out", "receiver": rx}  # dict literal: identity
            for lb in ("clean", "attack_jam_spoof", "attack_spoof", "uncertain"):  # for loop over labels
                y = x[x.label == lb]  # its seconds
                r[f"{lb}_s"] = len(y)  # seconds
                r[f"{lb}_self_coverage"] = round(float(y.self_any.mean()), 4) if len(y) else np.nan  # share with a self residual
                r[f"{lb}_known_coverage"] = round(float(y.known_any.mean()), 4) if len(y) else np.nan  # share with a known residual
            r["clean_self_unsolved_s"] = int(((x.label == "clean") & (x.self_solved == 0)).sum())  # clean seconds without a self-solved position
            p = pol[(pol.test_code == held) & (pol.receiver == rx)].iloc[0]  # the scoring module's policy row
            r["precision"], r["false_alarms"] = p.precision, p.false_alarms  # what the fold may report
            if held in POSITION_SESSIONS:  # if: a trained fold
                s = screen_summary[(screen_summary.held_out == held) & (screen_summary.receiver == rx) & (screen_summary.w == 10) & (screen_summary.set == "main")].set_index("rule")  # the w = 10 main-set screen, both rules
                r["screen_w10_main_flagged"] = int(s.loc["floor", "n_columns"] - s.loc["floor", "n_screened_list"])  # columns the adopted screen removes
                r["screen_w10_main_flagged_no_floor"] = int(s.loc["no_floor", "n_columns"] - s.loc["no_floor", "n_screened_list"])  # and the no-floor sensitivity
            rows.append(r)  # kept
    return pd.DataFrame(rows)  # the table


def coupling() -> pd.DataFrame:  # function: report: how often the self solver fails when the receivers' own position or time evidence (a label source) is set, against when it is not, on attack and uncertain seconds
    base = pd.read_csv(OUT / "features_base_1hz.csv.gz", usecols=KEYS + ["self_solved", "self_pos_held"])  # solver flags
    lab = pd.read_csv(PROC / "labels" / "labels_1hz.csv", usecols=KEYS + ["label", "own_pos_departure_x5", "own_pos_departure_ubx", "own_time_offset_ubx", "gain_effect_pooled"])  # labels and their own-fix evidence columns (report only)
    b = base.merge(lab, on=KEYS)  # attached
    b = b[b.label.isin(["attack_jam_spoof", "attack_spoof", "uncertain"])]  # seconds where the evidence can be set
    b["own_fix"] = (b.own_pos_departure_x5 + b.own_pos_departure_ubx + b.own_time_offset_ubx) > 0  # any own-position or own-time evidence that second
    rows = []  # results
    for rx in RXS:  # for loop over the receivers
        for ev, x in b[b.receiver == rx].groupby("own_fix"):  # for loop over evidence set / not set
            rows.append({"receiver": rx, "own_fix_evidence": bool(ev), "seconds": len(x), "unsolved_share": round(float((x.self_solved == 0).mean()), 4), "held_share": round(float((x.self_pos_held == 1).mean()), 4), "sessions": ",".join(sorted(x.test_code.unique()))})  # dict literal: the cell
    return pd.DataFrame(rows)  # the table


# ---------------------------------------------------------------- figures

def plot_jamming_screen(sc: pd.DataFrame, wc: pd.DataFrame) -> None:  # function: the figure, S_j against S_s per main-set column at w = 10, median over folds, coloured by reason
    x = wc[(wc.w == 10) & (wc.main == 1)]  # main-set columns at w = 10
    d = sc[(sc.w == 10) & sc.column.isin(x.column)]  # their screen rows
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)  # one panel per receiver
    for a, rx in zip(ax, RXS):  # for loop over the receivers
        y = d[d.receiver == rx].groupby("column").agg(sj=("S_jam", "median"), ss=("S_spoof", "median"), jam=("reason", lambda r: (r.str.contains("jamming")).mean()), miss=("reason", lambda r: (r.str.contains("missingness")).mean()))  # medians over folds and the share of folds with each reason
        cat = np.where(y.miss >= 0.5, "missingness in most folds", np.where(y.jam >= 0.5, "jamming in most folds", "kept in most folds"))  # nested where: a category per column
        for lab_, col in (("kept in most folds", "#0072B2"), ("jamming in most folds", "#D55E00"), ("missingness in most folds", "#999999")):  # for loop over categories
            m = cat == lab_  # its columns
            a.plot(y.ss[m], y.sj[m], ".", ms=3, color=col, label=f"{lab_} ({int(m.sum())})")  # points
        a.plot([0, 1], [0, 1], color="k", lw=0.8)  # the line S_j = S_s
        a.set_title(f"{RXN[rx]}: {len(y)} main-set columns, w = 10", fontsize=10)  # panel title
        a.set_xlabel("S_spoof = |2 AUROC - 1|, attack_spoof vs clean (median over folds)")  # axis label
        a.legend(fontsize=7, loc="upper left")  # legend
    ax[0].set_ylabel("S_jam = |2 AUROC - 1|, attack_jam_spoof vs clean")  # axis label
    fig.tight_layout()  # layout
    save_fig(fig, GRAPH / "jamming_screen.png")  # byte-stable save


def plot_session_probe(pr: pd.DataFrame) -> None:  # function: the figure, the session-fingerprint probe per receiver and feature set
    sets = list(dict.fromkeys(pr.feature_set))  # feature sets in order (dict keeps first occurrences)
    fig, ax = plt.subplots(figsize=(12, 5))  # one panel
    wbar = 0.11  # bar width
    for k, s in enumerate(sets):  # for loop over the sets
        y = [float(pr[(pr.receiver == rx) & (pr.feature_set == s)].balanced_accuracy.iloc[0]) for rx in RXS]  # list comprehension: balanced accuracy per receiver
        ax.bar(np.arange(3) + (k - len(sets) / 2) * wbar, y, wbar, label=s)  # bars
    ch = [float(pr[pr.receiver == rx].chance_balanced.iloc[0]) for rx in RXS]  # chance level per receiver
    ax.plot(np.arange(3), ch, "k_", ms=40, mew=2, label="chance (1 / number of sessions)")  # chance marks
    ax.set_xticks(np.arange(3))  # tick positions
    ax.set_xticklabels([RXN[rx] for rx in RXS])  # tick labels
    ax.set_ylabel("balanced accuracy of the session prediction (grouped CV)")  # axis label
    ax.set_ylim(0, 1)  # range
    ax.legend(fontsize=7, ncol=2)  # legend
    ax.set_title("Session fingerprint at w = 10: depth-4 tree, 5-fold CV over 300 s blocks with a 10 s purge", fontsize=10)  # title
    fig.tight_layout()  # layout
    save_fig(fig, GRAPH / "session_probe.png")  # byte-stable save


def main_registered() -> None:  # function: the registered set: part 4 from start to finish
    t_start = time.time()  # the clock
    A_mod = load("residuals_for_jamming_screen", "4_features/02_residuals.py")  # the residual script, for the protected-file check
    n_p, bad = A_mod.protected_ok()  # protected files at the start
    check("Protected files at start", "all unchanged", f"{n_p} compared, {len(bad)} changed {bad}", not bad)  # the check
    wc = pd.read_csv(OUT / "features_window_columns.csv")  # the window dictionary
    fo = folds()  # the folds
    check("Folds: five leave-one-session-out folds over the position sessions; 2.6.12 in no training list; held-out never in its own training list", "true", f"{[(h, t) for h, t in fo]}", all(h not in t and TRANSFER not in t and len(t) == 4 for h, t in fo))  # the check
    sc = run_screen(wc)  # the screen
    E_ = A_mod.E  # the observables script (loaded through the residual script), for its gzip writer with no time stamp
    E_.write_gz(sc.round(6), OUT / "screen.csv.gz")  # every flag
    lists, summ = lists_and_summary(sc, wc)  # lists and counts
    E_.write_gz(lists, OUT / "feature_lists.csv.gz")  # the lists
    summ.to_csv(CHECK_DIR / "screen_screen_summary.csv", index=False)  # the counts
    for name in ("no_residual", "main", "known_antenna"):  # for loop over the sets
        s = summ[(summ.set == name) & (summ.w == 10) & (summ.rule == "floor")]  # w = 10 rows, adopted rule
        NUM[f"screen_w10_{name}"] = f"columns {int(s.n_columns.iloc[0])}; screened list {int(s.n_screened_list.min())}-{int(s.n_screened_list.max())}; jamming {int(s.n_jamming.min())}-{int(s.n_jamming.max())}; not assessable {int(s.n_not_assessable.min())}-{int(s.n_not_assessable.max())}; missingness {int(s.n_missingness.min())}-{int(s.n_missingness.max())}"  # ranges over receivers and folds
    check("Screen at w = 10, ranges over receivers and folds", "reported", "; ".join(f"{k}: {v}" for k, v in NUM.items() if k.startswith("screen_w10")), None)  # reported
    pr = probe(wc)  # the probe
    pr.round(4).to_csv(CHECK_DIR / "screen_probe.csv", index=False)  # written
    ft = fold_table(summ)  # the fold table
    ft.to_csv(CHECK_DIR / "screen_fold_table.csv", index=False)  # written
    h = ft[(ft.held_out == TRANSFER) & (ft.receiver == "SBF_03")].clean_self_unsolved_s.iloc[0]  # the mosaic-H's unsolved clean seconds in 2.6.12
    check("Fold table: mosaic-H 2.6.12 clean seconds without a self-solved position", "74", int(h), int(h) == 74)  # the check against the residual engine
    cp = coupling()  # the coupling report
    cp.to_csv(CHECK_DIR / "screen_coupling.csv", index=False)  # written
    plot_jamming_screen(sc, wc)  # the figure
    plot_session_probe(pr)  # the figure
    n_p, bad = A_mod.protected_ok()  # protected files at the end
    check("Protected files at end", "all unchanged", f"{n_p} compared, {len(bad)} changed {bad}", not bad)  # the check
    pd.DataFrame(CHECKS).to_csv(CHECK_DIR / "screen_checks.csv", index=False)  # checks
    pd.DataFrame([{"key": k, "value": v} for k, v in NUM.items()]).to_csv(CHECK_DIR / "screen_numbers.csv", index=False)  # list comprehension: numbers
    print(f"done in {time.time() - t_start:.0f} s")  # run time


def main_final() -> None:  # function: (L5): the jamming screen and its lists on the final windows (features_final/), written under new names; nothing else
    t_start = time.time()  # the clock
    SET.update(root=PROC / "features_final", train_label=True)  # dict.update: the final windows, the training label for the screen's groups
    A_mod = load("residuals_for_jamming_screen", "4_features/02_residuals.py")  # the residual script, for the protected-file check and the observables script's writer
    n_p, bad = A_mod.protected_ok()  # protected files at the start
    check("Protected files at start", "all unchanged", f"{n_p} compared, {len(bad)} changed {bad}", not bad)  # the check
    wc = pd.read_csv(OUT / "features_window_columns.csv")  # the window dictionary
    sc = run_screen(wc)  # the screen on the final windows
    E_ = A_mod.E  # the observables script (loaded through the residual script), for its gzip writer with no time stamp
    E_.write_gz(sc.round(6), OUT / "screen_final.csv.gz")  # every flag, under the final set's name
    lists, summ = lists_and_summary(sc, wc)  # lists and counts
    E_.write_gz(lists, OUT / "feature_lists_final.csv.gz")  # the lists, under the final set's name
    summ.to_csv(CHECK_DIR / "screen_screen_summary_final.csv", index=False)  # the counts
    s = summ[(summ.set == "no_residual") & (summ.w == 10) & (summ.rule == "floor")]  # the primary's set at w = 10, adopted rule
    check("Final screen at w = 10, no_residual set: screened list sizes over receivers and folds", "reported", f"{int(s.n_screened_list.min())}-{int(s.n_screened_list.max())} of {int(s.n_columns.iloc[0])}", None)  # reported
    n_p, bad = A_mod.protected_ok()  # protected files at the end
    check("Protected files at end", "all unchanged", f"{n_p} compared, {len(bad)} changed {bad}", not bad)  # the check
    pd.DataFrame(CHECKS).to_csv(CHECK_DIR / "screen_checks_final.csv", index=False)  # checks
    print(f"done in {time.time() - t_start:.0f} s")  # run time


def main(*argv: str) -> None:  # function: runs the label set the command line names (the registered set by default; *argv collects the arguments)
    if list(argv) in ([], ['--labels', 'registered']):  # if: the registered set, the default, unchanged
        main_registered()  # the registered run
    elif list(argv) == ['--labels', 'final']:  # elif: the final set
        main_final()  # the final run, under new names
    else:  # anything else
        raise SystemExit('usage: 4_features/05_jamming_screen.py [--labels registered|final]')  # stop with the usage


if __name__ == "__main__":  # if: true only when the file is run directly
    main(*sys.argv[1:])  # the set named on the command line (main_registered() by default)
    exit_if_failed(FAILED, "4_features/05_jamming_screen.py")  # the helpers module: status 1 when any self-check failed, so a chained run stops
