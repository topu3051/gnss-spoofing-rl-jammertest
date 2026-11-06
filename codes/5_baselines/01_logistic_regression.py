#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
The logistic regression baseline: the pre-registered primary configuration and its
controls, under leave-one-session-out folds only.

Primary configuration (fixed before any fit, not tuned):
  per receiver (mosaic-X5, mosaic-H main, u-blox 01), five folds over 2.6.5, 2.6.2, 2.6.10, 2.6.4, 2.6.11; 2.6.12 is sealed
  (the unseen-attack test of row 13) and no table row of it survives the reader;
  target attack (attack_jam_spoof and attack_spoof) against clean; uncertain and excluded never trained on or scored;
  w = 10 s; the main set's screened list WITH the floor for that fold (feature_lists.csv.gz);
  one missing-indicator per block (orbit-free, self): 1 when any column of that block in the fold's candidate list, other than
  skewness and kurtosis, is missing; undefined skewness and kurtosis are imputed like any gap;
  columns with no training value dropped; median imputation, robust scaling, Pearson pruning at 0.89 and mutual-information
  top 36 (the screen script's fold-index functions, reused unchanged, MI random_state 20260914), every one fitted on the scored training
  windows of the fold only;
  L2 logistic regression, C = 1.0, class_weight balanced, intercept fitted; solver newton-cg, tol 1e-6, max_iter 100,000 (lbfgs,
  as first registered, did not converge on two folds; only the numerical method
  changed, before any result existed). For every fit the gradient of the full penalised objective (sum form, L2 term and
  intercept included) is recomputed with NumPy at the returned parameters and divided by the total training sample weight; a
  scipy LineSearchWarning is accepted only when its largest absolute value is <= 1e-4. A ConvergenceWarning, any other warning,
  a failed gradient check or a non-finite value stops the script; the three spurious "... encountered in matmul" warnings of
  NumPy 2.2.6 on Apple Accelerate are filtered and every decision value is recomputed elementwise against sklearn;
  threshold 0.5 fixed a priori, alarm when p >= 0.5, no persistence.
  Signed-log compression sign(z) log1p(|z|): part of the primary and of every
  control and variant; no parameter, fitted nowhere; order: columns with no training value dropped -> median imputation -> robust
  scaling -> compression -> Pearson pruning -> MI top 36 -> model, on training and held-out windows alike. In the compressed
  pipeline any warning stops the run (the three matmul messages excepted). The registered uncompressed pipeline is reported beside
  it for the primary and C5 only (configurations "primary_uncompressed" and "geometry_uncompressed"), under the rule (a
  LineSearchWarning accepted at a passing gradient); its C4a is not runnable and is reported as such.

Controls (C4a, C4b, C5, C6 here, C1, C2, C3 in the regression tests): C4a labels permuted across all training windows, 100 fixed seeds per cell,
2.6.2 and 2.6.4 folds, held-out AUROC median must lie within 0.5 +/- 0.1 in all six cells; C4b labels permuted inside each
training session, 20 seeds, report only; C5 the 370-column geometry block of the main variant alone, no
indicators; C6 mutual-information seed dispersion over 20260914-20260918, the jamming screen's SEED restored after every run.
Diagnostics, report only and never used to change the primary: the columns that reach a scaled value
beyond 1e3 in training, with their training median and IQR and whether pruning and MI keep them; lbfgs against newton-cg on the
primary folds (objective values, largest probability difference, windows that change side of 0.5).

Reads   Data/processed/features/w10/{receiver}_{index,orbitfree,geometry,self,known}.csv.gz (2.6.12 rows dropped at read),
        features_window_columns.csv, feature_lists.csv.gz, labels/labels_evidence.csv (through the scoring module), Data/processed/checks/expected_hashes_logistic_regression.txt
Writes  Data/processed/logistic_regression_registered/logistic_regression_*.csv(.gz), Data/processed/checks/logistic_regression_checks.csv, logistic_regression_numbers.csv, Graph/logistic_regression_*.png
"""

from __future__ import annotations  # future import: type hints kept as text (a convention across the project's scripts)

import gzip  # standard-library gzip, for compressed tables with no time stamp in their header
import hashlib  # standard-library hashing, for the protected-file check
import importlib.util  # standard-library module loader, for modules whose names start with a digit
import os  # operating-system access: MPLCONFIGDIR is set before matplotlib loads
import sys  # standard-library system access: sys.path and sys.modules
import time  # standard-library clock, for the run-time line
import warnings  # standard-library warnings: the fit's warnings are caught and inspected one by one
from pathlib import Path  # Path objects for building file paths

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
os.environ["MPLCONFIGDIR"] = str(PROJECT / "codes" / ".mplconfig")  # matplotlib's cache inside the project folder

import numpy as np  # NumPy: arrays
import pandas as pd  # pandas: tables
import matplotlib  # plotting library
matplotlib.use("Agg")  # file-only backend, no window
import matplotlib.pyplot as plt  # the pyplot interface
from scipy.special import expit  # the logistic function 1 / (1 + exp(-x)), computed stably
from scipy.optimize._linesearch import LineSearchWarning  # scipy's line-search warning class (not exported by scipy.optimize in 1.16, so read from its module)
from sklearn.exceptions import ConvergenceWarning  # the warning a solver raises when it stops before converging
from sklearn.linear_model import LogisticRegression  # the model: L2-penalised logistic regression
from sklearn.metrics import average_precision_score, roc_auc_score  # threshold-free scores of the held-out probabilities

sys.path.insert(0, str(PROJECT / "codes" / "common"))  # puts codes/ first on the import path so the helpers module is found
from helpers import save_fig  # byte-stable figure saving


def load(name: str, filename: str):  # function: loads codes/<filename> as a module (names start with a digit)
    spec = importlib.util.spec_from_file_location(name, PROJECT / "codes" / filename)  # where to load from
    mod = importlib.util.module_from_spec(spec)  # an empty module object
    sys.modules[name] = mod  # registered, so the module's own imports resolve
    spec.loader.exec_module(mod)  # runs the file's top level (its main() is guarded and not run)
    return mod  # the module


S = load("jamming_screen_for_logistic_regression", "4_features/05_jamming_screen.py")  # 07c: folds, fold-index functions (scaler, Pearson pruning, MI ranking), constants
SC = load("scoring_for_logistic_regression", "common/scoring.py")  # 06c: the scoring rules fixed before any detector existed

SEED = 20260914  # integer constant: the project's fixed seed
if S.SEED != SEED:  # if: the screen script's seed is not the project seed (it must be, since MI uses it)
    raise SystemExit("STOP: the jamming screen's SEED is not 20260914")  # stop
np.random.seed(SEED)  # seeds NumPy's legacy generator (the screen script also seeds it at import)

PROC = PROJECT / "Data" / "processed"  # processed tables
FEAT = PROC / "features"  # the feature scripts' tables and window files
OUT = PROC / "logistic_regression_registered"  # this section's tables
CHECK_DIR = PROJECT / "Data" / "processed" / "checks"  # check tables
GRAPH = PROJECT / "Graph"  # figures

SEALED = S.TRANSFER  # "2.6.12": the sealed session, dropped from every table at read
SESSIONS = S.POSITION_SESSIONS  # the five leave-one-session-out folds, in the screen script's order
RXS = S.RXS  # the three receivers
RXN = S.RXN  # their names for figures and tables
KEYS = S.KEYS  # key columns (test_code, receiver, sec)
ATTACK, SCORED = SC.ATTACK, SC.SCORED  # tuples: the attack labels, and the scored labels (clean plus attack)
W = 10  # the primary window length (s)
C_PRIMARY = 1.0  # inverse regularisation strength of the primary
SOLVER = "newton-cg"  # the optimiser for every fit in the logistic regression
TOL = 1e-6  # the solver's stopping tolerance (same decision)
MAX_ITER = 100_000  # iteration cap; reaching it counts as a failure
GRAD_TOL = 1e-4  # largest |gradient| / total sample weight at which a LineSearchWarning is accepted (same decision)
EXTREME_Z = 1e3  # diagnostic (a): a training value beyond this many interquartile ranges from the median
THR = 0.5  # the decision threshold, fixed a priori and set on no fold
BLOCKS = ("orbitfree", "self")  # the blocks that get a missing-indicator (orbit-free and self)
IND = {"orbitfree": "ind_missing_orbitfree", "self": "ind_missing_self"}  # dict: block -> indicator column name
CIRC_IND = "ind_missing_self"  # the self-block indicator counts as a circularity candidate
UNDEFINED_STATS = ("skew", "kurt")  # statistics undefined in a constant window; their gaps do not set an indicator
N_PERM_C4A, C4A_FOLDS = 100, ["2.6.2", "2.6.4"]  # C4a: 100 permutations per cell on the two folds with both classes
C4A_BAND = (0.4, 0.6)  # C4a pass band for the median held-out AUROC
N_PERM_C4B = 20  # C4b: 20 within-session permutations per cell
MI_SEEDS = [SEED + k for k in range(5)]  # list comprehension: the five MI seeds of C6, the primary's first
TOP_K = 10  # coefficients reported per fold
# the three RuntimeWarning texts NumPy 2.2.6 raises on Apple's Accelerate BLAS for ordinary finite products
MATMUL_MSGS = ("divide by zero encountered in matmul", "overflow encountered in matmul", "invalid value encountered in matmul")
DEC_TOL = 1e-9  # largest relative difference allowed between sklearn's decision values and the elementwise recomputation
READER = pd.read_csv  # the one function every table is read through; the regression tests replaces it to prove 2.6.12 is never needed

CHECKS: list = []  # list: the checks, written to logistic_regression_checks.csv
FIT_DIAG: list = []  # list: one row per logistic regression fit (iterations, gradient, warnings), written to logistic_regression_fit_diagnostics.csv
NUM: dict = {}  # dict: numbers the summary quotes


def check(name: str, expected: str, found, ok: bool | None) -> None:  # function: records one check and prints it
    CHECKS.append({"check": name, "expected": expected, "found": found, "ok": "" if ok is None else bool(ok)})  # append a dict row
    print(f"  [{'ok' if ok else ('FAIL' if ok is False else '--')}] {name}: expected {expected}; found {found}")  # progress line


def sha(path: Path) -> str:  # function: SHA-256 of a file's bytes
    return hashlib.sha256(path.read_bytes()).hexdigest()  # hex digest


def protected_ok() -> tuple:  # function: the 87 protected files (outputs of the ground-truth, preprocessing, label and feature scripts, 75 window files, feature lists) against their hashes
    rows = [l.split() for l in (CHECK_DIR / "expected_hashes_logistic_regression.txt").read_text().splitlines() if l.strip()]  # list comprehension: (hash, path) pairs
    bad = [p for h, p in rows if sha(PROJECT / p) != h]  # list comprehension: files whose hash changed
    return len(rows), bad  # how many were compared, and which differ


def write_gz(df: pd.DataFrame, path: Path) -> None:  # function: gzip CSV with no name or time in its header, so runs again are byte-identical
    with open(path, "wb") as fh, gzip.GzipFile(filename="", mode="wb", fileobj=fh, mtime=0, compresslevel=6) as gz:  # with: file and gzip stream closed afterwards
        gz.write(df.to_csv(index=False, lineterminator="\n").encode("utf-8"))  # the CSV text as bytes


def write_csv(df: pd.DataFrame, path: Path) -> None:  # function: plain CSV with fixed line endings
    df.to_csv(path, index=False, lineterminator="\n")  # written


# ---------------------------------------------------------------- reading, with 2.6.12 sealed

def read_table(path: Path) -> pd.DataFrame:  # function: reads a table through READER and drops every 2.6.12 row before anything else sees it
    df = READER(path, dtype={"test_code": str})  # read; test_code kept as text ("2.6.10" is not a number); ignored when absent
    if "test_code" in df.columns:  # if: the table has sessions
        df = df[df.test_code != SEALED].reset_index(drop=True)  # 2.6.12 rows dropped at once
        if (df.test_code == SEALED).any():  # if: one survived (cannot happen; asserted)
            raise SystemExit(f"STOP: a 2.6.12 row survived the reader in {path.name}")  # stop
    return df  # the table without 2.6.12


def load_cell(w: int, rx: str) -> tuple:  # function: one window length and receiver: the index (keys, label, scored) and every window column, 2.6.12 dropped
    d = FEAT / f"w{w}"  # the folder
    idx = read_table(d / f"{rx}_index.csv.gz")  # keys, label, scoring mask
    parts = [read_table(d / f"{rx}_{pt}.csv.gz") for pt in S.PARTS]  # list comprehension: the four feature parts
    X = parts[0]  # start from the first part
    for p in parts[1:]:  # for loop over the others
        X = X.merge(p, on=KEYS, how="inner", validate="one_to_one")  # joined on the keys (one row per window in each)
    X = idx[KEYS].merge(X, on=KEYS, how="left", validate="one_to_one")  # in the index's order
    if len(X) != len(idx) or X[KEYS].isna().any().any():  # if: a window lost its features in the join
        raise SystemExit(f"STOP: window parts do not match the index for w{w} {rx}")  # stop
    return idx.reset_index(drop=True), X.drop(columns=KEYS).reset_index(drop=True)  # index and features, aligned row by row


def load_meta() -> tuple:  # function: the window dictionary and the feature lists
    wc = read_table(FEAT / "features_window_columns.csv")  # the dictionary (no session column)
    fl = read_table(FEAT / "feature_lists.csv.gz")  # the lists per window length, receiver, fold, set and rule
    return wc, fl  # both


def cand_list(fl: pd.DataFrame, w: int, rx: str, held: str, set_: str = "main", list_: str = "screened") -> list:  # function: one candidate list from feature_lists
    r = fl[(fl.w == w) & (fl.receiver == rx) & (fl.held_out == held) & (fl.set == set_) & (fl["list"] == list_)]  # its row ("list" by key: the attribute is a method)
    if len(r) != 1:  # if: not exactly one row
        raise SystemExit(f"STOP: no unique list for w{w} {rx} {held} {set_} {list_}")  # stop
    return r["columns"].iloc[0].split(";")  # the column names in dictionary order


def geometry_block(wc: pd.DataFrame, w: int) -> list:  # function: the geometry block of the main variant (orbit-free and self origins), 370 columns at w = 10
    x = wc[(wc.w == w) & (wc.part == "geometry") & wc.origin.isin(["orbit-free", "self"])]  # its dictionary rows
    return list(x.column)  # names in dictionary order


# ---------------------------------------------------------------- the fold pipeline

def fold_rows(idx: pd.DataFrame, held: str, train: list) -> tuple:  # function: scored training rows and every held-out row, with the group checks
    tc, lab = idx.test_code.to_numpy(), idx.label.to_numpy()  # sessions and labels
    if held in train or SEALED in train or SEALED in set(tc):  # if: the fold is malformed or 2.6.12 is present
        raise SystemExit(f"STOP: fold {held} is not leave-one-session-out or 2.6.12 is present")  # stop
    tr = np.flatnonzero(np.isin(tc, train) & np.isin(lab, SCORED))  # scored windows (clean and attack) of the training sessions
    te = np.flatnonzero(tc == held)  # every window of the held-out session (uncertain kept, unscored)
    te = te[np.argsort(idx.sec.to_numpy()[te], kind="stable")]  # in time order, as the scoring module needs
    if set(tc[tr]) - set(train) or np.intersect1d(tr, te).size:  # if: a training row is from another session or also held out
        raise SystemExit(f"STOP: group leakage in fold {held}")  # stop
    return tr, te  # row positions


def target(idx: pd.DataFrame) -> np.ndarray:  # function: 1 for attack windows, 0 otherwise (only scored rows are ever used)
    return np.isin(idx.label.to_numpy(), ATTACK).astype(int)  # attack_jam_spoof and attack_spoof together


def with_indicators(X: pd.DataFrame, cand: list, wc: pd.DataFrame, w: int, blocks: tuple) -> pd.DataFrame:  # function: the candidate columns plus one missing-indicator per block
    x = wc[wc.w == w].set_index("column")  # dictionary rows by name
    V = X[cand].copy()  # the candidate columns
    for b in blocks:  # for loop over the blocks that get an indicator
        cols = [c for c in cand if x.at[c, "part"] == b and x.at[c, "stat"] not in UNDEFINED_STATS]  # list comprehension: the block's columns other than skewness and kurtosis
        if cols:  # if: the block has columns in this candidate list
            V[IND[b]] = X[cols].isna().any(axis=1).astype(float).to_numpy()  # 1 when any of them is missing in the window (a per-window value; nothing is fitted)
    return V  # candidates and indicators


def prepare(V: pd.DataFrame, tr: np.ndarray, compress: bool = True, ties: bool = False) -> dict:  # function: the label-free fitted steps on the scored training rows: drop, impute, scale, prune
    A = V.to_numpy(dtype=float)  # values
    names = list(V.columns)  # names
    has = np.isfinite(A[tr]).any(axis=0)  # columns with at least one training value
    dropped = [n for n, h in zip(names, has) if not h]  # list comprehension: columns with no training value
    A, names = A[:, has], [n for n, h in zip(names, has) if h]  # boolean column selection and the matching names
    med = np.nanmedian(A[tr], axis=0)  # training medians (imputation)
    A = np.where(np.isnan(A), med, A)  # every gap filled with its column's training median (broadcast over rows)
    scaler = S.fit_robust_scaler(A, tr)  # 07c: median and interquartile range from the training rows
    iqr_raw = np.percentile(A[tr], 75, axis=0) - np.percentile(A[tr], 25, axis=0)  # the training IQR before the screen script replaces a zero by 1 (diagnostic (a))
    Z = S.apply_scaler(A, scaler)  # 07c: centred and scaled with training parameters
    zmax_pre = np.abs(Z[tr]).max(axis=0)  # largest |scaled value| per column before compression (diagnostic (a))
    if compress:  # if: the compressed pipeline (the primary and every control and variant); False only for the uncompressed sensitivity
        Z = np.sign(Z) * np.log1p(np.abs(Z))  # monotone compression of the tails, no parameter, same on training and held-out rows
    kept = S.pearson_prune(Z, tr)  # 07c: columns kept after dropping the later of every pair with |r| >= 0.89 on the training rows
    n_ties = int((pd.DataFrame(Z[tr]).corr().abs().to_numpy() == S.PEARSON_T).sum()) if ties else -1  # conditional expression: pairs exactly at 0.89 (the screen script prunes at >=, the rule as written says >)
    return {"names": names, "dropped": dropped, "med": med, "scaler": scaler, "iqr_raw": iqr_raw, "zmax_pre": zmax_pre, "compressed": compress, "Z": Z, "kept": kept, "n_ties": n_ties}  # dict literal: everything fitted, and the scaled matrix


def mi_top(prep: dict, y: np.ndarray, tr: np.ndarray, seed: int = SEED) -> list:  # function: the screen script's MI ranking with a chosen seed; the screen script's SEED restored afterwards
    S.SEED = seed  # mi_rank reads the module constant at call time
    try:  # try/finally: the seed is restored even if the ranking fails
        top = S.mi_rank(prep["Z"][:, prep["kept"]], y, tr)  # 07c: the top 36 of the kept columns by mutual information on the training rows
    finally:  # always
        S.SEED = SEED  # restored
    if S.SEED != SEED:  # if: not restored (asserted)
        raise SystemExit("STOP: the jamming screen's SEED was not restored")  # stop
    return [prep["kept"][i] for i in top]  # list comprehension: positions in prep's columns


def balanced_weights(yt: np.ndarray) -> np.ndarray:  # function: the "balanced" class weight of each training row, n / (2 n_class), computed here independently of sklearn
    n, n1 = len(yt), int(yt.sum())  # rows and attack rows
    return np.where(yt == 1, n / (2.0 * n1), n / (2.0 * (n - n1)))  # conditional per row: attack rows get n / (2 n_attack), clean rows n / (2 n_clean)


def objective_and_gradient(Zt: np.ndarray, yt: np.ndarray, coef: np.ndarray, b: float, C: float) -> tuple:  # function: the penalised objective in sum form and its gradient, both divided by the total sample weight
    sw = balanced_weights(yt)  # sample weights
    z = (Zt * coef).sum(axis=1) + b  # linear score by elementwise products (no matrix product)
    loss = np.logaddexp(0.0, z) - yt * z  # logistic loss per row, log(1 + e^z) - y z, computed stably
    obj = (float((sw * loss).sum()) + 0.5 / C * float((coef * coef).sum())) / float(sw.sum())  # weighted loss plus the L2 term (intercept not penalised), per unit weight
    r = sw * (expit(z) - yt)  # weighted residual per row
    g = np.append((Zt * r[:, None]).sum(axis=0) + coef / C, r.sum())  # gradient: coefficients (loss plus L2 term) and intercept
    return obj, float(np.max(np.abs(g)) / float(sw.sum()))  # the objective and the largest |gradient| per unit weight


def fit_lr(Zt: np.ndarray, yt: np.ndarray, C: float = C_PRIMARY, tag: dict | None = None, solver: str = SOLVER, strict: bool = True) -> tuple:  # function: the logistic regression, with every warning inspected and the gradient checked; strict: any warning stops (compressed pipeline), else the rule
    m = LogisticRegression(C=C, class_weight="balanced", solver=solver, tol=TOL if solver == SOLVER else 1e-4, max_iter=MAX_ITER)  # L2 is sklearn's default penalty; lbfgs (diagnostic (b) only) keeps sklearn's default tol
    with warnings.catch_warnings(record=True) as caught:  # with: records every warning raised inside
        warnings.simplefilter("always")  # every occurrence recorded, none hidden
        m.fit(Zt, yt)  # fitted
    spurious, linesearch, conv = 0, 0, 0  # counts of the filtered BLAS messages, line-search warnings, convergence warnings
    for wn in caught:  # for loop over the warnings
        if issubclass(wn.category, ConvergenceWarning):  # if: the solver did not converge
            conv += 1  # counted
        elif issubclass(wn.category, LineSearchWarning):  # elif: scipy's line-search warning
            linesearch += 1  # counted
            if strict and solver == SOLVER:  # if: the compressed pipeline, where any warning stops
                raise SystemExit(f"STOP: LineSearchWarning in a compressed fit {tag}: {wn.message}")  # stops the script
        elif issubclass(wn.category, RuntimeWarning) and str(wn.message) in MATMUL_MSGS:  # elif: one of the three spurious Accelerate messages
            spurious += 1  # counted, not acted on
        elif solver == SOLVER:  # elif: anything else on a counted fit
            raise SystemExit(f"STOP: unexpected warning in the fit: {wn.category.__name__}: {wn.message}")  # stops the script
    n_iter = int(m.n_iter_[0])  # iterations used
    finite = bool(np.isfinite(m.coef_).all() and np.isfinite(m.intercept_).all())  # parameters finite
    obj, grad = objective_and_gradient(Zt, yt, m.coef_[0], float(m.intercept_[0]), C) if finite else (np.nan, np.nan)  # conditional expression: the independent gradient check
    diag = {**(tag or {}), "compressed": strict, "solver": solver, "tol": TOL if solver == SOLVER else 1e-4, "max_iter": MAX_ITER, "n_iter": n_iter, "max_abs_grad_per_weight": grad, "objective_per_weight": obj, "linesearch_warnings": linesearch, "convergence_warnings": conv, "spurious_matmul_warnings": spurious}  # dict unpacking: the fit's diagnostics
    if solver == SOLVER:  # if: a counted fit (not the lbfgs diagnostic)
        FIT_DIAG.append(diag)  # recorded
        if conv or n_iter >= MAX_ITER or not finite or not grad <= GRAD_TOL:  # if: no convergence, cap reached, non-finite, or the gradient check fails (NaN fails too)
            raise SystemExit(f"STOP: fit failed {tag}: convergence warnings {conv}, iterations {n_iter}, finite {finite}, max |gradient| per weight {grad}")  # stops the script
    return m, diag  # model and diagnostics


def predict(m: LogisticRegression, Ze: np.ndarray) -> tuple:  # function: held-out probabilities, checked against an elementwise recomputation
    with warnings.catch_warnings(record=True) as caught:  # with: records the warnings of the matrix products
        warnings.simplefilter("always")  # all recorded
        p = m.predict_proba(Ze)[:, 1]  # probability of attack
        dec = m.decision_function(Ze)  # the linear score sklearn computes
    for wn in caught:  # for loop over the warnings
        if not (issubclass(wn.category, RuntimeWarning) and str(wn.message) in MATMUL_MSGS):  # if: not one of the spurious messages
            raise SystemExit(f"STOP: unexpected warning in prediction: {wn.message}")  # stop
    manual = (Ze * m.coef_[0]).sum(axis=1) + m.intercept_[0]  # the same score by elementwise multiplication and sum (no matrix product)
    rel = float(np.max(np.abs(manual - dec) / np.maximum(1.0, np.abs(dec)))) if len(dec) else 0.0  # conditional expression: largest relative difference
    if not (np.isfinite(p).all() and np.isfinite(dec).all()) or rel > DEC_TOL or np.max(np.abs(expit(manual) - p), initial=0.0) > DEC_TOL:  # if: non-finite, or the two computations disagree
        raise SystemExit(f"STOP: decision values disagree with the elementwise recomputation ({rel:.3g})")  # stop
    return p, rel  # probabilities and the agreement


def fit_fold(prep: dict, y: np.ndarray, tr: np.ndarray, te: np.ndarray, C: float = C_PRIMARY, mi_seed: int = SEED, select: bool = True, tag: dict | None = None) -> dict:  # the warning rule follows prep["compressed"]  # function: selection, fit and held-out probabilities on a prepared fold
    sel = mi_top(prep, y, tr, mi_seed) if select else list(prep["kept"])  # conditional expression: MI top 36, or every kept column
    Z = prep["Z"][:, sel]  # the selected columns
    m, diag = fit_lr(Z[tr], y[tr], C, tag, strict=prep["compressed"])  # fitted on the scored training rows; any warning stops a compressed fit
    p, rel = predict(m, Z[te])  # held-out probabilities
    diag["decision_max_rel_diff"] = rel  # the elementwise agreement joins the fit's diagnostics
    return {"sel": sel, "names": [prep["names"][j] for j in sel], "coef": m.coef_[0].copy(), "intercept": float(m.intercept_[0]), "p": p, "n_iter": diag["n_iter"], "spurious": diag["spurious_matmul_warnings"], "dec_rel": rel, "grad": diag["max_abs_grad_per_weight"], "model": m}  # dict literal: the fitted model and its output


# ---------------------------------------------------------------- evaluation (the scoring module's rules)

def evaluate_fold(idx: pd.DataFrame, te: np.ndarray, p: np.ndarray, held: str) -> tuple:  # function: the metric row and the latency rows of one held-out fold
    sec, lab = idx.sec.to_numpy(np.int64)[te], idx.label.to_numpy(object)[te]  # held-out seconds and labels, in time order
    alarm = p >= THR  # alarm when p >= 0.5
    att, cln = np.isin(lab, ATTACK), lab == "clean"  # attack and clean windows
    scored = att | cln  # scored windows
    n_clean = int(cln.sum())  # clean windows
    pol = SC.fold_policy(held, n_clean)  # 06c: what this fold may report
    # the scoring module's guard: threshold_from_other_folds=True, recorded as "fixed a priori, set on no fold"
    ev = SC.evaluate(sec, lab, alarm, SC.pooled_intervals(held), test=held, threshold_from_other_folds=True)
    pc, fa = ev["per_class"], ev["false_alarms"]  # per-class figures and the false-alarm figure
    both = bool(att[scored].any() and cln[scored].any())  # both classes held out
    row = {"held_out": held, "windows": len(te), "clean": n_clean, "attack_jam_spoof": int((lab == "attack_jam_spoof").sum()), "attack_spoof": int((lab == "attack_spoof").sum()), "uncertain": int((lab == "uncertain").sum()),  # dict literal: counts
           "recall": pc["attack_recall"], "recall_jam_spoof": float(alarm[lab == "attack_jam_spoof"].mean()) if (lab == "attack_jam_spoof").any() else np.nan,  # recall overall and for attack_jam_spoof
           "recall_spoof": float(alarm[lab == "attack_spoof"].mean()) if (lab == "attack_spoof").any() else np.nan,  # recall for attack_spoof
           "precision_reported": bool(pol["precision"]), "precision": pc["attack_precision"] if pol["precision"] else np.nan, "f1": pc["attack_f1"] if pol["precision"] else np.nan,  # precision and F1 only where the scoring module allows
           "tp": pc["tp"], "fn": pc["fn"], "fp": pc["fp"], "tn": pc["tn"], "clean_in_alarm": pc["fp"], "clean_in_alarm_share": pc["fp"] / n_clean if n_clean else np.nan,  # cells, and clean windows in alarm
           # the figure follows the scoring module's fold_policy, which also keeps 2.6.5 and 2.6.10 at "count only"
           # instead of the scoring module's readable label (the 300-clean-second rule alone); a rate is carried only where the policy reads one
           "fa_figure": pol["false_alarms"], "fa_episodes": fa["episodes"], "fa_upper95_count": fa["upper95_count"],  # the scoring module's false-alarm figure under the fold policy
           "fa_rate_per_h": fa["rate_per_h"] if pol["false_alarms"] == "rate" else None, "fa_upper95_per_h": fa["upper95_per_h"] if pol["false_alarms"] == "rate" else None,  # conditional expressions: the rate only in a rate fold
           "auroc": float(roc_auc_score(att[scored], p[scored])) if both else np.nan, "average_precision": float(average_precision_score(att[scored], p[scored])) if both else np.nan,  # threshold-free scores where both classes exist
           "accuracy": float((alarm[scored] == att[scored]).mean()) if scored.any() else np.nan, "alarms_uncertain": pc["alarms_uncertain"], "p_equal_threshold": int((p == THR).sum()),  # accuracy (comparability only), unscored alarms, ties at 0.5
           "threshold_note": "0.5 fixed a priori, set on no fold"}  # how the threshold was set
    lat = []  # latency rows
    for k, L in enumerate(ev["latency"]):  # for loop over the pooled intervals in time order
        a, b = L["onset"], L["end"]  # onset and end (unix seconds)
        inside = (sec >= a) & (sec <= b)  # windows inside the interval
        first_alarm = sec[inside & alarm]  # alarm windows inside it
        lat.append({"held_out": held, "interval": k + 1, "onset": a, "end": b, "windows_in_interval": int(inside.sum()),  # dict literal: the interval
                    "first_window_offset_s": int(sec[inside].min() - a) if inside.any() else np.nan,  # how late the first window inside the interval comes (8-9 s at the 2.6.11 capture start)
                    "detected": bool(first_alarm.size), "latency_s": int(first_alarm.min() - a) if first_alarm.size else np.nan,  # the latency: first alarm window inside the interval minus the onset
                    "lead_06c_s": L["latency_s"] if L["detected"] else np.nan, "early_alarm_runs": len(L["early_alarms"]),  # the scoring module's signed lead and the early alarm runs
                    "pre_onset_uncertain_windows": L["pre_onset_uncertain_s"], "pre_onset_alarm_share": L["pre_onset_alarm_share"]})  # the pre-onset alarm share beside every latency
    return row, lat  # the metric row and the latency rows


def pooled(metrics: pd.DataFrame, latency: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:  # function: per receiver, pooled figures with the number of folds behind each
    rows = []  # results
    for rx in RXS:  # for loop over the receivers
        m, l, p = metrics[metrics.receiver == rx], latency[latency.receiver == rx], preds[preds.receiver == rx]  # its rows
        att = p.label.isin(ATTACK)  # attack windows over all folds
        pf = m[m.precision_reported]  # folds whose the scoring module policy reports precision (2.6.2 and 2.6.4)
        tp, fp = int(pf.tp.sum()), int(pf.fp.sum())  # pooled cells over those folds
        prec = tp / (tp + fp) if tp + fp else np.nan  # conditional expression: pooled precision
        rec2 = tp / (tp + int(pf.fn.sum())) if len(pf) else np.nan  # conditional expression: pooled recall over the same folds
        cnt = m[m.fa_figure == "count only"]  # folds that report a count
        rate = m[m.fa_figure == "rate"]  # folds that report a rate
        au = m[m.auroc.notna()]  # folds with both classes
        det = l[l.detected]  # detected intervals
        # dict literal over several lines: each pooled figure followed by the number of folds or intervals behind it
        rows.append({"receiver": rx,
                     "recall_pooled": float(p[att].alarm.mean()), "recall_median_over_folds": float(m.recall.median()), "recall_min": float(m.recall.min()), "recall_max": float(m.recall.max()), "recall_folds": int(m.recall.notna().sum()),  # recall pooled over every held-out attack window, its median and range over folds, and the folds behind it
                     "recall_jam_spoof_pooled": float(p[p.label == "attack_jam_spoof"].alarm.mean()), "recall_jam_spoof_folds": int(m.recall_jam_spoof.notna().sum()),  # recall for attack_jam_spoof pooled, with the folds that hold that class
                     "recall_spoof_pooled": float(p[p.label == "attack_spoof"].alarm.mean()), "recall_spoof_folds": int(m.recall_spoof.notna().sum()),  # recall for attack_spoof pooled, with its folds
                     # F1 = 2TP / (2TP + FP + FN) over the same folds: 0 when they hold attack windows and no true positive, NaN only when they hold no attack window
                     "precision_pooled": prec, "f1_pooled": np.nan if not np.isfinite(rec2) else (0.0 if tp == 0 else 2 * prec * rec2 / (prec + rec2)), "precision_folds": len(pf),  # precision and F1 pooled over the folds the scoring module allows (2.6.2, 2.6.4), with their number
                     "clean_in_alarm": int(m.clean_in_alarm.sum()), "clean_windows": int(m.clean.sum()), "clean_folds": int((m.clean > 0).sum()),  # clean windows in alarm over all folds, and the folds that have clean windows
                     "fa_episodes_count_folds": int(cnt.fa_episodes.sum()) if len(cnt) else np.nan, "fa_clean_windows_count_folds": int(cnt.clean.sum()), "fa_count_folds": len(cnt),  # false-alarm episodes summed over the count-only folds, with their clean windows and number
                     # episodes summed over every rate fold divided by their clean hours (one window per second), so a second rate
                     # fold no longer makes the pooled rate silently missing; with one rate fold (2.6.2 on the Septentrio receivers) it equals that fold's rate
                     "fa_rate_per_h": float(rate.fa_episodes.sum() / (rate.clean.sum() / 3600.0)) if len(rate) else np.nan, "fa_rate_folds": len(rate),
                     "auroc_median": float(au.auroc.median()) if len(au) else np.nan, "ap_median": float(au.average_precision.median()) if len(au) else np.nan, "auroc_folds": len(au),  # AUROC and average precision: medians over the folds with both classes, with their number
                     "intervals_detected": int(len(det)), "intervals": int(len(l)), "latency_median_s": float(det.latency_s.median()) if len(det) else np.nan,  # intervals with an alarm inside, of all intervals, and the median latency of the detected ones
                     "accuracy_pooled": float((p[p.label.isin(SCORED)].alarm == p[p.label.isin(SCORED)].label.isin(ATTACK)).mean())})  # accuracy over scored windows, for comparability with the literature only
    return pd.DataFrame(rows)  # the table


# ---------------------------------------------------------------- diagnostics (report only; never used to change the primary)

def extreme_columns(prep: dict, res: dict, tr: np.ndarray, rx: str, held: str) -> pd.DataFrame:  # function: diagnostic (a): columns whose scaled training values pass 1e3, with their median and IQR, and whether pruning and MI keep them
    mx = prep["zmax_pre"]  # largest |scaled value| per column on the scored training rows, before any compression
    kept, sel = set(prep["kept"]), set(res["sel"])  # sets: survived pruning, selected
    rows = [{"receiver": rx, "held_out": held, "column": prep["names"][j], "max_abs_scaled_train": float(mx[j]), "train_median": float(prep["scaler"][0][j]), "train_iqr": float(prep["iqr_raw"][j]), "zero_iqr_replaced_by_1": bool(prep["iqr_raw"][j] == 0), "kept_after_pearson": j in kept, "selected_by_mi": j in sel} for j in np.flatnonzero(mx > EXTREME_Z)]  # list comprehension: one row per extreme column
    return pd.DataFrame(rows, columns=["receiver", "held_out", "column", "max_abs_scaled_train", "train_median", "train_iqr", "zero_iqr_replaced_by_1", "kept_after_pearson", "selected_by_mi"])  # the table (columns fixed even when empty)


def lbfgs_vs_newton(prep: dict, res: dict, y: np.ndarray, tr: np.ndarray, te: np.ndarray, rx: str, held: str, config: str = "primary") -> dict:  # function: diagnostic (b): lbfgs (sklearn's default tol) on the primary's selected columns, against the newton-cg fit
    Z = prep["Z"][:, res["sel"]]  # the selected columns
    m, d = fit_lr(Z[tr], y[tr], C_PRIMARY, {"receiver": rx, "held_out": held}, solver="lbfgs")  # the lbfgs fit (not counted, never stops the script)
    conv = d["convergence_warnings"] == 0  # converged by sklearn's own report
    obj_n, _ = objective_and_gradient(Z[tr], y[tr], res["coef"], res["intercept"], C_PRIMARY)  # newton-cg's objective per unit weight
    with warnings.catch_warnings():  # with: the spurious matmul messages of this report-only prediction are silenced
        warnings.simplefilter("ignore", RuntimeWarning)  # RuntimeWarnings ignored inside
        pl = m.predict_proba(Z[te])[:, 1]  # lbfgs held-out probabilities
    # dict literal: the comparison, filled only where lbfgs converged
    return {"config": config, "receiver": rx, "held_out": held, "lbfgs_converged": conv, "lbfgs_iterations": d["n_iter"], "lbfgs_max_abs_grad_per_weight": d["max_abs_grad_per_weight"], "newton_cg_iterations": res["n_iter"], "newton_cg_max_abs_grad_per_weight": res["grad"],
            "objective_lbfgs": d["objective_per_weight"], "objective_newton_cg": obj_n,  # objectives per unit weight
            "max_abs_p_difference": float(np.max(np.abs(pl - res["p"]))) if conv else np.nan, "windows_flipping_at_0_5": int(((pl >= THR) != (res["p"] >= THR)).sum()) if conv else np.nan}  # held-out comparison where lbfgs converged


# ---------------------------------------------------------------- the primary and C5 (the regression tests runs again these through its reader guards)

def run_config(wc: pd.DataFrame, fl: pd.DataFrame, config: str, mi_seed: int = SEED, cells: dict | None = None, keep_prep: bool = False) -> dict:  # function: one configuration over every receiver and fold
    preds, metrics, lat, coefs, selrep, preps, dx_ext, dx_lb = [], [], [], [], [], {}, [], []  # outputs
    for rx in RXS:  # for loop over the receivers
        idx, X = cells[rx] if cells is not None else load_cell(W, rx)  # conditional expression: the windows, loaded or passed in
        y = target(idx)  # the target
        for held, train in S.folds():  # for loop over the folds
            tr, te = fold_rows(idx, held, train)  # rows
            if config.startswith("geometry"):  # if: C5 (compressed or not), the geometry block alone, no indicators
                V = with_indicators(X, geometry_block(wc, W), wc, W, ())  # the 370 columns
            else:  # else: the primary, the main set's screened list with the floor, and both indicators
                V = with_indicators(X, cand_list(fl, W, rx, held), wc, W, BLOCKS)  # candidates plus indicators
            prep = prepare(V, tr, compress=not config.endswith("_uncompressed"), ties=(config.startswith("primary") and mi_seed == SEED))  # the label-free fitted steps; compressed unless the registered-pipeline sensitivity
            res = fit_fold(prep, y, tr, te, mi_seed=mi_seed, tag={"config": config, "mi_seed": mi_seed, "receiver": rx, "held_out": held, "perm": -1})  # selection, fit, held-out probabilities
            if config.startswith("primary") and mi_seed == SEED:  # if: the primary (compressed or not): the two report-only diagnostics
                dx_ext.append(extreme_columns(prep, res, tr, rx, held).assign(config=config))  # (a) columns beyond 1e3 interquartile ranges in training (before compression)
                dx_lb.append(lbfgs_vs_newton(prep, res, y, tr, te, rx, held, config))  # (b) lbfgs against newton-cg on the same selected columns
            if keep_prep:  # if: the caller reuses the prepared fold (C4a, C4b)
                preps[(rx, held)] = (prep, tr, te, y, idx)  # tuple: kept
            row, lr = evaluate_fold(idx, te, res["p"], held)  # metrics and latency
            trl = idx.label.to_numpy()[tr]  # training labels
            cs = sorted(set(idx.test_code.to_numpy()[tr][trl == "clean"]))  # sessions that supply the clean training windows
            row.update({"receiver": rx, "config": config, "mi_seed": mi_seed, "train_clean": int((trl == "clean").sum()), "train_clean_sessions": "+".join(cs), "train_clean_independent": round(float((trl == "clean").sum()) / W, 1),  # dict update: training material (clean windows / 10)
                        "train_attack_jam_spoof": int((trl == "attack_jam_spoof").sum()), "train_attack_spoof": int((trl == "attack_spoof").sum()), "n_iter": res["n_iter"]})  # training attack windows and iterations
            metrics.append(row)  # kept
            for r in lr:  # for loop over the latency rows
                r.update({"receiver": rx, "config": config, "mi_seed": mi_seed})  # identity
                lat.append(r)  # kept
            preds.append(pd.DataFrame({"receiver": rx, "config": config, "mi_seed": mi_seed, "held_out": held, "sec": idx.sec.to_numpy()[te], "label": idx.label.to_numpy()[te], "p": res["p"], "alarm": res["p"] >= THR}))  # dict literal: the held-out windows' probabilities
            order = np.argsort(-np.abs(res["coef"]), kind="stable")  # coefficients by size, largest first
            for rank, j in enumerate(order):  # for loop over the selected columns by |coefficient|
                coefs.append({"receiver": rx, "config": config, "mi_seed": mi_seed, "held_out": held, "rank_abs_coef": rank + 1, "mi_rank": int(j) + 1, "column": res["names"][j], "coef": float(res["coef"][j]), "intercept": res["intercept"]})  # dict literal: one coefficient
            ind_kept = {k: (IND[k] in prep["names"] and prep["names"].index(IND[k]) in prep["kept"]) for k in BLOCKS}  # dict comprehension: indicator survived Pearson pruning
            ind_sel = {k: IND[k] in res["names"] for k in BLOCKS}  # dict comprehension: indicator among the 36 selected
            # dict literal: what was fitted in this fold
            selrep.append({"receiver": rx, "config": config, "mi_seed": mi_seed, "held_out": held, "candidates": V.shape[1], "dropped_no_training_value": len(prep["dropped"]), "kept_after_pearson": len(prep["kept"]), "selected": len(res["sel"]),
                           "ind_orbitfree_present": IND["orbitfree"] in V.columns, "ind_orbitfree_kept": ind_kept["orbitfree"], "ind_orbitfree_selected": ind_sel["orbitfree"],  # orbit-free indicator: built, survived pruning, selected
                           "ind_self_present": IND["self"] in V.columns, "ind_self_kept": ind_kept["self"], "ind_self_selected": ind_sel["self"],  # self indicator (circularity candidate): built, survived pruning, selected
                           "pearson_ties_at_0_89": prep["n_ties"], "n_iter": res["n_iter"], "max_abs_grad_per_weight": res["grad"], "spurious_matmul_warnings": res["spurious"], "decision_max_rel_diff": res["dec_rel"], "p_equal_threshold": int((res["p"] == THR).sum()),  # ties at 0.89, iterations, filtered BLAS messages, decision-value agreement, probabilities at 0.5
                           "train_rows": len(tr), "heldout_rows": len(te), "selected_columns": ";".join(res["names"])})  # rows behind the fit and the prediction, and the selected columns
    out = {"predictions": pd.concat(preds, ignore_index=True), "metrics": pd.DataFrame(metrics), "latency": pd.DataFrame(lat), "coefficients": pd.DataFrame(coefs), "selection": pd.DataFrame(selrep)}  # dict literal: the tables
    out["pooled"] = pooled(out["metrics"], out["latency"], out["predictions"])  # pooled per receiver
    if dx_ext:  # if: the primary's diagnostics were made
        out["diag_extreme"] = pd.concat(dx_ext, ignore_index=True)  # (a)
        out["diag_lbfgs"] = pd.DataFrame(dx_lb)  # (b)
    first = ["receiver", "config", "mi_seed", "held_out"]  # identity columns first
    for k in ("metrics", "latency"):  # for loop over the two tables with appended identity columns
        out[k] = out[k][first + [c for c in out[k].columns if c not in first]]  # list comprehension: reordered
    if keep_prep:  # if: the prepared folds were kept
        out["preps"] = preps  # attached (not written)
    return out  # everything


def tables_text(out: dict) -> dict:  # function: the written form of a configuration's tables (the regression tests compares these texts)
    return {k: v.to_csv(index=False, lineterminator="\n") for k, v in out.items() if k != "preps"}  # dict comprehension: CSV text per table


# ---------------------------------------------------------------- C4a, C4b and C6

def c4a(preps: dict) -> pd.DataFrame:  # function: labels permuted across all scored training windows, 100 seeds per cell, held-out AUROC with the true labels
    rows = []  # results
    for rx in RXS:  # for loop over the receivers
        for held in C4A_FOLDS:  # for loop over 2.6.2 and 2.6.4
            prep, tr, te, y, idx = preps[(rx, held)]  # the primary's prepared fold (its steps use no label)
            lab = idx.label.to_numpy()[te]  # held-out labels
            sc = np.isin(lab, SCORED)  # scored held-out windows
            for k in range(N_PERM_C4A):  # for loop over the permutations
                rng = np.random.default_rng(SEED + k)  # a fixed seed per permutation
                yp = y.copy()  # the labels
                yp[tr] = rng.permutation(y[tr])  # shuffled across all scored training windows (MI and the model see these)
                res = fit_fold(prep, yp, tr, te, tag={"config": "c4a", "mi_seed": SEED, "receiver": rx, "held_out": held, "perm": k})  # selection, fit and held-out probabilities on permuted labels
                rows.append({"receiver": rx, "held_out": held, "perm": k, "seed": SEED + k, "auroc": float(roc_auc_score(np.isin(lab[sc], ATTACK), res["p"][sc])), "n_iter": res["n_iter"]})  # dict literal: AUROC against the true held-out labels
            print(f"  C4a {rx} {held}: median AUROC {np.median([r['auroc'] for r in rows if r['receiver'] == rx and r['held_out'] == held]):.3f}")  # progress line
    return pd.DataFrame(rows)  # the distribution


def c4b(preps: dict) -> pd.DataFrame:  # function: labels permuted inside each training session, 20 seeds per cell; report only
    rows = []  # results
    for rx in RXS:  # for loop over the receivers
        for held in SESSIONS:  # for loop over the folds
            prep, tr, te, y, idx = preps[(rx, held)]  # the prepared fold
            tc = idx.test_code.to_numpy()  # sessions
            lab = idx.label.to_numpy()[te]  # held-out labels
            sc, att, cln = np.isin(lab, SCORED), np.isin(lab, ATTACK), lab == "clean"  # held-out masks
            both = bool(att.any() and cln.any())  # both classes held out
            mixed = sorted(s for s in set(tc[tr]) if len(set(y[tr][tc[tr] == s])) == 2)  # sorted generator: training sessions holding both classes (the only ones a permutation changes)
            for k in range(N_PERM_C4B):  # for loop over the permutations
                rng = np.random.default_rng(SEED + 1000 + k)  # a fixed seed per permutation, distinct from C4a's
                yp = y.copy()  # the labels
                for s in sorted(set(tc[tr])):  # for loop over the training sessions in a fixed order
                    r_ = tr[tc[tr] == s]  # that session's scored training rows
                    yp[r_] = rng.permutation(y[r_])  # shuffled inside the session: its attack share is kept
                res = fit_fold(prep, yp, tr, te, tag={"config": "c4b", "mi_seed": SEED, "receiver": rx, "held_out": held, "perm": k})  # selection, fit and held-out probabilities
                al = res["p"] >= THR  # alarms
                rows.append({"receiver": rx, "held_out": held, "perm": k, "seed": SEED + 1000 + k, "permuted_sessions": "+".join(mixed),  # dict literal: identity
                             "auroc": float(roc_auc_score(att[sc], res["p"][sc])) if both else np.nan, "recall": float(al[att].mean()), "clean_in_alarm_share": float(al[cln].mean()) if cln.any() else np.nan, "n_iter": res["n_iter"]})  # scores
            print(f"  C4b {rx} {held}: permuted sessions {mixed}")  # progress line
    return pd.DataFrame(rows)  # the results


def c6(wc: pd.DataFrame, fl: pd.DataFrame, cells: dict, primary: dict) -> tuple:  # function: the primary run again with four more MI seeds; selection overlap and score dispersion
    runs = [primary]  # the primary is the first seed's run
    for s in MI_SEEDS[1:]:  # for loop over the other seeds
        runs.append(run_config(wc, fl, "primary", mi_seed=s, cells=cells))  # run again with that seed
        if S.SEED != SEED:  # if: the screen script's seed was not restored
            raise SystemExit("STOP: the jamming screen's SEED not restored after a C6 run")  # stop
    met = pd.concat([r["metrics"] for r in runs], ignore_index=True)  # every run's metrics
    sel = pd.concat([r["selection"] for r in runs], ignore_index=True)  # every run's selection
    base = sel[sel.mi_seed == SEED].set_index(["receiver", "held_out"]).selected_columns  # the primary's selected sets
    sel["jaccard_with_primary"] = [len(set(a.split(";")) & set(base[(r, h)].split(";"))) / len(set(a.split(";")) | set(base[(r, h)].split(";"))) for a, r, h in zip(sel.selected_columns, sel.receiver, sel.held_out)]  # list comprehension: overlap with the primary's 36
    disp = met.groupby(["receiver", "held_out"]).agg(recall_min=("recall", "min"), recall_max=("recall", "max"), auroc_min=("auroc", "min"), auroc_max=("auroc", "max"), clean_alarm_min=("clean_in_alarm_share", "min"), clean_alarm_max=("clean_in_alarm_share", "max")).reset_index()  # named aggregation: range over the five seeds
    jac = sel[sel.mi_seed != SEED].groupby(["receiver", "held_out"]).jaccard_with_primary.agg(["min", "median"]).reset_index().rename(columns={"min": "jaccard_min", "median": "jaccard_median"})  # overlap range over the four other seeds
    return met[["receiver", "held_out", "mi_seed", "recall", "auroc", "clean_in_alarm_share"]].merge(sel[["receiver", "held_out", "mi_seed", "jaccard_with_primary", "n_iter"]], on=["receiver", "held_out", "mi_seed"]), disp.merge(jac, on=["receiver", "held_out"])  # per-run rows and the dispersion table


# ---------------------------------------------------------------- figures

COL = {"clean": "#009E73", "attack_jam_spoof": "#D55E00", "attack_spoof": "#CC79A7", "uncertain": "#BBBBBB"}  # dict: label -> colour (colour-blind safe)


def plot_heldout_probability(pr: pd.DataFrame) -> None:  # function: the figure, held-out probability per window, per receiver and fold, coloured by label
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
        ax[i, 0].set_ylabel("p(attack)")  # axis label
    ax[0, 0].legend(fontsize=7, markerscale=6, loc="center left")  # legend
    fig.tight_layout()  # layout
    save_fig(fig, GRAPH / "logistic_regression_heldout_probability.png")  # byte-stable save


def plot_controls(prim: pd.DataFrame, geo: pd.DataFrame, b: pd.DataFrame, a4: pd.DataFrame, unc: pd.DataFrame) -> None:  # function: the figure, the primary beside C5 (geometry), C4b (within-session permutation) and C4a (permutation)
    fig, ax = plt.subplots(2, 3, figsize=(16, 8), sharey="row")  # recall row and AUROC row, one column per receiver
    x = np.arange(len(SESSIONS))  # fold positions
    for k, rx in enumerate(RXS):  # for loop over the receivers
        p_, g_ = prim[prim.receiver == rx].set_index("held_out").loc[SESSIONS], geo[geo.receiver == rx].set_index("held_out").loc[SESSIONS]  # primary and C5 by fold
        bb = b[b.receiver == rx].groupby("held_out")  # C4b by fold
        bm = bb.median(numeric_only=True).loc[SESSIONS]  # C4b medians
        u_ = unc[unc.receiver == rx].set_index("held_out").loc[SESSIONS]  # the uncompressed sensitivity by fold
        ax[0, k].plot(x - 0.15, p_.recall, "o", color="#0072B2", label="primary")  # primary recall
        ax[0, k].plot(x - 0.3, u_.recall, "D", color="#56B4E9", ms=4, label="uncompressed primary (sensitivity)")  # uncompressed recall
        ax[0, k].plot(x, g_.recall, "s", color="#E69F00", label="C5 geometry only")  # C5 recall
        ax[0, k].plot(x + 0.15, bm.recall, "^", color="#999999", label="C4b within-session permutation (median of 20)")  # C4b recall
        ax[0, k].plot(x - 0.15, p_.clean_in_alarm_share, "o", mfc="none", color="#0072B2", label="primary: clean windows in alarm")  # primary false-alarm share
        ax[0, k].plot(x, g_.clean_in_alarm_share, "s", mfc="none", color="#E69F00", label="C5: clean windows in alarm")  # C5 false-alarm share
        ax[1, k].plot(x - 0.15, p_.auroc, "o", color="#0072B2", label="primary")  # primary AUROC
        ax[1, k].plot(x - 0.3, u_.auroc, "D", color="#56B4E9", ms=4, label="uncompressed primary (sensitivity)")  # uncompressed AUROC
        ax[1, k].plot(x, g_.auroc, "s", color="#E69F00", label="C5 geometry only")  # C5 AUROC
        ax[1, k].plot(x + 0.15, bm.auroc, "^", color="#999999", label="C4b median")  # C4b AUROC
        for j, held in enumerate(SESSIONS):  # for loop over the folds
            v = a4[(a4.receiver == rx) & (a4.held_out == held)].auroc  # C4a distribution
            if len(v):  # if: a C4a fold
                ax[1, k].plot([j + 0.3] * 2, np.percentile(v, [5, 95]), color="k", lw=1)  # 5th-95th percentile
                ax[1, k].plot(j + 0.3, np.median(v), "_", color="k", ms=10, label="C4a median, 5-95 % (100 permutations)" if j == 1 else None)  # median
        ax[1, k].axhspan(*C4A_BAND, color="#EEEEEE", zorder=0)  # the C4a pass band
        for r in (0, 1):  # for loop over the rows
            ax[r, k].set_xticks(x)  # fold ticks
            ax[r, k].set_xticklabels(SESSIONS)  # fold names
            ax[r, k].set_xlim(-0.6, len(SESSIONS) - 0.4)  # every fold shown in both rows, also where a row has no value
            ax[r, k].set_ylim(-0.03, 1.03)  # range
        ax[0, k].set_title(f"{RXN[rx]}", fontsize=10)  # column title
        ax[1, k].set_xlabel("held-out session")  # axis label
    ax[0, 0].set_ylabel("recall (filled) / clean windows in alarm (open)")  # axis label
    ax[1, 0].set_ylabel("held-out AUROC (where both classes exist)")  # axis label
    ax[0, 0].legend(fontsize=7, loc="lower left")  # legend
    ax[1, 0].legend(fontsize=7, loc="lower left")  # legend
    fig.tight_layout()  # layout
    save_fig(fig, GRAPH / "logistic_regression_controls.png")  # byte-stable save


def plot_permutation(a4: pd.DataFrame) -> None:  # function: the figure, the C4a AUROC distribution of every cell with the pass band
    fig, ax = plt.subplots(figsize=(10, 4.5))  # one panel
    cells = [(rx, h) for rx in RXS for h in C4A_FOLDS]  # list comprehension: the six cells
    for i, (rx, h) in enumerate(cells):  # for loop over the cells
        v = np.sort(a4[(a4.receiver == rx) & (a4.held_out == h)].auroc.to_numpy())  # its 100 AUROCs, sorted
        ax.plot(i + np.linspace(-0.3, 0.3, len(v)), v, ".", ms=3, color="#0072B2")  # spread along x in rank order (a strip plot)
        ax.plot([i - 0.35, i + 0.35], [np.median(v)] * 2, color="#D55E00", lw=2)  # the median
    ax.axhspan(*C4A_BAND, color="#EEEEEE", zorder=0)  # the pass band
    ax.axhline(0.5, color="k", lw=0.7)  # chance
    ax.set_xticks(range(len(cells)))  # cell ticks
    ax.set_xticklabels([f"{RXN[r]}\nheld out {h}" for r, h in cells], fontsize=8)  # list comprehension: cell names
    ax.set_ylabel("held-out AUROC, labels permuted in training")  # axis label
    ax.set_ylim(0, 1)  # range
    fig.tight_layout()  # layout
    save_fig(fig, GRAPH / "logistic_regression_permutation_c4a.png")  # byte-stable save


# ---------------------------------------------------------------- main

def main() -> None:  # function: the baseline from start to finish
    t0 = time.time()  # the clock
    OUT.mkdir(exist_ok=True)  # the output folder
    n_p, bad = protected_ok()  # protected files at the start
    check("Protected files at start (expected_hashes_logistic_regression.txt)", "87 unchanged", f"{n_p} compared, {len(bad)} changed {bad}", n_p == 87 and not bad)  # the check
    wc, fl = load_meta()  # dictionary and lists
    fo = S.folds()  # the screen script's folds
    check("Folds: five leave-one-session-out folds; 2.6.12 in no training list", "true", str(fo), all(h not in t and SEALED not in t and len(t) == 4 for h, t in fo))  # the check
    cells = {rx: load_cell(W, rx) for rx in RXS}  # dict comprehension: every receiver's windows, 2.6.12 dropped
    check("Sealed: no 2.6.12 row in any loaded window table", "0", sum(int((c[0].test_code == SEALED).sum()) for c in cells.values()), all(not (c[0].test_code == SEALED).any() for c in cells.values()))  # generator expressions: the check
    check("Geometry block of the main variant at w = 10 (C5)", "370 columns", len(geometry_block(wc, W)), len(geometry_block(wc, W)) == 370)  # the check
    prim = run_config(wc, fl, "primary", cells=cells, keep_prep=True)  # the primary
    geo = run_config(wc, fl, "geometry", cells=cells)  # C5
    prim_u = run_config(wc, fl, "primary_uncompressed", cells=cells)  # sensitivity: the registered uncompressed primary
    geo_u = run_config(wc, fl, "geometry_uncompressed", cells=cells)  # sensitivity: the uncompressed C5
    m = prim["metrics"]  # primary metrics
    # independent count: training and held-out windows from a plain crosstab of the index files, against the fold loop's counts
    ind = {rx: pd.crosstab(c[0].test_code, c[0].label) for rx, c in cells.items()}  # dict comprehension: windows per session and label
    ok_cnt = all(int(ind[r.receiver].loc[[t for t in SESSIONS if t != r.held_out], "clean"].sum()) == r.train_clean and int(ind[r.receiver].at[r.held_out, "clean"]) == r.clean and int(ind[r.receiver].loc[r.held_out, list(ATTACK)].sum()) == r.attack_jam_spoof + r.attack_spoof for r in m.itertuples())  # generator expression: every fold's counts agree
    check("Window counts per fold equal an independent crosstab of the index files (and the expected table)", "all equal", f"{len(m)} folds", ok_cnt)  # the check
    sel = pd.concat([prim["selection"], geo["selection"], prim_u["selection"], geo_u["selection"]], ignore_index=True)  # the four selections
    check("Convergence of the primary and C5, compressed and uncompressed (newton-cg, tol 1e-6): iterations below max_iter = 100,000 (a ConvergenceWarning would have stopped the script)", "< 100000", f"max {int(sel.n_iter.max())} (primary {int(prim['selection'].n_iter.min())}-{int(prim['selection'].n_iter.max())})", int(sel.n_iter.max()) < MAX_ITER)  # the check
    check("Decision values: sklearn against the elementwise recomputation, largest relative difference", f"<= {DEC_TOL}", f"{sel.decision_max_rel_diff.max():.3g}", float(sel.decision_max_rel_diff.max()) <= DEC_TOL)  # the check
    check("Spurious Accelerate matmul warnings filtered (only the three named messages)", "reported", int(sel.spurious_matmul_warnings.sum()), None)  # reported
    check("Held-out probabilities exactly equal to 0.5", "reported (0 expected)", int(sel.p_equal_threshold.sum()), None)  # reported
    check("Pearson pairs exactly at |r| = 0.89 (the screen script prunes at >=)", "reported (0 expected)", int(prim["selection"].pearson_ties_at_0_89.sum()), None)  # reported
    print("  C4a ...")  # progress line
    a4 = c4a(prim["preps"])  # C4a
    med = a4.groupby(["receiver", "held_out"]).auroc.median()  # median per cell
    ok4 = bool(((med >= C4A_BAND[0]) & (med <= C4A_BAND[1])).all() and len(med) == 6)  # all six medians inside the band
    check("C4a permutation control: median held-out AUROC within 0.5 +/- 0.1 in all six cells (100 permutations each)", "all six in [0.4, 0.6]", "; ".join(f"{RXN[r]} {h}: {v:.3f}" for (r, h), v in med.items()), ok4)  # the check
    print("  C4b ...")  # progress line
    b4 = c4b(prim["preps"])  # C4b
    print("  C6 ...")  # progress line
    c6_runs, c6_disp = c6(wc, fl, cells, prim)  # C6
    check("C6: the jamming screen's SEED restored after every run", str(SEED), S.SEED, S.SEED == SEED)  # the check
    for suffix, out_ in (("primary", prim), ("c5_geometry", geo), ("primary_uncompressed", prim_u), ("c5_geometry_uncompressed", geo_u)):  # for loop over the four configurations
        write_gz(out_["predictions"], OUT / f"logistic_regression_predictions_{suffix}.csv.gz")  # held-out probabilities, compressed file
        for k in ("metrics", "latency", "pooled", "coefficients", "selection"):  # for loop over the plain tables (the diagnostics are written below)
            write_csv(out_[k], OUT / f"logistic_regression_{k}_{suffix}.csv")  # written
    s2 = read_table(CHECK_DIR / "logistic_regression_stop2_diagnosis.csv")  # the stop-2 diagnosis of the uncompressed permutation fits
    s2 = s2[s2.config == "c4a"].assign(fail=lambda d: d.max_abs_grad_per_weight > GRAD_TOL)  # its C4a fits, failing ones marked
    st = s2.groupby(["receiver", "held_out"]).agg(fits=("fail", "size"), failing_gradient_rule=("fail", "sum"), largest_gradient=("max_abs_grad_per_weight", "max")).reset_index()  # per cell
    st.insert(2, "status", "not runnable")  # the status the decision names
    write_csv(st, OUT / "logistic_regression_c4a_uncompressed_status.csv")  # written
    check("Uncompressed sensitivity: C4a not runnable, stop-2 counts quoted", "416 of 600 failing", f"{int(st.failing_gradient_rule.sum())} of {int(st.fits.sum())}", int(st.failing_gradient_rule.sum()) == 416 and int(st.fits.sum()) == 600)  # the check against 41
    write_csv(a4, OUT / "logistic_regression_c4a_permutation.csv")  # C4a distribution
    write_csv(b4, OUT / "logistic_regression_c4b_within_session.csv")  # C4b
    write_csv(c6_runs, OUT / "logistic_regression_c6_mi_seeds.csv")  # C6 per run
    write_csv(c6_disp, OUT / "logistic_regression_c6_dispersion.csv")  # C6 ranges
    fd = pd.DataFrame(FIT_DIAG)  # one row per counted fit
    write_csv(fd, OUT / "logistic_regression_fit_diagnostics.csv")  # written
    check("Compressed fits: no warning of any kind (any would have stopped the script, matmul messages of the decision excepted)", "0", int(fd[fd.compressed].linesearch_warnings.sum() + fd[fd.compressed].convergence_warnings.sum()), int(fd[fd.compressed].linesearch_warnings.sum() + fd[fd.compressed].convergence_warnings.sum()) == 0)  # the check
    check("Uncompressed sensitivity fits: line-search warnings accepted under the rule", "reported", f"{int(fd[~fd.compressed].linesearch_warnings.sum())} warnings in {int((fd[~fd.compressed].linesearch_warnings > 0).sum())} fits", None)  # reported
    check("Every counted fit (primary, C5, C4a, C4b, C6, uncompressed primary and C5): max |gradient| / total sample weight <= 1e-4 (a failure would have stopped the script)", "<= 1e-4 in all", f"{len(fd)} fits; largest {fd.max_abs_grad_per_weight.max():.3g}; line-search warnings in {int((fd.linesearch_warnings > 0).sum())} fits; iterations {int(fd.n_iter.min())}-{int(fd.n_iter.max())}", bool((fd.max_abs_grad_per_weight <= GRAD_TOL).all() and (fd.convergence_warnings == 0).all()))  # the check
    check("Fits per configuration", "primary 15, C5 15, C4a 600, C4b 300, C6 60, uncompressed primary 15 and C5 15", fd.config.value_counts().to_dict() | {"c6": int((fd.mi_seed != SEED).sum())}, len(fd) == 15 + 15 + 600 + 300 + 60 + 30)  # dict union: counts by configuration, the C6 runs counted by seed
    write_csv(pd.concat([prim["diag_extreme"], prim_u["diag_extreme"]], ignore_index=True), OUT / "logistic_regression_diag_extreme_columns.csv")  # diagnostic (a), both pipelines (values before compression)
    write_csv(pd.concat([prim["diag_lbfgs"], prim_u["diag_lbfgs"]], ignore_index=True), OUT / "logistic_regression_diag_lbfgs_vs_newton_cg.csv")  # diagnostic (b), both pipelines
    co = prim["coefficients"]  # primary coefficients
    top = co[co.rank_abs_coef <= TOP_K]  # top 10 per fold
    rec = top.groupby(["receiver", "column"]).agg(folds_in_top10=("held_out", "nunique"), signs=("coef", lambda c: "".join("+" if v > 0 else "-" for v in c)), mean_coef=("coef", "mean")).reset_index()  # named aggregation: recurrence in the top 10 across folds
    sel36 = co.groupby(["receiver", "column"]).held_out.nunique().rename("folds_selected").reset_index()  # recurrence among the 36 selected
    rec = sel36.merge(rec, on=["receiver", "column"], how="left").fillna({"folds_in_top10": 0, "signs": ""})  # both counts per column
    rec = rec.sort_values(["receiver", "folds_in_top10", "folds_selected", "column"], ascending=[True, False, False, True], kind="stable")  # most recurrent first
    write_csv(rec, OUT / "logistic_regression_coef_recurrence_primary.csv")  # written
    plot_heldout_probability(prim["predictions"])  # the figure
    plot_controls(prim["metrics"], geo["metrics"], b4, a4, prim_u["metrics"])  # the figure
    plot_permutation(a4)  # the figure
    for rx in RXS:  # for loop over the receivers: numbers the summary quotes
        p_ = prim["pooled"][prim["pooled"].receiver == rx].iloc[0]  # pooled row
        NUM[f"{rx}_recall_pooled"] = round(float(p_.recall_pooled), 3)  # pooled recall
        NUM[f"{rx}_auroc_median"] = round(float(p_.auroc_median), 3)  # median AUROC
    n_p, bad = protected_ok()  # protected files at the end
    check("Protected files at end", "87 unchanged", f"{n_p} compared, {len(bad)} changed {bad}", n_p == 87 and not bad)  # the check
    write_csv(pd.DataFrame(CHECKS), CHECK_DIR / "logistic_regression_checks.csv")  # checks
    write_csv(pd.DataFrame([{"key": k, "value": v} for k, v in NUM.items()]), CHECK_DIR / "logistic_regression_numbers.csv")  # list comprehension: numbers
    print(f"done in {time.time() - t0:.0f} s; failed checks: {sum(c['ok'] is False for c in CHECKS)}")  # run time and failures
    if any(c["ok"] is False for c in CHECKS):  # if: a check failed
        sys.exit(1)  # exit status 1, so a failure cannot be missed


if __name__ == "__main__":  # guard: runs main() only when the file is executed, not when imported
    main()  # entry point
