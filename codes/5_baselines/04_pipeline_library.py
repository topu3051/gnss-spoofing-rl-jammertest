"""What the design choices cost: the shared library of the baseline runs (report only; nothing is selected here).

Every fitted step is the project's own code, imported unchanged: the screen script's jamming screen (screen_cell), the regression script's indicator, preparation,
MI ranking, logistic regression and fold evaluation (with_indicators, prepare, mi_top, fit_lr, predict, evaluate_fold, pooled),
the forest script's forest, permutation importance, C4b permutation and classification rule (fit_forest, proba, importance, c4b_permuted,
c4b_median_perm, classify) and the feature script's window statistics (window_rows, win_stats, read_full, per_satellite, free_block).
What is new here is only the plumbing that the design changes need: training labels that differ from scoring labels, pooled
training over the three receivers, a causal trailing-median baseline, a threshold set from inner out-of-fold
predictions, the Stave seconds, nested inner leave-one-session-out scores with the screen refitted inside every inner
fold, and the false-alarm pooling over the folds that the scoring module's fold_policy lets read a rate.
2.6.12 is sealed: it is dropped from every table at read, no 2.6.12 window enters any fit, and no base value of it is built.
"""
from __future__ import annotations  # future import: type hints kept as text

import gzip  # standard library: gzip files with a fixed header time (byte-stable)
import importlib.util  # standard library: loads the project scripts as modules (their names start with a digit)
import io  # standard library: in-memory text, for the CSV round trip of rebuilt windows
import sys  # standard library: module registry
from pathlib import Path  # standard library: paths

import numpy as np  # numpy: arrays
import pandas as pd  # pandas: tables
from sklearn.metrics import roc_auc_score  # threshold-free score of probabilities (as the forest script's C4b)

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
CODES = PROJECT / "codes"  # the project scripts
PROC = PROJECT / "Data" / "processed"  # processed tables
AA = PROC / "design_study"  # the design study outputs
LABELS = AA / "labels"  # the label review outputs (variant labels, quiet seconds, Stave base)
CHOICES = AA / "choices"  # this step's outputs
KW = AA / "k_windows"  # rebuilt base tables (in .gitignore by the decision)


def load(name: str, filename: str):  # function: loads codes/<filename> as a module, unchanged
    spec = importlib.util.spec_from_file_location(name, CODES / filename)  # where to load from
    mod = importlib.util.module_from_spec(spec)  # an empty module object
    sys.modules[name] = mod  # registered so the module's own imports resolve
    spec.loader.exec_module(mod)  # runs its top level (main() is guarded and does not run)
    return mod  # the module


R = load("random_forest_for_pipeline_library", "5_baselines/02_random_forest.py")  # 10, which loads 09 as its M; the regression script loads the screen script and the scoring module
M, S, SC = R.M, R.S, R.SC  # 09, the screen script and the scoring module exactly as 10 uses them
F = load("window_features_for_pipeline_library", "4_features/03_window_features.py")  # 07 (window statistics and the orbit-free block), which loads the residual script, the observables script, the evidence script and the position script

SEED = M.SEED  # 20260914, the project's seed
RXS, SESSIONS, KEYS = M.RXS, M.SESSIONS, M.KEYS  # receivers, the five folds in the screen script's order, key columns
SEALED, W = M.SEALED, M.W  # "2.6.12" and the window length 10
ATTACK, SCORED = M.ATTACK, M.SCORED  # attack labels and scored labels
BLOCKS = R.BLOCKS  # ("orbitfree",): the orbit-free primary's one missing-indicator block (the random forest's orbit-free set)
FAMILIES = R.FAMILIES  # the nine primary families of the random forest
QUIET_RES_M = 30.0  # strict: a known-antenna residual above 30 m disqualifies a second
QUIET_SIDE_S = 30  # strict: and 30 s on each side of such a second
TRAIL_S = 300  # the trailing-median baseline: the trailing median covers the previous 300 s
INNER_THRESHOLD_SHARE = 0.05  # the inner-fold threshold: the largest inner out-of-fold share of clean windows in alarm
N_PERM = R.N_PERM_C4B  # 100 C4b permutations per cell, as registered in the random forest
EARLY_S = 30  # early detection: recall over the first 30 s of each attack interval
S11_INTERVAL = ("2.6.11", 2)  # the label review the decision: the 1.1.27 interval of 2.6.11, the session's second pooled interval
IND_COLS = ["rx_is_SBF_03", "rx_is_UBX_01"]  # pooled training with the receiver indicator: two 0/1 columns, the X5 the reference
FAIL_COLS = ["run_id", "model", "held_out", "inner_held_out", "receivers", "reason"]  # columns of the failed-fit table
RES_COLS = ["known_G_res_maxabs", "known_E_res_maxabs", "known_C_res_maxabs", "known_R_res_db_maxabs"]  # known-antenna residual extremes per second (GLONASS only has the de-biased residual); label cleaning only, never a feature


def write_gz(df: pd.DataFrame, path: Path) -> None:  # function: gzip CSV with no file name or time in its header (byte-stable)
    path.parent.mkdir(parents=True, exist_ok=True)  # the folder
    with open(path, "wb") as fh, gzip.GzipFile(filename="", mode="wb", fileobj=fh, mtime=0) as gz:  # with-block over two context managers: the file and the gzip stream
        gz.write(df.to_csv(index=False, lineterminator="\n").encode("utf-8"))  # the table as CSV text (floats in their shortest round-trip form)


def write_csv(df: pd.DataFrame, path: Path) -> None:  # function: plain CSV with fixed line endings
    path.parent.mkdir(parents=True, exist_ok=True)  # the folder
    df.to_csv(path, index=False, lineterminator="\n")  # written


def read_exact(path: Path, **kw) -> pd.DataFrame:  # function: reads a CSV with exact float parsing (pandas' default parser can miss the last bit)
    return pd.read_csv(path, float_precision="round_trip", dtype={"test_code": str, "held_out": str, "outer_held_out": str}, **kw)  # dtype keeps "2.6.10" as text


def no_sealed(df: pd.DataFrame) -> pd.DataFrame:  # function: drops every 2.6.12 row before anything else sees it
    return df[df.test_code != SEALED].reset_index(drop=True)  # boolean filter


# ----------------------------------------------------------------------------------------------------------------------
# columns, labels and the base table
# ----------------------------------------------------------------------------------------------------------------------
def window_dict() -> pd.DataFrame:  # function: the feature script's committed window dictionary at w = 10
    wc = M.read_table(PROC / "features" / "features_window_columns.csv")  # read through the regression script's reader
    return wc[wc.w == W].reset_index(drop=True)  # the w = 10 rows, in the feature script's order


def d1_columns(wc: pd.DataFrame) -> list:  # function: the orbit-free primary's candidate set (no_residual), in the feature script's column order
    return list(wc[wc.no_residual == 1].column)  # 778 columns at w = 10


def d1_bases(wc: pd.DataFrame) -> list:  # function: the base columns behind the orbit-free set, in order of first appearance
    return list(dict.fromkeys(wc[wc.no_residual == 1].base_column))  # dict.fromkeys keeps first occurrences in order: 103 base columns


def labels(source: str) -> pd.DataFrame:  # function: second labels of one source (2.6.12 dropped)
    if source == "registered":  # the committed labels
        lab = M.read_table(PROC / "labels" / "labels_1hz.csv")[KEYS + ["label"]]  # labels/labels_1hz.csv through the regression script's reader
    else:  # an the label review variant: guard_only_g{0,10,60}, all_three_g{0,10,60} or k7
        lab = pd.read_csv(LABELS / f"labels_{source}.csv.gz", dtype={"test_code": str}, usecols=KEYS + ["label"])  # the label review's label file
    return no_sealed(lab)  # sealed rows dropped


def pooled_intervals() -> dict:  # function: session -> pooled attack intervals (the scoring module's function, labels/labels_evidence.csv)
    return {t: SC.pooled_intervals(t) for t in SESSIONS}  # dict comprehension over the five sessions


def build_free_base(k7: bool) -> pd.DataFrame:  # function: the orbit-free base columns of every pair, unrounded, by the feature script's own functions (2.6.5 from the end of the first drive)
    wc = window_dict()  # dictionary
    bases = d1_bases(wc)  # the base columns
    gt = pd.read_csv(F.PROC / "ground_truth" / "session_ground_truth.csv", usecols=F.GT_COLS).set_index("test_code")  # sessions, the columns the feature script reads
    grid_all = no_sealed(pd.read_csv(F.PROC / "labels" / "labels_1hz.csv", usecols=KEYS, dtype={"test_code": str}))  # the label grid, keys only (as in the window-feature script), 2.6.12 dropped
    lam = F.wavelengths()  # carrier wavelengths (the window-feature script)
    committed = dict(F.D.SITE_START)  # the evidence script's constant as committed
    if k7:  # the Stave variant
        F.D.SITE_START = {"2.6.5": "12:27:56"}  # in-memory override: 2.6.5 starts at the end of the first drive
    parts = []  # per pair
    try:  # try: the constant always comes back
        for test in SESSIONS:  # five sessions (2.6.12 is never built)
            for rx in RXS:  # three receivers
                o, (a, b) = F.read_full(gt, test, rx)  # the feature script's reader: observations in the capture span
                o = F.per_satellite(o, lam)  # the feature script's per-satellite quantities
                g = grid_all[(grid_all.test_code == test) & (grid_all.receiver == rx)]  # the pair's grid seconds
                inside = g.sec[(g.sec >= a) & (g.sec < b)].to_numpy()  # those inside the span
                blk = F.free_block(o, inside)  # the feature script's orbit-free block (both orbit-free and geometry-block quantities)
                parts.append(g[KEYS].merge(blk[["sec"] + bases], on="sec", how="left"))  # the pair on its full grid, as the feature script's build_base
    finally:  # finally: runs even after an error
        F.D.SITE_START = committed  # restored
    return pd.concat(parts, ignore_index=True)  # every pair


def free_base(k7: bool) -> pd.DataFrame:  # function: the cached base (built once and cached), read exactly
    return read_exact(KW / f"free_base_{'k7' if k7 else 'committed'}.csv.gz")  # the gzip CSV in k_windows/


def trailing(base: pd.DataFrame, cols: list) -> pd.DataFrame:  # function: the trailing-median baseline, each base column minus its median over [t-300 s, t-1 s] in the same receiver and session
    out = []  # per pair
    for _, g in base.groupby(["test_code", "receiver"], sort=False):  # pairs, in the table's order
        t = pd.to_datetime(g.sec.to_numpy(), unit="s")  # a time index, so the window is by time, not by rows
        x = g[cols].set_axis(t)  # the base values on that index
        med = x.rolling(f"{TRAIL_S}s", closed="left", min_periods=1).median()  # rolling median by time; closed="left": [t-300 s, t), non-missing values only, expanding until 300 s exist, missing at the first second
        out.append(pd.concat([g[KEYS].reset_index(drop=True), (x - med).reset_index(drop=True)], axis=1))  # base minus its trailing median
    return pd.concat(out, ignore_index=True)  # every pair


def windows_from_base(base: pd.DataFrame, lab: pd.DataFrame, wc: pd.DataFrame) -> dict:  # function: the feature script's windows (rows and statistics) of the orbit-free set, from a base table and labels
    cols = d1_columns(wc)  # orbit-free window columns
    bases = d1_bases(wc)  # their base columns
    rows = F.window_rows(lab, W)  # the feature script's rule: w consecutive grid seconds, none excluded, labelled by the last; "pos" = row within the pair
    out = {rx: ([], []) for rx in RXS}  # dict comprehension: receiver -> (index parts, value parts)
    for (test, rx), g in lab.groupby(["test_code", "receiver"], sort=False):  # pairs, in the label table's order (as in the window-feature script)
        bp = g[KEYS].merge(base, on=KEYS, how="left", validate="one_to_one")  # the base rows aligned to the label grid
        st = F.win_stats(bp[bases].to_numpy(dtype=float), W)  # the feature script's window statistics of every base column
        r = rows[(rows.test_code == test) & (rows.receiver == rx)]  # the kept windows of the pair
        pos = r.pos.to_numpy()  # their rows
        j = {b: k for k, b in enumerate(bases)}  # dict comprehension: base column -> its position
        # dict comprehension: every orbit-free column "base__stat" taken from the statistic array at the kept rows
        vals = {c: st[c.split("__")[1]][pos, j[c.split("__")[0]]] for c in cols}
        out[rx][0].append(r[KEYS + ["label"]].reset_index(drop=True))  # index part
        out[rx][1].append(pd.DataFrame(vals).round(4))  # values with the feature script's 4-decimal rounding (write_windows)
    # dict comprehension: receiver -> (index, values); the values pass through CSV text and pandas' default parser, exactly the path
    # of the feature script's write_windows and the regression script's read_table, so a rebuilt window holds the same double the committed pipeline would have read
    return {rx: (pd.concat(a, ignore_index=True), pd.read_csv(io.StringIO(pd.concat(b, ignore_index=True).to_csv(index=False)))) for rx, (a, b) in out.items()}


def committed_cells(wc: pd.DataFrame) -> dict:  # function: the committed w10 windows of the orbit-free set (the regression script's reader, 2.6.12 dropped)
    cols = d1_columns(wc)  # orbit-free columns
    out = {}  # receiver -> (index, values)
    for rx in RXS:  # receivers
        idx = M.read_table(PROC / "features" / f"w{W}" / f"{rx}_index.csv.gz")  # index
        X = idx[KEYS]  # keys, to align the parts
        for pt in ("orbitfree", "geometry"):  # the two parts that hold orbit-free columns
            X = X.merge(M.read_table(PROC / "features" / f"w{W}" / f"{rx}_{pt}.csv.gz"), on=KEYS, how="left", validate="one_to_one")  # merged one to one
        out[rx] = (idx[KEYS + ["label"]].reset_index(drop=True), X[cols].reset_index(drop=True))  # index and orbit-free values
    return out  # cells


def relabel(cells: dict, lab: pd.DataFrame) -> dict:  # function: the same windows under other second labels (window set unchanged: guard-only variants)
    wl = F.window_rows(lab, W)[KEYS + ["label"]]  # window labels by the feature script's rule
    out = {}  # receiver -> (index, values)
    for rx, (idx, X) in cells.items():  # receivers
        new = idx[KEYS].merge(wl, on=KEYS, how="left", validate="one_to_one")  # the variant's labels on the same windows
        if new.label.isna().any() or len(new) != len(idx):  # the window set must not change
            raise SystemExit(f"STOP: relabel changes the window set for {rx}")  # stop
        out[rx] = (new, X)  # same values
    return out  # cells


# ----------------------------------------------------------------------------------------------------------------------
# the strict quiet-second rule: quiet seconds as training negatives
# ----------------------------------------------------------------------------------------------------------------------
def quiet_sets() -> tuple:  # function: (the strict quiet-second rule-as-ruled, strict) quiet seconds, and the reconciliation table of strict
    q = pd.read_csv(LABELS / "label_rule2_quiet_seconds.csv", dtype={"test_code": str})  # label rule 2's quiet seconds
    q = no_sealed(q[q.rule == "b_feature_bands"][KEYS]).drop_duplicates()  # rule (b), the defining rule
    res = no_sealed(pd.read_csv(PROC / "features" / "features_base_1hz.csv.gz", usecols=KEYS + RES_COLS, dtype={"test_code": str}))  # known-antenna residual extremes (labelling evidence only)
    res["res_max"] = res[RES_COLS].max(axis=1)  # the largest over the four constellations (missing only where all are missing)
    big = res[res.res_max > QUIET_RES_M]  # seconds with a residual above 30 m, any label
    near = set()  # keys within 30 s of such a second
    for (t, rx), g in big.groupby(["test_code", "receiver"]):  # pairs
        s = g.sec.to_numpy()  # their seconds
        near |= {(t, rx, int(v)) for c in s for v in range(c - QUIET_SIDE_S, c + QUIET_SIDE_S + 1)}  # set comprehension: every second within +-30 s, the second itself included
    qk = list(zip(q.test_code, q.receiver, q.sec.astype(int)))  # quiet keys
    r1 = np.array([k in near for k in qk])  # reason 1: a residual above 30 m at the second or within 30 s of it
    r2 = (q.test_code == "2.6.10").to_numpy()  # reason 2: every 2.6.10 quiet second
    piv = pooled_intervals()  # pooled attack intervals
    a_end = max(b for _, b in piv["2.6.2"])  # end of 2.6.2's attack
    clear = {rx: int(big[(big.test_code == "2.6.2") & (big.receiver == rx) & (big.sec > a_end)].sec.max()) for rx in RXS}  # dict comprehension: last post-attack 2.6.2 second with a residual above 30 m per receiver (the residuals clear after it)
    r3 = np.array([t == "2.6.2" and a_end < s <= clear[rx] for t, rx, s in qk])  # reason 3: 2.6.2's post-attack counterfeit stretch until the residuals clear on that receiver
    drop = r1 | r2 | r3  # any reason
    strict = q[~drop].reset_index(drop=True)  # strict
    rec = []  # reconciliation rows
    for (t, rx), g in q.assign(r1=r1, r2=r2, r3=r3, kept=~drop).groupby(["test_code", "receiver"]):  # per pair
        # list.append of a dict over two lines: counts by reason (a second can carry several reasons)
        rec.append({"test_code": t, "receiver": rx, "quiet_rule_b": len(g), "res_over_30m_or_within_30s": int(g.r1.sum()), "session_2610": int(g.r2.sum()),
                    # dict entry: seconds removed per reason and in all, seconds kept, and the 2.6.2 clear time
                    "post_attack_2_6_2_until_clear": int(g.r3.sum()), "removed_any": int((~g.kept).sum()), "k2_strict": int(g.kept.sum()), "clear_utc": pd.Timestamp(clear[rx], unit="s").strftime("%H:%M:%S") if t == "2.6.2" else ""})
    return q, strict, pd.DataFrame(rec)  # the two sets and the table


def with_quiet(cells: dict, quiet: pd.DataFrame) -> dict:  # function: windows whose last second is quiet get the training label "clean"; their scoring label stays "uncertain"
    qk = set(zip(quiet.test_code, quiet.receiver, quiet.sec.astype(int)))  # quiet keys
    out = {}  # receiver -> (index, values)
    for rx, (idx, X) in cells.items():  # receivers
        idx = idx.copy()  # a copy, so the input is untouched
        k = np.array([key in qk for key in zip(idx.test_code, idx.receiver, idx.sec.astype(int))])  # windows ending on a quiet second
        if (idx.label[k] != "uncertain").any():  # every one must be scored "uncertain"
            raise SystemExit("STOP: a quiet window is not uncertain")  # stop
        idx["tlabel"] = np.where(k, "clean", idx.tlabel if "tlabel" in idx else idx.label)  # training label
        out[rx] = (idx, X)  # cells
    return out  # cells


def s11(cells: dict) -> dict:  # function: the label review the decision: the attack windows of 2.6.11's second interval set to uncertain (out of training and scoring)
    a, b = pooled_intervals()[S11_INTERVAL[0]][S11_INTERVAL[1] - 1]  # its onset and end
    out = {}  # receiver -> (index, values)
    for rx, (idx, X) in cells.items():  # receivers
        idx = idx.copy()  # a copy
        k = (idx.test_code == S11_INTERVAL[0]).to_numpy() & idx.sec.between(a, b).to_numpy() & idx.label.isin(ATTACK).to_numpy()  # its attack windows
        idx.loc[k, "label"] = "uncertain"  # scoring label
        if "tlabel" in idx:  # a training label exists
            idx.loc[k, "tlabel"] = "uncertain"  # training label too
        out[rx] = (idx, X)  # cells
    return out  # cells


def finish(cells: dict) -> dict:  # function: adds the training label where none was set (equal to the scoring label)
    return {rx: (idx.assign(tlabel=idx.tlabel if "tlabel" in idx else idx.label), X) for rx, (idx, X) in cells.items()}  # dict comprehension


# ----------------------------------------------------------------------------------------------------------------------
# one fit: screen, preparation, MI, both models
# ----------------------------------------------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------------------------------------------
# evaluation and pooling
# ----------------------------------------------------------------------------------------------------------------------
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


def fa_pool(metrics: pd.DataFrame) -> dict:  # function: false alarms pooled over the folds whose the scoring module fold_policy reads a rate, and over the count-only folds
    pol = [SC.fold_policy(h, int(c))["false_alarms"] for h, c in zip(metrics.held_out, metrics.clean)]  # the scoring module's verdict per fold
    m = metrics.assign(policy=pol)  # beside the rows
    rate, cnt = m[m.policy == "rate"], m[m.policy == "count only"]  # the two kinds
    eps, cs = float(rate.fa_episodes.sum()), float(rate.clean.sum())  # pooled episodes and clean windows of the rate folds
    ce, cc = float(cnt.fa_episodes.sum()), float(cnt.clean.sum())  # the same over count-only folds
    # dict literal over three lines: the pooled figures
    return {"fa_rate_folds_policy": "+".join(rate.held_out), "fa_rate_episodes": eps if len(rate) else np.nan, "fa_rate_clean_windows": cs if len(rate) else np.nan,
            # dict entry: pooled false-alarm rate per hour and its 95 per cent Poisson upper bound
            "fa_rate_per_h_pooled": eps / (cs / 3600.0) if len(rate) and cs else np.nan, "fa_upper95_per_h_pooled": SC.poisson_upper(int(eps)) / (cs / 3600.0) if len(rate) and cs else np.nan,
            # dict entry: folds and totals behind the false-alarm count, and folds whose policy differs
            "fa_count_folds_policy": "+".join(cnt.held_out), "fa_count_episodes": ce if len(cnt) else np.nan, "fa_count_clean_windows": cc if len(cnt) else np.nan, "fa_policy_differs_from_readable": "+".join(m.held_out[(m.policy != m.fa_figure)])}


def early(preds: pd.DataFrame, lat: pd.DataFrame, drop_s11: bool) -> dict:  # function: early detection per receiver: latency, the scoring module lead, pre-onset alarm share, recall in the first 30 s
    piv = pooled_intervals()  # intervals
    n30, a30 = 0, 0  # windows and alarms in the first 30 s
    for held, d in preds.groupby("held_out"):  # folds
        for k, (a, b) in enumerate(piv[held]):  # its intervals
            if drop_s11 and (held, k + 1) == S11_INTERVAL:  # the dropped interval
                continue  # skipped
            x = d[(d.sec >= a) & (d.sec <= min(a + EARLY_S - 1, b)) & d.label.isin(ATTACK)]  # attack windows in its first 30 s
            n30, a30 = n30 + len(x), a30 + int(x.alarm.sum())  # counted
    pre_n = lat.pre_onset_uncertain_windows.fillna(0)  # uncertain windows before each onset
    pre_a = (lat.pre_onset_alarm_share.fillna(0) * pre_n)  # alarms among them
    det = lat[lat.detected.astype(bool)]  # detected intervals
    # dict literal over three lines: the early-detection figures
    return {"intervals": len(lat), "intervals_detected": len(det), "latency_median_s": float(det.latency_s.median()) if len(det) else np.nan,
            # dict entry: median lead over the scoring script's onset, and uncertain windows before onset
            "lead_06c_median_s": float(det.lead_06c_s.median()) if len(det) else np.nan, "pre_onset_uncertain_windows": int(pre_n.sum()),
            # dict entry: share of pre-onset windows in alarm, and recall over the first 30 attack windows
            "pre_onset_alarm_share_pooled": float(pre_a.sum() / pre_n.sum()) if pre_n.sum() else np.nan, "first30_attack_windows": n30, "first30_recall": a30 / n30 if n30 else np.nan}


def summarise(metrics: pd.DataFrame, lat: pd.DataFrame, preds: pd.DataFrame, drop_s11: bool) -> pd.DataFrame:  # function: per receiver: the regression script's pooled figures, the policy false-alarm pooling, early detection and the pooled-prediction AUROC
    rows = []  # per receiver
    base = M.pooled(metrics, lat, preds)  # the regression script's pooled table (the committed definition)
    for rx in RXS:  # receivers
        m, l, p = metrics[metrics.receiver == rx], lat[lat.receiver == rx], preds[preds.receiver == rx]  # its rows
        sc = p[p.label.isin(SCORED)]  # scored windows
        att = sc.label.isin(ATTACK).to_numpy()  # attack
        auc = float(roc_auc_score(att, sc.p)) if att.any() and (~att).any() and sc.p.notna().all() else np.nan  # AUROC over the pooled predictions (report; the committed figure is the median over folds)
        cl = p[p.label == "clean"]  # clean windows
        # dict literal: the regression script's pooled row plus the added figures
        rows.append({**base[base.receiver == rx].iloc[0].to_dict(), "clean_in_alarm_share_pooled": float(cl.alarm.mean()) if len(cl) else np.nan, "auroc_pooled_predictions": auc, **fa_pool(m), **early(p, l, drop_s11)})
    return pd.DataFrame(rows)  # table


# ----------------------------------------------------------------------------------------------------------------------
# outer and nested runs
# ----------------------------------------------------------------------------------------------------------------------
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


def nested_summary(metrics: pd.DataFrame, lat: pd.DataFrame, preds: pd.DataFrame, drop_s11: bool) -> pd.DataFrame:  # function: per model, receiver and outer fold, the inner figures pooled over the four inner folds
    out = []  # rows
    for (mdl, held), m in metrics.groupby(["model", "outer_held_out"], sort=False):  # model and outer fold
        l = lat[(lat.model == mdl) & (lat.outer_held_out == held)]  # its latency rows
        p = preds[(preds.model == mdl) & (preds.outer_held_out == held)]  # its out-of-fold predictions
        s = summarise(m, l, p, drop_s11)  # the same pooled figures, the inner folds standing in for folds
        s["model"], s["outer_held_out"] = mdl, held  # identifiers
        s["inner_flags"] = s.receiver.map(lambda r: "; ".join(f"{h}: {x}" for h, x in zip(m[m.receiver == r].held_out, m[m.receiver == r].inner_flags) if x))  # lambda over receivers: named inner folds that cannot carry a metric
        s["inner_folds_scored"] = s.receiver.map(lambda r: int((m.receiver == r).sum()))  # inner folds behind the figures (4 unless a fit failed)
        out.append(s)  # kept
    return pd.concat(out, ignore_index=True)  # table


# ----------------------------------------------------------------------------------------------------------------------
# the inner-fold threshold: threshold from the inner out-of-fold predictions
# ----------------------------------------------------------------------------------------------------------------------
def k5_threshold(p_clean: np.ndarray) -> float:  # function: the lowest threshold t with share(p >= t) <= 0.05 over the clean out-of-fold windows
    p = np.sort(p_clean[np.isfinite(p_clean)])[::-1]  # descending
    if not len(p):  # no clean window
        return np.nan  # undefined
    k = int(np.floor(INNER_THRESHOLD_SHARE * len(p)))  # at most k clean windows may be in alarm
    if k >= len(p):  # every window may alarm
        return float(p[-1])  # the smallest value
    return float(np.nextafter(p[k], np.inf))  # just above the (k+1)-th largest value: at most k windows reach it


# ----------------------------------------------------------------------------------------------------------------------
# C4b (forest) and importance
# ----------------------------------------------------------------------------------------------------------------------
def permuted(idx: pd.DataFrame, y: np.ndarray, tr: np.ndarray, k: int, by_rx: bool) -> tuple:  # function: permutation k inside each training session (x receiver when pooled): 0/1 labels and label strings
    rng = np.random.default_rng(SEED + 1000 + k)  # the forest script's C4b seed
    key = (idx.test_code + "|" + idx.receiver).to_numpy() if by_rx else idx.test_code.to_numpy()  # the permutation groups
    yp, lp = y.copy(), idx.tlabel.to_numpy(object).copy()  # copies
    for s in sorted(set(key[tr])):  # groups in sorted order (the forest script's c4b_permuted)
        r_ = tr[key[tr] == s]  # its training rows
        pi = rng.permutation(len(r_))  # the permutation (Generator.permutation(array) shuffles exactly as array[permutation(len)])
        yp[r_], lp[r_] = y[r_][pi], lp[r_][pi]  # both permuted the same way
    return yp, lp  # labels


def family_importance(predict, Zte: np.ndarray, lab_te: np.ndarray, names: list, fam: dict, ident: dict) -> pd.DataFrame:  # function: the forest script's importance for the family grouping on all scored windows (importance_cell's family/all part)
    sc = np.isin(lab_te, SCORED)  # scored windows
    unplaced = [c for c in names if fam.get(c, "receiver_indicator") not in FAMILIES]  # columns outside the nine families
    fams = FAMILIES + sorted({fam.get(c, "receiver_indicator") for c in unplaced})  # families in the forest script's order
    groups = {f: [j for j, c in enumerate(names) if fam.get(c, "receiver_indicator") == f] for f in fams}  # dict comprehension: family -> selected column positions
    s, _ = R.importance(predict, Zte[sc], lab_te[sc], groups, {**ident, "grouping": "family", "subset": "all"}, keep_repeats=False)  # the forest script's permutation importance
    return pd.DataFrame(s)  # rows


def classify_cells(ip: pd.DataFrame, ib: pd.DataFrame, prim: pd.DataFrame, b4: pd.DataFrame, med: dict, cells: list) -> pd.DataFrame:  # function: the forest script's classification table (its loop, for a given list of cells) with the forest script's classify rule
    rows = []  # rows
    for rx, held in cells:  # cells
        pa = float(prim[(prim.receiver == rx) & (prim.held_out == held)].auroc.iloc[0])  # the true model's held-out AUROC
        ba = float(b4[(b4.receiver == rx) & (b4.held_out == held) & (b4.perm == med[(rx, held)])].auroc.iloc[0])  # the median C4b model's
        q = lambda d: d[(d.receiver == rx) & (d.held_out == held) & (d.grouping == "family") & (d.subset == "all") & (d.metric == "auroc")].set_index("group")  # lambda: the cell's family AUROC drops
        t, b = q(ip), q(ib)  # true and C4b
        for f in FAMILIES:  # the nine families
            sel_t, sel_b = t.at[f, "status"] == "selected", b.at[f, "status"] == "selected"  # selected in each model
            tm, tp5 = (float(t.at[f, "drop_mean"]), float(t.at[f, "drop_p5"])) if sel_t else (np.nan, np.nan)  # true drop
            bm = float(b.at[f, "drop_mean"]) if sel_b else 0.0  # C4b drop (0 when not selected, as in the forest script)
            # dict literal over two lines: the forest script's columns
            r = {"receiver": rx, "held_out": held, "family": f, "primary_auroc": pa, "c4b_median_perm": med[(rx, held)], "c4b_auroc": ba, "selected_primary": sel_t, "selected_c4b": sel_b,
                 # dict entry: mean and 5th percentile of the true drop, the permuted drop, and their ratio
                 "true_drop_mean": tm, "true_drop_p5": tp5, "c4b_drop_mean": bm, "ratio": bm / tm if sel_t and tm > 0 else np.nan}
            r["class"] = R.classify(tm, tp5, bm, pa, sel_t, R.RATIO)  # the forest script's rule at ratio 0.5
            for rr in R.RATIO_SENS:  # the forest script's sensitivity ratios
                r[f"class_ratio_{rr}"] = R.classify(tm, tp5, bm, pa, sel_t, rr)  # at that ratio
            rows.append(r)  # kept
    return pd.DataFrame(rows)  # table


def c4b_run(kept: dict, cells_c4b: list, wc: pd.DataFrame, cfg: dict, refit_screen: bool = False, n_perm: int = N_PERM) -> tuple:  # function: C4b for the forest on kept outer fits; refit_screen: the jamming screen and MI refitted on the permuted labels
    fam, _ = R.name_maps(wc)  # the forest script's column -> family map
    cols = d1_columns(wc)  # orbit-free columns
    b4, ip, ib, med = [], [], [], {}  # outputs
    for (rxs, held), kp in kept.items():  # kept fits
        targets = [r for r in rxs if (r, held) in cells_c4b]  # the receivers this fit serves
        if not targets:  # none
            continue  # next
        idx, X, tr, y, prep = kp["idx"], kp["X"], kp["tr"], kp["y"], kp["prep"]  # the fit's frames
        lab_te = {r: idx.label.to_numpy(object)[kp["ev"][r]] for r in targets}  # dict comprehension: true held-out labels per receiver
        if cfg["pooled"] is False:  # single receiver: the forest script's own permutation applies
            assert all(np.array_equal(permuted(idx, y, tr, k, False)[0], R.c4b_permuted({"idx": idx, "y": y, "tr": tr}, k)) for k in (0, 1, 2))  # generator in all(): this permutation equals the forest script's for three seeds
        fits = {}  # perm -> (model, Z blocks, names)
        for k in range(n_perm):  # permutations
            yp, lp = permuted(idx, y, tr, k, cfg["pooled"])  # within session (x receiver)
            if refit_screen:  # C4b-refit: screen and preparation from the permuted labels
                cand = screened(idx.assign(plabel=lp), X, kp["train"], cols, label_col="plabel")  # the screen script's screen on permuted labels
                V = M.with_indicators(X, cand, wc, W, BLOCKS)  # candidates and indicator
                pp = M.prepare(V, tr)  # label-free preparation of the new candidate set
            else:  # the forest script's C4b: the true-label screen and preparation reused
                pp, cand = prep, kp["cand"]  # the fit's preparation and candidate list
            sel = M.mi_top(pp, yp, tr, SEED)  # MI on the permuted labels (the forest script's C4b)
            Z = pp["Z"][:, sel]  # selected columns
            names = [pp["names"][j] for j in sel]  # names
            if cfg["rx_ind"]:  # receiver indicator forced in
                Z = np.column_stack([Z] + [(idx.receiver.to_numpy() == c.replace("rx_is_", "")).astype(float) for c in IND_COLS])  # two 0/1 columns
                names = names + IND_COLS  # names
            m = R.fit_forest(Z[tr], yp[tr], SEED)  # the forest script's forest on the permuted labels
            for r in targets:  # receivers
                p = R.proba(m, Z[kp["ev"][r]])  # held-out probabilities
                lab = lab_te[r]  # true labels
                sc, att, cln = np.isin(lab, SCORED), np.isin(lab, ATTACK), lab == "clean"  # masks
                al = p >= 0.5  # alarms
                b4.append({"run_id": cfg["run_id"], "receiver": r, "held_out": held, "perm": k, "seed": SEED + 1000 + k, "auroc": float(roc_auc_score(att[sc], p[sc])), "recall": float(al[att].mean()), "clean_in_alarm_share": float(al[cln].mean()) if cln.any() else np.nan, "n_selected": len(names), "n_candidates": len(cand), "candidates_equal_true": cand == kp["cand"], "selected_columns": ";".join(names)})  # one C4b row
            fits[k] = (m, Z, names)  # kept until the median is known
        bt = pd.DataFrame(b4)  # rows so far
        for r in targets:  # receivers
            mk = R.c4b_median_perm(bt[bt.run_id == cfg["run_id"]], r, held)  # the forest script's median model: the 50th of 100 by AUROC, ties by permutation
            if n_perm != N_PERM:  # a shortened run: the median rank scales
                d = bt[(bt.receiver == r) & (bt.held_out == held)].sort_values(["auroc", "perm"], kind="stable")  # sorted as in the forest script
                mk = int(d.perm.iloc[n_perm // 2 - 1])  # the same rank rule
            med[(r, held)] = mk  # kept
            m, Z, names = fits[mk]  # the median model
            ident = {"run_id": cfg["run_id"], "model": "forest", "receiver": r, "held_out": held}  # identifiers
            ib.append(family_importance(lambda Ze, m=m: R.proba(m, Ze), Z[kp["ev"][r]], lab_te[r], names, fam, {**ident, "config": "c4b_median"}))  # its importance
            mt = kp["model_rf"]  # the true-label forest
            ip.append(family_importance(lambda Ze, m=mt: R.proba(m, Ze), kp["Z"][kp["ev"][r]], lab_te[r], kp["names"], fam, {**ident, "config": "true"}))  # the true model's importance
        fits.clear()  # memory freed
    return pd.DataFrame(b4), pd.concat(ip, ignore_index=True), pd.concat(ib, ignore_index=True), med  # outputs
