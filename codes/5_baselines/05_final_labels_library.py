"""The library of 06_run_baselines.py: the adopted models on the final labels (report only; nothing is selected here).

Every fitted step is the project's own code, imported unchanged through the library (codes/5_baselines/04_pipeline_library.py), which
loads the window-feature script, the screen script, the scoring module, the regression script and the forest script: the screen script's jamming screen, the regression script's indicators, preparation, MI, logistic regression and fold evaluation, the forest script's
forest, grouped permutation importance and classification rule. P (the adopted primary), P-noStave and P-unrounded run through
choices_lib.outer() and choices_lib.nested() unchanged. What is new here is plumbing only:
  * load_cells(): the window files of a label set at any length with the training label (strict) beside the scoring label;
  * outer_v(): choices_lib.outer()'s loop generalised to a variant specification (feature set, screen rule, statistic filter, added
    columns, indicator blocks, selection, C, min_samples_leaf, compression, MI and forest seeds), for the variants, C5, C6 and
    P-nocompress; for P's specification it reproduces choices_lib.outer() bitwise (gate);
  * c4a() and c4b_refit() for both models: the logistic regression's fits that the regression script's warning rule stops are recorded as
    failed; a cell with more than 5 of 100 failed fits is "not assessable";
  * importance() and classification() around the forest script's importance_cell() and classify(), with the selection check;
  * e6(), k5(), probe(), unrounded_cells().
2.6.12 is sealed: the regression script's reader drops it at read, no 2.6.12 window enters a fit, a threshold, a selection, an importance or the probe.
"""
from __future__ import annotations  # future import: type hints kept as text

import contextlib  # standard library: context managers for the figure-name and print redirections
import importlib.util  # standard library: loads the project scripts as modules
import gzip  # standard library: gzip with a fixed header time (byte-stable)
import io  # standard library: in-memory text (the 6-significant-figure round trip; swallowed output)
import sys  # standard library: module registry
from pathlib import Path  # standard library: paths

import numpy as np  # numpy: arrays
import pandas as pd  # pandas: tables
from sklearn.ensemble import RandomForestClassifier  # the forest of the random forest (only for min_samples_leaf other than 5)
from sklearn.metrics import roc_auc_score  # threshold-free score of held-out probabilities

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
PROC = PROJECT / "Data" / "processed"  # processed tables
FINAL = PROC / "features_final"  # the final window files
REGISTERED = PROC / "features"  # the registered window files


def load(name: str, path: Path):  # function: loads a script as a module, unchanged (main() stays behind its guard)
    spec = importlib.util.spec_from_file_location(name, path)  # where from
    mod = importlib.util.module_from_spec(spec)  # an empty module object
    sys.modules[name] = mod  # registered so its own imports resolve
    spec.loader.exec_module(mod)  # its top level
    return mod  # the module


L = load("choices_lib", PROJECT / "codes" / "5_baselines" / "04_pipeline_library.py")  # the library under its own name (the runs import it by that name)
M, R, S, SC, F = L.M, L.R, L.S, L.SC, L.F  # the regression, forest, screen, scoring and window-feature scripts exactly as choices_lib holds them
C9 = load("sequential_baselines_for_final_labels", PROJECT / "codes" / "5_baselines" / "03_sequential_baselines.py")  # 09c: the causal k-of-n alarm rule and the variant figure
SB = load("stave_base_for_final", PROJECT / "codes" / "4_features" / "04_stave_stop_base.py")  # the final label set's Stave base builder (the feature script's functions)

SEED, RXS, KEYS, W = L.SEED, L.RXS, L.KEYS, L.W  # 20260914, the receivers, the key columns, w = 10
ATTACK, SCORED, SEALED = L.ATTACK, L.SCORED, L.SEALED  # attack labels, scored labels, "2.6.12"
CHANGE = {"std", "skew", "kurt"}  # the shift-invariant window statistics
N_PERM = 100  # permutations per cell for C4a and C4b-refit (E1)
C4A_FOLDS, C4A_BAND = ["2.6.2", "2.6.4"], (0.4, 0.6)  # C4a's two folds and its pass band (the logistic regression and the random forest)
FAIL_MAX = 5  # more than 5 failed fits of 100 make a cell "not assessable"
UNINF, Q95 = 0.995, 0.95  # the session control: a permuted 95th percentile at or above 0.995 is "uninformative"; the percentile itself
E6_FAMS, E6_LIMIT = ("sos", "sosdiff", "tk"), 6.9  # E6: the squared window statistics and the largest held-out excess allowed after compression
STOP = (int(pd.Timestamp("2026-09-14 12:27:56", tz="UTC").timestamp()), int(pd.Timestamp("2026-09-14 12:32:36", tz="UTC").timestamp()))  # tuple: the Stave stop
FAIL_COLS = ["run_id", "model", "held_out", "receivers", "reason"]  # columns of the failed-fit table
MODEL_NAME = {"lr": "logreg", "rf": "forest"}  # dict: model key -> the name the forest script writes in its importance tables


# ----------------------------------------------------------------------------------------------------------------------
# writing and reading
# ----------------------------------------------------------------------------------------------------------------------
def write_csv(df: pd.DataFrame, path: Path) -> None:  # function: plain CSV with fixed line endings, folder created
    path.parent.mkdir(parents=True, exist_ok=True)  # the folder
    df.to_csv(path, index=False, lineterminator="\n")  # written


def write_gz(df: pd.DataFrame, path: Path) -> None:  # function: gzip CSV without name or time in the header (byte-stable)
    path.parent.mkdir(parents=True, exist_ok=True)  # the folder
    with open(path, "wb") as fh, gzip.GzipFile(filename="", mode="wb", fileobj=fh, mtime=0, compresslevel=6) as gz:  # with-block over two context managers
        gz.write(df.to_csv(index=False, lineterminator="\n").encode("utf-8"))  # the table as CSV text


def read_exact(path: Path, **kw) -> pd.DataFrame:  # function: CSV with exact float parsing and session codes kept as text
    return pd.read_csv(path, float_precision="round_trip", dtype={"test_code": str, "held_out": str, "outer_held_out": str}, **kw)  # round_trip: each written number back to its float


@contextlib.contextmanager  # decorator: turns the generator below into a "with" context manager
def figures_to(graph: Path, suffix: str, *mods):  # function: while active, the named modules' save_fig writes <graph>/<stem><suffix>.png
    orig = [m.save_fig for m in mods]  # list comprehension: each module's own saver
    for m, o in zip(mods, orig):  # for loop over the modules
        m.save_fig = (lambda fig, path, o=o: o(fig, graph / f"{Path(path).stem}{suffix}.png"))  # lambda with a default argument: the same byte-stable saver, renamed path
    graph.mkdir(parents=True, exist_ok=True)  # the folder
    try:  # try/finally: the savers always come back
        yield  # the body of the with-block runs here
    finally:  # always
        for m, o in zip(mods, orig):  # for loop over the modules
            m.save_fig = o  # restored


def quiet(fn, *a, **k):  # function: calls fn with its printed progress swallowed (the run log keeps final_run's own lines)
    with contextlib.redirect_stdout(io.StringIO()):  # context manager: stdout into a discarded buffer
        return fn(*a, **k)  # the call's result


# ----------------------------------------------------------------------------------------------------------------------
# windows
# ----------------------------------------------------------------------------------------------------------------------
def use_labels(label_set: str) -> None:  # function: points the scoring module (through choices_lib) at the registered or the final label set
    SC.use_label_set(label_set)  # the scoring module's files: labels_1hz(_b), labels_evidence(_b), scoring_fold_policy(_b)


def wc_all() -> pd.DataFrame:  # function: the feature script's window dictionary at every length (the regression script's reader)
    return M.read_table(PROC / "features" / "features_window_columns.csv")  # the dictionary


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


def drop_stave(cells: dict) -> dict:  # function: the same cells without the Stave stop's windows (P-noStave, V1, V2)
    out = {}  # receiver -> (index, values)
    for rx, (idx, X) in cells.items():  # for loop over the receivers
        k = ~((idx.test_code == "2.6.5") & idx.sec.between(*STOP)).to_numpy()  # windows that do not end inside the stop
        out[rx] = (idx[k].reset_index(drop=True), X[k].reset_index(drop=True))  # kept
    return out  # cells


def d1_cells(cells: dict, wc10: pd.DataFrame) -> dict:  # function: the cells restricted to the orbit-free columns (what choices_lib.outer reads)
    cols = L.d1_columns(wc10)  # 778 columns
    return {rx: (idx, X[cols]) for rx, (idx, X) in cells.items()}  # dict comprehension


# ----------------------------------------------------------------------------------------------------------------------
# the generalised outer loop (variants, C5, C6, P-nocompress)
# ----------------------------------------------------------------------------------------------------------------------
# dict literal: P's specification (the logistic regression's logistic regression and the random forest's forest as registered, refitted screen on the training label)
P_SPEC = {"w": 10, "set": "no_residual", "screen": "refit", "rule": "reason", "change_only": False, "add": "", "blocks": ("orbitfree",), "known": False,
          "select": "mi", "C": 1.0, "leaf": 5, "compress": True, "mi_seed": SEED, "forest_seed": SEED}  # the rest of the specification


def set_columns(wc: pd.DataFrame, w: int, name: str) -> list:  # function: a feature set's window columns at length w, in the feature script's order
    if name == "geometry":  # if: C5's block
        return M.geometry_block(wc, w)  # the 370 columns at w = 10 (orbit-free and self origins)
    x = wc[wc.w == w]  # this length
    return list(x[x[name] == 1].column)  # the set's columns (no_residual, main or known_antenna)


def screened(idx: pd.DataFrame, X: pd.DataFrame, train: list, cols: list, rule: str, label_col: str = "tlabel") -> list:  # function: the screen script's screen refitted on the training sessions, under the floor rule ("reason") or the no-floor rule
    sc = S.screen_cell(idx.assign(label=idx[label_col]), X[cols], train)  # the screen script's screen with the chosen labels in the "label" column
    flag = dict(zip(sc.column, sc[rule].fillna("")))  # column -> reasons under the rule
    return [c for c in cols if not flag.get(c)]  # list comprehension: unflagged columns, in the feature script's order (the screen script's lists_and_summary)


def candidates(spec: dict, idx: pd.DataFrame, X: pd.DataFrame, wc: pd.DataFrame, train: list, fl, rx: str, held: str) -> list:  # function: one fit's candidate columns under a specification
    w = spec["w"]  # window length
    if spec["screen"] == "registered":  # if: the registered lists of feature_lists (gates only)
        cand = M.cand_list(fl, w, rx, held, spec["set"], spec["rule"])  # the regression script's reader of the lists ("screened", "screened_no_floor" or "main")
    elif spec["screen"] == "none":  # elif: no screen (unscreened, C5)
        cand = set_columns(wc, w, spec["set"])  # every column of the set
    else:  # refitted screen (the adopted configuration)
        cand = screened(idx, X, train, set_columns(wc, w, spec["set"]), spec["rule"])  # the screen script's screen on the training sessions under tlabel
    if spec["change_only"]:  # if: V3, std, skew and kurt only
        st = dict(zip(wc[wc.w == w].column, wc[wc.w == w].stat))  # column -> statistic
        cand = [c for c in cand if st[c] in CHANGE]  # list comprehension: the shift-invariant columns
    if spec["add"] == "geometry_levels":  # if: plus_geometry_levels: the geometry block's level statistics of the orbit-free origin, unscreened (the sequential-baseline script's main_plus_geometry on the orbit-free set)
        x = wc[(wc.w == w) & (wc.geometry_ablation == 1) & (wc.origin == "orbit-free")]  # their dictionary rows
        cand = cand + [c for c in x.column if c not in cand]  # list comprehension: appended in the feature script's order
    return cand  # the list


def with_blocks(X: pd.DataFrame, cand: list, wc: pd.DataFrame, spec: dict) -> pd.DataFrame:  # function: candidates plus indicators (the regression script's rule, the known-block indicator of 09c/10c for V2)
    V = M.with_indicators(X, cand, wc, spec["w"], spec["blocks"])  # the regression script's indicators of the named blocks
    if spec["known"]:  # if: V2
        x = wc[wc.w == spec["w"]].set_index("column")  # dictionary rows
        cols = [c for c in cand if x.at[c, "part"] == "known" and x.at[c, "stat"] not in M.UNDEFINED_STATS]  # list comprehension: the known block's columns other than skewness and kurtosis
        V[C9.KNOWN_IND] = X[cols].isna().any(axis=1).astype(float).to_numpy()  # 1 when any is missing (the sequential-baseline script's rule)
    return V  # candidates and indicators


def fit_rf(Zt: np.ndarray, yt: np.ndarray, seed: int, leaf: int) -> RandomForestClassifier:  # function: the random forest's forest with a chosen min_samples_leaf (the forest script's fit_forest when leaf is 5)
    if leaf == R.RF_PARAMS["min_samples_leaf"]:  # if: the registered leaf size
        return R.fit_forest(Zt, yt, seed)  # the forest script's own function
    m = RandomForestClassifier(**{**R.RF_PARAMS, "min_samples_leaf": leaf}, random_state=seed, n_jobs=R.FIT_JOBS).fit(Zt, yt)  # dict unpacking with one key replaced (the forest's fit_leaf variant)
    m.set_params(n_jobs=R.PRED_JOBS)  # serial prediction (bitwise reproducible)
    return m  # the forest


def fit_predict(model: str, Z: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray, spec: dict, strict: bool, tag: dict) -> tuple:  # function: one model fitted and its held-out probabilities; (model, p, failure reason)
    if model == "lr":  # the logistic regression's logistic regression
        try:  # try: 09 stops on any warning; here a stop is a failed fit
            m, _ = M.fit_lr(Z[tr], y[tr], spec["C"], tag, strict=strict)  # the regression script's fit with every warning rule (strict for a compressed pipeline)
            p = M.predict(m, Z[te])[0]  # the regression script's checked prediction
            fail = ""  # no failure
        except SystemExit as e:  # the regression script's stop
            m, p, fail = None, np.full(len(te), np.nan), str(e)  # tuple assignment: no model
        M.FIT_DIAG.clear()  # the regression script's diagnostics list is not needed here
        return m, p, fail  # the three
    m = fit_rf(Z[tr], y[tr], spec["forest_seed"], spec["leaf"])  # the random forest's forest
    return m, R.proba(m, Z[te]), ""  # model, probabilities, no failure


def outer_v(cells: dict, wc: pd.DataFrame, spec: dict, models: tuple, run_id: str, fl=None, keep: bool = False) -> dict:  # function: choices_lib.outer()'s loop under a specification; per receiver, five folds
    rows, lats, prs, sel, kept, fails = [], [], [], [], {}, []  # outputs
    for held, train in S.folds():  # the screen script's five folds (choices_lib.outer's order: fold, then receiver)
        for rx in RXS:  # receivers, one fit each
            idx, X = cells[rx]  # the receiver's windows
            cand = candidates(spec, idx, X, wc, train, fl, rx, held)  # candidate columns
            why = L.unfittable(idx, train, cand)  # choices_lib's fittability rule
            if why:  # not fittable: recorded, not scored
                fails += [{"run_id": run_id, "model": mdl, "held_out": held, "receivers": rx, "reason": why} for mdl in models]  # list comprehension: failure rows
                continue  # next fit
            tr, te = L.train_rows(idx, train), L.eval_rows(idx, held)  # tuple assignment: scored training rows (tlabel) and held-out rows in time order
            if np.intersect1d(tr, te).size or set(idx.test_code.to_numpy()[te]) & set(train):  # leakage checks (the regression script's)
                raise SystemExit("STOP: group leakage")  # stop
            y = np.isin(idx.tlabel.to_numpy(), ATTACK).astype(int)  # 1 for attack windows (on the training label)
            V = with_blocks(X, cand, wc, spec)  # candidates and indicators
            prep = M.prepare(V, tr, compress=spec["compress"])  # the regression script's label-free steps
            if spec["select"] == "none":  # if: no Pearson pruning and no MI (V4)
                s_ = list(range(len(prep["names"])))  # every column with a training value
            else:  # MI top 36 of the Pearson-kept columns
                s_ = M.mi_top(prep, y, tr, spec["mi_seed"])  # the regression script's MI with the chosen seed (the jamming screen's SEED restored inside)
            Z = prep["Z"][:, s_]  # the selected columns
            names = [prep["names"][j] for j in s_]  # their names
            rec = {"idx": idx, "X": X, "V": V, "tr": tr, "te": te, "y": y, "prep": prep, "sel": s_, "names": names, "cand": cand, "train": train, "model": {}, "p": {}}  # dict literal: the fit's record
            for mdl in models:  # models
                m, p, fail = fit_predict(mdl, Z, y, tr, te, spec, prep["compressed"], {"run": run_id, "held_out": held, "receivers": rx})  # fitted
                if fail:  # failed (the regression script's stop rule)
                    fails.append({"run_id": run_id, "model": mdl, "held_out": held, "receivers": rx, "reason": fail})  # recorded
                    continue  # not scored
                ident = {"run_id": run_id, "model": mdl, "receiver": rx}  # identifiers
                row, lat, pr = L.evaluate(idx, te, p, held, ident, False)  # the regression script's evaluation through choices_lib
                trl = idx.tlabel.to_numpy()[tr]  # training labels
                row.update({"train_rows": len(tr), "train_clean": int((trl == "clean").sum()), "train_attack": int(np.isin(trl, ATTACK).sum())})  # training counts
                rows.append(row)  # kept
                lats += lat  # kept
                prs.append(pr)  # kept
                rec["model"][mdl], rec["p"][mdl] = m, p  # tuple assignment: kept in the record
            sel.append({"run_id": run_id, "held_out": held, "receivers": rx, "candidates": V.shape[1], "kept_after_pearson": len(prep["kept"]), "selected": len(s_),  # dict literal: what was fitted
                        "kept_columns": ";".join(prep["names"][j] for j in prep["kept"]), "selected_columns": ";".join(names)})  # the kept and selected columns
            if keep:  # kept for later steps
                kept[(rx, held)] = rec  # by receiver and fold
    prd = pd.concat(prs, ignore_index=True) if prs else pd.DataFrame()  # predictions (empty when nothing was scored)
    return {"metrics": pd.DataFrame(rows), "latency": pd.DataFrame(lats), "predictions": prd, "selection": pd.DataFrame(sel), "fails": pd.DataFrame(fails, columns=FAIL_COLS), "kept": kept}  # dict literal: outputs


def summary(out: dict, models: tuple) -> pd.DataFrame:  # function: choices_lib.summarise per model (the regression script's pooled figures, the fold-policy false-alarm pooling, early detection, pooled-prediction AUROC)
    parts = []  # per model
    for md in models:  # for loop over the models
        m, lt, pr = out["metrics"], out["latency"], out["predictions"]  # the frames
        if len(m) and (m.model == md).any():  # if: the model has scored folds
            parts.append(L.summarise(m[m.model == md], lt[lt.model == md], pr[pr.model == md], False).assign(model=md))  # choices_lib's pooled table
    s = pd.concat(parts, ignore_index=True)  # one table
    s["early_warning"] = s.pre_onset_alarm_share_pooled - s.clean_in_alarm_share_pooled  # the early-warning margin: pre-onset alarm share minus clean-window alarm share
    return s  # the table


# ----------------------------------------------------------------------------------------------------------------------
# P through choices_lib (outer, nested), with the kept fits in a common record form
# ----------------------------------------------------------------------------------------------------------------------
P_CFG = {"run_id": "P", "pooled": False, "rx_ind": False, "cand": "refit", "s11": False, "models": ("lr", "rf")}  # choices_lib's configuration of the adopted primary


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


def run_nested(cells: dict, wc10: pd.DataFrame, run_id: str) -> dict:  # function: choices_lib.nested() (inner leave-one-session-out, screen refitted inside every inner fold)
    nm, nl, npr, nsel, nf = quiet(L.nested, cells, wc10, {**P_CFG, "run_id": run_id})  # the nested run, printing nothing
    ns = L.nested_summary(nm, nl, npr, False)  # per model, receiver and outer fold, the inner figures pooled
    ns["early_warning"] = ns.pre_onset_alarm_share_pooled - ns.clean_in_alarm_share_pooled  # the early-warning margin per outer fold
    return {"metrics": nm, "latency": nl, "predictions": npr, "selection": nsel, "fails": nf, "summary": ns}  # dict literal: outputs


# ----------------------------------------------------------------------------------------------------------------------
# C4a and C4b-refit, both models
# ----------------------------------------------------------------------------------------------------------------------
def scores(p: np.ndarray, lab: np.ndarray) -> dict:  # function: held-out AUROC, recall and clean share of one permuted model
    sc, att, cln = np.isin(lab, SCORED), np.isin(lab, ATTACK), lab == "clean"  # masks
    al = p >= M.THR  # alarms at 0.5
    return {"auroc": float(roc_auc_score(att[sc], p[sc])) if att[sc].any() and (~att[sc]).any() else np.nan, "recall": float(al[att].mean()) if att.any() else np.nan, "clean_in_alarm_share": float(al[cln].mean()) if cln.any() else np.nan}  # dict literal: the three


def c4a(kept: dict, models: tuple, run_id: str, perms: range = range(N_PERM)) -> pd.DataFrame:  # function: labels permuted across all scored training windows (seed 20260914 + k), the true-label screen and preparation kept, MI and model on the permuted labels
    rows = []  # results
    for rx in RXS:  # for loop over the receivers
        for held in C4A_FOLDS:  # for loop over 2.6.2 and 2.6.4
            kp = kept[(rx, held)]  # P's fit of the cell
            prep, tr, te, y = kp["prep"], kp["tr"], kp["te"], kp["y"]  # its parts
            lab = kp["idx"].label.to_numpy()[te]  # held-out scoring labels
            for k in perms:  # for loop over the permutations
                yp = y.copy()  # the labels
                yp[tr] = np.random.default_rng(SEED + k).permutation(y[tr])  # shuffled across all scored training windows (the regression script's and the forest script's c4a)
                s_ = M.mi_top(prep, yp, tr, SEED)  # MI on the permuted labels
                Z = prep["Z"][:, s_]  # selected columns
                for mdl in models:  # models
                    _, p, fail = fit_predict(mdl, Z, yp, tr, te, P_SPEC, True, {"run": run_id, "held_out": held, "receivers": rx, "perm": k})  # fitted on the permuted labels
                    rows.append({"run_id": run_id, "model": mdl, "receiver": rx, "held_out": held, "perm": k, "seed": SEED + k, "failed": bool(fail), "reason": fail, **(scores(p, lab) if not fail else {"auroc": np.nan, "recall": np.nan, "clean_in_alarm_share": np.nan})})  # dict unpacking: one row
    return pd.DataFrame(rows)  # the distribution


def c4b_prep(kp: dict, wc10: pd.DataFrame, k: int) -> dict:  # function: C4b-refit's label-free part of permutation k: within-session permutation, then the screen, indicator, preparation and MI refitted on the permuted labels
    idx, X, tr, y = kp["idx"], kp["X"], kp["tr"], kp["y"]  # the fit's frames
    yp, lp = L.permuted(idx, y, tr, k, False)  # choices_lib's permutation inside each training session (the forest script's c4b_permuted on 0/1 labels, also the strings)
    cand = screened(idx.assign(plabel=lp), X, kp["train"], L.d1_columns(wc10), "reason", label_col="plabel")  # the screen script's screen on the permuted label strings
    if not cand:  # if: the permuted screen keeps no column
        return {"cand": cand, "yp": yp, "fail": "the permuted screen keeps no column"}  # a failed fit for every model
    V = M.with_indicators(X, cand, wc10, W, L.BLOCKS)  # candidates and indicator
    pp = M.prepare(V, tr)  # label-free preparation of the new candidate set
    s_ = M.mi_top(pp, yp, tr, SEED)  # MI on the permuted labels
    return {"cand": cand, "yp": yp, "V": V, "prep": pp, "sel": s_, "names": [pp["names"][j] for j in s_], "fail": ""}  # dict literal: shared by both models


def c4b_model(kp: dict, f: dict, mdl: str, run_id: str, held: str, rx: str, k: int) -> tuple:  # function: one model on a C4b-refit preparation: (model, held-out probabilities, failure)
    if f["fail"]:  # if: no column to fit
        return None, np.full(len(kp["te"]), np.nan), f["fail"]  # failed
    Z = f["prep"]["Z"][:, f["sel"]]  # selected columns
    return fit_predict(mdl, Z, f["yp"], kp["tr"], kp["te"], P_SPEC, True, {"run": run_id, "held_out": held, "receivers": rx, "perm": k})  # fitted on the permuted labels


def c4b_refit(kept: dict, wc10: pd.DataFrame, models: tuple, cells: list, run_id: str, perms: range = range(N_PERM)) -> pd.DataFrame:  # function: the environment's label-free session control for both models, every permutation recorded
    rows = []  # results
    for rx, held in cells:  # for loop over the cells
        kp = kept[(rx, held)]  # P's fit of the cell
        lab = kp["idx"].label.to_numpy()[kp["te"]]  # held-out scoring labels
        for k in perms:  # for loop over the permutations
            f = c4b_prep(kp, wc10, k)  # the permuted preparation, shared by the models
            for mdl in models:  # models
                _, p, fail = c4b_model(kp, f, mdl, run_id, held, rx, k)  # fitted
                sc_ = scores(p, lab) if not fail else {"auroc": np.nan, "recall": np.nan, "clean_in_alarm_share": np.nan}  # conditional expression: its scores
                rows.append({"run_id": run_id, "model": mdl, "receiver": rx, "held_out": held, "perm": k, "seed": SEED + 1000 + k, "failed": bool(fail), "reason": fail, **sc_,  # dict unpacking: one row
                             "n_candidates": len(f["cand"]), "candidates_equal_true": f["cand"] == kp["cand"], "n_selected": len(f.get("names", [])), "selected_columns": ";".join(f.get("names", []))})  # the permuted model's columns
    return pd.DataFrame(rows)  # the runs


def median_perm(b4: pd.DataFrame, mdl: str, rx: str, held: str) -> int:  # function: the median permuted model of a cell: the forest script's rule (50th of 100 by AUROC, ties by permutation) over the completed fits
    d = b4[(b4.model == mdl) & (b4.receiver == rx) & (b4.held_out == held) & ~b4.failed.astype(bool)].sort_values(["auroc", "perm"], kind="stable")  # completed fits sorted as 10 sorts
    return int(d.perm.iloc[len(d) // 2 - 1])  # rank n/2 (50 of 100)


def cell_verdicts(b4: pd.DataFrame, true_auc: dict, cells: list, run_id: str) -> pd.DataFrame:  # function: the session control per cell and model (its precision, and the "not assessable" rule)
    rows = []  # results
    for (mdl, rx, held) in [(m_, r_, h_) for m_ in sorted(set(b4.model)) for r_, h_ in cells]:  # list comprehension: model x cell
        d = b4[(b4.model == mdl) & (b4.receiver == rx) & (b4.held_out == held)]  # the cell's permutations
        nf = int(d.failed.astype(bool).sum())  # failed fits
        ok = d[~d.failed.astype(bool)].auroc  # completed AUROCs
        t = true_auc.get((mdl, rx, held), np.nan)  # the adopted model's held-out AUROC
        if nf > FAIL_MAX:  # if: more than 5 failed fits
            v, med, p5, p95 = "not assessable", np.nan, np.nan, np.nan  # tuple assignment: no median, no percentile
        else:  # enough fits
            med, p5, p95 = float(ok.median()), float(ok.quantile(0.05)), float(ok.quantile(Q95))  # pandas quantile, linear interpolation (E1)
            v = "uninformative" if p95 >= UNINF else ("supports" if t > p95 else "fails")  # nested conditional expression: E1's precedence, strict ">"
        rows.append({"run_id": run_id, "model": mdl, "receiver": rx, "held_out": held, "perms": len(d), "failed_fits": nf, "true_auroc": t, "perm_auroc_median": med, "perm_auroc_p5": p5, "perm_auroc_p95": p95, "verdict": v})  # dict literal: one cell
    return pd.DataFrame(rows)  # the table


# ----------------------------------------------------------------------------------------------------------------------
# importance and classification (the forest script's code)
# ----------------------------------------------------------------------------------------------------------------------
def predictor(mdl: str, model):  # function: the probability function of a fitted model
    if mdl == "lr":  # the logistic regression
        return lambda Z, m=model: M.predict(m, Z)[0]  # lambda with a default argument: the regression script's checked prediction
    return lambda Z, m=model: R.proba(m, Z)  # the forest script's probability of attack


def importance(kept: dict, wc10: pd.DataFrame, mdl: str, config: str, order: list) -> tuple:  # function: the forest script's importance_cell (every grouping, both subsets, imputed shares) of one model in every cell named
    fam, xs = R.name_maps(wc10)  # the forest script's family and secondary-group maps
    summ, reps, shares = [], [], []  # outputs
    for rx, held in order:  # for loop over the cells in the order given (the forest script's: receiver, then fold)
        kp = kept[(rx, held)]  # the fit
        s, r, sh = quiet(R.importance_cell, predictor(mdl, kp["model"][mdl]), kp, kp["names"], kp["sel"], fam, xs, {"config": config, "model": MODEL_NAME[mdl], "receiver": rx, "held_out": held})  # the forest script's code
        summ += s  # kept
        reps += r  # kept
        shares += sh  # kept
    return pd.DataFrame(summ), pd.DataFrame(reps), pd.DataFrame(shares)  # three tables


def classification(ip: pd.DataFrame, ib: pd.DataFrame, true_auc: dict, b4: pd.DataFrame, med: dict, mdl: str, cells: list) -> pd.DataFrame:  # function: the forest script's classification per cell and family, with the selection check
    rows = []  # results
    for rx, held in cells:  # for loop over the cells
        pa = true_auc[(mdl, rx, held)]  # the adopted model's held-out AUROC
        if (rx, held) not in med:  # if: the cell is not assessable (no median model)
            continue  # no classification
        ba = float(b4[(b4.model == mdl) & (b4.receiver == rx) & (b4.held_out == held) & (b4.perm == med[(rx, held)])].auroc.iloc[0])  # the median permuted model's AUROC
        q = lambda d: d[(d.receiver == rx) & (d.held_out == held) & (d.grouping == "family") & (d.subset == "all") & (d.metric == "auroc")].set_index("group")  # lambda: the cell's family AUROC rows
        t, b = q(ip), q(ib)  # true and median permuted
        for f in R.FAMILIES:  # the nine families
            sel_t, sel_b = t.at[f, "status"] == "selected", b.at[f, "status"] == "selected"  # selected in each model
            tm, tp5 = (float(t.at[f, "drop_mean"]), float(t.at[f, "drop_p5"])) if sel_t else (np.nan, np.nan)  # true drop and its 5th percentile
            bm = float(b.at[f, "drop_mean"]) if sel_b else 0.0  # permuted drop (0 when not selected, as in the forest script)
            # dict literal over two lines: the forest script's columns
            r = {"model": MODEL_NAME[mdl], "receiver": rx, "held_out": held, "family": f, "primary_auroc": pa, "c4b_median_perm": med[(rx, held)], "c4b_auroc": ba, "selected_primary": sel_t, "selected_c4b": sel_b,
                 "true_drop_mean": tm, "true_drop_p5": tp5, "c4b_drop_mean": bm, "ratio": bm / tm if sel_t and tm > 0 else np.nan}  # the inputs of the forest script's rule
            r["class"] = R.classify(tm, tp5, bm, pa, sel_t, R.RATIO)  # the forest script's rule at ratio 0.5
            for rr in R.RATIO_SENS:  # the forest script's sensitivity ratios
                r[f"class_ratio_{rr}"] = R.classify(tm, tp5, bm, pa, sel_t, rr)  # at that ratio
            r["e14_selection_dependent"] = r["class"] == "attack evidence" and not sel_b  # E14: attack evidence that the median permuted model never selected
            rows.append(r)  # kept
    return pd.DataFrame(rows)  # the table


# ----------------------------------------------------------------------------------------------------------------------
# E6, the inner-fold threshold, probe, unrounded windows
# ----------------------------------------------------------------------------------------------------------------------
def e6(kept: dict, wc10: pd.DataFrame) -> tuple:  # function: E6 per receiver, fold and family: largest |z| before and after compression, training and held-out rows; per-column failures
    st = dict(zip(wc10.column, wc10.stat))  # column -> window statistic
    rows, bad = [], []  # family rows; failing columns
    for (rx, held), kp in kept.items():  # for loop over the fits
        tr, te = kp["tr"], kp["te"]  # training and held-out rows
        scd = te[np.isin(kp["idx"].label.to_numpy()[te], SCORED)]  # scored held-out rows
        pu = M.prepare(kp["V"], tr, compress=False)  # the same steps without the compression
        Zc, Zu = kp["prep"]["Z"], pu["Z"]  # compressed and uncompressed
        if pu["names"] != kp["prep"]["names"]:  # if: the column sets differ (cannot, the steps before compression are label-free and equal)
            raise SystemExit("STOP: E6 column sets differ")  # stop
        skipped = [n for j, n in enumerate(pu["names"]) if not np.array_equal(Zc[:, j], np.sign(Zu[:, j]) * np.log1p(np.abs(Zu[:, j])))]  # list comprehension: columns not compressed
        sel = set(kp["names"])  # selected columns
        for fam in E6_FAMS:  # for loop over the three families
            js = [j for j, n in enumerate(pu["names"]) if st.get(n) == fam]  # list comprehension: the family's prepared columns
            if not js:  # if: none in this fit
                continue  # next family
            mx = lambda Z, r_: np.abs(Z[np.ix_(r_, js)]).max(axis=0) if len(r_) else np.full(len(js), np.nan)  # lambda: per column, the largest |value| over rows r_
            exc = mx(Zc, te) - mx(Zc, tr)  # per column: held-out maximum minus training maximum, after compression
            over = [pu["names"][j] for j, e in zip(js, exc) if e > E6_LIMIT]  # list comprehension: columns beyond the limit
            bad += [{"receiver": rx, "held_out": held, "family": fam, "column": c, "selected": c in sel} for c in over]  # list comprehension: failing columns
            # list.append of a dict over three lines: the family's maxima
            rows.append({"receiver": rx, "held_out": held, "family": fam, "n_columns": len(js), "n_selected": sum(pu["names"][j] in sel for j in js),
                         "max_abs_z_train_pre": float(mx(Zu, tr).max()), "max_abs_z_heldout_pre": float(mx(Zu, te).max()), "max_abs_z_train_post": float(mx(Zc, tr).max()), "max_abs_z_heldout_post": float(mx(Zc, te).max()),  # the maxima before and after compression
                         "max_abs_z_heldout_scored_post": float(np.nanmax(mx(Zc, scd))) if len(scd) else np.nan, "largest_heldout_excess_post": float(np.max(exc)), "columns_over_6_9": len(over), "columns_skipping_compression": len([n for n in skipped if st.get(n) == fam])})  # the scored maximum, the largest excess and the counts
        bad += [{"receiver": rx, "held_out": held, "family": "skipped compression", "column": n, "selected": n in sel} for n in skipped]  # list comprehension: any column not compressed
    return pd.DataFrame(rows), pd.DataFrame(bad, columns=["receiver", "held_out", "family", "column", "selected"])  # the table and the failures


def k5(nested_pred: pd.DataFrame, outer: dict, cells: dict) -> dict:  # function: the inner-fold threshold (choices_run.cmd_k5's logic) on P's nested out-of-fold and outer predictions
    thr, rows, lats, prs = [], [], [], []  # outputs
    opr = outer["predictions"]  # P's outer predictions
    for (md, rx, held), d in nested_pred.groupby(["model", "receiver", "outer_held_out"], sort=False):  # per model, receiver and outer fold
        pc = d[d.label == "clean"].p.to_numpy()  # clean out-of-fold probabilities
        t = L.k5_threshold(pc)  # the lowest t with at most floor(0.05 n) clean windows at p >= t
        att = d[d.label.isin(ATTACK)]  # attack out-of-fold windows
        thr.append({"model": md, "receiver": rx, "outer_held_out": held, "oof_clean_windows": len(pc), "threshold": t, "in_sample_clean_share": float((pc >= t).mean()) if len(pc) and np.isfinite(t) else np.nan, "in_sample_recall": float((att.p >= t).mean()) if len(att) and np.isfinite(t) else np.nan, "note": "in-sample for the threshold"})  # threshold row
        if not np.isfinite(t):  # no threshold
            continue  # not scored
        idx_r = cells[rx][0]  # the receiver's index
        te = L.eval_rows(idx_r, held)  # its held-out rows
        p = opr[(opr.model == md) & (opr.receiver == rx) & (opr.held_out == held)].p.to_numpy()  # P's outer probabilities, same order
        row, lat, pr = L.evaluate(idx_r, te, p, held, {"run_id": "INNER_THRESHOLD", "model": md, "receiver": rx}, False, thr=t)  # scored at t
        row["threshold"] = t  # recorded
        rows.append(row)  # kept
        lats += lat  # kept
        prs.append(pr)  # kept
    out = {"thresholds": pd.DataFrame(thr), "metrics": pd.DataFrame(rows), "latency": pd.DataFrame(lats), "predictions": pd.concat(prs, ignore_index=True) if prs else pd.DataFrame()}  # dict literal
    return out  # thresholds and the scores at them


def probe(root: Path, train_label: bool) -> pd.DataFrame:  # function: the screen script's session-fingerprint probe (Table 58's method) on the five position sessions: 2.6.12 dropped at read
    orig, old_set = S.load_windows, dict(S.SET)  # the screen script's reader and its folder setting
    seen = []  # sessions the probe received
    def five(w: int, rx: str) -> tuple:  # nested function: the screen script's reader with 2.6.12 dropped before the probe sees a row
        idx, X = orig(w, rx)  # the screen script's reader
        k = (idx.test_code.astype(str) != SEALED).to_numpy()  # rows outside the sealed session
        seen.extend(sorted(set(idx.test_code.astype(str)[k])))  # sessions passed on
        return idx[k].reset_index(drop=True), X[k].reset_index(drop=True)  # 2.6.12 dropped
    S.SET.update(root=root, train_label=train_label)  # dict.update: the window folder
    S.load_windows = five  # the screen script's probe() calls load_windows from its module, so this replaces it
    try:  # try/finally: the reader and the setting always come back
        pr = quiet(S.probe, wc_all())  # the screen script's probe, unchanged
    finally:  # always
        S.load_windows = orig  # restored
        S.SET.clear()  # emptied
        S.SET.update(old_set)  # restored
    if SEALED in set(seen):  # if: a 2.6.12 row reached the probe (cannot)
        raise SystemExit("STOP: 2.6.12 reached the probe")  # stop
    return pr  # the table


def unrounded_cells(wc10: pd.DataFrame, final_lab: pd.DataFrame, train_lab: pd.DataFrame) -> dict:  # function: the final orbit-free windows rebuilt in memory, unrounded, Stave stop included as the feature script's final run fills it
    bases = L.d1_bases(wc10)  # the 103 base columns
    base = L.build_free_base(False)  # the feature script's functions, committed spans, five sessions, unrounded (choices_lib)
    sb = SB.build_265(SB.STOP_START)  # the final label set's Stave base: 2.6.5 read from 12:27:56, unrounded
    sb = sb[sb.sec.between(*STOP)].set_index(KEYS)  # the stop's seconds
    b = base.set_index(KEYS)  # keyed
    k = b.index.isin(sb.index)  # the stop's rows in the base
    if b.loc[k, bases].notna().any().any() or k.sum() != len(sb):  # if: a stop row already holds a value or rows are missing
        raise SystemExit("STOP: the Stave rows of the unrounded base are not empty")  # stop
    b.loc[k, bases] = sb.loc[b.index[k], bases].to_numpy()  # filled, as the feature script's stave_fill fills the final base
    base = b.reset_index()  # keys back as columns
    lab = L.no_sealed(final_lab[KEYS + ["label"]])  # final labels, 2.6.12 dropped
    rows = F.window_rows(lab, W)  # the feature script's rule
    out = {}  # receiver -> (index, values)
    for rx in RXS:  # receivers
        parts_i, parts_v = [], []  # per session
        for (test, r_), g in lab[lab.receiver == rx].groupby(["test_code", "receiver"], sort=False):  # pairs in the label table's order
            bp = g[KEYS].merge(base, on=KEYS, how="left", validate="one_to_one")  # the base on the pair's grid
            st = F.win_stats(bp[bases].to_numpy(dtype=float), W)  # the feature script's statistics, unrounded
            r = rows[(rows.test_code == test) & (rows.receiver == rx)]  # the pair's windows
            j = {b_: n for n, b_ in enumerate(bases)}  # dict comprehension: base column -> position
            parts_v.append(pd.DataFrame({c: st[c.split("__")[1]][r.pos.to_numpy(), j[c.split("__")[0]]] for c in L.d1_columns(wc10)}))  # dict comprehension: every orbit-free column, unrounded
            parts_i.append(r[KEYS + ["label"]].reset_index(drop=True))  # index part
        idx = pd.concat(parts_i, ignore_index=True).merge(train_lab[KEYS + ["train_label"]], on=KEYS, how="left", validate="one_to_one")  # with the training label of the decision second
        idx["tlabel"] = np.where(idx.train_label == "clean", "clean", idx.label)  # choices_lib's tlabel
        out[rx] = (idx[KEYS + ["label", "tlabel"]], pd.concat(parts_v, ignore_index=True))  # cells
    return out  # the unrounded cells
