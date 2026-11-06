"""The adopted baselines on the final labels: the run table, the gates, the adopted models and their controls, the report, and
the check of a second pass.

Usage   python3 codes/5_baselines/06_run_baselines.py table      writes Data/processed/baselines_controls/FINAL_RUN_TABLE.csv
        python3 codes/5_baselines/06_run_baselines.py gates      the gates on the registered material (no adopted number); gates.csv
        python3 codes/5_baselines/06_run_baselines.py [--root DIR] [--smoke] p | controls | c4b | sens | variants | probe | report
        python3 codes/5_baselines/06_run_baselines.py no_quiet_negatives  P-no-quiet: P without the quiet training negatives, beside P (final/no_quiet_negatives/)
        python3 codes/5_baselines/06_run_baselines.py compare DIR   every output of the main tree against the second pass in DIR, byte for byte
The commands run in the order p, controls, c4b, sens, variants, probe, report (later ones read earlier outputs from the same root);
p alone writes the held-out predictions, metrics and pooled tables of Table 10.
--root DIR writes everything under DIR with the same layout (Data/processed/logistic_regression, random_forest, baselines_controls, Graph):
the second pass. --smoke runs every command on the REGISTERED windows with the strict cells and two permutations into
Data/processed/final_labels/scratch/smoke_test: a test of the code paths that produces no adopted number.
Run times go to Data/processed/baselines_controls/timing/ (outside the identity check).
"""
from __future__ import annotations  # future import: type hints kept as text

import filecmp  # standard library: byte-for-byte file comparison
import gzip  # standard library: reads a committed gzip table as text
import importlib.util  # standard library: loads final_lib
import io  # standard library: in-memory text
import sys  # standard library: arguments, module registry, exit status
import time  # standard library: run times
from pathlib import Path  # standard library: paths

import numpy as np  # numpy: arrays
import pandas as pd  # pandas: tables

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
_spec = importlib.util.spec_from_file_location("final_lib", Path(__file__).resolve().parent / "05_final_labels_library.py")  # the library beside this file
B = importlib.util.module_from_spec(_spec)  # an empty module
sys.modules["final_lib"] = B  # registered
_spec.loader.exec_module(B)  # its top level (it loads choices_lib, the sequential-baseline script and stave_base)
L, M, R, S, F, C9 = B.L, B.M, B.R, B.S, B.F, B.C9  # choices_lib and the regression, forest, screen, window-feature and sequential-baseline scripts as the library holds them
PROC = B.PROC  # the project's processed folder (inputs are always read from the project)
AA3 = PROC / "design_study" / "choices"  # the outputs
TIMING = PROC / "baselines_controls" / "timing"  # run times, outside the identity check
ARGS = sys.argv[1:]  # the command line
SMOKE = "--smoke" in ARGS  # test mode on the registered windows
ROOT = PROJECT / "Data" / "processed" / "final_labels" / "scratch" / "smoke_test" if SMOKE else (Path(ARGS[ARGS.index("--root") + 1]).resolve() if "--root" in ARGS else PROJECT)  # nested conditional expression: where outputs go
LR_DIR, RF_DIR = ROOT / "Data" / "processed" / "logistic_regression", ROOT / "Data" / "processed" / "random_forest"  # tuple assignment: adopted the logistic regression and the random forest folders
FINAL_DIR = ROOT / "Data" / "processed" / "baselines_controls"  # cross-model tables of the adopted models on the final labels
GRAPH = ROOT / "Graph"  # adopted figures, "_b" suffix
PERMS = range(2) if SMOKE else range(B.N_PERM)  # permutations: 100, or 2 in the test mode
MI_SEEDS = M.MI_SEEDS[:2] if SMOKE else M.MI_SEEDS  # C6's MI seeds (20260914-20260918)
FOREST_SEEDS = R.FOREST_SEEDS[:2] if SMOKE else R.FOREST_SEEDS  # C6's forest seeds
CELLS_C4B = R.c4b_cells()  # 2.6.2 and 2.6.4 of every receiver, 2.6.10 of u-blox 01
MODELS = ("lr", "rf")  # both models
CHECKS: list = []  # checks of this command, written to final/checks_<command>.csv


def check(name: str, ok, detail: str) -> None:  # function: records one check (ok None: reported)
    CHECKS.append({"check": name, "result": "REPORT" if ok is None else ("PASS" if ok else "FAIL"), "detail": detail})  # nested conditional expression picks the word
    print(f"  [{CHECKS[-1]['result']}] {name}: {detail}", flush=True)  # printed


def done(cmd: str, t0: float, stop_if_failed: bool = True) -> None:  # function: writes the checks and the run time; exits 1 on a failed check
    B.write_csv(pd.DataFrame(CHECKS, columns=["check", "result", "detail"]), FINAL_DIR / f"checks_{cmd}.csv")  # the checks
    TIMING.mkdir(parents=True, exist_ok=True)  # the folder
    tag = "smoke" if SMOKE else ("pass2" if ROOT != PROJECT else "pass1")  # which pass
    (TIMING / f"{tag}__{cmd}.csv").write_text(f"{tag},{cmd},{time.time() - t0:.1f}\n")  # one line
    if stop_if_failed and any(c["result"] == "FAIL" for c in CHECKS):  # if: a check failed
        sys.exit(f"STOP: a check of '{cmd}' failed")  # exit status 1


# ----------------------------------------------------------------------------------------------------------------------
# the run table (pre-registered)
# ----------------------------------------------------------------------------------------------------------------------
# list of dicts: the pre-registered runs
FIN = "final (labels/labels_1hz_final.csv, Stave stop included)"  # the label set of every adopted run
STRICT = "strict (labels/train_labels_1hz.csv; training = scoring inside the Stave stop)"  # the training rule of every adopted run
W10 = "features_final/w10 (6 s.f.)"  # the windows of P
RUNS = [  # list literal: one dict per pre-registered run
    {"run_id": "P", "command": "p", "models": "lr+rf", "label_set": FIN, "training_rule": STRICT, "windows": W10, "candidates": "orbit-free no_residual, the screen script's screen refitted on the training sessions under the training label (floor rule); orbit-free indicator", "cells": "15 per model (3 receivers x 5 LOSO folds)", "seeds": "MI 20260914; forest 20260914", "permutations": "", "outer": True, "nested": True, "outputs": "logistic_regression/logistic_regression_*_primary, random_forest/random_forest_*_primary, final/p_nested_*, importance (random_forest/random_forest_importance_primary, random_forest_importance_logreg)"},  # dict literal: the adopted primary
    {"run_id": "E6", "command": "p", "models": "(model-free)", "label_set": FIN, "training_rule": STRICT, "windows": W10, "candidates": "P's prepared candidates, families sos, sosdiff, tk", "cells": "15", "seeds": "", "permutations": "", "outer": True, "nested": False, "outputs": "final/e6_range.csv, e6_failures.csv"},  # dict literal: E6
    {"run_id": "INNER_THRESHOLD", "command": "p", "models": "lr+rf", "label_set": FIN, "training_rule": STRICT, "windows": W10, "candidates": "P's", "cells": "15 per model", "seeds": "", "permutations": "", "outer": True, "nested": "reads P's", "outputs": "final/k5_*"},  # dict literal: the inner-fold threshold
    {"run_id": "C4A", "command": "controls", "models": "lr+rf", "label_set": FIN, "training_rule": STRICT, "windows": W10, "candidates": "P's (true-label screen and preparation kept)", "cells": "2.6.2, 2.6.4 x 3 receivers", "seeds": "20260914 + k", "permutations": "100 per cell and model", "outer": True, "nested": False, "outputs": "logistic_regression/logistic_regression_c4a_permutation.csv, random_forest/random_forest_c4a_permutation.csv, final/c4a_summary.csv"},  # dict literal: C4a
    {"run_id": "C5", "command": "controls", "models": "lr+rf", "label_set": FIN, "training_rule": STRICT, "windows": "features_final/w10", "candidates": "the regression script's 370-column geometry block, no screen, no indicator", "cells": "15 per model", "seeds": "20260914", "permutations": "", "outer": True, "nested": False, "outputs": "logistic_regression/logistic_regression_*_c5_geometry, random_forest/random_forest_*_c5_geometry"},  # dict literal: C5
    {"run_id": "C6", "command": "controls", "models": "lr (MI seeds); rf (forest seeds, MI seeds)", "label_set": FIN, "training_rule": STRICT, "windows": W10, "candidates": "P's specification", "cells": "15 per model and seed", "seeds": "20260914-20260918", "permutations": "", "outer": True, "nested": False, "outputs": "logistic_regression/logistic_regression_c6_*, random_forest/random_forest_c6_*, final/c6_pooled.csv"},  # dict literal: C6
    {"run_id": "C4BR_P", "command": "c4b", "models": "lr+rf", "label_set": FIN, "training_rule": STRICT, "windows": W10, "candidates": "screen, indicator, preparation and MI refitted on the permuted labels", "cells": "2.6.2, 2.6.4 x 3 receivers; 2.6.10 u-blox 01", "seeds": "20260914 + 1000 + k", "permutations": "100 per cell and model", "outer": True, "nested": False, "outputs": "logistic_regression/logistic_regression_c4b_within_session.csv, random_forest/random_forest_c4b_within_session.csv, final/r3_verdicts.csv, random_forest/random_forest_importance_c4b*.csv, random_forest/random_forest_c4b_classification*.csv"},  # dict literal: C4b-refit, the session control, classification
    {"run_id": "P_noStave", "command": "sens", "models": "lr+rf", "label_set": FIN + " without the Stave windows", "training_rule": STRICT, "windows": W10, "candidates": "P's specification", "cells": "15 per model", "seeds": "20260914", "permutations": "", "outer": True, "nested": False, "outputs": "final/p_nostave_*"},  # dict literal: P-noStave
    {"run_id": "P_unrounded", "command": "sens", "models": "lr+rf", "label_set": FIN, "training_rule": STRICT, "windows": "orbit-free windows rebuilt in memory, unrounded (the feature script's functions; Stave filled as the window-feature script does)", "candidates": "P's specification", "cells": "15 per model", "seeds": "20260914", "permutations": "", "outer": True, "nested": False, "outputs": "final/p_unrounded_*"},  # dict literal: P-unrounded
    {"run_id": "P_nocompress", "command": "sens", "models": "rf", "label_set": FIN, "training_rule": STRICT, "windows": W10, "candidates": "P's specification, compress=False (E3)", "cells": "15", "seeds": "20260914", "permutations": "", "outer": True, "nested": False, "outputs": "final/p_nocompress_*"},  # dict literal: P-nocompress
    {"run_id": "VARIANTS", "command": "variants", "models": "lr+rf (C0.01/C0.1/C10 lr only; leaf1/leaf20 rf only)", "label_set": FIN + " (V1, V2 without the Stave windows)", "training_rule": STRICT, "windows": "features_final/w1-w10", "candidates": "one change from P each", "cells": "15 per model and variant", "seeds": "20260914", "permutations": "", "outer": True, "nested": False, "outputs": "logistic_regression/logistic_regression_variants_*, random_forest/random_forest_variants_*"},  # dict literal: the variants
    {"run_id": "PROBE", "command": "probe", "models": "depth-4 tree", "label_set": "sessions only", "training_rule": "", "windows": "features/w10 and features_final/w10, 2.6.12 dropped at read", "candidates": "the screen script's seven sets", "cells": "3 receivers x 7 sets x 2 window sets", "seeds": "20260914", "permutations": "", "outer": False, "nested": False, "outputs": "final/probe_*"},  # dict literal: the probe
    {"run_id": "REPORT", "command": "report", "models": "lr+rf", "label_set": FIN, "training_rule": STRICT, "windows": "", "candidates": "", "cells": "", "seeds": "", "permutations": "", "outer": "", "nested": "", "outputs": "final/registered_vs_adopted.csv, final/per_fold_*.csv, final/nested_*.csv, final/r3_all.csv, final/class_counts.csv, figures"},  # dict literal: the report
]
# list of tuples: the variants (name, family, models, specification change, window length, without the Stave windows)
VARIANTS = [
    ("V1_main", "feature set", MODELS, {"set": "main", "blocks": ("orbitfree", "self")}, 10, True),  # the main set (orbit-free + self), two indicators
    ("V2_known", "feature set", MODELS, {"set": "known_antenna", "known": True}, 10, True),  # the known-antenna set, orbit-free and known indicators
    ("V3_shift_invariant", "feature set", MODELS, {"change_only": True, "blocks": ()}, 10, False),  # std, skew, kurt only, no indicator
    ("V4_no_selection", "feature set", MODELS, {"select": "none"}, 10, False),  # no Pearson pruning, no MI
    ("screen_no_floor", "feature set", MODELS, {"rule": "reason_no_floor"}, 10, False),  # the no-floor screen
    ("unscreened", "feature set", MODELS, {"screen": "none"}, 10, False),  # every orbit-free column
    ("plus_geometry_levels", "feature set", MODELS, {"add": "geometry_levels"}, 10, False),  # plus the orbit-free geometry level statistics
    ("w1", "window", MODELS, {"w": 1}, 1, False),  # w = 1
    ("w2", "window", MODELS, {"w": 2}, 2, False),  # w = 2
    ("w3", "window", MODELS, {"w": 3}, 3, False),  # w = 3
    ("w5", "window", MODELS, {"w": 5}, 5, False),  # w = 5
    ("C0.01", "regulariser", ("lr",), {"C": 0.01}, 10, False),  # C = 0.01
    ("C0.1", "regulariser", ("lr",), {"C": 0.1}, 10, False),  # C = 0.1
    ("C10", "regulariser", ("lr",), {"C": 10.0}, 10, False),  # C = 10
    ("leaf1", "leaf size", ("rf",), {"leaf": 1}, 10, False),  # min_samples_leaf 1
    ("leaf20", "leaf size", ("rf",), {"leaf": 20}, 10, False),  # min_samples_leaf 20
]


def cmd_table() -> None:  # function: the run table
    v = pd.DataFrame([{"run_id": f"VAR_{n}", "command": "variants", "models": "+".join(m), "label_set": FIN + (" without the Stave windows (flagged)" if ns else ""), "training_rule": STRICT, "windows": f"features_final/w{w}", "candidates": repr(d), "cells": "15 per model", "seeds": "20260914", "permutations": "", "outer": True, "nested": False, "outputs": "logistic_regression/logistic_regression_variants_*, random_forest/random_forest_variants_*"} for n, fam, m, d, w, ns in VARIANTS])  # list comprehension: one row per variant
    B.write_csv(pd.concat([pd.DataFrame(RUNS), v], ignore_index=True), PROC / "baselines_controls" / "FINAL_RUN_TABLE.csv")  # written
    print(f"run table: {len(RUNS) + len(v)} rows")  # summary


# ----------------------------------------------------------------------------------------------------------------------
# cells
# ----------------------------------------------------------------------------------------------------------------------
def k2s_registered(cells: dict) -> dict:  # function: the design-choice study's STRICT cells on the registered windows (quiet negatives of choices_lib.quiet_sets)
    _, strict, _ = L.quiet_sets()  # strict quiet seconds
    return L.finish(L.with_quiet(cells, strict))  # training label "clean" for them; tlabel set everywhere


def cells_for(w: int) -> dict:  # function: the cells of the adopted runs (final set) or of the test mode (the design-choice study's STRICT)
    if SMOKE:  # test mode
        return k2s_registered(B.load_cells(B.REGISTERED, w, False))  # registered windows with strict negatives
    return B.load_cells(B.FINAL, w, True)  # final windows with the training label


def labels_in_use() -> None:  # function: the scoring module points at the final set (registered in the test mode)
    B.use_labels("registered" if SMOKE else "final")  # the scoring module's three files


def save_model_tables(out: dict, model: str, folder: Path, prefix: str, suffix: str, summ: pd.DataFrame) -> None:  # function: one model's tables under the logistic regression and the random forest' file names
    m, lt, pr = out["metrics"], out["latency"], out["predictions"]  # frames
    B.write_gz(pr[pr.model == model], folder / f"{prefix}_predictions_{suffix}.csv.gz")  # held-out probabilities
    B.write_csv(m[m.model == model], folder / f"{prefix}_metrics_{suffix}.csv")  # per-fold figures
    B.write_csv(lt[lt.model == model], folder / f"{prefix}_latency_{suffix}.csv")  # latency rows
    B.write_csv(summ[summ.model == model], folder / f"{prefix}_pooled_{suffix}.csv")  # pooled figures (choices_lib.summarise)
    B.write_csv(out["selection"], folder / f"{prefix}_selection_{suffix}.csv")  # what each fit selected (shared by both models)
    B.write_csv(out["fails"][out["fails"].model == model], folder / f"{prefix}_fit_failures_{suffix}.csv")  # failed fits


def same_preds(a: pd.DataFrame, b: pd.DataFrame, keys: list) -> bool:  # function: same windows, labels and bitwise probabilities
    x = a[keys + ["label", "p"]].merge(b[keys + ["label", "p"]], on=keys, how="inner", suffixes=("", "_c"), validate="one_to_one")  # aligned
    return len(x) == len(a) == len(b) and bool((x.label == x.label_c).all()) and bool(np.array_equal(x.p.to_numpy(), x.p_c.to_numpy(), equal_nan=True))  # every window matched and equal


def true_auroc(metrics: pd.DataFrame) -> dict:  # function: (model, receiver, fold) -> held-out AUROC
    return {(r.model, r.receiver, r.held_out): r.auroc for r in metrics.itertuples()}  # dict comprehension


# ----------------------------------------------------------------------------------------------------------------------
# gates
# ----------------------------------------------------------------------------------------------------------------------
def cmd_gates() -> None:  # function: R0's gates on registered and the design study material
    t0 = time.time()  # clock
    B.use_labels("registered")  # the scoring module on the registered set
    wc, wc10 = B.wc_all(), L.window_dict()  # dictionaries
    wl, fl = M.load_meta()  # the regression script's dictionary and the registered lists
    k = ["receiver", "held_out", "sec"]  # prediction keys
    reg = {w: B.load_cells(B.REGISTERED, w, False) for w in (10, 5)}  # dict comprehension: registered windows at w = 10 and 5
    k0 = L.finish(B.d1_cells(reg[10], wc10))  # the registered labels' cells (orbit-free columns, tlabel = label)
    spec_reg = {**B.P_SPEC, "screen": "registered", "rule": "screened"}  # P's specification with the registered lists
    o = B.outer_v(L.finish(reg[10]), wc, spec_reg, MODELS, "REGISTERED", fl=fl, keep=True)  # outer_v on the registered labels
    rf_primary = B.read_exact(PROC / "random_forest_registered" / "random_forest_predictions_primary.csv.gz")  # the random forest's forest
    lr_variants = B.read_exact(PROC / "logistic_regression_registered" / "logistic_regression_variants_predictions.csv.gz")  # the logistic regression's variants
    pr = o["predictions"]  # outer_v's predictions
    check("G1 outer_v: forest = random_forest_predictions_primary, logistic regression = the logistic regression no_residual (bitwise); no failed fit", same_preds(pr[pr.model == "rf"], rf_primary, k) and same_preds(pr[pr.model == "lr"], lr_variants[lr_variants.variant == "no_residual"], k) and o["fails"].empty, f"{len(pr)} predictions")  # G1
    ks = k2s_registered(k0)  # the design-choice study's STRICT cells
    a = B.run_choices(ks, wc10, "STRICT", keep_all=True)  # choices_lib.outer on them
    ac = B.read_exact(AA3 / "runs" / "STRICT" / "outer_predictions.csv.gz")  # the design-choice study's committed STRICT predictions
    km = ["model", "receiver", "held_out", "sec"]  # keys with the model
    ov = B.outer_v(k2s_registered(L.finish(reg[10])), wc, B.P_SPEC, MODELS, "STRICT")  # outer_v with P's specification on the same cells
    check("G2 choices_lib.outer and outer_v(P's specification) on the design-choice study's STRICT cells = runs/STRICT/outer_predictions (bitwise)", same_preds(a["predictions"], ac, km) and same_preds(ov["predictions"], ac, km), f"{len(ac)} predictions")  # G2
    n = B.run_nested(ks, wc10, "STRICT")  # choices_lib.nested
    an = B.read_exact(AA3 / "runs" / "STRICT" / "nested_predictions.csv.gz")  # the design-choice study's committed nested predictions
    check("G3 choices_lib.nested on the design-choice study's STRICT cells = runs/STRICT/nested_predictions (bitwise)", same_preds(n["predictions"], an, km + ["outer_held_out"]), f"{len(an)} predictions")  # G3
    a4 = B.c4a(o["kept"], ("rf",), "C4A_REGISTERED", range(3))  # C4a, three permutations
    r4 = B.read_exact(PROC / "random_forest_registered" / "random_forest_c4a_permutation.csv")  # the random forest's C4a
    x = a4.merge(r4[r4.perm < 3][["receiver", "held_out", "perm", "auroc"]], on=["receiver", "held_out", "perm"], suffixes=("", "_r"))  # aligned
    check("G4 c4a() on the registered labels = the random forest's C4a, permutations 0-2 of the six cells (AUROC exact)", len(x) == 18 and bool((x.auroc == x.auroc_r).all()), f"{len(x)} rows")  # G4
    b4 = B.c4b_refit(a["kept"], wc10, ("rf",), [(rx, h) for rx in B.RXS for h in ("2.6.2", "2.6.4")], "C4BR_STRICT", range(3))  # C4b-refit, three permutations
    rb = B.read_exact(AA3 / "runs" / "C4BR_STRICT" / "c4b_runs.csv")  # the design-choice study's committed runs
    cols = ["auroc", "recall", "clean_in_alarm_share", "n_selected", "n_candidates", "candidates_equal_true", "selected_columns"]  # compared columns
    x = b4.merge(rb[rb.perm < 3][["receiver", "held_out", "perm"] + cols], on=["receiver", "held_out", "perm"], suffixes=("", "_r"))  # aligned
    check("G5 c4b_refit() on the design-choice study's STRICT fits = C4BR_STRICT/c4b_runs.csv, permutations 0-2 of the six cells (every compared column)", len(x) == 18 and all((x[c].astype(str) == x[c + "_r"].astype(str)).all() for c in cols), f"{len(x)} rows")  # G5 (text comparison: floats as their repr)
    order = [(rx, h) for rx in B.RXS for h, _ in S.folds()]  # the forest script's order: receiver, then fold
    for mdl, cfg_, files in (("rf", "primary", ("random_forest_importance_primary.csv", "random_forest_importance_repeats_primary.csv.gz", "random_forest_imputed_share.csv")), ("lr", "logreg_no_residual", ("random_forest_importance_logreg.csv", "random_forest_importance_repeats_logreg.csv.gz", None))):  # for loop over the two models
        ip, rp, sh = B.importance(o["kept"], wc10, mdl, cfg_, order)  # the forest script's importance on the registered labels' fits
        txt = lambda d: d.to_csv(index=False, lineterminator="\n")  # lambda: a table's CSV text, as the regression script's writers write it
        rf_ = lambda f: (gzip.decompress((PROC / "random_forest_registered" / f).read_bytes()) if f.endswith(".gz") else (PROC / "random_forest_registered" / f).read_bytes()).decode("utf-8")  # lambda with a conditional expression: a committed table's text as written
        ok = txt(ip) == rf_(files[0]) and txt(rp) == rf_(files[1]) and (files[2] is None or txt(sh) == rf_(files[2]))  # all three as written text
        check(f"G6 importance() on the registered labels ({mdl}) = {', '.join(f for f in files if f)} (as written text)", ok, f"{len(ip)} rows")  # G6
    ipc, ibc = B.read_exact(PROC / "random_forest_registered" / "random_forest_importance_primary.csv"), B.read_exact(PROC / "random_forest_registered" / "random_forest_importance_c4b.csv")  # the random forest's committed importance
    b4r = B.read_exact(PROC / "random_forest_registered" / "random_forest_c4b_within_session.csv").assign(model="rf", failed=False)  # the random forest's C4b (all completed)
    med = {c: R.c4b_median_perm(b4r, *c) for c in CELLS_C4B}  # dict comprehension: the forest script's median permutations
    mr = B.read_exact(PROC / "random_forest_registered" / "random_forest_metrics_primary.csv").assign(model="rf")  # the random forest's metrics
    cl = B.classification(ipc, ibc, true_auroc(mr), b4r, med, "rf", CELLS_C4B)  # mine
    rc = B.read_exact(PROC / "random_forest_registered" / "random_forest_c4b_classification.csv")  # the random forest's
    cc = [c for c in rc.columns]  # the random forest's columns
    check("G7 classification() on the random forest's committed tables = random_forest_c4b_classification.csv (every column)", cl[cc].astype(str).equals(rc.astype(str)), f"{len(rc)} rows")  # G7
    rf_variants = B.read_exact(PROC / "random_forest_registered" / "random_forest_variants_predictions.csv.gz")  # the random forest's variants
    for name, delta, w, fl_set in (("V1_main", {"set": "main", "blocks": M.BLOCKS, "rule": "screened"}, 10, None), ("V2_known", {"set": "known_antenna", "known": True, "rule": "screened"}, 10, None), ("V3_shift_invariant", {"change_only": True, "blocks": (), "rule": "screened"}, 10, None),  # tuples: registered variants
                                   ("V4_no_selection", {"select": "none", "rule": "screened"}, 10, None), ("w5", {"w": 5, "rule": "screened"}, 5, None), ("leaf20", {"leaf": 20, "rule": "screened"}, 10, None)):  # continued
        ov = B.outer_v(L.finish(reg[w]), wc, {**spec_reg, **delta}, ("rf",), name, fl=fl)  # outer_v with the registered lists
        check(f"G8 outer_v = the random forest {name} (bitwise)", same_preds(ov["predictions"], rf_variants[rf_variants.variant == name], k), f"{len(ov['predictions'])} predictions")  # G8
    lr_c01 = lr_variants[lr_variants.variant == "C0.1"]  # the logistic regression's C = 0.1 (main set)
    ov = B.outer_v(L.finish(reg[10]), wc, {**spec_reg, "set": "main", "blocks": M.BLOCKS, "C": 0.1}, ("lr",), "C0.1", fl=fl)  # outer_v with C = 0.1 on the logistic regression's main primary
    check("G8 outer_v = the logistic regression C0.1 (bitwise)", same_preds(ov["predictions"], lr_c01, k), f"{len(ov['predictions'])} predictions")  # G8 (logistic regression)
    pv = rf_variants[rf_variants.variant == "persist_3of5"]  # the random forest's 3-of-5 alarms
    al = np.concatenate([C9.k_of_n(d.sec.to_numpy(np.int64), d.alarm.to_numpy()) for _, d in rf_primary.groupby(["receiver", "held_out"], sort=False)])  # list comprehension: the sequential-baseline script's rule on the random forest's primary, fold by fold
    check("G8 the sequential-baseline script's k_of_n on random_forest_predictions_primary = the random forest persist_3of5 alarms", bool(np.array_equal(al, pv.alarm.to_numpy().astype(bool))), f"{len(al)} windows")  # G8 (alarm rule)
    for mdl, ref in (("rf", B.read_exact(PROC / "random_forest_registered" / "random_forest_predictions_c5_geometry.csv.gz")), ("lr", B.read_exact(PROC / "logistic_regression_registered" / "logistic_regression_predictions_c5_geometry.csv.gz"))):  # for loop over the two registered C5 runs
        ov = B.outer_v(L.finish(reg[10]), wc, {**B.P_SPEC, "set": "geometry", "screen": "none", "blocks": ()}, (mdl,), "C5")  # outer_v's C5
        check(f"G9 outer_v C5 ({mdl}) = registered C5 predictions (bitwise)", same_preds(ov["predictions"], ref, k), f"{len(ov['predictions'])} predictions")  # G9
    r6 = B.read_exact(PROC / "random_forest_registered" / "random_forest_c6_seeds.csv")  # the random forest's C6
    ov = B.outer_v(L.finish(reg[10]), wc, {**spec_reg, "forest_seed": B.SEED + 1}, ("rf",), "C6", fl=fl)  # forest seed 20260915 (registered lists)
    x = ov["metrics"].merge(r6[(r6.forest_seed == B.SEED + 1)], on=["receiver", "held_out"], suffixes=("", "_r"))  # aligned
    check("G9 outer_v C6 (forest seed 20260915) = random_forest_c6_seeds (recall, AUROC, clean share)", len(x) == 15 and all(np.allclose(x[c].astype(float), x[c + "_r"].astype(float), rtol=0, atol=0, equal_nan=True) for c in ("recall", "auroc", "clean_in_alarm_share")), f"{len(x)} folds")  # G9
    B.write_csv(pd.DataFrame(CHECKS), PROC / "baselines_controls" / "gates.csv")  # the gates
    TIMING.mkdir(parents=True, exist_ok=True)  # folder
    (TIMING / "gates.csv").write_text(f"gates,gates,{time.time() - t0:.1f}\n")  # run time
    if any(c["result"] == "FAIL" for c in CHECKS):  # if: a gate failed
        sys.exit("STOP: a gate failed")  # exit 1
    print("all gates pass")  # done


# ----------------------------------------------------------------------------------------------------------------------
# P: outer, nested, the inner-fold threshold, E6, importance
# ----------------------------------------------------------------------------------------------------------------------
def run_p(wc10: pd.DataFrame, keep_all: bool) -> tuple:  # function: the cells and P's outer run (deterministic; rebuilt by every command that needs the fits)
    cells = cells_for(10)  # the windows
    return cells, B.run_choices(B.d1_cells(cells, wc10), wc10, "P", keep_all)  # choices_lib.outer on the orbit-free columns


def cmd_p() -> None:  # function: P outer and nested, the outer_v identity, the inner-fold threshold, E6, importance, figures
    t0 = time.time()  # clock
    labels_in_use()  # the scoring module's label set
    wc, wc10 = B.wc_all(), L.window_dict()  # dictionaries
    cells, P = run_p(wc10, True)  # P's outer run with every fit kept
    check("P outer: failed fits", P["fails"].empty, f"{len(P['fails'])} failed fits")  # no fit may fail in P
    summ = B.summary(P, MODELS)  # pooled figures
    for mdl, folder, prefix in (("lr", LR_DIR, "logistic_regression"), ("rf", RF_DIR, "random_forest")):  # for loop over the two models
        save_model_tables(P, mdl, folder, prefix, "primary", summ)  # their tables
    ov = B.outer_v(cells, wc, B.P_SPEC, MODELS, "P")  # outer_v with P's specification
    km = ["model", "receiver", "held_out", "sec"]  # keys with the model
    check("outer_v(P's specification) = choices_lib.outer's P (bitwise)", same_preds(ov["predictions"], P["predictions"], km), f"{len(P['predictions'])} predictions")  # the generalised loop is P's loop
    B.write_csv(ov["selection"], FINAL_DIR / "adopted_selection_kept_columns.csv")  # P's Pearson-kept and selected columns (for P-nocompress)
    B.write_csv(P["metrics"][["model", "receiver", "held_out", "auroc"]], FINAL_DIR / "adopted_fold_auroc.csv")  # the held-out AUROC per model and fold (the session control's true AUROC)
    n = B.run_nested(B.d1_cells(cells, wc10), wc10, "P")  # nested scores
    B.write_csv(n["metrics"], FINAL_DIR / "adopted_nested_metrics.csv")  # per inner fold
    B.write_csv(n["latency"], FINAL_DIR / "adopted_nested_latency.csv")  # latency
    B.write_gz(n["predictions"], FINAL_DIR / "adopted_nested_predictions.csv.gz")  # inner out-of-fold predictions
    B.write_csv(n["selection"], FINAL_DIR / "adopted_nested_selection.csv")  # selections
    B.write_csv(n["fails"], FINAL_DIR / "adopted_nested_fit_failures.csv")  # unfittable inner folds
    B.write_csv(n["summary"], FINAL_DIR / "adopted_nested_summary.csv")  # per model, receiver and outer fold
    k5 = B.k5(n["predictions"], P, cells)  # the inner-fold threshold
    th = k5["thresholds"]  # thresholds
    B.write_csv(th, FINAL_DIR / "inner_threshold_thresholds.csv")  # written
    if len(k5["metrics"]):  # if: any fold scored at a threshold
        B.write_csv(k5["metrics"], FINAL_DIR / "inner_threshold_metrics.csv")  # per fold
        B.write_csv(B.summary({"metrics": k5["metrics"], "latency": k5["latency"], "predictions": k5["predictions"]}, tuple(sorted(set(k5["metrics"].model)))), FINAL_DIR / "inner_threshold_summary.csv")  # pooled
    clean_folds = P["metrics"][P["metrics"].clean > 0][["model", "receiver", "held_out"]]  # folds with held-out clean windows
    x = clean_folds.merge(th.rename(columns={"outer_held_out": "held_out"}), on=["model", "receiver", "held_out"], how="left")  # their thresholds
    inf = x[x.receiver.isin(["SBF_02", "SBF_03"]) & ~np.isfinite(x.threshold.astype(float))]  # Septentrio folds with clean windows and no threshold
    B.write_csv(pd.DataFrame([{"k5": "INFEASIBLE: dropped for good" if len(inf) else "feasible", "septentrio_folds_without_threshold": "; ".join(f"{r.model} {r.receiver} {r.held_out}" for r in inf.itertuples())}]), FINAL_DIR / "inner_threshold_decision.csv")  # the Stave-stop decision
    check("the inner-fold threshold", None, "INFEASIBLE" if len(inf) else "feasible")  # reported
    e6, bad = B.e6(P["kept"], wc10)  # E6
    B.write_csv(e6, FINAL_DIR / "e6_range.csv")  # the table
    B.write_csv(bad, FINAL_DIR / "e6_failures.csv")  # failing columns
    sk = bad[bad.family == "skipped compression"]  # columns that skipped the compression (any prepared column)
    ov_sel = bad[(bad.family != "skipped compression") & bad.selected.astype(bool)]  # MI-selected columns beyond 6.9
    check("extrapolation check (the logistic regression with its compression): no prepared column skipped the compression; no MI-selected column's held-out value beyond its training maximum by more than 6.9 after compression", sk.empty and ov_sel.empty, f"skipped {len(sk)}; selected columns over 6.9: {len(ov_sel)}")  # E6's stop
    check("E6 candidate level (reported in full, e6_failures.csv)", None, f"{len(bad) - len(sk)} column-fit cases over 6.9 in {bad[['receiver', 'held_out']].drop_duplicates().shape[0]} fits; largest excess {e6.largest_heldout_excess_post.max():.3f}")  # reported
    order = [(rx, h) for rx in B.RXS for h, _ in S.folds()]  # the forest script's order
    for mdl, cfg_, names in (("rf", "primary", ("random_forest_importance_primary.csv", "random_forest_importance_repeats_primary.csv.gz", "random_forest_imputed_share.csv")), ("lr", "logreg_no_residual", ("random_forest_importance_logreg.csv", "random_forest_importance_repeats_logreg.csv.gz", "random_forest_imputed_share_logreg.csv"))):  # the two models, the random forest's file names
        ip, rp, sh = B.importance(P["kept"], wc10, mdl, cfg_, order)  # the forest script's importance
        B.write_csv(ip, RF_DIR / names[0])  # summary
        B.write_gz(rp, RF_DIR / names[1])  # repeats
        B.write_csv(sh, RF_DIR / names[2])  # imputed shares
    pr = P["predictions"]  # predictions
    with B.figures_to(GRAPH, "_b", M, R):  # figures renamed with "_b"
        M.plot_heldout_probability(pr[pr.model == "lr"])  # the figure: the logistic regression's held-out probabilities
        R.plot_forest_heldout_probability(pr[pr.model == "rf"])  # the figure: the forest's
        ipr = B.read_exact(RF_DIR / "random_forest_importance_primary.csv")  # the forest's importance as written
        R.plot_importance_per_class(ipr)  # the figure: recall drop per family and class
        R.plot_imputation_diagnostic(ipr)  # the figure: the imputation diagnostic
    done("p", t0, stop_if_failed=True)  # checks; E6 or a failed fit stops


# ----------------------------------------------------------------------------------------------------------------------
# controls: C4a, C5, C6
# ----------------------------------------------------------------------------------------------------------------------
def cell_medians(a4: pd.DataFrame) -> pd.DataFrame:  # function: per model and cell, failed fits and the median AUROC (not assessable above 5 failures)
    g = a4.groupby(["model", "receiver", "held_out"], sort=False)  # cells
    out = g.agg(perms=("perm", "size"), failed=("failed", "sum")).reset_index()  # counts
    out["auroc_median"] = [np.nan if f > B.FAIL_MAX else float(d[~d.failed.astype(bool)].auroc.median()) for (_, d), f in zip(g, out.failed)]  # list comprehension: median over completed fits, or none
    out["auroc_p5"] = [np.nan if f > B.FAIL_MAX else float(d[~d.failed.astype(bool)].auroc.quantile(0.05)) for (_, d), f in zip(g, out.failed)]  # 5th percentile
    out["auroc_p95"] = [np.nan if f > B.FAIL_MAX else float(d[~d.failed.astype(bool)].auroc.quantile(0.95)) for (_, d), f in zip(g, out.failed)]  # 95th percentile
    out["status"] = np.where(out.failed > B.FAIL_MAX, "not assessable", np.where(out.auroc_median.between(*B.C4A_BAND), "inside [0.4, 0.6]", "OUTSIDE [0.4, 0.6]"))  # nested np.where: the cell's status
    return out  # the table


def seed_rows(runs: list, base_sel: pd.DataFrame) -> pd.DataFrame:  # function: C6's per-run rows (recall, AUROC, clean share, Jaccard with the reported model's selection)
    rows = []  # results
    base = {(r.receivers, r.held_out): set(r.selected_columns.split(";")) for r in base_sel.itertuples()}  # dict comprehension: the reported model's selections
    for mdl, mi, fs, varied, out in runs:  # for loop over the runs
        sel = {(r.receivers, r.held_out): set(r.selected_columns.split(";")) for r in out["selection"].itertuples()}  # dict comprehension: this run's selections
        for r in out["metrics"][out["metrics"].model == mdl].itertuples():  # for loop over its folds
            a, b = sel[(r.receiver, r.held_out)], base[(r.receiver, r.held_out)]  # the two sets
            rows.append({"model": mdl, "receiver": r.receiver, "held_out": r.held_out, "mi_seed": mi, "forest_seed": fs, "varied": varied, "recall": r.recall, "auroc": r.auroc, "clean_in_alarm_share": r.clean_in_alarm_share, "jaccard_with_primary": len(a & b) / len(a | b)})  # dict literal: one row
    return pd.DataFrame(rows)  # the table


def cmd_controls() -> None:  # function: C4a, C5, C6
    t0 = time.time()  # clock
    labels_in_use()  # the scoring module's label set
    wc, wc10 = B.wc_all(), L.window_dict()  # dictionaries
    cells, P = run_p(wc10, True)  # P with every fit
    a4 = B.c4a(P["kept"], MODELS, "C4A", PERMS)  # C4a
    B.write_csv(a4[a4.model == "lr"], LR_DIR / "logistic_regression_c4a_permutation.csv")  # the logistic regression's
    B.write_csv(a4[a4.model == "rf"], RF_DIR / "random_forest_c4a_permutation.csv")  # the forest's
    cm = cell_medians(a4)  # medians
    B.write_csv(cm, FINAL_DIR / "c4a_summary.csv")  # written
    ok4 = bool((cm.status != "OUTSIDE [0.4, 0.6]").all())  # every assessable cell inside the band
    check("C4a: median held-out AUROC inside [0.4, 0.6] in every assessable cell, both models", ok4 or SMOKE, "; ".join(f"{r.model} {r.receiver} {r.held_out}: {r.auroc_median:.3f} ({int(r.failed)} failed)" for r in cm.itertuples()))  # C4a (the test mode's two permutations are not judged)
    with B.figures_to(GRAPH, "_b", M, R):  # figures
        M.plot_permutation(a4[(a4.model == "lr") & ~a4.failed.astype(bool)])  # the figure: the logistic regression's C4a
        R.plot_forest_permutation(a4[(a4.model == "rf") & ~a4.failed.astype(bool)])  # the figure: the forest's
    c5 = B.outer_v(cells, wc, {**B.P_SPEC, "set": "geometry", "screen": "none", "blocks": ()}, MODELS, "C5")  # C5: the geometry block alone
    s5 = B.summary(c5, MODELS)  # pooled
    for mdl, folder, prefix in (("lr", LR_DIR, "logistic_regression"), ("rf", RF_DIR, "random_forest")):  # the two models
        save_model_tables(c5, mdl, folder, prefix, "c5_geometry", s5)  # their tables
    runs = [("lr", B.SEED, B.SEED, "primary", P), ("rf", B.SEED, B.SEED, "primary", P)]  # C6: the reported models first
    for s in MI_SEEDS[1:]:  # MI seeds
        runs.append(("lr", s, B.SEED, "mi_seed", B.outer_v(cells, wc, {**B.P_SPEC, "mi_seed": s}, ("lr",), "C6")))  # the logistic regression
        runs.append(("rf", s, B.SEED, "mi_seed", B.outer_v(cells, wc, {**B.P_SPEC, "mi_seed": s}, ("rf",), "C6")))  # the forest
    for s in FOREST_SEEDS[1:]:  # forest seeds
        runs.append(("rf", B.SEED, s, "forest_seed", B.outer_v(cells, wc, {**B.P_SPEC, "forest_seed": s}, ("rf",), "C6")))  # the forest
    sr = seed_rows(runs, P["selection"])  # per-run rows
    B.write_csv(sr[sr.model == "lr"].drop(columns="forest_seed"), LR_DIR / "logistic_regression_c6_mi_seeds.csv")  # the logistic regression's
    B.write_csv(sr[sr.model == "rf"], RF_DIR / "random_forest_c6_seeds.csv")  # the forest's
    disp = []  # dispersion rows
    for (mdl, v), d in [(k_, g) for k_, g in sr.groupby(["model", "varied"]) if k_[1] != "primary"]:  # list comprehension: the varied seeds of each model
        d = pd.concat([sr[(sr.model == mdl) & (sr.varied == "primary")], d])  # with the reported model
        g = d.groupby(["receiver", "held_out"]).agg(recall_min=("recall", "min"), recall_max=("recall", "max"), auroc_min=("auroc", "min"), auroc_max=("auroc", "max"), clean_alarm_min=("clean_in_alarm_share", "min"), clean_alarm_max=("clean_in_alarm_share", "max"), jaccard_min=("jaccard_with_primary", "min")).reset_index()  # named aggregation: range over the five seeds
        disp.append(g.assign(model=mdl, seed_varied=v))  # kept
    dp = pd.concat(disp, ignore_index=True)  # one table
    B.write_csv(dp[dp.model == "lr"], LR_DIR / "logistic_regression_c6_dispersion.csv")  # the logistic regression's
    B.write_csv(dp[dp.model == "rf"], RF_DIR / "random_forest_c6_dispersion.csv")  # the forest's
    pooled = pd.concat([B.summary(o, (mdl,)).assign(mi_seed=mi, forest_seed=fs, varied=v) for mdl, mi, fs, v, o in runs], ignore_index=True)  # list comprehension: pooled figures per seed
    B.write_csv(pooled, FINAL_DIR / "c6_pooled.csv")  # written
    done("controls", t0, stop_if_failed=not SMOKE)  # a C4a band failure stops


# ----------------------------------------------------------------------------------------------------------------------
# C4b-refit, the session control, importance of the median permuted models, classification
# ----------------------------------------------------------------------------------------------------------------------
def cmd_c4b() -> None:  # function: C4b-refit for both models, session-control verdicts, classification with the selection check
    t0 = time.time()  # clock
    labels_in_use()  # the scoring module's label set
    wc10 = L.window_dict()  # dictionary
    cells, P = run_p(wc10, True)  # P with every fit
    b4 = B.c4b_refit(P["kept"], wc10, MODELS, CELLS_C4B, "C4BR_P", PERMS)  # every permutation, both models
    B.write_csv(b4[b4.model == "lr"], LR_DIR / "logistic_regression_c4b_within_session.csv")  # the logistic regression's
    B.write_csv(b4[b4.model == "rf"], RF_DIR / "random_forest_c4b_within_session.csv")  # the forest's
    ta = true_auroc(P["metrics"])  # the adopted models' held-out AUROC
    v = B.cell_verdicts(b4, ta, CELLS_C4B, "C4BR_P")  # the session control
    B.write_csv(v, FINAL_DIR / "r3_verdicts.csv")  # written
    check("session control (adopted, both models)", None, "; ".join(f"{r.model} {r.receiver} {r.held_out}: {r.verdict}" for r in v.itertuples()))  # reported
    fam, xs = R.name_maps(wc10)  # the forest script's maps
    for mdl, tag, true_file in (("rf", "", "random_forest_importance_primary.csv"), ("lr", "_logreg", "random_forest_importance_logreg.csv")):  # the two models
        med, ib, ibr = {}, [], []  # median permutations; importance rows; repeats
        for rx, held in CELLS_C4B:  # cells
            vv = v[(v.model == mdl) & (v.receiver == rx) & (v.held_out == held)].verdict.iloc[0]  # the cell's verdict
            if vv == "not assessable":  # if: no median model
                continue  # not classified
            k = B.median_perm(b4, mdl, rx, held)  # the median permutation
            med[(rx, held)] = k  # kept
            kp = P["kept"][(rx, held)]  # P's fit
            f = B.c4b_prep(kp, wc10, k)  # the permuted preparation again (deterministic)
            m, p, fail = B.c4b_model(kp, f, mdl, "C4BR_P", held, rx, k)  # the median model refitted
            lab = kp["idx"].label.to_numpy()[kp["te"]]  # held-out labels
            a_ref = float(b4[(b4.model == mdl) & (b4.receiver == rx) & (b4.held_out == held) & (b4.perm == k)].auroc.iloc[0])  # its AUROC in the table
            if fail or B.scores(p, lab)["auroc"] != a_ref:  # if: the refit does not reproduce it
                raise SystemExit(f"STOP: median permuted model of {mdl} {rx} {held} not reproduced")  # stop
            kpm = {"te": kp["te"], "idx": kp["idx"], "prep": f["prep"], "V": f["V"]}  # the record importance_cell reads, with the permuted model's preparation
            s_, r_, _ = B.quiet(R.importance_cell, B.predictor(mdl, m), kpm, f["names"], f["sel"], fam, xs, {"config": "c4b_refit_median", "model": B.MODEL_NAME[mdl], "receiver": rx, "held_out": held})  # the forest script's importance
            ib += s_  # kept
            ibr += r_  # kept
        ibd = pd.DataFrame(ib)  # the median models' importance
        B.write_csv(ibd, RF_DIR / f"random_forest_importance_c4b{tag}.csv")  # written
        B.write_gz(pd.DataFrame(ibr), RF_DIR / f"random_forest_importance_repeats_c4b{tag}.csv.gz")  # repeats
        ip = B.read_exact(RF_DIR / true_file)  # the true model's importance (written by p)
        cl = B.classification(ip, ibd, ta, b4, med, mdl, CELLS_C4B)  # the forest script's classification with the selection check
        B.write_csv(cl, RF_DIR / f"random_forest_c4b_classification{tag}.csv")  # written
        check(f"classification ({mdl})", None, cl["class"].value_counts().to_dict().__repr__() + f"; E14 selection-dependent {int(cl.e14_selection_dependent.sum())}")  # reported
        if mdl == "rf":  # the forest's figure
            with B.figures_to(GRAPH, "_b", R):  # renamed
                R.plot_importance_within_session(cl, ip)  # the figure: importance against the median permuted model
    done("c4b", t0)  # checks


# ----------------------------------------------------------------------------------------------------------------------
# P-noStave, P-unrounded, P-nocompress
# ----------------------------------------------------------------------------------------------------------------------
def write_run(out: dict, models: tuple, stem: str) -> pd.DataFrame:  # function: one sensitivity run's tables under final/<stem>_*; returns its pooled table
    s = B.summary(out, models)  # pooled
    B.write_csv(out["metrics"], FINAL_DIR / f"{stem}_metrics.csv")  # per fold
    B.write_csv(out["latency"], FINAL_DIR / f"{stem}_latency.csv")  # latency
    B.write_gz(out["predictions"], FINAL_DIR / f"{stem}_predictions.csv.gz")  # predictions
    B.write_csv(out["selection"], FINAL_DIR / f"{stem}_selection.csv")  # selections
    B.write_csv(out["fails"], FINAL_DIR / f"{stem}_fit_failures.csv")  # failed fits
    B.write_csv(s, FINAL_DIR / f"{stem}_summary.csv")  # pooled
    return s  # pooled


def overlap(a: pd.DataFrame, b: pd.DataFrame, col: str, tag: str) -> pd.DataFrame:  # function: per fit, the common columns and the Jaccard overlap of two selections
    x = a[["receivers", "held_out", col]].merge(b[["receivers", "held_out", col]], on=["receivers", "held_out"], suffixes=("_P", f"_{tag}"))  # aligned
    sa, sb = [set(v.split(";")) for v in x[f"{col}_P"]], [set(v.split(";")) for v in x[f"{col}_{tag}"]]  # list comprehensions: the sets
    return x.assign(**{f"n_{col}_P": [len(s) for s in sa], f"n_{col}_{tag}": [len(s) for s in sb], f"common_{col}": [len(s & t) for s, t in zip(sa, sb)], f"jaccard_{col}": [len(s & t) / len(s | t) for s, t in zip(sa, sb)], f"equal_{col}": [s == t for s, t in zip(sa, sb)]})  # dict unpacking into assign: the overlap columns


def cmd_sens() -> None:  # function: P-noStave, P-unrounded, P-nocompress
    t0 = time.time()  # clock
    labels_in_use()  # the scoring module's label set
    wc, wc10 = B.wc_all(), L.window_dict()  # dictionaries
    cells = cells_for(10)  # the windows
    d1 = B.d1_cells(cells, wc10)  # orbit-free columns
    ns = B.run_choices(B.drop_stave(d1), wc10, "P_noStave", False)  # P without the Stave windows
    s_ns = write_run(ns, MODELS, "p_nostave")  # written
    if not SMOKE:  # the Stave stop relabel arm A comparison exists for the final set only
        k7 = B.read_exact(PROC / "features" / "stave_stop_vehicle_motion.csv")  # the final label set's six cells, exact float parsing (the default parser loses the last bit)
        rows = []  # comparison rows
        for r in k7.itertuples():  # for loop over the six cells
            md = {"logistic regression": "lr", "forest": "rf"}[r.model]  # model key
            pr = ns["predictions"][(ns["predictions"].model == md) & (ns["predictions"].receiver == r.receiver)]  # its predictions
            cs = float(pr[pr.label == "clean"].alarm.mean())  # clean-window alarm share over all its clean windows (arm A's common windows)
            fa = float(s_ns[(s_ns.model == md) & (s_ns.receiver == r.receiver)].fa_rate_per_h_pooled.iloc[0])  # pooled false-alarm rate
            rows.append({"model": md, "receiver": r.receiver, "clean_windows": int((pr.label == "clean").sum()), "common_clean_windows_k7": r.common_clean_windows, "clean_share": cs, "clean_share_A": r.clean_share_A, "fa_rate_per_h": fa, "fa_rate_per_h_A": r.fa_rate_per_h_A,  # dict literal: both
                         "equal": bool(cs == r.clean_share_A and ((np.isnan(fa) and np.isnan(r.fa_rate_per_h_A)) or fa == r.fa_rate_per_h_A) and int((pr.label == "clean").sum()) == r.common_clean_windows)})  # exact equality
        t = pd.DataFrame(rows)  # the table
        B.write_csv(t, FINAL_DIR / "adopted_nostave_stave_stop_check.csv")  # written
        check("P-noStave reproduces the Stave stop relabel arm A (stave_stop_vehicle_motion.csv): clean windows, clean share and pooled false-alarm rate, six cells, exactly", None, f"{int(t.equal.sum())} of 6 equal" + ("" if t.equal.all() else " - PROBLEM (reported, not a stop)"))  # reported
    fin = pd.read_csv(PROC / "labels" / "labels_1hz_final.csv", usecols=B.KEYS + ["label"], dtype={"test_code": str})  # final labels
    trl = pd.read_csv(PROC / "labels" / "train_labels_1hz.csv", usecols=B.KEYS + ["train_label"], dtype={"test_code": str})  # training labels
    uc = B.unrounded_cells(wc10, fin, trl)  # the unrounded orbit-free windows
    fc = B.d1_cells(B.load_cells(B.FINAL, 10, True), wc10)  # the final files' orbit-free windows
    okg = all(uc[rx][0][B.KEYS + ["label", "tlabel"]].equals(fc[rx][0][B.KEYS + ["label", "tlabel"]]) and bool(np.all((lambda a, b: (a == b) | (np.isnan(a) & np.isnan(b)))(pd.read_csv(io.StringIO(uc[rx][1].to_csv(index=False, float_format="%.6g"))).to_numpy(float), fc[rx][1].to_numpy(float)))) for rx in B.RXS)  # generator in all(): same windows, and every value equal once written at 6 s.f.
    check("P-unrounded gate: the unrounded windows written at 6 s.f. = the final files, every orbit-free cell, keys and labels", okg, f"{sum(len(uc[rx][0]) for rx in B.RXS)} windows")  # the gate
    if not SMOKE and okg:  # the run itself (final set only)
        pu = B.run_choices(uc, wc10, "P_unrounded", False)  # P on the unrounded windows
        write_run(pu, MODELS, "p_unrounded")  # written
        Psel = B.read_exact(LR_DIR / "logistic_regression_selection_primary.csv")  # P's selections
        B.write_csv(overlap(Psel, pu["selection"], "selected_columns", "unrounded"), FINAL_DIR / "adopted_unrounded_selection_compare.csv")  # the MI-selected columns beside P's
    pn = B.outer_v(cells, wc, {**B.P_SPEC, "compress": False}, ("rf",), "P_nocompress")  # the forest with compress=False (E3)
    write_run(pn, ("rf",), "p_nocompress")  # written
    pk = B.read_exact(FINAL_DIR / "adopted_selection_kept_columns.csv")  # P's kept and selected columns (outer_v, written by p)
    x = overlap(pk, pn["selection"], "kept_columns", "nocompress").merge(overlap(pk, pn["selection"], "selected_columns", "nocompress")[["receivers", "held_out", "n_selected_columns_P", "n_selected_columns_nocompress", "common_selected_columns", "jaccard_selected_columns", "equal_selected_columns"]], on=["receivers", "held_out"])  # both overlaps
    B.write_csv(x.drop(columns=["kept_columns_P", "kept_columns_nocompress"]), FINAL_DIR / "adopted_nocompress_columns_compare.csv")  # written
    done("sens", t0, stop_if_failed=True)  # the unrounded gate stops; the Stave stop relabel comparison is a reported check


# ----------------------------------------------------------------------------------------------------------------------
# variants
# ----------------------------------------------------------------------------------------------------------------------
def variant_table(allm: pd.DataFrame, summ: pd.DataFrame, model: str) -> pd.DataFrame:  # function: the random forest's variant table under the agent experiment: every headline figure, its difference from the baseline, the "moved" call for recall only
    rows = []  # results
    figs = ["recall_pooled", "recall_jam_spoof_pooled", "recall_spoof_pooled", "precision_pooled", "f1_pooled", "clean_in_alarm_share_pooled", "fa_rate_per_h_pooled", "fa_count_episodes", "auroc_median", "auroc_pooled_predictions", "intervals_detected", "latency_median_s", "pre_onset_alarm_share_pooled", "early_warning", "first30_recall"]  # list: the headline figures
    for rx in B.RXS:  # receivers
        for v in [x for x in summ.variant.unique() if x not in ("primary", "primary_noStave")]:  # list comprehension: the variants in run order
            base = "primary_noStave" if v in ("V1_main", "V2_known") else "primary"  # the baseline (V1, V2 without the Stave windows)
            pm = allm[(allm.variant == base) & (allm.receiver == rx)]  # the baseline's folds
            sd = float(pm.recall.std(ddof=0))  # the baseline's across-fold recall spread (ddof 0)
            b = summ[(summ.variant == base) & (summ.receiver == rx)].iloc[0]  # baseline pooled row
            r = summ[(summ.variant == v) & (summ.receiver == rx)]  # the variant's pooled row
            if r.empty:  # if: the variant has no scored fold for this receiver
                continue  # nothing to report
            r = r.iloc[0]  # the row
            row = {"model": model, "variant": v, "family": r.family, "receiver": rx, "baseline": base, "flag": "without the Stave windows" if base == "primary_noStave" else ""}  # dict literal: identity
            for f in figs:  # figures
                row[f] = r[f]  # the variant's
                row[f"d_{f}"] = r[f] - b[f] if pd.notna(r[f]) and pd.notna(b[f]) else np.nan  # difference from the baseline
            row["sd_recall_baseline"] = sd  # the spread
            row["moved_recall"] = bool(abs(row["d_recall_pooled"]) > sd) if pd.notna(row["d_recall_pooled"]) else False  # the agent experiment: recall only
            rows.append(row)  # kept
    return pd.DataFrame(rows)  # the table


def cmd_variants() -> None:  # function: the variants of both models
    t0 = time.time()  # clock
    labels_in_use()  # the scoring module's label set
    wc = B.wc_all()  # dictionary
    P = {k: B.read_exact(LR_DIR / f"logistic_regression_{k}_primary.csv") for k in ("metrics", "latency")}  # P's tables (logistic regression's files hold only its rows)
    out_m, out_l, out_p, out_s, summ = [], [], [], [], []  # long tables
    base_tabs = {"primary": "{n}/{n}_{k}_primary", "primary_noStave": "baselines_controls/adopted_nostave_{k}"}  # dict: baseline -> file pattern
    for mdl, n_ in (("lr", "logistic_regression"), ("rf", "random_forest")):  # for loop over the models
        for vname, stem in base_tabs.items():  # the two baselines
            mt = B.read_exact(ROOT / "Data" / "processed" / (stem.format(n=n_, k="metrics") + ".csv"))  # metrics
            lt = B.read_exact(ROOT / "Data" / "processed" / (stem.format(n=n_, k="latency") + ".csv"))  # latency
            pr = B.read_exact(ROOT / "Data" / "processed" / (stem.format(n=n_, k="predictions") + ".csv.gz"))  # predictions
            mt, lt, pr = mt[mt.model == mdl], lt[lt.model == mdl], pr[pr.model == mdl]  # this model's rows
            out_m.append(mt.assign(variant=vname, family="primary"))  # kept
            out_l.append(lt.assign(variant=vname))  # kept
            out_p.append(pr.assign(variant=vname))  # kept
            summ.append(B.summary({"metrics": mt, "latency": lt, "predictions": pr}, (mdl,)).assign(variant=vname, family="primary"))  # pooled
    cache = {}  # window length -> cells
    for name, fam, models, delta, w, nostave in VARIANTS:  # for loop over the variants
        if w not in cache:  # if: this length is not loaded
            cache = {k: v for k, v in cache.items() if k == 10}  # keep w = 10 only (memory)
            cache[w] = cells_for(w)  # this length's windows
        c = B.drop_stave(cache[w]) if nostave else cache[w]  # V1, V2 without the Stave windows
        o = B.outer_v(c, wc, {**B.P_SPEC, **delta}, models, name)  # the variant
        out_m.append(o["metrics"].assign(variant=name, family=fam))  # kept
        out_l.append(o["latency"].assign(variant=name))  # kept
        out_p.append(o["predictions"].assign(variant=name))  # kept
        out_s.append(o["selection"].assign(variant=name))  # kept
        summ.append(B.summary(o, models).assign(variant=name, family=fam))  # pooled
        if len(o["fails"]):  # if: failed fits
            check(f"variant {name}: failed fits", None, f"{len(o['fails'])}: " + "; ".join(f"{r.model} {r.receivers} {r.held_out}: {r.reason[:80]}" for r in o["fails"].itertuples()))  # reported
    cells10 = cache.get(10) or cells_for(10)  # w = 10 windows
    for mdl in MODELS:  # persist_3of5 for both models
        pr = pd.concat(out_p, ignore_index=True)  # every prediction so far
        pr = pr[(pr.variant == "primary") & (pr.model == mdl)]  # P's
        rows, lats, prs = [], [], []  # outputs
        for (rx, held), d in pr.groupby(["receiver", "held_out"], sort=False):  # P's folds in its order
            al = C9.k_of_n(d.sec.to_numpy(np.int64), d.alarm.to_numpy())  # the sequential-baseline script's causal 3 of 5
            idx = cells10[rx][0]  # the receiver's index
            te = L.eval_rows(idx, held)  # held-out rows in time order
            if not np.array_equal(idx.sec.to_numpy()[te], d.sec.to_numpy()):  # if: not P's windows in P's order
                raise SystemExit("STOP: persist_3of5 rows do not line up")  # stop
            row, lat, p_ = L.evaluate(idx, te, al.astype(float), held, {"run_id": "persist_3of5", "model": mdl, "receiver": rx}, False)  # the regression script's evaluation of the 0/1 alarms
            pm = P["metrics"] if mdl == "lr" else B.read_exact(RF_DIR / "random_forest_metrics_primary.csv")  # P's metrics of this model
            row["auroc"] = float(pm[(pm.receiver == rx) & (pm.held_out == held)].auroc.iloc[0])  # P's AUROC (persistence changes no probability)
            row["average_precision"] = float(pm[(pm.receiver == rx) & (pm.held_out == held)].average_precision.iloc[0])  # and its average precision
            rows.append(row)  # kept
            lats += lat  # kept
            prs.append(p_.assign(p=d.p.to_numpy()))  # P's probabilities kept beside the alarms
        o = {"metrics": pd.DataFrame(rows), "latency": pd.DataFrame(lats), "predictions": pd.concat(prs, ignore_index=True)}  # the frames
        out_m.append(o["metrics"].assign(variant="persist_3of5", family="alarm rule"))  # kept
        out_l.append(o["latency"].assign(variant="persist_3of5"))  # kept
        out_p.append(o["predictions"].assign(variant="persist_3of5"))  # kept
        summ.append(B.summary(o, (mdl,)).assign(variant="persist_3of5", family="alarm rule"))  # pooled (AUROC over pooled predictions is P's)
    allm, alll, allp, alls, sm = (pd.concat(x, ignore_index=True) for x in (out_m, out_l, out_p, out_s, summ))  # generator unpacked: the long tables
    for mdl, folder, prefix in (("lr", LR_DIR, "logistic_regression"), ("rf", RF_DIR, "random_forest")):  # the two models
        B.write_csv(allm[allm.model == mdl], folder / f"{prefix}_metrics_long.csv")  # every fold row
        B.write_csv(alll[alll.model == mdl], folder / f"{prefix}_latency_long.csv")  # latency
        B.write_gz(allp[allp.model == mdl], folder / f"{prefix}_predictions.csv.gz")  # predictions
        B.write_csv(alls, folder / f"{prefix}_selection.csv")  # selections (one row per fit; both models share a fit)
        vt = variant_table(allm[allm.model == mdl], sm[sm.model == mdl], mdl)  # the variant table
        B.write_csv(vt, folder / f"{prefix}_variants.csv")  # written
        check(f"variants ({mdl}): moved_recall (the agent experiment)", None, "; ".join(f"{r.variant} {r.receiver} {r.d_recall_pooled:+.3f}" for r in vt[vt.moved_recall].itertuples()) or "none")  # reported
    with B.figures_to(GRAPH, "_b", C9.M):  # the sequential-baseline script's figure function saves through its own 09
        lm, rm = allm[allm.model == "lr"], allm[allm.model == "rf"]  # each model's rows
        fs = ["primary", "V1_main", "V2_known", "V3_shift_invariant", "V4_no_selection", "screen_no_floor", "unscreened", "plus_geometry_levels"]  # the feature-set family
        C9.family_figure(lm, fs, M.GRAPH / "logistic_regression_variants_feature_sets.png", "Feature-set variants of the adopted logistic regression (V1, V2 without the Stave windows)")  # the figure
        C9.family_figure(lm, ["w1", "w2", "w3", "w5", "primary"], M.GRAPH / "logistic_regression_variants_windows.png", "Window length (adopted primary: w = 10 s)")  # the figure
        C9.family_figure(lm, ["C0.01", "C0.1", "primary", "C10"], M.GRAPH / "logistic_regression_variants_regulariser.png", "Regularisation strength (adopted primary: C = 1)")  # the figure
        C9.family_figure(lm, ["primary", "persist_3of5"], M.GRAPH / "logistic_regression_variants_persistence.png", "Alarm rule: every window at p >= 0.5 (adopted primary) against 3 of 5 windows", episodes=True)  # the figure
        C9.family_figure(rm, fs, M.GRAPH / "random_forest_variants_feature_sets.png", "Feature-set variants of the adopted forest (V1, V2 without the Stave windows)")  # the figure
        C9.family_figure(rm, ["w1", "w2", "w3", "w5", "primary"], M.GRAPH / "random_forest_variants_windows.png", "Window length (adopted primary: w = 10 s)")  # the figure
        C9.family_figure(rm, ["leaf1", "primary", "leaf20"], M.GRAPH / "random_forest_variants_leaf.png", "min_samples_leaf 1, 5 (adopted primary) and 20")  # the figure
        C9.family_figure(rm, ["primary", "persist_3of5"], M.GRAPH / "random_forest_variants_persistence.png", "Alarm rule: every window at p >= 0.5 (adopted primary) against 3 of 5 windows", episodes=True)  # the figure
    done("variants", t0)  # checks


# ----------------------------------------------------------------------------------------------------------------------
# probe
# ----------------------------------------------------------------------------------------------------------------------
def cmd_probe() -> None:  # function: Table 58's probe on the five position sessions, registered and final windows
    t0 = time.time()  # clock
    pr = B.probe(B.REGISTERED, False)  # registered windows without 2.6.12
    pr.round(4).to_csv(FINAL_DIR / "probe_registered_5sessions.csv", index=False, lineterminator="\n")  # written at the screen script's precision
    if not SMOKE:  # the final windows (adopted)
        pf = B.probe(B.FINAL, True)  # final windows without 2.6.12
        pf.round(4).to_csv(FINAL_DIR / "probe_final_5sessions.csv", index=False, lineterminator="\n")  # written
        with B.figures_to(GRAPH, "_b", S):  # renamed
            S.plot_session_probe(pf)  # the figure on the five sessions, final windows
    check("probe: sessions seen", None, f"{int(pr.n_sessions.max())} sessions; chance {pr.chance_balanced.iloc[0]:.3f}")  # reported
    done("probe", t0)  # checks


# ----------------------------------------------------------------------------------------------------------------------
# report
# ----------------------------------------------------------------------------------------------------------------------
FIGS = ["recall_pooled", "recall_jam_spoof_pooled", "recall_spoof_pooled", "precision_pooled", "f1_pooled", "clean_in_alarm", "clean_windows", "clean_in_alarm_share_pooled", "fa_rate_per_h_pooled", "fa_upper95_per_h_pooled", "fa_rate_folds_policy", "fa_count_episodes", "fa_count_folds_policy",  # list over two lines: the headline figures
        "auroc_median", "auroc_folds", "auroc_pooled_predictions", "intervals_detected", "intervals", "latency_median_s", "lead_06c_median_s", "pre_onset_alarm_share_pooled", "first30_recall", "early_warning"]  # continued
NFIGS = ["auroc_pooled_predictions", "recall_pooled", "clean_in_alarm_share_pooled", "fa_rate_per_h_pooled", "pre_onset_alarm_share_pooled", "early_warning"]  # nested figures


def choices_summary(run: str) -> pd.DataFrame:  # function: an the design-choice study run's outer summary with early warning
    s = B.read_exact(AA3 / "runs" / run / "outer_summary.csv")  # committed
    return s.assign(early_warning=s.pre_onset_alarm_share_pooled - s.clean_in_alarm_share_pooled)  # the early-warning margin


def nested_frames(run: str, root: Path | None) -> tuple:  # function: nested metrics, latency, predictions of the design-choice study's registered labels or of P
    if root is None:  # the design-choice study's committed run
        return tuple(B.read_exact(AA3 / "runs" / run / f"nested_{x}") for x in ("metrics.csv", "latency.csv", "predictions.csv.gz"))  # tuple of frames
    return tuple(B.read_exact(root / f"adopted_nested_{x}") for x in ("metrics.csv", "latency.csv", "predictions.csv.gz"))  # P's


def cmd_report() -> None:  # function: registered beside adopted, nested and early warning, common windows, per fold, the session control on all cells, class counts, figures
    t0 = time.time()  # clock
    labels_in_use()  # the scoring module's label set
    p = pd.concat([B.read_exact(LR_DIR / "logistic_regression_pooled_primary.csv"), B.read_exact(RF_DIR / "random_forest_pooled_primary.csv")], ignore_index=True)  # P's pooled
    srcs = {"registered_REGISTERED": choices_summary("REGISTERED"), "choices_STRICT": choices_summary("STRICT"), "adopted_P": p, "P_noStave": B.read_exact(FINAL_DIR / "adopted_nostave_summary.csv")}  # dict: the columns of the table
    if (FINAL_DIR / "adopted_unrounded_summary.csv").exists():  # if: run (final set)
        srcs["P_unrounded"] = B.read_exact(FINAL_DIR / "adopted_unrounded_summary.csv")  # unrounded windows
    srcs["P_nocompress"] = B.read_exact(FINAL_DIR / "adopted_nocompress_summary.csv")  # forest only
    rows = []  # long rows
    for mdl in MODELS:  # models
        for rx in B.RXS:  # receivers
            for f in FIGS:  # figures
                r = {"model": mdl, "receiver": rx, "figure": f}  # identity
                for k, d in srcs.items():  # sources
                    x = d[(d.model == mdl) & (d.receiver == rx)]  # the row
                    r[k] = x[f].iloc[0] if len(x) and f in x else np.nan  # its value
                rows.append(r)  # kept
    B.write_csv(pd.DataFrame(rows), FINAL_DIR / "registered_vs_adopted.csv")  # the agent experiment's table
    k0p = B.read_exact(AA3 / "runs" / "REGISTERED" / "outer_predictions.csv.gz")  # registered predictions
    pp = pd.concat([B.read_exact(LR_DIR / "logistic_regression_predictions_primary.csv.gz"), B.read_exact(RF_DIR / "random_forest_predictions_primary.csv.gz")], ignore_index=True)  # P's
    x = pp.merge(k0p, on=["model", "receiver", "held_out", "sec"], suffixes=("", "_k0"))  # windows in both
    com = []  # common-window rows
    for (md, rx), g in x.groupby(["model", "receiver"]):  # model x receiver
        cc, aa = g[(g.label == "clean") & (g.label_k0 == "clean")], g[g.label.isin(B.ATTACK) & g.label_k0.isin(B.ATTACK)]  # clean in both; attack in both
        com.append({"model": md, "receiver": rx, "common_clean_windows": len(cc), "clean_share_P": cc.alarm.mean(), "clean_share_REGISTERED": cc.alarm_k0.mean(), "common_attack_windows": len(aa), "recall_P": aa.alarm.mean(), "recall_REGISTERED": aa.alarm_k0.mean(),  # dict literal: the common-window figures
                    "clean_only_in_P": int(((g.label == "clean") & (g.label_k0 != "clean")).sum()), "windows_only_in_P": int(((pp.model == md) & (pp.receiver == rx)).sum()) - len(g)})  # windows that differ
    B.write_csv(pd.DataFrame(com), FINAL_DIR / "common_windows_vs_registered.csv")  # written
    pm = pd.concat([B.read_exact(LR_DIR / "logistic_regression_metrics_primary.csv"), B.read_exact(RF_DIR / "random_forest_metrics_primary.csv")], ignore_index=True)  # P per fold
    k0m = B.read_exact(AA3 / "runs" / "REGISTERED" / "outer_metrics.csv")  # the registered labels per fold
    cols = ["clean", "attack_jam_spoof", "attack_spoof", "recall", "recall_jam_spoof", "recall_spoof", "precision", "clean_in_alarm_share", "fa_figure", "fa_episodes", "auroc"]  # per-fold columns
    pf = pm[["model", "receiver", "held_out"] + cols].merge(k0m[["model", "receiver", "held_out"] + cols], on=["model", "receiver", "held_out"], suffixes=("_P", "_registered"), how="left")  # P beside the registered labels per fold
    B.write_csv(pf, FINAL_DIR / "per_fold_registered_vs_adopted.csv")  # the diagnosis table
    ck = ["model", "receiver", "outer_held_out", "held_out"]  # an inner fold's identity
    nP, nK = nested_frames("P", FINAL_DIR), nested_frames("REGISTERED", None)  # nested frames
    common = nP[0][ck].merge(nK[0][ck], on=ck)  # inner folds fitted in both
    sub = lambda t: tuple(d.merge(common, on=ck) for d in t)  # lambda: frames on the common inner folds
    sP, sK = L.nested_summary(*sub(nP), False), L.nested_summary(*sub(nK), False)  # per outer fold
    for s_ in (sP, sK):  # both
        s_["early_warning"] = s_.pre_onset_alarm_share_pooled - s_.clean_in_alarm_share_pooled  # the early-warning margin
    allP = B.read_exact(FINAL_DIR / "adopted_nested_summary.csv")  # P on all its inner folds
    nrows, calls = [], []  # nested rows; E2 calls
    for (md, rx), g in sP.groupby(["model", "receiver"], sort=False):  # model x receiver
        k = sK[(sK.model == md) & (sK.receiver == rx)].set_index("outer_held_out")  # the registered labels on the same inner folds
        g = g.set_index("outer_held_out")  # P's
        ga = allP[(allP.model == md) & (allP.receiver == rx)].set_index("outer_held_out")  # P on all its inner folds
        r = {"model": md, "receiver": rx, "inner_folds_P": int(((nP[0].model == md) & (nP[0].receiver == rx)).sum()), "inner_folds_common_with_REGISTERED": int(((common.model == md) & (common.receiver == rx)).sum())}  # identity
        for f in NFIGS:  # figures
            v = g[f].astype(float)  # P per outer fold (common inner folds)
            kk = k[f].astype(float).reindex(v.index)  # the registered labels'
            dd = v - kk  # change
            sign = -1 if f in ("clean_in_alarm_share_pooled", "fa_rate_per_h_pooled") else 1  # lower is better for these two
            r.update({f"{f}_P_median": v.median(), f"{f}_REGISTERED_median": kk.median(), f"{f}_P_by_fold": ";".join(f"{h}={x_:.4f}" for h, x_ in v.items() if pd.notna(x_)), f"{f}_REGISTERED_by_fold": ";".join(f"{h}={x_:.4f}" for h, x_ in kk.items() if pd.notna(x_)),  # dict.update: medians and per-fold values
                      f"d_{f}_median": dd.median(), f"{f}_folds_better": int((sign * dd > 0).sum()), f"{f}_folds_worse": int((sign * dd < 0).sum()), f"{f}_folds_compared": int(dd.notna().sum()), f"{f}_P_median_all_inner_folds": ga[f].astype(float).median()})  # changes and counts
        nrows.append(r)  # kept
        conf = r["early_warning_folds_better"] > r["early_warning_folds_compared"] / 2  # the early-warning margin: more than half of the compared outer folds better
        calls.append({"model": md, "receiver": rx, "folds_better": r["early_warning_folds_better"], "folds_compared": r["early_warning_folds_compared"], "d_median": r["d_early_warning_median"], "call": "confirmed" if conf else "not confirmed"})  # E2's call
    B.write_csv(pd.DataFrame(nrows), FINAL_DIR / "nested_registered_vs_adopted.csv")  # written
    B.write_csv(pd.DataFrame(calls), FINAL_DIR / "early_warning_nested_calls.csv")  # written
    ew = pd.DataFrame([{"model": r["model"], "receiver": r["receiver"], "early_warning_P": r["adopted_P"], "early_warning_REGISTERED": r["registered_REGISTERED"], "early_warning_STRICT": r["choices_STRICT"], "early_warning_P_noStave": r["P_noStave"]} for r in rows if r["figure"] == "early_warning"])  # list comprehension: outer early warning
    B.write_csv(ew, FINAL_DIR / "early_warning_outer.csv")  # written
    v = B.read_exact(FINAL_DIR / "r3_verdicts.csv")  # the adopted models' session control
    reg = []  # recounts
    for run in ("C4BR", "C4BR_STRICT"):  # the design-choice study's two refitted controls (forest)
        b = B.read_exact(AA3 / "runs" / run / "c4b_runs.csv").assign(model="rf", failed=False)  # their runs
        c = B.read_exact(AA3 / "runs" / run / "c4b_classification.csv")  # their true AUROC
        ta = {("rf", r.receiver, r.held_out): r.primary_auroc for r in c.drop_duplicates(["receiver", "held_out"]).itertuples()}  # dict comprehension
        reg.append(B.cell_verdicts(b, ta, sorted({(r.receiver, r.held_out) for r in b.itertuples()}, key=lambda t: (B.RXS.index(t[0]), t[1])), run))  # the session control on them
    allv = pd.concat([v] + reg, ignore_index=True)  # every verdict
    B.write_csv(allv, FINAL_DIR / "r3_all.csv")  # written
    cnt = []  # class counts
    srcc = [("the random forest registered (7 cells)", B.read_exact(PROC / "random_forest_registered" / "random_forest_c4b_classification.csv"))] + [(f"CHOICES {r_}", B.read_exact(AA3 / "runs" / r_ / "c4b_classification.csv")) for r_ in ("C4BR", "C4BR_STRICT")]  # list: registered sources
    srcc += [("adopted forest (C4b-refit)", B.read_exact(RF_DIR / "random_forest_c4b_classification.csv")), ("adopted logistic regression (C4b-refit, addition)", B.read_exact(RF_DIR / "random_forest_c4b_classification_logreg.csv"))]  # adopted sources
    for nme, d in srcc:  # sources
        cnt.append({"source": nme, "cells": int(d[["receiver", "held_out"]].drop_duplicates().shape[0]), **{k_: int((d["class"] == k_).sum()) for k_ in ["attack evidence", "session carrier", "not used", "not selected", "not interpretable"]}, "e14_selection_dependent": int(d.e14_selection_dependent.sum()) if "e14_selection_dependent" in d else np.nan})  # dict unpacking: one row
    B.write_csv(pd.DataFrame(cnt), FINAL_DIR / "class_counts.csv")  # written
    mr, ml = B.read_exact(RF_DIR / "random_forest_metrics_primary.csv"), B.read_exact(LR_DIR / "logistic_regression_metrics_primary.csv")  # per fold
    g5 = B.read_exact(RF_DIR / "random_forest_metrics_c5_geometry.csv")  # C5 forest
    b4r, a4r = B.read_exact(RF_DIR / "random_forest_c4b_within_session.csv"), B.read_exact(RF_DIR / "random_forest_c4a_permutation.csv")  # C4b-refit and C4a forest
    with B.figures_to(GRAPH, "_b", R):  # renamed
        R.plot_forest_controls(mr, ml, g5, b4r[~b4r.failed.astype(bool)], a4r[~a4r.failed.astype(bool)])  # the figure: forest beside the logistic regression, C5, C4b-refit (median) and C4a
    plot_control_verdicts(pd.concat([B.read_exact(LR_DIR / "logistic_regression_c4b_within_session.csv"), b4r], ignore_index=True), v)  # the session-control figure
    done("report", t0)  # checks


def plot_control_verdicts(b4: pd.DataFrame, v: pd.DataFrame) -> None:  # function: the session control per cell and model: the 100 permuted AUROCs, their 95th percentile and the adopted model's AUROC
    import matplotlib.pyplot as plt  # pyplot (09 chose the file-only backend)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharey=True)  # one panel per model
    for a, (mdl, nm) in zip(ax, (("lr", "logistic regression"), ("rf", "forest"))):  # panels
        for i, (rx, held) in enumerate(CELLS_C4B):  # cells
            d = np.sort(b4[(b4.model == mdl) & (b4.receiver == rx) & (b4.held_out == held) & ~b4.failed.astype(bool)].auroc.to_numpy())  # completed permuted AUROCs, sorted
            r = v[(v.model == mdl) & (v.receiver == rx) & (v.held_out == held)].iloc[0]  # the verdict row
            a.plot(i + np.linspace(-0.3, 0.3, len(d)), d, ".", ms=3, color="#999999", label="permuted AUROC (C4b-refit)" if i == 0 else None)  # strip of permuted AUROCs
            a.plot([i - 0.35, i + 0.35], [r.perm_auroc_p95] * 2, color="#0072B2", lw=1.5, label="95th percentile" if i == 0 else None)  # p95
            a.plot(i, r.true_auroc, "D", color="#D55E00", ms=7, label="adopted model's AUROC" if i == 0 else None)  # true AUROC
            a.text(i, 1.03, r.verdict, ha="center", fontsize=7, rotation=20)  # the verdict
        a.axhline(B.UNINF, color="k", lw=0.6, ls="--")  # the 0.995 line
        a.set_xticks(range(len(CELLS_C4B)))  # ticks
        a.set_xticklabels([f"{M.RXN[rx]}\n{h}" for rx, h in CELLS_C4B], fontsize=7)  # list comprehension: cell names
        a.set_title(f"{nm}session control per cell", fontsize=10, pad=28)  # title
        a.set_ylim(0, 1.12)  # range (room for the verdicts)
    ax[0].set_ylabel("held-out AUROC")  # axis label
    ax[0].legend(fontsize=7, loc="lower left")  # legend
    fig.tight_layout()  # layout
    M.save_fig(fig, GRAPH / "figure_r3_session_control_final.png")  # byte-stable save


# ----------------------------------------------------------------------------------------------------------------------
# P-no-quiet (Appendix A obligation 1): P with one change, the quiet training negatives of the training-label script not used
# ----------------------------------------------------------------------------------------------------------------------
PMK2 = FINAL_DIR / "no_quiet_negatives"  # Path joined with "/": P-no-quiet's own folder (new names only)
# list: the per-fold columns of the regression script's metrics table reported beside P (training counts, class recall, policy precision, false alarms and bound, AUROC)
PF_COLS = ["train_rows", "train_clean", "train_attack", "clean", "attack_jam_spoof", "attack_spoof", "recall", "recall_jam_spoof", "recall_spoof", "precision_reported", "precision", "clean_in_alarm_share", "fa_figure", "fa_episodes", "fa_upper95_count", "fa_rate_per_h", "fa_upper95_per_h", "auroc"]
C6_COLS = ["recall_min", "recall_max", "auroc_min", "auroc_max", "clean_alarm_min", "clean_alarm_max", "jaccard_min"]  # list: C6's spread columns over the five seeds


def without_quiet(cells: dict) -> dict:  # function: the single change: training label = scoring label, so the quiet negatives (scored uncertain, trained clean) leave training
    return {rx: (idx.assign(tlabel=idx.label.to_numpy()), X) for rx, (idx, X) in cells.items()}  # dict comprehension: the same windows and the same value frame, only tlabel replaced


def beside(p: pd.DataFrame, q: pd.DataFrame, keys: list, cols: list) -> pd.DataFrame:  # function: P's and P-no-quiet's columns side by side on the same keys
    return p[keys + cols].merge(q[keys + cols], on=keys, how="outer", suffixes=("_P", "_PmK2"), validate="one_to_one")  # outer merge: a row present on one side only stays visible


def c6_spread(sr: pd.DataFrame) -> pd.DataFrame:  # function: C6's range over the five seeds per model, varied seed, receiver and fold (cmd_controls' rule, for P-no-quiet's own rows)
    disp = []  # dispersion frames
    for (mdl, v), d in [(k_, g) for k_, g in sr.groupby(["model", "varied"]) if k_[1] != "primary"]:  # list comprehension: the varied seeds of each model
        d = pd.concat([sr[(sr.model == mdl) & (sr.varied == "primary")], d])  # with the reported model
        # named aggregation over two lines: the range over the five seeds, and the smallest selection overlap with the reported model
        g = d.groupby(["receiver", "held_out"]).agg(recall_min=("recall", "min"), recall_max=("recall", "max"), auroc_min=("auroc", "min"), auroc_max=("auroc", "max"),
                                                     clean_alarm_min=("clean_in_alarm_share", "min"), clean_alarm_max=("clean_in_alarm_share", "max"), jaccard_min=("jaccard_with_primary", "min")).reset_index()  # continuation: clean-window alarm share range and the smallest Jaccard overlap
        disp.append(g.assign(model=mdl, seed_varied=v))  # kept with its model and varied seed
    return pd.concat(disp, ignore_index=True)  # one table


def cmd_pmk2() -> None:  # function: P-no-quiet outer and nested, C4a, C4b-refit with the session control, C6 seeds, every table beside P; gates G0-G2 stop on failure
    t0 = time.time()  # clock
    labels_in_use()  # the scoring module's label set (final)
    wc, wc10 = B.wc_all(), L.window_dict()  # dictionaries
    cells = cells_for(10)  # P's windows with the strict training label
    km = ["model", "receiver", "held_out", "sec"]  # keys with the model
    pp = pd.concat([B.read_exact(LR_DIR / "logistic_regression_predictions_primary.csv.gz"), B.read_exact(RF_DIR / "random_forest_predictions_primary.csv.gz")], ignore_index=True)  # committed P predictions, both models
    off = B.run_choices(B.d1_cells(cells, wc10), wc10, "P", False)  # switch off: P fitted exactly as cmd_p fits it
    check("G0 switch off: P reproduced bitwise (committed logistic_regression and random_forest primary predictions)", same_preds(off["predictions"], pp, km), f"{len(pp)} predictions")  # the code path is P's
    pm_cells = without_quiet(cells)  # switch on: the single change
    rows = []  # one row per receiver
    for rx in B.RXS:  # for loop over the receivers
        a, b = cells[rx][0], pm_cells[rx][0]  # P's index and P-no-quiet's
        ch = a.tlabel.to_numpy() != b.tlabel.to_numpy()  # boolean array: windows whose training label changes
        # dict literal over three lines: what changed and what must not have changed
        rows.append({"receiver": rx, "windows": len(a), "changed_training_label": int(ch.sum()), "changed_from_clean": int((a.tlabel[ch] == "clean").sum()), "changed_scored_uncertain": int((a.label[ch] == "uncertain").sum()),
                     "changed_scored_clean_or_attack": int(a.label[ch].isin(["clean"] + list(B.ATTACK)).sum()), "keys_and_scoring_labels_equal": bool(a[B.KEYS + ["label"]].equals(b[B.KEYS + ["label"]])),  # continuation: no scored clean or attack window may change; keys and scoring labels equal
                     "values_same_frame": cells[rx][1] is pm_cells[rx][1], "training_clean_P": int((a.tlabel == "clean").sum()), "training_clean_PmK2": int((b.tlabel == "clean").sum())})  # continuation: the values are the same frame; training-clean counts before and after
    lc = pd.DataFrame(rows)  # the table
    B.write_csv(lc, PMK2 / "no_quiet_negatives_label_change.csv")  # written
    # boolean over the table: every changed window was a quiet negative (trained clean, scored uncertain), and nothing else moved
    g1 = bool(((lc.changed_training_label == lc.changed_from_clean) & (lc.changed_training_label == lc.changed_scored_uncertain) & (lc.changed_scored_clean_or_attack == 0) & lc.keys_and_scoring_labels_equal & lc.values_same_frame).all())
    check("G1 switch on: only quiet-window training labels change (counts per receiver)", g1, "; ".join(f"{r.receiver}: {r.changed_training_label} of {r.windows} windows, training clean {r.training_clean_P} -> {r.training_clean_PmK2}" for r in lc.itertuples()))  # reported with the counts
    d1 = B.d1_cells(pm_cells, wc10)  # the orbit-free columns of the changed cells
    PM = B.run_choices(d1, wc10, "PmK2", True)  # P-no-quiet's outer run, every fit kept for the controls
    ov = B.outer_v(pm_cells, wc, B.P_SPEC, MODELS, "PmK2")  # outer_v with P's specification on the changed cells
    check("G2 outer_v(P's specification) = choices_lib.outer on the P-no-quiet cells (bitwise; the C6 seed runs use outer_v)", same_preds(ov["predictions"], PM["predictions"], km), f"{len(PM['predictions'])} predictions")  # the seed runs vary only the seed
    if any(c["result"] == "FAIL" for c in CHECKS):  # if: a gate failed
        done("no_quiet_negatives", t0)  # writes the checks and stops with status 1
    check("P-no-quiet outer: failed fits (reported)", None, f"{len(PM['fails'])} failed fits")  # reported
    summ = write_run(PM, MODELS, "no_quiet_negatives/no_quiet_negatives")  # per-fold, latency, predictions, selection, failures, pooled tables under no_quiet_negatives/
    n = B.run_nested(d1, wc10, "PmK2")  # nested scores, as P's
    for k_ in ("metrics", "latency", "selection", "summary"):  # for loop over the nested tables
        B.write_csv(n[k_], PMK2 / f"no_quiet_negatives_nested_{k_}.csv")  # written
    B.write_gz(n["predictions"], PMK2 / "no_quiet_negatives_nested_predictions.csv.gz")  # inner out-of-fold predictions
    B.write_csv(n["fails"], PMK2 / "no_quiet_negatives_nested_fit_failures.csv")  # unfittable inner folds
    a4 = B.c4a(PM["kept"], MODELS, "C4A_PmK2", PERMS)  # C4a: 100 permutations per cell and model (seeds 20260914 + k, as P's)
    B.write_csv(a4, PMK2 / "no_quiet_negatives_c4a_permutation.csv")  # every permutation
    cm = cell_medians(a4)  # medians per cell
    B.write_csv(cm, PMK2 / "no_quiet_negatives_c4a_summary.csv")  # written
    check("C4a on P-no-quiet (reported: the control of an alternative is a result, not a stop)", None, "; ".join(f"{r.model} {r.receiver} {r.held_out}: {r.auroc_median:.3f} {r.status}" for r in cm.itertuples()))  # reported
    b4 = B.c4b_refit(PM["kept"], wc10, MODELS, CELLS_C4B, "C4BR_PmK2", PERMS)  # C4b-refit: 100 permutations per cell and model (seeds 20260914 + 1000 + k, as P's)
    B.write_csv(b4, PMK2 / "no_quiet_negatives_c4b_within_session.csv")  # every permutation
    v = B.cell_verdicts(b4, true_auroc(PM["metrics"]), CELLS_C4B, "C4BR_PmK2")  # the session control per cell and model
    B.write_csv(v, PMK2 / "no_quiet_negatives_r3_verdicts.csv")  # written
    check("session control on P-no-quiet (reported)", None, "; ".join(f"{r.model} {r.receiver} {r.held_out}: {r.verdict}" for r in v.itertuples()))  # reported
    runs = [("lr", B.SEED, B.SEED, "primary", PM), ("rf", B.SEED, B.SEED, "primary", PM)]  # C6: the reported models first
    for s_ in MI_SEEDS[1:]:  # for loop over the other MI seeds
        runs.append(("lr", s_, B.SEED, "mi_seed", B.outer_v(pm_cells, wc, {**B.P_SPEC, "mi_seed": s_}, ("lr",), "C6_PmK2")))  # the logistic regression
        runs.append(("rf", s_, B.SEED, "mi_seed", B.outer_v(pm_cells, wc, {**B.P_SPEC, "mi_seed": s_}, ("rf",), "C6_PmK2")))  # the forest
    for s_ in FOREST_SEEDS[1:]:  # for loop over the other forest seeds
        runs.append(("rf", B.SEED, s_, "forest_seed", B.outer_v(pm_cells, wc, {**B.P_SPEC, "forest_seed": s_}, ("rf",), "C6_PmK2")))  # the forest
    sr = seed_rows(runs, PM["selection"])  # per-run rows
    B.write_csv(sr, PMK2 / "no_quiet_negatives_c6_seeds.csv")  # written
    dp = c6_spread(sr)  # the range over the five seeds
    B.write_csv(dp, PMK2 / "no_quiet_negatives_c6_dispersion.csv")  # written
    B.write_csv(pd.concat([B.summary(o, (mdl,)).assign(mi_seed=mi, forest_seed=fs, varied=vv) for mdl, mi, fs, vv, o in runs], ignore_index=True), PMK2 / "no_quiet_negatives_c6_pooled.csv")  # list comprehension: pooled figures per seed
    mP = pd.concat([B.read_exact(LR_DIR / "logistic_regression_metrics_primary.csv"), B.read_exact(RF_DIR / "random_forest_metrics_primary.csv")], ignore_index=True)  # P per fold, both models
    B.write_csv(beside(mP, PM["metrics"], ["model", "receiver", "held_out"], PF_COLS), PMK2 / "no_quiet_negatives_vs_adopted_per_fold.csv")  # per receiver and fold, P beside P-no-quiet
    sP = pd.concat([B.read_exact(LR_DIR / "logistic_regression_pooled_primary.csv"), B.read_exact(RF_DIR / "random_forest_pooled_primary.csv")], ignore_index=True)  # P pooled
    B.write_csv(beside(sP, summ, ["model", "receiver"], [f for f in FIGS if f in sP and f in summ]), PMK2 / "no_quiet_negatives_vs_adopted_pooled.csv")  # list comprehension: the headline figures present in both
    nP = B.read_exact(FINAL_DIR / "adopted_nested_summary.csv")  # P's nested summary
    B.write_csv(beside(nP, n["summary"], ["model", "receiver", "outer_held_out"], NFIGS), PMK2 / "no_quiet_negatives_vs_adopted_nested.csv")  # nested figures per outer fold
    B.write_csv(beside(B.read_exact(FINAL_DIR / "c4a_summary.csv"), cm, ["model", "receiver", "held_out"], ["failed", "auroc_median", "auroc_p5", "auroc_p95", "status"]), PMK2 / "no_quiet_negatives_vs_adopted_c4a.csv")  # C4a beside P's
    B.write_csv(beside(B.read_exact(FINAL_DIR / "r3_verdicts.csv"), v, ["model", "receiver", "held_out"], ["failed_fits", "true_auroc", "perm_auroc_median", "perm_auroc_p95", "verdict"]), PMK2 / "no_quiet_negatives_vs_adopted_r3.csv")  # the session control beside P's
    dP = pd.concat([B.read_exact(LR_DIR / "logistic_regression_c6_dispersion.csv"), B.read_exact(RF_DIR / "random_forest_c6_dispersion.csv")], ignore_index=True)  # P's five-seed spread, both models
    B.write_csv(beside(dP, dp, ["model", "seed_varied", "receiver", "held_out"], C6_COLS), PMK2 / "no_quiet_negatives_vs_adopted_c6.csv")  # the spread beside P's
    done("no_quiet_negatives", t0)  # checks and run time


# ----------------------------------------------------------------------------------------------------------------------
# the pass comparison
# ----------------------------------------------------------------------------------------------------------------------
def outputs(root: Path) -> list:  # function: every T2 output under a root, as paths relative to it (run times and T0/T1 records excluded)
    out = []  # paths
    for d in ("Data/processed/logistic_regression", "Data/processed/random_forest", "Data/processed/baselines_controls"):  # the output folders
        for f in sorted((root / d).rglob("*")):  # every file
            rel = f.relative_to(root).as_posix()  # relative path
            if f.is_file() and "/timing/" not in rel and not Path(rel).name.startswith(("t0_", "t1_", "FINAL_", "pass_compare")) and "/t0_registered/" not in rel:  # T2 outputs only
                out.append(rel)  # kept
    out += sorted(f.relative_to(root).as_posix() for f in (root / "Graph").glob("*_final.png"))  # generator in sorted: the adopted figures
    return out  # the list


def cmd_compare(other: Path) -> None:  # function: pass 1 (the project) against pass 2, byte for byte
    a, b = outputs(PROJECT), outputs(other)  # both lists
    rows = [{"file": f, "in_pass1": f in a, "in_pass2": f in b, "identical": f in a and f in b and filecmp.cmp(PROJECT / f, other / f, shallow=False)} for f in sorted(set(a) | set(b))]  # list comprehension: every file
    t = pd.DataFrame(rows)  # the table
    B.write_csv(t, PROC / "baselines_controls" / "pass_compare.csv")  # written
    print(f"{int(t.identical.sum())} of {len(t)} files identical; {(~t.identical).sum()} different or missing")  # summary
    if not t.identical.all():  # if: any difference
        print(t[~t.identical].to_string(index=False))  # listed
        sys.exit(1)  # a problem


if __name__ == "__main__":  # run as a script
    cmd = [a for a in ARGS if not a.startswith("--") and (ARGS.index(a) == 0 or ARGS[ARGS.index(a) - 1] != "--root")][0]  # list comprehension: the command word
    {"table": cmd_table, "gates": cmd_gates, "p": cmd_p, "controls": cmd_controls, "c4b": cmd_c4b, "sens": cmd_sens, "variants": cmd_variants, "probe": cmd_probe, "report": cmd_report, "no_quiet_negatives": cmd_pmk2}.get(cmd, lambda: cmd_compare(Path(ARGS[1]).resolve()) if cmd == "compare" else sys.exit(__doc__))()  # dict dispatch: the command's function, called
