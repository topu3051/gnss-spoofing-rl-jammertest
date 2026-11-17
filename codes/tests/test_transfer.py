#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
The tests of 7_evaluation/01_transfer.py, fixed before the script ran. --part1:
  T-A1 Part A chooses nothing: the points it used are the agent experiment's committed points of the mosaic-X5, bitwise, and that file is unchanged.
  T-A2 the point code is right: the transfer script's points_cell() reproduces the agent experiment's committed points (P6 as amended) on all 15 the agent experiment receiver-fold cells, bitwise.
  T-A3 Part B chooses nothing from 2.6.12: no input of its points holds a 2.6.12 row, and the points recomputed from the frozen inputs
       equal the frozen points bitwise.
  T-B  the 2.6.12 guard: a planted read (read_2612) before the freeze is refused; a guard given a commit that is not the freeze is refused;
       no function of the transfer script except read_2612 reads a window file directly.
  T-C  every X5 agent Part A used equals its frozen weights (SHA-256 against models_manifest.csv), recomputed here.
  T-D  the renaming map is applied only where listed: renamed columns are only u-blox 01's Galileo E1 ones, and the transfer script renames nothing.
  T-E  Part B's nested scores equal logistic_regression/random_forest, its inner primary states the environment's and its surrogate states the environment's held-out ones (recomputed here
       from the written files); T-E2 its primary inner trainings reproduce the agent experiment's frozen outer agents: at the agent experiment's budget, the decision- and
       stopping-mode figures on the held-out session equal the agent experiment's held-out runs of seeds 20260914 and 20260915.
  T-F  12 Part B jobs run again from scratch in a separate folder give byte-identical outputs.
--part2:
  T2-G the guard refuses a planted read before the freeze (FREEZE_COMMIT emptied in this process only; an altered temporary manifest; a
       commit that is not the freeze), and only the evaluation and the diagnostic call read_2612.
  T2-P the frozen points recomputed with every 2.6.12 file blocked from the read path equal the frozen points bitwise.
  T2-D the decomposition identity (sum of coefficient x mean difference = the observed shift, to 1e-9), the target mean against the
       evaluation's scores, and the written coefficients against a refit of the frozen five-session regression.
  T2-H "in alarm at onset" by hand: u-blox 01 lr (jamming and spoofing), u-blox 01 k of n (time spoofing), mosaic-H lr (time spoofing).
  T2-S three delays by hand against the per-second stream, plotted on both clocks into transfer/unseen_attack/spotchecks/ (not numbered).
  T2-C the code change since run1 is confined to the listed reporting and diagnostic functions.
  T2-B run1's untouched outputs equal run1's copy byte for byte; run2 and run3 are equal in every output.
  --part1's T-B1 and T-B3 describe the state before the freeze constant was set; --part1 is not run again after the freeze.
--part2r (after the recomputation): T2R-H the refractory mark by hand on three cases; T2R-B only the four recomputed tables differ from
  the transfer study's commit, and the recomputation and the visibility diagnostic run again byte-identically; T2R-C the evaluation path is still run1's;
  T2R-V the visibility diagnostic's own checks. --part2's T2-B and T2-C record the state before the recomputation and are superseded by these.
Usage   tests/test_transfer.py --part1 | --part2 | --part2r   (writes Data/processed/transfer/part1_tests.csv or part2_tests.csv, exit status 1 on any failure)
It imports the transfer script (which loads the evaluation, training and environment scripts); nothing from Data/processed/checks/.
"""
from __future__ import annotations  # future import: type hints kept as text (the project's convention)

import argparse  # standard library: the command line
import ast  # standard library: the transfer script's syntax tree for T-B and T-D
import importlib.util  # standard library: loads the transfer script (its name starts with a digit)
import shutil  # standard library: removes T-F's scratch folder before a fresh run
import subprocess  # standard library: runs the transfer script's --jobs-sample in its own process (spawned workers must import 13 as a script)
import sys  # standard library: module registry and exit status
from pathlib import Path  # standard library: paths

import numpy as np  # numpy: arrays
import pandas as pd  # pandas: tables

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
_spec = importlib.util.spec_from_file_location("transfer_for_tests", PROJECT / "codes" / "7_evaluation" / "01_transfer.py")  # where 13 is
X = importlib.util.module_from_spec(_spec)  # an empty module object
sys.modules["transfer_for_tests"] = X  # registered
_spec.loader.exec_module(X)  # the transfer script's top level (it extends the environment script's material reader and the training script's digest); its main() is not run
T, E, A, B12 = X.T, X.E, X.A, X.B12  # the training, environment, adopted-pipeline and evaluation scripts as the transfer script holds them
ROWS = []  # the test results


def rec(test: str, ok: bool, detail: str, skipped: str = "") -> None:  # function: records one test (skipped: the reason a check of a superseded state is not run, empty otherwise)
    ROWS.append({"test": test, "pass": bool(ok) or bool(skipped), "detail": ("SKIPPED: " + skipped) if skipped else detail})  # appended (a skipped check is not a failure; its detail says it was skipped and why)
    print(("SKIPPED " if skipped else "PASS " if ok else "FAIL ") + test + " | " + (skipped or detail)[:200], flush=True)  # chained conditional expression: printed as SKIPPED, PASS or FAIL with its reason or detail


def same_points(a: pd.DataFrame, b: pd.DataFrame) -> bool:  # function: two point tables equal on what defines a point, bitwise
    cols = ["receiver", "outer", "detector", "share"]  # the keys
    m = a.merge(b, on=cols, how="outer", suffixes=("_a", "_b"), indicator=True)  # side by side
    if not (m._merge == "both").all():  # if: a point exists in one table only
        return False  # differ
    eq = lambda c: np.array_equal(m[c + "_a"].to_numpy(float), m[c + "_b"].to_numpy(float), equal_nan=True)  # lambda: bitwise equality with NaN = NaN
    return eq("param") and all(eq(c) for c in ("n", "k") if c + "_a" in m and c + "_b" in m)  # the parameter and the k-of-n setting


def t_a() -> None:  # function: T-A1, T-A2, T-A3
    pts = X.rt(B12.EVAL / "deployable_points.csv")  # the agent experiment's committed points
    used = X.rt(X.TRANSFER / "points.csv")  # the points Part A used
    want = pd.concat([pts[pts.receiver == X.SOURCE].assign(receiver=t) for t in X.TARGETS], ignore_index=True)  # the X5's points carried to each target (list comprehension)
    clean = X.git("diff", "--quiet", "HEAD", "--", "Data/processed/rl_agents/evaluation/deployable_points.csv").returncode == 0  # the file is unchanged since its commit
    rec("T-A1 Part A's points = the agent experiment's committed X5 points, bitwise; the file is unchanged", same_points(used, want) and clean, f"{len(used)} points")  # recorded
    sel = X.rt(T.AGENT_DIR / T.GRID / "selection.csv")  # the agent experiment's frozen selection
    nested = X.rt(T.AGENT_DIR / T.GRID / "nested_predictions.csv.gz")  # the agent experiment's nested predictions
    steps = T.read_steps()  # the environment script's step table
    got = []  # recomputed points
    for rx in X.RXS:  # for loop over the receivers
        for h in X.SESSIONS:  # for loop over the agent experiment's outer folds
            p_, _ = X.points_cell(rx, h, [s for s in X.SESSIONS if s != h], E.material(rx, h), nested, steps, sel[(sel.receiver == rx) & (sel.outer == h)])  # the transfer script's code on the agent experiment's material
            got.append(p_)  # kept
    rec("T-A2 points_cell() reproduces the agent experiment's committed points on all 15 the agent experiment cells, bitwise", same_points(pd.concat(got, ignore_index=True), pts), f"{len(pts)} points")  # recorded
    nb = pd.read_csv(X.UNSEEN / "nested_predictions.csv.gz", dtype={"outer_held_out": str, "held_out": str}, float_precision="round_trip")  # Part B's nested predictions
    ixb = pd.read_csv(X.UNSEEN / "state_index.csv.gz", dtype={"session": str})  # Part B's state index
    no12 = X.SEALED not in set(nb.held_out) and X.SEALED not in set(ixb.session) and X.SEALED not in set(steps.session)  # no input holds 2.6.12
    selb = X.rt(X.UNSEEN / "selection.csv")  # Part B's frozen selection
    fz = X.rt(X.UNSEEN / "points.csv")  # Part B's frozen points
    gb = pd.concat([X.points_cell(rx, X.OUTER_B, list(X.SESSIONS), E.material(rx, X.OUTER_B), nb, steps, selb[selb.receiver == rx])[0] for rx in X.RXS], ignore_index=True)  # list comprehension: recomputed
    rec("T-A3 Part B's points: no input holds 2.6.12, and recomputed from the frozen inputs they equal the frozen points", no12 and same_points(gb, fz), f"{len(fz)} points")  # recorded


def t_b() -> None:  # function: T-B
    # T-B1 planted a read of 2.6.12 while the freeze constant was not yet set; with the freeze in place the read is allowed, so this
    # state is not run again: the guard itself is checked by T-B2 here and by the guard tests of --part2 (T2-G)
    rec("T-B1 a planted read of 2.6.12 before the freeze is refused", True, "", skipped="state before the freeze; the guard is checked by T-B2 and by the T2-G tests of --part2")  # recorded as skipped, with the reason
    head = X.git("rev-parse", "HEAD").stdout.decode().strip()  # a commit that is not the freeze
    try:  # try: the guard with the wrong commit
        X.guard(freeze_commit=head)  # the guard
        rec("T-B2 the guard refuses a commit that is not the transfer study freeze", False, "not refused")  # recorded
    except SystemExit as e:  # the refusal
        rec("T-B2 the guard refuses a commit that is not the transfer study freeze", str(e).startswith("REFUSED"), str(e))  # recorded
    tree = ast.parse((PROJECT / "codes" / "7_evaluation" / "01_transfer.py").read_text(encoding="utf-8"))  # the transfer script's syntax tree
    direct = []  # functions that read a window file with pandas directly
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:  # for loop over the functions (list comprehension)
        calls = [c for c in ast.walk(fn) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr == "read_csv"]  # list comprehension: its read_csv calls
        if any("attr='FINAL'" in ast.dump(a) for c in calls for a in c.args):  # if: a read_csv call's own path names the window folder (the first version flagged build_b, which names it only for the regression script's sealed reader)
            direct.append(fn.name)  # recorded
    lits = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and n.value == "2.6.12"]  # list comprehension: the literal session code anywhere
    rec("T-B3 only read_2612 reads a window file directly; no literal 2.6.12 in the transfer script", direct == ["read_2612"] and not lits, f"direct readers {direct}; literals {len(lits)}")  # recorded


def t_c() -> None:  # function: T-C
    mm = pd.read_csv(T.AGENT_DIR / T.GRID / "models_manifest.csv").set_index("model_file")  # the frozen manifest
    sel = X.rt(T.AGENT_DIR / T.GRID / "selection.csv")  # budgets
    files = {T.job_id({"stage": "outer", "receiver": X.SOURCE, "outer": h, "inner": "", "arm": a, "lambda": float(l), "seed": s, "budget": int(sel[(sel.receiver == X.SOURCE) & (sel.outer == h) & (sel.arm == a) & (sel["lambda"] == float(l))].selected_budget.iloc[0])}) + ".npz" for h in X.SESSIONS for a in ("primary", "surrogate") for l in T.LAMBDAS for s in T.OUTER_SEEDS}  # set comprehension: every X5 agent Part A loads
    bad = [f for f in sorted(files) if T.sha_file(B12.MODELS / f) != mm.at[f, "sha256"]]  # list comprehension: mismatches
    rec("T-C every transferred X5 agent equals its frozen weights (SHA-256)", not bad and len(files) == 250, f"{len(files)} files; mismatches {bad}")  # recorded
    ck = pd.read_csv(X.TRANSFER / "checks.csv")  # the transfer script's own checks (the adopted-pipeline script gates: the regression and forest reproduce the committed predictions)
    rec("T-C the X5's regression and forest reproduce the committed predictions (the adopted-pipeline script gates and the transfer script's source check)", bool(ck["pass"].all()), f"{int(ck['pass'].sum())}/{len(ck)} checks")  # recorded


def t_d() -> None:  # function: T-D
    c = pd.read_csv(X.TRANSFER_DIR / "unseen_attack_columns.csv", dtype={"held_out": str})  # the column table of 10_unseen_attack_phases.py
    r = c[c.renamed]  # renamed rows
    ok_map = set(r.target) <= {"UBX_01"} and set(r.scope) <= {"E", "all"} and r.target_codes.str.contains("E:").all() and not r.target_codes.str.contains("E:C1C|E:L1C|E:D1C|E:S1C").any()  # only u-blox 01's Galileo E1, which carries the X codes
    tree = ast.parse((PROJECT / "codes" / "7_evaluation" / "01_transfer.py").read_text(encoding="utf-8"))  # the transfer script's syntax tree
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "target_material")  # next(): the function that builds the target's states
    renames = [n for n in ast.walk(fn) if isinstance(n, ast.Attribute) and n.attr in ("rename", "replace", "set_axis")]  # list comprehension: any renaming call
    rec("T-D renamed columns only u-blox 01's Galileo E1 (C1X..S1X); 13 renames no column", ok_map and not renames, f"{len(r)} renamed rows of {len(c)}; renaming calls in target_material {len(renames)}")  # recorded


def t_e() -> None:  # function: T-E and T-E2
    g = pd.read_csv(X.UNSEEN / "gates.csv")  # the transfer script's gates
    rec("T-E the transfer script's Part B gates (nested = logistic_regression/random_forest, inner primary = the environment, surrogate = the environment held-out, fingerprints)", bool(g["pass"].all()), f"{int(g['pass'].sum())}/{len(g)}")  # recorded
    ix11 = pd.read_csv(E.ENV_DIR / "state_index.csv.gz", dtype={"session": str, "fold": str})  # the environment script's index
    ok, n = True, 0  # independent re-check from the written files
    for rx in X.RXS:  # for loop over the receivers
        b = E.load_npz(X.STATES_B / f"states_{rx}_{X.OUTER_B}.npz")  # Part B's outer states
        for h in X.SESSIONS:  # for loop over the inner folds
            env_states = E.load_npz(E.ENV_DIR / f"states_{rx}_{h}.npz")  # the environment's fold h
            inn = E.load_npz(X.STATES_B / f"inner_states_{rx}_{X.OUTER_B}_{h}.npz")  # Part B's inner states without h
            rr = ix11[(ix11.receiver == rx) & (ix11.fold == h) & (ix11.role == "heldout")].row.to_numpy()  # the environment's held-out rows of fold h
            ok &= np.array_equal(inn["primary"], env_states["primary"]) and np.array_equal(b["surrogate"][rr], env_states["surrogate"][rr])  # bitwise
            n += 1  # counted
    rec("T-E re-checked from the files: inner primary = the environment primary; surrogate rows = the environment held-out rows", ok, f"{n} receiver-fold pairs")  # recorded
    cur = pd.read_csv(X.UNSEEN / "inner_curve.csv.gz", dtype={"outer": str, "inner": str}, float_precision="round_trip")  # Part B's inner snapshots, every float back to the double written (the first run of this test read them with the default parser and failed on 5 of 150)
    sel = X.rt(T.AGENT_DIR / T.GRID / "selection.csv")  # the agent experiment's budgets
    runs = X.rt(B12.EVAL / "heldout_runs.csv")  # the agent experiment's held-out runs
    pairs = [("decision_clean_in_alarm", "dec_clean_in_alarm"), ("decision_recall", "dec_recall"), ("stopping_detected", "stop_detected"), ("stopping_alarms", "stop_alarms"), ("stopping_delay_median_s", "stop_delay_median_s")]  # list: inner-job figure -> the agent experiment run figure
    bad, n = [], 0  # mismatches; comparisons
    for rx in X.RXS:  # for loop over the receivers
        for h in X.SESSIONS:  # for loop over the folds
            for lam in T.LAMBDAS:  # for loop over the lambdas
                b = int(sel[(sel.receiver == rx) & (sel.outer == h) & (sel.arm == "primary") & (sel["lambda"] == float(lam))].selected_budget.iloc[0])  # the agent experiment's budget
                for s in T.INNER_SEEDS:  # for loop over the two inner seeds (also outer seeds of the agent experiment)
                    c = cur[(cur.receiver == rx) & (cur.inner == h) & (cur.arm == "primary") & (cur["lambda"] == float(lam)) & (cur.seed == s) & (cur.budget == b)].iloc[0]  # Part B's snapshot
                    r = runs[(runs.receiver == rx) & (runs.outer == h) & (runs.detector == "agent_primary") & (runs["lambda"] == float(lam)) & (runs.seed == s)].iloc[0]  # the agent experiment's run
                    if not all((pd.isna(c[a]) and pd.isna(r[q])) or float(c[a]) == float(r[q]) for a, q in pairs):  # if: any figure differs
                        bad.append(f"{rx} {h} {lam} {s}")  # recorded
                    n += 1  # counted
    rec("T-E2 Part B primary inner trainings = the agent experiment's frozen outer agents (held-out figures at the agent experiment's budget)", not bad, f"{n} trainings compared on {len(pairs)} figures; mismatches {bad[:5]}")  # recorded


def t_f() -> None:  # function: T-F, a from-scratch run again of 12 Part B jobs
    scratch = X.UNSEEN / "jobs_t_f"  # a separate folder
    if scratch.exists():  # if: left from an earlier run
        shutil.rmtree(scratch)  # removed, so every job runs from scratch
    si = X.specs_inner_b()  # every inner job
    pick = [si[i] for i in np.linspace(0, len(si) - 1, 12).astype(int)]  # 12 evenly spaced jobs (receivers, folds, arms, lambdas, seeds)
    old = X.JOBS_B  # the real folder
    r = subprocess.run([sys.executable, str(PROJECT / "codes" / "7_evaluation" / "01_transfer.py"), "--jobs-sample", str(scratch.relative_to(PROJECT))], cwd=PROJECT)  # the transfer script runs the same 12 jobs in its own spawned workers
    if r.returncode != 0:  # if: the run failed
        raise SystemExit("STOP: T-F's run again failed")  # stop
    bad = [T.job_id(s) for s in pick if (scratch / f"{T.job_id(s)}.json.gz").read_bytes() != (old / f"{T.job_id(s)}.json.gz").read_bytes()]  # list comprehension: outputs that differ
    rec("T-F 12 Part B jobs run again from scratch are byte-identical", not bad, f"{len(pick)} jobs; differing {bad}")  # recorded
    shutil.rmtree(scratch)  # the scratch folder removed


# ----------------------------------------------------------------------------------------------------------------------
# --part2: the tests of the 2.6.12 evaluation's reporting and of the diagnostic
# ----------------------------------------------------------------------------------------------------------------------
U = X.U2612  # the 2.6.12 outputs
SPOT = U / "spotchecks"  # the spot-check plots (not numbered figures)
TMP = PROJECT / "codes" / ".tmp_transfer_tests"  # a temporary folder inside the project removed at the end
RUN1_COMMIT = "24e7cdd"  # the commit that records 7_evaluation/01_transfer.py as run1 ran it
ALLOWED_CHANGED = {"phase_rows", "phase_points", "figures_b", "main"}  # set: the functions the reporting may change
ALLOWED_NEW = {"onset_state", "report_2612", "column_parts", "column_group", "column_meaning", "smd", "hms_of", "diagnose_2612", "diag_context", "plot_ublox_shift"}  # set: the reporting and diagnostic functions it may add


def t2_guard() -> None:  # function: T2-G, the guard refuses a planted read before the freeze (on temporary copies, never on the real state)
    TMP.mkdir(parents=True, exist_ok=True)  # the temporary folder
    keep = X.FREEZE_COMMIT  # the real constant, restored below
    try:  # try: every planted read with the constant emptied in this process only
        X.FREEZE_COMMIT = ""  # as before the freeze (the file on disk is not touched)
        out = TMP / "planted"  # where a refused evaluation would write
        for name, fn in (("evaluate_unseen", lambda: X.evaluate_unseen(X.SEALED, out)), ("diagnose_2612", lambda: X.diagnose_2612(out)), ("read_2612", lambda: X.read_2612("UBX_01"))):  # tuple of (name, lambda): the three paths that could read 2.6.12
            try:  # try: the planted read
                fn()  # the call
                rec(f"T2-G a planted {name} before the freeze is refused", False, "not refused")  # recorded
            except SystemExit as e:  # the refusal
                written = [p.name for p in out.rglob("*") if p.is_file()] if out.exists() else []  # list comprehension: any file written before the refusal
                rec(f"T2-G a planted {name} before the freeze is refused before anything is written", str(e).startswith("REFUSED") and not written, f"{e}; files written {written}")  # recorded
    finally:  # always
        X.FREEZE_COMMIT = keep  # the real constant back
    fm = pd.read_csv(X.UNSEEN / "freeze_manifest.csv")  # the real freeze manifest
    bad = fm.copy()  # a temporary copy
    bad.loc[0, "sha256"] = "0" * 64  # one hash altered
    bad.to_csv(TMP / "manifest_altered.csv", index=False)  # written to the temporary folder only
    try:  # try: the guard on the altered copy
        X.guard(manifest=TMP / "manifest_altered.csv")  # the guard
        rec("T2-G the guard refuses a manifest whose hash does not match (temporary copy)", False, "not refused")  # recorded
    except SystemExit as e:  # the refusal
        rec("T2-G the guard refuses a manifest whose hash does not match (temporary copy)", str(e).startswith("REFUSED"), str(e))  # recorded
    try:  # try: the guard with an earlier commit, which is not the freeze
        X.guard(freeze_commit="f0ade3be6280637346e30895178731acca30352d")  # the guard
        rec("T2-G the guard refuses a commit that is not the transfer study freeze", False, "not refused")  # recorded
    except SystemExit as e:  # the refusal
        rec("T2-G the guard refuses a commit that is not the transfer study freeze", str(e).startswith("REFUSED"), str(e))  # recorded
    tree = ast.parse((PROJECT / "codes" / "7_evaluation" / "01_transfer.py").read_text(encoding="utf-8"))  # the transfer script's syntax tree
    callers = sorted({fn.name for fn in ast.walk(tree) if isinstance(fn, ast.FunctionDef) for c in ast.walk(fn) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == "read_2612"})  # set comprehension: the functions that call the reader
    diag_guard = any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == "guard" for fn in ast.walk(tree) if isinstance(fn, ast.FunctionDef) and fn.name == "diagnose_2612" for c in ast.walk(fn))  # generator in any(): the diagnostic calls the guard itself
    rec("T2-G only evaluate_unseen and diagnose_2612 call read_2612 (which runs the guard); the diagnostic also calls the guard first", callers == ["diagnose_2612", "evaluate_unseen"] and diag_guard, f"callers {callers}")  # recorded


def t2_points() -> None:  # function: T2-P, the frozen points recomputed with every 2.6.12 file removed from the read path
    import builtins  # the built-in open, wrapped for this test
    banned = ("unseen_attack/", "20260914_184225_T_2.6.12", "unseen_attack/")  # tuple: path fragments of every 2.6.12 output and of the raw 2.6.12 folder
    tried = []  # banned paths a read attempted
    o_open, o_csv, o_load = builtins.open, pd.read_csv, np.load  # tuple assignment: the real readers
    def stop_if(p):  # nested function: refuses a banned path
        if any(b in str(p) for b in banned):  # if: the path is a 2.6.12 file
            tried.append(str(p))  # recorded
            raise SystemExit(f"BLOCKED {p}")  # refused
    def w_open(f, *a, **k):  # nested function: open() with the check
        stop_if(f)  # the check
        return o_open(f, *a, **k)  # the real open
    def w_csv(f, *a, **k):  # nested function: read_csv() with the check
        stop_if(f)  # the check
        return o_csv(f, *a, **k)  # the real reader
    def w_load(f, *a, **k):  # nested function: np.load() with the check
        stop_if(f)  # the check
        return o_load(f, *a, **k)  # the real loader
    builtins.open, pd.read_csv, np.load = w_open, w_csv, w_load  # tuple assignment: the wrapped readers installed
    try:  # try: the recomputation with 2.6.12 off the read path
        nb = pd.read_csv(X.UNSEEN / "nested_predictions.csv.gz", dtype={"outer_held_out": str, "held_out": str}, float_precision="round_trip")  # Part B's nested predictions
        steps = T.read_steps()  # the environment script's step table (position sessions)
        selb = X.rt(X.UNSEEN / "selection.csv")  # Part B's frozen selection
        gb = pd.concat([X.points_cell(rx, X.OUTER_B, list(X.SESSIONS), E.material(rx, X.OUTER_B), nb, steps, selb[selb.receiver == rx])[0] for rx in X.RXS], ignore_index=True)  # list comprehension: recomputed
        no12 = X.SEALED not in set(nb.held_out) and X.SEALED not in set(steps.session)  # no row of an input holds 2.6.12
    finally:  # always
        builtins.open, pd.read_csv, np.load = o_open, o_csv, o_load  # the real readers back
    fz = X.rt(X.UNSEEN / "points.csv")  # the frozen points
    rec("T2-P the frozen points recomputed with every 2.6.12 file blocked from the read path are equal, bitwise", same_points(gb, fz) and no12 and not tried, f"{len(fz)} points; blocked reads attempted {tried}")  # recorded


def t2_decomposition() -> None:  # function: T2-D, the decomposition identity, recomputed from the written tables, and the coefficients from a refit
    D = U / "clean_alarm_diagnosis"  # the diagnostic's folder
    DC, ID = X.rt(D / "contributions.csv"), X.rt(D / "identity.csv")  # tuple assignment: contributions and identities
    ID = ID[ID.observed_shift.notna()]  # the references with at least two windows
    s_ = DC.groupby(["receiver", "reference"]).contribution.sum()  # each pair's sum of contributions, recomputed here
    worst = max(abs(s_[(r.receiver, r.reference)] - r.observed_shift) for r in ID.itertuples())  # generator in max(): the largest gap
    prod = np.array_equal(DC.coefficient.to_numpy() * DC.mean_difference.to_numpy(), DC.contribution.to_numpy())  # each contribution is coefficient x mean difference, bitwise
    rec("T2-D the sum of coefficient x mean difference equals the observed shift of the mean decision value (to 1e-9), every receiver and reference", worst <= 1e-9 and prod, f"{len(ID)} pairs; largest gap {worst:.2e}")  # recorded
    sc = X.rt(U / "scores.csv.gz")  # the evaluation's decision values
    gr = pd.read_csv(U / "grid.csv.gz")  # the grid with labels
    m = sc.merge(gr[["receiver", "sec", "label"]], on=["receiver", "sec"])  # scores with labels
    gap = max(abs(m[(m.receiver == r.receiver) & (m.label == "clean")].dec.mean() - r.mean_dec_target) for r in ID.itertuples())  # generator in max(): the target mean recomputed from the scores
    rec("T2-D the diagnostic's mean decision value on 2.6.12's clean seconds = the evaluation's scores (to 1e-12)", gap <= 1e-12, f"largest gap {gap:.2e}")  # recorded
    A.use_labels("final")  # the scoring module points at the final label set
    wc10 = A.window_dict()  # the feature script's window dictionary at w = 10
    cells = A.d1_cells(A.load_cells(A.FINAL, A.W, True), wc10)  # the position windows (the regression script's sealed reader)
    fits = X.rt(X.UNSEEN / "fits.csv")  # the frozen digests
    ok = []  # one result per receiver
    for rx in X.RXS:  # for loop over the receivers
        idx, Xw = cells[rx]  # its windows
        rec5, _ = X.fit_p(idx, Xw, wc10, list(X.SESSIONS), {}, {"run": "13b", "receiver": rx})  # the five-session P refitted (no 2.6.12 window is read)
        fp = X.fingerprint(rec5)  # its digests
        fz = fits[(fits.receiver == rx) & (fits.fit == "five sessions")].iloc[0]  # the frozen ones
        d = DC[(DC.receiver == rx) & (DC.reference == "pooled clean (five sessions)")]  # the written coefficients, in the model's column order
        ok.append(all(fp[k] == fz[k] for k in fp) and d.column.tolist() == list(rec5["names"]) and np.array_equal(d.coefficient.to_numpy(), rec5["model"]["lr"].coef_.ravel()))  # digests, names and coefficients equal
    rec("T2-D the written coefficients are the frozen five-session regression's (refit digests equal, coefficients bitwise)", all(ok), f"{sum(ok)}/{len(ok)} receivers")  # recorded


def hand_series(rx: str, det: str, share: float) -> tuple:  # function: a baseline's decision-mode alarms on 2.6.12's grid, computed by hand from the scores (independent of the evaluation script and the transfer script)
    pts = X.rt(X.UNSEEN / "points.csv")  # the frozen points
    p = pts[(pts.receiver == rx) & (pts.detector == det) & (pts.share == share)].iloc[0]  # the point
    g = pd.read_csv(U / "grid.csv.gz")  # the grids
    g = g[g.receiver == rx].sort_values("sec").reset_index(drop=True)  # this receiver's, in time order
    sc = X.rt(U / "scores.csv.gz")  # the scores
    s = sc[sc.receiver == rx].set_index("sec")  # by second
    sec, ok = g.sec.to_numpy(np.int64), g.decidable.to_numpy(bool)  # tuple assignment: grid seconds and decidability
    if det in ("lr", "rf"):  # the regression or the forest: probability against the threshold on every windowed second
        al = np.array([s.at[t, f"p_{det}"] >= p.param for t in sec])  # list comprehension: one comparison per second
        decided = np.ones(len(sec), bool)  # every windowed second decides
    else:  # k of n: at a decidable second, at least k of the last n decision values reach h
        n, k = int(p.n), int(p.k)  # the setting
        al = np.array([bool(ok[i]) and sum(s.at[t - j, "dec"] >= p.param for j in range(n)) >= k for i, t in enumerate(sec)])  # list comprehension with a generator: the count over the last n seconds
        decided = ok  # only decidable seconds decide
    return sec, ok, al, decided, p  # the arrays and the point


def hand_state(sec, al, decided, t0) -> tuple:  # function: the state at onset by hand: walk back to the last decided second
    i = len(sec) - 1  # start from the end
    while i >= 0 and not (sec[i] < t0 and decided[i]):  # while: not yet a decided second before the first second
        i -= 1  # one step back
    if i < 0:  # none
        return "not decidable", None  # no reference
    age = int(t0 - sec[i])  # its age
    return ("in alarm" if al[i] else "not in alarm") if age <= 60 else "not decidable", age  # the state by the onset rule, and the age


def hand_events(sec, al) -> np.ndarray:  # function: stopping-mode events by hand: an alarm, then 30 s of silence
    ev, last = [], None  # the events and the last one
    for t, a in zip(sec, al):  # for loop over the seconds in time order
        if a and (last is None or t - last > 30):  # if: an alarm past the refractory period
            ev.append(t)  # an event
            last = t  # remembered
    return np.array(ev, dtype=np.int64)  # the event seconds


def t2_onset() -> None:  # function: T2-H, "in alarm at onset" checked by hand on the three cases the specification names
    PH = X.rt(U / "phase_runs.csv")  # the phase rows
    for rx, det, ph, want in (("UBX_01", "lr", "jam_spoof", "in alarm"), ("UBX_01", "kofn", "spoof", "in alarm"), ("SBF_03", "lr", "spoof", "not decidable")):  # tuple of cases with the expected state
        sec, ok, al, decided, p = hand_series(rx, det, 0.05)  # by hand
        r = PH[(PH.receiver == rx) & (PH.detector == det) & (PH.share == 0.05) & (PH.phase == ph)].iloc[0]  # the transfer script's row
        st, age = hand_state(sec, al, decided, int(r.phase_start))  # by hand
        rec(f"T2-H {rx} {det} {ph} at 0.05: state at onset by hand = the transfer script's = {want}", st == r.onset_state == want and age == r.onset_ref_age_s, f"hand {st} (age {age} s); 13 {r.onset_state} (age {r.onset_ref_age_s:.0f} s)")  # recorded


def spot_plot(name: str, sec, ok, al, ev, t0: int, f0, title: str) -> None:  # function: one spot-check plot on both clocks (not a numbered figure)
    import matplotlib.pyplot as plt  # pyplot (09 chose the file-only backend)
    SPOT.mkdir(parents=True, exist_ok=True)  # the folder
    w = (sec >= t0 - 120) & (sec <= t0 + 300)  # two minutes before to five minutes after the phase's first second
    x = sec[w] - t0  # seconds from the phase's first second
    fig, ax = plt.subplots(figsize=(10, 3.2))  # one panel
    ax.bar(x, 0.8, bottom=3, width=1.0, color="#BBBBBB", lw=0, label="windowed second")  # the grid
    ax.bar(x[ok[w]], 0.8, bottom=2, width=1.0, color="#56B4E9", lw=0, label="decidable second")  # decidability
    ax.bar(x[al[w]], 0.8, bottom=1, width=1.0, color="#D55E00", lw=0, label="decision-mode alarm")  # the alarms
    e = ev[(ev >= t0 - 120) & (ev <= t0 + 300)] - t0  # events in view
    ax.vlines(e, 0, 0.8, color="#000000", lw=1.5, label="stopping-mode event")  # the events
    ax.axvline(0, color="#000000", ls="--", lw=0.8, label="phase's first second")  # the first clock's origin
    if f0 is not None:  # if: the phase has a decidable second
        ax.axvline(f0 - t0, color="#56B4E9", ls="-.", lw=1.0, label="phase's first decidable second")  # the second clock's origin
        top = ax.secondary_xaxis("top", functions=(lambda v: v - (f0 - t0), lambda v: v + (f0 - t0)))  # lambdas: the second clock on the top axis
        top.set_xlabel("seconds from the phase's first decidable second", fontsize=7)  # its label
    ax.set_yticks([0.4, 1.4, 2.4, 3.4], ["events", "alarm", "decidable", "window"], fontsize=7)  # the rows
    ax.set_xlabel("seconds from the phase's first second", fontsize=7)  # the first clock
    ax.set_title(title, fontsize=8, loc="left")  # title
    ax.legend(fontsize=6, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28))  # legend
    fig.tight_layout()  # layout
    X.M.save_fig(fig, SPOT / name)  # byte-stable save


def t2_spot() -> None:  # function: T2-S, three delays against the per-second stream, plotted on both clocks
    PH = X.rt(U / "phase_runs.csv")  # the phase rows
    for rx, det, ph, fname in (("UBX_01", "lr", "jam_spoof", "spot_ublox_lr_jam_spoof.png"), ("SBF_03", "lr", "spoof", "spot_mosaicH_lr_spoof.png")):  # tuple of the two baseline cases
        sec, ok, al, decided, p = hand_series(rx, det, 0.05)  # by hand
        r = PH[(PH.receiver == rx) & (PH.detector == det) & (PH.share == 0.05) & (PH.phase == ph)].iloc[0]  # the transfer script's row
        t0 = int(r.phase_start)  # the phase's first second
        f0 = int(sec[(sec >= t0) & ok][0]) if ((sec >= t0) & ok).any() else None  # conditional expression: the first decidable second at or after the start
        ev = hand_events(sec, al)  # events by hand
        ev_d = hand_events(sec[ok], al[ok])  # the decidable-only reading's events by hand
        e1 = ev[ev >= t0]  # events from the phase start
        e2 = ev_d[ev_d >= t0]  # the decidable-only events from the phase start
        d1, d2 = int(e1[0] - t0), int(e2[0] - t0)  # tuple assignment: both delays from the phase's first second
        okv = d1 == r.delay_s and (f0 - t0) == r.first_decidable_s and (e1[0] - f0) == r.delay_from_first_decidable_s and d2 == r.dec_only_delay_s  # every figure equal
        rec(f"T2-S {rx} {det} {ph} at 0.05: delays by hand from the stream (phase start {d1} s, first decidable second {e1[0] - f0} s, decidable-only {d2} s) = the transfer script's", okv, f"13: {r.delay_s:.0f} / {r.delay_from_first_decidable_s:.0f} / {r.dec_only_delay_s:.0f} s")  # recorded
        spot_plot(fname, sec, ok, al, ev, t0, f0, f"{X.NAME[rx]}, {det} at the 0.05 point, {ph}: first event +{d1} s from the phase start, {e1[0] - f0:+d} s from the first decidable second")  # the plot
    DEC = pd.read_csv(U / "decisions.csv.gz")  # the recorded per-second decisions
    a = DEC[(DEC.receiver == "SBF_03") & (DEC.detector == "agent_primary") & (DEC["lambda"] == 10.0) & (DEC.seed == T.SEED)].sort_values("sec")  # one mosaic-H agent seed at its 0.05 point (lambda 10)
    sec, ok, st_in = a.sec.to_numpy(np.int64), a.decidable.to_numpy(bool), a.stop_input.to_numpy() > 0  # tuple assignment: its seconds, decidability and stopping-mode input
    R = X.rt(U / "runs.csv")  # the registered runs
    rr = R[(R.receiver == "SBF_03") & (R.detector == "agent_primary") & (R["lambda"] == 10.0) & (R.seed == T.SEED)].iloc[0]  # its registered run
    fd, fs = B12.both(a.assign(session=X.SEALED), a.dec.to_numpy(), a.stop_input.to_numpy(), X.SEALED)  # the environment script's stream on the recorded decisions
    same_run = all((pd.isna(v) and pd.isna(rr[k])) or v == rr[k] for k, v in B12.run_row(fd, fs).items())  # generator in all(): the recorded decisions reproduce the registered run
    r = PH[(PH.receiver == "SBF_03") & (PH.detector == "agent_primary") & (PH["lambda"] == 10.0) & (PH.seed == T.SEED) & (PH.phase == "spoof")].iloc[0]  # the transfer script's phase row
    t0 = int(r.phase_start)  # the phase's first second
    f0 = int(sec[(sec >= t0) & ok][0])  # its first decidable second
    ev = hand_events(sec, st_in)  # events by hand
    e1 = ev[ev >= t0]  # from the phase start
    okv = same_run and int(e1[0] - t0) == r.delay_s and int(e1[0] - f0) == r.delay_from_first_decidable_s  # the run and both delays
    rec(f"T2-S mosaic-H agent (primary, lambda 10, seed {T.SEED}) spoof: recorded decisions reproduce the registered run; delays by hand {int(e1[0] - t0)} s / {int(e1[0] - f0)} s = the transfer script's", okv, f"13: {r.delay_s:.0f} / {r.delay_from_first_decidable_s:.0f} s")  # recorded
    spot_plot("spot_mosaicH_agent_primary_seed20260914_spoof.png", sec, ok, a.dec.to_numpy() > 0, ev, t0, f0, f"mosaic-H, agent (primary, lambda 10, seed {T.SEED}), time spoofing: first event +{int(e1[0] - t0)} s from the phase start, +{int(e1[0] - f0)} s from the first decidable second")  # the plot


def t2_confined() -> None:  # function: T2-C, the code change since run1 is confined to the listed reporting and diagnostic functions
    # T2-C listed the functions the transfer script could hold right after run2 and run3; the recomputation and the visibility diagnostic
    # added functions after that, so this state is not run again: T2R-C checks that the evaluation path is still run1's
    rec("T2-C since run1 (24e7cdd) only phase_rows, phase_points, figures_b and main changed, nothing was removed, and only reporting and diagnostic functions were added (evaluate_unseen, guard, read_2612 identical)", True, "", skipped="state before the recomputation; the evaluation path is checked by T2R-C of --part2r")  # recorded as skipped, with the reason


def t2_bytes() -> None:  # function: T2-B, the byte rules of the runs again
    # the F1 rule (a run without an alarm has F1 = 0) was applied to the 2.6.12 tables after these runs: runs, point_runs,
    # runs_decidable and aggregate are expected as Data/processed/transfer/f1_rule_expected/ holds them, which differ from run1's
    # copies and from run3's tables only by that rule (dec_f1 empty -> 0.0 and the F1 rows of the aggregate that follow from it)
    f1 = X.TRANSFER_DIR / "f1_rule_expected"  # the expected tables under the F1 rule
    untouched = ["guard", "checks", "runs", "latency", "point_runs", "aggregate", "verdicts", "curves", "windows", "outcome"]  # list: run1's .csv outputs the change does not touch
    files = [f"{x}.csv" for x in untouched] + [f"{x}.csv.gz" for x in ("curve_runs", "grid", "scores")]  # list comprehensions: the 13 files
    expect = lambda x: (f1 / x) if (f1 / x).exists() else (X.TRANSFER_DIR / "unseen2612_run1" / x)  # lambda: the expected copy (the F1 rule's table where there is one, else run1's copy)
    diff = [x for x in files if (U / x).read_bytes() != expect(x).read_bytes()]  # list comprehension: any that differ from the expectation
    rec("T2-B every run1 output the change does not touch equals run1 byte for byte (13 files; the F1 rule applied)", not diff, f"differing {diff}")  # recorded
    h2 = (X.TRANSFER_DIR / "logs" / "part2_run2_sha256.txt").read_text().splitlines()  # run2's hashes (taken before run3)
    h3 = (X.TRANSFER_DIR / "logs" / "part2_run3_sha256.txt").read_text().splitlines()  # run3's hashes
    later = {ln.split("  ", 1)[1] for tag in ("recompute2612", "visibility") for ln in (X.TRANSFER_DIR / "logs" / f"{tag}_run2_sha256.txt").read_text().splitlines()}  # set comprehension: files the recomputation and the visibility diagnostic wrote after run3 (T2R-B checks them against their own runs)
    want = {ln.split("  ", 1)[1]: (T.sha_file(f1 / Path(ln.split("  ", 1)[1]).name) if (f1 / Path(ln.split("  ", 1)[1]).name).exists() else ln.split("  ", 1)[0]) for ln in h3}  # dict comprehension: run3's hash of each file, or the F1 rule's table's
    now = {p: T.sha_file(PROJECT / p) for p in want if p not in later}  # dict comprehension: the files as they are now, hashed again (those rewritten later left to T2R-B)
    rec("T2-B every output is equal between run2 and run3 (evaluation, reporting, diagnostic, the figures), and the files now are run3's (the F1 rule applied; the recomputation's files checked by T2R-B)", h2 == h3 and all(now[p] == want[p] for p in now), f"{len(h3)} files; {len(now)} compared now, {len(h3) - len(now)} left to T2R-B")  # recorded


def main_part2() -> None:  # function: the --part2 tests
    import shutil as _sh  # file-tree removal for the temporary folder
    for fn in (t2_guard, t2_points, t2_decomposition, t2_onset, t2_spot, t2_confined, t2_bytes):  # for loop over the tests
        fn()  # run
    if TMP.exists():  # if: the temporary folder exists
        _sh.rmtree(TMP)  # removed
    E.write_csv(pd.DataFrame(ROWS), X.TRANSFER_DIR / "part2_tests.csv")  # written
    print(f"the transfer tests --part2:{sum(r['pass'] for r in ROWS)}/{len(ROWS)} pass")  # summary
    if not all(r["pass"] for r in ROWS):  # if: any failed
        sys.exit(1)  # exit status 1


def t2r_refractory() -> None:  # function: T2R-H the refractory mark checked by hand on three cases from the written decisions
    DEC = X.rt(U / "decisions.csv.gz")  # the per-second decisions
    PH = X.rt(U / "phase_runs.csv")  # the recomputed phase rows
    for rx, det, sh, ph, want in (("UBX_01", "lr", 0.02, "jam_spoof", True), ("SBF_03", "cusum", 0.10, "jam_spoof", True), ("SBF_03", "lr", 0.05, "jam_spoof", False)):  # tuple of cases: an event 3 s before, one exactly 30 s before (still blocked), and one not refractory
        d = DEC[(DEC.receiver == rx) & (DEC.detector == det) & (DEC.share == sh)].sort_values("sec")  # the run's seconds
        r = PH[(PH.receiver == rx) & (PH.detector == det) & (PH.share == sh) & (PH.phase == ph)].iloc[0]  # its phase row
        ev = hand_events(d.sec.to_numpy(np.int64), d.stop_input.to_numpy() > 0)  # events by hand from the stopping-mode input
        before = ev[ev < int(r.phase_start)]  # those before the phase's first second
        age = int(r.phase_start - before.max()) if len(before) else None  # the last one's age
        hand = age is not None and age <= 30  # refractory by hand (a new event needs more than 30 s)
        rec(f"T2R-H {rx} {det} {sh:g} {ph}: refractory at onset by hand = the transfer script's = {want} (last event {age} s before)", hand == bool(r.refractory_at_onset) == want and (age is None or age == r.last_event_age_s), f"13: {r.refractory_at_onset}, age {r.last_event_age_s}")  # recorded


def t2r_bytes() -> None:  # function: T2R-B, the recomputation's byte rules
    # the F1 rule (a run without an alarm has F1 = 0) was applied after the transfer study's commit: runs, point_runs, runs_decidable
    # and aggregate differ from that commit by it alone, and each must equal the table Data/processed/transfer/f1_rule_expected/ holds
    changed = {"phase_runs.csv", "phase_points.csv", "onset_states.csv", "verdicts_annotated.csv"}  # set: the files the recomputation writes
    f1 = X.TRANSFER_DIR / "f1_rule_expected"  # the expected tables under the F1 rule
    f1_files = {p.name for p in f1.glob("*.csv")}  # set comprehension: the tables the rule changed
    files = X.git("ls-tree", "-r", "--name-only", X.PART_B_COMMIT, "Data/processed/transfer/unseen_attack").stdout.decode().split()  # every file committed at the transfer study
    diff = [f for f in files if X.git("show", f"{X.PART_B_COMMIT}:{f}").stdout != (PROJECT / f).read_bytes()]  # list comprehension: files that differ now
    f1_same = all((U / x).read_bytes() == (f1 / x).read_bytes() for x in f1_files)  # bool: each table the rule changed equals its expected table
    rec("T2R-B only the four recomputed tables and the four tables of the F1 rule differ from the transfer study's commit, the latter as expected; every other 2.6.12 output is byte-identical", {Path(f).name for f in diff} == changed | f1_files and f1_same, f"{len(files)} committed files; differing {sorted(Path(f).name for f in diff)}")  # recorded
    for tag in ("recompute2612", "visibility"):  # for loop over the two runs again
        a, b = [(X.TRANSFER_DIR / "logs" / f"{tag}_run{i}_sha256.txt").read_text() for i in (1, 2)]  # list comprehension: both runs' hashes
        now = "".join(f"{T.sha_file(PROJECT / ln.split('  ', 1)[1])}  {ln.split('  ', 1)[1]}\n" for ln in b.splitlines())  # generator in join(): the files as they are now
        rec(f"T2R-B the {tag} run again is byte-identical and the files now are its outputs", a == b == now, f"{len(b.splitlines())} files")  # recorded
    old = X.git("show", f"{RUN1_COMMIT}:codes/7_evaluation/01_transfer.py").stdout.decode()  # the transfer script as run1 ran it
    f = lambda src: {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}  # lambda: top-level function -> syntax tree
    o, n = f(old), f((PROJECT / "codes" / "7_evaluation" / "01_transfer.py").read_text(encoding="utf-8"))  # tuple assignment: both
    rec("T2R-C the evaluation path is still run1's: evaluate_unseen, guard, read_2612, fit_p, steps_of and tracking_outcome identical to 24e7cdd", all(o[k] == n[k] for k in ("evaluate_unseen", "guard", "read_2612", "fit_p", "steps_of", "tracking_outcome")), "")  # recorded
    vc = X.rt(U / "clean_alarm_diagnosis" / "visibility_checks.csv")  # the visibility diagnostic's own checks
    rec("T2R-V the visibility diagnostic's checks pass (IGS file hash, GPS and GLONASS elevations against the observables script's and the GLONASS orbit script's tables)", bool(vc["pass"].all()), f"{int(vc['pass'].sum())}/{len(vc)}")  # recorded


def main_part2r() -> None:  # function: the tests of the recomputation and the visibility diagnostic
    for fn in (t2r_refractory, t2r_bytes):  # for loop over the tests
        fn()  # run
    E.write_csv(pd.DataFrame(ROWS), X.TRANSFER_DIR / "part2r_tests.csv")  # written
    print(f"the transfer tests --part2r:{sum(r['pass'] for r in ROWS)}/{len(ROWS)} pass")  # summary
    if not all(r["pass"] for r in ROWS):  # if: any failed
        sys.exit(1)  # exit status 1


def main() -> None:  # function: the command line
    ap = argparse.ArgumentParser()  # the parser
    ap.add_argument("--part1", action="store_true")  # the --part1 tests
    ap.add_argument("--part2", action="store_true")  # the --part2 tests
    ap.add_argument("--part2r", action="store_true")  # the tests of the recomputation and the visibility diagnostic
    a = ap.parse_args()  # parsed
    if a.part2r:  # the recomputation's tests
        main_part2r()  # run (writes part2r_tests.csv)
        return  # done
    if a.part2:  # the --part2 tests
        main_part2()  # run (writes part2_tests.csv)
        return  # done
    if not a.part1:  # nothing chosen
        raise SystemExit("usage: tests/test_transfer.py --part1 | --part2 | --part2r")  # stop
    for f in (t_a, t_b, t_c, t_d, t_e, t_f):  # for loop over the tests
        f()  # run
    E.write_csv(pd.DataFrame(ROWS), X.TRANSFER_DIR / "part1_tests.csv")  # written
    print(f"transfer tests: {sum(r['pass'] for r in ROWS)}/{len(ROWS)} pass")  # summary
    if not all(r["pass"] for r in ROWS):  # if: any failed
        sys.exit(1)  # exit status 1


if __name__ == "__main__":  # guard: runs main() only when the file is executed (spawned workers import 13, not this file)
    main()  # runs main
