#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
Transfer (RQ3): across hardware (Part A) and to the unseen attack type 2.6.12 (Part B).

  --transfer  Part A. For each outer fold h, the mosaic-X5's fitted P (transform, logistic regression, forest), its CUSUM and k-of-n
              points and its 50 frozen agents are applied unchanged to the held-out session h of u-blox 01 (primary) and of the
              mosaic-H main antenna (secondary). The target's windows go through the X5 transform BY NAME: every column name exists on all
              three receivers, and the u-blox's Galileo E1 columns were built by the window-feature script from C1X..S1X (the renaming map of
              1_ground_truth/10_unseen_attack_phases.py); no column is renamed here. Every detector at the X5's committed deployable points;
              the agent experiment's metrics, IQM intervals (P8) and verdicts (P9, the agent against the best transferred baseline); the
              target's own agent results beside them; the figures.
  --unseen    Part B, training and freezing (no 2.6.12 read). Per receiver (u-blox 01 primary; mosaic-H and mosaic-X5 secondary), P fitted
              on all five position sessions; nested leave-one-session-out over the five (each inner fit on four sessions is P's outer fit
              of the environment and the agent experiment, gated bitwise); the inner loop of the training script (5 inner folds x 2 seeds
              x 2 arms x 5 lambda, 100,000 steps, snapshots every 10,000) run by its own job runner on this material; the budget by the
              mean inner criterion; P10 on the inner folds; five outer seeds; deployable points (P6 as amended); P11; a freeze manifest.
              The freeze commit follows.
  --heldout2612  the single 2.6.12 evaluation. The guard first (every frozen file equal to the freeze commit), then the five-session
              refits must reproduce the frozen digests, then 2.6.12's windows are read once per receiver; every detector at its Part B point
              (agents: five seeds; every lambda for the curves), both modes through the environment script's stream with the label script's
              labels and fold policy, the delay per phase, the real windows beside the inferred ones, the loss of tracking as an outcome;
              the figures.
  --dryrun S  the same path on the training session S (no 2.6.12 read), into unseen2612_dryrun/, proving the plumbing first.
  --recompute2612  reporting and diagnostic code only (evaluate_unseen() and every function it scores with are unchanged since run1):
              phase_rows() adds the state at onset (in alarm, not in alarm, not decidable; the last decided second at most 60 s back, its
              age stored), the delay from the first decidable second, the decidable-only reading of the regression and the forest, and the
              minute before; report_2612() writes the realised shares with the "point not held" mark, the decidable-only runs, the states
              at the interval onset, the per-second decisions and the annotated verdict table (the registered verdicts.csv is untouched);
              the figures redrawn.
  --diag2612  the diagnostic of u-blox 01's shift on 2.6.12's clean seconds (finding F1), the mosaic-H and the mosaic-X5 as controls:
              standardised mean differences, the exact decomposition of the decision-value shift, level and change columns, 2.6.11 and
              2.6.5 at Stave as comparisons, the clock, satellite and capture context; the figure of the shift (Graph/ublox_shift.png). Nothing is chosen from it.
2.6.12 is sealed: the regression script's reader drops it from every window file, and the one function that may read it (read_2612)
refuses unless every frozen file equals its copy in the transfer study's freeze commit (FREEZE_COMMIT). It imports the evaluation script
(which loads the training, environment, adopted-pipeline, forest and regression scripts).
Reads   what the adopted-pipeline, environment and training scripts read; Data/processed/rl_agents/ (frozen points, selection, models),
        Data/processed/transfer/unseen_attack_columns.csv
Writes  Data/processed/transfer/receivers/*, Data/processed/transfer/unseen_attack_training/*, Data/processed/transfer/models/unseen_attack_training/*,
        Data/processed/transfer/unseen_attack/*
"""
from __future__ import annotations  # future import: type hints kept as text (the project's convention)

import argparse  # standard library: the command line
import hashlib  # standard library: SHA-256 fingerprints
import importlib.util  # standard library: loads the evaluation script (its name starts with a digit)
import json  # standard library: the digest's input as text
import multiprocessing  # standard library: the spawn context of the worker pool
import subprocess  # standard library: read-only git commands (the freeze guard)
import sys  # standard library: module registry and exit status
import time  # standard library: wall-clock timing (timing table only)
from concurrent.futures import ProcessPoolExecutor, as_completed  # standard library: the pool of worker processes
from pathlib import Path  # standard library: paths

import numpy as np  # numpy: arrays
import pandas as pd  # pandas: tables

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
_spec = importlib.util.spec_from_file_location("evaluate_agents_for_transfer", PROJECT / "codes" / "6_reinforcement_learning" / "03_evaluate_agents.py")  # where the evaluation script is
B12 = importlib.util.module_from_spec(_spec)  # an empty module object
sys.modules["evaluate_agents_for_transfer"] = B12  # registered, so its own references resolve
_spec.loader.exec_module(B12)  # the evaluation script's top level (it loads the training script, which loads the environment script and the adopted-pipeline script and sets torch to one thread); no main() is run
T, E = B12.T, B12.E  # the training script and the environment script
A, M = E.A, E.M  # the adopted-pipeline script and the regression script

TRANSFER_DIR = E.PROC / "transfer"  # this section's folder
TRANSFER = TRANSFER_DIR / "receivers"  # Part A's outputs
UNSEEN = TRANSFER_DIR / "unseen_attack_training"  # Part B's outputs
STATES_B = UNSEEN / "states"  # Part B's state files
JOBS_B = UNSEEN / "jobs"  # Part B's job outputs
MODELS_B = TRANSFER_DIR / "models" / "unseen_attack_training"  # Part B's frozen agents (small; committed, SHA-256 in the manifest)
SEALED = A.SEALED  # "2.6.12"
OUTER_B = SEALED  # Part B's single outer fold is named by its held-out session, 2.6.12, which no table here holds
SESSIONS, RXS = T.SESSIONS, T.RXS  # the five position sessions in the screen script's order; the three receivers
SOURCE = "SBF_02"  # Part A trains on the mosaic-X5
TARGETS = ("UBX_01", "SBF_03")  # Part A's targets: u-blox 01 (primary), mosaic-H main antenna (secondary)
NAME = E.NAME  # stream -> receiver name
FREEZE_COMMIT = "17d3f37da4d6f1dc88adccb5450969b5dc676c43"  # the transfer study FREEZE commit (set before --heldout2612; the guard checks every frozen file against it)
PART_B_COMMIT = "f4a9e2933b259f32cce47c89e6e0e86eced5d2ae"  # the commit of the first 2.6.12 tables, against which their recomputation is checked
N_WORKERS = T.N_WORKERS  # at most three worker processes
GRAPH = E.GRAPH  # the figure folder


# ----------------------------------------------------------------------------------------------------------------------
# Part B material for the training script's job runner: the environment script's material reader and the training script's digest are extended to the outer fold 2.6.12 only
# ----------------------------------------------------------------------------------------------------------------------
_MATERIAL_11 = E.material  # the environment script's own reader of the position folds' material, kept for every other fold
_DIGEST_12 = T.spec_digest  # the training script's own job digest, kept for every other fold


def material_b(rx: str, outer: str) -> dict:  # function: the environment script's material() for Part B (all five position sessions are training sessions); any other fold goes to the environment script's reader
    if outer != OUTER_B:  # if: a position fold
        return _MATERIAL_11(rx, outer)  # the environment script's material, unchanged
    mats = E.load_npz(STATES_B / f"states_{rx}_{OUTER_B}.npz")  # the five-session P's primary states and the nested surrogate states
    ix = pd.read_csv(UNSEEN / "state_index.csv.gz", dtype={"session": str})  # the state index
    ix = ix[ix.receiver == rx].sort_values("row").reset_index(drop=True)  # this receiver's rows, position = matrix row
    if not (ix.row.to_numpy() == np.arange(len(ix))).all() or len(ix) != len(mats["primary"]):  # if: the index does not cover the matrix row by row
        raise SystemExit(f"STOP: Part B state index and matrix disagree for {rx}")  # stop
    on = pd.read_csv(E.ENV_DIR / "onsets.csv", dtype={"session": str})  # the environment script's attack runs (they do not depend on the fold)
    on = on[(on.receiver == rx) & on.usable].reset_index(drop=True)  # the usable onsets of the five sessions, in the environment script's order
    key = {(s, int(t)): int(r) for s, t, r in zip(ix.session, ix.sec, ix.row)}  # dict comprehension: (session, second) -> row, every row (all are training rows)
    # dict literal over two lines: the material, field for field as the environment script's material() builds it
    return {"receiver": rx, "fold": OUTER_B, "primary": mats["primary"], "surrogate": mats["surrogate"], "index": ix, "key": key, "onsets": on, "m": float(np.median(on.length_s)),
            "type": ix.step.map(E.STEP).to_numpy(np.int8), "run": ix.run_id.to_numpy(), "session": ix.session.to_numpy(), "sec": ix.sec.to_numpy(np.int64), "pre_rows": ix.row[ix.step == "pre"].to_numpy()}  # dict literal continued


def material_code() -> str:  # function: SHA-256 of material_b()'s syntax tree, the only code of this file a training job depends on; for this folder's copy, the digest of material_b as the analysis ran it
    import ast, inspect, textwrap  # standard library: syntax trees, source of a function, indentation removal
    own = hashlib.sha256(ast.dump(ast.parse(textwrap.dedent(inspect.getsource(material_b))), include_attributes=False).encode()).hexdigest()  # the digest of the tree's text
    from helpers import analysis_code_digest  # the table of the analysis's code digests (codes/common/helpers.py)
    return analysis_code_digest(PROJECT, Path(__file__), own, "material_b")  # the analysis's digest when the table lists material_b with exactly this tree, else its own


def spec_digest_b(s: dict, states: Path) -> str:  # function: the training script's job digest, with Part B's files and the transfer script's code for the outer fold 2.6.12
    if s["outer"] != OUTER_B:  # if: a position fold
        return _DIGEST_12(s, states)  # the training script's digest, unchanged
    files = [STATES_B / f"states_{s['receiver']}_{OUTER_B}.npz", UNSEEN / "state_index.csv.gz", E.ENV_DIR / "onsets.csv"]  # list: the files the material reads
    if s["inner"] and s["arm"] == "primary":  # an inner primary job also reads its inner states
        files.append(states / f"inner_states_{s['receiver']}_{OUTER_B}_{s['inner']}.npz")  # that file
    # dict literal over two lines: the digest's input, as 12 builds it
    d = {"spec": s, "dqn": T.DQN_KW, "schedule": T.SCHEDULE, "budgets": T.BUDGETS, "scale": T.REWARD_SCALE, "eval": [T.ONSET_EVERY, T.CLEAN_EVERY], "sb3": T.sb3.__version__, "torch": T.torch.__version__,
         "files": [T.sha_file(f) for f in files], "code": [T.sha_code(PROJECT / "codes" / n) for n in ("6_reinforcement_learning/01_environment.py", "6_reinforcement_learning/02_train_agents.py")] + [material_code()]}  # list comprehensions: file and code digests; of 13, only the material reader a job depends on
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()  # the digest


E.material = material_b  # the environment script's reader extended (module attribute; the training script's job_material, implied() and 13 look it up at call time); done on import, so spawned workers have it too
T.spec_digest = spec_digest_b  # the training script's digest extended the same way


def run_job_b(spec: dict, states: str, jobs: str, models: str) -> dict:  # function: one training through the training script's own runner (a top-level function here, so spawned workers can find it)
    return T.run_job(spec, states, jobs, models)  # the training script's run_job, unchanged


# ----------------------------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------------------------
def git(*args) -> subprocess.CompletedProcess:  # function: a read-only git command in the folder; without a repository, the frozen-input table answers it
    from helpers import frozen_git, in_repository  # the table's reader and the repository test (codes/common/helpers.py, on the import path)
    if in_repository(PROJECT):  # if: the folder sits in a git repository with this history
        return subprocess.run(["git", *args], cwd=PROJECT, capture_output=True)  # its result (bytes)
    return frozen_git(PROJECT, args)  # the answer from Data/processed/checks/frozen_inputs.csv, in the same form


def rt(path: Path, **kw) -> pd.DataFrame:  # function: a committed table read with every float back to the double that was written
    return B12.rt(path, **kw)  # the evaluation script's reader


def lagged(Z: np.ndarray, secs: np.ndarray, need: np.ndarray, lags) -> np.ndarray:  # function: rows of Z (one per window second in secs, sorted) at need - lag for each lag, side by side
    out = []  # one block per lag
    for lag in lags:  # for loop over the lags
        k = np.searchsorted(secs, need - lag)  # positions by binary search
        if (k >= len(secs)).any() or (secs[np.minimum(k, len(secs) - 1)] != need - lag).any():  # if: a needed window does not exist (validity forbids it)
            raise SystemExit("STOP: a state needs a window that does not exist")  # stop
        out.append(Z[k] if Z.ndim == 2 else Z[k][:, None])  # conditional expression: a matrix block, or a column for a vector
    return np.hstack(out)  # side by side


def fingerprint(rec: dict) -> dict:  # function: SHA-256 of a fitted P (transform, regression, forest), so a refit can be proved identical
    p = rec["prep"]  # the fitted preparation
    tr = hashlib.sha256(np.ascontiguousarray(p["med"]).tobytes() + np.ascontiguousarray(p["scaler"][0]).tobytes() + np.ascontiguousarray(p["scaler"][1]).tobytes() + ";".join(rec["names"]).encode()).hexdigest()  # the environment script's fingerprint rule
    lr = rec["model"]["lr"]  # the regression
    h_lr = hashlib.sha256(np.ascontiguousarray(lr.coef_).tobytes() + np.ascontiguousarray(lr.intercept_).tobytes()).hexdigest()  # its coefficients
    h = hashlib.sha256()  # the forest's digest, tree by tree
    for est in rec["model"]["rf"].estimators_:  # for loop over the trees
        t_ = est.tree_  # the tree's arrays
        h.update(np.ascontiguousarray(t_.feature).tobytes() + np.ascontiguousarray(t_.threshold).tobytes() + np.ascontiguousarray(t_.value).tobytes())  # split features, thresholds and leaf values
    return {"transform_sha256": tr, "lr_sha256": h_lr, "rf_sha256": h.hexdigest()}  # dict literal: the three digests


# ----------------------------------------------------------------------------------------------------------------------
# Part A: the X5's fitted P and agents on a target's held-out session
# ----------------------------------------------------------------------------------------------------------------------
def target_material(Bp: dict, rx_t: str, held: str, steps: pd.DataFrame, m: float) -> tuple:  # function: the X5 fold's states and scores on a target's held-out session: (material, probabilities by second)
    rec = Bp["outer"]["kept"][(SOURCE, held)]  # the X5's outer fit of this fold
    idx, X = Bp["cells"][rx_t]  # the target's windows (D1 columns, every position session)
    te = A.eval_rows(idx, held)  # the held-out session's windows in time order
    Z = A.transform(rec, X.iloc[te], Bp["wc10"])  # the X5 transform, by column name, on the target's windows (C order)
    sc = A.score(rec, Z)  # the X5 regression's decision value and probability, the X5 forest's probability
    secs = idx.sec.to_numpy(np.int64)[te]  # the windows' seconds (sorted)
    v = steps[(steps.receiver == rx_t) & (steps.session == held) & steps.valid].sort_values("sec").reset_index(drop=True)  # the target's valid seconds (its own window runs)
    need = v.sec.to_numpy(np.int64)  # their seconds
    P = lagged(Z, secs, need, E.LAGS)  # primary: windows ending t, t-10, t-20
    S = lagged(sc["dec"], secs, need, range(E.HIST, -1, -1))  # surrogate: z at t-29 .. t, oldest first
    ix = v[["session", "sec", "label", "step", "run_id"]].assign(role="heldout", row=np.arange(len(v)))  # the index
    key = {(held, int(t)): int(r) for t, r in zip(ix.sec, ix.row)}  # dict comprehension: (session, second) -> row
    # dict literal over two lines: the material, field for field as the evaluation script's heldout_material() gives it (m only builds the environment; no reward is read)
    mat = {"receiver": rx_t, "fold": held, "primary": P, "surrogate": S, "index": ix, "key": key, "onsets": pd.DataFrame(columns=["session", "onset", "end", "start_lo", "start_hi", "length_s"]), "m": m,
           "type": ix.step.map(E.STEP).to_numpy(np.int8), "run": ix.run_id.to_numpy(), "session": ix.session.to_numpy(), "sec": ix.sec.to_numpy(np.int64), "pre_rows": ix.row[ix.step == "pre"].to_numpy()}  # dict literal continued
    pred = {"lr": pd.Series(sc["p_lr"], index=secs), "rf": pd.Series(sc["p_rf"], index=secs)}  # dict literal: the probabilities by second
    return mat, pred, sc  # material, probabilities, raw scores


def source_check(Bp: dict, steps: pd.DataFrame, folds: pd.DataFrame) -> list:  # function: target_material() run on the X5 itself must give the environment's held-out states and P's committed probabilities bitwise
    rows = []  # one row per fold
    lr_tab = rt(E.PROC / "logistic_regression" / "logistic_regression_predictions_primary.csv.gz")  # the committed regression probabilities
    rf_tab = rt(E.PROC / "random_forest" / "random_forest_predictions_primary.csv.gz")  # the committed forest probabilities
    ix = pd.read_csv(E.ENV_DIR / "state_index.csv.gz", dtype={"session": str, "fold": str})  # the environment script's state index
    for h in SESSIONS:  # for loop over the folds
        m = float(folds[(folds.receiver == SOURCE) & (folds.held_out == h)].m.iloc[0])  # the X5 fold's miss penalty
        mat, pred, _ = target_material(Bp, SOURCE, h, steps, m)  # on the X5 itself
        z = E.load_npz(E.ENV_DIR / f"states_{SOURCE}_{h}.npz")  # the environment's states of the fold
        r = ix[(ix.receiver == SOURCE) & (ix.fold == h) & (ix.role == "heldout")].sort_values("sec").row.to_numpy()  # its held-out rows in time order
        c9 = lr_tab[(lr_tab.receiver == SOURCE) & (lr_tab.held_out == h)].set_index("sec").p  # committed regression probabilities
        c10 = rf_tab[(rf_tab.receiver == SOURCE) & (rf_tab.held_out == h)].set_index("sec").p  # committed forest probabilities
        ok = np.array_equal(mat["primary"], z["primary"][r]) and np.array_equal(mat["surrogate"], z["surrogate"][r]) and np.array_equal(pred["lr"].to_numpy(), c9.reindex(pred["lr"].index).to_numpy()) and np.array_equal(pred["rf"].to_numpy(), c10.reindex(pred["rf"].index).to_numpy())  # bitwise
        rows.append({"check": f"target_material on the X5 = the environment held-out states and committed probabilities ({h})", "pass": bool(ok), "detail": f"{len(r)} states, {len(pred['lr'])} windows"})  # recorded
    return rows  # the rows


def transfer() -> None:  # function: Part A in full
    t0 = time.time()  # clock (printed only)
    Bp = A.build()  # P rebuilt (every outer fit kept)
    g = A.gates(Bp)  # the adopted-pipeline script's gates: the X5's outer fits reproduce the committed predictions and selections bitwise
    TRANSFER.mkdir(parents=True, exist_ok=True)  # the folder
    E.write_csv(g, TRANSFER / "adopted_pipeline_gates.csv")  # written
    checks = [{"check": f"the adopted-pipeline script gate {r.gate}: {r.check}", "pass": bool(r._3), "detail": r.detail} for r in g.itertuples()]  # list comprehension: the gates as checks (_3 is the 'pass' column)
    steps = T.read_steps()  # the environment script's step table
    folds = pd.read_csv(E.ENV_DIR / "folds.csv", dtype={"held_out": str})  # the environment script's folds (miss penalty per fold)
    checks += source_check(Bp, steps, folds)  # the builder proved on the X5 itself
    pts = rt(B12.EVAL / "deployable_points.csv")  # the agent experiment's committed points
    committed = git("ls-files", "--error-unmatch", "Data/processed/rl_agents/evaluation/deployable_points.csv").returncode == 0 and git("diff", "--quiet", "HEAD", "--", "Data/processed/rl_agents/evaluation/deployable_points.csv").returncode == 0  # tracked and unchanged
    checks.append({"check": "the X5's points are the agent experiment's committed points, unchanged", "pass": committed, "detail": "Data/processed/rl_agents/evaluation/deployable_points.csv"})  # recorded
    sel = rt(T.AGENT_DIR / T.GRID / "selection.csv")  # the frozen selection (budgets)
    mm = pd.read_csv(T.AGENT_DIR / T.GRID / "models_manifest.csv").set_index("model_file")  # the frozen agents' digests
    used, bad = 0, []  # model files checked; mismatches
    runs, lat, curve, cov = [], [], [], []  # run rows; per-interval delays; curve rows; grid coverage rows
    for rx_t in TARGETS:  # for loop over the targets
        for h in SESSIONS:  # for loop over the folds
            sd = T.session_seconds(steps, rx_t, h)  # the target's held-out windowed seconds: the stream grid
            m = float(folds[(folds.receiver == SOURCE) & (folds.held_out == h)].m.iloc[0])  # the X5 fold's miss penalty (builds the environment only)
            mat, pred, sc = target_material(Bp, rx_t, h, steps, m)  # the X5 fold on the target
            if any(set(p_.index) != set(sd.sec) for p_ in pred.values()):  # if: the scores do not cover the stream grid exactly
                raise SystemExit(f"STOP: {rx_t} {h} scores and stream grid differ")  # stop
            rows_, ok, Z = B12.grid_rows(mat, sd)  # each grid second's state row, validity and 30 log-odds
            cov.append({"target": rx_t, "outer": h, "grid_seconds": len(sd), "decidable": int(ok.sum()), "clean": int((sd.label == "clean").sum()), "attack": int(sd.label.isin(A.ATTACK).sum())})  # the grid's coverage
            P_ = pts[(pts.receiver == SOURCE) & (pts.outer == h)]  # the X5's points of this fold
            for r in P_[P_.detector.isin(B12.BASE)].itertuples():  # for loop over the baselines' points
                n, k = (int(r.n), int(r.k)) if r.detector == "kofn" else (0, 0)  # k-of-n's setting
                d_, s_ = B12.baseline_decisions(r.detector, float(r.param), n, k, sd, pred, ok, Z)  # the decisions
                fd, fs = B12.both(sd, d_, s_, h)  # through the environment script's stream
                runs.append({"receiver": rx_t, "outer": h, "detector": r.detector, "share": r.share, "param": float(r.param), "lambda": np.nan, "seed": -1, **B12.run_row(fd, fs)})  # dict unpacking: the run
                lat += [{"receiver": rx_t, "outer": h, "detector": r.detector, "share": r.share, "lambda": np.nan, "seed": -1, **x} for x in fs["latency"]]  # list comprehension: its intervals
            for arm in ("primary", "surrogate"):  # the X5's agents: every lambda and seed
                for lam in T.LAMBDAS:  # for loop over the lambdas
                    b = int(sel[(sel.receiver == SOURCE) & (sel.outer == h) & (sel.arm == arm) & (sel["lambda"] == float(lam))].selected_budget.iloc[0])  # its frozen budget
                    for seed in T.OUTER_SEEDS:  # for loop over the five seeds
                        spec = {"stage": "outer", "receiver": SOURCE, "outer": h, "inner": "", "arm": arm, "lambda": float(lam), "seed": seed, "budget": b}  # the frozen agent's specification
                        f = B12.MODELS / f"{T.job_id(spec)}.npz"  # its weights
                        used += 1  # counted
                        if T.sha_file(f) != mm.at[f.name, "sha256"]:  # if: the file differs from the frozen manifest
                            bad.append(f.name)  # recorded
                        model, _ = T.load_q(spec, B12.MODELS, mat)  # the frozen agent, unchanged
                        env = E.SpoofingDetectionEnv(mat, arm, float(lam))  # an environment over the target's rows (only its matrices and key are read)
                        d_, s_ = T.agent_decisions(model, env, sd)  # decision and stopping mode
                        fd, fs = B12.both(sd, d_, s_, h)  # through the stream
                        runs.append({"receiver": rx_t, "outer": h, "detector": f"agent_{arm}", "share": np.nan, "param": float(lam), "lambda": float(lam), "seed": seed, **B12.run_row(fd, fs)})  # dict unpacking: the run
                        lat += [{"receiver": rx_t, "outer": h, "detector": f"agent_{arm}", "share": np.nan, "lambda": float(lam), "seed": seed, **x} for x in fs["latency"]]  # list comprehension: its intervals
            kn = sorted({(int(r.n), int(r.k)) for r in pts[(pts.receiver == SOURCE) & (pts.detector == "kofn")].itertuples()})  # set comprehension: the X5's k-of-n settings at its points
            for det, grid_, nk in [("lr", B12.LR_GRID, (0, 0)), ("rf", B12.LR_GRID, (0, 0)), ("cusum", B12.H_GRID, (0, 0))] + [("kofn", B12.KN_GRID, x) for x in kn]:  # list concatenation: each baseline's curve parameters (the evaluation script's grids)
                for prm in grid_:  # for loop over the parameter values
                    d_, s_ = B12.baseline_decisions(det, float(prm), nk[0], nk[1], sd, pred, ok, Z)  # the decisions
                    fd, fs = B12.both(sd, d_, s_, h)  # through the stream
                    curve.append({"receiver": rx_t, "outer": h, "detector": det, "n": nk[0], "k": nk[1], "param": float(prm), "clean": fd["clean"], "clean_in_alarm": fd["clean_in_alarm"], "stop_clean_in_alarm": fs["clean_in_alarm"], "intervals": fs["intervals"], "delays": ";".join(str(x["delay_s"]) for x in fs["latency"] if x["detected"])})  # generator inside join(): the curve row
            print(f"{rx_t} {h}: {len(runs)} runs ({time.time() - t0:.0f} s)", flush=True)  # progress
    checks.append({"check": "every X5 agent used equals its frozen weights (SHA-256 = models_manifest.csv)", "pass": not bad, "detail": f"{used} uses of {len(set(T.job_id({'stage': 'outer', 'receiver': SOURCE, 'outer': h, 'inner': '', 'arm': a, 'lambda': float(l), 'seed': s, 'budget': int(sel[(sel.receiver == SOURCE) & (sel.outer == h) & (sel.arm == a) & (sel['lambda'] == float(l))].selected_budget.iloc[0])}) for h in SESSIONS for a in ('primary', 'surrogate') for l in T.LAMBDAS for s in T.OUTER_SEEDS))} files; mismatches {bad}"})  # set comprehension: distinct files
    R_, LAT, C = pd.DataFrame(runs), pd.DataFrame(lat), pd.DataFrame(curve)  # the tables
    pts_t = pd.concat([pts[pts.receiver == SOURCE].assign(receiver=t) for t in TARGETS], ignore_index=True)  # the X5's points, carried to each target unchanged (list comprehension)
    at = B12.point_runs(R_, pts_t)  # the runs at the points
    ag = B12.aggregate(at, LAT, pts_t)  # P8: IQM and 95 % stratified bootstrap intervals
    vd = B12.verdicts(at, LAT, pts_t)  # P9: agent against the best transferred baseline
    vd = vd[vd.receiver.isin(TARGETS)].reset_index(drop=True)  # the targets only (the evaluation script loops over every receiver)
    cv = B12.curves(R_, LAT, C)  # P7: the transferred operating curves
    within = within_table(ag)  # the target's own the agent experiment figures beside the transferred ones
    checks.append({"check": "2.6.12 absent from every Part A table", "pass": all(SEALED not in set(t_[c].astype(str)) for t_, c in ((R_, "outer"), (LAT, "outer"), (C, "outer"))), "detail": "runs, latency, curve"})  # recorded
    for name, df in (("runs.csv", R_), ("latency.csv", LAT), ("point_runs.csv", at), ("aggregate.csv", ag), ("verdicts.csv", vd),  # tuple of (file, table)
                     ("curves.csv", cv), ("vs_within.csv", within), ("points.csv", pts_t), ("grid.csv", pd.DataFrame(cov)), ("checks.csv", pd.DataFrame(checks))):  # tuple continued
        E.write_csv(df, TRANSFER / name)  # written
    E.write_gz(C, TRANSFER / "curve_runs.csv.gz")  # the curve runs, gzipped
    figures_a(ag, cv, pts_t)  # the figures
    ck = pd.DataFrame(checks)  # the checks
    print(ck.to_string(index=False))  # printed
    print(vd.to_string(index=False))  # printed
    print(f"Part A: {time.time() - t0:.0f} s")  # wall time (printed only)
    if not ck["pass"].all():  # if: a check failed
        sys.exit(1)  # exit status 1


def within_table(ag: pd.DataFrame) -> pd.DataFrame:  # function: the transferred figures beside the target's own the agent experiment figures, per receiver, share, detector and metric
    w = rt(B12.EVAL / "aggregate.csv")  # the agent experiment's within-receiver aggregate
    keep = ["median delay (s)", "miss rate", "clean-window alarm share", "recall"]  # list: the metrics compared
    a = ag[ag.metric.isin(keep)][["receiver", "share", "detector", "metric", "iqm", "ci_low", "ci_high", "runs_defined", "pooled_clean_share", "pooled_median_delay_s", "pooled_miss_share"]]  # the transferred rows
    b = w[w.metric.isin(keep) & w.receiver.isin(TARGETS)][["receiver", "share", "detector", "metric", "iqm", "ci_low", "ci_high", "runs_defined", "pooled_clean_share", "pooled_median_delay_s", "pooled_miss_share"]]  # the within-receiver rows
    return a.merge(b, on=["receiver", "share", "detector", "metric"], how="outer", suffixes=("_transfer", "_within")).sort_values(["receiver", "share", "detector", "metric"], kind="stable").reset_index(drop=True)  # side by side


def figures_a(ag: pd.DataFrame, cv: pd.DataFrame, pts_t: pd.DataFrame) -> None:  # function: the figures (transferred operating curves beside the within-receiver ones) and 62 (the deployable points)
    import matplotlib.pyplot as plt  # pyplot (09 chose the file-only backend)
    wc = rt(B12.EVAL / "curves.csv")  # the agent experiment's within-receiver curves
    wa = rt(B12.EVAL / "aggregate.csv")  # the agent experiment's within-receiver aggregate
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)  # delay (top) and misses (bottom) per target
    for j, rx in enumerate(TARGETS):  # for loop over the targets
        for det in B12.DET:  # for loop over the detectors
            col, mk, lab = B12.STYLE[det]  # its style
            for src, d, ls, lw in (("transferred from the mosaic-X5", cv, "-", 2.0), ("trained on the receiver itself", wc, ":", 1.0)):  # for loop over the two sources
                d = d[(d.receiver == rx) & (d.detector == det)]  # its curve points
                if det == "kofn":  # k of n: the setting kept most often at the points
                    nk = pts_t[(pts_t.receiver == rx) & (pts_t.detector == "kofn")][["n", "k"]].value_counts().idxmax() if src.startswith("transferred") else B12.rt(B12.EVAL / "deployable_points.csv").query("receiver == @rx and detector == 'kofn'")[["n", "k"]].value_counts().idxmax()  # the most frequent (n, k) of that source
                    d = d[(d.n == nk[0]) & (d.k == nk[1])]  # its curve
                d = d.sort_values("clean_share")  # along the share axis
                for i, y in enumerate(("median_delay_s", "miss_share")):  # for loop over the two rows
                    axes[i, j].plot(d.clean_share, d[y], color=col, ls=ls, lw=lw, marker=mk if det.startswith("agent") and ls == "-" else None, ms=5, label=f"{lab}, {src}" if i == 0 and j == 0 else None)  # the curve
        for i in range(2):  # for loop over the rows
            for s in T.SHARES:  # the registered shares
                axes[i, j].axvline(s, color="#888888", lw=0.8, ls=":")  # a dotted line
            axes[i, j].set_xscale("symlog", linthresh=0.01)  # the low shares spread out
            axes[i, j].set_xlim(0, 1)  # every share
            axes[i, j].grid(alpha=0.3, lw=0.5)  # recessive grid
        axes[0, j].set_title(f"{NAME[rx]}: detectors of the mosaic-X5 (solid) and of the receiver itself (dotted)", fontsize=9, loc="left")  # title
        axes[0, j].set_yscale("symlog", linthresh=10)  # signed log: long delays do not flatten the rest
        axes[1, j].set_xlabel("pooled clean-window alarm share on the target (decision mode)", fontsize=8)  # axis
    axes[0, 0].set_ylabel("median delay (s), stopping mode", fontsize=8)  # axis
    axes[1, 0].set_ylabel("share of intervals missed", fontsize=8)  # axis
    h_, l_ = axes[0, 0].get_legend_handles_labels()  # the legend entries
    fig.legend(h_, l_, loc="lower center", ncol=3, fontsize=6.5, frameon=False)  # one legend below the panels
    fig.tight_layout(rect=(0, 0.13, 1, 1))  # layout with room for the legend
    M.save_fig(fig, GRAPH / "curves.png")  # byte-stable save
    fig, axes = plt.subplots(3, 2, figsize=(12, 9))  # delay, misses and realised clean share per target
    for j, rx in enumerate(TARGETS):  # for loop over the targets
        for i, mt in enumerate(("median delay (s)", "miss rate", "clean-window alarm share")):  # for loop over the rows
            for q, det in enumerate(B12.DET):  # for loop over the detectors
                col, mk, lab = B12.STYLE[det]  # its style
                for src, d, off, face in (("transferred", ag, -0.05, col), ("within", wa, 0.05, "white")):  # for loop over the two sources
                    d = d[(d.receiver == rx) & (d.detector == det) & (d.metric == mt)].set_index("share").reindex(list(T.SHARES))  # its three shares
                    x = np.arange(len(T.SHARES)) + (q - 2.5) * 0.13 + off  # dodged positions
                    axes[i, j].errorbar(x, d.iqm, yerr=[d.iqm - d.ci_low, d.ci_high - d.iqm], fmt=mk, color=col, mfc=face, ms=6, lw=1.0, capsize=2, label=f"{lab}, {src}" if i == 0 and j == 0 else None)  # IQM with its interval
            if mt == "clean-window alarm share":  # the realised share against the registered one
                axes[i, j].scatter(np.arange(len(T.SHARES)), T.SHARES, marker="_", s=600, color="#000000", label="registered share" if j == 0 else None, zorder=4)  # the nominal shares as bars
                axes[i, j].set_yscale("symlog", linthresh=0.01)  # low shares spread out
            axes[i, j].set_xticks(range(len(T.SHARES)), [f"{s:g}" for s in T.SHARES])  # list comprehension: the shares
            axes[i, j].set_ylabel(f"{mt}: IQM, 95 % interval", fontsize=8)  # axis
            axes[i, j].grid(alpha=0.3, lw=0.5, axis="y")  # recessive grid
        axes[0, j].set_title(NAME[rx], fontsize=9, loc="left")  # title
        axes[2, j].set_xlabel("registered clean-window share (the mosaic-X5's points)", fontsize=8)  # axis
    h_, l_ = axes[0, 0].get_legend_handles_labels()  # legend entries
    h2, l2 = axes[2, 0].get_legend_handles_labels()  # the registered-share marker
    fig.legend(h_ + h2[-1:], l_ + l2[-1:], loc="lower center", ncol=4, fontsize=6.5, frameon=False)  # one legend below the panels
    fig.tight_layout(rect=(0, 0.09, 1, 1))  # layout
    M.save_fig(fig, GRAPH / "points.png")  # byte-stable save


# ----------------------------------------------------------------------------------------------------------------------
# Part B: five-session P, nested scores, states, gates
# ----------------------------------------------------------------------------------------------------------------------
def fit_p(idx: pd.DataFrame, X: pd.DataFrame, wc10: pd.DataFrame, train: list, evals: dict, tag: dict) -> tuple:  # function: P's fit on the given training sessions (the adopted-pipeline script's screened, unfittable and fit_unit), both models kept
    cand = A.screened(idx, X, train, A.d1_columns(wc10))  # the screen script's screen refitted on the training sessions
    why = A.unfittable(idx, train, cand)  # can it be fitted?
    if why:  # if: not fittable (no position fit is unfittable)
        raise SystemExit(f"STOP: {tag} cannot be fitted: {why}")  # stop
    f = A.fit_unit(idx, X, wc10, train, evals, cand, False, ("lr", "rf"), keep=True, tag=tag)  # the adopted-pipeline script's fit, models kept
    if any(f["fail"].values()):  # if: the regression script's stop rule refused a model
        raise SystemExit(f"STOP: {tag} failed: {f['fail']}")  # stop
    rec = {"prep": f["prep"], "cand": cand, "sel": f["sel"], "names": f["names"], "model": {"lr": f["model_lr"], "rf": f["model_rf"]}, "train_rows": len(f["tr"]), "n_cand": f["n_cand"]}  # dict literal: the record transform() and score() read
    return rec, f  # the record and the raw fit


def build_b() -> list:  # function: Part B's fits, states and nested scores for every receiver; returns the gate rows
    A.use_labels("final")  # the scoring module points at the final label set (as the adopted-pipeline script's build does)
    wc10 = A.window_dict()  # the feature script's window dictionary at w = 10
    cells = A.d1_cells(A.load_cells(A.FINAL, A.W, True), wc10)  # the final windows (2.6.12 dropped by the regression script's reader), D1 columns
    steps = T.read_steps()  # the environment script's step table
    ix11 = pd.read_csv(E.ENV_DIR / "state_index.csv.gz", dtype={"session": str, "fold": str})  # the environment script's state index
    lr_tab = rt(E.PROC / "logistic_regression" / "logistic_regression_predictions_primary.csv.gz")  # committed outer regression probabilities
    rf_tab = rt(E.PROC / "random_forest" / "random_forest_predictions_primary.csv.gz")  # committed outer forest probabilities
    fp11 = pd.read_csv(E.ENV_DIR / "transform_fingerprints.csv", dtype={"held_out": str})  # the environment script's transform fingerprints
    gates, fits, nested, index = [], [], [], []  # gate rows; fit rows; nested predictions; state index parts
    chk = lambda name, ok, detail: gates.append({"check": name, "pass": bool(ok), "detail": detail})  # lambda: records one gate
    STATES_B.mkdir(parents=True, exist_ok=True)  # the folder
    for rx in RXS:  # for loop over the receivers
        idx, X = cells[rx]  # its windows
        if (idx.test_code == SEALED).any():  # if: a 2.6.12 window survived the reader (cannot)
            raise SystemExit(f"STOP: 2.6.12 in {rx}'s cells")  # stop
        wr = E.window_rows(idx)  # session -> window lookup (the environment script's)
        order = ix11[(ix11.receiver == rx) & (ix11.fold == SESSIONS[0])].sort_values("row")  # the environment script's valid seconds of the five sessions, in the environment script's row order
        same = all(np.array_equal(order[["session", "sec"]].to_numpy(), ix11[(ix11.receiver == rx) & (ix11.fold == h)].sort_values("row")[["session", "sec"]].to_numpy()) for h in SESSIONS)  # generator in all(): every fold lists the same seconds in the same order
        chk(f"{rx}: every the environment fold indexes the same valid seconds in the same order", same, f"{len(order)} rows")  # recorded
        z = np.full(len(idx), np.nan)  # the nested decision value of every window
        for h, train in A.S.folds():  # for loop over the inner folds: each session held out once, the fit on the other four (the screen script's order)
            te = A.eval_rows(idx, h)  # the held-out session's windows in time order
            rec, f = fit_p(idx, X, wc10, train, {rx: te}, {"run": "transfer_part_b", "outer": OUTER_B, "inner": h, "receiver": rx})  # the inner fit
            z[te] = A.decision(rec["model"]["lr"], f["Z"][te])  # its decision values on the held-out windows (C order, as the fit scored them)
            p_lr, p_rf = f["p"]["lr"][rx], f["p"]["rf"][rx]  # its probabilities there
            sec_h = idx.sec.to_numpy(np.int64)[te]  # their seconds
            lab_h = idx.label.to_numpy()[te]  # their scoring labels
            c9 = lr_tab[(lr_tab.receiver == rx) & (lr_tab.held_out == h)].set_index("sec").reindex(sec_h)  # the committed regression rows
            c10 = rf_tab[(rf_tab.receiver == rx) & (rf_tab.held_out == h)].set_index("sec").reindex(sec_h)  # the committed forest rows
            chk(f"{rx} inner fit without {h} = P's outer fit of fold {h}: probabilities bitwise (logistic_regression_b, random_forest)", np.array_equal(p_lr, c9.p.to_numpy()) and np.array_equal(p_rf, c10.p.to_numpy()) and (c9.label.to_numpy() == lab_h).all(), f"{len(te)} windows")  # recorded
            fpr = fingerprint(rec)  # the fit's digests
            chk(f"{rx} inner fit without {h}: transform fingerprint = transform_fingerprints.csv", fpr["transform_sha256"] == fp11[(fp11.receiver == rx) & (fp11.held_out == h)].sha256.iloc[0], fpr["transform_sha256"][:16])  # recorded
            Zh = A.transform(rec, X, wc10)  # every window of the receiver under this inner fit's transform (C order)
            Ph = np.vstack([np.hstack([Zh[E.rows_at(wr, s, order[order.session == s].sec.to_numpy(np.int64) - lag)] for lag in E.LAGS]) for s in SESSIONS if (order.session == s).any()])  # nested comprehensions: the primary states of every valid second under this transform
            env_states = E.load_npz(E.ENV_DIR / f"states_{rx}_{h}.npz")  # the environment's states of fold h
            chk(f"{rx} inner primary states without {h} = the environment's fold-{h} primary states, bitwise", np.array_equal(Ph, env_states["primary"]), f"{Ph.shape}")  # recorded
            E.write_npz({"primary": Ph, "row": np.arange(len(Ph), dtype=np.int64), "sec": order.sec.to_numpy(np.int64)}, STATES_B / f"inner_states_{rx}_{OUTER_B}_{h}.npz")  # the inner states, under the name the training script's job_material() reads
            nested += [pd.DataFrame({"model": mdl, "receiver": rx, "outer_held_out": OUTER_B, "held_out": h, "sec": sec_h, "label": lab_h, "p": p, "dec": z[te] if mdl == "lr" else np.nan}) for mdl, p in (("lr", p_lr), ("rf", p_rf))]  # list comprehension: the nested predictions, both models
            fits.append({"receiver": rx, "fit": f"without {h}", "train_sessions": ";".join(train), "train_rows": rec["train_rows"], "candidates": rec["n_cand"], "kept_after_pearson": len(rec["prep"]["kept"]), "selected_columns": ";".join(rec["names"]), **fpr})  # the fit's row
        rec5, f5 = fit_p(idx, X, wc10, list(SESSIONS), {}, {"run": "transfer_part_b", "outer": OUTER_B, "inner": "", "receiver": rx})  # P on all five position sessions
        fits.append({"receiver": rx, "fit": "five sessions", "train_sessions": ";".join(SESSIONS), "train_rows": rec5["train_rows"], "candidates": rec5["n_cand"], "kept_after_pearson": len(rec5["prep"]["kept"]), "selected_columns": ";".join(rec5["names"]), **fingerprint(rec5)})  # its row
        Z5 = A.transform(rec5, X, wc10)  # every window under the five-session transform (C order)
        P5 = np.vstack([np.hstack([Z5[E.rows_at(wr, s, order[order.session == s].sec.to_numpy(np.int64) - lag)] for lag in E.LAGS]) for s in SESSIONS if (order.session == s).any()])  # the outer primary states
        S5 = np.vstack([np.column_stack([z[E.rows_at(wr, s, order[order.session == s].sec.to_numpy(np.int64) - lag)] for lag in range(E.HIST, -1, -1)]) for s in SESSIONS if (order.session == s).any()])  # the surrogate states: nested z at t-29 .. t
        ok_s = True  # the surrogate against the environment's held-out rows
        for s in SESSIONS:  # for loop over the sessions
            env_states = E.load_npz(E.ENV_DIR / f"states_{rx}_{s}.npz")  # fold s, whose held-out session is s
            rr = ix11[(ix11.receiver == rx) & (ix11.fold == s) & (ix11.role == "heldout")].row.to_numpy()  # its held-out rows
            ok_s &= np.array_equal(S5[(order.session == s).to_numpy()], env_states["surrogate"][rr])  # bitwise
        chk(f"{rx} surrogate states = the environment's held-out surrogate states of each session's own fold, bitwise", ok_s, f"{S5.shape}")  # recorded
        chk(f"{rx} every Part B state finite", np.isfinite(P5).all() and np.isfinite(S5).all(), "primary and surrogate")  # recorded
        E.write_npz({"primary": P5, "surrogate": S5, "sec": order.sec.to_numpy(np.int64)}, STATES_B / f"states_{rx}_{OUTER_B}.npz")  # the outer states
        index.append(order[["receiver", "session", "sec", "label", "train_label", "step", "run_id", "row"]].assign(role="train"))  # the index: every row a training row
    UNSEEN.mkdir(parents=True, exist_ok=True)  # the folder
    E.write_gz(pd.concat(index, ignore_index=True), UNSEEN / "state_index.csv.gz")  # the state index
    npr = pd.concat(nested, ignore_index=True)  # the nested predictions
    E.write_gz(npr, UNSEEN / "nested_predictions.csv.gz")  # written
    E.write_csv(pd.DataFrame(fits), UNSEEN / "fits.csv")  # the fits and their digests (the five-session fit is refitted in the held-out evaluation and must match)
    chk("2.6.12 absent from every Part B table", SEALED not in set(npr.held_out) and all(SEALED not in set(x.session) for x in index), "nested predictions, state index")  # recorded
    return gates  # the gate rows


# ----------------------------------------------------------------------------------------------------------------------
# Part B: jobs, selection, learning test, points, implied thresholds, freeze
# ----------------------------------------------------------------------------------------------------------------------
def specs_inner_b() -> list:  # function: every Part B inner job (3 receivers x 5 inner folds x 2 arms x 5 lambda x 2 seeds)
    # list comprehension over five loops: one specification per inner training, run to the full schedule
    return [{"stage": "inner", "receiver": rx, "outer": OUTER_B, "inner": h, "arm": arm, "lambda": float(lam), "seed": s, "budget": T.SCHEDULE}
            for rx in RXS for h in SESSIONS for arm in ("primary", "surrogate") for lam in T.LAMBDAS for s in T.INNER_SEEDS]  # list comprehension continued: the loops


def run_jobs_b(specs: list) -> pd.DataFrame:  # function: every job through three spawned workers (the training script's run_jobs with run_job_b, so the workers find this module's material)
    ctx = multiprocessing.get_context("spawn")  # spawn: each worker starts a fresh interpreter, which imports this file and so extends 11 and 12 again
    rows = []  # timings
    with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=ctx) as ex:  # with-block: the pool
        futs = {ex.submit(run_job_b, s, str(STATES_B), str(JOBS_B), str(MODELS_B)): s for s in specs}  # dict comprehension: future -> its specification
        for f in as_completed(futs):  # for loop over the jobs as they finish
            r = f.result()  # its timing (an exception in the worker is raised here)
            rows.append({**r, "arm": futs[f]["arm"], "stage": futs[f]["stage"]})  # kept
            print(f"  {r['job']}: {'skipped' if r['skipped'] else str(r['seconds']) + ' s'}", flush=True)  # progress
    return pd.DataFrame(rows).sort_values("job").reset_index(drop=True)  # in a fixed order


def select_b(curve: pd.DataFrame, n_inner: int) -> pd.DataFrame:  # function: the training script's select() for any number of inner folds, with the decision's learning test on the inner folds
    rows = []  # one row per cell
    for (rx, outer, arm, lam), d in curve.groupby(["receiver", "outer", "arm", "lambda"], sort=True):  # for loop over the cells
        if (d.inner == outer).any() or set(d.groupby("budget").size()) != {len(T.INNER_SEEDS) * n_inner} or set(d.budget) != set(T.BUDGETS):  # if: a held-out row, or a snapshot lacking a fold or a seed
            raise SystemExit(f"STOP: the inner loop of {rx} {outer} {arm} {lam} is incomplete")  # stop
        mean = d.groupby("budget").criterion.mean()  # the criterion averaged over the inner folds and seeds
        best = mean.min()  # the lowest mean
        b = int(min(x for x in mean.index if mean[x] <= best + T.TIE_TOL))  # generator in min(): the shortest budget within the tie tolerance
        triv = d.drop_duplicates(["inner"])[[f"{p}_criterion" for p in T.TRIVIAL]].mean()  # the trivial policies averaged over the inner folds
        at_b = d[d.budget == b]  # the inner runs at the selected budget
        per = at_b.groupby("inner").agg(agent=("criterion", "mean"), **{p: (f"{p}_criterion", "first") for p in T.TRIVIAL})  # dict comprehension in agg(): per inner fold, the agent's criterion (mean of the seeds) and the trivial ones
        below = int((per.agent < per[list(T.TRIVIAL)].min(axis=1)).sum())  # inner folds where the agent is below each trivial policy
        clean = at_b.stopping_clean.fillna(0).sum()  # pooled inner held-out clean seconds
        # dict literal over three lines: the row, the training script's columns first
        rows.append({"receiver": rx, "outer": outer, "arm": arm, "lambda": lam, **{f"criterion_{int(x)}": float(mean[x]) for x in mean.index}, "selected_budget": b, "selected_criterion": float(mean[b]),
                     **{f"{p}_criterion": float(triv[f"{p}_criterion"]) for p in T.TRIVIAL}, "best_trivial": float(triv.min()), "below_all_trivial": bool(float(mean[b]) < float(triv.min())),  # dict literal continued: the trivial policies
                     "inner_clean_s": int(clean), "inner_stopping_clean_share": float(at_b.stopping_clean_in_alarm.fillna(0).sum() / clean) if clean else None, "inner_decision_clean_share": float(at_b.decision_clean_in_alarm.fillna(0).sum() / clean) if clean else None, "inner_folds_below_all_trivial": below, "learned_inner": below >= 3})  # dict literal continued: shares and the decision
    return pd.DataFrame(rows)  # the table


def points_cell(rx: str, outer: str, train: list, mat: dict, nested: pd.DataFrame, steps: pd.DataFrame, sel_cell: pd.DataFrame) -> tuple:  # function: the deployable points of one receiver and outer fold (the training script's deployable() body and the evaluation script's agent rule, for any training sessions)
    rows, kn_rows = [], []  # the points; every k-of-n candidate
    for mdl in ("lr", "rf"):  # the supervised baselines: thresholds on the clean out-of-fold windows of the nested predictions
        q = nested[(nested.model == mdl) & (nested.receiver == rx) & (nested.outer_held_out == outer) & nested.held_out.isin(train) & (nested.label == "clean")]  # those windows
        for s in T.SHARES:  # for loop over the shares
            t = T.lowest_threshold(q.p.to_numpy(), len(q), s)  # the threshold
            rows.append({"receiver": rx, "outer": outer, "detector": mdl, "share": s, "param": t, "param_name": "threshold", "inner_clean": len(q), "inner_clean_share": float((q.p >= t).mean()) if len(q) else None})  # the point
    per = {}  # session -> (stream seconds, their valid-row positions, S per valid row, 30 log-odds per valid row)
    for sess in train:  # for loop over the training sessions
        d, r_, S = T.surrogate_runs(mat, sess)  # its valid rows and S
        sd = T.session_seconds(steps, rx, sess)  # its windowed seconds (the stream)
        pos = pd.Series(np.arange(len(d)), index=d.sec.to_numpy()).reindex(sd.sec.to_numpy()).to_numpy()  # each windowed second's position among the valid rows (NaN: no state)
        per[sess] = (sd, pos, S, mat["surrogate"][r_])  # kept
    n_clean = sum(int((sd.label == "clean").sum()) for sd, _, _, _ in per.values())  # generator in sum(): pooled clean windowed seconds
    on_grid = lambda stat: {k: np.where(np.isfinite(p_), stat[k][np.nan_to_num(p_, nan=0).astype(int)] if len(stat[k]) else np.nan, np.nan) for k, (_, p_, _, _) in per.items()}  # lambda with a dict comprehension: a per-valid-row statistic on each session's grid
    S_g = on_grid({k: v[2] for k, v in per.items()})  # S_t on the stream grid
    clean_v = lambda g: np.concatenate([g[k][(per[k][0].label == "clean").to_numpy()] for k in per])  # lambda: the statistic at the pooled clean seconds
    for s in T.SHARES:  # CUSUM: the smallest h whose pooled clean share is at most s
        h = T.lowest_threshold(clean_v(S_g), n_clean, s)  # the threshold on S
        dl = []  # stopping-mode delays on the training sessions, for the record
        for k, (sd, p_, _, Z) in per.items():  # for loop over the sessions
            ok = np.isfinite(p_)  # seconds with a state
            zg = np.where(ok, Z[np.nan_to_num(p_, nan=0).astype(int), -1] if len(Z) else np.nan, np.nan)  # z_t on the grid
            st = E.stream(sd.sec.to_numpy(np.int64), sd.label.to_numpy(object), E.cusum(sd.sec.to_numpy(np.int64), np.nan_to_num(zg), ok, h, "stopping"), k, "stopping")  # the environment script's cusum through the stream
            dl += [x["delay_s"] for x in st["latency"] if x["detected"]]  # delays
        rows.append({"receiver": rx, "outer": outer, "detector": "cusum", "share": s, "param": h, "param_name": "h", "inner_clean": n_clean, "inner_clean_share": float((clean_v(S_g) >= h).sum() / n_clean) if n_clean else None, "inner_median_delay": float(np.median(dl)) if dl else None})  # the point
    for n in E.KN_N:  # k of n: for every (n, k), the smallest h meeting the share
        for k_ in range(1, n + 1):  # for loop over k
            Q = on_grid({k: (np.sort(v[3][:, -n:], axis=1)[:, n - k_] if len(v[3]) else np.zeros(0)) for k, v in per.items()})  # the k-th largest of the last n log-odds, on the grid
            for s in T.SHARES:  # for loop over the shares
                h = T.lowest_threshold(clean_v(Q), n_clean, s)  # the threshold
                dl, det, iv = [], 0, 0  # delays; detected and all intervals
                for k, (sd, p_, _, _) in per.items():  # for loop over the sessions
                    dec = (np.nan_to_num(Q[k], nan=-np.inf) >= h).astype(int)  # k of the last n reach h
                    st = E.stream(sd.sec.to_numpy(np.int64), sd.label.to_numpy(object), dec, k, "stopping")  # stopping mode
                    dl += [x["delay_s"] for x in st["latency"] if x["detected"]]  # delays
                    det, iv = det + st["detected"], iv + st["intervals"]  # counts
                kn_rows.append({"receiver": rx, "outer": outer, "n": n, "k": k_, "share": s, "h": h, "inner_clean_share": float((clean_v(Q) >= h).sum() / n_clean) if n_clean else None, "inner_median_delay": float(np.median(dl)) if dl else np.inf, "inner_detected": det, "inner_intervals": iv})  # one candidate
    kn = pd.DataFrame(kn_rows)  # this cell's candidates
    for s in T.SHARES:  # the kept k-of-n setting per share (P6 as amended: lowest inner median delay among those meeting the share)
        c = kn[(kn.share == s) & (kn.inner_clean_share <= s)].sort_values(["inner_median_delay", "n", "k"], kind="stable").iloc[0]  # lowest delay, ties to the smaller n, then k
        rows.append({"receiver": rx, "outer": outer, "detector": "kofn", "share": s, "param": float(c.h), "param_name": f"h (n={int(c.n)}, k={int(c.k)})", "n": int(c.n), "k": int(c.k), "inner_clean": n_clean, "inner_clean_share": c.inner_clean_share, "inner_median_delay": c.inner_median_delay})  # the point
    for arm, a in sel_cell.groupby("arm", sort=True):  # the agent: P6 as amended
        a = a.sort_values("lambda")  # its five lambdas
        for s in T.SHARES:  # for loop over the shares
            ok = a[a.inner_decision_clean_share.notna() & (a.inner_decision_clean_share <= s)]  # the lambdas whose decision-mode inner share meets the share
            lam = float(ok["lambda"].min()) if len(ok) else None  # the smallest one, or no point
            at = a.set_index("lambda").loc[lam] if lam is not None else None  # its row
            # dict literal over two lines: the agent's point
            rows.append({"receiver": rx, "outer": outer, "detector": f"agent_{arm}", "share": s, "param": lam, "param_name": "lambda (smallest whose decision-mode inner share meets the share, P6 as amended)", "inner_clean": int(a.inner_clean_s.iloc[0]),
                         "inner_clean_share": float(at.inner_decision_clean_share) if at is not None else None, "inner_stopping_clean_share": float(at.inner_stopping_clean_share) if at is not None else None, "budget": int(at.selected_budget) if at is not None else None, "no_point": lam is None})  # dict literal continued
    order = {d: i for i, d in enumerate(B12.DET)}  # dict comprehension: detector -> position
    out = pd.DataFrame(rows)  # the table
    return out.assign(_d=out.detector.map(order)).sort_values(["_d", "share"], kind="stable").drop(columns="_d").reset_index(drop=True), kn  # the points in a fixed order, and the candidates


def guard(freeze_commit: str = None, manifest: Path = None) -> pd.DataFrame:  # function: refuses (SystemExit) unless every frozen Part B file equals its copy in the transfer study freeze commit
    freeze_commit = FREEZE_COMMIT if freeze_commit is None else freeze_commit  # conditional expression: the module's constant unless a test passes one
    manifest = UNSEEN / "freeze_manifest.csv" if manifest is None else manifest  # the freeze manifest
    if not freeze_commit:  # if: no freeze commit is set
        raise SystemExit("REFUSED: no the transfer study freeze commit is set; 2.6.12 stays sealed")  # refuse
    msg = git("log", "-1", "--format=%s", freeze_commit)  # the commit's subject
    if msg.returncode != 0 or not msg.stdout.decode().startswith("the transfer study FREEZE"):  # if: not the freeze commit
        raise SystemExit(f"REFUSED: {freeze_commit} is not the transfer study FREEZE commit")  # refuse
    fm = pd.read_csv(manifest)  # the manifest (file, sha256)
    rows = []  # one row per file
    for f, h in zip(fm.file, fm.sha256):  # for loop over the frozen files
        at = git("show", f"{freeze_commit}:{f}")  # its bytes in the freeze commit
        now = (PROJECT / f).read_bytes() if (PROJECT / f).exists() else b""  # its bytes now
        rows.append({"file": f, "in_freeze_commit": at.returncode == 0, "equal_to_commit": at.stdout == now, "sha256_matches_manifest": hashlib.sha256(now).hexdigest() == h})  # the row
    ck = pd.DataFrame(rows)  # the table
    if not (ck.in_freeze_commit & ck.equal_to_commit & ck.sha256_matches_manifest).all():  # if: any frozen file differs
        raise SystemExit("REFUSED: a frozen file differs from the transfer study freeze commit")  # refuse
    return ck  # the proof


def read_2612(rx: str) -> tuple:  # function: the ONLY reader of 2.6.12's windows (index and D1 values); it runs the guard first
    ck = guard()  # refuses unless the freeze holds
    wc10 = A.window_dict()  # the feature script's window dictionary at w = 10
    cols = A.d1_columns(wc10)  # the 778 D1 columns
    idx = pd.read_csv(A.FINAL / f"w{A.W}" / f"{rx}_index.csv.gz", dtype={"test_code": str})  # the index, read directly (the regression script's reader would drop 2.6.12)
    keep = (idx.test_code == SEALED).to_numpy()  # the 2.6.12 rows
    X = idx[A.KEYS]  # keys, to align the parts
    for pt in ("orbitfree", "geometry"):  # for loop over the parts that hold D1 columns
        X = X.merge(pd.read_csv(A.FINAL / f"w{A.W}" / f"{rx}_{pt}.csv.gz", dtype={"test_code": str}), on=A.KEYS, how="left", validate="one_to_one")  # merged one to one
    return idx[keep].reset_index(drop=True), X[keep].reset_index(drop=True)[cols], ck  # the 2.6.12 windows, their D1 values and the guard's proof


def unseen() -> None:  # function: Part B: fits, states, inner loop, selection, outer agents, points, implied thresholds, freeze manifest
    t0 = time.time()  # clock
    clock = {}  # phase wall times
    gates = build_b()  # fits, states and nested scores, with the gates
    clock["_prepare"] = time.time() - t0  # seconds
    ck = pd.DataFrame(gates)  # the gate table
    E.write_csv(ck, UNSEEN / "gates.csv")  # written
    print(ck.to_string(index=False))  # printed
    if not ck["pass"].all():  # if: a gate failed
        raise SystemExit("STOP: a Part B gate failed; nothing is trained")  # stop
    t1 = time.time()  # clock
    si = specs_inner_b()  # 300 inner jobs
    tm_in = run_jobs_b(si)  # run (finished jobs are skipped by hash)
    clock["_inner"] = time.time() - t1  # seconds
    cur = T.inner_curve(JOBS_B, si)  # every snapshot of every inner job (the training script's)
    E.write_gz(cur, UNSEEN / "inner_curve.csv.gz")  # written
    sel = select_b(cur, len(SESSIONS))  # the budgets and the decision's learning test
    E.write_csv(sel, UNSEEN / "selection.csv")  # written
    lr_ = sel.groupby(["receiver", "arm", "lambda"], sort=True).agg(inner_folds_below_all_trivial=("inner_folds_below_all_trivial", "first"), learned=("learned_inner", "first"), selected_budget=("selected_budget", "first")).reset_index()  # per receiver, arm and lambda (one outer fold)
    E.write_csv(lr_, UNSEEN / "learned.csv")  # written
    t2 = time.time()  # clock
    so = [{"stage": "outer", "receiver": r.receiver, "outer": OUTER_B, "inner": "", "arm": r.arm, "lambda": float(r["lambda"]), "seed": s, "budget": int(r.selected_budget)} for _, r in sel.iterrows() for s in T.OUTER_SEEDS]  # list comprehension: 150 outer jobs
    tm_out = run_jobs_b(so)  # run, each stopped at its budget
    clock["_outer"] = time.time() - t2  # seconds
    t3 = time.time()  # clock
    steps = T.read_steps()  # the environment script's step table
    nested = pd.read_csv(UNSEEN / "nested_predictions.csv.gz", dtype={"outer_held_out": str, "held_out": str}, float_precision="round_trip")  # read back exactly
    pts, kns = [], []  # points and k-of-n candidates
    for rx in RXS:  # for loop over the receivers
        p_, k_ = points_cell(rx, OUTER_B, list(SESSIONS), E.material(rx, OUTER_B), nested, steps, sel[(sel.receiver == rx) & (sel.outer == OUTER_B)])  # the receiver's points
        pts.append(p_)  # kept
        kns.append(k_)  # kept
    E.write_csv(pd.concat(pts, ignore_index=True), UNSEEN / "points.csv")  # the frozen points
    E.write_csv(pd.concat(kns, ignore_index=True), UNSEEN / "kofn_candidates.csv")  # every k-of-n candidate
    imp = T.implied(so, MODELS_B)  # P11 (the training script's, on Part B's material through the extended reader)
    E.write_csv(imp, UNSEEN / "implied_thresholds.csv")  # per agent
    med, rho = T.implied_summary(imp)  # medians and rho
    E.write_csv(med, UNSEEN / "implied_medians.csv")  # written
    E.write_csv(rho, UNSEEN / "implied_spearman.csv")  # written
    clock["_freeze"] = time.time() - t3  # seconds
    E.write_csv(pd.DataFrame([{"job": T.job_id(s), "stage": s["stage"], "output_sha256": T.sha_file(JOBS_B / f"{T.job_id(s)}.json.gz")} for s in si + so]), UNSEEN / "jobs_manifest.csv")  # list comprehension: the job manifest
    E.write_csv(pd.DataFrame([{"model_file": p.name, "sha256": T.sha_file(p), "bytes": p.stat().st_size} for p in sorted(MODELS_B.glob("*.npz"))]), UNSEEN / "models_manifest.csv")  # list comprehension: the model manifest
    frozen = [UNSEEN / n for n in ("fits.csv", "selection.csv", "learned.csv", "points.csv", "kofn_candidates.csv", "implied_thresholds.csv", "models_manifest.csv", "jobs_manifest.csv", "state_index.csv.gz", "nested_predictions.csv.gz")]  # list comprehension: the frozen tables
    frozen += sorted(STATES_B.glob("*.npz")) + sorted(MODELS_B.glob("*.npz"))  # the state files and the agents
    E.write_csv(pd.DataFrame([{"file": str(p.relative_to(PROJECT)), "sha256": T.sha_file(p)} for p in frozen]), UNSEEN / "freeze_manifest.csv")  # list comprehension: their digests (read_2612's guard checks them)
    tm = pd.concat([tm_in, tm_out, pd.DataFrame([{"job": k, "seconds": round(v, 1)} for k, v in clock.items()])], ignore_index=True)  # every job's timing and the phases
    E.write_csv(tm, UNSEEN / "timing.csv")  # wall clock by nature (declared non-deterministic)
    print({k: round(v / 3600, 2) for k, v in clock.items()}, "hours")  # dict comprehension: the phases in hours


# ----------------------------------------------------------------------------------------------------------------------
# --heldout2612: the single 2.6.12 evaluation (and its dry run on a training session)
# ----------------------------------------------------------------------------------------------------------------------
U2612 = TRANSFER_DIR / "unseen_attack"  # the held-out evaluation's outputs
PHASES = ("jam_spoof", "spoof")  # tuple: 2.6.12's two attack phases (the jamming-phase script): jamming and spoofing, then time spoofing alone


def steps_of(lab: pd.DataFrame, win_secs: np.ndarray) -> pd.DataFrame:  # function: every second of one receiver's session with its window, window run and validity (the environment script's seconds_table rule)
    g = lab.sort_values("sec").reset_index(drop=True)  # the label grid in time order
    s = g.sec.to_numpy(np.int64)  # its seconds
    if not (np.diff(s) == 1).all():  # if: the grid is not consecutive (the label script's grid is)
        raise SystemExit("STOP: label grid not consecutive")  # stop
    w = np.isin(s, win_secs)  # a window ends at the second
    run, valid = np.full(len(s), -1), np.zeros(len(s), bool)  # tuple assignment: window run (-1: none) and validity
    for k, (i, j) in enumerate(E.SC.runs(s, w)):  # for loop over the scoring module's maximal runs of windowed seconds
        run[i:j + 1] = k  # the run's number
        valid[i:j + 1] = s[i:j + 1] - s[i] >= E.HIST  # valid from 29 s after the run's first window
    step = np.select([g.label == "clean", g.label.isin(A.ATTACK), g.label != "excluded"], ["pre", "post", "uncertain"], "excluded")  # step type by the scoring label (no training label exists for an evaluation session)
    return g.assign(window=w, run_id=run, valid=valid, step=step)  # the table


DECISIONS = []  # list: every 2.6.12 run's per-second decisions, recorded by phase_rows() for report_2612() (reporting only; the
ONSET_MAX_AGE = 60  # the state at onset: the last decided second before a first second counts only if it lies at most 60 s earlier
OWN_ALL = ("lr", "rf")  # tuple: the detectors that decide on every windowed second; every other one decides on the decidable seconds only


def onset_state(sec: np.ndarray, dec: np.ndarray, decided: np.ndarray, t0: int) -> dict:  # function: the state of a detector as an interval or phase begins (in alarm, not in alarm, not decidable) and the age of the second it is read on
    k = np.flatnonzero(decided & (sec < t0))  # positions of the decided seconds before the first second
    if not k.size:  # if: no decided second before it at all
        return {"state": "not decidable", "ref_age_s": np.nan, "ref_alarm": None}  # dict literal: no reference second
    r = k[-1]  # the last of them (the grid is in time order)
    age = int(t0 - sec[r])  # its age in seconds
    al = bool(dec[r] > 0)  # the decision-mode output on it
    st = ("in alarm" if al else "not in alarm") if age <= ONSET_MAX_AGE else "not decidable"  # nested conditional expression: the state, or not decidable when the second is older than 60 s
    return {"state": st, "ref_age_s": age, "ref_alarm": al}  # dict literal: the state, the age and the output on that second


def refractory_at(sec: np.ndarray, ev: np.ndarray, t0: int) -> tuple:  # function: the refractory mark: does the refractory period of the last stopping-mode event before t0 cover t0? (flag, that event's age in seconds)
    e = sec[ev & (sec < t0)]  # the events before the first second
    if not e.size:  # if: none
        return False, np.nan  # not refractory, no age
    age = int(t0 - e.max())  # the last one's age
    return age <= E.REFRACTORY, age  # the environment script's rule allows a new event only more than 30 s after the last, so an age of 30 s or less blocks the first second


def phase_core(base: dict, sec: np.ndarray, lab: np.ndarray, phase: np.ndarray, ok: np.ndarray, dec: np.ndarray, stop: np.ndarray, starts: dict) -> tuple:  # function: the decision's delay per phase for one run with the state at onset, the refractory mark, both clocks and the minute before; returns (rows, events, decidable-only events)
    ev = E.events(sec, np.asarray(stop) > 0)  # the stopping-mode alarm events (the environment script's refractory rule; already events for the sequential detectors)
    okb = np.asarray(ok, bool)  # the decidable seconds (a state exists) as booleans
    decided = np.ones(len(sec), bool) if base["detector"] in OWN_ALL else okb  # conditional expression: the detector's own decided seconds (every windowed second for the regression and the forest)
    so = np.flatnonzero(okb)  # positions of the decidable seconds
    ev_d = np.zeros(len(sec), bool)  # the events of the decidable-only reading
    ev_d[so] = E.events(sec[so], np.asarray(stop)[so] > 0)  # the refractory rule applied to the decidable seconds alone
    rows = []  # one row per phase
    for ph in PHASES:  # for loop over the attack phases
        m = phase == ph  # the grid seconds of the phase
        a = starts.get(ph)  # the phase's first second on the label grid
        if a is None:  # if: the session has no such phase
            continue  # next phase
        hit = sec[ev & m]  # alarm events inside the phase
        hd = sec[ev_d & m]  # the decidable-only reading's events inside the phase
        fd = sec[m & okb]  # the phase's decidable seconds
        f0 = int(fd.min()) if fd.size else None  # conditional expression: its first decidable second, if any
        s1 = onset_state(sec, dec, decided, a)  # the state on the detector's own clock
        s2 = onset_state(sec, dec, okb, a)  # the state on the decidable seconds
        r1, g1 = refractory_at(sec, ev, a)  # tuple assignment: refractory at onset in the registered stream, and the last event's age
        r2, g2 = refractory_at(sec, ev_d, a)  # tuple assignment: the same in the decidable-only stream
        nd1, nd2 = s1["state"] == "in alarm" or r1, s2["state"] == "in alarm" or r2  # tuple assignment: either mark (no detection delay), both readings
        pre = (sec >= a - ONSET_MAX_AGE) & (sec < a)  # the minute before the phase's first second
        n_dc = int((pre & decided).sum())  # decided seconds in that minute
        n_al = int((pre & decided & (np.asarray(dec) > 0)).sum())  # of them in decision-mode alarm
        spoof_flag = ({"in alarm": "True", "not in alarm": "False"}.get(s1["state"], "not decidable")) if ph == "spoof" else None  # dict lookup with a default: F4's three-valued column (spoofing phase only)
        # dict literal over nine lines: the row (run1's columns first, then the onset state, then the refractory mark
        rows.append({**base, "phase": ph, "phase_start": int(a), "grid_seconds": int(m.sum()), "decidable": int((m & okb).sum()), "decidable_attack": int((m & okb & np.isin(lab, A.ATTACK)).sum()), "detected": bool(hit.size), "delay_s": float(hit.min() - a) if hit.size else np.nan,
                     "alarming_at_spoof_start": spoof_flag, "onset_state": s1["state"], "onset_ref_age_s": s1["ref_age_s"], "onset_ref_alarm": s1["ref_alarm"], "delay_counts": bool(hit.size) and not nd1,  # dict literal continued: the state at onset and whether the delay is a detection delay (either mark excludes it)
                     "first_decidable_s": float(f0 - a) if f0 is not None else np.nan, "delay_from_first_decidable_s": float(hit.min() - f0) if hit.size and f0 is not None else np.nan,  # dict literal continued: the second clock
                     "event_before_first_decidable": bool(hit.size) and (f0 is None or int(hit.min()) < f0),  # dict literal continued: the first event fell on a second without a state (regression and forest only)
                     "dec_only_detected": bool(hd.size), "dec_only_delay_s": float(hd.min() - a) if hd.size else np.nan, "dec_only_delay_from_first_decidable_s": float(hd.min() - f0) if hd.size and f0 is not None else np.nan,  # dict literal continued: the decidable-only reading
                     "onset_state_decidable": s2["state"], "onset_ref_age_decidable_s": s2["ref_age_s"], "dec_only_delay_counts": bool(hd.size) and not nd2,  # dict literal continued: its state at onset and its counted delay (either mark excludes it)
                     "pre60_decided_s": n_dc, "pre60_alarm_s": n_al, "pre60_share_of_60": n_al / ONSET_MAX_AGE, "pre60_share_of_decided": n_al / n_dc if n_dc else np.nan,  # dict literal continued: the minute before
                     "refractory_at_onset": r1, "last_event_age_s": g1, "no_detection_delay": nd1, "delay_counts_alarm_mark_only": bool(hit.size) and s1["state"] != "in alarm",  # dict literal continued: the refractory mark, the combined mark and the alarm-mark-only count for comparison
                     "dec_only_refractory_at_onset": r2, "dec_only_last_event_age_s": g2, "dec_only_no_detection_delay": nd2})  # dict literal continued: the same on the decidable seconds
    return rows, ev, ev_d  # the rows and both event sets


def phase_rows(base: dict, sec: np.ndarray, lab: np.ndarray, phase: np.ndarray, ok: np.ndarray, dec: np.ndarray, stop: np.ndarray, starts: dict) -> list:  # function: phase_core() for one run of the evaluation, with the run's per-second decisions recorded for report_2612()
    rows, ev, ev_d = phase_core(base, sec, lab, phase, ok, dec, stop, starts)  # tuple assignment: the rows and the two event sets
    # dict literal: the run's per-second record for report_2612() (decisions, stopping input and both event sets)
    DECISIONS.append({**base, "sec": sec, "label": lab, "phase": phase, "ok": np.asarray(ok, bool), "dec": np.asarray(dec), "stop": np.asarray(stop), "event": ev, "event_decidable": ev_d})
    return rows  # the rows


def evaluate_unseen(session: str, out: Path) -> None:  # function: every Part B detector on one session: 2.6.12 through the guard, or a training session as a dry run
    t0 = time.time()  # clock (printed only)
    out.mkdir(parents=True, exist_ok=True)  # the folder
    checks = []  # this step's checks
    chk = lambda name, ok, detail: checks.append({"check": name, "pass": bool(ok), "detail": detail})  # lambda: records one check
    if session == SEALED:  # the real evaluation: the guard before anything
        proof = guard()  # refuses unless every frozen file equals its copy in the freeze commit
        E.write_csv(proof, out / "guard.csv")  # the proof
        chk("guard: every frozen file equals its copy in the freeze commit and the manifest", True, f"{len(proof)} files, commit {FREEZE_COMMIT[:7]}")  # recorded
    A.use_labels("final")  # the scoring module points at the final label set
    wc10 = A.window_dict()  # the feature script's window dictionary at w = 10
    cells = A.d1_cells(A.load_cells(A.FINAL, A.W, True), wc10)  # the position sessions' windows (the regression script's sealed reader)
    fits = rt(UNSEEN / "fits.csv")  # the frozen fits and their digests
    pts = rt(UNSEEN / "points.csv")  # the frozen points
    sel = rt(UNSEEN / "selection.csv")  # the frozen budgets
    labs = pd.read_csv(E.PROC / "labels" / "labels_1hz_final.csv", dtype={"test_code": str}, usecols=["test_code", "receiver", "sec", "label", "phase", "exclusion_reason"])  # the final labels (label-side file, 2.6.12 included)
    steps11 = T.read_steps()  # the environment script's step table (position sessions), for the dry run's check
    phases = pd.read_csv(TRANSFER_DIR / "unseen_attack_2612_phases.csv")  # the inferred windows and decidable seconds (10_unseen_attack_phases.py)
    runs, lat, curve, phs, grids, wins, scores = [], [], [], [], [], [], []  # the output rows
    for rx in RXS:  # for loop over the receivers
        idx, X = cells[rx]  # the receiver's position-session windows
        rec5, _ = fit_p(idx, X, wc10, list(SESSIONS), {}, {"run": "transfer_part_b", "outer": OUTER_B, "inner": "", "receiver": rx})  # the five-session P refitted
        fp, fz = fingerprint(rec5), fits[(fits.receiver == rx) & (fits.fit == "five sessions")].iloc[0]  # tuple assignment: the refit's digests and the frozen ones
        same = all(fp[k] == fz[k] for k in fp)  # generator in all(): transform, regression and forest identical
        chk(f"{rx}: the five-session refit reproduces the frozen digests (transform, regression, forest)", same, fp["transform_sha256"][:12])  # recorded
        if not same:  # if: the refit differs
            raise SystemExit(f"STOP: {rx}'s five-session refit differs from the freeze; nothing is evaluated")  # stop before any read
        if session == SEALED:  # the real evaluation
            ix_s, X_s, _ = read_2612(rx)  # 2.6.12's windows: the single read (the guard runs inside again)
        else:  # the dry run on a training session
            r_ = A.eval_rows(idx, session)  # its windows in time order
            ix_s, X_s = idx.iloc[r_].reset_index(drop=True), X.iloc[r_].reset_index(drop=True)  # tuple assignment: index and values
        o_ = np.argsort(ix_s.sec.to_numpy(), kind="stable")  # time order
        ix_s, X_s = ix_s.iloc[o_].reset_index(drop=True), X_s.iloc[o_].reset_index(drop=True)  # tuple assignment: sorted
        Z = A.transform(rec5, X_s, wc10)  # the five-session transform (C order)
        sc = A.score(rec5, Z)  # decision value and both probabilities
        secs = ix_s.sec.to_numpy(np.int64)  # the window seconds
        lab = labs[(labs.test_code == session) & (labs.receiver == rx)]  # the session's labels
        st = steps_of(lab, secs)  # every second with window, run and validity
        chk(f"{rx}: every window's label equals the label file's", np.array_equal(st.set_index("sec").label.reindex(secs).to_numpy(), ix_s.label.to_numpy()), f"{len(secs)} windows")  # recorded
        if session != SEALED:  # the dry run: the step builder against the environment script's
            s11 = steps11[(steps11.receiver == rx) & (steps11.session == session)].sort_values("sec")  # the environment script's steps of the session
            chk(f"{rx}: steps_of() = the environment script's step table (window, run, validity) on {session}", np.array_equal(st.window.to_numpy(), s11.window.to_numpy()) and np.array_equal(st.run_id.to_numpy(), s11.run_id.to_numpy()) and np.array_equal(st.valid.to_numpy(), s11.valid.to_numpy()), f"{len(st)} seconds")  # recorded
        v = st[st.valid].reset_index(drop=True)  # the decidable seconds
        need = v.sec.to_numpy(np.int64)  # their seconds
        P = lagged(Z, secs, need, E.LAGS)  # primary states
        S = lagged(sc["dec"], secs, need, range(E.HIST, -1, -1))  # surrogate states
        ix = v[["sec", "label", "step", "run_id"]].assign(session=session, role="heldout", row=np.arange(len(v)))  # the index
        # dict literal over two lines: the material (m only builds the environment)
        mat = {"receiver": rx, "fold": OUTER_B, "primary": P, "surrogate": S, "index": ix, "key": {(session, int(t)): int(r) for t, r in zip(ix.sec, ix.row)}, "onsets": pd.DataFrame(columns=["session", "onset", "end", "start_lo", "start_hi", "length_s"]), "m": float(material_b(rx, OUTER_B)["m"]),
               "type": ix.step.map(E.STEP).to_numpy(np.int8), "run": ix.run_id.to_numpy(), "session": ix.session.to_numpy(), "sec": ix.sec.to_numpy(np.int64), "pre_rows": ix.row[ix.step == "pre"].to_numpy()}  # dict literal continued
        sd = st[st.window].assign(receiver=rx, session=session).reset_index(drop=True)  # the stream grid: windowed seconds
        pred = {"lr": pd.Series(sc["p_lr"], index=secs), "rf": pd.Series(sc["p_rf"], index=secs)}  # dict literal: probabilities by second
        rows_, ok, Zg = B12.grid_rows(mat, sd)  # each grid second's state row, validity and 30 log-odds
        sec, labv, phv = sd.sec.to_numpy(np.int64), sd.label.to_numpy(object), sd.phase.to_numpy(object)  # tuple assignment: grid arrays
        starts = {ph: int(lab[lab.phase == ph].sec.min()) for ph in PHASES if (lab.phase == ph).any()}  # dict comprehension: each phase's first second on the label grid
        grids.append(sd[["receiver", "session", "sec", "label", "phase", "valid"]].assign(decidable=ok))  # the grid for the RQ4 study and the figures
        scores.append(pd.DataFrame({"receiver": rx, "sec": secs, "p_lr": sc["p_lr"], "p_rf": sc["p_rf"], "dec": sc["dec"]}))  # every window's scores
        for ph in ("pre_test",) + PHASES + ("post_test",):  # for loop over the phases: real windows against the expected table's inference
            m_ = st.phase == ph  # its seconds
            c = phases[(phases.receiver == rx) & (phases.phase == ph)]  # the inferred row
            wins.append({"receiver": rx, "phase": ph, "seconds": int(m_.sum()), "windows": int((m_ & st.window).sum()), "decidable": int((m_ & st.valid).sum()), "windows_inspection": int(c.windows_inferred.iloc[0]) if session == SEALED and len(c) else None, "decidable_inspection": int(c.decidable_inferred.iloc[0]) if session == SEALED and len(c) else None})  # one row
        P_ = pts[(pts.receiver == rx) & (pts.outer == OUTER_B)]  # the receiver's frozen points
        for r in P_[P_.detector.isin(B12.BASE)].itertuples():  # for loop over the baselines' points
            n, k = (int(r.n), int(r.k)) if r.detector == "kofn" else (0, 0)  # k-of-n's setting
            d_, s_ = B12.baseline_decisions(r.detector, float(r.param), n, k, sd, pred, ok, Zg)  # the decisions
            fd, fs = B12.both(sd, d_, s_, session)  # through the environment script's stream (the label script's labels and fold policy)
            base = {"receiver": rx, "outer": session, "detector": r.detector, "share": r.share, "param": float(r.param), "lambda": np.nan, "seed": -1}  # the run's identity
            runs.append({**base, **B12.run_row(fd, fs)})  # dict unpacking: the run
            lat += [{**base, **x} for x in fs["latency"]]  # list comprehension: its intervals
            phs += phase_rows(base, sec, labv, phv, ok, d_, s_, starts)  # the decision
        for arm in ("primary", "surrogate"):  # the agents: every lambda and seed
            for lam in T.LAMBDAS:  # for loop over the lambdas
                b = int(sel[(sel.receiver == rx) & (sel.arm == arm) & (sel["lambda"] == float(lam))].selected_budget.iloc[0])  # its frozen budget
                for seed in T.OUTER_SEEDS:  # for loop over the five seeds
                    spec = {"stage": "outer", "receiver": rx, "outer": OUTER_B, "inner": "", "arm": arm, "lambda": float(lam), "seed": seed, "budget": b}  # the frozen agent
                    model, _ = T.load_q(spec, MODELS_B, mat)  # its weights
                    d_, s_ = T.agent_decisions(model, E.SpoofingDetectionEnv(mat, arm, float(lam)), sd)  # decision and stopping mode
                    fd, fs = B12.both(sd, d_, s_, session)  # through the stream
                    base = {"receiver": rx, "outer": session, "detector": f"agent_{arm}", "share": np.nan, "param": float(lam), "lambda": float(lam), "seed": seed}  # the run's identity
                    runs.append({**base, **B12.run_row(fd, fs)})  # dict unpacking: the run
                    lat += [{**base, **x} for x in fs["latency"]]  # list comprehension: its intervals
                    phs += phase_rows(base, sec, labv, phv, ok, d_, s_, starts)  # the decision
        kn = sorted({(int(r.n), int(r.k)) for r in P_[P_.detector == "kofn"].itertuples()})  # set comprehension: the k-of-n settings at the points
        for det, grid_, nk in [("lr", B12.LR_GRID, (0, 0)), ("rf", B12.LR_GRID, (0, 0)), ("cusum", B12.H_GRID, (0, 0))] + [("kofn", B12.KN_GRID, x) for x in kn]:  # each baseline's curve parameters (the evaluation script's grids)
            for prm in grid_:  # for loop over the parameter values
                d_, s_ = B12.baseline_decisions(det, float(prm), nk[0], nk[1], sd, pred, ok, Zg)  # the decisions
                fd, fs = B12.both(sd, d_, s_, session)  # through the stream
                curve.append({"receiver": rx, "outer": session, "detector": det, "n": nk[0], "k": nk[1], "param": float(prm), "clean": fd["clean"], "clean_in_alarm": fd["clean_in_alarm"], "stop_clean_in_alarm": fs["clean_in_alarm"], "intervals": fs["intervals"], "delays": ";".join(str(x["delay_s"]) for x in fs["latency"] if x["detected"])})  # the curve row
        print(f"{rx} {session}: {len(runs)} runs ({time.time() - t0:.0f} s)", flush=True)  # progress
    R_, LAT, C, PH = pd.DataFrame(runs), pd.DataFrame(lat), pd.DataFrame(curve), pd.DataFrame(phs)  # the tables
    pts_e = pts.assign(outer=session)  # the frozen points labelled with the evaluated session (unchanged for 2.6.12; the dry run's session otherwise)
    at = B12.point_runs(R_, pts_e)  # the runs at the frozen points
    ag = B12.aggregate(at, LAT, pts_e)  # IQM with bootstrap intervals (one session: the agents' intervals are over seeds only)
    vd = B12.verdicts(at, LAT, pts_e)  # P9 against the best baseline on this session
    cv = B12.curves(R_, LAT, C)  # the operating curves on this session
    pp = phase_points(PH, pts)  # the per-phase delays at the points
    outcome = tracking_outcome(session, labs)  # the loss of tracking, per receiver
    for name, df in (("runs.csv", R_), ("latency.csv", LAT), ("point_runs.csv", at), ("aggregate.csv", ag), ("verdicts.csv", vd), ("curves.csv", cv),  # tuple of (file, table)
                     ("phase_runs.csv", PH), ("phase_points.csv", pp), ("windows.csv", pd.DataFrame(wins)), ("outcome.csv", outcome), ("checks.csv", pd.DataFrame(checks))):  # tuple continued
        E.write_csv(df, out / name)  # written
    E.write_gz(C, out / "curve_runs.csv.gz")  # the curve runs
    E.write_gz(pd.concat(grids, ignore_index=True), out / "grid.csv.gz")  # the stream grids with decidability
    E.write_gz(pd.concat(scores, ignore_index=True), out / "scores.csv.gz")  # every window's scores
    if session == SEALED:  # the figures belong to the real evaluation only
        figures_b(out, pts)  # the figures
    ck = pd.DataFrame(checks)  # the checks
    print(ck.to_string(index=False))  # printed
    print(vd.to_string(index=False))  # printed
    print(f"{session}: {time.time() - t0:.0f} s")  # wall time (printed only)
    if not ck["pass"].all():  # if: a check failed
        sys.exit(1)  # exit status 1


def phase_points(PH: pd.DataFrame, pts: pd.DataFrame) -> pd.DataFrame:  # function: the per-phase delays at each detector's frozen point (agents: the five seeds at the point's lambda, summarised), with the onset states and clocks and the refractory mark
    med = lambda s: float(s.median()) if len(s) else np.nan  # lambda: the median of a Series, NaN when it is empty
    rows = []  # one row per receiver, share, detector and phase
    for r in pts.itertuples():  # for loop over the points
        if r.detector.startswith("agent"):  # an agent
            if str(r.no_point) == "True":  # no lambda meets the share
                continue  # nothing to report
            d = PH[(PH.receiver == r.receiver) & (PH.detector == r.detector) & (PH["lambda"] == r.param)]  # its five seeds
        else:  # a baseline
            d = PH[(PH.receiver == r.receiver) & (PH.detector == r.detector) & (PH.share == r.share)]  # its one run
        for ph, g in d.groupby("phase", sort=False):  # for loop over the phases
            dl = g.delay_s.dropna()  # the detected runs' delays
            sp = g.alarming_at_spoof_start.astype(str)  # F4's flag as text ("True", "False", "not decidable"; "None" in the first phase)
            # dict literal over nine lines: the row (run1's columns first, then the onset state
            rows.append({"receiver": r.receiver, "share": r.share, "detector": r.detector, "param": r.param, "phase": ph, "phase_start": int(g.phase_start.iloc[0]), "runs": len(g), "detected_runs": int(g.detected.sum()), "delay_median_s": float(dl.median()) if len(dl) else np.nan, "delay_min_s": float(dl.min()) if len(dl) else np.nan,
                         "delay_max_s": float(dl.max()) if len(dl) else np.nan, "decidable": int(g.decidable.iloc[0]), "decidable_attack": int(g.decidable_attack.iloc[0]), "alarming_at_spoof_start_runs": int((sp == "True").sum()) if ph == "spoof" else None,  # dict literal continued: run1's columns
                         "not_alarming_at_spoof_start_runs": int((sp == "False").sum()) if ph == "spoof" else None, "not_decidable_at_spoof_start_runs": int((sp == "not decidable").sum()) if ph == "spoof" else None,  # dict literal continued: F4's other two values, counted separately
                         "runs_in_alarm_at_onset": int((g.onset_state == "in alarm").sum()), "runs_not_in_alarm_at_onset": int((g.onset_state == "not in alarm").sum()), "runs_not_decidable_at_onset": int((g.onset_state == "not decidable").sum()),  # dict literal continued: the states at onset
                         "onset_ref_age_min_s": float(g.onset_ref_age_s.min()), "onset_ref_age_max_s": float(g.onset_ref_age_s.max()),  # dict literal continued: the ages of the seconds the states are read on
                         "counting_runs": int(g.delay_counts.sum()), "delay_median_counting_s": med(g.delay_s[g.delay_counts.astype(bool)]),  # dict literal continued: the counted detection delays (in-alarm runs left out)
                         "first_decidable_s": float(g.first_decidable_s.iloc[0]), "delay_from_first_decidable_median_s": med(g.delay_from_first_decidable_s.dropna()), "events_before_first_decidable_runs": int(g.event_before_first_decidable.sum()),  # dict literal continued: the second clock
                         "dec_only_detected_runs": int(g.dec_only_detected.sum()), "dec_only_delay_median_s": med(g.dec_only_delay_s.dropna()), "dec_only_delay_from_first_decidable_median_s": med(g.dec_only_delay_from_first_decidable_s.dropna()),  # dict literal continued: the decidable-only reading
                         "dec_only_counting_runs": int(g.dec_only_delay_counts.sum()), "dec_only_delay_median_counting_s": med(g.dec_only_delay_s[g.dec_only_delay_counts.astype(bool)]),  # dict literal continued: its counted delays
                         "pre60_share_of_60_median": med(g.pre60_share_of_60), "pre60_share_of_decided_median": med(g.pre60_share_of_decided.dropna()), "pre60_decided_s_median": med(g.pre60_decided_s),  # dict literal continued: the minute before
                         "grid_seconds": int(g.grid_seconds.iloc[0]), "runs_refractory_at_onset": int(g.refractory_at_onset.sum()), "runs_no_detection_delay": int(g.no_detection_delay.sum()),  # dict literal continued: the refractory mark and the combined mark (the counted delays above use the combined mark)
                         "counting_runs_alarm_mark_only": int(g.delay_counts_alarm_mark_only.sum()), "delay_median_counting_alarm_mark_only_s": med(g.delay_s[g.delay_counts_alarm_mark_only.astype(bool)]),  # dict literal continued: the alarm-mark-only count, for comparison
                         "dec_only_runs_refractory_at_onset": int(g.dec_only_refractory_at_onset.sum())})  # dict literal continued: the mark on the decidable seconds
    return pd.DataFrame(rows)  # the table


def tracking_outcome(session: str, labs: pd.DataFrame) -> pd.DataFrame:  # function: the loss of tracking inside the attack interval per receiver: an outcome in its own right
    rows = []  # one row per receiver
    iv = E.SC.pooled_intervals(session)  # the pooled attack interval(s)
    tl = pd.read_csv(E.PROC / "ground_truth" / "early_warning_timeline.csv", dtype={"test_code": str})  # the evidence script's traced instants
    for rx in RXS:  # for loop over the receivers
        l = labs[(labs.test_code == session) & (labs.receiver == rx)]  # its labels
        inside = np.zeros(len(l), bool)  # seconds inside an attack interval
        for a, b in iv:  # for loop over the intervals
            inside |= (l.sec >= a).to_numpy() & (l.sec <= b).to_numpy()  # inside this one
        no = inside & (l.exclusion_reason == "no_observations").to_numpy()  # no observation at all
        t = tl[(tl.test_code == session) & (tl.receiver == rx)]  # its traced instants
        # dict literal over two lines: the row
        rows.append({"receiver": rx, "session": session, "interval_seconds": int(inside.sum()), "no_observation_seconds": int(no.sum()), "no_observation_share": float(no.sum() / inside.sum()) if inside.sum() else np.nan, "first_no_observation": E.pd.Timestamp(int(l.sec[no].min()), unit="s").strftime("%H:%M:%S") if no.any() else "",
                     "last_no_observation": E.pd.Timestamp(int(l.sec[no].max()), unit="s").strftime("%H:%M:%S") if no.any() else "", "loss_of_tracking": str(t.t2612_loss_of_tracking.iloc[0]) if len(t) and "t2612_loss_of_tracking" in t else "", "reacquisition": str(t.t2612_reacquisition.iloc[0]) if len(t) and "t2612_reacquisition" in t else ""})  # dict literal continued
    return pd.DataFrame(rows)  # the table


def figures_b(out: Path, pts: pd.DataFrame) -> None:  # function: the figures (2.6.12 second by second) and 64 (delay per phase), with a hollow marker for no detection delay (in alarm or refractory at onset), the first decidable second, and no miss drawn where a phase has no window
    import matplotlib.pyplot as plt  # pyplot (09 chose the file-only backend)
    from matplotlib.lines import Line2D  # legend entries drawn by hand
    G_ = pd.read_csv(out / "grid.csv.gz")  # the grids
    SC_ = pd.read_csv(out / "scores.csv.gz")  # the scores
    pp = rt(out / "phase_points.csv")  # the per-phase delays
    lab_col = {"clean": "#009E73", "attack_jam_spoof": "#D55E00", "attack_spoof": "#CC79A7", "uncertain": "#BBBBBB"}  # dict: label -> colour (the regression script's palette)
    t0 = int(G_.sec.min())  # the first second
    starts = {ph: int(pp[pp.phase == ph].phase_start.iloc[0]) for ph in PHASES if (pp.phase == ph).any()}  # dict comprehension: each phase's first second on the label grid
    fig, axes = plt.subplots(3, 1, figsize=(12, 10.5), sharex=True)  # one panel per receiver
    for ax, rx in zip(axes, ("UBX_01", "SBF_03", "SBF_02")):  # for loop over the receivers, primary first
        g = G_[G_.receiver == rx]  # its grid
        x = (g.sec.to_numpy() - t0) / 60  # minutes
        for lb, col in lab_col.items():  # for loop over the labels
            m = (g.label == lb).to_numpy()  # its seconds
            ax.bar(x[m], 0.6, bottom=6.2, width=1 / 60, color=col, linewidth=0, label=lb)  # the label strip
        ax.bar(x[~g.decidable.to_numpy()], 0.25, bottom=5.85, width=1 / 60, color="#56B4E9", linewidth=0, label="no state (window or history missing)")  # undecidable seconds
        s = SC_[SC_.receiver == rx]  # its scores
        ax.plot((s.sec.to_numpy() - t0) / 60, 5.0 + 0.7 * s.p_lr.to_numpy(), color="#0072B2", lw=0.7, label="logistic regression probability (0 to 1, scaled)")  # the probability trace
        for j, det in enumerate(B12.DET):  # for loop over the detectors: the first alarm event of each phase at the 0.05 point
            col, mk, lbl = B12.STYLE[det]  # style
            ax.axhline(4 - 0.7 * j, color="#EEEEEE", lw=0.5, zorder=0)  # the detector's row
            ax.text(-0.3, 4 - 0.7 * j, lbl, fontsize=6, ha="right", va="center")  # its name
            q = pp[(pp.receiver == rx) & (pp.detector == det) & (pp.share == 0.05)]  # its per-phase delays at the point (none for an agent without a point)
            for r in q[q.delay_median_s.notna()].itertuples():  # for loop over the phases with a detection
                counted = pd.notna(r.delay_median_counting_s)  # a detection delay exists (some run has neither mark at onset)
                dly = r.delay_median_counting_s if counted else r.delay_median_s  # conditional expression: the counted median, else the median of the in-alarm runs
                ax.plot((r.phase_start + dly - t0) / 60, 4 - 0.7 * j, marker=mk, color=col, mfc=col if counted else "white", ms=7, ls="none")  # the first alarm event (agents: median over the seeds); hollow: in alarm at onset, no detection delay
        for ph in PHASES:  # for loop over the phases: the first decidable second
            d_ = g[(g.phase == ph) & g.decidable]  # the phase's decidable seconds
            if len(d_):  # if: any
                ax.axvline((d_.sec.min() - t0) / 60, color="#56B4E9", lw=1.0, ls="-.")  # its first decidable second
        ax.set_ylim(-0.2, 7.0)  # room for every strip
        ax.set_yticks([])  # no vertical scale
        ax.set_title(f"{NAME[rx]}, 2.6.12: labels, decidable seconds, the regression's probability, and each detector's first alarm in each phase at the 0.05 point", fontsize=9, loc="left")  # title
        for ph, ls in (("jam_spoof", "--"), ("spoof", ":")):  # for loop over the phase starts
            if ph in starts:  # if: the phase exists
                ax.axvline((starts[ph] - t0) / 60, color="#000000", lw=0.8, ls=ls)  # the phase start on the label grid
    axes[-1].set_xlabel(f"minutes after {pd.Timestamp(t0, unit='s').strftime('%H:%M:%S')} UTC (dashed: jamming and spoofing begins; dotted: time spoofing alone; dash-dot: first decidable second of a phase)", fontsize=8)  # axis
    h_, l_ = axes[0].get_legend_handles_labels()  # legend entries
    h_ += [Line2D([], [], marker="o", color="#555555", ls="none", ms=6), Line2D([], [], marker="o", color="#555555", mfc="white", ls="none", ms=6)]  # list concatenation: filled and hollow marker entries
    l_ += ["first alarm event: a detection delay", "hollow: in alarm or refractory at onset (no detection delay)"]  # their labels
    fig.legend(h_, l_, loc="lower center", ncol=3, fontsize=7, frameon=False)  # one legend
    fig.tight_layout(rect=(0.06, 0.06, 1, 1))  # layout with room for names and legend
    M.save_fig(fig, GRAPH / "timeline.png")  # byte-stable save
    fig, axes = plt.subplots(3, 2, figsize=(12, 9.5), sharey=True)  # receivers by phase
    top = 2000.0  # the height at which a miss is drawn (above every delay)
    for i, rx in enumerate(("UBX_01", "SBF_03", "SBF_02")):  # for loop over the receivers
        for j, ph in enumerate(PHASES):  # for loop over the phases
            ax = axes[i, j]  # the panel
            gsec = pp[(pp.receiver == rx) & (pp.phase == ph)].grid_seconds  # the phase's windowed seconds on this receiver
            nowin = not len(gsec) or int(gsec.max()) == 0  # no window at all: no detector could decide (the decision at the transfer study acceptance)
            if nowin:  # if: no window in the phase
                ax.text(0.5, 0.5, "no windows: no detector could decide", transform=ax.transAxes, ha="center", va="center", fontsize=9)  # stated, with no miss drawn
            for q, det in enumerate(B12.DET):  # for loop over the detectors (none when the phase has no window)
                if nowin:  # if: no window
                    break  # nothing to draw
                col, mk, lbl = B12.STYLE[det]  # style
                for k, s in enumerate(T.SHARES):  # for loop over the shares
                    r = pp[(pp.receiver == rx) & (pp.phase == ph) & (pp.detector == det) & (pp.share == s)]  # the point's row
                    xpos = q + (k - 1) * 0.22  # dodged position: 0.02, 0.05, 0.10 from left to right
                    if r.empty:  # no point (agent) at this share
                        continue  # nothing
                    r = r.iloc[0]  # the row
                    if pd.isna(r.delay_median_s):  # missed in every run
                        ax.plot(xpos, top, marker="x", color=col, ms=6, ls="none")  # an x at the top
                        continue  # next share
                    counted = pd.notna(r.delay_median_counting_s)  # a detection delay exists
                    y = r.delay_median_counting_s if counted else r.delay_median_s  # conditional expression: the counted median, else the in-alarm runs' median
                    lo, hi = min(y, r.delay_min_s), max(y, r.delay_max_s)  # tuple assignment: the range over the runs
                    ax.errorbar(xpos, y, yerr=[[y - lo], [hi - y]], fmt=mk, color=col, mfc=col if counted else "white", ms=6, capsize=2, lw=1.0, label=lbl if (i == 0 and j == 0 and k == 1) else None)  # median over the runs with the range; hollow: in alarm at onset
            dd = pp[(pp.receiver == rx) & (pp.phase == ph)]  # the phase's rows
            fdec = float(dd.first_decidable_s.iloc[0]) if len(dd) else np.nan  # its first decidable second, from the phase's first second
            if pd.notna(fdec):  # if: the phase has a decidable second
                ax.axhline(fdec, color="#56B4E9", lw=1.0, ls="--")  # delays below it fell before any sequential detector could decide
            ftxt = f"first decidable second +{fdec:.0f} s" if pd.notna(fdec) else "no decidable second"  # conditional expression: the title's clock note
            ax.set_title(f"{NAME[rx]}, {'jamming and spoofing' if ph == 'jam_spoof' else 'time spoofing alone'}: {int(dd.decidable_attack.max()) if len(dd) else 0} decidable attack seconds; {ftxt}", fontsize=8, loc="left")  # title
            ax.set_yscale("symlog", linthresh=1.0)  # delays span orders of magnitude, and 0 s stays visible
            ax.set_ylim(-0.3, top * 1.6)  # room for the misses at the top
            ax.set_xticks(range(len(B12.DET)), [B12.STYLE[d][2] for d in B12.DET], rotation=25, fontsize=7, ha="right")  # list comprehension: detector names
            ax.grid(alpha=0.3, lw=0.5, axis="y")  # recessive grid
        axes[i, 0].set_ylabel("delay from the phase's first second (s)", fontsize=8)  # axis
    h_, l_ = axes[0, 0].get_legend_handles_labels()  # legend entries
    h_ += [Line2D([], [], marker="o", color="#555555", mfc="white", ls="none"), Line2D([], [], marker="x", color="#555555", ls="none"), Line2D([], [], color="#56B4E9", ls="--"), Line2D([], [], ls="none")]  # list concatenation: hand-drawn entries
    l_ += ["hollow: in alarm or refractory at onset (no detection delay)", "x at the top: missed", "dashed: the phase's first decidable second", "each detector: shares 0.02, 0.05, 0.10 left to right; agents: median of five seeds with range"]  # their labels
    fig.legend(h_, l_, loc="lower center", ncol=5, fontsize=7, frameon=False)  # one legend
    fig.tight_layout(rect=(0, 0.07, 1, 1))  # layout
    M.save_fig(fig, GRAPH / "phase_delays.png")  # byte-stable save


def report_2612(out: Path, recompute: bool = False, extra_checks: list | None = None) -> None:  # function: the reporting tables of the 2.6.12 evaluation, from the decisions phase_rows() recorded or recompute_2612() read back (nothing is chosen from them)
    checks = list(extra_checks or [])  # this step's checks, after the caller's (recompute_2612's)
    chk = lambda name, ok, detail: checks.append({"check": name, "pass": bool(ok), "detail": detail})  # lambda: records one check
    pts = rt(UNSEEN / "points.csv")  # the frozen points
    at = rt(out / "point_runs.csv")  # the registered runs at the points
    LAT = rt(out / "latency.csv")  # the registered per-interval delays
    vd = rt(out / "verdicts.csv")  # the registered verdicts
    pp = rt(out / "phase_points.csv")  # the per-phase delays at the points
    held = DECISIONS[0]["outer"]  # the evaluated session (2.6.12, a training session only when the transfer tests dry-tests this function)
    iv = E.SC.pooled_intervals(held)  # its pooled attack interval(s)
    ident = ("receiver", "outer", "detector", "share", "param", "lambda", "seed")  # tuple: the columns that name a run
    dec_rows, runs_d, lat_d, ons = [], [], [], []  # the decisions table, the decidable-only runs and intervals, the states at the interval onsets
    for d in DECISIONS:  # for loop over every recorded run (baselines at their points; agents at every lambda and seed)
        base = {k: d[k] for k in ident}  # dict comprehension: the run's identity
        sec, okb = d["sec"], d["ok"]  # tuple assignment: the grid seconds and the decidable mask
        decided = np.ones(len(sec), bool) if d["detector"] in OWN_ALL else okb  # conditional expression: the detector's own decided seconds
        # DataFrame from a dict of columns: the run's seconds with every decision
        dec_rows.append(pd.DataFrame({**base, "sec": sec, "label": d["label"], "phase": d["phase"], "decidable": okb, "dec": np.asarray(d["dec"]).astype(int), "stop_input": np.asarray(d["stop"]).astype(int), "event": d["event"], "event_decidable": d["event_decidable"]}))
        fd = E.stream(sec, d["label"], d["dec"], held, "decision", decidable=okb)  # decision mode on the decidable seconds only
        fs = E.stream(sec, d["label"], d["stop"], held, "stopping", decidable=okb)  # stopping mode on the decidable seconds only
        runs_d.append({**base, **B12.run_row(fd, fs)})  # dict unpacking: the decidable-only run
        lat_d += [{**base, **x} for x in fs["latency"]]  # list comprehension: its intervals
        for a, b in iv:  # for loop over the pooled intervals: the state at each onset
            ins = (sec >= a) & (sec <= b)  # the interval's grid seconds
            h1, h2 = sec[d["event"] & ins], sec[d["event_decidable"] & ins]  # tuple assignment: events inside it, both readings
            s1, s2 = onset_state(sec, d["dec"], decided, a), onset_state(sec, d["dec"], okb, a)  # tuple assignment: the state on the own clock and on the decidable seconds
            (r1, g1), (r2, g2) = refractory_at(sec, d["event"], a), refractory_at(sec, d["event_decidable"], a)  # nested tuple assignment: the refractory mark in both readings, with the last event's age
            nd1, nd2 = s1["state"] == "in alarm" or r1, s2["state"] == "in alarm" or r2  # tuple assignment: either mark (no detection delay)
            # dict literal over four lines: the run's row at this onset
            ons.append({**base, "onset": int(a), "end": int(b), "onset_state": s1["state"], "onset_ref_age_s": s1["ref_age_s"], "onset_ref_alarm": s1["ref_alarm"], "detected": bool(h1.size), "delay_s": float(h1.min() - a) if h1.size else np.nan, "delay_counts": bool(h1.size) and not nd1,
                        "onset_state_decidable": s2["state"], "onset_ref_age_decidable_s": s2["ref_age_s"], "dec_only_detected": bool(h2.size), "dec_only_delay_s": float(h2.min() - a) if h2.size else np.nan, "dec_only_delay_counts": bool(h2.size) and not nd2,  # dict literal continued: the decidable-only reading
                        "refractory_at_onset": r1, "last_event_age_s": g1, "no_detection_delay": nd1, "delay_counts_alarm_mark_only": bool(h1.size) and s1["state"] != "in alarm",  # dict literal continued: the refractory mark, the combined mark, the alarm-mark-only count
                        "dec_only_refractory_at_onset": r2, "dec_only_last_event_age_s": g2, "dec_only_no_detection_delay": nd2})  # dict literal continued: the same on the decidable seconds
        if d["detector"] not in OWN_ALL:  # a sequential detector or an agent: silent where no state exists
            chk(f"{d['receiver']} {d['detector']} {d['share']} {d['lambda']} {d['seed']}: events on the decidable seconds alone = the registered events", np.array_equal(d["event"], d["event_decidable"]), "")  # recorded (failures only are listed below)
    DEC = pd.concat(dec_rows, ignore_index=True)  # every recorded second
    RD, LD, ON = pd.DataFrame(runs_d), pd.DataFrame(lat_d), pd.DataFrame(ons)  # the tables
    nos = [c for c in ident if c != "share"]  # list comprehension: a run's identity without the share (an agent's recorded share is NaN; its points carry one)
    # the registered figures recomputed from the recorded events must equal the registered files
    m_ = ON.merge(LAT, on=list(ident) + ["onset", "end"], how="inner", suffixes=("", "_reg"), validate="one_to_one")  # every run's interval beside its registered row
    same_delay = ((m_.delay_s == m_.delay_s_reg) | (m_.delay_s.isna() & m_.delay_s_reg.isna())).all() and (m_.detected == m_.detected_reg.astype(bool)).all()  # bitwise equal, NaN equal to NaN
    chk("the interval delays recomputed from the recorded events = latency.csv (every run)", same_delay and len(m_) == len(LAT), f"{len(m_)} of {len(LAT)} intervals")  # recorded
    rr = at.merge(pd.DataFrame([{**{k: d[k] for k in nos}, "n_clean": int((d["label"] == "clean").sum()), "n_clean_alarm": int(((np.asarray(d["dec"]) > 0) & (d["label"] == "clean")).sum())} for d in DECISIONS]), on=nos, how="left", validate="many_to_one")  # list comprehension inside a merge: every point run with its clean counts from the recorded decisions
    chk("the decision-mode clean counts from the recorded decisions = point_runs.csv", (rr.n_clean == rr.dec_clean).all() and (rr.n_clean_alarm == rr.dec_clean_in_alarm).all(), f"{len(rr)} point runs")  # recorded
    if held == SEALED:  # the real evaluation: the touched phase table against run1's
        run1 = rt(TRANSFER_DIR / "unseen2612_run1" / "phase_runs.csv")  # run1's phase rows (the committed copy)
        ph_now = rt(out / "phase_runs.csv")  # the phase rows of this run
        keep = ["receiver", "outer", "detector", "share", "param", "lambda", "seed", "phase", "phase_start", "grid_seconds", "decidable", "decidable_attack", "detected", "delay_s"]  # list: run1's columns whose meaning is unchanged
        chk("the phase rows keep run1's values in every column whose meaning is unchanged", ph_now[keep].equals(run1[keep]), f"{len(run1)} rows, {len(keep)} columns")  # recorded
    # (a) realised decision-mode shares at the points, against the target
    pr = at.assign(target=at.share, above_twice=at.dec_clean_share > 2 * at.share)  # every point run with its target and the per-run comparison
    rs = pr.groupby(["receiver", "detector", "share"], sort=True).agg(runs=("seed", "size"), clean=("dec_clean", "sum"), clean_in_alarm=("dec_clean_in_alarm", "sum"), runs_above_twice=("above_twice", "sum"), run_share_min=("dec_clean_share", "min"), run_share_max=("dec_clean_share", "max")).reset_index()  # per cell: pooled counts and the spread over runs
    rs["realised_share"] = rs.clean_in_alarm / rs.clean  # the pooled realised share (decision mode, every clean windowed second)
    rs["point_not_held"] = rs.realised_share > 2 * rs.share  # the 'point not held' mark: more than twice the target
    rdp_at = at[list(ident)].merge(RD.drop(columns="share"), on=nos, how="left", validate="many_to_one")  # the decidable-only runs at the points, each with its point's share
    rsd = rdp_at.groupby(["receiver", "detector", "share"], sort=True).agg(clean_decidable=("dec_clean", "sum"), clean_decidable_in_alarm=("dec_clean_in_alarm", "sum")).reset_index()  # per cell: the decidable clean seconds and those in alarm
    rs = rs.merge(rsd, on=["receiver", "detector", "share"], how="left", validate="one_to_one")  # beside the registered share
    rs["realised_share_decidable"] = rs.clean_decidable_in_alarm / rs.clean_decidable.where(rs.clean_decidable > 0)  # the share over decidable clean seconds only
    # (d) the annotated verdict table: the registered verdict copied, descriptive columns beside it, no second verdict
    onp = at[list(ident)].merge(ON.drop(columns="share"), on=nos, how="inner", validate="many_to_one")  # the states at onset of the point runs only, each with its point's share
    rows = []  # one row per registered verdict
    for v in vd.itertuples(index=False):  # for loop over the registered verdicts
        row = dict(v._asdict())  # the registered row, copied unchanged
        for b in B12.BASE:  # for loop over the baselines: each one's figures at this share
            o = onp[(onp.receiver == v.receiver) & (onp.detector == b) & (onp.share == v.share)]  # its run's state and delay
            c = rs[(rs.receiver == v.receiver) & (rs.detector == b) & (rs.share == v.share)]  # its realised share
            q = pp[(pp.receiver == v.receiver) & (pp.detector == b) & (pp.share == v.share) & (pp.phase == "spoof")]  # its time-spoofing phase row
            # dict update over three lines: the baseline's descriptive columns
            row.update({f"{b}_realised_share": float(c.realised_share.iloc[0]), f"{b}_point_not_held": bool(c.point_not_held.iloc[0]), f"{b}_onset_state": o.onset_state.iloc[0], f"{b}_onset_ref_age_s": o.onset_ref_age_s.iloc[0], f"{b}_refractory_at_onset": bool(o.refractory_at_onset.iloc[0]), f"{b}_no_detection_delay": bool(o.no_detection_delay.iloc[0]), f"{b}_delay_s": o.delay_s.iloc[0],
                        f"{b}_delay_counted_s": o.delay_s.iloc[0] if bool(o.delay_counts.iloc[0]) else np.nan, f"{b}_dec_only_delay_s": o.dec_only_delay_s.iloc[0], f"{b}_dec_only_onset_state": o.onset_state_decidable.iloc[0],  # dict update continued: the decidable-only reading
                        f"{b}_spoof_delay_s": float(q.delay_median_s.iloc[0]) if len(q) else np.nan, f"{b}_spoof_delay_from_first_decidable_s": float(q.delay_from_first_decidable_median_s.iloc[0]) if len(q) else np.nan, f"{b}_spoof_onset_state": ("in alarm" if q.runs_in_alarm_at_onset.iloc[0] else ("not in alarm" if q.runs_not_in_alarm_at_onset.iloc[0] else "not decidable")) if len(q) else "", f"{b}_spoof_refractory_at_onset": bool(q.runs_refractory_at_onset.iloc[0]) if len(q) else False})  # dict update continued: the time-spoofing phase
        p_ = pts[(pts.receiver == v.receiver) & (pts.detector == f"agent_{v.arm}") & (pts.share == v.share)]  # the agent's point
        if len(p_) and str(p_.no_point.iloc[0]) != "True":  # if: the agent has a point at this share
            o = onp[(onp.receiver == v.receiver) & (onp.detector == f"agent_{v.arm}") & (onp["lambda"] == float(p_.param.iloc[0])) & (onp.share == v.share)]  # its five seeds' states and delays at this share
            c = rs[(rs.receiver == v.receiver) & (rs.detector == f"agent_{v.arm}") & (rs.share == v.share)]  # its realised share
            q = pp[(pp.receiver == v.receiver) & (pp.detector == f"agent_{v.arm}") & (pp.share == v.share) & (pp.phase == "spoof")]  # its time-spoofing phase row
            # dict update over four lines: the agent's descriptive columns
            row.update({"agent_lambda": float(p_.param.iloc[0]), "agent_realised_share": float(c.realised_share.iloc[0]), "agent_point_not_held": bool(c.point_not_held.iloc[0]), "agent_seeds_above_twice": int(c.runs_above_twice.iloc[0]),
                        "agent_seeds_in_alarm_at_onset": int((o.onset_state == "in alarm").sum()), "agent_seeds_not_in_alarm_at_onset": int((o.onset_state == "not in alarm").sum()), "agent_seeds_not_decidable_at_onset": int((o.onset_state == "not decidable").sum()),  # dict update continued: the states
                        "agent_seeds_refractory_at_onset": int(o.refractory_at_onset.sum()), "agent_seeds_no_detection_delay": int(o.no_detection_delay.sum()),  # dict update continued: the refractory mark and the combined mark
                        "agent_delay_median_s": float(o.delay_s.dropna().median()) if o.detected.any() else np.nan, "agent_delay_median_counted_s": float(o.delay_s[o.delay_counts].median()) if o.delay_counts.any() else np.nan,  # dict update continued: the delays, all and counted
                        "agent_spoof_delay_s": float(q.delay_median_s.iloc[0]) if len(q) else np.nan, "agent_spoof_delay_from_first_decidable_s": float(q.delay_from_first_decidable_median_s.iloc[0]) if len(q) else np.nan})  # dict update continued: the time-spoofing phase
        bb = v.best_baseline  # the registered best baseline
        mark = "in alarm at onset" if row[f"{bb}_onset_state"] == "in alarm" else ("refractory at onset" if row[f"{bb}_refractory_at_onset"] else "")  # conditional expressions: the best baseline's mark, if any (the in-alarm mark first, then the refractory mark)
        row["annotation"] = "registered verdict unchanged" + (f"; best baseline {bb} is {mark} (no detection delay)" if mark else "") + ("; best baseline's point not held" if row[f"{bb}_point_not_held"] else "")  # conditional expressions: the descriptive annotation (no second verdict; the registered note column is kept as it is)
        rows.append(row)  # the row
    VA = pd.DataFrame(rows)  # the annotated table
    chk("verdicts_annotated.csv carries every registered verdict unchanged", VA[list(vd.columns)].equals(vd), f"{len(vd)} rows")  # recorded
    bad = [c for c in checks if not c["pass"]]  # the failed checks
    ck = pd.DataFrame([c for c in checks if not c["check"].endswith("= the registered events")] + [{"check": "every sequential detector's and agent's events on the decidable seconds alone = its registered events", "pass": not any(c["check"].endswith("= the registered events") and not c["pass"] for c in checks), "detail": f"{sum(c['check'].endswith('= the registered events') for c in checks)} runs"}])  # list comprehensions: the per-run event checks folded into one row
    if recompute:  # the recomputation: only the onset states and the annotated table change; the other tables must equal the committed files byte for byte
        tmp = PROJECT / "codes" / ".tmp_recompute2612"  # a temporary folder inside the project (removed at the end)
        tmp.mkdir(parents=True, exist_ok=True)  # created
        for name, df in (("realised_shares.csv", rs), ("runs_decidable.csv", RD), ("latency_decidable.csv", LD)):  # tuple of (file, table): the tables that must not change
            E.write_csv(df, tmp / name)  # rendered with the same writer
            ck.loc[len(ck)] = {"check": f"recomputed {name} = the committed file, byte for byte", "pass": (tmp / name).read_bytes() == (out / name).read_bytes(), "detail": ""}  # recorded
        E.write_gz(DEC, tmp / "decisions.csv.gz")  # the decisions rendered again
        ck.loc[len(ck)] = {"check": "recomputed decisions.csv.gz = the committed file, byte for byte", "pass": (tmp / "decisions.csv.gz").read_bytes() == (out / "decisions.csv.gz").read_bytes(), "detail": ""}  # recorded
        import shutil as _sh  # file-tree removal
        _sh.rmtree(tmp)  # the temporary folder removed
        bad = [c for c in ck.to_dict("records") if not c["pass"]]  # the failed checks, again with the new rows
        for name, df in (("onset_states.csv", ON), ("verdicts_annotated.csv", VA), ("checks_recompute.csv", ck)):  # tuple of (file, table): the recomputed tables
            E.write_csv(df, out / name)  # written
    else:  # the evaluation run
        for name, df in (("onset_states.csv", ON), ("realised_shares.csv", rs), ("runs_decidable.csv", RD), ("latency_decidable.csv", LD), ("verdicts_annotated.csv", VA), ("checks_part2.csv", ck)):  # tuple of (file, table)
            E.write_csv(df, out / name)  # written
        E.write_gz(DEC, out / "decisions.csv.gz")  # every recorded second
    print(ck.to_string(index=False))  # printed
    print(VA[["receiver", "share", "arm", "verdict", "best_baseline", "annotation"]].to_string(index=False))  # printed
    if bad:  # if: a check failed
        print([c["check"] for c in bad][:10])  # the first failures
        sys.exit(1)  # exit status 1


def recompute_2612(out: Path) -> None:  # function: the recomputation of the 2.6.12 phase tables, onset states, annotated verdicts and the figures from the per-second decisions already written (no read of 2.6.12's windows)
    checks = []  # this step's checks
    chk = lambda name, ok, detail: checks.append({"check": name, "pass": bool(ok), "detail": detail})  # lambda: records one check
    DEC = rt(out / "decisions.csv.gz")  # (the evaluation script's reader keeps "outer" as text) every recorded run's per-second decisions (written by report_2612)
    labs = pd.read_csv(E.PROC / "labels" / "labels_1hz_final.csv", dtype={"test_code": str}, usecols=["test_code", "receiver", "sec", "phase"])  # the final labels (label-side), for each phase's first second
    pts = rt(UNSEEN / "points.csv")  # the frozen points
    import io  # standard library: bytes as a file, for a table read from a commit
    old = rt(io.BytesIO(git("show", f"{PART_B_COMMIT}:Data/processed/transfer/unseen_attack/phase_runs.csv").stdout))  # the phase rows as committed at the transfer study (f4a9e29), whatever is on disk now
    ident = ["receiver", "outer", "detector", "share", "param", "lambda", "seed"]  # list: the columns that name a run
    DECISIONS.clear()  # the module's record, rebuilt from the file
    rows, same_ev = [], True  # the phase rows; whether every event set is reproduced
    for k, d in DEC.groupby(ident, dropna=False, sort=False):  # for loop over the recorded runs, in the order they were written
        base = {"receiver": str(k[0]), "outer": str(k[1]), "detector": str(k[2]), "share": float(k[3]), "param": float(k[4]), "lambda": float(k[5]), "seed": int(k[6])}  # dict literal: the run's identity with the evaluation's types
        lab = labs[(labs.test_code == base["outer"]) & (labs.receiver == base["receiver"])]  # the session's labels on this receiver
        starts = {ph: int(lab[lab.phase == ph].sec.min()) for ph in PHASES if (lab.phase == ph).any()}  # dict comprehension: each phase's first second on the label grid (as evaluate_unseen builds it)
        sec, okb = d.sec.to_numpy(np.int64), d.decidable.to_numpy(bool)  # tuple assignment: the grid seconds and the decidable mask
        lb, phv, dec, stop = d.label.to_numpy(object), d.phase.to_numpy(object), d.dec.to_numpy(), d.stop_input.to_numpy()  # tuple assignment: labels, phases, decisions, stopping-mode input
        r_, ev, ev_d = phase_core(base, sec, lb, phv, okb, dec, stop, starts)  # tuple assignment: the rows and both event sets
        same_ev = same_ev and np.array_equal(ev, d.event.to_numpy(bool)) and np.array_equal(ev_d, d.event_decidable.to_numpy(bool))  # the events recomputed equal the written ones
        rows += r_  # kept
        DECISIONS.append({**base, "sec": sec, "label": lb, "phase": phv, "ok": okb, "dec": dec, "stop": stop, "event": ev, "event_decidable": ev_d})  # the run's record, as phase_rows() keeps it
    chk("both event sets recomputed from the written decisions = the written events (every run)", same_ev, f"{len(DECISIONS)} runs")  # recorded
    PH = pd.DataFrame(rows)  # the recomputed phase rows
    E.write_csv(PH, out / "phase_runs.csv")  # written
    new = rt(out / "phase_runs.csv")  # read back as every reader reads it
    keep = [c for c in old.columns if c not in ("delay_counts", "dec_only_delay_counts")]  # list comprehension: the committed columns whose meaning the onset marks do not change
    chk("the recomputed phase rows equal the committed ones in every column whose meaning is unchanged", new[keep].equals(old[keep]), f"{len(old)} rows, {len(keep)} columns")  # recorded
    chk("the alarm-mark-only count (delay_counts_alarm_mark_only) equals the committed delay_counts", new.delay_counts_alarm_mark_only.equals(old.delay_counts) and new.dec_only_delay_counts.le(old.dec_only_delay_counts).all(), "")  # recorded (the combined mark can only remove delays)
    E.write_csv(phase_points(PH, pts), out / "phase_points.csv")  # the per-phase delays at the points
    report_2612(out, recompute=True, extra_checks=checks)  # the onset states and the annotated table (the other tables checked against the committed files)
    figures_b(out, pts)  # the figures


# ----------------------------------------------------------------------------------------------------------------------
# --diag2612: the diagnostic of u-blox 01's shift on 2.6.12's clean seconds (finding F1); nothing is chosen from it and no detector changes
# ----------------------------------------------------------------------------------------------------------------------
DIAG = U2612 / "clean_alarm_diagnosis"  # the diagnostic's outputs
CONST = {"G": "GPS L1", "R": "GLONASS G1", "E": "Galileo E1", "C": "BeiDou B1"}  # dict: constellation letter -> name and band
STAT = {"mean": "mean", "std": "standard deviation", "median": "median", "min": "minimum", "max": "maximum", "skew": "skewness", "kurt": "kurtosis", "sos": "sum of squares", "sosdiff": "last squared value minus first squared value", "tk": "mean Teager-Kaiser energy"}  # dict: the feature script's window statistic -> words


def column_parts(name: str) -> tuple:  # function: a D1 column name split into (constellation letter or "", the per-second quantity, the window statistic)
    base, stat = name.split("__")  # tuple assignment: 07 joins the base column and the statistic with a double underscore
    c = base[0] if base[:2] in ("G_", "R_", "E_", "C_") else ""  # conditional expression: the constellation, or none for an all-constellation quantity
    return c, base[2:] if c else base, stat  # the three parts


def column_group(name: str) -> tuple:  # function: the decision's grouping: (level or change, the kind)
    c, q, stat = column_parts(name)  # its parts
    if stat in ("sosdiff", "tk"):  # a statistic of change within the window
        return "change", stat  # change column
    for k in ("dC_dc", "cmc_rate", "d2C", "dop_rate"):  # for loop over the per-second differences (dop_rate, Doppler minus the previous epoch's, is grouped here by this script, not by the decision)
        if q.startswith(k):  # if: the quantity is that difference
            return "change", k  # change column
    if q.startswith("n_"):  # a satellite count
        return "level", "counts"  # level column
    if q.startswith("dop_"):  # the raw Doppler
        return "level", "raw Doppler"  # level column
    if q.startswith("cn0_"):  # the C/N0
        return "level", "C/N0 level"  # level column
    return "other", q  # anything else, named


def column_meaning(name: str, lin: pd.DataFrame) -> str:  # function: a column's meaning in words, from the feature script's lineage table (constellation, per-second quantity, window statistic)
    c, q, stat = column_parts(name)  # its parts
    base = name.split("__")[0]  # the base column
    d = lin[lin.column == base].definition  # its definition in features_base_lineage.csv
    dtext = str(d.iloc[0]).split(" [")[0] if len(d) else q  # the definition without the bracketed block note
    return f"{CONST.get(c, 'all four constellations')}; {dtext}; window statistic over 10 s: {STAT.get(stat, stat)}"  # the meaning


def smd(a: np.ndarray, b: np.ndarray) -> np.ndarray:  # function: standardised mean difference of each column, (mean a - mean b) / sqrt((var a + var b) / 2), ddof 1
    va, vb = a.var(axis=0, ddof=1), b.var(axis=0, ddof=1)  # tuple assignment: the column variances
    s = np.sqrt((va + vb) / 2)  # the pooled standard deviation
    return np.where(s > 0, (a.mean(axis=0) - b.mean(axis=0)) / np.where(s > 0, s, 1), np.nan)  # NaN where both groups are constant


def hms_of(sec) -> str:  # function: unix seconds -> HH:MM:SS (empty for NaN)
    return pd.Timestamp(int(sec), unit="s").strftime("%H:%M:%S") if pd.notna(sec) else ""  # conditional expression: the clock text


def diagnose_2612(out: Path) -> None:  # function: F1's diagnostic per receiver (u-blox 01 first; the mosaic-H and the mosaic-X5 as controls), through the guarded reader
    from scipy.special import expit  # the logistic function, for probabilities from decision values
    t0 = time.time()  # clock (printed only)
    out.mkdir(parents=True, exist_ok=True)  # the folder
    checks = []  # this step's checks
    chk = lambda name, ok, detail: checks.append({"check": name, "pass": bool(ok), "detail": detail})  # lambda: records one check
    guard()  # refuses unless every frozen file equals its copy in the freeze commit (read_2612 runs it again)
    A.use_labels("final")  # the scoring module points at the final label set
    wc10 = A.window_dict()  # the feature script's window dictionary at w = 10
    cells = A.d1_cells(A.load_cells(A.FINAL, A.W, True), wc10)  # the position sessions' windows (the regression script's sealed reader)
    fits = rt(UNSEEN / "fits.csv")  # the frozen fits and their digests
    lin = pd.read_csv(E.PROC / "features" / "features_base_lineage.csv").drop_duplicates("column")  # the feature script's per-second column definitions
    nest = pd.read_csv(UNSEEN / "nested_predictions.csv.gz", dtype={"held_out": str, "outer_held_out": str})  # the five-session P's nested out-of-fold scores
    sc_eval = rt(U2612 / "scores.csv.gz")  # the evaluation's decision values on 2.6.12 (this run's), every float parsed back to the double written
    labs = pd.read_csv(E.PROC / "labels" / "labels_1hz_final.csv", dtype={"test_code": str}, usecols=["test_code", "receiver", "sec", "label", "phase"])  # the final labels of every session
    sm_rows, dc_rows, id_rows, gp_rows, vals = [], [], [], [], []  # SMDs, contributions, identities, groups, the top columns' values
    for rx in ("UBX_01", "SBF_03", "SBF_02"):  # for loop over the receivers, primary first, then the controls
        idx, X = cells[rx]  # the receiver's position-session windows
        rec5, _ = fit_p(idx, X, wc10, list(SESSIONS), {}, {"run": "R13diag", "outer": OUTER_B, "inner": "", "receiver": rx})  # the five-session P refitted
        fp, fz = fingerprint(rec5), fits[(fits.receiver == rx) & (fits.fit == "five sessions")].iloc[0]  # tuple assignment: the refit's digests and the frozen ones
        same = all(fp[k] == fz[k] for k in fp)  # generator in all(): transform, regression and forest identical
        chk(f"{rx}: the five-session refit reproduces the frozen digests", same, fp["transform_sha256"][:12])  # recorded
        if not same:  # if: the refit differs
            raise SystemExit(f"STOP: {rx}'s refit differs from the freeze")  # stop before any read
        ix_s, X_s, _ = read_2612(rx)  # 2.6.12's windows: a diagnostic read through the guarded reader (counted in the report)
        o_ = np.argsort(ix_s.sec.to_numpy(), kind="stable")  # time order
        ix_s, X_s = ix_s.iloc[o_].reset_index(drop=True), X_s.iloc[o_].reset_index(drop=True)  # tuple assignment: sorted
        lr = rec5["model"]["lr"]  # the five-session regression
        w = lr.coef_.ravel()  # its 36 coefficients
        names = list(rec5["names"])  # its 36 column names, in the coefficients' order
        Zs, Zp = A.transform(rec5, X_s, wc10), A.transform(rec5, X, wc10)  # tuple assignment: the five-session transform on 2.6.12 and on every position window (C order)
        ds, dp = A.decision(lr, Zs), A.decision(lr, Zp)  # tuple assignment: the decision values
        Rs, Rp = X_s[names].to_numpy(float), X[names].to_numpy(float)  # tuple assignment: the same 36 columns' raw window values (the feature script's units, NaN where missing), for the raw means
        ev_ = sc_eval[sc_eval.receiver == rx].sort_values("sec")  # the evaluation's decision values on 2.6.12
        chk(f"{rx}: the diagnostic's 2.6.12 decision values = the evaluation's (scores.csv.gz), bitwise", np.array_equal(ev_.sec.to_numpy(), ix_s.sec.to_numpy()) and np.array_equal(ev_.dec.to_numpy(), ds), f"{len(ds)} windows")  # recorded
        tgt = (ix_s.label == "clean").to_numpy()  # 2.6.12's clean windows (the pre-test; on the Septentrio receivers the post-test too)
        tc, lb = idx.test_code.to_numpy(), idx.label.to_numpy()  # tuple assignment: the position windows' sessions and scoring labels
        refs = [(f"{s} clean" + (" (Stave stop)" if s == "2.6.5" else ""), (tc == s) & (lb == "clean")) for s in SESSIONS]  # list comprehension: each session's clean windows
        refs += [("pooled clean (five sessions)", lb == "clean"), ("2.6.11 all windows", tc == "2.6.11"), ("2.6.11 attack windows", (tc == "2.6.11") & np.isin(lb, A.ATTACK))]  # list concatenation: the pooled reference and the decision's 2.6.11 references
        for ref, mk in refs:  # for loop over the references
            if mk.sum() < 2:  # if: fewer than two windows (no variance): reported as such
                id_rows.append({"receiver": rx, "reference": ref, "n_target": int(tgt.sum()), "n_reference": int(mk.sum()), "note": "fewer than two windows"})  # the row
                continue  # next reference
            dm = Zs[tgt].mean(axis=0) - Zp[mk].mean(axis=0)  # the mean difference of each column
            con = w * dm  # coefficient times mean difference: each column's contribution to the shift
            dobs = float(ds[tgt].mean() - dp[mk].mean())  # the observed shift of the mean decision value
            sd_ = smd(Zs[tgt], Zp[mk])  # the standardised mean differences
            for j, nm in enumerate(names):  # for loop over the 36 columns
                g1, g2 = column_group(nm)  # its group
                # dict literal over two lines: the SMD row with the raw means and missing shares (a raw mean over the windows where the value exists)
                sm_rows.append({"receiver": rx, "reference": ref, "column": nm, "smd": float(sd_[j]), "mean_target": float(Zs[tgt, j].mean()), "mean_reference": float(Zp[mk, j].mean()), "group": g1, "kind": g2,
                                # dict entry: raw means of the feature in target and reference windows, and their missing shares
                                "raw_mean_target": float(np.nanmean(Rs[tgt, j])) if np.isfinite(Rs[tgt, j]).any() else np.nan, "raw_mean_reference": float(np.nanmean(Rp[mk, j])) if np.isfinite(Rp[mk, j]).any() else np.nan, "raw_missing_target": float(np.isnan(Rs[tgt, j]).mean()), "raw_missing_reference": float(np.isnan(Rp[mk, j]).mean())})
                dc_rows.append({"receiver": rx, "reference": ref, "column": nm, "coefficient": float(w[j]), "mean_difference": float(dm[j]), "contribution": float(con[j]), "group": g1, "kind": g2, "meaning": column_meaning(nm, lin)})  # the contribution row
            n_ = nest[(nest.model == "lr") & (nest.receiver == rx) & (nest.label == "clean") & (nest.held_out == ref.split(" ")[0])] if ref.endswith(("clean", "(Stave stop)")) and not ref.startswith("pooled") else pd.DataFrame(columns=["p"])  # conditional expression: the session's nested out-of-fold clean scores
            # dict literal over three lines: the identity row
            id_rows.append({"receiver": rx, "reference": ref, "n_target": int(tgt.sum()), "n_reference": int(mk.sum()), "mean_dec_target": float(ds[tgt].mean()), "mean_dec_reference": float(dp[mk].mean()), "observed_shift": dobs, "sum_of_contributions": float(con.sum()), "abs_difference": abs(float(con.sum()) - dobs),
                            "median_p_target": float(np.median(expit(ds[tgt]))), "median_p_reference_in_sample": float(np.median(expit(dp[mk]))), "median_p_reference_nested": float(n_.p.median()) if len(n_) else np.nan,  # dict literal continued: probabilities
                            "level_sum": float(sum(con[j] for j, nm in enumerate(names) if column_group(nm)[0] == "level")), "change_sum": float(sum(con[j] for j, nm in enumerate(names) if column_group(nm)[0] == "change")), "note": ""})  # generators in sum(): the two groups' totals
            chk(f"{rx} vs {ref}: sum of coefficient x mean difference = the observed shift of the mean decision value (to 1e-9)", abs(float(con.sum()) - dobs) <= 1e-9, f"{abs(float(con.sum()) - dobs):.2e}")  # recorded
        if rx == "UBX_01":  # the primary receiver: the values of its top three columns for the figure
            dpo = pd.DataFrame(dc_rows)  # the contributions so far
            top3 = dpo[(dpo.receiver == rx) & (dpo.reference == "pooled clean (five sessions)")].assign(a=lambda d: d.contribution.abs()).sort_values(["a", "column"], ascending=[False, True]).column.head(3).tolist()  # lambda in assign: the three largest absolute contributions
            for nm in top3:  # for loop over them
                j = names.index(nm)  # its position
                vals.append(pd.DataFrame({"column": nm, "group": "2.6.12 clean", "value": Zs[tgt, j]}))  # 2.6.12's clean windows
                for ref, mk in refs:  # for loop over the references
                    if ref.startswith(("pooled", "2.6.11 attack")) or mk.sum() == 0:  # the pooled set and the attack subset are not drawn; empty sets skipped
                        continue  # next
                    vals.append(pd.DataFrame({"column": nm, "group": ref, "value": Zp[mk, j]}))  # the reference's windows
        print(f"{rx}: diagnostic ({time.time() - t0:.0f} s)", flush=True)  # progress
    SM, DC, ID = pd.DataFrame(sm_rows), pd.DataFrame(dc_rows), pd.DataFrame(id_rows)  # the tables
    DC["rank"] = DC.assign(a=DC.contribution.abs()).groupby(["receiver", "reference"]).a.rank(ascending=False, method="first").astype(int)  # rank by absolute contribution within each receiver and reference
    GP = DC.groupby(["receiver", "reference", "group", "kind"], sort=True).contribution.agg(["sum", "size"]).reset_index().rename(columns={"sum": "contribution", "size": "columns"})  # (iii) the contributions summed per group
    # (ii) every session's UTC span of clean and attack seconds per receiver
    sp_rows = []  # one row per receiver, session and class
    for (t, rx), l in labs.groupby(["test_code", "receiver"], sort=True):  # for loop over the sessions and receivers
        for cls, mk in (("clean", l.label == "clean"), ("attack", l.label.isin(A.ATTACK))):  # for loop over the two classes
            s_ = l.sec[mk]  # their seconds
            sp_rows.append({"receiver": rx, "session": t, "class": cls, "seconds": int(mk.sum()), "first_utc": hms_of(s_.min()) if len(s_) else "", "last_utc": hms_of(s_.max()) if len(s_) else "", "phases": ";".join(sorted(l.phase[mk].dropna().unique()))})  # the row
    SP = pd.DataFrame(sp_rows)  # the table
    diag_context(out, labs, sc_eval, checks)  # the clock, satellite, capture and onset context, and the Stave check
    for name, df in (("smd.csv", SM), ("contributions.csv", DC), ("identity.csv", ID), ("groups.csv", GP), ("spans.csv", SP), ("checks.csv", pd.DataFrame(checks))):  # tuple of (file, table)
        E.write_csv(df, out / name)  # written
    E.write_gz(pd.concat(vals, ignore_index=True), out / "top3_values.csv.gz")  # the top columns' values for the figure
    plot_ublox_shift(out)  # the figure
    ck = pd.DataFrame(checks)  # the checks
    print(ck.to_string(index=False))  # printed
    print(ID[["receiver", "reference", "n_reference", "observed_shift", "level_sum", "change_sum", "median_p_target", "median_p_reference_nested"]].to_string(index=False))  # printed
    print(f"diagnostic: {time.time() - t0:.0f} s")  # wall time (printed only)
    if not ck["pass"].all():  # if: a check failed
        sys.exit(1)  # exit status 1


def diag_context(out: Path, labs: pd.DataFrame, sc_eval: pd.DataFrame, checks: list) -> None:  # function: the diagnostic's context from tables already in Data/processed (clocks, satellites and C/N0, capture boundaries, the README remark, the onset seconds of F5) and the Stave check
    chk = lambda name, ok, detail: checks.append({"check": name, "pass": bool(ok), "detail": detail})  # lambda: records one check in the caller's list
    stop = pd.read_csv(E.PROC / "preprocessing" / "vehicle_motion_stops.csv")  # the vehicle's stops
    st_ = stop[stop.nearest_recording_site.str.startswith("Stave")].iloc[0]  # the 2.6.5 stop at Stave
    a_, b_ = (pd.Timestamp(st_.start_utc).timestamp(), pd.Timestamp(st_.end_used_utc).timestamp())  # tuple: its span in unix seconds
    c5 = labs[(labs.test_code == "2.6.5") & (labs.label == "clean")]  # 2.6.5's clean seconds
    chk("2.6.5's clean seconds lie inside the vehicle's stop at Stave (preprocessing/vehicle_motion_stops.csv)", ((c5.sec >= a_) & (c5.sec <= b_)).all(), f"{hms_of(c5.sec.min())}-{hms_of(c5.sec.max())} inside {st_.start_utc[11:19]}-{st_.end_used_utc[11:19]}")  # recorded
    # context: clocks, satellites and C/N0, capture boundaries, the README remark, the onset seconds (F5)
    lat_h = pd.read_csv(E.PROC / "ground_truth" / "receiver_host_latency.csv", dtype={"test_code": str})  # the receiver-status script's host-minus-receiver latency per session
    ck_rows = [{"source": "ground_truth/receiver_host_latency.csv", "receiver": r.receiver, "session": r.test_code, "quantity": "median host minus receiver time over the margin seconds (s)", "n": int(r.n_kept), "value": r.median_s} for r in lat_h[lat_h.receiver == "u-blox_01"].itertuples()]  # list comprehension: u-blox 01's rows
    rc = pd.read_csv(E.PROC / "preprocessing" / "rinex_clock_2612_ubx01.csv")  # the conversion script's per-epoch clock table of u-blox 01 in 2.6.12
    pre = rc[rc.period == "before the attack window"]  # the pre-test epochs
    rt_ = pd.to_datetime(pre.receiver_time_utc)  # receiver time
    ht_ = pd.to_datetime(pre.receiver_time_utc.str[:11] + pre.host_time_utc)  # host time on the same date
    ck_rows.append({"source": "preprocessing/rinex_clock_2612_ubx01.csv", "receiver": "u-blox_01", "session": "2.6.12 pre-test", "quantity": "median host minus receiver time of the RINEX epochs (s)", "n": len(pre), "value": float(((ht_ - rt_).dt.total_seconds()).median())})  # the row
    ck_rows.append({"source": "preprocessing/rinex_clock_2612_ubx01.csv", "receiver": "u-blox_01", "session": "2.6.12 pre-test", "quantity": "median receiver clock bias cdt (ms)", "n": len(pre), "value": float(pre.cdt_ms.median())})  # the row
    rs_ = pd.read_csv(E.PROC / "ground_truth" / "receiver_status_timeseries.csv", dtype={"test_code": str}, usecols=["test_code", "receiver", "timestamp_utc", "reported_minus_host_s"])  # receiver status (context only, never a feature): the reported time minus host time
    rs_ = rs_[(rs_.receiver == "u-blox_01") & rs_.reported_minus_host_s.notna()]  # u-blox 01's rows with the quantity
    rs_ = rs_.assign(sec=np.floor(pd.to_datetime(rs_.timestamp_utc, utc=True).astype("int64") / 1e9).astype(np.int64))  # whole unix seconds
    cl_ = labs[(labs.receiver == "UBX_01") & (labs.label == "clean")][["test_code", "sec"]]  # u-blox 01's clean seconds
    rm_ = rs_.merge(cl_, on=["test_code", "sec"], how="inner")  # the status rows on clean seconds
    ck_rows += [{"source": "ground_truth/receiver_status_timeseries.csv", "receiver": "u-blox_01", "session": t + (" pre-test" if t == SEALED else ""), "quantity": "median reported minus host time on clean seconds (s)", "n": len(g), "value": float(g.reported_minus_host_s.median())} for t, g in rm_.groupby("test_code", sort=True)]  # list comprehension: per session
    CK = pd.DataFrame(ck_rows)  # the clock table
    obs = pd.read_csv(E.PROC / "features" / "observables_epochs_1hz.csv.gz", dtype={"test_code": str}, usecols=["test_code", "receiver", "sec", "n_G", "n_R", "n_E", "n_C", "n_tracked", "cn0_median", "r_cn0_median", "dop_drift_m_s"])  # the observables script's per-second epoch table
    ol = obs.merge(labs, on=["test_code", "receiver", "sec"], how="inner")  # with the labels
    ob_rows = []  # one row per receiver and group of seconds
    for rx in ("UBX_01", "SBF_03", "SBF_02"):  # for loop over the receivers
        o = ol[ol.receiver == rx]  # its seconds
        grp = [("2.6.12 clean", (o.test_code == SEALED) & (o.label == "clean"))] + [(f"{s} clean" + (" (Stave stop)" if s == "2.6.5" else ""), (o.test_code == s) & (o.label == "clean")) for s in SESSIONS] + [("2.6.11 all seconds", o.test_code == "2.6.11"), ("2.6.11 attack seconds", (o.test_code == "2.6.11") & o.label.isin(A.ATTACK))]  # list comprehension and concatenation: the groups
        for gname, mk in grp:  # for loop over the groups
            g = o[mk]  # its seconds
            ob_rows.append({"receiver": rx, "group": gname, "seconds": len(g), **{f"median_{c}": float(g[c].median()) if len(g) else np.nan for c in ("n_G", "n_R", "n_E", "n_C", "n_tracked", "cn0_median", "r_cn0_median", "dop_drift_m_s")}})  # dict comprehension inside a dict literal: the medians
    OB = pd.DataFrame(ob_rows)  # the table (the observables script's table starts 2.6.5 at 12:34:57, so its Stave seconds have no row here: the raw window means in smd.csv cover them)
    gt = pd.read_csv(E.PROC / "ground_truth" / "session_ground_truth.csv", dtype={"test_code": str}).set_index("test_code")  # the timeline script's sessions
    cp_ = pd.DataFrame([{"session": t, "capture_start_utc": gt.at[t, "capture_start_utc"], "capture_end_utc": gt.at[t, "capture_end_utc"], "readme_start": gt.at[t, "capture_start_raw_readme"], "readme_end": gt.at[t, "capture_end_raw_readme"]} for t in ("2.6.11", SEALED)])  # list comprehension: the two captures
    gap = (pd.Timestamp(gt.at[SEALED, "capture_start_utc"]) - pd.Timestamp(gt.at["2.6.11", "capture_end_utc"])).total_seconds()  # the seconds between 2.6.11's end and 2.6.12's start
    cp_["gap_after_2611_s"] = [np.nan, gap]  # the gap on 2.6.12's row
    readme = (PROJECT / "Data" / "Day1_14092026" / gt.at[SEALED, "session_folder"] / "README.txt").read_text(encoding="utf-8").splitlines()  # 2.6.12's capture README (read only)
    k = readme.index("User Remarks")  # the heading of the remark
    remark = next(x for x in readme[k + 2:] if x.strip())  # generator in next(): the first non-empty line after the heading's underline, verbatim
    on = int(E.SC.pooled_intervals(SEALED)[0][0])  # 2.6.12's onset
    f5 = sc_eval[(sc_eval.sec >= on - 10) & (sc_eval.sec <= on + 10)][["receiver", "sec", "p_lr"]].merge(obs[obs.test_code == SEALED][["receiver", "sec", "n_tracked", "cn0_median"]], on=["receiver", "sec"], how="left")  # F5: the regression's probability beside the C/N0 and the satellite count around the onset
    f5 = f5.assign(t_rel_s=f5.sec - on).sort_values(["receiver", "sec"]).reset_index(drop=True)  # seconds from the onset
    for name, df in (("clock.csv", CK), ("observables.csv", OB), ("capture.csv", cp_), ("onset_f5.csv", f5)):  # tuple of (file, table)
        E.write_csv(df, out / name)  # written
    (out / "readme_remark.txt").write_text(remark + "\n", encoding="utf-8")  # the README remark, verbatim


def load_script(name: str, filename: str):  # function: loads codes/<filename> as a module by path (its main() is guarded and not run)
    spec = importlib.util.spec_from_file_location(name, PROJECT / "codes" / filename)  # where it is
    mod = importlib.util.module_from_spec(spec)  # an empty module object
    sys.modules[name] = mod  # registered
    spec.loader.exec_module(mod)  # its top level: constants, functions, classes
    return mod  # the module


VIS_MASKS = (5.0, 10.0)  # tuple: the elevation masks of the visibility count, degrees (the decision at the transfer study acceptance)


def visibility(out: Path) -> None:  # function: the orbit-based count of satellites above 5 and 10 degrees at each session's site, once per minute over each group of seconds, beside the tracked counts (sky against obstruction); no 2.6.12 window is read; then the figure
    import datetime as dtm  # standard library: the recording date for the orbit records
    checks = []  # this step's checks
    chk = lambda name, ok, detail: checks.append({"check": name, "pass": bool(ok), "detail": detail})  # lambda: records one check
    M5 = load_script("reference_positions_for_transfer_visibility", "2_preprocessing/03_reference_positions.py")  # 05c: Orbits (GPS, Galileo, BeiDou from the IGS file), geometry(), the site rules
    S04 = load_script("convert_rinex_for_transfer_visibility", "2_preprocessing/01_convert_rinex.py")  # the conversion script: the IGS file, its SHA-256 and the Kepler orbit functions
    G8 = load_script("glonass_orbit_for_transfer_visibility", "2_preprocessing/04_glonass_orbit.py")  # 08a: the GLONASS broadcast propagator
    sha = hashlib.sha256(S04.IGS_BRDC.read_bytes()).hexdigest()  # the IGS broadcast file's SHA-256 (Data/external/igs_brdc, no new download)
    chk("the IGS broadcast file is the logged one (SHA-256)", sha == S04.IGS_SHA256, sha[:12])  # recorded
    orb, glo = M5.Orbits(S04, dtm.date(2026, 9, 14)), G8.GloOrbits()  # tuple assignment: the two orbit sources
    svs = sorted(orb.keph) + sorted(orb.bds) + sorted(glo.rec)  # list concatenation: every satellite with a broadcast record (GPS, Galileo, BeiDou, GLONASS)
    pos_tab = pd.read_csv(E.PROC / "preprocessing" / "reference_positions.csv")  # the position script's positions
    site_x = {r.site: np.array([r.x_m, r.y_m, r.z_m]) for r in pos_tab[pos_tab.scope == "site"].itertuples()}  # dict comprehension: site -> ECEF metres
    site_of = dict(M5.SITE_OF)  # session -> site (the position script's table)
    labs = pd.read_csv(E.PROC / "labels" / "labels_1hz_final.csv", dtype={"test_code": str}, usecols=["test_code", "receiver", "sec", "label"])  # the final labels (label-side)

    def counts(site: str, t: int) -> dict:  # nested function: satellites above each mask at a site and unix second, per constellation
        t_gps = float(t) + M5.LEAP_SECONDS - M5.GPS_UNIX  # unix seconds -> GPS seconds (the helpers module' leap seconds, through the position script)
        x = site_x[site]  # the site
        pr = np.full(len(svs), 2.2e7)  # a nominal pseudorange (about 73 ms of travel); it moves an elevation by far less than 0.01 degree
        p1, _, ok1 = orb.state(svs, t_gps, pr)  # GPS, Galileo and BeiDou positions (GLONASS rows stay unusable here)
        p2, _, ok2 = glo.state(svs, t_gps, pr)  # GLONASS positions (the others stay unusable there)
        pos, ok = np.where(ok1[:, None], p1, p2), ok1 | ok2  # tuple assignment: one position per satellite from whichever source has it
        _, _, el, _ = M5.geometry(x, pos[ok])  # elevations of the usable satellites
        sv_ok = np.array(svs)[ok]  # their names
        out_ = {}  # the counts
        for m in VIS_MASKS:  # for loop over the masks
            for c in ("G", "R", "E", "C"):  # for loop over the constellations
                out_[f"{c}_above_{m:g}"] = int(((np.degrees(el) >= m) & (np.char.startswith(sv_ok.astype(str), c))).sum())  # satellites of that constellation above the mask
        return out_  # the counts

    rows = []  # one row per receiver, group and minute
    for rx in RXS:  # for loop over the receivers (their clean seconds differ slightly)
        l = labs[labs.receiver == rx]  # its labels
        groups = [(f"{s} clean" + (" (Stave stop)" if s == "2.6.5" else ""), s, "Stave" if s == "2.6.5" else site_of[s], l[(l.test_code == s) & (l.label == "clean")].sec) for s in list(SESSIONS) + [SEALED]]  # list comprehension: each session's clean seconds and its site (2.6.5's clean seconds lie at the Stave stop)
        groups += [("2.6.11 attack seconds", "2.6.11", site_of["2.6.11"], l[(l.test_code == "2.6.11") & l.label.isin(A.ATTACK)].sec), ("2.6.11 all seconds", "2.6.11", site_of["2.6.11"], l[l.test_code == "2.6.11"].sec)]  # list concatenation: 2.6.11's attack seconds and all its seconds
        for gname, s, site, secs in groups:  # for loop over the groups
            mins = sorted({int(t) // 60 * 60 for t in secs})  # set comprehension: the minutes the group's seconds fall in
            for m0 in mins:  # for loop over those minutes
                t = int(min(t for t in secs if int(t) // 60 * 60 == m0))  # generator in min(): the group's first second in that minute
                rows.append({"receiver": rx, "group": gname, "session": s, "site": site, "sec": t, "utc": pd.Timestamp(t, unit="s").strftime("%H:%M:%S"), **counts(site, t)})  # the row
    V = pd.DataFrame(rows)  # one row per receiver, group and minute
    ob = rt(out / "observables.csv")  # the tracked counts (the observables script's per-second table, medians over the same groups)
    sm = rt(out / "smd.csv")  # the raw window means (u-blox 01's counts at the Stave stop, which the observables script's table lacks)
    sum_rows = []  # one row per receiver and group
    for (rx, gname), g in V.groupby(["receiver", "group"], sort=False):  # for loop over the groups
        o = ob[(ob.receiver == rx) & (ob.group == gname)]  # its tracked medians
        r = {"receiver": rx, "group": gname, "site": g.site.iloc[0], "minutes": len(g), "first_utc": g.utc.min(), "last_utc": g.utc.max()}  # dict literal: the group
        r.update({f"predicted_{k}_median": float(g[k].median()) for k in V.columns if k.endswith(("_5", "_10"))})  # dict comprehension: the predicted medians
        r.update({f"tracked_{c}_median": (float(o[f"median_n_{c}"].iloc[0]) if len(o) and pd.notna(o[f"median_n_{c}"].iloc[0]) else np.nan) for c in ("G", "R", "E", "C")})  # dict comprehension: the tracked medians
        ws = sm[(sm.receiver == rx) & (sm.column == "G_n_tracked__mean") & (sm.reference == gname.replace(" seconds", " windows"))]  # u-blox 01's raw window mean of the GPS count, where its fit selected the column
        r["tracked_G_window_mean"] = float(ws.raw_mean_reference.iloc[0]) if len(ws) else (float(sm[(sm.receiver == rx) & (sm.column == "G_n_tracked__mean")].raw_mean_target.iloc[0]) if gname == "2.6.12 clean" and rx == "UBX_01" else np.nan)  # conditional expressions: the window mean, or 2.6.12's for its own group
        sum_rows.append(r)  # the row
    VS = pd.DataFrame(sum_rows)  # the summary
    # the hand check: one GPS and one GLONASS elevation against the observables script's and the GLONASS orbit script's tables at the 2.6.2 site (no 2.6.12 row is used)
    ref = None  # the observables script's row
    for ch in pd.read_csv(E.PROC / "features" / "observables_1hz.csv.gz", dtype={"test_code": str}, usecols=["test_code", "receiver", "sec", "sv", "el_deg"], chunksize=500_000):  # for loop over chunks: only 2.6.2's rows are kept
        c_ = ch[(ch.test_code == "2.6.2") & (ch.receiver == "SBF_02") & ch.sv.str.startswith("G") & ch.el_deg.notna()]  # 2.6.2's GPS rows with an elevation
        if len(c_):  # if: this chunk has one
            ref = c_.sort_values(["sec", "sv"]).iloc[0]  # the earliest
            break  # one is enough
    gl = pd.read_csv(E.PROC / "preprocessing" / "glonass_residuals_clean.csv", dtype={"test_code": str})  # the GLONASS orbit script's GLONASS elevations (clean intervals)
    gref = gl[(gl.test_code == "2.6.2") & (gl.receiver == "SBF_02")].sort_values(["sec", "sv"]).iloc[0]  # the earliest 2.6.2 row
    for name_, r_ in (("GPS (features/observables_1hz.csv.gz)", ref), ("GLONASS (preprocessing/glonass_residuals_clean.csv)", gref)):  # tuple of (source, row)
        t_gps = float(r_.sec) + M5.LEAP_SECONDS - M5.GPS_UNIX  # its second in GPS time
        p1, _, ok1 = orb.state([r_.sv], t_gps, np.array([2.2e7]))  # the satellite by the position script's orbits
        p2, _, ok2 = glo.state([r_.sv], t_gps, np.array([2.2e7]))  # or by the GLONASS orbit script's
        p = p1 if ok1[0] else p2  # whichever has it
        el = float(np.degrees(M5.geometry(site_x[site_of["2.6.2"]], p)[2][0]))  # its elevation here
        chk(f"hand check: {r_.sv} at {pd.Timestamp(int(r_.sec), unit='s').strftime('%H:%M:%S')} on 2.6.2, elevation here = {name_} within 0.05 deg", abs(el - float(r_.el_deg)) <= 0.05, f"{el:.4f} vs {float(r_.el_deg):.4f} deg")  # recorded
    E.write_csv(V, out / "visibility.csv")  # per minute
    E.write_csv(VS, out / "visibility_summary.csv")  # per group, beside the tracked counts
    E.write_csv(pd.DataFrame(checks), out / "visibility_checks.csv")  # the checks
    plot_ublox_shift(out)  # the figure with the visibility panel
    print(pd.DataFrame(checks).to_string(index=False))  # printed
    print(VS[["receiver", "group", "minutes", "predicted_G_above_5_median", "predicted_G_above_10_median", "tracked_G_median", "tracked_G_window_mean"]].to_string(index=False))  # printed
    if not all(c["pass"] for c in checks):  # if: a check failed
        sys.exit(1)  # exit status 1


def plot_ublox_shift(out: Path) -> None:  # function: the figure, u-blox 01's shift: the largest contributions to the decision-value shift, and the top three columns' values on 2.6.12 against 2.6.5 at Stave and the other sessions
    import matplotlib.pyplot as plt  # pyplot (09 chose the file-only backend)
    from matplotlib.patches import Patch  # legend patches
    DC = rt(out / "contributions.csv")  # the contributions
    ID = rt(out / "identity.csv")  # the identities
    V = pd.read_csv(out / "top3_values.csv.gz", dtype={"group": str})  # the top columns' values
    d = DC[(DC.receiver == "UBX_01") & (DC.reference == "pooled clean (five sessions)")].sort_values("rank").head(12).iloc[::-1]  # the twelve largest, smallest at the bottom
    col = {"level": "#E69F00", "change": "#0072B2", "other": "#999999"}  # dict: group -> colour
    vis = out / "visibility_summary.csv"  # the visibility summary, once --visibility has run
    VS = rt(vis) if vis.exists() else None  # conditional expression: the table, or none
    fig = plt.figure(figsize=(13, 7.5 if VS is None else 10.5))  # the figure (taller with the visibility panel)
    gs = fig.add_gridspec(3 if VS is None else 4, 2, width_ratios=(1.25, 1), height_ratios=None if VS is None else (1, 1, 1, 1.15))  # three rows on the right, one tall panel on the left, and the visibility row below
    ax = fig.add_subplot(gs[:3, 0])  # the contributions panel
    y = np.arange(len(d))  # one row per column
    ax.barh(y, d.contribution, color=[col[g] for g in d.group], height=0.6)  # list comprehension: the contribution against the pooled clean seconds
    for ref, mk in (("2.6.5 clean (Stave stop)", "D"), ("2.6.11 all windows", "o")):  # for loop over the two comparison references
        e = DC[(DC.receiver == "UBX_01") & (DC.reference == ref)].set_index("column").contribution.reindex(d.column)  # the same columns' contributions against it
        ax.plot(e.to_numpy(), y, marker=mk, ls="none", color="#000000", mfc="white", ms=5, label=f"against {ref}")  # markers
    ax.set_yticks(y, d.column, fontsize=7)  # the column names
    ax.axvline(0, color="#000000", lw=0.6)  # zero
    i_ = ID[(ID.receiver == "UBX_01") & (ID.reference == "pooled clean (five sessions)")].iloc[0]  # the pooled identity row
    ax.set_xlabel("contribution to the shift of the mean decision value: coefficient x (mean on 2.6.12 clean - mean on the reference)", fontsize=7)  # axis
    ax.set_title(f"u-blox 01: shift {i_.observed_shift:+.2f} against the pooled clean seconds (level columns {i_.level_sum:+.2f}, change columns {i_.change_sum:+.2f})", fontsize=8, loc="left")  # title
    h_, l_ = ax.get_legend_handles_labels()  # legend entries
    ax.legend(h_ + [Patch(color=col["level"]), Patch(color=col["change"])], l_ + ["level column (counts, raw Doppler, C/N0 level)", "change column (dC_dc, cmc_rate, d2C, dop_rate, sosdiff, tk)"], fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.08), ncol=2)  # legend below the panel, clear of the bars
    ax.grid(alpha=0.3, lw=0.5, axis="x")  # recessive grid
    order = ["2.6.11 all windows", "2.6.4 clean", "2.6.10 clean", "2.6.2 clean", "2.6.5 clean (Stave stop)", "2.6.12 clean"]  # the groups drawn, bottom to top (2.6.12 on top)
    tops = d.sort_values("rank").column.head(3).tolist()  # the top three columns
    for i, nm in enumerate(tops):  # for loop over them
        a2 = fig.add_subplot(gs[i, 1])  # its panel
        groups = [g for g in order if ((V.column == nm) & (V.group == g)).any()]  # list comprehension: the groups with values
        a2.boxplot([V[(V.column == nm) & (V.group == g)].value.to_numpy() for g in groups], vert=False, widths=0.6, flierprops={"markersize": 2})  # list comprehension: one box per group
        a2.set_yticks(range(1, len(groups) + 1), [f"{g} (n {int(((V.column == nm) & (V.group == g)).sum())})" for g in groups], fontsize=6.5)  # list comprehension: group names with counts
        a2.set_title(f"{nm} (transformed value)", fontsize=8, loc="left")  # title
        a2.grid(alpha=0.3, lw=0.5, axis="x")  # recessive grid
    if VS is not None:  # the visibility panel: GPS satellites predicted above 5 and 10 degrees against those tracked
        av = fig.add_subplot(gs[3, :])  # the bottom row
        gord = ["2.6.12 clean", "2.6.5 clean (Stave stop)", "2.6.2 clean", "2.6.10 clean", "2.6.4 clean", "2.6.11 attack seconds"]  # the groups, in order
        u_ = VS[VS.receiver == "UBX_01"].set_index("group").reindex(gord)  # u-blox 01's rows (the predicted counts are the same for every receiver at a site and minute)
        xg = np.arange(len(gord))  # one position per group
        av.bar(xg - 0.3, u_.predicted_G_above_5_median, width=0.2, color="#BBBBBB", label="predicted above 5 deg (IGS broadcast orbits, site position)")  # predicted, 5 degrees
        av.bar(xg - 0.1, u_.predicted_G_above_10_median, width=0.2, color="#777777", label="predicted above 10 deg")  # predicted, 10 degrees
        for k, (rx, col) in enumerate((("UBX_01", "#009E73"), ("SBF_02", "#0072B2"), ("SBF_03", "#E69F00"))):  # for loop over the receivers' tracked counts
            t_ = VS[VS.receiver == rx].set_index("group").reindex(gord)  # its rows
            yv = t_.tracked_G_median.where(t_.tracked_G_median.notna(), t_.tracked_G_window_mean if rx == "UBX_01" else np.nan)  # the observables script's median, or u-blox 01's window mean where the observables script's table has no row (2.6.5 at Stave)
            av.bar(xg + 0.1 + 0.1 * k, yv, width=0.1, color=col, label=f"tracked, {NAME[rx]} (median; u-blox 01 at Stave: window mean)" if rx == "UBX_01" else f"tracked, {NAME[rx]} (median)")  # tracked
        av.set_xticks(xg, [f"{g}\n{u_.first_utc.iloc[i]}-{u_.last_utc.iloc[i]} UTC\n{u_.site.iloc[i]}" for i, g in enumerate(gord)], fontsize=6.5)  # list comprehension: group, time and site on three lines
        av.set_ylabel("GPS satellites", fontsize=8)  # axis
        av.set_title("GPS satellites predicted above the mask at the site against those tracked, per group of seconds", fontsize=8, loc="left")  # title
        av.set_ylim(0, 19)  # headroom above the bars for the legend
        av.legend(fontsize=6.5, frameon=False, ncol=3, loc="upper center")  # legend, above the bars
        av.grid(alpha=0.3, lw=0.5, axis="y")  # recessive grid
    fig.tight_layout()  # layout
    M.save_fig(fig, GRAPH / "ublox_shift.png")  # byte-stable save


def main() -> None:  # function: the command line
    ap = argparse.ArgumentParser()  # the parser
    ap.add_argument("--transfer", action="store_true")  # Part A
    ap.add_argument("--unseen", action="store_true")  # Part B training and freezing
    ap.add_argument("--jobs-sample", default="")  # the transfer tests' T-F: 12 evenly spaced inner jobs run again into this folder (relative to the project)
    ap.add_argument("--heldout2612", action="store_true")  # --heldout2612: the single 2.6.12 evaluation
    ap.add_argument("--dryrun", default="")  # --dryrun: the same path on a training session, before 2.6.12 is read
    ap.add_argument("--diag2612", action="store_true")  # --diag2612: the diagnostic of u-blox 01's shift (F1), through the guarded reader
    ap.add_argument("--recompute2612", action="store_true")  # --recompute2612: the 2.6.12 reporting tables recomputed from the written decisions (no window read)
    ap.add_argument("--visibility", action="store_true")  # the orbit-based satellite visibility per session and site (no 2.6.12 window read); the figure redrawn
    a = ap.parse_args()  # parsed
    if a.transfer:  # Part A
        transfer()  # run
    elif a.unseen:  # Part B
        unseen()  # run
    elif a.heldout2612:  # --heldout2612
        evaluate_unseen(SEALED, U2612)  # the evaluation through the frozen path (its code unchanged since run1)
        report_2612(U2612)  # the reporting tables from the decisions it recorded
    elif a.diag2612:  # --diag2612
        diagnose_2612(DIAG)  # F1's diagnostic (a further guarded read of 2.6.12's windows, counted)
    elif a.recompute2612:  # --recompute2612
        recompute_2612(U2612)  # from the written decisions only
    elif a.visibility:  # the visibility diagnostic
        visibility(DIAG)  # orbits and site positions only; then the figure
    elif a.dryrun:  # the plumbing check
        evaluate_unseen(a.dryrun, TRANSFER_DIR / "unseen2612_dryrun")  # a training session, its own folder
    elif a.jobs_sample:  # T-F's sample
        global JOBS_B  # the module's job folder, redirected for this run only
        JOBS_B = PROJECT / a.jobs_sample  # the scratch folder
        si = specs_inner_b()  # every inner job
        run_jobs_b([si[i] for i in np.linspace(0, len(si) - 1, 12).astype(int)])  # list comprehension: 12 evenly spaced jobs, run from scratch
    else:  # nothing chosen
        raise SystemExit("usage: 7_evaluation/01_transfer.py --transfer | --unseen | --heldout2612 | --diag2612 | --dryrun S")  # stop


if __name__ == "__main__":  # guard: runs main() only when the file is executed, not when imported (spawned workers import it)
    main()  # runs main
