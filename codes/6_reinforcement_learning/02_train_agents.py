#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
Agent training: the inner leave-one-session-out choice of the training length, the outer training, and the freezing of the agents.

The agents' design rules are named P1 to P11 throughout these scripts:
  P1 every training reward is divided by 100 (every reported cost is unscaled); P2 the surrogate arm's observations pass through
  sign(x) log(1 + |x|); P3 the budget is chosen on the inner folds (snapshots every 10,000 steps, two inner seeds, the plain mean over the
  evaluation episodes, a tie to the shorter budget); P4 deterministic evaluation episodes at fixed spacing; P5 the inner-loop states use
  the inner fit's own transform; P6 the deployable operating point of each detector from inner material only; P7 the operating curves;
  P8 interquartile means with 95 % stratified bootstrap intervals; P9 the verdicts against the best baseline; P10 the comparison with
  three trivial policies (an arm has learned at a lambda when it is below all three in at least 3 of 5 outer folds); P11 each frozen
  agent's implied threshold.

Configuration: Chowdhury's learning starts 500 and train frequency 1; one 100,000-step schedule for every training, outer runs stopped
at the selected budget by a callback; the inner miss penalty m from the three inner training sessions.
  States   the primary arm's inner-loop states are built with the inner fit's own transform (P5, which the adopted-pipeline script's
           nested_keep keeps); the surrogate arm uses the outer fold's state file, whose training-session log-odds are the inner
           out-of-fold scores (P5, disclosed).
  Jobs     one training per (step, receiver, outer fold, inner fold, arm, lambda, seed), each an independent, fully seeded process task:
           Monitor <- reward / 100 (P1) <- signed-log observations (surrogate only, P2) <- the environment script's SpoofingDetectionEnv;
           DQN in SB3 with Chowdhury's configuration and gamma 1. Inner jobs evaluate the greedy policy at every snapshot on the inner
           held-out session's evaluation episodes (P4) and run it through the environment script's stream on that session (P6 reads its
           stopping-mode clean share); outer jobs stop at the selected budget and save their weights.
  Output   one gzip JSON per job, written atomically, with a .sha256 entry; a repeated run skips a job whose output, hash and
           specification digest all match. Wall times go to a separate table (wall clock by nature, declared).
The freeze writes P6's deployable points, P10's verdicts and P11's implied thresholds, with a manifest of the frozen files' digests.
2.6.12 is sealed: the environment script's reader drops it, and no table read here holds it. The outer held-out session never enters a
training or an inner-evaluation episode. It imports the environment script (which loads the adopted-pipeline script) only.

Usage   python3 codes/6_reinforcement_learning/02_train_agents.py --grid   (--pilot times one cell)
Reads   Data/processed/rl_environment/ (states, index, onsets, folds, steps); P rebuilt through the adopted-pipeline script once per run
        (inner transforms, nested predictions)
Writes  Data/processed/rl_agents/training/ (states, jobs, tables) and Data/processed/rl_agents/models/training/ (the model files; their
        SHA-256 in the manifest)
"""
from __future__ import annotations  # future import: type hints kept as text (the project's convention)

import argparse  # standard library: the command line
import ast  # standard library: syntax trees, so a comment edit does not invalidate finished jobs
import gzip  # standard library: gzip with a fixed header time (byte-stable)
import hashlib  # standard library: SHA-256 digests
import importlib.util  # standard library: loads the environment script (its name starts with a digit)
import inspect  # standard library: SB3's default arguments, for the configuration table
import json  # standard library: the job outputs
import multiprocessing  # standard library: the spawn context of the worker pool
import os  # standard library: atomic replacement of a written file
import sys  # standard library: module registry and exit status
import time  # standard library: wall-clock timing (timing table only)
from concurrent.futures import ProcessPoolExecutor, as_completed  # standard library: the pool of worker processes
from pathlib import Path  # standard library: paths

import numpy as np  # numpy: arrays
import pandas as pd  # pandas: tables

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
_spec = importlib.util.spec_from_file_location("environment_for_training", PROJECT / "codes" / "6_reinforcement_learning" / "01_environment.py")  # where 11 is
E = importlib.util.module_from_spec(_spec)  # an empty module object
sys.modules["environment_for_training"] = E  # registered, so its own references resolve
_spec.loader.exec_module(E)  # the environment script's top level (it loads the adopted-pipeline script); its main() is not run

import gymnasium as gym  # gymnasium: the wrapper classes
import stable_baselines3 as sb3  # Stable-Baselines3 (version recorded)
import torch  # PyTorch, SB3's backend
# torch.set_num_threads(1): with the environment script's stack loaded (scikit-learn's OpenMP runtime in the process) multithreaded torch crashes (the environment Part E);
# every worker process imports this file, so every training runs on one thread
torch.set_num_threads(1)
from stable_baselines3 import DQN  # the Deep Q-Network
from stable_baselines3.common.callbacks import BaseCallback  # the hook SB3 calls after every environment step
from stable_baselines3.common.monitor import Monitor  # SB3's episode-return logger (T1 reads it)
from stable_baselines3.common.utils import set_random_seed  # seeds Python's random, numpy and torch at once

AGENT_DIR = E.PROC / "rl_agents"  # this section's folder
SEED = E.SEED  # 20260914
SESSIONS, RXS = E.SESSIONS, E.RXS  # the five folds in the screen script's order; the three receivers
LAMBDAS = E.LAMBDAS  # (10, 30, 100, 300, 1000)
SCHEDULE = 100_000  # every training runs under one 100,000-step schedule (exploration anneals over its first 30 %)
BUDGETS = tuple(range(10_000, SCHEDULE + 1, 10_000))  # tuple of a range with a step: P3's snapshots, every 10,000 steps from 10,000 to 100,000 (the pilot used 25k/50k/100k)
TRIVIAL = ("never", "first_step", "always")  # tuple: P10's trivial policies; "always" (an alarm at every step) added after the pilot
SHARES = E.SHARES  # (0.02, 0.05, 0.10): the registered clean-window shares
GRID = "training"  # the folder of the full grid under Data/processed/rl_agents/
INNER_SEEDS = (20260914, 20260915)  # tuple: P3's two inner seeds
OUTER_SEEDS = (20260914, 20260915, 20260916, 20260917, 20260918)  # tuple: the five outer seeds
REWARD_SCALE = 100.0  # P1: every training reward is divided by 100 (every reported cost is unscaled)
ONSET_EVERY, CLEAN_EVERY = 10, 30  # tuple assignment: P4's spacing of evaluation starts (onset and clean episodes), in s
N_WORKERS = 3  # at most three worker processes
TIE_TOL = 1e-9  # P3: criteria within 1e-9 of each other are a tie, and a tie goes to the shorter budget
# dict: Chowdhury's Table 8 (2025, p. 105) with gamma 1 and his learning starts and train frequency;
# every other DQN argument stays at SB3's default and is recorded in config.csv
DQN_KW = {"learning_rate": 1e-3, "gamma": E.GAMMA_TRAIN, "buffer_size": 10_000, "learning_starts": 500, "batch_size": 64, "train_freq": 1,
          "target_update_interval": 500, "exploration_fraction": 0.3, "exploration_initial_eps": 1.0, "exploration_final_eps": 0.01}  # dict literal continued
PILOT = {"receiver": "SBF_02", "outer": "2.6.2", "lambda": 100.0}  # dict: the pilot cell


# ----------------------------------------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------------------------------------
def sha_file(p: Path) -> str:  # function: a file's SHA-256
    return hashlib.sha256(p.read_bytes()).hexdigest()  # the digest of its bytes


def sha_code(p: Path) -> str:  # function: SHA-256 of a script's syntax tree, so comments and blank lines do not change it; for a script of this folder, the digest of the same script as the analysis ran it
    own = hashlib.sha256(ast.dump(ast.parse(p.read_text(encoding="utf-8")), include_attributes=False).encode()).hexdigest()  # the digest of the tree's text
    from helpers import analysis_code_digest  # the table of the analysis's code digests (codes/common/helpers.py)
    return analysis_code_digest(PROJECT, p, own)  # the analysis's digest when the table lists this script with exactly this tree, else the tree's own


def write_atomic(path: Path, data: bytes) -> None:  # function: writes a file so that it either exists complete or not at all
    path.parent.mkdir(parents=True, exist_ok=True)  # the folder
    tmp = path.with_name(path.name + ".tmp")  # a temporary name in the same folder
    tmp.write_bytes(data)  # the bytes
    os.replace(tmp, path)  # os.replace: an atomic rename on the same disk


def gz_json(obj) -> bytes:  # function: a JSON document gzipped with no name or time in its header, so runs again are byte-identical
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")  # sorted keys, no spaces: a fixed text
    return gzip.compress(raw, compresslevel=9, mtime=0)  # mtime=0: no time stamp in the header


def read_gz_json(path: Path):  # function: reads a job output back
    return json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))  # decompressed and parsed


def signed_log(x: np.ndarray) -> np.ndarray:  # function: P2's compression sign(x) * log(1 + |x|), P's own (the adopted-pipeline script's transform), no fitted parameter
    return np.sign(x) * np.log1p(np.abs(x))  # elementwise


# ----------------------------------------------------------------------------------------------------------------------
# inner-loop states (P5): the primary arm's states under the inner fit's own transform
# ----------------------------------------------------------------------------------------------------------------------
def inner_states(B: dict, cells: list, out: Path) -> pd.DataFrame:  # function: writes one .npz per (receiver, outer, inner) and returns the checks and fingerprints
    npr = B["nested"]["predictions"]  # the inner out-of-fold predictions with decision values
    rows = []  # one row per inner fit
    for rx, outer in cells:  # for loop over the requested (receiver, outer fold) cells
        idx, X = B["cells"][rx]  # the receiver's windows (every session) and their orbit-free values
        wr = E.window_rows(idx)  # session -> window lookup (the environment script's)
        ix = pd.read_csv(E.ENV_DIR / "state_index.csv.gz", dtype={"session": str, "fold": str})  # the environment script's state index
        ix = ix[(ix.receiver == rx) & (ix.fold == outer) & (ix.role == "train")].sort_values("row")  # the outer fold's training rows, in matrix order
        for inner in [s for s in SESSIONS if s != outer]:  # for loop over the four inner folds
            f = [r for r in B["nested"]["fits"] if r["receiver"] == rx and r["outer_held_out"] == outer and r["inner_held_out"] == inner]  # list comprehension: the inner fit's record
            if len(f) != 1:  # if: not exactly one record (cannot)
                raise SystemExit(f"STOP: {len(f)} inner fits for {rx} {outer} {inner}")  # stop
            f = f[0]  # the record
            rec = f["transform"]  # the adopted-pipeline script's kept transform: prep, cand, sel
            Z = E.A.transform(rec, X, B["wc10"])  # every window of the receiver -> the inner fit's 36 columns (C order)
            P = []  # the primary rows
            for s in SESSIONS:  # for loop over the sessions in the environment script's order (the index is ordered the same way)
                d = ix[ix.session == s]  # its training rows
                if len(d):  # if: a training session of the outer fold
                    sec = d.sec.to_numpy(np.int64)  # its valid seconds in time order
                    P.append(np.hstack([Z[E.rows_at(wr, s, sec - lag)] for lag in E.LAGS]))  # list comprehension: windows ending t, t-10, t-20
            Pm = np.vstack(P)  # the matrix, one row per outer training row
            if len(Pm) != len(ix) or not np.isfinite(Pm).all():  # if: the rows do not cover the index or a value is missing
                raise SystemExit(f"STOP: inner states of {rx} {outer} {inner} incomplete")  # stop
            q = npr[(npr.model == "lr") & (npr.receiver == rx) & (npr.outer_held_out == outer) & (npr.held_out == inner)].sort_values("sec")  # the committed inner decision values of the inner held-out session
            te = E.rows_at(wr, inner, q.sec.to_numpy(np.int64))  # their windows' rows
            dec = E.A.decision(f["model_lr"], np.ascontiguousarray(Z[te]))  # the inner model's decision values on the kept transform (C order, as every score in the environment)
            fit_sessions = [s for s in SESSIONS if s not in (outer, inner)]  # the three sessions the inner fit may use
            n_scored = int((idx.test_code.isin(fit_sessions) & idx.tlabel.isin(E.A.SCORED)).sum())  # their scored windows under the training label
            name = f"inner_states_{rx}_{outer}_{inner}.npz"  # the file name
            E.write_npz({"primary": Pm, "row": ix.row.to_numpy(np.int64), "sec": ix.sec.to_numpy(np.int64)}, out / name)  # written with fixed member dates (the environment script's writer)
            prep = rec["prep"]  # the fitted preparation
            h = hashlib.sha256(np.ascontiguousarray(prep["med"]).tobytes() + np.ascontiguousarray(prep["scaler"][0]).tobytes() + np.ascontiguousarray(prep["scaler"][1]).tobytes() + f["selected_columns"].encode()).hexdigest()  # SHA-256 of medians, scaler and selection (the environment script's fingerprint rule)
            # dict literal over two lines: the checks of this inner fit
            rows.append({"receiver": rx, "outer": outer, "inner": inner, "file": name, "rows": len(Pm), "screen_applied": f["screen_applied"], "fit_sessions": ";".join(fit_sessions), "train_rows": f["train_rows"], "scored_windows_of_fit_sessions": n_scored,
                         "train_rows_match": f["train_rows"] == n_scored, "dec_windows": len(q), "dec_bitwise": bool(np.array_equal(dec, q.dec.to_numpy())), "transform_sha256": h, "file_sha256": sha_file(out / name)})  # dict literal continued
    return pd.DataFrame(rows)  # the table


# ----------------------------------------------------------------------------------------------------------------------
# materials and evaluation episodes
# ----------------------------------------------------------------------------------------------------------------------
def job_material(spec: dict, states: Path) -> tuple:  # function: (training material, evaluation material or None) of one job, read from the written files
    rx, outer, inner, arm = spec["receiver"], spec["outer"], spec["inner"], spec["arm"]  # tuple assignment: the job's cell
    mat = E.material(rx, outer)  # the environment script's material of the outer fold (training rows only in its key)
    if inner == "":  # an outer job: all four training sessions
        return mat, None  # the outer material as 11 builds it
    ix = mat["index"]  # the state index, position = matrix row
    if arm == "primary":  # the primary arm: the inner fit's own states (P5)
        z = E.load_npz(states / f"inner_states_{rx}_{outer}_{inner}.npz")  # the inner states
        Xp = np.full(mat["primary"].shape, np.nan)  # NaN everywhere: the outer held-out rows stay unfilled and can never be read
        Xp[z["row"]] = z["primary"]  # the outer training rows under the inner transform
        mat = {**mat, "primary": Xp}  # dict unpacking: the material with the inner matrix
    itrain = [s for s in SESSIONS if s not in (outer, inner)]  # the three inner training sessions
    tr = ix[(ix.role == "train") & ix.session.isin(itrain)]  # their rows
    ev = ix[(ix.role == "train") & (ix.session == inner)]  # the inner held-out session's rows
    on = mat["onsets"]  # usable onsets of the four outer training sessions
    on_tr = on[on.session.isin(itrain)].reset_index(drop=True)  # those of the inner training sessions
    m = float(np.median(on_tr.length_s))  # the inner miss penalty from the three inner training sessions
    key = lambda d: {(s, int(t)): int(r) for s, t, r in zip(d.session, d.sec, d.row)}  # lambda with a dict comprehension: (session, second) -> row
    # dict literals over two lines: the training material (inner training sessions only) and the evaluation material (inner held-out session only)
    train = {**mat, "key": key(tr), "onsets": on_tr, "m": m, "pre_rows": tr.row[tr.step == "pre"].to_numpy()}  # dict unpacking with replacements
    evm = {**mat, "key": key(ev), "onsets": on[on.session == inner].reset_index(drop=True), "m": m, "pre_rows": ev.row[ev.step == "pre"].to_numpy()}  # the same for the evaluation session
    return train, evm  # both


def eval_episodes(env: E.SpoofingDetectionEnv) -> list:  # function: P4's deterministic evaluation episodes of an evaluation material
    mat, eps = env.mat, []  # the material; the episodes
    for o in mat["onsets"].itertuples():  # for loop over the session's usable onsets
        for st in range(int(o.start_lo), int(o.start_hi) + 1, ONSET_EVERY):  # range with a step: every 10 s from max(first usable second, onset - 300) to onset - 10
            eps.append(env.episode("onset", o.session, st, int(o.onset)))  # the environment script's episode definition
    pr = np.sort(mat["pre_rows"])  # the session's valid pre-change rows (rows are in time order within a session)
    sec, run = mat["sec"][pr], mat["run"][pr]  # their seconds and window runs
    new = np.r_[True, (np.diff(sec) != 1) | (np.diff(run) != 0)]  # np.r_ concatenation: True where a run of consecutive pre-change seconds begins
    first = np.maximum.accumulate(np.where(new, np.arange(len(pr)), 0))  # running maximum: the position of each second's run start
    for j in np.flatnonzero((sec - sec[first]) % CLEAN_EVERY == 0):  # for loop over every 30th second of each run, counted from its first second
        eps.append(env.episode("clean", str(mat["session"][pr[j]]), int(sec[j])))  # the environment script's clean episode from there
    return eps  # the list


def episode_obs(env: E.SpoofingDetectionEnv, ep: dict) -> np.ndarray:  # function: the observations of one episode along its never-alarm path (the agent's own inputs)
    x = env.X[ep["rows"]]  # the arm's state rows
    if env.arm != "surrogate":  # the primary arm: the rows themselves
        return x  # the matrix
    S, s_ = np.zeros(len(x)), 0.0  # S_t per step; the running value (S = 0 before the first step)
    for i in range(len(x)):  # for loop over the steps
        s_ = max(0.0, s_ + float(x[i, -1]))  # Page's recursion on z_t, the last of the 30 log-odds (the environment script's S_next)
        S[i] = s_  # kept
    return signed_log(np.column_stack([x, S]))  # P2: the 31 values compressed


def episode_cost(types: np.ndarray, actions: np.ndarray, kind: str, lam: float, m: float) -> tuple:  # function: closed-form undiscounted cost of an action sequence along an episode; (cost, outcome, delay)
    stop = np.flatnonzero((actions == 1) & (types != E.STEP["uncertain"]))  # alarms in pre- or post-change seconds end the episode; uncertain ones are masked
    j = int(stop[0]) if stop.size else len(types)  # the terminating position, or past the end
    post_before = int((types[:j] == E.STEP["post"]).sum())  # silent post-change seconds before it: one cost unit each
    if stop.size and types[j] == E.STEP["pre"]:  # a false alarm
        return post_before + lam, "false_alarm", np.nan  # lambda on top
    if stop.size:  # a detection
        return float(post_before), "detection", float(post_before)  # the delay is the silent post-change seconds
    if kind == "onset":  # a miss: the interval ended in silence
        return post_before + m, "miss", np.nan  # m on top
    return 0.0, "clean_end", np.nan  # a clean episode that ended without an alarm


def evaluate(model, env: E.SpoofingDetectionEnv, eps: list, lam: float) -> tuple:  # function: the greedy policy on the evaluation episodes: (summary, per-episode costs, action digest)
    obs = [episode_obs(env, ep) for ep in eps]  # list comprehension: every episode's observations
    acts = model.predict(np.vstack(obs), deterministic=True)[0] if model is not None else None  # the greedy actions of all of them at once (SB3's own greedy path)
    out, k = [], 0  # per-episode results; position in the stacked actions
    for ep, o in zip(eps, obs):  # for loop over the episodes
        a = acts[k:k + len(o)] if acts is not None else None  # this episode's actions
        k += len(o)  # next slice
        out.append((ep, a))  # kept
    return out  # (episode, actions) pairs


def summarise(pairs: list, env: E.SpoofingDetectionEnv, lam: float, policy: str = "agent") -> dict:  # function: the criterion and diagnostics of a policy over the evaluation episodes
    rows = []  # one row per episode
    for ep, a in pairs:  # for loop over the episodes
        ty = env.mat["type"][ep["rows"]]  # step types
        if policy == "never":  # P10's first trivial policy
            a = np.zeros(len(ty), int)  # never alarm
        elif policy == "first_step":  # P10's second trivial policy (the environment tests': an alarm at position 0 only, masked if that second is uncertain)
            a = np.zeros(len(ty), int)  # silent ...
            a[0] = 1  # ... except the first step
        elif policy == "always":  # P10's third trivial policy: an alarm at every step (stops at the first pre- or post-change second)
            a = np.ones(len(ty), int)  # alarm everywhere
        c, outcome, delay = episode_cost(ty, np.asarray(a), ep["kind"], lam, env.m)  # closed-form cost
        rows.append({"kind": ep["kind"], "cost": c, "outcome": outcome, "delay": delay})  # kept
    d = pd.DataFrame(rows)  # the table
    on, cl = d[d.kind == "onset"], d[d.kind == "clean"]  # the two kinds
    # dict literal over three lines: the criterion (P3: plain mean over all evaluation episodes) and the diagnostics
    return {"criterion": float(d.cost.mean()), "episodes": len(d), "onset_episodes": len(on), "clean_episodes": len(cl), "onset_cost": float(on.cost.mean()) if len(on) else None, "clean_cost": float(cl.cost.mean()) if len(cl) else None,
            "onset_detect_share": float((on.outcome == "detection").mean()) if len(on) else None, "onset_fa_share": float((on.outcome == "false_alarm").mean()) if len(on) else None, "onset_miss_share": float((on.outcome == "miss").mean()) if len(on) else None,  # dict literal continued: the outcome shares
            "clean_fa_share": float((cl.outcome == "false_alarm").mean()) if len(cl) else None, "median_delay": float(on.delay.median()) if (on.outcome == "detection").any() else None}  # dict literal continued


# ----------------------------------------------------------------------------------------------------------------------
# the agent through the environment script's evaluation stream (the inner held-out session; P6 needs its clean-window share)
# ----------------------------------------------------------------------------------------------------------------------
def read_steps() -> pd.DataFrame:  # function: the environment script's step table (every grid second: label, window, validity, run)
    return pd.read_csv(E.ENV_DIR / "steps.csv.gz", dtype={"session": str})  # read


def session_seconds(steps: pd.DataFrame, rx: str, session: str) -> pd.DataFrame:  # function: the windowed seconds of one session, in time order (the stream's grid, as the baselines')
    return steps[(steps.receiver == rx) & (steps.session == session) & steps.window].sort_values("sec").reset_index(drop=True)  # filtered and ordered


def agent_decisions(model, env: E.SpoofingDetectionEnv, d: pd.DataFrame) -> tuple:  # function: the greedy agent's decisions on every windowed second of a session, (decision mode, stopping mode)
    sec = d.sec.to_numpy(np.int64)  # the seconds
    rows = np.array([env.mat["key"].get((s, int(t)), -1) for s, t in zip(d.session, d.sec)])  # list comprehension: each second's state row, -1 where no state exists (silent there)
    ok = rows >= 0  # seconds with a state
    run = np.where(ok, env.mat["run"][np.maximum(rows, 0)], -1)  # np.where: the window run of each second with a state
    new = ok & np.r_[True, (np.diff(sec) != 1) | (np.diff(run) != 0) | ~ok[:-1]]  # True where a contiguous stretch of states begins (S restarts there)
    seg = np.cumsum(new)  # running count: a stretch number per second
    end = np.zeros(len(sec), int)  # each second's stretch end (exclusive)
    for g in np.unique(seg[ok]):  # for loop over the stretches
        ix = np.flatnonzero(ok & (seg == g))  # its positions
        end[ix] = ix[-1] + 1  # its end
    predict = lambda o: model.predict(o, deterministic=True)[0]  # lambda: SB3's greedy actions of a batch of observations

    def obs(i: int, j: int) -> np.ndarray:  # nested function: the observations of positions i .. j-1 of one stretch, S restarting at i
        x = env.X[rows[i:j]]  # the state rows
        if env.arm != "surrogate":  # the primary arm has no memory
            return x  # the rows
        S, s_ = np.zeros(j - i), 0.0  # S per second; the running value from 0
        for k in range(j - i):  # for loop over the seconds
            s_ = max(0.0, s_ + float(x[k, -1]))  # Page's recursion on z_t
            S[k] = s_  # kept
        return signed_log(np.column_stack([x, S]))  # P2's compression of the 31 values

    dec = np.zeros(len(sec), int)  # decision mode: the action at every second, no reset (S restarts only at a gap, as the environment script's cusum)
    for i in np.flatnonzero(new):  # for loop over the stretch starts
        dec[i:end[i]] = predict(obs(i, end[i]))  # the greedy actions of the stretch
    if env.arm != "surrogate":  # the primary arm: stopping mode is the decision-mode actions through the environment script's refractory rule
        return dec, E.events(sec, dec).astype(int)  # both modes
    stop, i, quiet_to = np.zeros(len(sec), int), 0, -np.inf  # stopping mode: events; position; end of the refractory period
    while i < len(sec):  # while loop over the seconds
        if not ok[i] or sec[i] <= quiet_to:  # no state, or inside the refractory period: silent
            i += 1  # next second
            continue  # on
        a = predict(obs(i, end[i]))  # S restarts here (an alarm's reset, or a stretch start), so the stretch's actions from i
        hit = np.flatnonzero(a == 1)  # the alarms
        if not hit.size:  # no alarm in the rest of the stretch
            i = end[i]  # the next stretch
            continue  # on
        k = i + int(hit[0])  # the first alarm
        stop[k], quiet_to, i = 1, sec[k] + E.REFRACTORY, k + 1  # tuple assignment: an event, 30 s of silence, then a restart
    return dec, stop  # both modes


def stream_row(d: pd.DataFrame, decisions: np.ndarray, session: str, mode: str) -> dict:  # function: the environment script's stream figures that the deployable points and the curves need
    f = E.stream(d.sec.to_numpy(np.int64), d.label.to_numpy(object), decisions, session, mode)  # the environment script's stream on the session (decisions are already events in stopping mode, so the refractory rule leaves them unchanged)
    keep = ("seconds", "clean", "clean_in_alarm", "alarms", "alarms_uncertain", "attack", "recall", "intervals", "detected", "delay_median_s", "pre_onset_uncertain", "early_warning")  # tuple: the figures kept
    return {f"{mode}_{k}": (None if isinstance(f[k], float) and np.isnan(f[k]) else f[k]) for k in keep}  # dict comprehension: prefixed by the mode, NaN as None (JSON)


# ----------------------------------------------------------------------------------------------------------------------
# the training wrappers
# ----------------------------------------------------------------------------------------------------------------------
class SignedLog(gym.ObservationWrapper):  # class: P2, the surrogate arm's observations through sign(x) * log(1 + |x|)
    def observation(self, observation):  # method: gymnasium calls it on every observation (reset and step, the final one included)
        return signed_log(observation)  # compressed


class ScaleReward(gym.RewardWrapper):  # class: P1, every training reward divided by 100
    def reward(self, reward):  # method: gymnasium calls it on every reward
        return reward / REWARD_SCALE  # scaled


class EpisodeLog(gym.Wrapper):  # class: records every episode's specification and actions, for T1's closed form
    def __init__(self, env):  # method: set-up
        super().__init__(env)  # gym.Wrapper's own set-up
        self.done_eps, self.cur = [], None  # tuple assignment: finished episodes; the current one

    def reset(self, **kw):  # method: a new episode starts
        out = self.env.reset(**kw)  # the environment's reset
        ep = self.env.unwrapped.ep  # the environment script's episode record
        self.cur = {"kind": ep["kind"], "session": ep["session"], "start": int(ep["start"]), "onset": int(ep["onset"]), "actions": []}  # dict literal: its specification
        return out  # unchanged

    def step(self, action):  # method: one step
        o, r, term, trunc, info = self.env.step(action)  # the environment's step
        self.cur["actions"].append(int(action))  # the action taken
        if term or trunc:  # the episode ended
            a = np.array(self.cur["actions"], np.uint8)  # its actions
            self.done_eps.append({**{k: v for k, v in self.cur.items() if k != "actions"}, "steps": len(a), "terminated": bool(term), "actions_hex": np.packbits(a).tobytes().hex()})  # dict comprehension: the record, actions packed into bits
        return o, r, term, trunc, info  # unchanged


def make_env(mat: dict, arm: str, lam: float) -> Monitor:  # function: the training environment stack
    env = E.SpoofingDetectionEnv(mat, arm, lam)  # the environment script's environment
    if arm == "surrogate":  # P2 applies to the surrogate arm only (the primary state is already compressed by P)
        env = SignedLog(env)  # compressed observations
    return Monitor(EpisodeLog(ScaleReward(env)))  # Monitor outermost, so it logs the scaled returns the agent learns from


class Snapshots(BaseCallback):  # class: evaluates at the inner snapshots, or stops an outer run at its budget
    def __init__(self, marks: tuple, on_mark, stop_at: int | None):  # method: the step counts to act at, what to do there, and where to stop
        super().__init__()  # BaseCallback's set-up
        self.marks, self.on_mark, self.stop_at = set(marks), on_mark, stop_at  # tuple assignment: kept

    def _on_step(self) -> bool:  # method: SB3 calls it after every environment step, before that step's training update
        if self.num_timesteps in self.marks:  # a snapshot
            self.on_mark(self.model, self.num_timesteps)  # evaluated (or saved) with the weights as they are now
        return self.stop_at is None or self.num_timesteps < self.stop_at  # False stops training: an outer run ends at its budget


# ----------------------------------------------------------------------------------------------------------------------
# jobs
# ----------------------------------------------------------------------------------------------------------------------
def job_id(s: dict) -> str:  # function: the job's file name stem
    return f"{s['stage']}_{s['receiver']}_{s['outer']}_{s['inner'] or 'all'}_{s['arm']}_lam{int(s['lambda'])}_s{s['seed']}_b{s['budget']}"  # every field of the specification


def spec_digest(s: dict, states: Path) -> str:  # function: SHA-256 of everything a job's result depends on: its specification, the configuration, the data files and the code
    files = [E.ENV_DIR / f"states_{s['receiver']}_{s['outer']}.npz", E.ENV_DIR / "state_index.csv.gz", E.ENV_DIR / "onsets.csv", E.ENV_DIR / "folds.csv"]  # list: the environment script's files the material reads
    if s["inner"] and s["arm"] == "primary":  # an inner primary job also reads its inner states
        files.append(states / f"inner_states_{s['receiver']}_{s['outer']}_{s['inner']}.npz")  # that file
    # dict literal over two lines: the digest's input
    d = {"spec": s, "dqn": DQN_KW, "schedule": SCHEDULE, "budgets": BUDGETS, "scale": REWARD_SCALE, "eval": [ONSET_EVERY, CLEAN_EVERY], "sb3": sb3.__version__, "torch": torch.__version__,
         "files": [sha_file(f) for f in files], "code": [sha_code(PROJECT / "codes" / n) for n in ("6_reinforcement_learning/01_environment.py", "6_reinforcement_learning/02_train_agents.py")]}  # list comprehensions: file digests and code digests
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()  # the digest


def run_job(spec: dict, states: str, jobs: str, models: str) -> dict:  # function: one training (runs in a worker process); returns its timing
    states, jobs, models = Path(states), Path(jobs), Path(models)  # tuple assignment: paths (passed as text across the process boundary)
    jid = job_id(spec)  # its name
    dig = spec_digest(spec, states)  # its digest
    out, hfile = jobs / f"{jid}.json.gz", jobs / f"{jid}.sha256"  # tuple assignment: the output and its hash entry
    if out.exists() and hfile.exists() and hfile.read_text().split() == [sha_file(out), dig]:  # if: finished before with the same inputs
        return {"job": jid, "skipped": True}  # nothing to do
    set_random_seed(spec["seed"])  # Python's random, numpy's global generator and torch seeded before anything is built
    train, evm = job_material(spec, states)  # the materials
    env = make_env(train, spec["arm"], spec["lambda"])  # the training stack
    model = DQN("MlpPolicy", env, seed=spec["seed"], device="cpu", verbose=0, **DQN_KW)  # the agent (seeds SB3, the env's first reset and the action space)
    res = {"spec": spec, "digest": dig, "checkpoints": []}  # the job's result
    if evm is not None:  # an inner job: evaluation episodes on the inner held-out session
        eenv = E.SpoofingDetectionEnv(evm, spec["arm"], spec["lambda"])  # the evaluation environment (never stepped; its episode() builds the episodes)
        eps = eval_episodes(eenv)  # P4's episodes
        res["trivial"] = {p: summarise(evaluate(None, eenv, eps, spec["lambda"]), eenv, spec["lambda"], p) for p in TRIVIAL}  # dict comprehension: P10's three trivial policies
        res["m"] = evm["m"]  # the inner miss penalty
        sd = session_seconds(read_steps(), spec["receiver"], spec["inner"])  # the inner held-out session's windowed seconds (the stream)

        def on_mark(mdl, n):  # nested function: one snapshot's evaluation
            pairs = evaluate(mdl, eenv, eps, spec["lambda"])  # greedy actions
            acts = np.concatenate([a for _, a in pairs]).astype(np.uint8)  # all of them in order
            dd, ds = agent_decisions(mdl, eenv, sd)  # the agent on the inner held-out stream, decision and stopping mode
            st = {**stream_row(sd, dd, spec["inner"], "decision"), **stream_row(sd, ds, spec["inner"], "stopping")}  # dict unpacking: both modes' figures (P6 reads the stopping-mode clean share)
            # dict literal over two lines: the snapshot's record
            res["checkpoints"].append({"budget": n, **summarise(pairs, eenv, spec["lambda"]), **st, "actions_sha256": hashlib.sha256(acts.tobytes()).hexdigest(),
                                       "stream_sha256": hashlib.sha256(np.concatenate([dd, ds]).astype(np.uint8).tobytes()).hexdigest(), "train_episodes": len(env.get_episode_rewards())})  # dict unpacking: the summary
        cb = Snapshots(BUDGETS, on_mark, None)  # every snapshot, no early stop
    else:  # an outer job: stop at the budget and save the weights
        res["m"] = train["m"]  # the outer miss penalty

        def on_mark(mdl, n):  # nested function: freezes the weights at the budget
            sd = {**{f"q_net.{k}": v.detach().cpu().numpy() for k, v in mdl.q_net.state_dict().items()}, **{f"q_net_target.{k}": v.detach().cpu().numpy() for k, v in mdl.q_net_target.state_dict().items()}}  # dict comprehensions: both networks' weights as arrays
            path = models / f"{jid}.npz"  # the model file
            E.write_npz(sd, path)  # fixed member dates, so the file is byte-stable
            res["checkpoints"].append({"budget": n, "model_file": path.name, "model_sha256": sha_file(path), "train_episodes": len(env.get_episode_rewards())})  # recorded
        cb = Snapshots((spec["budget"],), on_mark, spec["budget"])  # one mark, and the stop there
    t0 = time.perf_counter()  # clock
    model.learn(total_timesteps=SCHEDULE, callback=cb)  # the 100,000-step schedule (an outer run stops at its budget)
    dt = time.perf_counter() - t0  # seconds
    log = env.get_wrapper_attr("done_eps")  # EpisodeLog's finished episodes
    ret = env.get_episode_rewards()  # Monitor's returns of the same episodes, in the same order
    if len(log) != len(ret):  # if: the two logs disagree (cannot)
        raise SystemExit(f"STOP: {jid}: {len(log)} logged episodes, {len(ret)} Monitor returns")  # stop
    res["episodes"] = [{**e, "monitor_return": float(r)} for e, r in zip(log, ret)]  # list comprehension: each episode with its Monitor return
    write_atomic(out, gz_json(res))  # the output, atomically
    write_atomic(hfile, f"{sha_file(out)} {dig}\n".encode())  # then its hash entry
    return {"job": jid, "skipped": False, "seconds": round(dt, 1), "steps": int(model.num_timesteps), "steps_per_s": round(model.num_timesteps / dt, 1), "episodes": len(ret)}  # the timing


def run_jobs(specs: list, states: Path, jobs: Path, models: Path) -> pd.DataFrame:  # function: every job through a pool of three spawned worker processes; returns the timing table
    ctx = multiprocessing.get_context("spawn")  # spawn: each worker starts a fresh interpreter (no state inherited from the parent)
    rows = []  # timings
    with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=ctx) as ex:  # with-block: the pool, shut down at the end
        futs = {ex.submit(run_job, s, str(states), str(jobs), str(models)): s for s in specs}  # dict comprehension: future -> its specification
        for f in as_completed(futs):  # for loop over the jobs as they finish
            r = f.result()  # its timing (an exception in the worker is raised here)
            rows.append({**r, "arm": futs[f]["arm"], "stage": futs[f]["stage"]})  # kept
            print(f"  {r['job']}: {'skipped' if r['skipped'] else str(r['seconds']) + ' s'}", flush=True)  # progress
    return pd.DataFrame(rows).sort_values("job").reset_index(drop=True)  # in a fixed order


# ----------------------------------------------------------------------------------------------------------------------
# selection (P3) and the learning-failure comparison (P10)
# ----------------------------------------------------------------------------------------------------------------------
def inner_curve(jobs: Path, specs: list) -> pd.DataFrame:  # function: every snapshot of every inner job, with the trivial policies
    rows = []  # one row per job and snapshot
    for s in specs:  # for loop over the inner jobs
        r = read_gz_json(jobs / f"{job_id(s)}.json.gz")  # its output
        triv = {f"{p}_criterion": r["trivial"][p]["criterion"] for p in TRIVIAL}  # dict comprehension: the three trivial policies' criteria on this inner fold
        for c in r["checkpoints"]:  # for loop over its snapshots
            rows.append({"receiver": s["receiver"], "outer": s["outer"], "inner": s["inner"], "arm": s["arm"], "lambda": s["lambda"], "seed": s["seed"], "m": r["m"], **c, **triv})  # dict unpacking: the snapshot's fields
    return pd.DataFrame(rows)  # the table


def select(curve: pd.DataFrame) -> pd.DataFrame:  # function: P3's budget per (receiver, outer, arm, lambda) and P10's comparison with the three trivial policies
    rows = []  # one row per cell
    for (rx, outer, arm, lam), d in curve.groupby(["receiver", "outer", "arm", "lambda"], sort=True):  # for loop over the cells
        if (d.inner == outer).any() or set(d.groupby("budget").size()) != {len(INNER_SEEDS) * (len(SESSIONS) - 1)} or set(d.budget) != set(BUDGETS):  # if: a held-out row, or a snapshot lacking a fold or a seed
            raise SystemExit(f"STOP: the inner loop of {rx} {outer} {arm} {lam} is incomplete or holds its outer held-out session")  # stop
        mean = d.groupby("budget").criterion.mean()  # the criterion averaged over the four inner folds and two inner seeds
        best = mean.min()  # the lowest mean
        b = int(min(x for x in mean.index if mean[x] <= best + TIE_TOL))  # generator in min(): the shortest budget within the tie tolerance
        triv = d.drop_duplicates(["inner"])[[f"{p}_criterion" for p in TRIVIAL]].mean()  # list comprehension: the trivial policies' criteria averaged over the inner folds (they do not depend on the seed)
        at_b = d[d.budget == b]  # the inner runs at the selected budget
        clean = at_b.stopping_clean.fillna(0).sum()  # pooled inner held-out clean seconds (four folds x two seeds)
        # dict literal over three lines: the row
        rows.append({"receiver": rx, "outer": outer, "arm": arm, "lambda": lam, **{f"criterion_{int(x)}": float(mean[x]) for x in mean.index}, "selected_budget": b, "selected_criterion": float(mean[b]),
                     **{f"{p}_criterion": float(triv[f"{p}_criterion"]) for p in TRIVIAL}, "best_trivial": float(triv.min()), "below_all_trivial": bool(float(mean[b]) < float(triv.min())),  # dict literal continued: the trivial policies and the stream shares
                     "inner_clean_s": int(clean), "inner_stopping_clean_share": float(at_b.stopping_clean_in_alarm.fillna(0).sum() / clean) if clean else None, "inner_decision_clean_share": float(at_b.decision_clean_in_alarm.fillna(0).sum() / clean) if clean else None})  # dict comprehensions: the curve and the trivial policies
    return pd.DataFrame(rows)  # the table


def learned(sel: pd.DataFrame) -> pd.DataFrame:  # function: P10 as amended: an arm has learned at a lambda when it is below all three trivial policies in at least 3 of 5 outer folds
    g = sel.groupby(["receiver", "arm", "lambda"], sort=True).agg(folds=("outer", "size"), folds_below_all=("below_all_trivial", "sum")).reset_index()  # per receiver, arm and lambda
    return g.assign(learned=g.folds_below_all >= 3)  # the verdict


def config_table() -> pd.DataFrame:  # function: every DQN argument as used (Chowdhury's and SB3's defaults) and the versions (config.csv)
    sig = inspect.signature(DQN.__init__).parameters  # SB3's arguments and their defaults
    rows = [{"item": k, "value": repr(DQN_KW[k]) if k in DQN_KW else repr(p.default), "source": "Chowdhury (2025, p. 105, Table 8), with gamma 1" if k in DQN_KW else "SB3 default"} for k, p in sig.items() if k not in ("self", "policy", "env", "verbose", "seed", "device", "tensorboard_log", "_init_setup_model")]  # list comprehension: one row per argument
    pol = DQN("MlpPolicy", E.SpoofingDetectionEnv(E.material("SBF_02", "2.6.2"), "surrogate", 100.0), device="cpu", **DQN_KW)  # a model built only to read the policy's defaults (never trained)
    # list of rows over three lines: the defaults that live inside the policy and the buffer, and the run settings
    rows += [{"item": "net_arch", "value": repr(pol.policy.net_arch), "source": "SB3 default"}, {"item": "activation_fn", "value": pol.policy.activation_fn.__name__, "source": "SB3 default"}, {"item": "optimizer", "value": type(pol.policy.optimizer).__name__, "source": "SB3 default"},
             {"item": "handle_timeout_termination", "value": repr(pol.replay_buffer.handle_timeout_termination), "source": "SB3 default (kept on: a truncated clean episode is bootstrapped)"}, {"item": "schedule_steps", "value": repr(SCHEDULE), "source": "the decision"}, {"item": "budgets", "value": repr(BUDGETS), "source": "P3"},  # list literal continued: buffer and run settings
             {"item": "reward_scale", "value": repr(REWARD_SCALE), "source": "P1"}, {"item": "surrogate_compression", "value": "sign(x)*log1p(|x|)", "source": "P2"}, {"item": "stable-baselines3", "value": sb3.__version__, "source": "version"}, {"item": "torch", "value": torch.__version__, "source": "version"}, {"item": "torch_threads", "value": str(torch.get_num_threads()), "source": "the environment Part E"}]  # list literal continued
    return pd.DataFrame(rows)  # the table


# ----------------------------------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------------------------------
def specs_inner() -> list:  # function: every inner job of the grid (3 receivers x 5 outer folds x 4 inner folds x 2 arms x 5 lambda x 2 seeds)
    # list comprehension over six loops: one specification per inner training, run to the full schedule
    return [{"stage": "inner", "receiver": rx, "outer": outer, "inner": h, "arm": arm, "lambda": float(lam), "seed": s, "budget": SCHEDULE}
            for rx in RXS for outer in SESSIONS for h in SESSIONS if h != outer for arm in ("primary", "surrogate") for lam in LAMBDAS for s in INNER_SEEDS]  # list comprehension continued: the loops


def specs_outer(sel: pd.DataFrame) -> list:  # function: every outer job at its cell's selected budget (3 x 5 x 2 arms x 5 lambda x 5 seeds)
    return [{"stage": "outer", "receiver": r.receiver, "outer": r.outer, "inner": "", "arm": r.arm, "lambda": float(r["lambda"]), "seed": s, "budget": int(r.selected_budget)} for _, r in sel.iterrows() for s in OUTER_SEEDS]  # list comprehension: five seeds per cell


# ----------------------------------------------------------------------------------------------------------------------
# freezing: deployable operating points (P6), implied thresholds (P11)
# ----------------------------------------------------------------------------------------------------------------------
def lowest_threshold(v: np.ndarray, n: int, s: float) -> float:  # function: choices_lib.k5_threshold's rule generalised to the share s and to silent seconds: the lowest t with #(v >= t) <= floor(s n)
    v = np.sort(v[np.isfinite(v)])[::-1]  # the values, descending
    k = int(np.floor(s * n))  # at most k of the n clean seconds may be in alarm
    if not len(v):  # no value at all
        return np.nan  # undefined
    if k >= len(v):  # every value may alarm
        return float(v[-1])  # the smallest value
    return float(np.nextafter(v[k], np.inf))  # just above the (k+1)-th largest value: at most k values reach it


def surrogate_runs(mat: dict, session: str) -> tuple:  # function: one training session's valid rows in time order and its decision-mode S_t (restarting at every gap, as the environment script's cusum)
    ix = mat["index"]  # the state index
    d = ix[(ix.session == session)].sort_values("sec")  # the session's valid rows
    rows, sec, run = d.row.to_numpy(), d.sec.to_numpy(np.int64), d.run_id.to_numpy()  # arrays
    z = mat["surrogate"][rows, -1]  # z_t per row
    S, s_ = np.zeros(len(z)), 0.0  # S per row; the running value
    for i in range(len(z)):  # for loop over the rows
        if i and (sec[i] - sec[i - 1] != 1 or run[i] != run[i - 1]):  # a gap: S restarts
            s_ = 0.0  # reset
        s_ = max(0.0, s_ + float(z[i]))  # Page's recursion
        S[i] = s_  # kept
    return d, rows, S  # the rows, their matrix positions and S


def deployable(sel: pd.DataFrame, nested: pd.DataFrame, steps: pd.DataFrame, material=E.material) -> tuple:  # function: P6's operating point of every detector per receiver, outer fold and share, from inner material only
    rows, kn_rows = [], []  # the points; every k-of-n candidate
    for rx in RXS:  # for loop over the receivers
        for outer in SESSIONS:  # for loop over the outer folds
            train = [s for s in SESSIONS if s != outer]  # its training sessions (each the inner held-out session of one inner fold)
            mat = material(rx, outer)  # the environment script's material of the fold (the T5 check passes a reader with the held-out rows removed)
            for mdl in ("lr", "rf"):  # the two supervised baselines: thresholds on the clean out-of-fold windows of P's nested predictions
                q = nested[(nested.model == mdl) & (nested.receiver == rx) & (nested.outer_held_out == outer) & nested.held_out.isin(train) & (nested.label == "clean")]  # those windows
                for s in SHARES:  # for loop over the shares
                    t = lowest_threshold(q.p.to_numpy(), len(q), s)  # the threshold
                    rows.append({"receiver": rx, "outer": outer, "detector": mdl, "share": s, "param": t, "param_name": "threshold", "inner_clean": len(q), "inner_clean_share": float((q.p >= t).mean()) if len(q) else None})  # the point
            per = {}  # session -> (stream seconds, their valid-row positions, S per valid row, 30 log-odds per valid row)
            for sess in train:  # for loop over the training sessions
                d, r_, S = surrogate_runs(mat, sess)  # its valid rows and S
                sd = session_seconds(steps, rx, sess)  # its windowed seconds (the stream)
                pos = pd.Series(np.arange(len(d)), index=d.sec.to_numpy()).reindex(sd.sec.to_numpy()).to_numpy()  # each windowed second's position among the valid rows (NaN: no state)
                per[sess] = (sd, pos, S, mat["surrogate"][r_])  # kept
            n_clean = sum(int((sd.label == "clean").sum()) for sd, _, _, _ in per.values())  # generator in sum(): pooled clean windowed seconds (undecidable ones silent)
            on_grid = lambda stat: {k: np.where(np.isfinite(p_), stat[k][np.nan_to_num(p_, nan=0).astype(int)] if len(stat[k]) else np.nan, np.nan) for k, (_, p_, _, _) in per.items()}  # lambda with a dict comprehension: a per-valid-row statistic placed on each session's stream grid
            S_g = on_grid({k: v[2] for k, v in per.items()})  # S_t on the stream grid
            clean_v = lambda g: np.concatenate([g[k][(per[k][0].label == "clean").to_numpy()] for k in per])  # lambda: the statistic at the pooled clean seconds
            for s in SHARES:  # CUSUM: the smallest h (decision mode, S never reset by an alarm) whose pooled clean share is at most s
                h = lowest_threshold(clean_v(S_g), n_clean, s)  # the threshold on S
                dl = []  # stopping-mode delays on the inner sessions, for the record
                for k, (sd, p_, _, Z) in per.items():  # for loop over the sessions
                    ok = np.isfinite(p_)  # seconds with a state
                    zg = np.where(ok, Z[np.nan_to_num(p_, nan=0).astype(int), -1] if len(Z) else np.nan, np.nan)  # z_t on the grid
                    st = E.stream(sd.sec.to_numpy(np.int64), sd.label.to_numpy(object), E.cusum(sd.sec.to_numpy(np.int64), np.nan_to_num(zg), ok, h, "stopping"), k, "stopping")  # the environment script's cusum in stopping mode through the stream
                    dl += [x["delay_s"] for x in st["latency"] if x["detected"]]  # delays of the detected intervals
                rows.append({"receiver": rx, "outer": outer, "detector": "cusum", "share": s, "param": h, "param_name": "h", "inner_clean": n_clean, "inner_clean_share": float((clean_v(S_g) >= h).sum() / n_clean) if n_clean else None, "inner_median_delay": float(np.median(dl)) if dl else None})  # the point
            for n in E.KN_N:  # k-of-n: for every (n, k), the smallest h meeting the share; the setting with the lowest pooled inner median delay is kept
                for k_ in range(1, n + 1):  # for loop over k
                    Q = on_grid({k: (np.sort(v[3][:, -n:], axis=1)[:, n - k_] if len(v[3]) else np.zeros(0)) for k, v in per.items()})  # the k-th largest of the last n log-odds (from the state's own history), on the grid
                    for s in SHARES:  # for loop over the shares
                        h = lowest_threshold(clean_v(Q), n_clean, s)  # the threshold
                        dl, det, iv = [], 0, 0  # delays; detected and all intervals
                        for k, (sd, p_, _, _) in per.items():  # for loop over the sessions
                            dec = (np.nan_to_num(Q[k], nan=-np.inf) >= h).astype(int)  # k of the last n reach h
                            st = E.stream(sd.sec.to_numpy(np.int64), sd.label.to_numpy(object), dec, k, "stopping")  # stopping mode
                            dl += [x["delay_s"] for x in st["latency"] if x["detected"]]  # delays
                            det, iv = det + st["detected"], iv + st["intervals"]  # counts
                        kn_rows.append({"receiver": rx, "outer": outer, "n": n, "k": k_, "share": s, "h": h, "inner_clean_share": float((clean_v(Q) >= h).sum() / n_clean) if n_clean else None, "inner_median_delay": float(np.median(dl)) if dl else np.inf, "inner_detected": det, "inner_intervals": iv})  # one candidate
            kn = pd.DataFrame([r for r in kn_rows if r["receiver"] == rx and r["outer"] == outer])  # this fold's candidates
            for s in SHARES:  # the kept k-of-n setting per share
                c = kn[kn.share == s].sort_values(["inner_median_delay", "n", "k"], kind="stable").iloc[0]  # lowest delay, ties to the smaller n, then k
                rows.append({"receiver": rx, "outer": outer, "detector": "kofn", "share": s, "param": float(c.h), "param_name": f"h (n={int(c.n)}, k={int(c.k)})", "n": int(c.n), "k": int(c.k), "inner_clean": n_clean, "inner_clean_share": c.inner_clean_share, "inner_median_delay": c.inner_median_delay})  # the point
            for arm in ("primary", "surrogate"):  # the agent: lambda by the inner stopping-mode clean share at the selected budget, two inner seeds
                a = sel[(sel.receiver == rx) & (sel.outer == outer) & (sel.arm == arm)].sort_values("lambda")  # its five lambdas
                for s in SHARES:  # for loop over the shares
                    okl = a[a.inner_stopping_clean_share.notna() & (a.inner_stopping_clean_share <= s)]  # the lambdas meeting the share
                    # dict literal over two lines: P6 as written takes the largest such lambda; the smallest is recorded beside it for the reader
                    rows.append({"receiver": rx, "outer": outer, "detector": f"agent_{arm}", "share": s, "param": float(okl["lambda"].max()) if len(okl) else None, "param_name": "lambda (largest meeting the share, P6 as written)",
                                 "param_alt_smallest_lambda": float(okl["lambda"].min()) if len(okl) else None, "inner_clean": int(a.inner_clean_s.iloc[0]), "inner_clean_share": float(okl.set_index("lambda").inner_stopping_clean_share[okl["lambda"].max()]) if len(okl) else None, "no_point": not len(okl)})  # dict literal continued
    return pd.DataFrame(rows), pd.DataFrame(kn_rows)  # the points and the k-of-n candidates


def load_q(spec: dict, models: Path, mat: dict):  # function: an outer agent rebuilt from its frozen weights (a fresh DQN whose network takes the saved values)
    m = DQN("MlpPolicy", make_env(mat, spec["arm"], spec["lambda"]), seed=spec["seed"], device="cpu", verbose=0, **DQN_KW)  # the same architecture
    w = E.load_npz(models / f"{job_id(spec)}.npz")  # the saved weights
    m.q_net.load_state_dict({k[len("q_net."):]: torch.as_tensor(v) for k, v in w.items() if k.startswith("q_net.")})  # dict comprehension: the online network's weights
    m.q_net_target.load_state_dict({k[len("q_net_target."):]: torch.as_tensor(v) for k, v in w.items() if k.startswith("q_net_target.")})  # and the target network's
    return m, mat  # the agent and its material


def implied(specs: list, models: Path) -> pd.DataFrame:  # function: P11, each frozen agent's implied threshold on its training-session states
    from sklearn.linear_model import LogisticRegression  # one-variable logistic regression
    import warnings  # the fit's convergence warnings are recorded, not hidden
    rows, cache = [], {}  # one row per outer agent; (receiver, outer) -> the environment script's material, read once
    for sp in specs:  # for loop over the outer agents
        key = (sp["receiver"], sp["outer"])  # the cell
        if key not in cache:  # if: not read yet
            cache[key] = E.material(*key)  # the environment script's material of the outer fold (unpacking the tuple into the two arguments)
        m, mat = load_q(sp, models, cache[key])  # the agent and its material
        xs, acts = [], []  # the evidence variable and the greedy actions
        for sess in [s for s in SESSIONS if s != sp["outer"]]:  # training sessions only
            d, r_, S = surrogate_runs(mat, sess)  # valid rows and S_t (decision mode)
            if not len(r_):  # no valid row (cannot)
                continue  # next
            if sp["arm"] == "surrogate":  # the surrogate arm: actions on (30 log-odds, S_t), against S_t
                o, x = signed_log(np.column_stack([mat["surrogate"][r_], S])), S  # tuple assignment: observation and evidence
            else:  # the primary arm: actions on its 108 values, against the current log-odds
                o, x = mat["primary"][r_], mat["surrogate"][r_, -1]  # tuple assignment
            acts.append(m.predict(o, deterministic=True)[0])  # greedy actions
            xs.append(x)  # the evidence
        a, x = np.concatenate(acts).astype(int), np.concatenate(xs)  # all training seconds
        row = {**{k: sp[k] for k in ("receiver", "outer", "arm", "lambda", "seed", "budget")}, "seconds": len(a), "alarm_share": float(a.mean()), "variable": "S_t" if sp["arm"] == "surrogate" else "log-odds z_t"}  # dict comprehension: the agent's identity
        if a.min() == a.max():  # the agent always or never alarms on these states
            rows.append({**row, "threshold": None, "slope": None, "note": "always" if a[0] else "never"})  # no crossing
            continue  # next agent
        mu, sd = float(x.mean()), float(x.std())  # standardised for the fit, threshold converted back
        with warnings.catch_warnings(record=True) as w:  # with: the fit's warnings recorded
            warnings.simplefilter("always")  # every warning kept
            f = LogisticRegression(penalty=None, max_iter=10000).fit(((x - mu) / sd)[:, None], a)  # unpenalised one-variable fit
        b0, b1 = float(f.intercept_[0]), float(f.coef_[0, 0])  # intercept and slope (standardised)
        rows.append({**row, "threshold": mu - sd * b0 / b1, "slope": b1 / sd, "note": ";".join(sorted({str(x_.message)[:60] for x_ in w}))})  # the level where p = 0.5
    return pd.DataFrame(rows)  # the table


def implied_summary(imp: pd.DataFrame) -> pd.DataFrame:  # function: P11's median over seeds per lambda and Spearman's rho with lambda per receiver, outer fold and arm
    from scipy.stats import spearmanr  # rank correlation
    med = imp.groupby(["receiver", "outer", "arm", "lambda"], sort=True).agg(median_threshold=("threshold", "median"), defined=("threshold", "count")).reset_index()  # median over the seeds
    rows = []  # one row per receiver, outer fold and arm
    for (rx, outer, arm), d in med.groupby(["receiver", "outer", "arm"], sort=True):  # for loop over the cells
        d = d.dropna(subset=["median_threshold"])  # lambdas with a threshold
        rho = float(spearmanr(d["lambda"], d.median_threshold).statistic) if len(d) >= 3 and d.median_threshold.nunique() > 1 else np.nan  # rho, defined with three or more distinct points
        rows.append({"receiver": rx, "outer": outer, "arm": arm, "lambdas_with_threshold": len(d), "spearman_rho": rho, "non_negative": bool(rho >= 0) if np.isfinite(rho) else None})  # the row
    return med, pd.DataFrame(rows)  # both tables


# ----------------------------------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------------------------------
def grid() -> None:  # function: --grid: inner states, the inner loop, the selection, the outer agents, the freeze; a run again resumes
    out = AGENT_DIR / GRID  # the grid's folder
    states, jobs, models = out / "states", out / "jobs", AGENT_DIR / "models" / GRID  # tuple assignment: its sub-folders (the models kept apart, their SHA-256 in the manifest)
    cells = [(rx, outer) for rx in RXS for outer in SESSIONS]  # list comprehension: the 15 receiver and outer-fold cells
    clock = {}  # phase wall times
    t0 = time.perf_counter()  # clock
    B = E.A.build()  # P rebuilt (the adopted-pipeline script's gates prove it bitwise), with the inner transforms kept
    ck = inner_states(B, cells, states)  # the 60 inner state files and their checks
    npr = B["nested"]["predictions"]  # P's nested out-of-fold predictions (the adopted-pipeline script's, including the six fits with the screen not applied)
    E.write_gz(npr[["model", "receiver", "outer_held_out", "held_out", "sec", "label", "p", "dec", "screen_applied"]], out / "nested_predictions.csv.gz")  # written for P6 (no held-out row: each outer fold's rows are its training sessions)
    del B  # not needed by the jobs
    E.write_csv(ck, out / "inner_states_checks.csv")  # written
    if not (ck.dec_bitwise.all() and ck.train_rows_match.all()):  # if: an inner transform does not reproduce its fit
        raise SystemExit("STOP: an inner transform does not reproduce the committed inner decision values")  # stop
    E.write_csv(config_table(), out / "config.csv")  # the configuration
    clock["_prepare"] = time.perf_counter() - t0  # seconds
    t1 = time.perf_counter()  # clock
    si = specs_inner()  # 1,200 inner jobs
    tm_in = run_jobs(si, states, jobs, models)  # run (finished jobs are skipped)
    clock["_inner"] = time.perf_counter() - t1  # seconds
    cur = inner_curve(jobs, si)  # every snapshot
    E.write_gz(cur, out / "inner_curve.csv.gz")  # written
    sel = select(cur)  # P3's budgets and P10's comparison
    E.write_csv(sel, out / "selection.csv")  # written
    E.write_csv(learned(sel), out / "learned.csv")  # P10 as amended
    t2 = time.perf_counter()  # clock
    so = specs_outer(sel)  # 750 outer jobs
    tm_out = run_jobs(so, states, jobs, models)  # run, each stopped at its budget
    clock["_outer"] = time.perf_counter() - t2  # seconds
    t3 = time.perf_counter()  # clock
    steps = read_steps()  # the environment script's step table
    nested = pd.read_csv(out / "nested_predictions.csv.gz", dtype={"outer_held_out": str, "held_out": str}, float_precision="round_trip")  # read back exactly
    dp, kn = deployable(sel, nested, steps)  # P6
    E.write_csv(dp, out / "deployable_points.csv")  # the frozen points
    E.write_csv(kn, out / "kofn_candidates.csv")  # every k-of-n candidate
    imp = implied(so, models)  # P11
    E.write_csv(imp, out / "implied_thresholds.csv")  # per agent
    med, rho = implied_summary(imp)  # medians and rho
    E.write_csv(med, out / "implied_medians.csv")  # written
    E.write_csv(rho, out / "implied_spearman.csv")  # written
    clock["_freeze"] = time.perf_counter() - t3  # seconds
    man = [{"job": job_id(s), "stage": s["stage"], "output_sha256": sha_file(jobs / f"{job_id(s)}.json.gz")} for s in si + so]  # list comprehension: the job manifest
    E.write_csv(pd.DataFrame(man), out / "jobs_manifest.csv")  # written
    mm = [{"model_file": p.name, "sha256": sha_file(p), "bytes": p.stat().st_size} for p in sorted(models.glob("*.npz"))]  # list comprehension: the model manifest (the model files themselves are not kept in this folder)
    E.write_csv(pd.DataFrame(mm), out / "models_manifest.csv")  # written
    frozen = ["selection.csv", "learned.csv", "deployable_points.csv", "kofn_candidates.csv", "implied_thresholds.csv", "models_manifest.csv", "jobs_manifest.csv"]  # list: the files the freeze commit fixes
    E.write_csv(pd.DataFrame([{"file": f, "sha256": sha_file(out / f)} for f in frozen]), out / "freeze_manifest.csv")  # list comprehension: their digests (the evaluation script checks them)
    tm = pd.concat([tm_in, tm_out, pd.DataFrame([{"job": k, "seconds": round(v, 1)} for k, v in clock.items()])], ignore_index=True)  # every job's timing and the phases
    E.write_csv(tm, out / "timing.csv")  # wall clock by nature (declared non-deterministic)
    print({k: round(v / 3600, 2) for k, v in clock.items()}, "hours")  # dict comprehension: the phases in hours


def main() -> None:  # function: the command line
    ap = argparse.ArgumentParser()  # the parser
    ap.add_argument("--grid", action="store_true")  # --grid: the full grid
    a = ap.parse_args()  # parsed
    if not a.grid:  # nothing else is implemented (the pilot was run by this file with --pilot)
        raise SystemExit("usage: 6_reinforcement_learning/02_train_agents.py --grid")  # stop
    grid()  # run


if __name__ == "__main__":  # guard: runs main() only when the file is executed, not when imported (spawned workers import it)
    main()  # runs main
