#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
Why the agent loses: a diagnosis of the agent experiment's RQ2 result. No verdict is scored again.

Design: the pre-registered diagnostics D1 to D6, hypotheses H1 to H4 and checks C1 to C9, with the H2 test and the fixes F-a to F-d
added before any diagnostic was read. Every quantity is computed on inner leave-one-session-out folds of an outer fold's four training
sessions; the outer held-out sessions are not run again here (the stored tables of the agent experiment and of the RQ4 study are read
for D2 and D6 only); 2.6.12 is not read.
  --regen  runs each of the agent experiment's 1,200 inner jobs again exactly as the training script's run_job does and stops it at its
           cell's selected budget; the gate requires the greedy actions on P4's episodes and on the inner held-out stream to hash to the
           SHA-256 the agent experiment stored for that job and budget.
           Kept per job (Data/processed/rl_diagnosis/inner_q/): Q(continue) and Q(alarm) on every valid second of the four sessions along
           the decision-mode observation path, S_t on that path, the stopping-mode events of the inner held-out stream, and the greedy
           policy's per-episode outcomes on P4's episodes of the inner held-out session and on episodes built by P4's rule on the inner
           training sessions. Weights are not kept: the gate makes them regenerable bitwise.
  --diag   D1-D6 and H2 from those files and the stored tables of the environment, the agent experiment and the RQ4 study; tables in
           Data/processed/rl_diagnosis/, the figures.
It imports the evaluation script (which loads the training, environment and adopted-pipeline scripts).
Reads   Data/processed/rl_environment/, rl_agents/training/ (states, jobs, selection, inner curve, nested predictions), rl_agents/evaluation/,
        rl_agents/models/training/, rq4/
Writes  Data/processed/rl_diagnosis/ and Graph/diagnosis_* .. diagnosis_*
"""
from __future__ import annotations  # future import: type hints kept as text (the project's convention)

import argparse  # standard library: the command line
import hashlib  # standard library: SHA-256 digests
import importlib.util  # standard library: loads the evaluation script (its name starts with a digit)
import json  # standard library: the specification digest
import multiprocessing  # standard library: the spawn context of the worker pool
import sys  # standard library: module registry and exit status
import time  # standard library: wall-clock timing (timing table only, declared)
from concurrent.futures import ProcessPoolExecutor, as_completed  # standard library: the pool of worker processes
from pathlib import Path  # standard library: paths

import numpy as np  # numpy: arrays
import pandas as pd  # pandas: tables

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
_spec = importlib.util.spec_from_file_location("evaluate_agents_for_diagnosis", PROJECT / "codes" / "6_reinforcement_learning" / "03_evaluate_agents.py")  # where the evaluation script is
B12 = importlib.util.module_from_spec(_spec)  # an empty module object
sys.modules["evaluate_agents_for_diagnosis"] = B12  # registered, so its own references resolve
_spec.loader.exec_module(B12)  # the evaluation script's top level (it loads the training, environment and adopted-pipeline scripts); its main() is not run
T, E = B12.T, B12.E  # the training script and the environment script
A = E.A  # the adopted-pipeline script

import torch  # PyTorch, SB3's backend (12 has already set it to one thread)
torch.set_num_threads(1)  # one thread again, explicitly: multithreaded torch crashes beside scikit-learn's OpenMP runtime (the environment Part E)
from stable_baselines3 import DQN  # the Deep Q-Network (the training script's configuration)

DIAG_DIR = E.PROC / "rl_diagnosis"  # this section's folder
QDIR = DIAG_DIR / "inner_q"  # the regenerated inner agents' tables, one .npz per job
GRAPH = E.GRAPH  # the figure folder
GRID12 = T.AGENT_DIR / T.GRID  # the agent experiment's frozen grid folder
STATES12 = GRID12 / "states"  # the agent experiment's inner-loop primary states
JOBS12 = GRID12 / "jobs"  # the agent experiment's job outputs (episode logs, snapshot records)
SEED = T.SEED  # 20260914
SESSIONS, RXS, LAMBDAS = T.SESSIONS, T.RXS, T.LAMBDAS  # the five folds, the three receivers, the lambda grid
ARMS = ("primary", "surrogate")  # tuple: the two registered arms
SEALED = A.SEALED  # "2.6.12", never read here
REGEN_VERSION = "r15-regen-1"  # a label in the job digest: changing what a job stores changes it, so a run again recomputes
SESS_CODE = {s: i for i, s in enumerate(SESSIONS)}  # dict comprehension: session -> small integer, for the stored arrays
KIND_CODE = {"onset": 0, "clean": 1}  # dict: episode kind -> integer
OUTCOME_CODE = {"detection": 0, "false_alarm": 1, "miss": 2, "clean_end": 3}  # dict: episode outcome -> integer


# ----------------------------------------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------------------------------------
def rt(path: Path, **kw) -> pd.DataFrame:  # function: a committed table read with every float back to the double that was written (the evaluation script's reader)
    return B12.rt(path, **kw)  # the evaluation script's rt: round-trip floats, sessions as text


def assert_c(a: np.ndarray) -> np.ndarray:  # function: the C-order rule (every logistic-regression score on a C-ordered matrix); returns the array
    if not a.flags["C_CONTIGUOUS"]:  # if: not C-ordered (the score would not be bitwise reproducible on Apple Accelerate)
        raise SystemExit("STOP: a matrix for a logistic-regression score is not C-ordered")  # stop
    return a  # the same array


def q_values(model, obs: np.ndarray) -> np.ndarray:  # function: the online network's Q(continue), Q(alarm) for a batch, exactly as SB3's greedy predict computes them
    t, _ = model.policy.obs_to_tensor(obs)  # tuple unpacking: SB3's own conversion of the batch to a tensor (float32 inside the network)
    with torch.no_grad():  # with-block: no gradient bookkeeping
        return model.q_net(t).cpu().numpy()  # the (n, 2) Q values, float32 as the network computes them


def stretches(env: E.SpoofingDetectionEnv, d: pd.DataFrame) -> tuple:  # function: the training script's agent_decisions' grid: each windowed second's state row, the mask and the stretch starts and ends
    sec = d.sec.to_numpy(np.int64)  # the seconds
    rows = np.array([env.mat["key"].get((s, int(t)), -1) for s, t in zip(d.session, d.sec)])  # list comprehension: each second's state row, -1 where none
    ok = rows >= 0  # seconds with a state
    run = np.where(ok, env.mat["run"][np.maximum(rows, 0)], -1)  # np.where: the window run of each second with a state
    new = ok & np.r_[True, (np.diff(sec) != 1) | (np.diff(run) != 0) | ~ok[:-1]]  # np.r_ concatenation: True where a contiguous stretch of states begins (S restarts there)
    seg = np.cumsum(new)  # running count: a stretch number per second
    end = np.zeros(len(sec), int)  # each stretch start's end (exclusive)
    for g in np.unique(seg[ok]):  # for loop over the stretches
        ix = np.flatnonzero(ok & (seg == g))  # its positions
        end[ix[0]] = ix[-1] + 1  # its end, stored at its start
    return sec, rows, ok, new, end  # the grid


def stream_q(model, env: E.SpoofingDetectionEnv, d: pd.DataFrame) -> tuple:  # function: Q values and S_t of every windowed second on the decision-mode path (the training script's agent_decisions' observations, batched per stretch as it batches)
    sec, rows, ok, new, end = stretches(env, d)  # tuple unpacking: the grid
    Q = np.full((len(sec), 2), np.nan, np.float32)  # Q per second (NaN: no state)
    S = np.full(len(sec), np.nan)  # S_t per second (surrogate arm; NaN otherwise)
    for i in np.flatnonzero(new):  # for loop over the stretch starts
        x = env.X[rows[i:end[i]]]  # the stretch's state rows
        if env.arm == "surrogate":  # if: the surrogate arm, S restarting at the stretch start (the training script's agent_decisions' nested obs)
            s_, Sx = 0.0, np.zeros(len(x))  # tuple assignment: the running value from 0; S per second
            for k in range(len(x)):  # for loop over the seconds
                s_ = max(0.0, s_ + float(x[k, -1]))  # Page's recursion on z_t, the last of the 30 log-odds
                Sx[k] = s_  # kept
            S[i:end[i]] = Sx  # stored
            o = T.signed_log(np.column_stack([x, Sx]))  # P2's compression of the 31 values
        else:  # the primary arm has no memory
            o = x  # the rows themselves
        Q[i:end[i]] = q_values(model, o)  # the stretch's Q values in one batch, as predict batches it
    return sec, ok, Q, S  # seconds, mask, Q values, S


def episode_arrays(env: E.SpoofingDetectionEnv, pairs: list, lam: float, which: int) -> dict:  # function: per-episode specification and the greedy policy's closed-form outcome, as arrays
    rows = []  # one row per episode
    bits = []  # each episode's greedy actions, packed to bits
    for ep, a in pairs:  # for loop over (episode, greedy actions)
        ty = env.mat["type"][ep["rows"]]  # its step types
        c, out, dl = T.episode_cost(ty, np.asarray(a), ep["kind"], lam, env.m)  # the training script's closed-form undiscounted cost
        # tuple appended: set, kind, session, start, onset, length, cost, outcome, delay
        rows.append((which, KIND_CODE[ep["kind"]], SESS_CODE[ep["session"]], ep["start"], ep["onset"], len(ty), c, OUTCOME_CODE[out], dl))
        bits.append(np.asarray(a, np.uint8))  # the actions
    r = np.array(rows, dtype=float) if rows else np.zeros((0, 9))  # the table as a float matrix (integers are exact in a double)
    return {"spec": r, "acts": np.packbits(np.concatenate(bits)) if bits else np.zeros(0, np.uint8)}  # dict literal: the table and all actions packed in episode order


# ----------------------------------------------------------------------------------------------------------------------
# --regen: the agent experiment's inner jobs to their selected budgets, gated on the agent experiment's stored digests
# ----------------------------------------------------------------------------------------------------------------------
def regen_specs() -> list:  # function: every inner job of the agent experiment with its cell's selected budget and the digests the agent experiment stored at that budget
    sel = rt(GRID12 / "selection.csv")  # P3's selected budgets per (receiver, outer, arm, lambda)
    cur = rt(GRID12 / "inner_curve.csv.gz")  # every inner job's snapshot records
    x = cur.merge(sel[["receiver", "outer", "arm", "lambda", "selected_budget"]], on=["receiver", "outer", "arm", "lambda"], validate="many_to_one")  # each record with its cell's budget
    x = x[x.budget == x.selected_budget].sort_values(["receiver", "outer", "inner", "arm", "lambda", "seed"], kind="stable")  # the records at the selected budget, in a fixed order
    specs = []  # the job list
    for r in x.to_dict("records"):  # for loop over the records, each a dict (the column 'lambda' is a Python keyword, so no attribute access)
        sp = {"stage": "inner", "receiver": r["receiver"], "outer": r["outer"], "inner": r["inner"], "arm": r["arm"], "lambda": float(r["lambda"]), "seed": int(r["seed"]), "budget": T.SCHEDULE}  # dict literal: the agent experiment's own job specification (its budget field is the schedule)
        specs.append({"spec": sp, "stop": int(r["selected_budget"]), "actions_sha256": r["actions_sha256"], "stream_sha256": r["stream_sha256"], "criterion": float(r["criterion"])})  # with the stop and the stored digests
    if len(specs) != 1200:  # if: not the agent experiment's 1,200 inner jobs (cannot)
        raise SystemExit(f"STOP: {len(specs)} inner jobs at the selected budgets, not 1,200")  # stop
    return specs  # the list


def regen_digest(job: dict) -> str:  # function: SHA-256 of what a regenerated job depends on: the agent experiment's specification, its stop, the environment script's and the training script's code, and this step's version label
    d = {"job": job, "dqn": T.DQN_KW, "version": REGEN_VERSION, "code": [T.sha_code(PROJECT / "codes" / n) for n in ("6_reinforcement_learning/01_environment.py", "6_reinforcement_learning/02_train_agents.py")]}  # dict literal with a list comprehension: the digest's input
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()  # the digest


def regen_job(job: dict) -> dict:  # function: one inner job run again to its selected budget (runs in a worker process); returns the gate results
    torch.set_num_threads(1)  # one thread in the worker too
    if torch.get_num_threads() != 1:  # if: the setting did not take (cannot)
        raise SystemExit("STOP: torch is not on one thread")  # stop
    spec, b = job["spec"], job["stop"]  # the agent experiment's specification and the selected budget
    jid = T.job_id(spec)  # the agent experiment's name of the job
    out, hfile = QDIR / f"{jid}.npz", QDIR / f"{jid}.sha256"  # tuple assignment: the output and its hash entry
    dig = regen_digest(job)  # the job's digest
    if out.exists() and hfile.exists() and hfile.read_text().split() == [T.sha_file(out), dig]:  # if: finished before with the same inputs
        return {"job": jid, "skipped": True}  # nothing to do
    T.set_random_seed(spec["seed"])  # as the training script's run_job: Python's random, numpy and torch seeded before anything is built
    train, evm = T.job_material(spec, STATES12)  # the training script's materials: the three inner training sessions and the inner held-out session
    env = T.make_env(train, spec["arm"], spec["lambda"])  # the training script's training stack (Monitor <- reward/100 <- signed-log for the surrogate)
    model = DQN("MlpPolicy", env, seed=spec["seed"], device="cpu", verbose=0, **T.DQN_KW)  # the agent exactly as the agent experiment built it
    eenv = E.SpoofingDetectionEnv(evm, spec["arm"], spec["lambda"])  # the inner held-out evaluation environment (never stepped)
    tenv = E.SpoofingDetectionEnv(train, spec["arm"], spec["lambda"])  # an environment over the inner training sessions (never stepped; its episode() builds episodes)
    eps = T.eval_episodes(eenv)  # P4's evaluation episodes of the inner held-out session
    teps = T.eval_episodes(tenv)  # the same rule on the inner training sessions (new; D1's optimisation test)
    steps = T.read_steps()  # the environment script's step table
    sd = T.session_seconds(steps, spec["receiver"], spec["inner"])  # the inner held-out session's windowed seconds (the stream)
    itrain = [s for s in SESSIONS if s not in (spec["outer"], spec["inner"])]  # the three inner training sessions
    res = {}  # what the snapshot leaves behind

    def on_mark(mdl, n):  # nested function: everything kept, computed with the weights at the selected budget (before that step's update, as the agent experiment's snapshots)
        pairs = T.evaluate(mdl, eenv, eps, spec["lambda"])  # the training script's greedy actions on P4's episodes
        acts = np.concatenate([a for _, a in pairs]).astype(np.uint8)  # all of them in order
        dd, ds = T.agent_decisions(mdl, eenv, sd)  # the training script's decision- and stopping-mode decisions on the inner held-out stream
        res["actions_sha256"] = hashlib.sha256(acts.tobytes()).hexdigest()  # the agent experiment's digest of the episode actions
        res["stream_sha256"] = hashlib.sha256(np.concatenate([dd, ds]).astype(np.uint8).tobytes()).hexdigest()  # the agent experiment's digest of the stream decisions
        res["criterion"] = T.summarise(pairs, eenv, spec["lambda"])["criterion"]  # the training script's criterion, compared with the stored one
        sec, ok, Q, S = stream_q(mdl, eenv, sd)  # Q and S on the inner held-out stream
        res["q_argmax_equal"] = bool(np.array_equal(np.where(ok, np.argmax(np.nan_to_num(Q), axis=1), 0), dd))  # C2: the stored Q values give the greedy decisions bitwise
        tr = [stream_q(mdl, tenv, T.session_seconds(steps, spec["receiver"], s)) for s in itrain]  # list comprehension: the same on each inner training session
        tpairs = T.evaluate(mdl, tenv, teps, spec["lambda"])  # the greedy actions on the training-session episodes
        he, te = episode_arrays(eenv, pairs, spec["lambda"], 0), episode_arrays(tenv, tpairs, spec["lambda"], 1)  # tuple assignment: per-episode outcomes, held-out and training
        # dict literal over four lines: the arrays kept (Q in float32, the network's own precision)
        res["arrays"] = {"ho_sec": sec, "ho_ok": ok.astype(np.uint8), "ho_q": Q, "ho_S": S, "ho_dec": dd.astype(np.int8), "ho_stop": ds.astype(np.int8),
                         "tr_sess": np.concatenate([np.full(len(t_[0]), SESS_CODE[s], np.int8) for t_, s in zip(tr, itrain)]), "tr_sec": np.concatenate([t_[0] for t_ in tr]),  # dict literal continued
                         "tr_ok": np.concatenate([t_[1] for t_ in tr]).astype(np.uint8), "tr_q": np.concatenate([t_[2] for t_ in tr]), "tr_S": np.concatenate([t_[3] for t_ in tr]),  # dict literal continued
                         "ep_ho": he["spec"], "ep_ho_acts": he["acts"], "ep_tr": te["spec"], "ep_tr_acts": te["acts"]}  # dict literal continued

    cb = T.Snapshots((b,), on_mark, b)  # the training script's callback: one mark at the selected budget, and the stop there
    t0 = time.perf_counter()  # clock (timing table only)
    model.learn(total_timesteps=T.SCHEDULE, callback=cb)  # the training script's schedule, stopped at the budget
    dt = time.perf_counter() - t0  # seconds
    gate = res["actions_sha256"] == job["actions_sha256"] and res["stream_sha256"] == job["stream_sha256"]  # C1: the agent is the agent experiment's
    if not gate:  # if: not the agent experiment's agent
        return {"job": jid, "skipped": False, "gate": False, "q_argmax_equal": res["q_argmax_equal"], "seconds": round(dt, 1)}  # reported, nothing written
    E.write_npz(res["arrays"], out.with_name(out.name + ".part"))  # the arrays with fixed member dates, under a temporary name
    out.with_name(out.name + ".part").replace(out)  # an atomic rename on the same disk
    T.write_atomic(hfile, f"{T.sha_file(out)} {dig}\n".encode())  # then its hash entry
    return {"job": jid, "skipped": False, "gate": True, "q_argmax_equal": res["q_argmax_equal"], "criterion_equal": res["criterion"] == job["criterion"], "seconds": round(dt, 1), "steps": int(model.num_timesteps)}  # the record


def regen() -> None:  # function: --regen, every job through a pool of three spawned workers, then the manifest
    QDIR.mkdir(parents=True, exist_ok=True)  # the folder
    jobs = regen_specs()  # the 1,200 jobs
    ctx = multiprocessing.get_context("spawn")  # spawn: each worker starts a fresh interpreter
    rows = []  # one record per job
    with ProcessPoolExecutor(max_workers=T.N_WORKERS, mp_context=ctx) as ex:  # with-block: the pool of three, shut down at the end
        futs = {ex.submit(regen_job, j): j for j in jobs}  # dict comprehension: future -> its job
        for k, f in enumerate(as_completed(futs), 1):  # for loop over the jobs as they finish, counted from 1
            r = f.result()  # its record (a worker's exception is raised here)
            rows.append(r)  # kept
            print(f"  [{k}/{len(jobs)}] {r['job']}: {'skipped' if r['skipped'] else ('gate ok' if r['gate'] else 'GATE FAILED')} {r.get('seconds', '')}", flush=True)  # progress
    tim = pd.DataFrame(rows).sort_values("job").reset_index(drop=True)  # the records in a fixed order
    E.write_csv(tim, DIAG_DIR / "diagnosis_regen_timing.csv")  # wall times and gate flags of this run (timing varies by nature: declared, not byte-compared)
    man = []  # the manifest: every job's file and digest
    for j in jobs:  # for loop over the jobs in their fixed order
        jid = T.job_id(j["spec"])  # its name
        p = QDIR / f"{jid}.npz"  # its file
        man.append({"job": jid, "stop_budget": j["stop"], "file": p.name, "exists": p.exists(), "sha256": T.sha_file(p) if p.exists() else ""})  # one row
    man = pd.DataFrame(man)  # the table
    E.write_csv(man, DIAG_DIR / "diagnosis_regen_manifest.csv")  # written
    failed = [r["job"] for r in rows if not r["skipped"] and not r["gate"]]  # list comprehension: jobs whose agent is not the agent experiment's
    if failed or not man.exists.all():  # if: any gate failed or a file is missing
        raise SystemExit(f"STOP: regeneration gate failed for {len(failed)} jobs: {failed[:5]}")  # stop: nothing downstream may run
    print(f"regeneration: {len(man)} of {len(jobs)} jobs gated and written", flush=True)  # the summary


# ----------------------------------------------------------------------------------------------------------------------
# --diag: shared helpers
# ----------------------------------------------------------------------------------------------------------------------
CHECKS = []  # list: every check of this run, written to diagnosis_checks.csv
H_EP = np.r_[B12.H_GRID, np.inf]  # np.r_ concatenation: the evaluation script's CUSUM grid (0, then 1 to 10,000 on a log scale) and h = infinity (never alarm)
CELL_MIN = 8  # the pre-registered thresholds: a defect needs at least 8 of the 15 receiver x lambda cells
N_BOOT = B12.N_BOOT  # 10,000 bootstrap resamples (P8's)


def check(name: str, ok: bool, detail: str) -> None:  # function: records one check and prints it
    CHECKS.append({"check": name, "pass": bool(ok), "detail": detail})  # kept
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)  # printed


def load_job(jid: str) -> dict:  # function: one regenerated job's arrays
    return E.load_npz(QDIR / f"{jid}.npz")  # the environment script's reader of the fixed-date .npz


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:  # function: the area under the ROC curve of a score, positives against negatives (the Mann-Whitney statistic with ties counted half)
    from scipy.stats import rankdata  # average ranks, so ties count half
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]  # defined values only
    if not len(pos) or not len(neg):  # if: one class is empty
        return np.nan  # undefined
    r = rankdata(np.r_[pos, neg])  # np.r_ concatenation: ranks of all values together
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))  # U / (n_pos n_neg)


def kappa(a: np.ndarray, b: np.ndarray) -> float:  # function: Cohen's kappa of two binary decision series
    a, b = np.asarray(a, bool), np.asarray(b, bool)  # booleans
    if not len(a):  # if: no second
        return np.nan  # undefined
    po = float((a == b).mean())  # observed agreement
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))  # agreement expected by chance from the two alarm shares
    return (po - pe) / (1 - pe) if pe < 1 else np.nan  # conditional expression: kappa, undefined when both are constant and equal


def jaccard(a: np.ndarray, b: np.ndarray) -> float:  # function: shared alarm seconds over the seconds where either alarms
    a, b = np.asarray(a, bool), np.asarray(b, bool)  # booleans
    u = int((a | b).sum())  # either
    return float((a & b).sum() / u) if u else np.nan  # conditional expression: undefined when neither alarms


def implied(x: np.ndarray, a: np.ndarray) -> tuple:  # function: P11's method (the training script's implied): a one-variable unpenalised logistic fit of the actions on x; (level where p = 0.5, slope)
    from sklearn.linear_model import LogisticRegression  # the logistic fit
    import warnings  # its convergence warnings are expected at separable data and kept out of the output
    x, a = np.asarray(x, float), np.asarray(a, int)  # arrays
    k = np.isfinite(x)  # defined values only
    x, a = x[k], a[k]  # filtered
    if len(a) < 2 or a.min() == a.max():  # if: the agent always or never alarms here
        return np.nan, np.nan  # no crossing
    mu, sd = float(x.mean()), float(x.std())  # standardised for the fit, as the training script's implied
    if sd == 0:  # if: x does not vary
        return np.nan, np.nan  # no fit
    with warnings.catch_warnings():  # with-block: warnings silenced inside
        warnings.simplefilter("ignore")  # the training script's implied records them; here the slope column shows a separable fit
        f = LogisticRegression(penalty=None, max_iter=10000).fit(assert_c(np.ascontiguousarray(((x - mu) / sd)[:, None])), a)  # unpenalised one-variable fit on a C-ordered matrix
    b0, b1 = float(f.intercept_[0]), float(f.coef_[0, 0])  # intercept and slope (standardised)
    return (mu - sd * b0 / b1, b1 / sd) if b1 != 0 else (np.nan, 0.0)  # conditional expression: the level and the slope in x's units


def boot_median(v: np.ndarray, strata: np.ndarray, seed: int = SEED, n_boot: int = N_BOOT) -> tuple:  # function: the median with a 95 % stratified bootstrap percentile interval (runs resampled within each outer fold, as the evaluation script's strat_boot)
    v, s = np.asarray(v, float), np.asarray(strata)  # arrays
    k = np.isfinite(v)  # defined runs
    v, s = v[k], s[k]  # filtered
    if not len(v):  # if: nothing defined
        return np.nan, np.nan, np.nan, 0  # undefined
    rng = np.random.default_rng(seed)  # the seeded generator
    parts = [v[s == g][rng.integers(0, int((s == g).sum()), size=(n_boot, int((s == g).sum())))] for g in sorted(set(s))]  # list comprehension: n_boot resamples per stratum, in sorted order
    Bm = np.median(np.concatenate(parts, axis=1), axis=1)  # each resample's median
    return float(np.median(v)), float(np.percentile(Bm, 2.5)), float(np.percentile(Bm, 97.5)), len(v)  # point, interval, runs


def page_path(z: np.ndarray) -> np.ndarray:  # function: Page's S_t = max(0, S_(t-1) + z_t) from S = 0, by the same sequential recursion as the environment script's environment
    S, s_ = np.zeros(len(z)), 0.0  # S per step; the running value
    for i in range(len(z)):  # for loop over the steps
        s_ = max(0.0, s_ + float(z[i]))  # the recursion
        S[i] = s_  # kept
    return S  # the path


def refractory_from(sec: np.ndarray, ev: np.ndarray, lab: np.ndarray, t0: int) -> tuple:  # function: the refractory mark at t0 (the transfer script's refractory_at) and the label of the event that set it
    e = np.flatnonzero(ev & (sec < t0))  # positions of the events before the first second
    if not e.size:  # if: none
        return False, np.nan, ""  # not refractory
    j = e[-1]  # the last one (the grid is in time order)
    age = int(t0 - sec[j])  # its age
    return age <= E.REFRACTORY, age, str(lab[j])  # refractory when 30 s or less (the environment script's rule allows a new event only after more than 30 s), its age and its second's scoring label


def in_alarm_at(sec: np.ndarray, dec: np.ndarray, decided: np.ndarray, t0: int) -> str:  # function: the state at onset (the transfer script's onset_state): the decision on the last decided second at most 60 s before t0
    k = np.flatnonzero(decided & (sec < t0))  # decided seconds before the first second
    if not k.size or t0 - sec[k[-1]] > 60:  # if: none, or older than 60 s
        return "not decidable"  # no state
    return "in alarm" if dec[k[-1]] > 0 else "not in alarm"  # conditional expression: the state


def category(lab: np.ndarray, step: np.ndarray, sec: np.ndarray, pooled: list, runs: pd.DataFrame) -> np.ndarray:  # function: D3's step categories of a session's windowed seconds
    out = np.full(len(sec), "other", object)  # default
    out[step == "pre"] = "pre-change"  # clean or strict quiet negative (training label clean)
    near = np.zeros(len(sec), bool)  # uncertain seconds within 60 s before a pooled onset
    for a, _ in pooled:  # for loop over the pooled intervals
        near |= (sec >= a - 60) & (sec < a)  # the minute before it
    out[(step == "uncertain") & near] = "uncertain, 60 s before onset"  # the stretch every onset is approached through
    out[(step == "uncertain") & ~near] = "uncertain, other"  # every other uncertain second
    since = np.full(len(sec), np.inf)  # seconds since the attack run began
    for r in runs.itertuples():  # for loop over the attack runs of the session (training labels)
        k = (sec >= r.onset) & (sec <= r.end)  # inside it
        since[k] = np.minimum(since[k], sec[k] - r.onset)  # its age
    out[(step == "post") & (since < 30)] = "post-change, 0-29 s"  # the first 30 s of an attack
    out[(step == "post") & (since >= 30)] = "post-change, 30 s on"  # later
    return out  # the categories


# ----------------------------------------------------------------------------------------------------------------------
# --diag: the inner context (training sessions of one outer fold; no outer held-out second, no 2.6.12)
# ----------------------------------------------------------------------------------------------------------------------
def inner_ctx(rx: str, outer: str, steps: pd.DataFrame, nested: pd.DataFrame) -> dict:  # function: per training session of the fold: the stream grid, z_t, CUSUM's S_t, the regression's probability, labels and intervals
    mat = E.material(rx, outer)  # the environment script's material of the outer fold (its key holds training rows only)
    on = rt(E.ENV_DIR / "onsets.csv")  # every attack run (for D3's categories)
    per = {}  # session -> its arrays
    for s in [x for x in SESSIONS if x != outer]:  # for loop over the four training sessions (each an inner held-out session)
        d, r_, S = T.surrogate_runs(mat, s)  # its valid rows and decision-mode S_t (the training script's, restarting at every gap)
        sd = T.session_seconds(steps, rx, s)  # its windowed seconds (the stream grid)
        pos = pd.Series(np.arange(len(d)), index=d.sec.to_numpy()).reindex(sd.sec.to_numpy()).to_numpy()  # each windowed second's position among the valid rows (NaN: no state)
        ok = np.isfinite(pos)  # seconds with a state
        pi = np.nan_to_num(pos, nan=0).astype(int)  # positions as integers (0 where none; masked by ok)
        q = nested[(nested.model == "lr") & (nested.receiver == rx) & (nested.outer_held_out == outer) & (nested.held_out == s)].set_index("sec").p  # P's out-of-fold probability of the session (nested predictions)
        sec = sd.sec.to_numpy(np.int64)  # the seconds
        # dict literal over three lines: the session's arrays
        per[s] = {"sd": sd, "sec": sec, "lab": sd.label.to_numpy(object), "step": sd.step.to_numpy(object), "train_label": sd.train_label.to_numpy(object), "ok": ok,
                  "S": np.where(ok, S[pi] if len(S) else np.nan, np.nan), "z": np.where(ok, mat["surrogate"][r_][pi, -1] if len(r_) else np.nan, np.nan), "p": q.reindex(sec).to_numpy(),  # dict literal continued
                  "pooled": E.SC.pooled_intervals(s), "runs": on[(on.receiver == rx) & (on.session == s)]}  # dict literal continued
        if SEALED in set(sd.session) or outer in set(sd.session):  # if: a 2.6.12 or outer held-out second in the grid (cannot)
            raise SystemExit(f"STOP: {rx} {outer}: a sealed or held-out second in an inner stream")  # stop
    clean_S = np.concatenate([v["S"][v["lab"] == "clean"] for v in per.values()])  # list comprehension: CUSUM's S at the pooled clean seconds (NaN where no state: silent)
    clean_p = nested[(nested.model == "lr") & (nested.receiver == rx) & (nested.outer_held_out == outer) & nested.held_out.isin(list(per)) & (nested.label == "clean")].p.to_numpy()  # the regression's clean out-of-fold windows (the training script's deployable's)
    return {"mat": mat, "per": per, "clean_S": clean_S, "clean_p": clean_p}  # dict literal: the context


def thresholds(ctx: dict, q: float) -> tuple:  # function: CUSUM's h and the regression's threshold at the share q, by the training script's deployable's rule (the training script's lowest_threshold over the pooled inner clean seconds)
    return T.lowest_threshold(ctx["clean_S"], len(ctx["clean_S"]), q), T.lowest_threshold(ctx["clean_p"], len(ctx["clean_p"]), q)  # (h_q, t_q)


def baseline_runs(v: dict, h: float, t: float) -> dict:  # function: CUSUM at h and the regression at t on one inner held-out stream, both modes (the evaluation script's baseline_decisions' rules)
    z = np.nan_to_num(v["z"])  # z_t (0 where no state; those seconds are not valid)
    cd, cs = E.cusum(v["sec"], z, v["ok"], h, "decision"), E.cusum(v["sec"], z, v["ok"], h, "stopping")  # tuple assignment: the environment script's CUSUM in each mode
    ld = np.nan_to_num(v["p"], nan=-np.inf) >= t  # the regression alarms where its probability reaches t
    return {"cusum": (cd, cs, v["ok"]), "lr": (ld, E.events(v["sec"], ld), np.ones(len(v["sec"]), bool))}  # dict literal: (decision, stopping events, decided seconds) per baseline


def episode_frames(rx: str, outer: str, inner: str, spec_ho: np.ndarray, spec_tr: np.ndarray) -> dict:  # function: the stored episodes rebuilt from the environment script's definition: their step types and S_t, per set
    sp = {"receiver": rx, "outer": outer, "inner": inner, "arm": "surrogate", "lambda": 100.0, "seed": SEED, "stage": "inner", "budget": T.SCHEDULE}  # dict literal: any surrogate job of the cell (episodes do not depend on arm, lambda or seed)
    train, evm = T.job_material(sp, STATES12)  # the training script's materials
    out = {}  # set -> list of (types, S, kind)
    for name, spec, m_ in (("ho", spec_ho, evm), ("tr", spec_tr, train)):  # for loop over the two sets
        env = E.SpoofingDetectionEnv(m_, "surrogate", 100.0)  # an environment that only builds episodes
        L = []  # the episodes
        for r in spec:  # for loop over the stored rows: set, kind, session, start, onset, length, cost, outcome, delay
            kind = "onset" if int(r[1]) == 0 else "clean"  # its kind
            ep = env.episode(kind, SESSIONS[int(r[2])], int(r[3]), int(r[4]))  # the environment script's episode
            if len(ep["rows"]) != int(r[5]):  # if: not the stored length (cannot)
                raise SystemExit(f"STOP: a rebuilt episode of {rx} {outer} {inner} differs from the stored one")  # stop
            L.append((m_["type"][ep["rows"]], page_path(m_["surrogate"][ep["rows"], -1]), kind))  # step types, S_t along the episode from 0 (masked alarms do not reset it, as in the environment script's step) and kind
        out[name] = L  # kept
    out["m"], out["cost"] = train["m"], {}  # tuple assignment: the inner miss penalty; a cache of cost matrices per lambda
    return out  # the episodes


def mix_mean(c: np.ndarray, spec: np.ndarray) -> np.ndarray:  # function: a cost averaged as training samples episodes (half onset episodes, every onset weighted equally; half clean episodes), per column of c
    c = np.asarray(c, float).reshape(len(spec), -1)  # episodes x columns
    on = spec[:, 1] == 0  # onset episodes
    keys = spec[on][:, [2, 4]]  # their (session, onset)
    per = [c[on][(keys == k).all(axis=1)].mean(axis=0) for k in np.unique(keys, axis=0)]  # list comprehension: the mean over each onset's episodes
    return 0.5 * np.mean(per, axis=0) + 0.5 * c[~on].mean(axis=0)  # half onsets (equal weight each), half clean episodes


def cusum_costs(eps: list, lam: float, m: float) -> np.ndarray:  # function: the undiscounted Shiryaev cost of "alarm when S_t >= h" (S from 0 at the episode start, as in training) on each episode, for every h of H_EP
    C = np.zeros((len(eps), len(H_EP)))  # cost per episode and h
    for i, (ty, S, kind) in enumerate(eps):  # for loop over the episodes (S along each, from episode_frames)
        M = np.maximum.accumulate(np.where(ty != E.STEP["uncertain"], S, -np.inf))  # running maximum over the seconds where an alarm ends the episode
        j = np.searchsorted(M, H_EP, side="left")  # the first such second with S >= h, per h (len(ty) when none)
        cp = np.r_[0, np.cumsum(ty == E.STEP["post"])]  # np.r_ concatenation: silent post-change seconds before each position
        stop = j < len(ty)  # an alarm ended the episode
        ty_j = np.where(stop, ty[np.minimum(j, len(ty) - 1)], -1)  # the step type at that alarm
        miss = (m + cp[-1]) if kind == "onset" else 0.0  # conditional expression: a miss costs every post-change second plus m; a clean episode that ends silent costs 0
        C[i] = np.where(stop, cp[j] + np.where(ty_j == E.STEP["pre"], lam, 0.0), miss)  # np.where: delay, plus lambda for a false alarm; or the miss
    return C  # the matrix


# ----------------------------------------------------------------------------------------------------------------------
# D1 and H2 (and D3's out-of-sample margins), one pass over the 1,200 regenerated agents
# ----------------------------------------------------------------------------------------------------------------------
def smax_from_log(rx: str, outer: str, inner: str, lam: float, seed: int, b: int, mat: dict) -> float:  # function: the largest S_t along the training episodes the agent experiment's surrogate inner job ran before its selected budget
    sp = {"stage": "inner", "receiver": rx, "outer": outer, "inner": inner, "arm": "surrogate", "lambda": lam, "seed": seed, "budget": T.SCHEDULE}  # dict literal: the agent experiment's specification
    r = T.read_gz_json(JOBS12 / f"{T.job_id(sp)}.json.gz")  # its output with the episode log
    done = np.cumsum([e["steps"] for e in r["episodes"]]) <= b  # list comprehension in cumsum: episodes finished by the selected budget
    best = 0.0  # the largest S seen
    for e, d in zip(r["episodes"], done):  # for loop over the episodes
        if not d:  # past the budget
            break  # the agent at the budget never saw them
        rows = [mat["key"][(e["session"], e["start"] + i)] for i in range(e["steps"])]  # list comprehension: its rows (consecutive seconds)
        z = mat["surrogate"][rows, -1]  # z_t along them
        W = np.cumsum(z)  # running sum
        best = max(best, float((W - np.minimum(0.0, np.minimum.accumulate(W))).max()))  # Page's S_t = W_t - min(0, min W) along the steps taken, its maximum
    return best  # the value


def d1_h2(only: tuple | None = None) -> tuple:  # function: D1(a)-(c), H2 and the out-of-sample Q margins, from the regenerated agents; returns the tables (only: a (receiver, outer) pair, for a partial test)
    steps = T.read_steps()  # the environment script's step table (labels, step types)
    nested = rt(GRID12 / "nested_predictions.csv.gz")  # P's nested out-of-fold predictions (training sessions only)
    sel = rt(GRID12 / "selection.csv")  # the selection with each cell's inner decision-mode clean share
    pts = rt(T.AGENT_DIR / "evaluation" / "deployable_points.csv")  # the agent experiment's points (C4: h_q and t_q at the registered shares reproduce them)
    runs, onsets, margins, c4 = [], [], [], []  # D1 rows per run; H2 rows per onset; D3 margin rows; C4 comparisons
    for rx in RXS:  # for loop over the receivers
        for outer in SESSIONS:  # for loop over the outer folds
            if only is not None and (rx, outer) != only:  # if: a partial test restricted to one fold
                continue  # skip the others
            ctx = inner_ctx(rx, outer, steps, nested)  # the fold's inner context
            for s_ in T.SHARES:  # C4: the rule at the registered shares gives the agent experiment's CUSUM and regression points
                h, t = thresholds(ctx, s_)  # tuple assignment
                p_ = pts[(pts.receiver == rx) & (pts.outer == outer) & (pts.share == s_)].set_index("detector").param  # the agent experiment's points
                c4.append(h == p_["cusum"] and t == p_["lr"])  # equal to the bit
            for inner in [x for x in SESSIONS if x != outer]:  # for loop over the inner folds
                v = ctx["per"][inner]  # the inner held-out stream
                cat = category(v["lab"], v["step"], v["sec"], v["pooled"], v["runs"])  # D3's categories
                fr = None  # the rebuilt episodes, once per inner fold
                for arm in ARMS:  # for loop over the arms
                    for lam in LAMBDAS:  # for loop over the lambdas
                        sr = sel[(sel.receiver == rx) & (sel.outer == outer) & (sel.arm == arm) & (sel["lambda"] == float(lam))].iloc[0]  # the cell's selection row
                        q = float(sr.inner_decision_clean_share)  # P6's quantity: the agent's pooled inner decision-mode clean share
                        h_q, t_q = thresholds(ctx, q)  # the baselines at the agent's share
                        base = baseline_runs(v, h_q, t_q)  # both baselines on this inner held-out stream
                        for det, (dd, ev, decided) in base.items():  # for loop over the baselines: H2's onset rows
                            st = E.stream(v["sec"], v["lab"], ev, inner, "stopping")  # the stream's delays
                            for a, _ in v["pooled"]:  # for loop over the pooled intervals
                                rf, age, el = refractory_from(v["sec"], np.asarray(ev, bool), v["lab"], a)  # the refractory mark and its event's label
                                dl = [x["delay_s"] for x in st["latency"] if x["onset"] == a][0]  # list comprehension: this interval's delay
                                onsets.append({"receiver": rx, "outer": outer, "inner": inner, "arm": arm, "lambda": float(lam), "seed": -1, "detector": det, "onset": a, "refractory": rf, "event_age_s": age, "event_label": el, "refractory_uncertain": bool(rf and el == "uncertain"), "onset_state": in_alarm_at(v["sec"], np.asarray(dd, int), decided, a), "delay_s": dl})  # one row
                        for seed in T.INNER_SEEDS:  # for loop over the two inner seeds
                            sp = {"stage": "inner", "receiver": rx, "outer": outer, "inner": inner, "arm": arm, "lambda": float(lam), "seed": seed, "budget": T.SCHEDULE}  # dict literal: the agent experiment's specification
                            J = load_job(T.job_id(sp))  # the regenerated agent's arrays
                            if not np.array_equal(J["ho_sec"], v["sec"]):  # if: not the same grid (cannot)
                                raise SystemExit(f"STOP: {T.job_id(sp)}: its stream grid differs")  # stop
                            ok = J["ho_ok"].astype(bool)  # seconds with a state
                            dec, ev = J["ho_dec"] > 0, J["ho_stop"] > 0  # tuple assignment: decision-mode alarms; stopping-mode events
                            Sx = J["ho_S"] if arm == "surrogate" else v["S"]  # the agent's own S (surrogate) or CUSUM's (primary, for reference)
                            margin = (J["ho_q"][:, 1] - J["ho_q"][:, 0]).astype(float)  # Q(alarm) - Q(continue)
                            cd = base["cusum"][0]  # CUSUM's decision-mode alarms at h_q
                            st_a = E.stream(v["sec"], v["lab"], ev, inner, "stopping")  # the agent's stopping-mode figures
                            st_c = E.stream(v["sec"], v["lab"], base["cusum"][1], inner, "stopping")  # CUSUM's
                            ce = v["sec"][base["cusum"][1]]  # CUSUM's event seconds
                            ae = v["sec"][ev]  # the agent's
                            coinc = float(np.mean([np.abs(ce - t_).min() <= 5 for t_ in ae])) if len(ae) and len(ce) else np.nan  # list comprehension: share of agent events within 5 s of a CUSUM event
                            clean = (v["lab"] == "clean") & ok  # decidable clean seconds
                            row = {"receiver": rx, "outer": outer, "inner": inner, "arm": arm, "lambda": float(lam), "seed": seed, "q": q, "h_q": h_q, "t_q": t_q,  # dict literal: the run's identity and the matched thresholds
                                   "agent_decision_clean_alarms": int((dec & (v["lab"] == "clean")).sum()), "clean_seconds": int((v["lab"] == "clean").sum()),  # the agent's decision-mode clean alarms over the clean windowed seconds (P6's quantity)
                                   "kappa_cusum": kappa(dec[ok], cd[ok]), "jaccard_cusum": jaccard(dec[ok], cd[ok]), "kappa_lr": kappa(dec[ok], base["lr"][0][ok]),  # same-second agreement on the decidable seconds
                                   "share_alarms_S_ge_h": float((Sx[dec & ok] >= h_q).mean()) if (dec & ok).any() else np.nan, "event_coincidence_5s": coinc,  # the agent's alarms at S >= h_q; event coincidence
                                   "implied_S": implied(Sx[ok], dec[ok])[0] if arm == "surrogate" else np.nan, "implied_z": implied(v["z"][ok], dec[ok])[0]}  # dict literal continued: implied thresholds on the inner held-out seconds
                            for c_ in ("clean", "quiet", "uncertain", "attack"):  # for loop over the label categories: Jaccard with CUSUM per category
                                k_ = ok & {"clean": v["lab"] == "clean", "quiet": (v["lab"] == "uncertain") & (v["train_label"] == "clean"), "uncertain": (v["lab"] == "uncertain") & (v["train_label"] != "clean"), "attack": np.isin(v["lab"], A.ATTACK)}[c_]  # dict lookup: the category's seconds
                                row[f"jaccard_cusum_{c_}"] = jaccard(dec[k_], cd[k_])  # per category
                                row[f"agent_alarm_share_{c_}"] = float(dec[k_].mean()) if k_.any() else np.nan  # the agent's alarm share there
                            row["agent_delay_median_s"], row["cusum_delay_median_s"] = st_a["delay_median_s"], st_c["delay_median_s"]  # tuple assignment: stopping-mode median delays on the interval(s)
                            if arm == "surrogate":  # H1 (D1(c)): the S range the agent trained on
                                smax = smax_from_log(rx, outer, inner, float(lam), seed, int(sr.selected_budget), ctx["mat"])  # the largest training S before the budget
                                out_r = clean & (Sx > smax)  # clean seconds beyond it
                                row.update({"train_S_max": smax, "clean_share_beyond_train_S": float(out_r.sum() / clean.sum()) if clean.any() else np.nan,  # dict.update: H1's quantities
                                            "clean_alarms_beyond_share": float((dec & out_r).sum() / (dec & clean).sum()) if (dec & clean).any() else np.nan,  # share of the agent's clean alarms beyond the range
                                            "implied_S_in_range": implied(Sx[ok & (Sx <= smax)], dec[ok & (Sx <= smax)])[0]})  # the implied threshold on in-range seconds only
                            if fr is None:  # the episodes of this inner fold, rebuilt once
                                fr = episode_frames(rx, outer, inner, J["ep_ho"], J["ep_tr"])  # rebuilt and checked against the stored lengths
                            if lam not in fr["cost"]:  # the cost matrices of this lambda, computed once per inner fold
                                fr["cost"][lam] = (cusum_costs(fr["tr"], float(lam), fr["m"]), cusum_costs(fr["ho"], float(lam), fr["m"]))  # training and held-out
                            Ctr, Cho = fr["cost"][lam]  # tuple unpacking: CUSUM's costs per episode and h, training and held-out
                            hstar = int(np.argmin(Ctr.mean(axis=0)))  # h*: the lowest mean training cost (the first, i.e. the smallest h, on a tie)
                            hmix = int(np.argmin(mix_mean(Ctr, J["ep_tr"])))  # sensitivity: h chosen on the training mixture's weighting
                            ktr, kho = J["ep_tr"], J["ep_ho"]  # the stored episode tables (column 1 kind, 7 outcome)
                            cl_t, on_t = ktr[:, 1] == 1, ktr[:, 1] == 0  # clean and onset training episodes
                            S_fa = np.array([np.argmax(np.maximum.accumulate(np.where(ty != E.STEP["uncertain"], S, -np.inf)) >= H_EP[hstar]) if (np.where(ty != E.STEP["uncertain"], S, -np.inf) >= H_EP[hstar]).any() else -1 for ty, S, _ in fr["tr"]])  # list comprehension: CUSUM(h*)'s stopping position per training episode (-1: none)
                            c_fa = np.array([p_ >= 0 and ty[p_] == E.STEP["pre"] for p_, (ty, _, _) in zip(S_fa, fr["tr"])])  # list comprehension: CUSUM(h*) false alarms
                            row.update({"agent_train_cost_mix": float(mix_mean(ktr[:, 6], ktr)[0]), "cusum_train_cost_mix": float(mix_mean(Ctr[:, hmix], ktr)[0]), "h_mix": float(H_EP[hmix]),  # dict.update: the training-mixture sensitivity
                                        "agent_train_clean_fa_share": float((ktr[cl_t, 7] == OUTCOME_CODE["false_alarm"]).mean()), "cusum_train_clean_fa_share": float(c_fa[cl_t].mean()),  # false alarms on clean training episodes
                                        "agent_train_onset_fa_share": float((ktr[on_t, 7] == OUTCOME_CODE["false_alarm"]).mean()), "cusum_train_onset_fa_share": float(c_fa[on_t].mean()),  # false alarms on onset episodes (before the change)
                                        "agent_train_miss_share": float((ktr[on_t, 7] == OUTCOME_CODE["miss"]).mean()), "agent_heldout_miss_share": float((kho[kho[:, 1] == 0, 7] == OUTCOME_CODE["miss"]).mean())})  # misses
                            row.update({"h_star": float(H_EP[hstar]), "agent_train_cost": float(J["ep_tr"][:, 6].mean()), "cusum_train_cost": float(Ctr[:, hstar].mean()),  # dict.update: D1(b)
                                        "agent_heldout_cost": float(J["ep_ho"][:, 6].mean()), "cusum_heldout_cost": float(Cho[:, hstar].mean()), "train_episodes": len(fr["tr"]), "heldout_episodes": len(fr["ho"])})  # dict.update continued
                            runs.append(row)  # kept
                            for a, _ in v["pooled"]:  # H2: the agent's onset rows
                                rf, age, el = refractory_from(v["sec"], ev, v["lab"], a)  # its refractory mark
                                dl = [x["delay_s"] for x in st_a["latency"] if x["onset"] == a][0]  # list comprehension: this interval's delay
                                onsets.append({"receiver": rx, "outer": outer, "inner": inner, "arm": arm, "lambda": float(lam), "seed": seed, "detector": "agent", "onset": a, "refractory": rf, "event_age_s": age, "event_label": el, "refractory_uncertain": bool(rf and el == "uncertain"), "onset_state": in_alarm_at(v["sec"], dec.astype(int), ok, a), "delay_s": dl})  # one row
                            pre, post = margin[ok & (cat == "pre-change")], margin[ok & np.isin(cat, ["post-change, 0-29 s", "post-change, 30 s on"])]  # D3: the margins of pre- and post-change seconds
                            mrow = {"receiver": rx, "outer": outer, "inner": inner, "arm": arm, "lambda": float(lam), "seed": seed, "auroc_margin": auroc(post, pre),  # dict literal: D3's out-of-sample margin figures
                                    "auroc_z": auroc(v["z"][ok & np.isin(cat, ["post-change, 0-29 s", "post-change, 30 s on"])], v["z"][ok & (cat == "pre-change")])}  # the AUROC of z_t on the same seconds
                            for c_ in ("pre-change", "uncertain, 60 s before onset", "uncertain, other", "post-change, 0-29 s", "post-change, 30 s on"):  # for loop over the categories
                                x_ = margin[ok & (cat == c_)]  # its margins
                                mrow[f"median_margin[{c_}]"] = float(np.median(x_)) if len(x_) else np.nan  # the median margin
                                mrow[f"share_positive[{c_}]"] = float((x_ > 0).mean()) if len(x_) else np.nan  # the share where the greedy action is an alarm
                                mrow[f"n[{c_}]"] = len(x_)  # how many seconds
                            margins.append(mrow)  # kept
            print(f"  D1/H2: {rx} {outer} done", flush=True)  # progress
    check("C4 h_q and t_q reproduce the agent experiment's CUSUM and regression points at the registered shares", all(c4), f"{sum(c4)} of {len(c4)} (receiver, outer, share) cases equal to the bit")  # C4
    return pd.DataFrame(runs), pd.DataFrame(onsets), pd.DataFrame(margins)  # the three tables


# ----------------------------------------------------------------------------------------------------------------------
# cell summaries and the defect/limit verdicts of D1 and H2
# ----------------------------------------------------------------------------------------------------------------------
def d1_cells(R: pd.DataFrame) -> pd.DataFrame:  # function: D1's cells (arm x receiver x lambda): the optimisation test, the generalisation test and H1
    rows = []  # one row per cell
    for (arm, rx, lam), d in R.groupby(["arm", "receiver", "lambda"], sort=True):  # for loop over the cells
        dtr = (d.agent_train_cost - d.cusum_train_cost).to_numpy()  # paired training-episode differences, agent minus CUSUM(h*)
        dho = (d.agent_heldout_cost - d.cusum_heldout_cost).to_numpy()  # the same on the inner held-out episodes
        mt, lt, ht, n = boot_median(dtr, d.outer.to_numpy())  # median with its stratified interval
        mh, lh, hh, _ = boot_median(dho, d.outer.to_numpy())  # the same, held out
        r = {"arm": arm, "receiver": rx, "lambda": lam, "runs": n, "train_diff_median": mt, "train_diff_lo": lt, "train_diff_hi": ht, "heldout_diff_median": mh, "heldout_diff_lo": lh, "heldout_diff_hi": hh,  # dict literal: D1(b)
             "optimisation_failure": bool(mt > 0 and lt > 0), "train_at_or_below": bool(lt <= 0 or mt <= 0), "heldout_worse": bool(mh > 0 and lh > 0),  # the per-cell readings
             "train_diff_mix_median": float(np.median(d.agent_train_cost_mix - d.cusum_train_cost_mix)), "agent_train_clean_fa_share": float(d.agent_train_clean_fa_share.median()), "cusum_train_clean_fa_share": float(d.cusum_train_clean_fa_share.median()),  # sensitivity and mechanism
             "agent_train_onset_fa_share": float(d.agent_train_onset_fa_share.median()), "cusum_train_onset_fa_share": float(d.cusum_train_onset_fa_share.median()), "agent_train_miss_share": float(d.agent_train_miss_share.median()),  # dict literal continued
             "kappa_cusum_median": float(d.kappa_cusum.median()), "jaccard_cusum_median": float(d.jaccard_cusum.median()), "share_alarms_S_ge_h_median": float(d.share_alarms_S_ge_h.median()),  # D1(a)
             "event_coincidence_median": float(d.event_coincidence_5s.median()), "h_q_median": float(d.h_q.median()), "implied_S_median": float(d.implied_S.median()), "h_star_median": float(d.h_star.median())}  # dict literal continued
        if arm == "surrogate":  # H1's cell figures
            r.update({"clean_share_beyond_train_S_median": float(d.clean_share_beyond_train_S.median()), "clean_alarms_beyond_share_median": float(d.clean_alarms_beyond_share.median()),  # dict.update: medians over runs
                      "h1_cell": bool(d.clean_share_beyond_train_S.median() > 0.20 and d.clean_alarms_beyond_share.median() >= 0.5), "implied_S_in_range_median": float(d.implied_S_in_range.median())})  # the cell meets H1's rule
        rows.append(r)  # kept
    return pd.DataFrame(rows)  # the table


def h2_cells(O: pd.DataFrame) -> pd.DataFrame:  # function: the H2 test's cells (arm x receiver x lambda): the share of onsets refractory from an uncertain-second event, agent against the best of CUSUM and the regression at its share
    rows = []  # one row per cell
    for (arm, rx, lam), d in O.groupby(["arm", "receiver", "lambda"], sort=True):  # for loop over the cells
        sh = d.groupby("detector").refractory_uncertain.mean()  # the share per detector (agent over seeds, folds and onsets; baselines over folds and onsets)
        n = d.groupby("detector").size()  # onsets counted per detector
        best = min(sh["cusum"], sh["lr"])  # the best baseline: the lower share
        rows.append({"arm": arm, "receiver": rx, "lambda": lam, "agent_share": float(sh["agent"]), "cusum_share": float(sh["cusum"]), "lr_share": float(sh["lr"]), "best_baseline_share": float(best),  # dict literal: the shares
                     "agent_onsets": int(n["agent"]), "baseline_onsets": int(n["cusum"]), "excess": float(sh["agent"] - best), "h2_cell": bool(sh["agent"] - best > 0.20),  # the excess and the cell's reading
                     "agent_any_refractory": float(d[d.detector == "agent"].refractory.mean()), "agent_in_alarm": float((d[d.detector == "agent"].onset_state == "in alarm").mean())})  # context: any refractory mark, in alarm at onset
    return pd.DataFrame(rows)  # the table


# ----------------------------------------------------------------------------------------------------------------------
# D4 (and D3's training curves): the episode logs of the agent experiment's 750 outer and 1,200 inner jobs
# ----------------------------------------------------------------------------------------------------------------------
def rowmaps(mat: dict) -> dict:  # function: per training session of a material, an array from second to state row (-1: none), for vectorised episode rebuilding
    out = {}  # session -> (first second, array)
    ix = mat["index"]  # the state index
    for s in sorted(set(ix[ix.role == "train"].session)):  # for loop over the training sessions
        d = ix[(ix.role == "train") & (ix.session == s)]  # its rows
        lo = int(d.sec.min())  # its first second
        a = np.full(int(d.sec.max()) - lo + 1, -1, np.int64)  # -1 everywhere
        a[d.sec.to_numpy(np.int64) - lo] = d.row.to_numpy(np.int64)  # the rows at their seconds
        out[s] = (lo, a)  # kept
    return out  # the maps


def job_transitions(r: dict, mat: dict, maps: dict, lam: float, upto: int | None) -> dict:  # function: one job's logged training transitions in order: step type, action, episode, onset id, and the per-episode check of the Monitor return
    eps = r["episodes"]  # the log: finished episodes in order
    if upto is not None:  # an inner job: only the episodes finished by its selected budget (the agent there never saw later ones)
        keep = np.cumsum([e["steps"] for e in eps]) <= upto  # list comprehension in cumsum: finished in time
        eps = [e for e, k in zip(eps, keep) if k]  # list comprehension: those episodes
    n = np.array([e["steps"] for e in eps], np.int64)  # list comprehension: steps per episode
    rows = np.concatenate([maps[e["session"]][1][e["start"] - maps[e["session"]][0] + np.arange(e["steps"])] for e in eps]) if eps else np.zeros(0, np.int64)  # list comprehension: each episode's consecutive seconds -> rows
    if (rows < 0).any():  # if: a second without a state (cannot: episodes run over valid steps only)
        raise SystemExit("STOP: a logged episode crosses a second without a state")  # stop
    act = np.concatenate([np.unpackbits(np.frombuffer(bytes.fromhex(e["actions_hex"]), np.uint8))[:e["steps"]] for e in eps]) if eps else np.zeros(0, np.uint8)  # list comprehension: each episode's actions from its packed hex
    ty = mat["type"][rows]  # step types of every transition
    epi = np.repeat(np.arange(len(eps)), n)  # np.repeat: each transition's episode number
    last = np.zeros(len(rows), bool)  # the last transition of each episode
    last[np.cumsum(n) - 1] = True  # set
    onset_ep = np.array([e["kind"] == "onset" for e in eps], bool)  # list comprehension: onset episodes
    term = np.array([bool(e["terminated"]) for e in eps], bool)  # list comprehension: terminated (not truncated)
    rew = np.where((ty == E.STEP["post"]) & (act == 0), -1.0, 0.0) + np.where((ty == E.STEP["pre"]) & (act == 1), -lam, 0.0)  # np.where: the registered reward per step
    miss = last & onset_ep[epi] & term[epi] & (ty == E.STEP["post"]) & (act == 0)  # an onset episode ending in silence at its interval's end
    rew = rew - np.where(miss, r["m"], 0.0)  # the miss penalty m on top
    ret = np.bincount(epi, weights=rew, minlength=len(eps)) / T.REWARD_SCALE if eps else np.zeros(0)  # np.bincount with weights: each episode's return, scaled as the agent saw it
    mon = np.array([e["monitor_return"] for e in eps])  # list comprehension: the Monitor's logged returns
    ids = {k: i for i, k in enumerate(sorted({(e["session"], e["onset"]) for e in eps if e["kind"] == "onset"}))}  # dict comprehension over a sorted set comprehension: each distinct (session, onset) -> a number
    oid = np.where(onset_ep[epi], np.array([ids.get((e["session"], e["onset"]), -1) for e in eps], np.int64)[epi], -1) if eps else np.zeros(0, np.int64)  # list comprehension: each transition's onset number (-1 for clean episodes)
    return {"ty": ty, "act": act, "epi": epi, "oid": oid, "ret_err": float(np.abs(ret - mon).max()) if len(mon) else 0.0, "n": int(n.sum()), "episodes": len(eps), "mon": mon, "cum": np.cumsum(n)}  # dict literal: the arrays and the check


def buffer_stats(tr: dict, size: int, batch: int, start: int) -> dict:  # function: the replay buffer's composition at every update (SB3: the last `size` transitions, `batch` sampled uniformly with replacement from step `start` on)
    N = tr["n"]  # transitions
    if N <= start:  # if: no update (cannot at the agent experiment's budgets)
        return {}  # nothing
    u = np.arange(start + 1, N + 1)  # the update steps (SB3 trains once the step count exceeds learning_starts)
    lo = np.maximum(0, u - size)  # the buffer's first transition at each update
    cnt = (u - lo).astype(float)  # its fill
    cp = np.r_[0, np.cumsum(tr["ty"] == E.STEP["post"])]  # np.r_ concatenation: post-change transitions so far
    cd = np.r_[0, np.cumsum((tr["ty"] == E.STEP["post"]) & (tr["act"] == 1))]  # detections (the only terminal targets of 0 without bootstrapping)
    p = (cp[u] - cp[lo]) / cnt  # the post-change share of the buffer at each update
    dsh = (cd[u] - cd[lo]) / cnt  # the detection share
    half = u > N / 2  # the second half of training
    ons = sorted(set(tr["oid"][tr["oid"] >= 0]))  # the distinct onsets of the job's episodes
    pres = np.zeros(len(u))  # distinct onsets with a post-change transition in the buffer
    for o in ons:  # for loop over the onsets
        c = np.r_[0, np.cumsum((tr["oid"] == o) & (tr["ty"] == E.STEP["post"]))]  # its post-change transitions so far
        pres += (c[u] - c[lo]) > 0  # present at an update when any lies in the buffer
    # dict literal over two lines: the second-half and whole-training figures
    return {"post_per_batch_h2": float((batch * p[half]).mean()), "p_any_post_h2": float((1 - (1 - p[half]) ** batch).mean()), "detection_share_h2": float(dsh[half].mean()), "onsets_in_buffer_h2": float(pres[half].mean()),
            "post_per_batch_all": float((batch * p).mean()), "detection_share_all": float(dsh.mean()), "distinct_onsets": len(ons)}  # dict literal continued


def d4_logs(sel: pd.DataFrame, only: tuple | None = None) -> tuple:  # function: D4 per job and D3's training-return curves, from every logged episode (only: a (receiver, outer) pair, for a partial test)
    jobs, curves = [], []  # D4 rows per job; D3 curve rows
    marks = np.arange(5_000, T.SCHEDULE + 1, 5_000)  # the step marks of the training curves
    outer_specs = T.specs_outer(sel)  # the agent experiment's 750 outer jobs at their budgets
    inner_specs = [j["spec"] for j in regen_specs()]  # the 1,200 inner jobs
    budget = sel.set_index(["receiver", "outer", "arm", "lambda"]).selected_budget  # cell -> selected budget
    for rx in RXS:  # for loop over the receivers
        for outer in SESSIONS:  # for loop over the outer folds
            if only is not None and (rx, outer) != only:  # if: a partial test restricted to one fold
                continue  # skip the others
            mat = E.material(rx, outer)  # its material (training rows only in its key)
            maps = rowmaps(mat)  # second -> row per training session
            for sp in [s for s in outer_specs + inner_specs if s["receiver"] == rx and s["outer"] == outer]:  # list comprehension: the fold's jobs
                r = T.read_gz_json(JOBS12 / f"{T.job_id(sp)}.json.gz")  # the job's output with its log
                b = int(budget[(rx, outer, sp["arm"], float(sp["lambda"]))])  # its cell's selected budget
                tr = job_transitions(r, mat, maps, float(sp["lambda"]), b if sp["stage"] == "inner" else None)  # its transitions (an inner job up to the budget)
                st = buffer_stats(tr, T.DQN_KW["buffer_size"], T.DQN_KW["batch_size"], T.DQN_KW["learning_starts"])  # the buffer
                ty = tr["ty"]  # step types
                # dict literal over three lines: the job's D4 row
                jobs.append({"stage": sp["stage"], "receiver": rx, "outer": outer, "inner": sp["inner"], "arm": sp["arm"], "lambda": float(sp["lambda"]), "seed": sp["seed"], "budget": b, "transitions": tr["n"], "episodes": tr["episodes"],
                             "share_pre": float((ty == 0).mean()), "share_post": float((ty == 1).mean()), "share_uncertain": float((ty == 2).mean()), "detections": int(((ty == 1) & (tr["act"] == 1)).sum()),  # dict literal continued
                             "false_alarms": int(((ty == 0) & (tr["act"] == 1)).sum()), "masked_alarms": int(((ty == 2) & (tr["act"] == 1)).sum()), "monitor_return_max_err": tr["ret_err"], "unlogged_tail": b - tr["n"], **st})  # dict unpacking: the buffer figures
                if sp["stage"] == "inner":  # D3's training-return curve: an inner job's full log (100,000 steps)
                    full = job_transitions(r, mat, maps, float(sp["lambda"]), None)  # every logged episode
                    roll = pd.Series(full["mon"] * T.REWARD_SCALE).rolling(100, min_periods=20).mean().to_numpy()  # the unscaled return, rolling mean over 100 episodes
                    k = np.searchsorted(full["cum"], marks, side="right") - 1  # the last episode finished by each mark
                    curves.append({"receiver": rx, "outer": outer, "inner": sp["inner"], "arm": sp["arm"], "lambda": float(sp["lambda"]), "seed": sp["seed"], **{f"ret_{m_}": float(roll[j]) if j >= 0 else np.nan for m_, j in zip(marks, k)}})  # dict comprehension: the curve at each mark
            print(f"  D4: {rx} {outer} done", flush=True)  # progress
    return pd.DataFrame(jobs), pd.DataFrame(curves)  # both tables


def d4_cells(J: pd.DataFrame) -> pd.DataFrame:  # function: D4's cells (arm x receiver x lambda) over the 25 outer agents (5 folds x 5 seeds), with the inner jobs beside them
    rows = []  # one row per cell
    for (arm, rx, lam), d in J[J.stage == "outer"].groupby(["arm", "receiver", "lambda"], sort=True):  # for loop over the cells
        di = J[(J.stage == "inner") & (J.arm == arm) & (J.receiver == rx) & (J["lambda"] == lam)]  # the cell's inner jobs
        # dict literal over two lines: the medians
        rows.append({"arm": arm, "receiver": rx, "lambda": lam, "jobs": len(d), "post_per_batch_h2": float(d.post_per_batch_h2.median()), "p_any_post_h2": float(d.p_any_post_h2.median()), "detection_share_h2": float(d.detection_share_h2.median()),
                     "onsets_in_buffer_h2": float(d.onsets_in_buffer_h2.median()), "distinct_onsets": float(d.distinct_onsets.median()), "share_post": float(d.share_post.median()), "inner_post_per_batch_h2": float(di.post_per_batch_h2.median()),  # dict literal continued
                     "d4_cell": bool(d.post_per_batch_h2.median() < 1 or d.detection_share_h2.median() < 0.01)})  # the cell meets D4's defect rule
    return pd.DataFrame(rows)  # the table


# ----------------------------------------------------------------------------------------------------------------------
# D3: the rest (criterion against budget, variance, in-sample margins of the 750 outer agents, seed spread at the points)
# ----------------------------------------------------------------------------------------------------------------------
def d3_budget(cur: pd.DataFrame, sel: pd.DataFrame) -> pd.DataFrame:  # function: per cell (receiver, outer, arm, lambda): is the criterion still falling at 100k, and the seed and fold parts of its variance at the budget
    rows = []  # one row per cell
    for (rx, outer, arm, lam), d in cur.groupby(["receiver", "outer", "arm", "lambda"], sort=True):  # for loop over the 150 cells
        w = d.pivot_table(index=["inner", "seed"], columns="budget", values="criterion")  # runs x budgets
        c = w.mean()  # the mean over the 8 runs per budget
        diff = (w[80_000] - w[100_000]).to_numpy()  # per-run fall from 80k to 100k
        se = float(diff.std(ddof=1) / np.sqrt(len(diff)))  # its standard error
        falling = bool(c[80_000] > c[90_000] > c[100_000] and c[80_000] - c[100_000] > se)  # the pre-registered rule: the last three decreasing, by more than the standard error
        b = int(sel[(sel.receiver == rx) & (sel.outer == outer) & (sel.arm == arm) & (sel["lambda"] == lam)].selected_budget.iloc[0])  # the selected budget
        at = d[d.budget == b]  # the runs at it
        within = float(at.groupby("inner").criterion.var(ddof=1).mean())  # the seed part: variance between the two seeds, averaged over inner folds
        between = float(at.groupby("inner").criterion.mean().var(ddof=1))  # the fold part: variance of the inner-fold means
        rows.append({"receiver": rx, "outer": outer, "arm": arm, "lambda": lam, "selected_budget": b, "crit_80k": float(c[80_000]), "crit_90k": float(c[90_000]), "crit_100k": float(c[100_000]), "se_80_100": se, "still_falling": falling, "var_seed": within, "var_fold": between, "min_at_boundary": b in (10_000, 100_000)})  # the row
    return pd.DataFrame(rows)  # the table


def d3_outer_insample(sel: pd.DataFrame, steps: pd.DataFrame) -> pd.DataFrame:  # function: the 750 frozen outer agents' Q margins on their own TRAINING sessions (in-sample): AUROC for post- against pre-change seconds beside z_t's
    rows = []  # one row per agent
    specs = T.specs_outer(sel)  # the outer jobs
    for rx in RXS:  # for loop over the receivers
        for outer in SESSIONS:  # for loop over the outer folds
            mat = E.material(rx, outer)  # the fold's material
            grids = {s: T.session_seconds(steps, rx, s) for s in SESSIONS if s != outer}  # dict comprehension: the training sessions' stream grids
            for sp in [s for s in specs if s["receiver"] == rx and s["outer"] == outer]:  # list comprehension: the fold's 50 agents
                m, _ = T.load_q(sp, T.AGENT_DIR / "models" / T.GRID, mat)  # the training script's loader: the frozen weights in a fresh DQN
                env = E.SpoofingDetectionEnv(mat, sp["arm"], sp["lambda"])  # an environment over the fold's training rows (never stepped)
                post, pre, zp, zn = [], [], [], []  # margins and z_t of post- and pre-change seconds
                for s, sd in grids.items():  # for loop over the training sessions
                    sec, ok, Q, S = stream_q(m, env, sd)  # Q along the decision-mode path
                    mg = (Q[:, 1] - Q[:, 0]).astype(float)  # the margins
                    stp = sd.step.to_numpy(object)  # step types
                    rws = np.array([mat["key"].get((s, int(t)), -1) for t in sec])  # list comprehension: state rows
                    z = np.where(ok, mat["surrogate"][np.maximum(rws, 0), -1], np.nan)  # z_t (the out-of-fold log-odds)
                    post.append(mg[ok & (stp == "post")]); pre.append(mg[ok & (stp == "pre")])  # the margins by type (two statements on one line: both appends)
                    zp.append(z[ok & (stp == "post")]); zn.append(z[ok & (stp == "pre")])  # z_t by type (two statements on one line)
                rows.append({"receiver": rx, "outer": outer, "arm": sp["arm"], "lambda": float(sp["lambda"]), "seed": sp["seed"], "auroc_margin_insample": auroc(np.concatenate(post), np.concatenate(pre)), "auroc_z": auroc(np.concatenate(zp), np.concatenate(zn))})  # the row
            print(f"  D3 in-sample: {rx} {outer} done", flush=True)  # progress
    return pd.DataFrame(rows)  # the table


# ----------------------------------------------------------------------------------------------------------------------
# D2 and D6: readings of the agent experiment's and the RQ4 study's committed tables (no detector is run)
# ----------------------------------------------------------------------------------------------------------------------
def d2() -> tuple:  # function: D2 per cell (receiver x share x arm) and per fold
    pts = rt(T.AGENT_DIR / "evaluation" / "deployable_points.csv")  # the agent experiment's deployable points (P6 as amended)
    pr = rt(T.AGENT_DIR / "evaluation" / "point_runs.csv")  # the held-out runs at the points
    ver = rt(T.AGENT_DIR / "evaluation" / "verdicts.csv")  # the registered verdicts (read, never changed)
    cv = rt(T.AGENT_DIR / "evaluation" / "curves.csv")  # the held-out operating curves (P7, secondary)
    agg = rt(T.AGENT_DIR / "evaluation" / "aggregate.csv")  # P8's aggregates
    sel = rt(GRID12 / "selection.csv")  # the inner decision shares per lambda
    folds, cells = [], []  # per fold; per cell
    for v in ver.itertuples():  # for loop over the 18 registered cells
        det = f"agent_{v.arm}"  # the agent detector
        cmp_ = v.best_baseline  # the P9 comparator
        for outer in SESSIONS:  # for loop over the outer folds
            a = pts[(pts.receiver == v.receiver) & (pts.outer == outer) & (pts.detector == det) & (pts.share == v.share)].iloc[0]  # the agent's point
            c = pts[(pts.receiver == v.receiver) & (pts.outer == outer) & (pts.detector == cmp_) & (pts.share == v.share)].iloc[0]  # the comparator's point
            sh = sel[(sel.receiver == v.receiver) & (sel.outer == outer) & (sel.arm == v.arm)].sort_values("lambda").set_index("lambda").inner_decision_clean_share  # lambda -> inner decision share
            lam = a.param  # the chosen lambda (NaN: no point)
            smaller = sh[sh.index < lam] if np.isfinite(lam) else sh.iloc[:0]  # the lambdas below it
            coarse = bool(np.isfinite(lam) and sh[lam] < 0.5 * v.share and len(smaller) and smaller.iloc[-1] > v.share)  # the pre-registered rule: the chosen share under half the target while the next smaller lambda overshoots
            nonmono = bool(not np.isfinite(lam) and (np.diff(sh.to_numpy()) > 0).any())  # no point, but the share is non-monotone in lambda (a finer grid could fall between)
            folds.append({"receiver": v.receiver, "share": v.share, "arm": v.arm, "outer": outer, "lambda": lam, "agent_inner_share": float(sh[lam]) if np.isfinite(lam) else np.nan, "comparator": cmp_, "comparator_inner_share": float(c.inner_clean_share),  # dict literal: the fold's figures
                          "lost_to_grid": coarse, "no_point_nonmonotone": nonmono, **{f"share_lambda_{int(l)}": float(sh[l]) for l in sh.index}})  # dict comprehension: the lambda-share curve
        ra = pr[(pr.receiver == v.receiver) & (pr.share == v.share) & (pr.detector == det)]  # the agent's held-out runs
        rc = pr[(pr.receiver == v.receiver) & (pr.share == v.share) & (pr.detector == cmp_)]  # the comparator's
        sa = float(ra.dec_clean_in_alarm.sum() / ra.dec_clean.sum()) if len(ra) and ra.dec_clean.sum() else np.nan  # the agent's realised share, pooled over the folds and seeds
        sc = float(rc.dec_clean_in_alarm.sum() / rc.dec_clean.sum()) if len(rc) and rc.dec_clean.sum() else np.nan  # the comparator's
        curve = cv[(cv.receiver == v.receiver) & (cv.detector == cmp_)]  # the comparator's held-out curve
        at = E.read_at(curve.assign(param=curve.param), sa) if np.isfinite(sa) else pd.Series(dtype=float)  # the environment script's reading: the curve point at the largest share at or below the agent's
        ag = agg[(agg.receiver == v.receiver) & (agg.share == v.share) & (agg.detector == det) & (agg.metric == "median delay (s)")]  # the agent's delay IQM and interval
        lo, hi = (float(ag.ci_low.iloc[0]), float(ag.ci_high.iloc[0])) if len(ag) else (np.nan, np.nan)  # conditional expression: the interval
        cd = float(at.median_delay_s) if len(at) else np.nan  # the comparator's delay at the agent's share
        cm = float(at.miss_share) if len(at) else np.nan  # and its miss share
        has = np.isfinite(sa)  # the cell has a point
        art = bool(has and sa < 0.5 * sc and np.isfinite(cd) and lo <= cd <= hi)  # the pre-registered defect rule for the cell
        lim = bool(has and np.isfinite(cd) and (cd < lo or (np.isfinite(cm) and cm < v.agent_miss_rate)))  # the limit reading: still faster or missing fewer at the agent's share
        better = bool(has and np.isfinite(cd) and cd > hi and np.isfinite(cm) and cm >= v.agent_miss_rate)  # beyond the pre-registered rule, reported beside it: at the agent's share the comparator is slower and misses no fewer
        # dict literal over three lines: the cell's figures
        cells.append({"receiver": v.receiver, "share": v.share, "arm": v.arm, "comparator": cmp_, "verdict_registered": v.verdict, "agent_realised_share": sa, "comparator_realised_share": sc,
                      "agent_not_held": bool(has and sa > 2 * v.share), "comparator_not_held": bool(np.isfinite(sc) and sc > 2 * v.share), "agent_delay_iqm_lo": lo, "agent_delay_iqm_hi": hi, "agent_miss_rate": v.agent_miss_rate,  # dict literal continued
                      "comparator_delay_at_agent_share": cd, "comparator_miss_at_agent_share": cm, "operating_point_artefact": art, "still_beaten_at_equal_share": lim, "agent_better_at_equal_share": better, "has_point": has})  # dict literal continued
    return pd.DataFrame(cells), pd.DataFrame(folds)  # both tables


def d6(steps: pd.DataFrame) -> tuple:  # function: D6's secondary reading of the agent experiment's held-out intervals (agent_marks.csv), descriptive only
    mk = rt(E.PROC / "rq4" / "agent_marks.csv")  # the 618 interval-runs with both onset marks
    ver = rt(T.AGENT_DIR / "evaluation" / "verdicts.csv")  # the registered comparators
    agg = rt(T.AGENT_DIR / "evaluation" / "aggregate.csv")  # P8's aggregates (early warning)
    lab = steps.set_index(["receiver", "session", "sec"]).label  # the scoring label of every second (label data, read only)
    mk["counted"] = mk.detected & ~mk.no_detection_delay  # a counted delay: detected and neither in alarm nor refractory at onset
    ev_sec = mk.onset - mk.last_event_age_s  # the second of the last stopping-mode event before onset
    mk["event_label"] = [lab.get((r, o, int(t)), "") if np.isfinite(t) else "" for r, o, t in zip(mk.receiver, mk.outer, ev_sec)]  # list comprehension: that event's scoring label
    mk["refractory_uncertain"] = mk.refractory_at_onset & (mk.event_label == "uncertain")  # H2's mark on the held-out intervals
    pts = rt(T.AGENT_DIR / "evaluation" / "deployable_points.csv")  # the agent experiment's points: the agent's lambda per fold and share (its runs are stored by lambda, not by share)
    pairs, cells = [], []  # per pair; per cell
    for v in ver.itertuples():  # for loop over the 18 registered cells
        lam = pts[(pts.receiver == v.receiver) & (pts.share == v.share) & (pts.detector == f"agent_{v.arm}")].set_index("outer").param  # outer fold -> the agent's lambda at this share (NaN: no point)
        a = mk[(mk.receiver == v.receiver) & (mk.detector == f"agent_{v.arm}") & (mk["lambda"] == mk.outer.map(lam))]  # the agent's interval-runs at its point in each fold
        c = mk[(mk.receiver == v.receiver) & (mk.share == v.share) & (mk.detector == v.best_baseline)]  # the comparator's
        j = a.merge(c[["outer", "onset", "counted", "delay_s"]], on=["outer", "onset"], suffixes=("", "_cmp"))  # paired on the same fold and interval
        jj = j[j.counted & j.counted_cmp]  # both counted
        d_ = (jj.delay_s - jj.delay_s_cmp).to_numpy()  # paired differences, agent minus comparator
        i_, lo, hi, n = B12.strat_boot(d_, jj.outer.to_numpy()) if len(d_) else (np.nan, np.nan, np.nan, 0)  # the evaluation script's IQM with the stratified interval
        for r in jj.itertuples():  # the pairs, kept
            pairs.append({"receiver": v.receiver, "share": v.share, "arm": v.arm, "comparator": v.best_baseline, "outer": r.outer, "seed": r.seed, "onset": r.onset, "agent_delay_s": r.delay_s, "comparator_delay_s": r.delay_s_cmp})  # one row
        cells.append({"receiver": v.receiver, "share": v.share, "arm": v.arm, "comparator": v.best_baseline, "verdict_registered": v.verdict, "pairs_counted_both": int(n), "paired_runs_registered": v.paired_runs,  # dict literal: the cell
                      "diff_iqm_s": i_, "diff_lo": lo, "diff_hi": hi, "too_few": bool(n < 5)})  # dict literal continued
    mk["in_alarm"] = mk.onset_state == "in alarm"  # the in-alarm mark
    mk["refractory_only"] = mk.refractory_at_onset & ~mk.in_alarm  # the refractory mark where the first is absent (the two overlap)
    mk["missed"] = ~mk.detected & ~mk.no_detection_delay  # no event inside the interval and no mark
    marks = mk.groupby("detector").agg(in_alarm_n=("in_alarm", "sum"), refractory_only=("refractory_only", "sum"), missed=("missed", "sum"), interval_runs=("onset", "size"), counted=("counted", "sum"), in_alarm=("onset_state", lambda s: int((s == "in alarm").sum())), refractory=("refractory_at_onset", "sum"), refractory_uncertain=("refractory_uncertain", "sum"), no_delay=("no_detection_delay", "sum")).reset_index()  # per detector: the marks
    ew = agg[agg.metric == "early warning"][["receiver", "share", "detector", "iqm", "ci_low", "ci_high", "runs_defined"]]  # each detector's early warning at the points (P8's)
    per_share = []  # interval-runs and counted runs per detector and share, counted as the RQ4 study counts them (an agent run serves every share whose point is its lambda)
    for s_ in T.SHARES:  # for loop over the shares
        for det in sorted(mk.detector.unique()):  # for loop over the detectors
            if det.startswith("agent"):  # an agent: its runs at the share's point lambda in each fold
                rows_ = []  # its rows at this share
                for rx in RXS:  # for loop over the receivers
                    lam = pts[(pts.receiver == rx) & (pts.share == s_) & (pts.detector == det)].set_index("outer").param  # outer -> lambda
                    rows_.append(mk[(mk.receiver == rx) & (mk.detector == det) & (mk["lambda"] == mk.outer.map(lam))])  # kept
                x = pd.concat(rows_)  # all receivers
            else:  # a baseline: its rows at the share
                x = mk[(mk.detector == det) & (mk.share == s_)]  # filtered
            per_share.append({"detector": det, "share": s_, "interval_runs": len(x), "counted": int(x.counted.sum())})  # the row
    return pd.DataFrame(cells), pd.DataFrame(pairs), marks, ew, pd.DataFrame(per_share)  # the tables


# ----------------------------------------------------------------------------------------------------------------------
# D5: the reward and the stream re-derived by hand on episodes chosen by the pre-registered rule
# ----------------------------------------------------------------------------------------------------------------------
D5_CELL = {"stage": "inner", "receiver": "SBF_02", "outer": "2.6.2", "inner": "2.6.4", "arm": "surrogate", "lambda": 100.0, "seed": 20260914, "budget": T.SCHEDULE}  # dict: the pre-registered cell (the agent experiment's pilot cell)


def hand_return(ty: np.ndarray, a: np.ndarray, kind: str, lam: float, m: float) -> tuple:  # function: the reward written out step by step (neither the training script's episode_cost nor the environment script's step is called); (return, steps taken, outcome)
    total = 0.0  # the undiscounted return
    for i in range(len(ty)):  # for loop over the steps in order
        if ty[i] == 1:  # a post-change second (the environment script's code 1)
            if a[i] == 1:  # an alarm after the change ends the episode at no cost
                return total, i + 1, "detection"  # done
            total -= 1.0  # a silent post-change second costs 1
            if kind == "onset" and i == len(ty) - 1:  # the interval's last second passes in silence
                return total - m, i + 1, "miss"  # the miss penalty m on top
        elif ty[i] == 0 and a[i] == 1:  # an alarm before the change (the environment script's code 0)
            return total - lam, i + 1, "false_alarm"  # lambda, and the episode ends
        # an uncertain second (code 2), alarmed or not, and a silent pre-change second: no reward, the episode goes on
    return total, len(ty), "truncated" if kind == "clean" else "end"  # a clean episode that reached its end in silence


def env_return(env: E.SpoofingDetectionEnv, ep: dict, a: np.ndarray) -> tuple:  # function: the same episode replayed through the environment script's step(); (return, steps taken, last event)
    env.reset(options={"kind": ep["kind"], "session": ep["session"], "start": ep["start"], "onset": ep["onset"]})  # the environment script's replay of a given episode
    total, i, ev = 0.0, 0, ""  # tuple assignment: return, steps, last event
    while True:  # while loop until the episode ends
        _, r, term, trunc, info = env.step(int(a[i]))  # one step with the given action
        total, i, ev = total + r, i + 1, info["event"]  # tuple assignment: accumulated
        if term or trunc:  # the end
            return total, i, ev  # done


def d5(R: pd.DataFrame) -> pd.DataFrame:  # function: D5's hand derivations
    rows = []  # one row per item
    J = load_job(T.job_id(D5_CELL))  # the regenerated agent's arrays
    train, evm = T.job_material(D5_CELL, STATES12)  # the training script's materials of the cell
    eenv = E.SpoofingDetectionEnv(evm, "surrogate", D5_CELL["lambda"])  # the inner held-out environment
    lam, m = D5_CELL["lambda"], evm["m"]  # lambda and the inner miss penalty
    spec, acts = J["ep_ho"], np.unpackbits(J["ep_ho_acts"])  # the stored episodes and their packed actions
    pos = np.r_[0, np.cumsum(spec[:, 5].astype(int))]  # np.r_ concatenation: each episode's first position among the actions
    pick = None  # the chosen episode
    for k, r in enumerate(spec):  # for loop over the episodes in stored order: the pre-registered rule
        ep = eenv.episode("onset" if r[1] == 0 else "clean", SESSIONS[int(r[2])], int(r[3]), int(r[4]))  # the environment script's episode
        a = acts[pos[k]:pos[k + 1]]  # its greedy actions (along the never-alarm path, as the agent sees them)
        ty = evm["type"][ep["rows"]]  # step types
        end = hand_return(ty, a, ep["kind"], lam, m)[1]  # steps the episode actually takes
        masked = bool(((ty[:end] == 2) & (a[:end] == 1)).any())  # an alarm in an uncertain second before it ends
        if int(r[7]) == OUTCOME_CODE["detection"] and (masked or pick is None):  # a detection; with a masked alarm preferred
            pick = (k, ep, a, ty, masked)  # kept
            if masked:  # the rule's first choice found
                break  # stop looking
    k, ep, a, ty, masked = pick  # tuple unpacking: the episode
    h, ns, oc = hand_return(ty, a, ep["kind"], lam, m)  # by hand
    g, ne, ev = env_return(eenv, ep, a)  # through the environment script's step
    c = T.episode_cost(ty, a, ep["kind"], lam, m)[0]  # the training script's closed form (a cost: minus the return)
    rows.append({"item": "agent, inner held-out episode", "episode": k, "kind": ep["kind"], "masked_uncertain_alarm": masked, "outcome": oc, "steps": ns, "hand_return": h, "env_return": g, "closed_form_return": -c, "stored_cost": float(spec[k, 6]), "max_abs_diff": max(abs(h - g), abs(h + c), abs(h + spec[k, 6]))})  # the row
    S = page_path(evm["surrogate"][ep["rows"], -1])  # S along the episode from 0
    run = R[(R.receiver == "SBF_02") & (R.outer == "2.6.2") & (R.inner == "2.6.4") & (R.arm == "surrogate") & (R["lambda"] == lam) & (R.seed == D5_CELL["seed"])].iloc[0]  # the run's D1 row (h_q, h*)
    for nm, hh in (("CUSUM at h_q", run.h_q), ("CUSUM at h*", run.h_star)):  # for loop over the two CUSUM thresholds
        ac = (S >= hh).astype(int)  # CUSUM's actions on this episode
        h2_, _, oc2 = hand_return(ty, ac, ep["kind"], lam, m)  # by hand
        g2, _, _ = env_return(E.SpoofingDetectionEnv(evm, "surrogate", lam), ep, ac)  # through the environment script's step
        c2 = T.episode_cost(ty, ac, ep["kind"], lam, m)[0]  # the closed form
        vec = float(cusum_costs([(ty, S, ep["kind"])], lam, m)[0, int(np.argmin(np.abs(H_EP - hh)))]) if np.isfinite(hh) and np.isin(hh, H_EP) else np.nan  # the vectorised cost of D1(b) where h lies on its grid
        rows.append({"item": nm, "episode": k, "kind": ep["kind"], "h": float(hh), "outcome": oc2, "hand_return": h2_, "env_return": g2, "closed_form_return": -c2, "vectorised_return": -vec if np.isfinite(vec) else np.nan, "max_abs_diff": max(abs(h2_ - g2), abs(h2_ + c2), abs(h2_ + vec) if np.isfinite(vec) else 0.0)})  # the row
    log = T.read_gz_json(JOBS12 / f"{T.job_id(D5_CELL)}.json.gz")  # the agent experiment's training log of the job
    maps = rowmaps(E.material("SBF_02", "2.6.2"))  # second -> row
    want = {"false_alarm": None, "miss": None, "truncated": None}  # the first of each outcome in the log
    for e in log["episodes"]:  # for loop over the logged episodes in order
        rws = maps[e["session"]][1][e["start"] - maps[e["session"]][0] + np.arange(e["steps"])]  # its rows
        a_ = np.unpackbits(np.frombuffer(bytes.fromhex(e["actions_hex"]), np.uint8))[:e["steps"]]  # its actions
        hr, _, oc3 = hand_return(train["type"][rws], a_, e["kind"], lam, log["m"])  # by hand (training uses the inner m)
        key = "truncated" if not e["terminated"] else oc3  # a truncated clean episode, else the outcome
        if key in want and want[key] is None:  # the first of its kind
            want[key] = {"item": f"training log, first {key}", "kind": e["kind"], "outcome": oc3, "steps": e["steps"], "hand_return": hr, "monitor_return_x100": e["monitor_return"] * T.REWARD_SCALE, "max_abs_diff": abs(hr / T.REWARD_SCALE - e["monitor_return"])}  # the row (the difference in the agent's scaled units, as the agent experiment's T1)
    rows += [w for w in want.values() if w is not None]  # list comprehension: the rows found
    v_sec = J["ho_sec"]  # the inner held-out stream
    ev = J["ho_stop"] > 0  # the agent's stopping-mode events
    gaps = np.diff(v_sec[ev])  # seconds between successive events
    lab = T.session_seconds(T.read_steps(), "SBF_02", "2.6.4").label.to_numpy(object)  # scoring labels
    st = E.stream(v_sec, lab, ev, "2.6.4", "stopping")  # the environment script's stream figures
    for (a0, b0), L in zip(E.SC.pooled_intervals("2.6.4"), st["latency"]):  # for loop over the intervals with the stream's delays
        f = v_sec[ev & (v_sec >= a0) & (v_sec <= b0)]  # the events inside, by hand
        hd = float(f.min() - a0) if f.size else np.nan  # the hand delay
        rf = refractory_from(v_sec, ev, lab, a0)  # the refractory mark at onset and its event's label
        rows.append({"item": "stream, interval", "onset": a0, "hand_delay_s": hd, "stream_delay_s": L["delay_s"], "refractory_at_onset": rf[0], "event_age_s": rf[1], "event_label": rf[2], "max_abs_diff": 0.0 if (np.isnan(hd) and np.isnan(L["delay_s"])) else abs(hd - L["delay_s"])})  # the row
    rows.append({"item": "stream, refractory", "events": int(ev.sum()), "min_gap_s": float(gaps.min()) if len(gaps) else np.nan, "uncertain_events": int((ev & (lab == "uncertain")).sum()), "max_abs_diff": 0.0 if (not len(gaps) or gaps.min() > E.REFRACTORY) else 1.0})  # every event more than 30 s after the last
    return pd.DataFrame(rows)  # the table


# ----------------------------------------------------------------------------------------------------------------------
# verdicts, checks and figures
# ----------------------------------------------------------------------------------------------------------------------
def verdicts(D1C, H2C, D2C, D3C, BUD, D4C, D5T) -> pd.DataFrame:  # function: the pre-registered defect/limit verdict of every diagnostic, by its pre-registered thresholds
    rows = []  # one row per verdict
    for arm in ARMS:  # D1(b) per arm (decisive for the surrogate only)
        d = D1C[D1C.arm == arm]  # its 15 cells
        nf, nb = int(d.optimisation_failure.sum()), int(d.train_at_or_below.sum())  # cells failing; cells at or below CUSUM on training episodes
        v = "DEFECT (optimisation failure)" if nf >= CELL_MIN else ("LIMIT" if nb >= CELL_MIN else "MIXED")  # nested conditional expression: the pre-registered rule
        rows.append({"diagnostic": "D1(b) optimisation test", "arm": arm, "cells_meeting_defect": nf, "cells": len(d), "verdict": v if arm == "surrogate" else v + " (descriptive: not decisive for the primary arm)", "detail": f"{nb} of {len(d)} cells at or below CUSUM(h*) on the training episodes; {int(d.heldout_worse.sum())} worse on the inner held-out episodes; training-mixture weighting: agent above CUSUM in {int((d.train_diff_mix_median > 0).sum())} of {len(d)}"})  # the row
    d = D1C[D1C.arm == "surrogate"]  # H1's cells
    n1 = int(d.h1_cell.sum())  # cells meeting the interface rule
    rows.append({"diagnostic": "D1(c) H1, S_t beyond the training range", "arm": "surrogate", "cells_meeting_defect": n1, "cells": len(d), "verdict": "DEFECT (evaluation interface)" if n1 >= CELL_MIN else "no defect", "detail": f"median share of inner clean seconds beyond the training S range {d.clean_share_beyond_train_S_median.median():.3f}"})  # the row
    for arm in ARMS:  # H2 per arm
        d = H2C[H2C.arm == arm]  # its cells
        n2 = int(d.h2_cell.sum())  # cells where the agent exceeds the best baseline by more than 0.20
        rows.append({"diagnostic": "H2 refractory from an uncertain alarm", "arm": arm, "cells_meeting_defect": n2, "cells": len(d), "verdict": "DEFECT (training/deployment mismatch)" if n2 >= CELL_MIN else "LIMIT", "detail": f"median excess {d.excess.median():.3f}"})  # the row
    d = D2C[D2C.has_point]  # D2's cells with a point
    na, nl = int(d.operating_point_artefact.sum()), int(d.still_beaten_at_equal_share.sum())  # cells meeting each reading
    rows.append({"diagnostic": "D2 operating-point matching", "arm": "both", "cells_meeting_defect": na, "cells": len(d), "verdict": "DEFECT (operating-point artefact)" if na >= 9 else ("LIMIT" if nl >= 9 else "MIXED"), "detail": f"{nl} of {len(d)} cells still beaten at the agent's own share; agent point not held in {int(d.agent_not_held.sum())}"})  # the row
    d = D3C[D3C.arm == "surrogate"]  # D3's surrogate cells
    n3, nfall = int(d.margin_below_z.sum()), int(BUD.still_falling.sum())  # cells where the margin ranks worse than z_t; cells still falling at 100k
    v3 = "DEFECT (representation)" if n3 >= CELL_MIN else ("DEFECT (training too short)" if nfall >= len(BUD) / 3 else "LIMIT")  # nested conditional expression: the pre-registered rule
    rows.append({"diagnostic": "D3 learning", "arm": "surrogate (margin); all (budget)", "cells_meeting_defect": n3, "cells": len(d), "verdict": v3, "detail": f"criterion still falling at 100k in {nfall} of {len(BUD)} cells; selected budget at a boundary in {int(BUD.min_at_boundary.sum())}"})  # the row
    for arm in ARMS:  # D4 per arm
        d = D4C[D4C.arm == arm]  # its cells
        n4 = int(d.d4_cell.sum())  # cells meeting the sampling rule
        rows.append({"diagnostic": "D4 data budget", "arm": arm, "cells_meeting_defect": n4, "cells": len(d), "verdict": "DEFECT (sampling design)" if n4 >= CELL_MIN else "LIMIT", "detail": f"cells under 1 post-change transition per minibatch {int((d.post_per_batch_h2 < 1).sum())}, under 1 % detections {int((d.detection_share_h2 < 0.01).sum())}; median post-change transitions per minibatch (second half) {d.post_per_batch_h2.median():.2f}, median detection share {d.detection_share_h2.median():.4f}; distinct onsets {d.distinct_onsets.median():.0f}"})  # the row
    bad = int((D5T.max_abs_diff > 1e-12).sum())  # D5 items differing from the registered reward or rules
    rows.append({"diagnostic": "D5 reward bookkeeping", "arm": "surrogate (one cell)", "cells_meeting_defect": bad, "cells": len(D5T), "verdict": "DEFECT (code fault)" if bad else "no fault", "detail": f"largest difference {D5T.max_abs_diff.max():.2e}"})  # the row
    rows.append({"diagnostic": "D6 secondary reading", "arm": "both", "cells_meeting_defect": np.nan, "cells": np.nan, "verdict": "descriptive (no verdict; sizes H2)", "detail": "registered verdicts unchanged"})  # the row
    return pd.DataFrame(rows)  # the table


def figures(R, O, H2C, D1C, D2C, D2F, D3C, MG, D4C, D6M) -> None:  # function: the figures
    import matplotlib.pyplot as plt  # pyplot (09 chose the file-only backend)
    NAME = E.NAME  # receiver names
    RCOL = {"SBF_02": "#0072B2", "SBF_03": "#D55E00", "UBX_01": "#009E73"}  # dict: receiver -> colour (Okabe-Ito)
    AMK = {"primary": "D", "surrogate": "o"}  # dict: arm -> marker
    # the figure: one inner held-out stream, the raw trace behind the D1 and H2 numbers
    J = load_job(T.job_id(D5_CELL))  # the regenerated agent
    steps = T.read_steps()  # labels
    ctx = inner_ctx("SBF_02", "2.6.2", steps, rt(GRID12 / "nested_predictions.csv.gz"))  # the fold's context
    v = ctx["per"]["2.6.4"]  # the stream
    run = R[(R.receiver == "SBF_02") & (R.outer == "2.6.2") & (R.inner == "2.6.4") & (R.arm == "surrogate") & (R["lambda"] == 100.0) & (R.seed == D5_CELL["seed"])].iloc[0]  # its D1 row
    b = baseline_runs(v, run.h_q, run.t_q)  # CUSUM and the regression at the agent's share
    t = (v["sec"] - v["sec"][0]) / 60.0  # minutes from the stream's start
    fig, ax = plt.subplots(4, 1, figsize=(11, 7.5), sharex=True, gridspec_kw={"height_ratios": [0.6, 2, 1, 1]})  # four stacked panels
    lc = {"clean": "#009E73", "uncertain": "#BBBBBB", "attack_spoof": "#CC79A7", "attack_jam_spoof": "#D55E00"}  # dict: scoring label -> colour
    for lb, c in lc.items():  # for loop over the labels: a coloured band
        k = v["lab"] == lb  # its seconds
        ax[0].scatter(t[k], np.zeros(k.sum()), c=c, marker="|", s=200, label=lb)  # ticks along one line
    ax[0].set_yticks([]); ax[0].legend(fontsize=7, ncol=4, loc="upper right", frameon=False); ax[0].set_title("labels (scoring)", fontsize=9, loc="left")  # three statements: axis, legend, title
    ax[1].plot(t, np.where(v["ok"], J["ho_S"], np.nan), color="#0072B2", lw=1.2, label="S_t on the stream (decision mode, restarts only at gaps)")  # the agent's S
    ax[1].axhline(run.h_q, color="#009E73", ls="--", lw=1, label=f"CUSUM h_q = {run.h_q:.1f} (agent's inner share {run.q:.3f})")  # the matched threshold
    if np.isfinite(run.get("train_S_max", np.nan)):  # if: the training range is known
        ax[1].axhline(run.train_S_max, color="#CC79A7", ls=":", lw=1.2, label=f"largest S_t in the agent's training episodes = {run.train_S_max:.1f}")  # H1's line
    ax[1].set_yscale("symlog", linthresh=10); ax[1].set_ylim(bottom=0); ax[1].set_ylabel("S_t (symmetric log scale)"); ax[1].legend(fontsize=7, loc="upper left", frameon=False)  # four statements: scale, floor at 0 (S is never negative), label, legend
    ax[2].step(t, J["ho_dec"], where="post", color="#56B4E9", lw=1, label="agent, decision mode"); ax[2].step(t, b["cusum"][0].astype(int) * 0.9, where="post", color="#009E73", lw=1, label="CUSUM at h_q, decision mode (x 0.9)")  # two statements: decision-mode alarms
    ax[2].set_yticks([0, 1]); ax[2].legend(fontsize=7, loc="upper left", frameon=False)  # two statements
    ax[3].vlines(t[J["ho_stop"] > 0], 0.5, 1.0, color="#56B4E9", lw=1.5, label="agent events"); ax[3].vlines(t[b["cusum"][1]], 0.0, 0.5, color="#009E73", lw=1.5, label="CUSUM events")  # two statements: stopping-mode events
    for a0, b0 in v["pooled"]:  # for loop over the intervals: shaded on every panel
        for x_ in ax:  # every panel
            x_.axvspan((a0 - v["sec"][0]) / 60, (b0 - v["sec"][0]) / 60, color="#CC79A7", alpha=0.08)  # the interval
    ax[3].set_yticks([]); ax[3].set_xlabel("minutes from the first windowed second of 2.6.4"); ax[3].legend(fontsize=7, loc="upper left", frameon=False)  # three statements
    fig.suptitle("Inner stream of one agent: mosaic-X5, outer fold 2.6.2, inner held-out 2.6.4: surrogate agent (lambda 100, seed 20260914) against CUSUM at its share", fontsize=9)  # title
    fig.tight_layout(); E.M.save_fig(fig, GRAPH / "diagnosis_inner_stream.png"); plt.close(fig)  # three statements: layout, byte-stable save, close
    # the figure: D1(b), agent minus CUSUM(h*) on the training and the inner held-out episodes
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=False)  # two panels
    for k_, (col, lo, hi, ttl) in enumerate((("train_diff_median", "train_diff_lo", "train_diff_hi", "training episodes (optimisation)"), ("heldout_diff_median", "heldout_diff_lo", "heldout_diff_hi", "inner held-out episodes (generalisation)"))):  # for loop over the two panels
        for (arm, rx), d in D1C.groupby(["arm", "receiver"], sort=True):  # for loop over arm x receiver
            x = np.log10(d["lambda"].to_numpy()) + {"SBF_02": -0.06, "SBF_03": 0.0, "UBX_01": 0.06}[rx] + (0.02 if arm == "surrogate" else -0.02)  # dodged log-lambda positions
            ax[k_].errorbar(x, d[col], yerr=[d[col] - d[lo], d[hi] - d[col]], fmt=AMK[arm], color=RCOL[rx], ms=5, lw=1, mfc="none" if arm == "primary" else RCOL[rx], label=f"{NAME[rx]}, {arm}")  # median and interval
        ax[k_].axhline(0, color="black", lw=0.8); ax[k_].set_xticks(np.log10(LAMBDAS)); ax[k_].set_xticklabels([str(l) for l in LAMBDAS]); ax[k_].set_xlabel("lambda"); ax[k_].set_title(ttl, fontsize=9, loc="left")  # five statements: axes
        ax[k_].set_yscale("symlog", linthresh=10)  # a symmetric log scale: costs span 1 to hundreds
    ax[0].set_ylabel("agent minus CUSUM(h*), mean cost per episode"); ax[1].legend(fontsize=6.5, ncol=2, frameon=False)  # two statements
    fig.suptitle("Diagnosis D1: the surrogate's input contains S_t, so CUSUM(h*) is a policy it can represent; above 0 = the agent pays more", fontsize=9)  # title
    fig.tight_layout(); E.M.save_fig(fig, GRAPH / "diagnosis_d1_cusum_test.png"); plt.close(fig)  # three statements
    # the figure: D2, realised shares at the points and the inner lambda-share curves
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))  # two panels
    d = D2C[D2C.has_point]  # the cells with a point
    for r in d.itertuples():  # for loop over the cells
        ax[0].scatter(r.comparator_realised_share, r.agent_realised_share, marker=AMK[r.arm], color=RCOL[r.receiver], s=40, facecolors="none" if r.arm == "primary" else RCOL[r.receiver])  # one cell
    lim = [0.003, 0.6]  # axis limits
    ax[0].plot(lim, lim, color="black", lw=0.8); ax[0].plot(lim, [x / 2 for x in lim], color="grey", lw=0.8, ls=":")  # two statements: equality and half
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlim(lim); ax[0].set_ylim(lim); ax[0].set_xlabel("comparator's realised held-out clean share"); ax[0].set_ylabel("agent's realised held-out clean share")  # six statements
    ax[0].set_title("points (filled: surrogate; open: primary; dotted: half the comparator)", fontsize=8, loc="left")  # title
    for (rx, arm), g in D2F.drop_duplicates(["receiver", "arm", "outer"]).groupby(["receiver", "arm"], sort=True):  # for loop over receiver x arm: the inner lambda-share curve, median over folds
        y = [g[f"share_lambda_{l}"].median() for l in LAMBDAS]  # list comprehension: the median share per lambda
        ax[1].plot(LAMBDAS, y, marker=AMK[arm], color=RCOL[rx], mfc="none" if arm == "primary" else RCOL[rx], lw=1.2, label=f"{NAME[rx]}, {arm}")  # the curve
    for s_ in T.SHARES:  # for loop over the registered shares
        ax[1].axhline(s_, color="grey", lw=0.6, ls="--")  # the target
    ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("lambda"); ax[1].set_ylabel("inner decision-mode clean share (median over folds)"); ax[1].legend(fontsize=6.5, frameon=False)  # five statements
    fig.suptitle("Diagnosis D2: where the agent's points sit against its comparator's, and how its inner share moves with lambda", fontsize=9)  # title
    fig.tight_layout(); E.M.save_fig(fig, GRAPH / "diagnosis_d2_shares.png"); plt.close(fig)  # three statements
    # the figure: D3, the margin's ranking against z_t's and the greedy action by step category
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))  # two panels
    for r in D3C.itertuples():  # for loop over the cells
        ax[0].scatter(r.auroc_z_median, r.auroc_margin_median, marker=AMK[r.arm], color=RCOL[r.receiver], s=35, facecolors="none" if r.arm == "primary" else RCOL[r.receiver])  # one cell
    ax[0].plot([0.3, 1], [0.3, 1], color="black", lw=0.8); ax[0].plot([0.35, 1], [0.3, 0.95], color="grey", lw=0.8, ls=":")  # two statements: equality and 0.05 below
    ax[0].set_xlabel("AUROC of z_t (post- against pre-change, inner held-out)"); ax[0].set_ylabel("AUROC of the Q margin, same seconds"); ax[0].set_title("cells (receiver x lambda); dotted: 0.05 below", fontsize=8, loc="left")  # three statements
    cats = ["pre-change", "uncertain, other", "uncertain, 60 s before onset", "post-change, 0-29 s", "post-change, 30 s on"]  # list: the categories in order
    for j_, arm in enumerate(ARMS):  # for loop over the arms
        y = [MG[MG.arm == arm][f"share_positive[{c_}]"].median() for c_ in cats]  # list comprehension: the median share of seconds where the greedy action is an alarm
        ax[1].bar(np.arange(len(cats)) + (j_ - 0.5) * 0.38, y, width=0.38, color=["#CC79A7", "#56B4E9"][j_], label=arm)  # bars
    ax[1].set_xticks(np.arange(len(cats))); ax[1].set_xticklabels([c_.replace(", ", ",\n") for c_ in cats], fontsize=7); ax[1].set_ylabel("share of seconds with Q(alarm) > Q(continue)"); ax[1].legend(fontsize=7, frameon=False)  # four statements
    fig.suptitle("Diagnosis D3: does the value function keep what its inputs carry, and where does the greedy policy alarm", fontsize=9)  # title
    fig.tight_layout(); E.M.save_fig(fig, GRAPH / "diagnosis_d3_margins.png"); plt.close(fig)  # three statements
    # the figure: D4, post-change transitions per minibatch in the second half of training
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))  # one panel
    for j_, ((arm, rx), d) in enumerate(D4C.groupby(["arm", "receiver"], sort=True)):  # for loop over arm x receiver
        ax.plot(d["lambda"], d.post_per_batch_h2, marker=AMK[arm], color=RCOL[rx], mfc="none" if arm == "primary" else RCOL[rx], lw=1.2, label=f"{NAME[rx]}, {arm}")  # the curve
    ax.axhline(1, color="black", lw=0.8, ls="--"); ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("lambda"); ax.set_ylabel("expected post-change transitions per minibatch of 64"); ax.legend(fontsize=7, frameon=False)  # six statements
    fig.suptitle("Diagnosis D4: what the updates see (outer agents, median over 25 runs per cell; dashed: the defect line)", fontsize=9)  # title
    fig.tight_layout(); E.M.save_fig(fig, GRAPH / "diagnosis_d4_buffer.png"); plt.close(fig)  # three statements
    # the figure: H2 on the inner folds, and the held-out marks per detector 
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))  # two panels
    for r in H2C.itertuples():  # for loop over the cells
        ax[0].scatter(r.best_baseline_share, r.agent_share, marker=AMK[r.arm], color=RCOL[r.receiver], s=35, facecolors="none" if r.arm == "primary" else RCOL[r.receiver])  # one cell
    ax[0].plot([0, 1], [0, 1], color="black", lw=0.8); ax[0].plot([0, 0.8], [0.2, 1.0], color="grey", lw=0.8, ls=":")  # two statements: equality and +0.20
    ax[0].set_xlim(-0.02, 1); ax[0].set_ylim(-0.02, 1); ax[0].set_xlabel("best baseline (CUSUM or regression at the agent's share)"); ax[0].set_ylabel("agent"); ax[0].set_title("H2: share of inner onsets refractory from an uncertain-second event", fontsize=8, loc="left")  # five statements
    dm = D6M.set_index("detector")  # the marks per detector
    order = ["lr", "rf", "cusum", "kofn", "agent_primary", "agent_surrogate"]  # list: the detectors in the evaluation script's order
    n_ = dm.loc[order].interval_runs.to_numpy(float)  # interval-runs per detector
    parts = [("counted", "#009E73"), ("in_alarm_n", "#D55E00"), ("refractory_only", "#CC79A7"), ("missed", "#999999")]  # list of (column, colour): mutually exclusive parts (a detected run that is neither counted nor marked cannot occur)
    bot = np.zeros(len(order))  # stacking base
    for col, c in parts:  # for loop over the parts
        y = dm.loc[order][col].to_numpy(float) / n_  # the share
        ax[1].bar(range(len(order)), y, bottom=bot, color=c, label={"counted": "counted delay", "in_alarm_n": "in alarm at onset", "refractory_only": "refractory at onset only", "missed": "missed"}[col])  # stacked bar
        bot += y  # raised
    ax[1].set_xticks(range(len(order))); ax[1].set_xticklabels([B12.STYLE[o][2] for o in order], rotation=20, fontsize=7); ax[1].set_ylabel("share of held-out interval-runs at the points"); ax[1].legend(fontsize=7, frameon=False)  # four statements
    ax[1].set_title("D6 (the agent experiment held-out intervals at the points, read only)", fontsize=8, loc="left")  # title
    fig.suptitle("Diagnosis H2 and D6: alarms raised in the uncertain stretch before onset", fontsize=9)  # title
    fig.tight_layout(); E.M.save_fig(fig, GRAPH / "diagnosis_h2_d6.png"); plt.close(fig)  # three statements


def guard_job(J: dict, outer: str, inner: str) -> bool:  # function: C3, a regenerated job holds no outer held-out second and no second of a session outside the fold (2.6.12 has no code at all)
    tr = set(SESSIONS[int(c)] for c in np.unique(J["tr_sess"]))  # set comprehension of a generator: the stored training sessions
    return outer not in tr and inner not in tr and tr <= set(SESSIONS) - {outer, inner} and len(tr) == 3  # the three inner training sessions exactly


def diag() -> None:  # function: --diag, D1-D6 and H2, the checks, the figures and the verdicts
    if torch.get_num_threads() != 1:  # C8: one thread
        raise SystemExit("STOP: torch is not on one thread")  # stop
    man = rt(DIAG_DIR / "diagnosis_regen_manifest.csv")  # the regeneration manifest
    check("C1 every inner agent regenerated and gated", bool(man.exists.all() and len(man) == 1200 and all(T.sha_file(QDIR / f) == h for f, h in zip(man.file, man.sha256))), f"{int(man.exists.sum())} of 1200 files present, digests equal to the manifest")  # C1
    sel, cur, steps = rt(GRID12 / "selection.csv"), rt(GRID12 / "inner_curve.csv.gz"), T.read_steps()  # tuple assignment: selection, inner curve, step table
    c2, c3, n2 = True, True, 0  # C2 and C3 flags; seconds compared
    for j in regen_specs():  # for loop over the jobs: C2 and C3
        sp = j["spec"]  # its specification
        J = load_job(T.job_id(sp))  # its arrays
        ok = J["ho_ok"].astype(bool)  # seconds with a state
        c2 &= bool(np.array_equal(np.where(ok, np.argmax(np.nan_to_num(J["ho_q"]), axis=1), 0), J["ho_dec"]))  # the stored Q values give the greedy decisions bitwise
        n2 += int(ok.sum())  # counted
        c3 &= guard_job(J, sp["outer"], sp["inner"])  # no outer held-out or foreign session
    check("C2 stored Q values reproduce the greedy decisions", c2, f"{n2} inner held-out seconds over 1,200 agents")  # C2
    J0 = load_job(T.job_id(D5_CELL))  # one job for the negative test
    planted = {**J0, "tr_sess": np.r_[J0["tr_sess"], SESS_CODE[D5_CELL["outer"]]]}  # dict unpacking with a replacement: a planted outer held-out second
    check("C3 no outer held-out session in any inner agent's stored seconds; a planted one is caught", c3 and not guard_job(planted, D5_CELL["outer"], D5_CELL["inner"]), "1,200 jobs guarded; planted outer held-out second caught")  # C3
    R, O, MG = d1_h2()  # D1, H2 and the out-of-sample margins
    E.write_csv(R, DIAG_DIR / "diagnosis_optimisation_runs.csv"); E.write_gz(O, DIAG_DIR / "diagnosis_refractory_onsets.csv.gz"); E.write_csv(MG, DIAG_DIR / "diagnosis_learning_margins.csv")  # three statements: written
    D1C, H2C = d1_cells(R), h2_cells(O)  # tuple assignment: the cells
    E.write_csv(D1C, DIAG_DIR / "diagnosis_optimisation_cells.csv"); E.write_csv(H2C, DIAG_DIR / "diagnosis_refractory_cells.csv")  # two statements: written
    qq = R.groupby(["receiver", "outer", "arm", "lambda"]).agg(a=("agent_decision_clean_alarms", "sum"), c=("clean_seconds", "sum"), q=("q", "first")).reset_index()  # the agent's pooled inner decision share recomputed from the stored decisions
    eq = np.where(qq.c > 0, qq.a / qq.c.where(qq.c > 0, 1), np.nan)  # the recomputed share (NaN without clean seconds)
    check("C4b the stored decisions reproduce P6's inner decision-mode clean share", bool(np.allclose(eq, qq.q, rtol=0, atol=1e-15, equal_nan=True)), f"{len(qq)} cells (receiver, outer, arm, lambda)")  # C4b
    J, CURV = d4_logs(sel)  # D4 and the training curves
    E.write_csv(J, DIAG_DIR / "diagnosis_data_budget_jobs.csv"); E.write_csv(CURV, DIAG_DIR / "diagnosis_learning_training_curves.csv")  # two statements: written
    D4C = d4_cells(J)  # D4's cells
    E.write_csv(D4C, DIAG_DIR / "diagnosis_data_budget_cells.csv")  # written
    check("C6 the logs rebuilt: every Monitor return re-derived from the step types and actions", bool(J.monitor_return_max_err.max() <= 1e-9), f"largest |rebuilt - Monitor| = {J.monitor_return_max_err.max():.2e} (scaled) over {int(J.episodes.sum())} episodes in {len(J)} jobs")  # C6
    tail = J[J.stage == "outer"].unlogged_tail  # an outer job's steps after its last finished episode (the unfinished one is not logged)
    check("C6b outer logs end within one episode of the budget", bool(((tail >= 0) & (tail <= 766)).all()), f"unlogged tail 0..766 s in {int(((tail >= 0) & (tail <= 766)).sum())} of {len(tail)} outer jobs (largest {int(tail.max())})")  # C6b
    BUD = d3_budget(cur, sel)  # D3's budget table
    E.write_csv(BUD, DIAG_DIR / "diagnosis_learning_budget.csv")  # written
    INS = d3_outer_insample(sel, steps)  # D3's in-sample margins
    E.write_csv(INS, DIAG_DIR / "diagnosis_learning_outer_insample.csv")  # written
    D3C = MG.groupby(["arm", "receiver", "lambda"], sort=True).agg(auroc_margin_median=("auroc_margin", "median"), auroc_z_median=("auroc_z", "median")).reset_index()  # D3's cells (out of sample)
    D3C["gap"] = D3C.auroc_margin_median - D3C.auroc_z_median  # margin minus z_t
    D3C["margin_below_z"] = D3C.gap < -0.05  # the pre-registered rule per cell
    D3C = D3C.merge(INS.groupby(["arm", "receiver", "lambda"]).agg(auroc_margin_insample_median=("auroc_margin_insample", "median")).reset_index(), on=["arm", "receiver", "lambda"])  # with the in-sample margins
    E.write_csv(D3C, DIAG_DIR / "diagnosis_learning_cells.csv")  # written
    D2C, D2F = d2()  # D2
    E.write_csv(D2C, DIAG_DIR / "diagnosis_operating_point_cells.csv"); E.write_csv(D2F, DIAG_DIR / "diagnosis_operating_point_folds.csv")  # two statements: written
    D6C, D6P, D6M, EW, D6S = d6(steps)  # D6
    E.write_csv(D6C, DIAG_DIR / "diagnosis_secondary_reading_cells.csv"); E.write_csv(D6P, DIAG_DIR / "diagnosis_secondary_reading_pairs.csv"); E.write_csv(D6M, DIAG_DIR / "diagnosis_secondary_reading_marks.csv"); E.write_csv(EW, DIAG_DIR / "diagnosis_secondary_reading_early_warning.csv")  # four statements: written
    rq4_intervals = rt(E.PROC / "rq4" / "intervals.csv")  # the RQ4 study's interval table
    rq4_points = rq4_intervals[rq4_intervals.scope == "position sessions"].groupby(["detector", "share"]).agg(runs=("runs", "sum"), counted=("counted_runs", "sum"))  # its interval-runs and counted runs per detector and share
    rq4_points = rq4_points.reindex(pd.MultiIndex.from_frame(D6S[["detector", "share"]]))  # in D6's order
    check("C7 D6 reconciles with the RQ4 study's interval table, per detector and share", bool((rq4_points.runs.to_numpy() == D6S.interval_runs.to_numpy()).all() and (rq4_points.counted.to_numpy() == D6S.counted.to_numpy()).all()), f"{len(D6S)} detector x share cells; at 0.05: interval-runs {D6S[D6S.share == 0.05].interval_runs.tolist()}, counted {D6S[D6S.share == 0.05].counted.tolist()}")  # C7
    D5T = d5(R)  # D5
    E.write_csv(D5T, DIAG_DIR / "diagnosis_reward_bookkeeping_hand.csv")  # written
    check("C5 D5's hand returns equal the environment script's step, the training script's closed form, the stored costs and the Monitor", bool(D5T.max_abs_diff.max() <= 1e-12), f"{len(D5T)} items, largest difference {D5T.max_abs_diff.max():.2e}")  # C5
    V = verdicts(D1C, H2C, D2C, D3C, BUD, D4C, D5T)  # the verdicts
    E.write_csv(V, DIAG_DIR / "diagnosis_verdicts.csv")  # written
    figures(R, O, H2C, D1C, D2C, D2F, D3C, MG, D4C, D6M)  # the figures
    check("C8 torch on one thread throughout", torch.get_num_threads() == 1, f"torch.get_num_threads() = {torch.get_num_threads()}")  # C8
    E.write_csv(pd.DataFrame(CHECKS), DIAG_DIR / "diagnosis_checks.csv")  # the checks
    outs = sorted([p for p in DIAG_DIR.glob("diagnosis_*") if p.name not in ("diagnosis_regen_timing.csv", "diagnosis_outputs_manifest.csv", "diagnosis_tests.csv")]) + sorted(GRAPH.glob("diagnosis_*.png"))  # list concatenation: every output of this step (the timing table is declared variable, diagnosis_tests.csv belongs to the diagnosis tests)
    E.write_csv(pd.DataFrame({"file": [str(p.relative_to(PROJECT)) for p in outs], "sha256": [T.sha_file(p) for p in outs]}), DIAG_DIR / "diagnosis_outputs_manifest.csv")  # list comprehensions: the outputs' digests (the diagnosis tests compares two runs)
    print(V.to_string(index=False), flush=True)  # the verdicts
    if not all(c["pass"] for c in CHECKS):  # if: any check failed
        raise SystemExit("STOP: a check failed (diagnosis_checks.csv)")  # stop


# ----------------------------------------------------------------------------------------------------------------------
# command line
# ----------------------------------------------------------------------------------------------------------------------
def main() -> None:  # function: the command line
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])  # the parser, described by the docstring's first line
    ap.add_argument("--regen", action="store_true", help="run again the 1,200 inner jobs to their selected budgets, gated")  # the regeneration step
    ap.add_argument("--diag", action="store_true", help="D1-D6 and H2")  # the diagnostic step
    a = ap.parse_args()  # parsed
    if a.regen:  # if: the regeneration step
        regen()  # run
    if a.diag:  # if: the diagnostic step
        diag()  # run


if __name__ == "__main__":  # if: run as a script (not imported, e.g. by a spawned worker)
    main()  # the command line
