#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
The checks of agent training and evaluation (results in checks.csv).

With --pilot only these run, on the pilot's job outputs:
  T1  the episode returns SB3's Monitor logged during training equal the environment's closed-form cost of the same actions, divided by
      100 (P1), on a seeded sample of 1,000 episodes per arm. The closed form reads the step types from steps.csv.gz and the interval
      ends and the miss penalty from onsets.csv, never from the environment or from the training script.
  T1b the truncation decision of the agent experiment: a truncated clean episode's final observation is the next second's state when that second is a
      valid step of the same window run, and the last row's own state otherwise (the surrogate's S continuing its recursion).
  T2p no pilot episode lies in the outer held-out session or in 2.6.12, and no inner job's episode lies in its inner held-out session.
T2-T8 run with --part1 (training) and --part2 (evaluation).
Reads   Data/processed/rl_agents/<NAME>/ (job outputs), Data/processed/rl_environment/ (steps, onsets, episodes, states)
Writes  Data/processed/rl_agents/<NAME>/inspection_checks.csv and pilot_t1_sample.csv
"""
from __future__ import annotations  # future import: type hints kept as text (the project's convention)

import argparse  # standard library: the command line
import gzip  # standard library: reads the gzip job outputs
import importlib.util  # standard library: loads the training script (its name starts with a digit)
import json  # standard library: the job outputs
import sys  # standard library: module registry and exit status
from pathlib import Path  # standard library: paths

import numpy as np  # numpy: arrays
import pandas as pd  # pandas: tables

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
_spec = importlib.util.spec_from_file_location("train_agents_for_tests", PROJECT / "codes" / "6_reinforcement_learning" / "02_train_agents.py")  # where 12 is
T = importlib.util.module_from_spec(_spec)  # an empty module object
sys.modules["train_agents_for_tests"] = T  # registered
_spec.loader.exec_module(T)  # the training script's top level (it loads the environment script and the adopted-pipeline script); its main() is not run
E = T.E  # 11
_spec_b = importlib.util.spec_from_file_location("evaluate_agents_for_tests", PROJECT / "codes" / "6_reinforcement_learning" / "03_evaluate_agents.py")  # where the evaluation script is
B = importlib.util.module_from_spec(_spec_b)  # an empty module object
sys.modules["evaluate_agents_for_tests"] = B  # registered
_spec_b.loader.exec_module(B)  # the evaluation script's top level (constants and functions); its main() is not run
ENV_DIR, AGENT_DIR, SEED = E.ENV_DIR, T.AGENT_DIR, T.SEED  # folders and the seed
N_T1 = 1000  # T1's sample per arm
N_T1B = 400  # T1b's clean episodes per arm
TOL = 1e-9  # T1's tolerance on scaled returns (the sum of rewards divided by 100 rounds in the last bits)
CHECKS: list = []  # the results


def check(test: str, name: str, ok, detail: str) -> None:  # function: records one check
    CHECKS.append({"test": test, "check": name, "pass": bool(ok), "detail": detail})  # one row
    print(f"{test} {'PASS' if ok else 'FAIL'} {name}: {detail}")  # printed


def job_outputs(folder: Path) -> list:  # function: every job output of a run, with its specification
    return [json.loads(gzip.decompress(p.read_bytes())) for p in sorted((folder / "jobs").glob("*.json.gz"))]  # list comprehension: parsed in name order


def t1(folder: Path) -> pd.DataFrame:  # function: T1 on a seeded sample of training episodes per arm
    code = {"pre": 0, "post": 1, "uncertain": 2, "excluded": 3}  # dict: step -> code
    st = pd.read_csv(ENV_DIR / "steps.csv.gz", dtype={"session": str})  # the step table (independent of the environment)
    ty = {(r, s, int(t)): code[k] for r, s, t, k in zip(st.receiver, st.session, st.sec, st.step)}  # dict comprehension: (receiver, session, second) -> step code
    va = {(r, s, int(t)): (bool(v), int(u)) for r, s, t, v, u in zip(st.receiver, st.session, st.sec, st.valid, st.run_id)}  # dict comprehension: -> (valid, window run)
    on = pd.read_csv(ENV_DIR / "onsets.csv", dtype={"session": str})  # attack runs with usability and interval ends
    rows = []  # every episode of the pilot
    for r in job_outputs(folder):  # for loop over the jobs
        s = r["spec"]  # its specification
        train = [x for x in T.SESSIONS if x not in (s["outer"], s["inner"])]  # the sessions its training episodes may use
        u = on[(on.receiver == s["receiver"]) & on.session.isin(train) & on.usable]  # their usable onsets
        m = float(np.median(u.length_s))  # the miss penalty recomputed from onsets (not read from the training script)
        for k, e in enumerate(r["episodes"]):  # for loop over the job's episodes
            rows.append({"job": T.job_id(s), "arm": s["arm"], "receiver": s["receiver"], "outer": s["outer"], "inner": s["inner"], "lam": float(s["lambda"]), "m": m, "k": k, **e})  # dict unpacking: the episode
    ep = pd.DataFrame(rows)  # every episode
    out = []  # the sample with its comparison
    for arm, d in ep.groupby("arm", sort=True):  # for loop over the arms
        smp = d.sample(n=min(N_T1, len(d)), random_state=np.random.RandomState(SEED)).sort_values(["job", "k"])  # seeded sample
        for e in smp.itertuples():  # for loop over the sampled episodes
            secs = np.arange(e.start, e.start + e.steps)  # its seconds (episodes are consecutive seconds)
            tys = np.array([ty[(e.receiver, e.session, int(t))] for t in secs])  # list comprehension: step codes from the step table
            ok_path = all(va[(e.receiver, e.session, int(t))][0] for t in secs) and len({va[(e.receiver, e.session, int(t))][1] for t in secs}) == 1 and (tys != 3).all()  # generator in all() and set comprehension: valid, one run, no excluded second
            a = np.unpackbits(np.frombuffer(bytes.fromhex(e.actions_hex), np.uint8))[:e.steps]  # the actions taken
            stop = np.flatnonzero((a == 1) & (tys <= 1))  # alarms in pre- or post-change seconds
            if stop.size:  # a terminating alarm
                j = int(stop[0])  # its position
                ok_end = e.terminated and j == e.steps - 1  # the episode must have ended right there
                cost = float((tys[:j] == 1).sum()) + (e.lam if tys[j] == 0 else 0.0)  # silent post-change seconds before it, and lambda for a false alarm
            else:  # no terminating alarm
                iv = on[(on.receiver == e.receiver) & (on.session == e.session) & (on.onset == e.onset)]  # the onset's interval (onset episodes)
                miss = e.kind == "onset"  # an onset episode without an alarm is a miss
                ok_end = (e.terminated and miss and int(secs[-1]) == int(iv.end.iloc[0])) or (not e.terminated and not miss)  # a miss ends at the interval's last second; a clean episode is truncated
                cost = float((tys == 1).sum()) + (e.m if miss else 0.0)  # silent post-change seconds, and m for a miss
            out.append({"arm": arm, "job": e.job, "episode": e.k, "kind": e.kind, "session": e.session, "start": e.start, "steps": e.steps, "closed_form_cost": cost, "expected_return": -cost / T.REWARD_SCALE, "monitor_return": e.monitor_return, "path_ok": bool(ok_path), "end_ok": bool(ok_end)})  # the comparison row
    return pd.DataFrame(out), ep  # the sample and every episode


def t1b() -> pd.DataFrame:  # function: the truncation decision, replayed through the environment script's environment on the pilot cell's outer material
    rx, outer = T.PILOT["receiver"], T.PILOT["outer"]  # the pilot cell
    mat = E.material(rx, outer)  # the environment script's material
    ep = pd.read_csv(ENV_DIR / "episodes.csv.gz", dtype={"session": str, "fold": str})  # the environment script's episode table
    ep = ep[(ep.receiver == rx) & (ep.fold == outer) & (ep.kind == "clean")].head(N_T1B)  # its first clean episodes
    rows = []  # one row per arm and episode
    for arm in ("primary", "surrogate"):  # for loop over the arms
        env = E.SpoofingDetectionEnv(mat, arm, 100.0)  # the environment
        for e in ep.itertuples():  # for loop over the episodes
            env.reset(options={"kind": "clean", "session": e.session, "start": int(e.start)})  # replayed
            done, n, S = False, 0, 0.0  # finished; steps; the surrogate's running S (recomputed here, not read from the env)
            while not done:  # never alarm until the episode ends
                r_ = mat["key"][(e.session, int(e.start) + n)]  # this second's row
                S = max(0.0, S + float(mat["surrogate"][r_, -1]))  # Page's recursion
                obs, _, term, trunc, _ = env.step(0)  # silent
                done, n = term or trunc, n + 1  # tuple assignment
            nxt = (e.session, int(e.end) + 1)  # the next second
            same = nxt in mat["key"] and mat["run"][mat["key"][nxt]] == mat["run"][mat["key"][(e.session, int(e.end))]]  # a valid step of the same run?
            row = mat["key"][nxt] if same else mat["key"][(e.session, int(e.end))]  # the expected row
            x = mat[arm][row]  # its state
            if arm == "surrogate":  # the surrogate appends S
                x = np.append(x, max(0.0, S + float(x[-1])) if same else S)  # S continued to the next second, or S of the last second
            rows.append({"arm": arm, "session": e.session, "start": int(e.start), "end": int(e.end), "truncated": bool(trunc), "next_valid_same_run": bool(same), "equal": bool(np.array_equal(obs, x))})  # the comparison
    return pd.DataFrame(rows)  # the table


# ----------------------------------------------------------------------------------------------------------------------
# --part1: T1-T3, T5 and T7 on the full grid
# ----------------------------------------------------------------------------------------------------------------------
GRID = AGENT_DIR / T.GRID  # the grid's folder
SEALED_SCAN = ("session", "inner", "outer", "outer_held_out", "held_out", "fold")  # tuple: the columns of the grid's tables that name a session


def episode_ok(e: dict, spec: dict) -> bool:  # function: T2's rule for one training episode: never in the outer held-out session, never in an inner job's inner held-out session, never in 2.6.12
    return e["session"] not in (spec["outer"], E.SEALED) and (spec["inner"] == "" or e["session"] != spec["inner"])  # the rule


def t1_t2(jobs: Path) -> tuple:  # function: one streaming pass over every job output: T2's episode check, then T1 on a seeded sample of 1,000 episodes per arm
    code = {"pre": 0, "post": 1, "uncertain": 2, "excluded": 3}  # dict: step -> code
    st = pd.read_csv(ENV_DIR / "steps.csv.gz", dtype={"session": str})  # the step table (independent of the environment)
    ty = {(r, s, int(t)): code[k] for r, s, t, k in zip(st.receiver, st.session, st.sec, st.step)}  # dict comprehension: (receiver, session, second) -> step code
    va = {(r, s, int(t)): (bool(v), int(u)) for r, s, t, v, u in zip(st.receiver, st.session, st.sec, st.valid, st.run_id)}  # dict comprehension: -> (valid, window run)
    on = pd.read_csv(ENV_DIR / "onsets.csv", dtype={"session": str})  # attack runs
    files = sorted(jobs.glob("*.json.gz"))  # every job output
    count, bad, n_ep = [], [], 0  # (file, arm, episodes); failing episodes; all episodes
    for f in files:  # first pass: counts and the episode check
        r = json.loads(gzip.decompress(f.read_bytes()))  # the output
        count.append((f, r["spec"]["arm"], len(r["episodes"])))  # its count
        n_ep += len(r["episodes"])  # counted
        bad += [(f.name, k) for k, e in enumerate(r["episodes"]) if not episode_ok(e, r["spec"])]  # list comprehension: episodes breaking T2's rule
    rng = np.random.default_rng(SEED)  # the sample's generator
    out = []  # the sample with its comparison
    for arm in ("primary", "surrogate"):  # for loop over the arms
        c = [(f, n) for f, a_, n in count if a_ == arm]  # its jobs and counts
        cum = np.cumsum([n for _, n in c])  # running totals
        pick = np.sort(rng.choice(int(cum[-1]), size=N_T1, replace=False))  # 1,000 distinct episode numbers over the arm's jobs
        for f_i in np.unique(np.searchsorted(cum, pick, side="right")):  # for loop over the jobs holding sampled episodes
            f, _ = c[f_i]  # the job file
            r = json.loads(gzip.decompress(f.read_bytes()))  # its output
            s = r["spec"]  # its specification
            base = int(cum[f_i - 1]) if f_i else 0  # its first episode number
            train = [x for x in T.SESSIONS if x not in (s["outer"], s["inner"])]  # the sessions its training episodes may use
            m = float(np.median(on[(on.receiver == s["receiver"]) & on.session.isin(train) & on.usable].length_s))  # the miss penalty recomputed from onsets
            for k in pick[(pick >= base) & (pick < base + len(r["episodes"]))] - base:  # for loop over its sampled episodes
                e = r["episodes"][int(k)]  # the episode
                secs = np.arange(e["start"], e["start"] + e["steps"])  # its seconds
                tys = np.array([ty[(s["receiver"], e["session"], int(t))] for t in secs])  # list comprehension: step codes from the step table
                ok_path = all(va[(s["receiver"], e["session"], int(t))][0] for t in secs) and len({va[(s["receiver"], e["session"], int(t))][1] for t in secs}) == 1 and (tys != 3).all()  # generator and set comprehension: valid, one run, no excluded second
                a = np.unpackbits(np.frombuffer(bytes.fromhex(e["actions_hex"]), np.uint8))[:e["steps"]]  # the actions taken
                stop = np.flatnonzero((a == 1) & (tys <= 1))  # alarms in pre- or post-change seconds
                if stop.size:  # a terminating alarm
                    j = int(stop[0])  # its position
                    ok_end = e["terminated"] and j == e["steps"] - 1  # the episode must have ended there
                    cost = float((tys[:j] == 1).sum()) + (float(s["lambda"]) if tys[j] == 0 else 0.0)  # silent post-change seconds, and lambda for a false alarm
                else:  # no terminating alarm
                    miss = e["kind"] == "onset"  # an onset episode without an alarm is a miss
                    iv = on[(on.receiver == s["receiver"]) & (on.session == e["session"]) & (on.onset == e["onset"])]  # its interval
                    ok_end = (e["terminated"] and miss and int(secs[-1]) == int(iv.end.iloc[0])) or (not e["terminated"] and not miss)  # a miss ends at the interval's end; a clean episode is truncated
                    cost = float((tys == 1).sum()) + (m if miss else 0.0)  # silent post-change seconds, and m for a miss
                out.append({"arm": arm, "job": f.name, "episode": int(k), "kind": e["kind"], "session": e["session"], "steps": e["steps"], "closed_form_cost": cost, "expected_return": -cost / T.REWARD_SCALE, "monitor_return": e["monitor_return"], "path_ok": bool(ok_path), "end_ok": bool(ok_end)})  # the comparison row
    return pd.DataFrame(out), bad, n_ep, len(files)  # the sample, the failing episodes, the counts


def t2_eval_episodes() -> tuple:  # function: T2 for the inner evaluation: every inner fold's evaluation episodes and stream lie in its inner held-out session only
    bad, n = [], 0  # failing cells; episodes checked
    for rx in T.RXS:  # for loop over the receivers
        for outer in T.SESSIONS:  # for loop over the outer folds
            for inner in [s for s in T.SESSIONS if s != outer]:  # for loop over the inner folds
                for arm in ("primary", "surrogate"):  # for loop over the arms
                    tr, evm = T.job_material({"receiver": rx, "outer": outer, "inner": inner, "arm": arm}, GRID / "states")  # the materials
                    env = E.SpoofingDetectionEnv(evm, arm, 100.0)  # the evaluation environment
                    eps = T.eval_episodes(env)  # the episodes
                    sess = {str(evm["session"][r_]) for ep in eps for r_ in ep["rows"]}  # set comprehension: every session an evaluation step lies in
                    tr_s = {s for s, _ in tr["key"]}  # set comprehension: every session the training material holds
                    n += len(eps)  # counted
                    if sess - {inner} or outer in tr_s or inner in tr_s or E.SEALED in tr_s:  # if: an evaluation step outside the inner held-out session, or a held-out session in training
                        bad.append(f"{rx} {outer} {inner} {arm}")  # recorded
    return bad, n  # the failures and the count


def sealed_scan(folder: Path) -> list:  # function: every grid table naming 2.6.12 in a session column
    hits = []  # (file, column)
    for f in sorted(list(folder.glob("*.csv")) + list(folder.glob("*.csv.gz"))):  # for loop over the grid's tables
        d = pd.read_csv(f, dtype=str)  # every column as text
        hits += [(f.name, c) for c in SEALED_SCAN if c in d.columns and (d[c] == E.SEALED).any()]  # list comprehension: columns holding 2.6.12
    return hits  # the hits


def t3(n_jobs: int = 5) -> pd.DataFrame:  # function: T3, five randomly chosen seed-20260914 jobs run again from scratch into another folder; files and greedy actions compared
    man = pd.read_csv(GRID / "jobs_manifest.csv")  # every grid job
    cand = man[man.job.str.contains(f"_s{SEED}_")].job.tolist()  # the jobs with seed 20260914
    pick = sorted(np.random.default_rng(SEED).choice(cand, size=n_jobs, replace=False).tolist())  # five of them, seeded
    rr = AGENT_DIR / "training_t3"  # the run again's folder
    rows = []  # one row per job
    for jid in pick:  # for loop over the chosen jobs
        spec = T.read_gz_json(GRID / "jobs" / f"{jid}.json.gz")["spec"]  # its specification
        for p in [rr / "jobs" / f"{jid}.json.gz", rr / "jobs" / f"{jid}.sha256", AGENT_DIR / "models" / "training_t3" / f"{jid}.npz"]:  # for loop over earlier run again files
            p.unlink(missing_ok=True)  # removed, so the job runs from scratch
        T.run_job(spec, str(GRID / "states"), str(rr / "jobs"), str(AGENT_DIR / "models" / "training_t3"))  # the run again in this process
        same_out = T.sha_file(GRID / "jobs" / f"{jid}.json.gz") == T.sha_file(rr / "jobs" / f"{jid}.json.gz")  # the outputs byte-identical?
        same_model, same_act = None, None  # for outer jobs only
        if spec["stage"] == "outer":  # an outer job: the model files and the greedy actions
            same_model = T.sha_file(AGENT_DIR / "models" / T.GRID / f"{jid}.npz") == T.sha_file(AGENT_DIR / "models" / "training_t3" / f"{jid}.npz")  # byte-identical?
            mat = E.material(spec["receiver"], spec["outer"])  # its material
            m1, _ = T.load_q(spec, AGENT_DIR / "models" / T.GRID, mat)  # the grid's agent
            m2, _ = T.load_q(spec, AGENT_DIR / "models" / "training_t3", mat)  # the run again's agent
            acts = []  # both agents' actions on the training states
            for m_ in (m1, m2):  # for loop over the two agents
                a_ = []  # actions per session
                for sess in [s for s in T.SESSIONS if s != spec["outer"]]:  # training sessions
                    d, r_, S = T.surrogate_runs(mat, sess)  # rows and S
                    o = T.signed_log(np.column_stack([mat["surrogate"][r_], S])) if spec["arm"] == "surrogate" else mat["primary"][r_]  # conditional expression: the arm's observations
                    a_.append(m_.predict(o, deterministic=True)[0])  # greedy actions
                acts.append(np.concatenate(a_))  # all of them
            same_act = bool(np.array_equal(acts[0], acts[1]))  # identical?
        rows.append({"job": jid, "stage": spec["stage"], "output_identical": same_out, "model_identical": same_model, "actions_identical": same_act})  # the row
    return pd.DataFrame(rows)  # the table


def t5(points_fn=None, path: Path | None = None) -> tuple:  # function: T5, the deployable points recomputed fold by fold with the held-out session removed from everything read
    points_fn = points_fn or (lambda sel_, n_, st_, m_: T.deployable(sel_, n_, st_, m_))  # lambda: the training script's points by default; the evaluation script's amended points with --part2
    path = path or GRID / "deployable_points.csv"  # the file the points were written to
    sel = pd.read_csv(GRID / "selection.csv", dtype={"outer": str}, float_precision="round_trip")  # the frozen selection (from inner job outputs, which hold no held-out second)
    nested = pd.read_csv(GRID / "nested_predictions.csv.gz", dtype={"outer_held_out": str, "held_out": str}, float_precision="round_trip")  # P's nested predictions
    steps = T.read_steps()  # the environment script's step table
    frozen = pd.read_csv(path, dtype={"outer": str}, float_precision="round_trip")  # the written points
    parts = []  # each fold's recomputed points
    for outer in T.SESSIONS:  # for loop over the outer folds

        def material(rx, held, outer=outer):  # nested function: the environment script's material with the held-out session's rows removed from the index and blanked in the matrices
            m = E.material(rx, held)  # the environment script's material
            if held != outer:  # another fold (its rows are discarded below)
                return m  # unchanged
            h = m["index"].session == outer  # the held-out rows
            X = {k: m[k].copy() for k in ("primary", "surrogate")}  # dict comprehension: copies of the matrices
            for k in X:  # for loop over the two matrices
                X[k][h.to_numpy()] = np.nan  # the held-out rows blanked
            return {**m, **X, "index": m["index"][~h]}  # dict unpacking: the material without the held-out session
        dp, _ = points_fn(sel, nested[nested.held_out != outer], steps[steps.session != outer], material)  # the points, every file read without this fold's held-out session (other folds' points, computed on incomplete input, are discarded)
        parts.append(dp[dp.outer == outer])  # this fold's points only
    re = pd.concat(parts, ignore_index=True)  # every fold
    key = ["receiver", "outer", "detector", "share"]  # the identity of a point
    b = re.set_index(key).loc[pd.MultiIndex.from_frame(frozen[key])].reset_index()[list(frozen.columns)]  # the recomputed points in the written file's row and column order
    same = b.to_csv(index=False, lineterminator="\n") == path.read_text(encoding="utf-8")  # identical as written text (no float parsing in between)
    return same, len(frozen)  # identical?, how many


def t7(sel: pd.DataFrame) -> list:  # function: T7, three planted faults, each of which a check above must catch
    res = []  # (fault, caught)
    r = T.read_gz_json(sorted((GRID / "jobs").glob("inner_*.json.gz"))[0])  # one inner job's output
    e = dict(r["episodes"][0], session=r["spec"]["outer"])  # dict(): its first episode moved into the outer held-out session
    res.append(("a planted held-out episode", not episode_ok(e, r["spec"])))  # T2's rule must reject it
    cur = pd.read_csv(GRID / "inner_curve.csv.gz", dtype={"outer": str, "inner": str}, float_precision="round_trip")  # the inner curve
    c1 = cur[(cur.receiver == "SBF_02") & (cur.outer == "2.6.2") & (cur.arm == "primary") & (cur["lambda"] == 100.0)]  # one cell
    leak = pd.concat([c1, c1.head(1).assign(inner="2.6.2", criterion=-1.0)], ignore_index=True)  # assign(): a planted row whose criterion comes from the held-out session
    try:  # try: select() must stop
        T.select(leak)  # the selection on the planted curve
        res.append(("a budget chosen on held-out data (planted inner-curve row)", False))  # not caught
    except SystemExit:  # its stop
        res.append(("a budget chosen on held-out data (planted inner-curve row)", True))  # caught
    bad = sel.copy()  # a copy of the frozen selection
    i = bad.index[0]  # its first row
    bad.loc[i, "selected_budget"] = [b for b in T.BUDGETS if b != bad.loc[i, "selected_budget"]][0]  # list comprehension: a planted budget other than the one the rule gives
    res.append(("a frozen budget that the inner rule does not give", not T.select(cur).selected_budget.equals(bad.selected_budget)))  # the recomputation must differ
    tmp = GRID / "_sealed.csv"  # a planted table
    E.write_csv(pd.DataFrame({"session": [E.SEALED]}), tmp)  # holding 2.6.12
    res.append(("a planted 2.6.12 row", ("_sealed.csv", "session") in sealed_scan(GRID)))  # the scan must find it
    tmp.unlink()  # removed
    return res  # the results


def part1() -> None:  # function: the --part1 checks, written to Data/processed/rl_agents/training/checks.csv
    smp, bad, n_ep, n_jobs = t1_t2(GRID / "jobs")  # T1 and T2's episode check
    E.write_csv(smp, GRID / "t1_sample.csv")  # the sample
    for arm, d in smp.groupby("arm", sort=True):  # for loop over the arms
        diff = (d.expected_return - d.monitor_return).abs()  # absolute differences
        check("T1", f"Monitor returns = closed-form cost / {T.REWARD_SCALE:g} ({arm})", len(d) == N_T1 and diff.max() <= TOL and d.path_ok.all() and d.end_ok.all(), f"{len(d)} episodes; largest difference {diff.max():.2e}; kinds {dict(d.kind.value_counts())}; paths ok {int(d.path_ok.sum())}; ends ok {int(d.end_ok.sum())}")  # recorded
    tb = t1b()  # the truncation decision, replayed
    for arm, d in tb.groupby("arm", sort=True):  # for loop over the arms
        check("T1b", f"truncated clean episodes return the next second's state when valid in the same run, else the last row's ({arm})", d.truncated.all() and d.equal.all(), f"{len(d)} episodes; next second {int(d.next_valid_same_run.sum())}, last row {int((~d.next_valid_same_run).sum())}")  # recorded
    check("T2", "no training episode in the outer held-out session or 2.6.12; none of an inner job in its inner held-out session", not bad, f"{n_ep:,} episodes in {n_jobs} jobs; failing {bad[:3]}")  # recorded
    be, ne = t2_eval_episodes()  # the inner evaluation episodes
    check("T2", "every inner evaluation episode lies in its inner held-out session; no held-out session or 2.6.12 in any inner training material", not be, f"{ne:,} evaluation episodes in 120 inner cells; failing {be[:3]}")  # recorded
    hits = sealed_scan(GRID)  # 2.6.12 in a table
    check("T2", "2.6.12 absent from every grid table", not hits, f"hits {hits}")  # recorded
    fm = pd.read_csv(GRID / "freeze_manifest.csv")  # the freeze manifest
    check("T2", "the freeze manifest's digests match the frozen files (the evaluation script checks the freeze commit before --heldout)", all(T.sha_file(GRID / f) == h for f, h in zip(fm.file, fm.sha256)), f"{len(fm)} files")  # recorded
    sel = pd.read_csv(GRID / "selection.csv", dtype={"outer": str}, float_precision="round_trip")  # the frozen selection
    cur = pd.read_csv(GRID / "inner_curve.csv.gz", dtype={"outer": str, "inner": str}, float_precision="round_trip")  # the inner curve
    re = T.select(cur)  # recomputed
    check("T2", "the frozen budgets are the inner rule's (recomputed from the inner curve, which holds no outer held-out session)", re.selected_budget.equals(sel.selected_budget) and not (cur.inner == cur.outer).any(), f"{len(sel)} cells")  # recorded
    t3t = t3()  # T3
    E.write_csv(t3t, GRID / "t3_repeat.csv")  # written
    check("T3", "five seed-20260914 jobs run again from scratch: outputs, models and greedy actions identical", t3t.output_identical.all() and t3t.model_identical.dropna().all() and t3t.actions_identical.dropna().all(), f"jobs {list(t3t.job)}")  # recorded
    ok5, n5 = t5()  # T5
    check("T5", "deployable points recomputed with each fold's held-out session removed from every file read are identical", ok5, f"{n5} points")  # recorded
    for name, caught in t7(sel):  # T7
        check("T7", f"negative test caught: {name}", caught, "planted and caught" if caught else "NOT caught")  # recorded
    ck = pd.DataFrame(CHECKS)  # the table
    E.write_csv(ck, GRID / "checks.csv")  # written
    if not ck["pass"].all():  # if: a check failed
        sys.exit(1)  # exit status 1


# ----------------------------------------------------------------------------------------------------------------------
# --part2: T4, T5 (amended points), T6
# ----------------------------------------------------------------------------------------------------------------------
def t4() -> list:  # function: T4, the evaluation script's baseline path at the threshold 0.5 reproduces the committed decision-mode figures V2 checked
    steps = T.read_steps()  # the environment script's step table
    bad, n = [], 0  # mismatches; folds checked
    for mdl, f in (("lr", "logistic_regression/logistic_regression_metrics_primary.csv"), ("rf", "random_forest/random_forest_metrics_primary.csv")):  # for loop over the two models
        me = B.rt(E.PROC / f)  # the committed metrics, floats exact
        preds = B.rt(E.PROC / f.replace("metrics", "predictions").replace(".csv", ".csv.gz"))  # the committed outer probabilities
        for rx in T.RXS:  # for loop over the receivers
            for outer in T.SESSIONS:  # for loop over the folds
                sd = T.session_seconds(steps, rx, outer)  # the stream grid
                pred = {mdl: preds[(preds.receiver == rx) & (preds.held_out == outer)].set_index("sec").p}  # dict literal: the probabilities by second
                d_, _ = B.baseline_decisions(mdl, 0.5, 0, 0, sd, pred, np.zeros(len(sd), bool), np.zeros((len(sd), 30)))  # the evaluation script's decisions at 0.5
                fd = E.stream(sd.sec.to_numpy(np.int64), sd.label.to_numpy(object), d_, outer, "decision")  # decision mode
                c = me[(me.receiver == rx) & (me.held_out == outer)].iloc[0]  # the committed row
                for k in ("recall", "precision", "f1", "tp", "fn", "fp", "tn", "clean_in_alarm", "fa_episodes"):  # for loop over V2's figures
                    x, y = fd[k], c[k]  # new and committed
                    if not ((pd.isna(x) and pd.isna(y)) or (not pd.isna(x) and not pd.isna(y) and float(x) == float(y))):  # if: they differ (exact, NaN equal to NaN)
                        bad.append(f"{mdl} {rx} {outer} {k}: {x} vs {y}")  # recorded
                n += 1  # counted
    return bad, n  # mismatches and folds


def t6() -> list:  # function: T6, the aggregation against hand computations and the bootstrap against an independent seeded reference
    res = []  # (name, ok, detail)
    res.append(("IQM of three runs is their mean (floor(3/4) = 0 cut)", float(B.iqm(np.array([9.0, 1.0, 5.0]))) == 5.0, "iqm([9, 1, 5]) = 5"))  # a 3-run hand case
    res.append(("IQM of 1..8 cuts two from each end", float(B.iqm(np.arange(1.0, 9.0))) == 4.5, "mean of 3, 4, 5, 6 = 4.5"))  # an 8-run hand case
    toy = pd.DataFrame({"receiver": ["X"] * 3, "outer": ["a", "b", "c"], "detector": ["lr"] * 3, "share": [0.05] * 3,  # a 3-run toy table: one run per fold
                        "dec_recall": [0.2, 0.5, 0.8], "dec_precision": [np.nan, 0.5, 1.0], "dec_f1": [np.nan, 0.5, 0.9], "dec_clean": [100, 50, 0], "dec_clean_in_alarm": [10, 5, 0], "dec_clean_share": [0.1, 0.1, np.nan], "dec_early_warning": [0.0, 0.3, np.nan],  # dict literal continued: decision mode
                        "stop_delay_median_s": [10.0, 20.0, 60.0], "stop_miss_rate": [0.0, 0.5, 0.0], "stop_clean_in_alarm": [1, 0, 0], "stop_fa_rate_per_h": [8.0, np.nan, np.nan], "stop_clean_share": [0.01, 0.0, np.nan]})  # dict literal continued: stopping mode
    lat = pd.DataFrame({"receiver": ["X"] * 4, "outer": ["a", "b", "b", "c"], "detector": ["lr"] * 4, "share": [0.05] * 4, "lambda": [np.nan] * 4, "detected": [True, True, False, True], "delay_s": [10.0, 20.0, np.nan, 60.0]})  # its intervals
    ag = B.aggregate(toy, lat, pd.DataFrame(columns=["receiver", "detector", "share", "no_point"])).set_index("metric")  # the evaluation script's aggregation
    hand = {"recall": 0.5, "precision": 0.75, "F1": 0.7, "median delay (s)": 30.0, "miss rate": 0.5 / 3, "false alarms (events on clean seconds)": 1 / 3, "stopping-mode clean-window alarm share (at most about 1/31)": 0.005}  # dict: the hand means (three runs or fewer: the IQM is the mean of the defined runs)
    ok = all(abs(ag.loc[k, "iqm"] - v) < 1e-12 for k, v in hand.items())  # generator in all(): each against the hand value
    ok = ok and ag.pooled_median_delay_s.iloc[0] == 20.0 and abs(ag.pooled_miss_share.iloc[0] - 0.25) < 1e-12 and abs(ag.pooled_clean_share.iloc[0] - 15 / 150) < 1e-12  # the pooled figures by hand: median of 10, 20, 60; 1 of 4 missed; 15 of 150
    res.append(("aggregation of a 3-run toy table equals the hand computation", ok, f"IQMs {dict(ag.iqm.round(6))}"))  # recorded
    v = np.array([3.0, 7.0, 1.0, 9.0, 4.0, 4.0, 8.0, 2.0, 6.0, 5.0])  # ten toy runs
    g = np.array(["a"] * 4 + ["b"] * 3 + ["c"] * 3)  # in three folds
    p, lo, hi, n = B.strat_boot(v, g, n_boot=2000, seed=SEED)  # the evaluation script's bootstrap
    rng = np.random.default_rng(SEED)  # the reference: the same generator, drawing in the same stratum order
    draws = [v[g == s_][rng.integers(0, int((g == s_).sum()), size=(2000, int((g == s_).sum())))] for s_ in sorted(set(g))]  # list comprehension: each fold's resamples
    ref = []  # each resample's IQM, by an independent loop
    for b_ in range(2000):  # for loop over the resamples
        x = sorted(float(t) for d in draws for t in d[b_])  # generator in sorted(): its ten values in order
        c = len(x) // 4  # cut from each end
        ref.append(sum(x[c:len(x) - c]) / (len(x) - 2 * c))  # the middle mean
    ref = sorted(ref)  # in order, for the percentiles
    pct = lambda q: ref[int(np.floor(q / 100 * (len(ref) - 1)))] + (q / 100 * (len(ref) - 1) - np.floor(q / 100 * (len(ref) - 1))) * (ref[min(int(np.floor(q / 100 * (len(ref) - 1))) + 1, len(ref) - 1)] - ref[int(np.floor(q / 100 * (len(ref) - 1)))])  # lambda: the linear-interpolation percentile by hand
    res.append(("stratified bootstrap equals an independent seeded reference", abs(lo - pct(2.5)) < 1e-12 and abs(hi - pct(97.5)) < 1e-12 and abs(p - float(np.mean(sorted(v)[2:8]))) < 1e-12, f"interval {lo:.6f}..{hi:.6f} vs {pct(2.5):.6f}..{pct(97.5):.6f}"))  # recorded
    return res  # the results


def part2() -> None:  # function: the --part2 checks, written to Data/processed/rl_agents/evaluation/checks.csv
    bad, n = t4()  # T4
    check("T4", "the baselines through the evaluation script at the threshold 0.5 reproduce the committed decision-mode figures (recall, precision, F1, cells, clean alarms, false-alarm episodes)", not bad, f"{n} model x receiver x fold cases; mismatches {bad[:3]}")  # recorded
    ok, n5 = t5(B.points_table, B.EVAL / "deployable_points.csv")  # T5 on the amended points
    check("T5", "amended deployable points recomputed with each fold's held-out session removed from every file read are identical", ok, f"{n5} points")  # recorded
    for name, ok6, det in t6():  # T6
        check("T6", name, ok6, det)  # recorded
    fc = pd.read_csv(B.EVAL / "freeze_check.csv")  # the evaluation script's freeze check
    check("T2", "the evaluation script ran only after every frozen file matched the freeze commit and the amended points were committed", bool((fc.in_freeze_commit & fc.equal_to_commit & fc.sha256_matches_manifest).all()), f"{len(fc)} files; freeze commit {B.FREEZE_COMMIT[:10]}")  # recorded
    ck = pd.DataFrame(CHECKS)  # the table
    E.write_csv(ck, B.EVAL / "checks.csv")  # written
    if not ck["pass"].all():  # if: a check failed
        sys.exit(1)  # exit status 1


def main() -> None:  # function: the --pilot checks
    ap = argparse.ArgumentParser()  # the parser
    ap.add_argument("--pilot", action="store_true")  # the --pilot checks
    ap.add_argument("--part1", action="store_true")  # the --part1 checks
    ap.add_argument("--part2", action="store_true")  # the --part2 checks
    ap.add_argument("--t5amended", action="store_true")  # T5 on the amended points, before any held-out stream is read
    ap.add_argument("--out", default="pilot")  # the pilot folder
    a = ap.parse_args()  # parsed
    if a.part1:  # --part1
        return part1()  # the --part1 checks
    if a.part2:  # --part2
        return part2()  # the --part2 checks
    if a.t5amended:  # T5 on the amended points (the evaluation script --points)
        ok, n = t5(B.points_table, B.EVAL / "deployable_points.csv")  # recomputed with each held-out session removed
        check("T5", "amended deployable points recomputed with each fold's held-out session removed from every file read are identical", ok, f"{n} points")  # recorded
        E.write_csv(pd.DataFrame(CHECKS), B.EVAL / "t5_amended.csv")  # written
        return sys.exit(0 if ok else 1)  # exit status
    if not a.pilot:  # nothing else
        raise SystemExit("usage: tests/test_agents.py --pilot | --part1 | --part2 | --t5amended")  # stop
    folder = AGENT_DIR / a.out  # the pilot's folder
    smp, ep = t1(folder)  # T1
    E.write_csv(smp, folder / "pilot_t1_sample.csv")  # the sample
    for arm, d in smp.groupby("arm", sort=True):  # for loop over the arms
        diff = (d.expected_return - d.monitor_return).abs()  # absolute differences
        check("T1", f"Monitor returns = closed-form cost / {T.REWARD_SCALE:g} ({arm})", len(d) == N_T1 and diff.max() <= TOL and d.path_ok.all() and d.end_ok.all(), f"{len(d)} episodes of {int((ep.arm == arm).sum())}; largest difference {diff.max():.2e}; kinds {dict(d.kind.value_counts())}; paths ok {int(d.path_ok.sum())}; ends ok {int(d.end_ok.sum())}")  # recorded
    tb = t1b()  # T1b
    for arm, d in tb.groupby("arm", sort=True):  # for loop over the arms
        check("T1b", f"truncated clean episodes return the next second's state when it is a valid step of the same run, else the last row's ({arm})", d.truncated.all() and d.equal.all(), f"{len(d)} episodes; next second used {int(d.next_valid_same_run.sum())}, last row used {int((~d.next_valid_same_run).sum())}; all equal {bool(d.equal.all())}")  # recorded
    bad_outer = ep[(ep.session == ep.outer) | (ep.session == E.SEALED)]  # episodes in the outer held-out session or 2.6.12
    bad_inner = ep[(ep.inner != "") & (ep.session == ep.inner)]  # inner jobs' episodes in their inner held-out session
    check("T2p", "no training episode in the outer held-out session or 2.6.12, none of an inner job in its inner held-out session", bad_outer.empty and bad_inner.empty, f"{len(ep)} episodes; sessions {dict(ep.session.value_counts())}")  # recorded
    ck = pd.DataFrame(CHECKS)  # the table
    E.write_csv(ck, folder / "inspection_checks.csv")  # written
    if not ck["pass"].all():  # if: a check failed
        sys.exit(1)  # exit status 1


if __name__ == "__main__":  # guard: runs main() only when the file is executed, not when imported
    main()  # runs main
