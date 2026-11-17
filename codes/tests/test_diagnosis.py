#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
Tests of 6_reinforcement_learning/04_diagnosis.py and of the follow-up's gates.

  T1  run again: 15 --diag run again from the same inputs gives every output byte-identical (diagnosis_outputs_manifest.csv before and after).
  T2  regeneration reproduced from scratch: a seeded sample of 12 inner jobs run again into a scratch folder gives byte-identical .npz files.
      (Every one of the 1,200 jobs is already gated against the agent experiment's own stored digests, the sample proves the written files are stable.)
  T3  D1's arithmetic by independent code: CUSUM's episode costs by a plain loop over every h (no running maximum, no search), the AUROC by
      counting pairs, kappa against scikit-learn's cohen_kappa_score, the stratified median bootstrap replayed draw by draw.
  T4  H2 recounted from the stored decisions with the transfer script's own refractory_at and onset_state, on every inner held-out stream of one receiver.
  T5  negative tests: a planted outer held-out second in an inner stream and a planted 2.6.12 row are caught.
It imports the diagnosis script (which loads the evaluation, training and environment scripts) and the transfer script (its two onset functions only). Nothing is written outside
Data/processed/rl_diagnosis/ and the scratchpad folder given with --scratch.
Writes  Data/processed/rl_diagnosis/diagnosis_tests.csv
"""
from __future__ import annotations  # future import: type hints kept as text (the project's convention)

import argparse  # standard library: the command line
import importlib.util  # standard library: loads the diagnosis and transfer scripts (their names start with a digit)
import shutil  # standard library: copies the manifest before the run again
import subprocess  # standard library: runs 15 and the comment check as separate processes
import sys  # standard library: module registry and exit status
from pathlib import Path  # standard library: paths

import numpy as np  # numpy: arrays
import pandas as pd  # pandas: tables

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/


def _load(name: str, file: str):  # function: loads a script by path as a module (its main() is not run)
    spec = importlib.util.spec_from_file_location(name, PROJECT / "codes" / file)  # where it is
    mod = importlib.util.module_from_spec(spec)  # an empty module object
    sys.modules[name] = mod  # registered
    spec.loader.exec_module(mod)  # its top level
    return mod  # the module


D = _load("diagnosis_for_tests", "6_reinforcement_learning/04_diagnosis.py")  # the diagnosis script: the diagnosis
T, E = D.T, D.E  # 12 and 11 as 15 holds them
DIAG_DIR = D.DIAG_DIR  # the section's folder
TESTS = []  # list: every test's outcome


def test(name: str, ok: bool, detail: str) -> None:  # function: records one test and prints it
    TESTS.append({"test": name, "pass": bool(ok), "detail": detail})  # kept
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)  # printed


def t1_repeat() -> None:  # function: T1, a second --diag run gives byte-identical outputs
    first = DIAG_DIR / "diagnosis_outputs_manifest.csv"  # the manifest of the first run
    keep = DIAG_DIR / "logs" / "diagnosis_outputs_manifest_run1.csv"  # its copy
    keep.parent.mkdir(parents=True, exist_ok=True)  # the folder
    shutil.copyfile(first, keep)  # copied before the run again overwrites it
    r = subprocess.run([sys.executable, str(PROJECT / "codes" / "6_reinforcement_learning" / "04_diagnosis.py"), "--diag"], cwd=PROJECT, capture_output=True, text=True)  # the run again as its own process
    (DIAG_DIR / "logs" / "diagnosis_diag_run2.log").write_text(r.stdout + r.stderr)  # its output kept
    a, b = pd.read_csv(keep), pd.read_csv(first)  # tuple assignment: the two manifests
    same = a.equals(b)  # every file and digest equal
    diff = sorted(set(map(tuple, a.values)) ^ set(map(tuple, b.values)))  # set symmetric difference: the rows that differ
    test("T1 run again of 15 --diag byte-identical", r.returncode == 0 and same, f"{len(a)} outputs compared; differing {[d_[0] for d_ in diff][:6]}")  # T1


def t2_regen_sample(scratch: Path) -> None:  # function: T2, a seeded sample of jobs regenerated from scratch into a scratch folder
    jobs = D.regen_specs()  # the 1,200 jobs
    rng = np.random.default_rng(D.SEED)  # the project seed
    pick = [jobs[i] for i in sorted(rng.choice(len(jobs), size=12, replace=False))]  # list comprehension: 12 jobs, in their fixed order
    q0 = D.QDIR  # the real folder
    D.QDIR = scratch / "inner_q_t2"  # the scratch folder for the run again (module attribute, read by regen_job at call time)
    D.QDIR.mkdir(parents=True, exist_ok=True)  # created
    same, names = [], []  # per job: equal bytes; names
    for j in pick:  # for loop over the sample
        jid = T.job_id(j["spec"])  # its name
        for p in (D.QDIR / f"{jid}.npz", D.QDIR / f"{jid}.sha256"):  # any earlier scratch copy removed, so the job really runs again
            p.unlink(missing_ok=True)  # removed
        r = D.regen_job(j)  # run again in this process
        same.append(r.get("gate", False) and (D.QDIR / f"{jid}.npz").read_bytes() == (q0 / f"{jid}.npz").read_bytes())  # gated, and byte-identical to the committed file
        names.append(jid)  # kept
    D.QDIR = q0  # restored
    test("T2 regenerated jobs reproduced from scratch byte-identically (seeded sample)", all(same), f"{sum(same)} of {len(same)}: {names[0]} ... {names[-1]}")  # T2


def t3_arithmetic() -> None:  # function: T3, D1's arithmetic by independent code on one inner fold
    from sklearn.metrics import cohen_kappa_score  # scikit-learn's kappa, as the reference
    R = pd.read_csv(DIAG_DIR / "diagnosis_optimisation_runs.csv", dtype={"outer": str, "inner": str}, float_precision="round_trip")  # D1's runs
    J = D.load_job(T.job_id(D.D5_CELL))  # the D5 cell's arrays
    fr = D.episode_frames("SBF_02", "2.6.2", "2.6.4", J["ep_ho"], J["ep_tr"])  # its episodes
    lam, m = D.D5_CELL["lambda"], fr["m"]  # lambda and m
    C = D.cusum_costs(fr["tr"], lam, m)  # the diagnosis script's vectorised costs
    worst = 0.0  # the largest difference
    for i, (ty, S, kind) in enumerate(fr["tr"]):  # for loop over the training episodes
        for k, h in enumerate(D.H_EP):  # for loop over every h
            c = None  # the cost by a plain walk
            post = 0  # silent post-change seconds so far
            for t in range(len(ty)):  # for loop over the steps
                if S[t] >= h and ty[t] == 0:  # an alarm in a pre-change second
                    c = post + lam  # a false alarm
                    break  # the end
                if S[t] >= h and ty[t] == 1:  # an alarm after the change
                    c = post  # a detection
                    break  # the end
                post += int(ty[t] == 1)  # a silent post-change second
            if c is None:  # no alarm ended it
                c = post + m if kind == "onset" else 0.0  # conditional expression: a miss, or a clean end
            worst = max(worst, abs(c - C[i, k]))  # the difference
    run = R[(R.receiver == "SBF_02") & (R.outer == "2.6.2") & (R.inner == "2.6.4") & (R.arm == "surrogate") & (R["lambda"] == lam) & (R.seed == D.D5_CELL["seed"])].iloc[0]  # the D1 row
    test("T3a CUSUM's episode costs by a plain walk over every h", worst <= 1e-12 and C.mean(axis=0).min() == run.cusum_train_cost, f"{len(fr['tr'])} episodes x {len(D.H_EP)} h, largest difference {worst:.1e}; h* cost {C.mean(axis=0).min():.4f}")  # T3a
    rng = np.random.default_rng(D.SEED)  # the seed
    pos, neg = rng.normal(1, 1, 60), rng.normal(0, 1, 80)  # two samples
    pos[:5] = neg[:5]  # with ties
    pairs = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])  # list comprehension: every pair counted
    test("T3b AUROC equals the pair count", abs(D.auroc(pos, neg) - pairs) <= 1e-12, f"{D.auroc(pos, neg):.6f} against {pairs:.6f}")  # T3b
    a, b = rng.random(500) < 0.2, rng.random(500) < 0.3  # two binary series
    test("T3c kappa equals scikit-learn's", abs(D.kappa(a, b) - cohen_kappa_score(a, b)) <= 1e-12, f"{D.kappa(a, b):.6f}")  # T3c
    v, s = rng.normal(0, 1, 40), np.repeat(["a", "b", "c", "d", "e"], 8)  # values and strata
    g = np.random.default_rng(D.SEED)  # the same generator as boot_median's
    meds = np.zeros(D.N_BOOT)  # the replayed medians
    draws = {k: g.integers(0, 8, size=(D.N_BOOT, 8)) for k in sorted(set(s))}  # dict comprehension: the draws in boot_median's order
    for b_ in range(D.N_BOOT):  # for loop over the resamples, one by one
        meds[b_] = np.median(np.concatenate([v[s == k][draws[k][b_]] for k in sorted(set(s))]))  # list comprehension: the resample's median
    ref = D.boot_median(v, s)  # the diagnosis script's function
    test("T3d the stratified median bootstrap replayed draw by draw", abs(np.percentile(meds, 2.5) - ref[1]) <= 1e-12 and abs(np.percentile(meds, 97.5) - ref[2]) <= 1e-12, f"[{ref[1]:.4f}, {ref[2]:.4f}]")  # T3d


def t4_h2() -> None:  # function: T4, H2's onset rows recounted with the transfer script's own functions on one receiver
    X = _load("transfer_for_diagnosis_tests", "7_evaluation/01_transfer.py")  # 13 (only refractory_at and onset_state are used; both are pure)
    O = pd.read_csv(DIAG_DIR / "diagnosis_refractory_onsets.csv.gz", dtype={"outer": str, "inner": str})  # the diagnosis script's onset rows
    steps = T.read_steps()  # labels
    n, bad = 0, 0  # rows compared; disagreements
    for j in [j for j in D.regen_specs() if j["spec"]["receiver"] == "UBX_01"]:  # list comprehension: every u-blox 01 agent
        sp = j["spec"]  # its specification
        J = D.load_job(T.job_id(sp))  # its arrays
        sd = T.session_seconds(steps, "UBX_01", sp["inner"])  # its stream
        sec, lab = sd.sec.to_numpy(np.int64), sd.label.to_numpy(object)  # seconds and labels
        ev = J["ho_stop"] > 0  # its events
        for a, _ in E.SC.pooled_intervals(sp["inner"]):  # for loop over the intervals
            r_, age = X.refractory_at(sec, ev, int(a))  # the transfer script's mark
            lab_ev = lab[sec == a - age][0] if r_ else ""  # the event's label
            st = X.onset_state(sec, J["ho_dec"], J["ho_ok"].astype(bool), int(a))["state"]  # the transfer script's state at onset
            o = O[(O.receiver == "UBX_01") & (O.outer == sp["outer"]) & (O.inner == sp["inner"]) & (O.arm == sp["arm"]) & (O["lambda"] == sp["lambda"]) & (O.seed == sp["seed"]) & (O.detector == "agent") & (O.onset == a)].iloc[0]  # the diagnosis script's row
            bad += int(bool(o.refractory) != bool(r_) or bool(o.refractory_uncertain) != bool(r_ and lab_ev == "uncertain") or o.onset_state != st)  # disagreement
            n += 1  # counted
    test("T4 H2's agent onset rows equal the transfer script's refractory_at and onset_state (u-blox 01, every agent)", bad == 0 and n > 0, f"{n} onset rows, {bad} disagreements")  # T4


def t5_negative() -> None:  # function: T5, planted leaks are caught
    steps = T.read_steps()  # the step table
    nested = D.rt(D.GRID12 / "nested_predictions.csv.gz")  # nested predictions
    orig = T.session_seconds  # the training script's function
    T.session_seconds = lambda st, rx, s: pd.concat([orig(st, rx, s), orig(st, rx, "2.6.2").iloc[:1]])  # lambda: every inner stream with one planted 2.6.2 second (the outer fold below)
    try:  # try block: the planted context must stop
        D.inner_ctx("SBF_02", "2.6.2", steps, nested)  # the fold whose held-out session is 2.6.2
        caught = False  # not caught
    except SystemExit:  # except: stopped, as it must
        caught = True  # caught
    finally:  # always
        T.session_seconds = orig  # restored
    J0 = D.load_job(T.job_id(D.D5_CELL))  # a job
    sealed = {**J0, "tr_sess": np.r_[J0["tr_sess"], 99]}  # dict unpacking with a replacement: a session code outside the five (where a 2.6.12 row would have to sit)
    try:  # try block: an unknown code cannot be named
        caught2 = not D.guard_job(sealed, D.D5_CELL["outer"], D.D5_CELL["inner"])  # guarded
    except IndexError:  # except: the code has no session at all, so the guard cannot even name it
        caught2 = True  # caught
    test("T5 planted outer held-out second and planted foreign (2.6.12) session caught", caught and caught2, "inner_ctx stopped; guard_job rejected the planted code")  # T5



# ----------------------------------------------------------------------------------------------------------------------
# the follow-up's gates G1-G6 (G7, the held-out dry run, runs after each freeze through the follow-up script's --dryrun)
# ----------------------------------------------------------------------------------------------------------------------
def hand_fc(ty: np.ndarray, a: np.ndarray, kind: str, lam: float, m: float) -> float:  # function: the Run A reward written out step by step, independently of the follow-up script (the return, unscaled)
    total, quiet = 0.0, 0  # the return; refractory seconds left
    for i in range(len(ty)):  # for loop over the steps
        counts = quiet == 0  # an alarm counts only outside a refractory period
        if ty[i] == 1:  # a post-change second
            if a[i] == 1 and counts:  # a detection
                return total  # the end at no cost
            total -= 1.0  # one second of delay
            if kind == "onset" and i == len(ty) - 1:  # the interval ends in silence
                return total - m  # the miss penalty
        elif ty[i] == 0 and a[i] == 1 and counts:  # a false alarm
            return total - lam  # lambda
        elif ty[i] == 2 and a[i] == 1 and counts:  # an uncertain-second alarm opens a period
            quiet = 30  # the next 30 seconds
            continue  # this second does not count down
        if quiet > 0:  # a running period
            quiet -= 1  # one less
    return total  # a clean episode's end


def followup_gates(scratch: Path) -> None:  # function: the follow-up's gates G1-G6, before any grid
    C = _load("followup_for_tests", "6_reinforcement_learning/05_followup.py")  # the follow-up script
    spec = {"stage": "inner", "receiver": "SBF_02", "outer": "2.6.2", "inner": "2.6.4", "arm": "surrogate", "lambda": 100.0, "seed": 20260914, "budget": T.SCHEDULE}  # a fixed cell
    train, evm = T.job_material(spec, C.STATES12)  # its materials
    env = C.FcEnv(train, "surrogate", 100.0)  # an F-c environment over the training sessions
    eps = [e for e in T.eval_episodes(env) if e["kind"] == "onset"]  # its onset episodes
    worst, n = 0.0, 0  # the largest difference; sequences compared
    for e in eps[::7]:  # for loop over every seventh onset episode
        ty = env.mat["type"][e["rows"]]  # types
        u = np.flatnonzero(ty == 2)  # uncertain positions
        p = np.flatnonzero(ty == 1)  # post-change positions
        seqs = [np.zeros(len(ty), int)]  # list: the silent sequence (a miss) first
        if len(u) and len(p):  # planted: an uncertain alarm, then alarms on every later second (those inside the period are ignored)
            a = np.zeros(len(ty), int)  # silent
            a[u[-1]:] = 1  # alarm from the last uncertain second on
            seqs.append(a)  # kept
            b = np.zeros(len(ty), int)  # silent
            b[u[0]] = 1  # one uncertain alarm
            b[p[0]] = 1  # and one post-change alarm (inside the period when it falls within 30 s)
            b[p[-1]] = 1  # and one at the interval's last second
            seqs.append(b)  # kept
        for a in seqs:  # for loop over the sequences
            hr = hand_fc(ty, a, "onset", 100.0, env.m)  # by hand
            env.reset(options={"kind": "onset", "session": e["session"], "start": e["start"], "onset": e["onset"]})  # replayed
            g, i = 0.0, 0  # the environment's return; position
            while True:  # while loop through the episode
                _, r, term, trunc, _ = env.step(int(a[i]))  # one F-c step
                g, i = g + r, i + 1  # accumulated
                if term or trunc:  # the end
                    break  # done
            cf = -C.fc_cost(ty, a, "onset", 100.0, env.m)[0]  # the follow-up script's closed form
            st = C.make_env(train, "surrogate", 100.0)  # the training stack (Monitor)
            st.reset(options={"kind": "onset", "session": e["session"], "start": e["start"], "onset": e["onset"]})  # replayed
            i = 0  # position
            while True:  # while loop through the episode
                _, _, term, trunc, _ = st.step(int(a[i]))  # one step through the wrappers
                i += 1  # next
                if term or trunc:  # the end
                    break  # done
            mon = st.get_episode_rewards()[-1] * T.REWARD_SCALE  # Monitor's return, unscaled
            worst, n = max(worst, abs(hr - g), abs(hr - cf), abs(hr - mon)), n + 1  # tuple assignment: compared
    test("G1 F-c returns by hand = env.step = closed form = Monitor x 100 (planted sequences)", worst <= 1e-9 and n > 0, f"{n} sequences, largest difference {worst:.1e}")  # G1
    from stable_baselines3 import DQN as _DQN  # SB3's DQN, untrained, as a stand-in policy
    for arm in ("primary", "surrogate"):  # for loop over the arms: the closed-loop rollout equals stepping the environment with the same greedy policy
        e2 = C.FcEnv(train, arm, 30.0)  # F-c environment
        mdl = _DQN("MlpPolicy", C.make_env(train, arm, 30.0), seed=20260914, device="cpu", verbose=0, **T.DQN_KW)  # an untrained network (its random policy alarms often)
        bad, k = 0, 0  # disagreements; episodes
        for e in T.eval_episodes(e2)[::5]:  # for loop over every fifth episode
            acts, cost, oc, _ = C.rollout(mdl, e2, e)  # the follow-up script's batched rollout
            o, _ = e2.reset(options={"kind": e["kind"], "session": e["session"], "start": e["start"], "onset": e["onset"]})  # replayed step by step
            g, tak = 0.0, []  # return; actions
            while True:  # while loop: the greedy action on each observation as the agent sees it
                a_ = int(mdl.predict(T.signed_log(o)[None, :] if arm == "surrogate" else o[None, :], deterministic=True)[0][0])  # one greedy action
                tak.append(a_)  # kept
                o, r, term, trunc, _ = e2.step(a_)  # stepped
                g += r  # accumulated
                if term or trunc:  # the end
                    break  # done
            bad += int(abs(-g - cost) > 1e-9 or not np.array_equal(np.asarray(tak), acts[:len(tak)]))  # disagreement
            k += 1  # counted
        test(f"G1b closed-loop rollout = stepping the environment ({arm})", bad == 0 and k > 0, f"{k} episodes, {bad} disagreements")  # G1b
    rng = np.random.default_rng(T.SEED)  # the seed
    e_old, e_new = E.SpoofingDetectionEnv(train, "surrogate", 100.0), C.FcEnv(train, "surrogate", 100.0)  # the environment script's and F-c's environments
    bad, k = 0, 0  # disagreements; episodes
    for e in T.eval_episodes(e_old)[::4]:  # for loop over every fourth episode
        ty = e_old.mat["type"][e["rows"]]  # types
        a = ((ty == 1) & (rng.random(len(ty)) < 0.05)).astype(int)  # random alarms on post-change seconds only (no uncertain alarm)
        o1, _ = e_old.reset(options={k_: e[k_] for k_ in ("kind", "session", "start", "onset")})  # dict comprehension: the environment script's replay
        o2, _ = e_new.reset(options={k_: e[k_] for k_ in ("kind", "session", "start", "onset")})  # F-c's replay
        same, i = np.array_equal(o1, o2), 0  # first observations equal?
        while True:  # while loop: step both
            r1 = e_old.step(int(a[i]))  # 11
            r2 = e_new.step(int(a[i]))  # F-c
            same &= np.array_equal(r1[0], r2[0]) and r1[1] == r2[1] and r1[2] == r2[2] and r1[3] == r2[3]  # observation, reward and flags equal
            i += 1  # next
            if r1[2] or r1[3]:  # the end
                break  # done
        bad, k = bad + int(not same), k + 1  # tuple assignment: counted
    test("G2 F-c equals the environment script's environment step by step when no uncertain alarm is raised", bad == 0 and k > 0, f"{k} episodes, {bad} differing")  # G2
    teps = T.eval_episodes(env)  # P4-rule episodes of the training sessions
    hi, _ = C.fit_hstar(C.cusum_paths(env, teps), 100.0, train["m"])  # h*
    dm = C.demo_set(C.FcEnv(train, "surrogate", 100.0), teps, float(C.H_EP[hi]))  # the demonstrations
    tot = dm["rew"].sum() * T.REWARD_SCALE  # every demonstration reward, unscaled
    ref = sum(C.cusum_cost(*C.fc_cusum(ty, z, kd), 100.0, train["m"])[hi] for ty, z, kd in C.cusum_paths(env, teps))  # generator in sum(): CUSUM(h*)'s costs by fc_cusum
    sess = {e["session"] for e in teps}  # set comprehension: the demonstration sessions
    test("G3 demonstrations: their total cost equals fc_cusum at h*; no held-out or sealed session", abs(-tot - ref) <= 1e-6 * max(1.0, abs(ref)) and not (sess & {spec["outer"], spec["inner"], C.SEALED}), f"{len(dm['act'])} transitions, h* = {C.H_EP[hi]:g}, total cost {ref:.3f} against {-tot:.3f}; sessions {sorted(sess)}")  # G3
    mdl = C.DQfD("MlpPolicy", C.make_env(train, "surrogate", 100.0), seed=20260914, device="cpu", verbose=0, **T.DQN_KW)  # Run B's agent
    mdl.set_demos(dm, 20260914, ())  # its memory
    sizes = []  # minibatch sizes drawn from the agent's buffer
    orig = mdl.replay_buffer.sample  # SB3's sampler
    mdl.replay_buffer.sample = lambda n, env=None: (sizes.append(n), orig(n, env=env))[1]  # lambda: records each draw's size, then draws
    mdl.learn(total_timesteps=700)  # 199 updates after learning starts at 500
    test("G4a every Run B minibatch holds 48 own and 16 demonstration rows", len(sizes) > 0 and set(sizes) == {C.BATCH - C.DEMO}, f"{len(sizes)} minibatches, own sizes {sorted(set(sizes))}")  # G4a
    idx = np.arange(C.DEMO)  # a fixed demonstration batch
    import torch as _th  # PyTorch
    with _th.no_grad():  # with-block: no gradients
        qd = mdl.q_net(mdl.d["obs"][idx])  # Q of the batch
        got = float(mdl._demo_terms(qd, idx))  # the follow-up script's lambda1 J_n + lambda2 J_E
        q, ad = qd.numpy().astype(float), dm["act"][idx]  # numpy copies
        je = np.mean([max(q[i, a] + (0.0 if a == ad[i] else C.MARGIN) for a in (0, 1)) - q[i, ad[i]] for i in range(len(idx))])  # J_E by hand
        qt = mdl.q_net_target(_th.as_tensor(dm["nboot"][idx])).numpy().astype(float).max(axis=1)  # the target network's largest value at the bootstrap states
        tn = dm["nret"][idx] + (1 - dm["ndone"][idx]) * qt  # the 10-step targets
        x = q[np.arange(len(idx)), ad] - tn  # the errors
        jn = np.mean(np.where(np.abs(x) < 1, 0.5 * x ** 2, np.abs(x) - 0.5))  # the Huber loss by hand (smooth L1, beta 1)
    test("G4b J_E and J_n on a fixed batch equal a hand computation", abs(got - (C.LAM_N * jn + C.LAM_E * je)) <= 1e-5, f"the follow-up script {got:.6f}, by hand {C.LAM_N * jn + C.LAM_E * je:.6f}")  # G4b
    caught = 0  # planted leaks caught
    for bad_s in (spec["outer"], C.SEALED):  # for loop over a planted held-out and a planted sealed session
        try:  # try block: the guard must stop
            C.guard(spec, [{"session": bad_s}])  # a planted episode
        except SystemExit:  # except: stopped
            caught += 1  # caught
    test("G6 planted held-out and 2.6.12 episodes caught by the guard", caught == 2, f"{caught} of 2")  # G6
    C.FOLLOWUP_DIR = scratch / "followup_g5"  # G5: jobs written to a scratch folder (module attribute, read by rundir at call time)
    jobs = [{"spec": {"stage": "outer", "receiver": rx, "outer": o, "inner": "", "arm": arm, "lambda": lam, "seed": 20260914, "budget": 20_000}, "stop": 20_000} for rx, o, arm, lam in (("SBF_02", "2.6.5", "primary", 30.0), ("SBF_03", "2.6.10", "surrogate", 300.0), ("UBX_01", "2.6.4", "surrogate", 10.0), ("SBF_02", "2.6.11", "primary", 1000.0))]  # list comprehension: four short outer jobs
    jobs.append({"spec": {**spec, "lambda": 30.0}, "stop": T.SCHEDULE})  # dict unpacking: one inner job (the snapshot statistics path)
    same = []  # per run and job: byte-identical outputs
    for run in ("A", "B"):  # for loop over the runs
        out = {}  # the two passes' bytes
        for k in (1, 2):  # for loop over two passes
            for j in jobs:  # for loop over the jobs
                for sub in ("jobs", "models"):  # the folders
                    (C.rundir(run) / sub).mkdir(parents=True, exist_ok=True)  # created
                jid = T.job_id(j["spec"])  # its name
                for p_ in (C.rundir(run) / "jobs" / f"{jid}.json.gz", C.rundir(run) / "jobs" / f"{jid}.sha256", C.rundir(run) / "models" / f"{jid}.npz"):  # earlier files removed, so the job really runs again
                    p_.unlink(missing_ok=True)  # removed
                C.run_job(j, run)  # run in this process
                f1, f2 = C.rundir(run) / "jobs" / f"{jid}.json.gz", C.rundir(run) / "models" / f"{jid}.npz"  # tuple assignment: its output and model
                out.setdefault(jid, []).append((f1.read_bytes(), f2.read_bytes() if f2.exists() else b""))  # kept
        same += [v[0] == v[1] for v in out.values()]  # list comprehension: pass 1 against pass 2
    C.FOLLOWUP_DIR = D.DIAG_DIR / "followup"  # restored
    test("G5 ten jobs (five per run) run again byte-identically", all(same) and len(same) == 10, f"{sum(same)} of {len(same)}")  # G5
    E.write_csv(pd.DataFrame(TESTS), DIAG_DIR / "followup_gates.csv")  # the outcomes


def main() -> None:  # function: the command line
    ap = argparse.ArgumentParser(description="tests of 6_reinforcement_learning/04_diagnosis.py")  # the parser
    ap.add_argument("--scratch", required=True, help="a scratch folder for T2's regenerated files")  # T2's folder
    ap.add_argument("--part2", action="store_true", help="the follow-up's gates G1-G6 only")  # the follow-up's gates
    a = ap.parse_args()  # parsed
    if a.part2:  # if: the follow-up's gates
        followup_gates(Path(a.scratch))  # run
        if not all(t["pass"] for t in TESTS):  # if: any gate failed
            raise SystemExit("STOP: a follow-up gate failed (followup_gates.csv)")  # stop
        return  # done
    t3_arithmetic()  # T3
    t4_h2()  # T4
    t5_negative()  # T5
    t2_regen_sample(Path(a.scratch))  # T2
    t1_repeat()  # T1 last: it runs again the whole diagnosis
    E.write_csv(pd.DataFrame(TESTS), DIAG_DIR / "diagnosis_tests.csv")  # the outcomes
    if not all(t["pass"] for t in TESTS):  # if: any test failed
        raise SystemExit("STOP: a test failed (diagnosis_tests.csv)")  # stop


if __name__ == "__main__":  # if: run as a script
    main()  # the command line
