#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
The tests of the environment and the stream. Every test must pass.

  V1 hand-computed returns from an independent closed form over episodes.csv.gz (step types read from steps.csv.gz, never from the
     environment) against the environment's returns, to 1e-12, undiscounted and at 0.999: never alarm; alarm at the first step; alarm at the
     onset; alarm 1, 10 and 60 s after the onset; the onset oracle; alarm in every uncertain step (must equal never alarm: the masking);
     alarm always. Lambda 100 on the full table and all five lambdas on the first 200 episodes of every receiver and fold; the ordering
     of the policies under the training discount (1) must equal the undiscounted cost's; the
     orderings under 0.99, 0.999 and 0.9999 are a reported table (gamma_sensitivity.csv).
  V2 P's committed outer predictions at p >= 0.5 through the stream in decision mode reproduce the committed per-fold figures exactly
     (logistic_regression_b, random_forest metrics and latency); stopping-mode figures and the decidable-seconds column are reported.
  V3 episode check: counts; no episode crosses an excluded, missing-window or invalid second, a run or a session; none touches 2.6.12 or
     the held-out session; the attack runs recounted independently from labels/labels_intervals_final.csv.
  V4 leakage: states rebuilt from the adopted-pipeline script's objects with the windows read off the raw index (window end <= t) equal the written matrices
     bitwise; transform fingerprints identical and fitted without the held-out session; every surrogate training score comes from the
     inner fit that held its session out (its training-row count recounted from the index); the shares of uncertain and clean valid steps
     per session; the six no-screen inner fits' scores beside the outer models'.
  V5 monotonicity on TRAINING episodes: the cost-minimising CUSUM threshold h*(lambda) is non-decreasing and the false-alarm termination
     share non-increasing in lambda, through the environment.
  V6 negative tests: a reward with the masked alarm made terminal, a state with a future window and a 2.6.12 row are each caught.
  V7 gymnasium's env_checker and Stable-Baselines3's check_env pass on both arms.
V8 (the environment script and these tests run twice give byte-identical outputs) is checked outside this script.
Reads   Data/processed/rl_environment/*, logistic_regression/ and random_forest/ predictions, metrics and latency, labels/labels_intervals_final.csv; rebuilds P through the adopted-pipeline script once
Writes  Data/processed/rl_environment/checks.csv and returns.csv, baselines.csv, episode_counts.csv, label_shares.csv,
        noscreen_scores.csv, cusum.csv; Graph/hand_returns.png
"""
from __future__ import annotations  # future import: type hints kept as text (the project's convention)

import importlib.util  # standard library: loads the environment script (its name starts with a digit)
import sys  # standard library: module registry and exit status
import warnings  # standard library: Stable-Baselines3's checker reports through warnings
from pathlib import Path  # standard library: paths

import numpy as np  # numpy: arrays
import pandas as pd  # pandas: tables

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
_spec = importlib.util.spec_from_file_location("environment_for_tests", PROJECT / "codes" / "6_reinforcement_learning" / "01_environment.py")  # where 11 is
E = importlib.util.module_from_spec(_spec)  # an empty module object
sys.modules["environment_for_tests"] = E  # registered
_spec.loader.exec_module(E)  # the environment script's top level (it loads the adopted-pipeline script); its main() is not run
A, M, SC = E.A, E.M, E.SC  # the adopted-pipeline script, 09 and the scoring module
ENV_DIR, PROC, GRAPH = E.ENV_DIR, E.PROC, E.GRAPH  # folders
RXS, SESSIONS, SEALED = E.RXS, E.SESSIONS, E.SEALED  # receivers, folds, "2.6.12"
TOL = 1e-12  # V1's tolerance
GAMMA_CHECK = 0.999  # the discount first registered (a discount below 1, replaced before training); V1 still replays it, so the closed form's discounting stays tested
GAMMAS = (1.0, GAMMA_CHECK)  # the two discounts V1 replays through the environment
GAMMA_SENS = (0.99, 0.9999)  # report only: the ordering under two other discounts, from the closed form (validated against the environment at 1 and at GAMMA_CHECK)
LAM_FULL = 100.0  # V1's lambda on the full episode table
N_SUB = 200  # V1: episodes per receiver and fold that run under every lambda
N_V5 = 400  # V5: training episodes per receiver and fold
H_GRID = (0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, np.inf)  # tuple: V5's CUSUM thresholds
POLICIES = ("never", "first_step", "at_onset", "onset+1", "onset+10", "onset+60", "oracle", "uncertain_all", "always")  # tuple: V1's policies
CHECKS: list = []  # the results


def check(test: str, name: str, ok, detail: str) -> None:  # function: records one check
    CHECKS.append({"test": test, "check": name, "pass": bool(ok), "detail": detail})  # one row
    print(f"{test} {'PASS' if ok else 'FAIL'} {name}: {detail}")  # printed


# ----------------------------------------------------------------------------------------------------------------------
# shared readers
# ----------------------------------------------------------------------------------------------------------------------
def steps() -> pd.DataFrame:  # function: the step table written by the environment script
    return pd.read_csv(ENV_DIR / "steps.csv.gz", dtype={"session": str})  # read


def episodes() -> pd.DataFrame:  # function: the episode table written by the environment script
    return pd.read_csv(ENV_DIR / "episodes.csv.gz", dtype={"session": str, "fold": str})  # read


def types_of(st: pd.DataFrame) -> dict:  # function: (receiver, session) -> (first second, step-code array by second), from the step table (not the environment)
    code = {"pre": 0, "post": 1, "uncertain": 2, "excluded": 3}  # dict: step -> code (excluded never inside an episode)
    return {(rx, s): (int(d.sec.min()), d.step.map(code).to_numpy(np.int8)) for (rx, s), d in st.groupby(["receiver", "session"], sort=False)}  # dict comprehension


def ep_types(T: dict, e) -> np.ndarray:  # function: the step codes of one episode's seconds, start .. end
    t0, arr = T[(e.receiver, e.session)]  # the session's array
    return arr[e.start - t0:e.end - t0 + 1]  # its slice


# ----------------------------------------------------------------------------------------------------------------------
# V1: hand returns
# ----------------------------------------------------------------------------------------------------------------------
def alarm_positions(policy: str, ty: np.ndarray, a: int, kind: str) -> np.ndarray:  # function: the positions at which a policy alarms (open loop, from the step types)
    n = len(ty)  # episode length
    if policy == "never":  # never alarm
        return np.zeros(0, int)  # none
    if policy == "first_step":  # alarm at the first step only
        return np.array([0])  # position 0
    if policy in ("at_onset", "oracle"):  # alarm at the onset (the oracle does so in onset episodes only, as does at_onset)
        return np.array([a]) if kind == "onset" else np.zeros(0, int)  # conditional expression
    if policy.startswith("onset+"):  # alarm k seconds after the onset
        k = int(policy.split("+")[1])  # k
        return np.array([a + k]) if kind == "onset" and a + k < n else np.zeros(0, int)  # conditional expression: inside the episode, else never
    if policy == "uncertain_all":  # alarm in every uncertain second
        return np.flatnonzero(ty == 2)  # those positions
    return np.arange(n)  # always


def hand_return(policy: str, ty: np.ndarray, a: int, kind: str, lam: float, m: float, gamma: float) -> float:  # function: the closed-form discounted return (independent of the environment)
    pos = alarm_positions(policy, ty, a, kind)  # where it alarms
    stop = [j for j in pos if ty[j] in (0, 1)]  # list comprehension: alarms that end the episode (pre- or post-change second); uncertain alarms are masked
    j = stop[0] if stop else None  # the terminating alarm, if any
    g = gamma ** np.arange(len(ty))  # discount per position
    end = j if j is not None else len(ty) - 1  # the last position reached
    R = -float(np.sum(g[:end + 1][ty[:end + 1] == 1]))  # -1 for every post-change second reached ...
    if j is not None and ty[j] == 1:  # ... except the alarming one, which costs 0
        R += float(g[j])  # its -1 removed
    if j is not None and ty[j] == 0:  # a false alarm
        R -= lam * float(g[j])  # -lambda there
    if j is None and kind == "onset":  # a miss: the interval ended in silence
        R -= m * float(g[len(ty) - 1])  # -m at the last second
    return R  # the return


def env_return(env, e, policy: str, ty: np.ndarray, gamma_list: tuple) -> tuple:  # function: the environment's discounted returns under an open-loop policy (replayed episode)
    env.reset(options={"kind": e.kind, "session": e.session, "start": int(e.start), "onset": int(e.onset)})  # the episode replayed
    al = set(alarm_positions(policy, ty, int(e.onset_pos), e.kind).tolist())  # alarm positions
    out, j, done = [0.0] * len(gamma_list), 0, False  # returns per gamma; position; finished
    while not done:  # while loop: until terminated or truncated
        _, r, term, trunc, _ = env.step(1 if j in al else 0)  # one step
        for q, gm in enumerate(gamma_list):  # for loop over the discounts
            out[q] += (gm ** j) * r  # discounted reward
        done, j = term or trunc, j + 1  # tuple assignment: finished? next position
    if j != len(ty) and not term:  # if: truncated before the episode's end (cannot)
        raise SystemExit("STOP: an episode ended early")  # stop
    return tuple(out)  # the returns


def v1(ep: pd.DataFrame, T: dict, fo: pd.DataFrame) -> pd.DataFrame:  # function: V1 over the episode table
    rows = []  # one row per receiver, fold, policy, lambda
    pts = []  # (hand, env) pairs for the figure
    for (rx, held), d in ep.groupby(["receiver", "fold"], sort=False):  # for loop over the 15 tables
        mat = E.material(rx, held)  # the environment's material
        m = float(fo[(fo.receiver == rx) & (fo.held_out == held)].m.iloc[0])  # the miss penalty (from the folds table)
        for lam in E.LAMBDAS:  # for loop over the lambdas
            env = E.SpoofingDetectionEnv(mat, "primary", lam)  # the environment (returns do not depend on the arm)
            sub = d if lam == LAM_FULL else d.head(N_SUB)  # the full table at lambda 100, 200 episodes otherwise
            for pol in POLICIES:  # for loop over the policies
                h1, h2, e1, e2, hs = [], [], [], [], {g_: [] for g_ in GAMMA_SENS}  # hand and env returns per gamma; dict comprehension: hand returns under the report-only discounts
                for e in sub.itertuples():  # for loop over the episodes
                    ty = ep_types(T, e)  # step codes from the step table
                    if len(ty) != e.steps or (ty == 3).any():  # if: the episode's seconds disagree with its length, or hold an excluded second
                        raise SystemExit(f"STOP: episode {rx} {held} {e.episode} does not match the step table")  # stop
                    if pol in ("at_onset", "onset+1", "onset+10", "onset+60") and e.kind != "onset":  # onset policies apply to onset episodes only
                        continue  # skipped
                    h1.append(hand_return(pol, ty, e.onset_pos, e.kind, lam, m, 1.0))  # hand, gamma 1
                    h2.append(hand_return(pol, ty, e.onset_pos, e.kind, lam, m, GAMMA_CHECK))  # hand, gamma 0.999
                    for g_ in GAMMA_SENS:  # for loop over the report-only discounts
                        hs[g_].append(hand_return(pol, ty, e.onset_pos, e.kind, lam, m, g_))  # hand only
                    a, b = env_return(env, e, pol, ty, GAMMAS)  # environment, both gammas
                    e1.append(a)  # kept
                    e2.append(b)  # kept
                h1, h2, e1, e2 = map(np.array, (h1, h2, e1, e2))  # arrays
                # dict literal over two lines: the comparison
                rows.append({"receiver": rx, "fold": held, "lambda": lam, "policy": pol, "episodes": len(h1), "max_abs_diff_g1": float(np.max(np.abs(h1 - e1))) if len(h1) else 0.0, "max_abs_diff_g0999": float(np.max(np.abs(h2 - e2))) if len(h1) else 0.0,
                             "mean_return_g1": float(h1.mean()) if len(h1) else np.nan, "mean_return_g0999": float(h2.mean()) if len(h1) else np.nan,  # dict literal continued: the mean returns
                             **{f"mean_return_g{str(g_).replace('.', '')}": float(np.mean(hs[g_])) if len(h1) else np.nan for g_ in GAMMA_SENS}})  # dict comprehension unpacked: the report-only means
                if lam == LAM_FULL:  # the figure's points
                    pts.append(pd.DataFrame({"policy": pol, "hand_g1": h1, "env_g1": e1, "hand_g0999": h2, "env_g0999": e2}))  # kept
    return pd.DataFrame(rows), pd.concat(pts, ignore_index=True)  # the table and the points


def ordering_same(r: pd.DataFrame, col: str = "mean_return_g0999") -> tuple:  # function: is the order of the policies' mean returns under the discount in col the same as under gamma 1?
    bad = []  # cases that differ
    for (rx, held, lam), d in r.groupby(["receiver", "fold", "lambda"], sort=False):  # for loop over the cases
        d = d.set_index("policy")  # by policy
        x1, x2 = d.mean_return_g1.to_numpy(), d[col].to_numpy()  # mean returns
        s1 = np.sign(np.round(x1[:, None] - x1[None, :], 9))  # pairwise order under gamma 1 (ties within 1e-9 are ties)
        s2 = np.sign(np.round(x2[:, None] - x2[None, :], 9))  # under the other discount
        if not np.array_equal(s1, s2):  # if: the order differs
            bad.append(f"{rx} {held} {lam}")  # recorded
    return not bad, bad  # verdict and the differing cases


# ----------------------------------------------------------------------------------------------------------------------
# V2: the baselines through the stream
# ----------------------------------------------------------------------------------------------------------------------
def rt(name: str) -> pd.DataFrame:  # function: a committed table with exact float parsing (the regression script's reader uses pandas' default parser, which can lose the last bit)
    df = pd.read_csv(PROC / name, float_precision="round_trip", dtype={"test_code": str, "held_out": str})  # round_trip: each written number back to its double
    if (df.held_out.astype(str) == SEALED).any():  # if: a 2.6.12 row (none is committed)
        raise SystemExit(f"STOP: 2.6.12 in {name}")  # stop
    return df  # the table


def v2(st: pd.DataFrame) -> pd.DataFrame:  # function: P's committed predictions through the stream
    rows, bad, lat_bad, ivs = [], [], [], []  # results; mismatches; per-interval rows
    valid = st.set_index(["receiver", "session", "sec"]).valid  # decidable = a sequential detector has a state
    for mdl, pf, mf, lf in (("lr", "logistic_regression/logistic_regression_predictions_primary.csv.gz", "logistic_regression/logistic_regression_metrics_primary.csv", "logistic_regression/logistic_regression_latency_primary.csv"), ("rf", "random_forest/random_forest_predictions_primary.csv.gz", "random_forest/random_forest_metrics_primary.csv", "random_forest/random_forest_latency_primary.csv")):  # for loop over the models
        pr, me, la = rt(pf), rt(mf), rt(lf)  # committed predictions, metrics, latency, every float parsed exactly
        for (rx, held), d in pr.groupby(["receiver", "held_out"], sort=False):  # for loop over receivers and folds
            d = d.sort_values("sec")  # time order
            al = (d.p >= 0.5).to_numpy()  # the registered threshold
            f = E.stream(d.sec, d.label, al, held, "decision")  # decision mode
            c = me[(me.receiver == rx) & (me.held_out == held)].iloc[0]  # the committed row
            for k in ("recall", "precision", "f1", "tp", "fn", "fp", "tn", "clean_in_alarm", "fa_episodes"):  # for loop over the figures V2 must reproduce
                a, b = f[k], c[k]  # new and committed
                same = (pd.isna(a) and pd.isna(b)) or (not pd.isna(a) and not pd.isna(b) and float(a) == float(b))  # exact, NaN equal to NaN
                if not same:  # if: they differ
                    bad.append(f"{mdl} {rx} {held} {k}: {a} vs {b}")  # recorded
            cl = la[(la.receiver == rx) & (la.held_out == held)].sort_values("onset")  # committed latency rows
            for x, y in zip(f["latency"], cl.itertuples()):  # for loop over the intervals
                if x["detected"] != bool(y.detected) or (x["detected"] and x["delay_s"] != y.latency_s):  # if: detection or delay differs
                    lat_bad.append(f"{mdl} {rx} {held} {x['onset']}")  # recorded
            s = E.stream(d.sec, d.label, al, held, "stopping")  # stopping mode (new)
            ev = E.events(d.sec.to_numpy(np.int64), al)  # the stopping-mode alarm events
            for x, y in zip(f["latency"], s["latency"]):  # for loop over the pooled intervals, both modes
                pre = d.sec.to_numpy()[ev & (d.sec.to_numpy() < x["onset"])]  # events before the onset
                ivs.append({"model": mdl, "receiver": rx, "held_out": held, "onset": x["onset"], "end": x["end"], "length_s": x["end"] - x["onset"] + 1, "detected_decision": x["detected"], "delay_decision_s": x["delay_s"], "detected_stopping": y["detected"], "delay_stopping_s": y["delay_s"], "last_event_before_onset_s": int(x["onset"] - pre.max()) if len(pre) else np.nan})  # one row per interval
            dm = valid.loc[[(rx, held, int(t)) for t in d.sec]].to_numpy()  # decidable seconds
            fd = E.stream(d.sec, d.label, al, held, "decision", decidable=dm)  # decision mode on the decidable seconds only
            for mode, x in (("decision", f), ("stopping", s), ("decision_decidable", fd)):  # for loop over the three readings
                rows.append({"model": mdl, "receiver": rx, "held_out": held, "mode": mode, **{k: x[k] for k in ("seconds", "clean", "attack", "alarms", "clean_in_alarm", "clean_share", "alarms_uncertain", "recall", "precision", "f1", "fa_figure", "fa_episodes", "fa_rate_per_h", "intervals", "detected", "delay_median_s", "early_warning")}})  # dict unpacking: the figures
    check("V2", "decision mode reproduces the committed per-fold recall, precision, F1, cells, clean alarms and false-alarm episodes", not bad, f"{len(bad)} mismatches {bad[:3]}")  # recorded
    check("V2", "decision mode reproduces the committed per-interval detection and latency", not lat_bad, f"{len(lat_bad)} mismatches {lat_bad[:3]}")  # recorded
    E.write_csv(pd.DataFrame(ivs), ENV_DIR / "intervals.csv")  # per interval, both modes
    return pd.DataFrame(rows)  # the table


# ----------------------------------------------------------------------------------------------------------------------
# V3: episode check
# ----------------------------------------------------------------------------------------------------------------------
def sealed_ok(df: pd.DataFrame) -> bool:  # function: no 2.6.12 value in any session or fold column
    return not any((df[c].astype(str) == SEALED).any() for c in ("session", "fold", "held_out", "test_code") if c in df.columns)  # generator in any(): any column


def v3(ep: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:  # function: the episode check
    arr = {(rx, s): (int(d.sec.min()), d.valid.to_numpy(), d.run_id.to_numpy(), d.step.to_numpy(object)) for (rx, s), d in st.groupby(["receiver", "session"], sort=False)}  # dict comprehension: arrays by second
    bad = []  # failing episodes
    for e in ep.itertuples():  # for loop over every episode
        t0, va, ru, sp = arr[(e.receiver, e.session)]  # its session's arrays
        sl = slice(e.start - t0, e.end - t0 + 1)  # its seconds
        if not (va[sl].all() and len(set(ru[sl])) == 1 and (e.end - e.start + 1) == e.steps):  # if: an invalid second, a run boundary or a gap
            bad.append(f"{e.receiver} {e.fold} {e.episode}: invalid or boundary")  # recorded
        if e.session == e.fold:  # the held-out session
            bad.append(f"{e.receiver} {e.fold} {e.episode}: held-out session")  # recorded
        if e.kind == "clean" and (sp[sl] == "post").any():  # a clean episode holding an attack second
            bad.append(f"{e.receiver} {e.fold} {e.episode}: attack second in a clean episode")  # recorded
        if e.kind == "onset" and not (sp[e.end - t0] == "post" and (e.end - t0 + 1 == len(sp) or sp[e.end - t0 + 1] != "post")):  # an onset episode not ending at its interval's last second
            bad.append(f"{e.receiver} {e.fold} {e.episode}: onset episode end")  # recorded
    check("V3", "no episode crosses an excluded, missing-window or invalid second, a window run or a session; none in the held-out session", not bad, f"{len(ep)} episodes, {len(bad)} failing {bad[:3]}")  # recorded
    check("V3", "no episode touches 2.6.12", sealed_ok(ep), "session and fold columns")  # recorded
    iv = M.read_table(PROC / "labels" / "labels_intervals_final.csv")  # the label intervals (independent of the environment script's step table)
    iv = iv[iv.label.isin(A.ATTACK)].copy()  # the attack intervals
    t = lambda x: pd.to_datetime("2026-09-14 " + x, utc=True).astype("int64") // 10**9  # lambda: "HH:MM:SS" -> unix seconds
    iv["a"], iv["b"] = t(iv.start_utc), t(iv.end_utc)  # start and end seconds
    cnt = {}  # (receiver, session) -> attack runs
    for (rx, s), d in iv.sort_values("a").groupby(["receiver", "test_code"], sort=False):  # for loop over the pairs
        n, end = 0, None  # runs; end of the current run
        for a, b in zip(d.a, d.b):  # for loop over the intervals in time order
            if end is None or a != end + 1:  # a new run unless the interval continues the previous one
                n += 1  # counted
            end = b  # remembered
        cnt[(rx, s)] = n  # kept
    on = pd.read_csv(ENV_DIR / "onsets.csv", dtype={"session": str})  # the environment script's attack runs
    mine = on.groupby(["receiver", "session"]).size().to_dict()  # its counts
    check("V3", "attack runs recounted from labels/labels_intervals_final.csv equal onsets.csv", cnt == mine, f"{sum(cnt.values())} runs from the intervals, {sum(mine.values())} in the environment script; usable {int(on.usable.sum())}, re-entries {int(on.reentry.sum())}")  # recorded
    return ep.groupby(["receiver", "fold", "session", "kind"], sort=False).agg(episodes=("episode", "size"), steps_mean=("steps", "mean"), steps_max=("steps", "max"), clean_full_length=("clean_full_length", "sum")).reset_index()  # counts


# ----------------------------------------------------------------------------------------------------------------------
# V4: leakage
# ----------------------------------------------------------------------------------------------------------------------
def window_end_ok(B: dict, ix: pd.DataFrame, rx: str, held: str, mats: dict) -> tuple:  # function: the written states against states read off the raw window index (windows ending t, t-10, t-20; t-29 .. t)
    idx, X = B["cells"][rx]  # the receiver's raw window index and values
    rec = B["outer"]["kept"][(rx, held)]  # the outer fit
    Z = A.transform(rec, X, B["wc10"])  # every window's 36 columns
    pos = {(t, int(s)): k for k, (t, s) in enumerate(zip(idx.test_code, idx.sec))}  # dict comprehension: (session, window end) -> raw row
    d = ix[(ix.receiver == rx) & (ix.fold == held)].sort_values("row")  # the fold's index
    P = mats["primary"]  # the written primary matrix
    w = [np.array([pos[(t, int(s) - lag)] for t, s in zip(d.session, d.sec)]) for lag in E.LAGS]  # list comprehension: raw rows of the windows ending t - lag
    prim = np.hstack([Z[r] for r in w])  # the primary rebuilt
    return np.array_equal(prim, P), prim  # equal? and the rebuilt matrix


def v4(B: dict, st: pd.DataFrame, ep: pd.DataFrame) -> tuple:  # function: the leakage tests and the two V4 reports
    ix = pd.read_csv(ENV_DIR / "state_index.csv.gz", dtype={"session": str, "fold": str})  # the state index
    npr = B["nested"]["predictions"]  # the inner predictions with decision values
    bad_w, bad_z = [], []  # failures
    for rx in RXS:  # for loop over the receivers
        idx = B["cells"][rx][0]  # the raw window index
        for held, train in A.S.folds():  # for loop over the folds
            mats = E.load_npz(ENV_DIR / f"states_{rx}_{held}.npz")  # the written matrices
            ok, _ = window_end_ok(B, ix, rx, held, mats)  # primary from the raw index
            if not ok:  # if: it differs
                bad_w.append(f"{rx} {held}")  # recorded
            d = ix[(ix.receiver == rx) & (ix.fold == held)].sort_values("row")  # the fold's index
            rec = B["outer"]["kept"][(rx, held)]  # the outer fit
            te = rec["te"]  # held-out rows
            zo = dict(zip(idx.sec.to_numpy()[te], A.decision(rec["model"]["lr"], A.transform(rec, B["cells"][rx][1].iloc[te], B["wc10"]))))  # dict: held-out second -> outer decision value
            zi = {(s, int(t)): v for s, t, v in npr[(npr.model == "lr") & (npr.receiver == rx) & (npr.outer_held_out == held)][["held_out", "sec", "dec"]].itertuples(index=False)}  # dict comprehension: (session, second) -> inner decision value of the fit that held that session out
            for lag in range(E.HIST + 1):  # for loop over the 30 positions (t-29 .. t)
                exp = np.array([zo[int(t) - lag] if s == held else zi[(s, int(t) - lag)] for s, t in zip(d.session, d.sec)])  # list comprehension: the score the source rule names
                if not np.array_equal(exp, mats["surrogate"][:, E.HIST - lag]):  # if: the written column differs
                    bad_z.append(f"{rx} {held} lag {lag}")  # recorded
    check("V4", "primary states equal the windows ending t, t-10 and t-20 read off the raw index (no window ends after t), bitwise", not bad_w, f"15 folds; failing {bad_w}")  # recorded
    check("V4", "surrogate states equal the decision values at t-29 .. t: outer model on the held-out session, the inner fit that held the session out on training sessions, bitwise", not bad_z, f"15 folds x 30 lags; failing {bad_z[:3]}")  # recorded
    fp = pd.read_csv(ENV_DIR / "transform_fingerprints.csv", dtype={"held_out": str})  # the written fingerprints
    _, _, fp2 = E.states(B, E.seconds_table(B["cells"]))  # recomputed from the rebuilt objects
    check("V4", "transform fingerprints identical to the rebuilt the adopted-pipeline script objects; no transform fitted on its held-out session", fp.equals(fp2.astype(fp.dtypes.to_dict())) and not fp.held_out_in_fit.any(), f"{len(fp)} transforms")  # recorded
    fits = pd.DataFrame(B["nested"]["fits"])  # the inner fits
    bad_n = []  # failures
    for f in fits.itertuples():  # for loop over the 60 inner fits
        idx = B["cells"][f.receiver][0]  # the receiver's index
        itrain = [s for s in SESSIONS if s not in (f.outer_held_out, f.inner_held_out)]  # the three sessions the fit may use
        n = int((idx.test_code.isin(itrain) & idx.tlabel.isin(A.SCORED)).sum())  # their scored windows under the training label
        if n != f.train_rows:  # if: the fit used other rows (the held-out inner session included, for example)
            bad_n.append(f"{f.receiver} {f.outer_held_out} {f.inner_held_out}: {f.train_rows} vs {n}")  # recorded
    check("V4", "every inner fit was trained on exactly the scored windows of the other three sessions (no surrogate training score from a model that saw its session)", not bad_n, f"{len(fits)} inner fits; failing {bad_n[:3]}")  # recorded
    for arm in ("primary", "surrogate"):  # for loop over the arms
        dims = {E.load_npz(ENV_DIR / f"states_{rx}_{h}.npz")[arm].shape[1] for rx in RXS for h, _ in A.S.folds()}  # set comprehension: column counts
        check("V4", f"{arm} state holds only the registered columns (no step index, time, identity or label)", dims == ({108} if arm == "primary" else {30}), f"columns {sorted(dims)} (+ S_t appended by the environment for the surrogate); equality with the rebuilt windows above")  # recorded
    v = st[st.valid]  # valid steps
    shares = v.groupby(["receiver", "session"], sort=False).agg(valid=("sec", "size"), pre=("step", lambda s: int((s == "pre").sum())), post=("step", lambda s: int((s == "post").sum())), uncertain=("step", lambda s: int((s == "uncertain").sum())), scored_clean=("label", lambda s: int((s == "clean").sum()))).reset_index()  # lambdas: counts per type
    nv = st[(st.step != "excluded") & ~st.valid].groupby(["receiver", "session", "label"], sort=False).size().rename("no_state_s").reset_index()  # seconds without a state, by scoring label
    shares = shares.merge(nv.pivot_table(index=["receiver", "session"], columns="label", values="no_state_s", fill_value=0).add_prefix("no_state_").reset_index(), on=["receiver", "session"], how="left").fillna(0)  # beside the shares
    rows = []  # the no-screen diagnostic
    for f in fits[~fits.screen_applied].itertuples():  # for loop over the six fits with the screen not applied
        rx, k, s = f.receiver, f.outer_held_out, f.inner_held_out  # receiver, outer fold, inner held-out session
        idx, X = B["cells"][rx]  # its windows
        lab = idx.label.to_numpy(object)  # scoring labels
        q = npr[(npr.model == "lr") & (npr.receiver == rx) & (npr.held_out == s)]  # every inner fit that held s out (the no-screen one and the screened ones of other outer folds)
        src = [(f"inner, outer {o}, screen {'applied' if sa else 'NOT applied'}", dict(zip(g_.sec, g_.dec))) for (o, sa), g_ in q.groupby(["outer_held_out", "screen_applied"], sort=True)]  # list comprehension: each inner fit's scores
        r_out = B["outer"]["kept"][(rx, s)]  # the outer model that held s out (out of sample)
        r_in = B["outer"]["kept"][(rx, k)]  # the outer model of fold k, which trained on s (in sample)
        te = r_out["te"]  # s's rows
        secs = idx.sec.to_numpy()[te]  # their seconds
        src.append((f"outer, holding out {s} (out of sample)", dict(zip(secs, A.decision(r_out["model"]["lr"], A.transform(r_out, X.iloc[te], B["wc10"]))))))  # out-of-sample outer scores
        src.append((f"outer, fold {k} (trained on {s}: in sample)", dict(zip(secs, A.decision(r_in["model"]["lr"], A.transform(r_in, X.iloc[te], B["wc10"]))))))  # in-sample outer scores
        for name, sc in src:  # for loop over the sources
            for cl in ("attack_jam_spoof", "attack_spoof", "clean"):  # for loop over the scored labels
                vals = np.array([sc[t] for t, l in zip(secs, lab[te]) if l == cl and t in sc])  # list comprehension: the source's scores on those windows
                if len(vals):  # if: any
                    q1, md, q3 = np.percentile(vals, [25, 50, 75])  # quartiles
                    rows.append({"receiver": rx, "no_screen_fit_outer": k, "session": s, "source": name, "label": cl, "windows": len(vals), "median": float(md), "iqr": float(q3 - q1), "q1": float(q1), "q3": float(q3)})  # one row
    ns = pd.DataFrame(rows)  # the diagnostic
    check("V4", "the six no-screen inner fits are reported beside the outer models on the same windows", len(fits[~fits.screen_applied]) == 6 and len(ns) > 0, f"{len(ns)} rows in noscreen_scores.csv")  # recorded
    return shares, ns  # the two reports


# ----------------------------------------------------------------------------------------------------------------------
# V5: monotonicity of the reward under a CUSUM threshold
# ----------------------------------------------------------------------------------------------------------------------
def v5(ep: pd.DataFrame) -> pd.DataFrame:  # function: h*(lambda) and the false-alarm share on TRAINING episodes, through the environment (surrogate arm, lambda 0 plus the counted false alarms)
    rows = []  # per receiver, fold, h
    for (rx, held), d in ep.groupby(["receiver", "fold"], sort=False):  # for loop over the 15 tables
        env = E.SpoofingDetectionEnv(E.material(rx, held), "surrogate", 0.0)  # lambda 0: the return is delay and miss only; false alarms are counted from info
        sub = d.head(N_V5)  # the first 400 episodes
        for h in H_GRID:  # for loop over the thresholds
            ret, fa = 0.0, 0  # summed return; false-alarm terminations
            for e in sub.itertuples():  # for loop over the episodes
                obs, _ = env.reset(options={"kind": e.kind, "session": e.session, "start": int(e.start), "onset": int(e.onset)})  # replayed
                done = False  # finished?
                while not done:  # while loop: the CUSUM policy
                    obs, r, term, trunc, info = env.step(1 if obs[-1] >= h else 0)  # alarm when S_t reaches h
                    ret += r  # undiscounted
                    fa += info["event"] == "false_alarm"  # counted
                    done = term or trunc  # finished?
            rows.append({"receiver": rx, "fold": held, "h": h, "episodes": len(sub), "return_lambda0": ret, "false_alarms": fa})  # one row
    t = pd.DataFrame(rows)  # the table
    out = []  # h*(lambda)
    for (rx, held), d in t.groupby(["receiver", "fold"], sort=False):  # for loop over the cases
        for lam in E.LAMBDAS:  # for loop over the lambdas
            cost = -(d.return_lambda0 - lam * d.false_alarms) / d.episodes  # mean cost at this lambda (linear in lambda)
            j = int(np.argmin(cost.to_numpy()))  # the cost-minimising threshold (the smallest h among ties)
            out.append({"receiver": rx, "fold": held, "lambda": lam, "h_star": float(d.h.iloc[j]), "cost": float(cost.iloc[j]), "fa_share": float(d.false_alarms.iloc[j] / d.episodes.iloc[j])})  # one row
    o = pd.DataFrame(out)  # the table
    mono = all((np.diff(g_.h_star.to_numpy()) >= 0).all() and (np.diff(g_.fa_share.to_numpy()) <= 0).all() for _, g_ in o.groupby(["receiver", "fold"], sort=False))  # generator in all(): monotone in every case
    check("V5", "h*(lambda) non-decreasing and the false-alarm termination share non-increasing in lambda (training episodes, through the environment)", mono, f"{len(o) // len(E.LAMBDAS)} cases; h* range {o.h_star.min()} to {o.h_star.max()}")  # recorded
    return pd.concat([t.assign(table="grid"), o.assign(table="h_star")], ignore_index=True)  # both tables in one file


# ----------------------------------------------------------------------------------------------------------------------
# V6 and V7
# ----------------------------------------------------------------------------------------------------------------------
class TerminalMaskEnv(E.SpoofingDetectionEnv):  # class: a corrupted environment for V6: an alarm in an uncertain second ends the episode
    def step(self, action):  # method: the corrupted step
        typ = int(self.mat["type"][self.rows[self.i]])  # current step type
        if typ == E.STEP["uncertain"] and int(action) == 1:  # a masked alarm
            return self._obs_last(), 0.0, True, False, self._info(True, "corrupted")  # made terminal
        return super().step(action)  # otherwise unchanged


def v6(ep: pd.DataFrame, T: dict, fo: pd.DataFrame, B: dict) -> None:  # function: three negative tests, each must be caught
    rx, held = "SBF_02", "2.6.2"  # one receiver and fold
    d = ep[(ep.receiver == rx) & (ep.fold == held) & (ep.kind == "onset")].head(50)  # 50 onset episodes
    m = float(fo[(fo.receiver == rx) & (fo.held_out == held)].m.iloc[0])  # miss penalty
    env = TerminalMaskEnv(E.material(rx, held), "primary", LAM_FULL)  # the corrupted environment
    diff = max(abs(env_return(env, e, "uncertain_all", ep_types(T, e), (1.0,))[0] - hand_return("never", ep_types(T, e), e.onset_pos, e.kind, LAM_FULL, m, 1.0)) for e in d.itertuples())  # generator in max(): V1's masking test on the corrupted reward
    check("V6", "a reward with the masked alarm made terminal is caught by V1's masking test", diff > TOL, f"largest difference {diff:.1f}")  # recorded
    ix = pd.read_csv(ENV_DIR / "state_index.csv.gz", dtype={"session": str, "fold": str})  # the state index
    mats = E.load_npz(ENV_DIR / f"states_{rx}_{held}.npz")  # the written matrices
    idx, X = B["cells"][rx]  # raw windows
    rec = B["outer"]["kept"][(rx, held)]  # the outer fit
    Z = A.transform(rec, X, B["wc10"])  # every window's columns
    fi = ix[(ix.receiver == rx) & (ix.fold == held)].sort_values("row")  # the fold's index
    r0 = int(fi[(fi.sec + 1).isin(idx.sec[idx.test_code == fi.session.iloc[0]])].row.iloc[0])  # a row whose next window exists
    nxt = int(np.flatnonzero((idx.test_code == fi.session.iloc[0]).to_numpy() & (idx.sec == int(fi.sec.iloc[r0]) + 1).to_numpy())[0])  # the window ending one second later
    leaked = {**mats, "primary": mats["primary"].copy()}  # a copy
    leaked["primary"][r0, :36] = Z[nxt]  # the state at t given the window ending t+1
    ok, _ = window_end_ok(B, ix, rx, held, leaked)  # V4's window-end test on it
    check("V6", "a state built from a window ending after t is caught by V4's window-end test", not ok, f"row {r0}")  # recorded
    bad = ep.head(3).copy()  # a copy of three episodes
    bad.loc[bad.index[0], "session"] = SEALED  # one moved to 2.6.12
    check("V6", "a 2.6.12 row is caught by the sealed check", not sealed_ok(bad), "one episode row set to 2.6.12")  # recorded


def v7() -> None:  # function: gymnasium's and Stable-Baselines3's environment checkers on both arms
    from gymnasium.utils.env_checker import check_env as gym_check  # gymnasium's checker
    from stable_baselines3.common.env_checker import check_env as sb3_check  # SB3's checker
    for arm in ("primary", "surrogate"):  # for loop over the arms
        env = E.SpoofingDetectionEnv(E.material("SBF_02", "2.6.2"), arm, LAM_FULL)  # one fold's environment
        try:  # try: a failure raises
            with warnings.catch_warnings(record=True) as w1:  # with: gymnasium's warnings recorded
                warnings.simplefilter("always")  # all
                gym_check(env, skip_render_check=True)  # gymnasium's checker
            with warnings.catch_warnings(record=True) as w2:  # with: SB3's warnings recorded
                warnings.simplefilter("always")  # all
                sb3_check(env, warn=True)  # SB3's checker
            check("V7", f"gymnasium env_checker and SB3 check_env pass ({arm})", True, f"gymnasium warnings {sorted({str(x.message)[:80] for x in w1})}; SB3 warnings {sorted({str(x.message)[:80] for x in w2})}")  # recorded with the warnings
        except Exception as ex:  # any failure
            check("V7", f"gymnasium env_checker and SB3 check_env pass ({arm})", False, f"{type(ex).__name__}: {ex}")  # recorded


def plot_hand_returns(pts: pd.DataFrame) -> None:  # function: the figure, V1's hand returns against the environment's, on the identity line
    import matplotlib.pyplot as plt  # pyplot (09 chose the file-only backend)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))  # two panels: gamma 1 and gamma 0.999
    for ax, (h, e, gname) in zip(axes, (("hand_g1", "env_g1", "gamma = 1"), ("hand_g0999", "env_g0999", f"gamma = {GAMMA_CHECK}"))):  # for loop over the panels
        for pol, d in pts.groupby("policy", sort=False):  # for loop over the policies
            ax.scatter(d[h], d[e], s=6, alpha=0.5, label=pol)  # hand against environment
        lo, hi = float(min(pts[h].min(), pts[e].min())), float(max(pts[h].max(), pts[e].max()))  # the range
        ax.plot([lo, hi], [lo, hi], color="#000000", lw=0.8)  # the identity line
        ax.set_xlabel("hand-computed return (closed form)", fontsize=8)  # axis
        ax.set_ylabel("environment's return", fontsize=8)  # axis
        ax.set_title(f"{gname}, lambda = {LAM_FULL:.0f}; max |difference| {np.max(np.abs(pts[h] - pts[e])):.1e}", fontsize=9, loc="left")  # title with the largest difference
    axes[0].legend(fontsize=6, markerscale=2)  # legend once
    fig.tight_layout()  # layout
    M.save_fig(fig, GRAPH / "hand_returns.png")  # byte-stable save


def main() -> None:  # function: V1-V7
    st, ep = steps(), episodes()  # the step and episode tables
    T = types_of(st)  # step codes by second
    fo = pd.read_csv(ENV_DIR / "folds.csv", dtype={"held_out": str})  # per fold
    r1, pts = v1(ep, T, fo)  # V1
    check("V1", "hand returns = environment returns within 1e-12 (undiscounted and at 0.999, every policy, lambda 100 on the full table, all lambdas on 200 episodes)", (r1.max_abs_diff_g1 <= TOL).all() and (r1.max_abs_diff_g0999 <= TOL).all(), f"largest difference {max(r1.max_abs_diff_g1.max(), r1.max_abs_diff_g0999.max()):.2e} over {int(r1[r1['lambda'] == LAM_FULL].episodes.sum())} policy-episodes at lambda 100")  # recorded
    piv = r1.pivot_table(index=["receiver", "fold", "lambda"], columns="policy", values="mean_return_g1")  # mean returns by policy
    check("V1", "alarm in every uncertain step = never alarm (the masking)", np.allclose(piv["uncertain_all"], piv["never"], rtol=0, atol=TOL), "mean returns per receiver, fold and lambda")  # recorded
    train_col = {1.0: "mean_return_g1", GAMMA_CHECK: "mean_return_g0999"}[E.GAMMA_TRAIN]  # dict lookup: the mean-return column of the training discount
    ok, bad = ordering_same(r1, train_col)  # the ordering under the training discount against the undiscounted cost
    check("V1", f"the ordering of the policies under the training discount ({E.GAMMA_TRAIN:g}) equals the undiscounted cost's", ok, f"{len(bad)} cases differ {bad[:3]}")  # recorded
    sens = []  # report only: cases whose ordering differs from gamma 1, per discount
    for g_, col in ((0.99, "mean_return_g099"), (GAMMA_CHECK, "mean_return_g0999"), (0.9999, "mean_return_g09999")):  # for loop over the discounts
        _, b_ = ordering_same(r1, col)  # the differing cases
        sens.append({"gamma": g_, "cases": r1.groupby(["receiver", "fold", "lambda"]).ngroups, "cases_ordering_differs": len(b_), "which": "; ".join(b_)})  # one row
    E.write_csv(pd.DataFrame(sens), ENV_DIR / "gamma_sensitivity.csv")  # report only
    E.write_csv(r1, ENV_DIR / "returns.csv")  # V1's table
    plot_hand_returns(pts)  # the figure
    E.write_csv(v2(st), ENV_DIR / "baselines.csv")  # V2
    E.write_csv(v3(ep, st), ENV_DIR / "episode_counts.csv")  # V3
    B = A.build()  # P rebuilt once, for V4 and V6
    shares, ns = v4(B, st, ep)  # V4
    E.write_csv(shares, ENV_DIR / "label_shares.csv")  # V4: shares
    E.write_csv(ns, ENV_DIR / "noscreen_scores.csv")  # V4: the no-screen diagnostic
    E.write_csv(v5(ep), ENV_DIR / "cusum.csv")  # V5
    v6(ep, T, fo, B)  # V6
    v7()  # V7
    ck = pd.DataFrame(CHECKS)  # all checks
    E.write_csv(ck, ENV_DIR / "checks.csv")  # written
    print(f"environment tests: {int(ck['pass'].sum())}/{len(ck)} checks pass")  # summary
    if not ck["pass"].all():  # if: any check failed
        sys.exit(1)  # exit status 1


if __name__ == "__main__":  # guard: runs main() only when the file is executed, not when imported
    main()  # runs main
