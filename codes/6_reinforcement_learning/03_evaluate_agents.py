#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
Agent evaluation: deployable operating points, held-out runs, operating curves, aggregation and verdicts.

Two steps, so that the operating points are fixed before any held-out stream is read:
  --points   P6 as amended: the agent takes the SMALLEST lambda whose decision-mode inner clean-window share (the baselines' quantity)
             is at most the share; k-of-n the (n, k) with the lowest inner median delay among the settings whose inner share is at most
             the share. The logistic regression, the forest, CUSUM and k-of-n are computed by the training script's deployable() exactly
             as frozen with the agents; the stopping-mode shares stay beside the agent's (capped near 1/31 by the 30 s refractory period).
             Reads no held-out second (the agent tests' T5 proves it). Writes evaluation/deployable_points.csv.
  --heldout  refuses to run unless every frozen file equals its copy in the freeze commit and the points file is committed unchanged.
             Every detector runs once on each held-out session at its point (agent: five seeds), in decision and stopping mode through
             the environment script's stream; the full operating curves (P7); interquartile means with 95 % stratified bootstrap
             intervals (P8); the verdicts (P9); the figures.
Every frozen file is read with float_precision="round_trip". 2.6.12 is sealed. It imports the training script (which loads the
environment script and the adopted-pipeline script).
Reads   Data/processed/rl_agents/training/ (frozen), Data/processed/rl_environment/, the logistic regression and random forest outer
        predictions, Data/processed/rl_agents/models/training/
Writes  Data/processed/rl_agents/evaluation/*; Graph/inner_curves.png .. heldout_stream.png
"""
from __future__ import annotations  # future import: type hints kept as text (the project's convention)

import argparse  # standard library: the command line
import importlib.util  # standard library: loads the training script (its name starts with a digit)
import subprocess  # standard library: read-only git commands (the freeze check)
import sys  # standard library: module registry and exit status
from pathlib import Path  # standard library: paths

import numpy as np  # numpy: arrays
import pandas as pd  # pandas: tables

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
_spec = importlib.util.spec_from_file_location("train_agents_for_evaluation", PROJECT / "codes" / "6_reinforcement_learning" / "02_train_agents.py")  # where 12 is
T = importlib.util.module_from_spec(_spec)  # an empty module object
sys.modules["train_agents_for_evaluation"] = T  # registered
_spec.loader.exec_module(T)  # the training script's top level (it loads the environment script and the adopted-pipeline script); its main() is not run
E, M = T.E, T.E.M  # the environment script and the regression script
GRID, EVAL, MODELS = T.AGENT_DIR / T.GRID, T.AGENT_DIR / "evaluation", T.AGENT_DIR / "models" / T.GRID  # tuple assignment: the frozen grid, this step's folder, the frozen agents
GRAPH = E.GRAPH  # the figure folder
FREEZE_COMMIT = "416943248f57e987700e684180afe9104f11df2e"  # the freeze commit of the training
SEED, N_BOOT = T.SEED, 10_000  # tuple assignment: P8's bootstrap seed and resamples
SHARES, RXS, SESSIONS = T.SHARES, T.RXS, T.SESSIONS  # the shares, receivers and folds
BASE = ("lr", "rf", "cusum", "kofn")  # tuple: the baselines P9 chooses from
DET = BASE + ("agent_primary", "agent_surrogate")  # tuple: every detector, in a fixed order
# dict: detector -> (colour, marker, label); the Okabe-Ito palette in an order that passes the colour-vision checks, markers as second encoding
STYLE = {"lr": ("#0072B2", "o", "logistic regression"), "rf": ("#D55E00", "s", "random forest"), "cusum": ("#009E73", "^", "CUSUM"), "kofn": ("#E69F00", "v", "k of n"),
         "agent_primary": ("#CC79A7", "D", "agent, primary"), "agent_surrogate": ("#56B4E9", "P", "agent, surrogate")}  # dict literal continued
LR_GRID = np.round(np.linspace(0.0, 1.0, 101), 10)  # thresholds of the probability curves (P7)
H_GRID = np.r_[0.0, np.logspace(0, 4, 41)]  # np.r_ concatenation: CUSUM thresholds of the curve, 0 and 1 to 10,000 on a log scale
KN_GRID = np.round(np.linspace(-10.0, 15.0, 51), 10)  # k-of-n log-odds thresholds of the curve
DEC_KEYS = ("recall", "precision", "f1", "clean", "clean_in_alarm", "clean_share", "early_warning", "pre_onset_uncertain", "fa_episodes")  # tuple: decision-mode figures kept per run
STOP_KEYS = ("intervals", "detected", "delay_median_s", "clean", "clean_in_alarm", "alarms", "alarms_uncertain", "fa_figure", "fa_rate_per_h", "fa_upper95_per_h")  # tuple: stopping-mode figures kept per run


def rt(path: Path, **kw) -> pd.DataFrame:  # function: a frozen or committed table read with every float back to the double that was written
    return pd.read_csv(path, float_precision="round_trip", dtype={"outer": str, "held_out": str, "outer_held_out": str, "inner": str, "session": str}, **kw)  # sessions as text


# ----------------------------------------------------------------------------------------------------------------------
# --points: the deployable points (no held-out second is read)
# ----------------------------------------------------------------------------------------------------------------------
def points_table(sel: pd.DataFrame, nested: pd.DataFrame, steps: pd.DataFrame, material=E.material) -> tuple:  # function: P6 as amended; returns (points, k-of-n candidates)
    dp, kn = T.deployable(sel, nested, steps, material)  # the training script's computation: the baselines unchanged, the agent rows replaced below
    rows = [r for r in dp.to_dict("records") if not r["detector"].startswith("agent")]  # list comprehension: the baselines' points
    for r in rows:  # for loop over them
        if r["detector"] == "kofn":  # k-of-n: the kept setting must meet the share, as the amended rule says
            c = kn[(kn.receiver == r["receiver"]) & (kn.outer == r["outer"]) & (kn.share == r["share"]) & (kn.inner_clean_share <= r["share"])]  # the candidates meeting the share
            best = c.sort_values(["inner_median_delay", "n", "k"], kind="stable").iloc[0]  # the lowest inner median delay, ties to the smaller n, then k
            if (int(best.n), int(best.k)) != (r["n"], r["k"]):  # if: the training script's choice differs (it cannot: every candidate's h was set to meet the share)
                raise SystemExit(f"STOP: k-of-n choice differs for {r['receiver']} {r['outer']} {r['share']}")  # stop
    for (rx, outer, arm), a in sel.groupby(["receiver", "outer", "arm"], sort=False):  # for loop over the agent's cells
        a = a.sort_values("lambda")  # its five lambdas
        for s in SHARES:  # for loop over the shares
            ok = a[a.inner_decision_clean_share.notna() & (a.inner_decision_clean_share <= s)]  # the lambdas whose decision-mode inner share meets the share
            lam = float(ok["lambda"].min()) if len(ok) else None  # the smallest one (P6 as amended), or no point
            at = a.set_index("lambda").loc[lam] if lam is not None else None  # its row
            # dict literal over two lines: the agent's point, with the stopping-mode share beside it (capped near 1/31)
            rows.append({"receiver": rx, "outer": outer, "detector": f"agent_{arm}", "share": s, "param": lam, "param_name": "lambda (smallest whose decision-mode inner share meets the share, P6 as amended)", "inner_clean": int(a.inner_clean_s.iloc[0]),
                         "inner_clean_share": float(at.inner_decision_clean_share) if at is not None else None, "inner_stopping_clean_share": float(at.inner_stopping_clean_share) if at is not None else None, "budget": int(at.selected_budget) if at is not None else None, "no_point": lam is None})  # dict literal continued
    order = {d: i for i, d in enumerate(DET)}  # dict comprehension: detector -> position
    out = pd.DataFrame(rows)  # the table
    out = out.assign(_r=out.receiver.map({r: i for i, r in enumerate(RXS)}), _o=out.outer.map({o: i for i, o in enumerate(SESSIONS)}), _d=out.detector.map(order))  # dict comprehensions: sort keys in the project's orders
    return out.sort_values(["_r", "_o", "_d", "share"], kind="stable").drop(columns=["_r", "_o", "_d"]).reset_index(drop=True), kn  # the points in a fixed order


def evaluate_points() -> None:  # function: writes the amended points
    sel = rt(GRID / "selection.csv")  # the frozen selection
    nested = rt(GRID / "nested_predictions.csv.gz")  # P's nested out-of-fold predictions
    pts, _ = points_table(sel, nested, T.read_steps())  # the points
    old = rt(GRID / "deployable_points.csv")  # the frozen points
    base = pts[~pts.detector.str.startswith("agent")].reset_index(drop=True)  # the baselines now
    ob = old[~old.detector.str.startswith("agent")].reset_index(drop=True)  # and as frozen
    same = np.allclose(base.param.to_numpy(float), ob.param.to_numpy(float), rtol=0, atol=0, equal_nan=True) and base[["receiver", "outer", "detector", "share"]].equals(ob[["receiver", "outer", "detector", "share"]])  # identical, bitwise
    if not same:  # if: a baseline point moved (it must not)
        raise SystemExit("STOP: the baselines' points differ from the frozen ones")  # stop
    EVAL.mkdir(parents=True, exist_ok=True)  # the folder
    E.write_csv(pts, EVAL / "deployable_points.csv")  # written
    ag = pts[pts.detector.str.startswith("agent")]  # the agent's points
    print(ag.groupby(["detector", "share"]).agg(no_point=("no_point", "sum"), cells=("no_point", "size")))  # summary


# ----------------------------------------------------------------------------------------------------------------------
# the freeze check
# ----------------------------------------------------------------------------------------------------------------------
def git(*args) -> subprocess.CompletedProcess:  # function: a read-only git command in the folder; without a repository, the frozen-input table answers it
    from helpers import frozen_git, in_repository  # the table's reader and the repository test (codes/common/helpers.py, on the import path)
    if in_repository(PROJECT):  # if: the folder sits in a git repository with this history
        return subprocess.run(["git", *args], cwd=PROJECT, capture_output=True)  # its result (bytes)
    return frozen_git(PROJECT, args)  # the answer from Data/processed/checks/frozen_inputs.csv, in the same form


def check_freeze() -> pd.DataFrame:  # function: every frozen file equals its copy in the freeze commit; the amended points are committed and unchanged
    fm = pd.read_csv(GRID / "freeze_manifest.csv")  # the freeze manifest
    rows = []  # one row per file
    for f, h in zip(fm.file, fm.sha256):  # for loop over the frozen files
        rel = f"Data/processed/rl_agents/{T.GRID}/{f}"  # its path in the repository
        at = git("show", f"{FREEZE_COMMIT}:{rel}")  # its bytes in the freeze commit
        now = (GRID / f).read_bytes()  # its bytes now
        rows.append({"file": rel, "in_freeze_commit": at.returncode == 0, "equal_to_commit": at.stdout == now, "sha256_matches_manifest": T.sha_file(GRID / f) == h})  # the row
    rel = "Data/processed/rl_agents/evaluation/deployable_points.csv"  # the amended points
    committed = git("ls-files", "--error-unmatch", rel).returncode == 0 and git("diff", "--quiet", "HEAD", "--", rel).returncode == 0  # tracked and unchanged since the last commit
    rows.append({"file": rel, "in_freeze_commit": committed, "equal_to_commit": committed, "sha256_matches_manifest": committed})  # the row
    return pd.DataFrame(rows)  # the table


# ----------------------------------------------------------------------------------------------------------------------
# --heldout: held-out runs
# ----------------------------------------------------------------------------------------------------------------------
def heldout_material(rx: str, outer: str) -> dict:  # function: the environment script's material of the fold with its key over the held-out session's rows (agents and sequential baselines read these)
    mat = E.material(rx, outer)  # the environment script's material
    h = mat["index"][mat["index"].role == "heldout"]  # the held-out rows
    return {**mat, "key": {(s, int(t)): int(r) for s, t, r in zip(h.session, h.sec, h.row)}}  # dict comprehension: (session, second) -> row


def grid_rows(mat: dict, sd: pd.DataFrame) -> tuple:  # function: each windowed second's state row (-1: none), the validity mask, z_t and the 30 log-odds on the stream grid
    rows = np.array([mat["key"].get((s, int(t)), -1) for s, t in zip(sd.session, sd.sec)])  # list comprehension: rows
    ok = rows >= 0  # seconds with a state
    Z = np.full((len(sd), T.E.N_Z), np.nan)  # the 30 log-odds per second
    Z[ok] = mat["surrogate"][rows[ok]]  # filled where a state exists
    return rows, ok, Z  # the arrays


def run_row(stream_d: dict, stream_s: dict) -> dict:  # function: one run's decision- and stopping-mode figures
    out = {f"dec_{k}": stream_d[k] for k in DEC_KEYS}  # dict comprehension: decision mode
    out.update({f"stop_{k}": stream_s[k] for k in STOP_KEYS})  # stopping mode
    out["stop_miss_rate"] = 1 - stream_s["detected"] / stream_s["intervals"] if stream_s["intervals"] else np.nan  # misses over intervals
    return out  # the row


def both(sd: pd.DataFrame, dec_d: np.ndarray, dec_s: np.ndarray, held: str) -> tuple:  # function: the environment script's stream in both modes; (decision figures, stopping figures)
    sec, lab = sd.sec.to_numpy(np.int64), sd.label.to_numpy(object)  # arrays
    return E.stream(sec, lab, dec_d, held, "decision"), E.stream(sec, lab, dec_s, held, "stopping")  # both modes (stopping applies the refractory rule to the decisions)


def baseline_decisions(det: str, param: float, n: int, k: int, sd: pd.DataFrame, pred: dict, ok: np.ndarray, Z: np.ndarray) -> tuple:  # function: a baseline's decisions on the stream grid, (decision mode, stopping-mode input)
    sec = sd.sec.to_numpy(np.int64)  # seconds
    if det in ("lr", "rf"):  # the supervised baselines: their committed outer probabilities
        p = pred[det].reindex(sec).to_numpy()  # the probability at each windowed second
        d = (p >= param).astype(int)  # alarm where it reaches the threshold
        return d, d  # the same decisions; the stream applies the refractory rule in stopping mode
    if det == "cusum":  # CUSUM on z_t: the environment script's detector in each mode
        z = np.nan_to_num(Z[:, -1])  # z_t (0 where no state; those seconds are not valid)
        return E.cusum(sec, z, ok, param, "decision").astype(int), E.cusum(sec, z, ok, param, "stopping").astype(int)  # both modes
    q = np.where(ok, np.sort(np.nan_to_num(Z[:, -n:], nan=-np.inf), axis=1)[:, n - k], -np.inf)  # k of n: the k-th largest of the last n log-odds
    d = (q >= param).astype(int)  # alarm where it reaches h
    return d, d  # both modes from the same decisions


def evaluate_heldout() -> None:  # function: every detector on every held-out session, the curves, the aggregation, the verdicts and the figures
    ck = check_freeze()  # the freeze check
    EVAL.mkdir(parents=True, exist_ok=True)  # the folder
    E.write_csv(ck, EVAL / "freeze_check.csv")  # written
    if not (ck.in_freeze_commit & ck.equal_to_commit & ck.sha256_matches_manifest).all():  # if: a frozen file differs from the freeze commit, or the points are not committed
        raise SystemExit("STOP: the frozen files do not match the freeze commit, or the amended points are not committed; no held-out stream is read")  # refuse
    pts = rt(EVAL / "deployable_points.csv")  # the committed points
    sel = rt(GRID / "selection.csv")  # the frozen selection
    steps = T.read_steps()  # the environment script's step table
    preds = {m: rt(E.PROC / f"{'logistic_regression' if m == 'lr' else 'random_forest'}" / f"{'logistic_regression' if m == 'lr' else 'random_forest'}_predictions_primary.csv.gz") for m in ("lr", "rf")}  # dict comprehension: the committed outer probabilities
    runs, lat, curve = [], [], []  # run rows; per-interval delays; curve rows
    for rx in RXS:  # for loop over the receivers
        for outer in SESSIONS:  # for loop over the outer folds (held-out sessions)
            sd = T.session_seconds(steps, rx, outer)  # the held-out session's windowed seconds
            pred = {m: p[(p.receiver == rx) & (p.held_out == outer)].set_index("sec").p for m, p in preds.items()}  # dict comprehension: the probabilities by second
            if any(set(p_.index) != set(sd.sec) for p_ in pred.values()):  # if: the predictions do not cover the stream grid exactly (they must)
                raise SystemExit(f"STOP: {rx} {outer} predictions and stream grid differ")  # stop
            mat = heldout_material(rx, outer)  # the fold's material over the held-out rows
            rows_, ok, Z = grid_rows(mat, sd)  # the grid's state rows
            for r in pts[(pts.receiver == rx) & (pts.outer == outer) & pts.detector.isin(BASE)].itertuples():  # for loop over the baselines' points
                n, k = (int(r.n), int(r.k)) if r.detector == "kofn" else (0, 0)  # k-of-n's setting
                d_, s_ = baseline_decisions(r.detector, float(r.param), n, k, sd, pred, ok, Z)  # the decisions
                fd, fs = both(sd, d_, s_, outer)  # through the stream
                runs.append({"receiver": rx, "outer": outer, "detector": r.detector, "share": r.share, "param": float(r.param), "lambda": np.nan, "seed": -1, **run_row(fd, fs)})  # dict unpacking: the run
                lat += [{"receiver": rx, "outer": outer, "detector": r.detector, "share": r.share, "lambda": np.nan, "seed": -1, **x} for x in fs["latency"]]  # list comprehension: its intervals
            for arm in ("primary", "surrogate"):  # the agent: every lambda and seed (the points and the curve read from these)
                for lam in T.LAMBDAS:  # for loop over the lambdas
                    b = int(sel[(sel.receiver == rx) & (sel.outer == outer) & (sel.arm == arm) & (sel["lambda"] == float(lam))].selected_budget.iloc[0])  # its frozen budget
                    for seed in T.OUTER_SEEDS:  # for loop over the five seeds
                        spec = {"stage": "outer", "receiver": rx, "outer": outer, "inner": "", "arm": arm, "lambda": float(lam), "seed": seed, "budget": b}  # the frozen agent's specification
                        model, _ = T.load_q(spec, MODELS, mat)  # the frozen agent
                        env = E.SpoofingDetectionEnv(mat, arm, float(lam))  # an environment over the held-out rows (only its matrices and key are read)
                        d_, s_ = T.agent_decisions(model, env, sd)  # decision and stopping mode
                        fd, fs = both(sd, d_, s_, outer)  # through the stream
                        runs.append({"receiver": rx, "outer": outer, "detector": f"agent_{arm}", "share": np.nan, "param": float(lam), "lambda": float(lam), "seed": seed, **run_row(fd, fs)})  # dict unpacking: the run
                        lat += [{"receiver": rx, "outer": outer, "detector": f"agent_{arm}", "share": np.nan, "lambda": float(lam), "seed": seed, **x} for x in fs["latency"]]  # list comprehension: its intervals
            for det, grid_, nk in [("lr", LR_GRID, (0, 0)), ("rf", LR_GRID, (0, 0)), ("cusum", H_GRID, (0, 0))] + [("kofn", KN_GRID, (n, k)) for n, k in sorted({(int(r.n), int(r.k)) for r in pts[(pts.receiver == rx) & (pts.detector == "kofn")].itertuples()})]:  # list and set comprehensions: each baseline's curve parameters (k of n: the settings kept at the points)
                for prm in grid_:  # for loop over the parameter values
                    d_, s_ = baseline_decisions(det, float(prm), nk[0], nk[1], sd, pred, ok, Z)  # the decisions
                    fd, fs = both(sd, d_, s_, outer)  # through the stream
                    curve.append({"receiver": rx, "outer": outer, "detector": det, "n": nk[0], "k": nk[1], "param": float(prm), "clean": fd["clean"], "clean_in_alarm": fd["clean_in_alarm"], "stop_clean_in_alarm": fs["clean_in_alarm"], "intervals": fs["intervals"], "delays": ";".join(str(x["delay_s"]) for x in fs["latency"] if x["detected"])})  # generator inside join(): the curve row
            print(f"{rx} {outer}: {len(runs)} runs", flush=True)  # progress
    R, LAT, C = pd.DataFrame(runs), pd.DataFrame(lat), pd.DataFrame(curve)  # the tables
    E.write_csv(R, EVAL / "heldout_runs.csv")  # every run
    E.write_csv(LAT, EVAL / "heldout_latency.csv")  # every interval
    E.write_gz(C, EVAL / "curve_runs.csv.gz")  # the baselines' curve runs
    at = point_runs(R, pts)  # the runs at the deployable points
    E.write_csv(at, EVAL / "point_runs.csv")  # written
    ag = aggregate(at, LAT, pts)  # P8
    E.write_csv(ag, EVAL / "aggregate.csv")  # written
    vd = verdicts(at, LAT, pts)  # P9
    E.write_csv(vd, EVAL / "verdicts.csv")  # written
    cv = curves(R, LAT, C)  # P7
    E.write_csv(cv, EVAL / "curves.csv")  # written
    figures(pts, at, ag, cv, R, steps)  # the figures
    print(vd.to_string(index=False))  # printed


# ----------------------------------------------------------------------------------------------------------------------
# points, aggregation (P8) and verdicts (P9)
# ----------------------------------------------------------------------------------------------------------------------
def point_runs(R: pd.DataFrame, pts: pd.DataFrame) -> pd.DataFrame:  # function: the runs at each detector's deployable point; agents: the fold's lambda at that share, five seeds
    base = R[R.detector.isin(BASE)].copy()  # the baselines' runs carry their share already
    parts = [base]  # the pieces
    for r in pts[pts.detector.str.startswith("agent") & ~pts.no_point.astype(bool)].itertuples():  # for loop over the agent's points that exist
        a = R[(R.receiver == r.receiver) & (R.outer == r.outer) & (R.detector == r.detector) & (R["lambda"] == r.param)].assign(share=r.share)  # its five seed runs at the point
        parts.append(a)  # kept
    at = pd.concat(parts, ignore_index=True)  # every point run
    return at.assign(stop_clean_share=at.stop_clean_in_alarm / at.stop_clean.where(at.stop_clean > 0))  # the stopping-mode share per run (NaN without clean seconds), capped near 1/31 by the refractory period


def iqm(x: np.ndarray, axis: int = -1) -> np.ndarray:  # function: the interquartile mean (Agarwal et al., 2021, p. 7): the mean after cutting floor(n/4) values from each end (scipy's trim_mean convention, as rliable uses)
    x = np.sort(x, axis=axis)  # sorted
    n = x.shape[axis]  # the count
    c = int(np.floor(0.25 * n))  # cut from each end
    return np.take(x, np.arange(c, n - c), axis=axis).mean(axis=axis)  # the middle values' mean


def strat_boot(values: np.ndarray, strata: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> tuple:  # function: the 95 % stratified bootstrap percentile interval of the IQM (Agarwal et al., 2021, p. 6): runs resampled with replacement within each fold
    v, s = np.asarray(values, float), np.asarray(strata)  # arrays
    keep = np.isfinite(v)  # defined runs only
    v, s = v[keep], s[keep]  # filtered
    if not len(v):  # no defined run
        return np.nan, np.nan, np.nan, 0  # undefined
    rng = np.random.default_rng(seed)  # the seeded generator
    parts = []  # each stratum's resamples
    for g in sorted(set(s)):  # for loop over the strata in sorted order (the draw order T6 replays)
        x = v[s == g]  # its runs
        parts.append(x[rng.integers(0, len(x), size=(n_boot, len(x)))])  # n_boot resamples of the same size
    B = iqm(np.concatenate(parts, axis=1), axis=1)  # each resample's IQM
    return float(iqm(v)), float(np.percentile(B, 2.5)), float(np.percentile(B, 97.5)), len(v)  # point, interval, runs


METRICS = [("dec_recall", "recall"), ("dec_precision", "precision"), ("dec_f1", "F1"), ("dec_clean_share", "clean-window alarm share"), ("dec_early_warning", "early warning"),  # list of (column, name): P8's metrics
           ("stop_delay_median_s", "median delay (s)"), ("stop_miss_rate", "miss rate"), ("stop_clean_in_alarm", "false alarms (events on clean seconds)"), ("stop_fa_rate_per_h", "false alarms per clean hour (2.6.2 only)"),  # list continued
           ("stop_clean_share", "stopping-mode clean-window alarm share (at most about 1/31)")]  # list continued


def aggregate(at: pd.DataFrame, LAT: pd.DataFrame, pts: pd.DataFrame) -> pd.DataFrame:  # function: P8 per receiver, share and detector: IQM over seed x fold runs with 95 % stratified bootstrap intervals, and pooled figures
    rows = []  # one row per receiver, share, detector and metric
    for (rx, s, det), d in at.groupby(["receiver", "share", "detector"], sort=False):  # for loop over the cells
        lat = latency_at(LAT, pts, rx, s, det)  # its intervals
        pooled = {"pooled_median_delay_s": float(np.median(lat[lat.detected].delay_s)) if lat.detected.any() else np.nan, "pooled_miss_share": float(1 - lat.detected.mean()) if len(lat) else np.nan,  # dict literal: pooled over every interval
                  "pooled_clean_share": float(d.dec_clean_in_alarm.sum() / d.dec_clean.sum()) if d.dec_clean.sum() else np.nan, "folds": d.outer.nunique(), "runs": len(d)}  # dict literal continued: pooled clean share
        for col, name in METRICS:  # for loop over the metrics
            p, lo, hi, n = strat_boot(d[col].to_numpy(float), d.outer.to_numpy())  # IQM and interval
            rows.append({"receiver": rx, "share": s, "detector": det, "metric": name, "iqm": p, "ci_low": lo, "ci_high": hi, "runs_defined": n, **pooled})  # dict unpacking: the row
    return pd.DataFrame(rows)  # the table


def latency_at(LAT: pd.DataFrame, pts: pd.DataFrame, rx: str, s: float, det: str) -> pd.DataFrame:  # function: the intervals of a detector's point runs at a share
    if det in BASE:  # a baseline: its rows carry the share
        return LAT[(LAT.receiver == rx) & (LAT.detector == det) & (LAT.share == s)]  # filtered
    p = pts[(pts.receiver == rx) & (pts.detector == det) & (pts.share == s) & ~pts.no_point.astype(bool)]  # the agent's folds with a point and their lambdas
    return pd.concat([LAT[(LAT.receiver == rx) & (LAT.outer == r.outer) & (LAT.detector == det) & (LAT["lambda"] == r.param)] for r in p.itertuples()] or [LAT.iloc[:0]], ignore_index=True)  # list comprehension: each fold's intervals at its lambda


def verdicts(at: pd.DataFrame, LAT: pd.DataFrame, pts: pd.DataFrame) -> pd.DataFrame:  # function: P9 per receiver, share and arm against the best baseline at that share
    rows = []  # one row per receiver, share and arm
    for rx in RXS:  # for loop over the receivers
        for s in SHARES:  # for loop over the shares
            bl = {b: latency_at(LAT, pts, rx, s, b) for b in BASE}  # dict comprehension: each baseline's intervals at the share
            med = {b: (float(np.median(x[x.detected].delay_s)) if x.detected.any() else np.inf) for b, x in bl.items()}  # dict comprehension: pooled median delay (inf when nothing is detected)
            best = min(BASE, key=lambda b: (med[b], BASE.index(b)))  # lambda as sort key: the lowest median delay, ties in the listed order
            bruns = at[(at.receiver == rx) & (at.share == s) & (at.detector == best)].set_index("outer")  # its runs, one per fold
            b_miss = float(1 - bl[best].detected.mean()) if len(bl[best]) else np.nan  # its pooled miss rate
            for arm in ("primary", "surrogate"):  # for loop over the arms
                a = at[(at.receiver == rx) & (at.share == s) & (at.detector == f"agent_{arm}")]  # the agent's runs at its points
                np_folds = int(pts[(pts.receiver == rx) & (pts.share == s) & (pts.detector == f"agent_{arm}")].no_point.astype(bool).sum())  # folds where the agent has no point
                if a.empty:  # no point in any fold
                    rows.append({"receiver": rx, "share": s, "arm": arm, "best_baseline": best, "baseline_median_delay_s": med[best], "verdict": "no point (no lambda meets the share)", "folds_without_point": np_folds})  # the row
                    continue  # next arm
                diff = a.stop_delay_median_s.to_numpy(float) - bruns.loc[a.outer].stop_delay_median_s.to_numpy(float)  # paired differences, agent minus baseline, per seed x fold run
                p, lo, hi, n = strat_boot(diff, a.outer.to_numpy())  # IQM of the differences and its interval
                al = latency_at(LAT, pts, rx, s, f"agent_{arm}")  # the agent's intervals
                a_miss = float(1 - al.detected.mean()) if len(al) else np.nan  # its pooled miss rate
                b_miss_paired = float(1 - pd.concat([bl[best][bl[best].outer == o] for o in sorted(set(a.outer))]).detected.mean())  # generator inside concat(): the baseline's miss rate on the agent's folds
                higher = a_miss > b_miss_paired  # a higher miss rate?
                if np.isfinite(hi) and hi < 0 and not higher:  # P9: better
                    v = "better"  # the verdict
                elif np.isfinite(lo) and (lo > 0 or (higher and lo <= 0 <= hi)):  # P9: worse
                    v = "worse"  # the verdict
                else:  # otherwise
                    v = "no difference detectable"  # the verdict
                # dict literal over two lines: the row
                rows.append({"receiver": rx, "share": s, "arm": arm, "best_baseline": best, "baseline_median_delay_s": med[best], "baseline_miss_rate_all_folds": b_miss, "baseline_miss_rate_agent_folds": b_miss_paired, "agent_miss_rate": a_miss, "delay_diff_iqm_s": p,
                             "delay_diff_ci_low": lo, "delay_diff_ci_high": hi, "paired_runs": n, "runs": len(a), "folds_with_point": a.outer.nunique(), "folds_without_point": np_folds, "verdict": v,  # dict literal continued: the paired difference, the counts and the verdict
                             "note": "no seed x fold run with both median delays defined (the baseline detected nothing on the agent's folds)" if n == 0 else ""})  # dict literal continued
    return pd.DataFrame(rows)  # the table


def curves(R: pd.DataFrame, LAT: pd.DataFrame, C: pd.DataFrame) -> pd.DataFrame:  # function: P7's operating curves on the held-out stream, pooled over the folds (and the seeds for the agent)
    rows = []  # one row per receiver, detector and parameter
    for (rx, det, n, k, prm), d in C.groupby(["receiver", "detector", "n", "k", "param"], sort=False):  # for loop over the baselines' curve points
        dl = [float(x) for s in d.delays.fillna("") for x in str(s).split(";") if x not in ("", "nan")]  # nested comprehension: every detected interval's delay
        rows.append({"receiver": rx, "detector": det, "n": n, "k": k, "param": prm, "clean_share": d.clean_in_alarm.sum() / d.clean.sum() if d.clean.sum() else np.nan, "stop_clean_share": d.stop_clean_in_alarm.sum() / d.clean.sum() if d.clean.sum() else np.nan,  # dict literal: the shares
                     "median_delay_s": float(np.median(dl)) if dl else np.nan, "miss_share": 1 - len(dl) / d.intervals.sum() if d.intervals.sum() else np.nan, "intervals": int(d.intervals.sum())})  # dict literal continued: delay and misses
    for (rx, det, lam), d in R[R.detector.str.startswith("agent")].groupby(["receiver", "detector", "lambda"], sort=False):  # for loop over the agent's lambdas
        l_ = LAT[(LAT.receiver == rx) & (LAT.detector == det) & (LAT["lambda"] == lam)]  # its intervals over folds and seeds
        rows.append({"receiver": rx, "detector": det, "n": 0, "k": 0, "param": lam, "clean_share": d.dec_clean_in_alarm.sum() / d.dec_clean.sum(), "stop_clean_share": d.stop_clean_in_alarm.sum() / d.dec_clean.sum(),  # dict literal: the shares
                     "median_delay_s": float(np.median(l_[l_.detected].delay_s)) if l_.detected.any() else np.nan, "miss_share": float(1 - l_.detected.mean()), "intervals": len(l_)})  # dict literal continued
    return pd.DataFrame(rows)  # the table


# ----------------------------------------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------------------------------------
def figures(pts: pd.DataFrame, at: pd.DataFrame, ag: pd.DataFrame, cv: pd.DataFrame, R: pd.DataFrame, steps: pd.DataFrame) -> None:  # function: the figures
    import matplotlib.pyplot as plt  # pyplot (09 chose the file-only backend)
    NAME = E.NAME  # receiver names
    cur = rt(GRID / "inner_curve.csv.gz")  # the inner curve (frozen)
    sel = rt(GRID / "selection.csv")  # the selection (frozen)
    # the figure: the inner learning curves of the mosaic-X5
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)  # one panel per arm
    lam_col = dict(zip(T.LAMBDAS, ["#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC79A7"]))  # dict: lambda -> colour (fixed order)
    for ax, arm in zip(axes, ("primary", "surrogate")):  # for loop over the arms
        d = cur[(cur.receiver == "SBF_02") & (cur.arm == arm)]  # its inner runs
        for lam in T.LAMBDAS:  # for loop over the lambdas
            c = d[d["lambda"] == float(lam)].groupby("budget").criterion.mean()  # the criterion averaged over outer folds, inner folds and seeds
            ax.plot(c.index / 1000, c.to_numpy(), marker="o", ms=4, lw=1.8, color=lam_col[lam], label=f"λ = {lam}")  # one line per lambda
        ax.set_title(f"{NAME['SBF_02']}, {arm} arm", fontsize=9, loc="left")  # title
        ax.set_xlabel("training steps (thousands)", fontsize=8)  # axis
        ax.grid(alpha=0.3, lw=0.5)  # recessive grid
    axes[0].set_ylabel("inner criterion: mean undiscounted cost (s)", fontsize=8)  # axis
    axes[1].legend(fontsize=7, frameon=False)  # legend
    fig.tight_layout()  # layout
    M.save_fig(fig, GRAPH / "inner_curves.png")  # byte-stable save
    # the figure: the held-out operating curves per receiver, with the shares and the deployable points
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)  # delay (top) and miss share (bottom) per receiver
    for j, rx in enumerate(RXS):  # for loop over the receivers
        for det in DET:  # for loop over the detectors
            col, mk, lab = STYLE[det]  # its style
            d = cv[(cv.receiver == rx) & (cv.detector == det)]  # its curve points
            if det == "kofn":  # k of n: the setting kept most often at the points
                nk = pts[(pts.receiver == rx) & (pts.detector == "kofn")][["n", "k"]].value_counts().idxmax()  # the most frequent (n, k)
                d = d[(d.n == nk[0]) & (d.k == nk[1])]  # its curve
                lab = f"k of n ({int(nk[1])} of {int(nk[0])})"  # label
            d = d.sort_values("clean_share")  # along the share axis
            for i, y in enumerate(("median_delay_s", "miss_share")):  # for loop over the two rows
                axes[i, j].plot(d.clean_share, d[y], color=col, lw=2.2 if det.startswith("agent") else 1.2, marker=mk if det.startswith("agent") else None, ms=6, label=lab)  # the curve
        for i, y in enumerate(("stop_delay_median_s", "stop_miss_rate")):  # for loop over the two rows: the deployable points
            for det in DET:  # for loop over the detectors
                a = at[(at.receiver == rx) & (at.detector == det)]  # its point runs
                g = a.groupby("share").agg(x=("dec_clean_in_alarm", "sum"), c=("dec_clean", "sum"), v=(y, "median"))  # per share: pooled share and the median of the runs' figure
                axes[i, j].scatter(g.x / g.c, g.v, marker=STYLE[det][1], s=36, facecolor="white", edgecolor=STYLE[det][0], linewidth=1.5, zorder=3)  # the points, hollow
        for i in range(2):  # for loop over the rows
            for s in SHARES:  # the registered shares
                axes[i, j].axvline(s, color="#888888", lw=0.8, ls=":")  # a dotted line
            axes[i, j].set_xscale("symlog", linthresh=0.01)  # the low shares spread out
            axes[i, j].set_xlim(0, 1)  # every share
            axes[i, j].grid(alpha=0.3, lw=0.5)  # recessive grid
        axes[0, j].set_title(NAME[rx], fontsize=9, loc="left")  # title
        axes[0, j].set_yscale("symlog", linthresh=10)  # signed log: single long delays (hundreds of s) no longer flatten the rest
        axes[1, j].set_xlabel("pooled clean-window alarm share (decision mode)", fontsize=8)  # axis
    axes[0, 0].set_ylabel("median delay (s), stopping mode", fontsize=8)  # axis
    axes[1, 0].set_ylabel("share of intervals missed", fontsize=8)  # axis
    h_, l_ = axes[0, 0].get_legend_handles_labels()  # the legend entries
    fig.legend(h_, l_, loc="lower center", ncol=6, fontsize=7, frameon=False)  # one legend below the panels
    fig.tight_layout(rect=(0, 0.05, 1, 1))  # layout with room for the legend
    M.save_fig(fig, GRAPH / "operating_curves.png")  # byte-stable save
    # the figure: median delay and miss rate at the deployable points with intervals
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))  # delay (top) and miss rate (bottom) per receiver
    for j, rx in enumerate(RXS):  # for loop over the receivers
        for i, mt in enumerate(("median delay (s)", "miss rate")):  # for loop over the two rows
            for q, det in enumerate(DET):  # for loop over the detectors
                d = ag[(ag.receiver == rx) & (ag.detector == det) & (ag.metric == mt)].set_index("share").reindex(list(SHARES))  # its three shares
                x = np.arange(len(SHARES)) + (q - 2.5) * 0.12  # dodged positions
                axes[i, j].errorbar(x, d.iqm, yerr=[d.iqm - d.ci_low, d.ci_high - d.iqm], fmt=STYLE[det][1], color=STYLE[det][0], ms=6, lw=1.2, capsize=2, label=STYLE[det][2])  # IQM with its interval
            axes[i, j].set_xticks(range(len(SHARES)), [f"{s:g}" for s in SHARES])  # list comprehension: the shares
            axes[i, j].grid(alpha=0.3, lw=0.5, axis="y")  # recessive grid
        axes[0, j].set_title(NAME[rx], fontsize=9, loc="left")  # title
        axes[1, j].set_xlabel("registered clean-window share", fontsize=8)  # axis
    axes[0, 0].set_ylabel("median delay (s): IQM, 95 % interval", fontsize=8)  # axis
    axes[1, 0].set_ylabel("miss rate: IQM, 95 % interval", fontsize=8)  # axis
    h_, l_ = axes[0, 0].get_legend_handles_labels()  # the legend entries
    fig.legend(h_, l_, loc="lower center", ncol=6, fontsize=7, frameon=False)  # one legend below the panels
    fig.tight_layout(rect=(0, 0.05, 1, 1))  # layout
    M.save_fig(fig, GRAPH / "deployable_points.png")  # byte-stable save
    # the figure: the implied threshold against lambda per arm
    med = rt(GRID / "implied_medians.csv")  # medians over seeds (frozen with the agents)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))  # one panel per arm
    rx_col = dict(zip(RXS, ["#0072B2", "#D55E00", "#009E73"]))  # dict: receiver -> colour
    for ax, arm in zip(axes, ("primary", "surrogate")):  # for loop over the arms
        for rx in RXS:  # for loop over the receivers
            d = med[(med.receiver == rx) & (med.arm == arm)]  # its folds' medians
            ax.scatter(d["lambda"], d.median_threshold, s=14, color=rx_col[rx], alpha=0.5)  # every fold
            m = d.groupby("lambda").median_threshold.median()  # the median over folds
            ax.plot(m.index, m.to_numpy(), marker="o", lw=1.8, color=rx_col[rx], label=NAME[rx])  # the line
        ax.set_xscale("log")  # lambda is log-spaced
        ax.set_xlabel("λ (s of delay per false alarm)", fontsize=8)  # axis
        ax.set_ylabel("implied threshold on " + ("S_t" if arm == "surrogate" else "the log-odds z_t"), fontsize=8)  # axis
        if arm == "surrogate":  # the surrogate's thresholds reach thousands of both signs
            ax.set_yscale("symlog", linthresh=10)  # signed log scale
        ax.set_title(f"{arm} arm: median over five seeds (dots: folds; line: median over folds)", fontsize=9, loc="left")  # title
        ax.grid(alpha=0.3, lw=0.5)  # recessive grid
        ax.legend(fontsize=7, frameon=False)  # legend
    fig.tight_layout()  # layout
    M.save_fig(fig, GRAPH / "implied_thresholds.png")  # byte-stable save
    # the figure: one held-out stream (mosaic-X5, 2.6.2) with the labels, the agent's alarms in both modes, the logistic regression's and S_t
    rx, outer = "SBF_02", "2.6.2"  # the session
    s_ = next(s for s in (0.05, 0.10, 0.02) if not pts[(pts.receiver == rx) & (pts.outer == outer) & (pts.detector == "agent_primary") & (pts.share == s)].no_point.astype(bool).iloc[0])  # next(): the first share with a primary-arm point
    lam = float(pts[(pts.receiver == rx) & (pts.outer == outer) & (pts.detector == "agent_primary") & (pts.share == s_)].param.iloc[0])  # its lambda
    sd = T.session_seconds(steps, rx, outer)  # the stream
    mat = heldout_material(rx, outer)  # the material
    b = int(sel[(sel.receiver == rx) & (sel.outer == outer) & (sel.arm == "primary") & (sel["lambda"] == lam)].selected_budget.iloc[0])  # the budget
    model, _ = T.load_q({"stage": "outer", "receiver": rx, "outer": outer, "inner": "", "arm": "primary", "lambda": lam, "seed": SEED, "budget": b}, MODELS, mat)  # the seed-20260914 agent
    dd, ds = T.agent_decisions(model, E.SpoofingDetectionEnv(mat, "primary", lam), sd)  # its decisions
    thr = float(pts[(pts.receiver == rx) & (pts.outer == outer) & (pts.detector == "lr") & (pts.share == s_)].param.iloc[0])  # the logistic regression's threshold at the share
    p = rt(E.PROC / "logistic_regression" / "logistic_regression_predictions_primary.csv.gz")  # its probabilities
    p = p[(p.receiver == rx) & (p.held_out == outer)].set_index("sec").p.reindex(sd.sec.to_numpy()).to_numpy()  # on the grid
    _, ok, Z = grid_rows(mat, sd)  # z_t
    S, v = np.zeros(len(sd)), 0.0  # S_t without resets except at gaps
    for i in range(len(sd)):  # for loop over the seconds
        v = max(0.0, (v if i and ok[i - 1] and sd.sec.iloc[i] - sd.sec.iloc[i - 1] == 1 else 0.0) + (Z[i, -1] if ok[i] else 0.0)) if ok[i] else 0.0  # Page's recursion, 0 where no state
        S[i] = v  # kept
    t = (sd.sec.to_numpy() - sd.sec.min()) / 60  # minutes
    fig, axs = plt.subplots(5, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [1, 0.6, 0.6, 0.6, 2]})  # labels, three alarm strips, S_t
    for lab_, col in (("clean", "#009E73"), ("attack_jam_spoof", "#D55E00"), ("attack_spoof", "#CC79A7"), ("uncertain", "#BBBBBB")):  # for loop over the labels
        m_ = sd.label.to_numpy() == lab_  # its seconds
        axs[0].bar(t[m_], 1, width=1 / 60, color=col, linewidth=0, label=lab_)  # thin bars
    axs[0].legend(fontsize=6, ncol=4, loc="upper right", frameon=False)  # legend
    axs[0].set_title(f"{NAME[rx]}, held-out session {outer}: scoring labels; alarms at the share {s_:g} (agent λ = {lam:g}, seed {SEED}; logistic regression threshold {thr:.4f})", fontsize=9, loc="left")  # title
    for ax, a_, name in ((axs[1], dd, "agent, decision mode"), (axs[2], ds, "agent, stopping mode"), (axs[3], (p >= thr).astype(int), "logistic regression, decision mode")):  # for loop over the three alarm strips
        ax.bar(t[a_ == 1], 1, width=1 / 60, color="#000000", linewidth=0)  # one bar per alarm second
        ax.set_yticks([])  # no scale
        ax.set_ylabel(name, fontsize=7, rotation=0, ha="right", va="center")  # strip name
    axs[4].plot(t, S, color="#0072B2", lw=1.2, label="S_t (no reset at alarms)")  # the statistic
    axs[4].set_yscale("symlog", linthresh=1)  # S reaches hundreds
    axs[4].set_ylabel("S_t", fontsize=8)  # axis
    axs[4].set_xlabel(f"minutes after {pd.Timestamp(int(sd.sec.min()), unit='s').strftime('%H:%M:%S')} UTC", fontsize=8)  # axis
    axs[4].legend(fontsize=7, frameon=False)  # legend
    fig.tight_layout()  # layout
    M.save_fig(fig, GRAPH / "heldout_stream.png")  # byte-stable save


def main() -> None:  # function: the command line
    ap = argparse.ArgumentParser()  # the parser
    ap.add_argument("--points", action="store_true")  # --points
    ap.add_argument("--heldout", action="store_true")  # --heldout
    a = ap.parse_args()  # parsed
    if a.points:  # --points
        evaluate_points()  # run
    elif a.heldout:  # --heldout
        evaluate_heldout()  # run
    else:  # nothing chosen
        raise SystemExit("usage: 6_reinforcement_learning/03_evaluate_agents.py --points | --heldout")  # stop


if __name__ == "__main__":  # guard: runs main() only when the file is executed, not when imported
    main()  # runs main
