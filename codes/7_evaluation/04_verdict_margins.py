#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
How firmly the agent's verdicts of Table 11 hold (reported checks; every published verdict stays as it was set before the results).

Decision rule  the label-sensitivity decision rule with F1 counted as 0 for a class with instances and no true positive
               (Data/processed/data_checks/label_sensitivity/decision_rule.csv): the branch per receiver and which criterion carries it.
Bootstrap      for every receiver, share and state of Table 11, the share of 200 bootstrap seeds (20260914 + k, k = 0..199) under which
               the evaluation script's verdict rule returns the published verdict, beside the published interval (seed 20260914).
Comparator     the baseline the agent is paired with, chosen from training data only. The rule was fixed before computing: per
               receiver, share and outer fold, the baseline with the lowest pooled median stopping-mode delay on that fold's inner
               sessions (its four training sessions); ties in the order lr, rf, cusum, kofn; a baseline with no inner detection counts
               as +inf. CUSUM's and k-of-n's inner delays are the training script's saved values (rl_agents/training/deployable_points.csv).
               The regression's and the forest's are computed here from the saved inner out-of-fold predictions
               (rl_agents/training/nested_predictions.csv.gz) at their saved thresholds, through the environment's stream in stopping mode
               on each inner session, exactly as the training script computes k-of-n's (no refit). A gate first recomputes the held-out
               lr and rf delays of heldout_latency.csv the same way from the saved outer predictions. The verdicts are then formed with
               the evaluation script's rule, each run paired with its fold's comparator.
Usage   python3 codes/7_evaluation/04_verdict_margins.py
Writes  Data/processed/robustness_checks/decision_rule_branch.csv, bootstrap_margin.csv, inner_delays_lr_rf.csv, comparator_sensitivity.csv,
        margins_checks.csv
"""
# import: importlib loads the evaluation script as a module (its name starts with a digit)
import importlib.util
# import: sys registers the module
import sys
# import: time for the run time
import time
# import: Path for explicit paths
from pathlib import Path
# import: numpy for arrays and medians
import numpy as np
# import: pandas for the tables
import pandas as pd

# project root (this file is codes/7_evaluation/04_verdict_margins.py)
PROJECT = Path(__file__).resolve().parents[2]
# processed-data folder and the position check's folder
PROC, ROBUST_DIR = PROJECT / "Data" / "processed", PROJECT / "Data" / "processed" / "robustness_checks"
# where the evaluation script is
_spec = importlib.util.spec_from_file_location("evaluate_agents_for_margins", PROJECT / "codes" / "6_reinforcement_learning" / "03_evaluate_agents.py")
# an empty module object
B12 = importlib.util.module_from_spec(_spec)
# registered so its own references resolve
sys.modules["evaluate_agents_for_margins"] = B12
# its top level (it loads 12, 11 and the adopted-pipeline script); no main() runs
_spec.loader.exec_module(B12)
# 12 and 11 as the evaluation script holds them
T, E = B12.T, B12.E
# list of check rows
CHECKS = []
# the bootstrap seeds: the published one and 199 more
SEEDS = [B12.SEED + k for k in range(200)]


# function: records and prints one check
def check(name: str, ok: bool, detail: str = "") -> None:
    # appended
    CHECKS.append({"check": name, "pass": bool(ok), "detail": detail})
    # printed
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


# function: the evaluation script's verdict rule (verdicts(), lines 300-306) on given numbers
def rule(lo: float, hi: float, higher: bool) -> str:
    # better: the whole interval below zero and no more misses
    if np.isfinite(hi) and hi < 0 and not higher:
        return "better"  # the verdict
    # worse: the interval above zero, or more misses while the interval covers zero
    if np.isfinite(lo) and (lo > 0 or (higher and lo <= 0 <= hi)):
        return "worse"  # the verdict
    # otherwise
    return "no difference detectable"


# function: the decision rule's branch per receiver from the corrected table
def item4() -> pd.DataFrame:
    # the corrected decision table
    D = pd.read_csv(PROC / "data_checks" / "label_sensitivity" / "decision_rule.csv")
    # list comprehension over the receivers: which criteria are met
    rows = [{"receiver": r.receiver, "rank_swapped": r.rank_swapped, "f1_moved_over_0.10": r["f1_moved_over_0.10"], "rq1_families_changed": r.rq1_families_changed,
             "limitation_branch": r.limitation_branch, "carried_by": ", ".join(k for k in ("rank_swapped", "f1_moved_over_0.10", "rq1_families_changed") if r[k]) or "none"}  # list comprehension over the receivers: which criteria are met and which carries the branch
            for _, r in D.iterrows()]  # continued: the branch and the carrying criteria as text
    # DataFrame, written
    out = pd.DataFrame(rows)
    out.to_csv(ROBUST_DIR / "decision_rule_branch.csv", index=False, lineterminator="\n")  # written with fixed line endings
    # printed
    print(out.to_string(index=False))
    # the table
    return out


# function: the paired differences of one Table 11 cell (agent minus a comparator per run) and the inputs of the verdict
def paired(at: pd.DataFrame, rx: str, s: float, arm: str, comp: dict) -> tuple:
    # the agent's runs at its points
    a = at[(at.receiver == rx) & (at.share == s) & (at.detector == f"agent_{arm}")]
    # list comprehension: the comparator's run on each run's fold (comp: fold -> baseline)
    b = np.array([float(at[(at.receiver == rx) & (at.share == s) & (at.detector == comp[o]) & (at.outer == o)].stop_delay_median_s.iloc[0]) for o in a.outer])
    # agent minus comparator per run
    return a, a.stop_delay_median_s.to_numpy(float) - b


# function: the bootstrap margin, the share of 200 seeds that return the published verdict
def item5(at: pd.DataFrame, LAT: pd.DataFrame, pts: pd.DataFrame) -> pd.DataFrame:
    # the published verdicts
    pub = B12.rt(B12.EVAL / "verdicts.csv")
    # list of result rows
    rows = []
    # for loop over the published cells
    for v in pub.itertuples():
        # if: the agent had no point at this share
        if str(v.verdict).startswith("no point"):
            # recorded as not applicable
            rows.append({"receiver": v.receiver, "share": v.share, "arm": v.arm, "published_verdict": v.verdict, "published_ci_low": np.nan, "published_ci_high": np.nan, "seeds": 0, "seeds_same_verdict": 0, "share_same_verdict": np.nan, "lo_min": np.nan, "lo_max": np.nan, "hi_min": np.nan, "hi_max": np.nan})
            continue  # continue: next cell
        # the published comparator for every fold (one baseline per cell)
        a, d = paired(at, v.receiver, v.share, v.arm, {o: v.best_baseline for o in set(at.outer)})
        # the miss condition does not depend on the bootstrap: as published
        higher = v.agent_miss_rate > v.baseline_miss_rate_agent_folds
        # the evaluation script's IQM and stratified bootstrap under each seed
        res = [B12.strat_boot(d, a.outer.to_numpy(), seed=s) for s in SEEDS]
        # list comprehension: the verdict under each seed
        vv = [rule(lo, hi, higher) for _, lo, hi, _ in res]
        # the published seed must give the published interval exactly (it is SEEDS[0])
        check(f"seed {SEEDS[0]} reproduces the published interval of {v.receiver} {v.share} {v.arm}", np.allclose([res[0][1], res[0][2]], [v.delay_diff_ci_low, v.delay_diff_ci_high], equal_nan=True))
        # arrays of the interval bounds over the seeds
        los, his = np.array([r[1] for r in res]), np.array([r[2] for r in res])
        # the row
        rows.append({"receiver": v.receiver, "share": v.share, "arm": v.arm, "published_verdict": v.verdict, "published_ci_low": v.delay_diff_ci_low, "published_ci_high": v.delay_diff_ci_high,
                     "seeds": len(SEEDS), "seeds_same_verdict": sum(x == v.verdict for x in vv), "share_same_verdict": sum(x == v.verdict for x in vv) / len(SEEDS),  # the bounds over all seeds and the verdict counts (dict literal continued)
                     "lo_min": np.nanmin(los) if np.isfinite(los).any() else np.nan, "lo_max": np.nanmax(los) if np.isfinite(los).any() else np.nan,  # continued: the share of seeds and the range of the lower bound
                     "hi_min": np.nanmin(his) if np.isfinite(his).any() else np.nan, "hi_max": np.nanmax(his) if np.isfinite(his).any() else np.nan})  # continued: the range of the upper bound
    # DataFrame, written
    out = pd.DataFrame(rows)
    out.to_csv(ROBUST_DIR / "bootstrap_margin.csv", index=False, lineterminator="\n")  # written with fixed line endings
    # printed
    print(out.round(3).to_string(index=False))
    # the table
    return out


# function: stopping-mode delays of one per-second detector on one session's stream grid, from probabilities by second and a threshold
def delays(steps: pd.DataFrame, rx: str, session: str, p: pd.Series, thr: float) -> tuple:
    # the session's windowed seconds (the training script's stream grid)
    sd = T.session_seconds(steps, rx, session)
    # the probability at each windowed second (NaN where none: silent)
    pv = p.reindex(sd.sec).to_numpy()
    # alarm where the probability reaches the threshold (the evaluation script's baseline_decisions for lr and rf)
    d = (pv >= thr).astype(int)
    # the environment script's stream in stopping mode (the evaluation script's both's second half)
    st = E.stream(sd.sec.to_numpy(np.int64), sd.label.to_numpy(object), d, session, "stopping")
    # the intervals (detected flag and delay), and the coverage of the grid by the predictions
    return st["latency"], int(np.isfinite(pv).sum()), len(sd)


# function: the comparator, the inner delays of lr and rf, their gate, and the comparator verdicts
def item6(at: pd.DataFrame, LAT: pd.DataFrame, pts: pd.DataFrame) -> pd.DataFrame:
    # the environment script's step table (labels and windowed seconds of every session)
    steps = T.read_steps()
    # the agent experiment's grid points (with CUSUM's and k-of-n's saved inner delays)
    gp = B12.rt(T.AGENT_DIR / "training" / "deployable_points.csv")
    # the committed outer predictions of the two baselines, by model
    outer = {m: B12.rt(PROC / f"{'logistic_regression' if m == 'lr' else 'random_forest'}" / f"{'logistic_regression' if m == 'lr' else 'random_forest'}_predictions_primary.csv.gz") for m in ("lr", "rf")}
    # the agent experiment's inner out-of-fold predictions
    nested = B12.rt(T.AGENT_DIR / "training" / "nested_predictions.csv.gz")
    # gate: the held-out lr and rf delays of heldout_latency.csv recomputed with delays() from the outer predictions
    bad, n = 0, 0
    # for loop over the baselines' points of the two per-second models
    for r in pts[pts.detector.isin(["lr", "rf"])].itertuples():
        # the fold's outer predictions by second
        p = outer[r.detector][(outer[r.detector].receiver == r.receiver) & (outer[r.detector].held_out == r.outer)].set_index("sec").p
        # recomputed intervals
        lat, _, _ = delays(steps, r.receiver, r.outer, p, float(r.param))
        # the committed intervals of the same run
        ref = LAT[(LAT.receiver == r.receiver) & (LAT.outer == r.outer) & (LAT.detector == r.detector) & np.isclose(LAT.share, r.share)].sort_values("onset")
        # for loop over the pairs of intervals
        for x, y in zip(lat, ref.itertuples()):
            # one comparison
            n += 1
            # a mismatch in detection or delay
            bad += int(x["detected"] != bool(y.detected) or (x["detected"] and float(x["delay_s"]) != float(y.delay_s)))
        # interval counts must agree too
        bad += int(len(lat) != len(ref))
    check("gate: lr and rf held-out delays recomputed from the outer predictions = heldout_latency.csv", bad == 0, f"{n} intervals, {bad} mismatches")  # the gate's result
    # the inner delays of lr and rf: per receiver, outer fold, share and model, pooled over the four inner sessions
    rows = []
    # for loop over the lr and rf grid points (each carries its threshold from the inner clean windows)
    for r in gp[gp.detector.isin(["lr", "rf"])].itertuples():
        # the fold's training sessions (each the inner held-out session of one inner fold)
        train = [s for s in T.SESSIONS if s != r.outer]
        # pooled detected delays, detected and all intervals, coverage counts
        dl, det, iv, cov, grid = [], 0, 0, 0, 0
        # for loop over the inner sessions
        for sess in train:
            # the inner out-of-fold probabilities of this model, receiver, outer fold and inner session, by second
            q = nested[(nested.model == r.detector) & (nested.receiver == r.receiver) & (nested.outer_held_out == r.outer) & (nested.held_out == sess)].set_index("sec").p
            # the session's intervals at the point's threshold
            lat, c, g = delays(steps, r.receiver, sess, q, float(r.param))
            # pooled
            dl += [x["delay_s"] for x in lat if x["detected"]]
            det, iv, cov, grid = det + sum(x["detected"] for x in lat), iv + len(lat), cov + c, grid + g  # continued: the detection counts and the coverage of the grid
        # one row: the pooled inner median delay (+inf without a detection, as 12 does for k-of-n)
        rows.append({"receiver": r.receiver, "outer": r.outer, "detector": r.detector, "share": r.share, "param": r.param, "inner_median_delay": float(np.median(dl)) if dl else np.inf,
                     "inner_detected": det, "inner_intervals": iv, "grid_seconds": grid, "grid_seconds_with_prediction": cov})  # written
    # DataFrame of the computed inner delays, written
    ID = pd.DataFrame(rows)
    ID.to_csv(ROBUST_DIR / "inner_delays_lr_rf.csv", index=False, lineterminator="\n")  # concatenated: one table of every baseline's inner delay
    check("every inner stream second of lr and rf has an inner out-of-fold prediction", bool((ID.grid_seconds == ID.grid_seconds_with_prediction).all()), f"{int(ID.grid_seconds.sum())} seconds")  # dict: (receiver, outer fold, share) -> comparator
    # all four baselines' inner delays: lr and rf computed, CUSUM and k-of-n saved (NaN = no inner detection = +inf)
    cs = gp[gp.detector.isin(["cusum", "kofn"])][["receiver", "outer", "detector", "share", "inner_median_delay"]].assign(inner_median_delay=lambda d: d.inner_median_delay.fillna(np.inf))
    allin = pd.concat([ID[["receiver", "outer", "detector", "share", "inner_median_delay"]], cs], ignore_index=True)  # dict: baseline -> inner delay of this group
    # the fold-specific comparator: lowest inner delay, ties in the evaluation script's BASE order (min over (delay, position))
    comp = {}
    # for loop over the receiver, outer fold and share groups
    for (rx, o, s), g in allin.groupby(["receiver", "outer", "share"]):
        # dict: baseline -> inner delay
        dd = dict(zip(g.detector, g.inner_median_delay))
        # the comparator of this fold
        comp[(rx, o, round(s, 4))] = min(B12.BASE, key=lambda b: (dd[b], B12.BASE.index(b)))
    # the published verdicts, for the comparison
    pub = B12.rt(B12.EVAL / "verdicts.csv")
    # list of result rows
    out = []
    # for loop over the published cells
    for v in pub.itertuples():
        # the fold -> comparator map of this receiver and share
        cm = {o: comp[(v.receiver, o, round(v.share, 4))] for o in T.SESSIONS}
        # the comparators in fold order, as text
        ctext = "; ".join(f"{o}:{cm[o]}" for o in T.SESSIONS)
        # if: no agent point at this share
        if str(v.verdict).startswith("no point"):
            # recorded
            out.append({"receiver": v.receiver, "share": v.share, "arm": v.arm, "published_best_baseline": v.best_baseline, "published_verdict": v.verdict, "comparators_by_fold": ctext, "verdict_training_only": "no point", "differs": False})
            continue  # continue: next cell
        # the paired differences against each run's fold comparator
        a, d = paired(at, v.receiver, v.share, v.arm, cm)
        # the evaluation script's IQM and interval at the published seed
        p, lo, hi, nn = B12.strat_boot(d, a.outer.to_numpy())
        # the agent's pooled miss rate (as published)
        a_miss = float(v.agent_miss_rate)
        # the comparators' pooled miss rate on the agent's folds: each fold's comparator's intervals in that fold
        bl = pd.concat([B12.latency_at(LAT, pts, v.receiver, v.share, cm[o]).query("outer == @o") for o in sorted(set(a.outer))], ignore_index=True)
        b_miss = float(1 - bl.detected.mean()) if len(bl) else np.nan  # the comparators' detected flag over those intervals, as a miss rate (NaN without intervals)
        # the evaluation script's rule
        vt = rule(lo, hi, a_miss > b_miss)
        # the row
        out.append({"receiver": v.receiver, "share": v.share, "arm": v.arm, "published_best_baseline": v.best_baseline, "published_verdict": v.verdict, "comparators_by_fold": ctext,
                    "delay_diff_iqm_s": p, "ci_low": lo, "ci_high": hi, "paired_runs": nn, "agent_miss_rate": a_miss, "comparator_miss_rate_agent_folds": b_miss,  # continued: the paired difference, its interval and the miss rates
                    "verdict_training_only": vt, "differs": vt != v.verdict})  # continued: the verdict and whether it differs from the published one
    # DataFrame, written
    R = pd.DataFrame(out)
    R.to_csv(ROBUST_DIR / "comparator_sensitivity.csv", index=False, lineterminator="\n")  # written with fixed line endings
    # printed
    print(R.drop(columns=["comparators_by_fold"]).round(2).to_string(index=False))
    print("comparators by fold:\n" + R[["receiver", "share", "arm", "comparators_by_fold"]].drop_duplicates(["receiver", "share"]).to_string(index=False))  # printed: the comparator of each fold per receiver and share
    # the table
    return R


# function: main
def main() -> None:
    # the clock
    t0 = time.time()
    # the corrected point runs, every interval and the points of the confirmatory evaluation
    at, LAT, pts = B12.rt(B12.EVAL / "point_runs.csv"), B12.rt(B12.EVAL / "heldout_latency.csv"), B12.rt(B12.EVAL / "deployable_points.csv")
    # the three items
    item4()
    item5(at, LAT, pts)  # the bootstrap margin
    item6(at, LAT, pts)  # the comparator
    # the checks, written
    pd.DataFrame(CHECKS).to_csv(ROBUST_DIR / "margins_checks.csv", index=False, lineterminator="\n")
    # summary line
    print(f"done in {time.time() - t0:.0f} s; {len(CHECKS)} checks, {sum(not c['pass'] for c in CHECKS)} failed")


# if: run as a script
if __name__ == "__main__":
    # runs the three items
    main()
