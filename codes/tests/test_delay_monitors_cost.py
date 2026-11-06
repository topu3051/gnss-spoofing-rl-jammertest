#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
The tests of 7_evaluation/02_delay_monitors_cost.py (fixed before the script ran).
  T14-G  the gain monitor equals the evidence script's own gain_alarm_runs at the evidence script's fixed settings (1, 2, 3, 5 dB; 100-500 counts), persistence 10, on
         2.6.2 and 2.6.4, every receiver: the seconds 14 alarms on = the seconds from the 10th of each the evidence script run to its end.
  T14-P  the position monitor equals the evidence script's own first_hold_time on 2.6.2 and 2.6.4 (mosaic-X5, u-blox 01) at 0.5, 1, 2, 5 and 50 m: at
         every fix, 14 alarms exactly when first_hold_time, run on the current run of fixes beyond the threshold up to that fix, finds
         a hold of 10 s (the run split as the evidence script splits it: a flip of the flag or a gap over 5 s).
  T14-N  the monitors' points recomputed with each evaluated session's status rows removed from the read path (and, for 2.6.12, no 2.6.12
         row at all) equal monitor_points.csv bitwise.
  T14-R  rho recomputed by hand (average ranks, Pearson correlation of the ranks) for one detector, and its bootstrap replayed in full
         (seed 20260914, 10,000 resamples): the interval and the count of defined resamples equal spearman.csv.
  T14-C  coverage reconciled with transfer/inspection_monitor_coverage.csv: the gain monitor's clean and defined-clean seconds equal the table's; the
         position monitor's clean seconds equal, and every difference in its defined-clean seconds is explained second by second: the evidence script's hold
         needs its first and last fix at least 10 s apart in fix time, the inspection table's rule 10 consecutive whole seconds with a fix, and the evidence script's run may
         cross a gap of up to 5 s. (The first version of this test assumed the position coverage could only grow; it shrank by 2 seconds in
         three sessions.)
  T14-B  the byte rules: two full runs equal in every output except cost_timing.csv, cost_summary.csv and the figure, and a --no-cost
         run rebuilds the summary and the figure from the timing file byte for byte (hash lists written by the shell around the runs).
Usage   tests/test_delay_monitors_cost.py   (writes Data/processed/rq4/tests.csv; exit status 1 on any failure)
It imports the RQ4 script (which loads the transfer script, the evidence script and the rest); nothing from Data/processed/checks/.
"""
from __future__ import annotations  # future import: type hints kept as text (the project's convention)

import importlib.util  # standard library: loads the RQ4 script (its name starts with a digit)
import sys  # standard library: module registry and exit status
from pathlib import Path  # standard library: paths

import numpy as np  # numpy: arrays
import pandas as pd  # pandas: tables

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
_spec = importlib.util.spec_from_file_location("delay_monitors_cost_for_tests", PROJECT / "codes" / "7_evaluation" / "02_delay_monitors_cost.py")  # where 14 is
R = importlib.util.module_from_spec(_spec)  # an empty module object
sys.modules["delay_monitors_cost_for_tests"] = R  # registered
_spec.loader.exec_module(R)  # the RQ4 script's top level (it loads the transfer script, the evidence script and the rest); its main() is not run
D, E, T, X = R.D, R.E, R.T, R.X  # the evidence, environment, training and transfer scripts as the RQ4 script holds them
ROWS = []  # the test results
POS = {}  # (receiver, session) -> the evidence script's own fixes inside its span, read once


def rec(test: str, ok: bool, detail: str) -> None:  # function: records one test
    ROWS.append({"test": test, "pass": bool(ok), "detail": detail})  # appended
    print(("PASS " if ok else "FAIL ") + test + " | " + str(detail)[:200], flush=True)  # printed


def fixes(gt: pd.DataFrame, rx: str, t: str) -> pd.DataFrame:  # function: the evidence script's own fixes of a receiver and session inside the evidence script's span, cached (a position session only)
    if (rx, t) not in POS:  # if: not yet read
        own = D.own_positions(gt, t, rx)  # 05d: the first valid fix of each second
        a0, b0 = D.capture_spans(gt, t, rx)[0]  # the evidence script's span
        POS[(rx, t)] = own[(own.sec >= a0) & (own.sec < b0)].reset_index(drop=True)  # kept
    return POS[(rx, t)]  # the fixes


def t_gain(st: pd.DataFrame) -> None:  # function: T14-G
    bad, n = [], 0  # mismatches; comparisons
    for rx in R.RXS:  # for loop over the receivers
        for t in ("2.6.2", "2.6.4"):  # for loop over the two sessions
            g = D.gain_drop(D.gain_series(st, t, rx))  # the evidence script's gain fall
            H = R.persistent_min(g.index.to_numpy(np.int64), g.to_numpy(float), R.PERSIST)  # the RQ4 script's held statistic
            for thr in R.FIXED["gain"][rx]:  # for loop over the evidence script's fixed settings
                mine = set(H.index[(H >= thr).to_numpy()].astype(int))  # set: the seconds 14 alarms on
                runs = D.gain_alarm_runs(g, thr, R.PERSIST)  # the evidence script's own runs of 10 seconds or more
                theirs = {int(s) for a, b in zip(runs.start, runs.end) for s in range(int(a) + R.PERSIST - 1, int(b) + 1)}  # set comprehension: from each run's 10th second to its end
                n += 1  # counted
                if mine != theirs:  # if: they differ
                    bad.append(f"{rx} {t} {thr}: {len(mine ^ theirs)} seconds")  # recorded
    rec("T14-G the gain monitor's alarms equal the evidence script's gain_alarm_runs at its fixed settings, persistence 10 (2.6.2, 2.6.4, three receivers)", not bad, f"{n} comparisons; mismatches {bad}")  # recorded


def t_position(gt: pd.DataFrame) -> None:  # function: T14-P
    bad, n, alarms = [], 0, 0  # mismatches; comparisons; alarming fixes seen
    for rx in ("SBF_02", "UBX_01"):  # for loop over the receivers with a position
        for t in ("2.6.2", "2.6.4"):  # for loop over the two sessions
            own = fixes(gt, rx, t)  # the evidence script's fixes
            tt = own.t.to_numpy(float)  # fix times
            dh = D.horiz_from_site(own.lat.to_numpy(), own.lon.to_numpy(), own.h.to_numpy(), D.site_xyz(D.M.SITE_OF[t]))  # distances from the site
            H = R.hold_stat(tt, own.sec.to_numpy(np.int64), dh).to_numpy()  # the RQ4 script's held statistic
            for thr in (0.5, 1.0, 2.0, 5.0, 50.0):  # for loop over thresholds (the evidence script's 50 m and lower ones, so that runs exist)
                flag = dh > thr  # the evidence script's flag (strictly beyond the threshold)
                mine = H > thr  # 14 at the same comparison (a point is applied as H >= its threshold, one float step above a value)
                brk = np.r_[True, (flag[1:] != flag[:-1]) | (np.diff(tt) > 5)]  # the evidence script's run split, as first_hold_time writes it
                start = np.maximum.accumulate(np.where(brk, np.arange(len(tt)), 0))  # each fix's run start (the latest break at or before it)
                theirs = np.array([bool(flag[i]) and D.first_hold_time(tt[start[i]:i + 1], flag[start[i]:i + 1], R.HOLD_S) is not None for i in range(len(tt))])  # list comprehension: the evidence script's own function on the run so far
                n += 1  # counted
                alarms += int(theirs.sum())  # alarming fixes
                if not np.array_equal(mine, theirs):  # if: they differ
                    bad.append(f"{rx} {t} {thr}: {int((mine != theirs).sum())} fixes")  # recorded
    rec("T14-P the position monitor's alarms equal the evidence script's first_hold_time on the run so far, at every fix (2.6.2, 2.6.4; mosaic-X5, u-blox 01; 0.5-50 m)", not bad and alarms > 0, f"{n} comparisons; {alarms} alarming fixes; mismatches {bad}")  # recorded


def t_points(st: pd.DataFrame, gt: pd.DataFrame) -> None:  # function: T14-N
    G = R.grids()  # the stream grids (labels only for 2.6.12, not read here: 2.6.12 is never a training session)
    want = X.rt(R.RQ4_DIR / "monitor_points.csv")  # the written points
    got = []  # recomputed
    for held in R.ALL6:  # for loop over the evaluated sessions
        st_h = st[st.test_code != held]  # the status rows with the evaluated session removed
        train = [s for s in R.SESS5 if s != held]  # its training sessions
        for mon in R.MONITORS:  # for loop over the monitors
            for rx in R.RXS:  # for loop over the receivers
                if mon == "position" and rx == "SBF_03":  # no position monitor on the mosaic-H
                    continue  # next
                vals = []  # the held statistic at the training sessions' clean seconds
                for s in train:  # for loop over the training sessions
                    sec = G[(rx, s)].sec.to_numpy(np.int64)  # its grid
                    if mon == "gain":  # the gain fall from the reduced status table
                        g = D.gain_drop(D.gain_series(st_h, s, rx))  # the evidence script
                        v = R.persistent_min(g.index.to_numpy(np.int64), g.to_numpy(float), R.PERSIST).reindex(sec).to_numpy(float)  # on the grid
                    else:  # the position from the training session's own raw file only
                        own = fixes(gt, rx, s)  # the evidence script's fixes
                        dh = D.horiz_from_site(own.lat.to_numpy(), own.lon.to_numpy(), own.h.to_numpy(), D.site_xyz(D.M.SITE_OF[s])) if len(own) else np.array([])  # distances
                        v = (R.hold_stat(own.t.to_numpy(float), own.sec.to_numpy(np.int64), dh).reindex(sec).to_numpy(float)) if len(own) else np.full(len(sec), np.nan)  # on the grid
                    vals.append(v[G[(rx, s)].label.to_numpy() == "clean"])  # its clean seconds
                cv = np.concatenate(vals)  # pooled
                for sh in R.SHARES:  # for loop over the shares
                    got.append({"monitor": mon, "receiver": rx, "held_out": held, "share": sh, "threshold": T.lowest_threshold(cv, len(cv), sh)})  # the recomputed point
    g_ = pd.DataFrame(got).merge(want[["monitor", "receiver", "held_out", "share", "threshold"]], on=["monitor", "receiver", "held_out", "share"], suffixes=("", "_w"), validate="one_to_one")  # side by side
    same = len(g_) == len(want) and np.array_equal(g_.threshold.to_numpy(float), g_.threshold_w.to_numpy(float), equal_nan=True)  # bitwise, NaN = NaN
    rec("T14-N the monitors' points recomputed with each evaluated session's rows removed (2.6.12: no 2.6.12 row at all) equal monitor_points.csv bitwise", same, f"{len(g_)} points of {len(want)}")  # recorded


def ranks_rho(x: np.ndarray, y: np.ndarray) -> float:  # function: Spearman's rho by hand: average ranks, then Pearson's correlation of the ranks
    rx_, ry_ = pd.Series(x).rank(method="average").to_numpy(), pd.Series(y).rank(method="average").to_numpy()  # tuple assignment: the ranks
    a, b = rx_ - rx_.mean(), ry_ - ry_.mean()  # tuple assignment: centred
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))  # the correlation


def t_rho() -> None:  # function: T14-R
    RHO = X.rt(R.RQ4_DIR / "spearman.csv")  # the written rhos
    P = X.rt(R.RQ4_DIR / "strength_pairs.csv")  # the pairs
    c = RHO[RHO.subset.str.startswith("counted delays (primary") & RHO.level.str.startswith("per receiver") & RHO.rho.notna()].sort_values(["pairs", "receiver", "detector"], ascending=[False, True, True]).iloc[0]  # the primary cell with the most pairs
    g = P[(P.receiver == c.receiver) & (P.detector == c.detector) & P.counted.astype(bool) & P.depression.notna()]  # its pairs
    x, y = g.depression.to_numpy(float), g.delay_s.to_numpy(float)  # tuple assignment: depression and counted delay
    rho = ranks_rho(x, y)  # by hand
    rng = np.random.default_rng(R.SEED)  # the same seeded generator
    idx = rng.integers(0, len(x), size=(R.N_BOOT, len(x)))  # the same resamples
    bs = [ranks_rho(x[k], y[k]) for k in idx if np.unique(x[k]).size > 1 and np.unique(y[k]).size > 1]  # list comprehension: every defined resample's rho by hand
    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))  # tuple assignment: the interval
    ok = abs(rho - c.rho) <= 1e-12 and abs(lo - c.ci_low) <= 1e-12 and abs(hi - c.ci_high) <= 1e-12 and len(bs) == int(c.defined_resamples)  # every figure equal
    rec(f"T14-R rho by hand for {c.receiver} {c.detector} ({int(c.pairs)} pairs) and the full bootstrap replayed = spearman.csv", ok, f"rho {rho:+.6f} vs {c.rho:+.6f}; interval [{lo:+.4f}, {hi:+.4f}] vs [{c.ci_low:+.4f}, {c.ci_high:+.4f}]; resamples {len(bs)} vs {int(c.defined_resamples)}")  # recorded


def t_coverage(gt: pd.DataFrame) -> None:  # function: T14-C
    cov = X.rt(R.RQ4_DIR / "monitor_coverage.csv")  # the RQ4 script's coverage
    inspection = pd.read_csv(E.PROC / "transfer" / "inspection_monitor_coverage.csv", dtype={"session": str})  # the inspection table's
    inspection = inspection.assign(monitor=inspection.monitor.map({"gain fall": "gain", "position departure": "position"}))  # the same monitor names
    m = cov[cov.session != R.SEALED].merge(inspection, on=["receiver", "session", "monitor"], how="inner", validate="one_to_one")  # side by side on the position sessions
    mh = inspection[(inspection.receiver == "SBF_03") & (inspection.monitor == "position")]  # the inspection table lists the mosaic-H's position monitor as not available
    g, p = m[m.monitor == "gain"], m[m.monitor == "position"]  # tuple assignment: the two monitors
    ok_g = (g.clean_seconds == g.clean_grid_seconds).all() and (g.defined_clean_seconds == g.held_defined_clean).all()  # gain: equal
    ok_c = (p.clean_seconds == p.clean_grid_seconds).all()  # position: the same clean seconds
    G = R.grids()  # the stream grids
    expl, notes = True, []  # whether every difference is explained; the per-session account
    for r in p[p.defined_clean_seconds != p.held_defined_clean].itertuples():  # for loop over the sessions where the position coverage differs from the inspection table's
        own = fixes(gt, r.receiver, r.session)  # the evidence script's fixes
        sec, tt = own.sec.to_numpy(np.int64), own.t.to_numpy(float)  # tuple assignment: whole seconds and fix times
        dh = D.horiz_from_site(own.lat.to_numpy(), own.lon.to_numpy(), own.h.to_numpy(), D.site_xyz(D.M.SITE_OF[r.session]))  # distances
        g_ = G[(r.receiver, r.session)]  # the grid
        clean = g_.sec.to_numpy(np.int64)[g_.label.to_numpy() == "clean"]  # its clean seconds
        inspection_def = R.persistent_min(sec, dh, R.PERSIST).reindex(clean).notna().to_numpy()  # the inspection table's rule: all of the last 10 seconds have a fix
        now_def = R.hold_stat(tt, sec, dh).reindex(clean).notna().to_numpy()  # the evidence script's hold
        only0 = clean[inspection_def & ~now_def]  # defined by the inspection table's rule only
        only1 = clean[now_def & ~inspection_def]  # defined by the evidence script's hold only (a run that crosses a gap of up to 5 s)
        sset = set(sec.tolist())  # the seconds with a fix
        seg = np.cumsum(np.r_[True, np.diff(tt) > 5])  # the gap-free stretch of each fix (a gap over 5 s ends a run)
        t_of = dict(zip(sec.tolist(), tt.tolist()))  # dict: second -> its fix time
        start_of = {int(sc): float(tt[seg == seg[i]].min()) for i, sc in enumerate(sec)}  # dict comprehension: second -> the first fix time of its stretch
        tenth = all(all((x - k) in sset for k in range(10)) and t_of[int(x)] - start_of[int(x)] < R.HOLD_S for x in only0)  # generator in all(): the inspection table's rule sees 10 whole seconds with a fix, but in fix time the stretch has lasted under 10 s, so the evidence script's hold is not yet reached
        diff_ok = len(only0) - len(only1) == r.held_defined_clean - r.defined_clean_seconds  # the counts reconcile
        expl = expl and tenth and diff_ok  # accumulated
        notes.append(f"{r.receiver} {r.session}: the inspection table {r.held_defined_clean}, now {r.defined_clean_seconds} ({len(only0)} seconds have 10 whole seconds of fixes but under 10 s of fix time, {len(only1)} cross a short gap)")  # the account
    rec("T14-C coverage reconciled with the inspection table: gain clean and defined-clean seconds equal; position clean seconds equal, and every defined-clean difference is the evidence script's span rule (10 s between fix times against the inspection table's 10 whole seconds) or a run across a gap of up to 5 s", ok_g and ok_c and expl and len(m) == len(inspection) - len(mh) and (mh.held_defined_clean == 0).all(), "; ".join(notes) + f"; the inspection table's {len(mh)} mosaic-H position rows have no coverage, and 14 has no such monitor")  # recorded


def t_bytes() -> None:  # function: T14-B, from the hash lists the shell wrote around the runs
    L = E.PROC / "transfer" / "logs"  # the logs
    h2, h3 = (L / "run2_sha256.txt").read_text().splitlines(), (L / "run3_sha256.txt").read_text().splitlines()  # tuple assignment: both full runs' hashes
    timing = ("cost_timing.csv", "cost_summary.csv", "cost.png")  # tuple: the files that depend on wall clock
    det2 = [x for x in h2 if not x.endswith(timing)]  # list comprehension: run 2's deterministic files
    det3 = [x for x in h3 if not x.endswith(timing)]  # run 3's
    rec("T14-B two full runs are byte-identical in every output except cost_timing.csv, cost_summary.csv and the figure", det2 == det3 and len(det3) > 0, f"{len(det3)} files compared")  # recorded
    nc = (L / "nocost_sha256.txt").read_text().splitlines()  # the --no-cost run's hashes
    pick = lambda h: [x for x in h if x.endswith(("cost_summary.csv", "cost.png"))]  # lambda: the summary and the figure
    rec("T14-B a --no-cost run rebuilds the summary and the figure from the timing file byte for byte, and changes nothing else", pick(nc) == pick(h3) and [x for x in nc if not x.endswith(("checks_nocost.csv",))] == [x for x in h3 if not x.endswith(("checks_nocost.csv",))], f"{len(pick(nc))} files")  # recorded


def main() -> None:  # function: every test
    gt = pd.read_csv(E.PROC / "ground_truth" / "session_ground_truth.csv", dtype={"test_code": str}).set_index("test_code")  # the timeline script's sessions, as the evidence script expects
    st = pd.read_csv(E.PROC / "ground_truth" / "receiver_status_timeseries.csv", dtype={"test_code": str}, usecols=["test_code", "receiver", "timestamp_utc", "band", "agc_gain_db", "agc_cnt"])  # the gains
    st = st[st.test_code != R.SEALED]  # 2.6.12's rows dropped: no test reads them
    t_gain(st)  # T14-G
    t_position(gt)  # T14-P
    t_points(st, gt)  # T14-N
    t_rho()  # T14-R
    t_coverage(gt)  # T14-C
    t_bytes()  # T14-B
    E.write_csv(pd.DataFrame(ROWS), R.RQ4_DIR / "tests.csv")  # written
    print(f"delay, monitor and cost tests: {sum(r['pass'] for r in ROWS)}/{len(ROWS)} pass")  # summary
    if not all(r["pass"] for r in ROWS):  # if: any failed
        sys.exit(1)  # exit status 1


if __name__ == "__main__":  # guard: runs main() only when the file is executed
    main()  # runs main
