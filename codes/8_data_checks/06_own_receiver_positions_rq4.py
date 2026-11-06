#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
Each receiver at its own reference position, carried to the RQ4 tables the thesis prints. The RQ4 script's intervals, comparison,
gain depression, strength pairs and rank correlations are recomputed from the monitor tables of the two runs of
05_own_receiver_positions.py (today's site positions, the gate; each receiver's own position, per_receiver), which that script writes
into Data/processed/robustness_checks/mirror/<run>/rq4/.
Gate: the run at today's positions must reproduce intervals, comparison, depression, strength_pairs and spearman byte for byte before
any per-receiver value is read. Then every cell that differs is listed in long form. Nothing published is overwritten: the
per-receiver tables stay in the mirror (Figure 7 reads its comparison table from there).
Usage   python3 codes/8_data_checks/06_own_receiver_positions_rq4.py
Writes  Data/processed/robustness_checks/positions_compare_rq4_comparison.csv, positions_compare_rq4_spearman.csv,
        positions_rq4_checks.csv; the per-receiver RQ4 tables under Data/processed/robustness_checks/mirror/per_receiver/rq4/
"""
# import: importlib loads the RQ4 script and an independent evaluator by path
import importlib.util
# import: shutil copies the per-receiver tables into a shadow root for the evaluator
import shutil
# import: sys registers modules and gives the exit status
import sys
# import: Path for explicit paths
from pathlib import Path
# import: numpy for NaN-aware comparisons
import numpy as np
# import: pandas for the tables
import pandas as pd

# project root (this file is codes/8_data_checks/06_own_receiver_positions_rq4.py)
PROJECT = Path(__file__).resolve().parents[2]
# processed data, the position check's folder, the mirror and the published RQ4 folder
PROC, ROBUST_DIR = PROJECT / "Data" / "processed", PROJECT / "Data" / "processed" / "robustness_checks"
MIRROR, RQ4_PUBLISHED = ROBUST_DIR / "mirror", PROJECT / "Data" / "processed" / "rq4"  # the scratch root of the own-position check and the RQ4 script's folder
# list: the tables recomputed here, in the RQ4 script's main's names
TABLES = ["intervals.csv", "comparison.csv", "depression.csv", "strength_pairs.csv", "spearman.csv"]
# list of check rows
CHECKS = []


# function: loads a project script as a module under a name (its main() stays behind its guard)
def load(name: str, rel: str):
    # where the file is
    spec = importlib.util.spec_from_file_location(name, PROJECT / rel)
    # an empty module object
    mod = importlib.util.module_from_spec(spec)
    # registered so its own imports resolve
    sys.modules[name] = mod
    # its top level (definitions and module loads only)
    spec.loader.exec_module(mod)
    # the module
    return mod


# 14 (its interval, comparison, depression and strength functions, its writer and the evaluation script's exact reader)
RQ4 = load("delay_monitors_cost_for_own_positions_rq4", "codes/7_evaluation/02_delay_monitors_cost.py")

# function: records and prints one check
def check(name: str, ok: bool, detail: str = "") -> None:
    # appended
    CHECKS.append({"check": name, "pass": bool(ok), "detail": detail})
    # printed
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


# function: the RQ4 script's main's steps 2-3 from one mode's monitor tables, written into that mode's mirror folder
def tables(mode: str, st: pd.DataFrame, MK12: pd.DataFrame) -> dict:
    # the mode's monitor folder
    d = MIRROR / mode / "rq4"
    # the monitor runs, intervals and coverage, read back exactly
    RUNS, LATm, COV = (RQ4.B12.rt(d / n) for n in ("monitor_runs.csv", "monitor_latency.csv", "monitor_coverage.csv"))
    # every detector's and monitor's intervals at the points (the RQ4 script's main's line)
    IV = pd.concat([RQ4.detector_intervals(MK12), RQ4.monitor_intervals(LATm)], ignore_index=True)
    CMP = RQ4.comparison(IV, RUNS, COV)  # the comparison table
    DEP = RQ4.depression(IV, st)  # each interval's gain depression
    PAIRS, RHO = RQ4.strength(IV, DEP)  # tuple unpacking: the strength pairs and the rank correlations
    # dict: name -> table, in the RQ4 script's main's names
    out = dict(zip(TABLES, (IV, CMP, DEP, PAIRS, RHO)))
    # for loop: each written with the RQ4 script's writer
    for n, df in out.items():
        RQ4.E.write_csv(df, d / n)  # written into the mirror
    # the tables
    return out


# function: every cell that differs between two tables with the same keys, in long form (NaN equal to NaN)
def long_diff(a: pd.DataFrame, b: pd.DataFrame, keys: list) -> pd.DataFrame:
    # merged one to one on the keys
    m = a.merge(b, on=keys, how="outer", suffixes=("_published", "_per_receiver"), indicator=True, validate="one_to_one")
    # list of rows
    rows = []
    # for loop over the non-key columns
    for c in [c for c in a.columns if c not in keys]:
        # the two columns
        x, y = m[f"{c}_published"], m[f"{c}_per_receiver"]
        # boolean mask: cells that differ (both missing counts as equal)
        dif = ~((x == y) | (x.isna() & y.isna()))
        # for loop over the differing rows
        for i in np.flatnonzero(dif.to_numpy()):
            rows.append({**{k: m[k].iloc[i] for k in keys}, "column": c, "published": x.iloc[i], "per_receiver": y.iloc[i]})  # dict unpacking: the keys, then the cell
    # the table (empty with its columns when nothing differs)
    return pd.DataFrame(rows, columns=keys + ["column", "published", "per_receiver"])


# function: main
def main() -> None:
    # the transfer script's freeze guard before 2.6.12's gains are read again (as the RQ4 script's main)
    proof = RQ4.X.guard()
    check("the transfer script's freeze guard holds before 2.6.12's gains are read", bool((proof.in_freeze_commit & proof.equal_to_commit & proof.sha256_matches_manifest).all()), f"{len(proof)} frozen files")  # recorded
    # the gains, as the RQ4 script's main reads them
    st = pd.read_csv(RQ4.E.PROC / "ground_truth" / "receiver_status_timeseries.csv", dtype={"test_code": str}, usecols=["test_code", "receiver", "timestamp_utc", "band", "agc_gain_db", "agc_cnt"])
    # the agent experiment's repeated point runs with both marks, as 14 wrote them (the detectors do not use a reference position)
    MK12 = RQ4.B12.rt(RQ4_PUBLISHED / "agent_marks.csv")
    # the two modes
    T = {mode: tables(mode, st, MK12) for mode in ("gate", "per_receiver")}
    # the gate: every table byte for byte against the published one
    for n in TABLES:
        check(f"gate: {n} = published", (MIRROR / "gate" / "rq4" / n).read_bytes() == (RQ4_PUBLISHED / n).read_bytes())  # byte comparison
    # if: a gate failed, no per-receiver value is reported
    if not all(c["pass"] for c in CHECKS):
        pd.DataFrame(CHECKS).to_csv(ROBUST_DIR / "positions_rq4_checks.csv", index=False, lineterminator="\n")  # the checks so far
        raise SystemExit("STOP: a gate failed; no per-receiver result is reported")  # stop
    # the comparison table: every differing cell, and only the position monitor may differ
    k = ["scope", "receiver", "share", "detector"]
    cd = long_diff(T["gate"]["comparison.csv"], T["per_receiver"]["comparison.csv"], k)  # long form
    cd.to_csv(ROBUST_DIR / "positions_compare_rq4_comparison.csv", index=False, lineterminator="\n")  # written
    check("comparison: only the position monitor's rows differ", bool((cd.detector == "position").all()), f"{len(cd)} cells in {cd[k].drop_duplicates().shape[0]} rows")  # recorded
    # the rank correlations: keys are every text column
    rg, rp = T["gate"]["spearman.csv"], T["per_receiver"]["spearman.csv"]
    rk = [c for c in rg.columns if rg[c].dtype == object]  # list comprehension: the text columns name a row
    sd = long_diff(rg, rp, rk)  # long form
    sd.to_csv(ROBUST_DIR / "positions_compare_rq4_spearman.csv", index=False, lineterminator="\n")  # written
    check("rank correlations: rows that differ listed", True, f"{len(sd)} cells in {sd[rk].drop_duplicates().shape[0]} rows")  # recorded (a report, not a pass condition)
    # the checks written
    pd.DataFrame(CHECKS).to_csv(ROBUST_DIR / "positions_rq4_checks.csv", index=False, lineterminator="\n")
    # status 1 on a failed check
    sys.exit(0 if all(c["pass"] for c in CHECKS) else 1)


# if: run as a script
if __name__ == "__main__":
    # runs the RQ4 comparison
    main()
