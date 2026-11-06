#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
The 2.6.12 point runs with their onset marks, the table Section 5.4 counts: the runs at the operating points on the unseen attack
2.6.12 (Data/processed/transfer/unseen_attack/point_runs.csv) joined to every run's state at the onset (onset_states.csv in the same
folder). The baselines are joined by receiver, detector, share, lambda and seed; the agents by receiver, detector, lambda and seed,
because one agent run can sit at two shares. Every point run must find its onset state.

Usage   PYTHONPATH=codes/common python3 codes/7_evaluation/05_unseen_attack_point_marks.py
Writes  Data/processed/transfer/unseen_attack/point_marks.csv
"""

from __future__ import annotations  # future import: type hints kept as text, so they cost nothing at run time

from pathlib import Path  # Path objects for explicit file paths

import pandas as pd  # pandas: reads the data products

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/

PROC = "Data/processed"  # the data products, relative to the project (the manifest stores relative paths)
POINT_MARKS = f"{PROC}/transfer/unseen_attack/point_marks.csv"  # the derived table of 2.6.12's point runs with their marks




def point_marks() -> pd.DataFrame:  # function: the derived table of 2.6.12's point runs with their onset marks (written to transfer/unseen_attack/)
    pr = pd.read_csv(PROJECT / PROC / "transfer" / "unseen_attack" / "point_runs.csv")  # the 101 runs at the operating points
    os_ = pd.read_csv(PROJECT / PROC / "transfer" / "unseen_attack" / "onset_states.csv")  # every run's state at the onset
    keep = ["receiver", "detector", "share", "lambda", "seed", "onset_state", "refractory_at_onset", "no_detection_delay"]  # the columns kept
    base = pr[~pr.detector.str.startswith("agent")].merge(os_[os_.share.notna()], on=["receiver", "detector", "share", "lambda", "seed"], how="left", validate="one_to_one")  # the baselines by share
    agents = pr[pr.detector.str.startswith("agent")].merge(os_[os_.share.isna()].drop(columns="share"), on=["receiver", "detector", "lambda", "seed"], how="left", validate="many_to_one")  # the agents by lambda and seed (one run can sit at two shares)
    out = pd.concat([base, agents])[keep].sort_values(["receiver", "detector", "share", "seed"], kind="stable").reset_index(drop=True)  # together, in a fixed order
    assert len(out) == len(pr) and out.onset_state.notna().all(), "a point run without its onset state"  # every run found
    return out  # the table


def build() -> object:  # function: writes the table of 2.6.12's point runs with their onset marks (the rest of the source this file is taken from is left out)
    out = PROJECT / POINT_MARKS  # the derived table's path
    out.parent.mkdir(parents=True, exist_ok=True)  # its folder (exists after the diagnostic)
    point_marks().to_csv(out, index=False, lineterminator="\n")  # written before anything reads it


if __name__ == "__main__":  # if: run as a script, not imported
    build()  # writes Data/processed/transfer/unseen_attack/point_marks.csv, the call that wrote the packaged table
