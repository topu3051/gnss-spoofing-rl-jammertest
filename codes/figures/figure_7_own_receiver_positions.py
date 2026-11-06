#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
Figure 7 as the thesis prints it: the within-receiver intervals of the monitors, with the position monitor referenced to each
receiver's own position. The drawing is figures_6_to_9.fig_marks() unchanged; only its input folder points at the per-receiver RQ4
tables (Data/processed/robustness_checks/mirror/per_receiver/rq4/comparison.csv), and the file is saved as figure_7.png, so the
site-position drawing of figures_6_to_9.py (figure_7_site_positions.png) stays beside it. Before drawing, every row of the per-receiver
table that is not the position monitor's is proven equal to the published rq4/comparison.csv, so the two drawings differ only in the
position-monitor bars.

Usage   python3 codes/figures/figure_7_own_receiver_positions.py
Writes  Graph/figure_7.png; exit status 1 if a non-position row differs
"""

from __future__ import annotations  # future import: type hints kept as text, so they cost nothing at run time

import sys  # standard library: the import path and the exit status
from pathlib import Path  # Path objects for explicit file paths

import pandas as pd  # pandas: reads the two comparison tables for the equality proof

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
sys.path.insert(0, str(PROJECT / "codes" / "figures"))  # list method insert: codes/figures first on the import path, so figures_6_to_9 is found
import figures_6_to_9 as C  # noqa: E402  the module of Figures 6 to 9; importing it draws nothing (its main() runs only as a script)

PUBLISHED = PROJECT / "Data" / "processed" / "rq4" / "comparison.csv"  # the published RQ4 comparison (site positions)
PER_RX = PROJECT / "Data" / "processed" / "robustness_checks" / "mirror" / "per_receiver"  # the position check's per-receiver products; holds rq4/comparison.csv


def main() -> int:  # function definition: proof, then the figure; returns the exit status
    a = pd.read_csv(PUBLISHED)  # the published table, every scope, receiver, share and detector
    b = pd.read_csv(PER_RX / "rq4" / "comparison.csv")  # the same table with each receiver's own reference position
    keys = ["scope", "receiver", "share", "detector"]  # list: the columns that name a row
    a, b = a.sort_values(keys).reset_index(drop=True), b.sort_values(keys).reset_index(drop=True)  # tuple assignment: both in one row order
    same_rows = a[keys].equals(b[keys])  # bool: the two tables hold the same rows
    other = a.detector != "position"  # boolean Series: rows of every detector except the position monitor
    same_other = same_rows and a[other].equals(b[other])  # bool: every non-position row equal in every column (NaN equal to NaN)
    print(f"  rows identical: {same_rows}; non-position rows identical: {same_other} ({int(other.sum())} rows)")  # for the run log
    if not same_other:  # if: the per-receiver table changed more than the position monitor
        return 1  # exit status 1, and no figure is drawn
    C.PROC = PER_RX  # module attribute rebound: fig_marks() reads PROC / "the RQ4 study" / "comparison.csv", now the per-receiver table
    save = C.save  # the module's own save(): writes FIG_DIR / name at 300 dpi with no software tag
    C.save = lambda fig, name: save(fig, name.replace("_site_positions.png", ".png"))  # lambda: the same save, the name's "_site_positions" dropped, so this drawing is figure_7.png and the site-position drawing stays
    C.fig_marks()  # Figure 7, drawn by the unchanged function
    print(f"  written {(C.FIG_DIR / "figure_7.png").relative_to(PROJECT)}")  # for the run log
    return 0  # exit status 0


if __name__ == "__main__":  # run only when executed as a script, not when imported
    sys.exit(main())  # sys.exit(): pass main()'s status to the shell
