#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
Figures 6-9, drawn from the data products with the plain styling of Figures 3 to 5 (no title inside the image; the caption
is in the thesis). Colours: blue #0072B2, orange #E69F00, green #009E73 and greys, every class also
named in a legend or label, so identity never rests on colour alone.

Figure 6  fig_rq1    classification of each observable family per fold with clean windows, forest and regression, with the permutation
                     control's verdict above each column (random_forest/random_forest_c4b_classification*.csv, baselines_controls/r3_verdicts.csv)
Figure 7  fig_marks  within-receiver intervals at the share 0.05 per receiver and detector: counted, no detection delay, missed
                     (rq4/comparison.csv, position sessions)
Figure 8  fig_diag   the diagnosis on inner folds: agent minus cumulative-sum cost per training episode; margin AUROC against log-odds
                     AUROC (rl_diagnosis/diagnosis_optimisation_cells.csv, rl_diagnosis/diagnosis_learning_cells.csv; read only)
Figure 9  fig_shift  2.6.12's clean pre-test: the decision-value shift by group of columns per receiver; predicted against tracked GPS
                     satellites (transfer/unseen_attack/clean_alarm_diagnosis/contributions.csv, identity.csv, visibility_summary.csv)
Figure 6's legend names the grey class "used but neither", the words of its caption and of Section 5.1 (the data keep
          the class name "not used").
Usage   /opt/anaconda3/bin/python3 codes/figures/figures_6_to_9.py
Writes  Graph/figure_6.png, figure_7_site_positions.png, figure_8.png, figure_9.png
"""

from __future__ import annotations  # future import: type hints kept as text, so they cost nothing at run time

import os  # standard library: sets matplotlib's cache folder before matplotlib is imported
from pathlib import Path  # Path objects for explicit file paths

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
os.environ["MPLCONFIGDIR"] = str(PROJECT / "codes" / ".mplconfig")  # matplotlib's cache inside the project folder

import numpy as np  # NumPy: positions and arrays
import pandas as pd  # pandas: reads the CSV products
import matplotlib  # the plotting library; imported first so its backend can be set before pyplot loads
matplotlib.use("Agg")  # file-only backend: no window, PNG bytes independent of a screen
import matplotlib.pyplot as plt  # pyplot: the plotting interface
from matplotlib.patches import Patch  # legend entries drawn as filled boxes
from matplotlib.lines import Line2D  # legend entries drawn as markers

PROC = PROJECT / "Data" / "processed"  # the data products (read only)
FIG_DIR = PROJECT / "Graph"  # where the figures are written (Graph/figure_N.png)
RX_NAME = {"SBF_02": "mosaic-X5", "SBF_03": "mosaic-H", "UBX_01": "u-blox 01"}  # dict: stream -> receiver name
RX_COLOR = {"SBF_02": "#0072B2", "SBF_03": "#E69F00", "UBX_01": "#009E73"}  # dict: stream -> colour (as in Figures 3 to 5)
BLUE, ORANGE, GREY, LIGHT = "#0072B2", "#E69F00", "#8C8C8C", "#D9D9D9"  # tuple assignment: the colours of classes
FAMILY = {"cn0": "C/N₀", "code_second_difference": "code second difference", "code_doppler_consistency": "code–Doppler consistency", "carrier_phase": "carrier phase", "tracking_count": "tracking count", "doppler": "Doppler", "code_rate": "code rate", "clock_step": "clock step", "missing_indicator": "missing indicator"}  # dict: family code -> thesis name, in row order
CLASS_COLOR = {"attack evidence": BLUE, "session carrier": ORANGE, "not used": LIGHT, "not selected": "white"}  # dict: class -> cell colour
LEGEND = {"not used": "used but neither"}  # dict: the legend wording of a class where it differs from the data's name, so the legend says what the caption and Section 5.1 say
CONTROL = {"supports": "supports", "fails": "fails", "uninformative": "uninformative"}  # dict: control verdict -> label above a column (written out, finding C29)
DPI = 300  # print resolution
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})  # dict: one font and size for every label, as in Figures 3 to 5


def save(fig, name: str) -> None:  # function: writes a figure without a software tag, so runs again give the same bytes
    FIG_DIR.mkdir(parents=True, exist_ok=True)  # the folder
    fig.savefig(FIG_DIR / name, dpi=DPI, metadata={"Software": None})  # PNG for print
    plt.close(fig)  # memory freed


def fig_rq1() -> None:  # function: Figure 6, the classification grids with the control verdicts
    r3 = pd.read_csv(PROC / "baselines_controls" / "r3_verdicts.csv", dtype={"held_out": str})  # the control's verdicts
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.9), sharey=True)  # two panels side by side, one row axis (taller, for the written-out verdicts)
    for ax, (model, path, title) in zip(axes, (("rf", "random_forest_c4b_classification.csv", "random forest"), ("lr", "random_forest_c4b_classification_logreg.csv", "logistic regression"))):  # for loop over the two models
        d = pd.read_csv(PROC / "random_forest" / path, dtype={"held_out": str})  # the classification
        cols = list(dict.fromkeys(zip(d.receiver, d.held_out)))  # list of (receiver, fold) in file order, duplicates removed
        for j, (rx, h) in enumerate(cols):  # for loop over the columns
            for i, fam in enumerate(FAMILY):  # for loop over the rows
                row = d[(d.receiver == rx) & (d.held_out == h) & (d.family == fam)]  # the cell
                cls = row["class"].iloc[0] if len(row) else "not selected"  # its class
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=CLASS_COLOR.get(cls, LIGHT), edgecolor="white" if cls != "not selected" else LIGHT, linewidth=1.5))  # the cell, with a 2-px-like white gap
                if len(row) and bool(row.e14_selection_dependent.iloc[0]) and cls == "attack evidence":  # if: a call resting on the control's selection
                    ax.text(j + 0.5, i + 0.5, "s", ha="center", va="center", color="white", fontsize=7.5, fontweight="bold")  # marked "s"
            v = r3[(r3.model == model) & (r3.receiver == rx) & (r3.held_out == h)].verdict.iloc[0]  # the control's verdict on this fold
            ax.text(j + 0.5, -0.25, CONTROL[v], ha="center", va="bottom", rotation=90, fontsize=7.5, color="black" if v == "supports" else "#B00000")  # above the column
        ax.set_xlim(0, len(cols))  # columns
        ax.set_ylim(len(FAMILY), -3.4)  # rows top to bottom, room for the verdicts (more room for "uninformative")
        ax.set_xticks(np.arange(len(cols)) + 0.5, [f"{RX_NAME[rx]}\n{h}" for rx, h in cols], rotation=90, fontsize=7.5)  # list comprehension: column labels
        ax.set_yticks(np.arange(len(FAMILY)) + 0.5, list(FAMILY.values()))  # row labels
        ax.tick_params(length=0)  # no tick marks
        for s in ax.spines.values():  # for loop over the frame lines
            s.set_visible(False)  # removed (recessive frame)
        ax.set_xlabel(title)  # the model under the panel
    handles = [Patch(facecolor=c, edgecolor=LIGHT, label=LEGEND.get(k, k)) for k, c in CLASS_COLOR.items()]  # list comprehension: legend boxes, each labelled by its LEGEND wording
    handles.append(Patch(facecolor=BLUE, edgecolor=LIGHT, label="s: selection-dependent"))  # the mark's legend entry
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7.5)  # one legend under both panels, in two rows (the longer label no longer fits in one row)
    fig.tight_layout(rect=(0, 0.1, 1, 1))  # spacing, room for the two-row legend
    save(fig, "figure_6.png")  # written


def fig_marks() -> None:  # function: Figure 7, intervals at 0.05 counted / no detection delay / missed
    c = pd.read_csv(PROC / "rq4" / "comparison.csv")  # the RQ4 study's comparison table
    c = c[(c.scope == "position sessions") & (c.share == 0.05)]  # the five position sessions at the share 0.05
    order = [("lr", "regression"), ("rf", "forest"), ("cusum", "CUSUM"), ("kofn", "k of n"), ("agent_primary", "agent, primary"), ("agent_surrogate", "agent, surrogate"), ("gain", "gain monitor"), ("position", "position monitor")]  # list: detectors in display order
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.9), sharey=True, sharex=True)  # three panels, one per receiver
    for ax, rx in zip(axes, RX_NAME):  # for loop over the receivers
        for i, (det, name) in enumerate(order):  # for loop over the detectors
            r = c[(c.receiver == rx) & (c.detector == det)]  # its row
            if not len(r) or int(r.intervals.iloc[0]) == 0:  # if: no such detector on this receiver
                ax.text(0.2, i, "no position output" if det == "position" else "no point", va="center", fontsize=7.5, color=GREY)  # said so
                continue  # next detector
            n, dt, ct = int(r.intervals.iloc[0]), int(r.detected.iloc[0]), int(r.counted.iloc[0])  # tuple assignment: intervals, detected, counted
            ax.barh(i, ct, color=BLUE, height=0.7, edgecolor="white", linewidth=1)  # counted
            ax.barh(i, dt - ct, left=ct, color=ORANGE, height=0.7, edgecolor="white", linewidth=1)  # no detection delay
            ax.barh(i, n - dt, left=dt, color=LIGHT, height=0.7, edgecolor="white", linewidth=1)  # missed
            ax.text(n + 0.2, i, f"{ct}/{dt}/{n}", va="center", fontsize=7.5)  # counted / detected / intervals, at the bar's end
        ax.set_yticks(range(len(order)), [nm for _, nm in order])  # list comprehension: detector names
        ax.invert_yaxis()  # first detector on top
        ax.set_xlim(0, 11)  # room for the labels
        ax.set_xlabel(f"intervals, {RX_NAME[rx]}")  # the receiver under its panel
        ax.spines[["top", "right"]].set_visible(False)  # recessive frame
    handles = [Patch(facecolor=BLUE, label="counted delay"), Patch(facecolor=ORANGE, label="no detection delay (in alarm or refractory at onset)"), Patch(facecolor=LIGHT, label="missed")]  # list: legend boxes
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7.5)  # one legend
    fig.tight_layout(rect=(0, 0.08, 1, 1))  # spacing
    save(fig, "figure_7_site_positions.png")  # written


def fig_diag() -> None:  # function: Figure 8, the optimisation and representation tests
    d1 = pd.read_csv(PROC / "rl_diagnosis" / "diagnosis_optimisation_cells.csv")  # agent minus CUSUM(h*) per cell
    d3 = pd.read_csv(PROC / "rl_diagnosis" / "diagnosis_learning_cells.csv")  # margin and log-odds AUROC per cell
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.3, 2.9), gridspec_kw={"width_ratios": [1.35, 1]})  # two panels
    lams = sorted(d1["lambda"].unique())  # the five lambdas
    mk = {"primary": "o", "surrogate": "s"}  # dict: state -> marker shape (secondary encoding)
    for k, rx in enumerate(RX_NAME):  # for loop over the receivers
        for arm, off in (("primary", -0.12), ("surrogate", 0.12)):  # for loop over the states, dodged
            r = d1[(d1.receiver == rx) & (d1.arm == arm)].sort_values("lambda")  # the cells
            x = np.array([lams.index(v) for v in r["lambda"]]) + off + (k - 1) * 0.06  # positions on the lambda axis
            a1.errorbar(x, r.train_diff_median, yerr=[r.train_diff_median - r.train_diff_lo, r.train_diff_hi - r.train_diff_median], fmt=mk[arm], color=RX_COLOR[rx], ms=4, lw=0.8, capsize=0, mfc=RX_COLOR[rx] if arm == "surrogate" else "white")  # median and interval
            r3 = d3[(d3.receiver == rx) & (d3.arm == arm)]  # the same cells' AUROCs
            a2.scatter(r3.auroc_z_median, r3.auroc_margin_median, marker=mk[arm], s=22, facecolor=RX_COLOR[rx] if arm == "surrogate" else "white", edgecolor=RX_COLOR[rx], linewidth=1)  # one point per cell
    a1.axhline(0, color="black", lw=0.6)  # zero: agent as good as the cumulative sum
    a1.set_yscale("symlog", linthresh=1)  # costs from below zero to hundreds
    a1.set_yticks([-10, -1, 0, 1, 10, 100], ["−10", "−1", "0", "1", "10", "100"])  # plain tick labels instead of −10⁰ and 10¹
    a1.set_xticks(range(len(lams)), [f"{v:g}" for v in lams])  # list comprehension: lambda labels
    a1.set_xlabel("λ (s of delay per false alarm)")  # axis label
    a1.set_ylabel("agent − CUSUM cost per\ntraining episode (s)")  # axis label
    a2.plot([0.5, 1], [0.5, 1], color="black", lw=0.6)  # the diagonal
    a2.set_xlim(0.5, 1.0)  # AUROC axis
    a2.set_ylim(0.5, 1.0)  # AUROC axis
    a2.set_xlabel("AUROC of the regression's log-odds")  # axis label
    a2.set_ylabel("AUROC of the agent's\naction-value margin")  # axis label
    for a in (a1, a2):  # for loop over the panels
        a.spines[["top", "right"]].set_visible(False)  # recessive frame
    handles = [Line2D([], [], color=RX_COLOR[rx], marker="o", ls="", label=RX_NAME[rx]) for rx in RX_NAME]  # list comprehension: receivers
    handles += [Line2D([], [], color="black", marker="o", mfc="white", ls="", label="primary state"), Line2D([], [], color="black", marker="s", ls="", label="surrogate state")]  # the states
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=7.5)  # one legend
    fig.tight_layout(rect=(0, 0.08, 1, 1))  # spacing
    save(fig, "figure_8.png")  # written


def fig_shift() -> None:  # function: Figure 9, the shift by group and the visibility count
    con = pd.read_csv(PROC / "transfer" / "unseen_attack" / "clean_alarm_diagnosis" / "contributions.csv")  # contributions per column
    con = con[con.reference.str.startswith("pooled")]  # against the pooled clean seconds
    grp = np.where(con.group == "change", "change", np.where(con.kind == "counts", "satellite counts", np.where(con.kind == "C/N0 level", "C/N₀ level", "other level")))  # nested where: each column's group
    con = con.assign(g=grp)  # attached
    groups = ["satellite counts", "C/N₀ level", "other level", "change"]  # display order
    vs = pd.read_csv(PROC / "transfer" / "unseen_attack" / "clean_alarm_diagnosis" / "visibility_summary.csv")  # predicted against tracked
    fig = plt.figure(figsize=(6.3, 3.3))  # the figure
    gs = fig.add_gridspec(3, 2, width_ratios=[1.05, 1], hspace=0.5, wspace=0.55)  # left one panel, right three rows
    a1 = fig.add_subplot(gs[:, 0])  # the decomposition
    for k, rx in enumerate(RX_NAME):  # for loop over the receivers
        s = con[con.receiver == rx].groupby("g").contribution.sum().reindex(groups).fillna(0.0)  # the sum per group
        a1.barh(np.arange(len(groups)) + (k - 1) * 0.27, s.to_numpy(), height=0.25, color=RX_COLOR[rx], label=RX_NAME[rx], edgecolor="white", linewidth=0.8)  # one bar per group
    a1.set_yticks(range(len(groups)), groups)  # group names
    a1.invert_yaxis()  # first group on top
    a1.axvline(0, color="black", lw=0.6)  # zero
    a1.set_xlabel("contribution to the shift of the\nmean decision value (2.6.12 clean pre-test)")  # axis label
    a1.legend(frameon=False, fontsize=7.5, loc="lower right")  # receivers
    a1.spines[["top", "right"]].set_visible(False)  # recessive frame
    order = [("2.6.5 clean (Stave stop)", "2.6.5\nStave"), ("2.6.2 clean", "2.6.2"), ("2.6.10 clean", "2.6.10"), ("2.6.4 clean", "2.6.4"), ("2.6.12 clean", "2.6.12")]  # list: groups and their labels
    for k, rx in enumerate(RX_NAME):  # for loop over the receivers
        ax = fig.add_subplot(gs[k, 1])  # its row
        for i, (g, lab) in enumerate(order):  # for loop over the groups
            r = vs[(vs.receiver == rx) & (vs.group == g)]  # the row
            if not len(r):  # if: no clean group for this receiver (the Septentrio receivers have none in 2.6.10)
                continue  # skipped
            pred = float(r.predicted_G_above_5_median.iloc[0])  # predicted above the mask
            trk = r.tracked_G_median.iloc[0] if pd.notna(r.tracked_G_median.iloc[0]) else r.tracked_G_window_mean.iloc[0]  # tracked (the window mean where the per-second table has no row)
            ax.bar(i - 0.2, pred, width=0.38, color=LIGHT, edgecolor="white")  # predicted
            if pd.notna(trk):  # if: a tracked count exists
                ax.bar(i + 0.2, float(trk), width=0.38, color=RX_COLOR[rx], edgecolor="white")  # tracked
        ax.set_xticks(range(len(order)), [lab for _, lab in order] if k == 2 else [""] * len(order), fontsize=7.5)  # labels on the bottom row only
        ax.set_ylim(0, 14)  # satellites
        ax.set_yticks([0, 6, 12])  # few ticks
        ax.set_ylabel("GPS satellites", fontsize=7.5)  # the quantity on every row
        ax.text(0.02, 1.02, RX_NAME[rx], transform=ax.transAxes, fontsize=7.5, va="bottom")  # the receiver above its row
        ax.spines[["top", "right"]].set_visible(False)  # recessive frame
        if k == 2:  # if: the bottom row
            ax.set_xlabel("group of clean seconds", fontsize=7.5)  # what the bars are grouped by
    fig.legend(handles=[Patch(facecolor=LIGHT, label="predicted from the orbits"), Patch(facecolor=GREY, label="tracked (in the receiver's colour)")], frameon=False, fontsize=7.5, ncol=2, loc="upper right", bbox_to_anchor=(0.99, 1.0))  # one legend above the right column, inside the figure (it was clipped, finding C27)
    fig.subplots_adjust(left=0.2, right=0.98, top=0.87, bottom=0.17)  # margins, room for the legend at the top
    save(fig, "figure_9.png")  # written


def main() -> None:  # function: all four figures
    fig_rq1()  # Figure 6
    fig_marks()  # Figure 7
    fig_diag()  # Figure 8
    fig_shift()  # Figure 9


if __name__ == "__main__":  # if: run as a script
    main()  # the figures
