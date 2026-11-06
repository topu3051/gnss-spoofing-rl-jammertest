#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
Figure 2: the processing chain from the raw receiver files to the detectors, drawn as a plain box diagram.

The upper row is the feature path (raw observations -> RINEX -> 1 Hz observables -> windowed features -> detectors -> evaluation);
the lower row is the labelling path (test catalogue and transmission plan, receiver status, vehicle speed -> ground truth -> labels).
A dashed line separates the two: the lower row reaches the detectors only as labels, never as a feature input. The figure states no
number, so nothing in it needs a data source.

Usage   /opt/anaconda3/bin/python3 codes/figures/figure_2_processing_chain.py
Writes  Graph/figure_2.png
"""

from __future__ import annotations  # future import: type hints kept as text, so they cost nothing at run time

import os  # standard library: sets matplotlib's cache folder before matplotlib is imported
from pathlib import Path  # Path objects for explicit file paths

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
os.environ["MPLCONFIGDIR"] = str(PROJECT / "codes" / ".mplconfig")  # matplotlib's cache inside the project folder, as the other scripts keep it

import matplotlib  # the plotting library; imported first so its backend can be set before pyplot loads
matplotlib.use("Agg")  # file-only backend: no window is opened, and the PNG bytes do not depend on a screen
import matplotlib.pyplot as plt  # pyplot: the plotting interface
from matplotlib.patches import FancyBboxPatch  # a rectangle with rounded corners, used for every box of the chain

OUT = PROJECT / "Graph" / "figure_2.png"  # the figure file
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "svg.hashsalt": "20260914"})  # dict: one font for every label; a fixed hash salt keeps outputs stable

# list of (x, y, width, height, text, fill colour): every box, in data coordinates of a 100 x 46 canvas
BOXES = [
    (1, 31, 16, 11, "Raw receiver files\n(Septentrio SBF,\nu-blox UBX)", "#ffffff"),  # where the chain starts: the native recordings
    (20.5, 31, 16, 11, "Conversion to\nRINEX 3.04\n(observations)", "#ffffff"),  # the conversion step of Section 4.5
    (40, 31, 16, 11, "1 Hz observables\nand residuals\n(L1, G1, E1, B1)", "#ffffff"),  # the per-second quantities on the four feature bands
    (59.5, 31, 16, 11, "Windowed features\n(1 to 10 s, each\nlabelled by its\nlast second)", "#ffffff"),  # the feature step of Section 4.7
    (79, 31, 20, 11, "Detectors:\nlogistic regression,\nrandom forest,\nsequential (RL)", "#ffffff"),  # the models of Sections 4.8 and 4.10
    (40, 17.5, 16, 9, "Broadcast\nephemeris (causal\nuse; residual\nvariants only)", "#ffffff"),  # the orbit input of the residuals, which only the variant sets use
    (79, 19, 20, 7.5, "Leave-one-\nsession-out\nevaluation", "#ffffff"),  # the evaluation of Section 4.9
    (1, 1.5, 16, 9, "Test catalogue\nand transmission\nplan", "#e6e6e6"),  # labelling source 1
    (20.5, 1.5, 16, 9, "Receiver status\n(gain, radio-\nfrequency and\ninterference flags)", "#e6e6e6"),  # labelling source 2, excluded from features ("radio-frequency" written out: RF stands for random forest in the list of abbreviations)
    (40, 1.5, 16, 9, "Vehicle speed\nand receiver\nvelocity", "#e6e6e6"),  # labelling source 3 (stillness witness)
    (59.5, 1.5, 16, 9, "Ground truth\nand per-second\nlabels", "#e6e6e6"),  # the labels of Sections 4.3 to 4.6
]  # end of the box list

# list of (start x, start y, end x, end y): arrows between box edges, in the same coordinates
ARROWS = [
    (17, 36.5, 20.5, 36.5), (36.5, 36.5, 40, 36.5), (56, 36.5, 59.5, 36.5), (75.5, 36.5, 79, 36.5),  # the feature row, left to right
    (48, 26.5, 48, 31),  # ephemeris up into the residuals
    (89, 31, 89, 26.5),  # detectors down to the evaluation
    (67.5, 12.5, 67.5, 10.5),  # the bus of the three sources down into the ground-truth box
    (75.5, 8, 89, 19),  # labels up to the evaluation (scoring)
    (75.5, 9.5, 83, 31),  # labels up to the detectors (training targets)
]  # end of the arrow list

# list of ((x0, x1), (y0, y1)): plain lines of the bus that joins the three labelling sources (no arrow heads)
BUS = [((9, 9), (10.5, 12.5)), ((28.5, 28.5), (10.5, 12.5)), ((48, 48), (10.5, 12.5)), ((9, 67.5), (12.5, 12.5))]  # three risers and the horizontal bus


def main() -> None:  # function: draws the diagram and saves it
    fig, ax = plt.subplots(figsize=(6.3, 3.0))  # one axes; 6.3 x 3.0 inches fits the 15 cm text width of the thesis
    ax.set_xlim(0, 100)  # canvas width in data units
    ax.set_ylim(0, 46)  # canvas height in data units
    ax.axis("off")  # no axes, ticks or frame: the figure is a diagram, not a plot
    for x, y, w, h, text, fill in BOXES:  # for loop with tuple unpacking over the boxes
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.8", fc=fill, ec="black", lw=0.8))  # the box outline
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5, linespacing=1.1)  # the box label, centred
    for x0, y0, x1, y1 in ARROWS:  # for loop over the arrows
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "-|>", "lw": 0.8, "color": "black", "shrinkA": 0, "shrinkB": 0})  # an arrow with a filled head
    for xs, ys in BUS:  # for loop over the bus segments
        ax.plot(xs, ys, lw=0.8, color="black")  # one plain segment of the bus
    ax.plot([0, 77], [15.5, 15.5], ls=(0, (4, 3)), lw=0.8, color="black")  # dashed line: the boundary between the feature and the labelling paths
    ax.text(1, 16.3, "feature path (above)", fontsize=7.5, ha="left", va="bottom", style="italic")  # label of the upper side of the boundary
    ax.text(1, 14.7, "labelling and verification only, never a feature input (below)", fontsize=7.5, ha="left", va="top", style="italic")  # label of the lower side
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)  # fixed margins instead of tight_layout, so the PNG is byte-stable
    fig.savefig(OUT, dpi=300, metadata={"Software": None})  # PNG at 300 dpi for print; no software tag, so runs again give the same bytes
    plt.close(fig)  # frees the figure
    print(f"  wrote {OUT.relative_to(PROJECT)}")  # one line for the run log


if __name__ == "__main__":  # if: run as a script, not imported
    main()  # draws the figure
