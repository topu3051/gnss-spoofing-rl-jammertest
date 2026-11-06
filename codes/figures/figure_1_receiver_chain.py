#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
Figure 1: the receiver chain from the antenna to a RINEX observation file, as a plain box diagram.

The top row, inside a dashed frame for the receiver, follows the signal: antenna, front end (filter, ADC, AGC), acquisition and
tracking (correlators, DLL, PLL), measurement generation and the navigation solution. The second row shows what leaves each step:
grey boxes are quantities a commercial receiver outputs (AGC gain, the observables, its own status flags); the dashed white box holds
the tracking internals that only a software receiver makes available. The observables continue to the RINEX observation file. The
figure states no number, so nothing in it needs a data source; its content rests on the sources cited in Section 2.2.
Drawn in the plain print style of codes/figures/figure_2_processing_chain.py (not imported and not changed).

Usage   /opt/anaconda3/bin/python3 codes/figures/figure_1_receiver_chain.py
Writes  Graph/figure_1.png
"""

from __future__ import annotations  # future import: type hints kept as text, so they cost nothing at run time

import os  # standard library: sets matplotlib's cache folder before matplotlib is imported
import tempfile  # standard library: the system's temporary folder, so matplotlib's font cache is written outside the project folder
from pathlib import Path  # Path objects for explicit file paths

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "thesis2_ch2_mplconfig")  # matplotlib's cache outside the project: only paths inside the project are written

import matplotlib  # the plotting library; imported first so its backend can be set before pyplot loads
matplotlib.use("Agg")  # file-only backend: no window is opened, and the PNG bytes do not depend on a screen
import matplotlib.pyplot as plt  # pyplot: the plotting interface
from matplotlib.patches import FancyBboxPatch, Rectangle  # rounded boxes for the steps, a plain rectangle for the receiver frame

OUT = PROJECT / "Graph" / "figure_1.png"  # the figure file
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "svg.hashsalt": "20260914"})  # dict: one font for every label; a fixed hash salt keeps outputs stable
GREY = "#d9d9d9"  # fill of the boxes a commercial receiver outputs (prints as light grey)

# list of (x, y, width, height, text, fill, dashed edge): every box, in data coordinates of a 100 x 50 canvas
BOXES = [
    (1.5, 34, 10, 10, "Antenna", "#ffffff", False),  # where the signal enters
    (15, 34, 17, 10, "Front end\n(filter, ADC,\nAGC)", "#ffffff", False),  # amplification, filtering, digitisation and gain control
    (35.5, 34, 19, 10, "Acquisition and\ntracking\n(correlators,\nDLL, PLL)", "#ffffff", False),  # the search and the two loops
    (58, 34, 17, 10, "Measurement\ngeneration", "#ffffff", False),  # the measurement set of each tracked signal
    (78.5, 34, 20, 10, "Navigation\nsolution (position,\nvelocity, time)", "#ffffff", False),  # the least-squares solution of Section 2.1
    (15, 17, 17, 9, "AGC gain", GREY, False),  # output 1: the gain reading
    (35.5, 17, 19, 9, "Correlator outputs,\ndiscriminator\nvalues", "#ffffff", True),  # tracking internals: software receivers only
    (58, 17, 17, 9, "Pseudorange,\ncarrier phase,\nDoppler, C/N₀", GREY, False),  # output 2: the observables
    (78.5, 17, 20, 9, "Status flags:\ninterference, jamming,\nnon-authentic signals", GREY, False),  # output 3: the receiver's own verdict
    (58, 2, 17, 8.5, "RINEX\nobservation file", "#ffffff", False),  # where the observables end up after conversion
]  # end of the box list

# list of (start x, start y, end x, end y): arrows between box edges, in the same coordinates
ARROWS = [
    (11.5, 39, 15, 39), (32, 39, 35.5, 39), (54.5, 39, 58, 39), (75, 39, 78.5, 39),  # the signal path, left to right
    (23.5, 34, 23.5, 26), (45, 34, 45, 26), (66.5, 34, 66.5, 26),  # each step down to what it yields
    (88.5, 31, 88.5, 26),  # the receiver as a whole down to its status flags
    (66.5, 17, 66.5, 10.5),  # observables down to the RINEX file
]  # end of the arrow list


def main() -> None:  # function: draws the diagram and saves it
    OUT.parent.mkdir(parents=True, exist_ok=True)  # the figure folder Graph/ (created once)
    fig, ax = plt.subplots(figsize=(6.3, 3.2))  # one axes; 6.3 x 3.2 inches fits the 15 cm text width of the thesis
    ax.set_xlim(0, 100)  # canvas width in data units
    ax.set_ylim(0, 50)  # canvas height in data units
    ax.axis("off")  # no axes, ticks or frame: the figure is a diagram, not a plot
    ax.add_patch(Rectangle((0.5, 31), 99, 17, fill=False, ls=(0, (4, 3)), lw=0.8, ec="black"))  # dashed frame around the receiver's steps
    ax.text(1.5, 46.6, "receiver", fontsize=7.5, ha="left", va="center", style="italic")  # label of the frame
    for x, y, w, h, text, fill, dashed in BOXES:  # for loop with tuple unpacking over the boxes
        style = (0, (3, 2)) if dashed else "solid"  # conditional expression: a dashed edge for the software-only box
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.8", fc=fill, ec="black", lw=0.8, ls=style))  # the box outline
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5, linespacing=1.1)  # the box label, centred
    for x0, y0, x1, y1 in ARROWS:  # for loop over the arrows
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "-|>", "lw": 0.8, "color": "black", "shrinkA": 0, "shrinkB": 0})  # an arrow with a filled head
    ax.text(68, 13.8, "conversion\n(Section 4.5)", fontsize=7.5, ha="left", va="center", style="italic")  # the step between the observables and the file
    ax.add_patch(Rectangle((1.5, 9.5), 3.5, 2.5, fc=GREY, ec="black", lw=0.6))  # legend swatch: grey box
    ax.text(6, 10.75, "output by a commercial receiver", fontsize=7.5, ha="left", va="center")  # legend text 1
    ax.add_patch(Rectangle((1.5, 4.5), 3.5, 2.5, fc="#ffffff", ec="black", lw=0.6, ls=(0, (3, 2))))  # legend swatch: dashed white box
    ax.text(6, 5.75, "available only in a software receiver", fontsize=7.5, ha="left", va="center")  # legend text 2
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)  # fixed margins instead of tight_layout, so the PNG is byte-stable
    fig.savefig(OUT, dpi=300, metadata={"Software": None})  # PNG at 300 dpi for print; no software tag, so runs again give the same bytes
    plt.close(fig)  # frees the figure
    print(f"  wrote {OUT.relative_to(PROJECT)}")  # one line for the run record


if __name__ == "__main__":  # if: run as a script, not imported
    main()  # draws the figure
