#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
The residual shift that each receiver's own reference position causes, and two check figures.

positions_residual_shift.csv  compares, row for row, the residual tables of the two runs of 05_own_receiver_positions.py (today's site
    positions, equal to the published tables byte for byte; each receiver's own position), which that script writes into
    Data/processed/robustness_checks/mirror/ (run it first to rebuild them). A position shift of d metres moves each satellite's
    geometric range by at most d and the estimated receiver clock, a weighted mean of those moves, by at most d, so no residual may
    move by more than 2d.
bootstrap_margin.png  the bootstrap margin of every Table 11 cell: the published 95 % interval of the agent's delay difference, the
    range of its two ends over 200 bootstrap seeds, and the share of seeds that keep the published verdict.
reference_positions_per_receiver.png  the per-receiver reference positions against the site estimates of Table 4, and the change of
    the known-antenna code residual per receiver and session against the receiver's offset, with the bound a position shift allows.
Usage   python3 codes/8_data_checks/07_position_residual_shift.py
Writes  Data/processed/robustness_checks/positions_residual_shift.csv, figures_checks.csv, Graph/bootstrap_margin.png,
        Graph/reference_positions_per_receiver.png
"""
# import: importlib loads the position script for its session-to-site table (its name starts with a digit)
import importlib.util
# import: os sets matplotlib's cache folder before matplotlib loads
import os
# import: sys registers the loaded module and reaches the helpers module
import sys
# import: Path for explicit paths
from pathlib import Path
# import: numpy for arrays
import numpy as np
# import: pandas for the tables
import pandas as pd

# project root (this file is codes/8_data_checks/07_position_residual_shift.py)
PROJECT = Path(__file__).resolve().parents[2]
# matplotlib's cache inside the project folder, set before matplotlib is imported
os.environ["MPLCONFIGDIR"] = str(PROJECT / "codes" / ".mplconfig")
# import: the plotting library, to choose the file-only backend
import matplotlib
# Agg: draws to files only, no window
matplotlib.use("Agg")
# import: pyplot for figures and axes
import matplotlib.pyplot as plt
# codes/ on the import path, so the helpers module resolves
sys.path.insert(0, str(PROJECT / "codes" / "common"))
# import: the project's byte-stable figure saver
from helpers import save_fig

# processed data, the position check's folder, the item-9 mirror and the figure folder
PROC, ROBUST_DIR, GRAPH = PROJECT / "Data" / "processed", PROJECT / "Data" / "processed" / "robustness_checks", PROJECT / "Graph"
MIRROR = ROBUST_DIR / "mirror"  # the scratch root of the own-position check
# dict: stream -> readable name
NAME = {"SBF_02": "mosaic-X5", "SBF_03": "mosaic-H", "UBX_01": "u-blox 01"}
# dict: stream -> colour (the receiver colours of the earlier RQ4 figures)
COL = {"SBF_02": "#0072B2", "SBF_03": "#D55E00", "UBX_01": "#009E73"}
# dict: site -> marker
MARK = {"2.6.2 site": "o", "south-west site": "s", "Stave": "^"}
# list: the receivers in the order the tables use
RXS = ["SBF_02", "SBF_03", "UBX_01"]
# 05c: the session-to-site table
_spec = importlib.util.spec_from_file_location("reference_positions_for_residual_shift", PROJECT / "codes" / "2_preprocessing" / "03_reference_positions.py")
C05 = importlib.util.module_from_spec(_spec)  # an empty module object
sys.modules["reference_positions_for_residual_shift"] = C05  # registered
_spec.loader.exec_module(C05)  # its top level (constants and functions)
# list of check rows
CHECKS = []


# function: records and prints one check
def check(name: str, ok: bool, detail: str = "") -> None:
    # appended
    CHECKS.append({"check": name, "pass": bool(ok), "detail": detail})
    # printed
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


# function: the residual-shift table: per variant, session and receiver, how far the per-receiver positions move the code residuals
def residual_shift() -> pd.DataFrame:
    # the per-receiver positions and today's site estimates
    rp = pd.read_csv(ROBUST_DIR / "reference_positions_per_receiver.csv")
    ref = pd.read_csv(PROC / "preprocessing" / "reference_positions.csv")  # the position script's table
    site = {r.site: np.array([r.x_m, r.y_m, r.z_m]) for r in ref[ref.scope == "site"].itertuples()}  # dict comprehension: site -> ECEF
    # dict comprehension: (site, receiver) -> the 3-D distance between the receiver's own position and the site estimate
    off = {(r.used_for_site, r.receiver): float(np.linalg.norm(np.array([r.x_m, r.y_m, r.z_m]) - site[r.used_for_site])) for r in rp[rp.used_for_site.isin(list(site))].itertuples()}
    # list of rows
    rows = []
    # for loop over the two residual variants
    for v in ("known", "self"):
        # the gate's table (= published) and the per-receiver one
        a = pd.read_csv(MIRROR / "gate" / "features" / f"residuals_{v}.csv.gz", usecols=["test_code", "receiver", "sec", "sv", "res_m"], dtype={"test_code": str})
        b = pd.read_csv(MIRROR / "per_receiver" / "features" / f"residuals_{v}.csv.gz", usecols=["test_code", "receiver", "sec", "sv", "res_m"], dtype={"test_code": str})  # the per-receiver table
        # the two tables have the same rows in the same order (05_own_receiver_positions.py checks it); asserted again here
        check(f"{v}: same keys row for row", a[["test_code", "receiver", "sec", "sv"]].equals(b[["test_code", "receiver", "sec", "sv"]]), f"{len(a)} rows")
        # the difference per row, and whether a residual appears or disappears
        d = a.assign(res_rx=b.res_m.to_numpy(), diff=b.res_m.to_numpy() - a.res_m.to_numpy())
        # for loop over the sessions and receivers
        for (t, rx), g in d.groupby(["test_code", "receiver"], sort=False):
            # the rows with a residual in both runs
            both = g[g.res_m.notna() & g.res_rx.notna()]
            ad = both["diff"].abs()  # absolute changes
            o = off[(C05.SITE_OF[t], rx)]  # this receiver's offset at the session's site
            # one row with counts, quantiles and the bound
            rows.append({"variant": v, "test_code": t, "receiver": rx, "site": C05.SITE_OF[t], "reference_offset_3d_m": o, "residuals_both": len(both),
                         "only_published": int((g.res_m.notna() & g.res_rx.isna()).sum()), "only_per_receiver": int((g.res_m.isna() & g.res_rx.notna()).sum()),  # continued: residuals present in one run only
                         "median_abs_change_m": float(ad.median()), "p99_abs_change_m": float(ad.quantile(0.99)), "max_abs_change_m": float(ad.max()), "within_2x_offset": bool(ad.max() <= 2 * o + 1e-4)})  # continued: the change and the bound (1e-4 m: the tables' four decimals)
    # the table
    T = pd.DataFrame(rows)
    # the bound must hold everywhere
    check("every residual change within twice the receiver's offset", bool(T.within_2x_offset.all()), f"largest ratio {float((T.max_abs_change_m / T.reference_offset_3d_m).max()):.3f}")
    # written with fixed line endings
    T.to_csv(ROBUST_DIR / "positions_residual_shift.csv", index=False, lineterminator="\n")
    # the table
    return T


# function: the figure, the bootstrap margin per Table 11 cell
def plot_bootstrap_margin() -> None:
    # the margin table
    b = pd.read_csv(ROBUST_DIR / "bootstrap_margin.csv")
    # one row per cell, top to bottom in the table's order
    fig, ax = plt.subplots(figsize=(9.2, 7.2))
    y = np.arange(len(b))[::-1]  # the rows' heights, first cell on top
    xmax = 30.0  # the axis is cut here; the one interval beyond carries its end as text
    # dict: verdict -> colour
    vc = {"worse": "#D55E00", "no difference detectable": "#56B4E9", "better": "#0072B2"}
    # for loop over the cells
    for yi, r in zip(y, b.itertuples()):
        # the label at the left
        lab = f"{NAME[r.receiver]}, {r.share:g}, {r.arm}"
        # if: the cell has an interval
        if pd.notna(r.published_ci_low):
            c = vc.get(r.published_verdict, "#000000")  # the verdict's colour
            ax.plot([r.published_ci_low, min(r.published_ci_high, xmax)], [yi, yi], color=c, lw=2.2, solid_capstyle="butt")  # the published 95 % interval
            ax.plot([r.lo_min, r.lo_max], [yi + 0.22, yi + 0.22], color="#000000", lw=3, alpha=0.7, solid_capstyle="butt")  # the range of the lower end over 200 seeds
            ax.plot([min(r.hi_min, xmax), min(r.hi_max, xmax)], [yi + 0.22, yi + 0.22], color="#000000", lw=3, alpha=0.7, solid_capstyle="butt")  # the range of the upper end
            if r.published_ci_high > xmax:  # if: the interval runs past the cut
                ax.annotate(f"→ {r.published_ci_high:.0f} s", (xmax, yi), xytext=(-2, -9), textcoords="offset points", ha="right", fontsize=7)  # its end as text
            txt = f"{int(r.seeds_same_verdict)}/{int(r.seeds)}"  # the seeds that keep the verdict
        # else: no interval (no point, or no paired run)
        else:
            txt = "no point" if r.published_verdict.startswith("no point") else f"{int(r.seeds_same_verdict)}/{int(r.seeds)}, no interval"  # conditional expression: why there is none
        ax.text(xmax + 1.0, yi, txt, va="center", fontsize=8, fontweight="bold" if txt.startswith("3/") else "normal")  # the share at the right, bold where it is fragile
        ax.text(-14.5, yi, lab, va="center", ha="right", fontsize=8)  # the cell's name at the left
    ax.axvline(0, color="#000000", lw=0.8)  # zero: no difference
    ax.set_xlim(-14, xmax)  # the cut
    ax.set_ylim(-0.8, len(b) - 0.2)  # all rows
    ax.set_yticks([])  # names are drawn as text
    ax.set_xlabel("agent minus best baseline, interquartile-mean delay difference (s); positive = the agent is slower", fontsize=9)  # the axis
    ax.grid(axis="x", alpha=0.3, lw=0.5)  # light grid
    # the legend from proxy lines
    hs = [plt.Line2D([], [], color=vc["worse"], lw=2.2), plt.Line2D([], [], color=vc["no difference detectable"], lw=2.2), plt.Line2D([], [], color="#000000", lw=3, alpha=0.7)]
    ax.legend(hs, ["published interval, verdict 'worse'", "published interval, 'no difference detectable'", "range of each end over 200 seeds"], fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, frameon=False)  # the legend, below the axis
    ax.set_title("seeds keeping the published verdict at the right", fontsize=9, loc="right")  # the right column's meaning
    fig.tight_layout()  # spacing
    save_fig(fig, GRAPH / "bootstrap_margin.png")  # byte-stable save (it closes the figure)


# function: the figure, the per-receiver positions and the residual change they cause
def plot_reference_positions_per_receiver(T: pd.DataFrame) -> None:
    # the per-receiver positions
    rp = pd.read_csv(ROBUST_DIR / "reference_positions_per_receiver.csv")
    # two panels side by side
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.8))
    # panel (a): horizontal offsets from the site estimates, the positions a product uses
    used = rp[rp.used_for_site.isin(list(MARK))]  # boolean filter: the used positions
    for r in used.itertuples():  # for loop over them
        a1.scatter(r.dE_m, r.dN_m, marker=MARK[r.used_for_site], s=52, color=COL[r.receiver], zorder=3)  # one point
    a1.axhline(0, color="#000000", lw=0.6)  # the site estimate's north line
    a1.axvline(0, color="#000000", lw=0.6)  # and its east line
    lim = 1.5  # half-width of the panel (m), wider than every used offset
    a1.set_xlim(-lim, lim)  # east axis
    a1.set_ylim(-lim, lim)  # north axis
    a1.set_aspect("equal")  # metres equal on both axes
    # the unused 2.6.5 Stave stop lies off the panel: an arrow toward its mean offset, with its range in the label
    st = rp[~rp.used_for_site.isin(list(MARK))]
    ang = np.arctan2(st.dN_m.mean(), st.dE_m.mean())  # its mean direction from the origin
    tip = 0.92 * lim * np.array([np.cos(ang), np.sin(ang)]) / max(abs(np.cos(ang)), abs(np.sin(ang)))  # the arrow's tip near the panel edge in that direction
    a1.annotate(f"2.6.5 Stave stop (not used):\n{st.dH_m.min():.1f} to {st.dH_m.max():.1f} m this way", xy=tuple(tip), xytext=tuple(0.35 * tip), fontsize=7, ha="center", arrowprops={"arrowstyle": "->", "lw": 0.8})  # the arrow and its label
    a1.set_xlabel("east of the site estimate (m)", fontsize=9)  # axis
    a1.set_ylabel("north of the site estimate (m)", fontsize=9)  # axis
    a1.set_title("(a) each receiver's own position against Table 4", fontsize=9)  # the panel's title
    a1.grid(alpha=0.3, lw=0.5)  # light grid
    # legend: receivers by colour, sites by marker
    hs = [plt.Line2D([], [], color=COL[rx], marker="o", ls="", label=NAME[rx]) for rx in RXS] + [plt.Line2D([], [], color="#555555", marker=m, ls="", label=s) for s, m in MARK.items()]  # list concatenation: proxy markers
    a1.legend(handles=hs, fontsize=7, loc="upper left", frameon=True, framealpha=0.9)  # the legend, boxed over the grid
    # panel (b): largest residual change against the offset, known-antenna variant
    k = T[T.variant == "known"]
    for r in k.itertuples():  # for loop over sessions and receivers
        a2.scatter(r.reference_offset_3d_m, r.max_abs_change_m, marker=MARK[r.site], s=46, color=COL[r.receiver], zorder=3)  # the largest change
    xm = float(k.reference_offset_3d_m.max()) * 1.15  # the axis end
    a2.plot([0, xm], [0, 2 * xm], color="#000000", lw=0.8, ls="--", label="bound: twice the offset")  # the bound a position shift allows
    a2.plot([0, xm], [0, xm], color="#000000", lw=0.6, ls=":", label="equal to the offset")  # the one-to-one line
    a2.set_xlim(0, xm)  # from zero
    a2.set_ylim(0, 2 * xm)  # up to the bound's end
    a2.set_xlabel("3-D distance of the receiver's own position from the site estimate (m)", fontsize=9)  # axis
    a2.set_ylabel("largest change of a known-antenna code residual (m)", fontsize=9)  # axis
    a2.set_title("(b) what the shift does to the residuals", fontsize=9)  # the panel's title
    a2.grid(alpha=0.3, lw=0.5)  # light grid
    a2.legend(fontsize=7, loc="upper left", frameon=False)  # the legend
    fig.tight_layout()  # spacing
    save_fig(fig, GRAPH / "reference_positions_per_receiver.png")  # byte-stable save


# function: main
def main() -> None:
    # the table behind the figure
    T = residual_shift()
    print(T.to_string(index=False))  # printed
    plot_bootstrap_margin()  # the figure
    plot_reference_positions_per_receiver(T)  # the figure
    # the checks written
    pd.DataFrame(CHECKS).to_csv(ROBUST_DIR / "figures_checks.csv", index=False, lineterminator="\n")
    # status 1 on a failed check
    sys.exit(0 if all(c["pass"] for c in CHECKS) else 1)


# if: run as a script
if __name__ == "__main__":
    # runs both figures
    main()
