#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
Figures 3 to 5, drawn from the data products in UTC with plain styling legible in print.

Figure 3  scheduled activity at the three test areas against the capture windows of the recorded tests. Test Area 1 and 2 entries
          are those the transmission plan schedules during a capture (ground_truth/concurrent_transmissions.csv), with their stated power; the
          Test Area 3 bars are the scheduled windows of the six recorded tests and the dark bars the capture windows
          (ground_truth/session_ground_truth.csv). 2.6.12 is drawn only as its scheduled window, labelled as the held-out session, and its
          capture window is left out.
Vehicle speed (not in the thesis)  test 2.6.5: the vehicle's OBD-II speed and the mosaic-X5's horizontal speed against UTC, with the stationary segments
          of preprocessing/vehicle_motion.csv shaded. The two series are read by the functions of codes/2_preprocessing/02_vehicle_motion.py from the raw files
          (read only), the X5 only up to its front-end gain onset (ground_truth/ta3_schedule_confirmation.csv), as the motion script uses it.
Both are context figures: the OBD-II speed and the receiver's own velocity are verification quantities under the exclusion rule.

Figure 4  test 2.6.5: final labels per second (pooled and per receiver), the L1 gains and the own-position distances (the label script's inputs).
Windows (not in the thesis)  a schematic of the windows (no data).
Figure 5  a schematic of the sequential detection problem of Section 4.10 (no data): step types and what each action costs, the two
          state arms, and decision mode against stopping mode with the refractory mark; every number is read from codes/6_reinforcement_learning/01_environment.py.

Usage   /opt/anaconda3/bin/python3 codes/figures/figures_3_to_5.py [rl]      (rl: only the Section 4.10 schematic)
Writes  Graph/figure_3.png, Graph/figure_vehicle_speed_265.png, Graph/figure_4.png, Graph/figure_windows.png,
        Graph/figure_5.png
"""

from __future__ import annotations  # future import: type hints kept as text, so they cost nothing at run time

import importlib.util  # standard library: loads codes/2_preprocessing/02_vehicle_motion.py, whose file name starts with a digit
import os  # standard library: sets matplotlib's cache folder before matplotlib is imported
import sys  # standard library: registers the loaded module and the import path
from pathlib import Path  # Path objects for explicit file paths

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
os.environ["MPLCONFIGDIR"] = str(PROJECT / "codes" / ".mplconfig")  # matplotlib's cache inside the project folder

import numpy as np  # NumPy: arrays of times and speeds
import pandas as pd  # pandas: reads the CSV products
import matplotlib  # the plotting library; imported first so its backend can be set before pyplot loads
matplotlib.use("Agg")  # file-only backend: no window is opened, and the PNG bytes do not depend on a screen
import matplotlib.dates as mdates  # date helpers: tick positions and HH:MM labels on a time axis
import matplotlib.pyplot as plt  # pyplot: the plotting interface
from matplotlib.patches import Patch  # legend entries drawn as filled boxes

PROC = PROJECT / "Data" / "processed"  # the data products (read only)
GRAPH = PROJECT / "Graph"  # where the figures are written
OUT2 = GRAPH / "figure_3.png"  # Figure 3 file
OUT3 = GRAPH / "figure_vehicle_speed_265.png"  # the vehicle-speed figure's file (not in the thesis)
OUT4 = GRAPH / "figure_4.png"  # Figure 4 file
OUT5 = GRAPH / "figure_windows.png"  # the window schematic's file (not in the thesis)
OUT_RL = GRAPH / "figure_5.png"  # the schematic of Section 4.10
LAB_COLOR = {"clean": "#009E73", "attack_jam_spoof": "#7A0019", "attack_spoof": "#E69F00", "uncertain": "#C8C8C8", "excluded": "#404040"}  # dict: label -> colour (the label script's palette, readable in grey)
LAB_NAME = {"clean": "clean", "attack_jam_spoof": "attack (jamming and spoofing)", "attack_spoof": "attack (spoofing)", "uncertain": "uncertain", "excluded": "excluded"}  # dict: label -> thesis name
RX_NAME = {"SBF_02": "mosaic-X5", "SBF_03": "mosaic-H", "UBX_01": "u-blox 01"}  # dict: stream -> receiver name
RX_COLOR = {"SBF_02": "#0072B2", "SBF_03": "#E69F00", "UBX_01": "#009E73"}  # dict: stream -> line colour
HELD_OUT = "2.6.12"  # the session held out as the unseen attack type: drawn only as its scheduled window
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})  # dict: one font and size for every label


def utc(text: str) -> pd.Timestamp:  # function: "2026-09-14 12:30:00" -> a timezone-free timestamp read as UTC
    return pd.Timestamp(text)  # the products store UTC text without an offset, so no conversion is needed


def figure2() -> None:  # function: draws Figure 2
    gt = pd.read_csv(PROC / "ground_truth" / "session_ground_truth.csv")  # one row per recorded test: scheduled and capture windows
    cc = pd.read_csv(PROC / "ground_truth" / "concurrent_transmissions.csv")  # entries at Test Areas 1 and 2 scheduled during a capture
    cc = cc.drop_duplicates("concurrent_test_code")  # one bar per entry, although some overlap two captures
    lanes = {"Test Area 1": 3, "Test Area 2": 2, "Test Area 3\n(scheduled)": 1, "Captured": 0}  # dict: lane name -> y position
    fig, ax = plt.subplots(figsize=(6.3, 2.9))  # one axes, 6.3 inches wide for the 15 cm text width
    cc = cc.sort_values("concurrent_start_utc")  # entries in time order, so labels can alternate between neighbours
    ta2_i = 0  # counter of Test Area 2 entries, for alternating their power labels above and below the bar
    for _, r in cc.iterrows():  # for loop over the concurrent entries (iterrows yields index and row)
        y = lanes["Test Area 1"] if r.concurrent_test_area == 1 else lanes["Test Area 2"]  # conditional expression: the entry's lane
        t0, t1 = utc(r.concurrent_start_utc), utc(r.concurrent_end_utc)  # tuple assignment: start and end of the entry
        stated = not pd.isna(r.power_w)  # True when the plan or the catalogue states a power
        high = stated and r.power_w > 1.0  # True for an entry above one watt
        face = "#4d4d4d" if high else ("#cfcfcf" if stated else "white")  # dark above 1 W, light grey at or below, white if unstated
        ax.barh(y, mdates.date2num(t1) - mdates.date2num(t0), left=mdates.date2num(t0), height=0.55, color=face, edgecolor="black", lw=0.6, hatch=None if stated else "////")  # one horizontal bar
        if t1 - t0 >= pd.Timedelta(minutes=20):  # if: the bar is wide enough to hold its code at a printable size
            ax.text(mdates.date2num(t0 + (t1 - t0) / 2), y, r.concurrent_test_code, ha="center", va="center", fontsize=7.5, color="white" if high else "black")  # the entry's code on the bar
        ta2_i += int(r.concurrent_test_area == 2)  # counts this entry if it is at Test Area 2 (int(True) is 1)
        if stated:  # if: a power is stated, print it beside the bar
            below = r.concurrent_test_area == 2 and ta2_i % 2 == 1  # True for every second Test Area 2 entry: its label goes under the bar
            ax.text(mdates.date2num(t0 + (t1 - t0) / 2), y - 0.33 if below else y + 0.33, f"{r.power_w:.2g} W", ha="center", va="top" if below else "bottom", fontsize=7.5)  # the power to two significant figures
    for _, r in gt.iterrows():  # for loop over the six recorded tests
        s0, s1 = utc(r.scheduled_start_utc), utc(r.scheduled_end_utc)  # scheduled window of the test
        held = r.test_code == HELD_OUT  # True for the held-out session
        ax.barh(lanes["Test Area 3\n(scheduled)"], mdates.date2num(s1) - mdates.date2num(s0), left=mdates.date2num(s0), height=0.55, color="white", edgecolor="black", lw=0.6, hatch="...." if held else None)  # scheduled window
        if held:  # if: the held-out session is labelled beside its short bar, which is too narrow for the words
            ax.text(mdates.date2num(s1) + 0.002, lanes["Test Area 3\n(scheduled)"], f"{r.test_code}\n(held out)", ha="left", va="center", fontsize=7.5)  # label to the right of the bar
        else:  # else: every other test is labelled on its bar
            ax.text(mdates.date2num(s0 + (s1 - s0) / 2), lanes["Test Area 3\n(scheduled)"], r.test_code, ha="center", va="center", fontsize=7.5)  # label of the scheduled bar
        if held:  # if: the held-out session gets no capture bar
            continue  # skips the rest of this loop pass
        c0, c1 = utc(r.capture_start_utc), utc(r.capture_end_utc)  # capture window of the test
        ax.barh(lanes["Captured"], mdates.date2num(c1) - mdates.date2num(c0), left=mdates.date2num(c0), height=0.55, color="black", edgecolor="black", lw=0.6)  # capture window
        ax.text(mdates.date2num(c0 + (c1 - c0) / 2), lanes["Captured"] - 0.45, r.test_code, ha="center", va="top", fontsize=7.5)  # its code below the bar
    ax.set_yticks(list(lanes.values()))  # one tick per lane
    ax.set_yticklabels(list(lanes.keys()))  # lane names as tick labels
    ax.set_ylim(-0.95, 3.75)  # room for the codes below the capture bars and the powers above Test Area 1
    ax.set_xlim(mdates.date2num(utc("2026-09-14 12:15:00")), mdates.date2num(utc("2026-09-14 16:40:00")))  # the afternoon of the six captures
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30]))  # a tick every half hour
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))  # HH:MM labels
    ax.set_xlabel("UTC, 14 September 2026")  # axis title: the time base
    ax.grid(axis="x", color="#d9d9d9", lw=0.5)  # light vertical grid at the ticks
    ax.set_axisbelow(True)  # grid drawn behind the bars
    # list of legend patches: one filled box per bar shade, so the reader can tell the power classes apart
    handles = [Patch(fc="#4d4d4d", ec="black", label="above 1 W"), Patch(fc="#cfcfcf", ec="black", label="1 W or less"),
               Patch(fc="white", ec="black", hatch="////", label="power not stated"), Patch(fc="black", ec="black", label="capture window")]  # legend boxes
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4, frameon=False, fontsize=7.5)  # legend under the axis
    fig.subplots_adjust(left=0.14, right=0.97, top=0.97, bottom=0.27)  # fixed margins, so the PNG is byte-stable
    fig.savefig(OUT2, dpi=300, metadata={"Software": None})  # PNG for print; no software tag, so runs again give the same bytes
    plt.close(fig)  # frees the figure
    print(f"  wrote {OUT2.relative_to(PROJECT)}")  # one line for the run log


def load_05a():  # function: loads codes/2_preprocessing/02_vehicle_motion.py as a module without running its main()
    spec = importlib.util.spec_from_file_location("vehicle_motion_for_figures", PROJECT / "codes" / "2_preprocessing" / "02_vehicle_motion.py")  # where to load from
    mod = importlib.util.module_from_spec(spec)  # an empty module object from that spec
    sys.modules["vehicle_motion_for_figures"] = mod  # registers it, as a normal import would
    spec.loader.exec_module(mod)  # runs the file's top-level code (main() is guarded and not run)
    return mod  # the loaded module


def figure3() -> None:  # function: draws Figure 3
    m = load_05a()  # the motion script's readers and its X5 cut-off (ONSET, read there from ground_truth/ta3_schedule_confirmation.csv)
    folder = next(f for code, f in m.sessions() if code == "2.6.5")  # generator with next(): the raw folder of 2.6.5
    obd = m.load_speed(folder)  # every SPEED row: UTC nanoseconds and km/h (invalid rows are NaN)
    x5 = m.read_x5(folder)  # the X5 PVT rows to 12:52:00 on receiver time (UTC)
    x5 = x5[(x5.t_ns < m.ONSET.value) & x5.vel_ok]  # boolean mask: only before the X5 gain onset, and only usable velocities
    seg = pd.read_csv(PROC / "preprocessing" / "vehicle_motion.csv")  # the OBD segments of every session
    seg = seg[seg.test_code == "2.6.5"]  # boolean mask: the four segments of 2.6.5
    to_num = lambda ns: mdates.date2num(pd.to_datetime(ns, unit="ns"))  # lambda: integer nanoseconds -> matplotlib date numbers
    fig, ax = plt.subplots(figsize=(6.3, 2.6))  # one axes, 6.3 inches wide
    for _, r in seg[seg.state == "stationary"].iterrows():  # for loop over the stationary segments
        ax.axvspan(mdates.date2num(pd.Timestamp(r.start_utc)), mdates.date2num(pd.Timestamp(r.end_utc)), color="#e6e6e6", lw=0)  # grey band: the vehicle stood still
    ax.plot(to_num(obd.t_ns.to_numpy()), obd.kmh.to_numpy(), color="black", lw=0.9, label="OBD-II speed")  # the vehicle's own speed, whole km/h
    ax.plot(to_num(x5.t_ns.to_numpy()), x5.hspeed_ms.to_numpy() * 3.6, color="#7f7f7f", lw=0.8, ls=(0, (3, 1.5)), label="mosaic-X5 horizontal speed")  # receiver speed, m/s times 3.6 = km/h
    ax.set_ylabel("Speed (km/h)")  # axis title with unit
    ax.set_xlabel("UTC, 14 September 2026")  # axis title: the time base
    t_start = pd.Timestamp(seg.start_utc.iloc[0]).floor("min")  # the axis starts at the minute of the first segment
    t_end = pd.Timestamp(seg.end_utc.iloc[-1]).ceil("min")  # and ends at the minute after the last
    ax.set_xlim(mdates.date2num(t_start), mdates.date2num(t_end))  # the whole capture
    ax.set_ylim(-2, np.nanmax(obd.kmh.to_numpy()) * 1.15)  # from just below zero to 15 per cent above the highest speed
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 5)))  # a tick every five minutes
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))  # HH:MM labels
    handles, labels = ax.get_legend_handles_labels()  # the two line entries
    handles.append(Patch(fc="#e6e6e6", ec="#9e9e9e", lw=0.5))  # plus a grey box for the stationary segments
    labels.append("stationary (OBD-II)")  # its label
    ax.legend(handles, labels, loc="upper right", frameon=False, fontsize=7.5)  # legend inside the empty upper right
    fig.subplots_adjust(left=0.09, right=0.965, top=0.97, bottom=0.17)  # fixed margins, so the PNG is byte-stable
    fig.savefig(OUT3, dpi=300, metadata={"Software": None})  # PNG for print; no software tag
    plt.close(fig)  # frees the figure
    print(f"  wrote {OUT3.relative_to(PROJECT)}")  # one line for the run log


def load_05d():  # function: loads codes/1_ground_truth/04_gain_and_position_evidence.py (the receivers' own positions and the site geometry) without running its main()
    sys.path.insert(0, str(PROJECT / "codes" / "common"))  # list method insert: codes/ first on the import path, where the evidence script finds the helpers module
    spec = importlib.util.spec_from_file_location("gain_evidence_for_figures", PROJECT / "codes" / "1_ground_truth" / "04_gain_and_position_evidence.py")  # where to load from
    mod = importlib.util.module_from_spec(spec)  # empty module object
    sys.modules["gain_evidence_for_figures"] = mod  # registered as a normal import would be
    spec.loader.exec_module(mod)  # runs the top-level code (main() is guarded)
    return mod  # the module


def figure4() -> None:  # function: draws Figure 4, the labels of 2.6.5 with the gains and the own-position distances (the label script's the figure inputs)
    code = "2.6.5"  # the session shown
    lab = pd.read_csv(PROC / "labels" / "labels_1hz_final.csv", usecols=["test_code", "receiver", "sec", "label", "label_per_receiver_only"])  # the final labels
    lab = lab[lab.test_code == code]  # boolean mask: 2.6.5 only
    gt = pd.read_csv(PROC / "ground_truth" / "session_ground_truth.csv").set_index("test_code")  # windows of the session
    st = pd.read_csv(PROC / "ground_truth" / "receiver_status_timeseries.csv", usecols=["test_code", "receiver", "timestamp_utc", "band", "antenna_id", "agc_gain_db", "agc_cnt"], low_memory=False)  # the decoded status
    st = st[st.test_code == code]  # boolean mask: 2.6.5 only
    d5 = load_05d()  # the evidence script's own-position reader and site geometry
    site = d5.site_xyz(d5.M.SITE_OF[code])  # the site the session is referred to
    tnum = lambda sec: mdates.date2num(pd.to_datetime(sec, unit="s"))  # lambda: unix seconds -> matplotlib date numbers
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6.3, 5.6), sharex=True, gridspec_kw={"height_ratios": [1.6, 1.3, 1.0]})  # three stacked panels on one time axis
    rows = [(rx, "label", RX_NAME[rx]) for rx in RX_NAME] + [(rx, "label_per_receiver_only", f"{RX_NAME[rx]}, own evidence") for rx in RX_NAME]  # list: (receiver, label column, row name), pooled rows first
    for i, (rx, col, name) in enumerate(rows):  # for loop over the label rows
        L = lab[lab.receiver == rx].sort_values("sec")  # this receiver's seconds in time order
        y = len(rows) - 1 - i  # the row's height on the axis, first row on top
        for lname, colour in LAB_COLOR.items():  # for loop over the label names
            secs = L.sec[L[col] == lname].to_numpy()  # seconds with that label
            ax1.bar(tnum(secs), 0.8, width=1 / 86400, bottom=y - 0.4, color=colour, linewidth=0, align="edge")  # one bar per second (a second is 1/86400 of a day on the date axis)
    ax1.set_yticks(range(len(rows)))  # one tick per row
    ax1.set_yticklabels([r[2] for r in rows][::-1], fontsize=7.5)  # row names, bottom up
    ax1.legend(handles=[Patch(fc=c, label=LAB_NAME[k]) for k, c in LAB_COLOR.items()], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=7.5)  # legend above the panel (three columns, so the 7.5-pt labels fit the width)
    for rx, status in (("SBF_02", "Septentrio_02"), ("SBF_03", "Septentrio_03")):  # for loop over the Septentrio receivers
        g = st[(st.receiver == status) & (st.band == "GPSL1/E1") & (st.antenna_id == 0)].sort_values("timestamp_utc")  # L1/E1 gain on the main antenna
        ax2.plot(mdates.date2num(pd.to_datetime(g.timestamp_utc)), g.agc_gain_db, lw=0.8, color=RX_COLOR[rx], label=f"{RX_NAME[rx]} L1 gain (dB)")  # its trace
    ax2b = ax2.twinx()  # a second vertical axis for the u-blox gain count
    u = st[(st.receiver == "u-blox_01") & (st.band == "L1")].sort_values("timestamp_utc")  # u-blox 01 L1 gain count (host time)
    ax2b.plot(mdates.date2num(pd.to_datetime(u.timestamp_utc)), u.agc_cnt, lw=0.8, color=RX_COLOR["UBX_01"], ls=(0, (3, 1.5)), label="u-blox 01 L1 gain count")  # its trace, dashed
    ax2.set_ylabel("gain (dB)", fontsize=7.5)  # left axis title
    ax2b.set_ylabel("gain count", fontsize=7.5)  # right axis title
    h1, l1 = ax2.get_legend_handles_labels()  # left legend entries
    h2, l2 = ax2b.get_legend_handles_labels()  # right legend entries
    ax2.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="lower left", frameon=False)  # one legend for both axes
    for rx in ("SBF_02", "UBX_01"):  # for loop over the receivers that report a position
        own = d5.own_positions(gt, code, rx)  # first valid own fix of each second (u-blox on host time)
        a0, b0 = d5.capture_spans(gt, code, rx)[0]  # the usable span (from 12:34:57, after the drives)
        own = own[(own.sec >= a0) & (own.sec < b0)]  # boolean mask: inside it
        dh = d5.horiz_from_site(own.lat.to_numpy(), own.lon.to_numpy(), own.h.to_numpy(), site)  # horizontal distance from the site, m
        ax3.plot(tnum(own.sec.to_numpy()), dh, ".", ms=1.2, color=RX_COLOR[rx], label=f"{RX_NAME[rx]} own position")  # one dot per fix
    sys.path.insert(0, str(PROJECT / "codes" / "figures"))  # list method insert: codes/figures on the import path, for the constant reader
    from source_constants import module_const  # reads a constant from a script without running it
    ax3.axhline(module_const("codes/3_labels/01_labels.py", "POS_M"), color="black", lw=0.6, ls=(0, (4, 3)))  # the departure distance of the labels (the label script's POS_M, 50 m), not typed here
    ax3.set_yscale("log")  # logarithmic distance axis
    ax3.set_ylim(0.1, 2e3)  # 0.1 m to 2 km
    ax3.set_ylabel("distance from site (m)", fontsize=7.5)  # axis title
    ax3.legend(fontsize=7.5, loc="upper left", frameon=False, markerscale=6)  # legend with larger dots
    s0, s1 = pd.Timestamp(gt.loc[code, "scheduled_start_utc"]), pd.Timestamp(gt.loc[code, "scheduled_end_utc"])  # the scheduled window
    for ax in (ax1, ax2, ax3):  # for loop over the panels
        ax.axvspan(mdates.date2num(s0), mdates.date2num(s1), color="black", alpha=0.06, lw=0)  # the scheduled window, shaded
        ax.tick_params(labelsize=7)  # tick label size
    ax3.set_xlim(mdates.date2num(pd.Timestamp(gt.loc[code, "capture_start_utc"])), mdates.date2num(pd.Timestamp(gt.loc[code, "capture_end_utc"])))  # the capture
    ax3.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 5)))  # a tick every five minutes
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))  # HH:MM labels
    ax3.set_xlabel("UTC, 14 September 2026 (u-blox on host time)", fontsize=7.5)  # axis title
    ax2b.tick_params(labelsize=7)  # right-axis tick label size, as the other panels
    fig.subplots_adjust(left=0.26, right=0.89, top=0.93, bottom=0.08, hspace=0.12)  # fixed margins (room for the row names), so the PNG is byte-stable
    fig.savefig(OUT4, dpi=300, metadata={"Software": None})  # PNG for print; no software tag
    plt.close(fig)  # frees the figure
    print(f"  wrote {OUT4.relative_to(PROJECT)}")  # one line for the run log


def figure5() -> None:  # function: draws Figure 5, a schematic of the windows (no data: it states no result)
    sys.path.insert(0, str(PROJECT / "codes" / "figures"))  # list method insert: codes/figures on the import path, for the constant reader
    from source_constants import module_const  # reads constants from the scripts without running them
    w_long = module_const("codes/5_baselines/01_logistic_regression.py", "W")  # the primary window length (the regression script's W, 10 s)
    w_short = sorted(module_const("codes/4_features/03_window_features.py", "WINDOWS"))[-2]  # the next shorter window length of 07 (5 s)
    labels = ["clean"] * 6 + ["uncertain"] * 5 + ["attack_spoof"] * 7 + ["excluded"] + ["attack_spoof"] * 3  # list: an illustrative sequence of second labels
    fig, ax = plt.subplots(figsize=(6.3, 2.6))  # one axes
    for i, lname in enumerate(labels):  # for loop over the illustrative seconds
        ax.add_patch(plt.Rectangle((i, 0), 0.95, 0.6, color=LAB_COLOR[lname], lw=0))  # one second as a coloured cell
    t = 15  # the decision second of the example
    ax.text(t + 0.47, -0.25, "t", ha="center", va="center", fontsize=8, style="italic")  # its name under the cell
    spans = [(w_long, t, 1.0, f"{w_long:g} s window ending at t"), (w_short, t, 1.55, f"{w_short:g} s window ending at t"), (w_long, t + 1, 2.1, "the next window ends at t + 1")]  # list of (length, last second, height, text)
    for w, last, y, text in spans:  # for loop over the example windows
        a, b = last - w + 1, last + 1  # first second and the end of the last one
        ax.annotate("", xy=(a, y), xytext=(b - 0.05, y), arrowprops={"arrowstyle": "|-|", "lw": 0.8, "mutation_scale": 3})  # a bracket over the window
        ax.text(24, y, text, va="center", fontsize=7.5)  # its description, right of the cells
        ax.plot([last + 0.47, last + 0.47], [0.62, y - 0.08], color="black", lw=0.5, ls=(0, (2, 2)))  # a dotted line from the window to the second that labels it
    ax.annotate("", xy=(18.47, 0.62), xytext=(20.5, 0.75), arrowprops={"arrowstyle": "->", "lw": 0.7})  # pointer to the excluded second
    ax.text(20.6, 0.62, "no window may hold\nan excluded second", fontsize=7.5, va="bottom")  # its note, right of the cells
    ax.annotate("", xy=(9.47, -0.02), xytext=(9.47, -0.45), arrowprops={"arrowstyle": "->", "lw": 0.7})  # pointer to an uncertain second
    ax.text(0, -0.62, "a window ending on an uncertain second is kept but not scored", fontsize=7.5, va="center")  # its note, under the cells
    ax.set_xlim(-0.5, 33)  # room for the texts on the right
    ax.set_ylim(-0.8, 2.75)  # room above and below
    ax.axis("off")  # no axes: a schematic
    handles = [Patch(fc=LAB_COLOR[k], label=LAB_NAME[k]) for k in ("clean", "uncertain", "attack_spoof", "excluded")]  # legend boxes for the labels shown
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.03), ncol=4, frameon=False, fontsize=7.5)  # legend on top
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.02)  # fixed margins, so the PNG is byte-stable
    fig.savefig(OUT5, dpi=300, metadata={"Software": None})  # PNG for print; no software tag
    plt.close(fig)  # frees the figure
    print(f"  wrote {OUT5.relative_to(PROJECT)}")  # one line for the run log


def figure_rl() -> None:  # function: draws the schematic of the sequential detection problem of Section 4.10 (no data: it states no result)
    sys.path.insert(0, str(PROJECT / "codes" / "figures"))  # list method insert: codes/figures on the import path, for the constant readers
    from source_constants import module_const, source_regex  # reads constants and code literals without running the scripts
    c11 = "codes/6_reinforcement_learning/01_environment.py"  # the environment script whose constants the drawing names
    refr = module_const(c11, "REFRACTORY")  # the 30 s refractory period of stopping mode
    pre_max, pre_min = module_const(c11, "PRE_MAX"), module_const(c11, "PRE_MIN")  # an onset episode starts 300 to 10 s before the onset
    lags = module_const(c11, "LAGS")  # the primary state's windows end at t, t - 10 and t - 20 s
    n_z = module_const(c11, "N_Z")  # the surrogate state's 30 log-odds
    n_sel = module_const("codes/4_features/05_jamming_screen.py", "MI_K")  # the 36 selected columns per fold
    cost1 = source_regex(c11, r"reward = -(\d)\.0  # one second of delay")  # a silent post-change second costs 1
    fig, ax = plt.subplots(figsize=(6.3, 3.9))  # one axes for the whole schematic
    spans = [(0, 10, "clean", "pre-change"), (10, 18, "uncertain", "uncertain"), (18, 34, "attack_spoof", "post-change (attack)")]  # list of (first cell, end, colour key, name) of the example stretch
    for a, b, key, name in spans:  # for loop over the three stretches
        for i in range(a, b):  # for loop over their seconds
            ax.add_patch(plt.Rectangle((i, 3.0), 0.92, 0.45, color=LAB_COLOR[key], lw=0))  # one second as a coloured cell
        ax.text((a + b) / 2, 3.22, name, ha="center", va="center", fontsize=7.5, color="white" if key == "clean" else "black")  # the stretch's name on its cells (white on green, black on grey and orange for contrast)
    onset, end = 18, 34  # the onset and the end of the attack interval in the example (cell positions, not to scale)
    ax.annotate("onset", xy=(onset, 3.0), xytext=(onset, 2.72), ha="center", fontsize=7.5, arrowprops={"arrowstyle": "->", "lw": 0.7})  # the onset marked below the cells
    ax.plot([end, end], [2.95, 3.5], color="black", lw=0.8)  # the end of the attack interval
    notes = [(5, "alarm: −λ,\nepisode ends"), (14, "alarm: recorded,\nno reward, continues"), (24, f"silent: −{cost1} per second\nalarm: no cost, episode ends"), (34, "silent to the end:\n−m")]  # list of (centre, text): what each action costs where (the first note on two lines, so it clears its neighbour at 7.5 pt)
    for x, text in notes:  # for loop over the cost notes
        ax.text(x, 3.62, text, ha="center", va="bottom", fontsize=7.5)  # the note above the cells
    ax.annotate("", xy=(3, 2.82), xytext=(17, 2.82), arrowprops={"arrowstyle": "|-|", "lw": 0.7, "mutation_scale": 3})  # bracket: where onset episodes start
    ax.text(10, 2.62, f"onset episodes start {pre_max} to {pre_min} s before the onset", ha="center", va="center", fontsize=7.5)  # its label
    ax.text(-0.5, 2.15, "decision mode", ha="right", va="center", fontsize=7.5)  # row name: decision mode
    for x in (13, 14, 15, 19, 20, 21, 22):  # for loop over the example's alarms, second by second
        ax.plot([x + 0.46, x + 0.46], [2.0, 2.3], color="black", lw=1.0)  # one alarm mark per second in decision mode
    ax.text(-0.5, 1.55, "stopping mode", ha="right", va="center", fontsize=7.5)  # row name: stopping mode
    ax.add_patch(plt.Rectangle((13.92, 1.42), 6.0, 0.26, color="#DDDDDD", lw=0))  # the silent refractory period after the first event (not to scale)
    ax.plot([13.46, 13.46], [1.38, 1.72], color="black", lw=1.6)  # the first alarm event, before the onset
    ax.plot([20.38, 20.38], [1.38, 1.72], color="black", lw=1.6)  # the next event, the first one the stream can raise
    ax.text(16.9, 1.55, f"{refr} s silent", ha="center", va="center", fontsize=7.5)  # the refractory period's label
    ax.text(21.2, 1.55, "first event after the onset:\nrefractory at onset, no detection delay", ha="left", va="center", fontsize=7.5)  # what the mark means
    for y in (2.15, 1.55):  # for loop over the two stream rows
        ax.plot([0, 38], [y - 0.17, y - 0.17], color="#999999", lw=0.4)  # a faint time axis under each row
    ax.plot([onset, onset], [1.35, 2.35], color="black", lw=0.6, ls=(0, (2, 2)))  # the onset carried down through both rows
    boxes = [(0.0, f"primary state: the {n_sel} selected columns at the\nwindows ending at t, t − {lags[1]} s and t − {lags[2]} s\n({n_sel * len(lags)} values)"), (20.0, f"surrogate state: the last {n_z} log-odds of\nthe logistic regression and their\ncumulative sum Sₜ")]  # list of (left edge, text) of the two state boxes
    for x, text in boxes:  # for loop over the state boxes
        ax.add_patch(plt.Rectangle((x, 0.15), 18.0, 0.85, fill=False, lw=0.7))  # the box
        ax.text(x + 9.0, 0.575, text, ha="center", va="center", fontsize=7.5)  # its text
    ax.set_xlim(-6.5, 39)  # room for the row names on the left
    ax.set_ylim(0.0, 4.25)  # room for the notes on top
    ax.axis("off")  # no axes: a schematic
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)  # fixed margins, so the PNG is byte-stable
    fig.savefig(OUT_RL, dpi=300, metadata={"Software": None})  # PNG for print; no software tag
    plt.close(fig)  # frees the figure
    print(f"  wrote {OUT_RL.relative_to(PROJECT)}")  # one line for the run log


def main() -> None:  # function: draws the four data figures and the schematics; with the argument "rl", only the schematic of Section 4.10
    if sys.argv[1:] == ["rl"]:  # if: only the new schematic is asked for (the data figures read the raw files and are unchanged)
        figure_rl()  # the schematic of Section 4.10
        return  # nothing else
    figure2()  # Figure 2
    figure3()  # Figure 3
    figure4()  # Figure 4
    figure5()  # Figure 5
    figure_rl()  # the schematic of Section 4.10


if __name__ == "__main__":  # if: run as a script, not imported
    main()  # draws the figures
