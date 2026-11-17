#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
Figures 3 and 4, drawn from the data products in UTC with plain styling legible in print.

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

Usage   /opt/anaconda3/bin/python3 codes/figures/figures_3_and_4.py
Writes  Graph/figure_3.png, Graph/figure_vehicle_speed_265.png, Graph/figure_4.png, Graph/figure_windows.png
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
LAB_COLOR = {"clean": "#009E73", "attack_jam_spoof": "#7A0019", "attack_spoof": "#E69F00", "uncertain": "#C8C8C8", "excluded": "#404040"}  # dict: label -> colour (the label script's palette, readable in grey)
LAB_NAME = {"clean": "clean", "attack_jam_spoof": "attack (jamming and spoofing)", "attack_spoof": "attack (spoofing)", "uncertain": "uncertain", "excluded": "excluded"}  # dict: label -> thesis name
RX_NAME = {"SBF_02": "mosaic-X5", "SBF_03": "mosaic-H", "UBX_01": "u-blox 01"}  # dict: stream -> receiver name
RX_COLOR = {"SBF_02": "#0072B2", "SBF_03": "#E69F00", "UBX_01": "#009E73"}  # dict: stream -> line colour
HELD_OUT = "2.6.12"  # the session held out as the unseen attack type: drawn only as its scheduled window
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})  # dict: one font and size for every label


def utc(text: str) -> pd.Timestamp:  # function: "2026-09-14 12:30:00" -> a timezone-free timestamp read as UTC
    return pd.Timestamp(text)  # the products store UTC text without an offset, so no conversion is needed


def figure2() -> None:  # function: draws Figure 3, the scheduled activity at the three test areas against the capture windows
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


def figure3() -> None:  # function: draws the vehicle-speed figure of test 2.6.5 (not in the thesis)
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


def figure5() -> None:  # function: draws the schematic of the windows (not in the thesis; no data: it states no result)
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


def main() -> None:  # function: draws Figures 3 and 4 and the two figures that are not in the thesis (vehicle speed, windows)
    figure2()  # Figure 3
    figure3()  # the vehicle-speed figure (not in the thesis)
    figure4()  # Figure 4
    figure5()  # the window schematic (not in the thesis)


if __name__ == "__main__":  # if: run as a script, not imported
    main()  # draws the figures
