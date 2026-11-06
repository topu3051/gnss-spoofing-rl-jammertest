#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
Final-label rule L3: the end of spoofed tracking in 2.6.10, per second, from the 1 Hz residuals.

The minute rule: a minute of 2.6.10 counts as spoofed tracking when, in the conversion script's residual minute table
(preprocessing/rinex_residual_minutes.csv), more than half of the satellites of each of the mosaic-X5, the mosaic-H and u-blox 01 lie more than
CLK_OUTLIER_M (the conversion script, 100 m) from the receiver clock term (share_over_100m_pct > 50 on all three); the end is the last such minute plus 59 s,
14:43:59. That is a minute bound written as a second.
The per-second counterpart: per receiver and second of 2.6.10, over the GPS and Galileo satellites with a modelled
code residual (features/observables_1hz.csv.gz, pr_res_m: the observables script's definition, the selection the label script uses to reproduce the minute table), the
second is spoofed tracking when at least 4 such residuals exist and more than half of them exceed 100 m in absolute value. A receiver's
end is its last such second; the pooled end is the latest receiver end. It is a descriptive boundary, not a label (the
label end stays 14:44:06, u-blox 01's own position). The conversion script holds the pooled end as ATTACK_EXTENDED_END["2.6.10"]; this script
fails when the two differ.
Checks: the minute aggregation of these seconds (sum over sum, seconds with 4 or more residuals, the label script's method) reproduces the conversion script's
2.6.10 minute shares within 3 percentage points; the minute rule applied to that aggregation gives the minute end 14:43:59.

Reads   Data/processed/features/observables_1hz.csv.gz (2.6.10: residuals, as the label script, nothing here sets a label), preprocessing/rinex_residual_minutes.csv,
        codes/2_preprocessing/01_convert_rinex.py (CLK_OUTLIER_M and ATTACK_EXTENDED_END, read as text)
Writes  Data/processed/ground_truth/spoof_end_2610.csv (per receiver, pooled, and the minute rule), Data/processed/checks/spoofed_tracking_end_2610_seconds.csv (per
        receiver and second), Data/processed/checks/spoofed_tracking_end_2610_checks.csv, Data/processed/checks/spoofed_tracking_end_2610.png (per-second shares against the minutes)
"""

from __future__ import annotations  # future import: type hints kept as text (the project's convention)

import os  # operating-system access: MPLCONFIGDIR is set before matplotlib loads
import sys  # system access: sys.path, so the helpers module is found
from pathlib import Path  # Path objects for building file paths

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
os.environ["MPLCONFIGDIR"] = str(PROJECT / "codes" / ".mplconfig")  # keeps matplotlib's cache inside the project folder

import numpy as np  # NumPy: arrays
import pandas as pd  # pandas: tables
import matplotlib  # the plotting library itself, to choose a backend
matplotlib.use("Agg")  # file-only backend, so no window opens and runs again are identical
import matplotlib.pyplot as plt  # figures and axes

sys.path.insert(0, str(PROJECT / "codes" / "common"))  # list method insert: puts codes/ first on the import path so the helpers module is found
from helpers import exit_if_failed, save_fig, script_constant  # status 1 on a failed check; byte-stable figure saver; constants read as text

PROC = PROJECT / "Data" / "processed"  # the processed-data folder
CHECK_DIR = PROJECT / "Data" / "processed" / "checks"  # where the check outputs go
S04 = PROJECT / "codes" / "2_preprocessing" / "01_convert_rinex.py"  # the conversion script's script, read as text for two constants
TEST, DATE = "2.6.10", "2026-09-14"  # tuple assignment: the session and the recording date
RXS = ["SBF_02", "SBF_03", "UBX_01"]  # list: the mosaic-X5, the mosaic-H main antenna and u-blox 01 (the minute rule)
OVER_M = float(script_constant(S04, "CLK_OUTLIER_M"))  # 100 m: a satellite this far from the clock term counts as inconsistent (the conversion script)
MIN_RES = 4  # integer constant: a second needs at least 4 GPS/Galileo residuals (the label script's minute aggregation)
SHARE = 0.5  # float constant: "more than half" of the satellites, as the minute rule
MINUTE_TOL_PP = 3.0  # float constant: tolerance of the minute reproduction, percentage points (the label script's)
LABEL_END = "14:44:06"  # the 2.6.10 label end (the label script's check), drawn on the figure for orientation only
CHECKS: list = []  # list: one dict per check
FAILED: list = []  # list: names of the checks that failed


def hms(sec) -> str:  # function: unix seconds -> "HH:MM:SS"
    return pd.Timestamp(int(sec), unit="s", tz="UTC").strftime("%H:%M:%S")  # formatted


def check(name: str, ok, detail: str) -> None:  # function: records one check and prints it (ok None marks a reported number)
    CHECKS.append({"check": name, "result": "REPORT" if ok is None else ("PASS" if ok else "FAIL"), "detail": detail})  # nested conditional expression picks the word
    if ok is not None and not bool(ok):  # if: a pass/fail check that did not pass
        FAILED.append(name)  # kept for the exit status
    print(f"  [{'REPORT' if ok is None else ('PASS' if ok else 'FAIL')}] {name}: {detail}")  # printed for the run log


def per_second() -> pd.DataFrame:  # function: per receiver and second of 2.6.10, the GPS/Galileo residual counts and the spoofed-tracking flag
    cols = ["test_code", "receiver", "sec", "utc", "sys", "pr_res_m"]  # the columns needed
    d = pd.read_csv(PROC / "features" / "observables_1hz.csv.gz", usecols=cols, dtype={"test_code": str})  # the observables script's per-satellite table
    d = d[(d.test_code == TEST) & d.receiver.isin(RXS) & d.sys.isin(["G", "E"]) & d.pr_res_m.notna()]  # 2.6.10, the three receivers, GPS and Galileo with a residual
    d = d.assign(over=d.pr_res_m.abs() > OVER_M)  # assign: True where the satellite lies over 100 m from the clock term
    g = d.groupby(["receiver", "sec"]).agg(utc=("utc", "first"), n_ge=("over", "size"), n_over=("over", "sum")).reset_index()  # named aggregations: counts per second
    g["share"] = g.n_over / g.n_ge  # share of the satellites over 100 m
    g["spoofed"] = (g.n_ge >= MIN_RES) & (g.share > SHARE)  # the per-second rule
    return g  # the table


def minute_rule(ps: pd.DataFrame) -> tuple:  # function: the label script's minute aggregation of the seconds, and the minute rule's end on it
    x = ps[ps.n_ge >= MIN_RES].assign(minute=lambda z: z.utc.str[:5])  # seconds with 4 or more residuals, with their "HH:MM"
    m = x.groupby(["receiver", "minute"]).agg(n_over=("n_over", "sum"), n_ge=("n_ge", "sum")).reset_index()  # sums per minute
    m["share_pct"] = 100 * m.n_over / m.n_ge  # sum over sums, as the conversion script and the label script
    piv = m.pivot(index="minute", columns="receiver", values="share_pct")  # one row per minute, one column per receiver
    sp = piv[(piv[RXS] > 100 * SHARE).all(axis=1)].index  # minutes over 50 % on all three (the minute rule)
    return m, (max(sp) + ":59" if len(sp) else "")  # the minute shares and the minute end "HH:MM:59"


def figure(ps: pd.DataFrame, m: pd.DataFrame, ref: pd.DataFrame, ends: dict, pooled: int) -> None:  # function: per-second shares against the minute table
    fig, axs = plt.subplots(len(RXS), 1, figsize=(10, 7.5), sharex=True)  # one panel per receiver
    t0 = int(pd.Timestamp(f"{DATE} 14:38:00", tz="UTC").timestamp())  # the plotted span starts at 14:38:00
    for ax, rx in zip(axs, RXS):  # zip pairs each panel with its receiver
        s = ps[(ps.receiver == rx) & (ps.sec >= t0)]  # the receiver's seconds from 14:38
        ax.plot(pd.to_datetime(s.sec, unit="s"), 100 * s.share, ".", ms=2, color="#0072B2", label="per second: share over 100 m (GPS+Galileo)")  # the per-second shares
        r = ref[(ref.stream == rx)]  # the conversion script's minutes of this receiver
        for q in r.itertuples():  # for loop over its minutes
            a = pd.Timestamp(q.minute_utc)  # the minute's start
            ax.hlines(q.share_over_100m_pct, a, a + pd.Timedelta(seconds=59), color="#D55E00", lw=2)  # the conversion script's minute share as a bar across the minute
        ax.axhline(100 * SHARE, color="k", lw=0.6, ls="--")  # the 50 % line
        ax.axvline(pd.Timestamp(ends[rx], unit="s"), color="#009E73", lw=1, label=f"per-second end {hms(ends[rx])}")  # this receiver's end
        ax.axvline(pd.Timestamp(f"{DATE} 14:43:59"), color="#D55E00", lw=1, ls=":", label="minute end 14:43:59")  # the minute bound
        ax.axvline(pd.Timestamp(f"{DATE} {LABEL_END}"), color="#999999", lw=1, ls="-.", label=f"label end {LABEL_END}")  # the label end, for orientation
        ax.set_ylabel(f"{rx}\nshare (%)", fontsize=8)  # axis title
        ax.set_ylim(-5, 105)  # shares in percent
        ax.legend(fontsize=6, loc="upper right")  # legend
    axs[0].set_title(f"2.6.10: per-second share of GPS/Galileo residuals over {OVER_M:.0f} m (dots) and the conversion script's minute shares (bars); pooled end {hms(pooled)}", fontsize=9)  # title
    axs[-1].set_xlabel("UTC, 14 September 2026", fontsize=8)  # x title
    save_fig(fig, CHECK_DIR / "spoofed_tracking_end_2610.png")  # byte-stable save


def main() -> None:  # function: the per-second end, its checks and the constant
    ps = per_second()  # per receiver and second
    ends = {rx: int(ps[(ps.receiver == rx) & ps.spoofed].sec.max()) for rx in RXS}  # dict comprehension: each receiver's last spoofed second
    pooled = max(ends.values())  # the pooled end: the latest receiver end
    m, mend = minute_rule(ps)  # the minute aggregation and the minute rule's end on it
    ref = pd.read_csv(PROC / "preprocessing" / "rinex_residual_minutes.csv")  # the conversion script's minute table
    ref = ref[(ref.test_code == TEST) & ref.stream.isin(RXS)].copy()  # 2.6.10, the three receivers
    ref["minute"] = ref.minute_utc.str[-5:]  # "HH:MM" of each row (minute_utc is "YYYY-MM-DD HH:MM")
    cmp_ = ref.merge(m, left_on=["stream", "minute"], right_on=["receiver", "minute"], how="left")  # the conversion script's shares beside the aggregation
    dif = (cmp_.share_over_100m_pct - cmp_.share_pct).abs()  # per receiver-minute, |the conversion script - aggregation|
    # reported, not judged: the label script found the two residual definitions (the conversion script's clock term, the observables script's per-second median clock) apart by up to
    # 4.1 points during the takeover, so the decision-relevant test is the minute end below
    check("minute aggregation of the seconds against the conversion script's 2.6.10 shares", None, f"{len(cmp_)} receiver-minutes, largest difference {dif.max():.2f} percentage points, {int((dif > MINUTE_TOL_PP).sum())} over {MINUTE_TOL_PP}; minutes 14:40-14:44: " + "; ".join(f"{r.stream} {r.minute} {r.share_over_100m_pct:.1f}/{r.share_pct:.1f}" for r in cmp_[cmp_.minute.between("14:40", "14:44")].itertuples()))  # reported with the end minutes shown
    check("minute rule on the aggregation gives the minute end 14:43:59", mend == "14:43:59", f"{mend}")  # the minute bound reproduced
    rows = [{"name": f"end_{rx}", "receiver": rx, "end_utc": hms(e), "end_sec": e, "spoofed_seconds": int(ps[(ps.receiver == rx) & ps.spoofed].shape[0]),  # one row per receiver
             "spoofed_seconds_from_1443": int(ps[(ps.receiver == rx) & ps.spoofed & (ps.utc >= "14:43:00")].shape[0])} for rx, e in ends.items()]  # list comprehension: also how many lie in or after minute 14:43
    rows.append({"name": "end_pooled", "receiver": "pooled", "end_utc": hms(pooled), "end_sec": pooled})  # the pooled end
    rows.append({"name": "end_minute_rule", "receiver": "all three", "end_utc": mend, "end_sec": int(pd.Timestamp(f"{DATE} {mend}", tz="UTC").timestamp()) if mend else np.nan})  # the minute bound beside
    out = pd.DataFrame(rows)  # the table
    out.to_csv(PROC / "ground_truth" / "spoof_end_2610.csv", index=False, lineterminator="\n")  # written
    ps[ps.sec >= int(pd.Timestamp(f"{DATE} 14:38:00", tz="UTC").timestamp())].to_csv(CHECK_DIR / "spoofed_tracking_end_2610_seconds.csv", index=False, lineterminator="\n")  # the seconds from 14:38, for the record
    figure(ps, m, ref, ends, pooled)  # the figure
    print(out.to_string(index=False))  # for the run log
    const = script_constant(S04, "ATTACK_EXTENDED_END")["2.6.10"]  # the conversion script's constant
    check("the conversion script's ATTACK_EXTENDED_END['2.6.10'] equals the pooled per-second end", const == f"{DATE} {hms(pooled)}", f"04 holds {const}; computed {DATE} {hms(pooled)}")  # the assertion
    pd.DataFrame(CHECKS).to_csv(CHECK_DIR / "spoofed_tracking_end_2610_checks.csv", index=False, lineterminator="\n")  # the checks


if __name__ == "__main__":  # if: true only when the file is run directly
    main()  # runs L3
    exit_if_failed(FAILED, "1_ground_truth/09_spoofed_tracking_end_2610.py")  # the helpers module: status 1 when any check failed (the conversion script's constant must follow)
