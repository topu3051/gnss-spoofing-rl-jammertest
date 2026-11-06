#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
The observed jamming phase of 2.6.5, and the phase boundaries that can already be read from the front-end gain.

The 2.6.5 stretch 12:36:00-12:40:00 is the OBSERVED jamming phase. The evidence is independent of
the residuals: on the mosaic-X5 all bands step down together at about 12:36:00; the L1/E1, G1 and B1 bands step back at about
12:40:00 while L2 and L5 stay down, which is the catalogue's phase definition (L1, G1, B1 initial only; L2, L5, E5b, E6 continuous). The
scheduled phase 12:30:00-12:35:00 is recorded as "scheduled, no effect visible". Before the relabelling this script checks the same
pattern on the mosaic-H (main antenna) and on the u-blox 01 (its own L1 and L2 blocks); the group file is written only if all three agree on the
LEVEL criteria (every band below its level before, the L1 group back after, the continuous bands still down, both L1 edges within 8 s of the stated
boundaries). A first version gated on the largest single-second step near each boundary and stopped: on the mosaic-H a one-second upward spike
precedes the fall and on the u-blox a one-second dip precedes the recovery, so the criterion was too crude; those columns are kept for disclosure.

The status fields are read for labelling and verification only; they never enter a feature.

Reads   Data/processed/ground_truth/receiver_status_timeseries.csv, ground_truth/session_ground_truth.csv
Writes  Data/processed/ground_truth/gain_pattern_265.csv      per receiver and band: gain before, in, after the phase and the step times
        Data/processed/ground_truth/gain_checks_265.csv       the criteria per receiver
        Data/processed/ground_truth/phase_boundaries.csv      the gain steps of the L1 group and of the continuous bands, all four sessions, X5
        Data/processed/ground_truth/group_overrides.csv            the relabelling that codes/4_features/06_observable_effects.py applies
"""

from __future__ import annotations  # future import: type hints kept as text (a convention across the project's scripts)

import random  # standard-library random; seeded below
from pathlib import Path  # Path objects for building file paths

import numpy as np  # NumPy: arrays and medians
import pandas as pd  # pandas: tables

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
PROC = PROJECT / "Data" / "processed"  # where the CSVs are read and written
SEED = 20260914  # integer constant: the project's fixed random seed (nothing here draws from it)
random.seed(SEED)  # seeds Python's generator
np.random.seed(SEED)  # seeds NumPy's generator

DATE = "2026-09-14"  # the one recording date of all six sessions
PHASE_START, PHASE_END = "12:36:00", "12:40:00"  # the observed jamming phase of 2.6.5, as decided by the student
SCHEDULED_END = "12:35:00"  # the scheduled end of the initial jamming in 2.6.5
STEP_WINDOW_S = 12  # integer constant: seconds either side of a boundary in which a gain step must lie
L1_BANDS = ["GPSL1/E1", "B1"]  # list: the Septentrio bands of the initial-jamming group that carry a clear step (G1 shows almost no gain response on either mosaic)
CONT_BANDS = ["GPSL2", "E5b/B2b", "L5/E5a/B2a"]  # list: the continuously jammed Septentrio bands
RX = {"Septentrio_02": ("SBF_02", "agc_gain_db", 1.0), "Septentrio_03": ("SBF_03", "agc_gain_db", 1.0), "u-blox_01": ("UBX_01", "agc_cnt", 351.0)}  # status tag -> stream, field, one gain step (dB, or counts on the u-blox)
UB_BANDS = {"L1": "L1 group", "L2": "continuous"}  # dict: the u-blox's own two blocks and what they stand for


def T(hms: str) -> int:  # function: "HH:MM:SS" on the recording date -> unix seconds
    return int(pd.Timestamp(f"{DATE} {hms}", tz="UTC").timestamp())  # a UTC timestamp as whole seconds


def hms(sec) -> str:  # function: unix seconds -> HH:MM:SS
    return pd.to_datetime(int(sec), unit="s", utc=True).strftime("%H:%M:%S")  # clock text


def series(st: pd.DataFrame, tag: str, band: str, field: str, a: int, b: int) -> pd.Series:  # function: one band's gain per second on a full grid, gaps up to five seconds filled
    g = st[(st.receiver == tag) & (st.band == band)].groupby("sec")[field].median()  # the band's readings per second
    return g.reindex(np.arange(a, b)).interpolate(limit=5)  # a regular grid with short gaps filled


def largest_step(s: pd.Series, centre: int) -> tuple:  # function: time offset (s) and size of the largest one-second change within the window around `centre`
    d = s.diff().loc[centre - STEP_WINDOW_S:centre + STEP_WINDOW_S]  # signed one-second changes near the boundary
    if d.notna().sum() == 0:  # if: no data
        return np.nan, np.nan  # nothing
    k = d.abs().idxmax()  # the second of the largest change
    return float(k - centre), float(d.loc[k])  # offset from the boundary and the signed size


def pattern() -> tuple:  # function: the 2.6.5 gain pattern on the three receivers and the criteria
    st = pd.read_csv(PROC / "ground_truth" / "receiver_status_timeseries.csv", low_memory=False)  # the status table (gains), verification only
    st = st[(st.test_code == "2.6.5") & ~st.band.str.contains("Aux")].copy()  # 2.6.5, main antenna only
    st["sec"] = pd.to_datetime(st.timestamp_utc, utc=True).astype("int64") // 10 ** 9  # whole unix seconds
    a, b = T("12:30:00"), T("12:50:00")  # the window read (the scheduled test window)
    w = lambda s, x, y: float(s.loc[T(x):T(y)].median())  # median of a gain series between two clock times
    rows, checks = [], []  # per-band rows and per-receiver criteria
    for tag, (rx, field, unit) in RX.items():  # for loop over the three receivers
        bands = [bd for bd in st[st.receiver == tag].band.unique()]  # the bands this receiver reports
        res = {}  # band -> (pre, dip, post, late)
        for bd in sorted(bands):  # for loop over the bands
            s = series(st, tag, bd, field, a, b)  # the band's gain
            pre, dip, post, late = w(s, "12:34:00", "12:35:50"), w(s, "12:36:10", "12:39:50"), w(s, "12:40:10", "12:41:30"), w(s, "12:42:00", "12:46:00")  # four windows
            so, sz = largest_step(s, T(PHASE_START))  # the step near the phase start
            eo, ez = largest_step(s, T(PHASE_END))  # the step near the phase end
            res[bd] = (pre, dip, post, late, so, sz, eo, ez)  # kept
            rows.append({"receiver": rx, "band": bd, "field": field, "before": pre, "in_phase": dip, "after": post, "later": late, "start_step_offset_s": so, "start_step": sz, "end_step_offset_s": eo, "end_step": ez})  # one row
        l1 = [bd for bd in (L1_BANDS if rx != "UBX_01" else ["L1"]) if bd in res]  # the L1-group bands this receiver has
        cont = [bd for bd in (CONT_BANDS if rx != "UBX_01" else ["L2"]) if bd in res]  # the continuous bands this receiver has
        down_all = all(res[bd][5] <= -unit and abs(res[bd][4]) <= STEP_WINDOW_S for bd in l1 + cont)  # every shown band steps down within the window around the start
        l1_back = all(abs(res[bd][2] - res[bd][0]) <= unit for bd in l1)  # the L1 group is back within one step of its level before
        cont_down = all((res[bd][0] - res[bd][2]) >= (2 * unit if rx != "UBX_01" else unit) for bd in cont)  # the continuous bands stay at least two dB (one count step) below
        up_step = all(res[bd][7] >= unit and abs(res[bd][6]) <= STEP_WINDOW_S for bd in l1)  # the L1 group steps up within the window around the end
        gm = st[(st.receiver == tag) & st.band.isin(l1)].groupby("sec")[field].mean().reindex(np.arange(a, b)).interpolate(limit=5)  # the L1 group's mean gain per second
        pre_l, dip_l, post_l = float(gm.loc[T("12:34:00"):T("12:35:50")].mean()), float(gm.loc[T("12:36:10"):T("12:39:50")].mean()), float(gm.loc[T("12:40:10"):T("12:41:30")].mean())  # the group's level before, in and after the phase
        rm = gm.rolling(5, center=True, min_periods=3).mean()  # smoothed over five seconds
        mid1, mid2 = (pre_l + dip_l) / 2, (dip_l + post_l) / 2  # the midpoints of the two edges
        t_dn = [k for k, v in rm.loc[T("12:35:30"):T("12:37:30")].items() if v <= mid1]  # seconds below the first midpoint
        t_up = [k for k, v in rm.loc[T("12:39:00"):T("12:41:30")].items() if v >= mid2]  # seconds above the second midpoint
        off_dn, off_up = (t_dn[0] - T(PHASE_START)) if t_dn else np.nan, (t_up[0] - T(PHASE_END)) if t_up else np.nan  # offsets of the first crossings from the stated boundaries
        in_below = all(res[bd][0] - res[bd][1] >= unit for bd in l1 + cont)  # every shown band is at least one unit below its level before, inside the phase
        edges = bool(abs(off_dn) <= 8 and abs(off_up) <= 8)  # both edges of the L1 group lie within eight seconds of the stated boundaries
        # append a dict over three lines: one receiver's level criteria and the single-second criteria kept for disclosure
        checks.append({"receiver": rx, "l1_bands": ";".join(l1), "continuous_bands": ";".join(cont), "l1_group_before": pre_l, "l1_group_in_phase": dip_l, "l1_group_after": post_l,
                       # dict entry: L1 edge offsets, all bands low in the phase, and L1 back after it
                       "l1_edge_down_offset_s": off_dn, "l1_edge_up_offset_s": off_up, "all_bands_below_in_phase": bool(in_below), "l1_group_back_after": bool(l1_back),
                       # dict entry: continuous bands stay down, both L1 edges within 8 s, and whether all criteria hold
                       "continuous_bands_stay_down": bool(cont_down), "l1_edges_within_8s": edges, "level_criteria_all": bool(in_below and l1_back and cont_down and edges),
                       "single_second_step_down_at_start": bool(down_all), "single_second_step_up_at_end": bool(up_step)})  # the level criteria gate the relabelling; the single-second step criteria are kept for disclosure
    return pd.DataFrame(rows), pd.DataFrame(checks)  # the two tables


def boundaries() -> pd.DataFrame:  # function: the gain steps of the L1 group and of the continuous bands on the mosaic-X5, four sessions, beside the scheduled boundaries
    st = pd.read_csv(PROC / "ground_truth" / "receiver_status_timeseries.csv", low_memory=False)  # the status table, verification only
    st = st[(st.receiver == "Septentrio_02") & st.test_code.isin(["2.6.5", "2.6.4", "2.6.11", "2.6.12"])].copy()  # the X5 in the four sessions
    st["sec"] = pd.to_datetime(st.timestamp_utc, utc=True).astype("int64") // 10 ** 9  # whole unix seconds
    gt = pd.read_csv(PROC / "ground_truth" / "session_ground_truth.csv").set_index("test_code")  # scheduled windows
    ta3 = pd.read_csv(PROC / "ground_truth" / "ta3_schedule_confirmation.csv")  # the receiver-status script's initial-jamming end
    rows = []  # output rows
    for t in ["2.6.5", "2.6.4", "2.6.11", "2.6.12"]:  # for loop over the sessions
        y = st[st.test_code == t]  # the session
        a, b = int(y.sec.min()), int(y.sec.max())  # the span of the status data
        je = ta3[ta3.test_code == t].initial_jamming_end_utc.iloc[0]  # the scheduled end of the initial jamming
        for name, bands in (("L1 group (L1/E1, B1)", L1_BANDS), ("continuous (L2, L5, E5b)", CONT_BANDS)):  # for loop over the two groups
            s = y[y.band.isin(bands)].groupby("sec").agc_gain_db.mean().reindex(np.arange(a, b + 1)).interpolate(limit=6).rolling(5, center=True, min_periods=2).median()  # the group's mean gain, smoothed
            d = s.shift(-3) - s.shift(3)  # change across six seconds
            big = d[d.abs() >= 3.0]  # changes of at least three dB
            last = -999  # the previous event second
            for k, v in big.items():  # for loop over the large changes
                if k - last > 20:  # if: a new event, more than twenty seconds after the last
                    # append a dict over three lines: one gain step event with the scheduled boundaries and the offsets to them
                    rows.append({"test_code": t, "group": name, "event_utc": hms(k), "change_db": round(float(v), 1), "direction": "down" if v < 0 else "up",
                                 # dict entry: scheduled start, end of the initial jamming and scheduled end, as clock times
                                 "scheduled_start": gt.loc[t, "scheduled_start_utc"][11:], "scheduled_initial_jamming_end": je if isinstance(je, str) else "none (exempt)", "scheduled_end": gt.loc[t, "scheduled_end_utc"][11:],
                                 "offset_to_scheduled_start_s": int(k - T(gt.loc[t, "scheduled_start_utc"][11:])), "offset_to_scheduled_initial_end_s": int(k - T(je)) if isinstance(je, str) and je[:1].isdigit() else np.nan})  # one event
                last = k  # remember it
    return pd.DataFrame(rows)  # the table


def main() -> None:  # function: runs the checks, writes the group file
    pat, chk = pattern()  # the 2.6.5 pattern and criteria
    pat.round(2).to_csv(PROC / "ground_truth" / "gain_pattern_265.csv", index=False)  # written
    chk.to_csv(PROC / "ground_truth" / "gain_checks_265.csv", index=False)  # written
    print(pat.round(1).to_string())  # for the report
    print(chk.to_string())  # for the report
    bd = boundaries()  # the phase boundaries that can already be read
    bd.to_csv(PROC / "ground_truth" / "phase_boundaries.csv", index=False)  # written
    print(bd.to_string())  # for the report
    if not bool(chk.level_criteria_all.all()):  # if: any receiver disagrees on the level criteria
        raise SystemExit("the gain pattern does not hold on all three receivers: no relabelling written; see ground_truth/gain_checks_265.csv")  # stop, as instructed
    rows = []  # relabelling rows
    for rx in ("SBF_02", "SBF_03", "UBX_01"):  # for loop over the receivers
        rows.append({"test_code": "2.6.5", "receiver": rx, "start_sec": T("12:34:57"), "end_sec": T(SCHEDULED_END) - 1, "group": "scheduled-no-effect", "reason": "scheduled initial jamming 12:30-12:35, no effect visible in the gain"})  # the scheduled seconds that remain in the span
        rows.append({"test_code": "2.6.5", "receiver": rx, "start_sec": T(PHASE_START), "end_sec": T(PHASE_END), "group": "jamming", "reason": "observed jamming phase: L1/E1, G1, B1 gain steps down at 12:36:00 and back at 12:40:00, L2/L5 stay down"})  # the observed phase
    pd.DataFrame(rows).to_csv(PROC / "ground_truth" / "group_overrides.csv", index=False)  # written


if __name__ == "__main__":  # guard: runs main() only when the file is executed, not when imported
    main()  # entry point
