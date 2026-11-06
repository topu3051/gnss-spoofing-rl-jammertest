#!/usr/bin/env python3
# module docstring below: the file's help text
"""
Decision instants in one table.

Seven instants that the position script, the evidence script, the observables script and the label script use are derived here, each
from the data product it rests on, and written with their provenance; the four scripts read the table instead of typing the instants.
It runs after the motion script (it reads the motion script's preprocessing/vehicle_motion.csv) and before the position script.

The two satellite instants (u-blox 01's true G30 reacquisition in 2.6.2; the 3 s G08 code error of both Septentrio receivers in 2.6.10)
were diagnosed by codes/2_preprocessing/07_satellite_g30_g08_diagnostics.py, which itself imports the position script. They are read here from that script's stored output and recorded
with that provenance; the diagnostics script derives both again from the observations on every run and stops if they differ from what the position script reads.

Reads   Data/processed/preprocessing/vehicle_motion.csv, Data/processed/ground_truth/spoof_2612_timeline.csv and ground_truth/margin_contamination.csv (the receiver-status script),
        Data/processed/checks/g30_diagnostics.csv (the satellite diagnostics script)
Writes  Data/processed/ground_truth/decision_times.csv (name, value_utc, rule, source_file, source_quantity, source_value, used_by)
"""

from __future__ import annotations  # future import: type hints kept as text (a convention across the project's scripts)

import re  # regular expressions: pulls clock times out of text cells
import sys  # system access: sys.path, so the helpers module is found
from pathlib import Path  # Path objects for building file paths

import pandas as pd  # tables and timestamps

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
sys.path.insert(0, str(PROJECT / "codes" / "common"))  # list method insert: puts codes/ first on the import path so the helpers module is found
from helpers import exit_if_failed  # status 1 when a self-check failed

PROC = PROJECT / "Data" / "processed"  # processed data folder
CHECK_DIR = PROJECT / "Data" / "processed" / "checks"  # check folder (g30's outputs live here)
OUT = PROC / "ground_truth" / "decision_times.csv"  # the table written
DATE = "2026-09-14"  # the recording date (UTC)
FAILED: list = []  # list: names of failed self-checks


def floor_s(text: str) -> pd.Timestamp:  # function: a UTC time text with fractions -> the whole second at or before it
    return pd.Timestamp(text, tz="UTC").floor("s")  # Timestamp.floor("s") drops the fraction


def hms_on_date(hms: str) -> pd.Timestamp:  # function: "HH:MM:SS" on the recording date -> a UTC timestamp
    return pd.Timestamp(f"{DATE} {hms}", tz="UTC")  # f-string: date and time joined, tagged UTC


def g30_value(wp: pd.DataFrame, item: str, quantity_start: str) -> str:  # function: the value cell of one g30 row, found by item and quantity prefix
    hit = wp[(wp["item"] == item) & wp["quantity"].str.startswith(quantity_start)]  # boolean row selection on both columns
    if len(hit) != 1:  # if: missing or ambiguous
        sys.exit(f"STOP: g30_diagnostics.csv has {len(hit)} rows for {item!r} / {quantity_start!r}")  # halt: the provenance must be unique
    return str(hit["value"].iloc[0])  # the cell as text


def main() -> None:  # function: derives the seven instants and writes the table
    rows = []  # list of dicts: one row per instant
    vm = pd.read_csv(PROC / "preprocessing" / "vehicle_motion.csv")  # the motion script's motion segments
    mv = vm[(vm.test_code == "2.6.5") & (vm.state == "moving")]  # boolean row selection: the two 2.6.5 drives
    vehicle_motion_start, vehicle_motion_end = floor_s(mv.start_utc.min()), floor_s(mv.end_utc.max())  # tuple assignment: first drive start and last drive end, floored
    # dict literals: the two the 2.6.5 drive exclusion rows; the end second is also where 2.6.5 stands at the 2.6.2 site (the evidence script's SITE_START)
    rows.append({"name": "vehicle_motion_start", "value_utc": vehicle_motion_start, "rule": "first 2.6.5 moving segment start, floored to the second", "source_file": "Data/processed/preprocessing/vehicle_motion.csv", "source_quantity": "2.6.5 state=moving, min start_utc", "source_value": mv.start_utc.min(), "used_by": "06 (drive exclusion start)"})
    # list append: decision-time row for the end of the 2.6.5 drive, from the last moving segment's end
    rows.append({"name": "vehicle_motion_end", "value_utc": vehicle_motion_end, "rule": "last 2.6.5 moving segment end, floored to the second", "source_file": "Data/processed/preprocessing/vehicle_motion.csv", "source_quantity": "2.6.5 state=moving, max end_utc", "source_value": mv.end_utc.max(), "used_by": "06 (drive exclusion end, inclusive); the evidence script (SITE_START of 2.6.5)"})
    wp = pd.read_csv(CHECK_DIR / "g30_diagnostics.csv", dtype=str)  # g30's numbers, all as text
    g30 = g30_value(wp, "G30 u-blox", "reacquisition second")  # "13:02:41"
    # list append: decision-time row for the second u-blox 01's G30 code error falls back under 20 m in 2.6.2
    rows.append({"name": "G30_normal", "value_utc": hms_on_date(g30), "rule": "u-blox 01 2.6.2: first second with a G30 code error under 20 m (g30 g30())", "source_file": "Data/processed/checks/g30_diagnostics.csv", "source_quantity": "G30 u-blox / reacquisition second", "source_value": g30, "used_by": "the position script (G30_NORMAL); the evidence script (DATE_G30_NORMAL); the observables script and the evidence script known_invalid; 06 (invalid_G30 exclusion end)"})
    firsts, lasts = [], []  # two lists: the first and last bad G08 second of each Septentrio receiver
    for name in ("mosaic-X5", "mosaic-H"):  # for loop over the two Septentrio receivers named in g30's G08 diagnostic
        cell = g30_value(wp, f"G08 {name}", "seconds with |code error| over 10 m")  # e.g. "3 14:44:41 to 14:44:43"
        t = re.findall(r"\d\d:\d\d:\d\d", cell)  # every clock time in the cell
        if len(t) != 2:  # if: not exactly a first and a last second
            sys.exit(f"STOP: G08 {name} cell {cell!r} does not hold a first and a last second")  # halt
        firsts.append(t[0])  # the first bad second
        lasts.append(t[1])  # the last bad second
    # list append: decision-time row for the first second with a G08 code error over 10 m (both Septentrio)
    rows.append({"name": "G08_first", "value_utc": hms_on_date(min(firsts)), "rule": "earliest first second with |G08 code error| over 10 m over the two Septentrio receivers (g30 g08())", "source_file": "Data/processed/checks/g30_diagnostics.csv", "source_quantity": "G08 mosaic-X5 / G08 mosaic-H, seconds with |code error| over 10 m", "source_value": "; ".join(firsts), "used_by": "the observables script and the evidence script known_invalid; 06 (G08_transient exclusion start)"})
    # list append: decision-time row for the last second with a G08 code error over 10 m (both Septentrio)
    rows.append({"name": "G08_last", "value_utc": hms_on_date(max(lasts)), "rule": "latest last second with |G08 code error| over 10 m over the two Septentrio receivers (g30 g08())", "source_file": "Data/processed/checks/g30_diagnostics.csv", "source_quantity": "G08 mosaic-X5 / G08 mosaic-H, seconds with |code error| over 10 m", "source_value": "; ".join(lasts), "used_by": "the position script (G08_LAST); the observables script and the evidence script known_invalid; 06 (G08_transient exclusion end)"})
    tl = pd.read_csv(PROC / "ground_truth" / "spoof_2612_timeline.csv", dtype=str)  # the receiver-status script's 2.6.12 timeline
    rel = tl[(tl.receiver == "u-blox_01") & (tl.event == "measurement-engine release")]  # boolean row selection: u-blox 01's release row
    if len(rel) != 1:  # if: not exactly one row
        sys.exit("STOP: ground_truth/spoof_2612_timeline.csv has no unique u-blox 01 measurement-engine release")  # halt
    r_txt = str(rel.host_time_utc.iloc[0])  # "16:04:57.997" on host time
    ubx_release = hms_on_date(r_txt).floor("s") + pd.Timedelta(seconds=1)  # the first whole second after the release (the floored second is still offset, 06 line 310)
    # list append: decision-time row for u-blox 01's clock release in 2.6.12, one second after it on host time
    rows.append({"name": "ubx_release_time", "value_utc": ubx_release, "rule": "u-blox 01 2.6.12 measurement-engine release on host time, floored, plus 1 s (first second after the release)", "source_file": "Data/processed/ground_truth/spoof_2612_timeline.csv", "source_quantity": "u-blox_01 / measurement-engine release / host_time_utc", "source_value": r_txt, "used_by": "06 (ubx_clock_release exclusion start, inclusive)"})
    mc = pd.read_csv(PROC / "ground_truth" / "margin_contamination.csv", dtype=str)  # the receiver-status script's margin contamination table
    note = str(mc.loc[mc.test_code == "2.6.12", "note"].iloc[0])  # "possibly contaminated 15:42:29-15:45:12"
    m = re.search(r"-(\d\d:\d\d:\d\d)$", note)  # the end time after the dash, as the margin table writes it
    if m is None:  # if: the note does not end in a time range
        sys.exit(f"STOP: ground_truth/margin_contamination.csv 2.6.12 note {note!r} holds no end time")  # halt
    # list append: decision-time row for the end of the possibly contaminated 2.6.12 margin
    rows.append({"name": "margin_contamination_end", "value_utc": hms_on_date(m.group(1)), "rule": "end of the possibly contaminated 2.6.12 margin stretch", "source_file": "Data/processed/ground_truth/margin_contamination.csv", "source_quantity": "2.6.12 / note", "source_value": note, "used_by": "the position script (MARGIN_CONTAMINATION_END); the evidence script (2.6.12 pre-test margin start)"})
    df = pd.DataFrame(rows)  # the table
    df["value_utc"] = df.value_utc.map(lambda t: t.strftime("%Y-%m-%d %H:%M:%S"))  # lambda in map: every instant as "YYYY-MM-DD HH:MM:SS" UTC text
    # the instants the scripts typed before this table existed: each derived value must equal the old literal (no result may move in the final-label run)
    typed = {"vehicle_motion_start": "12:27:22", "vehicle_motion_end": "12:34:57", "G30_normal": "13:02:41", "G08_first": "14:44:41", "G08_last": "14:44:43", "ubx_release_time": "16:04:58", "margin_contamination_end": "15:45:12"}
    for _, r in df.iterrows():  # for loop over the rows; "_" ignores the index
        ok = r.value_utc.endswith(typed[r["name"]])  # str.endswith: the derived clock time equals the formerly typed one
        print(f"  [{'ok' if ok else 'FAIL'}] {r['name']}: derived {r.value_utc}, formerly typed {typed[r['name']]}")  # one line each
        if not ok:  # if: the derivation does not give the value the results were built with
            FAILED.append(r["name"])  # recorded for the exit
    df.to_csv(OUT, index=False, lineterminator="\n")  # written with Unix line ends, byte-stable
    print(f"decision_times.csv: {len(df)} instants")  # summary line


if __name__ == "__main__":  # guard: runs main() only when the file is executed, not when imported
    main()  # entry point
    exit_if_failed(FAILED, "1_ground_truth/05_decision_times.py")  # the helpers module: status 1 when a derived instant differs from the typed one
