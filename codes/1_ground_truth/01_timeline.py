#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text (help(module))
"""
Ground truth and time alignment.

Builds the ground-truth timeline for the six Jammertest 2026 capture sessions
recorded at Test Area 3 on Monday 14 September 2026.

This script touches NO receiver payload data. It reads only:
  - README.txt / session_events.log / stream_gaps.log / timestamps_system_capture.csv
  - Data/testcatalog.json
  - Data/transmission-plan-2026-09-14-to-2026-09-18.pdf
  - the Android GnssLogger .json metadata headers and .zip directories

It writes only to Data/processed/ and Graph/.

Outputs
  Data/processed/ground_truth/session_ground_truth.csv
  Data/processed/ground_truth/session_sensors.csv
  Data/processed/ground_truth/test_phases.csv
  Data/processed/ground_truth/concurrent_transmissions.csv
  Data/processed/ground_truth/phone_capture_inventory.csv
  Data/processed/ground_truth/time_base_hypotheses.csv
  Graph/session_vs_schedule.png
  Graph/test_area_activity.png
"""

from __future__ import annotations  # future import: hints kept as text, so "datetime | None" works on older Pythons

import json  # standard-library JSON reader, used to load the machine-readable test catalogue
import os  # standard-library operating-system module; os.environ sets matplotlib's cache folder below
import random  # standard-library random numbers; only seeded below, nothing here draws from it
import re  # standard-library regular expressions: matches folder names, plan lines and catalogue sentences
import sys  # standard-library system access: die() writes to stderr and exits with a failure status
import zipfile  # standard-library zip reader, lists the phone .zip archives' contents without unpacking them
from datetime import datetime, timedelta, timezone  # date-time classes: instants, durations and the UTC time zone
from pathlib import Path  # Path objects for building and reading file paths
from zoneinfo import ZoneInfo  # time-zone database lookup, used for Europe/Oslo with its summer-time rule

import numpy as np  # third-party numerical library under the short name np; only seeded in this script
import pandas as pd  # third-party table library under the short name pd; builds and writes the CSV outputs
import pypdf  # third-party PDF reader, extracts positioned text from the transmission plan

# Path: codes/.mplconfig, the folder matplotlib keeps its configuration and font cache in, inside the project folder
MPLCONFIG = Path(__file__).resolve().parent / ".mplconfig"
MPLCONFIG.mkdir(exist_ok=True)  # creates the folder if missing; exist_ok=True means no error when it exists
os.environ["MPLCONFIGDIR"] = str(MPLCONFIG)  # os.environ: tells matplotlib, imported next, to use that folder
import matplotlib  # third-party plotting library, imported first so its drawing back end can be chosen
matplotlib.use("Agg")  # picks the "Agg" back end: draws straight to image files, no screen window needed
import matplotlib.pyplot as plt  # plotting interface under the short name plt; must come after use("Agg")
from matplotlib.patches import Patch  # coloured-box shape used to build custom legend entries

# --------------------------------------------------------------------------
# Constants and configuration
# --------------------------------------------------------------------------

SEED = 20260914  # constant: the project's fixed seed (the capture date), recorded for reproducibility
random.seed(SEED)  # seeds Python's random generator; nothing here draws random numbers, it is fixed on principle
np.random.seed(SEED)  # seeds NumPy's global random generator for the same reason

# the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "Data"  # the '/' operator on a Path joins path parts: the project folder/Data
DAY = DATA / "Day1_14092026"  # the raw capture-day folder holding the six session folders and the phone folders
PROCESSED = DATA / "processed"  # folder this script writes its six CSV outputs into
GRAPH = PROJECT / "Graph"  # folder this script writes its two figures into

CATALOG_PATH = DATA / "testcatalog.json"  # path to the machine-readable test catalogue
PLAN_PATH = DATA / "transmission-plan-2026-09-14-to-2026-09-18.pdf"  # path to the transmission plan PDF

# ZoneInfo object for Norway; it knows the summer-time rule, so on 14 September it gives UTC+02:00
TZ_OSLO = ZoneInfo("Europe/Oslo")        # Norwegian local time (CEST on this date)
TZ_UTC = timezone.utc  # the UTC time-zone object; UTC is the single internal time base

TEST_DAY = "2026-09-14"  # the capture date as text, used to find the plan pages and to build datetimes
# string constant: the weekday word printed in the plan's page header for 14 September
PLAN_DAY_HEADER = "Monday"               # page header marker for the day we need

# compiled regex for folder names like 20260914_152717_T_2.6.5; its three groups capture date, time and test code
SESSION_DIR_RE = re.compile(r"^(\d{8})_(\d{6})_T_(\d+(?:\.\d+)+)$")
# compiled regex for a plan entry header like "14:30-14:50 - 2.6.5"; groups capture start, end and test code
ENTRY_RE = re.compile(r"^(\d{2}:\d{2})-(\d{2}:\d{2})\s*-\s*(\d+(?:\.\d+)+)\s*$")
# compiled regex for a plan line like "Power: 0.001 W", case-insensitive; the group captures the number of watts
POWER_RE = re.compile(r"^Power:\s*([0-9]*\.?[0-9]+)\s*W", re.IGNORECASE)

# Okabe-Ito colourblind-safe palette
C_ORANGE = "#E69F00"  # hex colour string: orange, used for the actual capture bars in Figure 1
C_SKY = "#56B4E9"  # hex colour: sky blue, used for scheduled windows and low-power tests
C_GREEN = "#009E73"  # hex colour: green, used for margins, our capture windows and the outline of our tests
C_VERMILLION = "#D55E00"  # hex colour: vermillion, used for tests above 1 W and for partial-coverage notes
C_GREY = "#999999"  # hex colour: grey, used for the time-axis grid lines
C_BLACK = "#000000"  # hex colour: black, used for bar edges and the clipping arrows

# tuple of two strings: the start and end of Figure 2's time axis, in Norwegian local time
FIG2_WINDOW = ("14:00", "18:30")         # Norwegian local clipping window for Figure 2
# float constant: power in watts; tests strictly above it are hatched and labelled in Figure 2
HIGH_POWER_W = 1.0                       # "above 1 W" threshold for Figure 2 emphasis

ANOMALIES: list[str] = []  # annotated empty list: collects every problem flagged during the run, printed at the end


def flag(msg: str) -> None:  # function definition: takes a message and returns nothing (-> None)
    """Record something that looks wrong, rather than smoothing it over."""
    ANOMALIES.append(msg)  # list.append adds the message to the module-level anomaly list


def die(msg: str) -> None:  # function: stops the whole script with a readable message
    print("\n*** STOPPING: " + msg, file=sys.stderr)  # string concatenation; file=sys.stderr prints to the error stream
    sys.exit(1)  # ends the program with status 1, which signals failure to the shell


def fmt_utc(dt: datetime | None) -> str:  # function: formats an instant as UTC text; the hint says dt may be None
    # conditional expression: "" for a missing instant, else convert to UTC and format as YYYY-MM-DD HH:MM:SS
    return "" if dt is None else dt.astimezone(TZ_UTC).strftime("%Y-%m-%d %H:%M:%S")


def fmt_no(dt: datetime | None) -> str:  # function: formats an instant as Norwegian local text, for display columns
    # same pattern, converting to Europe/Oslo instead, so the result reads against the transmission plan
    return "" if dt is None else dt.astimezone(TZ_OSLO).strftime("%Y-%m-%d %H:%M:%S")


def minutes(td: timedelta) -> float:  # function: turns a duration into minutes
    return round(td.total_seconds() / 60.0, 2)  # total_seconds() gives seconds; / 60 gives minutes; round to 2 places


# function: minutes of overlap between window a0-a1 and window b0-b1
def overlap_minutes(a0: datetime, a1: datetime, b0: datetime, b1: datetime) -> float:
    lo, hi = max(a0, b0), min(a1, b1)  # tuple assignment: overlap starts at the later start and ends at the earlier end
    # max(0.0, ...) turns a negative gap (windows not touching) into zero overlap; result in minutes to 2 places
    return round(max(0.0, (hi - lo).total_seconds()) / 60.0, 2)


# --------------------------------------------------------------------------
# 1. Session discovery
# --------------------------------------------------------------------------

def discover_sessions() -> list[dict]:  # function: returns one dict per capture-session folder
    """Find the capture-session folders by pattern, never by hardcoded name."""
    if not DAY.is_dir():  # if: the capture-day folder is missing or is not a folder
        die(f"capture day folder not found: {DAY}")  # f-string inserts the path; the script stops here
    found = []  # empty list that will collect the matching session folders
    for p in sorted(DAY.iterdir()):  # for loop over the folder's entries, sorted by name so the order is fixed
        if not p.is_dir():  # if: this entry is a file, not a folder
            continue  # continue skips to the next entry
        m = SESSION_DIR_RE.match(p.name)  # tries the folder-name pattern; gives a match object, or None if no match
        if m:  # if: the name matched, so this is a session folder (the phone folders do not match)
            # dict literal appended to the list: the path, the folder name and the three groups read from the name
            found.append({"path": p, "folder": p.name,
                          "folder_date": m.group(1), "folder_time": m.group(2),  # dict entry: the capture date and start time read from the session folder's name
                          "test_code": m.group(3)})  # dict entry: the test code read from the folder name, which ties the capture to its scheduled test
    if len(found) != 6:  # if: anything other than exactly six sessions is an error, never a silent skip
        # stops the script; the message (f-string + joined list, concatenated) names every folder that was found
        die(f"expected 6 session folders matching <date>_<time>_T_<code>, found {len(found)}: "
            + ", ".join(f["folder"] for f in found))  # str.join over a generator lists the folders that were found
    codes = [f["test_code"] for f in found]  # list comprehension: the test code of every session
    if len(set(codes)) != len(codes):  # if: a set drops duplicates, so a shorter set means a code appears twice
        die(f"duplicate test codes among session folders: {codes}")  # two folders claiming one test is fatal
    return found  # hands back the six session dicts


# --------------------------------------------------------------------------
# 2. README / event-log / gap-log / timestamp-CSV parsing
# --------------------------------------------------------------------------

def split_readme_sections(text: str) -> dict[str, list[str]]:  # function: README text -> {heading: lines under it}
    """
    Split a README into named sections.

    A heading is any line immediately followed by a rule of '-' or '='.
    This is section-aware rather than line-number based, because the six
    READMEs differ in remark length and sensor-table width.
    """
    lines = text.splitlines()  # str.splitlines: the README as a list of lines without line breaks
    heads = []  # empty list that will hold (line index, heading text) pairs
    for i in range(len(lines) - 1):  # for loop over every line index except the last, so i + 1 always exists
        nxt = lines[i + 1].strip()  # the following line, spaces removed
        # if: the next line is a rule (3+ characters, all '-' or '='; set <= is a subset test) and this line has text
        if len(nxt) >= 3 and set(nxt) <= {"-", "="} and lines[i].strip():
            heads.append((i, lines[i].strip()))  # appends a tuple: where the heading is and what it says
    sections: dict[str, list[str]] = {}  # annotated empty dict: heading text -> the lines under it
    for k, (idx, name) in enumerate(heads):  # enumerate with tuple unpacking: k counts headings; idx and name unpacked
        # conditional expression: a section ends where the next heading starts, or at the end of the file for the last
        end = heads[k + 1][0] if k + 1 < len(heads) else len(lines)
        sections[name] = lines[idx + 2:end]  # slice: skips the heading and its rule, keeps lines up to the next heading
    return sections  # hands back the dict of named sections


def parse_sensor_table(block: list[str]) -> list[dict]:  # function: turns the README's pipe table into row dicts
    """Parse the pipe-delimited sensor table from the README."""
    header, rows = None, []  # tuple assignment: no header yet, no rows yet
    for ln in block:  # for loop over the lines of the sensor section, which ends at the next README heading
        if "|" not in ln:  # if: the line has no pipe, so it is not a table row (the "----+----" rule has none either)
            continue  # skips this non-table line
        cells = [c.strip() for c in ln.split("|")]  # list comprehension: splits the row at each pipe, trims each cell
        if header is None:  # if: the first table line is the header
            header = cells  # keeps the column names
            continue  # the header is not a data row
        if len(cells) != len(header):  # if: the row has a different number of cells from the header
            # records the malformed row instead of guessing which cell belongs where
            flag(f"sensor table row has {len(cells)} cells against a {len(header)}-cell header: "
                 f"{ln.strip()[:70]}")  # f-string continued: the first 70 characters of the malformed row, so it can be found in the README
            continue  # the malformed row is skipped, not repaired
        rows.append(dict(zip(header, cells)))  # zip pairs column names with cells; dict() makes one row dict
    if header is None:  # if: no table was found at all
        die("no sensor table found in README")  # a README without its instrument table is fatal
    return rows  # hands back one dict per instrument


def parse_readme(path: Path) -> dict:  # function: reads one session README into the fields the script needs
    raw = path.read_text(encoding="utf-8")  # the whole README as one string
    sec = split_readme_sections(raw)  # splits it into named sections by heading

    # next() over a generator: the first heading containing "Capture Metadata", or None if no heading does
    head_key = next((k for k in sec if "Capture Metadata" in k), None)
    if head_key is None:  # if: no such heading
        die(f"{path}: no capture-metadata heading")  # without it there is no start or end time, so stop
    head = "\n".join(sec[head_key])  # str.join: glues the section's lines back into one block of text

    def field(name: str) -> str:  # nested function: finds "Name: value" inside the metadata block
        # rf-string builds the pattern; ^ and $ match at each line because of re.MULTILINE; the group is the value
        m = re.search(rf"^{name}:\s*(.+)$", head, re.MULTILINE)
        if not m:  # if: the field is absent
            die(f"{path}: README field '{name}' not found")  # a missing Start, End or Duration is fatal
        return m.group(1).strip()  # the captured value without surrounding spaces

    # tuple assignment: the three raw text fields exactly as written, e.g. "2026-09-14T15:27:22+03:00"
    start_raw, end_raw, dur_raw = field("Start"), field("End"), field("Duration")

    # next() over a generator: the heading that starts with "sensors detected", compared in lower case
    sensors_key = next((k for k in sec if k.lower().startswith("sensors detected")), None)
    if sensors_key is None:  # if: no sensor section
        die(f"{path}: no sensor-table section")  # stop rather than continue without an instrument list
    sensors = parse_sensor_table(sec[sensors_key])  # parses the instrument table

    # next() over a generator: the heading that starts with "user remarks", or None
    remarks_key = next((k for k in sec if k.lower().startswith("user remarks")), None)
    # join over a generator: the non-blank remark lines, trimmed and joined with spaces into one verbatim string
    remarks = " ".join(x.strip() for x in sec.get(remarks_key, []) if x.strip())

    gap_key = next((k for k in sec if "continuity" in k.lower()), None)  # the stream-continuity section's heading
    gap_thresh = ""  # default: threshold unknown
    if gap_key:  # if: a continuity section exists (a non-empty string is truthy)
        m = re.search(r"Gap threshold:\s*(.+)", "\n".join(sec[gap_key]))  # finds the "Gap threshold:" line
        gap_thresh = m.group(1).strip() if m else ""  # conditional expression: the value if found, else empty

    # returns a dict of the raw times, the instrument rows, the remarks and the gap threshold
    return {"start_raw": start_raw, "end_raw": end_raw, "duration_raw": dur_raw,
            "sensors": sensors, "remarks": remarks, "gap_threshold": gap_thresh}  # dict entry: the sensor rows, the operator remarks and the logger's stream-gap threshold


def parse_events(path: Path) -> dict:  # function: reads session_events.log and sorts events into kinds
    # list comprehension: every non-blank line of the event log (the variable l is one line)
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    evs = []  # empty list that collects parsed events
    for ln in lines:  # for loop over the log lines
        m = re.match(r"^(\S+)\s\s+(.*)$", ln)  # regex: a timestamp with no spaces, two or more spaces, then the text
        if not m:  # if: the line does not have that shape
            flag(f"{path.parent.name}: unparsed session_events line: {ln[:70]}")  # records it rather than dropping it
            continue  # moves on to the next line
        evs.append({"ts_raw": m.group(1), "text": m.group(2).strip()})  # dict: the raw timestamp and the event text
    start = next((e for e in evs if e["text"] == "START"), None)  # next() over a generator: the first START event
    stop = next((e for e in evs if e["text"] == "STOP"), None)  # likewise the first STOP event, or None
    other = [e for e in evs if e["text"] not in ("START", "STOP")]  # list comprehension: every other event
    # Any OBD-II "READY ELM=" line, at any time, is the adapter's handshake and is routine; each session's
    # session_events.log holds exactly one, at capture start. Everything else is reported.
    # list comprehension: the events whose text contains "READY ELM=" (\b = word boundary)
    routine = [e for e in other if re.search(r"\bREADY ELM=", e["text"])]
    problem = [e for e in other if e not in routine]  # list comprehension: everything not routine is reported
    # returns a dict with the line count, all events, START, STOP and the three groupings
    return {"n_lines": len(lines), "events": evs, "start": start, "stop": stop,
            "other": other, "routine": routine, "problem": problem}  # dict entry: the remaining events split into other, routine and problem lines


def parse_timestamp_csv(path: Path) -> dict:  # function: first and last rows of the 1 Hz system-clock log
    """First and last 1 Hz system-clock samples - an independent duration witness."""
    if not path.exists():  # if: the log is missing
        return {"n_rows": 0, "first": "", "last": ""}  # returns an empty result instead of stopping
    with path.open("r", encoding="utf-8") as fh:  # with block: opens the file and closes it automatically afterwards
        header = fh.readline().rstrip("\n").split(",")  # reads the first line and splits it into column names
        first = last = None  # chained assignment: both start as None
        n = 0  # counter of data rows
        for line in fh:  # for loop over the remaining lines, read one at a time without loading the whole file
            if not line.strip():  # if: a blank line
                continue  # skip it
            n += 1  # augmented assignment: one more data row
            if first is None:  # if: this is the first data row
                first = line.rstrip("\n").split(",")  # keeps it as a list of cells
            last = line.rstrip("\n").split(",")  # overwritten every time, so after the loop it holds the last row
    if first is None:  # if: the file had a header but no data rows
        return {"n_rows": 0, "first": "", "last": ""}  # empty result
    col = header.index("system_time_iso8601")  # list.index: the position of the timestamp column
    return {"n_rows": n, "first": first[col], "last": last[col]}  # row count and the first and last timestamps


# --------------------------------------------------------------------------
# 3. Test catalogue
# --------------------------------------------------------------------------

# set of catalogue group names (all of type "Supplemental periods") whose 0 W means "transmits nothing";
# the fourth supplemental group, "Ad hoc tests", is left out on purpose: an ad hoc test may transmit
NON_TRANSMITTING = {"Mandatory briefings", "Grace period", "Booking slots"}


# function: the one power rule for a scheduled entry p and its catalogue entry cat (None if uncatalogued); returns
# (power in W or None, source, field). Used by main() below and, through codes/common/helpers.py, by the handset script.
def effective_power(p: dict, cat: dict | None) -> tuple[float | None, str, str]:
    if p["power_w_plan"] is not None:  # if: the plan states a power
        return p["power_w_plan"], "plan", "plan_power_line"  # the plan's value is used, with its source and field
    if cat and cat["max_power_w"] is not None:  # if: otherwise the catalogue's maximum, if it has one
        return cat["max_power_w"], "catalog", "max_power_w"  # the catalogue maximum (the worst case) is used
    # if: a catalogue 0 W in a supplemental period that transmits nothing: briefings,
    # grace periods and booking slots are non-transmitting, so their 0 W is a real zero
    if cat and cat["max_power_zero"] and cat["type"] == "Supplemental periods" and cat["group_name"] in NON_TRANSMITTING:
        return 0.0, "catalog: non-transmitting period", "max_power_w"  # a known power of zero
    if cat and cat["max_power_zero"]:  # if: any other catalogue 0 W (ad hoc tests, jammers, spoofers): "not stated"
        return None, "catalog: not stated", "max_power_w"  # unknown (an empty CSV cell), never 0.0
    return None, "not_stated", ""  # neither source gives a power: unknown, kept as None rather than a guess


def load_catalog() -> dict[str, dict]:  # function: returns {test code: that test's catalogue fields}
    doc = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))  # json.loads parses the file's text into dicts and lists
    out: dict[str, dict] = {}  # annotated empty dict that will map each test code to its fields
    for tt in doc["test_types"]:  # outer for loop: each test type (e.g. Jamming, Spoofing)
        for grp in tt["test_groups"]:  # middle loop: each group within the type, e.g. group 2.6
            for te in grp["tests"]:  # inner loop: each individual test in the group
                code = te["test_code"]  # the test's code, e.g. "2.6.5"
                if code in out:  # if: the code was already seen, which would make lookups ambiguous
                    die(f"test code {code} appears more than once in testcatalog.json")  # fatal: stop
                out[code] = {  # dict literal stored under the code; the entries below flatten three catalogue levels
                    "test_code": code,  # dict entry: the test code, the key that joins the catalogue to the sessions
                    "test_name": te.get("name", ""),  # dict.get with default: "" if the test has no name
                    # (x or ""): falls back to "" when the value is None, so .strip() always has a string to work on
                    "test_description": (te.get("description") or "").strip(),
                    "type": tt.get("type", ""),  # dict entry: the interference type of the test (jamming, spoofing or meaconing)
                    "group_name": grp.get("group_name", ""),  # dict entry: the name of the catalogue group the test belongs to
                    "group_description": (grp.get("description") or "").strip(),  # dict entry: the group's description, the text the test phases are read from later
                    "group_rationale": (grp.get("rationale") or "").strip(),  # dict entry: the group's stated rationale, kept as context for the test
                    # (list or []) guards against None; ";".join turns the band list into one text cell
                    "bands": ";".join(te.get("constellation_bands") or []),
                    "equipment": ";".join(te.get("equipment") or []),  # dict entry: the transmitting equipment, joined into one text cell
                    # conditional expression: the power as a float, or None when the catalogue leaves it empty
                    "min_power_w": float(te["min_power_w"]) if te.get("min_power_w") not in (None, "") else None,
                    # the maximum as a float, or None when empty or stated as exactly 0 W, which means "not stated"
                    "max_power_w": (float(te["max_power_w"])
                                    # continuation: the condition, present and non-zero
                                    if te.get("max_power_w") not in (None, "") and float(te["max_power_w"]) != 0.0
                                    # continuation: otherwise None (not stated)
                                    else None),
                    # boolean: True when the catalogue writes 0 W, kept so the "not stated" is attributed to the catalogue
                    "max_power_zero": te.get("max_power_w") not in (None, "") and float(te["max_power_w"]) == 0.0,
                }
    return out  # hands back the lookup table


# --------------------------------------------------------------------------
# 4. Transmission plan PDF
# --------------------------------------------------------------------------

def pdf_page_chunks(page) -> list[tuple[float, float, str]]:  # function: every text piece on a PDF page with its place
    """Return (x, y_from_top, text) for every text chunk on the page."""
    height = float(page.mediabox.height)  # page height in PDF points, needed to measure y from the top
    out: list[tuple[float, float, str]] = []  # annotated empty list of (x, y, text) tuples

    def visitor(text, cm, tm, font_dict, font_size):  # nested callback: pypdf calls it once per text piece it draws
        t = text.strip()  # the piece's text without surrounding spaces
        if t:  # if: skip empty and whitespace-only pieces
            # appends a tuple: x = tm[4], y = page height minus tm[5] (so y counts from the top), both to 0.1 pt, text
            # ignoring cm is valid because this PDF draws all retained text with the identity transform (verified)
            out.append((round(float(tm[4]), 1), round(height - float(tm[5]), 1), t))

    page.extract_text(visitor_text=visitor)  # runs pypdf's text extraction, handing each piece to visitor
    return out  # hands back the list the visitor filled


# function: groups one column's text pieces into visual lines; tol is the y-distance (points) counted as one line
def group_into_lines(chunks, tol: float = 2.0) -> list[tuple[float, float, str]]:
    """
    Cluster chunks into visual lines by y, returning (x_of_first, y, joined_text).

    Callers pass the chunks of ONE test-area column only. Grouping across the
    whole page would merge three areas' entries, which sit at the same y, into
    a single unreadable line.
    """
    lines: list[dict] = []  # annotated empty list; each line is a dict with its first y and all member ys
    for y in sorted({c[1] for c in chunks}):  # set comprehension removes repeated ys; sorted walks them top to bottom
        placed = False  # flag: has this y joined an existing line yet?
        for ln in lines:  # for loop over the lines found so far
            if abs(ln["y"] - y) <= tol:  # if: within tol points of that line's FIRST y (not a running average)
                ln["ys"].append(y)  # this y joins the line
                placed = True  # remembers that it was placed
                break  # stops at the first line that fits
        if not placed:  # if: no existing line was close enough
            lines.append({"y": y, "ys": [y]})  # starts a new line at this y
    out = []  # empty list for the finished (x, y, text) lines
    for ln in lines:  # for loop over the grouped lines
        # list comprehension picks the pieces on this line; sorted with key=lambda orders them left to right by x
        members = sorted([c for c in chunks if c[1] in ln["ys"]], key=lambda c: c[0])
        if not members:  # if: no pieces (cannot normally happen, guarded anyway)
            continue  # skip it
        # joins the pieces' text with spaces, then re.sub collapses any run of whitespace to a single space
        text = re.sub(r"\s+", " ", " ".join(m[2] for m in members)).strip()
        out.append((members[0][0], min(ln["ys"]), text))  # tuple: x of the leftmost piece, the line's top y, the text
    return sorted(out, key=lambda r: (r[1], r[0]))  # sorted by a tuple key: top to bottom, then left to right


def parse_transmission_plan() -> tuple[list[dict], dict]:  # function: returns the day's scheduled entries and geometry
    """
    Parse every scheduled test on the target Monday, across all three test areas.

    Column assignment is geometric: each test area's entries begin at a fixed
    x-origin that is identical on every page of the day. That is asserted, not
    assumed, so a silent mis-assignment cannot pass unnoticed.
    """
    reader = pypdf.PdfReader(str(PLAN_PATH))  # opens the plan PDF; str() hands the reader a plain path string
    day_pages = []  # empty list for (page index, text pieces) of the pages belonging to 14 September
    for i, page in enumerate(reader.pages):  # enumerate: each page together with its 0-based index i
        chunks = pdf_page_chunks(page)  # every positioned text piece on the page
        joined = " ".join(c[2] for c in chunks)  # join over a generator: all the page's text as one string
        if PLAN_DAY_HEADER in joined and TEST_DAY in joined:  # if: both "Monday" and "2026-09-14" appear on the page
            day_pages.append((i, chunks))  # keeps the page
    if not day_pages:  # if: no page matched (an empty list is falsy)
        die(f"no page of the transmission plan carries the header '{PLAN_DAY_HEADER} - {TEST_DAY}'")  # fatal

    # Derive the three column x-origins from the entry headers themselves, using
    # raw chunks so that the derivation does not depend on line grouping.
    time_range = re.compile(r"^\d{2}:\d{2}-\d{2}:\d{2}$")  # compiled regex for a piece that is exactly "HH:MM-HH:MM"
    # set comprehension, two loops (pages, then pieces): distinct x where a time range starts; sorted left to right
    origins = sorted({x for _, chunks in day_pages for x, y, t in chunks if time_range.match(t)})
    if len(origins) != 3:  # if: not exactly three distinct x positions, one per test area, on all the day's pages
        # stops the script; two adjacent f-strings are joined into one message listing the x positions found
        die(f"expected 3 test-area column origins on the {PLAN_DAY_HEADER} pages, "
            f"found {len(origins)}: {origins}")  # stops rather than guess the columns
    # Each column runs from its own x-origin up to the NEXT column's x-origin.
    # The midpoint between origins would be wrong: a test area's body text is
    # indented under its entry header and runs most of the column's width, well
    # past the midpoint, so midpoint cuts file the tail of a long line under the
    # following test area.
    # list of three (low, high) x-ranges; the first starts 1 point early, the last runs to 1e9, i.e. the page edge
    bounds = [(origins[0] - 1.0, origins[1]), (origins[1], origins[2]), (origins[2], 1e9)]

    # dict recording how the columns were found; list comprehension keeps only the page numbers
    geometry = {"origins": origins, "bounds": bounds, "pages": [p for p, _ in day_pages]}

    entries: list[dict] = []  # annotated empty list that collects every parsed entry
    # dict: for each test area 1-3, the entry currently being read; it survives page breaks, so entries continue
    current: dict[int, dict | None] = {1: None, 2: None, 3: None}

    for page_no, chunks in day_pages:  # for loop over the day's pages
        for area, (lo, hi) in enumerate(bounds, start=1):  # enumerate from 1: area number with its x-range unpacked
            col = [c for c in chunks if lo <= c[0] < hi]  # comprehension, chained comparison: this column only
            for x, y, text in group_into_lines(col):  # for loop over the column's visual lines, top to bottom
                # if: page furniture (page numbers, column titles, the "Time" label) is not part of any entry
                if text.startswith("Page ") or text.startswith("Test Area") or text == "Time":
                    continue  # skip it
                if text.startswith(PLAN_DAY_HEADER):  # if: the "Monday ..." day header
                    continue  # skip it
                m = ENTRY_RE.match(text)  # tries the entry-header pattern "HH:MM-HH:MM - code"
                if m:  # if: this line opens a new entry
                    # dict literal for the new entry: place, window, code, empty text lists, power, title/comment flags
                    e = {"test_area": area, "page": page_no,
                         "start_hhmm": m.group(1), "end_hhmm": m.group(2),  # dict entry: the start and end clock times read from the entry's first line
                         "test_code": m.group(3), "title_parts": [], "comment_parts": [],  # dict entry: the test code, with empty lists that collect the title and comment lines below
                         "power_w_plan": None, "in_title": True, "in_comment": False}  # dict entry: no power read yet, and the parser starts inside the title, not the comment
                    entries.append(e)  # stores it
                    current[area] = e  # makes it the column's current entry
                    continue  # the header line is done
                e = current[area]  # otherwise the line belongs to the column's current entry
                if e is None:  # if: None means no entry has opened in this column yet
                    # continue: skips this line, which sits above the column's first entry header
                    continue                      # text before the column's first entry
                if text.startswith("Ongoing event"):  # if: a continuation banner, not content
                    continue  # skip it
                pm = POWER_RE.match(text)  # tries the "Power: N W" pattern
                if pm:  # if: this is the power line
                    e["power_w_plan"] = float(pm.group(1))  # stores the stated power as a number
                    e["in_title"] = e["in_comment"] = False  # chained assignment: title and comment have both ended
                    continue  # the power line is done
                if text.startswith("Contact:"):  # if: the contact line, which ends the title
                    e["in_title"] = e["in_comment"] = False  # stop collecting title or comment text
                    continue  # the contact line is not kept
                if text.startswith("Comment:"):  # if: the start of the entry's free-text comment
                    e["in_title"], e["in_comment"] = False, True  # tuple assignment: leave the title, enter the comment
                    # slice from len("Comment:") drops the label; the rest of the line is the comment's first part
                    e["comment_parts"].append(text[len("Comment:"):].strip())
                    continue  # this line is done
                if e["in_title"]:  # if: still inside the title
                    e["title_parts"].append(text)  # the line is part of the title
                elif e["in_comment"]:  # elif: inside the comment
                    e["comment_parts"].append(text)  # the line is part of the comment

    # Parse assertions. A boundary set too wide shows up as a line carrying a
    # second entry header, or as an entry whose title was taken by its neighbour.
    for e in entries:  # for loop over every parsed entry
        joined = " ".join(e["title_parts"])  # its title as one string
        if re.search(r"\d{2}:\d{2}-\d{2}:\d{2}\s*-\s*\d+\.\d+", joined):  # if: another entry header hides in the title
            # !r in the f-string prints the text with quotes, so stray spaces are visible; the script stops
            die(f"entry {e['test_code']} (Test Area {e['test_area']}) absorbed a second entry "
                f"header into its title: {joined[:100]!r}; column boundaries are wrong")  # f-string continued: the first 100 characters of the joined title, quoted, to show where parsing broke
        if not e["title_parts"]:  # if: the entry has no title at all
            # stops the script, naming the entry; an f-string and a plain string are joined into one message
            die(f"entry {e['test_code']} (Test Area {e['test_area']}) at {e['start_hhmm']} "
                "has an empty title; the plan did not parse as expected")  # the other failure mode; stop

    day = datetime.strptime(TEST_DAY, "%Y-%m-%d").date()  # strptime parses "2026-09-14"; .date() keeps only the date
    out = []  # empty list for the finished entries
    for e in entries:  # for loop over the parsed entries
        sh, sm = (int(v) for v in e["start_hhmm"].split(":"))  # generator unpacked into two ints: start hour, minute
        eh, em = (int(v) for v in e["end_hhmm"].split(":"))  # the same for the end
        # datetime.combine joins the date with midnight; .replace sets the hour and minute; no time zone yet (naive)
        start_naive = datetime.combine(day, datetime.min.time()).replace(hour=sh, minute=sm)
        end_naive = datetime.combine(day, datetime.min.time()).replace(hour=eh, minute=em)  # the same for the end
        if end_naive <= start_naive:  # if: the window ends at or before it starts
            # records it; the entry is not silently moved to the next day
            flag(f"transmission plan entry {e['test_code']} at Test Area {e['test_area']} "
                 f"ends ({e['end_hhmm']}) at or before it starts ({e['start_hhmm']}); "  # f-string continued: the end and start times that make the entry run backwards
                 f"not treated as crossing midnight")
        # appends a clean dict; re.sub collapses whitespace in the joined title and comment
        out.append({"test_area": e["test_area"], "test_code": e["test_code"],
                    "title": re.sub(r"\s+", " ", " ".join(e["title_parts"])).strip(),  # dict entry: the entry's title with its broken lines joined and white space collapsed
                    "start_hhmm": e["start_hhmm"], "end_hhmm": e["end_hhmm"],  # dict entry: the start and end clock times exactly as the plan prints them
                    "start_naive": start_naive, "end_naive": end_naive,  # dict entry: start and end as date-times on the test day, with no time zone attached yet
                    "power_w_plan": e["power_w_plan"],  # dict entry: the transmission power the plan states for the entry, if any
                    "comment": re.sub(r"\s+", " ", " ".join(e["comment_parts"])).strip()})  # dict entry: the entry's comment with its lines joined and white space collapsed
    # The plan's own free-text comments sometimes quote a clock time. Where they
    # do, check it against the tabled window; a whole-hour disagreement means the
    # document is internally inconsistent about its own time base.
    for r in out:  # for loop over the finished entries
        # re.findall returns every (hour, minute) pair written as H:MM, HH:MM or with a dot, e.g. "14.28"
        for hh, mm in re.findall(r"\b(\d{1,2})[:.](\d{2})\b", r["comment"]):
            quoted = int(hh) * 60 + int(mm)  # the quoted time as minutes after midnight
            # list of two values: the tabled start and end, also as minutes after midnight
            tabled = [r["start_naive"].hour * 60 + r["start_naive"].minute,
                      r["end_naive"].hour * 60 + r["end_naive"].minute]  # list element: the tabled end time in minutes after midnight, to compare with the comment's time
            diffs = [quoted - t for t in tabled]  # list comprehension: quoted minus start, quoted minus end
            # list comprehension: differences that are a non-zero whole number of hours, at most 3 h either way
            whole = [d for d in diffs if d != 0 and d % 60 == 0 and abs(d) <= 180]
            if whole and not any(d == 0 for d in diffs):  # if: a whole-hour gap and no exact match with either end
                # records the inconsistency; // is integer division, turning minutes into whole hours
                flag(f"transmission plan is internally inconsistent for {r['test_code']} "
                     f"(Test Area {r['test_area']}): tabled {r['start_hhmm']}-{r['end_hhmm']} "  # f-string continued: the test area and the start and end times the table gives
                     f"but its own comment quotes {hh}:{mm}, {abs(whole[0]) // 60} h apart")  # f-string continued: the time the comment quotes and how many whole hours separate the two

    out.sort(key=lambda r: (r["start_naive"], r["test_area"], r["test_code"]))  # in-place sort: time, area, code
    if not out:  # if: nothing was parsed
        die("parsed zero scheduled entries from the transmission plan")  # an empty schedule is fatal
    return out, geometry  # returns a tuple: the entries and the column geometry


# --------------------------------------------------------------------------
# 5. Time-base derivation
# --------------------------------------------------------------------------

# list of four tuples, one per hypothesis: tag, description, how to read the README times, and the plan's time zone
HYPOTHESES = [
    ("H1", "README ISO offset correct; plan times Norwegian local", "as_is", TZ_OSLO),  # list element: README offset correct and plan times in Norwegian local time
    ("H2", "README ISO offset correct; plan times UTC", "as_is", TZ_UTC),  # list element: README offset correct and plan times in UTC
    # list element: README digits are Norwegian wall-clock time and plan times are Norwegian local
    ("H3", "README digits are Norwegian wall clock; plan times Norwegian local", "digits_as_oslo", TZ_OSLO),
    ("H4", "README digits are Norwegian wall clock; plan times UTC", "digits_as_oslo", TZ_UTC),  # list element: README digits are Norwegian wall-clock time and plan times are UTC
]


def readme_instant(raw: str, mode: str) -> datetime:  # function: a README time string -> a UTC instant, per reading
    dt = datetime.fromisoformat(raw)  # parses ISO text such as "2026-09-14T15:27:22+03:00", offset included
    if mode == "as_is":  # if: the first reading, the +03:00 offset is taken as correct
        return dt.astimezone(TZ_UTC)  # converts to UTC, honouring the stated offset (15:27+03:00 -> 12:27 UTC)
    if mode == "digits_as_oslo":  # if: the rival reading, the digits are Norwegian wall-clock time
        # drops the offset (tzinfo=None), attaches Oslo instead, then converts: 15:27 Oslo -> 13:27 UTC
        return dt.replace(tzinfo=None).replace(tzinfo=TZ_OSLO).astimezone(TZ_UTC)
    raise ValueError(mode)  # raise: an unknown mode is a programming error, so an exception is thrown


# function: scores the four hypotheses and returns the winner plus the full score table as a DataFrame
def derive_time_base(sessions: list[dict], plan: list[dict]) -> tuple[dict, pd.DataFrame]:
    sched_by_code: dict[str, dict] = {}  # annotated empty dict: test code -> its Test Area 3 plan entry
    for p in plan:  # for loop over every plan entry
        if p["test_area"] == 3:  # if: only Test Area 3, where our six tests were scheduled
            sched_by_code.setdefault(p["test_code"], p)  # dict.setdefault: stores the first entry per code only

    rows = []  # empty list; one score row per hypothesis
    for tag, label, rmode, ptz in HYPOTHESES:  # for loop unpacking each hypothesis tuple into four names
        total, zero = 0.0, 0  # tuple assignment: total overlap minutes, and sessions with no overlap
        per = {}  # empty dict: test code -> overlap under this hypothesis
        for s in sessions:  # for loop over the six sessions
            code = s["test_code"]  # this session's test code
            if code not in sched_by_code:  # if: the plan has no Test Area 3 entry for this test
                die(f"test code {code} has no Test Area 3 entry in the transmission plan")  # cannot score it; stop
            cs = readme_instant(s["readme"]["start_raw"], rmode)  # capture start in UTC under this reading
            ce = readme_instant(s["readme"]["end_raw"], rmode)  # capture end in UTC under this reading
            # scheduled start: attach the hypothesised plan time zone to the naive time, then convert to UTC
            ss = sched_by_code[code]["start_naive"].replace(tzinfo=ptz).astimezone(TZ_UTC)
            se = sched_by_code[code]["end_naive"].replace(tzinfo=ptz).astimezone(TZ_UTC)  # scheduled end, likewise
            ov = overlap_minutes(cs, ce, ss, se)  # minutes the capture overlaps its own scheduled test
            per[code] = ov  # stores it per test
            total += ov  # adds to the hypothesis total
            if ov <= 0:  # if: no overlap at all
                zero += 1  # counts a session that misses its own test
        # appends the score row; "**{...}" unpacks a dict comprehension into extra columns, one per test code
        rows.append({"hypothesis": tag, "description": label,
                     "readme_interpretation": rmode,  # dict entry: how the README times are read under this hypothesis
                     # conditional expression: "is" compares identity with the Oslo zone object to name the zone
                     "plan_timezone": "Europe/Oslo" if ptz is TZ_OSLO else "UTC",
                     "sessions_with_zero_overlap": zero,  # dict entry: sessions whose capture does not overlap its scheduled test at all
                     "total_overlap_min": round(total, 2),  # dict entry: total minutes of capture-to-schedule overlap, the score that ranks the hypotheses
                     **{f"overlap_{c}_min": v for c, v in sorted(per.items())}})  # dict unpacking: adds one overlap column per test code, so each session's share of the score is visible

    df = pd.DataFrame(rows)  # builds a table from the list of row dicts, one column per key
    # sorted with a tuple key: fewest zero-overlap sessions first, then largest total (negated to sort descending)
    ranked = sorted(rows, key=lambda r: (r["sessions_with_zero_overlap"], -r["total_overlap_min"]))
    best, runner = ranked[0], ranked[1]  # tuple assignment: the winner and the runner-up

    if best["sessions_with_zero_overlap"] != 0:  # if: even the winner leaves a session outside its own test
        # stops the script: no reading places every session on its test, so labels cannot be made
        die("no time-base hypothesis gives every session a non-zero overlap with its own "
            "scheduled window; the time base is unresolved and labelling cannot proceed")  # stop
    # if: the runner-up scored something and the winner does not at least double it, the choice is not decisive
    if runner["total_overlap_min"] > 0 and best["total_overlap_min"] < 2.0 * runner["total_overlap_min"]:
        # stops the script, quoting both totals; three adjacent strings are joined into one message
        die(f"time-base hypotheses {best['hypothesis']} and {runner['hypothesis']} are not "
            f"separated decisively ({best['total_overlap_min']} vs {runner['total_overlap_min']} "  # f-string continued: the two overlap totals that are too close to separate the hypotheses
            "overlap minutes); refusing to choose between them")  # stop rather than pick on a narrow margin

    chosen = next(h for h in HYPOTHESES if h[0] == best["hypothesis"])  # next() over a generator: the winning tuple
    # returns a tuple of (dict describing the chosen base, with the schedule lookup, and the full score table)
    return {"tag": chosen[0], "label": chosen[1], "readme_mode": chosen[2],
            "plan_tz": chosen[3], "best": best, "runner": runner,  # dict entry: the plan's time zone and the best and runner-up score rows
            "sched_by_code": sched_by_code}, df  # dict entry: the scheduled window of each test code under the chosen time base


def check_operator_remark(sessions: list[dict], base: dict) -> dict:  # function: compares the 2.6.2 remark to the base
    """
    Cross-check the derived offset against the 2.6.2 operator remark, which
    independently states a Norwegian local start time.
    """
    hit = None  # no matching remark found yet
    for s in sessions:  # for loop over the sessions' operator remarks
        # re.search for the operator's phrase; the group captures a time like "14:56" or "14.56", case-insensitive
        m = re.search(r"started the capture early around\s*(\d{1,2}[:.]\d{2})\s*Norway time",
                      s["readme"]["remarks"], re.IGNORECASE)  # argument of re.search: the README remarks searched, ignoring letter case
        if m:  # if: this session's remark contains the phrase
            hit = (s, m.group(1).replace(".", ":"), m.group(0))  # tuple: session, time with ":" and the full sentence
            break  # stops at the first match
    if hit is None:  # if: no session's remark states a local time
        # returns a result that says so, rather than a pass or a fail
        return {"found": False, "agrees": None,
                "detail": "no operator remark stating a Norwegian local start time was found"}  # dict entry: the reason no verdict could be given
    s, hhmm, sentence = hit  # tuple unpacking of the match
    # the capture start under the chosen reading, converted to Norwegian local time for comparison with the remark
    derived_no = readme_instant(s["readme"]["start_raw"], base["readme_mode"]).astimezone(TZ_OSLO)
    h, mnt = (int(v) for v in hhmm.split(":"))  # generator unpacked into the stated hour and minute
    stated = derived_no.replace(hour=h, minute=mnt, second=0, microsecond=0)  # the stated time on the same date
    delta_min = abs((derived_no - stated).total_seconds()) / 60.0  # absolute gap between the two, in minutes
    # The remark says "around" a time given to the minute, so the test is whether
    # the derived value falls within a couple of minutes, not to the second.
    agrees = delta_min <= 2.0  # boolean: True if within two minutes
    # returns the verdict with the stated and derived times and a readable one-line summary
    return {"found": True, "agrees": agrees, "test_code": s["test_code"],
            "stated_no": hhmm, "derived_no": derived_no.strftime("%H:%M:%S"),  # dict entry: the start time the remark states and the one derived from the README, both Norwegian local
            "delta_min": round(delta_min, 2), "sentence": sentence,  # dict entry: the difference in minutes and the remark's sentence that was matched
            "detail": (f"remark states {hhmm} Norwegian local; derived value is "  # dict entry: a readable one-line verdict built from the stated and derived times
                       f"{derived_no.strftime('%H:%M:%S')} ({delta_min:.2f} min apart)")}  # f-string continued: the derived local time and how many minutes it lies from the stated one


# --------------------------------------------------------------------------
# 6. Phone captures - inventory and fourth time witness
# --------------------------------------------------------------------------

# compiled regex with named groups (?P<name>...) for GnssLogger file names: date, time, brand, model, SDK, session
PHONE_NAME_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})_(?P<brand>[^_]+)_(?P<model>[^_]+)_"
    r"(?P<sdk>\d+)_(?P<session>\d+)_")

PHONE_HEAD_BYTES = 4096  # integer constant: only the first 4 KiB of each phone JSON is read, where its header sits


def phone_header_fields(path: Path) -> dict:  # function: pulls named header fields out of a phone JSON file
    # opens in binary mode ("rb"), reads 4096 bytes, decodes as UTF-8 with "replace" so a cut character cannot crash
    head = path.open("rb").read(PHONE_HEAD_BYTES).decode("utf-8", "replace")
    out = {}  # empty dict for the fields found
    # for loop over a tuple of the JSON keys wanted; the header is searched as text, not parsed as JSON
    for key in ("sessionId", "sequence", "events", "durationInSeconds", "startTime",
                "endTime", "interruptionCause", "release", "campaign", "gnssApiType",
                "lastLogOfSession", "allMeasures"):  # tuple element: the last two JSON keys read from each phone log's header
        # rf-string regex: "key": optional quote, value up to a quote, comma, } or ] (}} in an f-string is a literal })
        m = re.search(rf'"{key}"\s*:\s*"?([^",}}\]]+)', head)
        out[key] = m.group(1).strip() if m else ""  # conditional expression: the value, or "" when absent
    m = re.search(r'"model"\s*:\s*"([^"]+)"', head)  # the handset model name, a quoted string value
    out["model_json"] = m.group(1) if m else ""  # stored under a separate name so it cannot clash with the file name's
    m = re.search(r'"gnssHwModel"\s*:\s*"([^"]+)"', head)  # the GNSS chipset description
    out["gnss_hw_model"] = m.group(1) if m else ""  # stored, or "" if absent
    m = re.search(r'"constellations"\s*:\s*\[([^\]]*)\]', head)  # the constellation list: everything inside [ ... ]
    # re.findall pulls each quoted name out of the list; ";".join makes one text cell; "" if the list was absent
    out["constellations"] = ";".join(re.findall(r'"([^"]+)"', m.group(1))) if m else ""
    m = re.search(r'"measurementDateTime"\s*:\s*"([^"]+)"', head)  # the first measurement's time stamp
    out["first_measurement_datetime"] = m.group(1) if m else ""  # stored as the raw text
    return out  # hands back the dict of header fields


def inventory_phones() -> list[dict]:  # function: one inventory row per Android capture
    rows: list[dict] = []  # annotated empty list of rows
    # generator filtered by two conditions: folders under the capture day that are NOT session folders, i.e. phones
    for folder in sorted(p for p in DAY.iterdir()
                         if p.is_dir() and not SESSION_DIR_RE.match(p.name)):  # generator clause: keeps folders that are not session folders, i.e. the phone captures
        for jf in sorted(folder.glob("*.json")):  # glob("*.json") finds the capture files; sorted fixes the order
            nm = PHONE_NAME_RE.match(jf.name)  # parses the file name with the named-group pattern
            if not nm:  # if: the name has an unexpected shape
                flag(f"phone file name does not match the GnssLogger pattern: {jf.name}")  # recorded, not guessed
                continue  # skip the file
            fields = phone_header_fields(jf)  # the header fields of this capture
            # match objects can be indexed by group name; strptime parses the file-name date and time (no zone)
            fname_ts = datetime.strptime(f"{nm['date']} {nm['time']}", "%Y-%m-%d %H-%M-%S")

            start_utc = end_utc = None  # chained assignment: both unknown until parsed
            tz_stated = ""  # records whether the file itself states its time zone
            if fields["startTime"]:  # if: the header gives a start time
                if fields["startTime"].endswith("Z"):  # if: a trailing "Z" is ISO 8601's explicit marker for UTC
                    tz_stated = "UTC (explicit Z suffix in startTime)"  # the file says UTC outright
                # "Z" replaced by "+00:00" so fromisoformat accepts it; the result is an aware UTC instant
                start_utc = datetime.fromisoformat(fields["startTime"].replace("Z", "+00:00"))
            if fields["endTime"]:  # if: the header gives an end time
                end_utc = datetime.fromisoformat(fields["endTime"].replace("Z", "+00:00"))  # parsed the same way

            fname_matches_start = None  # unknown until compared
            if start_utc is not None:  # if: there is a start time to compare with
                # strips the zone so the aware start can be subtracted from the naive file-name time; gap in seconds
                delta = abs((start_utc.replace(tzinfo=None) - fname_ts).total_seconds())
                fname_matches_start = delta < 1.0  # boolean: True if they agree to within a second
                if not fname_matches_start:  # if: they disagree
                    flag(f"{jf.name}: filename timestamp and startTime differ by {delta:.1f} s")  # recorded

            zpath = jf.with_suffix(".zip")  # with_suffix swaps .json for .zip: the matching data archive
            zip_entries, zip_size = "", None  # defaults for when there is no archive
            if zpath.exists():  # if: the archive is present
                zip_size = zpath.stat().st_size  # its size in bytes, from the file system
                with zipfile.ZipFile(zpath) as zf:  # with block: opens the archive and closes it afterwards
                    # for every member name: drop folders (rsplit "/"), keep the extension (rsplit "."); sort and join
                    zip_entries = ";".join(sorted(n.rsplit("/", 1)[-1].rsplit(".", 1)[-1]
                                                  for n in zf.namelist()))  # generator clause: runs over every member name inside the phone's zip archive
            else:  # else: no archive beside the JSON
                flag(f"{jf.name}: no matching .zip archive alongside the .json")  # recorded

            rows.append({  # appends one row dict; the keys below become the CSV columns
                "device_folder": folder.name,  # dict entry: the phone's folder name, which identifies the device
                # "a or b" returns a if it is non-empty, otherwise b: the JSON's model name, else the file name's
                "brand": nm["brand"], "model": fields["model_json"] or nm["model"],
                "android_sdk": nm["sdk"], "logger_session_id": fields["sessionId"],  # dict entry: Android version from the file name and the logger's own session number
                "json_filename": jf.name,  # dict entry: name of the JSON log the times are read from
                "filename_timestamp_raw": fname_ts.strftime("%Y-%m-%d %H:%M:%S"),  # dict entry: the time stamp written in the file name, as found
                "filename_timestamp_matches_startTime": fname_matches_start,  # dict entry: whether that time stamp equals the log's own start time
                "time_zone_stated_in_file": tz_stated,  # dict entry: the time zone the log states, if any; it decides how the phone times are read
                "start_utc": fmt_utc(start_utc), "end_utc": fmt_utc(end_utc),  # dict entry: start and end of the phone capture in UTC
                "start_local_no": fmt_no(start_utc), "end_local_no": fmt_no(end_utc),  # dict entry: the same start and end in Norwegian local time
                "duration_s_stated": fields["durationInSeconds"],  # dict entry: the capture length the log itself states, in seconds
                # conditional expression in brackets: the duration in minutes when both ends are known, else None
                "duration_min_computed": (round((end_utc - start_utc).total_seconds() / 60.0, 2)
                                          if start_utc and end_utc else None),  # conditional expression: the duration is left empty when either end of the capture is unknown
                "events": fields["events"],  # dict entry: the number of measurement events in the log
                "constellations": fields["constellations"],  # dict entry: the constellations the phone logged
                "gnss_hw_model": fields["gnss_hw_model"],  # dict entry: the phone's GNSS chipset model
                "interruption_cause": fields["interruptionCause"],  # dict entry: why the logger stopped, as the log records it
                "app_release": fields["release"], "campaign": fields["campaign"],  # dict entry: the logging app's release and the campaign name it was run under
                "first_measurement_datetime": fields["first_measurement_datetime"],  # dict entry: date and time of the first measurement in the log
                "json_size_bytes": jf.stat().st_size,  # dict entry: size of the JSON log in bytes
                "zip_filename": zpath.name if zpath.exists() else "",  # conditional expression: archive name or ""
                "zip_size_bytes": zip_size,  # dict entry: size of the zip archive in bytes
                "zip_content_extensions": zip_entries,  # dict entry: the file types inside the archive, joined into one cell
                # keys starting with "_" hold the datetime objects for later arithmetic; they are dropped from the CSV
                "_start_utc_dt": start_utc, "_end_utc_dt": end_utc,
            })
    return rows  # hands back all phone rows


# function: the fourth independent time check, testing whether the phone captures contain the laptop sessions
def phone_time_witness(sessions: list[dict], phones: list[dict], base: dict) -> dict:
    """
    Fourth independent check on the time base.

    The phones are not per-test captures, so the test is containment of each
    laptop session inside a phone capture window, plus coincidence of session
    and phone boundaries - evaluated for every hypothesis, not only the chosen one.
    """
    # list comprehension: (start, end, device, file) for every phone capture whose start and end are both known
    windows = [(p["_start_utc_dt"], p["_end_utc_dt"], p["device_folder"], p["json_filename"])
               for p in phones if p["_start_utc_dt"] and p["_end_utc_dt"]]  # generator clause: keeps only phone captures whose start and end are both known

    per_hypothesis = []  # empty list: one containment score per hypothesis
    for tag, label, rmode, ptz in HYPOTHESES:  # for loop over the four hypotheses (only rmode matters here)
        covered, total_cov = 0, 0.0  # sessions fully inside a phone capture, and total covered minutes
        for s in sessions:  # for loop over the six sessions
            cs = readme_instant(s["readme"]["start_raw"], rmode)  # session start in UTC under this reading
            ce = readme_instant(s["readme"]["end_raw"], rmode)  # session end in UTC under this reading
            dur = (ce - cs).total_seconds() / 60.0  # session length in minutes
            # max over a generator of overlaps with every phone window; default=0.0 if there are no windows
            best = max((overlap_minutes(cs, ce, w0, w1) for w0, w1, _, _ in windows), default=0.0)
            total_cov += best  # adds the best overlap to the total
            if dur > 0 and best / dur >= 0.99:  # if: at least 99 % of the session lies inside one phone capture
                covered += 1  # counts it as fully contained
        # appends this hypothesis's score row
        per_hypothesis.append({"hypothesis": tag, "sessions_fully_inside_a_phone_capture": covered,
                               "total_covered_min": round(total_cov, 2)})  # dict entry: minutes of session time covered by a phone capture under this hypothesis

    rows = []  # empty list: per-session detail under the chosen hypothesis only
    for s in sessions:  # for loop over the sessions
        cs = readme_instant(s["readme"]["start_raw"], base["readme_mode"])  # session start under the chosen reading
        ce = readme_instant(s["readme"]["end_raw"], base["readme_mode"])  # session end under the chosen reading
        dur = (ce - cs).total_seconds() / 60.0  # session length in minutes
        best = None  # the best-overlapping phone window so far
        for w0, w1, dev, fn in windows:  # for loop unpacking each phone window
            ov = overlap_minutes(cs, ce, w0, w1)  # overlap with this window
            if best is None or ov > best["overlap_min"]:  # if: first window, or a better one than before
                # keeps this window as the best so far
                best = {"overlap_min": ov, "device": dev, "file": fn,
                        "phone_start_utc": w0, "phone_end_utc": w1}  # dict entry: start and end of the best-overlapping phone capture
        rows.append({  # appends the session's alignment row; the offsets are in seconds
            "test_code": s["test_code"],  # dict entry: the test code of the session
            "session_start_utc": cs, "session_end_utc": ce,  # dict entry: start and end of the laptop capture in UTC under the chosen time base
            "phone_device": best["device"], "phone_file": best["file"],  # dict entry: which phone and which log overlap the session most
            "phone_start_utc": best["phone_start_utc"], "phone_end_utc": best["phone_end_utc"],  # dict entry: start and end of that phone capture in UTC
            "overlap_min": best["overlap_min"],  # dict entry: minutes the phone capture and the session share
            "fraction_covered": round(best["overlap_min"] / dur, 4) if dur > 0 else None,  # share of the session
            "start_offset_s": round((cs - best["phone_start_utc"]).total_seconds(), 1),  # + means phone started first
            "end_offset_s": round((best["phone_end_utc"] - ce).total_seconds(), 1),  # + means phone stopped later
        })

    boundary_hits = [r for r in rows if abs(r["start_offset_s"]) <= 180]  # sessions starting within 3 min of a phone
    pairs = []  # empty list: pairs of phone captures on different devices started within two minutes of each other
    by_time = sorted(phones, key=lambda p: p["start_utc"])  # phones sorted by their start-time text
    for i, a in enumerate(by_time):  # for loop: each phone capture a with its position i
        for b in by_time[i + 1:]:  # inner loop over the later captures only, so each pair is seen once
            # if: same device, or either start unknown, so the pair is not a cross-device comparison
            if a["device_folder"] == b["device_folder"] or not (a["_start_utc_dt"] and b["_start_utc_dt"]):
                continue  # skip it
            d = abs((a["_start_utc_dt"] - b["_start_utc_dt"]).total_seconds())  # gap between their starts, seconds
            if d <= 120:  # if: started within two minutes of each other
                pairs.append({"a": a["json_filename"], "b": b["json_filename"], "delta_s": round(d, 1)})  # keep it

    # The phones say nothing about the transmission plan's own time zone, so they
    # cannot separate H1 from H2. What they do test is the README interpretation,
    # which is the part that determines every label. Score on that.
    mode_of = {h[0]: h[2] for h in HYPOTHESES}  # dict comprehension: hypothesis tag -> its README reading
    by_mode: dict[str, int] = {}  # annotated empty dict: README reading -> best containment count
    for p in per_hypothesis:  # for loop over the four hypothesis scores
        m = mode_of[p["hypothesis"]]  # which README reading this hypothesis uses
        # keeps the larger count per reading; H1/H2 share a reading, as do H3/H4
        by_mode[m] = max(by_mode.get(m, 0), p["sessions_fully_inside_a_phone_capture"])
    chosen_mode = base["readme_mode"]  # the reading the plan alignment chose
    rival = max((v for k, v in by_mode.items() if k != chosen_mode), default=-1)  # the other reading's best count
    agrees = by_mode.get(chosen_mode, -1) > rival  # boolean: the phones favour the chosen reading strictly

    # returns everything the report needs; next() finds the first phone that states its time zone, or ""
    return {"rows": rows, "per_hypothesis": per_hypothesis, "pairs": pairs,
            "boundary_hits": boundary_hits, "agrees": agrees, "by_mode": by_mode,  # dict entry: README times that sit on a phone boundary, the verdict, and the score per reading
            "chosen_mode": chosen_mode,  # dict entry: the README reading the phones support
            "n_windows": len(windows),  # dict entry: how many phone capture windows were compared
            "tz_stated": next((p["time_zone_stated_in_file"] for p in phones  # dict entry: the first time zone any phone log states, or empty when none does
                               if p["time_zone_stated_in_file"]), "")}  # generator clause: skips phones that state no time zone


# --------------------------------------------------------------------------
# 7. Phase structure
# --------------------------------------------------------------------------

def sentences(text: str) -> list[str]:  # function: splits a catalogue description into sentences
    text = re.sub(r"\s+", " ", text or "").strip()  # (text or "") guards against None; whitespace runs become one space
    # re.split at spaces preceded by . ! or ? (the (?<=...) lookbehind keeps the punctuation); empty pieces dropped
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def extract_phases(code: str, cat: dict) -> list[dict]:  # function: the phase rows a test's catalogue text states
    """
    Record the internal phase structure a test actually states, with the sentence
    it came from. Nothing is inferred; a test that states nothing says so.
    """
    rows: list[dict] = []  # annotated empty list of phase rows
    grp, tst = cat["group_description"], cat["test_description"]  # tuple assignment: group text and test text

    override, override_sentence = False, ""  # default: the test does not override the group structure
    for s in sentences(tst):  # for loop over the test's own sentences
        # regex with | (either phrase) and \b word boundaries, case-insensitive: "no/without initial jamming"
        if re.search(r"\bno initial jamming\b|\bwithout initial jamming\b", s, re.IGNORECASE):
            override, override_sentence = True, s  # remembers the override and the sentence that states it
            break  # one statement is enough

    for s in sentences(grp):  # for loop over the group description's sentences
        # regex: ".*?" is a non-greedy gap; groups capture the minutes and the band list up to the next full stop
        m = re.search(r"During the first (\d+) minutes.*?spoofing and jamming on ([^.]+)", s, re.IGNORECASE)
        if m:  # if: the sentence states the opening jamming phase
            # appends a phase row that keeps its source sentence and whether this test overrides it
            rows.append({"phase_name": "simultaneous spoofing and jamming",
                         "offset_start_min": 0.0, "offset_end_min": float(m.group(1)),  # dict entry: the phase starts at the test start and lasts the minutes the sentence states
                         "bands": m.group(2).strip(), "offset_basis": "stated_minutes",  # dict entry: the bands jammed during the phase, and offsets read as stated minutes
                         "source_field": "group_description", "source_sentence": s,  # dict entry: the phase comes from the group description; its sentence is kept as evidence
                         "applies_by_default": True, "overridden_by_test": override,  # dict entry: the phase applies by default unless this test's own text overrides it
                         "override_sentence": override_sentence if override else ""})  # dict entry: the overriding sentence, kept only when there is an override
        # regex for the sentence saying when jamming stops; groups capture the minutes and the spoofed bands
        m = re.search(r"After (\d+) minutes, jamming will stop.*?spoofing will continue on ([^.]+)", s, re.IGNORECASE)
        if m:  # if: found
            # appends the spoofing-only phase, open-ended (offset_end_min None)
            rows.append({"phase_name": "spoofing only",
                         "offset_start_min": float(m.group(1)), "offset_end_min": None,  # dict entry: spoofing alone starts at the stated minute and runs to the end of the test
                         "bands": m.group(2).strip(), "offset_basis": "stated_minutes",  # dict entry: the spoofed bands, and offsets read as stated minutes
                         "source_field": "group_description", "source_sentence": s,  # dict entry: the phase comes from the group description; its sentence is kept as evidence
                         "applies_by_default": True, "overridden_by_test": override,  # dict entry: the phase applies by default unless this test's own text overrides it
                         "override_sentence": override_sentence if override else ""})  # dict entry: the overriding sentence, kept only when there is an override
        m = re.search(r"There will be continuous jamming on ([^.]+)", s, re.IGNORECASE)  # the always-on jamming bands
        if m:  # if: found
            # appends the continuous-jamming phase; no test exempts itself from it, so overridden is always False
            rows.append({"phase_name": "continuous jamming on other bands",
                         "offset_start_min": 0.0, "offset_end_min": None,  # dict entry: the continuous jamming runs from the test start to its end
                         "bands": m.group(1).strip(), "offset_basis": "stated_minutes",  # dict entry: the bands jammed throughout, and offsets read as stated minutes
                         "source_field": "group_description", "source_sentence": s,  # dict entry: the phase comes from the group description; its sentence is kept as evidence
                         "applies_by_default": True, "overridden_by_test": False,  # dict entry: applies by default, and no test exempts itself from it
                         "override_sentence": ""})  # dict entry: no overriding sentence exists for this phase

    if override:  # if: the test overrides the group default
        # appends a row recording the override itself, sourced from the test's own sentence
        rows.append({"phase_name": "no initial jamming (test-level override)",
                     "offset_start_min": 0.0, "offset_end_min": None, "bands": "",  # dict entry: the override covers the whole test and names no band
                     "offset_basis": "stated_minutes", "source_field": "test_description",  # dict entry: offsets read as stated minutes, taken from the test's own description
                     "source_sentence": override_sentence, "applies_by_default": False,  # dict entry: the sentence that removes the initial jamming; it is not a default phase
                     "overridden_by_test": False, "override_sentence": ""})  # dict entry: an override is not itself overridden

    for s in sentences(tst):  # for loop over the test's own sentences, looking for test-specific structure
        # regex: (.+?) lazily takes the route text up to the first " is <digits> minutes"; second group is the minutes
        m = re.search(r"estimated driving time between (.+?) is (\d+) minutes", s, re.IGNORECASE)
        if m:  # if: a driving time is stated
            # appends an "outbound leg" row; the f-string puts the route into the phase name
            rows.append({"phase_name": f"outbound leg ({m.group(1).strip()})",
                         "offset_start_min": 0.0, "offset_end_min": float(m.group(2)),  # dict entry: the leg starts at the test start and lasts the stated minutes
                         "bands": "", "offset_basis": "stated_minutes",  # dict entry: no band named for the leg, and offsets read as stated minutes
                         "source_field": "test_description", "source_sentence": s,  # dict entry: the leg comes from the test description; its sentence is kept as evidence
                         "applies_by_default": False, "overridden_by_test": False,  # dict entry: a test-specific phase, so no default and no override
                         "override_sentence": ""})  # dict entry: no overriding sentence for a test-specific phase
        m = re.search(r"on air for (\d+)[- ]minutes", s, re.IGNORECASE)  # "on air for 40-minutes" or "40 minutes"
        if m:  # if: a total transmission time is stated
            # appends a total-duration row
            rows.append({"phase_name": "total transmission duration",
                         "offset_start_min": 0.0, "offset_end_min": float(m.group(1)),  # dict entry: the transmission runs from the test start for the stated total minutes
                         "bands": "", "offset_basis": "stated_minutes",  # dict entry: no band named, and offsets read as stated minutes
                         "source_field": "test_description", "source_sentence": s,  # dict entry: the duration comes from the test description; its sentence is kept as evidence
                         "applies_by_default": False, "overridden_by_test": False,  # dict entry: a test-specific phase, so no default and no override
                         "override_sentence": ""})  # dict entry: no overriding sentence for a test-specific phase
        # regex: groups capture a distance like "1 km" and a speed like "160 km/h" from the acceleration sentence
        m = re.search(r"After approximately ([\d.]+ ?km)[^.]*accelerate up to ([\d]+ ?km/h)", s, re.IGNORECASE)
        if m:  # if: found
            # appends the row with NO time offset: the text gives a distance, which is kept as a distance
            rows.append({"phase_name": f"acceleration to {m.group(2)}",
                         "offset_start_min": None, "offset_end_min": None, "bands": "",  # dict entry: no time offsets, because the text gives a distance and not a time
                         "offset_basis": f"stated_distance ({m.group(1)}); no time offset stated",  # dict entry: records the stated distance and that no time offset is given
                         "source_field": "test_description", "source_sentence": s,  # dict entry: the phase comes from the test description; its sentence is kept as evidence
                         "applies_by_default": False, "overridden_by_test": False,  # dict entry: a test-specific phase, so no default and no override
                         "override_sentence": ""})  # dict entry: no overriding sentence for a test-specific phase

    if not any(r["source_field"] == "test_description" for r in rows):  # if: the test itself stated nothing
        # appends an explicit "nothing stated" row, so silence is recorded rather than left blank
        rows.append({"phase_name": "no test-level phase structure stated",
                     "offset_start_min": None, "offset_end_min": None, "bands": "",  # dict entry: no offsets and no bands, because the test states no phases
                     "offset_basis": "not_stated", "source_field": "test_description",  # dict entry: marks the silence as not stated, from the test description
                     "source_sentence": "", "applies_by_default": False,  # dict entry: no sentence to quote, and nothing applies by default
                     "overridden_by_test": False, "override_sentence": ""})  # dict entry: nothing overridden, no overriding sentence

    for r in rows:  # for loop over every row produced
        r["test_code"] = code  # tags it with the test code
    return rows  # hands back the phase rows for this test


# --------------------------------------------------------------------------
# 8. Figures
# --------------------------------------------------------------------------

def _no_axis(ax, t0: datetime, t1: datetime) -> None:  # function (leading _ = internal helper): Norwegian time axis
    import matplotlib.dates as mdates  # local import: matplotlib's date helpers, needed only inside this function
    ax.set_xlim(mdates.date2num(t0), mdates.date2num(t1))  # date2num: datetimes -> plot numbers; sets the x-range
    ax.xaxis.set_major_locator(mdates.HourLocator())  # a major tick on every whole hour
    ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[0, 15, 30, 45]))  # minor ticks every quarter hour
    # tick labels as HH:MM shown in Europe/Oslo time, although the plotted instants are UTC underneath
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=TZ_OSLO))
    ax.grid(axis="x", which="major", color=C_GREY, alpha=0.35, linewidth=0.7)  # faint vertical lines at each hour
    ax.grid(axis="x", which="minor", color=C_GREY, alpha=0.15, linewidth=0.4)  # fainter lines at each quarter hour


def save_deterministic(fig, path: Path) -> None:  # function: saves a figure so runs again give byte-identical files
    # 300 dpi, trimmed to the drawn content; metadata None removes the software and creation-time stamps from the PNG
    fig.savefig(path, dpi=300, bbox_inches="tight",
                metadata={"Software": None, "Creation Time": None})  # keyword argument metadata=: removes the software and time stamps, so runs again give identical PNG bytes
    plt.close(fig)  # frees the figure's memory


# dict literal: test code -> short scenario label printed under the code on Figure 1's y-axis
SHORT_NAME = {
    "2.6.2": "fixed spoofed position, large jump",  # dict entry: scenario label of 2.6.2 under its code on the y-axis
    "2.6.4": "spoofed position drifts into terrain",  # dict entry: scenario label of 2.6.4
    "2.6.5": "spoofed position travels south",  # dict entry: scenario label of 2.6.5
    "2.6.10": "power ramp over 40 minutes",  # dict entry: scenario label of 2.6.10
    "2.6.11": "position jump, spoofed tunnel transit",  # dict entry: scenario label of 2.6.11
    "2.6.12": "time jump forward by 1 y 1 d 1 h 15 min",  # dict entry: scenario label of 2.6.12, the time-spoofing test
}


def figure_one(inv: pd.DataFrame, path: Path) -> None:  # function: draws Figure 1, capture versus schedule per test
    import matplotlib.dates as mdates  # local import of matplotlib's date helpers
    rows = inv.sort_values("_cap_start").reset_index(drop=True)  # sessions in capture order, renumbered 0..5
    fig, ax = plt.subplots(figsize=(12.5, 5.6))  # a new figure with one set of axes, 12.5 x 5.6 inches
    # the axis starts 10 minutes before the earliest scheduled or captured start (timedelta is a duration)
    t0 = min(rows["_sched_start"].min(), rows["_cap_start"].min()) - timedelta(minutes=10)
    t1 = max(rows["_sched_end"].max(), rows["_cap_end"].max()) + timedelta(minutes=10)  # and ends 10 minutes after

    for i, r in rows.iterrows():  # iterrows: loops over the table giving each row's index i and the row r
        y = len(rows) - 1 - i  # y position, reversed so the earliest session is drawn at the top
        ss, se = r["_sched_start"], r["_sched_end"]  # tuple assignment: scheduled start and end
        cs, ce = r["_cap_start"], r["_cap_end"]  # tuple assignment: capture start and end
        # horizontal bar: the scheduled window, 0.19 above the row line, hatched "//", drawn underneath (zorder 2)
        ax.barh(y + 0.19, mdates.date2num(se) - mdates.date2num(ss),
                left=mdates.date2num(ss), height=0.26,  # keyword arguments of barh(): the bar starts at the scheduled start and is 0.26 rows tall
                color=C_SKY, edgecolor=C_BLACK, linewidth=0.8, hatch="//", zorder=2)  # keyword arguments of barh(): sky-blue hatched fill under the capture bar marks the schedule
        # horizontal bar for the actual capture window, lowered 0.19, solid orange, drawn on top (zorder 3)
        ax.barh(y - 0.19, mdates.date2num(ce) - mdates.date2num(cs),
                left=mdates.date2num(cs), height=0.20,  # keyword arguments of barh(): the bar starts at the capture start and is 0.20 rows tall
                color=C_ORANGE, edgecolor=C_BLACK, linewidth=0.8, zorder=3)  # keyword arguments of barh(): solid orange fill drawn above the scheduled bar marks the capture

        # Margins: candidate clean data, not verified clean (the receiver-status script found front-end disturbance in the
        # 2.6.5 and 2.6.10 margins); the code below only labels the margin minutes
        if r["minutes_before_scheduled_start"] > 0:  # if: the capture began before the test
            # green text "+N.N" just left of the capture bar's start: the pre-test margin in minutes
            ax.annotate(f"+{r['minutes_before_scheduled_start']:.1f}",
                        xy=(mdates.date2num(cs), y - 0.19), xytext=(-4, 0),  # keyword arguments of annotate(): anchored at the capture start, shifted 4 points to the left
                        textcoords="offset points", ha="right", va="center",  # keyword arguments of annotate(): the shift is in points, the text ends at the anchor and is centred
                        fontsize=7.5, color=C_GREEN)  # keyword arguments of annotate(): small green text, the colour of the margin figures
        if r["minutes_after_scheduled_end"] > 0:  # if: the capture ran past the test
            # green text just right of the capture bar's end: the post-test margin in minutes
            ax.annotate(f"+{r['minutes_after_scheduled_end']:.1f}",
                        xy=(mdates.date2num(ce), y - 0.19), xytext=(4, 0),  # keyword arguments of annotate(): anchored at the capture end, shifted 4 points to the right
                        textcoords="offset points", ha="left", va="center",  # keyword arguments of annotate(): the shift is in points, the text starts at the anchor and is centred
                        fontsize=7.5, color=C_GREEN)  # keyword arguments of annotate(): small green text, the colour of the margin figures
        if r["partial_coverage_flag"]:  # if: the capture misses part of the scheduled test
            miss = []  # empty list of "missing" phrases
            if cs > ss:  # if: recording began after the test started
                miss.append(f"{minutes(cs - ss):.1f} min missing at start")  # how much of the start is missing
            if ce < se:  # if: recording stopped before the test ended
                miss.append(f"{minutes(se - ce):.1f} min missing at end")  # how much of the end is missing
            # Anchor the note inside the axes by a fixed rule: right-aligned when the later of the scheduled and capture
            # starts lies beyond 55 % of the axis width, otherwise left-aligned (the text width is never measured)
            late = (max(ss, cs) - t0) / (t1 - t0) > 0.55
            # italic vermillion note below the row; conditional expressions pick its anchor and alignment
            ax.annotate("partial: " + "; ".join(miss),
                        xy=(mdates.date2num(min(se, ce) if late else max(ss, cs)), y - 0.36),  # keyword argument xy=: anchors the note at the end that is missing, just below the row
                        xytext=(0, -2), textcoords="offset points",  # keyword arguments of annotate(): a 2-point downward shift, measured in points
                        fontsize=7.2, color=C_VERMILLION, style="italic",  # keyword arguments of annotate(): small italic vermillion text marks the partial coverage
                        va="top", ha="right" if late else "left")  # keyword arguments of annotate(): text hangs below the anchor and grows away from the missing end

    ax.set_yticks(range(len(rows)))  # one y tick per session
    ax.set_yticklabels(  # sets the tick labels from the list built below
        # list comprehension: "code\nshort name" per row; iloc[::-1] reverses the rows to match the reversed y
        [f"{r['test_code']}\n{SHORT_NAME.get(r['test_code'], '')}"
         for _, r in rows.iloc[::-1].iterrows()], fontsize=8)  # generator clause: one label per session row, read bottom-up to match the reversed y-axis
    ax.set_ylim(-0.85, len(rows) - 0.35)  # vertical limits leaving room for the partial-coverage notes
    ax.set_xlabel("Norwegian local time on 14 September 2026 (CEST, UTC+02:00)")  # x-axis title
    _no_axis(ax, t0, t1)  # applies the shared Norwegian time-axis formatting
    ax.legend(handles=[  # legend built by hand from three proxy artists
        # proxy patch standing for the hatched scheduled bar
        Patch(facecolor=C_SKY, edgecolor=C_BLACK, hatch="//",
              label="Scheduled test window (transmission plan)"),  # keyword argument label=: legend text for the hatched scheduled bar
        Patch(facecolor=C_ORANGE, edgecolor=C_BLACK,  # proxy patch for the solid capture bar
              label="Actual capture window (receiver logs)"),  # keyword argument label=: legend text for the solid capture bar
        # an invisible line with a "+" marker (mathtext $+$), standing for the green margin figures
        plt.Line2D([], [], linestyle="none", marker="$+$", color=C_GREEN,
                   label="Green figures: minutes of margin outside the scheduled test")],  # keyword argument label=: legend text explaining the green margin figures
        loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False, fontsize=8.5)  # below the axes
    save_deterministic(fig, path)  # writes the PNG without time stamps


# function: draws Figure 2, all three test areas' schedules against our captures; returns a few facts about it
def figure_two(plan_rows: list[dict], inv: pd.DataFrame, path: Path) -> dict:
    import matplotlib.dates as mdates  # local import of matplotlib's date helpers
    day = datetime.strptime(TEST_DAY, "%Y-%m-%d").date()  # the capture date as a date object
    # combine date and "14:00" into an aware datetime in Oslo time: the left edge of the plot
    w0 = datetime.combine(day, datetime.strptime(FIG2_WINDOW[0], "%H:%M").time(), tzinfo=TZ_OSLO)
    w1 = datetime.combine(day, datetime.strptime(FIG2_WINDOW[1], "%H:%M").time(), tzinfo=TZ_OSLO)  # the right edge

    # Every test whose window OVERLAPS the plotted range, including those that
    # begin before it or end after it; a long-running emitter must not vanish
    # merely because its start time falls outside the clipping window.
    shown = [p for p in plan_rows if p["_start_utc"] < w1 and p["_end_utc"] > w0]  # overlap test for each entry
    our_codes = set(inv["test_code"])  # set of our six test codes, for fast membership tests

    fig, ax = plt.subplots(figsize=(14, 7.2))  # new figure, 14 x 7.2 inches
    lane_of = {1: 2.0, 2: 1.0, 3: 0.0}  # dict: test area -> the y of its lane (Area 1 highest)
    clipped_any = False  # flag: did any bar run past the plotted window?

    for area in (1, 2, 3):  # for loop over the three test areas
        base_y = lane_of[area]  # this area's lane
        # this area's shown entries, sorted by start time
        ents = sorted([p for p in shown if p["test_area"] == area], key=lambda r: r["_start_utc"])
        slot_end: list[datetime] = []  # end time of the last bar in each sub-row, so overlapping tests stack
        for e in ents:  # for loop over the area's entries in time order
            # next() over a generator: the first sub-row already free when this entry starts, or None
            slot = next((k for k, end in enumerate(slot_end) if end <= e["_start_utc"]), None)
            if slot is None:  # if: every sub-row is busy
                slot_end.append(e["_end_utc"])  # opens a new sub-row
                slot = len(slot_end) - 1  # its index is the last position
            else:  # else: a free sub-row was found
                slot_end[slot] = e["_end_utc"]  # it is now busy until this entry ends
            e["_slot"] = slot  # stores the sub-row on the entry for drawing
        n = max(len(slot_end), 1)  # number of sub-rows, at least 1
        h = 0.68 / n  # each sub-row's height, sharing 0.68 of the lane
        for e in ents:  # for loop drawing each entry
            y = base_y - 0.34 + e["_slot"] * h  # the bottom of this entry's sub-row
            s_ = max(e["_start_utc"], w0)  # the bar's start, clipped to the plotted window
            en = min(e["_end_utc"], w1)  # the bar's end, clipped to the plotted window
            pw = e["power_w_effective"]  # the power used for display (plan, else catalogue maximum, else None)
            high = pw is not None and pw > HIGH_POWER_W  # boolean: known and strictly above 1 W
            unknown = pw is None  # boolean: neither the plan nor the catalogue states a power
            ours = area == 3 and e["test_code"] in our_codes  # boolean: one of our six captured tests
            # the bar; conditional expressions pick colour, outline, line width and hatching from high and ours
            ax.barh(y + h / 2, mdates.date2num(en) - mdates.date2num(s_),
                    left=mdates.date2num(s_), height=h * 0.80,  # keyword arguments of barh(): the bar starts at the entry's start and fills 80 per cent of its lane
                    # fill: vermillion above 1 W, white when the power is not stated, otherwise sky blue
                    color=C_VERMILLION if high else ("white" if unknown else C_SKY),
                    # edge colour: green for our tests; grey for an unknown power, so its /// hatching is grey
                    edgecolor=C_GREEN if ours else (C_GREY if unknown else C_BLACK),
                    linewidth=2.2 if ours else (1.4 if high else (0.9 if unknown else 0.6)),  # outline width
                    linestyle="--" if unknown else "-",  # dashed outline marks an unknown power
                    hatch="xx" if high else ("///" if unknown else None), zorder=3)  # hatching per power class
            # the test code written at the bar's middle on a white box (bbox) so it stays readable over hatching
            ax.text(mdates.date2num(s_ + (en - s_) / 2), y + h / 2, e["test_code"],
                    fontsize=6.8, va="center", ha="center", zorder=6,  # keyword arguments of text(): small code text centred on the bar, drawn above the hatching
                    bbox=dict(facecolor="white", alpha=0.80, pad=0.8,  # keyword argument bbox=: a white box behind the code text, so it stays readable over the bar
                              edgecolor="none", boxstyle="square,pad=0.15"))  # keyword arguments of dict(...): the box has no edge and a tight square shape
            if high:  # if: above 1 W
                # prints the power (":g" drops trailing zeros) just above the bar's left end; allowed outside the axes
                ax.annotate(f"{pw:g} W",
                            xy=(mdates.date2num(s_), y + h / 2 + h * 0.40),  # keyword argument xy=: the power label sits just above the bar's left end
                            xytext=(0, 3), textcoords="offset points",  # keyword arguments of annotate(): shifted 3 points upward, measured in points
                            fontsize=6.8, fontweight="bold", ha="left", va="bottom",  # keyword arguments of annotate(): small bold text growing right and upward from the anchor
                            color=C_VERMILLION, zorder=9, annotation_clip=False)  # keyword arguments of annotate(): vermillion like the high-power bars, and allowed outside the axes
            if e["_start_utc"] < w0:  # if: the test began before the plotted window
                clipped_any = True  # remembers that an arrow is needed in the legend
                # a "<" marker on the left axis edge; clip_on=False lets it sit on the border
                ax.plot(mdates.date2num(w0), y + h / 2, marker="<", markersize=7,
                        color=C_BLACK, zorder=8, clip_on=False)  # keyword arguments of plot(): black marker drawn on top and allowed to sit on the axis border
            if e["_end_utc"] > w1:  # if: the test ran past the plotted window
                clipped_any = True  # remembers that an arrow is needed in the legend
                # a ">" marker on the right axis edge
                ax.plot(mdates.date2num(w1), y + h / 2, marker=">", markersize=7,
                        color=C_BLACK, zorder=8, clip_on=False)  # keyword arguments of plot(): black marker drawn on top and allowed to sit on the axis border

    # Our captures: their own lane at the top, plus shading down the figure.
    cap_y = 3.05  # y of the top lane holding our six capture windows
    for _, r in inv.sort_values("_cap_start").iterrows():  # for loop over sessions in capture order
        x0, x1 = mdates.date2num(r["_cap_start"]), mdates.date2num(r["_cap_end"])  # capture start and end as numbers
        ax.axvspan(x0, x1, color=C_GREEN, alpha=0.13, zorder=1)  # pale green band down the whole figure
        # solid green bar in the top lane
        ax.barh(cap_y, x1 - x0, left=x0, height=0.22, color=C_GREEN,
                edgecolor=C_BLACK, linewidth=0.7, zorder=4)  # keyword arguments of barh(): black outline around the green capture bar, above the plan bars
        # the test code above the bar, in dark green (a hex colour written inline)
        ax.text((x0 + x1) / 2, cap_y + 0.20, r["test_code"], fontsize=7.0,
                ha="center", va="bottom", color="#0a5f47", zorder=7)  # keyword arguments of text(): the code is centred above the bar in dark green, on top of the bars

    ax.set_yticks([cap_y, 2.0, 1.0, 0.0])  # four y ticks: our captures and the three area lanes
    # the four lane labels; "\n" breaks a label over two lines
    ax.set_yticklabels(["Our captures\n(Test Area 3)", "Test Area 1", "Test Area 2",
                        "Test Area 3\n(scheduled)"], fontsize=9)  # list element: the scheduled lane of Test Area 3, where the captured tests ran
    ax.set_ylim(-0.55, 3.55)  # vertical limits
    ax.set_xlabel("Norwegian local time on 14 September 2026 (CEST, UTC+02:00)")  # x-axis title
    _no_axis(ax, w0, w1)  # shared time-axis formatting over 14:00-18:30

    handles = [  # list of proxy patches for the legend, one per visual code
        # at or below 1 W: plain sky blue
        Patch(facecolor=C_SKY, edgecolor=C_BLACK,
              # continuation: legend text; ":g" prints 1.0 as 1
              label=f"Scheduled test at or below {HIGH_POWER_W:g} W"),
        # power not stated: white, grey /// hatching, dashed grey edge, so it reads as neither low nor high
        Patch(facecolor="white", edgecolor=C_GREY, hatch="///", linestyle="--", linewidth=0.9,
              # continuation: the legend text for an unknown power
              label="Power not stated in plan or catalogue"),
        Patch(facecolor=C_VERMILLION, edgecolor=C_BLACK, hatch="xx",  # above 1 W: vermillion, cross-hatched
              label=f"Scheduled test above {HIGH_POWER_W:g} W (power printed above the bar)"),  # keyword argument label=: legend text for the high-power bars, whose power is printed above them
        Patch(facecolor=C_SKY, edgecolor=C_GREEN, linewidth=2.2,  # green outline: our six tests
              label="Green outline: the six tests this study captured"),  # keyword argument label=: legend text for the green outline that marks the six captured tests
        Patch(facecolor=C_GREEN, edgecolor=C_BLACK,  # solid green: our capture windows
              label="Our capture windows (also shaded down the figure)"),  # keyword argument label=: legend text for the capture windows in the top lane
    ]
    if clipped_any:  # if: at least one arrow was drawn
        # adds a legend entry explaining the arrow marker
        handles.append(plt.Line2D([], [], marker=">", linestyle="none", color=C_BLACK,
                                  label="Arrow: the test continues beyond the plotted window"))  # keyword argument label=: legend text for the arrow that marks a test running past the window
    # the legend in two columns below the axes, without a frame
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.11),
              ncol=2, frameon=False, fontsize=8.5)  # keyword arguments of legend(): two columns without a frame, in a small font
    save_deterministic(fig, path)  # writes the PNG without time stamps
    return {"n_shown": len(shown), "window": FIG2_WINDOW, "clipped": clipped_any}  # facts for the printed report


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:  # function: runs the whole section in order; called only when the file is run directly
    PROCESSED.mkdir(parents=True, exist_ok=True)  # creates Data/processed if absent; exist_ok: no error if present
    GRAPH.mkdir(parents=True, exist_ok=True)  # creates Graph/ likewise

    print("=" * 78)  # string repetition: a rule of 78 "=" characters
    print("GROUND TRUTH AND TIME ALIGNMENT")  # report title
    print("=" * 78)  # closing rule

    # --- 1. discovery ----------------------------------------------------
    sessions = discover_sessions()  # the six session folders, found by name pattern
    print(f"\n[1] Discovered {len(sessions)} capture sessions by folder pattern:")  # f-string with the count
    for s in sessions:  # for loop over the sessions
        print(f"      {s['folder']}   -> test {s['test_code']}")  # folder name and its test code

    # --- 2. session metadata ---------------------------------------------
    for s in sessions:  # for loop reading each session's metadata files into its dict
        s["readme"] = parse_readme(s["path"] / "README.txt")  # the capture metadata
        s["events"] = parse_events(s["path"] / "session_events.log")  # the event log
        gap_path = s["path"] / "stream_gaps.log"  # path of the stream-continuity log
        # "A if cond else B" split by "\": sum() counts the gap log's non-blank lines (one per gap), or None if no log
        s["n_gaps"] = sum(1 for l in gap_path.read_text(encoding="utf-8").splitlines() if l.strip()) \
            if gap_path.exists() else None  # conditional expression: the gap count stays empty when the session has no gap log
        if gap_path.exists() is False:  # if: the log is missing ("is False" compares with the False object)
            flag(f"{s['folder']}: stream_gaps.log missing")  # recorded as an anomaly
        s["ts"] = parse_timestamp_csv(s["path"] / "timestamps_system_capture.csv")  # first/last 1 Hz clock rows

    # --- 3. catalogue -----------------------------------------------------
    catalog = load_catalog()  # the test catalogue as a lookup by test code
    print(f"\n[2] Test catalogue loaded: {len(catalog)} test codes.")  # reports its size
    for s in sessions:  # for loop matching each session to its catalogue entry
        if s["test_code"] not in catalog:  # if: the catalogue has no such code
            die(f"TEST CODE {s['test_code']} HAS NO MATCH IN testcatalog.json")  # unlabelable session: stop
        s["cat"] = catalog[s["test_code"]]  # attaches the catalogue fields to the session
    print("      All six session test codes matched.")  # reached only if every code matched

    # --- 4. transmission plan --------------------------------------------
    plan, geom = parse_transmission_plan()  # tuple unpacking: the scheduled entries and the column geometry
    # prints which PDF pages were used and where the three columns start
    print(f"\n[3] Transmission plan parsed from PDF pages {geom['pages']} "
          f"(0-indexed); column x-origins {geom['origins']}.")  # f-string continued: the x positions where the plan's three columns start
    # prints the entry count; the set comprehension counts distinct test areas
    print(f"      {len(plan)} scheduled entries on {PLAN_DAY_HEADER} {TEST_DAY} "
          f"across {len({p['test_area'] for p in plan})} test areas.")  # f-string continued: how many distinct test areas the entries cover

    # --- 5. time base -----------------------------------------------------
    base, hyp_df = derive_time_base(sessions, plan)  # the chosen time base and the table of all four scores
    print("\n[4] TIME BASE DERIVED FROM THE DATA")  # heading (a plain string: nothing to insert)
    for _, r in hyp_df.iterrows():  # for loop over the score table; "_" discards the row index
        mark = "  <== chosen" if r["hypothesis"] == base["tag"] else ""  # marker text for the winning row
        print(f"      {r['hypothesis']}: {r['description']}")  # hypothesis tag and description
        # its score: sessions missing their test, and total overlap minutes
        print(f"          zero-overlap sessions {int(r['sessions_with_zero_overlap'])}, "
              f"total overlap {r['total_overlap_min']:.1f} min{mark}")  # f-string continued: the hypothesis's total overlap minutes, with a mark on the winner
    print(f"\n      Conclusion: {base['label']}.")  # the winning description
    print("      The ISO offsets in the READMEs are CORRECT as written; it is the")  # fixed explanatory text
    print("      wall-clock digits that are misleading, because the logging laptop ran")  # continued
    print("      on Finnish time (UTC+03:00) while the range ran on Norwegian time")  # continued
    print("      (UTC+02:00). Norwegian local time is therefore the README digits minus")  # continued
    print("      one hour, and no correction to the recorded instants is required.")  # continued

    remark = check_operator_remark(sessions, base)  # cross-check against the operator's stated local time
    print(f"\n      Cross-check against the 2.6.2 operator remark: {remark['detail']}")  # its one-line result

    # --- 6. phones --------------------------------------------------------
    phones = inventory_phones()  # one row per Android capture
    witness = phone_time_witness(sessions, phones, base)  # the phone containment test
    print("\n[5] PHONE CAPTURES - FOURTH INDEPENDENT TIME WITNESS")  # heading (a plain string: nothing to insert)
    # capture count, and the number of distinct device folders (set comprehension)
    print(f"      {len(phones)} Android GnssLogger captures across "
          f"{len({p['device_folder'] for p in phones})} devices.")  # f-string continued: how many distinct phones made those captures
    if witness["tz_stated"]:  # if: a phone file states its own time zone
        print(f"      Time zone stated inside the files: {witness['tz_stated']}")  # says which
        print("      This settles the phone side directly rather than by inference.")  # fixed text
    print("      Sessions fully contained in a phone capture window, per hypothesis:")  # sub-heading
    for p in witness["per_hypothesis"]:  # for loop over the four hypothesis scores
        mark = "  <== chosen" if p["hypothesis"] == base["tag"] else ""  # marker for the chosen hypothesis
        # sessions contained out of six, and covered minutes
        print(f"          {p['hypothesis']}: {p['sessions_fully_inside_a_phone_capture']}/6 "
              f"({p['total_covered_min']:.1f} covered min){mark}")  # f-string continued: minutes of session time the phones cover, with a mark on the winner

    # --- 7. session inventory --------------------------------------------
    sched = base["sched_by_code"]  # the Test Area 3 schedule lookup by test code
    inv_rows, sensor_rows = [], []  # tuple assignment: two empty lists for inventory rows and sensor rows
    for s in sessions:  # for loop building one inventory row per session
        code = s["test_code"]  # this session's test code
        cs = readme_instant(s["readme"]["start_raw"], base["readme_mode"])  # capture start, UTC, chosen reading
        ce = readme_instant(s["readme"]["end_raw"], base["readme_mode"])  # capture end, UTC, chosen reading
        sp = sched[code]  # this test's plan entry
        ss = sp["start_naive"].replace(tzinfo=base["plan_tz"]).astimezone(TZ_UTC)  # scheduled start in UTC
        se = sp["end_naive"].replace(tzinfo=base["plan_tz"]).astimezone(TZ_UTC)  # scheduled end in UTC

        cap_min = minutes(ce - cs)  # capture length in minutes
        sch_min = minutes(se - ss)  # scheduled length in minutes
        ov = overlap_minutes(cs, ce, ss, se)  # minutes of the test that were recorded
        before = round(max(0.0, (ss - cs).total_seconds()) / 60.0, 2)  # pre-test margin in minutes, never negative
        after = round(max(0.0, (ce - se).total_seconds()) / 60.0, 2)  # post-test margin in minutes, never negative
        full = cs <= ss and ce >= se  # boolean: recording began no later and ended no earlier than the test

        h, m, sec = (int(v) for v in s["readme"]["duration_raw"].split(":"))  # "H:MM:SS" split into three ints
        stated_min = round(h * 60 + m + sec / 60.0, 2)  # the README's stated duration in minutes
        dur_ok = abs(stated_min - cap_min) <= 0.05  # boolean: agrees with End minus Start within 3 seconds
        if not dur_ok:  # if: they disagree
            # records the disagreement
            flag(f"{code}: README Duration {s['readme']['duration_raw']} ({stated_min} min) "
                 f"disagrees with End-Start ({cap_min} min)")  # f-string continued: the capture length from its own start and end times

        ts_min = None  # span of the 1 Hz clock log, unknown until computed
        if s["ts"]["first"] and s["ts"]["last"]:  # if: both first and last clock rows exist
            t0 = datetime.fromisoformat(s["ts"]["first"])  # first clock sample as a datetime
            t1 = datetime.fromisoformat(s["ts"]["last"])  # last clock sample
            ts_min = minutes(t1 - t0)  # the clock log's span in minutes
            if abs(ts_min - cap_min) > 0.5:  # if: it differs from the README duration by more than 30 s
                # records the disagreement
                flag(f"{code}: 1 Hz system-timestamp span ({ts_min} min) differs from the "
                     f"README capture duration ({cap_min} min) by more than 30 s")  # f-string continued: the README duration it is compared with, and the 30 s tolerance

        cat = s["cat"]  # this test's catalogue fields
        pw_plan = sp["power_w_plan"]  # the plan's stated power, or None
        agree = None  # unknown unless both sources give a value
        if pw_plan is not None and cat["max_power_w"] is not None:  # if: both the plan and the catalogue state a power
            agree = abs(pw_plan - cat["max_power_w"]) < 1e-9  # boolean: equal up to floating-point rounding
            if not agree:  # if: they differ (this is how 2.6.10's 0.001 W against 0-0.1 W is caught)
                # records both values rather than choosing one
                flag(f"{code}: transmission plan states {pw_plan} W but the catalogue gives "
                     f"min {cat['min_power_w']} W / max {cat['max_power_w']} W")  # f-string continued: the catalogue's minimum and maximum power for the same test

        if not full:  # if: the test was only partly recorded
            missing = []  # empty list of "missing" phrases
            if cs > ss:  # if: recording began late
                missing.append(f"{minutes(cs - ss):.1f} min missing at the start")  # how much
            if ce < se:  # if: recording stopped early
                missing.append(f"{minutes(se - ce):.1f} min missing at the end")  # how much
            # records the partial coverage, joining the phrases with "; "
            flag(f"{code}: capture does not cover the whole scheduled test - " + "; ".join(missing))
        if before == 0:  # if: no recording before the test
            flag(f"{code}: no pre-test margin; there is no clean data before the scheduled start")  # recorded
        if after == 0:  # if: no recording after the test
            flag(f"{code}: no post-test margin; there is no clean data after the scheduled end")  # recorded

        for ev in s["events"]["problem"]:  # for loop over non-routine session events
            flag(f"{code}: non-routine session event at {ev['ts_raw']} - {ev['text']}")  # each one recorded

        # Plain substring test: flags "meaconing" or "jamming" in the operator remark when the catalogue
        # test name and description lack the word; the group description is not searched and negations
        # ("no jamming") also count. It only adds to the printed anomaly list, never to a file.
        rem = s["readme"]["remarks"].lower()  # the remark in lower case, so the word test ignores capitals
        for word in ("meaconing", "jamming"):  # for loop over the two interference words checked
            # if: the word is in the remark but in neither the test name nor its description
            if word in rem and word not in (cat["test_name"] + " " + cat["test_description"]).lower():
                # records the mismatch, quoting the catalogue name cut to 70 characters
                flag(f"{code}: the operator remark describes the test as '{word}' but neither "
                     f"the catalogue test name nor its description mentions {word} "  # f-string continued: the word is missing from the catalogue's name and description of the test
                     f"(catalogue calls it: {cat['test_name'][:70]})")  # f-string continued: the catalogue's own name for the test, cut to 70 characters

        inv_rows.append({  # appends the session's inventory row; the keys become CSV columns
            "session_folder": s["folder"], "test_code": code,  # dict entry: the session folder and its test code
            "test_name": cat["test_name"], "test_group": cat["group_name"],  # dict entry: the catalogue's test name and group
            "interference_type": cat["type"],  # dict entry: jamming, spoofing or meaconing, from the catalogue
            "capture_start_utc": fmt_utc(cs), "capture_end_utc": fmt_utc(ce),  # dict entry: capture start and end in UTC, the project's time base
            "capture_start_local_no": fmt_no(cs), "capture_end_local_no": fmt_no(ce),  # dict entry: capture start and end in Norwegian local time, for reading against the plan
            "capture_start_raw_readme": s["readme"]["start_raw"],  # dict entry: the README start time exactly as written
            "capture_end_raw_readme": s["readme"]["end_raw"],  # dict entry: the README end time exactly as written
            "capture_duration_min": cap_min,  # dict entry: capture length in minutes from start and end
            "duration_readme_stated": s["readme"]["duration_raw"],  # dict entry: the duration the README states
            "duration_matches_readme": dur_ok,  # dict entry: whether the two durations agree
            "duration_min_from_1hz_timestamps": ts_min,  # dict entry: the capture length measured from the 1 Hz system time stamps
            "n_1hz_timestamp_rows": s["ts"]["n_rows"],  # dict entry: how many 1 Hz time-stamp rows the session has
            "scheduled_start_utc": fmt_utc(ss), "scheduled_end_utc": fmt_utc(se),  # dict entry: scheduled start and end of the test in UTC
            "scheduled_start_local_no": fmt_no(ss), "scheduled_end_local_no": fmt_no(se),  # dict entry: the same schedule in Norwegian local time
            "scheduled_duration_min": sch_min,  # dict entry: scheduled length of the test in minutes
            "overlap_min": ov,  # dict entry: minutes during which capture and test overlap
            "minutes_before_scheduled_start": before,  # dict entry: minutes captured before the test started, a candidate clean margin
            "minutes_after_scheduled_end": after,  # dict entry: minutes captured after the test ended, a candidate clean margin
            "capture_covers_full_test": full,  # dict entry: whether the capture spans the whole scheduled test
            "partial_coverage_flag": not full,  # dict entry: the opposite flag, set when part of the test was not captured
            "coverage_fraction_of_test": round(ov / sch_min, 4) if sch_min else None,  # recorded share of the test
            "power_w_plan": pw_plan,  # dict entry: the power the transmission plan states
            "power_w_catalog_min": cat["min_power_w"],  # dict entry: the catalogue's minimum power
            "power_w_catalog_max": cat["max_power_w"],  # dict entry: the catalogue's maximum power
            "power_sources_agree": agree,  # dict entry: whether the plan and the catalogue agree on the power
            "bands": cat["bands"], "equipment": cat["equipment"],  # dict entry: the affected bands and the transmitting equipment
            "n_sensors": len(s["readme"]["sensors"]),  # dict entry: how many sensors the README lists
            "n_session_event_lines": s["events"]["n_lines"],  # dict entry: lines in the session's event log
            "n_non_startstop_events": len(s["events"]["other"]),  # dict entry: events other than the start and stop lines
            "n_non_routine_events": len(s["events"]["problem"]),  # dict entry: events that are not routine and may point to a problem
            "non_routine_events": " | ".join(e["text"] for e in s["events"]["problem"]),  # joined into one cell
            "n_stream_gap_lines": s["n_gaps"],  # dict entry: lines in the stream-gap log, one per gap
            "stream_gap_threshold": s["readme"]["gap_threshold"],  # dict entry: the logger's threshold for recording a gap
            "remarks_full": s["readme"]["remarks"],  # dict entry: the operator's remarks in full
            # keys starting with "_" keep the datetime objects for the figures; they are dropped before writing the CSV
            "_cap_start": cs, "_cap_end": ce, "_sched_start": ss, "_sched_end": se,
        })

        for d in s["readme"]["sensors"]:  # for loop over the README's instrument rows
            # appends a row; "**" unpacks a dict comprehension that turns column names into snake_case keys
            sensor_rows.append({"session_folder": s["folder"], "test_code": code,
                                **{k.strip().lower().replace(" ", "_"): v for k, v in d.items()}})  # dict unpacking: adds the sensor's README columns, with names turned into lower-case keys

    inv = pd.DataFrame(inv_rows)  # the session inventory as a table

    # --- 8. phases --------------------------------------------------------
    phase_rows = []  # empty list for every test's phase rows
    for s in sessions:  # for loop over the sessions
        phase_rows.extend(extract_phases(s["test_code"], s["cat"]))  # list.extend adds all of a test's rows at once
    phases = pd.DataFrame(phase_rows)  # the phase table

    # --- 9. concurrent transmissions --------------------------------------
    for p in plan:  # for loop giving every scheduled entry at every test area UTC times and a power
        # attaches the chosen plan time zone to the naive start and converts to UTC
        p["_start_utc"] = p["start_naive"].replace(tzinfo=base["plan_tz"]).astimezone(TZ_UTC)
        p["_end_utc"] = p["end_naive"].replace(tzinfo=base["plan_tz"]).astimezone(TZ_UTC)  # the same for the end
        cat = catalog.get(p["test_code"])  # dict.get: the catalogue entry, or None if the code is not catalogued
        # tuple unpacking: power, source and field from the one power rule, effective_power()
        p["power_w_effective"], p["power_source"], p["power_field"] = effective_power(p, cat)
        p["bands"] = cat["bands"] if cat else ""  # conditional expression: catalogue bands, or "" if uncatalogued
        p["catalog_name"] = cat["test_name"] if cat else ""  # the catalogue name, or ""
        if cat is None:  # if: the scheduled test is missing from the catalogue
            # recorded as an anomaly
            flag(f"scheduled test {p['test_code']} (Test Area {p['test_area']}) has no entry "
                 "in testcatalog.json")
        # elif: an active test whose catalogue maximum is written as 0 W (stored as not stated above) is flagged
        elif (cat["max_power_zero"] and p["power_w_plan"] is None
              # continuation: and only for an active test type
              and cat["type"] in ("Jamming", "Spoofing", "Meaconing")):
            # recorded as an anomaly; .lower() writes the type in lower case
            flag(f"scheduled {cat['type'].lower()} test {p['test_code']} (Test Area "
                 f"{p['test_area']}) carries a stated maximum power of 0 W in the catalogue, "  # f-string continued: the test area and the impossible 0 W power stated in the catalogue
                 "which cannot be right for an active transmission")

    conc_rows = []  # empty list for the concurrent-transmission rows
    for _, r in inv.iterrows():  # for loop over our six sessions
        cands = []  # empty list: tests overlapping this capture
        for p in plan:  # for loop over every scheduled entry
            if p["test_area"] == 3 and p["test_code"] == r["test_code"]:  # if: it is the session's own test
                continue  # our own test is not a concurrent transmission; skip it
            # overlap of the capture with this entry, both in UTC
            ov = overlap_minutes(r["_cap_start"], r["_cap_end"], p["_start_utc"], p["_end_utc"])
            if ov <= 0:  # if: no overlap
                continue  # skip it
            cands.append({  # appends a row describing the concurrent test and its overlap
                "our_session_folder": r["session_folder"], "our_test_code": r["test_code"],  # dict entry: the captured session's folder and test code
                "concurrent_test_area": p["test_area"], "concurrent_test_code": p["test_code"],  # dict entry: the area and code of the test running at the same time
                "concurrent_title": p["title"] or p["catalog_name"],  # "or": the plan title, else the catalogue name
                "concurrent_start_utc": fmt_utc(p["_start_utc"]),  # dict entry: start of the concurrent test in UTC
                "concurrent_end_utc": fmt_utc(p["_end_utc"]),  # dict entry: end of the concurrent test in UTC
                "concurrent_start_local_no": fmt_no(p["_start_utc"]),  # dict entry: start of the concurrent test in Norwegian local time
                "concurrent_end_local_no": fmt_no(p["_end_utc"]),  # dict entry: end of the concurrent test in Norwegian local time
                "power_w": p["power_w_effective"], "power_source": p["power_source"],  # dict entry: the concurrent test's power and where that figure came from
                "power_field": p["power_field"], "bands": p["bands"],  # dict entry: the field the power was read from, and the affected bands
                "overlap_min": ov,  # dict entry: minutes the concurrent test overlaps the capture
                # conditional expression over two lines: the overlap as a share of our capture, or None if zero length
                "overlap_fraction_of_our_capture": round(ov / r["capture_duration_min"], 4)
                if r["capture_duration_min"] else None,  # conditional expression: the share is left empty when the capture has zero length
            })
        # in-place sort, tuple key, minus = descending: power (unknown as -1, below all stated), overlap, then code
        cands.sort(key=lambda c: (-(c["power_w"] if c["power_w"] is not None else -1.0),
                                  -c["overlap_min"], c["concurrent_test_code"]))  # tuple element: then longest overlap first, then the test code as a tie-breaker
        for k, c in enumerate(cands, start=1):  # enumerate from 1: k is the rank
            c["power_rank_within_session"] = k  # stores rank 1 = strongest concurrent emitter
        conc_rows.extend(cands)  # adds this session's rows to the full list
    conc = pd.DataFrame(conc_rows)  # the concurrent-transmission table

    # --- 10. figures ------------------------------------------------------
    fig1 = GRAPH / "session_vs_schedule.png"  # output path of Figure 1
    fig2 = GRAPH / "test_area_activity.png"  # output path of Figure 2
    figure_one(inv, fig1)  # draws and saves Figure 1
    f2meta = figure_two(plan, inv, fig2)  # draws and saves Figure 2, keeping the facts it returns
    print(f"\n[6] Figures written to {GRAPH.name}/ at 300 dpi:")  # Path.name is the folder's last part
    print(f"      {fig1.name}")  # Figure 1's file name
    # Figure 2's file name and how many entries it shows
    print(f"      {fig2.name}  ({f2meta['n_shown']} scheduled entries overlapping "
          f"{FIG2_WINDOW[0]}-{FIG2_WINDOW[1]})")  # f-string continued: the time window the figure shows

    # --- 11. write CSVs ---------------------------------------------------
    inv_out = inv.drop(columns=[c for c in inv.columns if c.startswith("_")])  # drops the helper "_" columns
    inv_out = inv_out.sort_values("capture_start_utc")  # fixed row order: by capture start
    # writes the CSV without the index column and with "\n" line endings, so the bytes are the same on every system
    inv_out.to_csv(PROCESSED / "ground_truth" / "session_ground_truth.csv", index=False, lineterminator="\n")

    # the instrument table as a CSV, in the order the rows were collected
    pd.DataFrame(sensor_rows).to_csv(PROCESSED / "ground_truth" / "session_sensors.csv", index=False, lineterminator="\n")

    # double-bracket indexing selects these columns in this fixed order
    phases = phases[["test_code", "phase_name", "offset_start_min", "offset_end_min", "bands",
                     "offset_basis", "source_field", "source_sentence", "applies_by_default",
                     "overridden_by_test", "override_sentence"]]
    # method chaining: sort by three columns, then write the phase CSV
    phases.sort_values(["test_code", "source_field", "phase_name"]).to_csv(
        PROCESSED / "ground_truth" / "test_phases.csv", index=False, lineterminator="\n")  # argument of to_csv(): the output path, without the index and with Unix line ends

    # sort by session, then by power rank, and write the concurrent-transmission CSV
    conc.sort_values(["our_test_code", "power_rank_within_session"]).to_csv(
        PROCESSED / "ground_truth" / "concurrent_transmissions.csv", index=False, lineterminator="\n")  # argument of to_csv(): the output path, without the index and with Unix line ends

    # dict comprehension inside a list comprehension: each phone row without its "_" helper keys
    phone_out = pd.DataFrame([{k: v for k, v in p.items() if not k.startswith("_")} for p in phones])
    # sort by device and start time, and write the phone inventory CSV
    phone_out.sort_values(["device_folder", "start_utc"]).to_csv(
        PROCESSED / "ground_truth" / "phone_capture_inventory.csv", index=False, lineterminator="\n")  # argument of to_csv(): the output path, without the index and with Unix line ends

    # writes the four-hypothesis score table
    hyp_df.to_csv(PROCESSED / "ground_truth" / "time_base_hypotheses.csv", index=False, lineterminator="\n")

    # --- 12. plain-language report ----------------------------------------
    print("\n" + "=" * 78)  # a blank line, then a rule of 78 "=" characters
    print("PLAIN-LANGUAGE REPORT")  # report heading
    print("=" * 78)  # closing rule

    total_cap = inv["capture_duration_min"].sum()  # pandas column sum: total minutes recorded
    # // is floor division (whole hours), % is the remainder (leftover minutes)
    print(f"\nTotal capture time across all six sessions: {total_cap:.2f} minutes "
          f"({int(total_cap // 60)} h {total_cap % 60:.1f} min).")  # f-string continued: the same total as whole hours and leftover minutes

    print("\nPer-session duration, coverage and margins:")  # sub-heading
    for _, r in inv.sort_values("_cap_start").iterrows():  # for loop over sessions in capture order
        # conditional expression: a phrase for full coverage, a loud one for partial coverage
        cov = "covers the whole scheduled test" if r["capture_covers_full_test"] else "PARTIAL COVERAGE"
        print(f"\n  {r['test_code']}  ({r['session_folder']})")  # test code and folder
        # [11:] slices off the date, leaving HH:MM:SS of the Norwegian local capture window
        print(f"      capture   {r['capture_start_local_no'][11:]} - {r['capture_end_local_no'][11:]} "
              f"Norwegian local, {r['capture_duration_min']:.2f} min")  # f-string continued: the capture length in minutes, labelled as Norwegian local time
        # the scheduled window, printed the same way
        print(f"      scheduled {r['scheduled_start_local_no'][11:]} - {r['scheduled_end_local_no'][11:]} "
              f"Norwegian local, {r['scheduled_duration_min']:.2f} min")  # f-string continued: the scheduled length in minutes, labelled as Norwegian local time
        # overlap in minutes and as a percentage of the scheduled test
        print(f"      overlap   {r['overlap_min']:.2f} min "
              f"({100 * r['coverage_fraction_of_test']:.1f} % of the scheduled test) - {cov}")  # f-string continued: the overlap as a percentage of the scheduled test, then the coverage verdict
        # the two margins, the only candidate clean data
        print(f"      margins   {r['minutes_before_scheduled_start']:.2f} min before the test, "
              f"{r['minutes_after_scheduled_end']:.2f} min after")  # f-string continued: minutes captured after the test

    print("\nStrongest concurrent emitter during each capture:")  # sub-heading
    for _, r in inv.sort_values("_cap_start").iterrows():  # for loop over sessions in capture order
        sub = conc[conc["our_test_code"] == r["test_code"]]  # boolean mask: this session's concurrent rows
        if sub.empty:  # if: no rows (DataFrame.empty is True for a table with no rows)
            print(f"  {r['test_code']}: no concurrent scheduled transmission found.")  # says so
            continue  # next session
        top = sub.sort_values("power_rank_within_session").iloc[0]  # iloc[0]: the first row, i.e. rank 1
        # conditional expression: pd.isna detects a missing power; ":g" prints the number compactly
        pw = "power not stated" if pd.isna(top["power_w"]) else f"{top['power_w']:g} W"
        # one line: code, area, power and source, overlap in minutes and as a share of the capture
        print(f"  {r['test_code']}: {top['concurrent_test_code']} at Test Area "
              f"{top['concurrent_test_area']}, {pw} ({top['power_source']}), "  # f-string continued: the concurrent test's area, its power and where that power figure came from
              f"overlapping {top['overlap_min']:.1f} min "  # f-string continued: the minutes it overlaps the capture
              f"({100 * top['overlap_fraction_of_our_capture']:.0f} % of the capture)")  # f-string continued: the overlap as a percentage of the capture
        print(f"        {top['concurrent_title'][:96]}")  # its title, cut to 96 characters

    print("\nSession events other than START and STOP:")  # sub-heading
    for s in sorted(sessions, key=lambda x: x["readme"]["start_raw"]):  # sessions sorted by their README start text
        ev = s["events"]  # this session's grouped events
        if not ev["problem"]:  # if: no non-routine events
            # the wording always says "line" (singular), whatever the count in brackets is
            print(f"  {s['test_code']}: none beyond the routine OBD-II ready handshake "
                  f"({len(ev['routine'])} line).")  # f-string continued: how many routine lines were seen
            continue  # next session
        print(f"  {s['test_code']}: {len(ev['problem'])} non-routine event(s)")  # how many
        for e in ev["problem"]:  # for loop over them
            print(f"        {e['ts_raw']}  {e['text']}")  # timestamp and text

    print("\nStream continuity: gap threshold and recorded gaps per session:")  # sub-heading
    for _, r in inv.sort_values("_cap_start").iterrows():  # for loop over sessions in capture order
        # gap count (int() because pandas stores it as a float) and the logger's threshold
        print(f"  {r['test_code']}: {int(r['n_stream_gap_lines'])} gap(s) recorded "
              f"(threshold {r['stream_gap_threshold']})")  # f-string continued: the logger's gap threshold

    print("\nOperator remarks, verbatim:")  # sub-heading
    for _, r in inv.sort_values("_cap_start").iterrows():  # for loop over sessions in capture order
        print(f"  {r['test_code']}: {r['remarks_full']}")  # the remark exactly as written

    print("\nPhone captures (inventory):")  # sub-heading
    for p in sorted(phones, key=lambda x: (x["device_folder"], x["start_utc"])):  # sorted by device, then start
        # format specs: ">12" right-aligns in 12 characters, "6.2f" is 6 wide with 2 decimals; sizes shown in MB
        print(f"  {p['model']:>12}  session {p['logger_session_id']:>2}  "
              f"{p['start_utc'][11:]} - {p['end_utc'][11:]} UTC  "  # f-string continued: the phone capture's start and end in UTC
              f"{p['duration_min_computed']:6.2f} min  "  # f-string continued: the capture length in minutes, six characters wide
              f"json {p['json_size_bytes'] / 1e6:5.1f} MB  zip {p['zip_size_bytes'] / 1e6:4.1f} MB  "  # f-string continued: sizes of the JSON log and the zip archive in megabytes
              f"contents: .{p['zip_content_extensions']}")  # f-string continued: the file types inside the archive

    print("\nPhone alignment against the derived time base:")  # sub-heading
    for r in witness["rows"]:  # for loop over the per-session alignment rows
        # share of the session inside the best phone capture, which device, its window, and which started first
        print(f"  {r['test_code']}: {r['fraction_covered'] * 100:5.1f} % of the session lies inside "
              f"{r['phone_device'].split('_')[0]} session "  # f-string continued: the phone model, from the device folder's name
              f"{r['phone_start_utc'].strftime('%H:%M:%S')}-{r['phone_end_utc'].strftime('%H:%M:%S')} UTC; "  # f-string continued: start and end of the phone capture in UTC
              f"phone started {abs(r['start_offset_s']):.0f} s "  # f-string continued: how many seconds apart the phone and laptop started
              f"{'after' if r['start_offset_s'] < 0 else 'before'} the laptop")  # f-string continued: whether the phone started after or before the laptop
    print(f"\n  Phone pairs started within 2 minutes of each other: {len(witness['pairs'])}")  # count of pairs
    for pr in witness["pairs"]:  # for loop over those pairs
        print(f"      {pr['delta_s']:.1f} s apart")  # the gap between their starts

    print("\n" + "-" * 78)  # a rule of "-" characters
    print("TIME-BASE AGREEMENT ACROSS FOUR INDEPENDENT SOURCES")  # summary heading
    print("-" * 78)  # closing rule
    print("  1. README ISO offsets           : +03:00, Finnish summer time - correct as written")  # fixed text, plain string
    # source 2: the winning hypothesis and its margin over the runner-up
    print(f"  2. Transmission plan alignment  : {base['tag']} wins with "
          f"{base['best']['total_overlap_min']:.1f} overlap minutes against "  # f-string continued: the winner's total overlap minutes
          f"{base['runner']['total_overlap_min']:.1f} for the runner-up")  # f-string continued: the runner-up's total, for comparison
    # source 3: the operator remark's verdict and detail
    print(f"  3. Operator remark (2.6.2)      : {'AGREES' if remark['agrees'] else 'DOES NOT AGREE'} "
          f"- {remark['detail']}")  # f-string continued: the remark check's one-line detail
    # source 4: the phones' verdict, printed over three lines
    print(f"  4. Android phone captures       : {'AGREES' if witness['agrees'] else 'DOES NOT AGREE'} "
          f"- the phone files state UTC explicitly; under the chosen reading of the")
    # the chosen reading's containment count out of six
    print(f"                                    README timestamps "
          f"{witness['by_mode'].get(witness['chosen_mode'])}/6 sessions fall inside a phone "  # f-string continued: sessions inside a phone capture under the chosen README reading
          f"capture, against")
    # the rival reading's count: max over a generator of the other readings' counts
    print(f"                                    "
          f"{max(v for k, v in witness['by_mode'].items() if k != witness['chosen_mode'])}/6 "  # f-string continued: the best count under any other reading, for comparison
          f"under the rival reading")
    all_agree = bool(remark["agrees"]) and bool(witness["agrees"])  # bool() turns a None into False
    print(f"\n  ALL FOUR SOURCES AGREE: {'YES' if all_agree else 'NO'}")  # the overall verdict

    print("\n" + "-" * 78)  # rule
    print(f"ANOMALIES AND THINGS THAT LOOK WRONG ({len(ANOMALIES)})")  # heading with the anomaly count
    print("-" * 78)  # rule
    if not ANOMALIES:  # if: the list is empty
        print("  none")  # says so explicitly
    for a in ANOMALIES:  # for loop over every flagged anomaly, in the order raised
        print(f"  - {a}")  # prints it as a bullet

    print("\n" + "=" * 78)  # rule
    print("Outputs written:")  # heading
    # list: exactly the six CSVs and two figures this script writes (other sections' files are not listed)
    mine = [PROCESSED / n for n in ("ground_truth/session_ground_truth.csv", "ground_truth/session_sensors.csv", "ground_truth/test_phases.csv",
                                    "ground_truth/concurrent_transmissions.csv", "ground_truth/phone_capture_inventory.csv",
                                    # continuation: the last CSV name, then "+" appends the two figure paths
                                    "ground_truth/time_base_hypotheses.csv")] + [fig1, fig2]
    for f in mine:  # for loop over this script's outputs
        print(f"  {f.relative_to(PROJECT)}")  # relative_to prints the path from the project folder
    print(f"\nRandom seed fixed at {SEED} (no randomness is used, but it is set and recorded).")  # seed note
    print("=" * 78)  # closing rule


if __name__ == "__main__":  # if: true only when the file is run directly, not when another script imports it
    main()  # runs the section
