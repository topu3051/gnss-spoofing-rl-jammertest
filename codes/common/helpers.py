#!/usr/bin/env python3
# module docstring below: the file's help text, describing what this module holds and who uses it
"""
Helpers shared by the handset and receiver-status scripts and the scripts after them (one definition each).

These functions used to be copied into 1_ground_truth/02_handset_check.py and
1_ground_truth/03_receiver_status.py, and several copies had drifted apart. Each
now exists once, here. the handset script imports from this module; 03 still holds its
own copies until it is switched over in 

Where the copies differed, the version kept is stated at the function:
  die, save_fig       identical copies
  haversine_m         the handset script's version (metres); the receiver-status script's haversine_km is the same formula in km
  iqr                 the handset script's body with the receiver-status script's conversion to a float array
  cliffs_delta        new: exact, NaN dropped first, no sub-sampling
  time_axis           the handset script's and the receiver-status script's are the same logic
  shade_windows       the handset script's version (the receiver-status script's shade_50w is a copy)
  parse_pyramid       the handset script's version with all its checks; the prediction is keyed by plan band name
  load_schedule       the handset script's build_schedule, with the plan time zone read from the timeline script
                      and the power from the timeline script's effective_power()
  tag_times           the handset script's version with its fall-through, plus the excluded unknown-power group (26)
"""

from __future__ import annotations  # future import: hints kept as text, so "float | None" works on older Pythons

import math  # standard-library maths: radians, sin, cos, asin, sqrt
import os  # standard-library operating-system access: sets matplotlib's cache folder before it is imported
import re  # standard-library regular expressions, used on the 1.8.2 comment
import sys  # standard-library system access: stderr and exit, used by die()
from datetime import datetime, timedelta, timezone  # date-time classes: instants, durations, the UTC zone
from pathlib import Path  # Path objects for building file paths
from zoneinfo import ZoneInfo  # time-zone database lookup, for the plan's time zone named by the timeline script

import numpy as np  # third-party numerical library: arrays, sorting, percentiles
import pandas as pd  # third-party table library: timestamps and the timeline script's CSV

# Path: codes/.mplconfig, matplotlib's configuration and font-cache folder inside the project folder
MPLCONFIG = Path(__file__).resolve().parent / ".mplconfig"
MPLCONFIG.mkdir(exist_ok=True)  # creates the folder if missing; exist_ok=True means no error when it exists
# os.environ.setdefault: sets MPLCONFIGDIR unless the importing script already did, before matplotlib loads
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG))
import matplotlib.dates as mdates  # matplotlib's date helpers: datetime <-> plot number, time-axis ticks
import matplotlib.pyplot as plt  # plotting interface, used here only to close a saved figure

TZ_UTC = timezone.utc  # the UTC time-zone object, the internal time base
# the GPS time origin: SEPT_DOC 4.1.3 gives WNc 0, TOW 0 as 6 January 1980 00:00:00 UTC
GPS_EPOCH = datetime(1980, 1, 6, tzinfo=TZ_UTC)
# GPS minus UTC on 14 September 2026, in seconds. Not stated in docs/; verified in the data: the receiver-status script reads the
# leapS field of every UBX-RXM-RAWX message in the 2.6.12 recordings and stops if it differs
LEAP_SECONDS = 18
C_GREY, C_VERMILLION = "#999999", "#D55E00"  # tuple assignment: Okabe-Ito grey (grids) and vermillion (50 W shading)

# Exposure-group labels shared by the tagging and every comparison 
G50 = "50 W at Test Area 1"  # a Test Area 1 test of 40 W or more is on air
GHI = "other above 1 W (not 50 W at TA1)"  # a test above 1 W is on air, but no 50 W Test Area 1 test
GBL = "baseline (nothing above 1 W)"  # every test on air has a stated power of 1 W or less (or nothing is on air)
GUNK = "unknown power only (excluded)"  # nothing known above 1 W, but a test whose power no document states is on air


def die(msg: str) -> None:  # function: stops the script with a readable message (identical in 01-04)
    print("\n*** STOPPING: " + msg, file=sys.stderr)  # prints to the error stream
    sys.exit(1)  # ends the program with status 1, meaning failure


def haversine_m(lat1, lon1, lat2, lon2) -> float:  # function: great-circle distance in metres between two points
    r = 6371008.8  # the Earth's mean radius in metres (a sphere is assumed)
    p1, p2 = math.radians(lat1), math.radians(lat2)  # tuple assignment: both latitudes converted to radians
    dp, dl = p2 - p1, math.radians(lon2 - lon1)  # the latitude and longitude differences, in radians
    # the haversine formula: h is the squared half-chord length between the points on a unit sphere
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))  # turns h into the arc length along the Earth's surface


# function: interquartile range = 75th minus 25th percentile, the spread of the middle half of the data
def iqr(a) -> float:
    a = np.asarray(a, float)  # np.asarray: lists, Series or arrays all become a float array (the receiver-status script's safeguard)
    a = a[np.isfinite(a)]  # boolean mask: keeps finite values only (/: one NaN or inf used to turn the spread into NaN)
    return float(np.percentile(a, 75) - np.percentile(a, 25)) if len(a) else float("nan")  # np.percentile(a, 75): the value 75 % of readings lie below (extreme readings barely move it); NaN for an empty set


# function: Cliff's delta, exactly, with no sub-sampling.
# Definition: delta = (#{(i, j): a_i > b_j} - #{(i, j): a_i < b_j}) / (n_a * n_b), over every pair of one a and one b;
# a tie (a_i = b_j) counts in neither count, i.e. as zero, which is the standard definition. Range -1 to +1;
# negative means a tends below b. NaN values are dropped first. Computed by sorting b once and, for every a,
# counting with searchsorted how many b lie strictly below it and how many lie at or below it.
def cliffs_delta(a, b) -> float:
    a = np.asarray(a, float)  # the first group as a float array
    b = np.asarray(b, float)  # the second group as a float array
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]  # boolean masks ("~" negates isnan) drop NaN before anything else
    if len(a) == 0 or len(b) == 0:  # if: either group is empty, no pairs exist
        return float("nan")  # undefined
    bs = np.sort(b)  # b sorted ascending, which searchsorted needs
    # searchsorted(side="left"): for each a, how many b are strictly smaller; summed = pairs with a > b
    greater = int(np.searchsorted(bs, a, side="left").sum())
    # side="right" counts b <= a; n_b minus that = b strictly larger; summed = pairs with a < b
    less = int((len(bs) - np.searchsorted(bs, a, side="right")).sum())
    return float((greater - less) / (len(a) * len(b)))  # the signed effect size; never made unsigned


def save_fig(fig, path: Path) -> None:  # function: saves a figure so runs again give byte-identical files
    # 300 dpi, trimmed to the drawn content; metadata None removes the software and creation-time stamps from the PNG
    fig.savefig(path, dpi=300, bbox_inches="tight",
                # keyword argument metadata: blanks the PNG's Software and Creation Time fields, so runs again are byte-identical
                metadata={"Software": None, "Creation Time": None})
    plt.close(fig)  # frees the figure's memory


def time_axis(ax, t0, t1) -> None:  # function: formats a UTC time axis, choosing tick spacing by span
    ax.set_xlim(mdates.date2num(t0), mdates.date2num(t1))  # date2num: datetimes -> plot numbers; sets the x-range
    span_min = (pd.Timestamp(t1) - pd.Timestamp(t0)).total_seconds() / 60.0  # plotted span in minutes
    if span_min <= 100:  # if: a short span
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))  # a labelled tick every 5 minutes
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=1))  # a small tick every minute
    else:  # else: a long span
        ax.xaxis.set_major_locator(mdates.HourLocator())  # a labelled tick every hour
        ax.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[0, 15, 30, 45]))  # small ticks each quarter hour
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=TZ_UTC))  # tick labels as HH:MM in UTC
    ax.grid(axis="x", which="major", color=C_GREY, alpha=0.30, lw=0.6)  # faint vertical lines at major ticks
    ax.grid(axis="x", which="minor", color=C_GREY, alpha=0.12, lw=0.4)  # fainter lines at minor ticks


# function: shades the windows of 50 W Test Area 1 tests (power >= fifty_w_min); True if any were drawn
def shade_windows(ax, plan, t0, t1, fifty_w_min: float) -> bool:
    drew = False  # nothing drawn yet
    for p in plan:  # for loop over every scheduled entry
        if p["power_w"] is None or p["power_w"] < fifty_w_min or p["test_area"] != 1:  # if: not a 50 W TA1 test
            continue  # skip it
        s, e = pd.Timestamp(p["start_utc"]), pd.Timestamp(p["end_utc"])  # its window as pandas timestamps
        if e <= t0 or s >= t1:  # if: entirely outside the plotted range
            continue  # skip it
        # vertical band across the axes, clipped to the plotted range; zorder 0 keeps it behind the data
        ax.axvspan(mdates.date2num(max(s, t0)), mdates.date2num(min(e, t1)),
                   # continuation of the call: vermillion, 18 % opaque, behind the data (zorder 0), no edge line
                   color=C_VERMILLION, alpha=0.18, zorder=0, lw=0)
        drew = True  # remembers that something was drawn
    return drew  # True if at least one window was shaded


# function: x and y arrays for a line plot with NaN inserted wherever consecutive times are more than max_gap_s
# apart, so matplotlib breaks the line there instead of drawing a straight segment across missing data
def break_gaps(times, values, max_gap_s: float = 5.0):
    t = pd.to_datetime(pd.Series(times)).reset_index(drop=True)  # the times as a fresh pandas Series
    v = pd.Series(values, dtype=float).reset_index(drop=True)  # the values as a float Series, same positions
    gap = t.diff().dt.total_seconds() > max_gap_s  # True where the step from the previous reading is too long
    x, y = [], []  # output lists: plot numbers and values, with NaN breaks
    for i in range(len(t)):  # for loop over the readings in order
        if gap.iloc[i]:  # if: a gap precedes this reading
            x.append(np.nan); y.append(np.nan)  # two statements: a NaN point that breaks the line
        x.append(mdates.date2num(t.iloc[i])); y.append(v.iloc[i])  # two statements: the reading itself
    return np.array(x), np.array(y)  # NumPy arrays for ax.plot


def plan_timezone(processed: Path) -> ZoneInfo:  # function: the plan's time zone as derived by the timeline script
    hyp = pd.read_csv(processed / "ground_truth" / "time_base_hypotheses.csv")  # the timeline script's four scored interpretations
    ok = hyp[hyp.sessions_with_zero_overlap == 0]  # boolean mask: interpretations placing every session in its test
    if len(ok) != 1:  # if: none or several qualify, the time base is not settled
        die(f"ground_truth/time_base_hypotheses.csv names {len(ok)} interpretations with no zero-overlap session, not one")  # stop
    return ZoneInfo(ok.iloc[0]["plan_timezone"])  # e.g. Europe/Oslo, taken from the timeline script rather than typed here


# function: every scheduled entry of the day from the timeline script's parser, with UTC times, the power from the timeline script's rule and
# the catalogue bands; returns (plan, catalog). s1 is the timeline script loaded as a module (the handset script's build_schedule).
def load_schedule(s1, processed: Path):
    plan, _ = s1.parse_transmission_plan()  # the timeline script's parser: the entries (the column geometry is not needed)
    catalog = s1.load_catalog()  # the timeline script's catalogue loader (a catalogue maximum of 0 W comes back as None)
    tz = plan_timezone(processed)  # the plan's time zone from the timeline script's derivation
    for p in plan:  # for loop over every scheduled entry at every test area
        p["start_utc"] = p["start_naive"].replace(tzinfo=tz).astimezone(TZ_UTC)  # attaches the zone, converts to UTC
        p["end_utc"] = p["end_naive"].replace(tzinfo=tz).astimezone(TZ_UTC)  # the end, converted likewise
        cat = catalog.get(p["test_code"])  # dict.get: the catalogue entry, or None if not catalogued
        # tuple unpacking: power, source and field from the timeline script's single rule; the field is not needed here
        p["power_w"], p["power_source"], _ = s1.effective_power(p, cat)
        p["bands"] = cat["bands"] if cat else ""  # conditional expression: catalogue bands, or ""
    return plan, catalog  # the schedule and the catalogue


# function: the 1.8.2 pyramid's step structure and prediction, parsed from the transmission plan's own comment.
# flag(msg) records an inconsistency; die(msg) stops. The prediction is a dict: plan band name -> steps jammed.
def parse_pyramid(plan, flag) -> dict:
    ent = [p for p in plan if p["test_code"] == "1.8.2"]  # list comprehension: every plan entry for test 1.8.2
    if len(ent) != 1:  # if: none, or more than one, which would make "the" pyramid ambiguous
        die(f"expected exactly one 1.8.2 entry in the transmission plan, found {len(ent)}")  # stop
    e = ent[0]  # the single 1.8.2 entry
    c = e["comment"]  # its free-text comment, where the step structure is written
    # regex over two joined raw strings: three groups capture the step length, jamming and break minutes
    m = re.search(r"Each step lasts (\d+) minutes with (\d+) minutes of jamming "
                  # second half of the regex (raw string, joined to the first): the break minutes; re.IGNORECASE ignores case
                  r"and a (\d+) minutes break", c, re.IGNORECASE)
    if not m:  # if: the sentence is not in the expected form
        die("the 1.8.2 comment does not state its step structure in the expected form")  # stop rather than guess
    step_min, jam_min, brk_min = (int(x) for x in m.groups())  # generator unpacked into three integers (5, 3, 2)
    if jam_min + brk_min != step_min:  # if: jamming plus break does not add up to the step length
        # records the inconsistency
        flag(f"1.8.2 step arithmetic does not close: {jam_min} min jamming plus "
             # continuation of the flag text: the break minutes and the step length that do not add up
             f"{brk_min} min break is not {step_min} min")
    tail = c.split("look like this:", 1)  # str.split at most once: the text before and after the phrase
    if len(tail) != 2:  # if: the phrase is missing, so there is no step list
        die("the 1.8.2 comment does not enumerate its steps after 'look like this:'")  # stop
    tokens = [t for t in re.split(r"[,\s]+", tail[1].strip()) if t]  # band names, split at commas and spaces
    steps: list[list[str]] = []  # annotated list of lists: one list of band names per step
    for t in tokens:  # for loop over the band names in order
        if t == "L1":  # if: every step of the pyramid begins with L1, so L1 opens a new step
            steps.append([t])  # starts a new step list holding L1
        elif steps:  # elif: a step is open (a non-empty list is truthy)
            steps[-1].append(t)  # steps[-1] is the last step; the band joins it
        else:  # else: a band before the first L1
            flag(f"1.8.2 step list begins with an unexpected token: {t}")  # recorded, not placed
    n = len(steps)  # number of steps recovered (15)
    sched_min = (e["end_utc"] - e["start_utc"]).total_seconds() / 60.0  # the scheduled window in minutes
    if abs(n * step_min - sched_min) > 1e-6:  # if: the steps do not fill the window exactly
        # records the mismatch; the arithmetic check guards against a wrong parse
        flag(f"1.8.2 pyramid arithmetic does not match its scheduled window: "
             # continuation of the flag text: steps x minutes against ...
             f"{n} steps of {step_min} min is {n * step_min:.0f} min against a scheduled "
             # ... the scheduled window in minutes
             f"{sched_min:.0f} min")
    rows = []  # empty list: one timing row per step
    for i, bands in enumerate(steps, start=1):  # enumerate from 1: step number i and its band list
        s = e["start_utc"] + timedelta(minutes=(i - 1) * step_min)  # the step's start: window start + whole steps
        # appends the step; jamming is read as its first jam_min minutes, the break as the rest
        rows.append({"step": i, "step_start_utc": s, "jam_start_utc": s,
                     # dict entries of one step: the end of its jamming minutes (start + jam_min)
                     "jam_end_utc": s + timedelta(minutes=jam_min),
                     # dict entry: the end of the step (start + step_min)
                     "step_end_utc": s + timedelta(minutes=step_min),
                     # dict entry: the step's band names joined with ';', then the dict and the append() call close
                     "bands": ";".join(bands)})
    names = sorted({t for st in steps for t in st})  # set comprehension: every band name in the step list, sorted
    # dict comprehension: band name -> the steps whose band list includes it (the prediction, stated before looking)
    pred = {nm: [r["step"] for r in rows if nm in r["bands"].split(";")] for nm in names}
    # returns the plan entry, the step table, the timing constants, the prediction and the source text
    return {"entry": e, "steps": rows, "n_steps": n, "step_min": step_min,
            # entries of the returned dict: jamming and break minutes and the prediction (band name -> steps)
            "jam_min": jam_min, "brk_min": brk_min, "predicted_by_band": pred,
            # entries of the returned dict: how many band names were parsed and the comment text they came from
            "n_tokens": len(tokens), "source_sentence": c}


# function: tags every distinct time with what was scheduled then; returns one row per time.
# Groups: a time at which no test of KNOWN power above 1 W is on air but a test
# of unknown power is, belongs to GUNK and is excluded from both the 50 W and the baseline comparison, because the
# unknown power could be high. A test of 40 W or more away from Test Area 1 falls in GHI (the handset script's fall-through).
def tag_times(ts, plan, sessions: pd.DataFrame, high_power_w: float, fifty_w_min: float) -> pd.DataFrame:
    t = pd.to_datetime(ts, utc=True)  # the epoch times as timezone-aware pandas timestamps
    uniq = pd.DatetimeIndex(sorted(set(t)))  # set() drops any repeat (phone times never coincide, check C10); sorted
    # list comprehension: (capture start, capture end, test code) for each of our six sessions
    sess = [(pd.Timestamp(r["capture_start_utc"], tz="UTC"),
             # continuation of the tuple: the capture end in UTC and the session's test code
             pd.Timestamp(r["capture_end_utc"], tz="UTC"), r["test_code"])
            # the comprehension's loop: one tuple per session row (iterrows yields index and row)
            for _, r in sessions.iterrows()]
    recs = []  # empty list of tag rows
    for u in uniq:  # for loop over the distinct epoch times
        # list comprehension with a chained comparison: tests scheduled at u (start inclusive, end exclusive)
        active = [p for p in plan if pd.Timestamp(p["start_utc"]) <= u < pd.Timestamp(p["end_utc"])]
        known = [p for p in active if p["power_w"] is not None]  # active tests whose power is stated
        unknown = [p for p in active if p["power_w"] is None]  # active tests whose power no document states
        # Test Area 1 tests of known power at or above 40 W
        fifty = [p for p in known if p["power_w"] >= fifty_w_min and p["test_area"] == 1]
        # every other known power above 1 W: 1-40 W at any area, or 40 W or more away from Test Area 1
        other_hi = [p for p in known if p["power_w"] > high_power_w and p not in fifty]
        # chained conditional expression: the first matching group wins, in this order
        grp = G50 if fifty else (GHI if other_hi else (GUNK if unknown else GBL))
        mx, mp = max(((p["power_w"], p) for p in known), key=lambda z: z[0], default=(None, None))  # strongest known
        ours = next((c for s, e, c in sess if s <= u <= e), "")  # next(): our session containing u, or ""
        recs.append({  # appends the tag row for this time
            # dict entries (one tag row per distinct time): the time itself
            "timestamp_utc": u,
            # dict entry: the exposure group decided above
            "exposure_group": grp,
            "fifty_w_test": ";".join(sorted(p["test_code"] for p in fifty)),  # the 50 W test codes, joined
            "active_tests": ";".join(sorted(p["test_code"] for p in active)),  # all active codes, joined
            "unknown_power_tests": ";".join(sorted(p["test_code"] for p in unknown)),  # those of unstated power
            # dict entry: how many tests were on air at this time
            "n_active_tests": len(active),
            "max_concurrent_power_w": mx,  # the strongest STATED power on air, or None
            "max_power_test_code": mp["test_code"] if mp else "",  # conditional expressions: blank when unknown
            # dict entry: the test area of the strongest stated power, or None when no power is stated
            "max_power_test_area": mp["test_area"] if mp else None,
            # dict entry: where that power came from (plan, catalog, ...), or '' when none
            "max_power_source": mp["power_source"] if mp else "",
            "no_scheduled_test_anywhere": len(active) == 0,  # boolean: nothing of any power scheduled anywhere
            # dict entry: which of our sessions contains this time, or ''
            "inside_our_capture": ours,
        })
    return pd.DataFrame(recs)  # the tag table, to be merged onto the epochs by time


# ---- host clock: the capture laptop's own clock, the independent time base when receiver time is attacked ----
# timestamps_system_capture.csv pairs the laptop's monotonic clock with its UTC (unix) clock once a second; each
# host_timing/<file>.hosttime.csv sidecar gives, for every serial read, the monotonic time it ended and the file
# offset reached. A message's host time is the end of the read that delivered its LAST byte, so it is always a little
# after the receiver produced it (the latency, measured in the receiver-status script).
def host_clock(session_dir: Path):  # function: returns a converter from monotonic nanoseconds to UTC timestamps
    cap = pd.read_csv(session_dir / "timestamps_system_capture.csv")  # the 1 Hz pairs of the two laptop clocks
    m = cap.monotonic_ns.to_numpy(np.int64)  # monotonic clock, ns, as 64-bit integers (exact)
    u = cap.unix_time_ns.to_numpy(np.int64)  # UTC clock (ns since 1970), as 64-bit integers
    base = int(u[0] - m[0])  # the offset unix - monotonic at the first pair: a large integer kept exact
    rel = (u - m - base).astype(float)  # how that offset drifts from the first pair, ns: small, so floats are exact
    # nested function (a closure: it remembers m, rel and base): monotonic ns -> UTC timestamps
    def to_utc(mono) -> pd.DatetimeIndex:
        mono = np.asarray(mono, np.int64)  # the monotonic times as 64-bit integers
        # np.interp: the drifting offset at each time; beyond the table's ends it holds the end value (slope 1)
        drift = np.round(np.interp(mono.astype(float), m.astype(float), rel)).astype(np.int64)
        return pd.to_datetime(mono + base + drift, unit="ns", utc=True)  # integer ns since 1970 -> UTC timestamps
    return to_utc  # the converter


def host_times(sidecar: Path, ends, to_utc) -> pd.DatetimeIndex:  # function: host UTC time of each message end
    sc = pd.read_csv(sidecar)  # the sidecar: one row per serial read
    off = sc.raw_offset_end.to_numpy(np.int64)  # file offset reached after each read, increasing
    ends = np.asarray(ends, np.int64)  # each message's end offset (one past its last byte)
    k = np.searchsorted(off, ends, side="left")  # binary search: the first read whose end offset reaches the message end
    if len(ends) and k.max() >= len(off):  # if: a message ends beyond the last read in the sidecar
        die(f"{sidecar.name}: a message ends past the last host read; the sidecar does not cover the file")  # stop
    return to_utc(sc.read_end_monotonic_ns.to_numpy(np.int64)[k])  # the end time of that read, converted to UTC


# function: an SBF receiver time stamp -> UTC, or None for the Do-Not-Use values. TOW is milliseconds into the
# GPS week and WNc the continuous week number (SEPT_DOC 4.1.3); GPS time runs LEAP_SECONDS ahead of UTC
def gps_to_utc(tow_ms: int, wnc: int):
    if tow_ms == 4294967295 or wnc == 65535:  # if: either field holds its Do-Not-Use value (SEPT_DOC 4.1.3)
        return None  # no valid time
    # epoch + weeks + milliseconds, minus the leap seconds
    return GPS_EPOCH + timedelta(weeks=wnc, milliseconds=tow_ms) - timedelta(seconds=LEAP_SECONDS)


# ---- SBF satellite and signal tables, from the Septentrio guide in docs/ ----------------------------
# function: SBF SVID -> RINEX satellite name such as "G05", or None where the guide gives no RINEX code (NA).
# Septentrio guide 4.1.9 (PDF p.248): every range of the table, including those an earlier copy in 04 lacked
def sbf_svid_to_rinex(svid: int):
    if 1 <= svid <= 37: return f"G{svid:02d}"  # 1-37: GPS PRN = SVID
    if 38 <= svid <= 61: return f"R{svid - 37:02d}"  # 38-61: GLONASS slots R01-R24, offset 37
    if svid == 62: return None  # 62: GLONASS satellite whose slot is not known: no RINEX code (NA)
    if 63 <= svid <= 68: return f"R{svid - 38:02d}"  # 63-68: GLONASS slots R25-R30, offset 38
    if 71 <= svid <= 106: return f"E{svid - 70:02d}"  # 71-106: Galileo, offset 70
    if 107 <= svid <= 119: return None  # 107-119: L-band (MSS) satellites: NA
    if 120 <= svid <= 140: return f"S{svid - 100:02d}"  # 120-140: SBAS S120-S140, RINEX Snn with nn = SVID-100
    if 141 <= svid <= 180: return f"C{svid - 140:02d}"  # 141-180: BeiDou, offset 140
    if 181 <= svid <= 187: return f"J{svid - 180:02d}"  # 181-187: QZSS, offset 180
    if 191 <= svid <= 197: return f"I{svid - 190:02d}"  # 191-197: NavIC I01-I07, offset 190
    if 198 <= svid <= 215: return f"S{svid - 157:02d}"  # 198-215: SBAS S141-S158, RINEX nn = SVID-157
    if 216 <= svid <= 222: return f"I{svid - 208:02d}"  # 216-222: NavIC I08-I14, offset 208
    if 223 <= svid <= 245: return f"C{svid - 182:02d}"  # 223-245: BeiDou C41-C63, offset 182
    return None  # undefined in the guide: to be ignored (4.1.9)


# dict: SBF signal number -> (constellation letter, RINEX obs code, carrier frequency in MHz); GLONASS FDMA bands
# carry None and use sbf_freq_mhz with the frequency number. Septentrio guide 4.1.10 (PDF p.249); signals the
# guide marks reserved, tentative or without a RINEX code (16, 18, 23, 27, 31, 35-38) are left out
SBF_SIGNALS = {0: ("G", "1C", 1575.42), 1: ("G", "1W", 1575.42), 2: ("G", "2W", 1227.60), 3: ("G", "2L", 1227.60),  # GPS L1CA, L1P, L2P, L2C
               4: ("G", "5Q", 1176.45), 5: ("G", "1L", 1575.42), 6: ("J", "1C", 1575.42), 7: ("J", "2L", 1227.60),  # GPS L5, GPS L1C, QZSS L1CA, QZSS L2C
               8: ("R", "1C", None), 9: ("R", "1P", None), 10: ("R", "2P", None), 11: ("R", "2C", None),  # GLONASS L1CA, L1P, L2P, L2CA (FDMA)
               12: ("R", "3Q", 1202.025), 13: ("C", "1P", 1575.42), 14: ("C", "5P", 1176.45),  # GLONASS L3, BeiDou B1C, BeiDou B2a
               15: ("I", "5A", 1176.45), 17: ("E", "1C", 1575.42), 19: ("E", "6C", 1278.75),  # NavIC L5, Galileo E1, Galileo E6
               20: ("E", "5Q", 1176.45), 21: ("E", "7Q", 1207.14), 22: ("E", "8Q", 1191.795),  # Galileo E5a, E5b, E5 AltBOC
               24: ("S", "1C", 1575.42), 25: ("S", "5I", 1176.45), 26: ("J", "5Q", 1176.45),  # SBAS L1, SBAS L5, QZSS L5
               28: ("C", "2I", 1561.098), 29: ("C", "7I", 1207.14), 30: ("C", "6I", 1268.52),  # BeiDou B1I, B2I, B3I
               32: ("J", "1L", 1575.42), 33: ("J", "1Z", 1575.42), 34: ("C", "7D", 1207.14),  # QZSS L1C, QZSS L1S, BeiDou B2b
               39: ("J", "5P", 1176.45)}  # continuation: the last QZSS signal (L5S)


# function: carrier frequency in MHz of an SBF signal; GLONASS L1/L2 FDMA from the frequency number FreqNr (offset
# 8, guide 4.1.9-4.1.10: L1 1602.00 + (FreqNr-8)*9/16, L2 1246.00 + (FreqNr-8)*7/16); None when it cannot be known
def sbf_freq_mhz(sig: int, freqnr):
    if sig in (8, 9):  # if: GLONASS L1CA or L1P
        return 1602.00 + (freqnr - 8) * 9 / 16 if freqnr else None  # FDMA L1; FreqNr 0 is Do-Not-Use
    if sig in (10, 11):  # if: GLONASS L2P or L2CA
        return 1246.00 + (freqnr - 8) * 7 / 16 if freqnr else None  # FDMA L2
    return SBF_SIGNALS[sig][2] if sig in SBF_SIGNALS else None  # every other signal: the fixed frequency


# function: the four segments of 2.6.5 (moving, stationary, moving, stationary) from the motion script's table, as times and lengths
def motion_265(vm: pd.DataFrame) -> dict:
    seg = vm[vm.test_code == "2.6.5"].sort_values("segment").reset_index(drop=True)  # the session's segments in time order
    # if: the pattern is not moving, stationary, moving, stationary (a split from the X5 would add rows): the prose cannot be written
    if list(seg.state) != ["moving", "stationary", "moving", "stationary"]:
        die(f"preprocessing/vehicle_motion.csv: 2.6.5 segments are {list(seg.state)}, not moving/stationary/moving/stationary")  # stop with a message
    sec = lambda t: pd.Timestamp(t).round("s").strftime("%H:%M:%S")  # lambda: a time rounded to the whole second as HH:MM:SS
    return {"m1_end": sec(seg.end_utc[0]), "stop1_start": sec(seg.start_utc[1]), "stop1_end": sec(seg.end_utc[1]),  # first stop
            "m2_start": sec(seg.start_utc[2]), "m2_end": sec(seg.end_utc[2]), "stop2_start": sec(seg.start_utc[3]),  # second drive, last stop
            "m1_s": float(seg.duration_s[0]), "stop1_s": float(seg.duration_s[1]), "m2_s": float(seg.duration_s[2])}  # lengths in seconds


# ---- helpers added for the final labels, placed at the end so every earlier line keeps its number ----

def exit_if_failed(failed: list, script: str) -> None:  # function: ends a script with status 1 when any self-check failed
    if failed:  # if: the list of failed check names is not empty
        print(f"*** {script}: {len(failed)} self-check(s) FAILED: {'; '.join(map(str, failed))}", file=sys.stderr)  # names them on the error stream
        sys.exit(1)  # status 1, so a chained run stops instead of passing the failure on unnoticed


def decision_times(processed: Path) -> dict:  # function: name -> "HH:MM:SS" of every instant in ground_truth/decision_times.csv (written by the decision-time script)
    d = pd.read_csv(processed / "ground_truth" / "decision_times.csv", dtype=str)  # the table, every cell as text
    return {n: v[-8:] for n, v in zip(d["name"], d["value_utc"])}  # dict comprehension: the last 8 characters of "YYYY-MM-DD HH:MM:SS" are the clock time


# function: the one definition of the diagnosed invalid measurements (not attack signatures), for seconds or run starts.
# u-blox 01 in 2.6.2: G30 before its true reacquisition; mosaic-X5 and mosaic-H in 2.6.10: G08 in its 3 s code error. sec are unix seconds.
def known_invalid(sv, sec, test: str, rx: str, dt: dict, date: str = "2026-09-14") -> np.ndarray:
    sv, sec = np.asarray(sv, dtype=object), np.asarray(sec, float)  # tuple assignment: satellite ids and times as plain arrays
    unix = lambda hms: pd.Timestamp(f"{date} {hms}", tz="UTC").timestamp()  # lambda: "HH:MM:SS" on the date -> unix seconds
    bad = np.zeros(len(sv), bool)  # none flagged yet
    if (test, rx) == ("2.6.2", "UBX_01"):  # if: u-blox 01 in 2.6.2
        bad |= (sv == "G30") & (sec < unix(dt["G30_normal"]))  # G30 before the true lock
    if test == "2.6.10" and rx in ("SBF_02", "SBF_03"):  # if: a Septentrio receiver in 2.6.10
        bad |= (sv == "G08") & (sec >= unix(dt["G08_first"])) & (sec <= unix(dt["G08_last"]))  # G08 inside its code-error seconds
    return bad  # one flag per row


# ---- helpers that read a number from the table or script that holds it ----

def script_constant(path: Path, name: str, index=None):  # function: a module-level constant of an analysis script, read without running the script
    import ast  # import inside the function: the syntax-tree module, kept local so no earlier line of this file moves
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))  # ast.parse: the script as a syntax tree; nothing in it is executed

    def lit(node):  # nested function: the value of a literal node; pd.Timestamp("<text>", ...) counts as its text
        if isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", "")) == "Timestamp" and node.args and isinstance(node.args[0], ast.Constant):  # if: a Timestamp of a literal string
            return node.args[0].value  # its text, e.g. "2026-09-14 12:35:30"
        if isinstance(node, (ast.Tuple, ast.List)):  # if: a tuple or list literal
            return tuple(lit(e) for e in node.elts)  # generator inside tuple(): each element's value, so a tuple of Timestamps works
        if isinstance(node, ast.Name):  # if: a name inside a tuple (such as ONSET): not evaluated
            return f"<{node.id}>"  # a placeholder text
        return ast.literal_eval(node)  # ast.literal_eval: numbers, strings, tuples, lists and dicts only, never a call

    for node in tree.body:  # for loop over the top-level statements only (constants inside functions are not module constants)
        targets = node.targets if isinstance(node, ast.Assign) else ([node.target] if isinstance(node, ast.AnnAssign) else [])  # conditional expression: the assigned names
        for t in targets:  # for loop over the targets of one assignment
            if isinstance(t, ast.Name) and t.id == name:  # if: NAME = value
                value = lit(node.value)  # its value
            elif isinstance(t, ast.Tuple) and name in [e.id for e in t.elts if isinstance(e, ast.Name)]:  # elif: A, NAME, C = 1, 2, 3
                value = lit(node.value.elts[[e.id if isinstance(e, ast.Name) else None for e in t.elts].index(name)])  # the element at NAME's position
            else:  # any other target
                continue  # not this one
            for part in ([] if index is None else (index if isinstance(index, (list, tuple)) else [index])):  # for loop over an index path, e.g. ["UBX_01", 2]
                value = value[part]  # one step down: a dict key or a tuple position
            return value  # the constant (or one element of it)
    raise KeyError(f"{Path(path).name}: no module-level constant {name}")  # raise: a missing constant stops the build rather than printing a guess


UNITS_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen",  # list literal continued on the next line: words for 0-12
               "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]  # list: English words for 0-19, the small mapping these helpers share
TENS_WORDS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]  # list: words for the tens 20-90


def num_words(n, capital: bool = False) -> str:  # function: a whole number 0-999,999 in English words ("fifty", "fifty thousand"), no package
    n = int(round(float(n)))  # whole number (a float such as 50.0 from a table is accepted)
    if n < 20:  # if: 0-19
        w = UNITS_WORDS[n]  # one word
    elif n < 100:  # elif: 20-99
        w = TENS_WORDS[n // 10] + ("" if n % 10 == 0 else "-" + UNITS_WORDS[n % 10])  # "thirty" or "thirty-nine"; "//" is integer division, "%" the remainder
    elif n < 1000:  # elif: 100-999
        w = UNITS_WORDS[n // 100] + " hundred" + ("" if n % 100 == 0 else " and " + num_words(n % 100))  # "one hundred and five"
    else:  # else: thousands
        w = num_words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + num_words(n % 1000))  # "fifty thousand"
    return w[0].upper() + w[1:] if capital else w  # conditional expression: the first letter in capitals at the start of a sentence


def hhmmss(t, digits: int = 0) -> str:  # function: a clock time text from a timestamp or "HH:MM:SS(.fff)" text, rounded to whole seconds or to tenths
    s = str(t)  # as text
    ts = pd.Timestamp(("2026-09-14 " + s) if len(s) <= 15 else s)  # pd.Timestamp: a bare clock time is put on the recording date
    if digits == 1:  # if: tenths of a second
        return ts.round("100ms").strftime("%H:%M:%S.%f")[:10]  # "HH:MM:SS.f"
    return ts.round("s").strftime("%H:%M:%S")  # "HH:MM:SS"


# ---- the values a script prints, written by that script as a facts table ----

def write_facts(script, rows: list) -> Path:  # function: writes Data/processed/printed_facts/<script stem>_facts.csv (key, value, unit, definition)
    import csv  # import inside the function: the standard-library CSV writer, kept local so no earlier line of this file moves
    out = Path(__file__).resolve().parents[2] / "Data" / "processed" / "printed_facts"  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
    out.mkdir(parents=True, exist_ok=True)  # the folder, created on first use
    path = out / f"{Path(script).stem}_facts.csv"  # f-string: one table per script, named after it (e.g. 01_convert_rinex_facts.csv)
    with path.open("w", newline="", encoding="utf-8") as fh:  # with-block: the file is closed afterwards; newline="" lets csv write its own line ends
        w = csv.writer(fh, lineterminator="\n")  # csv.writer with Unix line ends, so two runs give the same bytes on every system
        w.writerow(["key", "value", "unit", "definition"])  # the header
        for key, value, unit, definition in rows:  # for loop over the facts, each a 4-tuple unpacked
            # nested conditional expression: text as given; a whole number (Python or numpy integer) as its digits; any other number as repr of a
            # plain float, its shortest exact form (repr of a numpy float would print "np.float64(...)")
            v = value if isinstance(value, str) else (str(int(value)) if isinstance(value, (int, np.integer)) else repr(float(value)))
            w.writerow([key, v, unit, definition])  # one row
    return path  # where it went


def read_facts(script) -> dict:  # function: the facts table of a script as key -> value text (the values as the script printed them)
    path = Path(__file__).resolve().parents[2] / "Data" / "processed" / "printed_facts" / f"{Path(script).stem}_facts.csv"  # the table write_facts writes
    t = pd.read_csv(path, dtype=str, keep_default_na=False)  # every cell as written (dtype=str: no number is reformatted)
    return dict(zip(t.key, t.value))  # dict built from zip of two columns: key -> value text


def in_repository(project: Path) -> bool:  # function: True when the folder sits in a git repository (git installed and a repository found)
    import subprocess  # standard library: one read-only git command
    try:  # try block: git may not be installed at all
        return subprocess.run(["git", "rev-parse", "--git-dir"], cwd=project, capture_output=True).returncode == 0  # status 0 inside a repository
    except FileNotFoundError:  # except: no git program on this computer
        return False  # no repository


def frozen_git(project: Path, args: tuple):  # function: the read-only git commands of the sealed-test guard answered from the frozen-input table (no repository needed)
    import csv  # standard library: the table
    import hashlib  # standard library: SHA-256 of a file now
    import subprocess  # standard library: the result object the guard expects
    rows = list(csv.DictReader(open(project / "Data" / "processed" / "checks" / "frozen_inputs.csv", encoding="utf-8")))  # list: one row per freeze commit (no path) and per frozen file
    head = next(r["commit"] for r in rows if r["subject"] == "the version of this folder")  # generator in next(): the commit this folder was taken from, answered for HEAD
    a = [str(x) for x in args]  # list comprehension: the command's words as text
    def full(c: str):  # nested function: a commit named by its full hash, a prefix of seven or more characters, or HEAD; None when the table has no such commit
        if c == "HEAD":  # if: the folder's own version
            return head  # its commit
        return next((r["commit"] for r in rows if len(c) >= 7 and r["commit"].startswith(c)), None)  # generator in next(): the first commit with this prefix
    def answer(code: int, out: bytes = b""):  # nested function: a result like subprocess.run(..., capture_output=True) gives
        return subprocess.CompletedProcess(["git", *a], code, out, b"")  # status, stdout and an empty stderr
    def digest(rel: str) -> str:  # nested function: the SHA-256 of a file of the folder now, or "" when it does not exist
        p = project / rel  # the file
        return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else ""  # its digest
    def row(c, rel: str):  # nested function: the table's row of a file at a commit, or None
        return next((r for r in rows if r["commit"] == c and r["path"] == rel), None)  # generator in next(): the first match
    if a[:2] == ["rev-parse", "HEAD"]:  # if: the current commit
        return answer(0, (head + "\n").encode())  # the folder's version
    if a[:2] == ["log", "-1"] and len(a) == 4 and a[2] == "--format=%s":  # if: a commit's subject
        r = next((r for r in rows if r["commit"] == full(a[3]) and not r["path"]), None)  # generator in next(): the commit's own row
        return answer(0, (r["subject"] + "\n").encode()) if r else answer(128)  # its subject, or unknown
    if a[:2] == ["merge-base", "--is-ancestor"]:  # if: is one commit in the history of another
        return answer(0 if full(a[2]) and full(a[3]) else 1)  # every freeze commit is in the history of the folder's version
    if a[0] == "show" and ":" in a[1]:  # if: a file's bytes at a commit
        c, _, rel = a[1].partition(":")  # the commit and the path
        r = row(full(c), rel)  # its row
        if r is None:  # if: not a frozen file of that commit
            return answer(128)  # git's status for a missing path
        if r["frozen_copy"]:  # if: the committed version differs from the folder's file and is packaged beside the table
            return answer(0, (project / r["frozen_copy"]).read_bytes())  # those bytes
        if digest(rel) == r["sha256"]:  # if: the folder's file now is the frozen file (same SHA-256)
            return answer(0, (project / rel).read_bytes())  # its bytes are the committed bytes
        return answer(0, b"")  # in the commit, but the file now differs: bytes that never equal it, so every comparison fails
    if a[:2] == ["ls-files", "--error-unmatch"]:  # if: is a file tracked
        return answer(0 if row(head, a[2]) else 1)  # tracked when the table lists it at the folder's version
    if a[:3] == ["diff", "--quiet", "HEAD"] and a[3] == "--":  # if: is a file unchanged since the folder's version
        r = row(head, a[4])  # its row
        return answer(0 if r and digest(a[4]) == r["sha256"] else 1)  # unchanged when its SHA-256 is the table's
    if a[:3] == ["ls-tree", "-r", "--name-only"]:  # if: every file of a folder at a commit
        c, prefix = full(a[3]), a[4].rstrip("/") + "/"  # the commit and the folder
        return answer(0, "".join(r["path"] + "\n" for r in rows if r["commit"] == c and r["path"].startswith(prefix)).encode())  # generator in join: the listed paths
    return answer(128)  # any other command: not answered without a repository


def analysis_code_digest(project: Path, script: Path, own: str, part: str = "") -> str:  # function: the code digest the analysis stored for a script (or one function of it, named by part), when the code is this folder's copy of it (the table of code digests); else the code's own digest
    import csv  # standard library: the table
    rel = Path(script).resolve().relative_to(Path(project).resolve()).as_posix() + ("::" + part if part else "")  # the script's path in the folder (and the function, when one is meant)
    for r in csv.DictReader(open(project / "Data" / "processed" / "checks" / "code_digests.csv", encoding="utf-8")):  # for loop over the table's rows
        if r["path"] == rel and r["folder_digest"] == own:  # if: this code, unchanged since the folder was made
            return r["analysis_digest"]  # the digest of the same code as the analysis ran it (its saved jobs carry this digest)
    return own  # changed or unlisted code: its own digest, so every job that depends on it runs again
