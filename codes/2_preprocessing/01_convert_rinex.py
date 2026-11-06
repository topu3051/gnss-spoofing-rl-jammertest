#!/usr/bin/env python3
# module docstring below: help text; its Outputs list names the main files (the clock, physics, skip and toolchain tables also written)
"""
Conversion to RINEX and verification.

Converts the native receiver logs to standard RINEX observation and navigation
files and verifies that the conversion preserved what the method depends on.

NOTHING IS ACCEPTED BECAUSE A COMMAND EXITED ZERO. The verification is the work
of this section, and its centrepiece is a direct comparison of pseudorange and
carrier-to-noise density, satellite by satellite, between the RINEX output and
the MeasEpoch block decoded independently from the binary. Plausibility checks
catch gross corruption; only that comparison catches a systematic bias from a
block-revision mismatch, which is the failure mode that would poison every later
section while looking normal.

NO FEATURE IS COMPUTED AND NO LABEL IS BUILT HERE.

Scope:
  Septentrio mosaic-X5 (SBF_02)  primary receiver
  Septentrio mosaic-H  (SBF_03)  second Septentrio, for the transfer experiment
  u-blox               (UBX_01, UBX_02)
  Septentrio mosaic-G5 (SBF_01)  NOT converted; excluded as a primary source and
                                 documented, with its gap structure tested against
                                 the other streams rather than assumed.

Outputs (main files; the clock, physics, skip and toolchain tables in Data/processed are also written)
  Data/rinex/<session>/*.crx.gz, *.nav       observation (Hatanaka + gzip) and navigation only; a stream is
                                             converted only when its .crx.gz is absent
  Data/rinex/README_EXCLUSION.txt            the feature/label boundary, at the boundary
  Data/processed/preprocessing/rinex_conversion_log.csv    written only when something is converted on this run
  Data/processed/preprocessing/rinex_inventory.csv
  Data/processed/preprocessing/rinex_verification.csv
  Data/processed/preprocessing/rinex_measepoch_crosscheck.csv
  Data/processed/preprocessing/mosaic_g5_gap_alignment.csv
  Data/processed/preprocessing/mosaic_g5_gap_dependence.csv  gap rate of the mosaic-G5 by number of satellites tracked, per session
  Data/processed/preprocessing/rinex_ubx01_2612_pieces.csv   index of the three-piece conversion of u-blox 01 in 2.6.12
  Data/processed/preprocessing/rinex_residual_minutes.csv  per session, stream and minute, the residual spread after
      the common clock term; verification and labelling evidence only, never a feature source (circularity open)
  Data/processed/checks/fallback_records.csv    receiver records considered for satellites absent from IGS
  Graph/conversion_check.png        the figure, the conversion checked against the binary
  Graph/observables_l1.png          the figure, the feature-band observables per session

Inputs besides the raw data
  Data/external/igs_brdc/BRDC00IGS_R_20262570000_01D_MN.rnx.gz  IGS merged broadcast file, the only ephemeris source
"""

from __future__ import annotations  # future import: type hints kept as text (a convention across the project)

import binascii  # standard library: crc_hqx computes the SBF CRC-16-CCITT
import math  # standard-library maths: square roots and trigonometry for the physics bounds
import hashlib  # standard-library hashing: sha256 of the convbin binary against its build record
import gzip  # standard-library gzip reader: the IGS broadcast file is read compressed, in memory
import os  # standard-library operating-system access: sets MPLCONFIGDIR before matplotlib loads
import random  # standard-library random; seeded below although nothing here draws from it
import shutil  # standard-library file utilities: disk_usage, and rmtree of the fixed temporary folder inside codes/.tmp_rnx
import struct  # struct: unpacks little-endian binary fields from the SBF and UBX bytes
import subprocess  # subprocess: runs the external convbin program and captures its output
from datetime import datetime, timedelta, timezone  # date/time classes: the GPS epoch, time steps and the UTC zone
from collections import Counter, defaultdict  # Counter counts skips by reason; defaultdict groups records
from fractions import Fraction  # exact fractions: prints the C/N0 step as a fraction, computed
from pathlib import Path  # Path objects for building file paths

import numpy as np  # NumPy under the short name np: arrays, medians, checksum sums
import pandas as pd  # pandas under the short name pd: tables and CSV writing

# Path: codes/.mplconfig, matplotlib's configuration and font-cache folder inside the project folder (as in the handset and receiver-status scripts)
MPLCONFIG = Path(__file__).resolve().parent / ".mplconfig"
MPLCONFIG.mkdir(exist_ok=True)  # creates the folder if missing; exist_ok=True means no error when it exists
os.environ["MPLCONFIGDIR"] = str(MPLCONFIG)  # os.environ: tells matplotlib, imported next, to use that folder
import matplotlib  # the plotting library; imported first so its backend can be set before pyplot loads
matplotlib.use("Agg")  # "Agg" backend: draws to files only, no window (a script needs no screen)
import matplotlib.dates as mdates  # date helpers for the clock-time axis of the figure
import matplotlib.pyplot as plt  # pyplot: the plotting interface, under the short name plt
from matplotlib.patches import Patch  # Patch: a coloured rectangle used as a legend entry

# helpers shared with the handset and receiver-status scripts: one copy each, in codes/common/helpers.py next to this script;
# gps_to_utc returns None for the Do-Not-Use time stamps, and the GPS epoch and leap seconds live there too
from helpers import GPS_EPOCH, LEAP_SECONDS, die, gps_to_utc
# the SBF satellite and signal tables of the Septentrio guide (4.1.9, 4.1.10), shared so they exist once
from helpers import SBF_SIGNALS, sbf_freq_mhz, sbf_svid_to_rinex
# the laptop clock and the host-timing sidecars, so u-blox epochs can be placed on host time
from helpers import host_clock, host_times

SEED = 20260914  # integer constant: the project's fixed random seed
random.seed(SEED)  # seeds Python's random module (nothing here draws from it; fixed on principle)
np.random.seed(SEED)  # seeds NumPy's global generator (also unused; the epoch sampling below is deterministic)

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
DATA = PROJECT / "Data"  # the data folder
DAY = DATA / "Day1_14092026"  # the raw recordings; read only, never written
PROCESSED = DATA / "processed"  # where the CSVs are written
CHECK_DIR = PROJECT / "Data" / "processed" / "checks"  # the check folder: writes the fallback-record table there
RINEX = DATA / "rinex"  # where the RINEX files live
GRAPH = PROJECT / "Graph"  # where the figures are written
TOOLS = PROJECT / "tools"  # where convbin and its build record live
# the IGS merged daily broadcast navigation file for 14 September 2026 (downloaded under 
# addition 2; URL, time, size and SHA-256 in checks_log 1), the only ephemeris source of the clock term and of the
# physics bounds, because the receivers' own navigation files contain records broadcast by our spoofers
IGS_BRDC = DATA / "external" / "igs_brdc" / "BRDC00IGS_R_20262570000_01D_MN.rnx.gz"
IGS_SHA256 = "5d459f68b8d637955e9bc544034b734d7b2156829f3ad557371fe8adecf0f3ce"  # checked on every run (1 value)
# spoofing found to last beyond the scheduled window (2.6.10: per-second end, see below; 2.6.5 to the capture
# end); a receiver record broadcast inside a scheduled window or inside these extensions is never trusted
ATTACK_EXTENDED_END = {"2.6.10": "2026-09-14 14:44:04", "2.6.5": "2026-09-14 12:54:24"}  # dict: test -> UTC end; 2.6.10 is the per-second end of spoofed tracking (the spoofed-tracking script asserts it; the minute bound was 14:43:59)
# temporary files go in a folder inside the project folder, made in main() and removed when the run ends
TMP_ROOT = PROJECT / "codes" / ".tmp_rnx"
RUNLOG = PROJECT / "Data" / "processed" / "runlogs"  # run logs, kept apart from the results: 04_conversion_seconds.csv

TZ_UTC = timezone.utc  # the UTC time-zone object

DISK_FLOOR_GB = 5.0  # float constant: stop if less than 5 GB is free on the data disk
DPI = 300  # integer constant: figure resolution, dots per inch (printed from here, not typed)
# the date the home-folder prefix was removed from the conversion log's commands; fixed
# text, so runs again write the same bytes
PATH_NOTE = "home prefix removed on 2026-10-03; the RINEX headers of this conversion keep the absolute path"
EXCLUDED = {"UBX_02": "the decision: formally excluded from training and evaluation"}  # dict: stream -> exclusion
RINEX_VERSION = "3.04"  # text constant: the RINEX version passed to convbin's -v flag

# Which streams are converted, and the role of each.
# list of 4-tuples: (stream suffix, convbin -r format, receiver name, role); the G5 (SBF_01) is not in it
STREAMS = [
    ("SBF_02", "sbf", "Septentrio mosaic-X5", "primary"),  # the mosaic-X5, the primary receiver
    ("SBF_03", "sbf", "Septentrio mosaic-H", "second Septentrio"),  # the mosaic-H, second Septentrio
    ("UBX_01", "ubx", "u-blox F9 (inferred)", "u-blox 1"),  # first u-blox (model inferred in the receiver-status script)
    ("UBX_02", "ubx", "u-blox F9 (inferred)", "u-blox 2"),  # second u-blox
]
# list with one tuple: stream not converted, name, reason; "several hundred interruptions" is typed
NOT_CONVERTED = [("SBF_01", "Septentrio mosaic-G5",
                  "excluded as a primary source: its logging drops epochs (measured in (f) below)")]

# The feature bands (the research plan, Research_Plan_Topu.pdf, section 2.5: L1, G1, B1 and E1 only)
# GPS L1 C/A, GLONASS G1 C/A, Galileo E1 and BeiDou B1I. dict of dicts: receiver family -> constellation
# letter -> (RINEX band code, label); the u-blox writes Galileo E1 as code 1X, the Septentrio as 1C (both headers)
REQUIRED_BANDS = {
    "sbf": {"G": ("1C", "GPS L1 C/A"), "R": ("1C", "GLONASS G1 C/A"),  # Septentrio: GPS and GLONASS on 1C
            "E": ("1C", "Galileo E1"), "C": ("2I", "BeiDou B1I")},  # Galileo E1 on 1C, BeiDou B1I on 2I
    "ubx": {"G": ("1C", "GPS L1 C/A"), "R": ("1C", "GLONASS G1 C/A"),  # u-blox: GPS and GLONASS on 1C
            "E": ("1X", "Galileo E1"), "C": ("2I", "BeiDou B1I")},  # Galileo E1 on 1X, BeiDou B1I on 2I
}
CONSTELLATIONS = "GRECJIS"  # string: every constellation letter that gets its own code column

# Septentrio signal number -> (constellation letter, RINEX obs code, wavelength m)
C_LIGHT = 299792458.0  # float constant: speed of light in m/s (the value SBF ref 4.1.10 uses for wavelengths)

# tuple assignment: colour-blind-safe hex colours for the figures
C_ORANGE, C_SKY, C_GREEN = "#E69F00", "#56B4E9", "#009E73"
C_BLUE, C_VERMILLION, C_PURPLE = "#0072B2", "#D55E00", "#CC79A7"  # three more colours; C_PURPLE is never used
C_GREY, C_BLACK = "#999999", "#000000"  # grey and black

NOTES: list[str] = []  # empty list with a type hint: plain-language notes gathered during the run
ANOMALIES: list[str] = []  # empty list: anomalies gathered during the run, printed at the end
def note(m): NOTES.append(m)  # one-line function: appends a message to NOTES
def flag(m): ANOMALIES.append(m)  # one-line function: appends a message to ANOMALIES (the "flag() messages")


def free_gb(path: Path) -> float:  # function: free space in GB on the disk holding path
    st = shutil.disk_usage(path)  # shutil.disk_usage returns (total, used, free) in bytes
    return st.free / 1e9  # bytes divided by 1e9 = gigabytes (decimal)


# --------------------------------------------------------------------------
# The converter
# --------------------------------------------------------------------------

def locate_convbin() -> dict:  # function: returns the convbin build record as a dict, or stops
    """Find or build convbin, and record exactly what was built."""
    # creates tools/ if missing; parents=True makes parent folders, exist_ok=True means no error if present
    TOOLS.mkdir(parents=True, exist_ok=True)
    binary = TOOLS / "convbin"  # path of the convbin program
    meta = TOOLS / "convbin_build.txt"  # path of its build record
    if binary.exists() and meta.exists():  # if: both files exist
        # dict from a generator: each "key=value" line split at its first "="; lines without "=" are skipped
        info = dict(l.split("=", 1) for l in meta.read_text().splitlines() if "=" in l)
        info["path"] = str(binary)  # adds the program path to the record
        # the binary must be the one the record describes; hashlib.sha256 over its bytes, as hex text
        got = hashlib.sha256(binary.read_bytes()).hexdigest()
        if info.get("sha256") != got:  # if: the record's hash differs (or is missing)
            die(f"tools/convbin sha256 {got} does not match the build record ({info.get('sha256')})")  # stops
        info["sha256_checked"] = got  # kept, so the printout shows the check was made
        return info  # returns it (keys include source, commit, version_string, sha256)
    # otherwise stops: the program is missing
    die("tools/convbin not found. Run the build step documented in "
        "tools/convbin_build.txt (RTKLIB demo5; the commit is printed in the RINEX headers). This script does "
        "not build or install anything on its own.")


# function: runs convbin on one binary log and reports what happened
def convert(src: Path, fmt: str, out_obs: Path, out_nav: Path, conv: dict) -> dict:
    # command as a list of words: convbin, -r input format (sbf or ubx), -v RINEX version 3.04 (flags: convbin --help)
    rel = lambda p: os.path.relpath(p, PROJECT)  # lambda: a path relative to the project folder (no home folder recorded)
    # command as a list of words: convbin (relative path), -r input format, -v RINEX version (flags: convbin --help)
    cmd = [rel(conv["path"]), "-r", fmt, "-v", RINEX_VERSION,
           # -od Doppler, -os SNR, -oi iono and -ot time corr. in nav header, -hc header comment (convbin --help)
           "-od", "-os", "-oi", "-ot", "-hc", f"converted by {conv.get('version_string', 'convbin')}",
           # -o OBS file, -n NAV file, input; the header comment names the recorded version string
           "-o", rel(out_obs), "-n", rel(out_nav), rel(src)]
    t0 = datetime.now()  # wall-clock start, to time the conversion
    # subprocess.run waits for convbin; capture_output keeps stdout/stderr, text=True decodes them
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT)  # run from the project folder, so relative paths work
    dt = (datetime.now() - t0).total_seconds()  # elapsed seconds
    # returns a dict: the command as one string, and returncode from the CompletedProcess (0 = convbin success)
    return {"command": " ".join(cmd), "returncode": r.returncode,
            "seconds": round(dt, 1),  # seconds, rounded to 0.1
            # last line of convbin's stderr (status line), max 200 chars; "" when stderr is empty or blank
            "stderr_tail": (r.stderr or "").strip().splitlines()[-1][:200]
                           if (r.stderr or "").strip() else ""}  # conditional expression: "" for blank stderr


# ---- the decision: the 2.6.12 UBX_01 reconversion in pieces -----------------------------------------------------
PIECEWISE = {("2.6.12", "UBX_01")}  # set: the stream-sessions converted piece by piece (the decision)


# function: the RXM-RAWX messages of a UBX file in file order, as (receiver time, start offset, end offset,
# numMeas), with the same sequential, checksum-verified parse as count_binary_epochs
def rawx_messages(data: bytes) -> list:
    out, i, n = [], 0, len(data)  # results, read position, file size
    while i < n - 8:  # while a header fits
        if not (data[i] == 0xB5 and data[i + 1] == 0x62):  # if: not at the UBX sync
            j = data.find(b"\xb5\x62", i)  # the next sync, or -1
            if j < 0:  # if: none left
                break  # stop
            i = j  # jump to it
        ln = struct.unpack_from("<H", data, i + 4)[0]  # payload length
        if i + 8 + ln > n:  # if: the message runs past the end of the file
            break  # stop (the early stop is reported by count_binary_epochs)
        arr = np.frombuffer(data, np.uint8, 4 + ln, i + 2)  # class, ID, length and payload, without copying
        ok = (int(arr.sum() & 0xFF) == data[i + 6 + ln]  # CK_A matches
              and int((np.arange(len(arr), 0, -1, dtype=np.int64) * arr).sum() & 0xFF) == data[i + 7 + ln])  # CK_B
        if not ok:  # if: the checksum fails
            i += 2; continue  # step past and search again
        if data[i + 2] == 0x02 and data[i + 3] == 0x15 and ln >= 16:  # if: RXM-RAWX with its fixed part
            tow, week = struct.unpack_from("<dH", data, i + 6)  # rcvTow (s) and week
            t = GPS_EPOCH + timedelta(weeks=week, seconds=tow) - timedelta(seconds=LEAP_SECONDS)  # receiver time, UTC
            out.append((t, i, i + 8 + ln, data[i + 6 + 11]))  # time, start, end, numMeas
        i += 8 + ln  # sequential: the next message
    return out  # the RXM-RAWX list


# function: cuts a UBX file at every RXM-RAWX receiver-time reversal; each cut falls right after the last byte of
# the RXM-RAWX before the reversal, so every RXM-RAWX lands in exactly one piece and each piece's epoch times
# increase; returns the pieces' byte ranges and their RXM-RAWX lists
def cut_points(data: bytes):
    rx = rawx_messages(data)  # every RXM-RAWX in file order
    rev = [k for k in range(1, len(rx)) if rx[k][0] <= rx[k - 1][0]]  # indices where receiver time goes back
    cuts = [0] + [rx[k - 1][2] for k in rev] + [len(data)]  # byte offsets: start, each cut, end of file
    pieces = []  # (start, end, its RXM-RAWX)
    for a, b in zip(cuts[:-1], cuts[1:]):  # consecutive offsets
        pieces.append((a, b, [m for m in rx if a <= m[1] and m[2] <= b]))  # the messages inside the range
    return pieces, rx  # the pieces and all messages


# function: the decision for one stream: cuts the raw file at its time reversals into codes/.tmp_rnx, checks the
# cut reproduces (a second, independent cut must give the same sha256 per piece), converts each piece with the
# same convbin when its RINEX is absent, verifies each piece (strictly increasing RINEX epochs, one-to-one equal to
# the piece's measurement-bearing RXM-RAWX), renames the old merged files '_superseded', and returns the index rows
def reconvert_pieces(test_code: str, src: Path, outdir: Path, conv: dict, folder: Path, tmpdir: str) -> tuple:
    import hatanaka  # the compression package (imported where it is needed; its version is in preprocessing/rinex_toolchain.csv)
    data = src.read_bytes()  # the raw file, read once
    pieces, rx = cut_points(data)  # first cut
    pieces2, _ = cut_points(src.read_bytes())  # second, independent cut from a fresh read
    sha = [hashlib.sha256(data[a:b]).hexdigest() for a, b, _ in pieces]  # sha256 of each piece
    sha2 = [hashlib.sha256(data[a:b]).hexdigest() for a, b, _ in pieces2]  # and of the second cut's pieces
    if sha != sha2:  # if: the two cuts differ
        die(f"{src.name}: the second cut does not reproduce the first ({sha} vs {sha2})")  # stops
    # host UTC of every RXM-RAWX message end (sidecar and laptop clock)
    hts = host_times(folder / "host_timing" / f"{src.name}.hosttime.csv", [m[2] for m in rx], host_clock(folder))
    host_of = {m[1]: h for m, h in zip(rx, hts)}  # start offset -> host time of that RXM-RAWX
    old = [outdir / f"{src.stem}.crx.gz", outdir / f"{src.stem}.nav"]  # the merged conversion's files
    for p_ in old:  # each
        if p_.exists():  # if: still under its old name
            p_.rename(p_.with_name(p_.name.replace(src.stem, f"{src.stem}_superseded")))  # renamed, never deleted
    rows, logs = [], []  # index rows; conversion-log rows for pieces converted on this run
    for k, ((a, b, msgs), h_) in enumerate(zip(pieces, sha), start=1):  # each piece
        stem = f"{src.stem}_part{k:02d}"  # its file stem
        crx, nav = outdir / f"{stem}.crx.gz", outdir / f"{stem}.nav"  # its RINEX files
        piece = Path(tmpdir) / f"{stem}.ubx"  # the temporary piece, inside codes/.tmp_rnx
        piece.write_bytes(data[a:b])  # written
        if not crx.exists():  # if: not converted yet (only on the first run)
            obs = outdir / f"{stem}.obs"  # the uncompressed observation file
            r = convert(piece, "ubx", obs, nav, conv)  # the same convbin, same options, relative paths
            if r["returncode"] != 0 or not obs.exists():  # if: the conversion failed
                die(f"{stem}: convbin failed, rc={r['returncode']}")  # stops
            obs_bytes = obs.stat().st_size  # size before compression
            rnx = obs.with_suffix(".rnx"); obs.rename(rnx)  # the long RINEX suffix hatanaka recognises
            hatanaka.compress_on_disk(rnx, delete=True)  # Hatanaka + gzip, the .rnx removed
            # appends the piece's conversion-log row (same columns as every other row)
            logs.append({"test_code": test_code, "stream": f"UBX_01_part{k:02d}", "receiver": "u-blox F9 (inferred)",
                         # dict entry: the byte range and piece number of the source file, and the binary format
                         "source_file": f"{src.name} bytes {a}-{b - 1} (piece {k}, the decision)", "format": "ubx",
                         # dict unpacking: adds the conversion result's own columns, then the observation and navigation file sizes
                         **r, "obs_bytes": obs_bytes, "nav_bytes": nav.stat().st_size if nav.exists() else 0,
                         "convbin_commit": conv.get("commit", ""), "rinex_version": RINEX_VERSION})  # its log row
        # verification of this piece against its own binary epochs
        txt = hatanaka.decompress(crx)  # the piece's RINEX, decompressed in memory
        tmp_obs = Path(tmpdir) / f"{stem}.obs"  # a temporary text copy
        tmp_obs.write_bytes(txt if isinstance(txt, bytes) else txt.encode())  # written
        codes_, tsys_ = rinex_header(tmp_obs)  # its header
        ep = [t_ for t_, _, _ in iter_rinex(tmp_obs, codes_, tsys_, Counter())]  # its epoch times, file order
        tmp_obs.unlink(); piece.unlink()  # temporary files removed
        bin_t = [m[0] for m in msgs if m[3] > 0]  # the piece's measurement-bearing RXM-RAWX times
        key = lambda t_: round(t_.timestamp() * 1000)  # lambda: integer milliseconds, the match key
        # appends the piece's index row: bytes, hashes, counts, times and the checks of the pieces
        rows.append({"test_code": test_code, "piece": k, "rinex_file": crx.name, "byte_start": a, "byte_end": b - 1,
                     "bytes": b - a, "sha256": h_, "sha256_second_cut": sha2[k - 1],  # identity and hashes
                     "rawx_epochs": len(msgs), "rawx_epochs_with_measurements": len(bin_t),  # binary counts
                     "first_receiver_time": msgs[0][0].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] if msgs else "",  # dict entry: receiver time of the piece's first binary epoch, to the millisecond
                     "last_receiver_time": msgs[-1][0].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] if msgs else "",  # dict entry: receiver time of the piece's last binary epoch, to the millisecond
                     "first_host_time": host_of[msgs[0][1]].strftime("%H:%M:%S.%f")[:-3] if msgs else "",  # dict entry: host time of the first epoch, the independent clock for the u-blox
                     "last_host_time": host_of[msgs[-1][1]].strftime("%H:%M:%S.%f")[:-3] if msgs else "",  # dict entry: host time of the last epoch
                     "rinex_epochs": len(ep),  # the RINEX count
                     # the checks of the pieces: strictly increasing epochs, and one-to-one equal to the binary epochs
                     "rinex_strictly_increasing": all(x < y for x, y in zip(ep, ep[1:])),
                     "rinex_equals_binary_one_to_one": [key(t_) for t_ in ep] == [key(t_) for t_ in bin_t]})  # dict entry: whether the RINEX epochs match the binary epochs one for one, in order
    return rows, logs  # the index rows and any new log rows


# --------------------------------------------------------------------------
# Lightweight RINEX 3 observation reader
# --------------------------------------------------------------------------

# function: reads only the header of a RINEX 3 observation file: the code table per constellation and the time
# system of TIME OF FIRST OBS ("GPS" in every file here; epochs are then converted to UTC by the leap seconds)
def rinex_header(path: Path):
    codes: dict[str, list[str]] = {}  # constellation letter -> its observation codes, in column order
    tsys, cur = "GPS", None  # the time system (default GPS) and the constellation of the current types record
    with path.open("r", errors="replace") as fh:  # text file; undecodable bytes replaced
        for line in fh:  # header lines
            lab = line[60:].rstrip()  # the header label, column 61 on
            if lab == "SYS / # / OBS TYPES":  # if: an observation-types record
                if line[0] != " ":  # if: a new constellation starts (blank = continuation line)
                    cur = line[0]; codes[cur] = line[7:60].split()  # its letter and first codes
                else:  # else: continuation of the same constellation
                    codes[cur] += line[7:60].split()  # more codes
            elif lab == "TIME OF FIRST OBS" and line[48:51].strip():  # elif: the time-system field is filled
                tsys = line[48:51].strip()  # kept
            elif lab == "END OF HEADER":  # elif: the header ends
                break  # stop reading
    return codes, tsys  # the code table and the time system


# generator function: every epoch of a RINEX 3 observation file in file order, as (UTC time, satellite count on the
# epoch line, {satellite: {code: value}}); non-numeric fields and non-zero epoch flags are counted in `counts`
def iter_rinex(path: Path, codes: dict, tsys: str, counts: Counter):
    pos = {s: [(c, 3 + i * 16, 3 + i * 16 + 14) for i, c in enumerate(cs)] for s, cs in codes.items()}  # columns
    shift = timedelta(seconds=LEAP_SECONDS) if tsys == "GPS" else timedelta(0)  # GPS time -> UTC
    cur, nsat, sats = None, 0, {}  # the epoch being read, its count, its satellites
    with path.open("r", errors="replace") as fh:  # text file
        for line in fh:  # skips the header
            if line[60:].startswith("END OF HEADER"):  # if: the header has ended
                break  # the data follow
        for line in fh:  # the data records
            if line[0] == ">":  # if: an epoch line
                if cur is not None:  # if: an epoch is complete
                    yield cur, nsat, sats  # hands it out
                try:  # try: fixed columns that may not parse
                    sec = float(line[18:29])  # seconds with decimals
                    # the epoch time from the fixed columns, then GPS time to UTC by the shift. the fraction is added as a
                    # timedelta instead of a microsecond field, which int(round(frac * 1e6)) could push to 1,000,000 (ValueError, epoch dropped)
                    cur = (datetime(int(line[2:6]), int(line[7:9]), int(line[10:12]), int(line[13:15]),
                                    # argument of datetime(): minutes and whole seconds of the epoch line, in UTC; fraction and shift follow
                                    int(line[16:18]), int(sec), tzinfo=TZ_UTC) + timedelta(seconds=sec - int(sec))) - shift
                    counts["flags_nonzero"] += line[31:32] != "0"  # an event record counts
                    nsat, sats = int(line[32:35]), {}  # the epoch's satellite count and an empty table
                except ValueError:  # except: an unparseable epoch line
                    counts["bad_epoch_lines"] += 1; cur = None  # counted; its satellites are skipped
            elif cur is not None:  # elif: a satellite line of the current epoch
                p = pos.get(line[0])  # the column layout of its constellation
                if p is None:  # if: a constellation the header does not declare
                    counts["undeclared_constellation_lines"] += 1; continue  # counted
                vals = {}  # this satellite's values
                for c, a, b in p:  # each code's 14-character field
                    f = line[a:b]  # the field
                    if f.strip():  # if: not blank (blank = not observed)
                        try:  # try: converts it
                            vals[c] = float(f)  # stored
                        except ValueError:  # except: non-numeric
                            counts["bad_fields"] += 1  # counted
                sats[line[0:3]] = vals  # stored under the satellite name, e.g. "G05"
        if cur is not None:  # if: the last epoch is complete
            yield cur, nsat, sats  # hands it out


# function: the header of several RINEX files read as one stream (the decision's pieces): the union of their code
# tables, each constellation's codes in first-seen order, and the time system of the first
def rinex_header_multi(paths: list):
    codes, tsys = {}, None  # the union; the time system
    for p in paths:  # each file
        c_, t_ = rinex_header(p)  # its header
        tsys = tsys or t_  # the first file's time system
        for s_, cs in c_.items():  # each constellation
            codes.setdefault(s_, [])  # created if new
            codes[s_] += [c for c in cs if c not in codes[s_]]  # new codes appended in order
    return codes, tsys  # the union and the time system


# generator function: the epochs of several RINEX files in order, each read with its own header (the pieces of a stream converted in parts)
def iter_rinex_multi(paths: list, counts: Counter):
    for p in paths:  # each file, in piece order
        c_, t_ = rinex_header(p)  # its own header
        yield from iter_rinex(p, c_, t_, counts)  # its epochs


# --------------------------------------------------------------------------
# Independent MeasEpoch decode, for the cross-check
# --------------------------------------------------------------------------

# struct.Struct objects: precompiled layouts of the two MeasEpoch sub-blocks (Septentrio guide p.254-256):
# Type1 RxChannel u1, Type u1, SVID u1, Misc u1, CodeLSB u4, Doppler i4, CarrierLSB u2, CarrierMSB i1, CN0 u1,
# LockTime u2, ObsInfo u1, N2 u1 (20 bytes; CarrierLSB is unsigned, "H", per p.254)
T1 = struct.Struct("<BBBBIiHbBHBB")
# Type2 Type u1, LockTime u1, CN0 u1, OffsetsMSB u1, CarrierMSB i1, ObsInfo u1, CodeOffsetLSB u2, CarrierLSB u2,
# DopplerOffsetLSB u2 (12 bytes, p.255-256)
T2 = struct.Struct("<BBBBbBHHH")


# function: decodes one MeasEpoch body (after TOW and WNc) into {(satellite, RINEX code): (C m, L cycles, D Hz,
# S dB-Hz, master)}; anything that cannot be placed is counted in `skips` by reason instead of being dropped silently
def measepoch_signals(p: bytes, skips: Counter) -> dict:
    N1, SB1, SB2 = p[0], p[1], p[2]  # Type1 count and the two sub-block lengths (p.253)
    o = 6  # the first Type1 starts after N1, SB1Length, SB2Length, CommonFlags, CumClkJumps, Reserved
    out = {}  # the decoded signals of this epoch

    # nested function: converts one signal's raw fields to RINEX quantities and stores it, or counts why not
    def record(sv, sig, freqnr, ant, pr, carmsb, carlsb, cn0, dop, master):
        if ant != 0:  # if: Aux1 or Aux2 antenna (Type bits 5-7, p.254); RINEX here is the main antenna
            skips["antenna not main"] += 1; return  # counted, not stored
        if sv is None:  # if: an SVID the guide gives no RINEX code (62, L-band, undefined) (4.1.9, p.248)
            skips["SVID without RINEX code"] += 1; return  # counted
        if sig not in SBF_SIGNALS:  # if: a reserved or unlisted signal number (4.1.10, p.249)
            skips["signal not in table"] += 1; return  # counted
        con, code, _ = SBF_SIGNALS[sig]  # its constellation and RINEX code
        if con != sv[0]:  # if: the signal does not belong to the satellite's constellation
            skips["signal-constellation mismatch"] += 1; return  # counted
        f_ = sbf_freq_mhz(sig, freqnr)  # carrier frequency in MHz (GLONASS from FreqNr)
        # carrier phase, p.254: L = PR/lambda + (CarrierMSB*65536 + CarrierLSB)*0.001 cycles, lambda = c/f;
        # invalid when CarrierMSB is -128 and CarrierLSB is 0, or when the pseudorange or frequency is unknown
        L_ = (None if pr is None or f_ is None or (carmsb == -128 and carlsb == 0)
              else pr / (C_LIGHT / (f_ * 1e6)) + (carmsb * 65536 + carlsb) * 0.001)  # conditional expression: otherwise the carrier phase in cycles, from the pseudorange and the phase parts
        # C/N0, p.254: CN0*0.25 for signals 1 and 2, otherwise CN0*0.25 + 10; 255 is Do-Not-Use
        S_ = None if cn0 == 255 else (cn0 * 0.25 if sig in (1, 2) else cn0 * 0.25 + 10.0)
        for nm_, v_ in (("pseudorange", pr), ("carrier phase", L_), ("Doppler", dop), ("C/N0", S_)):  # each field
            if v_ is None:  # if: invalid or Do-Not-Use
                skips[f"{nm_} invalid"] += 1  # counted per field
        if (sv, code) in out:  # if: the same satellite and code twice in one epoch
            skips["duplicate signal"] += 1; return  # counted, the first kept
        out[(sv, code)] = (pr, L_, dop, S_, master)  # stored

    for _ in range(N1):  # for loop over the Type1 sub-blocks
        if o + 20 > len(p):  # if: the block ends early
            skips["truncated sub-block"] += 1; break  # counted, stop
        # tuple unpacking: the twelve Type1 fields at offset o (layout T1, p.254)
        rxch, typ, svid, misc, codelsb, dopp, carlsb, carmsb, cn0, lock, obsinfo, N2 = T1.unpack_from(p, o)
        lo = typ & 0x1F  # SigIdxLo, Type bits 0-4 (p.254)
        sig = (obsinfo >> 3) + 32 if lo == 31 else lo  # 31: the signal number is ObsInfo bits 3-7 plus 32 (p.255)
        freqnr = obsinfo >> 3 if lo in (8, 9, 10, 11) else None  # GLONASS FreqNr in ObsInfo bits 3-7 (p.255)
        # pseudorange, p.254: (CodeMSB*4294967296 + CodeLSB)*0.001 m, CodeMSB = Misc bits 0-3; invalid if both are 0
        pr1 = None if (misc & 0x0F) == 0 and codelsb == 0 else ((misc & 0x0F) * 4294967296 + codelsb) * 0.001
        d1 = None if dopp == -2147483648 else dopp * 0.0001  # Doppler Hz, p.254; -2147483648 is Do-Not-Use
        sv = sbf_svid_to_rinex(svid)  # RINEX satellite name (shared, guide 4.1.9)
        record(sv, sig, freqnr, typ >> 5, pr1, carmsb, carlsb, cn0, d1, True)  # the master signal
        o2 = o + SB1  # the first Type2 of this satellite
        for _ in range(N2):  # for loop over its Type2 sub-blocks
            if o2 + 12 > len(p):  # if: the block ends early
                skips["truncated sub-block"] += 1; break  # counted, stop
            # tuple unpacking: the nine Type2 fields at offset o2 (layout T2, p.255-256)
            typ2, lock2, cn02, offmsb, carmsb2, obsinfo2, codeoff, carlsb2, doppoff = T2.unpack_from(p, o2)
            lo2 = typ2 & 0x1F  # SigIdxLo of the Type2
            sig2 = (obsinfo2 >> 3) + 32 if lo2 == 31 else lo2  # as for Type1 (p.255)
            cm = (offmsb & 0x07) - 8 if (offmsb & 0x07) >= 4 else offmsb & 0x07  # CodeOffsetMSB, 3-bit two's complement
            dm = (offmsb >> 3) - 32 if (offmsb >> 3) >= 16 else offmsb >> 3  # DopplerOffsetMSB, 5-bit two's complement
            # pseudorange, p.256: PRtype1 + (CodeOffsetMSB*65536 + CodeOffsetLSB)*0.001; invalid if MSB -4 and LSB 0
            pr2 = None if pr1 is None or (cm == -4 and codeoff == 0) else pr1 + (cm * 65536 + codeoff) * 0.001
            f1, f2 = sbf_freq_mhz(sig, freqnr), sbf_freq_mhz(sig2, freqnr)  # master and this signal's frequency
            # Doppler, p.256: Dtype1*alpha + (DopplerOffsetMSB*65536 + DopplerOffsetLSB)*1e-4, alpha = f2/f1;
            # invalid if MSB -16 and LSB 0, or when a frequency is unknown
            d2 = (None if d1 is None or f1 is None or f2 is None or (dm == -16 and doppoff == 0)
                  else d1 * (f2 / f1) + (dm * 65536 + doppoff) * 0.0001)  # conditional expression: otherwise the second Doppler, scaled from the first by the frequency ratio
            record(sv, sig2, freqnr, typ2 >> 5, pr2, carmsb2, carlsb2, cn02, d2, False)  # stored or counted
            o2 += SB2  # next Type2
        o = o2  # next Type1 starts after this one's Type2 sub-blocks
    return out  # the epoch's signals


# generator function: walks an SBF file block by block and yields (time, signals) for every MeasEpoch, in file
# order; blocks failing the CRC are passed over (the file's integrity is the receiver-status script's report); Do-Not-Use time
# stamps and revisions other than the expected one are counted in `skips`
def iter_measepoch(path: Path, skips: Counter):
    data = path.read_bytes()  # the whole file as bytes (the largest is about 160 MB)
    n, i = len(data), 0  # its size, and the read position
    while i < n - 8:  # while an 8-byte block header still fits
        if not (data[i] == 0x24 and data[i + 1] == 0x40):  # if: not at the "$@" sync
            j = data.find(b"\x24\x40", i)  # the next sync, or -1
            if j < 0:  # if: none left
                break  # stop
            i = j  # jump to it
        crc, ident, length = struct.unpack_from("<HHH", data, i + 2)  # CRC, ID, Length after the sync
        if length < 8 or length % 4 or i + length > n:  # if: an implausible length (a false sync)
            i += 2; continue  # step past and search again
        if binascii.crc_hqx(data[i + 4:i + length], 0) != crc:  # if: the CRC fails
            i += 2; continue  # step past and search again
        if (ident & 0x1FFF) == 4027:  # if: MeasEpoch
            skips[f"info: MeasEpoch blocks of revision {ident >> 13}"] += 1  # its revision, counted (not a skip)
            tow, wnc = struct.unpack_from("<IH", data, i + 8)  # TOW (ms) and WNc
            t = gps_to_utc(tow, wnc)  # UTC, or None for a Do-Not-Use time stamp
            if t is None:  # if: no valid time
                skips["Do-Not-Use time stamp"] += 1  # counted
            else:  # else: decoded
                yield t, measepoch_signals(data[i + 14:i + length], skips)  # the epoch and its signals
        i += length  # sequential: the next block starts right after this one


# class: the MeasEpoch cross-check over every epoch of one Septentrio stream: for every signal decoded from
# the binary (Type1 and Type2), RINEX minus binary for pseudorange (C), carrier phase (L), Doppler (D) and C/N0 (S),
# per constellation and code; anything that cannot be compared is counted by reason, never skipped silently
class XAcc:
    def __init__(self):  # constructor
        self.s = {}  # (constellation, code) -> statistics
        self.skip = Counter()  # reason -> count
        self.pr_hist = Counter()  # pseudorange difference in 0.1 mm bins -> count (the figure)
        self.cn_steps = Counter()  # C/N0 difference in 1/32 dB-Hz steps -> count (the figure)
        self.epochs = 0  # epochs compared

    def add(self, msig: dict, rsats: dict):  # method: compares one epoch
        self.epochs += 1  # counted
        for (sv, code), vals in msig.items():  # each binary signal
            rv = rsats.get(sv)  # the satellite in the RINEX epoch
            if rv is None:  # if: absent there
                # counted under the satellite's name, so a converter that drops a satellite shows which
                self.skip[f"binary signal: satellite absent from the RINEX epoch ({sv})"] += 1; continue
            a = self.s.setdefault((sv[0], code), Counter())  # its statistics (a Counter starts at zero)
            a["n_signals"] += 1  # one more signal
            for kind, b in zip("CLDS", vals[:4]):  # the four observables: binary value b
                r = rv.get(kind + code)  # the RINEX value, or None
                if b is None or r is None:  # if: either side has no value
                    # counts the missing side: binary only, RINEX only, or neither (chained conditional expression)
                    a[f"n_{kind}_binary_only" if r is None and b is not None else f"n_{kind}_rinex_only"
                      if b is None and r is not None else f"n_{kind}_neither"] += 1  # counted by which side
                    continue  # next observable
                d = r - b  # RINEX minus binary
                a[f"n_{kind}"] += 1  # compared
                a[f"max_{kind}"] = max(a[f"max_{kind}"], abs(d))  # largest absolute difference
                a[f"sum_{kind}"] += d  # for the mean
                if kind == "C":  # if: pseudorange
                    self.pr_hist[round(d * 1e4)] += 1  # 0.1 mm bin
                    a["n_C_over_1mm"] += abs(d) > 1e-3  # beyond the format's last decimal
                elif kind == "L":  # elif: carrier phase - half-cycle corrections are counted apart
                    a["n_L_half_cycle"] += abs(abs(d) - 0.5) < 0.002  # a difference of half a cycle
                    a["n_L_over_0.002"] += abs(d) > 0.002 and abs(abs(d) - 0.5) >= 0.002  # anything else
                elif kind == "D":  # elif: Doppler
                    a["n_D_over_0.002"] += abs(d) > 0.002  # beyond rounding of the 3-decimal field
                else:  # else: C/N0, expected in steps of 1/32 dB-Hz (MeasExtra CN0HighRes)
                    k_ = round(d * 32)  # the step count
                    self.cn_steps[k_] += 1  # counted
                    a["n_S_off_32nd_grid"] += abs(d * 32 - k_) > 0.02  # off the 1/32 grid
        for sv, rv in rsats.items():  # RINEX signals without a binary counterpart
            for c_ in rv:  # each RINEX code
                if c_[0] == "C" and (sv, c_[1:]) not in msig:  # if: a pseudorange with no decoded signal
                    self.skip[f"RINEX signal without binary counterpart ({sv[0]})"] += 1  # counted

    def rows(self, test_code: str, stream: str) -> list:  # method: one CSV row per constellation and code
        out = []  # rows
        for (con, code), a in sorted(self.s.items()):  # sorted for a stable file
            # dict: the row's identity and signal count
            row = {"test_code": test_code, "stream": stream, "constellation": con, "obs_code": code,
                   "n_signals": a["n_signals"]}  # identity and count
            for kind, unit in (("C", "m"), ("L", "cyc"), ("D", "hz"), ("S", "dbhz")):  # each observable
                n_ = a[f"n_{kind}"]  # compared values
                row[f"n_{kind}"] = n_  # count
                row[f"max_abs_d{kind}_{unit}"] = round(a[f"max_{kind}"], 6) if n_ else None  # largest |diff|
                row[f"mean_d{kind}_{unit}"] = round(a[f"sum_{kind}"] / n_, 6) if n_ else None  # mean diff
                row[f"n_{kind}_binary_only"] = a[f"n_{kind}_binary_only"]  # binary value, RINEX blank
                row[f"n_{kind}_rinex_only"] = a[f"n_{kind}_rinex_only"]  # RINEX value, binary invalid
            # for loop over the tolerance counts
            for k_ in ("n_C_over_1mm", "n_L_half_cycle", "n_L_over_0.002", "n_D_over_0.002", "n_S_off_32nd_grid"):
                row[k_] = a[k_]  # the tolerance counts
            out.append(row)  # kept
        return out  # the rows


# --------------------------------------------------------------------------
# Verification helpers
# --------------------------------------------------------------------------

# function: counts observation epochs in a native file; fmt is "sbf" or "ubx"
def count_binary_epochs(path: Path, fmt: str):
    """
    Observation epochs in the native stream: MeasEpoch for SBF, RXM-RAWX for UBX.

    Returns the total count, the count that actually CARRIES measurements, and the
    time span. The distinction matters. A Septentrio MeasEpoch block is emitted
    every epoch whether or not the receiver is tracking anything, and its N1 field
    gives the number of satellites in it. When a spoofing attack denies the
    receiver service, N1 falls to zero and the block is an empty shell. RINEX has
    no way to represent an epoch with no observations, so a converter omits it.
    Comparing RINEX epochs against the raw block count would report that omission
    as conversion loss, which it is not: it is the attack.
    """
    data = path.read_bytes()  # reads the whole file into memory as bytes
    n = len(data)  # number of bytes
    i = 0  # current position
    times: list[datetime] = []  # list with a type hint: every epoch's time
    with_meas: list[datetime] = []  # times of the epochs that carry at least one measurement
    flags: list[bool] = []  # per epoch in file order: carries a measurement (for the longest empty run)
    dnu = 0  # MeasEpoch blocks with a Do-Not-Use time stamp, counted and skipped
    revs: set[int] = set()  # MeasEpoch revisions seen (one is expected)
    early_stop = None  # 42h: byte offset at which a UBX message ran past the end of the file, if it happens
    ubx_ep = []  # u-blox: (receiver time, end offset, carries a measurement) per RXM-RAWX, in file order
    sbf_ep = []  # Septentrio: (receiver time, N1 satellites, block length in bytes) per MeasEpoch
    if fmt == "sbf":  # if: a Septentrio file
        while i < n - 8:  # while loop: while an 8-byte header fits
            if not (data[i] == 0x24 and data[i + 1] == 0x40):  # if: not at the "$@" sync
                j = data.find(b"\x24\x40", i)  # next sync, or -1
                if j < 0:  # if: none left
                    break  # break: stop
                i = j  # jumps to it
            crc, ident, length = struct.unpack_from("<HHH", data, i + 2)  # "<HHH": CRC, ID, Length after the sync
            if length < 8 or length % 4 or i + length > n:  # if: implausible length (as in iter_measepoch)
                i += 2  # steps 2 bytes
                continue  # continue: next pass of the while loop
            if binascii.crc_hqx(data[i + 4:i + length], 0) != crc:  # CRC-16-CCITT check, as above
                i += 2  # steps 2 bytes
                continue  # continue: next pass of the while loop
            if (ident & 0x1FFF) == 4027:  # if: MeasEpoch (block number 4027)
                revs.add(ident >> 13)  # its revision, bits 13-15
                tow, wnc = struct.unpack_from("<IH", data, i + 8)  # "<IH" at byte 8: TOW (ms) and WNc (weeks)
                t = gps_to_utc(tow, wnc)  # to UTC; None for a Do-Not-Use time stamp
                if t is None:  # if: no valid time
                    dnu += 1  # counted
                    i += length  # next block
                    continue  # next pass
                times.append(t)  # records the epoch
                flags.append(data[i + 14] > 0)  # whether it carries a measurement, in file order
                sbf_ep.append((t, data[i + 14], length))  # time, satellite count and block size
                # byte 14 = body byte 6 = N1, the Type1 count (old comment right; SBF ref p.253)
                if data[i + 14] > 0:          # N1, satellites in this epoch
                    with_meas.append(t)  # records it as measurement-bearing
            i += length  # jumps to the next block
    else:  # else: a u-blox file
        while i < n - 8:  # while loop: while a header fits
            if not (data[i] == 0xB5 and data[i + 1] == 0x62):  # if: not at the UBX sync 0xB5 0x62
                j = data.find(b"\xb5\x62", i)  # next sync, or -1
                if j < 0:  # if: none left
                    break  # break: stop
                i = j  # jumps to it
            # "<H" at byte 4: payload length (U2); [0] takes the single value out of the tuple
            ln = struct.unpack_from("<H", data, i + 4)[0]
            if i + 6 + ln + 2 > n:  # if: the message would run past the end of the file
                early_stop = i  # 42h: recorded with its byte offset and reported, not passed over silently
                break  # break: the rest of the file cannot hold a complete message
            # np.frombuffer: a uint8 view of class, ID, length and payload (4 + ln bytes from i+2), without copying
            arr = np.frombuffer(data, np.uint8, 4 + ln, i + 2)
            # UBX 8-bit Fletcher checksum: CK_A = sum of those bytes & 0xFF (keeps the low 8 bits)...
            if (int(arr.sum() & 0xFF) == data[i + 6 + ln]
                    # ...CK_B = sum of (N, N-1, ..., 1) x byte & 0xFF, equal to the running sum of CK_A (as in the receiver-status script)
                    and int((np.arange(len(arr), 0, -1, dtype=np.int64) * arr).sum() & 0xFF)
                    == data[i + 7 + ln]):  # compared with the second checksum byte
                # if: class 0x02, ID 0x15 = UBX-RXM-RAWX, with at least its 16-byte fixed part
                if data[i + 2] == 0x02 and data[i + 3] == 0x15 and ln >= 16:
                    # "<dH" at the payload start: rcvTow (d = R8, s) and week (U2) (UBX doc p.198)
                    rcvtow, week = struct.unpack_from("<dH", data, i + 6)
                    # GPS epoch + week + s in receiver local time; the backslash continues the statement
                    t = GPS_EPOCH + timedelta(weeks=week, seconds=rcvtow) \
                        - timedelta(seconds=LEAP_SECONDS)  # minus 18 s; RAWX's leapS (offset 10) unused
                    times.append(t)  # records the epoch
                    flags.append(data[i + 6 + 11] > 0)  # whether it carries a measurement, in file order
                    ubx_ep.append((t, i + 6 + ln + 2, data[i + 6 + 11] > 0))  # time, end offset, measurement
                    # payload byte 11 = numMeas (UBX doc p.198); ln >= 16 above guarantees the byte exists
                    if data[i + 6 + 11] > 0:   # numMeas
                        with_meas.append(t)  # records it as measurement-bearing
                i += 6 + ln + 2  # skips header (6) + payload + checksum (2)
            else:  # else: checksum failed
                i += 2  # steps 2 bytes
    # the longest run of epochs without a measurement, in file order (start and end time, epoch count)
    best, cur = (None, None, 0), None  # best run so far; the start index of the current run
    for k, f_ in enumerate(flags + [True]):  # a sentinel True closes a run that reaches the end
        if not f_ and cur is None:  # if: a run starts
            cur = k  # its first index
        elif f_ and cur is not None:  # elif: a run ends at k - 1
            if k - cur > best[2]:  # if: longer than the best so far
                best = (times[cur], times[k - 1], k - cur)  # kept
            cur = None  # no run open
    first_file, last_file = (times[0] if times else None), (times[-1] if times else None)  # in file order
    times.sort()  # sorts the times (a spoofed u-blox clock writes them out of order)
    with_meas.sort()  # sorts these too
    # dict: totals, first and last time (sorted, as before), and the step-1 diagnostics
    return {"n": len(times), "n_meas": len(with_meas), "first": times[0] if times else None,
            # dict entry: last time stamp, and the first and last time in the order the file stores them
            "last": times[-1] if times else None, "first_file_order": first_file, "last_file_order": last_file,
            # dict entry: the step-one diagnostics: do-not-use times, revisions, early stop, longest empty run
            "dnu_times": dnu, "revisions": sorted(revs), "early_stop": early_stop, "longest_empty": best,
            "ubx_epochs": ubx_ep, "sbf_epochs": sbf_ep}  # the epochs in file order, for host time and gaps


# ---- physics bounds from the broadcast ephemeris ---------------------------------------------------------
MU_KM3_S2 = 398600.4418  # float constant: Earth's gravitational parameter GM in km^3/s^2 (WGS84), for orbital speeds
OMEGA_E = 7.2921151467e-5  # float constant: Earth's rotation rate in rad/s (WGS84), for the receiver's own motion
WGS84_A_KM, WGS84_B_KM = 6378.137, 6356.7523142  # WGS84 semi-major and semi-minor axes, km, for the site's radius
# named allowances, each with its reason:
# the pseudorange carries the receiver clock offset times c; the Septentrio keeps its clock within about a millisecond
# of GNSS time by millisecond jumps (the MeasEpoch CumClkJumps field, Septentrio guide p.253), so 1 ms. 
# correction: the u-blox does NOT - its common clock term reaches several milliseconds (preprocessing/rinex_clock_common_term.csv),
# so this allowance is too small for it; the count after removing a per-epoch clock term (n_outside_after_clock)
# is the meaningful test for both families
CLOCK_ALLOWANCE_S = 1e-3
# troposphere and ionosphere add metres to tens of metres at low elevation; 100 m covers both with room to spare
ATMOSPHERE_ALLOWANCE_M = 100.0
# a receiver can track a satellite slightly below the geometric horizon (refraction, an antenna above the
# surface), so the longest range is taken at -5 degrees rather than 0
ELEVATION_FLOOR_DEG = -5.0
# GLONASS and SBAS give a position vector at each record, not an orbit's a and e; the radius between records of a
# near-circular orbit can differ from the recorded ones, so the radius range is widened by this slack (km)
ORBIT_RADIUS_SLACK_KM = 100.0
# an ephemeris record whose time of clock is more than this many days from the recording day is not used: one
# record in this data is dated 2008 with a GPS-like orbit for a QZSS satellite and would set wrong bounds
EPHEMERIS_MAX_AGE_DAYS = 1.0
# receiver oscillator frequency offset shows up as a common Doppler bias; 1 part per million of the carrier
DRIFT_ALLOWANCE_PPM = 1.0
# C/N0 window kept from the first version of this script; values outside are reported, not judged as faults
SNR_RANGE_DBHZ = (15.0, 65.0)
# highest carrier frequency (MHz) per constellation and RINEX band digit, for the Doppler bound; GLONASS FDMA at
# the largest frequency number the guide allows (FreqNr 21, k = +13; guide 4.1.9-4.1.10), so the bound is generous
BAND_MHZ = {("G", "1"): 1575.42, ("G", "2"): 1227.60, ("G", "5"): 1176.45,  # GPS L1, L2, L5
            ("R", "1"): 1602.0 + 13 * 0.5625, ("R", "2"): 1246.0 + 13 * 0.4375, ("R", "3"): 1202.025,  # GLONASS
            ("E", "1"): 1575.42, ("E", "5"): 1176.45, ("E", "7"): 1207.14, ("E", "8"): 1191.795,  # Galileo
            ("E", "6"): 1278.75, ("C", "1"): 1575.42, ("C", "2"): 1561.098, ("C", "5"): 1176.45,  # Galileo, BeiDou
            ("C", "7"): 1207.14, ("C", "6"): 1268.52, ("J", "1"): 1575.42, ("J", "2"): 1227.60,  # BeiDou, QZSS
            ("J", "5"): 1176.45, ("J", "6"): 1278.75, ("I", "5"): 1176.45, ("I", "9"): 2492.028,  # QZSS, NavIC
            ("S", "1"): 1575.42, ("S", "5"): 1176.45}  # SBAS L1 and L5


# function: the lines of a RINEX navigation file, gzip-compressed (.gz, the IGS file) or plain (the receivers')
def nav_text_lines(path: Path) -> list:
    if path.suffix == ".gz":  # if: gzip-compressed
        # decompressed in memory, decoded as UTF-8 with undecodable bytes replaced, Windows line ends made "\n"
        return gzip.decompress(path.read_bytes()).decode("utf-8", errors="replace").replace("\r\n", "\n").split("\n")
    return path.read_text(errors="replace").split("\n")  # plain text: exactly how 04 read these files before


# function: the Galileo data-source field (RINEX 3.04 broadcast orbit 5, field 2) as text; bit 0 (1) is
# I/NAV E1-B, bit 2 (4) I/NAV E5b-I, bit 1 (2) F/NAV E5a-I; "&" is the bitwise AND of two integers
def galileo_source(ds: int) -> str:
    inav, fnav = bool(ds & 1) or bool(ds & 4), bool(ds & 2)  # tuple assignment: is an I/NAV bit set; is the F/NAV bit
    # conditional expression chain: both kinds, I/NAV, F/NAV, or neither
    return "both" if inav and fnav else ("I/NAV" if inav else ("F/NAV" if fnav else "unknown"))


# function: consistency of the IGS file's Galileo data-source fields; counts per class, records with
# both I/NAV and F/NAV bits, and records whose clock-pair bit disagrees (I/NAV needs bit 9 = 512, clocks E5b/E1;
# F/NAV needs bit 8 = 256, clocks E5a/E1)
def galileo_source_check(path: Path) -> dict:
    lines = nav_text_lines(path)  # the file's lines
    k = next(i for i, l in enumerate(lines) if l[60:].startswith("END OF HEADER")) + 1  # first data line
    n, both, bad = Counter(), [], 0  # counts per class; records with both kinds; clock-bit failures
    for i, l in enumerate(lines[k:], start=k):  # data lines with their index
        if l[:1] == "E":  # if: a Galileo record starts here
            ds = int(float(lines[i + 5][23:42].replace("D", "E")))  # orbit line 5, field 2: the data-source integer
            src = galileo_source(ds)  # its class
            n[src] += 1  # counted
            if src == "both":  # if: I/NAV and F/NAV bits together
                both.append(f"{l[:23]} ds={ds}")  # listed
            bad += (src == "I/NAV" and not ds & 512) or (src == "F/NAV" and not ds & 256)  # clock-bit mismatch
    return {"counts": dict(sorted(n.items())), "both": both, "clock_bit_failures": int(bad)}  # the result


# function: when a receiver's navigation record was broadcast, as UTC, from its transmission-time field
# (orbit line 7, field 1, seconds of the week given in orbit line 5, field 3); None where this is not defined
def broadcast_utc(sv: str, orb: list):
    if sv[0] in "GEJI":  # if: GPS, Galileo, QZSS or NavIC: GPS-aligned week numbers in RINEX 3.04
        gps_s = orb[4][2] * 604800 + orb[6][0]  # GPS seconds since the GPS epoch
    elif sv[0] == "C":  # elif: BeiDou: BDT week (from 2006-01-01, GPS week 1356) and BDT = GPS time - 14 s
        gps_s = (orb[4][2] + 1356) * 604800 + orb[6][0] + 14  # GPS seconds since the GPS epoch
    else:  # else: GLONASS and SBAS carry no week-based transmission time here
        return None  # unknown
    return GPS_EPOCH + timedelta(seconds=gps_s - LEAP_SECONDS)  # UTC


# function: every receiver record of the given satellites with its broadcast time, receivers and
# files; a record is trusted only if it was broadcast outside every attack period AND the same bytes appear in the
# navigation files of at least two receivers; returns the table rows and the set of trusted record texts
def fallback_records(nav_paths, sats: set, attack: list, windows: list, captures: list) -> tuple:
    fl = lambda s: float(s.replace("D", "E")) if s.strip() else 0.0  # lambda: a Fortran D-exponent field as float
    where = defaultdict(list)  # record text -> list of (receiver, file)
    for path in nav_paths:  # each receiver navigation file (superseded files already left out by the caller)
        lines = nav_text_lines(path)  # its lines
        k = next(i for i, l in enumerate(lines) if l[60:].startswith("END OF HEADER")) + 1  # first data line
        recs, cur = [], None  # records as line lists; the record being read (same grouping as the readers)
        for l in lines[k:]:  # data lines
            if l[:1].isalpha():  # if: a record starts
                cur = [l]; recs.append(cur)  # a new record
            elif cur is not None and l.strip():  # elif: a continuation line
                cur.append(l)  # appended
        rcv = path.stem.split("_", 2)[2].split("_part")[0]  # the receiver (SBF_02, SBF_03, UBX_01, UBX_02)
        for r in recs:  # each record
            if r[0][:3] in sats:  # if: a satellite that needs the fallback
                where["\n".join(r)].append((rcv, f"{path.parent.name}/{path.name}"))  # its receiver and file
    inside = lambda t, ivs: [c for c, a, b in ivs if t is not None and a <= t <= b]  # lambda: intervals holding t
    rows, trusted = [], set()  # table rows; trusted record texts
    for txt, w in sorted(where.items()):  # each distinct record text, sorted for a stable table
        r = txt.split("\n"); sv = r[0][:3]  # its lines and satellite
        orb = [[fl(x[4 + 19 * j:23 + 19 * j]) for j in range(4)] for x in r[1:]]  # broadcast orbit lines
        tb = broadcast_utc(sv, orb) if len(orb) >= 7 else None  # broadcast time, UTC
        att, rcvs = inside(tb, attack), sorted({x[0] for x in w})  # attack periods holding it; receivers
        ok = tb is not None and not att and len(rcvs) >= 2  # bool: the fallback rule
        trusted |= {txt} if ok else set()  # set union: kept when trusted
        # one table row: the record's identity, broadcast time, windows, receivers, files and the decision
        rows.append({"sv": sv, "toc": r[0][4:23], "iod_aode": orb[0][0], "toe_s_of_week": orb[2][0],
                     "broadcast_utc": tb.strftime("%Y-%m-%d %H:%M:%S") if tb else "",  # dict entry: broadcast time of the record in UTC, empty when it has none
                     "inside_scheduled_window_of": " ".join(inside(tb, windows)),  # dict entry: the scheduled test windows that contain the broadcast time
                     "inside_capture_of": " ".join(inside(tb, captures)), "inside_attack_period_of": " ".join(att),  # dict entry: the captures and the attack periods that contain the broadcast time
                     # dict entry: receivers and files that carry the record, and how many receivers do
                     "receivers": " ".join(rcvs), "n_receivers": len(rcvs), "files": " ".join(sorted({x[1] for x in w})),
                     "used": ok, "reason": "trusted" if ok else "; ".join(  # dict entry: whether the record is trusted; otherwise the reasons it is not, joined
                         # generator clause: the possible reasons; no broadcast time, or broadcast inside an attack
                         x for x in ["no broadcast time" if tb is None else "", "broadcast inside an attack period"
                                     if att else "", "in fewer than two receivers" if len(rcvs) < 2 else ""] if x)})  # conditional expression: and fewer than two receivers; empty reasons are dropped
    return rows, trusted  # the table and the trusted texts


# function: reads every RINEX 3 navigation file given and returns, per satellite, its orbit-radius range (km) and
# largest satellite clock term (s) over the usable records; Keplerian systems give a and e (r = a(1 -+ e)),
# GLONASS and SBAS a position vector (r = |r|, widened by ORBIT_RADIUS_SLACK_KM); `rejected` counts unusable records
def ephemeris_bounds(nav_paths, day, rejected: Counter, inav_only: bool = False, keep=None,
                     unhealthy: Counter | None = None) -> dict:  # parameter: optional counter of unhealthy records that still enter the bounds
    """Every thesis computation must pass inav_only=True, as for ephemeris_kepler; False keeps the
    pre-behaviour only so that the closed diagnosis stays reproducible. keep: optional function
    (satellite, record text) -> bool deciding whether a record is used (the fallback for satellites absent from the IGS file); unhealthy: optional
    Counter of records with a non-zero health field that enter the bounds, by constellation."""
    fl = lambda s: float(s.replace("D", "E")) if s.strip() else 0.0  # lambda: a Fortran D-exponent field as float
    out = {}  # satellite -> {"r_min", "r_max", "clk", "e", "a"}
    for path in nav_paths:  # each navigation file
        lines = nav_text_lines(path)  # its lines (a .gz file is decompressed first)
        k = next(i for i, l in enumerate(lines) if l[60:].startswith("END OF HEADER")) + 1  # first data line
        recs, cur = [], None  # records as line lists; the record being read
        for l in lines[k:]:  # data lines
            if l[:1].isalpha():  # if: a record starts (satellite name in column 1)
                cur = [l]; recs.append(cur)  # a new record
            elif cur is not None and l.strip():  # elif: a continuation line
                cur.append(l)  # appended
        for r in recs:  # each record
            sv = r[0][:3]  # satellite name
            if keep is not None and not keep(sv, "\n".join(r)):  # if: a record the caller does not trust
                continue  # not used
            try:  # try: a malformed record raises
                toc = datetime(int(r[0][4:8]), int(r[0][9:11]), int(r[0][12:14]), tzinfo=TZ_UTC)  # time of clock (day)
                v0 = [fl(r[0][23 + 19 * j:42 + 19 * j]) for j in range(3)]  # the three values of line 1
                orb = [[fl(x[4 + 19 * j:23 + 19 * j]) for j in range(4)] for x in r[1:]]  # broadcast orbit lines
                if abs((toc.date() - day).days) > EPHEMERIS_MAX_AGE_DAYS:  # if: dated far from the recording day
                    rejected[f"{sv[0]} record dated {toc.date()}"] += 1; continue  # counted, not used
                if inav_only and sv[0] == "E" and galileo_source(int(orb[4][1])) != "I/NAV":  # if: not I/NAV
                    rejected["E non-I/NAV record (I/NAV only)"] += 1; continue  # counted, not used
                if unhealthy is not None and (orb[0][3] if sv[0] in "RS" else orb[5][1]) != 0:  # if: health not 0
                    unhealthy[sv[0]] += 1  # counted; the record still enters the bounds (no health filter)
                if sv[0] in "RS":  # if: GLONASS or SBAS (position vector X, Y, Z in km, orbit lines 1-3, col 1)
                    rr = math.sqrt(orb[0][0] ** 2 + orb[1][0] ** 2 + orb[2][0] ** 2)  # radius, km
                    lo, hi, e_, a_ = rr - ORBIT_RADIUS_SLACK_KM, rr + ORBIT_RADIUS_SLACK_KM, 0.0, rr  # widened
                else:  # else: Keplerian elements: e is orbit line 2 field 2, sqrt(A) field 4 (in m^0.5)
                    e_, a_ = orb[1][1], (orb[1][3] ** 2) / 1000.0  # eccentricity and semi-major axis in km
                    lo, hi = a_ * (1 - e_), a_ * (1 + e_)  # perigee and apogee radius, km
            except (ValueError, IndexError):  # except: a field that does not parse or a short record
                rejected[f"{sv[0]} malformed record"] += 1; continue  # counted
            b = out.setdefault(sv, {"r_min": lo, "r_max": hi, "clk": 0.0, "e": e_, "a": a_})  # first record
            b["r_min"], b["r_max"] = min(b["r_min"], lo), max(b["r_max"], hi)  # widened by every record
            b["e"], b["clk"] = max(b["e"], e_), max(b["clk"], abs(v0[0]))  # largest e and clock term
    return out  # per-satellite bounds


# function: pseudorange (m) and Doppler (Hz, at a given carrier) bounds for one satellite seen from a site at
# geocentric radius R_km and latitude lat_deg; geometric range r - R at zenith, sqrt(r^2 - R^2 cos^2 E) - R sin E at
# the elevation floor; range rate at most v*R/r + e*v + omega*R*cos(lat) (satellite speed v at perigee, vis-viva)
# rx_clock: whether the window includes the receiver-clock allowance (False after a common clock term is removed)
def sat_bounds(b: dict, R_km: float, lat_deg: float, f_mhz: float | None, rx_clock: bool = True):
    E = math.radians(ELEVATION_FLOOR_DEG)  # the elevation floor in radians
    rho_min = b["r_min"] - R_km  # shortest geometric range, km (satellite overhead at its lowest radius)
    rho_max = math.sqrt(b["r_max"] ** 2 - (R_km * math.cos(E)) ** 2) - R_km * math.sin(E)  # longest, km
    allow = CLOCK_ALLOWANCE_S * C_LIGHT * rx_clock + b["clk"] * C_LIGHT + ATMOSPHERE_ALLOWANCE_M  # allowance, m
    pr = ((rho_min * 1000 - allow), (rho_max * 1000 + allow))  # pseudorange window, m
    if f_mhz is None:  # if: the carrier is unknown, no Doppler bound
        return pr, None  # pseudorange only
    v = math.sqrt(MU_KM3_S2 * (2 / b["r_min"] - 1 / b["a"]))  # speed at perigee, km/s (vis-viva)
    rr_max = v * R_km / b["r_min"] + b["e"] * v + OMEGA_E * R_km * math.cos(math.radians(lat_deg))  # km/s
    dop = rr_max * 1000 / (C_LIGHT / (f_mhz * 1e6)) + DRIFT_ALLOWANCE_PPM * 1e-6 * f_mhz * 1e6  # Hz
    return pr, dop  # the two bounds


# ----: a receiver-clock term common to all satellites, per epoch -----------------------------
# Verification only, never a feature. Per epoch, c*dt is the median over GPS L1 C/A and Galileo E1 satellites of
# pseudorange minus geometric range (broadcast ephemeris, IS-GPS-200 orbit; site from the handset script), with the satellite
# clock applied and the troposphere approximated; the ionosphere is not modelled (metres to tens of metres)
CLK_ELEV_MASK_DEG = 10.0  # satellites below 10 degrees are left out: their troposphere and multipath are largest
TROPO_ZENITH_M = 2.4  # zenith troposphere delay, mapped as 2.4 m / sin(elevation): an approximation, stated as such
RX_HEIGHT_M = 0.0  # 0 m is used: no height is known here, while the reference position is about 45 m ellipsoidal; this inflates the
# clean MAD of this table to 8.7-11.4 m. The old claim that such an error acts almost the same on every
# satellite and is absorbed in c*dt holds only in part, the differential part stays in the residuals
EPH_MAX_AGE_S = 4 * 3600.0  # an orbit record is used within 4 h of its reference time (a broadcast fit interval)
CLK_MIN_SATS = 4  # c*dt is estimated only when at least four satellites contribute
CLK_JUMP_M = 1000.0  # a change of c*dt above 1 km between consecutive estimates counts as a clock jump
CLK_OUTLIER_M = 100.0  # a satellite whose |residual - c*dt| exceeds 100 m counts as inconsistent
# criterion for 'common to all satellites' (chosen after the prototype run of the same computation): the
# median per-epoch residual spread after removing c*dt is at most twice the pre-test margin's
CLK_COMMON_RATIO = 2.0
MU_GPS, MU_GAL = 3.986005e14, 3.986004418e14  # gravitational parameters used by GPS and Galileo, m^3/s^2
F_REL = -4.442807633e-10  # relativistic clock constant F, s/m^0.5 (IS-GPS-200)


# function: the full Keplerian broadcast records of GPS and Galileo from every navigation file, healthy ones only,
# dated within EPHEMERIS_MAX_AGE_DAYS of the recording day; per satellite a dict of NumPy arrays, one entry per record
def ephemeris_kepler(nav_paths, day, inav_only: bool = False) -> dict:
    """Every thesis computation must pass inav_only=True: the receivers' Galileo observable is E1,
    whose message is I/NAV, and 04 applies the I/NAV group delay (BGD E5b/E1); False keeps the pre-
    behaviour only so that the closed diagnosis stays reproducible."""
    fl = lambda s: float(s.replace("D", "E")) if s.strip() else 0.0  # lambda: a Fortran D-exponent field as float
    # list: the fields kept per record, in the order of the stored lists
    keys = ["af0", "af1", "af2", "crs", "dn", "m0", "cuc", "e", "cus", "sqa", "toe", "cic", "om0", "cis", "i0", "crc",
            "w", "omd", "idot", "toe_abs", "toc_abs", "tgd"]  # the fields kept per record
    out = defaultdict(list)  # satellite -> list of records
    skipped = defaultdict(int)  # constellation letter -> records that are short or fail to parse (counted, and the run stops if any)
    for path in nav_paths:  # each navigation file
        lines = nav_text_lines(path)  # its lines (a .gz file is decompressed first)
        k = next(i for i, l in enumerate(lines) if l[60:].startswith("END OF HEADER")) + 1  # first data line
        recs, cur = [], None  # records; the record being read
        for l in lines[k:]:  # data lines
            if l[:1].isalpha():  # if: a record starts
                cur = [l]; recs.append(cur)  # a new record
            elif cur is not None and l.strip():  # elif: a continuation line
                cur.append(l)  # appended
        for r in recs:  # each record
            sv = r[0][:3]  # satellite
            if sv[0] not in "GE":  # if: not GPS or Galileo
                continue  # skipped
            if len(r) < 8:  # if: a GPS or Galileo record shorter than its eight lines
                skipped[sv[0]] += 1  # counted instead of skipped silently
                continue  # not usable
            try:  # try: a malformed record raises
                # time of clock from the record line (GPS time)
                toc = datetime(int(r[0][4:8]), int(r[0][9:11]), int(r[0][12:14]), int(r[0][15:17]),
                               int(r[0][18:20]), int(r[0][21:23]), tzinfo=TZ_UTC)  # time of clock (GPS time)
                v0 = [fl(r[0][23 + 19 * j:42 + 19 * j]) for j in range(3)]  # af0, af1, af2
                o = [[fl(x[4 + 19 * j:23 + 19 * j]) for j in range(4)] for x in r[1:8]]  # broadcast orbit lines 1-7
            except (ValueError, IndexError):  # except: a malformed record
                skipped[sv[0]] += 1  # counted instead of skipped silently
                continue  # next record
            if abs((toc.date() - day).days) > EPHEMERIS_MAX_AGE_DAYS or o[5][1] != 0:  # if: misdated or unhealthy
                continue  # not used
            if inav_only and sv[0] == "E" and galileo_source(int(o[4][1])) != "I/NAV":  # if: a non-I/NAV Galileo record
                continue  # not used (removes the F/NAV records and their wrong group delay)
            # appends the record: clock, orbit and reference times (toe and toc as absolute GPS seconds)
            out[sv].append([v0[0], v0[1], v0[2], o[0][1], o[0][2], o[0][3], o[1][0], o[1][1], o[1][2], o[1][3],
                            o[2][0], o[2][1], o[2][2], o[2][3], o[3][0], o[3][1], o[3][2], o[3][3], o[4][0],  # list element: the remaining orbit fields of the navigation record
                            o[4][2] * 604800 + o[2][0], (toc - GPS_EPOCH).total_seconds(),  # list element: time of ephemeris and time of clock as absolute GPS seconds
                            o[5][2] if sv[0] == "G" else o[5][3]])  # GPS TGD; Galileo BGD E5b/E1 (orbit 6, 4th)
    if skipped:  # if: any GPS or Galileo record was short or failed to parse
        die(f"broadcast records short or unparsed, per constellation: {dict(skipped)} in {[str(p) for p in nav_paths]}")  # stops: a dropped record would change which ephemeris a satellite uses
    # dict comprehension: satellite -> {field: array over its records}, records sorted by reference time
    return {sv: {k_: np.array(sorted(v, key=lambda x: x[19]))[:, j] for j, k_ in enumerate(keys)} for sv, v in out.items()}


# function: satellite ECEF positions (m) and clock offsets (s) at GPS times t (array) from per-observation records p
# (dict of arrays) - the IS-GPS-200 algorithm; mu chosen per system
def kepler_xyz(p: dict, t, mu):
    A = p["sqa"] ** 2; tk = t - p["toe_abs"]  # semi-major axis; time from the reference epoch
    M = p["m0"] + (np.sqrt(mu / A ** 3) + p["dn"]) * tk  # mean anomaly
    E = M.copy()  # eccentric anomaly, solved by fixed-point iteration
    for _ in range(10):  # ten iterations reach far below a millimetre for these eccentricities
        E = M + p["e"] * np.sin(E)  # Kepler's equation
    nu = np.arctan2(np.sqrt(1 - p["e"] ** 2) * np.sin(E), np.cos(E) - p["e"])  # true anomaly
    ph = nu + p["w"]  # argument of latitude
    s2, c2 = np.sin(2 * ph), np.cos(2 * ph)  # harmonic arguments
    u = ph + p["cus"] * s2 + p["cuc"] * c2  # corrected argument of latitude
    r = A * (1 - p["e"] * np.cos(E)) + p["crs"] * s2 + p["crc"] * c2  # corrected radius
    i = p["i0"] + p["cis"] * s2 + p["cic"] * c2 + p["idot"] * tk  # corrected inclination
    x, y = r * np.cos(u), r * np.sin(u)  # position in the orbital plane
    Om = p["om0"] + (p["omd"] - OMEGA_E) * tk - OMEGA_E * p["toe"]  # longitude of the ascending node (ECEF)
    X = x * np.cos(Om) - y * np.cos(i) * np.sin(Om)  # ECEF X
    Y = x * np.sin(Om) + y * np.cos(i) * np.cos(Om)  # ECEF Y
    Z = y * np.sin(i)  # ECEF Z
    dt_ = t - p["toc_abs"]  # time from the clock reference
    dts = p["af0"] + p["af1"] * dt_ + p["af2"] * dt_ ** 2 + F_REL * p["e"] * p["sqa"] * np.sin(E) - p["tgd"]  # clock
    return np.stack([X, Y, Z], axis=1), dts  # positions (n x 3) and clock offsets


# function: per epoch, the common clock term c*dt (m), the satellites used, and the spread of the residuals after
# removing it (median absolute residual and largest absolute residual, m); obs = DataFrame (key, t_gps, sv, pr)
def clock_terms(obs: pd.DataFrame, keph: dict, rx, up) -> pd.DataFrame:
    parts = []  # per-satellite results
    for sv, g in obs.groupby("sv"):  # each satellite
        if sv not in keph:  # if: no usable orbit
            continue  # skipped
        p_all = keph[sv]  # its records
        k = np.abs(g.t_gps.to_numpy()[:, None] - p_all["toe_abs"][None, :]).argmin(axis=1)  # nearest record each
        p = {k_: v[k] for k_, v in p_all.items()}  # the chosen record per observation
        ok = np.abs(g.t_gps.to_numpy() - p["toe_abs"]) <= EPH_MAX_AGE_S  # within the fit interval
        mu = MU_GPS if sv[0] == "G" else MU_GAL  # the system's constant
        tt = g.t_gps.to_numpy() - g.pr.to_numpy() / C_LIGHT  # transmission time, first guess
        pos, dts = kepler_xyz(p, tt, mu)  # position and clock then
        pos, dts = kepler_xyz(p, tt - dts, mu)  # corrected by the satellite clock
        th = OMEGA_E * np.linalg.norm(pos - rx, axis=1) / C_LIGHT  # Earth rotation during the travel time
        X = pos[:, 0] * np.cos(th) + pos[:, 1] * np.sin(th)  # rotated X (Sagnac)
        Y = -pos[:, 0] * np.sin(th) + pos[:, 1] * np.cos(th)  # rotated Y
        los = np.stack([X, Y, pos[:, 2]], axis=1) - rx  # line of sight, m
        fin = np.isfinite(los).all(axis=1) & ok  # finite geometry within the fit interval (epochs on a spoofed date
        los = np.where(fin[:, None], los, 1.0)  # give far-off times; their rows are replaced, then dropped by fin)
        rho = np.linalg.norm(los, axis=1)  # geometric range
        el = np.arcsin(np.clip(np.einsum("ij,j->i", los, up) / rho, -1, 1))  # elevation (einsum: row-wise dot)
        res = g.pr.to_numpy() - rho + C_LIGHT * dts - TROPO_ZENITH_M / np.sin(np.maximum(el, 0.05))  # residual
        keep = fin & (el >= math.radians(CLK_ELEV_MASK_DEG))  # usable observations
        parts.append(pd.DataFrame({"key": g.key.to_numpy()[keep], "res": res[keep]}))  # kept
    if not parts:  # if: nothing usable
        return pd.DataFrame(columns=["key", "cdt_m", "n_sats", "res_mad_m", "res_max_m", "n_over_100m"])  # empty
    r = pd.concat(parts)  # all residuals
    med = r.groupby("key").res.transform("median")  # each epoch's median, aligned to the rows
    r["dev"] = (r.res - med).abs()  # absolute residual after removing it
    r["over"] = r.dev > CLK_OUTLIER_M  # bool column: this satellite is more than 100 m from c*dt
    # one row per epoch: c*dt (median), satellites used, median and largest residual after removal
    out = r.groupby("key").agg(cdt_m=("res", "median"), n_sats=("res", "size"), res_mad_m=("dev", "median"),
                               res_max_m=("dev", "max"), n_over_100m=("over", "sum")).reset_index()  # one row per epoch
    return out[out.n_sats >= CLK_MIN_SATS]  # only epochs with enough satellites


# class: accumulates every pseudorange, Doppler and C/N0 value of one stream against the bounds, per constellation,
# code and period; values of satellites without a usable ephemeris are counted as not judged, never as passing
class PhysAcc:
    def __init__(self, eph: dict, R_km: float, lat_deg: float):  # constructor: the bounds' inputs
        self.eph, self.R, self.lat = eph, R_km, lat_deg  # kept on the object
        self.cache = {}  # (satellite, code) -> its bounds, computed once
        self.acc = {}  # (period, constellation, code) -> running statistics

    def add(self, period: str, sats: dict):  # method: feeds one epoch's satellites
        for sv, vals in sats.items():  # each satellite
            for code, v in vals.items():  # each observable
                kind = code[0]  # C pseudorange, L phase, D Doppler, S C/N0
                if kind == "L" or v == 0:  # if: phase (not range-checked) or an empty zero value
                    continue  # skipped
                key = (period, sv[0], code)  # the row this value belongs to
                # the statistics of this row, created empty on first use (setdefault)
                a = self.acc.setdefault(key, {"n": 0, "min": v, "max": v, "out": 0, "nojudge": 0, "lo": None,
                                              "hi": None, "ex": ""})  # its statistics, created on first use
                a["n"] += 1; a["min"] = min(a["min"], v); a["max"] = max(a["max"], v)  # count and range
                if kind == "S":  # if: C/N0 - the generic window
                    lo, hi = SNR_RANGE_DBHZ  # the window
                else:  # else: pseudorange or Doppler - per-satellite physics
                    bk = (sv, code)  # cache key
                    if bk not in self.cache:  # if: not computed yet
                        b = self.eph.get(sv)  # the satellite's ephemeris bounds, if any
                        # conditional expression: no bounds without an ephemeris, else the satellite's pseudorange and Doppler bounds
                        self.cache[bk] = (None if b is None else
                                          # conditional expression: otherwise the satellite's pseudorange and Doppler bounds on this band
                                          sat_bounds(b, self.R, self.lat, BAND_MHZ.get((sv[0], code[1]))))
                    sb = self.cache[bk]  # the bounds, or None
                    if sb is None or (kind == "D" and sb[1] is None):  # if: no ephemeris (or no carrier)
                        a["nojudge"] += 1; continue  # counted as not judged
                    lo, hi = sb[0] if kind == "C" else (-sb[1], sb[1])  # the window for this value
                a["lo"] = lo if a["lo"] is None else min(a["lo"], lo)  # widest window seen, for the table
                a["hi"] = hi if a["hi"] is None else max(a["hi"], hi)  # continuation
                if not lo <= v <= hi:  # if: outside
                    a["out"] += 1  # counted
                    if not a["ex"]:  # if: no example yet
                        a["ex"] = f"{sv} {v:.3f}"  # the first offending satellite and value

    # method: feeds one epoch's pseudoranges again with the epoch's common clock term cdt (m, or None):
    # each value minus cdt is checked against the bounds without the receiver-clock allowance
    def add_clock(self, period: str, sats: dict, cdt):
        for sv, vals in sats.items():  # each satellite
            for code, v in vals.items():  # each observable
                if code[0] != "C" or v == 0 or (period, sv[0], code) not in self.acc:  # if: not a counted pseudorange
                    continue  # skipped
                a = self.acc[(period, sv[0], code)]  # its row's statistics
                if cdt is None:  # if: no clock term for this epoch
                    a["noclk"] = a.get("noclk", 0) + 1; continue  # counted
                bk = (sv, code, "clk")  # cache key for the clock-free bounds
                if bk not in self.cache:  # if: not computed yet
                    b = self.eph.get(sv)  # ephemeris bounds
                    # the clock-free window, or None without an ephemeris (conditional expression)
                    self.cache[bk] = None if b is None else sat_bounds(b, self.R, self.lat, None, rx_clock=False)[0]
                lo_hi = self.cache[bk]  # the window, or None
                if lo_hi is None:  # if: no ephemeris
                    continue  # already counted as not judged
                a["n_clk"] = a.get("n_clk", 0) + 1  # judged after clock removal
                a["out_clk"] = a.get("out_clk", 0) + (not lo_hi[0] <= v - cdt <= lo_hi[1])  # outside after removal

    def rows(self, test_code: str, stream: str) -> list:  # method: the accumulated rows for the CSV
        out = []  # rows
        for (period, con, code), a in sorted(self.acc.items()):  # sorted for a stable file
            judged = a["n"] - a["nojudge"]  # values with a bound
            # appends one CSV row: identity, counts, range, bounds and the first offending value
            out.append({"test_code": test_code, "stream": stream, "period": period, "constellation": con,
                        "obs_code": code, "n_values": a["n"], "n_not_judged_no_ephemeris": a["nojudge"],  # dict entry: observation code, values seen, and values not judged for lack of an ephemeris
                        "min": round(a["min"], 3), "max": round(a["max"], 3),  # the value range
                        "bound_lo": round(a["lo"], 3) if a["lo"] is not None else None,  # widest bounds used
                        "bound_hi": round(a["hi"], 3) if a["hi"] is not None else None,  # dict entry: the highest bound used, rounded to the millimetre, or empty when none applied
                        # dict entry: values outside the bounds, as a count and as a percentage of those judged
                        "n_outside": a["out"], "pct_outside": round(100 * a["out"] / judged, 4) if judged else None,
                        "first_outside": a["ex"],  # the first offending satellite and value
                        # pseudoranges only - judged and outside after removing the epoch's common clock
                        # term, and values whose epoch had no clock term (blank for Doppler and C/N0)
                        "n_judged_after_clock": a.get("n_clk", 0) if code[0] == "C" else None,
                        "n_outside_after_clock": a.get("out_clk", 0) if code[0] == "C" else None,  # dict entry: pseudoranges still outside once the common clock term is removed
                        "n_no_clock_estimate": a.get("noclk", 0) if code[0] == "C" else None})  # dict entry: pseudoranges whose epoch had no clock estimate
        return out  # the rows


# function: epoch spacing statistics from (time, n_sat) pairs
# nominal_s: the true epoch spacing when the times given are host times (step 3): host arrival jitter would
# otherwise inflate the median spacing and turn every 1 Hz epoch arriving 10 ms late into a 'gap'
# span_s: the span the epochs should cover (step 3: for the u-blox, the binary's measurement-bearing span on
# host time, so epochs the converter dropped at the end of a capture count as missing)
def epoch_gaps(epochs: list[tuple[datetime, int]], nominal_s: float | None = None,
               span_s: float | None = None) -> dict:  # parameter: optional span in seconds, so gaps at the capture's end count as missing
    if len(epochs) < 3:  # if: fewer than 3 epochs
        # returns a placeholder dict of zeros and None
        return {"n_epochs": len(epochs), "nominal_hz": None, "gaps_gt_1s": 0, "gap_threshold_s": None,
                # rest of the placeholder (completeness None)
                "largest_gap_s": 0.0, "lost_s": 0.0, "completeness_pct": None}
    t = np.array([e[0].timestamp() for e in epochs])  # epoch times as POSIX seconds in a NumPy array
    d = np.diff(t)  # differences between consecutive epochs
    # typical spacing: the given true spacing, else the median difference (0.1 s at 10 Hz, 0.05 s at 20 Hz)
    nominal = nominal_s if nominal_s else float(np.median(d))
    thr = max(1.0, 1.5 * nominal)  # gap threshold: 1 s as before, 1.5 spacings for a 1 Hz stream on host time
    big = d[d > thr]  # gaps: differences over the threshold
    span = span_s if span_s else t[-1] - t[0]  # the given span, else first-to-last on the times given
    # epochs expected at the median spacing over that span
    expected = int(span / nominal) + 1 if nominal > 0 else len(t)
    return {"n_epochs": len(epochs),  # returns a dict:
            "nominal_hz": round(1 / nominal, 2) if nominal > 0 else None,  # nominal rate in Hz
            "gaps_gt_1s": int(big.size),  # number of gaps (over gap_threshold_s; the name is kept for continuity)
            "gap_threshold_s": round(thr, 2),  # the threshold used
            "largest_gap_s": round(float(d.max()), 2),  # largest single difference
            # total time in gaps (whole gap lengths, not only the excess over the spacing)
            "lost_s": round(float(big.sum()), 1),
            # completeness = epochs / expected in %; 0.0 when the span is a year
            "completeness_pct": round(100 * len(t) / expected, 1) if expected else None}


def fig_observables(inv_rows, sessions, path):  # function: the figure, the X5 satellite count and C/N0 per session
    order = list(sessions.test_code)  # session codes in capture order
    # one subplot row per session; sharex=False lets each row keep its own time range
    fig, axes = plt.subplots(len(order), 1, figsize=(13, 2.2 * len(order)), sharex=False)
    for k, code in enumerate(order):  # for loop over sessions with their row number (enumerate)
        ax = axes[k]  # this row's axes
        # next() over a generator: the first inv_rows entry for this session's X5, or None
        rec = next((r for r in inv_rows if r["test_code"] == code
                    and r["stream"] == "SBF_02"), None)  # second condition: the X5 stream
        sr = sessions[sessions.test_code == code].iloc[0]  # the session's ground-truth row
        ss = pd.Timestamp(sr["scheduled_start_utc"], tz="UTC")  # scheduled start as a UTC timestamp
        se = pd.Timestamp(sr["scheduled_end_utc"], tz="UTC")  # scheduled end
        if rec and rec.get("_series") is not None:  # if: an X5 series was stored
            s = rec["_series"]  # the 10-s series
            t0, t1 = s.index.min(), s.index.max()  # first and last times of the data
            # shades the scheduled window, clipped to the data range (date2num converts to plot units)
            ax.axvspan(mdates.date2num(max(ss, t0)), mdates.date2num(min(se, t1)),
                       color=C_GREEN, alpha=0.15, zorder=0, lw=0)  # style of the shading
            # blue line: satellites with a feature-band pseudorange, every epoch; 0 where nothing was tracked
            ax.plot(mdates.date2num(s.index), s["n_sat"], lw=0.8, color=C_BLUE,
                    label="satellites")  # its legend label
            ax.set_ylabel(f"{code}\nsatellites", fontsize=8.5)  # row label with the session code
            # y range from just below 0 (so a count of 0 stays visible above the axis line) to 60 or 10 % more
            ax.set_ylim(-3, max(60, float(s["n_sat"].max()) * 1.1))
            ax2 = ax.twinx()  # twinx: a second y axis on the right sharing the x axis
            # red line: median C/N0 of those feature-band signals, every epoch; absent where none was tracked
            ax2.plot(mdates.date2num(s.index), s["cn0"], lw=0.8, color=C_VERMILLION,
                     label="median C/N0")  # its legend label
            ax2.set_ylabel("C/N0 (dB-Hz)", fontsize=8)  # right-axis label
            ax2.set_ylim(15, 55)  # right-axis range 15-55 dB-Hz
            ax2.tick_params(labelsize=7.5)  # tick font size
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=TZ_UTC))  # clock times in UTC on the x axis
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))  # a tick every 5 minutes
            ax.grid(axis="x", color=C_GREY, alpha=0.25, lw=0.5)  # light vertical grid
        else:  # else: no X5 series for this session
            # centred grey text; transAxes means coordinates are fractions of the axes (0-1)
            ax.text(0.5, 0.5, "no converted data", transform=ax.transAxes,
                    ha="center", fontsize=9, color=C_GREY)  # text alignment, size, colour
        ax.tick_params(labelsize=7.5)  # tick font size
    axes[-1].set_xlabel("UTC on 14 September 2026")  # x label on the bottom row only
    fig.legend(handles=[  # figure-wide legend built from three hand-made handles:
        Patch(facecolor=C_GREEN, alpha=0.3, label="Scheduled test window"),  # the shading
        # the count is restricted to the feature bands, and the legend says so
        plt.Line2D([], [], color=C_BLUE, label="Satellites with a feature-band pseudorange: GPS L1, GLONASS G1, "
                                               "Galileo E1, BeiDou B1I (left axis; 0 when nothing was tracked)"),
        # the C/N0 legend: the same signals, every epoch
        plt.Line2D([], [], color=C_VERMILLION, label="Median C/N0 of those signals (right axis)")],
        # legend under the plots, 3 columns, no frame
        loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=2, frameon=False, fontsize=9)
    # figure title: what is counted, on which receiver, from which epochs
    fig.suptitle("First look at the observables on the feature bands (L1, G1, E1, B1): satellite count and "
                 # string continued: the receiver and the averaging of the title, with its size and height
                 "carrier-to-noise density, Septentrio mosaic-X5, every epoch (10 s medians)", fontsize=12, y=0.998)
    fig.tight_layout(rect=(0, 0.02, 1, 0.985))  # tight layout inside the rectangle left for title and legend
    # writes the PNG at 300 dpi; metadata None removes software and time stamps so runs again are byte-identical
    fig.savefig(path, dpi=DPI, bbox_inches="tight",
                metadata={"Software": None, "Creation Time": None})  # metadata argument
    plt.close(fig)  # frees the figure's memory


# function: the figure, the three-panel conversion check
def fig_check(ver: pd.DataFrame, pr_hist: Counter, cn_steps: Counter, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))  # one row of three subplots, 15 x 4.6 inches
    ax = axes[0]  # left panel
    for st, g in ver.groupby("stream"):  # for loop over streams; groupby yields (name, sub-table)
        # scatter: measurement-bearing binary epochs (x) against RINEX epochs (y)
        ax.scatter(g.binary_epochs_with_measurements, g.rinex_epochs, s=42,
                   label=st, alpha=0.85)  # marker size, legend label, transparency
    # axis limit: 0 to 8 % beyond the larger maximum of the two counts
    lim = [0, max(ver.binary_epochs_with_measurements.max(),
                  ver.rinex_epochs.max()) * 1.08]  # continuation
    # dashed grey 1:1 line, where the two counts would be equal
    ax.plot(lim, lim, color=C_GREY, ls="--", lw=1, zorder=0)
    ax.set_xlabel("measurement-bearing epochs in the binary")  # x label
    ax.set_ylabel("epochs in the RINEX")  # y label
    # points on the line converted every measurement-bearing epoch; a point below it lost epochs, and the
    # only such points are converter drops of time-reversed epochs, each labelled with its stream and status
    ax.set_title("Epoch counts: RINEX against measurement-bearing binary epochs\n(below the line: epochs the "
                 "converter dropped)", fontsize=10.5)  # panel title on two lines
    for r_ in ver[ver.epoch_ratio < 0.999].itertuples():  # each stream below the line
        ax.annotate(f"{r_.test_code} {r_.stream}" + (" (excluded)" if r_.excluded else ""),  # its label
                    # argument of annotate(): the point labelled, binary epochs against RINEX epochs; small text offset
                    (r_.binary_epochs_with_measurements, r_.rinex_epochs), fontsize=7.5, xytext=(6, -10),
                    textcoords="offset points")  # placed just below-right of the point
    ax.legend(fontsize=7.5, frameon=False)  # legend
    ax.grid(color=C_GREY, alpha=0.25, lw=0.5)  # grid

    ax = axes[1]  # middle panel
    if pr_hist:  # if: there are cross-check values (every epoch)
        k_ = sorted(pr_hist)  # the 0.1 mm bins present
        n_ = sum(pr_hist.values())  # all compared pseudoranges
        # bar chart: one bar per 0.1 mm bin with its count
        ax.bar([k / 1e4 for k in k_], [pr_hist[k] for k in k_], width=1e-4 * 0.8, color=C_SKY,
               edgecolor=C_BLACK, lw=0.4)  # one bar per 0.1 mm bin
        ax.set_xlabel("RINEX pseudorange minus binary (m)")  # x label
        ax.set_ylabel("signal-epochs")  # y label
        # title (f-string) with n and the largest |difference|
        ax.set_title(f"Pseudorange agreement, every epoch\nn={n_}, max |diff| "
                     f"{max(abs(k) for k in k_) / 1e4:.4f} m", fontsize=10.5)  # continuation
    ax.grid(color=C_GREY, alpha=0.25, lw=0.5)  # grid

    ax = axes[2]  # right panel
    if cn_steps:  # if: there are C/N0 values
        k_ = sorted(cn_steps)  # the 1/32 dB-Hz steps present
        n_ = sum(cn_steps.values())  # all compared C/N0 values
        # bar chart: one bar per 1/32 dB-Hz step with its count
        ax.bar([k / 32 for k in k_], [cn_steps[k] for k in k_], width=0.8 / 32, color=C_ORANGE,
               edgecolor=C_BLACK, lw=0.4)  # one bar per step
        ax.set_xlabel("RINEX C/N0 minus binary (dB-Hz)")  # x label
        ax.set_ylabel("signal-epochs")  # y label
        # title (f-string) with n and the largest |difference|
        ax.set_title(f"C/N0 agreement, every epoch\nn={n_}, max |diff| "
                     f"{max(abs(k) for k in k_) / 32:.4f} dB-Hz", fontsize=10.5)  # continuation
    ax.grid(color=C_GREY, alpha=0.25, lw=0.5)  # grid
    fig.suptitle("Conversion check: the RINEX output against the binary it came from",  # overall title
                 fontsize=12.5)  # title size
    fig.tight_layout(rect=(0, 0, 1, 0.93))  # tight layout below the title
    # writes the PNG, metadata removed as in the figure
    fig.savefig(path, dpi=DPI, bbox_inches="tight",
                metadata={"Software": None, "Creation Time": None})  # metadata argument
    plt.close(fig)  # frees the figure's memory


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:  # function: the whole section, start to finish
    PROCESSED.mkdir(parents=True, exist_ok=True)  # creates Data/processed if missing (no change when present)
    RINEX.mkdir(parents=True, exist_ok=True)  # same for Data/rinex
    GRAPH.mkdir(parents=True, exist_ok=True)  # same for Graph
    TMP_ROOT.mkdir(parents=True, exist_ok=True)  # the temporary folder's parent, inside the project folder
    # 42j: a temporary folder removed at the end of main. a fixed name instead of tempfile's random one, because the
    # 2.6.12 UBX_01 pieces are converted from this folder and convbin writes the input path into the RINEX header
    tmp_fixed = TMP_ROOT / "rinex_verify"  # the fixed temporary folder, codes/.tmp_rnx/rinex_verify (removed when the run ends)
    if tmp_fixed.exists():  # if: left over from an interrupted run
        shutil.rmtree(tmp_fixed)  # removed first, so the run starts from an empty folder
    tmp_fixed.mkdir()  # created empty
    TMPDIR = str(tmp_fixed)  # its path as text, as tempfile's .name was
    diag = []  # step-1 diagnostics per stream: Do-Not-Use times, revisions, early stop, bad fields, epoch flags
    print("=" * 78)  # banner: "=" repeated 78 times
    print("CONVERSION TO RINEX AND VERIFICATION")  # fixed text
    print("=" * 78)  # banner

    sessions = pd.read_csv(PROCESSED / "ground_truth" / "session_ground_truth.csv")  # the timeline script's session table
    sessions["_o"] = pd.to_datetime(sessions.capture_start_utc)  # new column: capture start as a datetime, for sorting
    # sorts sessions by capture start and renumbers the rows (drop=True discards the old index)
    sessions = sessions.sort_values("_o").reset_index(drop=True)

    conv = locate_convbin()  # the convbin build record (stops if missing)
    print("\n[1] CONVERTER")  # step header
    for k in ("source", "branch", "commit", "version_string", "platform", "sha256", "sha256_checked"):  # record keys
        if k in conv:  # if: the key is in the record
            print(f"      {k:<16} {conv[k]}")  # prints the key padded to 16 characters, then its value
    print("      The binary's sha256 matches the build record.")  # fixed text, reached only after the check
    print("      Full build record, including why one compiler flag differs from the")  # fixed text
    print("      shipped makefile, is in tools/convbin_build.txt.")  # continued

    print("\n[2] DISK")  # step header
    fg = free_gb(DATA)  # free space on the data disk
    # prints it against the floor
    print(f"      {fg:.1f} GB free; the floor is {DISK_FLOOR_GB:.0f} GB. Observation files")
    print("      are Hatanaka-compressed after verification, and the uncompressed file is")  # fixed text
    print("      then removed (the ratios are in the inventory below).")  # continued
    if fg < DISK_FLOOR_GB:  # if: below the floor
        die(f"only {fg:.1f} GB free, below the {DISK_FLOOR_GB:.0f} GB floor")  # die(): stops, exit code 1

    print("\n[3] SCOPE")  # step header
    for st, fmt, name, role in STREAMS:  # for loop over the four converted streams
        print(f"      convert      {st:<8} {name:<24} {role}")  # prints each one, padded into columns
    for st, name, why in NOT_CONVERTED:  # for loop over the excluded stream
        print(f"      NOT convert  {st:<8} {name:<24} {why}")  # prints it; its reason text is typed

    # ---- convert ------------------------------------------------------
    print("\n[4] CONVERTING")  # step header
    # four empty lists: conversion log, figure series, verification rows, cross-check rows
    log_rows, inv_rows, ver_rows, cross_rows = [], [], [], []
    piece_rows = []  # the decision: one index row per piece of the 2.6.12 UBX_01 reconversion
    xsk_rows = []  # per stream, epochs compared and every skip count by reason
    true_times = {}  # (test, stream) -> binary epoch times on a true clock, for the G5 gap comparison
    pr_hist, cn_steps = Counter(), Counter()  # pooled cross-check histograms for the figure
    for _, sr in sessions.iterrows():  # for loop over sessions in capture order
        folder = DAY / sr["session_folder"]  # the raw session folder (read only)
        outdir = RINEX / sr["session_folder"]  # the session's RINEX folder
        outdir.mkdir(parents=True, exist_ok=True)  # creates it if missing (no change when present)
        for st, fmt, name, role in STREAMS:  # for loop over the four streams; role is not used
            # next(iter(glob)): the first raw file named *<stream>.* (e.g. ..._SBF_02.sbf), or None
            src = next(iter(folder.glob(f"*{st}.*")), None)
            if src is None:  # if: none found
                flag(f"{sr['session_folder']}: no {st} stream found")  # records an anomaly
                continue  # next stream
            obs = outdir / f"{src.stem}.obs"  # path for the uncompressed observation file
            nav = outdir / f"{src.stem}.nav"  # path for the navigation file
            if (sr["test_code"], st) in PIECEWISE:  # if: the decision - converted piece by piece
                rows_, logs_ = reconvert_pieces(sr["test_code"], src, outdir, conv, folder, TMPDIR)  # cut, convert
                piece_rows += rows_; log_rows += logs_  # the index rows and any new log rows
                # one line: how many pieces, converted now or already converted
                print(f"      {sr['test_code']:<7} {st:<7} in {len(rows_)} pieces:"
                      f"{'converted now' if logs_ else 'already converted, checked'}")  # what happened
                continue  # the generic path below is not used for this stream
            # if: any <stem>.crx* (here .crx.gz) exists, the stream counts as converted and convbin is not run
            if any(outdir.glob(f"{src.stem}.crx*")):
                # the skip line, expected 24 times on a run again
                print(f"      {sr['test_code']:<7} {st:<7} already converted, skipping")
            else:  # else: convert this stream
                if free_gb(DATA) < DISK_FLOOR_GB:  # if: free space fell below the floor
                    die(f"free space fell below the {DISK_FLOOR_GB:.0f} GB floor during "  # stops before the disk fills
                        "conversion; stopping before the disk fills")
                # runs convbin (writes <stem>.obs and <stem>.nav into Data/rinex/<session>)
                r = convert(src, fmt, obs, nav, conv)
                # prints return code, seconds and OBS size in MB (0 if no file)
                print(f"      {sr['test_code']:<7} {st:<7} rc={r['returncode']} "
                      f"{r['seconds']:>6.1f} s  "  # seconds, 6 wide, 1 decimal
                      f"{obs.stat().st_size/1e6 if obs.exists() else 0:>7.1f} MB")  # OBS size in MB, 0 if no file
                # appends a log row: identity fields, then the convert() dict merged in with **r
                log_rows.append({"test_code": sr["test_code"], "stream": st,
                                 "receiver": name, "source_file": src.name,  # receiver name and raw file name
                                 # **r unpacks command, returncode, seconds and stderr_tail into this dict
                                 "format": fmt, **r,
                                 # OBS size in bytes, read now before compression (0 if missing)
                                 "obs_bytes": obs.stat().st_size if obs.exists() else 0,
                                 "nav_bytes": nav.stat().st_size if nav.exists() else 0,  # NAV size in bytes
                                 "convbin_commit": conv.get("commit", ""),  # convbin commit from the build record
                                 "rinex_version": RINEX_VERSION})  # RINEX version
                if r["returncode"] != 0 or not obs.exists():  # if: a non-zero return code or no OBS file
                    flag(f"{sr['test_code']} {st}: conversion failed, rc={r['returncode']}")  # records the failure
    logp = PROCESSED / "preprocessing" / "rinex_conversion_log.csv"  # path of the conversion log
    rewrite = False  # whether the existing log still holds absolute paths
    if logp.exists() and logp.stat().st_size > 0:  # if: a log exists
        try:  # try: if reading fails, the except branch runs
            lg0 = pd.read_csv(logp)  # reads it
        except Exception as e:  # except: an unreadable log would lose its old rows on the next write
            die(f"the conversion log {logp.name} cannot be read ({e}); stopping rather than overwrite it")  # stops
        rewrite = lg0.command.astype(str).str.contains(str(PROJECT) + "/", regex=False).any()  # absolute paths?
    for r_ in log_rows:  # the rows of this run's conversions
        r_["path_note"] = "relative paths at conversion"  # their note
    # wall-clock seconds are not reproducible, so they leave the committed log for an uncommitted run log
    if log_rows:  # if: something was converted on this run
        RUNLOG.mkdir(parents=True, exist_ok=True)  # Data/processed/runlogs (run logs, kept apart from the results)
        # list comprehension: one (test, stream, seconds) row per conversion, written over any earlier run log
        pd.DataFrame([{"test_code": r_["test_code"], "stream": r_["stream"], "seconds": r_["seconds"]} for r_ in log_rows]).to_csv(RUNLOG / "04_conversion_seconds.csv", index=False, lineterminator="\n")
        for r_ in log_rows:  # for loop over the rows again
            r_.pop("seconds", None)  # dict.pop with a default: the time removed from the committed row
    if log_rows or rewrite:  # if: new conversions, or old absolute paths to remove (the log is rewritten)
        new_log = pd.DataFrame(log_rows)  # new rows as a table
        if logp.exists() and logp.stat().st_size > 0:  # if: an earlier, non-empty log exists
            try:  # try: if reading fails, the except branch runs
                old_log = pd.read_csv(logp)  # reads it
            except Exception as e:  # except: an unreadable log would lose its old rows on the next write
                # stops with the reason: the old rows must not be lost
                die(f"the conversion log {logp.name} cannot be read ({e}); stopping rather than overwrite it")
            # the home-folder prefix removed from the old commands (paths relative to the project folder), noted per row;
            # nothing else in the old rows changes
            ab_ = old_log.command.astype(str).str.contains(str(PROJECT) + "/", regex=False)  # rows with it
            # the prefix removed from those commands (str.replace, plain text)
            old_log.loc[ab_, "command"] = old_log.loc[ab_, "command"].str.replace(str(PROJECT) + "/", "", regex=False)
            if "path_note" not in old_log:  # if: the column is new
                old_log["path_note"] = ""  # created empty
            old_log.loc[ab_, "path_note"] = PATH_NOTE  # the note on the rewritten rows
            old_log = old_log.drop(columns=["seconds"], errors="ignore")  # an older log's time column dropped (errors="ignore": absent is fine)
            if not old_log.empty and len(new_log):  # if: old rows exist and new ones too
                # keeps old rows whose (test, stream) pair was not just converted: a two-level index tested with isin()
                keep = old_log[~old_log.set_index(["test_code", "stream"]).index.isin(
                    # pairs just converted; ~ (not) inverts the isin mask
                    new_log.set_index(["test_code", "stream"]).index)]
                # stacks kept and new rows, renumbering the index
                new_log = pd.concat([keep, new_log], ignore_index=True)
            elif not len(new_log):  # elif: no new rows, only the path rewrite
                new_log = old_log  # the old log, rewritten
        # sorted by test and stream, written with "\n" line endings (same bytes on any system)
        new_log.sort_values(["test_code", "stream"]).to_csv(
            logp, index=False, lineterminator="\n")  # no index column, "\n" line endings
    if log_rows or rewrite:  # if: the log was written on this run
        # says what was written: new rows and whether paths were rewritten
        print(f"\n      Conversion log written ({len(log_rows)} new row(s); "
              f"absolute paths {'removed' if rewrite else 'none'}).")  # says so
    else:  # else: nothing converted, so nothing written
        print("\n      Nothing converted on this run; the conversion log is unchanged.")  # says so
    print(f"      Free space now {free_gb(DATA):.1f} GB.")  # prints free space now

    # ---- verify -------------------------------------------------------
    print("\n" + "=" * 78)  # banner
    print("[5] VERIFICATION")  # fixed text
    print("=" * 78)  # banner
    print("  Nothing here is accepted because a command exited zero.")  # fixed text
    band_failures = []  # empty list: streams missing a required band
    # per-satellite orbit bounds pooled over every navigation file of the day (the sky is the same; QZSS and
    # NavIC records exist in a few files only); records far from the recording day are rejected and counted
    # the bounds come from the IGS file only (I/NAV for Galileo, date filter, no health filter)
    eph_rej: Counter = Counter()  # rejected records by reason
    day_ = pd.Timestamp(sessions.capture_start_utc.iloc[0]).date()  # the recording day
    # the IGS file is checked against its logged SHA-256 before anything is read from it
    igs_sha_ = hashlib.sha256(IGS_BRDC.read_bytes()).hexdigest()  # the file's SHA-256, computed now
    if igs_sha_ != IGS_SHA256:  # if: not the file downloaded and logged in 1
        die(f"{IGS_BRDC.name}: SHA-256 {igs_sha_} differs from the logged {IGS_SHA256}")  # stops the run
    igs_lines_ = nav_text_lines(IGS_BRDC)  # its lines
    k_hdr_ = next(i for i, l in enumerate(igs_lines_) if l[60:].startswith("END OF HEADER")) + 1  # first data line
    igs_sats = {l[:3] for l in igs_lines_[k_hdr_:] if l[:1].isalpha()}  # set: every satellite with an IGS record
    # prints the file, its RINEX version (first header line, columns 1-9) and the hash check
    print(f"      IGS broadcast file {IGS_BRDC.name}: RINEX {igs_lines_[0][:9].strip()}, SHA-256 verified; "
          f"{len(igs_sats)} satellites")  # f-string continued: how many satellites the file holds
    gal_chk = galileo_source_check(IGS_BRDC)  # the Galileo data-source consistency
    # prints the data-source counts: records per class, both-kind records, clock-bit failures
    print(f"      IGS Galileo data-source field: {gal_chk['counts']}; both I/NAV and F/NAV bits: "
          # f-string continued: records with both data-source bits, and records failing the clock-pair check
          f"{len(gal_chk['both'])} {gal_chk['both']}; clock-pair bit failures: {gal_chk['clock_bit_failures']}")
    igs_unh: Counter = Counter()  # unhealthy IGS records entering the bounds, by constellation
    eph = ephemeris_bounds([IGS_BRDC], day_, eph_rej, inav_only=True, unhealthy=igs_unh)  # satellite -> bounds
    # D3: the receivers' navigation files, the superseded 2.6.12 UBX_01 file explicitly left out
    rx_navs = [p_ for p_ in sorted(RINEX.glob("*/*.nav")) if "_superseded" not in p_.name]  # list comprehension
    rx_rej: Counter = Counter()  # their rejected records (the off-date evidence, kept in the run log)
    ephemeris_bounds(rx_navs, day_, rx_rej)  # read only for that count; its bounds are not used
    # satellites with receiver records but none in the IGS file get bounds only from trusted receiver records
    rx_only = {l[:3] for p_ in rx_navs for l in nav_text_lines(p_) if l[:1].isalpha() and l[1:3].strip().isdigit()}
    rx_only -= igs_sats  # set difference: those absent from IGS
    ts_ = lambda x: pd.Timestamp(x, tz="UTC").to_pydatetime()  # lambda: a UTC text as a timezone-aware datetime
    # list comprehensions: (test, start, end) of every scheduled window, capture, and attack period (window extended)
    win_ = [(x.test_code, ts_(x.scheduled_start_utc), ts_(x.scheduled_end_utc)) for x in sessions.itertuples()]
    # captures: (test, start, end) of every capture
    cap_ = [(x.test_code, ts_(x.capture_start_utc), ts_(x.capture_end_utc)) for x in sessions.itertuples()]
    # attack periods: each scheduled window, its end extended where ATTACK_EXTENDED_END says so
    att_ = [(c_, a_, max(b_, ts_(ATTACK_EXTENDED_END[c_])) if c_ in ATTACK_EXTENDED_END else b_) for c_, a_, b_ in win_]
    fb_rows, fb_trusted = fallback_records(rx_navs, rx_only, att_, win_, cap_)  # the fallback table and decision
    fb_eph = ephemeris_bounds(rx_navs, day_, Counter(), keep=lambda sv_, t_: t_ in fb_trusted)  # trusted only
    eph.update({k_: v_ for k_, v_ in fb_eph.items() if k_ in rx_only})  # added only for satellites absent from IGS
    # the fallback table, written to the check folder (fixed column order, "\n" line endings)
    pd.DataFrame(fb_rows, columns=["sv", "toc", "iod_aode", "toe_s_of_week", "broadcast_utc",
                                   "inside_scheduled_window_of", "inside_capture_of", "inside_attack_period_of",
                                   "receivers", "n_receivers", "files", "used", "reason"]).to_csv(  # list element: the last column names of the fallback-record table
        CHECK_DIR / "fallback_records.csv", index=False, lineterminator="\n")  # argument of to_csv(): the output path of the fallback-record table, Unix line ends
    absent_cnt: Counter = Counter()  # (test, stream, satellite) -> pseudoranges of satellites absent from IGS
    # the site of each session (the handset script's measured median position; 2.6.5 uses its parked part) -> geocentric
    # radius of the WGS84 ellipsoid at that latitude (height ignored: tens of metres against thousands of km)
    dist_ = pd.read_csv(PROCESSED / "ground_truth" / "session_distance_to_sender.csv")  # the handset script's per-session positions
    site_ = dist_.dropna(subset=["median_lat"]).groupby("test_code").last()  # the measured part of each session
    # nested function: WGS84 geocentric radius in km at a geodetic latitude in degrees
    def r_wgs84(lat):
        c_, s_ = math.cos(math.radians(lat)), math.sin(math.radians(lat))  # cosine and sine of the latitude
        return math.sqrt(((WGS84_A_KM ** 2 * c_) ** 2 + (WGS84_B_KM ** 2 * s_) ** 2)  # the ellipse formula
                         / ((WGS84_A_KM * c_) ** 2 + (WGS84_B_KM * s_) ** 2))  # arithmetic continued: divided by the matching sum of squares, giving the radius at that latitude
    site_geo = {c_: (r_wgs84(r_.median_lat), r_.median_lat) for c_, r_ in site_.iterrows()}  # test -> (R km, lat)
    phys_rows = []  # physics rows of every stream (written to preprocessing/rinex_physics_ranges.csv)
    keph = ephemeris_kepler([IGS_BRDC], day_, inav_only=True)  # clock-term orbits from IGS only
    clk_rows, clk_tests, clk_ep = [], [], pd.DataFrame()  # summary rows, the u-blox 01's clock release tests, the 2.6.12 UBX_01 epochs
    minute_rows, no_true_time = [], Counter()  # per-minute rows; epochs without a true time, per stream
    for _, sr in sessions.iterrows():  # for loop over sessions
        folder = DAY / sr["session_folder"]  # raw folder
        outdir = RINEX / sr["session_folder"]  # RINEX folder
        for st, fmt, name, role in STREAMS:  # for loop over streams
            src = next(iter(folder.glob(f"*{st}.*")), None)  # the raw file
            if src is None:  # if: no raw file for this stream
                continue  # continue: next stream
            obs = outdir / f"{src.stem}.obs"  # the uncompressed OBS path (absent on a run again)
            tmp_obs = None  # None unless a temporary copy is made
            tmp_parts = []  # the pieces' temporary text copies, read in order as one stream
            if (sr["test_code"], st) in PIECEWISE:  # if: a stream converted piece by piece
                import hatanaka  # the compression package
                for k_ in range(1, len([x for x in piece_rows if x["test_code"] == sr["test_code"]]) + 1):  # pieces
                    tx_ = hatanaka.decompress(outdir / f"{src.stem}_part{k_:02d}.crx.gz")  # piece k, in memory
                    tp_ = Path(TMPDIR) / f"{src.stem}_part{k_:02d}.obs"  # its temporary text copy
                    tp_.write_bytes(tx_ if isinstance(tx_, bytes) else tx_.encode())  # written
                    tmp_parts.append(tp_)  # kept, in order
            elif not obs.exists():  # elif: no uncompressed OBS on disk (every stream on a run again)
                # On a run again the observation file has already been compressed and
                # removed, so it is restored to a temporary file for verification
                # and discarded again. The script stays reproducible and the stored
                # artefact stays compressed.
                # the compressed file; the TMPDIR folder is removed at the end of main
                comp = next(iter(outdir.glob(f"{src.stem}.crx*")), None)
                if comp is None:  # if: no compressed file either
                    die(f"{sr['test_code']} {st}: no RINEX observation file to verify")  # 42i: stops, never skips
                try:  # try: a decompression failure jumps to except
                    # import inside the function: the hatanaka package (2.8.1 here), needed only on this path
                    import hatanaka
                    # hatanaka.decompress: Hatanaka + gzip -> plain RINEX text in memory (str or bytes)
                    txt = hatanaka.decompress(comp)
                    tmp_obs = Path(TMPDIR) / f"{src.stem}.obs"  # temp file name in the TMPDIR folder
                    # writes it (encode() turns str into bytes); inside the project: TMP_ROOT is codes/.tmp_rnx, cleaned up at the end
                    tmp_obs.write_bytes(txt if isinstance(txt, bytes) else txt.encode())
                    obs = tmp_obs  # the temporary copy is read from now on
                except Exception as e:  # except: any failure
                    # 42i: a stream that cannot be restored would drop out of verification, so the run stops
                    die(f"could not restore {comp.name} for verification: {e}")
            # tuple unpacking: binary total, measurement-bearing count, first and last time
            cb = count_binary_epochs(src, fmt)  # the native stream's epoch counts and diagnostics
            nbin, nmeas = cb["n"], cb["n_meas"]  # all epochs and measurement-bearing epochs
            # 41d/44c: u-blox epochs are on the receiver's clock, which the attack moved, so their gaps and
            # completeness are measured on host time; each RINEX epoch is matched to its RXM-RAWX message by the
            # receiver time tag to the millisecond, and the message's host time read from the sidecar
            hfirst = hlast = hsp_ = None  # host time of the first and last RXM-RAWX, measurement-bearing span
            hmap = {}  # receiver time (ms) -> host time, first occurrence kept (u-blox only)
            ms = lambda t_: round(t_.timestamp() * 1000)  # lambda: a time as integer milliseconds (the match key)
            if fmt == "ubx" and cb["ubx_epochs"]:  # if: a u-blox stream with RXM-RAWX
                ue = cb["ubx_epochs"]  # (receiver time, end offset, has measurement) in file order
                # host UTC of every RXM-RAWX message end, from the sidecar and the laptop clock
                hts = host_times(folder / "host_timing" / f"{src.name}.hosttime.csv", [e for _, e, _ in ue],
                                 host_clock(folder))  # host UTC of every message end
                hfirst, hlast = hts[0], hts[-1]  # first and last in file order
                for (t_, _, _), h_ in zip(ue, hts):  # each RXM-RAWX
                    hmap.setdefault(ms(t_), h_)  # kept once
                hm_ = [h_ for (_, _, m_), h_ in zip(ue, hts) if m_]  # host times of the measurement-bearing RAWX
                hsp_ = (max(hm_) - min(hm_)).total_seconds() if hm_ else None  # their span on host time, s
            ss = pd.Timestamp(sr["scheduled_start_utc"], tz="UTC").to_pydatetime()  # scheduled start
            se = pd.Timestamp(sr["scheduled_end_utc"], tz="UTC").to_pydatetime()  # scheduled end
            # the 2.6.12 u-blox values inside the attack get their own rows, as an attack
            # effect; the window runs from the scheduled start to the later of the scheduled end and the clock
            # releases the receiver-status script measured for that receiver (ground_truth/spoof_2612_timeline.csv, host time)
            att_end = se  # the attack window's end, extended below if a release came later
            split_attack = fmt == "ubx" and sr["test_code"] == "2.6.12"  # bool: only the 2.6.12 u-blox streams
            if split_attack and (PROCESSED / "ground_truth" / "spoof_2612_timeline.csv").exists():  # if: the receiver-status script's timeline exists
                tq_ = pd.read_csv(PROCESSED / "ground_truth" / "spoof_2612_timeline.csv", low_memory=False)  # the timeline
                # boolean masks: this receiver's release events with a time
                rl_ = tq_[(tq_.receiver == f"u-blox_{st[-2:]}") & tq_.event.str.endswith("release")
                          & tq_.host_time_utc.notna()].host_time_utc  # this receiver's release times
                for h_ in rl_:  # each release time
                    # the later of the window end and this release, as a UTC datetime
                    att_end = max(att_end, datetime.fromisoformat(f"{ss.date()} {h_}").replace(tzinfo=TZ_UTC))

            # nested function: the physics period of an epoch - inside or outside the scheduled window of our own
            # spoofing (receiver time for the Septentrio, host time for the u-blox), and for the 2.6.12 u-blox the
            # attack window labelled as an attack effect; the label never calls a value a fault
            def period_of(t_):
                h_ = hmap.get(ms(t_)) if fmt == "ubx" else t_  # the time to classify by
                if h_ is None:  # if: a u-blox epoch with no RXM-RAWX matched
                    return "no host time"  # its own period
                if split_attack:  # if: a 2.6.12 u-blox stream - before, during or after the attack (host time)
                    if h_ < ss:  # if: before the scheduled start
                        return "before the attack window"  # its own period
                    # conditional expression: the attack window, or after the clocks were released (7: the
                    # post-release epochs still carry pseudoranges inconsistent with the orbits; their own rows)
                    return "attack window (attack effect)" if h_ <= att_end else "after the attack window (clock released)"
                # conditional expression: inside our scheduled window or not
                return "inside the scheduled window" if ss <= h_ < se else "outside the scheduled window"

            obs_files = tmp_parts or [obs]  # the file(s) to read: the pieces in order, or the one file
            codes, tsys = rinex_header_multi(obs_files)  # the code table (union over pieces) and time system
            req = REQUIRED_BANDS[fmt]  # this family's feature bands
            rcount: Counter = Counter()  # reader counts: bad fields, epoch flags, undeclared constellations
            pa = PhysAcc(eph, *site_geo.get(sr["test_code"], (WGS84_A_KM, 0.0)))  # physics accumulator
            epochs, band_ep = [], Counter()  # (time, n_sat) per epoch; epochs in which each feature band is present
            # the MeasEpoch cross-check walks the binary alongside the RINEX, epoch by epoch (Septentrio only)
            mskips: Counter = Counter()  # decoder skips by reason
            xa = XAcc() if fmt == "sbf" else None  # the cross-check accumulator
            mit = iter_measepoch(src, mskips) if fmt == "sbf" else None  # the binary's epochs in file order
            mcur = next(mit, None) if mit is not None else None  # the current binary epoch
            fser = [] if st == "SBF_02" else None  # the figure series (X5): feature-band satellites and C/N0 per epoch
            ge_rows, ep_info, tt_ = [], {}, {}  # GPS/Galileo pseudoranges; key -> (time, period); key -> true time
            e1_ = "C1C" if fmt == "sbf" else "C1X"  # Galileo E1 pseudorange code of this family
            for t_, nsat_, sats_ in iter_rinex_multi(obs_files, rcount):  # every epoch of the RINEX
                epochs.append((t_, nsat_))  # kept
                per_ = period_of(t_)  # the epoch's period
                pa.add(per_, sats_)  # every value against the physics bounds
                for sv_, v_ in sats_.items():  # each satellite of the epoch (those absent from IGS are counted)
                    if sv_ not in igs_sats:  # if: no IGS record for it
                        # adds the satellite's non-zero pseudoranges in this epoch (generator inside sum())
                        absent_cnt[(sr["test_code"], st, sv_)] += sum(1 for c_, x_ in v_.items() if c_[0] == "C" and x_)
                k_ = ms(t_); ep_info[k_] = (t_, per_)  # the epoch's key, time and period
                tt_[k_] = hmap.get(k_) if fmt == "ubx" else t_  # D4: its true time (host time for the u-blox)
                tg_ = (t_ - GPS_EPOCH).total_seconds() + LEAP_SECONDS  # GPS time of the epoch, s
                # GPS L1 C/A and Galileo E1 pseudoranges for the common clock term
                ge_rows += [(k_, tg_, sv, v["C1C"] if sv[0] == "G" else v[e1_]) for sv, v in sats_.items()
                            if (sv[0] == "G" and v.get("C1C")) or (sv[0] == "E" and v.get(e1_))]  # generator clause: keeps GPS and Galileo satellites that carry an L1 or E1 pseudorange
                if mit is not None:  # if: a Septentrio stream - match the binary epoch by time to the millisecond
                    k_ = ms(t_)  # this epoch's key
                    while mcur is not None and ms(mcur[0]) < k_:  # binary epochs with no RINEX epoch
                        # counts it by kind (conditional expression on whether it carries signals)
                        xa.skip["binary epoch without RINEX epoch, with signals" if mcur[1] else
                                "binary epoch without RINEX epoch, empty"] += 1  # counted by kind
                        mcur = next(mit, None)  # next binary epoch
                    if mcur is not None and ms(mcur[0]) == k_:  # if: the same epoch
                        xa.add(mcur[1], sats_)  # compared signal by signal
                        mcur = next(mit, None)  # next binary epoch
                    else:  # else: a RINEX epoch the binary does not have
                        xa.skip["RINEX epoch without binary epoch"] += 1  # counted
                if fser is not None:  # if: the X5, for the figure (feature bands only, every epoch)
                    fb_ = [v.get(f"S{req[sv[0]][0]}") for sv, v in sats_.items()  # C/N0 of each satellite with a
                           if sv[0] in req and f"C{req[sv[0]][0]}" in v]  # feature-band pseudorange
                    # appends (time, feature-band satellites, their median C/N0; NaN when none)
                    fser.append((t_, len(fb_), float(np.median([x for x in fb_ if x])) if any(fb_) else np.nan))
                for con, (code, _) in req.items():  # each feature band
                    # generator inside any(): a satellite of that constellation with code, phase, Doppler and C/N0
                    if any(sv[0] == con and all(f"{p}{code}" in v for p in "CLDS") for sv, v in sats_.items()):
                        band_ep[con] += 1  # the band is present in this epoch
            # ----: the common clock term per epoch, its residual spread, and the physics count without it --
            lat_, lon_ = (math.radians(site_.loc[sr["test_code"], c_]) for c_ in ("median_lat", "median_lon"))  # site
            e2_ = 6.69437999014e-3  # WGS84 first eccentricity squared
            Nn_ = WGS84_A_KM * 1000 / math.sqrt(1 - e2_ * math.sin(lat_) ** 2)  # prime-vertical radius, m
            rx_ = np.array([(Nn_ + RX_HEIGHT_M) * math.cos(lat_) * math.cos(lon_),  # site ECEF X
                            (Nn_ + RX_HEIGHT_M) * math.cos(lat_) * math.sin(lon_),  # site ECEF Y
                            (Nn_ * (1 - e2_) + RX_HEIGHT_M) * math.sin(lat_)])  # site ECEF Z
            # local vertical (geodetic) at the site, for the elevation
            up_ = np.array([math.cos(lat_) * math.cos(lon_), math.cos(lat_) * math.sin(lon_), math.sin(lat_)])
            ge_sv_ = {x[2] for x in ge_rows}  # set: the GPS and Galileo satellites observed in this stream
            if ge_sv_ - igs_sats:  # if: one of them has no record at all in the IGS file
                die(f"{sr['test_code']} {st}: no IGS record for {sorted(ge_sv_ - igs_sats)}; the clock term needs them")  # stops
            if ge_sv_ - set(keph):  # if: observed satellites whose IGS records are all unhealthy (or F/NAV only)
                # printed: those satellites are left out of the clock term, as before 
                print(f"      {sr['test_code']} {st}: no usable IGS record (unhealthy) for {sorted(ge_sv_ - set(keph))}")
            clk = clock_terms(pd.DataFrame(ge_rows, columns=["key", "t_gps", "sv", "pr"]), keph, rx_, up_)  # c*dt
            clk["period"] = clk.key.map(lambda k: ep_info[k][1])  # each estimate's period
            clk["t"] = clk.key.map(lambda k: ep_info[k][0])  # and its time (receiver time)
            cdt_of = dict(zip(clk.key, clk.cdt_m))  # epoch key -> c*dt, m
            for t_, nsat_, sats_ in iter_rinex_multi(obs_files, Counter()):  # a second pass over the RINEX
                pa.add_clock(period_of(t_), sats_, cdt_of.get(ms(t_)))  # pseudoranges minus the clock term
            # the u-blox 01's clock release test: after the release, do the residuals look like the pre-test margin's?
            pre_ = clk[clk.period == "before the attack window"].res_mad_m  # pre-test residual spread per epoch
            aft_ = clk[clk.period == "after the attack window (clock released)"].res_mad_m  # after the release
            if split_attack and len(pre_) and len(aft_):  # if: a 2.6.12 u-blox stream with both periods
                common_ = aft_.median() <= CLK_COMMON_RATIO * pre_.median()  # the criterion
                # keeps the decision with its two medians
                clk_tests.append({"test_code": sr["test_code"], "stream": st, "pre_test_median_mad_m": round(pre_.median(), 2),
                                  # dict entry: median residual spread after the clock release, and the ratio allowed
                                  "after_release_median_mad_m": round(aft_.median(), 2), "ratio_max": CLK_COMMON_RATIO,
                                  "common_to_all_satellites": bool(common_)})  # the decision, kept
                if common_:  # if: the residuals after removing c*dt look like the pre-test margin
                    new_ = "receiver clock offset after release (common to all satellites)"  # the new label
                    # dict comprehension: the period relabelled in every accumulated row
                    pa.acc = {((new_ if k[0] == "after the attack window (clock released)" else k[0]),) + k[1:]: v
                              for k, v in pa.acc.items()}  # dict comprehension: the period relabelled
                    clk.loc[clk.period == "after the attack window (clock released)", "period"] = new_  # likewise
            # ----: per minute of true time, the residual spread after the common clock term ----------
            mk_ = clk[clk.key.map(lambda k: tt_.get(k) is not None)].copy()  # epochs with a true time
            no_true_time[(sr["test_code"], st)] = len(clk) - len(mk_)  # u-blox epochs without host time, counted
            mk_["tt"] = pd.to_datetime([tt_[k] for k in mk_.key], utc=True)  # true times as UTC timestamps
            mk_["minute"] = mk_.tt.dt.floor("min").dt.strftime("%Y-%m-%d %H:%M")  # the minute each epoch falls in
            mk_["site_part"] = ""  # the handset script's moving/parked parts (2.6.5 only; other sessions are one part)
            # boolean mask: this session's part rows that split the capture (not "whole capture")
            pr_ = dist_[(dist_.test_code == sr["test_code"]) & (dist_.part != "whole capture")]
            for _, q_ in pr_.iterrows():  # each part, in table order (moving first, so the shared instant is parked)
                a_, b_ = pd.Timestamp(q_.part_start_utc, tz="UTC"), pd.Timestamp(q_.part_end_utc, tz="UTC")  # bounds
                mk_.loc[(mk_.tt >= a_) & (mk_.tt <= b_), "site_part"] = q_.part  # epochs inside it labelled
            for (mi_, pe_, sp_), g_ in mk_.groupby(["minute", "period", "site_part"]):  # one row per minute and label
                # appends the minute's row: epochs, satellites, spread, and the satellites more than 100 m off
                minute_rows.append({"test_code": sr["test_code"], "stream": st,
                                    "stream_note": "excluded stream, 43g" if st in EXCLUDED else "",  # dict entry: marks a stream that is excluded, so its rows are read with care
                                    "minute_utc": mi_, "period": pe_, "site_part": sp_, "epochs": len(g_),  # dict entry: the minute, its period, the site part and the number of epochs
                                    "median_sats": float(g_.n_sats.median()),  # dict entry: median number of satellites in the minute
                                    "residual_mad_m_median": round(g_.res_mad_m.median(), 2),  # dict entry: median residual spread in metres
                                    "share_over_100m_pct": round(100 * g_.n_over_100m.sum() / g_.n_sats.sum(), 1),  # dict entry: share of residuals over 100 m as a percentage
                                    "median_sats_over_100m": float(g_.n_over_100m.median())})  # dict entry: median number of satellites more than 100 m off
            for per_, g_ in clk.groupby("period"):  # per period: a summary row
                g_ = g_.sort_values("t")  # in time order
                dts_ = g_.t.diff().dt.total_seconds().to_numpy()[1:]  # time steps, s
                dc_ = np.diff(g_.cdt_m.to_numpy())  # c*dt steps, m
                sm_ = np.abs(dc_) <= CLK_JUMP_M  # steps that are not jumps
                # appends the period's summary row: estimates, c*dt range, drift, jumps, residual spread
                clk_rows.append({"test_code": sr["test_code"], "stream": st, "period": per_,
                                 "epochs_with_estimate": len(g_), "median_sats": float(g_.n_sats.median()),  # dict entry: epochs with a clock estimate, and the median satellites used
                                 "cdt_km_median": round(g_.cdt_m.median() / 1e3, 3), "cdt_km_min": round(g_.cdt_m.min() / 1e3, 3),  # dict entry: median and minimum receiver clock term, in kilometres
                                 "cdt_km_max": round(g_.cdt_m.max() / 1e3, 3),  # dict entry: maximum receiver clock term, in kilometres
                                 "cdt_ms_median": round(g_.cdt_m.median() / C_LIGHT * 1e3, 4),  # dict entry: median clock term as a time, in milliseconds
                                 # how fast it changes between jumps: the median of step/interval, m/s and ns/s
                                 "drift_m_per_s": round(float(np.median(dc_[sm_] / dts_[sm_])), 3) if sm_.any() else None,
                                 "drift_ns_per_s": round(float(np.median(dc_[sm_] / dts_[sm_])) / C_LIGHT * 1e9, 2)  # dict entry: the same drift in nanoseconds per second
                                 if sm_.any() else None,  # conditional expression: no drift when no step is free of a jump
                                 "n_jumps_over_1km": int((~sm_).sum()),  # clock jumps
                                 "residual_mad_m_median": round(g_.res_mad_m.median(), 2),  # spread after removal
                                 "residual_mad_m_p95": round(g_.res_mad_m.quantile(0.95), 2),  # dict entry: 95th percentile of the residual spread
                                 "residual_max_m_median": round(g_.res_max_m.median(), 2)})  # largest residual
            if (sr["test_code"], st) == ("2.6.12", "UBX_01"):  # if: the stream of u-blox 01's clock release - every epoch written out
                g_ = clk.sort_values("t")  # in time order
                # every 2.6.12 UBX_01 epoch with its clock term, written for u-blox 01's clock release
                clk_ep = pd.DataFrame({"receiver_time_utc": g_.t.dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3],
                                       "host_time_utc": [hmap[k].strftime("%H:%M:%S.%f")[:-3] if k in hmap else ""  # dict entry: host time of each epoch, empty when the epoch has none
                                                         for k in g_.key],  # host time of each epoch
                                       "period": g_.period, "n_sats": g_.n_sats,  # dict entry: the epoch's period and number of satellites
                                       "cdt_km": (g_.cdt_m / 1e3).round(4), "cdt_ms": (g_.cdt_m / C_LIGHT * 1e3).round(5),  # dict entry: the clock term in kilometres and in milliseconds
                                       # dict entry: residual spread and largest residual after the clock term is removed
                                       "residual_mad_m": g_.res_mad_m.round(2), "residual_max_m": g_.res_max_m.round(2)})
            phys_rows += pa.rows(sr["test_code"], st)  # this stream's physics rows
            if mit is not None:  # if: a Septentrio stream
                while mcur is not None:  # binary epochs after the last RINEX epoch
                    # counts it by kind
                    xa.skip["binary epoch without RINEX epoch, with signals" if mcur[1] else
                            "binary epoch without RINEX epoch, empty"] += 1  # counted by kind
                    mcur = next(mit, None)  # next
                cross_rows += xa.rows(sr["test_code"], st)  # the summary rows
                # appends the stream's epochs compared and its skips by reason (dict unpacking)
                xsk_rows.append({"test_code": sr["test_code"], "stream": st, "epochs_compared": xa.epochs,
                                 **{f"skip: {k}": v for k, v in sorted((xa.skip + mskips).items())}})  # skips
                pr_hist.update(xa.pr_hist); cn_steps.update(xa.cn_steps)  # pooled histograms for the figure
            le_ = cb["longest_empty"]  # (start, end, epochs) of the longest run without a measurement
            # median satellites per RINEX epoch before that run (None when there is no run or no epoch before it)
            pre_ = [n_ for t_, n_ in epochs if le_[0] is not None and t_ < le_[0]]
            if rcount["bad_epoch_lines"]:  # if: an epoch line did not parse (this count was never written anywhere)
                die(f"{sr['test_code']} {st}: {rcount['bad_epoch_lines']} RINEX epoch lines did not parse; their satellites would be dropped silently")  # stops
            # appends this stream's diagnostics (printed in (a3), never written to a CSV in this step)
            diag.append({"test_code": sr["test_code"], "stream": st, "dnu_times": cb["dnu_times"],
                         "revisions": cb["revisions"], "early_stop": cb["early_stop"], "longest_empty": le_,  # dict entry: revisions, early stop and the longest run of empty epochs
                         "median_sat_before_empty": float(np.median(pre_)) if pre_ else None,  # dict entry: median satellites before that empty run
                         "bad_fields": rcount["bad_fields"], "flags_nonzero": rcount["flags_nonzero"],  # dict entry: non-numeric RINEX fields and epochs with a non-zero flag
                         "rinex_days": sorted({t_.date().isoformat() for t_, _ in epochs})})  # dict entry: the distinct dates the RINEX epochs fall on
            # this stream's binary epoch times on a true clock (receiver time for the Septentrio, host time for
            # the u-blox), for the G5 gap comparison in (f)
            true_times[(sr["test_code"], st)] = ([t_ for t_, _, _ in cb["sbf_epochs"]] if fmt == "sbf"
                                                 else list(hts) if hmap else [])  # conditional expression: host times for the u-blox when available, otherwise nothing
            if hmap:  # if: a u-blox stream with host times
                ep_t = sorted((hmap[ms(t_)], n_) for t_, n_ in epochs if ms(t_) in hmap)  # RINEX epochs on host time
                # the true spacing: the median receiver-time difference (1 s or 0.2 s), robust to the few spoofed jumps
                rx_sp = float(np.median(np.diff([t_.timestamp() for t_, _ in epochs]))) if len(epochs) > 2 else None
                g = epoch_gaps(ep_t, nominal_s=rx_sp, span_s=hsp_)  # gaps and completeness on host time
                g["completeness_time_base"] = "host"  # stated
                g["rinex_epochs_without_host_time"] = len(epochs) - len(ep_t)  # unmatched epochs, counted
            else:  # else: Septentrio, whose MeasEpoch time stays true
                g = epoch_gaps(epochs)  # spacing statistics on receiver time
                g["completeness_time_base"] = "receiver"  # stated
                g["rinex_epochs_without_host_time"] = 0  # not applicable
            g["n_epochs"] = len(epochs)  # the RINEX epoch count, whatever the time base
            # binary measurement-bearing epochs that are absent from the RINEX, and the first one on a true clock
            rk_ = {ms(t_) for t_, _ in epochs}  # the RINEX epoch keys
            if fmt == "ubx":  # if: u-blox - receiver time keys, host time to report
                # list comprehension: host times of measurement-bearing RXM-RAWX absent from the RINEX
                miss_ = [h_ for (t_, _, m_), h_ in zip(cb["ubx_epochs"], hts if hmap else []) if m_ and ms(t_) not in rk_]
            else:  # else: Septentrio - receiver time is true
                miss_ = [t_ for t_, n_, _ in cb["sbf_epochs"] if n_ > 0 and ms(t_) not in rk_]  # absent times
            g["binary_epochs_absent_from_rinex"] = len(miss_)  # their count
            g["first_absent_utc"] = min(miss_).strftime("%H:%M:%S") if miss_ else ""  # the first (host time for u-blox)
            g["excluded"] = EXCLUDED.get(st, "")  # the exclusion, if any


            missing = []  # empty list of missing items
            for con, (code, label) in req.items():  # for loop over this family's feature bands
                have = codes.get(con, [])  # the header's codes for that constellation
                # the four needed: C code, L phase, D Doppler, S strength
                need = [f"{p}{code}" for p in ("C", "L", "D", "S")]
                miss = [c for c in need if c not in have]  # those absent from the header
                if con not in codes:  # if: the constellation is not declared at all
                    missing.append(f"{label}: constellation {con} absent")  # records it
                elif miss:  # elif: some codes missing
                    missing.append(f"{label}: {','.join(miss)} absent")  # records them
            # data presence, not only the header: a feature band present in no epoch at all counts as missing
            missing += [f"{lab}: in no epoch" for c_, (_, lab) in req.items() if c_ in codes and band_ep[c_] == 0]
            # if: missing on a Septentrio stream (header or data); u-blox gaps never stop the run
            if missing and fmt == "sbf":
                band_failures.append((sr["test_code"], st, missing))  # queues it for the stop below

            # Epochs carrying a spoofed time stamp. The u-blox writes RXM-RAWX on
            # receiver time, so when the clock is captured the measurement epochs
            # themselves move; the Septentrio's MeasEpoch time stays true and it
            # loses lock instead. Any later code that windows by time must know.
            cs_d = pd.Timestamp(sr["capture_start_utc"]).date()  # the capture date (the timeline script) as a date
            # epochs whose date differs from the capture date: the spoofed u-blox epochs
            off_date = [t for t, _ in epochs if t.date() != cs_d]
            # appends one verification row (a dict literal):
            ver_rows.append({
                "epochs_off_session_date": len(off_date),  # number of off-date epochs
                # first off-date epoch as text, or ""
                "off_date_first": off_date[0].strftime("%Y-%m-%d %H:%M:%S") if off_date else "",
                "off_date_last": off_date[-1].strftime("%Y-%m-%d %H:%M:%S") if off_date else "",  # last off-date epoch
                # first-to-last span in days (86400 s per day)
                "rinex_span_days": round((epochs[-1][0] - epochs[0][0]).total_seconds()
                                         / 86400, 3) if epochs else None,  # None if there are no epochs
                "test_code": sr["test_code"], "stream": st, "receiver": name,  # identity fields
                "binary_epochs": nbin,  # binary epoch total
                "binary_epochs_with_measurements": nmeas,  # binary epochs carrying measurements
                "binary_epochs_empty": nbin - nmeas,  # binary epochs with no satellites
                "rinex_epochs": g["n_epochs"],  # RINEX epoch count
                # RINEX / measurement-bearing epochs; None if zero
                "epoch_ratio": round(g["n_epochs"] / nmeas, 4) if nmeas else None,
                # RINEX / all binary epochs
                "epoch_ratio_vs_all_blocks": round(g["n_epochs"] / nbin, 4) if nbin else None,
                # first and last binary epoch in FILE order on the receiver's own time (no longer sorted)
                "binary_first_utc": cb["first_file_order"].strftime("%H:%M:%S") if cb["first_file_order"] else "",
                "binary_last_utc": cb["last_file_order"].strftime("%H:%M:%S") if cb["last_file_order"] else "",  # dict entry: last binary epoch in file order, on the receiver's own time
                # the same two on host time for the u-blox (empty for the Septentrio, whose own time is true)
                "binary_first_host_utc": hfirst.strftime("%H:%M:%S") if hfirst is not None else "",
                "binary_last_host_utc": hlast.strftime("%H:%M:%S") if hlast is not None else "",  # dict entry: last binary epoch on host time, empty for the Septentrio
                "rinex_first_utc": epochs[0][0].strftime("%H:%M:%S") if epochs else "",  # first RINEX time
                "rinex_last_utc": epochs[-1][0].strftime("%H:%M:%S") if epochs else "",  # last RINEX time
                # mean satellites per epoch, from the epoch lines (all constellations and bands)
                "mean_sat_per_epoch": round(float(np.mean([e[1] for e in epochs])), 2)
                                      if epochs else None,  # None if no epochs
                # fewest satellites in an epoch
                "min_sat_per_epoch": int(min(e[1] for e in epochs)) if epochs else None,
                "max_sat_per_epoch": int(max(e[1] for e in epochs)) if epochs else None,  # most satellites in an epoch
                "constellations": "".join(sorted(codes)),  # constellation letters declared in the header, sorted
                # one code column per constellation (G, R, E, C, J, I, S), dict unpacking of a comprehension
                **{f"obs_codes_{c_}": " ".join(codes.get(c_, [])) for c_ in CONSTELLATIONS},
                # the header test and the data test per feature band
                "required_codes_in_header": not [m_ for m_ in missing if "in no epoch" not in m_],
                "required_bands_missing": "; ".join(missing),  # the missing items as text (header or data)
                # share of RINEX epochs in which each feature band carries code, phase, Doppler and C/N0
                **{f"band_{c_}_present_pct": round(100 * band_ep[c_] / len(epochs), 2) if epochs else None
                   for c_ in "GREC"},  # generator clause: one presence share per feature constellation, GPS, GLONASS, Galileo, BeiDou
                **g})  # merges in the gap dict with ** (the physics ranges are in preprocessing/rinex_physics_ranges.csv)

            # series for the figure (X5 only): feature-band satellites and their median C/N0, from every epoch
            if fser is not None:  # if: the X5
                # binary epochs with nothing tracked (absent from the RINEX) enter as a count of 0, no C/N0
                fser = sorted(fser + [(t_, 0, np.nan) for t_, n_, _ in cb["sbf_epochs"] if n_ == 0])  # merged, by time
                ser = pd.DataFrame([(n_, c_) for _, n_, c_ in fser], columns=["n_sat", "cn0"],  # the two columns
                                   index=pd.DatetimeIndex([t_ for t_, _, _ in fser]))  # indexed by epoch time
                # stores the series as 10-s medians for the figure
                inv_rows.append({"test_code": sr["test_code"], "stream": st,
                                 "_series": ser.resample("10s").median()})  # "_series": 10-s medians via resample
            for tp_ in tmp_parts + ([tmp_obs] if tmp_obs is not None else []):  # every temporary copy
                if tp_.exists():  # if: it exists
                    tp_.unlink()  # deleted (the folder itself goes at the end of the run)

    if band_failures:  # if: a Septentrio band was missing
        print("\n  REQUIRED BANDS INCOMPLETE:")  # fixed text
        for code, st, miss in band_failures:  # for loop over the failures
            print(f"      {code} {st}: {'; '.join(miss)}")  # prints each one
        # stops the run; the text names Docker with sbf2rin as the escalation
        die("L1, G1 or B1 is incomplete in a converted stream. By design "
            "for this section the run stops here rather than working around it. The "
            "escalation is Docker with the manufacturer's sbf2rin for the Septentrio "
            "streams.")

    ver = pd.DataFrame(ver_rows)  # verification rows as a table
    if ver.empty:  # if: the verification table has no rows
        die("verification produced no rows; nothing was read")  # die(): stops the run with this reason
    # drops helper columns starting with "_" (none here) and writes preprocessing/rinex_verification.csv
    ver.drop(columns=[c for c in ver.columns if c.startswith("_")], errors="ignore").to_csv(
        # without the index, with "\n" line endings
        PROCESSED / "preprocessing" / "rinex_verification.csv", index=False, lineterminator="\n")
    # ---- the decision: the piece index and its checks --------------------------------------------------------
    pcs = pd.DataFrame(piece_rows)  # one row per piece
    pcs.to_csv(PROCESSED / "preprocessing" / "rinex_ubx01_2612_pieces.csv", index=False, lineterminator="\n")  # written (new)
    for code_, g_ in pcs.groupby("test_code"):  # each stream converted in pieces
        vr_ = ver[(ver.test_code == code_) & (ver.stream == "UBX_01")].iloc[0]  # its verification row
        tot_ = int(g_.rinex_epochs.sum())  # RINEX epochs over all pieces
        # bool: every check of the pieces passes (count, order, one-to-one, reproducible cut)
        ok_ = (tot_ == int(vr_.binary_epochs_with_measurements) and g_.rinex_strictly_increasing.all()
               and g_.rinex_equals_binary_one_to_one.all() and (g_.sha256 == g_.sha256_second_cut).all())  # the checks of the pieces
        # the checks, printed
        print(f"\n  (a4) u-blox 01 of 2.6.12 converted in pieces, {code_} UBX_01 in {len(g_)} pieces: RINEX epochs {tot_} against "
              # f-string continued: the binary epochs with measurements, and whether epochs strictly increase
              f"{int(vr_.binary_epochs_with_measurements)} measurement-bearing RXM-RAWX epochs; strictly increasing "
              f"in every piece {bool(g_.rinex_strictly_increasing.all())}; one-to-one with the binary "  # f-string continued: whether every piece increases, and whether the epochs match one for one
              f"{bool(g_.rinex_equals_binary_one_to_one.all())}; second cut identical "  # f-string continued: the one-to-one result, and whether a second cut gives the same pieces
              f"{bool((g_.sha256 == g_.sha256_second_cut).all())}")  # the checks, printed
        if not ok_:  # if: any check fails
            die(f"the decision checks fail for {code_} UBX_01 (see preprocessing/rinex_ubx01_2612_pieces.csv)")  # stops
    pd.DataFrame(clk_rows).to_csv(PROCESSED / "preprocessing" / "rinex_clock_common_term.csv", index=False, lineterminator="\n")  # new
    # the per-minute table, fixed column order, "\n" line endings
    pd.DataFrame(minute_rows, columns=["test_code", "stream", "stream_note", "minute_utc", "period", "site_part",
                                       "epochs", "median_sats", "residual_mad_m_median", "share_over_100m_pct",
                                       "median_sats_over_100m"]).to_csv(  # list element: the last column of the per-minute residual table
        PROCESSED / "preprocessing" / "rinex_residual_minutes.csv", index=False, lineterminator="\n")  # argument of to_csv(): the output path of the per-minute table, Unix line ends
    # prints the row count and the u-blox epochs left out for lack of a host time
    print(f"      rinex_residual_minutes.csv: {len(minute_rows)} rows; epochs without a true time (left out): "
          # string concatenation: lists streams with epochs left out for lack of host time, or none
          + (", ".join(f"{c_} {s_} {n_}" for (c_, s_), n_ in sorted(no_true_time.items()) if n_) or "none") + ".")
    clk_ep.to_csv(PROCESSED / "preprocessing" / "rinex_clock_2612_ubx01.csv", index=False, lineterminator="\n")  # new
    phys = pd.DataFrame(phys_rows)  # the physics table, one row per stream, period, constellation and code
    phys.to_csv(PROCESSED / "preprocessing" / "rinex_physics_ranges.csv", index=False, lineterminator="\n")  # written (new)
    cross = pd.DataFrame(cross_rows)  # the cross-check over every epoch, one row per stream, constellation, code
    cross.to_csv(PROCESSED / "preprocessing" / "rinex_measepoch_crosscheck.csv",  # writes preprocessing/rinex_measepoch_crosscheck.csv
                 index=False, lineterminator="\n")  # without the index
    xsk = pd.DataFrame(xsk_rows).fillna(0)  # skip counts per stream (0 where a reason did not occur)
    xsk.to_csv(PROCESSED / "preprocessing" / "rinex_measepoch_crosscheck_skips.csv", index=False, lineterminator="\n")  # written

    # ---- report the verification ---------------------------------------
    print("\n  (a) EPOCH COUNTS, RINEX against the native binary")  # fixed text
    print("      A Septentrio MeasEpoch block is emitted every epoch whether or not the")  # continued
    print("      receiver is tracking anything. Blocks carrying no satellites cannot be")  # continued
    print("      represented in RINEX and are correctly omitted, so the ratio that means")  # continued
    print("      anything is RINEX against MEASUREMENT-BEARING epochs, not against raw")  # continued
    print("      block count. Both are shown.")  # continued
    # header row (f-string; :<8 left-aligns, :>8 right-aligns in 8 characters)
    print(f"\n      {'test':<8}{'stream':<8}{'blocks':>8}{'empty':>8}{'with meas':>10}"
          f"{'rinex':>8}{'ratio':>8}{'vs blocks':>10}")  # continued
    for _, r in ver.iterrows():  # for loop over verification rows
        # one line per stream; "or 0" replaces a None ratio with 0
        print(f"      {r.test_code:<8}{r.stream:<8}{r.binary_epochs:>8}"
              f"{r.binary_epochs_empty:>8}{r.binary_epochs_with_measurements:>10}"  # continued
              f"{r.rinex_epochs:>8}{(r.epoch_ratio or 0):>8.4f}"  # continued
              f"{(r.epoch_ratio_vs_all_blocks or 0):>10.4f}")  # continued
    bad = ver[(ver.epoch_ratio < 0.999) | (ver.epoch_ratio > 1.001)]  # boolean mask: ratio outside 0.999-1.001
    if bad.empty:  # if: none outside
        print("\n      Every measurement-bearing epoch was converted: the ratio is 1.0000")  # fixed text
        print("      throughout. The shortfall against raw block count is entirely empty")  # continued
        print("      epochs, and those are the attack rather than a conversion loss.")  # continued
        # note; the stream-session count is computed
        note("Conversion preserved every measurement-bearing epoch: the RINEX epoch "
             "count equals the count of native epochs carrying satellites exactly, for "
             f"all {len(ver)} converted stream-sessions. The apparent shortfall against "  # f-string continued: the number of stream-sessions converted
             "the raw block count is entirely accounted for by epochs in which the "
             "receiver tracked no satellites at all, which cannot be represented in "
             "RINEX and which are a consequence of the attacks rather than of the "
             "converter.")
    else:  # else: some streams fall short
        print("\n      Streams where not every measurement-bearing epoch reached the RINEX:")  # fixed text
        for _, r in bad.iterrows():  # for loop over the short streams
            # measurement-bearing epochs missing from the RINEX
            lost = int(r.binary_epochs_with_measurements - r.rinex_epochs)
            print(f"      {r.test_code:<8}{r.stream:<8}{lost:>6} epochs absent "  # prints them
                  f"(ratio {r.epoch_ratio:.4f})")  # continued
        # convbin returns from convobs without writing any epoch not later than the last one written
        # (RTKLIB convrnx.c line 1045, quoted in check 8); a receiver whose clock the attack moved forward and
        # then released writes later epochs that are 'earlier', so they are dropped. Each short stream is reported
        # with its count, the first absent epoch and, for a u-blox, the receiver-status script's release time for that receiver
        tq_ = pd.read_csv(PROCESSED / "ground_truth" / "spoof_2612_timeline.csv", low_memory=False)  # the receiver-status script's timeline
        for _, r in bad.iterrows():  # for loop over the short streams
            # boolean masks: this receiver's measurement-engine release row
            rl_ = tq_[(tq_.receiver == f"u-blox_{r.stream[-2:]}") & (tq_.event == "measurement-engine release")]
            # its time to the second, or '-' if the receiver-status script has none
            rel_ = rl_.host_time_utc.iloc[0][:8] if len(rl_) and isinstance(rl_.host_time_utc.iloc[0], str) else "-"
            # one anomaly per short stream, every value computed
            flag(f"{r.test_code} {r.stream}: {int(r.binary_epochs_absent_from_rinex)} measurement-bearing epochs "
                 # f-string continued: the first missing epoch on host time, and the release time measured earlier
                 f"are absent from the RINEX, the first at {r.first_absent_utc} (host time); the receiver-status script measured "
                 # f-string continued: the release time, and why the converter drops the later epochs
                 f"this receiver's measurement clock released at {rel_}. convbin drops every epoch not later than "
                 "the last one written (convrnx.c line 1045), so the epochs after the release are lost. "
                 # string concatenation: states whether the stream was excluded or must be reconverted in pieces
                 + (f"Not reconverted: {r.excluded}." if r.excluded else "Reconvert in pieces (the decision)."))


    print("\n      TOTAL LOSS OF LOCK, which is a result and not a defect:")  # fixed text
    tot_empty = int(ver.binary_epochs_empty.sum())  # total empty epochs across streams
    # for loop over streams with more than 100 empty epochs (a typed threshold)
    for _, r in ver[ver.binary_epochs_empty > 100].iterrows():
        pct = 100 * r.binary_epochs_empty / r.binary_epochs  # per cent empty
        print(f"      {r.test_code:<8}{r.stream:<8}{r.binary_epochs_empty:>7} epochs "  # prints it
              f"({pct:5.1f} %) with zero satellites tracked")  # continued
    if tot_empty > 100:  # if: more than 100 in total
        # the note is built from the longest empty run measured in the binary, not typed
        lg = max(diag, key=lambda x: x["longest_empty"][2])  # the stream with the longest run without measurements
        a_, b_, k_ = lg["longest_empty"]  # its start, end and epoch count
        dur_ = (b_ - a_).total_seconds()  # its length in seconds
        tl_ = PROCESSED / "ground_truth" / "spoof_2612_timeline.csv"  # the receiver-status script's timeline, for the gain onset (labelling data)
        on_ = None  # the gain onset on host time, if the receiver-status script recorded one for this receiver
        if tl_.exists() and lg["test_code"] == "2.6.12":  # if: the timeline exists and the run is in 2.6.12
            tq_ = pd.read_csv(tl_, low_memory=False)  # the timeline
            rx_ = {"SBF_02": "Septentrio_02", "SBF_03": "Septentrio_03"}.get(lg["stream"])  # its name in the receiver-status tables
            r_ = tq_[(tq_.receiver == rx_) & (tq_.event == "gain onset")]  # the gain-onset row
            if len(r_) and isinstance(r_.host_time_utc.iloc[0], str):  # if: an onset time exists
                # the onset as a UTC datetime on the run's date (fromisoformat parses the text)
                on_ = datetime.fromisoformat(f"{a_.date()} {r_.host_time_utc.iloc[0]}").replace(tzinfo=TZ_UTC)
        # records the note: stream, run length, times, satellites before and the gain lead, all computed
        note(f"Several sessions contain epochs in which the receiver tracked no satellites at all. The longest "
             f"run is on {lg['stream']} in {lg['test_code']}: {k_} epochs with nothing tracked, from "  # f-string continued: the stream, the session and the number of empty epochs
             # f-string continued: start and end of the empty run to a tenth of a second, and its length
             f"{pd.Timestamp(a_).round('100ms'):%H:%M:%S.%f}"[:-5] + f" to {pd.Timestamp(b_).round('100ms'):%H:%M:%S.%f}"[:-5] + f" ({int(dur_ // 60)} min "
             f"{dur_ % 60:.0f} s), after a median of {lg['median_sat_before_empty']:.0f} satellites per epoch "  # f-string continued: the seconds of the length and the median satellites before it
             "beforehand."
             # string concatenation: adds when the front-end gain fell, if that time is known
             + (f" The front-end gain fell at {pd.Timestamp(on_).round('100ms'):%H:%M:%S.%f}"[:-5] + " (the receiver-status script, host time), "
                # f-string continued: how many seconds before the measurements stopped the gain fell
                f"{(a_ - on_).total_seconds():.1f} s before the measurements stopped." if on_ is not None else ""))

    print("\n  (a2) EPOCHS CARRYING A SPOOFED TIME STAMP")  # fixed text
    off = ver[ver.epochs_off_session_date > 0]  # streams with off-date epochs
    if off.empty:  # if: no stream has off-date epochs
        print("      None: every converted epoch falls on the session date.")  # fixed text
    else:  # else: at least one stream has them
        print("      These streams wrote epochs dated outside the recording day, because")  # fixed text
        print("      the receiver's measurement clock was captured and the converter writes")  # continued
        print("      RINEX on receiver time. This is the attack succeeding, not an error,")  # continued
        print("      but any later code that windows by time must be aware of it.")  # continued
        print(f"\n      {'test':<8}{'stream':<8}{'off-date':>10}{'span (days)':>13}"  # header row
              f"   first off-date epoch      last")  # continued
        for _, r in off.iterrows():  # for loop
            # one line per stream: count, span in days, first and last off-date epoch
            print(f"      {r.test_code:<8}{r.stream:<8}{int(r.epochs_off_session_date):>10}"
                  f"{(r.rinex_span_days or 0):>13.3f}   {r.off_date_first}   {r.off_date_last}")  # continued
        # flag, with the days read from the RINEX epochs of the stream with the most off-date epochs
        off_in_ = off[off.excluded == ""] if (off.excluded == "").any() else off  # prefer streams not excluded
        od_ = off_in_.loc[off_in_.epochs_off_session_date.idxmax()]  # that stream's verification row
        # generator inside next(): the RINEX days of that stream, from the diagnostics
        dd_ = next(x["rinex_days"] for x in diag if x["test_code"] == od_.test_code and x["stream"] == od_.stream)
        # records the anomaly with the computed days
        flag(f"The converted RINEX for session {od_.test_code} does not span one day. The u-blox "
             "writes measurement epochs on receiver time, and the time-spoofing attack "
             f"moved that clock, so its {od_.stream} observation file runs from {dd_[0]} to "  # f-string continued: the stream and the first date its observation file reaches
             f"{dd_[-1]} with epochs dated on {len(dd_)} separate days. The Septentrio "  # f-string continued: the last date and how many separate days the epochs fall on
             "behaves differently: its MeasEpoch time stamps stay true and it loses "
             "lock instead. Any windowing by time in a later section must handle this "
             "explicitly, and the receiver-family difference matters for the "
             "cross-receiver transfer experiment.")
        # note; the counts come from the same verification row (the typed '792 of 1636' was wrong)
        note("Receiver families respond differently to time spoofing at the measurement "
             "level. The u-blox accepted the spoofed time into its raw measurement "
             f"epochs, so {int(od_.epochs_off_session_date)} of the {int(od_.rinex_epochs)} RINEX epochs of "  # f-string continued: epochs dated off the recording day, out of all RINEX epochs
             # f-string continued: the stream and session they belong to
             f"{od_.stream} in {od_.test_code} carry a date other than the recording date. The Septentrio kept true measurement time stamps and "
             "instead tracked nothing for nearly seventeen minutes. Both are recorded "
             "as results.")

    # (a3) step-1 diagnostics: counts that were previously silent
    print("\n  (a3) DECODER AND READER DIAGNOSTICS")  # heading
    for x in diag:  # one line per stream
        # one diagnostics line per stream
        print(f"      {x['test_code']:<8}{x['stream']:<8} Do-Not-Use times {x['dnu_times']}, MeasEpoch revisions "
              f"{x['revisions'] or '-'}, UBX early stop {x['early_stop'] if x['early_stop'] is not None else '-'}, "  # f-string continued: block revisions and where the u-blox stream stopped early
              f"non-numeric RINEX fields {x['bad_fields']}, non-zero epoch flags {x['flags_nonzero']}")  # f-string continued: non-numeric RINEX fields and epochs with a non-zero flag
        if x.get("measepoch_skips"):  # if: the MeasEpoch decoder counted skipped signals (42c-42e)
            # one line: the counts by reason, sorted by reason
            print("            MeasEpoch skips: " + ", ".join(f"{k} {v}" for k, v in sorted(x["measepoch_skips"].items())))
        if x["early_stop"] is not None:  # if: a UBX message ran past the end of the file
            # records the truncated message as an anomaly
            flag(f"{x['test_code']} {x['stream']}: the last UBX message is truncated at byte {x['early_stop']}")
    print("\n  (b) OBSERVATION CODES, and whether the feature bands (L1, G1, E1, B1) are present")  # heading
    for fam_, st_ in (("sbf", "SBF_02"), ("ubx", "UBX_01")):  # one example stream per family
        r = ver[ver.stream == st_].iloc[0]  # its first row (2.6.5 in capture order)
        print(f"      {st_} in {r.test_code}:")  # which row
        for c_ in CONSTELLATIONS:  # each constellation with codes
            if str(r[f"obs_codes_{c_}"]).strip() and str(r[f"obs_codes_{c_}"]) != "nan":  # if: declared
                print(f"          {c_}  {r[f'obs_codes_{c_}']}")  # its codes
    # prints whether every header declares the four observables of every feature band (header test)
    print(f"      Feature-band codes declared in every header: {bool(ver.required_codes_in_header.all())}")
    for fam_, req_ in REQUIRED_BANDS.items():  # each family
        fv_ = ver[ver.stream.str.startswith(fam_.upper())]  # its rows
        for con, (code, label) in req_.items():  # each band
            pc_ = fv_[f"band_{con}_present_pct"]  # share of epochs with the band present, per stream
            # one line per family and band: the data test across streams
            print(f"          {fam_.upper()} {label:<16} {code}: present in {pc_.min():.1f} to {pc_.max():.1f} % of "
                  f"RINEX epochs across {len(fv_)} streams")  # f-string continued: the number of streams the range covers

    print("\n  (c) MEASEPOCH CROSS-CHECK OVER EVERY EPOCH - the check that a revision mismatch would fail")  # heading
    if cross.empty:  # if: no cross-check rows
        flag("no MeasEpoch cross-check rows were produced")  # records an anomaly
    else:  # else: there are cross-check rows to summarise
        n_ep = int(xsk.epochs_compared.sum())  # epochs compared over all Septentrio streams
        # the scope: epochs, signals, sessions, streams
        print(f"      {n_ep} epochs, {int(cross.n_signals.sum())} signals (Type1 and Type2) compared across "
              f"{cross.test_code.nunique()} sessions and {cross.stream.nunique()} streams.")  # the scope
        # for loop over the four observables with their units
        for kind, unit, nm in (("C", "m", "pseudorange"), ("L", "cyc", "carrier phase"), ("D", "hz", "Doppler"),
                               ("S", "dbhz", "C/N0")):  # each observable
            col = f"max_abs_d{kind}_{unit}"  # its largest-difference column
            # one line per observable: values compared and the largest difference
            print(f"      {nm:<14}: {int(cross[f'n_{kind}'].sum()):>10} values, max |RINEX - binary| "
                  f"{cross[col].max():.6f} {unit}")  # one line per observable
        # the tolerance counts on one line
        print(f"      pseudorange beyond 1 mm: {int(cross.n_C_over_1mm.sum())}; carrier phase half-cycle "
              f"differences: {int(cross.n_L_half_cycle.sum())}, other beyond 0.002 cycles: "  # f-string continued: phase differences of half a cycle, and other phase differences
              f"{int(cross['n_L_over_0.002'].sum())}; Doppler beyond 0.002 Hz: {int(cross['n_D_over_0.002'].sum())}; "  # f-string continued: phase and Doppler differences above their tolerances
              f"C/N0 off the 1/32 dB-Hz grid: {int(cross.n_S_off_32nd_grid.sum())}")  # tolerance counts
        sk_ = xsk.drop(columns=["test_code", "stream", "epochs_compared"]).sum()  # skips summed over streams
        # every skip reason with its count
        print("      Not compared, by reason (all streams): " + "; ".join(f"{k[6:]} {int(v)}" for k, v in sk_.items()
                                                                       if v and not k.startswith("skip: info:")))  # generator clause: skips reasons with no count and purely informational ones
        # the revision counts, printed apart because they are information, not skips
        print("      MeasEpoch blocks by revision: " + "; ".join(f"{k[12:]} {int(v)}" for k, v in sk_.items()
                                                             if v and k.startswith("skip: info:")))  # info only
        pr = pd.Series(np.repeat([k / 1e4 for k in pr_hist], list(pr_hist.values())))  # pseudorange diffs (0.1 mm)
        cn = pd.Series(np.repeat([k / 32 for k in cn_steps], list(cn_steps.values())))  # C/N0 diffs on the grid
        if pr.abs().max() < 1e-3:  # if: below 1 mm
            print("      The pseudorange agrees to the last decimal the format carries.")  # fixed text
        # sorted unique |C/N0 differences|, rounded to 5 decimals
        uniq = np.sort(cn.abs().round(5).unique())
        # smallest gap between them (the step); 0 with a single value
        step = np.min(np.diff(uniq)) if len(uniq) > 1 else 0.0
        if cn.abs().max() < 0.25:  # if: below 0.25 dB-Hz
            print("      The C/N0 differences take only discrete values, in steps of")  # fixed text
            # prints the step, then the fraction fr_ computed one line above, so "That is 1/32 dB-Hz." is not typed
            fr_ = Fraction(step).limit_denominator(64)  # the step as a fraction (computed, not typed)
            # the step and the range, with the step as a fraction
            print(f"      {step:.5f} dB-Hz, spanning 0 to {cn.abs().max():.5f}. That is {fr_} dB-Hz.")
            # fixed text; true per SBF ref p.254 (0.25 dB-Hz) and p.258 (MeasExtra 0.03125)
            print("      The MeasEpoch C/N0 field has 0.25 dB-Hz resolution and the SBF guide")
            print("      states that the Misc field of the MeasExtra block extends it to")  # continued
            print("      0.03125 dB-Hz. The converter is applying that refinement and this")  # continued
            print("      decoder is not, so the difference is the refinement itself rather")  # continued
            print("      than any disagreement about the measurement.")  # continued
            # note; "up to 7/32" typed; MeasExtra CN0HighRes is bits 0-2, 0-7 x 0.03125 dB-Hz (SBF ref p.258)
            note("The carrier-to-noise density differs between the RINEX and this "
                 f"decoder only in discrete steps of {fr_} dB-Hz, up to "  # f-string continued: the step size of the C/N0 differences
                 f"{Fraction(float(cn.abs().max())).limit_denominator(64)}. That is the "  # f-string continued: the largest C/N0 difference written as a fraction
                 "MeasExtra resolution extension described in the SBF reference guide, "
                 "which refines the 0.25 dB-Hz MeasEpoch field to 0.03125 dB-Hz. The "
                 "converter applies it and the cross-check decoder does not, so the "
                 "difference is the refinement and not a disagreement.")
        # note (f-string) with the computed maxima
        note(f"The converted pseudorange was compared signal by signal against "
             # the scope: every epoch and every signal of the Septentrio streams
             f"MeasEpoch decoded independently from the binary, over every epoch ({n_ep}) and every signal "
             f"({int(cross.n_signals.sum())}) of the {cross.stream.nunique() * cross.test_code.nunique()} "  # f-string continued: the number of signals and of Septentrio stream-sessions compared
             f"Septentrio stream-sessions. The maximum absolute difference is "
             f"{pr.abs().max():.6f} m. C/N0 agrees to within "  # max |PR difference|, 6 decimals
             # max |C/N0 difference|, 3 decimals
             f"{cn.abs().max():.3f} dB-Hz, below the 0.25 dB-Hz quantisation step of the "
             "source field. A block-revision mismatch would have produced a systematic "
             "offset here and did not.")
        if pr.abs().max() > 1.0:  # if: pseudorange off by more than 1 m
            die(f"the converted pseudorange differs from the binary by up to "  # die(): stops the run
                f"{pr.abs().max():.3f} m; the converter is not reproducing the "  # max |difference|, 3 decimals
                "measurements and nothing downstream can be trusted")

    print("\n  (d) OBSERVABLE RANGES ON EVERY EPOCH, against physics")  # heading
    print(f"      Ephemeris bounds for {len(eph)} satellites ("  # how many satellites have bounds, by constellation
          # string concatenation: satellites with bounds per constellation, then the label for rejects
          + ", ".join(f"{c_} {sum(1 for k in eph if k[0] == c_)}" for c_ in CONSTELLATIONS) + "); rejected records: "
          + (", ".join(f"{k} {v}" for k, v in sorted(eph_rej.items())) or "none") + ".")  # string concatenation: rejected records by reason, or none
    # where the bounds come from, unhealthy IGS records inside them, and the receivers' off-date records
    print(f"      Source: {IGS_BRDC.name} only (Galileo I/NAV; date filter; no health filter); unhealthy IGS records "
          f"entering the bounds: {dict(sorted(igs_unh.items())) or 'none'}.")  # f-string continued: unhealthy records that entered the bounds, or none
    # prints the receivers' rejected records, as before 
    print("      Receivers' navigation files (superseded excluded, not used for the bounds), rejected records: "
          + (", ".join(f"{k} {v}" for k, v in sorted(rx_rej.items())) or "none") + ".")  # the evidence
    tot_ab = Counter()  # satellite -> pseudoranges over all streams
    for (c_, s_, v_), n_ in absent_cnt.items():  # each (test, stream, satellite)
        tot_ab[v_] += n_  # summed
    for v_ in sorted(tot_ab):  # each satellite absent from IGS
        n_used = sum(1 for r_ in fb_rows if r_["sv"] == v_ and r_["used"])  # its trusted fallback records
        # prints its name, pseudorange count, streams, and whether the fallback gave it bounds
        print(f"      Absent from IGS: {v_}, {tot_ab[v_]} pseudoranges ("
              # string concatenation: the pseudorange counts of this satellite per session and stream
              + ", ".join(f"{c_} {s_} {n_}" for (c_, s_, x_), n_ in sorted(absent_cnt.items()) if x_ == v_) + "); "
              + (f"bounds from {n_used} trusted receiver record(s)" if v_ in eph else  # string concatenation: whether trusted receiver records gave it bounds, or it was not judged
                 "not judged: no trusted ephemeris (counted in n_not_judged_no_ephemeris)") + ".")
    # the named allowances, printed from their constants
    print(f"      Allowances: receiver clock {CLOCK_ALLOWANCE_S * 1e3:g} ms, atmosphere {ATMOSPHERE_ALLOWANCE_M:g} m, "
          f"elevation floor {ELEVATION_FLOOR_DEG:g} deg, GLONASS/SBAS radius slack {ORBIT_RADIUS_SLACK_KM:g} km, "  # f-string continued: elevation floor and orbit radius slack
          f"drift {DRIFT_ALLOWANCE_PPM:g} ppm; C/N0 window {SNR_RANGE_DBHZ[0]:g}-{SNR_RANGE_DBHZ[1]:g} dB-Hz.")  # f-string continued: oscillator drift and the accepted C/N0 window
    # rows outside every scheduled window of our own spoofing (the 2.6.12 u-blox before the attack included)
    out_ = phys[phys.period.str.startswith("outside") | (phys.period == "before the attack window")]
    ins_ = phys[phys.period == "inside the scheduled window"]  # rows inside a window (not the 2.6.12 u-blox)
    att_ = phys[phys.period.str.contains("attack effect")]  # the 2.6.12 u-blox attack rows
    aft_ = phys[phys.period.str.startswith("after the attack window")]  # the 2.6.12 u-blox after the clock release
    # for loop over the four groups: outside the windows, inside them, the attack rows, after the release
    for lab_, tb_ in (("Outside the scheduled windows", out_), ("Inside the scheduled windows", ins_),
                      ("2.6.12 u-blox attack window (attack effect)", att_),  # tuple element: the u-blox rows of the 2.6.12 attack window
                      # tuple element: the u-blox rows after its clock release, whose cause is not established
                      ("2.6.12 u-blox after the clock release (attack after-effect, cause not established)", aft_)):
        # the group's totals
        print(f"\n      {lab_}: {int(tb_.n_values.sum())} values, {int(tb_.n_not_judged_no_ephemeris.sum())} not "
              f"judged (no ephemeris), {int(tb_.n_outside.sum())} outside")  # the totals
        bad_ = tb_[tb_.n_outside > 0].groupby(["constellation", "obs_code"]).agg(  # outside counts per code
            # keyword arguments of agg(): values outside, all values, and the number of streams per code
            n_outside=("n_outside", "sum"), n_values=("n_values", "sum"), streams=("stream", lambda s: len(set(s))))
        for (c_, o_), r_ in bad_.iterrows():  # one line per constellation and code with values outside
            # one line per constellation and code
            print(f"          {c_} {o_:<4} {int(r_.n_outside):>7} of {int(r_.n_values):>9} outside, "
                  f"in {int(r_.streams)} stream(s)")  # f-string continued: the number of streams with values outside

    # (d2): the common receiver-clock term - verification only, never a feature
    ck_ = pd.read_csv(PROCESSED / "preprocessing" / "rinex_clock_common_term.csv")  # the summary just written
    print("\n  (d2) A RECEIVER-CLOCK TERM COMMON TO ALL SATELLITES (GPS L1 C/A and Galileo E1, per epoch)")  # heading
    # the method in one line
    print(f"      median of pseudorange minus geometric range; elevation mask {CLK_ELEV_MASK_DEG:g} deg, troposphere "
          f"{TROPO_ZENITH_M:g} m / sin(el), ionosphere not modelled, site height {RX_HEIGHT_M:g} m")  # the method
    for fam_ in ("SBF", "UBX"):  # per family: the range of c*dt and of the residual spread
        f_ = ck_[ck_.stream.str.startswith(fam_)]  # its rows
        # the family's c*dt range and residual spread
        print(f"      {fam_}: c*dt {f_.cdt_km_min.min():.1f} to {f_.cdt_km_max.max():.1f} km; residual spread (median "
              f"per period) {f_.residual_mad_m_median.min():.1f} to {f_.residual_mad_m_median.max():.1f} m")  # range
    for t_ in clk_tests:  # the u-blox 01's clock release test per 2.6.12 u-blox stream
        # the u-blox 01's clock release decision for this stream
        print(f"      ubx_release {t_['test_code']} {t_['stream']}: residual spread {t_['after_release_median_mad_m']} m after the "
              f"release against {t_['pre_test_median_mad_m']} m before the test -> common to all satellites: "  # f-string continued: the spread before the test, and the verdict on a common clock term
              f"{t_['common_to_all_satellites']} (criterion: at most {t_['ratio_max']:g} times)")  # the decision
        # records the result as a note
        note(f"u-blox 01's clock release test ({t_['stream']}, {t_['test_code']}): after removing a per-epoch clock term common to all "
             f"satellites, the residual spread after the clock release is {t_['after_release_median_mad_m']} m against "  # f-string continued: the residual spread after the clock release
             f"{t_['pre_test_median_mad_m']} m in the pre-test margin, so the post-release pseudoranges are "  # f-string continued: the spread before the test, the baseline it is compared with
             + ("shifted together by the receiver clock" if t_["common_to_all_satellites"] else  # string concatenation: the verdict, a shift shared by all satellites or not
                "not explained by a common clock term") + " (preprocessing/rinex_clock_common_term.csv, preprocessing/rinex_clock_2612_ubx01.csv).")
    pa_ = phys[phys.obs_code.str[0] == "C"]  # pseudorange rows
    # the physics comparison with and without the clock term
    print(f"      pseudoranges outside the bounds: {int(pa_.n_outside.sum())} with the 1 ms allowance; "
          # f-string continued: values outside once the clock term is removed, out of those judged
          f"{int(pa_.n_outside_after_clock.sum())} of {int(pa_.n_judged_after_clock.sum())} after removing the clock "
          f"term ({int(pa_.n_no_clock_estimate.sum())} without an estimate)")  # the comparison

    print("\n  (e) GAPS, and whether they correspond to anything already known")  # fixed text
    for _, r in ver.iterrows():  # for loop over streams
        if r.gaps_gt_1s:  # if: any gap over 1 s
            print(f"      {r.test_code:<8}{r.stream:<8}{int(r.gaps_gt_1s):>4} gaps over 1 s, "  # prints it
                  f"largest {r.largest_gap_s:>6.1f} s, {r.lost_s:>7.1f} s lost")  # continued
    gapless = ver[ver.gaps_gt_1s == 0]  # streams with no gap
    print(f"      {len(gapless)} of {len(ver)} converted streams have no gap over one second.")  # prints the count

    # ---- the mosaic-G5 gap question ---------------------------
    print("\n  (f) THE MOSAIC-G5 GAPS: logging bottleneck, or something in the radio environment?")  # heading
    print("      If the interruptions were caused by the signal environment they should coincide with")  # method
    print("      interruptions on the other streams, which record the same signal environment. Gaps are")  # continued
    print("      measured in time: receiver time for the Septentrio streams, host time for the u-blox.")  # continued

    # nested function: (start, end) of every spacing above the gap threshold, max(1 s, 1.5 x the median spacing)
    def gap_intervals(times):
        t = np.array(sorted(x.timestamp() for x in times))  # POSIX seconds, sorted
        if len(t) < 3:  # if: too few epochs
            return [], None  # no gaps, no spacing
        d = np.diff(t)  # spacings
        thr = max(1.0, 1.5 * float(np.median(d)))  # the threshold, as in epoch_gaps
        k = np.nonzero(d > thr)[0]  # indices of gaps
        return [(t[i], t[i + 1]) for i in k], float(np.median(d))  # the intervals and the median spacing

    g5rows, deprows = [], []  # one row per session; gap rate by satellite count
    for _, sr in sessions.iterrows():  # for loop over sessions
        folder = DAY / sr["session_folder"]  # raw folder
        src5 = next(iter(folder.glob("*SBF_01.sbf")), None)  # the G5 file (SBF_01), or None
        if src5 is None:  # if: no G5 file in this session
            continue  # continue: next session
        ep5 = sorted(count_binary_epochs(src5, "sbf")["sbf_epochs"])  # (time, N1, bytes) per MeasEpoch, by time
        gi5, sp5 = gap_intervals([t_ for t_, _, _ in ep5])  # the G5 gaps and its median spacing
        t5 = np.array([t_.timestamp() for t_, _, _ in ep5])  # its epoch times, s
        span5 = t5[-1] - t5[0]  # first to last, s
        # the other streams' gaps in this session, on their true clocks
        oth = {st_: gap_intervals(tt_)[0] for (c_, st_), tt_ in true_times.items() if c_ == sr["test_code"]}
        cap_ = (pd.Timestamp(sr["capture_end_utc"]) - pd.Timestamp(sr["capture_start_utc"])).total_seconds()  # s
        # per other stream: G5 gaps overlapping one of its gaps, and the number chance alone would give (the G5 gap
        # count times the share of the capture that stream spends inside its own gaps) - 3
        coinc = {s_: sum(any(a2 < b and b2 > a for a2, b2 in gl) for a, b in gi5) for s_, gl in oth.items()}
        chance = {s_: len(gi5) * sum(b - a for a, b in gl) / cap_ for s_, gl in oth.items()}  # expected by chance
        ss_ = pd.Timestamp(sr["scheduled_start_utc"], tz="UTC").timestamp()  # scheduled start, s
        se_ = pd.Timestamp(sr["scheduled_end_utc"], tz="UTC").timestamp()  # scheduled end, s
        inw = (t5 >= ss_) & (t5 < se_)  # boolean array: epochs inside our scheduled window
        n1 = np.array([n_ for _, n_, _ in ep5]); by = np.array([b_ for _, _, b_ in ep5])  # satellites, block bytes
        gap_after = np.r_[np.diff(t5) > max(1.0, 1.5 * sp5), False]  # boolean: this epoch is followed by a gap
        g5rows.append({"test_code": sr["test_code"], "g5_epochs": len(ep5),  # identity and epoch count
                       # dict entry: the G5's nominal rate and the share of nominal epochs it delivered
                       "g5_nominal_hz": round(1 / sp5, 1), "g5_delivered_pct": round(100 * len(ep5) / (span5 / sp5 + 1), 1),
                       "g5_gaps": len(gi5), "g5_median_gap_s": round(float(np.median([b - a for a, b in gi5])), 3)  # dict entry: number of gaps and their median length in seconds
                       if gi5 else None, "g5_lost_s": round(sum(b - a for a, b in gi5), 1),  # gap count, length, total
                       # the comparison, split by family, with what chance alone would give
                       "g5_gaps_coinciding_septentrio_gap": sum(v for s_, v in coinc.items() if s_.startswith("SBF")),
                       "g5_gaps_coinciding_ublox_gap": sum(v for s_, v in coinc.items() if s_.startswith("UBX")),  # dict entry: G5 gaps that coincide with a u-blox gap
                       "chance_expected_ublox": round(sum(v for s_, v in chance.items() if s_.startswith("UBX")), 1),  # dict entry: the number of u-blox coincidences chance alone would give
                       "chance_expected_septentrio": round(sum(v for s_, v in chance.items() if s_.startswith("SBF")), 1),  # dict entry: the number of Septentrio coincidences chance alone would give
                       "other_streams_total_gaps": sum(len(v) for v in oth.values()),  # their gaps on true clocks
                       "other_streams_total_lost_s": round(sum(b - a for v in oth.values() for a, b in v), 1),  # the time other streams lost
                       "other_streams_compared": " ".join(sorted(oth)),  # which streams
                       # where the gaps fall against where the time is, and what the blocks carried
                       "share_of_g5_gaps_in_window": round(float(gap_after[inw].sum() / max(gap_after.sum(), 1)), 3),
                       "share_of_g5_epochs_in_window": round(float(inw.mean()), 3),  # dict entry: share of all G5 epochs that fall inside the test window
                       "median_block_bytes_in_window": float(np.median(by[inw])) if inw.any() else None,  # dict entry: median block size in bytes inside the test window
                       "median_block_bytes_outside": float(np.median(by[~inw])) if (~inw).any() else None,  # dict entry: median block size in bytes outside the test window
                       "median_satellites_in_window": float(np.median(n1[inw])) if inw.any() else None,  # dict entry: median satellite count inside the test window
                       "median_satellites_outside": float(np.median(n1[~inw])) if (~inw).any() else None})  # dict entry: median satellite count outside the test window
        for lo_ in range(0, int(n1.max()) + 1, 10):  # satellite-count bins of ten (the leakage risk)
            m_ = (n1 >= lo_) & (n1 < lo_ + 10)  # boolean: epochs in the bin
            if m_.any():  # if: any epoch
                # appends the bin: epochs, epochs followed by a gap, and the rate
                deprows.append({"test_code": sr["test_code"], "satellites_from": lo_, "satellites_to": lo_ + 9,
                                "epochs": int(m_.sum()), "followed_by_gap": int(gap_after[m_].sum()),  # counts
                                "gap_rate": round(float(gap_after[m_].mean()), 4)})  # the rate in the bin
    g5 = pd.DataFrame(g5rows)  # list of dicts -> DataFrame, one row per session
    g5.to_csv(PROCESSED / "preprocessing" / "mosaic_g5_gap_alignment.csv", index=False, lineterminator="\n")  # written
    pd.DataFrame(deprows).to_csv(PROCESSED / "preprocessing" / "mosaic_g5_gap_dependence.csv", index=False, lineterminator="\n")  # new
    for r_ in g5.itertuples():  # one line per session
        # one measured line per session: rate, delivered share, gaps, coincidence
        print(f"      {r_.test_code:<7} G5 {r_.g5_nominal_hz:g} Hz, delivered {r_.g5_delivered_pct:5.1f} %, "
              f"{r_.g5_gaps:>4} gaps (median {r_.g5_median_gap_s} s, {r_.g5_lost_s:7.1f} s lost); coinciding with "  # f-string continued: gap count, median gap and seconds lost, then the coincidences
              f"a Septentrio gap {r_.g5_gaps_coinciding_septentrio_gap} (chance {r_.chance_expected_septentrio:g}), "  # f-string continued: coincidences with a Septentrio gap and the chance figure
              f"a u-blox gap {r_.g5_gaps_coinciding_ublox_gap} (chance {r_.chance_expected_ublox:g})")  # measured
    for r_ in g5.itertuples():  # the dependence on what the attack drives
        # one line per session: gaps and epochs inside the window, block sizes
        print(f"      {r_.test_code:<7} gaps in window {100 * r_.share_of_g5_gaps_in_window:5.1f} % against "
              f"{100 * r_.share_of_g5_epochs_in_window:5.1f} % of epochs; median block "  # f-string continued: the share of epochs inside the window, and the median block size
              f"{r_.median_block_bytes_in_window} B in window, {r_.median_block_bytes_outside} B outside")  # f-string continued: the median block sizes inside and outside the window
    tot_s = int(g5.g5_gaps_coinciding_septentrio_gap.sum())  # G5 gaps coinciding with an X5 or mosaic-H gap
    tot_u, ch_u = int(g5.g5_gaps_coinciding_ublox_gap.sum()), float(g5.chance_expected_ublox.sum())  # u-blox, chance
    tot_g = int(g5.g5_gaps.sum())  # all G5 gaps
    # the answer, computed: coincidences by family and the chance expectation
    print(f"      Of {tot_g} G5 gaps, {tot_s} coincide with a Septentrio gap and {tot_u} with a u-blox gap "
          f"(chance alone: about {ch_u:.0f}).")  # the answer, computed
    if tot_s == 0 and tot_u <= ch_u * 1.5:  # if: none with the Septentrio, and the u-blox ones near chance
        # records the finding, with the leakage caution
        note(f"The mosaic-G5 interruptions are specific to that stream: of {tot_g} gaps, none coincides with a gap "
             f"of the mosaic-X5 or mosaic-H (receiver time), and {tot_u} overlap a u-blox gap (host time), about "  # f-string continued: the coincidences with u-blox gaps on host time
             f"what chance gives ({ch_u:.0f}) because the u-blox, chiefly UBX_02, spend long stretches in their own "  # f-string continued: the chance figure and why the u-blox gaps overlap so often
             "gaps. The G5 gaps depend on the size of the MeasEpoch block, which our own attacks drive "
             "(satellites lost or gained), so its missing data are correlated with the label and its gaps must "
             "never be read as a property of the signal environment (preprocessing/mosaic_g5_gap_dependence.csv).")
    else:  # else: the comparison does not support the attribution - reported as an anomaly
        # records the anomaly with the counts
        flag(f"mosaic-G5 gaps: {tot_s} coincide with a Septentrio gap, {tot_u} with a u-blox gap (chance {ch_u:.0f})")

    # ---- what RINEX does not carry -------------------------------------
    print("\n" + "=" * 78)  # banner
    print("[6] WHAT THE CONVERSION DOES NOT CARRY")  # fixed text
    print("=" * 78)  # banner
    # list of (family, quantity) pairs that RINEX does not carry
    lost = [("Septentrio", "ReceiverStatus AGCState gain, sample variance, blanking"),
            ("Septentrio", "RFStatus band list, and its non-authentic and NMA flags"),  # RFStatus
            ("Septentrio", "QualityInd quality indicators"),  # QualityInd
            ("u-blox", "MON-RF agc_cnt, cw_suppression, noise_per_ms, jamming_state"),  # u-blox MON-RF fields
            ("u-blox", "NAV-PVT receiver-reported time, which the receiver-status script used to watch "  # u-blox reported time...
                       "the clock being captured")]
    for fam, what in lost:  # for loop
        print(f"      {fam:<12} {what}")  # prints each pair
    print("\n      None of these survives into RINEX, which is the point. They are")  # fixed text
    print("      available for LABELLING and are barred from the FEATURE SET by the")  # continued
    print("      exclusion rule. The separation is now visible in the file layout:")  # continued
    print("          Data/rinex/      observation and navigation only  -> features")  # continued
    print("          Data/processed/  receiver status and ground truth -> labelling only")  # continued
    # writes README_EXCLUSION.txt into Data/rinex with fixed text: the same bytes on every run
    (RINEX / "README_EXCLUSION.txt").write_text(
        "THE EXCLUSION RULE, AT THE BOUNDARY IT GOVERNS\n"
        "==============================================\n\n"
        "This directory holds RINEX observation and navigation files only. Every\n"
        "quantity here is one an ordinary commercial receiver emits in a standard\n"
        "interchange format: pseudorange, carrier phase, Doppler and carrier-to-noise\n"
        "density, with broadcast ephemeris alongside.\n\n"
        "FEATURES MAY BE COMPUTED FROM THIS DIRECTORY AND FROM NOWHERE ELSE.\n\n"
        "The receivers also report their own interference and authentication status:\n"
        "automatic gain control, the Septentrio RFStatus band list and its\n"
        "non-authentic flag, the quality indicators, the u-blox jamming state and\n"
        "noise level. None of it survives conversion to RINEX, and that is not an\n"
        "accident of the format but the reason the format was chosen. Those\n"
        "quantities live in Data/processed/ and are used to establish ground truth\n"
        "and to place label boundaries.\n\n"
        "USING THEM AS FEATURES WOULD MEAN THE DETECTOR WAS PARTLY READING THE\n"
        "MANUFACTURER'S OWN INTERFERENCE VERDICT. The research plan (Research_Plan_Topu.pdf, section 2.4) forbids it, and\n"
        "the directory split is how that prohibition is enforced in practice rather\n"
        "than in intention.\n\n"
        "Conversion: RTKLIB convbin, demo5 fork, commit recorded in\n"
        "tools/convbin_build.txt, with the exact command for every file in\n"
        "Data/processed/rinex_conversion_log.csv.\n",
        encoding="utf-8")  # encoding argument
    print("      Written to Data/rinex/README_EXCLUSION.txt.")  # fixed text

    # ---- inventory and compression -------------------------------------
    print("\n" + "=" * 78)  # banner
    print("[7] INVENTORY AND COMPRESSION")  # fixed text
    print("=" * 78)  # banner
    # The uncompressed size is taken from the conversion log, which records it at
    # the moment of conversion. Reading it from disk at this point would return
    # zero on a run again, because the file has since been compressed and removed.
    logp = PROCESSED / "preprocessing" / "rinex_conversion_log.csv"  # path of the conversion log
    rawsize = {}  # dict: (test, stream) -> uncompressed OBS size
    if logp.exists() and logp.stat().st_size > 0:  # if: the log exists and is not empty
        try:  # try: if reading the log fails, except runs
            prev = pd.read_csv(logp)  # reads it
        except Exception as e:  # except: an unreadable log would set every inventory size to 0
            die(f"the conversion log {logp.name} cannot be read ({e})")  # stops
        for _, q in prev.iterrows():  # for loop over the log rows
            # stores obs_bytes under (test, stream); q.get(..., 0) or 0 guards a missing value
            rawsize[(q["test_code"], q["stream"])] = int(q.get("obs_bytes", 0) or 0)
    inv = []  # empty list of inventory rows
    comp_mb = {}  # empty dict: (test, stream) -> compressed MB, printed below but never written
    for _, r in ver.iterrows():  # for loop over verification rows
        sr = sessions[sessions.test_code == r.test_code].iloc[0]  # the session's row
        outdir = RINEX / sr["session_folder"]  # RINEX folder
        # the raw file again, or None (the "if src" tests below guard against None)
        src = next(iter((DAY / sr["session_folder"]).glob(f"*{r.stream}.*")), None)
        obs = outdir / f"{src.stem}.obs" if src else None  # OBS path (absent on a run again)
        nav = outdir / f"{src.stem}.nav" if src else None  # NAV path
        raw = rawsize.get((r.test_code, r.stream), 0)  # uncompressed size from the log
        if (r.test_code, r.stream) in PIECEWISE:  # if: converted in pieces - sizes summed over the pieces
            # generator inside sum(): the pieces' uncompressed sizes from the log
            raw = sum(v for (c_, s_), v in rawsize.items() if c_ == r.test_code and s_.startswith(f"{r.stream}_part"))
        if not raw and obs and obs.exists():  # if: the log has none and an OBS file is on disk
            raw = obs.stat().st_size  # its size
        comp = 0  # compressed size, 0 so far
        # if: an OBS file exists and no .crx*: compress now (never on a run again)
        if obs and obs.exists() and not any(outdir.glob(f"{src.stem}.crx*")):
            try:  # try: a compression failure jumps to except
                import hatanaka  # import inside the function
                # hatanaka keys off the file name. A ".obs" suffix is not a name it
                # recognises, so it would gzip without Hatanaka-compressing and keep
                # the wrong extension. Renaming to the long RINEX convention first
                # gets both the compression and a name later tools will recognise.
                # the .rnx name; the old comment's claim about how hatanaka reads file names is not verified in docs/
                rnx = obs.with_suffix(".rnx")
                obs.rename(rnx)  # renames .obs to .rnx on disk
                # compresses (Hatanaka, then gzip: .crx.gz here) next to it and deletes the .rnx (delete=True)
                out = hatanaka.compress_on_disk(rnx, delete=True)
                comp = Path(out).stat().st_size  # compressed size
            except Exception as e:  # except Exception as e: e holds the error text
                # die() instead of flag(): a failed compression would leave a stream without its .crx.gz, so the run stops
                die(f"Hatanaka compression failed for {obs.name}: {e}")
        else:  # else: already compressed (every stream on a run again)
            for c in outdir.glob(f"{src.stem}.crx*"):  # for loop over the .crx* files (one here)
                comp = c.stat().st_size  # its size
        if (r.test_code, r.stream) in PIECEWISE:  # if: converted in pieces - compressed and NAV sizes summed
            comp = sum(c.stat().st_size for c in outdir.glob(f"{src.stem}_part*.crx.gz"))  # the pieces' sizes
            # their NAV sizes; the superseded file excluded explicitly (the pattern already misses it)
            nav_b = sum(c.stat().st_size for c in outdir.glob(f"{src.stem}_part*.nav") if "_superseded" not in c.name)
        else:  # else: one file
            nav_b = nav.stat().st_size if nav and nav.exists() else 0  # its NAV size
        # list comprehension (or one-item list): the stream's compressed files in rinex_files order, part01 first (sorted by name)
        crx_files = sorted(outdir.glob(f"{src.stem}_part*.crx.gz")) if (r.test_code, r.stream) in PIECEWISE else [outdir / f"{src.stem}.crx.gz"]
        comp_mb[(r.test_code, r.stream)] = comp / 1e6  # dict entry: the compressed size for the console summary only (not in the inventory)
        # appends one inventory row (dict literal):
        inv.append({
            "test_code": r.test_code, "session_folder": sr["session_folder"],  # identity fields
            "stream": r.stream, "receiver": r.receiver,  # stream and receiver
            # capture length in minutes: end minus start
            "duration_min": round((pd.Timestamp(sr["capture_end_utc"])
                                   - pd.Timestamp(sr["capture_start_utc"])  # minus the capture start
                                   ).total_seconds() / 60, 2),  # seconds / 60 = minutes, 2 decimals
            "epochs": r.rinex_epochs, "nominal_hz": r.nominal_hz,  # RINEX epochs and nominal rate
            # completeness on the time base named in completeness_time_base
            "completeness_pct": r.completeness_pct,
            "mean_satellites": r.mean_sat_per_epoch,  # mean satellites
            "min_satellites": r.min_sat_per_epoch,  # fewest
            "max_satellites": r.max_sat_per_epoch,  # most
            "constellations": r.constellations,  # constellation letters
            **{f"obs_codes_{c_}": r[f"obs_codes_{c_}"] for c_ in CONSTELLATIONS},  # code columns
            "required_codes_in_header": r.required_codes_in_header,  # the header test
            # the data test: share of epochs with each feature band present
            **{f"band_{c_}_present_pct": r[f"band_{c_}_present_pct"] for c_ in "GREC"},
            "completeness_time_base": r.completeness_time_base,  # receiver (Septentrio) or host (u-blox)
            "gaps_gt_1s": r.gaps_gt_1s, "largest_gap_s": r.largest_gap_s,  # gap count and largest gap
            "obs_bytes_uncompressed": raw,  # uncompressed size (the compressed size and the ratio left the inventory; they follow the date line)
            "obs_sha256_content": content_sha256(crx_files),  # SHA-256 of the decoded observation text without its date lines
            "nav_bytes": nav_b,  # NAV size (summed over the pieces), 0 if missing
            # dict entry: the observation file names, a numbered range when the session was split into pieces
            "rinex_files": (f"{src.stem}_part01..{len(list(outdir.glob(f'{src.stem}_part*.crx.gz'))):02d}.crx.gz"
                            if (r.test_code, r.stream) in PIECEWISE else f"{src.stem}.crx.gz"),  # which files
            "excluded": r.excluded})  # the exclusion, if any
    idf = pd.DataFrame(inv)  # list of dicts -> DataFrame, one row per stream-session
    idf.to_csv(PROCESSED / "preprocessing" / "rinex_inventory.csv", index=False, lineterminator="\n")  # writes preprocessing/rinex_inventory.csv
    print(f"      {'test':<8}{'stream':<8}{'epochs':>8}{'Hz':>6}{'sats':>7}"  # header row
          f"{'bands':>8}{'raw MB':>9}{'crx MB':>9}{'ratio':>7}")  # continued
    for _, r in idf.iterrows():  # for loop
        # one line per stream
        print(f"      {r.test_code:<8}{r.stream:<8}{r.epochs:>8}{(r.nominal_hz or 0):>6.0f}"
                    # the stream's own constellation letters
              f"{(r.mean_satellites or 0):>7.1f}{r.constellations:>8}"
              # the compressed size from comp_mb (dict lookup by the row's test and stream) and the ratio raw / compressed
              f"{r.obs_bytes_uncompressed/1e6:>9.1f}{comp_mb[(r.test_code, r.stream)]:>9.1f}"
              # conditional expression: the ratio, 0 when no compressed size was found
              f"{(r.obs_bytes_uncompressed / 1e6 / comp_mb[(r.test_code, r.stream)] if comp_mb[(r.test_code, r.stream)] else 0):>7.1f}")
    tot_raw = idf.obs_bytes_uncompressed.sum() / 1e9  # total uncompressed GB
    tot_cmp = sum(comp_mb.values()) / 1e3  # sum() over the dict's values: total compressed GB (MB / 1000)
    print(f"\n      Total {tot_raw:.2f} GB uncompressed, {tot_cmp:.2f} GB after Hatanaka "  # prints them
          f"compression.")  # continued
    print(f"      Free space now {free_gb(DATA):.1f} GB.")  # prints free space

    # ---- 41g: the toolchain record ----------------------------------------------------------------------------
    import gzip, importlib.metadata, platform  # gzip reads the CRINEX headers; metadata gives package versions
    crx_prog = set()  # the Hatanaka program lines found in the compressed observation files
    for c in sorted(RINEX.glob("*/*.crx.gz")):  # every compressed file, superseded included
        with gzip.open(c, "rt", errors="replace") as fh:  # opened as text through gzip
            for _ in range(3):  # the first lines hold the CRINEX header
                ln_ = fh.readline()  # one line
                if "CRINEX PROG / DATE" in ln_:  # if: the program line
                    crx_prog.add(ln_[:20].strip())  # its program name and version (columns 1-20)
    tool = [  # list of dicts: one row per tool, with where the value comes from
        {"tool": "convbin", "version": conv.get("version_string", ""), "detail": f"commit {conv.get('commit', '')}",  # list element: the converter's version and commit
         "source": "tools/convbin_build.txt (sha256 of tools/convbin checked against it on every run)"},  # dict entry: where the converter's version and commit are recorded and checked
        {"tool": "RINEX format", "version": RINEX_VERSION, "detail": "convbin -v", "source": "04 RINEX_VERSION"},  # list element: the RINEX format version the converter writes
        {"tool": "Hatanaka compression (CRINEX header)", "version": "; ".join(sorted(crx_prog)),  # list element: the Hatanaka compression program named in the file headers
         "detail": "program line of every .crx.gz", "source": "Data/rinex/*/*.crx.gz headers"},  # dict entry: where that program name is read from
        {"tool": "hatanaka (Python package)", "version": importlib.metadata.version("hatanaka"),  # list element: the version of the Python Hatanaka package
         "detail": "compress_on_disk / decompress", "source": "installed package metadata"},  # dict entry: what the package is used for, and where its version comes from
        {"tool": "Python", "version": platform.python_version(), "detail": "", "source": "platform"},  # list element: the Python version the run used
        {"tool": "numpy", "version": np.__version__, "detail": "", "source": "package"},  # NumPy version
        {"tool": "pandas", "version": pd.__version__, "detail": "", "source": "package"},  # pandas version
        {"tool": "matplotlib", "version": matplotlib.__version__, "detail": "", "source": "package"}]  # matplotlib
    pd.DataFrame(tool).to_csv(PROCESSED / "preprocessing" / "rinex_toolchain.csv", index=False, lineterminator="\n")  # written (new)
    # one line: every tool with its version
    print("\n      Toolchain written to preprocessing/rinex_toolchain.csv: " + "; ".join(f"{t['tool']} {t['version']}" for t in tool))

    # ---- figures --------------------------------------------------------
    fig_observables(inv_rows, sessions, GRAPH / "observables_l1.png")  # the figure (written every run)
    fig_check(ver, pr_hist, cn_steps, GRAPH / "conversion_check.png")  # the figure (written every run)
    print(f"\n[8] the figures written to Graph/ at {DPI} dpi.")  # the resolution constant

    # ---- report ---------------------------------------------------------
    print("\n" + "=" * 78)  # banner
    print("PLAIN-LANGUAGE REPORT")  # fixed text
    print("=" * 78)  # banner
    # prints the count of verified streams as "converted"
    print(f"\n  {len(ver)} stream-sessions were converted with RTKLIB convbin, demo5 fork,")
    # prints the commit, first 12 characters
    print(f"  commit {conv.get('commit','')[:12]}, to RINEX {RINEX_VERSION}. Every command is")
    print("  recorded in the conversion log with its exit status and timing.")  # fixed text
    # corrected: 974 epochs of u-blox 02 in 2.6.12 were dropped as time-reversed counted from ver
    print(f"\n  The conversion kept every measurement-bearing epoch except {int((ver.binary_epochs_with_measurements - ver.rinex_epochs).sum())} (u-blox 02, 2.6.12, dropped by convbin as time-reversed; u-blox 01 in 2.6.12 converted in three pieces). It preserved the pseudorange exactly,")
    print("  verified satellite by satellite against the binary. It preserved carrier-to-")  # continued
    print("  noise density to better than one quantisation step. L1, G1 and B1 are")  # continued
    # per-epoch presence of the feature bands is tested in the data, not in the header (4); E1 is named here
    print("  present in the data on every stream, Galileo E1 too (per-epoch presence test), tested on")
    print("  each, so the band restriction (Research_Plan_Topu.pdf, section 2.5) is supported by the data")  # continued
    print("  as converted.")  # continued
    print("\n  What it did not preserve is the receivers' own interference status, and")  # fixed text
    print("  that is the intended behaviour rather than a loss. The separation between")  # continued
    print("  what may become a feature and what may only inform a label is now a property")  # continued
    print("  of the directory layout.")  # continued
    if NOTES:  # if: any notes
        # three statements on one line joined by ";": a rule, the title with the count, a rule
        print("\n" + "-" * 78); print(f"NOTES ({len(NOTES)})"); print("-" * 78)
        for n in NOTES:  # for loop over the notes
            print(f"  - {n}")  # prints each
    # same three statements for the anomalies
    print("\n" + "-" * 78); print(f"ANOMALIES ({len(ANOMALIES)})"); print("-" * 78)
    # for loop over the anomalies (the flag() messages), or ["none"] if the list is empty
    for a_ in ANOMALIES or ["none"]:
        print(f"  - {a_}")  # prints each
    print("\n" + "=" * 78)  # banner
    print("No feature was computed and no label was built in this section.")  # fixed text
    print(f"Random seed fixed at {SEED}.")  # prints the seed
    print("=" * 78)  # banner
    # ---- the values this run used to print only, written to Data/processed/printed_facts/ as a table ----
    from helpers import write_facts  # import inside main: the facts-table writer, here so no earlier line of this script moves
    lg = max(diag, key=lambda x: x["longest_empty"][2])  # max with a key function: the stream whose run without measurements is longest
    a_, b_, k_ = lg["longest_empty"]  # tuple unpacking: its first and last empty epoch and its epoch count
    ts_ = lambda t: pd.Timestamp(t).strftime("%Y-%m-%d %H:%M:%S.%f")  # lambda: an instant as full text to the microsecond (the writer rounds it)
    facts = [("longest_empty_test", lg["test_code"], "", "session of the longest run of epochs without any measurement, over every stream"),  # tuple: key, value, unit, definition
             ("longest_empty_stream", lg["stream"], "", "stream of that run"),  # the stream
             ("longest_empty_start", ts_(a_), "UTC", "first epoch of the run (receiver time of the binary)"),  # its start
             ("longest_empty_end", ts_(b_), "UTC", "last epoch of the run"),  # its end
             ("longest_empty_epochs", int(k_), "epochs", "binary epochs in the run"),  # its length in epochs
             ("longest_empty_duration", (b_ - a_).total_seconds(), "s", "last epoch minus first epoch of the run"),  # its length in seconds (Timedelta.total_seconds)
             ("longest_empty_median_sats_before", lg["median_sat_before_empty"], "satellites", "median satellites per RINEX epoch of that stream before the run")]  # list of 4-tuples: the empty run
    # list += a list comprehension: the IGS file's Galileo records by data source, one row per class
    facts += [(f"igs_galileo_records_{k.replace('/', '').lower()}", int(v), "records", f"Galileo records of the IGS file whose data-source field is {k}") for k, v in gal_chk["counts"].items()]
    # list += a list comprehension: pseudoranges of each satellite absent from the IGS file, summed over streams
    facts += [(f"absent_from_igs_pseudoranges_{v_}", int(n_), "pseudoranges", f"pseudoranges of {v_}, absent from the IGS file, over every stream") for v_, n_ in sorted(tot_ab.items())]
    write_facts(__file__, facts)  # written; __file__ names this script, so the table is 01_convert_rinex_facts.csv
    shutil.rmtree(TMPDIR, ignore_errors=True)  # 42j: removes the temporary folder and anything left in it (the fixed folder)


def content_sha256(files: list) -> str:  # function: the final-label run - SHA-256 of the decoded observation text without its creation-date lines
    h = hashlib.sha256()  # a fresh SHA-256 digest object, fed piece by piece below
    for f in files:  # for loop over the stream's compressed files in order (one, or the pieces part01, part02, ...)
        import hatanaka  # import inside the function: the hatanaka package (2.8.1 here) decodes Hatanaka + gzip
        txt = hatanaka.decompress(f)  # the plain RINEX observation text in memory (bytes or str)
        txt = txt if isinstance(txt, bytes) else txt.encode()  # conditional expression: text turned into bytes when needed
        for ln_ in txt.splitlines(keepends=True):  # for loop over the lines, each kept with its own line ending
            # if: skip the one header line whose label (columns 61-80) is PGM / RUN BY / DATE - the converter's creation time
            if ln_[60:80].rstrip() != b"PGM / RUN BY / DATE":
                h.update(ln_)  # every other line goes into the digest unchanged
    return h.hexdigest()  # the digest as 64 hexadecimal characters


if __name__ == "__main__":  # true only when the file is run directly
    main()  # runs the section
