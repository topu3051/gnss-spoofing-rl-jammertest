#!/usr/bin/env python3
# module docstring below: text Python keeps as the file's help (help(module)); rewritten in 
"""
Interference status from the primary receivers.

THE VERDICT on whether the 50 W jamming transmitted from Test Area 1 reached the
recording antenna. The handset script gave an indication from consumer handsets; this
script answers it on survey-grade equipment.

SCOPE DISCIPLINE. No observable is decoded anywhere: no pseudorange, carrier
phase, Doppler or carrier-to-noise density is unpacked or written. The tests
read receiver-internal STATUS only:

  u-blox     UBX-MON-RF (0x0A 0x38); UBX-NAV-PVT (0x01 0x07) time, numSV, fix
  Septentrio RFStatus (4092), ReceiverStatus (4014), QualityInd (4082),
             ChannelStatus (4013) satellite count

The 2.6.12 timeline additionally reads,
for labelling and verification only and never as features: the HEADER of
MeasEpoch (4027) and of UBX-RXM-RAWX (0x02 0x15), i.e. time tag, counts and
satellite IDs; and time, mode and position from PVTGeodetic (4007) and NAV-PVT.

TIME BASE. Septentrio rows carry the receiver's own time stamp. u-blox rows carry
the host laptop's arrival time from the host-timing sidecars, whose
latency against true receiver time is measured in the capture margins.

FIELD DEFINITIONS are taken from the manufacturer documents in docs/ and are
cited in FIELDS below. Directions are inferred from the documented meaning.
No difference is converted to an unsigned magnitude anywhere in this script.

INTEGRITY. Every SBF block is CRC-checked and every UBX message checksum-checked;
block revisions are asserted single-valued; sub-block arithmetic and the bytes
each layout consumes are reconciled with the declared length. A failure in a
block or message type the tests decode stops the run.

Outputs (Data/processed/ unless stated)
  ground_truth/receiver_status_timeseries.csv          ground_truth/receiver_integrity_report.csv
  ground_truth/receiver_field_definitions.csv          ground_truth/receiver_host_latency.csv
  ground_truth/receiver_level_comparison.csv           ground_truth/receiver_within_capture_comparison.csv
  ground_truth/receiver_group_site_split.csv           ground_truth/receiver_transition_timing.csv
  ground_truth/receiver_pyramid_observed.csv           ground_truth/receiver_positive_control.csv
  ground_truth/jammer_1129_decomposition.csv           ground_truth/effective_attack_interval.csv
  ground_truth/straddle_test_1_3_1.csv                 ground_truth/margin_contamination.csv
  ground_truth/margin_clean_minutes.csv                ground_truth/own_jamming_assessment.csv
  ground_truth/receiver_frozen_check.csv               ground_truth/schedule_shift_sensitivity.csv
  ground_truth/interferer_1590mhz.csv                  ground_truth/spoof_2612_timeline.csv
  ground_truth/receiver_control_sensitivity.csv        ground_truth/margin_remnant_1_8_2.csv
  ground_truth/ta3_schedule_confirmation.csv           ground_truth/spoof_2612_navpvt_quality.csv
  Graph/receiver_status_timeseries.png  Graph/rfstatus_frequency_waterfall.png
  Graph/receiver_positive_control.png   Graph/front_end_excursions.png
"""

from __future__ import annotations  # future import: hints kept as text, so "dict | None" works on older Pythons

import binascii  # standard-library binary helpers; crc_hqx computes the SBF CRC-16-CCITT in C, fast
import importlib.util  # standard-library tools for loading the timeline script by its path
import random  # standard-library random numbers; only seeded, nothing here draws from it
import re  # standard-library regular expressions, used on the plan and catalogue text
import struct  # standard-library binary unpacking: turns raw bytes into numbers by a format string
import sys  # standard-library system access: stderr, exit, sys.modules
from collections import Counter, defaultdict  # Counter counts keys; defaultdict creates a missing key's value
from datetime import datetime, timedelta, timezone  # date-time classes: instants, durations, the UTC zone
from pathlib import Path  # Path objects for building and reading file paths

import os  # standard-library operating-system access: sets MPLCONFIGDIR before matplotlib loads

import numpy as np  # third-party numerical library under the short name np
import pandas as pd  # third-party table library under the short name pd

# Path: codes/.mplconfig, matplotlib's configuration and font-cache folder inside the project folder (as in the handset script)
MPLCONFIG = Path(__file__).resolve().parent / ".mplconfig"
MPLCONFIG.mkdir(exist_ok=True)  # creates the folder if missing; exist_ok=True means no error when it exists
os.environ["MPLCONFIGDIR"] = str(MPLCONFIG)  # os.environ: tells matplotlib, imported next, to use that folder
import matplotlib  # third-party plotting library, imported first so its back end can be chosen
matplotlib.use("Agg")  # "Agg" back end: draws straight to image files, no screen window needed
import matplotlib.dates as mdates  # matplotlib's date helpers: datetime <-> plot number, time ticks
import matplotlib.pyplot as plt  # plotting interface under the short name plt; must come after use("Agg")
from matplotlib.patches import Patch  # coloured-box shape used to build custom legend entries

# helpers shared with the handset script: one copy each, in codes/common/helpers.py next to this script
from helpers import (G50, GBL, GHI, GPS_EPOCH, GUNK, LEAP_SECONDS, break_gaps, cliffs_delta, die,
                            # continuation of the from-import: SBF time conversion and distance
                            gps_to_utc, haversine_m,
                            # continuation of the from-import: host clock, IQR, schedule, pyramid, saving, tagging, axis
                            host_clock, host_times, iqr, load_schedule, parse_pyramid, save_fig, tag_times, time_axis)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

SEED = 20260914  # constant: the project's fixed seed; nothing here draws a random number (cliffs_delta is exact)
random.seed(SEED)  # seeds Python's random generator (unused here, fixed on principle)
np.random.seed(SEED)  # seeds NumPy's global generator (unused, fixed on principle)

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
DATA = PROJECT / "Data"  # the '/' operator on a Path joins path parts
DAY = DATA / "Day1_14092026"  # the raw capture-day folder holding the .sbf and .ubx binaries (read only)
PROCESSED = DATA / "processed"  # where the 24 tables of this script are written (one to_csv call each)
GRAPH = PROJECT / "Graph"  # where the 4 figures of this script are written
DOCS = PROJECT / "docs"  # the manufacturer PDFs; only listed in the printout, never parsed by this script
TIMELINE_SCRIPT = PROJECT / "codes" / "1_ground_truth" / "01_timeline.py"  # the timeline script, whose plan parser is reused

TZ_UTC = timezone.utc  # the UTC time-zone object, the internal time base
TEST_DAY = "2026-09-14"  # the capture date as text (not used below)

HIGH_POWER_W = 1.0  # float constant (watts): the "above 1 W" boundary of the baseline group
FIFTY_W_MIN = 40.0  # float constant (watts): a Test Area 1 test at or above 40 W counts as "50 W"

# Appendix A, Table 1 survey points.
# tuple: the Test Area 2 origin, used as a stand-in for handheld jammer 1.1.29's position
TA2_ORIG = (69.2225166, 15.9334241)    # ORIG, "Origo Site2"
# The recording position is no longer a constant: each session's site and its distance to the 50 W transmitter are
# measured in the handset script (ground_truth/session_distance_to_sender.csv), and read from there.

# string constant (two adjacent strings joined): the Septentrio document cited in FIELDS
SEPT_DOC = ("Septentrio mosaic-X5 Firmware v4.14.10 Reference Guide, section 4 "
            "(SBF Reference)")
# string constant: the u-blox document cited in FIELDS
UBX_DOC = ("u-blox F9 HPG 1.51 Interface description, UBXDOC-963802114-13124, "
           "section 3.14.9")

# Septentrio AGCState FrontEndID bits 0-4, from SEPT_DOC ReceiverStatus.
# dict: front-end code (0-14) -> name, exactly as listed in SEPT_DOC's AGCState sub-block (verified in docs/)
FRONTEND = {0: "GPSL1/E1", 1: "GLOL1", 2: "E6", 3: "GPSL2", 4: "GLOL2",
            5: "L5/E5a/B2a", 6: "E5b/B2b", 7: "E5(a+b)", 8: "CombinedL1",  # dict entry: front-end codes 5 to 8, the L5, E5b, wideband E5 and combined L1 front ends
            9: "CombinedL2", 10: "MSS/Lband", 11: "B1", 12: "B3", 13: "Sband",  # dict entry: front-end codes 9 to 13, combined L2, L-band, BeiDou B1 and B3, and S-band
            14: "B3+E6"}  # dict entry: front-end code 14, the shared B3 and E6 front end
# Transmission-plan band names to Septentrio front-end codes.
# dict: plan band name -> front-end code (the mapping is the author's; the codes come from FRONTEND)
PLAN_BAND_TO_FRONTEND = {"L1": 0, "G1": 1, "L2": 3, "L5": 5, "E6": 2,
                         "B1I": 11, "E5b": 6, "G2": 4}  # dict entry: the plan's B1I, E5b and G2 bands mapped to their front-end codes
# The bands the group 2.6 description names.
# list of band names; not used below (parse_own_jamming re-reads the bands from the catalogue)
OWN_INITIAL_JAM_BANDS = ["L1", "G1", "B1"]          # first five minutes
# list of band names; likewise not used below
OWN_CONTINUOUS_JAM_BANDS = ["L2", "L5", "E5b", "E6"]  # throughout
# dict: catalogue band name -> front-end code, used by the own-jamming assessment
OWN_BAND_TO_FRONTEND = {"L1": 0, "G1": 1, "B1": 11, "L2": 3, "L5": 5,
                        "E5b": 6, "E6": 2}  # dict entry: the catalogue's E5b and E6 bands mapped to their front-end codes

C_ORANGE, C_SKY, C_GREEN = "#E69F00", "#56B4E9", "#009E73"  # tuple assignment of three hex colours (Okabe-Ito)
C_BLUE, C_VERMILLION, C_PURPLE = "#0072B2", "#D55E00", "#CC79A7"  # three more palette colours
C_GREY, C_BLACK = "#999999", "#000000"  # grey for grids, black for edges

NOTES: list[str] = []  # annotated empty list: findings printed at the end
ANOMALIES: list[str] = []  # annotated empty list: things that look wrong, printed at the end


def note(m): NOTES.append(m)  # one-line function: the body sits after the colon; records a finding
def flag(m): ANOMALIES.append(m)  # one-line function: records an anomaly


# --------------------------------------------------------------------------
# Field definitions, with citation and direction fixed in advance
# --------------------------------------------------------------------------
# direction: -1 means the field DECREASES under interference, +1 INCREASES,
# 0 means no direction is claimed and the field is carried for description only.

# list of dicts, one per status field: meaning and source verified in docs/; direction = author's pre-registration
FIELDS = [
    # agc_cnt: UBX_DOC MON-RF U2 at 18+n*24, "AGC Monitor ... 0 to 8191 (100%)" (verified in docs/)
    dict(receiver="u-blox", field="agc_cnt", units="0-8191 (percentage of maximum gain)",
         meaning="AGC Monitor, as percentage of maximum gain, range 0 to 8191 (100%)",  # keyword argument meaning=: the u-blox gain count as the documentation defines it
         direction=-1, source=UBX_DOC + ", UBX-MON-RF payload offset 18+n*24"),  # keyword arguments of dict(...): -1 means a lower gain count signals interference; then the source
    # cw_suppression: MON-RF U1 at 20+n*24, 0 = no CW jamming, 255 = strong (verified in docs/)
    dict(receiver="u-blox", field="cw_suppression", units="0-255 (scaled)",
         meaning=("CW interference suppression level, scaled (0 = no CW jamming, "  # keyword argument meaning=: the CW suppression level as the documentation defines it
                  "255 = strong CW jamming)"),
         direction=+1, source=UBX_DOC + ", UBX-MON-RF payload offset 20+n*24"),  # keyword arguments of dict(...): +1 means a higher suppression level signals interference
    # noise_per_ms: MON-RF U2 at 16+n*24, "Noise level as measured by the GPS core" (verified in docs/)
    dict(receiver="u-blox", field="noise_per_ms", units="counts",
         meaning="Noise level as measured by the GPS core",  # keyword argument meaning=: the noise level as the documentation defines it
         direction=+1, source=UBX_DOC + ", UBX-MON-RF payload offset 16+n*24"),  # keyword arguments of dict(...): +1 means a higher noise level signals interference
    # jamming_state: MON-RF flags bits 1..0 (verified in docs/); docs add it is always 0 where UBX-SEC-SIG exists
    dict(receiver="u-blox", field="jamming_state", units="0-3",
         meaning=("Output from jamming/interference monitor (0 = unknown or feature "  # keyword argument meaning=: the jamming monitor's four states as the documentation lists them
                  "disabled, 1 = ok, 2 = warning - interference visible but fix OK, "
                  "3 = critical - interference visible and no fix)"),
         direction=+1, source=UBX_DOC + ", UBX-MON-RF flags bits 1..0"),  # keyword arguments of dict(...): +1 means a higher jamming state signals interference
    # n_satellites: NAV-PVT U1 numSV at offset 23, "Number of satellites used in Nav Solution" (verified in docs/)
    dict(receiver="u-blox", field="n_satellites", units="count",
         meaning="Number of satellites used in the navigation solution (UBX-NAV-PVT numSV)",  # keyword argument meaning=: satellites used in the u-blox navigation solution
         direction=-1, source="u-blox F9 HPG 1.51 Interface description, UBX-NAV-PVT offset 23"),  # keyword arguments of dict(...): -1 means fewer satellites used signals interference
    # agc_gain_db: AGCState Gain i1, 1 dB, Do-Not-Use -128 = front-end PLL not locked (verified in docs/)
    dict(receiver="Septentrio", field="agc_gain_db", units="dB",
         meaning="AGC gain, in dB (Do-Not-Use -128 indicates the frontend PLL is not locked)",  # keyword argument meaning=: the Septentrio front-end gain and its do-not-use value
         direction=-1, source=SEPT_DOC + ", ReceiverStatus (4014) AGCState sub-block"),  # keyword arguments of dict(...): -1 means a lower gain signals interference
    # sample_var: AGCState SampleVar u1, nominal 100, Do-Not-Use 0 (verified in docs/; the 0 is not filtered)
    dict(receiver="Septentrio", field="sample_var", units="normalised, nominal 100",
         meaning="Normalized variance of the IF samples; the nominal value is 100",  # keyword argument meaning=: the normalised variance of the IF samples, nominally 100
         direction=0, source=SEPT_DOC + ", ReceiverStatus (4014) AGCState sub-block"),  # keyword arguments of dict(...): 0 means no direction is fixed in advance for this field
    # blanking_pct: AGCState BlankingStat u1, 1 %, always 0 without a pulse blanking unit (verified in docs/)
    dict(receiver="Septentrio", field="blanking_pct", units="%",
         meaning=("Current percentage of samples being blanked by the pulse blanking "  # keyword argument meaning=: the share of samples the pulse blanking unit removes
                  "unit; always 0 on receivers without a pulse blanking unit"),
         direction=+1, source=SEPT_DOC + ", ReceiverStatus (4014) AGCState sub-block"),  # keyword arguments of dict(...): +1 means more blanking signals interference
    # rf_n_bands: RFStatus N u1, number of RFBand sub-blocks (verified in docs/)
    dict(receiver="Septentrio", field="rf_n_bands", units="count",
         meaning=("Number of RFBand sub-blocks, i.e. frequency bands in which the "  # keyword argument meaning=: the number of bands the receiver declares interfered
                  "receiver has detected and/or mitigated interference"),
         direction=+1, source=SEPT_DOC + ", RFStatus (4092) field N"),  # keyword arguments of dict(...): +1 means more declared bands signal interference
    # rf_flag_notauthentic: RFStatus Flags bit 0 (verified in docs/; docs add it may be set with no RFBand at all)
    dict(receiver="Septentrio", field="rf_flag_notauthentic", units="0 or 1",
         meaning=("RFStatus Flags bit 0: set when the receiver determined that the "  # keyword argument meaning=: the receiver's own flag that the signals may not be authentic
                  "GNSS signals at its RF connector may not be authentic and that its "
                  "output may be misleading; the receiver may be connected to a GNSS "
                  "simulator or be subject to a spoofing attack"),
         direction=+1, source=SEPT_DOC + ", RFStatus (4092) Flags bit 0"),  # keyword arguments of dict(...): +1 means a raised flag signals an attack
    # rf_flag_nma_fail: RFStatus Flags bit 1, set when navigation message authentication fails (verified in docs/)
    dict(receiver="Septentrio", field="rf_flag_nma_fail", units="0 or 1",
         meaning=("RFStatus Flags bit 1: set when a non-authentic navigation message "  # keyword argument meaning=: the flag raised when navigation message authentication fails
                  "is detected by navigation message authentication checks, for "
                  "example Galileo OSNMA"),
         direction=+1, source=SEPT_DOC + ", RFStatus (4092) Flags bit 1"),  # keyword arguments of dict(...): +1 means a raised flag signals an attack
    # quality_main_signal: QualityInd type 1, value 0-10, 15 = unknown (verified in docs/)
    dict(receiver="Septentrio", field="quality_main_signal", units="0-10",
         meaning=("QualityInd indicator type 1, GNSS signals from main antenna; "  # keyword argument meaning=: signal quality from the main antenna on a 0 to 10 scale
                  "0 is poor quality and 10 very high quality"),
         direction=-1, source=SEPT_DOC + ", QualityInd (4082) indicator type 1"),  # keyword arguments of dict(...): -1 means a lower quality signals interference
    # quality_overall: QualityInd type 0 (verified in docs/)
    dict(receiver="Septentrio", field="quality_overall", units="0-10",
         meaning="QualityInd indicator type 0, overall quality",  # keyword argument meaning=: the receiver's overall quality indicator
         direction=-1, source=SEPT_DOC + ", QualityInd (4082) indicator type 0"),  # keyword arguments of dict(...): -1 means a lower quality signals interference
    # quality_rf_power_main: QualityInd type 11, RF power level from the main antenna (verified in docs/)
    dict(receiver="Septentrio", field="quality_rf_power_main", units="0-10",
         meaning="QualityInd indicator type 11, RF power level from the main antenna",  # keyword argument meaning=: the RF power level from the main antenna
         direction=-1, source=SEPT_DOC + ", QualityInd (4082) indicator type 11"),  # keyword arguments of dict(...): -1 means a lower power level signals interference
    # n_satellites: ChannelStatus N, "number of ChannelSatInfo sub-blocks", i.e. active channels (verified in docs/)
    dict(receiver="Septentrio", field="n_satellites", units="count",
         meaning="Number of active satellite channels (ChannelStatus N)",  # keyword argument meaning=: the number of active satellite channels
         direction=-1, source=SEPT_DOC + ", ChannelStatus (4013) field N"),  # keyword arguments of dict(...): -1 means fewer active channels signal interference
]
# dict comprehension: (receiver family, field) -> direction fixed in advance (-1, 0 or +1), for fast lookup
DIRECTION = {(f["receiver"], f["field"]): f["direction"] for f in FIELDS}

# Fields carried into the per-epoch table that the statistical tests run on.
SEPT_TEST_FIELDS = ["agc_gain_db", "blanking_pct"]  # list: the Septentrio fields the tests use
UBX_TEST_FIELDS = ["agc_cnt", "cw_suppression", "noise_per_ms"]  # list: the u-blox fields the tests use
# Fields that can respond to a spread-spectrum spoofing signal. blanking_pct
# responds to pulsed interference and cw_suppression to continuous-wave
# interference; a spoofer transmits neither, so both are structurally near zero
# and are excluded from any per-session summary.
# list; never used, and blanking_pct/cw_suppression do enter the control and 1.1.29 CSVs (this note said "not in corrections log" until 32)
RESPONSIVE_FIELDS = ["agc_gain_db", "agc_cnt", "noise_per_ms"]


# --------------------------------------------------------------------------
# Binary readers with full integrity verification
# --------------------------------------------------------------------------

class Integrity:  # class: a small container collecting one integrity row per file read
    def __init__(self):  # constructor: runs when Integrity() is created; self is the new object
        self.rows = []  # an empty list stored on the object
        # what the u-blox streams contain, so the model inference is printed from counts, not typed
        self.ubx_types: Counter = Counter()  # (class, id) -> valid messages, across every file read
        self.ubx_facts: defaultdict = defaultdict(set)  # fact name -> set of values seen (versions, lengths, blocks)

    def add(self, **kw):  # method: "**kw" gathers any keyword arguments into a dict
        self.rows.append(kw)  # stores that dict as one row


# generator function (it uses "yield"): walks one .sbf file block by block, handing back the wanted blocks
def sbf_blocks(path: Path, integ: Integrity, want: set[int]):
    """
    Yield (block_number, revision, tow_ms, wnc, payload, end_offset) for wanted blocks;
    end_offset (one past the block's last byte) locates the block in the host-timing sidecar.

    Every block in the file is CRC-16-CCITT verified (polynomial 0x1021,
    initial value 0, over the bytes from the ID field to the end of the block),
    which binascii.crc_hqx implements exactly.

    The parse is strictly sequential, for the same reason as the UBX reader: a
    scanner that searches for every occurrence of the sync pair finds the bytes
    $@ inside payloads and counts their failed CRCs as corruption. Under a
    sequential parse a CRC failure is a real one, and the resync count becomes a
    meaningful measure of how often the stream had to be re-acquired.
    """
    data = path.read_bytes()  # the whole file as one bytes object (up to ~160 MB); data[k] is byte k as an int 0-255
    n = len(data)  # file size in bytes
    i = 0  # read position: the byte where the current block should start
    total = crc_bad = len_bad = resync = 0  # chained assignment of four counters (len_bad is counted, never reported)
    synced = False  # True once a valid block has been read, so later failures count as real ones
    revs: dict[int, set] = defaultdict(set)  # block number -> set of revisions seen; defaultdict makes empty sets
    failed_blocks: Counter = Counter()  # block number -> CRC failures
    while i < n - 8:  # while loop: continue while at least a full 8-byte header could still fit
        if not (data[i] == 0x24 and data[i + 1] == 0x40):  # if: no sync pair "$@" (0x24 0x40, SEPT_DOC 4.1.1) here
            j = data.find(b"\x24\x40", i)  # bytes.find: position of the next "$@" at or after i, or -1
            if j < 0 or j + 8 > n:  # if: no further sync, or too close to the end for a header
                break  # leaves the loop: the file is finished
            if synced:  # if: we had been in step and lost it
                resync += 1  # counts a re-acquisition
            i = j  # jumps to the candidate block start
        # "<HHH" from byte i+2: "<" little-endian, "H" unsigned 2 bytes, three times = CRC u2, ID u2, Length u2
        crc, ident, length = struct.unpack_from("<HHH", data, i + 2)
        # ID bit field: "& 0x1FFF" keeps bits 0-12 = block number; ">> 13" shifts bits 13-15 down = revision
        blk, rev = ident & 0x1FFF, ident >> 13
        # if: Length below the 8-byte header, not a multiple of 4 ("%" remainder), or past the end: a false sync
        if length < 8 or length % 4 or i + length > n:
            if synced:  # if: we had been in step
                len_bad += 1  # counts a bad length (never written out)
            synced = False  # out of step now
            i += 2  # steps past this "$@" and searches again
            continue  # next loop iteration
        # CRC-16-CCITT (x^16+x^12+x^5+1, seed 0) over ID..end of block (SEPT_DOC 4.1.1); crc_hqx computes exactly that
        if binascii.crc_hqx(data[i + 4:i + length], 0) != crc:
            if synced:  # if: a failure while in step is a real one
                crc_bad += 1  # counts it
                failed_blocks[blk] += 1  # and which block type failed (from its possibly corrupt ID)
            synced = False  # out of step now
            i += 2  # steps past and searches again
            continue  # next loop iteration
        total += 1  # a valid block
        synced = True  # in step
        revs[blk].add(rev)  # records the revision of this block type
        if blk in want:  # if: one of the four status blocks this section decodes
            body = data[i + 8:i + length]  # slice: the block after the 8-byte header
            # "<IH": little-endian "I" unsigned 4 bytes = TOW (ms), "H" unsigned 2 bytes = WNc (weeks)
            tow, wnc = struct.unpack_from("<IH", body, 0)
            # yield hands one result to the caller; body[6:] skips TOW and WNc; i + length is the block's end offset
            yield blk, rev, tow, wnc, body[6:], i + length
        i += length  # sequential parse: the next block must begin right after this one
    # records one integrity row for this file ("framing_failures" is the resync count)
    integ.add(file=path.name, kind="SBF", blocks_or_messages=total,
              crc_failures=crc_bad, framing_failures=resync,  # keyword arguments of add(): blocks failing their checksum, and times the reader had to resynchronise
              failed_block_types=dict(failed_blocks),  # keyword argument failed_block_types=: failed blocks counted by block type
              revisions={b: sorted(v) for b, v in sorted(revs.items())},  # keyword argument revisions=: the block revisions seen, per block type, sorted
              early_stop_offset=None, file_bytes=n)  # SBF stops only on a missing sync, never mid-block


# generator function: walks one .ubx file message by message, handing back wanted (class, id) messages
def ubx_messages(path: Path, integ: Integrity, want: set[tuple[int, int]]):
    """
    Yield (cls, id, payload, end_offset) for wanted messages; end_offset (one past the checksum) locates the
    message in the host-timing sidecar.

    Every message is verified with the two-byte Fletcher checksum over the bytes
    from the class field to the end of the payload.

    The parse is strictly sequential: once a valid message has been read, the
    next must begin at the byte immediately after it. Only at the start of the
    file, or after a failure, is the stream searched for the next sync pair.
    This distinction matters, because a scanner that searches for every sync
    pair will find byte sequences b5 62 inside payloads and count their failed
    checksums as corruption. Under a sequential parse a checksum failure is a
    real one.
    """
    data = path.read_bytes()  # the whole file as bytes
    n = len(data)  # file size
    i = 0  # read position
    total = ck_bad = resync = 0  # counters: valid messages, checksum failures, re-acquisitions
    failed_ids: Counter = Counter()  # "0xCC 0xII" class/ID as read from each failed message -> count
    early_stop = None  # byte offset where a message ran past the end of the file, if it happens
    synced = False  # True once a valid message has been read
    while i < n - 8:  # while at least a header and checksum could still fit
        if not (data[i] == 0xB5 and data[i + 1] == 0x62):  # if: no UBX preamble 0xb5 0x62 (UBX_DOC 3.2) here
            j = data.find(b"\xb5\x62", i)  # the next preamble, or -1
            if j < 0:  # if: none left (the file tail is NMEA text only)
                break  # leaves the loop
            if synced:  # if: we had been in step (interleaved NMEA sentences cause these)
                resync += 1  # counts a re-acquisition
            i = j  # jumps to the candidate message
        # "<H" little-endian unsigned 2 bytes at i+4: the payload length (UBX_DOC 3.2); [0] takes it out of the tuple
        ln = struct.unpack_from("<H", data, i + 4)[0]
        # if: preamble 2 + class 1 + id 1 + length 2 + payload + checksum 2 runs past the end of the file
        if i + 6 + ln + 2 > n:
            early_stop = i  # recorded with its byte offset (reported by main) rather than passed over silently
            break  # the rest of the file cannot hold a complete message
        cls, mid = data[i + 2], data[i + 3]  # indexing bytes gives ints: the message class and ID
        # a NumPy view (no copy) of class, ID, length and payload: the bytes the checksum covers (UBX_DOC 3.4)
        arr = np.frombuffer(data, dtype=np.uint8, count=4 + ln, offset=i + 2)
        ck_a = int(arr.sum() & 0xFF)  # 8-bit Fletcher CK_A: sum of all bytes, "& 0xFF" keeps the low 8 bits
        # CK_B = running sum of CK_A = each byte weighted by how many sums include it (N, N-1, ..., 1); low 8 bits
        ck_b = int((np.arange(len(arr), 0, -1, dtype=np.int64) * arr).sum() & 0xFF)
        if ck_a == data[i + 6 + ln] and ck_b == data[i + 7 + ln]:  # if: both match the two bytes after the payload
            total += 1  # a valid message
            synced = True  # in step
            integ.ubx_types[(cls, mid)] += 1  # counts this message type
            if (cls, mid) == (0x02, 0x15) and ln > 13:  # if: RXM-RAWX - its version byte is payload offset 13 (UBX_DOC)
                integ.ubx_facts["RXM-RAWX version"].add(data[i + 6 + 13])  # the version, kept
            elif (cls, mid) == (0x02, 0x13) and ln > 6:  # elif: RXM-SFRBX - its version byte is payload offset 6
                integ.ubx_facts["RXM-SFRBX version"].add(data[i + 6 + 6])  # kept
            elif (cls, mid) == (0x01, 0x07):  # elif: NAV-PVT - its payload length
                integ.ubx_facts["NAV-PVT payload bytes"].add(ln)  # kept
            elif (cls, mid) == (0x0A, 0x38) and ln > 1:  # elif: MON-RF - nBlocks is payload offset 1 (UBX_DOC)
                integ.ubx_facts["MON-RF RF blocks"].add(data[i + 6 + 1])  # kept
            if (cls, mid) in want:  # if: MON-RF (0x0A 0x38) or NAV-PVT (0x01 0x07)
                yield cls, mid, data[i + 6:i + 6 + ln], i + 6 + ln + 2  # class, ID, payload slice, end offset
            i += 6 + ln + 2  # sequential parse: next message starts right after the checksum
        else:  # else: checksum mismatch
            if synced:  # if: we had been in step, so this is a real failure
                ck_bad += 1  # counts it
                # 30h: class and ID as read; they may themselves be damaged, so the attribution is probable, not certain
                failed_ids[f"0x{data[i + 2]:02X} 0x{data[i + 3]:02X}"] += 1
                synced = False  # out of step
            i += 2  # steps past this preamble and searches again
    # records the file's integrity row: failed messages keyed by class/ID as read, and any early stop
    integ.add(file=path.name, kind="UBX", blocks_or_messages=total,
              crc_failures=ck_bad, framing_failures=resync,  # keyword arguments of add(): messages failing their checksum, and times the reader had to resynchronise
              failed_block_types=dict(failed_ids), revisions={},  # keyword arguments of add(): failed messages by class and ID; UBX has no block revisions
              early_stop_offset=early_stop, file_bytes=n)  # keyword arguments of add(): where the file stopped early, if it did, and its size in bytes


# function: product and firmware of every Septentrio stream, read from its ReceiverSetup block (SBF 5902) rather than
# typed. Offsets are from the start of the block after TOW and WNc, i.e. 14 bytes into the block
# (SEPT_DOC 4.1 ReceiverSetup): RxVersion c1[20] at 182, GNSSFWVersion c1[40] at 274 (rev >= 2), ProductName
# c1[40] at 314 (rev >= 3); unused bytes are zero, so each string is cut at its first zero byte.
def receiver_setup(sessions: pd.DataFrame) -> pd.DataFrame:
    rows = []  # one row per stream that carries the block
    for _, sr in sessions.iterrows():  # for loop over the sessions
        for f in sorted((DAY / sr["session_folder"]).glob("*.sbf")):  # for loop over that session's SBF files
            # generator over this file's ReceiverSetup blocks; a throwaway Integrity keeps the main report unchanged
            for blk, rev, tow, wnc, p, end in sbf_blocks(f, Integrity(), {5902}):
                txt = lambda a, n: p[a:a + n].split(b"\0")[0].decode("ascii", "replace")  # lambda: a zero-ended string
                rows.append({"test_code": sr["test_code"], "file": f.name, "revision": rev,  # the row
                             "rx_version": txt(182, 20), "gnss_fw_version": txt(274, 40) if rev >= 2 else "",  # dict entry: receiver software version, and the GNSS firmware version from revision 2 on
                             "product_name": txt(314, 40) if rev >= 3 else ""})  # dict entry: the product name, present from block revision 3 on
                break  # the first block of the file is enough: the setup does not change within a recording
    return pd.DataFrame(rows).drop_duplicates(["file"])  # one row per file


# --------------------------------------------------------------------------
# Block decoders, each reporting bytes consumed against declared length
# --------------------------------------------------------------------------

# defaultdict whose missing keys get a fresh dict from the lambda: per block type, byte-accounting tallies
LAYOUT = defaultdict(lambda: {"consumed": Counter(), "declared": Counter(),
                              # tallies: sub-block overruns, fields read past the declared length, non-zero padding
                              "subblock_mismatch": 0, "consumed_over_declared": 0, "nonzero_padding": 0})


def _account(name: str, consumed: int, payload: bytes) -> None:  # function: tallies bytes read vs bytes present
    """Record bytes consumed against declared, and check the remainder is padding."""
    rec = LAYOUT[name]  # this block type's tallies (created on first use)
    rec["consumed"][consumed] += 1  # counts how often this many bytes were read
    rec["declared"][len(payload)] += 1  # counts how often the payload had this length
    if consumed > len(payload):  # if: the decoded fields ran past this block's own declared length
        rec["consumed_over_declared"] += 1  # counted per block; main() stops the run if above 0
    rest = payload[consumed:]  # slice: the bytes after the decoded fields
    # if: any of them is non-zero; SEPT_DOC 4.1.5 says padding values are "not defined", so this is counted and
    # reported, not treated as a decoding error
    if rest and any(rest):
        rec["nonzero_padding"] += 1  # counted and printed by main()


def decode_rfstatus(p: bytes) -> dict | None:  # function: RFStatus payload (after TOW/WNc) -> dict, or None
    """RFStatus (4092). Header N u1, SBLength u1, Flags u1, Reserved u1[3]."""
    if len(p) < 6:  # if: shorter than N, SBLength, Flags and Reserved[3] (6 bytes)
        return None  # too short to decode (never happened in this data)
    # "<BBB": little-endian, three "B" unsigned 1-byte values = N, SBLength, Flags (SEPT_DOC RFStatus)
    N, SB, flags = struct.unpack_from("<BBB", p, 0)
    # if: 6 header bytes + N sub-blocks of SB bytes exceed the payload (skipped when SB is 0)
    if SB and 6 + N * SB > len(p):
        LAYOUT["RFStatus"]["subblock_mismatch"] += 1  # counted; main() stops the run if above 0
        return None  # layout does not fit (never happened in this data)
    bands = []  # list of (frequency, bandwidth, mode, antenna) tuples
    for k in range(N):  # for loop over the N RFBand sub-blocks
        o = 6 + k * SB  # sub-block k starts after the 6-byte header and k earlier sub-blocks
        # "<IHB": "I" unsigned 4 bytes = Frequency (Hz), "H" unsigned 2 bytes = Bandwidth (kHz), "B" 1 byte = Info
        freq, bw, info = struct.unpack_from("<IHB", p, o)
        # Info: "& 0x0F" keeps bits 0-3 = Mode (1 notch, 2 cancelled, 8 detected); ">> 6 & 0x03" = bits 6-7 antenna
        bands.append((freq, bw, info & 0x0F, (info >> 6) & 0x03))
    _account("RFStatus", 6 + N * SB, p)  # byte accounting: header plus sub-blocks read
    # returns the count and the two Flags bits: "& 0x01" isolates bit 0, "& 0x02" bit 1; bool then int gives 0/1
    return {"rf_n_bands": N,
            "rf_flag_notauthentic": int(bool(flags & 0x01)),  # dict entry: flag bit 0, the receiver's own judgement that the signals may not be authentic
            "rf_flag_nma_fail": int(bool(flags & 0x02)),  # dict entry: flag bit 1, set when navigation message authentication fails
            "rf_bands": bands}  # dict entry: the list of bands declared interfered, with frequency, bandwidth and mode


def decode_receiverstatus(p: bytes) -> dict | None:  # function: ReceiverStatus payload -> dict with AGC list
    """ReceiverStatus (4014). Header 18 bytes then N AGCState sub-blocks of 4."""
    if len(p) < 18:  # if: shorter than the fixed 18-byte part
        return None  # too short (never happened in this data)
    # "<BBIIIBBBB" = 18 bytes, "<" little-endian, B u1, I u4: B CPULoad, B ExtError, I UpTime (s), I RxState,
    (cpu, exterr, uptime, rxstate, rxerr, N, SB, cmd, temp
     ) = struct.unpack_from("<BBIIIBBBB", p, 0)  # ... I RxError, B N, B SBLength, B CmdCount, B Temperature
    if SB and 18 + N * SB > len(p):  # if: the N AGCState sub-blocks would run past the payload
        LAYOUT["ReceiverStatus"]["subblock_mismatch"] += 1  # counted; stops the run if above 0
        return None  # does not fit (never happened in this data)
    agc = []  # list of per-front-end dicts
    for k in range(N):  # for loop over the AGCState sub-blocks
        o = 18 + k * SB  # sub-block k's start
        # "<BbBB": B FrontEndID u1, b Gain i1 (SIGNED 1 byte, -128..127 dB), B SampleVar u1, B BlankingStat u1
        fid, gain, svar, blank = struct.unpack_from("<BbBB", p, o)
        # FrontEndID: "& 0x1F" keeps bits 0-4 = front-end code; ">> 5 & 0x07" = bits 5-7 antenna (0 main, 1 Aux1)
        agc.append({"frontend_code": fid & 0x1F, "antenna_id": (fid >> 5) & 0x07,
                    # Gain Do-Not-Use -128 (PLL not locked) becomes None
                    "agc_gain_db": None if gain == -128 else gain,
                    "pll_unlocked": gain == -128,  # boolean flag kept alongside
                    # SampleVar's Do-Not-Use value 0 becomes None (SEPT_DOC 4014)
                    "sample_var": svar if svar != 0 else None, "blanking_pct": blank})
    _account("ReceiverStatus", 18 + N * SB, p)  # byte accounting
    # returns header fields; Temperature has an offset of 100 and Do-Not-Use 0, so 0 -> None (SEPT_DOC 4014)
    # CPULoad's Do-Not-Use value 255 becomes None (SEPT_DOC 4014)
    return {"cpu_load_pct": cpu if cpu != 255 else None, "ext_error": exterr, "uptime_s": uptime,
            "rx_state": rxstate, "rx_error": rxerr,  # dict entry: receiver state and error words as read
            "temperature_c": temp - 100 if temp else None, "agc": agc}  # dict entry: temperature in degrees Celsius after removing the offset of 100, and the gain blocks


def decode_qualityind(p: bytes) -> dict | None:  # function: QualityInd payload -> dict of four indicators
    """QualityInd (4082). N u1, Reserved u1, then N u2 indicators."""
    if len(p) < 2:  # if: too short for N and Reserved
        return None  # (never happened in this data)
    N = p[0]  # indexing bytes gives an int: the number of indicators
    if 2 + N * 2 > len(p):  # if: 2 header bytes + N two-byte indicators exceed the payload
        LAYOUT["QualityInd"]["subblock_mismatch"] += 1  # counted; stops the run if above 0
        return None  # (never happened in this data)
    out = {}  # dict: indicator type -> value
    for k in range(N):  # for loop over the indicators
        v = struct.unpack_from("<H", p, 2 + k * 2)[0]  # "<H" unsigned 2 bytes, little-endian: one indicator word
        # "& 0x00FF" keeps bits 0-7 = type; ">> 8 & 0x0F" takes bits 8-11 = value 0-10 (SEPT_DOC QualityInd)
        typ, val = v & 0x00FF, (v >> 8) & 0x0F
        out[typ] = None if val == 15 else val  # 15 means "unknown" in the docs, stored as None
    _account("QualityInd", 2 + N * 2, p)  # byte accounting
    # dict.get returns None for a type the block did not contain: 0 overall, 1 main signal, 11 RF power, 21 CPU
    return {"quality_overall": out.get(0), "quality_main_signal": out.get(1),
            "quality_rf_power_main": out.get(11), "quality_cpu_headroom": out.get(21)}  # dict entry: RF power from the main antenna and the processor headroom indicator


def decode_channelstatus(p: bytes) -> dict | None:  # function: ChannelStatus payload -> satellite count only
    """ChannelStatus (4013). Only the satellite count N is taken."""
    if len(p) < 3:  # if: too short for N, SB1Length, SB2Length
        return None  # (never happened in this data)
    # "<BBB": three unsigned bytes = N, SB1Length, SB2Length; three reserved bytes follow (SEPT_DOC 4013)
    N, SB1, SB2 = struct.unpack_from("<BBB", p, 0)
    o = 6  # the first ChannelSatInfo sub-block starts after the 6-byte header
    for _k in range(N):  # for loop over the N ChannelSatInfo sub-blocks, walking their lengths only
        # ChannelSatInfo: SVID u1, FreqNr u1, Reserved1 u1[2], Azimuth/RiseSet u2, HealthStatus u2, Elevation i1,
        # N2 u1, RxChannel u1, Reserved2 u1 (SEPT_DOC 4013), so N2 is the byte at offset 9
        if o + 10 > len(p):  # if: the sub-block's N2 byte (offset 9) would lie past the payload
            LAYOUT["ChannelStatus"]["subblock_mismatch"] += 1  # counted; main() stops the run if above 0
            return None  # does not fit
        o += SB1 + p[o + 9] * SB2  # p[o + 9] is N2, the count of nested ChannelStateInfo sub-blocks after this one
    if o > len(p):  # if: the walked length exceeds the payload
        LAYOUT["ChannelStatus"]["subblock_mismatch"] += 1  # counted; stops the run
        return None  # does not fit
    _account("ChannelStatus", o, p)  # byte accounting, now including ChannelStatus
    return {"n_satellites": N}  # N = number of active channels (ChannelSatInfo sub-blocks)


def decode_monrf(p: bytes) -> list[dict] | None:  # function: MON-RF payload -> one dict per RF block
    """UBX-MON-RF (0x0A 0x38). version u1, nBlocks u1, reserved0 u1[2], then 24 each."""
    if len(p) < 4:  # if: too short for the 4-byte header
        return None  # (never happened in this data)
    ver, nb = p[0], p[1]  # message version and number of RF blocks (UBX_DOC 3.14.9)
    if ver != 0:  # if: not the documented version 0x00
        flag(f"UBX-MON-RF reports version {ver}, not the documented 0")  # recorded, decoding continues
    if 4 + nb * 24 > len(p):  # if: header + nBlocks x 24 bytes exceed the payload
        LAYOUT["MON-RF"]["subblock_mismatch"] += 1  # counted; stops the run if above 0
        return None  # (never happened in this data)
    out = []  # list of RF-block dicts
    for k in range(nb):  # for loop over RF blocks (block 0 = L1, 1 = L2 or L5)
        o = 4 + k * 24  # block k's start
        # "<BBBBIIHHBbBbB" 24 bytes: blockId flags antStatus antPower (B u1), postStatus reserved (I u4), noise agc (H),
        (bid, flags, ant, antp, post, res1, noise, agc, cw, ofsI, magI, ofsQ, magQ
         ) = struct.unpack_from("<BBBBIIHHBbBbB", p, o)  # cw (B), ofsI (b = signed i1), magI (B), ofsQ (b), magQ (B)
        # flags "& 0x03" keeps bits 1..0 = jammingState; other fields copied as read
        out.append({"block_id": bid, "jamming_state": flags & 0x03,
                    "ant_status": ant, "ant_power": antp,  # dict entry: antenna status and antenna power as read
                    "noise_per_ms": noise, "agc_cnt": agc, "cw_suppression": cw,  # dict entry: noise level, gain count and CW suppression, the three u-blox interference fields
                    "mag_i": magI, "mag_q": magQ})  # dict entry: the in-phase and quadrature signal magnitudes
    _account("MON-RF", 4 + nb * 24, p)  # byte accounting
    return out  # list of RF-block dicts


# --------------------------------------------------------------------------
# Session parsing
# --------------------------------------------------------------------------

SEPT_WANT = {4092, 4014, 4082, 4013}  # set of SBF blocks decoded: RFStatus, ReceiverStatus, QualityInd, ChannelStatus
UBX_WANT = {(0x0A, 0x38), (0x01, 0x07)}  # set of UBX (class, id): MON-RF and NAV-PVT


# function: parses every session; returns ((sept, ubx, integ), rfb, lat); the hint below is wrong but stays: it is code
def parse_sessions(sessions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, Integrity]:
    integ = Integrity()  # one integrity collector for all files
    sept_rows, ubx_rows, rfband_rows = [], [], []  # three empty lists of rows
    lat_rows = []  # receiver time against host time, per message, for the latency measurement

    for _, sr in sessions.iterrows():  # for loop over the six sessions
        folder = DAY / sr["session_folder"]  # this session's raw folder
        code = sr["test_code"]  # its test code
        to_utc = host_clock(folder)  # this session's laptop clock: monotonic ns -> UTC

        for f in sorted(folder.glob("*.sbf")):  # for loop over the three .sbf files, sorted by name
            stream = f.name[-6:-4]  # slice: the two characters before ".sbf", i.e. "01", "02" or "03"
            cur: dict[datetime, dict] = {}  # annotated dict: epoch time -> merged record from the four blocks
            rs_t, rs_end = [], []  # ReceiverStatus receiver times and end offsets, for the latency measurement
            for blk, rev, tow, wnc, p, end in sbf_blocks(f, integ, SEPT_WANT):  # for loop over the generator's results
                t = gps_to_utc(tow, wnc)  # the block's receiver time stamp in UTC, or None for Do-Not-Use values
                if t is None:  # if: no valid time stamp
                    continue  # the block cannot be placed in time; skipped
                rec = cur.setdefault(t, {})  # the record for that time, created empty if new
                if blk == 4014:  # if: ReceiverStatus, output once per epoch: one latency sample per epoch
                    rs_t.append(t); rs_end.append(end)  # two statements: its receiver time and its end offset
                if blk == 4092:  # if: RFStatus
                    d = decode_rfstatus(p)  # decode it
                    if d:  # if: decoded (a non-empty dict is truthy)
                        # dict.pop removes "rf_bands" from d and returns it; loop unpacks each band tuple
                        for fr, bw, mode, ant in d.pop("rf_bands"):
                            rfband_rows.append({  # one row per declared-interfered band, for the waterfall figure
                                "test_code": code, "receiver": f"Septentrio_{stream}",  # dict entry: the session code and which Septentrio receiver declared the band
                                "timestamp_utc": t, "frequency_hz": fr,  # dict entry: time of the declaration and the band's frequency in hertz
                                "frequency_mhz": round(fr / 1e6, 3),  # hertz / 1e6 = MHz
                                "bandwidth_khz": bw, "mode": mode, "antenna_id": ant})  # dict entry: the band's width, the receiver's mitigation mode and the antenna
                        rec.update(d)  # dict.update merges the remaining fields into the epoch record
                elif blk == 4014:  # elif: ReceiverStatus
                    d = decode_receiverstatus(p)  # decode it
                    if d:  # if: decoded
                        for a in d.pop("agc"):  # for loop over the AGCState dicts, removed from d
                            rec.setdefault("_agc", []).append(a)  # kept in a list under "_agc"
                        rec.update(d)  # the header fields merge into the record
                elif blk == 4082:  # elif: QualityInd
                    d = decode_qualityind(p)  # decode it
                    if d:  # if: decoded
                        rec.update(d)  # merge
                elif blk == 4013:  # elif: ChannelStatus
                    d = decode_channelstatus(p)  # decode it
                    if d:  # if: decoded
                        rec.update(d)  # merge
            if rs_end:  # if: any ReceiverStatus block was read
                # host time of each, from this file's sidecar
                hs = host_times(folder / "host_timing" / (f.name + ".hosttime.csv"), rs_end, to_utc)
                # one latency row per block: list comprehension over (receiver time, host time) pairs
                lat_rows += [{"test_code": code, "receiver": f"Septentrio_{stream}", "receiver_utc": t_, "host_utc": h_}
                             for t_, h_ in zip(rs_t, hs)]  # generator clause: one row per pair of receiver time and host arrival time
            for t, rec in cur.items():  # for loop over (time, record) pairs
                base = {k: v for k, v in rec.items() if not k.startswith("_")}  # dict comprehension: all but "_agc"
                for a in rec.get("_agc", []):  # one output row per front end at this time
                    sept_rows.append({  # the row: identity, front-end fields, then "**base" adds the epoch fields
                        "test_code": code, "session_folder": sr["session_folder"],  # dict entry: session code and folder
                        "receiver": f"Septentrio_{stream}", "receiver_family": "Septentrio",  # dict entry: which Septentrio receiver, and its family for pooled summaries
                        "timestamp_utc": t,  # dict entry: receiver time of the status block, true time for the Septentrio
                        "frontend_code": a["frontend_code"],  # dict entry: the front-end code the gain reading belongs to
                        # The mosaic-H is a heading receiver with a second antenna
                        # whose gain runs some 20 dB above the main one. Pooling the
                        # two gives a bimodal baseline and destroys any threshold
                        # built from it, so the antenna is part of the band label.
                        # front-end name (FRONTEND.get falls back to "code<n>"), plus " [Aux1]" for a non-main antenna
                        "band": (FRONTEND.get(a["frontend_code"], f"code{a['frontend_code']}")
                                 + ("" if a["antenna_id"] == 0  # string concatenation: adds nothing for the main antenna
                                    else f" [Aux{a['antenna_id']}]")),  # conditional expression: otherwise the auxiliary antenna's number, so the two antennas stay apart
                        "antenna_id": a["antenna_id"],  # dict entry: which antenna the reading comes from
                        "agc_gain_db": a["agc_gain_db"], "pll_unlocked": a["pll_unlocked"],  # dict entry: front-end gain in dB and whether the front-end PLL was unlocked
                        "sample_var": a["sample_var"], "blanking_pct": a["blanking_pct"],  # dict entry: sample variance and blanking share of the same front end
                        **base})  # "**base" unpacks the epoch's other status fields into this row

        for f in sorted(folder.glob("*.ubx")):  # for loop over the two .ubx files
            stream = f.name[-6:-4]  # "01" or "02"
            # File order is preserved deliberately. During the time-spoofing test
            # the receiver's own reported date jumps forward by more than a year,
            # so sorting by reported time would reorder the stream and destroy the
            # pairing below. The reported time is kept as data, not as an index.
            pvt: list[tuple[datetime | None, int]] = []  # annotated list of (reported UTC or None, numSV), file order
            monrf: list[list[dict]] = []  # annotated list: one list of RF-block dicts per MON-RF message
            pvt_end, mon_end = [], []  # end offsets of the NAV-PVT and MON-RF messages, for host time
            for cls, mid, p, end in ubx_messages(f, integ, UBX_WANT):  # for loop over wanted messages in file order
                if (cls, mid) == (0x01, 0x07) and len(p) >= 92:  # if: NAV-PVT with its full 92-byte payload
                    # "<HBBBBB" from offset 4: H year u2, then B month, day, hour, min, sec (u1 each), UTC (UBX_DOC)
                    year, month, day, hh, mm, ss = struct.unpack_from("<HBBBBB", p, 4)
                    valid, numsv = p[11], p[23]  # validity flags (offset 11) and numSV (offset 23)
                    nano = struct.unpack_from("<i", p, 16)[0]  # "<i" signed 4 bytes at offset 16: nano, ns (UBX_DOC)
                    t = None  # the reported time, unknown until checked
                    # if: "& 0x03 == 0x03": bit 0 validDate AND bit 1 validTime, and a sane year
                    if valid & 0x03 == 0x03 and 1980 < year < 2200:
                        try:  # try block: some field combinations are not real dates
                            # the UTC fields plus the nano fraction (whole seconds alone mistimed UBX_02's 5 Hz epochs)
                            t = datetime(year, month, day, hh, mm, ss, tzinfo=TZ_UTC) + timedelta(microseconds=round(nano / 1000))
                        except ValueError:  # except: an impossible date such as month 13
                            t = None  # left unknown
                    pvt.append((t, numsv))  # keeps it in file order
                    pvt_end.append(end)  # and its end offset
                elif (cls, mid) == (0x0A, 0x38):  # elif: MON-RF
                    d = decode_monrf(p)  # decode it
                    if d:  # if: decoded (non-empty list)
                        monrf.append(d)  # keeps it in file order
                        mon_end.append(end)  # and its end offset
            # MON-RF carries no time stamp of its own. It is emitted once per
            # navigation epoch alongside NAV-PVT, so the nth MON-RF pairs with the
            # nth NAV-PVT in file order. The counts are checked before pairing.
            sc_path = folder / "host_timing" / (f.name + ".hosttime.csv")  # this file's host-timing sidecar
            pvt_host = host_times(sc_path, pvt_end, to_utc)  # host time of every NAV-PVT message
            mon_host = host_times(sc_path, mon_end, to_utc)  # host time of every MON-RF message
            # latency rows for every NAV-PVT with a valid reported time, UBX_02 included (measured for all)
            lat_rows += [{"test_code": code, "receiver": f"u-blox_{stream}", "receiver_utc": t_, "host_utc": h_}
                         for (t_, _), h_ in zip(pvt, pvt_host) if t_ is not None]  # generator clause: pairs each navigation epoch with its host time, skipping epochs with no valid time
            if not monrf or not pvt:  # if: either list is empty
                continue  # nothing to pair; next file
            if abs(len(monrf) - len(pvt)) > max(3, 0.02 * len(pvt)):  # if: counts differ by more than 3 or 2 %
                # records it and keeps the stream out of the verdict tests (option b); this
                # is UBX_02, whose NAV-PVT runs at 5 Hz and MON-RF at 1 Hz. It is still used in the 2.6.12 timeline.
                # the two message rates measured from the host times, not typed
                rate_ = lambda h: (len(h) - 1) / max((h.max() - h.min()).total_seconds(), 1e-9)  # lambda: messages/s
                # records UBX_02's message counts and measured rates
                note(f"{f.name}: {len(monrf)} MON-RF messages against {len(pvt)} NAV-PVT epochs "
                     f"({rate_(pvt_host):.1f} Hz navigation, {rate_(mon_host):.1f} Hz RF monitor, host time); this "  # f-string continued: the measured navigation and RF monitor rates on host time
                     "stream stays out of the verdict tests by and is used only in the 2.6.12 timeline")
                continue  # next file
            # The receiver's own clock is under attack in at least one session, so
            # epochs are timed by the host laptop's clock, read from the host-timing
            # sidecar: each MON-RF row takes the host time at which that
            # message arrived. This replaces the earlier index interpolation across
            # the capture window, which assumed perfectly even spacing.
            m = min(len(monrf), len(pvt))  # pairs: the shorter list decides
            for k in range(m):  # for loop over the pairs
                rep_t, numsv = pvt[k]  # tuple unpacking: the k-th reported time and satellite count
                host_t = mon_host[k].to_pydatetime()  # host arrival time of the k-th MON-RF message
                # conditional expression: reported time minus the host arrival time of the same NAV-PVT, s, or None
                offset_s = ((rep_t - pvt_host[k].to_pydatetime()).total_seconds()
                            if rep_t is not None else None)  # conditional expression: no offset when the receiver reported no valid time
                for b in monrf[k]:  # for loop over the RF blocks of the k-th MON-RF
                    ubx_rows.append({  # one row per RF block per epoch
                        "test_code": code, "session_folder": sr["session_folder"],  # dict entry: session code and folder
                        "receiver": f"u-blox_{stream}", "receiver_family": "u-blox",  # dict entry: which u-blox receiver, and its family for pooled summaries
                        "timestamp_utc": host_t,  # dict entry: host arrival time, the independent clock for the u-blox
                        "receiver_reported_time_utc": rep_t,  # dict entry: the time the receiver itself reported, which the time attack moves
                        "reported_minus_host_s": offset_s,  # dict entry: reported time minus host time in seconds, the measured clock offset
                        "rf_block_id": b["block_id"],  # dict entry: which RF block of the receiver the reading comes from
                        "band": "L1" if b["block_id"] == 0 else "L2",  # blockId 0 = L1, 1 = L2 on this F9
                        "agc_cnt": b["agc_cnt"],  # dict entry: the gain count of the RF block
                        "cw_suppression": b["cw_suppression"],  # dict entry: the CW suppression level
                        "noise_per_ms": b["noise_per_ms"],  # dict entry: the noise level
                        "jamming_state": b["jamming_state"],  # dict entry: the jamming monitor's state
                        "ant_status": b["ant_status"], "ant_power": b["ant_power"],  # dict entry: antenna status and antenna power
                        "mag_i": b["mag_i"], "mag_q": b["mag_q"],  # dict entry: in-phase and quadrature signal magnitudes
                        "n_satellites": numsv})  # dict entry: satellites used in the navigation solution at that epoch

    # returns a nested tuple ((Septentrio table, u-blox table, integrity), RFStatus band table, latency rows)
    return (pd.DataFrame(sept_rows), pd.DataFrame(ubx_rows),
            integ), pd.DataFrame(rfband_rows), pd.DataFrame(lat_rows)  # tuple element: the integrity table closes the inner tuple; the band and latency tables follow


# --------------------------------------------------------------------------
# Schedule, imported from the timeline script
# --------------------------------------------------------------------------

def load_section1():  # function: loads the timeline script as a module so its plan parser is reused, not copied (as in the handset script)
    # a module "spec" for the timeline script (its name starts with a digit, so no plain import)
    spec = importlib.util.spec_from_file_location("timeline", TIMELINE_SCRIPT)
    mod = importlib.util.module_from_spec(spec)  # an empty module object
    sys.modules["timeline"] = mod  # registers it as an import would
    spec.loader.exec_module(mod)  # runs the timeline script's top-level code; its main() is not run
    return mod  # the loaded module


def parse_own_jamming(catalog) -> dict:  # function: our motorcade's own jamming, parsed from the catalogue
    """
    The group 2.6 description states the jamming our own motorcade applied.
    Parsed from the catalogue text, and the tests that exempt themselves are
    identified from their own descriptions rather than assumed.
    """
    cat = catalog["2.6.2"]  # any group 2.6 test carries the group description; 2.6.2 is used
    grp = re.sub(r"\s+", " ", cat["group_description"])  # whitespace runs collapsed to one space
    # regex matching the catalogue's own spelling "simultanious"; groups: the minutes and the band list
    m_init = re.search(r"During the first (\d+) minutes there will be simultanious "
                       r"spoofing and jamming on ([^.]+)", grp, re.I)  # string continued: the minutes and the jammed band list are captured, ignoring letter case
    m_stop = re.search(r"After (\d+) minutes, jamming will stop", grp, re.I)  # the sentence ending the jamming
    m_cont = re.search(r"There will be continuous jamming on ([^.]+)", grp, re.I)  # the always-on bands
    if not (m_init and m_stop and m_cont):  # if: any of the three sentences is missing
        # stops rather than assess against an assumed structure
        die("the group 2.6 description does not state its jamming structure in the "
            "expected form; the own-jamming assessment cannot be fixed in advance")
    initial_min = int(m_init.group(1))  # the opening jamming duration (5)
    # band names split at commas or the whole word "and" (\b marks word boundaries, so "and" inside a word cannot
    # split it); the catalogue's misspelling "simultanious" is matched above on purpose; trimmed
    init_bands = [b.strip() for b in re.split(r",|\band\b", m_init.group(2)) if b.strip()]
    cont_bands = [b.strip() for b in re.split(r",|\band\b", m_cont.group(1)) if b.strip()]  # the continuous bands
    exempt = {}  # dict: test code -> the sentence exempting it
    for code, c in catalog.items():  # for loop over every catalogue test
        if not code.startswith("2.6."):  # if: not in group 2.6
            continue  # skip it
        # for loop over the test's sentences (split after . ! or ?, the lookbehind keeps the mark)
        for s in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", c["test_description"])):
            if re.search(r"\bno initial jamming\b|\bwithout initial jamming\b", s, re.I):  # if: it exempts itself
                exempt[code] = s.strip()  # records the sentence
                break  # one is enough
    # returns the parsed structure and the three source sentences, so the output can quote them
    return {"initial_min": initial_min, "initial_bands": init_bands,
            "continuous_bands": cont_bands, "exempt": exempt,  # dict entry: the bands jammed throughout, and whether this test is exempt from the initial jamming
            "group_sentence_initial": m_init.group(0), "group_sentence_stop": m_stop.group(0),  # dict entry: the catalogue sentences that state the initial jamming and when it stops
            "group_sentence_continuous": m_cont.group(0)}  # dict entry: the sentence that states the continuous jamming


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

# dict: receiver tag -> name for figures, so tags such as "Septentrio_02" never print as "Sept 02"
RX_NAME = {"Septentrio_01": "mosaic-G5", "Septentrio_02": "mosaic-X5", "Septentrio_03": "mosaic-H",
           "u-blox_01": "u-blox 01", "u-blox_02": "u-blox 02"}  # dict entry: display names of the two u-blox receivers
# dict: field -> axis label in plain words (no raw column names on axes)
FIELD_LABEL = {"agc_gain_db": "AGC gain (dB)", "agc_cnt": "AGC count (0-8191)",
               "cw_suppression": "CW suppression (0-255)", "noise_per_ms": "noise level (counts)"}  # dict entry: axis labels for the CW suppression and noise fields


def plot_gapped(ax, times, values, **kw) -> None:  # function: one trace with lines broken where data are missing
    x, y = break_gaps(times, values, max_gap_s=5.0)  # shared helper: NaN inserted at every gap over 5 s
    ax.plot(x, y, **kw)  # "**kw" passes the caller's style arguments (lw, color, label, zorder) through


def shade_50w(ax, plan, t0, t1) -> None:  # function: shades the scheduled 50 W Test Area 1 windows
    for p in plan:  # for loop over the schedule
        if p["power_w"] is None or p["power_w"] < FIFTY_W_MIN or p["test_area"] != 1:  # if: not a 50 W TA1 test
            continue  # skip it
        s, e = pd.Timestamp(p["start_utc"]), pd.Timestamp(p["end_utc"])  # its window
        if e <= t0 or s >= t1:  # if: outside the plotted range
            continue  # skip it
        # vertical band clipped to the plotted range, behind the data (zorder 0)
        ax.axvspan(mdates.date2num(max(s, t0)), mdates.date2num(min(e, t1)),
                   color=C_VERMILLION, alpha=0.18, zorder=0, lw=0)  # keyword arguments of axvspan(): pale vermillion band behind the data, without an outline


def shade_ours(ax, sessions, t0, t1, color=C_GREEN) -> None:  # function: shades our tests' scheduled windows
    for _, r in sessions.iterrows():  # for loop over sessions
        s = pd.Timestamp(r["scheduled_start_utc"], tz="UTC")  # scheduled start
        e = pd.Timestamp(r["scheduled_end_utc"], tz="UTC")  # scheduled end
        if e <= t0 or s >= t1:  # if: outside the plotted range
            continue  # skip it
        # pale band in the given colour, behind the data
        ax.axvspan(mdates.date2num(max(s, t0)), mdates.date2num(min(e, t1)),
                   color=color, alpha=0.14, zorder=0, lw=0)  # keyword arguments of axvspan(): pale band in the given colour behind the data, without an outline


def fig_status_timeseries(sept, ubx, plan, sessions, dist_txt, path):  # function: Figure 9, six stacked traces
    # list of (receiver, band, field, y label) tuples for the four Septentrio panels
    order = [("Septentrio_02", "GPSL1/E1", "agc_gain_db", "AGC gain (dB)"),
             ("Septentrio_02", "GLOL1", "agc_gain_db", "AGC gain (dB)"),  # list element: the mosaic-X5 gain on the GLONASS G1 front end
             ("Septentrio_02", "L5/E5a/B2a", "agc_gain_db", "AGC gain (dB)"),  # list element: the mosaic-X5 gain on the L5 front end
             ("Septentrio_03", "GPSL1/E1", "agc_gain_db", "AGC gain (dB)")]  # list element: the mosaic-H gain on the L1 front end
    t0 = min(sept.timestamp_utc.min(), ubx.timestamp_utc.min()) - pd.Timedelta(minutes=5)  # axis start
    t1 = max(sept.timestamp_utc.max(), ubx.timestamp_utc.max()) + pd.Timedelta(minutes=5)  # axis end
    fig, axes = plt.subplots(6, 1, figsize=(15, 15), sharex=True)  # six panels in one column, shared x
    for k, (rx, band, fld, lab) in enumerate(order):  # enumerate with tuple unpacking: panel index and spec
        ax = axes[k]  # this panel
        shade_50w(ax, plan, t0, t1); shade_ours(ax, sessions, t0, t1)  # two statements on one line: both shadings
        s = sept[(sept.receiver == rx) & (sept.band == band)].sort_values("timestamp_utc")  # this receiver and band
        for _, g in s.groupby("test_code"):  # one line per session, so gaps between sessions are not bridged
            plot_gapped(ax, g.timestamp_utc, g[fld], lw=0.6, color=C_BLUE, zorder=3)  # the trace, gaps kept as gaps
        ax.set_ylabel(f"{RX_NAME[rx]}\n{band}\n{lab}", fontsize=8)  # three-line y label with the receiver's name
        time_axis(ax, t0, t1); ax.tick_params(labelsize=8)  # two statements: axis format, small tick labels
    # for loop over the two u-blox panels: (field, label, colour), numbered by enumerate
    for k, (fld, lab, col) in enumerate([("agc_cnt", f"u-blox 01, L1\n{FIELD_LABEL['agc_cnt']}", C_ORANGE),
                                         ("cw_suppression", f"u-blox 01, L1\n{FIELD_LABEL['cw_suppression']}", C_PURPLE)]):  # list element: the second u-blox panel, its CW suppression level in purple
        ax = axes[4 + k]  # panels 5 and 6
        shade_50w(ax, plan, t0, t1); shade_ours(ax, sessions, t0, t1)  # both shadings
        u = ubx[(ubx.receiver == "u-blox_01") & (ubx.band == "L1")].sort_values("timestamp_utc")  # UBX_01 L1 only
        for _, g in u.groupby("test_code"):  # one line per session
            plot_gapped(ax, g.timestamp_utc, g[fld], lw=0.6, color=col, zorder=3)  # the trace, gaps kept as gaps
        ax.set_ylabel(lab, fontsize=8)  # y label
        time_axis(ax, t0, t1); ax.tick_params(labelsize=8)  # axis format and tick size
    axes[-1].set_xlabel("UTC on 14 September 2026")  # x label on the bottom panel
    fig.legend(handles=[  # figure legend from two proxy patches; the distance range is measured
        Patch(facecolor=C_VERMILLION, alpha=0.35, label=f"Scheduled 50 W jamming at Test Area 1 ({dist_txt})"),  # list element: legend patch for the 50 W jamming, with its measured distance range
        Patch(facecolor=C_GREEN, alpha=0.30, label="Scheduled window of our own spoofing (about 1 mW, on site)")],  # list element: legend patch for the windows of the on-site spoofing tests
        loc="lower center", bbox_to_anchor=(0.5, -0.015), ncol=2, frameon=False, fontsize=10)  # keyword arguments of legend(): placed below the panels in two columns without a frame
    fig.suptitle("Primary receiver interference status against time", fontsize=13, y=0.997)  # title
    fig.tight_layout(rect=(0, 0.015, 1, 0.985))  # packs the panels
    save_fig(fig, path)  # writes the figure


def fig_control(sept, ubx, plan, sessions, code, path):  # function: the figure, positive control for one test
    sr = sessions[sessions.test_code == code].iloc[0]  # that test's inventory row
    # two statements on one line: capture start and end as UTC timestamps
    cs = pd.Timestamp(sr["capture_start_utc"], tz="UTC"); ce = pd.Timestamp(sr["capture_end_utc"], tz="UTC")
    t0, t1 = cs - pd.Timedelta(minutes=2), ce + pd.Timedelta(minutes=2)  # plotted range: capture +/- 2 min
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)  # three stacked panels
    ax = axes[0]  # top panel: Septentrio gain
    shade_50w(ax, plan, t0, t1); shade_ours(ax, sessions[sessions.test_code == code], t0, t1)  # both shadings
    cols = {"GPSL1/E1": C_BLUE, "GLOL1": C_ORANGE, "L5/E5a/B2a": C_GREEN, "B1": C_PURPLE}  # dict: band -> colour
    s = sept[(sept.receiver == "Septentrio_02") & (sept.test_code == code)]  # the mosaic-X5 in this session
    for band, c in cols.items():  # for loop over the four bands
        g = s[s.band == band].sort_values("timestamp_utc")  # this band in time order
        if not g.empty:  # if: data present
            plot_gapped(ax, g.timestamp_utc, g.agc_gain_db, lw=0.9, color=c, label=band, zorder=3)  # trace
    ax.set_ylabel("Septentrio mosaic-X5\nAGC gain (dB)", fontsize=9)  # y label
    ax.legend(loc="lower left", fontsize=8, ncol=4, frameon=False)  # band legend
    ax = axes[1]  # middle panel: u-blox AGC
    shade_50w(ax, plan, t0, t1); shade_ours(ax, sessions[sessions.test_code == code], t0, t1)  # both shadings
    u = ubx[(ubx.receiver == "u-blox_01") & (ubx.test_code == code)]  # UBX_01 in this session
    for band, c in (("L1", C_BLUE), ("L2", C_GREEN)):  # for loop over the two RF blocks (L1 and L2)
        g = u[u.band == band].sort_values("timestamp_utc")  # this block in time order
        if not g.empty:  # if: data present
            plot_gapped(ax, g.timestamp_utc, g.agc_cnt, lw=0.9, color=c, label=band, zorder=3)  # trace
    ax.set_ylabel(f"u-blox 01\n{FIELD_LABEL['agc_cnt']}", fontsize=9)  # y label in plain words
    ax.legend(loc="lower left", fontsize=8, ncol=2, frameon=False)  # legend
    ax = axes[2]  # bottom panel: the receiver clock
    shade_50w(ax, plan, t0, t1); shade_ours(ax, sessions[sessions.test_code == code], t0, t1)  # both shadings
    g = u[u.band == "L1"].sort_values("timestamp_utc")  # one row per epoch (the L1 block)
    if not g.empty and g.reported_minus_host_s.notna().any():  # if: any reported-time offset exists
        # NAV-PVT reported time minus host time, divided by 86400 s to show days
        plot_gapped(ax, g.timestamp_utc, g.reported_minus_host_s / 86400.0, lw=1.1, color=C_VERMILLION, zorder=3)
    ax.set_ylabel("u-blox 01 reported time\nminus host time (days)", fontsize=9)  # y label
    ax.axhline(0, color=C_GREY, lw=0.6)  # zero line
    for a in axes:  # for loop over the three panels
        # three statements: time axis, faint y grid, small ticks
        time_axis(a, t0, t1); a.grid(axis="y", color=C_GREY, alpha=0.2, lw=0.4); a.tick_params(labelsize=8)
    axes[-1].set_xlabel("UTC on 14 September 2026")  # x label
    fig.legend(handles=[  # figure legend
        Patch(facecolor=C_GREEN, alpha=0.30, label=f"Scheduled window of our own test {code}"),  # list element: legend patch for the scheduled window of the session's own test
        Patch(facecolor=C_VERMILLION, alpha=0.35, label="Scheduled 50 W jamming at Test Area 1")],  # list element: legend patch for the 50 W jamming at Test Area 1
        loc="lower center", bbox_to_anchor=(0.5, -0.03), ncol=2, frameon=False, fontsize=9.5)  # keyword arguments of legend(): placed below the panels in two columns without a frame
    fig.suptitle(f"Positive control on the primary receivers: test {code}", fontsize=13, y=0.997)  # title
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))  # packs the panels
    save_fig(fig, path)  # writes the figure


def fig_waterfall(rfb, pyr, plan, sessions, path):  # function: the figure, every RFStatus band against time
    fig, ax = plt.subplots(figsize=(15, 7))  # one set of axes
    t0 = rfb.timestamp_utc.min() - pd.Timedelta(minutes=5)  # axis start
    t1 = rfb.timestamp_utc.max() + pd.Timedelta(minutes=5)  # axis end
    shade_50w(ax, plan, t0, t1)  # 50 W windows behind
    s = rfb[rfb.receiver == "Septentrio_02"]  # the mosaic-X5's declared bands only
    # scatter: one dot per (time, declared frequency in MHz)
    ax.scatter(mdates.date2num(s.timestamp_utc), s.frequency_mhz, s=1.2,
               c=C_BLUE, alpha=0.5, linewidths=0, zorder=3)  # keyword arguments of scatter(): blue, half transparent, no outline, above the reference lines
    # for loop over reference frequencies (MHz) with labels; B1I at its true centre 1561.098 MHz
    for mhz, lab in [(1575.42, "L1/E1"), (1602.0, "G1"), (1561.098, "B1"),
                     (1227.6, "L2"), (1176.45, "L5"), (1207.14, "E5b"), (1278.75, "E6")]:  # list element: the L2, L5, E5b and E6 reference frequencies with their labels
        ax.axhline(mhz, color=C_GREY, lw=0.5, ls=":", zorder=1)  # dotted horizontal guide
        ax.text(mdates.date2num(t1), mhz, " " + lab, fontsize=7.5, va="center", color="#444444")  # label at right
    for st in pyr["steps"]:  # for loop over the pyramid steps
        # a thin vertical line at each step start
        ax.axvline(mdates.date2num(pd.Timestamp(st["step_start_utc"])),
                   color=C_VERMILLION, lw=0.5, alpha=0.6, zorder=2)  # keyword arguments of axvline(): thin, pale vermillion line under the points
    for _, r in sessions.iterrows():  # for loop over our sessions
        # pale green band over each scheduled window
        ax.axvspan(mdates.date2num(pd.Timestamp(r["scheduled_start_utc"], tz="UTC")),
                   mdates.date2num(pd.Timestamp(r["scheduled_end_utc"], tz="UTC")),  # argument of axvspan(): the end of the scheduled window
                   color=C_GREEN, alpha=0.12, zorder=0, lw=0)  # keyword arguments of axvspan(): pale green band behind everything, without an outline
    ax.set_ylabel("Frequency declared interfered (MHz)")  # y label
    ax.set_xlabel("UTC on 14 September 2026")  # x label
    time_axis(ax, t0, t1)  # time-axis format
    ax.legend(handles=[  # legend of four proxy artists
        Patch(facecolor=C_VERMILLION, alpha=0.35, label="Scheduled 50 W jamming at Test Area 1"),  # list element: legend patch for the 50 W jamming at Test Area 1
        Patch(facecolor=C_GREEN, alpha=0.3, label="Scheduled window of our own spoofing"),  # list element: legend patch for the windows of the on-site spoofing tests
        plt.Line2D([], [], color=C_VERMILLION, lw=0.8, label="Pyramid step boundaries"),  # list element: legend line for the pyramid's step boundaries
        plt.Line2D([], [], marker="o", ls="none", color=C_BLUE, ms=4,  # list element: legend marker for a band the receiver declared interfered
                   label="RFStatus band declared interfered")],  # keyword argument label=: the text of that legend marker
        loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, frameon=False, fontsize=9)  # keyword arguments of legend(): placed below the axes in two columns without a frame
    # the axes title
    ax.set_title("Every frequency the Septentrio mosaic-X5 declared interfered, against time",
                 fontsize=12)  # keyword argument fontsize=: the title's font size
    save_fig(fig, path)  # writes the figure


# set of receivers left out of the interval metric: the mosaic-G5 (its logging gaps would split spans)
EXCLUDE_FROM_INTERVAL = {"Septentrio_01"}   # see the note printed in section [10]
# tuple assignment (seconds): smoothing window, largest gap bridged between runs, shortest run kept
SMOOTH_S, BRIDGE_S, MIN_SPAN_S = 15, 30, 30
# float: exit share of the entry drop (5 dB x 0.4 = 2 dB); never changed: main() passes other shares as exit_fraction
EXIT_FRACTION = 0.4   # a run is entered at `drop` and extended while above 0.4*drop
# dict: field -> entry drop below the quiet reference, in that field's own units
DROP_THRESHOLD = {"agc_gain_db": 5.0, "agc_cnt": 400.0, "noise_per_ms": 20.0}


# function: time spans where the smoothed trace departs from the reference in the expected direction
def disturbed_runs(series: pd.Series, ref: float, drop: float, direction: int, exit_fraction: float = EXIT_FRACTION):
    """
    Contiguous intervals in which a smoothed trace departs from the reference in
    the direction fixed in advance.

    Two thresholds are used. A run is ENTERED only where the departure exceeds
    `drop`, which keeps noise from opening one, and is then EXTENDED outwards
    while the departure exceeds `exit_fraction` of it (argument; module default
    EXIT_FRACTION). A single threshold cuts the shoulders off a smooth excursion,
    reporting only its deepest part as disturbed. Brief re-crossings are then
    bridged and a minimum span applied. The smoothing median is trailing, so a
    run's edges lag the trace by up to the smoothing window.
    """
    g = series.dropna()  # the trace without missing values (index = timestamps)
    if g.empty:  # if: nothing left
        return []  # no runs
    sm = g.rolling(f"{SMOOTH_S}s").median()  # time-based rolling median over the previous 15 s (trailing)
    # conditional expression: departure measured so positive always means "in the expected direction"
    dev = (ref - sm) if direction < 0 else (sm - ref)
    core = (dev > drop).to_numpy()  # boolean array: beyond the entry threshold
    halo = (dev > drop * exit_fraction).to_numpy()  # boolean array: beyond the lower exit threshold
    t = g.index.to_numpy()  # the timestamps as an array
    if not core.any():  # if: nothing crosses the entry threshold
        return []  # no runs
    runs, i = [], 0  # list of [start, end] runs, and a position
    while i < len(core):  # while loop over positions
        if core[i]:  # if: an entry point
            lo = i  # run start
            while lo > 0 and halo[lo - 1]:  # extends backwards while the previous point is above the exit threshold
                lo -= 1  # one step back
            hi = i  # run end
            while hi < len(core) - 1 and halo[hi + 1]:  # extends forwards likewise
                hi += 1  # one step on
            runs.append([t[lo], t[hi]])  # records the run's first and last timestamps
            i = hi + 1  # continues after it
        else:  # else: not an entry point
            i += 1  # next position
    merged = []  # runs after bridging short gaps
    for r in runs:  # for loop over runs in time order
        # if: this run starts within 30 s of the previous merged run's end
        if merged and (pd.Timestamp(r[0])
                       - pd.Timestamp(merged[-1][1])).total_seconds() <= BRIDGE_S:  # arithmetic continued: the gap from the last merged run's end, compared with the bridging limit
            merged[-1][1] = max(r[1], merged[-1][1])  # bridges them: extends the previous end
        else:  # else: a separate run
            merged.append(r)  # keeps it
    # list comprehension: only runs lasting at least 30 s
    return [r for r in merged
            if (pd.Timestamp(r[1]) - pd.Timestamp(r[0])).total_seconds() >= MIN_SPAN_S]  # generator clause: keeps runs lasting at least the minimum span


def margin_slices(g: pd.Series, cs, ss, se, ce) -> pd.Series:  # function: the pre- and post-test margins joined
    # pd.concat stacks the two slices: capture start to scheduled start, and scheduled end to capture end
    return pd.concat([g[(g.index >= cs) & (g.index < ss)],
                      g[(g.index > se) & (g.index <= ce)]])  # list element: the slice from the scheduled end to the capture end, the second margin


# function: the figure, each session's X5 GPS L1/E1 gain with detector thresholds and spans (refs unused inside)
def fig_excursions(traces: dict, sessions: pd.DataFrame, refs: dict, path: Path) -> None:
    """All six sessions stacked on a common axis, trace against detector."""
    order = list(sessions.test_code)  # test codes in session order
    fig, axes = plt.subplots(len(order), 1, figsize=(13, 2.15 * len(order)))  # one panel per session
    for k, code in enumerate(order):  # for loop over sessions
        ax = axes[k]  # this panel
        rec = traces.get(code)  # this session's stored trace and detector result, or None
        sr = sessions[sessions.test_code == code].iloc[0]  # inventory row
        ss = pd.Timestamp(sr["scheduled_start_utc"], tz="UTC")  # scheduled start
        se = pd.Timestamp(sr["scheduled_end_utc"], tz="UTC")  # scheduled end
        cs = pd.Timestamp(sr["capture_start_utc"], tz="UTC")  # capture start
        ce = pd.Timestamp(sr["capture_end_utc"], tz="UTC")  # capture end
        t0, t1 = cs - pd.Timedelta(minutes=1), ce + pd.Timedelta(minutes=1)  # plotted range
        # pale green band over the scheduled window
        ax.axvspan(mdates.date2num(max(ss, t0)), mdates.date2num(min(se, t1)),
                   color=C_GREEN, alpha=0.15, zorder=0, lw=0)  # keyword arguments of axvspan(): pale green band behind the data, without an outline
        if rec is not None and not rec["trace"].empty:  # if: a trace exists
            g = rec["trace"]  # the gain series
            ref = rec["ref"]  # the quiet reference
            ax.axhline(ref, color=C_GREY, lw=0.8, ls="--", zorder=2)  # dashed line: quiet reference
            # dotted line: entry threshold, 5 dB below
            ax.axhline(ref - DROP_THRESHOLD["agc_gain_db"], color=C_VERMILLION,
                       lw=0.8, ls=":", zorder=2)  # keyword arguments of axhline(): thin dotted line under the trace
            # dotted line: exit threshold, 5 x 0.4 = 2 dB below
            ax.axhline(ref - DROP_THRESHOLD["agc_gain_db"] * EXIT_FRACTION,
                       color=C_ORANGE, lw=0.8, ls=":", zorder=2)  # keyword arguments of axhline(): thin dotted orange line under the trace
            plot_gapped(ax, g.index, g.to_numpy(), lw=0.7, color=C_BLUE, zorder=3)  # the trace, gaps kept as gaps
            for a, b in rec["runs"]:  # for loop over detected spans (whole capture)
                # red band over each span
                ax.axvspan(mdates.date2num(pd.Timestamp(a)), mdates.date2num(pd.Timestamp(b)),
                           color=C_VERMILLION, alpha=0.22, zorder=1, lw=0)  # keyword arguments of axvspan(): vermillion band behind the trace, without an outline
            ax.set_ylim(-16, 46)  # fixed y range for comparability
            # a text line at the panel's bottom-left (axes coordinates) with the summary numbers
            ax.text(0.005, 0.06,
                    f"excursion {rec['excursion']:.0f} dB   {rec['n_passages']} passage(s)"  # argument of text(): the summary text, starting with the excursion in dB and the passage count
                    f"   longest {rec['longest']:.1f} min"  # f-string continued: the longest passage in minutes
                    f"   captured {rec['coverage']:.0f} % of window",  # f-string continued: the share of the scheduled window that was captured
                    transform=ax.transAxes, fontsize=7.5, color="#333333")  # keyword arguments of text(): positioned in axes coordinates, in small dark grey type
        else:  # else: no trace
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", fontsize=9)  # says so
        ax.set_ylabel(f"{code}\nmosaic-X5 L1\nAGC gain (dB)", fontsize=8.5)  # y label naming the receiver
        time_axis(ax, t0, t1)  # time-axis format
        ax.grid(axis="y", color=C_GREY, alpha=0.2, lw=0.4)  # faint y grid
        ax.tick_params(labelsize=7.5)  # small ticks
    axes[-1].set_xlabel("UTC on 14 September 2026")  # x label
    fig.legend(handles=[  # figure legend of six proxy artists (threshold values typed in the labels)
        Patch(facecolor=C_GREEN, alpha=0.3, label="Scheduled test window"),  # list element: legend patch for the scheduled test window
        Patch(facecolor=C_VERMILLION, alpha=0.4, label="Detected disturbed span"),  # list element: legend patch for a span the detector found disturbed
        plt.Line2D([], [], color=C_GREY, ls="--", label="Quiet reference (median of the per-session margin medians)"),  # list element: legend line for the quiet reference level
        plt.Line2D([], [], color=C_VERMILLION, ls=":", label="Entry threshold (5 dB below reference)"),  # list element: legend line for the entry threshold, 5 dB below the reference
        plt.Line2D([], [], color=C_ORANGE, ls=":", label="Exit threshold (2 dB below reference)"),  # list element: legend line for the exit threshold, 2 dB below the reference
        plt.Line2D([], [], color=C_BLUE, label="Septentrio mosaic-X5, GPS L1/E1 front-end gain")],  # list element: legend line for the mosaic-X5 L1 gain trace
        loc="lower center", bbox_to_anchor=(0.5, -0.012), ncol=2, frameon=False, fontsize=8.5)  # keyword arguments of legend(): placed below the panels in two columns without a frame
    # figure title (two joined strings)
    fig.suptitle("Front-end gain across all six sessions, with the detector shown against "
                 "the trace", fontsize=12.5, y=0.998)  # string continued: the end of the title, with its font size and height
    fig.tight_layout(rect=(0, 0.035, 1, 0.985))  # packs the panels
    save_fig(fig, path)  # writes the figure


# --------------------------------------------------------------------------
# analyses: frozen-value check, clean margins, site confound,
# schedule shift, the 1590 MHz interferer, 2.6.12 timeline
# --------------------------------------------------------------------------

def frozen_check(sept: pd.DataFrame, ubx: pd.DataFrame) -> pd.DataFrame:  # function: the mechanism first
    """
    Runs of an unchanged gain value lasting 30 s or more, per receiver and band, set against what the receiver
    reported during them. A held register (the handset finding of the handset script) would show as long runs coinciding with no
    tracked satellites or an unlocked front end. Nothing is removed here; the table reports what is found.
    """
    rows = []  # one row per receiver, band and field
    # list of (table, field) pairs: Septentrio gain in dB, u-blox AGC count
    for tab, fld in ((sept, "agc_gain_db"), (ubx[ubx.receiver == "u-blox_01"], "agc_cnt")):
        x = tab[tab[fld].notna()].sort_values(["receiver", "band", "test_code", "timestamp_utc"]).copy()  # in order
        key = ["receiver", "band", "test_code"]  # a run never crosses a band or a capture
        gap = x.timestamp_utc.diff().dt.total_seconds() > 5  # boolean: more than 5 s since the previous reading
        # a new run starts where the value, the band, the capture changes, or after a gap
        new = (x[fld] != x[fld].shift()) | (x[key] != x[key].shift()).any(axis=1) | gap
        x["run"] = new.cumsum()  # cumulative sum of the starts: a run number per reading
        span = x.groupby("run").timestamp_utc.agg(["min", "max"])  # each run's first and last time
        dur = (span["max"] - span["min"]).dt.total_seconds()  # its duration, s
        x["run_s"] = x.run.map(dur)  # Series.map: each reading gets its run's duration
        for (rx, band), g in x.groupby(["receiver", "band"]):  # per receiver and band
            lr = g[g.run_s >= 30]  # readings inside runs of 30 s or more
            el = g[g.run_s < 30]  # the rest
            d_ = g.groupby(["test_code"])[fld].diff().abs()  # one-step changes within a capture
            k_ = g.run_s.idxmax()  # index of a reading in the longest run
            rows.append({  # appends the summary row
                "receiver": rx, "band": band, "field": fld, "n": len(g),  # dict entry: receiver, band, field and the number of readings
                # dict entries: distinct values, typical step between them, and the series' floor and ceiling
                "n_distinct": int(g[fld].nunique()), "step": float(d_[d_ > 0].median()) if (d_ > 0).any() else None,
                "floor": float(g[fld].min()), "ceiling": float(g[fld].max()),  # dict entry: the lowest and highest value the field takes
                "share_in_runs_30s": round(len(lr) / len(g), 4),  # share of readings in long runs
                "longest_run_s": round(float(g.run_s.max()), 1), "longest_run_value": float(g.loc[k_, fld]),  # dict entry: length of the longest run of one repeated value, and that value
                # dict entry: when the longest run began, as text
                "longest_run_start_utc": pd.Timestamp(g[g.run == g.loc[k_, "run"]].timestamp_utc.min()).strftime(
                    "%Y-%m-%d %H:%M:%S"),
                # shares inside long runs: at the floor, with no tracked satellites, with the front-end PLL unlocked
                "share_runs_at_floor": round(float((lr[fld] == g[fld].min()).mean()), 4) if len(lr) else None,
                "share_runs_no_satellites": round(float((lr.n_satellites == 0).mean()), 4) if len(lr) else None,  # dict entry: share of long-run readings taken with no satellite tracked
                "share_elsewhere_no_satellites": round(float((el.n_satellites == 0).mean()), 4) if len(el) else None,  # dict entry: the same share for readings outside long runs, for comparison
                # conditional expression: the PLL flag exists for Septentrio only
                "share_runs_pll_unlocked": (round(float(lr.pll_unlocked.astype(bool).mean()), 4)
                                            if "pll_unlocked" in lr.columns and len(lr) else None)})  # conditional expression: left empty for the u-blox, which has no PLL flag
    out = pd.DataFrame(rows)  # the table
    # held-register signature: long runs in which most readings have no tracked satellite or an unlocked front end
    out["held_register_signature"] = ((out.share_runs_no_satellites.fillna(0) > 0.5)
                                      | (out.share_runs_pll_unlocked.fillna(0) > 0.5))  # element-wise condition: or most long-run readings come with the front-end PLL unlocked
    return out  # one row per receiver, band and field


# function: Test A - 50 W against baseline per receiver, band and field (never pooled); used by the verdict and
# by the +/-2 h shift of the schedule confirmation, so both run the identical computation
def group_comparison(long: pd.DataFrame) -> pd.DataFrame:
    comp = []  # comparison rows
    # groupby over three keys, tuple-unpacked: one group per receiver, band and field (never pooled)
    for (rx, band, fld), g in long.groupby(["receiver", "band", "field"]):
        a = g.loc[g.exposure_group == G50, "value"].to_numpy()  # values while a 50 W TA1 test was scheduled
        b = g.loc[g.exposure_group == GBL, "value"].to_numpy()  # baseline values
        if len(a) < 200 or len(b) < 200:  # if: fewer than 200 of either
            continue  # skip this group
        fam = "Septentrio" if rx.startswith("Sept") else "u-blox"  # receiver family from the name
        d = DIRECTION.get((fam, fld), 0)  # direction fixed in advance (0 if none)
        diff = float(np.median(a) - np.median(b))  # median 50 W minus median baseline (signed)
        comp.append({"receiver": rx, "band": band, "field": fld,  # appends the row
                     "n_50w": len(a), "n_baseline": len(b),  # dict entry: readings during the 50 W test and in the baseline
                     "median_50w": round(float(np.median(a)), 3),  # dict entry: median during the 50 W test
                     "median_baseline": round(float(np.median(b)), 3),  # dict entry: median in the baseline
                     "difference": round(diff, 3),  # dict entry: 50 W median minus baseline median
                     "cliffs_delta": round(cliffs_delta(a, b), 4),  # negative = 50 W values tend lower
                     "expected_direction": {-1: "decrease", 1: "increase", 0: "none"}[d],  # dict lookup -> word
                     # True only when the signed difference agrees with the direction fixed in advance
                     "matches_prediction": bool((d < 0 and diff < 0) or (d > 0 and diff > 0))})
    return pd.DataFrame(comp)  # the comparison table


def margin_parts(sr) -> list:  # function: a session's pre-test and post-test margins as (start, end) pairs
    cs = pd.Timestamp(sr["capture_start_utc"], tz="UTC")  # capture start
    ce = pd.Timestamp(sr["capture_end_utc"], tz="UTC")  # capture end
    ss = pd.Timestamp(sr["scheduled_start_utc"], tz="UTC")  # scheduled start
    se = pd.Timestamp(sr["scheduled_end_utc"], tz="UTC")  # scheduled end
    return [(cs, min(ss, ce)), (max(se, cs), ce)]  # list of two (start, end) pairs; an empty one has end <= start


# function: the margin cut; split = (test code, time): that session's pre-test margin is cut at the time and
# the part before it flagged as possibly contaminated (the 1.8.2 remnant of the possibly contaminated 2.6.12 margin)
def margin_clean(long: pd.DataFrame, sessions: pd.DataFrame, split=None) -> pd.DataFrame:
    """
    Clean margin minutes per indicator and combined, on a 0.1 s grid with each sample held until the next one
    (at most 60 s). Gain (mosaic-X5 and mosaic-H main antenna, L1/G1/B1) is contaminated below its quiet reference
    minus 3 dB; u-blox_01 L1 noise and CW suppression above reference plus 3 margin IQRs (noise also at twice its
    reference, reported but not in the combination). Reference = median of the per-session margin medians.
    """
    ind = ([("Septentrio_02", b, "agc_gain_db", "gain") for b in ("GPSL1/E1", "GLOL1", "B1")]  # mosaic-X5 gain
           + [("Septentrio_03", b, "agc_gain_db", "gain") for b in ("GPSL1/E1", "GLOL1", "B1")]  # mosaic-H gain
           # u-blox indicators: noise with two rules, CW suppression with one
           + [("u-blox_01", "L1", "noise_per_ms", "iqr"), ("u-blox_01", "L1", "noise_per_ms", "x2"),
              ("u-blox_01", "L1", "cw_suppression", "iqr")])  # list element: the CW suppression level of u-blox 01, judged by the interquartile rule
    ser, refs = {}, {}  # indicator -> its series; indicator -> (reference, IQR)
    for rx, band, fld, rule in ind:  # for loop over indicators
        # boolean masks on receiver, band and field: this indicator's readings, in time order
        g = long[(long.receiver == rx) & (long.band == band) & (long.field == fld)].sort_values("timestamp_utc")
        ser[(rx, band, fld)] = g  # kept
        meds, pooled = [], []  # per-session margin medians, pooled margin values
        for _, sr in sessions.iterrows():  # for loop over sessions
            # list comprehension: the readings inside each non-empty margin of this session
            parts = [g[(g.timestamp_utc >= a) & (g.timestamp_utc < b)] for a, b in margin_parts(sr) if b > a]
            m = pd.concat(parts) if parts else g.iloc[0:0]  # the session's margin readings
            if len(m):  # if: any
                meds.append(m.value.median()); pooled.append(m.value)  # two statements: median and values
        allm = pd.concat(pooled) if pooled else pd.Series(dtype=float)  # every margin reading of this indicator
        q = float(allm.quantile(0.75) - allm.quantile(0.25)) if len(allm) else 0.0  # pooled IQR
        refs[(rx, band, fld)] = (float(np.median(meds)), q if q > 0 else 1.0)  # (reference, IQR; 0 -> 1)
    rows = []  # one row per session, margin part and indicator, plus the combination
    for _, sr in sessions.iterrows():  # for loop over sessions
        parts = margin_parts(sr)  # its two margins
        if split is not None and sr["test_code"] == split[0] and parts[0][0] < split[1] < parts[0][1]:  # if: cut here
            parts = [(parts[0][0], split[1]), (split[1], parts[0][1]), parts[1]]  # the pre-test margin in two parts
        for a, b in parts:  # for loop over its margins
            if b <= a:  # if: this margin is empty
                continue  # skip
            flg = split is not None and sr["test_code"] == split[0] and b == split[1]  # boolean: the flagged part
            grid = pd.date_range(a, b, freq="100ms", inclusive="left")  # the 0.1 s grid over this margin
            bad_any = np.zeros(len(grid), bool)  # contaminated by any combined indicator
            cov_any = np.zeros(len(grid), bool)  # covered by any combined indicator
            for rx, band, fld, rule in ind:  # for loop over indicators
                ref, q = refs[(rx, band, fld)]  # its reference and IQR
                g = ser[(rx, band, fld)]  # its series
                # readings from 60 s before the margin (a held sample may reach into it) to its end
                gg = g[(g.timestamp_utc >= a - pd.Timedelta(seconds=60)) & (g.timestamp_utc < b)].set_index(
                    "timestamp_utc").value  # argument of set_index(): the time column becomes the index, so readings can be looked up by time
                gg = gg[~gg.index.duplicated()]  # one reading per time
                # reindex with forward fill: each grid point takes the last reading, if not older than 60 s
                v = (gg.reindex(grid, method="ffill", tolerance=pd.Timedelta(seconds=60)).to_numpy()
                     if len(gg) else np.full(len(grid), np.nan))  # conditional expression: an all-missing series when the receiver gave no reading near the margin
                cov = ~np.isnan(v)  # boolean: a reading covers this grid point
                # chained conditional expression: the indicator's contamination rule
                bad = (v < ref - 3.0) if rule == "gain" else (v > 2 * ref) if rule == "x2" else (v > ref + 3 * q)
                bad = bad & cov  # only covered points can be contaminated
                if rule != "x2":  # if: part of the combination
                    bad_any |= bad; cov_any |= cov  # two statements: "|=" accumulates the logical or
                rows.append({"test_code": sr["test_code"], "margin": f"{a:%H:%M:%S}-{b:%H:%M:%S}",  # the row
                             "indicator": f"{rx} {band} {fld} ({rule})", "margin_min": round(len(grid) / 600, 3),  # dict entry: the indicator and its rule, and the margin length in minutes
                             # dict entries: covered and contaminated minutes (600 grid points per minute)
                             "covered_min": round(cov.sum() / 600, 3), "contaminated_min": round(bad.sum() / 600, 3),
                             "possibly_contaminated_margin": flg})  # dict entry: the flag
            # the combined row: contaminated where any combined indicator is
            rows.append({"test_code": sr["test_code"], "margin": f"{a:%H:%M:%S}-{b:%H:%M:%S}",
                         "indicator": "ALL combined (six gains, u-blox noise IQR rule, u-blox CW)",  # dict entry: names the combined indicator and what it pools
                         "margin_min": round(len(grid) / 600, 3), "covered_min": round(cov_any.sum() / 600, 3),  # dict entry: margin length and the minutes any indicator covers
                         # dict entry: minutes any indicator marks contaminated, and the possibly contaminated flag
                         "contaminated_min": round(bad_any.sum() / 600, 3), "possibly_contaminated_margin": flg})
    out = pd.DataFrame(rows)  # the table
    out["surviving_min"] = (out.margin_min - out.contaminated_min).round(3)  # clean minutes
    return out  # per session, margin and indicator


def site_of(long: pd.DataFrame, site: pd.DataFrame, dist: pd.DataFrame) -> pd.Series:  # function: site per row
    km = site.dist_to_sender_km.round(1).astype(str) + " km"  # each session's site as text, e.g. "12.4 km"
    lab = long.test_code.map(km)  # Series.map: each row's session -> its site label
    mv = dist[dist.part == "moving"]  # 2.6.5's moving part (excluded from the site split)
    for _, r in mv.iterrows():  # for loop over moving parts (one)
        t1 = pd.Timestamp(r.part_end_utc, tz="UTC")  # when the vehicle stopped
        lab = lab.mask((long.test_code == r.test_code) & (long.timestamp_utc < t1), "moving")  # those rows: "moving"
    return lab  # one site label per row


def within_capture(long: pd.DataFrame) -> pd.DataFrame:  # function: the group comparison inside each capture
    rows = []  # one row per capture, receiver, band and field with both groups present
    for (code, rx, band, fld), g in long.groupby(["test_code", "receiver", "band", "field"]):  # per group
        fam = "Septentrio" if rx.startswith("Sept") else "u-blox"  # family
        d = DIRECTION.get((fam, fld), 0)  # direction
        a = g.loc[g.exposure_group == G50, "value"].to_numpy()  # 50 W values in this capture
        b = g.loc[g.exposure_group == GBL, "value"].to_numpy()  # baseline values in this capture
        if d == 0 or len(a) < 60 or len(b) < 60:  # if: no direction, or fewer than 60 of either
            continue  # skip
        diff = float(np.median(a) - np.median(b))  # 50 W minus baseline, signed
        rows.append({"test_code": code, "receiver": rx, "band": band, "field": fld,  # appends the row
                     "n_50w": len(a), "n_baseline": len(b), "difference": round(diff, 3),  # dict entry: readings during the 50 W test and in the baseline, and the median difference
                     # dict entries: the exact effect size and whether the sign agrees with the expected direction
                     "cliffs_delta": round(cliffs_delta(a, b), 4),
                     "matches_prediction": bool((d < 0 and diff < 0) or (d > 0 and diff > 0))})  # dict entry: whether the difference has the sign fixed in advance
    return pd.DataFrame(rows)  # the table


def pyramid_contrasts(long: pd.DataFrame, pyr: dict) -> list:  # function: Test C's contrasts for a given pyramid
    p0 = pd.Timestamp(pyr["entry"]["start_utc"]); p1 = pd.Timestamp(pyr["entry"]["end_utc"])  # its window
    ins = long[(long.timestamp_utc >= p0) & (long.timestamp_utc < p1)].copy()  # values inside it
    prows = []  # pyramid rows
    if ins.empty:  # if: no data inside the window
        return prows  # nothing to contrast
    off = (ins.timestamp_utc - p0).dt.total_seconds()  # seconds since the pyramid started
    ins["step"] = (off // (pyr["step_min"] * 60)).astype(int) + 1  # step number 1-15
    ins["jamming_third"] = (off % (pyr["step_min"] * 60)) < pyr["jam_min"] * 60  # True in the jamming minutes
    # dict comprehension: front-end name (e.g. "GPSL1/E1") -> plan band name (e.g. "L1")
    code_to_name = {FRONTEND[fe]: nm for nm, fe in PLAN_BAND_TO_FRONTEND.items() if fe in FRONTEND}
    for (rx, band, fld), g in ins.groupby(["receiver", "band", "field"]):  # per receiver, band, field
        nm = code_to_name.get(band)  # plan name of this Septentrio band (u-blox and Aux bands do not match)
        fam = "Septentrio" if rx.startswith("Sept") else "u-blox"  # family
        d = DIRECTION.get((fam, fld), 0)  # direction
        if nm is None or d == 0:  # if: not a pyramid band, or no direction
            continue  # skip
        pr = pyr["predicted_by_band"][nm]  # predicted steps
        s = g[g.step.between(min(pr), max(pr))]  # values in the predicted steps (inclusive)
        a = s.loc[s.jamming_third, "value"].to_numpy()  # jamming portions
        b = s.loc[~s.jamming_third, "value"].to_numpy()  # break portions
        if len(a) < 50 or len(b) < 50:  # if: too few
            continue  # skip
        diff = float(np.median(a) - np.median(b))  # jamming minus break
        dl = cliffs_delta(a, b)  # negative = lower while jamming
        prows.append({"receiver": rx, "band": band, "plan_band": nm, "field": fld,  # appends the row
                      "predicted_first": min(pr), "predicted_last": max(pr),  # dict entry: the first and last pyramid step in which the band should be jammed
                      "n_jamming": len(a), "n_break": len(b),  # dict entry: readings in the jamming minutes and in the break minutes
                      "median_jamming": round(float(np.median(a)), 3),  # dict entry: median in the jamming minutes
                      "median_break": round(float(np.median(b)), 3),  # dict entry: median in the break minutes
                      "jamming_minus_break": round(diff, 3), "cliffs_delta": round(dl, 4),  # dict entry: jamming minus break median, and Cliff's delta between them
                      "matches_prediction": bool((d < 0 and diff < 0) or (d > 0 and diff > 0))})  # dict entry: whether the difference has the sign fixed in advance
    return prows  # list of row dicts


def on_air_check(w0, w1, long, sessions, refs) -> dict:  # function: did a test on air in a capture margin show?
    """
    The margin parts of every capture that overlap the window [w0, w1) for at least 60 s are taken (our own
    scheduled windows are excluded by construction). There, the mosaic-X5 L1 gain must stay within 3 dB of its
    quiet reference and the u-blox_01 L1 noise within 3 margin IQRs; outcome survives, fails or untestable.
    """
    ov = []  # (capture, start, end) overlaps
    for _, sr in sessions.iterrows():  # for loop over sessions
        for a, b in margin_parts(sr):  # for loop over its margins
            lo, hi = max(a, w0), min(b, w1)  # the overlap with the window
            if (hi - lo).total_seconds() >= 60:  # if: at least a minute
                ov.append((sr["test_code"], lo, hi))  # kept
    if not ov:  # if: none
        # dict returned when no margin overlaps the window: outcome untestable, nothing measured
        return {"outcome": "untestable", "overlap": "", "overlap_min": 0.0, "x5_l1_gain_minus_ref_db": None,
                "ublox_l1_noise_minus_ref": None}  # nothing on air while we recorded a margin
    # boolean masks: the mosaic-X5 GPS L1/E1 gain readings
    x5 = long[(long.receiver == "Septentrio_02") & (long.band == "GPSL1/E1") & (long.field == "agc_gain_db")]
    # boolean masks: the u-blox_01 L1 noise readings
    nz = long[(long.receiver == "u-blox_01") & (long.band == "L1") & (long.field == "noise_per_ms")]
    msk = lambda t: np.logical_or.reduce([(t >= lo) & (t < hi) for _, lo, hi in ov])  # lambda: inside any overlap
    g_ = x5[msk(x5.timestamp_utc)].value  # X5 L1 gain during the overlaps
    n_ = nz[msk(nz.timestamp_utc)].value  # u-blox noise during the overlaps
    rg = refs.get(("Septentrio_02", "GPSL1/E1", "agc_gain_db"))  # the X5 L1 quiet reference
    # tuple assignment: the noise reference and its pooled margin IQR (stored by main under a '_iqr' key)
    rn, qn = refs.get(("u-blox_01", "L1", "noise_per_ms")), refs.get(("u-blox_01", "L1", "noise_per_ms_iqr"), 1.0)
    dg = float(g_.median() - rg) if len(g_) and rg is not None else None  # gain departure, dB
    dn = float(n_.median() - rn) if len(n_) and rn is not None else None  # noise departure, counts
    fails = (dg is not None and dg < -3.0) or (dn is not None and dn > 3 * qn)  # boolean: a departure
    # returned dict: the outcome (conditional expression: fails, survives, or untestable when nothing was read)
    return {"outcome": "fails" if fails else ("survives" if dg is not None or dn is not None else "untestable"),
            # dict entries: where and how long the test overlapped our margins, and the two departures
            "overlap": "; ".join(f"{c} {lo:%H:%M:%S}-{hi:%H:%M:%S}" for c, lo, hi in ov),
            "overlap_min": round(sum((hi - lo).total_seconds() for _, lo, hi in ov) / 60, 2),  # dict entry: total minutes of overlap with the session margins
            "x5_l1_gain_minus_ref_db": round(dg, 2) if dg is not None else None,  # dict entry: mosaic-X5 L1 gain minus its reference in dB, empty when not read
            "ublox_l1_noise_minus_ref": round(dn, 2) if dn is not None else None}  # dict entry: u-blox L1 noise minus its reference, empty when not read


# ---- the 2.6.12 timeline (D2 and change 2) -------------------------------------
# What each receiver and each of its outputs did during our time-spoofing test, on HOST time. The decoders below
# read, for labelling and verification only and never as features:
#   MeasEpoch (4027) and UBX-RXM-RAWX: the header (time tag, counts) and satellite IDs; no measurement field
#   (pseudorange, phase, Doppler, C/N0) is unpacked, so this script still decodes no observable;
#   PVTGeodetic (4007) and UBX-NAV-PVT: time, mode/fix and position.
CATALOGUE_JUMP_S = (366 * 86400 + 3600 + 15 * 60)  # the catalogue's 2.6.12 jump: 1 year (365 d), 1 day, 1 h, 15 min
JUMP_GATE_S = 5.0  # float (s): a time tag this far from host time has jumped (latency is about 0.1 s; see the CSV)
POS_THRESHOLDS_M = (50, 100, 500)  # chosen settings (m): a reported position this far from the site is deceived
HOLD_S = 10.0  # float (s): a state must hold this long to count (position, time and tracking events)


def decode_measepoch_header(p: bytes):  # function: MeasEpoch (4027) header -> (N1, distinct satellite IDs) or None
    """Header only: N1, SB1Length, SB2Length, then each Type1 sub-block's SVID (offset 2) and N2 (offset 19,
    used only to skip its nested Type2 sub-blocks). No measurement field is read (SEPT_DOC 4027)."""
    if len(p) < 6:  # if: shorter than N1, SB1Length, SB2Length, CommonFlags, CumClkJumps, Reserved
        return None  # cannot decode
    n1, sb1, sb2 = p[0], p[1], p[2]  # indexing bytes gives ints: the three header counts
    o, svids = 6, set()  # read position after the 6 header bytes, and the set of satellite IDs
    for _k in range(n1):  # for loop over the N1 Type1 sub-blocks
        if o + 20 > len(p):  # if: SVID and N2 would lie past the payload
            return None  # does not fit
        svids.add(p[o + 2])  # the SVID byte only
        o += sb1 + p[o + 19] * sb2  # skips this sub-block and its N2 nested Type2 sub-blocks, by length only
    return n1, len(svids)  # tuple: sub-block count and distinct satellites


def decode_pvtgeodetic_lead(p: bytes):  # function: PVTGeodetic (4007) leading fields -> (mode, error, lat, lon)
    """Mode u1, Error u1, Latitude f8, Longitude f8 (radians) - fields every revision shares (SEPT_DOC 4007)."""
    if len(p) < 18:  # if: too short for the four leading fields
        return None  # cannot decode
    mode, err = p[0], p[1]  # two unsigned bytes: PVT mode bit field and error code
    lat, lon = struct.unpack_from("<dd", p, 2)  # "<dd": two little-endian 8-byte floats, radians
    if lat < -1e10 or lon < -1e10:  # if: a Do-Not-Use value (-2e10), i.e. no solution
        lat = lon = None  # chained assignment: no position
    return mode & 0x0F, err, lat, lon  # "& 0x0F" keeps bits 0-3, the type of PVT solution (0 = none)


def first_hold(t: pd.Series, flag: pd.Series, hold_s: float = HOLD_S):  # function: first time a state holds
    """First time at which `flag` becomes True and stays True for at least hold_s seconds (a gap over 5 s in the
    data ends a run). Returns (start, end) of that run, or (None, None)."""
    t = pd.to_datetime(t).reset_index(drop=True); f = flag.reset_index(drop=True).astype(bool)  # aligned, renumbered
    brk = (f != f.shift()) | (t.diff().dt.total_seconds() > 5)  # a new run where the flag flips or the data gap
    run = brk.cumsum()  # run number per sample
    for _, g in t[f].groupby(run[f]):  # for loop over True runs, in time order
        if (g.iloc[-1] - g.iloc[0]).total_seconds() >= hold_s:  # if: it lasts long enough
            return g.iloc[0], g.iloc[-1]  # its first and last time
    return None, None  # never


def last_release(t: pd.Series, flag: pd.Series, after, hold_s: float = HOLD_S):  # function: when a state ends
    """After `after`, the first time at which `flag` is False and stays False for at least hold_s seconds, taken
    after the LAST True sample - i.e. the release from the state."""
    t = pd.to_datetime(t).reset_index(drop=True); f = flag.reset_index(drop=True).astype(bool)  # aligned
    tt = t[f & (t >= after)]  # times in the state after its start
    if tt.empty:  # if: none
        return None  # no release to find
    rest = t > tt.iloc[-1]  # boolean: after the last sample in the state
    s0, _ = first_hold(t[rest], ~f[rest], hold_s)  # the first sustained period out of the state
    return s0  # its start, or None if the capture ends first


# function: the 2.6.12 timeline; returns (table, integrity rows of these reads, leap-second checks)
def timeline_2612(sessions: pd.DataFrame, site: pd.DataFrame, latd: pd.DataFrame) -> pd.DataFrame:
    """One row per receiver and event, on host time, for test 2.6.12 (written to ground_truth/spoof_2612_timeline.csv)."""
    sr = sessions[sessions.test_code == "2.6.12"].iloc[0]  # the session row
    folder = DAY / sr["session_folder"]  # its raw folder
    to_utc = host_clock(folder)  # the laptop clock converter for this session
    cs, ss, se, ce = (pd.Timestamp(sr[c], tz="UTC") for c in  # generator unpacked: the four session instants
                      ("capture_start_utc", "scheduled_start_utc", "scheduled_end_utc", "capture_end_utc"))  # generator clause: the four column names, read as UTC time stamps in this order
    lat0, lon0 = site.loc["2.6.12", "median_lat"], site.loc["2.6.12", "median_lon"]  # the session's site (the handset script)
    integ = Integrity()  # separate integrity record: the timeline's reads do not enter the verdict tests' report
    rows, leap_checked = [], 0  # event rows, and how many RAWX leap-second fields were checked
    quality = []  # per-second NAV-PVT fix-quality tables of the u-blox receivers
    for f in sorted(folder.glob("*.sbf")) + sorted(folder.glob("*.ubx")):  # for loop over the five files
        rx = (f"Septentrio_{f.name[-6:-4]}" if f.suffix == ".sbf" else f"u-blox_{f.name[-6:-4]}")  # receiver tag
        sc_path = folder / "host_timing" / (f.name + ".hosttime.csv")  # its sidecar
        meas, pvt, gain = [], [], []  # (end, receiver time, count, satellites); (end, rx time, valid, lat, lon); (end, value)
        if f.suffix == ".sbf":  # if: a Septentrio file
            for blk, rev, tow, wnc, p, end in sbf_blocks(f, integ, {4027, 4007, 4014}):  # the three block types
                t_ = gps_to_utc(tow, wnc)  # the block's time tag (receiver time), or None
                if blk == 4027:  # if: MeasEpoch - header only
                    d = decode_measepoch_header(p)  # (N1, satellites) or None
                    if d:  # if: decoded
                        meas.append((end, t_, d[0], d[1]))  # kept
                elif blk == 4007:  # elif: PVTGeodetic - leading fields only
                    d = decode_pvtgeodetic_lead(p)  # (mode, error, lat, lon) or None
                    if d:  # if: decoded
                        # a COMPUTED solution with no error and a position: type 3 ("Fixed location", a position the
                        # user set, not computed) and 0 (none) are excluded (SEPT_DOC 4007 Mode bits 0-3)
                        ok = d[0] not in (0, 3) and d[1] == 0 and d[2] is not None
                        # np.degrees: the position from radians to degrees
                        pvt.append((end, t_, ok, np.degrees(d[2]) if ok else None, np.degrees(d[3]) if ok else None,
                                    # u-blox-only quality fields absent here: seven placeholders keep the columns aligned
                                    None, None, None, None, None, None, None))
                else:  # else: ReceiverStatus - the main-antenna GPS L1/E1 gain only
                    d = decode_receiverstatus(p)  # header and AGC list
                    if d:  # if: decoded
                        # list comprehension: the gain of the main-antenna GPS L1/E1 front end (code 0, antenna 0), if present
                        g_ = [a["agc_gain_db"] for a in d["agc"] if a["frontend_code"] == 0 and a["antenna_id"] == 0]
                        if g_ and g_[0] is not None:  # if: an L1 gain value exists
                            gain.append((end, g_[0]))  # kept
        else:  # else: a u-blox file
            for cls, mid, p, end in ubx_messages(f, integ, {(0x02, 0x15), (0x01, 0x07), (0x0A, 0x38)}):  # three types
                if (cls, mid) == (0x02, 0x15) and len(p) >= 16:  # if: RXM-RAWX - header only
                    rcv_tow, week, leaps, nmeas, recstat = struct.unpack_from("<dHbBB", p, 0)  # time tag and counts
                    # svId of each measurement: one byte at 16 + 32n + 21 (UBX_DOC); nothing else of it is read
                    svs = {(p[16 + 32 * n + 20], p[16 + 32 * n + 21]) for n in range(nmeas) if 16 + 32 * n + 22 <= len(p)}
                    ls = leaps if recstat & 0x01 else LEAP_SECONDS  # the leap seconds the receiver itself applies
                    t_ = GPS_EPOCH + timedelta(weeks=week, seconds=rcv_tow) - timedelta(seconds=ls)  # time tag, UTC
                    meas.append((end, t_, nmeas, len(svs), recstat & 0x01, leaps))  # kept, with the leap fields
                elif (cls, mid) == (0x01, 0x07) and len(p) >= 92:  # elif: NAV-PVT
                    year, month, day, hh, mm, s_ = struct.unpack_from("<HBBBBB", p, 4)  # UTC fields
                    valid, nano, fix, flags = p[11], struct.unpack_from("<i", p, 16)[0], p[20], p[21]  # validity etc.
                    lon, lat = (v * 1e-7 for v in struct.unpack_from("<ii", p, 24))  # degrees (scale 1e-7)
                    t_ = None  # reported UTC, if valid
                    if valid & 0x03 == 0x03 and 1980 < year < 2200:  # if: validDate and validTime
                        try:  # try: an impossible date raises
                            # the reported UTC fields plus the nano fraction, as a timezone-aware datetime
                            t_ = datetime(year, month, day, hh, mm, s_, tzinfo=TZ_UTC) + timedelta(microseconds=round(nano / 1000))
                        except ValueError:  # except: not a real date
                            t_ = None  # left unknown
                    ok = bool(flags & 0x01) and fix in (2, 3, 4)  # gnssFixOK and a 2D/3D fix (5 = time only, excluded)
                    # verification fields (never features): numSV, hAcc (U4 mm at 40), pDOP (U2 x 0.01 at 76)
                    hacc_mm, pdop = struct.unpack_from("<I", p, 40)[0], struct.unpack_from("<H", p, 76)[0]
                    # kept, with the fix type and the fix-quality fields; the position also kept when not valid
                    pvt.append((end, t_, ok, lat if ok else None, lon if ok else None, fix, int(flags & 0x01), p[23],
                                hacc_mm / 1000.0, pdop * 0.01, lat, lon))  # tuple element: horizontal accuracy in metres, position dilution, and the raw latitude and longitude
                elif (cls, mid) == (0x0A, 0x38):  # elif: MON-RF - the L1 block's AGC count
                    d = decode_monrf(p)  # list of RF blocks
                    if d:  # if: decoded
                        gain.append((end, d[0]["agc_cnt"]))  # block 0 is L1
        # host time for every kept message of this receiver
        M = pd.DataFrame(meas, columns=["end", "rx_t", "n_meas", "n_sv"] + (["leap_known", "leaps"] if f.suffix == ".ubx" else []))
        # PVT table; the last seven columns are u-blox verification fields (None for Septentrio)
        P = pd.DataFrame(pvt, columns=["end", "rx_t", "ok", "lat", "lon", "fix_type", "gnss_fix_ok", "num_sv", "hacc_m",
                                       "pdop", "lat_any", "lon_any"])
        G = pd.DataFrame(gain, columns=["end", "value"])  # gain table
        for T in (M, P, G):  # for loop over the three tables
            # conditional expression: host time of each message, or an empty time column when the table is empty
            T["host"] = host_times(sc_path, T.end, to_utc) if len(T) else pd.Series(dtype="datetime64[ns, UTC]")
        if f.suffix == ".ubx" and len(M):  # if: u-blox, check its leap seconds before the attack
            pre = M[(M.host < ss) & (M.leap_known == 1)]  # RAWX epochs before 2.6.12 starts, leap seconds known
            leap_checked += len(pre)  # counted
            # the leap seconds of every RAWX epoch of the capture, attack included (set.update adds all)
            integ.ubx_facts["RAWX leapS, whole 2.6.12 capture"].update(M[M.leap_known == 1].leaps.astype(int))
            if (pre.leaps != LEAP_SECONDS).any():  # if: any differs from the constant
                # stops: the receiver's leap seconds disagree with the constant all times rest on
                die(f"{f.name}: RXM-RAWX leapS {sorted(set(pre.leaps))} differs from LEAP_SECONDS {LEAP_SECONDS}")
        lt = latd[(latd.receiver == rx) & (latd.test_code == "2.6.12")]  # this receiver's measured latency
        lmed = lt.median_s.iloc[0] if len(lt) else None  # median, s
        lmax = lt.max_s.iloc[0] if len(lt) else None  # maximum, s

        # nested function: appends one timeline row; note carries a sentence where a number alone is not enough
        def add(event, definition, host_t, rx_t=None, value=None, unit="", threshold="", note=""):
            rows.append({"receiver": rx, "receiver_name": RX_NAME.get(rx, rx), "event": event,  # appends the row
                         "definition": definition, "threshold": threshold,  # dict entry: the definition of the event and the threshold that triggers it
                         # dict entries: host time, the receiver's own time tag where one applies, the value
                         "host_time_utc": host_t.strftime("%H:%M:%S.%f")[:-3] if host_t is not None else "",
                         "receiver_time_utc": rx_t.strftime("%Y-%m-%d %H:%M:%S") if rx_t is not None else "",  # dict entry: the receiver's own time tag, empty when none applies
                         "value": value, "unit": unit,  # dict entry: the measured value and its unit
                         # the timing uncertainty: the measured host-time latency for this receiver and session
                         "latency_median_s": lmed, "latency_max_s": lmax,
                         "observed": host_t is not None, "note": note})  # dict entries: the flag and the note (-)

        # nested function: one row per offset state of a time tag (each change of state, as decided)
        # only from the deception event on (s0): a single early time tag is not a state (the X5's buffered first
        # epochs are stamped about 5 s before host time at capture start)
        def states(host, off, what, s0):
            if s0 is None:  # if: no deception event, so no offset state to report
                return  # nothing
            st = off.round(0).where(off.abs() > JUMP_GATE_S, 0.0)  # the offset to the second; true time counts as 0
            st = st.where(host >= s0, 0.0)  # states only from the event on
            run = (st != st.shift()).cumsum()  # run number: a new run at every change of state
            for v_, g_ in st[st != 0].groupby(st[st != 0], sort=False):  # per deceived state, in order of first entry
                h_ = host[g_.index]  # the host times in that state
                n_in = run[g_.index].nunique()  # how many separate times the receiver entered it
                # seconds spent in this state: the span of each of its runs, summed (generator inside sum)
                secs = sum((host[run == r_].max() - host[run == r_].min()).total_seconds() for r_ in run[g_.index].unique())
                wk, tow = round(v_ / 604800), v_ - round(v_ / 604800) * 604800  # whole weeks and remainder in it
                # chained conditional expression: which part of the catalogue's jump this offset carries, from its
                # whole weeks and its time-of-week remainder; a small remainder is named as a
                # time-of-week step (u-blox 01's navigation time first stepped +15 s with the week and leap seconds
                # unchanged), never 'other'
                lab = ("full catalogue jump" if abs(v_ - CATALOGUE_JUMP_S) <= 60 else
                       "whole weeks only" if wk >= 1 and abs(tow) < 0.5 else  # conditional expression: whole weeks moved and the time of week kept
                       f"whole weeks plus the {tow:.0f} s time-of-week step" if wk >= 1 and abs(tow) <= 60 else  # conditional expression: whole weeks plus the time-of-week step
                       "time of week only" if wk == 0 and abs(v_ - CATALOGUE_JUMP_S % 604800) <= 60 else  # conditional expression: only the time of week moved, by the catalogue's amount
                       # conditional expression: only a time-of-week step, or else weeks and seconds spelled out
                       "time-of-week step only, week unchanged" if wk == 0 else f"{wk} weeks plus {tow:.0f} s")
                t_first = h_.min()  # first entry into the state
                # adds the offset-state row: first entry, the receiver's own time tag then, the offset and its meaning
                add(f"{what} offset state", f"{what} time tag offset from host time, to the second; a new row for "
                    # string continued: the event time, the receiver tag then, the offset in seconds and its unit
                    "each state", t_first, M_or_P_rx(what, t_first), round(float(v_), 0), "s (offset from host time)",
                    note=f"{lab}; entered {n_in} time(s), {secs:.0f} s in total")  # keyword argument note=: the offset's meaning, how often it was entered and for how long
                if lab == "full catalogue jump":  # if: the full offset - the only state compared with the catalogue
                    # adds the full state's offset minus the catalogue's, in seconds
                    add(f"{what} jump minus catalogue offset", "full-offset state minus the catalogue's 1 year 1 day "
                        "1 h 15 min", t_first, value=round(float(v_) - CATALOGUE_JUMP_S, 1), unit="s")  # string continued: the event time, and the full offset minus the catalogue's jump in seconds

        # nested function: the receiver's own time tag at a host instant, from the measurement or the PVT table
        def M_or_P_rx(what, t_):
            T_ = M if what == "measurement-engine" else P  # the table that holds this output
            return T_.loc[T_.host == t_, "rx_t"].iloc[0]  # its time tag at that instant

        # (1) gain onset: L1 gain below its pre-test reference by the entry threshold, held 30 s
        if len(G):  # if: gain data
            fld_ = "agc_gain_db" if rx.startswith("Sept") else "agc_cnt"  # the field
            ref = G[(G.host >= cs) & (G.host < ss)].value.median()  # pre-test margin reference
            low = G.value < ref - DROP_THRESHOLD[fld_]  # boolean: below the entry threshold
            s0, _ = first_hold(G.host, low, 30.0)  # first such state held 30 s
            # adds the gain-onset row; its value is the deepest fall below the reference after the onset
            add("gain onset", f"L1 {FIELD_LABEL[fld_]} below its pre-test reference by {DROP_THRESHOLD[fld_]:g}, "
                # string continued: the 30 s hold, the onset time, and the deepest fall below the reference afterwards
                "held 30 s", s0, value=round(float(G[G.host >= s0].value.min() - ref), 1) if s0 is not None else None,
                unit=FIELD_LABEL[fld_], threshold=f"{DROP_THRESHOLD[fld_]:g}")  # keyword arguments of add(): the unit of the gain field and the drop threshold that defines the onset
            # adds the gain-recovery row (conditional expression: only when an onset exists)
            add("gain recovery", "first time after the onset the gain stays within the threshold for 30 s",
                last_release(G.host, low, s0, 30.0) if s0 is not None else None)  # argument of add(): the recovery time, searched only when an onset was found
        # (2) loss of tracking and reacquisition (not deception): no measurement in the measurement epoch
        if len(M):  # if: measurement epochs
            zero = M.n_meas == 0  # boolean: nothing tracked
            s0, _ = first_hold(M.host, zero, 1.0)  # first epoch(s) with nothing tracked (any duration)
            # adds the loss-of-tracking row
            add("loss of tracking", "measurement epoch with no measurement (Septentrio N1 = 0; u-blox numMeas = 0)",
                s0)  # argument of add(): the time of the first epoch without measurements
            # adds the reacquisition row (only when a loss exists)
            add("reacquisition", f"first time after the loss that measurements return and hold {HOLD_S:g} s",
                last_release(M.host, zero, s0) if s0 is not None else None)  # argument of add(): the reacquisition time, searched only when a loss was found
            # what the epochs between the loss and the reacquisition held (the loss is the first empty
            # epoch, not a continuous outage)
            r0 = last_release(M.host, zero, s0) if s0 is not None else None  # the reacquisition instant, or None
            if s0 is not None and r0 is not None:  # if: both exist
                mid_ = M[(M.host >= s0) & (M.host < r0)].n_meas  # per-epoch measurement counts in between
                # adds the row: the largest count in one epoch, with the number of empty epochs and the range
                add("measurements between loss and reacquisition", "per-epoch measurement count from the loss up "
                    "to the reacquisition", s0, value=int(mid_.max()), unit="measurements (largest in one epoch)",  # string continued: the loss time, the largest count in one epoch, and its unit
                    # keyword argument note=: how many epochs were empty, and the range of the counts
                    note=f"{int((mid_ == 0).sum())} of {len(mid_)} epochs empty; counts {int(mid_.min())}-{int(mid_.max())}")
            # (3) measurement-engine deception: the measurement time tag jumps from true (host) time
            off = (pd.to_datetime(M.rx_t, utc=True) - M.host).dt.total_seconds()  # time tag minus host, s
            jump = off.abs() > JUMP_GATE_S  # boolean: jumped
            s0, _ = first_hold(M.host, jump)  # first jump held 10 s
            v_ = float(off[M.host == s0].iloc[0]) if s0 is not None else None  # the offset AT the event instant
            # adds the measurement-engine deception row, with the receiver's own jumped time tag and the offset then
            add("measurement-engine deception", f"measurement time tag more than {JUMP_GATE_S:g} s from host time, "
                f"held {HOLD_S:g} s", s0, M.loc[M.host == s0, "rx_t"].iloc[0] if s0 is not None else None,  # f-string continued: the hold time, the event time and the receiver's jumped time tag then
                round(v_, 1) if v_ is not None else None, "s (time tag minus host, at the event)")  # argument of add(): the time-tag offset from host time at the event, in seconds
            if r0 is not None and s0 is not None:  # if: a reacquisition and a jump both exist
                k_ = int(((M.host > r0) & (M.host <= s0)).sum())  # epochs after the reacquisition up to the first jump
                # adds the row: 1 means the very next epoch after the reacquisition carried the jumped time tag
                add("epochs from reacquisition to measurement-engine deception", "measurement epochs after the "
                    # string continued: the reacquisition time and the number of epochs until the first jumped one
                    "reacquisition epoch up to and including the first jumped one (1 = the next epoch)", s0, value=k_,
                    unit="epochs")  # keyword argument unit=: the count is in measurement epochs
            states(M.host, off, "measurement-engine", s0)  # one row per later offset state
            # adds the measurement-engine release row (only when a jump exists)
            add("measurement-engine release", f"first time after the last jumped epoch that the time tag is within "
                # f-string continued: the gate and hold times, and the release time if a jump was found
                f"{JUMP_GATE_S:g} s of host time for {HOLD_S:g} s", last_release(M.host, jump, s0) if s0 is not None else None)
        # (4) navigation-solution deception: reported time jumps, or the position leaves the site
        if len(P):  # if: PVT records
            have_t = P.rx_t.notna()  # boolean: a reported time exists
            off = (pd.to_datetime(P.rx_t, utc=True) - P.host).dt.total_seconds()  # reported minus host, s
            jump = have_t & (off.abs() > JUMP_GATE_S)  # boolean: jumped
            s0, _ = first_hold(P.host, jump)  # first jump held 10 s
            v_ = float(off[P.host == s0].iloc[0]) if s0 is not None else None  # the offset AT the event instant
            # adds the navigation time deception row, with the reported time and the offset then
            add("navigation time deception", f"reported time more than {JUMP_GATE_S:g} s from host time, held "
                f"{HOLD_S:g} s", s0, P.loc[P.host == s0, "rx_t"].iloc[0] if s0 is not None else None,  # f-string continued: the hold time, the event time and the reported time then
                round(v_, 1) if v_ is not None else None, "s (reported minus host, at the event)")  # argument of add(): the reported-time offset from host time at the event, in seconds
            states(P.host[have_t], off[have_t], "navigation", s0)  # one row per later offset state
            # adds the navigation time release row (only when a jump exists)
            add("navigation time release", f"first time after the last jumped epoch that the reported time is within "
                # f-string continued: the gate and hold times, and the release time if a jump was found
                f"{JUMP_GATE_S:g} s of host time for {HOLD_S:g} s", last_release(P.host, jump, s0) if s0 is not None else None)
            # position: distance of each valid fix from the session's site (median of the handsets' margin fixes)
            dm = P.apply(lambda r: haversine_m(r.lat, r.lon, lat0, lon0) if r.ok else np.nan, axis=1)  # metres
            pre = dm[(P.host >= cs) & (P.host < ss)]  # the receiver's own pre-test distances, as a check
            # adds the check row: the receiver's own pre-test distance from the site (no time; a check on the site)
            add("pre-test distance from the site", "median distance of the receiver's valid fixes from the session "
                # string continued: no event time, and the median pre-test distance from the site in metres
                "site before the test (a check on the site position)", None, value=round(float(pre.median()), 1)
                if pre.notna().any() else None, unit="m")  # conditional expression: no distance when the receiver had no valid pre-test fix; unit metres
            # a departure, not "deception" - the catalogue describes 2.6.12 as a time jump only, made while
            # stationary at Stave, so a position excursion under simultaneous jamming need not be a spoofed position
            dep50 = None  # the 50 m departure interval, for the fix-quality row
            for thr in POS_THRESHOLDS_M:  # for loop over the chosen thresholds
                far = dm > thr  # boolean: a valid fix beyond the threshold
                s0, _ = first_hold(P.host, far)  # first exceedance held 10 s
                r0 = last_release(P.host, far, s0) if s0 is not None else None  # when fixes come back within it
                # adds the departure row for this threshold, with the largest distance reached afterwards
                add(f"navigation position departure (> {thr} m)", f"valid fix more than {thr} m from the site, held "
                    # f-string continued: the hold time, the event time and the largest distance reached afterwards
                    f"{HOLD_S:g} s", s0, value=round(float(dm[P.host >= s0].max()), 0) if s0 is not None else None,
                    unit="m (largest distance afterwards)", threshold=f"{thr} m")  # keyword arguments of add(): the unit of that distance and the threshold of this row
                # adds the return row for this threshold (only when a departure exists)
                add(f"navigation position return (< {thr} m)", f"first time after the last fix beyond {thr} m that "
                    f"fixes stay within it for {HOLD_S:g} s", r0, threshold=f"{thr} m")  # f-string continued: the hold time, the return time and the threshold of this row
                if thr == POS_THRESHOLDS_M[0] and s0 is not None:  # if: the 50 m departure exists
                    dep50 = (s0, r0 if r0 is not None else ce)  # its interval (to the capture end if no return)
            if rx.startswith("u-blox"):  # if: a u-blox receiver, which reports fix quality
                pre_q = P[(P.host >= ss - pd.Timedelta(minutes=1)) & (P.host < ss)]  # the minute before the test
                if dep50 is None:  # if: no departure
                    # adds the fix-quality row with the medians before and during, and the verdict degraded or confident
                    add("position departure fix quality", "fix quality during the 50 m departure against the minute "
                        "before the test", None, note="no departure beyond 50 m")  # string continued: no event time, with a note that no departure beyond 50 m happened
                else:  # else: compare the departure with the minute before
                    dq = P[(P.host >= dep50[0]) & (P.host < dep50[1])]  # fixes during the departure
                    h0, h1 = pre_q.hacc_m.median(), dq.hacc_m.median()  # median hAcc before and during, m
                    n0, n1 = pre_q.num_sv.median(), dq.num_sv.median()  # median numSV before and during
                    bad = h1 > 3 * h0 or n1 < 0.7 * n0  # boolean: hAcc trebled or numSV fell by 30 % or more
                    # chained conditional expressions: the verdict on the fix, stated with the numbers behind it
                    add("position departure fix quality", "fix quality during the 50 m departure against the minute "
                        "before the test (degraded: hAcc more than 3 times, or numSV below 70 %)", dep50[0],  # string continued: what counts as degraded, and the time of the 50 m departure
                        value=round(float(h1), 1), unit="m (median hAcc during the departure)",  # keyword arguments of add(): the median horizontal accuracy during the departure, in metres
                        # keyword argument note=: the verdict, then the accuracy before and during the departure
                        note=(f"{'degraded fix' if bad else 'confident fix'}: hAcc {h0:.1f} m before -> {h1:.1f} m "
                              f"during; numSV {n0:.0f} -> {n1:.0f}; pDOP {pre_q.pdop.median():.2f} -> "  # f-string continued: satellites used before and during, and the position dilution before
                              # f-string continued: the position dilution during, and the fix types seen during the departure
                              f"{dq.pdop.median():.2f}; fix types during {sorted(dq.fix_type.dropna().astype(int).unique())}"))
                # per-second quality table, 15:49-15:56 host time (the first epoch in each host second)
                w_ = P[(P.host >= ss - pd.Timedelta(minutes=1)) & (P.host < ss + pd.Timedelta(minutes=6))].copy()
                w_["sec"] = w_.host.dt.floor("s")  # the host second each epoch falls in
                w_ = w_.groupby("sec").first().reset_index()  # one row per host second
                # distance of the reported position from the site, valid or not (verification only)
                w_["dist_from_site_m"] = [haversine_m(a_, b_, lat0, lon0) for a_, b_ in zip(w_.lat_any, w_.lon_any)]
                # keeps this receiver's per-second rows, in the columns of the quality CSV
                quality.append(w_.assign(receiver=rx)[["receiver", "sec", "rx_t", "fix_type", "gnss_fix_ok", "num_sv",
                                                       "hacc_m", "pdop", "dist_from_site_m"]])  # kept
            nofix = ~P.ok.astype(bool)  # boolean: no valid solution
            n_pre = int(P.ok[(P.host >= cs) & (P.host < ss)].astype(bool).sum())  # valid solutions before the test
            if n_pre == 0:  # if: the receiver had no computed solution before the test, so none could be lost
                # adds a check row instead of a fix-loss time: there was no fix to lose
                add("no valid solution before the test", "no computed navigation solution in the pre-test margin; "
                    "a loss of fix cannot be timed", None, value=0, unit="valid pre-test solutions")  # string continued: no event time, and zero valid pre-test solutions as the value
            else:  # else: time the loss and the return
                s0, _ = first_hold(P.host[P.host >= ss], nofix[P.host >= ss], 1.0)  # first loss after the start
                # adds the fix-lost row
                add("navigation fix lost", "first epoch after the scheduled start without a valid solution", s0)
                # adds the fix-regained row (only when a loss exists)
                add("navigation fix regained", f"first time after the loss that valid solutions hold {HOLD_S:g} s",
                    last_release(P.host, nofix, s0) if s0 is not None else None)  # argument of add(): the time the fix was regained, searched only when a loss was found
    out = pd.DataFrame(rows)  # the timeline
    q = pd.concat(quality, ignore_index=True) if quality else pd.DataFrame()  # the quality table, both receivers
    if len(q):  # if: any rows
        q["host_time_utc"] = q.sec.dt.strftime("%H:%M:%S")  # host second as text
        # the receiver's own time as text (empty when invalid)
        q["receiver_time_utc"] = pd.to_datetime(q.rx_t, utc=True).dt.strftime("%Y-%m-%d %H:%M:%S").fillna("")
        # the quality table's columns in order, rounded for the CSV
        q = q[["receiver", "host_time_utc", "receiver_time_utc", "fix_type", "gnss_fix_ok", "num_sv", "hacc_m",
               "pdop", "dist_from_site_m"]].round({"hacc_m": 2, "pdop": 2, "dist_from_site_m": 1})  # columns in order
    return out, q, integ, leap_checked  # tuple: the timeline, the quality table, integrity rows, leap checks


def margin_remnant(long: pd.DataFrame, sessions: pd.DataFrame, pyr: dict) -> pd.DataFrame:  # function: the possibly contaminated 2.6.12 margin/
    """
    The 15:45 step in 2.6.12, per receiver and field (an unexplained step,
    in time with the 1.8.2 scheduled end but not with its band; never a remnant). Median before the 1.8.2 scheduled end (capture start to the end) and after it (30 s after
    the end to the 2.6.12 scheduled start); the step time is the first time at or after the scheduled end at which the
    value has moved at least half the step in its direction and holds there 10 s.
    """
    sr = sessions[sessions.test_code == "2.6.12"].iloc[0]  # the session
    # tuple assignment: 2.6.12's capture start and scheduled start
    cs, ss = pd.Timestamp(sr["capture_start_utc"], tz="UTC"), pd.Timestamp(sr["scheduled_start_utc"], tz="UTC")
    pend = pd.Timestamp(pyr["entry"]["end_utc"])  # the 1.8.2 scheduled end
    st15 = pyr["steps"][-1]["bands"].split(";")  # the bands the last step jammed, as parsed from the plan
    # dict comprehension: front-end name -> plan band name, plus the u-blox block names
    name = {**{FRONTEND[fe]: nm for nm, fe in PLAN_BAND_TO_FRONTEND.items() if fe in FRONTEND}, "L1": "L1", "L2": "L2"}
    rows = []  # one row per receiver, band and field
    sub = long[(long.test_code == "2.6.12") & ~long.band.str.contains("Aux")]  # 2.6.12, main antennas only
    for (rx, band, fld), g in sub.groupby(["receiver", "band", "field"]):  # per receiver, band and field
        g = g.sort_values("timestamp_utc")  # in time order
        b_ = g[(g.timestamp_utc >= cs) & (g.timestamp_utc < pend)].value  # before the 1.8.2 end
        a_ = g[(g.timestamp_utc >= pend + pd.Timedelta(seconds=30)) & (g.timestamp_utc < ss)].value  # after it
        if len(b_) < 20 or len(a_) < 20 or fld == "blanking_pct":  # if: too little data, or a structurally zero field
            continue  # skip
        step = float(a_.median() - b_.median())  # the step, in the field's unit
        t_step = None  # when it happened, if it did
        if step != 0:  # if: the medians differ
            after = g[g.timestamp_utc >= pend]  # from the scheduled end on
            moved = (after.value - b_.median()) * np.sign(step) >= abs(step) / 2  # boolean: half the step, same sign
            t_step, _ = first_hold(after.timestamp_utc, moved, 10.0)  # first such state held 10 s
        rows.append({"receiver": rx, "band": band, "plan_band": name.get(band, ""), "field": fld,  # the row
                     "median_before_1_8_2_end": round(float(b_.median()), 2),  # dict entry: median of the field before the end of the pyramid test 1.8.2
                     "median_after_1_8_2_end": round(float(a_.median()), 2), "step": round(step, 2),  # dict entry: median after that end, and the size of the step between them
                     # dict entries: the step time, the scheduled end, their difference, and step the diagnosis script's bands
                     "step_time_utc": t_step.strftime("%H:%M:%S") if t_step is not None else "",
                     "scheduled_end_1_8_2_utc": f"{pend:%H:%M:%S}",  # dict entry: the scheduled end of test 1.8.2 as a clock time
                     # dict entry: step time minus scheduled end in seconds, empty when no step was found
                     "step_minus_scheduled_end_s": round((t_step - pend).total_seconds(), 1) if t_step is not None else None,
                     "step15_bands": ";".join(st15), "band_jammed_in_step15": name.get(band, "") in st15,  # dict entry: the bands jammed in step 15, and whether this band is one of them
                     "time_base": "host" if rx.startswith("u-blox") else "receiver (true time here)"})  # dict entry: host time for the u-blox, receiver time for the Septentrio
    return pd.DataFrame(rows)  # the table


# function: Test Area 3's scheduled times against receiver-detected onset, recovery and floor exit
def ta3_confirmation(long: pd.DataFrame, sessions: pd.DataFrame, own: dict, refs: dict, latd: pd.DataFrame):
    """
    Test Area 3's scheduled times against what the receivers recorded. For each of our six tests and
    four L1 indicators: onset = the gain first below its pre-test reference by the entry threshold for 30 s; recovery
    = the first time after it back within the threshold for 30 s; floor exit = the gain first back halfway from its
    floor (lowest value in the window) to its reference for 30 s after the onset, set against the end of the
    catalogue's initial jamming.
    """
    # list of (receiver, band, field): the four L1 indicators
    ind = [("Septentrio_02", "GPSL1/E1", "agc_gain_db"), ("Septentrio_03", "GPSL1/E1", "agc_gain_db"),
           ("Septentrio_01", "GPSL1/E1", "agc_gain_db"), ("u-blox_01", "L1", "agc_cnt")]  # (receiver, band, field)
    rows = []  # one row per test and indicator
    for _, sr in sessions.iterrows():  # for loop over our six tests
        code = sr["test_code"]  # the test
        cs, ss, se, ce = (pd.Timestamp(sr[c], tz="UTC") for c in  # generator unpacked: the four instants
                          ("capture_start_utc", "scheduled_start_utc", "scheduled_end_utc", "capture_end_utc"))  # generator clause: the four column names, read as UTC time stamps in this order
        ini = None if code in own["exempt"] else ss + pd.Timedelta(minutes=own["initial_min"])  # initial jamming end
        for rx, band, fld in ind:  # for loop over indicators
            # boolean masks: this indicator in this capture, in time order
            g = long[(long.test_code == code) & (long.receiver == rx) & (long.band == band)
                     & (long.field == fld)].sort_values("timestamp_utc")  # this indicator in this capture
            if g.empty:  # if: none
                continue  # skip
            pre = g[(g.timestamp_utc >= cs) & (g.timestamp_utc < ss)].value  # the pre-test margin
            ref = pre.median() if len(pre) >= 20 else refs.get((rx, band, fld))  # its reference (pooled if no margin)
            low = g.value < ref - DROP_THRESHOLD[fld]  # boolean: below the entry threshold
            on, _ = first_hold(g.timestamp_utc, low, 30.0)  # onset
            rec = last_release(g.timestamp_utc, low, on, 30.0) if on is not None else None  # recovery
            fx = None  # floor exit
            if on is not None and ini is not None:  # if: an onset, and a test with initial jamming
                w_ = g[g.timestamp_utc >= on]  # from the onset on
                fl = w_[w_.timestamp_utc < se].value.min()  # the floor reached inside the window
                fx, _ = first_hold(w_.timestamp_utc, w_.value >= (fl + ref) / 2, 30.0)  # back halfway, held 30 s
            lt = latd[(latd.receiver == rx) & (latd.test_code == code)]  # the measured latency here
            sec = lambda t_, ref_: round((t_ - ref_).total_seconds(), 1) if t_ is not None else None  # offset in s
            # appends the row: test, stationary or drive-past spoofer, indicator, times and offsets in seconds
            rows.append({"test_code": code, "spoofer": "stationary" if "tationary" in str(sr.get("test_name", "")) else
                         "drive-past", "receiver": rx, "field": fld,  # dict entries: test, spoofer, indicator
                         "time_base": "host" if rx.startswith("u-blox") else "receiver (true time)",  # dict entry: host time for the u-blox, true receiver time for the Septentrio
                         "scheduled_start_utc": f"{ss:%H:%M:%S}",  # dict entry: the scheduled start of the test
                         "onset_utc": on.strftime("%H:%M:%S.%f")[:-3] if on is not None else "",  # dict entry: when the indicator's onset was seen, to the millisecond
                         "onset_minus_start_s": sec(on, ss), "scheduled_end_utc": f"{se:%H:%M:%S}",  # dict entry: onset minus scheduled start in seconds, and the scheduled end
                         "recovery_utc": rec.strftime("%H:%M:%S.%f")[:-3] if rec is not None else "",  # dict entry: when the indicator recovered, to the millisecond
                         "recovery_minus_end_s": sec(rec, se),  # dict entry: recovery minus scheduled end in seconds
                         "initial_jamming_end_utc": f"{ini:%H:%M:%S}" if ini is not None else "exempt",  # dict entry: end of the initial jamming, or exempt for a test without it
                         "floor_exit_utc": fx.strftime("%H:%M:%S.%f")[:-3] if fx is not None else "",  # dict entry: when the indicator left its floor, to the millisecond
                         "floor_exit_minus_initial_end_s": sec(fx, ini) if ini is not None else None,  # dict entry: floor exit minus end of the initial jamming in seconds
                         "latency_median_s": lt.median_s.iloc[0] if len(lt) else None})  # dict entry: the median host-time latency of this receiver and session
    return pd.DataFrame(rows)  # the table


def main() -> None:  # function: runs the whole section in order
    PROCESSED.mkdir(parents=True, exist_ok=True)  # creates Data/processed if absent; no error if present
    GRAPH.mkdir(parents=True, exist_ok=True)  # creates Graph/ likewise
    print("=" * 78)  # rule of 78 "=" characters
    print("INTERFERENCE STATUS FROM THE PRIMARY RECEIVERS")  # title
    print("=" * 78)  # rule

    sessions = pd.read_csv(PROCESSED / "ground_truth" / "session_ground_truth.csv")  # the timeline script's session inventory
    sessions["_o"] = pd.to_datetime(sessions.capture_start_utc)  # helper column: capture start as datetime
    sessions = sessions.sort_values("_o").reset_index(drop=True)  # sessions in capture order, renumbered
    # each session's measured distance to the 50 W transmitter and its site position (the handset script)
    dist = pd.read_csv(PROCESSED / "ground_truth" / "session_distance_to_sender.csv")
    # the measured part of each session: 2.6.5's parked part (its moving part has no position)
    site = dist.dropna(subset=["dist_to_sender_km"]).groupby("test_code").last()
    d_lo, d_hi = site.dist_to_sender_km.min(), site.dist_to_sender_km.max()  # the range across sessions, km
    s1 = load_section1()  # the timeline script loaded as a module
    # shared loader: the timeline script's power rule (effective_power, incl.) and the timeline script's plan time zone
    plan, catalog = load_schedule(s1, PROCESSED)
    pyr = parse_pyramid(plan, flag)  # shared parser with all its safeguards: one 1.8.2 entry, arithmetic, window
    # every plan band 03 maps to a front end must be in the parsed prediction, or the pyramid test is wrong
    missing = sorted(set(PLAN_BAND_TO_FRONTEND) - set(pyr["predicted_by_band"]))  # set difference: unmatched names
    if missing:  # if: a band name 03 expects is absent from the parsed step list
        die(f"pyramid band names {missing} are not in the parsed 1.8.2 step list")  # stop rather than guess
    own = parse_own_jamming(catalog)  # our motorcade's own jamming structure

    # ---- manufacturer documents ------------------------------------------------------
    print("\n[1] MANUFACTURER DOCUMENTS")  # heading
    for p in sorted(DOCS.glob("*.pdf")):  # for loop over the PDFs in docs/ (listed, not parsed)
        print(f"      {p.name}  ({p.stat().st_size/1e6:.1f} MB)")  # name and size in MB
    print(f"\n      Septentrio: {SEPT_DOC}")  # the document cited
    # the product and firmware are read from the recordings' ReceiverSetup blocks, and the guide's
    # firmware from its file name in docs/, so no version below is typed
    setup = receiver_setup(sessions)  # one row per stream with a ReceiverSetup block
    # list comprehension: the version in the Septentrio guide's file name, one match object per PDF
    gd = [re.search(r"Firmware_v([\d.]+)", q.name) for q in sorted(DOCS.glob("Septentrio*Firmware_v*.pdf"))]
    guide_fw = gd[0].group(1) if gd and gd[0] else "unknown"  # the guide's firmware version, from its file name
    su = setup.drop_duplicates(["product_name", "gnss_fw_version"])  # distinct product/firmware pairs
    # the distinct pairs, e.g. "'mosaic-G5 P8' at firmware 2026.01.2-g83d1ca6770 (SBF_01 streams)"
    pairs = "; ".join(f"'{r.product_name}' at firmware {r.gnss_fw_version} (receiver version {r.rx_version}; "
                      f"{', '.join(sorted({x[-10:-4] for x in setup[setup.product_name == r.product_name].file}))})"  # f-string continued: the stream numbers whose setup blocks name this product
                      for r in su.itertuples())  # generator clause: one text per distinct product and firmware pair
    print(f"      The recordings' own ReceiverSetup blocks ({len(setup)} files) report {pairs}.")  # read, not typed
    # the guide's firmware, read from its file name
    print(f"      The guide obtained is for the mosaic-X5 at firmware {guide_fw}. The product and firmware")
    print("      therefore DO NOT match. SBF is versioned by block revision rather than")  # continued
    print("      firmware, and the revision check below establishes that every block this")  # continued
    print("      script decodes carries exactly one revision across all files, matching")  # continued
    print("      the layouts in the guide field for field. The definitions are sound at")  # continued
    print("      block-revision level; a mosaic-G5 guide could not be obtained.")  # continued
    # records the mismatch as a note
    note(f"The Septentrio reference guide obtained is for the mosaic-X5 at firmware {guide_fw}, while the "
         f"recordings' ReceiverSetup blocks report {pairs}; the other streams carry no ReceiverSetup block. "  # f-string continued: the products and firmware the recordings report
         "SBF is versioned by block revision, every "
         "decoded block carries a single revision across all files, and the decoded "
         "layouts reconcile exactly with the declared lengths, so the definitions hold "
         "at block-revision level.")
    print(f"\n      u-blox: {UBX_DOC}")  # the u-blox document cited
    print("      UBX-MON-VER is absent from every stream (counted in [2]), so the model is INFERRED,")  # fixed text
    print("      not confirmed. The evidence is the message inventory printed in [2]; UBX-MON-RF")  # continued
    print("      exists from protocol generation 9 onward (UBX_DOC).")  # continued (sourced)
    print("      That set identifies a u-blox F9-generation dual-band high-precision")  # continued
    print("      receiver, most plausibly the ZED-F9P, which is what the cited interface")  # continued
    print("      description covers.")  # continued
    # records the inference as a note
    note("The u-blox model is inferred, not confirmed: UBX-MON-VER is absent from every "
         "stream. The message set identifies an F9-generation dual-band high-precision "
         "receiver, most plausibly the ZED-F9P.")

    # ---- parse ----------------------------------------------------------
    print("\n[2] PARSING (status blocks only; no observable is decoded anywhere)")  # heading
    (sept, ubx, integ), rfb, lat = parse_sessions(sessions)  # nested tuple unpacking of parse_sessions' result
    # row counts of the three tables
    print(f"      Septentrio rows {len(sept)}, u-blox rows {len(ubx)}, "
          f"RFStatus band rows {len(rfb)}")  # f-string continued: the number of RFStatus band rows
    # the u-blox message inventory behind the model inference in [1], counted over every valid message
    ut, uf = integ.ubx_types, integ.ubx_facts  # message counts and facts
    # the inventory line: counts of the deciding message types and the version and size facts
    print(f"      u-blox inventory: MON-VER {ut[(0x0A, 0x04)]}, SEC-SIG {ut[(0x27, 0x09)]}, NAV-SIG {ut[(0x01, 0x43)]}, "
          # f-string continued: RF monitor message count, RF blocks per message, and the raw-measurement version
          f"MON-RF {ut[(0x0A, 0x38)]} messages; MON-RF RF blocks {sorted(uf['MON-RF RF blocks'])}; RXM-RAWX "
          f"version {sorted(uf['RXM-RAWX version'])}; RXM-SFRBX version {sorted(uf['RXM-SFRBX version'])}; "  # f-string continued: the raw-measurement and subframe message versions
          f"NAV-PVT payload {sorted(uf['NAV-PVT payload bytes'])} bytes")  # f-string continued: the navigation message payload sizes in bytes

    # ---- host-time latency: measured before the u-blox rows were moved onto host time ----------------
    # In the capture margins, where the receiver's own UTC is true, host arrival time minus receiver UTC is the
    # latency of the host time base (receiver output delay, serial link, laptop clock error together). An epoch
    # counts only if the two agree within LATENCY_GATE_S, which drops any epoch whose receiver time is deceived.
    LATENCY_GATE_S = 5.0  # float constant (s): the agreement gate for "receiver time is true"
    lat["latency_s"] = (pd.to_datetime(lat.host_utc, utc=True)  # host minus receiver time, in seconds
                        - pd.to_datetime(lat.receiver_utc, utc=True)).dt.total_seconds()  # arithmetic continued: minus the receiver time, giving the latency in seconds
    sw = sessions.set_index("test_code")  # session rows by test code, for the margin windows
    cs_ = pd.to_datetime(lat.test_code.map(sw.capture_start_utc), utc=True)  # capture start per row
    ss_ = pd.to_datetime(lat.test_code.map(sw.scheduled_start_utc), utc=True)  # scheduled start per row
    se_ = pd.to_datetime(lat.test_code.map(sw.scheduled_end_utc), utc=True)  # scheduled end per row
    ce_ = pd.to_datetime(lat.test_code.map(sw.capture_end_utc), utc=True)  # capture end per row
    h_ = pd.to_datetime(lat.host_utc, utc=True)  # host times as timestamps
    in_margin = ((h_ >= cs_) & (h_ < ss_)) | ((h_ > se_) & (h_ <= ce_))  # boolean: pre- or post-test margin
    lat["in_margin"] = in_margin  # kept as a column
    lat["kept"] = in_margin & (lat.latency_s.abs() < LATENCY_GATE_S)  # boolean: margin AND receiver time true
    lrows = []  # one summary row per receiver and session
    for (code_, rx_), g_ in lat.groupby(["test_code", "receiver"], sort=False):  # per session and receiver
        k_ = g_[g_.kept].latency_s  # the kept latencies
        lrows.append({"test_code": code_, "receiver": rx_,  # appends the summary row
                      # dict entries: margin samples, how many failed the gate, and the kept count
                      "n_margin": int(g_.in_margin.sum()), "n_gate_failed": int((g_.in_margin & ~g_.kept).sum()),
                      "n_kept": len(k_),  # dict entry: samples kept after the gate
                      # conditional expressions: statistics only when samples were kept, else None
                      "median_s": round(float(k_.median()), 3) if len(k_) else None,
                      "iqr_s": round(float(k_.quantile(0.75) - k_.quantile(0.25)), 3) if len(k_) else None,  # dict entry: interquartile range of the kept latencies, empty when none was kept
                      "min_s": round(float(k_.min()), 3) if len(k_) else None,  # dict entry: smallest kept latency
                      "max_s": round(float(k_.max()), 3) if len(k_) else None})  # dict entry: largest kept latency
    latd = pd.DataFrame(lrows).sort_values(["receiver", "test_code"])  # the summary table, by receiver then session
    latd = latd.astype(object).where(latd.notna(), None)  # .where: missing statistics become None, printed as "-"
    latd.to_csv(PROCESSED / "ground_truth" / "receiver_host_latency.csv", index=False, lineterminator="\n")  # written (new)
    print("\n      Host-time latency in the capture margins (host arrival minus receiver UTC, s):")  # sub-heading
    for _, r in latd.iterrows():  # for loop over the summary rows
        # one printed line per receiver and session; "-" when nothing was kept
        print(f"        {r.receiver:<14} {r.test_code:<7} n {r.n_kept:>5}  median "
              # f-string continued: median and interquartile range, or a dash when nothing was kept
              f"{r.median_s if r.median_s is not None else '-':>7}  IQR {r.iqr_s if r.iqr_s is not None else '-':>6}  "
              f"max {r.max_s if r.max_s is not None else '-':>6}  gate failures {r.n_gate_failed}")  # f-string continued: the largest latency and the number of gate failures

    # ---- integrity ------------------------------------------------------
    print("\n[3] INTEGRITY CHECKS")  # heading
    ir = pd.DataFrame(integ.rows)  # one row per file
    ir["revisions"] = ir["revisions"].astype(str)  # the revision dicts turned into text for the CSV
    tot_blocks = int(ir.blocks_or_messages.sum())  # all valid blocks and messages
    tot_crc = int(ir.crc_failures.sum())  # all checksum failures
    # the two totals and the rate, all computed
    print(f"  (a) Checksums. {tot_blocks} blocks and messages verified; "
          f"{tot_crc} failure(s), a rate of {tot_crc / tot_blocks:.1e}.")  # f-string continued: the number of failures and the failure rate
    sbf_n = int(ir[ir.kind == "SBF"].blocks_or_messages.sum())  # valid SBF blocks
    ubx_n = int(ir[ir.kind == "UBX"].blocks_or_messages.sum())  # valid UBX messages
    sbf_fail = int(ir[ir.kind == "SBF"].crc_failures.sum())  # SBF failures only
    ubx_fail = int(ir[ir.kind == "UBX"].crc_failures.sum())  # UBX failures only
    print(f"        SBF CRC-16-CCITT failures: {sbf_fail} in {sbf_n} blocks")  # printed with its denominator
    print(f"        UBX checksum failures:     {ubx_fail} in {ubx_n} messages")  # printed with its denominator
    failed_types: Counter = Counter()  # key -> failure count, across all files (SBF block number or UBX "0xCC 0xII")
    for r in integ.rows:  # for loop over files
        for b_, c_ in (r.get("failed_block_types") or {}).items():  # "or {}" guards a missing entry
            failed_types[b_] += c_  # adds the counts
    print(f"        failed types, as read from the failed block or message itself: {dict(failed_types)}")  # 
    # the types the verdict tests decode: four SBF block numbers and two UBX class/ID keys
    USED_SBF, USED_UBX = {4092, 4014, 4082, 4013}, {"0x0A 0x38", "0x01 0x07"}
    # dict comprehension: failures in a decoded type (an int key is SBF, a text key is UBX)
    used_failed = {b_: c_ for b_, c_ in failed_types.items()
                   # generator clause: keeps failed types that a verdict test decodes, SBF by number, UBX by name
                   if (isinstance(b_, int) and b_ in USED_SBF) or (isinstance(b_, str) and b_ in USED_UBX)}
    if used_failed:  # if: a decoded type failed (the stop rule, now for both families)
        # stops rather than report from possibly corrupt blocks
        die(f"checksum failures occurred in block or message types this section decodes: {used_failed}. "
            "The run stops rather than reporting results from a stream that may be "
            "corrupt in the blocks the conclusion rests on.")
    if tot_crc:  # if: any failure at all
        # records it with computed counts
        flag(f"{sbf_fail} SBF CRC failures in {sbf_n} blocks and {ubx_fail} UBX checksum failures in {ubx_n} "
             # f-string continued: the failed types as read, with the caution that the type field may be damaged
             f"messages, of types {dict(failed_types)} as read from the failed data (probable, not certain: the "
             "type field may itself be damaged). None is a type the verdict tests decode; the parser "
             "resynchronises after each.")
    print("        No block or message type that the verdict tests decode failed its checksum;")  # fixed text, true here
    print("        the run would have stopped otherwise.")  # continued
    stops = ir[ir.early_stop_offset.notna()]  # files whose last message ran past the end of the file
    for _, r in stops.iterrows():  # for loop over them
        # reported with the byte offset and how many bytes were left unread
        print(f"        {r.file}: a message at byte {int(r.early_stop_offset)} runs past the end of the file "
              f"({int(r.file_bytes - r.early_stop_offset)} bytes unread); the file is read up to it")  # f-string continued: the bytes left unread after that message
        # records the truncated last message as an anomaly, with its byte offset
        flag(f"{r.file}: the last message, at byte {int(r.early_stop_offset)}, is truncated by the end of the file "
             f"({int(r.file_bytes - r.early_stop_offset)} bytes); it is not decoded")  # recorded
    nmea = int(ir[ir.kind == "UBX"].framing_failures.sum())  # resyncs in the u-blox files
    # the count; the NMEA explanation is fixed text
    print(f"        Non-UBX gaps in the u-blox streams: {nmea}. These are interleaved")
    print("        NMEA sentences ($GNRMC, $GLGSV, $GAGSV, $GBGSV), not corruption; the")  # continued
    print("        receivers were configured to output NMEA and UBX on the same port.")  # continued

    allrev = defaultdict(set)  # block number -> revisions seen in any file
    for r in integ.rows:  # for loop over files
        for b_, v_ in (r["revisions"] or {}).items():  # the file's revision dict
            allrev[b_].update(v_)  # set.update: adds that file's revisions
    print("\n  (b) Block revisions, across every file and session:")  # sub-heading
    for b_ in sorted(USED_SBF):  # for loop over the decoded blocks
        print(f"        block {b_}: revisions {sorted(allrev[b_])}")  # their revisions
    multi_used = [b_ for b_ in USED_SBF if len(allrev[b_]) > 1]  # decoded blocks seen at more than one revision
    if multi_used:  # if: any
        # stops: fixed offsets would misread one revision
        die(f"blocks {multi_used} carry more than one revision; a fixed-offset decoder "
            "would misparse them silently")
    multi_other = sorted(b_ for b_, v_ in allrev.items() if len(v_) > 1)  # other blocks with several revisions
    print("        Every decoded block carries exactly one revision. Blocks elsewhere")  # fixed text
    print(f"        in the streams that carry more than one: {multi_other}")  # the list
    if multi_other:  # if: any
        # records it; whether PVTGeodetic (4007) is among them is now checked, not typed
        note(f"SBF blocks {multi_other} appear at more than one revision in these recordings"
             # string concatenation: adds why the satellite count avoids that block, when it is affected
             + (". PVTGeodetic (4007) is among them, which is why the satellite count is taken from ChannelStatus "
                "(4013, single revision) and the 2.6.12 timeline reads only its leading fields, which every "
                "revision shares" if 4007 in multi_other else "") + ".")  # string continued: the end of that sentence, which is added only when the block is affected

    print("\n  (c) and (d) Sub-block arithmetic and byte accounting per block type:")  # sub-heading
    # table header, two lines
    print(f"        {'block':<16} {'decoded':>9} {'sub-block':>11} {'consumed>':>11} "
          f"{'non-zero':>10}")  # f-string continued: the last header cell
    print(f"        {'':<16} {'':>9} {'mismatch':>11} {'declared':>11} {'padding':>10}")  # header, second line
    bad_layout = False  # no layout problem found yet
    for name, rec in sorted(LAYOUT.items()):  # for loop over the accounted block types (ChannelStatus now included)
        n = sum(rec["consumed"].values())  # how many blocks of this type were accounted
        over = rec["consumed_over_declared"]  # blocks whose decoded fields ran past the declared length
        # the row, now printing the computed count
        print(f"        {name:<16} {n:>9} {rec['subblock_mismatch']:>11} "
              f"{over:>11} {rec['nonzero_padding']:>10}")  # f-string continued: the decoded layouts that overrun, and the blocks with non-zero padding
        if rec["subblock_mismatch"] or over:  # if: a mismatch, or fields read past the declared length
            bad_layout = True  # remembers it
    if bad_layout:  # if: any problem
        # stops
        die("a block layout does not reconcile with its declared length; the decoder "
            "is wrong and no result from it can be trusted")
    print("        Every block reconciles: header plus sub-blocks never exceeds the declared")  # fixed text, checked above
    print("        length. Non-zero padding is counted only: SEPT_DOC 4.1.5 leaves padding undefined.")  # continued
    ir.to_csv(PROCESSED / "ground_truth" / "receiver_integrity_report.csv", index=False, lineterminator="\n")  # writes the report

    # ---- field definitions ----------------------------------------------
    fd = pd.DataFrame(FIELDS)  # the field table
    # Series.map with a dict: -1/1/0 -> words
    fd["direction_under_interference"] = fd.direction.map(
        {-1: "decreases", 1: "increases", 0: "no direction claimed"})  # argument of map(): turns the direction codes into the words written in the table
    # drops the numeric column and writes the CSV
    fd.drop(columns=["direction"]).to_csv(
        PROCESSED / "ground_truth" / "receiver_field_definitions.csv", index=False, lineterminator="\n")  # argument of to_csv(): the output path of the field-definition table, Unix line ends
    print("\n[4] FIELD DEFINITIONS AND DIRECTIONS FIXED IN ADVANCE")  # heading
    print(f"      {'receiver':<11} {'field':<24} {'units':<34} {'under interference'}")  # table header
    for _, r in fd.iterrows():  # for loop over fields
        # one row per field
        print(f"      {r.receiver:<11} {r.field:<24} {r.units:<34} "
              f"{r.direction_under_interference}")  # f-string continued: the direction in which the field moves under interference
    print("      Full definitions with document, version and section are written to")  # fixed text
    print("      receiver_field_definitions.csv.")  # continued

    # ---- tag -------------------------------------------------------------
    # tags every distinct time from both receiver families (pd.concat stacks the two time columns)
    # shared tagging: the handset script's fall-through to "other above 1 W" and the excluded unknown-power group
    tags = tag_times(pd.concat([sept.timestamp_utc, ubx.timestamp_utc]), plan, sessions, HIGH_POWER_W, FIFTY_W_MIN)
    sept = sept.merge(tags, on="timestamp_utc", how="left")  # left join: tag columns added by time
    ubx = ubx.merge(tags, on="timestamp_utc", how="left")  # likewise

    # the group labels G50, GBL, GHI and GUNK now come from the helpers module (one definition for the handset and receiver-status scripts)
    print("      exposure groups (epoch-times, both receiver families): "  # count of distinct tagged times per group
          + ", ".join(f"{g_}: {n_}" for g_, n_ in tags.exposure_group.value_counts().sort_index().items()))  # string concatenation: the number of tagged epoch times in each exposure group
    # melt: wide -> long, one row per (identity columns, field, value) for the Septentrio test fields
    sept_long = sept.melt(
        id_vars=["test_code", "receiver", "receiver_family", "timestamp_utc", "band",  # keyword argument id_vars=: the identity columns kept on every long row
                 "exposure_group", "inside_our_capture"],
        value_vars=[c for c in SEPT_TEST_FIELDS if c in sept.columns],  # keyword argument value_vars=: the Septentrio fields under test that the table holds
        var_name="field", value_name="value")  # keyword arguments of melt(): the new columns that hold the field name and its value
    # the same for the u-blox test fields
    ubx_long = ubx.melt(
        id_vars=["test_code", "receiver", "receiver_family", "timestamp_utc", "band",  # keyword argument id_vars=: the identity columns kept on every long row
                 "exposure_group", "inside_our_capture"],
        value_vars=[c for c in UBX_TEST_FIELDS if c in ubx.columns],  # keyword argument value_vars=: the u-blox fields under test that the table holds
        var_name="field", value_name="value")  # keyword arguments of melt(): the new columns that hold the field name and its value
    long = pd.concat([sept_long, ubx_long], ignore_index=True)  # one long table, index renumbered
    long["value"] = pd.to_numeric(long["value"], errors="coerce")  # numbers; None/text -> NaN
    # a code that means "unknown or feature disabled" is not a level and is excluded before any comparison.
    # u-blox jamming_state 0 is such a code (UBX_DOC MON-RF flags). The field is not among UBX_TEST_FIELDS, so this
    # guard removes nothing today; it keeps the rule true if the field is ever tested.
    UNKNOWN_CODE = {"jamming_state": 0}  # dict: field -> its "unknown" code
    for fld_, code_ in UNKNOWN_CODE.items():  # for loop over the fields that have such a code
        long.loc[(long.field == fld_) & (long.value == code_), "value"] = np.nan  # .loc mask: set to NaN, dropped next
    long = long[long.value.notna()]  # drops missing values (e.g. PLL-unlocked gain)

    # ---- 3a group comparison ---------------------------------------------
    print("\n" + "=" * 78)  # rule
    print("[5] TEST A - GROUP COMPARISON, 50 W windows against baseline")  # heading
    print("=" * 78)  # rule
    print("  Never pooled across receivers. Direction is fixed in advance per field;")  # fixed text
    print("  a difference in the opposite direction is evidence AGAINST.")  # continued
    cdf = group_comparison(long)  # Test A, one row per receiver, band and field (function shared with the schedule confirmation)
    cdf.to_csv(PROCESSED / "ground_truth" / "receiver_level_comparison.csv", index=False, lineterminator="\n")  # written
    # table header
    print(f"\n      {'receiver':<14} {'band':<12} {'field':<16} {'med 50W':>9} "
          f"{'med base':>9} {'diff':>8} {'Cliff d':>8} {'expected':>9} {'match':>6}")  # f-string continued: header cells for the baseline median, difference, effect size and verdict
    for _, r in cdf.iterrows():  # for loop over rows
        # one printed row
        print(f"      {r.receiver:<14} {r.band:<12} {r.field:<16} {r.median_50w:>9.2f} "
              f"{r.median_baseline:>9.2f} {r.difference:>+8.2f} {r.cliffs_delta:>8.3f} "  # f-string continued: the baseline median, the difference and Cliff's delta
              f"{r.expected_direction:>9} {'yes' if r.matches_prediction else 'no':>6}")  # f-string continued: the expected direction and whether the result matches it
    n_match = int(cdf.matches_prediction.sum())  # contrasts in the predicted direction
    big = cdf[(cdf.matches_prediction) & (cdf.cliffs_delta.abs() > 0.3)]  # of those, |delta| above 0.3
    # summary line
    print(f"\n  {n_match} of {len(cdf)} contrasts move in the direction fixed in advance; "
          f"{len(big)} of those exceed an effect size of 0.30.")  # f-string continued: how many of those exceed an effect size of 0.30

    # ---- 3b transition timing ---------------------------------------------
    print("\n" + "=" * 78)  # rule
    print("[6] TEST B - TRANSITION TIMING against the scheduled 50 W edges")  # heading
    print("=" * 78)  # rule
    edges = []  # (test code, "on"/"off", instant) for each 50 W TA1 test
    for p in plan:  # for loop over the schedule
        if p["power_w"] is None or p["power_w"] < FIFTY_W_MIN or p["test_area"] != 1:  # if: not 50 W TA1
            continue  # skip
        edges.append((p["test_code"], "on", pd.Timestamp(p["start_utc"])))  # switch-on
        edges.append((p["test_code"], "off", pd.Timestamp(p["end_utc"])))  # switch-off
    win = pd.Timedelta(minutes=5)  # search half-window
    # an edge is usable only if its whole +/- 5 min search window lies inside one capture (as in the handset script)
    caps = [(pd.Timestamp(r["capture_start_utc"], tz="UTC"), pd.Timestamp(r["capture_end_utc"], tz="UTC"))
            for _, r in sessions.iterrows()]  # list comprehension: (capture start, capture end) per session
    inside_edge = {(c_, k_, t_): any(a_ <= t_ - win and t_ + win <= b_ for a_, b_ in caps)  # dict: edge -> usable?
                   for c_, k_, t_ in edges}  # generator clause: one entry per scheduled switch-on or switch-off edge
    trows = []  # timing rows
    for (rx, band, fld), g in long.groupby(["receiver", "band", "field"]):  # per receiver, band, field
        fam = "Septentrio" if rx.startswith("Sept") else "u-blox"  # family
        d = DIRECTION.get((fam, fld), 0)  # direction
        if d == 0:  # if: no direction claimed
            continue  # skip
        g = g.sort_values("timestamp_utc")  # time order
        for code, kind, t in edges:  # for loop over edges; one outside a capture is timed but flagged by inside_edge
            s = g[(g.timestamp_utc >= t - win) & (g.timestamp_utc <= t + win)]  # values within +/- 5 min
            if len(s) < 120:  # if: too few values
                continue  # skip
            roll = s["value"].rolling(15, center=True, min_periods=5).median()  # centred 15-sample median
            diffs = roll.diff().to_numpy()  # step from one sample to the next
            if np.all(np.isnan(diffs)):  # if: nothing usable
                continue  # skip
            # boolean equality: look for a drop at switch-on of a decreasing field (or switch-off of an increasing one)
            want_drop = (d < 0) == (kind == "on")
            k = int(np.nanargmin(diffs)) if want_drop else int(np.nanargmax(diffs))  # index of the largest such step
            obs = pd.Timestamp(s["timestamp_utc"].to_numpy()[k])  # when it happened
            trows.append({"receiver": rx, "band": band, "field": fld,  # appends the row
                          "test_code": code, "edge": kind,  # dict entry: the test code and whether the edge is a switch-on or a switch-off
                          "scheduled_utc": t.strftime("%Y-%m-%d %H:%M:%S"),  # dict entry: when the plan says the edge happens
                          "observed_utc": obs.strftime("%Y-%m-%d %H:%M:%S"),  # dict entry: when the indicator shows its largest change near the edge
                          "offset_s": round((obs - t).total_seconds(), 1),  # observed minus scheduled, s
                          "inside_capture": inside_edge[(code, kind, t)]})  # True if the window lies in a capture
    tdf = pd.DataFrame(trows)  # the timing table
    tdf.to_csv(PROCESSED / "ground_truth" / "receiver_transition_timing.csv", index=False, lineterminator="\n")  # written
    if tdf.empty:  # if: no edge testable
        print("  No scheduled edge falls inside a recording with enough margin to test.")  # says so
    else:  # else: summarise
        ti = tdf[tdf.inside_capture]  # only edges whose search window lies wholly inside a capture
        near = int((ti.offset_s.abs() <= 30).sum())  # within 30 s of schedule, usable edges only
        n_edges = ti.groupby(["test_code", "edge"]).ngroups  # distinct usable edges
        n_series = ti.groupby(["receiver", "band", "field"]).ngroups  # distinct receiver-band-field series
        # counts, all edges and usable edges
        print(f"  {len(tdf)} edge contrasts computed; {len(ti)} come from {n_edges} edge(s) whose +/- 5 min "
              f"window lies inside a capture, seen by {n_series} receiver-band-field series.")  # f-string continued: how many receiver, band and field series see those edges
        print(f"  Largest change within 30 s of the scheduled instant: {near} of {len(ti)} usable contrasts.")  # count
        # withdrawn as evidence; the dependence is stated instead of a chance comparison
        print(f"  WITHDRAWN AS EVIDENCE: the {len(ti)} contrasts are {n_edges} edge(s) seen many times over by")
        print("  series of the same receivers, so they are not independent, and no chance rate is quoted.")  # continued
        for (c_, k_), g_ in ti.groupby(["test_code", "edge"]):  # per usable edge
            # one line per edge: how many series, how many near the schedule, the median |offset|
            print(f"      {c_} {k_:<3} {g_.scheduled_utc.iloc[0]}  series {len(g_):>3}  within 30 s "
                  # f-string continued: series within 30 s of the schedule, and the median absolute offset
                  f"{int((g_.offset_s.abs() <= 30).sum()):>3}  median |offset| {g_.offset_s.abs().median():.0f} s")

    # ---- 3c pyramid --------------------------------------------------------
    print("\n" + "=" * 78)  # rule
    print("[7] TEST C - THE 1.8.2 PYRAMID")  # heading
    print("=" * 78)  # rule
    # the parsed structure
    print(f"  Structure from the plan: {pyr['n_steps']} steps of {pyr['step_min']} min, "
          f"{pyr['jam_min']} min jamming then {pyr['brk_min']} min break.")  # f-string continued: the jamming and break minutes of each step
    print("  The Septentrio resolves front-ends for SEVEN of the eight pyramid bands,")  # fixed text
    print("  against four on the handsets. Predicted steps per band:")  # continued
    for nm, fe in PLAN_BAND_TO_FRONTEND.items():  # for loop over plan bands and front-end codes
        a = pyr["predicted_by_band"][nm]  # predicted steps (shared parser's key)
        # band, front end and step range; FRONTEND.get(fe, '?') falls back to "?"
        print(f"      {nm:<4} (front-end {fe:>2}, {FRONTEND.get(fe,'?'):<12}) "
              f"steps {min(a)}-{max(a)}")  # f-string continued: the pyramid steps in which the band should be jammed
    # two statements: the pyramid's scheduled start and end
    p0 = pd.Timestamp(pyr["entry"]["start_utc"]); p1 = pd.Timestamp(pyr["entry"]["end_utc"])
    ins = long[(long.timestamp_utc >= p0) & (long.timestamp_utc < p1)].copy()  # values inside the pyramid
    prows = []  # pyramid rows
    if not ins.empty:  # if: data inside the window
        off = (ins.timestamp_utc - p0).dt.total_seconds()  # seconds since the pyramid started (vectorised .dt)
        ins["step"] = (off // (pyr["step_min"] * 60)).astype(int) + 1  # floor division by 300 s, plus 1: step 1-15
        # "%": seconds into the step; True in the first 180 s (the first 3 of 5 minutes, not a third)
        ins["jamming_third"] = (off % (pyr["step_min"] * 60)) < pyr["jam_min"] * 60
        # dict comprehension: front-end name (e.g. "GPSL1/E1") -> plan band name (e.g. "L1")
        code_to_name = {FRONTEND[fe]: nm for nm, fe in PLAN_BAND_TO_FRONTEND.items()
                        if fe in FRONTEND}  # generator clause: keeps plan bands whose front-end code is known
        # table header
        print(f"\n      {'receiver':<14} {'band':<12} {'plan':<5} {'pred steps':<11} "
              f"{'n jam':>7} {'n brk':>7} {'med jam':>9} {'med brk':>9} {'jam-brk':>9} "  # f-string continued: header cells for counts and medians in jamming and break minutes
              f"{'Cliff d':>8} {'match':>6}")  # f-string continued: header cells for the effect size and the verdict
        for (rx, band, fld), g in ins.groupby(["receiver", "band", "field"]):  # per receiver, band, field
            nm = code_to_name.get(band)  # plan name of this Septentrio band (u-blox bands "L1" do not match)
            if nm is None:  # if: not a pyramid band (so u-blox and Aux bands drop out)
                continue  # skip
            fam = "Septentrio" if rx.startswith("Sept") else "u-blox"  # family
            d = DIRECTION.get((fam, fld), 0)  # direction
            if d == 0:  # if: none
                continue  # skip
            pr = pyr["predicted_by_band"][nm]  # predicted steps (shared parser's key)
            s = g[g.step.between(min(pr), max(pr))]  # values in the predicted steps (inclusive)
            a = s.loc[s.jamming_third, "value"].to_numpy()  # jamming portions
            b = s.loc[~s.jamming_third, "value"].to_numpy()  # break portions
            if len(a) < 50 or len(b) < 50:  # if: too few
                continue  # skip
            diff = float(np.median(a) - np.median(b))  # jamming minus break
            dl = cliffs_delta(a, b)  # negative = lower while jamming
            match = (d < 0 and diff < 0) or (d > 0 and diff > 0)  # agrees with the direction?
            # the printed row; nested f-string for the step range
            print(f"      {rx:<14} {band:<12} {nm:<5} {f'{min(pr)}-{max(pr)}':<11} "
                  f"{len(a):>7} {len(b):>7} {np.median(a):>9.2f} {np.median(b):>9.2f} "  # f-string continued: the counts and medians in jamming and break minutes
                  f"{diff:>+9.2f} {dl:>8.3f} {'yes' if match else 'no':>6}")  # f-string continued: the difference, Cliff's delta and whether the prediction holds
            prows.append({"receiver": rx, "band": band, "plan_band": nm, "field": fld,  # appends the row
                          "predicted_first": min(pr), "predicted_last": max(pr),  # dict entry: the first and last pyramid step in which the band should be jammed
                          "n_jamming": len(a), "n_break": len(b),  # dict entry: readings in the jamming minutes and in the break minutes
                          "median_jamming": round(float(np.median(a)), 3),  # dict entry: median in the jamming minutes
                          "median_break": round(float(np.median(b)), 3),  # dict entry: median in the break minutes
                          "jamming_minus_break": round(diff, 3),  # dict entry: jamming median minus break median
                          "cliffs_delta": round(dl, 4),  # dict entry: Cliff's delta of the jamming readings against the break readings
                          "matches_prediction": bool(match)})  # dict entry: whether the difference has the predicted sign
    pdf = pd.DataFrame(prows)  # the pyramid table
    pdf.to_csv(PROCESSED / "ground_truth" / "receiver_pyramid_observed.csv", index=False, lineterminator="\n")  # written
    n_pyr_match = int(pdf.matches_prediction.sum()) if not pdf.empty else 0  # predicted-direction count
    # of those, |delta| above 0.3 (conditional expression guards an empty table)
    pyr_big = pdf[(pdf.matches_prediction) & (pdf.cliffs_delta.abs() > 0.3)] if not pdf.empty else pdf
    # summary line
    print(f"\n  {n_pyr_match} of {len(pdf)} contrasts move in the predicted direction; "
          f"{len(pyr_big)} exceed an effect size of 0.30.")  # f-string continued: how many of those exceed an effect size of 0.30

    # ---- positive control ---------------------------------------------------
    print("\n" + "=" * 78)  # rule
    print("[8] POSITIVE CONTROL - our own spoofing, per session, not pooled")  # heading
    print("=" * 78)  # rule
    crows = []  # control rows
    csens = []  # unrounded direction, effect size, difference and scale per contrast, for the rule grid
    for _, sr in sessions.iterrows():  # for loop over our six sessions
        ss = pd.Timestamp(sr["scheduled_start_utc"], tz="UTC")  # scheduled start
        se = pd.Timestamp(sr["scheduled_end_utc"], tz="UTC")  # scheduled end
        cs = pd.Timestamp(sr["capture_start_utc"], tz="UTC")  # capture start (unused below)
        ce = pd.Timestamp(sr["capture_end_utc"], tz="UTC")  # capture end (unused below)
        sub = long[(long.test_code == sr["test_code"])]  # this session's values
        for (rx, band, fld), g in sub.groupby(["receiver", "band", "field"]):  # per receiver, band, field
            fam = "Septentrio" if rx.startswith("Sept") else "u-blox"  # family
            d = DIRECTION.get((fam, fld), 0)  # direction
            if d == 0:  # if: none
                continue  # skip
            a = g.loc[(g.timestamp_utc >= ss) & (g.timestamp_utc < se), "value"].to_numpy()  # inside the window
            b = g.loc[(g.timestamp_utc < ss) | (g.timestamp_utc >= se), "value"].to_numpy()  # in the margins
            if len(a) < 60 or len(b) < 60:  # if: too few
                continue  # skip
            diff = float(np.median(a) - np.median(b))  # inside minus margins
            dl = cliffs_delta(a, b)  # negative = inside tends lower
            scale = iqr(b) if iqr(b) > 1e-9 else 1.0  # the margins' IQR, or 1 if it is ~0
            # kept for the sensitivity grid: does the difference go the documented way, and its size
            csens.append({"test_code": sr["test_code"], "dir_ok": (d < 0 and diff < 0) or (d > 0 and diff > 0),
                          "abs_delta": abs(dl), "abs_diff": abs(diff), "scale": scale})  # dict entry: the size of the effect and of the difference, and the margin spread that scales it
            # detection rule: expected direction AND |delta| > 0.5 AND |difference| > 3 margin IQRs
            det = bool(((d < 0 and diff < 0) or (d > 0 and diff > 0))
                       and abs(dl) > 0.5 and abs(diff) > 3 * scale)  # condition continued: and the effect is large and the shift exceeds three margin spreads
            crows.append({"test_code": sr["test_code"], "receiver": rx, "band": band,  # appends the row
                          "field": fld, "n_inside": len(a), "n_outside": len(b),  # dict entry: the field, and the readings inside the scheduled test and outside it
                          "median_inside": round(float(np.median(a)), 3),  # dict entry: median inside the scheduled test
                          "median_outside": round(float(np.median(b)), 3),  # dict entry: median outside it
                          "inside_minus_outside": round(diff, 3),  # dict entry: inside median minus outside median
                          "cliffs_delta": round(dl, 4), "deflection_detected": det})  # dict entry: Cliff's delta, and whether the detection rule calls it a deflection
    ctrl = pd.DataFrame(crows)  # the control table
    ctrl.to_csv(PROCESSED / "ground_truth" / "receiver_positive_control.csv", index=False, lineterminator="\n")  # written
    per_sess = ctrl.groupby("test_code").deflection_detected.agg(["sum", "count"])  # per session: hits, contrasts
    for code in sessions.test_code:  # for loop in session order
        if code in per_sess.index:  # if: the session has contrasts
            r = per_sess.loc[code]  # its row
            sub = ctrl[(ctrl.test_code == code) & ctrl.deflection_detected]  # its hits
            # conditional expression: largest |difference| and its field, or "none"
            worst = (f"largest {sub.inside_minus_outside.abs().max():.1f} "
                     f"in {sub.loc[sub.inside_minus_outside.abs().idxmax(),'field']}"  # f-string continued: the field in which that largest difference occurs
                     if not sub.empty else "none")  # conditional expression: none when no contrast exists for the session
            # the per-session line
            print(f"      {code:<7} {int(r['sum']):>3} of {int(r['count']):>3} contrasts "
                  f"show a deflection   ({worst})")  # f-string continued: the largest difference and its field, in brackets
    control_passes = bool(ctrl.deflection_detected.any())  # passes if any contrast detected
    n_sess_ctrl = int(ctrl[ctrl.deflection_detected].test_code.nunique())  # sessions with a hit
    # the session count
    print(f"\n  Sessions in which the primary receivers register our own spoofing: "
          f"{n_sess_ctrl} of {len(sessions)}.")  # f-string continued: the number of such sessions out of all six
    # the handset script's handset result read from its own table instead of typed
    hpc = pd.read_csv(PROCESSED / "ground_truth" / "handset_positive_control.csv")  # the handset script's positive-control table
    hs_ = sorted(hpc[hpc.deflection_detected].our_test_code.astype(str).unique())  # sessions with a handset hit
    # the sessions in which the handsets registered our spoofer, read from the handset script's table
    print(f"  the handset script found a deflection on the handsets in {len(hs_)} session(s): {', '.join(hs_) or 'none'}.")
    # the rule's sensitivity - effect-size threshold 0.3 / 0.5 / 0.7 against IQR multiple 2 / 3 / 4
    cs_df = pd.DataFrame(csens)  # one row per contrast, unrounded
    order_ = list(sessions.test_code)  # sessions in capture order, for the listing
    grid = []  # one row per rule
    for dthr in (0.3, 0.5, 0.7):  # for loop over effect-size thresholds
        for kmul in (2, 3, 4):  # for loop over IQR multiples
            hit = cs_df[cs_df.dir_ok & (cs_df.abs_delta > dthr) & (cs_df.abs_diff > kmul * cs_df.scale)]  # detected
            ss_ = [c_ for c_ in order_ if c_ in set(hit.test_code)]  # sessions detected, in capture order
            # appends one rule's row: thresholds, sessions and contrasts detected, and whether it is the rule in use
            grid.append({"delta_threshold": dthr, "iqr_multiple": kmul, "n_sessions_detected": len(ss_),
                         "sessions_detected": ";".join(ss_), "n_contrasts_detected": len(hit),  # dict entry: the detected sessions by name, and the number of contrasts detected
                         "n_contrasts": len(cs_df), "rule_in_use": dthr == 0.5 and kmul == 3})  # the row
    gdf = pd.DataFrame(grid)  # the grid
    gdf.to_csv(PROCESSED / "ground_truth" / "receiver_control_sensitivity.csv", index=False, lineterminator="\n")  # written (new)
    print("\n  Rule sensitivity: |delta| threshold x IQR multiple -> sessions detected")  # heading
    for r in gdf.itertuples():  # one line per rule
        # one printed line per rule: sessions detected out of six, with the rule in use marked
        print(f"      {r.delta_threshold:.1f} x {r.iqr_multiple}: {r.n_sessions_detected} of {len(sessions)} "
              f"({r.sessions_detected.replace(';', ', ')}){'   <- rule in use' if r.rule_in_use else ''}")  # f-string continued: the detected sessions, with the rule in use marked

    # ---- 1.1.29 confound -----------------------------------------------------
    print("\n" + "=" * 78)  # rule
    print("[9] THE 1.1.29 CONFOUND - does the deflection begin at 2.6.12 or at 1.1.29?")  # heading
    print("=" * 78)  # rule
    j = [p for p in plan if p["test_code"] == "1.1.29"]  # the 1.1.29 plan entry, as a list
    s12 = site.loc["2.6.12"]  # the recording site of 2.6.12, the session 1.1.29 overlaps (measured)
    d1129 = haversine_m(s12.median_lat, s12.median_lon, *TA2_ORIG) / 1000  # that site to the TA2 origin, km
    # the assumed location, from the constant rather than typed
    print(f"  Test Area 2 is taken to be at survey point ORIG ({TA2_ORIG[0]} N, "
          f"{TA2_ORIG[1]} E),")  # f-string continued: the survey point's longitude
    # the two distances, both from the 2.6.12 recording site (1)
    print(f"  {d1129:.2f} km from the 2.6.12 recording site, against {s12.dist_to_sender_km:.2f} km "
          "for Test Area 1.")
    print("  ASSUMPTION, not a measurement: 1.1.29 is a handheld jammer and handheld")  # caveat
    print("  jammers move. ORIG is the surveyed origin of Test Area 2 and is the best")  # continued
    print("  available proxy for where it was, not its transmitter location.")  # continued
    # records the assumption ("3.17 km" typed)
    note("The location of test 1.1.29 is an assumption. Appendix A gives survey point "
         f"ORIG as the origin of Test Area 2, {d1129:.2f} km from the 2.6.12 recording site, but 1.1.29 is a "  # f-string continued: the computed distance from the 2.6.12 recording site
         "handheld jammer and its actual position during the test is unknown.")
    drows = []  # decomposition rows
    if j:  # if: 1.1.29 is in the plan
        js, je = pd.Timestamp(j[0]["start_utc"]), pd.Timestamp(j[0]["end_utc"])  # its window
        sr = sessions[sessions.test_code == "2.6.12"].iloc[0]  # the 2.6.12 session
        ss = pd.Timestamp(sr["scheduled_start_utc"], tz="UTC")  # 2.6.12 scheduled start
        se = pd.Timestamp(sr["scheduled_end_utc"], tz="UTC")  # 2.6.12 scheduled end
        cs = pd.Timestamp(sr["capture_start_utc"], tz="UTC")  # capture start
        # list of (label, start, end): the four intervals built from overlaps (max of starts, min of ends)
        iv = [("A: 1.1.29 alone", max(js, cs), min(ss, je)),
              ("B: 1.1.29 and 2.6.12", max(js, ss), min(je, se)),  # list element: interval B, when the jammer and the 2.6.12 test overlap
              ("C: 2.6.12 alone", max(je, ss), se),  # list element: interval C, from the jammer's end to the 2.6.12 test's end
              ("D: neither (pre-test margin)", cs, min(js, ss))]  # list element: interval D, the pre-test margin before either transmits, the baseline
        # the windows
        print(f"\n  1.1.29 scheduled {js:%H:%M}-{je:%H:%M} UTC; 2.6.12 scheduled "
              f"{ss:%H:%M}-{se:%H:%M} UTC; capture starts {cs:%H:%M:%S}.")  # f-string continued: the 2.6.12 schedule and the capture start
        # table header
        print(f"\n      {'interval':<30} {'window (UTC)':<16} {'receiver':<14} "
              f"{'band':<12} {'field':<16} {'median':>9} {'vs D':>9}")  # f-string continued: header cells for band, field, median and difference to interval D
        sub = long[(long.test_code == "2.6.12")]  # 2.6.12 values
        for (rx, band, fld), g in sub.groupby(["receiver", "band", "field"]):  # per receiver, band, field
            fam = "Septentrio" if rx.startswith("Sept") else "u-blox"  # family
            if DIRECTION.get((fam, fld), 0) == 0:  # if: no direction
                continue  # skip
            if band not in ("GPSL1/E1", "L1"):  # if: not the L1 band (Septentrio or u-blox name)
                continue  # skip
            # interval D's values: iv[3][1] and iv[3][2] are its start and end
            base = g.loc[(g.timestamp_utc >= iv[3][1]) & (g.timestamp_utc < iv[3][2]),
                         "value"].to_numpy()  # tuple element: the value column of the rows inside interval D, as an array
            if len(base) < 20:  # if: too few
                continue  # skip
            for lab, a0, a1 in iv:  # for loop over the four intervals
                # the interval's values
                v = g.loc[(g.timestamp_utc >= a0) & (g.timestamp_utc < a1),
                          "value"].to_numpy()  # tuple element: the value column of the rows inside this interval, as an array
                if len(v) < 20:  # if: too few
                    continue  # skip
                dv = float(np.median(v) - np.median(base))  # median minus interval D's median
                # the printed row
                print(f"      {lab:<30} {a0:%H:%M}-{a1:%H:%M}    {rx:<14} {band:<12} "
                      f"{fld:<16} {np.median(v):>9.2f} {dv:>+9.2f}")  # f-string continued: the field, its median and the difference to interval D
                drows.append({"interval": lab, "start_utc": a0.strftime("%H:%M:%S"),  # appends the row
                              "end_utc": a1.strftime("%H:%M:%S"), "receiver": rx,  # dict entry: interval end and the receiver
                              "band": band, "field": fld, "n": len(v),  # dict entry: band, field and the number of readings
                              "median": round(float(np.median(v)), 3),  # dict entry: median value over the interval
                              "minus_interval_D": round(dv, 3)})  # dict entry: the median minus that of interval D, the baseline
    ddf = pd.DataFrame(drows)  # the decomposition table
    ddf.to_csv(PROCESSED / "ground_truth" / "jammer_1129_decomposition.csv", index=False, lineterminator="\n")  # written
    verdict_1129 = None  # which start the deflection follows; set below and printed in the verdict
    if not ddf.empty:  # if: rows exist
        # the rule is applied per receiver and field, so dB and u-blox counts are never pooled
        print("\n  Largest absolute departure from the quiet pre-test margin, per receiver and field:")  # sub-heading
        # table header for the per-field 1.1.29 departures
        print(f"      {'receiver':<14} {'field':<14} {'A alone':>9} {'B both':>9} {'C 2.6.12':>9} {'A / max(B,C)':>13}")
        ratios = []  # A / max(B, C) per receiver and field
        for (rx_, fld_), g_ in ddf.groupby(["receiver", "field"]):  # per receiver and field
            # dict comprehension: interval letter -> largest |departure| (NaN when the interval has no row)
            m_ = {iv_: g_[g_.interval.str.startswith(iv_)].minus_interval_D.abs().max() for iv_ in "ABC"}
            den = max(m_["B"], m_["C"])  # the larger of the two intervals that contain 2.6.12
            r_ = m_["A"] / den if den > 0 else float("nan")  # the ratio, undefined when neither moves
            ratios.append(r_)  # kept for the rule
            print(f"      {rx_:<14} {fld_:<14} {m_['A']:>9.2f} {m_['B']:>9.2f} {m_['C']:>9.2f} {r_:>13.3f}")  # the row
        ok_ = [r_ for r_ in ratios if not np.isnan(r_)]  # the defined ratios
        A = max(ok_) if ok_ else float("nan")  # the largest ratio decides: every field must pass
        if ok_ and A < 0.2:  # if: in every field interval A is under a fifth of the larger of B and C
            verdict_1129 = "2.6.12"  # the deflection follows our test
            print("\n  The deflection begins at the 2.6.12 scheduled start, NOT at 1.1.29.")  # conclusion
            print("  the handset script's attribution to our own spoofer is CORRECT, and 1.1.29")  # continued
            print(f"  becomes a SECOND independent negative test: a jammer at {d1129:.1f} km")  # continued
            print("  produced no measurable response either.")  # continued
            # records it
            note(f"Test 1.1.29, a jammer at an assumed {d1129:.2f} km, produced no "
                 f"measurable response in any primary receiver during the {(iv[0][2] - iv[0][1]).total_seconds() / 60:.0f} "  # f-string continued: how many minutes the jammer transmitted alone
                 "minutes it transmitted alone. It is a second independent negative test, and it "
                 "confirms that the deflection seen in the handset script belongs to our own "
                 "spoofer and not to it.")
        elif not ok_:  # elif: no field moved in B or C, so no attribution is possible
            verdict_1129 = "undetermined"  # records that, without claiming either start
            print("\n  No field departs in interval B or C; the attribution cannot be made.")  # says so
        else:  # else: in at least one field the deflection already shows in interval A
            verdict_1129 = "1.1.29"  # records that
            # flags the handset script's attribution as wrong
            flag("THE DEFLECTION ATTRIBUTED BY THE HANDSET SCRIPT TO OUR OWN SPOOFER BEGINS AT THE 1.1.29 START, NOT AT THE 2.6.12 START. The handset script's attribution "
                 "is wrong and must be revised.")
            print("\n  *** The deflection begins at the 1.1.29 start. the handset script's")  # says so
            print("  *** attribution is WRONG and must be revised.")  # continued

    # ---- effective attack interval -------------------------------------------
    print("\n" + "=" * 78)  # rule
    print("[10] EFFECTIVE ATTACK INTERVAL - measurement, not labelling")  # heading
    print("=" * 78)  # rule
    print("  CONFIDENCE LIMIT, stated before the numbers. A spoofer that matches the")  # caveat, fixed text
    print("  authentic signal power can capture a receiver without moving its gain")  # continued
    print("  control at all. Absence of a deflection is therefore NOT evidence of")  # continued
    print("  absence of spoofing. What follows bounds when the receiver's front end was")  # continued
    print("  disturbed, not when it was deceived. Nothing here is a label.")  # continued
    print("\n  WHAT IS REPORTED, and why it changed. The primary quantities are the")  # method, fixed text
    print("  maximum excursion depth against a quiet reference and the longest")  # continued
    print("  contiguous disturbed span, because both can be read straight off the trace")  # continued
    print("  and checked by eye. The fraction is secondary, is taken against CAPTURED")  # continued
    print("  in-window time rather than the scheduled window, and is never quoted")  # continued
    print("  without the capture coverage beside it. An earlier version divided by the")  # continued
    print("  scheduled window, which capped the answer at the capture coverage (the")  # continued (no typed value)
    print("  coverage column below) before any threshold applied.")  # continued
    # the G5's logging measured here from its own status records instead of typed: the share of capture
    # seconds holding at least one record and the gaps longer than 1.5 s, per session, beside the mosaic-X5
    cov_ = {}  # receiver -> list of (share of seconds covered, gaps over 1.5 s) per session
    for rx_ in ("Septentrio_01", "Septentrio_02"):  # the G5 and, for comparison, the X5
        for _, sr in sessions.iterrows():  # for loop over sessions
            # boolean mask: this receiver's L1 gain records in this session; drop_duplicates keeps one per time
            tt_ = long[(long.receiver == rx_) & (long.test_code == sr["test_code"]) & (long.band == "GPSL1/E1")
                       & (long.field == "agc_gain_db")].timestamp_utc.drop_duplicates().sort_values()  # record times
            # capture length in seconds
            dur_ = (pd.Timestamp(sr["capture_end_utc"]) - pd.Timestamp(sr["capture_start_utc"])).total_seconds()
            if len(tt_) and dur_ > 0:  # if: any records
                # tuple: share of capture seconds with a record, and the number of gaps over 1.5 s
                cov_.setdefault(rx_, []).append((tt_.dt.floor("s").nunique() / dur_,
                                                 int((tt_.diff().dt.total_seconds() > 1.5).sum())))  # tuple element: the number of gaps longer than 1.5 s between records
    g5c, x5c = cov_.get("Septentrio_01", [(0, 0)]), cov_.get("Septentrio_02", [(0, 0)])  # the two lists
    # the excluded receiver, named from the constant
    print(f"\n  The mosaic-G5 ({', '.join(sorted(EXCLUDE_FROM_INTERVAL))}) is EXCLUDED from")
    # the G5's measured share of covered seconds (minimum and maximum over sessions)
    print(f"  this metric and reported separately below: its status records cover "
          f"{100 * min(c[0] for c in g5c):.0f} to {100 * max(c[0] for c in g5c):.0f} per")  # measured
    # the G5's measured count of gaps over 1.5 s per session
    print(f"  cent of the seconds of each capture, with {min(c[1] for c in g5c)} to {max(c[1] for c in g5c)} gaps "
          f"longer than 1.5 s per session")  # measured
    # the same two measures for the mosaic-X5, for comparison
    print(f"  (mosaic-X5: {100 * min(c[0] for c in x5c):.0f} to {100 * max(c[0] for c in x5c):.0f} per cent, "
          f"{min(c[1] for c in x5c)} to {max(c[1] for c in x5c)} gaps), so a contiguous-span rule on that stream")  # f-string continued: the range of gap counts for the mosaic-X5
    print("  measures its logging bottleneck rather than the radio environment.")  # continued

    # Quiet reference per (receiver, band, field): the median of the per-session
    # margin medians, which resists a margin that is itself contaminated.
    refs: dict = {}  # (receiver, band, field) -> quiet reference value
    permargin = defaultdict(list)  # (receiver, band, field) -> list of per-session margin summaries
    for _, sr in sessions.iterrows():  # for loop over sessions
        ss = pd.Timestamp(sr["scheduled_start_utc"], tz="UTC")  # scheduled start
        se = pd.Timestamp(sr["scheduled_end_utc"], tz="UTC")  # scheduled end
        cs = pd.Timestamp(sr["capture_start_utc"], tz="UTC")  # capture start
        ce = pd.Timestamp(sr["capture_end_utc"], tz="UTC")  # capture end
        # per receiver, band and field of this session (the groupby call continues on the next line)
        for (rx, band, fld), g in long[long.test_code == sr["test_code"]].groupby(
                ["receiver", "band", "field"]):  # argument of groupby(): one group per receiver, band and field
            if fld not in DROP_THRESHOLD:  # if: not one of the three thresholded fields
                continue  # skip
            ser = g.set_index("timestamp_utc")["value"].sort_index()  # a time-indexed series
            m = margin_slices(ser, cs, ss, se, ce)  # its pre- and post-test margins
            if len(m) >= 20:  # if: at least 20 margin values
                # appends (session, margin median, margin 1st percentile); the percentile is never used
                permargin[(rx, band, fld)].append((sr["test_code"], float(np.nanmedian(m)),
                                                   float(np.nanpercentile(m, 1))))  # tuple element: the margin's 1st percentile, kept but never used
    for k, v in permargin.items():  # for loop over keys and their session lists
        refs[k] = float(np.median([x[1] for x in v]))  # reference = median of the per-session margin medians

    erows, traces = [], {}  # interval rows, and the traces kept for the figure
    for _, sr in sessions.iterrows():  # for loop over sessions
        code = sr["test_code"]  # test code
        ss = pd.Timestamp(sr["scheduled_start_utc"], tz="UTC")  # scheduled start
        se = pd.Timestamp(sr["scheduled_end_utc"], tz="UTC")  # scheduled end
        cs = pd.Timestamp(sr["capture_start_utc"], tz="UTC")  # capture start
        ce = pd.Timestamp(sr["capture_end_utc"], tz="UTC")  # capture end
        cap_lo, cap_hi = max(cs, ss), min(ce, se)  # the captured part of the scheduled window
        cap_min = max((cap_hi - cap_lo).total_seconds() / 60.0, 0.0)  # its length in minutes, never negative
        sched_min = (se - ss).total_seconds() / 60.0  # scheduled length in minutes
        # per receiver, band and field of this session
        for (rx, band, fld), g in long[long.test_code == code].groupby(
                ["receiver", "band", "field"]):  # argument of groupby(): one group per receiver, band and field
            if fld not in DROP_THRESHOLD or rx in EXCLUDE_FROM_INTERVAL:  # if: unthresholded field, or the G5
                continue  # skip
            fam = "Septentrio" if rx.startswith("Sept") else "u-blox"  # family
            d_ = DIRECTION.get((fam, fld), 0)  # direction
            if d_ == 0 or band not in ("GPSL1/E1", "L1"):  # if: no direction, or not the L1 band
                continue  # skip
            ref = refs.get((rx, band, fld))  # quiet reference
            if ref is None or cap_min <= 0:  # if: no reference or no captured window
                continue  # skip
            ser = g.set_index("timestamp_utc")["value"].sort_index()  # time-indexed series
            ins = ser[(ser.index >= cap_lo) & (ser.index < cap_hi)]  # inside the captured window
            if len(ins) < 30:  # if: too few
                continue  # skip
            runs = disturbed_runs(ins, ref, DROP_THRESHOLD[fld], d_)  # disturbed spans inside the window
            # The detector is also run over the WHOLE capture, because a passage
            # need not fall inside the scheduled window: in 2.6.5 the deepest
            # excursion lies in the post-test margin.
            whole = ser[(ser.index >= cs) & (ser.index <= ce)]  # the whole capture
            wruns = disturbed_runs(whole, ref, DROP_THRESHOLD[fld], d_)  # spans over the whole capture
            # longest whole-capture span in minutes; max(..., default=0.0) handles an empty list
            w_longest = max([(pd.Timestamp(b_) - pd.Timestamp(a_)).total_seconds() / 60.0
                             for a_, b_ in wruns], default=0.0)  # generator clause: runs over the disturbed spans of the whole capture
            # count of whole-capture spans that overlap the captured window
            in_window = sum(1 for a_, b_ in wruns
                            if pd.Timestamp(a_) < cap_hi and pd.Timestamp(b_) > cap_lo)  # generator clause: keeps whole-capture spans that overlap the captured part of the window
            # total in-window disturbed minutes
            tot = sum((pd.Timestamp(b_) - pd.Timestamp(a_)).total_seconds()
                      for a_, b_ in runs) / 60.0  # generator clause: sums the in-window spans, then converts seconds to minutes
            # longest in-window span in minutes
            longest = max([(pd.Timestamp(b_) - pd.Timestamp(a_)).total_seconds() / 60.0
                           for a_, b_ in runs], default=0.0)  # generator clause: runs over the in-window disturbed spans
            exc = float(ref - np.nanmin(ins)) if d_ < 0 else float(np.nanmax(ins) - ref)  # deepest in-window excursion
            marg = margin_slices(ser, cs, ss, se, ce)  # the margins
            # nested conditional expression: deepest margin excursion, or None when there is no margin
            marg_exc = (float(ref - np.nanmin(marg)) if len(marg) and d_ < 0
                        else (float(np.nanmax(marg) - ref) if len(marg) else None))  # conditional expression: the highest rise for a field that rises, or nothing without a margin
            erows.append({  # appends the interval row
                "test_code": code, "receiver": rx, "band": band, "field": fld,  # dict entry: session, receiver, band and field of the interval
                "quiet_reference": round(ref, 2),  # dict entry: the quiet reference level the excursion is measured from
                "max_excursion": round(exc, 2),  # dict entry: the largest excursion from the reference inside the window
                "longest_disturbed_span_min": round(longest, 2),  # dict entry: the longest disturbed span in minutes
                "total_disturbed_min": round(tot, 2),  # dict entry: the total disturbed minutes
                "n_disturbed_runs": len(runs),  # dict entry: the number of disturbed spans
                "captured_in_window_min": round(cap_min, 2),  # dict entry: minutes of the scheduled window that were captured
                "scheduled_window_min": round(sched_min, 2),  # dict entry: length of the scheduled window in minutes
                "capture_coverage_pct": round(100 * cap_min / sched_min, 1),  # captured share of the window
                "fraction_of_captured_in_window": round(tot / cap_min, 4),  # denominator: captured minutes
                "max_excursion_in_margin": (round(marg_exc, 2)  # dict entry: the largest excursion inside the margins, a check on the quiet reference
                                            if marg_exc is not None else None),  # conditional expression: empty when the session has no margin
                "n_passages_whole_capture": len(wruns),  # dict entry: the number of disturbed passages over the whole capture
                "n_passages_inside_window": in_window,  # dict entry: how many of those passages overlap the scheduled window
                "longest_passage_whole_capture_min": round(w_longest, 2)})  # dict entry: the longest passage over the whole capture, in minutes
            if rx == "Septentrio_02" and band == "GPSL1/E1" and fld == "agc_gain_db":  # if: the X5 L1 gain
                # keeps the trace (capture +/- 1 min) and the whole-capture detector result for the figure
                traces[code] = {"trace": ser[(ser.index >= cs - pd.Timedelta(minutes=1))
                                             & (ser.index <= ce + pd.Timedelta(minutes=1))],  # element-wise condition: up to one minute after the capture ends
                                "ref": ref, "runs": wruns, "excursion": exc,  # dict entry: the quiet reference, the disturbed spans and the largest excursion
                                "longest": w_longest, "n_passages": len(wruns),  # dict entry: the longest passage and the number of passages, printed on the panel
                                "coverage": 100 * cap_min / sched_min}  # dict entry: the captured share of the scheduled window as a percentage
    edf = pd.DataFrame(erows)  # the interval table
    edf.to_csv(PROCESSED / "ground_truth" / "effective_attack_interval.csv", index=False, lineterminator="\n")  # written

    prim = edf[(edf.receiver == "Septentrio_02") & (edf.field == "agc_gain_db")]  # the primary receiver's rows
    print("\n  PRIMARY QUANTITIES, Septentrio mosaic-X5, GPS L1/E1 front-end gain.")  # sub-heading
    # table header, first line
    print(f"      {'test':<8}{'excursion':>11}{'longest span':>14}{'total':>9}"
          f"{'captured':>10}{'coverage':>10}{'fraction':>10}")  # f-string continued: header cells for captured minutes, coverage and fraction
    # table header, units line
    print(f"      {'':<8}{'(dB)':>11}{'(min)':>14}{'(min)':>9}{'(min)':>10}"
          f"{'of window':>10}{'of capture':>10}")  # f-string continued: the unit cells under coverage and fraction
    for code in sessions.test_code:  # for loop in session order
        r = prim[prim.test_code == code]  # its row, as a one-row table
        if r.empty:  # if: none
            print(f"      {code:<8}{'no data':>11}")  # says so
            continue  # next session
        r = r.iloc[0]  # the row itself
        # the printed row
        print(f"      {code:<8}{r.max_excursion:>11.0f}{r.longest_disturbed_span_min:>14.2f}"
              f"{r.total_disturbed_min:>9.2f}{r.captured_in_window_min:>10.2f}"  # f-string continued: total disturbed minutes and captured minutes in the window
              f"{r.capture_coverage_pct:>9.0f}%{100*r.fraction_of_captured_in_window:>9.0f}%")  # f-string continued: captured share of the window, and disturbed share of the captured time
    print("\n  PASSAGES ACROSS THE WHOLE CAPTURE, not only the scheduled window.")  # sub-heading
    # table header
    print(f"      {'test':<8}{'passages':>10}{'of which in window':>20}{'longest (min)':>15}"
          f"{'where the deepest lies':>26}")  # f-string continued: the header cell saying where the deepest excursion lies
    for code in sessions.test_code:  # for loop in session order
        r = prim[prim.test_code == code]  # its row
        if r.empty:  # if: none
            continue  # skip
        r = r.iloc[0]  # the row itself
        # conditional expression: "in the margin" if the margin excursion exists and is deeper than in-window
        where = ("in the margin" if (r.max_excursion_in_margin is not None
                                     and not pd.isna(r.max_excursion_in_margin)  # condition continued: and the margin excursion is a number
                                     and r.max_excursion_in_margin > r.max_excursion)  # condition continued: and it is deeper than the excursion inside the window
                 else "inside the scheduled window")  # conditional expression: otherwise the deepest excursion lies inside the scheduled window
        # the printed row
        print(f"      {code:<8}{int(r.n_passages_whole_capture):>10}"
              f"{int(r.n_passages_inside_window):>20}"  # f-string continued: how many passages overlap the scheduled window
              f"{r.longest_passage_whole_capture_min:>15.2f}{where:>26}")  # f-string continued: the longest passage in minutes and where the deepest excursion lies

    if not prim.empty:  # if: primary rows exist
        # records the summary, computed: the old typed 'Five show a single passage' was wrong (2.6.4 and
        # 2.6.11 show two passages each); passages and coverage are now read from the table
        pc_ = prim.groupby("n_passages_whole_capture").test_code.apply(lambda s: ", ".join(s))  # sessions by passages
        # records the summary: passages per session and coverage read from Table the transfer script's rows
        note(f"Every one of the {len(prim)} sessions shows a substantial excursion in front-end gain; passages "
             "of a transmitter approaching and receding per session: "
             + "; ".join(f"{k} in {v}" for k, v in pc_.items())  # string concatenation: the number of passages in each session
             + f". The maximum excursion ranges from {prim.max_excursion.min():.0f} to "  # string concatenation: the start of the sentence on the range of maximum excursions
             f"{prim.max_excursion.max():.0f} dB. Fractions are quoted against captured "  # f-string continued: the largest excursion, and why fractions use captured time
             "in-window time and never against the scheduled window, because capture "
             f"coverage ranges from {prim.capture_coverage_pct.min():.0f} to {prim.capture_coverage_pct.max():.0f} "  # f-string continued: the range of capture coverage across sessions
             "per cent.")

    # Threshold sensitivity, so the reader can see the span is not a knife-edge.
    # the spread of the per-session quiet levels and the two thresholds, computed instead of typed
    qv_ = [x[1] for x in permargin.get(("Septentrio_02", "GPSL1/E1", "agc_gain_db"), [])]  # per-session margin medians
    spread_ = (max(qv_) - min(qv_)) if qv_ else float("nan")  # their range, dB
    dthr_ = DROP_THRESHOLD["agc_gain_db"]  # the entry threshold, dB
    # the per-session quiet-level spread, computed
    print(f"\n  THRESHOLD CALIBRATION. The per-session quiet level of this trace spans {spread_:.0f} dB,")
    print(f"  so a run is entered at {dthr_:.0f} dB below the reference and extended while the")  # continued
    # the exit threshold: the entry threshold times EXIT_FRACTION
    print(f"  departure exceeds {dthr_ * EXIT_FRACTION:.0f} dB. Longest span in minutes at other exit thresholds:")
    # table header
    print(f"\n      {'test':<8}{'exit 1 dB':>11}{'exit 2 dB':>11}{'exit 3 dB':>11}"
          f"{'exit 5 dB':>11}   (entry fixed at 5 dB)")  # f-string continued: the last header cell, with the entry threshold held fixed
    for _, sr in sessions.iterrows():  # for loop over sessions
        code = sr["test_code"]  # test code
        # this session's X5 L1 gain values
        sub = long[(long.test_code == code) & (long.receiver == "Septentrio_02")
                   & (long.band == "GPSL1/E1") & (long.field == "agc_gain_db")]  # element-wise condition: only the L1 front-end gain rows
        if sub.empty:  # if: none
            continue  # skip
        ref = refs.get(("Septentrio_02", "GPSL1/E1", "agc_gain_db"))  # its quiet reference
        ss = pd.Timestamp(sr["scheduled_start_utc"], tz="UTC")  # scheduled start
        se = pd.Timestamp(sr["scheduled_end_utc"], tz="UTC")  # scheduled end
        cs = pd.Timestamp(sr["capture_start_utc"], tz="UTC")  # capture start
        ce = pd.Timestamp(sr["capture_end_utc"], tz="UTC")  # capture end
        ser = sub.set_index("timestamp_utc")["value"].sort_index()  # time-indexed series
        ins = ser[(ser.index >= max(cs, ss)) & (ser.index < min(ce, se))]  # captured window only
        if ref is None or len(ins) < 30:  # if: no reference or too few values
            continue  # skip
        out = []  # longest spans at the four exit thresholds
        for ex in (1.0, 2.0, 3.0, 5.0):  # for loop over exit thresholds in dB
            # spans with entry 5 dB, direction "decreases", exit as a share of the entry passed as an argument
            rr = disturbed_runs(ins, ref, 5.0, -1, exit_fraction=ex / 5.0)
            # the longest span in minutes (0.0 if none)
            out.append(max([(pd.Timestamp(b_) - pd.Timestamp(a_)).total_seconds() / 60.0
                            for a_, b_ in rr], default=0.0))  # generator clause: runs over the disturbed spans found with this exit threshold
        print(f"      {code:<8}" + "".join(f"{v:>11.2f}" for v in out))  # the four spans on one line

    # rows whose margin excursion exceeds twice the field's own entry threshold (10 dB for gain, 800 counts for
    # u-blox AGC, 40 for u-blox noise): one rule in each field's unit, where 10 was applied to all units before
    marg_lim = edf.field.map(lambda f_: 2 * DROP_THRESHOLD[f_])  # Series.map with a lambda: the limit per row
    # boolean masks: rows whose margin excursion is known and beyond the field's own limit
    marg_hit = edf[(edf.max_excursion_in_margin.notna())
                   & (edf.max_excursion_in_margin > marg_lim)]  # element-wise condition: and the margin excursion is beyond the field's own limit
    if not marg_hit.empty:  # if: any
        print("\n  EXCURSIONS FOUND IN THE CAPTURE MARGINS, which the timeline script and the")  # sub-heading
        print("  research plan (Research_Plan_Topu.pdf, section 2.3) treat as the only candidate clean data:")  # continued
        for _, r in marg_hit.iterrows():  # for loop over those rows
            # session, receiver, field, margin excursion and in-window excursion
            print(f"      {r.test_code:<8}{r.receiver:<15}{r.field:<14}"
                  f"margin excursion {r.max_excursion_in_margin:5.0f} "  # f-string continued: the excursion inside the margin
                  f"(in-window {r.max_excursion:5.0f})")  # f-string continued: the excursion inside the window, for comparison
        # records it; the joined session list is computed, the rest ("10 dB", "2.6.5 ... deeper") is typed
        flag("The capture margins are not uniformly quiet. In session "
             + ", ".join(sorted(marg_hit.test_code.unique()))  # string concatenation: the sessions whose margins hold such an excursion
             + " a front-end indicator departs from its quiet reference by more than twice its entry threshold "
               "INSIDE the margin, which the timeline script and the research plan (Research_Plan_Topu.pdf, section 2.3) treat as "
               "candidate clean data. The margin of 2.6.5 in particular contains a "
               "deeper excursion than its own scheduled window does, consistent with the "
               "motorcade passing after the scheduled end. Clean-class intervals must be "
               "verified against this measurement at labelling time rather than taken "
               "from the schedule.")

    # ---- the straddle test: a 50 W L1 jammer switching on inside a capture ----
    print("\n" + "-" * 78)  # rule
    print("  A FIFTH NEGATIVE TEST: the 2.6.5 capture straddles a 50 W switch-on")  # heading
    print("-" * 78)  # rule
    j131 = [p for p in plan if p["test_code"] == "1.3.1"]  # the 1.3.1 plan entry, as a list
    if j131:  # if: present (the straddle CSV is only written in this branch)
        e = j131[0]  # the entry
        js, je = pd.Timestamp(e["start_utc"]), pd.Timestamp(e["end_utc"])  # its window
        sr = sessions[sessions.test_code == "2.6.5"].iloc[0]  # the 2.6.5 session
        cs = pd.Timestamp(sr["capture_start_utc"], tz="UTC")  # capture start
        ce = pd.Timestamp(sr["capture_end_utc"], tz="UTC")  # capture end
        ss = pd.Timestamp(sr["scheduled_start_utc"], tz="UTC")  # scheduled start
        se5 = pd.Timestamp(sr["scheduled_end_utc"], tz="UTC")  # 2.6.5 scheduled end
        # the excursion is the detector's own whole-capture run on the X5 L1 gain that reaches past the
        # scheduled end, replacing the typed 12:50:00-12:52:30; the trailing smoother delays its edges
        r5 = [r_ for r_ in traces.get("2.6.5", {}).get("runs", []) if pd.Timestamp(r_[1]) > se5]
        if not r5:  # if: the detector found no such run
            die("no X5 L1 excursion reaching past the 2.6.5 scheduled end; the straddle test cannot be set up")  # stop
        exc_a, exc_b = pd.Timestamp(r5[0][0]), pd.Timestamp(r5[0][1])  # its start and end
        # the jammer's code, title (first 46 characters) and power
        print(f"  {e['test_code']}, {e['title'][:46]}, {e['power_w']:g} W at Test Area 1,")
        print(f"  transmits {js:%H:%M:%S}-{je:%H:%M:%S} UTC on L1, the band in question.")  # its window
        # how long the capture ran past the switch-on
        print(f"  The 2.6.5 capture runs to {ce:%H:%M:%S}, so it continues "
              f"{(ce - js).total_seconds()/60:.2f} min past that switch-on.")  # f-string continued: minutes the capture runs past the switch-on
        # table header
        print(f"\n      {'receiver':<15}{'field':<15}{'quiet':>8}{'during excursion':>18}"
              f"{'after it, jammer on':>21}{'departure':>11}")  # f-string continued: header cells for the level after the excursion and the departure from quiet
        rows = []  # straddle rows
        # for loop over five (receiver, band, field) combinations written out as a list of tuples
        for rx, band, fld in [("Septentrio_01", "GPSL1/E1", "agc_gain_db"),
                              ("Septentrio_02", "GPSL1/E1", "agc_gain_db"),  # list element: the mosaic-X5 L1 gain
                              ("Septentrio_03", "GPSL1/E1", "agc_gain_db"),  # list element: the mosaic-H L1 gain
                              ("u-blox_01", "L1", "agc_cnt"),  # list element: the u-blox 01 L1 gain count
                              ("u-blox_01", "L1", "noise_per_ms")]:  # list element: the u-blox 01 L1 noise level
            # this combination in 2.6.5
            sub = long[(long.test_code == "2.6.5") & (long.receiver == rx)
                       & (long.band == band) & (long.field == fld)]  # element-wise condition: and this band and field
            if sub.empty:  # if: none
                continue  # skip
            g = sub.set_index("timestamp_utc")["value"].sort_index()  # time-indexed series
            quiet = g[(g.index >= cs) & (g.index < ss)]  # the pre-test margin
            during = g[(g.index >= exc_a) & (g.index <= exc_b)]  # the excursion
            after = g[(g.index > exc_b) & (g.index <= ce)]  # after it, jammer still on
            if len(quiet) < 20 or len(after) < 20:  # if: too few
                continue  # skip
            q, a_ = float(np.nanmedian(quiet)), float(np.nanmedian(after))  # the two medians (NaN-safe)
            # the row: quiet, during, after and the departure after-minus-quiet
            print(f"      {rx:<15}{fld:<15}{q:>8.0f}{np.nanmedian(during):>18.0f}"
                  f"{a_:>21.0f}{a_ - q:>+11.0f}")  # f-string continued: the level after the excursion with the jammer on, and its departure from quiet
            rows.append({"receiver": rx, "field": fld, "quiet_median": round(q, 2),  # appends the row
                         "median_during_excursion": round(float(np.nanmedian(during)), 2),  # dict entry: median during the excursion
                         "median_after_excursion_jammer_on": round(a_, 2),  # dict entry: median after the excursion, with the jammer still on
                         "departure_from_quiet": round(a_ - q, 2),  # dict entry: that median minus the quiet median
                         "minutes_jammer_on_after_excursion":  # dict entry: how long the jammer stayed on after the excursion; the value is on the next line
                             round((ce - exc_b).total_seconds() / 60, 2)})  # dict value: minutes from the excursion's end to the capture's end
        # writes the straddle table
        pd.DataFrame(rows).to_csv(PROCESSED / "ground_truth" / "straddle_test_1_3_1.csv",
                                  index=False, lineterminator="\n")  # keyword arguments of to_csv(): no index column, and Unix line ends
        # minutes of capture after the excursion; the rest is fixed text
        print(f"\n  For the final {(ce - exc_b).total_seconds()/60:.2f} minutes of the capture a "
              f"50 W sweep jammer was")
        # the distance from the 2.6.5 parked position (measured)
        print(f"  radiating on L1 from {site.loc['2.6.5', 'dist_to_sender_km']:.2f} km away and every receiver sat "
              "at its quiet value.")
        # the excursion's detected start and end, so the reader can see it is no longer typed
        print(f"  The excursion is the detector's run {exc_a:%H:%M:%S}-{exc_b:%H:%M:%S} UTC (trailing 15 s smoother).")
        print("  This is a direct on-air test rather than a statistical contrast, and it is")  # continued
        print("  the most economical of the five negative results.")  # continued
        print("\n  WHAT THE EXCURSION PROBABLY WAS, stated as a candidate and not as fact.")  # continued
        # durations, times and candidates computed from the plan and the sessions
        nxt_ = sessions[pd.to_datetime(sessions.scheduled_start_utc) > pd.Timestamp(sr["scheduled_end_utc"])]  # later
        nxt_ = nxt_.iloc[0] if len(nxt_) else None  # the next session, if any
        # 1.1.12 and 1.1.22 (the latter only under the +2 h reading) are candidates too
        tw_ = [p for p in plan if p["test_code"] in ("1.1.12", "1.1.22")]  # their plan entries
        # 1.3.1's duration from its plan window
        print(f"  It cannot be 1.3.1: that transmission ran for {(je - js).total_seconds() / 60:.0f} minutes without a "
              "break,")  # computed duration
        print("  and the gain recovered fully while it was still on air. The candidates are our")  # continued
        print(f"  own motorcade (catalogue: '{sr['test_name']}'), arriving as the")  # the catalogue's own words
        # the excursion's duration and the next session's window, from the detector and the sessions table
        print(f"  excursion begins and lasting {(exc_b - exc_a).total_seconds() / 60:.1f} min; repositioning before "
              f"{nxt_['test_code'] if nxt_ is not None else 'the next test'}, whose window opens at "  # f-string continued: the next test, named when there is one
              f"{pd.Timestamp(nxt_['scheduled_start_utc']):%H:%M};" if nxt_ is not None else "")  # f-string continued: when that test's window opens, printed only when there is a next test
        for p_ in tw_:  # one line per Test Area 2 candidate
            # one Test Area 2 candidate: area, power from the plan, window
            print(f"  and {p_['test_code']} (Test Area {p_['test_area']}, {p_['power_w'] if p_['power_w'] is not None else 'unknown'} W), "
                  f"scheduled {pd.Timestamp(p_['start_utc']):%H:%M}-{pd.Timestamp(p_['end_utc']):%H:%M} UTC"  # f-string continued: the candidate's scheduled window in UTC
                  # f-string continued: a remark for the one candidate that overlaps only under a ruled-out reading
                  f"{' (overlaps only under the +2 h reading, which section [13](d) rules out)' if p_['test_code'] == '1.1.22' else ''}.")
        print("  The evidence cannot separate them; the odometer and inertial records would settle")  # continued
        print("  the motorcade's part.")  # continued
        # records the fifth negative test (typed text)
        # every time and duration computed; the excursion's cause stated as candidates
        note(f"A fifth negative test is available and is more direct than the statistical ones. Test "
             f"{e['test_code']}, {e['power_w']:g} W from Test Area 1, switched on at {js:%H:%M:%S} UTC inside the "  # f-string continued: the test code, its power and when it switched on
             f"2.6.5 capture, which continued for {(ce - js).total_seconds() / 60:.1f} minutes afterwards. Apart "  # f-string continued: how many minutes the capture continued after the switch-on
             f"from a {(exc_b - exc_a).total_seconds() / 60:.1f}-minute excursion whose candidates are our own "  # f-string continued: the length of the one excursion and its candidate causes
             "motorcade and the Test Area 2 jammer 1.1.12 (0.1 W, catalogue), every receiver's gain sat at its quiet value "
             "for the remainder of the capture while that jammer was on air.")

    # ---- quantify the margin contamination rather than warning about it ----
    print("\n" + "-" * 78)  # rule
    print("  HOW MUCH CANDIDATE CLEAN DATA SURVIVES")  # heading
    print("-" * 78)  # rule
    # the unexplained 15:45 step in 2.6.12, quantified per receiver
    # and field; the pre-test margin before it is flagged as possibly contaminated, as a precaution
    rem = margin_remnant(long, sessions, pyr)  # one row per receiver, band and field
    rem.to_csv(PROCESSED / "ground_truth" / "margin_remnant_1_8_2.csv", index=False, lineterminator="\n")  # written (new)
    sep_ = rem[(rem.field == "agc_gain_db") & (rem.step.abs() >= 1) & (rem.step_time_utc != "")]  # Septentrio steps
    # the flag ends at the median step time of those series (a computed time, not typed): midnight of the day plus the
    # median of the step times as durations, floored to the second
    t_flag = ((pd.Timestamp(pyr["entry"]["end_utc"]).normalize() + pd.to_timedelta(sep_.step_time_utc).median())
              .floor("s") if len(sep_) else None)  # method chain: floors the median step time to the whole second, or nothing without steps
    s12_ = sessions[sessions.test_code == "2.6.12"].iloc[0]  # the 2.6.12 session
    cs12 = pd.Timestamp(s12_["capture_start_utc"], tz="UTC")  # its capture start
    mtot = mbad = 0.0  # chained assignment: total margin minutes and contaminated minutes
    mrows = []  # per-session rows
    for _, sr in sessions.iterrows():  # for loop over sessions
        cs = pd.Timestamp(sr["capture_start_utc"], tz="UTC")  # capture start
        ce = pd.Timestamp(sr["capture_end_utc"], tz="UTC")  # capture end
        ss = pd.Timestamp(sr["scheduled_start_utc"], tz="UTC")  # scheduled start
        se = pd.Timestamp(sr["scheduled_end_utc"], tz="UTC")  # scheduled end
        # the X5 L1 gain in this session
        sub = long[(long.test_code == sr["test_code"]) & (long.receiver == "Septentrio_02")
                   & (long.band == "GPSL1/E1") & (long.field == "agc_gain_db")]  # element-wise condition: only the L1 front-end gain rows
        if sub.empty:  # if: none
            continue  # skip
        g = sub.set_index("timestamp_utc")["value"].sort_index()  # time-indexed series
        # the typed 36.0 dB fallback is gone; a missing X5 L1 reference stops the run instead of being replaced by a constant
        if ("Septentrio_02", "GPSL1/E1", "agc_gain_db") not in refs:  # if: the quiet reference of the X5 L1 gain was not computed
            die("margin_contamination: no quiet reference for the mosaic-X5 GPSL1/E1 gain; refusing to fall back to a typed value")  # stops with status 1
        ref = refs[("Septentrio_02", "GPSL1/E1", "agc_gain_db")]  # the reference
        # The pre-test and post-test margins are measured SEPARATELY. Concatenating
        # them and differencing the result inserts one spurious interval spanning
        # the whole scheduled window at the join, which the clip then charges to
        # the margin as up to a minute of time that was never in it.
        # nested function (defined inside the loop, so it sees this session's ref): (total, contaminated) minutes
        def _span(x: pd.Series) -> tuple[float, float]:
            if len(x) < 2:  # if: fewer than two samples, no interval
                return 0.0, 0.0  # nothing
            # gaps between consecutive timestamps: diff, as milliseconds, as float, then / 1000 = seconds
            st_ = np.diff(x.index.to_numpy()).astype("timedelta64[ms]").astype(float) / 1000.0
            st_ = np.clip(np.append(st_, st_[-1]), 0, 60)  # repeats the last gap for the last sample; caps at 60 s
            # total minutes, and minutes where the sample lies more than 3 dB below the reference
            return st_.sum() / 60.0, (st_ * (x.to_numpy() < ref - 3.0)).sum() / 60.0
        pre_t, pre_b = _span(g[(g.index >= cs) & (g.index < ss)])  # pre-test margin
        post_t, post_b = _span(g[(g.index > se) & (g.index <= ce)])  # post-test margin
        tot_m, bad_m = pre_t + post_t, pre_b + post_b  # both margins together
        if tot_m <= 0:  # if: no margin
            # a row of zeros
            mrows.append({"test_code": sr["test_code"], "margin_min": 0.0,
                          "contaminated_min": 0.0, "surviving_min": 0.0})  # dict entry: no contaminated and no surviving minutes
            continue  # next session
        mtot += tot_m  # adds to the total
        mbad += bad_m  # adds to the contaminated total
        mrows.append({"test_code": sr["test_code"], "margin_min": round(tot_m, 2),  # appends the row
                      "contaminated_min": round(bad_m, 2),  # dict entry: margin minutes more than 3 dB below the quiet reference
                      "surviving_min": round(tot_m - bad_m, 2)})  # dict entry: margin minutes that survive as candidate clean data
    mdf = pd.DataFrame(mrows)  # the margin table
    # minutes of 2.6.12's pre-test margin before the step, flagged possibly contaminated; 0 elsewhere
    mdf["possibly_contaminated_margin_min"] = np.where(mdf.test_code == "2.6.12",
                                                     # argument of where(): the flagged minutes for 2.6.12, zero when no step was found
                                                     round((t_flag - cs12).total_seconds() / 60, 2) if t_flag is not None
                                                     else 0.0, 0.0)  # np.where: value per row by condition
    # np.where: the flagged interval in words for 2.6.12, empty elsewhere
    mdf["note"] = np.where(mdf.test_code == "2.6.12", f"possibly contaminated {cs12:%H:%M:%S}-"
                           f"{t_flag:%H:%M:%S}" if t_flag is not None else "", "")  # the flag in words
    mdf.to_csv(PROCESSED / "ground_truth" / "margin_contamination.csv", index=False, lineterminator="\n")  # written
    # the same total from the timeline script's schedule arithmetic, for comparison
    sched_tot = float(sessions.minutes_before_scheduled_start.sum()
                      + sessions.minutes_after_scheduled_end.sum())  # arithmetic continued: plus the minutes after each scheduled end
    print(f"      {'test':<8}{'margin (min)':>14}{'>3 dB below quiet':>20}{'surviving':>12}")  # header
    for _, r in mdf.iterrows():  # for loop over sessions
        # the row
        print(f"      {r.test_code:<8}{r.margin_min:>14.2f}{r.contaminated_min:>20.2f}"
              f"{r.surviving_min:>12.2f}")  # f-string continued: the surviving minutes
    print(f"      {'TOTAL':<8}{mtot:>14.2f}{mbad:>20.2f}{mtot - mbad:>12.2f}")  # totals
    print(f"\n  Two independent routes to the same total: {sched_tot:.2f} min from the")  # comparison
    print(f"  schedule arithmetic of the timeline script, {mtot:.2f} min from the receiver's own")  # continued
    print(f"  epoch timestamps, agreeing to {abs(mtot - sched_tot)*60:.0f} seconds.")  # difference in seconds
    # contaminated minutes and share; "all of it in 2.6.5 and 2.6.10" is typed
    if t_flag is not None:  # if: the possibly contaminated 2.6.12 margin step was found
        fl_ = float(mdf.possibly_contaminated_margin_min.sum())  # its minutes
        # the flagged interval and the surviving total with and without it
        print(f"  the possibly contaminated 2.6.12 margin: 2.6.12 {cs12:%H:%M:%S}-{t_flag:%H:%M:%S} flagged possibly contaminated ({fl_:.2f} min, the "
              # f-string continued: the reason for the flag, and the surviving total with and without it
              f"unexplained step, the possibly contaminated 2.6.12 margin); surviving {mtot - mbad:.2f} min with it, {mtot - mbad - fl_:.2f} min without it.")
    # contaminated minutes and their share; the session names that follow are fixed text
    bad_s = ", ".join(mdf[mdf.contaminated_min > 0].test_code)  # the sessions holding them, computed
    # the contaminated minutes and their share
    print(f"  Contaminated: {mbad:.2f} min, which is {100*mbad/mtot:.1f} per cent, all of it in")
    print(f"  {bad_s}. Surviving candidate clean data: {mtot - mbad:.2f} minutes.")  # continued
    # records it, every number and session name computed
    note(f"The candidate clean margin totals {mtot:.2f} minutes measured from the "
         f"receiver's own timestamps, against {sched_tot:.2f} minutes from the schedule "  # f-string continued: the total from the schedule, the second independent route
         f"arithmetic of the timeline script, two independent routes agreeing to within {abs(mtot - sched_tot) * 60:.0f} "  # f-string continued: how many seconds the two routes differ
         f"seconds. Of that, {mbad:.2f} minutes lies more than 3 dB below the quiet "  # f-string continued: minutes more than 3 dB below the quiet reference
         f"reference, {100*mbad/mtot:.1f} per cent of the total, confined to {bad_s}. "  # f-string continued: their share of the total, and the sessions they fall in
         f"{mtot - mbad:.2f} minutes survives as candidate clean data.")  # f-string continued: minutes that survive as candidate clean data

    print("\n  The mosaic-G5 is excluded above. For the record, its excursion depths are")  # fixed text
    print("  still meaningful even though its spans are not:")  # continued
    for _, sr in sessions.iterrows():  # for loop over sessions
        # the G5 L1 gain in this session
        sub = long[(long.test_code == sr["test_code"])
                   & (long.receiver == "Septentrio_01")  # element-wise condition: the mosaic-G5 receiver only
                   & (long.band == "GPSL1/E1") & (long.field == "agc_gain_db")]  # element-wise condition: and its L1 front-end gain
        if sub.empty:  # if: none
            continue  # skip
        ref = refs.get(("Septentrio_01", "GPSL1/E1", "agc_gain_db"))  # the G5's own reference
        ss = pd.Timestamp(sr["scheduled_start_utc"], tz="UTC")  # scheduled start
        se = pd.Timestamp(sr["scheduled_end_utc"], tz="UTC")  # scheduled end
        ins = sub.set_index("timestamp_utc")["value"].sort_index()  # time-indexed series
        ins = ins[(ins.index >= ss) & (ins.index < se)]  # inside the scheduled window
        if ref is None or ins.empty:  # if: nothing to report
            continue  # skip
        print(f"      {sr['test_code']:<8}excursion {ref - np.nanmin(ins):5.0f} dB")  # reference minus minimum

    # ---- our own jamming ------------------------------------------------------
    print("\n" + "=" * 78)  # rule
    print("[11] OUR OWN JAMMING - a separate finding, not part of the verdict")  # heading
    print("=" * 78)  # rule
    print("  Parsed from the catalogue, not assumed:")  # sub-heading
    print(f'      "{own["group_sentence_initial"]}"')  # the source sentence, quoted (single-quoted f-string)
    print(f'      "{own["group_sentence_stop"]}"')  # the second sentence
    print(f'      "{own["group_sentence_continuous"]}"')  # the third sentence
    # the parsed structure
    print(f"\n      initial jamming: first {own['initial_min']} min on "
          f"{', '.join(own['initial_bands'])}")  # f-string continued: the bands jammed in the first minutes
    print(f"      continuous jamming: {', '.join(own['continuous_bands'])} throughout")  # continuous bands
    ours_exempt = sorted(c for c in own["exempt"] if c in set(sessions.test_code))  # exempt tests we recorded
    # the exempt codes, or "none"
    print(f"      tests exempting themselves from initial jamming: "
          f"{', '.join(ours_exempt) if ours_exempt else 'none'}")  # f-string continued: the exempt test codes, or none
    for c in ours_exempt:  # for loop over them
        print(f'          {c}: "{own["exempt"][c][:90]}"')  # the exempting sentence, first 90 characters
    orows = []  # own-jamming rows
    for _, sr in sessions.iterrows():  # for loop over sessions
        code = sr["test_code"]  # test code
        ss = pd.Timestamp(sr["scheduled_start_utc"], tz="UTC")  # scheduled start
        se = pd.Timestamp(sr["scheduled_end_utc"], tz="UTC")  # scheduled end
        cs = pd.Timestamp(sr["capture_start_utc"], tz="UTC")  # capture start
        init_end = ss + pd.Timedelta(minutes=own["initial_min"])  # end of the first five minutes
        exempt = code in own["exempt"]  # boolean: this test exempts itself
        # per receiver, band and field of this session (the expression continues on the next line)
        for (rx, band, fld), g in long[(long.test_code == code)
                                       ].groupby(["receiver", "band", "field"]):  # method chain: groups this session's rows by receiver, band and field
            fam = "Septentrio" if rx.startswith("Sept") else "u-blox"  # family
            d = DIRECTION.get((fam, fld), 0)  # direction
            if d == 0 or fam != "Septentrio":  # if: no direction, or not a Septentrio receiver
                continue  # skip
            # dict comprehension (front-end name -> catalogue band) built afresh per group, then .get(band)
            nm = {FRONTEND[fe]: b for b, fe in OWN_BAND_TO_FRONTEND.items()
                  if fe in FRONTEND}.get(band)  # generator clause: keeps known front-end codes; the band is then looked up in the result
            if nm is None:  # if: not one of the named bands (Aux1 bands and others drop out)
                continue  # skip
            # chained conditional expression: the band's class under the catalogue text
            cls = ("initial+continuous" if nm in own["initial_bands"] else
                   "continuous" if nm in own["continuous_bands"] else "other")  # conditional expression: continuous jamming only, or a band the catalogue does not name
            base = g.loc[(g.timestamp_utc >= cs) & (g.timestamp_utc < ss), "value"].to_numpy()  # pre-test margin
            early = g.loc[(g.timestamp_utc >= ss) & (g.timestamp_utc < init_end), "value"].to_numpy()  # first 5 min
            late = g.loc[(g.timestamp_utc >= init_end) & (g.timestamp_utc < se), "value"].to_numpy()  # the rest
            if len(early) < 30 or len(late) < 30:  # if: too few in either part
                continue  # skip
            orows.append({  # appends the row
                "test_code": code, "receiver": rx, "band": band, "plan_band": nm,  # dict entry: session, receiver, front-end band and the catalogue's band name
                "field": fld, "band_class": cls, "exempt_from_initial": exempt,  # dict entry: field, the band's class, and whether the test exempts itself from initial jamming
                "n_baseline": len(base), "n_first5": len(early), "n_rest": len(late),  # dict entry: readings in the margin, in the first five minutes, and in the rest of the window
                # conditional expression: the margin median, or None when there is no margin
                "median_baseline": round(float(np.median(base)), 3) if len(base) else None,
                "median_first5": round(float(np.median(early)), 3),  # dict entry: median in the first five minutes
                "median_rest": round(float(np.median(late)), 3),  # dict entry: median in the rest of the window
                "first5_minus_rest": round(float(np.median(early) - np.median(late)), 3),  # negative = lower early
                # rest minus margin, only with at least 20 margin values
                "rest_minus_baseline": (round(float(np.median(late) - np.median(base)), 3)
                                        if len(base) >= 20 else None)})  # conditional expression: only with at least 20 margin readings
    odf = pd.DataFrame(orows)  # the own-jamming table
    odf.to_csv(PROCESSED / "ground_truth" / "own_jamming_assessment.csv", index=False, lineterminator="\n")  # written
    if not odf.empty:  # if: rows exist
        m = odf[odf.receiver == "Septentrio_02"]  # the mosaic-X5 rows
        m = m[m.field == "agc_gain_db"]  # its gain only
        print("\n  Septentrio mosaic-X5, AGC gain in dB. 'first5' is the first five")  # explanation
        print("  minutes of the scheduled window, 'rest' the remainder, 'base' the")  # continued
        print("  pre-test margin.")  # continued
        # table header
        print(f"\n      {'test':<7} {'band':<12} {'plan':<5} {'class':<19} {'exempt':<7} "
              f"{'first5':>9} {'rest':>9} {'f5-rest':>9} {'rest-base':>10}")  # f-string continued: header cells for the medians and the two differences
        for _, r in m.iterrows():  # for loop over the X5 rows
            # conditional expression: the signed rest-minus-base, or a dash when missing
            rb = f"{r.rest_minus_baseline:+10.2f}" if pd.notna(r.rest_minus_baseline) else f"{'-':>10}"
            # the printed row
            print(f"      {r.test_code:<7} {r.band:<12} {r.plan_band:<5} {r.band_class:<19} "
                  f"{str(r.exempt_from_initial):<7} {r.median_first5:>9.2f} "  # f-string continued: the exempt flag and the median in the first five minutes
                  f"{r.median_rest:>9.2f} {r.first5_minus_rest:>+9.2f} {rb}")  # f-string continued: the median of the rest and the two differences
        print("\n  THE SECOND PREDICTION FIXED IN ADVANCE: L1, G1 and B1 should be jammed")  # the prediction, fixed text
        print("  for the first five minutes of every test that does not exempt itself,")  # continued
        print("  and not afterwards. A test that exempts itself should show no such step.")  # continued
        ini = m[m.band_class == "initial+continuous"]  # rows on L1, G1, B1
        # table header
        print(f"\n      {'test':<7} {'exempt':<7} {'median first5 minus rest on L1, G1, B1':<42} "
              f"{'prediction'}")  # f-string continued: the header cell for the prediction
        res_ = []  # (test, exempt, step seen, matches) per testable session, for the computed summary
        for code, g in ini.groupby("test_code", sort=False):  # per test, in table order
            step = float(g.first5_minus_rest.median())  # the median over the three bands
            ex = bool(g.exempt_from_initial.iloc[0])  # exempt or not
            # A drop during the first five minutes shows as first5 BELOW rest.
            observed_step = step < -3.0  # boolean: a drop of more than 3 dB
            ok = (observed_step != ex)  # matches if a step appears exactly when the test is NOT exempt
            # the row; the label says "no step seen" whenever it does not match (also for an unexpected step)
            print(f"      {code:<7} {str(ex):<7} {step:>+10.2f} dB"
                  f"{'   (a step is present)' if observed_step else '   (no step)':<30} "  # f-string continued: whether a step is present
                  f"{'matches' if ok else 'no step seen'}")  # f-string continued: whether that agrees with the catalogue's prediction
            res_.append((code, ex, observed_step, ok))  # kept for the summary
        # the summary computed from the rows above and from each test's catalogue name
        mode_ = {c_: ("stationary" if "tationary" in str(t_) else "drive-past")  # dict comprehension: test -> mode
                 for c_, t_ in zip(sessions.test_code, sessions.test_name)}  # generator clause: runs over the test codes and catalogue names of the sessions
        no_ = [c_ for c_, _, _, ok_ in res_ if not ok_]  # tests that do not match the prediction
        st_ = [c_ for c_, _, seen_, _ in res_ if seen_]  # tests that show the step
        # how many testable sessions match, and which do not with their catalogue mode
        print(f"\n  {sum(r_[3] for r_ in res_)} of the {len(res_)} testable sessions match. The {len(no_)} that do "
              f"not are {', '.join(no_)} ({', '.join(sorted({mode_[c_] for c_ in no_}))}),")  # f-string continued: the sessions that do not match, and their spoofer modes
        print("  so at the scheduled start the transmitter was still far away. The step shows in")  # continued
        # the sessions that show the step, with their catalogue mode
        print(f"  {', '.join(st_)} ({', '.join(sorted({mode_[c_] for c_ in st_}))}). The catalogue describes what "
              "was transmitted; the receiver records")  # computed lists
        print("  what arrived, and the two differ by range. This is the same finding as")  # continued
        print("  the effective attack interval in section [10] and it is the central")  # continued
        print("  result of this section, arrived at from a second direction.")  # continued
        # records it, every number and list computed
        f12_ = ini[ini.test_code.isin(st_)].first5_minus_rest.abs()  # the step's depth per band where it shows
        # list comprehension: our sessions that exempt themselves, parsed from the catalogue text
        ex_ = sorted(c_ for c_ in own["exempt"] if c_ in set(sessions.test_code))
        # records the finding with every list and range computed
        note(f"The catalogue's initial-jamming structure is visible only in {', '.join(st_)}. "
             f"{', '.join(no_)} ({', '.join(sorted({mode_[c_] for c_ in no_}))}) show no initial-jamming step, which "  # f-string continued: the sessions without a step, with their spoofer modes
             "is consistent with the transmitter being out of effective range at the scheduled start rather than "
             "with the catalogue being wrong. The catalogue describes what was transmitted; the receiver records "
             f"what arrived. In {', '.join(st_)} the AGC on L1, G1 and B1 is depressed by {f12_.min():.0f} to "  # f-string continued: the sessions with a step, and the smallest depression in dB
             f"{f12_.max():.0f} dB during the first five minutes of the scheduled window relative to the remainder, "  # f-string continued: the largest depression in dB
             f"and the {len(ex_)} tests that exempt themselves in the catalogue text ({', '.join(ex_)}) "  # f-string continued: the tests that exempt themselves from the initial jamming
             "show no such step where they can be tested.")

        # continuously jammed bands; the field filter repeats the one already applied to m
        cont = m[(m.band_class == "continuous") & (m.field == "agc_gain_db")]
        n_cont_dist = int((cont.rest_minus_baseline < -3).sum()) if not cont.empty else 0  # depressed > 3 dB
        # sub-heading naming the bands
        print(f"\n  Bands the catalogue says our own motorcade jammed continuously "
              f"({', '.join(own['continuous_bands'])}):")  # f-string continued: the continuously jammed bands, from the parsed catalogue
        # the count
        print(f"      {n_cont_dist} of {len(cont)} show the AGC depressed by more than "
              "3 dB against the pre-test margin.")
        print("\n  CONSEQUENCE FOR THE FEATURE SET. Any feature computed on a band our")  # fixed text
        print("  own motorcade jammed throughout measures jamming, not spoofing, and the")  # continued
        print("  opening minutes of the sessions that did not exempt themselves are a")  # continued
        print("  different class of event from the remainder of those sessions.")  # continued
        # records the consequence (typed text)
        # bands and the exempt count read from the parsed catalogue
        note(f"The catalogue states that our own motorcade jammed {', '.join(own['continuous_bands'])} "
             "continuously throughout every test in group 2.6, and additionally jammed "
             f"{', '.join(own['initial_bands'])} for the first five minutes of every test; of our sessions "  # f-string continued: the bands jammed in the first five minutes, and the exempt count
             f"{len(ex_)} ({', '.join(ex_)}) exempt themselves. Features computed on the continuously jammed "  # f-string continued: the exempt sessions by name
             "bands would measure our own jamming rather than spoofing.")

    # ---- analyses -----------------------------------------------------------------------------------------
    print("\n" + "=" * 78)  # rule
    print("[13] ANALYSES: frozen values, clean margins, site confound, schedule shift, 1590 MHz, 2.6.12")  # heading
    print("=" * 78)  # rule

    # (a) 30g: frozen or held receiver gain values, mechanism first
    frz = frozen_check(sept, ubx)  # one row per receiver, band and field
    frz.to_csv(PROCESSED / "ground_truth" / "receiver_frozen_check.csv", index=False, lineterminator="\n")  # written (new)
    print(f"  (a) Frozen-value check, {len(frz)} receiver-band series:")  # sub-heading
    # the range of the share of readings inside unchanged runs of 30 s or more
    print(f"      readings inside unchanged runs of 30 s or more: {frz.share_in_runs_30s.min():.0%} to "
          f"{frz.share_in_runs_30s.max():.0%} of each series")  # f-string continued: the highest share, as a percentage of each series
    # what the receiver reported during those runs, against elsewhere
    print(f"      in those runs, no tracked satellite in at most {frz.share_runs_no_satellites.max():.1%} of readings "
          # f-string continued: the same share elsewhere, and the share with the front end unlocked
          f"(elsewhere at most {frz.share_elsewhere_no_satellites.max():.1%}); front end unlocked in at most "
          f"{frz.share_runs_pll_unlocked.max(skipna=True):.1%}")  # f-string continued: the highest share of unlocked front-end readings
    ub = frz[frz.field == "agc_cnt"]  # the u-blox row(s)
    # quantisation: distinct values and the u-blox step
    print(f"      distinct values per series {int(frz.n_distinct.min())} to {int(frz.n_distinct.max())}; Septentrio "
          f"gain moves in whole dB, u-blox AGC count in steps of {ub.step.median():.0f}")  # f-string continued: the step sizes of the Septentrio gain and the u-blox count
    print(f"      series with a held-register signature: {int(frz.held_register_signature.sum())}")  # the count
    # the conclusion, from the computed counts: nothing removed either way
    print("      No reading is removed. Long runs coincide with normal tracking; they are a quantised gain holding")
    # continuation of the conclusion: the floors, read from the frozen-check table
    print("      steady, and gains resting on their floor (Septentrio "
          f"{sorted(set(frz[frz.field == 'agc_gain_db'].floor))} dB, u-blox {sorted(set(ub.floor))}) under our jamming.")  # f-string continued: the floors of the Septentrio gain and the u-blox count

    # (b) clean margin minutes per indicator and combined, and the u-blox 01 noise after the 2.6.10 scheduled end
    # per session, margin part and indicator; 2.6.12's pre-test margin cut at the possibly contaminated 2.6.12 margin step and the first part flagged
    mcl = margin_clean(long, sessions, split=("2.6.12", t_flag) if t_flag is not None else None)
    # the single-indicator figure of section [10] is kept as one row of the same table
    mcl = pd.concat([mcl, pd.DataFrame([{
        "test_code": "all", "margin": "all", "indicator": "Septentrio_02 GPSL1/E1 agc_gain_db (section [10] method)",  # dict entry: names the single-indicator method as one row of the same table
        # dict entries: section [10]'s totals (35.71 / 1.59 / 34.13 in the run before)
        "margin_min": round(mtot, 3), "covered_min": round(mtot, 3), "contaminated_min": round(mbad, 3),
        "surviving_min": round(mtot - mbad, 3), "possibly_contaminated_margin": False}])], ignore_index=True)  # dict entry: surviving minutes, and this row carries no possibly contaminated flag
    mcl.to_csv(PROCESSED / "ground_truth" / "margin_clean_minutes.csv", index=False, lineterminator="\n")  # written (new)
    # total minutes per indicator over all sessions (the 'all' row excluded)
    tot = mcl[mcl.test_code != "all"].groupby("indicator")[["margin_min", "contaminated_min", "surviving_min"]].sum()
    print("\n  (b) Clean margin minutes per indicator, all sessions (method of the check's R07):")  # sub-heading
    for ind_, r in tot.iterrows():  # for loop over indicators
        print(f"      {ind_:<58} {r.margin_min:6.2f} {r.contaminated_min:6.2f} {r.surviving_min:6.2f}")  # the row
    # the section [10] single-indicator row printed alongside, for comparison
    print(f"      {'section [10] single indicator (mosaic-X5 L1 gain)':<58} {mtot:6.2f} {mbad:6.2f} {mtot - mbad:6.2f}")
    # per-session totals of the combined indicator
    allc = mcl[mcl.indicator.str.startswith("ALL")].groupby("test_code")[["margin_min", "surviving_min"]].sum()
    al_ = mcl[mcl.indicator.str.startswith("ALL")]  # the combined rows
    # combined clean minutes with and without the flagged part of 2.6.12's pre-test margin
    print(f"      combined, all sessions: {al_.surviving_min.sum():.2f} of {al_.margin_min.sum():.2f} min with "
          # f-string continued: the surviving and margin minutes without the flagged part
          f"the possibly contaminated 2.6.12 margin-flagged part, {al_[~al_.possibly_contaminated_margin].surviving_min.sum():.2f} of "
          f"{al_[~al_.possibly_contaminated_margin].margin_min.sum():.2f} without it")  # f-string continued: the margin total without the flagged part
    # one line: surviving combined minutes of each session (generator joined with ', ')
    print("      combined, per session: " + ", ".join(f"{c_} {r.surviving_min:.2f} of {r.margin_min:.2f}"
                                                   for c_, r in allc.iterrows()))  # per-session survivors
    # the u-blox noise in the 2.6.10 post-test margin, in 30 s medians, to see when it falls back
    s10 = sessions[sessions.test_code == "2.6.10"].iloc[0]  # the 2.6.10 session
    se10 = pd.Timestamp(s10["scheduled_end_utc"], tz="UTC")  # its scheduled end
    # boolean masks: u-blox_01 L1 noise in 2.6.10 from its scheduled end on, indexed by time
    nz10 = long[(long.test_code == "2.6.10") & (long.receiver == "u-blox_01") & (long.band == "L1")
                & (long.field == "noise_per_ms") & (long.timestamp_utc >= se10)].set_index("timestamp_utc").value  # element-wise condition: the noise field from the scheduled end on, indexed by time
    b30 = nz10.resample("30s").median()  # 30 s medians from the scheduled end
    # one line of 30 s medians (when the noise falls back after the scheduled end)
    print(f"      u-blox_01 L1 noise after the 2.6.10 scheduled end ({se10:%H:%M:%S}), 30 s medians: "
          + ", ".join(f"{t_:%H:%M:%S} {v_:.0f}" for t_, v_ in b30.items()))  # the medians in time order

    # (c) site confound: where the 50 W and baseline seconds were recorded, and the comparison within captures
    long["site"] = site_of(long, site, dist)  # each row's recording site, "moving" for 2.6.5's moving part
    grp = long[long.exposure_group.isin([G50, GBL])]  # the two compared groups only
    spl = (grp.groupby(["receiver", "exposure_group", "site"]).timestamp_utc.nunique()  # distinct seconds
           .unstack("site", fill_value=0).reset_index())  # one column per site
    spl.to_csv(PROCESSED / "ground_truth" / "receiver_group_site_split.csv", index=False, lineterminator="\n")  # written (new)
    # sub-heading of the site split
    print("\n  (c) Where the compared seconds were recorded (distinct time stamps per receiver, group and site):")
    for _, r in spl.iterrows():  # for loop over rows
        print(f"      {r.receiver:<14} {r.exposure_group:<30} " + ", ".join(  # the row, one site at a time
            f"{c_}: {int(r[c_])}" for c_ in spl.columns if c_ not in ("receiver", "exposure_group")))  # generator clause: one count per site column, skipping the identity columns
    n_unk = long[long.exposure_group == GUNK].timestamp_utc.nunique()  # seconds excluded as unknown power
    n_ghi = long[long.exposure_group == GHI].timestamp_utc.nunique()  # seconds in the other-above-1-W group
    print(f"      excluded as unknown power only: {n_unk} distinct seconds; other above 1 W: {n_ghi}")  # the two counts
    # the comparison inside each capture, twice: on all its seconds, and outside our own scheduled windows and
    # 2.6.5's moving part, because inside a capture the 50 W time can coincide with our own jamming or driving
    own_win = np.zeros(len(long), bool)  # boolean array: inside one of our own scheduled windows
    for _, sr_ in sessions.iterrows():  # for loop over sessions
        own_win |= ((long.timestamp_utc >= pd.Timestamp(sr_["scheduled_start_utc"], tz="UTC"))  # "|=": accumulates
                & (long.timestamp_utc < pd.Timestamp(sr_["scheduled_end_utc"], tz="UTC"))).to_numpy()  # element-wise condition: and before the scheduled end
    keep = ~own_win & (long.site != "moving").to_numpy()  # outside our windows and not while the vehicle moved
    wc = pd.concat([within_capture(long).assign(restriction="all seconds of the capture"),  # .assign adds a column
                    # list element: the same comparison outside the on-site tests and the moving part of 2.6.5
                    within_capture(long[keep]).assign(restriction="outside our own windows and 2.6.5's moving part")],
                   ignore_index=True)  # both restrictions in one table
    wc["status"] = "descriptive, not a test"  # the label every row carries
    wc.to_csv(PROCESSED / "ground_truth" / "receiver_within_capture_comparison.csv", index=False, lineterminator="\n")  # written
    for res_, w_ in wc.groupby("restriction", sort=False):  # per restriction
        wbig = w_[w_.matches_prediction & (w_.cliffs_delta.abs() > 0.3)]  # strong predicted-direction contrasts
        # one line per restriction: contrasts, predicted-direction count, strong ones and their captures
        print(f"      within captures (descriptive, not a test), {res_}: {len(w_)} contrasts in "
              f"{', '.join(sorted(w_.test_code.unique()))}; "  # f-string continued: the sessions that contribute contrasts
              f"{int(w_.matches_prediction.sum())} in the predicted direction, {len(wbig)} beyond 0.30 "  # f-string continued: contrasts in the predicted direction and those beyond 0.30
              f"({', '.join(sorted(wbig.test_code.unique())) or 'none'})")  # f-string continued: the sessions with contrasts beyond 0.30, or none

    # (d) the schedule confirmation and its shift sensitivity: Test Area 1 and 2 windows shifted by +2 h and -2 h
    nzq = long[(long.receiver == "u-blox_01") & (long.band == "L1") & (long.field == "noise_per_ms")]
    mq = pd.concat([nzq[(nzq.timestamp_utc >= a_) & (nzq.timestamp_utc < b_)].value  # pooled margin values
                    for _, sr_ in sessions.iterrows() for a_, b_ in margin_parts(sr_) if b_ > a_])  # generator clause: every margin part of every session, skipping empty parts
    q_ = float(mq.quantile(0.75) - mq.quantile(0.25))  # pooled margin IQR of the u-blox noise
    refs[("u-blox_01", "L1", "noise_per_ms_iqr")] = q_ if q_ > 0 else 1.0  # kept with the references, 0 -> 1
    srows = []  # one row per shift and test
    for h in (0, 2, -2):  # for loop over the shifts in hours (0 is the schedule as adopted, for reference)
        dt = pd.Timedelta(hours=h)  # the shift
        # list comprehension: a copy of each plan entry, with Test Area 1 and 2 windows moved by the shift
        splan = [dict(p, start_utc=p["start_utc"] + dt, end_utc=p["end_utc"] + dt) if p["test_area"] in (1, 2)
                 else p for p in plan]  # conditional expression: plan entries of other test areas stay as they are
        for code in ("1.3.1", "1.1.29"):  # the straddle jammer and the handheld jammer
            e = next(p for p in splan if p["test_code"] == code)  # its shifted entry
            w0, w1 = pd.Timestamp(e["start_utc"]), pd.Timestamp(e["end_utc"])  # its shifted window
            r = on_air_check(w0, w1, long, sessions, refs)  # was it seen in a margin it overlapped?
            srows.append({"shift_h": h, "test": f"{code} on air during a capture margin",  # the row
                          "window_utc": f"{w0:%H:%M}-{w1:%H:%M}", **r})  # dict entry: the shifted window, and the overlap result's own columns
        # Test A on the shifted schedule - the seconds re-tagged, the identical group comparison run again
        tg_ = tag_times(pd.Series(long.timestamp_utc.unique()), splan, sessions, HIGH_POWER_W, FIFTY_W_MIN)
        # the seconds with the shifted groups: the old group column dropped, the new one merged on time
        ls_ = long.drop(columns=["exposure_group"]).merge(tg_[["timestamp_utc", "exposure_group"]], on="timestamp_utc")
        ca_ = group_comparison(ls_)  # Test A's table on the shifted groups
        nbig_a = int((ca_.matches_prediction & (ca_.cliffs_delta.abs() > 0.3)).sum()) if len(ca_) else 0  # strong
        srows.append({"shift_h": h, "test": "Test A group comparison", "window_utc": "all TA1/TA2 windows",  # the row
                      # conditional expression: untestable without contrasts; fails by the verdict's own rule
                      "outcome": "untestable" if ca_.empty else ("fails" if nbig_a >= max(3, 0.5 * len(ca_)) else "survives"),
                      "overlap": f"{len(ca_)} contrasts, {int(ca_.matches_prediction.sum()) if len(ca_) else 0} in the "  # dict entry: the number of contrasts and those in the predicted direction
                                 f"predicted direction, {nbig_a} beyond 0.30",  # f-string continued: the contrasts beyond 0.30
                      "overlap_min": None, "x5_l1_gain_minus_ref_db": None, "ublox_l1_noise_minus_ref": None})  # dict entry: the columns that do not apply to this test stay empty
        spyr = parse_pyramid(splan, flag)  # the pyramid on the shifted schedule
        pr = pyramid_contrasts(long, spyr)  # Test C on it
        nbig = sum(1 for x_ in pr if x_["matches_prediction"] and abs(x_["cliffs_delta"]) > 0.3)  # strong predicted
        posc = nbig >= max(2, 0.4 * max(len(pr), 1))  # Test C's positive rule, as in the verdict
        srows.append({"shift_h": h, "test": "1.8.2 pyramid (Test C)",  # the row
                      "window_utc": f"{pd.Timestamp(spyr['entry']['start_utc']):%H:%M}-"  # dict entry: the shifted window of the pyramid test, start first
                                    f"{pd.Timestamp(spyr['entry']['end_utc']):%H:%M}",  # f-string continued: the end of the shifted window
                      # conditional expression: no contrast at all means no data in the shifted window
                      "outcome": "untestable" if not pr else ("fails" if posc else "survives"),
                      "overlap": f"{len(pr)} contrasts, {nbig} in the predicted direction beyond 0.30",  # dict entry: the number of contrasts and those beyond 0.30 in the predicted direction
                      "overlap_min": None, "x5_l1_gain_minus_ref_db": None, "ublox_l1_noise_minus_ref": None})  # dict entry: the columns that do not apply to this test stay empty
        # 11a: does 1.1.22 (up to 1 W, L1) overlap a capture, and what did the receivers show? (descriptive)
        e = next(p for p in splan if p["test_code"] == "1.1.22")  # its shifted entry
        w0, w1 = pd.Timestamp(e["start_utc"]), pd.Timestamp(e["end_utc"])  # its window
        hits = []  # (capture, start, end) overlaps with whole captures
        for _, sr_ in sessions.iterrows():  # for loop over sessions
            lo = max(w0, pd.Timestamp(sr_["capture_start_utc"], tz="UTC"))  # overlap start
            hi = min(w1, pd.Timestamp(sr_["capture_end_utc"], tz="UTC"))  # overlap end
            if hi > lo:  # if: they overlap
                hits.append((sr_["test_code"], lo, hi))  # kept
        # boolean masks: the mosaic-X5 GPS L1/E1 gain readings
        x5 = long[(long.receiver == "Septentrio_02") & (long.band == "GPSL1/E1") & (long.field == "agc_gain_db")]
        # lambda: boolean mask inside any overlap; with no overlap an all-False mask (t != t is False for times)
        inn = lambda t_: np.logical_or.reduce([(t_ >= lo) & (t_ < hi) for _, lo, hi in hits]) if hits else t_ != t_
        gv = x5[inn(x5.timestamp_utc)].value  # X5 L1 gain during the overlap
        nv = nzq[inn(nzq.timestamp_utc)].value  # u-blox noise during the overlap
        # tuple assignment: the gain and noise references
        rg_, rn_ = refs.get(("Septentrio_02", "GPSL1/E1", "agc_gain_db")), refs.get(("u-blox_01", "L1", "noise_per_ms"))
        srows.append({"shift_h": h, "test": "1.1.22 overlap with a capture (descriptive)",  # the row
                      "window_utc": f"{w0:%H:%M}-{w1:%H:%M}", "outcome": "overlaps" if hits else "no overlap",  # dict entry: the shifted window, and whether it overlaps a capture
                      "overlap": "; ".join(f"{c_} {lo:%H:%M:%S}-{hi:%H:%M:%S}" for c_, lo, hi in hits),  # dict entry: each overlapping capture with its overlap times
                      "overlap_min": round(sum((hi - lo).total_seconds() for _, lo, hi in hits) / 60, 2),  # dict entry: total overlap in minutes
                      # dict entries: median departures over the whole overlap (it may include our own windows)
                      "x5_l1_gain_minus_ref_db": round(float(gv.median() - rg_), 2) if len(gv) else None,
                      "ublox_l1_noise_minus_ref": round(float(nv.median() - rn_), 2) if len(nv) else None})  # dict entry: median u-blox noise during the overlap minus its reference
    sdf = pd.DataFrame(srows)  # the sensitivity table
    sdf.to_csv(PROCESSED / "ground_truth" / "schedule_shift_sensitivity.csv", index=False, lineterminator="\n")  # written (new)
    print("\n  (d) Test Area 1 and 2 windows shifted by +2 h and -2 h (the adopted-pipeline script for 1.1.22):")  # sub-heading
    for _, r in sdf.iterrows():  # for loop over rows
        # one line per shift and test
        print(f"      {r.shift_h:+d} h  {r.test:<46} {r.window_utc:<12} {r.outcome:<11} {r.overlap}"
              + (f"  gain {r.x5_l1_gain_minus_ref_db:+.1f} dB" if pd.notna(r.x5_l1_gain_minus_ref_db) else "")  # string concatenation: adds the gain departure when it was measured
              + (f"  noise {r.ublox_l1_noise_minus_ref:+.1f}" if pd.notna(r.ublox_l1_noise_minus_ref) else ""))  # string concatenation: adds the noise departure when it was measured
    if (sdf.outcome == "fails").any():  # if: any shifted test fails
        # records any failing shift as an anomaly
        flag("SCHEDULE SHIFT: a test fails under a +/-2 h shift of the Test Area 1/2 windows: "
             + "; ".join(f"{r.shift_h:+d} h {r.test}" for _, r in sdf[sdf.outcome == "fails"].iterrows()))  # string concatenation: lists each failing shift and test
        print("      *** A SHIFTED TEST FAILS: see the anomalies. stops here for the reader.")  # loud
    # Test Area 3's times against what the receivers recorded (item the environment script's direct evidence)
    ta3 = ta3_confirmation(long, sessions, own, refs, latd)  # one row per test and indicator
    ta3.to_csv(PROCESSED / "ground_truth" / "ta3_schedule_confirmation.csv", index=False, lineterminator="\n")  # written (new)
    # sub-heading of the TA3 table
    print("      Test Area 3 (our own tests) against the receivers (onset - start, recovery - end, floor exit - end of "
          "initial jamming, s):")  # sub-heading
    for _, r in ta3.iterrows():  # for loop over rows
        # one line per test and indicator: the three offsets in seconds ('!s' prints None as text)
        print(f"        {r.test_code:<7} {r.spoofer:<10} {r.receiver:<14} {r.onset_minus_start_s!s:>8} "
              f"{r.recovery_minus_end_s!s:>8} {r.floor_exit_minus_initial_end_s!s:>8}")  # one line per test and indicator
    for h in (2, -2):  # for loop over the two real shifts: how many verdict tests became untestable
        u_ = sdf[(sdf.shift_h == h) & ~sdf.test.str.contains("11a")]  # the three tests at this shift
        # one line per shift: survives / untestable / fails counts
        print(f"      at {h:+d} h: {int((u_.outcome == 'survives').sum())} survive, "
              f"{int((u_.outcome == 'untestable').sum())} untestable, {int((u_.outcome == 'fails').sum())} fail")  # f-string continued: the untestable and failing counts

    # (e) 31d: the 1590 MHz narrowband interferer, descriptive only
    ifr = rfb[(rfb.frequency_mhz >= 1589.5) & (rfb.frequency_mhz <= 1590.5)].sort_values("timestamp_utc")  # ~1590 MHz
    irows = []  # one row per receiver and continuous episode
    for (rx_, code_), g_ in ifr.groupby(["receiver", "test_code"]):  # per receiver and session
        ts_ = pd.to_datetime(g_.timestamp_utc, utc=True)  # its time stamps
        ep_ = (ts_.diff().dt.total_seconds() > 5).cumsum()  # episode number: a new one after a gap over 5 s
        sr_ = sessions[sessions.test_code == code_].iloc[0]  # the session row
        # tuple: the session's pre-test margin (capture start to scheduled start)
        pre_ = (pd.Timestamp(sr_["capture_start_utc"], tz="UTC"), pd.Timestamp(sr_["scheduled_start_utc"], tz="UTC"))
        for _, e_ in g_.groupby(ep_.to_numpy()):  # for loop over episodes
            a_, b_ = pd.Timestamp(e_.timestamp_utc.min()), pd.Timestamp(e_.timestamp_utc.max())  # its span
            # pyramid steps the episode overlaps, and whether in their jamming minutes
            stp = [st["step"] for st in pyr["steps"] if st["step_start_utc"] < b_ and st["step_end_utc"] > a_]
            def med(rxx, band, fld, t0_, t1_):  # nested function: median of one series between two times
                # boolean masks: one series between two times
                v_ = long[(long.receiver == rxx) & (long.band == band) & (long.field == fld)
                          & (long.timestamp_utc >= t0_) & (long.timestamp_utc <= t1_)].value  # the values
                return round(float(v_.median()), 2) if len(v_) else None  # median, or None
            irows.append({"receiver": rx_, "test_code": code_, "start_utc": f"{a_:%H:%M:%S}",  # the row
                          "end_utc": f"{b_:%H:%M:%S}", "duration_s": round((b_ - a_).total_seconds(), 1),  # dict entry: the end of the episode and its length in seconds
                          "n_epochs": len(e_), "frequency_mhz": sorted(set(e_.frequency_mhz)),  # dict entry: the number of epochs and the declared frequencies
                          "bandwidth_khz": sorted(set(e_.bandwidth_khz)), "mode": sorted(set(e_["mode"])),  # ["mode"]: the column, not .mode()
                          "antenna_id": sorted(set(e_.antenna_id)), "pyramid_steps_overlapped": stp,  # dict entry: the antennas, and the pyramid steps the episode overlaps
                          # dict entries: the X5 gains and the u-blox CW suppression during it, and before the test
                          "x5_l1_gain_db": med("Septentrio_02", "GPSL1/E1", "agc_gain_db", a_, b_),
                          "x5_l1_gain_pre_test_db": med("Septentrio_02", "GPSL1/E1", "agc_gain_db", *pre_),  # dict entry: median mosaic-X5 L1 gain before the test
                          "x5_g1_gain_db": med("Septentrio_02", "GLOL1", "agc_gain_db", a_, b_),  # dict entry: median mosaic-X5 G1 gain during the episode
                          "x5_g1_gain_pre_test_db": med("Septentrio_02", "GLOL1", "agc_gain_db", *pre_),  # dict entry: median mosaic-X5 G1 gain before the test
                          "ublox_l1_cw": med("u-blox_01", "L1", "cw_suppression", a_, b_),  # dict entry: median u-blox CW suppression during the episode
                          "ublox_l1_cw_pre_test": med("u-blox_01", "L1", "cw_suppression", *pre_)})  # dict entry: median u-blox CW suppression before the test
    idf = pd.DataFrame(irows)  # the table
    idf.to_csv(PROCESSED / "ground_truth" / "interferer_1590mhz.csv", index=False, lineterminator="\n")  # written (new)
    seen = sorted(set(idf.receiver)) if len(idf) else []  # receivers that declared it
    # sub-heading: which receivers declared the 1590 MHz band and which did not
    print(f"\n  (e) The 1590 MHz narrowband interferer: declared by {seen or 'none'}; not by "
          f"{sorted(set(rfb.receiver) - set(seen))}")  # f-string continued: the receivers that did not declare it
    for _, r in idf.iterrows():  # for loop over episodes
        # one line per episode: when, how long, frequency, bandwidth, mode, pyramid steps and the indicators
        print(f"      {r.receiver:<14} {r.test_code:<7} {r.start_utc}-{r.end_utc} ({r.duration_s:.0f} s, {r.n_epochs} "
              f"epochs), {r.frequency_mhz} MHz, {r.bandwidth_khz} kHz, mode {r['mode']}, pyramid steps "  # f-string continued: the frequency, bandwidth, mode and overlapped pyramid steps
              f"{r.pyramid_steps_overlapped}; X5 L1 gain {r.x5_l1_gain_db} dB (before the test "  # f-string continued: the steps, and the mosaic-X5 L1 gain during the episode
              # f-string continued: the gain before the test, and the u-blox CW suppression during and before
              f"{r.x5_l1_gain_pre_test_db}); u-blox CW {r.ublox_l1_cw} (before {r.ublox_l1_cw_pre_test})")

    # (f) the 2.6.12 timeline on host time
    tl, tq, tinteg, nleap = timeline_2612(sessions, site, latd)  # timeline, quality table, integrity rows, leap checks
    tl.to_csv(PROCESSED / "ground_truth" / "spoof_2612_timeline.csv", index=False, lineterminator="\n")  # written (new)
    # per-second NAV-PVT fix quality, u-blox 01 and 02, 15:49-15:56 host time (verification only)
    tq.to_csv(PROCESSED / "ground_truth" / "spoof_2612_navpvt_quality.csv", index=False, lineterminator="\n")
    # sub-heading of the timeline, with the leap-second check count
    print(f"\n  (f) The 2.6.12 timeline on host time; RXM-RAWX leap seconds checked in {nleap} "
          f"pre-test epochs: all {LEAP_SECONDS} s; over the whole capture, attack included:"  # f-string continued: the leap-second value in the pre-test epochs
          f"{sorted(tinteg.ubx_facts['RAWX leapS, whole 2.6.12 capture'])} s")  # f-string continued: the leap-second values over the whole capture, attack included
    for _, r in tl.iterrows():  # for loop over events
        # one line per receiver and event; the value, when there is one, with its unit
        # host time, or "check" for the two rows that are not events, or "not observed"
        when = r.host_time_utc or ("check" if r.event.startswith(("pre-test", "no valid")) else "not observed")
        # one line per event
        print(f"      {r.receiver_name:<10} {r.event:<46} {r.threshold:<6} {when:<14} "
              + (f"{r.value} {r.unit}" if pd.notna(r.value) else "") + (f"  [{r.note}]" if r.note else ""))  # string concatenation: the value with its unit, and the note when there is one
    ti_ = pd.DataFrame(tinteg.rows)  # the timeline reads' integrity rows
    tfail = {}  # failure type -> count, for the timeline's own block and message types
    for d_ in ti_.failed_block_types:  # for loop over files
        for k_, c_ in (d_ or {}).items():  # failure types in that file
            tfail[k_] = tfail.get(k_, 0) + c_  # added
    # the timeline reads' own integrity: blocks read and failures (reported, not a stop)
    print(f"      timeline reads: {int(ti_.blocks_or_messages.sum())} blocks/messages, failures {tfail or 'none'} "
          "(descriptive outputs; a failed block is skipped, never decoded)")

    # ---- figures ---------------------------------------------------------------
    # Figure 9
    fig_status_timeseries(sept, ubx, plan, sessions, f"{d_lo:.2f} to {d_hi:.2f} km",
                          GRAPH / "receiver_status_timeseries.png")  # argument of the figure function: the output path of the receiver status figure
    best = "2.6.12"  # default test for the figure
    # if: the control found a deflection in Septentrio gain (the unit the figure's choice uses)
    if not ctrl.empty and (ctrl.deflection_detected & (ctrl.field == "agc_gain_db")).any():
        # method chain: the hits, a new column a = |difference| (assign with a lambda), max per test, idxmax
        # chosen in one unit only (Septentrio gain, dB), so a u-blox count can no longer win by its scale
        best = (ctrl[ctrl.deflection_detected & (ctrl.field == "agc_gain_db")]
                .assign(a=lambda x: x.inside_minus_outside.abs())  # method chain: adds the absolute inside-minus-outside difference as a column
                .groupby("test_code").a.max().idxmax())  # method chain: the session whose largest difference is greatest
    # the figure, for the session with the largest gain deflection in dB
    fig_control(sept, ubx, plan, sessions, best, GRAPH / "receiver_positive_control.png")
    if (rfb.receiver == "Septentrio_02").any() if not rfb.empty else False:  # if: any mosaic-X5 band rows
        # the figure
        fig_waterfall(rfb, pyr, plan, sessions, GRAPH / "rfstatus_frequency_waterfall.png")
    fig_excursions(traces, sessions, refs, GRAPH / "front_end_excursions.png")  # the figure
    # the resolution is set in the helpers module.save_fig, so it is named there rather than typed here
    print("\n[12] Figures 9 to 12 written to Graph/ (resolution set by the helpers module.save_fig).")

    # ---- write the tidy table ---------------------------------------------------
    # list of Septentrio columns kept in the combined status table
    keep_s = ["test_code", "session_folder", "receiver", "receiver_family",
              "timestamp_utc", "band", "frontend_code", "antenna_id", "agc_gain_db",
              "pll_unlocked", "sample_var", "blanking_pct", "rf_n_bands",
              "rf_flag_notauthentic", "rf_flag_nma_fail", "quality_overall",
              "quality_main_signal", "quality_rf_power_main", "n_satellites",
              "cpu_load_pct", "temperature_c", "exposure_group", "inside_our_capture"]
    # list of u-blox columns kept
    keep_u = ["test_code", "session_folder", "receiver", "receiver_family",
              "timestamp_utc", "band", "rf_block_id", "agc_cnt", "cw_suppression",
              "noise_per_ms", "jamming_state", "ant_status", "ant_power", "mag_i",
              "mag_q", "n_satellites", "receiver_reported_time_utc",
              "reported_minus_host_s", "exposure_group", "inside_our_capture"]
    a = sept[[c for c in keep_s if c in sept.columns]].copy()  # the kept Septentrio columns that exist
    b = ubx[[c for c in keep_u if c in ubx.columns]].copy()  # the kept u-blox columns that exist
    out = pd.concat([a, b], ignore_index=True)  # stacked; missing columns become empty
    # timestamps as fixed text with microseconds (method call continues on the next line)
    out["timestamp_utc"] = pd.to_datetime(out.timestamp_utc, utc=True).dt.strftime(
        "%Y-%m-%d %H:%M:%S.%f")
    if "receiver_reported_time_utc" in out.columns:  # if: the u-blox reported-time column exists
        # formatted to whole seconds; errors="coerce" turns unparseable or missing values into NaT (empty)
        out["receiver_reported_time_utc"] = pd.to_datetime(
            out.receiver_reported_time_utc, utc=True, errors="coerce").dt.strftime(  # argument of to_datetime(): unparseable times become empty; then formatted to whole seconds
            "%Y-%m-%d %H:%M:%S")
    # sort by session, receiver, time and band, then write the 38 MB table
    out.sort_values(["test_code", "receiver", "timestamp_utc", "band"]).to_csv(
        PROCESSED / "ground_truth" / "receiver_status_timeseries.csv", index=False, lineterminator="\n")  # argument of to_csv(): the output path of the status table, Unix line ends

    # ---- verdict ------------------------------------------------------------------
    print("\n" + "=" * 78)  # rule
    print("PLAIN-LANGUAGE REPORT AND VERDICT")  # heading
    print("=" * 78)  # rule
    # the question, with the measured distances (1)
    print(f"\n  The question. Did the 50 W jamming transmitted from Test Area 1, {d_lo:.2f} to {d_hi:.2f} km")
    print("  from the recording positions, reach the recording antenna?")  # continued
    # the instruments; the antenna arrangement is not stated in any capture file, so none is claimed
    print("\n  The instruments. Three Septentrio receivers and two u-blox receivers, recording")
    print("  the same signal environment and reporting their own interference status. The")  # continued
    print("  Septentrio reports AGC gain per front-end and, in RFStatus, the frequencies in")  # continued
    print("  which it has detected interference.")  # continued
    # test A summary
    print(f"\n  Test A, group comparison: {n_match} of {len(cdf)} contrasts in the "
          f"direction fixed in advance, {len(big)} beyond an effect size of 0.30.")  # f-string continued: the contrasts beyond an effect size of 0.30
    if not tdf.empty:  # if: timing results exist
        # test B: reported for transparency only, withdrawn as evidence
        print(f"  Test B, transition timing (withdrawn as evidence):{near} of {len(ti)} usable contrasts "
              "within 30 s of a scheduled edge.")
    # test C summary
    print(f"  Test C, the pyramid: {n_pyr_match} of {len(pdf)} contrasts in the predicted "
          f"direction, {len(pyr_big)} beyond 0.30.")  # f-string continued: the contrasts beyond 0.30
    # the control summary (the number of sessions)
    print(f"\n  Positive control: our own spoofing is registered in {n_sess_ctrl} of "
          f"{len(sessions)} sessions.")  # f-string continued: the number of sessions in all
    print(f"  1.1.29 attribution: the deflection follows {verdict_1129}.")  # the 1.1.29 rule's outcome
    # decision rule over the two tests that remain evidence (timing is withdrawn): a test is positive when
    # half its contrasts (at least 3) are in the predicted direction beyond 0.30 for A, or 40 % (at least 2) for C
    pos_a = len(big) >= max(3, 0.5 * len(cdf))  # boolean: test A positive
    pos_c = len(pyr_big) >= max(2, 0.4 * max(len(pdf), 1))  # boolean: test C positive
    print("\n  VERDICT")  # verdict heading
    print("  " + "-" * 74)  # rule
    if pos_a and pos_c:  # if: both tests positive
        print("  THE JAMMING REACHED THE RECORDING ANTENNA.")  # verdict text
        print("  Both remaining tests are positive on survey-grade equipment.")  # continued
    elif pos_a or pos_c:  # elif: one positive and one negative (a mixed branch, as in the handset script)
        print("  MIXED EVIDENCE: ONE TEST IS POSITIVE AND ONE NEGATIVE.")  # verdict text
        print(f"  Group comparison positive: {pos_a}; pyramid positive: {pos_c}. No verdict is drawn.")  # continued
    elif control_passes:  # elif: both negative and the control passed (the branch this data reaches)
        print("  THE 50 W JAMMING AT TEST AREA 1 DID NOT REACH THE RECORDING ANTENNA")  # verdict text
        print("  AT A LEVEL THE RECEIVERS COULD DETECT.")  # continued
        print("  The group comparison and the pyramid test are negative on survey-grade")  # continued
        print("  equipment, and the positive control passes: the same receivers register")  # continued
        print("  our own milliwatt spoofer, and in 2.6.12 the u-blox clock is pulled forward")  # continued
        # the ratio is computed from the two scheduled powers
        print("  by about the interval the catalogue specifies (ground_truth/spoof_2612_timeline.csv). A source")
        # the power ratio, computed from the two scheduled powers (50 W and 1 mW), and the measured distances
        # both powers sourced - the largest Test Area 1 power in the plan and our tests' planned power
        p50_ = max(p["power_w"] for p in plan if p["power_w"] is not None and p["test_area"] == 1)  # W
        p1_ = float(sessions.power_w_plan.max())  # our spoofer's planned power, W (ground_truth/session_ground_truth.csv)
        # the power ratio from the two sourced powers, and the measured distances
        print(f"  radiating {p50_ / p1_:,.0f} times more power, {d_lo:.2f} to {d_hi:.2f} km away, produced")
        print(f"  no corresponding response. Test 1.1.29, a jammer at an assumed {d1129:.2f} km,")  # continued
        print("  produced none either.")  # continued
        print("\n  CONFIDENCE: high. It rests on instruments designed to report interference,")  # confidence
        print("  agreeing with each other and with the consumer handsets of the handset script,")  # continued
        print("  and on a control that demonstrates the measurement would have seen the")  # continued
        print("  jamming had it arrived.")  # continued
        print("\n  CONSEQUENCE. The clean margins identified in the timeline script are clean of")  # consequence
        print("  Test Area 1 jamming (ground_truth/margin_clean_minutes.csv gives what survives every")  # continued
        print("  indicator), and the three sessions that lie inside the 1.8.2 window describe")  # continued
        print("  spoofing rather than spoofing against a jammed background. Our own")  # continued
        print("  motorcade's jamming, reported in section [11] above, is a separate and real")  # continued
        print("  constraint on the feature set.")  # continued
    else:  # else: the control failed too
        print("  INSUFFICIENT EVIDENCE.")  # verdict text
        print("  The tests are negative but the positive control did not pass, so the")  # continued
        print("  measurement cannot distinguish absence of jamming from insensitivity.")  # continued
    print("  " + "-" * 74)  # rule

    if NOTES:  # if: any notes
        print("\n" + "-" * 78)  # rule
        print(f"NOTES ({len(NOTES)})")  # heading with the count
        print("-" * 78)  # rule
        for n in NOTES:  # for loop over notes
            print(f"  - {n}")  # printed as bullets
    print("\n" + "-" * 78)  # rule
    print(f"ANOMALIES ({len(ANOMALIES)})")  # heading with the count
    print("-" * 78)  # rule
    for a_ in ANOMALIES or ["none"]:  # "or" substitutes ["none"] when the list is empty
        print(f"  - {a_}")  # printed as bullets
    print("\n" + "=" * 78)  # rule
    print(f"Random seed fixed at {SEED}.")  # seed note
    print("=" * 78)  # closing rule


if __name__ == "__main__":  # if: true only when the file is run directly
    main()  # runs the section
