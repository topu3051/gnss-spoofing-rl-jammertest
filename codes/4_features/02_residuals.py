#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
The residual engine. Nothing written here is a feature: the window-feature script builds the features from these
tables, and its barred-column test keeps the check columns (position offsets, clock terms) out of every feature table.

For the mosaic-X5 (SBF_02), the mosaic-H main antenna (SBF_03) and u-blox 01 (UBX_01) in all six sessions, at 1 Hz (first epoch of each
second, as the observables script), on GPS C1C, GLONASS C1C, Galileo C1C/C1X and BeiDou C2I only (the L1-band codes):
  - satellite orbits and clocks come from the IGS merged broadcast file under a CAUSAL rule : at
    receiver time t a record is used only if its transmission time is at or before t and t lies inside its validity span, the record
    with the nearest reference time winning (GLONASS through the GLONASS orbit script, the message frame time playing the transmission time);
  - every model quantity is evaluated at the time the receiver itself reports. The u-blox 01 epochs of 2.6.12 whose time the
    attack moved have no valid record and therefore no residual. Host time only places a row on the label grid (the evidence script's index);
  - two variants: "self", at a position solved here each epoch from the same pseudoranges (GPS, Galileo, BeiDou MEO/IGSO;
    GLONASS is computed at that position with its own clock), and "known", at the position script site reference;
  - a screen for whole-millisecond pseudorange jumps, run before any clock is estimated;
  - the residual about each constellation's median clock, at least four satellites above 10 degrees (the observables script's definition);
  - a causal per-satellite baseline of each satellite's own past residuals and the de-biased residual (the robust band and the 10 m cap as the feature-time version of the freeze: a sample with
    |res| > 10 m never enters the baseline, so a pass that starts on a counterfeit signal cannot seed it).

Pre-registered checks: R1 reproduction of the observables script (G, E, C) and of the GLONASS broadcast check (R), R2 clean level, R3 position sanity, R4
causality, R5 the R19 10 ms rows of 2.6.4; and the BeiDou transmission-time check.

Reads   Data/rinex/<session>/*.crx.gz, *.nav (BeiDou transmission time and toe only), the IGS file (Data/external/igs_brdc),
        Data/processed/{session_ground_truth,reference_positions,reference_clean_intervals,glonass_channels,rinex_ubx01_2612_pieces}.csv,
        the 2.6.12 u-blox 01 binary and host-time sidecar (grid placement only, through the evidence script), Data/processed/features/observables_1hz.csv.gz and
        preprocessing/glonass_residuals_clean.csv (R1 and the BeiDou check only), labels/labels_1hz.csv (the label column picks clean seconds for the checks)
Writes  Data/processed/features/residuals_{self,known}.csv.gz   one row per satellite-second with a pseudorange
        Data/processed/features/epochs_{self,known}.csv.gz      one row per second with a pseudorange (solver check)
        Data/processed/checks/residuals_checks.csv, residuals_numbers.csv, beidou_transmission_time.csv, residual_reproduction.png, residuals_gate.csv,
        residuals_gate_diagnosis.csv
        Graph/residual_engine.png, Graph/glonass_baseline.png
"""

from __future__ import annotations  # future import: type hints kept as text (a convention across the project's scripts)

import gzip  # standard-library gzip, to read the IGS file
import hashlib  # standard-library hashing, for the protected-file check
import importlib.util  # standard-library module loader, for modules whose names start with a digit
import os  # operating-system access: MPLCONFIGDIR is set before matplotlib loads
import random  # standard-library random; seeded below although nothing here draws from it
import sys  # standard-library system access: sys.path and sys.modules
import time  # standard-library clock, for the run-time lines printed to the terminal
import datetime as dtm  # standard-library dates and times
from collections import defaultdict, deque  # a dict with a default value, and a queue with a fixed maximum length
from pathlib import Path  # Path objects for building file paths

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
os.environ["MPLCONFIGDIR"] = str(PROJECT / "codes" / ".mplconfig")  # matplotlib's cache inside the project folder

import numpy as np  # NumPy: arrays, medians, linear algebra
import pandas as pd  # pandas: tables
import matplotlib  # plotting library
matplotlib.use("Agg")  # file-only backend, no window
import matplotlib.pyplot as plt  # the pyplot interface
import matplotlib.dates as mdates  # date tick locators and formatters for the time axes

sys.path.insert(0, str(PROJECT / "codes" / "common"))  # puts codes/ first on the import path so the helpers module is found
from helpers import exit_if_failed, save_fig  # byte-stable figure saving; exit_if_failed ends the run on a failed check

SEED = 20260914  # integer constant: the project's fixed random seed
random.seed(SEED)  # seeds Python's generator (nothing here draws from it)
np.random.seed(SEED)  # seeds NumPy's generator (nothing here draws from it)

PROC = PROJECT / "Data" / "processed"  # where the input CSVs are
OUT = PROC / "features"  # where this section's tables go
CHECK_DIR = PROJECT / "Data" / "processed" / "checks"  # where the check tables go
GRAPH = PROJECT / "Graph"  # where the figures go


def load(name: str, filename: str):  # function: loads codes/<filename> as a module (names start with a digit)
    spec = importlib.util.spec_from_file_location(name, PROJECT / "codes" / filename)  # where to load from
    mod = importlib.util.module_from_spec(spec)  # an empty module object
    sys.modules[name] = mod  # registered, so the module's own imports resolve
    spec.loader.exec_module(mod)  # runs the file's top level (its main() is guarded and not run)
    return mod  # the module


E = load("observables_for_residuals", "4_features/01_observables.py")  # 05e: the 1 Hz reader of all four constellations, and through it the evidence script and the position script
D, M = E.D, E.M  # the evidence script (spans, site positions, the 2.6.12 u-blox host-time index) and the position script (orbits, geometry, corrections)
S04 = D.S04  # the conversion script: the GPS/Galileo record reader, the Kepler orbit, the IGS file path
G = load("glonass_orbit_for_residuals", "2_preprocessing/04_glonass_orbit.py")  # 08a: the GLONASS broadcast-orbit propagator

DAY = dtm.date(2026, 9, 14)  # the one recording date
# the BeiDou eligibility lag is a constant, fixed at the value the transmission-time check found (0 s: no qualifying
# record, residuals_numbers bds_qualifying 0). The check still runs, but it chooses its evidence seconds from labels/labels_1hz.csv, so letting it set the
# lag would let a relabel move every BeiDou residual: labels must never shape a feature
BDS_LAG_S = 0.0  # float constant (s): the checked BeiDou lag
C = M.C_LIGHT  # speed of light, m/s
MASK = D.MASK_DEG  # the 10 degree elevation mask of the clock and the residuals (the observables script)
MS_M = C * 1.0e-3  # one millisecond of range, 299,792.458 m
JUMP_MIN_M = 0.5 * MS_M  # a jump is at least half a millisecond of range from the constellation's median
JUMP_TOL_M = 1000.0  # and within 1 km of a whole number of milliseconds (R19's remainder is about 7 m)
JUMP_KMAX = 20  # at most 20 ms, the length of one navigation-message bit
MAX_IT = 10  # Gauss-Newton iterations allowed before an epoch counts as not converged
CONV_M = 1.0e-4  # convergence: the position update falls below 0.1 mm (the position script's rule)
REJECT_M = M.REJECT_M  # 30 m: a post-fit residual beyond this is rejected, by deletion residual
HOLD_S = 30.0  # a previous solution is reused (start point or held position) only if at most 30 s old on the receiver's clock
EARTH_R = (6.0e6, 6.8e6)  # a Bancroft start whose radius lies outside this band (m) is not used
BASE_N, BASE_MIN, BASE_FREE = 300, 30, 5  # baseline: last 300 accepted samples; defined from 30; the first 5 of a pass are accepted freely
BAND_K, BAND_FLOOR_M, BASE_CAP_M = 4.0, 2.0, 10.0  # robust band: 4 scaled MADs, at least 2 m; the subtracted baseline is capped at +-10 m
MAD_K = 1.4826  # the factor that makes a median absolute deviation estimate a standard deviation for normal data
PASS_GAP_S = 60.0  # a satellite's new pass starts after 60 s (receiver time) without a residual
VALID_S = {"G": 2 * 3600.0, "E": 4 * 3600.0, "C": 4 * 3600.0}  # dict: how long after toe a record may be used (GPS: end of the 4 h fit; Galileo, BeiDou: the project's 4 h rule)
PERTURB_M = 1000.0  # the causality test shifts a record's clock by 1 km of range
SESSIONS, RXS = D.SESSIONS, D.RXS  # the six sessions in time order and the three receivers
RXN = {"SBF_02": "mosaic-X5", "SBF_03": "mosaic-H main", "UBX_01": "u-blox 01"}  # receiver names for figures and tables
VARIANTS = ("self", "known")  # tuple: the two residual variants

CHECKS: list = []  # list: the checks, written to residuals_checks.csv
FAILED: list = []  # list: names of the self-checks that failed; the script ends the run with status 1 when it is not empty
NUM: dict = {}  # dict: every number the summary quotes, written to residuals_numbers.csv


def check(name: str, expected: str, found, ok: bool | None) -> None:  # function: records one check and prints it
    CHECKS.append({"check": name, "expected": expected, "found": found, "ok": "" if ok is None else bool(ok)})  # append a dict row
    if ok is not None and not bool(ok):  # if: a pass/fail check (None marks a reported number) that did not pass
        FAILED.append(name)  # its name kept for the exit at the end of the run
    print(f"  [{'ok' if ok else ('FAIL' if ok is False else '--')}] {name}: expected {expected}; found {found}")  # progress line


def sha(path: Path) -> str:  # function: SHA-256 of a file as hex text
    return hashlib.sha256(path.read_bytes()).hexdigest()  # the whole file hashed


def protected_ok() -> tuple:  # function: compares the protected files with the hashes recorded before the residual engine first ran
    rows = [l.split() for l in (CHECK_DIR / "expected_hashes_features.txt").read_text().splitlines() if l.strip()]  # list comprehension: (hash, path) pairs
    bad = [p for h, p in rows if sha(PROJECT / p) != h]  # list comprehension: files whose hash changed
    return len(rows), bad  # how many were compared, and which differ


# ---------------------------------------------------------------- records and their transmission times

def igs_ttm() -> dict:  # function: transmission time of every GPS, Galileo (I/NAV) and BeiDou record that 04 and the position script keep, per satellite, in their order
    fl = lambda s: float(s.replace("D", "E")) if s.strip() else 0.0  # lambda: a Fortran D-exponent field as a float, 0 when blank (as the conversion script reads it)
    lines = S04.nav_text_lines(S04.IGS_BRDC)  # the IGS file's lines, decompressed in memory
    k = next(i for i, l in enumerate(lines) if l[60:].startswith("END OF HEADER")) + 1  # generator with next(): the first data line
    recs, cur = [], None  # the records; the record being read
    for l in lines[k:]:  # for loop over the data lines
        if l[:1].isalpha():  # if: a record starts (its first character is the constellation letter)
            cur = [l]  # a new record
            recs.append(cur)  # kept
        elif cur is not None and l.strip():  # elif: a continuation line
            cur.append(l)  # appended to the current record
    out = defaultdict(list)  # satellite -> list of (toe, transmission time) in GPS seconds
    skipped = defaultdict(int)  # constellation letter -> records short or unparsed (counted, and the run stops if any, as the conversion script and the position script now do)
    for r in recs:  # for loop over the records
        sv = r[0][:3]  # the satellite name
        if sv[0] not in "GEC":  # if: not GPS, Galileo or BeiDou
            continue  # skipped
        if len(r) < 8:  # if: a record shorter than its eight lines
            skipped[sv[0]] += 1  # counted instead of skipped silently
            continue  # not usable
        try:  # try: a malformed record raises, and 04 skips it
            toc = S04.datetime(int(r[0][4:8]), int(r[0][9:11]), int(r[0][12:14]), int(r[0][15:17]), int(r[0][18:20]), int(r[0][21:23]), tzinfo=S04.TZ_UTC)  # time of clock, calendar
            o = [[fl(x[4 + 19 * j:23 + 19 * j]) for j in range(4)] for x in r[1:8]]  # nested list comprehension: broadcast orbit lines 1-7, four values each
        except (ValueError, IndexError):  # except: a record that does not parse
            skipped[sv[0]] += 1  # counted instead of skipped silently
            continue  # next record
        if abs((toc.date() - DAY).days) > S04.EPHEMERIS_MAX_AGE_DAYS or o[5][1] != 0:  # if: dated off the day or unhealthy (the 04 and the position script filter)
            continue  # not used
        if sv[0] == "E" and S04.galileo_source(int(o[4][1])) != "I/NAV":  # if: a Galileo record that is not I/NAV (04 with inav_only=True)
            continue  # not used
        shift = M.BDT_EPOCH_GPS_S if sv[0] == "C" else 0.0  # conditional expression: BeiDou weeks count from 2006 on BDT, shifted to GPS seconds as the position script does
        out[sv].append((o[4][2] * 604800 + o[2][0] + shift, o[4][2] * 604800 + o[6][0] + shift))  # (toe, transmission time): week times 604800 s plus seconds of week
    if skipped:  # if: any record was short or failed to parse
        sys.exit(f"STOP: broadcast records short or unparsed in the IGS file, per constellation: {dict(skipped)}")  # stops instead of choosing among fewer records
    return {sv: np.array(sorted(v, key=lambda x: x[0])) for sv, v in out.items()}  # dict comprehension: sorted by toe; Python's sort is stable, so equal toe keep file order as in the conversion script


class Ephem:  # class: GPS, Galileo and BeiDou records (read by the conversion script and the position script) with their transmission times, and the record choice
    def __init__(self, causal: bool, bds_lag: float = 0.0, perturb: tuple | None = None):  # constructor: causal or nearest choice; BeiDou lag; optional test perturbation (T, after)
        self.causal = causal  # True: only records already sent; False: the position script's nearest record
        self.rec = dict(S04.ephemeris_kepler([S04.IGS_BRDC], DAY, inav_only=True))  # GPS and Galileo records exactly as the position script loads them
        self.rec.update(M.bds_records(S04, S04.IGS_BRDC, DAY))  # BeiDou records exactly as the position script loads them
        tt = igs_ttm()  # the transmission times of the same records, same order
        for sv, r in self.rec.items():  # for loop over the satellites
            if not np.array_equal(tt[sv][:, 0], r["toe_abs"]):  # if: the two readers disagree on the list of records
                raise SystemExit(f"STOP: record lists differ for {sv}")  # the transmission times would sit on the wrong records
            r["ttm_abs"] = tt[sv][:, 1] + (bds_lag if sv[0] == "C" else 0.0)  # transmission time; BeiDou shifted by the checked lag
            if perturb is not None:  # if: the causality test asks for a perturbation
                sel = r["ttm_abs"] > perturb[0] if perturb[1] else r["ttm_abs"] <= perturb[0]  # conditional expression: records sent after T (the test) or by T (the control)
                r["af0"] = r["af0"] + sel * (PERTURB_M / C)  # their clock bias moved by 1 km of range (a new array; the reader's data stay untouched)

    def pick(self, sv: str, t: np.ndarray) -> tuple:  # method: the record used at each receiver time, whether one exists, and why not
        r = self.rec[sv]  # the satellite's records
        d = np.abs(t[:, None] - r["toe_abs"][None, :])  # broadcasting: time from every epoch (rows) to every record's toe (columns)
        near = d <= S04.EPH_MAX_AGE_S  # within the 4 h the conversion script allows
        if self.causal:  # if: the causal rule
            near &= (t[:, None] - r["toe_abs"][None, :]) <= VALID_S[sv[0]]  # inside the constellation's validity span after toe
            elig = near & (r["ttm_abs"][None, :] <= t[:, None])  # and already sent at the epoch
        else:  # else: the position script's nearest record
            elig = near  # any record within 4 h
        k = np.where(elig, d, np.inf).argmin(axis=1)  # nearest eligible record per epoch; argmin takes the first of equal values, as the position script does
        ok = elig[np.arange(len(t)), k]  # fancy indexing: whether that record is eligible at all
        why = np.where(ok, "", np.where(near.any(axis=1), "not_yet_sent", "no_record"))  # nested where: why an epoch has no record
        return k, ok, why  # three arrays, one value per epoch

    def state(self, sv: str, t: np.ndarray, pr: np.ndarray, k: np.ndarray, dt: float = 0.0) -> tuple:  # method: positions (n x 3) and clocks (s) at transmission time, receive time moved by dt
        p = {key: v[k] for key, v in self.rec[sv].items()}  # dict comprehension: each field of the chosen record, one value per epoch
        tt = t + dt - pr / C  # transmission time, first guess: receive time minus travel time
        if sv[0] == "C":  # if: BeiDou
            geo = np.full(len(t), int(sv[1:]) in M.BDS_GEO)  # GEO flag per epoch (all equal; GEO rows are dropped before this)
            x1, d1 = M.bds_xyz(p, tt, geo)  # position and clock at the first guess (the position script's BeiDou code)
            x2, d2 = M.bds_xyz(p, tt - d1, geo)  # again, the time corrected by the satellite clock
        else:  # else: GPS or Galileo
            mu = S04.MU_GPS if sv[0] == "G" else S04.MU_GAL  # conditional expression: the system's gravitational parameter
            x1, d1 = S04.kepler_xyz(p, tt, mu)  # position and clock at the first guess (the conversion script's Kepler code)
            x2, d2 = S04.kepler_xyz(p, tt - d1, mu)  # again, the time corrected by the satellite clock
        return x2, d2  # positions and clock offsets


def frame_times() -> dict:  # function: (satellite, tb in GPS seconds) -> message frame time in GPS seconds, for every GLONASS record of the IGS file
    with gzip.open(G.IGS_GZ, "rt", errors="replace") as fh:  # the IGS file opened as text
        lines = fh.read().split("\n")  # its lines
    k0 = next(i for i, l in enumerate(lines) if l[60:].startswith("END OF HEADER")) + 1  # the first data line
    out = {}  # the result
    for l in lines[k0:]:  # for loop over the data lines
        if l[:1] == "R" and l[1:3].isdigit():  # if: the first line of a GLONASS record
            utc = dtm.datetime(int(l[4:8]), int(l[9:11]), int(l[12:14]), int(l[15:17]), int(l[18:20]), int(l[21:23]), tzinfo=dtm.timezone.utc)  # tb, UTC calendar
            tbu = (utc - G.GPS_EPOCH).total_seconds()  # tb on the UTC scale, seconds since the GPS epoch
            tf = np.floor(tbu / 604800.0) * 604800.0 + G._f(l[61:80])  # frame time: seconds of the UTC week (RINEX 3.04, first line, fourth value) plus the week's start
            tf += -604800.0 if tf - tbu > 302400.0 else (604800.0 if tbu - tf > 302400.0 else 0.0)  # nested conditional: the frame time belongs to the week nearest tb
            out[(l[:3], tbu + G.LEAP_SECONDS)] = tf + G.LEAP_SECONDS  # both on the GPS scale (UTC plus 18 s), keyed exactly as the GLONASS orbit script computes tb
    return out  # the table


class GloCausal(G.GloOrbits):  # class: the GLONASS orbit script's GLONASS orbits with the causal record choice (subclass: everything else is the GLONASS orbit script's)
    def __init__(self, causal: bool, perturb: tuple | None = None):  # constructor: causal or the GLONASS orbit script's choice; optional test perturbation (T, after)
        super().__init__()  # the GLONASS orbit script reads the records and the channel table
        self.causal = causal  # True: only records already sent
        ft = frame_times()  # (satellite, tb) -> frame time
        self.ttm = {}  # satellite -> frame times of its healthy records, aligned with the GLONASS orbit script's self.tb
        for sv, g in self.rec.items():  # for loop over the satellites (g is the GLONASS orbit script's own table of that satellite's healthy records)
            self.ttm[sv] = np.array([ft[(sv, tb)] for tb in g.tb_gps])  # list comprehension: each record's frame time
            if perturb is not None:  # if: the causality test
                sel = self.ttm[sv] > perturb[0] if perturb[1] else self.ttm[sv] <= perturb[0]  # conditional expression: records sent after T, or by T
                g["clk"] = g.clk + sel * (PERTURB_M / C)  # their clock moved by 1 km of range (this object's own copy of the table)

    def pick(self, sv: str, t_gps: float):  # method: the GLONASS orbit script's record choice, restricted to records already sent when causal (overrides the GLONASS orbit script's pick)
        if not self.causal or sv not in self.tb:  # if: the GLONASS orbit script's rule, or a satellite without healthy records
            return super().pick(sv, t_gps)  # exactly the GLONASS orbit script
        d = np.abs(t_gps - self.tb[sv])  # time to every healthy record's tb
        d = np.where(self.ttm[sv] <= t_gps, d, np.inf)  # records not yet sent are out of reach
        k = int(d.argmin())  # the nearest; argmin returns the first minimum, so the earlier tb wins a tie, as in the GLONASS orbit script
        if d[k] > G.MAX_AGE_S:  # if: nothing usable within 15 min
            return None  # not usable
        return k  # the record index

    def why(self, sv: str, t_gps: float) -> str:  # method: why a GLONASS satellite-epoch has no record (same rule as pick)
        if sv not in self.tb:  # if: no healthy record at all
            return "no_record"  # unhealthy or absent from the file
        near = np.abs(t_gps - self.tb[sv]) <= G.MAX_AGE_S  # records within 15 min
        return "not_yet_sent" if near.any() else "no_record"  # conditional expression: one exists but was not yet sent, or none exists


# ---------------------------------------------------------------- observations and satellite states

def read_obs(gt: pd.DataFrame, test: str, rx: str) -> pd.DataFrame:  # function: 1 Hz observations of a session and receiver; t_rx keeps the receiver's own time
    if (test, rx) == ("2.6.12", "UBX_01"):  # if: the u-blox whose time the attack moved
        orig = D.read_full  # the evidence script's own reader, remembered
        D.read_full = lambda path, spans, r: E.read_all(path, spans, r).assign(t_rx=lambda d: d.t_gps)  # lambda: the observables script's four-constellation reader, receiver time copied before the evidence script re-times the rows
        try:  # try: the original reader always comes back
            obs = D.ubx_2612_obs(gt)  # the three pieces placed on the label grid by host time; t_rx keeps the receiver's time
        finally:  # finally: runs even after an error
            D.read_full = orig  # the evidence script's reader restored
    else:  # else: every other session and receiver
        obs = E.read_all(M.obs_path(gt.loc[test, "session_folder"], test, rx), D.capture_spans(gt, test, rx), rx)  # observations in the capture span (2.6.5 from the site start)
        obs["t_rx"] = obs.t_gps  # the receiver's time is the epoch time itself
    D.LINES.clear()  # frees the decompressed RINEX
    obs = obs[obs.pr.notna() & obs.sv.str[0].isin(list("GREC"))]  # rows with a pseudorange of the four constellations
    return obs.sort_values(["sec", "sv"]).reset_index(drop=True)[["sec", "t_rx", "sv", "pr"]]  # time order, the four columns the engine needs


def sat_states(obs: pd.DataFrame, eph: Ephem, glo: GloCausal) -> dict:  # function: orbit, clock and range-rate inputs for every row; they depend on the pseudorange and the receiver time only, not on the variant
    n = len(obs)  # number of rows
    S = {"sv": obs.sv.to_numpy(), "sys": obs.sv.str[0].to_numpy(), "pr": obs.pr.to_numpy(), "t": obs.t_rx.to_numpy()}  # dict: the row inputs as arrays
    for key in ("pos", "pa", "pb", "vel"):  # for loop over the three-column quantities
        S[key] = np.full((n, 3), np.nan)  # position, position 0.5 s later and earlier (G, E, C), velocity (R)
    for key in ("dts", "da", "db", "gam", "chan"):  # for loop over the one-column quantities
        S[key] = np.full(n, np.nan)  # clock, clock 0.5 s later and earlier, GLONASS clock rate and channel
    S["ok"], S["why"] = np.zeros(n, bool), np.full(n, "", dtype=object)  # usable flag and the reason when not
    for sv, idx in obs.groupby("sv").indices.items():  # for loop over the satellites; idx holds that satellite's row positions
        if sv[0] == "R":  # if: GLONASS, handled per epoch below
            continue  # next satellite
        if sv[0] == "C" and int(sv[1:]) in M.BDS_GEO:  # if: a BeiDou GEO satellite (orbit code not validated, 5.2.7)
            S["why"][idx] = "bds_geo"  # reason recorded
            continue  # no orbit computed
        if sv not in eph.rec:  # if: the IGS file has no healthy record for it at all
            S["why"][idx] = "no_record"  # reason recorded
            continue  # next satellite
        t, pr = S["t"][idx], S["pr"][idx]  # its receiver times and pseudoranges
        k, ok, why = eph.pick(sv, t)  # the record per epoch, chosen at the receiver's time
        x0, d0 = eph.state(sv, t, pr, k)  # position and clock at transmission
        xa, da = eph.state(sv, t, pr, k, +0.5)  # half a second later, same record (no look-ahead at a record change)
        xb, db = eph.state(sv, t, pr, k, -0.5)  # half a second earlier, same record
        ok = ok & np.isfinite(x0).all(axis=1)  # a finite position is required
        S["pos"][idx], S["dts"][idx], S["pa"][idx], S["da"][idx], S["pb"][idx], S["db"][idx] = x0, d0, xa, da, xb, db  # stored
        S["ok"][idx], S["why"][idx] = ok, np.where(ok, "", why)  # usable flag and reason
    rows = np.flatnonzero(S["sys"] == "R")  # GLONASS row positions
    sub = obs.iloc[rows]  # their rows
    for sec, ii in sub.groupby("sec").indices.items():  # for loop over the seconds with GLONASS rows
        j = rows[ii]  # their positions in the full table
        svs, t = list(S["sv"][j]), float(S["t"][j[0]])  # satellites and the epoch's receiver time
        p, v, d, o = glo.state_full(svs, t, S["pr"][j])  # 08a: positions, velocities, clocks and usable mask at transmission time
        S["pos"][j], S["vel"][j], S["dts"][j], S["ok"][j] = p, v, d, o  # stored (unusable rows hold zeros, masked by ok)
        for jj, s in zip(j, svs):  # for loop over the epoch's GLONASS satellites
            if S["ok"][jj]:  # if: usable
                k = glo.pick(s, t)  # the record used (the same call the GLONASS orbit script made)
                S["gam"][jj] = glo.rec[s].gam.iloc[k]  # its clock rate GammaN, for the range rate
                S["chan"][jj] = glo.channel(int(s[1:]))  # the slot's frequency channel, for the ionosphere scale
            else:  # else: no record
                S["why"][jj] = glo.why(s, t)  # the reason
    S["pos"][~S["ok"]] = np.nan  # unusable rows carry no position
    return S  # the dict of arrays


# ---------------------------------------------------------------- one epoch: z, the screen, the clock, the solver

def evaluate(x: np.ndarray, t: float, j: np.ndarray, S: dict, kl: tuple) -> tuple:  # function: z (receiver clock still in), elevation, azimuth and line of sight of rows j (one epoch) at position x
    sysl = S["sys"][j]  # constellation letters
    rho, u, el, az = M.geometry(x, S["pos"][j])  # Sagnac-corrected range, unit vector satellite->receiver, elevation, azimuth (radians)
    corr = np.zeros(len(j))  # troposphere plus ionosphere per row
    g = sysl != "R"  # GPS, Galileo, BeiDou rows
    if g.any():  # if: any
        corr[g] = M.corrections(x, t, list(S["sv"][j][g]), el[g], az[g], kl)  # 05c: Saastamoinen plus Klobuchar (scaled to B1I for BeiDou)
    if (~g).any():  # if: any GLONASS rows
        corr[~g] = G.path_delay(x, t, S["chan"][j][~g].astype(int), el[~g], az[~g], kl)  # 08a: the same, Klobuchar scaled to each slot's G1 frequency
    z = S["pr"][j] - rho + C * S["dts"][j] - corr  # measured minus modelled range, receiver clock still in (the evidence script and the GLONASS orbit script form)
    return z, np.degrees(el), np.degrees(az), -u  # z, elevation and azimuth in degrees, unit vector receiver->satellite


def screen(z: np.ndarray, sysl: np.ndarray, on: bool = True) -> np.ndarray:  # function: whole-millisecond jump per row (0 = none), judged against the constellation's median before any clock is estimated (D7a)
    k = np.zeros(len(z), dtype=int)  # no jump yet
    if not on:  # if: the screen is switched off (R1 reproduces the published tables, which had none)
        return k  # nothing flagged
    for s in sorted(set(sysl)):  # for loop over the constellations present, in a fixed order
        m = (sysl == s) & np.isfinite(z)  # that constellation's rows with a value
        if m.sum() < 3:  # if: fewer than three, a median cannot single one out
            continue  # not screened
        d = z[m] - np.median(z[m])  # each satellite's distance from the constellation's median
        kk = np.rint(d / MS_M)  # the nearest whole number of milliseconds
        hit = (np.abs(d) >= JUMP_MIN_M) & (np.abs(d - kk * MS_M) <= JUMP_TOL_M) & (np.abs(kk) >= 1) & (np.abs(kk) <= JUMP_KMAX)  # at least half a millisecond away and within 1 km of whole milliseconds
        k[np.flatnonzero(m)[hit]] = kk[hit].astype(int)  # the jump in milliseconds, signed
    return k  # one value per row


def clock_res(z: np.ndarray, el: np.ndarray, sysl: np.ndarray, jump: np.ndarray) -> tuple:  # function: each constellation's median clock (at least 4 above the mask, jumps left out) and every such satellite's residual
    clk, res = np.full(len(z), np.nan), np.full(len(z), np.nan)  # the clock of each row's constellation and the residual
    for s in sorted(set(sysl)):  # for loop over the constellations present
        m = (sysl == s) & np.isfinite(z) & (el >= MASK) & (jump == 0)  # the satellites that enter the clock
        if m.sum() >= 4:  # if: four or more, so one bad pseudorange cannot move the median far (the observables script)
            c = float(np.median(z[m]))  # the constellation's clock term this second, metres
            clk[sysl == s] = c  # stored on every row of the constellation (check)
            res[m] = z[m] - c  # residual about it
    return clk, res  # two arrays


def bancroft(pos: np.ndarray, prc: np.ndarray) -> np.ndarray | None:  # function: Bancroft's closed-form position from four or more clock-corrected pseudoranges; a start point that needs no prior position
    B = np.column_stack([pos, prc])  # one row per satellite: x, y, z, pseudorange
    lor = lambda a, b: a[0] * b[0] + a[1] * b[1] + a[2] * b[2] - a[3] * b[3]  # lambda: the Lorentz inner product of two 4-vectors
    alpha = 0.5 * (np.sum(pos ** 2, axis=1) - prc ** 2)  # half the Lorentz norm of each row
    Bp = np.linalg.pinv(B)  # least-squares inverse of B
    u, v = Bp @ np.ones(len(prc)), Bp @ alpha  # the two vectors of the solution y = M (v + lambda u)
    a, b, c = lor(u, u), 2.0 * (lor(u, v) - 1.0), lor(v, v)  # quadratic in lambda: a lambda^2 + b lambda + c = 0
    disc = b * b - 4.0 * a * c  # discriminant
    if a == 0.0 or disc < 0.0:  # if: no real root
        return None  # no start point
    best = None  # the root closest to the Earth's radius
    for lam in ((-b + np.sqrt(disc)) / (2.0 * a), (-b - np.sqrt(disc)) / (2.0 * a)):  # for loop over the two roots
        xx = (v + lam * u)[:3]  # the position part (M only flips the fourth component)
        r = np.linalg.norm(xx)  # its distance from the Earth's centre
        if EARTH_R[0] <= r <= EARTH_R[1] and (best is None or abs(r - 6.371e6) < abs(np.linalg.norm(best) - 6.371e6)):  # if: on the Earth's surface and closer than the other root
            best = xx  # kept
    return best  # position or None


def solve(x0: np.ndarray, t: float, sv: np.ndarray, sysl: np.ndarray, pr: np.ndarray, pos: np.ndarray, dts: np.ndarray, kl: tuple) -> dict | None:  # function: one epoch's weighted least squares, position plus one clock per constellation with 4 or more above the mask, the position script's rejection; None if unsolvable or not converged
    use = np.ones(len(sv), bool)  # satellites still in the solution
    x = x0.copy()  # the position, refined from the start point
    n_rej, n_it = 0, 0  # rejections and iterations so far
    while True:  # loop: solve, then reject the worst residual if beyond 30 m, then solve again
        conv = False  # not converged yet
        for _ in range(MAX_IT):  # up to MAX_IT Gauss-Newton iterations
            idx = np.flatnonzero(use)  # satellites in use
            rho, u, el, az = M.geometry(x, pos[idx])  # geometry from the current position
            m = np.degrees(el) >= MASK  # the elevation mask, from the current position
            idx, rho, u, el, az = idx[m], rho[m], u[m], el[m], az[m]  # only those above it
            systems = [s for s in "GEC" if (sysl[idx] == s).sum() >= 4]  # list comprehension: constellations with four or more satellites
            keep = np.isin(sysl[idx], systems)  # rows of those constellations
            idx, rho, u, el, az = idx[keep], rho[keep], u[keep], el[keep], az[keep]  # only those
            if not systems or len(idx) < 3 + len(systems) + 1:  # if: no constellation, or no redundancy left
                return None  # cannot be solved this epoch
            z = pr[idx] - rho + C * dts[idx] - M.corrections(x, t, list(sv[idx]), el, az, kl)  # measured minus modelled, clock still in
            H = np.zeros((len(idx), 3 + len(systems)))  # design matrix: three position columns plus one clock column per constellation
            H[:, :3] = u  # position partials: unit vectors from the satellite to the receiver
            for c, s in enumerate(systems):  # for loop over the constellations
                H[:, 3 + c] = sysl[idx] == s  # that constellation's clock column (1 for its satellites)
            w = np.sin(el) ** 2  # elevation weights
            A = H.T @ (H * w[:, None])  # weighted normal matrix
            try:  # try: a degenerate geometry makes A singular
                th = np.linalg.solve(A, H.T @ (w * z))  # position correction and the clocks
            except np.linalg.LinAlgError:  # except: singular
                return None  # cannot be solved
            x = x + th[:3]  # updated position
            n_it += 1  # one more iteration
            if np.linalg.norm(th[:3]) < CONV_M:  # if: converged to 0.1 mm
                conv = True  # converged
                break  # stop iterating
        if not conv:  # if: MAX_IT iterations without converging
            return None  # no solution (the rule fixed beforehand)
        r = z - H @ th  # post-fit residuals
        if np.abs(r).max() > REJECT_M:  # if: any beyond 30 m
            Hw = H * np.sqrt(w)[:, None]  # design matrix scaled by the square root of the weights
            lev = np.einsum("ij,jk,ik->i", Hw, np.linalg.inv(A), Hw)  # einsum: leverage of each observation (diagonal of the hat matrix)
            use[idx[(np.abs(r) / np.maximum(1 - lev, 1e-6)).argmax()]] = False  # the largest deletion residual goes (the position script's rule)
            n_rej += 1  # counted
            continue  # solve again from the current position
        return {"x": x, "n_rej": n_rej, "n_it": n_it, "n": {s: int((sysl[idx] == s).sum()) for s in "GEC"}}  # the solution, rejections, iterations and satellites per constellation


def run_variant(variant: str, obs: pd.DataFrame, S: dict, site: np.ndarray, kl: tuple, screen_on: bool = True) -> tuple:  # function: residuals and solver check of one session and receiver under one variant
    n = len(obs)  # rows
    out = {key: np.full(n, np.nan) for key in ("z", "el", "az", "ex", "ey", "ez", "rr", "clk", "res")}  # dict comprehension: the per-row results
    out["jump"] = np.zeros(n, dtype=int)  # jump in milliseconds, 0 for none
    reason = S["why"].copy()  # why a row has no residual, starting from the orbit reasons
    eps = []  # one dict per second: the solver check
    last_x, last_t = None, None  # the most recent real solution and its receiver time (self variant)
    ix = obs.groupby("sec").indices  # dict: second -> its row positions
    for sec in sorted(ix):  # for loop over the seconds in time order
        j_all = ix[sec]  # the second's rows
        j = j_all[S["ok"][j_all]]  # the rows with an orbit
        t = float(S["t"][j_all[0]])  # the receiver's time of this epoch
        ep = {"sec": int(sec), "start": "", "solved": 0, "pos_held": 0, "n_it": 0, "n_rej": 0, "n_pos_G": 0, "n_pos_E": 0, "n_pos_C": 0}  # dict literal: this second's check row
        if variant == "known":  # if: the known-antenna variant
            x0, ep["start"] = site, "site"  # the position script reference is both start and position
        elif last_x is not None and abs(t - last_t) <= HOLD_S:  # elif: a recent real solution exists
            x0, ep["start"] = last_x, "previous"  # it is the start point
        else:  # else: a cold start
            g = j[S["sys"][j] != "R"]  # GPS, Galileo, BeiDou rows with an orbit
            x0 = bancroft(S["pos"][g], S["pr"][g] + C * S["dts"][g]) if len(g) >= 4 else None  # conditional expression: Bancroft's closed form, or nothing with fewer than four
            ep["start"] = "bancroft" if x0 is not None else "none"  # which start was used
        k0 = None  # the screen at the start point
        if x0 is not None and len(j):  # if: a start point and rows with an orbit
            z0 = evaluate(x0, t, j, S, kl)[0]  # z at the start point
            k0 = screen(z0, S["sys"][j], screen_on)  # millisecond jumps found there, before any clock
        x = None  # the position the residuals are computed at
        if variant == "known":  # if: known antenna
            x = site  # the site reference
        else:  # else: self-solved
            sol = None  # no solution yet
            if k0 is not None:  # if: the screen ran
                g = (S["sys"][j] != "R") & (k0 == 0)  # GPS, Galileo, BeiDou rows without a jump
                sol = solve(x0, t, S["sv"][j][g], S["sys"][j][g], S["pr"][j][g], S["pos"][j][g], S["dts"][j][g], kl)  # the epoch's solution
            if sol is not None:  # if: solved
                x, last_x, last_t = sol["x"], sol["x"], t  # used, and remembered for the next epochs
                ep.update({"solved": 1, "n_it": sol["n_it"], "n_rej": sol["n_rej"], "n_pos_G": sol["n"]["G"], "n_pos_E": sol["n"]["E"], "n_pos_C": sol["n"]["C"]})  # check
            elif last_x is not None and abs(t - last_t) <= HOLD_S:  # elif: no solution, but a real one at most 30 s old
                x, ep["pos_held"] = last_x, 1  # the held position, marked (pos_held)
        if x is None or not len(j):  # if: no position, or no row with an orbit
            reason[j] = "no_position"  # rows with an orbit get this reason
            ep.update({"n_jump": 0, "n_jump_moved": 0, "h_off_m": np.nan, "v_off_m": np.nan})  # nothing else in the check row
            eps.append(ep)  # kept
            continue  # next second
        z, el, az, e = evaluate(x, t, j, S, kl)  # z, elevation, azimuth and line of sight at the position used
        kk = screen(z, S["sys"][j], screen_on)  # the screen again at that position (it decides the residual)
        clk, res = clock_res(z, el, S["sys"][j], kk)  # median clocks and residuals
        rr = np.full(len(j), np.nan)  # modelled range rate at a receiver at rest, m/s
        g = S["sys"][j] != "R"  # GPS, Galileo, BeiDou rows
        if g.any():  # if: any
            ea = M.geometry(x, S["pa"][j][g])[0] - C * S["da"][j][g]  # clock-corrected range half a second later (the evidence script model Doppler)
            eb = M.geometry(x, S["pb"][j][g])[0] - C * S["db"][j][g]  # and half a second earlier
            rr[g] = ea - eb  # their difference over one second: the range rate
        if (~g).any():  # if: GLONASS rows
            rr[~g] = np.sum(e[~g] * S["vel"][j][~g], axis=1) - C * S["gam"][j][~g]  # line of sight times satellite velocity, minus the clock rate
        for key, val in (("z", z), ("el", el), ("az", az), ("ex", e[:, 0]), ("ey", e[:, 1]), ("ez", e[:, 2]), ("rr", rr), ("clk", clk), ("res", res)):  # for loop over the results
            out[key][j] = val  # stored on the second's rows
        out["jump"][j] = kk  # the jumps
        rs = np.where(kk != 0, "jump", np.where(el < MASK, "below_mask", np.where(np.isnan(res), "clock_n<4", "")))  # nested where: why a row with an orbit has no residual
        reason[j] = rs  # stored
        hoff, voff = D.enu_dist(x, site)  # horizontal and vertical offset of the position used from the site (check only, never a feature)
        ep.update({"n_jump": int((kk != 0).sum()), "n_jump_moved": int((k0 != kk).sum()) if k0 is not None else 0, "h_off_m": float(hoff), "v_off_m": float(voff)})  # check
        eps.append(ep)  # kept
    df = obs[["sec", "sv"]].copy()  # the row keys
    for key in ("el", "az", "ex", "ey", "ez", "rr", "z", "clk", "res"):  # for loop over the results
        df[key] = out[key]  # attached
    df["jump_k"], df["reason"], df["t_rx"] = out["jump"], reason, S["t"]  # jumps, reasons and the receiver time (used by the baseline, not written)
    return df, pd.DataFrame(eps)  # the satellite table and the second table


def baseline(df: pd.DataFrame) -> pd.DataFrame:  # function: per satellite, the running median of its own accepted past residuals (D7b), the capped baseline, the de-biased residual and each sample's state (warm, in, out, gate)
    res, trx = df.res.to_numpy(), df.t_rx.to_numpy()  # residuals and receiver times
    base = np.full(len(df), np.nan)  # the baseline subtracted from each sample
    state = np.full(len(df), "", dtype=object)  # warm (baseline not yet defined), in or out of the band, or gate (|res| beyond the 10 m cap)
    for sv, idx in df.groupby("sv").indices.items():  # for loop over the satellites; rows are in time order within each
        acc = deque(maxlen=BASE_N)  # the accepted past samples; the oldest drops out when 300 are held
        last = None  # receiver time of the previous residual sample of this satellite
        for i in idx:  # for loop over the satellite's rows
            r = res[i]  # its residual
            if not np.isfinite(r):  # if: no residual (screened, masked, no record ...): not a sample
                continue  # skipped, the baseline untouched
            if last is not None and abs(trx[i] - last) >= PASS_GAP_S:  # if: 60 s or more since the previous sample on the receiver's clock
                acc.clear()  # a new pass: start again
            last = trx[i]  # remembered
            a = np.array(acc)  # the accepted past samples as an array (none from the present or the future)
            med = float(np.median(a)) if len(a) else np.nan  # conditional expression: their median, the baseline before capping
            if len(a) >= BASE_MIN:  # if: enough samples
                base[i] = min(max(med, -BASE_CAP_M), BASE_CAP_M)  # the baseline, capped at +-10 m
            if len(a) < BASE_FREE:  # if: the first few samples of a pass
                accept = True  # accepted without a band
            else:  # else: the robust band
                accept = abs(r - med) <= max(BAND_K * MAD_K * float(np.median(np.abs(a - med))), BAND_FLOOR_M)  # within 4 scaled MADs of the running median, at least 2 m
            gate = abs(r) > BASE_CAP_M  # the seeding gate: a sample beyond the cap never enters the baseline
            accept = accept and not gate  # gated samples are never accepted, during warm-up or after
            state[i] = "gate" if gate else (("in" if accept else "out") if len(a) >= BASE_MIN else "warm")  # nested conditional: the sample's state
            if accept:  # if: accepted
                acc.append(r)  # it joins the baseline's samples
    df = df.copy()  # a copy, so the caller's table is not changed
    df["base"], df["res_db"], df["base_state"] = base, res - base, state  # baseline, de-biased residual (NaN while warm), state
    return df  # the table


# ---------------------------------------------------------------- the BeiDou transmission-time check

def beidou_check(gt: pd.DataFrame) -> float:  # function: when does a new BeiDou record first appear in a receiver's own navigation file (transmission time and toe only); returns the eligibility lag
    lab = pd.read_csv(PROC / "labels" / "labels_1hz.csv", usecols=["test_code", "receiver", "sec", "label"])  # labels: only to pick clean seconds (a check, never a feature)
    clean = set(map(tuple, lab[lab.label == "clean"][["test_code", "receiver", "sec"]].to_numpy()))  # set of (session, receiver, second) that are clean
    ob = pd.read_csv(PROC / "features" / "observables_1hz.csv.gz", usecols=["test_code", "receiver", "sec", "sv"])  # 05e: which satellites each receiver tracked each second
    trk = set(map(tuple, ob[ob.sv.str[0] == "C"].to_numpy()))  # set of (session, receiver, second, satellite) with a BeiDou observation
    g0 = (pd.Timestamp(DAY, tz="UTC") - pd.Timestamp(M.GPS_EPOCH)).total_seconds() + M.LEAP_SECONDS  # the day's start in GPS seconds
    bdt_week0 = M.BDT_EPOCH_GPS_S + np.floor((g0 - M.BDT_EPOCH_GPS_S) / 604800.0) * 604800.0  # start of the day's BDT week in GPS seconds (from the date, not from the file)
    to_unix = lambda sow: bdt_week0 + sow - M.LEAP_SECONDS + M.GPS_UNIX  # lambda: BDT seconds of week -> UTC unix seconds
    fl = lambda s: float(s.replace("D", "E"))  # lambda: a Fortran D-exponent field as a float
    rows = []  # one row per receiver, session, satellite and toe
    for test in SESSIONS:  # for loop over the sessions
        folder = PROJECT / "Data" / "rinex" / gt.loc[test, "session_folder"]  # the session's RINEX folder
        for rx in RXS:  # for loop over the receivers
            if (test, rx) == ("2.6.12", "UBX_01"):  # if: the u-blox whose time the attack moved (its stamps are on that moved clock)
                continue  # not used for this check
            first = {}  # (satellite, toe) -> first transmission time seen
            for nav in sorted(folder.glob(f"*_{rx}*.nav")):  # for loop over the receiver's navigation files of the session
                lines = nav.read_text(errors="replace").split("\n")  # its lines
                k = next(i for i, l in enumerate(lines) if "END OF HEADER" in l) + 1  # the first data line
                for i in range(k, len(lines) - 7):  # for loop over the lines that can start a record
                    if lines[i][:1] == "C" and lines[i][1:3].isdigit():  # if: a BeiDou record starts here
                        toe, ttm = fl(lines[i + 3][4:23]), fl(lines[i + 7][4:23])  # ONLY toe (orbit line 3, first value) and the transmission time (line 7, first value)
                        key = (lines[i][:3], toe)  # the record's identity: satellite and toe
                        first[key] = min(first.get(key, np.inf), ttm)  # the earliest transmission time seen for it
            cap0 = D.T(gt.loc[test, "capture_start_utc"][11:])  # the capture start, unix seconds
            for (sv, toe), ttm in sorted(first.items()):  # for loop over the records, in a fixed order
                tu, fu = to_unix(toe), to_unix(ttm)  # toe and first transmission time, UTC unix seconds
                new = tu >= cap0 + 60  # the record's toe lies inside the capture (not a record dumped at start-up)
                before = all((test, rx, int(s), sv) in trk for s in range(int(tu) - 60, int(tu)))  # tracked in each of the 60 s before toe
                at_first = (test, rx, int(np.floor(fu)), sv) in trk  # tracked in the second of the first transmission time
                cln = all((test, rx, int(s)) in clean for s in range(int(tu) - 60, int(np.floor(fu)) + 1))  # every second from toe - 60 s to the first stamp is clean
                rows.append({"receiver": rx, "test_code": test, "sv": sv, "toe_utc": D.hms(tu), "first_ttm_utc": D.hms(fu), "lag_s": fu - tu, "new_in_capture": int(new), "tracked_60s_before": int(before), "tracked_at_first_ttm": int(at_first), "clean": int(cln), "qualifying": int(new and before and cln and rx == "SBF_02"), "role": "primary" if rx == "SBF_02" else "supplementary"})  # dict literal: one record's result
    df = pd.DataFrame(rows)  # the table
    df.to_csv(CHECK_DIR / "beidou_transmission_time.csv", index=False, float_format="%.1f")  # for the reader
    q = df[df.qualifying == 1]  # the records the pre-registered rule reads
    lag = float(max(q.lag_s.max(), 0.0)) if len(q) else 0.0  # conditional expression: the largest lag of a qualifying record, if positive (conservative: no record before this receiver had it)
    nw = df[(df.new_in_capture == 1) & (df.tracked_60s_before == 1)]  # new records of continuously tracked satellites, any label (supplementary)
    for rx in RXS:  # for loop over the receivers
        s = nw[nw.receiver == rx]  # its supplementary records
        NUM[f"bds_new_tracked_{rx}"] = len(s)  # how many
        NUM[f"bds_lag_counts_{rx}"] = "; ".join(f"{int(k)} s: {v}" for k, v in s.lag_s.value_counts().sort_index().items())  # generator expression: lag -> count
    odd = df[(df.new_in_capture == 1) & (df.tracked_at_first_ttm == 0) & (df.lag_s == 0)]  # stamped at toe in a second the receiver did not track the satellite
    NUM["bds_stamp_without_tracking"] = "; ".join(f"{r.receiver} {r.test_code} {r.sv} {r.toe_utc}" for r in odd.itertuples())  # listed
    check("BeiDou transmission time: records new in the capture on clean seconds, X5 (pre-registered evidence)", "lag 0 keeps toe as the start; a positive lag shifts eligibility", f"{len(q)} qualifying records; lag applied {lag:.0f} s; new records of tracked satellites on any second: X5 {NUM['bds_lag_counts_SBF_02'] or 'none'}; mosaic-H {NUM['bds_lag_counts_SBF_03'] or 'none'}; u-blox 01 {NUM['bds_lag_counts_UBX_01'] or 'none'}; stamped at toe while not tracked: {len(odd)}", None)  # reported
    NUM["bds_qualifying"], NUM["bds_lag_applied_s"] = len(q), lag  # kept
    return lag  # the lag the record choice uses


# ---------------------------------------------------------------- the run

def build(gt: pd.DataFrame, kl: tuple, lag: float) -> tuple:  # function: production tables (causal, screen on) for both variants, plus the R1 table (nearest record, screen off, known antenna, clean intervals)
    eph, glo = Ephem(True, lag), GloCausal(True)  # causal record choice
    eph_n, glo_n = Ephem(False), GloCausal(False)  # the position script's and the GLONASS orbit script's nearest-record choice, for R1
    tabs, eps, r1, OBS = defaultdict(list), defaultdict(list), [], {}  # outputs per variant; R1 rows; observations kept for R4
    for test in SESSIONS:  # for loop over the sessions
        site = D.site_xyz(M.SITE_OF[test])  # the session's site reference
        for rx in RXS:  # for loop over the receivers
            t0 = time.time()  # the clock, for the progress line
            obs = read_obs(gt, test, rx)  # its observations with a pseudorange
            OBS[(test, rx)] = obs  # kept for the causality test
            S = sat_states(obs, eph, glo)  # orbits and clocks under the causal rule
            for v in VARIANTS:  # for loop over the variants
                df, ep = run_variant(v, obs, S, site, kl)  # residuals and check
                df = baseline(df)  # the per-satellite baseline
                df.insert(0, "receiver", rx)  # receiver column first
                df.insert(0, "test_code", test)  # session column first
                ep.insert(0, "receiver", rx)  # likewise in the check
                ep.insert(0, "test_code", test)  # likewise
                tabs[v].append(df)  # kept
                eps[v].append(ep)  # kept
            cs = D.clean_spans(test, rx)  # the position script clean intervals of the session and receiver
            if cs:  # if: any
                o1 = obs[D.in_spans(obs.sec.to_numpy().astype(float), cs)].reset_index(drop=True)  # their observations
                d1, _ = run_variant("known", o1, sat_states(o1, eph_n, glo_n), site, kl, screen_on=False)  # the R1 computation
                d1.insert(0, "receiver", rx)  # receiver column
                d1.insert(0, "test_code", test)  # session column
                r1.append(d1)  # kept
            print(f"  {test} {rx}: {len(obs)} satellite-seconds, {time.time() - t0:.0f} s")  # progress line
    return {v: pd.concat(tabs[v], ignore_index=True) for v in VARIANTS}, {v: pd.concat(eps[v], ignore_index=True) for v in VARIANTS}, pd.concat(r1, ignore_index=True), OBS  # dict comprehensions: one table per variant


def write_tables(tab: dict, ep: dict) -> None:  # function: the residual and check tables, rounded so runs again are byte-identical
    OUT.mkdir(parents=True, exist_ok=True)  # the features folder
    r4 = {"el": 4, "az": 4, "ex": 7, "ey": 7, "ez": 7, "rr": 4, "z": 4, "clk": 4, "res": 4, "base": 4, "res_db": 4}  # dict: decimals per column (0.1 mm for metres)
    names = {"el": "el_deg", "az": "az_deg", "rr": "rr_m_s", "z": "z_m", "clk": "clk_m", "res": "res_m", "base": "base_m", "res_db": "res_db_m"}  # dict: written column names with units
    for v in VARIANTS:  # for loop over the variants
        d = tab[v].drop(columns="t_rx").round(r4).rename(columns=names)  # the receiver time is not written; rounded; units in the names
        E.write_gz(d, OUT / f"residuals_{v}.csv.gz")  # gzip with no time stamp (the observables script's writer)
        E.write_gz(ep[v].round({"h_off_m": 4, "v_off_m": 4}), OUT / f"epochs_{v}.csv.gz")  # the check table


# ---------------------------------------------------------------- checks

def same(a: np.ndarray, b: np.ndarray) -> np.ndarray:  # function: element-wise equality where NaN equals NaN
    return (a == b) | (pd.isna(a) & pd.isna(b))  # equal, or both missing


def r1_check(r1: pd.DataFrame) -> None:  # function: R1, the engine against the observables script (G, E, C) and the GLONASS broadcast check (R) on the clean reference intervals
    ob = pd.read_csv(PROC / "features" / "observables_1hz.csv.gz", usecols=["test_code", "receiver", "sec", "sv", "sys", "pr_res_m"])  # the observables script's published residuals
    m = r1[r1.sv.str[0] != "R"].merge(ob[ob.sys != "R"], on=["test_code", "receiver", "sec", "sv"], how="left")  # engine rows of G, E, C with the observables script's value attached
    both = m.res.notna() & m.pr_res_m.notna()  # rows where both have a residual
    one = int((m.res.notna() != m.pr_res_m.notna()).sum())  # rows where only one has
    dif = (m.res[both] - m.pr_res_m[both]).abs()  # absolute differences, metres (the observables script stores four decimals)
    NUM["r1_gec_rows"], NUM["r1_gec_both"], NUM["r1_gec_one_only"], NUM["r1_gec_maxdiff_mm"] = len(m), int(both.sum()), one, float(dif.max() * 1e3)  # kept
    check("R1 G/E/C: engine (nearest record, reference position, screen off) against the observables script pr_res_m on the clean intervals", "max |difference| <= 1 mm; no row with a residual in only one", f"{int(both.sum())} rows, max {dif.max() * 1e3:.3f} mm, {one} in only one", bool(dif.max() <= 1e-3 and one == 0))  # the test
    glo_res = pd.read_csv(PROC / "preprocessing" / "glonass_residuals_clean.csv")  # the GLONASS broadcast check's published GLONASS table
    g = glo_res.merge(r1[r1.sv.str[0] == "R"], on=["test_code", "receiver", "sec", "sv"], how="left")  # the GLONASS broadcast check rows with the engine's attached
    dz, dr = (g.z - g.z_clock_in_m).abs(), (g.res - g.res_median_m).abs()  # differences in z and the residual
    eq = (g.el.round(4) == g.el_deg) & (g.z.round(4) == g.z_clock_in_m) & (g.res.round(4) == g.res_median_m)  # rows equal after the GLONASS broadcast check's own rounding
    extra = int(r1[(r1.sv.str[0] == "R") & r1.res.notna()].merge(glo_res[["test_code", "receiver", "sec", "sv"]], how="left", indicator=True)._merge.eq("left_only").sum())  # engine residuals the GLONASS broadcast check does not have
    NUM["r1_r_rows"], NUM["r1_r_equal_rounded"], NUM["r1_r_maxdiff_mm"], NUM["r1_r_extra"] = len(g), int(eq.sum()), float(max(dz.max(), dr.max()) * 1e3), extra  # kept
    check("R1 R: engine against the GLONASS broadcast check's preprocessing/glonass_residuals_clean.csv (z and residual)", "byte-identical after the GLONASS broadcast check's rounding, or within 1 mm", f"{len(g)} rows, {int(eq.sum())} equal after rounding, max {max(dz.max(), dr.max()) * 1e3:.3f} mm, {extra} engine rows the GLONASS broadcast check lacks", bool(max(dz.max(), dr.max()) <= 1e-3 and extra == 0))  # the test
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))  # check figure: two panels
    ax[0].hist((m.res[both] - m.pr_res_m[both]) * 1e3, bins=60, color="#0072B2")  # G/E/C differences in mm
    ax[0].set_xlabel("engine minus the observables script residual (mm), G/E/C")  # axis label
    ax[0].set_ylabel("satellite-seconds")  # axis label
    ax[1].hist((g.res - g.res_median_m) * 1e3, bins=60, color="#D55E00")  # GLONASS differences in mm
    ax[1].set_xlabel("engine minus the GLONASS broadcast check residual (mm), GLONASS")  # axis label
    fig.suptitle("R1: engine in reproduction mode against the published tables (clean reference intervals)", fontsize=10)  # title
    fig.tight_layout()  # layout
    save_fig(fig, CHECK_DIR / "residual_reproduction.png")  # byte-stable save


def labelled(tab: pd.DataFrame) -> pd.DataFrame:  # function: the table with each row's label attached (checks only)
    lab = pd.read_csv(PROC / "labels" / "labels_1hz.csv", usecols=["test_code", "receiver", "sec", "label"])  # labels
    return tab.merge(lab, on=["test_code", "receiver", "sec"], how="left")  # attached


def pct(a, q) -> float:  # function: percentile of an array, NaN if empty
    a = np.asarray(a, dtype=float)  # as a float array
    return float(np.percentile(a, q)) if len(a) else float("nan")  # conditional expression: NumPy percentile


def r2_r3(tab: dict, ep: dict) -> None:  # function: R2 (clean residual level, self-solved) and R3 (self-solved position against the reference)
    rows = []  # one row per variant, receiver and constellation
    for v in VARIANTS:  # for loop over the variants
        t = labelled(tab[v])  # with labels
        t = t[t.label == "clean"]  # clean seconds only
        for rx in RXS:  # for loop over the receivers
            for s in "GECR":  # for loop over the constellations
                x = t[(t.receiver == rx) & (t.sv.str[0] == s)]  # this cell
                a = (x.res_db if s == "R" else x.res).abs().dropna()  # GLONASS: de-biased; others: raw
                rows.append({"variant": v, "receiver": rx, "sys": s, "quantity": "res_db" if s == "R" else "res", "n": len(a), "median_m": pct(a, 50), "p95_m": pct(a, 95), "raw_median_m": pct(x.res.abs().dropna(), 50), "raw_p95_m": pct(x.res.abs().dropna(), 95)})  # dict literal: the cell
    r2 = pd.DataFrame(rows)  # the table
    r2.round(4).to_csv(CHECK_DIR / "residuals_r2_levels.csv", index=False)  # written
    s = r2[r2.variant == "self"]  # the pre-registered rows
    fail = [f"{r.receiver} {r.sys} {r.median_m:.2f}/{r.p95_m:.2f}" for r in s.itertuples() if (r.median_m >= (1.5 if r.sys == "R" else 1.0)) or (r.p95_m >= (5.0 if r.sys == "R" else 4.0))]  # list comprehension: cells over the limits
    check("R2 clean level, self-solved: |residual| median and 95th pct per receiver; G, E, C raw < 1.0 / 4 m; R de-biased < 1.5 / 5 m", "all 12 cells inside", "; ".join(f"{r.receiver} {r.sys} {r.median_m:.2f}/{r.p95_m:.2f} (n {r.n})" for r in s.itertuples()) + (f"; FAILING: {fail}" if fail else ""), not fail)  # the test
    e = labelled(ep["self"])  # self-solved check with labels
    e = e[(e.label == "clean") & (e.solved == 1)]  # clean seconds with a real solution
    out = []  # per receiver
    for rx in RXS:  # for loop over the receivers
        x = e[e.receiver == rx]  # its clean solutions
        out.append((rx, len(x), float(x.h_off_m.median()), float(x.v_off_m.abs().median()), pct(x.h_off_m, 95), pct(x.v_off_m.abs(), 95)))  # medians and 95th percentiles
        NUM[f"r3_{rx}_h_med"], NUM[f"r3_{rx}_v_med"], NUM[f"r3_{rx}_n"] = out[-1][2], out[-1][3], out[-1][1]  # kept
    ok = all(o[2] <= 10.0 and o[3] <= 20.0 for o in out)  # generator expression: every receiver inside both limits
    check("R3 self-solved position on clean seconds against the position script reference, median", "horizontal <= 10 m and vertical <= 20 m", "; ".join(f"{RXN[o[0]]} n {o[1]}: {o[2]:.2f} / {o[3]:.2f} m (95th {o[4]:.2f} / {o[5]:.2f})" for o in out), ok)  # the test
    cl = labelled(ep["self"])  # the check again, with labels
    cl = cl[cl.label == "clean"]  # clean seconds
    NUM["clean_self_unsolved"], NUM["clean_self_held"] = int((cl.solved == 0).sum()), int((cl.pos_held == 1).sum())  # clean seconds without a solution, and with a held position
    tb = labelled(tab["self"])  # self table with labels
    tb = tb[tb.label == "clean"]  # clean rows
    NUM["clean_out_band"] = int((tb.base_state == "out").sum())  # clean samples outside the baseline band
    NUM["clean_in_band"] = int((tb.base_state == "in").sum())  # clean samples inside it
    NUM["clean_jumps"] = int((tb.jump_k != 0).sum())  # clean rows screened as jumps
    check("Baseline on clean seconds (self): samples outside the band; position holds; unsolved seconds", "reported (few expected)", f"out {NUM['clean_out_band']} of {NUM['clean_out_band'] + NUM['clean_in_band']}; held {NUM['clean_self_held']}; unsolved {NUM['clean_self_unsolved']}; jumps {NUM['clean_jumps']}", None)  # reported


def gate_report(tab: dict) -> None:  # function: the first answer: clean samples the seeding gate keeps out of the baseline, per receiver and satellite, and every clean sample beyond 10 m described
    ob = pd.read_csv(PROC / "features" / "observables_1hz.csv.gz", usecols=["test_code", "receiver", "sec", "sv", "cn0", "has_carrier", "cmc_rate_m_s"])  # 05e: C/N0 and carrier presence (check only)
    rows, flag = [], []  # per satellite counts; satellites over 2 %
    for v in VARIANTS:  # for loop over the variants
        t = labelled(tab[v])  # with labels
        c = t[(t.label == "clean") & t.res.notna()]  # clean samples with a residual
        g = c.assign(gated=c.base_state == "gate").groupby(["receiver", "sv"]).agg(n_clean=("res", "size"), n_gated=("gated", "sum")).reset_index()  # per receiver and satellite
        g["share"] = g.n_gated / g.n_clean  # share of its clean samples kept out
        g.insert(0, "variant", v)  # the variant
        rows.append(g[g.n_gated > 0])  # only satellites the gate touches
        flag += [f"{v} {r.receiver} {r.sv} {r.n_gated}/{r.n_clean}" for r in g[g.share > 0.02].itertuples()]  # list comprehension: over 2 %
        NUM[f"gate_clean_{v}"] = int(g.n_gated.sum())  # clean samples gated
    pd.concat(rows, ignore_index=True).round(4).to_csv(CHECK_DIR / "residuals_gate.csv", index=False)  # written
    check("Seeding gate on clean samples: per receiver and satellite, flag where it keeps out more than 2 % of the satellite's clean samples", "reported (flags listed)", f"self {NUM['gate_clean_self']}, known {NUM['gate_clean_known']} clean samples gated; over 2 %: {flag}", None)  # reported
    t = labelled(tab["self"])  # the self table with labels
    c = t[(t.label == "clean") & (t.res.abs() > BASE_CAP_M)].copy()  # clean samples beyond 10 m
    k = tab["known"][["test_code", "receiver", "sec", "sv", "res"]].rename(columns={"res": "res_known"})  # the known-antenna residual of the same rows
    c = c.merge(k, on=["test_code", "receiver", "sec", "sv"], how="left").merge(ob, on=["test_code", "receiver", "sec", "sv"], how="left")  # attached, with C/N0 and carrier
    c["ms_remainder_m"] = (c.z - c.clk) - np.rint((c.z - c.clk) / MS_M) * MS_M  # distance from a whole number of milliseconds (a jump would leave metres here and be over 150 km in size)
    c["utc"] = [D.hms(x) for x in c.sec]  # list comprehension: clock text
    c[["test_code", "receiver", "sv", "sec", "utc", "el", "az", "res", "res_known", "cn0", "has_carrier", "cmc_rate_m_s", "ms_remainder_m", "base_state"]].round(4).to_csv(CHECK_DIR / "residuals_gate_diagnosis.csv", index=False)  # for the reader
    NUM["clean_beyond10"] = "; ".join(f"{a} {b} {s_}: {n}" for (a, b, s_), n in c.groupby(["test_code", "receiver", "sv"]).size().items())  # generator expression: where they are


def r5_check(tab: dict) -> None:  # function: R5, the R19 10 ms rows of 2.6.4 are caught and do not move the other GLONASS satellites' clock
    glo_res = pd.read_csv(PROC / "preprocessing" / "glonass_residuals_clean.csv")  # the GLONASS broadcast check's table (anomaly = 1 marks the 128 rows)
    an = glo_res[glo_res.anomaly == 1][["test_code", "receiver", "sec", "sv"]]  # the anomalous rows
    for v in VARIANTS:  # for loop over the variants
        t = tab[v]  # the production table
        a = an.merge(t, on=["test_code", "receiver", "sec", "sv"], how="left")  # those rows in the engine
        caught = int((a.jump_k == -10).sum())  # flagged as a jump of -10 ms
        oth = t[(t.sv.str[0] == "R") & t.sec.isin(an.sec) & (t.test_code == "2.6.4")]  # the GLONASS rows of the same session and seconds
        oth = oth.merge(an[["receiver", "sec"]].drop_duplicates(), on=["receiver", "sec"])  # only the anomalous receiver-seconds
        ref = oth[(oth.jump_k == 0) & (oth.el >= MASK)].groupby(["receiver", "sec"]).z.median()  # independent median of the other satellites' z
        clk = oth.groupby(["receiver", "sec"]).clk.first()  # the engine's clock at those seconds
        dclk = (clk - ref).abs().max()  # largest difference
        glo_clean = glo_res[(glo_res.anomaly == 0)].merge(an[["test_code", "receiver", "sec"]].drop_duplicates(), on=["test_code", "receiver", "sec"])  # the GLONASS broadcast check's other rows at those seconds
        glo_clock = (glo_clean.z_clock_in_m - glo_clean.res_median_m).groupby([glo_clean.receiver, glo_clean.sec]).first()  # the GLONASS broadcast check's clock there (which still contained R19)
        shift = (clk.reindex(glo_clock.index) - glo_clock)  # how far the screen moves the clock
        NUM[f"r5_{v}_caught"], NUM[f"r5_{v}_shift_med_m"], NUM[f"r5_{v}_shift_max_m"] = caught, float(shift.abs().median()), float(shift.abs().max())  # kept
        check(f"R5 ({v}): R19 10 ms rows of 2.6.4 caught by the screen; clock of the other GLONASS satellites equals their own median", "128 of 128 caught (k = -10); clock difference 0", f"{caught} of {len(an)} caught; max |engine clock - median of others| {dclk:.2e} m; the screen moves the GLONASS broadcast check's clock by {shift.abs().median():.3f} m median, {shift.abs().max():.3f} m max", bool(caught == len(an) and dclk < 1e-9))  # the test


def r4_check(gt: pd.DataFrame, kl: tuple, lag: float, tab: dict, OBS: dict) -> None:  # function: R4 causality: records sent after T perturbed by 1 km of clock; nothing at receiver time <= T may change; with two positive controls
    cols = ["el", "az", "ex", "ey", "ez", "rr", "z", "clk", "res", "jump_k", "reason", "base", "res_db", "base_state"]  # every column compared
    n_cmp, n_bad, ctl_n, ctl_c = 0, 0, [], []  # rows compared, rows changed; control counts
    for test in SESSIONS:  # for loop over the sessions
        site = D.site_xyz(M.SITE_OF[test])  # site reference
        for rx in RXS:  # for loop over the receivers
            obs = OBS[(test, rx)]  # the observations
            prod = tab["known"][(tab["known"].test_code == test) & (tab["known"].receiver == rx)]  # production rows (known variant) of the pair
            if prod.res.notna().sum() == 0:  # if: no residual at all
                continue  # nothing to test
            T = float(np.median(prod.t_rx[prod.res.notna()]))  # the cut: median receiver time of the rows with a residual
            secT = int(prod.sec[prod.t_rx <= T].max())  # its grid second
            o = obs[obs.sec <= secT].reset_index(drop=True)  # the observations up to the cut
            S = sat_states(o, Ephem(True, lag, (T, True)), GloCausal(True, (T, True)))  # states with every record sent after T perturbed
            for v in VARIANTS:  # for loop over the variants
                d = baseline(run_variant(v, o, S, site, kl)[0])  # the perturbed run
                p = tab[v][(tab[v].test_code == test) & (tab[v].receiver == rx) & (tab[v].sec <= secT)].reset_index(drop=True)  # production rows of the same span
                keep = (p.t_rx <= T).to_numpy()  # rows at receiver time <= T
                eq = np.ones(len(p), bool)  # all equal so far
                for c in cols:  # for loop over the compared columns
                    eq &= same(d[c].to_numpy(), p[c].to_numpy())  # equality, NaN equal to NaN
                n_cmp += int(keep.sum())  # rows compared
                n_bad += int((~eq & keep).sum())  # rows that changed
            win = o[o.sec > secT - 900].reset_index(drop=True)  # the last 15 min before the cut, for the controls
            z_ref = run_variant("known", win, sat_states(win, Ephem(True, lag), GloCausal(True)), site, kl)[0].z.to_numpy()  # unperturbed z there
            z_n = run_variant("known", win, sat_states(win, Ephem(False, 0.0, (T, True)), GloCausal(False, (T, True))), site, kl)[0].z.to_numpy()  # nearest-record rule, same perturbation
            z_c = run_variant("known", win, sat_states(win, Ephem(True, lag, (T, False)), GloCausal(True, (T, False))), site, kl)[0].z.to_numpy()  # causal rule, records sent BY T perturbed
            ctl_n.append(int((~same(z_n, z_ref)).sum()))  # control 1: rows the look-ahead rule lets the future change
            ctl_c.append(int((~same(z_c, z_ref)).sum()))  # control 2: rows the records in use change
    NUM["r4_rows_compared"], NUM["r4_rows_changed"], NUM["r4_ctl_nearest"], NUM["r4_ctl_inuse"] = n_cmp, n_bad, sum(ctl_n), sum(ctl_c)  # kept
    check("R4 causality: every record sent after T (median receiver time of each pair) given +1 km of clock; rows at receiver time <= T, both variants, 14 columns incl. baseline", "0 rows change", f"{n_bad} of {n_cmp} rows changed", n_bad == 0)  # the test
    check("R4 positive controls, last 15 min before T, known variant: (1) the nearest-record rule with the same perturbation; (2) the causal rule with the records sent by T perturbed", "both change rows (the test can see a change)", f"(1) {sum(ctl_n)} rows changed; (2) {sum(ctl_c)} rows changed", sum(ctl_n) > 0 and sum(ctl_c) > 0)  # the controls


def other_checks(tab: dict, ep: dict, OBS: dict) -> None:  # function: counts, reconciliations and magnitudes
    ob = pd.read_csv(PROC / "features" / "observables_1hz.csv.gz", usecols=["test_code", "receiver", "sys", "has_pr"])  # the observables script's table
    n05 = ob[ob.has_pr == 1].groupby(["test_code", "receiver"]).size()  # satellite-seconds with a pseudorange per pair
    n07 = tab["self"].groupby(["test_code", "receiver"]).size()  # the engine's
    dn = (n07.reindex(n05.index) - n05).abs().sum()  # total difference
    check("Rows: engine satellite-seconds with a pseudorange against the observables script's has_pr count, all 18 pairs", "equal", f"{int(n07.sum())} against {int(n05.sum())}, total difference {int(dn)}", bool(dn == 0))  # the reconciliation
    t = tab["self"]  # the self table
    u = t[(t.test_code == "2.6.12") & (t.receiver == "UBX_01")]  # u-blox 01 in 2.6.12
    g0 = u.sec + M.LEAP_SECONDS - M.GPS_UNIX  # the grid second in GPS seconds (host-based for this pair)
    moved = u[(u.t_rx - g0).abs() > 2.0]  # rows whose receiver time is off the grid by more than 2 s
    secs_moved = moved.sec.nunique()  # seconds with moved receiver time
    with_res = int(moved.res.notna().sum())  # any residual among them (must be none)
    reasons = moved.reason.value_counts().to_dict()  # their reasons
    NUM["ubx2612_moved_seconds"], NUM["ubx2612_moved_rows"], NUM["ubx2612_moved_with_res"] = secs_moved, len(moved), with_res  # kept
    NUM["ubx2612_moved_first"], NUM["ubx2612_moved_last"] = D.hms(moved.sec.min()), D.hms(moved.sec.max())  # first and last such second (host)
    check("D2: u-blox 01 in 2.6.12, seconds whose receiver time is moved, against (792 of 1,624 RINEX epochs dated off the day)", "no residual in any of them", f"{secs_moved} grid seconds ({NUM['ubx2612_moved_first']}-{NUM['ubx2612_moved_last']} host), {len(moved)} rows, {with_res} with a residual; reasons {reasons}", with_res == 0)  # the test
    for v in VARIANTS:  # for loop over the variants
        x = tab[v]  # the table
        NUM[f"{v}_rows"], NUM[f"{v}_res_rows"], NUM[f"{v}_res_db_rows"] = len(x), int(x.res.notna().sum()), int(x.res_db.notna().sum())  # sizes
        NUM[f"{v}_reasons"] = "; ".join(f"{k or 'residual'} {n}" for k, n in x.reason.value_counts().sort_index().items())  # generator expression: rows per reason
        NUM[f"{v}_jumps"] = "; ".join(f"{a} {b} {c} {d}: {n}" for (a, b, c, d), n in x[x.jump_k != 0].assign(s=x.sv.str[0]).groupby(["test_code", "receiver", "s", "jump_k"]).size().items())  # where jumps were found
        NUM[f"{v}_not_yet_sent"] = int((x.reason == "not_yet_sent").sum())  # rows lost to the causal rule
        e = ep[v]  # the check
        NUM[f"{v}_seconds"], NUM[f"{v}_solved"], NUM[f"{v}_held"] = len(e), int(e.solved.sum()), int(e.pos_held.sum())  # seconds, solved, held
        NUM[f"{v}_starts"] = "; ".join(f"{k} {n}" for k, n in e.start.value_counts().sort_index().items())  # how each second started
        NUM[f"{v}_jump_moved"] = int(e.n_jump_moved.sum())  # rows whose screen verdict changed between the start point and the final position
        mag = x[x.res.notna()]  # rows with a residual
        check(f"Magnitudes ({v}): elevation, range rate and |z - clock| plausible", "el 10-90 deg; |rr| < 1000 m/s (MEO); |residual| reported", f"el {mag.el.min():.1f}-{mag.el.max():.1f}; rr {x.rr.min():.0f} to {x.rr.max():.0f} m/s; |res| median {mag.res.abs().median():.2f} m, 99.9th {pct(mag.res.abs(), 99.9):.0f} m, max {mag.res.abs().max():.0f} m", bool(mag.el.min() >= MASK and mag.el.max() <= 90.0 and np.nanmax(np.abs(x.rr)) < 1000.0))  # sanity
    check("Rows lost to the causal rule (reason not_yet_sent), self / known", "0 expected from the planned count", f"{NUM['self_not_yet_sent']} / {NUM['known_not_yet_sent']}", None)  # reported
    check("Screen verdict at the start point against the final position (self)", "reported", f"{NUM['self_jump_moved']} rows differ", None)  # reported


# ---------------------------------------------------------------- figures

def plot_residual_engine(tab: dict, ep: dict) -> None:  # function: the figure, the self-solved position and the residual levels on clean seconds
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))  # three panels
    e = labelled(ep["self"])  # check with labels
    e = e[(e.label == "clean") & (e.solved == 1)]  # clean seconds with a real solution
    for rx in RXS:  # for loop over the receivers
        x = e[e.receiver == rx]  # its rows
        ax[0].hist(x.h_off_m, bins=np.linspace(0, 8, 81), histtype="step", color=M.COLORS[rx], label=f"{RXN[rx]} (n {len(x)})")  # horizontal offset distribution
    ax[0].set_xlabel("horizontal offset of the self-solved position from the position script reference (m)")  # axis label
    ax[0].set_ylabel("clean seconds")  # axis label
    ax[0].legend(fontsize=8)  # legend
    ax[0].set_title("(a) position, clean seconds", fontsize=10)  # panel title
    sty = {"G": "#0072B2", "E": "#009E73", "C": "#CC79A7", "R": "#D55E00"}  # dict: a colour per constellation
    for v, ls in (("self", "-"), ("known", ":")):  # for loop over the variants, solid and dotted
        t = labelled(tab[v])  # with labels
        t = t[t.label == "clean"]  # clean rows
        for s in "GEC":  # for loop over three constellations
            a = np.sort(t[t.sv.str[0] == s].res.abs().dropna().to_numpy())  # sorted |residual|
            ax[1].plot(a, np.arange(1, len(a) + 1) / max(len(a), 1), ls=ls, color=sty[s], label=f"{s} {v}")  # empirical CDF
    ax[1].axvline(1.0, color="0.5", lw=0.8)  # the R2 median limit
    ax[1].axvline(4.0, color="0.5", lw=0.8, ls="--")  # the R2 95th percentile limit
    ax[1].set_xlim(0, 6)  # range
    ax[1].set_xlabel("|code residual| (m), GPS / Galileo / BeiDou, all receivers")  # axis label
    ax[1].set_ylabel("cumulative share")  # axis label
    ax[1].legend(fontsize=7, ncol=2)  # legend
    ax[1].set_title("(b) G, E, C residuals, clean seconds", fontsize=10)  # panel title
    t = labelled(tab["self"])  # self table with labels
    t = t[(t.label == "clean") & (t.sv.str[0] == "R")]  # clean GLONASS rows
    for col, ls, lab in (("res", ":", "raw (about the median clock)"), ("res_db", "-", "de-biased (minus own baseline)")):  # for loop over the two quantities
        a = np.sort(t[col].abs().dropna().to_numpy())  # sorted absolute values
        ax[2].plot(a, np.arange(1, len(a) + 1) / max(len(a), 1), ls=ls, color=sty["R"], label=f"{lab}, n {len(a)}")  # empirical CDF
    ax[2].axvline(1.5, color="0.5", lw=0.8)  # the R2 median limit for GLONASS
    ax[2].axvline(5.0, color="0.5", lw=0.8, ls="--")  # the R2 95th percentile limit for GLONASS
    ax[2].set_xlim(0, 10)  # range
    ax[2].set_xlabel("|GLONASS G1 residual| (m), self-solved, all receivers")  # axis label
    ax[2].legend(fontsize=8)  # legend
    ax[2].set_title("(c) GLONASS raw and de-biased, clean seconds", fontsize=10)  # panel title
    fig.tight_layout()  # layout
    save_fig(fig, GRAPH / "residual_engine.png")  # byte-stable save


def plot_glonass_baseline(tab: dict) -> None:  # function: the figure, the GLONASS baseline at work (one biased satellite through a session) and the R19 jump screen of 2.6.4
    t = tab["self"]  # self table
    fig = plt.figure(figsize=(13, 10))  # one figure, panels placed on a grid below
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 0.8, 1.0])  # grid: two full-width rows for (a), one row split into (b) and (c)
    a1, a2 = fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, :])  # the two full-width axes of panel (a)
    b, c = fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])  # panels (b) and (c)
    cand = t[(t.sv.str[0] == "R") & t.base.notna()].groupby(["test_code", "receiver", "sv"]).base.median().abs()  # median baseline magnitude per session, receiver and satellite
    key = cand[[k[1] == "SBF_02" for k in cand.index]].idxmax()  # the X5 satellite with the largest median baseline
    x = t[(t.test_code == key[0]) & (t.receiver == key[1]) & (t.sv == key[2])]  # its rows
    tm = pd.to_datetime(x.sec, unit="s")  # time axis
    a1.plot(tm, x.res, ".", ms=2, color="0.6", label="residual about the GLONASS median clock")  # raw
    a1.plot(tm, x.base, "-", lw=1.2, color="#0072B2", label="baseline: running median of its own accepted past residuals (cap +-10 m)")  # baseline
    a1.set_ylim(np.nanpercentile(x.res, 0.5) - 2, np.nanpercentile(x.res, 99.5) + 2)  # range around the data
    a1.set_ylabel("m")  # axis label
    a1.legend(fontsize=8, loc="lower right")  # legend
    a1.set_title(f"(a) {key[2]} on the {RXN[key[1]]}, session {key[0]}: the X5's GLONASS satellite with the largest median baseline", fontsize=10)  # panel title
    a2.plot(tm, x.res_db, ".", ms=2, color="#D55E00", label="de-biased residual (residual minus baseline); empty while the baseline warms up")  # de-biased
    a2.axhline(0.0, color="0.5", lw=0.8)  # zero line
    a2.set_ylim(-8, 8)  # a fixed range, so the scale is comparable with (a)
    a2.set_ylabel("m")  # axis label
    a2.legend(fontsize=8, loc="lower right")  # legend
    r = t[(t.test_code == "2.6.4") & (t.receiver == "UBX_01") & (t.sv == "R19")].copy()  # R19 on u-blox 01 in 2.6.4
    r["d_ms"] = (r.z - r.clk) / MS_M  # its distance from the clock of the other GLONASS satellites, in milliseconds of range
    for lab_, m_, col in (("screened (whole ms, D7a)", r.jump_k != 0, "k"), ("not whole ms: kept as a residual", (r.jump_k == 0) & (r.res.abs() > 1e5), "#D55E00"), ("normal", (r.jump_k == 0) & (r.res.abs() <= 1e5), "0.6")):  # for loop over the three kinds of row
        g = r[m_]  # those rows
        b.plot(pd.to_datetime(g.sec, unit="s"), g.d_ms, ".", ms=3, color=col, label=f"{lab_} ({len(g)} s)")  # their distance in ms
    b.axhline(-10.0, color="0.5", lw=0.8, ls="--")  # the -10 ms level
    b.set_ylabel("R19 z minus GLONASS clock (ms of range)")  # axis label
    b.legend(fontsize=7)  # legend
    b.set_title("(b) u-blox 01, 2.6.4: R19 against its constellation's clock", fontsize=10)  # panel title
    glo_res = pd.read_csv(PROC / "preprocessing" / "glonass_residuals_clean.csv")  # the GLONASS broadcast check's table (its clock still contained R19)
    an = glo_res[glo_res.anomaly == 1][["test_code", "receiver", "sec"]].drop_duplicates()  # the receiver-seconds with an R19 anomaly
    o = glo_res[glo_res.anomaly == 0].merge(an, on=["test_code", "receiver", "sec"])  # the GLONASS broadcast check's other rows at those seconds
    old = (o.z_clock_in_m - o.res_median_m).groupby([o.receiver, o.sec]).first()  # the GLONASS broadcast check's clock there
    new = t[t.sv.str[0] == "R"].merge(an, on=["test_code", "receiver", "sec"]).groupby(["receiver", "sec"]).clk.first()  # the engine's clock there
    sh = (new.reindex(old.index) - old)  # how far leaving R19 out moves the clock
    for rx in ("SBF_02", "UBX_01"):  # for loop over the two receivers with R19 anomalies
        v = sh[sh.index.get_level_values(0) == rx]  # its values
        c.plot(pd.to_datetime(v.index.get_level_values(1), unit="s"), v.to_numpy(), "o", ms=3, color=M.COLORS[rx], label=f"{RXN[rx]} ({len(v)} s)")  # the shift per second
    c.axhline(0.0, color="0.5", lw=0.8)  # zero line
    c.set_ylabel("engine clock minus the GLONASS broadcast check clock (m)")  # axis label
    c.legend(fontsize=8)  # legend
    c.set_title("(c) clean interval of 2.6.4: the GLONASS clock without R19 against the GLONASS broadcast check's with it", fontsize=10)  # panel title
    for axx, fmt in ((a1, "%H:%M"), (a2, "%H:%M"), (b, "%H:%M"), (c, "%H:%M:%S")):  # for loop over the time axes and their clock format
        axx.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=6))  # at most about six ticks, so labels do not overlap
        axx.xaxis.set_major_formatter(mdates.DateFormatter(fmt))  # UTC clock time without the day
    fig.tight_layout()  # layout
    save_fig(fig, GRAPH / "glonass_baseline.png")  # byte-stable save


def main() -> None:  # function: the residual engine from start to finish
    t_start = time.time()  # the clock
    n_p, bad = protected_ok()  # protected files at the start
    check("Protected files at start (label hash list, observables, residual minutes, the GLONASS broadcast check table, the GLONASS precise check CSVs)", "all unchanged", f"{n_p} compared, {len(bad)} changed {bad}", not bad)  # the check
    gt = pd.read_csv(PROC / "ground_truth" / "session_ground_truth.csv").set_index("test_code")  # session folders and windows
    kl = M.igs_klobuchar(S04.IGS_BRDC)  # the GPS Klobuchar parameters of the IGS header
    print("BeiDou transmission-time check")  # heading
    lag = beidou_check(gt)  # the BeiDou transmission-time check: the lag its label-chosen evidence would set (reported only)
    if lag != BDS_LAG_S:  # if: the evidence now points to another lag (possible after a relabel)
        print(f"  NOTE: the BeiDou check's evidence gives a lag of {lag} s; the engine keeps the checked constant {BDS_LAG_S} s (labels never set a feature)")  # said, not applied
    print("residual engine")  # heading
    tab, ep, r1, OBS = build(gt, kl, BDS_LAG_S)  # production and R1 tables, with the constant lag
    write_tables(tab, ep)  # written
    print("checks")  # heading
    r1_check(r1)  # R1
    r2_r3(tab, ep)  # R2 and R3
    r5_check(tab)  # R5
    gate_report(tab)  # the seeding gate on clean samples, and the clean samples beyond 10 m
    other_checks(tab, ep, OBS)  # counts and magnitudes
    print("R4 causality runs")  # heading
    r4_check(gt, kl, lag, tab, OBS)  # R4
    plot_residual_engine(tab, ep)  # the figure
    plot_glonass_baseline(tab)  # the figure
    n_p, bad = protected_ok()  # protected files at the end
    check("Protected files at end", "all unchanged", f"{n_p} compared, {len(bad)} changed {bad}", not bad)  # the check
    pd.DataFrame(CHECKS).to_csv(CHECK_DIR / "residuals_checks.csv", index=False)  # the checks
    pd.DataFrame([{"key": k, "value": v} for k, v in NUM.items()]).to_csv(CHECK_DIR / "residuals_numbers.csv", index=False)  # list comprehension: the numbers
    print(f"done in {time.time() - t_start:.0f} s")  # run time


if __name__ == "__main__":  # guard: runs main() only when the file is executed, not when imported
    main()  # entry point
    exit_if_failed(FAILED, "4_features/02_residuals.py")  # the helpers module: status 1 when any self-check failed, so a chained run stops
