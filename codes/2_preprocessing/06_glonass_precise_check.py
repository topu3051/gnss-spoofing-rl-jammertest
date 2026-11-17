#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
the GLONASS precise check - is the GLONASS V3 failure of the GLONASS broadcast check a broadcast-ephemeris error or a model error?. A diagnostic only:
it changes neither codes/2_preprocessing/04_glonass_orbit.py nor any output of the GLONASS broadcast check, and it builds no feature, label or model.

Products (Data/external/igs_precise, read only): the ESA final orbit (5 min) and clock (30 s) and the GFZ final orbit (5 min
despite the 15M in its name) and clock (30 s) of 14 September 2026. Both carry the same 20 GLONASS satellites; R01, R10 and R13, healthy
in the broadcast file, are in neither. Each product is used alone in every epoch (each has its own clock datum); ESA is primary.

Conventions
  SP3 positions are the satellite centre of mass in IGS20, GPS time; clocks are GPS time, ionosphere-free P-code, periodic relativistic
  term removed. No antenna offset is applied (none matching the products is available; its size is bounded instead).
  rel = +2 r.v / c in metres (the GLONASS orbit script's rel_m). The full apparent precise clock is P = c dts_p - rel (the IGS convention). The broadcast
  variants are B0 = c dts_b (the GLONASS orbit script as it is), B- = B0 - rel (the GPS sign), B+ = B0 + rel (the opposite sign); D_V = B_V - P.
  With y = c (dts_b - dts_p): D0 = y + rel, D- = y, D+ = y + 2 rel. The two-way fixed-effects slope beta of y on rel is -1 if B0 is
  right, 0 if B- is right, -2 if B+ is right. At the ranging level w = (radial broadcast minus precise) - y and eps_V = radial - D_V;
  the slope gamma of w on rel is +1, 0 or +2 for B0, B-, B+.
  A pseudorange residual (observed minus modelled) sees - (radial - D0); the predicted residual offset is P = -(E - epoch median of E),
  E = radial - D0.

Pre-registered readings (set before running)
  T1  the constellation-common broadcast-minus-precise radial median should be near -2 m (antenna phase centre about 2-2.5 m towards
      the Earth from the centre of mass); "near" means inside -3.0 to -1.5 m. Reported, nothing adjusted.
  T2  a variant wins if its pooled 95th percentile |D| (epoch median removed) is at most 0.9 times the runner-up's and the slope is within
      0.3 of the variant's value; otherwise undecided. The same rule at the ranging level (eps_V, gamma) governs the advice for the GLONASS orbit script.
  T3  the GLONASS broadcast check offset regressed on P: correlation >= 0.8 and slope 0.7-1.3 means the V3 offsets are broadcast-ephemeris error.
  T4  V3 run again with the precise orbit and clock: |residual| median < 1.5 m and 95th percentile < 5 m (unchanged thresholds).
  T5  two runs of this script give byte-identical outputs (compared from outside).

Reads   the two products, the IGS broadcast file, Data/rinex/*, Data/processed/{glonass_residuals_clean, reference_clean_intervals,
        reference_positions, session_ground_truth, glonass_channels}.csv
Writes  Data/processed/checks/glonass_precise_check.csv   one row per check: check, expected, found, ok
        Data/processed/checks/glonass_precise_per_satellite.csv   long table: table, product, sv, test_code, receiver, metric, value
        Data/processed/checks/glonass_precise_summary.csv         every number the report quotes (key, value)
        Graph/glonass_broadcast_vs_precise.png  four panels: T1 per satellite, T2 slopes, T3 scatter, T4 distributions
"""

from __future__ import annotations  # future import: type hints kept as text (a convention across the project's scripts)

import gzip  # standard-library gzip reader: opens the compressed products as text
import hashlib  # standard-library hashing, for the SHA-256 comparisons
import importlib.util  # standard-library module loader, for modules whose names start with a digit
import os  # operating-system access: MPLCONFIGDIR is set before matplotlib loads
import random  # standard-library random; seeded below
import sys  # standard-library system access: sys.path
from datetime import datetime, timezone  # date-time classes, to read the product epochs
from pathlib import Path  # Path objects for building file paths

PROJECT = Path(__file__).resolve().parent.parent.parent  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
os.environ["MPLCONFIGDIR"] = str(PROJECT / "codes" / ".mplconfig")  # matplotlib's cache inside the project folder

import numpy as np  # NumPy: arrays, interpolation, least squares
import pandas as pd  # pandas: tables
import matplotlib  # plotting library
matplotlib.use("Agg")  # file-only backend, no window
import matplotlib.pyplot as plt  # the pyplot interface

sys.path.insert(0, str(PROJECT / "codes" / "common"))  # puts codes/ first on the import path so the helpers module is found
from helpers import GPS_EPOCH, save_fig  # the GPS epoch and the byte-stable figure saver

SEED = 20260914  # integer constant: the project's fixed random seed
random.seed(SEED)  # seeds Python's generator (nothing here draws from it)
np.random.seed(SEED)  # seeds NumPy's legacy generator (nothing here draws from it)
RNG = np.random.default_rng(SEED)  # NumPy generator object: the only random draws here, the bootstrap over satellites in T2

CHECK_DIR = PROJECT / "Data" / "processed" / "checks"  # where the check tables go
GRAPH = PROJECT / "Graph"  # where the figure goes
PROC = PROJECT / "Data" / "processed"  # where the GLONASS broadcast check residual table is read
EXT = PROJECT / "Data" / "external" / "igs_precise"  # the downloaded products (read only)


def load(name: str, path: Path):  # function: loads a Python file as a module (names start with a digit or sit in Data/processed/checks)
    spec = importlib.util.spec_from_file_location(name, path)  # where to load from
    mod = importlib.util.module_from_spec(spec)  # an empty module object
    sys.modules[name] = mod  # registered under name, as a normal import would
    spec.loader.exec_module(mod)  # runs its top level (definitions only; its main() is guarded)
    return mod  # the module


A = load("glonass_orbit_check_for_precise_check", PROJECT / "codes" / "2_preprocessing" / "05_glonass_orbit_check.py")  # the GLONASS broadcast check's check script: its v3_run is the V3 computation reused unchanged in T4
G, D, M, E = A.G, A.D, A.M, A.E  # the GLONASS orbit script (broadcast propagator), the evidence script (residual definition), the position script (geometry), the observables script (observation reader)
C = G.C_LIGHT  # speed of light, m/s
OM = np.array([0.0, 0.0, G.OMEGA])  # Earth rotation vector, rad/s, for the inertial velocity of the orbit frame
# dict literal: product name -> (orbit file, clock file); ESA first because it is the primary product
PRODUCTS = {"ESA": (EXT / "ESA0OPSFIN_20262570000_01D_05M_ORB.SP3.gz", EXT / "ESA0OPSFIN_20262570000_01D_30S_CLK.CLK.gz"),
            "GFZ": (EXT / "GFZ0OPSFIN_20262570000_01D_15M_ORB.SP3.gz", EXT / "GFZ0OPSFIN_20262570000_01D_30S_CLK.CLK.gz")}  # GFZ second
# dict literal: the SHA-256 of each product as logged at download (checks_log 28, 29)
SHA_LOGGED = {"ESA0OPSFIN_20262570000_01D_05M_ORB.SP3.gz": "1316a8a2c06367e2376227cadb0026b8948ac573ff9bf8ed8ede66cfa3bf3d4d",  # ESA orbit
              "ESA0OPSFIN_20262570000_01D_30S_CLK.CLK.gz": "36ed556b16a759d4909a0555eba1abe59e16300882def9929f7a24fd3484d041",  # ESA clock
              "GFZ0OPSFIN_20262570000_01D_15M_ORB.SP3.gz": "dfaf90a08ce19e15a8623f947a290c667d8d4f718797c2dbcb4d22da2e92b353",  # GFZ orbit
              "GFZ0OPSFIN_20262570000_01D_30S_CLK.CLK.gz": "d831c75d1a1fd5b4bd7ceae15ac288cd94197928bbfc55ac23a6bd7d40dd088e"}  # GFZ clock
# dict literal: the protected files outside the label hash list and their SHA-256 as recorded when they were written
PROTECTED_EXTRA = {"Data/processed/features/observables_1hz.csv.gz": "f6b86c8dc107ba89193b5432a458752af625a7607d8e6eb28bcba754ee0ea989",  # the observables script table
                   "Data/processed/preprocessing/rinex_residual_minutes.csv": "e4e9acd4c4d25676bfd39075d700736d088f1e48c62d7b0696678aadbae4b094",  # the conversion script's minute table
                   "Data/processed/preprocessing/glonass_residuals_clean.csv": "606c7e38cb0ff21b58a10b3575a0225018f20114571cba6bb9235d7d64cd75b2"}  # the GLONASS broadcast check
PCO_WINDOW = (-3.0, -1.5)  # tuple: the pre-registered "near -2 m" window for the common radial median, metres
T3_CORR, T3_SLOPE = 0.8, (0.7, 1.3)  # pre-registered T3 reading: minimum correlation and the slope range
V3_MED, V3_P95 = 1.5, 5.0  # the unchanged V3 thresholds in metres
WIN_RATIO, COEF_TOL = 0.9, 0.3  # T2 winner rule: at most 0.9 of the runner-up's 95th percentile and slope within 0.3 of the target
BETA_T = {"none": -1.0, "gps": 0.0, "opp": -2.0}  # dict: clock-level slope beta expected if each variant is right
GAMMA_T = {"none": 1.0, "gps": 0.0, "opp": 2.0}  # dict: ranging-level slope gamma expected if each variant is right
VNAME = {"none": "none (the GLONASS orbit script as it is)", "gps": "GPS sign (-rel)", "opp": "opposite sign (+rel)"}  # dict: variant labels
RXN = {"SBF_02": "mosaic-X5", "SBF_03": "mosaic-H main", "UBX_01": "u-blox 01"}  # receiver names for the figure and the checks
PCOL = {"broadcast": "#0072B2", "ESA": "#D55E00", "GFZ": "#CC79A7"}  # Okabe-Ito colours of the orbit sources (validated, CVD-safe)
CHK: list = []  # list: the checks, written to glonass_precise_check.csv
NUM: dict = {}  # dict: every quoted number, written to glonass_precise_summary.csv
LONG: list = []  # list: rows of the long per-satellite table


def add(check: str, expected: str, found, ok: bool | None) -> None:  # function: records one check and prints it
    CHK.append({"check": check, "expected": expected, "found": found, "ok": "" if ok is None else bool(ok)})  # append a dict row
    print(f"  [{'ok' if ok else ('FAIL' if ok is False else '--')}] {check}: expected {expected}; found {found}")  # progress line


def pct(a, q) -> float:  # function: percentile of an array, NaN if empty
    a = np.asarray(a, dtype=float)  # as a float array
    return float(np.percentile(a, q)) if len(a) else float("nan")  # NumPy percentile


def long(table: str, product: str, sv: str, metric: str, value, test: str = "", rx: str = "") -> None:  # function: one row of the long table
    LONG.append((table, product, sv, test, rx, metric, float(value)))  # tuple appended in a fixed column order


def sha(path: Path) -> str:  # function: SHA-256 of a file
    return hashlib.sha256(path.read_bytes()).hexdigest()  # hex digest of the whole file


def protected(when: str) -> None:  # function: the protected-file check, run at the start and at the end
    bad = []  # files whose hash differs
    for line in (CHECK_DIR / "expected_hashes_labels.txt").read_text().splitlines():  # for loop over the label hash list (hash, two spaces, path)
        h, p = line.split()  # the recorded hash and the path
        if sha(PROJECT / p) != h:  # if: the file changed
            bad.append(p)  # remembered
    for p, h in PROTECTED_EXTRA.items():  # for loop over the three extra protected files
        if sha(PROJECT / p) != h:  # if: changed
            bad.append(p)  # remembered
    add(f"protected SHA-256 ({when}): label hash list (5) + observables_1hz + rinex_residual_minutes + the GLONASS broadcast check residuals", "all 8 unchanged", "unchanged" if not bad else f"CHANGED {bad}", not bad)  # the check


def gps_seconds(parts: list) -> float:  # function: GPS seconds of a calendar epoch written in GPS time (year, month, day, hour, minute, second)
    y, mo, d, h, mi = (int(v) for v in parts[:5])  # generator expression: the five integer fields
    return (datetime(y, mo, d, h, mi, tzinfo=timezone.utc) - GPS_EPOCH).total_seconds() + float(parts[5])  # no leap seconds: the scale is already GPS


def read_sp3(path: Path) -> tuple:  # function: the GLONASS part of an SP3-c file: epochs, positions (m), clocks (s) and the time system
    lines = gzip.open(path, "rt").read().split("\n")  # all lines of the decompressed file
    tsys = next(l[9:12] for l in lines if l.startswith("%c"))  # the time-system field of the first %c line ("GPS" expected)
    t, rows = [], []  # epoch times; (epoch index, satellite, x, y, z, clock) rows
    for l in lines:  # for loop over the lines
        if l.startswith("* "):  # if: an epoch line
            t.append(gps_seconds(l[1:].split()))  # its GPS seconds
        elif l.startswith("PR"):  # elif: a GLONASS position line
            rows.append((len(t) - 1, l[1:4], float(l[4:18]), float(l[18:32]), float(l[32:46]), float(l[46:60])))  # km and microseconds
    T = np.array(t)  # epoch times as an array
    pos, clk = {}, {}  # satellite -> (N x 3) metres and satellite -> N seconds
    for sv in sorted({r[1] for r in rows}):  # for loop over the satellites, in name order (set comprehension gives the names)
        pos[sv] = np.full((len(T), 3), np.nan)  # NaN where the satellite has no line
        clk[sv] = np.full(len(T), np.nan)  # likewise for the clock
    for i, sv, x, y, z, c in rows:  # for loop over the position lines
        if not (x == 0.0 and y == 0.0 and z == 0.0):  # if: not the SP3 "bad position" marker (all zeros)
            pos[sv][i] = (x * 1e3, y * 1e3, z * 1e3)  # kilometres to metres
        if c < 999999.0:  # if: not the SP3 "bad clock" marker
            clk[sv][i] = c * 1e-6  # microseconds to seconds
    return T, pos, clk, tsys  # epochs, positions, SP3 clocks, time system


def read_clk(path: Path) -> tuple:  # function: the GLONASS satellite clocks (AS records) of a RINEX clock file, GPS time, seconds
    out = {}  # satellite -> list of (time, clock)
    for l in gzip.open(path, "rt"):  # for loop over the lines
        if l.startswith("AS R"):  # if: a GLONASS satellite clock record
            p = l.split()  # whitespace-separated fields: AS, sat, y, m, d, h, mi, s, n, value(s)
            out.setdefault(p[1], []).append((gps_seconds(p[2:8]), float(p[9])))  # time and the first value (the bias)
    ct = {sv: np.array([a for a, _ in v]) for sv, v in sorted(out.items())}  # dict comprehension: satellite -> times
    cv = {sv: np.array([b for _, b in v]) for sv, v in sorted(out.items())}  # satellite -> clock values
    return ct, cv  # the two dicts


class Precise:  # class: one precise product as an orbit source with the GLONASS orbit script's interface (state_full, channel)
    def __init__(self, name: str, sp3: Path, clk: Path):  # constructor: reads the product once
        self.name = name  # "ESA" or "GFZ"
        self.T, self.P, self.Psc, self.tsys = read_sp3(sp3)  # epochs, positions, SP3 clocks, time system
        self.ct, self.cv = read_clk(clk)  # the 30 s clocks
        self.H = float(np.median(np.diff(self.T)))  # node spacing in seconds (the median, not the first interval)
        self.svs = sorted(set(self.P) & set(self.ct))  # satellites with both an orbit and a clock
        self.brdc = G.GloOrbits()  # a broadcast object of its own: the record-availability rule and the channel numbers come from the GLONASS orbit script

    def pos(self, sv: str, t, pad: float = 0.0) -> np.ndarray:  # method: Earth-fixed position (n x 3, m) by 11-point Lagrange interpolation; pad: seconds allowed beyond the file
        t = np.atleast_1d(np.asarray(t, dtype=float))  # the times as a 1-D array
        P = self.P[sv]  # this satellite's node positions
        n = len(self.T)  # number of nodes
        u = (t - self.T[0]) / self.H  # time in node units
        i0 = np.clip(np.floor(u).astype(int) - 5, 0, n - 11)  # first node of the centred 11-node window, clipped at the ends
        x = u - i0  # time inside the window, node units
        w = np.ones((len(t), 11))  # Lagrange weights, one row per time
        for j in range(11):  # for loop over the nodes of the window
            for m in range(11):  # for loop over the other nodes
                if m != j:  # if: not the node itself
                    w[:, j] *= (x - m) / (j - m)  # the Lagrange basis product
        idx = i0[:, None] + np.arange(11)[None, :]  # broadcasting: node indices of each time's window (n x 11)
        out = np.einsum("nj,njk->nk", w, P[idx])  # einsum: weighted sum of the window's positions
        out[(t < self.T[0] - pad) | (t > self.T[-1] + pad)] = np.nan  # no extrapolation beyond the file (pad only for the velocity's half-second)
        return out  # positions

    def vel(self, sv: str, t) -> np.ndarray:  # method: Earth-fixed velocity (m/s) by a 1 s central difference of the interpolant
        t = np.atleast_1d(np.asarray(t, dtype=float))  # the times as an array
        v = (self.pos(sv, t + 0.5, pad=1.0) - self.pos(sv, t - 0.5, pad=1.0)) / 1.0  # central difference; at the file's two end epochs the end window's polynomial is used 0.5 s beyond (a one-sided difference there would be off by about 0.13 m/s)
        v[(t < self.T[0]) | (t > self.T[-1])] = np.nan  # undefined outside the file itself
        return v  # velocities

    def clock(self, sv: str, t) -> np.ndarray:  # method: satellite clock (s) by linear interpolation of the 30 s values, NaN across gaps
        t = np.atleast_1d(np.asarray(t, dtype=float))  # the times as an array
        ct, cv = self.ct[sv], self.cv[sv]  # node times and values
        lo = np.clip(np.searchsorted(ct, t, side="right") - 1, 0, len(ct) - 2)  # left node of each time's interval
        hi = lo + 1  # right node
        f = (t - ct[lo]) / (ct[hi] - ct[lo])  # fraction of the interval
        out = cv[lo] + f * (cv[hi] - cv[lo])  # linear interpolation
        out[(t < ct[0]) | (t > ct[-1]) | (ct[hi] - ct[lo] > 30.5)] = np.nan  # outside the file or across a missing record
        return out  # clocks

    def channel(self, slot: int) -> int:  # method: frequency channel of a slot (the GLONASS orbit script's table first, IGS record second)
        return self.brdc.channel(slot)  # delegated to the GLONASS orbit script

    def state_full(self, svs: list, t_rx: float, pr: np.ndarray) -> tuple:  # method: the GLONASS orbit script's state_full with the precise orbit and clock
        n = len(svs)  # number of satellites
        pos, vel, dts, ok = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros(n), np.zeros(n, bool)  # results
        for j, sv in enumerate(svs):  # for loop over the satellites
            if sv[0] != "R" or sv not in self.svs:  # if: not GLONASS or not in the product
                continue  # not handled
            if self.brdc.pick(sv, t_rx) is None:  # if: the GLONASS orbit script has no usable broadcast record here (same rows as the broadcast run)
                continue  # skip, as the GLONASS orbit script would
            tt = np.array([t_rx - float(pr[j]) / C])  # transmission time on the satellite clock, first guess (as the GLONASS orbit script)
            tt2 = tt - self.clock(sv, tt)  # corrected by the satellite clock (the relativistic ns change nothing here)
            p, v = self.pos(sv, tt2), self.vel(sv, tt2)  # position and velocity at the true transmission time
            rel = 2.0 * np.einsum("ij,ij->i", p, v) / C  # periodic relativistic term 2 r.v / c, metres
            d2 = self.clock(sv, tt2) - rel / C  # full apparent clock: precise clock plus the periodic term with the IGS sign
            pos[j], vel[j], dts[j] = p[0], v[0], d2[0]  # stored
            ok[j] = bool(np.isfinite(p).all() and np.isfinite(d2[0]))  # usable if everything is defined
        return pos, vel, dts, ok  # the arrays


class BrdcKeep(G.GloOrbits):  # class: the GLONASS orbit script's broadcast orbit, optionally limited to a set of satellites (for identical rows)
    def __init__(self, keep: set | None = None):  # constructor: keep = satellites allowed, None for all
        super().__init__()  # the GLONASS orbit script's constructor unchanged
        self.keep = keep  # the allowed satellites

    def state_full(self, svs: list, t_rx: float, pr: np.ndarray) -> tuple:  # method: the GLONASS orbit script's state_full, then the excluded satellites masked
        pos, vel, dts, ok = super().state_full(svs, t_rx, pr)  # the GLONASS orbit script's computation unchanged
        if self.keep is not None:  # if: limited
            ok = ok & np.array([sv in self.keep for sv in svs], dtype=bool)  # list comprehension: True only for allowed satellites
        return pos, vel, dts, ok  # the arrays


def rec_arrays(orb) -> dict:  # function: each satellite's healthy records as a NumPy array (columns below)
    cols = ["x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az", "tb_gps", "clk", "gam"]  # state, acceleration, tb, -TauN, GammaN
    return {sv: g[cols].to_numpy(dtype=float) for sv, g in orb.rec.items()}  # dict comprehension over the GLONASS orbit script's record tables


def brdc_eval(orb, ra: dict, svs, t_pick, t_eval) -> tuple:  # function: broadcast position, velocity and clock at t_eval with the GLONASS orbit script's record chosen at t_pick
    n = len(svs)  # number of rows
    R = np.full((n, 12), np.nan)  # the chosen record of each row
    ok = np.zeros(n, bool)  # rows with a usable record
    for i, (sv, tp) in enumerate(zip(svs, t_pick)):  # for loop over the rows
        k = orb.pick(sv, float(tp))  # the GLONASS orbit script's rule: nearest healthy record within 15 min, earlier at a tie
        if k is not None:  # if: a record exists
            R[i], ok[i] = ra[sv][k], True  # stored
    pos, vel, dts = np.full((n, 3), np.nan), np.full((n, 3), np.nan), np.full(n, np.nan)  # results
    if ok.any():  # if: anything usable
        te = np.asarray(t_eval, dtype=float)[ok]  # evaluation times of the usable rows
        s = G.propagate(R[ok, :6], R[ok, 6:9], te - R[ok, 9])  # the GLONASS orbit script's RK4 propagation from tb to t_eval
        pos[ok], vel[ok] = s[:, :3], s[:, 3:]  # positions and velocities
        dts[ok] = R[ok, 10] + R[ok, 11] * (te - R[ok, 9])  # the GLONASS orbit script's clock: -TauN + GammaN (t - tb)
    return pos, vel, dts, ok  # the arrays


def prec_eval(prec: Precise, svs, t) -> tuple:  # function: precise position, velocity and clock (no relativistic term) per row
    svs, t = np.asarray(svs), np.asarray(t, dtype=float)  # arrays
    pos, vel, clk = np.full((len(t), 3), np.nan), np.full((len(t), 3), np.nan), np.full(len(t), np.nan)  # results
    for sv in sorted(set(svs)):  # for loop over the satellites present
        m = svs == sv  # rows of this satellite
        if sv in prec.svs:  # if: the product carries it
            pos[m], vel[m], clk[m] = prec.pos(sv, t[m]), prec.vel(sv, t[m]), prec.clock(sv, t[m])  # interpolated values
    return pos, vel, clk  # the arrays


def rac(dp: np.ndarray, p: np.ndarray, v: np.ndarray) -> tuple:  # function: a difference vector in radial, along-track and cross-track
    r = p / np.linalg.norm(p, axis=1)[:, None]  # radial unit vector
    vi = v + np.cross(OM, p)  # inertial velocity: Earth-fixed velocity plus omega x r
    c = np.cross(p, vi)  # orbit normal
    c = c / np.linalg.norm(c, axis=1)[:, None]  # cross-track unit vector
    a = np.cross(c, r)  # along-track unit vector completing the triad
    return np.einsum("ij,ij->i", dp, r), np.einsum("ij,ij->i", dp, a), np.einsum("ij,ij->i", dp, c)  # the three projections


def fe_slope(df: pd.DataFrame, y: str, x: str = "rel") -> tuple:  # function: two-way fixed-effects slope of y on x (one effect per epoch and per satellite)
    te = pd.factorize(df.t, sort=True)[0]  # epoch index of each row
    se = pd.factorize(df.sv, sort=True)[0]  # satellite index of each row
    n_t, n_s = te.max() + 1, se.max() + 1  # numbers of epochs and satellites
    X = np.zeros((len(df), n_t + n_s))  # design: epoch dummies, satellite dummies (the first dropped), slope column
    X[np.arange(len(df)), te] = 1.0  # epoch dummies
    X[np.arange(len(df)), n_t + se] = 1.0  # satellite dummies
    X = np.delete(X, n_t, axis=1)  # drops the first satellite's dummy (otherwise collinear with the epoch dummies)
    X = np.column_stack([X, df[x].to_numpy()])  # the slope column last
    b = np.linalg.lstsq(X, df[y].to_numpy(), rcond=None)[0]  # least-squares coefficients
    ep = pd.Series(b[:n_t], index=np.sort(df.t.unique()))  # the epoch effects, indexed by epoch time
    return float(b[-1]), ep  # the slope and the epoch effects


def obs_rows() -> pd.DataFrame:  # function: the GLONASS observations of the clean intervals (time, pseudorange) for every session and receiver
    gt = pd.read_csv(PROC / "ground_truth" / "session_ground_truth.csv").set_index("test_code")  # session folders
    out = []  # tables
    for test in D.SESSIONS:  # for loop over the sessions in time order
        for rx in D.RXS:  # for loop over the receivers
            spans = D.clean_spans(test, rx)  # the clean intervals
            if not spans:  # if: none
                continue  # nothing to read
            ob = E.read_all(M.obs_path(gt.loc[test, "session_folder"], test, rx), spans, rx)  # all four constellations, as the GLONASS broadcast check read them
            D.LINES.clear()  # frees the decompressed file
            ob = ob[ob.sv.str[0] == "R"][["sec", "sv", "t_gps", "pr"]].copy()  # GLONASS rows only
            ob.insert(0, "receiver", rx)  # receiver column
            ob.insert(0, "test_code", test)  # session column
            out.append(ob)  # kept
    return pd.concat(out, ignore_index=True)  # one table


def row_states(orb, ra: dict, prec: Precise, df: pd.DataFrame) -> pd.DataFrame:  # function: broadcast and precise state at each residual row's transmission time (the GLONASS orbit script's arithmetic)
    sv, t, pr = df.sv.to_numpy(), df.t_gps.to_numpy(), df.pr.to_numpy()  # satellite, receive time, pseudorange
    tt = t - pr / C  # transmission time, first guess
    _, _, d1, _ = brdc_eval(orb, ra, sv, t, tt)  # broadcast clock at the first guess
    tt2 = tt - d1  # true transmission time, as the GLONASS orbit script
    pb, vb, db, okb = brdc_eval(orb, ra, sv, t, tt2)  # broadcast state at that time
    pp, vp, cp = prec_eval(prec, sv, tt2)  # precise state at the same instant
    out = df.copy()  # the rows with new columns
    out["tt2"] = tt2  # the transmission time
    for k, a in (("b", pb), ("p", pp)):  # for loop over the two positions
        out[[f"x{k}", f"y{k}", f"z{k}"]] = a  # position columns
    out["dts_b"], out["dts_p"] = db, cp  # broadcast clock and precise clock (no relativistic term)
    out["rel"] = 2.0 * np.einsum("ij,ij->i", pp, vp) / C  # periodic term from the precise orbit, metres
    out["dr"], out["da"], out["dc"] = rac(pb - pp, pp, vp)  # broadcast minus precise, radial / along / cross
    out["d0"] = C * db - (C * cp - out.rel.to_numpy())  # D0: the GLONASS orbit script's clock minus the full apparent precise clock, metres
    out["ok"] = okb & np.isfinite(pp).all(axis=1) & np.isfinite(vp).all(axis=1) & np.isfinite(cp)  # rows where both sources are defined
    return out  # the table


def grid(orb, ra: dict, prec: Precise, svs: list) -> pd.DataFrame:  # function: broadcast against precise at every product epoch of the day (T1, T2)
    rows = []  # tables per satellite
    for sv in svs:  # for loop over the satellites
        t = prec.T  # the product's epochs
        pb, vb, db, okb = brdc_eval(orb, ra, [sv] * len(t), t, t)  # broadcast at each epoch, record chosen at that epoch
        pp, vp, cp = prec.pos(sv, t), prec.vel(sv, t), prec.clock(sv, t)  # precise at the same epochs
        g = pd.DataFrame({"t": t, "sv": sv})  # the rows
        g["dr"], g["da"], g["dc"] = rac(pb - pp, pp, vp)  # broadcast minus precise in the orbit frame
        g["d3"] = np.linalg.norm(pb - pp, axis=1)  # 3-D difference
        g["y"] = C * (db - cp)  # y = c (dts_b - dts_p), metres
        g["rel"] = 2.0 * np.einsum("ij,ij->i", pp, vp) / C  # rel from the precise orbit
        g["rel_b"] = 2.0 * np.einsum("ij,ij->i", pb, vb) / C  # rel from the broadcast orbit (check only)
        g["radius_km"] = np.linalg.norm(pp, axis=1) / 1e3  # precise orbit radius (check only)
        g["ok"] = okb & np.isfinite(pp).all(axis=1) & np.isfinite(vp).all(axis=1) & np.isfinite(cp)  # position, velocity and clock all defined
        rows.append(g)  # kept
    return pd.concat(rows, ignore_index=True)  # one table


def demed(df: pd.DataFrame, col: str, by: str | list = "t") -> pd.Series:  # function: a column minus its median over satellites within each epoch (group)
    return df[col] - df.groupby(by)[col].transform("median")  # one datum per epoch removed


def winner(stats: dict, coef: float, target: dict) -> str:  # function: the pre-registered winner rule of T2 (95th percentile and slope)
    order = sorted(stats, key=lambda k: stats[k])  # variants by their pooled 95th percentile, best first
    best, run = order[0], order[1]  # the best and the runner-up
    if stats[best] <= WIN_RATIO * stats[run] and abs(coef - target[best]) <= COEF_TOL:  # if: both parts of the rule hold
        return best  # decided
    return "undecided"  # otherwise


def resid_stats(res: pd.DataFrame) -> dict:  # function: V3 statistics of a residual table (anomalies removed and all rows)
    ok = res[res.anomaly == 0]  # rows without the milliseconds anomalies
    out = {"n": len(ok), "med": pct(np.abs(ok.res_median_m), 50), "p95": pct(np.abs(ok.res_median_m), 95)}  # pooled
    out["all_med"], out["all_p95"], out["all_n"] = pct(np.abs(res.res_median_m), 50), pct(np.abs(res.res_median_m), 95), len(res)  # all rows
    for rx in D.RXS:  # for loop over the receivers
        q = np.abs(ok[ok.receiver == rx].res_median_m)  # that receiver's |residuals|
        out[f"med_{rx}"], out[f"p95_{rx}"], out[f"n_{rx}"] = pct(q, 50), pct(q, 95), len(q)  # per receiver
    return out  # the dict


def main() -> None:  # function: runs T1-T5 for both products and writes the outputs
    print("the GLONASS precise check - GLONASS broadcast against precise orbit and clock")  # heading
    protected("start")  # protected files before anything runs
    add("IGS broadcast file SHA-256 equals the hash logged at download", D.S04.IGS_SHA256[:12], sha(G.IGS_GZ)[:12], sha(G.IGS_GZ) == D.S04.IGS_SHA256)  # same broadcast file as the GLONASS broadcast check
    for name, files in PRODUCTS.items():  # for loop over the products
        for f in files:  # for loop over the orbit and the clock file
            add(f"{name} product SHA-256 equals the logged one ({f.name})", SHA_LOGGED[f.name][:12], sha(f)[:12], sha(f) == SHA_LOGGED[f.name])  # file identity
    kl = M.igs_klobuchar(D.S04.IGS_BRDC)  # the GPS Klobuchar parameters, as the GLONASS broadcast check
    pub = pd.read_csv(PROC / "preprocessing" / "glonass_residuals_clean.csv")  # the published the GLONASS broadcast check residual table
    # self-check: the GLONASS broadcast check's V3 computation run through the subclass with no limit reproduces the published table byte for byte
    rep, _ = A.v3_run(BrdcKeep(None), kl)  # the V3 computation with the broadcast orbit
    same = A.csv_bytes(rep) == (PROC / "preprocessing" / "glonass_residuals_clean.csv").read_bytes()  # byte comparison with the file on disk
    add("reproduction: the GLONASS broadcast check's v3_run through this script's path reproduces preprocessing/glonass_residuals_clean.csv", "byte-identical", f"{len(rep)} rows, {'identical' if same else 'DIFFERENT'}", same)  # the check
    if not same:  # if: the path differs from the GLONASS broadcast check
        raise SystemExit("STOP: the broadcast path does not reproduce the GLONASS broadcast check; nothing else is meaningful")  # stop
    obs = obs_rows()  # the GLONASS observations of the clean intervals
    dup = int(obs.duplicated(["test_code", "receiver", "sec", "sv"]).sum())  # duplicated satellite-seconds
    obs = obs.drop_duplicates(["test_code", "receiver", "sec", "sv"])  # one row per satellite-second
    add("observation rows re-read for T3 (GLONASS, clean intervals), duplicates dropped", "reported", f"{len(obs)} rows, {dup} duplicates", None)  # the count
    precs = {name: Precise(name, *files) for name, files in PRODUCTS.items()}  # dict comprehension: the two products
    ra = rec_arrays(G.GloOrbits())  # the healthy broadcast records as arrays
    healthy = sorted(ra)  # the healthy broadcast satellites
    figdat = {}  # data handed to the figure
    for name, prec in precs.items():  # for loop over the products
        print(f"--- {name}")  # progress
        # product content
        bad_p = int(sum(np.isnan(prec.P[s]).any(axis=1).sum() for s in prec.P))  # epochs with a bad position marker
        add(f"{name}: time system / epochs / spacing / GLONASS satellites (orbit and clock)", "GPS / 289 / 300 s", f"{prec.tsys} / {len(prec.T)} / {prec.H:.0f} s / {len(prec.svs)}: {' '.join(prec.svs)}; bad position epochs {bad_p}", prec.tsys == "GPS" and len(prec.T) == 289 and prec.H == 300.0)  # header facts
        miss = [s for s in healthy if s not in prec.svs]  # healthy broadcast satellites the product lacks
        rows_miss = int(pub.sv.isin(miss).sum())  # the GLONASS broadcast check V3 rows on them
        add(f"{name}: healthy broadcast satellites not in the product; the GLONASS broadcast check V3 rows on them", "R01, R10 (and R13) expected missing", f"{miss}; {rows_miss} of {len(pub)} rows ({', '.join(f'{s} {int((pub.sv == s).sum())}' for s in miss if (pub.sv == s).any())})", None)  # coverage
        NUM[f"{name}_n_glo"], NUM[f"{name}_missing"], NUM[f"{name}_rows_missing"] = len(prec.svs), " ".join(miss), rows_miss  # kept
        dsc = []  # SP3 clock minus clock-file clock at the 5 min nodes, seconds
        for s in prec.svs:  # for loop over the satellites
            dsc.append(prec.Psc[s] - prec.clock(s, prec.T))  # SP3 clock column minus the clock file at the same epochs
        dsc = np.concatenate(dsc)  # one array
        dsc = dsc[np.isfinite(dsc)]  # defined values only
        add(f"{name}: SP3 clock column against the clock file at the 5 min nodes (parsing and time alignment)", "|difference| < 1 ns", f"max {np.abs(dsc).max() * 1e9:.4f} ns, median {np.median(np.abs(dsc)) * 1e9:.4f} ns over {len(dsc)}", bool(np.abs(dsc).max() < 1e-9))  # the check
        # interpolation leave-one-out: orbit (node k from 5 nodes each side, k left out) and clock (node from the +-30 s neighbours)
        lo_o, lo_c = [], []  # orbit and clock leave-one-out errors (m)
        for s in prec.svs:  # for loop over the satellites
            P = prec.P[s]  # node positions
            for k in range(6, len(prec.T) - 6):  # for loop over the interior nodes
                idx = [i for i in range(k - 5, k + 6) if i != k]  # list comprehension: the ten neighbour nodes
                x = np.array(idx, dtype=float) - k  # their offsets in node units (the target is at 0)
                w = np.array([np.prod([(0.0 - x[m]) / (x[j] - x[m]) for m in range(10) if m != j]) for j in range(10)])  # Lagrange weights at the target
                lo_o.append(np.linalg.norm(w @ P[idx] - P[k]))  # error of the interpolated position
            c = prec.cv[s]  # the 30 s clock values
            lo_c.append(np.abs((c[:-2] + c[2:]) / 2.0 - c[1:-1]) * C)  # middle value against the mean of its two neighbours, metres
        lo_o, lo_c = np.array(lo_o), np.concatenate(lo_c)  # arrays
        add(f"{name}: orbit interpolation leave-one-out (10 nodes, 600 s gap around the target; a bound for the 11-node 300 s use)", "centimetres", f"median {np.median(lo_o) * 1e3:.2f} mm, 95th {pct(lo_o, 95) * 1e3:.2f} mm, max {lo_o.max() * 1e3:.2f} mm", bool(lo_o.max() < 0.10))  # the orbit bound
        add(f"{name}: clock interpolation leave-one-out (linear over 60 s; a bound for the 30 s use)", "centimetres", f"median {np.median(lo_c) * 100:.2f} cm, 95th {pct(lo_c, 95) * 100:.2f} cm, max {lo_c.max() * 100:.2f} cm", bool(pct(lo_c, 95) < 0.10))  # the clock bound
        NUM[f"{name}_loo_orb_max_mm"], NUM[f"{name}_loo_clk_p95_cm"], NUM[f"{name}_loo_clk_max_cm"] = lo_o.max() * 1e3, pct(lo_c, 95) * 100, lo_c.max() * 100  # kept
        # ---------------- T1 and T2 over the day ----------------
        svs = [s for s in healthy if s in prec.svs]  # satellites healthy in the broadcast and in the product
        gd = grid(G.GloOrbits(), ra, prec, svs)  # broadcast against precise at every product epoch
        n_all = len(gd)  # candidate rows
        gd = gd[gd.ok].reset_index(drop=True)  # rows where both are defined
        add(f"{name} T1/T2 day grid: satellites, epochs, rows used of candidates", "reported", f"{len(svs)} satellites, {gd.t.nunique()} epochs, {len(gd)} of {n_all}", None)  # the funnel
        add(f"{name}: precise orbit radius over the day grid", "25400-25600 km", f"{gd.radius_km.min():.1f}-{gd.radius_km.max():.1f} km", bool(gd.radius_km.min() >= 25400 and gd.radius_km.max() <= 25600))  # physics
        drl = np.abs(gd.rel - gd.rel_b)  # rel from the two orbits
        add(f"{name}: rel from the precise orbit against rel from the broadcast orbit", "< 1 mm", f"max {drl.max() * 1e3:.4f} mm; |rel| median {np.median(np.abs(gd.rel)):.3f} m, max {np.abs(gd.rel).max():.3f} m", bool(drl.max() < 1e-3))  # convention consistency
        add(f"{name}: largest 3-D broadcast-minus-precise difference over the day", "< 50 m (else flagged)", f"{gd.d3.max():.2f} m ({gd.loc[gd.d3.idxmax(), 'sv']})", bool(gd.d3.max() < 50.0))  # magnitude
        # T1: the common radial and the antenna-offset check
        common = float(np.median(gd.dr))  # pooled median of the radial difference over every satellite-epoch
        common_ep = float(np.median(gd.groupby("t").dr.median()))  # median of the per-epoch constellation medians
        satmed = gd.groupby("sv").dr.median()  # per-satellite radial medians
        spread = satmed - common  # per-satellite medians about the common value
        near = PCO_WINDOW[0] <= common <= PCO_WINDOW[1]  # inside the pre-registered window
        add(f"{name} T1 antenna-offset check: constellation-common broadcast-minus-precise radial median (pre-registered near -2 m: -3.0 to -1.5)", "about -2 m (APC 2-2.5 m Earthward of CoM)", f"{common:+.3f} m pooled ({common_ep:+.3f} m as median of epoch medians); per-satellite medians about it: sd {spread.std(ddof=1):.3f} m, {spread.min():+.2f} ({spread.idxmin()}) to {spread.max():+.2f} m ({spread.idxmax()})", near)  # the check
        NUM[f"{name}_t1_common"], NUM[f"{name}_t1_common_ep"], NUM[f"{name}_t1_spread_sd"], NUM[f"{name}_t1_spread_min"], NUM[f"{name}_t1_spread_max"] = common, common_ep, spread.std(ddof=1), spread.min(), spread.max()  # kept
        NUM[f"{name}_t1_spread_min_sv"], NUM[f"{name}_t1_spread_max_sv"] = spread.idxmin(), spread.idxmax()  # which satellites
        for comp in ("dr", "da", "dc", "d3"):  # for loop over the components
            add(f"{name} T1 day: {comp} pooled signed median / |median| / 95th |.|", "reported", f"{gd[comp].median():+.3f} / {np.median(np.abs(gd[comp])):.3f} / {pct(np.abs(gd[comp]), 95):.3f} m", None)  # pooled
            NUM[f"{name}_t1_{comp}_med"], NUM[f"{name}_t1_{comp}_absmed"], NUM[f"{name}_t1_{comp}_p95"] = float(gd[comp].median()), float(np.median(np.abs(gd[comp]))), pct(np.abs(gd[comp]), 95)  # kept
            for s, g in gd.groupby("sv"):  # for loop over the satellites
                long("T1_day", name, s, f"{comp}_signed_median_m", g[comp].median())  # signed median
                long("T1_day", name, s, f"{comp}_abs_median_m", np.median(np.abs(g[comp])))  # |median|
                long("T1_day", name, s, f"{comp}_abs_p95_m", pct(np.abs(g[comp]), 95))  # 95th percentile
        # T2: the three clock variants against the full apparent precise clock
        gd["D_none"], gd["D_gps"], gd["D_opp"] = gd.y + gd.rel, gd.y, gd.y + 2.0 * gd.rel  # D_V = B_V - P
        gd["w"] = gd.dr - gd.y  # ranging-level quantity w = radial - y
        gd["e_none"], gd["e_gps"], gd["e_opp"] = gd.w - gd.rel, gd.w, gd.w - 2.0 * gd.rel  # eps_V = radial - D_V
        st_c, st_r = {}, {}  # pooled 95th percentiles: clock level and ranging level
        for v in ("none", "gps", "opp"):  # for loop over the variants
            gd["Dt_" + v] = demed(gd, "D_" + v)  # clock difference with one constellation datum per epoch removed
            gd["et_" + v] = demed(gd, "e_" + v)  # ranging difference likewise
            st_c[v], st_r[v] = pct(np.abs(gd["Dt_" + v]), 95), pct(np.abs(gd["et_" + v]), 95)  # pooled 95th percentiles
            add(f"{name} T2 clock, variant {VNAME[v]}: pooled |D| median / 95th (epoch median removed)", "reported", f"{np.median(np.abs(gd['Dt_' + v])):.3f} / {st_c[v]:.3f} m", None)  # clock level
            add(f"{name} T2 ranging (radial - clock), variant {VNAME[v]}: pooled |eps| median / 95th (epoch median removed)", "reported", f"{np.median(np.abs(gd['et_' + v])):.3f} / {st_r[v]:.3f} m", None)  # ranging level
            NUM[f"{name}_t2c_{v}_med"], NUM[f"{name}_t2c_{v}_p95"] = float(np.median(np.abs(gd["Dt_" + v]))), st_c[v]  # kept
            NUM[f"{name}_t2r_{v}_med"], NUM[f"{name}_t2r_{v}_p95"] = float(np.median(np.abs(gd["et_" + v]))), st_r[v]  # kept
            for s, g in gd.groupby("sv"):  # for loop over the satellites
                long("T2_day", name, s, f"clock_{v}_signed_median_m", g["Dt_" + v].median())  # signed median
                long("T2_day", name, s, f"clock_{v}_abs_p95_m", pct(np.abs(g["Dt_" + v]), 95))  # 95th percentile
                long("T2_day", name, s, f"clock_{v}_within_sd_m", g["Dt_" + v].std(ddof=1))  # scatter about the satellite's own mean
                long("T2_day", name, s, f"ranging_{v}_signed_median_m", g["et_" + v].median())  # signed median
                long("T2_day", name, s, f"ranging_{v}_abs_p95_m", pct(np.abs(g["et_" + v]), 95))  # 95th percentile
                long("T2_day", name, s, f"ranging_{v}_within_sd_m", g["et_" + v].std(ddof=1))  # scatter
        beta, ep_y = fe_slope(gd, "y")  # clock-level slope with epoch and satellite effects
        gamma, ep_w = fe_slope(gd, "w")  # ranging-level slope
        gd["y_ep"] = gd.y - gd.t.map(ep_y)  # y with the fitted epoch effects removed
        gd["w_ep"] = gd.w - gd.t.map(ep_w)  # w likewise
        bs, gs = {}, {}  # per-satellite slopes
        for s, g in gd.groupby("sv"):  # for loop over the satellites
            bs[s] = float(np.polyfit(g.rel, g.y_ep, 1)[0])  # beta of this satellite (with its own intercept)
            gs[s] = float(np.polyfit(g.rel, g.w_ep, 1)[0])  # gamma of this satellite
            long("T2_day", name, s, "beta_clock", bs[s])  # kept
            long("T2_day", name, s, "gamma_ranging", gs[s])  # kept
            long("T2_day", name, s, "rel_amplitude_m", np.abs(g.rel).max())  # size of the periodic term for this satellite
        bv, gv = np.array(list(bs.values())), np.array(list(gs.values()))  # the per-satellite slopes as arrays
        bb = RNG.choice(len(bv), size=(2000, len(bv)), replace=True)  # bootstrap: satellites drawn with replacement, 2000 times
        b_ci, g_ci = np.percentile(bv[bb].mean(axis=1), [2.5, 97.5]), np.percentile(gv[bb].mean(axis=1), [2.5, 97.5])  # percentile intervals of the mean slopes
        wc, wr = winner(st_c, beta, BETA_T), winner(st_r, gamma, GAMMA_T)  # the pre-registered winners
        sat_c = pd.Series({s: min(("none", "gps", "opp"), key=lambda v: gd[gd.sv == s]["Dt_" + v].std(ddof=1)) for s in bs}).value_counts()  # per satellite: variant with the smallest scatter (clock)
        sat_r = pd.Series({s: min(("none", "gps", "opp"), key=lambda v: gd[gd.sv == s]["et_" + v].std(ddof=1)) for s in bs}).value_counts()  # the same at the ranging level
        add(f"{name} T2 clock slope beta (two-way fixed effects; -1 none, 0 GPS sign, -2 opposite)", "reported", f"pooled {beta:+.3f}; satellite mean {bv.mean():+.3f}, 95% bootstrap {b_ci[0]:+.3f} to {b_ci[1]:+.3f}; range {bv.min():+.2f} to {bv.max():+.2f}", None)  # beta
        add(f"{name} T2 ranging slope gamma (+1 none, 0 GPS sign, +2 opposite)", "reported", f"pooled {gamma:+.3f}; satellite mean {gv.mean():+.3f}, 95% bootstrap {g_ci[0]:+.3f} to {g_ci[1]:+.3f}; range {gv.min():+.2f} to {gv.max():+.2f}", None)  # gamma
        add(f"{name} T2 winner, clock level (pre-registered rule)", "none / gps / opp / undecided", f"{wc}; satellites best by scatter: {dict(sat_c)}", None)  # clock winner
        add(f"{name} T2 winner, ranging level (pre-registered rule, governs the advice for the GLONASS orbit script)", "none / gps / opp / undecided", f"{wr}; satellites best by scatter: {dict(sat_r)}", None)  # ranging winner
        NUM.update({f"{name}_beta": beta, f"{name}_beta_mean": bv.mean(), f"{name}_beta_lo": b_ci[0], f"{name}_beta_hi": b_ci[1], f"{name}_beta_min": bv.min(), f"{name}_beta_max": bv.max(),  # beta numbers
                    f"{name}_gamma": gamma, f"{name}_gamma_mean": gv.mean(), f"{name}_gamma_lo": g_ci[0], f"{name}_gamma_hi": g_ci[1], f"{name}_gamma_min": gv.min(), f"{name}_gamma_max": gv.max(),  # gamma numbers
                    f"{name}_win_clock": wc, f"{name}_win_ranging": wr, f"{name}_satbest_clock": "; ".join(f"{k} {v}" for k, v in sat_c.items()), f"{name}_satbest_ranging": "; ".join(f"{k} {v}" for k, v in sat_r.items())})  # winners
        # ---------------- T4: V3 run again with the precise orbit and clock ----------------
        keep = set(prec.svs)  # the satellites the product carries
        red, _ = A.v3_run(BrdcKeep(keep), kl)  # broadcast V3 on the same satellites (identical rows to the precise run)
        pre, _ = A.v3_run(prec, kl)  # the V3 computation with the precise orbit and clock, everything else the GLONASS broadcast check's
        key = ["test_code", "receiver", "sec", "sv"]  # the row identity
        same_rows = red[key].equals(pre[key])  # identical rows in identical order
        add(f"{name} T4 rows: precise run against the broadcast run on the product's satellites", "identical rows", f"{len(pre)} and {len(red)} rows, {'identical' if same_rows else 'DIFFERENT'}; published the GLONASS broadcast check {len(pub)}, of them on product satellites {int(pub.sv.isin(keep).sum())}", same_rows)  # the row check
        same_anom = bool((red.anomaly.to_numpy() == pre.anomaly.to_numpy()).all()) if same_rows else False  # the anomaly flags agree
        add(f"{name} T4 anomaly flags (|residual| > 100 km) identical in the two runs", "identical", f"{int(pre.anomaly.sum())} and {int(red.anomaly.sum())}", same_anom)  # same rows flagged
        s_red, s_pre, s_pub = resid_stats(red), resid_stats(pre), resid_stats(pub)  # V3 statistics of the three tables
        pas = s_pre["med"] < V3_MED and s_pre["p95"] < V3_P95  # the pre-registered pass
        add(f"{name} T4 V3 with precise orbit and clock, pooled (anomalies removed): |residual| median / 95th", "< 1.5 m / < 5 m", f"{s_pre['med']:.3f} / {s_pre['p95']:.3f} m over {s_pre['n']} (broadcast on the same rows {s_red['med']:.3f} / {s_red['p95']:.3f}; published the GLONASS broadcast check, all satellites {s_pub['med']:.3f} / {s_pub['p95']:.3f})", pas)  # the T4 test
        add(f"{name} T4 V3 all rows (anomalies kept): median / 95th", "reported", f"{s_pre['all_med']:.3f} / {s_pre['all_p95']:.3f} m (broadcast same rows {s_red['all_med']:.3f} / {s_red['all_p95']:.3f})", None)  # with the anomalies
        for rx in D.RXS:  # for loop over the receivers
            add(f"{name} T4 {RXN[rx]}: |residual| median / 95th (broadcast on the same rows)", "< 1.5 m / < 5 m", f"{s_pre['med_' + rx]:.3f} / {s_pre['p95_' + rx]:.3f} m ({s_red['med_' + rx]:.3f} / {s_red['p95_' + rx]:.3f}), n {s_pre['n_' + rx]}", s_pre["med_" + rx] < V3_MED and s_pre["p95_" + rx] < V3_P95)  # per receiver
        for k_, v_ in s_pre.items():  # for loop over the precise statistics
            NUM[f"{name}_t4_{k_}"] = v_  # kept
        for k_, v_ in s_red.items():  # for loop over the broadcast same-row statistics
            NUM[f"{name}_t4brdc_{k_}"] = v_  # kept
        NUM["pub_med"], NUM["pub_p95"], NUM["pub_n"] = s_pub["med"], s_pub["p95"], s_pub["n"]  # the published the GLONASS broadcast check numbers, for reference
        okp = pre[pre.anomaly == 0]  # precise rows without the anomalies
        okb = red[red.anomaly == 0]  # broadcast rows without the anomalies
        for s, g in okp.groupby("sv"):  # for loop over the satellites
            gb = okb[okb.sv == s]  # the broadcast rows of the satellite
            long("T4_sat", name, s, "precise_signed_median_m", g.res_median_m.median())  # precise signed median
            long("T4_sat", name, s, "precise_abs_p95_m", pct(np.abs(g.res_median_m), 95))  # precise 95th
            long("T4_sat", name, s, "broadcast_signed_median_m", gb.res_median_m.median())  # broadcast signed median, same rows
            long("T4_sat", name, s, "broadcast_abs_p95_m", pct(np.abs(gb.res_median_m), 95))  # broadcast 95th, same rows
            long("T4_sat", name, s, "n_residuals", len(g))  # rows
        pt = okp.groupby(["sv", "test_code", "receiver"]).res_median_m.median()  # precise satellite-session-receiver medians
        add(f"{name} T4 satellite-session-receiver medians with precise products: rms / largest |median|", "small if the offsets were broadcast error", f"rms {np.sqrt((pt ** 2).mean()):.2f} m, largest {pt.abs().max():.2f} m ({' '.join(pt.abs().idxmax())}); broadcast same rows rms {np.sqrt((okb.groupby(['sv', 'test_code', 'receiver']).res_median_m.median() ** 2).mean()):.2f} m", None)  # what is left of the offsets
        NUM[f"{name}_t4_grp_rms"], NUM[f"{name}_t4_grp_max"] = float(np.sqrt((pt ** 2).mean())), float(pt.abs().max())  # kept
        cb, cp_ = float(np.corrcoef(okb.res_median_m, okb.rel_m)[0, 1]), float(np.corrcoef(okp.res_median_m, okp.rel_m)[0, 1])  # residual against rel, broadcast and precise, same rows
        add(f"{name} T4: correlation of the residual with rel_m, broadcast against precise (same rows, the GLONASS broadcast check found -0.57 on all rows)", "reported: if it vanishes with precise products, the GLONASS broadcast check correlation came from broadcast error", f"broadcast {cb:+.3f}, precise {cp_:+.3f}", None)  # where the GLONASS broadcast check correlation comes from
        NUM[f"{name}_t4_relcorr_brdc"], NUM[f"{name}_t4_relcorr_prec"] = cb, cp_  # kept
        gk = ["sv", "test_code", "receiver"]  # one group per satellite-session-receiver (one pass in a clean interval)
        rw = float((okp.rel_m - okp.groupby(gk).rel_m.transform("mean")).std(ddof=1))  # spread of rel inside the passes
        rbt = float(okp.groupby(gk).rel_m.mean().std(ddof=1))  # spread of the pass means of rel
        gb_, gp_ = okb.groupby(gk)[["res_median_m", "rel_m"]].median(), okp.groupby(gk)[["res_median_m", "rel_m"]].median()  # pass medians of residual and rel
        cgb, cgp = float(gb_.corr().iloc[0, 1]), float(gp_.corr().iloc[0, 1])  # pass-level correlations
        add(f"{name} T4: rel_m within the clean passes against between them; pass-level residual-rel correlation (broadcast / precise)", "reported: in 2-5 min passes rel is almost constant, so the row correlation rests on the number of passes", f"within-pass sd {rw:.3f} m, sd of pass means {rbt:.3f} m; {len(gp_)} passes, correlation {cgb:+.3f} / {cgp:+.3f}", None)  # effective sample size of the GLONASS broadcast check relativity correlation
        NUM[f"{name}_rel_within_sd"], NUM[f"{name}_rel_between_sd"], NUM[f"{name}_rel_npass"], NUM[f"{name}_rel_passcorr_brdc"], NUM[f"{name}_rel_passcorr_prec"] = rw, rbt, len(gp_), cgb, cgp  # kept
        NUM[f"{name}_t4brdc_grp_rms"] = float(np.sqrt((okb.groupby(["sv", "test_code", "receiver"]).res_median_m.median() ** 2).mean()))  # kept
        # ---------------- T3: predicted residual offset from radial minus clock ----------------
        rr = okb[key].merge(obs, on=key, how="left")  # the broadcast same-row residual rows with their time and pseudorange
        add(f"{name} T3 rows with an observation (time and pseudorange) found", "all", f"{int(rr.pr.notna().sum())} of {len(rr)}", bool(rr.pr.notna().all()))  # join check
        rs = row_states(G.GloOrbits(), ra, prec, rr)  # broadcast and precise state at every row
        add(f"{name} T3 rows where both sources are defined", "all", f"{int(rs.ok.sum())} of {len(rs)}", bool(rs.ok.all()))  # definition check
        # self-check: this script's broadcast arithmetic against the GLONASS orbit script's state_full on every row
        o8 = G.GloOrbits()  # a fresh the GLONASS orbit script object
        dmax_p, dmax_t = 0.0, 0.0  # largest position and clock differences
        for (_, _, _), g in rs.groupby(["test_code", "receiver", "sec"]):  # for loop over the receiver-epochs
            p8, _, d8, k8 = o8.state_full(list(g.sv), float(g.t_gps.iloc[0]), g.pr.to_numpy())  # the GLONASS orbit script's own computation
            dmax_p = max(dmax_p, float(np.abs(p8[k8] - g[["xb", "yb", "zb"]].to_numpy()[k8]).max()))  # position difference
            dmax_t = max(dmax_t, float(np.abs(d8[k8] - g.dts_b.to_numpy()[k8]).max()))  # clock difference
        add(f"{name} T3 self-check: broadcast state here against the GLONASS orbit script's state_full on every row", "< 1 mm, < 1 ps", f"max {dmax_p * 1e3:.6f} mm, {dmax_t * 1e12:.6f} ps", bool(dmax_p < 1e-3 and dmax_t < 1e-12))  # replica check
        add(f"{name} T3: broadcast minus precise clock at the rows (datum difference; moves the evaluation instant)", "reported; < 1 us keeps the instant within 4 mm", f"median {np.median(rs.dts_b - rs.dts_p) * 1e6:+.4f} us, max |.| {np.abs(rs.dts_b - rs.dts_p).max() * 1e6:.4f} us", None)  # datum size
        rs["E"] = rs.dr - rs.d0  # radial minus clock (T2 primary), metres
        sites = {t_: D.site_xyz(M.SITE_OF[t_]) for t_ in sorted(rs.test_code.unique())}  # dict comprehension: session -> site position
        rho_b, rho_p = np.zeros(len(rs)), np.zeros(len(rs))  # line-of-sight ranges to the two positions
        for t_, x_ in sites.items():  # for loop over the sessions
            m = (rs.test_code == t_).to_numpy()  # rows of this session
            # with-block: NumPy's Apple Accelerate matrix product raises spurious divide/overflow warnings on large finite arrays (reproduced on random data); the ranges are checked below instead
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                rho_b[m] = M.geometry(x_, rs.loc[m, ["xb", "yb", "zb"]].to_numpy())[0]  # Sagnac-corrected range to the broadcast position
                rho_p[m] = M.geometry(x_, rs.loc[m, ["xp", "yp", "zp"]].to_numpy())[0]  # and to the precise position
        add(f"{name} T3 line-of-sight ranges finite and plausible (GLONASS seen from the ground: about 19,100-25,000 km)", "all finite, 19,000-25,500 km", f"{rho_p.min() / 1e3:.0f}-{rho_p.max() / 1e3:.0f} km, finite {bool(np.isfinite(rho_b).all() and np.isfinite(rho_p).all())}", bool(np.isfinite(rho_b).all() and np.isfinite(rho_p).all() and rho_p.min() > 1.9e7 and rho_p.max() < 2.55e7))  # range sanity
        rs["E_los"] = (rho_b - rho_p) - rs.d0  # line-of-sight version at the actual site
        ep = ["test_code", "receiver", "sec"]  # the receiver-epoch
        rs["P"] = -demed(rs, "E", ep)  # predicted residual: minus E about the receiver-epoch median
        rs["P_los"] = -demed(rs, "E_los", ep)  # the line-of-sight prediction
        # T1 at the clean rows (the clean intervals hold only three 5 min epochs, so the rows themselves are used)
        for comp in ("dr", "da", "dc"):  # for loop over the components
            NUM[f"{name}_t1clean_{comp}_med"], NUM[f"{name}_t1clean_{comp}_p95"] = float(rs[comp].median()), pct(np.abs(rs[comp]), 95)  # pooled
            for s, g in rs.groupby("sv"):  # for loop over the satellites
                long("T1_clean", name, s, f"{comp}_signed_median_m", g[comp].median())  # signed median at the clean rows
        add(f"{name} T1 at the clean rows: radial / along / cross signed median (95th |.|)", "reported", ", ".join(f"{rs[c_].median():+.2f} ({pct(np.abs(rs[c_]), 95):.2f})" for c_ in ("dr", "da", "dc")) + " m", None)  # pooled
        # T3 regression: published the GLONASS broadcast check offsets (primary), reduced-set offsets and the line-of-sight variant
        gp = rs.groupby(["sv", "test_code", "receiver"])[["P", "P_los"]].median()  # predicted offsets per satellite-session-receiver
        off_pub = pub[(pub.anomaly == 0) & pub.sv.isin(keep)].groupby(["sv", "test_code", "receiver"]).res_median_m.median()  # published the GLONASS broadcast check medians
        off_red = okb.groupby(["sv", "test_code", "receiver"]).res_median_m.median()  # same-row broadcast medians
        tab = gp.join(off_pub.rename("off_pub"), how="inner").join(off_red.rename("off_red"), how="inner")  # one row per satellite-session-receiver
        res3 = {}  # results of the three comparisons
        for lab, xcol, ycol in (("primary", "P", "off_pub"), ("reduced", "P", "off_red"), ("los", "P_los", "off_pub")):  # for loop over the three variants
            sl, ic = np.polyfit(tab[xcol], tab[ycol], 1)  # slope and intercept of offset on prediction
            r = float(np.corrcoef(tab[xcol], tab[ycol])[0, 1])  # correlation
            rms = float(np.sqrt(((tab[ycol] - tab[xcol]) ** 2).mean()))  # rms of offset minus prediction
            res3[lab] = (r, sl, ic, rms)  # kept
            NUM[f"{name}_t3_{lab}_r"], NUM[f"{name}_t3_{lab}_slope"], NUM[f"{name}_t3_{lab}_icpt"], NUM[f"{name}_t3_{lab}_rms"] = r, sl, ic, rms  # kept
        r, sl, ic, rms = res3["primary"]  # the pre-registered comparison
        t3 = r >= T3_CORR and T3_SLOPE[0] <= sl <= T3_SLOPE[1]  # the pre-registered reading
        add(f"{name} T3 PRIMARY: the GLONASS broadcast check satellite-session-receiver offset regressed on predicted P = -(radial - clock), published offsets", "correlation >= 0.8 and slope 0.7-1.3", f"r {r:.3f}, slope {sl:.3f}, intercept {ic:+.3f} m, rms(offset - P) {rms:.3f} m, n {len(tab)} (rms of the offsets {np.sqrt((tab.off_pub ** 2).mean()):.3f} m)", t3)  # T3
        for lab in ("reduced", "los"):  # for loop over the secondary comparisons
            r_, s_, i_, m_ = res3[lab]  # their numbers
            add(f"{name} T3 secondary ({'offsets recomputed on the product satellites' if lab == 'reduced' else 'line-of-sight prediction at the site'})", "reported", f"r {r_:.3f}, slope {s_:.3f}, intercept {i_:+.3f} m, rms {m_:.3f} m", None)  # secondary
        NUM[f"{name}_t3_n"], NUM[f"{name}_t3_offrms"], NUM[f"{name}_t3_pass"] = len(tab), float(np.sqrt((tab.off_pub ** 2).mean())), int(t3)  # kept
        sh_ = tab.off_pub - tab.off_red  # published offset minus the offset recomputed without R01 and R10 in the epoch medians
        add(f"{name} T3: published minus same-row the GLONASS broadcast check offsets (the epoch medians of the GLONASS broadcast check also contained R01 and R10)", "reported; explains an intercept", f"mean {sh_.mean():+.3f} m, median {sh_.median():+.3f} m, range {sh_.min():+.2f} to {sh_.max():+.2f} m", None)  # the median shift
        NUM[f"{name}_t3_shift_mean"], NUM[f"{name}_t3_shift_min"], NUM[f"{name}_t3_shift_max"] = float(sh_.mean()), float(sh_.min()), float(sh_.max())  # kept
        for (s, t_, rx), row in tab.iterrows():  # for loop over the satellite-session-receiver points
            long("T3_group", name, s, "predicted_P_m", row.P, t_, rx)  # the radial prediction
            long("T3_group", name, s, "predicted_P_los_m", row.P_los, t_, rx)  # the line-of-sight prediction
            long("T3_group", name, s, "broadcast_offset_published_m", row.off_pub, t_, rx)  # the published offset
            long("T3_group", name, s, "broadcast_offset_same_rows_m", row.off_red, t_, rx)  # the same-row offset
        # exact check: broadcast residual minus precise residual, row by row, against the line-of-sight prediction
        dd = okb[key + ["res_median_m"]].merge(okp[key + ["res_median_m"]], on=key, suffixes=("_b", "_p")).merge(rs[key + ["P_los"]], on=key)  # rows of both runs with the prediction
        dif = dd.res_median_m_b - dd.res_median_m_p  # what changing the orbit and clock did to each residual
        add(f"{name} T3 exact check: broadcast minus precise residual per row against P_los (median-of-differences vs difference-of-medians)", "close agreement", f"r {np.corrcoef(dif, dd.P_los)[0, 1]:.4f}, rms difference {np.sqrt(((dif - dd.P_los) ** 2).mean()):.3f} m, n {len(dd)}", None)  # the exact comparison
        NUM[f"{name}_exact_r"], NUM[f"{name}_exact_rms"] = float(np.corrcoef(dif, dd.P_los)[0, 1]), float(np.sqrt(((dif - dd.P_los) ** 2).mean()))  # kept
        figdat[name] = {"gd": gd, "bs": bs, "gs": gs, "beta": beta, "gamma": gamma, "tab": tab, "res3": res3, "pre": okp, "red": okb, "satmed": satmed, "common": common}  # for the figure
    # ---------------- ESA against GFZ on the shared satellites ----------------
    pe, pg = precs["ESA"], precs["GFZ"]  # the two products
    sh = sorted(set(pe.svs) & set(pg.svs))  # shared satellites
    t = np.array(sorted(set(pe.T) & set(pg.T)))  # shared epochs
    rows = []  # per satellite tables
    for s in sh:  # for loop over the shared satellites
        p1, p2, v1 = pe.pos(s, t), pg.pos(s, t), pe.vel(s, t)  # positions and ESA velocity
        g = pd.DataFrame({"t": t, "sv": s})  # rows
        g["dr"], g["da"], g["dc"] = rac(p1 - p2, p1, v1)  # ESA minus GFZ in the orbit frame
        g["dclk"] = C * (pe.clock(s, t) - pg.clock(s, t))  # clock difference, metres (each product's own datum)
        rows.append(g)  # kept
    eg = pd.concat(rows, ignore_index=True).dropna().reset_index(drop=True)  # one table, defined rows only
    eg["dclk_t"] = demed(eg, "dclk")  # one datum per epoch removed
    eg["rng_t"] = demed(eg.assign(rng=eg.dr - eg.dclk), "rng")  # radial minus clock, datum removed (what a range sees)
    for s, g in eg.groupby("sv"):  # for loop over the satellites
        for c_ in ("dr", "da", "dc", "dclk_t", "rng_t"):  # for loop over the quantities
            long("ESA_GFZ_day", "ESA-GFZ", s, f"{c_}_signed_median_m", g[c_].median())  # signed median
            long("ESA_GFZ_day", "ESA-GFZ", s, f"{c_}_abs_p95_m", pct(np.abs(g[c_]), 95))  # 95th percentile
    satc = eg.groupby("sv").dclk_t.median()  # per-satellite clock offsets between the products
    satr = eg.groupby("sv").rng_t.median()  # per-satellite ranging offsets between the products
    add("ESA-GFZ: shared satellites and epochs", "20 and 288 (the 24:00 epoch has no clock)", f"{len(sh)} and {eg.t.nunique()} ({len(eg)} rows)", None)  # sizes
    add("ESA-GFZ orbit: radial / along / cross signed median (95th |.|)", "centimetres to decimetres", ", ".join(f"{eg[c_].median():+.3f} ({pct(np.abs(eg[c_]), 95):.3f})" for c_ in ("dr", "da", "dc")) + " m", None)  # orbit agreement
    add("ESA-GFZ clock after the epoch datum: per-satellite medians range / pooled 95th |.|", "reported (antenna models differ: ESA23 against IGS20)", f"{satc.min():+.3f} ({satc.idxmin()}) to {satc.max():+.3f} m ({satc.idxmax()}); {pct(np.abs(eg.dclk_t), 95):.3f} m", None)  # clock agreement
    add("ESA-GFZ ranging (radial - clock) after the datum: per-satellite medians range / pooled 95th |.|", "small: what a pseudorange sees", f"{satr.min():+.3f} ({satr.idxmin()}) to {satr.max():+.3f} m ({satr.idxmax()}); {pct(np.abs(eg.rng_t), 95):.3f} m", None)  # ranging agreement
    NUM.update({"eg_n_sv": len(sh), "eg_n_ep": int(eg.t.nunique()), "eg_dr_med": float(eg.dr.median()), "eg_dr_p95": pct(np.abs(eg.dr), 95), "eg_da_p95": pct(np.abs(eg.da), 95), "eg_dc_p95": pct(np.abs(eg.dc), 95),  # orbit numbers
                "eg_clk_satmin": float(satc.min()), "eg_clk_satmax": float(satc.max()), "eg_clk_p95": pct(np.abs(eg.dclk_t), 95), "eg_rng_satmin": float(satr.min()), "eg_rng_satmax": float(satr.max()), "eg_rng_p95": pct(np.abs(eg.rng_t), 95)})  # clock and ranging numbers
    a3 = figdat["ESA"]["tab"].join(figdat["GFZ"]["tab"][["P"]].rename(columns={"P": "P_gfz"}), how="inner")  # the predictions of the two products per point
    add("ESA-GFZ: T3 predicted offsets per satellite-session-receiver, correlation and rms difference", "close agreement", f"r {np.corrcoef(a3.P, a3.P_gfz)[0, 1]:.3f}, rms {np.sqrt(((a3.P - a3.P_gfz) ** 2).mean()):.3f} m, n {len(a3)}", None)  # T3 consistency between products
    NUM["eg_t3_r"], NUM["eg_t3_rms"] = float(np.corrcoef(a3.P, a3.P_gfz)[0, 1]), float(np.sqrt(((a3.P - a3.P_gfz) ** 2).mean()))  # kept
    figure(figdat)  # the figure
    protected("end")  # protected files after everything
    pd.DataFrame(CHK).to_csv(CHECK_DIR / "glonass_precise_check.csv", index=False)  # one row per check
    lt = pd.DataFrame(LONG, columns=["table", "product", "sv", "test_code", "receiver", "metric", "value"])  # the long table
    lt.to_csv(CHECK_DIR / "glonass_precise_per_satellite.csv", index=False, float_format="%.6g")  # written with fixed rounding
    pd.DataFrame({"key": list(NUM), "value": [NUM[k_] if isinstance(NUM[k_], str) else float(NUM[k_]) for k_ in NUM]}).to_csv(CHECK_DIR / "glonass_precise_summary.csv", index=False, float_format="%.6g")  # the numbers
    print("done")  # finished


def figure(fd: dict) -> None:  # function: the four-panel figure (T1, T2, T3, T4)
    fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))  # a 2 x 2 grid of panels
    # (a) T1 per satellite, ESA, whole day: radial, along, cross and clock (D0 about the epoch median)
    d = fd["ESA"]["gd"]  # the ESA day grid
    svs = sorted(d.sv.unique())  # satellites on the x axis
    x = np.arange(len(svs))  # their positions
    comps = (("dr", "radial", "#0072B2", "o"), ("da", "along-track", "#E69F00", "s"), ("dc", "cross-track", "#009E73", "^"), ("Dt_none", "clock D0 (epoch median out)", "#56B4E9", "D"))  # tuple of (column, label, colour, marker)
    for j, (c_, lab, col, mk) in enumerate(comps):  # for loop over the four quantities
        med = d.groupby("sv")[c_].median().reindex(svs)  # per-satellite signed medians
        ax[0, 0].scatter(x + (j - 1.5) * 0.18, med, s=30, color=col, marker=mk, label=lab, edgecolor="white", lw=0.6, zorder=3)  # one marker per satellite
    ax[0, 0].axhline(fd["ESA"]["common"], color="#0072B2", lw=1.0, ls="--", label=f"common radial median {fd['ESA']['common']:+.2f} m")  # the common radial value
    # the panel shaded only the physical -2.5 to -2.0 m range, narrower than the band T1 actually tested; now the tested
    # band PCO_WINDOW is shaded, and the physical range is drawn inside it with its own label
    ax[0, 0].axhspan(PCO_WINDOW[0], PCO_WINDOW[1], color="0.92", zorder=0, label=f"T1 tested band ({PCO_WINDOW[0]:.1f} to {PCO_WINDOW[1]:.1f} m)")  # the pre-registered window
    ax[0, 0].axhspan(-2.5, -2.0, color="0.80", zorder=0, label="expected antenna offset (-2.5 to -2 m)")  # the physical range inside it
    ax[0, 0].axhline(0, color="k", lw=0.6)  # zero line
    ax[0, 0].set_xticks(x)  # one tick per satellite
    ax[0, 0].set_xticklabels(svs, rotation=90, fontsize=8)  # satellite names
    ax[0, 0].set_ylabel("broadcast minus precise (ESA), median over the day (m)")  # y label
    ax[0, 0].set_title("(a) T1: orbit and clock differences per satellite")  # title
    ax[0, 0].set_ylim(-8.0, 8.5)  # y range: room above the data for the legend (R03's along-track median is near -7 m)
    ax[0, 0].legend(fontsize=7, loc="upper center", ncol=3)  # legend above the data, so no point is hidden
    # (b) T2 slopes per satellite: beta (clock) and -gamma (ranging), both products; reference lines for the three variants
    for k, name in enumerate(("ESA", "GFZ")):  # for loop over the products
        f = fd[name]  # this product's data
        sv2 = sorted(f["bs"])  # its satellites
        xi = np.array([svs.index(s) if s in svs else np.nan for s in sv2], dtype=float)  # list comprehension: x positions on panel (a)'s order
        ax[0, 1].scatter(xi + (k - 0.5) * 0.3, [f["bs"][s] for s in sv2], s=30, color=PCOL[name], marker="o", edgecolor="white", lw=0.6, label=f"{name} beta, clock (pooled {f['beta']:+.2f})", zorder=3)  # beta per satellite
        ax[0, 1].scatter(xi + (k - 0.5) * 0.3, [-f["gs"][s] for s in sv2], s=30, facecolor="none", edgecolor=PCOL[name], marker="s", lw=1.2, label=f"{name} -gamma, ranging (pooled {-f['gamma']:+.2f})", zorder=3)  # -gamma per satellite
    for yv, lab in ((-1.0, "none"), (0.0, "GPS sign"), (-2.0, "opposite sign")):  # for loop over the reference values
        ax[0, 1].axhline(yv, color="0.4", lw=0.8, ls=":")  # the reference line
        ax[0, 1].text(len(svs) - 0.5, yv + 0.05, lab, fontsize=8, ha="right", va="bottom", color="0.25")  # its label
    ax[0, 1].set_xticks(x)  # one tick per satellite
    ax[0, 1].set_xticklabels(svs, rotation=90, fontsize=8)  # satellite names
    ax[0, 1].set_ylabel("slope on rel = 2 r.v/c (beta; -gamma)")  # y label
    ax[0, 1].set_title("(b) T2: which relativistic variant matches the precise clock")  # title
    ax[0, 1].legend(fontsize=7, loc="upper left")  # legend
    # (c) T3: published the GLONASS broadcast check offset against the predicted offset (ESA), coloured by receiver
    tab = fd["ESA"]["tab"].reset_index()  # the points
    for rx in D.RXS:  # for loop over the receivers
        q = tab[tab.receiver == rx]  # that receiver's points
        ax[1, 0].scatter(q.P, q.off_pub, s=28, color=M.COLORS[rx], edgecolor="white", lw=0.6, label=f"{RXN[rx]} (n {len(q)})", zorder=3)  # one dot per satellite-session
    lim = float(np.ceil(max(np.abs(tab.P).max(), np.abs(tab.off_pub).max()) + 0.5))  # symmetric axis limit
    xx = np.array([-lim, lim])  # line ends
    ax[1, 0].fill_between(xx, 0.7 * xx, 1.3 * xx, color="0.88", zorder=0, label="slopes 0.7-1.3 (pre-registered)")  # the slope band through the origin
    ax[1, 0].plot(xx, xx, color="k", lw=0.8, label="1:1")  # the 1:1 line
    r, sl, ic, rms = fd["ESA"]["res3"]["primary"]  # the fit numbers
    ax[1, 0].plot(xx, ic + sl * xx, color="#D55E00", lw=1.2, label=f"fit: slope {sl:.2f}, r {r:.2f}")  # the fitted line
    ax[1, 0].set_xlim(-lim, lim)  # x range
    ax[1, 0].set_ylim(-lim, lim)  # y range
    ax[1, 0].set_xlabel("predicted residual offset P = -(radial - clock), ESA (m)")  # x label
    ax[1, 0].set_ylabel("the GLONASS broadcast check residual offset, satellite-session-receiver median (m)")  # y label
    ax[1, 0].set_title("(c) T3: the GLONASS broadcast check offsets against broadcast-ephemeris error")  # title
    ax[1, 0].legend(fontsize=7, loc="upper left")  # legend
    # (d) T4: distribution of |residual|, broadcast against the two precise products on identical rows
    for lab, res, col in (("broadcast (same rows)", fd["ESA"]["red"], PCOL["broadcast"]), ("ESA precise", fd["ESA"]["pre"], PCOL["ESA"]), ("GFZ precise", fd["GFZ"]["pre"], PCOL["GFZ"])):  # for loop over the three tables
        a = np.sort(np.abs(res.res_median_m.to_numpy()))  # sorted |residuals|
        ax[1, 1].plot(a, np.arange(1, len(a) + 1) / len(a), color=col, lw=2.0, label=f"{lab}: median {np.median(a):.2f}, 95th {pct(a, 95):.2f} m")  # empirical cumulative distribution
    for v in (V3_MED, V3_P95):  # for loop over the thresholds
        ax[1, 1].axvline(v, color="0.4", lw=0.8, ls="--")  # the threshold
    for q_ in (0.5, 0.95):  # for loop over the two quantiles
        ax[1, 1].axhline(q_, color="0.7", lw=0.6, ls=":")  # the quantile level
    ax[1, 1].set_xlim(0, 10)  # x range (the tail beyond 10 m is in the numbers)
    ax[1, 1].set_xlabel("|G1 C1C residual| about the epoch median clock (m)")  # x label
    ax[1, 1].set_ylabel("cumulative fraction of residuals")  # y label
    ax[1, 1].set_title("(d) T4: V3 run again with precise orbit and clock (thresholds dashed)")  # title
    ax[1, 1].legend(fontsize=7, loc="lower right")  # legend
    fig.tight_layout()  # spacing
    save_fig(fig, GRAPH / "glonass_broadcast_vs_precise.png")  # byte-stable save


if __name__ == "__main__":  # if: run as a script
    main()  # run everything
