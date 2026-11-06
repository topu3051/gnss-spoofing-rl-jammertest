#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
Row 8 - GLONASS broadcast-orbit propagator. A module meant to be imported by later row 8 code; it builds no feature, no label and
no model, and it has no main().

Source: the R records of the IGS merged broadcast file BRDC00IGS_R_20262570000_01D_MN.rnx.gz only (the file the reference-position and evidence scripts read, precise products
are not used). Each record gives, at its reference time tb, the satellite position, velocity and lunisolar acceleration in the
Earth-fixed PZ-90 frame (km, km/s, km/s^2 in the file, metres here), the clock terms -TauN and +GammaN, a health flag and the
frequency channel number.

Time. RINEX 3 writes the GLONASS record epoch (tb) in UTC, that is tb in Moscow time minus 3 h, so the 3 h are already removed. GPS time
is UTC + 18 s in September 2026 (the file header's LEAP SECONDS line and the project's LEAP_SECONDS agree), so tb_gps = tb_utc + 18 s.
The header's GLUT term (-0.93 ns, 0.28 m) is the constant offset of GLONASS time from UTC(SU): the per-constellation receiver clock term
absorbs it. GLONASS time carries leap seconds like UTC, so no further conversion is needed.

Record choice. Healthy records only (health flag 0). At an epoch the record with the nearest tb is used, provided it is within 15 min
(records come every 30 min on:15 and:45, so a healthy satellite always has one). Ties happen at EVERY half hour: an epoch at exactly
:00 or:30 is 15 min from two records, and the rule is that the EARLIER tb wins. The tie is judged on the receiver epoch time. A
satellite with no healthy record within 15 min is not used (no extrapolation) and the reason is kept: unhealthy (the slot has records
but none healthy), no_record (the slot has none), gap (healthy records exist but none within 15 min).

Propagation. Equations of motion in the rotating Earth-fixed PZ-90 frame (GLONASS ICD edition 5.1, Appendix 3, integrates them in an absolute geocentric frame instead), integrated by fourth-order
Runge-Kutta: central gravity, the J2 term, the rotation terms (omega^2 x, 2 omega v) and the lunisolar acceleration of the record held
constant. The result is PZ-90 Earth-fixed position and velocity at the integration time. Steps are at most 30 s. PZ-90.11 differs from
WGS-84/ITRF by a few centimetres, which is ignored (far below the metre-level thresholds).

Clock. dts = (-TauN) + GammaN * (t - tb), the RINEX value of the first record line plus the frequency term. RINEX stores -TauN already,
so there is no second negation (RTKLIB does the same). No relativistic term is applied: the broadcast TauN and GammaN already contain the
periodic relativistic correction (the orbits are not exactly circular and the term can reach decimetres, which is why standard GLONASS
processing applies none). The ICD broadcasts only the G1-G2 delay; I know of no G1 C1C group-delay term, so none is applied. The size
of the periodic term is reported as a diagnostic by the check script and is never applied.

Frequency. G1 is 1602 MHz + k * 0.5625 MHz. The channel k comes from Data/processed/ground_truth/glonass_channels.csv first (the union of the RINEX
headers), and from the IGS record's own frequency number where the table has no entry, so no healthy satellite is dropped for a
missing channel.

Reads   the IGS file; Data/processed/ground_truth/glonass_channels.csv
Writes  nothing (a module)
"""

from __future__ import annotations  # future import: type hints kept as text (a convention across the project's scripts)

import gzip  # standard-library gzip reader: opens the compressed IGS file as text
import importlib.util  # standard-library module loader, to load the position script whose name starts with a digit
import os  # operating-system access: MPLCONFIGDIR is set before matplotlib loads
import sys  # standard-library system access: sys.path
from datetime import datetime, timezone  # date-time classes, to read the record epochs
from pathlib import Path  # Path objects for building file paths

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
os.environ["MPLCONFIGDIR"] = str(PROJECT / "codes" / ".mplconfig")  # matplotlib's cache inside the project folder (the position script imports matplotlib)

import numpy as np  # NumPy: arrays and the vectorised integrator
import pandas as pd  # pandas: the channel table

sys.path.insert(0, str(PROJECT / "codes" / "common"))  # puts codes/ first on the import path so the helpers module is found
from helpers import GPS_EPOCH, LEAP_SECONDS  # the GPS epoch and the 18 s between GPS time and UTC


def load_module(name: str, filename: str):  # function: loads codes/<filename> as a module (names start with a digit)
    spec = importlib.util.spec_from_file_location(name, PROJECT / "codes" / filename)  # where to load from
    mod = importlib.util.module_from_spec(spec)  # an empty module object from that spec
    sys.modules[name] = mod  # registers it under name, as a normal import would
    spec.loader.exec_module(mod)  # runs the file's top-level code (its main() is guarded and not run)
    return mod  # hands back the loaded module


M = load_module("reference_positions_for_glonass_orbit", "2_preprocessing/03_reference_positions.py")  # the position script: geometry, Saastamoinen, Klobuchar, ecef_to_llh

IGS_GZ = PROJECT / "Data" / "external" / "igs_brdc" / "BRDC00IGS_R_20262570000_01D_MN.rnx.gz"  # the IGS merged broadcast file (read only)
CHANNELS_CSV = PROJECT / "Data" / "processed" / "ground_truth" / "glonass_channels.csv"  # the slot -> channel table written by the observables script
C_LIGHT = 299792458.0  # float constant: speed of light, m/s
# constants this module uses: gravitational parameter (m^3/s^2), Earth rotation rate (rad/s), semi-major axis (m), J2; ICD edition 5.1 (PZ-90.02, Appendix 3) prints mu 398600.44 km^3/s^2, a_e 6378.136 km, C20 -1082.63e-6
# correction to the comments of this line and the next: ICD 5.1 also prints, in Table 3.2 (PZ-90.02 constants, PDF p. 24), mu 398 600.4418e9 m^3/s^2 and omega 7.292115e-5 rad/s
# so MU below IS the ICD 5.1 value (Table 3.2); OMEGA 7.2921151467e-5 is not (the ICD prints 7.292115e-5, Table 3.2 and A.3.1.2, PDF pp. 24, 57), nor is J2 (A.3.1.2 prints 1082625.7e-9)
MU, OMEGA, A_E, J2 = 398600.4418e9, 7.2921151467e-5, 6378136.0, 1082625.75e-9  # tuple assignment: the four values used here; a_e equals the ICD 5.1 value, mu, omega and J2 are not taken from that edition
# correction: the next line's values are not merely rounded, they are themselves ICD 5.1 values: mu 398600.44 km^3/s^2 of Appendix 3 (PDF p. 51) and omega 7.292115e-5 rad/s of Table 3.2 (PDF p. 24)
# so the GLONASS broadcast check's comparison (05_glonass_orbit_check.py) measures how much the values of the line above change a position against these ICD values; of the line above's MU and OMEGA, only OMEGA is not an ICD 5.1 value
MU_OLD, OMEGA_OLD = 398600.44e9, 7.292115e-5  # the rounded values first used, kept only to report how much the ICD values change a position
STEP_S = 30.0  # float constant: longest integration step in seconds (the method allows up to 60)
MAX_AGE_S = 900.0  # float constant: a record is used only within 15 min of its tb
F_G1_0, F_G1_DF, F_L1 = 1602.0e6, 0.5625e6, 1575.42e6  # GLONASS G1 centre frequency, channel spacing and GPS L1 frequency in Hz


def _f(s: str) -> float:  # function: a Fortran D-exponent field as a float
    return float(s.replace("D", "E"))  # swap the exponent letter and convert


def read_records(path: Path = IGS_GZ) -> tuple:  # function: every GLONASS record of the IGS file as a table, plus the header's leap-second count
    with gzip.open(path, "rt", errors="replace") as fh:  # opens the compressed file as text
        lines = fh.read().split("\n")  # all its lines
    leap = None  # the header's leap-second count, not found yet
    k0 = 0  # index of the first data line
    for i, l in enumerate(lines):  # for loop over the lines, with their index
        if l[60:].startswith("LEAP SECONDS"):  # if: the leap-second header line
            leap = int(l[:6])  # its first integer is the GPS-UTC offset
        if l[60:].startswith("END OF HEADER"):  # if: the header ends
            k0 = i + 1  # data start on the next line
            break  # stop looking
    rows = []  # one dict per record
    i = k0  # running line index
    while i < len(lines):  # while loop: walks the data lines
        l = lines[i]  # the current line
        if l[:1] == "R" and l[1:3].isdigit() and i + 3 < len(lines):  # if: a GLONASS record starts here (four lines long)
            a, b, c = lines[i + 1], lines[i + 2], lines[i + 3]  # the three continuation lines
            utc = datetime(int(l[4:8]), int(l[9:11]), int(l[12:14]), int(l[15:17]), int(l[18:20]), int(l[21:23]), tzinfo=timezone.utc)  # tb as UTC calendar time
            fld = lambda s, j: _f(s[4 + 19 * j:23 + 19 * j])  # lambda: the j-th value (0-3) of a continuation line
            rows.append({  # dict literal: this record's numbers, kilometres converted to metres
                "sv": l[:3], "slot": int(l[1:3]),  # satellite name and slot number
                "tb_gps": (utc - GPS_EPOCH).total_seconds() + LEAP_SECONDS,  # tb as GPS seconds (UTC calendar plus 18 s)
                "clk": _f(l[23:42]), "gam": _f(l[42:61]),  # -TauN (s) and +GammaN (s/s)
                "x": fld(a, 0) * 1e3, "y": fld(b, 0) * 1e3, "z": fld(c, 0) * 1e3,  # position, m
                "vx": fld(a, 1) * 1e3, "vy": fld(b, 1) * 1e3, "vz": fld(c, 1) * 1e3,  # velocity, m/s
                "ax": fld(a, 2) * 1e3, "ay": fld(b, 2) * 1e3, "az": fld(c, 2) * 1e3,  # lunisolar acceleration, m/s^2
                "health": int(round(fld(a, 3))), "k_igs": int(round(fld(b, 3))), "age": fld(c, 3)})  # health flag (0 healthy), IGS channel number, operation age
            i += 4  # skip the whole record
        else:  # else: some other line
            i += 1  # next line
    return pd.DataFrame(rows), leap  # the table and the header's leap seconds


def deriv(s: np.ndarray, acc: np.ndarray, mu: float, om: float) -> np.ndarray:  # function: time derivative of the state (x, y, z, vx, vy, vz) in the rotating Earth-fixed PZ-90 frame (ICD 5.1, Appendix 3, uses an absolute frame)
    x, y, z, vx, vy, vz = s.T  # unpack the six columns (one row per satellite)
    r2 = x * x + y * y + z * z  # squared radius
    r = np.sqrt(r2)  # radius
    g = -mu / (r2 * r)  # central-gravity factor: acceleration = g * position
    c = 1.5 * J2 * mu * A_E ** 2 / (r2 * r2 * r)  # J2 factor: (3/2) J2 mu a^2 / r^5
    q = 5.0 * z * z / r2  # the 5 z^2 / r^2 term shared by the J2 components
    ax = g * x - c * x * (1.0 - q) + om * om * x + 2.0 * om * vy + acc[:, 0]  # x acceleration: gravity, J2, centrifugal, Coriolis, lunisolar
    ay = g * y - c * y * (1.0 - q) + om * om * y - 2.0 * om * vx + acc[:, 1]  # y acceleration, Coriolis sign reversed
    az = g * z - c * z * (3.0 - q) + acc[:, 2]  # z acceleration: no rotation terms along the axis
    return np.stack([vx, vy, vz, ax, ay, az], axis=1)  # derivative: velocities, then accelerations


def propagate(s0: np.ndarray, acc: np.ndarray, dt: np.ndarray, step: float = STEP_S, mu: float = MU, om: float = OMEGA) -> np.ndarray:  # function: integrates each row's state over its own dt (seconds, either sign) by RK4
    s = np.array(s0, dtype=float)  # working copy of the states (n x 6)
    dt = np.asarray(dt, dtype=float)  # the time spans, one per row
    n = np.maximum(np.ceil(np.abs(dt) / step), 1.0).astype(int)  # steps per row, none longer than `step`
    h = dt / n  # each row's own step length (signed)
    for i in range(int(n.max())):  # for loop over the step index, up to the longest row
        m = i < n  # rows that still have steps left
        hm = h[m][:, None]  # their step lengths as a column
        sm, am = s[m], acc[m]  # their states and accelerations
        k1 = deriv(sm, am, mu, om)  # RK4 slope at the start
        k2 = deriv(sm + 0.5 * hm * k1, am, mu, om)  # slope at the midpoint using k1
        k3 = deriv(sm + 0.5 * hm * k2, am, mu, om)  # slope at the midpoint using k2
        k4 = deriv(sm + hm * k3, am, mu, om)  # slope at the end using k3
        s[m] = sm + hm / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)  # weighted RK4 update
    return s  # the states at tb + dt


def g1_frequency(k) -> np.ndarray:  # function: G1 carrier frequency in Hz for channel number(s) k
    return F_G1_0 + np.asarray(k, dtype=float) * F_G1_DF  # centre plus k steps


def iono_scale(k) -> np.ndarray:  # function: Klobuchar scale to G1: the L1 delay times (f_L1 / f_G1,k)^2
    return (F_L1 / g1_frequency(k)) ** 2  # dispersive delay goes with the inverse square of the frequency


class GloOrbits:  # class: GLONASS satellite positions, velocities and clocks at transmission time
    def __init__(self, path: Path = IGS_GZ, channels_csv: Path = CHANNELS_CSV):  # constructor: reads the records and the channel table once
        self.all, self.leap = read_records(path)  # every record and the header's leap seconds
        self.healthy = self.all[self.all.health == 0].sort_values(["sv", "tb_gps"]).reset_index(drop=True)  # healthy records, in time order per satellite
        self.rec = {sv: g.reset_index(drop=True) for sv, g in self.healthy.groupby("sv")}  # dict comprehension: satellite -> its healthy records
        self.tb = {sv: g.tb_gps.to_numpy() for sv, g in self.rec.items()}  # satellite -> its tb array, for the nearest-record search
        self.igs_k = self.all.groupby("slot").k_igs.agg(lambda v: sorted(set(v)))  # slot -> the channel numbers its records carry (one expected)
        tab = pd.read_csv(channels_csv)  # the observables script channel table
        self.table_k = {int(s[1:]): int(k) for s, k in zip(tab.slot, tab.channel)}  # slot number -> channel from the table
        self.drops = {"unhealthy": 0, "no_record": 0, "gap": 0}  # counts of dropped satellite-epochs by reason
        self.drop_sv = {}  # dict: (reason, satellite) -> number of dropped satellite-epochs, to say which satellites were dropped
        self.ties = set()  # set of (satellite, epoch) pairs that sat exactly between two records
        self.k_source = {"table": 0, "igs": 0}  # counts of satellite-epochs by where the channel came from

    def channel(self, slot: int) -> int:  # method: the channel number of a slot, table first, IGS record second
        if slot in self.table_k:  # if: the table has it
            return self.table_k[slot]  # use the table
        return int(self.igs_k[slot][0])  # else the IGS record's frequency number

    def pick(self, sv: str, t_gps: float):  # method: index of the healthy record used at epoch t_gps, or None with the reason counted
        tb = self.tb.get(sv)  # this satellite's healthy tb values
        if tb is None:  # if: no healthy record at all
            same = (self.all.sv == sv).any()  # whether the slot has any record
            why = "unhealthy" if same else "no_record"  # unhealthy if records exist but none healthy, else no record
            self.drops[why] += 1  # counted
            self.drop_sv[(why, sv)] = self.drop_sv.get((why, sv), 0) + 1  # and counted per satellite
            return None  # not usable
        d = np.abs(t_gps - tb)  # time distance to every healthy record
        k = int(d.argmin())  # the nearest; argmin returns the first minimum, which is the earlier tb at a tie
        if d[k] > MAX_AGE_S:  # if: nothing within 15 min
            self.drops["gap"] += 1  # a gap in the healthy records
            self.drop_sv[("gap", sv)] = self.drop_sv.get(("gap", sv), 0) + 1  # counted per satellite
            return None  # not usable
        if (d == d[k]).sum() > 1:  # if: two records equally near (an epoch exactly on :00 or :30)
            self.ties.add((sv, t_gps))  # counted once per satellite and epoch
        return k  # the record index

    def state_full(self, svs: list, t_rx: float, pr: np.ndarray) -> tuple:  # method: positions (n x 3), velocities (n x 3), clock offsets (s) and a usable mask at transmission time
        n = len(svs)  # number of satellites
        pos, vel, dts, ok = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros(n), np.zeros(n, bool)  # results
        use, recs = [], []  # indices of usable satellites and their records
        for j, sv in enumerate(svs):  # for loop over the satellites
            if sv[0] != "R":  # if: not GLONASS
                continue  # not handled here
            k = self.pick(sv, t_rx)  # the record index or None
            if k is None:  # if: no usable record
                continue  # skip the satellite
            use.append(j)  # remember which satellite
            recs.append(self.rec[sv].iloc[k])  # and its record
        if not use:  # if: nothing usable
            return pos, vel, dts, ok  # all empty
        s0 = np.array([[r.x, r.y, r.z, r.vx, r.vy, r.vz] for r in recs])  # list comprehension: the records' states at tb
        acc = np.array([[r.ax, r.ay, r.az] for r in recs])  # their lunisolar accelerations
        tb = np.array([r.tb_gps for r in recs])  # their tb values
        clk = np.array([r.clk for r in recs])  # -TauN
        gam = np.array([r.gam for r in recs])  # GammaN
        tt = t_rx - np.asarray(pr, dtype=float)[use] / C_LIGHT  # transmission time on the satellite's clock, first guess
        d1 = clk + gam * (tt - tb)  # satellite clock offset at that time
        tt2 = tt - d1  # true transmission time, corrected by the satellite clock
        s = propagate(s0, acc, tt2 - tb)  # the PZ-90 state at the true transmission time
        d2 = clk + gam * (tt2 - tb)  # clock offset again, at the corrected time
        idx = np.array(use)  # the usable indices as an array
        pos[idx], vel[idx], dts[idx], ok[idx] = s[:, :3], s[:, 3:], d2, np.isfinite(s).all(axis=1)  # stored
        for r in recs:  # for loop over the used records
            self.k_source["table" if int(r.slot) in self.table_k else "igs"] += 1  # where each channel number came from
        return pos, vel, dts, ok  # the arrays

    def state(self, svs: list, t_rx: float, pr: np.ndarray) -> tuple:  # method: same call as the position script's Orbits.state: positions, clock offsets and a usable mask
        pos, _, dts, ok = self.state_full(svs, t_rx, pr)  # the full state without the velocities
        return pos, dts, ok  # the three arrays

    def at(self, sv: str, t_gps: float) -> tuple:  # method: Earth-fixed position and velocity of one satellite at a GPS time, no clock (used by the checks)
        k = self.pick(sv, t_gps)  # the record
        if k is None:  # if: none usable
            return None, None  # nothing
        r = self.rec[sv].iloc[k]  # the record
        s = propagate(np.array([[r.x, r.y, r.z, r.vx, r.vy, r.vz]]), np.array([[r.ax, r.ay, r.az]]), np.array([t_gps - r.tb_gps]))  # propagate it
        return s[0, :3], s[0, 3:]  # position and velocity


def path_delay(x: np.ndarray, t_gps: float, k: np.ndarray, el: np.ndarray, az: np.ndarray, kl: tuple) -> np.ndarray:  # function: troposphere plus ionosphere at G1, metres
    lat, lon, h = M.ecef_to_llh(*x)  # receiver geodetic position
    e5 = np.maximum(el, np.radians(5.0))  # elevation floored at 5 degrees against blow-up, as the position script does
    trop = M.saastamoinen(h, lat, e5)  # Saastamoinen troposphere
    iono = M.klobuchar(t_gps, lat, lon, e5, az, kl[0], kl[1])  # Klobuchar delay at L1
    return trop + iono * iono_scale(k)  # the ionosphere scaled to each satellite's own G1 frequency


def glonass_z(orb: GloOrbits, obs: pd.DataFrame, x: np.ndarray, kl: tuple, refine: bool = False) -> pd.DataFrame:  # function: per GLONASS satellite and second, measured minus modelled range (receiver clock still in), elevation, channel and relativity diagnostic
    out = []  # rows: second, satellite, z, elevation (degrees), channel, relativity term in metres
    for sec, g in obs[obs.sv.str[0] == "R"].groupby("sec", sort=True):  # for loop over the seconds, GLONASS observations only
        svs, t, pr = list(g.sv), float(g.t_gps.iloc[0]), g.pr.to_numpy()  # satellites, GPS time, pseudoranges
        pos, vel, dts, ok = orb.state_full(svs, t, pr)  # positions, velocities and clocks at transmission time
        sel = np.flatnonzero(ok)  # usable satellites
        if len(sel) == 0:  # if: none usable
            continue  # next second
        rho, _, el, az = M.geometry(x, pos[sel])  # Sagnac-corrected range, elevation, azimuth
        k = np.array([orb.channel(int(svs[i][1:])) for i in sel])  # the channel of each usable satellite
        z = pr[sel] - rho + C_LIGHT * dts[sel] - path_delay(x, t, k, el, az, kl)  # observed minus modelled, receiver clock still in
        if refine and (el >= np.radians(M.ELEV_MASK_DEG)).any():  # if: the refinement is asked for and a satellite is above the mask
            b = float(np.median(z[el >= np.radians(M.ELEV_MASK_DEG)]))  # the receiver's GLONASS clock term of this second in metres (median, as in the residual definition)
            pos, vel, dts, ok = orb.state_full(svs, t, pr - b)  # transmission time again, with the receiver clock term taken out of the travel time
            sel = np.flatnonzero(ok)  # usable satellites again
            rho, _, el, az = M.geometry(x, pos[sel])  # geometry again
            k = np.array([orb.channel(int(svs[i][1:])) for i in sel])  # channels again
            z = pr[sel] - rho + C_LIGHT * dts[sel] - path_delay(x, t, k, el, az, kl)  # the second-pass observed minus modelled
        rel = 2.0 * np.einsum("ij,ij->i", pos[sel], vel[sel]) / C_LIGHT  # periodic relativistic term 2 r.v / c in metres (diagnostic only, never applied)
        for j, i in enumerate(sel):  # for loop over the usable satellites
            out.append((int(sec), svs[i], float(z[j]), float(np.degrees(el[j])), int(k[j]), float(rel[j])))  # one row
    return pd.DataFrame(out, columns=["sec", "sv", "z", "el", "k", "rel_m"])  # the table
