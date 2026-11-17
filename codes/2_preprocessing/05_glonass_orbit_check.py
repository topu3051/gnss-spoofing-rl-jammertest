#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
the GLONASS broadcast check - validation of the GLONASS broadcast-orbit propagator (codes/2_preprocessing/04_glonass_orbit.py). Pre-registered checks V1-V5,
plus the added checks (V2b mid-point test, RK4 round trip, step halving, constants comparison,
channel cross-check, relativity diagnostic). Nothing here is a feature, a label or a model.

  V1  physics: orbit radius 25 400-25 600 km and INERTIAL speed 3.9-4.0 km/s (the broadcast velocity is Earth-fixed; the inertial speed
      adds omega x r, which is why the Earth-fixed speed alone is not tested against 3.9-4.0)
  V2  record continuity: record k propagated 30 min to tb of record k+1 against that record's position; V2b: both records propagated
      15 min to the mid-point (the distance actually used in service)
  V3  residuals on the clean reference intervals (preprocessing/reference_clean_intervals.csv), the position script site positions fixed, G1 C1C, one GLONASS clock
      term per epoch (median, n >= 4 above 10 degrees: the evidence script / definition, the elevation-weighted mean of the position script's BeiDou
      test is a sensitivity row)
  V4  median residual per frequency channel
  V5  two complete in-process computations give byte-identical output (a second run of this script is compared from outside too)

Reads   Data/processed/{reference_clean_intervals,reference_positions,session_ground_truth,glonass_channels}.csv, Data/rinex/*, the IGS file
Writes  Data/processed/preprocessing/glonass_residuals_clean.csv   one row per GLONASS satellite-second in the clean intervals (a derived product)
        Data/processed/checks/glonass_broadcast_check.csv                one row per check: check, expected, found, ok
        Data/processed/checks/glonass_broadcast_detail.csv               per receiver x satellite statistics (V3) and per receiver x channel (V4)
        Graph/glonass_residuals.png                  residual against elevation per receiver, and per-channel medians
"""

from __future__ import annotations  # future import: type hints kept as text (a convention across the project's scripts)

import hashlib  # standard-library hashing, for the SHA-256 comparisons
import importlib.util  # standard-library module loader, for modules whose names start with a digit
import os  # operating-system access: MPLCONFIGDIR is set before matplotlib loads
import random  # standard-library random; seeded below although nothing here draws from it
import sys  # standard-library system access: sys.path
from pathlib import Path  # Path objects for building file paths

PROJECT = Path(__file__).resolve().parent.parent.parent  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
os.environ["MPLCONFIGDIR"] = str(PROJECT / "codes" / ".mplconfig")  # matplotlib's cache inside the project folder

import numpy as np  # NumPy: arrays and percentiles
import pandas as pd  # pandas: tables
import matplotlib  # plotting library
matplotlib.use("Agg")  # file-only backend, no window
import matplotlib.pyplot as plt  # the pyplot interface

sys.path.insert(0, str(PROJECT / "codes" / "common"))  # puts codes/ first on the import path so the helpers module is found
from helpers import save_fig  # byte-stable figure saving

SEED = 20260914  # integer constant: the project's fixed random seed
random.seed(SEED)  # seeds Python's generator (nothing here draws from it)
np.random.seed(SEED)  # seeds NumPy's generator (nothing here draws from it)

PROC = PROJECT / "Data" / "processed"  # where the CSVs are read and written
CHECK_DIR = PROJECT / "Data" / "processed" / "checks"  # where the check tables go
GRAPH = PROJECT / "Graph"  # where the figure goes


def load(name: str, filename: str):  # function: loads codes/<filename> as a module (names start with a digit)
    spec = importlib.util.spec_from_file_location(name, PROJECT / "codes" / filename)  # where to load from
    mod = importlib.util.module_from_spec(spec)  # an empty module object
    sys.modules[name] = mod  # registered
    spec.loader.exec_module(mod)  # runs the top level (definitions only)
    return mod  # the module


G = load("glonass_orbit", "2_preprocessing/04_glonass_orbit.py")  # the propagator under test
E = load("observables_for_glonass_check", "4_features/01_observables.py")  # 05e: the observation reader (all four constellations) and, through it, the evidence script and the position script
D, M = E.D, E.M  # the evidence script (clean spans, site positions, residual definition) and the position script (orbits, geometry)
MASK = D.MASK_DEG  # the 10 degree elevation mask
C = G.C_LIGHT  # speed of light, m/s
ANOM_M = 1.0e5  # float constant: a residual beyond 100 km is a pseudorange error of whole milliseconds (1 ms = 299.8 km), not an orbit effect
NUM: dict = {}  # dict: every number the report quotes, written to glonass_broadcast_summary.csv
CHK: list = []  # list: the checks, written to glonass_broadcast_check.csv
COLORS = M.COLORS  # Okabe-Ito colours of the three receivers
RXN = {"SBF_02": "mosaic-X5", "SBF_03": "mosaic-H main", "UBX_01": "u-blox 01"}  # receiver names for the figure


def add(check: str, expected: str, found, ok: bool | None) -> None:  # function: records one check and prints it
    CHK.append({"check": check, "expected": expected, "found": found, "ok": "" if ok is None else bool(ok)})  # append a dict row
    print(f"  [{'ok' if ok else ('FAIL' if ok is False else '--')}] {check}: expected {expected}; found {found}")  # progress line


def pct(a, q) -> float:  # function: percentile of an array, NaN if empty
    return float(np.percentile(a, q)) if len(a) else float("nan")  # NumPy percentile


def v1_v2(orb) -> None:  # function: V1 (physics over the day) and V2/V2b (record continuity) and the integrator self-checks
    rec = orb.healthy  # the healthy records
    # V1 sweep: every healthy slot, every 60 s of the UTC day, nearest healthy record
    t0 = (pd.Timestamp("2026-09-14", tz="UTC") - pd.Timestamp(G.GPS_EPOCH)).total_seconds() + G.LEAP_SECONDS  # the day's start as GPS seconds
    rad, spd, spd_ef = [], [], []  # radius, inertial speed and Earth-fixed speed (km, km/s)
    om = np.array([0.0, 0.0, G.OMEGA])  # Earth rotation vector, rad/s
    for sv in sorted(orb.rec):  # for loop over the healthy satellites
        for t in np.arange(t0, t0 + 86400.0, 60.0):  # for loop over the day at 60 s
            p, v = orb.at(sv, t)  # position and Earth-fixed velocity
            if p is None:  # if: no usable record
                continue  # skip
            rad.append(np.linalg.norm(p) / 1e3)  # radius in km
            spd.append(np.linalg.norm(v + np.cross(om, p)) / 1e3)  # inertial speed: Earth-fixed velocity plus omega x r, km/s
            spd_ef.append(np.linalg.norm(v) / 1e3)  # the Earth-fixed speed, reported only
    rad, spd, spd_ef = np.array(rad), np.array(spd), np.array(spd_ef)  # arrays
    add("V1 radius, day sweep (60 s, every healthy slot)", "25400-25600 km", f"{rad.min():.1f}-{rad.max():.1f} km, {len(rad)} epochs", rad.min() >= 25400 and rad.max() <= 25600)  # the radius test
    add("V1 inertial speed, day sweep", "3.9-4.0 km/s", f"{spd.min():.4f}-{spd.max():.4f} km/s", spd.min() >= 3.9 and spd.max() <= 4.0)  # the speed test
    add("V1 Earth-fixed speed, day sweep (not tested; the broadcast frame rotates)", "informational", f"{spd_ef.min():.3f}-{spd_ef.max():.3f} km/s", None)  # reported only
    # V2: consecutive healthy records, 30 min apart
    rows = []  # per satellite: errors of V2 and V2b
    for sv, g in orb.rec.items():  # for loop over the satellites
        g = g.reset_index(drop=True)  # a clean index
        e30, e15 = [], []  # the 30 min and mid-point errors
        for i in range(len(g) - 1):  # for loop over consecutive record pairs
            a, b = g.iloc[i], g.iloc[i + 1]  # the two records
            if abs(b.tb_gps - a.tb_gps - 1800.0) > 1.0:  # if: not 30 min apart (a gap from an unhealthy record)
                continue  # not a continuity pair
            sa = np.array([[a.x, a.y, a.z, a.vx, a.vy, a.vz]])  # record k state
            ac = np.array([[a.ax, a.ay, a.az]])  # its acceleration
            fwd = G.propagate(sa, ac, np.array([1800.0]))[0, :3]  # record k propagated to tb of k+1
            e30.append(np.linalg.norm(fwd - np.array([b.x, b.y, b.z])))  # its distance from record k+1's own position
            sb = np.array([[b.x, b.y, b.z, b.vx, b.vy, b.vz]])  # record k+1 state
            bc = np.array([[b.ax, b.ay, b.az]])  # its acceleration
            mid_a = G.propagate(sa, ac, np.array([900.0]))[0, :3]  # record k forward 15 min
            mid_b = G.propagate(sb, bc, np.array([-900.0]))[0, :3]  # record k+1 backward 15 min
            e15.append(np.linalg.norm(mid_a - mid_b))  # the two mid-point positions' distance
        rows.append((sv, len(e30), pct(e30, 50), pct(e30, 95), pct(e15, 50), pct(e15, 95)))  # one satellite's summary
    v2 = pd.DataFrame(rows, columns=["sv", "n_pairs", "v2_median_m", "v2_p95_m", "v2b_median_m", "v2b_p95_m"])  # the table
    v2.to_csv(CHECK_DIR / "glonass_broadcast_per_satellite.csv", index=False, float_format="%.3f")  # kept for the reader
    # pooled and worst satellites
    allp = []  # every pair error, pooled
    allm = []  # every mid-point error, pooled
    for sv, g in orb.rec.items():  # for loop over the satellites again, collecting all pair errors
        for i in range(len(g) - 1):  # for loop over consecutive pairs
            a, b = g.iloc[i], g.iloc[i + 1]  # the two records
            if abs(b.tb_gps - a.tb_gps - 1800.0) > 1.0:  # if: not 30 min apart
                continue  # skip
            sa = np.array([[a.x, a.y, a.z, a.vx, a.vy, a.vz]])  # record k state
            ac = np.array([[a.ax, a.ay, a.az]])  # its acceleration
            allp.append(np.linalg.norm(G.propagate(sa, ac, np.array([1800.0]))[0, :3] - np.array([b.x, b.y, b.z])))  # pair error
            sb = np.array([[b.x, b.y, b.z, b.vx, b.vy, b.vz]])  # record k+1 state
            bc = np.array([[b.ax, b.ay, b.az]])  # its acceleration
            allm.append(np.linalg.norm(G.propagate(sa, ac, np.array([900.0]))[0, :3] - G.propagate(sb, bc, np.array([-900.0]))[0, :3]))  # mid-point error
    w = v2.sort_values("v2_p95_m").iloc[-1]  # the satellite with the largest 95th percentile
    add("V2 record continuity, 30 min, pooled (median / 95th pct)", "metres to a few tens of metres (reported, not tuned)", f"{pct(allp, 50):.2f} / {pct(allp, 95):.2f} m over {len(allp)} pairs; worst satellite {w.sv} p95 {w.v2_p95_m:.1f} m; per-satellite medians {v2.v2_median_m.min():.2f}-{v2.v2_median_m.max():.2f} m", None)  # reported
    add("V2b mid-point, 15 min each way, pooled (median / 95th pct)", "smaller than V2 (reported)", f"{pct(allm, 50):.2f} / {pct(allm, 95):.2f} m; worst satellite {v2.sort_values('v2b_p95_m').iloc[-1].sv} p95 {v2.v2b_p95_m.max():.1f} m", None)  # reported
    add("records: all / healthy / unhealthy, satellites with any / healthy", "1224 / 1078 / 146 (R21 49, R26 48, R27 48, R23 1)", f"{len(orb.all)} / {len(rec)} / {len(orb.all) - len(rec)}, {orb.all.sv.nunique()} / {len(orb.rec)}", len(orb.all) == 1224)  # record counts reconcile
    gaps = {sv: int((np.diff(g.tb_gps.to_numpy()) > 1801).sum()) for sv, g in orb.rec.items()}  # healthy-record gaps over 30 min per satellite
    add("healthy-record gaps over 30 min (any satellite)", "none expected", f"{sum(gaps.values())}", sum(gaps.values()) == 0)  # the gap count
    # integrator self-checks on every healthy record
    s0 = rec[["x", "y", "z", "vx", "vy", "vz"]].to_numpy()  # all healthy states at tb
    ac = rec[["ax", "ay", "az"]].to_numpy()  # their accelerations
    out = G.propagate(s0, ac, np.full(len(rec), 900.0))  # forward 15 min
    back = G.propagate(out, ac, np.full(len(rec), -900.0))  # and back
    rt = np.linalg.norm(back[:, :3] - s0[:, :3], axis=1)  # round-trip position error
    add("RK4 round trip, +15 min then -15 min, all healthy records", "< 1 mm", f"max {rt.max() * 1e3:.4f} mm", rt.max() < 1e-3)  # the round trip
    f30 = G.propagate(s0, ac, np.full(len(rec), 900.0), step=30.0)  # 30 s steps
    f15 = G.propagate(s0, ac, np.full(len(rec), 900.0), step=15.0)  # 15 s steps
    sh = np.linalg.norm(f30[:, :3] - f15[:, :3], axis=1)  # their difference
    add("RK4 step halving, 30 s against 15 s over 15 min", "< 1 mm", f"max {sh.max() * 1e3:.4f} mm", sh.max() < 1e-3)  # step halving
    for dt in (900.0, 1800.0):  # for loop over the two spans
        o1 = G.propagate(s0, ac, np.full(len(rec), dt))  # with the ICD 5.1 constants
        o2 = G.propagate(s0, ac, np.full(len(rec), dt), mu=G.MU_OLD, om=G.OMEGA_OLD)  # with the rounded values first used
        dd = np.linalg.norm(o1[:, :3] - o2[:, :3], axis=1)  # position change
        add(f"constants: ICD 5.1 (mu 398600.4418e9, omega 7.2921151467e-5) against the rounded values first used, {dt / 60:.0f} min", "< 1 mm", f"max {dd.max() * 1e3:.3f} mm, median {np.median(dd) * 1e3:.3f} mm", dd.max() < 1e-3)  # the change
    # channel cross-check
    shared = sorted(set(orb.table_k) & set(orb.igs_k.index))  # slots in both sources
    multi = [s for s in orb.igs_k.index if len(orb.igs_k[s]) != 1]  # slots whose IGS records carry more than one number
    dis = [(s, orb.table_k[s], orb.igs_k[s][0]) for s in shared if orb.igs_k[s][0] != orb.table_k[s]]  # slots where the two disagree
    only_igs = sorted(set(orb.igs_k.index) - set(orb.table_k))  # slots only the IGS file has
    only_tab = sorted(set(orb.table_k) - set(orb.igs_k.index))  # slots only the table has
    add("channel numbers: slots in both sources, agreeing, disagreeing", "all shared slots agree", f"{len(shared)}, {len(shared) - len(dis)}, {len(dis)} {dis}; IGS slots with more than one number: {multi}", len(dis) == 0 and not multi)  # the cross-check
    add("channel numbers: slots only in the IGS file / only in the table", "informational", f"{['R%02d=%d' % (s, orb.igs_k[s][0]) for s in only_igs]} / {['R%02d' % s for s in only_tab]}", None)  # the single-source slots
    NUM.update({"v1_rad_min": rad.min(), "v1_rad_max": rad.max(), "v1_n": len(rad), "v1_spd_min": spd.min(), "v1_spd_max": spd.max(), "v1_spdef_min": spd_ef.min(), "v1_spdef_max": spd_ef.max(),  # V1 numbers
                "v2_med": pct(allp, 50), "v2_p95": pct(allp, 95), "v2_n": len(allp), "v2_worst_sv": w.sv, "v2_worst_p95": w.v2_p95_m, "v2_satmed_min": v2.v2_median_m.min(), "v2_satmed_max": v2.v2_median_m.max(),  # V2 numbers
                "v2b_med": pct(allm, 50), "v2b_p95": pct(allm, 95), "v2b_worst_sv": v2.sort_values("v2b_p95_m").iloc[-1].sv, "v2b_worst_p95": v2.v2b_p95_m.max(),  # V2b numbers
                "rec_all": len(orb.all), "rec_healthy": len(rec), "rec_unhealthy": len(orb.all) - len(rec), "sv_all": orb.all.sv.nunique(), "sv_healthy": len(orb.rec), "gaps30": sum(gaps.values()),  # record counts
                "rt_max_mm": rt.max() * 1e3, "sh_max_mm": sh.max() * 1e3,  # integrator checks
                "ch_shared": len(shared), "ch_agree": len(shared) - len(dis), "ch_igs_only": "; ".join(f"R{s_:02d}={orb.igs_k[s_][0]}" for s_ in only_igs), "ch_tab_only": len(only_tab)})  # channel cross-check
    for dt in (900.0, 1800.0):  # for loop over the two spans of the constants comparison
        o1 = G.propagate(s0, ac, np.full(len(rec), dt))  # with the ICD 5.1 constants
        o2 = G.propagate(s0, ac, np.full(len(rec), dt), mu=G.MU_OLD, om=G.OMEGA_OLD)  # with the rounded values first used
        d_ = np.linalg.norm(o1[:, :3] - o2[:, :3], axis=1)  # position change
        NUM[f"const_max_mm_{int(dt / 60)}"], NUM[f"const_med_mm_{int(dt / 60)}"] = d_.max() * 1e3, np.median(d_) * 1e3  # kept
    sr = orb.all.groupby("sv").agg(n=("tb_gps", "size"), healthy=("health", lambda v: int((v == 0).sum())))  # records per satellite and how many are healthy
    NUM["rec_r21"], NUM["rec_r26"], NUM["rec_r27"], NUM["rec_r23"] = int(sr.loc["R21", "n"]), int(sr.loc["R26", "n"]), int(sr.loc["R27", "n"]), int(sr.loc["R23", "n"])  # the unhealthy slots' record counts
    if dis:  # if: any disagreement
        raise SystemExit(f"STOP: channel table and IGS frequency numbers disagree: {dis}")  # the specification says stop and report


def v3_run(orb, kl, refine: bool = False) -> tuple:  # function: the whole V3 computation: residuals on the clean intervals for every receiver; returns the residual table and the diagnostics
    gt = pd.read_csv(PROC / "ground_truth" / "session_ground_truth.csv").set_index("test_code")  # session folders
    out, sizes = [], []  # residual tables; (session, receiver, minutes, GLONASS rows)
    for test in D.SESSIONS:  # for loop over the sessions in time order
        site = D.site_xyz(M.SITE_OF[test])  # the session's site reference (with its height)
        for rx in D.RXS:  # for loop over the receivers
            spans = D.clean_spans(test, rx)  # the clean intervals of the session and receiver
            if not spans:  # if: none of positive length
                continue  # nothing to read
            path = M.obs_path(gt.loc[test, "session_folder"], test, rx)  # the RINEX file (2.6.12's u-blox 01 is piece 1, as in the position script)
            obs = E.read_all(path, spans, rx)  # all four constellations at 1 Hz inside the spans
            D.LINES.clear()  # frees the decompressed file
            z = G.glonass_z(orb, obs, site, kl, refine)  # GLONASS measured minus modelled range, clock still in (refine: transmission time without the receiver clock term)
            sizes.append((test, rx, sum(b - a for a, b in spans) / 60.0, int((obs.sv.str[0] == "R").sum()), int(len(z))))  # minutes, raw GLONASS rows, modelled rows
            if len(z) == 0:  # if: no GLONASS satellite could be modelled
                continue  # nothing to add
            z["sys"] = "R"  # the constellation letter the evidence script's residual function groups by
            rs = D.resid_mask(z[["sec", "sv", "z", "el", "sys"]], MASK)  # median-clock residuals above the mask, n >= 4 (the evidence script definition)
            bad = rs[np.abs(rs.res) > ANOM_M][["sec", "sv"]].assign(bad=1)  # satellite-seconds whose residual is beyond 100 km: a pseudorange error of whole milliseconds, not an orbit effect
            zz = z[z.el >= MASK].merge(bad, on=["sec", "sv"], how="left")  # the satellites above the mask with the anomaly flag attached
            zz = zz[zz.bad.isna()].drop(columns="bad").copy()  # the weighted-mean sensitivity is computed without them (one such value would move the whole epoch's mean)
            zz["n_w"] = zz.groupby("sec").z.transform("count")  # satellites per second
            zz = zz[zz.n_w >= 4].copy()  # at least four, as for the median
            zz["w"] = np.sin(np.radians(zz.el)) ** 2  # elevation weights as in the position script's BeiDou test
            zz["clk_w"] = (zz.w * zz.z).groupby(zz.sec).transform("sum") / zz.w.groupby(zz.sec).transform("sum")  # weighted-mean clock per second
            zz["res_wmean_m"] = zz.z - zz.clk_w  # residual about the weighted-mean clock
            m = rs.merge(z[["sec", "sv", "z", "k", "rel_m"]], on=["sec", "sv"], how="left").merge(zz[["sec", "sv", "n_w", "res_wmean_m"]], on=["sec", "sv"], how="left")  # attach z, channel, relativity (from z) and the weighted-mean residual and satellite count (from zz)
            m["anomaly"] = (np.abs(m.res) > ANOM_M).astype(int)  # 1 for the milliseconds-level pseudorange anomalies
            m.insert(0, "receiver", rx)  # receiver column
            m.insert(0, "test_code", test)  # session column
            out.append(m)  # kept
    res = pd.concat(out, ignore_index=True).rename(columns={"res": "res_median_m", "el": "el_deg", "z": "z_clock_in_m", "n_w": "n_wmean"})  # all rows
    res = res[["test_code", "receiver", "sec", "sv", "k", "el_deg", "z_clock_in_m", "res_median_m", "res_wmean_m", "n_wmean", "anomaly", "rel_m"]]  # column order
    return res, pd.DataFrame(sizes, columns=["test_code", "receiver", "minutes", "glonass_rows", "modelled_rows"])  # the table and the sizes


def csv_bytes(df: pd.DataFrame) -> bytes:  # function: the table as CSV bytes with fixed rounding, for byte comparison
    return df.round({"el_deg": 4, "z_clock_in_m": 4, "res_median_m": 4, "res_wmean_m": 4, "rel_m": 4}).to_csv(index=False).encode()  # rounded, fixed format


def passes(df: pd.DataFrame) -> int:  # function: the number of passes: runs of seconds with gaps of at most 5 s, per session and receiver
    s = np.sort(df.sec.to_numpy())  # the seconds in order
    return int(1 + (np.diff(s) > 5).sum()) if len(s) else 0  # one run plus one per gap


def main() -> None:  # function: runs every check and writes the outputs
    print("the GLONASS broadcast check - GLONASS propagator checks")  # heading
    sha = hashlib.sha256(G.IGS_GZ.read_bytes()).hexdigest()  # the IGS file's SHA-256
    add("IGS file SHA-256 equals the logged one (1)", D.S04.IGS_SHA256[:12], sha[:12], sha == D.S04.IGS_SHA256)  # same file as every earlier section
    orb = G.GloOrbits()  # the propagator
    add("header LEAP SECONDS equals the project's LEAP_SECONDS", str(G.LEAP_SECONDS), str(orb.leap), orb.leap == G.LEAP_SECONDS)  # the time conversion
    v1_v2(orb)  # V1, V2, V2b, integrator checks, constants, channels
    kl = M.igs_klobuchar(D.S04.IGS_BRDC)  # the GPS Klobuchar parameters
    orb = G.GloOrbits()  # fresh counters for V3
    res, sizes = v3_run(orb, kl)  # the V3 computation
    res2, _ = v3_run(G.GloOrbits(), kl)  # a second complete computation
    same = csv_bytes(res) == csv_bytes(res2)  # byte comparison
    add("V5 two in-process computations, byte-identical residual table", "identical", f"{len(res)} rows, SHA-256 {hashlib.sha256(csv_bytes(res)).hexdigest()[:12]}", same)  # reproducibility
    print(sizes.to_string(index=False))  # the sizes
    # counts and reconciliation
    add("V3 clean minutes read / GLONASS rows in them / rows with an orbit / residual rows (n>=4, >=10 deg)", "rows with orbit <= rows read; residuals <= rows with orbit", f"{sizes.minutes.sum():.2f} min / {sizes.glonass_rows.sum()} / {sizes.modelled_rows.sum()} / {len(res)}", sizes.modelled_rows.sum() <= sizes.glonass_rows.sum() and len(res) <= sizes.modelled_rows.sum())  # funnel counts
    add("drops among GLONASS satellite-epochs read (orbit not usable)", "unhealthy / no_record / gap, reported", f"{orb.drops}", None)  # drop counts; these map to no_orbit_record
    add("tie epochs (exactly :00 or :30, equally near two records) in the clean intervals", "reported; ties occur at every half hour", f"{len(orb.ties)} satellite-epochs", None)  # ties in V3
    g1 = orb.rec["R01"].tb_gps.to_numpy()  # R01's healthy tb values, as the example satellite
    mid = (g1[:-1] + g1[1:]) / 2.0  # midpoints between consecutive records (all on :00 and :30)
    day0 = (pd.Timestamp("2026-09-14", tz="UTC") - pd.Timestamp(G.GPS_EPOCH)).total_seconds() + G.LEAP_SECONDS  # the UTC day's start as GPS seconds
    add("tie epochs per UTC day per healthy satellite (record midpoints inside the day)", "48 (every :00 and :30)", f"{int(((mid >= day0) & (mid < day0 + 86400.0)).sum())}", int(((mid >= day0) & (mid < day0 + 86400.0)).sum()) == 48)  # the analytic count
    add("channel source in V3 (satellite-epochs from the table / from the IGS record)", "no satellite dropped for a missing channel", f"{orb.k_source}", None)  # channel provenance
    # relativity diagnostic
    rel = np.abs(res.rel_m.dropna().to_numpy())  # |2 r.v / c| in metres
    add("relativity diagnostic |2 r.v/c| over the V3 residual rows (not applied)", "decimetres possible", f"median {np.median(rel):.3f} m, max {rel.max():.3f} m", None)  # reported only
    # V3 statistics
    an = res[res.anomaly == 1]  # the milliseconds-level pseudorange anomalies
    add("V3 pseudorange anomalies (|residual| > 100 km): rows, by receiver / satellite / session, residual + c x 10 ms", "reported; kept in the table, flagged", "; ".join(f"{k}: {len(g)} rows, residual {g.res_median_m.median() + 0.01 * C:+.1f} m after adding 10 ms" for k, g in an.groupby(["receiver", "sv", "test_code"])) or "none", None)  # what and where
    ok = res[res.anomaly == 0]  # the rows without the anomalies
    a = np.abs(res.res_median_m.to_numpy())  # |median-clock residual|, all rows (as pre-registered)
    b = np.abs(ok.res_median_m.to_numpy())  # |residual| without the anomalies
    aw = np.abs(ok.res_wmean_m.dropna().to_numpy())  # |weighted-mean residual| without the anomaly seconds
    add("V3 pooled |residual|, median clock, all rows: median", "< 1.5 m", f"{np.median(a):.3f} m over {len(a)}", np.median(a) < 1.5)  # pass criterion 1
    add("V3 pooled |residual|, median clock, all rows: 95th percentile", "< 5 m", f"{pct(a, 95):.3f} m", pct(a, 95) < 5.0)  # pass criterion 2
    add("V3 pooled |residual|, median clock, anomalies removed: median / 95th percentile", "< 1.5 m / < 5 m", f"{np.median(b):.3f} / {pct(b, 95):.3f} m over {len(b)}", bool(np.median(b) < 1.5 and pct(b, 95) < 5.0))  # both criteria without the anomalies
    add("V3 sensitivity: weighted-mean clock (anomaly seconds out), median / 95th percentile", "reported", f"{np.median(aw):.3f} / {pct(aw, 95):.3f} m over {len(aw)}", None)  # the sensitivity row
    add("V3 residual signed median and mean (median clock, anomalies removed)", "near 0", f"{np.median(ok.res_median_m):.3f} / {ok.res_median_m.mean():.3f} m", None)  # signed
    rf, _ = v3_run(G.GloOrbits(), kl, refine=True)  # sensitivity: the same computation with the receiver clock term taken out of the transmission time
    rf = rf[rf.anomaly == 0]  # anomalies out as above
    for rx in [None] + list(D.RXS):  # for loop over all receivers pooled, then each
        q = rf if rx is None else rf[rf.receiver == rx]  # the rows
        q0 = ok if rx is None else ok[ok.receiver == rx]  # the primary rows for comparison
        add(f"V3 sensitivity, transmission time without the receiver clock term, {'all receivers' if rx is None else RXN[rx]}: |residual| median / 95th percentile (primary in brackets)", "reported; physical refinement, not tuned to the residual", f"{np.median(np.abs(q.res_median_m)):.3f} / {pct(np.abs(q.res_median_m), 95):.3f} m ({np.median(np.abs(q0.res_median_m)):.3f} / {pct(np.abs(q0.res_median_m), 95):.3f}), n {len(q)}", None)  # the sensitivity rows
    # sensitivity: the periodic relativistic term added to the satellite clock (reported, not adopted, the GLONASS orbit script is unchanged)
    # sign convention: rel_m = +2 r.v / c in metres; the clock correction is -2 r.v / c^2 seconds (GPS: F e sqrt(A) sin E with F = -2 sqrt(mu)/c^2), i.e. -rel_m in metres, so the clock-in value z becomes z - rel_m
    rr = res.copy()  # all rows including the anomalies (the epoch median must see the same satellites as the primary)
    rr["z_rel"] = rr.z_clock_in_m - rr.rel_m  # clock-in value with the periodic term added to the satellite clock
    rr["z_opp"] = rr.z_clock_in_m + rr.rel_m  # the opposite sign, kept only to show that the sign matters
    for col, name in (("z_rel", "added as -2 r.v/c^2 (the GPS convention)"), ("z_opp", "opposite sign, diagnostic only")):  # for loop over the two signs
        rr["r_" + col] = rr[col] - rr.groupby(["test_code", "receiver", "sec"])[col].transform("median")  # the clock re-estimated per epoch as the median, as in the primary definition
        q = rr[rr.anomaly == 0]  # without the anomalies
        add(f"V3 sensitivity, relativistic term {name}, clock re-estimated per epoch: |residual| median / 95th percentile (primary in brackets)", "reported, not adopted", f"{np.median(np.abs(q['r_' + col])):.3f} / {pct(np.abs(q['r_' + col]), 95):.3f} m ({np.median(b):.3f} / {pct(b, 95):.3f}), n {len(q)}", None)  # the row
    add("V3 correlation between the primary residual and rel_m (anomalies removed)", "reported; negative means the term has the right sign", f"{np.corrcoef(ok.res_median_m, ok.rel_m)[0, 1]:+.3f} over {len(ok)}", None)  # the correlation
    add("V3 rel_m sign convention", "documented", "rel_m = +2 r.v/c in metres; the correction applied in the sensitivity is -rel_m on the clock-in value (clock term -2 r.v/c^2 s)", None)  # the convention
    # test of the R19 explanation: a 10 ms pseudorange error also moves the transmission time by 10 ms
    # a pseudorange short by 10 ms puts the model satellite 10 ms later on its orbit: the residual left after adding c x 10 ms is -(range rate) x 0.01 s
    r19 = res[(res.sv == "R19") & (res.anomaly == 1)]  # the anomaly rows of R19
    gt3 = pd.read_csv(PROC / "ground_truth" / "session_ground_truth.csv").set_index("test_code")  # session folders
    o19 = G.GloOrbits()  # a fresh propagator, so the counters of the main run (drops, ties, channel source) are not touched by this test
    rate_all, left_all = [], []  # range rates and the residual left after adding c x 10 ms
    for (test_, rx_), g in r19.groupby(["test_code", "receiver"]):  # for loop over the session and receiver pairs that carry the anomaly
        ob = E.read_all(M.obs_path(gt3.loc[test_, "session_folder"], test_, rx_), D.clean_spans(test_, rx_), rx_)  # the observations again
        D.LINES.clear()  # frees the file
        ob = ob[(ob.sv == "R19") & ob.sec.isin(g.sec)].drop_duplicates("sec").set_index("sec")  # R19's rows of the anomalous seconds
        site_ = D.site_xyz(M.SITE_OF[test_])  # the site position
        for sec_, row in ob.iterrows():  # for loop over the seconds
            t_, pr_ = float(row.t_gps), np.array([row.pr])  # GPS time and pseudorange
            pa, _, da, _ = o19.state_full(["R19"], t_ + 0.5, pr_)  # position half a second later
            pb, _, db, _ = o19.state_full(["R19"], t_ - 0.5, pr_)  # and half a second earlier
            ra = M.geometry(site_, pa)[0][0] - C * da[0]  # clock-corrected effective range after
            rb = M.geometry(site_, pb)[0][0] - C * db[0]  # and before
            rate_all.append((ra - rb, rx_))  # range rate in m/s (positive when the satellite recedes)
            left_all.append(float(g[g.sec == sec_].res_median_m.iloc[0]) + C * 0.01)  # residual left after adding c x 10 ms
    rate = np.array([r_[0] for r_ in rate_all])  # the range rates
    left = np.array(left_all)  # the residuals left
    pred = -rate * 0.01  # predicted leftover
    add("R19 anomaly: range rate over the anomalous seconds (m/s), mean (min to max)", "reported", f"{rate.mean():.1f} ({rate.min():.1f} to {rate.max():.1f}), {len(rate)} seconds", None)  # the range rate
    add("R19 anomaly: residual left after adding c x 10 ms, observed against -(range rate) x 0.01 s", "match if the 10 ms explanation holds", f"observed mean {left.mean():.2f} m ({left.min():.2f} to {left.max():.2f}), predicted mean {pred.mean():.2f} m ({pred.min():.2f} to {pred.max():.2f}), rms difference {np.sqrt(np.mean((left - pred) ** 2)):.2f} m", bool(np.sqrt(np.mean((left - pred) ** 2)) < 1.0))  # the comparison
    # structure of the residual: per satellite and session offsets common to the receivers
    t = ok.groupby(["sv", "test_code", "receiver"]).res_median_m.median().reset_index()  # median residual per satellite, session and receiver
    p = t.pivot_table(index=["sv", "test_code"], columns="receiver", values="res_median_m")  # one row per satellite and session, one column per receiver
    for a_, b_ in (("SBF_02", "SBF_03"), ("SBF_02", "UBX_01"), ("SBF_03", "UBX_01")):  # for loop over the receiver pairs
        q = p[[a_, b_]].dropna()  # satellite-sessions both saw
        add(f"V3 offset structure: satellite-session medians, {RXN[a_]} against {RXN[b_]}", "reported", f"{len(q)} satellite-sessions, correlation {q.corr().iloc[0, 1]:.2f}, rms difference {np.sqrt(((q[a_] - q[b_]) ** 2).mean()):.2f} m, rms level {np.sqrt((q[a_] ** 2).mean()):.2f} m", None)  # how common the offsets are
    ok2 = ok.copy()  # a copy to add columns to
    ok2["dm"] = ok2.res_median_m - ok2.groupby(["sv", "receiver", "test_code"]).res_median_m.transform("median")  # residual with each satellite-session-receiver median removed
    add("V3 residual after removing each satellite-session-receiver median: |median| / 95th percentile", "the noise left when the offsets are removed (reported)", f"{np.median(np.abs(ok2.dm)):.3f} / {pct(np.abs(ok2.dm), 95):.3f} m", None)  # the within-pass scatter
    xs = ok[ok.receiver == "SBF_02"].groupby(["sv", "test_code"]).res_median_m.median().unstack()  # X5 median per satellite and session
    rng = (xs.max(axis=1) - xs.min(axis=1)).dropna()  # largest change of a satellite's median between sessions
    add("V3 offset structure: change of one satellite's median between sessions (mosaic-X5), satellites with two or more sessions", "a constant inter-frequency bias would be zero", f"median {rng.median():.2f} m, max {rng.max():.2f} m ({rng.idxmax()}), {len(rng)} satellites", None)  # not constant in time
    # clock continuity of the broadcast records
    cj = []  # clock jumps in metres
    for sv, g in orb.rec.items():  # for loop over the satellites
        g = g.reset_index(drop=True)  # clean index
        for i in range(len(g) - 1):  # for loop over consecutive records
            a1, b1 = g.iloc[i], g.iloc[i + 1]  # the two records
            if abs(b1.tb_gps - a1.tb_gps - 1800.0) <= 1.0:  # if: 30 min apart
                cj.append(abs((a1.clk + a1.gam * 1800.0 - b1.clk) * C))  # clock of record k carried to tb of k+1, minus record k+1's clock, metres
    add("V2c broadcast clock continuity, record k carried 30 min against record k+1, |difference|", "reported (a clock-update test, not an absolute accuracy test)", f"median {np.median(cj):.2f} m, 95th {pct(cj, 95):.2f} m, max {max(cj):.2f} m over {len(cj)} pairs", None)  # the clock jumps
    # context: the other constellations in the same clean intervals (the observables script table, read only)
    ob = pd.read_csv(PROC / "features" / "observables_1hz.csv.gz", usecols=["test_code", "receiver", "sec", "sys", "pr_res_m"])  # the existing per-satellite table
    ci = pd.read_csv(PROC / "preprocessing" / "reference_clean_intervals.csv")  # the clean intervals
    ci = ci[ci.minutes > 0]  # those with positive length
    msk = np.zeros(len(ob), bool)  # rows inside a clean interval
    for r in ci.itertuples():  # for loop over the intervals
        msk |= ((ob.test_code == r.test_code) & (ob.receiver == r.receiver) & (ob.sec >= D.T(r.start_utc[11:])) & (ob.sec < D.T(r.end_utc[11:]) + 1)).to_numpy()  # inside this interval
    ctx = ob[msk & ob.pr_res_m.notna()]  # the residuals of the other constellations there
    for sy, g in ctx.groupby("sys"):  # for loop over the constellations
        add(f"context: {sy} residuals in the same clean intervals (the observables script table), |median| / 95th percentile", "reported", f"{g.pr_res_m.abs().median():.2f} / {pct(g.pr_res_m.abs(), 95):.2f} m over {len(g)}", None)  # the comparison rows
    det = []  # detail rows
    for rx in D.RXS:  # for loop over the receivers
        r = ok[ok.receiver == rx]  # that receiver's rows without the anomalies
        ar = np.abs(r.res_median_m.to_numpy())  # its absolute residuals
        add(f"V3 {RXN[rx]}: |residual| median / 95th percentile, residuals, satellites, epochs", "reported", f"{pct(ar, 50):.3f} / {pct(ar, 95):.3f} m, {len(r)}, {r.sv.nunique()}, {r.sec.nunique()}", None)  # per receiver
        for sv, g in r.groupby("sv"):  # for loop over the satellites of this receiver
            npass = sum(passes(h) for _, h in g.groupby("test_code"))  # passes: runs per session
            det.append(("V3", rx, sv, "", len(g), npass, len(g) / 60.0, pct(np.abs(g.res_median_m), 50), pct(np.abs(g.res_median_m), 95), float(g.el_deg.median()), float(g.el_deg.min()), float(g.el_deg.max()), float(g.res_median_m.median())))  # one row
        for k, g in r.groupby("k"):  # for loop over the channels of this receiver
            det.append(("V4", rx, "", int(k), len(g), sum(passes(h) for _, h in g.groupby(["test_code", "sv"])), len(g) / 60.0, pct(np.abs(g.res_median_m), 50), pct(np.abs(g.res_median_m), 95), float(g.el_deg.median()), float(g.el_deg.min()), float(g.el_deg.max()), float(g.res_median_m.median())))  # one row
    for sv, g in ok.groupby("sv"):  # for loop over the satellites, all receivers pooled (anomalies removed)
        det.append(("V3", "ALL", sv, "", len(g), sum(passes(h) for _, h in g.groupby(["test_code", "receiver"])), len(g) / 60.0, pct(np.abs(g.res_median_m), 50), pct(np.abs(g.res_median_m), 95), float(g.el_deg.median()), float(g.el_deg.min()), float(g.el_deg.max()), float(g.res_median_m.median())))  # one row
    for k, g in ok.groupby("k"):  # for loop over the channels, pooled (anomalies removed)
        det.append(("V4", "ALL", "", int(k), len(g), sum(passes(h) for _, h in g.groupby(["test_code", "receiver", "sv"])), len(g) / 60.0, pct(np.abs(g.res_median_m), 50), pct(np.abs(g.res_median_m), 95), float(g.el_deg.median()), float(g.el_deg.min()), float(g.el_deg.max()), float(g.res_median_m.median())))  # one row
    dd = pd.DataFrame(det, columns=["test", "receiver", "sv", "channel", "n_residuals", "n_passes", "sat_minutes", "abs_median_m", "abs_p95_m", "el_median_deg", "el_min_deg", "el_max_deg", "signed_median_m"])  # the table
    dd.round(3).to_csv(CHECK_DIR / "glonass_broadcast_detail.csv", index=False)  # written
    bad = dd[(dd.test == "V3") & (dd.receiver == "ALL") & (dd.abs_p95_m > 6.0)]  # satellites whose 95th percentile exceeds 6 m (all receivers pooled)
    add("V3 satellites with 95th percentile |residual| above 6 m (all receivers pooled)", "listed with elevation", "none" if bad.empty else "; ".join(f"{r.sv}: p95 {r.abs_p95_m:.1f} m, n {r.n_residuals}, median el {r.el_median_deg:.0f} deg ({r.el_min_deg:.0f}-{r.el_max_deg:.0f})" for r in bad.itertuples()), None)  # the list
    badr = dd[(dd.test == "V3") & (dd.receiver != "ALL") & (dd.abs_p95_m > 6.0)]  # the same per receiver
    add("V3 receiver x satellite pairs with 95th percentile above 6 m", "listed", "none" if badr.empty else "; ".join(f"{r.receiver} {r.sv}: {r.abs_p95_m:.1f} m (n {r.n_residuals})" for r in badr.itertuples()), None)  # the list
    # V4
    chm = dd[(dd.test == "V4") & (dd.receiver == "ALL")]  # pooled channel medians
    add("V4 median residual per channel, pooled (k: signed median m, n)", "reported", "; ".join(f"{int(r.channel)}: {r.signed_median_m:+.2f} (n {r.n_residuals})" for r in chm.itertuples()), None)  # the channel medians
    for rx in D.RXS:  # for loop over the receivers
        c = dd[(dd.test == "V4") & (dd.receiver == rx)]  # that receiver's channel medians
        c5 = c[c.n_residuals >= 30]  # channels with at least 30 residuals
        spread = float(c5.signed_median_m.max() - c5.signed_median_m.min()) if len(c5) > 1 else float("nan")  # max minus min of the channel medians
        add(f"V4 {RXN[rx]}: channel-median spread (channels with >= 30 residuals), largest |channel median|", "pattern above 1 m is a finding", f"spread {spread:.2f} m over {len(c5)} channels, largest {c5.signed_median_m.abs().max():.2f} m; " + "; ".join(f"{int(r.channel)}:{r.signed_median_m:+.2f}({r.n_residuals})" for r in c.itertuples()), None)  # per receiver
    NUM.update({"v3_minutes": sizes.minutes.sum(), "v3_rows": int(sizes.glonass_rows.sum()), "v3_modelled": int(sizes.modelled_rows.sum()), "v3_resid": len(res), "v3_anom": len(an),  # V3 funnel
                "drop_unhealthy": orb.drops["unhealthy"], "drop_no_record": orb.drops["no_record"], "drop_gap": orb.drops["gap"], "drop_svs": "; ".join(f"{k_[1]} {v_}" for k_, v_ in sorted(orb.drop_sv.items())),  # drops
                "ties_clean": len(orb.ties), "ties_day": int(((mid >= day0) & (mid < day0 + 86400.0)).sum()), "k_table": orb.k_source["table"], "k_igs": orb.k_source["igs"],  # ties and channel source
                "rel_med": np.median(rel), "rel_max": rel.max(),  # relativity magnitude
                "p_all_med": np.median(a), "p_all_p95": pct(a, 95), "p_ok_med": np.median(b), "p_ok_p95": pct(b, 95), "p_ok_n": len(b), "p_w_med": np.median(aw), "p_w_p95": pct(aw, 95),  # pooled V3
                "p_signed_med": float(np.median(ok.res_median_m)), "p_signed_mean": float(ok.res_median_m.mean()),  # signed
                "after_med": float(np.median(np.abs(ok2.dm))), "after_p95": pct(np.abs(ok2.dm), 95), "sess_change_med": float(rng.median()), "sess_change_max": float(rng.max()), "sess_change_sv": str(rng.idxmax()), "sess_change_n": len(rng),  # offset structure
                "clkjump_med": float(np.median(cj)), "clkjump_p95": pct(cj, 95), "clkjump_max": float(max(cj)), "clkjump_n": len(cj),  # clock continuity
                "rel_corr": float(np.corrcoef(ok.res_median_m, ok.rel_m)[0, 1]),  # correlation with rel_m
                "r19_rate_mean": float(rate.mean()), "r19_rate_min": float(rate.min()), "r19_rate_max": float(rate.max()), "r19_n": len(rate), "r19_left_mean": float(left.mean()), "r19_left_min": float(left.min()), "r19_left_max": float(left.max()),  # R19 test
                "r19_pred_mean": float(pred.mean()), "r19_pred_min": float(pred.min()), "r19_pred_max": float(pred.max()), "r19_rms": float(np.sqrt(np.mean((left - pred) ** 2)))})  # R19 prediction
    for col, key in (("z_rel", "relminus"), ("z_opp", "relplus")):  # for loop over the two signs of the relativistic sensitivity
        q = rr[rr.anomaly == 0]  # without the anomalies
        NUM[f"{key}_med"], NUM[f"{key}_p95"] = float(np.median(np.abs(q["r_" + col]))), pct(np.abs(q["r_" + col]), 95)  # kept
    for rx in [None] + list(D.RXS):  # for loop over all receivers pooled, then each
        q = rf if rx is None else rf[rf.receiver == rx]  # the refined rows
        q0 = ok if rx is None else ok[ok.receiver == rx]  # the primary rows
        tag = "all" if rx is None else rx  # the key part
        NUM[f"refine_med_{tag}"], NUM[f"refine_p95_{tag}"] = float(np.median(np.abs(q.res_median_m))), pct(np.abs(q.res_median_m), 95)  # refined statistics
        NUM[f"prim_med_{tag}"], NUM[f"prim_p95_{tag}"], NUM[f"prim_n_{tag}"] = float(np.median(np.abs(q0.res_median_m))), pct(np.abs(q0.res_median_m), 95), len(q0)  # primary statistics
        NUM[f"prim_sv_{tag}"], NUM[f"prim_ep_{tag}"] = q0.sv.nunique(), q0.sec.nunique()  # satellites and epochs
    for a_, b_ in (("SBF_02", "SBF_03"), ("SBF_02", "UBX_01"), ("SBF_03", "UBX_01")):  # for loop over the receiver pairs
        q = p[[a_, b_]].dropna()  # satellite-sessions both saw
        NUM[f"corr_{a_}_{b_}"], NUM[f"rmsd_{a_}_{b_}"], NUM[f"n_{a_}_{b_}"] = float(q.corr().iloc[0, 1]), float(np.sqrt(((q[a_] - q[b_]) ** 2).mean())), len(q)  # offset structure
    NUM["rms_level"] = float(np.sqrt((p["SBF_02"].dropna() ** 2).mean()))  # rms level of the X5 satellite-session medians
    for sy, g in ctx.groupby("sys"):  # for loop over the other constellations
        NUM[f"ctx_med_{sy}"], NUM[f"ctx_p95_{sy}"], NUM[f"ctx_n_{sy}"] = float(g.pr_res_m.abs().median()), pct(g.pr_res_m.abs(), 95), len(g)  # context
    pd.DataFrame({"key": list(NUM), "value": [NUM[k_] if isinstance(NUM[k_], str) else float(NUM[k_]) for k_ in NUM]}).to_csv(CHECK_DIR / "glonass_broadcast_summary.csv", index=False, float_format="%.6g")  # the numbers
    res.round({"el_deg": 4, "z_clock_in_m": 4, "res_median_m": 4, "res_wmean_m": 4, "rel_m": 4}).to_csv(PROC / "preprocessing" / "glonass_residuals_clean.csv", index=False)  # the derived product
    figure(ok, dd)  # the figure (anomalies left out of the scatter)
    pd.DataFrame(CHK).to_csv(CHECK_DIR / "glonass_broadcast_check.csv", index=False)  # one row per check
    from helpers import write_facts  # the facts-table writer, imported here so no earlier line moves
    # the receiver's GLONASS clock term of each second (the epoch median clock, z minus the residual about it), in milliseconds
    ck = ((res.z_clock_in_m - res.res_median_m) / C * 1000.0).groupby([res.test_code, res.receiver, res.sec]).median().abs().reset_index(name="ms")
    fam = {"Septentrio": ck[ck.receiver != "UBX_01"].ms, "u-blox 01": ck[ck.receiver == "UBX_01"].ms}  # dict: the two receiver families (mosaic-X5 and mosaic-H; u-blox 01)
    # write_facts with a list comprehension of 4-tuples: the smallest and largest |clock term| of each family, over the V3 seconds
    write_facts(__file__, [(f"glonass_clock_term_{k.split()[0].replace('-', '').lower()}_{agg}", float(getattr(v, agg)()), "ms",
                            f"{ {'min': 'minimum', 'max': 'maximum'}[agg]} over the V3 clean-interval seconds of the absolute GLONASS receiver clock term (epoch median clock) of {k}")  # its definition
                           for k, v in fam.items() for agg in ("min", "max")])  # two for clauses: each family, then min and max
    print("done")  # finished


def figure(res: pd.DataFrame, dd: pd.DataFrame) -> None:  # function: residual against elevation per receiver, and per-channel medians
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.5, 1]})  # two panels side by side
    for rx in D.RXS:  # for loop over the receivers
        r = res[res.receiver == rx]  # that receiver's rows
        ax[0].scatter(r.el_deg, r.res_median_m, s=5, alpha=0.35, color=COLORS[rx], label=f"{RXN[rx]} (n {len(r)})", rasterized=True)  # scatter of residual against elevation
    ax[0].axhline(0, color="k", lw=0.6)  # the zero line
    for v in (-5, 5):  # for loop over the 5 m limit lines
        ax[0].axhline(v, color="0.5", lw=0.8, ls="--")  # the 95th percentile criterion
    ax[0].set_xlabel("elevation (degrees)")  # x label
    ax[0].set_ylabel("G1 C1C residual about the epoch median clock (m)")  # y label
    ax[0].set_title("GLONASS residuals on the clean reference intervals")  # title
    ax[0].legend(loc="upper right", fontsize=8, markerscale=3)  # legend
    for j, rx in enumerate(D.RXS):  # for loop over the receivers
        c = dd[(dd.test == "V4") & (dd.receiver == rx)]  # that receiver's channel medians
        ax[1].scatter(c.channel + (j - 1) * 0.18, c.signed_median_m, s=18 + 3 * np.sqrt(c.n_residuals), color=COLORS[rx], label=RXN[rx], edgecolor="k", lw=0.4)  # one dot per channel, area grows with count
    ax[1].axhline(0, color="k", lw=0.6)  # the zero line
    ax[1].set_xlabel("GLONASS frequency channel k")  # x label
    ax[1].set_ylabel("median residual (m)")  # y label
    ax[1].set_title("per-channel medians (dot area grows with n)")  # title
    ax[1].legend(fontsize=8)  # legend
    fig.tight_layout()  # spacing
    save_fig(fig, GRAPH / "glonass_residuals.png")  # byte-stable save


if __name__ == "__main__":  # if: run as a script
    main()  # run the checks
