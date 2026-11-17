#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
The per-second observable table. Exploration only: nothing produced here is a feature or a label, and the
group column is PROVISIONAL (the labels come from the label script).

For the mosaic-X5 (SBF_02), the mosaic-H main antenna (SBF_03) and u-blox 01 (UBX_01) in all six sessions, at 1 Hz (the first epoch of
each UTC second), on GPS L1, GLONASS G1, Galileo E1 and BeiDou B1I only, it writes
  Data/processed/features/observables_1hz.csv.gz         one row per satellite and second (every satellite with any L1 observation)
  Data/processed/features/observables_epochs_1hz.csv.gz  one row per second of the capture span (a second without any observation is a row)
  Data/processed/features/observables_data_dictionary.csv
  Data/processed/features/observables_missingness.csv    what exists, by session, receiver and provisional group
  Data/processed/features/observables_checks.csv         the self-checks, long form
  Data/processed/ground_truth/glonass_channels.csv           the GLONASS slot -> frequency channel table (union of the RINEX headers)
  Data/processed/checks/observables_spot_*.png            spot plots

Never read: receiver status, AGC, authentication, the OBD-II odometer, the IMU, navigation-message content, mosaic-G5, UBX_02, and the
u-blox time jump (2.6.12 u-blox 01 is put back on true time by the host-time index, exactly as in the evidence script). Orbits and clocks come from the
IGS merged broadcast file only (functions of the reference-position and evidence scripts); there is no GLONASS orbit in the project, so GLONASS carries observation-intrinsic
quantities only and its modelled residuals are flagged "no_orbit".

Reads   Data/rinex/<session>/*.crx.gz, Data/processed/{session_ground_truth,reference_positions,reference_clean_intervals,
        ta3_schedule_confirmation,early_warning_g30}.csv, the IGS file
"""

from __future__ import annotations  # future import: type hints kept as text (a convention across the project's scripts)

import gzip  # standard-library gzip, to write compressed tables with a fixed header (byte-identical runs again)
import importlib.util  # standard-library module loader, to load the evidence script whose name starts with a digit
import os  # operating-system access: MPLCONFIGDIR is set before matplotlib loads
import random  # standard-library random; seeded below
import re  # regular expressions, to read the GLONASS slot records of the RINEX headers
import sys  # standard-library system access: sys.path
import datetime as dtm  # standard-library dates, for the day of the recording
from datetime import datetime, timezone  # date-time classes, to read the RINEX epoch lines
from pathlib import Path  # Path objects for building file paths

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
os.environ["MPLCONFIGDIR"] = str(PROJECT / "codes" / ".mplconfig")  # matplotlib's cache inside the project folder

import numpy as np  # NumPy: arrays and medians
import pandas as pd  # pandas: tables

sys.path.insert(0, str(PROJECT / "codes" / "common"))  # puts codes/ first on the import path so the helpers module is found
from helpers import exit_if_failed, save_fig, known_invalid as known_invalid_rule  # byte-stable figure saving; the one known-invalid rule; exit_if_failed ends the run on a failed check

SEED = 20260914  # integer constant: the project's fixed random seed
random.seed(SEED)  # seeds Python's generator
np.random.seed(SEED)  # seeds NumPy's generator

PROC = PROJECT / "Data" / "processed"  # where the CSVs are read and written
CHECK_DIR = PROJECT / "Data" / "processed" / "checks"  # where the spot plots go

spec = importlib.util.spec_from_file_location("gain_evidence_for_observables", PROJECT / "codes" / "1_ground_truth" / "04_gain_and_position_evidence.py")  # describes where the evidence script lives
D = importlib.util.module_from_spec(spec)  # creates an empty module object for it
sys.modules["gain_evidence_for_observables"] = D  # registers it so its own imports resolve
spec.loader.exec_module(D)  # runs the evidence script's top level (definitions only; main() is guarded)
M = D.M  # the position script module that the evidence script loaded (orbits, geometry, corrections, the solver)
T = D.T  # clock text -> unix seconds on the recording date

SESSIONS, RXS = D.SESSIONS, D.RXS  # the six sessions in time order and the three receivers
C_LIGHT = M.C_LIGHT  # speed of light, m/s
LAM = {"G": C_LIGHT / 1575.42e6, "E": C_LIGHT / 1575.42e6, "C": C_LIGHT / 1561.098e6}  # dict: carrier wavelength (m) of GPS L1, Galileo E1 and BeiDou B1I; GLONASS depends on the slot
LAM_L1 = D.LAMBDA_L1  # the wavelength the evidence script used for its model Doppler (GPS L1 = Galileo E1)
CODES = {"SBF": {"G": "C1C", "R": "C1C", "E": "C1C", "C": "C2I"}, "UBX": {"G": "C1C", "R": "C1C", "E": "C1X", "C": "C2I"}}  # dict: the L1-type code per constellation and receiver family (the L1-band codes)
MASK = D.MASK_DEG  # the elevation mask in degrees (10)
BOUNDS = {"G": (15.0, 65.0), "R": (15.0, 65.0), "E": (15.0, 65.0), "C": (15.0, 65.0)}  # dict: plausible C/N0 range in dB-Hz for the plausibility check
GLO_F0, GLO_DF = 1602.0e6, 0.5625e6  # GLONASS G1 centre frequency and channel spacing, Hz
PAIRS = [(n, n + 4) for b in (1, 9, 17) for n in range(b, b + 4)]  # list comprehension: the antipodal slot pairs (1,5) (2,6) ... (20,24)

CHECKS: list = []  # list: the self-checks, written to features/observables_checks.csv
FAILED: list = []  # list: names of the self-checks that failed; the script ends the run with status 1 when it is not empty
HDR_SLOTS: dict = {}  # dict: file name -> {slot: channel} from that file's header
USED_SLOTS: dict = {}  # dict: file name -> set of slots seen in the file's observations
LINE_COUNT: dict = {}  # dict: file name -> (satellite lines in kept epochs, rows kept, empty lines dropped)


def check(item: str, quantity: str, value, ok: bool | None = None) -> None:  # function: records one self-check and prints it
    CHECKS.append({"item": item, "quantity": quantity, "value": value, "pass": "" if ok is None else bool(ok)})  # append a dict row to the list
    if ok is not None and not bool(ok):  # if: a pass/fail check (None marks a reported number) that did not pass
        FAILED.append(f"{item}: {quantity}")  # its name kept for the exit at the end of the run
    print(f"  [{'ok' if ok else ('FAIL' if ok is False else '--')}] {item}: {quantity} = {value}")  # progress line


def read_all(path: Path, spans: list, rx: str) -> pd.DataFrame:  # function: GPS, GLONASS, Galileo and BeiDou L1-type code, carrier, Doppler, C/N0 and lock flags inside the spans, first epoch of each UTC second
    lines, k0, codes = D.rinex_lines(path)  # the file's lines, the first data line and the observation codes (the evidence script's reader, cached)
    slots = {}  # this file's GLONASS slot -> channel table
    for l in lines[:k0]:  # for loop over the header lines
        if l[60:].strip() == "GLONASS SLOT / FRQ #":  # if: a slot record
            for m in re.finditer(r"R(\d\d)\s+(-?\d+)", l[:60]):  # regular expression: every "Rnn  k" pair on the line
                slots[int(m.group(1))] = int(m.group(2))  # slot number -> channel number
    HDR_SLOTS[path.name] = slots  # kept for the union table
    USED_SLOTS[path.name] = {int(l[1:3]) for l in lines[k0:] if l[:1] == "R"}  # set comprehension: every GLONASS slot with a line anywhere in the file
    want = CODES[rx[:3]]  # the code letters per constellation for this receiver family
    cols = {}  # per constellation: column index of code, carrier, Doppler, C/N0
    for s in "GREC":  # for loop over the four constellations used
        c = want[s]  # the pseudorange code, e.g. C1C
        names = [c, "L" + c[1:], "D" + c[1:], "S" + c[1:]]  # the four related observation codes: code, carrier, Doppler, C/N0
        cols[s] = [codes[s].index(n) if (s in codes and n in codes[s]) else None for n in names]  # their columns (None if the header lacks the code)
    lo, hi = [a for a, _ in spans], [b for _, b in spans]  # span starts and ends in unix seconds
    rows, keep, last, sec, t_gps = [], False, None, 0, 0.0  # kept rows; epoch state
    n_lines = n_empty = 0  # satellite lines in kept epochs; lines with all four values empty
    for l in lines[k0:]:  # for loop over the data lines
        if l[:1] == ">":  # if: an epoch line
            t_gps = (datetime(int(l[2:6]), int(l[7:9]), int(l[10:12]), int(l[13:15]), int(l[16:18]), tzinfo=timezone.utc) - M.GPS_EPOCH).total_seconds() + float(l[18:29])  # GPS seconds
            t_unix = t_gps - M.LEAP_SECONDS + M.GPS_UNIX  # UTC as unix seconds
            sec = int(np.floor(t_unix))  # its UTC second
            keep = sec != last and any(a <= t_unix < b for a, b in zip(lo, hi))  # the first epoch of a second inside a span
            if keep:  # if: kept
                last = sec  # this second is taken
        elif keep and l[:1] in cols:  # elif: a satellite line of a kept epoch for a constellation we use
            n_lines += 1  # one more satellite line
            vals, lli = [], []  # the four values and their lock flags
            for i in cols[l[0]]:  # for loop over the four columns
                f = l[3 + 16 * i:3 + 16 * i + 16] if i is not None else ""  # the 16-character field: 14 value, LLI, SSI
                vals.append(float(f[:14]) if f[:14].strip() else np.nan)  # the value or NaN
                lli.append(int(f[14]) if len(f) > 14 and f[14].isdigit() else 0)  # the loss-of-lock indicator digit, 0 if blank
            if all(np.isnan(v) for v in vals):  # if: nothing observed on this band
                n_empty += 1  # counted, not kept
                continue  # next line
            rows.append((sec, t_gps, l[:3], vals[0], vals[1], vals[2], vals[3], lli[0], lli[1]))  # one observation row
    LINE_COUNT[path.name] = (n_lines, len(rows), n_empty)  # for the reconciliation check
    return pd.DataFrame(rows, columns=["sec", "t_gps", "sv", "pr", "cp", "dop", "cn0", "lli_pr", "lli_cp"])  # the table


def glonass_union() -> dict:  # function: the slot -> channel table from all headers, with the conflict, pairing and gap checks
    union, seen = {}, {}  # slot -> channel; slot -> number of files that list it
    conflicts = []  # slots given two different channels
    for name, sl in HDR_SLOTS.items():  # for loop over the files
        for k, v in sl.items():  # for loop over that file's slots
            if k in union and union[k] != v:  # if: a different channel from before
                conflicts.append((name, k, union[k], v))  # a conflict
            union.setdefault(k, v)  # keeps the first channel seen
            seen[k] = seen.get(k, 0) + 1  # one more file lists the slot
    check("GLONASS channels", "files read, distinct records, slots in the union, conflicting channels", f"{len(HDR_SLOTS)}, {len({tuple(sorted(s.items())) for s in HDR_SLOTS.values()})}, {len(union)}, {len(conflicts)}", len(conflicts) == 0)  # no slot may have two channels
    both = [(a, b) for a, b in PAIRS if a in union and b in union]  # the antipodal pairs with both slots present
    bad = [(a, b, union[a], union[b]) for a, b in both if union[a] != union[b]]  # pairs whose channels differ
    check("GLONASS channels", "antipodal pairs (n, n+4) with both slots present, pairs sharing a channel", f"{len(both)}, {len(both) - len(bad)}", len(bad) == 0)  # slot n and n+4 must share a channel
    gaps = [(name, sorted(USED_SLOTS[name] - set(HDR_SLOTS[name]))) for name in HDR_SLOTS if USED_SLOTS[name] - set(HDR_SLOTS[name])]  # files with an observed slot missing from their own header
    check("GLONASS channels", "files with a slot observed but absent from the file's own header", "; ".join(f"{n}: R{', R'.join(f'{s:02d}' for s in sl)}" for n, sl in gaps) or "none", None)  # the per-file gaps
    check("GLONASS channels", "every observed slot is in the union", str(all(s in union for u in USED_SLOTS.values() for s in u)), all(s in union for u in USED_SLOTS.values() for s in u))  # the union covers all
    pd.DataFrame([{"slot": f"R{k:02d}", "channel": union[k], "files_listing": seen[k], "paired_with": next((f"R{b:02d}" if a == k else f"R{a:02d}" for a, b in PAIRS if k in (a, b)), ""), "wavelength_m": C_LIGHT / (GLO_F0 + union[k] * GLO_DF)} for k in sorted(union)]).round(6).to_csv(PROC / "ground_truth" / "glonass_channels.csv", index=False)  # the table
    return union  # slot -> channel


def jam_end_fn(gt: pd.DataFrame):  # function: returns a function giving the scheduled end of the initial jamming of a session, or None for the exempt sessions
    ta3 = pd.read_csv(PROC / "ground_truth" / "ta3_schedule_confirmation.csv")  # the receiver-status script's schedule confirmation
    def jam_end(test: str):  # nested function: unix seconds or None
        v = ta3[ta3.test_code == test].initial_jamming_end_utc.iloc[0]  # the text
        return T(v) if isinstance(v, str) and v[:1].isdigit() else None  # a clock time, or 'exempt'
    return jam_end  # the function


def groups_for(gt: pd.DataFrame, jam_end, test: str, rx: str, secs: np.ndarray) -> np.ndarray:  # function: the PROVISIONAL group of each second: clean, jamming, spoofing or other
    s0, s1 = T(gt.loc[test, "scheduled_start_utc"][11:]), T(gt.loc[test, "scheduled_end_utc"][11:])  # the scheduled window
    je = jam_end(test)  # the end of the initial jamming, or None
    sp_end = max(s1, T("14:43:59")) if test == "2.6.10" else s1  # the end of spoofed tracking in 2.6.10: in 2.6.10 the spoofed tracking lasted to 14:43:59, counted as spoofing; the per-second end is later, 14:44:04 pooled (X5 and mosaic-H 14:44:03, u-blox 01 14:44:04; ground_truth/spoof_end_2610.csv); this minute bound is kept
    jam = (secs >= s0) & (secs < je) if je is not None else np.zeros(len(secs), bool)  # the simultaneous jamming and spoofing phase
    spf = (secs >= (je if je is not None else s0)) & (secs <= sp_end)  # the spoofing-only phase (or the whole window where there was no initial jamming)
    cln = D.in_spans(secs.astype(float), D.clean_spans(test, rx))  # the kept clean intervals of the position script
    over = int((cln & (jam | spf)).sum())  # clean seconds that also fall in a phase (there must be none)
    check("groups", f"{test} {rx}: clean seconds inside a jamming or spoofing window", over, over == 0)  # the two definitions must not overlap
    out = np.full(len(secs), "other", dtype=object)  # everything else
    out[jam], out[spf], out[cln] = "jamming", "spoofing", "clean"  # fancy indexing: labels set by mask
    return out  # one label per second


def known_invalid(sv: pd.Series, sec: pd.Series, test: str, rx: str) -> np.ndarray:  # function: the diagnosed invalid measurements (not attack signatures), flagged
    return known_invalid_rule(sv, sec, test, rx, D.DT)  # one shared rule in the helpers module, instants from ground_truth/decision_times.csv (the evidence script reads them)


def solver_epochs(orb, kl: tuple, obs: pd.DataFrame, site: np.ndarray) -> pd.DataFrame:  # function: per second, the position script solver on GPS and Galileo: clocks, PDOP and HDOP of the satellites it kept, position offset from the site
    rows = []  # one dict per second
    ge = obs[obs.sv.str[0].isin(["G", "E"]) & obs.pr.notna()]  # GPS and Galileo pseudoranges only, as in the position script
    for sec, g in ge.groupby("sec", sort=True):  # for loop over the seconds
        svs, t_gps, pr = list(g.sv), float(g.t_gps.iloc[0]), g.pr.to_numpy()  # satellites, GPS time, pseudoranges
        pos, dts, ok = orb.state(svs, t_gps, pr)  # positions and clocks at transmission time
        sel = np.flatnonzero(ok)  # usable satellites
        if len(sel) < 5:  # if: too few for a solution with redundancy
            continue  # no solution this second
        sv_s, pr_s, pos_s, dts_s = [svs[i] for i in sel], pr[sel], pos[sel], dts[sel]  # the usable subset
        sol = M.solve_epoch(sv_s, pr_s, pos_s, dts_s, t_gps, site, kl)  # the position script weighted least squares with 30 m rejection
        if sol is None:  # if: no solution
            continue  # next second
        x, idx = sol["x"], sol["idx"]  # the position and the satellites the solver kept
        rho, e, el, az = M.geometry(x, pos_s[idx])  # geometry at the solution
        z = pr_s[idx] - rho + C_LIGHT * dts_s[idx] - M.corrections(x, t_gps, [sv_s[i] for i in idx], el, az, kl)  # measured minus modelled, clock still in
        w = np.sin(el) ** 2  # the solver's elevation weights
        sysl = np.array([sv_s[i][0] for i in idx])  # constellation letter of each kept satellite
        H = np.zeros((len(idx), 3 + len(sol["systems"])))  # design matrix: three position columns plus one clock column per constellation
        H[:, :3] = e  # unit vectors
        for c, s in enumerate(sol["systems"]):  # for loop over the constellations in the solution
            H[:, 3 + c] = (sysl == s)  # that constellation's clock column
        try:  # try: the cofactor matrix may be singular in a degenerate geometry
            Q = np.linalg.inv(H.T @ H)  # cofactor matrix of the unweighted geometry
            lat, lon, _ = M.ecef_to_llh(*x)  # geodetic position of the solution
            Re = M.enu_axes(lat, lon)  # rows: east, north, up unit vectors
            Qe = Re @ Q[:3, :3] @ Re.T  # position cofactors in the local frame
            pdop, hdop = float(np.sqrt(np.trace(Q[:3, :3]))), float(np.sqrt(Qe[0, 0] + Qe[1, 1]))  # dilution of precision
        except np.linalg.LinAlgError:  # except: singular matrix
            pdop = hdop = np.nan  # no DOP this second
        clk = {s: float(np.sum(w[sysl == s] * z[sysl == s]) / np.sum(w[sysl == s])) for s in sol["systems"]}  # weighted mean of the clock-in residuals per constellation = that constellation's clock at the converged position
        hoff, voff = D.enu_dist(x, site)  # horizontal and vertical offset of the solution from the site, metres
        row = {"sec": int(sec), "solver_ok": 1, "n_used": len(idx)}  # dict literal: the row starts with the second, the success flag and the satellites kept
        row["n_used_G"], row["n_used_E"], row["n_rej"] = int((sysl == "G").sum()), int((sysl == "E").sum()), sol["n_rej"]  # kept per constellation and the count the 30 m rule rejected
        row["pdop"], row["hdop"] = pdop, hdop  # dilution of precision of the kept satellites
        row["clk_gps_m"], row["clk_gal_m"] = clk.get("G", np.nan), clk.get("E", np.nan)  # the two constellation clocks (NaN if a constellation is not in the solution)
        row["h_off_m"], row["v_off_m"] = float(hoff), float(voff)  # offset of the solver's position from the site
        rows.append(row)  # list append: keeps the finished row
    return pd.DataFrame(rows)  # the table


def per_satellite(orb, kl: tuple, obs: pd.DataFrame, site: np.ndarray, test: str, rx: str, glo: dict) -> pd.DataFrame:  # function: the per-satellite quantities of one session and receiver
    o = obs.sort_values(["sv", "sec"]).reset_index(drop=True).copy()  # satellite-major order, for the time differences
    o["sys"] = o.sv.str[0]  # constellation letter
    lam = np.array([LAM.get(s, np.nan) for s in o.sys])  # wavelength per row (GLONASS filled below)
    for i in np.flatnonzero(o.sys.to_numpy() == "R"):  # for loop over the GLONASS rows
        k = glo.get(int(o.sv.iloc[i][1:]))  # the slot's channel from the union table
        lam[i] = C_LIGHT / (GLO_F0 + k * GLO_DF) if k is not None else np.nan  # its G1 wavelength
    o["lam"] = lam  # kept
    g = o.groupby("sv")  # one group per satellite
    p1, p2 = g.sec.shift(1), g.sec.shift(2)  # the seconds of the previous two observations of the same satellite
    ok1, ok2 = (o.sec - p1) == 1, ((o.sec - p1) == 1) & ((p1 - p2) == 1)  # consecutive-second masks for the first and second difference
    o["tracked_prev"] = ok1.astype(int)  # 1 if the satellite was also tracked in the previous second
    o["dC_m"] = np.where(ok1, o.pr - g.pr.shift(1), np.nan)  # single time difference of the pseudorange, metres per second (includes the receiver clock drift)
    o["d2C_m"] = np.where(ok2, o.pr - 2 * g.pr.shift(1) + g.pr.shift(2), np.nan)  # second difference, metres per second squared
    o["cmc"] = o.pr - o.lam * o.cp  # code minus carrier, metres (carrier in cycles times wavelength)
    o["cmc_rate_m_s"] = np.where(ok1, o.cmc - o.groupby("sv").cmc.shift(1), np.nan)  # first difference of code minus carrier, metres per second
    gec = o[o.sys.isin(["G", "E", "C"]) & o.pr.notna()][["sec", "t_gps", "sv", "pr"]]  # the rows an orbit exists for, with a pseudorange
    z = D.epoch_z(orb, gec, site, kl, with_rate=True)  # measured minus modelled range (clock in), elevation and model Doppler per satellite and second
    z["rr"] = -z.dop_true * LAM_L1  # range rate in m/s recovered from the evidence script's model Doppler (it divides by the GPS wavelength)
    z["geo"] = [s[0] == "C" and int(s[1:]) in M.BDS_GEO for s in z.sv]  # list comprehension: BeiDou GEO satellites, whose orbits are not validated
    z["sys"] = z.sv.str[0]  # constellation letter
    rs = D.resid_mask(z[~z.geo][["sec", "sv", "z", "el", "sys"]], MASK)  # code residuals about each constellation's median clock above the mask (the evidence script definition), n >= 4 per constellation
    o = o.merge(z[["sec", "sv", "el", "rr", "geo"]].rename(columns={"el": "el_deg"}), on=["sec", "sv"], how="left").merge(rs[["sec", "sv", "res"]].rename(columns={"res": "pr_res_m"}), on=["sec", "sv"], how="left")  # the model quantities attached
    o["geo"] = o.geo.eq(True)  # comparison with True: NaN (no model) becomes False, and the column is a plain boolean
    lam_m = np.where(o.sys.isin(["G", "E", "C"]), o.lam, np.nan)  # wavelength where a model exists
    o["dop_model_hz"] = -o.rr / lam_m  # model Doppler in each satellite's own band (RINEX sign: positive when the range shortens)
    usable = (o.el_deg >= MASK) & ~o.geo & o.dop.notna() & o.dop_model_hz.notna()  # satellites that enter the Doppler drift
    e_ms = np.where(usable, -o.lam * (o.dop - o.dop_model_hz), np.nan)  # measured minus modelled range rate, m/s (the common part is the receiver clock drift)
    tmp = pd.DataFrame({"sec": o.sec, "e": e_ms})  # a small table for the per-second median
    cnt, med = tmp.groupby("sec").e.count(), tmp.groupby("sec").e.median()  # satellites and median per second
    drift = med.where(cnt >= 4)  # the drift needs four satellites, as the code clock does
    o["dop_drift_m_s"] = o.sec.map(drift)  # the per-second receiver drift (same for every satellite of the second)
    o["dop_res_hz"] = np.where(usable, -(e_ms - o.dop_drift_m_s) / o.lam, np.nan)  # Doppler residual in Hz: measured minus modelled minus the receiver drift
    reason = np.full(len(o), "", dtype=object)  # why a model quantity is missing, per row
    reason[(o.sys == "R").to_numpy()] = "no_orbit"  # GLONASS: no broadcast-orbit code in the project
    reason[((o.sys != "R") & o.pr.isna()).to_numpy()] = "no_pseudorange"  # no code observation, so no transmission time
    reason[((o.sys != "R") & o.pr.notna() & o.el_deg.isna()).to_numpy()] = "no_orbit_record"  # the IGS file has no usable record for the satellite
    reason[o.geo.to_numpy()] = "bds_geo"  # BeiDou GEO: not validated
    reason[(reason == "") & (o.el_deg < MASK).to_numpy()] = "below_mask"  # under 10 degrees
    reason[(reason == "") & o.pr_res_m.isna().to_numpy()] = "clock_n<4"  # fewer than four satellites of the constellation for a median clock
    o["no_model_reason"] = reason  # kept
    o["known_invalid"] = known_invalid(o.sv, o.sec, test, rx).astype(int)  # the diagnosed invalid measurements
    for c in ("pr", "cp", "dop", "cn0"):  # for loop over the four observations
        o["has_" + {"pr": "pr", "cp": "carrier", "dop": "doppler", "cn0": "cn0"}[c]] = o[c].notna().astype(int)  # presence flags
    o["lock_loss"] = (o.lli_cp % 2 == 1).astype(int)  # RINEX LLI bit 0 on the carrier: loss of lock / possible cycle slip
    return o  # the table


def agg_stats(d: pd.DataFrame, col: str, name: str, std: bool = False) -> pd.DataFrame:  # function: per-second median and max |value| (and optionally the standard deviation) of one per-satellite quantity
    x = d[["sec", col]].dropna()  # rows where the quantity exists
    gb = x.groupby("sec")[col]  # one group per second
    out = pd.DataFrame({f"{name}_median": gb.median(), f"{name}_maxabs": x.assign(a=x[col].abs()).groupby("sec").a.max(), f"n_{name}": gb.count()})  # median, largest absolute value and the count behind them
    if std:  # if: the spread is wanted
        out[f"{name}_std"] = gb.std().where(gb.count() >= 2)  # sample standard deviation, needs two values
    return out  # indexed by second


def epoch_table(test: str, rx: str, grid: np.ndarray, sat: pd.DataFrame, sol: pd.DataFrame, group: np.ndarray) -> pd.DataFrame:  # function: one row per second of the capture span
    e = pd.DataFrame({"test_code": test, "receiver": rx, "sec": grid, "group_provisional": group})  # the skeleton: every second, observed or not
    cnt = sat.groupby(["sec", "sys"]).size().unstack(fill_value=0).reindex(columns=list("GREC"), fill_value=0)  # satellites tracked per constellation
    for s in "GREC":  # for loop over the constellations
        e["n_" + s] = e.sec.map(cnt[s]).fillna(0).astype(int)  # tracked count (0 when nothing was tracked)
    e["n_tracked"] = e[["n_G", "n_R", "n_E", "n_C"]].sum(axis=1)  # all four together
    for s in ["tracked", "G", "R", "E", "C"]:  # for loop over the totals and the constellations
        e["d_n_" + s] = e["n_" + s].diff()  # change from the previous second (NaN for the first second)
    cont = sat.groupby("sec").tracked_prev.sum()  # satellites tracked now that were also tracked in the previous second
    e["n_continuing"] = e.sec.map(cont).fillna(0).astype(int)  # kept
    e["n_new"] = e.n_tracked - e.n_continuing  # newly appearing satellites (also: re-appearing after a gap)
    e["n_lost"] = e.n_tracked.shift(1) - e.n_continuing  # satellites tracked in the previous second and not now
    e.loc[e.index[0], "n_lost"] = np.nan  # the first second has no previous second
    e["n_lock_loss"] = e.sec.map(sat.groupby("sec").lock_loss.sum()).fillna(0).astype(int)  # satellites with a carrier loss-of-lock flag
    e["n_known_invalid"] = e.sec.map(sat.groupby("sec").known_invalid.sum()).fillna(0).astype(int)  # diagnosed invalid measurements (left out of the aggregates)
    e["has_observations"] = (e.n_tracked > 0).astype(int)  # at least one satellite tracked
    e = e.merge(sol, on="sec", how="left")  # the solver quantities
    e["solver_ok"] = e.solver_ok.fillna(0).astype(int)  # 0 where no solution was possible
    e["clk_gps_drift_m_s"] = e.clk_gps_m.diff()  # clock drift: first difference of the GPS clock bias, metres per second
    e["dop_drift_m_s"] = e.sec.map(sat.drop_duplicates("sec").set_index("sec").dop_drift_m_s)  # the Doppler-based drift (median over satellites)
    main = sat[sat.sys.isin(["G", "E", "C"]) & (sat.el_deg >= MASK) & ~sat.geo & (sat.known_invalid == 0)]  # the main set: GPS, Galileo and BeiDou MEO/IGSO above the mask, diagnosed invalid seconds left out
    for col, name, sd in (("cn0", "cn0", True), ("dop_res_hz", "dopres", True), ("pr_res_m", "prres", False), ("dC_m", "dC", False), ("d2C_m", "d2C", False), ("cmc_rate_m_s", "cmc", False)):  # for loop over the six quantities
        e = e.merge(agg_stats(main, col, name, sd), left_on="sec", right_index=True, how="left")  # their per-second aggregates
    glo = sat[(sat.sys == "R") & (sat.known_invalid == 0)]  # GLONASS rows (no elevation known, so no mask)
    for col, name, sd in (("cn0", "r_cn0", True), ("dC_m", "r_dC", False), ("d2C_m", "r_d2C", False), ("cmc_rate_m_s", "r_cmc", False)):  # for loop over GLONASS's observation-intrinsic quantities
        e = e.merge(agg_stats(glo, col, name, sd), left_on="sec", right_index=True, how="left")  # their per-second aggregates
    e["utc"] = pd.to_datetime(e.sec, unit="s", utc=True).dt.strftime("%H:%M:%S")  # clock text for reading
    return e  # the table


def write_gz(df: pd.DataFrame, path: Path) -> None:  # function: writes a gzip CSV with a fixed header so runs again are byte-identical
    with open(path, "wb") as fh, gzip.GzipFile(filename="", mode="wb", fileobj=fh, mtime=0, compresslevel=6) as gz:  # open file and gzip stream with no name or time in the header
        gz.write(df.to_csv(index=False, lineterminator="\n").encode("utf-8"))  # the CSV text as bytes; lineterminator="\n" fixes Unix line ends on every system, as 09 does


def same(a: pd.Series, b: pd.Series) -> int:  # function: number of positions where two series differ (NaN equals NaN)
    return int((~((a == b) | (a.isna() & b.isna()))).sum())  # count of mismatches


def main() -> None:  # function: builds the tables
    gt = pd.read_csv(PROC / "ground_truth" / "session_ground_truth.csv").set_index("test_code")  # session windows and folders
    jam_end = jam_end_fn(gt)  # the jamming-end function
    orb = M.Orbits(D.S04, dtm.date(2026, 9, 14))  # orbits and clocks from the IGS file only
    kl = M.igs_klobuchar(D.S04.IGS_BRDC)  # the GPS Klobuchar parameters
    OBS, SPAN = {}, {}  # (test, rx) -> observations; the capture span
    print("phase 1: reading the RINEX files")  # heading
    for test in SESSIONS:  # for loop over the sessions
        for rx in RXS:  # for loop over the receivers
            spans = D.capture_spans(gt, test, rx)  # the capture span (2.6.5 from the site start)
            SPAN[(test, rx)] = spans[0]  # kept
            if (test, rx) == ("2.6.12", "UBX_01"):  # if: the u-blox whose time tag was shifted
                orig = D.read_full  # remember the evidence script's own reader
                D.read_full = read_all  # the evidence script's host-time function reads the three pieces through this reader instead
                try:  # try: always restore the reader
                    OBS[(test, rx)] = D.ubx_2612_obs(gt)  # host-corrected observations of the whole capture, all four constellations
                    D.read_full = orig  # the original reader back
                    ref = D.ubx_2612_obs(gt)  # the same function with the evidence script's own reader (GPS and Galileo), for the equality check
                finally:  # finally: runs even after an error
                    D.read_full = orig  # the original reader back
                ob = OBS[(test, rx)]  # mine
                m = ob.merge(ref, on=["sec", "sv"], suffixes=("", "_ref"))  # rows both have
                for c in ("pr", "cp", "dop", "cn0"):  # for loop over the four observations
                    check("raw equality", f"{test} {rx} {c}: rows compared, differing", f"{len(m)}, {same(m[c], m[c + '_ref'])}", same(m[c], m[c + '_ref']) == 0)  # must be identical
                check("raw equality", f"{test} {rx}: G/E rows in the evidence script's reader missing from this table", int(len(ref) - len(m)), len(ref) == len(m))  # nothing lost
                D.LINES.clear()  # frees memory
                continue  # next receiver
            path = M.obs_path(gt.loc[test, "session_folder"], test, rx)  # the RINEX file
            ob = read_all(path, spans, rx)  # all four constellations, 1 Hz
            OBS[(test, rx)] = ob  # kept
            ref = D.read_full(path, spans, rx)  # the evidence script's own reader (GPS and Galileo), for the equality check
            m = ob.merge(ref, on=["sec", "sv"], suffixes=("", "_ref"))  # rows both have
            bad = {c: same(m[c], m[c + "_ref"]) for c in ("pr", "cp", "dop", "cn0")}  # mismatches per observation
            check("raw equality", f"{test} {rx}: rows compared, rows missing from this table, differing pr/cp/dop/cn0", f"{len(m)}, {len(ref) - len(m)}, {bad['pr']}/{bad['cp']}/{bad['dop']}/{bad['cn0']}", len(ref) == len(m) and sum(bad.values()) == 0)  # must be identical
            ep = D.rinex_epoch_times(path)  # every epoch line of the file (the evidence script's separate reader)
            su = np.floor(ep - M.LEAP_SECONDS + M.GPS_UNIX).astype(np.int64)  # their UTC seconds
            n_sec = len({int(s) for s in su if spans[0][0] <= s < spans[0][1]})  # distinct seconds of the span that have an epoch
            n_l, n_r, n_e = LINE_COUNT[path.name]  # satellite lines, kept rows, empty lines
            check("reconcile", f"{test} {rx}: seconds with a RINEX epoch in the span, seconds with an L1 observation, satellite lines = rows + empty", f"{n_sec}, {ob.sec.nunique()}, {n_l} = {n_r} + {n_e}", n_l == n_r + n_e and ob.sec.nunique() <= n_sec)  # the counts must reconcile
            D.LINES.clear()  # frees the decompressed file
    glo = glonass_union()  # the slot -> channel table
    sat_all, ep_all = [], []  # per-satellite and per-epoch tables of all sessions
    print("phase 2: quantities")  # heading
    for test in SESSIONS:  # for loop over the sessions
        site = D.site_xyz(M.SITE_OF[test])  # the session's site reference
        for rx in RXS:  # for loop over the receivers
            sat = per_satellite(orb, kl, OBS[(test, rx)], site, test, rx, glo)  # per-satellite quantities
            sol = solver_epochs(orb, kl, OBS[(test, rx)], site)  # per-second solver quantities
            a, b = SPAN[(test, rx)]  # the capture span
            grid = np.arange(int(a), int(b))  # every UTC second of the span
            grp = groups_for(gt, jam_end, test, rx, grid)  # provisional groups
            sat["group_provisional"] = pd.Series(grp, index=grid).reindex(sat.sec).to_numpy()  # each row inherits its second's group
            ep = epoch_table(test, rx, grid, sat, sol, grp)  # per-second table
            sat.insert(0, "receiver", rx)  # receiver column
            sat.insert(0, "test_code", test)  # session column
            print(f"  {test:7s} {rx}: {len(sat):7d} satellite-seconds, {int(ep.has_observations.sum()):5d} of {len(ep):5d} seconds tracked, {int(ep.solver_ok.sum()):5d} solved")  # progress line
            sat_all.append(sat)  # kept
            ep_all.append(ep)  # kept
    sat = pd.concat(sat_all, ignore_index=True)  # all sessions and receivers
    ep = pd.concat(ep_all, ignore_index=True)  # all sessions and receivers
    sat["utc"] = pd.to_datetime(sat.sec, unit="s", utc=True).dt.strftime("%H:%M:%S")  # clock text
    sat["geo"] = sat.geo.astype(int)  # BeiDou GEO flag as 0/1
    cols = ["test_code", "receiver", "sec", "utc", "group_provisional", "sv", "sys", "el_deg", "cn0", "dop"]  # the identifying columns and the observed ones
    cols += ["pr_res_m", "dop_res_hz", "dC_m", "d2C_m", "cmc_rate_m_s", "lock_loss", "lli_pr", "tracked_prev"]  # list +=: the derived quantities and the lock indicators
    cols += ["has_pr", "has_carrier", "has_doppler", "has_cn0", "no_model_reason", "known_invalid", "geo", "dop_drift_m_s"]  # the missingness columns
    sat_out = sat[cols].rename(columns={"dop": "dop_hz"}).round(4)  # rounded for byte-stable text
    ep_out = ep.round(4)  # rounded likewise
    for c in ("test_code", "receiver", "sec"):  # for loop over the key columns
        assert c in ep_out.columns  # sanity check on the keys
    write_gz(sat_out, PROC / "features" / "observables_1hz.csv.gz")  # the per-satellite table
    write_gz(ep_out, PROC / "features" / "observables_epochs_1hz.csv.gz")  # the per-epoch table
    counterfeit_facts(sat_out, ep_out)  # 5.4.2's counterfeit stretch, to Data/processed/printed_facts/
    selfchecks(gt, jam_end, orb, kl, sat_out, ep_out)  # the checks that need the finished tables
    missingness(sat_out, ep_out)  # the missingness table
    dictionary()  # the data dictionary
    pd.DataFrame(CHECKS).to_csv(PROC / "features" / "observables_checks.csv", index=False)  # the checks
    print(f"rows: {len(sat_out)} satellite-seconds, {len(ep_out)} epochs")  # final line


COUNTERFEIT_M = 1.0e6  # float constant: "millions of metres", the size of code residual that marks 5.4.2's counterfeit stretch of 2.6.12 u-blox 01


def counterfeit_facts(sat: pd.DataFrame, ep: pd.DataFrame) -> None:  # function: the counterfeit stretch of u-blox 01 in 2.6.12, written as a facts table
    from helpers import write_facts  # import inside the function: the facts-table writer
    k = lambda t: t[(t.test_code.astype(str) == "2.6.12") & (t.receiver == "UBX_01")]  # lambda: the rows of 2.6.12 u-blox 01
    s, e = k(sat), k(ep)  # its satellite rows and its epoch rows
    big = s.pr_res_m.abs().groupby(s.sec).max() >= COUNTERFEIT_M  # per second with a residual: is its largest |code residual| at least a million metres
    obs = e[e.has_observations == 1].copy()  # the seconds with an observation
    obs["minute"] = obs.utc.str[:5]  # their minute, "HH:MM"
    obs["big"] = obs.sec.map(big).fillna(False).astype(bool)  # map: the flag of each observed second (False where no residual could be formed)
    full = obs.groupby("minute").big.all()  # per minute: True when every observed second is beyond a million metres
    m0, m1 = full[full].index.min(), full[full].index.max()  # the first and the last such minute
    if not full.loc[m0:m1].all():  # if: a minute inside the stretch does not qualify
        raise SystemExit(f"2.6.12 u-blox 01: the counterfeit minutes {m0}-{m1} are not contiguous")  # stop: the stretch would not be one window
    win = obs[(obs.minute >= m0) & (obs.minute <= m1)]  # the observed seconds of the window
    r = s[s.sec.isin(win.sec)].pr_res_m.dropna()  # every code residual in the window, m
    # write_facts with a list of 4-tuples: key, value, unit, definition (each fixed from the meaning of 5.4.2's sentence before its value was printed)
    write_facts(__file__, [("ubx2612_counterfeit_first_minute", m0, "UTC", f"first whole minute of 2.6.12 u-blox 01 in which every second with an observation has a code residual of at least {COUNTERFEIT_M:.0f} m"),
                           ("ubx2612_counterfeit_last_minute", m1, "UTC", "last such minute (the window runs to its end)"),  # the window's last minute
                           ("ubx2612_counterfeit_seconds", int(len(win)), "s", "seconds with an observation in the window"),  # its size
                           ("ubx2612_counterfeit_solved", int(win.solver_ok.sum()), "s", "seconds of the window in which the solver produced a position"),  # never converges
                           ("ubx2612_counterfeit_tracked_median", float(win.n_tracked.median()), "satellites", "median per-second count of tracked satellites over the window"),  # decision (C)
                           ("ubx2612_counterfeit_res_min", float(r.min()), "m", "minimum code residual about the median clock over the window"),  # decision (C): the lower end of the spread
                           ("ubx2612_counterfeit_res_max", float(r.max()), "m", "maximum code residual about the median clock over the window")])  # the upper end; written as 01_observables_facts.csv


def selfchecks(gt, jam_end, orb, kl, sat: pd.DataFrame, ep: pd.DataFrame) -> None:  # function: the checks that need the finished tables
    print("self-checks")  # heading
    for c, (lo, hi) in (("cn0", (10.0, 65.0)),):  # tuple of one: the plausible range
        v = sat[c].dropna()  # observed values
        check("plausibility", "C/N0 (dB-Hz) min, 0.1 / 50 / 99.9 percentiles, max, share outside 10-65", f"{v.min():.1f}, {v.quantile(.001):.1f}/{v.median():.1f}/{v.quantile(.999):.1f}, {v.max():.1f}, {((v < lo) | (v > hi)).mean():.5f}", ((v < lo) | (v > hi)).mean() < 0.001)  # almost all values must be inside
    cl = sat[(sat.group_provisional == "clean") & sat.dop_res_hz.notna()]  # clean rows with a Doppler residual
    check("plausibility", "Doppler residual in clean data: median |x|, 99th percentile of |x| (Hz)", f"{cl.dop_res_hz.abs().median():.3f}, {cl.dop_res_hz.abs().quantile(.99):.3f}", cl.dop_res_hz.abs().quantile(.99) < 5.0)  # a stationary receiver: a few Hz at most
    cr = sat[(sat.group_provisional == "clean") & sat.pr_res_m.notna()]  # clean rows with a code residual
    check("plausibility", "code residual in clean data: median |x|, 99th percentile of |x| (m)", f"{cr.pr_res_m.abs().median():.2f}, {cr.pr_res_m.abs().quantile(.99):.2f}", cr.pr_res_m.abs().quantile(.99) < 20.0)  # metres, as in the evidence script
    s = ep[ep.solver_ok == 1]  # solved seconds
    # the rule below was applied to every solved second and failed unnoticed (99th percentile PDOP 99.69: all 645 seconds above 15
    # are spoofing seconds of 2.6.10 and 2.6.4, where 5-7 satellites form a spoofed geometry); the plausibility rule now reads the clean seconds, as the
    # residual checks above do, and the all-seconds percentiles stay as a reported number
    sc = s[s.group_provisional == "clean"]  # boolean row selection: the solved clean seconds
    check("plausibility", "PDOP and HDOP in clean seconds: 1st percentile, median, 99th percentile", f"{sc.pdop.quantile(.01):.2f}/{sc.pdop.median():.2f}/{sc.pdop.quantile(.99):.2f}; {sc.hdop.quantile(.01):.2f}/{sc.hdop.median():.2f}/{sc.hdop.quantile(.99):.2f}", sc.pdop.min() > 0.5 and sc.pdop.quantile(.99) < 15)  # a plausible geometry where nothing is attacked
    check("plausibility", "PDOP and HDOP: 1st percentile, median, 99th percentile", f"{s.pdop.quantile(.01):.2f}/{s.pdop.median():.2f}/{s.pdop.quantile(.99):.2f}; {s.hdop.quantile(.01):.2f}/{s.hdop.median():.2f}/{s.hdop.quantile(.99):.2f}", None)  # all solved seconds, attack included: reported, no pass/fail
    check("plausibility", "GPS clock bias (km) range of the solved seconds, and the largest |drift| (m/s)", f"{s.clk_gps_m.min() / 1e3:.1f} to {s.clk_gps_m.max() / 1e3:.1f}, {s.clk_gps_drift_m_s.abs().max():.1f}", None)  # for reading
    n_dups = int(sat.duplicated(["test_code", "receiver", "sec", "sv"]).sum())  # duplicate keys
    check("keys", "duplicate (session, receiver, second, satellite) rows", n_dups, n_dups == 0)  # none allowed
    # residuals equal the evidence script's own series on the 2.6.5 window (12:45:00 to 12:54:24)
    a, b = T("12:45:00"), T("12:54:25")  # the G30 window of the evidence script's G30 case study
    site = D.site_xyz(M.SITE_OF["2.6.5"])  # the 2.6.2 site
    g30 = pd.read_csv(PROC / "ground_truth" / "early_warning_g30.csv")  # the evidence script's G30 series (u-blox 01)
    for rx in RXS:  # for loop over the receivers
        res, _, _, _ = D.residual_series(orb, kl, gt, "2.6.5", rx, site)  # the evidence script's residuals (GPS, Galileo), computed again by the evidence script's own function
        D.LINES.clear()  # frees memory
        res = res[(res.sec >= a) & (res.sec < b)]  # the window
        mine = sat[(sat.test_code == "2.6.5") & (sat.receiver == rx) & (sat.sec >= a) & (sat.sec < b) & sat.pr_res_m.notna() & sat.sys.isin(["G", "E"])]  # my residuals, GPS and Galileo
        m = res.merge(mine, on=["sec", "sv"], how="outer", indicator=True)  # outer join to see rows only one side has
        both = m[m._merge == "both"]  # rows both have
        d = (both.res - both.pr_res_m).abs().max()  # the largest difference
        check("the evidence script equality", f"2.6.5 {rx} 12:45:00-12:54:24: the evidence script rows, my rows, rows only on one side, max |difference| (m, rounding 5e-5)", f"{len(res)}, {len(mine)}, {int((m._merge != 'both').sum())}, {d:.5f}", (m._merge != "both").sum() == 0 and d < 1e-3)  # must agree
        if rx == "UBX_01":  # if: the receiver of the G30 file
            my = mine[mine.sv == "G30"].set_index("sec").pr_res_m  # my G30 residual
            gs = g30.set_index("sec").g30_res_m  # the G30 residual of the evidence script's case study (clock without G30)
            j = pd.concat([my, gs], axis=1, keys=["mine", "g30"]).dropna()  # common seconds
            pre = j[(j.index < T("12:48:40"))]  # before the anomaly, where the two clocks coincide
            check("the evidence script equality", "G30 residual on u-blox 01, before 12:48:40: seconds, max |mine - the evidence script's G30 case study| (m); whole window max (m)", f"{len(pre)}, {(pre.mine - pre.g30).abs().max():.3f}; {(j.mine - j.g30).abs().max():.3f}", (pre.mine - pre.g30).abs().max() < 1.0)  # the two clock definitions differ by a fraction of a metre before the anomaly
            dopm = sat[(sat.test_code == "2.6.5") & (sat.receiver == rx) & (sat.sv == "G30")].set_index("sec").dop_res_hz  # my G30 Doppler residual
            dp = pd.concat([dopm, g30.set_index("sec").g30_dop_err_hz], axis=1, keys=["mine", "g30"]).dropna()  # common seconds
            dp = dp[dp.index < T("12:48:40")]  # before the anomaly
            check("the evidence script equality", "G30 Doppler error on u-blox 01, before 12:48:40: seconds, median and max |mine - the evidence script's G30 case study| (Hz); the two differ in the drift definition", f"{len(dp)}, {(dp.mine - dp.g30).abs().median():.3f}, {(dp.mine - dp.g30).abs().max():.3f}", None)  # for reading
    # spot plot 1: G30 on u-blox 01
    import matplotlib  # the plotting library
    matplotlib.use("Agg")  # file-only backend
    import matplotlib.pyplot as plt  # pyplot
    w = sat[(sat.test_code == "2.6.5") & (sat.receiver == "UBX_01") & (sat.sv == "G30") & (sat.sec >= a) & (sat.sec < b)]  # the G30 rows
    fig, ax = plt.subplots(3, 1, figsize=(8, 7), sharex=True)  # three stacked panels
    t = (w.sec - a) / 60.0  # minutes from 12:45:00
    ax[0].plot(t, w.pr_res_m, label="this table"); ax[0].plot((g30.sec - a) / 60.0, g30.g30_res_m, "--", label="the evidence script's G30 case study")  # code residual against the evidence script's  # noqa: E702
    ax[0].set_ylabel("G30 code residual (m)"); ax[0].legend()  # noqa: E702
    ax[1].plot(t, w.cn0); ax[1].set_ylabel("G30 C/N0 (dB-Hz)")  # noqa: E702
    ax[2].plot(t, w.dop_res_hz, label="this table"); ax[2].plot((g30.sec - a) / 60.0, g30.g30_dop_err_hz, "--", label="the evidence script's G30 case study"); ax[2].set_ylabel("G30 Doppler residual (Hz)"); ax[2].set_xlabel("minutes after 12:45:00 UTC"); ax[2].legend()  # noqa: E702
    save_fig(fig, CHECK_DIR / "observables_spot_g30.png")  # saved
    # spot plot 2: satellites tracked per constellation, mosaic-X5, all sessions
    fig, axs = plt.subplots(6, 1, figsize=(9, 11))  # one panel per session
    for ax_, test in zip(axs, SESSIONS):  # for loop over the sessions
        e = ep[(ep.test_code == test) & (ep.receiver == "SBF_02")]  # the X5's seconds
        for s_, col in zip("GREC", ("#0072B2", "#D55E00", "#009E73", "#CC79A7")):  # for loop over the constellations
            ax_.plot((e.sec - e.sec.min()) / 60.0, e["n_" + s_], color=col, label=s_, lw=0.8)  # tracked count
        for gname, gcol in (("jamming", "#E69F00"), ("spoofing", "#CC0000"), ("clean", "#228822")):  # for loop over the groups
            sel = (e.group_provisional == gname).to_numpy()  # the group's seconds
            ax_.scatter(((e.sec - e.sec.min()) / 60.0)[sel], np.full(int(sel.sum()), -1.0), s=2, color=gcol)  # a coloured strip under the lines
        ax_.set_ylabel(f"{test}"); ax_.set_ylim(-2, 22)  # noqa: E702
    axs[0].legend(ncol=4, fontsize=7); axs[-1].set_xlabel("minutes from the start of the span (strip: green clean, orange jamming, red spoofing, PROVISIONAL)")  # noqa: E702
    save_fig(fig, CHECK_DIR / "observables_spot_counts_x5.png")  # saved


def missingness(sat: pd.DataFrame, ep: pd.DataFrame) -> None:  # function: what exists, by session, receiver and provisional group
    rows = []  # output rows
    for (t, r, g), e in ep.groupby(["test_code", "receiver", "group_provisional"]):  # for loop over every session, receiver and group
        s = sat[(sat.test_code == t) & (sat.receiver == r) & (sat.group_provisional == g)]  # the group's satellite rows
        main = s[s.sys.isin(["G", "E", "C"]) & (s.el_deg >= MASK)]  # the rows a residual could exist for
        row = {"test_code": t, "receiver": r, "group": g, "seconds": len(e), "seconds_tracked": int(e.has_observations.sum()), "seconds_solved": int(e.solver_ok.sum())}  # dict literal: the group's seconds and how many were tracked and solved
        row["seconds_with_dopres_ge4"] = int((e.n_dopres.fillna(0) >= 4).sum())  # seconds with at least four Doppler residuals
        row["seconds_with_prres_ge4"] = int((e.n_prres.fillna(0) >= 4).sum())  # seconds with at least four code residuals
        row["satellite_seconds"] = len(s)  # satellite-seconds in the group
        row["share_with_carrier"] = s.has_carrier.mean() if len(s) else np.nan  # conditional expression: share with a carrier observation
        row["share_with_doppler"] = s.has_doppler.mean() if len(s) else np.nan  # share with a Doppler observation
        row["share_masked_with_pr_residual"] = main.pr_res_m.notna().mean() if len(main) else np.nan  # of satellites above the mask, the share with a code residual
        row["share_masked_with_dop_residual"] = main.dop_res_hz.notna().mean() if len(main) else np.nan  # of satellites above the mask, the share with a Doppler residual
        row["share_lock_loss"] = s.lock_loss.mean() if len(s) else np.nan  # share of satellite-seconds with a carrier loss-of-lock flag
        row["glonass_satellite_seconds"] = int((s.sys == "R").sum())  # GLONASS satellite-seconds (observation-intrinsic quantities only)
        rows.append(row)  # list append: keeps the finished row
    pd.DataFrame(rows).round(4).to_csv(PROC / "features" / "observables_missingness.csv", index=False)  # written


def dictionary() -> None:  # function: the data dictionary
    d = []  # list: the dictionary rows, each a (table, column, unit, definition) tuple
    d.append(("both", "test_code, receiver, sec, utc", "", "Session, stream (SBF_02 mosaic-X5, SBF_03 mosaic-H main antenna, UBX_01 u-blox 01), UTC unix second and its clock text. 2.6.12 u-blox 01 is on true time by the host-time index."))  # list append of a tuple: one data-dictionary entry
    d.append(("both", "group_provisional", "", "PROVISIONAL group of the second: clean (the position script reference_clean_intervals), jamming (scheduled start to the end of the initial jamming), spoofing (to the scheduled end, 2.6.10 to 14:43:59, the end of spoofed tracking in 2.6.10), other. Final labels come from the label script (only independent evidence sets a label). Never a feature."))  # list append of a tuple: one data-dictionary entry
    d.append(("satellite", "sv, sys", "", "Satellite (G GPS, R GLONASS, E Galileo, C BeiDou) and its constellation letter. Bands: GPS C1C, GLONASS C1C, BeiDou C2I (B1I), Galileo C1C (Septentrio) or C1X (u-blox)."))  # list append of a tuple: one data-dictionary entry
    d.append(("satellite", "el_deg", "deg", "Elevation from the IGS orbit and the site reference position (5.2). Empty for GLONASS (no orbit), for satellites without a pseudorange or an orbit record."))  # list append of a tuple: one data-dictionary entry
    d.append(("satellite", "cn0", "dB-Hz", "S1C / S1X / S2I as recorded in the RINEX."))  # list append of a tuple: one data-dictionary entry
    d.append(("satellite", "dop_hz", "Hz", "D1C / D1X / D2I as recorded (RINEX sign: positive when the range decreases)."))  # list append of a tuple: one data-dictionary entry
    d.append(("satellite", "pr_res_m", "m", "Code residual: measured pseudorange minus the modelled range (IGS orbit and clock, Sagnac, Saastamoinen, Klobuchar) minus the median of that constellation at that second, satellites above 10 deg, at least four per constellation (the evidence script definition)."))  # list append of a tuple: one data-dictionary entry
    d.append(("satellite", "dop_res_hz", "Hz", "Measured minus modelled Doppler, minus the per-second receiver drift (median over satellites above 10 deg of the same difference in m/s, at least four). Needs the site reference position, so it is meaningful for a stationary receiver only."))  # list append of a tuple: one data-dictionary entry
    d.append(("satellite", "dC_m, d2C_m", "m/s, m/s^2", "Single and second time difference of the pseudorange over consecutive seconds (raw, the receiver clock drift is still in them)."))  # list append of a tuple: one data-dictionary entry
    d.append(("satellite", "cmc_rate_m_s", "m/s", "First difference of code minus carrier (C - lambda L) over consecutive seconds. GLONASS wavelength from the slot's channel (ground_truth/glonass_channels.csv)."))  # list append of a tuple: one data-dictionary entry
    d.append(("satellite", "lock_loss, lli_pr", "0/1", "RINEX loss-of-lock indicator, bit 0 of the carrier flag (lock_loss) and the code flag digit (lli_pr)."))  # list append of a tuple: one data-dictionary entry
    d.append(("satellite", "tracked_prev", "0/1", "1 if the satellite also has an L1 observation in the previous second."))  # list append of a tuple: one data-dictionary entry
    d.append(("satellite", "has_pr, has_carrier, has_doppler, has_cn0", "0/1", "Which observations exist. A row exists when any of the four does."))  # list append of a tuple: one data-dictionary entry
    d.append(("satellite", "no_model_reason", "", "Why pr_res_m / dop_res_hz are empty: no_orbit (GLONASS: a broadcast-orbit propagator is not yet built, register row), no_pseudorange, no_orbit_record, bds_geo (BeiDou GEO orbits not validated), below_mask, clock_n<4."))  # list append of a tuple: one data-dictionary entry
    d.append(("satellite", "known_invalid, geo", "0/1", "Known_invalid: the diagnosed invalid measurements (u-blox 01 G30 in 2.6.2 until 13:02:41; Septentrio G08 in 2.6.10, 14:44:41-14:44:43); they stay in the table but not in the epoch aggregates. geo: BeiDou GEO."))  # list append of a tuple: one data-dictionary entry
    d.append(("satellite", "dop_drift_m_s", "m/s", "The per-second Doppler-based receiver drift used for dop_res_hz."))  # list append of a tuple: one data-dictionary entry
    d.append(("epoch", "n_G, n_R, n_E, n_C, n_tracked", "", "Satellites with an L1-type observation, per constellation and in total (0 when nothing is tracked: the second is a row anyway)."))  # list append of a tuple: one data-dictionary entry
    d.append(("epoch", "d_n_*", "", "Change of those counts from the previous second."))  # list append of a tuple: one data-dictionary entry
    d.append(("epoch", "n_continuing, n_new, n_lost", "", "Satellites tracked in both seconds, newly tracked or re-appearing, and tracked before but not now."))  # list append of a tuple: one data-dictionary entry
    d.append(("epoch", "n_lock_loss, n_known_invalid, has_observations", "", "Counts of carrier lock-loss flags and of diagnosed invalid measurements; has_observations is 1 when n_tracked > 0."))  # list append of a tuple: one data-dictionary entry
    d.append(("epoch", "solver_ok, n_used, n_used_G, n_used_E, n_rej", "", "The position script's solver on GPS and Galileo (free position, one clock per constellation, 30 m rejection): solved or not, satellites kept and rejected."))  # list append of a tuple: one data-dictionary entry
    d.append(("epoch", "pdop, hdop", "", "Dilution of precision of the satellites the solver KEPT. During an attack it therefore reflects the 30 m rejection and not the full constellation."))  # list append of a tuple: one data-dictionary entry
    d.append(("epoch", "clk_gps_m, clk_gal_m, clk_gps_drift_m_s", "m, m, m/s", "Receiver clock bias on GPS and Galileo from the solver (metres of range) and the first difference of the GPS bias."))  # list append of a tuple: one data-dictionary entry
    d.append(("epoch", "h_off_m, v_off_m", "m", "Horizontal and vertical offset of the solver's own position from the 5.2 site reference (context for attacks that move the position)."))  # list append of a tuple: one data-dictionary entry
    d.append(("epoch", "dop_drift_m_s", "m/s", "The Doppler-based receiver drift of the second."))  # list append of a tuple: one data-dictionary entry
    d.append(("epoch", "cn0_*, dopres_*, prres_*, dC_*, d2C_*, cmc_*", "", "Per-second median, largest absolute value (maxabs), count (n_*) and, for C/N0 and the Doppler residual, standard deviation (std) over the main set: GPS, Galileo and BeiDou MEO/IGSO above 10 deg, without known_invalid rows."))  # list append of a tuple: one data-dictionary entry
    d.append(("epoch", "r_cn0_*, r_dC_*, r_d2C_*, r_cmc_*", "", "The same statistics for GLONASS, observation-intrinsic quantities only, with no elevation mask (no orbit)."))  # list append of a tuple: one data-dictionary entry

    pd.DataFrame(d, columns=["table", "column", "unit", "definition"]).to_csv(PROC / "features" / "observables_data_dictionary.csv", index=False)  # written


if __name__ == "__main__":  # guard: runs main() only when the file is executed, not when imported
    main()  # entry point
    exit_if_failed(FAILED, "4_features/01_observables.py")  # the helpers module: status 1 when any self-check failed, so a chained run stops
