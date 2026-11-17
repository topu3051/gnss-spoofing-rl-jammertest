#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
Early warning, exploration only. Nothing produced here is a feature (exclusion rule); receiver status flags are
comparison events and never inputs.

Part 1 is the G30 case study of session 2.6.5 on u-blox 01, 12:45:00-12:54:24 UTC at 1 Hz. G30's code residual is measured
C1C minus the Sagnac-corrected geometric range to the 2.6.2-site reference position, minus the satellite clock and the ionosphere and
troposphere corrections (the position script functions), minus a receiver clock. Primary clock: the median of the other GPS satellites (G30 left
out), 10 degree mask. Sensitivity clock: GPS and Galileo pooled, with the GPS-Galileo bias fixed from the 2.6.2 clean intervals of the
same receiver (preprocessing/reference_clean_intervals.csv) and never estimated inside the attack window.

Reads   Data/processed/preprocessing/reference_positions.csv, preprocessing/reference_clean_intervals.csv, ground_truth/ta3_schedule_confirmation.csv,
        ground_truth/receiver_status_timeseries.csv (gain, drawn and timed only), ground_truth/session_ground_truth.csv, Data/processed/checks/g30_onset.csv (comparison)
        Data/rinex/<session>/..._SBF_02, _SBF_03, _UBX_01 .crx.gz and the IGS merged broadcast file (orbits and clocks only)
Writes  Data/processed/ground_truth/early_warning_g30.csv        per-second G30 series with the clock, flags, Doppler and u-blox position
        Data/processed/ground_truth/early_warning_g30_tests.csv  tests T1-T5 and the checks, long form (item, quantity, value)
        Graph/g30_early_warning.png        the figure
"""

from __future__ import annotations  # future import: type hints kept as text (a convention across the project's scripts)

import os  # operating-system access: MPLCONFIGDIR is set before matplotlib loads
import random  # standard-library random; seeded below although nothing here draws from it
import sys  # standard-library system access: sys.path
import datetime as dtm  # standard-library dates, for the day of the recording
from datetime import datetime, timezone  # date-time classes, to print clock times
from pathlib import Path  # Path objects for building file paths

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
os.environ["MPLCONFIGDIR"] = str(PROJECT / "codes" / ".mplconfig")  # matplotlib's cache inside the project folder

import numpy as np  # NumPy: arrays of times and readings
import pandas as pd  # pandas: reads the CSVs and handles timestamps

sys.path.insert(0, str(PROJECT / "codes" / "common"))  # puts codes/ first on the import path so the helpers module is found
from helpers import exit_if_failed, save_fig, decision_times, known_invalid as known_invalid_rule  # byte-stable figure saving; the decision-times table and the one known-invalid rule; exit_if_failed ends the run on a failed check

SEED = 20260914  # integer constant: the project's fixed random seed
random.seed(SEED)  # seeds Python's generator (nothing here draws from it)
np.random.seed(SEED)  # seeds NumPy's generator likewise

PROC = PROJECT / "Data" / "processed"  # where the CSVs are read and written
GRAPH = PROJECT / "Graph"  # where the figures go
DATE = "2026-09-14"  # the one recording date of all six sessions
DT = decision_times(PROC)  # dict: decision instants from Data/processed/ground_truth/decision_times.csv (the decision-time script), so none is typed here
LAMBDA_L1 = 299792458.0 / 1575.42e6  # float constant: the L1 wavelength in metres, to turn a Doppler into a range rate
ANOM_M = 30.0  # float constant: a residual above this many metres counts as an anomaly
ANOM_RUN = 3  # integer constant: consecutive seconds above ANOM_M that make an onset
POS_M = 50.0  # float constant: horizontal distance from the site that counts as a position deviation
MASK_DEG = 10.0  # float constant: elevation mask for the clock and the counts, as in the position script
ERR_END_M = 20.0  # float constant: the G30 error counts as ended when |residual| is below this (as g30's first_normal)
WIN_START, WIN_END = "12:45:00", "12:54:24"  # strings: the G30 case-study window of 2.6.5
FIT_A0, FIT_A1 = "12:48:46", "12:50:59"  # strings: the pre-passage window of T1 and T4
X5_ONSET, UBX_ONSET = "12:50:59", "12:51:01.9"  # strings: gain onsets of the X5 and u-blox 01 (ground_truth/ta3_schedule_confirmation.csv)

ROWS: list = []  # list of long-form output rows: item, quantity, value
CHECKS: list = []  # list of self-check rows
FAILED: list = []  # list: names of the self-checks that failed; the script ends the run with status 1 when it is not empty


def put(item: str, quantity: str, value) -> None:  # function: records one number and prints it
    ROWS.append({"item": item, "quantity": quantity, "value": value})  # one long-form row
    print(f"  {item:30s} {quantity:62s} {value}")  # printed for the log


def check(name: str, expected, found, ok: bool) -> None:  # function: records one self-check and prints it
    CHECKS.append({"item": "check", "quantity": name, "value": f"expected {expected}; found {found}; {'ok' if ok else 'FAIL'}"})  # one row
    if ok is not None and not bool(ok):  # if: a pass/fail check (None marks a reported number) that did not pass
        FAILED.append(name)  # its name kept for the exit at the end of the run
    print(f"  [{'ok' if ok else 'FAIL'}] {name}: expected {expected}, found {found}")  # printed for the log


def T(hms: str) -> float:  # function: "HH:MM:SS(.f)" on the recording date -> unix seconds
    return pd.Timestamp(f"{DATE} {hms}", tz="UTC").timestamp()  # a UTC timestamp as seconds


def hms(sec: float) -> str:  # function: unix seconds -> HH:MM:SS clock text
    return datetime.fromtimestamp(sec, timezone.utc).strftime("%H:%M:%S")  # clock text


def load_module(name: str, filename: str):  # function: loads codes/<filename> as a module (names start with a digit)
    import importlib.util  # standard-library tools for loading a Python file by its path
    spec = importlib.util.spec_from_file_location(name, PROJECT / "codes" / filename)  # where to load from
    mod = importlib.util.module_from_spec(spec)  # an empty module object
    sys.modules[name] = mod  # registered
    spec.loader.exec_module(mod)  # runs the file's top level (its main() is guarded and not run)
    return mod  # the module


M = load_module("reference_positions_for_gain_evidence", "2_preprocessing/03_reference_positions.py")  # the position script: orbits, corrections, geometry, solver, observation paths
S04 = M.load_module("convert_rinex_for_gain_evidence", "2_preprocessing/01_convert_rinex.py")  # the conversion script, for the orbit functions and the IGS file path
LINES: dict = {}  # dict: decompressed RINEX lines per file, so each file is decompressed once per run


def rinex_lines(path: Path) -> tuple:  # function: the lines of a RINEX file and the index where its data start, cached
    if path not in LINES:  # if: not yet decompressed
        import hatanaka  # the compression package (Hatanaka plus gzip)
        txt = hatanaka.decompress(path)  # plain RINEX in memory (bytes)
        LINES[path] = (txt.decode("utf-8", errors="replace") if isinstance(txt, bytes) else txt).split("\n")  # its lines
    lines = LINES[path]  # the cached lines
    codes, cur, k = {}, None, 0  # observation codes per constellation; current constellation; header end
    for k, l in enumerate(lines):  # for loop over the header lines
        lab = l[60:].strip()  # the header label
        if lab == "SYS / # / OBS TYPES":  # if: an observation-types record
            if l[0] != " ":  # if: a new constellation
                cur = l[0]  # its letter
                codes[cur] = l[7:60].split()  # its first codes
            else:  # else: a continuation line
                codes[cur] += l[7:60].split()  # more codes
        elif lab == "END OF HEADER":  # elif: the header ends
            break  # stop
    return lines, k + 1, codes  # the lines, first data line, observation codes


def read_full(path: Path, spans: list, rx: str) -> pd.DataFrame:  # function: GPS and Galileo L1/E1 code, carrier, Doppler, C/N0 and lock flags inside the spans, first epoch of each UTC second
    lines, k0, codes = rinex_lines(path)  # the file's lines and codes
    want = M.OBS_CODES[rx[:3]]  # the code letters per constellation for this receiver family (C1C, or C1X for u-blox Galileo)
    cols = {}  # per constellation: column index of code, carrier, Doppler, C/N0
    for s in "GE":  # for loop over the two constellations used
        c = want[s]  # the pseudorange code, e.g. C1C
        names = [c, "L" + c[1:], "D" + c[1:], "S" + c[1:]]  # the four related observation codes: code, carrier, Doppler, C/N0
        cols[s] = [codes[s].index(n) if n in codes[s] else None for n in names]  # their columns (None if absent)
    lo, hi = [a for a, _ in spans], [b for _, b in spans]  # span starts and ends in unix seconds
    rows, keep, last, sec, t_gps = [], False, None, 0, 0.0  # kept rows; epoch state
    for l in lines[k0:]:  # for loop over the data lines
        if l[:1] == ">":  # if: an epoch line
            t_gps = (datetime(int(l[2:6]), int(l[7:9]), int(l[10:12]), int(l[13:15]), int(l[16:18]), tzinfo=timezone.utc) - M.GPS_EPOCH).total_seconds() + float(l[18:29])  # GPS seconds
            t_unix = t_gps - M.LEAP_SECONDS + M.GPS_UNIX  # UTC as unix seconds
            sec = int(np.floor(t_unix))  # its UTC second
            keep = sec != last and any(a <= t_unix < b for a, b in zip(lo, hi))  # the first epoch of a second inside a span
            if keep:  # if: kept
                last = sec  # this second is taken
        elif keep and l[:1] in cols:  # elif: a satellite line of a kept epoch for a constellation we use
            vals, lli = [], []  # the four values and their lock flags
            for i in cols[l[0]]:  # for loop over the four columns
                f = l[3 + 16 * i:3 + 16 * i + 16] if i is not None else ""  # the 16-character field: 14 value, LLI, SSI
                vals.append(float(f[:14]) if f[:14].strip() else np.nan)  # the value or NaN
                lli.append(f[14:15].strip() if len(f) > 14 else "")  # the loss-of-lock flag
            if not np.isnan(vals[0]):  # if: a pseudorange exists
                rows.append((sec, t_gps, l[:3], vals[0], vals[1], vals[2], vals[3], lli[0], lli[1]))  # one observation row
    return pd.DataFrame(rows, columns=["sec", "t_gps", "sv", "pr", "cp", "dop", "cn0", "lli_pr", "lli_cp"])  # the table


def epoch_z(orb, obs: pd.DataFrame, x: np.ndarray, kl: tuple, with_rate: bool = False) -> pd.DataFrame:  # function: per satellite and second, measured minus modelled range (clock still in), elevation and optionally the model Doppler
    out = []  # rows: second, satellite, z, elevation, true Doppler
    for sec, g in obs.groupby("sec", sort=True):  # for loop over the UTC seconds
        svs, t, pr = list(g.sv), float(g.t_gps.iloc[0]), g.pr.to_numpy()  # satellites, GPS time, pseudoranges
        pos, dts, ok = orb.state(svs, t, pr)  # positions and clock offsets at transmission time
        sel = np.flatnonzero(ok)  # usable satellites
        if len(sel) == 0:  # if: none usable
            continue  # next second
        rho, _, el, az = M.geometry(x, pos[sel])  # Sagnac-corrected range, elevation, azimuth
        svsel = [svs[i] for i in sel]  # the usable satellites' names
        z = pr[sel] - rho + M.C_LIGHT * dts[sel] - M.corrections(x, t, svsel, el, az, kl)  # observed minus modelled, receiver clock still in
        if with_rate:  # if: the model Doppler is wanted
            pa, da, _ = orb.state(svs, t + 0.5, pr)  # positions and clocks half a second later
            pb, db, _ = orb.state(svs, t - 0.5, pr)  # and half a second earlier
            ea = M.geometry(x, pa[sel])[0] - M.C_LIGHT * da[sel]  # clock-corrected effective range after
            eb = M.geometry(x, pb[sel])[0] - M.C_LIGHT * db[sel]  # clock-corrected effective range before
            dtrue = -(ea - eb) / LAMBDA_L1  # RINEX sign: Doppler is positive when the range decreases, so D = -rate / lambda
        else:  # else: no Doppler
            dtrue = np.full(len(sel), np.nan)  # empty
        for j, i in enumerate(sel):  # for loop over the usable satellites
            out.append((int(sec), svs[i], float(z[j]), float(np.degrees(el[j])), float(dtrue[j])))  # one row
    return pd.DataFrame(out, columns=["sec", "sv", "z", "el", "dop_true"])  # the table


def gps_galileo_bias(orb, kl: tuple, rx: str, site: np.ndarray, gt: pd.DataFrame) -> tuple:  # function: the GPS-Galileo clock bias of one receiver from the 2.6.2 clean intervals (median of per-epoch differences), its spread and epoch count
    ci = pd.read_csv(PROC / "preprocessing" / "reference_clean_intervals.csv")  # the clean intervals of the position script
    ci = ci[(ci.test_code == "2.6.2") & (ci.receiver == rx) & (ci.minutes > 0)]  # this receiver's kept 2.6.2 intervals
    spans = [(T(a[11:]), T(b[11:]) + 1) for a, b in zip(ci.start_utc, ci.end_utc)]  # each as unix seconds, the last second included
    obs = read_full(M.obs_path(gt.loc["2.6.2", "session_folder"], "2.6.2", rx), spans, rx)  # its observations
    z = epoch_z(orb, obs, site, kl)  # the per-satellite z values
    z = z[(z.el >= MASK_DEG) & (z.sv != "G30")]  # the mask and G30 left out (same rule as the clock)
    g = z[z.sv.str[0] == "G"].groupby("sec").z.median()  # GPS clock per epoch
    e = z[z.sv.str[0] == "E"].groupby("sec").z.median()  # Galileo clock per epoch
    d = (e - g).dropna()  # the bias per epoch (Galileo minus GPS), metres
    return float(d.median()), float(d.quantile(0.05)), float(d.quantile(0.95)), len(d)  # median, 5th and 95th percentile, epochs


def residual_table(z: pd.DataFrame, bias: float) -> pd.DataFrame:  # function: per-second clocks (primary and pooled), residuals, other-satellite counts and the reliability flag, from the z table
    z = z.copy()  # work on a copy
    z["sys"] = z.sv.str[0]  # constellation letter
    z["zb"] = np.where(z.sys == "E", z.z - bias, z.z)  # Galileo shifted by the fixed bias onto the GPS clock
    m = (z.el >= MASK_DEG) & (z.sv != "G30")  # satellites that may enter a clock: above the mask, G30 never
    clk = z[m & (z.sys == "G")].groupby("sec").z.agg(["median", "count"])  # primary clock: median of the other GPS satellites
    clkp = z[m].groupby("sec").zb.median().rename("clk_pooled")  # pooled GPS+Galileo clock with the bias removed
    z = z.merge(clk.rename(columns={"median": "clk", "count": "n_other_gps"}), on="sec", how="left").merge(clkp, on="sec", how="left")  # clocks attached to every row
    z["res"] = z.zb - z.clk  # residual about the primary clock (Galileo bias-corrected)
    z["res_pooled"] = z.zb - z.clk_pooled  # residual about the pooled clock
    z["over"] = m & (z.res.abs() > ANOM_M)  # an 'other' satellite beyond the limit
    cnt = z[m].groupby("sec").agg(n_over=("over", "sum"), n_other_gal=("sys", lambda s: int((s == "E").sum())))  # counts per second
    z = z.merge(cnt, on="sec", how="left")  # attached
    z["n_other_gps"] = z.n_other_gps.fillna(0)  # no GPS satellites counts as zero
    z["unreliable"] = (z.n_over > 1) | (z.n_other_gps < 4)  # unreliable clock reference: more than one other satellite beyond 30 m, or fewer than four GPS references
    return z  # the table


def onset(sec: np.ndarray, flag: np.ndarray, run: int = ANOM_RUN) -> float | None:  # function: the first second of the first run of `run` consecutive seconds with flag true
    s = pd.Series(flag.astype(int), index=sec)  # flags indexed by second
    full = s.reindex(range(int(sec.min()), int(sec.max()) + 1), fill_value=0)  # missing seconds count as not flagged
    r = full.rolling(run).sum() == run  # true at the end of each full run
    ends = full.index[r.to_numpy()]  # run-ending seconds
    return float(ends[0] - run + 1) if len(ends) else None  # the start of the first run


def fit(t: np.ndarray, y: np.ndarray) -> dict:  # function: straight-line fit with standard error and rms of the fit
    n = len(t)  # sample size
    if n < 3:  # if: too few points
        return {"n": n, "slope": np.nan, "se": np.nan, "rms": np.nan}  # nothing to report
    p = np.polyfit(t - t[0], y, 1)  # slope and intercept
    r = y - np.polyval(p, t - t[0])  # fit residuals
    se = float(np.sqrt((r ** 2).sum() / (n - 2) / ((t - t.mean()) ** 2).sum()))  # standard error of the slope
    return {"n": n, "slope": float(p[0]), "se": se, "rms": float(np.sqrt((r ** 2).mean()))}  # the fit


def enu_dist(x: np.ndarray, site: np.ndarray) -> tuple:  # function: horizontal and vertical offset of a position from the site, metres
    lat, lon, _ = M.ecef_to_llh(*site)  # the site's geodetic position
    d = M.enu_axes(lat, lon) @ (x - site)  # the offset in east, north, up
    return float(np.hypot(d[0], d[1])), float(d[2])  # horizontal and vertical


def main_part1() -> None:  # function: part 1, the G30 case study
    gt = pd.read_csv(PROC / "ground_truth" / "session_ground_truth.csv").set_index("test_code")  # session folders
    pos = pd.read_csv(PROC / "preprocessing" / "reference_positions.csv")  # the position script positions
    r = pos[(pos.scope == "site") & (pos.site == "2.6.2 site")].iloc[0]  # the 2.6.2 site's median-of-medians row
    site = np.array([r.x_m, r.y_m, r.z_m])  # its ECEF position
    orb = M.Orbits(S04, dtm.date(2026, 9, 14))  # satellite orbits and clocks from the IGS file only
    kl = M.igs_klobuchar(S04.IGS_BRDC)  # the GPS Klobuchar parameters
    span = [(T(WIN_START), T(WIN_END) + 1)]  # the case-study window as unix seconds, the last second included
    ta3 = pd.read_csv(PROC / "ground_truth" / "ta3_schedule_confirmation.csv")  # the receiver-status script's gain onsets
    ta3 = ta3[ta3.test_code == "2.6.5"].set_index("receiver").onset_utc  # 2.6.5 onsets by receiver
    check("gain onsets used here match ground_truth/ta3_schedule_confirmation.csv", f"{X5_ONSET} / {UBX_ONSET}", f"{ta3['Septentrio_02']} / {ta3['u-blox_01']}", ta3["Septentrio_02"] == X5_ONSET + ".000" and ta3["u-blox_01"] == UBX_ONSET + "00")  # constants agree with the file
    own_gain = {"SBF_02": T(ta3["Septentrio_02"]), "SBF_03": T(ta3["Septentrio_03"]), "UBX_01": T(ta3["u-blox_01"])}  # each receiver's own gain onset, unix seconds
    folder = gt.loc["2.6.5", "session_folder"]  # the session folder of 2.6.5
    # ---- GPS-Galileo biases from the 2.6.2 clean intervals, never from the attack window ----
    bias = {}  # receiver -> bias in metres
    print("GPS-Galileo bias from the 2.6.2 clean intervals")  # heading
    for rx in ("UBX_01", "SBF_02", "SBF_03"):  # for loop over the three receivers
        b, lo, hi, n = gps_galileo_bias(orb, kl, rx, site, gt)  # estimate
        bias[rx] = b  # kept
        put(f"bias {rx}", "Galileo minus GPS clock, median / p5 / p95 (m), epochs", f"{b:.2f} / {lo:.2f} / {hi:.2f}, {n}")  # reported
    # ---- u-blox 01: observations, z, residuals ----
    obs = read_full(M.obs_path(folder, "2.6.5", "UBX_01"), span, "UBX_01")  # u-blox 01 observations in the window
    z = epoch_z(orb, obs, site, kl, with_rate=True)  # per-satellite z values and model Doppler
    z = residual_table(z, bias["UBX_01"])  # clocks, residuals, counts, flag
    z = z.merge(obs[["sec", "sv", "cp", "dop", "cn0", "lli_pr", "lli_cp"]], on=["sec", "sv"], how="left")  # carrier, Doppler, C/N0, flags attached
    ok_bias = z[(z.sec < T(FIT_A0)) & (z.sys == "E") & (z.el >= MASK_DEG)].groupby("sec").zb.median() - z[(z.sec < T(FIT_A0)) & (z.sys == "G") & (z.el >= MASK_DEG) & (z.sv != "G30")].groupby("sec").z.median()  # Galileo minus GPS clock before the passage, with the bias already removed from Galileo: should be about zero
    check("2.6.5 pre-passage Galileo-GPS clock difference after removing the 2.6.2 bias (median |m|, check only)", "small (< 3 m)", f"{float(ok_bias.dropna().median()):.2f}", abs(float(ok_bias.dropna().median())) < 3.0)  # the bias carries over to 2.6.5
    g30 = z[z.sv == "G30"].sort_values("sec").reset_index(drop=True)  # G30's rows
    print(f"u-blox 01 G30 observed in {len(g30)} of {int(T(WIN_END) - T(WIN_START) + 1)} seconds; elevation {g30.el.min():.1f}-{g30.el.max():.1f} deg")  # coverage line
    # ---- Doppler: the model, its sign convention and the receiver's clock drift ----
    z["dd"] = z.dop - z.dop_true  # measured minus model Doppler: the receiver clock drift plus any error
    others = z[(z.sv != "G30") & (z.sys == "G") & (z.el >= MASK_DEG)]  # clean-reference GPS satellites
    drift = others.groupby("sec").dd.median().rename("drift")  # per-second receiver clock drift term in Hz (median over the others)
    z = z.merge(drift, on="sec", how="left")  # attached
    z["dop_err"] = z.dd - z.drift  # Doppler error after removing the clock drift
    clean = z[(z.sv != "G30") & (z.sys == "G") & (z.el >= MASK_DEG) & (z.sec < T(FIT_A0))]  # clean satellites before the passage
    wrong = (clean.dop + clean.dop_true).abs()  # what the opposite sign convention would give
    put("Doppler convention", "RINEX: D positive when the range decreases, D = -rate/lambda; clean GPS sats, median |D - D_model - drift| (Hz)", f"{clean.dop_err.abs().median():.3f}")  # the convention as used
    put("Doppler convention", "opposite sign, median |D + D_model| (Hz) for the same satellites", f"{wrong.median():.0f}")  # the wrong convention is off by kHz
    check("Doppler sign convention holds on clean satellites", "median error < 1 Hz, wrong sign > 100 Hz", f"{clean.dop_err.abs().median():.3f} / {wrong.median():.0f}", clean.dop_err.abs().median() < 1.0 and wrong.median() > 100)  # decisive
    # carrier check, independent of the orbits: D should equal -d(carrier)/dt on a clean satellite
    cc = []  # per satellite differences
    for sv, gg in clean.groupby("sv"):  # for loop over the clean satellites
        gg = gg.dropna(subset=["cp", "dop"]).set_index("sec")  # carrier and Doppler present
        sec = gg.index.to_numpy()  # their seconds
        ok = np.isin(sec + 1, sec) & np.isin(sec - 1, sec)  # both neighbours present
        cp_p, cp_m = gg.cp.reindex(sec[ok] + 1).to_numpy(), gg.cp.reindex(sec[ok] - 1).to_numpy()  # carrier a second later and earlier
        cc.append(pd.DataFrame({"d": gg.dop.reindex(sec[ok]).to_numpy(), "dcp": (cp_p - cp_m) / 2.0}))  # Doppler and central carrier rate (cycles/s)
    cc = pd.concat(cc) if cc else pd.DataFrame(columns=["d", "dcp"])  # all clean satellites
    cc = cc[(cc.d.abs() < 6000) & (cc.dcp.abs() < 6000)]  # drops slips and cycle-ambiguity jumps
    if len(cc):  # if: carrier data exist
        put("Doppler convention", "carrier rate: median |D + dL/dt| (Hz) / opposite sign median |D - dL/dt| (Hz), clean sats", f"{(cc.d + cc.dcp).abs().median():.3f} / {(cc.d - cc.dcp).abs().median():.0f}")  # D = -dL/dt
    # ---- G30 series ----
    g30 = z[z.sv == "G30"].sort_values("sec").reset_index(drop=True)  # G30's rows with everything attached
    ok_ = ~g30.unreliable  # reliable epochs
    g30["rate_res"] = np.nan  # central-difference rate of the residual, m/s, on reliable epochs with reliable neighbours
    s = g30.set_index("sec")  # indexed by second
    for sec in s.index:  # for loop over the seconds
        if sec + 1 in s.index and sec - 1 in s.index and not s.loc[[sec - 1, sec, sec + 1], "unreliable"].any():  # if: both neighbours exist and all three are reliable
            g30.loc[g30.sec == sec, "rate_res"] = (s.loc[sec + 1, "res"] - s.loc[sec - 1, "res"]) / 2.0  # the central difference
    g30["rate_dop"] = -LAMBDA_L1 * g30.dop_err  # the range rate the Doppler error implies: -lambda x (measured - true)
    # ---- clean-satellite check of the rate comparison (same statistic on every other GPS satellite, before the passage) ----
    cl_rows = []  # per-satellite median differences
    for sv, gg in clean.groupby("sv"):  # for loop over the clean satellites
        s2 = gg.set_index("sec")  # indexed by second
        d = []  # differences
        for sec in s2.index:  # for loop over the seconds
            if sec + 1 in s2.index and sec - 1 in s2.index and not s2.loc[[sec - 1, sec, sec + 1], "unreliable"].any():  # neighbours present and reliable
                d.append((s2.loc[sec + 1, "res"] - s2.loc[sec - 1, "res"]) / 2.0 + LAMBDA_L1 * s2.loc[sec, "dop_err"])  # residual rate minus the Doppler-implied rate
        if len(d) > 20:  # if: enough epochs
            cl_rows.append((sv, float(np.median(d)), float(np.median(np.abs(np.array(d) - np.median(d)))), len(d)))  # satellite, median, MAD, n
    cl = pd.DataFrame(cl_rows, columns=["sv", "median", "mad", "n"])  # the table
    put("T2 clean satellites", "residual rate minus Doppler-implied rate before the passage: median of per-satellite medians (m/s), MAD (m/s), satellites", f"{cl['median'].median():.3f}, {cl['mad'].median():.3f}, {len(cl)}")  # the baseline of the comparison
    # ---- T1 sign reversal and T2 code-Doppler consistency, on three post-passage readings ----
    a0, a1 = T(FIT_A0), T(FIT_A1)  # pre-passage window in unix seconds
    pre = g30[(g30.sec >= a0) & (g30.sec <= a1) & ok_]  # reliable epochs before the passage
    t_gain = T("12:51:00")  # the instant after which the post fit starts
    post_all = g30[(g30.sec > t_gain) & ok_]  # reliable epochs after 12:51:00
    first_rel = float(post_all.sec.iloc[0]) if len(post_all) else np.nan  # the first reliable epoch after 12:51:00
    back = post_all[post_all.res.abs() < ERR_END_M]  # epochs back within 20 m (a momentary return of the true signal)
    t_end = float(back.sec.iloc[0]) if len(back) else float(g30.sec.max()) + 1  # the error 'ends' at the first such epoch, else after the window
    t_rec = T(pd.read_csv(PROC / "ground_truth" / "ta3_schedule_confirmation.csv").query("test_code == '2.6.5' and receiver == 'Septentrio_02'").recovery_utc.iloc[0])  # the X5 gain recovery (the receiver-status script), the end of the passage: 12:52:21
    put("T1", "passage = X5 gain onset to X5 gain recovery (the receiver-status script)", f"{X5_ONSET} to {hms(t_rec)}; the literal window {hms(first_rel)}-{hms(t_end - 1)} lies inside it")  # why the literal window cannot test reversal
    seg = {  # dict: reading -> (first epoch, end exclusive, minimum |res|)
        "a literal, pre-specified (first reliable epoch after 12:51:00 until |res| < 20 m)": (first_rel, t_end, 0.0),  # the specification as written
        "b revised after inspection (reliable epochs from the X5 gain recovery to the end of capture with |res| > 30 m)": (t_rec, T(WIN_END) + 1, ANOM_M),  # the counterfeit tracked after the passage
        "b2 sensitivity of b (from the first reliable epoch after the last unreliable one, 12:52:57, with |res| > 30 m)": (float(g30[g30.unreliable].sec.max()) + 1, T(WIN_END) + 1, ANOM_M),  # drops the few non-counterfeit epochs at the end of the passage
        "c low-C/N0 signal during the passage (12:51:25-12:51:33), information only": (T("12:51:25"), T("12:51:34"), 0.0),  # a third behaviour seen in the series
    }
    f1 = fit(pre.sec.to_numpy(float), pre.res.to_numpy())  # the pre-passage fit
    put("T1", "pre-passage fit window, reliable epochs", f"{FIT_A0}-{FIT_A1}, n={f1['n']}")  # window
    put("T1", "pre-passage slope (m/s), s.e., rms of fit (m)", f"{f1['slope']:.3f}, {f1['se']:.3f}, {f1['rms']:.2f}")  # slope
    put("T1", "first epoch within 20 m after 12:51:00 (the 'error ends' rule)", f"{hms(float(back.sec.iloc[0])) if len(back) else 'none'} (C/N0 {float(back.cn0.iloc[0]) if len(back) else float('nan'):.0f} dB-Hz)")  # what ended the literal window
    for name, (s0, s1, m0) in seg.items():  # for loop over the three readings
        sub = g30[(g30.sec >= s0) & (g30.sec < s1) & ~g30.unreliable & (g30.res.abs() > m0)]  # reliable epochs in the reading's window with |res| above its minimum
        f2 = fit(sub.sec.to_numpy(float), sub.res.to_numpy())  # its fit
        opp = bool(np.isfinite(f1["slope"]) and np.isfinite(f2["slope"]) and f1["slope"] * f2["slope"] < 0)  # opposite to the pre-passage slope
        put("T1 " + name.split()[0], f"{name}: window {hms(s0)}-{hms(s1 - 1)}, n, slope (m/s), s.e., rms (m)", f"{f2['n']}, {f2['slope']:.3f}, {f2['se']:.3f}, {f2['rms']:.2f}")  # the fit
        put("T1 " + name.split()[0], "slope sign opposite to the pre-passage slope", opp)  # verdict
        sb = sub.dropna(subset=["rate_res", "rate_dop"])  # epochs with both rates
        d = sb.rate_res - sb.rate_dop  # residual rate minus Doppler-implied rate
        put("T2 " + name.split()[0], "reliable epochs, median(rate_res - rate_dop) m/s, MAD, median rate_res, median rate_dop", f"{len(sb)}, {d.median():.3f}, {(d - d.median()).abs().median():.3f}, {sb.rate_res.median():.3f}, {sb.rate_dop.median():.3f}")  # agreement
    sb = g30[(g30.sec >= a0) & (g30.sec <= a1)].dropna(subset=["rate_res", "rate_dop"])  # the pre-passage epochs with both rates
    d = sb.rate_res - sb.rate_dop  # the difference
    put("T2 pre", "pre-passage reliable epochs, median(rate_res - rate_dop) m/s, MAD, median rate_res, median rate_dop", f"{len(sb)}, {d.median():.3f}, {(d - d.median()).abs().median():.3f}, {sb.rate_res.median():.3f}, {sb.rate_dop.median():.3f}")  # agreement
    ramp = g30[(g30.sec >= a0) & (g30.sec <= T("12:51:05"))]  # the whole ramp up to the first track break
    put("carrier", "G30 carrier present, fraction of seconds before the ramp / during it", f"{g30[g30.sec < a0].cp.notna().mean():.2f} / {ramp.cp.notna().mean():.2f}")  # carrier presence
    lli = g30[(g30.sec >= T("12:48:00")) & (g30.sec <= a1) & (g30.lli_cp.fillna("").astype(str).str.strip() != "")]  # carrier loss-of-lock flags actually set (blank means none) before the gain onset
    put("carrier", "carrier LLI flags set 12:48:00-12:50:59: count, first second", f"{len(lli)}, {hms(lli.sec.iloc[0]) if len(lli) else 'none'}")  # LLI timing
    put("C/N0", "G30 C/N0 median before the ramp / during the ramp up to 12:51:05 (dB-Hz)", f"{g30[g30.sec < a0].cn0.median():.1f} / {ramp.cn0.median():.1f}")  # power change
    # ---- T3 lead time ----
    sec_all = g30.sec.to_numpy()  # G30's seconds
    on = onset(sec_all, (g30.res.abs() > ANOM_M).to_numpy())  # the anomaly onset by the 30 m / 3 s rule
    put("T3", "G30 anomaly onset (3 consecutive s with |residual| > 30 m)", hms(on) if on else "none in window")  # onset
    if on:  # if: found
        put("T3", "lead over u-blox 01 gain onset (s) = gain onset - anomaly onset", round(T(UBX_ONSET) - on, 1))  # positive: anomaly first
        put("T3", "lead over X5 gain onset (s)", round(T(X5_ONSET) - on, 1))  # positive: anomaly first
    # ---- comparison with g30's inter-receiver difference ----
    w = pd.read_csv(PROJECT / "Data" / "processed" / "checks" / "g30_onset.csv")  # g30's G30 error against the mosaic-H
    w = w[w.test == "2.6.5"].copy()  # 2.6.5 only
    w["sec"] = [T(h) for h in w.hms]  # unix seconds
    cmp_ = g30.merge(w[["sec", "err"]], on="sec", how="inner")  # common seconds
    cmp_ = cmp_[(cmp_.sec >= a0 - 120) & (cmp_.sec <= t_gain) & ~cmp_.unreliable]  # from two minutes before the ramp to 12:51:00, reliable epochs
    dd_ = (cmp_.res - cmp_.err).abs()  # difference between the two methods
    put("g30 comparison", "12:46:46-12:51:00 reliable common seconds; |res - g30 err| median / p95 / max (m)", f"{len(cmp_)}, {dd_.median():.2f} / {dd_.quantile(0.95):.2f} / {dd_.max():.2f}")  # the two methods agree
    cmp2 = g30.merge(w[["sec", "err"]], on="sec", how="inner")  # all common seconds
    cmp2 = cmp2[~cmp2.unreliable & (cmp2.sec >= a0 - 120) & (cmp2.sec <= t_gain)]  # reliable ones to 12:51:00
    cmp2["d"] = (cmp2.res - cmp2.err).abs()  # difference of the two methods
    big_ = cmp2[cmp2.d > 20]  # the large differences
    put("g30 comparison", "differences > 20 m: seconds, first, last; max difference inside 12:48:46-12:49:35 / 12:49:36-12:51:00 / before 12:48:46 (m)", f"{len(big_)}, {hms(big_.sec.min()) if len(big_) else '-'}, {hms(big_.sec.max()) if len(big_) else '-'}; {cmp2[(cmp2.sec >= a0) & (cmp2.sec <= T('12:49:35'))].d.max():.1f} / {cmp2[(cmp2.sec > T('12:49:35'))].d.max():.1f} / {cmp2[cmp2.sec < a0].d.max():.1f}")  # where the differences are
    w_on = onset(w.sec.to_numpy(), (w.err.abs() > 20).to_numpy()) if len(w) else None  # g30's first run of 3 s beyond 20 m
    put("g30 comparison", "first run of 3 s with |inter-receiver difference| > 20 m", hms(w_on) if w_on else "none")  # its onset
    # ---- sensitivity: pooled clock ----
    dif = (g30.res - g30.res_pooled).abs()[~g30.unreliable]  # primary minus pooled, reliable epochs
    put("clock sensitivity", "|primary - pooled| for G30, reliable epochs: median / p95 / max (m), epochs", f"{dif.median():.2f} / {dif.quantile(0.95):.2f} / {dif.max():.2f}, {len(dif)}")  # the difference
    put("clock sensitivity", "pooled-clock anomaly onset", hms(onset(sec_all, (g30.res_pooled.abs() > ANOM_M).to_numpy())) if onset(sec_all, (g30.res_pooled.abs() > ANOM_M).to_numpy()) else "none")  # onset with the pooled clock
    put("reliability", "G30 epochs flagged 'clock reference unreliable' / epochs", f"{int(g30.unreliable.sum())} / {len(g30)}")  # counts
    put("reliability", "all other-satellite epochs with more than one beyond 30 m (seconds in window)", int(z.drop_duplicates("sec").n_over.gt(1).sum()))  # seconds
    # ---- T4 the other receivers ----
    for rx in ("SBF_02", "SBF_03"):  # for loop over X5 and mosaic-H
        o = read_full(M.obs_path(folder, "2.6.5", rx), span, rx)  # observations
        zz = residual_table(epoch_z(orb, o, site, kl), bias[rx])  # residuals with the same clock rule
        win = zz[(zz.sec >= a0) & (zz.sec <= a1) & (zz.el >= MASK_DEG)]  # the pre-passage window above the mask
        mx = win.groupby("sv").res.agg(lambda s: float(s.abs().max()))  # largest |residual| per satellite
        nb = win[win.res.abs() > ANOM_M].sv.nunique()  # satellites beyond 30 m at least once
        gx = zz[(zz.sv == "G30") & (zz.sec >= a0) & (zz.sec <= a1)]  # G30 on this receiver
        put(f"T4 {M.RECEIVERS[rx]}", "satellites above mask / with |res| > 30 m at least once / worst satellite and its max |res| (m)", f"{win.sv.nunique()} / {nb} / {mx.idxmax()} {mx.max():.1f}")  # verdict numbers
        put(f"T4 {M.RECEIVERS[rx]}", "G30: epochs, median res (m), max |res| (m), elevation (deg)", f"{len(gx)}, {gx.res.median():.2f}, {gx.res.abs().max():.2f}, {gx.el.median():.1f}")  # G30 here
        on_x = onset(gx.sec.to_numpy(), (gx.res.abs() > ANOM_M).to_numpy()) if len(gx) else None  # this receiver's G30 onset by the same rule
        put(f"T4 {M.RECEIVERS[rx]}", "G30 seconds with |res| > 30 m / onset (3 s rule) / first second beyond 30 m", f"{int((gx.res.abs() > ANOM_M).sum())} / {hms(on_x) if on_x else 'none'} / {hms(gx[gx.res.abs() > ANOM_M].sec.iloc[0]) if (gx.res.abs() > ANOM_M).any() else 'none'}")  # onset and extent
        beyond = gx[gx.res.abs() > ANOM_M]  # G30 seconds beyond 30 m on this receiver
        put(f"T4 {M.RECEIVERS[rx]}", "G30 last second beyond 30 m / lead of the 3 s onset over this receiver's own gain onset (s)", f"{hms(beyond.sec.iloc[-1]) if len(beyond) else 'none'} / {round(own_gain[rx] - on_x, 1) if on_x else 'none'}")  # extent and lead
        fx = fit(gx[gx.sec >= a0].sec.to_numpy(float), gx[gx.sec >= a0].res.to_numpy())  # G30 slope on this receiver over the same window
        put(f"T4 {M.RECEIVERS[rx]}", "G30 slope over 12:48:46-12:50:59 (m/s), rms (m)", f"{fx['slope']:.3f}, {fx['rms']:.2f}")  # slope for comparison with u-blox
        oth = win[(win.sv != "G30") & (win.res.abs() > ANOM_M)]  # other satellites beyond 30 m
        put(f"T4 {M.RECEIVERS[rx]}", "satellites other than G30 beyond 30 m: count of satellites, seconds", f"{oth.sv.nunique()}, {len(oth)}")  # the 'or other' part of T4
        put(f"T4 {M.RECEIVERS[rx]}", "u-blox 01 on the same window: satellites with |res| > 30 m at least once", int(z[(z.sec >= a0) & (z.sec <= a1) & (z.el >= MASK_DEG) & (z.res.abs() > ANOM_M)].sv.nunique()))  # for contrast
    # ---- T5 the u-blox position from all its satellites ----
    pr_all = obs[["sec", "t_gps", "sv", "pr"]]  # the pseudoranges only
    ep, _ = M.solve_session(orb, pr_all, site, kl)  # the position script solver, all satellites (GPS and Galileo), rejection of residuals over 30 m on
    hv = [enu_dist(np.array([a, b, c]), site) for a, b, c in zip(ep.x, ep.y, ep.z)]  # horizontal and vertical offsets from the site
    ep["dh"], ep["dv"] = [h for h, _ in hv], [v for _, v in hv]  # attached
    first50 = ep[ep.dh > POS_M]  # epochs beyond 50 m
    put("T5", "epochs solved / window seconds; median horizontal offset before the passage (m)", f"{len(ep)} / {int(T(WIN_END) - T(WIN_START) + 1)}; {ep[ep.sec < a0].dh.median():.2f}")  # coverage
    put("T5", "first epoch with horizontal offset > 50 m / epochs beyond 50 m / their span", f"{hms(first50.sec.iloc[0]) if len(first50) else 'none in window'} / {len(first50)} / {hms(first50.sec.iloc[0]) + '-' + hms(first50.sec.iloc[-1]) if len(first50) else ''}")  # first deviation
    put("T5", "max horizontal offset in window (m) at", f"{ep.dh.max():.1f} at {hms(ep.loc[ep.dh.idxmax(), 'sec'])}")  # largest
    put("T5", "observations rejected by the 30 m rule before the passage / in 12:48:46-12:54:24; G30 among rejected, seconds", f"{int(ep[ep.sec < a0].n_rejected.sum())} / {int(ep[ep.sec >= a0].n_rejected.sum())}; {int(ep.rejected_sv.str.contains('G30').sum())}")  # rejection counts
    saved_reject = M.REJECT_M  # the position script rejection limit in force (30 m), kept so it can be put back exactly
    M.REJECT_M = 1.0e9  # sensitivity: switch the 30 m rejection off (module-level constant of the position script)
    try:  # try/finally: the finally block runs even if the solve raises, so the rejection can never stay switched off
        ep2, _ = M.solve_session(orb, pr_all, site, kl)  # the same solver with no rejection
    finally:  # always executed after the try block
        M.REJECT_M = saved_reject  # restored to the saved value, not to a typed 30.0
    hv2 = [enu_dist(np.array([a, b, c]), site) for a, b, c in zip(ep2.x, ep2.y, ep2.z)]  # offsets without rejection
    ep2["dh"] = [h for h, _ in hv2]  # attached
    f2_ = ep2[ep2.dh > POS_M]  # beyond 50 m without rejection
    put("T5 sensitivity", "no rejection: first epoch with horizontal offset > 50 m / epochs beyond 50 m / max offset (m)", f"{hms(f2_.sec.iloc[0]) if len(f2_) else 'none'} / {len(f2_)} / {ep2.dh.max():.1f}")  # sensitivity
    # ---- output series ----
    ser = g30.merge(ep[["sec", "dh", "dv", "n_rejected"]].rename(columns={"dh": "ubx_horiz_m", "dv": "ubx_vert_m", "n_rejected": "ubx_n_rejected"}), on="sec", how="outer").sort_values("sec")  # G30 series with the position columns
    ser["hms"] = [hms(s) for s in ser.sec]  # clock text
    cols = ["sec", "hms", "res", "res_pooled", "el", "n_other_gps", "n_other_gal", "n_over", "unreliable", "cn0", "lli_pr", "lli_cp", "cp", "dop", "dop_true", "dop_err", "rate_res", "rate_dop", "ubx_horiz_m", "ubx_vert_m", "ubx_n_rejected"]  # kept columns
    ser = ser[cols].rename(columns={"res": "g30_res_m", "res_pooled": "g30_res_pooled_m", "el": "g30_el_deg", "cn0": "g30_cn0_dbhz", "cp": "g30_carrier_cycles", "dop": "g30_dop_hz", "dop_true": "g30_dop_true_hz", "dop_err": "g30_dop_err_hz", "unreliable": "clock_unreliable"})  # clearer names
    ser["sec"] = ser.sec.astype(int)  # integer seconds
    ser.round(4).to_csv(PROC / "ground_truth" / "early_warning_g30.csv", index=False)  # the series
    pd.DataFrame(ROWS + CHECKS).to_csv(PROC / "ground_truth" / "early_warning_g30_tests.csv", index=False)  # the tests in long form
    plot_g30_early_warning(ser, on)  # the figure


def plot_g30_early_warning(ser: pd.DataFrame, on: float | None) -> None:  # function: the figure, G30 residual, C/N0 and the two gains on one time axis
    import matplotlib  # the plotting library
    matplotlib.use("Agg")  # file-only backend
    import matplotlib.pyplot as plt  # pyplot
    st = pd.read_csv(PROC / "ground_truth" / "receiver_status_timeseries.csv", low_memory=False)  # the status table (gain values, for the figure only)
    st = st[st.test_code == "2.6.5"].copy()  # 2.6.5 only
    st["sec"] = pd.to_datetime(st.timestamp_utc, utc=True).astype("int64") / 1e9  # unix seconds
    x5 = st[(st.receiver == "Septentrio_02") & (st.band == "GPSL1/E1")]  # the X5's GPS L1 / Galileo E1 gain
    ub = st[(st.receiver == "u-blox_01") & (st.band == "L1")]  # u-blox 01's L1 AGC count
    t0, t1 = T(WIN_START), T(WIN_END)  # the window
    rel = ser[(ser.clock_unreliable == False) & ser.g30_res_m.notna()]  # reliable G30 epochs
    unr = ser[(ser.clock_unreliable == True) & ser.g30_res_m.notna()]  # unreliable epochs, greyed
    fig, ax = plt.subplots(4, 1, figsize=(9, 8.6), sharex=True)  # four stacked panels on one time axis
    ax[0].plot(rel.sec, rel.g30_res_m, ".", ms=3, color="#0072B2", label="reliable clock reference")  # reliable residuals
    ax[0].plot(unr.sec, unr.g30_res_m, ".", ms=3, color="#BBBBBB", label="clock reference unreliable")  # unreliable, grey
    ax[0].axhline(30, color="k", lw=0.6, ls=":")  # the 30 m limit
    ax[0].axhline(-30, color="k", lw=0.6, ls=":")  # its mirror
    ax[0].set_ylim(-1600, 1500)  # frame the ramp; a few unreliable passage epochs lie outside
    n_out = int(((ser.g30_res_m > 1500) | (ser.g30_res_m < -1600)).sum())  # epochs outside the frame
    ax[0].text(0.01, 0.04, f"{n_out} epochs outside the frame (all in the passage, all flagged unreliable)", transform=ax[0].transAxes, fontsize=7)  # say so on the panel
    ax[0].set_ylabel("G30 residual (m)")  # label
    ax[0].legend(fontsize=7, loc="best")  # legend
    ax[1].plot(ser.sec, ser.g30_cn0_dbhz, ".", ms=3, color="#009E73")  # G30 C/N0
    ax[1].set_ylabel("G30 C/N0 (dB-Hz)")  # label
    ax[2].plot(x5.sec, x5.agc_gain_db, lw=0.8, color="#E69F00")  # X5 gain
    ax[2].set_ylabel("X5 gain (dB)")  # label
    ax[3].plot(ub.sec, ub.agc_cnt, lw=0.8, color="#CC79A7")  # u-blox AGC count
    ax[3].set_ylabel("u-blox 01 AGC count")  # label
    for a in ax:  # for loop over the panels
        a.axvline(T(X5_ONSET), color="#E69F00", lw=1.0, ls="--")  # X5 gain onset
        a.axvline(T(UBX_ONSET), color="#CC79A7", lw=1.0, ls="--")  # u-blox gain onset
        if on:  # if: an anomaly onset exists
            a.axvline(on, color="#D55E00", lw=1.0)  # G30 anomaly onset
        a.grid(color="#999999", alpha=0.3, lw=0.5)  # faint grid
        a.set_xlim(t0, t1 + 1)  # the window
    ticks = np.arange(t0, t1 + 1, 60)  # a tick each minute
    ax[3].set_xticks(ticks)  # set them
    ax[3].set_xticklabels([hms(t)[:5] for t in ticks], fontsize=7)  # as HH:MM
    ax[3].set_xlabel("UTC, 14 September 2026 (solid: G30 anomaly onset; dashed: X5 and u-blox gain onsets)")  # axis label
    fig.tight_layout()  # spacing
    save_fig(fig, GRAPH / "g30_early_warning.png")  # written, byte-stable


# =====================================================================================================================
# Part 2: the early-warning timeline, all sessions, three receivers
# =====================================================================================================================
SESSIONS = ["2.6.5", "2.6.2", "2.6.10", "2.6.4", "2.6.11", "2.6.12"]  # list: the six sessions in time order
RXS = ["SBF_02", "SBF_03", "UBX_01"]  # list: the three receivers (mosaic-X5, mosaic-H main antenna, u-blox 01)
RX_STATUS = {"SBF_02": "Septentrio_02", "SBF_03": "Septentrio_03", "UBX_01": "u-blox_01"}  # dict: stream -> the receiver tag of the status and the receiver-status script tables
FLAG_BANDS = {"SBF_02": ["GPSL1/E1", "GLOL1", "B1"], "SBF_03": ["GPSL1/E1", "GLOL1", "B1"]}  # dict: the main-antenna L1-type bands whose authentication flags are read (events only)
PERSIST = (3, 5, 10)  # tuple: the persistences tried, in seconds (3 -> 5 -> 10)
PCT = 99.9  # float constant: the percentile of the clean-interval distribution used as the threshold
HOLD_POS_S = 10.0  # float constant: seconds a receiver's own position must stay beyond 50 m (the receiver-status script's rule, so the 2.6.12 rows are comparable)
HOLD_SOL_S = 3.0  # float constant: seconds our own solver's position must stay beyond 50 m (the anomaly persistence)
SITE_START = {"2.6.5": DT["vehicle_motion_end"]}  # dict: 2.6.5 stands at the 2.6.2 site only from here (12:34:57, read from ground_truth/decision_times.csv); before it the receiver was moving
UBX_2612_END = "15:51:17"  # string: u-blox 01 in 2.6.12 carries a shifted time tag from here (the receiver-status script), so observations are modelled only before it
DATE_G30_NORMAL = DT["G30_normal"]  # (read from ground_truth/decision_times.csv) string: u-blox 01 reacquires the true G30 here in 2.6.2 (g30); before it G30 is an invalid measurement, not an attack signature
GAIN_REF_LAG, GAIN_REF_LEN = 30, 90  # integers: the gain reference is the median over the 90 s that ended 30 s ago


def ts(sec) -> str:  # function: unix seconds -> "HH:MM:SS" (or "HH:MM:SS.f" if fractional), or an empty string for no event
    if sec is None or (isinstance(sec, float) and not np.isfinite(sec)):  # if: no event
        return ""  # empty cell
    sec = float(sec)  # make it a float
    base = datetime.fromtimestamp(np.floor(sec), timezone.utc).strftime("%H:%M:%S")  # whole seconds as clock text
    return base if abs(sec - np.floor(sec)) < 1e-6 else base + f"{sec - np.floor(sec):.3f}"[1:]  # add the fraction only when there is one


def site_xyz(site: str) -> np.ndarray:  # function: a site's ECEF estimate from preprocessing/reference_positions.csv
    pos = pd.read_csv(PROC / "preprocessing" / "reference_positions.csv")  # the position script positions
    r = pos[(pos.scope == "site") & (pos.site == site)].iloc[0]  # the site row
    return np.array([r.x_m, r.y_m, r.z_m])  # x, y, z in metres


def capture_spans(gt: pd.DataFrame, test: str, rx: str) -> list:  # function: the observation span of a session and receiver, as (start, end) unix seconds
    a, b = T(gt.loc[test, "capture_start_utc"][11:]), T(gt.loc[test, "capture_end_utc"][11:]) + 1  # the capture, last second included
    if test in SITE_START:  # if: a session with a moving start
        a = max(a, T(SITE_START[test]))  # start where the receiver is at the site
    return [(a, b)]  # a single span


def clean_spans(test: str, rx: str) -> list:  # function: the kept clean intervals of a session and receiver, as (start, end+1) unix seconds
    ci = pd.read_csv(PROC / "preprocessing" / "reference_clean_intervals.csv")  # the position script intervals
    ci = ci[(ci.test_code == test) & (ci.receiver == rx) & (ci.minutes > 0)]  # kept ones for this session and receiver
    return [(T(a[11:]), T(b[11:]) + 1) for a, b in zip(ci.start_utc, ci.end_utc)]  # unix seconds, last second included


def in_spans(sec: np.ndarray, spans: list) -> np.ndarray:  # function: boolean mask of seconds inside any span
    m = np.zeros(len(sec), bool)  # nothing inside yet
    for a, b in spans:  # for loop over the spans
        m |= (sec >= a) & (sec < b)  # add this span
    return m  # the mask


def raw_z(orb, kl: tuple, obs: pd.DataFrame, site: np.ndarray) -> pd.DataFrame:  # function: per satellite and second, measured minus modelled range above the 10 degree mask, clock still in
    z = epoch_z(orb, obs, site, kl)  # measured minus modelled, receiver clock still in
    z = z[z.el >= MASK_DEG].copy()  # the elevation mask
    z["sys"] = z.sv.str[0]  # constellation letter
    return z[["sec", "sv", "z", "el", "sys"]].reset_index(drop=True)  # the table


def resid_mask(z: pd.DataFrame, mask: float) -> pd.DataFrame:  # function: residuals about a per-constellation median clock, using only satellites above `mask` degrees
    z = z[z.el >= mask].copy()  # the mask applies to the clock and to the residuals alike
    grp = z.groupby(["sec", "sys"]).z  # groups: one per second and constellation
    z["n"] = grp.transform("count")  # satellites in each group
    z["clk"] = grp.transform("median")  # the robust clock of each group
    z = z[z.n >= 4].copy()  # a median needs four satellites to be robust to one bad one
    z["res"] = z.z - z.clk  # residual about the constellation's own clock
    return z[["sec", "sv", "res", "el"]].reset_index(drop=True)  # the table


def rinex_epoch_times(path: Path) -> np.ndarray:  # function: GPS seconds of every epoch line of a RINEX file, in file order
    lines, k0, _ = rinex_lines(path)  # the file's lines and the first data line
    out = []  # epoch times
    for l in lines[k0:]:  # for loop over the data lines
        if l[:1] == ">":  # if: an epoch line
            out.append((datetime(int(l[2:6]), int(l[7:9]), int(l[10:12]), int(l[13:15]), int(l[16:18]), tzinfo=timezone.utc) - M.GPS_EPOCH).total_seconds() + float(l[18:29]))  # GPS seconds
    return np.array(out)  # the array


def ubx_2612_obs(gt: pd.DataFrame) -> pd.DataFrame:  # function: u-blox 01 in 2.6.12 over the whole capture, the three RINEX pieces with every epoch's time tag put back on true time by the host-time index
    from helpers import host_clock, host_times  # the project's laptop-clock readers
    S03 = load_module("receiver_status_for_gain_evidence", "1_ground_truth/03_receiver_status.py")  # the receiver-status script's checked UBX reader
    folder = DAY_DIR / gt.loc["2.6.12", "session_folder"]  # the raw session folder (read only)
    f = next(folder.glob("*_UBX_01.ubx"))  # the u-blox 01 file
    ends = []  # end offsets of the RXM-RAWX epochs that carry measurements (the epochs the RINEX pieces hold)
    for cls, mid, p, end in S03.ubx_messages(f, S03.Integrity(), {(0x02, 0x15)}):  # RAWX messages, checksum verified
        if len(p) >= 16 and p[11] > 0:  # numMeas at offset 11: an epoch with measurements
            ends.append(end)  # kept
    ends = np.array(ends, np.int64)  # as an array
    host = np.array([h.timestamp() for h in host_times(folder / "host_timing" / (f.name + ".hosttime.csv"), ends, host_clock(folder))])  # host time of each epoch, unix seconds
    pcs = pd.read_csv(PROC / "preprocessing" / "rinex_ubx01_2612_pieces.csv")  # the pieces' byte ranges (the conversion script)
    out = []  # corrected observation tables
    for r in pcs.itertuples():  # for loop over the three pieces
        m = (ends > r.byte_start) & (ends <= r.byte_end + 1)  # the epochs whose RAWX message lies in this piece
        ob = read_full(PROJECT / "Data" / "rinex" / gt.loc["2.6.12", "session_folder"] / r.rinex_file, [(0.0, 4.0e9)], "UBX_01")  # the piece's observations, every epoch
        LINES.clear()  # frees the decompressed file
        ep = rinex_epoch_times(PROJECT / "Data" / "rinex" / gt.loc["2.6.12", "session_folder"] / r.rinex_file)  # every epoch of the piece, in file order, whether or not it has a C1C observation
        if len(ep) != int(m.sum()):  # if: the piece and the binary disagree
            raise SystemExit(f"2.6.12 u-blox piece {r.piece}: {len(ep)} RINEX epochs against {int(m.sum())} RAWX epochs")  # stop: the mapping would be wrong
        rx_unix = ep - M.LEAP_SECONDS + M.GPS_UNIX  # time tag as unix seconds
        off = np.round(rx_unix - host[m])  # the whole-second offset between the time tag and the laptop clock (latency is under 0.2 s, so rounding is safe)
        frac = rx_unix - host[m] - off  # what is left after removing whole seconds: the laptop's reading latency, if the shift is a whole number of seconds
        put(f"u-blox 01 2.6.12 piece {r.piece}", "epochs; distinct whole-second offsets (s) with counts; max |remainder| (s)", f"{len(ep)}; " + ", ".join(f"{int(k)}:{int(v)}" for k, v in pd.Series(off).value_counts().items()) + f"; {np.abs(frac).max():.3f}")  # the check that the shift is a whole number of seconds
        corr = dict(zip(ep, ep - off))  # epoch -> time tag put back by that many whole seconds
        ob["t_gps"] = ob.t_gps.map(corr)  # corrected GPS time
        ob["sec"] = np.floor(ob.t_gps - M.LEAP_SECONDS + M.GPS_UNIX).astype(np.int64)  # corrected UTC second
        ob["piece"] = r.piece  # which piece
        ob["offset_s"] = ob.sec * 0.0 + float(np.median(off))  # the median offset applied in this piece (for the log; the offset varies within piece 1)
        out.append(ob)  # kept
    ob = pd.concat(out, ignore_index=True).sort_values(["sec", "piece"]).drop_duplicates(["sec", "sv"], keep="first")  # one observation per satellite and second, in time order
    a, b = capture_spans(gt, "2.6.12", "UBX_01")[0]  # the capture span
    return ob[(ob.sec >= a) & (ob.sec < b)].reset_index(drop=True)  # inside the capture


def residual_series(orb, kl: tuple, gt: pd.DataFrame, test: str, rx: str, site: np.ndarray) -> tuple:  # function: per-satellite residuals about a per-constellation median clock over a session's capture, the raw pseudoranges and the clock-in table
    if (test, rx) == ("2.6.12", "UBX_01"):  # if: the u-blox whose time tag was shifted
        obs = ubx_2612_obs(gt)  # evaluated on host-corrected time, over the whole capture
    else:  # else: every other session and receiver
        obs = read_full(M.obs_path(gt.loc[test, "session_folder"], test, rx), capture_spans(gt, test, rx), rx)  # observations in the capture span
        LINES.clear()  # frees the decompressed file (memory)
    zr = raw_z(orb, kl, obs, site)  # clock still in
    return resid_mask(zr, MASK_DEG), obs[["sec", "t_gps", "sv", "pr"]], zr, obs[["sec", "sv", "cn0"]]  # residuals at 10 degrees, the pseudoranges, the clock-in table, the C/N0 of every observation


def anomaly_runs(df: pd.DataFrame, thr: float, persist: int) -> pd.DataFrame:  # function: per-satellite runs of at least `persist` consecutive seconds with |residual| above the threshold
    out = []  # (satellite, first second, last second)
    d = df[df.res.abs() > thr].sort_values(["sv", "sec"])  # only the exceedances
    for sv, g in d.groupby("sv"):  # for loop over satellites
        sec = g.sec.to_numpy()  # their seconds
        rid = np.cumsum(np.r_[True, np.diff(sec) != 1])  # run number: a new run wherever a second is missing
        for r in np.unique(rid):  # for loop over runs
            m = sec[rid == r]  # the seconds of this run
            if len(m) >= persist:  # if: long enough
                out.append((sv, float(m[0]), float(m[-1])))  # kept
    return pd.DataFrame(out, columns=["sv", "start", "end"]).sort_values("start").reset_index(drop=True)  # time order


def zero_alarm_threshold(df: pd.DataFrame, persist: int) -> float:  # function: the smallest threshold at which no satellite has a run of `persist` consecutive seconds above it (diagnostic: how large the clean tail really is)
    best = 0.0  # the largest, over satellites and windows, of the smallest |residual| in the window
    for sv, g in df.groupby("sv"):  # for loop over satellites
        g = g.sort_values("sec")  # time order
        a = g.res.abs().to_numpy()  # |residual| values
        sec = g.sec.to_numpy()  # their seconds
        rid = np.cumsum(np.r_[True, np.diff(sec) != 1])  # consecutive-second runs
        for r in np.unique(rid):  # for loop over runs
            x = a[rid == r]  # the run's values
            if len(x) >= persist:  # if: long enough to hold a window
                best = max(best, float(pd.Series(x).rolling(persist).min().max()))  # the highest 'window minimum' is the level this run holds for persist seconds
    return best  # metres


def episodes(runs: pd.DataFrame) -> int:  # function: number of alarm episodes: runs of different satellites that touch or overlap count once
    n, cur_end = 0, -1e18  # count and the end of the current episode
    for a, b in zip(runs.start, runs.end):  # for loop over runs in time order
        if a > cur_end + 1:  # if: a new episode begins
            n += 1  # counted
        cur_end = max(cur_end, b)  # extend the current episode
    return n  # the count


def rate_bound(n: int, hours: float) -> tuple:  # function: false alarms per hour and the 95 % upper bound (rule of three at zero; Poisson tail otherwise)
    if hours <= 0:  # if: no clean time
        return np.nan, np.nan  # nothing to say
    from scipy.stats import chi2  # the chi-square quantile gives the exact Poisson upper limit
    ub = chi2.ppf(0.95, 2 * (n + 1)) / 2.0 / hours  # exact one-sided 95 % upper limit on the rate
    return n / hours, float(ub)  # estimate and bound


def gain_series(st: pd.DataFrame, test: str, rx: str) -> pd.Series:  # function: one receiver's L1 gain on a one-second grid (NaN where there is no reading)
    r = st[(st.test_code == test) & (st.receiver == RX_STATUS[rx])]  # this session and receiver
    r = r[r.band == ("L1" if rx == "UBX_01" else "GPSL1/E1")]  # the L1 gain (u-blox AGC count; Septentrio main-antenna GPS L1/E1 gain in dB)
    val = r.agc_cnt if rx == "UBX_01" else r.agc_gain_db  # the gain column of that receiver
    t = np.floor(pd.to_datetime(r.timestamp_utc, utc=True).astype("int64") / 1e9).astype(np.int64)  # whole unix seconds
    g = pd.Series(val.to_numpy(float), index=t.to_numpy()).groupby(level=0).last()  # the last reading of each second
    return g.reindex(range(int(g.index.min()), int(g.index.max()) + 1))  # a full one-second grid


def gain_drop(g: pd.Series) -> pd.Series:  # function: gain deviation: the median of the gain over the 90 s that ended 30 s ago, minus the gain now (positive when the gain falls)
    ref = g.shift(GAIN_REF_LAG).rolling(GAIN_REF_LEN, min_periods=30).median()  # trailing reference, no information from the future
    return ref - g  # a fall in gain is a positive deviation


def first_hold_time(t: np.ndarray, flag: np.ndarray, hold: float) -> float | None:  # function: start of the first run of flagged samples lasting at least `hold` seconds (a gap over 5 s ends a run)
    if len(t) == 0:  # if: no samples
        return None  # no event
    brk = np.r_[True, (flag[1:] != flag[:-1]) | (np.diff(t) > 5)]  # a new run where the flag flips or the data have a gap
    rid = np.cumsum(brk)  # run number
    for r in np.unique(rid[flag]):  # for loop over flagged runs in time order
        m = t[(rid == r)]  # the run's times
        if m[-1] - m[0] >= hold:  # if: lasts long enough
            return float(m[0])  # its start
    return None  # never


def own_positions(gt: pd.DataFrame, test: str, rx: str) -> pd.DataFrame:  # function: the receiver's own reported position (X5 and mosaic-H PVTGeodetic, u-blox NAV-PVT), first fix of each second, with its time
    import struct  # standard-library binary decoding
    from helpers import gps_to_utc, host_clock, host_times  # time conversions of the project
    S03 = load_module("receiver_status_for_gain_evidence", "1_ground_truth/03_receiver_status.py")  # the receiver-status script's checked readers
    folder = DAY_DIR / gt.loc[test, "session_folder"]  # the raw session folder (read only)
    integ = S03.Integrity()  # the readers' integrity container
    rows = []  # (unix seconds, lat, lon, h)
    if rx.startswith("SBF"):  # if: a Septentrio file
        f = next(folder.glob(f"*_{rx}.sbf"))  # the file
        for blk, rev, tow, wnc, p, end in S03.sbf_blocks(f, integ, {4007}):  # CRC-verified PVTGeodetic blocks
            t = gps_to_utc(tow, wnc)  # receiver time (true time on the Septentrio) as UTC
            if t is None or len(p) < 26:  # if: no valid time or a short payload
                continue  # skip
            mode, err = p[0] & 0x0F, p[1]  # solution type and error code
            lat, lon, h = struct.unpack_from("<ddd", p, 2)  # radians, radians, metres
            if mode in (0, 3) or err != 0 or lat < -1e10 or lon < -1e10 or h < -1e10:  # if: no computed solution
                continue  # skip
            rows.append((t.timestamp(), np.degrees(lat), np.degrees(lon), h))  # kept
    else:  # else: the u-blox file, timed by the laptop clock because its own time tag is attacked in 2.6.12
        f = next(folder.glob(f"*_{rx}.ubx"))  # the file
        to_utc = host_clock(folder)  # the laptop clock converter of this session
        ends, vals = [], []  # message end offsets and decoded values
        for cls, mid, p, end in S03.ubx_messages(f, integ, {(0x01, 0x07)}):  # NAV-PVT messages
            if len(p) < 92:  # if: not the full payload
                continue  # skip
            fix, flags = p[20], p[21]  # fix type and flags
            if not (flags & 0x01 and fix in (2, 3, 4)):  # if: not a valid 2D/3D fix
                continue  # skip
            lon, lat, hgt = struct.unpack_from("<iii", p, 24)  # 1e-7 degrees, 1e-7 degrees, millimetres
            ends.append(end)  # where the message ends in the file
            vals.append((lat * 1e-7, lon * 1e-7, hgt / 1000.0))  # decoded position
        host = host_times(folder / "host_timing" / (f.name + ".hosttime.csv"), ends, to_utc)  # host time of each message
        rows = [(h_.timestamp(), v[0], v[1], v[2]) for h_, v in zip(host, vals)]  # unix seconds and position
    d = pd.DataFrame(rows, columns=["t", "lat", "lon", "h"]).sort_values("t")  # the table
    d["sec"] = np.floor(d.t).astype(np.int64)  # whole second
    return d.drop_duplicates("sec", keep="first").reset_index(drop=True)  # first fix of each second


def horiz_from_site(lat: np.ndarray, lon: np.ndarray, h: np.ndarray, site: np.ndarray) -> np.ndarray:  # function: horizontal distance of geodetic positions from the site, metres
    return np.array([enu_dist(M.llh_to_ecef(a, b, c), site)[0] for a, b, c in zip(lat, lon, h)])  # one distance per fix


DAY_DIR = PROJECT / "Data" / "Day1_14092026"  # the raw recordings (read only), for the receivers' own position messages


def solver_positions(orb, kl: tuple, pr: pd.DataFrame, site: np.ndarray, reject: bool) -> pd.DataFrame:  # function: our own per-epoch position from all GPS and Galileo satellites, with or without the 30 m rejection, and its horizontal offset from the site
    M.REJECT_M = 30.0 if reject else 1.0e9  # the position script's rejection limit: normal, or switched off
    ep, _ = M.solve_session(orb, pr, site, kl)  # the position script solver
    M.REJECT_M = 30.0  # restored
    if len(ep) == 0:  # if: nothing solved
        return pd.DataFrame(columns=["sec", "dh"])  # empty
    ep["dh"] = [enu_dist(np.array([a, b, c]), site)[0] for a, b, c in zip(ep.x, ep.y, ep.z)]  # horizontal offsets
    return ep[["sec", "dh"]]  # the table


def status_flags(st: pd.DataFrame, test: str, rx: str) -> pd.DataFrame:  # function: seconds at which the receiver's own interference or authentication flag is set (events only, never inputs)
    r = st[(st.test_code == test) & (st.receiver == RX_STATUS[rx])]  # this session and receiver
    if rx == "UBX_01":  # if: the u-blox
        r = r[r.jamming_state.notna()]  # MON-RF rows
        on = r.jamming_state >= 2  # jamming state warning (2) or critical (3)
        kind = "u-blox jamming state >= 2"  # label
    else:  # else: a Septentrio receiver
        r = r[r.band.isin(FLAG_BANDS[rx])]  # the main-antenna L1-type bands
        on = (r.rf_flag_notauthentic == 1) | (r.rf_flag_nma_fail == 1)  # not-authentic or navigation-message-authentication-failed flag
        kind = "RFStatus not-authentic / NMA fail (L1, GLO L1, B1 main)"  # label
    t = np.floor(pd.to_datetime(r.timestamp_utc, utc=True).astype("int64") / 1e9).astype(np.int64)  # whole unix seconds
    return pd.DataFrame({"sec": t.to_numpy(), "on": on.to_numpy(), "kind": kind}).sort_values("sec")  # the table


def lever_arm_test(add) -> None:  # function: the X5 minus mosaic-H horizontal difference at Stave, pre-test against post-test on common epochs
    ep = pd.read_csv(PROC / "preprocessing" / "reference_position_epochs.csv")  # the position script 1 Hz solutions
    ep = ep[ep.test_code == "2.6.12"]  # Stave
    sched = T("15:50:00")  # the scheduled start
    a = ep[ep.receiver == "SBF_02"].set_index("sec")  # X5 solutions
    b = ep[ep.receiver == "SBF_03"].set_index("sec")  # mosaic-H solutions
    com = a.index.intersection(b.index)  # common seconds
    site = site_xyz("Stave")  # the Stave reference position
    lat, lon, _ = M.ecef_to_llh(*site)  # its geodetic position
    A = M.enu_axes(lat, lon)  # its east, north, up axes
    d = (a.loc[com, ["x", "y", "z"]].to_numpy() - b.loc[com, ["x", "y", "z"]].to_numpy()) @ A.T  # X5 minus mosaic-H in east, north, up per common second
    d = pd.DataFrame(d, index=com, columns=["e", "n", "u"])  # as a table
    res = {}  # period -> (n, east, north, length, azimuth)
    for name, m in (("pre-test", d.index < sched), ("post-test", d.index >= sched)):  # for loop over the two periods
        x = d[m]  # the period's differences
        if len(x) == 0:  # if: none
            continue  # skip
        e, n = float(x.e.median()), float(x.n.median())  # median east and north components
        res[name] = (len(x), e, n, float(np.hypot(e, n)), float(np.degrees(np.arctan2(e, n)) % 360), float(x.u.median()))  # count, components, length, azimuth, up
        add("Stave offset X5 - mosaic-H", f"{name}: common epochs, east, north (m), horizontal length (m), azimuth (deg), up (m)", ", ".join(f"{v:.3f}" if i else str(int(v)) for i, v in enumerate(res[name])))  # reported
    if len(res) == 2:  # if: both periods exist
        dl = abs(res["pre-test"][3] - res["post-test"][3])  # length change
        da = abs((res["pre-test"][4] - res["post-test"][4] + 180) % 360 - 180)  # azimuth change, wrapped to 0-180
        add("Stave offset X5 - mosaic-H", "difference pre-test vs post-test: length (m), azimuth (deg)", f"{dl:.3f}, {da:.1f}")  # the comparison
        add("Stave offset X5 - mosaic-H", "agree within 0.2 m and 20 deg", bool(dl <= 0.2 and da <= 20))  # the criterion fixed beforehand


def main_part2() -> dict:  # function: part 2, the early-warning timeline
    ROWS.clear()  # the summary of part 2 starts empty
    CHECKS.clear()  # and its checks
    gt = pd.read_csv(PROC / "ground_truth" / "session_ground_truth.csv").set_index("test_code")  # session windows and folders
    st = pd.read_csv(PROC / "ground_truth" / "receiver_status_timeseries.csv", low_memory=False)  # the status table: gains and flags (events and comparison only)
    ta3 = pd.read_csv(PROC / "ground_truth" / "ta3_schedule_confirmation.csv")  # the receiver-status script's gain onsets
    sp = pd.read_csv(PROC / "ground_truth" / "spoof_2612_timeline.csv")  # the receiver-status script's traced 2.6.12 instants
    orb = M.Orbits(S04, dtm.date(2026, 9, 14))  # satellite orbits and clocks from the IGS file only
    kl = M.igs_klobuchar(S04.IGS_BRDC)  # the GPS Klobuchar parameters
    # ---- residuals for every session and receiver ----
    RES, PRS, ZRAW, CN0 = {}, {}, {}, {}  # (test, rx) -> residual table; pseudoranges; clock-in table; C/N0
    for test in SESSIONS:  # for loop over the sessions
        site = site_xyz(M.SITE_OF[test])  # the session's site reference
        for rx in RXS:  # for loop over the receivers
            RES[(test, rx)], PRS[(test, rx)], ZRAW[(test, rx)], CN0[(test, rx)] = residual_series(orb, kl, gt, test, rx, site)  # residuals, pseudoranges, clock-in table, C/N0
            r = RES[(test, rx)]  # the table
            print(f"  residuals {test:7s} {rx}: {r.sec.nunique():5d} seconds, {len(r):7d} satellite-seconds, median |res| {r.res.abs().median():.2f} m")  # progress line
            put(f"residual scale {test} {M.RECEIVERS[rx]}", "median |residual| (m), satellite-seconds, seconds", f"{r.res.abs().median():.2f}, {len(r)}, {r.sec.nunique()}")  # kept for the report
    # ---- thresholds, leave-one-session-out ----
    thr, nthr = {}, {}  # (test, rx) -> threshold and sample size
    for rx in RXS:  # for loop over the receivers
        for test in SESSIONS:  # for loop over the held-out session
            vals = []  # |residuals| from the other sessions' clean intervals
            for other in SESSIONS:  # for loop over the other sessions
                if other == test:  # if: the held-out session
                    continue  # leave it out
                r = RES[(other, rx)]  # that session's residuals
                vals.append(r.res.abs().to_numpy()[in_spans(r.sec.to_numpy(float), clean_spans(other, rx))])  # those inside its clean intervals
            v = np.concatenate(vals)  # pooled values
            thr[(test, rx)], nthr[(test, rx)] = float(np.percentile(v, PCT)), len(v)  # 99.9th percentile and the sample size
    for rx in RXS:  # for loop over the receivers
        put(f"threshold {M.RECEIVERS[rx]}", "99.9th percentile of |residual| on the other sessions' clean intervals (m), by held-out session; sample size", "; ".join(f"{t}: {thr[(t, rx)]:.2f} (n={nthr[(t, rx)]})" for t in SESSIONS))  # reported
    # ---- false alarms on each session's own clean intervals, for each persistence ----
    fa_rows = []  # rows of the false-alarm table
    for rx in RXS:  # for loop over the receivers
        for P in PERSIST:  # for loop over the persistences
            tot_n, tot_s = 0, 0  # episodes and clean seconds with data, summed over sessions
            for test in SESSIONS:  # for loop over the sessions
                cs = clean_spans(test, rx)  # its clean intervals
                if not cs:  # if: none for this receiver
                    continue  # skip
                r = RES[(test, rx)]  # residuals
                r = r[in_spans(r.sec.to_numpy(float), cs)]  # clean ones only
                ru = anomaly_runs(r, thr[(test, rx)], P)  # runs above the leave-one-out threshold
                n_ep, secs = episodes(ru), int(r.sec.nunique())  # episodes and seconds with data
                rate, ub = rate_bound(n_ep, secs / 3600.0)  # per hour and its 95 % bound
                tot_n += n_ep  # accumulate
                tot_s += secs  # accumulate
                fa_rows.append({"detector": "residual", "receiver": rx, "persistence_s": P, "test_code": test, "threshold": thr[(test, rx)], "clean_seconds_with_data": secs, "episodes": n_ep, "per_hour": rate, "upper95_per_hour": ub, "zero_alarm_threshold": zero_alarm_threshold(r, P)})  # one row, with the threshold that would have given none
            rate, ub = rate_bound(tot_n, tot_s / 3600.0)  # the receiver's total
            fa_rows.append({"detector": "residual", "receiver": rx, "persistence_s": P, "test_code": "all", "threshold": np.nan, "clean_seconds_with_data": tot_s, "episodes": tot_n, "per_hour": rate, "upper95_per_hour": ub})  # total row
            put(f"false alarms {M.RECEIVERS[rx]}", f"residual detector, persistence {P} s: episodes over all clean intervals / clean hours / per hour (95 % bound)", f"{tot_n} / {tot_s / 3600.0:.3f} / {rate:.2f} ({ub:.1f})")  # reported
    # ---- choose the persistence per receiver: the smallest with no false alarm over all clean intervals, else 10 s ----
    chosen = {}  # receiver -> persistence
    fa = pd.DataFrame(fa_rows)  # the table so far
    for rx in RXS:  # for loop over the receivers
        tot = fa[(fa.receiver == rx) & (fa.test_code == "all")].set_index("persistence_s").episodes  # total episodes per persistence
        ok_p = [P for P in PERSIST if tot[P] == 0]  # persistences with none
        chosen[rx] = ok_p[0] if ok_p else PERSIST[-1]  # the smallest, else the largest tried
        put(f"persistence {M.RECEIVERS[rx]}", "chosen (smallest with zero false-alarm episodes on clean data; the largest tried if none)", f"{chosen[rx]} s" + ("" if ok_p else " (false alarms remain)"))  # reported
    # ---- gain series and their deviations (the fixed-fall gain detectors are evaluated in part 2b) ----
    G, D = {}, {}  # (test, rx) -> gain series, deviation series
    for test in SESSIONS:  # for loop over the sessions
        for rx in RXS:  # for loop over the receivers
            G[(test, rx)] = gain_series(st, test, rx)  # the gain on a one-second grid
            D[(test, rx)] = gain_drop(G[(test, rx)])  # its deviation: the reference minus the gain now
    fdf = pd.DataFrame(fa_rows)  # the false-alarm table
    for r in fdf[(fdf.detector == "residual") & (fdf.test_code != "all") & (fdf.persistence_s == PERSIST[-1])].itertuples():  # for loop over session x receiver at the longest persistence
        put(f"zero-alarm level {M.RECEIVERS[r.receiver]}", f"{r.test_code}: threshold (m) that would give no {PERSIST[-1]} s episode on its clean intervals, against the leave-one-out threshold used (m)", f"{r.zero_alarm_threshold:.2f} vs {r.threshold:.2f}")  # how far the held-out clean tail lies beyond the threshold
    fdf.round(4).to_csv(PROC / "ground_truth" / "early_warning_false_alarms.csv", index=False)  # written (the per-satellite residual detector of part 2; gain detectors are in part 2b)
    # ---- events per session and receiver ----
    tl = []  # timeline rows
    for test in SESSIONS:  # for loop over the sessions
        g = gt.loc[test]  # the session's windows
        s0, s1 = T(g.scheduled_start_utc[11:]), T(g.scheduled_end_utc[11:])  # scheduled window
        c0, c1 = T(g.capture_start_utc[11:]), T(g.capture_end_utc[11:])  # capture
        in_prog = bool(s0 <= c0 < s1)  # the capture begins inside the scheduled window
        site = site_xyz(M.SITE_OF[test])  # the session's site reference
        for rx in RXS:  # for loop over the receivers
            P = chosen[rx]  # persistence of this receiver
            # dict literal over two lines: the fixed identity, window and setting columns of this row of the timeline
            row = {"test_code": test, "receiver": rx, "capture_start": ts(c0), "capture_end": ts(c1), "scheduled_start": ts(s0), "scheduled_end": ts(s1),
                   "attack_in_progress_at_capture_start": in_prog, "persistence_s": P, "residual_threshold_m": thr[(test, rx)]}  # fixed columns
            ru = anomaly_runs(RES[(test, rx)], thr[(test, rx)], P)  # all runs in the capture
            row["first_anomaly"] = ts(ru.start.min()) if len(ru) else ""  # first anomaly in the capture
            row["first_anomaly_sv"] = ru.sv.iloc[ru.start.argmin()] if len(ru) else ""  # its satellite
            known = known_invalid(ru, test, rx)  # the diagnosed invalid measurements (u-blox 01 G30 in 2.6.2, Septentrio G08 in 2.6.10), one shared rule
            ru = ru.assign(known=known)  # marked
            late = ru[(ru.start >= s0) & ~ru.known]  # runs from the scheduled start on, without the diagnosed invalid measurements
            row["first_anomaly_from_sched_start"] = ts(late.start.min()) if len(late) else ""  # first anomaly at or after the scheduled start
            row["first_anomaly_from_sched_start_sv"] = late.sv.iloc[late.start.argmin()] if len(late) else ""  # its satellite
            row["anomaly_episodes_in_capture"] = episodes(ru)  # how many episodes in all
            row["observation_seconds"] = int(RES[(test, rx)].sec.nunique())  # seconds with residuals
            # gain events
            a = ta3[(ta3.test_code == test) & (ta3.receiver == RX_STATUS[rx])]  # the receiver-status script's row
            row["gain_onset_section3"] = a.onset_utc.iloc[0] if len(a) and isinstance(a.onset_utc.iloc[0], str) else ""  # its 5 dB onset (X5, H) or 400-count onset (u-blox)
            # position events
            own = own_positions(gt, test, rx)  # the receiver's own reported positions
            own = own[(own.sec >= capture_spans(gt, test, rx)[0][0]) & (own.sec < capture_spans(gt, test, rx)[0][1])]  # inside the usable capture span
            dh = horiz_from_site(own.lat.to_numpy(), own.lon.to_numpy(), own.h.to_numpy(), site) if len(own) else np.array([])  # their distances from the site
            row["own_position_fixes"] = len(own)  # how many fixes
            row["own_position_first_over_50m"] = ts(first_hold_time(own.t.to_numpy(), dh > POS_M, HOLD_POS_S)) if len(own) else ""  # first held beyond 50 m
            row["own_position_max_m"] = float(dh.max()) if len(dh) else np.nan  # the largest distance
            for lab, rej in (("solver_norej", False), ("solver_rej", True)):  # for loop over the two solver settings
                sp_ = solver_positions(orb, kl, PRS[(test, rx)], site, rej)  # our own positions
                row[f"{lab}_first_over_50m"] = ts(first_hold_time(sp_.sec.to_numpy(float), (sp_.dh > POS_M).to_numpy(), HOLD_SOL_S)) if len(sp_) else ""  # first held beyond 50 m
            # receiver flag
            fl = status_flags(st, test, rx)  # the flag series
            fon = fl[fl.on]  # seconds with the flag set
            row["flag_kind"] = fl.kind.iloc[0] if len(fl) else ""  # which flag
            row["first_flag"] = ts(float(fon.sec.min())) if len(fon) else ""  # first set
            row["flag_seconds_set"] = int(len(fon))  # how many seconds in all
            row["flag_set_before_sched_start"] = bool(len(fon) and fon.sec.min() < s0)  # set before the test began
            cs = clean_spans(test, rx)  # clean intervals
            row["flag_fraction_in_clean"] = float(fl[in_spans(fl.sec.to_numpy(float), cs)].on.mean()) if cs and in_spans(fl.sec.to_numpy(float), cs).any() else np.nan  # how often it is set on clean data
            # 2.6.12 traced instants (the receiver-status script)
            if test == "2.6.12":  # if: the time-spoofing session
                e = sp[sp.receiver == RX_STATUS[rx]].set_index("event").host_time_utc  # this receiver's traced events
                for ev, col in (("loss of tracking", "t2612_loss_of_tracking"), ("reacquisition", "t2612_reacquisition"), ("measurement-engine deception", "t2612_measurement_engine_deception"), ("navigation time deception", "t2612_navigation_time_deception"), ("navigation fix lost", "t2612_navigation_fix_lost")):  # for loop over the events
                    row[col] = e.get(ev, "") if isinstance(e.get(ev, ""), str) else ""  # the host time or empty
            tl.append(row)  # one row per session and receiver
            print(f"  timeline {test:7s} {rx}: anomaly {row['first_anomaly'] or '-':>8s} ({row['first_anomaly_sv']}), gain S3 {row['gain_onset_section3'] or '-':>12s}, own position {row['own_position_first_over_50m'] or '-':>8s}, flag {row['first_flag'] or '-':>8s}")  # progress line
    tdf = pd.DataFrame(tl)  # the timeline table
    # ---- lead times, seconds; positive = the observation anomaly came first ----
    def to_s(txt: str) -> float:  # nested function: "HH:MM:SS(.fff)" -> unix seconds, NaN if empty
        return T(txt) if isinstance(txt, str) and txt else np.nan  # parsed or NaN
    rec = {(r.test_code, r.receiver): to_s(r.recovery_utc) for r in ta3.assign(receiver=ta3.receiver.map({v: k for k, v in RX_STATUS.items()})).itertuples() if isinstance(r.recovery_utc, str)}  # the receiver-status script's gain recovery per session and receiver
    tdf["gain_recovery_section3"] = [ts(rec.get((t, r), np.nan)) for t, r in zip(tdf.test_code, tdf.receiver)]  # shown beside the events
    ok = []  # whether a lead is defined for each row
    note = []  # why not, or a caution
    for r in tdf.itertuples():  # for loop over the rows
        a = to_s(r.first_anomaly_from_sched_start)  # the anomaly
        rc = rec.get((r.test_code, r.receiver), np.nan)  # the gain recovery
        if r.attack_in_progress_at_capture_start:  # if: the attack was already on at capture start
            ok.append(False)  # a lead cannot be timed
            note.append("attack in progress at capture start: the first anomaly is the state at the start, no lead")  # reason
        elif not np.isfinite(a):  # if: no anomaly
            ok.append(False)  # nothing to time
            note.append("no observation anomaly from the scheduled start" + ("; no observations from 15:50:10 to 16:07:04 (X5 tracked nothing)" if (r.test_code, r.receiver) == ("2.6.12", "SBF_02") else "") + ("; observations usable only to 15:51:16 (time tag shifted from 15:51:17)" if (r.test_code, r.receiver) == ("2.6.12", "UBX_01") else ""))  # reason
        elif np.isfinite(rc) and a > rc:  # if: the anomaly appears after the interference has ended
            ok.append(False)  # an after-effect, not a warning
            note.append("first anomaly after the receiver-status script's gain recovery: an after-effect of the release, not a warning")  # reason
        else:  # else: usable
            ok.append(True)  # defined
            note.append("")  # no caution
    tdf["lead_defined"] = ok  # flag
    tdf["lead_note"] = note  # text
    for col, ev in (("lead_over_gain_section3_s", "gain_onset_section3"), ("lead_over_own_position_s", "own_position_first_over_50m"), ("lead_over_solver_norej_s", "solver_norej_first_over_50m"), ("lead_over_flag_s", "first_flag")):  # for loop over the comparison events
        tdf[col] = [(to_s(e) - to_s(a)) if k else np.nan for a, e, k in zip(tdf.first_anomaly_from_sched_start, tdf[ev], tdf.lead_defined)]  # event minus anomaly: positive when the anomaly was earlier
    tdf.round(3).to_csv(PROC / "ground_truth" / "early_warning_timeline.csv", index=False)  # the timeline
    u = tdf[(tdf.test_code == "2.6.12") & (tdf.receiver == "UBX_01")].iloc[0]  # u-blox 01 in 2.6.12
    s3 = sp[(sp.receiver == "u-blox_01") & (sp.event == "navigation position departure (> 50 m)")].host_time_utc.iloc[0]  # the receiver-status script's own-position departure
    check("u-blox 01 2.6.12 own-position departure equals the receiver-status script's (10 s hold)", s3, u.own_position_first_over_50m, abs(T(s3) - T(u.own_position_first_over_50m)) < 0.01)  # same decoder logic, same answer
    c5 = tdf[tdf.test_code == "2.6.5"].set_index("receiver")  # the 2.6.5 rows
    check("2.6.5 first anomalies are G30 on all three receivers and within 15 s of part 1 (u-blox 12:48:46-47, X5 12:49:00, mosaic-H 12:48:57-49:02)", "G30 at 12:48:46, 12:49:00, 12:48:57", ", ".join(f"{c5.loc[r, 'first_anomaly_sv']} {c5.loc[r, 'first_anomaly']}" for r in RXS), all(c5.loc[r, "first_anomaly_sv"] == "G30" for r in RXS) and abs(T(c5.loc["UBX_01", "first_anomaly"]) - T("12:48:47")) < 15)  # agrees with part 1
    lever_arm_test(put)  # Stave offset X5 - mosaic-H
    return {"gt": gt, "st": st, "ta3": ta3, "sp": sp, "orb": orb, "kl": kl, "RES": RES, "ZRAW": ZRAW, "CN0": CN0, "PRS": PRS, "G": G, "D": D, "tdf": tdf}  # handed to part 2b


def plot_early_warning_timeline(tdf: pd.DataFrame, gt: pd.DataFrame) -> None:  # function: the figure, the event timeline per session and receiver
    import matplotlib  # the plotting library
    matplotlib.use("Agg")  # file-only backend
    import matplotlib.pyplot as plt  # pyplot
    fig, axes = plt.subplots(3, 2, figsize=(11, 8.6))  # one panel per session
    names = {"SBF_02": "X5", "SBF_03": "mosaic-H", "UBX_01": "u-blox 01"}  # short receiver names
    ev = [("T_first_alarm", "o", "#D55E00", "first observation alarm (T statistic)"), ("gain_onset_section3", "s", "#E69F00", "gain onset, the receiver-status script"), ("own_position_first_over_50m", "^", "#0072B2", "own position > 50 m"), ("first_flag", "v", "#009E73", "receiver flag")]  # events: column, marker, colour, label
    for ax, test in zip(axes.ravel(), SESSIONS):  # for loop over the panels
        g = gt.loc[test]  # the session's windows
        s0, s1, c0, c1 = (T(g[k][11:]) for k in ("scheduled_start_utc", "scheduled_end_utc", "capture_start_utc", "capture_end_utc"))  # windows in unix seconds
        ax.axvspan(s0, s1, color="#DDDDDD", alpha=0.6, lw=0)  # the scheduled window
        for y, rx in enumerate(RXS):  # for loop over the receivers
            ax.plot([c0, c1 + 1], [y, y], color="#888888", lw=1.2)  # the capture
            r = tdf[(tdf.test_code == test) & (tdf.receiver == rx)].iloc[0]  # this row
            for col, mk, colr, lab in ev:  # for loop over the events
                if isinstance(r[col], str) and r[col]:  # if: the event exists
                    hollow = col == "T_first_alarm" and not bool(r.T_lead_defined)  # an anomaly that is an after-effect or the state of an attack in progress
                    ax.plot(T(r[col]), y, mk, color="white" if hollow else colr, ms=6, mec=colr if hollow else "black", mew=1.2 if hollow else 0.5)  # its marker
        ax.set_xlim(c0 - 30, c1 + 30)  # the capture, not the whole scheduled window
        ax.set_yticks(range(3))  # one row per receiver
        ax.set_yticklabels([names[r] for r in RXS], fontsize=8)  # names
        ax.set_ylim(-0.6, 2.6)  # frame
        ticks = np.linspace(c0, c1, 5)  # five ticks over the capture
        ax.set_xticks(ticks)  # set
        ax.set_xticklabels([ts(t)[:5] for t in ticks], fontsize=7)  # HH:MM
        ax.set_title(f"{test}" + (" (attack in progress at capture start)" if bool(s0 <= c0 < s1) else ""), fontsize=9)  # title
        ax.grid(color="#999999", alpha=0.3, lw=0.5)  # faint grid
    from matplotlib.lines import Line2D  # legend handles drawn by hand
    handles = [Line2D([], [], marker=mk, color=colr, mec="black", mew=0.5, ls="", ms=6, label=lab) for _, mk, colr, lab in ev]  # one per event type
    handles.append(Line2D([], [], marker="o", color="white", mec="#D55E00", mew=1.2, ls="", ms=6, label="alarm outside the spoofing phase, after-effect or attack already on"))  # the hollow variant
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7, frameon=False)  # one legend for all panels
    fig.supxlabel("UTC, 14 September 2026 (grey band: scheduled window; line: capture)", fontsize=8, y=0.07)  # axis label
    fig.tight_layout(rect=(0, 0.1, 1, 1))  # spacing, leaving room for the legend
    save_fig(fig, GRAPH / "early_warning_timeline.png")  # written, byte-stable


# =====================================================================================================================
# Part 2b: a per-epoch detector statistic normalised by an elevation-dependent sigma, fixed-threshold operating points
# for the residual and gain detectors, and false alarms counted on schedule-defined margins
# =====================================================================================================================
MASKS = (10.0, 15.0)  # tuple: the elevation masks tried (degrees)
BIN_DEG = 5.0  # float constant: elevation bin width for the sigma(el) fit
T_FIXED = (4.0, 6.0, 8.0, 12.0)  # tuple: fixed thresholds on the normalised statistic T, in sigma units
GAIN_FIXED = {"SBF_02": (1.0, 2.0, 3.0, 5.0), "SBF_03": (1.0, 2.0, 3.0, 5.0), "UBX_01": (100.0, 200.0, 300.0, 500.0)}  # dict: fixed gain falls per receiver (dB for the Septentrio, AGC counts for the u-blox at about 100 counts per dB, from the 2.6.12 onset: -46 dB against -4563 counts)
JAM_TAIL_S = 120.0  # float constant: seconds after the scheduled end of the initial jamming during which the gain can still show it (the receiver-status script saw lags up to 110 s)
SPOOF_START_265 = "12:48:46"  # string: the start of the spoofing effect in 2.6.5, the first G30 anomaly (part 1)


def schedule_margins(gt: pd.DataFrame, test: str, rx: str) -> list:  # function: all capture time outside the scheduled window and the decided exclusions, as (start, end+1) unix seconds, with the 30 s guard at edges next to an excluded or attack interval; no selection on the gain
    g = gt.loc[test]  # the session's windows
    s0, s1 = T(g.scheduled_start_utc[11:]), T(g.scheduled_end_utc[11:])  # scheduled window
    c0, c1 = T(g.capture_start_utc[11:]), T(g.capture_end_utc[11:]) + 1  # capture, last second included
    gd = M.GUARD_S  # the guard in seconds
    if test in ("2.6.5", "2.6.11"):  # if: 2.6.5 (excluded entirely) or 2.6.11 (no margin)
        return []  # none
    if test == "2.6.10":  # if: spoofed tracking lasted to 14:43:59; the capture starts inside the window
        sp_ = [(max(s1, T("14:44:00")) + gd, c1)]  # the post-test margin after the spoofed tracking; the per-second end is later, 14:44:04 pooled (X5 and mosaic-H 14:44:03, u-blox 01 14:44:04; ground_truth/spoof_end_2610.csv); this minute bound is kept
    elif test == "2.6.12":  # if: Stave
        pre = (T(DT["margin_contamination_end"]) + gd, s0 - gd)  # the possibly contaminated 2.6.12 margin: the margin before 15:45:12 (ground_truth/decision_times.csv) is possibly contaminated
        post = {"SBF_02": (T("16:07:04") + gd, c1), "SBF_03": (s1 + gd, c1)}.get(rx)  # X5 tracks again from 16:07:04; u-blox after its clock release is excluded
        sp_ = [pre] + ([post] if post else [])  # the margins
    elif test == "2.6.2":  # if: the 2.6.2 site
        sp_ = ([] if rx == "UBX_01" else [(c0, s0 - gd)]) + [(s1 + gd, c1)]  # u-blox 01's pre-test margin holds the invalid G30 measurement (until 13:02:41), so it is excluded
    else:  # else: 2.6.4, whose capture ends before the scheduled end
        sp_ = [(c0, s0 - gd)]  # the pre-test margin
    return [(a, b) for a, b in sp_ if b > a]  # drops empty ones


def fit_sigma(res: pd.DataFrame) -> tuple:  # function: sigma(el) = a + b / sin(el) fitted to robust residual scatter in 5 degree elevation bins (non-negative least squares)
    from scipy.optimize import nnls  # non-negative least squares
    el, ab = res.el.to_numpy(), np.abs(res.res.to_numpy())  # elevations and |residuals|
    xs, ys, ws = [], [], []  # bin centre in 1/sin(el), robust sigma, weight
    for e0 in np.arange(np.floor(el.min()), 90.0, BIN_DEG):  # for loop over elevation bins
        m = (el >= e0) & (el < e0 + BIN_DEG)  # the bin
        if m.sum() >= 100:  # if: enough samples
            xs.append(float(np.mean(1.0 / np.sin(np.radians(el[m])))))  # mean 1/sin(el) in the bin
            ys.append(float(1.4826 * np.median(ab[m])))  # robust sigma: 1.4826 x the median absolute residual
            ws.append(float(np.sqrt(m.sum())))  # weight
    A = np.stack([np.ones(len(xs)), np.array(xs)], axis=1) * np.array(ws)[:, None]  # weighted design matrix: constant and 1/sin(el)
    coef, _ = nnls(A, np.array(ys) * np.array(ws))  # non-negative a and b
    return float(max(coef[0], 0.05)), float(coef[1])  # a floor of 5 cm keeps sigma positive


def t_stat(res: pd.DataFrame, ab: tuple) -> pd.DataFrame:  # function: per epoch, T = max over satellites of |r| / sigma(el), and the satellite that holds the maximum
    r = res.reset_index(drop=True).copy()  # fresh index
    r["x"] = r.res.abs() / (ab[0] + ab[1] / np.sin(np.radians(r.el)))  # |r| in sigma units
    idx = r.groupby("sec").x.idxmax()  # the row of the largest in each second
    out = r.loc[idx, ["sec", "x", "sv"]].rename(columns={"x": "T"}).reset_index(drop=True)  # one row per second
    out["sec"] = out.sec.astype(float)  # float seconds
    return out  # the table


def consec_runs(sec: np.ndarray, persist: int) -> list:  # function: index pairs (first, last) of runs of at least `persist` consecutive whole seconds in a sorted array
    if len(sec) == 0:  # if: nothing
        return []  # no runs
    rid = np.cumsum(np.r_[True, np.diff(sec) != 1])  # a new run wherever a second is missing
    out = []  # (first index, last index)
    for r in np.unique(rid):  # for loop over runs
        ix = np.flatnonzero(rid == r)  # the run's positions
        if len(ix) >= persist:  # if: long enough
            out.append((int(ix[0]), int(ix[-1])))  # kept
    return out  # the runs


def t_runs(ts_: pd.DataFrame, thr: float, persist: int) -> pd.DataFrame:  # function: runs of at least `persist` consecutive seconds with T above the threshold
    d = ts_[ts_["T"] > thr].sort_values("sec")  # the exceedances in time order
    sec, sv = d.sec.to_numpy(), d.sv.to_numpy()  # their seconds and satellites
    return pd.DataFrame([(float(sec[a]), float(sec[b]), sv[a]) for a, b in consec_runs(sec, persist)], columns=["start", "end", "sv"])  # start, end, satellite at the start


def gain_alarm_runs(d: pd.Series, thr: float, persist: int, spans: list | None = None) -> pd.DataFrame:  # function: runs of at least `persist` consecutive seconds with the gain fall at or above the threshold (falls only)
    d = d.dropna()  # seconds with a reading
    if spans is not None:  # if: restricted to margins
        d = d[in_spans(d.index.to_numpy(float), spans)]  # keep those seconds
    sec = d.index.to_numpy(float)[d.to_numpy() >= thr - 1e-9]  # seconds with a fall of at least the threshold
    return pd.DataFrame([(float(sec[a]), float(sec[b]), "gain") for a, b in consec_runs(sec, persist)], columns=["start", "end", "sv"])  # runs


def known_invalid(runs: pd.DataFrame, test: str, rx: str) -> pd.Series:  # function: runs that are the diagnosed invalid measurements (u-blox 2.6.2 G30 until 13:02:41; the 2.6.10 Septentrio G08 until 14:44:43), not attack signatures
    # the shared rule replaces this function's own copy, which flagged every G08 run starting at or before 14:44:43;
    # the two agree on this data because neither Septentrio receiver tracks G08 before 14:44:41 (g30)
    return pd.Series(known_invalid_rule(runs.sv, runs.start, test, rx, DT), index=runs.index)  # Series aligned with the runs, as before


def phase_of(t: float, s0: float, je: float | None) -> str:  # function: which phase of the scheduled test an alarm time falls in
    if je is None or t < s0:  # if: no initial jamming, or before the test
        return "no jamming phase" if (je is None and t >= s0) else "before the scheduled start"  # labels
    if t <= je:  # if: inside the scheduled simultaneous jamming and spoofing
        return "jamming phase"  # label
    return "jamming tail" if t <= je + JAM_TAIL_S else "spoofing phase"  # label


def main_part2b(S: dict) -> None:  # function: part 2b
    gt, ta3, sp, tdf, ZRAW, D = S["gt"], S["ta3"], S["sp"], S["tdf"], S["ZRAW"], S["D"]  # what part 2 hands over
    RESM = {(t, rx, m): resid_mask(ZRAW[(t, rx)], m) for t in SESSIONS for rx in RXS for m in MASKS}  # residuals per session, receiver and mask
    # ---- sigma(el) and the percentile threshold of T, leave-one-session-out ----
    AB, TTHR, TSER, srows = {}, {}, {}, []  # fitted (a, b); threshold; T series; rows of the output table
    for rx in RXS:  # for loop over receivers
        for mask in MASKS:  # for loop over masks
            for test in SESSIONS:  # for loop over the held-out session
                clean = []  # clean-interval residuals of the other sessions
                for o in SESSIONS:  # for loop over the others
                    if o == test:  # if: the held-out one
                        continue  # leave out
                    r = RESM[(o, rx, mask)]  # that session's residuals
                    clean.append(r[in_spans(r.sec.to_numpy(float), clean_spans(o, rx))])  # inside its clean intervals
                pool = pd.concat(clean, ignore_index=True)  # pooled clean residuals
                ab = fit_sigma(pool)  # sigma(el) on the others
                tv = np.concatenate([t_stat(c, ab)["T"].to_numpy() for c in clean if len(c)])  # T on the others' clean epochs
                AB[(test, rx, mask)], TTHR[(test, rx, mask)] = ab, float(np.percentile(tv, PCT))  # sigma fit and the 99.9th percentile of T
                TSER[(test, rx, mask)] = t_stat(RESM[(test, rx, mask)], ab)  # T over the held-out session's whole capture
                srows.append({"receiver": rx, "mask_deg": mask, "held_out": test, "sigma_a_m": ab[0], "sigma_b_m": ab[1], "sigma_at_90deg_m": ab[0] + ab[1], "sigma_at_20deg_m": ab[0] + ab[1] / np.sin(np.radians(20)), "T_threshold_p999": TTHR[(test, rx, mask)], "fit_samples": len(pool), "clean_epochs_for_threshold": len(tv)})  # one row
    pd.DataFrame(srows).round(4).to_csv(PROC / "ground_truth" / "early_warning_sigma_thresholds.csv", index=False)  # written
    for rx in RXS:  # for loop over receivers
        for mask in MASKS:  # for loop over masks
            q = [r for r in srows if r["receiver"] == rx and r["mask_deg"] == mask]  # this receiver's rows
            put(f"sigma(el) {M.RECEIVERS[rx]} mask {int(mask)}", "median over held-out sessions: a (m), b (m), sigma at 20 deg (m), T threshold (99.9th percentile)", f"{np.median([r['sigma_a_m'] for r in q]):.2f}, {np.median([r['sigma_b_m'] for r in q]):.2f}, {np.median([r['sigma_at_20deg_m'] for r in q]):.2f}, {np.median([r['T_threshold_p999'] for r in q]):.2f}")  # reported
    # ---- false alarms of the percentile-T detector ----
    far = []  # false-alarm rows
    tot = {}  # (receiver, mask, persistence, margin type) -> [episodes, seconds]
    for rx in RXS:  # for loop over receivers
        for mask in MASKS:  # for loop over masks
            for P in PERSIST:  # for loop over persistences
                for test in SESSIONS:  # for loop over sessions
                    for mt, spans in (("clean_05c", clean_spans(test, rx)), ("schedule", schedule_margins(gt, test, rx))):  # for loop over the two kinds of margin
                        if not spans:  # if: none
                            continue  # skip
                        ts_ = TSER[(test, rx, mask)]  # T of the held-out session
                        x = ts_[in_spans(ts_.sec.to_numpy(float), spans)]  # inside the margin
                        n_ep, secs = len(t_runs(x, TTHR[(test, rx, mask)], P)), len(x)  # episodes and seconds with data
                        rate, ub = rate_bound(n_ep, secs / 3600.0)  # per hour and its bound
                        far.append({"detector": "T percentile", "receiver": rx, "mask_deg": mask, "persistence_s": P, "margin": mt, "test_code": test, "T_threshold": TTHR[(test, rx, mask)], "seconds_with_data": secs, "episodes": n_ep, "per_hour": rate, "upper95_per_hour": ub})  # one row
                        c = tot.setdefault((rx, mask, P, mt), [0, 0])  # running totals
                        c[0] += n_ep  # episodes
                        c[1] += secs  # seconds
                for mt in ("clean_05c", "schedule"):  # for loop over margin types
                    n_ep, secs = tot.get((rx, mask, P, mt), [0, 0])  # totals
                    rate, ub = rate_bound(n_ep, secs / 3600.0)  # per hour and bound
                    far.append({"detector": "T percentile", "receiver": rx, "mask_deg": mask, "persistence_s": P, "margin": mt, "test_code": "all", "T_threshold": np.nan, "seconds_with_data": secs, "episodes": n_ep, "per_hour": rate, "upper95_per_hour": ub})  # total row
                    put(f"false alarms T {M.RECEIVERS[rx]}", f"mask {int(mask)}, persistence {P} s, {mt} margins: episodes / hours / per hour (95 % bound)", f"{n_ep} / {secs / 3600.0:.3f} / {rate:.2f} ({ub:.1f})")  # reported
    pd.DataFrame(far).round(4).to_csv(PROC / "ground_truth" / "early_warning_false_alarms_T.csv", index=False)  # written
    # ---- choose the setting on clean data only: the lowest persistence with zero false alarms over all held-out sessions, 10 degree mask first; else 10 s ----
    chosen = {}  # receiver -> (mask, persistence)
    for rx in RXS:  # for loop over receivers
        pick = None  # nothing yet
        for P in PERSIST:  # for loop over persistences, lowest first
            for mask in MASKS:  # for loop over masks, 10 degrees first
                if tot[(rx, mask, P, "clean_05c")][0] == 0:  # if: no false alarm on any held-out clean interval
                    pick = (mask, P)  # chosen
                    break  # stop
            if pick:  # if: found
                break  # stop
        if pick is None:  # if: none reaches zero
            mask = min(MASKS, key=lambda m: (tot[(rx, m, PERSIST[-1], "clean_05c")][0], m))  # the mask with fewer false alarms at 10 s
            pick = (mask, PERSIST[-1])  # keep 10 s
            n_ep, secs = tot[(rx, pick[0], pick[1], "clean_05c")]  # its residual rate
            put(f"setting {M.RECEIVERS[rx]}", "no setting reaches zero; kept 10 s, mask, residual false alarms per hour on clean data (95 % bound)", f"{int(pick[0])}, {n_ep / (secs / 3600.0):.2f} ({rate_bound(n_ep, secs / 3600.0)[1]:.1f})")  # stated
        chosen[rx] = pick  # kept
        put(f"setting {M.RECEIVERS[rx]}", "chosen on clean data: mask (deg), persistence (s)", f"{int(pick[0])}, {pick[1]}")  # reported
    # ---- the chosen setting per session: alarms, phases, leads ----
    ta3r = ta3.assign(rx=ta3.receiver.map({v: k for k, v in RX_STATUS.items()}))  # the receiver-status script's rows keyed by stream
    def jam_end(test: str) -> float | None:  # nested function: the scheduled end of the initial jamming, or None for 'exempt' sessions
        v = ta3r[ta3r.test_code == test].initial_jamming_end_utc.iloc[0]  # the text
        return T(v) if isinstance(v, str) and v[:1].isdigit() else None  # unix seconds or None
    def s3(test: str, rx: str, col: str) -> float:  # nested function: the receiver-status script's gain onset or recovery in unix seconds, NaN if absent
        r = ta3r[(ta3r.test_code == test) & (ta3r.rx == rx)]  # the row
        return T(r[col].iloc[0]) if len(r) and isinstance(r[col].iloc[0], str) else np.nan  # parsed or NaN
    def alarms_info(runs: pd.DataFrame, test: str, rx: str) -> dict:  # nested function: first alarm from the scheduled start, its phase, and the first alarm in the spoofing phase
        g = gt.loc[test]  # windows
        s0 = T(g.scheduled_start_utc[11:])  # scheduled start
        je = jam_end(test)  # initial jamming end
        runs = runs[(runs.start >= s0) & ~known_invalid(runs, test, rx)] if len(runs) else runs  # from the scheduled start, without the diagnosed invalid measurements
        info = {"first_alarm": np.nan, "first_alarm_sv": "", "first_alarm_phase": "", "first_alarm_spoofing_phase": np.nan, "first_alarm_spoofing_phase_sv": ""}  # defaults
        if len(runs):  # if: any alarm
            info.update(first_alarm=float(runs.start.min()), first_alarm_sv=runs.sv.iloc[int(runs.start.argmin())], first_alarm_phase=phase_of(float(runs.start.min()), s0, je))  # the first alarm and its phase
            sp_ = runs[[phase_of(t, s0, je) in ("spoofing phase", "no jamming phase") for t in runs.start]]  # alarms in the spoofing phase
            if len(sp_):  # if: any
                info.update(first_alarm_spoofing_phase=float(sp_.start.min()), first_alarm_spoofing_phase_sv=sp_.sv.iloc[int(sp_.start.argmin())])  # the first one
        return info  # dictionary
    def leads(test: str, rx: str, info: dict) -> dict:  # nested function: lead times in seconds (positive = the alarm came first), defined only for a spoofing-phase alarm in a session that did not start inside the attack and before the gain recovery
        g = gt.loc[test]  # windows
        inprog = bool(T(g.scheduled_start_utc[11:]) <= T(g.capture_start_utc[11:]) < T(g.scheduled_end_utc[11:]))  # attack in progress at capture start
        a = info["first_alarm_spoofing_phase"]  # the alarm
        rc = s3(test, rx, "recovery_utc")  # gain recovery
        ok_ = (not inprog) and np.isfinite(a) and not (np.isfinite(rc) and a > rc)  # lead defined
        out = {"lead_defined": ok_, "lead_over_gain_s3_s": (s3(test, rx, "onset_utc") - a) if ok_ else np.nan}  # lead over the receiver-status script's gain onset
        out["lead_over_spoof_effect_s"] = (T(SPOOF_START_265) - a) if (ok_ and test == "2.6.5") else np.nan  # 2.6.5 only: against the first G30 anomaly
        return out  # dictionary
    lrows = []  # leads at the chosen setting
    for test in SESSIONS:  # for loop over sessions
        for rx in RXS:  # for loop over receivers
            mask, P = chosen[rx]  # the chosen setting
            thr = TTHR[(test, rx, mask)]  # the percentile threshold
            ru = t_runs(TSER[(test, rx, mask)], thr, P)  # all runs
            info = alarms_info(ru, test, rx)  # first alarms
            row = {"test_code": test, "receiver": rx, "mask_deg": mask, "persistence_s": P, "T_threshold": thr, "seconds_with_T": len(TSER[(test, rx, mask)]), "attack_in_progress_at_capture_start": bool(tdf[(tdf.test_code == test) & (tdf.receiver == rx)].attack_in_progress_at_capture_start.iloc[0])}  # fixed columns
            row.update({"first_alarm": ts(info["first_alarm"]), "first_alarm_sv": info["first_alarm_sv"], "first_alarm_phase": info["first_alarm_phase"], "first_alarm_spoofing_phase": ts(info["first_alarm_spoofing_phase"]), "first_alarm_spoofing_phase_sv": info["first_alarm_spoofing_phase_sv"]})  # times as clock text
            row.update(leads(test, rx, info))  # leads over the receiver-status script's gain onset and (2.6.5) the spoofing effect
            t0 = tdf[(tdf.test_code == test) & (tdf.receiver == rx)].iloc[0]  # the part 2 row, for the position and flag events
            a = info["first_alarm_spoofing_phase"]  # the alarm
            for col, ev in (("lead_over_own_position_s", "own_position_first_over_50m"), ("lead_over_solver_norej_s", "solver_norej_first_over_50m"), ("lead_over_flag_s", "first_flag")):  # for loop over the other reference events
                row[col] = (T(t0[ev]) - a) if (row["lead_defined"] and isinstance(t0[ev], str) and t0[ev]) else np.nan  # event minus alarm
            if (test, rx) == ("2.6.12", "UBX_01"):  # if: the u-blox with the shifted time tag, evaluated on host-corrected time
                aa = info["first_alarm"]  # the first alarm of any phase
                for ev, col in (("measurement-engine deception", "lead_over_meas_engine_deception_s"), ("navigation time deception", "lead_over_nav_time_deception_s")):  # for loop over the two traced instants
                    tt = sp[(sp.receiver == "u-blox_01") & (sp.event == ev)].host_time_utc.iloc[0]  # its host time
                    row[col] = (T(tt) - aa) if np.isfinite(aa) else np.nan  # instant minus alarm
                    row[col.replace("lead_over_", "instant_").replace("_s", "")] = tt  # the instant itself
            lrows.append(row)  # kept
    ldf = pd.DataFrame(lrows)  # the table
    ldf["lead_over_gain_s3_at_detection_s"] = ldf.lead_over_gain_s3_s - (ldf.persistence_s - 1)  # the alarm is only certain P-1 seconds after the run starts, so the lead at the detection instant is shorter by that much
    ldf.round(3).to_csv(PROC / "ground_truth" / "early_warning_leads_T.csv", index=False)  # written
    for r in ldf.itertuples():  # for loop over the rows
        put(f"alarm T {r.test_code} {M.RECEIVERS[r.receiver]}", "first alarm (any phase, sv, phase) / first alarm in the spoofing phase / lead over the receiver-status gain onset (s) / lead over spoofing effect (s)", f"{r.first_alarm or '-'} ({r.first_alarm_sv or '-'}, {r.first_alarm_phase or '-'}) / {r.first_alarm_spoofing_phase or '-'} / {r.lead_over_gain_s3_s if r.lead_defined else 'undefined'} / {r.lead_over_spoof_effect_s if r.lead_defined and r.test_code == '2.6.5' else '-'}")  # reported
    # ---- fixed-threshold operating points, schedule margins ----
    ops, tots = [], {}  # per-session rows; totals per (detector, receiver, setting)
    for rx in RXS:  # for loop over receivers
        mask, P = chosen[rx]  # persistence and mask shared by both detectors
        for det, settings in (("residual T", T_FIXED), ("gain fall", GAIN_FIXED[rx])):  # for loop over detectors
            for th in settings:  # for loop over fixed settings
                for test in SESSIONS:  # for loop over sessions
                    spans = schedule_margins(gt, test, rx)  # the schedule-defined margins
                    if det == "residual T":  # if: the residual detector
                        ts_ = TSER[(test, rx, mask)]  # T series
                        ru = t_runs(ts_, th, P)  # all alarms
                        x = ts_[in_spans(ts_.sec.to_numpy(float), spans)] if spans else ts_.iloc[:0]  # T inside the margins
                        n_fa, secs = len(t_runs(x, th, P)), len(x)  # false alarms and seconds
                    else:  # else: the gain detector
                        d = D[(test, rx)]  # deviation series
                        ru = gain_alarm_runs(d, th, P)  # all alarms
                        n_fa = len(gain_alarm_runs(d, th, P, spans)) if spans else 0  # false alarms inside the margins
                        dd = d.dropna()  # seconds with a reading
                        secs = int(in_spans(dd.index.to_numpy(float), spans).sum()) if spans else 0  # margin seconds with data
                    info = alarms_info(ru, test, rx)  # first alarms and phases
                    row = {"detector": det, "receiver": rx, "setting": th, "mask_deg": mask if det == "residual T" else np.nan, "persistence_s": P, "test_code": test, "margin_seconds_with_data": secs, "false_alarm_episodes": n_fa, "first_alarm": ts(info["first_alarm"]), "first_alarm_phase": info["first_alarm_phase"], "first_alarm_spoofing_phase": ts(info["first_alarm_spoofing_phase"])}  # one row
                    row.update(leads(test, rx, info))  # leads
                    row["lead_over_gain_s3_at_detection_s"] = row["lead_over_gain_s3_s"] - (P - 1)  # the lead at the detection instant (the run start plus P-1 seconds)
                    if test == "2.6.5":  # if: the one session with a spoofing effect in the middle of the capture
                        je5 = jam_end("2.6.5")  # the scheduled end of the initial jamming
                        row["alarms_265_after_jamming_before_spoof_effect"] = int(((ru.start > je5 + JAM_TAIL_S) & (ru.start < T(SPOOF_START_265))).sum()) if len(ru) else 0  # alarm runs between the end of the jamming tail and the first G30 anomaly: unlabelled, so not counted as false alarms, but each is an alarm the attack cannot explain
                    ops.append(row)  # kept
                    c = tots.setdefault((det, rx, th), [0, 0])  # totals
                    c[0] += n_fa  # episodes
                    c[1] += secs  # seconds
    odf = pd.DataFrame(ops)  # the table
    odf["total_episodes"] = [tots[(d_, r_, s_)][0] for d_, r_, s_ in zip(odf.detector, odf.receiver, odf.setting)]  # totals attached to every row
    odf["total_margin_hours"] = [tots[(d_, r_, s_)][1] / 3600.0 for d_, r_, s_ in zip(odf.detector, odf.receiver, odf.setting)]  # hours
    odf["false_alarms_per_hour"] = [rate_bound(tots[(d_, r_, s_)][0], tots[(d_, r_, s_)][1] / 3600.0)[0] for d_, r_, s_ in zip(odf.detector, odf.receiver, odf.setting)]  # rate
    odf["upper95_per_hour"] = [rate_bound(tots[(d_, r_, s_)][0], tots[(d_, r_, s_)][1] / 3600.0)[1] for d_, r_, s_ in zip(odf.detector, odf.receiver, odf.setting)]  # exact Poisson bound
    odf.round(4).to_csv(PROC / "ground_truth" / "early_warning_operating_points.csv", index=False)  # written
    for (det, rx, th), (n_ep, secs) in tots.items():  # for loop over the totals
        rate, ub = rate_bound(n_ep, secs / 3600.0)  # per hour
        put(f"operating point {det} {M.RECEIVERS[rx]}", f"setting {th}: false-alarm episodes on schedule margins / hours / per hour (95 % bound)", f"{n_ep} / {secs / 3600.0:.3f} / {rate:.2f} ({ub:.1f})")  # reported
    w5 = (T(SPOOF_START_265) - (jam_end("2.6.5") + JAM_TAIL_S)) / 3600.0  # hours between the end of the jamming tail and the first G30 anomaly in 2.6.5
    put("2.6.5 pre-effect window", "hours between the end of the jamming tail and the first G30 anomaly (12:37:00-12:48:46)", f"{w5:.3f}")  # its length
    for (det, rx, th), (n_ep, secs) in tots.items():  # for loop again, for the 2.6.5 leads
        r5 = odf[(odf.detector == det) & (odf.receiver == rx) & (odf.setting == th) & (odf.test_code == "2.6.5")].iloc[0]  # its 2.6.5 row
        put(f"2.6.5 {det} {M.RECEIVERS[rx]}", f"setting {th}: first alarm (any phase, phase) / first alarm in the spoofing phase / lead over the receiver-status gain onset (s) / lead over the first G30 anomaly (s) / alarm runs before the first G30 anomaly after the jamming tail", f"{r5.first_alarm or '-'} ({r5.first_alarm_phase or '-'}) / {r5.first_alarm_spoofing_phase or '-'} / {r5.lead_over_gain_s3_s:.1f} / {r5.lead_over_spoof_effect_s:.1f} / {int(r5.alarms_265_after_jamming_before_spoof_effect)}")  # reported
    S.update({"alarms_info": alarms_info, "leads": leads, "jam_end": jam_end, "s3": s3, "chosen_elev": chosen, "TSER_elev": TSER, "TTHR_elev": TTHR})  # handed to part 2c
    # ---- the timeline table: the T detector at the chosen setting added ----
    add = ldf[["test_code", "receiver", "mask_deg", "T_threshold", "first_alarm", "first_alarm_sv", "first_alarm_phase", "first_alarm_spoofing_phase", "lead_defined", "lead_over_gain_s3_s", "lead_over_spoof_effect_s", "lead_over_own_position_s", "lead_over_solver_norej_s", "lead_over_flag_s"]].rename(columns=lambda c: c if c in ("test_code", "receiver") else "T_" + c)  # prefixed
    tdf2 = tdf.merge(add, on=["test_code", "receiver"], how="left")  # attached
    tdf2.round(3).to_csv(PROC / "ground_truth" / "early_warning_timeline.csv", index=False)  # rewritten with the T columns
    plot_early_warning_timeline(tdf2, gt)  # the figure on the T alarms
    plot_lead_against_false_alarms(odf, ldf)  # the figure
    check("a false-alarm episode of the percentile detector is counted once per run: totals equal the sum over sessions", "equal", "equal", all(tot[(rx, m, P, mt)][0] == sum(r["episodes"] for r in far if r["receiver"] == rx and r["mask_deg"] == m and r["persistence_s"] == P and r["margin"] == mt and r["test_code"] != "all") for rx in RXS for m in MASKS for P in PERSIST for mt in ("clean_05c", "schedule")))  # internal consistency


def plot_lead_against_false_alarms(odf: pd.DataFrame, ldf: pd.DataFrame) -> None:  # function: the figure, lead time over the receiver-status script's gain onset against false alarms per hour, per detector and receiver
    import matplotlib  # the plotting library
    matplotlib.use("Agg")  # file-only backend
    import matplotlib.pyplot as plt  # pyplot
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), sharey=True)  # one panel per receiver
    names = {"SBF_02": "X5", "SBF_03": "mosaic-H", "UBX_01": "u-blox 01"}  # short receiver names
    col = {"residual T": "#D55E00", "gain fall": "#0072B2"}  # one colour per detector
    mk = {"2.6.5": "o", "2.6.2": "s", "2.6.4": "^", "2.6.10": "v", "2.6.11": "D", "2.6.12": "P"}  # marker per session
    for ax, rx in zip(axes, RXS):  # for loop over the panels
        for det in ("residual T", "gain fall"):  # for loop over detectors
            d = odf[(odf.detector == det) & (odf.receiver == rx) & odf.lead_defined]  # rows with a defined lead
            for test, g in d.groupby("test_code"):  # for loop over the sessions
                g = g.sort_values("false_alarms_per_hour")  # along the false-alarm axis
                ax.plot(g.false_alarms_per_hour, g.lead_over_gain_s3_s, "-" if test == "2.6.5" else ":", color=col[det], lw=1.1, label=f"{det}, {test}")  # one curve per detector and session
                un = g.get("alarms_265_after_jamming_before_spoof_effect", pd.Series(0, index=g.index)).fillna(0) > 0  # settings that also alarmed between the jamming tail and the first G30 anomaly: the alarm cannot be attributed to the attack
                ax.scatter(g.false_alarms_per_hour[~un], g.lead_over_gain_s3_s[~un], marker=mk[test], s=28, color=col[det], edgecolors="black", linewidths=0.4, zorder=3)  # filled markers: first spoofing-phase alarm is not preceded by an unexplained alarm
                ax.scatter(g.false_alarms_per_hour[un], g.lead_over_gain_s3_s[un], marker=mk[test], s=34, facecolors="white", edgecolors=col[det], linewidths=1.2, zorder=3)  # hollow markers: an unexplained alarm came earlier in 2.6.5
        ax.axhline(0, color="black", lw=0.6)  # lead zero: the alarm coincides with the receiver-status script's gain onset
        ax.set_title(names[rx], fontsize=9)  # title
        ax.set_xlabel("false alarms per hour (schedule margins)")  # axis label
        ax.grid(color="#999999", alpha=0.3, lw=0.5)  # faint grid
        ax.legend(fontsize=6, loc="best")  # legend
    axes[0].set_ylabel("lead over the receiver-status gain onset (s)")  # shared axis label
    fig.text(0.5, -0.02, "Hollow markers: the setting also alarmed in 2.6.5 between the jamming tail and the first G30 anomaly. One false alarm is about 4 per hour here (0.2-0.3 h of margin per receiver), so the rates are coarse; a point at 0 means none was seen, not that the rate is zero.", ha="center", fontsize=7)  # the caution, on the figure
    fig.tight_layout()  # spacing
    save_fig(fig, GRAPH / "lead_against_false_alarms.png")  # written, byte-stable


# =====================================================================================================================
# Part 2c: why the Stave margin gives false alarms, and a C/N0-dependent sigma instead of the elevation model
# =====================================================================================================================
CN0_BIN_DB = 2.0  # float constant: C/N0 bin width for the sigma(C/N0) fit
STAVE_PRE = ("15:45:42", "15:49:30")  # tuple: the Stave pre-test margin of the reference positions
BIG_RES_M = 5.0  # float constant: a residual above this many metres counts as large in the diagnosis


def with_cn0(res: pd.DataFrame, cn0: pd.DataFrame) -> pd.DataFrame:  # function: attaches each observation's C/N0 to a residual table and drops rows without one
    out = res.merge(cn0.drop_duplicates(["sec", "sv"]), on=["sec", "sv"], how="left")  # C/N0 of the same satellite and second
    return out.dropna(subset=["cn0"]).reset_index(drop=True)  # rows with a C/N0


def stave_diagnosis(S: dict) -> None:  # function: C/N0 and residuals at Stave against the other sites, the concurrent transmissions and the possibly contaminated 2.6.12 margin gain step
    from scipy.stats import spearmanr  # rank correlation
    RES, CN0, G = S["RES"], S["CN0"], S["G"]  # residuals, C/N0, gains
    a0, a1 = T(STAVE_PRE[0]), T(STAVE_PRE[1]) + 1  # the margin in unix seconds
    for rx in RXS:  # for loop over the receivers
        nm = M.RECEIVERS[rx]  # name
        st = with_cn0(RES[("2.6.12", rx)], CN0[("2.6.12", rx)])  # Stave rows with C/N0
        st = st[(st.sec >= a0) & (st.sec < a1)].copy()  # the pre-test margin
        oth = []  # the other sites' clean rows
        for o in SESSIONS:  # for loop over sessions
            if o == "2.6.12":  # if: Stave itself
                continue  # skip
            r = with_cn0(RES[(o, rx)], CN0[(o, rx)])  # that session's rows with C/N0
            oth.append(r[in_spans(r.sec.to_numpy(float), clean_spans(o, rx))])  # clean ones
        oth = pd.concat(oth, ignore_index=True)  # pooled
        for d_ in (st, oth):  # for loop over both
            d_["bin"] = (np.floor(d_.el / 10.0) * 10).astype(int)  # 10 degree elevation bin
        # C/N0 in matched elevation bins and for the same satellites
        gs = st.groupby(["sv", "bin"]).cn0.agg(["median", "size"]).rename(columns={"median": "m_s", "size": "n_s"})  # Stave per satellite and bin
        go = oth.groupby(["sv", "bin"]).cn0.agg(["median", "size"]).rename(columns={"median": "m_o", "size": "n_o"})  # others per satellite and bin
        j = gs.join(go, how="inner")  # satellites and bins both have
        j = j[(j.n_s >= 20) & (j.n_o >= 20)]  # enough samples on both sides
        j["diff"] = j.m_s - j.m_o  # Stave minus the other sites, dB-Hz
        put(f"Stave C/N0 {nm}", "same satellite and 10 deg bin, Stave pre-test minus other sites' clean: pairs, median difference (dB-Hz), share of pairs lower at Stave, n-weighted median difference", f"{len(j)}, {j['diff'].median():.2f}, {(j['diff'] < 0).mean():.2f}, {np.average(j['diff'], weights=j.n_s):.2f}")  # reported
        bb = pd.concat([st.groupby("bin").cn0.median().rename("s"), oth.groupby("bin").cn0.median().rename("o"), st.groupby("bin").size().rename("ns"), oth.groupby("bin").size().rename("no")], axis=1).dropna()  # every satellite pooled, matched on the 10 degree elevation bin only
        bb = bb[(bb.ns >= 50) & (bb.no >= 50)]  # enough samples in both
        put(f"Stave C/N0 {nm}", "all satellites, elevation bins only: bins, median of (Stave - other sites) per bin (dB-Hz), and by bin (bin start: difference)", f"{len(bb)}, {(bb.s - bb.o).median():.2f}, " + ", ".join(f"{int(b)}: {v:.1f}" for b, v in (bb.s - bb.o).items()))  # reported
        big = st[st.res.abs() > BIG_RES_M]  # the large residuals
        bs = sorted(big.sv.unique())  # their satellites
        jb = j[j.index.get_level_values("sv").isin(bs)]  # the same satellites' differences
        put(f"Stave C/N0 {nm}", f"satellites with |r| > {BIG_RES_M:.0f} m at Stave ({len(bs)}): pairs, median difference (dB-Hz)", f"{', '.join(bs)}: {len(jb)}, {jb['diff'].median() if len(jb) else float('nan'):.2f}")  # reported
        for sv_, g_ in big.groupby("sv"):  # for loop over the satellites with large residuals
            a_s = st[st.sv == sv_]  # all Stave rows of the satellite
            put(f"Stave C/N0 {nm}", f"{sv_}: Stave rows, elevation range (deg), median C/N0 (dB-Hz), rows with |r| > {BIG_RES_M:.0f} m, their median C/N0, rows of that satellite in the other sites' clean data", f"{len(a_s)}, {a_s.el.min():.0f}-{a_s.el.max():.0f}, {a_s.cn0.median():.1f}, {len(g_)}, {g_.cn0.median():.1f}, {int((oth.sv == sv_).sum())}")  # reported
        # do large residuals coincide with low C/N0?
        rho, pv = spearmanr(st.res.abs(), st.cn0)  # rank correlation of |r| with C/N0 at Stave
        rho_o, _ = spearmanr(oth.res.abs(), oth.cn0)  # the same on the other sites' clean data
        within = np.nanmean([spearmanr(g.res.abs(), g.cn0)[0] for _, g in st.groupby("bin") if len(g) >= 100])  # mean rank correlation within elevation bins
        put(f"Stave C/N0 {nm}", "Spearman of |r| with C/N0: Stave (n), p, within 10 deg bins; other sites' clean (n)", f"{rho:.3f} (n={len(st)}), p={pv:.2g}, {within:.3f}; {rho_o:.3f} (n={len(oth)})")  # reported
        put(f"Stave C/N0 {nm}", f"median C/N0 of rows with |r| > {BIG_RES_M:.0f} m against the rest at Stave (dB-Hz), n; share of rows with |r| > {BIG_RES_M:.0f} m at Stave / other sites", f"{big.cn0.median():.1f} (n={len(big)}) vs {st[st.res.abs() <= BIG_RES_M].cn0.median():.1f}; {len(big) / len(st):.4f} / {(oth.res.abs() > BIG_RES_M).mean():.4f}")  # reported
        # the possibly contaminated 2.6.12 margin gain step
        g = G[("2.6.12", rx)].dropna()  # the gain series
        b_ = g[(g.index >= T("15:42:30")) & (g.index < T("15:45:00"))]  # before the step
        a_ = g[(g.index >= a0) & (g.index < a1)]  # in the margin
        put(f"Stave gain {nm}", "median gain 15:42:30-15:45:00, in the margin 15:45:42-15:49:30, largest one-second change 15:44:30-15:45:42 (time)", f"{b_.median():.1f}, {a_.median():.1f}, {g.loc[T('15:44:30'):T('15:45:42')].diff().abs().max():.1f} ({ts(g.loc[T('15:44:30'):T('15:45:42')].diff().abs().idxmax())})")  # reported
    # concurrent transmissions
    ct = pd.read_csv(PROC / "ground_truth" / "concurrent_transmissions.csv")  # the timeline script's overlap table
    ct = ct[ct.our_test_code == "2.6.12"]  # 2.6.12 only
    for r in ct.itertuples():  # for loop over the concurrent tests
        ov = max(0.0, min(T(r.concurrent_end_utc[11:]), a1) - max(T(r.concurrent_start_utc[11:]), a0)) / 60.0  # minutes of overlap with the Stave margin
        put("Stave concurrent transmission", "test, area, window, power (W), bands, overlap with 15:45:42-15:49:30 (min)", f"{r.concurrent_test_code} ({r.concurrent_title}), area {r.concurrent_test_area}, {r.concurrent_start_utc[11:]}-{r.concurrent_end_utc[11:]}, {r.power_w if pd.notna(r.power_w) else 'power not stated'}, {r.bands}, {ov:.1f}")  # reported


def stave_sky(S: dict) -> None:  # function: azimuth and elevation of the satellites with large residuals at Stave during the margin, whether they share an azimuth sector, and the check sky plot
    import matplotlib  # the plotting library
    matplotlib.use("Agg")  # file-only backend
    import matplotlib.pyplot as plt  # pyplot
    a0, a1 = T(STAVE_PRE[0]), T(STAVE_PRE[1]) + 1  # the margin in unix seconds
    site = site_xyz("Stave")  # the Stave reference position
    pr = pd.concat([S["PRS"][("2.6.12", rx)] for rx in ("SBF_02", "UBX_01")], ignore_index=True)  # pseudoranges of the X5 and the u-blox (the satellites are the same, the geometry does not depend on the receiver)
    pr = pr[(pr.sec >= a0) & (pr.sec < a1)].drop_duplicates(["sec", "sv"])  # the margin, one row per satellite and second
    rows = []  # (second, satellite, azimuth, elevation)
    for sec, g in pr.groupby("sec"):  # for loop over the seconds
        svs, t, p_ = list(g.sv), float(g.t_gps.iloc[0]), g.pr.to_numpy()  # satellites, GPS time, pseudoranges
        pos, dts, ok = S["orb"].state(svs, t, p_)  # satellite positions at transmission time
        sel = np.flatnonzero(ok)  # usable satellites
        if len(sel) == 0:  # if: none
            continue  # next second
        _, _, el, az = M.geometry(site, pos[sel])  # elevation and azimuth from the site
        rows += [(int(sec), svs[i], float(np.degrees(az[j]) % 360), float(np.degrees(el[j]))) for j, i in enumerate(sel)]  # one row each
    sky = pd.DataFrame(rows, columns=["sec", "sv", "az", "el"])  # the table
    sky = sky[sky.el >= MASK_DEG]  # above the mask
    hi = ["G02", "G17", "G14", "E10"]  # the four satellites with large residuals (diagnosis above)
    means = {}  # satellite -> mean azimuth, circular
    for sv_ in hi:  # for loop over the four
        g = sky[sky.sv == sv_]  # its track
        ca = np.degrees(np.arctan2(np.sin(np.radians(g.az)).mean(), np.cos(np.radians(g.az)).mean())) % 360  # circular mean azimuth
        means[sv_] = float(ca)  # kept
        put(f"Stave sky {sv_}", "seconds in the margin, azimuth mean (range) (deg), elevation mean (range) (deg)", f"{len(g)}, {ca:.0f} ({g.az.min():.0f}-{g.az.max():.0f}), {g.el.mean():.0f} ({g.el.min():.0f}-{g.el.max():.0f})")  # reported
    az4 = np.sort(np.array(list(means.values())))  # the four mean azimuths in order
    gaps = np.diff(np.r_[az4, az4[0] + 360.0])  # the gaps between neighbours around the circle
    arc = 360.0 - gaps.max()  # the smallest arc that holds all four
    put("Stave sky", "smallest azimuth arc holding the four (deg); a quarter of the horizon is 90", f"{arc:.0f}")  # reported
    others = sky[~sky.sv.isin(hi)].groupby("sv").agg(az=("az", "mean"), el=("el", "mean"), n=("sec", "size"))  # the other satellites above the mask
    start = az4[(int(np.argmax(gaps)) + 1) % 4]  # the azimuth where the four's arc begins (just after the largest gap)
    inside = [sv_ for sv_, a_ in others.az.items() if (a_ - start) % 360 <= arc]  # other satellites whose mean azimuth falls inside that arc
    put("Stave sky", "other satellites above 10 deg in the margin: count; mean azimuth inside the four's arc: count (satellites)", f"{len(others)}; {len(inside)} ({', '.join(inside)})")  # how many others sit in the same sector
    fig = plt.figure(figsize=(5.2, 5.2))  # a square figure
    ax = fig.add_subplot(111, projection="polar")  # polar axes: azimuth around, zenith distance outwards
    ax.set_theta_zero_location("N")  # north at the top
    ax.set_theta_direction(-1)  # azimuth clockwise
    for sv_, g in sky.groupby("sv"):  # for loop over every satellite above the mask
        if sv_ in hi:  # if: one of the four, drawn below in colour
            continue  # skip for now
        ax.plot(np.radians(g.az), 90 - g.el, ".", ms=1.5, color="#BBBBBB")  # grey track
        ax.text(np.radians(g.az.iloc[len(g) // 2]), 90 - g.el.iloc[len(g) // 2], sv_, fontsize=5, color="#777777")  # label
    for sv_, c in zip(hi, ["#D55E00", "#0072B2", "#009E73", "#CC79A7"]):  # for loop over the four, highlighted
        g = sky[sky.sv == sv_]  # its track
        ax.plot(np.radians(g.az), 90 - g.el, ".", ms=3, color=c)  # coloured track
        ax.text(np.radians(g.az.iloc[len(g) // 2]), 90 - g.el.iloc[len(g) // 2], " " + sv_, fontsize=8, color=c, weight="bold")  # label
    ax.set_ylim(0, 80)  # up to the 10 degree mask
    ax.set_yticks([20, 40, 60, 80])  # rings at 70, 50, 30 and 10 degrees elevation
    ax.set_yticklabels(["70", "50", "30", "10"], fontsize=6)  # labelled by elevation
    ax.set_title("Stave 15:45:42-15:49:30: satellites above 10 deg (elevation rings),\nG02, G17, G14, E10 highlighted", fontsize=8)  # title
    save_fig(fig, PROJECT / "Data" / "processed" / "checks" / "stave_skyplot.png")  # the check plot, byte-stable


def fit_sigma_cn0(res: pd.DataFrame) -> tuple:  # function: sigma(C/N0) = sqrt(a + b x 10^(-C/N0/10)) fitted to robust residual variance in 2 dB C/N0 bins (non-negative least squares on the variance)
    from scipy.optimize import nnls  # non-negative least squares
    c, ab = res.cn0.to_numpy(float), np.abs(res.res.to_numpy())  # C/N0 and |residuals|
    xs, ys, ws = [], [], []  # regressor, robust variance, weight
    for c0 in np.arange(np.floor(c.min()), np.ceil(c.max()), CN0_BIN_DB):  # for loop over C/N0 bins
        m = (c >= c0) & (c < c0 + CN0_BIN_DB)  # the bin
        if m.sum() >= 100:  # if: enough samples
            xs.append(float(np.mean(10.0 ** (-c[m] / 10.0))))  # mean of 10^(-C/N0/10) in the bin
            ys.append(float((1.4826 * np.median(ab[m])) ** 2))  # robust variance: (1.4826 x median absolute residual) squared
            ws.append(float(np.sqrt(m.sum())))  # weight
    SC = 1.0e4  # scale of the regressor, so that the two columns have similar size
    A = np.stack([np.ones(len(xs)), np.array(xs) * SC], axis=1) * np.array(ws)[:, None]  # weighted design matrix for the variance
    coef, _ = nnls(A, np.array(ys) * np.array(ws))  # non-negative a and scaled b
    return float(max(coef[0], 0.0025)), float(coef[1] * SC)  # a floor of (5 cm)^2 keeps sigma positive; b back on the 10^(-C/N0/10) scale


def t_stat_cn0(res: pd.DataFrame, ab: tuple) -> pd.DataFrame:  # function: per epoch, T = max over satellites of |r| / sigma(C/N0), and the satellite that holds the maximum
    r = res.reset_index(drop=True).copy()  # fresh index
    r["x"] = r.res.abs() / np.sqrt(ab[0] + ab[1] * 10.0 ** (-r.cn0 / 10.0))  # |r| in sigma units
    idx = r.groupby("sec").x.idxmax()  # the row of the largest in each second
    out = r.loc[idx, ["sec", "x", "sv"]].rename(columns={"x": "T"}).reset_index(drop=True)  # one row per second
    out["sec"] = out.sec.astype(float)  # float seconds
    return out  # the table


def main_part2c(S: dict) -> None:  # function: part 2c
    gt = S["gt"]  # the session windows, handed over by the earlier parts
    stave_diagnosis(S)  # the diagnosis
    stave_sky(S)  # the sky positions of the satellites with large residuals
    RC = {(t, rx): with_cn0(S["RES"][(t, rx)], S["CN0"][(t, rx)]) for t in SESSIONS for rx in RXS}  # residuals with C/N0, 10 degree mask
    AB, THR, TS, rows = {}, {}, {}, []  # fits, thresholds, T series, output rows
    for rx in RXS:  # for loop over receivers
        for test in SESSIONS:  # for loop over the held-out session
            clean = []  # the others' clean residuals
            for o in SESSIONS:  # for loop over the others
                if o == test:  # if: held out
                    continue  # leave out
                r = RC[(o, rx)]  # that session
                clean.append(r[in_spans(r.sec.to_numpy(float), clean_spans(o, rx))])  # inside its clean intervals
            pool = pd.concat(clean, ignore_index=True)  # pooled
            ab = fit_sigma_cn0(pool)  # sigma(C/N0) on the others
            tv = np.concatenate([t_stat_cn0(c, ab)["T"].to_numpy() for c in clean if len(c)])  # T on the others' clean epochs
            AB[(test, rx)], THR[(test, rx)] = ab, float(np.percentile(tv, PCT))  # fit and threshold
            TS[(test, rx)] = t_stat_cn0(RC[(test, rx)], ab)  # T over the held-out session
            rows.append({"receiver": rx, "held_out": test, "a_m2": ab[0], "b_m2": ab[1], "sigma_at_50dBHz_m": float(np.sqrt(ab[0] + ab[1] * 10 ** -5.0)), "sigma_at_30dBHz_m": float(np.sqrt(ab[0] + ab[1] * 10 ** -3.0)), "T_threshold_p999": THR[(test, rx)], "fit_samples": len(pool), "clean_epochs_for_threshold": len(tv)})  # one row
    pd.DataFrame(rows).round(5).to_csv(PROC / "ground_truth" / "early_warning_sigma_cn0.csv", index=False)  # written
    for rx in RXS:  # for loop over receivers
        q = [r for r in rows if r["receiver"] == rx]  # its rows
        put(f"sigma(C/N0) {M.RECEIVERS[rx]}", "median over held-out sessions: a (m^2), b, sigma at 50 and 30 dB-Hz (m), T threshold", f"{np.median([r['a_m2'] for r in q]):.3f}, {np.median([r['b_m2'] for r in q]):.1f}, {np.median([r['sigma_at_50dBHz_m'] for r in q]):.2f}, {np.median([r['sigma_at_30dBHz_m'] for r in q]):.2f}, {np.median([r['T_threshold_p999'] for r in q]):.2f}")  # reported
    far, tot = [], {}  # false-alarm rows and totals
    for rx in RXS:  # for loop over receivers
        for P in PERSIST:  # for loop over persistences
            for test in SESSIONS:  # for loop over sessions
                for mt, spans in (("clean_05c", clean_spans(test, rx)), ("schedule", schedule_margins(gt, test, rx))):  # for loop over the two margin kinds
                    if not spans:  # if: none
                        continue  # skip
                    ts_ = TS[(test, rx)]  # T series
                    x = ts_[in_spans(ts_.sec.to_numpy(float), spans)]  # inside the margin
                    n_ep, secs = len(t_runs(x, THR[(test, rx)], P)), len(x)  # episodes, seconds
                    rate, ub = rate_bound(n_ep, secs / 3600.0)  # per hour
                    far.append({"detector": "T percentile, sigma(C/N0)", "receiver": rx, "persistence_s": P, "margin": mt, "test_code": test, "T_threshold": THR[(test, rx)], "seconds_with_data": secs, "episodes": n_ep, "per_hour": rate, "upper95_per_hour": ub})  # one row
                    c = tot.setdefault((rx, P, mt), [0, 0])  # totals
                    c[0] += n_ep  # episodes
                    c[1] += secs  # seconds
            for mt in ("clean_05c", "schedule"):  # for loop over margin kinds
                n_ep, secs = tot[(rx, P, mt)]  # totals
                rate, ub = rate_bound(n_ep, secs / 3600.0)  # per hour
                far.append({"detector": "T percentile, sigma(C/N0)", "receiver": rx, "persistence_s": P, "margin": mt, "test_code": "all", "T_threshold": np.nan, "seconds_with_data": secs, "episodes": n_ep, "per_hour": rate, "upper95_per_hour": ub})  # total row
                put(f"false alarms C/N0 {M.RECEIVERS[rx]}", f"persistence {P} s, {mt} margins: episodes / hours / per hour (95 % bound)", f"{n_ep} / {secs / 3600.0:.3f} / {rate:.2f} ({ub:.1f})")  # reported
    fdf = pd.DataFrame(far)  # the table
    fdf.round(4).to_csv(PROC / "ground_truth" / "early_warning_false_alarms_cn0.csv", index=False)  # written
    for r in fdf[(fdf.test_code != "all") & (fdf.margin == "clean_05c") & (fdf.episodes > 0)].itertuples():  # for loop over the sessions with false alarms
        put(f"false alarms C/N0 {M.RECEIVERS[r.receiver]}", f"session {r.test_code}, persistence {r.persistence_s} s, clean intervals: episodes / seconds with data / per hour (95 % bound)", f"{r.episodes} / {r.seconds_with_data} / {r.per_hour:.1f} ({r.upper95_per_hour:.1f})")  # which sessions
    chosen = {}  # receiver -> persistence
    for rx in RXS:  # for loop over receivers
        ok_p = [P for P in PERSIST if tot[(rx, P, "clean_05c")][0] == 0]  # persistences with zero false alarms on clean data
        chosen[rx] = ok_p[0] if ok_p else PERSIST[-1]  # the lowest, else 10 s
        n_ep, secs = tot[(rx, chosen[rx], "clean_05c")]  # totals at the chosen persistence
        put(f"setting C/N0 {M.RECEIVERS[rx]}", "chosen persistence (s); residual false alarms per hour on clean data (95 % bound)", f"{chosen[rx]}; {n_ep / (secs / 3600.0):.2f} ({rate_bound(n_ep, secs / 3600.0)[1]:.1f})" + ("" if ok_p else " (false alarms remain)"))  # reported
    S["zero_cn0"] = all(tot[(rx, chosen[rx], "clean_05c")][0] == 0 for rx in RXS)  # whether every receiver reaches zero
    put("sigma(C/N0) outcome", "every receiver reaches zero false alarms on all held-out clean intervals", S["zero_cn0"])  # the decision input
    lrows = []  # leads at the chosen setting
    for test in SESSIONS:  # for loop over sessions
        for rx in RXS:  # for loop over receivers
            P = chosen[rx]  # persistence
            ru = t_runs(TS[(test, rx)], THR[(test, rx)], P)  # runs
            info = S["alarms_info"](ru, test, rx)  # first alarms and phases
            row = {"test_code": test, "receiver": rx, "persistence_s": P, "T_threshold": THR[(test, rx)], "seconds_with_T": len(TS[(test, rx)]), "first_alarm": ts(info["first_alarm"]), "first_alarm_sv": info["first_alarm_sv"], "first_alarm_phase": info["first_alarm_phase"], "first_alarm_spoofing_phase": ts(info["first_alarm_spoofing_phase"])}  # fixed columns
            row.update(S["leads"](test, rx, info))  # lead times
            row["lead_over_gain_s3_at_detection_s"] = row["lead_over_gain_s3_s"] - (P - 1)  # at the detection instant
            if test == "2.6.5":  # if: 2.6.5, the unlabelled stretch
                je5 = S["jam_end"]("2.6.5")  # end of the initial jamming
                row["alarm_runs_265_after_jamming_before_spoof_effect"] = int(((ru.start > je5 + JAM_TAIL_S) & (ru.start < T(SPOOF_START_265))).sum()) if len(ru) else 0  # alarms in the unlabelled stretch
            lrows.append(row)  # kept
    ldf = pd.DataFrame(lrows)  # the table
    ldf.round(3).to_csv(PROC / "ground_truth" / "early_warning_leads_cn0.csv", index=False)  # written
    for r in ldf[ldf.test_code == "2.6.5"].itertuples():  # for loop over the 2.6.5 rows
        put(f"2.6.5 C/N0 T {M.RECEIVERS[r.receiver]}", "first alarm (phase) / spoofing-phase alarm / lead over the receiver-status gain onset (s) / at detection (s) / lead over first G30 anomaly (s) / alarm runs in the unlabelled stretch", f"{r.first_alarm} ({r.first_alarm_phase}, {r.first_alarm_sv}) / {r.first_alarm_spoofing_phase} / {r.lead_over_gain_s3_s:.1f} / {r.lead_over_gain_s3_at_detection_s:.1f} / {r.lead_over_spoof_effect_s:.1f} / {int(r.alarm_runs_265_after_jamming_before_spoof_effect)}")  # reported
    for r in ldf[ldf.test_code != "2.6.5"].itertuples():  # for loop over the other sessions
        put(f"{r.test_code} C/N0 T {M.RECEIVERS[r.receiver]}", "first alarm (phase, sv) / lead over the receiver-status gain onset (s)", f"{r.first_alarm or '-'} ({r.first_alarm_phase or '-'}, {r.first_alarm_sv or '-'}) / {r.lead_over_gain_s3_s if r.lead_defined else 'undefined'}")  # reported


def ubx_2612_facts(S: dict) -> None:  # function: what u-blox 01's host-corrected 2.6.12 residuals show after its time tag shifted, for 5.3's prose
    from helpers import write_facts  # import inside the function: the facts-table writer (no earlier line of this script moves)
    r = S["RES"][("2.6.12", "UBX_01")]  # the host-corrected residual table of u-blox 01 in 2.6.12 (sec, sv, res, el)
    sec = np.unique(r.sec[r.sec >= T(UBX_2612_END)].to_numpy())  # np.unique: the sorted seconds from the shift on that hold at least one residual
    run = np.cumsum(np.r_[True, np.diff(sec) != 1])  # run number: a new run wherever a second is missing
    size = pd.Series(run).map(pd.Series(run).value_counts()).to_numpy()  # map of value_counts: each second's run length
    first = float(sec[size >= ANOM_RUN][0])  # the first second of the first run of ANOM_RUN consecutive seconds with a residual: tracking resumes
    sp = S["sp"]  # the receiver-status script's traced 2.6.12 instants
    rel = sp[(sp.receiver == "u-blox_01") & (sp.event == "measurement-engine release")].host_time_utc.iloc[0]  # the end of the counterfeit measurement time
    w = r[(r.sec >= first) & (r.sec < T(rel))]  # residuals from the resumption to the release
    mag = float(w.res.abs().groupby(w.sec).median().median())  # median over seconds of each second's median |residual|, m
    # write_facts with a list of 4-tuples: key, value, unit, definition
    write_facts(__file__, [("ubx2612_residuals_resume", ts(first), "UTC", f"first second after {UBX_2612_END} that starts a run of {ANOM_RUN} consecutive seconds with a residual (four satellites of one constellation above {MASK_DEG:.0f} degrees)"),
                           ("ubx2612_measurement_engine_release", rel, "UTC", "u-blox 01's measurement-engine release (ground_truth/spoof_2612_timeline.csv)"),  # the window's end, as the receiver-status script traced it
                           ("ubx2612_residual_magnitude", mag, "m", "median over seconds of the per-second median |residual| from the resumption to the measurement-engine release")])  # written as 04_gain_and_position_evidence_facts.csv


def main() -> None:  # function: part 1, the G30 case study, then part 2, the timeline
    print("part 1: G30 case study")  # heading
    main_part1()  # runs part 1
    print("part 2: early-warning timeline")  # heading
    S = main_part2()  # runs part 2
    print("part 2b: detector statistic, operating points")  # heading
    main_part2b(S)  # runs part 2b
    print("part 2c: Stave diagnosis, sigma(C/N0)")  # heading
    main_part2c(S)  # runs part 2c
    ubx_2612_facts(S)  # u-blox 01's 2.6.12 values for 5.3's prose, to Data/processed/printed_facts/
    pd.DataFrame(ROWS + CHECKS).to_csv(PROC / "ground_truth" / "early_warning_summary.csv", index=False)  # the numbers of parts 2 and 2b, long form


if __name__ == "__main__":  # guard: runs main() only when the file is executed, not when imported
    main()  # entry point
    exit_if_failed(FAILED, "1_ground_truth/04_gain_and_position_evidence.py")  # the helpers module: status 1 when any self-check failed, so a chained run stops
