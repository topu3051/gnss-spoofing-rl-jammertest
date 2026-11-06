#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
Diagnostics behind two exclusions of the reference-position intervals (run after codes/2_preprocessing/03_reference_positions.py).

1. u-blox 01, 2.6.2, satellite G30: a constant code error of about -559 m from the first second of the capture, at 21-23 dB-Hz and
   without carrier phase, while the mosaic-X5 and mosaic-H track the true signal at 43-47 dB-Hz. Is it a deception remnant or a
   false lock? Evidence: when does it normalise, what changes at that moment, does the Doppler agree with the true Doppler, was
   does any other satellite show an offset, what does the transmission plan say. A second function (g30_onset) traces G30 against the
   mosaic-H through 2.6.5 and across the gap, to find when the error first appears; a third (mask20) repeats the same-antenna check at a 20 degree mask.
2. mosaic-X5 and mosaic-H, 2.6.10, satellite G08, 14:44:41-14:44:43: three seconds with a -30 m code error that tipped the mosaic-H
   solution; is it a reacquisition transient after the end of spoofed tracking in 2.6.10 capture?

3. the per-minute residual spread of the conversion script's minute table for 2.6.5, 12:28 to 12:32 (the stop at Stave), recomputed with the conversion script's
   clock_terms using the Stave position of the stop (the motion script) instead of the 2.6.2 site that the conversion script assumed. Diagnostic only; the
   interval stays excluded (the 2.6.5 drive exclusion).

4. L1 after 15:55:00 in 2.6.12: the catalogue has the motorcade's jamming on L1, G1 and B1 stop after five minutes, so 15:55:00-16:05:00
   is spoofing alone. Is the L1 front end still deflected against the margins, and how deep? (Decides whether the expected 46 dB is the
   spoofing's.)

Reads   Data/rinex/*/*.crx.gz (RINEX observations, decompressed in memory), Data/processed/preprocessing/reference_positions.csv (site estimates)
Writes  Data/processed/checks/g30_series.csv       G30 per second on the three receivers (residual, code, carrier present, Doppler, C/N0, lock flags)
        Data/processed/checks/g30_diagnostics.csv      every number quoted in about the two events, in long form
        Data/processed/checks/g30_diagnostics_checks.csv  the constants of the position script checked against what the data show
Residuals use the site estimate as a fixed position, the IGS orbits and the same corrections as the position script; the clock of each constellation is the
median of the other satellites of that epoch.
"""

from __future__ import annotations  # future import: type hints kept as text (a convention across the project's scripts)

import sys  # standard-library system access: sys.path
from pathlib import Path  # Path objects for building file paths

PROJECT = Path(__file__).resolve().parent.parent.parent  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
sys.path.insert(0, str(PROJECT / "codes" / "common"))  # puts codes/ first on the import path

import numpy as np  # NumPy: arrays and medians
import pandas as pd  # pandas: tables and timestamps
import importlib.util  # standard-library tools for loading a Python file by its path
import datetime as dtm  # standard-library dates, for the day of the recording
from datetime import datetime, timezone  # date-time classes for the RINEX epoch times

PROC = PROJECT / "Data" / "processed"  # processed CSVs
CHECK_DIR = PROJECT / "Data" / "processed" / "checks"  # where the outputs go
LAMBDA_L1 = 299792458.0 / 1575.42e6  # float constant: the L1 wavelength in metres, to turn a Doppler into a range rate


def load_module(name: str, filename: str):  # function: loads codes/<filename> as a module (names start with a digit)
    spec = importlib.util.spec_from_file_location(name, PROJECT / "codes" / filename)  # where to load from
    mod = importlib.util.module_from_spec(spec)  # an empty module object
    sys.modules[name] = mod  # registered
    spec.loader.exec_module(mod)  # runs the file's top level (constants and functions)
    return mod  # the module


M = load_module("reference_positions_for_satellite_diagnostics", "2_preprocessing/03_reference_positions.py")  # the position script: orbits, corrections, geometry, observation reader
S04 = M.load_module("convert_rinex_for_satellite_diagnostics", "2_preprocessing/01_convert_rinex.py")  # the conversion script for the orbit functions
ORB = M.Orbits(S04, dtm.date(2026, 9, 14))  # satellite orbits and clocks from the IGS file
KL = M.igs_klobuchar(S04.IGS_BRDC)  # the GPS Klobuchar parameters
GT = pd.read_csv(PROC / "ground_truth" / "session_ground_truth.csv").set_index("test_code")  # session folders
POS = pd.read_csv(PROC / "preprocessing" / "reference_positions.csv")  # the site estimates of the position script
ROWS: list = []  # long-form output rows: item, quantity, value


def put(item: str, quantity: str, value) -> None:  # function: records one number and prints it
    ROWS.append({"item": item, "quantity": quantity, "value": value})  # one row
    print(f"  {item:34s} {quantity:58s} {value}")  # printed for the log


def T(hms: str) -> float:  # function: "HH:MM:SS" on the recording date -> unix seconds
    return pd.Timestamp(f"2026-09-14 {hms}", tz="UTC").timestamp()  # a UTC timestamp as seconds


def site_x(site: str) -> np.ndarray:  # function: a site's ECEF estimate from preprocessing/reference_positions.csv
    r = POS[(POS.scope == "site") & (POS.site == site)].iloc[0]  # the site row
    return np.array([r.x_m, r.y_m, r.z_m])  # x, y, z


def raw_sat(test: str, rx: str, sv: str, t0: float, t1: float) -> pd.DataFrame:  # function: one satellite's RINEX observations with lock flags, first epoch of each second
    import hatanaka  # the compression package
    folder = GT.loc[test, "session_folder"]  # the session folder
    stem = M.OBS_FILES.get(test, {}).get(rx, rx)  # the stream's file part
    path = PROJECT / "Data" / "rinex" / folder / f"{folder.split('_T_')[0]}_{stem}.crx.gz"  # the observation file
    lines = hatanaka.decompress(path).decode("utf-8", errors="replace").split("\n")  # its lines
    codes, cur, k = {}, None, 0  # observation codes per constellation
    for k, l in enumerate(lines):  # for loop over the header
        lab = l[60:].strip()  # its label
        if lab == "SYS / # / OBS TYPES":  # if: an observation-types record
            if l[0] != " ":  # if: a new constellation
                cur = l[0]  # its letter
                codes[cur] = l[7:60].split()  # first codes
            else:  # else: continuation
                codes[cur] += l[7:60].split()  # more codes
        elif lab == "END OF HEADER":  # elif: header ends
            break  # stop
    ci = codes[sv[0]]  # this constellation's codes
    rows, ok, last, ts = [], False, None, 0.0  # output rows and epoch state
    for l in lines[k + 1:]:  # for loop over the data
        if l[:1] == ">":  # if: an epoch line
            tg = (datetime(int(l[2:6]), int(l[7:9]), int(l[10:12]), int(l[13:15]), int(l[16:18]), tzinfo=timezone.utc) - M.GPS_EPOCH).total_seconds() + float(l[18:29])  # GPS seconds
            ts = tg - M.LEAP_SECONDS + M.GPS_UNIX  # unix seconds
            ok = t0 <= ts < t1 and int(np.floor(ts)) != last  # inside the window and the first epoch of this second
            if ok:  # if: kept
                last = int(np.floor(ts))  # this second is taken
        elif ok and l[:3] == sv:  # elif: this satellite's line in a kept epoch
            d = {"sec": int(np.floor(ts))}  # the row
            for i, c in enumerate(ci):  # for loop over the codes
                f = l[3 + 16 * i:3 + 16 * i + 16]  # value, LLI and SSI characters
                d[c] = float(f[:14]) if f[:14].strip() else np.nan  # the value
                d[c + "_lli"] = f[14:15].strip()  # loss-of-lock indicator
            rows.append(d)  # kept
    return pd.DataFrame(rows)  # the table


def residuals(test: str, rx: str, t0: float, t1: float, x: np.ndarray) -> pd.DataFrame:  # function: per-satellite residual per second against a fixed position, clock = median of the other satellites
    spans = [(pd.Timestamp(t0, unit="s", tz="UTC"), pd.Timestamp(t1, unit="s", tz="UTC"))]  # the window
    stem = M.OBS_FILES.get(test, {}).get(rx, rx)  # the stream's file part
    folder = GT.loc[test, "session_folder"]  # the session folder
    obs = M.read_obs(PROJECT / "Data" / "rinex" / folder / f"{folder.split('_T_')[0]}_{stem}.crx.gz", spans, M.OBS_CODES[rx[:3]])  # observations
    out = []  # (second, satellite, z, elevation)
    for sec, g in obs[obs.sv.str[0].isin(["G", "E"])].groupby("sec"):  # for loop over the seconds
        svs, t = list(g.sv), float(g.t_gps.iloc[0])  # satellites and time
        P, D, ok = ORB.state(svs, t, g.pr.to_numpy())  # positions and clocks
        sel = np.flatnonzero(ok)  # usable
        rho, e, el, az = M.geometry(x, P[sel])  # geometry
        z = g.pr.to_numpy()[sel] - rho + M.C_LIGHT * D[sel] - M.corrections(x, t, [svs[i] for i in sel], el, az, KL)  # observed minus modelled, clock still in
        out += [(sec, svs[i], zz, float(np.degrees(e_))) for i, zz, e_ in zip(sel, z, el)]  # rows
    d = pd.DataFrame(out, columns=["sec", "sv", "z", "el"])  # the table
    d["sys"] = d.sv.str[0]  # constellation
    clk = d.groupby(["sec", "sys"]).z.transform("median")  # per-epoch clock: median over all satellites of the constellation
    d["res"] = d.z - clk  # residual about that clock
    return d  # the table


def first_normal(d: pd.DataFrame, after: float) -> int:  # function: the first second after a time with a residual within 20 m
    return int(d[(d.sec > after) & (d.res.abs() < 20)].sec.iloc[0])  # that second


def hms(sec: float) -> str:  # function: unix seconds -> HH:MM:SS
    return datetime.fromtimestamp(sec, timezone.utc).strftime("%H:%M:%S")  # clock text


def g30() -> dict:  # function: the G30 diagnostic; returns the instants the position script uses
    print("1. u-blox 01, 2.6.2, G30")  # heading
    x = site_x("2.6.2 site")  # the 2.6.2 site estimate
    ub = residuals("2.6.2", "UBX_01", T("12:56:58"), T("13:12:00"), x)  # u-blox residuals
    g = ub[ub.sv == "G30"].copy()  # G30 on u-blox
    # the clock of each epoch must not include G30 itself: recompute it from the other satellites
    clk = ub[ub.sv != "G30"].groupby(["sec", "sys"]).z.median().rename("clk2").reset_index()  # clock from the other satellites
    g = g.merge(clk, on=["sec", "sys"], how="left")  # attached
    g["res"] = g.z - g.clk2  # residual about the other satellites' clock
    raw = raw_sat("2.6.2", "UBX_01", "G30", T("12:56:58"), T("13:12:00"))  # code, carrier, Doppler, C/N0, flags
    g = g.merge(raw, on="sec", how="outer").sort_values("sec")  # joined by second (outer: a second can have raw but no residual)
    g["hms"] = g.sec.map(hms)  # clock text
    norm = first_normal(g.dropna(subset=["res"]), T("13:00:00"))  # the reacquisition second
    before = g[(g.sec >= T("12:56:58")) & (g.sec < norm)]  # the counterfeit stretch
    after = g[(g.sec >= norm) & (g.sec < norm + 120)]  # the two minutes after
    put("G30 u-blox", "first second tracked", before.dropna(subset=["C1C"]).hms.iloc[0])  # when it starts
    put("G30 u-blox", "code error before reacquisition, median (m)", round(float(before.res.median()), 1))  # about -559
    put("G30 u-blox", "code error before reacquisition, min and max (m)", f"{before.res.min():.1f} {before.res.max():.1f}")  # its range
    put("G30 u-blox", "reacquisition second (first with error under 20 m)", hms(norm))  # when it normalises
    put("G30 u-blox", "C/N0 before / after (dB-Hz, median)", f"{before.S1C.median():.0f} / {after.S1C.median():.0f}")  # the step
    put("G30 u-blox", "carrier phase present before / after (fraction of seconds)", f"{before.L1C.notna().mean():.2f} / {after.L1C.notna().mean():.2f}")  # carrier appears
    put("G30 u-blox", "loss-of-lock flags set on C1C or L1C, before / after", f"{int((before.C1C_lli != '').sum() + (before.L1C_lli.fillna('') != '').sum())} / {int((after.C1C_lli.fillna('') != '').sum() + (after.L1C_lli.fillna('') != '').sum())}")  # LLI counts
    span_s = int(norm - T("12:56:59"))  # seconds from the first tracked second to the reacquisition
    miss = span_s - int(before.C1C.notna().sum())  # seconds in that stretch without a G30 code observation
    put("G30 u-blox", "seconds without a G30 code observation in the counterfeit stretch (of %d)" % span_s, miss)  # tracking dropouts
    # Doppler against the true Doppler: fit a line to the true Doppler after the reacquisition and extrapolate back to the last counterfeit second
    tr = after.dropna(subset=["D1C"]).head(40)  # true Doppler, first 40 s
    lastfalse = before.dropna(subset=["D1C"]).tail(1)  # the last counterfeit second
    k1, k0 = np.polyfit(tr.sec - norm, tr.D1C, 1)  # slope and intercept of the true Doppler
    pred = k0 + k1 * float(lastfalse.sec.iloc[0] - norm)  # true Doppler extrapolated to that second
    put("G30 u-blox", "Doppler of the counterfeit at its last second (Hz)", round(float(lastfalse.D1C.iloc[0]), 1))  # observed
    put("G30 u-blox", "true Doppler extrapolated to that second (Hz)", round(float(pred), 1))  # expected
    put("G30 u-blox", "difference (Hz)", round(float(lastfalse.D1C.iloc[0] - pred), 1))  # how close
    # self-consistency of the counterfeit: range rate from successive codes against -lambda * Doppler
    cc = before.dropna(subset=["C1C", "D1C"]).copy()  # seconds with both
    cc = cc[cc.sec.diff() == 1]  # consecutive seconds only
    rr = np.diff(before.dropna(subset=["C1C"]).C1C.to_numpy())  # code differences
    dd = -LAMBDA_L1 * before.dropna(subset=["C1C"]).D1C.to_numpy()[1:]  # Doppler range rates
    ok = np.diff(before.dropna(subset=["C1C"]).sec.to_numpy()) == 1  # consecutive pairs
    put("G30 u-blox", "code rate minus Doppler rate, median over consecutive seconds (m/s)", round(float(np.median((rr - dd)[ok])), 2))  # near zero
    # the X5 and the mosaic-H on the same seconds
    for rx, name in (("SBF_02", "mosaic-X5"), ("SBF_03", "mosaic-H")):  # for loop over the Septentrio receivers
        rw = raw_sat("2.6.2", rx, "G30", T("12:56:58"), norm)  # their raw G30
        rd = residuals("2.6.2", rx, T("12:56:58"), norm, x)  # their residuals
        put(f"G30 {name}", "C/N0 over the same seconds (dB-Hz, median)", round(float(rw.S1C.median()), 1))  # strong
        put(f"G30 {name}", "residual over the same seconds (m, median)", round(float(rd[rd.sv == "G30"].res.median()), 1))  # small
    # other satellites of u-blox 01 in 12:56:58-12:59:30, above the 10 degree mask
    w = ub[(ub.sec < T("12:59:30")) & (ub.el >= 10) & (ub.sv != "G30")]  # the others
    w = w.merge(clk, on=["sec", "sys"], how="left")  # their clocks from the others
    med = w.assign(r=w.z - w.clk2).groupby("sv").r.median().abs()  # median residual magnitude per satellite
    put("other u-blox satellites", "largest median |residual| above 10 degrees, 12:56:58-12:59:30 (m)", f"{med.idxmax()} {med.max():.1f}")  # none stands out
    # the transmission plan between 12:54 and 13:03
    s1 = load_module("timeline_for_satellite_diagnostics", "1_ground_truth/01_timeline.py")  # the timeline script for the schedule rule
    from helpers import load_schedule  # the schedule loader
    sch, _ = load_schedule(s1, PROC)  # the transmission plan with effective powers
    w0, w1 = pd.Timestamp("2026-09-14 12:54:00", tz="UTC"), pd.Timestamp("2026-09-14 13:03:00", tz="UTC")  # the window of interest
    for e in sch:  # for loop over the entries
        if pd.Timestamp(e["start_utc"]) < w1 and pd.Timestamp(e["end_utc"]) > w0:  # if: overlaps 12:54-13:03
            put("plan entry 12:54-13:03", f"Test Area {e['test_area']}", f"{e['title']} {e['start_utc']:%H:%M}-{e['end_utc']:%H:%M} {e['power_w']} W {e['bands']}")  # recorded
    g[["hms", "res", "C1C", "L1C", "D1C", "S1C", "C1C_lli", "L1C_lli"]].round(3).to_csv(CHECK_DIR / "g30_series.csv", index=False)  # the series
    return {"normal": hms(norm)}  # the instant the position script uses


def g08() -> dict:  # function: the G08 diagnostic in 2.6.10
    print("2. mosaic-X5 and mosaic-H, 2.6.10, G08")  # heading
    x = site_x("south-west site")  # the south-west site estimate
    last = None  # the last anomalous second
    first = None  # the first anomalous second over the Septentrio receivers (checked against ground_truth/decision_times.csv)
    for rx, name in (("SBF_02", "mosaic-X5"), ("SBF_03", "mosaic-H"), ("UBX_01", "u-blox 01")):  # for loop over the receivers
        d = residuals("2.6.10", rx, T("14:44:30"), T("14:45:03"), x)  # residuals
        c = d[d.sv != "G08"].groupby(["sec", "sys"]).z.median().rename("clk2").reset_index()  # clock from the others
        g = d[d.sv == "G08"].merge(c, on=["sec", "sys"]).assign(r=lambda q: q.z - q.clk2)  # G08 about that clock
        g["hms"] = g.sec.map(hms)  # clock text
        raw = raw_sat("2.6.10", rx, "G08", T("14:44:30"), T("14:45:03"))  # raw observations
        bad = g[g.r.abs() > 10]  # seconds with a code error over 10 m
        put(f"G08 {name}", "seconds tracked in 14:44:30-14:45:02 (count) and first", f"{len(g)} {g.hms.iloc[0]}")  # when it is seen
        put(f"G08 {name}", "seconds with |code error| over 10 m", f"{len(bad)} {bad.hms.iloc[0] if len(bad) else '-'} to {bad.hms.iloc[-1] if len(bad) else '-'}")  # the transient
        if len(bad):  # if: some
            put(f"G08 {name}", "code error in those seconds (m)", " ".join(f"{v:.1f}" for v in bad.r))  # values
            put(f"G08 {name}", "C/N0 in those seconds (dB-Hz)", " ".join(f"{v:.1f}" for v in raw[raw.sec.isin(bad.sec)].S1C))  # strength
            put(f"G08 {name}", "code error after reacquisition 14:44:51-14:45:02 (m, median)", round(float(g[g.sec >= T("14:44:51")].r.median()), 1))  # back to normal
            if rx != "UBX_01":  # if: a Septentrio receiver (the docstring's G08 instant is theirs)
                first = int(bad.sec.iloc[0]) if first is None else min(first, int(bad.sec.iloc[0]))  # conditional expression: the earliest first anomalous second
                last = int(bad.sec.iloc[-1]) if last is None else max(last, int(bad.sec.iloc[-1]))  # conditional expression: the latest last anomalous second over the Septentrio receivers, not the last one in loop order
    return {"last": hms(last), "first": hms(first)}  # for the position script and the decision-times check


def stave_stop_spread() -> None:  # function: the 2.6.5 minutes 12:28-12:32 with the Stave position instead of the 2.6.2 site
    print("3. residual spread per minute, 2.6.5 12:28-12:32, 2.6.2 site against Stave-stop position")  # heading
    vs = pd.read_csv(PROC / "preprocessing" / "vehicle_motion_stops.csv")  # the motion script's stop positions
    st = vs[(vs.scope == "OBD stop") & (vs.segment == 2)].iloc[0]  # the stop at Stave
    sv_ = POS[(POS.scope == "site") & (POS.site == "Stave")].iloc[0]  # the Stave reference estimate, for its height
    x_stave = M.llh_to_ecef(float(st.median_lat), float(st.median_lon), float(sv_.h_m))  # the stop's position at the Stave height
    x_262 = site_x("2.6.2 site")  # the position the conversion script assumed for the whole session
    old = pd.read_csv(PROC / "preprocessing" / "rinex_residual_minutes.csv", low_memory=False)  # the conversion script's minute table
    old = old[(old.test_code == "2.6.5") & (old.site_part == "moving")]  # the minutes it labels moving
    for rx, stream, name in (("SBF_02", "SBF_02", "mosaic-X5"), ("SBF_03", "SBF_03", "mosaic-H"), ("UBX_01", "UBX_01", "u-blox 01")):  # for loop over the three receivers
        spans = [(pd.Timestamp("2026-09-14 12:28:00", tz="UTC"), pd.Timestamp("2026-09-14 12:33:00", tz="UTC"))]  # the five minutes
        folder = GT.loc["2.6.5", "session_folder"]  # the session folder
        obs = M.read_obs(PROJECT / "Data" / "rinex" / folder / f"{folder.split('_T_')[0]}_{M.OBS_FILES.get('2.6.5', {}).get(rx, rx)}.crx.gz", spans, M.OBS_CODES[rx[:3]])  # 1 Hz observations
        obs = obs[obs.sv.str[0].isin(["G", "E"])].assign(key=lambda q: q.sec)  # GPS and Galileo, keyed by the second (the conversion script uses these two constellations)
        for label, x in (("2.6.2 site", x_262), ("Stave stop", x_stave)):  # for loop over the two assumed positions
            lat, lon, _ = M.ecef_to_llh(*x)  # the geodetic position
            up = M.enu_axes(lat, lon)[2]  # the local up vector
            ct = S04.clock_terms(obs[["key", "t_gps", "sv", "pr"]], ORB.keph, x, up)  # the conversion script's per-epoch common clock term and residual spread
            ct["minute"] = pd.to_datetime(ct.key, unit="s", utc=True).dt.strftime("%H:%M")  # the minute of each epoch
            per = ct.groupby("minute").res_mad_m.median()  # the median spread per minute, as the conversion script's table defines it
            for mn, v in per.items():  # for loop over the minutes
                put(f"{name} {label}", f"minute {mn} median residual spread (m)", round(float(v), 1))  # recorded
        o = old[(old.stream == stream) & old.minute_utc.astype(str).str[11:16].between("12:28", "12:32")]  # the conversion script's own values
        put(f"{name} minute table of the conversion script", "minutes 12:28-12:32 range (m)", f"{o.residual_mad_m_median.min():.1f} to {o.residual_mad_m_median.max():.1f}")  # the reproduction target


def l1_spoof_only() -> None:  # function: the L1 deflection in 2.6.12 after 15:55:00 against the margins
    print("4. L1 front end in 2.6.12, spoofing only (15:55:00-16:05:00) against the margins")  # heading
    cols = ["test_code", "receiver", "timestamp_utc", "band", "agc_gain_db", "agc_cnt", "noise_per_ms", "antenna_id"]  # needed columns
    st = pd.read_csv(PROC / "ground_truth" / "receiver_status_timeseries.csv", low_memory=False, usecols=cols)  # the status readings
    st = st[st.test_code == "2.6.12"].copy()  # this session
    st["t"] = pd.to_datetime(st.timestamp_utc, utc=True)  # times as UTC timestamps
    t_ = lambda x: pd.Timestamp(f"2026-09-14 {x}", tz="UTC")  # clock text -> timestamp
    # for loop over a tuple of tuples: visits each Septentrio receiver's L1 gain column so the margin test runs on every one.
    for rx, name, band, fld in (("Septentrio_02", "mosaic-X5", "GPSL1/E1", "agc_gain_db"), ("Septentrio_03", "mosaic-H", "GPSL1/E1", "agc_gain_db"),
                                ("Septentrio_01", "mosaic-G5", "GPSL1/E1", "agc_gain_db"), ("u-blox_01", "u-blox 01", "L1", "agc_cnt"), ("u-blox_01", "u-blox 01 noise", "L1", "noise_per_ms")):  # the L1 indicators
        g = st[(st.receiver == rx) & (st.band == band) & st[fld].notna()]  # that indicator's readings
        if rx == "Septentrio_03":  # if: the mosaic-H
            g = g[g.antenna_id.fillna(0) == 0]  # main antenna only
        marg = g[((g.t >= t_("15:45:12")) & (g.t < t_("15:50:00"))) | (g.t >= t_("16:05:00"))][fld].median()  # margin median
        first5 = g[(g.t >= t_("15:50:00")) & (g.t < t_("15:55:00"))][fld]  # the first five minutes
        only = g[(g.t >= t_("15:55:00")) & (g.t < t_("16:05:00"))][fld]  # spoofing alone
        put(f"L1 2.6.12 {name}", "margin median", round(float(marg), 1))  # the quiet level
        put(f"L1 2.6.12 {name}", "first five minutes median", round(float(first5.median()), 1))  # jamming plus spoofing
        put(f"L1 2.6.12 {name}", "spoofing alone median", round(float(only.median()), 1))  # after 15:55
        put(f"L1 2.6.12 {name}", "spoofing alone, change against the margin", round(float(only.median() - marg), 1))  # the deflection
        put(f"L1 2.6.12 {name}", "spoofing alone, largest change against the margin", round(float((only - marg).abs().max()), 1))  # the deepest point
        put(f"L1 2.6.12 {name}", "first five minutes, largest change against the margin", round(float((first5 - marg).abs().max()), 1))  # the expected 46 dB


def g30_onset() -> dict:  # function: when does the G30 error of u-blox 01 first appear? Differences against the Septentrio receivers on the same antenna, through 2.6.5 and across the gap into 2.6.2
    print("1b. u-blox 01 G30 onset, 2.6.5 and the start of 2.6.2")  # heading
    out = {}  # numbers handed back to main()
    series = []  # per-second rows kept for the CSV
    for test, t0, t1 in (("2.6.5", "12:27:00", "12:54:30"), ("2.6.2", "12:56:58", "12:58:30")):  # for loop over the two sessions on either side of the gap
        folder = GT.loc[test, "session_folder"]  # the session folder
        def gps(rx):  # nested function: the GPS pseudoranges of one receiver in the window, first epoch of each second
            spans = [(pd.Timestamp(T(t0), unit="s", tz="UTC"), pd.Timestamp(T(t1), unit="s", tz="UTC"))]  # the window as one span
            o = M.read_obs(PROJECT / "Data" / "rinex" / folder / f"{folder.split('_T_')[0]}_{rx}.crx.gz", spans, M.OBS_CODES[rx[:3]])  # pseudoranges
            return o[o.sv.str[0] == "G"]  # GPS only
        ub, hh = gps("UBX_01"), gps("SBF_03")  # u-blox 01 and the mosaic-H (it logs to the end of 2.6.5)
        m = ub.merge(hh, on=["sec", "sv"], suffixes=("_u", "_h"))  # satellites seen by both, per second
        m["d"] = m.pr_u - m.pr_h  # the difference of the two receivers' pseudoranges: the common clock offset plus any satellite-specific error
        base = m[m.sv != "G30"].groupby("sec").d.median().rename("base")  # the clock offset between the receivers from the other satellites
        g = m[m.sv == "G30"].merge(base, on="sec")  # G30 with that baseline attached
        g["err"] = g.d - g.base  # the G30-specific error of the u-blox against the mosaic-H, metres
        g["test"] = test  # session label
        g["hms"] = g.sec.map(hms)  # clock text
        series.append(g[["test", "hms", "err"]])  # kept
        if test == "2.6.5":  # if: the session that holds the ramp
            late = g[g.sec >= T("12:53:09")]  # from the first second after the disturbance
            first_dev = g[(g.err.abs() > 20) & (g.sec >= T("12:45:00"))].iloc[0]  # the first second with an error beyond 20 m
            put("G30 u-blox onset 2.6.5", "first second with a G30 error beyond 20 m (against the mosaic-H)", hms(first_dev.sec))  # onset of the ramp
            put("G30 u-blox onset 2.6.5", "error at the first second beyond 20 m (m)", round(float(first_dev.err), 1))  # its size
            ramp = g[(g.sec >= T("12:48:50")) & (g.sec <= T("12:51:00"))]  # the ramp before the spoofer's passage
            put("G30 u-blox onset 2.6.5", "error growth over 12:48:50-12:51:00 (m per s, linear fit)", round(float(np.polyfit(ramp.sec - ramp.sec.min(), ramp.err, 1)[0]), 2))  # the slope
            put("G30 u-blox onset 2.6.5", "error at 12:51:00 (m)", round(float(g[g.sec == T("12:51:00")].err.iloc[0]), 1))  # where the ramp ends in the data
            put("G30 u-blox onset 2.6.5", "other satellites, median |difference from the clock offset| 12:48:50-12:51:00 (m)", round(float(m[(m.sv != "G30") & (m.sec >= T("12:48:50")) & (m.sec <= T("12:51:00"))].assign(r=lambda q: (q.d - q.sec.map(base)).abs()).r.median()), 2))  # the others stay put
            put("G30 u-blox onset 2.6.5", "G30 error 12:53:09-12:54:24, median and max |error| (m)", f"{late.err.median():.1f} {late.err.abs().max():.1f}")  # normal again
            put("G30 u-blox onset 2.6.5", "last second logged in 2.6.5", hms(g.sec.max()))  # end of the data before the gap
            out["last265"] = hms(g.sec.max())  # kept
        else:  # else: the start of 2.6.2
            put("G30 u-blox onset 2.6.2", "first second logged and the error there (m)", f"{hms(g.sec.min())} {g.err.iloc[0]:.1f}")  # already in place
            out["first262"] = hms(g.sec.min())  # kept
    pd.concat(series).round(3).to_csv(CHECK_DIR / "g30_onset.csv", index=False)  # the series
    return out  # the two instants


def mask20() -> None:  # function: check 1b (same antenna, identical satellites) at the 10 degree mask and again at 20 degrees, diagnostic only
    print("3b. same-antenna differences at a 20 degree mask (diagnostic)")  # heading
    iv = pd.read_csv(PROC / "preprocessing" / "reference_clean_intervals.csv")  # the kept intervals
    iv = iv[iv.minutes > 0]  # only those with time
    for test in ("2.6.12", "2.6.2"):  # for loop over the sessions with two or more receivers and both margins
        sched = pd.Timestamp(GT.loc[test, "scheduled_start_utc"], tz="UTC").timestamp()  # the scheduled start separates the two margins
        folder = GT.loc[test, "session_folder"]  # the session folder
        obs = {}  # receiver -> observations
        for rx, g in iv[iv.test_code == test].groupby("receiver"):  # for loop over the receivers with kept time
            spans = [(pd.Timestamp(a, tz="UTC"), pd.Timestamp(b, tz="UTC")) for a, b in zip(g.start_utc, g.end_utc)]  # its kept intervals
            stem = M.OBS_FILES.get(test, {}).get(rx, rx)  # the stream's file part
            obs[rx] = M.read_obs(PROJECT / "Data" / "rinex" / folder / f"{folder.split('_T_')[0]}_{stem}.crx.gz", spans, M.OBS_CODES[rx[:3]])  # observations
        site = iv[iv.test_code == test].site.iloc[0]  # the site name
        xs = site_x(site)  # its estimate as a starting point
        lat, lon, _ = M.ecef_to_llh(*xs)  # geodetic position for the axes
        axes = M.enu_axes(lat, lon)  # rotation to east, north, up
        for a_, b_ in (("SBF_02", "SBF_03"), ("SBF_02", "UBX_01"), ("SBF_03", "UBX_01")):  # for loop over the three receiver pairs
            if a_ not in obs or b_ not in obs:  # if: a receiver has no kept time here
                continue  # skip
            gA, gB = dict(tuple(obs[a_].groupby("sec"))), dict(tuple(obs[b_].groupby("sec")))  # observations per second
            for mask in (10.0, 20.0):  # for loop over the two masks
                M.ELEV_MASK_DEG = mask  # the module-level mask that solve_epoch reads
                rows = []  # (second, horizontal, vertical)
                for sec in sorted(set(gA) & set(gB)):  # for loop over the seconds both receivers logged
                    sats = sorted(set(gA[sec].sv) & set(gB[sec].sv))  # the satellites both observed
                    xa, xb = M.solve_subset(ORB, gA[sec], sats, xs, KL), M.solve_subset(ORB, gB[sec], sats, xs, KL)  # each receiver from the same set
                    if xa is None or xb is None:  # if: either failed (too few above the mask)
                        continue  # skip
                    d = (xa - xb) @ axes.T  # east, north, up difference
                    rows.append((sec, float(np.hypot(d[0], d[1])), float(d[2])))  # one row
                df = pd.DataFrame(rows, columns=["sec", "hz", "vt"])  # the differences
                for per, q in (("pre-test", df[df.sec < sched]), ("post-test", df[df.sec >= sched])):  # for loop over the two margins
                    if len(q) == 0:  # if: nothing in this margin
                        continue  # skip
                    lab = f"{test} {per} {M.RECEIVERS[a_]} minus {M.RECEIVERS[b_]}"  # group label
                    put("mask check " + f"{mask:.0f} deg", lab, f"n={len(q)} horizontal median {q.hz.median():.2f} m, vertical median {q.vt.median():+.2f} m, vertical |median| {q.vt.abs().median():.2f} m, vertical p95 {q.vt.abs().quantile(0.95):.2f} m")  # recorded
    M.ELEV_MASK_DEG = 10.0  # restored


def main() -> None:  # function: the diagnostics, then the check of the position script's constants
    a = g30()  # G30
    g30_onset()  # when the G30 error first appears
    mask20()  # the 20 degree mask check
    b = g08()  # G08
    stave_stop_spread()  # the residual spread at the Stave stop
    l1_spoof_only()  # the L1 deflection after 15:55:00
    # list of two dicts: the constants G30_NORMAL and G08_LAST of the position script against the instants this script found in the data
    chk = [{"check": "the position script G30_NORMAL equals the reacquisition second found here", "expected": f"{M.G30_NORMAL:%H:%M:%S}", "found": a["normal"], "ok": f"{M.G30_NORMAL:%H:%M:%S}" == a["normal"]},
           {"check": "the position script G08_LAST equals the last anomalous second found here", "expected": f"{M.G08_LAST:%H:%M:%S}", "found": b["last"], "ok": f"{M.G08_LAST:%H:%M:%S}" == b["last"]},  # the two constants
           # the first G08 second, which the evidence script, the observables script and 06 read from ground_truth/decision_times.csv, re-derived here from the observations
           {"check": "ground_truth/decision_times.csv G08_first equals the first anomalous second found here", "expected": M.DT["G08_first"], "found": b["first"], "ok": M.DT["G08_first"] == b["first"]}]
    pd.DataFrame(ROWS).to_csv(CHECK_DIR / "g30_diagnostics.csv", index=False)  # every number
    pd.DataFrame(chk).to_csv(CHECK_DIR / "g30_diagnostics_checks.csv", index=False)  # the constants checked
    for c in chk:  # for loop over the three
        print(f"  [{'ok' if c['ok'] else 'FAIL'}] {c['check']}: expected {c['expected']}, found {c['found']}")  # printed
    if not all(c["ok"] for c in chk):  # if: a constant disagrees with the data
        sys.exit("STOP: a constant of the position script does not match the diagnostic")  # halt


if __name__ == "__main__":  # if: true only when the file is run directly
    main()  # runs both
