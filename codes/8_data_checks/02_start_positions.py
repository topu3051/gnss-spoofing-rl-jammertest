#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
Data check 3 (the supervisor's checks of 19 October 2026, item d), with the main location after the drives: the position of each receiver from the clean seconds at the
START of each recording, before the first interference, against the thesis' per-site reference positions (Table 4).

The positions are the position script's own single-point solutions, its functions loaded unchanged: GPS C1C and Galileo E1 pseudoranges, IGS broadcast
orbits, a 10 degree mask, Klobuchar and Saastamoinen, one clock per constellation, sin^2(elevation) weights and iterative rejection
of residuals over 30 m; the 1 Hz solutions' median per window and receiver. Windows:
  2.6.5   the Stave stop before the scheduled start (12:27:56-12:30:00) and with 30 s guards (12:28:26-12:29:30), against Stave;
          after the drives: the main location, from 12:35:27 (the last drive's 30 s guard after its end 12:34:57) to
          12:36:00, the second before the first interference evidence of 2.6.5 (the observed jamming phase from 12:36:01,
          labels/labels_evidence_final.csv and ground_truth/jam_phase_boundaries.csv), against the 2.6.2 site (the position script's SITES puts 2.6.5 there);
          also with a 30 s guard before that onset (12:35:27-12:35:30)
  2.6.2   capture start to the scheduled start (12:56:58-13:00:00) and the thesis' pre-test interval (to 12:59:30), against the 2.6.2 site
  2.6.4   capture start to the scheduled start (14:47:25-14:50:00) and the thesis' pre-test interval (to 14:49:30), south-west site
  2.6.10  starts inside its scheduled window: informational only, 14:30:43 to 30 s before the first evidence (14:34:51)
  2.6.11  starts inside its window and its first evidence is at 15:23:02: informational only, the quiet stretch 15:24:14-15:28:34
  2.6.12  sealed: the logged 1 Hz solutions of the thesis' pre-test interval (preprocessing/reference_position_epochs.csv) only
The supervisor's criterion is met only by the windows before the first scheduled interference that pass every indicator (2.6.5's
Stave stop, 2.6.2, 2.6.4, 2.6.12); the main location's window lies inside 2.6.5's scheduled window but before any evidence, and is flagged so.
A difference over 2 m is reported; no residual is recomputed here (report only).

Reads   Data/rinex/<session>/*.crx.gz (through the position script), Data/external IGS broadcast file (through 04), preprocessing/reference_positions.csv,
        preprocessing/reference_position_epochs.csv, ground_truth/session_ground_truth.csv, ground_truth/session_distance_to_sender.csv, labels/labels_evidence_final.csv,
        ground_truth/margin_clean_minutes.csv, features/features_base_1hz.csv.gz (known-antenna residual columns, to count them in the Stave stop)
Writes  Data/processed/data_checks/start_positions.csv, start_positions_epochs.csv.gz, start_positions_indicators.csv, start_positions_checks.csv
"""

from __future__ import annotations  # future import: type hints kept as text (the project's convention)

import datetime as dtm  # standard-library dates: the recording day for the orbit records
import gzip  # standard-library gzip: the epoch table written with a fixed header (byte-stable)
import importlib.util  # standard-library module loader: the position script's and the conversion script's names start with a digit
import os  # operating-system access: MPLCONFIGDIR is set before the position script loads matplotlib
import sys  # standard-library system access: sys.path and sys.modules
from pathlib import Path  # Path objects for building file paths

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
os.environ["MPLCONFIGDIR"] = str(PROJECT / "codes" / ".mplconfig")  # matplotlib's cache inside the project folder

import numpy as np  # NumPy: arrays
import pandas as pd  # pandas: tables

sys.path.insert(0, str(PROJECT / "codes" / "common"))  # codes/ first on the import path, so the helpers module is found
from helpers import exit_if_failed  # status 1 on a failed check

PROC = PROJECT / "Data" / "processed"  # the processed-data folder
OUT = PROC / "data_checks"  # this check's output folder
DATE = "2026-09-14"  # the recording date
LIMIT_M = 2.0  # float constant: the 2 m rule
RXS = ["SBF_02", "SBF_03", "UBX_01"]  # list: mosaic-X5, mosaic-H main antenna, u-blox 01
CHECKS: list = []  # list: one dict per self-check
FAILED: list = []  # list: names of the checks that failed
# list of (session, window name, start, end, reference site, meets the supervisor's criterion): the windows of the module docstring
WINDOWS = [("2.6.5", "Stave stop before the scheduled start, no guard", "12:27:56", "12:30:00", "Stave", True),
           ("2.6.5", "Stave stop before the scheduled start, 30 s guards", "12:28:26", "12:29:30", "Stave", True),  # continued: more items of the list or the row
           ("2.6.5", "main location after the drives, to the first interference (item 2)", "12:35:27", "12:36:00", "2.6.2 site", False),  # continued: more items of the list or the row
           ("2.6.5", "main location after the drives, 30 s guard before the first interference (item 2)", "12:35:27", "12:35:30", "2.6.2 site", False),  # continued: more items of the list or the row
           ("2.6.2", "capture start to the scheduled start, no guard", "12:56:58", "13:00:00", "2.6.2 site", True),  # continued: more items of the list or the row
           ("2.6.2", "thesis pre-test interval (30 s end guard)", "12:56:58", "12:59:30", "2.6.2 site", True),  # continued: more items of the list or the row
           ("2.6.10", "INFORMATIONAL: in-window start, to 30 s before the first evidence (14:34:51)", "14:30:43", "14:34:21", "south-west site", False),  # continued: more items of the list or the row
           ("2.6.4", "capture start to the scheduled start, no guard", "14:47:25", "14:50:00", "south-west site", True),  # continued: more items of the list or the row
           ("2.6.4", "thesis pre-test interval (30 s end guard)", "14:47:25", "14:49:30", "south-west site", True),  # continued: more items of the list or the row
           ("2.6.11", "INFORMATIONAL: in-window quiet stretch between pooled intervals, 30 s guards", "15:24:14", "15:28:34", "south-west site", False)]  # continued: more items of the list or the row


def check(name: str, ok, detail: str) -> None:  # function: records one self-check and prints it (ok None marks a reported number)
    CHECKS.append({"check": name, "result": "REPORT" if ok is None else ("PASS" if ok else "FAIL"), "detail": detail})  # nested conditional expression picks the word
    if ok is not None and not bool(ok):  # if: a pass/fail check that did not pass
        FAILED.append(name)  # kept for the exit status
    print(f"  [{CHECKS[-1]['result']}] {name}: {detail}")  # printed for the run log


def load(name: str, filename: str):  # function: loads codes/<filename> as a module (its main() stays behind its guard)
    spec = importlib.util.spec_from_file_location(name, PROJECT / "codes" / filename)  # where from
    mod = importlib.util.module_from_spec(spec)  # an empty module object
    sys.modules[name] = mod  # registered so its imports resolve
    spec.loader.exec_module(mod)  # its top level (definitions; nothing is written)
    return mod  # the module


C5 = load("reference_positions_for_start_positions", "2_preprocessing/03_reference_positions.py")  # 05c: reader, orbits, solver and geodesy, unchanged
S04 = load("convert_rinex_for_start_positions", "2_preprocessing/01_convert_rinex.py")  # the conversion script: the IGS broadcast file and its record reader


def U(hms: str) -> pd.Timestamp:  # function: "HH:MM:SS" on the recording date -> UTC timestamp
    return pd.Timestamp(f"{DATE} {hms}", tz="UTC")  # the date and the time, tagged UTC


def enu(x: np.ndarray, ref: np.ndarray) -> np.ndarray:  # function: east, north, up offset of x from ref, in metres, at ref
    lat, lon, _ = C5.ecef_to_llh(*ref)  # the reference's geodetic position, for the local axes
    return (np.asarray(x) - np.asarray(ref)) @ C5.enu_axes(lat, lon).T  # the ECEF difference rotated into east, north, up


def row(t: str, name: str, a: str, b: str, site: str, ok: bool, rx: str, ep: pd.DataFrame, n_read: int, ref: pd.DataFrame, sxr: pd.DataFrame) -> dict:  # function: one window's result for one receiver
    base = {"test_code": t, "window": name, "start_utc": a, "end_utc": b, "meets_criterion": ok, "receiver": rx, "reference_site": site, "seconds_read": n_read, "seconds_solved": len(ep)}  # dict literal: identity
    if not len(ep):  # if: no solution in the window
        return base  # identity only
    xm = np.array([ep.x.median(), ep.y.median(), ep.z.median()])  # the median ECEF position of the window
    r_ = ref.loc[site]  # Table 4's site row
    d = enu(xm, np.array([r_.x_m, r_.y_m, r_.z_m]))  # offset from the site estimate
    lat, lon, h = C5.ecef_to_llh(*xm)  # geodetic form
    st = C5.stats_row(ep, xm)  # the position script's scatter: 95th percentiles horizontal and vertical
    own = sxr[(sxr.test_code == t) & (sxr.receiver == rx)]  # the thesis's own session x receiver median, if any
    do = enu(xm, own[["x_m", "y_m", "z_m"]].to_numpy()[0]) if len(own) else np.full(3, np.nan)  # conditional expression: offset from it
    rej = sorted(set(";".join(ep.rejected_sv.fillna("").astype(str)).split(";")) - {""})  # the satellites rejected in any epoch
    # dict literal: the window's median, scatter, rejections and the offsets from Table 4 (site) and from the thesis's own receiver median
    return {**base, "lat_deg": round(lat, 9), "lon_deg": round(lon, 9), "h_m": round(h, 3), "p95_h_m": round(st["h_p95_m"], 2), "p95_v_m": round(st["v_p95_m"], 2),
            "obs_rejected": int(ep.n_rejected.sum()), "rejected_sv": ";".join(rej), "dE_m": round(d[0], 2), "dN_m": round(d[1], 2), "dH_m": round(float(np.hypot(d[0], d[1])), 2),  # continued: more items of the list or the row
            "dU_m": round(d[2], 2), "over_2m": bool(np.hypot(d[0], d[1]) > LIMIT_M or abs(d[2]) > LIMIT_M), "vs_thesis_session_rx_dH_m": round(float(np.hypot(do[0], do[1])), 2), "vs_thesis_session_rx_dU_m": round(float(do[2]), 2)}  # the row


def main() -> None:  # function: check 3 and the main location after the drives, from start to finish
    ref_all = pd.read_csv(PROC / "preprocessing" / "reference_positions.csv")  # Table 4's source
    ref = ref_all[ref_all.scope == "site"].set_index("site")  # the three site estimates
    sxr = ref_all[ref_all.scope == "session x receiver"]  # the session x receiver medians
    gt = pd.read_csv(PROC / "ground_truth" / "session_ground_truth.csv", dtype={"test_code": str}).set_index("test_code")  # session folders
    dist = pd.read_csv(PROC / "ground_truth" / "session_distance_to_sender.csv", dtype={"test_code": str}).dropna(subset=["median_lat"]).drop_duplicates("test_code", keep="last").set_index("test_code")  # starting points for the iteration only (the position script's choice)
    ev = pd.read_csv(PROC / "labels" / "labels_evidence_final.csv", dtype={"test_code": str})  # the final evidence intervals
    first = {t: g[g.source != "pooled_interval"].sort_values("start").iloc[0] for t, g in ev.groupby("test_code")}  # dict comprehension: each session's first evidence interval
    f5 = first["2.6.5"]  # 2.6.5's
    check("item 2 bound: 2.6.5's first interference evidence after parking", f5.start_utc == "12:36:01" and f5.source == "observed_jamming_phase", f"{f5.start_utc} ({f5.source}, labels/labels_evidence_final.csv)")  # the window's end + 1 s
    check("2.6.10's first evidence (informational window ends 30 s before it)", first["2.6.10"].start_utc == "14:34:51", f"{first['2.6.10'].start_utc} {first['2.6.10'].source} {first['2.6.10'].receiver}")  # bound of the 2.6.10 window
    check("2.6.11's first evidence: one second after capture start, so no start window exists", first["2.6.11"].start_utc == "15:23:02", f"{first['2.6.11'].start_utc} {first['2.6.11'].source} {first['2.6.11'].receiver}")  # nothing available
    orb = C5.Orbits(S04, dtm.date(2026, 9, 14))  # orbits and clocks from the IGS file only
    kl = C5.igs_klobuchar(S04.IGS_BRDC)  # the Klobuchar parameters from its header
    rows, eps = [], []  # result rows; epoch tables
    for t, name, a, b, site, ok in WINDOWS:  # for loop over the windows
        for rx in RXS:  # for loop over the receivers
            obs = C5.read_obs(C5.obs_path(gt.loc[t, "session_folder"], t, rx), [(U(a), U(b))], C5.OBS_CODES[rx[:3]])  # the position script's reader: the 1 Hz pseudoranges in the window
            x0 = C5.llh_to_ecef(float(dist.loc[t, "median_lat"]), float(dist.loc[t, "median_lon"]), 0.0)  # a starting point only
            ep, _ = C5.solve_session(orb, obs, x0, kl) if len(obs) else (pd.DataFrame(), None)  # conditional expression: the position script's solver, or nothing
            rows.append(row(t, name, a, b, site, ok, rx, ep, int(obs.sec.nunique()) if len(obs) else 0, ref, sxr))  # the result
            if len(ep):  # if: solutions exist
                eps.append(ep.assign(test_code=t, receiver=rx, window=name))  # kept for the epoch table
    re_ = pd.read_csv(PROC / "preprocessing" / "reference_position_epochs.csv", dtype={"test_code": str})  # the logged 1 Hz solutions (2.6.12 is sealed: no new solution)
    for rx in RXS:  # for loop over the receivers
        g = re_[(re_.test_code == "2.6.12") & (re_.receiver == rx) & (re_.sec < U("15:50:00").timestamp())]  # the logged pre-test solutions
        name = "LOGGED solutions of the thesis's pre-test interval (sealed; 15:42:29-15:45:42 not on file)"  # the window's name
        rows.append(row("2.6.12", name, pd.Timestamp(int(g.sec.min()), unit="s").strftime("%H:%M:%S"), pd.Timestamp(int(g.sec.max()), unit="s").strftime("%H:%M:%S"), "Stave", True, rx, g, len(g), ref, sxr))  # the result
    T = pd.DataFrame(rows)  # the table
    OUT.mkdir(parents=True, exist_ok=True)  # the folder
    T.to_csv(OUT / "start_positions.csv", index=False, lineterminator="\n")  # written
    E = pd.concat(eps, ignore_index=True)  # every new 1 Hz solution
    with open(OUT / "start_positions_epochs.csv.gz", "wb") as fh, gzip.GzipFile(filename="", mode="wb", fileobj=fh, mtime=0) as gz:  # with-block over two context managers: no name or time in the gzip header
        gz.write(E.round(6).to_csv(index=False, lineterminator="\n").encode("utf-8"))  # six decimals (a micrometre in ECEF), text encoded to bytes
    sel = T[(T.test_code == "2.6.4") & (T.window == "thesis pre-test interval (30 s end guard)")]  # 2.6.4's thesis interval: the thesis's own median used exactly these seconds
    check("the position script's solver reproduces Table 4's 2.6.4 medians on the thesis interval (offset from them, m)", bool((sel.vs_thesis_session_rx_dH_m.abs() <= 0.005).all() and (sel.vs_thesis_session_rx_dU_m.abs() <= 0.005).all()), "; ".join(f"{r.receiver} {r.vs_thesis_session_rx_dH_m:.2f}/{r.vs_thesis_session_rx_dU_m:.2f}" for r in sel.itertuples()))  # reproduction
    crit = T[T.meets_criterion & T.dH_m.notna()]  # the windows that meet the supervisor's criterion
    over = crit[crit.over_2m]  # those over 2 m
    check("criterion windows over 2 m (reported; report only, no recompute)", None, "; ".join(f"{r.test_code} {r.receiver} dH {r.dH_m} dU {r.dU_m} ({r.window})" for r in over.itertuples()) or "none")  # the 2 m rule
    rest = crit[crit.test_code != "2.6.5"]  # the criterion windows at the reference spots
    check("criterion windows at the reference spots: largest horizontal and vertical offsets (m)", None, f"dH {rest.dH_m.max():.2f}, |dU| {rest.dU_m.abs().max():.2f}")  # reported
    i2 = T[T.window.str.contains("item 2")]  # the rows of the main location after the drives
    check("item 2 (2.6.5 main location after the drives) against the 2.6.2 site (report only)", None, "; ".join(f"{r.window[-25:]} {r.receiver}: n {r.seconds_solved}, dH {r.dH_m}, dU {r.dU_m}, over 2 m {r.over_2m}" for r in i2.itertuples()))  # reported
    fb = pd.read_csv(PROC / "features" / "features_base_1hz.csv.gz", usecols=["test_code", "receiver", "sec", "known_G_res_median", "known_E_res_median"], dtype={"test_code": str})  # two known-antenna residual columns
    st = fb[(fb.test_code == "2.6.5") & fb.sec.between(int(U("12:27:56").timestamp()), int(U("12:32:36").timestamp()))]  # the Stave stop's rows
    check("2.6.5 Stave stop: known-antenna residuals are missing in every row (nothing to recompute)", bool(st[["known_G_res_median", "known_E_res_median"]].isna().all().all()), f"{len(st)} receiver-seconds")  # the residual question
    mc = pd.read_csv(PROC / "ground_truth" / "margin_clean_minutes.csv")  # the receiver-status script's per-indicator margin minutes
    pre = mc[mc.margin.isin(["12:27:21-12:30:00", "12:56:58-13:00:00", "14:47:25-14:50:00", "15:42:29-15:45:12", "15:45:12-15:50:00"])].copy()  # the pre-test margins
    pre["covered_s"] = (pre.covered_min * 60).round().astype(int)  # seconds each indicator covers
    pre["contaminated_s"] = (pre.contaminated_min * 60).round().astype(int)  # seconds it calls contaminated
    pre[["test_code", "margin", "indicator", "covered_s", "contaminated_s", "possibly_contaminated_margin"]].to_csv(OUT / "start_positions_indicators.csv", index=False, lineterminator="\n")  # written
    pd.DataFrame(CHECKS).to_csv(OUT / "start_positions_checks.csv", index=False, lineterminator="\n")  # the checks
    print(T.drop(columns=["lat_deg", "lon_deg"]).to_string(index=False))  # the table for the run log


if __name__ == "__main__":  # if: true only when the file is run directly
    main()  # runs check 3 and the main location after the drives
    exit_if_failed(FAILED, "8_data_checks/02_start_positions.py")  # the helpers module: status 1 when any check failed
