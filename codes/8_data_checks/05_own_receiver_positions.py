#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
Each receiver at its own reference position. The labels, the features and the RQ4 position monitor use one reference position per
site; the receivers shared the site but each had its own antenna. Here every receiver gets its own position from its own clean first
minutes (Data/processed/data_checks/start_positions.csv: the windows that meet the criterion, with the 30 s guards); a session without a
clean start takes the same receiver's position at the same site from another session (2.6.10 and 2.6.11 from 2.6.4; 2.6.5 at the
south site from 2.6.2). Every product that uses the reference position is then recomputed with the project's own functions, run in
this process with the site lookup replaced for the receiver being computed, into Data/processed/robustness_checks/mirror/ (nothing
published is written):
  the known- and self-position residuals (the residual script), the base table's residual columns and the 10 s windows of every part
  as the feature script builds them on the final labels, the quiet-negative screen and the training labels, the labels' own-position
  evidence (the label script), the RQ4 position monitor (the RQ4 script; 2.6.12's own fixes are read through the transfer script's
  freeze guard), and the 2.6.10 spoofed-tracking check.
Gate first: the same run with today's site positions must reproduce the published files byte for byte (or row for row where the
product is a set of intervals); only then is the per-receiver run compared. Run it after the feature chain (4_features), whose window
files the gate compares against.
Usage   python3 codes/8_data_checks/05_own_receiver_positions.py
Writes  Data/processed/robustness_checks/reference_positions_per_receiver.csv, positions_checks.csv, positions_compare_*.csv (one per
        product), positions_summary.csv; the recomputed products under Data/processed/robustness_checks/mirror/
"""
# import: importlib loads the project scripts as modules
import importlib.util
# import: gzip reads the committed window files as text
import gzip
# import: sys registers modules
import sys
# import: time for the run time
import time
# import: Path for explicit paths
from pathlib import Path
# import: numpy for arrays
import numpy as np
# import: pandas for the tables
import pandas as pd

# project root (this file is codes/8_data_checks/05_own_receiver_positions.py)
PROJECT = Path(__file__).resolve().parents[2]
# processed-data folder and the position check's folder
PROC, ROBUST_DIR = PROJECT / "Data" / "processed", PROJECT / "Data" / "processed" / "robustness_checks"
# the scratch root of this item
MIRROR = ROBUST_DIR / "mirror"
# list of check rows
CHECKS = []
# dict: the receiver whose quantities are being computed (set by the wrappers below)
CUR = {"rx": None}
# dict: (site, receiver) -> ECEF position used in the per-receiver run (filled by positions())
POS = {}
# dict: site name -> today's ECEF site estimate (filled by positions())
SITE = {}


# function: records and prints one check
def check(name: str, ok: bool, detail: str = "") -> None:
    # appended
    CHECKS.append({"check": name, "pass": bool(ok), "detail": detail})
    # printed
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


# function: loads a project script as a module under a name (its main() stays behind its guard)
def load(name: str, rel: str):
    # where the file is
    spec = importlib.util.spec_from_file_location(name, PROJECT / rel)
    # an empty module object
    mod = importlib.util.module_from_spec(spec)
    # registered so its own imports resolve
    sys.modules[name] = mod
    # its top level
    spec.loader.exec_module(mod)
    # the module
    return mod


# the residual script (the residual engine, it loads the evidence script as D, the position script as M and 04 as S04)
A7 = load("residuals_for_own_positions", "codes/4_features/02_residuals.py")
# 07 (the base table and the windows, it loads its own the residual script as A)
F7 = load("window_features_for_own_positions", "codes/4_features/03_window_features.py")
# 06 (the labels, it loads the evidence script as)
G6 = load("labels_for_own_positions", "codes/3_labels/01_labels.py")
# the training-label script (the training labels)
TL = load("training_labels_for_own_positions", "codes/3_labels/03_training_labels.py")
# 14 (RQ4, it loads the evidence script as D and 13 as X)
RQ4 = load("delay_monitors_cost_for_own_positions", "codes/7_evaluation/02_delay_monitors_cost.py")
# the observables script (the observables, for the 2.6.10 check)
O5 = load("observables_for_own_positions", "codes/4_features/01_observables.py")
# the spoofed-tracking script (the spoofed-tracking rule, for its constants)
S5 = load("spoofed_tracking_end_for_own_positions", "codes/1_ground_truth/09_spoofed_tracking_end_2610.py")
# dict: each script's own copy of the evidence script -> how its site is replaced. the residual script and the observables script look the site up inside their receiver loop (this script
# sets the receiver), the RQ4 script reads the receiver's fixes just before it looks the site up, 06 looks the site up before its receiver loop and
# measures the distance inside it, so the label script's distance function is the one replaced
LOOKUPS = {"07a": (A7.D, "site"), "05e": (O5.D, "site"), "14": (RQ4.D, "site+fixes"), "06": (G6.D05, "distance+fixes")}


# function: the per-receiver positions from the start-position check, and today's site estimates
def positions() -> pd.DataFrame:
    # the start-position table of the data checks
    sp = pd.read_csv(PROC / "data_checks" / "start_positions.csv", dtype={"test_code": str})
    # dict: the window of each session used (the 30 s guarded windows that meet the criterion; 2.6.12's logged pre-test solutions)
    win = {"2.6.2": "thesis pre-test interval (30 s end guard)", "2.6.4": "thesis pre-test interval (30 s end guard)",
           "2.6.12": "LOGGED solutions of the thesis's pre-test interval (sealed; 15:42:29-15:45:42 not on file)", "2.6.5": "Stave stop before the scheduled start, 30 s guards"}  # dict literal continued: 2.6.12's logged solutions and 2.6.5's Stave stop
    # dict: which session's position each site takes (2.6.5 at the south site from 2.6.2, 2.6.10 and 2.6.11 from 2.6.4)
    site_src = {"Stave": "2.6.12", "2.6.2 site": "2.6.2", "south-west site": "2.6.4"}
    # today's site estimates (the position script's rows of scope "site")
    ref = pd.read_csv(PROC / "preprocessing" / "reference_positions.csv")
    # for loop over the site rows
    for r in ref[ref.scope == "site"].itertuples():
        # ECEF array
        SITE[r.site] = np.array([r.x_m, r.y_m, r.z_m])
    # list of output rows
    rows = []
    # for loop over the sessions with a clean start
    for t, w in win.items():
        # for loop over the three receivers
        for rx in ("SBF_02", "SBF_03", "UBX_01"):
            # the window's row of this receiver
            r = sp[(sp.test_code == t) & (sp.window == w) & (sp.receiver == rx)].iloc[0]
            # geodetic to ECEF with the position script's own conversion
            x = np.array(A7.M.llh_to_ecef(float(r.lat_deg), float(r.lon_deg), float(r.h_m)))
            # the site this position stands for (the 2.6.5 Stave-stop spot is 8 m from the Stave estimate and stands for no site)
            site = next((s for s, src in site_src.items() if src == t), None)
            # if: a site's per-receiver position
            if site is not None:
                # kept for the per-receiver run
                POS[(site, rx)] = x
            # one row: where it comes from and its offset from today's site estimate
            rows.append({"session": t, "window": w, "receiver": rx, "used_for_site": site or "none (the 2.6.5 Stave stop, 8 m from the Stave estimate; no product uses it)",
                         "used_for_sessions": {"Stave": "2.6.12", "2.6.2 site": "2.6.2, 2.6.5 (after its drives)", "south-west site": "2.6.4, 2.6.10, 2.6.11"}.get(site, ""),  # continued: the sessions each site's position serves
                         "seconds": int(r.seconds_solved), "lat_deg": r.lat_deg, "lon_deg": r.lon_deg, "h_m": r.h_m, "x_m": x[0], "y_m": x[1], "z_m": x[2],  # continued: the seconds solved, the geodetic and ECEF position
                         "reference_site": r.reference_site, "dE_m": r.dE_m, "dN_m": r.dN_m, "dH_m": r.dH_m, "dU_m": r.dU_m})  # continued: the offset from today's site estimate (the start-position check's columns)
    # DataFrame, written
    out = pd.DataFrame(rows)
    out.to_csv(ROBUST_DIR / "reference_positions_per_receiver.csv", index=False, lineterminator="\n")  # written with fixed line endings
    # 2.6.5's spot after the drives against the south-site estimate and against each receiver's 2.6.2 position : report the offset)
    s5 = sp[(sp.test_code == "2.6.5") & sp.window.str.startswith("main location after the drives, to")]
    # for loop over its three rows
    for r in s5.itertuples():
        # the spot's ECEF position
        x = np.array(A7.M.llh_to_ecef(float(r.lat_deg), float(r.lon_deg), float(r.h_m)))
        # horizontal offset from the receiver's 2.6.2 position (the evidence script's ENU distance)
        d = A7.D.enu_dist(x, POS[("2.6.2 site", r.receiver)])[0]
        # reported
        check(f"2.6.5's spot after the drives ({r.receiver}): offset from the south-site estimate {r.dH_m:.2f} m, from its own 2.6.2 position {d:.2f} m", True, " (reported)")
    # printed
    print(out[["session", "receiver", "used_for_site", "dH_m", "dU_m"]].to_string(index=False))
    # the table
    return out


# function: the site lookup of the per-receiver run: the position of the receiver being computed at the named site
def site_rx(site: str) -> np.ndarray:
    # the per-receiver position (a KeyError would mean a receiver or site without a position: it must not happen)
    return POS[(site, CUR["rx"])]


# function: replaces the site in every the evidence script copy for the receiver being computed (mode 'per_receiver'), or restores today's (mode 'gate')
def patch(mode: str) -> None:
    # for loop over the evidence script copies and how each is replaced
    for D, how in LOOKUPS.values():
        # the originals kept once on the module object
        if not hasattr(D, "_site_lookup_original"):
            # tuple: today's site lookup, own-position reader and distance function
            D._site_lookup_original = (D.site_xyz, D.own_positions, D.horiz_from_site)
        # unpacked
        sx, op, hz = D._site_lookup_original
        # today's functions restored first (the gate keeps them)
        D.site_xyz, D.own_positions, D.horiz_from_site = sx, op, hz
        # if: the gate
        if mode == "gate":
            # nothing replaced
            continue
        # if: this copy reads the receiver's fixes before the site matters (the RQ4 script and the label script): the reader notes the receiver, then reads unchanged
        if "fixes" in how:
            # lambda with a tuple: sets CUR then returns the reader's result
            D.own_positions = lambda gt, test, rx, _op=op: (CUR.__setitem__("rx", rx), _op(gt, test, rx))[1]
        # if: the site lookup itself is replaced (the RQ4 script)
        if "site" in how:
            # the current receiver's position at the named site
            D.site_xyz = site_rx
        # if: the distance is replaced (the label script): its site argument is today's estimate, mapped to its name, then to the receiver's position there
        if "distance" in how:
            # lambda: today's distance function on the receiver's own position at that site
            D.horiz_from_site = lambda lat, lon, h, site, _hz=hz: _hz(lat, lon, h, POS[(next(n for n, x in SITE.items() if np.allclose(x, site, rtol=0, atol=1e-6)), CUR["rx"])])


# function: the residual tables (the residual script's build, both variants) of one mode, written into the mirror
def residuals(mode: str, gt: pd.DataFrame) -> Path:
    # the mode's features folder in the mirror
    out = MIRROR / mode / "features"
    # created
    out.mkdir(parents=True, exist_ok=True)
    # the GPS Klobuchar parameters (the residual script's main's call)
    kl = A7.M.igs_klobuchar(A7.S04.IGS_BRDC)
    # if: the gate: one build over every receiver, today's site
    if mode == "gate":
        # the residual script's production tables
        tab, ep, _, _ = A7.build(gt, kl, A7.BDS_LAG_S)
    # else: one build per receiver with that receiver's positions, then the rows put back in the residual script's order (sessions, then receivers)
    else:
        # lists of per-receiver tables per variant
        tl, el = {v: [] for v in A7.VARIANTS}, {v: [] for v in A7.VARIANTS}
        # the receivers the residual script loops over
        rxs = list(A7.RXS)
        # for loop over them
        for rx in rxs:
            # one receiver in the residual script's loop
            A7.RXS = [rx]
            # its site lookup
            CUR["rx"] = rx
            # the residual script's build for this receiver
            t_, e_, _, _ = A7.build(gt, kl, A7.BDS_LAG_S)
            # for loop over the variants: kept
            for v in A7.VARIANTS:
                tl[v].append(t_[v])  # the receiver's residual table of this variant
                el[v].append(e_[v])  # and its check table
        # the residual script's receivers restored
        A7.RXS = rxs
        # function: the rows of one variant sorted back into (session, receiver) order, stable within a pair
        order = lambda d: d.assign(_t=d.test_code.map({s: i for i, s in enumerate(A7.SESSIONS)}), _r=d.receiver.map({r: i for i, r in enumerate(rxs)})).sort_values(["_t", "_r"], kind="stable").drop(columns=["_t", "_r"]).reset_index(drop=True)
        # dict comprehensions: concatenated and ordered
        tab = {v: order(pd.concat(tl[v], ignore_index=True)) for v in A7.VARIANTS}
        ep = {v: order(pd.concat(el[v], ignore_index=True)) for v in A7.VARIANTS}  # the check tables likewise
    # the residual script's writer into the mirror
    A7.OUT = out
    A7.write_tables(tab, ep)  # the residual script's writer (rounded, byte-stable gzip)
    # the folder
    return out


# function: the base table, the 10 s windows of every part and the training labels of one mode, from that mode's residuals
def base_windows_labels(mode: str, gt: pd.DataFrame, feat: Path, tr_override=None) -> tuple:
    # the feature script reads the residual script's tables from its OUT: pointed at the mode's mirror folder
    F7.OUT = feat
    # the feature script's base table, unrounded in memory (as its final run computes its windows)
    base, _ = F7.build_base(gt)
    # written at four decimals as the feature script writes it
    F7.E.write_gz(base.round(4), feat / "features_base_1hz.csv.gz")
    # the training-label script's quiet screen reads the residual extremes from the base table: its reader pointed at this mode's base
    TL.residuals = lambda: TL.no_sealed(pd.read_csv(feat / "features_base_1hz.csv.gz", usecols=TL.KEYS + TL.RES_COLS, dtype={"test_code": str})).assign(res_max=lambda d: d[TL.RES_COLS].max(axis=1))
    # the training-label script main's final-label block: in-window passes, the final labels and pooled intervals, the quiet rule, the Stave stop, strict
    wp = TL.in_window_pass()
    fin = TL.read_labels("labels/labels_1hz_final.csv")  # the final labels
    pv = TL.pooled("labels/labels_evidence_final.csv")  # the final pooled attack intervals
    qf = TL.quiet_rule_b(fin, pv, wp)  # the quiet seconds on the final labels
    stop = TL.L.stave_stop()  # the Stave stop (12:27:56-12:32:36)
    in_stop = ((qf.test_code == "2.6.5") & qf.sec.between(*stop)).to_numpy()  # boolean mask: quiet seconds inside the stop
    qf = qf[~in_stop].reset_index(drop=True)  # removed before strict, as the training-label script does
    strict_f, _, why_f = TL.k2_strict(qf, TL.residuals(), pv)  # strict with this mode's residual extremes
    tr = TL.train_table(fin, strict_f, why_f)  # the training-label script's training table
    sm = ((tr.test_code == "2.6.5") & tr.sec.between(*stop)).to_numpy()  # boolean mask: the stop's seconds in the table
    tr.loc[sm, "reason"] = "Stave stop: training label = scoring label (" + fin.label[sm] + ")"  # the stop's reason text, as the training-label script writes it
    # written into the mirror as the training-label script writes it
    tr.to_csv(MIRROR / mode / "labels" / "train_labels_1hz.csv", index=False, lineterminator="\n")
    # 07 main_final's window block at 10 s: the Stave stop's orbit-free fill, the final labels, the windows, the training label in the index
    fr = pd.read_csv(PROC / "labels" / "labels_1hz_final.csv", usecols=F7.KEYS + ["final_rule"], dtype={"final_rule": str}, keep_default_na=False)
    st = fr[(fr.test_code == "2.6.5") & fr.final_rule.str.contains("the Stave stop relabel", regex=False)]  # the Stave stop's seconds as the final labels moved them
    stopsec = {rx: set(st[st.receiver == rx].sec.astype(int)) for rx in F7.RXS}  # dict comprehension: receiver -> its stop seconds
    F7.STOP_SECS.update(*stopsec.values())  # set.update with unpacking: the feature script's module record of the stop seconds
    F7.stave_fill(base, gt, stopsec)  # the feature script's orbit-free fill of the stop, in place
    lab = pd.read_csv(PROC / "labels" / "labels_1hz_final.csv", usecols=F7.KEYS + ["label"])  # the final scoring labels
    trk = pd.read_csv(MIRROR / mode / "labels" / "train_labels_1hz.csv", usecols=F7.KEYS + ["train_label"])  # this mode's training labels
    idx, parts = F7.build_windows(base, lab, 10)  # the feature script's windows at 10 s on the final labels
    idx = idx.merge(trk, on=F7.KEYS, how="left", validate="one_to_one")  # the decision second's training label beside its scoring label
    # written with the feature script's six-significant-figure writer into the mirror
    d = MIRROR / mode / "features_final" / "w10"
    d.mkdir(parents=True, exist_ok=True)  # folder created
    for rx in F7.RXS:  # for loop over the receivers
        F7.write_gz_6sf(idx[idx.receiver == rx], d / f"{rx}_index.csv.gz")  # the index file
        for pt in F7.PARTS:  # for loop over the four parts
            F7.write_gz_6sf(parts[rx][pt], d / f"{rx}_{pt}.csv.gz")  # one part at six significant figures
    # the base, the training labels and the window folder
    return base, tr, d


# function: the label script's own-position evidence of one mode (departures beyond 50 m for at least 10 s, X5 and u-blox 01)
def evidence(mode: str) -> pd.DataFrame:
    # the label script's inputs and session windows
    d = G6.load_inputs()
    W = G6.windows(d["gt"])  # the label script's session windows
    # the label script's function (its distance call goes through the wrapper in the per-receiver mode)
    ev, _ = G6.position_evidence(d["gt"], W)
    # the evidence rows
    return pd.DataFrame(ev)


# function: the RQ4 monitors of one mode (both monitors; only the position monitor can move)
def monitors(mode: str) -> tuple:
    # the transfer script's freeze guard first, as the RQ4 script's main runs it before any 2.6.12 read
    proof = RQ4.X.guard()
    check(f"{mode}: the transfer script's freeze guard holds before 2.6.12's own fixes are read", bool((proof.in_freeze_commit & proof.equal_to_commit & proof.sha256_matches_manifest).all()), f"{len(proof)} frozen files")  # recorded: every frozen file equal to the freeze commit
    # the RQ4 script's main's inputs: the sessions and the gains
    gt = pd.read_csv(RQ4.E.PROC / "ground_truth" / "session_ground_truth.csv", dtype={"test_code": str}).set_index("test_code")
    st = pd.read_csv(RQ4.E.PROC / "ground_truth" / "receiver_status_timeseries.csv", dtype={"test_code": str}, usecols=["test_code", "receiver", "timestamp_utc", "band", "agc_gain_db", "agc_cnt"])  # the gains, as the RQ4 script's main reads them
    # the RQ4 script's grids and monitors
    G = RQ4.grids()
    _, PTS, RUNS, LATm, _, COV = RQ4.monitors(st, gt, G)  # the RQ4 script's two monitors on every grid
    # written into the mirror with the RQ4 script's writer
    out = MIRROR / mode / "rq4"
    out.mkdir(parents=True, exist_ok=True)  # folder created
    for name, df in (("monitor_points.csv", PTS), ("monitor_runs.csv", RUNS), ("monitor_latency.csv", LATm), ("monitor_coverage.csv", COV)):  # for loop over the four tables
        RQ4.E.write_csv(df, out / name)  # the RQ4 script's writer
    # the tables
    return PTS, RUNS, LATm, COV


# function: the observables script's per-satellite code residual in 2.6.10 for G and E, and the spoofed-tracking script's per-second spoofed-tracking flag (a check, never a label)
def tracking_2610(mode: str) -> pd.DataFrame:
    # sessions, orbits and the Klobuchar parameters as the observables script's main builds them
    gt = pd.read_csv(PROC / "ground_truth" / "session_ground_truth.csv").set_index("test_code")
    orb = O5.M.Orbits(O5.D.S04, O5.dtm.date(2026, 9, 14))  # orbits and clocks from the IGS file (the observables script's main)
    kl = O5.M.igs_klobuchar(O5.D.S04.IGS_BRDC)  # the GPS Klobuchar parameters
    # list of per-receiver tables
    rows = []
    # for loop over the receivers
    for rx in O5.RXS:
        # the observables script's capture span and observations of 2.6.10
        spans = O5.D.capture_spans(gt, "2.6.10", rx)
        ob = O5.read_all(O5.M.obs_path(gt.loc["2.6.10", "session_folder"], "2.6.10", rx), spans, rx)  # the observables script's reader: all four constellations at 1 Hz in the span
        # the site lookup for this receiver
        CUR["rx"] = rx
        site = O5.D.site_xyz(O5.M.SITE_OF["2.6.10"])  # the site the observables script would use: in the per-receiver mode this receiver's own position
        # the observables script's glonass_union() writes its table to its module's PROC: pointed at the mirror for the call, so no published file is written
        keep, O5.PROC = O5.PROC, MIRROR / mode / "05e"
        O5.PROC.mkdir(parents=True, exist_ok=True)  # the mirror folder created
        glo = O5.glonass_union()  # the GLONASS channel table from the files read in this run (GPS and Galileo do not use it)
        O5.PROC = keep  # the observables script's folder restored
        # the observables script's per-satellite quantities
        sat = O5.per_satellite(orb, kl, ob, site, "2.6.10", rx, glo)
        # GPS and Galileo rows with a residual
        sat = sat[sat.sys.isin(["G", "E"]) & sat.pr_res_m.notna()]
        # kept with the receiver
        rows.append(sat[["sec", "sv", "sys", "pr_res_m"]].assign(receiver=rx))
    # one table
    d = pd.concat(rows, ignore_index=True)
    # the spoofed-tracking script's per-second rule: at least MIN_RES residuals and more than SHARE of them beyond OVER_M
    d = d.assign(over=d.pr_res_m.abs() > S5.OVER_M)
    g = d.groupby(["receiver", "sec"]).agg(n_ge=("over", "size"), n_over=("over", "sum")).reset_index()  # named aggregations: residuals per second and how many lie beyond OVER_M
    g["spoofed"] = (g.n_ge >= S5.MIN_RES) & (g.n_over / g.n_ge > S5.SHARE)  # the spoofed-tracking script's rule: at least MIN_RES residuals and more than SHARE of them beyond
    # the per-satellite residuals and the per-second flags
    return d, g


# function: the evidence intervals in a fixed order (session, receiver, start), so two lists compare independently of the order found
def srt(d: pd.DataFrame) -> pd.DataFrame:
    # the four key columns, sorted and renumbered
    return d[["test_code", "receiver", "start", "end"]].sort_values(["test_code", "receiver", "start"], kind="stable").reset_index(drop=True)


# function: largest absolute difference per numeric column between two tables of the same rows (NaN where the column is not numeric)
def maxdiff(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    # list comprehension: one row per shared column
    rows = [{"column": c, "max_abs_diff": float(np.nanmax(np.abs(a[c].to_numpy(float) - b[c].to_numpy(float)))) if np.issubdtype(a[c].dtype, np.number) and (a[c].notna() | b[c].notna()).any() else np.nan,
             "cells_differ": int((~((a[c] == b[c]) | (a[c].isna() & b[c].isna()))).sum())} for c in a.columns if c in b.columns]  # continued: the number of cells that differ (NaN equal to NaN)
    # the table
    return pd.DataFrame(rows)


# function: main: positions, the gate, the per-receiver run, the comparisons
def main() -> None:
    # the clock
    t0 = time.time()
    # the mirror created
    MIRROR.mkdir(parents=True, exist_ok=True)
    # 9a
    positions()
    # sessions as the residual script and 07 read them
    gt = pd.read_csv(PROC / "ground_truth" / "session_ground_truth.csv").set_index("test_code")
    # dict: mode -> results
    res = {}
    # for loop over the two modes, the gate first
    for mode in ("gate", "per_receiver"):
        print(f"== {mode} ==", flush=True)  # printed: the mode
        # the site lookups for this mode
        patch(mode)
        CUR["rx"] = None  # no receiver noted yet
        # residuals, base, windows and training labels
        feat = residuals(mode, gt)
        base, tr, wdir = base_windows_labels(mode, gt, feat)  # the feature script's base, the w10 windows and the training-label script's training labels
        # labels' own-position evidence
        CUR["rx"] = None
        ev = evidence(mode)  # the label script's own-position evidence
        # RQ4 monitors
        CUR["rx"] = None
        mon = monitors(mode)  # RQ4's monitors
        # the 2.6.10 check
        trk = tracking_2610(mode)
        # kept
        res[mode] = {"feat": feat, "base": base, "tr": tr, "wdir": wdir, "ev": ev, "mon": mon, "trk": trk}
        print(f"  {mode} done at {time.time() - t0:.0f} s", flush=True)  # printed: progress
        # if: the gate: every product against the published one
        if mode == "gate":
            # residual and epoch tables, byte for byte
            for f in ("residuals_known.csv.gz", "residuals_self.csv.gz", "epochs_known.csv.gz", "epochs_self.csv.gz"):
                check(f"gate: {f} = published", (feat / f).read_bytes() == (PROC / "features" / f).read_bytes())  # byte comparison
            # the base table, byte for byte
            check("gate: features_base_1hz.csv.gz = published", (feat / "features_base_1hz.csv.gz").read_bytes() == (PROC / "features" / "features_base_1hz.csv.gz").read_bytes())
            # the training labels, byte for byte
            check("gate: labels/train_labels_1hz.csv = published", (MIRROR / mode / "labels" / "train_labels_1hz.csv").read_bytes() == (PROC / "labels" / "train_labels_1hz.csv").read_bytes())
            # the w10 window files, byte for byte
            bad = [f.name for f in sorted(wdir.glob("*.csv.gz")) if f.read_bytes() != (PROC / "features_final" / "w10" / f.name).read_bytes()]
            check("gate: features_final/w10 (index and four parts per receiver) = published", not bad, f"{len(list(wdir.glob('*.csv.gz')))} files, differing {bad}")  # recorded
            # the own-position evidence intervals, row for row
            pub = pd.read_csv(PROC / "labels" / "labels_evidence_final.csv", dtype={"test_code": str})
            pub = srt(pub[pub.source == "own_position"])  # boolean filter: the own-position rows, in a fixed order
            mine = srt(ev)  # this run's intervals in the same order (the label script writes its file sorted; position_evidence returns them as found)
            check("gate: own-position evidence = labels/labels_evidence_final.csv (own_position rows)", mine.equals(pub), f"{len(mine)} intervals")  # row-for-row equality after the same sort
            # the monitor tables, byte for byte
            for name in ("monitor_points.csv", "monitor_runs.csv", "monitor_latency.csv", "monitor_coverage.csv"):
                check(f"gate: {name} = published", (MIRROR / mode / "rq4" / name).read_bytes() == (PROC / "rq4" / name).read_bytes())  # byte comparison
            # the observables script's 2.6.10 G/E code residuals against the published per-satellite table (four decimals)
            ob = pd.read_csv(PROC / "features" / "observables_1hz.csv.gz", usecols=["test_code", "receiver", "sec", "sv", "sys", "pr_res_m"], dtype={"test_code": str})
            ob = ob[(ob.test_code == "2.6.10") & ob.sys.isin(["G", "E"]) & ob.pr_res_m.notna()]  # boolean filter: 2.6.10, GPS and Galileo with a residual
            m = trk[0].merge(ob, on=["receiver", "sec", "sv"], suffixes=("", "_pub"))  # merge: this run's residuals beside the published ones
            check("gate: 2.6.10 G/E code residuals = features/observables_1hz.csv.gz (to 1e-4 m)", len(m) == len(ob) == len(trk[0]) and bool((np.abs(m.pr_res_m.round(4) - m.pr_res_m_pub) < 1e-9).all()), f"{len(m)} satellite-seconds")  # recorded: equal at the published four decimals
            # if: any gate failed, stop before any per-receiver value
            if any(not c["pass"] for c in CHECKS):
                pd.DataFrame(CHECKS).to_csv(ROBUST_DIR / "positions_checks.csv", index=False, lineterminator="\n")  # the checks so far, written
                raise SystemExit("STOP: a gate failed; no per-receiver result is reported")  # stop: no per-receiver number without a passed gate
    # the comparisons: per-receiver run against the gate (= published)
    g, p = res["gate"], res["per_receiver"]
    # residual tables: per column largest absolute difference
    for v in ("known", "self"):
        a = pd.read_csv(g["feat"] / f"residuals_{v}.csv.gz", keep_default_na=False, na_values=[""])  # the gate's residual table (= published)
        b = pd.read_csv(p["feat"] / f"residuals_{v}.csv.gz", keep_default_na=False, na_values=[""])  # the per-receiver one
        check(f"residuals {v}: same rows and keys", len(a) == len(b) and a[["test_code", "receiver", "sec", "sv"]].equals(b[["test_code", "receiver", "sec", "sv"]]), f"{len(a)} rows")  # recorded
        maxdiff(a, b).assign(product=f"residuals_{v}").to_csv(ROBUST_DIR / f"positions_compare_residuals_{v}.csv", index=False, lineterminator="\n")  # the per-column differences, written
    # base table: per column largest absolute difference (orbit-free columns must be identical)
    bd = maxdiff(g["base"], p["base"]).merge(pd.read_csv(PROC / "features" / "features_base_lineage.csv")[["column", "block"]], on="column", how="left")
    bd.to_csv(ROBUST_DIR / "positions_compare_base.csv", index=False, lineterminator="\n")  # written
    free = bd[bd.block == "orbit-free"]  # boolean filter: the orbit-free block (the adopted pipeline's inputs)
    check("base table: every orbit-free column identical", bool((free.cells_differ == 0).all()), f"{len(free)} columns")  # recorded
    # training labels second by second, and the counts per class, receiver and session
    tg, tp = g["tr"], p["tr"]
    diff_tr = tg[tg.train_label != tp.train_label]  # seconds whose training label differs
    check("training labels: identical second by second", diff_tr.empty, f"{len(tg)} seconds, {len(diff_tr)} differ")  # recorded
    cnt = lambda t: t.groupby(["test_code", "receiver", "train_label"]).size()  # lambda: counts per session, receiver and training label
    pd.concat([cnt(tg).rename("published"), cnt(tp).rename("per_receiver")], axis=1).fillna(0).astype(int).reset_index().to_csv(ROBUST_DIR / "positions_compare_train_labels.csv", index=False, lineterminator="\n")  # both counts side by side, written
    # windows: every part of every receiver; the orbit-free and geometry parts and the index must be byte-identical
    wrows = []
    for f in sorted(g["wdir"].glob("*.csv.gz")):  # for loop over the gate's window files
        same_bytes = f.read_bytes() == (p["wdir"] / f.name).read_bytes()  # byte comparison with the per-receiver file
        if same_bytes:  # if: identical
            wrows.append({"file": f.name, "identical": True, "columns_differing": 0, "max_abs_diff": 0.0})  # recorded as identical
            continue  # continue: next file
        a, b = pd.read_csv(f), pd.read_csv(p["wdir"] / f.name)  # both tables read
        md = maxdiff(a, b)  # per-column differences
        wrows.append({"file": f.name, "identical": False, "columns_differing": int((md.cells_differ > 0).sum()), "max_abs_diff": float(md.max_abs_diff.max())})  # recorded with the differing columns and the largest difference
    WR = pd.DataFrame(wrows)  # the window table
    WR.to_csv(ROBUST_DIR / "positions_compare_windows_w10.csv", index=False, lineterminator="\n")  # written
    adopted = WR[WR.file.str.contains("_index|_orbitfree")]  # boolean filter: the index and orbit-free files, which the adopted pipeline reads whole
    check("windows w10: index and orbit-free parts byte-identical", bool(adopted.identical.all()), f"{len(adopted)} files")  # recorded
    # the geometry part: the adopted pipeline reads only its orbit-free (no_residual) columns (choices_lib.d1_columns)
    wcd = pd.read_csv(PROC / "features" / "features_window_columns.csv")  # the feature script's window dictionary
    d1 = set(wcd[(wcd.w == 10) & (wcd.no_residual == 1)].column)  # set: the orbit-free columns at w = 10
    gbad, gdiff = [], set()  # tuple assignment: differing orbit-free columns, and every differing column
    for rx in F7.RXS:  # for loop over the receivers
        a = pd.read_csv(g["wdir"] / f"{rx}_geometry.csv.gz")  # the gate's geometry part (= published)
        b = pd.read_csv(p["wdir"] / f"{rx}_geometry.csv.gz")  # the per-receiver one
        for c in a.columns:  # for loop over its columns
            if not bool(((a[c] == b[c]) | (a[c].isna() & b[c].isna())).all()):  # if: any cell differs (NaN equal to NaN)
                gdiff.add(c)  # recorded
                if c in d1:  # if: a column the adopted pipeline reads
                    gbad.append(f"{rx} {c}")  # recorded as a failure
    check("windows w10: every orbit-free column of the geometry part identical (the adopted pipeline's inputs)", not gbad, f"{sum(c in d1 for c in a.columns)} orbit-free columns per receiver; differing columns, none orbit-free: {sorted(gdiff)}")  # recorded
    # labels: the own-position evidence intervals
    eg, ep_ = srt(g["ev"]), srt(p["ev"])
    check("labels: own-position evidence intervals identical (so every scoring label is identical)", eg.equals(ep_), f"{len(eg)} published, {len(ep_)} per receiver")  # recorded: interval for interval
    pd.concat([eg.assign(run="published"), ep_.assign(run="per_receiver")]).to_csv(ROBUST_DIR / "positions_compare_evidence.csv", index=False, lineterminator="\n")  # both sets written
    # position monitor: points, runs and intervals of the position monitor; the gain monitor must not move
    PG, RG, LG, CG = g["mon"]
    PP, RP, LP, CP = p["mon"]  # tuple unpacking: the per-receiver monitor tables
    gain_same = PG[PG.monitor == "gain"].equals(PP[PP.monitor == "gain"]) and RG[RG.monitor == "gain"].equals(RP[RP.monitor == "gain"])  # the gain monitor's points and runs equal
    check("RQ4: the gain monitor unchanged", bool(gain_same))  # recorded
    pos = lambda d: d[d.monitor == "position"].reset_index(drop=True)  # lambda: the position monitor's rows
    mp = maxdiff(pos(RG).select_dtypes("number"), pos(RP).select_dtypes("number"))  # per-column differences of its runs
    mp.to_csv(ROBUST_DIR / "positions_compare_monitor_runs.csv", index=False, lineterminator="\n")  # written
    lat_same = pos(LG)[["receiver", "outer", "share", "onset", "detected", "delay_s"]].equals(pos(LP)[["receiver", "outer", "share", "onset", "detected", "delay_s"]])  # its detections and delays equal
    check("RQ4: the position monitor's detections and delays identical at every point", bool(lat_same), f"{len(pos(LG))} intervals")  # recorded
    check("RQ4: the position monitor's realised shares and thresholds", bool((mp.cells_differ == 0).all()), f"{int((mp.cells_differ > 0).sum())} numeric columns differ; largest {mp.max_abs_diff.max():.6g}")  # recorded: realised shares and thresholds
    # 2.6.10 spoofed-tracking flag per second
    fg, fp = g["trk"][1], p["trk"][1]
    mm = fg.merge(fp, on=["receiver", "sec"], suffixes=("_pub", "_rx"))  # merge: published-position flags beside per-receiver flags
    check("2.6.10 check: per-second spoofed-tracking flags identical", bool(len(mm) == len(fg) == len(fp) and (mm.spoofed_pub == mm.spoofed_rx).all()), f"{len(mm)} receiver-seconds; last spoofed second per receiver {fp[fp.spoofed].groupby('receiver').sec.max().to_dict()}")  # recorded
    # summary
    pd.DataFrame(CHECKS).to_csv(ROBUST_DIR / "positions_checks.csv", index=False, lineterminator="\n")
    print(f"done in {time.time() - t0:.0f} s; {len(CHECKS)} checks, {sum(not c['pass'] for c in CHECKS)} failed")  # printed: run time and failures


# if: run as a script
if __name__ == "__main__":
    # runs the own-position check
    main()
