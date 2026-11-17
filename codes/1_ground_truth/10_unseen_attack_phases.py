#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
The unseen attack (test 2.6.12) and the transfer across receivers: the tables that must exist before anything is trained or evaluated
on them. No model is fitted, no agent is trained and no detector is run here.

Writes (Data/processed/transfer/)
  unseen_attack_columns.csv            per outer fold and target receiver, each of the mosaic-X5's 36 selected columns: its name on the
                                       target, the observation codes behind it on both receivers, renamed or not, its non-missing share
  unseen_attack_renaming.csv           the renaming map (target, constellation, observable: X5 code -> target code, admitted)
  unseen_attack_renaming_evidence.csv  same satellite, same second, clean seconds of the five position sessions: target minus X5, GPS as control
  unseen_attack_renaming_caveats.csv   clean-window medians per receiver behind the three caveats of the map
  unseen_attack_2612_phases.csv        2.6.12 per receiver and phase: span, label counts, windows and decidable seconds inferred from the labels
  unseen_attack_2612_events.csv        2.6.12 per receiver: the traced instants of its timeline and the observed jamming-phase edges
  unseen_attack_onset_sources.csv      every pooled attack interval and the evidence that set its onset
  inspection_monitors.csv              the two reference monitors as they are run (the evidence script's definitions), per receiver
  inspection_monitor_coverage.csv      their coverage on the shared stream grid of the five position sessions (2.6.12 not read)
  unseen_attack_jobs.csv               job counts and hours, from the training log of the agent experiment
  unseen_attack_checks.csv             this script's self-checks
2.6.12 is read only from the label-side files (labels_1hz_final.csv, scoring_fold_policy_final.csv), its spoofing timeline and the
label script's phase boundaries; every feature, observable, status and position table is read with its 2.6.12 rows dropped at once.
Usage   python3 codes/1_ground_truth/10_unseen_attack_phases.py
"""
from __future__ import annotations  # future import: type hints kept as text (the project's convention)

import ast  # standard library: reads the observables script's code table as a literal without running the observables script
import importlib.util  # standard library: loads the evidence script (its name starts with a digit)
import re  # standard library: regular expressions, to parse the agent experiment's run log
import sys  # standard library: module search path and exit status
from pathlib import Path  # standard library: paths

import numpy as np  # numpy: arrays
import pandas as pd  # pandas: tables

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
CODES = PROJECT / "codes"  # the project's scripts
PROC = PROJECT / "Data" / "processed"  # processed tables
OUT = PROC / "transfer"  # this step's folder
FEAT = PROC / "features_final" / "w10"  # the final window files at w = 10
SEALED = "2.6.12"  # the unseen attack type, never read from a feature, observable, status or position table here
SESSIONS = ("2.6.5", "2.6.2", "2.6.10", "2.6.4", "2.6.11")  # tuple: the five position sessions in the screen script's fold order
RXS = ("SBF_02", "SBF_03", "UBX_01")  # tuple: mosaic-X5, mosaic-H main antenna, u-blox 01
TARGETS = ("UBX_01", "SBF_03")  # tuple: the transfer targets, primary first
NAME = {"SBF_02": "mosaic-X5", "SBF_03": "mosaic-H", "UBX_01": "u-blox 01"}  # dict: stream -> name in the text
FAMILY = {"SBF_02": "SBF", "SBF_03": "SBF", "UBX_01": "UBX"}  # dict: stream -> receiver family of the observables script's code table
KEYS = ["test_code", "receiver", "sec"]  # list: the key columns of every window file
W = 10  # the window length of P (seconds)
HIST = 29  # the environment script's rule: a step is valid when the windows ending t-29 .. t lie in one run
PERSIST = 10  # the evidence script's persistence of the gain monitor on every receiver (ground_truth/early_warning_operating_points.csv), used for the position monitor too
GAIN_FIXED = {"SBF_02": "1, 2, 3, 5 dB", "SBF_03": "1, 2, 3, 5 dB", "UBX_01": "100, 200, 300, 500 counts"}  # dict: the evidence script's fixed settings, drawn on the curves
OBS_ORDER = (("code", "C"), ("carrier", "L"), ("doppler", "D"), ("cn0", "S"))  # tuple of pairs: observable -> RINEX letter, in a fixed order
PHASES = ("pre_test", "jam_spoof", "spoof", "post_test")  # tuple: the label script's phases of 2.6.12 in time order
CHECKS = []  # list: the self-checks, one dict each


def check(name: str, ok: bool, detail: str) -> None:  # function: records one self-check
    CHECKS.append({"check": name, "pass": bool(ok), "detail": detail})  # appended as a row


def write_csv(df: pd.DataFrame, name: str) -> None:  # function: a plain CSV with fixed line endings, so runs again are byte-identical
    OUT.mkdir(parents=True, exist_ok=True)  # the folder
    df.to_csv(OUT / name, index=False, lineterminator="\n")  # written


def read_sealed(path: Path, **kw) -> pd.DataFrame:  # function: reads a table and drops every 2.6.12 row before anything else sees it
    df = pd.read_csv(path, dtype={"test_code": str}, **kw)  # read; session codes kept as text
    df = df[df.test_code != SEALED].reset_index(drop=True)  # 2.6.12 dropped at once
    if (df.test_code == SEALED).any():  # if: one survived (cannot)
        raise SystemExit(f"STOP: a 2.6.12 row survived in {path.name}")  # stop
    return df  # the table


def hms(sec) -> str:  # function: unix seconds -> "HH:MM:SS" UTC, or "" when undefined
    return "" if sec is None or (isinstance(sec, float) and np.isnan(sec)) else pd.Timestamp(int(sec), unit="s").strftime("%H:%M:%S")  # conditional expression: text or empty


def load_module(name: str, path: Path):  # function: loads a script as a module without running its main()
    spec = importlib.util.spec_from_file_location(name, path)  # where it is
    mod = importlib.util.module_from_spec(spec)  # an empty module
    sys.modules[name] = mod  # registered, so its own imports resolve
    spec.loader.exec_module(mod)  # its top level (constants and functions)
    return mod  # the module


def codes_05e() -> dict:  # function: the observables script's table of L1-type codes per receiver family and constellation, read as a literal (the observables script is not run)
    tree = ast.parse((CODES / "4_features" / "01_observables.py").read_text(encoding="utf-8"))  # the observables script's syntax tree
    for n in tree.body:  # for loop over the top-level statements
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CODES" for t in n.targets):  # if: the CODES assignment
            return ast.literal_eval(n.value)  # its value as a Python dict
    raise SystemExit("STOP: CODES not found in the observables script")  # stop if absent


# ----------------------------------------------------------------------------------------------------------------------
# 1. column availability
# ----------------------------------------------------------------------------------------------------------------------
def scope_of(base: str) -> str:  # function: the constellation a base column describes, or "all" for the cross-constellation ones
    p = base.split("_")[0]  # the prefix before the first underscore
    return p if p in ("G", "E", "R", "C") else "all"  # conditional expression: one constellation, or all of them (all_, n_, d_, ms_)


def observables_of(sources: str) -> list:  # function: which of code, carrier, Doppler and C/N0 a column is built from (the feature script's lineage text)
    toks = str(sources).split(";")  # the source tokens
    out = []  # the RINEX letters used
    for obs, L in OBS_ORDER:  # for loop over the four observables in a fixed order
        if any(t.startswith(f"RINEX:{L}1") for t in toks) or (L == "L" and any("LLI(carrier)" in t for t in toks)):  # if: a token names that observable (lock-loss flags come with the carrier)
            out.append(L)  # kept
    return out  # the letters


def code_text(fam: str, scope: str, letters: list, CODE: dict) -> str:  # function: the observation codes behind a column on one receiver family, e.g. "E:C1X,D1X"
    systems = [scope] if scope != "all" else ["G", "R", "E", "C"]  # conditional expression: one constellation or all four
    return ";".join(f"{s}:{','.join(L + CODE[fam][s][1:] for L in letters)}" for s in systems) if letters else ""  # nested generators: system -> its codes


def columns_table(CODE: dict) -> tuple:  # function: the availability table and the name-presence counts
    wc = pd.read_csv(PROC / "features" / "features_window_columns.csv")  # the feature script's window dictionary
    wc = wc[(wc.w == W) & (wc.no_residual == 1)].set_index("column")  # the 778 candidate columns at w = 10
    names, present = {}, {}  # receiver -> column names of its four part files; receiver -> candidate names present
    for rx in RXS:  # for loop over the receivers
        names[rx] = set().union(*[set(pd.read_csv(FEAT / f"{rx}_{pt}.csv.gz", nrows=0).columns) for pt in ("orbitfree", "geometry", "self", "known")])  # set union of the four headers (no row read)
        present[rx] = sum(c in names[rx] for c in wc.index)  # generator in sum(): candidate names present
    check("every candidate column name exists on every receiver", all(v == len(wc) for v in present.values()), f"{present} of {len(wc)}")  # recorded
    sel = pd.read_csv(PROC / "random_forest" / "random_forest_selection_primary.csv", dtype={"held_out": str})  # P's outer selections (the random forest's table)
    sel9 = pd.read_csv(PROC / "logistic_regression" / "logistic_regression_selection_primary.csv", dtype={"held_out": str})  # and the logistic regression's
    s2 = sel[sel.receivers == "SBF_02"].set_index("held_out").selected_columns  # the mosaic-X5's 36 columns per fold
    s9 = sel9[sel9.receivers == "SBF_02"].set_index("held_out").selected_columns  # the same from the logistic regression
    check("the X5's selected columns are the same in the logistic regression and the random forest", all(s2[h] == s9[h] for h in SESSIONS), "5 folds compared")  # recorded
    need = sorted(set(";".join(s2[h] for h in SESSIONS).split(";")))  # every column selected in some fold
    vals = {}  # receiver -> windows with the needed columns
    for rx in RXS:  # for loop over the receivers
        v = read_sealed(FEAT / f"{rx}_index.csv.gz", usecols=KEYS)  # the window keys, 2.6.12 dropped
        for pt in ("orbitfree", "geometry"):  # for loop over the two parts that hold candidate columns
            cols = [c for c in need if wc.at[c, "part"] == pt]  # list comprehension: the needed columns in this part
            v = v.merge(read_sealed(FEAT / f"{rx}_{pt}.csv.gz", usecols=KEYS + cols), on=KEYS, how="left", validate="one_to_one")  # merged one to one
        vals[rx] = v  # kept
    rows = []  # one row per fold, target and column
    for h in SESSIONS:  # for loop over the outer folds
        cs = s2[h].split(";")  # the fold's 36 columns in selection order
        x5 = vals["SBF_02"][vals["SBF_02"].test_code == h]  # the X5's held-out windows
        for t in TARGETS:  # for loop over the targets
            tg = vals[t][vals[t].test_code == h]  # the target's held-out windows
            for k, c in enumerate(cs, 1):  # for loop over the columns, numbered from 1
                sc, letters = scope_of(wc.at[c, "base_column"]), observables_of(wc.at[c, "sources"])  # tuple assignment: constellation scope and observables
                renamed = FAMILY[t] != "SBF" and sc in ("E", "all") and bool(letters)  # the u-blox's Galileo E1 codes differ from the X5's
                # dict literal over three lines: the row
                rows.append({"held_out": h, "target": t, "position": k, "column": c, "base_column": wc.at[c, "base_column"], "stat": wc.at[c, "stat"], "part": wc.at[c, "part"], "scope": sc,
                             "x5_codes": code_text("SBF", sc, letters, CODE), "target_codes": code_text(FAMILY[t], sc, letters, CODE), "renamed": renamed, "name_on_target": c in names[t],  # dict literal continued: codes, renaming flag and presence
                             "x5_windows": len(x5), "x5_nonmissing": float(x5[c].notna().mean()), "target_windows": len(tg), "target_nonmissing": float(tg[c].notna().mean()), "is_indicator": c.startswith("ind_missing_")})  # dict literal continued
    return pd.DataFrame(rows), present, len(wc)  # the table, the counts and the number of candidates


# ----------------------------------------------------------------------------------------------------------------------
# 2. renaming map and its evidence
# ----------------------------------------------------------------------------------------------------------------------
def renaming(CODE: dict) -> tuple:  # function: the map, and the same-satellite comparison on clean seconds of the five position sessions
    rows = []  # the map
    for t in TARGETS:  # for loop over the targets
        for s in ("G", "R", "E", "C"):  # for loop over the constellations
            for obs, L in OBS_ORDER:  # for loop over the observables
                a, b = L + CODE["SBF"][s][1:], L + CODE[FAMILY[t]][s][1:]  # tuple assignment: the X5's code and the target's
                rows.append({"target": t, "constellation": s, "observable": obs, "x5_code": a, "target_code": b, "renamed": a != b, "admitted": True,  # dict literal: one map row
                             "basis": "same signal and frequency (Galileo E1: C = pilot on the Septentrio, X = data + pilot on the u-blox); same-satellite evidence in unseen_attack_renaming_evidence.csv; the decision of the transfer study" if a != b else "identical code"})  # the reason
    o = read_sealed(PROC / "features" / "observables_1hz.csv.gz", usecols=["test_code", "receiver", "sec", "sv", "sys", "cn0", "dop_hz", "cmc_rate_m_s"])  # the observables script's per-satellite table, 2.6.12 dropped
    lab = read_sealed(PROC / "labels" / "labels_1hz_final.csv", usecols=["test_code", "receiver", "sec", "label"])  # the final labels, 2.6.12 dropped
    o = o.merge(lab[lab.label == "clean"][KEYS], on=KEYS)  # clean seconds only
    a = o[o.receiver == "SBF_02"].set_index(["test_code", "sec", "sv"])  # the X5 by session, second and satellite
    ev = []  # evidence rows
    for t in TARGETS:  # for loop over the targets
        b = o[o.receiver == t].set_index(["test_code", "sec", "sv"])  # the target, same keys
        j = a.join(b, lsuffix="_x5", rsuffix="_t", how="inner")  # same satellite, same second
        j["dcn0"] = j.cn0_t - j.cn0_x5  # C/N0, target minus X5
        j["ddop"] = j.dop_hz_t - j.dop_hz_x5  # Doppler, target minus X5 (holds the two clocks' drift difference)
        gps = j[j.sys_x5 == "G"].groupby(level=[0, 1]).ddop.median()  # per second, the GPS median of that difference: the common clock term
        j["ddop_c"] = j.ddop - gps.reindex(j.index.droplevel(2)).to_numpy()  # the difference with the GPS common term removed
        j["dcmc"] = j.cmc_rate_m_s_t - j.cmc_rate_m_s_x5  # code-minus-carrier rate, target minus X5
        mad = lambda s: float((s - s.median()).abs().median())  # lambda: median absolute deviation
        for s, g in j.groupby("sys_x5", sort=True):  # for loop over the constellations
            # dict literal over three lines: the evidence row
            ev.append({"target": t, "constellation": s, "pairs": len(g), "cn0_diff_median_db": float(g.dcn0.median()), "cn0_diff_iqr_db": float(g.dcn0.quantile(0.75) - g.dcn0.quantile(0.25)),
                       "doppler_minus_gps_term_median_hz": float(g.ddop_c.median()), "doppler_minus_gps_term_mad_hz": mad(g.ddop_c), "cmc_rate_diff_mad_m_s": mad(g.dcmc),  # dict literal continued: the Doppler and code-minus-carrier comparisons
                       "x5_cn0_median": float(g.cn0_x5.median()), "target_cn0_median": float(g.cn0_t.median()), "target_cn0_noninteger_share": float((g.cn0_t.dropna() % 1 != 0).mean()), "x5_cn0_step_db": float(np.diff(np.unique(g.cn0_x5.dropna())).min())})  # dict literal continued
    E = pd.DataFrame(ev)  # the table
    gp = E[E.constellation == "G"]  # GPS rows: the term was removed on them, so their median is 0 by construction
    check("GPS Doppler median after removing the GPS term is 0 (construction)", bool((gp.doppler_minus_gps_term_median_hz.abs() < 1e-9).all()), "sanity of the control")  # recorded
    win = {}  # receiver -> clean windows of the change and level columns that the caveats quote
    cols = [f"{s}_{q}" for q in ("dC_dc_median__median", "cn0_median__median") for s in ("G", "R", "E", "C")]  # list comprehension: the eight columns
    for rx in RXS:  # for loop over the receivers
        ix = read_sealed(FEAT / f"{rx}_index.csv.gz", usecols=KEYS + ["label"])  # window keys and labels
        v = read_sealed(FEAT / f"{rx}_orbitfree.csv.gz", usecols=KEYS + cols)  # the columns
        win[rx] = ix.merge(v, on=KEYS, validate="one_to_one").query("label == 'clean'")  # clean windows only
    cav = []  # caveat rows
    for c in cols:  # for loop over the columns
        x = win["SBF_02"][c]  # the X5's clean values
        cav.append({"column": c, "x5_median": float(x.median()), "x5_iqr": float(x.quantile(0.75) - x.quantile(0.25)), "mosaic_h_median": float(win["SBF_03"][c].median()), "ublox_median": float(win["UBX_01"][c].median()), "clean_windows": f"{len(win['SBF_02'])}/{len(win['SBF_03'])}/{len(win['UBX_01'])}"})  # one row
    return pd.DataFrame(rows), E, pd.DataFrame(cav)  # the map, the evidence and the caveat medians


# ----------------------------------------------------------------------------------------------------------------------
# 3. 2.6.12 phases, inferred windows and decidable seconds; the traced instants; onset sources
# ----------------------------------------------------------------------------------------------------------------------
def infer_steps(d: pd.DataFrame) -> pd.DataFrame:  # function: the feature script's window rule and the environment script's validity rule applied to one receiver's labels of one session
    d = d.sort_values("sec").reset_index(drop=True)  # in time order
    s, ok = d.sec.to_numpy(np.int64), (d.label != "excluded").to_numpy()  # tuple assignment: seconds and non-excluded mask
    if not (np.diff(s) == 1).all():  # if: the label grid is not consecutive (it is, the label script's grid)
        raise SystemExit("STOP: label grid not consecutive")  # stop
    c = np.convolve(ok.astype(int), np.ones(W, int))[:len(ok)]  # running count of non-excluded seconds over the last W seconds
    win = (c == W) & (np.arange(len(ok)) >= W - 1)  # a window ends at t when t-9 .. t are all in the capture and none excluded (07.5)
    new = win & np.r_[True, ~win[:-1]]  # True where a run of windows begins
    start = np.maximum.accumulate(np.where(new, s, 0))  # running maximum: each second's run start
    return d.assign(window=win, valid=win & (s - start >= HIST))  # the environment script's rule: valid from 29 s after the run's first window


def phases(lab12: pd.DataFrame, pol: pd.DataFrame) -> pd.DataFrame:  # function: 2.6.12 per receiver and phase
    rows = []  # one row per receiver and phase
    for rx in RXS:  # for loop over the receivers
        d = infer_steps(lab12[lab12.receiver == rx])  # its seconds with inferred windows and steps
        att = d.label.str.startswith("attack")  # attack seconds
        for ph in PHASES:  # for loop over the phases
            p = d[d.phase == ph]  # the phase's seconds
            fv = p.sec[p.valid]  # its decidable seconds
            fa = p.sec[p.valid & att[p.index]]  # its decidable attack seconds
            # dict literal over four lines: the row
            rows.append({"receiver": rx, "phase": ph, "start_utc": hms(p.sec.min()), "end_utc": hms(p.sec.max()), "seconds": len(p), **{f"n_{k}": int((p.label == k).sum()) for k in ("clean", "attack_jam_spoof", "attack_spoof", "uncertain", "excluded")},
                         "windows_inferred": int(p.window.sum()), "decidable_inferred": int(p.valid.sum()), "decidable_attack_inferred": int(len(fa)), "first_decidable_utc": hms(fv.min()) if len(fv) else "", "first_decidable_attack_utc": hms(fa.min()) if len(fa) else "",  # dict literal continued: inferred windows and decidable seconds
                         "meaning": {"pre_test": "before the scheduled start (Stave margin)", "jam_spoof": "jamming on L1/E1, G1 and B1 together with the spoofing (observed L1 gain step, jamming-phase script)", "spoof": "time spoofing alone after the jamming stopped (jamming-phase script)", "post_test": "after the attack interval"}[ph],  # dict lookup: what the phase is
                         "fold_policy": f"precision {bool(pol.at[rx, 'precision'])}; false alarms: {pol.at[rx, 'false_alarms']}"})  # the scoring module's policy for 2.6.12
    return pd.DataFrame(rows)  # the table


def validate_inference(lab: pd.DataFrame) -> tuple:  # function: the inferred windows and steps against the real window index and the environment script's step table on the five position sessions
    st = pd.read_csv(PROC / "rl_environment" / "steps.csv.gz", dtype={"session": str})  # the environment script's step table (position sessions only)
    mm_w, mm_v, n = 0, 0, 0  # mismatched windows; mismatched validity; seconds compared
    for rx in RXS:  # for loop over the receivers
        ix = read_sealed(FEAT / f"{rx}_index.csv.gz", usecols=KEYS)  # the real windows
        real = set(zip(ix.test_code, ix.sec))  # set of (session, second)
        for t in SESSIONS:  # for loop over the sessions
            d = infer_steps(lab[(lab.receiver == rx) & (lab.test_code == t)])  # inferred
            got = set(zip([t] * int(d.window.sum()), d.sec[d.window]))  # inferred windows as a set
            mm_w += len(got ^ {x for x in real if x[0] == t})  # symmetric difference: windows in one set only
            s11 = st[(st.receiver == rx) & (st.session == t)].set_index("sec").valid  # the environment script's validity
            mm_v += int((d.set_index("sec").valid != s11.reindex(d.sec).fillna(False).to_numpy()).sum())  # seconds whose validity differs
            n += len(d)  # counted
    return mm_w, mm_v, n  # the counts


def events_2612() -> pd.DataFrame:  # function: the evidence script's traced instants of 2.6.12 and the label script's observed jamming-phase edges, long format
    tl = pd.read_csv(PROC / "ground_truth" / "early_warning_timeline.csv", dtype={"test_code": str})  # the evidence script's timeline (label-side evidence, not a feature)
    tl = tl[tl.test_code == SEALED]  # the 2.6.12 rows
    ev = {"gain_onset_section3": "gain onset (the receiver-status script)", "t2612_loss_of_tracking": "loss of tracking", "t2612_reacquisition": "reacquisition", "t2612_measurement_engine_deception": "measurement engine on shifted time",  # dict: column -> event name
          "t2612_navigation_time_deception": "navigation time shifted", "t2612_navigation_fix_lost": "navigation fix lost", "own_position_first_over_50m": "own position > 50 m from the site", "gain_recovery_section3": "gain recovery (the receiver-status script)"}  # dict literal continued
    rows = [{"receiver": r.receiver, "event": name, "utc": str(getattr(r, col)) if pd.notna(getattr(r, col)) else "", "source": "ground_truth/early_warning_timeline.csv (the evidence script)"} for r in tl.itertuples() for col, name in ev.items()]  # nested list comprehension: one row per receiver and event
    jp = pd.read_csv(PROC / "ground_truth" / "jam_phase_boundaries.csv", dtype={"test_code": str})  # the jamming-phase script's observed jamming phases
    for r in jp[jp.test_code == SEALED].itertuples():  # for loop over its 2.6.12 rows
        rows.append({"receiver": r.receiver, "event": "observed jamming phase (L1 gain step)", "utc": f"{r.start_utc if pd.notna(r.start_utc) else ''}-{r.end_utc if pd.notna(r.end_utc) else ''}", "source": f"ground_truth/jam_phase_boundaries.csv ({r.basis}); scheduled {r.sched_start_utc}-{r.sched_end_utc}"})  # one row
    return pd.DataFrame(rows)  # the table


def onset_sources() -> pd.DataFrame:  # function: every pooled interval and the receiver evidence that set its onset and end (labels/labels_evidence_final.csv)
    e = pd.read_csv(PROC / "labels" / "labels_evidence_final.csv", dtype={"test_code": str})  # the label script's evidence table (labels, not features)
    p = e[e.receiver == "pooled"].copy()  # the pooled intervals
    gainish = ("gain_onset_recovery", "gain_disturbed_run", "gain_short_dip", "observed_jamming_phase")  # tuple: evidence read from the front-end gain
    p["onset_from_gain"] = p.start_source.apply(lambda s: all(x.split(":")[1] in gainish for x in str(s).split(";")))  # lambda: every source of the onset is a gain quantity
    for rx in RXS:  # for loop over the receivers: does that receiver's own gain or position set the onset?
        p[f"onset_own_gain_{rx}"] = p.start_source.apply(lambda s, rx=rx: any(x.split(":")[0] == rx and x.split(":")[1] in gainish for x in str(s).split(";")))  # lambda with a default argument: this receiver's gain among the sources
        p[f"onset_own_position_{rx}"] = p.start_source.apply(lambda s, rx=rx: any(x == f"{rx}:own_position" for x in str(s).split(";")))  # this receiver's own position among the sources
    return p.drop(columns=["receiver", "source"]).reset_index(drop=True)  # the table


# ----------------------------------------------------------------------------------------------------------------------
# 4. the RQ4 study reference monitors: definitions and coverage on the position sessions
# ----------------------------------------------------------------------------------------------------------------------
def persistent_min(sec: np.ndarray, x: np.ndarray, p: int) -> np.ndarray:  # function: at each second, the smallest value over the last p consecutive seconds, NaN unless all p exist
    s = pd.Series(x, index=sec).reindex(range(int(sec.min()), int(sec.max()) + 1))  # a full one-second grid (missing seconds NaN)
    m = s.rolling(p, min_periods=p).min()  # rolling minimum over p seconds; NaN if any is missing
    return m.reindex(sec).to_numpy()  # back on the given seconds


def monitors() -> tuple:  # function: the definitions and the coverage table
    sys.path.insert(0, str(CODES))  # the evidence script's own_positions imports the helpers module from codes/
    D = load_module("gain_evidence_for_unseen_attack_phases", CODES / "1_ground_truth" / "04_gain_and_position_evidence.py")  # the evidence script, unchanged (its main() is not run)
    gt = pd.read_csv(PROC / "ground_truth" / "session_ground_truth.csv", dtype={"test_code": str}).set_index("test_code")  # the timeline script's sessions, as own_positions expects
    st = read_sealed(PROC / "ground_truth" / "receiver_status_timeseries.csv", usecols=["test_code", "receiver", "timestamp_utc", "band", "agc_gain_db", "agc_cnt"])  # the gains, 2.6.12 dropped
    steps = pd.read_csv(PROC / "rl_environment" / "steps.csv.gz", dtype={"session": str})  # the environment script's stream grid (position sessions only)
    eai = pd.read_csv(PROC / "ground_truth" / "effective_attack_interval.csv", dtype={"test_code": str})  # the receiver-status script's quiet references
    defs, cov = [], []  # the two tables
    for rx in RXS:  # for loop over the receivers
        q = eai[(eai.receiver == D.RX_STATUS[rx]) & eai.field.isin(["agc_gain_db", "agc_cnt"])].quiet_reference.unique()  # its quiet reference(s)
        # dict literals over several lines: the two definitions
        defs.append({"receiver": rx, "monitor": "gain fall", "available": True, "source": "receiver_status_timeseries.csv: " + ("u-blox L1 AGC count (agc_cnt)" if rx == "UBX_01" else "GPSL1/E1 main-antenna gain (agc_gain_db)"),
                     "statistic": "the evidence script gain_drop: median gain over the 90 s that ended 30 s ago minus the gain now (positive when the gain falls); the evidence script gain_series: last reading of each second",  # dict literal continued: the evidence script's statistic
                     "alarm": f"statistic >= threshold in each of the last {PERSIST} seconds (the evidence script's persistence; an alarm at the {PERSIST}th second, nothing from the future)", "unit": "counts" if rx == "UBX_01" else "dB",  # dict literal continued: the alarm rule
                     "deployable_point": "lowest threshold whose share of the training sessions' clean stream seconds in alarm is at most the share (the logistic regression's rule); 2.6.12: the five position sessions",  # dict literal continued: how the deployable point is chosen
                     "fixed_settings_on_curve": GAIN_FIXED[rx], "quiet_reference": ";".join(f"{v:g}" for v in q), "missing": "a second without a reading, or without 30 readings in the reference window, is silent"})  # dict literal continued
        # list append: one row describing this receiver's position-departure monitor, its source named first
        defs.append({"receiver": rx, "monitor": "position departure", "available": rx != "SBF_03", "source": {"SBF_02": "PVTGeodetic (the evidence script own_positions)", "SBF_03": "none: the mosaic-H outputs no computed position in any session", "UBX_01": "NAV-PVT on host time (the evidence script own_positions)"}[rx],
                     "statistic": "horizontal distance of the receiver's own first fix of each second from the session's site reference (the evidence script horiz_from_site); 2.6.5 from 12:34:57 (the evidence script capture_spans)",  # dict literal continued: the position statistic
                     "alarm": f"distance >= threshold in each of the last {PERSIST} seconds (the evidence script's 'held 10 s'; an alarm at the {PERSIST}th second)", "unit": "m",  # dict literal continued: the alarm rule
                     "deployable_point": "lowest threshold whose share of the training sessions' clean stream seconds in alarm is at most the share; 2.6.12: the five position sessions",  # dict literal continued: how the deployable point is chosen
                     "fixed_settings_on_curve": "50 m" if rx != "SBF_03" else "", "quiet_reference": "", "missing": "a second without a valid fix is silent"})  # dict literal continued
        for t in SESSIONS:  # for loop over the position sessions
            grid = steps[(steps.receiver == rx) & (steps.session == t) & steps.window]  # the stream grid: windowed seconds
            sec, lab = grid.sec.to_numpy(np.int64), grid.label.to_numpy()  # tuple assignment: seconds and labels
            g = D.gain_series(st, t, rx)  # 05d: the gain on a one-second grid
            dr = D.gain_drop(g)  # 05d: the gain fall statistic
            gx = dr.reindex(sec).to_numpy()  # on the stream grid
            gp = persistent_min(dr.index.to_numpy(np.int64), dr.to_numpy(float), PERSIST)  # held over 10 s, on the gain grid
            gp = pd.Series(gp, index=dr.index).reindex(sec).to_numpy()  # on the stream grid
            own = D.own_positions(gt, t, rx)  # 05d: the receiver's own first valid fix of each second (u-blox on host time)
            if rx == "SBF_03":  # the mosaic-H: no position monitor
                check(f"the mosaic-H has no own position fix ({t})", len(own) == 0, f"{len(own)} fixes")  # recorded
                px, pp = np.full(len(sec), np.nan), np.full(len(sec), np.nan)  # nothing to measure
            else:  # the X5 and u-blox 01
                a0, b0 = D.capture_spans(gt, t, rx)[0]  # the evidence script's usable span (2.6.5 from 12:34:57)
                own = own[(own.sec >= a0) & (own.sec < b0)]  # inside it
                dh = D.horiz_from_site(own.lat.to_numpy(), own.lon.to_numpy(), own.h.to_numpy(), D.site_xyz(D.M.SITE_OF[t])) if len(own) else np.array([])  # horizontal distance from the site
                ds = pd.Series(dh, index=own.sec.to_numpy(np.int64))  # by second
                px = ds.reindex(sec).to_numpy()  # on the stream grid
                pp = pd.Series(persistent_min(ds.index.to_numpy(), ds.to_numpy(), PERSIST), index=ds.index).reindex(sec).to_numpy() if len(ds) else np.full(len(sec), np.nan)  # held over 10 s
            cl = lab == "clean"  # clean seconds of the grid
            for mon, x, xp in (("gain fall", gx, gp), ("position departure", px, pp)):  # for loop over the two monitors
                c = xp[cl & np.isfinite(xp)]  # the held statistic on clean seconds where defined
                # dict literal over two lines: the coverage row
                cov.append({"receiver": rx, "session": t, "monitor": mon, "grid_seconds": len(sec), "clean_grid_seconds": int(cl.sum()), "statistic_defined_share": float(np.isfinite(x).mean()) if len(x) else np.nan, "held_defined_share": float(np.isfinite(xp).mean()) if len(xp) else np.nan,
                            "held_defined_clean": int(len(c)), "clean_held_p50": float(np.median(c)) if len(c) else np.nan, "clean_held_p99": float(np.percentile(c, 99)) if len(c) else np.nan, "clean_held_max": float(c.max()) if len(c) else np.nan})  # dict literal continued
    return pd.DataFrame(defs), pd.DataFrame(cov)  # both tables


# ----------------------------------------------------------------------------------------------------------------------
# 5. jobs and hours (the agent experiment's first run as the measured rate)
# ----------------------------------------------------------------------------------------------------------------------
def jobs() -> tuple:  # function: the job table and the agent experiment's measured rates
    log = (PROC / "rl_agents" / "training" / "logs" / "grid_run1.log").read_text(encoding="utf-8")  # the agent experiment's committed first-run log
    hours = ast.literal_eval(re.search(r"(\{'_prepare'.*?\}) hours", log).group(1))  # regular expression: the phase hours dict printed at the end
    sec = {k: [float(x) for x in re.findall(rf"^  {k}_\S+: ([0-9.]+) s$", log, re.M)] for k in ("inner", "outer")}  # dict comprehension: every job's seconds per step
    rate = {k: hours[f"_{k}"] * 3600 / len(v) for k, v in sec.items()}  # wall seconds per job with 3 workers
    pts = pd.read_csv(PROC / "rl_agents" / "evaluation" / "deployable_points.csv", dtype={"outer": str})  # the agent experiment's committed deployable points
    base_pts = int(((pts.receiver == "SBF_02") & ~pts.detector.str.startswith("agent")).sum())  # the X5's baseline points (5 folds x 4 detectors x 3 shares)
    mm = pd.read_csv(PROC / "rl_agents" / "training" / "models_manifest.csv")  # the frozen agents
    agents = int(mm.model_file.str.contains("_SBF_02_").sum())  # the X5's frozen agents
    cr = pd.read_csv(PROC / "rl_agents" / "evaluation" / "curve_runs.csv.gz", dtype={"outer": str}, usecols=["receiver"])  # the agent experiment's baseline curve runs
    curve = int((cr.receiver == "SBF_02").sum())  # the X5's
    nB_in = len(RXS) * len(SESSIONS) * 2 * 5 * 2  # Part B inner jobs: 3 receivers x 5 inner folds x 2 arms x 5 lambdas x 2 seeds
    nB_out = len(RXS) * 2 * 5 * 5  # Part B outer jobs: 3 receivers x 2 arms x 5 lambdas x 5 seeds
    rows = [  # list of dicts: one row per item
        {"part": "PART1", "item": "P rebuilt through the adopted-pipeline script (Part A, the X5's outer fits; Part B, three receivers' five-session fits)", "count": 1, "basis": f"the agent experiment _prepare {hours['_prepare']} h", "est_minutes": round(hours["_prepare"] * 60 * 2, 1)},  # twice the agent experiment's preparation
        {"part": "PART1", "item": "Part A stream runs at the X5's points: per target, baselines + frozen agents", "count": len(TARGETS) * (base_pts + agents), "basis": f"{base_pts} baseline points + {agents} frozen X5 agents per target; not timed in the agent experiment", "est_minutes": 30.0},  # an assumption, stated
        {"part": "PART1", "item": "Part A baseline curve runs (the X5's curve grid on each target)", "count": len(TARGETS) * curve, "basis": f"{curve} X5 curve runs in the agent experiment per target; included above", "est_minutes": 0.0},  # inside the 30 minutes
        {"part": "PART1", "item": "Part B inner jobs (3 receivers x 5 inner folds x 2 arms x 5 lambda x 2 seeds), 100,000 steps", "count": nB_in, "basis": f"the agent experiment inner {hours['_inner']} h / {len(sec['inner'])} jobs = {rate['inner']:.1f} s wall per job (median single job {np.median(sec['inner']):.1f} s)", "est_minutes": round(nB_in * rate["inner"] / 60, 1)},  # the rate times the count
        {"part": "PART1", "item": "Part B outer jobs (3 receivers x 2 arms x 5 lambda x 5 seeds), stopped at their budgets", "count": nB_out, "basis": f"the agent experiment outer {hours['_outer']} h / {len(sec['outer'])} jobs = {rate['outer']:.1f} s wall per job", "est_minutes": round(nB_out * rate["outer"] / 60, 1)},  # the rate times the count
        {"part": "PART1", "item": "from-scratch run again of a sample of Part B jobs (T3 analogue)", "count": 12, "basis": "12 jobs at the inner rate", "est_minutes": round(12 * rate["inner"] / 60, 1)},  # the sample
        {"part": "PART2", "item": "2.6.12: every detector once per receiver at its point (agents five seeds), both modes", "count": len(RXS) * (12 + 50), "basis": "12 baseline points + 50 agents per receiver", "est_minutes": 10.0},  # an assumption, stated
        {"part": "PART2", "item": "the RQ4 study monitors: 2 monitors x 3 receivers x 5 folds + 2.6.12 (no position monitor on the mosaic-H)", "count": 2 * 3 * 6 - 6, "basis": "threshold sweeps; seconds", "est_minutes": 5.0},  # an assumption, stated
        {"part": "PART2", "item": "the RQ4 study cost: 10,000 timed decisions x 6 detectors x 3 receivers, model and transform separately", "count": 6 * 3 * 2 * 10000, "basis": "single-row calls on one thread", "est_minutes": 10.0},  # an assumption, stated
    ]  # end of the list
    return pd.DataFrame(rows), rate, hours  # the table and the rates


# ----------------------------------------------------------------------------------------------------------------------
# the plan text
# ----------------------------------------------------------------------------------------------------------------------
def wrap(text: str, width: int = 140) -> str:  # function: wraps a paragraph to the report's width
    import textwrap  # standard library: line wrapping
    return "\n".join(textwrap.wrap(text, width=width))  # the wrapped text




def main() -> None:  # function: builds every table and the report
    CODE = codes_05e()  # the observables script's code table
    check("the observables script's code table: Galileo C1C on the Septentrio, C1X on the u-blox, the rest identical", CODE["SBF"]["E"] == "C1C" and CODE["UBX"]["E"] == "C1X" and all(CODE["SBF"][s] == CODE["UBX"][s] for s in "GRC"), str(CODE))  # recorded
    C, present, nd1 = columns_table(CODE)  # the column table
    RM, EV, CAV = renaming(CODE)  # the renaming map
    lab = pd.read_csv(PROC / "labels" / "labels_1hz_final.csv", dtype={"test_code": str}, usecols=["test_code", "receiver", "sec", "label", "phase"])  # the final labels, 2.6.12 included (a READ FIRST label file)
    pol = pd.read_csv(PROC / "labels" / "scoring_fold_policy_final.csv", dtype={"test_code": str})  # the scoring module's fold policy
    pol12 = pol[pol.test_code == SEALED].set_index("receiver")  # its 2.6.12 rows
    PH = phases(lab[lab.test_code == SEALED], pol12)  # the 2.6.12 phases and events
    cnt = PH.groupby("receiver")[["n_clean", "n_attack_jam_spoof", "n_attack_spoof", "n_uncertain", "n_excluded"]].sum()  # label counts per receiver
    check("2.6.12 label counts per receiver equal labels/scoring_fold_policy_final.csv", all(cnt.at[r, "n_clean"] == pol12.at[r, "clean_s"] and cnt.at[r, "n_attack_jam_spoof"] + cnt.at[r, "n_attack_spoof"] == pol12.at[r, "attack_s"] and cnt.at[r, "n_uncertain"] == pol12.at[r, "uncertain_s"] and cnt.at[r, "n_excluded"] == pol12.at[r, "excluded_s"] for r in RXS), cnt.to_dict("index").__repr__())  # recorded
    val = validate_inference(lab[lab.test_code != SEALED])  # the inference rule on the position sessions
    check("inferred windows and validity equal the real index and the environment script's steps on the five position sessions", val[0] == 0 and val[1] == 0, f"{val[0]} windows and {val[1]} validity values differ over {val[2]} receiver-seconds")  # recorded
    EVT = events_2612()  # the 2.6.12 phases and events
    rq = pd.Timestamp("2026-09-14 " + EVT[(EVT.receiver == "SBF_03") & (EVT.event == "reacquisition")].utc.iloc[0]).floor("s")  # the mosaic-H's reacquisition, floored to its second
    exp = (rq + pd.Timedelta(seconds=W - 1 + HIST)).strftime("%H:%M:%S")  # first window 9 s after it, first valid state 29 s after that
    got = PH[(PH.receiver == "SBF_03") & (PH.phase == "spoof")].first_decidable_attack_utc.iloc[0]  # the inferred first decidable attack second
    check("the mosaic-H's first decidable attack second = reacquisition + 9 s (window) + 29 s (history)", got == exp, f"inferred {got}, expected {exp}")  # recorded
    ON = onset_sources()  # the onset sources
    DEF, COV = monitors()  # the reference monitors
    J, rate, hours = jobs()  # the job counts
    check("no 2.6.12 row in any table written except the 2.6.12 tables", all(SEALED not in set(df[c].astype(str)) for df, c in ((C, "held_out"), (COV, "session"))), "columns and coverage")  # recorded
    write_csv(C, "unseen_attack_columns.csv")  # written
    write_csv(RM, "unseen_attack_renaming.csv")  # written
    write_csv(EV, "unseen_attack_renaming_evidence.csv")  # the same-satellite evidence
    write_csv(CAV, "unseen_attack_renaming_caveats.csv")  # the caveat medians per receiver
    write_csv(PH, "unseen_attack_2612_phases.csv")  # written
    write_csv(EVT, "unseen_attack_2612_events.csv")  # written
    write_csv(ON, "unseen_attack_onset_sources.csv")  # written
    write_csv(DEF, "inspection_monitors.csv")  # written
    write_csv(COV, "inspection_monitor_coverage.csv")  # written
    write_csv(J, "unseen_attack_jobs.csv")  # written
    write_csv(pd.DataFrame(CHECKS), "unseen_attack_checks.csv")  # written
    for c in CHECKS:  # for loop over the checks
        print(("PASS " if c["pass"] else "FAIL ") + c["check"] + " | " + c["detail"][:160])  # printed
    if not all(c["pass"] for c in CHECKS):  # if: a check failed
        sys.exit(1)  # exit status 1, so a failure cannot be missed


if __name__ == "__main__":  # guard: runs main() only when the file is executed
    main()  # runs main
