#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
The automated feature tests. Part 2 of the feature build: the barred-column test (the BARRED list below;
the research plan, Research_Plan_Topu.pdf, section 2.4, keeps receiver status out of every feature).

It fails (exit status 1) if any barred column, or a column derived from one, reaches the base feature table:
  T1 names     no feature column carries a barred word (gain, status, jamming, authentication, position, velocity, distance, time, clock,
               label, navigation-message fields ...), and no column repeats a label-table column other than the three keys;
  T2 lineage   every feature column has a row in features_base_lineage.csv, every source named there is on the allowed list (the L1-band
               observation codes, the IGS broadcast file used causally, the atmosphere models, the receiver's own time differences, the GLONASS
               channel table of the RINEX headers, the self-solved position, the position script reference), the position script reference appears only in known_
               columns and the self-solved position only in self_ columns;
  T3 barred files unreadable   07_features.build_base() is run again with every file under the project folder made unreadable except an allowed list
               (the RINEX observation files, the residual script's tables, three processed tables, the 2.6.12 u-blox 01 binary and host-time files used only to
               place rows on the grid), the label table readable only for its three key columns and the session table only for its folder
               and capture times; the table it returns must equal the published one byte for byte;
  T4 the guard works   under the same guard, reading a barred file (receiver status, the label column, the schedule) must raise.
part 3:
  T5 no future     for every window length, every pair and three cut seconds per pair, every base value after the cut is moved by
               +1e6 (a missing one set to 1e6) and the window statistics are recomputed: no statistic of any window ending at or before
               the cut may change; positive control: moving only the cut second itself must change its window;
  T6 window files  every feature-file column is in features_window_columns.csv and carries no barred word, only the index files hold a
               label, geometry level statistics are outside the main sets and geometry change statistics inside.
part 4, the six leakage classes (report Data/processed/checks/screen_leakage_report.csv):
  T7 group         the folds are disjoint, 2.6.12 is in no training list, and a published screen cell is unchanged when the held-out
               session's and 2.6.12's windows are deleted altogether;
  T8 scaling       the robust scaler, Pearson pruning, mutual-information ranking and the screen give identical results when the held-out
               and 2.6.12 windows are moved by 1e6; control: moving one training session changes the scaler;
  T9 navigation    the residual script's orbits and residuals for one pair, recomputed with every receiver navigation file and every raw receiver binary
               unreadable, equal the published rows exactly (orbits come from the IGS file alone);
  T10 circularity  the circularity candidates are named in the dictionary and the screen summary, and their coupling with the label's
               own-fix evidence is reported; no label source in the lineage (T2) and none readable by the build (T3, T4).
  Time leakage is T5 with part 1's R4 and part 3's last-second labelling check; label leakage is T1-T4 and T6.

Reads   Data/processed/features/features_base_1hz.csv.gz, features_base_lineage.csv, labels/labels_1hz.csv (column names only), and whatever
        build_base() reads under the guard
Writes  Data/processed/checks/feature_tests.csv
"""

from __future__ import annotations  # future import: type hints kept as text (a convention across the project's scripts)

import builtins  # the built-in namespace, whose open() the guard replaces during T3 and T4
import gzip  # standard-library gzip, to read the published table back as text
import importlib.util  # standard-library module loader, for modules whose names start with a digit
import io  # standard-library io, whose open() pathlib uses (also replaced by the guard)
import sys  # standard-library system access: sys.path, sys.modules and the exit status
from pathlib import Path  # Path objects for building file paths

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/

import numpy as np  # NumPy: arrays
import pandas as pd  # pandas: tables

PROC = PROJECT / "Data" / "processed"  # processed tables
OUT = PROC / "features"  # this section's tables
CHECK_DIR = PROJECT / "Data" / "processed" / "checks"  # where the result goes
RESULTS: list = []  # list: one row per test


def load(name: str, filename: str):  # function: loads codes/<filename> as a module (names start with a digit)
    spec = importlib.util.spec_from_file_location(name, PROJECT / "codes" / filename)  # where to load from
    mod = importlib.util.module_from_spec(spec)  # an empty module object
    sys.modules[name] = mod  # registered, so the module's own imports resolve
    spec.loader.exec_module(mod)  # runs the file's top level (its main() is guarded and not run)
    return mod  # the module


BARRED_WORDS = {"agc", "gain", "rfstatus", "rf", "status", "quality", "qualityind", "jam", "jammed", "jamming", "cw", "cwsuppression", "noise", "auth", "authentic", "authentication", "osnma", "flag", "flags",  # set: receiver self-reports (exclusion rule)
                "pvt", "navpvt", "lat", "lon", "latitude", "longitude", "height", "alt", "velocity", "vel", "speed", "dist", "distance", "offset", "hoff", "voff", "ecef", "own", "departure",  # the receiver's position and velocity
                "time", "utc", "trx", "clock", "clk", "tow", "week",  # the receiver's time or clock state as such
                "label", "labels", "group", "provisional", "phase", "evidence", "scheduled", "window", "concurrent", "indicator", "exclusion", "reason",  # the label table
                "iod", "iode", "iodc", "iodnav", "aode", "aodc", "toe", "toc", "ttm", "af0", "af1", "af2", "tgd", "bgd", "ephemeris", "nav"}  # navigation-message content
ALLOWED_SOURCES = {"RINEX:C1C/C1X/C2I", "RINEX:L1C/L1X/L2I", "RINEX:D1C/D1X/D2I", "RINEX:S1C/S1X/S2I", "RINEX:LLI(carrier)", "RINEX:header GLONASS SLOT/FRQ",  # set: the decision-14 observations and the header's channel table
                   "receiver time (differences only)", "IGS broadcast (causal)", "Klobuchar+Saastamoinen", "self-solved position", "reference position"}  # own time differences, orbits, atmosphere, the two positions
KEYS = ["test_code", "receiver", "sec"]  # the key columns
GT_COLS = {"test_code", "session_folder", "capture_start_utc", "capture_end_utc"}  # the session-table columns the build may read
ALLOW_FILES = ["Data/rinex/*/*.crx.gz", "Data/processed/features/residuals_*.csv.gz", "Data/processed/features/epochs_*.csv.gz",  # list of glob patterns: files the build may open
               "Data/processed/ground_truth/session_ground_truth.csv", "Data/processed/ground_truth/glonass_channels.csv", "Data/processed/preprocessing/rinex_ubx01_2612_pieces.csv", "Data/processed/labels/labels_1hz.csv",  # three processed tables, and the grid (keys only, enforced below)
               "Data/Day1_14092026/20260914_184225_T_2.6.12/*_UBX_01.ubx", "Data/Day1_14092026/20260914_184225_T_2.6.12/host_timing/*_UBX_01.ubx.hosttime.csv",  # the 2.6.12 u-blox 01 binary and its host-time sidecar (grid placement only)
               "Data/Day1_14092026/20260914_184225_T_2.6.12/timestamps_system_capture.csv", "codes/*", "codes/.mplconfig/*", "codes/.mplconfig/*/*"]  # the laptop clock table; code and matplotlib's cache


def record(test: str, expected: str, found, ok: bool) -> None:  # function: records one test and prints it
    RESULTS.append({"test": test, "expected": expected, "found": found, "ok": bool(ok)})  # append a dict row
    print(f"  [{'ok' if ok else 'FAIL'}] {test}: expected {expected}; found {found}")  # progress line


def words(col: str) -> set:  # function: the lower-case words of a column name, split at underscores
    return {w.lower() for w in col.split("_")}  # set comprehension


def t1_names(cols: list, label_cols: list) -> None:  # function: T1
    feat = [c for c in cols if c not in KEYS]  # list comprehension: the feature columns
    hit = [c for c in feat if words(c) & BARRED_WORDS]  # list comprehension: columns with a barred word
    rep = [c for c in feat if c in label_cols]  # list comprehension: columns repeating a label-table column
    record("T1 names: no barred word, no label-table column", "0 and 0", f"{len(feat)} feature columns; barred words {hit}; label-table columns {rep}", not hit and not rep)  # the test


def t2_lineage(cols: list, lin: pd.DataFrame) -> None:  # function: T2
    feat = [c for c in cols if c not in KEYS]  # the feature columns
    missing = sorted(set(feat) - set(lin.column))  # columns without a lineage row
    extra = sorted(set(lin.column) - set(feat))  # lineage rows without a column
    bad = [(r.column, s) for r in lin.itertuples() for s in str(r.sources).split(";") if s not in ALLOWED_SOURCES]  # nested comprehension: sources off the allowed list
    ref_wrong = [r.column for r in lin.itertuples() if "reference position" in str(r.sources) and not r.column.startswith("known_")]  # the reference outside the known block
    self_wrong = [r.column for r in lin.itertuples() if "self-solved position" in str(r.sources) and not r.column.startswith("self_")]  # the self-solved position outside the self block
    free_orbit = [r.column for r in lin.itertuples() if r.block == "orbit-free" and ("IGS" in str(r.sources) or "position" in str(r.sources))]  # an orbit in the orbit-free block
    record("T2 lineage: every column has allowed sources only; the reference only in known_, the self position only in self_, no orbit in the orbit-free block", "all empty", f"missing {missing}; extra {extra}; off-list {bad[:5]}; reference elsewhere {ref_wrong[:5]}; self position elsewhere {self_wrong[:5]}; orbit in orbit-free {free_orbit[:5]}", not (missing or extra or bad or ref_wrong or self_wrong or free_orbit))  # the test


class Guard:  # class: makes every file under the project folder unreadable except the allowed list, while active
    def __init__(self):  # constructor
        self.blocked = []  # files a read was refused for
        self.opened = set()  # allowed files under the project folder that were opened (proof the reads pass through the guard)
        self.real_open, self.real_io_open, self.real_read_csv = builtins.open, io.open, pd.read_csv  # the real functions, kept to restore them

    def allowed(self, p: Path) -> bool:  # method: whether a path under the project folder may be opened
        rel = p.relative_to(PROJECT)  # path relative to the project folder
        return any(rel.match(pat) for pat in ALLOW_FILES)  # generator expression: matches an allowed pattern

    def check(self, file) -> None:  # method: refuses a barred path
        if isinstance(file, (str, Path)):  # if: a path (not an already-open file descriptor)
            p = Path(file).resolve()  # absolute path
            if PROJECT in p.parents and not self.allowed(p):  # if: inside the project folder and not allowed
                self.blocked.append(str(p.relative_to(PROJECT)))  # remembered
                raise PermissionError(f"barred source: {p.relative_to(PROJECT)}")  # refused
            if PROJECT in p.parents:  # if: an allowed file under the project folder
                self.opened.add(str(p.relative_to(PROJECT).parent if p.suffix == ".gz" and "rinex" in p.parts else p.relative_to(PROJECT)))  # remembered (RINEX by session folder)

    def __enter__(self):  # method: switches the guard on (the "with" statement calls it)
        def g_open(file, *a, **k):  # nested function: open() that checks the path first
            self.check(file)  # refuses barred paths
            return self.real_open(file, *a, **k)  # the real open
        def g_read_csv(path, *a, **k):  # nested function: read_csv that also limits the columns of the label and session tables
            name = Path(path).name if isinstance(path, (str, Path)) else ""  # the file name
            cols = set(k.get("usecols") or [])  # the columns asked for
            if name == "labels_1hz.csv" and (not cols or not cols <= set(KEYS)):  # if: the label table read beyond its keys
                self.blocked.append("labels/labels_1hz.csv beyond the keys")  # remembered
                raise PermissionError("barred source: labels/labels_1hz.csv beyond test_code, receiver, sec")  # refused
            if name == "session_ground_truth.csv" and (not cols or not cols <= GT_COLS):  # if: the session table read beyond folders and capture times
                self.blocked.append("ground_truth/session_ground_truth.csv beyond folders and capture times")  # remembered
                raise PermissionError("barred source: ground_truth/session_ground_truth.csv schedule columns")  # refused
            self.check(path)  # the path rule as well
            return self.real_read_csv(path, *a, **k)  # the real read_csv
        builtins.open, io.open, pd.read_csv = g_open, g_open, g_read_csv  # switched on
        return self  # the guard

    def __exit__(self, *exc):  # method: switches the guard off, also after an error
        builtins.open, io.open, pd.read_csv = self.real_open, self.real_io_open, self.real_read_csv  # restored
        return False  # an error inside is not swallowed


def t3_unreadable(published_text: str) -> None:  # function: T3, the build under the guard returns the published table byte for byte
    F = load("window_features_for_tests", "4_features/03_window_features.py")  # the feature script
    with Guard() as g:  # the guard on
        gt = pd.read_csv(PROC / "ground_truth" / "session_ground_truth.csv", usecols=F.GT_COLS).set_index("test_code")  # the session table, allowed columns only
        base, _ = F.build_base(gt)  # the whole build
    text = base.round(4).to_csv(index=False)  # its CSV text, as write_base writes it
    record("T3 build with every barred file unreadable", "runs; equals the published table byte for byte; nothing refused", f"refused {g.blocked}; equal {text == published_text}; {len(base)} rows, {base.shape[1]} columns; allowed files opened through the guard: {sorted(g.opened)}", bool(not g.blocked and text == published_text))  # the test


def t4_guard_works() -> None:  # function: T4, the guard does refuse barred reads (so T3 can fail)
    tries = {"receiver status": lambda: pd.read_csv(PROC / "ground_truth" / "receiver_status_timeseries.csv", nrows=1),  # dict of lambdas: a receiver self-report
             "label column": lambda: pd.read_csv(PROC / "labels" / "labels_1hz.csv", usecols=["test_code", "receiver", "sec", "label"], nrows=1),  # the label itself
             "schedule": lambda: pd.read_csv(PROC / "ground_truth" / "session_ground_truth.csv", nrows=1),  # the schedule columns
             "receiver navigation file": lambda: open(PROJECT / "Data" / "rinex" / "20260914_155653_T_2.6.2" / "20260914_155653_SBF_02.nav").read(10)}  # navigation-message content
    refused = []  # tries that raised
    with Guard():  # the guard on
        for name, f in tries.items():  # for loop over the tries
            try:  # try: each must raise
                f()  # the read
            except PermissionError:  # except: refused, as it should be
                refused.append(name)  # counted
    record("T4 the guard refuses barred reads", f"{len(tries)} of {len(tries)} refused", f"{len(refused)} refused: {refused}", len(refused) == len(tries))  # the test


def t5_no_future() -> None:  # function: T5, a window never uses a second after its decision second
    F = load("window_features_for_tests_t5", "4_features/03_window_features.py")  # the feature script (its window statistics)
    base = pd.read_csv(OUT / "features_base_1hz.csv.gz")  # the published base table
    feat = [c for c in base.columns if c not in KEYS]  # feature columns
    changed, controls, compared = 0, 0, 0  # statistics that changed (must stay 0); control windows that changed; values compared
    for w in F.WINDOWS:  # for loop over the window lengths
        for (test, rx), g in base.groupby(["test_code", "receiver"], sort=False):  # for loop over the pairs
            X = g[feat].to_numpy(dtype=float)  # the pair's base values, time order
            ref = F.win_stats(X, w)  # unperturbed statistics
            for q in (0.25, 0.5, 0.75):  # for loop over three cut positions
                cut = int(q * (len(X) - 1))  # the cut row
                Y = X.copy()  # a copy
                fut = Y[cut + 1:]  # every row after the cut (a view into Y)
                fut[:] = np.where(np.isnan(fut), 1e6, fut + 1e6)  # moved by a large value, missing ones filled
                pert = F.win_stats(Y, w)  # recomputed
                for st in ref:  # for loop over the statistics
                    a, b = ref[st][:cut + 1], pert[st][:cut + 1]  # windows ending at or before the cut
                    changed += int((~((a == b) | (np.isnan(a) & np.isnan(b)))).sum())  # any difference, NaN equal to NaN
                    compared += a.size  # values compared
                Z = X.copy()  # control: only the cut row moved
                Z[cut] = np.where(np.isnan(Z[cut]), 1e6, Z[cut] + 1e6)  # the decision second itself
                ctl = F.win_stats(Z, w)  # recomputed
                controls += int(any((~((ref[st][cut] == ctl[st][cut]) | (np.isnan(ref[st][cut]) & np.isnan(ctl[st][cut])))).any() for st in ref))  # generator expression: the window at the cut changed
    n_cuts = 5 * 18 * 3  # windows lengths x pairs x cuts
    record("T5 no future: base values after the cut moved by 1e6; statistics of windows ending at or before the cut", f"0 changed of every value compared; control changes in {n_cuts} of {n_cuts}", f"{changed} changed of {compared:,}; control changed in {controls} of {n_cuts}", changed == 0 and controls == n_cuts)  # the test


def t6_window_files() -> None:  # function: T6, the window files' columns against the dictionary, barred words, labels and set membership
    wc = pd.read_csv(OUT / "features_window_columns.csv")  # the window dictionary
    base_cols = set(pd.read_csv(OUT / "features_base_lineage.csv").column)  # base columns
    problems = []  # what is wrong
    for f in sorted(OUT.glob("w*/*.csv.gz")):  # for loop over every window file
        w = int(f.parent.name[1:])  # its window length
        with gzip.open(f, "rt") as fh:  # opened as text
            cols = fh.readline().strip().split(",")  # its header only
        part = f.name.split("_", 2)[2].replace(".csv.gz", "")  # index, orbitfree, geometry, self or known
        if part == "index":  # if: the index file
            if cols != KEYS + ["label", "scored"]:  # if: anything but keys, label and mask
                problems.append(f"{f.name}: index columns {cols}")  # recorded
            continue  # next file
        feat = cols[len(KEYS):]  # feature columns
        want = list(wc[(wc.w == w) & (wc.part == part)].column)  # what the dictionary says
        if cols[:len(KEYS)] != KEYS or feat != want:  # if: keys or columns differ
            problems.append(f"{f.parent.name}/{f.name}: columns differ from the dictionary")  # recorded
        bad = [c for c in feat if words(c) & BARRED_WORDS]  # list comprehension: barred words
        if bad:  # if: any
            problems.append(f"{f.name}: barred words {bad[:3]}")  # recorded
    geo = wc[wc.part == "geometry"]  # geometry window columns
    lvl_in_main = int(((geo.stat.isin(["std", "skew", "kurt"]) == False) & ((geo.main == 1) | (geo.no_residual == 1) | (geo.known_antenna == 1))).sum())  # level statistics in a main set (must be 0)
    chg_out = int((geo.stat.isin(["std", "skew", "kurt"]) & (geo.main == 0) & (geo.known_antenna == 0) & (geo.no_residual == 0)).sum())  # change statistics left out of every main set (must be 0)
    orphan = sorted(set(wc.base_column) - base_cols)  # window columns without a base column
    record("T6 window files: columns as the dictionary, no barred word, a label only in the index files, geometry levels out of the main sets and changes in", "no problem", f"{len(list(OUT.glob('w*/*.csv.gz')))} files; problems {problems or 'none'}; geometry level statistics in a main set {lvl_in_main}; geometry change statistics outside {chg_out}; orphans {orphan}", not problems and lvl_in_main == 0 and chg_out == 0 and not orphan)  # the test


def t7_group() -> None:  # function: T7 group leakage
    S = load("jamming_screen_for_tests", "4_features/05_jamming_screen.py")  # the screen script
    fo = S.folds()  # the folds
    ok_f = all(h not in t and S.TRANSFER not in t and len(set(t)) == 4 and set(t) | {h} == set(S.POSITION_SESSIONS) for h, t in fo)  # generator expression: each fold is one held-out session plus the other four
    sc = pd.read_csv(OUT / "screen.csv.gz", keep_default_na=False, na_values=[""])  # the published screen
    idx, X = S.load_windows(10, "SBF_02")  # one receiver at w = 10
    held, train = fo[1]  # the 2.6.2 fold
    keep = idx.test_code.isin(train).to_numpy()  # only the training sessions' windows
    cell = S.screen_cell(idx[keep].reset_index(drop=True), X[keep].reset_index(drop=True), train)  # the cell without any held-out or 2.6.12 window
    pub = sc[(sc.w == 10) & (sc.receiver == "SBF_02") & (sc.held_out == held)].drop(columns=["w", "receiver", "held_out"]).reset_index(drop=True)  # the published cell
    num = [c for c in cell.columns if c not in ("column", "reason", "reason_no_floor")]  # numeric columns
    same = bool((cell.column == pub.column).all() and np.allclose(cell[num].to_numpy(float), pub[num].to_numpy(float), atol=1e-6, equal_nan=True) and all((cell[r].fillna("") == pub[r].fillna("")).all() for r in ("reason", "reason_no_floor")))  # same columns, values to the published 6 decimals, same reasons under both rules
    record("T7 group: folds disjoint, 2.6.12 never trained, a screen cell unchanged with the held-out and 2.6.12 windows deleted", "true and true", f"folds {ok_f}; cell (w10, X5, held-out 2.6.2) unchanged {same}", ok_f and same)  # the test


def t8_scaling() -> None:  # function: T8 scaling leakage: fold-fitted quantities do not depend on held-out windows
    S = load("jamming_screen_for_tests_t8", "4_features/05_jamming_screen.py")  # the screen script and its fold-index functions
    idx, X = S.load_windows(10, "SBF_02")  # one receiver at w = 10
    held, train = S.folds()[1]  # the 2.6.2 fold
    tr = idx.test_code.isin(train).to_numpy()  # training windows
    sc_rows = tr & idx.label.isin(["clean", "attack_jam_spoof", "attack_spoof"]).to_numpy()  # scored training windows
    wc = pd.read_csv(OUT / "features_window_columns.csv")  # dictionary
    cols = list(wc[(wc.w == 10) & (wc.main == 1)].column)[:200]  # 200 main-set columns (pruning and mutual information are slow on all)
    V = X[cols].to_numpy(dtype=float)  # values
    P = V.copy()  # perturbed copy
    P[~tr] = np.where(np.isnan(P[~tr]), 1e6, P[~tr] + 1e6)  # held-out and 2.6.12 windows moved
    y = idx.label.isin(["attack_jam_spoof", "attack_spoof"]).to_numpy().astype(int)  # attack against clean (training rows only are used)
    r1 = (S.fit_robust_scaler(V, np.flatnonzero(tr)), S.pearson_prune(V, np.flatnonzero(tr)), S.mi_rank(V, y, np.flatnonzero(sc_rows)))  # fitted on the original
    r2 = (S.fit_robust_scaler(P, np.flatnonzero(tr)), S.pearson_prune(P, np.flatnonzero(tr)), S.mi_rank(P, y, np.flatnonzero(sc_rows)))  # fitted with the held-out moved
    same_sc = all(np.array_equal(a, b, equal_nan=True) for a, b in zip(r1[0], r2[0]))  # generator expression: scaler parameters identical
    same_pp, same_mi = r1[1] == r2[1], r1[2] == r2[2]  # pruning and ranking identical
    c1 = S.screen_cell(idx, X, train)  # the screen
    XP = X.copy()  # perturbed windows
    XP.loc[~tr] = np.where(np.isnan(XP.loc[~tr].to_numpy(float)), 1e6, XP.loc[~tr].to_numpy(float) + 1e6)  # every held-out and 2.6.12 window moved
    c2 = S.screen_cell(idx, XP, train)  # the screen again
    same_scr = bool(np.array_equal(c1.select_dtypes("number").to_numpy(float), c2.select_dtypes("number").to_numpy(float), equal_nan=True) and (c1.reason == c2.reason).all() and (c1.reason_no_floor == c2.reason_no_floor).all())  # identical, under both rules
    Q = V.copy()  # control: one training session moved
    q = (idx.test_code == train[0]).to_numpy()  # its windows
    Q[q] = Q[q] + 1e6  # moved
    ctl = not np.array_equal(S.fit_robust_scaler(Q, np.flatnonzero(tr))[0], r1[0][0], equal_nan=True)  # the scaler must notice
    record("T8 scaling: scaler, Pearson pruning, mutual information and the screen fitted on training windows ignore held-out and 2.6.12 values", "all identical; control changes", f"scaler {same_sc}, pruning {same_pp} ({len(r1[1])} kept of 200), mutual information {same_mi}, screen {same_scr}; control (a training session moved) changes the scaler {ctl}", same_sc and same_pp and same_mi and same_scr and ctl)  # the test


class NavGuard(Guard):  # class: the guard of T3 turned around: everything readable except the receivers' navigation files and raw binaries
    def allowed(self, p: Path) -> bool:  # method: overrides the allowed list
        rel = p.relative_to(PROJECT)  # path relative to the project folder
        return not (p.suffix == ".nav" or rel.parts[:2] == ("Data", "Day1_14092026") or "receiver_status" in p.name)  # navigation files, raw captures and receiver status are refused


def t9_navigation() -> None:  # function: T9 navigation-message leakage: the residual engine's orbits come from the IGS file alone
    A = load("residuals_for_tests", "4_features/02_residuals.py")  # the engine
    gt = pd.read_csv(PROC / "ground_truth" / "session_ground_truth.csv").set_index("test_code")  # session folders
    test, rx = "2.6.2", "SBF_02"  # one pair (its RINEX needs no raw binary)
    with NavGuard() as g:  # navigation files and binaries unreadable
        kl = A.M.igs_klobuchar(A.S04.IGS_BRDC)  # Klobuchar from the IGS header
        obs = A.read_obs(gt, test, rx)  # observations
        S_ = A.sat_states(obs, A.Ephem(True, 0.0), A.GloCausal(True))  # orbits under the causal rule
        site = A.D.site_xyz(A.M.SITE_OF[test])  # the reference (known variant)
        got = {v: A.baseline(A.run_variant(v, obs, S_, site, kl)[0]) for v in A.VARIANTS}  # dict comprehension: both variants
    diff = 0  # rows that differ
    for v in A.VARIANTS:  # for loop over the variants
        pub = pd.read_csv(OUT / f"residuals_{v}.csv.gz", keep_default_na=False, na_values=[""])  # the published table
        pub = pub[(pub.test_code == test) & (pub.receiver == rx)].reset_index(drop=True)  # the pair
        mine = got[v].drop(columns="t_rx").round({"el": 4, "az": 4, "ex": 7, "ey": 7, "ez": 7, "rr": 4, "z": 4, "clk": 4, "res": 4, "base": 4, "res_db": 4})  # rounded as the residual script writes
        for a, b in (("el", "el_deg"), ("rr", "rr_m_s"), ("z", "z_m"), ("res", "res_m"), ("res_db", "res_db_m")):  # for loop over compared columns
            x, y = mine[a].to_numpy(float), pub[b].to_numpy(float)  # the two
            diff += int((~((x == y) | (np.isnan(x) & np.isnan(y)))).sum())  # differences
    record("T9 navigation message: the residual script recomputed for 2.6.2 X5 with every .nav file, raw binary and receiver-status file unreadable", "no read refused; 0 differences from the published rows", f"refused {g.blocked or 'nothing'}; differences {diff} over both variants", not g.blocked and diff == 0)  # the test


def t10_circularity() -> None:  # function: T10 label-feature circularity: the candidates are named, measured and kept out of the label sources
    wc = pd.read_csv(OUT / "features_window_columns.csv")  # dictionary
    summ = pd.read_csv(CHECK_DIR / "screen_screen_summary.csv")  # screen summary
    cp = pd.read_csv(CHECK_DIR / "screen_coupling.csv")  # coupling report
    named = set(wc[wc.base_column.isin(["self_G_n_res", "self_R_n_res", "self_E_n_res", "self_C_n_res", "self_solved", "self_pos_held"])].base_column)  # candidates present in the dictionary
    m = summ[summ.set == "main"]  # main-set cells (the no-residual and known-antenna sets hold no candidate by construction)
    in_summary = bool((m.n_circularity > 0).all() and (summ[summ.set != "main"].n_circularity == 0).all())  # every main-set cell counts them, and no other set holds one
    t = cp[cp.own_fix_evidence]  # seconds with own-fix evidence
    f = cp[~cp.own_fix_evidence]  # seconds without
    gap = f"{t.unsolved_share.min():.3f}-{t.unsolved_share.max():.3f} unsolved with own-fix evidence against {f.unsolved_share.min():.3f}-{f.unsolved_share.max():.3f} without"  # the coupling
    lin = pd.read_csv(OUT / "features_base_lineage.csv")  # base lineage
    label_src = [s_ for s_ in set(";".join(lin.sources).split(";")) if any(k in s_.lower() for k in ("gain", "status", "schedule", "pvt", "label", "flag"))]  # list comprehension: a label source named as a feature source (must be none)
    record("T10 circularity: candidates named (6), counted in every main-set cell and absent from the other sets, coupling with own-fix evidence reported, no label source in any lineage", "6; true; reported; none", f"{len(named)}; {in_summary}; {gap}; label sources {label_src or 'none'}", len(named) == 6 and in_summary and not label_src)  # the test


def leakage_report() -> None:  # function: the six leakage classes, each with the tests that prove it
    residual_checks = pd.read_csv(CHECK_DIR / "residuals_checks.csv")  # part 1 checks
    base_checks = pd.read_csv(CHECK_DIR / "base_checks.csv")  # part 2 and part 3 checks (last-second labelling)
    r4 = residual_checks[residual_checks.check.str.startswith("R4 causality:")].iloc[0]  # the causality test
    lab = base_checks[base_checks.check.str.startswith("Each window labelled by its decision")].iloc[0]  # the labelling test
    res = {r["test"].split(" ")[0]: r for r in RESULTS}  # dict comprehension: this run's tests by id
    rows = [("group", "windows of one session on both sides of a split", "T7", res["T7"]["ok"]),  # list of tuples: class, threat, proof, outcome
            ("time", "a window or an orbit record reaching beyond the decision second", "T5 (windows); part 1 R4 (causal records and baseline); part 3 last-second labelling", res["T5"]["ok"] and str(r4.ok) == "True" and str(lab.ok) == "True"),  # time
            ("label", "a label source or a column derived from one among the features", "T1-T4, T6", all(res[k]["ok"] for k in ("T1", "T2", "T3", "T4", "T6"))),  # label
            ("scaling", "normalisation or selection fitted beyond the training sessions", "T8", res["T8"]["ok"]),  # scaling
            ("navigation message", "decoded navigation content (IOD, ephemeris) reaching a feature", "T1/T6 (no navigation word), T4 (the build cannot read a .nav file), T9 (orbits from the IGS file alone)", res["T9"]["ok"] and res["T1"]["ok"] and res["T4"]["ok"]),  # navigation
            ("label-feature circularity", "one physical event setting a label and serving as a feature", "T10, T2, T3; candidates reported with and without", res["T10"]["ok"])]  # circularity
    pd.DataFrame(rows, columns=["class", "threat", "proof", "pass"]).to_csv(CHECK_DIR / "screen_leakage_report.csv", index=False)  # written
    record("Leakage report: the six leakage classes", "6 of 6 pass", f"{sum(bool(r[3]) for r in rows)} of 6", all(bool(r[3]) for r in rows))  # summary


def main() -> None:  # function: runs the tests and exits with status 1 on any failure
    with gzip.open(OUT / "features_base_1hz.csv.gz", "rt") as fh:  # the published table, as text
        published = fh.read()  # its CSV text
    cols = published.split("\n", 1)[0].split(",")  # its header
    label_cols = list(pd.read_csv(PROC / "labels" / "labels_1hz.csv", nrows=0).columns)  # the label table's column names only (no rows)
    lin = pd.read_csv(OUT / "features_base_lineage.csv")  # the data dictionary
    t1_names(cols, label_cols)  # T1
    t2_lineage(cols, lin)  # T2
    t4_guard_works()  # T4 first, so a broken guard is known before T3 relies on it
    t3_unreadable(published)  # T3
    t5_no_future()  # T5
    t6_window_files()  # T6
    t7_group()  # T7
    t8_scaling()  # T8
    t9_navigation()  # T9
    t10_circularity()  # T10
    leakage_report()  # the six classes
    pd.DataFrame(RESULTS).to_csv(CHECK_DIR / "feature_tests.csv", index=False)  # written
    sys.exit(0 if all(r["ok"] for r in RESULTS) else 1)  # exit status 1 if any test failed


if __name__ == "__main__":  # guard: runs main() only when the file is executed, not when imported
    main()  # entry point
