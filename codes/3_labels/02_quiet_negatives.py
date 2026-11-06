"""Label rules 1, 2 and 5 - the label guard, the quiet in-window seconds and the Stave stop, measured (nothing selected).

The label script (3_labels/01_labels.py) is imported unchanged; its inputs are rebuilt exactly as its main() builds them, and its label_all() makes the labels.
Two in-memory variants of the label script's build() are made by substituting source text, nothing else (the file on disk is untouched):
  EXCL   the two exclusion extensions (u-blox 01's invalid G30 in 2.6.2 to 13:02:41 + 30 s, the Septentrio G08 transient in 2.6.10
         to 14:44:43 + 30 s) read EXCL_GUARD_S instead of GUARD_S (the primary variant moves the label guard only);
  STAVE  the drive exclusion 12:27:22-12:34:57 in 2.6.5 is split into the two drives (12:27:22-12:27:55, 12:32:37-12:34:57), so the
         Stave stop 12:27:56-12:32:36 is labelled by the ordinary rule.
Gate G1: with GUARD_S = EXCL_GUARD_S = 30 and the Stave stop relabel off, the substituted build reproduces labels/labels_1hz.csv (label and exclusion reason)
on all 22,356 rows; both the guard-only variant and the "all three move" variant at 30 must do so.

label rule 1  guard 0, 10, 30, 60, in the guard-only variant and with all three moving ("all three"): clean seconds and scored
       clean windows at w = 10 per receiver and session; the scoring module's fold policy and the false-alarm-eligible clean minutes; seconds whose
       exclusion moved with the coupling; per pooled onset, the last clean second before it and the clean run ending there
       (counted: a clean run of at least 60 s ending within 300 s of the onset). The seconds the guard removes,
       R = clean at guard 0 and not clean at guard 30 (guard-only variant), are split by side: "post" when they lie at the start of
       a guard-0 clean run (they follow the boundary), "pre" when at its end (they precede it); the boundary second, its label, and
       the adjacent attack seconds (attack seconds within 30 s on the far side of the boundary) are recorded. Per family of the
       main-set base columns (orbit-free and self blocks; families of the forest script's rule), Cliff's delta of R against core clean (clean at
       guard 60) and against the adjacent attack seconds: median |delta| over the family's columns and the largest; per receiver
       pooled over sessions and per session where both groups hold at least 20 seconds. Descriptive only (autocorrelated seconds).
label rule 2  quiet seconds: in a scheduled window, labelled uncertain, covered by no evidence interval of any receiver, passing the receiver-status script's
       indicators evaluated inside the window, and at least 60 s from any attack, uncertain-by-evidence or excluded second of the same
       receiver (uncertain-by-evidence: uncertain outside the window, or inside 2.6.5's fixed uncertain interval after the attack). The indicators inside the
       window are the label script's combined rule with the label script's own references (indicator_mask), on a 0.1 s grid with readings held up to 60 s;
       gate G2: applied to the margins, the evaluator reproduces the label script's indicator_seconds() second for second. Rule (a) is the full
       combined rule, rule (b) its feature-band indicators; every indicator of the combined rule is a feature-band one
       (GPSL1/E1, GLOL1, B1 on the two Septentrio main antennas; u-blox L1 noise and CW suppression), so (b) equals (a), which is
       checked, not assumed. Quiet seconds are counted per receiver and session, located against two stretches per session (the
       earliest gain onset to the first deceived output; the first residual anomaly to the earliest gain onset), and compared with
       core clean and with attack seconds as in label rule 1.
label rule 5  the Stave stop: the Stave stop relabel labels (guard 30, guard-only variant) give the candidate clean seconds of 12:27:56-12:30:00; per receiver their
       count, scored clean windows (w = 10, from the Stave stop relabel labels), the concurrent feature-band tests over them (labels_1hz column), the
       first-minute behaviour after the drive (tracked count and GPS C/N0 median from stave_base_265.csv.gz), Cliff's delta of the
       orbit-free columns against core clean and against 2.6.5's attack seconds, and the scoring module's policy for 2.6.5 with those seconds.
       The in-window part of the stop (12:30:00-12:32:36) is evaluated by the in-window indicators and reported.
Writes Data/processed/design_study/labels/: labels_<variant>_g<guard>.csv.gz (and labels_stave_stop.csv.gz), labels_checks.csv, label_rule1_*.csv,
label_rule2_*.csv, label_rule5_*.csv.
"""
import gzip  # standard library: byte-stable gzip output
import importlib.util  # standard library: loads the project scripts as modules
import inspect  # standard library: the source text of the label script's build(), for the in-memory variants
import sys  # standard library: module registry, exit
from pathlib import Path  # standard library: paths

import numpy as np  # numpy: per-second arrays
import pandas as pd  # pandas: tables

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
PROC = PROJECT / "Data" / "processed"  # committed inputs
OUT = PROC / "design_study" / "labels"  # outputs
GUARDS = [0, 10, 30, 60]  # the guard values of label rule 1 (seconds)
KEYS = ["test_code", "receiver", "sec"]  # key columns
ATT = ("attack_jam_spoof", "attack_spoof")  # attack labels
MIN_N = 20  # a per-session comparison needs at least this many seconds in each group
CHECKS = []  # list of dicts: every check


def load(name: str, path: Path):  # function: loads a project file as a module, unchanged
    spec = importlib.util.spec_from_file_location(name, path)  # where to load from
    mod = importlib.util.module_from_spec(spec)  # an empty module
    sys.modules[name] = mod  # registered
    spec.loader.exec_module(mod)  # executed (main() is guarded)
    return mod  # the module


def check(name: str, ok: bool, detail: str) -> None:  # function: records and prints one check; a failed gate stops the script
    CHECKS.append({"check": name, "pass": bool(ok), "detail": detail})  # recorded
    print(("PASS " if ok else "FAIL ") + name + ": " + detail, flush=True)  # printed


def write_gz(df: pd.DataFrame, path: Path) -> None:  # function: gzip CSV with no name or time in the header (byte-stable)
    with open(path, "wb") as fh, gzip.GzipFile(filename="", mode="wb", fileobj=fh, mtime=0) as gz:  # with-block: file and stream
        gz.write(df.to_csv(index=False, lineterminator="\n").encode())  # the CSV text


def T(h: str) -> int:  # function: "HH:MM:SS" on 14 September 2026 -> unix seconds
    return int(pd.Timestamp(f"2026-09-14 {h}", tz="UTC").timestamp())  # whole seconds


# ---------------------------------------------------------------- the label script's inputs and the in-memory variants of build()

def setup():  # function: 06 loaded, its inputs rebuilt as its main() does, build() replaced by the variant with two switches
    L = load("labels_for_quiet_negatives", PROJECT / "codes" / "3_labels" / "01_labels_as_used_by_quiet_negatives.py")  # 06, unchanged on disk
    src = inspect.getsource(L.build)  # build()'s source text
    subs = [('excl[idx(w["cs"], T("13:02:41") + GUARD_S)] = "invalid_G30"', 'excl[idx(w["cs"], T("13:02:41") + EXCL_GUARD_S)] = "invalid_G30"'),  # tuple: guard-only switch, G30
            ('excl[idx(T("14:44:41"), T("14:44:43") + GUARD_S)] = "G08_transient"', 'excl[idx(T("14:44:41"), T("14:44:43") + EXCL_GUARD_S)] = "G08_transient"'),  # tuple: guard-only switch, G08
            ('excl[idx(T("12:27:22"), T("12:34:57"))] = "vehicle_motion"', 'excl[idx(T("12:27:22"), T("12:34:57") if not STAVE_STOP else T("12:27:55"))] = "vehicle_motion"\n        if STAVE_STOP: excl[idx(T("12:32:37"), T("12:34:57"))] = "vehicle_motion"')]  # tuple: the Stave stop relabel switch
    for a, b in subs:  # for loop over the three substitutions
        if src.count(a) != 1:  # each must occur exactly once
            sys.exit(f"STOP: 06 build() text not found once: {a}")  # stop
        src = src.replace(a, b)  # substituted
    L.EXCL_GUARD_S, L.STAVE_STOP = 30, False  # the switches' defaults (= committed behaviour)
    exec(compile(src, "06_labels.build (the design study variant)", "exec"), L.__dict__)  # build() redefined in the label script's own namespace
    d = L.load_inputs()  # the label script's inputs
    W = L.windows(d["gt"])  # windows
    J = L.jam_phases(d["ph"], d["ov"], W)  # jamming phases
    long = L.status_long(d["st"])  # status long table
    split = ("2.6.12", pd.Timestamp(f"{L.DATE} {L.MARGIN_CONTAMINATION_START}", tz="UTC"))  # the receiver-status script's possibly contaminated 2.6.12 margin split
    parts, refs = L.indicator_mask(long, d["gt"], split)  # the label script's indicator rule with its references
    ind = L.indicator_seconds(parts)  # passing / failing margin seconds
    gev, _ = L.gain_evidence(d, long, W, d["eai"])  # gain evidence
    pev, _ = L.position_evidence(d["gt"], W)  # position evidence
    tev = L.time_evidence(d["sp"])  # own-time evidence
    ev = pd.DataFrame(gev + pev + tev).sort_values(["test_code", "receiver", "start", "source"]).reset_index(drop=True)  # evidence, as main()
    return {"L": L, "d": d, "W": W, "J": J, "long": long, "split": split, "parts": parts, "refs": refs, "ind": ind, "ev": ev}  # dict


def labels(S: dict, guard: int, excl_guard: int, k7: bool = False) -> pd.DataFrame:  # function: the labels under one setting
    L = S["L"]  # 06
    L.GUARD_S, L.EXCL_GUARD_S, L.STAVE_STOP = guard, excl_guard, k7  # the switches
    d = S["d"]  # the label script's inputs
    if k7:  # the Stave stop relabel: the observables script's grid has no 2.6.5 row before 12:34:57 (its span starts there), so the stop's seconds would read "no observations"
        stb = pd.read_csv(OUT / "stave_base_265.csv.gz", usecols=KEYS + ["n_tracked"])  # extended orbit-free base (4_features/04_stave_stop_base.py)
        add = stb[stb.sec < T("12:34:57")].assign(has_observations=lambda x: (x.n_tracked > 0).astype(int))[KEYS + ["has_observations"]]  # the observables script's rule: any satellite tracked
        d = {**d, "ep": pd.concat([d["ep"], add], ignore_index=True)}  # dict unpacking: the grid with the stop's rows (in memory only)
    try:  # try/finally: the committed values are restored whatever happens
        lab, _ = L.label_all(d, S["ev"], S["W"], S["J"], S["ind"])  # the label script's labels
    finally:  # always
        L.GUARD_S, L.EXCL_GUARD_S, L.STAVE_STOP = 30, 30, False  # restored
    return lab  # per-second table (the label script's columns)


# ---------------------------------------------------------------- windows, policy, onsets

def windows(lab: pd.DataFrame, w: int = 10) -> pd.DataFrame:  # function: windows (the feature script's rule): w consecutive capture seconds, none excluded, labelled by the last
    out = []  # per pair
    for (t, rx), g in lab.groupby(["test_code", "receiver"], sort=False):  # pairs
        lb = g.label.to_numpy(object)  # labels in time order (06 builds every capture second)
        ex = np.convolve((lb == "excluded").astype(int), np.ones(w, int))[:len(lb)]  # excluded seconds in each window
        keep = (np.arange(len(lb)) >= w - 1) & (ex == 0)  # complete windows without an excluded second
        out.append(pd.DataFrame({"test_code": t, "receiver": rx, "sec": g.sec.to_numpy()[keep], "label": lb[keep]}))  # kept windows
    return pd.concat(out, ignore_index=True)  # all


def pooled_iv() -> dict:  # function: session -> list of pooled attack intervals (labels/labels_evidence.csv; the same under every guard)
    ev = pd.read_csv(PROC / "labels" / "labels_evidence.csv", dtype={"test_code": str})  # evidence table
    p = ev[ev.source == "pooled_interval"]  # pooled rows
    return {t: sorted(zip(g.start, g.end)) for t, g in p.groupby("test_code")}  # dict comprehension


def onset_stretches(lab: pd.DataFrame, piv: dict) -> pd.DataFrame:  # function: per receiver and onset, the clean run before it
    rows = []  # one per receiver and onset
    for (t, rx), g in lab.groupby(["test_code", "receiver"], sort=False):  # pairs
        sec, cl = g.sec.to_numpy(), (g.label == "clean").to_numpy()  # seconds, clean mask
        prev_end = -10**12  # end of the previous interval
        for k, (on, en) in enumerate(piv.get(t, []), start=1):  # each onset
            m = (sec > prev_end) & (sec < on) & cl  # clean seconds before the onset, after the previous interval
            if m.any():  # any
                last = int(sec[m].max())  # the last clean second
                i = int(np.flatnonzero(sec == last)[0])  # its position
                j = i  # walk back over the contiguous clean run
                while j > 0 and cl[j - 1] and sec[j - 1] == sec[j] - 1:  # previous second clean and adjacent
                    j -= 1  # one back
                run = i - j + 1  # run length
                gap = on - last  # seconds from the last clean second to the onset
            else:  # no clean second before the onset
                last, run, gap = None, 0, None  # nothing
            # list.append of a dict over two lines: the onset row
            rows.append({"test_code": t, "receiver": rx, "interval": k, "onset": on, "last_clean_sec": last, "gap_to_onset_s": gap,
                         "clean_run_s": run, "qualifies_60_within_300": bool(run >= 60 and gap is not None and gap <= 300)})  # one row
            prev_end = en  # the next onset looks after this interval
    return pd.DataFrame(rows)  # the table


# ---------------------------------------------------------------- the indicator inside the windows

def eval_span(S: dict, a: pd.Timestamp, b: pd.Timestamp, bands: set | None = None) -> tuple:  # function: (passing seconds, failing seconds) of the label script's rule over [a, b)
    sel = [k for k in S["refs"] if bands is None or k[1] in bands]  # indicators used (all for rule (a); the feature-band ones for (b))
    grid = pd.date_range(a, b, freq="100ms", inclusive="left")  # 0.1 s grid
    bad = np.zeros(len(grid), bool)  # any indicator contaminated
    long = S["long"]  # status long table
    for rx, band, fld in sel:  # for loop over indicators
        ref, q = S["refs"][(rx, band, fld)]  # the label script's reference and IQR
        g = long[(long.receiver == rx) & (long.band == band) & (long.field == fld)].sort_values("timestamp_utc")  # readings in time order, as indicator_mask
        gg = g[(g.timestamp_utc >= a - pd.Timedelta(seconds=60)) & (g.timestamp_utc < b)].set_index("timestamp_utc").value  # from 60 s before
        gg = gg[~gg.index.duplicated()]  # one reading per time (the first)
        v = gg.reindex(grid, method="ffill", tolerance=pd.Timedelta(seconds=60)).to_numpy() if len(gg) else np.full(len(grid), np.nan)  # held up to 60 s
        rule = "gain" if fld == "agc_gain_db" else "iqr"  # the label script's two rules
        bad |= ((v < ref - 3.0) if rule == "gain" else (v > ref + 3 * q)) & ~np.isnan(v)  # contaminated where covered
    sec = (grid.asi8 // 1_000_000_000).astype(np.int64)  # whole seconds
    bs = set(sec[bad].tolist())  # failing seconds
    return set(sec.tolist()) - bs, bs  # passing, failing


# ---------------------------------------------------------------- Cliff's delta by family

def family_table(base: pd.DataFrame, A: pd.DataFrame, B: pd.DataFrame, cols: list, fam: dict, ident: dict) -> list:  # function: per family, the delta summary of A against B
    from helpers import cliffs_delta  # the project's exact Cliff's delta (NaN dropped)
    a = base.merge(A[KEYS], on=KEYS)  # group A's base rows
    b = base.merge(B[KEYS], on=KEYS)  # group B's base rows
    rows = []  # one per family
    per = {c: cliffs_delta(a[c].to_numpy(float), b[c].to_numpy(float)) for c in cols}  # dict comprehension: delta per column
    for f in sorted(set(fam[c] for c in cols)):  # for loop over families
        cs = [c for c in cols if fam[c] == f and np.isfinite(per[c])]  # its columns with a defined delta
        if not cs:  # none
            continue  # skipped
        ab = {c: abs(per[c]) for c in cs}  # absolute deltas
        top = max(ab, key=ab.get)  # the column with the largest |delta|
        # list.append of a dict over two lines: the family row
        rows.append({**ident, "family": f, "n_A": len(a), "n_B": len(b), "columns": len(cs), "median_abs_delta": float(np.median(list(ab.values()))),
                     "max_abs_delta": ab[top], "max_column": top, "max_delta_signed": per[top]})  # the family's row
    return rows  # list of dicts


def compare(base, A, B, cols, fam, label: str) -> list:  # function: pooled per receiver and per session (both groups >= MIN_N)
    out = []  # rows
    for rx in sorted(set(A.receiver)):  # receivers
        a, b = A[A.receiver == rx], B[B.receiver == rx]  # this receiver's groups
        if len(a) and len(b):  # both present
            out += family_table(base, a, b, cols, fam, {"comparison": label, "receiver": rx, "session": "pooled"})  # pooled
        for t in sorted(set(a.test_code)):  # sessions
            at, bt = a[a.test_code == t], b[b.test_code == t]  # within the session
            if len(at) >= MIN_N and len(bt) >= MIN_N:  # enough seconds
                out += family_table(base, at, bt, cols, fam, {"comparison": label, "receiver": rx, "session": t})  # per session
    return out  # rows


def main() -> None:  # function: everything, in order
    OUT.mkdir(parents=True, exist_ok=True)  # output folder
    sys.path.insert(0, str(PROJECT / "codes" / "common"))  # the helpers module importable
    S = setup()  # 06 and its inputs
    L, W = S["L"], S["W"]  # 06 and windows
    committed = pd.read_csv(PROC / "labels" / "labels_1hz.csv", usecols=KEYS + ["label", "exclusion_reason", "in_scheduled_window", "evidence_receivers", "concurrent_feature_band_test", "indicator_pass"], dtype={"test_code": str}, keep_default_na=False)  # committed labels
    # ---- gate G1
    for nm, eg in (("guard only", 30), ("all three", 30)):  # both variants at 30
        lab = labels(S, 30, eg)  # labels
        m = committed.merge(lab[KEYS + ["label", "exclusion_reason"]], on=KEYS, suffixes=("", "_v"))  # aligned
        ok = len(m) == len(committed) == len(lab) and (m.label == m.label_v).all() and (m.exclusion_reason == m.exclusion_reason_v).all()  # equal
        check(f"G1 {nm} at guard 30 reproduces labels/labels_1hz.csv", bool(ok), f"{len(m)} rows; label differences {int((m.label != m.label_v).sum())}; exclusion differences {int((m.exclusion_reason != m.exclusion_reason_v).sum())}")  # gate
    if not all(c["pass"] for c in CHECKS):  # gate failed
        sys.exit("STOP: gate G1 failed")  # stop
    piv = pooled_iv()  # pooled intervals
    # ---- label rule 1: every guard, both variants
    labs, cnt_rows, ons = {}, [], []  # labels per variant; counts; onset rows
    for var, coupled in (("guard_only", False), ("all_three", True)):  # the two variants
        for g in GUARDS:  # guard values
            lab = labels(S, g, g if coupled else 30)  # labels
            labs[(var, g)] = lab  # kept
            write_gz(lab[KEYS + ["label", "phase", "exclusion_reason", "in_scheduled_window", "indicator_pass"]].astype({"in_scheduled_window": int, "indicator_pass": int}), OUT / f"labels_{var}_g{g}.csv.gz")  # written for the design-choice study
            win = windows(lab)  # windows
            for (t, rx), x in lab.groupby(["test_code", "receiver"], sort=False):  # pairs
                cs = int((x.label == "clean").sum())  # clean seconds
                wn = win[(win.test_code == t) & (win.receiver == rx)]  # its windows
                pol = load_policy()(t, cs)  # the scoring module's fold policy
                # dict literal over five lines: the pair's counts and policy
                rows = {"variant": var, "guard_s": g, "test_code": t, "receiver": rx, "clean_s": cs, "attack_s": int(x.label.isin(ATT).sum()),
                        "uncertain_s": int((x.label == "uncertain").sum()), "excluded_s": int((x.label == "excluded").sum()),  # dict entry: uncertain and excluded seconds of this session and receiver
                        # dict entry: clean and attack 10 s windows that are scored
                        "scored_clean_windows_w10": int((wn.label == "clean").sum()), "scored_attack_windows_w10": int(wn.label.isin(ATT).sum()),
                        "windows_w10": len(wn), "fa_policy": pol["false_alarms"], "precision": pol["precision"],  # dict entry: all 10 s windows, and the false-alarm and precision policy of the pair
                        "fa_eligible_clean_min": round(cs / 60, 2) if pol["false_alarms"] != "none" else 0.0}  # dict: the pair's counts
                cnt_rows.append(rows)  # kept
            o = onset_stretches(lab, piv)  # onset stretches
            ons.append(o.assign(variant=var, guard_s=g))  # kept
    cnt = pd.DataFrame(cnt_rows)  # counts
    base30 = labs[("all_three", 30)][KEYS + ["exclusion_reason"]]  # exclusion reasons at 30
    moved = []  # seconds whose exclusion moved with the coupling
    for g in GUARDS:  # guard values
        e = labs[("all_three", g)][KEYS + ["exclusion_reason"]].merge(base30, on=KEYS, suffixes=("", "_30"))  # aligned
        mv = e[e.exclusion_reason != e.exclusion_reason_30]  # moved
        moved.append({"guard_s": g, "seconds_exclusion_changed": len(mv), "by_reason": "; ".join(f"{k} {v}" for k, v in (mv.exclusion_reason + "|" + mv.exclusion_reason_30).value_counts().sort_index().items())})  # summary
    cnt.to_csv(OUT / "label_rule1_counts.csv", index=False, lineterminator="\n")  # written
    pd.concat(ons, ignore_index=True).to_csv(OUT / "label_rule1_onsets.csv", index=False, lineterminator="\n")  # written
    pd.DataFrame(moved).to_csv(OUT / "label_rule1_coupling.csv", index=False, lineterminator="\n")  # written
    # ---- label rule 1: the removed seconds and their deltas (guard-only variant)
    l0, l30, l60 = labs[("guard_only", 0)], labs[("guard_only", 30)], labs[("guard_only", 60)]  # three guard settings
    c0 = l0.label.to_numpy() == "clean"  # clean at 0
    c30 = (l30.label.to_numpy() == "clean")  # clean at 30
    R = l0[c0 & ~c30][KEYS].copy()  # removed seconds
    side, bnd, blab = [], [], []  # side, boundary second, boundary label
    lab0 = l0.set_index(KEYS).label  # labels at guard 0 by key
    for (t, rx), g in R.groupby(["test_code", "receiver"], sort=False):  # pairs
        x = l0[(l0.test_code == t) & (l0.receiver == rx)]  # the pair at guard 0
        sec, cl = x.sec.to_numpy(), (x.label == "clean").to_numpy()  # seconds and clean
        st = np.flatnonzero(cl & ~np.r_[False, cl[:-1]])  # starts of clean runs
        en = np.flatnonzero(cl & ~np.r_[cl[1:], False])  # ends of clean runs
        for s in g.sec.to_numpy():  # each removed second
            i = int(np.flatnonzero(sec == s)[0])  # its position
            k = int(np.searchsorted(st, i, side="right") - 1)  # the run it belongs to
            a_, b_ = st[k], en[k]  # run start and end
            if i - a_ <= b_ - i:  # nearer the run start: after the boundary
                side.append("post"); b0 = sec[a_] - 1  # two statements: side and the boundary second before the run
            else:  # nearer the end: before the boundary
                side.append("pre"); b0 = sec[b_] + 1  # the boundary second after the run
            bnd.append(int(b0))  # boundary
            blab.append(lab0.get((t, rx, int(b0)), "outside capture"))  # its label at guard 0
    R["side"], R["boundary_sec"], R["boundary_label"] = side, bnd, blab  # columns
    R.to_csv(OUT / "label_rule1_removed_seconds.csv", index=False, lineterminator="\n")  # written
    per_b = R.groupby(["test_code", "receiver", "side", "boundary_sec", "boundary_label"]).size().rename("removed_s").reset_index()  # per boundary
    per_b.to_csv(OUT / "label_rule1_removed_per_boundary.csv", index=False, lineterminator="\n")  # written
    check("label rule 1 removed seconds (clean at guard 0, not at 30), guard-only variant", True, f"{len(R)} receiver-seconds at {len(per_b)} boundaries; per boundary median {per_b.removed_s.median():.0f} s (min {per_b.removed_s.min()}, max {per_b.removed_s.max()}); decision about 30 s per boundary, 602 in all")  # reported
    adj = []  # adjacent attack seconds
    att_keys = set(map(tuple, l0[l0.label.isin(ATT)][KEYS].itertuples(index=False)))  # attack seconds (labels do not depend on the guard)
    for r in per_b.itertuples():  # each boundary
        rng_ = range(r.boundary_sec, r.boundary_sec + 30) if r.side == "pre" else range(r.boundary_sec - 29, r.boundary_sec + 1)  # the 30 s beyond the boundary
        adj += [(r.test_code, r.receiver, s, r.side) for s in rng_ if (r.test_code, r.receiver, s) in att_keys]  # attack seconds there
    ADJ = pd.DataFrame(adj, columns=KEYS + ["side"]).drop_duplicates()  # adjacent attack seconds
    CORE = l60[l60.label == "clean"][KEYS]  # core clean
    base = pd.read_csv(PROC / "features" / "features_base_1hz.csv.gz")  # committed base table
    lin = pd.read_csv(PROC / "features" / "features_base_lineage.csv")  # base dictionary
    cols = list(lin[lin.block.isin(["orbit-free", "self"])].column)  # main-set base columns
    FOREST = load("random_forest_for_quiet_negatives", PROJECT / "codes" / "5_baselines" / "02_random_forest.py")  # 10, for its family rule
    fam = {c: FOREST.family_of(c, c) for c in cols}  # family per base column
    dl = []  # delta rows
    for sd in ("pre", "post"):  # the two sides
        Rs = R[R.side == sd]  # removed seconds of the side
        dl += compare(base, Rs, CORE, cols, fam, f"removed ({sd}) vs core clean (guard 60)")  # against core clean
        dl += compare(base, Rs, ADJ[ADJ.side == sd], cols, fam, f"removed ({sd}) vs adjacent attack")  # against the adjacent attack
    pd.DataFrame(dl).to_csv(OUT / "label_rule1_removed_deltas.csv", index=False, lineterminator="\n")  # written
    # ---- label rule 2: the in-window indicator and the quiet seconds
    gt = S["d"]["gt"]  # sessions
    passd, faild = {}, {}  # gate G2: per session, the evaluator's margin pass and fail sets
    for p in S["parts"]:  # the label script's margin parts
        ps, fs = eval_span(S, p["a"], p["b"])  # evaluated
        passd.setdefault(p["test_code"], set()).update(ps); faild.setdefault(p["test_code"], set()).update(fs)  # two statements: union over parts
    ok2 = all((passd.get(t, set()) - faild.get(t, set())) == S["ind"].get(t, {"pass": set()})["pass"] and faild.get(t, set()) == S["ind"].get(t, {"fail": set()})["fail"] for t in L.SESSIONS)  # generator in all(): equal per session
    check("G2 in-window evaluator applied to the margins reproduces the label script's indicator_seconds() second for second", ok2, f"{sum(len(v) for v in passd.values())} margin seconds evaluated")  # gate
    if not ok2:  # gate failed
        sys.exit("STOP: gate G2 failed")  # stop
    FB = {"GPSL1/E1", "GLOL1", "B1", "L1"}  # feature-band names of the status bands
    win_pass = {}  # rule -> session -> passing in-window seconds
    for rule, bands in (("a_full_rule", None), ("b_feature_bands", FB)):  # the two rules of the decision
        win_pass[rule] = {}  # per session
        for r in gt.itertuples():  # sessions
            a = pd.Timestamp(max(W[r.test_code]["cs"], W[r.test_code]["ss"]), unit="s", tz="UTC")  # captured part of the window, start
            b = pd.Timestamp(min(W[r.test_code]["ce"], W[r.test_code]["se"]) + 1, unit="s", tz="UTC")  # end (exclusive)
            win_pass[rule][r.test_code] = eval_span(S, a, b, bands)[0] if b > a else set()  # passing seconds
    same_ab = all(win_pass["a_full_rule"][t] == win_pass["b_feature_bands"][t] for t in win_pass["a_full_rule"])  # rule (b) equals rule (a)?
    check("rule (b) feature-band indicators equals rule (a) the full combined rule inside the windows", same_ab, f"indicators in the combined rule: {sorted(S['refs'])}")  # reported
    l30k = l30.merge(committed[KEYS + ["evidence_receivers"]], on=KEYS)  # guard-30 labels with the evidence column
    p5 = [b_ for a_, b_ in piv.get("2.6.5", [])]  # 2.6.5 interval ends
    post54 = (l30k.test_code == "2.6.5") & (l30k.sec > max(p5)) & (l30k.sec <= T(L.POST_265_END))  # the 2.6.5 post-test interval
    unc_ev = (l30k.label == "uncertain") & (~l30k.in_scheduled_window.astype(bool) | post54)  # uncertain by evidence
    trig = l30k.label.isin(ATT) | unc_ev | (l30k.label == "excluded")  # seconds that keep quiet seconds 60 s away
    qrows = []  # quiet seconds
    for rule in win_pass:  # both rules
        for (t, rx), x in l30k.groupby(["test_code", "receiver"], sort=False):  # pairs
            xi = x.index  # row positions in l30k
            sec = x.sec.to_numpy()  # seconds
            tg = np.flatnonzero(trig.loc[xi].to_numpy())  # trigger positions
            near = np.zeros(len(x), bool)  # within 60 s of a trigger
            if len(tg):  # any
                pos = np.searchsorted(tg, np.arange(len(x)))  # nearest trigger after
                da = np.where(pos < len(tg), tg[np.minimum(pos, len(tg) - 1)] - np.arange(len(x)), 10**9)  # distance after
                db = np.where(pos > 0, np.arange(len(x)) - tg[np.maximum(pos - 1, 0)], 10**9)  # distance before
                near = np.minimum(da, db) < 60  # closer than 60 s
            # boolean array over two lines: the quiet rule
            q = (x.in_scheduled_window.astype(bool).to_numpy() & (x.label == "uncertain").to_numpy() & (x.evidence_receivers == "").to_numpy()
                 & np.isin(sec, list(win_pass[rule][t])) & ~near)  # the quiet rule
            qrows.append(pd.DataFrame({"rule": rule, "test_code": t, "receiver": rx, "sec": sec[q]}))  # kept
    Q = pd.concat(qrows, ignore_index=True)  # quiet seconds
    Q.to_csv(OUT / "label_rule2_quiet_seconds.csv", index=False, lineterminator="\n")  # written
    tl = pd.read_csv(PROC / "ground_truth" / "early_warning_timeline.csv", dtype={"test_code": str})  # 5.3's timeline
    st_rows = []  # the two stretches per session
    for t, g in tl.groupby("test_code"):  # sessions
        go = min((T(v[:8]) for v in g.gain_onset_section3.dropna()), default=None)  # earliest gain onset
        dec = [T(v[:8]) for v in g.own_position_first_over_50m.dropna()] + [T(v[:8]) for v in g.t2612_measurement_engine_deception.dropna() if isinstance(v, str)]  # deceived outputs
        fa = min((T(v[:8]) for v in g.first_anomaly.dropna()), default=None)  # first residual anomaly
        st_rows.append({"test_code": t, "gain_onset": go, "first_deceived_output": min(dec) if dec else None, "first_anomaly": fa})  # row
    ST = pd.DataFrame(st_rows).set_index("test_code")  # stretches
    qs = Q[Q.rule == "b_feature_bands"].copy()  # quiet seconds by the defining rule
    qs["in_gain_to_deceived"] = [bool(ST.at[t, "gain_onset"] is not None and pd.notna(ST.at[t, "first_deceived_output"]) and ST.at[t, "gain_onset"] <= s <= ST.at[t, "first_deceived_output"]) for t, s in zip(qs.test_code, qs.sec)]  # list comprehension: in stretch 1
    qs["in_anomaly_to_gain"] = [bool(pd.notna(ST.at[t, "first_anomaly"]) and ST.at[t, "gain_onset"] is not None and ST.at[t, "first_anomaly"] <= s < ST.at[t, "gain_onset"]) for t, s in zip(qs.test_code, qs.sec)]  # list comprehension: in stretch 2
    inwin = l30k[l30k.in_scheduled_window.astype(bool)]  # in-window seconds
    qc = []  # quiet counts
    for (t, rx), x in inwin.groupby(["test_code", "receiver"], sort=False):  # pairs
        y = qs[(qs.test_code == t) & (qs.receiver == rx)]  # quiet (rule b)
        ya = Q[(Q.rule == "a_full_rule") & (Q.test_code == t) & (Q.receiver == rx)]  # quiet (rule a)
        # list.append of a dict over four lines: the quiet counts
        qc.append({"test_code": t, "receiver": rx, "in_window_s": len(x), "in_window_uncertain_s": int((x.label == "uncertain").sum()),
                   "uncertain_no_evidence_s": int(((x.label == "uncertain") & (x.evidence_receivers == "")).sum()),  # dict entry: uncertain seconds that no receiver's evidence supports
                   "pass_in_window_indicators_s": int(np.isin(x.sec, list(win_pass["b_feature_bands"][t])).sum()),  # dict entry: seconds inside the window in which the indicators pass
                   # dict entry: quiet seconds under rules b and a, and quiet seconds between gain change and deception
                   "quiet_rule_b": len(y), "quiet_rule_a": len(ya), "quiet_in_gain_to_deceived": int(y.in_gain_to_deceived.sum()),
                   "quiet_in_anomaly_to_gain": int(y.in_anomaly_to_gain.sum()), "sealed_session": t == "2.6.12"})  # counts
    pd.DataFrame(qc).to_csv(OUT / "label_rule2_quiet_counts.csv", index=False, lineterminator="\n")  # written
    ST.reset_index().to_csv(OUT / "label_rule2_stretches.csv", index=False, lineterminator="\n")  # written
    # residual evidence inside the quiet seconds (a measurement for the strict quiet-second rule decision, residuals are features and never set a label)
    rs = pd.read_csv(PROC / "labels" / "labels_1hz_residual_sensitivity.csv", usecols=KEYS + ["label_residual_assisted"], dtype={"test_code": str})  # the residual-assisted labels
    kr = base[KEYS + ["known_G_res_maxabs", "known_E_res_maxabs", "known_C_res_maxabs"]].copy()  # known-antenna residual extremes per second
    kr["known_res_maxabs"] = kr[["known_G_res_maxabs", "known_E_res_maxabs", "known_C_res_maxabs"]].max(axis=1)  # the largest over the three constellations
    qe = qs.merge(kr[KEYS + ["known_res_maxabs"]], on=KEYS, how="left").merge(rs, on=KEYS, how="left")  # quiet seconds with both
    tl_rx = tl.set_index(["test_code", "receiver"])  # 5.3's timeline per receiver
    qe["in_own_anomaly_to_gain"] = [bool(isinstance(tl_rx.at[(t, r), "first_anomaly"], str) and ST.at[t, "gain_onset"] is not None and T(tl_rx.at[(t, r), "first_anomaly"][:8]) <= s_ < ST.at[t, "gain_onset"]) for t, r, s_ in zip(qe.test_code, qe.receiver, qe.sec)]  # list comprehension: the receiver's own first anomaly to the pooled gain onset
    # groupby with named aggregations over four lines: the residual evidence per pair
    qr = qe.groupby(["test_code", "receiver"]).agg(quiet_s=("sec", "size"), res_over_30m_s=("known_res_maxabs", lambda v: int((v > 30).sum())),
                                                   # keyword arguments of agg(): seconds with a residual over 100 m, and seconds with no residual
                                                   res_over_100m_s=("known_res_maxabs", lambda v: int((v > 100).sum())), res_missing_s=("known_res_maxabs", lambda v: int(v.isna().sum())),
                                                   # keyword argument of agg(): seconds labelled attack under the residual-assisted label rule
                                                   attack_under_06d_residual_rule_s=("label_residual_assisted", lambda v: int(v.isin(ATT).sum())),
                                                   in_own_anomaly_to_gain_s=("in_own_anomaly_to_gain", "sum")).reset_index()  # per pair
    qr.to_csv(OUT / "label_rule2_quiet_residual_evidence.csv", index=False, lineterminator="\n")  # written
    ATTK = l30[l30.label.isin(ATT)][KEYS]  # attack seconds
    qd = compare(base, qs[KEYS], CORE, cols, fam, "quiet vs core clean (guard 60)") + compare(base, qs[KEYS], ATTK, cols, fam, "quiet vs attack")  # deltas
    pd.DataFrame(qd).to_csv(OUT / "label_rule2_quiet_deltas.csv", index=False, lineterminator="\n")  # written
    # ---- label rule 5: the Stave stop
    k7 = labels(S, 30, 30, k7=True)  # the Stave stop relabel labels (guard 30, guard-only variant)
    write_gz(k7[KEYS + ["label", "phase", "exclusion_reason", "in_scheduled_window", "indicator_pass"]].astype({"in_scheduled_window": int, "indicator_pass": int}), OUT / "labels_stave_stop.csv.gz")  # written for the design-choice study
    diff = k7.merge(l30[KEYS + ["label"]], on=KEYS, suffixes=("", "_30"))  # against guard 30
    chg = diff[diff.label != diff.label_30]  # changed seconds
    check("the Stave stop relabel labels differ from guard 30, and only inside 2.6.5's Stave stop", len(chg) > 0 and bool(((chg.test_code == "2.6.5") & (chg.sec >= T("12:27:56")) & (chg.sec <= T("12:32:36"))).all()), f"{len(chg)} receiver-seconds changed: " + "; ".join(f"{a}->{b} {n}" for (a, b), n in chg.groupby(["label_30", "label"]).size().items()))  # check
    kw = windows(k7)  # the Stave stop relabel windows
    stb = pd.read_csv(OUT / "stave_base_265.csv.gz")  # extended 2.6.5 orbit-free base (4_features/04_stave_stop_base.py)
    fcols = [c for c in stb.columns if c not in KEYS]  # its orbit-free columns
    rows5, cand = [], []  # per receiver; candidate seconds
    for rx in L.RXS:  # receivers
        x = k7[(k7.test_code == "2.6.5") & (k7.receiver == rx) & (k7.sec >= T("12:27:56")) & (k7.sec <= T("12:30:00"))]  # the pre-window part of the stop
        cl = x[x.label == "clean"]  # candidate clean seconds
        cand.append(cl[KEYS])  # kept
        cm = committed[(committed.test_code == "2.6.5") & (committed.receiver == rx) & committed.sec.isin(cl.sec)]  # their committed columns
        wn = kw[(kw.test_code == "2.6.5") & (kw.receiver == rx) & kw.sec.isin(cl.sec)]  # their windows
        sb = stb[(stb.receiver == rx)]  # the stop's base values
        first = sb[(sb.sec >= T("12:27:56")) & (sb.sec < T("12:28:56"))]  # first minute after the drive
        later = sb[(sb.sec >= T("12:28:56")) & (sb.sec < T("12:30:00"))]  # the rest before the window
        y = k7[(k7.test_code == "2.6.5") & (k7.receiver == rx) & (k7.sec >= T("12:30:00")) & (k7.sec <= T("12:32:36"))]  # in-window part of the stop
        # list.append of a dict over eight lines: the Stave row of one receiver
        rows5.append({"receiver": rx, "candidate_clean_s": len(cl), "first_clean": L.hms(cl.sec.min()) if len(cl) else "", "last_clean": L.hms(cl.sec.max()) if len(cl) else "",
                      # dict entry: scored clean 10 s windows and the feature-band tests running at the same time
                      "scored_clean_windows_w10": int((wn.label == "clean").sum()), "concurrent_feature_band_tests": ";".join(sorted({v for s in cm.concurrent_feature_band_test for v in s.split(";") if v})),
                      # dict entry: seconds before the window in which the indicators pass, out of all seconds before it
                      "indicator_pass_pre_window_s": int(x.indicator_pass.sum()), "pre_window_s": len(x),
                      # dict entry: median tracked satellites in the first minute and later on
                      "n_tracked_first_minute_median": float(first.n_tracked.median()), "n_tracked_later_median": float(later.n_tracked.median()),
                      # dict entry: median GPS C/N0 in the first minute and later on
                      "G_cn0_median_first_minute": float(first.G_cn0_median.median()), "G_cn0_median_later": float(later.G_cn0_median.median()),
                      # dict entry: seconds of the stop inside the window, and those in which the indicators pass
                      "in_window_stop_s": len(y), "in_window_stop_pass_indicators_s": int(np.isin(y.sec, list(win_pass["b_feature_bands"]["2.6.5"])).sum()),
                      "policy_2_6_5_with_these": load_policy()("2.6.5", len(cl))["false_alarms"]})  # dict: the receiver's row
    pd.DataFrame(rows5).to_csv(OUT / "label_rule5_stave.csv", index=False, lineterminator="\n")  # written
    CAND = pd.concat(cand, ignore_index=True)  # candidate seconds
    basex = pd.concat([base[~((base.test_code == "2.6.5") & base.sec.isin(stb.sec))][KEYS + fcols], stb[KEYS + fcols]], ignore_index=True)  # committed base with the stop's rows added
    ATT5 = l30[(l30.test_code == "2.6.5") & l30.label.isin(ATT)][KEYS]  # 2.6.5 attack seconds
    sd_ = compare(basex, CAND, CORE, fcols, fam, "Stave candidates vs core clean (guard 60)") + compare(basex, CAND, ATT5, fcols, fam, "Stave candidates vs 2.6.5 attack")  # deltas
    pd.DataFrame(sd_).to_csv(OUT / "label_rule5_stave_deltas.csv", index=False, lineterminator="\n")  # written
    pd.DataFrame(CHECKS).to_csv(OUT / "labels_checks.csv", index=False, lineterminator="\n")  # checks
    print("done")  # end


def load_policy():  # function: the scoring module's fold_policy (the scoring module is a module, nothing runs on import)
    if "scoring_for_quiet_negatives" not in sys.modules:  # loaded once
        load("scoring_for_quiet_negatives", PROJECT / "codes" / "common" / "scoring.py")  # the scoring module
    return sys.modules["scoring_for_quiet_negatives"].fold_policy  # the function


if __name__ == "__main__":  # run as a script
    main()  # label rule 1, label rule 2, label rule 5
