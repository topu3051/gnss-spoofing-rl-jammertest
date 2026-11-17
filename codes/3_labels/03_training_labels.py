#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
Final-label rule L4: the strict training label, written as a data product.

The training label differs from the scoring label in one way: quiet in-window seconds become training negatives. Two steps, both ported
from the design study with their logic unchanged:
  (1) the quiet seconds (rule b, codes/3_labels/02_quiet_negatives.py, main()): in a scheduled window, labelled uncertain, covered by
      no evidence interval of any receiver (evidence_receivers empty), passing the receiver-status script's feature-band indicators evaluated inside the
      window (quiet_labels.eval_span, imported unchanged, with the label script's own references from indicator_mask), and at least 60 s from any attack,
      uncertain-by-evidence (uncertain outside the window, or inside 2.6.5's fixed uncertain interval after the attack) or excluded second of the same receiver;
  (2) strict (choices_lib.quiet_sets()): a quiet second is removed when a known-antenna residual extreme (features_base_1hz.csv.gz, the
      largest of known_G/E/C_res_maxabs and known_R_res_db_maxabs) exceeds 30 m at it or within 30 s of it, when it lies in 2.6.10, or
      when it lies in 2.6.2 after the pooled attack interval and up to that receiver's last second with such a residual.
  2.6.12 carries no training label (sealed). The residuals enter the training label only, as adopted; no scoring label moves.
Gates on the REGISTERED labels, before the final file is written: G-A the ported quiet rule reproduces label rule 2's committed 5,537 rule-b
quiet seconds second for second (label_rule2_quiet_seconds.csv); G-B the ported strict keeps 4,187 of them, equal second for second to
choices_lib.quiet_sets(), and its reconciliation table equals choices_strict_reconciliation.csv. Then the same steps run on the final labels.
train_label: clean (scoring clean, or a strict quiet second), attack (scoring attack), none (everything else); reason names why.
inside the Stave stop (12:27:56-12:32:36, which the final set now holds as relabelled)
the training label equals the scoring label; the quiet rule's seconds there are removed before strict, and every training label
outside the stop must equal the final label set's file (checked), so strict keeps 4,181.


Reads   Data/processed/labels/labels_1hz.csv and labels/labels_evidence.csv (gates), labels/labels_1hz_final.csv and labels/labels_evidence_final.csv (final),
        features/features_base_1hz.csv.gz (known-antenna residual extremes), design_study/labels/label_rule2_quiet_seconds.csv,
        design_study/choices/choices_strict_reconciliation.csv; the label script's inputs through 06 (status, sessions) for the indicators
Writes  Data/processed/labels/train_labels_1hz.csv, Data/processed/checks/strict_reconciliation.csv, Data/processed/checks/train_label_counts.csv,
        Data/processed/checks/train_labels_checks.csv
"""

from __future__ import annotations  # future import: type hints kept as text (the project's convention)

import importlib.util  # standard-library tools to load a Python file whose name starts with a digit
import io  # standard library: in-memory text, to compare two tables as the CSV text they would be written as
import sys  # system access: sys.path and sys.modules
from pathlib import Path  # Path objects for building file paths

import numpy as np  # NumPy: boolean arrays
import pandas as pd  # pandas: tables

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
sys.path.insert(0, str(PROJECT / "codes" / "common"))  # list method insert: puts codes/ first on the import path so the helpers module is found
from helpers import exit_if_failed  # status 1 when a check failed

PROC = PROJECT / "Data" / "processed"  # the processed-data folder
CHECK_DIR = PROJECT / "Data" / "processed" / "checks"  # where the check tables go
KEYS = ["test_code", "receiver", "sec"]  # the key columns
RXS = ["SBF_02", "SBF_03", "UBX_01"]  # list: the three receivers
ATT = ("attack_jam_spoof", "attack_spoof")  # tuple: the two attack labels
SEALED = "2.6.12"  # the sealed session
FB = {"GPSL1/E1", "GLOL1", "B1", "L1"}  # set: feature-band names of the status bands (label rule 2's rule b)
NEAR_S = 60  # integer constant: quiet seconds keep 60 s from any trigger
QUIET_RES_M, QUIET_SIDE_S = 30.0, 30  # tuple assignment: strict's residual limit and its sides (choices_lib)
RES_COLS = ["known_G_res_maxabs", "known_E_res_maxabs", "known_C_res_maxabs", "known_R_res_db_maxabs"]  # list: the residual extremes choices_lib reads
CHECKS: list = []  # list: one dict per check
FAILED: list = []  # list: names of the checks that failed


def load(name: str, path: Path):  # function: loads a .py file as a module, unchanged (main() stays behind its __name__ guard)
    spec = importlib.util.spec_from_file_location(name, path)  # where the module comes from
    mod = importlib.util.module_from_spec(spec)  # an empty module object
    sys.modules[name] = mod  # registered, so the module's own imports resolve
    spec.loader.exec_module(mod)  # runs the file's top level
    return mod  # the module


L = load("labels_for_training_labels", PROJECT / "codes" / "3_labels" / "01_labels.py")  # the label script: its inputs, indicator references and constants
A2 = load("quiet_labels_for_06g", PROJECT / "codes" / "3_labels" / "02_quiet_negatives.py")  # the script: eval_span, used unchanged


def check(name: str, ok, detail: str) -> None:  # function: records one check and prints it (ok None marks a reported number)
    CHECKS.append({"check": name, "result": "REPORT" if ok is None else ("PASS" if ok else "FAIL"), "detail": detail})  # nested conditional expression picks the word
    if ok is not None and not bool(ok):  # if: a pass/fail check that did not pass
        FAILED.append(name)  # kept for the exit status
    print(f"  [{'REPORT' if ok is None else ('PASS' if ok else 'FAIL')}] {name}: {detail}")  # printed for the run log


def no_sealed(df: pd.DataFrame) -> pd.DataFrame:  # function: drops every 2.6.12 row (choices_lib.no_sealed)
    return df[df.test_code != SEALED].reset_index(drop=True)  # boolean filter, renumbered


def in_window_pass() -> dict:  # function: per session, the in-window seconds passing the feature-band indicators (label rule 2 rule b)
    d = L.load_inputs()  # the label script's inputs
    W = L.windows(d["gt"])  # its windows
    long = L.status_long(d["st"])  # the receiver-status script's long status table, rebuilt by the label script
    split = ("2.6.12", pd.Timestamp(f"{L.DATE} {L.MARGIN_CONTAMINATION_START}", tz="UTC"))  # the receiver-status script's possibly contaminated 2.6.12 margin split
    _, refs = L.indicator_mask(long, d["gt"], split)  # the label script's indicator references
    S = {"long": long, "refs": refs}  # the dict eval_span reads
    out = {}  # session -> passing seconds
    for r in d["gt"].itertuples():  # for loop over the sessions
        a = pd.Timestamp(max(W[r.test_code]["cs"], W[r.test_code]["ss"]), unit="s", tz="UTC")  # captured part of the window, start
        b = pd.Timestamp(min(W[r.test_code]["ce"], W[r.test_code]["se"]) + 1, unit="s", tz="UTC")  # its end (exclusive)
        out[r.test_code] = A2.eval_span(S, a, b, FB)[0] if b > a else set()  # conditional expression: passing seconds, or none
    return out  # the sets


def pooled(evidence: str) -> dict:  # function: session -> pooled attack intervals (onset, end) from an evidence file
    ev = pd.read_csv(PROC / evidence, dtype={"test_code": str})  # the evidence table
    p = ev[ev.source == "pooled_interval"].sort_values(["test_code", "start"])  # its pooled rows in time order
    return {t: [(int(a), int(b)) for a, b in zip(g.start, g.end)] for t, g in p.groupby("test_code")}  # dict comprehension over sessions


def quiet_rule_b(lab: pd.DataFrame, piv: dict, wp: dict) -> pd.DataFrame:  # function: label rule 2's quiet seconds (rule b), ported line for line from quiet_labels.main()
    p5 = [b_ for a_, b_ in piv.get("2.6.5", [])]  # 2.6.5 interval ends
    post54 = (lab.test_code == "2.6.5") & (lab.sec > max(p5)) & (lab.sec <= L.T(L.POST_265_END))  # the 2.6.5 post-test interval
    unc_ev = (lab.label == "uncertain") & (~lab.in_scheduled_window.astype(bool) | post54)  # uncertain by evidence
    trig = lab.label.isin(ATT) | unc_ev | (lab.label == "excluded")  # seconds that keep quiet seconds 60 s away
    qrows = []  # quiet seconds
    for (t, rx), x in lab.groupby(["test_code", "receiver"], sort=False):  # pairs, in the table's order
        xi = x.index  # row positions in lab
        sec = x.sec.to_numpy()  # seconds
        tg = np.flatnonzero(trig.loc[xi].to_numpy())  # trigger positions
        near = np.zeros(len(x), bool)  # within 60 s of a trigger
        if len(tg):  # any
            pos = np.searchsorted(tg, np.arange(len(x)))  # nearest trigger after
            da = np.where(pos < len(tg), tg[np.minimum(pos, len(tg) - 1)] - np.arange(len(x)), 10**9)  # distance after
            db = np.where(pos > 0, np.arange(len(x)) - tg[np.maximum(pos - 1, 0)], 10**9)  # distance before
            near = np.minimum(da, db) < NEAR_S  # closer than 60 s
        # boolean array over two lines: the quiet rule
        q = (x.in_scheduled_window.astype(bool).to_numpy() & (x.label == "uncertain").to_numpy() & (x.evidence_receivers == "").to_numpy()
             & np.isin(sec, list(wp[t])) & ~near)  # the quiet rule
        qrows.append(pd.DataFrame({"rule": "b_feature_bands", "test_code": t, "receiver": rx, "sec": sec[q]}))  # kept
    return pd.concat(qrows, ignore_index=True)  # quiet seconds


def k2_strict(q: pd.DataFrame, res: pd.DataFrame, piv: dict) -> tuple:  # function: choices_lib.quiet_sets() ported line for line, on a given quiet set and pooled intervals
    q = no_sealed(q[q.rule == "b_feature_bands"][KEYS]).drop_duplicates()  # rule (b), the defining rule
    big = res[res.res_max > QUIET_RES_M]  # seconds with a residual above 30 m, any label
    near = set()  # keys within 30 s of such a second
    for (t, rx), g in big.groupby(["test_code", "receiver"]):  # pairs
        s = g.sec.to_numpy()  # their seconds
        near |= {(t, rx, int(v)) for c in s for v in range(c - QUIET_SIDE_S, c + QUIET_SIDE_S + 1)}  # set comprehension: every second within +-30 s, the second itself included
    qk = list(zip(q.test_code, q.receiver, q.sec.astype(int)))  # quiet keys
    r1 = np.array([k in near for k in qk])  # reason 1: a residual above 30 m at the second or within 30 s of it
    r2 = (q.test_code == "2.6.10").to_numpy()  # reason 2: every 2.6.10 quiet second
    a_end = max(b for _, b in piv["2.6.2"])  # end of 2.6.2's attack
    clear = {rx: int(big[(big.test_code == "2.6.2") & (big.receiver == rx) & (big.sec > a_end)].sec.max()) for rx in RXS}  # dict comprehension: last post-attack 2.6.2 second with a residual above 30 m per receiver
    r3 = np.array([t == "2.6.2" and a_end < s <= clear[rx] for t, rx, s in qk])  # reason 3: 2.6.2's post-attack counterfeit stretch until the residuals clear on that receiver
    drop = r1 | r2 | r3  # any reason
    strict = q[~drop].reset_index(drop=True)  # strict
    rec = []  # reconciliation rows
    for (t, rx), g in q.assign(r1=r1, r2=r2, r3=r3, kept=~drop).groupby(["test_code", "receiver"]):  # per pair
        # list.append of a dict over two lines: counts by reason (a second can carry several reasons)
        rec.append({"test_code": t, "receiver": rx, "quiet_rule_b": len(g), "res_over_30m_or_within_30s": int(g.r1.sum()), "session_2610": int(g.r2.sum()),
                    "post_attack_2_6_2_until_clear": int(g.r3.sum()), "removed_any": int((~g.kept).sum()), "k2_strict": int(g.kept.sum()), "clear_utc": pd.Timestamp(clear[rx], unit="s").strftime("%H:%M:%S") if t == "2.6.2" else ""})  # the reasons' counts and, in 2.6.2, when the residuals clear
    reasons = q.assign(r1=r1, r2=r2, r3=r3)  # the quiet seconds with their removal reasons (for the training label's reason column)
    return strict, pd.DataFrame(rec), reasons  # the strict set, the table and the reasons


def residuals() -> pd.DataFrame:  # function: the known-antenna residual extremes per second, 2.6.12 dropped (choices_lib.quiet_sets' read)
    res = no_sealed(pd.read_csv(PROC / "features" / "features_base_1hz.csv.gz", usecols=KEYS + RES_COLS, dtype={"test_code": str}))  # labelling evidence only
    res["res_max"] = res[RES_COLS].max(axis=1)  # the largest over the four constellations (missing only where all are missing)
    return res  # the table


def csv_text(df: pd.DataFrame) -> str:  # function: a table as the CSV text it would be written as
    buf = io.StringIO()  # an in-memory text file
    df.to_csv(buf, index=False, lineterminator="\n")  # written into it
    return buf.getvalue()  # the text


def read_labels(name: str) -> pd.DataFrame:  # function: a label table with the columns the quiet rule reads (empty text kept as "")
    cols = KEYS + ["utc", "label", "in_scheduled_window", "evidence_receivers", "exclusion_reason"]  # the columns needed
    return pd.read_csv(PROC / name, usecols=cols, dtype={"test_code": str, "evidence_receivers": str, "exclusion_reason": str}, keep_default_na=False)  # keep_default_na=False: "" stays ""


def train_table(lab: pd.DataFrame, strict: pd.DataFrame, reasons: pd.DataFrame) -> pd.DataFrame:  # function: the training label of every grid second, with its reason
    sk = set(zip(strict.test_code, strict.receiver, strict.sec.astype(int)))  # strict keys
    why = {(t, rx, int(s)): "; ".join(n for n, f in (("residual over 30 m within 30 s", a), ("2.6.10", b), ("2.6.2 post-attack until clear", c)) if f)  # generator in join: the removal reasons
           for t, rx, s, a, b, c in zip(reasons.test_code, reasons.receiver, reasons.sec, reasons.r1, reasons.r2, reasons.r3)}  # dict comprehension: quiet key -> its removal reasons
    tl, rs = [], []  # training labels; reasons
    for t, rx, s, lb, ex in zip(lab.test_code, lab.receiver, lab.sec, lab.label, lab.exclusion_reason):  # for loop over every grid second
        k = (t, rx, int(s))  # its key
        if t == SEALED:  # if: the sealed session
            tl.append("none"); rs.append("sealed: no training label")  # two statements: no label, and why
        elif lb == "clean":  # elif: a scoring clean second
            tl.append("clean"); rs.append("scoring clean")  # clean in training too
        elif lb in ATT:  # elif: a scoring attack second
            tl.append("attack"); rs.append(f"scoring {lb}")  # attack in training
        elif k in sk:  # elif: a strict quiet second
            tl.append("clean"); rs.append("strict quiet second (scoring uncertain)")  # a training negative
        elif k in why:  # elif: a quiet second that strict removed
            tl.append("none"); rs.append(f"quiet, removed: {why[k]}")  # no training label, with the reasons
        elif lb == "excluded":  # elif: excluded
            tl.append("none"); rs.append(f"excluded ({ex})")  # with its exclusion reason
        else:  # else: uncertain and not quiet
            tl.append("none"); rs.append("uncertain")  # no training label
    return pd.DataFrame({"test_code": lab.test_code, "receiver": lab.receiver, "sec": lab.sec, "utc": lab.utc, "train_label": tl, "reason": rs})  # the table


def main() -> None:  # function: gates on the registered labels, then the final training labels
    wp = in_window_pass()  # in-window passing seconds (label-independent)
    res = residuals()  # residual extremes (label-independent)
    print("== gates on the registered labels ==")  # header
    reg = read_labels("labels/labels_1hz.csv")  # registered labels
    q = quiet_rule_b(reg, pooled("labels/labels_evidence.csv"), wp)  # the ported quiet rule
    labels = pd.read_csv(PROC / "design_study" / "labels" / "label_rule2_quiet_seconds.csv", dtype={"test_code": str})  # label rule 2's committed quiet seconds
    labelsb = labels[labels.rule == "b_feature_bands"].reset_index(drop=True)  # rule (b)
    ga = csv_text(q) == csv_text(labelsb)  # equal as written: same seconds, same order
    check("G-A ported quiet rule = label rule 2's rule-b quiet seconds, second for second", ga, f"{len(q)} ported, {len(labelsb)} committed")  # gate A
    strict, rec, _ = k2_strict(q, res, pooled("labels/labels_evidence.csv"))  # the ported strict on the registered labels
    L3 = load("choices_lib_for_06g", PROJECT / "codes" / "5_baselines" / "04_pipeline_library.py")  # the library, for its own quiet_sets()
    _, strict3, rec3 = L3.quiet_sets()  # the design study's strict, from the committed label rule 2 file
    rc = pd.read_csv(PROC / "design_study" / "choices" / "choices_strict_reconciliation.csv", dtype={"test_code": str}, keep_default_na=False)  # the design-choice study's committed reconciliation
    gb = len(strict) == 4187 and len(no_sealed(q)) == 5537 and csv_text(strict) == csv_text(strict3) and csv_text(rec) == csv_text(rec3) and csv_text(rec) == csv_text(rc)  # every part equal
    check("G-B ported strict keeps 4,187 of 5,537, = choices_lib.quiet_sets() second for second, reconciliation = choices_strict_reconciliation.csv", gb, f"kept {len(strict)} of {len(no_sealed(q))}; quiet_sets() {len(strict3)}")  # gate B
    rec_all = [rec.assign(label_set="registered (gate)")]  # reconciliation tables to write
    if not (ga and gb):  # if: a gate failed
        pd.DataFrame(CHECKS).to_csv(CHECK_DIR / "train_labels_checks.csv", index=False, lineterminator="\n")  # the checks
        raise SystemExit("STOP: an L4 gate failed on the registered labels; labels/train_labels_1hz.csv is not written")  # stop
    print("== final labels ==")  # header
    fin = read_labels("labels/labels_1hz_final.csv")  # final labels
    pv = pooled("labels/labels_evidence_final.csv")  # final pooled intervals
    qf = quiet_rule_b(fin, pv, wp)  # quiet seconds on the final labels
    stop = L.stave_stop()  # the Stave stop, 12:27:56-12:32:36, which the final set now holds
    in_stop = ((qf.test_code == "2.6.5") & qf.sec.between(*stop)).to_numpy()  # quiet seconds inside the stop (between with tuple unpacking: inclusive)
    check("Stave stop: quiet seconds the rule would add there, removed (training label = scoring label inside the stop)", None, f"{int(in_stop.sum())} receiver-seconds ({', '.join(f'{rx} {int((in_stop & (qf.receiver == rx).to_numpy()).sum())}' for rx in RXS)})")  # reported
    qf = qf[~in_stop].reset_index(drop=True)  # removed before strict: the Stave stop relabel measured the stop with its scoring labels only
    strict_f, rec_f, why_f = k2_strict(qf, res, pv)  # strict on them
    rec_all.append(rec_f.assign(label_set="final"))  # kept
    tr = train_table(fin, strict_f, why_f)  # the training labels
    sm = ((tr.test_code == "2.6.5") & tr.sec.between(*stop)).to_numpy()  # the stop's seconds in the training table
    tr.loc[sm, "reason"] = "Stave stop: training label = scoring label (" + fin.label[sm] + ")"  # the reason, with the scoring label (string concatenation over a column)
    old = pd.read_csv(PROC / "labels" / "train_labels_1hz.csv", dtype={"test_code": str}, keep_default_na=False)  # the final label set's training labels, before this run overwrites them
    same_out = bool(old[KEYS].equals(tr[KEYS]) and (old.train_label.to_numpy()[~sm] == tr.train_label.to_numpy()[~sm]).all() and (old.reason.to_numpy()[~sm] == tr.reason.to_numpy()[~sm]).all())  # outside the stop: same rows, labels and reasons
    check("outside the Stave stop every training label and reason equals the final label set's labels/train_labels_1hz.csv; strict keeps 4,181", same_out and len(strict_f) == 4181, f"{int((~sm).sum())} rows compared; kept {len(strict_f)}; inside the stop {int(sm.sum())} rows: {tr[sm].train_label.value_counts().to_dict()}")  # the decision
    if old.equals(tr):  # if: this run writes what is already on disk (a run again)
        print("  (labels/train_labels_1hz.csv unchanged on disk)")  # said
    tr.to_csv(PROC / "labels" / "train_labels_1hz.csv", index=False, lineterminator="\n")  # written
    pd.concat(rec_all, ignore_index=True).to_csv(CHECK_DIR / "strict_reconciliation.csv", index=False, lineterminator="\n")  # both reconciliations
    cnt = tr.groupby(["test_code", "receiver", "train_label", "reason"], sort=False).size().rename("seconds").reset_index()  # counts by label and reason, in table order
    cnt.to_csv(CHECK_DIR / "train_label_counts.csv", index=False, lineterminator="\n")  # written
    key = tr.test_code + "|" + tr.receiver + "|" + tr.sec.astype(str)  # one key per row
    check("every final grid second has one training label", key.is_unique and len(tr) == len(fin) and tr.train_label.isin(["clean", "attack", "none"]).all(), f"{len(tr)} rows")  # complete and valid
    ns = tr.test_code != SEALED  # outside the sealed session
    check("training attack = scoring attack and scoring clean is training clean (outside 2.6.12)", bool(((tr.train_label == "attack") == fin.label.isin(ATT))[ns].all() and (tr.train_label[ns & (fin.label == "clean")] == "clean").all()), "per second")  # the training label only adds negatives
    added = (tr.train_label == "clean") & (fin.label != "clean")  # the training negatives added
    check("every added training negative is an in-window uncertain second", bool(((fin.label == "uncertain") & (fin.in_scheduled_window == 1))[added].all()), f"{int(added.sum())} added")  # quiet seconds only
    check("no 2.6.10 second added; 2.6.12 carries none", bool(not (added & (tr.test_code == "2.6.10")).any() and (tr.train_label[~ns] == "none").all()), "per second")  # reasons 2 and E15
    check("final strict count", None, f"{len(strict_f)} of {len(no_sealed(qf))} quiet seconds kept (registered: {len(strict)} of {len(no_sealed(q))})")  # reported
    pd.DataFrame(CHECKS).to_csv(CHECK_DIR / "train_labels_checks.csv", index=False, lineterminator="\n")  # the checks
    print(pd.concat(rec_all, ignore_index=True).to_string(index=False))  # for the run log


if __name__ == "__main__":  # if: true only when the file is run directly
    main()  # runs L4
    exit_if_failed(FAILED, "3_labels/03_training_labels.py")  # the helpers module: status 1 when any check failed
