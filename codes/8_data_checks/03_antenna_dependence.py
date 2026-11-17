#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
Data check 4 (the supervisor's checks of 19 October 2026): how far each receiver's labels rest on the other receivers' evidence, now that
each receiver is known to have had its own antenna; and the per-receiver label set that the sensitivity (8_data_checks/04_label_sensitivity.py) uses.

The labels of Section 4.6 pool the evidence of the three receivers: an attack interval is the union of their evidence intervals and
every receiver receives it, and a clean second must pass every receiver's the receiver-status indicators. This script rebuilds, in memory and with
the label script's own functions unchanged, three label sets per receiver:
  pooled     the label script's final rule (must equal labels/labels_1hz_final.csv's label, every row: the reproduction check);
  option 1   own evidence, pooled indicators (must equal labels/labels_1hz_final.csv's label_per_receiver_only, every row);
  own        own evidence AND own indicators (the per-receiver label set of the sensitivity).
The quiet training negatives are rebuilt with the training-label script's functions the same two ways (pooled: must equal labels/train_labels_1hz.csv's 4,181
strict seconds, own: the receiver's own evidence, own indicators and own attack intervals). 2.6.12's labels are read (allowed: labels
and gains are logged quantities); it carries no training label. Evidence intervals are read from labels/labels_evidence_final.csv (logged),
so no raw file and no 2.6.12 position is recomputed.

Decision rule (19 October 2026), fixed before any number: if, for every receiver, fewer than 5 % of its attack
seconds and fewer than 5 % of its clean seconds depend only on other receivers, the labels stand; otherwise the sensitivity runs.

Reads   Data/processed/labels/labels_1hz_final.csv, labels/labels_evidence_final.csv, labels/train_labels_1hz.csv, and the label script's and the training-label script's inputs through them
        (session_ground_truth, receiver_status_timeseries, observables_epochs_1hz, jam_phase_boundaries, ta2_test_effects,
        features/features_base_1hz.csv.gz residual extremes, features/stave_base_265_final.csv.gz, design_study/labels/label_rule5_stave.csv)
Writes  Data/processed/data_checks/antenna_dependence_by_session.csv   per session x receiver: attack, clean and quiet-negative dependence
        Data/processed/data_checks/antenna_dependence_by_receiver.csv  per receiver over the five training sessions, with the shares
        Data/processed/data_checks/antenna_only_others_runs.csv        every run of attack seconds supported only by other receivers
        Data/processed/data_checks/antenna_own_clean_2610.csv          2.6.10: own-rule clean seconds that the pooled rule labels otherwise
        Data/processed/data_checks/antenna_checks.csv                  the self-checks
        Data/processed/data_checks/labels_per_receiver_1hz.csv         every grid second: pooled and per-receiver scoring and training labels
        Data/processed/data_checks/labels_per_receiver_counts.csv      seconds per label, both sets, per session x receiver
"""

from __future__ import annotations  # future import: type hints kept as text (the project's convention)

import importlib.util  # standard-library module loader: the training-label script's name starts with a digit, so no plain import
import os  # operating-system access: MPLCONFIGDIR is set before the label script loads matplotlib
import sys  # standard-library system access: sys.path and sys.modules
from pathlib import Path  # Path objects for building file paths

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
os.environ["MPLCONFIGDIR"] = str(PROJECT / "codes" / ".mplconfig")  # matplotlib's cache inside the project folder (the label script imports matplotlib)

import numpy as np  # NumPy: boolean arrays over seconds
import pandas as pd  # pandas: tables

sys.path.insert(0, str(PROJECT / "codes" / "common"))  # list method insert: codes/ first on the import path, so the helpers module is found
from helpers import exit_if_failed  # ends the run with status 1 when a check failed

SEED = 20260914  # integer constant: the project's fixed seed (nothing here draws a random number)
np.random.seed(SEED)  # seeds NumPy's generator for form's sake, as every script does
PROC = PROJECT / "Data" / "processed"  # the '/' operator joins path parts: the processed-data folder
OUT = PROC / "data_checks"  # this check's output folder (its only write location for data)
CHECKS: list = []  # list: one dict per self-check
FAILED: list = []  # list: names of the checks that failed


def load(name: str, path: Path):  # function: loads a .py file as a module, unchanged (its main() stays behind its __name__ guard)
    spec = importlib.util.spec_from_file_location(name, path)  # a module "spec": where the module comes from
    mod = importlib.util.module_from_spec(spec)  # an empty module object built from that spec
    sys.modules[name] = mod  # registered, so the module's own imports resolve
    spec.loader.exec_module(mod)  # runs the file's top level (definitions only; nothing is written)
    return mod  # the loaded module


G = load("training_labels_for_antenna_check", PROJECT / "codes" / "3_labels" / "03_training_labels.py")  # the training-label script: the quiet rule and strict; it loads the label script and the quiet-negatives script itself
L = G.L  # 06_labels as the training-label script loaded it: build(), the indicator rule, the windows and the Stave stop
RXS, SESSIONS = L.RXS, L.SESSIONS  # tuple assignment: the three receivers and the six sessions in time order (the label script's lists)
ATT = ("attack_jam_spoof", "attack_spoof")  # tuple: the two attack labels
STATUS_RX = L.STATUS_RX  # dict: stream -> the status table's receiver tag (SBF_02 -> Septentrio_02 ...)
SEALED = "2.6.12"  # the sealed session: labels read, no training label, no feature
TRAINING = [s for s in SESSIONS if s != SEALED]  # list comprehension: the five training (position) sessions
SPOOF_END_2610 = "ground_truth/spoof_end_2610.csv"  # the logged per-receiver end of spoofed tracking in 2.6.10, for the caveat


def check(name: str, ok, detail: str) -> None:  # function: records one self-check and prints it (ok None marks a reported number)
    CHECKS.append({"check": name, "result": "REPORT" if ok is None else ("PASS" if ok else "FAIL"), "detail": detail})  # nested conditional expression picks the word
    if ok is not None and not bool(ok):  # if: a pass/fail check that did not pass
        FAILED.append(name)  # kept for the exit status
    print(f"  [{CHECKS[-1]['result']}] {name}: {detail}")  # printed for the run log


def write_csv(df: pd.DataFrame, name: str) -> None:  # function: one table into Data/processed/data_checks with fixed line endings (byte-stable)
    OUT.mkdir(parents=True, exist_ok=True)  # the folder, created on first use
    df.to_csv(OUT / name, index=False, lineterminator="\n")  # index=False: no row numbers; "\n": the same bytes on every system


def runs_text(secs: np.ndarray) -> list:  # function: sorted whole seconds -> list of (first, last) runs of consecutive seconds
    s = np.sort(np.asarray(secs, dtype=np.int64))  # the seconds in order
    if not len(s):  # if: none
        return []  # no run
    br = np.flatnonzero(np.diff(s) > 1)  # np.diff: gaps between neighbours; flatnonzero: where a gap exceeds one second
    return list(zip(np.r_[s[0], s[br + 1]], np.r_[s[br], s[-1]]))  # np.r_ joins arrays: run starts and run ends, paired by zip


def hms(sec) -> str:  # function: unix seconds -> "HH:MM:SS" (the label script's helper, named here for brevity)
    return L.hms(sec)  # the label script's own formatter


# ---------------------------------------------------------------- the label script's inputs, exactly as the label script's main_final() builds them
def inputs() -> dict:  # function: every input of the label rebuild, read once
    d = L.load_inputs()  # the label script's tables (sessions, status, grid, phases, overrides)
    W = L.windows(d["gt"])  # per session: capture and scheduled window in whole seconds
    J = L.jam_phases(d["ph"], d["ov"], W)  # the registered jamming phases
    d, J, l2 = L.final_inputs(d, W, J)  # the label script's final-set changes: L1's phases and L2's Test Area 2 intervals
    long = L.status_long(d["st"])  # the receiver-status script's long status table, rebuilt by the label script
    split = ("2.6.12", pd.Timestamp(f"{L.DATE} {L.MARGIN_CONTAMINATION_START}", tz="UTC"))  # the receiver-status script's possibly contaminated 2.6.12 margin split of 2.6.12's pre-test margin
    parts, refs = L.indicator_mask(long, d["gt"], split)  # the label script's wrapped indicator rule: per-point mask and the references
    ind = L.indicator_seconds(parts)  # per session, the seconds passing every (pooled) indicator
    evf = pd.read_csv(PROC / "labels" / "labels_evidence_final.csv", dtype={"test_code": str})  # the logged evidence intervals of the final set
    ev = evf[evf.source != "pooled_interval"].reset_index(drop=True)  # the receivers' own intervals (the pooled rows are the label script's output, not input)
    stop = L.stave_stop()  # the Stave stop, 12:27:56-12:32:36 (the label script's function: it stops if preprocessing/vehicle_motion.csv disagrees)
    sb = pd.read_csv(L.STAVE_BASE, usecols=["test_code", "receiver", "sec", "n_tracked"], dtype={"test_code": str})  # the final label set's Stave base: the stop's observed seconds
    sb = sb[(sb.test_code == "2.6.5") & sb.sec.between(*stop)]  # the stop's rows only (between with tuple unpacking: inclusive)
    # pd.concat: the observables script's grid of observed seconds plus the stop's seconds, as the label script's main_final builds its grid
    ep_n = pd.concat([d["ep"], sb[["test_code", "receiver", "sec"]].assign(has_observations=(sb.n_tracked > 0).astype(int).to_numpy())], ignore_index=True)
    return {"d": d, "W": W, "J": J, "l2": l2, "long": long, "split": split, "refs": refs, "ind": ind, "ev": ev, "stop": stop, "ep_n": ep_n}  # dict literal: everything


def own_indicator_seconds(I: dict, rx: str) -> dict:  # function: the label script's indicator rule with one receiver's indicators only (references unchanged)
    tag = STATUS_RX[rx]  # the status table's name of this receiver
    out = []  # one dict per session and margin part, as the label script's indicator_mask returns
    for _, sr in I["d"]["gt"].iterrows():  # for loop over the sessions, in the label script's (capture) order
        pp = L.S03.margin_parts(sr)  # the receiver-status script's two margins of the session
        if sr["test_code"] == I["split"][0] and pp[0][0] < I["split"][1] < pp[0][1]:  # if: 2.6.12's pre-test margin, cut at the possibly contaminated 2.6.12 margin time as 06 cuts it
            pp = [(pp[0][0], I["split"][1]), (I["split"][1], pp[0][1]), pp[1]]  # the pre-test margin in two parts
        for a, b in pp:  # for loop over the margin parts
            if b <= a:  # if: an empty part
                continue  # skipped, as in the label script
            grid = pd.date_range(a, b, freq="100ms", inclusive="left")  # the label script's 0.1 s grid over the part
            bad = np.zeros(len(grid), bool)  # contaminated by any of this receiver's indicators
            for (r_, band, fld), (ref, q) in I["refs"].items():  # for loop over the label script's indicators and their references (reference level, IQR)
                if r_ != tag:  # if: another receiver's indicator
                    continue  # left out: this is the own-indicator rule
                g = I["long"][(I["long"].receiver == r_) & (I["long"].band == band) & (I["long"].field == fld)].sort_values("timestamp_utc")  # its readings in time order
                gg = g[(g.timestamp_utc >= a - pd.Timedelta(seconds=60)) & (g.timestamp_utc < b)].set_index("timestamp_utc").value  # from 60 s before the part, as in the label script
                gg = gg[~gg.index.duplicated()]  # one reading per time (the first), as in the label script
                # reindex with forward fill: each grid point takes the last reading, at most 60 s old (the label script's rule); none -> NaN
                v = gg.reindex(grid, method="ffill", tolerance=pd.Timedelta(seconds=60)).to_numpy() if len(gg) else np.full(len(grid), np.nan)
                rule_bad = (v < ref - 3.0) if fld == "agc_gain_db" else (v > ref + 3 * q)  # conditional expression: the label script's gain rule or IQR rule
                bad |= rule_bad & ~np.isnan(v)  # "|=": contaminated where covered and beyond the rule
            out.append({"test_code": sr["test_code"], "a": a, "b": b, "grid": grid, "bad": bad})  # one part, in the label script's form
    return L.indicator_seconds(out)  # the label script's own conversion to passing and failing whole seconds


# ---------------------------------------------------------------- the three label sets
def build_all(I: dict, ind_own: dict) -> tuple:  # function: per session x receiver, the pooled, option-1 and own labels, plus the own attack intervals
    rows, own_iv = [], {}  # per-second tables; (session, receiver) -> own attack intervals
    for t in SESSIONS:  # for loop over the sessions
        for rx in RXS:  # for loop over the receivers
            k_all = I["l2"].get(t, {}).get("antenna", [])  # L2's intervals with an effect on any receiver (the pooled rule's)
            k_own = I["l2"].get(t, {}).get(rx, [])  # L2's intervals with an effect on this receiver (Option 1's)
            # the label script's build(): pooled evidence, pooled indicators; vehicle_motion_open: the Stave stop as the label script's final set holds it (proven equal to the Stave stop relabel by the label script)
            Lp, _ = L.build(t, rx, I["ev"], I["ep_n"], I["W"], I["J"], I["ind"], own_rx_only=False, l2=k_all, vehicle_motion_open=I["stop"])
            L1, _ = L.build(t, rx, I["ev"], I["ep_n"], I["W"], I["J"], I["ind"], own_rx_only=True, l2=k_own, vehicle_motion_open=I["stop"])  # Option 1: own evidence, pooled indicators
            Lo, oiv = L.build(t, rx, I["ev"], I["ep_n"], I["W"], I["J"], ind_own[rx], own_rx_only=True, l2=k_own, vehicle_motion_open=I["stop"])  # own evidence and own indicators
            own_iv[(t, rx)] = oiv  # this receiver's own attack intervals (the label script's merge across gaps under 30 s)
            # pd.DataFrame from a dict of columns: the three labels of every capture second, with the exclusion reason and window flag
            rows.append(pd.DataFrame({"test_code": t, "receiver": rx, "sec": Lp.sec.to_numpy(), "label_pooled": Lp.label.to_numpy(), "label_option1": L1.label.to_numpy(),
                                      "label_own": Lo.label.to_numpy(), "excl_pooled": Lp.exclusion_reason.to_numpy(), "excl_own": Lo.exclusion_reason.to_numpy(),  # continued: more fields of the row
                                      "in_window": Lp.in_scheduled_window.astype(int).to_numpy()}))  # one table per pair
    return pd.concat(rows, ignore_index=True), own_iv  # one table in the label script's grid order, and the intervals


# ---------------------------------------------------------------- quiet negatives and training labels
def in_window_pass_own(I: dict, rx: str) -> dict:  # function: the training-label script's in-window indicator pass with one receiver's indicators only
    S = {"long": I["long"], "refs": {k: v for k, v in I["refs"].items() if k[0] == STATUS_RX[rx]}}  # dict comprehension: only this receiver's references
    out = {}  # session -> passing seconds
    for r in I["d"]["gt"].itertuples():  # for loop over the sessions
        w = I["W"][r.test_code]  # its windows
        a = pd.Timestamp(max(w["cs"], w["ss"]), unit="s", tz="UTC")  # captured part of the scheduled window, start (the training-label script's rule)
        b = pd.Timestamp(min(w["ce"], w["se"]) + 1, unit="s", tz="UTC")  # its end, exclusive
        out[r.test_code] = G.A2.eval_span(S, a, b, G.FB)[0] if b > a else set()  # conditional expression: the design study's eval_span, unchanged, or nothing
    return out  # the sets


def quiet_and_train(I: dict, lab: pd.DataFrame, own_iv: dict) -> tuple:  # function: strict quiet seconds and training labels, pooled (reproduction) and own
    res = G.residuals()  # the training-label script's residual extremes per second (2.6.12 dropped)
    fin = G.read_labels("labels/labels_1hz_final.csv")  # the final labels with the columns the training-label script's quiet rule reads
    pv = G.pooled("labels/labels_evidence_final.csv")  # the final pooled intervals per session
    stop = I["stop"]  # the Stave stop
    qf = G.quiet_rule_b(fin, pv, G.in_window_pass())  # the training-label script's quiet rule on the pooled labels
    qf = qf[~((qf.test_code == "2.6.5") & qf.sec.between(*stop))].reset_index(drop=True)  # the training-label script removes the stop's quiet seconds before strict
    strict_p, _, why_p = G.k2_strict(qf, res, pv)  # the training-label script's strict: the pooled quiet negatives and the removal reasons
    tr_p = G.train_table(fin, strict_p, why_p)  # the training-label script's training labels on the pooled set
    parts_s, parts_w, parts_t = [], [], []  # per receiver: own strict seconds, own removal reasons, own training labels
    for rx in RXS:  # for loop over the receivers
        fo = fin[fin.receiver == rx].reset_index(drop=True).copy()  # this receiver's rows of the final label table
        m = lab[lab.receiver == rx].reset_index(drop=True)  # the rebuilt labels of the same rows (same grid order, checked in main)
        fo["label"] = m.label_own.to_numpy()  # the own label in place of the pooled one
        fo["exclusion_reason"] = m.excl_own.to_numpy()  # the own exclusion reason (equal to the pooled one, checked in main)
        ev_rx = np.array([rx in str(e).split(";") for e in fo.evidence_receivers])  # list comprehension: seconds this receiver's own evidence covers
        fo["evidence_receivers"] = np.where(ev_rx, rx, "")  # the quiet rule's "no evidence" test now reads own evidence only
        pv_r = {t: own_iv[(t, rx)] for t in SESSIONS}  # dict comprehension: this receiver's own attack intervals per session
        q = G.quiet_rule_b(fo, pv_r, in_window_pass_own(I, rx))  # the training-label script's quiet rule on own labels, own evidence and own indicators
        q = q[~((q.test_code == "2.6.5") & q.sec.between(*stop))].reset_index(drop=True)  # the stop's quiet seconds removed, as the training-label script
        s_r, _, w_r = G.k2_strict(q, res, pv_r)  # strict with this receiver's own attack end in 2.6.2
        parts_s.append(s_r[s_r.receiver == rx])  # its own quiet negatives
        parts_w.append(w_r[w_r.receiver == rx])  # its removal reasons
        parts_t.append(G.train_table(fo, s_r[s_r.receiver == rx], w_r[w_r.receiver == rx]))  # the training-label script's training table on the own labels
    strict_o = pd.concat(parts_s, ignore_index=True)  # every receiver's own quiet negatives
    tr_o = pd.concat(parts_t, ignore_index=True)  # own training labels, receiver by receiver
    order = fin[["test_code", "receiver", "sec"]].reset_index()  # the final table's row order, with its position
    tr_o = order.merge(tr_o, on=["test_code", "receiver", "sec"], how="left", validate="one_to_one").sort_values("index").drop(columns="index").reset_index(drop=True)  # back into the grid order
    return strict_p, tr_p, strict_o, tr_o  # four tables


# ---------------------------------------------------------------- dependence metrics
def dependence(lab: pd.DataFrame, ev_rx: pd.Series, ind_own: dict, QP: set, QO: set) -> pd.DataFrame:  # function: per session x receiver, every count of check 4
    out = []  # one dict per pair
    for t in SESSIONS:  # for loop over the sessions
        for rx in RXS:  # for loop over the receivers
            m = ((lab.test_code == t) & (lab.receiver == rx)).to_numpy()  # boolean mask: the pair's rows
            x, e = lab[m], ev_rx[m]  # the pair's labels and evidence receivers
            fo = set().union(*[ind_own[o].get(t, {"fail": set()})["fail"] for o in RXS if o != rx])  # set union: seconds failing another receiver's indicator
            A, A1 = x.label_pooled.isin(ATT).to_numpy(), x.label_option1.isin(ATT).to_numpy()  # attack under the pooled rule and under own evidence
            own_ev = np.array([rx in str(v).split(";") for v in e])  # list comprehension: own evidence covers the second
            oth_ev = np.array([str(v) != "" and rx not in str(v).split(";") for v in e])  # only other receivers' evidence covers it
            C, Co = (x.label_pooled == "clean").to_numpy(), (x.label_own == "clean").to_numpy()  # clean under the pooled and the own rule
            sec = x.sec.to_numpy()  # the seconds
            qp = np.array([(t, rx, int(s)) in QP for s in sec])  # list comprehension: pooled quiet negatives
            qo = np.array([(t, rx, int(s)) in QO for s in sec])  # own quiet negatives
            fo_m = np.isin(sec, list(fo))  # np.isin: seconds failing another receiver's indicator
            # dict literal: the pair's counts and the share of its attack seconds that only other receivers support
            out.append({"test_code": t, "receiver": rx, "attack_pooled": int(A.sum()), "attack_own_evidence": int((A & A1).sum()), "attack_only_others": int((A & ~A1).sum()),
                        "attack_only_others_share": round(float((A & ~A1).sum() / A.sum()), 4) if A.sum() else np.nan,  # continued: more fields of the row
                        "attack_s_own_evidence_direct": int((A & own_ev).sum()), "attack_s_only_other_evidence_direct": int((A & oth_ev).sum()), "attack_s_merge_gap": int((A & (e == "").to_numpy()).sum()),  # continued: more fields of the row
                        "attack_own_not_pooled": int((A1 & ~A).sum()), "clean_pooled": int(C.sum()), "clean_own": int(Co.sum()), "clean_pooled_not_own": int((C & ~Co).sum()),  # continued: more fields of the row
                        "clean_own_lost_to_others": int((Co & ~C).sum()), "clean_own_failing_other_indicator": int((Co & fo_m).sum()), "clean_pooled_failing_other_indicator": int((C & fo_m).sum()),  # continued: more fields of the row
                        "quietneg_pooled": int(qp.sum()), "quietneg_own": int(qo.sum()), "quietneg_pooled_not_own": int((qp & ~qo).sum()), "quietneg_own_lost_to_others": int((qo & ~qp).sum())})  # the row
    return pd.DataFrame(out)  # the table


def main() -> None:  # function: check 4 from start to finish
    I = inputs()  # the label script's inputs
    ind_own = {rx: own_indicator_seconds(I, rx) for rx in RXS}  # dict comprehension: per receiver, its own passing and failing seconds
    for t in I["ind"]:  # for loop over the sessions with margins
        u = set().union(*[ind_own[rx].get(t, {"fail": set()})["fail"] for rx in RXS])  # the union over the receivers of their own failing seconds
        check(f"{t}: union of the receivers' own indicator failures = the label script's pooled failures", u == I["ind"][t]["fail"], f"{len(u)} and {len(I['ind'][t]['fail'])} seconds")  # the own rule splits the pooled one exactly
    lab, own_iv = build_all(I, ind_own)  # the three label sets
    fb = pd.read_csv(PROC / "labels" / "labels_1hz_final.csv", dtype={"test_code": str}, keep_default_na=False)  # the published final labels
    same_keys = bool((lab[["test_code", "receiver", "sec"]].to_numpy() == fb[["test_code", "receiver", "sec"]].to_numpy()).all())  # the same grid in the same order
    check("rebuild: grid = labels/labels_1hz_final.csv, row for row", same_keys and len(lab) == len(fb), f"{len(lab)} rows")  # grid identity
    check("rebuild: pooled label = labels/labels_1hz_final.csv label, every row", bool((lab.label_pooled.to_numpy() == fb.label.to_numpy()).all()), f"{int((lab.label_pooled.to_numpy() != fb.label.to_numpy()).sum())} differ")  # reproduction of the label script
    check("rebuild: option-1 label = labels/labels_1hz_final.csv label_per_receiver_only, every row", bool((lab.label_option1.to_numpy() == fb.label_per_receiver_only.to_numpy()).all()), f"{int((lab.label_option1.to_numpy() != fb.label_per_receiver_only.to_numpy()).sum())} differ")  # reproduction of Option 1
    check("own rule: excluded seconds and their reasons equal the pooled ones (the window set cannot move)", bool((lab.excl_own == lab.excl_pooled).all() and ((lab.label_own == "excluded") == (lab.label_pooled == "excluded")).all()), "every row")  # exclusion is evidence-free
    ev_rx = fb.evidence_receivers  # which receivers' evidence covers each second (logged)
    strict_p, tr_p, strict_o, tr_o = quiet_and_train(I, lab, own_iv)  # quiet negatives and training labels
    tr_pub = pd.read_csv(PROC / "labels" / "train_labels_1hz.csv", dtype={"test_code": str}, keep_default_na=False)  # the published training labels
    check("pooled strict quiet negatives reproduced: 4,181", len(strict_p) == 4181, f"{len(strict_p)}")  # the training-label script's count
    stop = I["stop"]  # the Stave stop
    sm = ((tr_p.test_code == "2.6.5") & tr_p.sec.between(*stop)).to_numpy()  # the stop's rows (the training-label script rewrites only their reason text)
    check("pooled training labels = labels/train_labels_1hz.csv, every row (reason outside the Stave stop)", bool((tr_p.train_label.to_numpy() == tr_pub.train_label.to_numpy()).all() and (tr_p.reason.to_numpy()[~sm] == tr_pub.reason.to_numpy()[~sm]).all()), f"{len(tr_p)} rows")  # reproduction of the training-label script
    QP = set(zip(strict_p.test_code, strict_p.receiver, strict_p.sec.astype(int)))  # set of (session, receiver, second): pooled quiet negatives
    QO = set(zip(strict_o.test_code, strict_o.receiver, strict_o.sec.astype(int)))  # own quiet negatives
    dep = dependence(lab, ev_rx, ind_own, QP, QO)  # per session x receiver
    write_csv(dep, "antenna_dependence_by_session.csv")  # written
    agg = dep[dep.test_code != SEALED].drop(columns=["test_code", "attack_only_others_share"]).groupby("receiver", sort=False).sum().reset_index()  # summed over the five training sessions
    agg["attack_only_others_share"] = (agg.attack_only_others / agg.attack_pooled).round(4)  # share of the attack seconds supported only by others
    agg["clean_lost_share_of_own"] = (agg.clean_own_lost_to_others / agg.clean_own).round(4)  # share of the own-rule clean seconds the pooled rule removes
    agg["quietneg_lost_share_of_own"] = (agg.quietneg_own_lost_to_others / agg.quietneg_own).round(4)  # the same for the quiet negatives
    agg["clean_pooled_depending_on_others_share"] = (agg.clean_pooled_not_own / agg.clean_pooled).round(4)  # pooled clean seconds that are clean only through other receivers (0 by construction)
    # boolean column: the 5 % rule per receiver (attack share and clean share both under 5 %)
    agg["under_5pct_rule"] = (agg.attack_only_others_share < 0.05) & (agg.clean_lost_share_of_own < 0.05)
    write_csv(agg, "antenna_dependence_by_receiver.csv")  # written
    check("decision rule (5 % for every receiver, attack and clean)", None, "labels stand" if agg.under_5pct_rule.all() else "FAILS: the sensitivity runs (" + "; ".join(f"{r.receiver} attack {r.attack_only_others_share:.4f}, clean {r.clean_lost_share_of_own:.4f}" for r in agg.itertuples()) + ")")  # the branch
    rr = []  # runs of attack seconds supported only by others
    for (t, rx), x in lab.groupby(["test_code", "receiver"], sort=False):  # for loop over the pairs
        s = x.sec[x.label_pooled.isin(ATT) & ~x.label_option1.isin(ATT)].to_numpy()  # those seconds
        for a, b in runs_text(s):  # for loop over their runs
            ev_here = ";".join(sorted(set(";".join(ev_rx[(lab.test_code == t).to_numpy() & (lab.receiver == rx).to_numpy() & lab.sec.between(a, b).to_numpy()]).split(";")) - {""}))  # the receivers whose evidence covers the run
            rr.append({"test_code": t, "receiver": rx, "start_utc": hms(a), "end_utc": hms(b), "seconds": int(b - a + 1), "evidence_from": ev_here})  # one run
    write_csv(pd.DataFrame(rr), "antenna_only_others_runs.csv")  # written
    se = pd.read_csv(PROC / SPOOF_END_2610)  # the logged end of spoofed tracking per receiver in 2.6.10 (from its own residuals)
    rows_c = []  # 2.6.10's own-rule clean seconds the pooled rule labels otherwise
    for rx in ("SBF_02", "SBF_03"):  # the two Septentrio receivers
        x = lab[(lab.test_code == "2.6.10") & (lab.receiver == rx) & (lab.label_own == "clean") & (lab.label_pooled != "clean")]  # those seconds
        end = int(se[se.receiver == rx].end_sec.iloc[0])  # the receiver's own last spoofed second
        for a, b in runs_text(x.sec.to_numpy()):  # for loop over their runs
            # rows_c.append adds a dict: one run of own-rule clean seconds with the pooled labels there and the seconds up to the receiver's own spoof end
            rows_c.append({"receiver": rx, "start_utc": hms(a), "end_utc": hms(b), "seconds": int(b - a + 1), "pooled_labels": ";".join(f"{k}={v}" for k, v in x[x.sec.between(a, b)].label_pooled.value_counts().sort_index().items()),
                           "own_spoofed_tracking_end_utc": hms(end), "seconds_at_or_before_own_spoof_end": int((x.sec.between(a, b) & (x.sec <= end)).sum())})  # one run
    write_csv(pd.DataFrame(rows_c), "antenna_own_clean_2610.csv")  # written
    for r in rows_c:  # for loop over the runs, reported
        check(f"2.6.10 {r['receiver']}: own-rule clean seconds the pooled rule does not call clean", None, f"{r['start_utc']}-{r['end_utc']} ({r['seconds']} s, pooled {r['pooled_labels']}); {r['seconds_at_or_before_own_spoof_end']} at or before its own spoofed-tracking end {r['own_spoofed_tracking_end_utc']}")  # the caveat
    # the per-receiver label product: pooled and own labels side by side, with both training labels and the own training reason
    prod = pd.DataFrame({"test_code": fb.test_code, "receiver": fb.receiver, "sec": fb.sec, "utc": fb.utc, "label_pooled": lab.label_pooled.to_numpy(), "label_per_receiver": lab.label_own.to_numpy(),
                         "exclusion_reason": fb.exclusion_reason, "in_scheduled_window": fb.in_scheduled_window, "train_label_pooled": tr_p.train_label.to_numpy(),  # continued: more fields of the row
                         "train_label_per_receiver": tr_o.train_label.to_numpy(), "train_reason_per_receiver": tr_o.reason.to_numpy()})  # the table
    sm_o = ((prod.test_code == "2.6.5") & prod.sec.between(*stop)).to_numpy()  # the stop's rows
    prod.loc[sm_o, "train_reason_per_receiver"] = "Stave stop: training label = scoring label (" + prod.label_per_receiver[sm_o] + ")"  # the training-label script's reason text inside the stop, applied to the own labels
    ok_tl = bool((((prod.train_label_per_receiver == "attack") == prod.label_per_receiver.isin(ATT)) | (prod.test_code == SEALED)).all() and (prod.train_label_per_receiver[(prod.label_per_receiver == "clean") & (prod.test_code != SEALED)] == "clean").all() and (prod.train_label_per_receiver[prod.test_code == SEALED] == "none").all())  # the training-label script's three invariants on the own set
    check("own training labels: attack = own scoring attack, own clean is training clean, 2.6.12 none", ok_tl, "every row")  # the invariants
    write_csv(prod, "labels_per_receiver_1hz.csv")  # written
    cnt = []  # seconds per label, both sets
    for col, name in (("label_pooled", "pooled"), ("label_per_receiver", "per receiver (own evidence and indicators)")):  # for loop over the two sets
        c = prod.groupby(["test_code", "receiver"], sort=False)[col].value_counts().unstack(fill_value=0).reindex(columns=L.LABELS, fill_value=0).reset_index()  # value_counts then unstack: one column per label
        c.insert(0, "rule", name)  # the set named first
        cnt.append(c)  # kept
    for col, name in (("train_label_pooled", "training, pooled"), ("train_label_per_receiver", "training, per receiver")):  # for loop over the two training sets
        c = prod.groupby(["test_code", "receiver"], sort=False)[col].value_counts().unstack(fill_value=0).reindex(columns=["clean", "attack", "none"], fill_value=0).reset_index()  # the same for training labels
        c.insert(0, "rule", name)  # named
        cnt.append(c)  # kept
    write_csv(pd.concat(cnt, ignore_index=True), "labels_per_receiver_counts.csv")  # written
    pd.DataFrame(CHECKS).to_csv(OUT / "antenna_checks.csv", index=False, lineterminator="\n")  # the checks
    print(agg.to_string(index=False))  # the per-receiver table for the run log


if __name__ == "__main__":  # if: true only when the file is run directly
    main()  # runs check 4
    exit_if_failed(FAILED, "8_data_checks/03_antenna_dependence.py")  # the helpers module: status 1 when any check failed
