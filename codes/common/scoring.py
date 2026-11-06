#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
The scoring rules every detector is judged by. A small importable module:
nothing runs on import. Run directly, it writes the fold policy table from the registered labels.

Rules:
  * Per-class precision, recall and F1 are computed on clean and attack seconds only. An alarm in an uncertain or excluded
    second counts as neither true nor false; such alarms are counted separately. Uncertain seconds are masked from scoring and
    reward but stay in the time series, because a sequential model needs the history before an onset.
  * Detection latency is measured from the start of the pooled attack interval (the independent onset). A negative latency
    counts only if the alarm run is unbroken from its own start to the onset; earlier alarm runs that end before the onset
    are reported separately as early alarms and never as detections. An alarm first raised after the interval ends is a miss.
    A lead is never counted back past the end of the previous pooled interval of the same session.
  * False alarms are episodes per clean hour: an episode is a maximal run of consecutive alarm seconds that contains at least
    one clean second, so a run that crosses an uncertain stretch between two clean ones counts once. The bound is the exact
    one-sided Poisson 95 % upper limit on the count (chi-square, 2n + 2 degrees of freedom; 2.996 at zero, the rule of three).
  * With fewer than MIN_RATE_S (300) clean seconds, the count and its bound are reported and no rate is read; with none, no
    false-alarm figure is reported.
  * Thresholds for a target false-alarm rate are set on training folds only (matched comparisons).
  * Fold policy: precision for the folds holding out 2.6.2 and 2.6.4, and
    for 2.6.12 per receiver with the Stave limitation named beside it; false-alarm counts over stated clean
    seconds for 2.6.5 and 2.6.10; nothing for 2.6.11.
  * In a fold without a readable false-alarm rate, latency is reported only at a threshold set on other folds by
    threshold_for_rate (evaluate() refuses otherwise), and every latency is shown with the share of pre-onset uncertain seconds
    in alarm, so a lead bought by alarming all the time is visible.

Reads   (when run) Data/processed/labels/labels_1hz.csv, labels/labels_evidence.csv (with --labels final: labels/labels_1hz_final.csv, labels/labels_evidence_final.csv)
Writes  (when run) Data/processed/labels/scoring_fold_policy.csv (with --labels final: labels/scoring_fold_policy_final.csv)
"""

from __future__ import annotations  # future import: type hints kept as text (the project's convention)

from pathlib import Path  # Path objects for building file paths

import numpy as np  # NumPy: boolean arrays over seconds
import pandas as pd  # pandas: reading the label tables
from scipy.stats import chi2  # the chi-square distribution, for the exact Poisson bound

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
PROC = PROJECT / "Data" / "processed"  # the processed-data folder
ATTACK = ("attack_jam_spoof", "attack_spoof")  # tuple: the two attack labels
SCORED = ("clean",) + ATTACK  # tuple concatenation: the labels that are scored
MIN_RATE_S = 300  # integer constant: fewer clean seconds than this and no false-alarm rate is read
CONF = 0.95  # float constant: confidence of the Poisson upper bound
PRECISION_FOLDS = ("2.6.2", "2.6.4", "2.6.12")  # tuple: folds that report precision ; 2.6.12 per receiver, 6 October)
FA_COUNT_FOLDS = ("2.6.5", "2.6.10")  # tuple: folds that report false-alarm counts over stated clean seconds only
NO_FA_FOLDS = ("2.6.11",)  # tuple: the fold with no false-alarm figure at all (no clean margin)
# the label set the module reads; use_label_set("final") switches to the files with the "_b" suffix
FILES = {"labels": "labels/labels_1hz.csv", "evidence": "labels/labels_evidence.csv", "policy": "labels/scoring_fold_policy.csv"}  # dict: role -> file name (registered set by default)


def as_arrays(labels, alarms) -> tuple:  # function: labels and alarms as NumPy arrays of equal length
    lab = np.asarray(labels, dtype=object)  # labels as an object array (text)
    al = np.asarray(alarms, dtype=bool)  # alarms as booleans
    if lab.shape != al.shape:  # if: the lengths differ
        raise ValueError("labels and alarms must have the same length")  # stop: a mismatch would silently misalign seconds
    return lab, al  # the two arrays


def scoring_mask(labels) -> np.ndarray:  # function: True where a second is scored and enters the reward (clean or attack)
    return np.isin(np.asarray(labels, dtype=object), SCORED)  # np.isin: membership test per element


def runs(sec, flag) -> list:  # function: maximal runs of consecutive seconds where flag is True, as (first index, last index)
    s = np.asarray(sec, dtype=np.int64)  # seconds as integers
    f = np.asarray(flag, dtype=bool)  # the flag
    out, i, n = [], 0, len(f)  # result list, position, count
    while i < n:  # while loop over positions
        if f[i]:  # if: a run starts here
            j = i  # its last position so far
            while j + 1 < n and f[j + 1] and s[j + 1] - s[j] == 1:  # extends while the next second is flagged and adjacent
                j += 1  # one on
            out.append((i, j))  # the run
            i = j + 1  # continue after it
        else:  # else: not flagged
            i += 1  # next position
    return out  # the runs


def confusion(labels, alarms) -> dict:  # function: true/false positives and negatives on scored seconds, and the alarms that are neither
    lab, al = as_arrays(labels, alarms)  # arrays
    att = np.isin(lab, ATTACK)  # attack seconds
    cln = lab == "clean"  # clean seconds
    # dict literal over three lines: the four cells, plus alarms in uncertain and excluded seconds (counted, never scored)
    return {"tp": int((al & att).sum()), "fn": int((~al & att).sum()), "fp": int((al & cln).sum()), "tn": int((~al & cln).sum()),
            # dict entry: alarms raised in uncertain and in excluded seconds, counted but never scored
            "alarms_uncertain": int((al & (lab == "uncertain")).sum()), "alarms_excluded": int((al & (lab == "excluded")).sum()),
            "attack_s": int(att.sum()), "clean_s": int(cln.sum())}  # dict entry: the number of attack and of clean seconds, the denominators of the rates


def _ratio(a: int, b: int) -> float:  # function: a / b, or NaN when b is zero (an undefined value is never shown as 0 or 1)
    return a / b if b else float("nan")  # conditional expression


def _f1(p: float, r: float, tp: int, n_pos: int) -> float:  # function: F1 = 2TP / (2TP + FP + FN) of one class: 0 when the class has instances and no true positive, NaN only when the class has none
    if n_pos == 0:  # if: the class has no instance in the fold (TP + FN = 0): F1 is undefined
        return float("nan")  # undefined, never shown as 0 or 1
    if tp == 0:  # if: instances but no true positive (no alarm at all, or only wrong ones): 2TP = 0 over a positive denominator
        return 0.0  # zero, not undefined (the old rule returned NaN here and the interquartile means then dropped the run)
    return 2 * p * r / (p + r)  # TP > 0: the harmonic mean of precision and recall, the same double as before


def per_class(labels, alarms) -> dict:  # function: precision, recall and F1 for the attack class (alarm) and the clean class (no alarm)
    c = confusion(labels, alarms)  # the cells
    pa, ra = _ratio(c["tp"], c["tp"] + c["fp"]), _ratio(c["tp"], c["tp"] + c["fn"])  # attack precision and recall
    pc, rc = _ratio(c["tn"], c["tn"] + c["fn"]), _ratio(c["tn"], c["tn"] + c["fp"])  # clean precision and recall (no alarm = predicted clean)
    # dict literal: the six values plus the cells they came from; : each F1 also takes its class's true positives and instances (attack: tp, tp + fn; clean: tn, tn + fp)
    return {"attack_precision": pa, "attack_recall": ra, "attack_f1": _f1(pa, ra, c["tp"], c["tp"] + c["fn"]), "clean_precision": pc, "clean_recall": rc, "clean_f1": _f1(pc, rc, c["tn"], c["tn"] + c["fp"]), **c}


def poisson_upper(n: int, conf: float = CONF) -> float:  # function: exact one-sided upper confidence limit on a Poisson count
    return float(chi2.ppf(conf, 2 * n + 2) / 2.0)  # the chi-square quantile with 2n + 2 degrees of freedom, halved (Garwood)


def false_alarm_episodes(sec, labels, alarms) -> list:  # function: alarm runs that contain at least one clean second
    lab, al = as_arrays(labels, alarms)  # arrays
    return [(i, j) for i, j in runs(sec, al) if (lab[i:j + 1] == "clean").any()]  # list comprehension: runs touching a clean second


def false_alarms(sec, labels, alarms) -> dict:  # function: false-alarm episodes, their bound, and a rate only where one can be read
    lab, _ = as_arrays(labels, alarms)  # arrays (validates the lengths)
    n = len(false_alarm_episodes(sec, labels, alarms))  # episodes
    cs = int((lab == "clean").sum())  # clean seconds
    up = poisson_upper(n)  # upper bound on the count
    h = cs / 3600.0  # clean hours
    if cs == 0:  # if: no clean second
        return {"episodes": None, "clean_s": 0, "upper95_count": None, "rate_per_h": None, "upper95_per_h": None, "readable": "none", "note": "no clean seconds: no false-alarm figure"}  # nothing
    if cs < MIN_RATE_S:  # if: too few clean seconds to read a rate
        # dict literal over two lines: the count and its bound, and a note saying plainly that no rate can be read
        return {"episodes": n, "clean_s": cs, "upper95_count": up, "rate_per_h": None, "upper95_per_h": None, "readable": "count only",
                "note": f"{n} episodes in {cs} clean seconds (95 % upper bound {up:.2f}); fewer than {MIN_RATE_S} clean seconds, no rate can be read"}  # count and bound only
    # dict literal over two lines: the count, its bound, the rate per clean hour and the bound as a rate
    return {"episodes": n, "clean_s": cs, "upper95_count": up, "rate_per_h": n / h, "upper95_per_h": up / h, "readable": "rate",
            "note": f"{n} episodes in {cs} clean seconds ({h:.3f} h)"}  # count, bound and rate


def latency(sec, labels, alarms, onset: int, end: int, prev_end: int | None = None) -> dict:  # function: detection latency for one pooled attack interval
    s = np.asarray(sec, dtype=np.int64)  # seconds
    _, al = as_arrays(labels, alarms)  # alarms (validates the lengths)
    rr = runs(s, al)  # alarm runs
    lo = -np.inf if prev_end is None else prev_end  # conditional expression: early alarms are looked for after the previous interval
    det, early = None, []  # the detecting run; the early alarm runs
    for i, j in rr:  # for loop over runs in time order
        a, b = int(s[i]), int(s[j])  # run start and end seconds
        if b < onset:  # if: the run ends before the onset
            if a > lo:  # if: after the previous interval
                early.append((a, b))  # an early alarm, never a detection
            continue  # next run
        if a > end:  # if: the run starts after the interval ended
            break  # no detection in this interval
        det = (a, b)  # the first run reaching the onset or later, starting no later than the interval end
        break  # stop at the first
    if det is None:  # if: no run qualified
        return {"detected": False, "latency_s": None, "alarm_start": None, "early_alarms": early}  # a miss
    start = det[0] if prev_end is None else max(det[0], prev_end + 1)  # a lead never reaches back past the end of the previous attack interval
    lat = start - onset  # negative when the run started before the onset; it reached the onset unbroken, since it did not end before it
    return {"detected": True, "latency_s": int(lat), "alarm_start": start, "early_alarms": early}  # the detection


def alarms_from_scores(sec, scores, thr: float, persist: int = 1) -> np.ndarray:  # function: alarm per second: score at or above thr for persist consecutive seconds
    s = np.asarray(sec, dtype=np.int64)  # seconds
    # ">=" as the models alarm (09 and 10: p >= 0.5), so the scoring module and the models share one convention (was strictly ">")
    above = np.asarray(scores, dtype=float) >= thr  # at or above the threshold (NaN gives False)
    out = np.zeros(len(above), bool)  # no alarm yet
    for i, j in runs(s, above):  # for loop over runs above the threshold
        if j - i + 1 >= persist:  # if: long enough
            out[i + persist - 1:j + 1] = True  # alarm from the persist-th second of the run (an alarm cannot predate its evidence)
    return out  # the alarms


def threshold_for_rate(folds: dict, train: list, target_per_h: float, persist: int = 1) -> float:  # function: lowest threshold whose pooled training false-alarm rate, and that of every higher candidate, is at or below target
    # folds: name -> (sec, labels, scores). Only the folds named in train are read; the held-out fold is never touched.
    if not train:  # if: no training fold
        raise ValueError("at least one training fold is needed")  # stop
    cand = set()  # candidate thresholds: score values seen on clean seconds of the training folds
    clean_s = 0  # pooled clean seconds
    for k in train:  # for loop over training folds
        sec, lab, sc = folds[k]  # the fold
        lab = np.asarray(lab, dtype=object)  # labels
        sc = np.asarray(sc, dtype=float)  # scores
        # with ">=" a threshold equal to a clean score would alarm on it, so each candidate is the next double above a clean score
        # (np.nextafter toward +inf), the convention the design-choice study's inner-fold threshold uses; the alarms of every candidate are therefore the same as under the old ">" rule
        cand |= set(np.nextafter(sc[(lab == "clean") & np.isfinite(sc)], np.inf).tolist())  # "|=": adds the candidates
        clean_s += int((lab == "clean").sum())  # adds its clean seconds
    if clean_s == 0:  # if: no clean second in training
        raise ValueError("the training folds hold no clean second")  # stop: a rate cannot be set
    best = float("inf")  # start above every score: no alarm at all
    for thr in sorted(cand, reverse=True):  # for loop from the highest candidate down
        n = sum(len(false_alarm_episodes(folds[k][0], folds[k][1], alarms_from_scores(folds[k][0], folds[k][2], thr, persist))) for k in train)  # generator inside sum(): pooled episodes
        if n / (clean_s / 3600.0) > target_per_h:  # if: this threshold exceeds the target
            break  # stop: lower thresholds are not considered (the first violation from the top ends the search)
        best = thr  # still within the target: keep going down
    return best  # the threshold (alarm when the score is at or above it; it lies one double above a clean score)


def fold_policy(test: str, clean_s: int) -> dict:  # function: what a held-out fold reports, by decision and by its clean seconds
    if test in NO_FA_FOLDS:  # if: 2.6.11
        return {"precision": False, "false_alarms": "none", "note": "no clean margin; no false-alarm figure "}  # nothing
    prec = test in PRECISION_FOLDS and clean_s > 0  # precision only in the decided folds, and only with clean seconds
    fa = "none" if clean_s == 0 else ("count only" if clean_s < MIN_RATE_S else "rate")  # nested conditional expression: what the false-alarm figure can be
    if test in FA_COUNT_FOLDS and fa == "rate":  # if: a counts-only fold that would allow a rate
        fa = "count only"  # the decision keeps it a count over stated clean seconds
    note = f"{clean_s} clean seconds"  # the stated clean seconds
    if fa == "none":  # if: no figure
        note += "; no false-alarm figure"  # said
    elif fa == "count only":  # elif: count only
        note += "; count with its exact Poisson 95 % bound, no rate can be read" + (f" (fewer than {MIN_RATE_S})" if clean_s < MIN_RATE_S else "")  # said, with the reason
    if test == "2.6.12":  # if: the attack-type transfer session
        note += "; Stave clean margin limited"  # the limitation named beside the figure
    return {"precision": bool(prec), "false_alarms": fa, "note": note}  # the policy


def use_label_set(label_set: str) -> None:  # function: points the module at the registered or the final label set
    stem = {"registered": "", "final": "_final"}[label_set]  # dict lookup: the file-name suffix (a KeyError stops an unknown set)
    FILES.update(labels=f"labels/labels_1hz{stem}.csv", evidence=f"labels/labels_evidence{stem}.csv", policy=f"labels/scoring_fold_policy{stem}.csv")  # dict.update: the three names


def load_fold(test: str, receiver: str, column: str = "label") -> tuple:  # function: one held-out fold from labels/labels_1hz.csv: seconds and labels in time order
    lab = pd.read_csv(PROC / FILES["labels"], usecols=["test_code", "receiver", "sec", column], dtype={"test_code": str})  # the label table, four columns (FILES: the set in use)
    f = lab[(lab.test_code == test) & (lab.receiver == receiver)].sort_values("sec")  # the fold, in time order
    return f.sec.to_numpy(np.int64), f[column].to_numpy(object)  # seconds and labels


def pooled_intervals(test: str) -> list:  # function: the pooled attack intervals of a session, as (onset, end) unix seconds, from labels/labels_evidence.csv
    ev = pd.read_csv(PROC / FILES["evidence"], dtype={"test_code": str})  # the evidence table (FILES: the set in use)
    p = ev[(ev.test_code == test) & (ev.source == "pooled_interval")].sort_values("start")  # its pooled rows, in time order
    return [(int(a), int(b)) for a, b in zip(p.start, p.end)]  # list comprehension: (onset, end)


def pre_onset_alarm_share(sec, labels, alarms, onset: int, prev_end: int | None = None) -> dict:  # function: share of the uncertain seconds before an onset that are in alarm
    s = np.asarray(sec, dtype=np.int64)  # seconds
    lab, al = as_arrays(labels, alarms)  # arrays
    lo = -np.inf if prev_end is None else prev_end  # conditional expression: only after the previous interval
    m = (s < onset) & (s > lo) & (lab == "uncertain")  # boolean mask: pre-onset uncertain seconds
    n = int(m.sum())  # how many
    return {"pre_onset_uncertain_s": n, "pre_onset_alarm_share": (float(al[m].sum()) / n) if n else float("nan")}  # the share, NaN when there is none


def evaluate(sec, labels, alarms, intervals: list, test: str | None = None, threshold_from_other_folds: bool | None = None) -> dict:  # function: every figure for one fold: per-class scores, false alarms, a latency per interval
    # Guard: in a fold that cannot report a false-alarm rate (2.6.5, 2.6.10 and every fold under
    # MIN_RATE_S clean seconds), a latency means something only at a threshold set on other folds (threshold_for_rate), never at
    # one tuned on the fold itself; the caller must say so. Every latency carries the share of pre-onset uncertain seconds in alarm.
    fa = false_alarms(sec, labels, alarms)  # the false-alarm figure of this fold
    if test is not None and fold_policy(test, fa["clean_s"])["false_alarms"] != "rate" and threshold_from_other_folds is not True:  # if: no rate here and the threshold is not declared as set elsewhere
        raise ValueError(f"fold {test}: no false-alarm rate can be read, so latency is reported only at a threshold set on other folds (threshold_for_rate); pass threshold_from_other_folds=True")  # stop
    out = {"per_class": per_class(labels, alarms), "false_alarms": fa, "latency": []}  # dict literal: the parts
    prev = None  # end of the previous interval
    for a, b in intervals:  # for loop over the pooled intervals in time order
        out["latency"].append({"onset": a, "end": b, **latency(sec, labels, alarms, a, b, prev), **pre_onset_alarm_share(sec, labels, alarms, a, prev)})  # dict unpacking: interval, latency and the pre-onset alarm share beside it
        prev = b  # remembered
    return out  # the figures


def main() -> None:  # function: writes the fold policy table from the registered labels
    lab = pd.read_csv(PROC / FILES["labels"], usecols=["test_code", "receiver", "label"], dtype={"test_code": str})  # the labels (FILES: the set in use)
    rows = []  # one row per session and receiver
    for (t, rx), g in lab.groupby(["test_code", "receiver"], sort=False):  # for loop over folds in table order
        cs, at = int((g.label == "clean").sum()), int(g.label.isin(ATTACK).sum())  # clean and attack seconds
        p = fold_policy(t, cs)  # its policy
        rows.append({"test_code": t, "receiver": rx, "clean_s": cs, "attack_s": at, "uncertain_s": int((g.label == "uncertain").sum()), "excluded_s": int((g.label == "excluded").sum()), **p})  # dict unpacking: counts and policy
    out = pd.DataFrame(rows)  # the table
    out.to_csv(PROC / FILES["policy"], index=False, lineterminator="\n")  # written (labels/scoring_fold_policy.csv, or _b for the final set)
    print(out.to_string(index=False))  # printed for the report


if __name__ == "__main__":  # if: true only when the file is run directly
    import sys  # system access: the command-line arguments (imported here so the module's import lines stay as they were)
    if sys.argv[1:] == ["--labels", "final"]:  # if: the final set
        use_label_set("final")  # the "_b" files
    elif sys.argv[1:] not in ([], ["--labels", "registered"]):  # elif: anything but the registered set
        raise SystemExit("usage: common/scoring.py [--labels registered|final]")  # stop with the usage
    main()  # writes the fold policy table
