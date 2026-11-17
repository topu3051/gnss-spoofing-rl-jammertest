#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
Do the satellite-count columns drive u-blox 01's within-receiver result in the fold that holds out 2.6.11?

The fold trains on 2.6.2, 2.6.4, 2.6.5 and 2.6.10 and holds out 2.6.11, which on u-blox 01 has attack and uncertain windows only, so
the fold reports recall and delay. The same columns drive u-blox 01's alarms on 2.6.12's clean pre-test (the transfer study's diagnostic); this
script asks whether they also carry its result on a position session. No 2.6.12 window is read: the windows come from the adopted-pipeline script's reader,
which drops 2.6.12 at read, and fold() refuses the sealed session before anything is read (test T2 plants it).

Steps:
  a. refit P for u-blox 01, outer 2.6.11, with the transfer script's fit_p(); gate: the held-out probabilities of both models equal the committed ones
     (logistic_regression/logistic_regression_predictions_primary.csv.gz, random_forest/random_forest_predictions_primary.csv.gz) bitwise;
  b. the satellite-count columns among the 36 selected columns of every fold of every receiver (the transfer script's column_group: "counts");
  c. the exact decomposition of the regression's mean decision value on 2.6.11's attack windows, and on its uncertain windows, minus
     that on the training sessions' clean windows: coefficient x mean difference per column, grouped (identity to 1e-9);
  d. the counterfactual: the count columns set to their training-clean mean (regression) or median (forest) in the transformed space,
     and recall on 2.6.11's attack windows at 0.5 and at the fold's 0.05 operating threshold (rl_agents/training/deployable_points.csv);
  e. context from committed tables: the fold's interval delays with both onset marks (rq4/agent_marks.csv) and the predicted
     against tracked GPS satellites in 2.6.11 (transfer/unseen_attack/clean_alarm_diagnosis/visibility_summary.csv).
Reading rule, fixed before anything was computed: the counts "drive" the fold when (i) they carry at least half of the
regression's mean decision-value shift on the attack windows against training clean, and (ii) setting them to training clean lowers
the regression's recall at the 0.05 threshold by at least 0.10; "contribute" when one holds; "do not drive" when neither. The forest
is reported beside it, descriptively.

Usage   /opt/anaconda3/bin/python3 codes/7_evaluation/03_satellite_count_diagnosis.py
Writes  Data/processed/satellite_count_diagnosis/chapter5_count_diag_{columns,decomposition,recall,context,verdict,checks}.csv, _COUNT_DIAG_REPORT.txt,
        Graph/chapter5_count_diag.png
"""

from __future__ import annotations  # future import: type hints kept as text, so they cost nothing at run time

import importlib.util  # standard library: loads codes/7_evaluation/01_transfer.py, whose file name starts with a digit
import os  # standard library: sets matplotlib's cache folder before matplotlib is imported
import sys  # standard library: module registry and exit status
from pathlib import Path  # Path objects for explicit file paths

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
os.environ["MPLCONFIGDIR"] = str(PROJECT / "codes" / ".mplconfig")  # matplotlib's cache inside the project folder

import numpy as np  # NumPy: arrays of transformed values and decisions
import pandas as pd  # pandas: tables
from scipy.special import expit  # the logistic function: decision value -> probability
import matplotlib  # the plotting library; imported first so its backend can be set before pyplot loads
matplotlib.use("Agg")  # file-only backend: no window, PNG bytes independent of a screen
import matplotlib.pyplot as plt  # pyplot: the plotting interface

_spec = importlib.util.spec_from_file_location("transfer_for_satellite_count", PROJECT / "codes" / "7_evaluation" / "01_transfer.py")  # where 13 is
T13 = importlib.util.module_from_spec(_spec)  # an empty module object
sys.modules["transfer_for_satellite_count"] = T13  # registered, so its own references resolve
_spec.loader.exec_module(T13)  # the transfer script's top level (it loads the evaluation, training, environment, adopted-pipeline and regression scripts); no main() is run
A, M, R = T13.A, T13.M, T13.A.R  # the adopted-pipeline script (the adopted pipeline), 09 (the regression's helpers), 10 (the forest's helpers)

PROC = PROJECT / "Data" / "processed"  # the data products
OUT = PROC / "satellite_count_diagnosis"  # this diagnostic's folder (apart from rl_diagnosis/)
FIG = PROJECT / "Graph" / "chapter5_count_diag.png"  # the records figure
RX, HELD = "UBX_01", "2.6.11"  # the receiver and the held-out session of the question
SEALED = "2.6.12"  # the session never read here
SHARE = 0.05  # the operating share whose threshold condition (ii) uses
DRIVE_SHARE, DRIVE_RECALL = 0.5, 0.10  # the reading rule's two lines: half of the shift; a recall drop of 0.10
GROUP_ORDER = ["counts", "C/N0 level", "raw Doppler", "other level", "change"]  # the groups of the decomposition, in report order
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})  # dict: one font and size for every label


def group_of(name: str) -> str:  # function: a selected column's group for the decomposition, from the transfer script's column_group()
    g1, g2 = T13.column_group(name)  # tuple assignment: (level / change / other, the kind)
    if g1 == "change":  # a change column
        return "change"  # one group for every change column
    if g2 in ("counts", "C/N0 level", "raw Doppler"):  # a level column of a named kind
        return g2  # its kind
    return "other level"  # every other level column (the transfer script's "other", e.g. the missing indicator or d_n_tracked)


def fold(held: str) -> tuple:  # function: the held-out session's guard and the fold's training sessions; refuses 2.6.12 before any read
    if held == SEALED:  # if: the sealed session was asked for
        raise SystemExit(f"STOP: {SEALED} is not read by this diagnostic")  # refusal before anything is read
    return held, [s for s in T13.SESSIONS if s != held]  # list comprehension: the four training sessions


def recall_at(p: np.ndarray, att: np.ndarray, thr: float) -> float:  # function: share of attack windows whose probability reaches the threshold
    return float((p[att] >= thr).mean())  # boolean mean = share in alarm (the regression script's rule: p >= threshold)


def main() -> None:  # function: the whole diagnostic
    OUT.mkdir(parents=True, exist_ok=True)  # the output folder
    checks = []  # one row per check
    chk = lambda name, ok, detail: checks.append({"check": name, "pass": bool(ok), "detail": str(detail)})  # lambda: records one check
    held, train = fold(HELD)  # tuple assignment: the guard passes for 2.6.11; the training sessions
    try:  # try: T2, the planted sealed session must be refused
        fold(SEALED)  # planted request
        refused = False  # not refused (a failure)
    except SystemExit:  # the refusal
        refused = True  # refused, as required
    chk("T2: a planted request for 2.6.12 is refused before any read", refused, "fold('2.6.12') raised SystemExit")  # recorded
    A.use_labels("final")  # the scoring module points at the final label set
    wc10 = A.window_dict()  # the feature script's window dictionary at w = 10
    cells = A.d1_cells(A.load_cells(A.FINAL, A.W, True), wc10)  # the final windows, D1 columns; the adopted-pipeline script's reader drops 2.6.12 at read
    chk("no 2.6.12 window in any receiver's cells", all(not (ix.test_code == SEALED).any() for ix, _ in cells.values()), "the adopted-pipeline script's reader")  # generator in all(): recorded
    idx, X = cells[RX]  # u-blox 01's windows
    te = A.eval_rows(idx, held)  # the held-out rows in time order (the regression script's fold_rows)
    rec, f = T13.fit_p(idx, X, wc10, train, {held: te}, {"run": "W5diag", "outer": held, "inner": "", "receiver": RX})  # a. the fold's P refitted, both models kept

    # a. gate: the refit's held-out probabilities equal the committed ones bitwise
    for mdl, path in (("lr", PROC / "logistic_regression" / "logistic_regression_predictions_primary.csv.gz"), ("rf", PROC / "random_forest" / "random_forest_predictions_primary.csv.gz")):  # for loop over the two committed files
        c = A.read_round_trip(path)  # committed predictions, floats as written
        c = c[(c.model == mdl) & (c.receiver == RX) & (c.held_out == held)].sort_values("sec")  # this fold's rows
        new_p, new_s = f["p"][mdl][held], idx.sec.to_numpy()[te]  # tuple assignment: the refit's probabilities and their seconds
        same = np.array_equal(c.sec.to_numpy(), new_s) and np.array_equal(c.p.to_numpy(), new_p)  # bitwise equality of seconds and probabilities
        chk(f"a. refit {mdl} held-out probabilities = {path.name} bitwise", same, f"{len(new_p)} windows")  # recorded
        if not same:  # if: the refit differs
            raise SystemExit(f"STOP: the {mdl} refit differs from the committed fold")  # stop before any reading

    # b. the count columns among every fold's 36 selected columns
    sel = pd.read_csv(PROC / "baselines_controls" / "adopted_selection_kept_columns.csv", dtype={"held_out": str})  # P's selections per fold
    sel = sel[sel.run_id == "P"]  # the adopted primary's rows
    col_rows = []  # one row per receiver and fold
    for _, r in sel.iterrows():  # for loop over the folds
        names = r.selected_columns.split(";")  # the 36 selected columns
        cnt = [n for n in names if group_of(n) == "counts"]  # list comprehension: the satellite-count columns
        col_rows.append({"receiver": r.receivers, "held_out": r.held_out, "selected": len(names), "count_columns": len(cnt), "names": ";".join(cnt)})  # the row
    cols = pd.DataFrame(col_rows)  # the table
    row = cols[(cols.receiver == RX) & (cols.held_out == held)].iloc[0]  # this fold's row
    chk("b. the refit selects the committed 36 columns in order", row["names"].split(";") == [n for n in rec["names"] if group_of(n) == "counts"] and len(rec["names"]) == row.selected, f"{row.count_columns} count columns")  # recorded

    # c. the exact decomposition of the regression's decision value
    names = list(rec["names"])  # the 36 columns in the coefficients' order
    w = rec["model"]["lr"].coef_.ravel()  # the regression's 36 coefficients
    grp = np.array([group_of(n) for n in names])  # list comprehension in an array: each column's group
    Z_all = A.transform(rec, X, wc10)  # every u-blox 01 window through the fold's fitted transform
    d_all = A.decision(rec["model"]["lr"], Z_all)  # the regression's decision values on every window
    held_mask = np.zeros(len(idx), bool)  # boolean array: the held-out rows
    held_mask[te] = True  # set
    chk("c. transform + decision reproduce the fold's held-out probabilities bitwise", np.array_equal(expit(d_all[te]), f["p"]["lr"][held]), "expit(decision)")  # recorded
    lab, tc = idx.label.to_numpy(), idx.test_code.to_numpy()  # tuple assignment: the scoring labels and the sessions
    ref = np.isin(tc, train) & (lab == "clean")  # the training sessions' clean windows (the reference)
    targets = {"attack": held_mask & np.isin(lab, A.ATTACK), "uncertain": held_mask & (lab == "uncertain")}  # dict: the two held-out window sets
    dec_rows = []  # the decomposition rows
    for tname, tm in targets.items():  # for loop over the two window sets
        dm = Z_all[tm].mean(axis=0) - Z_all[ref].mean(axis=0)  # each column's mean difference
        con = w * dm  # coefficient times mean difference: each column's contribution
        obs = float(d_all[tm].mean() - d_all[ref].mean())  # the observed shift of the mean decision value
        chk(f"c. {tname}: sum of contributions = observed shift (1e-9)", abs(float(con.sum()) - obs) <= 1e-9, f"{abs(float(con.sum()) - obs):.2e}")  # recorded
        for g in GROUP_ORDER:  # for loop over the groups
            m_ = grp == g  # the group's columns
            dec_rows.append({"target": tname, "n_target": int(tm.sum()), "n_reference": int(ref.sum()), "group": g, "columns": int(m_.sum()), "contribution": float(con[m_].sum()), "observed_shift": obs, "share_of_shift": float(con[m_].sum() / obs)})  # the row
        for j in np.argsort(-np.abs(con), kind="stable")[:5]:  # for loop over the five largest contributions by size
            dec_rows.append({"target": tname, "n_target": int(tm.sum()), "n_reference": int(ref.sum()), "group": "column: " + names[j], "columns": 1, "contribution": float(con[j]), "observed_shift": obs, "share_of_shift": float(con[j] / obs)})  # the row
    dec = pd.DataFrame(dec_rows)  # the table

    # d. the counterfactual recall
    thr = pd.read_csv(PROC / "rl_agents" / "training" / "deployable_points.csv", dtype={"outer": str}, float_precision="round_trip")  # the frozen operating points
    thr = thr[(thr.receiver == RX) & (thr.outer == held) & (thr.share == SHARE)].set_index("detector")["param"]  # the fold's thresholds at 0.05
    cmask = grp == "counts"  # the count columns
    att = targets["attack"][te]  # the attack windows among the held-out rows (time order)
    unc = targets["uncertain"][te]  # the uncertain windows among them (descriptive only: the alarm share before the onsets, outside the rule)
    Zh = Z_all[te].copy()  # the held-out windows, transformed
    Zl, Zf = Zh.copy(), Zh.copy()  # tuple assignment: copies for the regression's and the forest's counterfactual
    Zl[:, cmask] = Z_all[ref][:, cmask].mean(axis=0)  # the counts at their training-clean mean
    Zf[:, cmask] = np.median(Z_all[ref][:, cmask], axis=0)  # the counts at their training-clean median
    p_l, p_l0 = expit(A.decision(rec["model"]["lr"], Zh)), expit(A.decision(rec["model"]["lr"], Zl))  # tuple assignment: regression probabilities as fitted and with counts neutral
    p_f, p_f0 = R.proba(rec["model"]["rf"], Zh), R.proba(rec["model"]["rf"], Zf)  # tuple assignment: forest probabilities as fitted and with counts neutral
    met = {}  # committed recall at 0.5, per model
    for mdl, path in (("lr", PROC / "logistic_regression" / "logistic_regression_metrics_primary.csv"), ("rf", PROC / "random_forest" / "random_forest_metrics_primary.csv")):  # for loop over the committed metric tables
        m_ = pd.read_csv(path, dtype={"held_out": str})  # the table
        met[mdl] = float(m_[(m_.model.isin([mdl, {"lr": "logreg", "rf": "forest"}[mdl]])) & (m_.receiver == RX) & (m_.held_out == held)].recall.iloc[0])  # nested dict lookup: the fold's recall
    rec_rows = []  # the recall rows
    for mdl, p0, p1 in (("lr", p_l, p_l0), ("rf", p_f, p_f0)):  # for loop over the two models
        for tname, t in (("0.5", 0.5), ("share 0.05", float(thr[mdl]))):  # for loop over the two thresholds
            r0, r1 = recall_at(p0, att, t), recall_at(p1, att, t)  # tuple assignment: recall as fitted and with counts neutral
            rec_rows.append({"model": mdl, "threshold_kind": tname, "threshold": t, "attack_windows": int(att.sum()), "recall": r0, "recall_counts_neutral": r1, "recall_drop": r0 - r1, "uncertain_windows": int(unc.sum()), "uncertain_alarm_share": recall_at(p0, unc, t), "uncertain_alarm_share_counts_neutral": recall_at(p1, unc, t)})  # the row
        chk(f"d. {mdl} recall at 0.5 as fitted = the committed fold recall", recall_at(p0, att, 0.5) == met[mdl], f"{recall_at(p0, att, 0.5)} vs {met[mdl]}")  # recorded
    recl = pd.DataFrame(rec_rows)  # the table

    # e. context from committed tables
    mk = pd.read_csv(PROC / "rq4" / "agent_marks.csv", dtype={"outer": str})  # the agent experiment's interval runs with both marks
    mk = mk[(mk.receiver == RX) & (mk.outer == held) & (mk.share == SHARE)]  # this fold at 0.05
    ctx_rows = [{"item": f"interval {int(r.onset)} {r.detector}" + (f" seed {int(r.seed)}" if r.seed >= 0 else ""), "value": r.delay_s, "detail": f"detected {r.detected}; onset state {r.onset_state}; refractory {r.refractory_at_onset}; no detection delay {r.no_detection_delay}"} for r in mk.itertuples()]  # list comprehension: one row per interval run
    vis = pd.read_csv(PROC / "transfer" / "unseen_attack" / "clean_alarm_diagnosis" / "visibility_summary.csv")  # predicted against tracked satellites per group
    for g in ("2.6.11 attack seconds", "2.6.11 all seconds"):  # for loop over 2.6.11's groups
        v = vis[(vis.receiver == RX) & (vis.group == g)].iloc[0]  # the row
        ctx_rows.append({"item": f"GPS predicted above 5 deg, {g}", "value": float(v.predicted_G_above_5_median), "detail": "median over minutes"})  # predicted
        ctx_rows.append({"item": f"GPS tracked (window mean), {g}", "value": float(v.tracked_G_window_mean), "detail": "u-blox 01 windows"})  # tracked
    ctx = pd.DataFrame(ctx_rows)  # the table

    # the reading rule
    share_c = float(dec[(dec.target == "attack") & (dec.group == "counts")].share_of_shift.iloc[0])  # (i) the counts' share of the attack-window shift
    drop_c = float(recl[(recl.model == "lr") & (recl.threshold_kind == "share 0.05")].recall_drop.iloc[0])  # (ii) the regression's recall drop at the 0.05 threshold
    c1, c2 = share_c >= DRIVE_SHARE, drop_c >= DRIVE_RECALL  # tuple assignment: the two conditions
    verdict = "drive" if (c1 and c2) else ("contribute" if (c1 or c2) else "do not drive")  # conditional expression: the rule's outcome
    ver = pd.DataFrame([{"receiver": RX, "held_out": held, "count_columns": int(cmask.sum()), "share_of_shift": share_c, "condition_i": c1, "recall_drop_lr_share_0.05": drop_c, "condition_ii": c2, "verdict": verdict}])  # the one-row table

    # outputs, rounded nowhere (floats as Python writes them)
    cols.to_csv(OUT / "columns.csv", index=False, lineterminator="\n")  # b
    dec.to_csv(OUT / "decomposition.csv", index=False, lineterminator="\n")  # c
    recl.to_csv(OUT / "recall.csv", index=False, lineterminator="\n")  # d
    ctx.to_csv(OUT / "context.csv", index=False, lineterminator="\n")  # e
    ver.to_csv(OUT / "verdict.csv", index=False, lineterminator="\n")  # the rule's outcome
    figure(dec, recl)  # the records figure
    pd.DataFrame(checks).to_csv(OUT / "checks.csv", index=False, lineterminator="\n")  # the checks
    report(cols, dec, recl, ctx, ver, checks)  # the text report
    bad = [c for c in checks if not c["pass"]]  # list comprehension: failed checks
    print(f"verdict: {verdict}; share {share_c:.3f}; recall drop {drop_c:.3f}; checks {len(checks) - len(bad)}/{len(checks)}")  # summary line
    sys.exit(1 if bad else 0)  # exit status


def figure(dec: pd.DataFrame, recl: pd.DataFrame) -> None:  # function: Graph/chapter5_count_diag.png, the decomposition and the recall with and without the counts
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.4, 1]})  # two panels
    for k, t in enumerate(("attack", "uncertain")):  # for loop over the two window sets
        d = dec[(dec.target == t) & dec.group.isin(GROUP_ORDER)].set_index("group").loc[GROUP_ORDER]  # the grouped rows in order
        a1.barh(np.arange(len(GROUP_ORDER)) + (k - 0.5) * 0.38, d.contribution, height=0.36, color=["#7A0019", "#C8C8C8"][k], label=f"2.6.11 {t} windows")  # bars: contribution per group
    a1.set_yticks(np.arange(len(GROUP_ORDER)), GROUP_ORDER)  # group names
    a1.invert_yaxis()  # first group on top
    a1.axvline(0, color="black", lw=0.6)  # zero line
    a1.set_xlabel("contribution to the shift of the mean\ndecision value against training clean")  # axis label
    a1.legend(frameon=False, fontsize=7.5)  # legend
    r = recl.copy()  # the recall rows
    lbl = [f"{m} {k}" for m, k in zip(r.model, r.threshold_kind)]  # list comprehension: row labels
    y = np.arange(len(r))  # positions
    a2.barh(y - 0.19, r.uncertain_alarm_share, height=0.36, color="#0072B2", label="as fitted")  # uncertain alarm share as fitted
    a2.barh(y + 0.19, r.uncertain_alarm_share_counts_neutral, height=0.36, color="#E69F00", label="counts at training clean")  # uncertain alarm share with counts neutral
    a2.set_yticks(y, lbl)  # labels
    a2.invert_yaxis()  # first row on top
    a2.set_xlim(0, 1.05)  # recall axis
    a2.set_xlabel(f"share of 2.6.11 uncertain windows in alarm\n(attack recall {r.recall.min():.2f} as fitted, {r.recall_counts_neutral.min():.2f} with counts neutral)", fontsize=7.5)  # axis label, with the attack recall range read from the table
    a2.legend(frameon=False, fontsize=7.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)  # legend above the panel
    fig.tight_layout()  # spacing
    fig.savefig(FIG, dpi=200, metadata={"Software": None})  # PNG; no software tag, so runs again give the same bytes
    plt.close(fig)  # free memory


def report(cols, dec, recl, ctx, ver, checks) -> None:  # function: _COUNT_DIAG_REPORT.txt from the tables just written
    v = ver.iloc[0]  # the verdict row
    lines = ["COUNT DIAGNOSTIC: u-blox 01, fold holding out 2.6.11 (no 2.6.12 window read)", "=" * 100, ""]  # header lines
    lines.append(f"Reading rule (fixed before anything was computed): drive when the counts carry >= {DRIVE_SHARE} of the regression's attack-window shift AND neutralising them lowers its recall at the {SHARE} threshold by >= {DRIVE_RECALL}.")  # the rule
    lines.append(f"Result: count columns {v.count_columns} of 36; share of shift {v.share_of_shift:.4f} (condition i {v.condition_i}); recall drop {v['recall_drop_lr_share_0.05']:.4f} (condition ii {v.condition_ii}); VERDICT: {v.verdict}.")  # the outcome
    lines += ["", "Count columns selected per fold:"] + [f"  {r.receiver} {r.held_out}: {r.count_columns} ({r.names})" for r in cols.itertuples()]  # list comprehension: per fold
    lines += ["", "Decomposition (contribution, share of shift):"] + [f"  {r.target:9s} {r.group:40s} {r.contribution:+.4f}  {r.share_of_shift:+.4f}  (shift {r.observed_shift:+.4f}; n {r.n_target} vs {r.n_reference})" for r in dec.itertuples()]  # list comprehension: decomposition rows
    lines += ["", "Recall on 2.6.11's attack windows:"] + [f"  {r.model} at {r.threshold_kind} ({r.threshold:.6g}): {r.recall:.4f} -> {r.recall_counts_neutral:.4f} (drop {r.recall_drop:+.4f})" for r in recl.itertuples()]  # list comprehension: recall rows
    lines += ["", "Descriptive, outside the rule: share of 2.6.11's uncertain windows in alarm:"] + [f"  {r.model} at {r.threshold_kind}: {r.uncertain_alarm_share:.4f} -> {r.uncertain_alarm_share_counts_neutral:.4f} ({r.uncertain_windows} windows)" for r in recl.itertuples()]  # list comprehension: the uncertain alarm shares
    lines += ["", "Context:"] + [f"  {r.item}: {r.value} {r.detail}" for r in ctx.itertuples()]  # list comprehension: context rows
    lines += ["", "Checks:"] + [f"  {'PASS' if c['pass'] else 'FAIL'} {c['check']}: {c['detail']}" for c in checks]  # list comprehension: checks
    (OUT / "CHAPTER5_COUNT_DIAG_REPORT.txt").write_text("\n".join(lines) + "\n")  # written


if __name__ == "__main__":  # if: run as a script, not imported
    main()  # the diagnostic
