#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
The sequential detection environment, its material and the evaluation stream (the reinforcement-learning formulation).

Checks C1 to C5 of this script:
  C1 episode material per receiver: every second's step type (pre-change = training label clean, which includes the strict quiet
     negatives, post-change = attack, uncertain = none and not excluded), its window run and validity (the windows ending t-29 .. t lie in
     one contiguous run), the attack runs with their onsets, and per outer fold the usable onsets and the miss penalty m (the median length
     of the usable onset intervals of the fold's training sessions).
  C2 states for every valid second of the five sessions under every outer fold: primary = the fold's 36 fitted, selected columns at the
     windows ending t, t-10 and t-20 (108 values); surrogate = the logistic regression's decision values (log-odds; balanced class weights
     make the prior odds 1, so nothing is subtracted) at t-29 .. t (30 values), to which the environment appends the cumulative sum S_t,
     which restarts with every episode. Training sessions take the inner out-of-fold scores, the held-out session the outer model's.
     Every logistic-regression score is computed on a C-ordered matrix (the adopted-pipeline script's transform), which reproduces the
     stored probabilities bitwise.
  C3 SpoofingDetectionEnv (gymnasium): Discrete(2) actions, a float64 Box observation, rewards in the Shiryaev cost form with no positive
     reward; a deterministic table of 2,000 episodes per receiver and fold from seed 20260914, each replayable through reset(options=...).
  C4 the evaluation stream shared by every detector, in decision mode and stopping mode (30 s refractory), with operating curves.
  C5 the CUSUM and k-of-n detectors on the surrogate scores; the environment runs them through the stream on TRAINING sessions only (smoke test).
2.6.12 is sealed: the regression script's reader drops it, and every table written here is asserted free of it. It imports the
adopted-pipeline script only.

Reads   everything the adopted-pipeline script reads (it rebuilds P bitwise), and baselines_controls/adopted_nested_predictions.csv.gz
        through the adopted-pipeline script's gates
Writes  Data/processed/rl_environment/steps.csv.gz, runs.csv, onsets.csv, folds.csv, state_index.csv.gz,
        states_<receiver>_<fold>.npz (15 files), episodes.csv.gz, transform_fingerprints.csv, smoke_detectors.csv,
        c_checks.csv; Graph/step_timeline.png, surrogate_trace.png
"""
from __future__ import annotations  # future import: type hints kept as text (the project's convention)

import gzip  # standard library: gzip with a fixed header time (byte-stable)
import hashlib  # standard library: SHA-256 fingerprints of the fitted transforms
import importlib.util  # standard library: loads the adopted-pipeline script (its name starts with a digit)
import io  # standard library: in-memory bytes for the .npz members
import sys  # standard library: module registry and exit status
import zipfile  # standard library: the .npz container, written with fixed member dates
from pathlib import Path  # standard library: paths

import numpy as np  # numpy: arrays
import pandas as pd  # pandas: tables
import gymnasium as gym  # gymnasium: the environment interface Stable-Baselines3 expects
from gymnasium import spaces  # gymnasium's action and observation spaces

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
PROC = PROJECT / "Data" / "processed"  # processed tables
ENV_DIR = PROC / "rl_environment"  # this section's tables
GRAPH = PROJECT / "Graph"  # figures
_spec = importlib.util.spec_from_file_location("adopted_pipeline_for_environment", PROJECT / "codes" / "5_baselines" / "07_adopted_pipeline.py")  # where the adopted-pipeline script is
A = importlib.util.module_from_spec(_spec)  # an empty module object
sys.modules["adopted_pipeline_for_environment"] = A  # registered, so its own references resolve
_spec.loader.exec_module(A)  # the adopted-pipeline script's top level (it loads the forest and regression scripts); its main() is not run
M, SC = A.M, A.SC  # 09 and the scoring module exactly as the adopted-pipeline script holds them
A.use_labels("final")  # the scoring module points at the final label set on import, so every importer reads labels/labels_evidence_final.csv

SEED = A.SEED  # 20260914
RXS, SESSIONS, SEALED = A.RXS, A.SESSIONS, A.SEALED  # receivers, the five folds, "2.6.12"
LAGS = (0, 10, 20)  # tuple: the primary state's windows end at t, t-10 and t-20
N_Z = 30  # the surrogate's log-odds at t-29 .. t
HIST = N_Z - 1  # 29 s of window history: a step is valid when the windows ending t-29 .. t lie in one run (both arms share the steps)
PRE_MAX, PRE_MIN = 300, 10  # tuple assignment: an onset episode starts between onset - 300 s and onset - 10 s
L_CLEAN = 120  # a clean episode lasts at most 120 steps, a truncation
REFRACTORY = 30  # stopping mode: silent for 30 s after an alarm
LAMBDAS = (10, 30, 100, 300, 1000)  # tuple: the lambda grid in seconds of delay per false alarm
SHARES = (0.02, 0.05, 0.10)  # tuple: the headline pooled clean-window alarm shares
GAMMA_TRAIN = 1.0  # the training discount: 1 (a discount below 1 was registered first and replaced before training); every return is finite and every evaluation undiscounted
N_EPISODES = 2000  # episodes per receiver and fold in the deterministic table
KN_N = (3, 5, 10)  # tuple: the k-of-n window lengths (k = 1 .. n)
STEP = {"pre": 0, "post": 1, "uncertain": 2}  # dict: step type -> the code the environment stores
NAME = {"SBF_02": "mosaic-X5", "SBF_03": "mosaic-H", "UBX_01": "u-blox 01"}  # dict: stream -> name in figures
N_FIG53 = 16  # episodes drawn per session in the figure, each on its own height


# ----------------------------------------------------------------------------------------------------------------------
# writing (byte-stable)
# ----------------------------------------------------------------------------------------------------------------------
def write_csv(df: pd.DataFrame, path: Path) -> None:  # function: plain CSV with fixed line endings
    path.parent.mkdir(parents=True, exist_ok=True)  # the folder
    df.to_csv(path, index=False, lineterminator="\n")  # written


def write_gz(df: pd.DataFrame, path: Path) -> None:  # function: gzip CSV with no name or time in its header, so runs again are byte-identical
    path.parent.mkdir(parents=True, exist_ok=True)  # the folder
    with open(path, "wb") as fh, gzip.GzipFile(filename="", mode="wb", fileobj=fh, mtime=0) as gz:  # with-block over the file and the gzip stream
        gz.write(df.to_csv(index=False, lineterminator="\n").encode("utf-8"))  # the table as CSV text


def write_npz(arrays: dict, path: Path) -> None:  # function: a .npz numpy can load, with every member dated 1980-01-01 (np.savez stamps the current time)
    path.parent.mkdir(parents=True, exist_ok=True)  # the folder
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:  # with-block: the zip archive
        for name, arr in arrays.items():  # for loop over the arrays in the given order
            buf = io.BytesIO()  # the member's bytes
            np.lib.format.write_array(buf, np.ascontiguousarray(arr), allow_pickle=False)  # numpy's .npy format, no pickled objects
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))  # fixed date
            info.compress_type = zipfile.ZIP_DEFLATED  # compressed
            z.writestr(info, buf.getvalue())  # written


def load_npz(path: Path) -> dict:  # function: the arrays of a .npz written by write_npz
    with np.load(path, allow_pickle=False) as f:  # with-block: numpy's reader closes the file afterwards
        return {k: f[k] for k in f.files}  # dict comprehension: every array, read into memory


# ----------------------------------------------------------------------------------------------------------------------
# C1: episode material
# ----------------------------------------------------------------------------------------------------------------------
def seconds_table(cells: dict) -> pd.DataFrame:  # function: every grid second of the five sessions per receiver: labels, step type, window, run, validity
    lab = M.read_table(PROC / "labels" / "labels_1hz_final.csv")[A.KEYS + ["label"]]  # the scoring labels (the regression script's reader drops 2.6.12)
    tl = M.read_table(PROC / "labels" / "train_labels_1hz.csv")[A.KEYS + ["train_label"]]  # the training labels (strict)
    g = lab.merge(tl, on=A.KEYS, how="left", validate="one_to_one")  # one row per receiver-second
    win = pd.concat([idx[A.KEYS] for idx, _ in cells.values()], ignore_index=True).assign(window=True)  # every window of P's cells, by its last second
    g = g.merge(win, on=A.KEYS, how="left", validate="one_to_one")  # the window flag beside the second
    g["window"] = g.window.eq(True)  # eq(True): True where a window ends at the second, False where none does
    g = g.sort_values(["receiver", "test_code", "sec"], key=lambda c: c.map({s: i for i, s in enumerate(SESSIONS)}) if c.name == "test_code" else c, kind="stable").reset_index(drop=True)  # lambda key: sessions in the screen script's order, then time
    g["step"] = np.select([g.train_label == "clean", g.train_label == "attack", g.label != "excluded"], ["pre", "post", "uncertain"], "excluded")  # step type by the training label
    g["run_id"], g["valid"] = -1, False  # window run within the session (-1: no window); validity
    for (rx, t), d in g.groupby(["receiver", "test_code"], sort=False):  # for loop over the pairs
        s = d.sec.to_numpy(np.int64)  # seconds in time order
        if not (np.diff(s) == 1).all():  # if: the grid is not consecutive (cannot)
            raise SystemExit(f"STOP: {rx} {t} grid is not consecutive")  # stop
        for k, (i, j) in enumerate(SC.runs(s, d.window.to_numpy())):  # the scoring module's runs(): maximal runs of windowed seconds
            g.loc[d.index[i:j + 1], "run_id"] = k  # the run's number
            g.loc[d.index[i:j + 1], "valid"] = s[i:j + 1] - s[i] >= HIST  # valid from 29 s after the run's first window
    return g.rename(columns={"test_code": "session"})  # the table


def runs_table(g: pd.DataFrame) -> pd.DataFrame:  # function: the window runs per receiver and session
    w = g[g.run_id >= 0]  # windowed seconds
    t = w.groupby(["receiver", "session", "run_id"], sort=False).agg(start=("sec", "min"), end=("sec", "max"), windows=("sec", "size"), valid_steps=("valid", "sum")).reset_index()  # one row per run
    t["first_valid"] = t.start + HIST  # the first second a state exists
    return t  # the table


def onsets_table(g: pd.DataFrame) -> pd.DataFrame:  # function: every attack run of the training labels, with what an onset episode needs
    rows = []  # one row per run
    for (rx, t), d in g.groupby(["receiver", "session"], sort=False):  # for loop over the pairs
        s, post, val, run = d.sec.to_numpy(np.int64), (d.step == "post").to_numpy(), d.valid.to_numpy(), d.run_id.to_numpy()  # arrays
        piv = SC.pooled_intervals(t)  # the pooled attack intervals (labels/labels_evidence_final.csv through the scoring module)
        prev = None  # end of the previous attack run
        for k, (i, j) in enumerate(SC.runs(s, post)):  # the scoring module's runs(): maximal runs of attack seconds
            a, b = int(s[i]), int(s[j])  # onset and last second
            pin = [q for q, (x, y) in enumerate(piv) if x <= a <= y]  # list comprehension: the pooled interval holding the onset
            reentry = prev is not None and bool(pin) and piv[pin[0]][0] <= prev  # the run starts inside the pooled interval of the previous run
            lo = None  # earliest start
            if val[i]:  # if: the onset second is a valid step
                first = int(s[(run == run[i]) & val].min())  # the first valid second of its window run
                lo = max(first, (prev + 1) if prev is not None and prev >= first else first, a - PRE_MAX)  # after any earlier attack run there, and no earlier than onset - 300 s
            usable = lo is not None and lo <= a - PRE_MIN  # at least one start second
            if usable and not val[i:j + 1].all():  # if: a usable interval is not valid to its end
                raise SystemExit(f"STOP: {rx} {t} interval at {a} is not valid to its end")  # stop
            # dict literal over two lines: the run
            rows.append({"receiver": rx, "session": t, "attack_run": k, "onset": a, "end": b, "length_s": b - a + 1, "reentry": reentry, "onset_valid": bool(val[i]), "usable": bool(usable),
                         "start_lo": lo if usable else -1, "start_hi": a - PRE_MIN if usable else -1, "pooled_interval": pin[0] + 1 if pin else 0})  # dict literal continued: the remaining fields of the row
            prev = b  # remembered
    return pd.DataFrame(rows)  # the table


def folds_table(g: pd.DataFrame, on: pd.DataFrame) -> pd.DataFrame:  # function: per receiver and outer fold: usable onsets, m and the training material
    rows = []  # one row per receiver and fold
    for rx in RXS:  # for loop over the receivers
        for held, train in A.S.folds():  # the screen script's folds
            u = on[(on.receiver == rx) & on.session.isin(train) & on.usable]  # usable onsets of the training sessions
            a = on[(on.receiver == rx) & on.session.isin(train)]  # every attack run there
            d = g[(g.receiver == rx) & g.session.isin(train) & g.valid]  # valid training steps
            h = g[(g.receiver == rx) & (g.session == held)]  # the held-out session
            # dict literal over two lines: the fold
            rows.append({"receiver": rx, "held_out": held, "usable_onsets": len(u), "attack_runs": len(a), "reentries": int(a.reentry.sum()), "m": float(np.median(u.length_s)), "m_all_runs": float(np.median(a.length_s)),
                         "valid_pre": int((d.step == "pre").sum()), "valid_post": int((d.step == "post").sum()), "valid_uncertain": int((d.step == "uncertain").sum()), "heldout_windows": int(h.window.sum()), "heldout_valid": int(h.valid.sum())})  # dict literal continued: the remaining fields of the row
    return pd.DataFrame(rows)  # the table


# ----------------------------------------------------------------------------------------------------------------------
# C2: states
# ----------------------------------------------------------------------------------------------------------------------
def window_rows(idx: pd.DataFrame) -> dict:  # function: session -> (sorted window seconds, their rows in the receiver's cells)
    out = {}  # session -> arrays
    for t in SESSIONS:  # for loop over the sessions
        r = np.flatnonzero(idx.test_code.to_numpy() == t)  # its rows
        o = np.argsort(idx.sec.to_numpy()[r], kind="stable")  # in time order
        out[t] = (idx.sec.to_numpy(np.int64)[r][o], r[o])  # seconds and rows
    return out  # the lookup


def rows_at(wr: dict, t: str, secs: np.ndarray) -> np.ndarray:  # function: the cell rows of the windows ending at secs (every one must exist)
    s, r = wr[t]  # the session's window seconds and rows
    k = np.searchsorted(s, secs)  # positions by binary search
    if (k >= len(s)).any() or (s[np.minimum(k, len(s) - 1)] != secs).any():  # if: a needed window does not exist (validity forbids it)
        raise SystemExit(f"STOP: a state needs a window that does not exist in {t}")  # stop
    return r[k]  # the rows


def states(B: dict, g: pd.DataFrame) -> tuple:  # function: the two state matrices of every receiver and fold, their index, and the transform fingerprints
    npr = B["nested"]["predictions"]  # the inner out-of-fold predictions (with decision values)
    lr = npr[npr.model == "lr"]  # the logistic regression's
    index, fps, mats = [], [], {}  # state index parts; fingerprints; (rx, fold) -> arrays
    for rx in RXS:  # for loop over the receivers
        idx, X = B["cells"][rx]  # the receiver's windows (every session) and their orbit-free values
        wr = window_rows(idx)  # session -> window lookup
        for held, train in A.S.folds():  # for loop over the folds
            rec = B["outer"]["kept"][(rx, held)]  # the outer fit
            Z = A.transform(rec, X, B["wc10"])  # every window -> the fold's 36 columns (C order)
            z = np.full(len(idx), np.nan)  # decision value per window
            te = rec["te"]  # the held-out session's rows in time order
            z[te] = A.decision(rec["model"]["lr"], A.transform(rec, X.iloc[te], B["wc10"]))  # the outer model on the held-out windows (C order, as the fit scored them)
            for s in train:  # for loop over the training sessions
                q = lr[(lr.receiver == rx) & (lr.outer_held_out == held) & (lr.held_out == s)]  # the inner fit that held s out
                z[rows_at(wr, s, q.sec.to_numpy(np.int64))] = q.dec.to_numpy()  # its decision values
            v = g[(g.receiver == rx) & g.valid]  # the valid seconds of the five sessions
            P, Sg, parts = [], [], []  # primary rows, surrogate rows, index parts
            for t in SESSIONS:  # for loop over the sessions in order
                d = v[v.session == t]  # its valid seconds
                sec = d.sec.to_numpy(np.int64)  # in time order
                P.append(np.hstack([Z[rows_at(wr, t, sec - lag)] for lag in LAGS]))  # list comprehension: windows ending t, t-10, t-20, side by side
                Sg.append(np.column_stack([z[rows_at(wr, t, sec - lag)] for lag in range(HIST, -1, -1)]))  # list comprehension: decision values at t-29 .. t, oldest first
                parts.append(d[["receiver", "session", "sec", "label", "train_label", "step", "run_id"]].assign(fold=held, role="heldout" if t == held else "train", z_source=f"outer:{held}" if t == held else f"inner:{held}:{t}"))  # where the row's scores came from
            Pm, Sm = np.vstack(P), np.vstack(Sg)  # the matrices
            if not (np.isfinite(Pm).all() and np.isfinite(Sm).all()):  # if: a value is missing (every window has a score since the decision)
                raise SystemExit(f"STOP: a non-finite state value for {rx} {held}")  # stop
            ix = pd.concat(parts, ignore_index=True)  # this fold's index
            ix["row"] = np.arange(len(ix))  # row in the matrices
            index.append(ix)  # kept
            mats[(rx, held)] = {"primary": Pm, "surrogate": Sm, "sec": ix.sec.to_numpy(np.int64)}  # dict literal: the arrays
            prep = rec["prep"]  # the fitted preparation
            h = hashlib.sha256(np.ascontiguousarray(prep["med"]).tobytes() + np.ascontiguousarray(prep["scaler"][0]).tobytes() + np.ascontiguousarray(prep["scaler"][1]).tobytes() + ";".join(rec["names"]).encode()).hexdigest()  # SHA-256 of medians, scaler and selection
            trs = sorted(set(rec["idx"].test_code.to_numpy()[rec["tr"]]))  # the sessions of the training rows the preparation was fitted on
            fps.append({"receiver": rx, "held_out": held, "fit_sessions": ";".join(trs), "held_out_in_fit": held in trs, "selected": ";".join(rec["names"]), "sha256": h})  # one row
    return pd.concat(index, ignore_index=True), mats, pd.DataFrame(fps)  # index, matrices, fingerprints


# ----------------------------------------------------------------------------------------------------------------------
# C3: the environment
# ----------------------------------------------------------------------------------------------------------------------
def material(rx: str, held: str) -> dict:  # function: what the environment of one receiver and fold needs, read from the written files
    mats = load_npz(ENV_DIR / f"states_{rx}_{held}.npz")  # the state matrices
    ix = pd.read_csv(ENV_DIR / "state_index.csv.gz", dtype={"session": str, "fold": str})  # the state index
    ix = ix[(ix.receiver == rx) & (ix.fold == held)].sort_values("row").reset_index(drop=True)  # this fold's rows, position = matrix row
    if not (ix.row.to_numpy() == np.arange(len(ix))).all():  # if: the index does not cover the matrix row by row (cannot)
        raise SystemExit(f"STOP: state index and matrix disagree for {rx} {held}")  # stop
    tr = ix[ix.role == "train"]  # the training rows only: the held-out session never enters an episode
    on = pd.read_csv(ENV_DIR / "onsets.csv", dtype={"session": str})  # the attack runs
    on = on[(on.receiver == rx) & (on.session != held) & on.usable].reset_index(drop=True)  # usable onsets of the training sessions
    fo = pd.read_csv(ENV_DIR / "folds.csv", dtype={"held_out": str})  # per fold
    m = float(fo[(fo.receiver == rx) & (fo.held_out == held)].m.iloc[0])  # the miss penalty
    key = {(s, int(t)): int(r) for s, t, r in zip(tr.session, tr.sec, tr.row)}  # dict comprehension: (session, second) -> state row, training rows only
    # dict literal over two lines: the material
    return {"receiver": rx, "fold": held, "primary": mats["primary"], "surrogate": mats["surrogate"], "index": ix, "key": key, "onsets": on, "m": m,
            "type": ix.step.map(STEP).to_numpy(np.int8), "run": ix.run_id.to_numpy(), "session": ix.session.to_numpy(), "sec": ix.sec.to_numpy(np.int64), "pre_rows": tr.row[tr.step == "pre"].to_numpy()}  # arrays by matrix row; clean starts from training pre-change rows


class SpoofingDetectionEnv(gym.Env):  # class: a gymnasium environment; one episode is one stretch of one receiver's training stream
    metadata = {"render_modes": []}  # dict: no rendering

    def __init__(self, mat: dict, arm: str = "primary", lam: float = 100.0):  # method: built from material(); arm "primary" (108 values) or "surrogate" (31 values)
        super().__init__()  # gym.Env's own set-up
        self.mat, self.arm, self.lam, self.m = mat, arm, float(lam), mat["m"]  # tuple assignment: material, arm, lambda and miss penalty
        self.X = mat[arm]  # the arm's state matrix
        dim = self.X.shape[1] + (1 if arm == "surrogate" else 0)  # conditional expression: the surrogate appends S_t
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float64)  # unbounded float64 vector
        self.action_space = spaces.Discrete(2)  # 0 silent, 1 alarm
        self.rows = np.zeros(0, int)  # the episode's state rows (set by reset)

    def episode(self, kind: str, session: str, start: int, onset: int = -1) -> dict:  # method: the episode starting at (session, start): its rows and its last post-change position
        k = self.mat["key"]  # (session, second) -> row
        r0 = k[(session, start)]  # the first row
        if kind == "onset":  # an onset episode
            o = self.mat["onsets"]  # usable onsets
            end = int(o[(o.session == session) & (o.onset == onset)].end.iloc[0])  # the interval's last second
            rows = np.array([k[(session, t)] for t in range(start, end + 1)])  # list comprehension: every second to the interval's end (each must be a valid step)
            last_post = len(rows) - 1  # the miss is decided at the last row
        else:  # a clean episode
            rows, t = [r0], start  # its first row and second
            while len(rows) < L_CLEAN and (session, t + 1) in k and self.mat["run"][k[(session, t + 1)]] == self.mat["run"][r0] and self.mat["type"][k[(session, t + 1)]] != STEP["post"]:  # while loop: the next second is a valid, same-run, non-attack step
                t += 1  # one on
                rows.append(k[(session, t)])  # kept
            rows, last_post = np.array(rows), -1  # no post-change step
        return {"kind": kind, "session": session, "start": start, "onset": onset, "rows": rows, "last_post": last_post}  # dict literal: the episode

    def reset(self, *, seed: int | None = None, options: dict | None = None):  # method: gymnasium's reset; options={"kind", "session", "start", "onset"} replays a given episode
        super().reset(seed=seed)  # seeds self.np_random when a seed is given
        if options and "kind" in options:  # if: a replay
            ep = self.episode(options["kind"], options["session"], int(options["start"]), int(options.get("onset", -1)))  # that episode
        elif self.np_random.random() < 0.5:  # elif: an onset episode (probability 0.5, from the env's own generator)
            o = self.mat["onsets"].iloc[int(self.np_random.integers(len(self.mat["onsets"])))]  # an onset, uniformly
            ep = self.episode("onset", o.session, int(self.np_random.integers(int(o.start_lo), int(o.start_hi) + 1)), int(o.onset))  # its start, uniformly
        else:  # a clean episode
            r = int(self.mat["pre_rows"][int(self.np_random.integers(len(self.mat["pre_rows"])))])  # a valid pre-change second, uniformly
            ep = self.episode("clean", self.mat["session"][r], int(self.mat["sec"][r]))  # the episode from there
        self.ep, self.rows, self.i, self.S = ep, ep["rows"], 0, 0.0  # tuple assignment: episode, rows, position, CUSUM statistic
        return self._obs(), self._info(False)  # the first observation and its info

    def _obs(self) -> np.ndarray:  # method: the observation at the current position
        x = self.X[self.rows[self.i]]  # the arm's state row
        if self.arm == "surrogate":  # if: the surrogate arm
            return np.append(x, self.S_next())  # its 30 log-odds and S_t
        return x.copy()  # a copy, so the caller cannot change the matrix

    def S_next(self) -> float:  # method: S_t = max(0, S_(t-1) + z_t) at the current position (S = 0 before the first step)
        self.S_t = max(0.0, self.S + float(self.X[self.rows[self.i], -1]))  # z_t is the last of the 30 log-odds
        return self.S_t  # the value

    def _info(self, masked: bool, event: str = "") -> dict:  # method: what step() reports beside the reward
        r = self.rows[self.i]  # the current row
        return {"step_type": ["pre", "post", "uncertain"][int(self.mat["type"][r])], "sec": int(self.mat["sec"][r]), "masked_alarm": masked, "event": event}  # dict literal: type, second, masked-alarm flag, outcome

    def step(self, action):  # method: gymnasium's step: (obs, reward, terminated, truncated, info)
        typ = int(self.mat["type"][self.rows[self.i]])  # the current second's step type
        alarm = int(action) == 1  # an alarm?
        reward, term, masked, event = 0.0, False, False, ""  # defaults: silent in a pre-change or uncertain second
        if typ == STEP["post"]:  # a post-change second
            if alarm:  # an alarm after the change
                term, event = True, "detection"  # the end of the cost, no reward
            else:  # silent after the change
                reward = -1.0  # one second of delay
                if self.i == self.ep["last_post"]:  # if: the interval's last second passes in silence
                    reward, term, event = -1.0 - self.m, True, "miss"  # the miss penalty on top
        elif typ == STEP["pre"] and alarm:  # an alarm before the change (clean or quiet negative)
            reward, term, event = -self.lam, True, "false_alarm"  # the false-alarm penalty
        elif typ == STEP["uncertain"] and alarm:  # an alarm in an uncertain second
            masked = True  # recorded, no reward, the episode goes on
        if self.arm == "surrogate":  # if: the surrogate arm
            self.S = self.S_t  # S_t becomes S_(t-1) of the next step
        info = self._info(masked, event)  # the info of the step just taken
        if term:  # the episode ended
            return self._obs_last(), reward, True, False, info  # terminated
        self.i += 1  # the next second
        if self.i == len(self.rows):  # past the last row (clean episodes only)
            self.i -= 1  # stays on the last row for the observation
            return self._obs_truncated(), reward, False, True, {**info, "event": "truncated"}  # truncated
        return self._obs(), reward, False, False, info  # the next observation

    def _obs_last(self) -> np.ndarray:  # method: the observation returned with a terminal or truncated step (the current row, S unchanged)
        x = self.X[self.rows[self.i]]  # the state row
        return np.append(x, self.S) if self.arm == "surrogate" else x.copy()  # conditional expression: with S for the surrogate

    def _obs_truncated(self) -> np.ndarray:  # method: the final observation of a truncated clean episode, which the agent bootstraps from
        r = self.rows[self.i]  # the last row of the episode
        nxt = (str(self.mat["session"][r]), int(self.mat["sec"][r]) + 1)  # tuple: (session, next second)
        if nxt not in self.mat["key"] or self.mat["run"][self.mat["key"][nxt]] != self.mat["run"][r]:  # if: the next second is not a valid step of the same window run
            return self._obs_last()  # the last row's own state, as before the decision
        x = self.X[self.mat["key"][nxt]]  # the next second's state row
        if self.arm == "surrogate":  # if: the surrogate arm
            return np.append(x, max(0.0, self.S + float(x[-1])))  # its 30 log-odds and S at that second, continuing the episode's recursion (self.S already holds S_t)
        return x.copy()  # a copy, so the caller cannot change the matrix


def episode_table(rx: str, held: str) -> pd.DataFrame:  # function: N_EPISODES consecutive resets from SEED: the deterministic, replayable episode table
    env = SpoofingDetectionEnv(material(rx, held), "primary")  # the environment (the draws do not depend on the arm)
    rows = []  # one row per episode
    for e in range(N_EPISODES):  # for loop over the episodes
        env.reset(seed=SEED if e == 0 else None)  # the first reset seeds the generator, the rest continue it
        ep, t = env.ep, env.mat["type"][env.rows]  # the episode and its step types
        sec = env.mat["sec"][env.rows]  # its seconds
        # dict literal over two lines: the episode
        rows.append({"receiver": rx, "fold": held, "episode": e, "kind": ep["kind"], "session": ep["session"], "onset": ep["onset"], "start": ep["start"], "end": int(sec[-1]), "steps": len(env.rows),
                     "pre_steps": int((t == 0).sum()), "post_steps": int((t == 1).sum()), "uncertain_steps": int((t == 2).sum()), "onset_pos": int(np.flatnonzero(sec == ep["onset"])[0]) if ep["kind"] == "onset" else -1, "clean_full_length": ep["kind"] == "clean" and len(env.rows) == L_CLEAN})  # dict literal continued: the remaining fields of the row
    return pd.DataFrame(rows)  # the table


# ----------------------------------------------------------------------------------------------------------------------
# C4: the evaluation stream (shared by every detector)
# ----------------------------------------------------------------------------------------------------------------------
def events(sec: np.ndarray, decisions: np.ndarray, refractory: int = REFRACTORY) -> np.ndarray:  # function: stopping mode: an alarm, then silence for `refractory` seconds; returns the alarm events
    out = np.zeros(len(sec), bool)  # no event yet
    last = -np.inf  # the last event's second
    for i in np.flatnonzero(decisions):  # for loop over the seconds the detector wants to alarm
        if sec[i] - last > refractory:  # if: past the refractory period
            out[i], last = True, sec[i]  # an event, remembered
    return out  # the events


def stream(sec, labels, decisions, held: str, mode: str = "decision", decidable=None) -> dict:  # function: every figure of one detector on one held-out session (decision or stopping mode); decidable: optional mask, figures on those seconds only
    sec, lab = np.asarray(sec, np.int64), np.asarray(labels, object)  # arrays
    dec = np.nan_to_num(np.asarray(decisions, float), nan=0.0) > 0  # decisions as booleans (NaN, an undefined state, is silent)
    if decidable is not None:  # if: restricted to the decidable seconds
        k = np.asarray(decidable, bool)  # the mask
        sec, lab, dec = sec[k], lab[k], dec[k]  # tuple assignment: the subset
    al = dec if mode == "decision" else events(sec, dec)  # conditional expression: the alarms that count
    att, cln, unc = np.isin(lab, A.ATTACK), lab == "clean", lab == "uncertain"  # masks
    pc = SC.per_class(lab, al)  # the scoring module's per-class figures (clean and attack seconds only)
    fa = SC.false_alarms(sec, lab, al)  # the scoring module's false-alarm episodes, bound and rate where readable
    pol = SC.fold_policy(held, int(cln.sum()))  # the scoring module's fold policy
    out = {"held_out": held, "mode": mode, "seconds": len(sec), "clean": int(cln.sum()), "attack": int(att.sum()), "uncertain": int(unc.sum()), "alarms": int(al.sum()), "clean_in_alarm": int((al & cln).sum()),  # dict literal: counts
           "clean_share": float((al & cln).sum() / cln.sum()) if cln.any() else np.nan, "alarms_uncertain": int((al & unc).sum()), "recall": pc["attack_recall"], "precision": pc["attack_precision"] if pol["precision"] else np.nan, "f1": pc["attack_f1"] if pol["precision"] else np.nan,  # shares and per-class figures
           "tp": pc["tp"], "fn": pc["fn"], "fp": pc["fp"], "tn": pc["tn"], "clean_precision": pc["clean_precision"], "clean_recall": pc["clean_recall"], "clean_f1": pc["clean_f1"], "fa_figure": pol["false_alarms"], "fa_episodes": fa["episodes"], "fa_upper95_count": fa["upper95_count"],  # cells and false alarms
           "fa_rate_per_h": fa["rate_per_h"] if pol["false_alarms"] == "rate" else None, "fa_upper95_per_h": fa["upper95_per_h"] if pol["false_alarms"] == "rate" else None}  # the rate only where the policy reads one
    lat, prev, pre_n, pre_a = [], None, 0, 0  # per interval; previous interval end; pre-onset uncertain seconds and alarms
    for a, b in SC.pooled_intervals(held):  # for loop over the pooled intervals (the labelled onsets)
        inside = (sec >= a) & (sec <= b)  # seconds inside the interval
        first = sec[inside & al]  # alarms inside it
        ps = SC.pre_onset_alarm_share(sec, lab, al, a, prev)  # the scoring module's pre-onset uncertain seconds since the previous interval, and their alarm share (the baselines' early-warning measure)
        pre_n += ps["pre_onset_uncertain_s"]  # counted
        pre_a += ps["pre_onset_alarm_share"] * ps["pre_onset_uncertain_s"] if ps["pre_onset_uncertain_s"] else 0  # alarms among them
        lat.append({"onset": a, "end": b, "seconds_inside": int(inside.sum()), "detected": bool(first.size), "delay_s": int(first.min() - a) if first.size else np.nan})  # delay from the labelled onset
        prev = b  # remembered
    out["intervals"], out["detected"] = len(lat), sum(x["detected"] for x in lat)  # generator inside sum(): detected intervals
    out["delay_median_s"] = float(np.median([x["delay_s"] for x in lat if x["detected"]])) if out["detected"] else np.nan  # median delay of the detected intervals
    out["pre_onset_uncertain"] = pre_n  # the baselines' denominator
    out["early_warning"] = (pre_a / pre_n - out["clean_share"]) if pre_n and cln.any() else np.nan  # pre-onset alarm share minus clean alarm share
    out["latency"] = lat  # per interval
    return out  # the figures


def pooled_curve(rows: list) -> dict:  # function: one parameter's figures pooled over folds: clean share (decision mode), median delay and miss share (stopping mode)
    cl = sum(r["clean"] for r in rows if r["mode"] == "decision")  # generator inside sum(): pooled clean seconds
    ca = sum(r["clean_in_alarm"] for r in rows if r["mode"] == "decision")  # pooled clean seconds in alarm
    lat = [x for r in rows if r["mode"] == "stopping" for x in r["latency"]]  # list comprehension: every interval in stopping mode
    det = [x["delay_s"] for x in lat if x["detected"]]  # delays of the detected ones
    return {"clean_share": ca / cl if cl else np.nan, "delay_median_s": float(np.median(det)) if det else np.nan, "miss_share": 1 - len(det) / len(lat) if lat else np.nan, "intervals": len(lat)}  # dict literal: the curve point


def read_at(curve: pd.DataFrame, share: float) -> pd.Series:  # function: the curve point with the largest pooled clean share at or below a registered share (no interpolation)
    c = curve[curve.clean_share <= share]  # the admissible points
    return c.sort_values(["clean_share", "param"], kind="stable").iloc[-1] if len(c) else pd.Series(dtype=float)  # the closest from below, or empty


# ----------------------------------------------------------------------------------------------------------------------
# C5: CUSUM and k-of-n on the surrogate scores
# ----------------------------------------------------------------------------------------------------------------------
def cusum(sec: np.ndarray, z: np.ndarray, valid: np.ndarray, h: float, mode: str = "decision") -> np.ndarray:  # function: Page's CUSUM on z over valid seconds; decision mode: alarm while S >= h; stopping mode: alarm, reset, silent 30 s
    out = np.zeros(len(sec), bool)  # decisions
    S, last, quiet_to = 0.0, None, -np.inf  # statistic, previous valid second, end of the refractory period
    for i in range(len(sec)):  # for loop over the seconds in time order
        if not valid[i]:  # no state here: silent, and S restarts when states resume
            S, last = 0.0, None  # tuple assignment: reset
            continue  # next second
        if last is not None and sec[i] - last != 1:  # a gap in the valid seconds
            S = 0.0  # restarts
        last = sec[i]  # remembered
        if sec[i] <= quiet_to:  # inside the refractory period (stopping mode)
            continue  # silent; S stays 0
        S = max(0.0, S + z[i])  # Page's recursion
        if S >= h:  # the threshold is reached
            out[i] = True  # an alarm
            if mode == "stopping":  # stopping mode
                S, quiet_to = 0.0, sec[i] + REFRACTORY  # reset, silent for 30 s
    return out  # the decisions


def k_of_n(sec: np.ndarray, z: np.ndarray, valid: np.ndarray, k: int, n: int, h: float) -> np.ndarray:  # function: alarm when at least k of the last n decision values reach h (valid seconds; the n values lie in the state's history)
    hit = (np.nan_to_num(z, nan=-np.inf) >= h).astype(int)  # 1 where a value reaches h
    c = np.convolve(hit, np.ones(n, int))[:len(hit)]  # running count over the last n seconds (convolution with a window of ones)
    return valid & (c >= k)  # only where a state exists


# ----------------------------------------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------------------------------------
COL = {"pre": "#009E73", "post": "#D55E00", "uncertain": "#BBBBBB", "excluded": "#000000", "clean": "#009E73", "attack_jam_spoof": "#D55E00", "attack_spoof": "#CC79A7"}  # dict: type or label -> colour (the regression script's colour-blind palette)


def plot_step_timeline(g: pd.DataFrame, on: pd.DataFrame, ep: pd.DataFrame, rx: str, held: str) -> None:  # function: the figure, one training fold's step types per session, onsets, episodes as bars, masked stretches
    import matplotlib.pyplot as plt  # matplotlib's pyplot (09 chose the file-only backend)
    train = [s for s in SESSIONS if s != held]  # the training sessions
    fig, axes = plt.subplots(len(train), 1, figsize=(11, 2.3 * len(train)))  # one panel per session
    for ax, s in zip(axes, train):  # for loop over the panels
        d = g[(g.receiver == rx) & (g.session == s)]  # the session's seconds
        t0 = d.sec.min()  # its first second
        for typ, y in (("pre", 0), ("post", 0), ("uncertain", 0), ("excluded", 0)):  # for loop over the step types
            x = d[d.step == typ]  # their seconds
            ax.bar((x.sec - t0) / 60, 0.6, bottom=y, width=1 / 60, color=COL[typ], linewidth=0, label={"pre": "pre-change (clean or quiet negative)", "post": "post-change (attack)", "uncertain": "uncertain (alarm masked)", "excluded": "excluded"}[typ])  # one thin bar per second
        nv = d[~d.valid & (d.step != "excluded")]  # non-excluded seconds without a state
        ax.bar((nv.sec - t0) / 60, 0.15, bottom=0.62, width=1 / 60, color="#56B4E9", linewidth=0, label="no state (no window or < 29 s of run)")  # their marks
        for o in on[(on.receiver == rx) & (on.session == s)].itertuples():  # for loop over the attack runs
            ax.axvline((o.onset - t0) / 60, color="#000000" if o.usable else "#999999", lw=1.2, ls="-" if o.usable else ":")  # the onset (dotted: not usable)
        e = ep[(ep.receiver == rx) & (ep.fold == held) & (ep.session == s)].head(N_FIG53)  # the session's first episodes of the table
        for j, r in enumerate(e.itertuples()):  # for loop over them
            ax.plot([(r.start - t0) / 60, (r.end - t0) / 60], [0.82 + 0.022 * j] * 2, color=COL["post"] if r.kind == "onset" else COL["pre"], lw=1.3)  # the episode as a horizontal bar, each on its own height
        ax.set_xlim(0, (d.sec.max() - t0) / 60)  # the session's span
        ax.set_ylim(0, 0.84 + 0.022 * N_FIG53)  # room for the episode bars
        ax.set_yticks([])  # no vertical scale
        ax.set_title(f"{s}: {NAME[rx]}, fold holding out {held} (training session)", fontsize=9, loc="left")  # panel title
        ax.set_xlabel(f"minutes after {pd.Timestamp(int(t0), unit='s').strftime('%H:%M:%S')} UTC", fontsize=8)  # time axis
    h_, l_ = axes[0].get_legend_handles_labels()  # the legend entries of one panel
    fig.legend(h_, l_, loc="lower center", fontsize=7, ncol=5, frameon=False)  # the legend once, below the panels, covering no data
    fig.tight_layout(rect=(0, 0.03, 1, 1))  # layout, leaving room for the legend
    M.save_fig(fig, GRAPH / "step_timeline.png")  # byte-stable save (the regression script's helper)


def plot_surrogate_trace(ix: pd.DataFrame, mats: dict, rx: str, held: str) -> None:  # function: the figure, the surrogate trace on a held-out session beside the labels, with P's committed probabilities overlaid
    import matplotlib.pyplot as plt  # pyplot
    from scipy.special import expit  # logistic function
    d = ix[(ix.receiver == rx) & (ix.fold == held) & (ix.role == "heldout")].reset_index(drop=True)  # the held-out session's valid seconds
    z = mats[(rx, held)]["surrogate"][d.row.to_numpy(), -1]  # z_t (the last of the 30 values)
    sec = d.sec.to_numpy(np.int64)  # seconds
    Sv, s_ = np.zeros(len(z)), 0.0  # S without resets except at gaps
    for i in range(len(z)):  # for loop over the seconds
        s_ = max(0.0, (s_ if i and sec[i] - sec[i - 1] == 1 else 0.0) + z[i])  # Page's recursion, restarted at a gap
        Sv[i] = s_  # kept
    p = M.read_table(PROC / "logistic_regression" / "logistic_regression_predictions_primary.csv.gz")  # P's committed logistic-regression probabilities (the raw quantity)
    p = p[(p.receiver == rx) & (p.held_out == held)].set_index("sec").p  # this session's
    t0 = sec.min()  # first second
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [1, 2, 2]})  # three panels
    for lab_, col in COL.items():  # for loop over the label colours
        if lab_ in ("clean", "attack_jam_spoof", "attack_spoof"):  # if: a scored label
            x = d[d.label == lab_]  # its seconds
            a1.bar((x.sec - t0) / 60, 1, width=1 / 60, color=col, linewidth=0, label=lab_)  # thin bars
    x = d[d.label == "uncertain"]  # uncertain seconds
    a1.bar((x.sec - t0) / 60, 1, width=1 / 60, color=COL["uncertain"], linewidth=0, label="uncertain")  # thin bars
    a1.set_yticks([])  # no scale
    a1.legend(fontsize=6, ncol=4, loc="upper right")  # legend
    a1.set_title(f"{NAME[rx]}, held-out session {held}: scoring labels", fontsize=9, loc="left")  # title
    a2.plot((sec - t0) / 60, expit(z), color="#0072B2", lw=1.6, label="expit(z_t), z_t = decision value (derived)")  # the derived trace on the probability scale
    a2.plot((p.index.to_numpy() - t0) / 60, p.to_numpy(), color="#E69F00", lw=0.6, ls="--", label="P's committed probability (raw, logistic_regression)")  # the raw quantity, every windowed second
    a2.set_ylabel("probability", fontsize=8)  # axis
    a2.legend(fontsize=7, loc="upper right")  # legend
    a3.plot((sec - t0) / 60, z, color="#0072B2", lw=0.8, label="z_t (log-odds)")  # the increment
    a3.plot((sec - t0) / 60, Sv, color="#000000", lw=1.2, label="CUSUM S_t (no reset)")  # the statistic
    a3.set_yscale("symlog")  # signed log scale: S grows to hundreds
    a3.set_ylabel("log-odds / S", fontsize=8)  # axis
    a3.set_xlabel(f"minutes after {pd.Timestamp(int(t0), unit='s').strftime('%H:%M:%S')} UTC", fontsize=8)  # time axis
    a3.legend(fontsize=7, loc="upper left")  # legend
    fig.tight_layout()  # layout
    M.save_fig(fig, GRAPH / "surrogate_trace.png")  # byte-stable save


# ----------------------------------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------------------------------
def smoke(ix: pd.DataFrame, mats: dict) -> pd.DataFrame:  # function: C5 smoke test: CUSUM and k-of-n through the stream on TRAINING sessions only (scoring labels), a few settings
    rows = []  # results
    for rx in RXS:  # for loop over the receivers
        for held, train in A.S.folds():  # for loop over the folds
            d = ix[(ix.receiver == rx) & (ix.fold == held) & (ix.role == "train")]  # training rows only
            z = mats[(rx, held)]["surrogate"][:, -1]  # z_t per row
            for s in train:  # for loop over the training sessions
                q = d[d.session == s]  # its valid seconds
                sec, zs, lab_ = q.sec.to_numpy(np.int64), z[q.row.to_numpy()], q.label.to_numpy(object)  # arrays
                ok = np.ones(len(sec), bool)  # every row here is a valid step
                for det, dec in (("cusum h=50", cusum(sec, zs, ok, 50.0)), ("cusum h=50 stopping", cusum(sec, zs, ok, 50.0, "stopping")), ("3-of-5 h=0", k_of_n(sec, zs, ok, 3, 5, 0.0))):  # for loop over three settings
                    for mode in ("decision", "stopping"):  # both stream modes
                        f = stream(sec, lab_, dec, s, mode)  # through the stream (the scoring module's policy of that session)
                        rows.append({"receiver": rx, "fold": held, "session": s, "detector": det, "mode": mode, **{k: f[k] for k in ("seconds", "clean", "attack", "alarms", "clean_share", "recall", "detected", "intervals", "delay_median_s", "early_warning")}})  # dict unpacking: the figures
    return pd.DataFrame(rows)  # the table


def main() -> None:  # function: builds the material, the states, the episodes, the smoke test and the figures
    B = A.build()  # P, rebuilt bitwise (the adopted-pipeline script's gates prove it, the environment tests repeats the identity checks)
    g = seconds_table(B["cells"])  # C1: every second
    if (g.session == SEALED).any():  # if: 2.6.12 slipped in (cannot)
        raise SystemExit("STOP: 2.6.12 in the step table")  # stop
    rn, on = runs_table(g), onsets_table(g)  # C1: runs and attack runs
    fo = folds_table(g, on)  # C1: per fold
    on_flag = on[["receiver", "session", "onset"]].rename(columns={"onset": "sec"}).assign(onset=True)  # onset seconds
    g = g.merge(on_flag, on=["receiver", "session", "sec"], how="left")  # beside the step table
    g["onset"] = g.onset.eq(True)  # True at an onset second
    write_gz(g[["receiver", "session", "sec", "label", "train_label", "step", "window", "run_id", "valid", "onset"]], ENV_DIR / "steps.csv.gz")  # C1
    write_csv(rn, ENV_DIR / "runs.csv")  # C1
    write_csv(on, ENV_DIR / "onsets.csv")  # C1
    write_csv(fo, ENV_DIR / "folds.csv")  # C1
    ix, mats, fps = states(B, g)  # C2
    write_gz(ix, ENV_DIR / "state_index.csv.gz")  # C2: the index
    for (rx, held), a in mats.items():  # for loop over the 15 matrix files
        write_npz(a, ENV_DIR / f"states_{rx}_{held}.npz")  # C2
    write_csv(fps, ENV_DIR / "transform_fingerprints.csv")  # C2: what each transform was fitted on
    ep = pd.concat([episode_table(rx, held) for rx in RXS for held, _ in A.S.folds()], ignore_index=True)  # C3: the episode table
    write_gz(ep, ENV_DIR / "episodes.csv.gz")  # C3
    sm = smoke(ix, mats)  # C5 through C4, training sessions only
    write_csv(sm, ENV_DIR / "smoke_detectors.csv")  # C5
    checks = [  # list: the checks of this script (the environment tests holds V1-V8)
        {"check": "2.6.12 absent from every table written", "pass": not any((t.astype(str) == SEALED).any() for t in (g.session, on.session, fo.held_out, ix.session, ix.fold, ep.session, ep.fold, sm.session))},  # sealed
        {"check": "every valid second has finite primary and surrogate states", "pass": all(np.isfinite(a["primary"]).all() and np.isfinite(a["surrogate"]).all() for a in mats.values())},  # finite
        {"check": "usable onsets per receiver = 7", "pass": set(on[on.usable].groupby("receiver").size()) == {7}},  # the planned count
        {"check": "episode table: 2,000 episodes per receiver and fold", "pass": set(ep.groupby(["receiver", "fold"]).size()) == {N_EPISODES}},  # size
        {"check": "no transform was fitted on its held-out session", "pass": not fps.held_out_in_fit.any()},  # leakage
    ]
    ck = pd.DataFrame(checks)  # the table
    write_csv(ck, ENV_DIR / "c_checks.csv")  # written
    plot_step_timeline(g, on, ep, "SBF_02", "2.6.2")  # the figure: the mosaic-X5 fold holding out 2.6.2
    plot_surrogate_trace(ix, mats, "SBF_02", "2.6.2")  # the figure: its held-out session
    print(ck.to_string(index=False))  # printed
    print(f"episodes: {len(ep)}; kinds {dict(ep.kind.value_counts())}; states {sum(len(a['sec']) for a in mats.values())} rows")  # summary
    if not ck["pass"].all():  # if: a check failed
        sys.exit(1)  # exit status 1


if __name__ == "__main__":  # guard: runs main() only when the file is executed, not when imported
    main()  # runs main
