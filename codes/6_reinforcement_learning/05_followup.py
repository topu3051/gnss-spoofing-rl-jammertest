#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
The exploratory follow-up of the diagnosis, designed after the agent experiment's results were seen. The agent experiment's verdicts
stay the confirmatory answer to RQ2; nothing here replaces them.
  Run A (F-c): an alarm in an uncertain second opens the 30 s refractory period in training as it does in the evaluation stream; during it
         alarms neither end the episode, nor detect, nor cost lambda; silent post-change seconds still cost 1; S_t is held at 0 through it
         and accumulates again from the first second after it, as the training script's agent_decisions restarts S after a stopping-mode event.
  Run B (F-c + F-e): Run A plus demonstrations of CUSUM(h*) fitted on each job's own training sessions, in the manner of deep Q-learning
         from demonstrations (Hester et al., 2018): a permanent demonstration memory, 10,000 pre-training updates on it alone, then minibatches
         of 48 own + 16 demonstration transitions; 1-step TD on all, a large-margin loss (0.05) and a 10-step loss on demonstration rows only.
Steps: --pilot RUN (timing of one cell), --grid RUN (inner jobs, selection, outer jobs, freeze tables), --heldout RUN --freeze HASH (the
single outer read, refused when outer_read.json exists), --recompute RUN (every held-out table again from the stored per-second decisions,
no second read), --diag RUN (D3 and H2 on inner folds at the selected budgets, closed or open by the diagnosis rules, absorption and demo share).
Selection of budget and lambda follows the registered rules on inner folds and is the only selection. 2.6.12 is not read.
It imports the diagnosis script (which loads the evaluation, training and environment scripts). Writes Data/processed/rl_diagnosis/followup/
(the model files kept apart, their SHA-256 in the freeze manifest).
"""
from __future__ import annotations  # future import: type hints kept as text (the project's convention)

import argparse  # standard library: the command line
import hashlib  # standard library: SHA-256 digests
import importlib.util  # standard library: loads the diagnosis script (its name starts with a digit)
import json  # standard library: the job digest and outer_read.json
import multiprocessing  # standard library: the spawn context of the worker pool
import subprocess  # standard library: read-only git commands (the freeze check)
import sys  # standard library: module registry and exit status
import time  # standard library: wall-clock timing (timing tables only, declared)
from concurrent.futures import ProcessPoolExecutor, as_completed  # standard library: the pool of worker processes
from pathlib import Path  # standard library: paths

import numpy as np  # numpy: arrays
import pandas as pd  # pandas: tables

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
_spec = importlib.util.spec_from_file_location("diagnosis_for_followup", PROJECT / "codes" / "6_reinforcement_learning" / "04_diagnosis.py")  # where 15 is
D = importlib.util.module_from_spec(_spec)  # an empty module object
sys.modules["diagnosis_for_followup"] = D  # registered, so its own references resolve
_spec.loader.exec_module(D)  # the diagnosis script's top level (the training and environment scripts); its main() is not run
B12, T, E, A = D.B12, D.T, D.E, D.A  # the evaluation, training, environment and adopted-pipeline scripts as the diagnosis script holds them

import torch as th  # PyTorch (12 and 15 have set it to one thread)
th.set_num_threads(1)  # one thread again, explicitly (the environment Part E)
import torch.nn.functional as F  # PyTorch's loss functions (SB3's DQN uses smooth_l1_loss)
from stable_baselines3 import DQN  # the Deep Q-Network
from stable_baselines3.common.utils import polyak_update  # SB3's target-network copy (tau 1 = a full copy)

FOLLOWUP_DIR = D.DIAG_DIR / "followup"  # this step's folder
SEALED = D.SEALED  # "2.6.12", never read
SESSIONS, RXS, LAMBDAS, ARMS = T.SESSIONS, T.RXS, T.LAMBDAS, D.ARMS  # folds, receivers, lambda grid, arms
REFR = E.REFRACTORY  # 30 s
PRE, POST, UNC = E.STEP["pre"], E.STEP["post"], E.STEP["uncertain"]  # tuple assignment: the step codes
H_EP = D.H_EP  # CUSUM's threshold grid (0, 1 to 10,000 in 41 log steps) and infinity
DEMO, BATCH, PRETRAIN = 16, 64, 10_000  # tuple assignment: the follow-up's settings: demonstration rows per minibatch, minibatch, pre-training updates
MARGIN, NSTEP, LAM_N, LAM_E = 0.05, 10, 1.0, 1.0  # tuple assignment: Table 13a: margin, n-step length, n-step weight, supervised weight
FC_VERSION = "r15-cp2-1"  # a label in the job digest
OUTCOME = {"detection": 0, "false_alarm": 1, "miss": 2, "clean_end": 3}  # dict: episode outcome -> integer
LABEL = "exploratory follow-up, designed after the agent experiment results were seen"  # every table and figure carries this label


def rundir(run: str) -> Path:  # function: a run's folder
    return FOLLOWUP_DIR / f"run{run}"  # followup/runA (refractory-consistent training) or followup/runB (with demonstrations)


# ----------------------------------------------------------------------------------------------------------------------
# Run A's environment (F-c)
# ----------------------------------------------------------------------------------------------------------------------
class FcEnv(E.SpoofingDetectionEnv):  # class: the environment script's environment with the refractory period of an uncertain-second alarm, Run A)
    def reset(self, *, seed=None, options=None):  # method: a new episode; no refractory period running
        self.quiet = 0  # seconds of refractory period left (0: none)
        return super().reset(seed=seed, options=options)  # the environment script's reset (its first observation calls S_next below)

    def S_next(self) -> float:  # method: S_t at the current position: 0 inside a refractory period, else Page's recursion (the environment script's)
        if self.quiet > 0:  # if: inside a refractory period
            self.S_t = 0.0  # held at 0, as the stream recomputes S only from the first second after the period
            return 0.0  # the value
        return super().S_next()  # the environment script's max(0, S + z_t)

    def step(self, action):  # method: gymnasium's step under F-c: (obs, reward, terminated, truncated, info)
        typ = int(self.mat["type"][self.rows[self.i]])  # the current second's step type
        inref = self.quiet > 0  # inside a refractory period?
        alarm = int(action) == 1 and not inref  # an alarm that counts (one inside the period is treated as silence)
        reward, term, masked, event, opened = 0.0, False, False, "", False  # defaults
        if typ == POST:  # a post-change second
            if alarm:  # an alarm after the change
                term, event = True, "detection"  # the end of the cost
            else:  # silent (or an alarm inside the period)
                reward = -1.0  # one second of delay
                if self.i == self.ep["last_post"]:  # the interval's last second passes in silence
                    reward, term, event = -1.0 - self.m, True, "miss"  # the miss penalty on top
        elif typ == PRE and alarm:  # an alarm before the change
            reward, term, event = -self.lam, True, "false_alarm"  # the false-alarm penalty
        elif typ == UNC and alarm:  # an alarm in an uncertain second
            masked, opened = True, True  # no reward, no end; it opens the refractory period
        if self.arm == "surrogate":  # the surrogate's S
            self.S = 0.0 if (opened or inref) else self.S_t  # restarted at the alarm and held through the period
        if opened:  # a new period
            self.quiet = REFR  # the next 30 seconds
        elif inref:  # one second of a running period has passed
            self.quiet -= 1  # one less
        info = {**self._info(masked, event), "refractory": inref}  # dict unpacking: the environment script's info plus the period flag
        if term:  # the episode ended
            return self._obs_last(), reward, True, False, info  # terminated
        self.i += 1  # the next second
        if self.i == len(self.rows):  # past the last row (clean episodes only)
            self.i -= 1  # stays on the last row for the observation
            return self._obs_truncated(), reward, False, True, {**info, "event": "truncated"}  # truncated
        return self._obs(), reward, False, False, info  # the next observation

    def _obs_truncated(self) -> np.ndarray:  # method: the environment script's bootstrap observation of a truncated clean episode, with S = 0 inside a period
        o = super()._obs_truncated()  # the environment script's
        if self.arm == "surrogate" and self.quiet > 0:  # if: the next second lies inside a refractory period
            o = o.copy()  # a copy
            o[-1] = 0.0  # S held at 0
        return o  # the observation


def fc_cost(ty: np.ndarray, a: np.ndarray, kind: str, lam: float, m: float) -> tuple:  # function: the F-c undiscounted cost of an action sequence (open loop); (cost, outcome, delay)
    quiet, post = 0, 0  # refractory seconds left; silent post-change seconds
    for i in range(len(ty)):  # for loop over the steps
        inref = quiet > 0  # inside a period?
        al = int(a[i]) == 1 and not inref  # an alarm that counts
        if ty[i] == POST:  # a post-change second
            if al:  # detection
                return float(post), "detection", float(post)  # the delay is the silent post-change seconds
            post += 1  # one more silent post-change second
            if kind == "onset" and i == len(ty) - 1:  # the interval ended in silence
                return post + m, "miss", np.nan  # m on top
        elif ty[i] == PRE and al:  # a false alarm
            return post + lam, "false_alarm", np.nan  # lambda on top
        elif ty[i] == UNC and al:  # an uncertain-second alarm opens a period
            quiet = REFR  # 30 s
            continue  # this second does not count down
        if inref:  # a running period
            quiet -= 1  # counts down
    return 0.0, "clean_end", np.nan  # a clean episode that ended without a counted alarm


def rollout(model, env: FcEnv, ep: dict) -> tuple:  # function: the greedy agent's closed-loop episode under F-c; (actions, cost, outcome, delay)
    rows = ep["rows"]  # the episode's rows
    ty = env.mat["type"][rows]  # step types
    X = env.X[rows]  # state rows
    L = len(rows)  # length

    def seg_obs(start: int, hold: int) -> np.ndarray:  # nested function: the observations from position start on, S from 0 and held at 0 for the first `hold` positions
        if env.arm != "surrogate":  # the primary arm has no S
            return X[start:]  # the rows
        S, s_ = np.zeros(L - start), 0.0  # S per position; the running value
        for k in range(L - start):  # for loop over the positions
            s_ = 0.0 if k < hold else max(0.0, s_ + float(X[start + k, -1]))  # held at 0, then Page's recursion on z_t
            S[k] = s_  # kept
        return T.signed_log(np.column_stack([X[start:], S]))  # P2's compression

    a = model.predict(seg_obs(0, 0), deterministic=True)[0]  # the greedy actions along the uninterrupted path
    base, quiet, post = 0, 0, 0  # where a's first action sits; refractory seconds left; silent post-change seconds
    acts = np.zeros(L, np.int8)  # the actions taken
    for i in range(L):  # for loop over the positions
        ai = int(a[i - base])  # the greedy action here
        acts[i] = ai  # kept
        inref = quiet > 0  # inside a period?
        al = ai == 1 and not inref  # an alarm that counts
        if ty[i] == POST:  # a post-change second
            if al:  # detection
                return acts[:i + 1], float(post), "detection", float(post)  # done
            post += 1  # one silent post-change second
            if ep["kind"] == "onset" and i == L - 1:  # a miss
                return acts, post + env.m, "miss", np.nan  # done
        elif ty[i] == PRE and al:  # a false alarm
            return acts[:i + 1], post + env.lam, "false_alarm", np.nan  # done
        elif ty[i] == UNC and al:  # an uncertain-second alarm: a period opens and S restarts
            quiet = REFR  # 30 s
            if env.arm == "surrogate" and i + 1 < L:  # the surrogate's observations change from here on
                a, base = model.predict(seg_obs(i + 1, REFR), deterministic=True)[0], i + 1  # tuple assignment: re-predicted with S held at 0 for 30 positions
            continue  # this second does not count down
        if inref:  # a running period
            quiet -= 1  # counts down
    return acts, 0.0, "clean_end", np.nan  # a clean episode without a counted alarm


def fc_cusum(ty: np.ndarray, z: np.ndarray, kind: str) -> tuple:  # function: CUSUM "alarm when S >= h" under F-c, every h of H_EP at once; (outcome codes, silent post-change seconds)
    nh = len(H_EP)  # thresholds
    S, quiet, post = np.zeros(nh), np.zeros(nh, int), np.zeros(nh, int)  # tuple assignment: S, refractory left, silent post-change seconds per h
    done, out = np.zeros(nh, bool), np.full(nh, OUTCOME["clean_end"])  # tuple assignment: ended; outcome per h
    for i in range(len(ty)):  # for loop over the steps
        act = ~done  # still running
        inref = act & (quiet > 0)  # inside a period
        S = np.where(inref, 0.0, np.where(act, np.maximum(0.0, S + float(z[i])), S))  # np.where: held at 0, else Page's recursion
        al = act & ~inref & (S >= H_EP)  # an alarm that counts
        if ty[i] == POST:  # a post-change second
            out[al], done[al] = OUTCOME["detection"], True  # detections end
            sil = act & ~al  # silent ones
            post[sil] += 1  # one more silent post-change second
            if kind == "onset" and i == len(ty) - 1:  # the interval's end
                out[sil], done[sil] = OUTCOME["miss"], True  # misses
        elif ty[i] == PRE:  # a pre-change second
            out[al], done[al] = OUTCOME["false_alarm"], True  # false alarms end
        else:  # an uncertain second
            S[al] = 0.0  # S restarts at the alarm
        quiet = np.where(inref, quiet - 1, quiet)  # running periods count down
        if ty[i] == UNC:  # an uncertain second
            quiet[al] = REFR  # alarms open a period
    return out, post  # per h


def cusum_cost(out: np.ndarray, post: np.ndarray, lam: float, m: float) -> np.ndarray:  # function: the F-c cost per h from fc_cusum's outcomes
    return np.where(out == OUTCOME["false_alarm"], post + lam, np.where(out == OUTCOME["miss"], post + m, np.where(out == OUTCOME["detection"], post, 0.0)))  # nested np.where: the registered reward


# ----------------------------------------------------------------------------------------------------------------------
# Run B: demonstrations and DQfD
# ----------------------------------------------------------------------------------------------------------------------
def cusum_paths(env: FcEnv, eps: list) -> list:  # function: (step types, z_t, kind) of each episode
    return [(env.mat["type"][e["rows"]], env.mat["surrogate"][e["rows"], -1], e["kind"]) for e in eps]  # list comprehension


def fit_hstar(paths: list, lam: float, m: float) -> tuple:  # function: h* = the h with the lowest mean F-c cost over the episodes (tie: the smaller h); (index, cost per h)
    C = np.vstack([cusum_cost(*fc_cusum(ty, z, kind), lam, m) for ty, z, kind in paths])  # list comprehension: episodes x h
    mean = C.mean(axis=0)  # mean per h
    return int(np.argmin(mean)), mean  # argmin returns the first, i.e. the smallest h, on a tie


def demo_set(env: FcEnv, eps: list, h: float) -> dict:  # function: CUSUM(h)'s transitions through F-c on the episodes, stored as the agent would store them, Run B)
    obs, act, rew, nxt, done, epi = [], [], [], [], [], []  # the transitions
    wrap = (lambda o: T.signed_log(o)) if env.arm == "surrogate" else (lambda o: np.asarray(o, float))  # lambda: the agent's observation wrapper (P2 for the surrogate)
    for k, e in enumerate(eps):  # for loop over the episodes
        o, _ = env.reset(options={"kind": e["kind"], "session": e["session"], "start": e["start"], "onset": e["onset"]})  # the environment script's replay of the episode
        s_ = 0.0  # the demonstrator's own S (F-c rules)
        while True:  # while loop until the episode ends
            z = float(env.mat["surrogate"][env.rows[env.i], -1])  # z_t at this second
            s_ = 0.0 if env.quiet > 0 else max(0.0, s_ + z)  # S held at 0 inside a period, else Page's recursion
            if env.arm == "surrogate" and abs(s_ - env.S_t) > 1e-9:  # if: the demonstrator's S differs from the agent's input S (it cannot)
                raise SystemExit("STOP: the demonstrator's S differs from the environment's")  # stop
            a = int(s_ >= h)  # the demonstrator's action
            o2, r, term, trunc, info = env.step(a)  # one F-c step
            if a == 1 and info["step_type"] == "uncertain" and not info["refractory"]:  # an uncertain-second alarm that opened a period
                s_ = 0.0  # S restarts
            obs.append(wrap(o)); act.append(a); rew.append(r / T.REWARD_SCALE); nxt.append(wrap(o2)); done.append(float(term)); epi.append(k)  # the transition (six statements on one line: the stored fields)
            o = o2  # the next observation
            if term or trunc:  # the end
                break  # next episode
    O, Nx = np.asarray(obs, np.float32), np.asarray(nxt, np.float32)  # tuple assignment: observations as the network sees them
    rw, dn, ep_ = np.asarray(rew), np.asarray(done, np.float32), np.asarray(epi)  # tuple assignment: rewards, dones, episode numbers
    n = len(rw)  # transitions
    nret, nboot, ndone = np.zeros(n), np.zeros_like(O), np.zeros(n, np.float32)  # tuple assignment: n-step returns, bootstrap observations, n-step dones
    for t in range(n):  # for loop over the transitions: the 10-step targets, cut at the episode's end
        j = t  # the last transition inside the window
        while j + 1 < n and ep_[j + 1] == ep_[t] and j + 1 < t + NSTEP:  # while loop: up to n transitions of the same episode
            j += 1  # one further
        nret[t] = rw[t:j + 1].sum()  # undiscounted sum (gamma 1)
        nboot[t], ndone[t] = Nx[j], dn[j]  # tuple assignment: bootstrap from the window's last next-observation, unless the episode terminated there
    return {"obs": O, "act": np.asarray(act, np.int64), "rew": rw.astype(np.float32), "next": Nx, "done": dn, "nret": nret.astype(np.float32), "nboot": nboot, "ndone": ndone}  # dict literal: the demonstration memory


class DQfD(DQN):  # class: SB3's DQN with a permanent demonstration memory, mixed minibatches, large-margin and n-step losses and pre-training
    def set_demos(self, demo: dict, seed: int, marks: tuple) -> None:  # method: the permanent demonstration memory, its generator and the share-log marks
        self.d = {k: th.as_tensor(v) for k, v in demo.items()}  # dict comprehension: tensors (float32 / int64)
        self.nd = len(demo["act"])  # its size
        self.demo_rng = np.random.default_rng(seed)  # the generator seeded with the job's seed
        self.share_log, self.share_marks = [], set(marks)  # the realised-share records; the steps at which to record

    def _demo_terms(self, qd: th.Tensor, idx: np.ndarray) -> th.Tensor:  # method: lambda1 J_n + lambda2 J_E on the demonstration rows
        ad = self.d["act"][idx].reshape(-1, 1)  # the demonstrator's actions
        q_e = th.gather(qd, 1, ad).squeeze(1)  # Q(s, a_E)
        l = MARGIN * (1.0 - F.one_hot(ad.squeeze(1), qd.shape[1]).float())  # the margin l(a_E, a): 0 for a_E, MARGIN otherwise
        j_e = ((qd + l).max(dim=1)[0] - q_e).mean()  # J_E = max_a [Q(s,a) + l(a_E,a)] - Q(s,a_E), averaged
        with th.no_grad():  # with-block: the n-step target without gradients
            tn = self.d["nret"][idx] + (1.0 - self.d["ndone"][idx]) * self.q_net_target(self.d["nboot"][idx]).max(dim=1)[0]  # 10-step return plus the target network's largest value (gamma 1)
        j_n = F.smooth_l1_loss(q_e, tn)  # SB3's Huber loss, as the 1-step TD loss
        return LAM_N * j_n + LAM_E * j_e  # the weighted sum

    def _step(self, loss: th.Tensor) -> None:  # method: one optimiser step exactly as SB3's train() takes it
        self.policy.optimizer.zero_grad()  # gradients cleared
        loss.backward()  # back-propagated
        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)  # SB3's gradient clipping
        self.policy.optimizer.step()  # the update

    def pretrain(self, k: int) -> None:  # method: k updates on demonstration minibatches alone (Hester et al., 2018, Algorithm 1), target copied every target_update_interval updates
        self.policy.set_training_mode(True)  # training mode (no effect on an MLP, as SB3 sets it)
        for u in range(1, k + 1):  # for loop over the updates
            idx = self.demo_rng.integers(0, self.nd, BATCH)  # a demonstration minibatch
            with th.no_grad():  # with-block: the 1-step target
                tgt = self.d["rew"][idx] + (1.0 - self.d["done"][idx]) * self.q_net_target(self.d["next"][idx]).max(dim=1)[0]  # 1-step TD target (gamma 1)
            q = self.q_net(self.d["obs"][idx])  # Q of the batch
            cur = th.gather(q, 1, self.d["act"][idx].reshape(-1, 1)).squeeze(1)  # Q of the demonstrated actions
            self._step(F.smooth_l1_loss(cur, tgt) + self._demo_terms(q, idx))  # 1-step TD + lambda1 J_n + lambda2 J_E
            if u % self.target_update_interval == 0:  # every 500 updates
                polyak_update(self.q_net.parameters(), self.q_net_target.parameters(), 1.0)  # the target network copied

    def absorption(self) -> dict:  # method: the share of demonstration states where the greedy action equals the demonstrator's (reported, never used)
        with th.no_grad():  # with-block: no gradients
            g = self.q_net(self.d["obs"]).argmax(dim=1).numpy()  # greedy actions
        ad = self.d["act"].numpy()  # the demonstrator's
        return {"agree": float((g == ad).mean()), "agree_on_alarm": float((g[ad == 1] == 1).mean()) if (ad == 1).any() else np.nan, "agree_on_silence": float((g[ad == 0] == 0).mean()) if (ad == 0).any() else np.nan, "demo_transitions": int(self.nd), "demo_alarms": int((ad == 1).sum())}  # dict literal

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:  # method: SB3's DQN.train with 48 own + 16 demonstration rows and the demonstration losses
        self.policy.set_training_mode(True)  # as SB3
        self._update_learning_rate(self.policy.optimizer)  # as SB3 (constant 1e-3)
        for _ in range(gradient_steps):  # for loop over the gradient steps (one per environment step)
            own = self.replay_buffer.sample(batch_size - DEMO, env=self._vec_normalize_env)  # 48 of the agent's own transitions, SB3's uniform draw
            idx = self.demo_rng.integers(0, self.nd, DEMO)  # 16 demonstration transitions, uniform with replacement
            obs = th.cat([own.observations.float(), self.d["obs"][idx]])  # the 64 observations in float32 (SB3's own preprocessing casts them so too)
            nxt = th.cat([own.next_observations.float(), self.d["next"][idx]])  # the 64 next observations
            rew = th.cat([own.rewards.reshape(-1).float(), self.d["rew"][idx]])  # rewards
            dn = th.cat([own.dones.reshape(-1).float(), self.d["done"][idx]])  # dones (SB3's dones already exclude time-outs)
            act = th.cat([own.actions.long().reshape(-1), self.d["act"][idx]]).reshape(-1, 1)  # actions
            with th.no_grad():  # with-block: SB3's 1-step target
                tgt = rew + (1.0 - dn) * self.gamma * self.q_net_target(nxt).max(dim=1)[0]  # 1-step TD target
            q = self.q_net(obs)  # Q of the 64
            cur = th.gather(q, 1, act).squeeze(1)  # Q of the taken actions
            loss = F.smooth_l1_loss(cur, tgt) + self._demo_terms(q[batch_size - DEMO:], idx)  # TD on all 64 + the demonstration losses on the 16
            self._step(loss)  # the update
            if self.num_timesteps in self.share_marks:  # a share-log mark (start, middle, end)
                self.share_log.append({"step": int(self.num_timesteps), "demo_rows": int(len(idx)), "rows": int(len(idx) + len(own.rewards)), "own_buffer": int(self.replay_buffer.size()), "demo_memory": int(self.nd)})  # the realised share
        self._n_updates += gradient_steps  # as SB3


# ----------------------------------------------------------------------------------------------------------------------
# jobs
# ----------------------------------------------------------------------------------------------------------------------
def make_env(mat: dict, arm: str, lam: float):  # function: the training script's training stack around the F-c environment
    env = FcEnv(mat, arm, lam)  # Run A's environment
    if arm == "surrogate":  # P2
        env = T.SignedLog(env)  # compressed observations
    return T.Monitor(T.EpisodeLog(T.ScaleReward(env)))  # the training script's wrappers, Monitor outermost


def guard(spec: dict, eps: list) -> None:  # function: G6's guard: no episode in the outer held-out session, the inner held-out session (training material) or 2.6.12
    bad = {spec["outer"], SEALED} | ({spec["inner"]} if spec["inner"] else set())  # set literal and union: forbidden sessions for training material
    if any(e["session"] in bad for e in eps):  # generator in any(): a forbidden session
        raise SystemExit(f"STOP: {T.job_id(spec)}: a training or demonstration episode in a held-out or sealed session")  # stop


def job_digest(spec: dict, run: str, stop: int) -> str:  # function: SHA-256 of what a job depends on
    d = {"spec": spec, "run": run, "stop": stop, "dqn": T.DQN_KW, "version": FC_VERSION, "consts": [DEMO, BATCH, PRETRAIN, MARGIN, NSTEP, LAM_N, LAM_E, REFR],  # dict literal: the job, the run and the constants
         "code": [T.sha_code(PROJECT / "codes" / n) for n in ("6_reinforcement_learning/01_environment.py", "6_reinforcement_learning/02_train_agents.py", "6_reinforcement_learning/05_followup.py")]}  # list comprehension: the code's syntax trees
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()  # the digest


def snapshot_stats(mdl, eenv: FcEnv, eps: list, tenv: FcEnv, teps: list, sd: pd.DataFrame, cat: np.ndarray, lam: float, held: str) -> dict:  # function: everything an inner snapshot records
    ho = [rollout(mdl, eenv, e) for e in eps]  # closed-loop greedy episodes on the inner held-out session
    tr = [rollout(mdl, tenv, e) for e in teps]  # and on the inner training sessions 
    acts = np.concatenate([x[0] for x in ho]).astype(np.uint8)  # the held-out episodes' actions in order
    hc = np.array([x[1] for x in ho])  # held-out costs
    on = np.array([e["kind"] == "onset" for e in eps])  # onset episodes
    oc = np.array([x[2] for x in ho])  # outcomes
    dl = np.array([x[3] for x in ho], float)  # delays
    tc, tk, to = np.array([x[1] for x in tr]), np.array([e["kind"] for e in teps]), np.array([x[2] for x in tr])  # tuple assignment: training costs, kinds, outcomes
    dd, ds = T.agent_decisions(mdl, eenv, sd)  # the deployed stream, decision and stopping mode (the training script's, unchanged)
    sec, ok, Q, S = D.stream_q(mdl, eenv, sd)  # Q on the decision-mode path (the diagnosis script's)
    mg = (Q[:, 1] - Q[:, 0]).astype(float)  # the margin
    z = np.array([eenv.mat["surrogate"][eenv.mat["key"][(s, int(t))], -1] if (s, int(t)) in eenv.mat["key"] else np.nan for s, t in zip(sd.session, sd.sec)])  # list comprehension: z_t on the grid
    post = ok & np.isin(cat, ["post-change, 0-29 s", "post-change, 30 s on"])  # post-change seconds with a state
    pre = ok & (cat == "pre-change")  # pre-change seconds with a state
    ev = np.asarray(ds) > 0  # stopping-mode events
    lab = sd.label.to_numpy(object)  # scoring labels
    h2 = [D.refractory_from(sec, ev, lab, int(a))[0] and D.refractory_from(sec, ev, lab, int(a))[2] == "uncertain" for a, _ in E.SC.pooled_intervals(held)]  # list comprehension: per onset, refractory from an uncertain-second event
    # dict literal over four lines: the snapshot's record (the 12-compatible fields first)
    return {"criterion": float(hc.mean()), "episodes": len(eps), "onset_detect_share": float((oc[on] == "detection").mean()) if on.any() else None, "onset_miss_share": float((oc[on] == "miss").mean()) if on.any() else None,
            "clean_fa_share": float((oc[~on] == "false_alarm").mean()) if (~on).any() else None, "median_delay": float(np.nanmedian(dl[on])) if np.isfinite(dl[on]).any() else None, "actions_sha256": hashlib.sha256(acts.tobytes()).hexdigest(),  # dict literal continued
            **T.stream_row(sd, dd, held, "decision"), **T.stream_row(sd, ds, held, "stopping"), "train_cost": float(tc.mean()), "train_clean_fa_share": float((to[tk == "clean"] == "false_alarm").mean()) if (tk == "clean").any() else None,  # dict unpacking: the stream figures
            "auroc_margin": D.auroc(mg[post], mg[pre]), "auroc_z": D.auroc(z[post], z[pre]), "h2_flags": [bool(x) for x in h2]}  # dict literal continued


def run_job(job: dict, run: str) -> dict:  # function: one training (runs in a worker process); returns its timing
    th.set_num_threads(1)  # one thread in the worker
    spec, stop = job["spec"], job["stop"]  # the specification and the step to stop at (inner: the schedule; outer: the budget)
    rd = rundir(run)  # the run's folder
    jid = T.job_id(spec)  # the training script's job name
    out, hfile = rd / "jobs" / f"{jid}.json.gz", rd / "jobs" / f"{jid}.sha256"  # tuple assignment: output and hash entry
    dig = job_digest(spec, run, stop)  # the digest
    if out.exists() and hfile.exists() and hfile.read_text().split() == [T.sha_file(out), dig]:  # if: done before with the same inputs
        return {"job": jid, "skipped": True}  # nothing to do
    T.set_random_seed(spec["seed"])  # as the training script's run_job
    train, evm = T.job_material(spec, STATES12)  # the training script's materials
    env = make_env(train, spec["arm"], spec["lambda"])  # the F-c training stack
    model = (DQfD if run == "B" else DQN)("MlpPolicy", env, seed=spec["seed"], device="cpu", verbose=0, **T.DQN_KW)  # conditional expression: Run B's subclass or SB3's DQN
    tenv = FcEnv(train, spec["arm"], spec["lambda"])  # an environment over the training sessions (episodes and rollouts)
    teps = T.eval_episodes(tenv)  # P4's rule on the training sessions (D1; Run B's demonstration episodes)
    guard(spec, teps)  # G6
    paths = cusum_paths(tenv, teps)  # (types, z, kind) of each
    hi, hmean = fit_hstar(paths, spec["lambda"], train["m"])  # h* on the job's own training sessions under F-c
    res = {"spec": spec, "run": run, "digest": dig, "h_star": float(H_EP[hi]), "cusum_train_cost": float(hmean[hi]), "checkpoints": []}  # the job's result
    if run == "B":  # demonstrations
        demo = demo_set(FcEnv(train, spec["arm"], spec["lambda"]), teps, float(H_EP[hi]))  # CUSUM(h*)'s transitions
        model.set_demos(demo, spec["seed"], (T.DQN_KW["learning_starts"] + 1, stop // 2, stop - 1))  # the memory, its generator, the share-log marks (stop - 1: an outer run ends before the update at its budget)
        res["demo_sha256"] = hashlib.sha256(b"".join(np.ascontiguousarray(v).tobytes() for v in demo.values())).hexdigest()  # generator in join(): the demonstration memory's digest
        model.pretrain(PRETRAIN)  # 10,000 updates on demonstrations alone
        res["absorption"] = model.absorption()  # the agreement after pre-training (reported, never used)
    if evm is not None:  # an inner job
        eenv = FcEnv(evm, spec["arm"], spec["lambda"])  # the inner held-out environment
        eps = T.eval_episodes(eenv)  # P4's episodes
        res["trivial"] = {}  # the three trivial policies under F-c
        for p in T.TRIVIAL:  # for loop over them
            cs = []  # costs
            for e in eps:  # for loop over the episodes
                ty = eenv.mat["type"][e["rows"]]  # types
                a = np.zeros(len(ty), int) if p == "never" else (np.r_[1, np.zeros(len(ty) - 1, int)] if p == "first_step" else np.ones(len(ty), int))  # nested conditional expression: the policy's actions
                cs.append(fc_cost(ty, a, e["kind"], spec["lambda"], eenv.m)[0])  # its cost
            res["trivial"][p] = {"criterion": float(np.mean(cs))}  # the mean
        res["m"] = evm["m"]  # the inner miss penalty
        ehi = cusum_paths(eenv, eps)  # the held-out episodes' paths
        res["cusum_heldout_cost"] = float(np.mean([cusum_cost(*fc_cusum(ty, z, k), spec["lambda"], evm["m"])[hi] for ty, z, k in ehi]))  # list comprehension: CUSUM(h*) on the inner held-out episodes
        sd = T.session_seconds(T.read_steps(), spec["receiver"], spec["inner"])  # the inner held-out stream
        on = pd.read_csv(E.ENV_DIR / "onsets.csv", dtype={"session": str})  # attack runs
        cat = D.category(sd.label.to_numpy(object), sd.step.to_numpy(object), sd.sec.to_numpy(np.int64), E.SC.pooled_intervals(spec["inner"]), on[(on.receiver == spec["receiver"]) & (on.session == spec["inner"])])  # D3's categories

        def on_mark(mdl, n):  # nested function: one snapshot
            res["checkpoints"].append({"budget": n, **snapshot_stats(mdl, eenv, eps, tenv, teps, sd, cat, spec["lambda"], spec["inner"]), "train_episodes": len(env.get_episode_rewards())})  # dict unpacking: the record
        cb = T.Snapshots(T.BUDGETS, on_mark, None)  # every snapshot
    else:  # an outer job: stop at the budget and save the weights
        res["m"] = train["m"]  # the outer miss penalty

        def on_mark(mdl, n):  # nested function: the freeze of the weights
            w = {**{f"q_net.{k}": v.detach().cpu().numpy() for k, v in mdl.q_net.state_dict().items()}, **{f"q_net_target.{k}": v.detach().cpu().numpy() for k, v in mdl.q_net_target.state_dict().items()}}  # dict comprehensions: both networks
            path = rd / "models" / f"{jid}.npz"  # the model file
            E.write_npz(w, path)  # fixed member dates
            res["checkpoints"].append({"budget": n, "model_file": path.name, "model_sha256": T.sha_file(path), "train_episodes": len(env.get_episode_rewards())})  # recorded
        cb = T.Snapshots((stop,), on_mark, stop)  # one mark and the stop
    t0 = time.perf_counter()  # clock
    model.learn(total_timesteps=T.SCHEDULE, callback=cb)  # the schedule (an outer run stops at its budget)
    dt = time.perf_counter() - t0  # seconds
    if run == "B":  # the realised demonstration share at start, middle and end
        res["share_log"] = model.share_log  # recorded
    log, ret = env.get_wrapper_attr("done_eps"), env.get_episode_rewards()  # EpisodeLog's episodes and Monitor's returns
    res["episodes"] = [{**e, "monitor_return": float(r)} for e, r in zip(log, ret)]  # list comprehension: each episode with its return
    T.write_atomic(out, T.gz_json(res))  # the output
    T.write_atomic(hfile, f"{T.sha_file(out)} {dig}\n".encode())  # its hash entry
    return {"job": jid, "skipped": False, "seconds": round(dt, 1), "steps": int(model.num_timesteps)}  # the timing


STATES12 = T.AGENT_DIR / T.GRID / "states"  # the agent experiment's inner primary states (read only)


def run_jobs(jobs: list, run: str, tag: str) -> pd.DataFrame:  # function: jobs through a pool of three spawned workers; returns the timing table
    ctx = multiprocessing.get_context("spawn")  # fresh interpreters
    rows = []  # timings
    with ProcessPoolExecutor(max_workers=T.N_WORKERS, mp_context=ctx) as ex:  # with-block: the pool
        futs = {ex.submit(run_job, j, run): j for j in jobs}  # dict comprehension: future -> job
        for k, f in enumerate(as_completed(futs), 1):  # for loop over finished jobs
            r = f.result()  # its timing (a worker's exception is raised here)
            rows.append(r)  # kept
            print(f"  [{tag} {k}/{len(jobs)}] {r['job']}: {'skipped' if r['skipped'] else str(r['seconds']) + ' s'}", flush=True)  # progress
    return pd.DataFrame(rows).sort_values("job").reset_index(drop=True)  # in a fixed order


def inner_jobs(cells=None) -> list:  # function: every inner job (or those of the given (receiver, outer, lambda) cells), run to the full schedule
    return [{"spec": s, "stop": T.SCHEDULE} for s in T.specs_inner() if cells is None or (s["receiver"], s["outer"], s["lambda"]) in cells]  # list comprehension: the training script's inner specifications


# ----------------------------------------------------------------------------------------------------------------------
# steps
# ----------------------------------------------------------------------------------------------------------------------
def pilot(run: str) -> None:  # function: the one-cell timing pilot (mosaic-X5, outer 2.6.2, lambda 100, both arms: 16 inner jobs and one outer job per arm)
    cell = {("SBF_02", "2.6.2", 100.0)}  # set literal: the pilot cell
    rd = rundir(run)  # the run's folder
    for sub in ("jobs", "models"):  # the folders
        (rd / sub).mkdir(parents=True, exist_ok=True)  # created
    t0 = time.perf_counter()  # clock
    tim = run_jobs(inner_jobs(cell), run, "pilot inner")  # the inner jobs
    outer = [{"spec": {"stage": "outer", "receiver": "SBF_02", "outer": "2.6.2", "inner": "", "arm": arm, "lambda": 100.0, "seed": 20260914, "budget": 50_000}, "stop": 50_000} for arm in ARMS]  # list comprehension: one outer job per arm at a mid budget
    tim = pd.concat([tim, run_jobs(outer, run, "pilot outer")], ignore_index=True)  # and the outer ones
    wall = time.perf_counter() - t0  # seconds
    n_inner, n_outer = 1200, 750  # the grid's job counts
    sel = D.rt(T.AGENT_DIR / T.GRID / "selection.csv")  # the agent experiment's budgets, for the outer estimate only
    est = (tim[tim.job.str.startswith("inner")].seconds.mean() * n_inner + tim[tim.job.str.startswith("outer")].seconds.mean() / 50_000 * sel.selected_budget.mean() * n_outer) / T.N_WORKERS / 3600  # the grid's estimated wall hours (3 workers)
    E.write_csv(tim.assign(pilot_wall_s=round(wall, 1), grid_estimate_h=round(est, 2)), rd / "followup_pilot_timing.csv")  # declared variable (wall clock)
    print(f"pilot run {run}: wall {wall:.0f} s; grid estimate {est:.2f} h", flush=True)  # the estimate


def grid(run: str) -> None:  # function: --grid: inner jobs, selection, outer jobs, the freeze tables (a run again resumes)
    rd = rundir(run)  # the run's folder
    for sub in ("jobs", "models", "freeze"):  # the folders
        (rd / sub).mkdir(parents=True, exist_ok=True)  # created
    t0 = time.perf_counter()  # clock
    tim = run_jobs(inner_jobs(), run, "inner")  # 1,200 inner jobs
    specs = T.specs_inner()  # their specifications
    cur = T.inner_curve(rd / "jobs", specs)  # every snapshot of every inner job (the training script's reader)
    E.write_gz(cur.drop(columns=["h2_flags"]), rd / "freeze" / "inner_curve.csv.gz")  # the inner curve (H2 flags kept in the job files)
    sel = T.select(cur)  # P3 and P10 (the training script's rules, unchanged)
    E.write_csv(sel, rd / "freeze" / "selection.csv")  # written
    E.write_csv(T.learned(sel), rd / "freeze" / "learned.csv")  # P10 as amended
    outer = [{"spec": s, "stop": s["budget"]} for s in T.specs_outer(sel)]  # list comprehension: 750 outer jobs at their budgets
    tim = pd.concat([tim, run_jobs(outer, run, "outer")], ignore_index=True)  # trained
    E.write_csv(tim.assign(grid_wall_h=round((time.perf_counter() - t0) / 3600, 2)), rd / "followup_grid_timing.csv")  # declared variable
    nested = D.rt(T.AGENT_DIR / T.GRID / "nested_predictions.csv.gz")  # P's nested predictions (training sessions only)
    pts, _ = B12.points_table(sel, nested, T.read_steps())  # P6 as amended: the baselines recomputed, the agents' lambdas
    old = D.rt(T.AGENT_DIR / "evaluation" / "deployable_points.csv")  # the agent experiment's points
    b, o = pts[~pts.detector.str.startswith("agent")].reset_index(drop=True), old[~old.detector.str.startswith("agent")].reset_index(drop=True)  # tuple assignment: the baselines now and in the agent experiment
    if not (np.array_equal(b.param.to_numpy(float), o.param.to_numpy(float), equal_nan=True) and b[["receiver", "outer", "detector", "share"]].equals(o[["receiver", "outer", "detector", "share"]])):  # if: a baseline point moved (it cannot)
        raise SystemExit("STOP: the baselines' points differ from the agent experiment's")  # stop
    E.write_csv(pts.assign(label=LABEL), rd / "freeze" / "points.csv")  # the points, labelled
    imp = T.implied(T.specs_outer(sel), rd / "models")  # P11 on the frozen agents (the training script's)
    E.write_csv(imp, rd / "freeze" / "implied.csv")  # written
    E.write_csv(T.implied_summary(imp)[1], rd / "freeze" / "implied_spearman.csv")  # rho per receiver, fold and arm
    man = []  # the models' manifest
    for s in T.specs_outer(sel):  # for loop over the outer agents
        p = rd / "models" / f"{T.job_id(s)}.npz"  # the model file
        r = T.read_gz_json(rd / "jobs" / f"{T.job_id(s)}.json.gz")  # its job output
        man.append({"model_file": p.name, "sha256": T.sha_file(p), "job_sha256": T.sha_file(rd / "jobs" / f"{T.job_id(s)}.json.gz"), "demo_sha256": r.get("demo_sha256", ""), "h_star": r["h_star"]})  # the row
    E.write_csv(pd.DataFrame(man), rd / "freeze" / "models_manifest.csv")  # written (the models themselves are kept apart, their SHA-256 in the manifest)
    cfg = {**{k: str(v) for k, v in T.DQN_KW.items()}, "run": run, "DEMO": DEMO, "BATCH": BATCH, "PRETRAIN": PRETRAIN, "MARGIN": MARGIN, "NSTEP": NSTEP, "LAM_N": LAM_N, "LAM_E": LAM_E, "REFRACTORY": REFR,  # dict literal: the configuration
           "sb3": T.sb3.__version__, "torch": th.__version__, **{f"code_{n}": T.sha_code(PROJECT / "codes" / n) for n in ("6_reinforcement_learning/01_environment.py", "6_reinforcement_learning/02_train_agents.py", "6_reinforcement_learning/03_evaluate_agents.py", "6_reinforcement_learning/05_followup.py")}}  # dict comprehension: the code digests
    E.write_csv(pd.DataFrame([cfg]), rd / "freeze" / "config.csv")  # written
    fz = sorted(p for p in (rd / "freeze").iterdir() if p.name != "freeze_manifest.csv")  # generator in sorted(): the frozen files
    E.write_csv(pd.DataFrame({"file": [p.name for p in fz], "sha256": [T.sha_file(p) for p in fz]}), rd / "freeze" / "freeze_manifest.csv")  # the freeze manifest
    print(f"run {run}frozen tables written; commit them as 'follow-up FREEZE run {run}' before --heldout", flush=True)  # the next step


def git(*args) -> subprocess.CompletedProcess:  # function: a read-only git command in the folder; without a repository, the frozen-input table answers it
    from helpers import frozen_git, in_repository  # the table's reader and the repository test (codes/common/helpers.py, on the import path)
    if in_repository(PROJECT):  # if: the folder sits in a git repository with this history
        return subprocess.run(["git", *args], cwd=PROJECT, capture_output=True)  # its result (bytes)
    return frozen_git(PROJECT, args)  # the answer from Data/processed/checks/frozen_inputs.csv, in the same form


def check_freeze(run: str, commit: str) -> pd.DataFrame:  # function: the freeze guard: every frozen file equals its blob in the freeze commit; every model equals the manifest
    rd = rundir(run)  # the run's folder
    rows = []  # one row per file
    ok_c = git("merge-base", "--is-ancestor", commit, "HEAD").returncode == 0 and b'follow-up FREEZE run ' + run.encode() in git("log", "-1", "--format=%s", commit).stdout  # the commit is in HEAD's history and is this run's freeze
    for f in sorted((rd / "freeze").iterdir()):  # for loop over the frozen files
        rel = str(f.relative_to(PROJECT))  # its path
        at = git("show", f"{commit}:{rel}")  # its bytes in the freeze commit
        rows.append({"file": rel, "in_freeze_commit": at.returncode == 0 and ok_c, "equal": at.stdout == f.read_bytes()})  # the row
    man = pd.read_csv(rd / "freeze" / "models_manifest.csv")  # the models' manifest
    for mf, h in zip(man.model_file, man.sha256):  # for loop over the models
        rows.append({"file": f"models/{mf}", "in_freeze_commit": ok_c, "equal": T.sha_file(rd / "models" / mf) == h})  # equal to the manifest
    return pd.DataFrame(rows)  # the table


def write_gz_csv(df: pd.DataFrame, path: Path) -> None:  # function: the environment script's byte-stable gzip CSV
    E.write_gz(df, path)  # written


def heldout(run: str, commit: str, dry: str | None = None) -> None:  # function: --heldout: the single outer read; dry: an inner training session read instead (G7, no outer read)
    rd = rundir(run)  # the run's folder
    tag = "heldout" if dry is None else f"dryrun_{dry}"  # the output folder
    out = rd / tag  # it
    if dry is None and (rd / "outer_read.json").exists():  # refused once read
        raise SystemExit(f"STOP: run {run}'s outer sessions were read already (outer_read.json); a second read needs --deviation")  # refuse
    ck = check_freeze(run, commit)  # the freeze guard
    out.mkdir(parents=True, exist_ok=True)  # the folder
    E.write_csv(ck, out / "freeze_check.csv")  # written
    if not (ck.in_freeze_commit & ck.equal).all():  # if: anything differs from the freeze
        raise SystemExit("STOP: the frozen files or models do not match the freeze commit; nothing is read")  # refuse
    sel = D.rt(rd / "freeze" / "selection.csv")  # the frozen selection
    steps = T.read_steps()  # labels
    t_start = time.strftime("%Y-%m-%dT%H:%M:%S")  # start time (declared variable)
    runs, lat, reads = [], [], []  # run rows; interval rows; inputs read
    for rx in RXS:  # for loop over the receivers
        for outer in SESSIONS:  # for loop over the outer folds
            sess = outer if dry is None else dry  # the session read: the held-out one, or a training session in the dry run
            if dry is not None and (sess == outer or (rx, outer) != ("SBF_02", "2.6.2")):  # the dry run never reads a held-out session, and runs one fold only (G7 proves the plumbing)
                continue  # skip
            mat = B12.heldout_material(rx, outer) if dry is None else {**E.material(rx, outer)}  # the held-out rows' key, or the training rows' (dry run)
            sd = T.session_seconds(steps, rx, sess)  # the stream grid
            reads.append({"receiver": rx, "outer": outer, "session": sess, "states_sha256": T.sha_file(E.ENV_DIR / f"states_{rx}_{outer}.npz")})  # the input read
            dec_rows = []  # per-second decisions of this fold
            for arm in ARMS:  # for loop over the arms
                for lam in LAMBDAS:  # for loop over the lambdas
                    b = int(sel[(sel.receiver == rx) & (sel.outer == outer) & (sel.arm == arm) & (sel["lambda"] == float(lam))].selected_budget.iloc[0])  # the frozen budget
                    for seed in T.OUTER_SEEDS:  # for loop over the seeds
                        spec = {"stage": "outer", "receiver": rx, "outer": outer, "inner": "", "arm": arm, "lambda": float(lam), "seed": seed, "budget": b}  # the frozen agent
                        model, _ = T.load_q(spec, rd / "models", mat)  # loaded (the training script's loader)
                        env = E.SpoofingDetectionEnv(mat, arm, float(lam))  # matrices and key only
                        d_, s_ = T.agent_decisions(model, env, sd)  # decision and stopping mode (the deployed stream, unchanged)
                        fd, fs = B12.both(sd, d_, s_, sess)  # through the environment script's stream
                        runs.append({"receiver": rx, "outer": outer, "detector": f"agent_{arm}", "share": np.nan, "param": float(lam), "lambda": float(lam), "seed": seed, **B12.run_row(fd, fs)})  # dict unpacking: the run
                        lat += [{"receiver": rx, "outer": outer, "detector": f"agent_{arm}", "share": np.nan, "lambda": float(lam), "seed": seed, **x} for x in fs["latency"]]  # list comprehension: its intervals
                        dec_rows.append(pd.DataFrame({"sec": sd.sec.to_numpy(np.int64), "label": sd.label.to_numpy(object), "arm": arm, "lambda": float(lam), "seed": seed, "dec": np.asarray(d_, np.int8), "stop": np.asarray(s_, np.int8)}))  # the per-second decisions
            write_gz_csv(pd.concat(dec_rows, ignore_index=True), out / f"decisions_{rx}_{outer}.csv.gz")  # written (every table can be recomputed from these)
            print(f"  {tag}: {rx} {outer} read", flush=True)  # progress
    R, LAT = pd.DataFrame(runs), pd.DataFrame(lat)  # the agents' tables
    E.write_csv(R, out / "agent_runs.csv"); E.write_csv(LAT, out / "agent_latency.csv")  # two statements: written
    if dry is None:  # the real read: its log
        rec = {"run": run, "freeze_commit": commit, "head": git("rev-parse", "HEAD").stdout.decode().strip(), "start": t_start, "end": time.strftime("%Y-%m-%dT%H:%M:%S"), "reads": reads, "label": LABEL}  # dict literal: the read's record
        (rd / "outer_read.json").write_text(json.dumps(rec, indent=1, sort_keys=True))  # outer_read.json (declared variable: times)
        log = FOLLOWUP_DIR / "diagnosis_outer_reads.csv"  # the log of every outer read
        row = pd.DataFrame([{"run": run, "freeze_commit": commit, "head": rec["head"], "start": rec["start"], "end": rec["end"], "folds_read": len(reads)}])  # the row
        E.write_csv(pd.concat([pd.read_csv(log), row], ignore_index=True) if log.exists() else row, log)  # appended
        tables(run)  # the tables from what was just written


def tables(run: str) -> None:  # function: every held-out table of a run from the stored agent runs and the agent experiment's committed baseline runs (also --recompute)
    rd = rundir(run)  # the run's folder
    out = rd / "heldout"  # the folder
    ev = T.AGENT_DIR / "evaluation"  # the agent experiment's committed evaluation
    pts = D.rt(rd / "freeze" / "points.csv").drop(columns=["label"])  # the frozen points
    Ra, La = D.rt(out / "agent_runs.csv"), D.rt(out / "agent_latency.csv")  # the agents
    Rb, Lb = D.rt(ev / "heldout_runs.csv"), D.rt(ev / "heldout_latency.csv")  # the agent experiment's runs (baselines kept, the agent experiment's agents dropped)
    R = pd.concat([Rb[Rb.detector.isin(B12.BASE)], Ra], ignore_index=True)  # every run
    LAT = pd.concat([Lb[Lb.detector.isin(B12.BASE)], La], ignore_index=True)  # every interval
    C = D.rt(ev / "curve_runs.csv.gz")  # the agent experiment's baseline curve runs
    at = B12.point_runs(R, pts)  # the runs at the points
    ag = B12.aggregate(at, LAT, pts)  # P8
    vd = B12.verdicts(at, LAT, pts)  # P9
    cv = B12.curves(R, LAT, C)  # P7
    for name, df in (("point_runs", at), ("aggregate", ag), ("verdicts", vd), ("curves", cv)):  # for loop over the tables
        E.write_csv(df.assign(label=LABEL), out / f"followup_{run}_{name}.csv")  # written, labelled


def recompute(run: str) -> None:  # function: --recompute: the agent tables again from the per-second decisions alone, then every table; byte comparison (no second read)
    rd = rundir(run)  # the run's folder
    out = rd / "heldout"  # the folder
    before = {p.name: T.sha_file(p) for p in sorted(out.glob("*.csv"))}  # dict comprehension: digests now
    steps = T.read_steps()  # labels
    runs, lat = [], []  # rebuilt rows
    for rx in RXS:  # for loop over the receivers
        for outer in SESSIONS:  # for loop over the folds
            dd = pd.read_csv(out / f"decisions_{rx}_{outer}.csv.gz")  # the stored decisions
            sd = T.session_seconds(steps, rx, outer)  # the grid
            for (arm, lam, seed), g in dd.groupby(["arm", "lambda", "seed"], sort=False):  # for loop over the runs in stored order
                fd, fs = B12.both(sd, g.dec.to_numpy(), g.stop.to_numpy(), outer)  # the environment script's stream from the stored decisions
                runs.append({"receiver": rx, "outer": outer, "detector": f"agent_{arm}", "share": np.nan, "param": float(lam), "lambda": float(lam), "seed": int(seed), **B12.run_row(fd, fs)})  # the run
                lat += [{"receiver": rx, "outer": outer, "detector": f"agent_{arm}", "share": np.nan, "lambda": float(lam), "seed": int(seed), **x} for x in fs["latency"]]  # list comprehension: its intervals
    E.write_csv(pd.DataFrame(runs), out / "agent_runs.csv"); E.write_csv(pd.DataFrame(lat), out / "agent_latency.csv")  # two statements: rewritten
    tables(run)  # every table
    after = {p.name: T.sha_file(p) for p in sorted(out.glob("*.csv"))}  # dict comprehension: digests after
    diff = sorted(k for k in before if before.get(k) != after.get(k) and k != "freeze_check.csv")  # generator in sorted(): files that changed
    E.write_csv(pd.DataFrame([{"files": len(before), "differing": ";".join(diff), "identical": not diff}]), rd / "followup_recompute.csv")  # the outcome
    print(f"recompute run {run}: {'identical' if not diff else 'DIFFERS: ' + ';'.join(diff)}", flush=True)  # the outcome


def diag(run: str) -> None:  # function: --diag: D1, D3 and H2 on inner folds at the selected budgets, judged closed or open by the diagnosis rules; absorption; demo share
    rd = rundir(run)  # the run's folder
    sel = D.rt(rd / "freeze" / "selection.csv")  # the frozen selection
    steps, nested = T.read_steps(), D.rt(T.AGENT_DIR / T.GRID / "nested_predictions.csv.gz")  # tuple assignment: labels; nested predictions (training sessions only)
    rows, onsets, absorb, share = [], [], [], []  # D1/D3 rows per run; H2 rows; absorption; demo share
    for rx in RXS:  # for loop over the receivers
        for outer in SESSIONS:  # for loop over the outer folds
            ctx = D.inner_ctx(rx, outer, steps, nested)  # the diagnosis script's inner context (training sessions only)
            for s in [x for x in T.specs_inner() if x["receiver"] == rx and x["outer"] == outer]:  # list comprehension: the fold's inner jobs
                r = T.read_gz_json(rd / "jobs" / f"{T.job_id(s)}.json.gz")  # the job
                b = int(sel[(sel.receiver == rx) & (sel.outer == outer) & (sel.arm == s["arm"]) & (sel["lambda"] == s["lambda"])].selected_budget.iloc[0])  # the selected budget
                c = [x for x in r["checkpoints"] if x["budget"] == b][0]  # list comprehension: its snapshot
                base = {k: s[k] for k in ("receiver", "outer", "inner", "arm", "lambda", "seed")}  # dict comprehension: the run's identity
                rows.append({**base, "budget": b, "agent_train_cost": c["train_cost"], "cusum_train_cost": r["cusum_train_cost"], "agent_heldout_cost": c["criterion"], "cusum_heldout_cost": r["cusum_heldout_cost"], "h_star": r["h_star"],  # dict literal: D1
                             "agent_train_clean_fa_share": c["train_clean_fa_share"], "auroc_margin": c["auroc_margin"], "auroc_z": c["auroc_z"]})  # and D3
                q = float(sel[(sel.receiver == rx) & (sel.outer == outer) & (sel.arm == s["arm"]) & (sel["lambda"] == s["lambda"])].inner_decision_clean_share.iloc[0])  # the agent's inner share (P6's quantity)
                for a_, f in zip(E.SC.pooled_intervals(s["inner"]), c["h2_flags"]):  # for loop over the onsets: the agent's H2 flags
                    onsets.append({**base, "detector": "agent", "onset": a_[0], "refractory_uncertain": f})  # the row
                if s["seed"] == T.INNER_SEEDS[0]:  # once per (fold, arm, lambda): the baselines at the agent's share (seed-free)
                    h_q, t_q = D.thresholds(ctx, q)  # CUSUM and the regression at q
                    bl = D.baseline_runs(ctx["per"][s["inner"]], h_q, t_q)  # both on the inner held-out stream
                    v = ctx["per"][s["inner"]]  # the stream
                    for det, (_, ev, _) in bl.items():  # for loop over the baselines
                        for a_, _ in v["pooled"]:  # for loop over the onsets
                            rf, _, el = D.refractory_from(v["sec"], np.asarray(ev, bool), v["lab"], a_)  # the mark and its event's label
                            onsets.append({**base, "seed": -1, "detector": det, "onset": a_, "refractory_uncertain": bool(rf and el == "uncertain")})  # the row
                if run == "B":  # Run B's records
                    absorb.append({**base, **r["absorption"]})  # the agreement after pre-training
                    share += [{**base, **x, "share": x["demo_rows"] / x["rows"]} for x in r["share_log"]]  # list comprehension: the realised share at start, middle, end
            print(f"  diag {run}: {rx} {outer}", flush=True)  # progress
    Rr, O = pd.DataFrame(rows), pd.DataFrame(onsets)  # the tables
    d1 = D.d1_cells(Rr.assign(agent_train_cost_mix=np.nan, cusum_train_cost_mix=np.nan, cusum_train_clean_fa_share=np.nan, agent_train_onset_fa_share=np.nan, cusum_train_onset_fa_share=np.nan, agent_train_miss_share=np.nan,  # the diagnosis script's cell function (columns it does not need here are blank)
                                  kappa_cusum=np.nan, jaccard_cusum=np.nan, share_alarms_S_ge_h=np.nan, event_coincidence_5s=np.nan, h_q=np.nan, implied_S=np.nan, clean_share_beyond_train_S=np.nan, clean_alarms_beyond_share=np.nan, implied_S_in_range=np.nan))  # assign continued
    d3 = Rr.groupby(["arm", "receiver", "lambda"], sort=True).agg(auroc_margin_median=("auroc_margin", "median"), auroc_z_median=("auroc_z", "median")).reset_index()  # D3's cells
    d3["margin_below_z"] = (d3.auroc_margin_median - d3.auroc_z_median) < -0.05  # the diagnosis rule per cell
    h2 = D.h2_cells(O.assign(event_age_s=np.nan, event_label="", refractory=O.refractory_uncertain, onset_state=""))  # the diagnosis script's cell function
    s_ = d1[d1.arm == "surrogate"]  # D1's surrogate cells
    st = [{"defect": "D1 optimisation (surrogate)", "cells_meeting_rule": int(s_.optimisation_failure.sum()), "cells": len(s_)}]  # list literal: D1
    st += [{"defect": f"D3 representation ({arm})", "cells_meeting_rule": int(d3[d3.arm == arm].margin_below_z.sum()), "cells": int((d3.arm == arm).sum())} for arm in ARMS]  # list comprehension: D3 per arm
    st += [{"defect": "H2 refractory at onset (primary)", "cells_meeting_rule": int(h2[h2.arm == "primary"].h2_cell.sum()), "cells": int((h2.arm == "primary").sum())}]  # H2 primary
    S = pd.DataFrame(st)  # the closed/open table
    S["status"] = np.where(S.cells_meeting_rule >= D.CELL_MIN, "open", "closed")  # open while the diagnosis defect rule is met (8 of 15)
    for name, df in (("d1_runs", Rr), ("d1_cells", d1), ("d3_cells", d3), ("h2_onsets", O), ("h2_cells", h2), ("closed_open", S)):  # for loop over the tables
        E.write_csv(df.assign(label=LABEL), rd / f"followup_{run}_{name}.csv")  # written, labelled
    if run == "B":  # Run B's records
        E.write_csv(pd.DataFrame(absorb).assign(label=LABEL), rd / "followup_B_absorption.csv")  # the agreement after pre-training
        E.write_csv(pd.DataFrame(share).assign(label=LABEL), rd / "followup_B_demo_share.csv")  # the realised share
    print(S.to_string(index=False), flush=True)  # the outcome


def paired(at1: pd.DataFrame, at2: pd.DataFrame, n1: str, n2: str) -> pd.DataFrame:  # function: paired differences of median delay, run 2 minus run 1, on the same receiver, share, arm, fold and seed (descriptive)
    rows = []  # one row per receiver, share and arm
    for (rx, s, det), d2 in at2[at2.detector.str.startswith("agent")].groupby(["receiver", "share", "detector"], sort=False):  # for loop over the cells of run 2
        d1 = at1[(at1.receiver == rx) & (at1.share == s) & (at1.detector == det)]  # run 1's runs
        j = d2.merge(d1[["outer", "seed", "stop_delay_median_s", "stop_miss_rate"]], on=["outer", "seed"], suffixes=("", "_1"))  # paired by fold and seed
        x = (j.stop_delay_median_s - j.stop_delay_median_s_1).to_numpy(float)  # the differences
        p, lo, hi, n = B12.strat_boot(x, j.outer.to_numpy()) if len(j) else (np.nan, np.nan, np.nan, 0)  # IQM with the stratified interval
        rows.append({"receiver": rx, "share": s, "arm": det.replace("agent_", ""), "comparison": f"{n2} minus {n1}", "pairs": n, "delay_diff_iqm_s": p, "ci_low": lo, "ci_high": hi, "miss_diff_mean": float((j.stop_miss_rate - j.stop_miss_rate_1).mean()) if len(j) else np.nan})  # the row
    return pd.DataFrame(rows)  # the table


def report_all() -> None:  # function: --report: the agent experiment's registered verdicts beside Runs A and B, the closed/open table beside the diagnosis, paired differences, the figures
    import matplotlib.pyplot as plt  # pyplot (09 chose the file-only backend)
    agent_runs = D.rt(T.AGENT_DIR / "evaluation" / "verdicts.csv")  # the registered verdicts (confirmatory, unchanged)
    va, vb = (D.rt(rundir(r) / "heldout" / f"followup_{r}_verdicts.csv") for r in ("A", "B"))  # generator unpacked: the two runs' verdicts
    key = ["receiver", "share", "arm"]  # the cell
    cols = ["best_baseline", "verdict", "delay_diff_iqm_s", "delay_diff_ci_low", "delay_diff_ci_high", "agent_miss_rate", "baseline_miss_rate_agent_folds", "folds_with_point"]  # the columns kept
    side = agent_runs[key + cols].merge(va[key + cols], on=key, suffixes=("_main", "_A")).merge(vb[key + cols].rename(columns={c: c + "_B" for c in cols}), on=key)  # dict comprehension in rename: the three side by side
    E.write_csv(side.assign(label=LABEL), FOLLOWUP_DIR / "followup_verdicts_side_by_side.csv")  # written
    diag_verdicts = D.rt(D.DIAG_DIR / "diagnosis_verdicts.csv")  # the diagnosis verdicts
    d1 = int(diag_verdicts[(diag_verdicts.diagnostic == "D1(b) optimisation test") & (diag_verdicts.arm == "surrogate")].cells_meeting_defect.iloc[0])  # the diagnosis D1 count
    d3c = D.rt(D.DIAG_DIR / "diagnosis_learning_cells.csv")  # the diagnosis D3 cells
    h2c = D.rt(D.DIAG_DIR / "diagnosis_refractory_cells.csv")  # the diagnosis H2 cells
    diagnosis_counts = {"D1 optimisation (surrogate)": d1, "D3 representation (primary)": int(d3c[d3c.arm == "primary"].margin_below_z.sum()), "D3 representation (surrogate)": int(d3c[d3c.arm == "surrogate"].margin_below_z.sum()), "H2 refractory at onset (primary)": int(h2c[h2c.arm == "primary"].h2_cell.sum())}  # dict literal: the diagnosis counts
    co = pd.DataFrame({"defect": list(diagnosis_counts), "cells": 15, "main_agents": list(diagnosis_counts.values())})  # the table
    for r in ("A", "B"):  # for loop over the runs
        s = D.rt(rundir(r) / f"followup_{r}_closed_open.csv").set_index("defect")  # the run's closed/open table
        co[f"run{r}"] = co.defect.map(s.cells_meeting_rule)  # its counts
        co[f"run{r}_status"] = co.defect.map(s.status)  # and status
    co["main_status"] = np.where(co.main_agents >= D.CELL_MIN, "open", "closed")  # the diagnosis status by the same rule
    E.write_csv(co.assign(label=LABEL), FOLLOWUP_DIR / "followup_closed_open.csv")  # written
    pa, pb = (D.rt(rundir(r) / "heldout" / f"followup_{r}_point_runs.csv") for r in ("A", "B"))  # generator unpacked: the point runs
    p12 = D.rt(T.AGENT_DIR / "evaluation" / "point_runs.csv")  # the agent experiment's point runs
    pdiff = pd.concat([paired(p12, pa, "main", "Run A"), paired(pa, pb, "Run A", "Run B")], ignore_index=True)  # A minus the agent experiment, B minus A
    E.write_csv(pdiff.assign(label=LABEL), FOLLOWUP_DIR / "followup_paired_diffs.csv")  # written
    NAME, VC = E.NAME, {"better": "#009E73", "no difference detectable": "#E69F00", "worse": "#D55E00"}  # tuple assignment: receiver names; verdict colours
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True)  # the figure: one panel per source
    for ax, sfx, ttl in zip(axes, ("_main", "_A", "_B"), ("the agent experiment (registered, confirmatory)", "Run A (F-c)", "Run B (F-c + demonstrations)")):  # for loop over the panels
        for i, r in enumerate(side.itertuples()):  # for loop over the 18 cells
            v = getattr(r, "verdict" + sfx)  # the verdict
            lo, mid, hi = getattr(r, "delay_diff_ci_low" + sfx), getattr(r, "delay_diff_iqm_s" + sfx), getattr(r, "delay_diff_ci_high" + sfx)  # tuple assignment: the paired delay difference
            col = VC.get(v, "#999999")  # its colour (grey: no point)
            if np.isfinite(mid):  # a point with a difference
                ax.errorbar(mid, i, xerr=[[mid - lo], [hi - mid]] if np.isfinite(lo) else None, fmt="o", color=col, ms=5, lw=1)  # the IQM and its interval
            else:  # no point or no pair
                ax.plot(0, i, "x", color=col, ms=6)  # marked at 0
        ax.axvline(0, color="black", lw=0.8); ax.set_title(ttl, fontsize=9, loc="left"); ax.set_xlabel("agent minus best baseline, median delay (s)")  # three statements
        ax.set_xscale("symlog", linthresh=10)  # a symmetric log scale
    axes[0].set_yticks(range(len(side))); axes[0].set_yticklabels([f"{NAME[r.receiver]} {r.share:g} {r.arm}" for r in side.itertuples()], fontsize=7)  # two statements: cell labels
    fig.suptitle(f"Follow-up runs against the agent experiment: P9 verdicts (green better, orange no difference, red worse, grey x no point); {LABEL}", fontsize=8.5)  # title
    fig.tight_layout(); E.M.save_fig(fig, E.GRAPH / "diagnosis_verdicts.png"); plt.close(fig)  # three statements
    fig, ax = plt.subplots(figsize=(8, 4))  # the figure: the defects' cell counts
    x = np.arange(len(co))  # positions
    for k, (c_, col, lab) in enumerate((("main_agents", "#999999", "the agent experiment's agents"), ("runA", "#56B4E9", "Run A"), ("runB", "#CC79A7", "Run B"))):  # for loop over the three sources
        ax.bar(x + (k - 1) * 0.27, co[c_], width=0.27, color=col, label=lab)  # bars
    ax.axhline(D.CELL_MIN, color="black", ls="--", lw=0.8); ax.set_xticks(x); ax.set_xticklabels([d_.replace(" (", "\n(") for d_ in co.defect], fontsize=7.5); ax.set_ylabel("cells meeting the defect rule (of 15)"); ax.legend(fontsize=7, frameon=False)  # five statements
    fig.suptitle(f"Defects diagnosed per run: each diagnosed defect on inner folds, open at 8 or more cells (dashed); {LABEL}", fontsize=8.5)  # title
    fig.tight_layout(); E.M.save_fig(fig, E.GRAPH / "diagnosis_defects.png"); plt.close(fig)  # three statements
    ab = D.rt(rundir("B") / "followup_B_absorption.csv")  # Run B's absorption
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)  # the figure: agreement after pre-training
    for k, arm in enumerate(ARMS):  # for loop over the arms
        g = ab[ab.arm == arm]  # its jobs
        ax[k].boxplot([g[g.receiver == rx].agree_on_alarm.dropna() for rx in RXS] + [g[g.receiver == rx].agree_on_silence.dropna() for rx in RXS], labels=[f"{NAME[rx]}\nalarm" for rx in RXS] + [f"{NAME[rx]}\nsilence" for rx in RXS])  # list comprehensions: agreement on the demonstrator's alarms and silences
        ax[k].set_title(f"{arm} arm", fontsize=9, loc="left"); ax[k].tick_params(axis="x", labelsize=6.5)  # two statements
    ax[0].set_ylabel("agreement with the demonstrator")  # label (short, so it fits)
    fig.suptitle(f"Imitation of CUSUM after pre-training. Run B: how far pre-training made the network imitate CUSUM(h*), per inner job; {LABEL}", fontsize=8.5)  # title
    fig.tight_layout(); E.M.save_fig(fig, E.GRAPH / "diagnosis_absorption.png"); plt.close(fig)  # three statements
    print(side[["receiver", "share", "arm", "verdict_main", "verdict_A", "verdict_B"]].to_string(index=False)); print(co.to_string(index=False))  # two statements: printed


def main() -> None:  # function: the command line
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])  # the parser
    ap.add_argument("step", choices=["pilot", "grid", "heldout", "dryrun", "recompute", "diag", "report"])  # the step
    ap.add_argument("run", choices=["A", "B"], nargs="?", default="A")  # the run (report: both)
    ap.add_argument("--freeze", default="", help="the run's freeze commit (heldout, dryrun)")  # the freeze commit
    ap.add_argument("--session", default="2.6.4", help="dryrun: the training session read instead of the held-out one")  # G7's session
    a = ap.parse_args()  # parsed
    if th.get_num_threads() != 1:  # one thread, asserted
        raise SystemExit("STOP: torch is not on one thread")  # stop
    {"pilot": lambda: pilot(a.run), "grid": lambda: grid(a.run), "heldout": lambda: heldout(a.run, a.freeze), "dryrun": lambda: heldout(a.run, a.freeze, a.session),  # dict of lambdas: the step's function
     "recompute": lambda: recompute(a.run), "diag": lambda: diag(a.run), "report": report_all}[a.step]()  # looked up and called


if __name__ == "__main__":  # if: run as a script
    main()  # the command line
