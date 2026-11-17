#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
Data checks 1 and 2 (the supervisor's checks of 19 October 2026): provenance of the six session folders, and the motion of the vehicle.

Check 1  the six folders used, their capture start times (README, first 1 Hz system stamp, first receiver epoch), every file's size
         and SHA-256, compared with the hash list made after the captures were copied (rawcheck_end.txt), and whether any folder is one of the three the
         supervisor calls archived and corrupted.
Check 2  per session: the README's "User Remarks" verbatim; the OBD-II SPEED count, maximum, the seconds with speed above zero and
         their runs; the mosaic-X5's own velocity (PVTGeodetic Vn, Ve, Vu decoded with the receiver-status script's CRC-checked reader) with its
         maximum and the seconds above 0.5 m/s, and the label of each such second; every labelled second within 30 s of OBD motion.
2.6.12 is sealed: its OBD figures are read from the logged preprocessing/vehicle_motion_obd_sessions.csv and its X5 velocity is not computed; its
labels (allowed) are read. The mosaic-X5 velocity existed before only for 2.6.5 up to the X5 gain onset; here it is computed for
the five unsealed sessions from the raw SBF, read only.

Reads   Data/Day1_14092026/<six folders>/* (read only), Data/processed/design_study/hashes/rawcheck_end.txt, preprocessing/rinex_verification.csv,
        ground_truth/session_ground_truth.csv, preprocessing/vehicle_motion_obd_sessions.csv, labels/labels_1hz_final.csv, labels/train_labels_1hz.csv
Writes  Data/processed/data_checks/provenance_files.csv, provenance_sessions.csv, motion_obd.csv, motion_obd_runs.csv,
        motion_x5_velocity.csv, motion_x5_fast_runs.csv, motion_labelled_near.csv, provenance_motion_checks.csv
"""

from __future__ import annotations  # future import: type hints kept as text (the project's convention)

import hashlib  # standard-library hashing: SHA-256 of each raw file
import importlib.util  # standard-library module loader: the receiver-status script's name starts with a digit
import os  # operating-system access: MPLCONFIGDIR is set before the receiver-status script loads matplotlib
import re  # standard-library regular expressions: the README's Start line and User Remarks
import struct  # standard-library binary decoding: the velocity fields of PVTGeodetic
import sys  # standard-library system access: sys.path and sys.modules
from pathlib import Path  # Path objects for building file paths

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
os.environ["MPLCONFIGDIR"] = str(PROJECT / "codes" / ".mplconfig")  # matplotlib's cache inside the project folder

import numpy as np  # NumPy: arrays
import pandas as pd  # pandas: tables

sys.path.insert(0, str(PROJECT / "codes" / "common"))  # codes/ first on the import path, so the helpers module is found
from helpers import exit_if_failed, gps_to_utc  # status 1 on a failed check; receiver week and time of week -> UTC

PROC = PROJECT / "Data" / "processed"  # the processed-data folder
RAW = PROJECT / "Data" / "Day1_14092026"  # the raw recordings: read only
OUT = PROC / "data_checks"  # this check's output folder
SEALED = "2.6.12"  # the sealed session
# list: the folders the supervisor calls archived and corrupted (folder names as he gave them)
ARCHIVED = ["20260914_155633_T_2.6.2", "20260914_183945_T_2.6.12", "20260914_184030_T_2.6.12"]
V_FAST = 0.5  # float constant: the speed limit in m/s for the X5 velocity
NEAR_S = 30  # integer constant: "within 30 s of motion"
RXS = ["SBF_02", "SBF_03", "UBX_01"]  # list: the three receivers that carry labels
CHECKS: list = []  # list: one dict per self-check
FAILED: list = []  # list: names of the checks that failed


def check(name: str, ok, detail: str) -> None:  # function: records one self-check and prints it (ok None marks a reported number)
    CHECKS.append({"check": name, "result": "REPORT" if ok is None else ("PASS" if ok else "FAIL"), "detail": detail})  # nested conditional expression picks the word
    if ok is not None and not bool(ok):  # if: a pass/fail check that did not pass
        FAILED.append(name)  # kept for the exit status
    print(f"  [{CHECKS[-1]['result']}] {name}: {detail}")  # printed for the run log


def write_csv(df: pd.DataFrame, name: str) -> None:  # function: one table into Data/processed/data_checks with fixed line endings (byte-stable)
    OUT.mkdir(parents=True, exist_ok=True)  # the folder, created on first use
    df.to_csv(OUT / name, index=False, lineterminator="\n")  # index=False: no row numbers; "\n": the same bytes everywhere


def sha(path: Path) -> str:  # function: SHA-256 of a file, read in 1 MB pieces
    h = hashlib.sha256()  # a fresh hash object
    with open(path, "rb") as fh:  # with-block: the file opened for binary reading, closed afterwards
        for chunk in iter(lambda: fh.read(1 << 20), b""):  # iter with a sentinel: reads until the empty bytes object at the end
            h.update(chunk)  # adds the piece
    return h.hexdigest()  # 64 hex characters


def runs(secs, gap: int = 1) -> list:  # function: sorted whole seconds -> (first, last) runs whose neighbours lie at most gap seconds apart
    s = np.sort(np.asarray(list(secs), dtype=np.int64))  # the seconds in order
    if not len(s):  # if: none
        return []  # no run
    br = np.flatnonzero(np.diff(s) > gap)  # positions where the gap to the next second exceeds gap
    return list(zip(np.r_[s[0], s[br + 1]], np.r_[s[br], s[-1]]))  # np.r_ joins arrays: starts and ends paired by zip


def hms(sec) -> str:  # function: unix seconds -> "HH:MM:SS" UTC
    return pd.Timestamp(int(sec), unit="s").strftime("%H:%M:%S")  # integer seconds formatted as clock time


def load(name: str, filename: str):  # function: loads codes/<filename> as a module (its main() stays behind its guard)
    spec = importlib.util.spec_from_file_location(name, PROJECT / "codes" / filename)  # where from
    mod = importlib.util.module_from_spec(spec)  # an empty module object
    sys.modules[name] = mod  # registered so its imports resolve
    spec.loader.exec_module(mod)  # its top level (definitions only)
    return mod  # the module


# ---------------------------------------------------------------- check 1
def check1(gt: pd.DataFrame) -> None:  # function: folders, start times, files and hashes
    ref = {}  # relative path -> SHA-256 in the design study's list
    for line in (PROC / "design_study" / "hashes" / "rawcheck_end.txt").read_text().splitlines():  # for loop over the list's lines ("raw\t<sha>  <path>")
        parts = line.split("\t", 1)  # the tag and the rest
        if len(parts) == 2 and parts[0] == "raw":  # if: a raw-file line
            h, p = parts[1].split("  ", 1)  # the hash and the path, separated by two spaces
            ref[p] = h  # kept
    ver = pd.read_csv(PROC / "preprocessing" / "rinex_verification.csv", dtype=str)  # the logged first and last receiver epochs (the conversion script)
    files, sess = [], []  # one dict per file; one dict per session
    for r in gt.itertuples():  # for loop over the six sessions
        d = RAW / r.session_folder  # the folder
        readme = (d / "README.txt").read_text()  # its README
        start = re.search(r"Start:\s+(\S+)", readme).group(1)  # the README's start, laptop clock with its offset
        first_ns = int(pd.read_csv(d / "timestamps_system_capture.csv", nrows=1).unix_time_ns.iloc[0])  # the first 1 Hz system stamp, nanoseconds since 1970
        v = ver[ver.test_code == r.test_code].set_index("stream").binary_first_utc  # the first receiver epoch of each stream
        fl = sorted(p for p in d.rglob("*") if p.is_file() and p.name != ".DS_Store")  # every file of the folder, macOS view settings left out
        for p in fl:  # for loop over its files
            rel = str(p.relative_to(PROJECT))  # path relative to the project folder, as the design study wrote it
            h = sha(p)  # its hash now
            files.append({"folder": r.session_folder, "file": str(p.relative_to(d)), "bytes": p.stat().st_size, "sha256": h, "sha256_rawcheck": ref.get(rel, ""), "equal_to_rawcheck": h == ref.get(rel, "")})  # one row
        # dict literal: the session's identity, its three start readings and whether the folder is archived
        sess.append({"test_code": r.test_code, "folder": r.session_folder, "readme_start_local": start, "readme_start_utc": pd.Timestamp(start).tz_convert("UTC").strftime("%H:%M:%S.%f")[:-3],
                     "first_system_stamp_utc": pd.Timestamp(first_ns, unit="ns").strftime("%H:%M:%S.%f")[:-3], "first_epoch_x5": v.get("SBF_02", ""), "first_epoch_mosaic_h": v.get("SBF_03", ""),  # continued: more fields of the row
                     "first_epoch_ublox01": v.get("UBX_01", ""), "files": len(fl), "bytes": int(sum(p.stat().st_size for p in fl)), "archived_name": r.session_folder in ARCHIVED})  # the row
    F, Sx = pd.DataFrame(files), pd.DataFrame(sess)  # the two tables
    write_csv(F, "provenance_files.csv")  # every file
    write_csv(Sx, "provenance_sessions.csv")  # every session
    check("every raw file's SHA-256 equals the hash list made after the captures were copied (rawcheck_end.txt)", bool(F.equal_to_rawcheck.all()), f"{int(F.equal_to_rawcheck.sum())} of {len(F)}")  # provenance unchanged
    check("no session folder is one of the three archived folders", not Sx.archived_name.any() and not any((RAW / a).exists() for a in ARCHIVED), ", ".join(Sx.folder))  # the supervisor's item (b)


# ---------------------------------------------------------------- check 2
def remarks(gt: pd.DataFrame) -> dict:  # function: session -> the README's User Remarks, verbatim
    out = {}  # the remarks
    for r in gt.itertuples():  # for loop over the sessions
        txt = (RAW / r.session_folder / "README.txt").read_text()  # the README
        out[r.test_code] = txt.split("User Remarks\n------------\n", 1)[1].strip()  # the text after the heading and its underline
    return out  # the dict


def obd(gt: pd.DataFrame, rem: dict) -> tuple:  # function: the OBD-II SPEED figures per session, and the moving seconds
    rows, run_rows, moving = [], [], {}  # summary rows; run rows; session -> set of moving seconds
    logged = pd.read_csv(PROC / "preprocessing" / "vehicle_motion_obd_sessions.csv", dtype={"test_code": str}).set_index("test_code")  # the motion script's logged summary
    for r in gt.itertuples():  # for loop over the sessions
        t = r.test_code  # the session
        if t == SEALED:  # if: the sealed session: logged figures only
            m = logged.loc[t]  # its row
            # rows.append adds a dict: the sealed session's figures from the motion script's logged summary
            rows.append({"test_code": t, "remarks": rem[t], "source": "preprocessing/vehicle_motion_obd_sessions.csv (logged; sealed session)", "speed_rows": int(m.speed_rows), "valid_rows": int(m.valid_rows),
                         "max_kmh": float(m.max_speed_kmh), "samples_above_0": int(m.moving_samples), "seconds_above_0": 0 if float(m.moving_s) == 0 else np.nan, "first_above_0_utc": "", "last_above_0_utc": ""})  # the row
            moving[t] = set()  # no motion logged
            continue  # next session
        d = pd.read_csv(next((RAW / r.session_folder).glob("*_OBD-II_CSV_01.csv")), usecols=["signal", "value", "valid_value", "unix_time_ns"])  # the raw OBD file, four columns
        d = d[d.signal == "SPEED"].copy()  # SPEED rows only
        d["valid"] = d.valid_value.astype(str) == "True"  # the logger's validity flag as a boolean
        d["kmh"] = pd.to_numeric(d.value, errors="coerce").where(d.valid)  # km/h; an invalid reading is missing, not zero
        d["sec"] = d.unix_time_ns.astype(np.int64) // 1_000_000_000  # the UTC whole second (integer division of nanoseconds)
        mv = d[d.kmh > 0]  # samples above zero
        moving[t] = set(mv.sec.tolist())  # the seconds that hold one
        for a, b in runs(moving[t]):  # for loop over the runs of moving seconds
            run_rows.append({"test_code": t, "start_utc": hms(a), "end_utc": hms(b), "seconds": int(b - a + 1)})  # one run
        # dict literal: the session's figures from the raw file
        rows.append({"test_code": t, "remarks": rem[t], "source": "raw OBD-II CSV", "speed_rows": len(d), "valid_rows": int(d.valid.sum()), "max_kmh": float(d.kmh.max()), "samples_above_0": len(mv),
                     "seconds_above_0": len(moving[t]), "first_above_0_utc": pd.Timestamp(int(mv.unix_time_ns.iloc[0]), unit="ns").strftime("%H:%M:%S.%f")[:-3] if len(mv) else "",  # continued: more fields of the row
                     "last_above_0_utc": pd.Timestamp(int(mv.unix_time_ns.iloc[-1]), unit="ns").strftime("%H:%M:%S.%f")[:-3] if len(mv) else ""})  # the row
    O = pd.DataFrame(rows)  # the table
    x = O.set_index("test_code").join(logged[["speed_rows", "valid_rows", "moving_samples"]], rsuffix="_05a")  # beside the motion script's logged counts
    check("OBD counts (rows, valid rows, samples above zero) = the motion script's preprocessing/vehicle_motion_obd_sessions.csv", bool((x.speed_rows == x.speed_rows_05a).all() and (x.valid_rows == x.valid_rows_05a).all() and (x.samples_above_0 == x.moving_samples).all()), "six sessions")  # an independent count
    return O, pd.DataFrame(run_rows), moving  # the three


def x5_velocity(gt: pd.DataFrame, lab: pd.DataFrame, moving: dict) -> tuple:  # function: the mosaic-X5's own velocity per session, and its seconds above 0.5 m/s
    S03 = load("receiver_status_for_provenance", "1_ground_truth/03_receiver_status.py")  # the receiver-status script: the CRC-checked SBF reader
    rows, fast_rows = [], []  # summary rows; runs of fast seconds
    for r in gt.itertuples():  # for loop over the sessions
        t = r.test_code  # the session
        if t == SEALED:  # if: sealed
            rows.append({"test_code": t, "note": "not computed: 2.6.12 is sealed and no logged X5 velocity series exists"})  # said
            continue  # next session
        rec = []  # one tuple per PVTGeodetic block
        for blk, rev, tow, wnc, p, end in S03.sbf_blocks(next((RAW / r.session_folder).glob("*_SBF_02.sbf")), S03.Integrity(), {4007}):  # generator: CRC-verified PVTGeodetic blocks (4007)
            tt = gps_to_utc(tow, wnc)  # receiver time -> UTC, or None for a Do-Not-Use time
            if tt is None or len(p) < 42:  # if: no valid time, or too short to hold Vn, Ve and Vu
                continue  # skipped
            mode, err = p[0] & 0x0F, p[1]  # "& 0x0F" keeps bits 0-3: the solution type; the next byte is the error code
            vn, ve, vu = struct.unpack_from("<fff", p, 30)  # "<fff": three little-endian 4-byte floats at bytes 30-41, m/s
            ok = mode not in (0, 3) and err == 0 and min(vn, ve, vu) > -1e10  # a computed solution (not "no PVT" or fixed) and no Do-Not-Use value (-2e10)
            rec.append((pd.Timestamp(tt).value, ok, vn, ve, vu))  # nanoseconds and the fields
        x = pd.DataFrame(rec, columns=["t_ns", "ok", "vn", "ve", "vu"])  # the blocks
        n_all = len(x)  # every block
        x = x[x.ok].copy()  # those with a velocity
        x["v3"] = np.sqrt(x.vn ** 2 + x.ve ** 2 + x.vu ** 2)  # 3-D speed, m/s
        x["vh"] = np.hypot(x.vn, x.ve)  # horizontal speed, m/s
        x["sec"] = x.t_ns // 1_000_000_000  # UTC whole second
        fs = sorted(set(x.sec[x.v3 > V_FAST].tolist()))  # seconds holding a sample above 0.5 m/s
        Lx = lab[(lab.test_code == t) & (lab.receiver == "SBF_02")].set_index("sec").label  # the X5's labels by second
        lab_of = Lx.reindex(fs).fillna("no label row")  # the label of each fast second (none before the capture start)
        near = [s for s in fs if any(abs(s - m) <= 5 for m in moving[t])]  # list comprehension: fast seconds within 5 s of OBD motion
        k = int(x.v3.idxmax())  # the row of the largest speed
        # dict literal: the session's velocity figures
        rows.append({"test_code": t, "pvt_blocks": n_all, "blocks_with_velocity": len(x), "max_v3_ms": round(float(x.v3.max()), 3), "max_vh_ms": round(float(x.vh.max()), 3),
                     "max_at_utc": pd.Timestamp(int(x.loc[k, "t_ns"]), unit="ns").strftime("%H:%M:%S.%f")[:-3], "seconds_above_0.5": len(fs), "within_5s_of_obd_motion": len(near),  # continued: more fields of the row
                     "labels_of_those_seconds": ";".join(f"{a}={b}" for a, b in lab_of.value_counts().sort_index().items()), "clean_seconds_above_0.5": int((lab_of == "clean").sum())})  # the row
        for a, b in runs([s for s in fs if s not in near], gap=2):  # for loop over the runs of fast seconds away from OBD motion
            fast_rows.append({"test_code": t, "start_utc": hms(a), "end_utc": hms(b), "x5_label_at_start": Lx.get(a, "no label row"), "max_v3_ms": round(float(x.v3[(x.sec >= a) & (x.sec <= b)].max()), 3)})  # one run
    return pd.DataFrame(rows), pd.DataFrame(fast_rows)  # two tables


def labelled_near(lab: pd.DataFrame, tr: pd.DataFrame, moving: dict) -> pd.DataFrame:  # function: every labelled second within 30 s of OBD motion, as runs per receiver
    rows = []  # one dict per run
    for t, mv in moving.items():  # for loop over the sessions with their moving seconds
        if not mv:  # if: no motion
            continue  # nothing near it
        near = set()  # seconds within 30 s of a moving second
        for m in mv:  # for loop over the moving seconds
            near |= set(range(m - NEAR_S, m + NEAR_S + 1))  # "|=": set union with the second's neighbourhood
        for rx in RXS:  # for loop over the receivers
            x = lab[(lab.test_code == t) & (lab.receiver == rx) & lab.sec.isin(near)]  # the receiver's seconds near motion
            T = tr[(tr.test_code == t) & (tr.receiver == rx)].set_index("sec").train_label  # its training labels by second
            for moving_flag in (False, True):  # for loop: the seconds next to motion, then the moving seconds themselves
                y = x[x.sec.isin(mv) == moving_flag]  # those seconds
                y = y[y.label != "excluded"]  # labelled ones only
                for lb, g in y.groupby("label", sort=True):  # for loop over the labels present
                    for a, b in runs(g.sec):  # for loop over their runs
                        # rows.append adds a dict: one run of labelled seconds with its training labels
                        rows.append({"test_code": t, "receiver": rx, "moving_itself": moving_flag, "label": lb, "start_utc": hms(a), "end_utc": hms(b), "seconds": int(b - a + 1),
                                     "train_labels": ";".join(f"{k}={v}" for k, v in T.reindex(range(int(a), int(b) + 1)).value_counts().sort_index().items())})  # one run with its training labels
    return pd.DataFrame(rows)  # the table


def main() -> None:  # function: checks 1 and 2 from start to finish
    gt = pd.read_csv(PROC / "ground_truth" / "session_ground_truth.csv", dtype={"test_code": str})  # the six sessions and their folders
    check1(gt)  # check 1
    lab = pd.read_csv(PROC / "labels" / "labels_1hz_final.csv", dtype={"test_code": str}, keep_default_na=False, usecols=["test_code", "receiver", "sec", "label"])  # the final labels
    tr = pd.read_csv(PROC / "labels" / "train_labels_1hz.csv", dtype={"test_code": str}, keep_default_na=False, usecols=["test_code", "receiver", "sec", "train_label"])  # the training labels
    O, OR, moving = obd(gt, remarks(gt))  # OBD
    write_csv(O, "motion_obd.csv")  # written
    write_csv(OR, "motion_obd_runs.csv")  # written
    vm = pd.read_csv(PROC / "preprocessing" / "vehicle_motion.csv", dtype={"test_code": str}).drop_duplicates()  # the motion script's segments (the file repeats two rows)
    mv = vm[(vm.test_code == "2.6.5") & (vm.state == "moving")].sort_values("segment")  # 2.6.5's two drives
    check("thesis statement: 2.6.5 drives of 33.8 s and 140.4 s, the second ending at 12:34:57", [round(v, 1) for v in mv.duration_s] == [33.8, 140.4] and str(mv.end_utc.iloc[-1])[11:19] == "12:34:57", f"{list(mv.duration_s)}; end {mv.end_utc.iloc[-1]}")  # the thesis's numbers
    check("thesis statement: the other five sessions never move (OBD above zero)", bool((O[O.test_code != "2.6.5"].samples_above_0 == 0).all()), "five sessions")  # the thesis's claim
    V, VF = x5_velocity(gt, lab, moving)  # X5 velocity
    write_csv(V, "motion_x5_velocity.csv")  # written
    write_csv(VF, "motion_x5_fast_runs.csv")  # written
    N = labelled_near(lab, tr, moving)  # labelled seconds near motion
    write_csv(N, "motion_labelled_near.csv")  # written
    nn = N[~N.moving_itself]  # the seconds next to motion
    check("labelled seconds within 30 s of motion carry no clean or attack label and no training label", bool(nn.label.isin(["uncertain"]).all() and nn.train_labels.str.fullmatch(r"none=\d+").all()), f"{int(nn.seconds.sum())} receiver-seconds, labels {sorted(set(nn.label))}")  # the guard
    mi = N[N.moving_itself]  # moving seconds that carry a label other than excluded
    check("moving seconds labelled other than excluded (reported)", None, "; ".join(f"{r.receiver} {r.start_utc}-{r.end_utc} {r.label} ({r.train_labels})" for r in mi.itertuples()) or "none")  # 12:32:36
    pd.DataFrame(CHECKS).to_csv(OUT / "provenance_motion_checks.csv", index=False, lineterminator="\n")  # the checks


if __name__ == "__main__":  # if: true only when the file is run directly
    main()  # runs checks 1 and 2
    exit_if_failed(FAILED, "8_data_checks/01_provenance_and_motion.py")  # the helpers module: status 1 when any check failed
