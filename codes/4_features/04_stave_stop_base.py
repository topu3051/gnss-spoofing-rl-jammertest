"""The 2.6.5 base quantities extended back to the Stave stop, now with the 36
geometry-block quantities (raw Doppler, Doppler rate, single difference of the pseudorange; median, largest absolute value and standard
deviation over each constellation's satellites) beside the orbit-free block that label rule 5's file held.

Method: the earlier Stave base script, unchanged in substance. the feature script's own functions are imported and the evidence script's module constant SITE_START is
overridden in memory to start 2.6.5 at 12:27:56, the end of the first drive. build_265() also gives the Stave windows of the final label set
(unrounded, on the full label grid), with SITE_START either the Stave start or the committed 12:34:57.
Preconditions (label rule 5's, extended to the geometry columns): (i) from 12:34:59 on, every orbit-free and geometry cell equals the committed
base table at its 4 decimals; (ii) at 12:34:57-12:34:58 every differing cell is a difference-type cell the committed base leaves missing
for want of a previous epoch; (iii) the orbit-free columns of the new file equal label rule 5's committed stave_base_265.csv.gz cell for cell.
If any fails, nothing is written.
Writes Data/processed/features/stave_base_265_final.csv.gz (2.6.5, the three receivers, 12:27:56-12:54:24, orbit-free and geometry
columns, 4 decimals) and stave_base_b_checks.csv.
"""
import gzip  # standard library: gzip with a fixed header time (byte-stable)
import importlib.util  # standard library: loads the project scripts as modules (their names start with a digit)
import sys  # standard library: module registry and exit
from pathlib import Path  # standard library: paths

import pandas as pd  # pandas: tables

PROJECT = Path(__file__).resolve().parents[2]  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
OUT = PROJECT / "Data" / "processed" / "features"  # this step's outputs
STOP_START = "12:27:56"  # the end of the first 2.6.5 drive (preprocessing/vehicle_motion.csv segment 1 ends 12:27:55.85)
SITE_START_COMMITTED = "12:34:57"  # the evidence script's committed site start for 2.6.5
STOP_END = "12:32:36"  # the last second of the Stave stop (the second drive starts 12:32:37)
DIFF_TYPES = ("dC", "d2C", "dop_rate", "cmc_rate", "n_new", "n_lost", "d_n_tracked")  # tuple: name parts of the columns that need a previous epoch (label rule 5's list)


def load(name: str, filename: str):  # function: loads codes/<filename> as a module, unchanged
    spec = importlib.util.spec_from_file_location(name, PROJECT / "codes" / filename)  # where to load from
    mod = importlib.util.module_from_spec(spec)  # an empty module
    sys.modules[name] = mod  # registered so its own imports resolve
    spec.loader.exec_module(mod)  # runs its top level (main() is guarded)
    return mod  # the module


F = load("window_features_for_stave_stop_base", "4_features/03_window_features.py")  # 07, which loads the residual script, the observables script, the evidence script and the position script itself
F.free_lineage()  # 07 records the orbit-free and geometry column dictionary
FREE = [c for c, v in F.LINEAGE.items() if v[0] == "orbit-free"]  # list comprehension: the orbit-free base columns, in the feature script's order
GEOM = [c for c, v in F.LINEAGE.items() if v[0] == "geometry"]  # the geometry-block base columns (36)


def T(h: str) -> int:  # function: "HH:MM:SS" on 14 September 2026 -> unix seconds
    return int(pd.Timestamp(f"2026-09-14 {h}", tz="UTC").timestamp())  # whole seconds


def build_265(site_start: str) -> pd.DataFrame:  # function: 2.6.5's unrounded orbit-free and geometry base, three receivers, on the full label grid, from site_start
    D = F.D  # the evidence script as 07 uses it
    gt = pd.read_csv(F.PROC / "ground_truth" / "session_ground_truth.csv", usecols=F.GT_COLS).set_index("test_code")  # sessions, the columns the feature script reads
    grid = pd.read_csv(F.PROC / "labels" / "labels_1hz.csv", usecols=F.KEYS, dtype={"test_code": str})  # the label grid, keys only (as in the window-feature script)
    lam = F.wavelengths()  # carrier wavelengths (the window-feature script)
    committed = dict(D.SITE_START)  # the constant as committed
    if committed != {"2.6.5": SITE_START_COMMITTED}:  # if: the evidence script's constant is not what the committed run used
        raise SystemExit(f"STOP: the evidence script SITE_START is {committed}")  # stop
    D.SITE_START = {"2.6.5": site_start}  # the in-memory override
    parts = []  # per receiver
    try:  # try/finally: the constant always comes back
        for rx in F.RXS:  # the three receivers
            o, (a, b) = F.read_full(gt, "2.6.5", rx)  # observations from site_start (the feature script's reader, the evidence script's span)
            o = F.per_satellite(o, lam)  # the feature script's per-satellite quantities
            g = grid[(grid.test_code == "2.6.5") & (grid.receiver == rx)]  # the pair's grid seconds
            inside = g.sec[(g.sec >= a) & (g.sec < b)].to_numpy()  # those inside the span
            blk = F.free_block(o, inside)  # the feature script's orbit-free block (orbit-free and geometry quantities)
            parts.append(g[F.KEYS].merge(blk[["sec"] + FREE + GEOM], on="sec", how="left"))  # the pair on its full grid, as the feature script's build_base
    finally:  # always
        D.SITE_START = committed  # restored
    return pd.concat(parts, ignore_index=True)  # every receiver


def main() -> None:  # function: builds, checks and writes the Stave base file
    OUT.mkdir(parents=True, exist_ok=True)  # output folder
    cols = FREE + GEOM  # the columns checked and written
    ext = build_265(STOP_START).round(4)  # the extended base at the feature script's rounding
    com = pd.read_csv(F.OUT / "features_base_1hz.csv.gz")  # the committed base table
    com = com[com.test_code == "2.6.5"][F.KEYS + cols]  # its 2.6.5 rows
    m = ext.merge(com, on=F.KEYS, how="inner", suffixes=("", "_c"))  # aligned on the keys
    same = lambda x, y: (x == y) | (pd.isna(x) & pd.isna(y))  # lambda: equal values, or both missing
    rows = []  # checks
    late = m[m.sec >= T("12:34:59")]  # from 12:34:59
    n_bad = int(sum((~same(late[c], late[c + "_c"])).sum() for c in cols))  # differing cells there
    rows.append({"check": "orbit-free and geometry base from 12:34:59 to the capture end equals the committed base", "cells": int(len(late) * len(cols)), "differing": n_bad, "pass": n_bad == 0})  # (i)
    edge = m[(m.sec >= T("12:34:57")) & (m.sec <= T("12:34:58"))]  # the two edge seconds
    diff = [(c, r) for c in cols for r in edge.index if not same(edge.at[r, c], edge.at[r, c + "_c"])]  # list comprehension: differing cells
    allowed = all(pd.isna(edge.at[r, c + "_c"]) and any(k in c for k in DIFF_TYPES) for c, r in diff)  # generator in all(): each a missing committed difference-type cell
    rows.append({"check": "at 12:34:57-12:34:58 every difference is a difference-type cell the committed base leaves missing", "cells": int(len(edge) * len(cols)), "differing": len(diff), "pass": bool(allowed)})  # (ii)
    stb = ext[(ext.sec >= T(STOP_START))].reset_index(drop=True)  # the extended rows, Stave start to the capture end (label rule 5's span)
    old = pd.read_csv(PROJECT / "Data" / "processed" / "design_study" / "labels" / "stave_base_265.csv.gz")  # label rule 5's committed file (orbit-free only)
    k = stb[F.KEYS + FREE].merge(old, on=F.KEYS, how="outer", suffixes=("", "_o"), indicator=True)  # outer join: every row of either
    n_old = int(sum((~same(k[c], k[c + "_o"])).sum() for c in FREE)) + int((k._merge != "both").sum())  # differing cells, plus rows on one side only
    rows.append({"check": "orbit-free columns equal label rule 5's stave_base_265.csv.gz", "cells": int(len(k) * len(FREE)), "differing": n_old, "pass": n_old == 0})  # (iii)
    chk = pd.DataFrame(rows)  # the checks
    chk.to_csv(OUT / "stave_base_b_checks.csv", index=False, lineterminator="\n")  # written
    print(chk.to_string(index=False))  # printed
    if not chk["pass"].all():  # a precondition failed
        sys.exit("STOP: the extended base does not reproduce the committed base or label rule 5's file; nothing written")  # stop
    stop = stb[stb.sec <= T(STOP_END)]  # the Stave stop's rows
    print(f"Stave stop {STOP_START}-{STOP_END}: {len(stop)} rows; geometry cells filled {int(stop[GEOM].notna().sum().sum())} of {stop[GEOM].size}")  # summary
    with open(OUT / "stave_base_265_final.csv.gz", "wb") as fh, gzip.GzipFile(filename="", mode="wb", fileobj=fh, mtime=0) as gz:  # with-block: gzip without name or time
        gz.write(stb[F.KEYS + cols].to_csv(index=False, lineterminator="\n").encode())  # the extended 2.6.5 base, orbit-free and geometry


if __name__ == "__main__":  # run as a script
    main()  # build, check, write
