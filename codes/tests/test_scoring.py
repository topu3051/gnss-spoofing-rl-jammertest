#!/usr/bin/env python3
# module docstring below: a string placed first in a file, which Python keeps as the file's help text
"""
Unit tests for codes/common/scoring.py on hand-made toy sequences. Every expected value is worked out by hand in the comment
beside it. Run:  python3 codes/tests/test_scoring.py   (standard-library unittest; also runs under pytest).

Toy fold used by most tests (one character per second, seconds 1000 to 1029):
    index   0         1         2
            012345678901234567890123456789
    label   cccccccccuuuuuAAAAAAAAuuuuxccc   (c clean, u uncertain, A attack_spoof, x excluded)
    i.e. clean 0-8 (9 s), uncertain 9-13 (5 s), attack 14-21 (8 s), uncertain 22-25 (4 s), excluded 26, clean 27-29 (3 s): 30 s
"""

import sys  # system access: sys.path and sys.modules
import unittest  # the standard-library test framework
import importlib.util  # loads a .py file whose name starts with a digit
from pathlib import Path  # Path objects

import numpy as np  # NumPy arrays

ROOT = Path(__file__).resolve().parent.parent.parent  # the script's folder sits in codes/, so two parents up is the folder that holds codes/ and Data/
spec = importlib.util.spec_from_file_location("scoring", ROOT / "codes" / "common" / "scoring.py")  # where the module under test is
S = importlib.util.module_from_spec(spec)  # an empty module object
sys.modules["scoring"] = S  # registered
spec.loader.exec_module(S)  # loaded (nothing runs on import)

CODE = {"c": "clean", "u": "uncertain", "A": "attack_spoof", "J": "attack_jam_spoof", "x": "excluded"}  # dict: one letter -> label


def fold(pattern: str, start: int = 1000) -> tuple:  # function: a toy fold from a letter pattern: seconds and labels
    lab = np.array([CODE[ch] for ch in pattern], dtype=object)  # list comprehension: one label per letter
    return np.arange(start, start + len(pattern), dtype=np.int64), lab  # consecutive seconds and the labels


def alarms(pattern: str) -> np.ndarray:  # function: a toy alarm sequence from "1" and "0" characters
    return np.array([ch == "1" for ch in pattern], dtype=bool)  # list comprehension: True where "1"


LAB = "cccccccccuuuuuAAAAAAAAuuuuxccc"  # 30 s: clean 0-8, uncertain 9-13, attack 14-21 (onset 1014, end 1021), uncertain 22-25, excluded 26, clean 27-29
ONSET, END = 1014, 1021  # the pooled interval of the toy fold


class TestScoring(unittest.TestCase):  # class: a group of tests (unittest runs every method whose name starts with test)

    def test_never_alarm(self):  # never raising an alarm
        sec, lab = fold(LAB)  # the toy fold
        al = np.zeros(len(lab), bool)  # no alarm at all
        pc = S.per_class(lab, al)  # scores
        self.assertEqual((pc["tp"], pc["fp"], pc["fn"], pc["tn"]), (0, 0, 8, 12))  # 8 attack seconds missed, 12 clean seconds correct
        self.assertTrue(np.isnan(pc["attack_precision"]))  # 0 / 0: undefined, never shown as 0 or 1
        self.assertEqual(pc["attack_recall"], 0.0)  # 0 of 8
        self.assertEqual(pc["clean_recall"], 1.0)  # 12 of 12
        lt = S.latency(sec, lab, al, ONSET, END)  # latency
        self.assertFalse(lt["detected"])  # a miss
        self.assertIsNone(lt["latency_s"])  # no latency
        fa = S.false_alarms(sec, lab, al)  # false alarms
        self.assertEqual(fa["episodes"], 0)  # none
        self.assertAlmostEqual(fa["upper95_count"], 2.9957, places=4)  # the rule of three: -ln(0.05)

    def test_always_alarm(self):  # alarm in every second
        sec, lab = fold(LAB)  # the toy fold
        al = np.ones(len(lab), bool)  # always
        pc = S.per_class(lab, al)  # scores
        self.assertEqual((pc["tp"], pc["fp"], pc["fn"], pc["tn"]), (8, 12, 0, 0))  # every attack caught, every clean second a false positive
        self.assertAlmostEqual(pc["attack_precision"], 8 / 20)  # 8 of 20 scored alarms
        self.assertEqual(pc["alarms_uncertain"], 9)  # 5 + 4 uncertain seconds alarmed, neither true nor false
        self.assertEqual(pc["alarms_excluded"], 1)  # the excluded second
        lt = S.latency(sec, lab, al, ONSET, END)  # latency
        self.assertTrue(lt["detected"])  # detected
        self.assertEqual(lt["latency_s"], 1000 - 1014)  # the run is unbroken from the first second: -14 s
        self.assertEqual(lt["early_alarms"], [])  # no separate early alarm
        self.assertEqual(S.false_alarms(sec, lab, al)["episodes"], 1)  # one run touching clean seconds: one episode

    def test_run_from_uncertain_before_onset(self):  # an alarm run that starts in uncertain seconds and runs unbroken into the attack
        sec, lab = fold(LAB)  # the toy fold
        al = alarms("000000000001111111110000000000")  # alarm at 11-19: starts in uncertain at 11, reaches the onset at 14 unbroken
        lt = S.latency(sec, lab, al, ONSET, END)  # latency
        self.assertEqual(lt["latency_s"], 1011 - 1014)  # -3 s: counted, because the run is unbroken through the onset
        pc = S.per_class(lab, al)  # scores
        self.assertEqual((pc["tp"], pc["fp"]), (6, 0))  # attack 14-19 caught; the uncertain alarms 11-13 are not false
        self.assertEqual(pc["alarms_uncertain"], 3)  # counted separately
        self.assertEqual(S.false_alarms(sec, lab, al)["episodes"], 0)  # no clean second touched

    def test_isolated_early_alarm(self):  # an early alarm that stops before the onset is never a detection
        sec, lab = fold(LAB)  # the toy fold
        al = alarms("000000000011000001110000000000")  # early alarm 10-11 (uncertain), gap, then 17-19
        lt = S.latency(sec, lab, al, ONSET, END)  # latency
        self.assertEqual(lt["latency_s"], 1017 - 1014)  # +3 s from the first alarm at or after the onset
        self.assertEqual(lt["early_alarms"], [(1010, 1011)])  # the early run, reported separately
        self.assertEqual(S.confusion(lab, al)["fp"], 0)  # it fell in uncertain seconds: not a false alarm

    def test_early_alarm_one_second_gap(self):  # a run broken one second before the onset does not earn a negative latency
        sec, lab = fold(LAB)  # the toy fold
        al = alarms("000000000001101111000000000000")  # 11-12, gap at 13, then 14-17
        lt = S.latency(sec, lab, al, ONSET, END)  # latency
        self.assertEqual(lt["latency_s"], 0)  # the detection is the run starting at the onset
        self.assertEqual(lt["early_alarms"], [(1011, 1012)])  # the broken run is an early alarm

    def test_lead_not_through_previous_interval(self):  # one alarm run covering two attack intervals earns no lead through the first
        sec, lab = fold("ccuuAAAuuuAAAuu")  # attack 4-6 (1004-1006) and 10-12 (1010-1012)
        al = np.ones(15, bool)  # always alarm
        lt = S.latency(sec, lab, al, 1010, 1012, prev_end=1006)  # the second interval, after the first ended at 1006
        self.assertEqual(lt["latency_s"], 1007 - 1010)  # -3 s: counted from the second after the previous interval, not from 1000

    def test_always_alarm_lead_shown_with_full_preonset_share(self):  # an always-alarm lead appears beside a 100 % pre-onset alarm share (toy fold shaped like 2.6.5)
        # this test used to read the committed 2.6.5 labels (u-blox 01: leads -519/-658 s, 62/658 uncertain seconds), so any
        # relabel failed it although the scoring module was unchanged. The same situation is now built by hand: excluded seconds first (as the 2.6.5 drive exclusion), then two intervals.
        sec, lab = fold("xxxuuAAAuuuAA")  # 1000-1002 excluded, 1003-1004 uncertain, 1005-1007 attack, 1008-1010 uncertain, 1011-1012 attack
        iv = [(1005, 1007), (1011, 1012)]  # list of tuples: the two pooled intervals
        al = np.ones(len(sec), bool)  # always alarm
        with self.assertRaises(ValueError):  # context manager: 2.6.5 has no false-alarm rate, so an undeclared threshold is refused
            S.evaluate(sec, lab, al, iv, test="2.6.5")  # no declaration
        e = S.evaluate(sec, lab, al, iv, test="2.6.5", threshold_from_other_folds=True)  # declared as set on other folds
        self.assertEqual([x["latency_s"] for x in e["latency"]], [-5, -3])  # leads: the alarm run from 1000 to the onset 1005; from 1008 (after the first interval) to 1011
        self.assertEqual([x["pre_onset_alarm_share"] for x in e["latency"]], [1.0, 1.0])  # beside each, 100 % of the pre-onset uncertain seconds in alarm
        self.assertEqual([x["pre_onset_uncertain_s"] for x in e["latency"]], [2, 3])  # 1003-1004 and 1008-1010 (the excluded seconds 1000-1002 are not counted)

    def test_rate_fold_needs_no_declaration(self):  # a fold with a readable rate may be evaluated without the declaration
        sec, lab = fold("c" * 300 + "u" * 5 + "A" * 5)  # 300 clean seconds, then uncertain, then attack
        e = S.evaluate(sec, lab, np.zeros(310, bool), [(1305, 1309)], test="2.6.2")  # no declaration needed
        self.assertEqual(e["latency"][0]["pre_onset_alarm_share"], 0.0)  # none of the 5 uncertain seconds alarmed

    def test_alarm_in_excluded_seconds(self):  # alarms in excluded seconds are neither true nor false and are no detection
        sec, lab = fold(LAB)  # the toy fold
        al = alarms("000000000000000000000000001000")  # alarm only at 26 (excluded)
        c = S.confusion(lab, al)  # cells
        self.assertEqual((c["tp"], c["fp"], c["alarms_excluded"]), (0, 0, 1))  # not scored, counted
        self.assertFalse(S.latency(sec, lab, al, ONSET, END)["detected"])  # after the interval: a miss
        self.assertEqual(S.false_alarms(sec, lab, al)["episodes"], 0)  # no clean second touched

    def test_hand_confusion(self):  # a mixed sequence checked cell by cell
        sec, lab = fold(LAB)  # the toy fold
        al = alarms("110000000000000011110000000011")  # clean 0-1 alarm, attack 16-19 alarm, clean 28-29 alarm
        pc = S.per_class(lab, al)  # scores
        self.assertEqual((pc["tp"], pc["fn"], pc["fp"], pc["tn"]), (4, 4, 4, 8))  # attack 4 of 8; clean alarms 0,1,28,29; clean quiet 8
        self.assertAlmostEqual(pc["attack_precision"], 0.5)  # 4 / 8
        self.assertAlmostEqual(pc["attack_f1"], 0.5)  # precision 0.5, recall 0.5
        self.assertAlmostEqual(pc["clean_precision"], 8 / 12)  # 8 quiet clean / (8 + 4 quiet attack)
        self.assertEqual(S.false_alarms(sec, lab, al)["episodes"], 2)  # runs 0-1 and 28-29

    def test_episode_across_uncertain_counts_once(self):  # a run crossing uncertain seconds between clean ones is one episode
        sec, lab = fold("cccuuuccc")  # clean, uncertain, clean
        self.assertEqual(len(S.false_alarm_episodes(sec, lab, np.ones(9, bool))), 1)  # one run, one episode
        self.assertEqual(len(S.false_alarm_episodes(sec, lab, alarms("110001100"))), 2)  # two separate runs touching clean

    def test_poisson_bounds(self):  # exact one-sided 95 % upper limits (Garwood), against tabulated values
        self.assertAlmostEqual(S.poisson_upper(0), 2.9957, places=4)  # n = 0
        self.assertAlmostEqual(S.poisson_upper(1), 4.7439, places=4)  # n = 1
        self.assertAlmostEqual(S.poisson_upper(2), 6.2958, places=4)  # n = 2

    def test_rate_only_with_300_clean_seconds(self):  # count only below 300 clean seconds, no figure at zero, rate from 300
        sec, lab = fold("c" * 26)  # 26 clean seconds, as u-blox 01 in 2.6.10
        fa = S.false_alarms(sec, lab, np.zeros(26, bool))  # no alarm
        self.assertEqual((fa["readable"], fa["episodes"], fa["rate_per_h"]), ("count only", 0, None))  # count only, no rate
        self.assertIn("no rate can be read", fa["note"])  # said plainly
        sec, lab = fold("u" * 10)  # no clean second
        self.assertEqual(S.false_alarms(sec, lab, np.zeros(10, bool))["readable"], "none")  # no figure
        sec, lab = fold("c" * 300)  # exactly 300 clean seconds
        fa = S.false_alarms(sec, lab, alarms("1" + "0" * 299))  # one episode
        self.assertEqual(fa["readable"], "rate")  # a rate
        self.assertAlmostEqual(fa["rate_per_h"], 12.0)  # 1 episode in 300 s = 12 per hour

    def test_fold_policy(self):  # the decided policy per fold
        self.assertEqual(S.fold_policy("2.6.11", 0)["false_alarms"], "none")  # nothing for 2.6.11
        self.assertFalse(S.fold_policy("2.6.11", 0)["precision"])  # and no precision
        self.assertEqual(S.fold_policy("2.6.5", 0)["false_alarms"], "none")  # 2.6.5: no clean seconds, no figure
        self.assertEqual(S.fold_policy("2.6.10", 0)["false_alarms"], "none")  # 2.6.10 Septentrios: none
        self.assertEqual(S.fold_policy("2.6.10", 26)["false_alarms"], "count only")  # 2.6.10 u-blox 01: count and bound
        self.assertFalse(S.fold_policy("2.6.10", 26)["precision"])  # no precision there
        self.assertEqual(S.fold_policy("2.6.10", 1000)["false_alarms"], "count only")  # stays a count by decision even with many seconds
        self.assertTrue(S.fold_policy("2.6.2", 440)["precision"])  # precision for 2.6.2
        self.assertEqual(S.fold_policy("2.6.2", 440)["false_alarms"], "rate")  # and a rate with 440 s
        self.assertEqual(S.fold_policy("2.6.4", 125)["false_alarms"], "count only")  # 125 s: count only
        self.assertIn("Stave clean margin limited", S.fold_policy("2.6.12", 388)["note"])  # the Stave limitation named

    def test_persistence(self):  # an alarm with persistence k is raised at the k-th second above threshold
        sec = np.arange(10)  # ten seconds
        sc = np.array([0, 5, 5, 5, 0, 5, 5, 0, 5, 5], float)  # runs above 1 at 1-3, 5-6, 8-9
        self.assertEqual(S.alarms_from_scores(sec, sc, 1.0, 3).tolist(), [False, False, False, True] + [False] * 6)  # only the 3 s run, from its 3rd second

    def test_threshold_uses_training_folds_only(self):  # the held-out fold never changes the threshold
        a = (np.arange(3600), np.array(["clean"] * 3600, dtype=object), np.r_[np.zeros(3598), 9.0, 7.0])  # one hour of clean, two high scores at the end
        b = (np.arange(10), np.array(["clean"] * 10, dtype=object), np.zeros(10))  # a held-out fold
        t1 = S.threshold_for_rate({"a": a, "b": b}, ["a"], target_per_h=0.0)  # zero false alarms allowed
        # alarms are now raised at score >= threshold, so the threshold is the next double above 9 (np.nextafter): there nothing
        # alarms; the next candidate down, just above 7, alarms on the 9 at 3598, so it is refused
        self.assertEqual(t1, np.nextafter(9.0, np.inf))
        b2 = (b[0], b[1], np.full(10, 100.0))  # the held-out fold changed completely
        self.assertEqual(S.threshold_for_rate({"a": a, "b": b2}, ["a"], target_per_h=0.0), t1)  # same threshold
        with self.assertRaises(ValueError):  # context manager: the block must raise ValueError
            S.threshold_for_rate({"a": a}, [], 1.0)  # no training fold

    def test_mask_keeps_series(self):  # uncertain seconds are masked from scoring but the series keeps its length
        sec, lab = fold(LAB)  # the toy fold
        m = S.scoring_mask(lab)  # the mask
        self.assertEqual(len(m), len(lab))  # same length: nothing dropped from the time series
        self.assertEqual(int(m.sum()), 20)  # 12 clean + 8 attack seconds scored

    def test_length_mismatch_raises(self):  # misaligned inputs stop the scoring
        with self.assertRaises(ValueError):  # context manager: must raise
            S.confusion(["clean", "clean"], [True])  # two labels, one alarm

    # F1 = 2TP / (2TP + FP + FN); 0 when the class has instances and no true positive, NaN only without instances
    def test_f1_no_alarm_with_attack(self):  # a fold with attack seconds and no alarm at all
        sec, lab = fold(LAB)  # the toy fold: 8 attack, 12 clean seconds
        pc = S.per_class(lab, np.zeros(len(lab), bool))  # no alarm
        self.assertTrue(np.isnan(pc["attack_precision"]))  # precision 0 / 0 stays undefined and is reported as such
        self.assertEqual(pc["attack_f1"], 0.0)  # 2*0 / (0 + 0 + 8) = 0: the class has instances and no true positive
        self.assertAlmostEqual(pc["clean_f1"], 0.75)  # clean: tn 12, fn 8, fp 0: 2*12 / (24 + 8 + 0) = 0.75

    def test_f1_wrong_alarms_only(self):  # alarms only on clean seconds: no true positive, false positives present
        sec, lab = fold(LAB)  # the toy fold
        pc = S.per_class(lab, alarms("110000000000000000000000000011"))  # alarms at clean seconds 0, 1, 28, 29 only
        self.assertEqual(pc["attack_precision"], 0.0)  # 0 / 4: defined and zero
        self.assertEqual(pc["attack_f1"], 0.0)  # 2*0 / (0 + 4 + 8) = 0 (the old rule gave NaN because precision + recall = 0)

    def test_f1_no_attack_instances(self):  # a fold without any attack second
        sec, lab = fold("cccccuuuccc")  # clean and uncertain only
        pc = S.per_class(lab, alarms("10000000000"))  # one alarm on a clean second
        self.assertTrue(np.isnan(pc["attack_recall"]))  # recall 0 / 0: undefined
        self.assertTrue(np.isnan(pc["attack_f1"]))  # TP + FN = 0: the class has no instance, so F1 is undefined
        self.assertAlmostEqual(pc["clean_f1"], 2 * 7 / (14 + 0 + 1))  # clean: tn 7, fn 0, fp 1: 14 / 15

    def test_f1_clean_class_mirror(self):  # the clean class by the same rule
        sec, lab = fold("AAAAuuu")  # attack and uncertain only: no clean second
        pc = S.per_class(lab, alarms("1100000"))  # two of four attack seconds in alarm
        self.assertTrue(np.isnan(pc["clean_f1"]))  # tn + fp = 0: the clean class has no instance, F1 undefined
        self.assertAlmostEqual(pc["attack_f1"], 2 * 2 / (4 + 0 + 2))  # attack: tp 2, fp 0, fn 2: 4 / 6
        sec2, lab2 = fold("ccccAAAA")  # clean and attack
        pc2 = S.per_class(lab2, np.ones(8, bool))  # every second in alarm: no clean second without alarm (tn = 0)
        self.assertEqual(pc2["clean_f1"], 0.0)  # clean instances present, tn = 0: 2*0 / (0 + 0 + 4) = 0
        self.assertAlmostEqual(pc2["attack_f1"], 2 * 4 / (8 + 4 + 0))  # attack: tp 4, fp 4, fn 0: 8 / 12

    def test_f1_equals_count_form_when_defined(self):  # with TP > 0 the harmonic mean equals the count form
        sec, lab = fold(LAB)  # the toy fold
        pc = S.per_class(lab, alarms("110000000000000011110000000011"))  # tp 4, fn 4, fp 4 (test_hand_confusion's sequence)
        self.assertAlmostEqual(pc["attack_f1"], 2 * 4 / (8 + 4 + 4))  # 8 / 16 = 0.5, the same as the harmonic mean of 0.5 and 0.5


if __name__ == "__main__":  # if: true only when the file is run directly
    unittest.main(verbosity=2)  # runs every test and prints each name and result
