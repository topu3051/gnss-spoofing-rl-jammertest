# Reinforcement learning for GNSS spoofing detection from receiver observation data

Code of the MSc thesis of Jamiual Amin Topu, University of Vaasa, 2026; supervisor Dr. Mahmoud Elsanhoury.

## What the code does

The code is the analysis from the raw receiver recordings of the Jammertest 2026 event to every number, table and figure of the thesis. It builds the ground truth (schedule, capture times, handsets, receiver status and gain), converts the receiver recordings to RINEX, and derives labels per receiver-second and window features from the observations. It fits two baselines, a logistic regression and a random forest, together with the sequential detectors CUSUM and k-of-n. It then trains a deep Q-network agent on the Shiryaev cost. It tests how the models transfer to the other receivers and to an unseen attack, and it measures detection delay and cost. A fixed seed, 20260914, is used throughout.

## Repository layout

`codes/` holds one folder per part of the work, with the scripts numbered in the order they run. Every line of code carries a comment that says what the line is and what it does for the analysis; each script's opening text says what it reads and writes.

```
codes/
  common/                    helpers.py (functions shared by many scripts) and scoring.py (the scoring rules every
                             evaluation uses: folds, operating points, recall, precision, F1, the false-alarm bounds)
  1_ground_truth/            the schedule, captures and UTC time base; the handsets; the receivers' own gain and status;
                             gain drops and own-position evidence; decision times; the jamming phases; the 2.6.5 gain
                             pattern; the Test Area 2 effects; the end of spoofed tracking in 2.6.10; the phases of the
                             unseen attack 2.6.12
  2_preprocessing/           RINEX conversion and its verification; vehicle motion; reference positions; GLONASS orbit
                             propagation and its broadcast and precise-orbit checks; the G30 and G08 diagnostics
  3_labels/                  the labels per receiver-second, the quiet negative sets, the training labels;
                             01_labels_as_used_by_quiet_negatives.py is the label script as it stood when the quiet
                             negative sets were made (02_quiet_negatives.py loads it; it runs nothing by itself)
  4_features/                per-satellite observables from RINEX, the residuals, the window features, the Stave stop
                             base, the jamming screen, and the observable effects of RQ1
  5_baselines/               the logistic regression, the random forest, the sequential baselines (k of n, CUSUM,
                             variants), the pipeline library, the library for the final labels, the run of the adopted
                             baselines on the final labels (Table 10), and the adopted pipeline the environment uses
  6_reinforcement_learning/  the environment, agent training, agent evaluation (Table 11), the diagnosis (Figure 8) and
                             its exploratory follow-up (Table 12)
  7_evaluation/              transfer across receivers and to the unseen attack 2.6.12, delay, monitors and cost (RQ4),
                             the satellite-count diagnosis, the margins of the agent's verdicts, and the 2.6.12 point
                             runs with their onset marks (05_unseen_attack_point_marks.py)
  8_data_checks/             the checks made after the supervisor's review of 19 October 2026: provenance and vehicle
                             motion, the start positions of every receiver, the dependence of each receiver's labels on
                             the other receivers' evidence, the per-receiver label sensitivity, and every receiver at its
                             own reference position (its products, its RQ4 tables and the residual shift it causes)
  figures/                   the scripts of Figures 1 to 9 (figure_7_own_receiver_positions.py draws Figure 7 as printed;
                             figures_6_to_9.py also draws Figure 7 at the site positions, as figure_7_site_positions.png),
                             and source_constants.py, which reads constants out of the scripts so a figure states the
                             values the analysis used
  tests/                     the automated tests of the features, both baselines, the environment, the agents, transfer,
                             RQ4, the diagnosis and the scoring rules
tools/convbin_build.txt      the build record of RTKLIB's convbin (see "RTKLIB convbin")
manifest/RAW_MANIFEST.csv    path, size and SHA-256 of every raw recording (no data; see "Data")
environment.yml              conda environment
requirements.txt             pip requirements
```

## Requirements

Python 3.13.9. Create the environment with conda:

```bash
conda env create -f environment.yml
```

or install the packages with pip:

```bash
pip install -r requirements.txt
```

The packages and versions, exactly as in `requirements.txt`:

- gymnasium==1.3.0
- hatanaka==2.8.1
- matplotlib==3.10.6
- numpy==2.2.6
- pandas==2.3.3
- pypdf==6.17.0
- scipy==1.16.3
- scikit-learn==1.7.2
- stable-baselines3==2.9.0
- torch==2.14.0

## RTKLIB convbin

The converter binary is not included. `codes/2_preprocessing/01_convert_rinex.py` runs RTKLIB's convbin (demo5 branch) from `tools/convbin`. Build it from the record in `tools/convbin_build.txt` (repository `https://github.com/rtklibexplorer/RTKLIB.git`, branch demo5, commit `75a2e56275485b21a67bd35bc94bbeb8936e1a74`, version string `CONVBIN demo5 b34L`) and place it at `tools/convbin`:

```bash
git clone https://github.com/rtklibexplorer/RTKLIB.git && cd RTKLIB && git checkout 75a2e56275485b21a67bd35bc94bbeb8936e1a74
cd app/consapp/convbin/gcc && make
```

The second line is the Linux build; on Windows the same makefile is used under MSYS2 with gcc. The macOS build used `-D_DARWIN_C_SOURCE`; the full build command is recorded in `tools/convbin_build.txt`. Then copy the program to `tools/convbin` and write its SHA-256 into the line `sha256=` of `tools/convbin_build.txt`: a binary built elsewhere has another checksum, and the script refuses a binary its record does not describe.

## Data

The Jammertest 2026 recordings (test group 2.6, Andøya, 14 September 2026) are not included. They are research data of the University of Vaasa research group and are available on request from the supervisor.

The scripts expect this layout, with all commands started from the folder that holds it:

```
<working folder>/
  codes/        from this repository
  tools/        from this repository, with the convbin binary you build
  Data/         the recordings, with the raw files under Data/Day1_14092026/
  Graph/        an empty folder; the scripts save their figures here
```

`manifest/RAW_MANIFEST.csv` lists every raw recording with its path (relative to the folder above), size in bytes and SHA-256, so that a copy of the recordings can be checked file by file.

## How to run

Set up once:

1. Put `codes/` and `tools/` of this repository and the `Data/` folder side by side in one empty folder, and create an empty folder `Graph/` beside them (the scripts save their figures there). Every command below is started from that folder.
2. Python 3.13.9 with the packages of `environment.yml` (conda: `conda env create -f environment.yml`) or `requirements.txt` (pip: `pip install -r requirements.txt`).
3. The shared helpers on the import path, for example:

```bash
PYTHONPATH=codes/common python3 codes/figures/figures_6_to_9.py
```

Each script's opening text says what it reads and writes.

The scripts in the order they ran, with the runtime measured when every script was run from a fresh copy on a laptop (Apple M4, 10 cores, 16 GB; the agent training jobs ran beside the other scripts, three worker processes each):

```
1_ground_truth/
  codes/1_ground_truth/01_timeline.py                                          under a minute  schedule, captures, UTC time base
  codes/1_ground_truth/02_handset_check.py                                     under a minute  handset time base and distances
  codes/1_ground_truth/03_receiver_status.py                                   under a minute  receivers' gain and status
  codes/1_ground_truth/04_gain_and_position_evidence.py                       about 4 minutes  gain drops and own-position evidence
  codes/1_ground_truth/05_decision_times.py                                    under a minute  decision times
  codes/1_ground_truth/06_jamming_phases.py                                    under a minute  jamming phases
  codes/1_ground_truth/07_gain_pattern_265.py                                  under a minute  2.6.5 gain pattern
  codes/1_ground_truth/08_test_area_2_effects.py                               under a minute  Test Area 2 effects
  codes/1_ground_truth/09_spoofed_tracking_end_2610.py                         under a minute  end of spoofed tracking in 2.6.10
  codes/1_ground_truth/10_unseen_attack_phases.py                              under a minute  phases of the unseen attack 2.6.12
2_preprocessing/
  codes/2_preprocessing/01_convert_rinex.py                                   about 5 minutes  RINEX conversion with tools/convbin and its verification
  codes/2_preprocessing/02_vehicle_motion.py                                   under a minute  vehicle motion
  codes/2_preprocessing/03_reference_positions.py                              under a minute  reference positions
  codes/2_preprocessing/04_glonass_orbit.py                                    under a minute  GLONASS orbit propagation
  codes/2_preprocessing/05_glonass_orbit_check.py                             about 2 minutes  broadcast GLONASS check
  codes/2_preprocessing/06_glonass_precise_check.py                           about 2 minutes  precise-orbit check
  codes/2_preprocessing/07_satellite_g30_g08_diagnostics.py                    about a minute  G30 and G08 diagnostics
3_labels/
  codes/3_labels/01_labels.py                                                  under a minute  labels, registered set
  codes/3_labels/01_labels.py --labels final                                   under a minute  labels, final set
  codes/common/scoring.py                                                      under a minute  fold policy, registered labels
  codes/common/scoring.py --labels final                                       under a minute  fold policy, final labels
  codes/3_labels/02_quiet_negatives.py                                         under a minute  quiet negative sets
  codes/3_labels/03_training_labels.py                                         under a minute  training labels
4_features/
  codes/4_features/01_observables.py                                          about 5 minutes  per-satellite observables
  codes/4_features/02_residuals.py                                            about 6 minutes  residuals
  codes/4_features/03_window_features.py                                      about 3 minutes  registered windows (features/w1 to w10)
  codes/4_features/03_window_features.py --labels final --precision 6sf       about 5 minutes  final windows (features_final/)
  codes/4_features/04_stave_stop_base.py                                       under a minute  Stave stop base
  codes/4_features/05_jamming_screen.py                                       about 2 minutes  jamming screen, registered windows
  codes/4_features/05_jamming_screen.py --labels final                         under a minute  jamming screen, final windows
  codes/4_features/06_observable_effects.py                                   about 2 minutes  observable effects (RQ1)
5_baselines/
  codes/5_baselines/01_logistic_regression.py                                about 40 minutes  logistic regression, registered labels
  codes/5_baselines/03_sequential_baselines.py                               about 28 minutes  sequential baselines and variants
  codes/5_baselines/02_random_forest.py                                      about 65 minutes  random forest, registered labels
  codes/5_baselines/06_run_baselines.py table                                  under a minute  the run table
  codes/5_baselines/06_run_baselines.py gates                                about 13 minutes  gates on the registered material
  codes/5_baselines/06_run_baselines.py p                                     about 9 minutes  adopted baselines on the final labels (Table 10)
  codes/5_baselines/06_run_baselines.py controls                             about 26 minutes  controls
  codes/5_baselines/06_run_baselines.py c4b                                  about 44 minutes  within-session permutation control
  codes/5_baselines/06_run_baselines.py sens                                  about 2 minutes  sensitivity runs
  codes/5_baselines/06_run_baselines.py variants                             about 10 minutes  variants
  codes/5_baselines/06_run_baselines.py probe                                 about 2 minutes  probe
  codes/5_baselines/06_run_baselines.py report                                 under a minute  report tables
  codes/5_baselines/06_run_baselines.py no_quiet_negatives                   about 39 minutes  P without the quiet negatives
  codes/5_baselines/07_adopted_pipeline.py                                    about 4 minutes  adopted pipeline and its gates
6_reinforcement_learning/
  codes/6_reinforcement_learning/01_environment.py                            about 4 minutes  environment, states, episodes
  codes/6_reinforcement_learning/02_train_agents.py --grid                     about 17 hours  agent training (the full grid)
  codes/6_reinforcement_learning/03_evaluate_agents.py --points                under a minute  deployable points
  codes/6_reinforcement_learning/03_evaluate_agents.py --heldout               under a minute  held-out evaluation (Table 11)
  codes/6_reinforcement_learning/04_diagnosis.py --regen                        about 4 hours  the 1,200 inner jobs of the diagnosis, regenerated
  codes/6_reinforcement_learning/04_diagnosis.py --diag                        about a minute  diagnosis (Figure 8)
tests/
  codes/tests/test_diagnosis.py --scratch scratch_diagnosis_tests             about 4 minutes  diagnosis tests (before the follow-up draws its figures)
6_reinforcement_learning/
  codes/6_reinforcement_learning/05_followup.py grid A                         about 17 hours  follow-up run A, training
  codes/6_reinforcement_learning/05_followup.py grid B                         about 22 hours  follow-up run B, training
  codes/6_reinforcement_learning/05_followup.py recompute A                    under a minute  run A's held-out tables
  codes/6_reinforcement_learning/05_followup.py recompute B                    under a minute  run B's held-out tables
  codes/6_reinforcement_learning/05_followup.py diag A                         under a minute  run A's inner diagnosis
  codes/6_reinforcement_learning/05_followup.py diag B                         under a minute  run B's inner diagnosis
  codes/6_reinforcement_learning/05_followup.py report                         under a minute  follow-up tables (Table 12)
7_evaluation/
  codes/7_evaluation/01_transfer.py --transfer                                about 3 minutes  transfer across receivers (Part A)
  codes/7_evaluation/01_transfer.py --unseen                                    about 3 hours  Part B training and freezing
  codes/7_evaluation/01_transfer.py --heldout2612                              under a minute  Part B on 2.6.12 (sealed test)
  codes/7_evaluation/01_transfer.py --diag2612                                 under a minute  clean-alarm diagnosis on 2.6.12
  codes/7_evaluation/01_transfer.py --recompute2612                            under a minute  2.6.12 tables from the written decisions
  codes/7_evaluation/01_transfer.py --visibility                               under a minute  satellite visibility
  codes/7_evaluation/02_delay_monitors_cost.py                                about 5 minutes  delay, monitors and cost (RQ4)
  codes/7_evaluation/03_satellite_count_diagnosis.py                           under a minute  satellite-count diagnosis
  codes/7_evaluation/04_verdict_margins.py                                     under a minute  bootstrap margin, comparator sensitivity
  codes/7_evaluation/05_unseen_attack_point_marks.py                           under a minute  2.6.12 point runs with onset marks
8_data_checks/
  codes/8_data_checks/01_provenance_and_motion.py                              under a minute  provenance and motion
  codes/8_data_checks/02_start_positions.py                                    under a minute  start positions
  codes/8_data_checks/03_antenna_dependence.py                                 under a minute  antenna dependence
  codes/8_data_checks/04_label_sensitivity.py                                about 57 minutes  per-receiver label sensitivity
  codes/8_data_checks/05_own_receiver_positions.py                            about 6 minutes  each receiver's own position
  codes/8_data_checks/06_own_receiver_positions_rq4.py                        about 2 minutes  the same carried to RQ4
  codes/8_data_checks/07_position_residual_shift.py                            under a minute  residual shift and check figures
figures/
  codes/figures/figure_1_receiver_chain.py                                     under a minute  Figure 1
  codes/figures/figure_2_processing_chain.py                                   under a minute  Figure 2
  codes/figures/figures_3_to_5.py                                              under a minute  Figures 3 to 5
  codes/figures/figures_6_to_9.py                                              under a minute  Figures 6, 8, 9
  codes/figures/figure_7_own_receiver_positions.py                             under a minute  Figure 7 as printed
tests/
  codes/tests/test_scoring.py                                                  under a minute  scoring
  codes/tests/test_features.py                                                 about a minute  features
  codes/tests/test_logistic_regression.py                                     about 8 minutes  logistic regression
  codes/tests/test_random_forest.py                                          about 13 minutes  random forest
  codes/tests/test_environment.py                                             about 4 minutes  environment
  codes/tests/test_agents.py --pilot                                           under a minute  agents, timing pilot
  codes/tests/test_agents.py --part1                                          about 4 minutes  agents, training
  codes/tests/test_agents.py --t5amended                                       under a minute  agents, amended points
  codes/tests/test_agents.py --part2                                           under a minute  agents, evaluation
  codes/tests/test_transfer.py --part1                                        about 3 minutes  transfer, Part A
  codes/tests/test_transfer.py --part2                                         under a minute  transfer, Part B
  codes/tests/test_transfer.py --part2r                                        under a minute  transfer, recomputation and visibility
  codes/tests/test_delay_monitors_cost.py                                      under a minute  RQ4
  codes/tests/test_diagnosis.py --scratch scratch_diagnosis_tests --part2     about 7 minutes  follow-up gates
```

## Reproducibility

- A fixed seed, 20260914, is used throughout.
- The five agents are trained with the seeds 20260914 to 20260918 (thesis Table 9).
- Every fitted step uses the training sessions of its fold only (thesis Tables 7 and 9).
- Every table a script writes, every figure of the thesis and every trained model was reproduced byte for byte from a fresh copy of the thesis folder (`Thesis/README.txt` of the thesis folder). The exceptions it names are wall-clock timing tables, the start and end times in the sealed evaluations' read logs, two date lines in the RINEX headers, and the check images that carry the folder names in their titles; each is a property of the file and not of a result.

## Citation

Topu, J. A. (2026). Reinforcement learning for GNSS spoofing detection from receiver observation data [Master's thesis, University of Vaasa]. (link to be added after publication in OSUVA)

## License

No license has been chosen yet; all rights reserved.
