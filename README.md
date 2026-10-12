# Patient-referenced and trajectory features from serial ECGs

Code and result tables for

> Wang Y-H. *Patient-Referenced and Trajectory Features from Serial ECGs for Haemodynamic
> Monitoring: An In Silico Study with External Feasibility Analysis.* Physiological
> Measurement (under review, manuscript PMEA-107380).

The study asks, in simulation, whether serial electrocardiograms read against a patient's
own earlier recording carry information about haemodynamic state, and whether the recent
trajectory of that change adds to its current value, using acute haemorrhage as the test
case; it then examines the result in an external feasibility cohort of intraoperative
recordings (VitalDB). Nothing here is a clinical tool.

## Layout

```
configs/default.yaml       main world: every parameter of the generator, features and models
configs/v2.yaml            second world: random observation window, onset may precede it
configs/xgen.yaml          cross-generator arm
hemo/                      state-space model, observation model, features, models, metrics
build_data.py              simulate the world, render windows, extract features
run_*.py                   the analyses of the original submission
rev1_*.py                  analyses added in revision (alternative observation models,
                           unknown onset, calibration and alarms, QT correction and leads,
                           fresh test cohort, anaesthetic confounding in VitalDB)
make_figures_*.py          figures, redrawn from the result tables
vitaldb_pull.py            retrieval of the VitalDB windows
results/                   every result table behind the manuscript and its supplement
```

## Data

* **PTB-XL** 1.0.3 (Wagner et al 2020), https://physionet.org/content/ptb-xl/ — set
  `PTBXL_DIR` or place it under `data/PTB-XL`.
* **VitalDB** (Lee et al 2022), https://vitaldb.net, CC BY 4.0 — set `VITALDB_DIR` or place
  `cases.csv`, `trks.csv` and `vital/<caseid>.vital` under `data/VitalDB`. The case
  identifiers used are in `results/out_vitaldb/selected_cases.csv` and the exclusion cascade
  in `results/out/vitaldb_cohort_cascade.csv`.

No patient data are included in this repository. Synthetic subjects, their parameters and
their train/validation/test assignment are regenerated exactly from the seeds in the
configuration files; the assignments are also listed in `results/*/subjects_split.csv`.

## Running

Python 3.12; package versions in `requirements.txt`. No GPU is needed.

```bash
./run_full.sh                                  # main world: build, models, controls (about 4 h, 14 threads)
HEMO_CFG=configs/v2.yaml ./run_v2.sh           # second world
python3 run_leakage_and_matching.py; python3 run_incremental.py; python3 run_longitudinal.py
python3 rev1_alt_observation.py all            # alternative observation models + Latin hypercube
python3 rev1_v2_reference.py                   # references without knowledge of onset
python3 rev1_calibration_alarm.py              # calibration, alarm burden, decision curve
python3 rev1_qtc_leads_fresh.py                # QT correction, lead configuration, fresh cohort
python3 rev1_vitaldb_confounders.py            # anaesthetic and ventilatory confounding
./make_figures.sh                               # figures
```

## Licence

MIT (code). The result tables are released under CC BY 4.0. Please cite the article.

The code was written by the author, with Claude (Anthropic) used as an assistant under the
author's direction; the author reviewed and tested all code and takes full responsibility
for the results it produces.
