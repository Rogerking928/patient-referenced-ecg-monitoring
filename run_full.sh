#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=14
rm -f out/X_ratenorm.npy
echo "build $(date +%T)"
python3 build_data.py    > out/build_data.stdout   2>&1 || { echo BUILD_FAILED; exit 1; }
echo "main $(date +%T)"
python3 run_main.py      > out/run_main.stdout     2>&1 || echo MAIN_FAILED
echo "tte $(date +%T)"
python3 run_tte.py       > out/run_tte.stdout      2>&1 || echo TTE_FAILED
echo "controls $(date +%T)"
python3 run_controls.py  > out/run_controls.stdout 2>&1 || echo CONTROLS_FAILED
echo "interpret $(date +%T)"
python3 run_interpret.py > out/run_interpret.stdout 2>&1 || echo INTERPRET_FAILED
echo "ptbxl $(date +%T)"
python3 run_ptbxl.py     > out/run_ptbxl.stdout    2>&1 || echo PTBXL_FAILED
echo "ALL_STAGES_COMPLETE $(date +%T)"
