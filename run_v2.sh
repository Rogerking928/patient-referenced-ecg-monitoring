#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=14
export HEMO_CFG=configs/v2.yaml
echo "build $(date +%T)"
python3 build_data.py > out_v2/build.stdout 2>&1 || { echo BUILD_FAILED; exit 1; }
echo "leakage baselines $(date +%T)"
python3 run_leakage_and_matching.py > out_v2/leakage.stdout 2>&1 || echo LEAK_FAILED
echo "main $(date +%T)"
python3 run_main.py > out_v2/main.stdout 2>&1 || echo MAIN_FAILED
echo "tte $(date +%T)"
python3 run_tte.py > out_v2/tte.stdout 2>&1 || echo TTE_FAILED
echo "matched hr $(date +%T)"
python3 run_matched_hr_main.py > out_v2/matchedhr.stdout 2>&1 || echo MHR_FAILED
echo "leakage baselines rerun $(date +%T)"
python3 run_leakage_and_matching.py > out_v2/leakage2.stdout 2>&1 || echo LEAK2_FAILED
echo "V2_COMPLETE $(date +%T)"
