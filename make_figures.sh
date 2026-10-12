#!/bin/bash
# Redraw every figure from the result tables.
cd "$(dirname "$0")"
set -e
mkdir -p figures
python3 make_figures_model.py
python3 make_figures_results.py
python3 make_figures_main.py
