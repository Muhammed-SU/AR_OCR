#!/usr/bin/env bash
# Runs the entire pipeline.
# Assumes you have activated a Python environment with the deps installed.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"

echo "==> 01 build corpus"
python 01_build_corpus.py

echo "==> 02 compute metrics"
python 02_compute_metrics.py

echo "==> 03 make figures"
python 03_make_figures.py

echo "==> 04 error typology"
python 04_error_typology.py

echo "Done. See ../results/"
