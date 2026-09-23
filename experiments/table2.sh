#!/usr/bin/env bash
# Reproduce Table 2 — inverse design on the 100 hardest PCA-split targets.
#
# Prerequisites: simulated FMNIST+CIFAR corpus (data/README.md), the
# gps_pais_big surrogate checkpoint, and a trained diffusion checkpoint
# (bash experiments passing configs/inverse/diffusion_train.yaml).
#
# Each step exports candidate geometries (STL + metadata) under output_dir;
# validate them in CST, then aggregate with scripts/evaluate_cst.py.
set -euo pipefail

# Diff. + surr. (500 candidates/target, surrogate top-5 -> CST)
python -m scripts.run_inverse --config_file configs/inverse/diffusion.yaml
# Diff. - 5 rand. (5 unfiltered samples/target)
python -m scripts.run_inverse --config_file configs/inverse/diffusion_random5.yaml
# NN retrieval
python -m scripts.run_inverse --config_file configs/inverse/nearest_neighbor.yaml
# CMA-ES
python -m scripts.run_inverse --config_file configs/inverse/cma_baseline.yaml
# GA, SA, NN-seeded SA (500-evaluation budget each)
python -m scripts.run_inverse --config_file configs/inverse/baselines.yaml

echo "Validate the exported candidates in CST, then run:"
echo "  python -m scripts.evaluate_cst --cst_output_dirs <dir-with-CST-results>"
