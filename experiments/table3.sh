#!/usr/bin/env bash
# Reproduce Table 3 — inverse design on the 500 hardest PCA-split targets
# (NN retrieval vs surrogate-filtered diffusion, both CST-validated).
set -euo pipefail
python -m scripts.run_inverse --config_file configs/inverse/diffusion.yaml --hardest_n 500
python -m scripts.run_inverse --config_file configs/inverse/nearest_neighbor.yaml
echo "Validate the exported candidates in CST, then run:"
echo "  python -m scripts.evaluate_cst --cst_output_dirs <dir> --hardest_n 500"
