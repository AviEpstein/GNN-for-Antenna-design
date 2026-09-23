#!/usr/bin/env bash
# Fig. 3 — hand-designed targets (multi-lobe / broad-lobe), CFG w=2.
# Place one .pt far-field tensor per target in the manual_targets_dir
# configured in configs/base_inverse.yaml; outputs (geometry/pattern plots,
# STLs for CST validation) are written next to the targets.
set -euo pipefail
python -m scripts.run_inverse --config_file configs/inverse/hand_designed.yaml
