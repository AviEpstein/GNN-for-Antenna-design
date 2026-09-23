#!/usr/bin/env bash
# Fig. 2 — zero-shot predictions on square patch antennas with a parasitic
# element. Runs the trained big-dataset GPS+PAIS surrogate on the 64-example
# canonical square+parasitic set (no fine-tuning); the trainer's test pass
# writes per-example prediction/ground-truth plots.
#
# Requires: the canonical-patch corpus simulated in CST (see data/README.md)
# and the gps_pais_big checkpoint at the path in the config.
set -euo pipefail
python -m scripts.evaluate_forward \
  --config_file configs/forward/zeroshot_classic_square_parasitic.yaml "$@"
