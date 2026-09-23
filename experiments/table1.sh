#!/usr/bin/env bash
# Reproduce one row of Table 1 (forward radiation-pattern prediction).
#
#   bash experiments/table1.sh <row> [--seed 42] [other --key value overrides]
#
# Rows and the runs that produced the published numbers are listed in
# REPRODUCIBILITY.md. GNN rows for DGCNN/MGN/GPS report means over seeds
# 0/42/123 (run each seed with --seed).
#
# Trainable rows:
#   mlp unet resnet50                     (grid baselines)
#   gcn gcn_pais gat gat_pais graph_unet graph_unet_pais
#   dgcnn dgcnn_pais mgn mgn_pais
#   gps gps_pais gps_shuffled_sc gps_pos_aux
#   gps_pais_directional gps_pais_dir_phys_s0 gps_pais_dir_phys
#   gps_pais_mixed gps_pais_big
# Evaluation-only rows (need the gps_pais_big checkpoint):
#   gps_pais_big_pca zeroshot_classic_square zeroshot_classic_square_parasitic
#   zeroshot_classic_rectangular
set -euo pipefail
ROW=${1:?usage: bash experiments/table1.sh <row-config-name> [overrides]}
shift || true

case "$ROW" in
  mlp|unet|resnet50)
    python -m scripts.train_baseline --config_file "configs/forward/$ROW.yaml" "$@" ;;
  gps_pais_big_pca|zeroshot_*)
    python -m scripts.evaluate_forward --config_file "configs/forward/$ROW.yaml" "$@" ;;
  *)
    python -m scripts.train_forward --config_file "configs/forward/$ROW.yaml" "$@" ;;
esac
