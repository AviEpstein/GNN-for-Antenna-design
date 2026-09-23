# Physics-Augmented Graph Transformers for Patch-Antenna Forward and Inverse Design

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AviEpstein/GNN-for-Antenna-design/blob/public/notebooks/demo.ipynb)

Official code for the IEEE MLSP 2026 paper
**"Physics-Augmented Graph Transformers for Patch-Antenna Forward and Inverse Design"**
— Avi Epstein, Snir Nehemia, Haim Suchowski, Lior Wolf (Tel Aviv University).
The accepted manuscript is included at [docs/paper.pdf](docs/paper.pdf).

> Full-wave electromagnetic simulation enables accurate patch-antenna analysis but is
> computationally expensive for large-scale forward prediction and inverse design. We present a
> mesh-native, physics-augmented graph-learning framework that treats radiation-pattern
> prediction as signal reconstruction on an irregular surface mesh. A GPS graph transformer is
> trained with Physics-Augmented Intermediate Supervision (PAIS) — an auxiliary node-level
> objective predicting complex surface currents, the physical intermediate linking geometry to
> radiation — plus a direction-conditioned decoder and a differentiable radiation-integral
> consistency loss. For inverse design, a conditional diffusion model proposes candidate
> geometries which the trained surrogate ranks before CST validation.

Pipeline (paper Fig. 1): mesh graph G → GPS(+PAIS) → surface currents Ĵ → radiation pattern F̂;
inverse: target F* → conditional diffusion U-Net → 500 candidates → surrogate ranking → top-5 CST validation.

## Quick demo (Colab)

[notebooks/demo.ipynb](notebooks/demo.ipynb) runs the released models with zero setup — click
the badge above. Three tiers, stop at any point: **Tier 1** (~3 min, free CPU runtime) loads
one antenna, predicts its radiation pattern and surface currents against CST ground truth, and
lets you design zero-shot square patches interactively; **Tier 2** (~5 min, CPU) reproduces
Table-1-style metrics on a bundled sample set, makes the PAIS ablation visible, and plots the
direction-conditioned attention maps; **Tier 3** (~8 min, needs a free T4 GPU runtime) runs
diffusion-based inverse design with surrogate ranking. The notebook auto-downloads a ~100 MB
bundle of sample data and checkpoints (built by `scripts/build_demo_bundle.py`). To run
locally instead: `pip install -r requirements.txt`, download or build the bundle, then open
the notebook from the repo root.

## Repository layout

```
data/generation/   geometry generators (FMNIST / FMNIST+CIFAR / random-pixel / canonical patches)
data/README.md     CST simulation instructions + expected data layout
src/
  geometry/ graph/ dataset/    mesh construction, graph features (LapPE k=10), PixelDataset, splits
  models/                      GPS, GPSDCC + GCN/GAT/GraphUNet/DGCNN/MGN and grid baselines
  losses/ physics/ metrics/    PAIS + physics-consistency losses, radiation integral, metrics
  training/                    forward/baseline trainers
  diffusion/ inverse/          conditional DDPM, surrogate-filtered generation, CMA-ES/GA/SA/NN baselines
configs/           base_forward.yaml + one yaml per Table 1 row (configs/forward/) and per
                   inverse method (configs/inverse/)
scripts/           entry points (train_forward, train_baseline, train_diffusion, run_inverse, ...)
experiments/       one command per paper table/figure
splits/            the exact split index files used in the paper (<1 MB)
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126   # match your CUDA
pip install -r requirements.txt
```

Tested with Python 3.12, CUDA 12.x, a single NVIDIA RTX 4090. Some graph baselines
(GCN/GAT/Graph U-Net/DGCNN) construct CUDA tensors directly and require a GPU.
Weights & Biases logging is on by default; run with `WANDB_MODE=disabled` to opt out.

## Data

The benchmark pairs generated geometries with CST Microwave Studio simulations at 5.6 GHz
(three sub-corpora — FMNIST-only, FMNIST+CIFAR, random-pixel — union ≈80k samples, with
radiation-pattern and surface-current ground truth). Geometry generation is fully scripted;
the CST step is manual — see [data/README.md](data/README.md) for solver settings, export
naming, and the expected on-disk layout. A download link for the simulated corpora will be
added to data/README.md.

## Forward problem (Table 1)

Each Table 1 row has a config under `configs/forward/`; the config header names the
experiment-tracking run that produced the published number. From the repo root:

```bash
# GPS + PAIS on FMNIST+CIFAR (random split)
python -m scripts.train_forward --config_file configs/forward/gps_pais.yaml
# seeds: Table 1 GNN rows report means over seeds 0/42/123
python -m scripts.train_forward --config_file configs/forward/gps_pais.yaml --seed 42

# grid baselines
python -m scripts.train_baseline --config_file configs/forward/mlp.yaml

# + directional decoder + physics-consistency loss (single-frequency GPSDCC)
python -m scripts.train_forward --config_file configs/forward/gps_pais_dir_phys.yaml

# scaling rows
python -m scripts.train_forward --config_file configs/forward/gps_pais_big.yaml
python -m scripts.train_forward --config_file configs/forward/gps_pais_mixed.yaml

# evaluation-only rows (PCA split, zero-shot canonical patches)
python -m scripts.evaluate_forward --config_file configs/forward/gps_pais_big_pca.yaml
python -m scripts.evaluate_forward --config_file configs/forward/zeroshot_classic_square.yaml
```

See [experiments/](experiments/) for the full row-by-row command list and
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the run map and known caveats
(note: the configs record the training recipe of the logged runs — 300 epochs,
batch 64, AdamW 2e-4 — which differs from the abbreviated recipe stated in the paper text).

## Inverse problem (Tables 2–3, Fig. 3)

```bash
# 1) train the conditional diffusion model
python -m scripts.train_diffusion --config_file configs/inverse/diffusion_train.yaml

# 2) surrogate-filtered generation on the 100 hardest PCA-split targets
#    (500 candidates/target, top-5 by GPS+PAIS surrogate score exported for CST validation)
python -m scripts.run_inverse --config_file configs/inverse/diffusion.yaml

# 3) baselines under the same 500-evaluation budget
python -m scripts.run_inverse --config_file configs/inverse/nearest_neighbor.yaml
python -m scripts.run_inverse --config_file configs/inverse/cma_baseline.yaml
python -m scripts.run_inverse --config_file configs/inverse/baselines.yaml   # GA, SA, NN-seeded SA
python -m scripts.run_inverse --config_file configs/inverse/diffusion_random5.yaml

# 4) after validating exported candidates in CST, aggregate the tables
python -m scripts.evaluate_cst --help
```

## Citation

```bibtex
@inproceedings{epstein2026physics,
  title     = {Physics-Augmented Graph Transformers for Patch-Antenna Forward and Inverse Design},
  author    = {Epstein, Avi and Nehemia, Snir and Suchowski, Haim and Wolf, Lior},
  booktitle = {2026 IEEE International Workshop on Machine Learning for Signal Processing (MLSP)},
  year      = {2026},
  publisher = {IEEE}
}
```

## License

MIT — see [LICENSE](LICENSE).
