# Reproducibility guide

This file maps every table and figure of the paper to the command that
regenerates it, records which experiment-tracking run produced each published
number, and lists what cannot be reproduced from this repository alone.

## What you need beyond this repository

| Requirement | Why | Status |
|---|---|---|
| CST Studio Suite license | All labels (far-fields, surface currents, S-parameters) and all inverse-design validation come from CST; the simulation step is manual (see data/README.md) | not distributable |
| Simulated corpora (~530 GB raw / ~80k samples) | training + evaluation data | hosting in preparation; link will be added to data/README.md |
| Trained checkpoints (curated release, one per released row: gps_pais_big surrogate genial-bush-2194; diffusion U-Net; +Phys representative crisp-totem-2365; GPS confused-smoke-2225; GPS+PAIS lively-disco-2160; +PAIS baselines ancient-capybara-2172 (GCN) / expert-sound-2179 (GAT) / fragrant-sun-2223 (Graph U-Net) / rose-hill-2309 (DGCNN) / usual-firebrand-2191 (MGN); each with its config and a metrics.json of re-evaluated Table 1 numbers) | evaluation-only rows, inverse pipeline, per-row load verification | hosting in preparation |

Not hosted (deliberately): the shuffled-SC / pos-aux control runs, the grid
baselines (MLP / U-Net / ResNet-50), the mixed-row run
(gallant-wildflower-2442; see caveat 18), and the extra seed runs of the
multi-seed rows — all retrainable from the shipped configs. The control- and
seed-level re-evaluations recorded in caveats 15-16 used local copies of
those checkpoints.

The split index files are NOT a gap: the exact `.pth` files behind every
split-dependent number ship in `splits/`.

## Table 1 — forward radiation-pattern prediction

Every row: `bash experiments/table1.sh <row>` (see the script for the row
list). DGCNN/MGN/GPS rows are means over seeds 0/42/123 (`--seed`).

| Row | Config | Run(s) (seed 0 / 42 / 123) |
|---|---|---|
| Nearest neighbor | (retrieval; `configs/inverse/nearest_neighbor.yaml`) | — |
| MLP | forward/mlp.yaml | lemon-firefly-2216 |
| U-Net | forward/unet.yaml | hopeful-mountain-2211 |
| ResNet-50 | forward/resnet50.yaml | smooth-fire-2214 |
| GCN / +PAIS | forward/gcn{,_pais}.yaml | copper-darkness-2174 / ancient-capybara-2172 |
| GAT / +PAIS | forward/gat{,_pais}.yaml | desert-dragon-2180 / expert-sound-2179 |
| Graph U-Net / +PAIS | forward/graph_unet{,_pais}.yaml | lemon-terrain-2219 / fragrant-sun-2223 |
| DGCNN | forward/dgcnn.yaml | classic-armadillo-2231 / amber-lake-2306 / fresh-durian-2307 |
| DGCNN + PAIS | forward/dgcnn_pais.yaml | jolly-vortex-2184 / rose-hill-2309 / elated-sky-2308 |
| MGN | forward/mgn.yaml | still-shadow-2186 / lemon-dust-2305 / comic-valley-2304 |
| MGN + PAIS | forward/mgn_pais.yaml | usual-firebrand-2191 / prime-fire-2301 / gentle-cloud-2302 |
| GPS | forward/gps.yaml | confused-smoke-2225 / vocal-breeze-2275 / rose-capybara-2274 |
| GPS + PAIS | forward/gps_pais.yaml | lively-disco-2160 / proud-pine-2271 / warm-dawn-2273 |
| GPS + PAIS + Directional | forward/gps_pais_directional.yaml | (GPSDCC family; see notes) |
| GPS + PAIS + Dir. + Phys (s=0) | forward/gps_pais_dir_phys_s0.yaml | worthy-snowball-2336 |
| GPS + PAIS + Directional + Phys | forward/gps_pais_dir_phys.yaml | mean over fragrant-night-2334 / exalted-cherry-2360 / bright-tree-2363 / honest-music-2364 / crisp-totem-2365 / lucky-star-2366 (all six go in trained_models/ for the mean; the hosted release ships one representative, crisp-totem-2365 — the run closest to the six-run re-evaluated mean; see notes) |
| GPS + shuffled SC | forward/gps_shuffled_sc.yaml | skilled-firebrand-2289 / worldly-puddle-2298 / dutiful-breeze-2299 |
| GPS + pos. aux | forward/gps_pos_aux.yaml | vocal-water-2291 |
| GPS + PAIS (mixed, matched size) | forward/gps_pais_mixed.yaml | gallant-wildflower-2442 |
| GPS + PAIS (big) | forward/gps_pais_big.yaml | genial-bush-2194 |
| GPS + PAIS (big-PCA split) | forward/gps_pais_big_pca.yaml (eval-only) | genial-bush-2194 |
| Zero-shot classic square / +parasitic / rectangular | forward/zeroshot_classic_*.yaml (eval-only) | genial-bush-2194 |

The +Phys row's six shipped checkpoints re-evaluate on this repository's test
split (5.6 GHz) to MAE 0.338–0.345, MSE 0.326–0.338, PSNR 16.71–16.91 per run
(mean 0.341 / 0.332 / 16.82, matching the published 0.34 / 0.33 / 16.78):

```bash
python -m scripts.evaluate_forward --config_file configs/forward/gps_pais_dir_phys.yaml \
    --load_trained_model true --trained_model_path trained_models/<run>_34_34_2400.pt
```

Physics-layer ablation numbers (analytic-integral MSE against ground truth):
`python -m scripts.evaluate_physics_integral_gt --root <corpus>` reproduces
the MSE-1.333 ground-truth-currents figure; the per-model analytic MSEs came
from runs major-monkey-2317 (no physics, 1.392), worthy-snowball-2336 (s=0,
0.838), fragrant-night-2334 (s=0.1, 1.38) via `scripts/evaluate_physics_integral.py`.

Computational-cost paragraph (params/GFLOPs/latency):
`python -m scripts.forward_model_size_and_runtime` and
`python -m scripts.diffusion_model_size_and_runtime`.

## Tables 2 and 3 — inverse design

`bash experiments/table2.sh` and `bash experiments/table3.sh` run every
method; each exports its selected candidates (STL + metadata) under
`output_dir`. Validate those in CST manually, then aggregate:
`python -m scripts.evaluate_cst --cst_output_dirs <dir>` (the docstring of
scripts/evaluate_cst.py maps each Table 2/3 row to the CST output-directory
naming pattern used originally). N-ablation (N=100/1000):
`--number_of_samples_to_generate 100` on the diffusion config.

## Figures

- **Fig. 1**: drawn manually (no generating script).
- **Fig. 2**: `bash experiments/fig2.sh` — per-example prediction plots from
  the zero-shot square+parasitic evaluation.
- **Fig. 3**: `bash experiments/fig3.sh` — hand-designed targets with CFG
  w=2; place one `.pt` far-field per target in `manual_targets_dir`.

## Known discrepancies and caveats

These were found while curating the research code and are deliberately
documented rather than silently changed; **no numerical logic was altered**.

1. **Training recipe**: the paper text says 150 epochs / batch 16; the logged
   configs of the runs behind Table 1 record **300 epochs / batch 64**
   (AdamW 2e-4, wd 1e-5, 5-epoch warmup, cosine to 1e-6 — as stated).
   Similarly the diffusion model trained with batch 128, not 32. The shipped
   configs record what ran.
2. **Corpus sizes**: on disk the three sub-corpora hold ≈26.6k (FMNIST-only),
   ≈30.7k (FMNIST+CIFAR) and ≈13.0k (random-pixel) simulated samples
   (union ≈70k), vs. the paper's "approximately 26,000 each / 80,000 union".
3. **Model classes**: the Directional, Phys(s=0) and Phys rows were trained
   with the `GPSDCC` class. The published +Phys number is the mean over the
   six repeat runs listed in the Table 1 map (fragrant-night-2334 through
   lucky-star-2366).
4. **+Phys repeat runs**: the experiment tracker's logged configs from the
   +Phys era record neither a seed key nor a reliable `model_type`, so the
   six runs are identifiable as repeats of the same configuration only by
   their state-dict architecture, dates and metrics — which of the paper's
   seeds each one used is not recoverable. Five of them
   (exalted-cherry-2360 through lucky-star-2366) were saved with a legacy
   S11 auxiliary head trained on pooled feed-node embeddings alone, which
   predates the current `GPSDCC` S11 input (embeddings concatenated with
   predicted currents) and therefore cannot be loaded by the shipped class.
   The shipped copies have only those unused `s11_pred_head` tensors
   removed; all far-field weights are bit-identical to the originals.
5. **Shuffled-SC control**: the code permutes surface-current rows *within*
   each example (deterministically per example), while the paper says
   "permuted across the dataset". Both destroy node-level physical
   correspondence, which is the point of the control.
6. **HPBW-IoU**: as implemented (`src/metrics/peak_power_metric.py`), the
   denominator is the ground-truth mask area, not the union.
7. **N-hardest subsets**: selected by nearest-neighbor far-field-MSE distance
   from the training set (see `src/dataset/create_N_hardest_split.py`), not
   by distance from the mean PCA projection as the paper text suggests. The
   shipped `splits/hardest_indices_{100,500}.pth` are the exact sets used.
8. **Far-field loss extra term**: `FarfeildLoss` (used when PAIS is off) adds
   a 0.1-weighted radiated-power-normalization regularizer on top of MSE,
   not mentioned in the paper.
9. **Rectangular zero-shot set**: the committed generator defaults do not
   yield 108 examples; `--patch-rows-min 9 --patch-rows-max 9` reproduces
   exactly 108 (see the docstring of
   `data/generation/create_classic_rectangle_patch.py`). The original CLI
   invocation was not recorded.
10. **CMA-ES budget**: expressed as 100 CMA iterations (sigma 0.2) rather
   than an explicit 500-evaluation cap like GA/SA.
11. **Device assumptions**: the GCN/GAT/Graph-U-Net/DGCNN wrappers construct
    CUDA tensors directly and require a GPU; the baseline trainer internally
    overrides its epoch count to 500. These wrappers originally hijacked the
    `load_trained_model` flag in their constructors for a legacy
    sub-model-initialization path (dead code — its
    `trained_model_path_for_*` keys are never set, and GCNWrapper referenced
    a nonexistent attribute), which made checkpoint-based evaluation crash
    before the whole-model load in `scripts/train_forward.py`. The legacy
    blocks are now gated on their own `trained_model_path_for_*` keys
    instead; no numerical logic changed.
12. **Metrics paths**: training-time metrics come from
    `src/losses/losses.py:compute_metrics` (MS-SSIM via pytorch-msssim,
    win 3); the CST-validated tables use `src/metrics/ff_metrics.py:Metrics`
    (torchmetrics MS-SSIM, kernel 5). Both are shipped unchanged.
13. **Random-pixel corpus driver**: the original corpus was generated from a
    CST-side Windows script that is not in the repository;
    `data/generation/create_random_pixel_dataset.py` is a reconstructed
    driver around the original geometry function.
14. **Classic-square zero-shot MAE**: re-running
    `configs/forward/zeroshot_classic_square.yaml` with the shipped
    gps_pais_big checkpoint gives MAE **0.153**, not the 0.25 printed in
    Table 1 — while the row's other metrics reproduce to full published
    precision (MSE 0.0557 → 0.06, MS-SSIM 0.9705 → 0.97, PSNR 24.7558 →
    24.76). The published 0.25 is a transcription error, likely duplicated
    from the "big-PCA split" row directly above. The parasitic and
    rectangular zero-shot rows reproduce on all four metrics.
15. **Control-row re-evaluation**: re-running the original checkpoints
    through `scripts.evaluate_forward` (same metrics path as training)
    reproduces both ablation rows at published rounding except for two
    boundary cases. GPS + shuffled SC per run
    (skilled-firebrand-2289 / worldly-puddle-2298 / dutiful-breeze-2299):
    MAE 0.4081/0.4026/0.4043, MSE 0.4564/0.4481/0.4558, MS-SSIM
    0.8325/0.8362/0.8354, PSNR 15.109/15.243/15.206 — three-run means
    0.405 / 0.453 / 0.835 / **15.19**, vs the published 0.40 / 0.45 /
    0.83 / **15.12**; the first run alone gives 15.11, so the published
    PSNR may reflect one run rather than the mean. GPS + pos. aux
    (vocal-water-2291): 0.4065 / 0.4562 / **0.8347** / 15.133 vs the
    published 0.41 / 0.45 / **0.84** / 15.14 — the MS-SSIM sits at the
    0.835 rounding boundary. Neither affects the rows' conclusion (the
    PAIS gain vanishes under both controls).
16. **GPS / GPS+PAIS rows**: re-evaluating the original checkpoints, the
    GPS row reproduces at published rounding (per run
    confused-smoke-2225 / vocal-breeze-2275 / rose-capybara-2274:
    MAE 0.4052/0.4067/0.4055, MSE 0.4507/0.4545/0.4448, MS-SSIM
    0.8355/0.8336/0.8347, PSNR 15.186/15.147/15.166; means
    0.406 / 0.450 / 0.835 / 15.17 vs the published
    0.40 / 0.45 / 0.83 / 15.20). The GPS+PAIS three-run mean
    (lively-disco-2160 / proud-pine-2271 / warm-dawn-2273: MAE
    0.3677/0.3808/0.3814, MSE 0.3875/0.4075/0.4071, MS-SSIM
    0.8583/0.8493/0.8495, PSNR 16.152/15.737/15.767) is
    **0.377 / 0.401 / 0.852 / 15.89**, one rounding step worse than the
    published **0.37 / 0.39 / 0.86 / 16.10** on every metric — while
    lively-disco-2160 alone rounds to 0.37 / 0.39 / 0.86 / 16.15,
    matching the published row; the published number evidently reflects
    that run rather than the seed mean (cf. the shuffled-SC PSNR in
    item 15). The PAIS gain holds on the honest means (MSE 0.450 →
    0.401, PSNR +0.72 dB), with a somewhat smaller margin than the
    published 0.45 → 0.39 / +0.9 dB.
17. **GPS + PAIS (big) row**: per the authors, the published big row was
    evaluated on the PCA split (`dataset_variant: pca`, as in
    forward/gps_pais_big_pca.yaml), not on a random split of the big
    corpus. On that split the shipped genial-bush-2194 checkpoint
    re-evaluates to MAE 0.2516 / MSE 0.1831 / MS-SSIM 0.9361 /
    PSNR 19.413 — consistent with the printed 0.26 / 0.17 / 0.92 /
    19.67 up to small residual differences (MSE 0.18 vs 0.17, PSNR
    19.41 vs 19.67), and matching the "big-PCA split" row (0.25 / 0.18 /
    0.94 / 19.40) at published rounding; the two printed rows likely
    reflect the same PCA-split evaluation at different epochs. For
    reference, on the shipped big *random* split (test n=6705) the same
    checkpoint gives 0.237 / 0.202 / 0.923 / 22.12, bit-exact the run's
    best logged epoch. All three zero-shot rows from this checkpoint
    reproduce their published numbers.
18. **Mixed-row model class and split**: the "GPS + PAIS (mixed, matched
    size)" run (gallant-wildflower-2442) was logged with `model_type:
    GPSDCC`, directional decoder and physics loss enabled — not plain
    GPS as the row label suggests; `configs/forward/gps_pais_mixed.yaml`
    now records what ran. Its published MAE / MS-SSIM / PSNR match the
    run's logged test metrics (0.3065 / 0.8845 / 19.196 at the final
    epoch), but the printed MSE 0.23 matches no logged epoch (the run's
    minimum is 0.2957). The row also cannot be re-evaluated on disk: the
    shipped `splits/random_split_big_dataset_x3_subset_0.pth` covers
    35,570 samples while the on-disk subset corpus yields 35,569, and on
    such a mismatch the loader silently regenerates **and overwrites**
    the shipped split file (it was restored from git after this was
    discovered). The checkpoint is not part of the hosted release; for
    local copies, a deterministic load-verification fingerprint on the
    zero-shot square set (`--dataset_variant classic_square`) is
    MAE 0.1988 / MSE 0.0928 / MS-SSIM 0.9582 / PSNR 22.169 (n=64).
19. **DGCNN + PAIS seed 0**: the jolly-vortex-2184 checkpoint predates
    the shipped `EdgeCNNClassicWSC` architecture (512-wide vs 1024-wide
    EdgeCNN) and cannot be loaded by the shipped class. The other two
    seeds re-evaluate to 0.4429 / 0.5287 / 0.8106 / 14.234
    (rose-hill-2309) and 0.4437 / 0.5270 / 0.8106 / 14.207
    (elated-sky-2308), whose two-run mean rounds to the published
    0.44 / 0.53 / 0.81 / 14.23.
