# Data: generation, simulation, and expected layout

The benchmark couples generated 3D patch-antenna geometries with full-wave CST
simulations. The geometry side is fully scripted here; **the CST simulation
step is manual** — the automation macros used originally ran on a separate
Windows/CST machine and are not part of this repository.

## 1. Generate geometries

Each generator writes, per example, a folder of STL files plus a
`matrix_and_env_dict.pkl` with the occupancy matrices and environment
parameters:

```bash
# FMNIST-only corpus (patch layer from Fashion-MNIST)
python -m data.generation.create_fmnist_dataset --output-dir <out>

# FMNIST+CIFAR corpus (adds the CIFAR-derived parasitic layer)
python -m data.generation.create_fmnist_cifar_dataset --output-dir <out>

# random-pixel corpus (i.i.d. random binary patches)
python -m data.generation.create_random_pixel_dataset --output-dir <out> --num-examples N

# zero-shot canonical sets (Table 1 lower block)
python -m data.generation.create_classic_square_patch --output-dir <out>                 # 64 + 64 (with/without parasitic)
python -m data.generation.create_classic_rectangle_patch --output-dir <out>              # rectangular sweep
```

Every antenna is a stack of: PEC parasitic layer (optional), PEC patch layer,
a PEC feed column at the patch's maximum-activation pixel, an FR4 substrate
(h = 4 mm, eps_r = 3.55, tan_d = 0.0027) and a PEC ground plane
(50 x 50 mm; patch area 28 x 28 mm on a 16 x 16 pixel grid).

## 2. Simulate in CST Studio (manual)

For each example folder, in CST Studio Suite (2024 was used):

1. Import the STLs (`Dielectric.stl`, `Feed.stl`, `PEC_ground.stl`,
   `PEC_pixel.stl`, and `PEC_Reflector.stl` when present); assign PEC to all
   metal parts and FR4 (eps_r 3.55, tan d 0.0027) to the dielectric.
2. Excite with a discrete port across the feed gap (between the feed column
   and the ground plane).
3. Time-domain solver, **open (add space) boundary conditions**, simulation
   band covering 2.4-6.0 GHz.
4. Export, per frequency in {2400, 2800, 5200, 5600, 6000} MHz
   (the paper evaluates at **5.6 GHz**):
   - the far-field gain over the elevation-azimuth sphere, saved as
     `farfield_<freq>.npy` (linear gain; the loader resizes to 34x34 and
     normalizes to directivity),
   - the surface-current distribution as an exported point cloud, saved as
     `surface current (f=<freq>) [1].pkl`,
   - S-parameters as `S_parameters.pickle`.

## 3. Expected on-disk layout

`configs/*.yaml` point `data_root` at a directory containing one folder per
corpus (see `src/dataset/datasets.py` for the expected names). Each corpus
folder must look like:

```
<corpus>/
  raw/
    meshes/<id>/          Antenna_Feed_PEC_STEP.stl, Antenna_PEC_STEP.stl,
                          Dielectric.stl, Feed.stl, PEC_ground.stl,
                          PEC_pixel.stl, [PEC_Reflector.stl],
                          matrix_and_env_dict.pkl
    CST_results/<id>/     farfield_2400.npy ... farfield_6000.npy,
                          "surface current (f=5600) [1].pkl", S_parameters.pickle
  processed/              processed_data_<id>.pt   (written by the loader on first pass)
```

## 4. Download

The simulated corpora (~80k examples, far-field + surface-current ground
truth, **processed form**: the `processed/processed_data_*.pt` files the
loader reads) are published as per-corpus tar.gz shards with a sha256
manifest, mirrored in two places:

- **Hugging Face Hub** (primary): https://huggingface.co/datasets/AviEpstein/gnn-antenna-design-dataset
- **Zenodo** (archival, citable DOI): see the DOI badge in the top-level README

From the repo root (stdlib only, no extra installs):

```bash
python scripts/download_corpora.py --list            # corpora + sizes
python scripts/download_corpora.py                   # everything: ~36 GB download, ~110 GB on disk
python scripts/download_corpora.py --corpora classic_rectangle_patch fmnist_test
# from Zenodo instead of Hugging Face:
python scripts/download_corpora.py --base-url https://zenodo.org/records/<id>/files
```

The script verifies every shard's sha256 and checks per-corpus file counts
against `manifest.json` — the split files in `splits/` index into these exact
file sets, so counts must match. It also places
`data/corpora/dataset_ff_stats.csv` (far-field normalization stats needed by
training). Manual alternative: download any shard and
`tar -xzf <shard> -C data/corpora/`.

The **raw CST exports** (`raw/`: meshes + per-frequency far-field /
surface-current dumps / S-parameters, ~625 GB) are not hosted; they are only
needed to regenerate the processed files and are available from the authors on
request. The packaging side lives in `scripts/package_corpora.py` and
`scripts/upload_corpora_zenodo.py`.

The trained checkpoints (curated release: forward surrogates + diffusion
model; manifest in REPRODUCIBILITY.md) are packaged separately; a download
link will be added here. Forward checkpoints go in `trained_models/`, the
diffusion model at `checkpoints/diffusion/diffusion_model.pt`; each ships
with its config and a `metrics.json` of re-evaluated Table 1 numbers so a
correct load can be confirmed with `python -m scripts.evaluate_forward`.
