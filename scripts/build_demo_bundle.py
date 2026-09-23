#!/usr/bin/env python3
"""Package the sample data + checkpoints that notebooks/demo.ipynb downloads.

Run locally from the repo root on a machine that has the full ``data/corpora/``
tree and the staged ``release/`` checkpoint folder:

    python scripts/build_demo_bundle.py [--out demo_bundle_staging/] [--force]

It stages a ``work/`` tree whose paths mirror the repo root (so the notebook can
extract each archive with ``tar -xzf <archive> -C <repo_root>``) and produces
three archives plus ``manifest.json`` and ``checksums.sha256``:

    demo_checkpoints.tar.gz   4 forward checkpoints -> trained_models/,
                              diffusion checkpoint  -> checkpoints/diffusion/
    demo_data_tier1.tar.gz    the two 64-example classic-patch corpora
                              (processed/ only) + dataset_ff_stats.csv
    demo_data_tier23.tar.gz   20 held-out PCA-test samples, the 10 hardest
                              inverse targets, and precomputed nearest-neighbor
                              retrievals for those targets

Upload the contents of ``<out>/archives/`` anywhere that serves plain files
(GitHub Release assets work well), then paste the base URL into the
``BUNDLE_BASE_URL`` constant near the top of ``notebooks/demo.ipynb``.
"""

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
from pathlib import Path

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent

# release-dir -> destination (paths relative to repo root)
CHECKPOINTS = {
    'release/forward/gps_pais_big/genial-bush-2194GNN_ff_foward_34_34_2400.pt': 'trained_models/',
    'release/forward/gps/confused-smoke-2225GNN_ff_foward_34_34_2400.pt': 'trained_models/',
    'release/forward/gps_pais/lively-disco-2160GNN_ff_foward_34_34_2400.pt': 'trained_models/',
    'release/forward/gps_pais_dir_phys/crisp-totem-2365_34_34_2400.pt': 'trained_models/',
    'release/inverse/diffusion/diffusion_model.pt': 'checkpoints/diffusion/',
}

CLASSIC_CORPORA = ['classic_patch_no_reflector', 'classic_patch_with_reflector']

NUM_TEST_SAMPLES = 20
NUM_HARD_TARGETS = 10
SAMPLE_SEED = 0
IDX_FREQ = 3  # 5.6 GHz, matches config['idx_freq'] in base_forward.yaml / base_inverse.yaml


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def sorted_processed_files(corpus: str) -> list[Path]:
    """Numeric-suffix sort, exactly as PixelDataset.processed_file_names does."""
    files = list((REPO_ROOT / 'data/corpora' / corpus / 'processed').glob('processed_data_*.pt'))
    files.sort(key=lambda p: int(p.stem.split('_')[-1]))
    return files


def pca_combined_files() -> list[Path]:
    """File list matching ConcatDataset([fc_train, fc_test]) global indexing."""
    return sorted_processed_files('fmnist_cifar_train') + sorted_processed_files('fmnist_cifar_test')


def stage_checkpoints(work: Path) -> None:
    known = {}
    checksums_path = REPO_ROOT / 'release/checksums.sha256'
    if checksums_path.exists():
        for line in checksums_path.read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            known['release/' + name.strip()] = digest
    for src_rel, dest_rel in CHECKPOINTS.items():
        src = REPO_ROOT / src_rel
        if not src.exists():
            sys.exit(f'missing checkpoint: {src} (is release/ staged on this machine?)')
        if src_rel in known:
            actual = sha256_of(src)
            if actual != known[src_rel]:
                sys.exit(f'checksum mismatch for {src_rel}: {actual} != {known[src_rel]}')
        dest = work / dest_rel
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest / src.name)
        print(f'  staged {src.name} -> {dest_rel}')


def stage_classic_corpora(work: Path) -> None:
    for corpus in CLASSIC_CORPORA:
        src = REPO_ROOT / 'data/corpora' / corpus / 'processed'
        files = sorted_processed_files(corpus)
        if len(files) != 64:
            sys.exit(f'{corpus}: expected 64 processed files, found {len(files)}')
        dest = work / 'data/corpora' / corpus / 'processed'
        dest.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy2(f, dest / f.name)
        for stub in ('pre_filter.pt', 'pre_transform.pt'):
            if (src / stub).exists():
                shutil.copy2(src / stub, dest / stub)
        print(f'  staged {corpus}/processed ({len(files)} samples)')
    stats = REPO_ROOT / 'data/corpora/dataset_ff_stats.csv'
    dest = work / 'data/corpora'
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(stats, dest / stats.name)
    print('  staged dataset_ff_stats.csv')


def select_pca_test_samples(work: Path) -> None:
    split = torch.load(REPO_ROOT / 'splits/pca_extrapolation_split.pth', weights_only=False)
    combined = pca_combined_files()
    assert len(combined) == 30689, f'combined corpus size drifted: {len(combined)}'
    test_idx = list(split['test'])
    perm = torch.randperm(len(test_idx), generator=torch.Generator().manual_seed(SAMPLE_SEED))
    chosen_positions = perm[:NUM_TEST_SAMPLES].tolist()

    dest = work / 'data/demo/pca_test_20'
    dest.mkdir(parents=True, exist_ok=True)
    manifest = []
    for k, pos in enumerate(chosen_positions):
        gidx = int(test_idx[pos])
        src = combined[gidx]
        corpus = src.parents[1].name
        out_name = f'sample_{k:02d}_g{gidx}.pt'
        shutil.copy2(src, dest / out_name)
        manifest.append({
            'file': out_name,
            'global_idx': gidx,
            'test_subset_pos': pos,
            'source_corpus': corpus,
            'source_file': src.name,
            'sha256': sha256_of(src),
        })
    (dest / 'samples_manifest.json').write_text(json.dumps(
        {'seed': SAMPLE_SEED, 'split': 'splits/pca_extrapolation_split.pth', 'samples': manifest},
        indent=2))
    print(f'  staged {NUM_TEST_SAMPLES} PCA-test samples')


def export_hard_targets(work: Path) -> list[dict]:
    split = torch.load(REPO_ROOT / 'splits/pca_extrapolation_split.pth', weights_only=False)
    hardest = torch.load(REPO_ROOT / 'splits/hardest_indices_100.pth', weights_only=False)
    combined = pca_combined_files()
    test_idx = list(split['test'])

    dest = work / 'data/demo/manual_targets_hard10'
    dest.mkdir(parents=True, exist_ok=True)
    manifest = []
    for rank, pos in enumerate(hardest[:NUM_HARD_TARGETS]):
        gidx = int(test_idx[int(pos)])
        src = combined[gidx]
        ex = torch.load(src, weights_only=False)
        target = ex['farfeilds'][IDX_FREQ].clone()  # bare [34,34], linear scale
        out_name = f'hard_{rank:02d}_g{gidx}.pt'
        torch.save(target, dest / out_name)
        manifest.append({
            'file': out_name,
            'hardest_rank': rank,
            'test_subset_pos': int(pos),
            'global_idx': gidx,
            'source_corpus': src.parents[1].name,
            'source_file': src.name,
        })
    (dest / 'targets_manifest.json').write_text(json.dumps(
        {'idx_freq': IDX_FREQ, 'frequency_mhz': 5600, 'targets': manifest}, indent=2))
    print(f'  staged {NUM_HARD_TARGETS} hard targets')
    return manifest


def precompute_nn(work: Path, targets: list[dict]) -> None:
    """Nearest-neighbor retrieval baseline, mirroring AntennaNearestNeighbor's
    'calc_nn_from_farfeild' mode: features are the raw flattened linear-scale
    far-fields at idx_freq, no normalization, sklearn NearestNeighbors."""
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    split = torch.load(REPO_ROOT / 'splits/pca_extrapolation_split.pth', weights_only=False)
    combined = pca_combined_files()
    train_files = [combined[int(g)] for g in split['train']]

    feats = np.empty((len(train_files), 34 * 34), dtype=np.float32)
    for i, f in enumerate(tqdm(train_files, desc='reading PCA-train far-fields', unit=' files')):
        ex = torch.load(f, weights_only=False)
        feats[i] = ex['farfeilds'][IDX_FREQ].numpy().astype(np.float32).flatten()

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(feats)

    results = {}
    for t in targets:
        target_path = work / 'data/demo/manual_targets_hard10' / t['file']
        target = torch.load(target_path, weights_only=False)
        _, indices = nbrs.kneighbors([target.numpy().flatten()])
        j = int(indices[0][0])
        src = train_files[j]
        ex = torch.load(src, weights_only=False)
        params = ex['example_paramters']['ant_parameters']
        nn_ff = ex['farfeilds'][IDX_FREQ].clone()
        clamped = lambda a, b: ((torch.clamp(a, -15, 10) - torch.clamp(b, -15, 10)) ** 2).mean().item()
        results[t['file']] = {
            'nn_matrix': torch.as_tensor(params['matrix']).clone(),
            'nn_reflector': torch.as_tensor(params['reflector_matrix']).clone()
                            if 'reflector_matrix' in params else None,
            'nn_farfield': nn_ff,
            'nn_mse': ((nn_ff - target) ** 2).mean().item(),
            'nn_mse_clamped': clamped(nn_ff, target),
            'nn_source': {'corpus': src.parents[1].name, 'file': src.name,
                          'global_idx': int(split['train'][j])},
        }
        print(f"  {t['file']}: NN from {src.name} (mse {results[t['file']]['nn_mse']:.4f})")

    dest = work / 'data/demo/nn_precomputed'
    dest.mkdir(parents=True, exist_ok=True)
    torch.save(results, dest / 'nn_results.pt')


ARCHIVES = {
    'demo_checkpoints.tar.gz': {
        'paths': ['trained_models', 'checkpoints'],
        'description': 'GPS+PAIS (big), GPS, GPS+PAIS, GPS+Dir+Phys forward checkpoints '
                       'and the conditional diffusion U-Net',
    },
    'demo_data_tier1.tar.gz': {
        'paths': ['data/corpora'],
        'description': 'Zero-shot classic-patch corpora (64 square + 64 parasitic, '
                       'processed only) and far-field normalization stats',
    },
    'demo_data_tier23.tar.gz': {
        'paths': ['data/demo'],
        'description': '20 held-out PCA-test samples, 10 hardest inverse targets, '
                       'precomputed nearest-neighbor retrievals',
    },
}


def build_archives(work: Path, archives_dir: Path, force: bool) -> None:
    archives_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, spec in ARCHIVES.items():
        out = archives_dir / name
        if out.exists() and not force:
            print(f'  {name} exists, skipping (use --force to rebuild)')
        else:
            with tarfile.open(out, 'w:gz') as tar:
                for rel in spec['paths']:
                    tar.add(work / rel, arcname=rel)
            print(f'  built {name} ({out.stat().st_size / 1e6:.1f} MB)')
        manifest[name] = {
            'sha256': sha256_of(out),
            'size_bytes': out.stat().st_size,
            'top_level_paths': spec['paths'],
            'description': spec['description'],
        }
    (archives_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    with open(archives_dir / 'checksums.sha256', 'w') as f:
        for name, info in manifest.items():
            f.write(f"{info['sha256']}  {name}\n")
        f.write(f"{sha256_of(archives_dir / 'manifest.json')}  manifest.json\n")
    print('  wrote manifest.json + checksums.sha256')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--out', type=Path, default=REPO_ROOT / 'demo_bundle_staging')
    parser.add_argument('--force', action='store_true', help='restage and rebuild everything')
    parser.add_argument('--skip-nn', action='store_true',
                        help='skip the (slow) nearest-neighbor precompute')
    args = parser.parse_args()

    work = args.out / 'work'
    archives_dir = args.out / 'archives'
    if args.force and work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    print('staging checkpoints...')
    stage_checkpoints(work)
    print('staging classic corpora...')
    stage_classic_corpora(work)
    print('selecting PCA-test samples...')
    select_pca_test_samples(work)
    print('exporting hard targets...')
    targets = export_hard_targets(work)
    if args.skip_nn:
        print('skipping NN precompute (--skip-nn)')
    elif (work / 'data/demo/nn_precomputed/nn_results.pt').exists() and not args.force:
        print('NN precompute exists, skipping')
    else:
        print('precomputing nearest-neighbor retrievals (reads all PCA-train files, slow)...')
        precompute_nn(work, targets)
    print('building archives...')
    build_archives(work, archives_dir, args.force)

    print(f"""
Done. Upload the files in {archives_dir}/ (all three .tar.gz + manifest.json)
to any static host — GitHub Release assets on the public repo work well:
    gh release create demo-bundle-v1 {archives_dir}/*.tar.gz {archives_dir}/manifest.json
Then set BUNDLE_BASE_URL in notebooks/demo.ipynb to the directory URL, e.g.
    https://github.com/<owner>/<repo>/releases/download/demo-bundle-v1/
""")


if __name__ == '__main__':
    main()
