#!/usr/bin/env python3
"""Package the processed corpora into tar.gz shards for public hosting.

Run locally from the repo root on a machine that has the full ``data/corpora/``
tree:

    python scripts/package_corpora.py [--out corpora_release/] [--corpora NAME ...]

For each corpus it tars ``processed/`` (plus the per-corpus stats CSVs) into
independent ~10 GB-uncompressed gzip shards, so any shard extracts on its own
with ``tar -xzf <shard> -C data/corpora/``. raw/ CST results and meshes are NOT
packaged (only needed to regenerate processed files; available on request).

The staging dir ends up holding everything to upload as-is:

    <out>/<corpus>/<corpus>.shard000.tar.gz   per-corpus shards
    <out>/dataset_ff_stats.csv                far-field normalization stats
    <out>/splits/*.pth                        train/test split indices
    <out>/manifest.json                       shard sha256s + expected file counts
    <out>/checksums.sha256

Interrupted runs resume: a corpus whose manifest entry and shard files already
exist is skipped (use --force to rebuild). Download side: scripts/download_corpora.py.
"""

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATASET_VERSION = 'corpora-v1'

# keep in sync with src/dataset/datasets.py:_DEFAULT_SUBDIRS
CORPORA = [
    'classic_patch_no_reflector',
    'classic_patch_with_reflector',
    'classic_rectangle_patch',
    'fmnist_test',
    'fmnist_cifar_test',
    'random_pixel',
    'fmnist_train',
    'fmnist_cifar_train',
]

# loose per-corpus files worth shipping (dataset_split.csv / train_test_indices.pkl
# are legacy and unreferenced by the code; raw/ and s11_pictures/ are excluded)
CORPUS_CSVS = ('dataset_ff_stats.csv', 'dataset_graph_stats.csv')

SHARD_UNCOMPRESSED_BYTES = 10 * (1 << 30)  # ~10 GiB per shard before gzip (~3.3 GiB after)
GZIP_LEVEL = 6


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


def corpus_members(corpus: str) -> list[tuple[Path, str]]:
    """(source path, arcname) pairs; arcnames are relative to data/corpora/."""
    root = REPO_ROOT / 'data/corpora' / corpus
    members: list[tuple[Path, str]] = []
    for csv in CORPUS_CSVS:
        if (root / csv).exists():
            members.append((root / csv, f'{corpus}/{csv}'))
    for stub in ('pre_filter.pt', 'pre_transform.pt'):
        if (root / 'processed' / stub).exists():
            members.append((root / 'processed' / stub, f'{corpus}/processed/{stub}'))
    for f in sorted_processed_files(corpus):
        members.append((f, f'{corpus}/processed/{f.name}'))
    return members


def plan_shards(members: list[tuple[Path, str]]) -> list[list[tuple[Path, str]]]:
    """Greedy split (in id order) into shards of <= SHARD_UNCOMPRESSED_BYTES."""
    shards, current, current_bytes = [], [], 0
    for src, arcname in members:
        size = src.stat().st_size
        if current and current_bytes + size > SHARD_UNCOMPRESSED_BYTES:
            shards.append(current)
            current, current_bytes = [], 0
        current.append((src, arcname))
        current_bytes += size
    if current:
        shards.append(current)
    return shards


def build_shard(out_path: Path, members: list[tuple[Path, str]]) -> dict:
    tmp = out_path.with_suffix(out_path.suffix + '.part')
    uncompressed = 0
    start = time.time()
    with tarfile.open(tmp, 'w:gz', compresslevel=GZIP_LEVEL) as tar:
        for src, arcname in members:
            tar.add(src, arcname=arcname, recursive=False)
            uncompressed += src.stat().st_size
    tmp.rename(out_path)
    info = {
        'file': f'{out_path.parent.name}/{out_path.name}',
        'sha256': sha256_of(out_path),
        'size_bytes': out_path.stat().st_size,
        'uncompressed_bytes': uncompressed,
        'num_members': len(members),
    }
    mb = info['size_bytes'] / 1e6
    print(f'    {out_path.name}: {len(members)} files, {mb:,.0f} MB compressed '
          f'({uncompressed / 1e6:,.0f} MB raw, {time.time() - start:,.0f}s)', flush=True)
    return info


def corpus_is_complete(entry: dict | None, out_dir: Path) -> bool:
    if not entry:
        return False
    for shard in entry.get('shards', []):
        p = out_dir.parent / shard['file']
        if not p.exists() or p.stat().st_size != shard['size_bytes']:
            return False
    return True


def package_corpus(corpus: str, out: Path, manifest: dict, force: bool) -> None:
    out_dir = out / corpus
    if not force and corpus_is_complete(manifest['corpora'].get(corpus), out_dir):
        print(f'  {corpus}: already packaged, skipping (--force to rebuild)', flush=True)
        return
    members = corpus_members(corpus)
    processed_count = sum(1 for _, a in members if 'processed_data_' in a)
    if processed_count == 0:
        sys.exit(f'{corpus}: no processed files found under data/corpora/{corpus}/processed/')
    shards = plan_shards(members)
    total_bytes = sum(src.stat().st_size for src, _ in members)
    print(f'  {corpus}: {processed_count} samples, {total_bytes / 1e9:.1f} GB raw '
          f'-> {len(shards)} shard(s)', flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_infos = []
    for i, shard_members in enumerate(shards):
        shard_infos.append(build_shard(out_dir / f'{corpus}.shard{i:03d}.tar.gz', shard_members))
    manifest['corpora'][corpus] = {
        'processed_file_count': processed_count,
        'uncompressed_bytes': total_bytes,
        'shards': shard_infos,
    }
    save_manifest(out, manifest)


def load_manifest(out: Path) -> dict:
    path = out / 'manifest.json'
    if path.exists():
        return json.loads(path.read_text())
    return {'version': DATASET_VERSION, 'corpora': {}, 'files': {}}


def save_manifest(out: Path, manifest: dict) -> None:
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def stage_extras(out: Path, manifest: dict) -> None:
    """Top-level stats CSV + split files, uploaded uncompressed."""
    extras = {'dataset_ff_stats.csv': REPO_ROOT / 'data/corpora/dataset_ff_stats.csv'}
    for split in sorted((REPO_ROOT / 'splits').glob('*.pth')):
        extras[f'splits/{split.name}'] = split
    for rel, src in extras.items():
        if not src.exists():
            sys.exit(f'missing expected file: {src}')
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        manifest['files'][rel] = {'sha256': sha256_of(dest), 'size_bytes': dest.stat().st_size}
    print(f'  staged {len(extras)} flat files (stats CSV + splits)', flush=True)


def write_checksums(out: Path, manifest: dict) -> None:
    lines = []
    for corpus in sorted(manifest['corpora']):
        for shard in manifest['corpora'][corpus]['shards']:
            lines.append(f"{shard['sha256']}  {shard['file']}")
    for rel in sorted(manifest['files']):
        lines.append(f"{manifest['files'][rel]['sha256']}  {rel}")
    (out / 'checksums.sha256').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--out', type=Path, default=REPO_ROOT / 'corpora_release')
    parser.add_argument('--corpora', nargs='+', choices=CORPORA, default=CORPORA,
                        help='subset of corpora to package (default: all)')
    parser.add_argument('--force', action='store_true', help='rebuild existing shards')
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.out)

    print('staging stats CSV + splits...', flush=True)
    stage_extras(args.out, manifest)
    save_manifest(args.out, manifest)

    print('packaging corpora...', flush=True)
    for corpus in args.corpora:
        package_corpus(corpus, args.out, manifest, args.force)

    write_checksums(args.out, manifest)
    n_shards = sum(len(c['shards']) for c in manifest['corpora'].values())
    total = sum(s['size_bytes'] for c in manifest['corpora'].values() for s in c['shards'])
    print(f'\nDone: {len(manifest["corpora"])} corpora, {n_shards} shards, '
          f'{total / 1e9:.1f} GB compressed in {args.out}/')


if __name__ == '__main__':
    main()
