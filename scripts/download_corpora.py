#!/usr/bin/env python3
"""Download the published processed corpora into data/corpora/.

Stdlib-only (no pip installs needed). From the repo root:

    python scripts/download_corpora.py                          # everything (~36 GB download)
    python scripts/download_corpora.py --corpora classic_rectangle_patch fmnist_test
    python scripts/download_corpora.py --list                   # show sizes, download nothing

Fetches per-corpus tar.gz shards from the Hugging Face dataset repo (override
with --base-url, e.g. a Zenodo record's files URL or a local mirror), verifies
each shard's sha256 against manifest.json, extracts into --dest, and checks the
final processed_data_*.pt count per corpus — the shipped split files in
splits/ index into these exact file sets, so a count mismatch is an error.

Also drops data/corpora/dataset_ff_stats.csv (needed by training) and, with
--with-splits, the splits/*.pth files (already in git for repo users).
"""

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HF_REPO_ID = 'AviEpstein/gnn-antenna-design-dataset'
DEFAULT_BASE_URL = f'https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main'
USER_AGENT = 'gnn-antenna-design/download_corpora'


def fetch(base_url: str, rel: str, dest: Path,
          expected_bytes: int | None = None, quiet: bool = False) -> None:
    """Download base_url/rel -> dest atomically (via .part), with a progress line.

    Zenodo records store files flat (no directories), so a 404 on
    '<corpus>/<shard>' is retried as just '<shard>'.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + '.part')
    candidates = [f'{base_url}/{rel}']
    if '/' in rel:
        candidates.append(f'{base_url}/{rel.rsplit("/", 1)[1]}')
    for i, url in enumerate(candidates):
        req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
        start = time.time()
        try:
            with urllib.request.urlopen(req) as resp, open(tmp, 'wb') as out:
                total = expected_bytes or int(resp.headers.get('Content-Length') or 0)
                done = 0
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if not quiet and total and sys.stdout.isatty():
                        mbps = done / 1e6 / max(time.time() - start, 1e-9)
                        print(f'\r    {dest.name}: {done / 1e6:,.0f}/{total / 1e6:,.0f} MB '
                              f'({100 * done / total:.0f}%, {mbps:.0f} MB/s)',
                              end='', flush=True)
        except urllib.error.HTTPError as e:
            tmp.unlink(missing_ok=True)
            if e.code == 404 and i + 1 < len(candidates):
                continue
            sys.exit(f'\ndownload failed ({e.code} {e.reason}): {url}')
        if not quiet:
            print(flush=True)
        tmp.rename(dest)
        return


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def safe_extract(archive: Path, dest: Path) -> None:
    with tarfile.open(archive, 'r:gz') as tar:
        try:
            tar.extractall(dest, filter='data')  # py >= 3.12 (or 3.8+ with backport)
        except TypeError:
            for m in tar.getmembers():
                if m.name.startswith(('/', '..')) or '/../' in m.name:
                    sys.exit(f'unsafe path in archive {archive.name}: {m.name}')
            tar.extractall(dest)


def processed_count(dest: Path, corpus: str) -> int:
    return len(list((dest / corpus / 'processed').glob('processed_data_*.pt')))


def download_corpus(corpus: str, entry: dict, base_url: str, dest: Path,
                    cache: Path, keep_archives: bool) -> None:
    have = processed_count(dest, corpus)
    want = entry['processed_file_count']
    if have == want:
        print(f'  {corpus}: {have} files already present, skipping', flush=True)
        return
    if have:
        print(f'  {corpus}: found {have}/{want} files, re-extracting all shards', flush=True)
    for shard in entry['shards']:
        archive = cache / Path(shard['file']).name
        if archive.exists() and sha256_of(archive) == shard['sha256']:
            print(f'    {archive.name}: cached, verified', flush=True)
        else:
            fetch(base_url, shard['file'], archive, shard['size_bytes'])
            digest = sha256_of(archive)
            if digest != shard['sha256']:
                archive.unlink()
                sys.exit(f'checksum mismatch for {shard["file"]}: got {digest}, '
                         f'expected {shard["sha256"]} — deleted, re-run to retry')
        safe_extract(archive, dest)
        if not keep_archives:
            archive.unlink()
    have = processed_count(dest, corpus)
    if have != want:
        sys.exit(f'{corpus}: extracted {have} processed files, manifest says {want} — '
                 'the split files in splits/ index into this exact file set. Delete '
                 f'{dest / corpus}/processed/ (it may hold stale extra files) and re-run.')
    print(f'  {corpus}: OK ({want} files)', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--base-url', default=DEFAULT_BASE_URL,
                        help='directory URL serving manifest.json and the shards')
    parser.add_argument('--corpora', nargs='+', default=None,
                        help='subset of corpora (default: all in the manifest)')
    parser.add_argument('--dest', type=Path, default=REPO_ROOT / 'data/corpora')
    parser.add_argument('--with-splits', action='store_true',
                        help='also download splits/*.pth (repo users already have them in git)')
    parser.add_argument('--keep-archives', action='store_true',
                        help='keep downloaded .tar.gz files in the cache dir')
    parser.add_argument('--list', action='store_true', help='print available corpora and exit')
    args = parser.parse_args()

    base_url = args.base_url.rstrip('/')
    cache = args.dest / '.download_cache'
    cache.mkdir(parents=True, exist_ok=True)

    manifest_path = cache / 'manifest.json'
    fetch(base_url, 'manifest.json', manifest_path, quiet=True)
    manifest = json.loads(manifest_path.read_text())

    if args.list:
        print(f'{manifest["version"]} @ {base_url}')
        for name, entry in manifest['corpora'].items():
            dl = sum(s['size_bytes'] for s in entry['shards']) / 1e9
            print(f'  {name:32s} {entry["processed_file_count"]:6d} samples  '
                  f'{dl:6.1f} GB download  {entry["uncompressed_bytes"] / 1e9:6.1f} GB on disk')
        return

    corpora = args.corpora or list(manifest['corpora'])
    unknown = [c for c in corpora if c not in manifest['corpora']]
    if unknown:
        sys.exit(f'unknown corpora {unknown}; available: {list(manifest["corpora"])}')

    dl = sum(s['size_bytes'] for c in corpora for s in manifest['corpora'][c]['shards'])
    disk = sum(manifest['corpora'][c]['uncompressed_bytes'] for c in corpora)
    print(f'downloading {len(corpora)} corpora: {dl / 1e9:.1f} GB transfer, '
          f'{disk / 1e9:.1f} GB extracted (into {args.dest})', flush=True)

    for corpus in corpora:
        download_corpus(corpus, manifest['corpora'][corpus], base_url,
                        args.dest, cache, args.keep_archives)

    for rel, info in manifest.get('files', {}).items():
        if rel.startswith('splits/'):
            if not args.with_splits:
                continue
            target = REPO_ROOT / rel
        else:
            target = args.dest / rel
        if target.exists() and sha256_of(target) == info['sha256']:
            continue
        fetch(base_url, rel, target, info['size_bytes'], quiet=True)
        print(f'  fetched {rel}', flush=True)

    if not args.keep_archives:
        shutil.rmtree(cache, ignore_errors=True)
    print('done.')


if __name__ == '__main__':
    main()
