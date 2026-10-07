#!/usr/bin/env python3
"""Upload the packaged corpora (scripts/package_corpora.py output) to Zenodo.

Creates a DRAFT deposition, uploads every release file, and sets the record
metadata. Nothing is public until --publish (or the Publish button in the web
UI) — review the draft first; publishing is irreversible and mints the DOI.

    export ZENODO_TOKEN=...          # personal access token, 'deposit' scopes
    python scripts/upload_corpora_zenodo.py --staging corpora_release/ [--sandbox]
    # review the printed draft URL, then either publish in the UI or:
    python scripts/upload_corpora_zenodo.py --staging corpora_release/ \
        --deposition-id <id> --publish

Zenodo stores files flat (no directories), so shards upload under their
basename (e.g. fmnist_train.shard002.tar.gz); scripts/download_corpora.py
falls back to basenames automatically when pointed at a Zenodo record with
--base-url https://zenodo.org/records/<id>/files.

Re-running with --deposition-id resumes: files already present with the right
size are skipped.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent

SKIP = {'packaging.log', 'README.md',  # HF dataset card + local log stay off Zenodo
        'examples_overview.png', 'example_frequencies.png'}  # card figures likewise

METADATA = {
    'title': 'GNN for Antenna Design — simulated patch-antenna corpora '
             '(processed graphs, far-fields, surface currents)',
    'upload_type': 'dataset',
    'description': (
        '<p>Dataset for the paper <a href="https://arxiv.org/abs/2610.05004">'
        'Physics-Augmented Graph Transformers for Patch-Antenna Forward and Inverse '
        'Design</a> (arXiv:2610.05004).</p>'
        '<p>Processed training/evaluation corpora for the GNN-for-Antenna-design '
        'benchmark: ~80k pixelated patch-antenna geometries (Fashion-MNIST, '
        'Fashion-MNIST+CIFAR parasitic, random-pixel, and canonical classic-patch '
        'sets) simulated in CST Studio Suite. Each sample is a PyTorch file with the '
        'antenna graph, far-field gain maps at 2.4/2.8/5.2/5.6/6.0 GHz, surface '
        'currents, and S-parameters. Includes the exact train/test split index files '
        'used in the paper and sha256 manifests.</p>'
        '<p>Corpora are packaged as independent tar.gz shards; extract with '
        '<code>tar -xzf &lt;shard&gt; -C data/corpora/</code> or use '
        '<code>scripts/download_corpora.py</code> from the code repository: '
        '<a href="https://github.com/AviEpstein/GNN-for-Antenna-design">'
        'github.com/AviEpstein/GNN-for-Antenna-design</a>.</p>'
        '<p>Raw CST exports (~625 GB: meshes, per-frequency far-field/surface-current '
        'dumps) are available from the authors on request.</p>'
    ),
    'creators': [
        {'name': 'Epstein, Avi', 'affiliation': 'Tel Aviv University'},
        {'name': 'Nehemia, Snir', 'affiliation': 'Tel Aviv University'},
        {'name': 'Suchowski, Haim', 'affiliation': 'Tel Aviv University'},
        {'name': 'Wolf, Lior', 'affiliation': 'Tel Aviv University'},
    ],
    'keywords': ['antenna design', 'graph neural networks', 'electromagnetic simulation',
                 'patch antenna', 'far-field', 'surrogate model', 'inverse design',
                 'machine learning dataset'],
    'license': 'cc-by-4.0',
    'related_identifiers': [
        {'relation': 'isSupplementTo',
         'identifier': 'https://github.com/AviEpstein/GNN-for-Antenna-design',
         'resource_type': 'software'},
        {'relation': 'isSupplementTo',
         'identifier': 'arXiv:2610.05004'},
    ],
}


def release_files(staging: Path) -> list[Path]:
    files = [p for p in sorted(staging.rglob('*'))
             if p.is_file() and p.name not in SKIP and not p.name.endswith('.part')
             and not any(part.startswith('.') for part in p.relative_to(staging).parts)]
    names = [p.name for p in files]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        sys.exit(f'flat-name collision, Zenodo cannot hold these: {dupes}')
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--staging', type=Path, default=REPO_ROOT / 'corpora_release')
    parser.add_argument('--deposition-id', type=int, default=None,
                        help='resume/publish an existing draft instead of creating one')
    parser.add_argument('--publish', action='store_true',
                        help='publish the deposition (IRREVERSIBLE, mints the DOI)')
    parser.add_argument('--sandbox', action='store_true',
                        help='use sandbox.zenodo.org (separate account/token)')
    args = parser.parse_args()

    token = os.environ.get('ZENODO_SANDBOX_TOKEN' if args.sandbox else 'ZENODO_TOKEN')
    if not token:
        sys.exit('set ZENODO_TOKEN (or ZENODO_SANDBOX_TOKEN with --sandbox) first — '
                 'create one at https://zenodo.org/account/settings/applications/ '
                 "with the 'deposit:write' + 'deposit:actions' scopes")
    base = 'https://sandbox.zenodo.org' if args.sandbox else 'https://zenodo.org'
    auth = {'Authorization': f'Bearer {token}'}

    if args.deposition_id:
        r = requests.get(f'{base}/api/deposit/depositions/{args.deposition_id}', headers=auth)
    else:
        r = requests.post(f'{base}/api/deposit/depositions', json={}, headers=auth)
    if r.status_code >= 400:
        sys.exit(f'deposition request failed ({r.status_code}): {r.text[:500]}')
    dep = r.json()
    dep_id = dep['id']
    bucket = dep['links']['bucket']
    print(f'deposition {dep_id}: {base}/uploads/{dep_id}')

    r = requests.put(f'{base}/api/deposit/depositions/{dep_id}',
                     json={'metadata': METADATA}, headers=auth)
    if r.status_code >= 400:
        sys.exit(f'metadata update failed ({r.status_code}): {r.text[:500]}')
    print('metadata set')

    existing = {f['filename']: f['filesize']
                for f in requests.get(f'{base}/api/deposit/depositions/{dep_id}/files',
                                      headers=auth).json()}
    files = release_files(args.staging)
    total = sum(p.stat().st_size for p in files)
    print(f'uploading {len(files)} files, {total / 1e9:.1f} GB total')
    for p in files:
        size = p.stat().st_size
        if existing.get(p.name) == size:
            print(f'  {p.name}: already uploaded, skipping')
            continue
        print(f'  {p.name} ({size / 1e6:,.0f} MB)...', flush=True)
        with open(p, 'rb') as fh:
            r = requests.put(f'{bucket}/{p.name}', data=fh, headers=auth)
        if r.status_code >= 400:
            sys.exit(f'upload of {p.name} failed ({r.status_code}): {r.text[:500]}')
    print('all files uploaded')

    if args.publish:
        confirm = input(f'PUBLISH deposition {dep_id} on {base}? This is irreversible '
                        f'and mints the DOI. Type "publish" to confirm: ')
        if confirm.strip() != 'publish':
            sys.exit('aborted; draft left intact')
        r = requests.post(f'{base}/api/deposit/depositions/{dep_id}/actions/publish',
                          headers=auth)
        if r.status_code >= 400:
            sys.exit(f'publish failed ({r.status_code}): {r.text[:500]}')
        rec = r.json()
        print(f"published: DOI {rec.get('doi')}  ->  {rec['links'].get('record_html')}")
    else:
        print(f'\nDraft ready — review at {base}/uploads/{dep_id} then publish there, '
              f'or re-run with --deposition-id {dep_id} --publish')


if __name__ == '__main__':
    main()
