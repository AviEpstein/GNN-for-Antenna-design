"""On-disk cache for `prepare_graph`'s output, keyed on the input processed-file
identity plus a config fingerprint.

Motivation: with `connect_components` set, `prepare_graph` bridges disconnected
components (`knn_component_bridge_edges`), recomputes `edge_attr` with the 5th
(mesh=0/bridge=1) type column, and recomputes LapPE on the bridged graph -- all of
which is pure, deterministic work given (raw graph, config). Measured at ~73ms/graph,
this dropped patch training (batch 64) from ~3 batches/sec to ~0.45. None of the three
depends on anything but the on-disk processed graph and the config, so the result can
be computed once and replayed.

What is cached: the full `(graph, graph_dict)` tuple `prepare_graph` returns --
edge_index, edge_attr, pe, pos, x, and every other field it sets. NOT anything
`PixelDataset.get()` attaches afterwards (surface_currents, s11, freq_hz, farfield
selection) and NOT anything touched by training-time augmentation -- those are still
computed fresh on every `get()` call, cache or no cache.

Augmentation guard (see `_cache_enabled`): PixelDataset.get() applies rotation/shift
augmentation to the graph *before* calling prepare_graph, when `is_training` and the
augment flags are set. Rotation changes positions, normals, edge_attr, and invalidates
LapPE, so a cached prepared graph is only ever valid for an unaugmented input. Chosen
fix is option (a) from the task: caching is refused outright whenever
`augment_rotation`/`augment_shift` are configured on, regardless of the
`prepared_graph_cache` flag's value -- not option (b) (apply augmentation after
retrieval), which would need a from-scratch argument that rotating a cached edge_attr/
pe post-hoc is still correct, and the current patch runs don't need it (no rotation
augmentation planned).
"""

import hashlib
import json
import os
from pathlib import Path

import torch

from src.graph.GNN_functions import prepare_graph

# Bump whenever prepare_graph's or knn_component_bridge_edges' *logic* changes in a
# way that would change their output for the same input -- this is folded into the
# cache directory fingerprint, so bumping it invalidates every existing cache entry
# instead of silently serving stale output computed by old code.
PREPARE_GRAPH_CACHE_VERSION = "v1"

# Config keys whose value affects prepare_graph's output. "add_sphere and its params"
# from the task means add_sphere plus sphere_radius/num_sphere_nodes (only consulted
# when add_sphere is truthy, but harmless -- and safer -- to always fingerprint).
_CACHE_RELEVANT_CONFIG_KEYS = (
    'connect_components',
    'add_radius_graph_edges',
    'merge_duplicate_nodes',
    'add_sphere',
    'sphere_radius',
    'num_sphere_nodes',
    'physical_antenna',
    'use_node_probs',
)

_warned_reasons = set()


def _warn_once(msg):
    if msg not in _warned_reasons:
        _warned_reasons.add(msg)
        print(f'[prepared_graph_cache] {msg}')


def _cache_enabled(config):
    """Whether prepare_graph_cached should attempt the cache for this config.

    Returns (enabled, reason_if_disabled).
    """
    connect_components_cfg = config.get('connect_components')
    if not connect_components_cfg:
        # Legacy path: connect_components unset -> untouched, zero behavior change.
        return False, 'connect_components not set'

    augment_rotation = bool(config.get('augment_rotation', False))
    augment_shift = bool(config.get('augment_shift', False))
    geometric_augmentation_on = augment_rotation or augment_shift

    default = not geometric_augmentation_on
    explicit = 'prepared_graph_cache' in config
    requested = config.get('prepared_graph_cache', default)

    if requested and geometric_augmentation_on:
        if explicit:
            _warn_once(
                'prepared_graph_cache=True requested but augment_rotation/augment_shift is enabled -- '
                'rotation/shift change pos, normals, edge_attr and pe, so a graph prepared before '
                'augmentation cannot be safely cached. Disabling prepared_graph_cache for this run.'
            )
        return False, 'geometric augmentation enabled'

    if not requested:
        reason = 'prepared_graph_cache=False in config' if explicit else 'geometric augmentation enabled (default off)'
        return False, reason

    return True, None


def dataset_root_hash(dataset_root):
    return hashlib.sha256(os.path.abspath(dataset_root).encode('utf-8')).hexdigest()


def compute_config_fingerprint(config, pe_k):
    """Canonical fingerprint over every config key that affects prepare_graph's
    output, plus the input pe dimension (k) -- which is not a config key but is set
    once per dataset by the pre_transform used at process()-time -- and the hard-coded
    cache-format/logic version.
    """
    relevant = {key: config.get(key) for key in _CACHE_RELEVANT_CONFIG_KEYS}
    relevant['_pe_k'] = pe_k
    relevant['_cache_version'] = PREPARE_GRAPH_CACHE_VERSION
    canonical = json.dumps(relevant, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def get_prepared_graph_cache_dir(dataset_root, config, pe_k):
    fingerprint = compute_config_fingerprint(config, pe_k)
    return (Path.home() / '.cache' / 'antenna_loader' / dataset_root_hash(dataset_root)
            / 'prepared_graphs' / fingerprint)


def prepare_graph_cached(graph, config, *, dataset_root, raw_idx, source_path):
    """Drop-in replacement for `prepare_graph(graph, config)` that caches the result
    to disk when `_cache_enabled(config)` allows it, keyed on `raw_idx` under a
    directory fingerprinted by the config (see `compute_config_fingerprint`) and
    validated per-entry against `source_path`'s mtime+size (so a re-processed raw
    example invalidates just its own entry, not the whole cache).

    On any cache miss/disable, falls back to calling `prepare_graph` directly --
    behavior (including the strict_pe_check tripwire) is unchanged from before this
    cache existed. On a cache hit, `prepare_graph` is not called at all: the tripwire
    does not re-run, since it already ran (and passed) against these exact bytes at
    write time, and re-running it against unchanged bytes on every hit would defeat
    the point of caching the expensive path it inspects. See test_prepared_graph_cache.py.
    """
    enabled, _reason = _cache_enabled(config)
    if not enabled:
        return prepare_graph(graph, config)

    pe_k = int(graph.pe.shape[-1]) if getattr(graph, 'pe', None) is not None else None
    cache_dir = get_prepared_graph_cache_dir(dataset_root, config, pe_k)
    cache_file = cache_dir / f'{raw_idx}.pt'

    try:
        stat = os.stat(source_path)
        src_mtime, src_size = stat.st_mtime, stat.st_size
    except OSError:
        src_mtime, src_size = None, None

    if cache_file.exists():
        try:
            cached = torch.load(cache_file, weights_only=False)
        except Exception:
            cached = None
        if (cached is not None
                and cached.get('cache_version') == PREPARE_GRAPH_CACHE_VERSION
                and cached.get('src_mtime') == src_mtime
                and cached.get('src_size') == src_size):
            return cached['graph'], cached['graph_dict']

    graph_out, graph_dict_out = prepare_graph(graph, config)

    cache_dir.mkdir(parents=True, exist_ok=True)
    # Write-then-atomic-rename so concurrent DataLoader workers racing on the same
    # (unlikely but possible, e.g. across epoch boundaries) idx never observe a
    # partially-written file.
    tmp_file = cache_dir / f'{raw_idx}.pt.tmp.{os.getpid()}'
    torch.save({
        'cache_version': PREPARE_GRAPH_CACHE_VERSION,
        'src_mtime': src_mtime,
        'src_size': src_size,
        'graph': graph_out,
        'graph_dict': graph_dict_out,
    }, tmp_file)
    os.replace(tmp_file, cache_file)

    return graph_out, graph_dict_out
