import hashlib
import os

import torch


def sha256_of_file(path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()


def load_checkpoint_relaxed(model, checkpoint_path, device='cpu'):
    """Loads a raw state_dict into model with strict=False.

    Returns a report of which parameters were loaded, dropped (present in the
    checkpoint but not in the model), and randomly-initialized (present in the
    model but not in the checkpoint).
    """
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    result = model.load_state_dict(state_dict, strict=False)
    missing_keys = list(result.missing_keys)
    unexpected_keys = list(result.unexpected_keys)
    loaded_keys = [k for k in state_dict.keys() if k not in unexpected_keys]
    return {
        'checkpoint_path': checkpoint_path,
        'n_loaded': len(loaded_keys),
        'loaded_keys': loaded_keys,
        'n_missing': len(missing_keys),
        'missing_keys': missing_keys,
        'n_unexpected': len(unexpected_keys),
        'unexpected_keys': unexpected_keys,
    }


def print_checkpoint_report(report):
    print(f"[init_from_checkpoint] loaded {report['n_loaded']} tensors from {report['checkpoint_path']}")
    print(f"[init_from_checkpoint] dropped (unexpected_keys, in checkpoint not in model): {report['n_unexpected']}")
    for k in report['unexpected_keys']:
        print(f'    dropped: {k}')
    print(f"[init_from_checkpoint] randomly-initialized (missing_keys, in model not in checkpoint): {report['n_missing']}")
    for k in report['missing_keys']:
        print(f'    randomly-initialized: {k}')


def validate_checkpoint_report(report, allowed_dropped_prefixes=()):
    """Hard stop if any backbone parameter is missing or unexpectedly dropped.

    missing_keys must always be empty: the patch checkpoint is a full model
    instance, so a fine-tune config should never need to randomly-initialize
    a parameter -- it can only ever remove a head. unexpected_keys (dropped)
    are only acceptable when every one of them starts with an allowed prefix
    (e.g. 'current_decoder' when predict_surface_current=False); anything
    else means a backbone/architecture mismatch and must stop the run.
    """
    if report['n_missing'] > 0:
        raise RuntimeError(
            'init_from_checkpoint: found randomly-initialized (missing) parameters, '
            f"expected none: {report['missing_keys']}"
        )
    offending = [k for k in report['unexpected_keys']
                 if not any(k.startswith(p) for p in allowed_dropped_prefixes)]
    if offending:
        raise RuntimeError(
            'init_from_checkpoint: dropped parameters outside the allowed prefixes '
            f'{allowed_dropped_prefixes} -- this looks like a backbone/architecture '
            f'mismatch, not an expected head removal: {offending}'
        )


def print_run_identity(config, checkpoint_path=None, log_path=None):
    lines = ['=== Run identity ===']
    if checkpoint_path:
        lines.append(f'checkpoint_path: {checkpoint_path}')
        lines.append(f'checkpoint_sha256: {sha256_of_file(checkpoint_path)}')
    lines.append(f'dataset_root: {config.get("wire_data_root") or config.get("path_to_root_data_folder")}')
    lines.append(f'split_manifest: {config.get("split_manifest")}')
    lines.append(f'predict_surface_current (PAIS): {config.get("predict_surface_current")}')
    text = '\n'.join(lines)
    print(text)
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'a') as f:
            f.write(text + '\n')
    return text
