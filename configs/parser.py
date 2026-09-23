"""YAML config loading with base-config merging and CLI overrides.

Every entry point loads one YAML from configs/ via::

    config = ArgParser(config_file='configs/forward/gps_pais.yaml').get_config()

Resolution order (highest wins): command line > row YAML > its `base_config` YAML.

A row YAML may name a base file to inherit from::

    base_config: configs/base_forward.yaml
    model_type: GPS

Any config key can be overridden from the command line as ``--key value``;
values are parsed with yaml.safe_load, so ``--use_physics_loss true`` and
``--residual_scale 0.0`` get proper types (this replaces the old
``type=bool`` argparse flags, which silently treated any string as True).
The string ``None`` in a YAML is normalized to Python None.
"""

import argparse
import os
import sys

import yaml


def _load_yaml(path):
    with open(path) as f:
        config = yaml.safe_load(f) or {}
    for k, v in config.items():
        if v == 'None':
            config[k] = None
    return config


class ArgParser:
    def __init__(self, config_file=None):
        self.parser = argparse.ArgumentParser(
            description='Config loader (yaml + --key value overrides)')
        self.parser.add_argument('--config_file', type=str, default=config_file,
                                 help='path to yaml config file')

    def get_config(self):
        args, unknown = self.parser.parse_known_args()
        if args.config_file is None:
            raise SystemExit('a --config_file yaml is required')

        config = _load_yaml(args.config_file)
        base_path = config.pop('base_config', None)
        if base_path:
            base = _load_yaml(self._resolve(base_path, args.config_file))
            base.update(config)
            config = base

        config.update(self._parse_overrides(unknown))
        return config

    @staticmethod
    def _resolve(base_path, row_path):
        """Resolve a (possibly relative) base_config path.

        Tried in order: as given (relative to CWD), relative to the row
        yaml's directory, relative to the row yaml's parent directory
        (so 'base_forward.yaml' works from configs/forward/*.yaml).
        """
        if os.path.isabs(base_path):
            return base_path
        row_dir = os.path.dirname(os.path.abspath(row_path))
        for candidate in (base_path,
                          os.path.join(row_dir, base_path),
                          os.path.join(os.path.dirname(row_dir), base_path)):
            if os.path.exists(candidate):
                return candidate
        raise SystemExit(f'base_config not found: {base_path}')

    @staticmethod
    def _parse_overrides(argv):
        overrides = {}
        key = None
        for token in argv:
            if token.startswith('--'):
                if key is not None:
                    overrides[key] = True  # bare flag
                if '=' in token:
                    k, v = token[2:].split('=', 1)
                    overrides[k] = yaml.safe_load(v)
                    key = None
                else:
                    key = token[2:]
            else:
                if key is None:
                    raise SystemExit(f'unexpected argument: {token}')
                overrides[key] = yaml.safe_load(token)
                key = None
        if key is not None:
            overrides[key] = True
        for k, v in overrides.items():
            if v == 'None':
                overrides[k] = None
        return overrides


if __name__ == '__main__':
    print(ArgParser().get_config())
