"""Run one inverse-design method (Tables 2-3, Fig. 3).

    python -m scripts.run_inverse --config_file configs/inverse/<method>.yaml

The entry module is inferred from the config file name:

    diffusion.yaml           surrogate-filtered diffusion (Diff. + surr.)
    diffusion_random5.yaml   5 unfiltered diffusion samples (Diff. - 5 rand.)
    hand_designed.yaml       hand-designed targets, w=2 (Fig. 3)
    baselines.yaml           GA -> SA -> NN-seeded SA (500-eval budget each)
    cma_baseline.yaml        CMA-ES
    random_filtered_baseline.yaml  best-of-500 random training geometries
    nearest_neighbor.yaml    NN retrieval

Selected candidates/STLs are written under output_dir for CST validation;
aggregate validated results with scripts/evaluate_cst.py.
"""

import os
import runpy
import sys

_MODULES = {
    'diffusion': 'src.diffusion.diffusion_trainer',
    'diffusion_train': 'src.diffusion.diffusion_trainer',
    'diffusion_random5': 'src.diffusion.diffusion_trainer',
    'hand_designed': 'src.diffusion.diffusion_trainer',
    'baselines': 'src.inverse.driver',
    'cma_baseline': 'src.inverse.cma_baseline',
    'random_filtered_baseline': 'src.inverse.random_filtered_baseline',
    'nearest_neighbor': 'src.inverse.nearest_neighbor',
}

if __name__ == '__main__':
    config_file = None
    for i, a in enumerate(sys.argv):
        if a == '--config_file' and i + 1 < len(sys.argv):
            config_file = sys.argv[i + 1]
        elif a.startswith('--config_file='):
            config_file = a.split('=', 1)[1]
    if config_file is None:
        raise SystemExit(__doc__)
    name = os.path.splitext(os.path.basename(config_file))[0]
    module = _MODULES.get(name)
    if module is None:
        raise SystemExit(f"unknown inverse method config '{name}'; known: {sorted(_MODULES)}")
    runpy.run_module(module, run_name='__main__')
