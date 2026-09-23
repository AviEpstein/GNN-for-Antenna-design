"""Evaluate a trained forward model without training.

Thin wrapper around scripts/train_forward.py that forces evaluation mode
(train: false, test: true). The config must set trained_model_path (and
load_trained_model: true) or those can be passed as overrides:

    python -m scripts.evaluate_forward --config_file configs/forward/gps_pais_big_pca.yaml
    python -m scripts.evaluate_forward --config_file configs/forward/gps_pais.yaml \
        --load_trained_model true --trained_model_path checkpoints/my_run.pt
"""

import runpy
import sys

if __name__ == '__main__':
    sys.argv += ['--train', 'false', '--test', 'true']
    runpy.run_module('scripts.train_forward', run_name='__main__')
