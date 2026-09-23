"""Train the conditional diffusion model (Sec. 4.2).

    python -m scripts.train_diffusion --config_file configs/inverse/diffusion_train.yaml

Thin wrapper around src/diffusion/diffusion_trainer.py (which is also runnable
directly); the DDPM hyperparameters (T=700, betas 1e-4..0.02, p_uncond=0.1,
Adam lr 1e-4, batch 128, 1000 epochs) are pinned there.
"""

import runpy
import sys

if __name__ == '__main__':
    if not any(a.startswith('--config_file') for a in sys.argv[1:]):
        sys.argv += ['--config_file', 'configs/inverse/diffusion_train.yaml']
    runpy.run_module('src.diffusion.diffusion_trainer', run_name='__main__')
