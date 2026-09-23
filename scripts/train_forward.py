"""Train/evaluate a graph forward surrogate (Table 1 GNN rows).

Usage (from the repo root):
    python -m scripts.train_forward --config_file configs/forward/gps_pais.yaml
Any config key can be overridden on the command line, e.g.:
    python -m scripts.train_forward --config_file configs/forward/gps_pais.yaml --seed 42
"""

import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch_geometric.transforms as T
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch_geometric.loader import DataLoader

from configs.parser import ArgParser
from src.dataset.data_stats import calculate_farfeild_statistics
from src.dataset.datasets import get_dataset
from src.losses.losses import FarfeildLoss
from src.losses.surface_current_loss import (FFAndSurfaceCurrentAndPhysicsLoss,
                                             FFAndSurfaceCurrentLoss)
from src.models.registry import build_graph_model
from src.training.checkpoint_utils import (load_checkpoint_relaxed,
                                           print_checkpoint_report,
                                           print_run_identity,
                                           validate_checkpoint_report)
from src.training.forward_trainer import Trainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == '__main__':
    arg_parser = ArgParser(config_file=None)
    config = arg_parser.get_config()

    if config.get('preflight_only'):
        os.environ['WANDB_MODE'] = 'disabled'

    device_num = config.get('device_num', '0')
    os.environ["CUDA_VISIBLE_DEVICES"] = device_num
    cuda_device = "cuda:" + device_num
    device = torch.device(cuda_device if torch.cuda.is_available() else "cpu")

    # 1) dataset and loaders:
    pe_transform = T.Compose([
        T.ToUndirected(),
        T.AddLaplacianEigenvectorPE(k=10, attr_name='pe')
    ])

    torch.manual_seed(0)
    combined_dataset, train_dataset, test_dataset = get_dataset(config, pe_transform)

    seed = config.get('seed', 0)
    set_seed(seed)
    seed_worker(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    num_cpus = os.cpu_count() or 4
    num_workers = min(8, max(1, num_cpus // 2))
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              follow_batch=['pos_surface_current'],
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              persistent_workers=True,
                              prefetch_factor=2,
                              drop_last=True,
                              worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             follow_batch=['pos_surface_current'])

    # far-field statistics (computed once per dataset root, then cached as csv)
    ff_stats_csv = os.path.join(config.get('path_to_root_data_folder'),
                                'dataset_ff_stats.csv')
    if os.path.exists(ff_stats_csv):
        ff_stats = pd.read_csv(ff_stats_csv).iloc[0].to_dict()
    else:
        ff_stats = calculate_farfeild_statistics(train_loader)
        pd.DataFrame([ff_stats]).to_csv(ff_stats_csv, index=False)
    graph_stats = None

    # 2) model:
    model = build_graph_model(config, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in model: {total_params}")
    print('Model Name: ', model.__class__.__name__)

    init_checkpoint_path = config.get('init_from_checkpoint')
    if init_checkpoint_path:
        assert not config.get('load_trained_model') and not config.get('load_from_checkpoint'), \
            'init_from_checkpoint is a separate mechanism from load_trained_model/load_from_checkpoint; these must not be combined.'
        report = load_checkpoint_relaxed(model, init_checkpoint_path, device='cpu')
        print_checkpoint_report(report)
        allowed_dropped_prefixes = () if config.get('predict_surface_current') else ('current_decoder',)
        validate_checkpoint_report(report, allowed_dropped_prefixes=allowed_dropped_prefixes)
        log_path = None
        if config.get('checkpoints_path'):
            os.makedirs(config['checkpoints_path'], exist_ok=True)
            log_path = os.path.join(config['checkpoints_path'], 'run_identity.log')
        print_run_identity(config, checkpoint_path=init_checkpoint_path, log_path=log_path)

    if config.get('load_trained_model'):
        print('loading trained model from: ', config.get('trained_model_path'))
        model.load_state_dict(torch.load(config.get('trained_model_path')))

    # 3) loss function (Eq. 2 of the paper):
    if config.get('predict_surface_current'):
        if config.get('use_physics_loss'):
            loss_function = FFAndSurfaceCurrentAndPhysicsLoss(
                config,
                ff_loss_weight=config.get('ff_loss_weight'),
                surface_current_loss_weight=config.get('surface_current_loss_weight'),
                device=device)
        else:
            loss_function = FFAndSurfaceCurrentLoss(
                config,
                ff_loss_weight=config.get('ff_loss_weight'),
                surface_current_loss_weight=config.get('surface_current_loss_weight'))
    else:
        loss_function = FarfeildLoss(config, ff_stats)

    # 4) optimizer:
    base_lr = config.get('finetune_lr') if config.get('init_from_checkpoint') else config.get('base_lr', 0.0002)
    weight_decay = config.get('optimizer_weight_decay', 1e-5)
    warmup_epochs = config.get('warmup_epochs', 5)
    total_epochs = config.get('total_epochs', 300)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=base_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay
    )

    # 5) LR schedule: linear warmup to base_lr, then cosine annealing to 1e-6.
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.01,
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=(total_epochs - warmup_epochs),
        eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    # 6) trainer:
    trainer = Trainer(config, model, loss_function, optimizer=optimizer,
                      scheduler=scheduler, graph_stats=graph_stats, ff_stats=ff_stats)

    if config.get('preflight_only'):
        trainer.preflight(train_loader=train_loader, max_steps=config.get('max_steps', 5))
        print('done (preflight only)')
        sys.exit(0)

    if config['train']:
        trainer.train(train_loader=train_loader, test_loader=test_loader)
    if config['test']:
        print('Testing network')
        trainer.test_network(test_loader=test_loader)
        trainer.validate(test_loader=test_loader)
        trainer.wandb_run.log(trainer.wandb_log_dict)
    print('done')
