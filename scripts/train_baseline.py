"""Train/evaluate a grid-baseline forward model (Table 1 MLP / U-Net / ResNet-50 rows).

Usage (from the repo root):
    python -m scripts.train_baseline --config_file configs/forward/mlp.yaml
"""

import os

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
from src.models.registry import build_baseline_model
from src.training.baseline_trainer import Trainer

if __name__ == '__main__':
    arg_parser = ArgParser(config_file=None)
    config = arg_parser.get_config()

    cuda_device = "cuda:" + config.get('device_num', '0')
    device = torch.device(cuda_device if torch.cuda.is_available() else "cpu")

    # 1) dataset and loaders:
    pe_transform = T.Compose([
        T.ToUndirected(),
        T.AddLaplacianEigenvectorPE(k=10, attr_name='pe')
    ])

    torch.manual_seed(0)
    combined_dataset, train_dataset, test_dataset = get_dataset(config, pe_transform)

    num_cpus = os.cpu_count() or 4
    num_workers = min(8, max(1, num_cpus // 2))
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              follow_batch=['pos_surface_current'],
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True,
                              persistent_workers=True,
                              prefetch_factor=2,
                              drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             follow_batch=['pos_surface_current'])

    ff_stats_csv = os.path.join(config.get('path_to_root_data_folder'),
                                'dataset_ff_stats.csv')
    if os.path.exists(ff_stats_csv):
        ff_stats = pd.read_csv(ff_stats_csv).iloc[0].to_dict()
    else:
        ff_stats = calculate_farfeild_statistics(train_loader)
        pd.DataFrame([ff_stats]).to_csv(ff_stats_csv, index=False)
    graph_stats = None

    # 2) model:
    model = build_baseline_model(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in model: {total_params}")
    print('Model Name: ', model.__class__.__name__)

    if config.get('load_trained_model'):
        print('loading trained model from: ', config.get('trained_model_path'))
        model.load_state_dict(torch.load(config.get('trained_model_path')))

    # 3) loss function:
    loss_function = FarfeildLoss(config, ff_stats)

    # 4) optimizer (identical recipe to the graph models):
    base_lr = config.get('base_lr', 0.0002)
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

    # 5) LR schedule: linear warmup, then cosine annealing to 1e-6.
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                                total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(total_epochs - warmup_epochs),
                                         eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                             milestones=[warmup_epochs])

    # 6) trainer:
    trainer = Trainer(config, model, loss_function, optimizer=optimizer,
                      scheduler=scheduler, graph_stats=graph_stats, ff_stats=ff_stats)

    if config['train']:
        trainer.train(train_loader=train_loader, test_loader=test_loader)
    if config['test']:
        print('Testing network')
        trainer.test_network(test_loader=test_loader)
    print('done')
