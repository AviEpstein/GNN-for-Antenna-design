"""Synthetic smoke test: run every paper model + loss for two optimizer steps
on a tiny random batch (no real data, no CST). Usage, from the repo root:

    python -m tests.smoke_test            # all checks
    python -m tests.smoke_test forward    # graph models only
    python -m tests.smoke_test baseline   # grid baselines only
    python -m tests.smoke_test diffusion  # DDPM step + a tiny CFG sampling pass

This verifies wiring (imports, config plumbing, loss branching), not accuracy.
"""

import sys

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from configs.parser import _load_yaml

H, W = 34, 34
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_row(row):
    cfg = _load_yaml(f'configs/forward/{row}.yaml')
    base = _load_yaml('configs/base_forward.yaml')
    base.update({k: v for k, v in cfg.items() if k != 'base_config'})
    return base


def synthetic_graph(num_nodes=60, num_sc=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    pos = torch.rand(num_nodes, 3, generator=g) * 50.0
    normals = torch.randn(num_nodes, 3, generator=g)
    normals = normals / normals.norm(dim=-1, keepdim=True)
    node_type = torch.zeros(num_nodes, 10)
    node_type[torch.arange(num_nodes), torch.randint(0, 10, (num_nodes,), generator=g)] = 1.0
    x = torch.cat([pos, normals, node_type], dim=-1)
    num_edges = num_nodes * 4
    edge_index = torch.randint(0, num_nodes, (2, num_edges), generator=g)
    rel = pos[edge_index[1]] - pos[edge_index[0]]
    dist = rel.norm(dim=-1, keepdim=True)
    edge_attr = torch.cat([dist, rel / dist.clamp(min=1e-9)], dim=-1)
    data = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr,
                pe=torch.randn(num_nodes, 10, generator=g),
                node_normals=normals, node_type=node_type,
                surface_currents=torch.randn(num_sc, 6, generator=g),
                pos_surface_current=torch.rand(num_sc, 3, generator=g) * 50.0,
                node_probs=torch.rand(num_nodes, 1, generator=g),
                freq_hz=torch.tensor([5.6e9]))
    return data


def make_batch(batch_size=2):
    loader = DataLoader([synthetic_graph(seed=i) for i in range(batch_size)],
                        batch_size=batch_size, follow_batch=['pos_surface_current'])
    return next(iter(loader))


def run_forward_row(row):
    from src.models.registry import build_graph_model
    from src.losses.losses import FarfeildLoss
    from src.losses.surface_current_loss import (FFAndSurfaceCurrentAndPhysicsLoss,
                                                 FFAndSurfaceCurrentLoss)
    config = load_row(row)
    # smoke runs single-frequency without the S11 head; the flags below only
    # narrow the exercised path for the best-config row
    config.update({'multipule_freqs': False, 'use_freq_conditioning': False,
                   'pred_s11_single_freq': False})
    model = build_graph_model(config, DEVICE).to(DEVICE)
    ff_stats = {'min': 0.0, 'max': 4.0, 'mean': 0.5, 'std': 0.5}
    if config.get('predict_surface_current'):
        if config.get('use_physics_loss'):
            loss_fn = FFAndSurfaceCurrentAndPhysicsLoss(
                config, ff_loss_weight=config['ff_loss_weight'],
                surface_current_loss_weight=config['surface_current_loss_weight'],
                device=DEVICE)
        else:
            loss_fn = FFAndSurfaceCurrentLoss(
                config, ff_loss_weight=config['ff_loss_weight'],
                surface_current_loss_weight=config['surface_current_loss_weight'])
    else:
        loss_fn = FarfeildLoss(config, ff_stats)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    for step in range(2):
        batch = make_batch().to(DEVICE)
        ff_label = torch.rand(2, 1, H, W, device=DEVICE) * 4.0
        opt.zero_grad()
        pred = model(batch, radiation_image_shape=config['radiation_image_shape'])
        if config.get('predict_surface_current'):
            ff_pred = pred['radiation_image'].permute(0, 3, 1, 2)
            sc_pred = pred['surface_current']
            if config.get('use_physics_loss'):
                ff_phys = pred['radiation_image_physics'].unsqueeze(1)
                loss, _, _ = loss_fn(batch, sc_pred, ff_pred, ff_phys, ff_label)
            else:
                loss, _, _ = loss_fn(batch, sc_pred, ff_pred, ff_label)
        elif config.get('position_auxiliary_task'):
            ff_pred = pred['radiation_image'].permute(0, 3, 1, 2)
            loss = torch.nn.functional.mse_loss(ff_pred, ff_label) \
                + 0.1 * torch.nn.functional.mse_loss(pred['positions'], batch.pos)
        else:
            out = pred['radiation_image'] if isinstance(pred, dict) else pred
            loss = loss_fn(out.permute(0, 3, 1, 2), ff_label)
        loss.backward()
        opt.step()
    return loss.item()


def run_baseline_row(row):
    from src.models.registry import build_baseline_model
    config = load_row(row)
    model = build_baseline_model(config).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    for step in range(2):
        x = torch.rand(2, 2, 16, 16, device=DEVICE)
        target = torch.rand(2, 1, H, W, device=DEVICE)
        opt.zero_grad()
        if row == 'mlp':
            out = model(x.reshape(2, -1))
        else:
            out = model(x)
        out = out.reshape(2, 1, H, W)
        loss = torch.nn.functional.mse_loss(out, target)
        loss.backward()
        opt.step()
    return loss.item()


def run_diffusion():
    from src.diffusion.ddpm import DDPM
    from src.diffusion.unet_attention import ConditionalDenoisingUNetSmallAttentionWithConcat
    config = {'T': 700}
    unet = ConditionalDenoisingUNetSmallAttentionWithConcat(
        config=config, in_channels=2, num_hiddens=128, ff_in_ch=1).to(DEVICE)
    ddpm = DDPM(unet, num_ts=config['T']).to(DEVICE)
    opt = torch.optim.Adam(ddpm.parameters(), lr=1e-4)
    for step in range(2):
        x = (torch.rand(4, 2, 16, 16, device=DEVICE) > 0.5).float()
        c = torch.rand(4, 1, H, W, device=DEVICE) * 4.0
        opt.zero_grad()
        loss = ddpm(x, c)
        loss.backward()
        opt.step()
    with torch.no_grad():
        samples = ddpm.sample(c=torch.rand(2, 1, H, W, device=DEVICE),
                              img_wh=(16, 16), guidance_scale=1.0, in_channels=2)
    assert samples.shape[-2:] == (16, 16)
    return loss.item()


FORWARD_ROWS = ['gps', 'gps_pais', 'gps_pos_aux', 'gps_shuffled_sc',
                'gps_pais_directional', 'gps_pais_dir_phys_s0', 'gps_pais_dir_phys',
                'gcn', 'gcn_pais', 'gat', 'gat_pais', 'graph_unet', 'graph_unet_pais',
                'dgcnn', 'dgcnn_pais', 'mgn', 'mgn_pais']
BASELINE_ROWS = ['mlp', 'unet', 'resnet50']


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    failures = []
    if what in ('all', 'forward'):
        for row in FORWARD_ROWS:
            try:
                loss = run_forward_row(row)
                print(f'forward  {row:24s} OK  (loss {loss:.3f})')
            except Exception as e:
                failures.append(row)
                print(f'forward  {row:24s} FAIL: {type(e).__name__}: {e}')
    if what in ('all', 'baseline'):
        for row in BASELINE_ROWS:
            try:
                loss = run_baseline_row(row)
                print(f'baseline {row:24s} OK  (loss {loss:.3f})')
            except Exception as e:
                failures.append(row)
                print(f'baseline {row:24s} FAIL: {type(e).__name__}: {e}')
    if what in ('all', 'diffusion'):
        try:
            loss = run_diffusion()
            print(f'diffusion ddpm+cfg_sample       OK  (loss {loss:.3f})')
        except Exception as e:
            failures.append('diffusion')
            print(f'diffusion                        FAIL: {type(e).__name__}: {e}')
    if failures:
        raise SystemExit(f'{len(failures)} smoke failures: {failures}')
    print('all smoke checks passed')


if __name__ == '__main__':
    main()
