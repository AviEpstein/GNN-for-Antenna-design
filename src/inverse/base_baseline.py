"""
base_baseline.py
----------------
Shared infrastructure for the discrete inverse-design baselines.
  - MSELoss         : per-sample clamped MSE, matches the CMA baseline
  - seed_everything : global RNG seeding
  - DiscreteSearchBaseline : abstract base class that owns the surrogate,
      the pixel-to-graph pipeline, score_population, and the test loop.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

from src.geometry.create_pixel_antenna import add_reflector_to_graph, create_pixel_ant
from src.graph.GNN_functions import prepare_graph
from src.geometry.mesh_functions import decompose_pyg_graph
from src.geometry.saving_functions import save_graph_dict_as_stl
from src.diffusion.smooth_binarize import smooth_binarize
from src.diffusion.diffusion_utils import create_parameter_dict
from src.metrics.ff_metrics import Metrics


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class MSELoss(nn.Module):
    """Per-sample MSE with value clamping, matching the CMA baseline."""
    def forward(self, pred, target):
        pred   = torch.clamp(pred,   -15, 10)
        target = torch.clamp(target, -15, 10)
        return ((pred - target) ** 2).mean(dim=(1, 2))


def seed_everything(seed: int):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class DiscreteSearchBaseline:
    """
    Common infrastructure for GA / SA / NN-seeded baselines.

    Subclasses override optimize() only; everything else is shared.

    Parameters
    ----------
    config : dict
        Experiment configuration dict (from ArgParser).
    simulation_model : nn.Module
        Instantiated GPS model (weights loaded in __init__).
    eval_budget : int
        Total number of surrogate evaluations allowed per target (default 500).
    """

    def __init__(self, config: dict, simulation_model: nn.Module, eval_budget: int = 500):
        self.config      = config
        self.device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eval_budget = eval_budget
        self.ff_h        = 34
        self.ff_w        = 34
        self.channels    = 2 if config.get('data_set_type') == 'pixel_data_with_reflectors' else 1
        self.grid        = 16

        # Load surrogate (GPS+PAIS forward model)
        self.model = simulation_model
        state_path = config.get('surrogate_checkpoint')
        if state_path is None:
            raise ValueError("config['surrogate_checkpoint'] must point to the trained GPS+PAIS forward-model .pt")
        self.model.load_state_dict(torch.load(state_path))
        self.model.eval()
        self.model.to(self.device)

        self.ff_loss_fn = MSELoss()
        self.metrics    = Metrics(config)
        self.compute_pe = T.AddLaplacianEigenvectorPE(k=10, attr_name='pe')

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def get_batch_data(self, batch):
        """Extract occupancy matrices, GT farfield, class labels, raw_idx."""
        graph, s11, farfields, graph_identifier, example_parameters, raw_idx = batch
        data = example_parameters['ant_parameters'][0].unsqueeze(1)
        data = smooth_binarize(data, 300, 300)
        if self.channels == 2:
            refl = example_parameters['reflector_matrix'].unsqueeze(1)
            refl = smooth_binarize(refl, 300, 300)
            data = torch.cat([data, refl], dim=1)
        label        = farfields[:, self.config['idx_freq'], :, :]
        class_labels = example_parameters['class_label']
        return data, label, class_labels, raw_idx

    def _prepare_target(self, target):
        """Interpolate raw GT farfield to (1, 34, 34) on self.device."""
        gt = target.reshape(1, -1, self.config['radiation_image_shape'][1])
        gt = nn.functional.interpolate(
            gt.unsqueeze(0), size=(self.ff_h, self.ff_w),
            mode='bilinear', align_corners=False,
        ).squeeze(0)
        return gt.to(self.device)

    # ------------------------------------------------------------------
    # Surrogate scoring
    # ------------------------------------------------------------------

    def score_population(self, population: torch.Tensor, target_34: torch.Tensor):
        """
        Score a batch of binary candidates with the surrogate.

        Parameters
        ----------
        population : (B, C, 16, 16) float tensor in {0, 1}
        target_34  : (1, 34, 34) interpolated GT, on self.device

        Returns
        -------
        losses       : list[float]  – one entry per valid candidate
        preds        : list[Tensor] – (34, 34) CPU tensors
        graphs       : list[PyG Data]
        valid_indices: list[int]    – original population indices that succeeded
        """
        env_dict, reflectors_dict = create_parameter_dict(self.config)
        all_losses, all_preds, all_graphs, valid_idx = [], [], [], []

        B         = population.size(0)
        max_chunk = 32
        for start in range(0, B, max_chunk):
            end          = min(start + max_chunk, B)
            chunk_graphs = []
            chunk_valid  = []

            for i in range(start, end):
                patch_mat = smooth_binarize(population[i, 0], 300, 300)
                try:
                    antenna, _ = create_pixel_ant(
                        patch_mat,
                        threshold             = self.config.get('threshold'),
                        size_of_patch_in_mm   = env_dict['patch_x'],
                        size_of_FR4_in_mm     = env_dict['ground_x'],
                        size_of_ground        = env_dict['ground_x'],
                        height                = env_dict['h'],
                        reflectors_dict       = reflectors_dict,
                        create_physical_pixel_mesh = True,
                    )
                    if self.channels == 2:
                        refl    = smooth_binarize(population[i, 1], 300, 300)
                        antenna = add_reflector_to_graph(antenna, refl, env_dict)
                    antenna, _ = prepare_graph(antenna.to(self.device), self.config)
                    antenna    = self.compute_pe(antenna)
                    chunk_graphs.append(antenna)
                    chunk_valid.append(i)
                except Exception:
                    # Empty / degenerate mesh: skip, do not crash the search.
                    continue

            if not chunk_graphs:
                continue

            batch_g = Batch.from_data_list(chunk_graphs).to(self.device)
            with torch.no_grad():
                pred = self.model(batch_g, radiation_image_shape=self.config['radiation_image_shape'])
            if isinstance(pred, dict):
                pred = pred['radiation_image']
            pred        = pred.reshape(-1, self.ff_h, self.ff_w)
            gt_expanded = target_34.expand(pred.size(0), -1, -1)
            losses      = self.ff_loss_fn(pred, gt_expanded)

            all_losses.extend(losses.tolist())
            all_preds.extend([p.cpu() for p in pred])
            all_graphs.extend(chunk_graphs)
            valid_idx.extend(chunk_valid)

        return all_losses, all_preds, all_graphs, valid_idx

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def _plot_save(self, ant_matrix, pred_ff, gt_ff, raw_idx, loss, class_label, save_path):
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        if self.channels == 2:
            combined = torch.cat([ant_matrix[0], ant_matrix[1]], dim=1).cpu().numpy().squeeze()
            plt.imshow(combined, cmap='gray')
            a0      = ant_matrix[0].detach().cpu().squeeze()
            max_idx = torch.argmax(a0).item()
            r, c    = divmod(max_idx, a0.shape[-1])
            plt.scatter(c, r, c='red', s=30)
        else:
            plt.imshow(ant_matrix.cpu().squeeze(), cmap='gray')
        plt.title(f'Antenna (loss={loss:.4f})'); plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(pred_ff.reshape(self.ff_h, self.ff_w).cpu(), cmap='jet', vmin=0, vmax=10)
        plt.title('Predicted farfield'); plt.colorbar(); plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(gt_ff.reshape(self.ff_h, self.ff_w).cpu(), cmap='jet', vmin=0, vmax=10)
        plt.title('Ground-truth farfield'); plt.colorbar(); plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(
            save_path, f'farfield_idx_{raw_idx}_class_{class_label}.png'))
        plt.close()

    def save_output(self, graph, ant_matrix, gt_ant_matrix,
                    pred_ff, gt_ff, loss, class_label, raw_idx):
        """Save STL mesh, farfield tensors, and diagnostic plot."""
        os.makedirs(self.config['output_dir'], exist_ok=True)
        save_path = os.path.join(self.config['output_dir'], f'{raw_idx}')
        os.makedirs(save_path, exist_ok=True)

        antenna_graph_dict = decompose_pyg_graph(graph.to(ant_matrix.device))
        save_graph_dict_as_stl(antenna_graph_dict, output_dir=save_path)

        torch.save(pred_ff.cpu(),     os.path.join(save_path, 'pred_farfield.pt'))
        torch.save(gt_ff.cpu(),       os.path.join(save_path, 'gt_farfield.pt'))
        torch.save(ant_matrix.cpu(),  os.path.join(save_path, 'pred_antenna_matrix.pt'))
        if gt_ant_matrix is not None:
            torch.save(gt_ant_matrix.cpu(), os.path.join(save_path, 'gt_antenna_matrix.pt'))

        self._plot_save(ant_matrix, pred_ff, gt_ff, raw_idx, loss, class_label, save_path)

    # ------------------------------------------------------------------
    # Optimize – subclasses override this
    # ------------------------------------------------------------------

    def optimize(self, target_34: torch.Tensor, init_matrix=None):
        """
        Run the search for a single target farfield.

        Parameters
        ----------
        target_34   : (1, 34, 34) interpolated GT on self.device
        init_matrix : optional (C, 16, 16) seed geometry

        Returns
        -------
        best_matrix : (C, 16, 16) CPU tensor
        best_loss   : float
        best_graph  : PyG Data
        best_pred   : (34, 34) CPU tensor
        gt_ff       : (34, 34) CPU tensor
        (all None on complete failure)
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Test loop
    # ------------------------------------------------------------------

    def test(self, eval_loader: DataLoader):
        self.metrics.reset()
        min_losses = []

        for idx, batch in tqdm(enumerate(eval_loader), total=len(eval_loader),
                               desc=self.__class__.__name__):
            with torch.no_grad():
                gt_ant_matrix, eval_labels, class_labels, raw_idx = self.get_batch_data(batch)
                target_34 = self._prepare_target(eval_labels)

                best_matrix, best_loss, best_graph, best_pred, gt_ff = self.optimize(target_34)

                if best_loss is None:
                    print(f"[{self.__class__.__name__}] no valid samples for {raw_idx[0]}, skipping.")
                    continue

                self.metrics.update(
                    best_pred.unsqueeze(0).unsqueeze(0),
                    gt_ff.unsqueeze(0).unsqueeze(0),
                )
                min_losses.append(best_loss)

                self.save_output(
                    best_graph,
                    best_matrix,
                    gt_ant_matrix[0][0] if gt_ant_matrix.dim() >= 2 else None,
                    best_pred,
                    gt_ff,
                    best_loss,
                    class_labels[0] if len(class_labels) > 0 else 0,
                    f"{raw_idx[0]}",
                )

        computed = self.metrics.compute()
        print(f"\n=== {self.__class__.__name__} results ===")
        for k, v in computed.items():
            print(f"  {k.upper()}: {v:.4f}")
        if min_losses:
            print(f"  Mean surrogate loss : {np.mean(min_losses):.4f} ± {np.std(min_losses):.4f}")
            print(f"  Min  surrogate loss : {min(min_losses):.4f}")
        return computed
