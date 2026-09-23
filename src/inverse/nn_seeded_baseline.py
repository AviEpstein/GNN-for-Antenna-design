"""
nn_seeded_baseline.py
---------------------
NN-seeded local search: retrieve the nearest-neighbour antenna from the
training set (matched by farfield MSE) and warm-start SA from its
*continuous* occupancy matrix.

Why continuous matters here too
-------------------------------
We cache the raw `ant_parameters` from the dataset (which are the
original FMNIST/CIFAR pixel intensities, naturally in [0, 1] with a
clear maximum) rather than binarising them. This:
  - gives a unique argmax → well-defined feed placement
  - preserves the NN antenna's own feed location at the start of SA
  - lets continuous SA perturbations explore around it without breaking
    the argmax-based feed convention

`NNSeededBaseline` inherits from `SABaseline`. It overrides `optimize`
only to retrieve and pass the seed; the SA loop, scoring, saving, and
test driver all live in the parent classes.

IMPORTANT: `train_loader` MUST use `shuffle=False`. The cached occupancy
matrices in `self.train_matrices` line up with `AntennaNearestNeighbor`'s
internal ordering only when both iterations of the loader produce items
in the same order.
"""

import torch
import torch.nn as nn

from src.inverse.nearest_neighbor import AntennaNearestNeighbor
from src.inverse.sa_baseline import SABaseline


class NNSeededBaseline(SABaseline):
    """
    Parameters
    ----------
    config           : dict
    simulation_model : GPS
    train_loader     : DataLoader  – MUST have shuffle=False
    eval_budget      : int          – SA steps after warm-start (default 500)
    T0, T_end, k_perturb, proposal_sigma
                                    – passed through to SABaseline
    """

    def __init__(
        self,
        config,
        simulation_model,
        train_loader,
        eval_budget:    int   = 500,
        T0:             float = 0.05,
        T_end:          float = 1e-3,
        k_perturb:      int   = 1,
        proposal_sigma: float = 0.3,
    ):
        super().__init__(
            config, simulation_model,
            eval_budget    = eval_budget,
            T0             = T0,
            T_end          = T_end,
            k_perturb      = k_perturb,
            proposal_sigma = proposal_sigma,
        )
        self.train_loader = train_loader
        self._build_nn_index()

    # ------------------------------------------------------------------
    # Build index once at construction
    # ------------------------------------------------------------------

    def _build_nn_index(self):
        print("[NNSeeded] Building NN index over training farfields...")
        self.nn = AntennaNearestNeighbor(
            self.train_loader,
            freq_idx                = self.config['idx_freq'],
            Nearest_neighbor_feture = 'calc_nn_from_farfeild',
            loss_function           = nn.MSELoss(),
        )
        np_train  = [t.numpy().flatten() for t in self.nn.train_ff_images]
        self.nbrs = self.nn.train_nearest_neighbor(np_train)

        # Cache CONTINUOUS occupancy matrices in the same iteration order
        # as the NN index. We do NOT binarise — argmax of the original
        # FMNIST/CIFAR intensities defines the feed location.
        print("[NNSeeded] Caching training occupancy matrices (continuous)...")
        self.train_matrices = []
        for batch in self.train_loader:
            _, _, _, _, params, _ = batch
            ant = params['ant_parameters']           # (B, 16, 16)
            if self.channels == 2:
                refl = params['reflector_matrix']    # (B, 16, 16)
                m    = torch.stack([ant[0], refl], dim=1)  # (B, 2, 16, 16)
            else:
                m = ant.unsqueeze(1)
            for i in range(m.size(0)):
                self.train_matrices.append(m[i].cpu())
        print(f"[NNSeeded] Cached {len(self.train_matrices)} training matrices.")

    # ------------------------------------------------------------------
    # NN retrieval
    # ------------------------------------------------------------------

    def _retrieve_nn_matrix(self, target_34: torch.Tensor) -> torch.Tensor:
        """
        Return the (continuous) occupancy matrix of the training antenna
        whose farfield is closest (L2) to target_34.
        """
        target_flat   = target_34[0].cpu().numpy().reshape(1, -1)
        _, positions  = self.nbrs.kneighbors(target_flat, n_neighbors=1)
        return self.train_matrices[int(positions[0][0])].clone()

    # ------------------------------------------------------------------
    # Optimize: warm-start SA from the (continuous) NN geometry
    # ------------------------------------------------------------------

    def optimize(self, target_34, init_matrix=None):
        seed = self._retrieve_nn_matrix(target_34)
        # Keep values in [0, 1] for the proposal/clip logic in SA.
        seed = seed.clamp(0.0, 1.0)
        return super().optimize(target_34, init_matrix=seed)
