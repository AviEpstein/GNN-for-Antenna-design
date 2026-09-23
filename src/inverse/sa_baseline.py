"""
sa_baseline.py
--------------
Simulated annealing on the 2x16x16 occupancy lattice, operating in
*continuous* [0, 1] space.

Why continuous?
---------------
The feed column is placed at the argmax of the patch matrix. Discrete
bit-flip proposals leave many tied "1" pixels, so the argmax — and
hence the feed location — would be decided arbitrarily by floating
point ordering. Continuous values give a unique maximum that
meaningfully encodes feed placement.

Proposal mechanism
------------------
At each step, pick `k_perturb` pixels uniformly at random and add
N(0, proposal_sigma^2) noise to each, then clip to [0, 1]. With
sigma=0.3 these perturbations can cross the 0.5 binarization threshold
often enough to actually change the geometry — they behave like
"soft bit flips" while preserving a unique argmax.

Acceptance is the standard Metropolis rule with geometric cooling
from T0 to T_end over the full eval_budget.

Also serves as the parent of NNSeededBaseline: subclasses call
super().optimize(target, init_matrix=...) to warm-start the same SA loop
from any starting geometry.
"""

import numpy as np
import torch

from src.inverse.base_baseline import DiscreteSearchBaseline


class SABaseline(DiscreteSearchBaseline):
    """
    Parameters
    ----------
    config           : dict
    simulation_model : GPS
    eval_budget      : int   – number of SA steps / surrogate evals (default 500)
    T0               : float – initial temperature                  (default 0.10)
    T_end            : float – final  temperature                   (default 1e-3)
    k_perturb        : int   – pixels perturbed per proposal        (default 2)
    proposal_sigma   : float – stddev of per-pixel Gaussian noise   (default 0.3)
    """

    def __init__(
        self,
        config,
        simulation_model,
        eval_budget:    int   = 500,
        T0:             float = 0.10,
        T_end:          float = 1e-3,
        k_perturb:      int   = 2,
        proposal_sigma: float = 0.3,
    ):
        super().__init__(config, simulation_model, eval_budget)
        self.T0             = T0
        self.T_end          = T_end
        self.k_perturb      = k_perturb
        self.proposal_sigma = proposal_sigma
        # Geometric cooling factor: T_k = T0 * alpha^k
        self.alpha          = (T_end / T0) ** (1.0 / max(1, eval_budget - 1))

    # ------------------------------------------------------------------

    def _propose(self, current: torch.Tensor) -> torch.Tensor:
        """
        Continuous proposal: perturb k_perturb pixels with Gaussian noise.

        current : (1, C, 16, 16) float in [0, 1]
        returns : (1, C, 16, 16) float in [0, 1], clipped
        """
        proposal = current.clone()
        flat     = proposal.view(-1)
        idxs     = torch.randperm(flat.numel(), device=flat.device)[:self.k_perturb]
        noise    = torch.randn(self.k_perturb, device=flat.device) * self.proposal_sigma
        flat[idxs] = (flat[idxs] + noise).clamp(0.0, 1.05)
        return flat.view_as(current)

    # ------------------------------------------------------------------

    def optimize(self, target_34, init_matrix=None):
        # Starting point
        if init_matrix is None:
            current = torch.rand(1, self.channels, self.grid, self.grid)
        else:
            current = init_matrix.clone()
            if current.dim() == 3:
                current = current.unsqueeze(0)
        current = current.to(self.device)

        # Score initial state
        losses, preds, graphs, valid = self.score_population(current, target_34)
        if not valid:
            return None, None, None, None, None

        cur_loss    = losses[0]
        best_loss   = cur_loss
        best_matrix = current[0].clone()
        best_graph  = graphs[0]
        best_pred   = preds[0]

        T = self.T0
        for step in range(1, self.eval_budget):
            proposal = self._propose(current)

            losses, preds, graphs, valid = self.score_population(proposal, target_34)
            if not valid:
                T *= self.alpha
                continue

            new_loss = losses[0]
            delta    = new_loss - cur_loss

            # Metropolis acceptance
            if delta < 0 or np.random.rand() < np.exp(-delta / max(T, 1e-12)):
                current  = proposal
                cur_loss = new_loss
                if new_loss < best_loss:
                    best_loss   = new_loss
                    best_matrix = current[0].clone()
                    best_graph  = graphs[0]
                    best_pred   = preds[0]

            T *= self.alpha

        return best_matrix, best_loss, best_graph, best_pred, target_34[0].cpu()
