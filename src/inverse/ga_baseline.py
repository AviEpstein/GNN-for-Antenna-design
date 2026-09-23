"""
ga_baseline.py
--------------
Genetic algorithm on the 2x16x16 occupancy lattice, operating in
*continuous* [0, 1] space rather than discrete {0, 1}.

Why continuous?
---------------
The feed column is placed at the argmax of the patch matrix. If the
search operated on pure {0, 1} bits, argmax would be ambiguous (any of
the many "1" pixels could win, decided arbitrarily by floating-point
ordering). Operating on continuous values guarantees a unique maximum
that meaningfully determines feed location, while `smooth_binarize`
inside `score_population` still produces near-binary geometry.

Default budget: population 50 x 10 generations = 500 surrogate evaluations.

Operators
---------
  Selection : tournament (k=3), min-loss wins
  Crossover : uniform (each gene independently from either parent)
  Mutation  : per-gene Gaussian perturbation with probability
              `mutation_rate`, clipped to [0, 1]
  Elitism   : top-`elitism` carried over unchanged
"""

import numpy as np
import torch

from src.inverse.base_baseline import DiscreteSearchBaseline


class GABaseline(DiscreteSearchBaseline):
    """
    Parameters
    ----------
    config           : dict
    simulation_model : GPS
    eval_budget      : int   – total surrogate evals (default 500)
    pop_size         : int   – individuals per generation (default 50)
    mutation_rate    : float – per-gene probability of being perturbed (default 0.05)
    mutation_sigma   : float – stddev of Gaussian perturbation, in [0,1] units (default 0.3)
    elitism          : int   – individuals copied unchanged each generation (default 2)
    tournament_k     : int   – candidates per tournament (default 3)
    """

    def __init__(
        self,
        config,
        simulation_model,
        eval_budget:    int   = 500,
        pop_size:       int   = 50,
        mutation_rate:  float = 0.05,
        mutation_sigma: float = 0.3,
        elitism:        int   = 2,
        tournament_k:   int   = 3,
    ):
        super().__init__(config, simulation_model, eval_budget)
        self.pop_size       = pop_size
        self.generations    = max(1, eval_budget // pop_size)
        self.mutation_rate  = mutation_rate
        self.mutation_sigma = mutation_sigma
        self.elitism        = elitism
        self.tournament_k   = tournament_k

    # ------------------------------------------------------------------
    # Genetic operators (continuous-valued)
    # ------------------------------------------------------------------

    def _random_population(self) -> torch.Tensor:
        """Uniform [0, 1] random initialisation – natural continuous prior."""
        return torch.rand(self.pop_size, self.channels, self.grid, self.grid)

    def _tournament(self, fitness: list) -> int:
        idxs = np.random.choice(len(fitness), self.tournament_k, replace=False)
        return int(idxs[np.argmin([fitness[i] for i in idxs])])

    def _crossover(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """Uniform crossover: each gene independently from p1 or p2."""
        mask = (torch.rand_like(p1) > 0.5).float()
        return mask * p1 + (1.0 - mask) * p2

    def _mutate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sparse Gaussian mutation.

        Each gene is perturbed with probability `mutation_rate`. Perturbed
        values receive additive N(0, mutation_sigma^2) noise, then the
        whole tensor is clipped to [0, 1].

        With sigma=0.3, perturbations can routinely cross the 0.5
        binarization threshold (~18% of single perturbations have
        |delta| > 0.4), so geometry can genuinely change while continuous
        values keep argmax unique for feed placement.
        """
        perturb_mask = (torch.rand_like(x) < self.mutation_rate).float()
        noise        = torch.randn_like(x) * self.mutation_sigma
        mutated      = x + perturb_mask * noise
        return mutated.clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def optimize(self, target_34, init_matrix=None):
        pop = self._random_population().to(self.device)

        # Optional warm-start: seed half the population around init_matrix
        if init_matrix is not None:
            init = init_matrix.to(self.device)
            if init.dim() == 3:
                init = init.unsqueeze(0)
            n_seed = self.pop_size // 2
            seeded = init.expand(n_seed, -1, -1, -1).clone()
            seeded = torch.stack([self._mutate(seeded[i]) for i in range(n_seed)])
            pop[:n_seed] = seeded

        best_loss   = float('inf')
        best_matrix = best_graph = best_pred = None

        for gen in range(self.generations):
            losses, preds, graphs, valid = self.score_population(pop, target_34)

            # Track global best across all generations
            for li, vi in enumerate(valid):
                if losses[li] < best_loss:
                    best_loss   = losses[li]
                    best_matrix = pop[vi].clone()
                    best_graph  = graphs[li]
                    best_pred   = preds[li]

            # Build fitness vector aligned to pop indices; invalid -> inf
            fitness = [float('inf')] * pop.size(0)
            for li, vi in enumerate(valid):
                fitness[vi] = losses[li]

            # Next generation: elites + tournament / crossover / mutation
            order   = np.argsort(fitness)
            new_pop = [pop[order[i]].clone() for i in range(self.elitism)]
            while len(new_pop) < self.pop_size:
                p1    = pop[self._tournament(fitness)]
                p2    = pop[self._tournament(fitness)]
                child = self._mutate(self._crossover(p1, p2))
                new_pop.append(child)
            pop = torch.stack(new_pop).to(self.device)

        if best_matrix is None:
            return None, None, None, None, None

        return best_matrix, best_loss, best_graph, best_pred, target_34[0].cpu()
