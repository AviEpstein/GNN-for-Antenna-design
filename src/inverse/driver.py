"""
driver.py
---------
Runs the three continuous-search inverse-design baselines sequentially:

    1. GABaseline        – genetic algorithm, uniform [0,1] init
    2. SABaseline        – simulated annealing, uniform [0,1] init
    3. NNSeededBaseline  – SA warm-started from continuous NN occupancy

All three operate in continuous [0, 1] space (not pure binary) so that
argmax of the patch matrix gives a unique, meaningful feed location.
`smooth_binarize` inside the surrogate pipeline still produces a binary
geometry at evaluation time.

Each baseline:
  - uses the trained GPS+PAIS surrogate for fitness (500 evals / target)
  - evaluates on the 100 hardest PCA-split targets
  - saves STL meshes + farfield tensors to its own output directory, in
    the same layout as the CMA baseline so evaluate_cst.py works unchanged

Runtime paths come from configs/inverse/baselines.yaml:
  data_root, split_dir, surrogate_checkpoint, and
  output_dir (each baseline writes to its own subdirectory of it).
"""

import os
import torch
import torch_geometric.transforms as T

from torch_geometric.loader import DataLoader

from configs.parser import ArgParser
from src.dataset.datasets import dataset_hardest_pca
from src.models.gps import GPS

from src.inverse.base_baseline      import seed_everything
from src.inverse.ga_baseline        import GABaseline
from src.inverse.sa_baseline        import SABaseline
from src.inverse.nn_seeded_baseline import NNSeededBaseline


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def make_gps_model(config: dict, device: torch.device) -> GPS:
    """Instantiate a fresh GPS model (base class loads the weights)."""
    return GPS(
        config       = config,
        in_channels  = 16,
        channels     = 128,
        pe_dim       = 10,
        num_layers   = 10,
        attn_type    = 'multihead',
        attn_kwargs  = {'dropout': 0.3},
        out_channels = config['radiation_image_shape'][0] * config['radiation_image_shape'][1],
    ).to(device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    seed_everything(0)

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    arg_parser = ArgParser(config_file=os.path.join(_REPO_ROOT, 'configs', 'inverse', 'baselines.yaml'))
    config     = arg_parser.get_config()
    config.update({
        'batch_size':             128,
        'eval_batch_size':        1,
        'idx_freq':               3,
        'radiation_image_shape':  [34, 34, 1],
        'threshold':              0.5,
        'physical_antenna':       True,
        'use_node_probs':         False,
        'add_sphere':             False,
        'merge_duplicate_nodes':  True,
        'data_set_type':          'pixel_data_with_reflectors',
        'add_radius_graph_edges': True,
        'split_type':             'hardest_indices',
        'predict_surface_current': True,
    })
    for required_key in ('data_root', 'split_dir', 'surrogate_checkpoint', 'output_dir'):
        if config.get(required_key) is None:
            raise ValueError(f"config['{required_key}'] must be set in the yaml config file")

    OUT_ROOT = config['output_dir']
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    pe_transform = T.Compose([
        T.ToUndirected(),
        T.AddLaplacianEigenvectorPE(k=10, attr_name='pe'),
    ])

    combined_dataset, train_dataset, test_dataset = dataset_hardest_pca(
        config, N=100, pe_transform=pe_transform, dataset_name='FMNIST_CIFAR',
    )

    eval_loader = DataLoader(
        test_dataset, batch_size=1,
        follow_batch=['pos_surface_current'], shuffle=False,
    )
    num_cpus = os.cpu_count() or 4
    num_workers = min(8, max(1, num_cpus // 2))
    # shuffle=False is REQUIRED: NNSeededBaseline caches matrices in the
    # same iteration order as AntennaNearestNeighbor walks the loader.
    train_loader = DataLoader(
        train_dataset, batch_size=32,
        follow_batch=['pos_surface_current'], shuffle=False, num_workers=num_workers,
                                        pin_memory=True,
                                        prefetch_factor=2,
                                        drop_last=True

    )

    # ------------------------------------------------------------------
    # 1. Genetic algorithm  (50 individuals × 10 generations = 500 evals)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Baseline 1 / 3 : Genetic Algorithm (continuous search)")
    print("=" * 60)
    config['output_dir'] = os.path.join(
        OUT_ROOT, 'diffusion_output_MNIST_CIFAR_GA_baseline_hardest_100/')

    GABaseline(
        config           = config,
        simulation_model = make_gps_model(config, device),
        eval_budget      = 500,
        pop_size         = 50,
        mutation_rate    = 0.05,
        mutation_sigma   = 0.30,
        elitism          = 2,
        tournament_k     = 3,
    ).test(eval_loader)

    # ------------------------------------------------------------------
    # 2. Simulated annealing  (500 Gaussian-perturbation steps, random init)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Baseline 2 / 3 : Simulated Annealing (continuous, random init)")
    print("=" * 60)
    config['output_dir'] = os.path.join(
        OUT_ROOT, 'diffusion_output_MNIST_CIFAR_SA_baseline_hardest_100/')

    SABaseline(
        config           = config,
        simulation_model = make_gps_model(config, device),
        eval_budget      = 500,
        T0               = 0.10,
        T_end            = 1e-3,
        k_perturb        = 2,
        proposal_sigma   = 0.30,
    ).test(eval_loader)

    # ------------------------------------------------------------------
    # 3. NN-seeded SA  (warm-start from nearest-neighbour geometry)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Baseline 3 / 3 : NN-seeded Simulated Annealing")
    print("=" * 60)
    config['output_dir'] = os.path.join(
        OUT_ROOT, 'diffusion_output_MNIST_CIFAR_NNseed_SA_baseline_hardest_100/')

    NNSeededBaseline(
        config           = config,
        simulation_model = make_gps_model(config, device),
        train_loader     = train_loader,
        eval_budget      = 500,
        T0               = 0.05,        # cooler start: we're already in a good basin
        T_end            = 1e-3,
        k_perturb        = 1,           # finer-grained refinement
        proposal_sigma   = 0.30,
    ).test(eval_loader)

    print("\nAll baselines complete.")
