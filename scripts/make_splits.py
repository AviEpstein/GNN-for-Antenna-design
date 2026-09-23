"""(Re)generate dataset split files.

The exact split files used for every number in the paper ship in ``splits/``
— regenerating is only needed for new corpora.

Random 90/10 split (Sec. 3 'Splits'):
    python -m scripts.make_splits --config_file configs/base_forward.yaml \
        [--dataset_variant random|big|mixed] [--seed 0]

PCA extrapolation split (top-10% |PC1 projection| of flattened patterns):
    python -m src.dataset.create_extrapolation_split_pca --config_file configs/base_forward.yaml

N-hardest subsets of the PCA test split (Tables 2/3):
    python -m src.dataset.create_N_hardest_split --config_file configs/base_forward.yaml
"""

import torch
import torch_geometric.transforms as T

from configs.parser import ArgParser
from src.dataset.datasets import (big_dataset, create_random_split,
                                  get_big_dataset_subset, get_combined_dataset)

if __name__ == '__main__':
    config = ArgParser(config_file=None).get_config()
    pe_transform = T.Compose([
        T.ToUndirected(),
        T.AddLaplacianEigenvectorPE(k=10, attr_name='pe')
    ])
    torch.manual_seed(0)

    variant = config.get('dataset_variant', 'random')
    if variant == 'random':
        combined, _, _ = get_combined_dataset(config, pe_transform,
                                              dataset_name=config.get('data_set_name',
                                                                      'FMNIST_CIFAR'))
        name = config.get('data_set_name', 'FMNIST_CIFAR')
    elif variant == 'big':
        # build the union without requiring an existing split file
        from src.dataset.datasets import _all_five_corpora
        parts = _all_five_corpora(config, pe_transform)
        combined = torch.utils.data.ConcatDataset(list(parts))
        name = 'big_dataset_x3'
    elif variant == 'mixed':
        combined, _, _ = get_big_dataset_subset(config, pe_transform)
        print('mixed variant creates/refreshes its own split file; done')
        raise SystemExit(0)
    else:
        raise SystemExit(f'unsupported dataset_variant for make_splits: {variant}')

    create_random_split(combined, split_dir=config['split_dir'], dataset_name=name,
                        train_ratio=0.9, seed=config.get('seed', 0))
