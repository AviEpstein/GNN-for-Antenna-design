"""Dataset registry for the paper's corpora and splits.

All corpora live under ``config['data_root']`` using these default subfolder
names (each holds the ``raw/meshes``, ``raw/CST_results``, ``processed`` layout
described in data/README.md):

    fmnist_cifar_train/   FMNIST patch + CIFAR parasitic, training portion
    fmnist_cifar_test/    FMNIST patch + CIFAR parasitic, held-out portion
    fmnist_train/         FMNIST-only, training portion
    fmnist_test/          FMNIST-only, held-out portion
    random_pixel/         i.i.d. random binary patches
    classic_patch_no_reflector/    64 canonical square patches (zero-shot)
    classic_patch_with_reflector/  64 square patches + parasitic (zero-shot)
    classic_rectangle_patch/       108 canonical rectangular patches (zero-shot)

Any of them can be overridden individually with a ``<name>_root`` config key
(e.g. ``fmnist_cifar_train_root``).  Split index files (.pth) are read from
``config['split_dir']`` — the ones used for the paper ship in ``splits/``.

Selection for training/eval scripts goes through :func:`get_dataset`, keyed by
``config['dataset_variant']``; the individual builders below are unchanged
from the code that produced the paper's tables (only the hardcoded absolute
paths were replaced by config lookups).
"""

import os

import torch

from src.dataset.pixel_dataloader import PixelDataset

_DEFAULT_SUBDIRS = {
    'fmnist_cifar_train': 'fmnist_cifar_train/',
    'fmnist_cifar_test': 'fmnist_cifar_test/',
    'fmnist_train': 'fmnist_train/',
    'fmnist_test': 'fmnist_test/',
    'random_pixel': 'random_pixel/',
    'classic_patch_no_reflector': 'classic_patch_no_reflector/',
    'classic_patch_with_reflector': 'classic_patch_with_reflector/',
    'classic_rectangle_patch': 'classic_rectangle_patch/',
}


def corpus_root(config, name):
    """Directory of one corpus: explicit ``<name>_root`` key, else data_root/<name>/."""
    explicit = config.get(f'{name}_root')
    if explicit:
        return explicit
    data_root = config.get('data_root')
    if not data_root:
        raise KeyError(
            f"config must set 'data_root' (or '{name}_root') to locate the '{name}' corpus")
    return os.path.join(data_root, _DEFAULT_SUBDIRS[name])


def _corpus_pair(config, pe_transform, dataset_name):
    """The train/test PixelDataset pair for 'FMNIST_CIFAR' or 'FMNIST'."""
    if dataset_name == 'FMNIST_CIFAR':
        train_root = corpus_root(config, 'fmnist_cifar_train')
        test_root = corpus_root(config, 'fmnist_cifar_test')
    elif dataset_name == 'FMNIST':
        train_root = corpus_root(config, 'fmnist_train')
        test_root = corpus_root(config, 'fmnist_test')
    else:
        raise ValueError(f'unknown dataset_name: {dataset_name}')
    train_dataset = PixelDataset(train_root, config=config, pre_transform=pe_transform)
    test_dataset = PixelDataset(test_root, config=config, pre_transform=pe_transform)
    return train_dataset, test_dataset


def dataset_random(config, pe_transform, dataset_name='FMNIST_CIFAR'):
    """90/10 random split (Table 1 main block)."""
    torch.manual_seed(0)
    train_dataset, test_dataset = _corpus_pair(config, pe_transform, dataset_name)
    combined_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])

    path_to_random_split = config.get('path_to_split')
    if path_to_random_split and os.path.exists(path_to_random_split):
        print("Loading existing random split from: ", path_to_random_split)
        split = torch.load(path_to_random_split, weights_only=False)
        train_dataset = torch.utils.data.Subset(combined_dataset, split['train_idx'])
        test_dataset = torch.utils.data.Subset(combined_dataset, split['test_idx'])
        print(f"Combined dataset size: {len(combined_dataset)}, "
              f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    else:
        raise FileNotFoundError(
            f"random split file not found at config['path_to_split']={path_to_random_split}; "
            "generate one with scripts/make_splits.py or point at splits/random_split.pth")
    return combined_dataset, train_dataset, test_dataset


def dataset_pca(config, pe_transform, dataset_name='FMNIST_CIFAR'):
    """PCA extrapolation split (Table 1 'big-PCA split' evaluation)."""
    torch.manual_seed(0)
    train_dataset, test_dataset = _corpus_pair(config, pe_transform, dataset_name)
    combined_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])

    path_to_pca_split = os.path.join(config.get('split_dir'), 'pca_extrapolation_split.pth')
    if os.path.exists(path_to_pca_split):
        print("Loading existing PCA extrapolation split from: ", path_to_pca_split)
        split = torch.load(path_to_pca_split, weights_only=False)
        train_dataset = torch.utils.data.Subset(combined_dataset, split['train'])
        test_dataset = torch.utils.data.Subset(combined_dataset, split['test'])
        print(f"Combined dataset size: {len(combined_dataset)}, "
              f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    else:
        raise FileNotFoundError(
            f"PCA split not found at {path_to_pca_split}; "
            "generate it with src/dataset/create_extrapolation_split_pca.py")
    return combined_dataset, train_dataset, test_dataset


def dataset_hardest_pca(config, pe_transform, N=100, dataset_name='FMNIST_CIFAR'):
    """The N hardest PCA-split test targets (Tables 2 and 3)."""
    torch.manual_seed(0)
    combined_dataset, train_dataset, test_dataset = dataset_pca(
        config, pe_transform, dataset_name=dataset_name)

    path_to_hardest_indices = os.path.join(config.get('split_dir'), f'hardest_indices_{N}.pth')
    if os.path.exists(path_to_hardest_indices):
        print("Loading existing hardest indices split from: ", path_to_hardest_indices)
        hardest_indices = torch.load(path_to_hardest_indices, weights_only=False)
        test_dataset = torch.utils.data.Subset(test_dataset, hardest_indices)
        print(f"Test size after applying hardest indices split: {len(test_dataset)}")
    else:
        raise FileNotFoundError(
            f"hardest-indices file not found at {path_to_hardest_indices}; "
            "generate it with src/dataset/create_N_hardest_split.py")
    return combined_dataset, train_dataset, test_dataset


def dataset_random_easy(config, pe_transform, N=100, dataset_name='FMNIST_CIFAR'):
    """First N random-split test samples (an 'easy' control set)."""
    torch.manual_seed(0)
    train_dataset, test_dataset = _corpus_pair(config, pe_transform, dataset_name)
    combined_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
    path_to_random_split = os.path.join(config.get('split_dir'), 'random_split.pth')
    if os.path.exists(path_to_random_split):
        print("Loading existing random split from: ", path_to_random_split)
        split = torch.load(path_to_random_split, weights_only=False)
        train_dataset = torch.utils.data.Subset(combined_dataset, split['train_idx'])
        test_dataset = torch.utils.data.Subset(combined_dataset, split['test_idx'])
        print(f"Combined dataset size: {len(combined_dataset)}, "
              f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    easy_indices = list(range(N))
    test_dataset = torch.utils.data.Subset(test_dataset, easy_indices)
    print(f"Test size after applying easy indices split: {len(test_dataset)}")
    return combined_dataset, train_dataset, test_dataset


def _all_five_corpora(config, pe_transform):
    train_fc = PixelDataset(corpus_root(config, 'fmnist_cifar_train'), config=config,
                            pre_transform=pe_transform)
    train_f = PixelDataset(corpus_root(config, 'fmnist_train'), config=config,
                           pre_transform=pe_transform)
    test_fc = PixelDataset(corpus_root(config, 'fmnist_cifar_test'), config=config,
                           pre_transform=pe_transform)
    test_f = PixelDataset(corpus_root(config, 'fmnist_test'), config=config,
                          pre_transform=pe_transform)
    random_pixel = PixelDataset(corpus_root(config, 'random_pixel'), config=config,
                                pre_transform=pe_transform)
    return train_fc, train_f, test_fc, test_f, random_pixel


def big_dataset(config, pe_transform):
    """Union of all three sub-corpora ('big' rows of Table 1).

    Concatenation order matters: the shipped split file
    splits/random_split_big_dataset_x3.pth indexes into exactly this order.
    """
    torch.manual_seed(0)
    train_fc, train_f, test_fc, test_f, random_pixel = _all_five_corpora(config, pe_transform)
    combined_dataset = torch.utils.data.ConcatDataset(
        [train_fc, train_f, test_fc, test_f, random_pixel])

    path_to_random_split = os.path.join(config.get('split_dir'),
                                        'random_split_big_dataset_x3.pth')
    if os.path.exists(path_to_random_split):
        print("Loading existing random split from: ", path_to_random_split)
        split = torch.load(path_to_random_split, weights_only=False)
        train_dataset = torch.utils.data.Subset(combined_dataset, split['train_idx'])
        test_dataset = torch.utils.data.Subset(combined_dataset, split['test_idx'])
        print(f"Combined dataset size: {len(combined_dataset)}, "
              f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    else:
        raise FileNotFoundError(
            f"big-dataset split not found at {path_to_random_split}; "
            "copy splits/random_split_big_dataset_x3.pth into split_dir or "
            "generate a new split with scripts/make_splits.py")
    return combined_dataset, train_dataset, test_dataset


def get_big_dataset_subset(config, pe_transform):
    """Stratified matched-size mixture of the three sub-corpora
    (Table 1 'GPS + PAIS (mixed, matched size)': 4/9 of each pool)."""
    seed = 0
    torch.manual_seed(seed)
    train_fc, train_f, test_fc, test_f, random_pixel = _all_five_corpora(config, pe_transform)

    fmnist_only_pool = torch.utils.data.ConcatDataset([train_f, test_f])
    fmnist_cifar_pool = torch.utils.data.ConcatDataset([train_fc, test_fc])
    random_pixel_pool = random_pixel

    def stratified_subset(pool, n):
        n = min(n, len(pool))
        indices = torch.randperm(len(pool))[:n]
        return torch.utils.data.Subset(pool, indices)

    fmnist_only_subset = stratified_subset(fmnist_only_pool, len(fmnist_only_pool) * 4 // 9)
    fmnist_cifar_subset = stratified_subset(fmnist_cifar_pool, len(fmnist_cifar_pool) * 4 // 9)
    random_pixel_subset = stratified_subset(random_pixel_pool, len(random_pixel_pool) * 4 // 9)
    print(f"FMNIST-only subset size: {len(fmnist_only_subset)}, "
          f"FMNIST+CIFAR subset size: {len(fmnist_cifar_subset)}, "
          f"Random-pixel subset size: {len(random_pixel_subset)}")

    combined_dataset = torch.utils.data.ConcatDataset(
        [fmnist_only_subset, fmnist_cifar_subset, random_pixel_subset])

    path_to_random_split = os.path.join(config.get('split_dir'),
                                        f'random_split_big_dataset_x3_subset_{seed}.pth')
    split = None
    if os.path.exists(path_to_random_split):
        print("Loading existing random split from: ", path_to_random_split)
        split = torch.load(path_to_random_split, weights_only=False)
        if len(split['train_idx']) + len(split['test_idx']) != len(combined_dataset):
            print(f"Cached split at {path_to_random_split} covers "
                  f"{len(split['train_idx']) + len(split['test_idx'])} samples "
                  f"but combined_dataset now has {len(combined_dataset)}. "
                  "Treating cache as stale and regenerating.")
            split = None
    if split is None:
        print('creating a new random split and saving it to: ', path_to_random_split)
        split = create_random_split(combined_dataset, split_dir=config.get('split_dir'),
                                    dataset_name='big_dataset_x3_subset',
                                    train_ratio=0.9, seed=0)
        split = torch.load(path_to_random_split, weights_only=False)

    train_dataset = torch.utils.data.Subset(combined_dataset, split['train_idx'])
    test_dataset = torch.utils.data.Subset(combined_dataset, split['test_idx'])
    print(f"Combined dataset size: {len(combined_dataset)}, "
          f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    return combined_dataset, train_dataset, test_dataset


def get_combined_dataset(config, pe_transform, dataset_name='FMNIST_CIFAR'):
    torch.manual_seed(0)
    train_dataset, test_dataset = _corpus_pair(config, pe_transform, dataset_name)
    combined_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
    return combined_dataset, train_dataset, test_dataset


def get_FMNIST_dataset(config, pe_transform):
    train_dataset, test_dataset = _corpus_pair(config, pe_transform, 'FMNIST')
    return torch.utils.data.ConcatDataset([train_dataset, test_dataset])


def get_random_pixel_dataset(config, pe_transform):
    return PixelDataset(corpus_root(config, 'random_pixel'), config=config,
                        pre_transform=pe_transform)


def get_classic_patch_dataset(config, pe_transform, variant='classic_patch_with_reflector'):
    """Zero-shot canonical patch sets (Table 1 lower block, Fig. 2).

    variant: classic_patch_no_reflector | classic_patch_with_reflector |
             classic_rectangle_patch
    """
    return PixelDataset(corpus_root(config, variant), config=config,
                        pre_transform=pe_transform)


def get_CST_retrained_dataset(path_to_dataset, config, pe_transform):
    return PixelDataset(path_to_dataset, config=config, pre_transform=pe_transform)


def combine_datasets(dataset1, dataset2):
    return torch.utils.data.ConcatDataset([dataset1, dataset2])


def create_random_split(dataset, split_dir, dataset_name='FMNIST_CIFAR',
                        train_ratio=0.9, seed=0):
    torch.manual_seed(seed)
    num_total_samples = len(dataset)
    num_train_samples = int(train_ratio * num_total_samples)
    shuffled_indices = torch.randperm(num_total_samples)
    split = {
        'train_idx': shuffled_indices[:num_train_samples],
        'test_idx': shuffled_indices[num_train_samples:],
    }
    path_to_split = os.path.join(split_dir, f'random_split_{dataset_name}_{seed}.pth')
    os.makedirs(split_dir, exist_ok=True)
    if os.path.exists(path_to_split):
        print(f"Random split already exists at: {path_to_split}. Overwriting it.")
    torch.save(split, path_to_split)
    print("Saved new random split to: ", path_to_split)
    return split


_VARIANTS = {}


def get_dataset(config, pe_transform):
    """Dispatch on config['dataset_variant'] → (combined, train, test).

    Variants: random (default) | pca | hardest_pca_100 | hardest_pca_500 |
    random_easy | big | mixed | classic_square | classic_square_parasitic |
    classic_rectangle.  The classic_* variants are evaluation-only: the same
    canonical-patch dataset is returned in all three slots.
    """
    variant = config.get('dataset_variant', 'random')
    name = config.get('data_set_name', 'FMNIST_CIFAR')
    if variant == 'random':
        return dataset_random(config, pe_transform, dataset_name=name)
    if variant == 'pca':
        return dataset_pca(config, pe_transform, dataset_name=name)
    if variant == 'hardest_pca_100':
        return dataset_hardest_pca(config, pe_transform, N=100, dataset_name=name)
    if variant == 'hardest_pca_500':
        return dataset_hardest_pca(config, pe_transform, N=500, dataset_name=name)
    if variant == 'random_easy':
        return dataset_random_easy(config, pe_transform, dataset_name=name)
    if variant == 'big':
        return big_dataset(config, pe_transform)
    if variant == 'mixed':
        return get_big_dataset_subset(config, pe_transform)
    if variant in ('classic_square', 'classic_square_parasitic', 'classic_rectangle'):
        corpus = {'classic_square': 'classic_patch_no_reflector',
                  'classic_square_parasitic': 'classic_patch_with_reflector',
                  'classic_rectangle': 'classic_rectangle_patch'}[variant]
        dataset = get_classic_patch_dataset(config, pe_transform, variant=corpus)
        return dataset, dataset, dataset
    raise ValueError(f'unknown dataset_variant: {variant}')
