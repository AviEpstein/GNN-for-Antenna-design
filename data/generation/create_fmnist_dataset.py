#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from src.geometry.create_pixel_antenna import create_pixel_ant
from src.geometry.mesh_functions import plot_3d_points_edges, decompose_pyg_graph
from src.geometry.saving_functions import save_graph_dict_as_stl, save_matrix_and_dict


def get_dataset(name: str, root: Path, train: bool, imagenet_dir: Path | None):
    name = name.lower()
    common_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),        # ensure single channel
        transforms.Resize((16, 16), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),                              # -> [0,1]
        transforms.Lambda(lambda t: t.clamp(0.0, 1.0)),     # just to be explicit
    ])

    if name == "mnist":
        return datasets.MNIST(root=str(root), train=train, transform=common_tf, download=True)
    elif name in ["emnist", "e-mnist"]:
        # EMNIST has several splits; default to 'balanced' which includes digits+letters
        return datasets.EMNIST(root=str(root), split='balanced', train=train, transform=common_tf, download=True)
    elif name == "kmnist":
        return datasets.KMNIST(root=str(root), train=train, transform=common_tf, download=True)
    elif name in ["fashion", "fashion-mnist", "fashion_mnist"]:
        return datasets.FashionMNIST(root=str(root), train=train, transform=common_tf, download=True)
    elif name == "cifar10":
        return datasets.CIFAR10(root=str(root), train=train, transform=common_tf, download=True)
    elif name == "imagenet":
        if imagenet_dir is None:
            raise ValueError("For ImageNet, please pass --imagenet-dir pointing to the dataset root.")
        # For ImageNet, torchvision expects ImageFolder structure:
        # imagenet_dir/
        #   train/<class>/*.JPEG
        #   val/<class>/*.JPEG
        subdir = "train" if train else "val"
        data_dir = Path(imagenet_dir) / subdir
        if not data_dir.exists():
            raise ValueError(f"Expected {data_dir} to exist for ImageNet {subdir} split.")
        return datasets.ImageFolder(root=str(data_dir), transform=common_tf)
    else:
        raise ValueError(f"Unknown dataset: {name}. Choose from mnist, fashion-mnist, cifar10, imagenet.")

def process_batch(x: torch.Tensor, threshold: float = 0.5):
    """
    x: (B, 1, 16, 16) in [0,1]
    Returns:
      x_bin: thresholded (B, 1, 16, 16) in {0,1}
      max_xy: (B, 2) tensor with (x, y) coord of max value BEFORE threshold, 0-based
      max_vals: (B,) max values (for reference)
    """
    B, C, H, W = x.shape
    assert C == 1 and H == 16 and W == 16, f"Expected (B,1,16,16), got {x.shape}"

    # Threshold
    x_bin = (x >= threshold).to(x.dtype)

    # Max (pre-threshold)
    flat = x.view(B, -1)                # (B, H*W)
    max_vals, idx = flat.max(dim=1)     # (B,)
    y = idx // W
    x_coord = idx % W
    max_xy = torch.stack([x_coord, y], dim=1)  # (B, 2) -> (x, y)


    return x_bin, max_xy, max_vals


def create_antenna_FMNIST_dataset(dataloader, config, display=False):

    output_dir = config.get('output_dir', 'outputs/antenna_FMNIST/')
    all_images = []
    all_labels = []
    skipped_idxs = []

    for idx, (imgs, labels) in tqdm(enumerate(dataloader), total=len(dataloader)):
        x_bin, max_yx, max_vals = process_batch(imgs, threshold=config.get('threshold'))
        y,x = max_yx[:,1], max_yx[:,0] # x,y are swaped in the antenna coridnant system comaperd to the image
        # print(f"Max positions (x,y): {list(zip(x.tolist(), y.tolist()))}")
        
        env_dict, reflectors_dict = create_parameter_dict(config)

        matrix = imgs[0][0]
        if torch.all(x_bin == 0):
            print(f"All pixels are below threshold for idx {idx}, skipping.")
            skipped_idxs.append(idx)
            continue
        antenna, _ = create_pixel_ant(
                matrix,  # matrix is now always a Parameter with gradients
                threshold = config.get('threshold'),
                size_of_patch_in_mm=env_dict['patch_x'],
                size_of_FR4_in_mm=env_dict['ground_x'],
                size_of_ground=env_dict['ground_x'],
                height=env_dict['h'],
                reflectors_dict=reflectors_dict,
                create_physical_pixel_mesh=True
            )
        antenna_graph_dict = decompose_pyg_graph(antenna)

        
        paramter_dict = env_dict.copy()
        # paramter_dict.update(reflectors_dict)
        paramter_dict['matrix'] = matrix
        paramter_dict['class_label'] = labels[0].item()
        save_matrix_and_dict_path = os.path.join(output_dir, str(idx), 'matrix_and_env_dict.pkl')
        save_matrix_and_dict(matrix, paramter_dict, save_matrix_and_dict_path)

        output_dir_stl = output_dir + str(idx)
        save_graph_dict_as_stl(antenna_graph_dict, output_dir=output_dir_stl)



        if display:
            plot_3d_points_edges(antenna.pos.detach().cpu(), antenna.edge_index.T.detach().cpu())
            # Display the matrix with the (y, x) pixel highlighted
            import matplotlib.pyplot as plt

            matrix_np = matrix.detach().cpu().numpy()
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(matrix_np, cmap="gray", vmin=0, vmax=1)
            ax.scatter(x[0].item(), y[0].item(), color="red", label="Max Pixel")
            ax.legend()
            ax.set_title("Matrix with Highlighted Max Pixel")
            plt.show()

        # if idx>500:
        #     break

        

    # Save skipped indices to output directory
    skipped_idxs_path = os.path.join(output_dir, 'skipped_idxs.pkl')
    torch.save(skipped_idxs, skipped_idxs_path)
    print(f'create_antenna_FMNIST_dataset skipped indices: {len(skipped_idxs)}')
    return all_images, all_labels

def create_parameter_dict(config):

    env_dict = {
        'patch_x': config.get('patch_x', 28),
        'patch_y': config.get('patch_y', 28),
        'ground_x': config.get('ground_x', 50),
        'ground_y': config.get('ground_y', 50),
        'radius': config.get('radius', 75),
        'num_reflectors': config.get('num_reflectors', 2),
        'h': config.get('h', 4),
        'box_size': config.get('box_size', 50),
        'tan_d': config.get('tan_d', 0.0027),
        'eps_r': config.get('eps_r', 3.55)

    }
    reflectors_dict = None
    return env_dict, reflectors_dict

def args_to_dict(args):
    return vars(args)

def main():
    parser = argparse.ArgumentParser(description="Load dataset, resize to 16x16, normalize [0,1], threshold at 0.5, find max (x,y).")
    parser.add_argument("--dataset", type=str, default="fashion-mnist",
                        help="mnist | fashion-mnist | cifar10 | imagenet | emnist | kmnist")
    parser.add_argument("--root", type=str, default="./data/downloads", help="Where to store/download datasets")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "val"],
                        help="Split to use (ImageNet uses train/val; others use train/test)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", type=str, default="outputs/antenna_FMNIST/", help="Where to write the generated dataset")
    parser.add_argument("--imagenet-dir", type=str, default=None, help="Root of ImageNet (with train/ and val/)")
    parser.add_argument("--show",default=True, action="store_true", help="Print a small summary for one batch")
    args = parser.parse_args()

    root = Path(args.root)
    train_flag = args.split in ["train"]  # for MNIST/Fashion/CIFAR10
    if args.dataset.lower() == "imagenet":
        # ImageNet uses "val" instead of "test"
        train_flag = args.split == "train"

    ds = get_dataset(args.dataset, root, train_flag, Path(args.imagenet_dir) if args.imagenet_dir else None)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    config = args_to_dict(args)
    
    create_antenna_FMNIST_dataset(dl, config)

    # Take one batch to demonstrate
    imgs, labels = next(iter(dl))  # imgs: (B,1,16,16) after our transform
    x_bin, max_xy, max_vals = process_batch(imgs, threshold=args.threshold)

    # Example: print results for the batch
    print(f"Dataset: {args.dataset} | Split: {args.split} | Batch size: {imgs.size(0)}")
    print(f"Images shape (after transform): {tuple(imgs.shape)}  (dtype={imgs.dtype}, range=[{imgs.min().item():.3f},{imgs.max().item():.3f}])")
    print(f"Binary images shape: {tuple(x_bin.shape)}  (unique values: {torch.unique(x_bin).tolist()})")
    print("Top-5 examples in this batch:")
    B = imgs.size(0)
    import math
    import numpy as np

    # prepare numpy arrays on CPU
    imgs_np = imgs.detach().cpu().numpy()        # (B,1,16,16)
    thresh_np = x_bin.detach().cpu().numpy()     # (B,1,16,16)
    labels_list = labels.detach().cpu().tolist() if isinstance(labels, torch.Tensor) else list(labels)
    max_vals_list = max_vals.detach().cpu().tolist()


    # Print brief textual summary for the first few as before
    for i in range(B):
        x_y = tuple(max_xy[i].tolist())
        mv = max_vals_list[i]
        lbl = labels_list[i]
        print(f"  idx {i:>2}: label={lbl}, max_val={mv:.4f} at (x,y)={x_y}")

    # Display the full batch (all images) using matplotlib if available
    try:
        import matplotlib.pyplot as plt

        cols = min(8, B)
        rows = math.ceil(B / cols)

        # Originals grid
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
        axes = np.array(axes).reshape(-1)
        for i in range(rows * cols):
            ax = axes[i]
            if i < B:
                ax.imshow(imgs_np[i, 0], cmap="gray", vmin=0, vmax=1)
                ax.set_title(f"idx {i}\nlbl={labels_list[i]}\nmax={max_vals_list[i]:.3f}")
            ax.axis("off")
        fig.suptitle("Original (16x16) - full batch")
        plt.tight_layout()
        plt.show()

        # Thresholded grid
        fig2, axes2 = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.6))
        axes2 = np.array(axes2).reshape(-1)
        for i in range(rows * cols):
            ax = axes2[i]
            if i < B:
                ax.imshow(thresh_np[i, 0], cmap="gray", vmin=0, vmax=1)
                ax.set_title(f"idx {i}\nmax@{tuple(max_xy[i].tolist())}")
            ax.axis("off")
        fig2.suptitle(f"Thresholded @ {args.threshold} - full batch")
        plt.tight_layout()
        plt.show()

    except Exception:
        print("Could not display images with matplotlib (not available or no DISPLAY).")
        # Fallback: print compact numeric arrays for each image
        for i in range(B):
            print(f"\n--- idx {i} | label={labels_list[i]} | max@{tuple(max_xy[i].tolist())} val={max_vals[i].item():.4f} ---")
            print("Original (rounded 16x16):")
            print(np.array2string(np.round(imgs_np[i, 0], 2), max_line_width=200))
            print("Thresholded (0/1):")
            print(np.array2string(thresh_np[i, 0].astype(int), max_line_width=200))
    print("Done.")

if __name__ == "__main__":
    main()

