"""
Sanity-check the far-field radiation integral (src.physics.ff_from_surface_currents.compute_farfield_from_currents)
directly against CST ground truth, bypassing the model and the processed PyG dataset entirely.

For each raw example this reads:
    <root>/raw/CST_results/<idx>/surface current (f=<freq>) [1].pkl
    <root>/raw/CST_results/<idx>/farfield_<freq>.npy
computes the far-field from the raw surface currents via the analytic radiation integral, and
compares it (MSE) against the CST-simulated far-field.
"""
import argparse
import glob
import os
import pickle

import numpy as np
import torch

from src.dataset.dataloader_utils import farfeild_txt_to_np, resize_farfeild, normalize_gain
from src.graph.surface_current_functions import process_surface_current, process_surface_current_downsample
from src.physics.ff_from_surface_currents import compute_farfield_from_currents


def load_ground_truth_farfield(result_path, freq, image_shape, device):
    npy_path = os.path.join(result_path, f'farfield_{freq}.npy')
    txt_path = os.path.join(result_path, f'farfield_{freq}.txt')
    if os.path.exists(npy_path):
        farfeild = torch.tensor(np.load(npy_path), dtype=torch.float32, device=device)
    elif os.path.exists(txt_path):
        farfeild = torch.tensor(farfeild_txt_to_np(txt_path), dtype=torch.float32, device=device)
    else:
        raise FileNotFoundError(f'no farfield_{freq}.npy or .txt found in {result_path}')

    farfeild = resize_farfeild(farfeild, image_shape).unsqueeze(0)
    gt_gain = normalize_gain(farfeild[0][:, :, 0], farfeild[0][:, :, 1], img_shape=image_shape, device=device)
    return gt_gain


def load_surface_current(result_path, freq, downsample_factor=None):
    surface_current_path = os.path.join(result_path, f'surface current (f={freq}) [1].pkl')
    with open(surface_current_path, 'rb') as f:
        surface_current_data = pickle.load(f)
    if downsample_factor is not None and downsample_factor > 1:
        surface_current, pos_surface_current = process_surface_current_downsample(surface_current_data, downsample_factor=downsample_factor)
    else:
        surface_current, pos_surface_current, area = process_surface_current(surface_current_data)
    return surface_current, pos_surface_current, area


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=str, required=True,
                        help='Dataset root containing raw/CST_results/<idx>/ example directories.')
    parser.add_argument('--frequencies', type=int, nargs='+', default=[2400, 2800, 5200, 5600, 6000])
    parser.add_argument('--frequencies_hz', type=float, nargs='+', default=[2.4e9, 2.8e9, 5.2e9, 5.6e9, 6.0e9])
    parser.add_argument('--image_shape', type=int, nargs=2, default=[34, 34])
    parser.add_argument('--downsample_factor', type=int, default=None,
                         help='If set, use process_surface_current_downsample with this stride instead of full-resolution process_surface_current.')
    parser.add_argument('--num_examples', type=int, default=None, help='Limit the number of examples evaluated (default: all).')
    args = parser.parse_args()

    assert len(args.frequencies) == len(args.frequencies_hz), '--frequencies and --frequencies_hz must be the same length'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_shape = tuple(args.image_shape)

    cst_results_dir = os.path.join(args.root, 'raw', 'CST_results')
    example_dirs = [d for d in glob.glob(os.path.join(cst_results_dir, '*')) if os.path.isdir(d)]
    example_dirs.sort(key=lambda p: int(os.path.basename(p)) if os.path.basename(p).isdigit() else os.path.basename(p))
    if args.num_examples is not None:
        example_dirs = example_dirs[:args.num_examples]

    mse_loss_fn = torch.nn.MSELoss()
    per_freq_losses = {freq: [] for freq in args.frequencies}

    for example_dir in example_dirs:
        example_number = os.path.basename(example_dir)
        for freq, freq_hz in zip(args.frequencies, args.frequencies_hz):
            try:
                surface_current, pos_surface_current, area = load_surface_current(example_dir, freq, args.downsample_factor)
                gt_gain = load_ground_truth_farfield(example_dir, freq, image_shape, device)
            except FileNotFoundError:
                continue

            surface_data = {'pos': pos_surface_current.to(device), 'J': surface_current.to(device), 'area': area.to(device)}
            ff_from_current_integral = compute_farfield_from_currents(surface_data, freq_hz, image_shape=image_shape, device=device)

            loss = mse_loss_fn(ff_from_current_integral, gt_gain.to(device))
            per_freq_losses[freq].append(loss.item())
            print(f'example {example_number} | freq {freq} MHz | MSE = {loss.item():.4f}')

    print('\n=== Summary ===')
    all_losses = []
    for freq in args.frequencies:
        losses = per_freq_losses[freq]
        if losses:
            avg = sum(losses) / len(losses)
            print(f'freq {freq} MHz: avg MSE = {avg:.4f} over {len(losses)} examples')
            all_losses.extend(losses)
    if all_losses:
        print(f'Overall avg MSE = {sum(all_losses) / len(all_losses):.4f} over {len(all_losses)} (example, freq) pairs')
    else:
        print('No examples were evaluated — check --root and the CST_results directory layout.')


if __name__ == '__main__':
    main()
