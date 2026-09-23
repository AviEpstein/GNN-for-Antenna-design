#!/usr/bin/env python3
import argparse
import os
import torch
from tqdm import tqdm
from src.geometry.create_pixel_antenna import create_pixel_ant, create_reflector_matrix
from src.geometry.mesh_functions import decompose_pyg_graph
from src.geometry.mesh_functions_pytorch_2 import merge_and_connect_graphs_from_dict
from src.geometry.saving_functions import save_graph_dict_as_stl, save_matrix_and_dict

FEED_EDGES = ['top', 'bottom', 'left', 'right']


def make_patch_matrix(grid_size, patch_size, row_off, col_off, feed_edge):
    """Build a (grid_size x grid_size) tensor with a solid patch and a marked feed pixel.

    All patch pixels = 0.9 (above threshold 0.5); the feed pixel = 1.0 so
    create_pixel_ant's max-pixel logic places the feed there.
    """
    matrix = torch.zeros(grid_size, grid_size)
    matrix[row_off:row_off + patch_size, col_off:col_off + patch_size] = 0.9

    mid_r = row_off + patch_size // 2
    mid_c = col_off + patch_size // 2

    if feed_edge == 'top':
        fr, fc = row_off, mid_c
    elif feed_edge == 'bottom':
        fr, fc = row_off + patch_size - 1, mid_c
    elif feed_edge == 'left':
        fr, fc = mid_r, col_off
    else:  # right
        fr, fc = mid_r, col_off + patch_size - 1

    matrix[fr, fc] = 1.0
    return matrix


def make_reflector_matrix(grid_size, reflector_patch_size):
    """Centered solid patch for the reflector (no specific feed edge needed)."""
    matrix = torch.zeros(grid_size, grid_size)
    r_off = (grid_size - reflector_patch_size) // 2
    c_off = (grid_size - reflector_patch_size) // 2
    matrix[r_off:r_off + reflector_patch_size, c_off:c_off + reflector_patch_size] = 1.0
    return matrix


def create_parameter_dict(config):
    return {
        'patch_x': config.get('patch_x', 28),
        'patch_y': config.get('patch_y', 28),
        'ground_x': config.get('ground_x', 50),
        'ground_y': config.get('ground_y', 50),
        'radius': config.get('radius', 75),
        'num_reflectors': config.get('num_reflectors', 2),
        'h': config.get('h', 4),
        'box_size': config.get('box_size', 50),
        'tan_d': config.get('tan_d', 0.0027),
        'eps_r': config.get('eps_r', 3.55),
        'reflector_distance': config.get('reflector_distance', 3),
        'reflector_scale': config.get('reflector_scale', 1.5),
        'threshold': config.get('threshold', 0.5),
    }


def create_classic_patch_dataset(config, with_reflector=False):
    output_dir = config['output_dir']
    grid_size = 16
    patch_size = config['patch_size']
    offset_step = config['offset_step']
    reflector_extra = config['reflector_extra']

    env_dict = create_parameter_dict(config)
    threshold = env_dict['threshold']
    reflector_distance = env_dict['reflector_distance']

    max_off = grid_size - patch_size
    offsets = list(range(0, max_off + 1, offset_step))

    total = len(offsets) * len(offsets) * len(FEED_EDGES)
    skipped = []
    idx = 0

    with tqdm(total=total) as pbar:
        for row_off in offsets:
            for col_off in offsets:
                for feed_edge in FEED_EDGES:
                    pbar.set_description(f"row={row_off} col={col_off} edge={feed_edge}")
                    # try:
                    matrix = make_patch_matrix(grid_size, patch_size, row_off, col_off, feed_edge)

                    antenna, _ = create_pixel_ant(
                        matrix,
                        threshold=threshold,
                        size_of_patch_in_mm=env_dict['patch_x'],
                        size_of_FR4_in_mm=env_dict['ground_x'],
                        size_of_ground=env_dict['ground_x'],
                        height=env_dict['h'],
                        reflectors_dict=None,
                        create_physical_pixel_mesh=True,
                    )
                    antenna_graph_dict = decompose_pyg_graph(antenna)

                    if with_reflector:
                        reflector_patch_size = patch_size + reflector_extra
                        
                        reflector_matrix = make_reflector_matrix(grid_size, reflector_patch_size)
                        reflector = create_reflector_matrix(reflector_matrix, env_dict)
                        # antenna_ref, _ = create_pixel_ant(
                        #     reflector_matrix,
                        #     threshold=threshold,
                        #     size_of_patch_in_mm=env_dict['patch_x'],
                        #     size_of_FR4_in_mm=env_dict['ground_x'],
                        #     size_of_ground=env_dict['ground_x'],
                        #     height=env_dict['h'],
                        #     reflectors_dict=None,
                        #     create_physical_pixel_mesh=True,
                        # )
                        # ref_dict = decompose_pyg_graph(antenna_ref)
                        # reflector = ref_dict['Antenna_PEC_STEP']
                        # reflector.pos[:, 2] += reflector_distance
                        antenna_graph_dict['PEC_Reflector'] = reflector
                        full_antenna = merge_and_connect_graphs_from_dict(
                            antenna_graph_dict, 'sphere', k=2, add_sphere=False
                        )

                    sample_dir = os.path.join(output_dir, str(idx))
                    save_graph_dict_as_stl(antenna_graph_dict, output_dir=sample_dir)

                    param_dict = env_dict.copy()
                    param_dict['matrix'] = matrix
                    param_dict['patch_size'] = patch_size
                    param_dict['feed_edge'] = feed_edge
                    param_dict['row_off'] = row_off
                    param_dict['col_off'] = col_off
                    if with_reflector:
                        param_dict['reflector_matrix'] = reflector_matrix
                        param_dict['reflector_patch_size'] = reflector_patch_size

                    pkl_path = os.path.join(sample_dir, 'matrix_and_env_dict.pkl')
                    save_matrix_and_dict(matrix, param_dict, pkl_path)

                    # except Exception as e:
                    #     print(f"Error at idx={idx} row={row_off} col={col_off} edge={feed_edge}: {e}")
                    #     skipped.append(idx)

                    idx += 1
                    pbar.update(1)

    skipped_path = os.path.join(output_dir, 'skipped_idxs.pkl')
    torch.save(skipped, skipped_path)
    print(f"Done. {idx - len(skipped)}/{idx} samples saved. Skipped: {len(skipped)}")


def main():
    parser = argparse.ArgumentParser(description="Generate classic rectangular patch antenna dataset.")
    parser.add_argument('--output-dir', type=str, default='outputs/antenna_classic_patch/')
    parser.add_argument('--patch-size', type=int, default=7)
    parser.add_argument('--offset-step', type=int, default=3,
                        help='Step for patch row/col offsets within the 16x16 grid')
    parser.add_argument('--with-reflector',default=True, action='store_true',
                        help='Add a larger solid patch reflector above each antenna')
    parser.add_argument('--reflector-extra', type=int, default=2,
                        help='Reflector patch size = patch_size + reflector_extra')
    parser.add_argument('--reflector-distance', type=float, default=3.0,
                        help='Z-offset of reflector in mm')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--patch-x', type=float, default=28.0, help='Patch size in mm')
    parser.add_argument('--ground-x', type=float, default=50.0, help='Ground plane size in mm')
    parser.add_argument('--h', type=float, default=4.0, help='Substrate height in mm')
    args = parser.parse_args()

    config = vars(args)
    # normalise hyphen keys to underscore
    config = {k.replace('-', '_'): v for k, v in config.items()}

    os.makedirs(config['output_dir'], exist_ok=True)
    create_classic_patch_dataset(config, with_reflector=config['with_reflector'])


if __name__ == '__main__':
    main()
