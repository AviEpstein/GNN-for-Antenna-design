#!/usr/bin/env python3
"""Generate the classic *rectangular* patch antenna dataset (non-square patches
swept over sizes, positions and a left-edge feed on a 16x16 pixel grid).

Note on the released 108-example set: the committed defaults
(--patch-rows-min 7 --patch-rows-max 9 --patch-cols-min 1 --patch-cols-max 9
--offset-step-location 3 --offset-step-patch 1, feed edge fixed to 'left')
produce 356 examples, not the 108 in the paper's rectangular set. Enumerating
the sweep arithmetic, the released 108-example set was generated with a single
fixed patch height and the default column sweep -- most plausibly
`--patch-rows-min 9 --patch-rows-max 9` (patch height 9, widths 1..8, left-edge
feed, offset step 3: 3 row offsets x 36 (col-offset x width) combinations = 108;
this is the only single-argument change from the committed defaults that yields
exactly 108). The equivalent `--patch-rows-min 8 --patch-rows-max 8` sweep
(height 8, widths 1..7 and 9) also yields exactly 108. Caveat: the exact
generation arguments were not recorded, so this is a reconstruction from the
sweep logic, not a log.
"""
import argparse
import os
import torch
from tqdm import tqdm
from src.geometry.create_pixel_antenna import create_pixel_ant, create_reflector_matrix
from src.geometry.mesh_functions import decompose_pyg_graph
from src.geometry.mesh_functions_pytorch_2 import merge_and_connect_graphs_from_dict
from src.geometry.saving_functions import save_graph_dict_as_stl, save_matrix_and_dict

FEED_EDGES = ['left'] #'top']#, 'bottom', 'left', 'right']


def make_patch_matrix(grid_size, patch_rows, patch_cols, row_off, col_off, feed_edge):
    """Build a (grid_size x grid_size) tensor with a rectangular patch and a marked feed pixel.

    All patch pixels = 0.9 (above threshold 0.5); the feed pixel = 1.0 so
    create_pixel_ant's max-pixel logic places the feed there.
    """
    matrix = torch.zeros(grid_size, grid_size)
    matrix[row_off:row_off + patch_rows, col_off:col_off + patch_cols] = 0.9

    mid_r = row_off + patch_rows // 2
    mid_c = col_off + patch_cols // 2

    if feed_edge == 'top':
        fr, fc = row_off, mid_c
    elif feed_edge == 'bottom':
        fr, fc = row_off + patch_rows - 1, mid_c
    elif feed_edge == 'left':
        fr, fc = mid_r, col_off
    else:  # right
        fr, fc = mid_r, col_off + patch_cols - 1

    matrix[fr, fc] = 1.0
    return matrix


def make_reflector_matrix(grid_size, reflector_rows, reflector_cols):
    """Centered rectangular solid patch for the reflector."""
    matrix = torch.zeros(grid_size, grid_size)
    r_off = (grid_size - reflector_rows) // 2
    c_off = (grid_size - reflector_cols) // 2
    matrix[r_off:r_off + reflector_rows, c_off:c_off + reflector_cols] = 1.0
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


def create_classic_rectangle_patch_dataset(config, with_reflector=False):
    output_dir = config['output_dir']
    grid_size = 16
    patch_rows_min = config['patch_rows_min']
    patch_rows_max = config['patch_rows_max']
    patch_cols_min = config['patch_cols_min']
    patch_cols_max = config['patch_cols_max']
    offset_step_patch_length = config['offset_step_patch']
    offset_step_location = config['offset_step_location']

    reflector_extra = config['reflector_extra']

    env_dict = create_parameter_dict(config)
    threshold = env_dict['threshold']

    all_patch_shapes = [
        (pr, pc)
        for pr in range(patch_rows_min, patch_rows_max + 1, offset_step_patch_length)
        for pc in range(patch_cols_min, patch_cols_max + 1, offset_step_patch_length)
        if pr != pc  # skip squares
    ]

    total = sum(
        len(range(0, grid_size - pr + 1, offset_step_location)) *
        len(range(0, grid_size - pc + 1, offset_step_location)) *
        len(FEED_EDGES)
        for pr, pc in all_patch_shapes
    )

    skipped = []
    idx = 0

    with tqdm(total=total) as pbar:
        for patch_rows, patch_cols in all_patch_shapes:
            row_offsets = list(range(0, grid_size - patch_rows + 1, offset_step_location))
            col_offsets = list(range(0, grid_size - patch_cols + 1, offset_step_location))

            for row_off in row_offsets:
                for col_off in col_offsets:
                    for feed_edge in FEED_EDGES:
                        pbar.set_description(
                            f"shape=({patch_rows},{patch_cols}) row={row_off} col={col_off} edge={feed_edge}"
                        )

                        matrix = make_patch_matrix(
                            grid_size, patch_rows, patch_cols, row_off, col_off, feed_edge
                        )

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
                            reflector_rows = patch_rows + reflector_extra
                            reflector_cols = patch_cols + reflector_extra
                            reflector_matrix = make_reflector_matrix(
                                grid_size, reflector_rows, reflector_cols
                            )
                            reflector = create_reflector_matrix(reflector_matrix, env_dict)
                            antenna_graph_dict['PEC_Reflector'] = reflector
                            merge_and_connect_graphs_from_dict(
                                antenna_graph_dict, 'sphere', k=2, add_sphere=False
                            )

                        sample_dir = os.path.join(output_dir, str(idx))
                        save_graph_dict_as_stl(antenna_graph_dict, output_dir=sample_dir)

                        param_dict = env_dict.copy()
                        param_dict['matrix'] = matrix
                        param_dict['patch_rows'] = patch_rows
                        param_dict['patch_cols'] = patch_cols
                        param_dict['feed_edge'] = feed_edge
                        param_dict['row_off'] = row_off
                        param_dict['col_off'] = col_off
                        if with_reflector:
                            param_dict['reflector_matrix'] = reflector_matrix
                            param_dict['reflector_rows'] = reflector_rows
                            param_dict['reflector_cols'] = reflector_cols

                        pkl_path = os.path.join(sample_dir, 'matrix_and_env_dict.pkl')
                        save_matrix_and_dict(matrix, param_dict, pkl_path)

                        idx += 1
                        pbar.update(1)

    skipped_path = os.path.join(output_dir, 'skipped_idxs.pkl')
    torch.save(skipped, skipped_path)
    print(f"Done. {idx - len(skipped)}/{idx} samples saved. Skipped: {len(skipped)}")


def main():
    parser = argparse.ArgumentParser(description="Generate classic rectangular patch antenna dataset.")
    parser.add_argument('--output-dir', type=str, default='outputs/antenna_classic_rectangle_patch/')
    parser.add_argument('--patch-rows-min', type=int, default=7, help='Min patch height in grid cells')
    parser.add_argument('--patch-rows-max', type=int, default=9, help='Max patch height in grid cells')
    parser.add_argument('--patch-cols-min', type=int, default=1, help='Min patch width in grid cells')
    parser.add_argument('--patch-cols-max', type=int, default=9, help='Max patch width in grid cells')
    parser.add_argument('--offset-step-location', type=int, default=3,
                        help='Step for patch row/col offsets within the 16x16 grid')
    parser.add_argument('--offset-step-patch', type=int, default=1,
                        help='Step for patch size (rows and cols) to control the number of shapes')
    parser.add_argument('--with-reflector', default=False, action='store_true',
                        help='Add a larger rectangular reflector above each antenna')
    parser.add_argument('--reflector-extra', type=int, default=2,
                        help='Reflector size = (patch_rows + extra) x (patch_cols + extra)')
    parser.add_argument('--reflector-distance', type=float, default=3.0,
                        help='Z-offset of reflector in mm')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--patch-x', type=float, default=28.0, help='Physical patch x-size in mm')
    parser.add_argument('--patch-y', type=float, default=28.0, help='Physical patch y-size in mm')
    parser.add_argument('--ground-x', type=float, default=50.0, help='Ground plane size in mm')
    parser.add_argument('--h', type=float, default=4.0, help='Substrate height in mm')
    args = parser.parse_args()

    config = {k.replace('-', '_'): v for k, v in vars(args).items()}

    os.makedirs(config['output_dir'], exist_ok=True)
    create_classic_rectangle_patch_dataset(config, with_reflector=config['with_reflector'])


if __name__ == '__main__':
    main()
