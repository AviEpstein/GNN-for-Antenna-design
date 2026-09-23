"""Random pixel-antenna STL generation.

Note: this is a reconstructed driver; the original corpus was generated from a
CST-side Windows script not included in the repo. `randomize_ant_with_reflector`
is the original mesh-construction routine (unchanged numerically); only its
path handling was converted from Windows-style separators to os.path.join, and
a minimal argparse __main__ was added to generate N examples to an output dir.
"""

import argparse
import os
import pickle

import torch
import trimesh
from src.geometry.create_pixel_antenna import create_pixel_mesh, create_feed_PEC, create_feed, combine_and_merge, create_ground, create_box_with_cube_hole



def randomize_ant_with_reflector(path_to_save_mesh, model_parameters, grid_size = 16, threshold = 0.5, seed=0):
    if seed > 0:
        torch.manual_seed(seed)
    size_of_patch_in_mm = model_parameters['patch_x']
    size_of_ground = model_parameters['ground_x']
    scale = size_of_patch_in_mm / grid_size
    # create ground:
    size_of_FR4_in_mm = size_of_ground
    height = model_parameters['h']
    antenna_reltive_shift = (size_of_ground - size_of_patch_in_mm) / 2

    # Create an external learnable matrix (logits) of shape (16,16)
    matrix = torch.randn((grid_size, grid_size), requires_grad=True)
    pixel_data, pixel_probs = create_pixel_mesh(matrix, threshold=threshold)
    # scale to mm:
    pixel_data.pos[:, :2] = pixel_data.pos[:, :2] * scale + antenna_reltive_shift

    # Create a trimesh object
    pixel_mesh = trimesh.Trimesh(vertices=pixel_data.pos, faces=pixel_data.faces)

    # create feed PEC:
    max_val = torch.max(matrix)
    x, y = torch.nonzero(matrix == max_val)[0]

    feed_PEC_data = create_feed_PEC(x, y, height)
    feed_PEC_data.pos[:, :2] = feed_PEC_data.pos[:, :2] * scale + antenna_reltive_shift
    feed_PEC_mesh = trimesh.Trimesh(vertices=feed_PEC_data.pos, faces=feed_PEC_data.faces)

    feed_PEC_and_pixel_data = combine_and_merge(feed_PEC_data, pixel_data)  # add this to combine the node are and faces

    # create feed
    feed_data = create_feed(x, y, height)

    # scale to mm:
    feed_data.pos[:, :2] = feed_data.pos[:, :2] * scale + antenna_reltive_shift

    # shift relitive to ground:
    feed_PEC_and_pixel_data.pos[:, 2] = feed_PEC_and_pixel_data.pos[:, 2] + height
    feed_PEC_and_pixel_mesh = trimesh.Trimesh(vertices=feed_PEC_and_pixel_data.pos, faces=feed_PEC_and_pixel_data.faces)
    feed_PEC_and_pixel_mesh.export(os.path.join(path_to_save_mesh, 'PEC_pixel.stl'))

    feed_data.pos[:, 2] = feed_data.pos[:, 2] + height
    feed_mesh = trimesh.Trimesh(vertices=feed_data.pos, faces=feed_data.faces)
    blue_color = [0, 0, 255, 125]  # [R, G, B, A] where A is opacity
    feed_mesh.visual.face_colors = blue_color
    feed_mesh.export(os.path.join(path_to_save_mesh, 'Feed.stl'))

    ground = create_ground()
    ground.pos[:, :2] = ground.pos[:, :2] * size_of_ground
    ground_mesh = trimesh.Trimesh(vertices=ground.pos, faces=ground.faces)
    ground_mesh.export(os.path.join(path_to_save_mesh, 'PEC_ground.stl'))

    FR4_data = create_box_with_cube_hole(cube_size=scale, outerbox_length=size_of_FR4_in_mm, height=model_parameters['h'], hole_center=(
        x * scale + antenna_reltive_shift - size_of_FR4_in_mm / 2 + scale / 2,
        y * scale + antenna_reltive_shift - size_of_FR4_in_mm / 2 + scale / 2))
    # shift to 0, o cordinant
    FR4_data.pos[:, :2] = FR4_data.pos[:, :2] + size_of_FR4_in_mm / 2
    FR4_data.pos[:, 2] = FR4_data.pos[:, 2] + height / 2  # + height-1
    FR4_data.pos[:, :2] = FR4_data.pos[:, :2]  # + antenna_reltive_shift

    FR4_mesh = trimesh.Trimesh(vertices=FR4_data.pos, faces=FR4_data.faces)
    FR4_mesh.export(os.path.join(path_to_save_mesh, 'Dielectric.stl'))

    print('created STLs')
    return matrix, threshold


def main():
    parser = argparse.ArgumentParser(description="Generate N random pixel antennas as STL files (reconstructed driver).")
    parser.add_argument('--output-dir', type=str, default='outputs/random_pixel_antennas/',
                        help='Directory to write one numbered sub-folder of STLs per example')
    parser.add_argument('--num-examples', type=int, default=10, help='Number of examples to generate')
    parser.add_argument('--grid-size', type=int, default=16)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--patch-x', type=float, default=28.0, help='Physical patch size in mm')
    parser.add_argument('--ground-x', type=float, default=50.0, help='Ground plane size in mm')
    parser.add_argument('--h', type=float, default=4.0, help='Substrate height in mm')
    parser.add_argument('--seed', type=int, default=0, help='If > 0, seed torch once before generation')
    args = parser.parse_args()

    if args.seed > 0:
        torch.manual_seed(args.seed)

    model_parameters = {'patch_x': args.patch_x, 'ground_x': args.ground_x, 'h': args.h}

    for idx in range(args.num_examples):
        example_dir = os.path.join(args.output_dir, str(idx))
        os.makedirs(example_dir, exist_ok=True)
        matrix, threshold = randomize_ant_with_reflector(
            example_dir, model_parameters, grid_size=args.grid_size, threshold=args.threshold, seed=0
        )
        # save the matrix/threshold and the environment parameters alongside the STLs,
        # matching the (matrix, threshold) ant_parameters format the dataloader expects
        with open(os.path.join(example_dir, 'ant_parameters.pickle'), 'wb') as f:
            pickle.dump([matrix.detach(), threshold], f)
        with open(os.path.join(example_dir, 'model_parameters.pickle'), 'wb') as f:
            pickle.dump(model_parameters, f)


if __name__ == '__main__':
    main()
