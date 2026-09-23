"""
evaluate_cst.py
---------------
Canonical Tables-2/3 aggregator: evaluates CST-simulated far-fields of
optimized antennas against their targets and against the nearest-neighbor
far-field from the training set.

Each CST output directory is expected to contain:
    <dir>/results/<example>_top_<k>/farfield_<frequency>.npy
    <dir>/models/<example>_top_<k>/gt_farfield.pt
Folders are grouped by the "<example>" prefix, the best of the top-K CST
runs (lowest MSE against the target) represents each group, and metrics are
aggregated over all groups (target vs. optimized, and target vs. NN).

Paper-table row -> CST output directory naming patterns
-------------------------------------------------------
The directories fed to this script during the paper runs followed these
naming patterns (one run directory per table row):

    Table row / figure                          Directory naming pattern
    ------------------------------------------  ------------------------------------------------------------------
    Diffusion (ours), 100 hardest PCA targets   CST_output_MNIST_CIFAR_diffusion_gps_outputs_gudince_1_34_34_full_train_net_100_hardest_fix/
    Diffusion (ours), N=500 cand., hardest 100  CST_output_MNIST_CIFAR_diffusion_gps_outputs_gudince_1_34_34_full_train_net_100_500_hardest_fix/
    Diffusion (ours), 100 easy targets          CST_output_MNIST_CIFAR_diffusion_gps_outputs_gudince_1_34_34_full_train_net_100_easy/
    Diffusion, random-5-of-N ablation           CST_output_MNIST_CIFAR_diffusion_random_5_outputs/
    CMA-ES baseline, hardest 100                CST_output_MNIST_CIFAR_cma_baseline_hardest_100_random/
    GA baseline, hardest 100                    CST_output_MNIST_SANN_SA_EG_baselines_hardest_100/CST_output_MNIST_CIFAR_GA_baseline_hardest_100/
    SA baseline, hardest 100                    CST_output_MNIST_SANN_SA_EG_baselines_hardest_100/CST_output_MNIST_CIFAR_SA_baseline_hardest_100/
    NN-seeded SA baseline, hardest 100          CST_output_MNIST_SANN_SA_EG_baselines_hardest_100/CST_output_MNIST_CIFAR_NNseed_SA_baseline_hardest_100/
    Classic rectangular patch reference         CST_output_classic_rectangle_patch_no_reflector/
    Hand-designed targets (Table 3 / Fig. 3)    CST_output_diffusion_outputs_manual_new_test_big_full/
    Hand-designed targets, guidance 2 variants  CST_output_MNIST_CIFAR_diffusion_manual_diff/  (and *_guidince_2_* seed/batch variants)
    Early HDGCNN-surrogate run (not in tables)  CST_output_MNIST_diffusion_HGCNN_guidence_1_top_5/

Usage
-----
    python scripts/evaluate_cst.py --cst_output_dirs <dir>
    python scripts/evaluate_cst.py --cst_output_dirs "[<dir1>, <dir2>]"

Dataset/split paths come from the yaml config (configs/inverse/evaluate_cst.yaml
by default); any key can be overridden on the command line as --key value:
    cst_output_dirs           (a single directory, or a yaml list of them)
    data_root
    split_dir
    path_to_split
"""
import os
import numpy as np
import torch
from configs.parser import ArgParser
from src.dataset.dataloader_utils import resize_farfeild, normalize_gain
from src.dataset.datasets import  dataset_hardest_pca
from src.metrics.peak_power_metric import calculate_HPBW, calculate_Boresight_error
from src.metrics.ff_metrics import Metrics
from matplotlib import pyplot as plt
from src.inverse.nearest_neighbor import AntennaNearestNeighbor
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

###########################################################

plot = False


def evaluate_cst_output_dir(CST_output_dir, config, NN, nbrs, loss_function, frequency):
    """Evaluate a single CST output directory (one paper-table row)."""
    CST_result_dir = os.path.join(CST_output_dir, 'results/')
    CST_models_dir = os.path.join(CST_output_dir, 'models/')

    ff_metrics_target_vs_opt = Metrics(config={})
    ff_metrics_target_vs_nn = Metrics(config={})

    # Gather all folders matching the pattern and evaluate each optimized antenna against its target and nearest neighbor:
    groups = {}
    for folder in os.listdir(CST_result_dir):
        if "_top_" in folder:
            group = folder.split("_top_")[0]
            groups.setdefault(group, []).append(folder)

    best_example_loss = float('inf')

    for group, folders in sorted(groups.items()):
        # 1) load gt farfield data of the optimized antenna:
        target_farfeild_path = os.path.join(CST_models_dir, folders[0] + '/gt_farfield' + '.pt')
        target_farfeild = torch.load(target_farfeild_path)
        target_farfeild = target_farfeild.unsqueeze(0).unsqueeze(0)
        best_mse = float('inf')
        # 2) load CST farfield data of the optimized antenna get the best one among the top 5:
        for folder in folders:
            farfeild_path = os.path.join(CST_result_dir, folder + '/farfield_'+ str(frequency) + '.npy')
            farfield_data = np.load(farfeild_path)
            farfield_data = torch.tensor(farfield_data,dtype=torch.float32)
            farfeild = resize_farfeild(farfield_data, config['radiation_image_shape'][:2]).unsqueeze(0)
            CST_opt_normalized_gain_farfeild = normalize_gain(farfeild[0][:,:,0],farfeild[0][:,:,1], img_shape=config['radiation_image_shape'][:2])
            mse = loss_function(CST_opt_normalized_gain_farfeild, target_farfeild[0][0])
            if mse < best_mse:
                best_mse = mse
                best_CST_opt_normalized_gain_farfeild = CST_opt_normalized_gain_farfeild
                best_example_name = folder


        # get the nearest neighbor in the training set:
        flatened_target_farfeild = [target_farfeild.numpy().flatten()]
        nearest_neighbors_mesh_id, nearest_neighbors_farfeild, nn_distance, matched_raw_idx, matched_graph, matched_s11 = NN.find_nearest_neighbor(flatened_target_farfeild, nbrs)

        if matched_raw_idx[0][0] == group:
            nearest_neighbors_farfeild  = nearest_neighbors_farfeild[1]
        else:
            nearest_neighbors_farfeild  = nearest_neighbors_farfeild[0]

        ff_metrics_target_vs_opt.update(best_CST_opt_normalized_gain_farfeild.unsqueeze(0).unsqueeze(0), target_farfeild)
        ff_metrics_target_vs_nn.update(nearest_neighbors_farfeild.unsqueeze(0).unsqueeze(0), target_farfeild)

        mse_target_vs_opt = loss_function(best_CST_opt_normalized_gain_farfeild, target_farfeild[0][0])
        mse_target_vs_nn = loss_function(nearest_neighbors_farfeild, target_farfeild[0][0])
        mssim_target_vs_opt = ff_metrics_target_vs_opt.ssim_metric(best_CST_opt_normalized_gain_farfeild.unsqueeze(0).unsqueeze(0), target_farfeild)
        mssim_target_vs_nn = ff_metrics_target_vs_nn.ssim_metric(nearest_neighbors_farfeild.unsqueeze(0).unsqueeze(0), target_farfeild)
        hpbw_target_vs_opt = calculate_HPBW(best_CST_opt_normalized_gain_farfeild, target_farfeild[0][0])
        hpbw_target_vs_nn = calculate_HPBW(nearest_neighbors_farfeild, target_farfeild[0][0])
        boresight_error_target_vs_opt = calculate_Boresight_error(best_CST_opt_normalized_gain_farfeild, target_farfeild[0][0])
        boresight_error_target_vs_nn = calculate_Boresight_error(nearest_neighbors_farfeild, target_farfeild[0][0])

        if hpbw_target_vs_opt > hpbw_target_vs_nn or   mse_target_vs_opt < mse_target_vs_nn or mssim_target_vs_opt > mssim_target_vs_nn:
            print(f"Example {group}: Optimized farfield is closer to target than nearest neighbor. MSE Target vs Opt: {mse_target_vs_opt.item():.4f}, MSE Target vs NN: {mse_target_vs_nn.item():.4f}")
            print('metrics for this example:')
            print(f"  MSE Target vs Opt: {mse_target_vs_opt.item():.6f}")
            print(f"  MSE Target vs NN: {mse_target_vs_nn.item():.6f}")
            print(f"  SSIM Target vs Opt: {mssim_target_vs_opt.item():.6f}")
            print(f"  SSIM Target vs NN: {mssim_target_vs_nn.item():.6f}")
            print('MS-MSSIM Target vs Opt:', ff_metrics_target_vs_opt.mssim_metric(best_CST_opt_normalized_gain_farfeild.unsqueeze(0).unsqueeze(0), target_farfeild).item(), ff_metrics_target_vs_nn.mssim_metric(nearest_neighbors_farfeild.unsqueeze(0).unsqueeze(0), target_farfeild).item())
            print('MS-MSSIM Target vs NN:', ff_metrics_target_vs_opt.mssim_metric(best_CST_opt_normalized_gain_farfeild.unsqueeze(0).unsqueeze(0), target_farfeild).item(), ff_metrics_target_vs_nn.mssim_metric(nearest_neighbors_farfeild.unsqueeze(0).unsqueeze(0), target_farfeild).item())
            print('MAE Target vs Opt:', ff_metrics_target_vs_opt.mae_metric(best_CST_opt_normalized_gain_farfeild.unsqueeze(0).unsqueeze(0), target_farfeild).item(), ff_metrics_target_vs_nn.mae_metric(nearest_neighbors_farfeild.unsqueeze(0).unsqueeze(0), target_farfeild).item())
            print('MAE Target vs NN:', ff_metrics_target_vs_opt.mae_metric(best_CST_opt_normalized_gain_farfeild.unsqueeze(0).unsqueeze(0), target_farfeild).item(), ff_metrics_target_vs_nn.mae_metric(nearest_neighbors_farfeild.unsqueeze(0).unsqueeze(0), target_farfeild).item())
            print('PSNR Target vs Opt:', ff_metrics_target_vs_opt.psnr_metric(best_CST_opt_normalized_gain_farfeild.unsqueeze(0).unsqueeze(0), target_farfeild).item(), ff_metrics_target_vs_nn.psnr_metric(nearest_neighbors_farfeild.unsqueeze(0).unsqueeze(0), target_farfeild).item())
            print('PSNR Target vs NN:', ff_metrics_target_vs_opt.psnr_metric(best_CST_opt_normalized_gain_farfeild.unsqueeze(0).unsqueeze(0), target_farfeild).item(), ff_metrics_target_vs_nn.psnr_metric(nearest_neighbors_farfeild.unsqueeze(0).unsqueeze(0), target_farfeild).item())
            print('HPBW Target vs Opt:', hpbw_target_vs_opt.item(), hpbw_target_vs_nn.item())
            print('Boresight Error Target vs Opt:', boresight_error_target_vs_opt, boresight_error_target_vs_nn)
        if plot:
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            axs[0].imshow(best_CST_opt_normalized_gain_farfeild, cmap='jet', vmin=0, vmax=10)
            axs[0].set_title(f'Optimized Normalized Gain Farfield\nMSE: {mse_target_vs_opt.item():.4f}')
            axs[0].axis('off')
            axs[1].imshow(target_farfeild[0][0], cmap='jet', vmin=0, vmax=10)
            axs[1].set_title('Ground Truth Target Farfield')
            axs[1].axis('off')
            # Show nearest neighbor farfield
            if isinstance(nearest_neighbors_farfeild, torch.Tensor):
                nn_ff_img = nearest_neighbors_farfeild.squeeze().detach().cpu().numpy()
            else:
                nn_ff_img = np.array(nearest_neighbors_farfeild)
            axs[2].imshow(nn_ff_img, cmap='jet', vmin=0, vmax=10)
            axs[2].set_title(f'Nearest Neighbor Farfield\nMSE: {mse_target_vs_nn.item():.4f}')
            axs[2].axis('off')
            plt.tight_layout()
            plt.show()
            print('Finished displaying farfield for example:', best_example_name)
            # plot the antenna geometry of the optimized design, target, and nearest neighbor:
            # load the antenna geometry of the optimized design:

            optimized_antenna_path = os.path.join(CST_models_dir, folder + '/optimized_antenna.pt')
            optimized_antenna = torch.load(optimized_antenna_path)
            # load the antenna geometry of the target design:
            target_antenna_path = os.path.join(CST_models_dir, folders[0] + '/target_antenna.pt')
            target_antenna = torch.load(target_antenna_path)


    metrics_result_target_vs_opt = ff_metrics_target_vs_opt.compute()
    print("Overall metrics across all examples (Target vs Optimized):")
    for key, value in metrics_result_target_vs_opt.items():
        print(f"  {key}: {value:.6f}")

    metrics_result_target_vs_nn = ff_metrics_target_vs_nn.compute()
    print("Overall metrics across all examples (Target vs Nearest Neighbor):")
    for key, value in metrics_result_target_vs_nn.items():
        print(f"  {key}: {value:.6f}")


if __name__ == '__main__':
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    arg_parser = ArgParser(config_file=os.path.join(_REPO_ROOT, 'configs', 'inverse', 'evaluate_cst.yaml'))
    config = arg_parser.get_config()

    cst_output_dirs = config.get('cst_output_dirs')
    if not cst_output_dirs:
        raise ValueError("Pass --cst_output_dirs (or set 'cst_output_dirs' in the yaml config) "
                         "with at least one CST output directory")
    if isinstance(cst_output_dirs, str):
        cst_output_dirs = [cst_output_dirs]
    for required_key in ('data_root', 'split_dir', 'path_to_split'):
        if config.get(required_key) is None:
            raise ValueError(f"config['{required_key}'] must be set in the yaml config file or on the command line")

    # set up dataset and nearest neighbor search:
    config['num_sphere_nodes'] = 162
    config['add_sphere'] = False
    config['radiation_image_shape'] = [34,34,1]
    frequency = 5600  # 5.6 GHz
    freq_idx = 3  # Choose frequency index for far-field comparison
    config['idx_freq'] = freq_idx
    config['frequencies'] = [2400, 2800, 5200, 5600,6000]

    config['data_set_name'] = 'FMNIST_CIFAR' # 'combine_datasets' or 'FMNIST' or 'FMNIST_CIFAR'
    config['split_type'] = 'easy_indices' # 'easy_indices' or 'random_split' or 'pca_extrapolation_split' or 'farfield_split_with_buckets' or 'hardest_indices'

    pe_transform = T.Compose([
        T.ToUndirected(),
        T.AddLaplacianEigenvectorPE(k=10, attr_name='pe')
    ])

    # set seed for reproducibility
    torch.manual_seed(0)
    combined_dataset, train_dataset, test_dataset = dataset_hardest_pca(config, N=config.get('hardest_n', 100), pe_transform=pe_transform)

    # load the nearest neighbor model or create it if it doesn't exist:
    num_cpus = os.cpu_count() or 4
    num_workers = min(8, max(1, num_cpus // 2))
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers,
                                pin_memory=True,
                                prefetch_factor=2,)
    loss_function = torch.nn.MSELoss()

    NN = AntennaNearestNeighbor(train_loader,freq_idx=freq_idx, Nearest_neighbor_feture='calc_nn_from_farfeild', loss_function=loss_function)
    np_train_set = [tensor.numpy().flatten() for tensor in NN.train_ff_images]
    nbrs = NN.train_nearest_neighbor(np_train_set)
    # save the nearest neighbor model:
    torch.save(nbrs, os.path.join(config['data_root'], 'ff_nearest_neighbor_model.pth'))

    for CST_output_dir in cst_output_dirs:
        print(f"\n================ Evaluating {CST_output_dir} ================")
        evaluate_cst_output_dir(CST_output_dir, config, NN, nbrs, loss_function, frequency)

    print("Evaluation complete.")
