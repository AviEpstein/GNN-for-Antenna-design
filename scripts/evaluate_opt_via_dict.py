"""
Aggregate metrics.json files produced by the CST optimization runs.

Expects a base directory containing folders named "<group>_nn_<k>", each with a
metrics.json holding "best_metrics" (MSE / MAE / mssim) and, for the "<group>_nn_0"
reference folder, "nn_ff_metrics". For every group the best value across the
folders is compared against the nn_0 reference, and averages / win counts are
reported.
"""
import argparse
import os
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-dir', '--base_dir', dest='base_dir', type=str, required=True,
                        help='Directory containing the "<group>_nn_<k>" result folders with metrics.json files.')
    args = parser.parse_args()

    best_mse_list = []
    best_ref_nn0_mse_list = []
    best_mae_list = []
    best_ref_nn0_mae_list = []
    best_mssim_list = []
    best_ref_nn0_mssim_list = []

    base_dir = args.base_dir
    groups = {}

    # Gather all folders matching the pattern
    for folder in os.listdir(base_dir):
        if "_nn_" in folder:
            group = folder.split("_nn_")[0]
            groups.setdefault(group, []).append(folder)

    for group, folders in sorted(groups.items()):
        mse_dict = {}
        mae_dict = {}
        mssim_dict = {}
        best_mse = float('inf')
        best_mae = float('inf')
        best_mssim = float('-inf')
        for folder in folders:
            metrics_path = os.path.join(base_dir, folder, "metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path) as f:
                    metrics = json.load(f)
                    mse = metrics.get("best_metrics", {}).get("MSE")
                    mae = metrics.get("best_metrics", {}).get("MAE")
                    mssim = metrics.get("best_metrics", {}).get("mssim")
                    mse_dict[folder] = mse
                    mae_dict[folder] = mae
                    mssim_dict[folder] = mssim

        ref_folder = f"{group}_nn_0"
        ref_metrics_path = os.path.join(base_dir, ref_folder, "metrics.json")
        ref_mse = None
        ref_mae = None
        ref_mssim = None
        if os.path.exists(ref_metrics_path):
            with open(ref_metrics_path) as f:
                metrics = json.load(f)
                ref_mse = metrics.get("nn_ff_metrics", {}).get("MSE")
                ref_mae = metrics.get("nn_ff_metrics", {}).get("MAE")
                ref_mssim = metrics.get("nn_ff_metrics", {}).get("mssim")

        print(f"\nGroup {group}: Reference ({ref_folder}) MSE = {ref_mse}")
        print(f"Group {group}: Reference ({ref_folder}) MAE = {ref_mae}")
        print(f"Group {group}: Reference ({ref_folder}) MSSIM = {ref_mssim}")
        for folder in sorted(folders):
            mse = mse_dict.get(folder)
            mae = mae_dict.get(folder)
            mssim = mssim_dict.get(folder)
            if mse is not None:
                diff = mse - ref_mse if ref_mse is not None else None
                if mse < best_mse:
                    best_mse = mse
            if mae is not None:
                if mae < best_mae:
                    best_mae = mae
            if mssim is not None:
                if mssim > best_mssim:
                    best_mssim = mssim
        best_mse_list.append(best_mse)
        best_ref_nn0_mse_list.append(ref_mse)
        best_mae_list.append(best_mae)
        best_ref_nn0_mae_list.append(ref_mae)
        best_mssim_list.append(best_mssim)
        best_ref_nn0_mssim_list.append(ref_mssim)

    average_best_mse = sum(best_mse_list) / len(best_mse_list) if best_mse_list else float('nan')
    average_best_ref_nn0_mse = sum(best_ref_nn0_mse_list) / len(best_ref_nn0_mse_list) if best_ref_nn0_mse_list else float('nan')
    print(f"\nAverage Best MSE across all groups: {average_best_mse}")
    print(f"Average Best Reference nn_0 MSE across all groups: {average_best_ref_nn0_mse}")
    print(f"Improvement: {average_best_ref_nn0_mse - average_best_mse}")
    win_count = sum(1 for best, ref in zip(best_mse_list, best_ref_nn0_mse_list) if best < ref)
    print(f"Number of wins (best_mse_list winning over best_ref_nn0): {win_count}")

    average_best_mae = sum(best_mae_list) / len(best_mae_list) if best_mae_list else float('nan')
    average_best_ref_nn0_mae = sum(best_ref_nn0_mae_list) / len(best_ref_nn0_mae_list) if best_ref_nn0_mae_list else float('nan')
    print(f"\nAverage Best MAE across all groups: {average_best_mae}")
    print(f"Average Best Reference nn_0 MAE across all groups: {average_best_ref_nn0_mae}")
    print(f"Improvement: {average_best_ref_nn0_mae - average_best_mae}")
    win_count_mae = sum(1 for best, ref in zip(best_mae_list, best_ref_nn0_mae_list) if best < ref)
    print(f"Number of wins (best_mae_list winning over best_ref_nn0): {win_count_mae}")

    average_best_mssim = sum(best_mssim_list) / len(best_mssim_list) if best_mssim_list else float('nan')
    average_best_ref_nn0_mssim = sum(best_ref_nn0_mssim_list) / len(best_ref_nn0_mssim_list) if best_ref_nn0_mssim_list else float('nan')
    print(f"\nAverage Best MSSIM across all groups: {average_best_mssim}")
    print(f"Average Best Reference nn_0 MSSIM across all groups: {average_best_ref_nn0_mssim}")
    print(f"Improvement: {average_best_mssim - average_best_ref_nn0_mssim}")
    win_count_mssim = sum(1 for best, ref in zip(best_mssim_list, best_ref_nn0_mssim_list) if best > ref)
    print(f"Number of wins (best_mssim_list winning over best_ref_nn0): {win_count_mssim}")

    print("Done")
