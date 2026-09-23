import os
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random

from configs.parser import ArgParser
from src.dataset.datasets import corpus_root
from src.dataset.pixel_dataloader import PixelDataset
from src.models.gps import GPS
from src.metrics.ff_metrics import Metrics
from src.geometry.create_pixel_antenna import add_reflector_to_graph, create_pixel_ant
from torch_geometric.data import Batch
from src.graph.GNN_functions import prepare_graph
from src.geometry.mesh_functions import decompose_pyg_graph
from src.geometry.saving_functions import save_graph_dict_as_stl
from src.diffusion.smooth_binarize import smooth_binarize
from src.diffusion.diffusion_utils import create_parameter_dict
import torch_geometric.transforms as T

class MSELoss(nn.Module):
    def __init__(self, reduce=True):
        super(MSELoss, self).__init__()

    def forward(self, pred, target, reduce=False):
        #pred and target are of shape (batch_size, H, W)
        pred = torch.clamp(pred, -15, 10)
        target = torch.clamp(target, -15, 10)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=(1, 2))  # Compute mean over spatial dimensions for each sample
        if reduce:
            return loss.mean()  # Return the mean loss across the batch
        return loss  # Return the loss for each sample


def seed_everything(seed):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

class RandomFilterBaseline:
    def __init__(self, config, simulation_model):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = simulation_model
        surrogate_checkpoint = config.get('surrogate_checkpoint')
        if surrogate_checkpoint is None:
            raise ValueError("config['surrogate_checkpoint'] must point to the trained GPS+PAIS forward-model .pt")
        self.model.load_state_dict(torch.load(surrogate_checkpoint))
        self.model.eval()
        self.model.to(self.device)

        self.ff_loss_fn = MSELoss()
        self.metrics = Metrics(config)
        self.compute_pe = T.AddLaplacianEigenvectorPE(k=10, attr_name='pe')

        self.train_dataset = None  # Will be set before running

    def get_batch_data(self, batch):
        graph, s11, farfeilds, graph_identifier, example_paramters, raw_idx = batch
        data = example_paramters['ant_parameters'][0].unsqueeze(1)
        data = smooth_binarize(data, 300, 300)

        if self.config.get('data_set_type') == 'pixel_data_with_reflectors':
            reflector_matrix = example_paramters['reflector_matrix'].unsqueeze(1)
            reflector_matrix = smooth_binarize(reflector_matrix, 300, 300)
            data = torch.cat([data, reflector_matrix], dim=1)

        label = farfeilds[:,self.config['idx_freq'],:,:]
        class_labels = example_paramters['class_label']
        return data, label, class_labels, raw_idx

    def plot_antenna_and_farfield(self, antenna, pred_farfield, gt_farfield, title_suffix='', vmin=0, vmax=10):
        plt.figure(figsize=(15, 10))
        plt.subplot(1, 3, 1)

        if self.config.get('data_set_type') == 'pixel_data_with_reflectors':
            combined_image = torch.cat([antenna[0], antenna[1]], dim=1)
            combined_image = combined_image.cpu().numpy().squeeze()
            plt.imshow(combined_image, cmap='gray')
            antenna0 = antenna[0].detach().cpu()
            if antenna0.dim() > 2:
                antenna0 = antenna0.squeeze()
            max_idx = torch.argmax(antenna0).item()
            max_row, max_col = divmod(max_idx, antenna0.shape[-1])
            plt.scatter(max_col, max_row, c='red', s=30)
        else:
            plt.imshow(antenna.cpu().squeeze(), cmap='gray')

        plt.title(f"Antenna {title_suffix}")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(pred_farfield.reshape(34, 34).cpu(), cmap='jet', vmin=vmin, vmax=vmax)
        plt.title(f"Farfield {title_suffix}")
        plt.colorbar()
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(gt_farfield.reshape(34, 34).cpu(), cmap='jet', vmin=vmin, vmax=vmax)
        plt.title(f"Ground Truth Farfield {title_suffix}")
        plt.colorbar()
        plt.axis('off')

        plt.tight_layout()
        return plt

    def save_output(self, min_graph, min_ant_matrix, gt_ant_matrix, min_farfield, gt_farfield, min_loss, class_label, raw_idx):
        os.makedirs(self.config['output_dir'], exist_ok=True)
        save_path = os.path.join(self.config['output_dir'], f'{raw_idx}')
        os.makedirs(save_path, exist_ok=True)

        antenna_graph_dict = decompose_pyg_graph(min_graph.to(min_ant_matrix.device))
        save_graph_dict_as_stl(antenna_graph_dict, output_dir=save_path)

        plt_fig = self.plot_antenna_and_farfield(min_ant_matrix.cpu(), min_farfield, gt_farfield, title_suffix=f"(Sample {raw_idx}, Loss: {min_loss:.4f})")
        plt_fig.savefig(os.path.join(save_path, f'antenna_and_farfield_idx_{raw_idx}_class_label_{class_label}.png'))
        plt_fig.close()

        torch.save(min_farfield.cpu(), os.path.join(save_path, f'pred_farfield.pt'))
        torch.save(gt_farfield.cpu(), os.path.join(save_path, f'gt_farfield.pt'))
        torch.save(min_ant_matrix.cpu(), os.path.join(save_path, f'pred_antenna_matrix.pt'))
        if gt_ant_matrix is not None:
            torch.save(gt_ant_matrix.cpu(), os.path.join(save_path, f'gt_antenna_matrix.pt'))

    def run_gnn(self, samples, eval_labels):
        env_dict, reflectors_dict = create_parameter_dict(self.config)
        losses, pred_list, gt_list, antenna_list, valid_indices = [], [], [], [], []

        batch_size = samples.size(0)
        max_batch_size = 32
        num_splits = (batch_size + max_batch_size - 1) // max_batch_size

        for split_idx in range(num_splits):
            start_idx = split_idx * max_batch_size
            end_idx = min((split_idx + 1) * max_batch_size, batch_size)
            antenna_chunk_list = []
            chunk_valid_indices = []

            for i in range(start_idx, end_idx):
                matrix = smooth_binarize(samples[i, 0], 300, 300)
                antenna, _ = create_pixel_ant(
                    matrix,
                    threshold=self.config.get('threshold'),
                    size_of_patch_in_mm=env_dict['patch_x'],
                    size_of_FR4_in_mm=env_dict['ground_x'],
                    size_of_ground=env_dict['ground_x'],
                    height=env_dict['h'],
                    reflectors_dict=reflectors_dict,
                    create_physical_pixel_mesh=True
                )

                if self.config.get('data_set_type') == 'pixel_data_with_reflectors':
                    reflector_matrix = smooth_binarize(samples[i, 1], 300, 300)
                    antenna = add_reflector_to_graph(antenna, reflector_matrix, env_dict)

                antenna, graph_dict = prepare_graph(antenna.to(self.device), self.config)

                try:
                    antenna = self.compute_pe(antenna)
                    antenna_chunk_list.append(antenna)
                    antenna_list.append(antenna)
                    chunk_valid_indices.append(i)
                except Exception as e:
                    print(f"Skipping sample {i} due to PE error: {e}")
                    continue

            if not antenna_chunk_list:
                continue

            antenna_batch = Batch.from_data_list(antenna_chunk_list)

            with torch.no_grad():
                pred_radiation = self.model(antenna_batch.to(self.device), radiation_image_shape=self.config['radiation_image_shape'])
            if isinstance(pred_radiation, dict):
                pred_radiation = pred_radiation['radiation_image']

            gt = eval_labels[chunk_valid_indices].reshape(-1, self.config['radiation_image_shape'][0], self.config['radiation_image_shape'][1])
            gt = nn.functional.interpolate(gt.unsqueeze(0), size=(34, 34), mode='bilinear', align_corners=False).squeeze(0)
            pred = pred_radiation.reshape(-1, 34, 34)

            loss = self.ff_loss_fn(pred, gt.to(self.device))

            losses.extend(loss.tolist())
            pred_list.extend(pred.cpu())
            gt_list.extend(gt.cpu())
            valid_indices.extend(chunk_valid_indices)

        return pred_list, gt_list, losses, antenna_list, valid_indices

    def test(self, eval_loader):
        min_losses = []
        self.metrics.reset()

        num_gen = self.config['number_of_samples_to_generate']
        num_best = self.config['number_of_best_samples']

        for idx, batch in tqdm(enumerate(eval_loader), total=len(eval_loader), desc="Testing random baseline"):
            with torch.no_grad():
                gt_ant_matrix, eval_labels, class_labels, raw_idx = self.get_batch_data(batch)

                # Fetch N random samples from train_dataset
                random_indices = random.sample(range(len(self.train_dataset)), num_gen)
                random_subset = torch.utils.data.Subset(self.train_dataset, random_indices)
                # Ensure each element doesn't get matched to something incorrectly by forcing batch size
                random_loader = DataLoader(random_subset, batch_size=num_gen, shuffle=False)

                # Grab a single mega-batch
                random_batch = next(iter(random_loader))
                train_ant_data, _, _, _ = self.get_batch_data(random_batch)

                train_ant_data = train_ant_data.to(self.device)

                # Repeat the test eval_labels to match the selected N random samples
                repeated_eval_labels = eval_labels[0].expand(num_gen, -1, -1).to(self.device)

                pred_list, gt_list, losses, antenna_list, valid_indices = self.run_gnn(train_ant_data, repeated_eval_labels)

                if not losses:
                    print(f"No valid samples generated for test batch idx {idx}.")
                    continue

                best_k = min(num_best, len(losses))
                best_indices = np.argsort(losses)[:best_k]

                for rank, b_idx in enumerate(best_indices):
                    min_loss = losses[b_idx]
                    original_idx = valid_indices[b_idx]

                    min_antenna = train_ant_data[original_idx]
                    min_graph = antenna_list[b_idx]
                    min_farfield = pred_list[b_idx]
                    gt_farfield = gt_list[b_idx]

                    if rank == 0:
                        self.metrics.update(min_farfield.unsqueeze(0).unsqueeze(0), gt_farfield.unsqueeze(0).unsqueeze(0))

                    min_losses.append(min_loss)

                    self.save_output(
                        min_graph,
                        min_antenna,
                        gt_ant_matrix[0][0] if gt_ant_matrix.dim() >= 2 else None,
                        min_farfield,
                        gt_farfield,
                        min_loss,
                        class_labels[0] if len(class_labels) > 0 else 0,
                        f"{raw_idx[0]}_top_{rank+1}"
                    )

        computed_metrics = self.metrics.compute()
        print("Evaluation Metrics:")
        for metric_name, metric_value in computed_metrics.items():
            print(f"{metric_name.upper()}: {metric_value:.4f}")

        avg_loss = sum(min_losses) / len(min_losses) if min_losses else float('inf')
        std_loss = np.std(min_losses) if min_losses else float('inf')
        print(f"Average Farfield Loss: {avg_loss:.4f}")
        print(f"Standard Deviation of Loss: {std_loss:.4f}")
        if min_losses:
            print(f"Min loss: {min(min_losses):.4f}")

if __name__ == "__main__":
    seed_everything(0)

    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    arg_parser = ArgParser(config_file=os.path.join(_REPO_ROOT, 'configs', 'inverse', 'random_filtered_baseline.yaml'))
    config = arg_parser.get_config()

    config.update({
        'batch_size': 128,
        'eval_batch_size': 1,
        'number_of_best_samples': 5,
        'number_of_samples_to_generate': 500,
        'idx_freq': 3,
        'radiation_image_shape': [34, 34, 1],
        'threshold': 0.5,
        'physical_antenna': True,
        'use_node_probs': False,
        'add_sphere': False,
        'merge_duplicate_nodes': True,
        'data_set_type': 'pixel_data_with_reflectors',
        'add_radius_graph_edges': True,
        'split_type': 'hardest_indices',
        'predict_surface_current': True,
    })
    # Runtime paths come from the yaml config:
    #   data_root      : dataset root (corpora fmnist_cifar_train / fmnist_cifar_test)
    #   path_to_split  : pca_extrapolation_split.pth file
    #   split_dir      : directory holding hardest_indices_100.pth
    #   surrogate_checkpoint, output_dir
    for required_key in ('data_root', 'path_to_split',
                         'split_dir', 'surrogate_checkpoint', 'output_dir'):
        if config.get(required_key) is None:
            raise ValueError(f"config['{required_key}'] must be set in the yaml config file")

    os.makedirs(config["output_dir"], exist_ok=True)

    train_root = corpus_root(config, 'fmnist_cifar_train')
    train_dataset = PixelDataset(train_root, config=config)
    test_root = corpus_root(config, 'fmnist_cifar_test')
    test_dataset = PixelDataset(test_root, config=config)

    combined_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])

    split_path = config.get('path_to_split')
    loaded_split = torch.load(split_path, weights_only=False)

    train_ds = torch.utils.data.Subset(combined_dataset, loaded_split['train'])
    test_ds = torch.utils.data.Subset(combined_dataset, loaded_split['test'])

    if config['split_type'] == 'hardest_indices':
        try:
            hardest_indices = torch.load(os.path.join(config['split_dir'], 'hardest_indices_100.pth'), weights_only=False)
            test_ds = torch.utils.data.Subset(test_ds, hardest_indices[:100])
        except Exception as e:
            print(f"Warning: Could not load hardest_indices from {config['split_dir']}. Defaulting to first 100.")
            test_ds = torch.utils.data.Subset(test_ds, list(range(100)))

    eval_loader = DataLoader(test_ds, batch_size=1, follow_batch=['pos_surface_current'], shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    simulation_model = GPS(
        config=config,
        in_channels=16,
        channels=128,
        pe_dim=10,
        num_layers=10,
        attn_type='multihead',
        attn_kwargs={'dropout': 0.3},
        out_channels=config.get('radiation_image_shape')[0]*config.get('radiation_image_shape')[1]
    ).to(device)

    baseline = RandomFilterBaseline(config, simulation_model)
    baseline.train_dataset = train_ds
    baseline.test(eval_loader)
