import torch.nn as nn
from tqdm import tqdm
from configs.parser import ArgParser
import os
from sklearn.neighbors import NearestNeighbors
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.metrics.ff_metrics import Metrics


"""
this class implements a nearest neighbor search for antenna designs based on either farfield patterns or mesh identifiers asa  matrix.
there are two main modes of operation:
1. calc_nn_from_mesh_identifier: finds nearest neighbors based on mesh identifier similarity. then calculate the ff loss between the quary and the nearest neighbor ff.
2. calc_nn_from_farfeild: finds nearest neighbors based on farfeild pattern similarity. then calculate the ff loss between the quary and the nearest neighbor ff.
"""

class AntennaNearestNeighbor:
    def __init__(self, train_loader, freq_idx=0, Nearest_neighbor_feture = 'calc_nn_from_farfeild',loss_function = nn.MSELoss()):
        assert(Nearest_neighbor_feture == 'calc_nn_from_farfeild' or Nearest_neighbor_feture == 'calc_nn_from_mesh_identifier')

        self.freq_idx = freq_idx # 2400 MHz
        self.train_ff_images, self.train_mesh_identfiers, self.idx_to_raw_idx_map, self.train_graphs, self.train_s11 = self.parse_train_loader(train_loader)
        self.loss_function = loss_function
        self.Nearest_neighbor_feture = Nearest_neighbor_feture


    def parse_train_loader(self, train_loader):
        print('\nparsing train loader for nearest neighbor\n')
        farfeld_list = []
        mesh_identfier_list = []
        idx_to_raw_idx_list = []
        graph_list = []
        s11_list = []
        for _, (graph, s11, ff_image, _, example_parameters, raw_index) in tqdm(enumerate(train_loader), desc="parsing train dataset", unit=' batches', leave=True):
            batch_size = len(raw_index)
            for i in range(batch_size):
                idx_to_raw_idx_list.append(raw_index[i])
                farfeld_list.append(ff_image[i][self.freq_idx])
                matrix = example_parameters['ant_parameters'][0][i]
                if 'reflector_matrix' in example_parameters:
                    matrix = torch.cat((matrix, example_parameters['reflector_matrix'][i]), dim=0)
                mesh_identfier_list.append(matrix.flatten())
                s11_list.append(s11['s11_abs_db'][i])

            if hasattr(graph, 'to_data_list'):
                graph_list.extend(graph.to_data_list())
            else:
                graph_list.extend(graph if isinstance(graph, list) else [graph])
        return farfeld_list, mesh_identfier_list, idx_to_raw_idx_list, graph_list, s11_list

    def train_nearest_neighbor(self, training_set, n_neighbors=2):
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(training_set)
        return nbrs

    def find_nearest_neighbor(self, quary, nbrs):
        distances, indices = nbrs.kneighbors(quary)
        return [[self.train_mesh_identfiers[i] for i in indices[0]], [self.train_ff_images[i] for i in indices[0]], distances[0], [self.idx_to_raw_idx_map[i] for i in indices[0]], [self.train_graphs[i] for i in indices[0]], [self.train_s11[i] for i in indices[0]] ]

    def load_nearest_neighbor_model(self, path):
        if os.path.exists(path):
            nbrs = torch.load(path, weights_only=False)
            print("Loaded nearest neighbor model from: ", path)
            return nbrs
        else:
            print("No nearest neighbor model found at: ", path)
            return None

    def save_nearest_neighbor_model(self, nbrs, path):
        torch.save(nbrs, path)
        print("Saved nearest neighbor model to: ", path)


    def calculate_nearest_neighbor_loss_on_test_set(self, test_loader, nbrs):
        total_loss = 0
        losses = []
        num_examples = len(test_loader)
        ff_metrics_target_vs_nn = Metrics(config={})
        if self.Nearest_neighbor_feture == 'calc_nn_from_mesh_identifier':
            print('\ncalculating nn avg loss in  (mesh->ff)...\n')
            for idx, (graph, ant, ff_image, graph_identifier, example_parameters, raw_index) in tqdm(enumerate(test_loader), desc="runing nn on test dataset", unit=' examples', leave=True):
                matrix = example_parameters['ant_parameters'][0]
                if 'reflector_matrix' in example_parameters:
                    matrix = torch.cat((matrix, example_parameters['reflector_matrix']), dim=0)
                quary = [matrix.flatten()]
                nearest_neighbors_mesh_id, nearest_neighbors_farfeild, nn_distance, matched_raw_idx, matched_graph, matched_s11  = self.find_nearest_neighbor(quary, nbrs)
                loss = self.loss_function(nearest_neighbors_farfeild[0].unsqueeze(0), ff_image[:,self.freq_idx, :, :])
                ff_metrics_target_vs_nn.update(nearest_neighbors_farfeild[0].unsqueeze(0).unsqueeze(0), ff_image[:,self.freq_idx,:,:].unsqueeze(0))
                matrix_distance = torch.mean((matrix.flatten() - nearest_neighbors_mesh_id[0].flatten()))
                losses.append(loss.item())
                total_loss += loss

        if self.Nearest_neighbor_feture == 'calc_nn_from_farfeild':
            print('\ncalculating nn avg loss in farfeild domain (ff->ff)...\n')
            for idx, (graph, ant, ff_image, graph_identifier, example_parameters, raw_index) in tqdm(enumerate(test_loader), desc="runing nn on test dataset", unit=' examples', leave=True):
                flatened_ff_image = [ff_image[:, self.freq_idx,:,:].numpy().flatten()]
                nearest_neighbors_mesh_id, nearest_neighbors_farfeild, nn_distance, matched_raw_idx, matched_graph, matched_s11 = self.find_nearest_neighbor(flatened_ff_image, nbrs)
                freq_idx = self.freq_idx  # 2400 MHz
                farfeild_distance = self.loss_function(nearest_neighbors_farfeild[0].unsqueeze(0), ff_image[:,freq_idx, :, :])
                ff_metrics_target_vs_nn.update(nearest_neighbors_farfeild[0].unsqueeze(0).unsqueeze(0), ff_image[:,self.freq_idx,:,:].unsqueeze(0))
                loss = farfeild_distance
                losses.append(loss.item())
                total_loss += loss

        avg_loss = total_loss /num_examples
        std_loss = torch.std(torch.tensor(losses))

        metrics_result_target_vs_nn = ff_metrics_target_vs_nn.compute()
        print("Overall metrics across all examples (Target vs Nearest Neighbor):")
        for key, value in metrics_result_target_vs_nn.items():
            print(f"  {key}: {value:.6f}")
        # Create a histogram
        plt.hist(losses, bins=150, edgecolor='black')

        # Add labels and title
        plt.xlabel('Loss')
        plt.ylabel('Frequency')
        plt.title('Loss Histogram dB')

        # Show the plot
        plt.show()

        return avg_loss, std_loss

    def calculate_nearest_neighbor_loss(self,train_loader, test_loader, nbrs, max_examples=10000):
        ff_metrics_target_vs_nn = Metrics(config={})
        for idx, (graph, ant, ff_image, graph_identifier, example_parameters, raw_index) in tqdm(enumerate(test_loader), desc="runing nn on test dataset", unit=' examples', leave=True):
            flatened_ff_image = [ff_image[:, self.freq_idx,:,:].numpy().flatten()]
            nearest_neighbors_mesh_id, nearest_neighbors_farfeild, nn_distance, matched_raw_idx, matched_graph, matched_s11 = NN.find_nearest_neighbor(flatened_ff_image, nbrs)
            ff_metrics_target_vs_nn.update(nearest_neighbors_farfeild[0].unsqueeze(0).unsqueeze(0), ff_image[:,self.freq_idx,:,:].unsqueeze(0))
            if idx >= max_examples:
                break
        metrics_result_target_vs_nn = ff_metrics_target_vs_nn.compute()
        print("Overall metrics across all examples (Target vs Nearest Neighbor):")
        for key, value in metrics_result_target_vs_nn.items():
            print(f"  {key}: {value:.6f}")

        return

    def get_example_ff_nearest_neighbor(self, ff_image, nbrs, plot=False):
        flatened_ff_image = [ff_image.flatten()]

        ff_image = torch.tensor(ff_image, dtype=torch.float32)
        ff_metrics_target_vs_nn = Metrics(config={})
        nearest_neighbors_mesh_id, nearest_neighbors_farfeild, nn_distance, matched_raw_idx, matched_graph, matched_s11 = self.find_nearest_neighbor(flatened_ff_image, nbrs)
        ff_metrics_target_vs_nn.update(nearest_neighbors_farfeild[0].unsqueeze(0).unsqueeze(0), ff_image.unsqueeze(0).unsqueeze(0))
        metrics_result_target_vs_nn = ff_metrics_target_vs_nn.compute()
        print("Metrics for the nearest neighbor example:")
        for key, value in metrics_result_target_vs_nn.items():
            print(f"  {key}: {value:.6f}")
        if plot:
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(ff_image.numpy(), cmap='jet', origin='upper', vmin=0, vmax=10)
            plt.title("Query Far-Field")
            plt.colorbar()
            plt.subplot(1, 2, 2)
            plt.imshow(nearest_neighbors_farfeild[0].numpy(), cmap='jet', origin='upper', vmin=0, vmax=10)
            plt.title("Nearest Neighbor Far-Field")
            plt.colorbar()
            plt.show()
        return nearest_neighbors_mesh_id, nearest_neighbors_farfeild, nn_distance, matched_raw_idx, matched_graph, matched_s11, metrics_result_target_vs_nn

    def __call__(self, test_loader):
        if self.Nearest_neighbor_feture == 'calc_nn_from_mesh_identifier':
            nbrs = self.train_nearest_neighbor(self.train_mesh_identfiers)
        if self.Nearest_neighbor_feture == 'calc_nn_from_farfeild':
            np_train_set = [tensor.numpy().flatten() for tensor in self.train_ff_images]
            nbrs = self.train_nearest_neighbor(np_train_set)

        avg_loss, std_loss = self.calculate_nearest_neighbor_loss_on_test_set(test_loader, nbrs)
        print("avg_loss: ", avg_loss, 'std_loss: ', std_loss)
        return avg_loss, std_loss


if __name__ == '__main__':

    import torch_geometric.transforms as T
    _REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    arg_parser = ArgParser(config_file=os.path.join(_REPO_ROOT, 'configs', 'inverse', 'nearest_neighbor.yaml'))
    config = arg_parser.get_config()

    # Runtime paths come from the yaml config:
    #   data_root, split_dir, manual_targets_dir,
    #   surrogate_checkpoint (only when load_trained_model is True),
    #   checkpoints_path (only when load_from_checkpoint is True)
    for required_key in ('data_root', 'split_dir'):
        if config.get(required_key) is None:
            raise ValueError(f"config['{required_key}'] must be set in the yaml config file")
    config['number_of_validation_samples'] = 15
    config['load_precomputed_nn'] = False
    config['data_set_type'] = 'pixel_data'
    config['k_sphere_neighbores'] = 2
    config['num_sphere_nodes'] = 162
    config['augment_shift'] = 0 # 2
    config['augment_rotation'] = False
    config['augment_compress_towards_threshold'] = False
    config['sphere_radius'] = 400
    config['frequencies'] = [2400, 2800, 5200, 5600,6000]
    device_num = '0'
    config['threshold'] = 0.5
    os.environ["CUDA_VISIBLE_DEVICES"] = device_num
    cuda_device = "cuda:" + device_num
    config['learning_rate'] = 0.0001 # the currnt standerd 0.001 HDGCNN 0.00001
    config['path_to_save_model_weights'] = 'GNN_ff_foward_isosphere_max_' + str(config['data_set_type']) +'_node_prob'+'_34_34_2400.pt'
    config['radiation_image_shape'] = [34,34,1]
    config['raw_node_feture_size'] = 17
    config['weight_decay'] = 0.0001
    config['add_sphere'] = False
    config['batch_size'] = 32 # 32 #use: 48 # 16 for HDGCNN
    config['adapt_output_size'] = True
    config['idx_freq'] = 3
    config['add_radius_graph_edges'] = True # False for meshgraphnets
    config['model_type'] = 'HDGCNN' # 'MeshGraphNet' 'HDGCNN'
    config['use_node_probs'] = False
    config['physical_antenna'] = True
    config['gps'] = True

    config['hetero_graph'] = False
    config['predict_surface_current'] = True

    # load from checkpoint?
    config['load_from_checkpoint'] = False
    if config.get('checkpoints_path') is None:
        config['checkpoints_path'] = os.path.join(config['data_root'], 'checkpoints/')
    config['load_trained_model'] = False
    # config['surrogate_checkpoint'] comes from the yaml (used when load_trained_model is True)
    config['dataset'] = 'FMNIST_CIFAR' # options: 'FMNIST_CIFAR', 'FMNIST'
    config['split_type'] = 'hardest_indices'#'easy_indices' # options: 'pca_extrapolation_split', 'farfield_split_with_buckets', 'random_split', 'hardest_indices'
    ###########################################################
    # 1) dataset and loaders:
    pe_transform = T.Compose([
        T.ToUndirected(),
        T.AddLaplacianEigenvectorPE(k=10, attr_name='pe')
    ])


    # set seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    from src.dataset.datasets import dataset_random
    combined_dataset, train_dataset, test_dataset = dataset_random(config, pe_transform, dataset_name='FMNIST_CIFAR')


    # create loaders with num_workers and pin_memory for performance
    num_cpus = os.cpu_count() or 4
    num_workers = min(8, max(1, num_cpus // 2))
    from torch_geometric.loader import DataLoader

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                            follow_batch=['pos_surface_current'],
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True,
                            prefetch_factor=2,)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             follow_batch=['pos_surface_current'],
                            )
    loss_function = torch.nn.MSELoss()

    config['nn_type'] = 'single_ff_example' # options: 'calc_nn_from_mesh_identifier', 'calc_nn_from_farfeild', 'single_ff_example'

    NN = AntennaNearestNeighbor(train_loader,freq_idx=config['idx_freq'], Nearest_neighbor_feture='calc_nn_from_mesh_identifier', loss_function=loss_function)
    if config['nn_type'] == 'calc_nn_from_mesh_identifier':
        # for calc_nn_from_mesh_identifier:
        NN.calculate_nearest_neighbor_loss_on_test_set(test_loader, NN.train_nearest_neighbor(NN.train_mesh_identfiers))

    if config['nn_type'] == 'calc_nn_from_farfield':
        # for calc_nn_from_farfeild:
        np_train_set = [tensor.numpy().flatten() for tensor in NN.train_ff_images]
        nbrs = NN.train_nearest_neighbor(np_train_set)
        NN.calculate_nearest_neighbor_loss(train_loader, test_loader, nbrs=nbrs)

    if config['nn_type'] == 'single_ff_example':
        np_train_set = [tensor.numpy().flatten() for tensor in NN.train_ff_images]
        nbrs = NN.train_nearest_neighbor(np_train_set)

        # Hand-designed target far-fields (Fig. 3) live in manual_targets_dir.
        path_to_samples_dir = config.get('manual_targets_dir')
        if path_to_samples_dir is None:
            raise ValueError("config['manual_targets_dir'] must be set in the yaml config file for nn_type 'single_ff_example'")

        sample_name = 'batch_8_lin_0.pt'
        path_to_sample = os.path.join(path_to_samples_dir, sample_name)
        ff_image = torch.load(path_to_sample,weights_only=False)

        log_scale = False
        if log_scale:
            #return to linear scale
            linear_image = torch.pow(10, ff_image / 10)
            ff_image = linear_image

        nearest_neighbors_mesh_id, nearest_neighbors_farfeild, nn_distance, matched_raw_idx, matched_graph, matched_s11, metrics_result_target_vs_nn = NN.get_example_ff_nearest_neighbor(ff_image, nbrs)


        # create a directory for diffusion outputs if it doesn't exist
        diffusion_outputs_dir = os.path.join(path_to_samples_dir, 'diffusion_outputs/')
        if not os.path.exists(diffusion_outputs_dir):
            os.makedirs(diffusion_outputs_dir)
        avrage_metrics ={'mse': 0, 'mae': 0, 'psnr': 0, 'ssim': 0, 'mssim': 0}
        for sample_name in os.listdir(path_to_samples_dir):
            if not sample_name.endswith('.pt'):
                continue
            path_to_sample = os.path.join(path_to_samples_dir, sample_name)
            ff_image = torch.load(path_to_sample,weights_only=False)

            log_scale = False
            if log_scale:
                #return to linear scale
                linear_image = torch.pow(10, ff_image / 10)
                ff_image = linear_image

            nearest_neighbors_mesh_id, nearest_neighbors_farfeild, nn_distance, matched_raw_idx, matched_graph, matched_s11, metrics_result_target_vs_nn = NN.get_example_ff_nearest_neighbor(ff_image, nbrs)
            # save the nearest neighbor metrics to a text file
            with open(os.path.join(diffusion_outputs_dir, sample_name + '_nn_metrics.txt'), 'w') as f:
                f.write(f"Metrics for the nearest neighbor example:\n")
                for key, value in metrics_result_target_vs_nn.items():
                    f.write(f"  {key}: {value:.6f}\n")
            for key, value in metrics_result_target_vs_nn.items():
                avrage_metrics[key] += value
        num_samples = len([name for name in os.listdir(path_to_samples_dir) if name.endswith('.pt')])
        print(f"Avrage metrics across {num_samples} examples:")
        for key, value in avrage_metrics.items():
            avrage_metrics[key] = value / num_samples
            print(f"  {key}: {avrage_metrics[key]:.6f}")


    print('finished nearest neighbor evaluation')
