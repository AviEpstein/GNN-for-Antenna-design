



import os

import torch
from tqdm import tqdm

from configs.parser import ArgParser
from src.dataset.datasets import dataset_pca
from src.dataset.pixel_dataloader import PixelDataset
from src.metrics.ff_metrics import Metrics
from src.inverse.nearest_neighbor import  AntennaNearestNeighbor
from src.diffusion.smooth_binarize import smooth_binarize
from torch_geometric.loader import DataLoader

import numpy as np


def create_N_hardest_split(train_loader: PixelDataset, test_loader: PixelDataset, N: int, freq_idx: int = 2):
    
    loss_function = torch.nn.MSELoss()
    metrics = Metrics(config={})
    device =  "cuda" if torch.cuda.is_available() else "cpu"

    # Create a nearest neighbor search object for the dataset
    NN = AntennaNearestNeighbor(train_loader,freq_idx=freq_idx, Nearest_neighbor_feture='calc_nn_from_farfeild', loss_function=loss_function)
    np_train_set = [tensor.numpy().flatten() for tensor in NN.train_ff_images]
    nbrs = NN.train_nearest_neighbor(np_train_set)


    dmin_test = []
    test_idx_perm = []
    raw_index_list = []
    example_paths = []

    # Calculate the nearest neighbor distance for each example in the dataset
    for idx, (graph, s11, ff_image, graph_identifier, example_parameters, raw_index) in tqdm(enumerate(test_loader), desc="creating ff split dataset", unit=' examples', leave=True):
        flatened_ff_image = [ff_image[:,freq_idx, :, :].numpy().flatten()]
        # if s11['s11_abs_db'][0][800]> -4: # 100 is for frequancy 2400 ,800 for frequancy 5200
        #     continue
        matrix = smooth_binarize(example_parameters['ant_parameters'][0][0], 300, 300).numpy()
        # if np.sum(matrix) < 50:
        #     continue

        nearest_neighbors_mesh_id, nearest_neighbors_farfeild, nn_distance, matched_raw_idx, matched_graph, matched_s11 = NN.find_nearest_neighbor(flatened_ff_image, nbrs)
        farfeild_distance = loss_function(nearest_neighbors_farfeild[0].unsqueeze(0), ff_image[:, freq_idx, :, :])
        metrics.update(nearest_neighbors_farfeild[0].unsqueeze(0).unsqueeze(0).to(device), ff_image[:, freq_idx, :, :].unsqueeze(0).to(device))
        dmin_test.append(farfeild_distance)
        test_idx_perm.append(idx)
        raw_index_list.append(raw_index)
        # example_path = graph.file_path 
        # example_paths.append(example_path)

    # Get the indices of the N examples with the largest nearest neighbor distance (hardest examples)
    hardest_indices = torch.argsort(torch.tensor(dmin_test), descending=True)[:N].tolist()
    # print the smallest loss in hardest indecies:
    print("Smallest nearest neighbor distance among the hardest examples: ", torch.tensor(dmin_test)[hardest_indices].min().item())

    metrics_result = metrics.compute()
    print("Nearest neighbor metrics on test set:")
    for key, value in metrics_result.items():
        print(f"  {key}: {value:.6f}")

    #claculate the hardest_indices metrics:
    print("Hardest examples nearest neighbor metrics:")
    hard_metrics = Metrics(config={})
    
    # We will get elements from the dataset which don't have a batch dimension. 
    # But since the previous loop used the dataloader we need to make sure we unsqueeze if needed,
    # or just convert our test_loader to index its dataset properly.
    for idx in hardest_indices:
        graph, s11, ff_image, graph_identifier, example_parameters, raw_index = test_loader.dataset[idx]
        
        # Add batch dimension to match DataLoader output
        if isinstance(ff_image, torch.Tensor):
            ff_image = ff_image.unsqueeze(0)
            
        flatened_ff_image = [ff_image[:, freq_idx, :, :].numpy().flatten()]
        nearest_neighbors_mesh_id, nearest_neighbors_farfeild, nn_distance, matched_raw_idx, matched_graph, matched_s11 = NN.find_nearest_neighbor(flatened_ff_image, nbrs)
        farfeild_distance = loss_function(nearest_neighbors_farfeild[0].unsqueeze(0), ff_image[:, freq_idx, :, :])
        hard_metrics.update(nearest_neighbors_farfeild[0].unsqueeze(0).unsqueeze(0).to(device), ff_image[:, freq_idx, :, :].unsqueeze(0).to(device))
        # print(f"Example {idx} (raw index {raw_index_list[idx]}): Nearest neighbor distance: {farfeild_distance.item():.6f}")

    print("Hardest ", N ," examples nearest neighbor metrics:")
    hard_metrics_result = hard_metrics.compute()
    for key, value in hard_metrics_result.items():
        print(f"  {key}: {value:.6f}")

    

    

    return hardest_indices




if __name__ == "__main__":


    # set up dataset and nearest neighbor search:
    arg_parser = ArgParser()
    config = arg_parser.get_config()
    config['num_sphere_nodes'] = 162
    config['add_sphere'] = False
    config['radiation_image_shape'] = [34,34,1]
    frequency = 5600  # 5.6 GHz
    freq_idx = 3  # Choose frequency index for far-field comparison
    config['idx_freq'] = freq_idx
    config['batch_size'] = 32
    config['frequencies'] = [2400, 2800, 5200, 5600,6000]
    # the dataset root comes from the config/CLI (--path_to_root_data_folder)
    import torch_geometric.transforms as T

    pe_transform = T.Compose([
        T.ToUndirected(),
        T.AddLaplacianEigenvectorPE(k=10, attr_name='pe')
    ])


    # set seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)

    config['dataset'] = 'FMNIST_CIFAR'
    config['split_dir'] = config.get('split_dir', os.path.join('splits', config['dataset']) + '/')
    combined_dataset, train_dataset, test_dataset = dataset_pca(config, pe_transform, dataset_name='FMNIST_CIFAR')


    
    num_cpus = os.cpu_count() or 4
    num_workers = min(8, max(1, num_cpus // 2))
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                            follow_batch=['pos_surface_current'], 
                            shuffle=False, 
                            num_workers=num_workers,
                            pin_memory=True,
                            # persistent_workers=True,
                            prefetch_factor=2,)
                            # drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, 
                             follow_batch=['pos_surface_current'],
                            # num_workers=num_workers,
                            # pin_memory=True,
                            # persistent_workers=True,
                            # prefetch_factor=2,
                            )

    print('Calculating the hardest examples based on nearest neighbor distance...')
    hardesrt_indices = create_N_hardest_split(train_loader, test_loader, N=100, freq_idx=freq_idx)
    
    print("hardesrt_indices: ", hardesrt_indices)
    # save the hardest indices to a file:
    torch.save(hardesrt_indices, config['split_dir'] + 'hardest_indices_100.pth')
    print("Saved hardest indices to: ", config['split_dir'] + 'hardest_indices_100.pth')
