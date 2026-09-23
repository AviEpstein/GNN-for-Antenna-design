import os
import numpy as np
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm

from configs.parser import ArgParser
from src.metrics.ff_metrics import Metrics
from src.dataset.pixel_dataloader import PixelDataset
from src.inverse.nearest_neighbor import AntennaNearestNeighbor
from src.diffusion.smooth_binarize import smooth_binarize



def create_extrapolation_split_pca(dataset, freq_idx=3, split_ratio=0.9):
    """
    Creates an Out-of-Distribution (OOD) train-test split by isolating the 
    most extreme farfield examples into the test set using PCA.
    
    Args:
        dataset (PixelDataset): The dataset instance.
        split_ratio (float): Ratio of the dataset to use for training (default: 0.9).

    Returns:
        dict: A dictionary containing 'train' and 'test' indices.
    """
    num_samples = len(dataset)
    farfields_flat = []
    
    print(f"Extracting farfields from {num_samples} examples for PCA...")
    # 1. Iterate through the dataset and extract the farfields
    for i in tqdm(range(num_samples), desc="Processing farfields"):
        # The get() method returns: 
        # graph_full_data, s11, farfeilds, surface_currents, example_paramters, raw_idx
        batch = dataset[i]
        farfields = batch[2] 
        farfield = farfields[freq_idx]  # Select the farfield for the specified frequency index
        
        # 2. Flatten the farfield tensor into a 1D vector and store
        if isinstance(farfield, torch.Tensor):
            farfields_flat.append(farfield.detach().cpu().numpy().flatten())
        else:
            farfields_flat.append(np.array(farfield).flatten())
            
    # 3. Stack into a single matrix (Shape: num_samples x flattened_features)
    data_matrix = np.vstack(farfields_flat)
    
    print("Fitting PCA to find the primary axis of variance...")
    # 4. Fit PCA to find the 1st Principal Component
    pca = PCA(n_components=1)
    projections = pca.fit_transform(data_matrix).flatten()
    
    # 5. Calculate distance from the central cluster (mean projection)
    mean_projection = np.mean(projections)
    distances_from_center = np.abs(projections - mean_projection)
    
    # 6. Sort indices based on their distance from the center
    # The smallest distances are the most "normal" (Train)
    # The largest distances are the most "anomalous" (Test)
    sorted_indices = np.argsort(distances_from_center)
    
    split_idx = int(split_ratio * num_samples)
    
    train_indices = sorted_indices[:split_idx].tolist()
    test_indices = sorted_indices[split_idx:].tolist()
    
    print(f"Split complete! Train: {len(train_indices)} | Test (OOD): {len(test_indices)}")
    
    return {"train": train_indices, "test": test_indices}




def evaluate_ood_split(
    train_loader: PixelDataset,
    test_loader: PixelDataset,
    device: str = "cuda",
    freq_idx: int = 3,
):
    """
    Evaluates the OOD split by computing nearest neighbor distances and metrics.
    """

    metrics = Metrics(config={})


    # (Optional) You can normalize here, e.g. to unit norm or mean/std.
    # For plain MSE distances, you can skip normalization.
    # Example L2 norm normalization (comment out if not wanted):
    # Y_flat = Y_flat / (Y_flat.norm(dim=1, keepdim=True) + 1e-8)


    loss_function = torch.nn.MSELoss()
    NN = AntennaNearestNeighbor(train_loader,freq_idx=freq_idx, Nearest_neighbor_feture='calc_nn_from_farfeild', loss_function=loss_function)
    np_train_set = [tensor.numpy().flatten() for tensor in NN.train_ff_images]
    nbrs = NN.train_nearest_neighbor(np_train_set)
    # save the nearest neighbor model:
    torch.save(nbrs, os.path.join(config['path_to_root_data_folder'], 'ff_nearest_neighbor_model.pth'))

    dmin_test = []
    test_idx_perm = []
    raw_index_list = []

    for idx, (graph, s11, ff_image, graph_identifier, example_parameters, raw_index) in tqdm(enumerate(test_loader), desc="evaluating pca ff split dataset", unit=' examples', leave=True):
        flatened_ff_image = [ff_image[:, freq_idx, :, :].numpy().flatten()]
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
    metrics_result = metrics.compute()
    print("Nearest neighbor metrics on test set:")
    for key, value in metrics_result.items():
        print(f"  {key}: {value:.6f}")


if __name__ == "__main__":
    from torch_geometric.loader import DataLoader
    import torch_geometric.transforms as T

    arg_parser = ArgParser()
    config = arg_parser.get_config()
    working_directory = os.getcwd() # ..../GNN-for-Antenna-design/
    parent_dir = os.path.dirname(working_directory) # .../git
    data_dir = parent_dir + '/data_sets' #  .../git/data_sets
    # dataset roots come from the config/CLI (--path_to_root_data_folder / --path_to_testset)
    config['number_of_validation_samples'] = 15
    config['load_precomputed_nn'] = False
    config['data_set_type'] = 'pixel_data'
    config['k_sphere_neighbores'] = 2
    config['num_sphere_nodes'] = 162
    config['augment_shift'] = 0 # 2    
    config['augment_rotation'] = False
    config['augment_compress_towards_threshold'] = False
    config['sphere_radius'] = 400
    config['frequencies'] = [2400, 2800, 5200, 5600,6000] # [2400, 2800, 5200, 5600, 6000,8500,10000,'1.2e+4','1.5e+4']
    device_num = '0'
    config['threshold'] = 0.5
    os.environ["CUDA_VISIBLE_DEVICES"] = device_num
    cuda_device = "cuda:" + device_num
    config['learning_rate'] = 0.001 # the currnt standerd 0.001 HDGCNN 0.00001
    config['path_to_save_model_weights'] = 'GNN_ff_foward_isosphere_max_' + str(config['data_set_type']) +'_node_prob'+'_34_34_2400.pt'
    config['radiation_image_shape'] = [34,34,1]
    config['raw_node_feture_size'] = 17
    config['weight_decay'] = 0.0001
    config['add_sphere'] = False
    config['batch_size'] = 48 # 32 #use: 48 # 16 for HDGCNN
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
    config['checkpoints_path'] = config['path_to_root_data_folder'] + '/checkpoints/'
    config['load_trained_model'] = False
    
    ###########################################################
    # 1) dataset and loaders:
    pe_transform = T.Compose([
        T.ToUndirected(),
        T.AddLaplacianEigenvectorPE(k=10, attr_name='pe')
    ])
    # set seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)


    # for seperate train and test folders (train root from --path_to_root_data_folder,
    # test root from --path_to_testset; both are existing config keys):
    train_root = config['path_to_root_data_folder']
    train_dataset = PixelDataset(train_root, config=config, pre_transform=pe_transform)
    test_root = config.get('path_to_testset', config['path_to_root_data_folder'])
    test_dataset = PixelDataset(test_root, config=config, pre_transform=pe_transform)

    # combine the 2 datasets and make a new split of train and test:
    combined_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
    


    # # Save the split indices to a file
    split_save_path = config.get('path_to_root_data_folder') + 'pca_extrapolation_split_new.pth'
    if os.path.exists(split_save_path):
        print(f"Split file already exists at {split_save_path}. Loading existing split.")
    else:
        # # Assuming 'dataset' is already initialized
        print("Creating OOD split...")
        split_indices = create_extrapolation_split_pca(combined_dataset, split_ratio=0.9)
        torch.save(split_indices, split_save_path)
        print(f"OOD split saved to {split_save_path}") 

    # load the split indices from the file (for verification)
    split = torch.load(split_save_path)

    train_dataset = torch.utils.data.Subset(combined_dataset, split['train'])
    test_dataset = torch.utils.data.Subset(combined_dataset, split['test'])
    print(f"Combined dataset size: {len(combined_dataset)}, Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
    
    num_cpus = os.cpu_count() or 4
    num_workers = min(8, max(1, num_cpus // 2))
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=False, follow_batch=['pos_surface_current'],
                              num_workers=num_workers, pin_memory=True, prefetch_factor=2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, follow_batch=['pos_surface_current'],)
    print("Evaluating OOD split...")
    split = evaluate_ood_split(train_loader, test_loader, device=cuda_device, freq_idx=config['idx_freq'])
    print("OOD split evaluation complete!")

