from data.generation.create_classic_rectangle_patch import create_parameter_dict
from src.geometry.create_pixel_antenna import add_reflector_to_graph, create_pixel_ant, create_reflector_matrix
from pathlib import Path
from torch_geometric.data import Dataset
import os
import pickle
import torch
from src.dataset.dataloader_utils import get_data_idx_to_example_number_map, get_example_number_to_idx_map, get_example_number, get_icosphere, load_obj_as_graph
from src.dataset.augmentation import RotateTransform3D, ShiftTransform3D, compress_towards_threshold, augment_input_matrix
import time
import os.path as osp
from src.dataset.dataloader_utils import resize_farfeild, normalize_gain
import random
import pandas as pd
from src.diffusion.smooth_binarize import smooth_binarize
from src.geometry.utils import read_s11_pickle, get_s11_single_freq
import numpy as np
from src.geometry.mesh_functions import decompose_pyg_graph
from src.geometry.mesh_functions_pytorch_2 import merge_and_connect_graphs_from_dict
from src.graph.surface_current_functions import process_surface_current_downsample
import trimesh
from src.graph.prepared_graph_cache import prepare_graph_cached
from src.graph.graph_functions import graph_dict_merging_duplicate_nodes, split_disconnected_graphs_pytorch

class PixelDataset(Dataset):
    def __init__(self, root, config, transform=None, pre_transform=None, pre_filter=None,):
        self.config = config
        super().__init__(root, transform, pre_transform, pre_filter)  
        """
        Args:
            root of the data set
        """
        self.data_idx_to_example_number_map = get_data_idx_to_example_number_map(root)
        self.data_example_number_to_idx_map  = get_example_number_to_idx_map(self.data_idx_to_example_number_map)
        if config.get('augment_rotation', False):
            self.rotate_transform = RotateTransform3D()
        if config.get('augment_shift', False):
            self.shift_transform = ShiftTransform3D(config['augment_shift'])
        self.frequencies = config.get('frequencies')
        self.frequencies_list_idxs = [0, 1, 2, 3, 4]#,5,6,7,8]  #[config['frequencies'].index(freq) for freq in self.frequencies if freq in config['frequencies']]

        self.dataset_id = str(int(time.time()))
        
        
        
    @property
    def raw_file_names(self):
        """
        iterate over 'raw' directory and grab all the mesh files, and their corisponding simulation result
        """
        mesh_paths = Path(self.raw_dir).glob('meshes/*')
        mesh_paths = [mesh_path for mesh_path in mesh_paths]
        return mesh_paths
        
    @property
    def processed_file_names(self):
        processed_files = Path(self.processed_dir).glob('processed_data_*.pt')
        file_name_list = [processed_file for processed_file in processed_files ]
        # Sort by the integer index at the end of the filename before .pt
        def extract_index(path):
            # Assumes filename like 'processed_data_123.pt'
            stem = path.stem
            idx_str = stem.split('_')[-1]
            try:
                return int(idx_str)
            except ValueError:
                return -1
        file_name_list.sort(key=extract_index)
        
        # file_name_list = []
        self.processed_file_paths = file_name_list

        return file_name_list

    def download(self):
            # Download to `self.raw_dir`.
            ...
            
    def process(self):
        idx = 0
        start_time = time.time()  
        raw_paths = [raw_path for raw_path in self.raw_paths]
        def extract_index(path):
            # Assumes filename like 'processed_data_123.pt'
            idx_str = path.split('/')[-1]
            try:
                return int(idx_str)
            except ValueError:
                return -1
        raw_paths.sort(key=extract_index)
        for raw_path in raw_paths:
            self.process_mesh(raw_path)
        end_time = time.time()
        total_time = end_time - start_time
        print("Total Time:", total_time, "seconds")
            
    def len(self):
        # return int(len(self.processed_file_names)/2)-1
        return len(self.processed_file_paths)

    def processed_path_exists(self, raw_path):
        """
        Given a raw path to a mesh, check if its processed path exists.
        raw_path example: <root>/raw/meshes/1
        processed_path example: <root>/processed/processed_data_1.pt
        """
        # Extract the file number from the raw path
        raw_filename = osp.basename(raw_path)
        
        # Form the corresponding processed path
        processed_filename = f'processed_data_{raw_filename}.pt'
        processed_path = osp.join(self.processed_dir, processed_filename)
        
        # Check if the processed path exists
        return osp.exists(processed_path)
    
    def process_mesh(self,raw_path):
        try:
            if self.processed_path_exists(raw_path):
                return
            example_number = get_example_number(raw_path)
            # create the pixel data and center it:
            threshold = self.config.get('threshold', 0.5)
            pixel_data, pixel_probs = self.load_pixel_data(raw_path, threshold)
            PyG_data = pixel_data
            # PyG_data = self.add_sphere_around_graph(pixel_data)
            
            if self.pre_transform is not None:
                PyG_data = self.pre_transform(PyG_data)
            result_path = self.convert_mesh_path_to_result_path(raw_path)
            farfeilds = []
            surface_currents = []
            pos_surface_current = []
            
            for freq in self.config.get('frequencies'):
                farfeild = self.get_farfeild(result_path, freq)
                farfeild = resize_farfeild(farfeild, self.config.get('radiation_image_shape')[:2]).unsqueeze(0)
                normlized_gain_farfeild = normalize_gain(farfeild[0][:,:,0],farfeild[0][:,:,1], img_shape=self.config.get('radiation_image_shape')[:2])
                farfeilds.append(normlized_gain_farfeild)
                
                # get suface current data:
                surface_current_data, pos_surface_current_data = self.get_surface_current(result_path, freq)
                surface_currents.append(surface_current_data)
                pos_surface_current.append(pos_surface_current_data)

            farfeilds = torch.stack(farfeilds)
            surface_currents = torch.stack(surface_currents)
            pos_surface_current = torch.stack(pos_surface_current)
            s11 = self.get_s11(result_path)
            example_paramters = self.get_example_paramters(raw_path)
                
            processed_example = {'graph':PyG_data, 'farfeilds':farfeilds, 'surface_currents':surface_currents, 'pos_surface_current':pos_surface_current, 's11':s11, 'example_paramters':example_paramters}
            processed_path = osp.join(self.processed_dir, f'processed_data_{example_number}.pt')
            torch.save(processed_example, processed_path)
        
        except:
            print('failed to process example: ', example_number)
            
        
    @staticmethod
    def create_split(dataset, split_ratio=0.9, seed=None):
        """
        Create a random train-test split based on the given ratio.
        Args:
            dataset (Dataset): The dataset to split.
            split_ratio (float): Ratio of the dataset to use for training.
            seed (int, optional): Seed for reproducibility of random split.

        Returns:
            dict: A dictionary containing 'train' and 'test' indices.
        """
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        if seed is not None:
            random.seed(seed)
        random.shuffle(indices)

        split_idx = int(dataset_size * split_ratio)
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        return {"train": train_indices, "test": test_indices}

    @staticmethod
    def save_split(data_idx_to_example_number_map , split_indices, csv_path):
        """
        Save the split indices to a CSV file.
        """
        split_data = []
        for split_name, indices in split_indices.items():
            for idx in indices:
                split_data.append({"split": split_name, "index": idx, 'example_number':  data_idx_to_example_number_map[idx] })
        pd.DataFrame(split_data).to_csv(csv_path, index=False)
        print(f"Dataset split saved to {csv_path}")

    @staticmethod
    def load_split(csv_path):
        """
        Load the split indices from a CSV file.

        Args:
            csv_path (str): Path to the CSV file containing the saved split.

        Returns:
            dict: A dictionary with 'train' and 'test' splits containing indices.
        """
        df = pd.read_csv(csv_path)
        split_indices = {split_name: df[df['split'] == split_name]['index'].tolist()
                         for split_name in df['split'].unique()}
        print(f"Dataset split loaded from {csv_path}")
        return split_indices
    
    @staticmethod
    def load_split_by_raw_idx(csv_path, data_example_number_to_idx_map):
        df = pd.read_csv(csv_path)
        df['index'] = df["ExampleNumber"].astype(str).map(data_example_number_to_idx_map).astype(int)
        
        split_indices = {split_name: df[df['Type'] == split_name]['index'].tolist()
                    for split_name in df['Type'].unique()}
        return split_indices
    
    # @staticmethod
    def load_pixel_data(self, example_path, threshold):
        """
        Load the pixel data from the given example path and center it.
        Args:
            example_path (str): Path to the example folder.
            threshold (float): Threshold for the antenna parameters.
        Returns:
            torch_geometric.data.Data: The loaded pixel data.
        """
        paramspath = example_path + '/matrix_and_env_dict.pkl'
        if os.path.exists(paramspath):
            with open(paramspath, 'rb') as f:
                params = pickle.load(f)
            matrix = torch.tensor(params['reflectors_dict']['matrix'])
            size_of_patch_in_mm = params['reflectors_dict']['patch_x']
            size_of_ground = params['reflectors_dict']['ground_x']
            size_of_FR4_in_mm = size_of_ground
            height = params['reflectors_dict']['h']
            if 'thetas' in params and 'phis' in params:
                reflectors_dict = {'thetas': params['thetas'], 'phis': params['phis'], 
                                    'radius': params['radius'], 'box_size': params['box_size'],
                                    'num_of_reflectors': params['num_of_reflectors']}
                if params['thetas'] is not None and params['phis'] is not None:
                    pixel_data_with_reflectors = True
                else:
                    reflectors_dict = {'thetas': -1, 'phis': -1, 
                    'radius': -1, 'box_size': -1,
                    'num_of_reflectors': 0}
                    pixel_data_with_reflectors = False
            else:
                pixel_data_with_reflectors = False
            if pixel_data_with_reflectors:
                pixel_data, pixel_probs = create_pixel_ant(matrix, threshold, size_of_patch_in_mm , size_of_FR4_in_mm, size_of_ground, height, reflectors_dict=reflectors_dict )
            else:
                pixel_data, pixel_probs =  create_pixel_ant(matrix, threshold, size_of_patch_in_mm , size_of_FR4_in_mm, size_of_ground, height, create_physical_pixel_mesh = True )
            
            if 'reflector_matrix' in params['reflectors_dict'].keys():
                reflector_matrix = torch.tensor(params['reflectors_dict']['reflector_matrix'])
                reflector = create_reflector_matrix(reflector_matrix, params['reflectors_dict'], threshold)
                graph_dict = decompose_pyg_graph(pixel_data)
                graph_dict['Reflector'] = reflector
                pixel_data =  merge_and_connect_graphs_from_dict(graph_dict, 'sphere', k=2, add_sphere=False)
            
            if self.config.get('merge_duplicate_nodes'):
                graph_dict = decompose_pyg_graph(pixel_data)
                graph_dict = graph_dict_merging_duplicate_nodes(graph_dict)
                pixel_data = merge_and_connect_graphs_from_dict(graph_dict, 'sphere', k=2, add_sphere=False)
            return pixel_data, pixel_probs
        
        pred_ant_matrix_path = example_path + '/pred_antenna_matrix.pt'
        if os.path.exists(pred_ant_matrix_path):
            ant_params = torch.load(pred_ant_matrix_path)
            base_matrix = ant_params[0]
            reflector_matrix = ant_params[1]
            env_dict = create_parameter_dict(self.config)
            pixel_data, pixel_probs =  create_pixel_ant(base_matrix, threshold, size_of_patch_in_mm=env_dict['patch_x'], size_of_FR4_in_mm=env_dict['ground_x'], size_of_ground=env_dict['ground_x'], height=env_dict['h'], create_physical_pixel_mesh = True )
            antenna = add_reflector_to_graph(pixel_data, reflector_matrix, env_dict)
            antenna = antenna.cpu()
            return antenna, pixel_probs


        ant_prams_path = example_path + '/ant_parameters.pickle'
        env_prams_path = example_path + '/model_parameters.pickle'
        with open(ant_prams_path, 'rb') as f:
            ant_prams = pickle.load(f)
        with open(env_prams_path, 'rb') as f:
            env_prams = pickle.load(f) 
        # some have the threshold with the ma
        if len(ant_prams) == 2:
            matrix =  ant_prams[0]
            threshold = ant_prams[1]
        if len(ant_prams) == 4:
            matrix =  ant_prams['matrix']
            threshold = ant_prams['threshold']

            # check if the reflectors are in the antenna parameters
            if 'phis' in ant_prams and 'thetas' in ant_prams and 'radius' in env_prams and 'box_size' in env_prams:
                reflectors_dict = {'thetas': ant_prams['thetas'], 'phis': ant_prams['phis'],
                                    'radius': env_prams['radius'], 'box_size': env_prams['box_size'],
                                    'num_of_reflectors': env_prams['num_of_reflectors']}
                if ant_prams['thetas'] is not None and ant_prams['phis'] is not None:
                    pixel_data_with_reflectors = True
                else:
                    reflectors_dict = {'thetas': -1, 'phis': -1,
                    'radius': -1, 'box_size': -1,
                    'num_of_reflectors': 0}
                    pixel_data_with_reflectors = False

                    # reflectors_dict = {'thetas': ant_prams['thetas'], 'phis': ant_prams['phis'],
                    #                 'radius': env_prams['radius'], 'box_size': env_prams['box_size'],
                    #                 'num_of_reflectors': env_prams['num_of_reflectors']}

        else:
            matrix = ant_prams
        if isinstance(matrix, torch.Tensor):
            matrix = matrix.detach()
        size_of_patch_in_mm = env_prams['patch_x']
        size_of_ground = env_prams['ground_x']
        size_of_FR4_in_mm = size_of_ground
        height = env_prams['h']

        if pixel_data_with_reflectors:
            pixel_data, pixel_probs = create_pixel_ant(matrix, threshold, size_of_patch_in_mm , size_of_FR4_in_mm, size_of_ground, height, reflectors_dict=reflectors_dict, create_physical_pixel_mesh=self.config.get('physical_antenna') )
        else:
            pixel_data, pixel_probs =  create_pixel_ant(matrix, threshold, size_of_patch_in_mm , size_of_FR4_in_mm, size_of_ground, height )
        
        if self.config.get('merge_duplicate_nodes'):
            graph_dict = decompose_pyg_graph(pixel_data)
            graph_dict = graph_dict_merging_duplicate_nodes(graph_dict)
            pixel_data = merge_and_connect_graphs_from_dict(graph_dict, 'sphere', k=2, add_sphere=False)
        return pixel_data.cpu(), pixel_probs.cpu()
    
    def add_sphere_around_graph(self, PyG_data):
        graph_dict = decompose_pyg_graph(PyG_data)
        vertices, faces = get_icosphere(num_sphere_vertices=self.config.get('num_sphere_nodes'),radius=self.config.get('sphere_radius'))
        sphere_mesh  = trimesh.Trimesh(vertices=vertices, faces=faces)
        graph_dict['sphere'] = load_obj_as_graph(sphere_mesh, 'sphere')
        pyg_graph = merge_and_connect_graphs_from_dict(graph_dict, 'sphere', k=2)
        return pyg_graph
    
    @staticmethod
    def convert_mesh_path_to_result_path(mesh_path):
        result_path = mesh_path.replace('meshes', 'CST_results')
        return result_path
    
    @staticmethod
    def get_farfeild(result_path:str, frequency = 2400):
        farfeild_path = result_path + '/farfield_' + str(frequency) + '.npy'
        farfield = np.load(farfeild_path)
        return torch.tensor(farfield,dtype=torch.float32)
    
    @staticmethod
    def get_s11(result_path:str):
        s11_path = result_path +'/S_parameters.pickle'
        s11 = read_s11_pickle(s11_path)
        return s11
   
    @staticmethod
    def get_surface_current(result_path:str, frequency = 2400):
        surface_current_path = result_path + '/surface current ' +'(f='+str(frequency)+') [1].pkl'
        with open(surface_current_path, 'rb') as f: 
            surface_current = pickle.load(f)
        # surface_current0, pos_surface_current0 = process_surface_current(surface_current)
        surface_current, pos_surface_current = process_surface_current_downsample(surface_current)
        return surface_current, pos_surface_current
    
    @staticmethod
    def get_example_paramters(raw_path:str):
        paramspath = raw_path + '/matrix_and_env_dict.pkl'
        if os.path.exists(paramspath):
            with open(paramspath, 'rb') as f:
                params = pickle.load(f)
            params['reflectors_dict']['threshold'] = params.get('threshold', 0.5)
            return {'ant_parameters': params['reflectors_dict']}
        pred_matrix_path = raw_path + '/pred_antenna_matrix.pt'
        if os.path.exists(pred_matrix_path):
            ant_params = torch.load(pred_matrix_path)
            params = {}
            params['reflectors_dict'] = {
                    'matrix': ant_params[0].detach().cpu(),
                    'reflector_matrix': ant_params[1].detach().cpu(),
                    'threshold': 0.5,
                    'class_label': -1
                }
            return {'ant_parameters': params['reflectors_dict']}
        # read antenna paramters:
        ant_parameters_path = raw_path + '/ant_parameters.pickle'
        with open(ant_parameters_path, 'rb') as file:
            ant_parameters = pickle.load(file)
        for key, value in ant_parameters.items():
            if isinstance(value, torch.Tensor):
                ant_parameters[key] = value.detach()
        # read envierment paramters:
        env_parameters_path = raw_path + '/model_parameters.pickle'
        with open(env_parameters_path, 'rb') as file:
            env_parameters = pickle.load(file)

        return {'ant_parameters': ant_parameters, 'env_parameters': env_parameters}
    
    def _resolve_freq_index(self, raw_idx):
        """Choose which frequency index this __getitem__ call uses.
    
        Modes (set self.config['freq_mode']):
        'fixed'   : use self.config['idx_freq'] (original behaviour).
        'train'   : randomly sample from self.config['train_freq_idxs'] every call
                    (multi-frequency training; held-out freq never selected).
        'eval'    : use a single pinned index self.config['eval_freq_idx']
                    (use this for val and for the zero-shot 5.2 GHz test set).
        'all'     : deterministic round-robin over train_freq_idxs keyed by
                    raw_idx, so coverage is balanced and reproducible.
        """
        mode = self.config.get('freq_mode', 'train')
    
        if mode == 'fixed':
            return self.config.get('idx_freq')
    
        if mode == 'eval':
            return int(self.config['eval_freq_idx'])
    
        train_idxs = self.config['train_freq_idxs']  # e.g. [0,1,3,4] (excludes 5.2)
    
        if mode == 'train':
            # fresh random frequency per access -> each geometry seen at many freqs
            freq_hz = random.choice(train_idxs)
            return freq_hz
    
        if mode == 'all':
            # deterministic, balanced: key off the integer raw_idx
            try:
                key = int(raw_idx)
            except (TypeError, ValueError):
                key = hash(str(raw_idx))
            return train_idxs[key % len(train_idxs)]
    
        raise ValueError(f"unknown freq_mode {mode!r}")

    
    def get(self, idx):
        processed_file_path = self.processed_file_paths[idx]
        raw_idx = str(processed_file_path).split('/')[-1].split('.')[0].split('_')[-1]
        example = torch.load(processed_file_path, weights_only=False)
        graph_full_data = example['graph']
        surface_currents = [] 
        farfeilds = example['farfeilds']
        s11 = example['s11']
        example_paramters = example['example_paramters']
        if self.config.get('multipule_freqs', True):
            idx_frequency = self._resolve_freq_index(raw_idx)
        else:
            idx_frequency = self.config.get('idx_freq')

        # augmentations:
        if self.config.get('augment_rotation', False) and self.config.get('is_training', False):
            graph_full_data, farfeilds = self.rotate_transform(graph_full_data, farfeilds)
        if self.config.get('augment_shift', False) and self.config.get('is_training', False):
            graph_full_data = self.shift_transform(graph_full_data)
        if self.config.get('augment_compress_towards_threshold', False) and self.config.get('is_training', False):
            # Only compress in 30% of calls; adjust the probability as needed.
            if random.random() < 0.40:
                example_paramters['ant_parameters']['matrix'] = compress_towards_threshold(example_paramters['ant_parameters']['matrix'], threshold=0.5, strength=0.1, jitter=0.0)
        
        # get paramters for the antenna and envierment:
        ant_parameters = []
        if isinstance(example_paramters['ant_parameters'], torch.Tensor):
            ant_parameters.append(example_paramters['ant_parameters'])
            ant_parameters.append(torch.tensor(self.config.get('threshold')))
        if isinstance(example_paramters['ant_parameters'], list):
            ant_parameters.append(torch.tensor(example_paramters['ant_parameters'][0]))
            # get the threshold
            ant_parameters.append(torch.tensor(example_paramters['ant_parameters'][1]))
        else:
            ant_parameters.append(example_paramters['ant_parameters']['matrix'])
            ant_parameters.append(example_paramters['ant_parameters']['threshold'])
            if 'phis' in example_paramters['ant_parameters'] and 'thetas' in example_paramters['ant_parameters']:
                example_paramters['reflectors_params'] = {
                    'thetas': example_paramters['ant_parameters']['thetas'],  
                    'phis': example_paramters['ant_parameters']['phis']}  

                if example_paramters['ant_parameters']['thetas'] is  None and example_paramters['ant_parameters']['phis'] is None:
                    example_paramters['reflectors_params'] = {
                                    'thetas': -1,  
                                    'phis': -1}  
        # class labels are only present for the FMNIST reflector dataset; default to -1
        # otherwise so every sample has the same dict keys (required for batch collation)
        example_paramters['class_label'] = example['example_paramters']['ant_parameters'].get('class_label', -1)
        if example_paramters['class_label'] is None:
            example_paramters['class_label'] = -1
        if 'class_label' in example['example_paramters']['ant_parameters'] and example['example_paramters']['ant_parameters']['class_label'] is not None:
            # no paramtere so set to -1
            example_paramters['reflectors_params'] = {
                'thetas': -1,  
                'phis': -1} 
            example_paramters['env_parameters'] = {'h': torch.tensor(-1.0), 'patch_x': torch.tensor(-1.0), 'patch_y': torch.tensor(-1.0),
                                                    'ground_x': torch.tensor(-1.0), 'ground_y': torch.tensor(-1.0), 'box_size': torch.tensor(-1.0), 'radius': torch.tensor(-1.0)}
            if self.config.get('augment_matrix', False) and self.config.get('is_training', False):
                example_paramters['ant_parameters']['matrix'] = augment_input_matrix(example_paramters['ant_parameters']['matrix'])
            if self.config.get('smooth_binarize_matrix', False):
                example_paramters['ant_parameters']['matrix'] = smooth_binarize(example_paramters['ant_parameters']['matrix'], 250, 300)
                # from matplotlib import pyplot as plt
                # plt.imshow(example_paramters['ant_parameters']['matrix'])
                # plt.show()


        if 'reflector_matrix' in example_paramters['ant_parameters'].keys():
            example_paramters['reflector_matrix'] = example_paramters['ant_parameters']['reflector_matrix']
        else:
            # Provide a fallback/dummy tensor depending on expected shape
            example_paramters['reflector_matrix'] = torch.zeros((16, 16)) # Change shape to whatever your model expects
        example_paramters['ant_parameters'] = ant_parameters
        
        # uncomment to only use the frequencies in the config:
        # farfeilds = farfeilds[self.frequencies_list_idxs]
        # radiation_image_shape = self.config.get('radiation_image_shape', None)
        # if radiation_image_shape is not None:
        #     farfeilds = torch.nn.functional.interpolate(farfeilds.unsqueeze(0), size=radiation_image_shape[:2], mode='bilinear', align_corners=False).squeeze(0)
        # prepare_graph is pure given (graph, config); when connect_components is set
        # its output (bridged edges, 5-dim edge_attr, bridged-graph LapPE) is cached to
        # disk per example -- see utilities/prepared_graph_cache.py for the cache-key/
        # invalidation scheme and the augmentation-guard rationale. This call happens
        # after the rotation/shift augmentation above, so the guard disables caching
        # outright whenever those flags are configured on.
        graph_full_data, graph_dict = prepare_graph_cached(
            graph_full_data, self.config,
            dataset_root=self.root, raw_idx=raw_idx, source_path=str(processed_file_path),
        )

        graph_full_data.x = torch.cat([graph_full_data.pos, graph_full_data.node_normals, graph_full_data.node_type], dim=-1)
        if self.config.get('hetero_graph', False):
            # Heterogeneous-graph experiments are not part of the paper and the
            # hetero_graph module is not included in this release.
            raise NotImplementedError(
                "config['hetero_graph'] is not supported in the public release")
        
        if 'surface_currents' in example and 'pos_surface_current' in example:
            surface_currents = {}
            # surface_currents['surface_currents'] = example['surface_currents']
            # surface_currents['pos_surface_current'] = example['pos_surface_current']
            sc = example['surface_currents'][idx_frequency]
            pos_sc = example['pos_surface_current'][idx_frequency]
            if self.config.get('shuffle_surface_currents', False):
                seed = hash(str(processed_file_path)) % (2 ** 32)
                perm = torch.randperm(sc.shape[0], generator=torch.Generator().manual_seed(seed))
                sc = sc[perm]
                # pos_sc = pos_sc[perm]
            graph_full_data.surface_currents = sc
            graph_full_data.pos_surface_current = pos_sc
        graph_full_data.s11 = get_s11_single_freq(s11, self.config['frequencies'][idx_frequency])
        graph_full_data.idx = idx
        graph_full_data.example_idx = raw_idx
        graph_full_data.example_path = self.processed_file_paths[idx]
        
        freq_hz = float(self.config['frequencies_in_scale'][idx_frequency])
        graph_full_data.freq_hz = torch.tensor([freq_hz], dtype=torch.float32)
        farfeilds = farfeilds[idx_frequency].unsqueeze(0)

        # Uncoment if you need the example parms
        example_paramters = {}
        return graph_full_data,  s11, farfeilds, surface_currents, example_paramters,  raw_idx
