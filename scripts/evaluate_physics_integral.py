from configs.parser import ArgParser
from src.dataset.datasets import dataset_random
from src.models.gpsdcc import GPSDCC
import os
import torch_geometric.transforms as T
import torch
from src.physics.ff_from_surface_currents import compute_farfield_from_currents

from torch_geometric.loader import DataLoader

import numpy as np, random

from src.graph.surface_current_functions import knn_interpolate_features

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

# Also seed the DataLoader worker init
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs', 'inverse', 'physics_integral.yaml')
    arg_parser = ArgParser(config_file=config_path)
    config = arg_parser.get_config()

    device_num = config.get('device_num', '0')
    os.environ["CUDA_VISIBLE_DEVICES"] = device_num
    cuda_device = "cuda:" + device_num

    ###########################################################
    # 1) dataset and loaders:
    pe_transform = T.Compose([
        T.ToUndirected(),
        T.AddLaplacianEigenvectorPE(k=10, attr_name='pe')
    ])
    device = torch.device(cuda_device if torch.cuda.is_available() else "cpu")



    # set seed for reproducibility
    torch.manual_seed(0)
    combined_dataset, train_dataset, test_dataset = dataset_random(config, pe_transform, dataset_name='FMNIST_CIFAR')
    seed = 123 #123 #42 # 0, 123
    set_seed(seed)
    seed_worker(seed)
    g = torch.Generator()
    g.manual_seed(seed)


    # create loaders with num_workers and pin_memory for performance
    num_cpus = os.cpu_count() or 4
    num_workers = min(8, max(1, num_cpus // 2))
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                            follow_batch=['pos_surface_current'],
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=True,
                            persistent_workers=True,
                            prefetch_factor=2,
                            drop_last=True,
                            worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                             follow_batch=['pos_surface_current'],
                            )

    model = GPSDCC(config=config, in_channels=16, channels=128, heads=4, pe_dim=10, num_layers=10, attn_type='multihead', attn_kwargs = {'dropout': 0.0}, out_channels=config.get('radiation_image_shape')[0]*config.get('radiation_image_shape')[1]) #attn_type='performer', attn_kwargs={'num_random_features': 64})

    config['gps'] = True
    if config.get('load_trained_model'):
        surrogate_checkpoint = config.get('surrogate_checkpoint')
        if surrogate_checkpoint is None:
            raise ValueError("config['surrogate_checkpoint'] must point to the trained GPS+PAIS forward-model .pt")
        print('loading trained model from: ', surrogate_checkpoint)
        model.load_state_dict(torch.load(surrogate_checkpoint))
        run_id = config.get('wandb_run_id')
    model.to(device)

    mse_loss_fn = torch.nn.MSELoss()
    mse_loss_ff_from_gt_sc_list = []
    mse_loss_ff_from_pred_sc_list = []
    physics_to_residual_ratio_list = []
    residual_norm_greater_then_ff_analytic_norm_count = 0

    for data in test_loader:
        batch_idx = 0
        graph, s11, ff_images, graph_identifier, params_vector, raw_idx = data

        model.eval()
        with torch.no_grad():
            graph = graph.to(device)
            pred = model(graph)
        pred_surface_current = pred['surface_current']



        surface_current_pos = graph.pos_surface_current
        target_surface_current = graph.surface_currents
        graph_batch_mask = graph.batch == batch_idx
        graph_pos = graph.pos[graph_batch_mask]
        pred_surface_current_batch = pred_surface_current[graph_batch_mask]
        pos_surface_current_mask = (graph.pos_surface_current_batch == batch_idx)
        surface_current_pos = surface_current_pos[pos_surface_current_mask]
        target_surface_current = target_surface_current[pos_surface_current_mask]
        interpolated_target_currents = knn_interpolate_features(surface_current_pos, graph_pos, target_surface_current, k=5)

        # from gt sc compute ff
        surface_data = {'pos': graph_pos, 'J': interpolated_target_currents}
        ff_from_gt_sc = compute_farfield_from_currents(surface_data, graph.freq_hz, image_shape=(34, 34), device=interpolated_target_currents.device)
        loss_ff_from_gt_sc = mse_loss_fn(ff_from_gt_sc, ff_images[batch_idx][0].to(device))
        mse_loss_ff_from_gt_sc_list.append(loss_ff_from_gt_sc)

        # from pred sc compute ff
        surface_data_pred = {'pos': graph_pos, 'J': pred_surface_current_batch}
        ff_from_pred_sc = compute_farfield_from_currents(surface_data_pred, graph.freq_hz, image_shape=(34, 34), device=pred_surface_current_batch.device)
        loss_ff_from_pred_sc = mse_loss_fn(ff_from_pred_sc, ff_images[batch_idx][0].to(device))
        mse_loss_ff_from_pred_sc_list.append(loss_ff_from_pred_sc)

        # meassure r=∥Fanalytic​(J^)∥2​ / ∥sMLP(h)∥2​​.
        # ff_from_pred_sc is F_analytic(J_hat) (the standalone radiation integral over the
        # predicted surface currents); radiation_image_residual is the model's learned
        # residual term sMLP(h) from RadiationLayerWithResidual, scaled by residual_scale.
        if 'radiation_image_residual' in pred:
            ff_analytic_norm = ff_from_pred_sc.norm(p=2)
            residual_norm = pred['radiation_image_residual'][batch_idx].norm(p=2)
            r = ff_analytic_norm / residual_norm
            physics_to_residual_ratio_list.append(r)
            if r < 1:
                residual_norm_greater_then_ff_analytic_norm_count += 1



    print(f"Average MSE Loss on Test Set: {torch.mean(torch.tensor(mse_loss_ff_from_gt_sc_list))}")
    print(f"Average MSE Loss on Test Set (from predicted surface currents): {torch.mean(torch.tensor(mse_loss_ff_from_pred_sc_list))}")
    if physics_to_residual_ratio_list:
        print(f"Average r = ||F_analytic(J_hat)||_2 / ||sMLP(h)||_2: {torch.mean(torch.tensor(physics_to_residual_ratio_list))}")
    print(f"Count of cases where ||sMLP(h)||_2 > ||F_analytic(J_hat)||_2: {residual_norm_greater_then_ff_analytic_norm_count} out of {len(test_loader)}")
