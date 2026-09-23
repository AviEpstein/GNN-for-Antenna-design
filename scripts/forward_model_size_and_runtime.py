

"""
calclating this: | Model | Params | GFLOPs  | Inference |
|---|---|---|---|---|
| GPS+PAIS | X.XM | X.X | X ms/sample |


*Avg. over test meshes, batch 16, one RTX 4090; PAIS surface-current head runs at
inference too, since the direction decoder is conditioned on its output
(use_current_in_direction_decoder=True) -- its cost is included above.*
"""



from configs.parser import ArgParser
from src.dataset.datasets import dataset_random
from src.models.gpsdcc import GPSDCC
import os
import torch_geometric.transforms as T
import torch

from torch_geometric.loader import DataLoader
from src.dataset.stats import calculate_farfeild_statistics
import pandas as pd

import numpy as np, random
import time

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
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'configs', 'inverse', 'forward_model_benchmark.yaml')
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


    # calculate and save dataset statistics (far-field and graph stats) for normalization and analysis
    if config.get('data_root') is None:
        raise ValueError("config['data_root'] must be set in the yaml config file")
    ff_stats_csv = os.path.join(config.get('data_root'), 'dataset_ff_stats.csv')
    graph_stats_csv = os.path.join(config.get('data_root'), 'dataset_graph_stats.csv')
    if os.path.exists(ff_stats_csv) or os.path.exists(graph_stats_csv):
        loaded_ff_stats_df = pd.read_csv(ff_stats_csv)
        ff_stats = loaded_ff_stats_df.iloc[0].to_dict()
        graph_stats = None
    else:
        ff_stats = calculate_farfeild_statistics(train_loader)
        df_ff = pd.DataFrame([ff_stats])
        df_ff.to_csv(ff_stats_csv, index=False)
        graph_stats = None


    # the GPS+PAIS (surrogate) checkpoint benchmarked in the paper table
    if config.get('surrogate_checkpoint') is None:
        raise ValueError("config['surrogate_checkpoint'] must point to the trained GPS+PAIS forward-model .pt")
    config['load_trained_model'] = True
    model = GPSDCC(config=config, in_channels=16, channels=128, heads=4, pe_dim=10, num_layers=10, attn_type='multihead', attn_kwargs = {'dropout': 0.0}, out_channels=config.get('radiation_image_shape')[0]*config.get('radiation_image_shape')[1]) #attn_type='performer', attn_kwargs={'num_random_features': 64})


    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in model: {total_params}")
    print('Model Name: ', model.__class__.__name__)

    if config.get('load_trained_model'):
        print('loading trained model from: ', config.get('surrogate_checkpoint'))
        # strict=False: this checkpoint predates the s11/physics heads, so those
        # stay randomly initialized; irrelevant here since we only measure
        # params/FLOPs/latency, not prediction correctness.
        model.load_state_dict(torch.load(config.get('surrogate_checkpoint')), strict=False)
        run_id = config.get('wandb_run_id')

    ###########################################################
    # 2) size / FLOPs / latency benchmark
    model.to(device)
    model.eval()

    # PAIS (surface-current) predictions feed the direction decoder's
    # attention keys/values here (use_current_in_direction_decoder=True), so
    # they run at inference and are included in the numbers below. The
    # physics radiation head (use_physics_loss) is a separate training-only
    # regularizer never consulted in test_network/validate, so it's switched
    # off. s11_pred_head predates this checkpoint (untrained, and its input
    # dim doesn't match the current+embedding concat it's fed -- a pre-
    # existing bug in GPSDCC.forward, unrelated to this benchmark), so it's
    # also switched off. The reported param count still reflects the full
    # checkpoint including these unused heads.
    bench_config = dict(config)
    bench_config['use_physics_loss'] = False
    bench_config['pred_s11_single_freq'] = False
    model.config = bench_config

    radiation_image_shape = config.get('radiation_image_shape')
    batch_size = 16
    bench_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                               follow_batch=['pos_surface_current'])

    def run_model(batch):
        return model(batch, radiation_image_shape=radiation_image_shape, is_training=False)

    # PixelDataset.get() returns (graph, s11, farfields, surface_currents,
    # example_params, raw_idx); only the graph (index 0) feeds the model.
    sample_batch = next(iter(bench_loader))[0].to(device)

    # ---- GFLOPs per sample, via torch.profiler operator-level FLOP counts ----
    with torch.no_grad():
        for _ in range(3):
            run_model(sample_batch)
        with torch.profiler.profile(with_flops=True) as prof:
            run_model(sample_batch)
    total_flops = sum(evt.flops for evt in prof.key_averages() if evt.flops is not None)
    gflops_per_sample = total_flops / sample_batch.num_graphs / 1e9

    # ---- Inference latency (ms/sample), averaged over the test set ----
    n_warmup = 5
    it = iter(bench_loader)
    with torch.no_grad():
        for _ in range(n_warmup):
            try:
                b = next(it)
            except StopIteration:
                it = iter(bench_loader)
                b = next(it)
            run_model(b[0].to(device))
        if device.type == 'cuda':
            torch.cuda.synchronize()

        total_time_s = 0.0
        total_samples = 0
        for b in bench_loader:
            b = b[0].to(device)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            run_model(b)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            total_time_s += time.perf_counter() - start
            total_samples += b.num_graphs

    ms_per_sample = (total_time_s / total_samples) * 1000

    print(f"| Model | Params | GFLOPs | Inference |")
    print(f"|---|---|---|---|")
    print(f"| GPS+PAIS | {total_params / 1e6:.1f}M | {gflops_per_sample:.1f} | {ms_per_sample:.1f} ms/sample |")

    model.config = config
