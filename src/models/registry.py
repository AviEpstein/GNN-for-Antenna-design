"""Config-driven model construction.

Replaces the comment-block model selection of the original research scripts:
every constructor call below is verbatim from the scripts that produced the
paper's tables (training_scripts/train_forward_ff_pixel_model_GNN.py and
train_forward_ff_pixel_model_baseline.py in the research repo) — only the
selection mechanism changed from "uncomment a line" to config['model_type'].

Graph models (trained with scripts/train_forward.py):
    GPS | GPSDCC | GCN | GAT | GRAPH_UNET | DGCNN | MGN
    PAIS (the surface-current auxiliary head) is toggled with
    config['predict_surface_current']; for DGCNN it selects the
    EdgeCNNClassicWSC variant.

Grid baselines (trained with scripts/train_baseline.py):
    MLP | UNET | RESNET50
"""


def build_graph_model(config, device):
    model_type = config.get('model_type')
    out_channels = (config.get('radiation_image_shape')[0]
                    * config.get('radiation_image_shape')[1])
    gps_kwargs = dict(config=config, in_channels=16, channels=128, heads=4, pe_dim=10,
                      num_layers=10, attn_type='multihead', attn_kwargs={'dropout': 0.0},
                      out_channels=out_channels)

    if model_type == 'GPS':
        from src.models.gps import GPS
        return GPS(**gps_kwargs)
    if model_type == 'GPSDCC':
        from src.models.gpsdcc import GPSDCC
        return GPSDCC(**gps_kwargs)
    if model_type == 'GCN':
        from src.models.gcn import GCNWrapper
        return GCNWrapper(config, output_channels=out_channels)
    if model_type == 'GAT':
        from src.models.gat import GATWrapper
        return GATWrapper(config, output_channels=out_channels)
    if model_type == 'GRAPH_UNET':
        from src.models.graph_unet import GraphUNetWrapper
        return GraphUNetWrapper(config, output_channels=out_channels)
    if model_type == 'DGCNN':
        from src.models.dgcnn import EdgeCNNClassic, EdgeCNNClassicWSC
        cls = EdgeCNNClassicWSC if config.get('predict_surface_current') else EdgeCNNClassic
        return cls(config=config, output_channels=out_channels)
    if model_type == 'MGN':
        from src.models.mesh_graph_nets import MeshGraphNetWrapper
        return MeshGraphNetWrapper(config, device)
    raise ValueError(f'unknown graph model_type: {model_type}')


def build_baseline_model(config):
    model_type = config.get('model_type')
    if model_type == 'MLP':
        from src.models.pixel_baseline_models import MLPBaseline
        return MLPBaseline(in_features=16 * 16 * 2)
    if model_type == 'UNET':
        from src.models.unet_baseline import UNetBaselineBasicModel
        return UNetBaselineBasicModel(num_hiddens=128, in_channels=2, out_channels=1)
    if model_type == 'RESNET50':
        from src.models.pixel_baseline_models import ResNet50
        return ResNet50(in_channels=2, out_channels=1)
    raise ValueError(f'unknown baseline model_type: {model_type}')
