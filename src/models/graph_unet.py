from torch_geometric.nn.models import GraphUNet
import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_mean_pool

class GraphUNetWrapper(nn.Module):
    def __init__(self, config, output_channels=40):
        super(GraphUNetWrapper, self).__init__()
        self.config = config
        self.setup_config()
        hidden_channels=512
        # depth = 4 or 3
        self.graph_unet = GraphUNet(in_channels=17, hidden_channels=hidden_channels, depth=4 , out_channels=hidden_channels).to(torch.device('cuda'))
        # Legacy sub-model init, gated on its own key: 'load_trained_model' is the
        # whole-model eval flag consumed in scripts/train_forward.py.
        if self.config.get('trained_model_path_for_GraphUNet'):
            print('loading trained model')
            self.graph_unet.load_state_dict(torch.load(self.config.get('trained_model_path_for_GraphUNet')))
            print('model loaded')
        if self.config.get('freeze_trained_model', False):
            for param in self.graph_unet.parameters():
                param.requires_grad = False
        if self.config.get('adapt_input_size', False):
            self.adapt_input_linear = nn.Linear(17, 3).to(torch.device('cuda'))

        if self.config.get('adapt_output_size', False):
            self.adapt_output_linear = nn.Linear(output_channels, self.config.get('radiation_image_shape')[0]*self.config.get('radiation_image_shape')[1]).to(torch.device('cuda'))

        dropout_p = self.config.get('dropout', 0.0)
        self.ff_decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels * 2),
            torch.nn.LayerNorm(hidden_channels * 2),
            torch.nn.GELU(),
            torch.nn.Dropout(p=dropout_p),
            torch.nn.Linear(hidden_channels * 2, hidden_channels * 2),
            torch.nn.LayerNorm(hidden_channels * 2),
            torch.nn.GELU(),
            torch.nn.Dropout(p=dropout_p),
            torch.nn.Linear(hidden_channels * 2, self.config.get('radiation_image_shape')[0]*self.config.get('radiation_image_shape')[1])
        )

        if self.config.get('predict_surface_current', False):
            current_dim = 6
            self.current_decoder = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels * 2),
                torch.nn.LayerNorm(hidden_channels * 2),
                torch.nn.GELU(),
                torch.nn.Dropout(p=dropout_p),
                torch.nn.Linear(hidden_channels * 2, hidden_channels),
                torch.nn.LayerNorm(hidden_channels),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_channels, current_dim) # e.g., 2 or 3 for complex vector components
                )

    def forward(self, data, radiation_image_shape=None, is_training=True):
        pos, mask = to_dense_batch(data.pos, data.batch)
        node_normals, _ = to_dense_batch(data.node_normals, data.batch)
        node_type, _ = to_dense_batch(data.node_type, data.batch)
        node_probs, _ = to_dense_batch(data.node_probs, data.batch)
        x = torch.cat([pos, node_normals, node_type, node_probs], dim=2)  # [batch, max_nodes, features]

        batch_size, max_nodes, feat_dim = x.shape
        x = x.reshape(-1, feat_dim)  # [batch * max_nodes, features]
        mask = mask.reshape(-1)      # [batch * max_nodes]

        # Only keep real nodes
        x = x[mask]
        edge_index = data.edge_index
        if edge_index.device != x.device:
            edge_index = edge_index.to(x.device)

        x = self.graph_unet(x, edge_index, batch=data.batch)
        global_mean = global_mean_pool(x, batch=data.batch)
        ff = self.ff_decoder(global_mean)
        ff = ff.view(-1, self.config.get('radiation_image_shape')[0], self.config.get('radiation_image_shape')[1]).unsqueeze(-1)

        if self.config.get('predict_surface_current', False):
            surface_current = self.current_decoder(x)
            output = {
                'radiation_image': ff,
                'surface_current': surface_current
                }
            return output
        if torch.isnan(x).any():
            raise ValueError("The output tensor x contains NaN values.")

        return ff

    def setup_config(self):
        self.config['k'] = 20
        self.config['emb_dims'] = 1024
        self.config['dropout'] = 0.5
