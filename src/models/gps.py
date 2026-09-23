import torch

from torch_geometric.nn import GINEConv, GPSConv, global_add_pool
from torch_geometric.nn.attention import PerformerAttention
from typing import Any, Dict, Optional


class GPS(torch.nn.Module):
    def __init__(self,config: Dict[str, Any], channels: int, pe_dim: int, num_layers: int,
                 attn_type: str, attn_kwargs: Dict[str, Any],in_channels: 16, heads=4, out_channels=None):
        super().__init__()
        self.config = config


        if self.config.get('pointcloud_model'):
            self.pe_lin = torch.nn.Linear(24, pe_dim)
            self.pe_norm = torch.nn.BatchNorm1d(24)
            in_channels = 6
        else:
            self.pe_lin = torch.nn.Linear(10, pe_dim)
            self.pe_norm = torch.nn.BatchNorm1d(10)
        self.edge_emb = torch.nn.Linear(config.get('edge_feature_size', 4), channels) # Embedding(4, channels)
        self.node_emb = torch.nn.Linear(in_channels, channels - pe_dim) #Embedding(28, channels - pe_dim)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = torch.nn.Sequential(
                torch.nn.Linear(channels, channels),
                torch.nn.ReLU(),
                torch.nn.Linear(channels, channels),
            )
            conv = GPSConv(channels, GINEConv(mlp), heads=4,
                           attn_type=attn_type, attn_kwargs=attn_kwargs)
            self.convs.append(conv)

        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=1000 if attn_type == 'performer' else None)

        # Assuming self.dropout is defined elsewhere or meant to be passed in, passing 0.0 as default if missing
        dropout_p = getattr(self, 'dropout', 0.0)

        self.ff_decoder = torch.nn.Sequential(
            torch.nn.Linear(channels, channels * 2),
            torch.nn.LayerNorm(channels * 2),
            torch.nn.GELU(),
            torch.nn.Dropout(p=dropout_p),
            torch.nn.Linear(channels * 2, channels * 2),
            torch.nn.LayerNorm(channels * 2),
            torch.nn.GELU(),
            torch.nn.Dropout(p=dropout_p),
            torch.nn.Linear(channels * 2, out_channels)
        )
        if self.config.get('predict_surface_current', False):
            current_dim = 6 # For example, 3 for real and 3 for imaginary parts of the surface current vector
            self.current_decoder = torch.nn.Sequential(
                torch.nn.Linear(channels, channels * 2),
                torch.nn.LayerNorm(channels * 2),
                torch.nn.GELU(),
                torch.nn.Dropout(p=dropout_p),
                torch.nn.Linear(channels * 2, channels),
                torch.nn.LayerNorm(channels),
                torch.nn.GELU(),
                torch.nn.Linear(channels, current_dim) # e.g., 2 or 3 for complex vector components
                )
        if self.config.get('position_auxiliary_task', False):
            self.position_decoder = torch.nn.Sequential(
                torch.nn.Linear(channels, channels * 2),
                torch.nn.LayerNorm(channels * 2),
                torch.nn.GELU(),
                torch.nn.Dropout(p=dropout_p),
                torch.nn.Linear(channels * 2, channels),
                torch.nn.LayerNorm(channels),
                torch.nn.GELU(),
                torch.nn.Linear(channels, 3) # Predicting 3D coordinates as an auxiliary task
            )


    def forward(self, data, radiation_image_shape=(34, 34), is_training=False):
        x, pe, edge_index, edge_attr, batch = data.x, data.pe, data.edge_index, data.edge_attr, data.batch
        x_pe = self.pe_norm(pe)
        x = torch.cat((self.node_emb(x), self.pe_lin(x_pe)), 1)
        edge_attr = self.edge_emb(edge_attr)

        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)
        x_embeding = x
        x = global_add_pool(x_embeding, batch)
        out_ff = self.ff_decoder(x)
        out_ff = out_ff.reshape(-1, radiation_image_shape[0], radiation_image_shape[1]).unsqueeze(-1)
        if self.config.get('predict_surface_current', False):
            out_currents = self.current_decoder(x_embeding)
            output = {
            'radiation_image': out_ff,
            'surface_current': out_currents
            }
            return output
        if self.config.get('position_auxiliary_task', False):
            out_positions = self.position_decoder(x_embeding)
            output = {
                'radiation_image': out_ff,
                'positions': out_positions
            }
            return output
        return out_ff


class RedrawProjection:
    def __init__(self, model: torch.nn.Module,
                 redraw_interval: Optional[int] = None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            fast_attentions = [
                module for module in self.model.modules()
                if isinstance(module, PerformerAttention)
            ]
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1
