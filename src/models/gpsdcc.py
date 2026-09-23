import math
from typing import Any, Dict

import torch
import torch.nn as nn
from torch_geometric.nn import GPSConv, GINEConv, global_add_pool
from torch_geometric.utils import to_dense_batch

from src.physics.ff_from_surface_currents import compute_farfield_from_currents_for_batch
from src.geometry.mesh_functions import get_nodes_of_type, model_stl_types


class GPSDCC(torch.nn.Module):
    def __init__(self, config: Dict[str, Any], channels: int, pe_dim: int, num_layers: int,
                 attn_type: str, attn_kwargs: Dict[str, Any], in_channels: int = 16,
                 heads=4, out_channels=None):
        super().__init__()
        self.config = config
        self.channels = channels
        self.heads = heads

        self.use_direction_decoder = self.config.get("direction_conditioned_decoder", True)
        self.use_current_in_direction_decoder = self.config.get(
            "use_current_in_direction_decoder", False
        )

        if self.config.get('pointcloud_model'):
            self.pe_lin = torch.nn.Linear(24, pe_dim)
            self.pe_norm = torch.nn.BatchNorm1d(24)
            in_channels = 6
        else:
            self.pe_lin = torch.nn.Linear(10, pe_dim)
            self.pe_norm = torch.nn.BatchNorm1d(10)

        self.edge_emb = torch.nn.Linear(4, channels)
        self.node_emb = torch.nn.Linear(in_channels, channels - pe_dim)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = torch.nn.Sequential(
                torch.nn.Linear(channels, channels),
                torch.nn.ReLU(),
                torch.nn.Linear(channels, channels),
            )
            conv = GPSConv(
                channels,
                GINEConv(mlp),
                heads=heads,
                attn_type=attn_type,
                attn_kwargs=attn_kwargs
            )
            self.convs.append(conv)

        self.redraw_projection = RedrawProjection(
            self.convs,
            redraw_interval=1000 if attn_type == 'performer' else None
        )

        # FIX: pull dropout from config instead of getattr(self, 'dropout', 0.0),
        # which was always 0.0 because self.dropout is never set.
        dropout_p = float(self.config.get('dropout', 0.0))

        # Old global decoder, kept as fallback.
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

        # PAIS surface-current head.
        if self.config.get('predict_surface_current', False):
            current_dim = 6
            self.current_decoder = torch.nn.Sequential(
                torch.nn.Linear(channels, channels * 2),
                torch.nn.LayerNorm(channels * 2),
                torch.nn.GELU(),
                torch.nn.Dropout(p=dropout_p),
                torch.nn.Linear(channels * 2, channels),
                torch.nn.LayerNorm(channels),
                torch.nn.GELU(),
                torch.nn.Linear(channels, current_dim)
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
                torch.nn.Linear(channels, 3)
            )
        if self.config.get('use_physics_loss', False):
            self.radiation_layer = RadiationLayerWithResidual(
                channels,
                grid_shape=self.config.get('radiation_image_shape')[:2],
                residual_scale=self.config.get('residual_scale', 0.1)
            )

        # ------------------------------------------------------------------
        # Direction-conditioned far-field decoder
        # ------------------------------------------------------------------
        # FIX: direction features are now 6-dim and free of the duplicate
        # cos(theta) and the discontinuous phi/pi:
        #   [kx, ky, kz, sin(theta), cos(theta), sin(phi), cos(phi)]
        # cos(theta) appears once (it equals kz, but kept explicit as the
        # elevation feature is harmless); phi is encoded periodically so the
        # +-pi seam carries no discontinuity.
        dir_in_dim = 6

        self.direction_mlp = torch.nn.Sequential(
            torch.nn.Linear(dir_in_dim, channels),
            torch.nn.LayerNorm(channels),
            torch.nn.GELU(),
            torch.nn.Linear(channels, channels),
            torch.nn.LayerNorm(channels),
            torch.nn.GELU(),
        )

        # FIX: currents now feed BOTH the key and value paths so attention
        # *scores* (not just the aggregated values) can depend on the current
        # at each node. Two separate projections from the (embedding [+ current])
        # space down to `channels`.
        kv_in_dim = channels
        if self.use_current_in_direction_decoder:
            kv_in_dim += 6

        self.direction_key_proj = torch.nn.Linear(kv_in_dim, channels)
        self.direction_value_proj = torch.nn.Linear(kv_in_dim, channels)

        self.direction_cross_attn = torch.nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            dropout=dropout_p,
            batch_first=True
        )

        self.direction_ffn = torch.nn.Sequential(
            torch.nn.Linear(channels, channels * 2),
            torch.nn.LayerNorm(channels * 2),
            torch.nn.GELU(),
            torch.nn.Dropout(p=dropout_p),
            torch.nn.Linear(channels * 2, channels),
            torch.nn.LayerNorm(channels),
            torch.nn.GELU(),
        )

        self.direction_out = torch.nn.Sequential(
            torch.nn.Linear(channels, channels),
            torch.nn.GELU(),
            torch.nn.Linear(channels, 1)
        )

        if self.config.get('pred_s11_single_freq', False):
            self.s11_pred_head = torch.nn.Sequential(
                torch.nn.Linear(channels, 2 * channels ),
                torch.nn.LayerNorm(2 * channels ),
                torch.nn.GELU(),
                torch.nn.Dropout(p=dropout_p),
                torch.nn.Linear( 2*channels, channels),
                torch.nn.LayerNorm(channels),
                torch.nn.GELU(),
                torch.nn.Linear(channels, 2)  # real and imaginary parts
            )

    def make_direction_grid(self, radiation_image_shape, device):
        """
        Creates direction queries for a regular theta-phi radiation image.

        Returns:
            dirs: [M, 6], where M = H * W
            features: [kx, ky, kz, sin(theta), cos(theta), sin(phi), cos(phi)]
        """
        H, W = radiation_image_shape[0], radiation_image_shape[1]

        theta = torch.linspace(0.0, math.pi, H, device=device)
        phi = torch.linspace(-math.pi, math.pi, W, device=device)

        theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing="ij")

        sin_t = torch.sin(theta_grid)
        cos_t = torch.cos(theta_grid)
        sin_p = torch.sin(phi_grid)
        cos_p = torch.cos(phi_grid)

        kx = sin_t * cos_p
        ky = sin_t * sin_p
        kz = cos_t

        dirs = torch.stack(
            [
                kx,
                ky,
                kz,
                sin_t,
                sin_p,   # periodic phi encoding, no +-pi discontinuity
                cos_p,
            ],
            dim=-1
        )

        dirs = dirs.reshape(H * W, 6)
        return dirs

    def direction_conditioned_decode(
        self,
        node_embeddings,
        batch,
        radiation_image_shape,
        predicted_currents=None
    ):
        """
        node_embeddings: [total_nodes, channels]
        batch: [total_nodes]
        predicted_currents: optional [total_nodes, 6]

        Returns:
            out_ff: [B, H, W, 1]
        """

        # Convert sparse PyG batch into dense tensor.
        node_dense, node_mask = to_dense_batch(node_embeddings, batch)
        # node_dense: [B, N_max, C]; node_mask: [B, N_max] True for real nodes.

        # FIX: build a single key/value source that includes currents (when
        # enabled), then derive BOTH keys and values from it. Previously keys
        # used raw node_dense while only values saw the currents, so attention
        # scores were blind to the current distribution.
        if self.use_current_in_direction_decoder and predicted_currents is not None:
            current_dense, _ = to_dense_batch(predicted_currents, batch)
            kv_source = torch.cat([node_dense, current_dense], dim=-1)  # [B,N,C+6]
        else:
            kv_source = node_dense                                      # [B,N,C]

        keys = self.direction_key_proj(kv_source)      # [B, N, C]
        values = self.direction_value_proj(kv_source)  # [B, N, C]

        B = node_dense.size(0)
        H, W = radiation_image_shape[0], radiation_image_shape[1]

        dirs = self.make_direction_grid(radiation_image_shape, node_embeddings.device)
        dir_queries = self.direction_mlp(dirs)                 # [M, C]
        dir_queries = dir_queries.unsqueeze(0).repeat(B, 1, 1)  # [B, M, C]

        attn_out, _ = self.direction_cross_attn(
            query=dir_queries,
            key=keys,
            value=values,
            key_padding_mask=~node_mask,
            need_weights=False
        )

        # Residual directional FFN.
        attn_out = attn_out + self.direction_ffn(attn_out)

        out = self.direction_out(attn_out)  # [B, M, 1]
        out = out.reshape(B, H, W, 1)

        return out

    def forward(self, data, radiation_image_shape=(34, 34), is_training=False):
        x, pe, edge_index, edge_attr, batch = (
            data.x,
            data.pe,
            data.edge_index,
            data.edge_attr,
            data.batch
        )

        x_pe = self.pe_norm(pe)
        x = torch.cat((self.node_emb(x), self.pe_lin(x_pe)), dim=1)
        edge_attr = self.edge_emb(edge_attr)

        for conv in self.convs:
            x = conv(x, edge_index, batch, edge_attr=edge_attr)

        x_embedding = x

        out_currents = None
        out_positions = None
        ff_from_physics = None
        ff_physics_residual = None

        if self.config.get('predict_surface_current', False):
            out_currents = self.current_decoder(x_embedding)
            if self.config.get('use_physics_loss', False):
                ff_physics_analytic, ff_physics_residual = self.radiation_layer(
                    data, out_currents, global_add_pool(x_embedding, batch)
                )
                ff_from_physics = 0.0*ff_physics_analytic + ff_physics_residual

        if self.config.get('pred_s11_single_freq', False):
            # we pool only the feed nodes for s11 prediction, as s11 is a feed parameter and should be predicted from the feed node embeddings. This also reduces the risk of overfitting to the training data by relying on non-feed nodes that may have different distributions in the test set.
            node_type = data.node_type
            Feed_type_index = model_stl_types['Antenna_Feed_STEP']
            Feed_type_index = get_nodes_of_type(node_type, Feed_type_index-1)
            feed_node_embeddings = x_embedding[Feed_type_index]  # [num_feed_nodes, channels]
            feed_out_currents = out_currents[Feed_type_index] if out_currents is not None else None
            feed_combined = torch.cat([feed_node_embeddings, feed_out_currents], dim=1) if feed_out_currents is not None else feed_node_embeddings
            batch_feed = batch[Feed_type_index]  # [num_feed_nodes]
            node_embeddings_pool = global_add_pool(feed_combined, batch_feed)
            s11_pred = self.s11_pred_head(node_embeddings_pool)  # [B, 2] real and imaginary parts
            s11_pred = s11_pred.view(-1, 2)  # real and imaginary parts



        if self.config.get('position_auxiliary_task', False):
            out_positions = self.position_decoder(x_embedding)

        if self.use_direction_decoder:
            out_ff = self.direction_conditioned_decode(
                node_embeddings=x_embedding,
                batch=batch,
                radiation_image_shape=radiation_image_shape,
                predicted_currents=out_currents
            )
        else:
            x_graph = global_add_pool(x_embedding, batch)
            out_ff = self.ff_decoder(x_graph)
            out_ff = out_ff.reshape(
                -1,
                radiation_image_shape[0],
                radiation_image_shape[1]
            ).unsqueeze(-1)

        output = {"radiation_image": out_ff}
        if out_currents is not None:
            output["surface_current"] = out_currents
        if out_positions is not None:
            output["positions"] = out_positions
        if ff_from_physics is not None:
            output["radiation_image_physics"] = ff_from_physics
            output["radiation_image_residual"] = ff_physics_residual
        if self.config.get('pred_s11_single_freq', False):
            output["s11_pred"] = s11_pred
        return output


class RedrawProjection:
    def __init__(self, model, redraw_interval=None):
        self.model = model
        self.redraw_interval = redraw_interval
        self.num_last_redraw = 0

    def redraw_projections(self):
        if not self.model.training or self.redraw_interval is None:
            return
        if self.num_last_redraw >= self.redraw_interval:
            from torch_geometric.nn.attention import PerformerAttention
            fast_attentions = [
                m for m in self.model.modules()
                if isinstance(m, PerformerAttention)
            ]
            for fa in fast_attentions:
                fa.redraw_projection_matrix()
            self.num_last_redraw = 0
            return
        self.num_last_redraw += 1



class RadiationLayerWithResidual(nn.Module):
    """Analytic radiation integral + small learned residual.

    The idealized discrete integral misses meshing density effects, the
    ground-plane image, and feed-excitation phase reference. A lightweight
    residual head on the pooled embedding absorbs these without letting the
    network bypass the physics (the analytic term carries the gradient to J).
    """

    def __init__(self, channels, grid_shape=(34, 34), residual_scale=0.1):
        super().__init__()
        self.grid_shape = grid_shape
        H, W = grid_shape
        self.residual_scale = residual_scale
        self.residual_head = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, H * W),
        )


    def forward(self, graph, output_currents, pooled_embedding):
        F_phys = compute_farfield_from_currents_for_batch(graph, output_currents, image_shape=self.grid_shape, device=output_currents.device)
        H, W = self.grid_shape
        res = self.residual_head(pooled_embedding)          # [B, H*W]
        res = res.reshape(-1, H, W) * self.residual_scale
        return F_phys, res
