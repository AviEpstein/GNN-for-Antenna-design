#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""


import torch
import torch_geometric
from torch_geometric.utils import to_dense_batch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


class DGCNN(nn.Module):
    def __init__(self, config, output_channels=40):
        super(DGCNN, self).__init__()
        self.config = config
        self.k = config.get('k')

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(config.get('emb_dims'))
        # this is changed from the original implimintaion from 6 to 34:
        self.conv1 = nn.Sequential(nn.Conv2d(34, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, config.get('emb_dims'), kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(config.get('emb_dims')*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=config.get('dropout'))
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=config.get('dropout'))
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):

        batch_size = x.size(0)

        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x


class DGCNNRapper(nn.Module):
    def __init__(self, config, output_channels=40):
        super(DGCNNRapper, self).__init__()
        self.config = config
        self.setup_config()
        self.dgcnn = DGCNN(self.config, output_channels).to(torch.device('cuda'))
        # Legacy sub-model init, gated on its own key: 'load_trained_model' is the
        # whole-model eval flag consumed in scripts/train_forward.py.
        if self.config.get('trained_model_path_for_DGCNN'):
            print('loading trained model')
            self.dgcnn.load_state_dict(torch.load(self.config.get('trained_model_path_for_DGCNN')))
            print('model loaded')
        if self.config.get('freeze_trained_model', False):
            for param in self.dgcnn.parameters():
                param.requires_grad = False
        if self.config.get('adapt_input_size', False):
            self.adapt_input_linear = nn.Linear(17, 3).to(torch.device('cuda'))

        if self.config.get('adapt_output_size', False):
            self.adapt_output_linear = nn.Linear(output_channels, self.config.get('radiation_image_shape')[0]*self.config.get('radiation_image_shape')[1]).to(torch.device('cuda'))

        if self.config.get('predict_surface_current', False):
            hidden_channels = 128
            out_channels = 1024
            current_dim = 6
            self.dropout = self.config.get('dropout', 0.2)

            self.current_decoder = nn.Sequential(
            nn.Linear(512, hidden_channels * 2),
            nn.LayerNorm(hidden_channels * 2),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, current_dim) # e.g., 2 or 3 for complex vector components
            )
            self.traget_pos_encoder = nn.Sequential(
                    nn.Linear(3, 512), # Assuming 3D positions for surface current points
                    nn.LayerNorm(512),
                    nn.GELU()
                )
    def forward(self, data, radiation_image_shape=None, is_training=True):

        pos, mask = to_dense_batch(data.pos, data.batch)
        node_normals, _ = to_dense_batch(data.node_normals, data.batch)
        node_type, _ = to_dense_batch(data.node_type, data.batch)
        node_probs, _ = to_dense_batch(data.node_probs, data.batch)
        x = torch.cat([pos, node_normals, node_type, node_probs],dim = 2)


        if self.config.get('adapt_input_size', False):
            x = self.adapt_input_linear(x)
        elif self.config.get('raw_position_input', False):
            x = pos

        x = x.transpose(2,1)
        encoding = self.dgcnn(x)

        if self.config.get('adapt_output_size', False):
            ff = self.adapt_output_linear(encoding)
        ff = encoding
        ff = ff.view(-1, self.config.get('radiation_image_shape')[0], self.config.get('radiation_image_shape')[1])


        if self.config.get('predict_surface_current', False):
            # Interpolate features for the current batch
            interpolated_features = knn_interpolate(
                    x=encoding,
                    pos_x=pos,   # Original coordinates of the mesh nodes
                    pos_y=data.pos_surface_current,  # Target coordinates where you want to predict currents
                    k=3,                # Number of nearest neighbors to use for interpolation
                    batch_x=data.batch,       # No batching for interpolation
                    batch_y=data.pos_surface_current_batch        # No batching for interpolation
                    )
            pe_encoded = self.traget_pos_encoder(data.pos_surface_current)
            interpolated_features = interpolated_features + pe_encoded # Add positional encoding to the interpolated features

            out_currents = self.current_decoder(interpolated_features)
            output = {
                'radiation_image': ff.unsqueeze(-1),
                'surface_current': out_currents
                }
            return output
        return ff.unsqueeze(-1)

    def setup_config(self):
        self.config['k'] = 20
        self.config['emb_dims'] = 1024
        self.config['dropout'] = 0.5


from torch_geometric.nn import EdgeCNN, knn_interpolate

class EdgeCNNClassic(torch.nn.Module):
    def __init__(self, config, output_channels=40):
        super(EdgeCNNClassic, self).__init__()
        self.config = config
        raw_node_feature_size = config.get('raw_node_feture_size')
        linear_output_size = config.get('radiation_image_shape')[0]*config.get('radiation_image_shape')[1]
        self.edge_cnn = EdgeCNN(in_channels=raw_node_feature_size,
                       hidden_channels=128,
                       num_layers=4,
                       out_channels=1024)
        self.linear_out = nn.Linear(1024, linear_output_size)
        self.dropout = config.get('dropout', 0.2)


    def forward(self, data, radiation_image_shape=None, is_training=True):
        pos, node_normals, node_type, edge_index, node_probs = data.pos, data.node_normals, data.node_type, data.edge_index, data.node_probs
        x = torch.cat([pos, node_normals, node_type, node_probs],dim = 1)

        encoded_graph = self.edge_cnn(x, edge_index=edge_index, batch=data.batch)
        global_feature = torch_geometric.nn.global_mean_pool(encoded_graph, data.batch)
        output = self.linear_out(global_feature)
        ff = output.reshape(-1, self.config.get('radiation_image_shape')[0], self.config.get('radiation_image_shape')[1]).unsqueeze(-1)
        return ff


class EdgeCNNClassicWSC(torch.nn.Module):
    def __init__(self, config, output_channels=40):
        super(EdgeCNNClassicWSC, self).__init__()
        self.config = config
        raw_node_feature_size = config.get('raw_node_feture_size')
        hidden_channels = 128
        out_channels = 1024
        current_dim = 6
        linear_output_size = config.get('radiation_image_shape')[0]*config.get('radiation_image_shape')[1]
        self.dropout = config.get('dropout', 0.0)
        self.edge_cnn = EdgeCNN(in_channels=raw_node_feature_size,
                       hidden_channels=128,
                       num_layers=4,
                       out_channels=out_channels)

        self.linear_out = nn.Linear(out_channels, linear_output_size)
        self.current_decoder = nn.Sequential(
            nn.Linear(out_channels, hidden_channels * 2),
            nn.LayerNorm(hidden_channels * 2),
            nn.GELU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, current_dim) # e.g., 2 or 3 for complex vector components
            )
        self.traget_pos_encoder = nn.Sequential(
                nn.Linear(3, 512), # Assuming 3D positions for surface current points
                nn.LayerNorm(512),
                nn.GELU()
            )

    def forward(self, data, radiation_image_shape=None, is_training=True):
        pos, node_normals, node_type, edge_index, node_probs = data.pos, data.node_normals, data.node_type, data.edge_index, data.node_probs
        x = torch.cat([pos, node_normals, node_type, node_probs],dim = 1)

        encoded_graph = self.edge_cnn(x, edge_index=edge_index, batch=data.batch)

        global_feature = torch_geometric.nn.global_mean_pool(encoded_graph, data.batch)
        output = self.linear_out(global_feature)
        ff = output.reshape(-1, self.config.get('radiation_image_shape')[0], self.config.get('radiation_image_shape')[1]).unsqueeze(-1)

        out_currents = self.current_decoder(encoded_graph)
        output = {
            'radiation_image': ff,
            'surface_current': out_currents
            }
        return output
