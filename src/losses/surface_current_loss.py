import torch
from torch_geometric.utils import subgraph
from src.graph.surface_current_functions import knn_interpolate_features, knn_interpolate_features_per_component
from src.graph.graph_functions import connected_components_pytorch
from src.losses.losses import FarfeildLoss


def filter_surface_current(surface_current, pos_surface_current, threshold):
    norms = torch.norm(surface_current, dim=-1)
    mask = norms > threshold
    return surface_current[mask], pos_surface_current[mask]


def mesh_only_component_labels(pos, edge_index, edge_attr, num_nodes=None):
    """Connected-component labels for a mesh, from mesh-connectivity edges only.

    connect_components (src/graph/GNN_functions.py:prepare_graph) adds a binary
    edge-type column to edge_attr when it bridges disconnected components: mesh edges
    (incl. radius-graph edges) = 0, bridge edges = 1. Those bridge edges exist purely
    for message passing/LapPE -- they say nothing about which raw current samples
    physically belong to which antenna element, so component labels used to decide
    that must come from the edge_type==0 subgraph only. When edge_attr has no
    edge-type column (connect_components off, 4 columns), every edge is a mesh edge.

    Returns (labels: LongTensor[num_nodes], n_components: int).
    """
    if num_nodes is None:
        num_nodes = pos.shape[0]
    if edge_attr is not None and edge_attr.numel() > 0 and edge_attr.shape[-1] >= 5:
        mesh_mask = edge_attr[:, 4] == 0
        mesh_edge_index = edge_index[:, mesh_mask]
    else:
        mesh_edge_index = edge_index
    # connected_components_pytorch does not symmetrize its input itself -- see
    # knn_component_bridge_edges in src/graph/graph_functions.py for why we always
    # pre-symmetrize before calling it.
    sym_edge_index = (torch.cat([mesh_edge_index, mesh_edge_index.flip(0)], dim=1)
                       if mesh_edge_index.numel() else mesh_edge_index)
    n_components, labels = connected_components_pytorch(sym_edge_index, num_nodes)
    return labels, n_components


def nearest_node_component_labels(query_pos, node_pos, node_labels):
    """Assigns each query point (e.g. a raw surface-current sample) the component
    label of its nearest node_pos entry (1-NN by Euclidean distance)."""
    if query_pos.shape[0] == 0:
        return torch.zeros(0, dtype=torch.long, device=query_pos.device)
    nearest = torch.cdist(query_pos, node_pos).argmin(dim=1)
    return node_labels.to(query_pos.device)[nearest]


def interpolate_target_currents_per_component(surface_current_pos, target_surface_current, graph_pos,
                                               edge_index, edge_attr, k=5):
    """per_component_interpolation path: derives mesh-only component labels for
    graph_pos (see mesh_only_component_labels), assigns each raw surface_current_pos
    point to a component by nearest mesh node, then interpolates
    target_surface_current (source) onto graph_pos (target) restricted per-component
    (knn_interpolate_features_per_component falls back to an unrestricted global knn
    only for a component with zero assigned source points).
    """
    mesh_labels, n_components = mesh_only_component_labels(graph_pos, edge_index, edge_attr,
                                                            num_nodes=graph_pos.shape[0])
    raw_point_labels = nearest_node_component_labels(surface_current_pos, graph_pos, mesh_labels)
    return knn_interpolate_features_per_component(
        surface_current_pos, raw_point_labels, graph_pos, mesh_labels, target_surface_current, k=k)


class SurfaceCurrentLossOutputInterpulation(torch.nn.Module):
    def __init__(self, config={}):
        super(SurfaceCurrentLossOutputInterpulation, self).__init__()
        self.mse_loss = torch.nn.MSELoss()
        self.config = config

    def forward(self, graph, pred_surface_current):
        '''
        will find all nodes of type PEC and then interpolate the predicted surface current at those positions,
        and then calculate MSE loss with the target surface current at those positions.
        '''
        loss_avg = 0
        for batch_idx in range(graph.num_graphs):
            surface_current_pos = graph.pos_surface_current
            target_surface_current = graph.surface_currents
            graph_batch_mask = graph.batch == batch_idx
            graph_pos = graph.pos[graph_batch_mask]
            pred_surface_current_batch = pred_surface_current[graph_batch_mask]
            pos_surface_current_mask = (graph.pos_surface_current_batch == batch_idx)
            surface_current_pos = surface_current_pos[pos_surface_current_mask]
            target_surface_current = target_surface_current[pos_surface_current_mask]

            if self.config.get('pointcloud_model'):
                # interpolate the predicted currents (defined at the metal graph nodes) to the raw
                # surface-current sample positions and compare with the target there.
                target_surface_current, surface_current_pos = filter_surface_current(target_surface_current, surface_current_pos, self.config.get('threshold', 0.5))
                graph_pos_metal = graph_pos[graph.pec_mask[graph_batch_mask]]
                pred_surface_current_batch_metal = pred_surface_current_batch[graph.pec_mask[graph_batch_mask]]

                interpolated_pred_currents = knn_interpolate_features(graph_pos_metal, surface_current_pos, pred_surface_current_batch_metal, k=3)
                loss = self.mse_loss(interpolated_pred_currents, target_surface_current)
            else:
                # interpolate target_surface_current to graph_pos and compare with pred_surface_current_batch:
                if self.config.get('per_component_interpolation', False):
                    # graph.edge_index/.edge_attr cover the WHOLE batch; restrict to
                    # this batch item's subgraph (relabeled to match graph_pos's
                    # local node ordering, i.e. graph_batch_mask's node order) before
                    # deriving mesh-only component labels.
                    node_idx = graph_batch_mask.nonzero(as_tuple=True)[0]
                    sub_edge_index, sub_edge_attr = subgraph(
                        node_idx, graph.edge_index, edge_attr=getattr(graph, 'edge_attr', None),
                        relabel_nodes=True, num_nodes=graph.num_nodes)
                    interpolated_target_currents = interpolate_target_currents_per_component(
                        surface_current_pos, target_surface_current, graph_pos, sub_edge_index, sub_edge_attr, k=5)
                else:
                    interpolated_target_currents = knn_interpolate_features(surface_current_pos, graph_pos, target_surface_current, k=5)

                loss = self.mse_loss(pred_surface_current_batch, interpolated_target_currents)

            loss_avg += loss
        loss_avg /= graph.num_graphs
        return loss_avg


class SurfaceCurrentLoss(torch.nn.Module):
    """Hetero-graph variant, selected by FFAndSurfaceCurrentLoss when
    config['hetero_graph'] is set. NOTE: this class accumulates loss_avg over the
    batch but returns the LAST item's loss (preserved as-is from the original
    implementation)."""
    def __init__(self):
        super(SurfaceCurrentLoss, self).__init__()
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, graph, pred_surface_current):
        '''
        will find all nodes of type PEC and then calculate MSE loss between the predicted surface current and the target surface current at those positions.
        '''
        loss_avg = 0
        for batch_idx in range(graph.num_graphs):
            surface_current_pos = graph.pos_surface_current
            target_surface_current = graph.surface_currents
            graph_batch_mask = graph['metal'].batch == batch_idx
            graph_pos = graph['metal'].x[graph_batch_mask, :3]
            pred_surface_current_batch = pred_surface_current[graph_batch_mask]
            pos_surface_current_mask = (graph.pos_surface_current_batch == batch_idx)
            surface_current_pos = surface_current_pos[pos_surface_current_mask]
            target_surface_current = target_surface_current[pos_surface_current_mask]

            # interpolate target_surface_current to graph_pos and compare with pred_surface_current_batch:
            interpolated_target_currents = knn_interpolate_features(surface_current_pos, graph_pos, target_surface_current, k=3)
            loss = self.mse_loss(pred_surface_current_batch, interpolated_target_currents)
            loss_avg += loss
        loss_avg /= graph.num_graphs

        return loss


class FFAndSurfaceCurrentLoss(torch.nn.Module):
    def __init__(self, config, ff_loss_weight=1.0, surface_current_loss_weight=0.5):
        super(FFAndSurfaceCurrentLoss, self).__init__()
        self.ff_loss_function = FarfeildLoss(config, {}) # ff stats should be passed here if needed for normalization, but currently not used in the loss function
        if config.get('hetero_graph'):
            self.surface_current_loss_function = SurfaceCurrentLoss() # for hetero graph, we can directly calculate loss at the PEC nodes since they are included in the graph and have target surface current values.
        else:
            self.surface_current_loss_function = SurfaceCurrentLossOutputInterpulation(config)
        self.ff_loss_weight = ff_loss_weight
        self.surface_current_loss_weight = surface_current_loss_weight

    def forward(self, graph, pred_surface_current, pred_ff_image, target_ff_image):
        ff_loss = self.ff_loss_function(pred_ff_image, target_ff_image)
        surface_current_loss = self.surface_current_loss_function(graph, pred_surface_current)

        combined_loss = self.ff_loss_weight * ff_loss + self.surface_current_loss_weight * surface_current_loss
        return combined_loss, ff_loss, surface_current_loss


class FFAndSurfaceCurrentAndPhysicsLoss(torch.nn.Module):
    def __init__(self, config, ff_loss_weight=1.0, surface_current_loss_weight=0.5, physics_loss_weight=0.3, constincy_loss_weight=0.0, ff_image_shape=(34, 34), device='cpu'):
        super(FFAndSurfaceCurrentAndPhysicsLoss, self).__init__()
        self.ff_loss_function = FarfeildLoss(config, {}) # ff stats should be passed here if needed for normalization, but currently not used in the loss function
        self.surface_current_loss_function = SurfaceCurrentLossOutputInterpulation(config)
        self.ff_loss_weight = ff_loss_weight
        self.surface_current_loss_weight = surface_current_loss_weight
        self.physics_loss_weight = physics_loss_weight
        self.constincy_loss_weight = constincy_loss_weight
        self.ff_image_shape = ff_image_shape

        warmup_steps = 2000
        self.warmup_steps = warmup_steps
        self.warmup = max(1, warmup_steps)
        self.register_buffer('_step', torch.tensor(0, dtype=torch.long))

    def forward(self, graph, pred_surface_current, pred_ff_image, ff_from_sc, target_ff_image):
        ff_loss = self.ff_loss_function(pred_ff_image, target_ff_image)
        surface_current_loss = self.surface_current_loss_function(graph, pred_surface_current)
        physics_loss = self.ff_loss_function(ff_from_sc, target_ff_image, train=False)  # compare ff from surface currents to target ff, without backpropagating through the surface current prediction
        constincy_loss = self.ff_loss_function(pred_ff_image, ff_from_sc, train=False)  # currently not in use
        physics_loss = physics_loss * min(1.0, self._step / self.warmup)  # Linear warmup for physics loss
        constincy_loss = constincy_loss * min(1.0, self._step / self.warmup)  # Linear warmup for consistency loss
        combined_loss = self.ff_loss_weight * ff_loss + self.surface_current_loss_weight * surface_current_loss + self.physics_loss_weight * physics_loss + self.constincy_loss_weight * constincy_loss
        self._step += 1
        return combined_loss, ff_loss, surface_current_loss
