
import time
from src.graph.graph_functions import (compute_geometric_edge_attr, knn_component_bridge_edges,
                                        connected_components_pytorch, count_localized_pe_columns)
from src.geometry.mesh_functions import decompose_pyg_graph, add_PEC_to_node_type, get_icosphere
from src.geometry.mesh_functions_pytorch_2 import merge_and_connect_graphs_from_dict
import trimesh
from src.dataset.dataloader_utils import load_obj_as_graph
from torch_geometric.transforms  import ToUndirected, AddSelfLoops, RadiusGraph, AddLaplacianEigenvectorPE
from torch_geometric.data import Batch
import torch
from src.graph.graph_functions import graph_dict_merging_duplicate_nodes


# Module-level state for the strict_pe_check tripwire (Task 3b/B): a per-process
# sample of the first 10 calls' pass/fail *and* wall-time, used to decide whether the
# check is cheap enough to keep running on every call, or should cap itself. Not
# exhaustive verification -- a tripwire against a future caching regression
# reintroducing a stale/pre-bridging pe.
_PE_TRIPWIRE_STATE = {'count': 0, 'total_time': 0.0, 'capped': False}


def _check_pe_tripwire(graph, k):
    state = _PE_TRIPWIRE_STATE
    if state['capped']:
        return
    t0 = time.perf_counter()
    num_nodes = graph.pos.shape[0]
    # full (mesh + bridge) connectivity: graph.edge_index at this point already
    # includes bridge edges (added before ToUndirected, above) unioned with mesh/
    # radius-graph edges. Pre-symmetrize -- see knn_component_bridge_edges for why.
    sym_edge_index = (torch.cat([graph.edge_index, graph.edge_index.flip(0)], dim=1)
                       if graph.edge_index.numel() else graph.edge_index)
    n_components, labels = connected_components_pytorch(sym_edge_index, num_nodes)
    localized = count_localized_pe_columns(graph.pe, labels, n_components)
    elapsed = time.perf_counter() - t0
    assert localized == 0, (
        f"prepare_graph strict_pe_check tripwire: {localized}/{k} pe column(s) are localized to a single "
        f"connected component (of {n_components} components) after connect_components bridging -- the "
        f"bridged-graph LapPE refresh appears stale or incorrect. This should never happen once bridging "
        f"has fully connected the graph; check for a caching regression reintroducing a pre-bridging pe.")
    state['count'] += 1
    state['total_time'] += elapsed
    if state['count'] == 10:
        avg_ms = (state['total_time'] / state['count']) * 1000
        if avg_ms > 2.0:
            state['capped'] = True
            print(f"[prepare_graph] strict_pe_check tripwire passed on the first {state['count']} calls "
                  f"(avg {avg_ms:.2f} ms/call > 2ms threshold) -- capping further checks this process to "
                  f"avoid per-call overhead. This is a tripwire, not exhaustive verification.")


def prepare_graph(graph, config):
    # save the positional encoding if it exists:
    if hasattr(graph, 'pe'):
        pe = graph.pe
        has_pe = True
    else:
        has_pe = False
    # single graph
    add_sphere = config.get('add_sphere', True)
    # Decompose the graph
    graph_dict = decompose_pyg_graph(graph)
    graph_dict = add_PEC_to_node_type(graph_dict,device=graph.pos.device)
    if config.get('merge_duplicate_nodes'):
        graph_dict = graph_dict_merging_duplicate_nodes(graph_dict)
    
    for key in graph_dict:
        if graph_dict[key] is not None and graph_dict[key].num_nodes > 0:
            graph_dict[key].x = torch.cat([graph_dict[key].pos, graph_dict[key].node_normals, graph_dict[key].node_type], dim=1)
            graph_dict[key] = compute_geometric_edge_attr(graph_dict[key])

    
    if add_sphere:
        # add the sphere graph to the dict:
        vertices, faces = get_icosphere(num_sphere_vertices=config.get('num_sphere_nodes'),radius=config.get('sphere_radius'))
        sphere_mesh  = trimesh.Trimesh(vertices=vertices, faces=faces)
        graph_dict['sphere'] = load_obj_as_graph(sphere_mesh, 'sphere').to(graph.pos.device)
    else:
        graph_dict['sphere'] = []
    


    graph = merge_and_connect_graphs_from_dict(graph_dict, 'sphere', k=2, add_sphere=add_sphere)
    
    # set edges:
    if config.get('add_radius_graph_edges'):
        original_edge_index = graph.edge_index
        rg = RadiusGraph(1)
        graph = rg(graph)
        graph.edge_index = torch.cat([original_edge_index, graph.edge_index], dim=1)

    # connect_components: {mode: 'knn', k: int} -- OFF by default (config key absent/
    # falsy leaves everything below bit-identical to before this option existed).
    # When on, bridges every pair of distinct connected components with k nearest
    # cross-component node pairs (Euclidean, undirected) *before* ToUndirected/LapPE,
    # so downstream LapPE sees a single connected component. bridge_edge_key is a
    # (row * num_nodes + col) encoding of the bridge edges, looked up again after
    # ToUndirected + compute_geometric_edge_attr (which may reorder/coalesce edges)
    # to build the mesh=0/bridge=1 edge-type column -- robust regardless of how
    # ToUndirected dedupes, rather than trying to carry a mask tensor through it.
    connect_components_cfg = config.get('connect_components')
    bridge_edge_key = None
    if connect_components_cfg:
        num_nodes = graph.pos.shape[0]
        bridge_edge_index = knn_component_bridge_edges(
            graph.pos, graph.edge_index,
            mode=connect_components_cfg.get('mode', 'knn'),
            k=connect_components_cfg.get('k', 3),
        )
        if bridge_edge_index.numel() > 0:
            graph.edge_index = torch.cat([graph.edge_index, bridge_edge_index], dim=1)
            bridge_edge_key = bridge_edge_index[0] * num_nodes + bridge_edge_index[1]

    to_undirected = ToUndirected()
    graph = to_undirected(graph)
    if has_pe:
        if connect_components_cfg:
            # `pe` here is whatever was already on the input graph -- for PixelDataset
            # that's the pre_transform-computed, process()-time-cached LapPE on the
            # *raw* (pre-bridging) graph, now stale: bridging changes connectivity, so
            # a pe computed before it decouples into per-component blocks (see
            # check_lap_pe in validate_wire_pipeline.py) exactly like the disconnected
            # case connect_components exists to fix. Recompute fresh here, on the final
            # bridged+undirected graph, at the same k so downstream shapes don't change,
            # overwriting the stale cached value. (TrimeshDataset passes has_pe=False
            # into this function -- it computes pe itself, after prepare_graph, via its
            # own transform, so it already reflects bridging without needing this path.)
            k = pe.shape[-1]
            graph = AddLaplacianEigenvectorPE(k=k, attr_name='pe')(graph)
            if config.get('strict_pe_check', True):
                _check_pe_tripwire(graph, k)
        else:
            graph.pe = pe
    graph = compute_geometric_edge_attr(graph)

    if connect_components_cfg:
        num_nodes = graph.pos.shape[0]
        final_key = graph.edge_index[0] * num_nodes + graph.edge_index[1]
        if bridge_edge_key is not None and bridge_edge_key.numel() > 0:
            is_bridge = torch.isin(final_key, bridge_edge_key.to(final_key.device))
        else:
            is_bridge = torch.zeros_like(final_key, dtype=torch.bool)
        edge_type = is_bridge.to(graph.edge_attr.dtype).unsqueeze(-1)
        graph.edge_attr = torch.cat([graph.edge_attr, edge_type], dim=-1)

    # set node probs:
    if config.get('use_node_probs') == False:
        graph.node_probs = torch.ones(graph.pos.size(0),1).to(graph.pos.device)
    # asl = AddSelfLoops()
    # graph = asl(graph)
    return graph, graph_dict


def compose_batch_graph(graph_batch, config, device='cpu'):
    """ compose a batch of graphs with sphere added and connected or without sphere
    Input:
        data: a batch of graphs from dataloader
        config: configuration dict
    Output:
        batched_graph: a batch of graphs
        graph_dict_list: a list of decomposed graphs 
    """
    graph_batch_list = graph_batch.to_data_list()
    graph_dict_list = []
    new_graph_list = []
    for idx, graph in enumerate(graph_batch_list):
        graph, graph_dict = prepare_graph(graph, config)
        new_graph_list.append(graph)
        graph_dict_list.append(graph_dict)
    batched_graph = Batch.from_data_list(new_graph_list)
    return batched_graph, graph_dict_list



import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import coalesce
from torch_scatter import scatter

def merge_nodes_by_position_explicit(data: Data, 
                                     type_col_index: int = None, 
                                     probs_col_index: int = None) -> Data:
    """
    Merges nodes sharing the same 'pos', handling features independently.
    
    Assumes data.x is [pos, node_normals, node_type, node_probs].
    
    Args:
        data: The input PyG graph.
        type_col_index: The index in 'x' where node_type starts.
        probs_col_index: The index in 'x' where node_probs starts.
        
    Returns:
        Data: New graph with merged nodes.
    """
    
    # 1. Find Unique Positions
    # inverse_indices maps every old node to its new cluster index
    unique_pos, inverse_indices = torch.unique(data.pos, dim=0, return_inverse=True)
    num_new_nodes = unique_pos.size(0)

    # 2. Slice the input tensor 'x'
    # We assume standard 3D dimensions for pos and normals
    # x structure: [pos (3), normals (3), type (1), probs (1)]
    # You can adjust these slice indices if your dimensions differ.
    
    # Slice: Positions (3 dims)
    # We already have unique_pos, so we don't strictly need to slice/aggregate this from x,
    # but for consistency with x's structure:
    old_pos = data.pos
    
    # Slice: Normals (3 dims)
    old_normals = data.node_normals
    
    # Slice: Type (1 dim) - Assumed to be at index 6
    old_types = data.node_type
    
    # Slice: Probs (1 dim) - Assumed to be at index 7
    old_probs = data.node_probs

    # 3. Aggregate Features Independently
    
    # A. Normals: Mean aggregation + Re-normalization
    # We average normals of merged nodes, then normalize to ensure length 1.
    new_normals = scatter(old_normals, inverse_indices, dim=0, dim_size=num_new_nodes, reduce='mean')
    new_normals = F.normalize(new_normals, p=2, dim=1)

    # B. Types: 'Max' aggregation (or Mode)
    # For categorical integers, 'mean' is invalid. 
    # 'max' is a simple heuristic (picks the higher class ID).
    # If you want the most frequent type, that requires a more complex 'mode' operation.
    new_types = scatter(old_types, inverse_indices, dim=0, dim_size=num_new_nodes, reduce='mean')

    # C. Probs: Mean aggregation
    # Probabilities should generally be averaged.
    new_probs = scatter(old_probs, inverse_indices, dim=0, dim_size=num_new_nodes, reduce='mean')

    # 4. Re-concatenate to form the new 'x'
    # Structure: [unique_pos, new_normals, new_types, new_probs]
    new_x = torch.cat([unique_pos, new_normals, new_types, new_probs], dim=1)

    # 5. Remap Edges
    new_edge_index = inverse_indices[data.edge_index]
    
    new_edge_index, new_edge_attr = coalesce(
        new_edge_index, 
        edge_attr=data.edge_attr, 
        num_nodes=num_new_nodes,
        reduce='mean' # Average edge features if collisions occur
    )

    # 6. Create New Data Object
    new_data = Data(
        x=new_x,
        pos=unique_pos,
        node_normals=new_normals,
        node_type=new_types,
        node_probs=new_probs,
        edge_index=new_edge_index,
        edge_attr=new_edge_attr
    )

    return new_data
            
            

