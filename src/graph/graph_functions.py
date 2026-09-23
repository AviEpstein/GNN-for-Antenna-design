from torch_geometric.utils import scatter
import torch


def custom_global_pool( x, node_probs, batch, pool_type='mean'):
    """Custom global pooling function that considers node probabilities.
    Args:
        x (_type_): _description_
        node_probs (_type_): _description_
        batch (_type_): _description_
        pool_type (str, optional): _description_. Defaults to 'mean'.

    Returns:
        _type_: _description_
    """
    if pool_type == 'mean':
        bias = torch.log(torch.clamp(node_probs, min=1e-6))
        weighted_x = x + bias  # Weight features by node probabilities
        sum_x = scatter(weighted_x, batch, dim=0, reduce='sum')
        sum_probs = scatter(node_probs, batch, dim=0, reduce='sum') + 1e-8  # Avoid division by zero
        pooled = sum_x / sum_probs
    elif pool_type == 'max':
        # For max pooling, we can use a mask to ignore low-probability nodes
        masked_x = x.clone()
        masked_x[node_probs < 1e-5] = float('-inf')  # Set low-probability nodes to -inf
        pooled = scatter(masked_x, batch, dim=0, reduce='max')
    else:
        raise ValueError(f"Unsupported pool_type: {pool_type}")
    return pooled



import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_scipy_sparse_matrix, subgraph


def connected_components_pytorch(edge_index, num_nodes):
    """
    Finds the connected components of an undirected graph using purely PyTorch operations.
    
    Args:
        edge_index (Tensor): Shape [2, num_edges].
        num_nodes (int): Total number of nodes in the graph.
        
    Returns:
        num_components (int): The total number of disconnected subgraphs.
        component_labels (Tensor): Shape [num_nodes], labeling each node with its component ID.
    """
    device = edge_index.device
    
    # 1. Handle edge case of a graph with no edges
    if edge_index.size(1) == 0:
        return num_nodes, torch.arange(num_nodes, device=device)
        
    # 2. Symmetrize the edge_index to ensure the graph is undirected
    row, col = edge_index
    sym_edge_index = torch.cat([edge_index, torch.stack([col, row])], dim=1)
    sym_edge_index = edge_index
    
    # 3. Initialize each node's label as its own node ID
    labels = torch.arange(num_nodes, dtype=torch.long, device=device)
    
    # 4. Label Propagation: Iteratively update labels to the maximum neighbor label
    while True:
        prev_labels = labels.clone()
        
        # Get the current labels of all source nodes
        neighbor_labels = labels[sym_edge_index[0]]
        
        # Scatter the maximum label to the destination nodes
        labels.scatter_reduce_(
            dim=0, 
            index=sym_edge_index[1], 
            src=neighbor_labels, 
            reduce="amax", 
            include_self=True
        )
        
        # If no labels changed in this iteration, we've converged!
        if torch.equal(labels, prev_labels):
            break
            
    # 5. Format the output to strictly match SciPy's behavior (labels from 0 to N-1)
    unique_labels, component_labels = torch.unique(labels, return_inverse=True)
    num_components = unique_labels.size(0)
    
    return num_components, component_labels

def count_localized_pe_columns(pe, component_labels, n_components):
    """Counts how many columns of a Laplacian-eigenvector positional encoding (pe) are
    localized to a single connected component -- i.e. carry zero cross-component
    positional signal, which breaks LapPE for structural reasoning across disconnected
    antenna elements (director/reflector/driven spacing on a Yagi, etc.).

    A graph with C disconnected components decouples into C independent Laplacian
    blocks: every eigenvector of the whole graph is a single component's own
    eigenvector, zero-padded everywhere else. Detected per column by counting how many
    of the C components are internally flat (std < 1e-3, or <=1 node); if all but at
    most one component is flat, that column carries no cross-component signal.

    This is the single shared implementation of the check originally written inline in
    validate_wire_pipeline.py's check_lap_pe() -- other callers (prepare_graph's
    connect_components strict_pe_check tripwire, pipeline_validation/test_pe_bridging.py)
    should import this rather than re-deriving the loop.

    Args:
        pe (Tensor): [num_nodes, k] positional encoding.
        component_labels (Tensor or array-like): [num_nodes] integer component id per
            node, in the range [0, n_components).
        n_components (int): total number of connected components.

    Returns:
        int: number of columns (0..k) localized to a single component. Always 0 when
        n_components <= 1 (nothing to be localized *relative to* with a single
        component -- that is the success state connect_components bridging aims for).
    """
    if n_components <= 1:
        return 0
    comp_t = torch.as_tensor(component_labels, dtype=torch.long, device=pe.device)
    localized_cols = 0
    for j in range(pe.shape[1]):
        col = pe[:, j]
        n_flat = 0
        for c in range(n_components):
            mask = comp_t == c
            if mask.sum() <= 1 or float(col[mask].std().item()) < 1e-3:
                n_flat += 1
        if n_flat >= n_components - 1:
            localized_cols += 1
    return localized_cols


def knn_component_bridge_edges(pos, edge_index, mode='knn', k=3):
    """Finds every pair of distinct connected components in (pos, edge_index) and, for
    each pair, adds edges between the k nearest cross-component node pairs by Euclidean
    distance (the k globally-smallest pairwise distances between the two components'
    node sets, not k-per-node), in both directions.

    Args:
        pos (Tensor): [num_nodes, 3] node positions.
        edge_index (Tensor): [2, num_edges] current edges (used only for component
            detection -- not modified).
        mode (str): only 'knn' is currently supported.
        k (int): number of nearest cross-component node pairs to bridge, per pair of
            components.

    Returns:
        bridge_edge_index (Tensor): [2, num_bridge_edges] new edges to add (both
            directions already included), or an empty [2, 0] tensor if the graph is
            already a single connected component. Does not include the original edges.
    """
    if mode != 'knn':
        raise ValueError(f"connect_components mode {mode!r} is not supported (only 'knn' is implemented)")

    device = pos.device
    num_nodes = pos.shape[0]
    # connected_components_pytorch does not actually symmetrize its input (see its
    # docstring vs. implementation -- the symmetrized edge_index it builds is
    # immediately discarded and only the given, possibly one-directional, edge_index
    # is used for label propagation). Mesh edges here are frequently one-directional
    # pre-ToUndirected (e.g. one edge per triangle side), so pre-symmetrize ourselves
    # to get a correct *undirected* component count -- matching what
    # scipy.sparse.csgraph.connected_components(..., directed=False) would report,
    # which is what the acceptance check (validate_wire_pipeline.py) actually uses.
    sym_edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1) if edge_index.numel() else edge_index
    num_components, labels = connected_components_pytorch(sym_edge_index, num_nodes)
    if num_components <= 1:
        return torch.zeros((2, 0), dtype=torch.long, device=device)

    src_parts, dst_parts = [], []
    for a in range(num_components):
        a_nodes = torch.where(labels == a)[0]
        if a_nodes.numel() == 0:
            continue
        for b in range(a + 1, num_components):
            b_nodes = torch.where(labels == b)[0]
            if b_nodes.numel() == 0:
                continue
            dmat = torch.cdist(pos[a_nodes], pos[b_nodes])  # [nA, nB]
            k_eff = min(k, dmat.numel())
            flat_idx = torch.topk(dmat.reshape(-1), k=k_eff, largest=False).indices
            row = torch.div(flat_idx, dmat.shape[1], rounding_mode='floor')
            col = flat_idx % dmat.shape[1]
            src = a_nodes[row]
            dst = b_nodes[col]
            # undirected: add both directions ourselves so bridge-edge identity
            # survives ToUndirected()/coalescing downstream regardless of its
            # internal reduce behavior.
            src_parts.append(torch.cat([src, dst]))
            dst_parts.append(torch.cat([dst, src]))

    if not src_parts:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    return torch.stack([torch.cat(src_parts), torch.cat(dst_parts)], dim=0)


def remove_redondent_nodes_pytorch(data):
    """
    remove redondent nodes that have the same position and normal and node type, by merging them into one node with node_prob attribute that indicates how many nodes were merged.
    """
    # 1. Collect features to check for uniqueness (pos, normal, node_type)
    feats_to_check = [data.pos]
    if hasattr(data, 'node_normals') and data.node_normals is not None:
        feats_to_check.append(data.node_normals)
    if hasattr(data, 'node_type') and data.node_type is not None:
        # Ensure node_type is float/compatible or convert all to string/bytes if mixed, 
        # but usually tensors are same type. Just concatenation for now.
        feats_to_check.append(data.node_type)
    
    # helper to ensure 2D for cat
    feats_to_check = [f if f.dim() > 1 else f.unsqueeze(-1) for f in feats_to_check]
    
    # Concatenate features along dimension 1
    combined_features = torch.cat(feats_to_check, dim=1)

    # 2. Find unique nodes
    # return_inverse gives us the mapping from old_idx -> new_unique_idx
    # return_counts gives us how many duplicates were merged
    unique_features, inverse_indices, counts = torch.unique(
        combined_features, dim=0, return_inverse=True, return_counts=True
    )
    
    num_unique_nodes = unique_features.size(0)

    # 3. Create new attributes
    # The 'counts' tensor is effectively our unnormalized probability (or weight)
    new_node_probs = counts.float().unsqueeze(-1)
    
    # For 'pos', 'node_normals', 'node_type', we can just take the unique values 
    # (reconstructing from unique_features would be complex if sizes differ, 
    # but we can scatter/select based on inverse_indices).
    # A safer way to preserve separate attributes: Select the first occurrence of each unique node.
    # However, torch.unique sorts results, so we can't just pick index 0..N.
    # instead scatter_reduce with "mean" (since they are identical) or just reconstruct from unique_features?
    # Simpler approach: scatter mean for float attributes.
    
    new_pos = scatter(data.pos, inverse_indices, dim=0, reduce='mean')
    
    new_data = Data(num_nodes=num_unique_nodes)
    new_data.pos = new_pos
    new_data.node_probs = new_node_probs

    if hasattr(data, 'x') and data.x is not None:
        # Sum features of merged nodes, or mean? Usually sum for "pooling" logic, or mean if intensive.
        # Let's use mean to preserve scale, or sum if count matters. 
        # Given "node_probs" captures count, mean seems safer for 'x'.
        new_data.x = scatter(data.x, inverse_indices, dim=0, reduce='mean')
    
    if hasattr(data, 'node_normals') and data.node_normals is not None:
         new_data.node_normals = scatter(data.node_normals, inverse_indices, dim=0, reduce='mean')

    if hasattr(data, 'node_type') and data.node_type is not None:
        # node_type should be identical for merged nodes, so 'max' or 'min' or 'mean' works
        new_data.node_type = scatter(data.node_type, inverse_indices, dim=0, reduce='max')
    
    if hasattr(data, 'pe') and data.pe is not None:
        new_data.pe = scatter(data.pe, inverse_indices, dim=0, reduce='mean')

    # 4. Remap Edge Indices
    # The inverse_indices tensor is exactly the mapping: old_node_idx -> new_node_idx
    row, col = data.edge_index
    new_row = inverse_indices[row]
    new_col = inverse_indices[col]
    
    # Filter self-loops if merging created them? Usually we keep them or remove them.
    # Let's keep edges as is, just remapped.
    # Remove duplicates edges that might result from merging nodes
    new_edge_index = torch.stack([new_row, new_col], dim=0)
    new_edge_index = torch.unique(new_edge_index, dim=1) # Remove duplicate edges
    new_data.edge_index = new_edge_index
    
    # 5. remap edge attributes if they exist
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        edge_attr = data.edge_attr
        # We need to aggregate edge attributes for edges that got merged together.
        new_edge_indices_for_old_edges = inverse_indices[row] * num_unique_nodes + inverse_indices[col]
        unique_new_edge_indices, inverse_edge_indices = torch.unique(new_edge_indices_for_old_edges, return_inverse=True)   
        new_edge_attr = scatter(edge_attr, inverse_edge_indices, dim=0, reduce='mean')
        
        # Make edges bidirectional if they aren't already
        row, col = new_data.edge_index
        symmetric_edge_index = torch.stack([torch.cat([row, col]), torch.cat([col, row])], dim=0)
        symmetric_edge_attr = torch.cat([new_edge_attr, new_edge_attr], dim=0)
        
        # Remove duplicates
        new_edge_index, unique_edge_indices = torch.unique(symmetric_edge_index, dim=1, return_inverse=True)
        final_edge_attr = scatter(symmetric_edge_attr, unique_edge_indices, dim=0, reduce='mean')
        
        # new_data.edge_index = new_edge_index
        new_data.edge_attr = final_edge_attr
    
    return new_data
    




def split_disconnected_graphs_pytorch(graph_dict):
    """
    Takes a dict of PyG Data objects and returns a new dict where 
    all disconnected subgraphs are split into their own separate graphs.
    """
    new_dict = {}

    for key, data in graph_dict.items():
        if data == []:
            continue
        data = remove_redondent_nodes_pytorch(data)
        
        # data = compute_geometric_edge_attr(data)

        num_nodes = data.pos.size(0)

        # 1. Convert edge_index to a sparse matrix to find connected components
        num_components, component_labels = connected_components_pytorch(data.edge_index, num_nodes)
        
        # 2. Iterate over each discovered component
        for i in range(num_components):
            # component_labels is already a PyTorch tensor on the correct device!
            subset = torch.where(component_labels == i)[0]
            
            # Skip empty components (safety check)
            if subset.size(0) == 0:
                continue

            # 3. Extract and relabel the edge_index
            # new_edge_index, _ = subgraph(
            #     subset, 
            #     data.edge_index, 
            #     relabel_nodes=True, 
            #     num_nodes=num_nodes
            # )
            # Extract and relabel the eddge attributes if they exist
            edge_attr = None
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                edge_attr = data.edge_attr
                new_edge_index, new_edge_attr = subgraph(
                    subset, 
                    data.edge_index, 
                    edge_attr=edge_attr, 
                    relabel_nodes=True, 
                    num_nodes=num_nodes
                )
                edge_attr = new_edge_attr

            new_x = data.x[subset]

            # 5. Extract and remap faces (triangles)
            new_faces = None
            if hasattr(data, 'faces') and data.faces is not None:
                # Create a mapping array. Old node index -> New node index.
                # Nodes not in this subgraph get mapped to -1.
                mapping = torch.full((num_nodes,), -1, dtype=torch.long)
                mapping[subset] = torch.arange(subset.size(0), dtype=torch.long)

                # Map the faces (assuming shape [3, num_faces] standard to PyG)
                mapped_faces = mapping[data.faces]

                # Keep only the faces where ALL 3 vertices exist in this subgraph
                # (i.e., none of the vertices mapped to -1)
                valid_face_mask = (mapped_faces != -1).all(dim=0)
                new_faces = mapped_faces[:, valid_face_mask]
            
            # 6. Extract and remap node_normals if they exist
            new_node_normals = None
            if hasattr(data, 'node_normals') and data.node_normals is not None:
                new_node_normals = data.node_normals[subset]
            
            # 7. Extract the new positions
            new_pos = data.pos[subset]

            # 8. Extract and remap node types if they exist
            new_node_type = None
            if hasattr(data, 'node_type') and data.node_type is not None:
                new_node_type = data.node_type[subset]
            
            # 8. Extract and remap pe positional encodings if they exist
            new_pe = None
            if hasattr(data, 'pe') and data.pe is not None:
                new_pe = data.pe[subset]
            
            # 8. Extract and remap node_probs if they exist
            if hasattr(data, 'node_probs') and data.node_probs is not None:
                #TODO if needed fix:
                new_node_probs = torch.ones(subset.size(0), device=data.node_probs.device).unsqueeze(-1) # default to 1 for all nodes in this subgraph
            

            # 9. Construct the new Data object
            new_data = Data(x=new_x, edge_index=new_edge_index, pos=new_pos, edge_attr=edge_attr)
            if new_node_normals is not None:
                new_data.node_normals = new_node_normals
            if new_faces is not None:
                new_data.faces = new_faces
            if new_node_type is not None:
                new_data.node_type = new_node_type
            if hasattr(data, 'pe') and data.pe is not None:
                new_data.pe = new_pe
            if hasattr(data, 'node_probs') and data.node_probs is not None:
                new_data.node_probs = new_node_probs

            # 7. Add to the new dictionary with a unique key
            new_key = f"{key}_subgraph_{i}"
            new_dict[new_key] = new_data

    return new_dict



def compute_geometric_edge_attr_dict(graph_dict) -> dict:
    """
    Computes distance and direction vectors for edges and 
    assigns them to the graph.edge_attr.
    """
    # Check if the graph has positions and edges
    for key, graph in graph_dict.items():
        graph_dict[key] = compute_geometric_edge_attr(graph)
    return graph_dict


def compute_geometric_edge_attr(graph):
    if not hasattr(graph, 'pos') or not hasattr(graph, 'edge_index'):
        print("Graph is missing 'pos' or 'edge_index'. Skipping.")
        return graph

    pos = graph.pos
    edge_index = graph.edge_index

    # Get the source and target node indices for each edge
    row, col = edge_index
    
    # Extract the (x, y, z) coordinates for source and target nodes
    pos_source = pos[row]
    pos_target = pos[col]
    
    # 1. Calculate the Direction Vector (Delta x, Delta y, Delta z)
    direction_vector = pos_target - pos_source
    
    # 2. Calculate the Euclidean Distance (R)
    # Using keepdim=True so it remains shape [num_edges, 1] instead of [num_edges]
    distance = torch.norm(direction_vector, p=2, dim=-1, keepdim=True)
    
    # (Optional) Normalize the direction vector
    # This is often preferred so the network separates magnitude from direction
    # Add a small epsilon to avoid division by zero
    normalized_direction = direction_vector / (distance + 1e-8)
    
    # 3. Concatenate to create the final edge_attr tensor
    # Shape will be [num_edges, 4] -> [Distance, Dir_X, Dir_Y, Dir_Z]
    edge_attr = torch.cat([distance, normalized_direction], dim=-1)
    
    # Assign to the graph
    graph.edge_attr = edge_attr
    return graph


import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, remove_self_loops

def merge_duplicate_nodes(data: Data, tolerance: float = 1e-5) -> Data:
    """
    Merges positionally duplicated nodes in a torch_geometric Data object.

    Args:
        data (Data): The input PyG Data object.
        tolerance (float): Tolerance for positional equality to handle float inaccuracies.

    Returns:
        Data: A new PyG Data object with merged nodes.
    """
    # Round positions to handle floating point inaccuracies
    rounded_pos = torch.round(data.pos / tolerance) * tolerance

    # Find unique positions and get the mapping from old to new indices
    unique_pos, inverse_indices = torch.unique(rounded_pos, dim=0, return_inverse=True)

    num_old_nodes = data.pos.size(0)
    num_new_nodes = unique_pos.size(0)

    if num_new_nodes == num_old_nodes:
        # print("No duplicated nodes found.")
        return data

    # print(f"Merged {num_old_nodes - num_new_nodes} duplicated nodes. New node count: {num_new_nodes}")

    # 1. Update edge_index: map old node IDs to new node IDs
    new_edge_index = inverse_indices[data.edge_index]

    # 2. Clean up edges: remove self-loops and merge duplicate edges
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        new_edge_index, new_edge_attr = remove_self_loops(new_edge_index, data.edge_attr)
        new_edge_index, new_edge_attr = coalesce(new_edge_index, new_edge_attr, num_nodes=num_new_nodes, reduce='mean')
    else:
        new_edge_index, _ = remove_self_loops(new_edge_index)
        new_edge_index = coalesce(new_edge_index, num_nodes=num_new_nodes)
        new_edge_attr = None

    # 3. Create a mapping from new to old indices to transfer node features
    # This keeps the features of the last duplicate found for each unique position
    new_to_old = torch.empty(num_new_nodes, dtype=torch.long, device=data.pos.device)
    new_to_old[inverse_indices] = torch.arange(num_old_nodes, device=data.pos.device)

    # 4. Build the new Data object
    new_data_kwargs = {}
    for key, value in data:
        if key == 'edge_index':
            new_data_kwargs[key] = new_edge_index
        elif key == 'edge_attr':
            new_data_kwargs[key] = new_edge_attr
        elif key == 'pos':
            # Use the original exact positions from the representative nodes
            new_data_kwargs[key] = data.pos[new_to_old]
        elif isinstance(value, torch.Tensor) and value.size(0) == num_old_nodes:
            # Filter node-level attributes (x, node_type, node_normals, etc.)
            new_data_kwargs[key] = value[new_to_old]
        else:
            # Keep graph-level attributes (like faces or sphere data)
            new_data_kwargs[key] = value

    return Data(**new_data_kwargs)

def graph_dict_merging_duplicate_nodes(graph_dict):
    """Applies merge_duplicate_nodes to each graph in the dictionary."""
    new_graph_dict = {}
    for key, graph in graph_dict.items():
        new_graph_dict[key] = merge_duplicate_nodes(graph)
    return new_graph_dict

