import torch
import torch.nn as nn
import torch.optim as optim


def process_surface_current(surface_current_data):
    #  Prepare surface current targets and their coordinates (y_surface_current, pos_surface_current)
    # a. Extract and concatenate Kx, Ky, Kz real/imaginary parts for y_surface_current
    KxRe = torch.tensor(surface_current_data['KxRe [A/m]'], dtype=torch.float32).unsqueeze(-1)
    KxIm = torch.tensor(surface_current_data['KxIm [A/m]'], dtype=torch.float32).unsqueeze(-1)
    KyRe = torch.tensor(surface_current_data['KyRe [A/m]'], dtype=torch.float32).unsqueeze(-1)
    KyIm = torch.tensor(surface_current_data['KyIm [A/m]'], dtype=torch.float32).unsqueeze(-1)
    KzRe = torch.tensor(surface_current_data['KzRe [A/m]'], dtype=torch.float32).unsqueeze(-1)
    KzIm = torch.tensor(surface_current_data['KzIm [A/m]'], dtype=torch.float32).unsqueeze(-1)
    surface_current = torch.cat((KxRe, KxIm, KyRe, KyIm, KzRe, KzIm), dim=-1)
    area = torch.tensor(surface_current_data['Area [mm^2]'], dtype=torch.float32).unsqueeze(-1)

    # b. Extract and concatenate x, y, z coordinates for pos_surface_current
    pos_x = torch.tensor(surface_current_data['#x [mm]'], dtype=torch.float32).unsqueeze(-1)
    pos_y = torch.tensor(surface_current_data['y [mm]'], dtype=torch.float32).unsqueeze(-1)
    pos_z = torch.tensor(surface_current_data['z [mm]'], dtype=torch.float32).unsqueeze(-1)
    pos_surface_current = torch.cat((pos_x, pos_y, pos_z), dim=-1)
    return surface_current, pos_surface_current, area


import numpy as np
import torch
from scipy.spatial import cKDTree

def map_currents_to_graph(raw_data, graph_node_positions):
    """
    Maps raw surface current data onto the graph nodes using Nearest Neighbor Interpolation.
    
    Args:
        raw_data (dict): The dictionary from your pickle file.
        graph_node_positions (Tensor or array): (N_nodes, 3) coordinates of your graph nodes.
    
    Returns:
        torch.Tensor: (N_nodes, 6) tensor of currents aligned with the graph.
    """
    # 1. Prepare Raw Source Data (The "Field")
    # Stack the 6 current components
    K_src = np.stack([
        raw_data['KxRe [A/m]'], raw_data['KxIm [A/m]'],
        raw_data['KyRe [A/m]'], raw_data['KyIm [A/m]'],
        raw_data['KzRe [A/m]'], raw_data['KzIm [A/m]']
    ], axis=-1)  # Shape: (N_source, 6)
    
    # Stack the source positions
    pos_src = np.stack([
        raw_data['#x [mm]'], raw_data['y [mm]'], raw_data['z [mm]']
    ], axis=-1)  # Shape: (N_source, 3)

    # 2. Prepare Target Data (The Graph Nodes)
    if isinstance(graph_node_positions, torch.Tensor):
        pos_target = graph_node_positions.cpu().numpy()
    else:
        pos_target = graph_node_positions

    # 3. Build KD-Tree and Query
    # This finds the index of the nearest source point for each graph node
    tree = cKDTree(pos_src)
    dist, indices = tree.query(pos_target, k=1)  # k=1 for nearest neighbor

    # 4. Map Values
    # We take the current values from the nearest source indices
    mapped_currents = K_src[indices]

    # Optional: Zero out currents if the nearest point is too far away
    # (e.g., if a graph node is in empty space far from any metal)
    # tolerance = 2.0 # mm
    # mapped_currents[dist > tolerance] = 0.0

    return torch.tensor(mapped_currents, dtype=torch.float32)

# --- Differentiable k-NN Interpolation Helper Function ---
def knn_interpolate_features(source_coords, target_coords, source_features, k=3):
    """
    Performs k-NN interpolation of features from source_coords to target_coords using PyTorch operations.
    Ensures differentiability for gradient propagation.
    Args:
        source_coords (torch.Tensor): Coordinates of source points (N_source, D).
        target_coords (torch.Tensor): Coordinates of target points (N_target, D).
        source_features (torch.Tensor): Features at source points (N_source, F).
        k (int): Number of nearest neighbors to consider for interpolation.
    Returns:
        torch.Tensor: Interpolated features at target points (N_target, F).
    """
    # Calculate squared Euclidean distances between target points and source points
    # Shape: (N_target, N_source)
    distances_sq = torch.cdist(target_coords, source_coords, p=2.0).pow(2)

    # Find the k nearest neighbors for each target point
    # torch.topk returns (values, indices)
    # values: (N_target, k) - k smallest distances
    # indices: (N_target, k) - indices of k nearest neighbors in source_coords
    distances_topk, indices_topk = torch.topk(distances_sq, k=k, largest=False, sorted=True)

    # Retrieve the features of the k nearest neighbors
    # Shape: (N_target, k, F)
    neighbor_features = source_features[indices_topk]

    # Calculate weights based on inverse distance (avoid division by zero)
    # Add a small epsilon to distances_topk to prevent division by zero for identical points
    # Handle cases where distance is zero (points are identical or very close)
    weights = 1.0 / (distances_topk + 1e-8) # shape (N_target, k)

    # Normalize weights so they sum to 1 for each target point
    weights_sum = weights.sum(dim=1, keepdim=True) # shape (N_target, 1)
    normalized_weights = weights / weights_sum # shape (N_target, k)

    # Expand normalized_weights to (N_target, k, 1) for element-wise multiplication with neighbor_features
    # Sum across the k dimension to get interpolated features
    interpolated_features = (normalized_weights.unsqueeze(-1) * neighbor_features).sum(dim=1)

    return interpolated_features




def process_surface_current_downsample(surface_current_data, downsample_factor=5):
    """
    Processes surface current data with an optional downsampling factor to save memory.
    
    Args:
        surface_current_data (dict): Dictionary containing the surface current arrays.
        downsample_factor (int): Step size for slicing (e.g., 5 means take every 5th point).
                                 Set to 1 to keep original resolution.
    """
    
    # Helper function to slice numpy array first, then convert to tensor
    # This ensures we don't create massive tensors in memory before downsampling
    def extract_and_slice(key):
        # [::downsample_factor] selects every N-th element
        raw_data = surface_current_data[key][::downsample_factor]
        return torch.tensor(raw_data, dtype=torch.float32).unsqueeze(-1)

    # a. Extract and concatenate Kx, Ky, Kz real/imaginary parts
    KxRe = extract_and_slice('KxRe [A/m]')
    KxIm = extract_and_slice('KxIm [A/m]')
    KyRe = extract_and_slice('KyRe [A/m]')
    KyIm = extract_and_slice('KyIm [A/m]')
    KzRe = extract_and_slice('KzRe [A/m]')
    KzIm = extract_and_slice('KzIm [A/m]')
    
    surface_current = torch.cat((KxRe, KxIm, KyRe, KyIm, KzRe, KzIm), dim=-1)

    # b. Extract and concatenate x, y, z coordinates
    pos_x = extract_and_slice('#x [mm]')
    pos_y = extract_and_slice('y [mm]')
    pos_z = extract_and_slice('z [mm]')
    
    pos_surface_current = torch.cat((pos_x, pos_y, pos_z), dim=-1)

    return surface_current, pos_surface_current



def knn_interpolate_features_per_component(source_coords, source_component_ids, target_coords, target_component_ids, source_features, k=5):
    """
    Same as knn_interpolate_features, but restricts each target point's neighbor
    search to source points that share its component id (e.g. the same physically
    connected antenna element), preventing interpolation from smearing current
    values across disjoint metal parts (e.g. Yagi director/reflector/driven
    elements) that happen to be geometrically close but not galvanically connected.

    Args:
        source_coords (torch.Tensor): (N_source, D) source point coordinates.
        source_component_ids (torch.Tensor): (N_source,) integer component id per source point.
        target_coords (torch.Tensor): (N_target, D) target point coordinates.
        target_component_ids (torch.Tensor): (N_target,) integer component id per target point.
        source_features (torch.Tensor): (N_source, F) features at source points.
        k (int): number of nearest neighbors to consider (per component).

    Returns:
        torch.Tensor: (N_target, F) interpolated features, restricted per-component.
    """
    out = torch.zeros((target_coords.shape[0], source_features.shape[-1]),
                       dtype=source_features.dtype, device=target_coords.device)

    for comp in torch.unique(target_component_ids):
        t_mask = target_component_ids == comp
        s_mask = source_component_ids == comp
        if s_mask.sum() == 0:
            # no source points share this component; fall back to the global (unrestricted) knn
            k_eff = min(k, source_coords.shape[0])
            out[t_mask] = knn_interpolate_features(source_coords, target_coords[t_mask], source_features, k=k_eff)
            continue
        k_eff = min(k, int(s_mask.sum().item()))
        out[t_mask] = knn_interpolate_features(source_coords[s_mask], target_coords[t_mask], source_features[s_mask], k=k_eff)

    return out


def plot_surface_current_vectors(surface_current, pos_surface_current):
    import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D  # Not needed, just importing mpl_toolkits.mplot3d enables 3d

    fig = plt.figure(figsize=(12, 5))
    
    # Plot Real Part
    ax1 = fig.add_subplot(121, projection='3d')
    KxRe = surface_current[:, 0]
    KyRe = surface_current[:, 2]
    KzRe = surface_current[:, 4]
    magnitude_re = (KxRe**2 + KyRe**2 + KzRe**2).sqrt()
    norm_re = plt.Normalize(magnitude_re.min().item(), magnitude_re.max().item())
    colors_re = plt.cm.viridis(norm_re(magnitude_re.detach().cpu().numpy()))
    ax1.quiver(pos_surface_current[:, 0], pos_surface_current[:, 1], pos_surface_current[:, 2],
              KxRe, KyRe, KzRe, length=0.1, normalize=True, color=colors_re)
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title('Surface Current Vectors (Real Part)')

    # Plot Imaginary Part
    ax2 = fig.add_subplot(122, projection='3d')
    KxIm = surface_current[:, 1]
    KyIm = surface_current[:, 3]
    KzIm = surface_current[:, 5]
    magnitude_im = (KxIm**2 + KyIm**2 + KzIm**2).sqrt()
    norm_im = plt.Normalize(magnitude_im.min().item(), magnitude_im.max().item())
    colors_im = plt.cm.plasma(norm_im(magnitude_im.detach().cpu().numpy()))
    ax2.quiver(pos_surface_current[:, 0], pos_surface_current[:, 1], pos_surface_current[:, 2],
              KxIm, KyIm, KzIm, length=0.1, normalize=True, color=colors_im)
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_zlabel('Z (mm)')
    ax2.set_title('Surface Current Vectors (Imaginary Part)')

    plt.tight_layout()
    plt.show()  


def plot_surface_current_magnitude(surface_current, pos_surface_current):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(12, 5))
    
    # Plot Real Part Magnitude
    ax1 = fig.add_subplot(121, projection='3d')
    KxRe = surface_current[:, 0]
    KyRe = surface_current[:, 2]
    KzRe = surface_current[:, 4]
    magnitude_re = (KxRe**2 + KyRe**2 + KzRe**2).sqrt()
    sc1 = ax1.scatter(pos_surface_current[:, 0], pos_surface_current[:, 1], pos_surface_current[:, 2],
                      c=magnitude_re.detach().cpu().numpy(), cmap='viridis')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title('Surface Current Magnitude (Real Part)')
    fig.colorbar(sc1, ax=ax1, shrink=0.5)

    # Plot Imaginary Part Magnitude
    ax2 = fig.add_subplot(122, projection='3d')
    KxIm = surface_current[:, 1]
    KyIm = surface_current[:, 3]
    KzIm = surface_current[:, 5]
    magnitude_im = (KxIm**2 + KyIm**2 + KzIm**2).sqrt()
    sc2 = ax2.scatter(pos_surface_current[:, 0], pos_surface_current[:, 1], pos_surface_current[:, 2],
                      c=magnitude_im.detach().cpu().numpy(), cmap='plasma')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_zlabel('Z (mm)')
    ax2.set_title('Surface Current Magnitude (Imaginary Part)')
    fig.colorbar(sc2, ax=ax2, shrink=0.5)

    plt.tight_layout()
    plt.show()

def plot_surface_current_euclidian_norm(surface_current, pos_surface_current):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    KxRe = surface_current[:, 0]
    KyRe = surface_current[:, 2]
    KzRe = surface_current[:, 4]
    KxIm = surface_current[:, 1]
    KyIm = surface_current[:, 3]
    KzIm = surface_current[:, 5]

    magnitude = torch.sqrt(KxRe**2 + KyRe**2 + KzRe**2 + KxIm**2 + KyIm**2 + KzIm**2)
    
    sc = ax.scatter(pos_surface_current[:, 0], pos_surface_current[:, 1], pos_surface_current[:, 2],
                    c=magnitude.detach().cpu().numpy(), cmap='viridis')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('Surface Current Euclidean Norm')
    #set colorbar limits to the same for all plots for better comparison if needed
    sc.set_clim(0, 40) # example limits, adjust based on your data
    fig.colorbar(sc, ax=ax, shrink=0.5)

    plt.tight_layout()
    plt.show()
    # save the figure if needed
    fig.savefig('surface_current_euclidean_norm.png', dpi=300)

    



if __name__ == "__main__":
    # Example usage
    source_coords = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], requires_grad=True) # (3, 2)
    target_coords = torch.tensor([[0.5, 0.5], [1.5, 1.5]], requires_grad=True) # (2, 2)
    source_features = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], requires_grad=True) # (3, 2)

    interpolated_features = knn_interpolate_features(source_coords, target_coords, source_features, k=2)
    print("Interpolated Features at Target Points:\n", interpolated_features)

    # Check differentiability by performing a backward pass
    loss = interpolated_features.sum()
    loss.backward()
    print("Gradients w.r.t Source Features:\n", source_features.grad)