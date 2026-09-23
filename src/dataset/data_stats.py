import torch

from src.dataset.dataloader_utils import example_paramters_to_vector


def calculate_farfeild_statistics(data_loader):
    """
    Calculate the average, standard deviation, minimum, and maximum of tensors in a PyTorch DataLoader.

    Args:
    data_loader (torch.utils.data.DataLoader): DataLoader containing the dataset.

    Returns:
    dict: Dictionary containing the mean, std, min, and max of the dataset.
    """
    mean_sum = 0
    square_sum = 0
    min_val = float('inf')
    max_val = float('-inf')
    n_samples = 0

    for data in data_loader:
        # Assuming the data is a tensor
        if isinstance(data, (list, tuple)):
            data = data[2]  # if data is in tuple format, e.g., (data, target)

        # Update sums and counts
        mean_sum += data.sum()
        square_sum += (data ** 2).sum()
        min_val = min(min_val, data.min().item())
        max_val = max(max_val, data.max().item())
        n_samples += data.numel()

    # Calculate mean and std
    mean = mean_sum / n_samples
    std = (square_sum / n_samples - mean ** 2).sqrt()

    return {
        'mean': mean.item(),
        'std': std.item(),
        'min': min_val,
        'max': max_val
    }


def get_graph_stats(data_loader):

    max_accumulations = 10**6
    eps = torch.tensor(1e-8)
    num_accs_x_graph = 0
    num_accs_x_antenna = 0
    mean_vec_x_graph = 0
    std_vec_x_graph = 0
    for idx, (graph, antenna, ff_image, _, example_paramters, raw_idx) in enumerate(data_loader):
        pos, node_normals, edge_index, node_type = graph.pos, graph.node_normals, graph.edge_index, graph.node_type
        x_graph = torch.cat([pos[:], node_type[:]], dim=1).to(torch.float32)
        mean_vec_x_graph += torch.sum(x_graph[:], dim=0)
        std_vec_x_graph += torch.sum(x_graph[:]**2, dim=0)
        num_accs_x_graph += x_graph[:].shape[0]

        if(num_accs_x_graph > max_accumulations or num_accs_x_antenna > max_accumulations):
            break

    mean_vec_x_graph = mean_vec_x_graph/num_accs_x_graph
    std_vec_x_graph = torch.maximum(torch.sqrt(std_vec_x_graph/num_accs_x_graph - mean_vec_x_graph**2), eps)

    stats = {'mean_vec_x_graph': mean_vec_x_graph,
             'std_vec_x_graph': std_vec_x_graph}
    return stats


def get_parameter_stats(data_loader):
    """
    Calculate element-wise mean, standard deviation, minimum, and maximum for vectors in data_loader.

    Args:
        data_loader (iterable): DataLoader yielding individual data points.

    Returns:
        dict: Dictionary containing element-wise 'mean', 'std', 'min', and 'max'.
    """
    sum_vector = None
    sum_of_squares_vector = None
    min_vector = None
    max_vector = None
    total_count = 0
    print('calculating parameter stats')

    for graph, antenna, ff_image, _, example_parameters, raw_idx in data_loader:
        # Convert example parameters to a vector
        params_vector = example_paramters_to_vector(
            example_parameters['ant_parameters'],
            example_parameters['env_parameters']
        )

        if sum_vector is None:
            # Initialize the stats with the shape of params_vector
            sum_vector = params_vector.clone()
            sum_of_squares_vector = params_vector.clone() ** 2
            min_vector = params_vector.clone()
            max_vector = params_vector.clone()
        else:
            # Update element-wise statistics
            sum_vector += params_vector
            sum_of_squares_vector += params_vector ** 2
            min_vector = torch.min(min_vector, params_vector)
            max_vector = torch.max(max_vector, params_vector)

        total_count += 1

    # Compute element-wise mean and standard deviation
    mean_vector = sum_vector / total_count
    std_vector = ((sum_of_squares_vector / total_count) - mean_vector ** 2).sqrt()
    return {
        'mean': mean_vector,
        'std': std_vector,
        'min': min_vector,
        'max': max_vector,
    }


def standerize(vector_to_normalize, mean_vec, std_vec, epsilon=1e-8):
    # Add epsilon only where std_vec is zero
    std_vec = torch.where(std_vec == 0, epsilon, std_vec)
    return (vector_to_normalize-mean_vec)/std_vec


def unstanerize(vector_to_normalize, mean_vec, std_vec):
    return vector_to_normalize*std_vec+mean_vec
