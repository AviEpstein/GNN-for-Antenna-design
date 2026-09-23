"""Dataset statistics helpers (copied from stats/data_stats.py in the research repo)."""


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
