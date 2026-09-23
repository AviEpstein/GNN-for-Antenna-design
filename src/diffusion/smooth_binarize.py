import torch

def projection_schedule_beta(t: int, N: int, beta_max: float = 1000.0) -> float:
    """Beta(t) = exp((t/N) * ln(beta_max))."""
    return float(beta_max) ** (float(t) / float(N))

def smooth_binarize(x: torch.Tensor, t: int, N: int, beta_max: float = 1000.0, clamp: bool = False, max_value_offset=0.1) -> torch.Tensor:
    """
    Elementwise projection:
        P(x; t) = [tanh(0.5*beta) + tanh((x-0.5)*beta)] / [2*tanh(0.5*beta)]
    Args:
        x: tensor of any shape, typically in [0,1]
        t: current optimization step (0..N)
        N: total number of steps
        beta_max: final sharpness (default 1000)
        clamp: clip result to [0,1] to guard tiny numeric drift
        max_value: maximum value to clamp the maximum element to (default 1.05)
    Returns:
        Tensor with same shape as x.
    """
    beta = projection_schedule_beta(t, N, beta_max)
    half_beta = 0.5 * beta
    num = torch.tanh(torch.tensor(half_beta, dtype=x.dtype, device=x.device)) + torch.tanh((x - 0.5) * beta)
    den = 2.0 * torch.tanh(torch.tensor(half_beta, dtype=x.dtype, device=x.device))
    y = num / den
    if clamp:
        y = torch.clamp(y, 0.0, 1.0)

    # we add this to ensure that the maximum value remains at the same position after smooth binarization
    if x.dim() == 2:
        max_value = torch.max(y) + max_value_offset
        row, col =  find_argmax_2d(x.unsqueeze(0))
        y[row, col] = max_value
    elif x.dim() == 4:
        # Find the maximum value per element in the batch
        max_values, _ = torch.max(y.view(y.size(0), -1), dim=1)
        max_values += max_value_offset
        row, col =  find_argmax_2d(x.squeeze(1))
        y[torch.arange(x.size(0)), 0, row, col] = max_values
    
    # if y has no value over threshold 0.5, then set the center pixel to 1.0
    threshold = 0.5
    if x.dim() == 2:
        if torch.max(y) < threshold:
            center_row = x.size(0) // 2
            center_col = x.size(1) // 2
            y[center_row, center_col] = 1.0
    elif x.dim() == 4:
        for i in range(x.size(0)):
            if torch.max(y[i]) < threshold:
                center_row = x.size(2) // 2
                center_col = x.size(3) // 2
                y[i, 0, center_row, center_col] = 1.0
    return y

def find_argmax_2d(x: torch.Tensor) -> tuple:
    """
    Find the indices of the maximum value in a batch of 2D tensors.
    Args:
        x: batch of 2D tensors with shape (batch_size, H, W)
    Returns:
        Tuple of two tensors:
            - row_indices: Tensor of shape (batch_size,) containing row indices of the maximum values.
            - col_indices: Tensor of shape (batch_size,) containing column indices of the maximum values.
    """
    batch_size, height, width = x.size()
    argmax = torch.argmax(x.view(batch_size, -1), dim=1)  # Flatten H x W to a single dimension
    row_indices = argmax // width
    col_indices = argmax % width
    return row_indices, col_indices

if __name__ == "__main__":
    # simple test
    x1 = torch.rand((2, 5))
    argmax = torch.argmax(x1)
    row = argmax // x1.size(1)
    col = argmax % x1.size(1)
    print(f"x={x1.numpy()}, argmax_x={row.numpy()}, argmax_y={col.numpy()}")
    for t in range(0, 100):
        print('---- t=', t, ':\n')
        x = smooth_binarize(x1, t, 300)
        argmax = torch.argmax(x)
        row = argmax // x.size(1)
        col = argmax % x.size(1)
        print(f"x={x.numpy()}, argmax_x={row.numpy()}, argmax_y={col.numpy()}")
