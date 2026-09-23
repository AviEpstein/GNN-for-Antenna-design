import torch
import torch.nn.functional as F


def scale_to_log(image):
    """
    Convert antenna far-field absolute value image to 10 dB.

    Parameters:
        image (torch.Tensor): Input image tensor containing absolute values.

    Returns:
        torch.Tensor: Output image tensor in 10 dB.
    """
    # Ensure the input is a torch tensor
    if not isinstance(image, torch.Tensor):
        raise ValueError("Input must be a torch tensor")

    # Ensure the input tensor has no zero values to avoid log(0)
    epsilon = 1e-5
    image = torch.clamp(image, min=epsilon)

    # Convert to dB
    log_image = 10*torch.log10(image)

    return log_image


def scale_to_20_log(image):
    """
    Convert antenna s prams absolute value image to dB.

    Parameters:
        image (torch.Tensor): Input image tensor containing absolute values.

    Returns:
        torch.Tensor: Output image tensor in dB.
    """
    # Ensure the input is a torch tensor
    if not isinstance(image, torch.Tensor):
        raise ValueError("Input must be a torch tensor")

    # Ensure the input tensor has no zero values to avoid log(0)
    epsilon = 1e-1
    image = torch.clamp(image, min=epsilon)

    # Convert to dB
    log_image = 20*torch.log10(image)

    return log_image


def unscale_from_log(log_image):
    """
    Convert dB image back to absolute values.

    Parameters:
        log_image (torch.Tensor): Input image tensor in dB.

    Returns:
        torch.Tensor: Output image tensor with absolute values.
    """
    # Ensure the input is a torch tensor
    if not isinstance(log_image, torch.Tensor):
        raise ValueError("Input must be a torch tensor")

    # Convert from dB to absolute value
    abs_image = torch.pow(10, log_image/10)

    return abs_image


def interpolate_1d(array: torch.Tensor, target_size: int) -> torch.Tensor:
    """
    Downsample a 1D tensor to a specified size.

    Args:
        array (torch.Tensor): The input 1D tensor to be downsampled.
        target_size (int): The size of the output tensor.

    Returns:
        torch.Tensor: The downsampled 1D tensor.
    """
    # Ensure the input tensor is 1D
    if array.dim() != 1:
        raise ValueError("Input tensor must be 1D")

    # Add batch and channel dimensions
    array = array.unsqueeze(0).unsqueeze(0)

    # Use interpolation to downsample
    downsampled_array = F.interpolate(array, size=target_size, mode='linear', align_corners=False)

    # Remove unnecessary dimensions
    downsampled_array = downsampled_array.squeeze()

    return downsampled_array
