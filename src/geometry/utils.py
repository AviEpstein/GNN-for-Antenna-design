import torch
import torch_geometric.utils as pyg_utils

def faces_to_edge_index(faces):
        """
        Convert a (num_faces x 3) tensor to an edge_index tensor.
        Each face (i, j, k) produces edges (i, j), (j, k), (k, i).
        
        Args:
            faces (torch.Tensor): Tensor of shape (num_faces, 3).
            
        Returns:
            edge_index (torch.Tensor): Tensor of shape (2, num_edges).
        """
        edges = []
        for face in faces:
            i, j, k = face
            edges.append([i.item(), j.item()])
            edges.append([j.item(), k.item()])
            edges.append([k.item(), i.item()])
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_index = pyg_utils.to_undirected(edge_index)
        return edge_index



import torch
import pickle
import numpy as np
import torch.nn.functional as F

def get_s11_single_freq(s11_dict,frequency):
    """
    Get the S11 value for a specific frequency from the S11 dictionary.
    
    Parameters:
        s11_dict (dict): A dictionary containing 'frequancy' and 's11_abs_db' tensors.
        frequency (float): The frequency for which to retrieve the S11 value.

    Returns:
        torch.Tensor: The S11 value at the specified frequency.
    """
    freq = s11_dict['frequancy']
    s11_abs_db = s11_dict['s11_abs_db']
    s11_abs = s11_dict['s11_abs']
    s11_phase = s11_dict['phase']
    
    # Find the index of the closest frequency
    idx = (torch.abs(freq - frequency)).argmin()
    s11_abs_db_value = s11_abs_db[idx]
    s11_abs = s11_abs[idx]
    s11_phase = s11_phase[idx]
    s11_complex = to_complex(s11_abs, s11_phase)
    s11_single_freq ={'s11_abs_db': s11_abs_db_value, 's11_abs': s11_abs, 's11_phase': s11_phase, 's11_complex': s11_complex}

    return s11_single_freq

def to_complex(abs, phase):
    """
    Convert absolute value and phase to complex tensor.
    
    Parameters:
        abs (torch.Tensor): Absolute value tensor.
        phase (torch.Tensor): Phase tensor in radians.
    
    Returns:
        torch.Tensor: B,2 real and imaginary parts.
    """
    real = abs * torch.cos(phase)
    imaginary = abs * torch.sin(phase)
    complex_tensor = torch.tensor([real, imaginary]).unsqueeze(0) #torch.stack([real, imaginary], dim=-1)
    return complex_tensor



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
    log_image =  10*torch.log10(image)
    
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
    epsilon = 1e-1#1e-5
    image = torch.clamp(image, min=epsilon)
    
    # Convert to dB
    log_image =  20*torch.log10(image)
    
    return log_image



def unscale_from_log(log_image):
    """
    Convert dB image back to absolute values.
    
    Parameters:
        db_image (torch.Tensor): Input image tensor in dB.
    
    Returns:
        torch.Tensor: Output image tensor with absolute values.
    """
    # Ensure the input is a torch tensor
    if not isinstance(log_image, torch.Tensor):
        raise ValueError("Input must be a torch tensor")
    
    # Convert from dB to absolute value
    abs_image = torch.pow(10, log_image/10) 
    
    return abs_image


def unstanderize_ff(ff, avg_val=1.004577, std_val=0.46855):
    scale = (avg_val/std_val)
    standerized = ((ff-scale)*std_val + avg_val)
    return standerized

def standerize_ff(ff, avg_val=1.004577, std_val=0.46855):
    scale = (avg_val/std_val)
    standerized = (ff - avg_val)/std_val + scale
    return standerized

class PositionalEncoding:
    def __init__(self, num_encoding_functions=6, include_input=True):
        self.num_encoding_functions = num_encoding_functions
        self.include_input = include_input
        self.encoding_functions = [lambda x: x] if include_input else []
        
        for i in range(num_encoding_functions):
            self.encoding_functions.append(lambda x, i=i: torch.sin((2.0**i) * x))
            self.encoding_functions.append(lambda x, i=i: torch.cos((2.0**i) * x))
    
    def encode(self, x):
        return torch.cat([fn(x) for fn in self.encoding_functions], dim=-1)
    
    
    
    

def read_s11_pickle(file_path):
    """
    Reads a pickle file and returns a complex tensor stored in it.
    
    Parameters:
        file_path (str): The path to the pickle file.
        
    Returns:
        torch.Tensor: The complex tensor from the pickle file.
    """
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        
        
    complex_tensor = torch.tensor(data[0])
    frequancy = torch.tensor(data[1])
    s11_abs = abs(complex_tensor)
    # Convert tensor to dB
    s11_abs_db = 20 * torch.log10(torch.abs(complex_tensor))
    phase = torch.angle(complex_tensor)
    # return {
    #     's11_abs':torch.tensor(s11_abs, dtype=torch.float32) if s11_abs is not torch else s11_abs,
    #     's11_abs_db':torch.tensor(s11_abs_db,dtype=torch.float32),
    #     'phase':torch.tensor(phase, dtype=torch.float32),
    #     'frequancy':torch.tensor(frequancy,dtype=torch.float32)}
    return {
        's11_abs': s11_abs.to(torch.float32) if isinstance(s11_abs, torch.Tensor) else torch.tensor(s11_abs, dtype=torch.float32),
        's11_abs_db':s11_abs_db.to(torch.float32) if isinstance(s11_abs_db, torch.Tensor) else torch.tensor(s11_abs_db, dtype=torch.float32),
        'phase':phase.to(torch.float32) if isinstance(phase, torch.Tensor) else torch.tensor(phase, dtype=torch.float32),
        'frequancy':frequancy.to(torch.float32) if isinstance(frequancy, torch.Tensor) else torch.tensor(frequancy, dtype=torch.float32)}
    

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


def mask_random_patch_and_average(loss_tensor, patch_size=(32, 32)):
    # Get the original shape of the loss tensor (1, H, W, 1)
    _, H, W, _ = loss_tensor.shape

    # Ensure patch_size is smaller than the loss tensor dimensions
    patch_height, patch_width = patch_size
    assert patch_height <= H and patch_width <= W, "Patch size must be smaller than the loss tensor dimensions."

    # Randomly choose the top-left corner for the patch
    top = torch.randint(0, H - patch_height + 1, (1,)).item()
    left = torch.randint(0, W - patch_width + 1, (1,)).item()

    # Extract the patch from the loss tensor
    patch = loss_tensor[:, top:top + patch_height, left:left + patch_width, :]

    # Compute the average loss in the patch
    average_loss = patch.mean()

    return patch, average_loss