import torch
from src.geometry.utils import faces_to_edge_index
from torch_geometric.data import Data
import torch.nn.functional as F
from src.geometry.mesh_functions import get_submesh_type

def spherical_to_cartesian(r, theta, phi):
    if hasattr(r, 'device') and r.device != theta.device:
        r = r.to(theta.device)
    x = r * torch.cos(phi) * torch.cos(theta)
    y = r * torch.cos(phi) * torch.sin(theta)
    z = r * torch.sin(phi)
    if z.dim()==1:
         xyz = torch.cat([x, y, z],dim=0)
    else:
        xyz = torch.stack([x, y, z])
    return xyz


def create_box(width=1.0, height=1.0, depth=1.0, device='cpu'):
    """
    Creates a box mesh centered at origin with the given dimensions.
    
    Args:
        width (float): Width of the box (x dimension)
        height (float): Height of the box (y dimension)
        depth (float): Depth of the box (z dimension)
        device (str): Device to create tensors on ('cpu' or 'cuda')
        
    Returns:
        Data: PyTorch Geometric Data object containing the box mesh
    """
    # Define the vertices (8 corners of the box)
    half_width = width / 2
    half_height = height / 2
    half_depth = depth / 2
    
    vertices = torch.tensor([
        [-half_width, -half_height, -half_depth],  # 0
        [-half_width, -half_height,  half_depth],  # 1
        [-half_width,  half_height, -half_depth],  # 2
        [-half_width,  half_height,  half_depth],  # 3
        [ half_width, -half_height, -half_depth],  # 4
        [ half_width, -half_height,  half_depth],  # 5
        [ half_width,  half_height, -half_depth],  # 6
        [ half_width,  half_height,  half_depth],  # 7
    ], dtype=torch.float32, device=device)

    # Define the faces to match trimesh box faces
    faces = torch.tensor([
        [1, 3, 0],  # Left face
        [4, 1, 0],  # Bottom face
        [0, 3, 2],  # Front face
        [2, 4, 0],  # Top face
        [1, 7, 3],  # Back face
        [5, 1, 4],  # Right face
        [5, 7, 1],
        [3, 7, 2],
        [6, 4, 2],
        [2, 7, 6],
        [6, 5, 4],
        [7, 5, 6]
    ], dtype=torch.long, device=device)

    # Compute edge indices from faces
    edge_index = faces_to_edge_index(faces).to(device)

    # Calculate vertex normals (normalized vectors from center to vertices)
    node_normals = F.normalize(vertices, dim=1).to(device)

    # Create PyTorch Geometric Data object
    data = Data(
        pos=vertices,
        faces=faces,
        edge_index=edge_index,
        node_normals=node_normals
    )

    return data





def look_at_rotation(position):
    z = F.normalize(-position, dim=0)  # pointing toward origin
    up = torch.tensor([0.0, 0.0, 1.0], device=position.device)
    if torch.allclose(z, up):  # Avoid degenerate cross product
        up = torch.tensor([0.0, 1.0, 0.0], device=position.device)
    if up.dim()==1 and z.dim()==2:
        z = z.squeeze(1)
    x = F.normalize(torch.cross(up, z), dim=0)
    y = torch.cross(z, x)
    rot = torch.stack([x, y, z], dim=1)  # 3x3 rotation matrix

    # Ensure orthonormality using QR decomposition
    q, _ = torch.linalg.qr(rot)
    return q


def transform_vertcies(vertices, center, rotation, scale=1.0):
    return vertices * scale @ rotation.T + center


def rotate_normals(normals, rotation):
    """
    Rotates the node normals using the given rotation matrix.
    
    Args:
        normals (torch.Tensor): Normals to rotate (N x 3).
        rotation (torch.Tensor): Rotation matrix (3 x 3).
        
    Returns:
        torch.Tensor: Rotated normals (N x 3).
    """
    return normals @ rotation.T


def create_reflector_on_sphere(radius, theta, phi, box_size, device='cpu'):
    center = spherical_to_cartesian(radius, theta, phi)
    rotation = look_at_rotation(center)
    box = create_box(width=box_size, height=box_size, depth=1, device=device)
    box.pos = transform_vertcies(box.pos, center, rotation, scale=1)
    box.node_normals = rotate_normals(box.node_normals, rotation)
    box.edge_index = box.edge_index
    return box

def create_reflectors_on_sphere(reflectors_dict, device='cpu'):
    """
    Create multiple reflectors on a sphere based on the provided dictionary.
    
    Args:
        reflectors_dict (dict): Dictionary containing reflector parameters.
        
    Returns:
        list: List of reflector meshes.
    """
    thetas = reflectors_dict['thetas']
    phis = reflectors_dict['phis']
    radius = reflectors_dict['radius']  
    box_size = reflectors_dict['box_size']
    reflectors_list = []
    
    for i in range(reflectors_dict['num_of_reflectors']):
        theta = thetas[i]
        phi = phis[i] 
        reflector = create_reflector_on_sphere(radius, theta=theta, phi=phi, box_size=box_size, device=device)
        one_hot_type = get_submesh_type('PEC_Reflector')
        node_type = torch.tensor(len(reflector.pos) * [one_hot_type], dtype=torch.float32, device=device)
        node_probs = torch.ones((len(reflector.pos), 1), dtype=torch.float32, device=device)
        reflector.node_type = node_type
        reflector.node_probs = node_probs
        reflectors_list.append(reflector)
    
    return reflectors_list
