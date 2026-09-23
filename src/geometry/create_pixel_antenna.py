import torch_geometric.utils as pyg_utils
import torch
from torch_geometric.data import Data
from src.geometry.mesh_functions import decompose_pyg_graph, get_submesh_type
from src.geometry.mesh_functions_pytorch_2 import merge_and_connect_graphs_from_dict, merge_graphs_from_dict
from src.geometry.create_reflector_on_sphere import create_reflectors_on_sphere

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
    
def compute_node_normals(points, faces):
    """
    Compute normals at each node given 3D points and faces.

    Args:
        points (torch.Tensor): (N, 3) tensor of 3D points.
        faces (torch.Tensor): (M, 3) tensor of face indices.

    Returns:
        torch.Tensor: (N, 3) tensor of node normals.
    """
    # Step 1: Compute face normals
    v0 = points[faces[:, 0]]
    v1 = points[faces[:, 1]]
    v2 = points[faces[:, 2]]
    
    # Edge vectors
    edge1 = v1 - v0
    edge2 = v2 - v0
    
    # Compute face normals
    face_normals = torch.cross(edge1, edge2, dim=1)
    face_normals = torch.nn.functional.normalize(face_normals, dim=1)

    # Step 2: Aggregate face normals to nodes
    node_normals = torch.zeros_like(points, device=face_normals.device)
    for i in range(3):
        node_normals.index_add_(0, faces[:, i].to(face_normals.device), face_normals)
    
    # Normalize node normals
    node_normals = torch.nn.functional.normalize(node_normals, dim=1)
    return node_normals



class HardThreshold(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold=0.5):
        return (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-through estimator: just pass gradients as-is
        return grad_output, None
    
def soft_step(x, threshold=0.5, epsilon=0.3):
    return torch.clamp((x - threshold + epsilon) / (2 * epsilon), 0, 1)

def differentiable_threshold(x, threshold=0.5, sharpness=50):
    return torch.sigmoid((x - threshold) * sharpness)

def gumbel_sigmoid(logits, tau=1.0):
    U = torch.rand_like(logits)
    g = -torch.log(-torch.log(U + 1e-9) + 1e-9)
    return torch.sigmoid((logits + g) / tau)

def create_pixel_mesh(matrix, threshold=0.5):
    """
    Creates a mesh from a matrix of logits while preserving gradients.
    pixel probs should be between 0 and 1
    """
    grid_size = matrix.shape[0]
    pixel_probs = matrix # assume pixel_probs are already between 0 and 1
    
    # pixel_probs = torch.sigmoid(matrix)
    # Compute pixel probabilities using sigmoid, preserving gradients
    # pixel_probs = torch.sigmoid(0.5*(matrix-threshold))
    
    # pixel_probs = differentiable_threshold(matrix, threshold=threshold)
    # Apply smooth threshold
    
    # pixel_probs = torch.sigmoid((matrix - threshold) * 10)  # Sharper sigmoid for better binary approximation
    # pixel_probs = gumbel_sigmoid(matrix, tau=0.01)
    # Create a mask for active pixels
    pixel_mask = pixel_probs >= threshold
    # pixel_scale = HardThreshold.apply(pixel_probs, threshold)
    # pixel_probs = differentiable_threshold(matrix, threshold=threshold)
    vertices = []
    faces = []
    colors = []
    vertex_count = 0

    # Move template tensors to same device as input
    pixel_vertices_template = torch.tensor([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0]
    ], dtype=torch.float32, device=matrix.device)

    pixel_faces_template = torch.tensor([
        [0, 1, 2],
        [0, 2, 3]
    ], dtype=torch.long, device=matrix.device)

    # Iterate over grid positions
    for i in range(grid_size):
        for j in range(grid_size):
            if pixel_mask[i, j]:
                # Preserve gradients by multiplying vertices by probability
                shift = torch.tensor([i, j, 0], dtype=torch.float32, device=matrix.device)
                # Scale vertices by probability to maintain gradients
                pixel_vertices = pixel_vertices_template + shift
                # pixel_vertices = pixel_vertices * pixel_probs[i, j]#.unsqueeze(0)  # Scale by probability
                vertices.append(pixel_vertices)

                pixel_faces = pixel_faces_template + vertex_count
                faces.append(pixel_faces)

                pixel_color = torch.full((4, 3), 255.0, dtype=torch.float32, device=matrix.device)
                colors.append(pixel_color)

                vertex_count += 4

    if vertices:
        vertices = torch.cat(vertices, dim=0)
        faces = torch.cat(faces, dim=0)
        colors = torch.cat(colors, dim=0)
    else:
        vertices = torch.empty((0, 3), dtype=torch.float32, device=matrix.device)
        faces = torch.empty((0, 3), dtype=torch.long, device=matrix.device)
        colors = torch.empty((0, 3), dtype=torch.float32, device=matrix.device)

    # Create edges and compute normals
    edge_index = faces_to_edge_index(faces).to(matrix.device)

    node_normals = compute_node_normals(vertices, faces)
    
    one_hot_type = get_submesh_type('Antenna_PEC_STEP')
    node_type = torch.full((len(vertices),), one_hot_type, dtype=torch.float32, device=matrix.device) if isinstance(one_hot_type, (int, float)) else torch.tensor([one_hot_type] * len(vertices), dtype=torch.float32, device=matrix.device)
    
    data = Data(
        x=colors, 
        pos=vertices, 
        faces=faces, 
        edge_index=edge_index, 
        node_type=node_type, 
        node_normals=node_normals
    )
    return data, pixel_probs


def create_pixel_mesh_with_pixel_probabilty(matrix, threshold=0.5):
    """
    Creates a mesh as a a fl squared pixel and each pixel has its confidance the matrix 
    """
    grid_size = matrix.shape[0]

    vertices = []
    faces = []
    colors = []
    pixel_probs = []
    vertex_count = 0

    # Move template tensors to same device as input
    pixel_vertices_template = torch.tensor([
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0]
    ], dtype=torch.float32, device=matrix.device)

    pixel_faces_template = torch.tensor([
        [0, 1, 2],
        [0, 2, 3]
    ], dtype=torch.long, device=matrix.device)

    # Iterate over grid positions
    for i in range(grid_size):
        for j in range(grid_size):
                # Preserve gradients by multiplying vertices by probability
                shift = torch.tensor([i, j, 0], dtype=torch.float32, device=matrix.device)
                # Scale vertices by probability to maintain gradients
                pixel_vertices = pixel_vertices_template + shift
                # pixel_vertices = pixel_vertices * pixel_probs[i, j]#.unsqueeze(0)  # Scale by probability
                vertices.append(pixel_vertices)

                pixel_faces = pixel_faces_template + vertex_count
                faces.append(pixel_faces)
                pixel_prob = matrix[i, j].unsqueeze(0).repeat(4, 1)
                pixel_probs.append(pixel_prob)
                pixel_color = torch.full((4, 3), 255, dtype=torch.float32, device=matrix.device)
                colors.append(pixel_color)

                vertex_count += 4

    if vertices:
        vertices = torch.cat(vertices, dim=0)
        faces = torch.cat(faces, dim=0)
        colors = torch.cat(colors, dim=0)
        pixel_probs = torch.cat(pixel_probs, dim=0)
    else:
        vertices = torch.empty((0, 3), dtype=torch.float32, device=matrix.device)
        faces = torch.empty((0, 3), dtype=torch.long, device=matrix.device)
        colors = torch.empty((0, 3), dtype=torch.float32, device=matrix.device)

    # Create edges and compute normals
    edge_index = faces_to_edge_index(faces).to(matrix.device)

    node_normals = compute_node_normals(vertices, faces)
    
    one_hot_type = get_submesh_type('Antenna_PEC_STEP')
    node_type = torch.full((len(vertices),), one_hot_type, dtype=torch.float32, device=matrix.device) if isinstance(one_hot_type, (int, float)) else torch.tensor([one_hot_type] * len(vertices), dtype=torch.float32, device=matrix.device)
    
    data = Data(
        x=colors, 
        pos=vertices, 
        faces=faces, 
        edge_index=edge_index, 
        node_type=node_type, 
        node_normals=node_normals,
        node_probs=pixel_probs
    )
    return data, matrix

def create_feed_PEC(x,y,height = 4):
    # Desired location for the center of the box


    # Define the vertices for a cube (box) centered at the origin.
    # The cube has side length 1, so vertices range from -0.5 to 0.5.
    verts = torch.tensor([
        [0.0, 0.0, 0.0],
        [ 1.0, 0.0, 0.0],
        [ 1.0,  1.0, 0.0],
        [0.0,  1.0, 0.0],
        [0.0, 0.0,  -height+1],
        [ 1.0, 0.0,  -height+1],
        [ 1.0,  1.0,  -height+1],
        [0.0,  1.0,  -height+1]
    ], dtype=torch.float, device=x.device)

    # Define the faces of the cube.
    # Each face of the cube is represented as two triangles (thus 12 triangles in total).
    faces = torch.tensor([
        # Top face (z = -height+1)
        [0, 1, 2], [0, 2, 3],
        # Bottom face (z = -height)
        [4, 6, 5], [4, 7, 6],
        # Front face (y = 0)
        [0, 5, 1], [0, 4, 5],
        # Back face (y = 1)
        [3, 2, 6], [3, 6, 7],
        # Left face (x = 0)
        [0, 3, 7], [0, 7, 4],
        # Right face (x = 1)
        [1, 5, 6], [1, 6, 2]
    ], dtype=torch.long, device=x.device)
    node_normals = torch.tensor([[-0.57735027, -0.57735027,  0.57735027],
       [ 0.57735027, -0.57735027,  0.57735027],
       [ 0.57735027,  0.57735027,  0.57735027],
       [-0.57735027,  0.57735027,  0.57735027],
       [-0.57735027, -0.57735027, -0.57735027],
       [ 0.57735027, -0.57735027, -0.57735027],
       [ 0.57735027,  0.57735027, -0.57735027],
       [-0.57735027,  0.57735027, -0.57735027]], device=x.device)
    
    # Create a translation vector to move the cube so that its center is at (x, y, 0)
    translation = torch.tensor([x, y, 0.0],device=x.device)
    vertices = verts + translation
    edge_index = faces_to_edge_index(faces).to(x.device)
    
    one_hot_type = get_submesh_type('Antenna_Feed_PEC_STEP')
    node_type = torch.tensor(len(vertices) * [one_hot_type], dtype=torch.float32, device=x.device)
    node_probs = torch.ones((vertices.shape[0], 1), dtype=torch.float32, device=x.device)
    data = Data(pos=vertices, faces=faces, node_normals=node_normals, edge_index = edge_index, node_type=node_type, node_probs=node_probs)
    return data

def create_feed(x, y, height=4):
    # Define the vertices for a box (cube) that has a thickness of 1 in z.
    # Here we define the top face at z = -height+1 and the bottom face at z = -height.
    verts = torch.tensor([
        [0.0, 0.0, -height+1],  # vertex 0: top face
        [1.0, 0.0, -height+1],  # vertex 1
        [1.0, 1.0, -height+1],  # vertex 2
        [0.0, 1.0, -height+1],  # vertex 3
        [0.0, 0.0, -height],    # vertex 4: bottom face
        [1.0, 0.0, -height],    # vertex 5
        [1.0, 1.0, -height],    # vertex 6
        [0.0, 1.0, -height]     # vertex 7
    ], dtype=torch.float, device=x.device)

    # Define the faces (as triangles) with a consistent winding order.
    # We create two triangles for each of the six faces.
    faces = torch.tensor([
        # Top face (z = -height+1)
        [0, 1, 2], [0, 2, 3],
        # Bottom face (z = -height)
        [4, 6, 5], [4, 7, 6],
        # Front face (y = 0)
        [0, 5, 1], [0, 4, 5],
        # Back face (y = 1)
        [3, 2, 6], [3, 6, 7],
        # Left face (x = 0)
        [0, 3, 7], [0, 7, 4],
        # Right face (x = 1)
        [1, 5, 6], [1, 6, 2]
    ], dtype=torch.long, device=x.device)
    
    node_normals = torch.tensor([[-0.57735027, -0.57735027,  0.57735027],
       [ 0.57735027, -0.57735027,  0.57735027],
       [ 0.57735027,  0.57735027,  0.57735027],
       [-0.57735027,  0.57735027,  0.57735027],
       [-0.57735027, -0.57735027, -0.57735027],
       [ 0.57735027, -0.57735027, -0.57735027],
       [ 0.57735027,  0.57735027, -0.57735027],
       [-0.57735027,  0.57735027, -0.57735027]], device=x.device)

    # Translate the vertices so that the box is centered at (x, y, 0)
    translation = torch.tensor([x, y, 0.0], device=x.device)
    vertices = verts + translation
    node_normals = node_normals
    edge_index = faces_to_edge_index(faces).to(x.device)
    
    one_hot_type = get_submesh_type('Antenna_Feed_STEP')
    node_type = torch.tensor(len(vertices) * [one_hot_type], dtype=torch.float32, device=x.device)
    node_probs = torch.ones((vertices.shape[0], 1), dtype=torch.float32, device=x.device)
    
    data = Data(pos=vertices, faces=faces, edge_index = edge_index , node_normals=node_normals, node_type=node_type, node_probs=node_probs)
    return data

def create_ground(device='cpu'):
    # Define the vertices of the square (4 vertices in 2D)
    vertices = torch.tensor([
        [0.0, 0.0, 0.0],  # Vertex 0
        [1.0, 0.0, 0.0],  # Vertex 1
        [1.0, 1.0, 0.0],  # Vertex 2
        [0.0, 1.0, 0.0]   # Vertex 3
    ], dtype=torch.float, device=device)

    # Define the faces (triangles) of the square by splitting it into two triangles.
    # Here, the square is split along the diagonal from vertex 0 to vertex 2.
    faces = torch.tensor([
        [0, 1, 2],  # First triangle
        [0, 2, 3]   # Second triangle
    ], dtype=torch.long, device=device)
    
    edge_index = faces_to_edge_index(faces).to(device)
    node_normals = compute_node_normals(vertices, faces).to(device)
    
    one_hot_type = get_submesh_type('PEC_ground')
    node_type = torch.tensor(len(vertices) * [one_hot_type], dtype=torch.float32, device=device)
    node_probs = torch.ones((vertices.shape[0], 1), dtype=torch.float32, device=device)

    data = Data(pos=vertices, faces=faces, edge_index = edge_index, node_normals=node_normals , node_type=node_type, node_probs=node_probs)
    return data



def create_box_with_cube_hole(cube_size=1.0, outerbox_length=1.5, height=0.5, hole_center=(0.0, 0.0)):
    """
    Creates a mesh (vertices and faces) for a box (a thin frame) whose inner cavity is a cube.
    
    Parameters:
      cube_size     : The side length of the inner cube (hole).
      outerbox_length: The side length of the outer frame. The outer box is centered at (0,0).
      hight     : Total hight (z-extent) of the box (should be less than cube_size for a thin frame).
      hole_center   : Tuple (x, y) giving the 2D location at which the inner hole is centered.
                      (The outer box remains centered at (0,0).)
    
    The mesh is built with 16 vertices:
      - Outer vertices (indices 0–7): corners of the outer box, computed from outerbox_length.
      - Inner vertices (indices 8–15): corners of the inner cube (hole), computed relative to hole_center.
    
    Returns:
      vertices: FloatTensor of shape (16, 3)
      faces   : LongTensor of shape (32, 3) with the vertex indices for each triangle.
    """
    s = cube_size      # side length of the inner cube (hole)
    L = outerbox_length  # side length of the outer box (frame)
    h = height      # total thickness (z-extent) of the box
    cx, cy = hole_center  # center of the inner cube (hole)

    # Compute half-lengths:
    half_outer = L / 2.0   # half-length for the outer box (centered at (0,0))
    half_inner = s / 2.0   # half-length for the inner cube (hole)

    # z coordinates (the box is centered in z)
    z_bottom = -h / 2.0
    z_top    =  h / 2.0

    # Define 16 vertices:
    #
    # Outer vertices (indices 0–7) come from the outer box which is centered at (0,0).
    #   Bottom face (z = z_bottom): indices 0-3
    #   Top face    (z = z_top):    indices 4-7
    outer_vertices = [
        [ -half_outer, -half_outer, z_bottom],  # 0: bottom-left
        [  half_outer, -half_outer, z_bottom],  # 1: bottom-right
        [  half_outer,  half_outer, z_bottom],  # 2: top-right
        [ -half_outer,  half_outer, z_bottom],  # 3: top-left
        [ -half_outer, -half_outer, z_top   ],  # 4: bottom-left (top face)
        [  half_outer, -half_outer, z_top   ],  # 5: bottom-right
        [  half_outer,  half_outer, z_top   ],  # 6: top-right
        [ -half_outer,  half_outer, z_top   ],  # 7: top-left
    ]

    # Inner vertices (indices 8–15) come from the inner cube (hole), centered at (cx, cy).
    #   Bottom face (z = z_bottom): indices 8-11
    #   Top face    (z = z_top):    indices 12-15
    inner_vertices = [
        [ cx - half_inner, cy - half_inner, z_bottom],  # 8
        [ cx + half_inner, cy - half_inner, z_bottom],  # 9
        [ cx + half_inner, cy + half_inner, z_bottom],  # 10
        [ cx - half_inner, cy + half_inner, z_bottom],  # 11
        [ cx - half_inner, cy - half_inner, z_top   ],  # 12
        [ cx + half_inner, cy - half_inner, z_top   ],  # 13
        [ cx + half_inner, cy + half_inner, z_top   ],  # 14
        [ cx - half_inner, cy + half_inner, z_top   ],  # 15
    ]

    # Combine the vertices into one tensor.
    vertices = torch.tensor(outer_vertices + inner_vertices, dtype=torch.float32,device=cube_size.device)

    faces = []  # to collect triangle faces

    # Two helper functions to add a quad (as two triangles) with different vertex orderings.
    def add_quad_reversed(v0, v1, v2, v3):
        """
        For the bottom ring: adds a quad defined by vertices (v0, v1, v2, v3) 
        as two triangles with reversed ordering:
            Triangle 1: (v2, v1, v0)
            Triangle 2: (v3, v2, v0)
        """
        faces.append([v2, v1, v0])
        faces.append([v3, v2, v0])
        
    def add_quad_default(v0, v1, v2, v3):
        """
        Adds a quad defined by vertices (v0, v1, v2, v3) as two triangles 
        with default ordering:
            Triangle 1: (v0, v1, v2)
            Triangle 2: (v0, v2, v3)
        """
        faces.append([v0, v1, v2])
        faces.append([v0, v2, v3])
    
    # -------------------------------------------------------------------
    # 1. Bottom ring (z = z_bottom)
    #    Four quads around the inner hole (using reversed ordering)
    add_quad_reversed(0, 1, 9, 8)    # front edge
    add_quad_reversed(1, 2, 10, 9)   # right edge
    add_quad_reversed(2, 3, 11, 10)  # back edge
    add_quad_reversed(3, 0, 8, 11)   # left edge

    # -------------------------------------------------------------------
    # 2. Top ring (z = z_top)
    #    Four quads around the inner hole (using default ordering)
    add_quad_default(4, 5, 13, 12)   # front edge
    add_quad_default(5, 6, 14, 13)   # right edge
    add_quad_default(6, 7, 15, 14)   # back edge
    add_quad_default(7, 4, 12, 15)   # left edge

    # -------------------------------------------------------------------
    # 3. Outer vertical side faces
    #    These connect the outer bottom and outer top boundaries.
    add_quad_default(0, 1, 5, 4)    # front side
    add_quad_default(1, 2, 6, 5)    # right side
    add_quad_default(2, 3, 7, 6)    # back side
    add_quad_default(3, 0, 4, 7)    # left side

    # -------------------------------------------------------------------
    # 4. Inner vertical side faces
    #    These line the inner cavity.
    add_quad_default(12, 13, 9, 8)    # front inner side
    add_quad_default(13, 14, 10, 9)   # right inner side
    add_quad_default(14, 15, 11, 10)  # back inner side
    add_quad_default(15, 12, 8, 11)   # left inner side
    
    node_normals = torch.tensor([[-0.57735027, -0.57735027, -0.57735027],
       [ 0.57735027, -0.57735027, -0.57735027],
       [ 0.57735027,  0.57735027, -0.57735027],
       [-0.57735027,  0.57735027, -0.57735027],
       [-0.57735027, -0.57735027,  0.57735027],
       [ 0.57735027, -0.57735027,  0.57735027],
       [ 0.57735027,  0.57735027,  0.57735027],
       [-0.57735027,  0.57735027,  0.57735027],
       [ 0.30151134,  0.30151134, -0.90453403],
       [-0.30151134,  0.30151134, -0.90453403],
       [-0.30151134, -0.30151134, -0.90453403],
       [ 0.30151134, -0.30151134, -0.90453403],
       [ 0.30151134,  0.30151134,  0.90453403],
       [-0.30151134,  0.30151134,  0.90453403],
       [-0.30151134, -0.30151134,  0.90453403],
       [ 0.30151134, -0.30151134,  0.90453403]], dtype=torch.float32, device=vertices.device)

    faces = torch.tensor(faces, dtype=torch.long, device=vertices.device)
    edge_index = faces_to_edge_index(faces).to(vertices.device)
    
    one_hot_type = get_submesh_type('Env_FR4_STEP')
    node_type = torch.tensor(len(vertices) * [one_hot_type], dtype=torch.float32, device=vertices.device)
    node_probs = torch.ones((vertices.shape[0], 1), dtype=torch.float32, device=vertices.device)
    
    data = Data(pos=vertices, faces=faces, edge_index = edge_index, node_normals=node_normals, node_type=node_type, node_probs=node_probs)
    return data


def combine_and_merge(data1, data2):
    # 1. Concatenate positions.
    pos1 = data1.pos         # shape: [N1, D]
    pos2 = data2.pos         # shape: [N2, D]
    offset = pos1.size(0)    # for adjusting indices of data2
    pos_cat = torch.cat([pos1, pos2], dim=0)  # shape: [N1+N2, D]

    # 2. Merge duplicate vertices.
    # `inv` maps each vertex in pos_cat to its unique index.
    unique_pos, inv = torch.unique(pos_cat, dim=0, return_inverse=True)

    # 3. Adjust faces and remap indices.
    # Assume faces are stored as [num_faces, 3]. Adjust data2 faces by offset.
    faces1 = data1.faces                   # shape: [F1, 3]
    faces2 = data2.faces + offset          # shape: [F2, 3]
    faces_cat = torch.cat([faces1, faces2], dim=0)  # shape: [F1+F2, 3]
    # Remap faces indices to the new unique vertex indices.
    faces_cat = inv[faces_cat]

    # 4. Adjust edge_index and remap indices.
    # Assume edge_index is of shape [2, num_edges].
    edge1 = data1.edge_index               # shape: [2, E1]
    edge2 = data2.edge_index + offset      # shape: [2, E2]
    edge_cat = torch.cat([edge1, edge2], dim=1)  # shape: [2, E1+E2]
    # Remap edge_index using the same inverse mapping.
    edge_cat = inv[edge_cat]

    # 5. Create the new combined data object.
    new_data = Data(pos=unique_pos, faces=faces_cat, edge_index=edge_cat)
    return new_data




def create_pixel_ant(matrix, threshold, size_of_patch_in_mm, size_of_FR4_in_mm, size_of_ground, height, reflectors_dict=None, create_physical_pixel_mesh=False):
    size_of_patch_in_mm = torch.tensor(size_of_patch_in_mm, dtype=torch.float32, device=matrix.device)
    grid_size = matrix.shape[0]
    scale = size_of_patch_in_mm / grid_size
    size_of_ground = torch.tensor(size_of_ground, dtype=torch.float32, device=matrix.device)
    antenna_reltive_shift = (size_of_ground - size_of_patch_in_mm) / 2
    antenna_reltive_shift = torch.tensor(antenna_reltive_shift, dtype=torch.float32, device=matrix.device)
    height = torch.tensor(height, dtype=torch.float32, device=matrix.device)
    size_of_FR4_in_mm = torch.tensor(size_of_FR4_in_mm, dtype=torch.float32, device=matrix.device)
    
    #experiment
    # basis_patterns = create_basis_patterns(grid_size, device=matrix.device)
    # num_basis = len(basis_patterns)
     # Initialize coefficients instead of direct matrix
    # coefficients = torch.randn(num_basis, device=matrix.device, requires_grad=True)
    # pixel_data, pixel_probs = create_pixel_mesh_with_basis(coefficients, basis_patterns, threshold=0)
    
    if create_physical_pixel_mesh == True:
        pixel_data, pixel_probs = create_pixel_mesh(matrix, threshold)
    else:
        pixel_data, pixel_probs = create_pixel_mesh_with_pixel_probabilty(matrix, threshold)
        
    pixel_data.pos[:,:2] = pixel_data.pos[:,:2]*scale + antenna_reltive_shift   # scale to mm:
    pixel_data.pos[:,2] = pixel_data.pos[:,2] + height
    
    # create feed PEC:
    max_val = torch.max(pixel_probs)
    x, y = torch.nonzero(pixel_probs == max_val)[0].to(matrix.device)  # feed xy location

    feed_PEC_data = create_feed_PEC(x, y, height)
    feed_PEC_data.pos[:,:2] = feed_PEC_data.pos[:,:2]*scale + antenna_reltive_shift 
    feed_PEC_data.pos[:,2] = feed_PEC_data.pos[:,2] + height
    
    # create feed
    feed_data = create_feed(x, y, height)
    feed_data.pos[:,:2] = feed_data.pos[:,:2]*scale + antenna_reltive_shift     # scale to mm:
    feed_data.pos[:,2] = feed_data.pos[:,2] + height  # shift relative to ground
    
    # create ground
    ground = create_ground(matrix.device)
    ground.pos[:,:2] = ground.pos[:,:2]*size_of_ground
    
    # create FR4
    FR4_data = create_box_with_cube_hole(
        cube_size=scale,
        outerbox_length=size_of_FR4_in_mm,
        height=height,
        hole_center=(
            x*scale + antenna_reltive_shift - size_of_FR4_in_mm/2 + scale/2,
            y*scale + antenna_reltive_shift - size_of_FR4_in_mm/2 + scale/2
        )
    )
    FR4_data.pos[:,:2] = FR4_data.pos[:,:2] + size_of_FR4_in_mm/2  # shift to 0,0 coordinate
    FR4_data.pos[:,2] = FR4_data.pos[:,2] + height/2 
    FR4_data.pos[:,:2] = FR4_data.pos[:,:2]
    

    # Merge all graphs
    graph_dict = {
        'Antenna_Feed_STEP': feed_data,
        'Antenna_Feed_PEC_STEP': feed_PEC_data,
        'Antenna_PEC_STEP': pixel_data,
        'Antenna_ground': ground,
        'Env_FR4_STEP': FR4_data
    }
    
    # If reflectors_dict is provided, create reflectors and add to graph_dict
    if reflectors_dict is not None and isinstance(reflectors_dict, dict):
        reflectors_list = create_reflectors_on_sphere(reflectors_dict, device=matrix.device)
        for i, reflector in enumerate(reflectors_list):
            graph_dict[f'Reflector_{i}'] = reflector

    merged_pos, merged_node_types, merged_node_normals, merged_edge_index, merged_batch, offsets, key_to_offset, merged_node_probs, merged_faces, merged_node_areas = merge_graphs_from_dict(graph_dict)
    Pyg_graph = Data(
        x= torch.cat([merged_pos, merged_node_normals, merged_node_types], dim=-1),
        pos=merged_pos,
        node_type=merged_node_types,
        node_normals=merged_node_normals,
        edge_index=merged_edge_index,
        batch=merged_batch,
        node_probs=merged_node_probs
    )
    # Center the antenna positions
    # Pyg_graph.pos = Pyg_graph.pos - Pyg_graph.pos.mean(dim=0, keepdim=True)
    Pyg_graph.pos = Pyg_graph.pos - torch.tensor([size_of_FR4_in_mm/2, size_of_FR4_in_mm/2, height/2], device=Pyg_graph.pos.device)

    return Pyg_graph, pixel_probs

def add_reflector_to_graph(graph, reflector_matrix, params):
    threshold = params.get('reflector_threshold', 0.5)
    reflector = create_reflector_matrix(reflector_matrix, params, threshold)
    graph = graph.to(reflector.pos.device)
    graph_dict = decompose_pyg_graph(graph)
    graph_dict['Reflector'] = reflector
    graph =  merge_and_connect_graphs_from_dict(graph_dict, 'sphere', k=2, add_sphere=False)
    return graph




def create_reflector_matrix(reflector_matrix, reflectors_dict, threshold=0.5):
    antenna_reflector, _ = create_pixel_ant(
            reflector_matrix,  # matrix is now always a Parameter with gradients
            threshold = threshold,
            size_of_patch_in_mm=reflectors_dict['patch_x'],
            size_of_FR4_in_mm=reflectors_dict['ground_x'],
            size_of_ground=reflectors_dict['ground_x'],
            height=reflectors_dict['h'],
            create_physical_pixel_mesh=True
        )
    antenna_reflector = antenna_reflector.to(reflector_matrix.device)
    antenna_reflector_dict = decompose_pyg_graph(antenna_reflector)
    reflector = antenna_reflector_dict['Antenna_PEC_STEP']
    reflector_distance = reflectors_dict.get('reflector_distance', 3) #3
    # move reflector up in z direction
    reflector.pos[:,2] = reflector.pos[:,2] + reflector_distance
    # Scale reflector in x/y
    scale = reflectors_dict.get('reflector_scale', 1.5)
    reflector.pos[:, :2] = reflector.pos[:, :2] * scale
    reflector = reflector.to(reflector_matrix.device)
    return reflector

    

def create_basis_patterns(grid_size, device='cpu'):
    """
    Creates a set of basis patterns for the pixel antenna.
    Returns both simple and Fourier basis patterns.
    """
    basis_patterns = []
    
    # Add simple basis patterns (individual pixels)
    for i in range(grid_size):
        for j in range(grid_size):
            pattern = torch.zeros((grid_size, grid_size), device=device)
            pattern[i, j] = 1.0
            basis_patterns.append(pattern)
    
    # Add Fourier basis patterns
    for kx in range(grid_size//2):
        for ky in range(grid_size//2):
            pattern = torch.zeros((grid_size, grid_size), device=device)
            for i in range(grid_size):
                for j in range(grid_size):
                    pattern[i,j] = torch.cos(2*torch.pi*(kx*i/torch.tensor(grid_size, dtype=torch.float32, device=device) + ky*j/torch.tensor(grid_size, dtype=torch.float32, device=device)))
            basis_patterns.append(pattern/pattern.abs().max())
            
            # Add sine patterns too
            pattern = torch.zeros((grid_size, grid_size), device=device)
            for i in range(grid_size):
                for j in range(grid_size):
                    pattern[i,j] = torch.sin(2*torch.pi*(kx*i/torch.tensor(grid_size, dtype=torch.float32, device=device) + ky*j/torch.tensor(grid_size, dtype=torch.float32, device=device)))
            basis_patterns.append(pattern/pattern.abs().max())
    
    return torch.stack(basis_patterns)

def create_pixel_mesh_with_basis(coefficients, basis_patterns, threshold=0.5):
    """
    Creates a mesh from coefficients of basis patterns while preserving gradients.
    
    Args:
        coefficients: Tensor of coefficients for each basis pattern
        basis_patterns: Tensor of basis patterns [num_patterns, grid_size, grid_size]
        threshold: Threshold for converting to binary
    """
    # Compute matrix as linear combination of basis patterns
    # Ensure coefficients and basis_patterns are finite and avoid NaN propagation
    coefficients = torch.nan_to_num(coefficients, nan=0.0, posinf=1.0, neginf=-1.0)
    basis_patterns = torch.nan_to_num(basis_patterns, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Compute the matrix while ensuring numerical stability
    matrix = torch.sum(coefficients.view(-1, 1, 1) * basis_patterns, dim=0)
    matrix = torch.clamp(matrix, min=-1e6, max=1e6)  # Clamp to avoid overflow
    

    # Create mesh using the smoothed probabilities
    return create_pixel_mesh(matrix, threshold)
