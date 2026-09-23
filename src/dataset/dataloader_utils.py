
from pathlib import Path
import trimesh
import torch
from torch_geometric.data import Data
from src.geometry.mesh_functions import get_mesh_generator, model_stl_types, one_hot_encoding, get_icosphere
import math
from src.geometry.mesh_functions_pytorch_2 import add_edges_between_sphere_and_dict_pytorch
import numpy as np
from torchvision.transforms import v2
import pandas as pd



def get_example_number_to_idx_map(data_idx_to_example_number_map):
    return {v: k for k, v in data_idx_to_example_number_map.items()}


def get_data_idx_to_example_number_map(root_path):
    # set idx -> raw_idx maping
    data_idx_map = {}
    data_path =  sorted(Path(root_path).glob('processed/*.pt'))
    idx = 0
    for path in data_path:
        parts = str(path).split('/')
        parts = parts[-1].split('_')
        parts = parts[-1].split('.')
        example_number = parts[0]
        if example_number not in data_idx_map.values() and example_number != 'transform' and example_number != 'filter':
            data_idx_map.update({idx : example_number})
            idx += 1
    return data_idx_map


def get_example_number(stem_path):
    parts = stem_path.split('/')
    return parts[-1]


def load_obj_as_graph(mesh, material_type=None):
    """
    Loads an OBJ mesh and converts it to a PyTorch Geometric graph.
    
    :param mesh: The mesh object (with attributes such as vertices, faces, centroid, etc.).
    :param material_type: Optional material label for the mesh.
    :return: PyTorch Geometric Data object.
    """
    # Compute the average centroid from the mesh
    avg_centroid = torch.tensor(mesh.centroid, dtype=torch.float32)
    
    # Get node positions from the mesh vertices.
    pos = torch.tensor(mesh.vertices, dtype=torch.float32)
    
    # Get face connectivity.
    faces = torch.tensor(mesh.faces, dtype=torch.long)
    
    # Compute vertex normals.
    node_normals = torch.tensor(mesh.vertex_normals, dtype=torch.float32)
    
    # Compute node areas.
    vertex_areas = compute_vertex_areas(pos, faces).unsqueeze(-1)
    
    # Create graph connectivity (edge_index) from mesh edges.
    edges = mesh.edges_unique
    edge_index = torch.tensor(edges.T, dtype=torch.long)  # Shape: (2, num_edges)
    
    # --- One-hot encode the material type ---
    # Determine material_id from model_stl_types.
    material_id = model_stl_types.get(material_type, -1)
    num_materials = len(model_stl_types)
    
    if material_id < 0:
        # If material_type is not recognized, use a vector of zeros.
        vertex_types = torch.zeros((pos.shape[0], num_materials), dtype=torch.float32)
    else:
        # Create a one-hot vector for the given material_id.
        one_hot = torch.tensor(one_hot_encoding(material_id), dtype=torch.float32)
        # Repeat the one-hot vector for every vertex.
        vertex_types = one_hot.unsqueeze(0).repeat(pos.shape[0], 1)
    
    # Combine all node features into a single tensor.
    # Here we concatenate position, normal, area, and the one-hot material type.
    node_features = torch.cat([pos, node_normals, vertex_areas, vertex_types], dim=1)
    if material_type == 'sphere':
        node_probs = torch.zeros((pos.shape[0], 1), dtype=torch.float32)
    else:
        # For other materials, we can set a dummy probability or leave it as zeros.
        node_probs = torch.ones((pos.shape[0], 1), dtype=torch.float32)

    
    # Create the PyTorch Geometric Data object.
    graph_data = Data(
        x=node_features,       # Node feature matrix.
        pos=pos,               # Node positions.
        node_normals=node_normals,
        node_type=vertex_types,
        node_area=vertex_areas,
        edge_index=edge_index, 
        faces=faces.T,         # Transpose to shape (3, num_faces) if needed.
        avg_centroid=avg_centroid,
        node_probs = node_probs
    )
    
    return graph_data




def compute_vertex_areas(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    Compute vertex areas for a triangular mesh using a barycentric area assignment.
    
    Each vertex is assigned one third of the area of each face that touches it.
    
    Parameters:
        vertices (torch.Tensor): Tensor of shape (N, 3) containing the vertex positions.
        faces (torch.Tensor): Tensor of shape (F, 3) containing indices (into vertices) for each face.
    
    Returns:
        torch.Tensor: Tensor of shape (N,) containing the computed vertex areas.
    """
    # Ensure the input tensors are of the proper type
    if vertices.dtype != torch.float32 and vertices.dtype != torch.float64:
        vertices = vertices.float()
    
    # --- Step 1: Compute Face Areas ---
    # For each face, get its three vertices
    v0 = vertices[faces[:, 0]]  # shape: (F, 3)
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    
    # Compute two edge vectors for each face
    edge1 = v1 - v0
    edge2 = v2 - v0
    
    # Compute the cross product of the edge vectors for each face
    # The norm of the cross product gives twice the area of the triangle.
    cross_prod = torch.cross(edge1, edge2, dim=1)
    
    # Face areas: 0.5 * ||cross_prod||
    face_areas = 0.5 * torch.norm(cross_prod, dim=1)  # shape: (F,)
    
    # --- Step 2: Distribute Face Areas to Vertices ---
    num_vertices = vertices.shape[0]
    vertex_areas = torch.zeros(num_vertices, dtype=face_areas.dtype, device=face_areas.device)
    
    # Each face contributes one-third of its area to each of its three vertices.
    # Flatten the face indices and repeat each face area three times (divided by 3).
    faces_flat = faces.view(-1)  # shape: (F*3,)
    face_areas_repeated = face_areas.repeat_interleave(3) / 3.0  # shape: (F*3,)
    
    # Use index_add to accumulate contributions into vertex_areas.
    vertex_areas = vertex_areas.index_add(0, faces_flat, face_areas_repeated)
    
    return vertex_areas



def create_sphere_mesh(radius: float = 200,
                       n_lat: int = None,
                       n_lon: int = None,
                       n_points: int = None,
                       device=None):
    """
    Create a sphere mesh of a given radius using latitude and longitude sampling.
    
    You can either specify the resolution directly with:
      - n_lat: number of segments in the latitude direction (including poles, so rings will be n_lat-1)
      - n_lon: number of segments in the longitude direction.
      
    Alternatively, you can specify a target total number of vertices via n_points.
    In that case, the function will choose approximate values of n_lat and n_lon so that:
        total_vertices = (n_lat - 1)*n_lon + 2
    is approximately equal to n_points.
    
    Parameters:
        radius (float): The radius of the sphere (default is 1.0).
        n_lat (int): Number of latitude segments (must be >= 2 if provided).
        n_lon (int): Number of longitude segments (must be >= 3 if provided).
        n_points (int): Desired total number of vertices on the sphere. If provided, this overrides n_lat and n_lon.
        device: The device on which the tensors will be allocated (e.g., 'cpu' or 'cuda').
        
    Returns:
        vertices (torch.Tensor): Tensor of shape (N, 3) with the vertex positions.
        faces (torch.Tensor): Tensor of shape (F, 3) with indices into the vertices tensor forming triangles.
    """
    if device is None:
        device = torch.device('cpu')
    
    # If n_points is provided, compute approximate n_lat and n_lon.
    if n_points is not None:
        # A simple heuristic: choose n_lat approximately sqrt(n_points)
        n_lat = max(2, round(math.sqrt(n_points)))
        # Then, n_lon is approximated from the formula:
        # n_points = (n_lat - 1)*n_lon + 2  -->  n_lon = (n_points - 2)/(n_lat - 1)
        n_lon = max(3, round((n_points - 2) / (n_lat - 1)))
        print(f"Using n_points={n_points} results in n_lat={n_lat} and n_lon={n_lon}.")
    else:
        # Ensure that n_lat and n_lon are provided
        if n_lat is None or n_lon is None:
            raise ValueError("Either n_points or both n_lat and n_lon must be provided.")
        if n_lat < 2:
            raise ValueError("n_lat must be at least 2.")
        if n_lon < 3:
            raise ValueError("n_lon must be at least 3.")
    
    vertices = []
    faces = []
    
    # --- Create vertices ---
    # Top pole
    vertices.append([0.0, 0.0, radius])
    
    # Generate vertices for each ring (excluding the poles)
    # There will be (n_lat - 1) rings.
    for i in range(1, n_lat):
        theta = math.pi * i / n_lat  # theta in (0, pi)
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)
        
        for j in range(n_lon):
            phi = 2 * math.pi * j / n_lon  # phi in [0, 2*pi)
            sin_phi = math.sin(phi)
            cos_phi = math.cos(phi)
            
            x = radius * sin_theta * cos_phi
            y = radius * sin_theta * sin_phi
            z = radius * cos_theta
            vertices.append([x, y, z])
    
    # Bottom pole
    vertices.append([0.0, 0.0, -radius])
    
    vertices = torch.tensor(vertices, dtype=torch.float32, device=device)
    
    # --- Create faces ---
    # Index conventions:
    #   top pole has index 0.
    #   middle rings: rings 0 to (n_lat - 2) with each ring having n_lon vertices.
    #   bottom pole has index = len(vertices) - 1.
    top_index = 0
    bottom_index = vertices.shape[0] - 1
    
    # Top cap: Connect the top pole to the first ring.
    for j in range(n_lon):
        next_j = (j + 1) % n_lon
        face = [top_index, 1 + j, 1 + next_j]
        faces.append(face)
    
    # Intermediate rings: connect adjacent rings with two triangles per quad.
    for i in range(n_lat - 2):
        curr_ring_start = 1 + i * n_lon
        next_ring_start = 1 + (i + 1) * n_lon
        for j in range(n_lon):
            next_j = (j + 1) % n_lon
            # First triangle of the quad.
            face1 = [
                curr_ring_start + j,
                next_ring_start + j,
                next_ring_start + next_j
            ]
            # Second triangle of the quad.
            face2 = [
                curr_ring_start + j,
                next_ring_start + next_j,
                curr_ring_start + next_j
            ]
            faces.append(face1)
            faces.append(face2)
    
    # Bottom cap: Connect the last ring to the bottom pole.
    last_ring_start = 1 + (n_lat - 2) * n_lon
    for j in range(n_lon):
        next_j = (j + 1) % n_lon
        face = [bottom_index, last_ring_start + next_j, last_ring_start + j]
        faces.append(face)
    
    faces = torch.tensor(faces, dtype=torch.long, device=device)
    
    return vertices, faces

def merge_and_connect_graphs_from_dict_with_area(graph_dict, sphere_key, k):
    """
    Merges a dictionary of graphs and connects each sphere node to its k nearest neighbors in other graphs.

    Args:
        graph_dict (dict): Dictionary of graphs, where keys are graph identifiers and values are `Data` objects.
        sphere_key (str): Key of the sphere graph in the dictionary.
        k (int): Number of nearest neighbors to connect.

    Returns:
        Data: A merged graph with additional edges.
    """
    # Merge all graphs
    merged_pos, merged_node_types, merged_node_normals,merged_node_area, merged_edge_index, merged_batch, offsets, key_to_offset= merge_graphs_from_dict_with_area(graph_dict)

    # Add edges between sphere nodes and other graphs
    new_edges = add_edges_between_sphere_and_dict_pytorch(sphere_key, graph_dict, offsets, key_to_offset, k)

    # Combine new edges with existing edges
    merged_edge_index = torch.cat([merged_edge_index, new_edges], dim=1)

    # Return the merged graph
    return Data(x=torch.tensor(range(len(merged_pos))), pos=merged_pos, node_type=merged_node_types, node_normals=merged_node_normals,node_area=merged_node_area,  edge_index=merged_edge_index, batch=merged_batch)


def merge_graphs_from_dict_with_area(graph_dict):
    """
    Merges a dictionary of PyTorch Geometric graphs into a single graph.

    Args:
        graph_dict (dict): Dictionary of graphs, where keys are graph identifiers and values are `Data` objects.

    Returns:
        tuple: Merged node features, positions, edges, batch indices, and offsets, along with the key-to-offset mapping.
    """
    all_x, all_pos, all_node_types, all_node_normals, all_edge_index, all_batch, all_node_areas = [], [], [], [],[],[],[]
    offsets = [0]  # Track node index offsets for each graph
    offset = 0
    key_to_offset = {}

    for key, graph in graph_dict.items():
        num_nodes = graph.pos.shape[0]

        # Append node features, positions, edges, and batch indices
        #all_x.append(torch.tensor(graph.x) if graph.x is not None else torch.zeros((num_nodes, 1)))
        all_pos.append(graph.pos)
        all_node_types.append(graph.node_type)
        all_node_normals.append(graph.node_normals)
        all_node_areas.append(graph.node_area)
        all_edge_index.append(graph.edge_index + offset)
        all_batch.append(torch.full((num_nodes,), len(key_to_offset)))

        # Record offset and update
        key_to_offset[key] = offset
        offset += num_nodes
        offsets.append(offset)

    #merged_x = torch.cat(all_x, dim=0)
    merged_pos = torch.cat(all_pos, dim=0)
    merged_node_types = torch.cat(all_node_types, dim=0)
    merged_node_normals = torch.cat(all_node_normals, dim=0)
    merged_node_area = torch.cat(all_node_areas, dim=0)
    merged_edge_index = torch.cat(all_edge_index, dim=1)
    merged_batch = torch.cat(all_batch, dim=0)

    return  merged_pos, merged_node_types, merged_node_normals, merged_node_area, merged_edge_index, merged_batch, offsets, key_to_offset



def get_pyg_graph(mesh_paths, n_sphere_points):
    """_summary_

    Args:
        mesh_paths (_type_): path to .obj file directory
        n_sphere_points (int, optional): _description_. Defaults to 256.

    Returns:
        Pyg graph: with position areas and sphere (with sphere faces)
    """
    graph_dict = {}
    submeshes_generator = get_mesh_generator(mesh_paths)
    submesh_list = []
    submesh_type_list = []
    for submesh_path in submeshes_generator:
        if submesh_path.stem == 'Whole_Model_STEP' or submesh_path.stem == 'Antenna_STEP' or submesh_path.stem == 'Env_Vacuum_STEP':
            continue
        subgmesh = trimesh.interfaces.gmsh.load_gmsh(str(submesh_path))
        submesh = trimesh.Trimesh(**subgmesh,validate=True)
        submesh_list.append(submesh)
        submesh_type_list.append(submesh_path.stem)
        
    for submesh, submesh_type in zip(submesh_list, submesh_type_list):
        graph_dict[submesh_type] = load_obj_as_graph(submesh, submesh_type)
    
    # compute the avg cetroid:
    sum_pos = sum([sum(graph.pos) for graph in graph_dict.values()])
    num_verts = sum([len(graph.pos) for graph in graph_dict.values()])
    overall_avg_centroid = sum_pos / num_verts
    
    #shift the postions around the centroid
    for graph in graph_dict.values():
        # Ensure that the graph has a pos attribute.
        if hasattr(graph, 'pos'):
            graph.pos = graph.pos - overall_avg_centroid
    
    vertices, faces = get_icosphere(num_sphere_vertices=n_sphere_points,radius=200) #create_sphere_mesh(n_points = n_sphere_points)
    sphere_mesh  = trimesh.Trimesh(vertices=vertices, faces=faces)
    graph_dict['sphere'] = load_obj_as_graph(sphere_mesh, 'sphere')
    # graph_dict['sphere'].pos = graph_dict['sphere'].pos + overall_avg_centroid
    
    pyg_graph = merge_and_connect_graphs_from_dict_with_area(graph_dict, 'sphere', k=2)
    # plot_3d_points_edges(pyg_graph.pos, pyg_graph.edge_index.T.tolist())
    return pyg_graph
    
    
def resize_farfeild(farfeild, new_shape = (91,181)):
    # permute to fit v2.Resize format
    permuted_farfeild = farfeild.permute(2, 0, 1)
    resized_farfeild = v2.Resize(new_shape)(permuted_farfeild)
    resized_farfeild = resized_farfeild.permute(1, 2, 0)
    return resized_farfeild 


    
    
def farfeild_txt_to_np(txt_file_path:str):
    """
    Methode: parses farfeild text file
    text file header:
        theta [deg.]  Phi   [deg.]  Abs(Grlz)[]   Abs(Theta)[ ]  Phase(Theta)[deg.]  Abs(Phi  )[]  Phase(Phi )[deg.]  Ax.Ratio[]  
    """
    # Check if the file path ends with '.txt'
    assert txt_file_path.endswith('.txt'), "Input file should have a .txt extension."
    parts = txt_file_path.split('/')
    if isinstance(parts[-2] , int):
        example_num = int(parts[-2])
    else:
        example_num = 700000 # made up number to deal with test data set that doesnt have example numbers but is configured like examples 8000 and up
        
    df = pd.read_csv(txt_file_path, delim_whitespace=True)

    # parse the .txt file
    with open(txt_file_path, 'r') as file:
        # Read all lines in the file
        lines = file.readlines()
        data = []
        # Initialize a flag to skip the header
        skip_header = True
        # Iterate over each line
        for line in lines:
            if skip_header or '--' in line:
                skip_header = False
                continue
            # Split the line by whitespace
            columns = line.split()
            # Convert each column value to float and append to the data list
            data.append([np.float32(column) for column in columns])

    # Convert the data list to a NumPy array
    data_array = np.array(data)
    # Extract phi and theta columns
    theta_phi = data_array[:,:2]
    values = data_array[:, 2:]
    
    # examples ubder 8000 have resulutoin of 2.5
    if example_num <= 8000 :
        # Initialize a tensor with zeros
        farfeild = np.zeros((73, 144, 6))  # 73 rows for Theta (0 to 180 with 2.5 increments), 144 columns for Phi (0 to 357.5 with 2.5 increments), 6 channels
        scale = 2.5
    else:
        farfeild = np.zeros((37, 72, 6))  # 37 rows for Theta (0 to 180 with 5 increments), 72 columns for Phi (0 to 357.5 with 5 increments), 6 channels
        scale = 5
    # Iterate over each image index and set tensor values
    for i, (theta, phi) in enumerate(theta_phi):
        # Map theta and phi to indices in the tensor
        theta_index = int(theta / scale)  # Scale theta to match tensor indices
        phi_index = int(phi / scale)  # Scale phi to match tensor indices
        # Set tensor values at corresponding index
        farfeild[theta_index, phi_index, :] = values[i]
    # extract and orginize the abs and phase of E and B andd disgarding Abs(Grlz) and Ax.Ratio
    theta_abs = farfeild[:,:,1]
    theta_phase = farfeild[:,:,2]
    phi_abs = farfeild[:,:,3]
    phi_phase = farfeild[:,:,4]
    farfeild =  np.stack((theta_abs, phi_abs, theta_phase, phi_phase), axis=-1)
    return np.float32(farfeild)


def normalize_gain(abs_theta, abs_phi, img_shape=(64,64), gain_pol=None, device='cpu'):
    gain = abs_theta + abs_phi
    theta_rad = (torch.linspace(0, 180,img_shape[0], dtype=torch.float32, device=device) * torch.pi / 180)   #91
    phi_rad = (torch.linspace(0, 360,img_shape[1], dtype=torch.float32, device=device) * torch.pi / 180)     #181
    d_theta = torch.max(torch.diff(theta_rad))
    d_phi = torch.max(torch.diff(phi_rad))
    efficiency = torch.sum(torch.multiply(gain, torch.sin(theta_rad).unsqueeze(1))) * d_theta * d_phi / (4*torch.pi)
    directivity = gain / efficiency
    if gain_pol == None:
        return directivity
    else:
        directivity_pol = gain_pol / efficiency
        return directivity_pol 
    

def example_paramters_to_vector(ant_parameters: dict, model_parameters: dict):
    """_summary_ 
    Takes Antenne parameters dict and Envierment(model) parameters dict in reletive corudinants and returns them as a vector in abslute coirdenants 
    """
    ant_parameters_abs = ant_rel2abs(ant_parameters, model_parameters)
    # Convert all values to float32
    ant_parameters_abs = {key: torch.tensor(value, dtype=torch.float32) for key, value in ant_parameters_abs.items()}
    ant_abs = torch.tensor(list(ant_parameters_abs.values()), dtype=torch.float32)
    #ant_abs = torch.stack([value for value in ant_parameters_abs.values()], dim=1).reshape(ant_parameters_abs['fx'].shape[0], -1)
    model_parameters_abs = model_rel2abs(model_parameters)
    env_abs = torch.tensor(list(model_parameters_abs.values()), dtype=torch.float32)
    #env_abs = torch.stack([value for value in model_parameters_abs.values()], dim=1).reshape(model_parameters_abs['adx'].shape[0], -1)
    example_paramters_abs = torch.cat((ant_abs, env_abs), dim=0)
    return example_paramters_abs

 
def model_rel2abs(model_parameters):
    model_parameters_abs = model_parameters.copy()
    if model_parameters['type'] == 5:
        model_parameters_abs = model_parameters.copy()
        model_parameters_abs['Lz'] = model_parameters['Sz'] * model_parameters_abs['Lz']
        model_parameters_abs['Ly'] = model_parameters['Sy'] * model_parameters_abs['Ly']
        # model_parameters_abs['d'] = model_parameters['d'] * model_parameters['height']
    if model_parameters['type'] == 3:
        axes = ['x','y','z']
        dimensions = ['width','height','length']

        elements = ['a','b','c','d']
        for e in elements:
            for [i_axis, axis] in enumerate(axes):
                model_parameters_abs[e+'d' + axis] = model_parameters[e+'d' + axis] * model_parameters[dimensions[i_axis]]
                model_parameters_abs[e+'r' + axis] = model_parameters[e+'r' + axis] * model_parameters[e+'d' + axis] * model_parameters[dimensions[i_axis]]
        model_parameters_abs['a'] = model_parameters['a'] *model_parameters['width']
        model_parameters_abs['b'] = model_parameters['b'] * model_parameters['height']
        model_parameters_abs['c'] = model_parameters['c'] * model_parameters['height']
        # model_parameters_abs['d'] = model_parameters['d'] * model_parameters['height']
    return model_parameters_abs


def ant_rel2abs(ant_parameters: dict, model_parameters: dict):
    ant_parameters_abs = ant_parameters.copy()
    if model_parameters['type'] == 3:
        Sz = (model_parameters['length'] * model_parameters['adz'] * model_parameters['arz'] / 2 - ant_parameters['w'] / 2
              - model_parameters['feed_length'] / 2)
        Sy = model_parameters['height'] * model_parameters['ady'] * model_parameters['ary'] - ant_parameters['w']
        for key, value in ant_parameters.items():
            if len(key) == 4:
                if key[2] == 'z':
                    ant_parameters_abs[key] = np.round(value * Sz, decimals=2)
                if key[2] == 'y':
                    ant_parameters_abs[key] = np.round(value * Sy, decimals=2)
            if key == 'fx':
                ant_parameters_abs[key] = np.round(value * Sy, decimals=2)
    if model_parameters['type'] == 6:
        ant_parameters_abs['L1_rel'] = ant_parameters_abs['L1_rel'] * model_parameters['LG_y']
        ant_parameters_abs['L2_rel'] = ant_parameters_abs['L2_rel'] * (model_parameters['A_z'] - ant_parameters_abs['W2'])
        ant_parameters_abs['L3_rel'] = ant_parameters_abs['L3_rel'] * (model_parameters['LG_y'] - ant_parameters_abs['W1']*3 - - ant_parameters_abs['gap'])
        ant_parameters_abs['L4_rel'] = ant_parameters_abs['L4_rel'] * ant_parameters_abs['L2_rel']
    if model_parameters['type'] == 5:
        ant_parameters_abs = ant_parameters.copy()
        Sz = model_parameters['Sz'] - ant_parameters['w'] / 2 - model_parameters['feed_length'] / 2
        Sy = model_parameters['Sy'] - ant_parameters['w']
        for key, value in ant_parameters.items():
            if len(key) == 4:
                if key[2] == 'z':
                    ant_parameters_abs[key] = np.round(value * Sz, decimals=2)
                if key[2] == 'y':
                    ant_parameters_abs[key] = np.round(value * Sy, decimals=2)
            if key == 'fx':
                ant_parameters_abs[key] = np.round(value * Sy, decimals=2)
    return ant_parameters_abs
