import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors
import trimesh
from pathlib import Path
import torch
import torch_geometric.data as data
import gmsh 
from torch_geometric.transforms  import RadiusGraph
from torch_geometric.nn import knn
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import os
from collections import defaultdict
import math

model_stl_types = {
    'Antenna_Feed_PEC_STEP':7,# 10 for dataset-3, 7 for dataset-5
    'Antenna_Feed_STEP':10,
    'Antenna_PEC_STEP':9,
    'Env_FR4_STEP':3,
    'Env_PEC_STEP':1,
    'Env_Polycarbonate_STEP':8,
    'PEC_ground':4,
    'sphere':0,
    'PEC':6,
    'PEC_Reflector':2
    
}

def add_PEC_to_node_type(graph_dict,device='cpu'):
    one_hot_type = torch.tensor(get_submesh_type('PEC'),device=device)#.to('cuda')# 
    # node_type = torch.tensor(len(vertices) * [one_hot_type], dtype=torch.float32)
    for subgraph_name, subgraph in graph_dict.items():
        if 'PEC' in subgraph_name and subgraph != []:
            # Set to 1 wherever either original or PEC type had a non-zero value
            graph_dict[subgraph_name].node_type = torch.clamp(
                (graph_dict[subgraph_name].node_type + one_hot_type.to(graph_dict[subgraph_name].node_type.device)).bool().float(),
                min=0,
                max=1
            )
        # print(f"subgraph name: {subgraph_name} node type: {graph_dict[subgraph_name].node_type[0]}")
    return graph_dict


def one_hot_encoding(integer, num_classes = 10):
    """
    Encode an integer into a one-hot vector.

    Parameters:
    integer (int): The integer to be encoded.
    num_classes (int): The total number of classes.

    Returns:
    numpy.ndarray: A one-hot encoded vector with shape (1, num_classes).
    """
    integer = integer - 1
    if integer == -1:
        encoding = [0] * num_classes
        return encoding
        
    if integer < -1 or integer >= num_classes:
        raise ValueError("Integer should be within the range [0, num_classes).")
    
    encoding = [0] * num_classes
    encoding[integer] = 1
    return encoding


def one_hot_to_int(one_hot_vector):
    """
    Converts a one-hot encoded vector to an integer.

    Parameters:
    one_hot_vector (list or np.ndarray): A one-hot encoded vector.

    Returns:
    int: The integer corresponding to the one-hot encoded vector.
    """
    if isinstance(one_hot_vector, list):
        one_hot_vector = np.array(one_hot_vector)
    
    if not np.any(one_hot_vector):
        raise ValueError("The input vector is not a valid one-hot encoded vector.")
    
    return np.argmax(one_hot_vector)+1


def get_subgraph(graph, node_type:str):
    node_type_index = model_stl_types[node_type]
    subgrph_indexes = get_nodes_of_type( graph.node_type, node_type_index -1)
    subgraph = graph.subgraph(subgrph_indexes)
    return subgraph

def get_antenna_and_sphere( graph):
    graph_antenna, graph_antenna_indexs = get_subgraph_and_indexs(graph, 'PEC:ant')
    sphere_indexs = torch.arange(graph.pos.size(0) - (256 + len(graph_antenna_indexs)),graph.pos.size(0) - (len(graph_antenna_indexs)) )
    ant_and_sphere = graph.subgraph(torch.cat((graph_antenna_indexs, sphere_indexs)))
    return ant_and_sphere

def get_subgraph_and_indexs(graph, node_type:str):
    node_type_index = model_stl_types[node_type]
    subgrph_indexes = get_nodes_of_type( graph.node_type, node_type_index -1)
    subgraph = graph.subgraph(subgrph_indexes)
    return subgraph, subgrph_indexes


def get_sphere_node_idx(node_type_list):
    desired_type_nodes_idxs = zero_vector_indices = torch.all(node_type_list == 0, dim=1).nonzero(as_tuple=True)[0]
    return desired_type_nodes_idxs


def get_nodes_of_type(node_type_list, node_type_index):
    """
    Get the indices of nodes that have a specific type in a PyG graph.

    Parameters:
    - node_type_list: list object containing the graph node types
    - node_type_index: int, the index of the one-hot encoded column representing the desired node type

    Returns:
    - torch.Tensor: Indices of nodes that have the specified type
    """
    desired_type_nodes_idxs = torch.where(node_type_list[:, node_type_index] == 1)[0]
    return desired_type_nodes_idxs


def load_and_parce_scene_mesh(raw_path:str,remesh = False, save_submeshes = False):
    """
    Methode:
        This function loads a .stl file to a scene mesh wich is a dictionary of submeshes 
        each corisponding to a node type string (the key) and depending on the 

    Args:
        raw_path (str): path_to .stl file
        remesh (bool): add nodes and faces
    Returns:
        trimesh: defined on face and nodes comprised from the submeshes extravted from the scene mesh
        type_list (list): list holding the type of each node
    """
    raw_mesh_scene = trimesh.load(raw_path)
    node_list = []
    face_list = []
    type_list = []
    # the raw mesh is loaded as a scene so it is a compasition of meshes each with the mesh type ['feed:hot'],['pec:ant'],['fr4:subtrate'] ..
    # so we go over each and add their vertcies and faces however the faces need to be advanced to repersent the new vertex index
    acumulater = 0
    for idx, key in enumerate(raw_mesh_scene.geometry):
        #print(raw_mesh.geometry[key].faces)
        #print(len(raw_mesh.geometry[key].vertices))
        submesh = raw_mesh_scene.geometry[key]
        if remesh == True:
            submesh = submesh.subdivide()  # .subdivide_to_size(25) # #.subivide_loop
            if key != 'fr4:subtrate' :
                submesh = submesh.subdivide()
                submesh = submesh.subdivide()

        node_list.extend(submesh.vertices)
        type_list.extend(len(submesh.vertices) * [model_stl_types[key]])
        if len(type_list) != len(node_list):
            print("len(type_list) != len(node_list)")
        faces = submesh.faces + acumulater
        face_list.extend(faces)
        acumulater += len(submesh.vertices)

    parsed_mesh = trimesh.Trimesh(vertices=node_list,faces=face_list, process = False)

    return parsed_mesh, type_list


def load_and_parce_scene_mesh_with_sphere(raw_path:str,remesh = False, save_submeshes = False):
    """
    Methode:
        This function loads a .stl file to a scene mesh wich is a dictionary of submeshes 
        each corisponding to a node type string (the key) and depending on the 

    Args:
        raw_path (str): path_to .stl file
        remesh (bool): add nodes and faces
    Returns:
        trimesh: defined on face and nodes comprised from the submeshes extravted from the scene mesh
        type_list (list): list holding the type of each node
    """
    raw_mesh_scene = trimesh.load(raw_path)
    node_list = []
    face_list = []
    type_list = []
    submesh_list = []
    # the raw mesh is loaded as a scene so it is a compasition of meshes each with the mesh type ['feed:hot'],['pec:ant'],['fr4:subtrate'] ..
    # so we go over each and add their vertcies and faces however the faces need to be advanced to repersent the new vertex index
    acumulater = 0
    for idx, key in enumerate(raw_mesh_scene.geometry):
        #print(raw_mesh.geometry[key].faces)
        #print(len(raw_mesh.geometry[key].vertices))
        submesh = raw_mesh_scene.geometry[key]
        submesh_list.append(submesh)
        if remesh == True:
            submesh = submesh.subdivide()  # .subdivide_to_size(25) # #.subivide_loop
            if key != 'fr4:subtrate' :
                submesh = submesh.subdivide()
                submesh = submesh.subdivide()

        node_list.extend(submesh.vertices)
        type_list.extend(len(submesh.vertices) * [one_hot_encoding(model_stl_types[key])])
        if len(type_list) != len(node_list):
            print("len(type_list) != len(node_list)")
        faces = submesh.faces + acumulater
        face_list.extend(faces)
        acumulater += len(submesh.vertices)

    parsed_mesh = trimesh.Trimesh(vertices=node_list,faces=face_list, process = False)
    
    # add the sphere: 
    sphere_points = add_sphere_around_model_np(parsed_mesh.vertices, num_points = 256, radius = 200)
    node_list.extend(sphere_points)
    avg_centroid = parsed_mesh.centroid
    sphere_points = [[x1 + x2, y1 + y2, z1 + z2] for (x1, y1, z1), (x2, y2, z2) in zip(sphere_points, [avg_centroid.tolist()] * len(sphere_points))]
    sphere_edges = get_sphere_edges(submesh_list, sphere_points, len(parsed_mesh.vertices))
    sphere_vertex_face_normals = np.zeros((len(sphere_points), 3))
    node_normals_list = np.vstack((parsed_mesh.vertex_normals, sphere_vertex_face_normals))
    sphere_vertex_type  = np.tile(one_hot_encoding(0),(len(sphere_points),1)) 
    type_list.extend(sphere_vertex_type)
    total_edges = np.vstack((faces_to_edges(faces) ,sphere_edges))
    
    graph_mesh = {'vertcies':np.array(node_list),
                  'vertex_types':np.array(type_list),
                  'faces':np.array(face_list),
                  'edges':total_edges,
                  'vertex_normals':np.array(node_normals_list),
                  'sphere_edges':sphere_edges}

    return graph_mesh






def get_submesh_type(stem_path):
    # example: 'Antenna_PEC_STEP' returns 'PEC' number represintation 
    #parts = stem_path.split('_')
    intger_type_repesntaion = model_stl_types[stem_path]
    return one_hot_encoding(intger_type_repesntaion)


def plot_3d_points_edges(points, edges, sphere_edges = [], title = None, ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # Plot points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', marker='o')

    # Plot edges
    for edge in edges:
        start_point = points[edge[0]]
        end_point = points[edge[1]]
        ax.plot([start_point[0], end_point[0]], 
                [start_point[1], end_point[1]], 
                [start_point[2], end_point[2]], c='r')
    
    if sphere_edges:
        # Plot edges
        for edge in sphere_edges:
            start_point = points[edge[0]]
            end_point = points[edge[1]]
            ax.plot([start_point[0], end_point[0]], 
                    [start_point[1], end_point[1]], 
                    [start_point[2], end_point[2]], c='g')
            

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if title:
        ax.set_title(title, fontsize=10)
    if ax is not None:
        plt.show()
    return ax
   
    
def add_self_edges(points, edges=[]):
    """
    Add self-edges for each point in the mesh.

    Parameters:
        points (list): List of points in the mesh.
        edges (list, optional): List of edges in the mesh. Default is an empty list.

    Returns:
        list: Updated list of edges including self-edges.
    """
    self_edges = []
    num_points = len(points)

    # Adding self-edges for each point
    for i in range(num_points):
        self_edges.append((i, i))

    # Combining existing edges with self-edges
    all_edges = np.vstack((edges ,self_edges))
    return all_edges
    
    
    
    
    
def add_sphere_around_model_np(model_nodes, radius = 50, num_points = 100, debug = False):
    """
    Create a spherical points around mesh   
    """
    num_points_per_angle = int(np.sqrt(num_points))
    theta = np.linspace(0, np.pi, num_points_per_angle)
    phi = np.linspace(0, 2*np.pi, num_points_per_angle)
    theta, phi = np.meshgrid(theta, phi)
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    spherical_vertcies_pos = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=-1)
    number_nodes_in_graph = len(model_nodes)
    sphere_node_list = [number_nodes_in_graph + i for i in range(num_points)]
    return spherical_vertcies_pos


    
def create_box(width, height, depth, orientation=(0, 0, 0)):
    """
    Create a box with specified width, height, depth, and orientation centered around (0, 0, 0).

    Parameters:
        width (float): Width of the box.
        height (float): Height of the box.
        depth (float): Depth of the box.
        orientation (tuple): Orientation of the box as a tuple of Euler angles (in radians),
                             default is (0, 0, 0) for no rotation.

    Returns:
        numpy.ndarray: Vertices of the box.
        list of tuples: Edges of the box.
    """
    # Define half-dimensions to center the box around the origin
    half_width = width / 2
    half_height = height / 2
    half_depth = depth / 2

    # Define vertices of the box
    vertices = np.array([
        [-half_width, -half_height, -half_depth],    # Vertex 0
        [half_width, -half_height, -half_depth],     # Vertex 1
        [half_width, half_height, -half_depth],      # Vertex 2
        [-half_width, half_height, -half_depth],     # Vertex 3
        [-half_width, -half_height, half_depth],     # Vertex 4
        [half_width, -half_height, half_depth],      # Vertex 5
        [half_width, half_height, half_depth],       # Vertex 6
        [-half_width, half_height, half_depth]       # Vertex 7
    ])

    # Rotate the vertices based on orientation
    if orientation != (0, 0, 0):
        rotation_matrix = np.array([
            [np.cos(orientation[2]), -np.sin(orientation[2]), 0],
            [np.sin(orientation[2]), np.cos(orientation[2]), 0],
            [0, 0, 1]
        ]) @ np.array([
            [np.cos(orientation[1]), 0, np.sin(orientation[1])],
            [0, 1, 0],
            [-np.sin(orientation[1]), 0, np.cos(orientation[1])]
        ]) @ np.array([
            [1, 0, 0],
            [0, np.cos(orientation[0]), -np.sin(orientation[0])],
            [0, np.sin(orientation[0]), np.cos(orientation[0])]
        ])
        vertices = vertices @ rotation_matrix

    # Define edges of the box
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),      # Bottom square
        (4, 5), (5, 6), (6, 7), (7, 4),      # Top square
        (0, 4), (1, 5), (2, 6), (3, 7)       # Connecting edges
    ]

    return vertices, edges

def add_bidirctional_edges_between_unconnected_meshes(nodes, edges = []):
    '''
    goes over all nodes and sets edges betwen nodes of the same location
    '''
    # Fit NearestNeighbors model
    nn_model = NearestNeighbors(n_neighbors=8, algorithm='ball_tree')
    nn_model.fit(nodes)
    distances, indices = nn_model.kneighbors(nodes)
    
    new_edges = []
    #nn_graph = nn_model.kneighbors_graph(nodes).toarray()
    for idx_source, nn_distance_list in enumerate(distances):
        for idx_dest, distance in enumerate(nn_distance_list):
            if distance == 0 and idx_source != indices[idx_source][idx_dest]:
                new_edges.append((idx_source, indices[idx_source][idx_dest]))
    
    edges = edges + new_edges
    return edges    

def add_directional_edges_between_unconnected_meshes(nodes, edges=[]):
    '''
    Goes over all nodes and sets directional edges between nodes of the same location.
    '''
    # Fit NearestNeighbors model
    nn_model = NearestNeighbors(n_neighbors=8, algorithm='ball_tree')
    nn_model.fit(nodes)
    distances, indices = nn_model.kneighbors(nodes)
    
    new_edges = []
    for idx_source, nn_distance_list in enumerate(distances):
        for idx_dest, distance in enumerate(nn_distance_list):
            if distance == 0 and idx_source != indices[idx_source][idx_dest]:
                new_edges.append((idx_source, indices[idx_source][idx_dest]))
    
    edges = edges + new_edges
    return edges

def add_nearest_neighbor_edges(graph1: Data, graph2: Data, num_nodes: int = 256) -> Data:
    """
    Adds edges from the last `num_nodes` nodes of `graph1` to their nearest neighbors in `graph2`.
    # in our case 
    Parameters:
    - graph1 (Data): The first PyTorch Geometric graph. (graph without antenna)
    - graph2 (Data): The second PyTorch Geometric graph. (new antenna)
    - num_nodes (int): The number of nodes from the end of `graph1` to connect to `graph2`.
    
    Returns:
    - new edge index lis: The updated `graph1 edges` with additional edges.
    """
    # Step 1: Extract the last `num_nodes` nodes of graph1
    sphere_nodes = graph1.pos[-num_nodes:]

    # Step 2: Find the nearest neighbors in graph2
    _, nearest_neighbor_indices = knn(graph2.pos, sphere_nodes, k=1)

    # set the node idxs to fit graph1
    nearest_neighbor_indices += graph1.pos.shape[0]
    sphere_indexs = torch.arange(graph1.pos.size(0) - num_nodes, graph1.pos.size(0))
    
    new_edges_to_sphear = torch.cat((nearest_neighbor_indices.unsqueeze(0), sphere_indexs.unsqueeze(0)), dim=0)
    # Update graph1's edge_index
    edge_index = torch.cat((graph1.edge_index, new_edges_to_sphear), dim=1)
    
    return edge_index


def extract_faces_from_edges(edge_index):
    """
    Extract triangular faces from edge_index by finding node triplets.
    """
    edge_index = to_undirected(edge_index)

    adj_dict = {}
    for i, j in edge_index.t().tolist():
        if i not in adj_dict:
            adj_dict[i] = set()
        adj_dict[i].add(j)

    faces = set()
    for i in adj_dict:
        for j in adj_dict[i]:
            for k in adj_dict[j]:
                if k in adj_dict[i] and i < j < k:  # Avoid duplicates
                    faces.add((i, j, k))

    faces = torch.tensor(list(faces), dtype=torch.long)  # Shape [num_faces, 3]
    return faces

def pyg_graph_to_trimesh(pyg_graph, process=True):
    subgraph_faces =  extract_faces_from_edges(pyg_graph.edge_index)
    vertices_np = pyg_graph.pos.cpu().numpy()
    faces_np = subgraph_faces.cpu().numpy()
    trimesh_mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np, process=process)
    return trimesh_mesh

def save_subgraph(subgraph, path_to_save_trimesh):
    trimesh_mesh = pyg_graph_to_trimesh(subgraph)
    trimesh_mesh.export(path_to_save_trimesh)
    print(f"Mesh saved as {path_to_save_trimesh}")

def save_full_graph_as_trimesh(graph, path_to_save_trimesh, debug= False):
    os.makedirs(path_to_save_trimesh, exist_ok=True)
    trimesh_submesh_dict = {}
    
    for stl_type in model_stl_types.keys():
        node_type_index = model_stl_types[stl_type]
        subgrph_indexes = get_nodes_of_type( graph.node_type, node_type_index -1)
        subgraph = graph.subgraph(subgrph_indexes)
        trimesh_mesh = pyg_graph_to_trimesh(subgraph)
        # Save mesh to STL
        filename = path_to_save_trimesh +'/' + stl_type + '.stl'
        trimesh_mesh.export(filename)
        print(f"Mesh saved as {filename}")
        trimesh_submesh_dict[stl_type] = trimesh_mesh
        if debug:
            plot_3d_points_edges(trimesh_mesh.vertices , trimesh_mesh.edges.tolist())        
    return  trimesh_submesh_dict




def get_boundary_edges_axis_aligned(mesh):
    trimesh_mesh =  pyg_graph_to_trimesh(mesh)
    # Step 1: Identify boundary edges (edges with only one face)
    boundary_edges = trimesh_mesh.edges_sorted[trimesh_mesh.edges_unique_inverse == 1]
    boundary_vertices = np.unique(boundary_edges.flatten())
    # Step 2: Extract positions of boundary vertices
    vertex_positions = mesh.vertices.copy()
    return boundary_edges, vertex_positions

def group_axis_aligned(vertices, tolerance=1e-5):
    """
    Group vertices into axis-aligned vertical and horizontal groups.

    Args:
        vertices (np.ndarray): An array of shape (N, 3) representing vertex positions (x, y, z).
        tolerance (float): Numerical tolerance for comparing floating-point values.

    Returns:
        vertical_groups (list): List of groups of vertex indices that lie on vertical lines.
        horizontal_groups (list): List of groups of vertex indices that lie on horizontal lines.
    """
    # Dictionaries to store groups of vertices
    x_to_indices = defaultdict(list)  # For vertical alignment
    y_to_indices = defaultdict(list)  # For horizontal alignment

    # Group vertex indices by unique x-coordinates and y-coordinates
    for i, (x, y, z) in enumerate(vertices):
        x_to_indices[round(x, int(-np.log10(tolerance)))].append(i)
        y_to_indices[round(y, int(-np.log10(tolerance)))].append(i)

    # Extract vertical and horizontal groups
    vertical_groups = [indices for indices in x_to_indices.values() if len(indices) > 1]
    horizontal_groups = [indices for indices in y_to_indices.values() if len(indices) > 1]

    return vertical_groups, horizontal_groups
  
def align_boundary_edges_to_average(pyg_mesh):
    """
    Align boundary edges to their average coordinate (axis-aligned).

    Args:
        mesh (trimesh.Trimesh): The input mesh.

    Returns:
        trimesh.Trimesh: The updated mesh with boundary edges aligned to their average.
    """
    mesh =  pyg_graph_to_trimesh(pyg_mesh, process=False)
    # Step 1: Identify boundary edges
    boundary_edges = mesh.edges_sorted[mesh.edges_unique_inverse == 1]
    vertex_positions = mesh.vertices.copy()  # Copy vertex positions

    for edge in boundary_edges:
        v1_idx, v2_idx = edge
        v1, v2 = vertex_positions[v1_idx], vertex_positions[v2_idx]

        # Step 2: Calculate average coordinates
        avg_x = (v1[0] + v2[0]) / 2.0
        avg_y = (v1[1] + v2[1]) / 2.0

        # Step 3: Decide whether to align x or y based on the dominant axis
        if abs(v1[0] - v2[0]) < abs(v1[1] - v2[1]):  # Align x (vertical edge)
            vertex_positions[v1_idx][0] = avg_x
            vertex_positions[v2_idx][0] = avg_x
        else:  # Align y (horizontal edge)
            vertex_positions[v1_idx][1] = avg_y
            vertex_positions[v2_idx][1] = avg_y

    # Step 4: Update mesh vertices
    mesh.vertices = vertex_positions
    return mesh.vertices  
    
def add_one_directiona_edges_from_model_to_sphere(model_nodes, sphere_nodes, edges = []):
    '''
    1) find for each node on the sphere its model node nearest neighbor and coneect one directional edge
    2) find for each node on the sphere its sphere node nearest neighbor 
    3) calculate the found sphere node nearest neighbor nearest neighbor to the model
    4) connect one directional sphere node to the found nearest neighbor nearest neighbor
    '''
    sphere_edges = []
    number_of_model_nodes = len(model_nodes) 
    number_of_sphere_nodes = len(sphere_nodes) 
    nn_model = NearestNeighbors(n_neighbors=2, algorithm='ball_tree')
    nn_model.fit(model_nodes)
    _, model_indices = nn_model.kneighbors(sphere_nodes)
    model_neighbors = [idx[0] for idx in model_indices]
    sphere_node_indeces = [ number_of_model_nodes + idx for idx in range(number_of_sphere_nodes)]
    # Add edges from model nodes to sphere nodes
    sphere_edges.extend(zip(model_neighbors, sphere_node_indeces))
    
    
    
    # add the sphere neaeres nerest neighbor:
    nn_sphere = NearestNeighbors(n_neighbors=2, algorithm='ball_tree')
    nn_sphere.fit(sphere_nodes)
    _, sphere_indices = nn_sphere.kneighbors(sphere_nodes)
    
    
    for idx, sphere_idx in enumerate(sphere_indices):
        sphere_nodes_nn_idxs = sphere_idx[1]
        sphere_nodes_nn = sphere_nodes[sphere_nodes_nn_idxs]
        _, model_indices = nn_model.kneighbors([sphere_nodes_nn])
        model_neighbors = [idx[0] for idx in model_indices]
        sphere_edges.append((model_neighbors[0], number_of_model_nodes + idx))
        
    return edges, sphere_edges
    
    
    


def add_nodes_from_mesh2_to_mesh1(mesh1, mesh2):
    
    # Iterate over nodes in mesh1
    for node in mesh2.vertices:
        # Cast ray from node in mesh1
        origin = node
        directions = [
            [1, 0, 0],  # +X
            #[-1, 0, 0], # -X not nedded
            [0, 1, 0],  # +Y
            #[0, -1, 0], # -Y
            [0, 0, 1],  # +Z
            #[0, 0, -1]  # -Z
        ]
        ray_mesh_intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh1)

        # Find intersections
        intersections, index_ray, index_tri = ray_mesh_intersector.intersects_location(np.tile(origin,(len(directions),1)), directions)

        # Check intersections
        for i in range(len(intersections)):
            point = intersections[i]
            ray_index = index_ray[i]
            tri_index = index_tri[i]

            # If intersection distance is not zero, add as new node in mesh2
            if sum(point - origin) == 0:
                # Add point as new node in mesh1
                new_node_index = len(mesh1.vertices)
                mesh1.vertices = np.vstack((mesh1.vertices, [point]))
                
                # Connect new node to the vertices of the intersected face
                face_vertices = mesh1.faces[tri_index]
                new_faces = [[new_node_index, face_vertices[0],face_vertices[1]],
                             [new_node_index, face_vertices[0], face_vertices[2]],
                             [new_node_index, face_vertices[1],face_vertices[2]]]
                mesh1.faces = np.vstack((mesh1.faces, new_faces))
    
    return mesh1


def get_mesh_generator(mesh_path:str, mesh_format:str = 'stp', Debug = False):
    """
    methode: collects all submeshes in a folder
    Args: mesh_path 
    Returns: combined mesh generator
    """
    submesh_generator = Path(mesh_path).glob('*.' + mesh_format)
    return submesh_generator
    
def get_submesh_list(submeshes_generator):
    submesh_list = []
    submesh_type_list = []
    for submesh_path in submeshes_generator:
        if submesh_path.stem == 'Whole_Model_STEP' or submesh_path.stem == 'Antenna_STEP' or submesh_path.stem == 'Env_Vacuum_STEP':
            continue
        subgmesh = trimesh.interfaces.gmsh.load_gmsh(str(submesh_path))
        submesh = trimesh.Trimesh(**subgmesh,validate=True)
        submesh_list.append(submesh)
        vertex_type = np.array(get_submesh_type(submesh_path.stem))
        submesh_type_list.append(vertex_type)
        
    return submesh_list, submesh_type_list

def resample_meshes_on_tangent_faces(submesh_list, Debug = False):
    
    mesh_volume = trimesh.Trimesh(vertices=[], faces=[])
    resamples_submesh_list = []

    for idx1, submesh1 in enumerate(submesh_list): 
        if submesh1.is_volume :
            # Union all the original meshes into a single mesh
            mesh_volume = mesh_volume + submesh1
        
        
        for idx2, submesh2 in enumerate(submesh_list):
            if idx1 == idx2:
                continue
            submesh1 = add_nodes_from_mesh2_to_mesh1(submesh1, submesh2)
            
        resamples_submesh_list.append(submesh1)
        
    return resamples_submesh_list, mesh_volume

def create_rectangle(p1, p2, width):
    """
    Create a rectangular mesh in trimesh between two points p1 and p2 with given width.
    """
    p1, p2 = map(np.array, [p1, p2])
    
    # Direction vector of the line
    direction = p2 - p1
    # Perpendicular vector for the width
    perp_direction = np.array([-direction[1], direction[0]])
    perp_direction = (width / 2) * perp_direction / np.linalg.norm(perp_direction)
    
    # Compute rectangle corners
    v1 = p1 + perp_direction
    v2 = p1 - perp_direction
    v3 = p2 + perp_direction
    v4 = p2 - perp_direction
    # Compute rectangle corners
    # v1 = p1 + [- (width / 2), -width / 2]
    # v2 = p1 - [- (width / 2), + width / 2]
    # v3 = p2 - (width / 2)
    # v4 = p2 - [(-width / 2), -width / 2]
    
 

    # Define vertices and faces
    vertices = np.array([v1, v2, v3, v4])
    faces = np.array([[0, 1, 2], [1, 3, 2]])  # Two triangles for the rectangle
    
    # Create the trimesh object
    return trimesh.Trimesh(vertices=vertices, faces=faces)


    

 
 
def create_buffered_closed_path(points, buffer=0.5):
    points = np.array(points)  # Convert to numpy array for easier manipulation
    n = len(points)

    # Compute segment directions
    directions = np.diff(points, axis=0)
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)  # Normalize

    # Compute normals
    normals = np.zeros_like(directions)
    normals[:, 0] = -directions[:, 1]  # Perpendicular x
    normals[:, 1] = directions[:, 0]  # Perpendicular y

    # Offset points
    offset_points_positive = []
    offset_points_negative = []

    # First point
    offset_points_positive.append(points[0] + buffer * np.sign(normals[0]))
    offset_points_negative.append(points[0] - buffer * np.sign(normals[0]))

    # Middle points
    for i in range(1, n - 1):
        # Average normals for smooth transitions
        avg_normal = (normals[i - 1] + normals[i]) / 2
        avg_normal /= np.linalg.norm(avg_normal)  # Normalize
        offset_points_positive.append(points[i] + buffer  /  np.sign(avg_normal))
        offset_points_negative.append(points[i] - buffer  /  np.sign(avg_normal))

    # Last point
    offset_points_positive.append(points[-1] + buffer * np.sign(normals[-1]))
    offset_points_negative.append(points[-1] - buffer * np.sign(normals[-1]))

    # Combine positive and negative offsets to form a closed path
    closed_path = np.vstack([
        offset_points_positive, 
        offset_points_negative[::-1]  # Reverse negative offsets
    ])

    return closed_path

def create_mesh_from_2d_path(path, width, debug=False):
    buffer = width/2
    closed_path = create_buffered_closed_path(path, buffer) # without the closing point
    closed_path = np.vstack((closed_path,[closed_path[0, 0], closed_path[0, 1]])) #adding closing point
    closed_path_tri = trimesh.load_path(closed_path)
    points, triangles = closed_path_tri.triangulate()
    points = np.hstack((np.zeros((points.shape[0], 1)),points[:,0:1], points[:,1:2]))
    mesh = trimesh.Trimesh(vertices=points, faces=triangles, process=True)
    if debug:
        plot_3d_points_edges(mesh.vertices , mesh.edges.tolist())
    return mesh 
  
def create_mesh_from_2d_paths(paths, width, debug=False):
    full_mesh = []
    for path_2d in paths:
        path_2d_mesh = create_mesh_from_2d_path(path_2d, width)
        full_mesh.append(path_2d_mesh)
    full_mesh = trimesh.util.concatenate(full_mesh)
    return full_mesh

        
def close_and_offset_path(path, offset_distance=1):
        path = np.array(path)
        offset_path = path + offset_distance
        offset_path = offset_path[::-1]
        closed_path = np.vstack((path,offset_path,path[0]))
        closed_path -= offset_distance/2
        return closed_path

    

def voxilize_bounding_box(min_bound, max_bound, voxel_size):
    """
    Generate a voxelized representation of a bounding box.

    Parameters:
    - min_bound: (numpy.ndarray) A 3-element array specifying the minimum x, y, z coordinates of the bounding box.
    - max_bound: (numpy.ndarray) A 3-element array specifying the maximum x, y, z coordinates of the bounding box.
    - voxel_size: (float) The size of each voxel (assumed to be cubic).

    Returns:
    - combined_voxel_mesh: (trimesh.Trimesh) A mesh consisting of all the voxel meshes combined to fill the bounding box.
    
    This function calculates the number of voxels needed along each axis based on the provided bounding box dimensions and voxel size.
    It then iterates through each position within the bounding box, creates a voxel mesh at that position, and combines all these
    voxel meshes into a single Trimesh object which is returned.
    """
    num_voxels = np.ceil((max_bound - min_bound) / voxel_size).astype(int)

    voxel_meshes = trimesh.Trimesh(vertices=[], faces=[])

    # Iterate through each voxel position
    for x in range(num_voxels[0]):
        for y in range(num_voxels[1]):
            for z in range(num_voxels[2]):
                # Define the voxel bounds
                voxel_min = min_bound + np.array([x, y, z]) * voxel_size
                voxel_max = voxel_min + voxel_size
                
                # Create a box (voxel) mesh
                voxel_mesh = trimesh.creation.box(extents=[voxel_size, voxel_size, voxel_size], transform=trimesh.transformations.translation_matrix(voxel_min + voxel_size / 2))
                
                # Add the voxel mesh to the list
                voxel_meshes = voxel_meshes + voxel_mesh
    return voxel_meshes
        
def create_complimentry_mesh(tri_mesh, mesh_volume, voxlize = True):
    scene = trimesh.Scene([tri_mesh])
    # Compute the bounding box
    bounding_box = scene.bounds
    min_corner = bounding_box[0]
    max_corner = bounding_box[1]
    size = max_corner - min_corner
    combined_boundbox = trimesh.creation.box(extents=size, transform=trimesh.transformations.translation_matrix(min_corner + size / 2))
    complementary_mesh = trimesh.boolean.difference([combined_boundbox, mesh_volume])
    if voxlize:
        box_resultion = 10
        min_bound, max_bound =  complementary_mesh.bounds
        voxel_size = (abs(max_bound - min_bound)).max()/box_resultion
        voxel_meshes =  voxilize_bounding_box(min_bound, max_bound, voxel_size)
        complementary_mesh = trimesh.boolean.intersection([complementary_mesh, voxel_meshes])
           
    return complementary_mesh
   
def make_edges_bidirectional(edge_list):
    bidirectional_edges = []
    for edge in edge_list:
        # Original edge
        bidirectional_edges.append(edge)
        # Reverse edge
        reverse_edge = (edge[1], edge[0])
        bidirectional_edges.append(reverse_edge)
    return bidirectional_edges  

def get_antenna_mesh(antenna_PEC_mesh, vertex_types_antenna):
    antenna_PEC_edges = make_edges_bidirectional( antenna_PEC_mesh.edges.tolist())
    antenna_PEC_edges = add_bidirctional_edges_between_unconnected_meshes(antenna_PEC_mesh.vertices, antenna_PEC_edges)
    antenna_PEC_sphere_points = add_sphere_around_model_np(antenna_PEC_mesh.vertices, num_points = 256, radius = 300)
    antenna_avg_centroid = antenna_PEC_mesh.centroid
    antenna_PEC_sphere_points = [[x1 + x2, y1 + y2, z1 + z2] for (x1, y1, z1), (x2, y2, z2) in zip(antenna_PEC_sphere_points, [antenna_avg_centroid.tolist()] * len(antenna_PEC_sphere_points))]
    
    sphere_vertex_face_normals = np.zeros((len(antenna_PEC_sphere_points), 3))
    total_vertex_face_normals = np.vstack((antenna_PEC_mesh.vertex_normals, sphere_vertex_face_normals))

    edges, sphere_edges = add_one_directiona_edges_from_model_to_sphere(antenna_PEC_mesh.vertices,antenna_PEC_sphere_points, antenna_PEC_edges)


    total_edges = np.vstack((edges ,sphere_edges))
    points = np.vstack((antenna_PEC_mesh.vertices, antenna_PEC_sphere_points))
    total_edges = add_self_edges(points, total_edges)
    #plot_3d_points_edges(points , total_edges)
    
    duplicate_vertex_type  = np.tile(one_hot_encoding(model_stl_types['PEC']),(len(antenna_PEC_sphere_points),1)) 
    vertex_types_antenna.append(duplicate_vertex_type)
    points_type = np.concatenate(vertex_types_antenna)
    return {'vertcies':points, 'edges':total_edges,'vertex_types':points_type,  'vertex_normals':total_vertex_face_normals, 'faces':antenna_PEC_mesh.faces, 'sphere_edges':sphere_edges}


def get_sphere_edges(resampled_mesh_list, sphere_nodes, number_of_model_nodes):
    '''
    connects each sphere node to its neaerest neghibor in the submesh
    '''
    sphere_edges = []
    number_of_sphere_nodes = len(sphere_nodes) 
    sphere_node_indeces = [ number_of_model_nodes + idx for idx in range(number_of_sphere_nodes)]
    nn_submesh = NearestNeighbors(n_neighbors=2, algorithm='ball_tree')
    vertex_index_offset = 0

    for submesh in resampled_mesh_list:
        nn_submesh.fit(submesh.vertices)
        _, model_indices = nn_submesh.kneighbors(sphere_nodes)
        model_neighbors = [idx[0] for idx in model_indices]
        model_neighbors_indeces = [vertex_index + vertex_index_offset for vertex_index in model_neighbors]
        
        vertex_index_offset += len(submesh.vertices)
        # Add edges from model nodes to sphere nodes
        sphere_edges.extend(zip(model_neighbors_indeces, sphere_node_indeces))
        
        
    return sphere_edges



def convert_mesh_to_pyg(submesh, submesh_type):
    submesh_type = get_submesh_type(submesh_type)
    type_list = len(submesh.vertices) * [submesh_type]
    graph_dict = {'vertcies':np.array(submesh.vertices),
                'vertex_types':np.array(type_list),
                'faces':np.array(submesh.faces),
                'edges':submesh.edges,
                'vertex_normals':np.array(submesh.vertex_normals),
                'graph_identifier': submesh.identifier}
    graph = convert_graph_to_Pyg(graph_dict)
    return graph
    
    
def decompose_pyg_graph(graph, debug=False):
    """_summary_
        go over all node types and get there subgraphs then get the sphere subgraph and return them as a list in trimesh format
        make sure the number of nodes are the same after decompisition
    Args:
        graph (_type_): _description_
    """

    #unique_node_type_indexs = set(vertex_type_to_integer_mapping.values())
    subgraph_dict = {}
    for stl_type in model_stl_types.keys():
        node_type_index = model_stl_types[stl_type]
        # if sphere continue we assighn it sepretly
        if node_type_index == 0:
            continue
        subgrph_indexes = get_nodes_of_type( graph.node_type, node_type_index -1)
        subgraph = graph.subgraph(subgrph_indexes)
        # if the type is not in the data set
        if subgraph.pos.numel() == 0:
            continue
        subgraph_dict[stl_type] = subgraph
        if debug:
            plot_3d_points_edges(subgraph.pos , subgraph.edge_index.T.tolist())
    # sphere nodes typs are 0 vectores
    sphere_node_idx = get_sphere_node_idx(graph.node_type)
    subgraph = graph.subgraph(sphere_node_idx)
    subgraph_dict['sphere'] = subgraph
    if debug:
        plot_3d_points_edges(subgraph.pos , subgraph.edge_index.T.tolist())
    return subgraph_dict
  
def compose_pyg_graph(submesh_list, submesh_type_list, num_sphere_nodes):
    # takes the submesh list combines them and adds the sphere were each sphere is connected to its nearest neighbor in the submesh
    # also recomposes the decompesd graph by decompose_pyg_graph and reconstructs them as a pyg graph the sumesh list is to be without the sphere:
    node_list = []
    node_normals_list = []
    ant_node_normals = []
    face_list = []
    type_list = []
    ant_type_list = []
    antenna = trimesh.Trimesh(vertices=[], faces=[])
    graph = trimesh.Trimesh(vertices=[], faces=[])

    acumulater = 0
    for idx, (submesh, submesh_type) in enumerate(zip(submesh_list,submesh_type_list)):
        #submesh = submesh.subdivide()  # .subdivide_to_size(25) # #.subivide_loop
        # if (one_hot_to_int(submesh_type) != vertex_type_to_integer_mapping['FR4:subtrate']) and one_hot_to_int(submesh_type) != vertex_type_to_integer_mapping['Vacuum'] and one_hot_to_int(submesh_type) != vertex_type_to_integer_mapping['Polycarbonate']:
        #     submesh = submesh.subdivide()
        # if (one_hot_to_int(submesh_type) == vertex_type_to_integer_mapping['PEC']):
        #     submesh = submesh.subdivide()
        #     antenna += submesh
        #     ant_type_list.extend(len(submesh.vertices) * [submesh_type])    
        graph += submesh

            
        node_list.extend(submesh.vertices)
        type_list.extend(len(submesh.vertices) * [submesh_type])
        # node_normals_list.extend(submesh.vertex_normals)
        if len(type_list) != len(node_list):
            print("len(type_list) != len(node_list)")
    
    avg_centroid = graph.centroid
    node_list = node_list - avg_centroid
    # add the sphere: 
    sphere_points = add_sphere_around_model_np(graph.vertices, num_points = num_sphere_nodes, radius = 200)
    node_list = np.vstack((node_list,sphere_points))

    # sphere_points = [[x1 + x2, y1 + y2, z1 + z2] for (x1, y1, z1), (x2, y2, z2) in zip(sphere_points, [avg_centroid.tolist()] * len(sphere_points))]
    sphere_edges = get_sphere_edges(submesh_list, sphere_points, len(graph.vertices))
    sphere_vertex_face_normals = np.zeros((len(sphere_points), 3))
    node_normals_list = np.vstack((graph.vertex_normals, sphere_vertex_face_normals))
    sphere_vertex_type  = np.tile(one_hot_encoding(0),(len(sphere_points),1)) 
    type_list.extend(sphere_vertex_type)
    total_edges = np.vstack((faces_to_edges(graph.faces) ,sphere_edges))
    

    graph_mesh = {'vertcies':np.array(node_list),
                  'vertex_types':np.array(type_list),
                  'faces':np.array(graph.faces),
                  'edges':total_edges,
                  'vertex_normals':np.array(node_normals_list),
                  'graph_identifier': graph.identifier}
    antenna_graph_mesh = {}
    # antenna_graph_mesh = {'vertcies':antenna.vertices, 
    #                       'edges':np.array(antenna.edges),'vertex_types':np.array(ant_type_list),  'vertex_normals':antenna.vertex_normals, 'faces':antenna.faces }
    
    return graph_mesh, antenna_graph_mesh
    
    
    
def get_mesh_diopole_format(mesh_path:str, num_sphere_nodes, mesh_format:str = 'stp', Debug = False):
    """
    methode: collects all submeshes and combines them into one mesh in the diopole dataset format
    Args: mesh_path 
    Returns: combined mesh in trimesh format
    """
    submeshes_generator = get_mesh_generator(mesh_path)
    submesh_list, submesh_type_list = get_submesh_list(submeshes_generator)
    graph_mesh, antenna_graph_mesh = compose_pyg_graph(submesh_list, submesh_type_list, num_sphere_nodes)
    return graph_mesh, antenna_graph_mesh
       
       
       
       
def pyg_graphs_to_trimesh(graphs):
    """
    Converts a dictionary of PyTorch Geometric graphs to trimesh meshes.
    
    Args:
        graphs (dict): A dictionary where keys are graph names (str) and values are PyG DataBatch objects.

    Returns:
        dict: A dictionary where keys are graph names (str) and values are trimesh.Trimesh objects.
    """
    trimesh_meshes = {}
    for name, graph in graphs.items():
        # Extract vertices (pos) and faces
        vertices = graph.pos.numpy()  # Convert tensor to numpy
        faces = edges_to_faces(graph.edge_index.T.tolist())
        #faces = faces.numpy()  # Convert tensor to numpy

        # Create the trimesh object
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        # Store the trimesh object in the dictionary
        trimesh_meshes[name] = mesh

    return trimesh_meshes

def edges_to_faces(edges):
    """
    Given a list of edges, return a list of triangular faces.

    Args:
        edges (list of tuple): A list of edges where each edge is represented as a tuple (v1, v2).

    Returns:
        list of tuple: A list of triangular faces where each face is represented as a tuple (v1, v2, v3).
    """
    from collections import defaultdict

    # Create an adjacency list to store neighboring vertices for each vertex
    adjacency = defaultdict(set)
    for v1, v2 in edges:
        adjacency[v1].add(v2)
        adjacency[v2].add(v1)

    # Generate faces
    faces = set()
    for v1 in adjacency:
        for v2 in adjacency[v1]:
            for v3 in adjacency[v1].intersection(adjacency[v2]):
                # Ensure each face is added only once by sorting vertex indices
                face = tuple(sorted([v1, v2, v3]))
                faces.add(face)

    return list(faces)
 
def snap_to_grid(coordinate, grid_size):
    return torch.round(coordinate / grid_size) * grid_size

def snap_trimesh_to_grid(mesh, grid_size):
    """
    Snaps all vertices of a trimesh object to the nearest grid points.

    Args:
        mesh (trimesh.Trimesh): The input mesh.
        grid_size (float): The grid spacing for snapping.

    Returns:
        trimesh.Trimesh: The adjusted mesh with snapped vertices.
    """
    # Snap function for each coordinate
    def snap_to_grid(coordinate, grid_size):
        return np.round(coordinate / grid_size) * grid_size

    # Snap all vertices to the grid
    snapped_vertices = snap_to_grid(mesh.vertices, grid_size)

    # Create a new mesh with snapped vertices and the same faces
    snapped_mesh = trimesh.Trimesh(vertices=snapped_vertices, faces=mesh.faces)

    return snapped_mesh
  
  
def clip_coordinates(tensor, bbox):
    """
    Clips the x and z coordinates of a tensor to fit within a bounding box.

    Args:
        tensor (torch.Tensor): Tensor of shape (N, 3) containing x, y, and z coordinates.
        bbox (list or tuple): Bounding box as [min_x, min_z, max_x, max_z].

    Returns:
        torch.Tensor: Tensor with x and z coordinates clipped to the bounding box.
    """
    min_x, min_z, max_x, max_z = bbox

    # Extract x and z columns
    x = tensor[:, 0]
    z = tensor[:, 2]

    # Clip x and z to fit within the bounding box
    x_clipped = torch.clamp(x, min_x, max_x)
    z_clipped = torch.clamp(z, min_z, max_z)

    # Replace x and z in the original tensor
    clipped_tensor = tensor.clone()
    clipped_tensor[:, 0] = x_clipped
    clipped_tensor[:, 2] = z_clipped

    return clipped_tensor  
        
def compare_mesh_structure(mesh1, mesh2):
    '''
    if meshe positions and edge indexes are diffrent the meshes are diffrent so output fals
    ''' 
    # Check if the number of nodes and edges are the same
    if mesh1.pos.size() != mesh2.pos.size() or mesh1.edge_index.size() != mesh2.edge_index.size():
        return False
    
    # Compare node positions
    if not torch.allclose(mesh1.pos, mesh2.pos, atol=1e-01):
        return False
    
    # Sort the edge indices to ensure the order doesn't affect comparison
    # Sorting each edge and then the whole edge_index tensor
    edges1_sorted = torch.sort(mesh1.edge_index, dim=0).values
    edges1_sorted = torch.sort(edges1_sorted, dim=1).values
    
    edges2_sorted = torch.sort(mesh2.edge_index, dim=0).values
    edges2_sorted = torch.sort(edges2_sorted, dim=1).values
    
    # Compare the sorted edge indices
    if not torch.allclose(edges1_sorted, edges2_sorted):
        return False
    
    return True


def remove_nodes_from_graph(graph, nodes_to_exclude):
    """
    Removes the specified nodes from the graph and returns the resulting subgraph.

    Parameters:
    - graph: The PyTorch Geometric graph object.
    - nodes_to_exclude: A list or tensor of node indices to be excluded from the graph. node indexes

    Returns:
    - new_graph: A new PyTorch Geometric graph object with the specified nodes removed.
    """
    # Get all nodes in the graph
    all_nodes = torch.arange(graph.num_nodes)

    # Find nodes to keep by excluding the nodes in `nodes_to_exclude`
    nodes_to_keep = torch.tensor([node for node in all_nodes if node not in nodes_to_exclude])

    # Use the `subgraph` function to get the new graph
    new_graph = graph.subgraph(nodes_to_keep)
    
    return new_graph   
    

def get_mesh(mesh_path:str, mesh_format:str = 'stp', Debug = False):
    """
    methode: collects all submeshes and combines them into one mesh
    Args: mesh_path 
    Returns: combined mesh in trimesh format
    """
    submeshes_generator = get_mesh_generator(mesh_path)
    submesh_list, submesh_type_list = get_submesh_list(submeshes_generator)
    resampled_mesh_list, mesh_volume = resample_meshes_on_tangent_faces(submesh_list)
    combined_mesh = trimesh.Trimesh(vertices=[], faces=[])
    antenna_PEC_mesh = trimesh.Trimesh(vertices=[], faces=[])

    vertex_types_combined = []
    vertex_types_antenna = []
    for idx, submesh in enumerate(resampled_mesh_list):
        combined_mesh = combined_mesh + submesh
        duplicate_vertex_type = np.tile(submesh_type_list[idx],(submesh.vertices.shape[0],1)) 
        vertex_types_combined.append(duplicate_vertex_type)
        if (submesh_type_list[idx]== one_hot_encoding(9)).all() or (submesh_type_list[idx]== one_hot_encoding(10)).all():
            antenna_PEC_mesh = antenna_PEC_mesh + submesh
            vertex_types_antenna.append(duplicate_vertex_type)
    # get the antenna mesh:
    antenna_graph_mesh = get_antenna_mesh(antenna_PEC_mesh, vertex_types_antenna)

        
    # create complimentry mesh made out of air
    complementary_mesh = create_complimentry_mesh(mesh_volume, mesh_volume)
    duplicate_vertex_type  = np.tile(one_hot_encoding(0),(complementary_mesh.vertices.shape[0],1)) 
    vertex_types_combined.append(duplicate_vertex_type)
    full_mesh = complementary_mesh + combined_mesh
    faces = combined_mesh.faces
    vertex_face_normals = full_mesh.vertex_normals
    
    
    sphere_points = add_sphere_around_model_np(full_mesh.vertices, num_points = 256, radius = 600)
    avg_centroid = full_mesh.centroid
    sphere_points = [[x1 + x2, y1 + y2, z1 + z2] for (x1, y1, z1), (x2, y2, z2) in zip(sphere_points, [avg_centroid.tolist()] * len(sphere_points))]
    sphere_edges = get_sphere_edges(resampled_mesh_list, sphere_points, len(full_mesh.vertices))
    sphere_vertex_face_normals = np.zeros((len(sphere_points), 3))
    total_vertex_face_normals = np.vstack((vertex_face_normals, sphere_vertex_face_normals))
    sphere_vertex_type  = np.tile(one_hot_encoding(0),(len(sphere_points),1)) 
    vertex_types_combined.append(sphere_vertex_type)
    
    # post process:
    edges = full_mesh.edges.tolist()
    edges = make_edges_bidirectional(edges)
    edges = add_bidirctional_edges_between_unconnected_meshes(full_mesh.vertices,  edges)

    
    total_edges = np.vstack((edges ,sphere_edges))
    points = np.vstack((full_mesh.vertices, sphere_points))
    points_type = np.concatenate(vertex_types_combined)
    graph_mesh = {'vertcies':points, 'edges':total_edges,
                  'vertex_types':points_type,
                  'vertex_normals':total_vertex_face_normals,
                  'faces':faces,
                  'sphere_edges':sphere_edges}


    return graph_mesh, antenna_graph_mesh
    
    

def get_mesh_old(mesh_path:str, mesh_format:str = 'stp', Debug = False):
    """
    methode: collects all submeshes and combines them into one mesh
    Args: mesh_path 
    Returns: combined mesh in trimesh format
    """
    submeshes_generator = get_mesh_generator(mesh_path)
    submesh_list, submesh_type_list = get_submesh_list(submeshes_generator)
    resampled_mesh_list, mesh_volume = resample_meshes_on_tangent_faces(submesh_list)
    
    # set vertex types per vetrex:
    vertex_types_combined = []
    vertex_types_antenna = []
    vertex_face_normals = []
    antenna_PEC_mesh = trimesh.Trimesh(vertices=[], faces=[])
    combined_mesh = trimesh.Trimesh(vertices=[], faces=[])
    for idx, submesh in enumerate(resampled_mesh_list):
        combined_mesh = combined_mesh + submesh
        duplicate_vertex_type = np.tile(submesh_type_list[idx],(submesh.vertices.shape[0],1)) 
        vertex_types_combined.append(duplicate_vertex_type)
        if (submesh_type_list[idx]== one_hot_encoding(9)).all() or (submesh_type_list[idx]== one_hot_encoding(10)).all():
            antenna_PEC_mesh = antenna_PEC_mesh + submesh
            vertex_types_antenna.append(duplicate_vertex_type)
        
    # get the antenna mesh:
    antenna_graph_mesh = get_antenna_mesh(antenna_PEC_mesh, vertex_types_antenna)

    # create complimentry mesh made out of air
    complementary_mesh = create_complimentry_mesh(combined_mesh, mesh_volume)
    duplicate_vertex_type  = np.tile(one_hot_encoding(0),(complementary_mesh.vertices.shape[0],1)) 
    vertex_types_combined.append(duplicate_vertex_type)
    #plot_3d_points_edges(complementary_mesh.vertices , complementary_mesh.edges.tolist())
    
 
    full_mesh = complementary_mesh + combined_mesh
    faces = combined_mesh.faces
    vertex_face_normals = full_mesh.vertex_normals
    #plot_3d_points_edges(full_mesh.vertices , full_mesh.edges.tolist())
    edges = add_bidirctional_edges_between_unconnected_meshes(full_mesh.vertices,  full_mesh.edges.tolist())
    edges = make_edges_bidirectional(edges)
    # create sphere ponits
    sphere_points = add_sphere_around_model_np(full_mesh.vertices, num_points = 256, radius = 600)
    avg_centroid = full_mesh.centroid
    sphere_points = [[x1 + x2, y1 + y2, z1 + z2] for (x1, y1, z1), (x2, y2, z2) in zip(sphere_points, [avg_centroid.tolist()] * len(sphere_points))]
    duplicate_vertex_type  = np.tile(one_hot_encoding(0),(len(sphere_points),1)) 
    vertex_types_combined.append(duplicate_vertex_type)
    sphere_vertex_face_normals = np.zeros((len(sphere_points), 3))
    total_vertex_face_normals = np.vstack((vertex_face_normals, sphere_vertex_face_normals))
    
    edges, sphere_edges = add_one_directiona_edges_from_model_to_sphere(full_mesh.vertices,sphere_points, edges)
    total_edges = np.vstack((edges ,sphere_edges))
    points = np.vstack((full_mesh.vertices, sphere_points))
    total_edges = add_self_edges(points, total_edges)
    points_type = np.concatenate(vertex_types_combined)
    #plot_3d_points_edges(points, edges, sphere_edges)
    
    graph_mesh = {'vertcies':points, 'edges':total_edges, 'vertex_types':points_type, 'vertex_normals':total_vertex_face_normals, 'faces':faces}
    return graph_mesh, antenna_graph_mesh

def convert_graph_to_Pyg(graph):
    PyG_data = data.Data()
    PyG_data.x = [i for i in range(len(graph['vertcies']))]
    PyG_data.pos = torch.tensor(graph['vertcies'],dtype = torch.float32)
    PyG_data.node_normals = torch.tensor(graph['vertex_normals'],dtype = torch.float32)
    #PyG_data.faces = tri_mesh.faces
    if 'vertex_types' in graph.keys():
        PyG_data.node_type =  torch.tensor(graph['vertex_types'],dtype = torch.float32)
    PyG_data.faces = torch.tensor(graph['faces'],dtype = torch.int32)
    PyG_data.edge_index = torch.tensor(graph['edges'], dtype = torch.int32).T
    PyG_data.validate(raise_on_error=True)
    return PyG_data

def faces_to_edges(faces):
    edges = set()
    for face in faces:
        num_vertices = len(face)
        for i in range(num_vertices):
            # Create an edge with ordered vertices (smaller index first)
            edge = tuple(sorted((face[i], face[(i + 1) % num_vertices])))
            edges.add(edge)
    return list(edges)
            
# Function to generate the edge list
def tetrahedra_to_edges(tetrahedra):
    edges = set()
    for tetra in tetrahedra:
        # Extract the indices of the vertices of the tetrahedron
        v0, v1, v2, v3 = tetra
        # Generate the six edges (as tuples of vertex indices, sorted to avoid duplicates)
        edges.add(tuple(sorted((v0, v1))))
        edges.add(tuple(sorted((v0, v2))))
        edges.add(tuple(sorted((v0, v3))))
        edges.add(tuple(sorted((v1, v2))))
        edges.add(tuple(sorted((v1, v3))))
        edges.add(tuple(sorted((v2, v3))))
    # Convert the set to a list for the final edge list
    return list(edges)


def generate_volume_mesh_from_stp(file_path, min_edge_length = 0.01, max_edge_length=0.01):
    gmsh.initialize()

    # Merge the STEP file into the Gmsh model
    gmsh.model.occ.importShapes(file_path)
    gmsh.model.occ.synchronize()

    # Create a volume from imported shapes
    gmsh.model.occ.synchronize()

    # Define meshing parameters
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", min_edge_length)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", max_edge_length)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)

    # Generate 3D mesh
    gmsh.model.mesh.generate(3)

    # Extract the mesh data
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    tetra_tags, tetra_nodes = gmsh.model.mesh.getElementsByType(4)

    gmsh.finalize()

    points = np.array(node_coords).reshape(-1, 3)
    tetrahedra = np.array(tetra_nodes).reshape(-1, 4) - 1  # Convert to 0-based indexing

    return points, tetrahedra


def create_meshgrid_from_mesh(mesh):
    bounding_box = mesh.bounds
    x_min, y_min, z_min = bounding_box[0]
    x_max, y_max, z_max = bounding_box[1]
    # Create 64 points along each axis based on the bounding box
    x = np.linspace(x_min, x_max, 64)
    y = np.linspace(y_min , y_max, 16)
    z = np.linspace(z_min, z_max, 64)

    # Create the meshgrid
    X,Y,Z = np.meshgrid(x, y, z, indexing='ij')
    meshgrid = [X,Y,Z ]
    # Reshape the meshgrid arrays to create a list of 3D points
    points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    return points

def create_and_align_meshgrid(full_mesh, ant_mesh):
    # Calculate the current center of the plane mesh
    bounding_box = full_mesh.bounds
    x_min, y_min, z_min = bounding_box[0]
    x_max, y_max, z_max = bounding_box[1]
    
    ant_center = ant_mesh.centroid
    resolution = ( y_max - y_min) / 15
    
    meshgrid =  create_meshgrid_from_mesh(full_mesh)
    #the number 3 needs  to be updated to lower bounds of ant_center%res
    distance_between_centers = ant_center[1] - (4*resolution + y_min) #y_mid - ant_center[1]
    meshgrid[ :, 1] += distance_between_centers # Translate along y-axis
    
    return meshgrid


def extrude_mesh(mesh, extrude_distance = 0.1):
    '''
    accepts 2d trimesh and outputs is extrudtion
    '''
    extruded_mesh = trimesh.Trimesh(vertices=[], faces=[])
    # get the outline of the mesh, move it to 2D, save the transform                                
    on_plane, to_3D = mesh.outline().to_planar()
    try:
        for extruded_plane in on_plane.extrude(extrude_distance):
            # extrude the outline into a solid                                                              
            extruded_mesh += extruded_plane.to_mesh().apply_transform(to_3D)
    except:
        extruded_plane = on_plane.extrude(extrude_distance)
        extruded_mesh += extruded_plane.to_mesh().apply_transform(to_3D)
    return extruded_mesh
    

   




def combine_graphs( graph1: Data, graph2: Data) -> Data:
    # Combine node features
    # combined_x = torch.cat([graph1.x, graph2.x], dim=0)

    # Combine node positions
    combined_pos = torch.cat([graph1.pos, graph2.pos], dim=0)

    # Combine node normals
    combined_node_normals = torch.cat([graph1.node_normals, graph2.node_normals], dim=0)

    # Combine node types
    combined_node_type = torch.cat([graph1.node_type, graph2.node_type], dim=0)

    # Combine faces (adjusting indices for the second graph)
    combined_faces = torch.cat([graph1.faces, graph2.faces + graph1.pos.size(0)], dim=0)

    # Combine edge indices (adjusting indices for the second graph)
    combined_edge_index = torch.cat([graph1.edge_index, graph2.edge_index + graph1.pos.size(0)], dim=1)

    # Combine batch indices
    combined_batch = torch.cat([graph1.batch, graph2.batch + graph1.batch.max() + 1], dim=0)

    # Combine ptr
    #combined_ptr = torch.cat([graph1.ptr, graph2.ptr[1:] + graph1.ptr[-1]], dim=0)

    # Create a new DataBatch object
    combined_graph = Data(
        #x=combined_x,
        pos=combined_pos,
        node_normals=combined_node_normals,
        node_type=combined_node_type,
        faces=combined_faces,
        edge_index=combined_edge_index,
        batch=combined_batch#,
        #ptr=combined_ptr
    )
    return combined_graph




def create_icosphere(subdivisions=0, radius=1.0, device=torch.device("cpu")):
    """
    Creates an icosphere mesh using PyTorch.

    Parameters:
        subdivisions (int): Number of subdivision iterations.
        radius (float): Radius of the sphere.
        device (torch.device): The device on which tensors are allocated.

    Returns:
        vertices (torch.Tensor): Tensor of vertex coordinates with shape (N, 3).
        faces (torch.Tensor): Tensor of face indices with shape (M, 3).
    """
    # Compute golden ratio using torch operations
    t = (1.0 + torch.sqrt(torch.tensor(5.0, device=device))) / 2.0

    # Define the 12 vertices of an icosahedron
    vertices = torch.tensor([
        [-1,  t,  0],
        [ 1,  t,  0],
        [-1, -t,  0],
        [ 1, -t,  0],
        [ 0, -1,  t],
        [ 0,  1,  t],
        [ 0, -1, -t],
        [ 0,  1, -t],
        [ t,  0, -1],
        [ t,  0,  1],
        [-t,  0, -1],
        [-t,  0,  1],
    ], dtype=torch.float64, device=device)

    # Normalize vertices to lie on the sphere of given radius
    vertices = vertices / torch.norm(vertices, dim=1, keepdim=True) * radius

    # Define the 20 triangular faces of the icosahedron
    faces = torch.tensor([
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ], dtype=torch.int64, device=device)

    # Perform subdivisions
    for _ in range(subdivisions):
        new_faces = []
        # Use a cache to store computed midpoints to avoid duplicates
        midpoint_cache = {}
        # Convert vertices tensor to a list for dynamic appending
        vertices_list = vertices.tolist()

        def get_midpoint(i1, i2):
            # Sort the indices to have a unique key for each edge
            key = tuple(sorted((i1, i2)))
            if key in midpoint_cache:
                return midpoint_cache[key]
            else:
                # Convert the vertices back to torch tensors for arithmetic
                v1 = torch.tensor(vertices_list[i1], dtype=torch.float64, device=device)
                v2 = torch.tensor(vertices_list[i2], dtype=torch.float64, device=device)
                mid = (v1 + v2) / 2.0
                # Normalize the midpoint to lie on the sphere
                mid = mid / torch.norm(mid) * radius
                vertices_list.append(mid.tolist())
                index = len(vertices_list) - 1
                midpoint_cache[key] = index
                return index

        # Subdivide each triangle into four smaller triangles
        for tri in faces.tolist():
            i0, i1, i2 = tri
            a = get_midpoint(i0, i1)
            b = get_midpoint(i1, i2)
            c = get_midpoint(i2, i0)
            new_faces.append([i0, a, c])
            new_faces.append([i1, b, a])
            new_faces.append([i2, c, b])
            new_faces.append([a, b, c])

        # Update vertices and faces for next subdivision iteration
        vertices = torch.tensor(vertices_list, dtype=torch.float64, device=device)
        faces = torch.tensor(new_faces, dtype=torch.int64, device=device)

    return vertices, faces

def get_icosphere(num_sphere_vertices, radius=200.0, device=torch.device("cpu")):
    """
    Creates an icosphere mesh with at least the specified number of vertices.

    The vertex count for an icosphere is given by: V = 10 * 4^n + 2.
    This function computes the minimum number of subdivisions required so that V >= target_vertices.

    Parameters:
        target_vertices (int): Minimum desired number of vertices.
        radius (float): Radius of the sphere.
        device (torch.device): The device on which tensors are allocated.

    Returns:
        vertices (torch.Tensor): Tensor of vertex coordinates.
        faces (torch.Tensor): Tensor of face indices.
    """
    # Determine required subdivisions
    if num_sphere_vertices <= 12:
        subdivisions = 0
    else:
        subdivisions = math.ceil(math.log((num_sphere_vertices - 2) / 10, 4))
    
    vertices, faces = create_icosphere(subdivisions, radius, device)
    # print(f"Using {subdivisions} subdivisions to achieve {vertices.shape[0]} vertices.")
    return vertices, faces








    

def display_points_and_normals(graph, arrow_length=0.1):
    """
    Visualizes 3D points and their corresponding normals.

    Parameters:
    - graph_dict (dict): A dictionary that contains the 3D data.
    - key (str): The key in graph_dict corresponding to the object holding
                 the attributes 'pos' and 'normals'. Default is 'Antenna_PEC_STEP'.
    - arrow_length (float): The scaling factor for the normal arrows.
    """
    # Retrieve points and normals from the provided key
    data = graph
    points = data.pos       # Expected shape: [N, 3]
    normals = data.node_normals  # Expected shape: [N, 3]

    # Convert to NumPy arrays if they are PyTorch tensors.
    if torch.is_tensor(points):
        points = points.detach().cpu().numpy()
    if torch.is_tensor(normals):
        normals = normals.detach().cpu().numpy()

    # Create a 3D plot
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the points as blue dots.
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue', label='Points')

    # Plot the normals using a quiver plot (red arrows).
    ax.quiver(points[:, 0], points[:, 1], points[:, 2],
              normals[:, 0], normals[:, 1], normals[:, 2],
              length=arrow_length, normalize=True, color='red', label='Normals')

    # Label axes and set a title.
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Points with Normals')

    # Add a legend for clarity.
    ax.legend()

    # Show the plot.
    plt.show()
