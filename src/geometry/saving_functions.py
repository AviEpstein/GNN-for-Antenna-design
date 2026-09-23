

import pickle
import os
import torch
import trimesh
from src.geometry.mesh_functions import edges_to_faces



def save_matrix_and_dict(matrix, reflectors_dict, path):
    """Saves the matrix and reflectors dictionary to a specified path.
    
    Args:
        matrix (torch.Tensor): The matrix to be saved.
        reflectors_dict (dict): The reflectors dictionary to be saved.
        path (str): The file path where the data will be saved.

        matrix (torch.Tensor): The matrix to be saved.
        reflectors_dict (dict): The reflectors dictionary to be saved.
        path (str): The file path where the data will be saved.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data_to_save = {
        'matrix': matrix.cpu().numpy().tolist() if isinstance(matrix, torch.Tensor) else matrix,
        'reflectors_dict': {k: [v_i.item() if torch.is_tensor(v_i) else v_i for v_i in v] if isinstance(v, list) else v for k, v in reflectors_dict.items()}
    }
    with open(path, 'wb') as f:
        pickle.dump(data_to_save, f)
    # print(f"Saved matrix and reflectors dictionary to {path}")



def save_graph_dict_as_stl( graph_dict, output_dir):
    """
    Saves each element in a graph dictionary as a trimesh STL file.
    
    Args:
        graph_dict (dict): Dictionary of PyG Data objects where each key is a component name
        output_dir (str): Directory where STL files will be saved
    """
    os.makedirs(output_dir, exist_ok=True)
    
    ant_pec_and_feed_pec_mesh_list = []
    ant_pec_and_reflector_mesh_list = []
    
    for component_name, component in graph_dict.items():
        # Skip if this is an empty component
        if not hasattr(component, 'pos') or len(component.pos) == 0 or component_name == 'sphere':
            # print(f"Skipping empty component: {component_name}")
            continue
            
        # Get vertices as numpy array
        vertices = component.pos.detach().cpu().numpy()
        
        # Get faces - either from existing faces attribute or by computing from edges
        if hasattr(component, 'faces') and component.faces is not None and len(component.faces) > 0:
            # Use existing faces if available
            faces = component.faces.cpu().numpy()
        else:
            # Convert edge_index to list of edge tuples
            edges = component.edge_index.T.cpu().numpy().tolist()
            # Use edges_to_faces to generate faces
            faces = edges_to_faces(edges)
        
        # Create trimesh object
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        

        # Save as STL
        if component_name ==  'Antenna_PEC_STEP':
            component_name = 'Antenna_PEC_STEP'
            ant_pec_and_feed_pec_mesh_list.append(mesh)
            ant_pec_and_reflector_mesh_list.append(mesh)
        if component_name ==  'Antenna_Feed_PEC_STEP':
            component_name = 'Antenna_Feed_PEC_STEP'
            ant_pec_and_feed_pec_mesh_list.append(mesh)
        if component_name ==  'Antenna_Feed_STEP':
            component_name = 'Feed'
        if component_name ==  'Env_FR4_STEP':
            component_name = 'Dielectric'
        if component_name ==  'PEC_ground':
            component_name = 'PEC_ground'
        if component_name ==  'PEC_Reflector':
            component_name = 'PEC_Reflector'
            ant_pec_and_feed_pec_mesh_list.append(mesh)
            ant_pec_and_reflector_mesh_list.append(mesh)

        stl_path = os.path.join(output_dir, f"{component_name}.stl")
        mesh.export(stl_path)
        # print(f"Saved {component_name} to {stl_path}")
        

    combined_stl_path = os.path.join(output_dir, 'PEC_pixel.stl')
    if len(ant_pec_and_feed_pec_mesh_list) > 0:
        ant_pec_and_feed_pec_mesh = ant_pec_and_feed_pec_mesh_list[0]
        for mesh in ant_pec_and_feed_pec_mesh_list[1:]:
            ant_pec_and_feed_pec_mesh += mesh
    ant_pec_and_feed_pec_mesh.export(combined_stl_path) 
    
    # this was added so CST wil load them as one file
    combined_stl_path_reflector = os.path.join(output_dir, 'Antenna_PEC_STEP.stl')
    if len(ant_pec_and_reflector_mesh_list) > 0:
        ant_pec_and_reflector_mesh = ant_pec_and_reflector_mesh_list[0]
        for mesh in ant_pec_and_reflector_mesh_list[1:]:
            ant_pec_and_reflector_mesh += mesh
    ant_pec_and_reflector_mesh.export(combined_stl_path_reflector)
    
    # print(f"Combined STL saved as {combined_stl_path}")
        
