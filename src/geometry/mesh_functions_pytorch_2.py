import torch
from src.geometry.mesh_functions import plot_3d_points_edges
from torch_geometric.data import Data
from src.geometry.mesh_functions import one_hot_encoding, model_stl_types, edges_to_faces

def check_ant_validity(ant_parameters, model_parameters):
    assert int(model_parameters["type"]) in [3, 5], 'model_parameters["type"] must be either 3 or 5'
    if int(model_parameters["type"]) == 3:
        Sz = (model_parameters['length'] * model_parameters['adz'] * model_parameters['arz'] / 2 - ant_parameters['w'] / 2
              - model_parameters['feed_length'] / 2)
        Sy = model_parameters['height'] * model_parameters['ady'] * model_parameters['ary'] - ant_parameters['w']
    else:
        Sz = model_parameters['Sz'] - ant_parameters['w'] / 2 - model_parameters['feed_length'] / 2
        Sy = model_parameters['Sy'] - ant_parameters['w']
    wings = ['w1','w2','q1','q2']
    for key in ant_parameters:
        if ant_parameters[key] < 0: return 0
        if key != 'w' and ant_parameters[key] > 1: return 0
        
    for key, value in ant_parameters.items():
        if key != 'w':  # If the key is not 'w'
            if not (0 <= value <= 1):  # Check if the value is between 0 and 1
                return False

    for wing in wings:
        if (ant_parameters[f'{wing}z3'] > ant_parameters[f'{wing}z1'] > ant_parameters[f'{wing}z2'] and
            ant_parameters[f'{wing}y1'] > ant_parameters[f'{wing}y2']):
            return 0
        if (ant_parameters[f'{wing}z2'] > ant_parameters[f'{wing}z1'] > ant_parameters[f'{wing}z3'] and
            ant_parameters[f'{wing}y1'] > ant_parameters[f'{wing}y2']):
            return 0
        if (ant_parameters[f'{wing}z1'] > ant_parameters[f'{wing}z3'] > ant_parameters[f'{wing}z2'] and
            ant_parameters[f'{wing}y3'] > ant_parameters[f'{wing}y1'] > ant_parameters[f'{wing}y2']):
            return 0
        if (ant_parameters[f'{wing}z2'] > ant_parameters[f'{wing}z3'] > ant_parameters[f'{wing}z1'] and
            ant_parameters[f'{wing}y3'] > ant_parameters[f'{wing}y1'] > ant_parameters[f'{wing}y2']):
            return 0
        if (ant_parameters[f'{wing}z2'] > ant_parameters[f'{wing}z3'] > ant_parameters[f'{wing}z1'] and
            ant_parameters[f'{wing}y2'] > ant_parameters[f'{wing}y1'] > ant_parameters[f'{wing}y3']):
            return 0
        if (ant_parameters[f'{wing}z1'] > ant_parameters[f'{wing}z3'] > ant_parameters[f'{wing}z2'] and
            ant_parameters[f'{wing}y2'] > ant_parameters[f'{wing}y1'] > ant_parameters[f'{wing}y3']):
            return 0
        if torch.abs(torch.tensor(ant_parameters[f'{wing}z2'] - ant_parameters[f'{wing}z1'])) < ant_parameters['w']/Sz: return 0
        if torch.abs(torch.tensor(ant_parameters[f'{wing}z1'] - ant_parameters[f'{wing}z3'])) < ant_parameters['w']/Sz: return 0
        if torch.abs(torch.tensor(ant_parameters[f'{wing}z2'] - ant_parameters[f'{wing}z3'])) < ant_parameters['w']/Sz: return 0
        if ant_parameters[f'{wing}y1'] < ant_parameters['w'] / Sy: return 0
        if ant_parameters[f'{wing}y2'] < ant_parameters['w'] / Sy: return 0
        if torch.abs(torch.tensor(ant_parameters[f'{wing}y2'] - ant_parameters[f'{wing}y1'])) < ant_parameters['w']/Sy: return 0
        if torch.abs(torch.tensor(ant_parameters[f'{wing}y1'] - ant_parameters[f'{wing}y3'])) < ant_parameters['w']/Sy: return 0
        if torch.abs(torch.tensor(ant_parameters[f'{wing}y2'] - ant_parameters[f'{wing}y3'])) < ant_parameters['w']/Sy: return 0
    if (Sz * ant_parameters[f'q1z3'] - ant_parameters['w']/2 <= 5
        and (ant_parameters[f'q1y3'] < ant_parameters['fx'] < ant_parameters[f'q1y2'] or
            ant_parameters[f'q1y2'] < ant_parameters['fx'] < ant_parameters[f'q1y3'])): return 0
    if (Sz * ant_parameters[f'w1z3'] - ant_parameters['w']/2 <= 5
        and (ant_parameters[f'w1y3'] < ant_parameters['fx'] < ant_parameters[f'w1y2'] or
            ant_parameters[f'w1y2'] < ant_parameters['fx'] < ant_parameters[f'w1y3'])): return 0
    wings = ['w1', 'w2','w3', 'q1', 'q2','q3']
    for wing in wings:
        if torch.abs(torch.tensor(ant_parameters[f'{wing}z0'] - ant_parameters[f'{wing}z1'])) <= ant_parameters['w'] / Sz: return 0
    if torch.min(torch.tensor([ant_parameters[f'q3z0'],ant_parameters[f'w3z0']])) > 0.2: return 0
    return 1

def ant_to_dict_representation_pytorch(ant: torch.Tensor, example_keys: list):
    """
    Convert a batch of antenna tensors to a dictionary representation while maintaining differentiability.
    
    Args:
        ant (torch.Tensor): A 2D tensor with shape (batch, features).
        example_keys (list): A list of keys corresponding to antenna parameters.
        
    Returns:
        dict: A dictionary where each key corresponds to a parameter and the value is a tensor of that parameter for the batch.
    """
    #assert ant.ndim == 2, 'Antenna tensor should have 2 dimensions (batch, features)'
    assert len(example_keys) == ant.shape[0], (
        f"Number of features in tensor ({ant.size(1)}) does not match the number of keys ({len(example_keys)})."
    )
    
    # Create a dictionary where each key maps to a column (feature) in the tensor
    ant_dict = {key: ant[i] for i, key in enumerate(example_keys)}
    
    return ant_dict



def ant_abs2rel_pytorch(ant_parameters_abs: dict, model_parameters: dict):
    """
    Converts absolute antenna parameters to relative parameters.
    Ensures compatibility with backpropagation by using PyTorch tensor operations.

    Args:
        ant_parameters_abs (dict): Dictionary of absolute antenna parameters as PyTorch tensors.
        model_parameters (dict): Dictionary of model parameters as PyTorch tensors.
        
    Returns:
        dict: Dictionary of relative antenna parameters.
    """
    # Create a new dictionary for relative parameters
    # Calculate scaling factors using PyTorch tensors
    ant_parameters_rel = {}
    if model_parameters['type'] == 3.0:
        Sz = (
            model_parameters['length'] * model_parameters['adz'] * model_parameters['arz'] / 2
            - ant_parameters_abs['w'] / 2 
            - model_parameters['feed_length'] / 2
        )
        Sy = (
            model_parameters['height'] * model_parameters['ady'] * model_parameters['ary']
            - ant_parameters_abs['w']
        )

        # Compute relative parameters
        for key, value in ant_parameters_abs.items():
            if len(key) == 4:  # Check if key length matches
                if key[2] == 'z':
                    ant_parameters_rel[key] = (value  / Sz)
                elif key[2] == 'y':
                    ant_parameters_rel[key] = (value / Sy)
            elif key == 'fx':
                ant_parameters_rel[key] = (value  / Sy)
                
        ant_parameters_rel['w'] = ant_parameters_abs['w']
    
    if model_parameters['type'] == 5.0:
        Sz = model_parameters['Sz'] - ant_parameters_abs['w'] / 2 - model_parameters['feed_length'] / 2
        Sy = model_parameters['Sy'] - ant_parameters_abs['w']
        for key, value in ant_parameters_abs.items():
            if len(key) == 4:
                if key[2] == 'z':
                    ant_parameters_rel[key] = torch.round(value / Sz, decimals=2)
                if key[2] == 'y':
                    ant_parameters_rel[key] = torch.round(value / Sy, decimals=2)
            if key == 'fx':
                ant_parameters_rel[key] = torch.round(value / Sy, decimals=2)
        ant_parameters_rel['w'] = ant_parameters_abs['w']
        
    if model_parameters['type'] == 6.0:
        ant_parameters_rel['L4_rel'] = ant_parameters_abs['L4_rel'] / ant_parameters_abs['L2_rel']
        ant_parameters_rel['L1_rel'] = ant_parameters_abs['L1_rel'] / model_parameters['LG_y']
        ant_parameters_rel['L2_rel'] = ant_parameters_abs['L2_rel'] / (
                    model_parameters['A_z'] - ant_parameters_abs['W2'])
        ant_parameters_rel['L3_rel'] = ant_parameters_abs['L3_rel'] / (
            
        model_parameters['LG_y'] - ant_parameters_abs['W1'] * 3 - - ant_parameters_abs['gap'])
    return ant_parameters_rel



def create_antenna_points_list_pytorch(example_vector, key_to_index_example):
    
    Sz = (example_vector[key_to_index_example['length']] * example_vector[key_to_index_example['adz']] * example_vector[key_to_index_example['arz']] / 2 
        - example_vector[key_to_index_example['w']] / 2 - example_vector[key_to_index_example['feed_length']] / 2)
    Sy = example_vector[key_to_index_example['height']] * example_vector[key_to_index_example['ady']] * example_vector[key_to_index_example['ary']] - example_vector[key_to_index_example['w']]

    
    sign1 = 1.0
    wing = 'w1'
    ant_PEC_points = []

    z1 = sign1 * torch.stack([Sz * example_vector[key_to_index_example[f'{wing}z0']], Sz * example_vector[key_to_index_example[f'{wing}z{1}']], Sz * example_vector[key_to_index_example[f'{wing}z{1}']],
                        Sz * example_vector[key_to_index_example[f'{wing}z{2}']], Sz * example_vector[key_to_index_example[f'{wing}z{2}']],
                        Sz * example_vector[key_to_index_example[f'{wing}z{3}']], Sz * example_vector[key_to_index_example[f'{wing}z{3}']]
                    ])
    a = Sy * example_vector[key_to_index_example[f'{wing}y{2}']]
    b = Sy * example_vector[key_to_index_example[f'{wing}y{2}']]
    c = Sy * example_vector[key_to_index_example[f'{wing}y{3}']]
    #d = Sy * example_vector[key_to_index_example[f'{wing}y{3}']]
    e = Sy * example_vector[key_to_index_example[f'{wing}y{1}']]
    f = Sy* example_vector[key_to_index_example['w1y1']]
    # dummy_loss = f.sum()
    # dummy_loss.backward()
    # optimizer.step()

    # Recompute f and dummy_loss for the next step
    # f = 3 * example_vector[key_to_index_example['w1y1']]
    # dummy_loss = f.sum()
    # optimizer.zero_grad()
    # dummy_loss.backward()
    # optimizer.step()
    y1 = torch.stack([torch.zeros_like(example_vector[key_to_index_example['w1y1']]), torch.zeros_like(example_vector[key_to_index_example['w1y1']]), e, f,
            a, b,
            c#, d
            ])

    #y = torch.stack(y[:-1])  # Remove the last element and stack tensors.
    #z = torch.stack(z)
    #y1 = y1[:-1]
    #z1 = sign1 * z1
    wing_points1 = torch.stack([y1, z1], dim=1)  # Shape: [num_points, 2]
    ant_PEC_points.append(wing_points1)
    return ant_PEC_points




def create_points_list_model3_pytorch(model_parameters, ant_parameters):
    wings = ['w1', 'w2', 'q1', 'q2']
    Sz = (model_parameters['length'] * model_parameters['adz'] * model_parameters['arz'] / 2 
          - ant_parameters['w'] / 2 - model_parameters['feed_length'] / 2)
    Sy = model_parameters['height'] * model_parameters['ady'] * model_parameters['ary'] - ant_parameters['w']

    ant_PEC_points = []
    
    for wing in wings:
        # Use a tensor-based approach for the sign
        if wing == 'q1' or wing == 'q2':
            sign1 = torch.tensor([-1.0], dtype=Sz.dtype, device=Sz.device)
        else:
            sign1 = torch.tensor([1.0], dtype=Sz.dtype, device=Sz.device)

        z1 = torch.stack([Sz * ant_parameters[f'{wing}z0'], Sz * ant_parameters[f'{wing}z{1}'], Sz * ant_parameters[f'{wing}z{1}'],
                        Sz * ant_parameters[f'{wing}z{2}'], Sz * ant_parameters[f'{wing}z{2}'],
                        Sz * ant_parameters[f'{wing}z{3}'], Sz * ant_parameters[f'{wing}z{3}']
                        ])
        y1 = torch.stack([torch.tensor([0.0], dtype=Sz.dtype, device=Sz.device), torch.tensor([0.0], dtype=Sz.dtype, device=Sz.device),
                        Sy * ant_parameters[f'{wing}y{1}'], Sy * ant_parameters[f'{wing}y{1}'],
                        Sy * ant_parameters[f'{wing}y{2}'], Sy * ant_parameters[f'{wing}y{2}'],
                        Sy * ant_parameters[f'{wing}y{3}'], Sy * ant_parameters[f'{wing}y{3}']
                        ])   
        
        # Remove the last element and stack tensors.
        y1 = y1[:-1]
        z1 = sign1 * z1
        
        if wing == 'q1' or wing == 'q2':
            z1 = z1 - model_parameters['feed_length']
            
        wing_points1 = torch.stack([y1, z1], dim=1)  # Shape: [num_points, 2]
        ant_PEC_points.append(wing_points1)

    wings = ['w3', 'q3']
    for wing in wings:
        # Use the same tensor-based approach for the sign
        if wing == 'q3':
            sign = torch.tensor(-1.0, dtype=Sz.dtype, device=Sz.device)
        else:
            sign = torch.tensor(1.0, dtype=Sz.dtype, device=Sz.device)
            
        z = torch.stack([Sz * ant_parameters[f'{wing}z0'], Sz * ant_parameters[f'{wing}z1'], Sz * ant_parameters[f'{wing}z1']])
        y = torch.stack([Sy * ant_parameters['fx'], Sy * ant_parameters['fx'], Sy * ant_parameters[f'{wing}y1']])
        z = sign * z
        
        if wing == 'q3':
            z = z - model_parameters['feed_length']
        
        wing_points = torch.stack([y, z], dim=1)  # Shape: [num_points, 2]
        
        # if the last point is duplicated remove it
        if wing_points[-1].all() ==  wing_points[-2].all():
            wing_points = wing_points[:-1]
        
        ant_PEC_points.append(wing_points)
    
    feed_PEC_points = [
        torch.stack([
            torch.stack([Sy * ant_parameters['fx'], torch.tensor([-12.0], dtype=Sz.dtype, device=Sz.device)]),
            torch.stack([Sy * ant_parameters['fx'], -model_parameters['feed_length']])
        ], dim=0),
        torch.stack([
            torch.stack([Sy * ant_parameters['fx'], torch.tensor([10.0], dtype=Sz.dtype, device=Sz.device)]),
            torch.stack([Sy * ant_parameters['fx'], torch.tensor([0.0], dtype=Sz.dtype, device=Sz.device)])
        ], dim=0),
    ]
    
    feed_points = [torch.stack([
        torch.stack([Sy * ant_parameters['fx'], torch.tensor([0.0], dtype=Sz.dtype, device=Sz.device)]),
        torch.stack([Sy * ant_parameters['fx'], -model_parameters['feed_length']])
    ], dim=0)]
    
    return feed_points, feed_PEC_points, ant_PEC_points



def create_points_list_model5_pytorch(model_parameters, ant_parameters):
        
        wings = ['w1', 'w2']
        Sz = model_parameters['Sz'] - ant_parameters['w'] / 2 - model_parameters['feed_length'] / 2
        Sy = model_parameters['Sy'] - ant_parameters['w']
        
        feed_PEC_points = [[[Sy * ant_parameters['fx'], -torch.tensor([10.0], dtype=Sz.dtype, device=Sz.device) - model_parameters['feed_length']],
                            [Sy * ant_parameters['fx'], -model_parameters['feed_length']]],
                           [[Sy * ant_parameters['fx'], torch.tensor([10.0], dtype=Sz.dtype, device=Sz.device)], [Sy * ant_parameters['fx'], torch.tensor([0.0], dtype=Sz.dtype, device=Sz.device)]]]
        ant_PEC_points = []
        for wing in wings:
            sign = torch.tensor([1.0], dtype=Sz.dtype, device=Sz.device)
            z = [Sz * ant_parameters[f'{wing}z0']]
            y = [torch.tensor([0.0], dtype=Sz.dtype, device=Sz.device), torch.tensor([0.0], dtype=Sz.dtype, device=Sz.device)]
            for i1 in range(3):
                z.append(Sz * ant_parameters[f'{wing}z{i1 + 1:d}'])
                z.append(Sz * ant_parameters[f'{wing}z{i1 + 1:d}'])
                y.append(Sy * ant_parameters[f'{wing}y{i1 + 1:d}'])
                y.append(Sy * ant_parameters[f'{wing}y{i1 + 1:d}'])
            y.pop()
            wing_points = [[y[ii], sign * torch.tensor(z[ii])] for [ii, temp] in enumerate(y)]
            ant_PEC_points.append(wing_points)
        wings = ['w3']
        for wing in wings:
            sign = 1
            z = [Sz * ant_parameters[f'{wing}z0']]
            y = [Sy * ant_parameters['fx'], Sy * ant_parameters['fx']]
            z.append(Sz * ant_parameters[f'{wing}z{1:d}'])
            z.append(Sz * ant_parameters[f'{wing}z{1:d}'])
            y.append(Sy * ant_parameters[f'{wing}y{1:d}'])
            # wing_points = [[y[ii], sign * np.array(z[ii])] for [ii, temp] in enumerate(y)] # TODO:
            wing_points = [[y[ii], sign * torch.tensor(z[ii])] for [ii, temp] in enumerate(y)]
            if wing_points[-1] ==  wing_points[-2]:
                wing_points = wing_points[:-1]
            ant_PEC_points.append(wing_points)
        
        # feed_points = [[Sy * ant_parameters['fx'], model_parameters['feed_length'] / 2], # TODO:
        #                     [Sy * ant_parameters['fx'], -model_parameters['feed_length'] / 2]]
        feed_points = [[[Sy * ant_parameters['fx'], torch.tensor([0.0], dtype=Sz.dtype, device=Sz.device)],  
                       [Sy * ant_parameters['fx'], -model_parameters['feed_length']]]]
        
        return feed_points, feed_PEC_points, ant_PEC_points
    










def create_buffered_closed_path(points, buffer=0.5):
    """
    Create a closed path with a buffer around the input points.

    Args:
        points (list or torch.Tensor): Input points of shape (n, 2). If a list is provided,
                                       it should contain tensors or numerical values.
        buffer (float): Buffer distance.

    Returns:
        torch.Tensor: Buffered closed path points.
    """
    # Ensure all points are tensors and stack them
    #buffer = torch.tensor(buffer, dtype=torch.float32, requires_grad=False)
    
    if isinstance(points, list):
        stacked_points = []
        for p in points:
            stacked_point = torch.stack((p))
            stacked_points.append(stacked_point)
        points = torch.stack([p for p in stacked_points])
    else:
        points = points.to(torch.float32)

    n = points.shape[0]

    # Compute segment directions
    directions = points[1:] - points[:-1]
    directions = directions / (directions.norm(dim=1, keepdim=True) + 1e-6)  # Normalize, avoid division by zero

    # Compute normals (perpendicular vectors)
    normals = torch.zeros_like(directions)
    normals[:, 0] = normals[:, 0] - directions[:, 1]  # Perpendicular x
    normals[:, 1] = normals[:, 1] + directions[:, 0]  # Perpendicular y
    normals = normals#.detach().clone()
    # Offset points
    offset_points_positive = []
    offset_points_negative = []

    # First point
    offset_points_positive.append(points[0] + buffer * normals[0])
    offset_points_negative.append(points[0] - buffer * normals[0])

    # Middle points
    #m = torch.nn.Hardtanh(-1, 1)
    for i in range(1, n - 1):
        # Average normals for smooth transitions
        avg_normal = (normals[i - 1] + normals[i]) / 2
        avg_normal = avg_normal / avg_normal.norm()  # Normalize
        avg_normal = avg_normal/abs(avg_normal)
        offset_points_positive.append(points[i] + buffer * avg_normal)
        offset_points_negative.append(points[i] - buffer * avg_normal)

    # Last point
    last_point_offset_positive = points[-1] + buffer * normals[-1]
    last_point_offset_negative = points[-1] - buffer * normals[-1]
    offset_points_positive.append(last_point_offset_positive)
    offset_points_negative.append(last_point_offset_negative)

    # Combine positive and negative offsets to form a closed path
    closed_path = torch.cat([
        torch.stack(offset_points_positive), 
        torch.stack(offset_points_negative[::-1])  # Reverse negative offsets
    ], dim=0)

    return closed_path





# coverted to pytorch from https://github.com/joelibaceta/triangulator/blob/main/triangulator/ear_clipping_method.py

def signed_area(polygon):
    """Calculate the signed area of a polygon to determine its winding order."""
    area = 0
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]
        area += (x1 * y2 - x2 * y1)
    return area / 2

def ensure_winding_order(polygon):
    """Ensure that the polygon vertices are in counter-clockwise order."""
    if signed_area(polygon) < 0:
        polygon.reverse()  # Reverse if the order is clockwise
    return polygon

def calculate_clockwise_angle(a, b, c):
    """
    Calculate the counterclockwise angle between two vectors BA and BC using PyTorch.
    Args:
        a, b, c (tuple or tensor): 2D coordinates of triangle vertices (x, y)
    Returns:
        float: counterclockwise angle in degrees
    """
    # Convert input to tensors if they are not already
    # a = torch.tensor(a, dtype=torch.float32)
    # b = torch.tensor(b, dtype=torch.float32)
    # c = torch.tensor(c, dtype=torch.float32)

    # vector BA and BC
    ba = a - b
    bc = c - b
    # ba = ba[:,0]
    # bc = bc[:,0]
    
    # cross product and magnitude
    dot_product = torch.dot(ba, bc)
    magnitude_ba = torch.norm(ba)
    magnitude_bc = torch.norm(bc)
    
    # calculate cos(theta)
    cos_theta = dot_product / (magnitude_ba * magnitude_bc)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # avoid invalid value
    angle = torch.acos(cos_theta) * 180.0 / torch.pi  # in degrees
    
    # check the direction of the angle
    cross_product = ba[0] * bc[1] - ba[1] * bc[0]
    if cross_product > 0:
        # anticlockwise angle
        angle = 360 - angle
    
    return angle.item()  # Convert tensor to scalar value if needed


def is_point_in_triangle(a, b, c, p):
    """
    Check if point P is inside triangle ABC using PyTorch.
    Args:
        a, b, c (tuple or tensor): vertices of the triangle, format (x, y)
        p (tuple or tensor): point to be checked, format (x, y)
    Returns:
        bool: True -> inside or on boundaries, False -> outside
    """
    # Convert inputs to tensors if they are not already
    a = torch.tensor(a, dtype=torch.float32)
    b = torch.tensor(b, dtype=torch.float32)
    c = torch.tensor(c, dtype=torch.float32)
    p = torch.tensor(p, dtype=torch.float32)

    # Calculate vectors
    ab = b - a
    ap = p - a
    bc = c - b
    bp = p - b
    ca = a - c
    cp = p - c

    # Cross product
    cross1 = ab[0] * ap[1] - ab[1] * ap[0]
    cross2 = bc[0] * bp[1] - bc[1] * bp[0]
    cross3 = ca[0] * cp[1] - ca[1] * cp[0]

    # Check the sign of the cross product
    return (cross1 >= 0 and cross2 >= 0 and cross3 >= 0) or (cross1 <= 0 and cross2 <= 0 and cross3 <= 0)


def triangulate(polygon: tuple) -> list:
    """
    This function will triangulate any polygon and return a list of triangles.

        Parameters: 
            polygon (tuple): A tuple of tuples with the points of the polygon
        Returns:
            list: A list of tuples with the triangles vertices
    """

    final_triangles = []
    vertices = list(polygon)
    original_vertices = list(polygon)
    triangles_found = -1

    # Ensure the winding order is counter-clockwise
    vertices = ensure_winding_order(vertices)

    # While there are triangles to be found
    while triangles_found != 0: 
        triangles_found = 0

        for index, _ in enumerate(vertices):
            prev_vertex = vertices[index - 1]
            next_vertex = vertices[(index + 1) % len(vertices)]  # using mod to avoid index out of range
            vertex = vertices[index]

        
            angle = calculate_clockwise_angle(prev_vertex, vertex, next_vertex)

            if angle >= 180:
                # Skip because angle is greater than or equal to 180
                continue
            else:
                # Build a triangle with the three vertices
                triangle = (prev_vertex,vertex,next_vertex)
                # Get vertices that are not part of the triangl
                points = [p for p in original_vertices if not any(torch.equal(p, t) for t in triangle)]
                        
                # Check if there is a vertex inside the triangle using barycentric coordinates
                inside_evaluation = [is_point_in_triangle(prev_vertex,vertex,next_vertex, point) for point in points]
                # If no points are inside the triangle
                if not any(inside_evaluation):
                    # Add triangle to final triangles
                    final_triangles.append(triangle)
                    # Remove vertex from vertices
                    vertices.pop(index)
                    # Increment triangles found
                    triangles_found += 1
                    break

        # Check for infinite loop
        if triangles_found == 0:
            # print(f"Loop detected. Exiting. found {len(final_triangles)} triangles")
            break
    return final_triangles





def faces_to_edges(faces):
    edges = set()
    for face in faces:
        num_vertices = len(face)
        for i in range(num_vertices):
            # Create an edge with ordered vertices (smaller index first)
            edge = tuple(sorted((face[i], face[(i + 1) % num_vertices])))
            edges.add(edge)
    return list(edges)



def combine_list_of_graphs(graph_list):
    # Initialize lists to store merged data
    x_list, edge_index_list, edge_attr_list = [], [], []
    pos_list, node_normals_list, node_type_list = [], [], []

    node_offset = 0  # Tracks the index offset for edge_index

    for graph in graph_list:
        # Append node features
        x_list.append(graph.x + node_offset)

        # Adjust and append edge indices
        edge_index_list.append(graph.edge_index + node_offset)

        # Append edge attributes if they exist
        # if graph.edge_attr is not None:
        #     edge_attr_list.append(graph.edge_attr)

        # Append other attributes if they exist
        if graph.pos is not None:
            pos_list.append(graph.pos)
        if graph.node_normals is not None:
            node_normals_list.append(graph.node_normals)
        if graph.node_type is not None:
            node_type_list.append(graph.node_type)

        # Update node offset
        node_offset += len(graph.pos)

    # Combine all attributes
    combined_x = torch.cat(x_list, dim=0)
    combined_edge_index = torch.cat(edge_index_list, dim=1)
    #combined_edge_attr = torch.cat(edge_attr_list, dim=0) if edge_attr_list else None
    combined_pos = torch.cat(pos_list, dim=0) if pos_list else None
    combined_node_normals = torch.cat(node_normals_list, dim=0) if node_normals_list else None
    combined_node_type = torch.cat(node_type_list, dim=0) if node_type_list else None

    # Create a new graph
    combined_graph = Data(
        x=combined_x,
        edge_index=combined_edge_index,
        # edge_attr=combined_edge_attr,
        pos=combined_pos,
        node_normals=combined_node_normals,
        node_type=combined_node_type,
    )

    return combined_graph




from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, out_channels)

    def forward(self, data):
        pos, edge_index = data.pos, data.edge_index
        x = F.relu(self.conv1(pos, edge_index))
        x = self.conv2(x, edge_index)
        return x
    
def compute_normals_with_faces(points, faces):
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
    
def compose_graph_from_point_list(elements,ant_candidate_dict_rel_pytorch, elements_type):
    graph_list = []
    for element in elements:
        closed_path = create_buffered_closed_path(element, buffer=ant_candidate_dict_rel_pytorch['w']/2)[:,:,0]#, buffer=example_vector[key_to_index_example['w']]/2)
        triangles = triangulate(closed_path)
        
        # Step 1: Create a mapping of points (converted to tuples) to indices
        point_to_index = {tuple(point.tolist()): idx for idx, point in enumerate(closed_path)}

        # Step 2: Convert faces into index-based faces
        faces_with_indices = [
            [point_to_index[tuple(point.tolist())] for point in face] for face in triangles
        ]
        # print(faces_with_indices)
        edges = faces_to_edges(faces_with_indices)
        edges_pytorch = torch.tensor(edges)
        # Add another plane with x = 0
        points= torch.cat([torch.zeros(closed_path.shape[0], 1, device=closed_path.device), closed_path], dim=1)
        node_normals = compute_normals_with_faces(points, torch.tensor(faces_with_indices))
        node_type = torch.tensor(len(points) * [one_hot_encoding(model_stl_types[elements_type])])
        
        # plot_3d_points_edges(points.detach(), edges)
        # Create the PyTorch Geometric graph
        graph_element = Data(x=torch.tensor(range(len(points))),pos=points, edge_index=edges_pytorch.T, node_normals=node_normals, node_type=node_type)
        graph_list.append(graph_element)
    graph = combine_list_of_graphs(graph_list)
    return graph


import torch
from torch_geometric.data import Data

def merge_graphs_from_dict(graph_dict):
    """
    Merges a dictionary of PyTorch Geometric graphs into a single graph.

    Args:
        graph_dict (dict): Dictionary of graphs, where keys are graph identifiers and values are `Data` objects.

    Returns:
        tuple: Merged node features, positions, edges, batch indices, and offsets, along with the key-to-offset mapping.
    """
    all_x, all_pos, all_node_types, all_node_normals, all_edge_index, all_batch , all_node_probs, all_faces, all_node_areas = [], [], [], [], [],[],[],[],[]
    offsets = [0]  # Track node index offsets for each graph
    offset = 0
    key_to_offset = {}

    for key, graph in graph_dict.items():
        if graph == []:
            continue
        num_nodes = graph.pos.shape[0]

        # Append node features, positions, edges, and batch indices
        #all_x.append(torch.tensor(graph.x) if graph.x is not None else torch.zeros((num_nodes, 1)))
        all_pos.append(graph.pos)
        all_node_types.append(graph.node_type)
        all_node_normals.append(graph.node_normals)
        all_edge_index.append(graph.edge_index + offset)
        all_batch.append(torch.full((num_nodes,), len(key_to_offset)))
        if hasattr(graph, 'node_probs'):
            all_node_probs.append(graph.node_probs)
        else:
            all_node_probs.append(torch.zeros(num_nodes, 1, device=graph.pos.device))
        if hasattr(graph, 'faces'):
            all_faces.append(graph.faces + offset)
        else:            
            all_faces.append(torch.zeros((0, 3), dtype=torch.long, device=graph.pos.device))
        if hasattr(graph, 'node_areas'):
            all_node_areas.append(graph.node_areas)
        else:
            all_node_areas.append(torch.zeros(num_nodes, 1, device=graph.pos.device))


        # Record offset and update
        key_to_offset[key] = offset
        offset += num_nodes
        offsets.append(offset)

    #merged_x = torch.cat(all_x, dim=0)
    merged_pos = torch.cat(all_pos, dim=0)
    merged_node_types = torch.cat(all_node_types, dim=0)
    merged_node_normals = torch.cat(all_node_normals, dim=0)
    merged_edge_index = torch.cat(all_edge_index, dim=1)
    merged_batch = torch.cat(all_batch, dim=0)
    merged_node_probs = torch.cat(all_node_probs, dim=0) 
    merged_faces = torch.cat(all_faces, dim=0)
    merged_node_areas = torch.cat(all_node_areas, dim=0)


    return  merged_pos, merged_node_types, merged_node_normals, merged_edge_index, merged_batch, offsets, key_to_offset, merged_node_probs, merged_faces, merged_node_areas

def find_k_nearest_neighbors_pytorch(sphere_nodes, target_nodes, k):
    """
    Finds k nearest neighbors for each sphere node in the target graph using PyTorch.

    Args:
        sphere_nodes (torch.Tensor): Positions of the sphere nodes (N, 3).
        target_nodes (torch.Tensor): Positions of the target graph's nodes (M, 3).
        k (int): Number of nearest neighbors to find.

    Returns:
        torch.Tensor: Indices of k nearest neighbors for each sphere node (N, k).
    """
    # Compute pairwise distances
    distances = torch.cdist(sphere_nodes, target_nodes)  # Shape: (N, M)

    # Get the indices of the k smallest distances
    _, indices = distances.topk(k, largest=False, dim=1)  # Shape: (N, k)
    return indices

def add_edges_between_sphere_and_dict_pytorch(sphere_key, graph_dict, offsets, key_to_offset, k):
    """
    Adds edges connecting each sphere node to its k nearest neighbors in all other graphs.

    Args:
        sphere_key (str): Key of the sphere graph in the dictionary.
        graph_dict (dict): Dictionary of graphs, where keys are graph identifiers and values are `Data` objects.
        offsets (list[int]): Node index offsets for all graphs in the merged graph.
        key_to_offset (dict): Mapping of graph keys to their offsets.
        k (int): Number of nearest neighbors to connect.

    Returns:
        torch.Tensor: New edges connecting sphere nodes to other graphs.
    """
    sphere_graph = graph_dict[sphere_key]
    sphere_nodes = sphere_graph.pos
    sphere_offset = key_to_offset[sphere_key]

    new_edges = []
    for key, graph in graph_dict.items():
        k_copy = k
        if key == sphere_key or graph==[]:
            continue
        if k_copy> graph.pos.shape[0]:
            k_copy = graph.pos.shape[0]
        # Find k nearest neighbors
        current_indices = find_k_nearest_neighbors_pytorch(sphere_nodes, graph.pos, k_copy)

        # Adjust indices to account for graph's position in merged graph
        current_indices += key_to_offset[key]
        sphere_indices = torch.arange(sphere_nodes.shape[0]) + sphere_offset
        sphere_indices_expanded = sphere_indices.unsqueeze(1).repeat(1, k_copy)

        # Create edge pairs
        new_edges.append(torch.stack([sphere_indices_expanded.to(current_indices.device).flatten(), current_indices.flatten()], dim=0))

    return torch.cat(new_edges, dim=1) if new_edges else torch.empty((2, 0))

def merge_and_connect_graphs_from_dict(graph_dict, sphere_key, k, add_sphere=True):
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
    merged_pos, merged_node_types, merged_node_normals, merged_edge_index, merged_batch, offsets, key_to_offset, merged_node_probs, merged_faces, merged_node_areas = merge_graphs_from_dict(graph_dict)
    if add_sphere: 
        # Add edges between sphere nodes and other graphs
        new_edges = add_edges_between_sphere_and_dict_pytorch(sphere_key, graph_dict, offsets, key_to_offset, k)

        # Combine new edges with existing edges
        merged_edge_index = torch.cat([merged_edge_index, new_edges], dim=1)

        # Return the merged graph
        x = torch.cat([merged_pos, merged_node_normals, merged_node_areas, merged_node_types, ], dim=-1)
        return Data(x=x,
                    pos=merged_pos, node_type=merged_node_types,
                    node_normals=merged_node_normals,
                    edge_index=merged_edge_index,
                    batch=merged_batch,
                    node_probs=merged_node_probs,
                    faces=merged_faces,
                    node_areas=merged_node_areas

                   )
    else:
        # x = torch.cat([merged_pos, merged_node_normals, merged_node_areas, merged_node_types], dim=-1)
        x = torch.cat([merged_pos, merged_node_normals,  merged_node_types], dim=-1)
        return Data(x=x,
                    pos=merged_pos,
                    node_type=merged_node_types,
                    node_normals=merged_node_normals,
                    edge_index=merged_edge_index,
                    batch=merged_batch,
                    node_probs=merged_node_probs, 
                    faces=merged_faces,
                    node_areas=merged_node_areas
                   )


def find_yz_plane_bounding_box( graph) -> torch.Tensor:
    """
    Finds the bounding box of the plane in a PyTorch Geometric mesh graph
    that is parallel to the XZ axis and has a specified y-coordinate.

    Args:
        data (torch_geometric.data.Data): The mesh data object containing
            vertex positions (`pos`).
        min_y (float): The y-coordinate of the plane.

    Returns:
        torch.Tensor: A tensor containing the bounding box coordinates:
                    [min_y, min_z, max_y, max_z].
    """
    # Step 1: Identify vertices on the plane with the specified y-coordinate
    vertices_on_plane = graph.pos #graph.pos[graph.pos[:, 1] == min_y]

    # Step 2: Calculate the bounding box in the XZ plane
    min_y = torch.min(vertices_on_plane[:, 1])
    max_y = torch.max(vertices_on_plane[:, 1])
    min_z = torch.min(vertices_on_plane[:, 2])
    max_z = torch.max(vertices_on_plane[:, 2])
    bounding_box = torch.tensor([min_y, min_z, max_y, max_z], dtype=torch.float)
    return bounding_box


def mask_negative_x_plane(graph, normal: torch.Tensor = torch.tensor([0, -1, 0], dtype=torch.float)):
    """
    Masks the plane on a PyTorch Geometric mesh graph where the plane's normal
    coincides with the specified normal vector (default is (-1, 0, 0)).    
    Specifically, it identifies the plane with the most negative y-coordinate
    """
    # Step 1: Identify the minimal y-coordinate value
    min_x = torch.min(graph.pos[:, 0])
    vertex_indices = torch.nonzero(graph.pos[:, 0] == min_x).squeeze()
    return vertex_indices
            

def get_FR4_antenna_plane( graph, debug=False):
    antenna_FR4_plan_vertex_indices = mask_negative_x_plane(graph)
    FR4_antenna_plane_subgraph = graph.subgraph(antenna_FR4_plan_vertex_indices)
    if debug:
        plot_3d_points_edges(FR4_antenna_plane_subgraph.pos, FR4_antenna_plane_subgraph.edge_index.T.tolist())
    return FR4_antenna_plane_subgraph


def check_bbox_fit_and_center_offset(fr4_bbox, ant_bbox):
    """
    Check if the ant_bbox fits within fr4_bbox and calculate the offset between their centers.
    
    Args:
        fr4_bbox (torch.Tensor): Bounding box of the FR4 region in the format [y_min, z_min, y_max, z_max].
        ant_bbox (torch.Tensor): Bounding box of the antenna in the format [y_min, z_min, y_max, z_max].
    
    Returns:
        center_offset (torch.Tensor): Offset in the [y, z] directions between the centers of the bounding boxes.
    """
    # Check if ant_bbox fits within fr4_bbox


    # Calculate center of each bounding box
    fr4_center = torch.tensor([
        (fr4_bbox[0] + fr4_bbox[2]) / 2,  # Center y
        (fr4_bbox[1] + fr4_bbox[3]) / 2   # Center z
    ])
    ant_center = torch.tensor([
        (ant_bbox[0] + ant_bbox[2]) / 2,  # Center y
        (ant_bbox[1] + ant_bbox[3]) / 2   # Center z
    ])

    # Calculate offset between centers
    center_offset = ant_center - fr4_center
    
    return center_offset


from torch_cluster import knn_graph


def compute_node_surface_area(data: Data):
    """
    Compute the surface area associated with each node in a PyTorch Geometric mesh graph.

    Args:
        data (torch_geometric.data.Data): A PyTorch Geometric mesh graph with `pos` and `faces`.

    Returns:
        torch.Tensor: A tensor of shape (num_nodes,) containing the surface area per node.
    """
    pos = data.pos  # Node positions [num_nodes, 3]
    
    faces = edges_to_faces(data.edge_index.T)  # Face indices [num_faces, 3]

    # Get triangle vertices
    v0, v1, v2 = pos[faces[:, 0]], pos[faces[:, 1]], pos[faces[:, 2]]

    # Compute face normals using cross product
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)

    # Compute face areas (half the magnitude of the cross product)
    face_areas = 0.5 * torch.norm(face_normals, dim=1)

    # Initialize node surface areas
    node_surface_area = torch.zeros(pos.shape[0], device=pos.device)

    # Distribute area to nodes (each face contributes equally to its 3 nodes)
    for i in range(3):
        node_surface_area.index_add_(0, faces[:, i], face_areas / 3)

    return node_surface_area


def sphere_to_surface_mesh(pos: torch.Tensor, k: int = 6, num_nodes_main: int = 0) -> torch.Tensor:
    """
    Convert a set of points on a sphere into a surface mesh and shift indices 
    for integration into a larger graph.

    Args:
        pos (torch.Tensor): Tensor of shape [N, 3] representing the points on the sphere.
        k (int): Number of nearest neighbors to consider for constructing edges.
        num_nodes_main (int): Number of nodes in the main graph before adding the sphere.

    Returns:
        torch.Tensor: Adjusted edge_index tensor for PyG.
    """
    # Step 1: Construct edges using KNN
    edge_index = knn_graph(pos, k=k, loop=False)
    
    # Step 2: Extract triangular faces
    faces = []
    num_nodes = pos.shape[0]
    
    # Convert edges to a set for fast lookup
    edge_set = {tuple(edge_index[:, i].tolist()) for i in range(edge_index.shape[1])}

    # Find triangles (3-cycle in the edge graph)
    for i in range(edge_index.shape[1]):
        u, v = edge_index[:, i].tolist()
        for w in range(num_nodes):
            if (v, w) in edge_set and (w, u) in edge_set:
                face = tuple(sorted([u, v, w]))  # Ensure uniqueness
                faces.append(face)
    
    # Remove duplicate faces
    faces = list(set(faces))
    
    if len(faces) == 0:
        raise ValueError("No faces were generated. Try increasing k.")

    # Convert faces to tensor
    face_tensor = torch.tensor(faces, dtype=torch.long).T  # Shape: [3, num_faces]
    
    # Step 3: Convert faces to edges
    edge_index_sphere = faces_to_edges(face_tensor)
    
    # Step 4: Shift indices for integration into the main graph
    edge_index_sphere += num_nodes_main
    
    return edge_index_sphere
