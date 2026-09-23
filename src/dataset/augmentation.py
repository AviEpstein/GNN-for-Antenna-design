import numpy as np
import trimesh
import cv2
import random
import torch



def cyclic_shift_image(image, shift_horizontol, shift_vertical=0):
    # Convert the image to a NumPy array if it is not already one
    image = np.asarray(image)
    # Shift the image
    shifted_image = np.roll(image, shift_horizontol, axis=2)  # Shift along the z-axis
    #shifted_image = np.roll(shifted_image, shift_vertical, axis=0)  # Shift along the y-axis
    return shifted_image  


def rotate_image(image, theta_z):
    # For the image rotation, we assume a 2D rotation on the plane
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    theta_2d = np.degrees(theta_z)  # Assuming rotation around z-axis for 2D image
    rotation_matrix_cv = cv2.getRotationMatrix2D(image_center, theta_2d, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix_cv, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return rotated_image

    
def rotate_mesh(mesh_vertices_pos,theta_z,  theta_x=0, theta_y=0 ):
    # Define rotation matrices for each axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x)],
        [0, np.sin(theta_x), np.cos(theta_x)]
    ],dtype=np.float32)
    
    Ry = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y)],
        [0, 1, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y)]
    ],dtype=np.float32)
    
    Rz = np.array([
        [np.cos(theta_z), -np.sin(theta_z), 0],
        [np.sin(theta_z), np.cos(theta_z), 0],
        [0, 0, 1]
    ],dtype=np.float32)
    
    # Combined rotation matrix
    R = Rz @ Ry @ Rx
    
    # Rotate the mesh vertices
    rotated_vertices = mesh_vertices_pos @ R.T
    return rotated_vertices


class RotateTransform3D():
    def __init__(self, angle_range = [0, 2*np.pi]):
        '''
        Rotating the mesh and the farfeild image by  a random pixel each pixel in the z axis rotation is 2*pi/64 angels
        TODO: add rotation around horizontal shoul be pi/64 around the y axis
        '''
        self.angle_range = angle_range

    def __call__(self, mesh, ff_image, num_of_horizontol_shfted_pixels=None):
        
        # Define the angle range in terms of the number of pi/64 steps for th vertical axis, 
        # TODO: 2*pi/64 for the horizontal axis of the ff image
        #angle_range_steps = (int(self.angle_range[0] * 64 / 2*np.pi), int(self.angle_range[1] * 64 / 2*np.pi))
        
        if num_of_horizontol_shfted_pixels == None:
            pixel_step = (0,3)
            # Randomly select angles for each axis within the given range, in steps of pi/64
            num_of_horizontol_shfted_pixels = random.randint(*pixel_step) * 16
            
        #theta_x = np.radians(random.randint(*angle_range_steps) * (2*np.pi / 64))
        #theta_y = np.radians(random.randint(*angle_range_steps) * (2*np.pi / 64))
        theta_z = num_of_horizontol_shfted_pixels * (2*np.pi / 64)
        
        # transform mesh:
        mesh.pos = rotate_mesh(mesh.pos, theta_z=theta_z)
        
        # transform ff image:
        shifted_image = cyclic_shift_image(ff_image,  shift_horizontol=num_of_horizontol_shfted_pixels)
        ff_image = torch.tensor(shifted_image)
        #ff_image = rotate_image(shifted_image, theta_z)
    
        return mesh, ff_image

class ShiftTransform3D(torch.nn.Module):
    def __init__(self, max_shift):
        super(ShiftTransform3D, self).__init__()
        self.max_shift = max_shift

    def forward(self, data_from_get):
        # Shift node locations by a random value
        random_vector_translation = torch.empty(data_from_get.pos.shape[1]).uniform_(-self.max_shift, self.max_shift)
        data_from_get.pos += random_vector_translation.to(data_from_get.pos.device)

        return data_from_get


def rotate_mesh_around_z_axis(parced_mesh: trimesh.Trimesh , rotation_angle_in_degrees: int) -> trimesh.Trimesh:


    # Define your transformation matrixs
    #let's rotate the mesh around the z-axis 
    angle = np.radians(rotation_angle_in_degrees)
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                [np.sin(angle), np.cos(angle), 0],
                                [0, 0, 1]])

    # Apply the transformation to the vertices of the mesh
    transformed_vertices = np.dot(parced_mesh.vertices, rotation_matrix)

    # Create a new mesh with the transformed vertices
    transformed_mesh = trimesh.Trimesh(vertices=transformed_vertices, faces=parced_mesh.faces)
    
    return transformed_mesh

    # Save or visualize the transformed mesh
    # For visualization, you can use:
    # transformed_mesh.show()
    # For saving, you can use:
    # transformed_mesh.export('transformed_mesh.obj')



def compress_towards_threshold(x: torch.Tensor,
                                     threshold: float = 0.5,
                                     strength: float = 0.5,
                                     jitter: float = 0.0,
                                     eps: float = 1e-4) -> torch.Tensor:
    """
    Compress values in x toward a threshold while preserving side of threshold.
    If strength=0, values collapse to just-below or just-above threshold using eps.
    """
    thr = torch.as_tensor(threshold, dtype=x.dtype, device=x.device)

    # Deterministic compression
    compressed = thr + strength * (x - thr)

    # If strength=0, force just under/over threshold
    if strength == 0:
        mask_above = x >= thr
        compressed = torch.where(mask_above, thr + eps, thr - eps)

    if jitter > 0:
        # Random noise proportional to distance to threshold
        noise_scale = (x - thr).abs() * jitter
        u = torch.rand_like(compressed) * 2.0 - 1.0
        compressed = compressed + u * noise_scale

        # Make sure side is preserved
        mask_above = x >= thr
        compressed = torch.where(mask_above,
                                 torch.maximum(compressed, thr + eps),
                                 torch.minimum(compressed, thr - eps))

    return compressed.clamp(0.0, 1.0)

def augment_input_matrix(x_raw, tau=0.5, lambda_soft=0.3, sigma=0.05):
    """
    x_raw: (B,H,W) or (H,W) in [0,1]
    return: augmented x for training forward surrogate, preserving max .
    """
    # Ensure x_raw is at least 3D for batch processing
    if x_raw.dim() == 2:
        x_raw = x_raw.unsqueeze(0)

    # Find the argmax indices for each batch
    argmax = torch.argmax(x_raw.view(x_raw.size(0), -1), dim=1)
    argmax_y = argmax // x_raw.shape[-1]
    argmax_x = argmax % x_raw.shape[-1]

    # Step 1: binary geometry actually used in CST
    x_bin = (x_raw >= tau).float()

    # Step 2: soft mixing
    x_mixed = (1 - lambda_soft) * x_bin + lambda_soft * x_raw

    # Step 3: noise augmentation (but preserve ones)
    x_aug = x_mixed + sigma * torch.randn_like(x_mixed)
    x_aug = x_aug.clamp(0, 1)

    # Ensure max stays at the same place for each batch
    for i in range(x_raw.size(0)):
        x_aug[i, argmax_y[i], argmax_x[i]] = torch.max(x_aug[i]) + 0.05

    # Verify max position and binary geometry for each batch
    for i in range(x_raw.size(0)):
        argmax = torch.argmax(x_aug[i])
        argmax_y_aug = argmax // x_raw.shape[-1]
        argmax_x_aug = argmax % x_raw.shape[-1]
        assert argmax_y_aug == argmax_y[i] and argmax_x_aug == argmax_x[i], "Max position changed after augmentation!"
        aug_binary = (x_aug[i] >= tau).float()
        assert (aug_binary == x_bin[i]).all(), "Binary geometry changed after augmentation!"

    return x_aug
