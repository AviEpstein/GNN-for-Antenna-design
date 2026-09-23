import torch
from src.dataset.dataloader_utils import normalize_gain


def _compute_farfield_core(surface_data, f_Hz=5.6e9, num_theta=37, num_phi=72, device=None, raw_sc=False):
    """
    Computes far-field electric fields (E_theta, E_phi) from surface currents.
    Vectorized over the angular grid using PyTorch broadcasting.
    Returns complex tensors of shape (num_theta, num_phi).
    if raw_sc is True, expects surface_data to have keys like '#x [mm]', 'y [mm]', 'z [mm]', 'KxRe [A/m]', 'KxIm [A/m]', etc.
        If raw_sc is False, expects surface_data to have 'pos' (N, 3) and 'J' (N, 6) tensors.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _to_tensor(arr, dtype=torch.float32):
        if isinstance(arr, torch.Tensor):
            return arr.to(device=device, dtype=dtype)
        return torch.tensor(arr, dtype=dtype, device=device)
    if raw_sc:
        print("Raw surface currents detected. Ensure that the input data is correctly formatted and scaled.")

        x = _to_tensor(surface_data['#x [mm]']) * 1e-3
        y = _to_tensor(surface_data['y [mm]']) * 1e-3
        z = _to_tensor(surface_data['z [mm]']) * 1e-3
        zeros = torch.zeros_like(x)
        Jx_re = _to_tensor(surface_data.get('KxRe [A/m]', zeros))
        Jx_im = _to_tensor(surface_data.get('KxIm [A/m]', zeros))
        Jy_re = _to_tensor(surface_data.get('KyRe [A/m]', zeros))
        Jy_im = _to_tensor(surface_data.get('KyIm [A/m]', zeros))
        Jz_re = _to_tensor(surface_data.get('KzRe [A/m]', zeros))
        Jz_im = _to_tensor(surface_data.get('KzIm [A/m]', zeros))
    else:
        x = surface_data['pos'][:, 0] * 1e-3
        y = surface_data['pos'][:, 1] * 1e-3
        z = surface_data['pos'][:, 2] * 1e-3
        Jx_re = surface_data['J'][:, 0]
        Jx_im = surface_data['J'][:, 1]
        Jy_re = surface_data['J'][:, 2]
        Jy_im = surface_data['J'][:, 3]
        Jz_re = surface_data['J'][:, 4]
        Jz_im = surface_data['J'][:, 5]
        area = surface_data.get('area', None)


    Jx = torch.complex(Jx_re, Jx_im)  # (N,)
    Jy = torch.complex(Jy_re, Jy_im)
    Jz = torch.complex(Jz_re, Jz_im)

    c = 299792458.0
    k = torch.tensor(2 * torch.pi / (c / f_Hz), dtype=torch.float32, device=device)
    eta = 376.73

    theta = torch.linspace(0, torch.pi, num_theta, device=device)   # (T,)
    phi   = torch.linspace(0, 2 * torch.pi, num_phi, device=device) # (P,)

    sin_th = torch.sin(theta)  # (T,)
    cos_th = torch.cos(theta)  # (T,)
    sin_ph = torch.sin(phi)    # (P,)
    cos_ph = torch.cos(phi)    # (P,)

    # Observation unit vectors flattened to (T*P, 3) — avoids a 3D (T,P,N) tensor
    r_hat_x = (sin_th[:, None] * cos_ph[None, :]).reshape(-1)
    r_hat_y = (sin_th[:, None] * sin_ph[None, :]).reshape(-1)
    r_hat_z = cos_th[:, None].expand(num_theta, num_phi).reshape(-1)
    R = torch.stack([r_hat_x, r_hat_y, r_hat_z], dim=1)  # (T*P, 3)

    # Phase argument via matmul: (T*P, N)
    coords = torch.stack([x, y, z], dim=0)  # (3, N)
    phase_arg = k * (R @ coords)             # (T*P, N)
    phase_term = torch.polar(torch.ones_like(phase_arg), phase_arg)  # (T*P, N) complex

    # Radiation vectors via matmul: (3, T*P) = (3, N) @ (N, T*P)
    J = torch.stack([Jx, Jy, Jz], dim=0)    # (3, N) complex

    L = J @ phase_term.T                     # (3, T*P)

    Lx = L[0].reshape(num_theta, num_phi)
    Ly = L[1].reshape(num_theta, num_phi)
    Lz = L[2].reshape(num_theta, num_phi)

    # Spherical projection
    cos_th_c = cos_th[:, None].to(torch.complex64)
    sin_th_c = sin_th[:, None].to(torch.complex64)
    cos_ph_c = cos_ph[None, :].to(torch.complex64)
    sin_ph_c = sin_ph[None, :].to(torch.complex64)

    L_theta = Lx * cos_th_c * cos_ph_c + Ly * cos_th_c * sin_ph_c - Lz * sin_th_c
    L_phi   = -Lx * sin_ph_c + Ly * cos_ph_c

    const = torch.tensor(-1j * k.item() * eta / (4 * torch.pi), dtype=torch.complex64, device=device)
    E_theta = const * L_theta
    E_phi   = const * L_phi

    return E_theta, E_phi


def compute_farfield_from_currents(surface_data, f_Hz=5.6e9, image_shape=(37, 72), device=None):
    """
    Computes normalized far-field gain pattern from surface currents.
    Returns a gain array of shape determined by normalize_gain.
    """
    num_theta, num_phi = image_shape
    E_theta, E_phi = _compute_farfield_core(surface_data, f_Hz, num_theta, num_phi, device=device)

    U_theta = E_theta.abs() ** 2
    U_phi   = E_phi.abs() ** 2
    ff = normalize_gain(U_theta, U_phi, img_shape=image_shape, device=device)
    return ff

def compute_farfield_from_currents_for_batch(graph, surface_current_batches, f_Hz=5.6e9, image_shape=(37, 72), device=None):
    """
    Computes far-field gain patterns for a batch of surface current data.
    Expects batch_surface_data to be a list of dicts, each with 'pos' and 'J' tensors.
    Returns a tensor of shape (B, num_theta, num_phi) with normalized gain patterns.
    """
    batch_ff = []
    for batch_idx in range(graph.num_graphs):
        graph_batch_mask = graph.batch == batch_idx
        graph_pos = graph.pos[graph_batch_mask]
        surface_current = surface_current_batches[graph_batch_mask]
        surface_data = {'pos': graph_pos, 'J': surface_current}
        ff = compute_farfield_from_currents(surface_data, f_Hz, image_shape, device)
        batch_ff.append(ff)
    return torch.stack(batch_ff, dim=0)
