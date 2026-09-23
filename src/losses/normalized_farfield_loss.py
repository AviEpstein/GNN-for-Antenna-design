import torch


def normalized_farfeild_loss(farfeild, tol=1e-3):
    if farfeild.dim() == 2:
        farfeild = farfeild.unsqueeze(0)
    batch_size, C, H, W = farfeild.shape
    theta_rad = (torch.linspace(0, 180, H, dtype=torch.float32) * torch.pi / 180).to(farfeild.device)  # H
    phi_rad = (torch.linspace(0, 360, W, dtype=torch.float32) * torch.pi / 180).to(farfeild.device)    # W
    d_theta = torch.max(torch.diff(theta_rad)).to(farfeild.device)
    d_phi = torch.max(torch.diff(phi_rad)).to(farfeild.device)

    # Reshape theta_rad to (H, 1) and broadcast for batch computation
    sin_theta = torch.sin(theta_rad).unsqueeze(1)  # Shape: (H, 1)

    # Compute efficiency for each batch
    efficiency = torch.sum(farfeild * sin_theta, dim=(1, 2)) * d_theta * d_phi / (4 * torch.pi)

    # Compute loss for each batch
    losses = torch.abs(efficiency - 1.0)
    loss = torch.mean(losses)
    return loss
