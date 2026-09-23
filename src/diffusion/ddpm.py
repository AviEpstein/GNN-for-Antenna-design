

import torch
from torch import nn


def ddpm_schedule(beta1: float, beta2: float, num_ts: int, device: str = 'cuda') -> dict:
    """Constants for DDPM training and sampling.

    Arguments:
        beta1: float, starting beta value.
        beta2: float, ending beta value.
        num_ts: int, number of timesteps.

    Returns:
        dict with keys:
            betas: linear schedule of betas from beta1 to beta2.
            alphas: 1 - betas.
            alpha_bars: cumulative product of alphas.
    """
    assert beta1 < beta2 < 1.0, "Expect beta1 < beta2 < 1.0."

    betas = torch.linspace(
        beta1, beta2, num_ts, device=device
    )

    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return {
        "beta_array": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
    }



def ddpm_forward(
    unet: nn.Module,
    ddpm_schedule: dict,
    x_0: torch.Tensor,
    c: torch.Tensor,
    p_uncond: float,
    num_ts: int,
) -> torch.Tensor:
    """Algorithm 3 (not including gradient step).

    Args:
        unet: conditional denoising U-Net
        ddpm_schedule: dict
        x_0: (N, C, H, W) input tensor.
        c: (N,) int64 condition tensor.
        p_uncond: float, probability of unconditioning the condition.
        num_ts: int, number of timesteps.

    Returns:
        (,) diffusion loss.
    """
    unet.train()
    b = x_0.shape[0]

    c_vec = c

    # set c_vec to zero with probability p_uncond
    mask = (torch.rand(b, 1, device=x_0.device) > p_uncond).float()

    t = torch.randint(1, num_ts, (b,), device=x_0.device).unsqueeze(-1)
    t_norm = t.float() / num_ts  # Normalize t to [0, 1]

    epsilon = torch.randn_like(x_0).to(x_0.device)
    alpha_bars = ddpm_schedule['alpha_bars'][t]

    # extend alpha bars to match x_0 shape
    alpha_bars = alpha_bars.view(-1, 1, 1, 1).expand_as(x_0)

    x_t = (torch.sqrt(alpha_bars) * x_0) + (torch.sqrt(1 - alpha_bars) * epsilon)

    pred = unet(x_t, c=c_vec, t=t_norm, mask=mask)
    loss = nn.functional.mse_loss(pred, epsilon, reduction='mean')

    return loss



@torch.inference_mode()
def ddpm_cfg_sample(
    unet: nn.Module,
    ddpm_schedule: dict,
    c: torch.Tensor,
    img_wh: tuple[int, int],
    num_ts: int,
    guidance_scale: float = 5.0,
    in_channels: int = 1
) -> torch.Tensor:
    """Algorithm 4.

    Args:
        unet: conditional denoising U-Net
        ddpm_schedule: dict
        c: (N,) int64 condition tensor. Only for class-conditional
        img_wh: (H, W) output image width and height.
        num_ts: int, number of timesteps.
        guidance_scale: float, CFG scale.

    Returns:
        (N, C, H, W) final sample.
    """
    unet.eval()
    batch_size = c.shape[0]

    device = next(unet.parameters()).device
    x_t = torch.randn(batch_size, in_channels, *img_wh, device=device)

    betas = ddpm_schedule['beta_array']
    alphas = ddpm_schedule['alphas']
    alpha_bars = ddpm_schedule['alpha_bars']

    for t in range(num_ts - 1, 0, -1):

        if t > 0:
            z = torch.randn_like(x_t, device=device)
        else:
            z = torch.zeros_like(x_t, device=device)

        t_norm = torch.ones(batch_size, 1, device=device) * (t / num_ts)
        c_vec = c

        pred_cond = unet(x_t, t=t_norm, c=c_vec)
        pred_uncond = unet(x_t, t=t_norm, c=torch.zeros_like(c_vec))
        pred = pred_uncond + (guidance_scale * (pred_cond - pred_uncond))

        alpha_t = alphas[t]
        alpha_bar_t = alpha_bars[t]
        beta_t = betas[t]

        alpha_bar_t_prev = alpha_bars[t - 1] if t > 0 else 1.0

        x_0_hat = (1./torch.sqrt(alpha_bar_t)) * (x_t - (torch.sqrt(1. - alpha_bar_t) * pred))
        x_t_prev = ((torch.sqrt(alpha_bar_t_prev) * beta_t) / (1 - alpha_bar_t) * x_0_hat) + \
                   ((torch.sqrt(alpha_t)*(1 - alpha_bar_t_prev))/(1 - alpha_bar_t) * x_t) + \
                   torch.sqrt(beta_t) * z

        x_t = x_t_prev


    return x_t_prev



# Do Not Modify
class DDPM(nn.Module):
    def __init__(
        self,
        unet: nn.Module,
        betas: tuple[float, float] = (1e-4, 0.02),
        num_ts: int = 300,
        p_uncond: float = 0.1,
    ):
        super().__init__()
        self.unet = unet
        self.betas = betas
        self.num_ts = num_ts
        self.p_uncond = p_uncond
        self.ddpm_schedule = ddpm_schedule(betas[0], betas[1], num_ts)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, C, H, W) input tensor.
            c: (N,) int64 condition tensor.

        Returns:
            (,) diffusion loss.
        """
        return ddpm_forward(
            self.unet, self.ddpm_schedule, x, c, self.p_uncond, self.num_ts
        )

    @torch.inference_mode()
    def sample(
        self,
        c: torch.Tensor,
        img_wh: tuple[int, int],
        guidance_scale: float = 5.0,
        in_channels: int = 1
    ):
        return ddpm_cfg_sample(
            self.unet, self.ddpm_schedule, c, img_wh, self.num_ts, guidance_scale, in_channels
        )
