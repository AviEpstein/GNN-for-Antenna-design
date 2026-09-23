"""
| Model | Params | GFLOPs  | Inference |
|---|---|---|---|
| Diffusion U-Net | X.XM | X.X | X s/500 cand. |

Uses the model/config from src/diffusion/diffusion_trainer.py (the
ConditionalDenoisingUNetSmallAttentionWithConcat denoiser wrapped in the DDPM
class) to produce a raw size/runtime estimate:
  - Params: total parameters of the denoiser U-Net.
  - GFLOPs: cost of a single conditional U-Net forward pass (i.e. one
    denoising step, batch size 1), measured via torch.profiler.
  - Inference: wall-clock time to run the full reverse-diffusion sampling
    loop (T denoising steps, classifier-free guidance -> 2 U-Net forward
    passes per step) for a batch of 500 candidate antennas, matching
    `number_of_samples_to_generate` in diffusion_trainer.py. Random noise and
    a random far-field condition are used -- no dataset/trained checkpoint is
    required, since only shapes/timing matter here.
"""

import time

import torch

from src.diffusion.unet_attention import ConditionalDenoisingUNetSmallAttentionWithConcat
from src.diffusion.ddpm import DDPM


if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Matches the diffusion_trainer.py config for the trained checkpoint
    # (data_set_type='pixel_data_with_reflectors' -> 2 image channels).
    config = {
        'T': 700,
        'img_wh': (16, 16),
        'radiation_image_shape': [34, 34, 1],
        'number_of_samples_to_generate': 500,
        'guidance_scale': 2.0,
    }
    in_channels = 2
    num_hiddens = 128
    ff_in_ch = 1
    ff_h, ff_w = config['radiation_image_shape'][0], config['radiation_image_shape'][1]

    denoiser_unet = ConditionalDenoisingUNetSmallAttentionWithConcat(
        config=config, in_channels=in_channels, num_hiddens=num_hiddens, ff_in_ch=ff_in_ch
    ).to(device)
    denoiser_unet.eval()

    total_params = sum(p.numel() for p in denoiser_unet.parameters())

    ddpm = DDPM(denoiser_unet, num_ts=config['T']).to(device)

    # ---- GFLOPs for a single conditional U-Net forward pass (one denoising step) ----
    x = torch.randn(1, in_channels, *config['img_wh'], device=device)
    c = torch.randn(1, ff_h, ff_w, device=device)
    t = torch.full((1, 1), 0.5, device=device)
    with torch.no_grad():
        for _ in range(3):
            denoiser_unet(x, c=c, t=t)
        with torch.profiler.profile(with_flops=True) as prof:
            denoiser_unet(x, c=c, t=t)
    total_flops = sum(evt.flops for evt in prof.key_averages() if evt.flops is not None)
    gflops_per_step = total_flops / 1e9

    # ---- Inference latency: full DDPM sampling loop for 500 candidates ----
    batch_size = config['number_of_samples_to_generate']
    c_batch = torch.randn(batch_size, ff_h, ff_w, device=device)

    # Warm-up run (compiles cuDNN kernels etc.), not timed.
    ddpm.sample(c=c_batch, img_wh=config['img_wh'], guidance_scale=config['guidance_scale'], in_channels=in_channels)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    start = time.perf_counter()
    ddpm.sample(c=c_batch, img_wh=config['img_wh'], guidance_scale=config['guidance_scale'], in_channels=in_channels)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - start

    print(f"| Model | Params | GFLOPs | Inference |")
    print(f"|---|---|---|---|")
    print(f"| Diffusion U-Net | {total_params / 1e6:.1f}M | {gflops_per_step:.1f} | {elapsed_s:.1f} s/{batch_size} cand. |")
