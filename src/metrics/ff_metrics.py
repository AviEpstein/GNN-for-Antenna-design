import torch
from torchmetrics import MeanSquaredError, MeanAbsoluteError, PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, MultiScaleStructuralSimilarityIndexMeasure
from src.metrics.peak_power_metric import calculate_HPBW

class Metrics:
    def __init__(self, config, ff_stats=None, clamp=True, device='cuda'):
        self.clamp = clamp
        self.config = config
        self.device = device
        self.data_range = 8 # assuming input images are in range [-15, 10]
        if ff_stats is not None:
            self.data_range = ff_stats['max_ff'] - ff_stats['min_ff']
        self.reset()


    def reset(self):
        self.mse_metric = MeanSquaredError().to(self.device)
        self.mae_metric = MeanAbsoluteError().to(self.device)
        self.psnr_metric = PeakSignalNoiseRatio().to(self.device)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=self.data_range).to(self.device)
        self.mssim_metric = MultiScaleStructuralSimilarityIndexMeasure(data_range=self.data_range,kernel_size=5, betas=(0.0448, 0.2856, 0.3001)).to(self.device)
        self.peak_power_sum = 0.0
        self.peak_power_count = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        if self.clamp:
            preds = torch.clamp(preds, -15, 10)
            targets = torch.clamp(targets, -15, 10)
        self.mse_metric.update(preds, targets)
        self.mae_metric.update(preds, targets)
        self.psnr_metric.update(preds, targets)
        self.ssim_metric.update(preds, targets)
        self.mssim_metric.update(preds, targets)

        batch_peak_power = calculate_HPBW(preds, targets)
        self.peak_power_sum += batch_peak_power.item() * preds.size(0)
        self.peak_power_count += preds.size(0)

    def compute(self):
        mse = self.mse_metric.compute()
        mae = self.mae_metric.compute()
        psnr = self.psnr_metric.compute()
        ssim = 1 - self.ssim_metric.compute()
        mssim = self.mssim_metric.compute()
        peak_power = self.peak_power_sum / self.peak_power_count if self.peak_power_count > 0 else 0.0
        return {
            'mse': mse,
            'mae': mae,
            'psnr': psnr,
            'ssim': ssim,
            'mssim': mssim,
            'peak_power': peak_power
        }
