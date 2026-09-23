import torch.nn as nn
import torch
from pytorch_msssim import ms_ssim
from src.utils import interpolate_1d, unscale_from_log, scale_to_20_log

from src.losses.normalized_farfield_loss import normalized_farfeild_loss


class S11Loss(nn.Module):
    def __init__(self, config):
        super(S11Loss, self).__init__()
        self.scale_pred_to_DB = True

    def forward(self, pred_magnitutde, target_S11_abs):
        if self.scale_pred_to_DB == True:
            pred_magnitutde        = scale_to_20_log(pred_magnitutde)
        target_S11 = scale_to_20_log(target_S11_abs)
        target_S11 = interpolate_1d(target_S11, target_size=256)

        non_zero_mask = (target_S11 <= -1).float()  # Regions with negative spikes
        zero_mask = (target_S11 >= -0.5).float()     # Regions that should remain zero

        # Match the negative spikes
        non_zero_loss = torch.nn.functional.mse_loss(pred_magnitutde * non_zero_mask, target_S11 * non_zero_mask)

        # Penalize unnecessary predictions in zero regions
        false_positive_loss = torch.mean((pred_magnitutde * zero_mask) ** 2)

        # Combined loss
        loss = non_zero_loss + 0.1 * false_positive_loss  # Adjust weight as needed
        s11_magnitud_loss = nn.MSELoss()(pred_magnitutde, target_S11)
        s11_magnitud_loss + 3*loss
        return s11_magnitud_loss


class S11SingleFreqLoss(nn.Module):
    def __init__(self, config):
        super(S11SingleFreqLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.db_weight = config.get('s11_db_weight', 0.01)
        self.MAE_loss = nn.L1Loss()
        warmup_steps = 15000
        self.warmup_steps = warmup_steps
        self.warmup = max(1, warmup_steps)
        self.register_buffer('_step', torch.tensor(0, dtype=torch.long))

    def forward(self, pred, target, return_db_loss=False):
        s11_complex_loss = self.mse_loss(pred, target)  # MSE loss for complex values
        # Convert to dB
        pred_abs = torch.sqrt(pred[:, 0]**2 + pred[:, 1]**2)
        target_abs = torch.sqrt(target[:, 0]**2 + target[:, 1]**2)
        pred_abs_db = 20 * torch.log10(pred_abs + 1e-8)
        target_abs_db = 20 * torch.log10(target_abs + 1e-8)
        s11_db_loss = self.mse_loss(pred_abs_db, target_abs_db)
        s11_db_loss_mae = self.MAE_loss(pred_abs_db, target_abs_db)
        db_weight = self.db_weight * min(1.0, self._step / self.warmup)  # Linear warmup for the dB loss
        s11_complex_weight = min(1.0, self._step / self.warmup)
        loss = s11_complex_weight * s11_complex_loss + db_weight * s11_db_loss
        self._step += 1
        if return_db_loss:
            return {'complex_loss': s11_complex_loss, 'db_loss': s11_db_loss, 'db_loss_mae': s11_db_loss_mae, 'loss': loss}
        return loss


class FarfeildLoss(nn.Module):
    def __init__(self, config, ff_stats):
        super(FarfeildLoss, self).__init__()
        self.ff_stats = ff_stats
        self.mssim_data_range = 8.865
        self.scale_to_DB = config.get('output_in_DB')

    def forward(self, pred, target, train=True):
        if target.shape != pred.shape:
            target = torch.nn.functional.interpolate(target, size=pred.shape[2:], mode='bilinear', align_corners=False)
        mse_loss = nn.MSELoss()(pred, target)
        if train == True:
            normalized_farfeild_loss_value = normalized_farfeild_loss(pred)
        else:
            normalized_farfeild_loss_value = 0.0
        loss = mse_loss + 0.1*normalized_farfeild_loss_value
        return loss


def compute_metrics(prediction: torch.Tensor, ground_truth: torch.Tensor, ff_stats: dict, clip=False):
    """calculats the metrics between the pridictionn and the ground truth
    contains MAE,MSE, max avarge error, snr, mssim, and more to come...
    [batchsize,C,H,W]
    Args:
        prediction (torch.Tensor): ff_pred in db
        ground_truth (torch.Tensor):ff_target in db
    """
    if clip:
        min, max = [-15, 5]
        prediction = torch.clamp(prediction, min, max)
        ground_truth = torch.clamp(ground_truth, min, max)

    if ground_truth.shape != prediction.shape:
        ground_truth = torch.nn.functional.interpolate(ground_truth, size=prediction.shape[2:], mode='bilinear', align_corners=False)

    abs_error = abs(ground_truth-prediction)
    mae = abs_error.mean()
    mse = MSELoss()(prediction, ground_truth)
    max_error = abs_error.max()
    snr = compute_snr(prediction.squeeze(-1), ground_truth.squeeze(-1))
    mssim = 1 - MSSSIMLoss()(prediction, ground_truth, ff_stats)
    return {'MAE': mae, 'MSE': mse, 'max_error': max_error, 'SNR': snr, 'mssim': mssim}


"""_summary_
    MSSSIMLoss class
    Compute MultiScaleSSIM, Multi-scale Structural Similarity Index Measure.
    This metric is is a generalization of Structural Similarity Index Measure by incorporating image details at different resolution scores.
"""
class MSSSIMLoss(nn.Module):
    def __init__(self):
        super(MSSSIMLoss, self).__init__()

    def forward(self, prediction, target_image, ff_stats=None, win_size=3):
        # Calculate MS-SSIM (loss change to image shape [batchsize,C,H,W]
        prediction = prediction.clone()
        target_image = target_image.clone()
        prediction += ff_stats['min']
        target_image += ff_stats['min']
        data_range = ff_stats['min'] + ff_stats['max']
        loss = 1 - ms_ssim(prediction, target_image, data_range=data_range, win_size=win_size)
        return loss


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, target):
        return self.mse_loss(pred, target)


def compute_snr(prediction: torch.Tensor, ground_truth: torch.Tensor) -> float:
    """
    Calculate the Signal-to-Noise Ratio (SNR) between the predicted and ground truth images.

    Args:
        prediction (torch.Tensor): The predicted far-field image.
        ground_truth (torch.Tensor): The ground truth far-field image.

    Returns:
        float: The SNR value in decibels (dB).
    """
    # Ensure both tensors have the same shape
    assert prediction.shape == ground_truth.shape, "Prediction and ground truth must have the same shape."

    # input is in dB so we scale back to linear scale
    ground_truth, prediction = unscale_from_log(ground_truth), unscale_from_log(prediction)

    # Compute the signal (mean squared value of the ground truth)
    signal = torch.mean(ground_truth ** 2)

    # Compute the noise (mean squared error between prediction and ground truth)
    noise = torch.mean((prediction - ground_truth) ** 2)

    # Compute SNR in decibels
    snr = 10 * torch.log10(signal / noise)

    return snr.item()
