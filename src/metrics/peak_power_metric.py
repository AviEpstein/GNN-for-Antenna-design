import torch


def calculate_HPBW(predicted_farfield: torch.Tensor, ground_truth_farfield: torch.Tensor):
    """
    half power beamwidth (HPBW) metric for farfield patterns. This metric measures the angular width of the main lobe of the farfield pattern at half of its maximum power level.
    The steps to calculate this metric are as follows:
    1. Find the max value in the ground truth farfield
    2. Set a threshold at: max value - 3 ( this corisponds to half power point in linear scale, or -3dB point in log scale)
    3. Threshold the ground truth farfield (creating a mask of 1s and 0s)
    4. Threshold the predicted farfield using the same threshold
    5. Find the angle difference between the two thresholded farfields (this is the HPBW)
    """
    # 1. Find the max value in the ground truth farfield
    max_val = torch.max(ground_truth_farfield)

    # 2. Set a threshold at: max value -  max_val * 0.5 ( this corisponds to half power point in linear scale, or -3dB point in log scale)
    threshold = max_val - max_val *0.5

    # 3. Threshold the ground truth farfield (creating a mask of 1s and 0s)
    gt_mask = (ground_truth_farfield >= threshold).float()
    pred_mask = (predicted_farfield >= threshold).float()
    intersection_mask = gt_mask * pred_mask

    result = intersection_mask.sum() / (gt_mask.sum() + 1e-8)  # Avoid division by zero, return ratio of predicted to ground truth above threshold

    return result

def calculate_Boresight_error(predicted_farfield: torch.Tensor, ground_truth_farfield: torch.Tensor):
    """
    Finds the angle of the max value in the ground truth farfield, and the angle of the max value in the predicted farfield, and returns the absolute difference between them.
    """
    # 1. Find the angle of the max value in the ground truth farfield
    gt_max_idx = torch.argmax(ground_truth_farfield)
    gt_angle = gt_max_idx.item()  # Assuming angles correspond to indices

    # 2. Find the angle of the max value in the predicted farfield
    pred_max_idx = torch.argmax(predicted_farfield)
    pred_angle = pred_max_idx.item()  # Assuming angles correspond to indices

    # 3. Calculate the absolute difference between the two angles
    boresight_error = abs(gt_angle - pred_angle)

    return boresight_error
