import torch
import numpy as np
from sklearn.metrics import f1_score


def calculate_IOU(ref, pred):
    """
    ref, pred: tensors of shape (B, H, W)
    Binary (0,1)
    """

    # Compute intersection and union for each sample
    B = ref.shape[0]
    ious = []

    for i in range(B):
        r = ref[i]
        p = pred[i]

        intersection = torch.sum(r * p)
        union = torch.sum((r + p) >= 1)

        iou = (intersection + 1e-6) / (union + 1e-6)
        ious.append(iou.item())

    return np.mean(ious)


def calculate_F1(ref, pred):
    """
    ref, pred: tensors of shape (B, H, W)
    """
    ref = ref.cpu().numpy().ravel()
    pred = pred.cpu().numpy().ravel()
    return f1_score(ref, pred)


class metric_calculation:
    def __init__(self):
        self.iou = []
        self.f1 = []

    def compute_metrics(self, ref, pred):

        # Convert logits/probabilities → binary
        pred_bin = (pred > 0.5).long()

        # Remove channel dimension if needed
        if ref.dim() == 4:
            ref = ref.squeeze(1)
        if pred_bin.dim() == 4:
            pred_bin = pred_bin.squeeze(1)

        iou_val = calculate_IOU(ref, pred_bin)
        f1_val = calculate_F1(ref, pred_bin)

        self.iou.append(iou_val)
        self.f1.append(f1_val)

    def reset(self):
        self.iou = []
        self.f1 = []

    def get_metrics(self):
        return {
            "iou": float(np.mean(self.iou)),
            "f1": float(np.mean(self.f1)),
        }