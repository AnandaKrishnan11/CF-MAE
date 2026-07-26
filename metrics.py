import torch


class metric_calculation:
    """Global confusion matrix over the whole evaluation set.

    Metrics are computed for the change class (positive class).
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0   
        self.fp = 0   
        self.fn = 0 
        
    @torch.no_grad()
    def compute_metrics(self, ref, pred):
        """ref, pred: matching shapes, binary values (already thresholded)."""
        if ref.shape != pred.shape:
            raise ValueError(f"shape mismatch: ref {tuple(ref.shape)} vs pred {tuple(pred.shape)}")

        ref = (ref > 0.5).flatten()
        pred = (pred > 0.5).flatten()

        self.tp += torch.count_nonzero(ref & pred).item()
        self.fp += torch.count_nonzero(~ref & pred).item()
        self.fn += torch.count_nonzero(ref & ~pred).item()

    def get_metrics(self):
        eps = 1e-9
        tp, fp, fn = self.tp, self.fp, self.fn

        f1 = 2 * tp / (2 * tp + fp + fn + eps)
        iou = tp / (tp + fp + fn + eps)

        return {"f1": float(f1), "iou": float(iou)}