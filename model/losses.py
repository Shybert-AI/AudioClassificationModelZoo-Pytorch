import torch
import torch.nn as nn


class BCELoss(nn.Module):
    """docstring for BCELoss"""
    def __init__(self):
        super().__init__()
        self.name = "BCELoss"

    def forward(self, clip_prob, tar):
        return nn.functional.binary_cross_entropy(input=clip_prob, target=tar.float())
        #return nn.BCEWithLogitsLoss(clip_prob, tar.float())

class BCELossWithLabelSmoothing(nn.Module):
    """docstring for BCELoss"""
    def __init__(self, label_smoothing=0.1):
        super().__init__()
        self.label_smoothing = label_smoothing

    def forward(self, clip_prob, frame_prob, tar):
        n_classes = clip_prob.shape[-1]
        with torch.no_grad():
            tar = tar * (1 - self.label_smoothing) + (
                1 - tar) * self.label_smoothing / (n_classes - 1)
        return nn.functional.binary_cross_entropy(clip_prob, tar)




