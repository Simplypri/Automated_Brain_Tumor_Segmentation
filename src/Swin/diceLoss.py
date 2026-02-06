import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        """
        Args:
            preds: (B, C, H, W) — raw logits
            targets: (B, H, W) or (H, W) — integer class labels
        """
        preds = F.softmax(preds, dim=1)
        num_classes = preds.shape[1]

        # Ensure targets has batch dimension
        if targets.ndim == 2:
            targets = targets.unsqueeze(0)  # (1, H, W)

        # One-hot encode targets
        targets_onehot = F.one_hot(targets.long(), num_classes=num_classes)  # (B, H, W, C)
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()          # (B, C, H, W)

        # Flatten to (B, C, H*W)
        preds_flat = preds.contiguous().view(preds.shape[0], preds.shape[1], -1)
        targets_flat = targets_onehot.contiguous().view(targets_onehot.shape[0], targets_onehot.shape[1], -1)

        intersection = (preds_flat * targets_flat).sum(-1)
        denominator = preds_flat.sum(-1) + targets_flat.sum(-1)
        dice_score = (2 * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1 - dice_score.mean()

        return dice_loss
