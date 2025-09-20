
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: logits (B, ...) single channel
        targets: binary mask (B, ...)
        """
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        inputs: probabilities (B, ...) single channel
        targets: binary mask (B, ...)
        """
        intersection = (inputs * targets).sum()
        dice_coeff = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice_coeff


class DiceFocalBCELoss(nn.Module):
    def __init__(self, alpha=0.35, gamma=3,
                 use_regions=False, use_ignore_label=False,
                 dice_weight=0.5, bce_weight=0.25, focal_weight=0.25):
        super().__init__()
        self.use_regions = use_regions
        self.use_ignore_label = use_ignore_label

        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight

        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha, gamma)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        """
        inputs: logits (B, C, ...)
        targets:
          - if use_regions=True → one-hot (B, C, ...)
          - else → integer mask (B, 1, ...)
        """
        if self.use_regions:
            # Region-based → multi-channel one-hot
            targets = targets.float()          # (B, C, ...)
            bce = self.bce_loss(inputs, targets)

            probs = torch.sigmoid(inputs)      # (B, C, ...)
            dice = self._multiclass_dice(probs, targets)
            focal = self._multiclass_focal(inputs, targets)
        else:
            # Binary → use only foreground channel
            inputs_fg = inputs[:, 1, ...]       # (B, ...)
            targets = targets.squeeze(1).float()

            probs = torch.sigmoid(inputs_fg)
            dice = self.dice_loss(probs, targets)
            bce = self.bce_loss(inputs_fg, targets)
            focal = self.focal_loss(inputs_fg, targets)

        return self.dice_weight * dice + self.bce_weight * bce + self.focal_weight * focal

    def _multiclass_dice(self, probs, targets):
        # Compute Dice per channel and average
        dims = tuple(range(2, probs.ndim))
        intersection = (probs * targets).sum(dims)
        dice_per_class = (2. * intersection + self.dice_loss.smooth) / (
            probs.sum(dims) + targets.sum(dims) + self.dice_loss.smooth
        )
        return 1 - dice_per_class.mean()

    def _multiclass_focal(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal = (0.25 * (1 - pt) ** 2 * bce_loss).mean()  # default alpha=0.25, gamma=2
        return focal
