
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


# class DiceFocalBCELoss(nn.Module):
#     def __init__(self, alpha=0.35, gamma=3, smooth=1e-6,
#                  use_regions=False, use_ignore_label=False,
#                  dice_weight=0.5, bce_weight=0.25, focal_weight=0.25):
#         super(DiceFocalBCELoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.smooth = smooth
#         self.use_regions = use_regions
#         self.use_ignore_label = use_ignore_label
#         self.dice_weight = dice_weight
#         self.bce_weight = bce_weight
#         self.focal_weight = focal_weight
#         self.bce = nn.BCEWithLogitsLoss(reduction='none' if use_ignore_label else 'mean')

#     def forward(self, inputs, targets):
#         """
#         inputs: [B, C, ...] logits from network
#         targets: one-hot or region-based [B, C, ...]
#         """
#         if self.use_ignore_label:
#             mask = (1 - targets[:, -1:]).bool()
#             targets = targets[:, :-1]  # drop ignore channel
#         else:
#             mask = None

#         # ---- Dice Loss ----
#         probs = torch.sigmoid(inputs)
#         intersection = (probs * targets).sum(dim=list(range(2, targets.ndim)))
#         dice_coeff = (2 * intersection + self.smooth) / (
#             probs.sum(dim=list(range(2, targets.ndim))) + targets.sum(dim=list(range(2, targets.ndim))) + self.smooth
#         )
#         dice_loss = 1 - dice_coeff.mean()

#         # ---- BCE Loss ----
#         if mask is not None:
#             bce_raw = self.bce(inputs, targets)
#             bce_loss = (bce_raw * mask).sum() / torch.clamp(mask.sum(), min=1e-8)
#         else:
#             bce_loss = self.bce(inputs, targets)

#         # ---- Focal Loss ----
#         bce_each = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
#         pt = torch.exp(-bce_each)
#         focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_each
#         if mask is not None:
#             focal_loss = (focal_loss * mask).sum() / torch.clamp(mask.sum(), min=1e-8)
#         else:
#             focal_loss = focal_loss.mean()

#         return (
#             self.dice_weight * dice_loss +
#             self.bce_weight * bce_loss +
#             self.focal_weight * focal_loss
#         )