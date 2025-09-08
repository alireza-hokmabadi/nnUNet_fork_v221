
import numpy as np
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice_focal_loss import DiceFocalBCELoss  # Import the custom loss


class nnUNetTrainerFocalDiceBCELoss(nnUNetTrainer):
    def _build_loss(self):
        self.print_to_log_file("Using Dice + BCE + Focal Loss for training.")

        # Define the loss function
        loss = DiceFocalBCELoss()  # Adjust parameters if needed

        deep_supervision_scales = self._get_deep_supervision_scales()
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights[-1] = 0

        # Normalize weights
        weights = weights / weights.sum()

        # Wrap with Deep Supervision
        loss = DeepSupervisionWrapper(loss, weights)

        return loss