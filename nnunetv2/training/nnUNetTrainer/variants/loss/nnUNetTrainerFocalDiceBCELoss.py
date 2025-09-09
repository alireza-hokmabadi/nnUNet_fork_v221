
import numpy as np
import torch
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


class nnUNetTrainerFocalDiceBCELoss_4000epochs(nnUNetTrainerFocalDiceBCELoss):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 4000