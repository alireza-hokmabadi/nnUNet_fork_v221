
import numpy as np
import torch
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.loss.dice_focal_loss import DiceFocalBCELoss  # Import the custom loss


class nnUNetTrainerFocalDiceBCELoss(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda'),
                 dice_weight: float = 0.5,
                 bce_weight: float = 0.25,
                 focal_weight: float = 0.25):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight

    def _build_loss(self):
        self.print_to_log_file(
            f"Using Dice + BCE + Focal Loss. "
            f"Weights: Dice={self.dice_weight}, BCE={self.bce_weight}, Focal={self.focal_weight}"
        )

        if self.label_manager.has_regions:
            loss = DiceFocalBCELoss(
                use_regions=True,
                use_ignore_label=self.label_manager.ignore_label is not None,
                dice_weight=self.dice_weight,
                bce_weight=self.bce_weight,
                focal_weight=self.focal_weight
            )
        else:
            loss = DiceFocalBCELoss(
                use_regions=False,
                use_ignore_label=self.label_manager.ignore_label is not None,
                dice_weight=self.dice_weight,
                bce_weight=self.bce_weight,
                focal_weight=self.focal_weight
            )

        deep_supervision_scales = self._get_deep_supervision_scales()
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights[-1] = 0
        weights = weights / weights.sum()

        return DeepSupervisionWrapper(loss, weights)


class nnUNetTrainerFocalDiceBCELoss_250epochs(nnUNetTrainerFocalDiceBCELoss):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 250


class nnUNetTrainerFocalDiceBCELoss_500epochs(nnUNetTrainerFocalDiceBCELoss):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 500


class nnUNetTrainerFocalDiceBCELoss_1000epochs(nnUNetTrainerFocalDiceBCELoss):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 1000


class nnUNetTrainerFocalDiceBCELoss_4000epochs(nnUNetTrainerFocalDiceBCELoss):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 4000


class nnUNetTrainerFocalDiceBCELoss_8000epochs(nnUNetTrainerFocalDiceBCELoss):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 8000


class nnUNetTrainerFocalDiceBCELoss_10000epochs(nnUNetTrainerFocalDiceBCELoss):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 10000


class nnUNetTrainerFocalDiceBCELoss_4000epochs_CustomWeights1(nnUNetTrainerFocalDiceBCELoss):
    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device,
                         dice_weight=0.4, bce_weight=0.2, focal_weight=0.4)
        self.num_epochs = 4000


class nnUNetTrainerFocalDiceBCELoss_8000epochs_CustomWeights1(nnUNetTrainerFocalDiceBCELoss):
    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device,
                         dice_weight=0.4, bce_weight=0.2, focal_weight=0.4)
        self.num_epochs = 8000


class nnUNetTrainerFocalDiceBCELoss_10000epochs_CustomWeights1(nnUNetTrainerFocalDiceBCELoss):
    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device,
                         dice_weight=0.4, bce_weight=0.2, focal_weight=0.4)
        self.num_epochs = 10000
