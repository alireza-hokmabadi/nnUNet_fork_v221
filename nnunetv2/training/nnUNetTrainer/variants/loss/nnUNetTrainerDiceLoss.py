import numpy as np
import torch

from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import softmax_helper_dim1

#%% ---------------------------------------------------------------------------
class nnUNetTrainerDiceLoss(nnUNetTrainer):
    def _build_loss(self):
        loss = MemoryEfficientSoftDiceLoss(
            **{
                'batch_dice': self.configuration_manager.batch_dice,
                'do_bg': self.label_manager.has_regions,
                'smooth': 1e-5,
                'ddp': self.is_ddp
            },
            apply_nonlin=torch.sigmoid if self.label_manager.has_regions else softmax_helper_dim1
        )

        deep_supervision_scales = self._get_deep_supervision_scales()
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights[-1] = 0
        weights = weights / weights.sum()

        loss = DeepSupervisionWrapper(loss, weights)
        return loss

#%% ---------------------------------------------------------------------------
class nnUNetTrainerDiceCELoss_noSmooth(nnUNetTrainer):
    def _build_loss(self):
        # --- set smooth to 0 ---
        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss(
                {},
                {
                    'batch_dice': self.configuration_manager.batch_dice,
                    'do_bg': True,
                    'smooth': 0,
                    'ddp': self.is_ddp
                },
                weight_ce=1,
                weight_dice=1,
                use_ignore_label=self.label_manager.ignore_label is not None,
                dice_class=MemoryEfficientSoftDiceLoss
            )
        else:
            loss = DC_and_CE_loss(
                {},
                {
                    'batch_dice': self.configuration_manager.batch_dice,
                    'do_bg': False,
                    'smooth': 0,
                    'ddp': self.is_ddp
                },
                weight_ce=1,
                weight_dice=1,
                ignore_label=self.label_manager.ignore_label,
                dice_class=MemoryEfficientSoftDiceLoss
            )

        deep_supervision_scales = self._get_deep_supervision_scales()
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights[-1] = 0
        weights = weights / weights.sum()

        loss = DeepSupervisionWrapper(loss, weights)
        return loss


class nnUNetTrainerDiceCELoss_noSmooth_100epochs(nnUNetTrainerDiceCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 100

class nnUNetTrainerDiceCELoss_noSmooth_250epochs(nnUNetTrainerDiceCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 250

class nnUNetTrainerDiceCELoss_noSmooth_500epochs(nnUNetTrainerDiceCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 500

class nnUNetTrainerDiceCELoss_noSmooth_2000epochs(nnUNetTrainerDiceCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 2000

class nnUNetTrainerDiceCELoss_noSmooth_4000epochs(nnUNetTrainerDiceCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 4000

class nnUNetTrainerDiceCELoss_noSmooth_8000epochs(nnUNetTrainerDiceCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 8000


#%% ---------------------------------------------------------------------------
class nnUNetTrainerDiceCELoss_noSmooth_2000epochs_NoMirroring(nnUNetTrainerDiceCELoss_noSmooth):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 2000

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes


#%% ---------------------------------------------------------------------------
class nnUNetTrainerFlexibleLoss(nnUNetTrainer):

    # ---- default experiment parameters ----
    class_weights = None
    weight_ce = 1.0
    weight_dice = 1.0
    num_epochs = 1000  # default
    oversample_foreground_percent = 0.33  # default

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = self.__class__.num_epochs
        self.oversample_foreground_percent = self.__class__.oversample_foreground_percent

    def _build_loss(self):

        # prepare CE/BCE kwargs
        if self.class_weights is not None:
            ce_kwargs = {
                'weight': torch.tensor(
                    self.class_weights,
                    dtype=torch.float32,
                    device=self.device
                )
            }
        else:
            ce_kwargs = {}

        if self.label_manager.has_regions:
            loss = DC_and_BCE_loss(
                ce_kwargs,
                {
                    'batch_dice': self.configuration_manager.batch_dice,
                    'do_bg': True,
                    'smooth': 0,
                    'ddp': self.is_ddp
                },
                weight_ce=self.weight_ce,
                weight_dice=self.weight_dice,
                use_ignore_label=self.label_manager.ignore_label is not None,
                dice_class=MemoryEfficientSoftDiceLoss
            )
        else:
            loss = DC_and_CE_loss(
                {
                    'batch_dice': self.configuration_manager.batch_dice,
                    'do_bg': False,
                    'smooth': 0,
                    'ddp': self.is_ddp
                },
                ce_kwargs,
                weight_ce=self.weight_ce,
                weight_dice=self.weight_dice,
                ignore_label=self.label_manager.ignore_label,
                dice_class=MemoryEfficientSoftDiceLoss
            )

        deep_supervision_scales = self._get_deep_supervision_scales()
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights[-1] = 0
        weights = weights / weights.sum()

        return DeepSupervisionWrapper(loss, weights)


class nnUNetTrainerFlexibleLoss_ep100(nnUNetTrainerFlexibleLoss):
    class_weights = None
    weight_ce = 1.0
    weight_dice = 1.0
    num_epochs = 100
    oversample_foreground_percent = 0.33

class nnUNetTrainerFlexibleLoss_ep250(nnUNetTrainerFlexibleLoss):
    class_weights = None
    weight_ce = 1.0
    weight_dice = 1.0
    num_epochs = 250
    oversample_foreground_percent = 0.33

class nnUNetTrainerFlexibleLoss_ep500(nnUNetTrainerFlexibleLoss):
    class_weights = None
    weight_ce = 1.0
    weight_dice = 1.0
    num_epochs = 500
    oversample_foreground_percent = 0.33

class nnUNetTrainerFlexibleLoss_ep100_ce2(nnUNetTrainerFlexibleLoss):
    class_weights = None
    weight_ce = 2.0
    weight_dice = 1.0
    num_epochs = 100
    oversample_foreground_percent = 0.33

class nnUNetTrainerFlexibleLoss_ep250_ce2(nnUNetTrainerFlexibleLoss):
    class_weights = None
    weight_ce = 2.0
    weight_dice = 1.0
    num_epochs = 250
    oversample_foreground_percent = 0.33

class nnUNetTrainerFlexibleLoss_ep500_ce2(nnUNetTrainerFlexibleLoss):
    class_weights = None
    weight_ce = 2.0
    weight_dice = 1.0
    num_epochs = 500
    oversample_foreground_percent = 0.33

class nnUNetTrainerFlexibleLoss_ep1000_ce2(nnUNetTrainerFlexibleLoss):
    class_weights = None
    weight_ce = 2.0
    weight_dice = 1.0
    num_epochs = 1000
    oversample_foreground_percent = 0.33


class nnUNetTrainerFlexibleLoss_ep250_ce2__class_weights(nnUNetTrainerFlexibleLoss):
    class_weights = [1.0, 2.0, 1., 2.5]  # {0: background, 1: LV_Endo, 2: LV_Ep}
    weight_ce = 2.0
    weight_dice = 1.0
    num_epochs = 250
    oversample_foreground_percent = 0.5


class nnUNetTrainerFlexibleLoss_ep250_strongSampling(nnUNetTrainerFlexibleLoss):
    class_weights = None
    weight_ce = 1.0
    weight_dice = 1.0
    num_epochs = 250
    oversample_foreground_percent = 0.66

#%% ---------------------------------------------------------------------------

