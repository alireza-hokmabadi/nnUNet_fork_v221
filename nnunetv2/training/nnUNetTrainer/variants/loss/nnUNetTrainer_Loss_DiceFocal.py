import torch

from nnunetv2.training.loss.dice_loss import DC_and_Focal_loss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


# class nnUNetTrainer_Loss_DiceFocal(nnUNetTrainer):
#     def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, **kwargs):
#         super().__init__(plans, configuration, fold, dataset_json, **kwargs)

#         self.loss = DC_and_Focal_loss(
#             {'batch_dice': self.configuration.batch_dice, 'smooth': 1e-5, 'do_bg': False},
#             {'alpha': 0.5, 'gamma': 2, 'smooth': 1e-5}
#         )


# class nnUNetTrainerV2_Loss_DiceFocal(nnUNetTrainerV2):
#     def __init__(self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
#                  unpack_data=True, deterministic=True, fp16=False):
#         super().__init__(plans_file, fold, output_folder, dataset_directory, batch_dice, stage,
#                                               unpack_data, deterministic, fp16)

#         self.loss = DC_and_Focal_loss({'batch_dice':self.batch_dice, 'smooth':1e-5,
#         	'do_bg':False}, {'alpha':0.5, 'gamma':2, 'smooth':1e-5})




class nnUNetTrainer_Loss_DiceFocal(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        self.loss = DC_and_Focal_loss(
            {'batch_dice': self.configuration.batch_dice, 'smooth': 1e-5, 'do_bg': False},
            {'alpha': 0.5, 'gamma': 2, 'smooth': 1e-5}
        )

