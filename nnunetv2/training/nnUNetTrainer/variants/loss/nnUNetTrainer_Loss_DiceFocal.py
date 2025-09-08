
from nnunetv2.training.loss.dice_loss import DC_and_Focal_loss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_Loss_DiceFocal(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, **kwargs):
        super().__init__(plans, configuration, fold, dataset_json, **kwargs)

        self.loss = DC_and_Focal_loss(
            {'batch_dice': self.configuration.batch_dice, 'smooth': 1e-5, 'do_bg': False},
            {'alpha': 0.5, 'gamma': 2, 'smooth': 1e-5}
        )
