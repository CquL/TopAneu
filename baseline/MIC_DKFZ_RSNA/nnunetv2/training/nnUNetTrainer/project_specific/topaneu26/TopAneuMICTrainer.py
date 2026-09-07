import torch

from nnunetv2.training.nnUNetTrainer.project_specific.kaggle2025_rsna.Kaggle2025RSNATrainer import (
    Kaggle2025RSNATrainer,
)


class TopAneuMICTrainer(Kaggle2025RSNATrainer):
    """MIC-DKFZ heatmap trainer adapted to the 52 TopAneu location labels."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        # nnU-Net uses 250 optimizer steps per epoch. At 250 epochs this gives
        # 62,500 updates, a practical first schedule for the 416-case dataset.
        self.num_epochs = 250
        # Preserve the MIC-DKFZ RSNA heatmap target setting.
        self.blobb_radius = 65
