"""Compatibility shim for the nnU-Net v2 Stage-1 checkpoint."""

from nnxnet.training.nnXNetTrainer.nnXNetTrainer import nnXNetTrainer


class nnUNetTrainer(nnXNetTrainer):
    """Build the standard nnU-Net architecture using nnXNet's compatible API."""

    pass
