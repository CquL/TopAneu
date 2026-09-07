"""Compatibility shim for the published nnU-Net v2 Stage-1 checkpoint."""

from nnxnet.training.nnXNetTrainer.nnXNetTrainer import nnXNetTrainer


class nnUNetTrainer(nnXNetTrainer):
    """Build the standard nnU-Net architecture through nnXNet's compatible API."""

    pass
