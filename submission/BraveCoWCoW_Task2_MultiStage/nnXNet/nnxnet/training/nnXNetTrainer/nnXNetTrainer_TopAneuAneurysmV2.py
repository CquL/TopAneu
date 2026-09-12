"""Fold-0 pilot: staged fine-tuning and deterministic full-ROI selection."""
import json
import os
from time import time

import torch
from torch._dynamo import OptimizedModule

from nnxnet.training.nnXNetTrainer.nnXNetTrainer_TopAneuAneurysmOnly import (
    nnXNetTrainer_TopAneuAneurysmOnly,
)


class nnXNetTrainer_TopAneuAneurysmV2(nnXNetTrainer_TopAneuAneurysmOnly):
    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset, device)
        self.warmup_epochs = 20
        self.full_validation_interval = 10
        self.initial_lr = 1e-3

    def _module(self):
        return self.network._orig_mod if isinstance(self.network, OptimizedModule) else self.network

    def configure_optimizers(self):
        if self.is_ddp:
            raise RuntimeError("AneurysmV2 pilot supports one GPU per process only")
        module = self._module()
        branch = [p for name in ("transpconvs_last_two_2", "decoder_blocks_last_two_2", "seg_layers_2")
                  for p in getattr(module, name).parameters()]
        high = [p for blocks in (module.conv_encoder_blocks[-2:],
                                module.transpconvs[-2:], module.decoder_blocks[-2:])
                for p in blocks.parameters()]
        if set(map(id, branch)) & set(map(id, high)):
            raise RuntimeError("Optimizer parameter groups overlap")
        optimizer = torch.optim.SGD([
            {"params": branch, "lr": 1e-3},
            {"params": high, "lr": 1e-4},
        ], momentum=0.99, nesterov=True, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: max(0., 1. - epoch / self.num_epochs) ** 0.9)
        return optimizer, scheduler

    def _set_training_phase(self):
        for p in self._module().parameters():
            p.requires_grad_(False)
        for p in self.optimizer.param_groups[0]["params"]:
            p.requires_grad_(True)
        for p in self.optimizer.param_groups[1]["params"]:
            p.requires_grad_(self.current_epoch >= self.warmup_epochs)

    def initialize(self):
        super().initialize()
        self._set_training_phase()

    def on_train_epoch_start(self):
        self._set_training_phase()
        super().on_train_epoch_start()
        self.print_to_log_file(
            "V2 phase:", "branch only" if self.current_epoch < self.warmup_epochs else "high-level fine-tuning",
            "group learning rates:", [g["lr"] for g in self.optimizer.param_groups])

    def _validate_for_selection(self):
        # Reuse the same no-augmentation, full-ROI inference as final validation.
        original_folder = self.output_folder
        was_training = self.network.training
        self.output_folder = os.path.join(original_folder, "selection_validation")
        os.makedirs(self.output_folder, exist_ok=True)
        try:
            super().perform_actual_validation()
            with open(os.path.join(self.output_folder, "validation_aneurysm_summary.json")) as handle:
                summary = json.load(handle)
        finally:
            self.output_folder = original_folder
            self.set_deep_supervision_enabled(self.enable_deep_supervision)
            self.network.train(was_training)
        summary["completed_epochs"] = self.current_epoch + 1
        summary["selection_metric"] = "full_roi_micro_dice_at_argmax"
        with open(os.path.join(original_folder, "full_validation_history.jsonl"), "a") as handle:
            handle.write(json.dumps(summary) + "\n")
        score = summary["dice"]
        if score is not None and (self._best_ema is None or score > self._best_ema):
            # Keep the upstream checkpoint field for resume compatibility, but
            # V2 stores full-ROI Dice here, never the patch EMA.
            self._best_ema = score
            self.save_checkpoint(os.path.join(original_folder, "checkpoint_best.pth"))
            with open(os.path.join(original_folder, "best_full_validation.json"), "w") as handle:
                json.dump(summary, handle, indent=2)
            self.print_to_log_file("V2 new best full-ROI Dice:", score)

    def on_epoch_end(self):
        if (self.current_epoch + 1) % self.full_validation_interval == 0 or self.current_epoch == self.num_epochs - 1:
            self._validate_for_selection()
        self.logger.log("epoch_end_timestamps", time(), self.current_epoch)
        for label, key in (("train_loss", "train_losses"), ("val_loss", "val_losses"),
                           ("Pseudo dice (diagnostic only)", "dice_per_class_or_region")):
            self.print_to_log_file(label, self.logger.my_fantastic_logging[key][-1])
        if (self.current_epoch + 1) % self.save_every == 0 and self.current_epoch != self.num_epochs - 1:
            self.save_checkpoint(os.path.join(self.output_folder, "checkpoint_latest.pth"))
        self.logger.plot_progress_png(self.output_folder)
        self.current_epoch += 1
