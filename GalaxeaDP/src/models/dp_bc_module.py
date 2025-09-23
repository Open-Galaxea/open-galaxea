from __future__ import annotations

from typing import Any

import torch
from lightning import LightningModule

from src import utils as U
from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class DiffusionPolicyBCModule(LightningModule):
    def __init__(
        self,
        policy,
        optimizer,
        lr_scheduler,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(
            logger=False,
            ignore=["policy"],
        )
        self.policy = policy

    def setup(self, stage: str) -> None:
        self.policy.set_normalizer(self.trainer.datamodule.data_train.get_normalizer())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.policy.compute_loss(x)
        else:
            return self.policy.predict_action(x)

    def model_step(
        self, batch: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.training:
            return self.policy.compute_loss(batch)
        else:
            return self.policy.predict_action(batch)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        super().on_train_start()
    
    def on_train_epoch_start(self) -> None:
        self.policy.current_epoch = self.current_epoch

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, loss_log = self.model_step(batch)

        # update and log metrics
        self.log_dict(
            {"train/_loss": loss.detach()}, 
            on_step=True, 
            on_epoch=True, 
            prog_bar=True, 
            sync_dist=True, 
        )

        self.log_dict(
            loss_log, 
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        
        # return loss or backpropagation will fail
        return loss

    def on_validation_epoch_start(self) -> None:
        return

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        return

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        return
    
    def test_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        raise NotImplementedError

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = self.policy.get_optimizer(self.hparams.optimizer)
        assert self.check_params_in_optimizer(optimizer)

        self.hparams.lr_scheduler.scheduler.total_steps = (
            self.trainer.estimated_stepping_batches
        )
        scheduler = U.build_scheduler(
            self.hparams.lr_scheduler.scheduler, optimizer=optimizer
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.hparams.lr_scheduler.get("monitor", "val/loss"),
                "interval": self.hparams.lr_scheduler.get("interval", "step"),
                "frequency": self.hparams.lr_scheduler.get("frequency", 1),
            },
        }

    def check_params_in_optimizer(self, optimizer):
        model_param_ids = {id(p) for p in self.policy.parameters() if p.requires_grad}
        optim_param_ids = set()
        for group in optimizer.param_groups:
            for p in group['params']:
                optim_param_ids.add(id(p))
        
        missing_ids = model_param_ids - optim_param_ids
        if missing_ids:
            missing_params = [p for p in self.policy.parameters() if id(p) in missing_ids]
            log.info("The following parameters are not tracked by optimizer:")
            for p in missing_params:
                log.info(f" - name={next((n for n, pp in self.policy.named_parameters() if pp is p), '?')}, "
                    f"shape={tuple(p.shape)}, device={p.device}")
            return False

        return True