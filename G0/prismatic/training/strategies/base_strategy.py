"""
base_strategy.py

Abstract class definition of a (distributed) training strategy, with full annotations of class methods, utility
functions, and initialization logic.

Training Strategies (DDP, FSDP-Grad, FSDP-Full) tend to have a lot of repeated components; this class does a lot of
heavy lifting.
"""

from contextlib import nullcontext
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tqdm import tqdm
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.vlms import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.training.metrics import Metrics, VLAMetrics
from prismatic.util import check_bloat16_supported
from prismatic.util.batching_utils import SplitModalitySampler
from prismatic.util.data_utils import PaddedCollatorForActionPrediction, PaddedCollatorForLanguageModeling
from prismatic.vla.action_tokenizer import ActionTokenizer

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

import signal
def handle_resize(signum, frame):
    tqdm.tqdm._instances.clear()
    for instance in tqdm.tqdm._instances:
        instance.refresh()

signal.signal(signal.SIGWINCH, handle_resize)

# === Abstract Base Class for an arbitrary Training Strategy ===
class TrainingStrategy(ABC):
    def __init__(
        self,
        vlm: PrismaticVLM,
        device_id: int,
        stage: str,
        epochs: int,
        max_steps: Optional[int],
        global_batch_size: int,
        per_device_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        lr_scheduler_type: str,
        warmup_ratio: float,
        enable_gradient_checkpointing: bool = True,
        enable_mixed_precision_training: bool = True,
        reduce_in_full_precision: bool = False,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        **_: str,
    ) -> None:
        self.vlm, self.device_id, self.stage = vlm, device_id, stage

        # Get relevant VLM instance parameters before they get (potentially) wrapped
        self.all_module_keys, self.trainable_module_keys = self.vlm.all_module_keys, self.vlm.trainable_module_keys
        self.llm_transformer_layer_cls = self.vlm.transformer_layer_cls

        # Optimization Parameters
        self.epochs, self.max_steps = epochs, max_steps
        self.global_batch_size, self.per_device_batch_size = global_batch_size, per_device_batch_size

        self.learning_rate, self.weight_decay, self.max_grad_norm = learning_rate, weight_decay, max_grad_norm
        self.lr_scheduler_type, self.warmup_ratio = lr_scheduler_type, warmup_ratio

        # Generic Strategy Parameters
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.reduce_in_full_precision = reduce_in_full_precision
        self.mixed_precision_dtype = mixed_precision_dtype

        # DataLoader Parameters
        self.worker_init_fn = worker_init_fn

        # Optimizers & Scheduler (initialized in `run_setup`)
        self.optimizer, self.lr_scheduler = None, None

        # Lightweight Validation
        assert (
            self.global_batch_size % self.per_device_batch_size == 0
        ), "Per-device batch size must evenly divide global batch size!"
        self.grad_accumulation_steps = self.global_batch_size // self.per_device_batch_size // overwatch.world_size()
        if self.enable_mixed_precision_training:
            assert self.mixed_precision_dtype == torch.bfloat16, "Only BF16 mixed precision training is supported!"
            assert check_bloat16_supported(), "BFloat16 is not supported on this hardware; unset `mixed_precision`"

    @abstractmethod
    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None: ...

    @abstractmethod
    def run_setup(self, run_dir: Path, n_train_examples: int) -> None: ...

    @abstractmethod
    def clip_grad_norm(self) -> None: ...

    # === VLA Training ===

    def run_vla_training(
        self,
        vla_dataset: IterableDataset,
        collator: PaddedCollatorForActionPrediction,
        action_tokenizer: ActionTokenizer,
        metrics: VLAMetrics,
        save_interval: int = 2500,
        save_full_model: bool = True,
    ) -> None:
        """Run the VLA training loop for the given `dataset` and `collator`; log losses, action metrics to `metrics`."""
        assert isinstance(vla_dataset, IterableDataset), "VLA training expects an IterableDataset!"
        # assert self.grad_accumulation_steps == 1, "VLA training does not support gradient accumulation!"

        # Create a DataLoader =>> Set `num_workers` to 0; RLDS loader handles parallelism!
        dataloader = DataLoader(
            vla_dataset,
            batch_size=self.per_device_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,
            worker_init_fn=self.worker_init_fn,
        )

        # === Train ===
        status = metrics.get_status()
        with tqdm(
            total=(self.epochs * len(dataloader)) if self.max_steps is None else self.max_steps,
            desc=status,
            leave=False,
            dynamic_ncols=True,
            disable=not overwatch.is_rank_zero(),
        ) as progress:
            self.vlm.train()

            # Zero Gradients (just in case)
            self.optimizer.zero_grad()

            # [Contract] DataLoader wraps RLDS Loader (`.as_numpy_iterator() =>> implicit `.repeat()`)
            #   => This means looping over the DataLoader is basically "infinite" (so no outer loop over epochs).
            #      Slightly breaks default PyTorch semantics, which is why we adaptively compute `epoch` below.
            for batch_idx, batch in enumerate(dataloader):
                # Note that we'll unpack batch (and let AMP/FSDP do its thing) in the VLM.forward() call
                #   => Basically, if we're using mixed precision (or not), autocast()/FSDP will move to device!
                
                
                gradient_step_idx = batch_idx // self.grad_accumulation_steps
                maybe_no_sync = nullcontext
                # Don't sync gradients until the final batch for FSDP.
                if isinstance(self.vlm, FSDP) and not ((batch_idx + 1) % self.grad_accumulation_steps == 0):
                    maybe_no_sync = self.vlm.no_sync
                
                with maybe_no_sync():
                    with torch.autocast(
                        "cuda", dtype=self.mixed_precision_dtype, enabled=self.enable_mixed_precision_training
                    ):
                        
                        
                        # [Contract] self.vlm.forward() must automatically compute `loss` and return!
                        output: CausalLMOutputWithPast = self.vlm(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            pixel_values=batch["pixel_values"],
                            labels=batch["labels"],
                            actions=batch["actions"],
                            proprio=batch["proprio"],
                            pred_action = gradient_step_idx % 1000 == 0
                        )
                        loss = output.loss

                    # Commit Loss =>> Backward!
                    metrics.commit(loss=loss)
                    loss.backward()
                # if True:
                #     # layer weight gradients
                #     # for name, param in self.vlm.named_parameters():
                #     self.vlm.llm_backbone.model.layers[-1].self_attn.k_proj.weight.requires_grad = True
                # import ipdb; ipdb.set_trace()
                # === Compute Action Token Accuracy & L1 Loss ===

                if gradient_step_idx % 1000 == 0:
                    # To compute action token accuracy, we need to identify the locations of the action tokens
                    # in both `output.logits` and `batch["labels"]`. We know that when "right" padding, we
                    # insert `self.vlm.vision_backbone.num_patches` at index 1.
                    #
                    # Computing `action_prediction_accuracy` is then pretty straightforward:
                    #   1) Extract "aligned" predictions & labels
                    #   2) Compute boolean "mask" where "labels > 2" (where 2 is ID for `EOS_TOKEN`)
                    #           => If masking out EOS, then it's just "labels != -100 (IGNORE_INDEX)
                    #   3) Compute masked accuracy as `(preds == logits) & mask` --> sum/divide by # unmasked!
                    
                    # Compute L1 Loss on Predicted (Continuous) Actions
                    action_preds = output.actions.detach().cpu()
                    action_gt = batch["actions"].cpu()
                    
                    correct_preds = torch.abs(action_preds - action_gt) < (1/256)
                    action_accuracy = correct_preds.sum().float() / torch.ones_like(action_gt).sum().float()

                    
                    action_l1_loss = torch.nn.functional.l1_loss(action_preds, action_gt)

                    # Commit Metrics
                    metrics.commit(action_accuracy=action_accuracy, l1_loss=action_l1_loss, update_step_time=True)

                    # Compute metrics per dataset --> only on rank_zero since we don't log them on other workers anyways
                    if overwatch.is_rank_zero():
                        datasets = set(batch["dataset_names"])
                        if len(datasets) > 1:
                            for ds in datasets:
                                ds_mask = torch.tensor([elem == ds for elem in batch["dataset_names"]])
                                
                                action_accuracy_ds = correct_preds[ds_mask].sum().float() / torch.ones_like(action_gt[ds_mask]).sum().float()
                                action_l1_loss_ds = torch.nn.functional.l1_loss(
                                    action_preds[ds_mask], action_gt[ds_mask]
                                )
                                metrics.commit_for_dataset(
                                    dataset_name=ds.decode(), l1_loss=action_l1_loss_ds, action_accuracy=action_accuracy_ds
                                )
                                
                    del output

                # === Gradient Step ===
                # Optimizer Step
                if (batch_idx + 1) % self.grad_accumulation_steps == 0:
                    # Clip Gradients --> this is custom, per-strategy because of DDP vs. FSDP locality assumptions
                    self.clip_grad_norm()
                    # Optimizer & LR Scheduler Step
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    # Update Progress Bar
                    progress.update()
                    progress.set_description(status)

                # Compute epoch value using number of completed gradient steps
                epoch = (gradient_step_idx + 1) // (len(vla_dataset) // self.global_batch_size)

                # Push Metrics
                metrics.commit(global_step=gradient_step_idx + 1, epoch=epoch, lr=self.lr_scheduler.get_last_lr()[0])
                status = metrics.push()

                # Check for Save Interval or Max Steps & Save Checkpoint
                if (terminate := (self.max_steps is not None and gradient_step_idx >= self.max_steps)) or (
                    (gradient_step_idx % save_interval) == 0
                ):
                    self.save_checkpoint(
                        metrics.run_dir, gradient_step_idx, epoch, loss.item(), only_trainable=not save_full_model
                    )
                    dist.barrier()

                    if terminate:
                        return

