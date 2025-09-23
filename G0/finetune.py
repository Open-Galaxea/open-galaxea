import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import sys
import signal

import draccus
import torch
import torch.distributed as dist
import tqdm
from accelerate import PartialState
import bitsandbytes as bnb
from time import time
from omegaconf import OmegaConf
from datetime import datetime

from ema_pytorch import EMA

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader

import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction4Galaxea
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.datasets import (RLDSBatchTransform4GalaxeaMultiObsMultiCams,
                                    RLDSDataset)
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics
from prismatic.util import set_global_seed

from prismatic.overwatch import initialize_overwatch

from vla.helper import get_scheduler
from vla.galaxea_zero import GalaxeaModelOutput
from vla.load import load_from_checkpoint, load
from vla.config.config import get_cfg
from vla.config.import_utils import get_obj_from_str

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def log_allocated_gpu_memory(log=None, stage="loading model", device=0):
    if torch.cuda.is_available():
        allocated_memory = torch.cuda.memory_allocated(device)
        msg = f"Allocated GPU memory after {stage}: {allocated_memory/1024/1024/1024:.2f} GB"
        print(msg) if log is None else log.info(msg)

def handle_resize(signum, frame):
    tqdm.tqdm._instances.clear()
    for instance in tqdm.tqdm._instances:
        instance.refresh()

signal.signal(signal.SIGWINCH, handle_resize)

def finetune():
    cfg = get_cfg()

    name_tag = str(Path(cfg.config.split('/')[-1]).stem)

    print(f"Fine-tuning GalaxeaZero Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    if distributed_state.is_main_process:
        exp_id = name_tag
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        exp_name = cfg.get("exp_name", "")
        if not exp_name:
            exp_id += f"--{timestamp}"
        else:
            exp_id += f"--{exp_name}"

        run_dir, adapter_dir = Path(cfg.run_root_dir) / exp_id, Path(cfg.adapter_tmp_dir) / exp_id
        os.makedirs(run_dir, exist_ok=True)
        with open(run_dir / "config.yaml", "w") as f:
            OmegaConf.save(config=cfg, f=f)

    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)

    hf_token = Path(cfg.hf_token).read_text().strip()
    if cfg.get("resume", None):
        vla = load_from_checkpoint(cfg.resume, hf_token=hf_token, load_for_training=True,
                                   load_pretrained_data_stats=cfg.DATASET.get("use_pretrained_data_stats", False),
                                   model_cfg=cfg.MODEL, strict=False)
    elif cfg.ckpt:
        vla = load_from_checkpoint(cfg.ckpt, hf_token=hf_token, load_for_training=True,
                                   load_pretrained_data_stats=cfg.DATASET.get("use_pretrained_data_stats", False),
                                   model_cfg=cfg.MODEL, strict=False)
    else:
        vla_cls = get_obj_from_str(cfg.MODEL.get("name", "vla.galaxea_zero.GalaxeaZeroWrapper"))
        vla = load(vla_cls, cfg.vla_path, hf_token=hf_token, load_for_training=True, 
                action_expert_only=cfg.MODEL.action_expert_only, model_cfg=cfg.MODEL)

    vla.model.freeze_by_stage(stage=cfg.get("vla_training_strategy", "vla-full-train"))

    if cfg.get("model_param_to_bf16", False):
        vla = vla.to(torch.bfloat16)

    use_ema = cfg.get("use_ema", False)
    if use_ema:
        ema_model = EMA(vla, 
                        update_after_step=cfg.ema.update_after_step, 
                        beta=cfg.ema.power).to(device_id) 

    if cfg.get("use_torch_compile", False):  # model being compiled in the first batch which takes some time
        # torch._dynamo.config.suppress_errors = True
        vla.model = torch.compile(
            vla.model,
            mode="default",
        )
    vla = vla.to(device_id)

    if distributed_state.is_main_process:
        log_allocated_gpu_memory(stage="loading model", device=0)# 22.8G for model_bf=False, 

    # Create Action Tokenizer
    action_tokenizer = ActionTokenizer(vla.tokenizer)

    # Load Fine-tuning Dataset =>> note that we use an RLDS-formatted dataset following Open X-Embodiment by default.
    #   =>> If you want to use a non-RLDS dataset (e.g., a standard PyTorch Dataset) see the following commented block.
    #   =>> Note that our training code does not loop over epochs because the RLDS loader does this implicitly; if using
    #       your own Dataset, make sure to add the appropriate logic to the training loop!
    #
    # ---
    # from prismatic.vla.datasets import DummyDataset
    #
    # vla_dataset = DummyDataset(
    #     action_tokenizer,
    #     processor.tokenizer,
    #     image_transform=processor.image_processor.apply_transform,
    #     prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
    # )
    # ---
    assert cfg.DATASET.get("short_prompt", True) == True, "short_prompt=false is deprecated!"
    batch_transform = RLDSBatchTransform4GalaxeaMultiObsMultiCams(
        action_tokenizer,
        vla.action_expert.language_encoder.tokenizer if cfg.MODEL.action_expert_only else vla.tokenizer,
        image_transform=vla.image_transform,
        prompt_builder_fn=PurePromptBuilder,
        input_id_max_length=cfg.MODEL.max_text_tokens,
        window_size=cfg.DATASET.get("window_size", 1),
        future_action_window_size=cfg.DATASET.get("future_action_window_size", 7),
        camera_views=cfg.DATASET.get("camera_views", ["primary"]),
        aug_instruction=cfg.DATASET.get("aug_instruction", False),
        aug_instruction_kwargs=cfg.DATASET.get("aug_instruction_kwargs", None),
    )

    if cfg.DATASET.get("use_pretrained_data_stats", False):
        assert hasattr(vla, "norm_stats"), "Model does not have `norm_stats` attribute!"
    vla_dataset = RLDSDataset(
        Path(cfg.data_root_dir),
        cfg.dataset_name,
        batch_transform,
        resize_resolution=(cfg.MODEL.vision.image_size, cfg.MODEL.vision.image_size),
        shuffle_buffer_size=cfg.DATASET.get("shuffle_buffer_size", 100000),
        balance_weights=cfg.DATASET.get("balance_weights", True),
        image_aug=cfg.image_aug,
        window_size=cfg.DATASET.get("window_size", 1),
        future_action_window_size=cfg.DATASET.get("future_action_window_size", 7),
        goal_relabeling_strategy=cfg.DATASET.get("goal_relabeling_strategy", "uniform"),
        repeat_last_timestep_ratio=cfg.DATASET.get("repeat_last_timestep_ratio", 0),
        load_camera_views=cfg.DATASET.get("camera_views", ["primary"]),
        image_augment_kwargs=OmegaConf.to_container(cfg.DATASET).get("image_augment_kwargs", None),
        use_last_action=cfg.DATASET.get("use_last_action", False),
        last_action_drop_prob=cfg.DATASET.get("last_action_drop_prob", 0.0),
        last_action_dim=cfg.DATASET.get("last_action_dim", None),
        proprio_noise_std=cfg.DATASET.get("proprio_noise_std", 0.0),
        action_proprio_normalization_type=cfg.DATASET.get("action_proprio_normalization_type", "q99"),
        preset_datasets_statistics=vla.norm_stats if cfg.DATASET.get("use_pretrained_data_stats", False) else None,
        max_trajectories_per_dataset=cfg.DATASET.get("max_trajectories_per_dataset", None),
        use_relative_joint_action=cfg.DATASET.get("use_relative_joint_action", False),
    )

    vla.norm_stats = vla_dataset.dataset_statistics  # Update norm_stats
    # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
    if distributed_state.is_main_process:
        dataset_statistics = save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

    if cfg.get("max_epochs", False):
        assert not cfg.get("max_steps", False), "Cannot set both `max_epochs` and `max_steps`!"
        total_transitions = int(vla_dataset.dataset_statistics["__total__"]["num_transitions"])
        effective_batch_size = cfg.batch_size * cfg.grad_accumulation_steps * torch.distributed.get_world_size()
        cfg.max_steps = int(total_transitions * cfg.max_epochs / effective_batch_size)
    
    # Create Collator and DataLoader
    collator = PaddedCollatorForActionPrediction4Galaxea(
        cfg.MODEL.max_text_tokens, vla.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
        worker_init_fn=worker_init_fn
    )

    # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
    vla = DDP(vla, device_ids=[device_id], find_unused_parameters=cfg.get("find_unused_parameters", False), gradient_as_bucket_view=True)

    # Create Optimizer
    param_groups = vla.module.get_optim_param_groups(
        lr=cfg.learning_rate,
        backbone_lr_multiplier=cfg.get("backbone_lr_multiplier", 1.0),
    )
    # trainable_params = [param for param in vla.parameters() if param.requires_grad]
    if cfg.get("use_8bit_optimizer", False):
        optimizer = bnb.optim.AdamW8bit(param_groups, weight_decay=cfg.weight_decay)
    else:
        optimizer = AdamW(param_groups, weight_decay=cfg.weight_decay)
    scheduler = get_scheduler(
        name=cfg.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=cfg.max_steps,
    )

    gradient_step_idx = 0
    # Resume Training
    # we only resume the optimizer, scheduler, ema_model, and gradient_step_idx
    # resume the dataloader state
    if cfg.get("resume", None):
        checkpoint = torch.load(cfg.resume, map_location=lambda storage, loc: storage.cuda(device_id))
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        if use_ema:
            try:
                ema_model.ema_model.load_state_dict(checkpoint['ema_model'])
            except KeyError:
                overwatch.warning("EMA model not found in checkpoint, skipping EMA update")
        gradient_step_idx = checkpoint['gradient_step_idx']
        del checkpoint  # Clean up checkpoint to avoid OOM
        torch.cuda.empty_cache()
        overwatch.info(f"Resuming training from step {gradient_step_idx}")

    # Initialize Logging =>> W&B
    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"{exp_id}",
                   config=OmegaConf.to_container(cfg))

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_loss_dict = deque(maxlen=cfg.grad_accumulation_steps)
    recent_metrics_log = {}
    for k in ["original", "ema"]:
        recent_metrics_log[k] = {
            # "action_exec_acc": deque(maxlen=cfg.grad_accumulation_steps),
            "action_all_acc": deque(maxlen=cfg.grad_accumulation_steps),
            # "action_exec_l1": deque(maxlen=cfg.grad_accumulation_steps),
            "action_all_l1": deque(maxlen=cfg.grad_accumulation_steps),
        }
    # gradient_step_idx = 0
    # Train!
    with tqdm.tqdm(initial=gradient_step_idx, total=cfg.max_steps, leave=False, dynamic_ncols=True) as progress:
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            # Compute gradient step index
            pred_action = (batch_idx + 1) % cfg.grad_accumulation_steps == 0 and \
                gradient_step_idx % cfg.get('log_steps', 100) == 0

            dtype = torch.bfloat16 if cfg.get("enable_bf16", True) else torch.float32
            # add no_sync to speed up
            if (batch_idx + 1) % cfg.grad_accumulation_steps != 0:
                with vla.no_sync():
                    with torch.autocast("cuda", dtype=dtype, enabled=cfg.get("enable_bf16", True)):
                        output: GalaxeaModelOutput = vla(
                            input_ids=batch["input_ids"].to(device_id),
                            attention_mask=batch["attention_mask"].to(device_id),
                            pixel_values=batch["pixel_values"].to(dtype).to(device_id),
                            labels=batch["labels"].to(device_id),
                            actions=batch["actions"].to(dtype).to(device_id),
                            action_pad_masks=batch["action_pad_masks"].to(device_id),
                            proprio=batch["proprio"].to(dtype).to(device_id),
                            instructions=batch["instructions"],
                            pred_action=False
                        )
                        loss = output.loss  
                    # Normalize loss to account for gradient accumulation
                    normalized_loss = loss / cfg.grad_accumulation_steps
                    # Backward pass
                    normalized_loss.backward()
            else:
                with torch.autocast("cuda", dtype=dtype, enabled=cfg.get("enable_bf16", True)):
                    output: GalaxeaModelOutput = vla(
                        input_ids=batch["input_ids"].to(device_id),
                        attention_mask=batch["attention_mask"].to(device_id),
                        pixel_values=batch["pixel_values"].to(dtype).to(device_id),
                        labels=batch["labels"].to(device_id),
                        actions=batch["actions"].to(dtype).to(device_id),
                        action_pad_masks=batch["action_pad_masks"].to(device_id),
                        proprio=batch["proprio"].to(dtype).to(device_id),
                        instructions=batch["instructions"],
                        pred_action=False
                    )
                    loss = output.loss  
                # Normalize loss to account for gradient accumulation
                normalized_loss = loss / cfg.grad_accumulation_steps
                # Backward pass
                normalized_loss.backward()

            recent_losses.append(loss.item())
            recent_loss_dict.append(output.loss_dict)
            smoothened_loss = sum(recent_losses) / len(recent_losses)
            smoothened_loss_dict = {k: sum([loss_dict[k] for loss_dict in recent_loss_dict]) / len(recent_loss_dict) for k in recent_loss_dict[0]}

            log_dict = {"loss": smoothened_loss, "lr":optimizer.param_groups[0]["lr"]}
            for k in smoothened_loss_dict.keys():
                log_dict[k] = smoothened_loss_dict[k]

            if pred_action:
                start = time()
                for policy_to_eval in ["original", "ema"]:
                    if policy_to_eval == "ema" and not use_ema:
                        continue
                    policy_net = ema_model if policy_to_eval == "ema" else vla.module
                    recent_metrics = recent_metrics_log[policy_to_eval]

                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.get("enable_bf16", True)):
                        action_preds = policy_net(
                            input_ids=batch["input_ids"].to(device_id),
                            attention_mask=batch["attention_mask"].to(device_id),
                            pixel_values=batch["pixel_values"].to(dtype).to(device_id),
                            labels=batch["labels"].to(device_id),
                            actions=batch["actions"].to(dtype).to(device_id),
                            action_pad_masks=batch["action_pad_masks"].to(device_id),
                            proprio=batch["proprio"].to(dtype).to(device_id),
                            instructions=batch["instructions"],
                            inference_mode=True,
                        )

                    action_preds = action_preds.cpu()
                    action_gt = batch["actions"].cpu()

                    correct_preds = torch.abs(action_preds - action_gt) < (1/256)
                    action_accuracy = correct_preds.sum().float() / torch.ones_like(action_gt).sum().float()

                    action_l1_loss = torch.nn.functional.l1_loss(action_preds, action_gt)
                    # Store recent train metrics
                    recent_metrics["action_all_acc"].append(action_accuracy.item())
                    recent_metrics["action_all_l1"].append(action_l1_loss.item())

                    # Compute smoothened train metrics
                    #   =>> Equal to current step metrics when not using gradient accumulation
                    #   =>> Otherwise, equal to the average of metrics observed over micro-batches used for gradient accumulation
                    suffix = "_ema" if policy_to_eval == "ema" else ""
                    log_dict.update(
                        {
                            # "action_exec_accuracy" + suffix: sum(recent_metrics['action_exec_acc']) / len(recent_metrics['action_exec_acc']),
                            "action_accuracy" + suffix: sum(recent_metrics['action_all_acc']) / len(recent_metrics['action_all_acc']),
                            # "action_exec_l1" + suffix: sum(recent_metrics['action_exec_l1']) / len(recent_metrics['action_exec_l1']),
                            "action_l1" + suffix: sum(recent_metrics['action_all_l1']) / len(recent_metrics['action_all_l1']),
                        }
                    )

                    torch.cuda.empty_cache()
                end = time()
                # if distributed_state.is_main_process:
                #     print(f"Pred action time at step {gradient_step_idx}: {end - start:.2f}s")

            # Push Metrics to W&B (every 10 gradient steps)
            if distributed_state.is_main_process and gradient_step_idx % 10 == 0:
                wandb.log(
                    log_dict,
                    step=gradient_step_idx,
                    
                )

            # Optimizer Step
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                # vla.clip_grad_norm_(1.0)
                torch.nn.utils.clip_grad_norm_(vla.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress.update()
                progress.refresh()  
                if use_ema:
                    ema_model.update()
                gradient_step_idx += 1

            # Save Model Checkpoint =>> by default, only keeps the latest checkpoint, continually overwriting it!
            if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0 and (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                if distributed_state.is_main_process:
                    print(f"Saving Model Checkpoint for Step {gradient_step_idx}")
                    save_dir = run_dir
                    to_save = vla.module.state_dict()
                    dataset_statistics.update(vla.module.norm_stats)
                    to_save.update({"dataset_statistics": dataset_statistics})
                    # save optimizer and scheduler
                    to_save["optimizer"] = optimizer.state_dict()
                    to_save["scheduler"] = scheduler.state_dict()
                    # save gradient_step_idx
                    to_save["gradient_step_idx"] = gradient_step_idx
                    torch.save(to_save, save_dir / f"model_{gradient_step_idx}.pt")
                    if use_ema:
                        to_save = ema_model.ema_model.state_dict()
                        to_save.update({"dataset_statistics": dataset_statistics})
                        torch.save(to_save, save_dir / f"model_ema_{gradient_step_idx}.pt")

                # Block on Main Process Checkpointing
                dist.barrier()

            # Stop training when max_steps is reached
            if gradient_step_idx == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                if distributed_state.is_main_process:
                    print(f"Saving Model Checkpoint for Step {gradient_step_idx}")
                    save_dir = run_dir
                    to_save = vla.module.state_dict()
                    to_save.update({"dataset_statistics": dataset_statistics})
                    torch.save(to_save, save_dir / f"model_{gradient_step_idx}.pt")
                    if use_ema:
                        to_save = ema_model.ema_model.state_dict()
                        to_save.update({"dataset_statistics": dataset_statistics})
                        torch.save(to_save, save_dir / f"model_ema_{gradient_step_idx}.pt")
                break

    distributed_state.destroy_process_group()


if __name__ == "__main__":
    finetune()
