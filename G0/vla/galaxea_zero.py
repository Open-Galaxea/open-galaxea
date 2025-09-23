from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, Union, Tuple, Any

import einops

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import AutoTokenizer


from paligemma.modeling_paligemma import PaliGemmaMultiModalProjector, PaliGemmaConfig

from transformers.models.gemma import GemmaTokenizer
from transformers.models.siglip import SiglipVisionModel, SiglipImageProcessor


from vla.helper import ImageProcessorToTransform
from vla.config.import_utils import get_obj_from_str
from omegaconf import DictConfig


from dataclasses import dataclass
from transformers.utils import ModelOutput

from vla.vla_base import VLABase
from .model.g0.galaxeazero import GalaxeaZero

import logging

logger = logging.getLogger(__name__)

@dataclass
class GalaxeaModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    actions: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    loss_dict: Optional[Dict[str, torch.FloatTensor]] = None

@dataclass
class FakeTensor():
    shape: Tuple[int] = None
    dtype: torch.dtype = None
    device: torch.device = None
    requires_grad: bool = None


class GalaxeaZeroWrapper(VLABase):
    def __init__(
        self,
        model_id: str,
        config: PaliGemmaConfig,
        enable_mixed_precision_training: bool = True,
        local_files_only: bool = True,
        action_expert_only: bool = False,
        model_cfg: DictConfig = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.config = config

        self.model_family, self.model_id = 'galaxea_zero', model_id
        self.norm_stats = {}

        self.model: GalaxeaZero = get_obj_from_str(model_cfg.get("model_name", "vla.model.g0.galaxeazero.GalaxeaZero"))(
            model_cfg
        )

        self.pad_token_id = 0
        # Instance Attributes for a generic VLM
        self.all_module_keys, self.trainable_module_keys = None, None
        # Set Weight Initialization Seed for Projector Consistency
        # torch.manual_seed(self.config.hidden_size)
        # Set Module Keys =>> used in Checkpoint Saving / Model Loading
        self.all_module_keys = ["vision_backbone", "llm_backbone", "projector", "action_expert"]
        self.trainable_module_keys = []

        # Action Expert
        self.flow_sampling = "beta"
        flow_alpha = 1.5
        flow_beta = 1
        self.flow_t_max = 1 - 0.001
        self.flow_beta_dist = torch.distributions.Beta(flow_alpha, flow_beta)

        self.action_expert_only = action_expert_only  
        self.model_config = model_cfg  
        self.enable_mixed_precision_training = enable_mixed_precision_training
        self.vision_backbone_identifier = config.vision_config.model_type
        self.llm_backbone_identifier = config.text_config.model_type
        self.cached_key_values = None
        # Trackers
        self.vision_backbone_requires_grad = False
        self.num_input_images = model_cfg.get("num_input_images", 1)

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        model_config: PaliGemmaConfig,
        tokenizer: GemmaTokenizer,
        image_processor: SiglipImageProcessor,
        enable_mixed_precision_training: bool = True,
        training: bool = True,
        **model_kwargs,
    ) -> GalaxeaZeroWrapper:
        """Initialize a VLM from a pretrained checkpoint."""
        vlm = cls(
            model_id,
            model_config,
            enable_mixed_precision_training=enable_mixed_precision_training,
            **model_kwargs,
        )

        # Load from Checkpoint (Custom --> should load both *projector* and *llm* weights)
        logger.info(f"Loading [bold blue] state dict[/] from Checkpoint")
        vlm.model.load_pretrained_weights()
        vlm.model.tie_action_proprio_weights()
        vlm.model.freeze_unused_weights()
        logger.info(f"Loading [bold blue]tokenizer and image processor[/] from Checkpoint")
        vlm.image_transform = ImageProcessorToTransform(image_processor)
        vlm.tokenizer = tokenizer

        # Freeze Weights
        if not training:
            vlm.requires_grad_(False)
            vlm.eval()

        return vlm

    @classmethod
    def from_checkpoint(
        cls,
        model_id: str,
        model_config: PaliGemmaConfig,
        state_dict: Dict[str, Any],
        norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]],
        tokenizer: GemmaTokenizer,
        image_processor: SiglipImageProcessor,
        enable_mixed_precision_training: bool = True,
        training: bool = True,
        strict: bool = True,
        **model_kwargs,
    ) -> GalaxeaZeroWrapper:
        """Initialize a VLA from a pretrained checkpoint."""
        vlm = cls(
            model_id,
            model_config,
            enable_mixed_precision_training=enable_mixed_precision_training,
            **model_kwargs,
        )

        new_state_dict = {}
        for k, v in state_dict.items():
            if "_orig_mod" in k:
                new_state_dict[k.replace("_orig_mod.", "")] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
        filtered = {}
        for k, v in state_dict.items():
            if k in vlm.state_dict():
                if v.shape != vlm.state_dict()[k].shape:
                    logger.warning(f"Shape Mismatch for {k}: {v.shape} vs {vlm.state_dict()[k].shape}"
                                      f"skipping")
                    continue
            filtered[k] = v

        vlm.norm_stats = norm_stats
        incompatable_keys = vlm.load_state_dict(filtered, strict=strict)
        if len(incompatable_keys.missing_keys) > 0:
            logger.warning(f"Missing Keys in checkpoint: {incompatable_keys.missing_keys}")
        if len(incompatable_keys.unexpected_keys) > 0:
            logger.warning(f"Unexpected Keys in checkpoint: {incompatable_keys.unexpected_keys}")

        vlm.model.tie_action_proprio_weights()
        logger.info(f"Loading [bold blue]tokenizer and image processor[/] from Checkpoint")
        vlm.image_transform = ImageProcessorToTransform(image_processor)
        vlm.tokenizer = tokenizer

        # Freeze Weights
        if training:
            vlm.model.freeze_unused_weights()
        else:
            vlm.requires_grad_(False)
            vlm.eval()

        return vlm

    def get_optim_param_groups(self, lr: float, backbone_lr_multiplier: float = 1.0):
        """
        This function returns a list of parameter groups for the optimizer
        """
        assert len(list(self.parameters())) == len(list(self.model.parameters()))
        action_expert_params_id = set(id(p) for p in self.model.action_expert_parameters)

        action_expert_params = [p for p in self.model.parameters() if id(p) in action_expert_params_id]
        backbone_params = [p for p in self.model.parameters() if id(p) not in action_expert_params_id]

        param_groups = [
            {"params": [p for p in backbone_params if p.requires_grad], "lr": lr * backbone_lr_multiplier},
            {"params": [p for p in action_expert_params if p.requires_grad], "lr": lr},
        ]

        all_requires_grad_params = [p for p in self.parameters() if p.requires_grad]
        assert len(all_requires_grad_params) == sum([len(g['params']) for g in param_groups])

        return param_groups

    @property
    def num_patches(self) -> int:
        return self.vision_backbone.vision_model.embeddings.num_patches

    def sample_fm_time(self, bsz: int) -> torch.FloatTensor:
        if self.flow_sampling == "uniform":  # uniform between 0 and 1
            """https://github.com/gle-bellier/flow-matching/blob/main/Flow_Matching.ipynb"""
            eps = 1e-5
            t = (torch.rand(1) + torch.arange(bsz) / bsz) % (1 - eps)
        elif self.flow_sampling == "beta":  # from pi0 paper
            z = self.flow_beta_dist.sample((bsz,))
            t = self.flow_t_max * (1 - z)  # flip and shift
        return t

    def forward(
        self,
        *args,
        inference_mode=False,
        **kwargs
    ):

        if inference_mode:
            was_training = self.training
            self.model.eval()
            out = self.forward_inference(*args, **kwargs)
            self.model.train(was_training)
            return out
        else:
            return self.forward_train(*args, **kwargs)

    def pre_process_inputs(self, input_ids, attention_mask):
        num_image_tokens = self.model.cfg.vision.num_image_tokens * self.num_input_images
        # pad input_ids to self.max_image_text_tokens
        max_text_tokens = self.model.cfg.max_image_text_tokens - num_image_tokens
        if input_ids.shape[1] < max_text_tokens:
            input_ids = nn.functional.pad(input_ids, 
                (0, max_text_tokens - input_ids.shape[1]),value=self.pad_token_id)
        elif input_ids.shape[1] > max_text_tokens:
            input_ids = input_ids[:, :max_text_tokens]
        # add image tokens
        input_ids = torch.cat([input_ids[:, :1], 
                            torch.full(
                                (input_ids.shape[0], num_image_tokens),
                                self.model.cfg.image_token_index).to(input_ids),
                            input_ids[:,1:]], dim=1)
        attention_mask = torch.cat([attention_mask[:, :1],
                                    torch.full(
                                        (attention_mask.shape[0], num_image_tokens),
                                        1).to(attention_mask),
                                    attention_mask[:, 1:]], dim=1)
        return input_ids, attention_mask

    def forward_train(self,
                    input_ids: Optional[torch.LongTensor] = None,
                    attention_mask: Optional[torch.Tensor] = None,
                    pixel_values: Optional[torch.FloatTensor] = None,
                    actions: Optional[torch.FloatTensor] = None,
                    action_pad_masks: Optional[torch.BoolTensor] = None,
                    proprio: Optional[torch.FloatTensor] = None,
                    **kwargs):

        input_ids_processed, attention_mask_processed = \
            self.pre_process_inputs(input_ids.clone(), attention_mask.clone())
        device = input_ids_processed.device
        dtype = pixel_values.dtype
        t = self.sample_fm_time(len(input_ids_processed)).to(dtype)
        if proprio.ndim == 2:
            proprio = proprio.unsqueeze(1)
        # proprio_processd = proprio[:, None] # add cond_steps dimension NOTE: added at batchtransform
        assert pixel_values.shape[1] == self.num_input_images, \
            f"pixel_values.shape[1] ({pixel_values.shape[1]}) should be equal to self.num_input_images ({self.num_input_images})"
        loss_dict = self.model(
            input_ids=input_ids_processed,
            attention_mask=attention_mask_processed,
            pixel_values=pixel_values,
            proprios=proprio,
            actions=actions,
            action_pad_masks=action_pad_masks,
            t=t.to(dtype).to(device)
        )
        loss = sum(loss_dict.values())

        outputs = GalaxeaModelOutput(loss=loss, loss_dict=loss_dict)        

        return outputs

    @torch.no_grad()
    def forward_inference(self,                    
                    input_ids: Optional[torch.LongTensor] = None,
                    attention_mask: Optional[torch.Tensor] = None,
                    pixel_values: Optional[torch.FloatTensor] = None,
                    proprio: Optional[torch.FloatTensor] = None,
                    **kwargs):

        input_ids, attention_mask = self.pre_process_inputs(input_ids, attention_mask)
        if proprio.ndim == 2:
            proprio = proprio.unsqueeze(1)

        assert pixel_values.shape[1] == self.num_input_images, \
            f"pixel_values.shape[1] ({pixel_values.shape[1]}) should be equal to self.num_input_images ({self.num_input_images})"

        sampled_actions = self.model.infer_action(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            proprios=proprio,
        )

        return sampled_actions
