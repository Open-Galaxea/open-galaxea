from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, Union, Tuple, Any, Literal

import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
import torch
from PIL import Image
from transformers import LlamaTokenizerFast

import logging

logger = logging.getLogger(__name__)

class VLABase(nn.Module):

    def build_inputs(
        self, 
        images: Dict[str, List[Image.Image]], 
        instruction: str, 
        proprio: np.ndarray, # (t, c)
        unnorm_key: Optional[str] = None, 
        unnorm_type: Optional[Literal["q99", "normal", "max"]] = "q99",
        select_keys: Optional[List[str]] = ["primary"],
    ) -> Tuple:
        image_transform, tokenizer = self.image_transform, self.tokenizer

        imgs = []

        for cam in select_keys:
            single_views = images[cam]
            assert len(single_views) == self.model.cond_steps, f"# Images should be equal to self.cond_steps for {cam}"
            imgs.append(single_views)

        # [[c0_t0, c0_t1, c0_t2], ..., [cn_t0, cn_t1, cn_t2], ...] -> [[c0_t0, c1_t0, ..., cn_t0], ..., [c0_to, ..., cn_to]]
        imgs = list(zip(*imgs))
        # [[c0_t0, c1_t0, ..., cn_t0], ..., [c0_to, ..., cn_to]] -> [c0_t0, c1_t0, ..., cn_t0, ..., c0_to, ..., cn_to]
        imgs = [i for sublist in imgs for i in sublist] # flatten


        # Build VLA Prompt
        prompt_text = instruction.lower().strip()
        # Prepare Inputs
        input_ids = tokenizer(
            f'{tokenizer.bos_token}Task: {prompt_text}, ', return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        input_ids = input_ids[:, :-1] # remove eos token
        
        # Preprocess Image
        pixel_values = [image_transform(img) for img in imgs] # it will automatically genearte batch dimension
        pixel_values = torch.cat(pixel_values, dim=0) # (n, c, h, w), n = window_size * len(camera_views)
        pixel_values = pixel_values.unsqueeze(0).to(self.device) # (1, n, c, h, w)
        
        # process proprio
        if not isinstance(proprio, torch.Tensor):
            proprio = torch.tensor(proprio, dtype=torch.float32)
        if unnorm_type == "q99":
            proprio_low = torch.tensor(self.norm_stats[unnorm_key]["proprio"]["q01"])
            proprio_high = torch.tensor(self.norm_stats[unnorm_key]["proprio"]["q99"])
            proprio = (proprio - proprio_low) / (proprio_high - proprio_low + 1e-8)
            proprio = proprio * 2 - 1 # normalize to [-1, +1]
        
        elif unnorm_type == "max":
            proprio_max = torch.tensor(self.norm_stats[unnorm_key]["proprio"]["max"])
            proprio_min = torch.tensor(self.norm_stats[unnorm_key]["proprio"]["min"])
            proprio = (proprio - proprio_min) / (proprio_max - proprio_min + 1e-8)
            proprio = proprio * 2 - 1 # normalize to [-1, +1]

        elif unnorm_type == "normal":
            mean = torch.tensor(self.norm_stats[unnorm_key]["proprio"]["mean"])
            std = torch.tensor(self.norm_stats[unnorm_key]["proprio"]["std"])
            proprio = (proprio - mean) / (std + 1e-8) # already in [-1, +1] (in term of scales)

        zeros_mask = torch.tensor(self.norm_stats[unnorm_key]["proprio"]["min"]) == \
            torch.tensor(self.norm_stats[unnorm_key]["proprio"]["max"])
        proprio = torch.where(zeros_mask, torch.zeros_like(proprio), proprio)
        proprio = proprio.to(self.device) 
        
        # return input_ids, pixel_values, proprio
        return {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "proprio": proprio,
        }

    def infer(self, obs: Dict) -> Dict:
        return self.predict_action(**obs)

    @torch.inference_mode()
    def predict_action(
        self, 
        images: Dict[str, List[np.ndarray]], 
        instruction: str, 
        proprio: Union[np.ndarray, torch.Tensor], 
        unnorm_key: Optional[str] = None,
        unnorm_type: Optional[Literal["q99", "normal", "max"]] = "q99",
        select_keys: Optional[List[str]] = ["primary"],
        center_crop: Optional[bool] = True,
        autocast_dtype: Optional[Union[str, torch.dtype]] = torch.float32,
    ) -> np.ndarray:
        """
        Core function for VLA inference; maps input image and task instruction to continuous action.

        @param images: dict of list of np.ndarray Images (h, w, c), [..., t_-1, t_0]
        @param instruction: Task instruction string
        @param proprio: Proprioceptive state, (t, c)
        @param unnorm_key: Optional dataset name for retrieving un-normalizing statistics; if None, checks that model
                           was trained only on a single dataset, and retrieves those statistics.
        @param unnorm_type: Optional type of un-normalization to apply to actions.
        @param select_keys: Optional list of image keys
        @param center_crop: Whether to center-crop the input images before passing them to the model.

        """
        images = {k: [process_single_image(i, center_crop=center_crop) for i in imgs] for k, imgs in images.items()}

        batch = self.build_inputs(
            images, instruction, proprio, unnorm_key, unnorm_type, select_keys)
        
        if isinstance(autocast_dtype, str):
            dtype = getattr(torch, autocast_dtype)
            assert isinstance(dtype, torch.dtype)
            autocast_dtype = dtype

        # check all finite
        if not torch.isfinite(batch["pixel_values"]).all():
            logger.warning("Pixel values contain NaN or Inf")
        if not torch.isfinite(batch["proprio"]).all():
            logger.warning("Proprio state contains NaN or Inf")
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=False):
            # fmt: off
            actions = self.forward(
                input_ids=batch["input_ids"],
                pixel_values=batch["pixel_values"].to(dtype=autocast_dtype),
                attention_mask=torch.ones_like(batch["input_ids"]).bool(),
                proprio=batch["proprio"].to(dtype=autocast_dtype),
                inference_mode=True,
                verbose=False,
            )
            # fmt: ond c x
        if isinstance(actions, dict):
            actions = actions["actions"]

        # following code only works for float32.
        actions = actions.to(torch.float32)
        actions = actions.squeeze(0)[..., -self.get_action_dim(unnorm_key):].cpu().numpy()

        # Un-normalize Actions
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        if unnorm_type == "q99":
            action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
            actions = 0.5 * (actions + 1) # from [-1, +1] to [0, 1], including gripper
            actions = np.where(
                mask,
                actions * (action_high[np.newaxis,:] - action_low[np.newaxis,:]) + action_low[np.newaxis,:],
                actions,
            )
        elif unnorm_type == "max":
            action_max, action_min = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
            actions = 0.5 * (actions + 1) # from [-1, +1] to [0, 1], including gripper
            actions = np.where(
                mask,
                actions * (action_max[np.newaxis,:] - action_min[np.newaxis,:]) + action_min[np.newaxis,:],
                actions,
            )
        elif unnorm_type == "normal":
            mean = np.array(action_norm_stats["mean"])
            std = np.array(action_norm_stats["std"])
            actions = np.where(
                mask,
                (actions * (std[np.newaxis,:] + 1e-8) + mean[np.newaxis,:]),
                0.5 * (actions + 1), # specialize for gripper
            )

        else:
            raise ValueError(f"Invalid unnorm_type: {unnorm_type}")

        return actions

    @staticmethod
    def _check_unnorm_key(norm_stats: Dict, unnorm_key: str) -> str:
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, please pass a `unnorm_key` from the following "
                f"options to choose the statistics used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        # Error Handling
        assert (
            unnorm_key in norm_stats
        ), f"The `unnorm_key` you chose is not in the set of available statistics; choose from: {norm_stats.keys()}"

        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)

        return len(self.norm_stats[unnorm_key]["action"]["mean"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict:
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)

        return self.norm_stats[unnorm_key]["action"]

    @property
    def device(self) -> torch.device:
        """Borrowed from `transformers.modeling_utils.py` -- checks parameter device; assumes model on *ONE* device!"""
        return next(self.parameters()).device

def process_single_image(image, center_crop=False):
    image = Image.fromarray(image)
    image = image.convert("RGB")
    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
    #            the original height and width by sqrt(0.9) -- not 0.9!
    if center_crop:
        batch_size = 1
        crop_scale = 0.9

        # Convert to TF Tensor and record original data type (should be tf.uint8)
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype

        # Convert to data type tf.float32 and values between [0,1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Crop and then resize back to original size
        image = crop_and_resize(image, crop_scale, batch_size)

        # Convert back to original data type
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

        # Convert back to PIL Image
        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")
    return image

def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image

            
if __name__ == '__main__':

    pass