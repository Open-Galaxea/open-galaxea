"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type, List, Sequence, Optional, Union
import copy

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase
from einops import rearrange

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tree_map
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType
from prismatic.vla.datasets.instructions import augment_instruction


# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


class InstructionOverLengthError(Exception):
    pass


@dataclass
class RLDSBatchTransform:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name, action = rlds_batch["dataset_name"], rlds_batch["action"][0]
        img = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        lang = rlds_batch["task"]["language_instruction"].decode().lower()

        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(img)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, 
                    input_ids=input_ids, 
                    labels=labels, 
                    dataset_name=dataset_name)

@dataclass
class RLDSBatchTransform4Galaxea:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    predict_stop_token: bool = True
    window_size: int = 1
    future_action_window_size: int = 7

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Converts a RLDS batch to the format expected by the OpenVLA collator/models."""
        dataset_name = rlds_batch["dataset_name"]
        imgs = [Image.fromarray(i) for i in rlds_batch["observation"]["image_primary"]]
        lang = rlds_batch["task"]["language_instruction"].decode().lower()
        proprio = rlds_batch["observation"]["proprio"][-1]
        
        action = rlds_batch["action"][self.window_size-1:] # only take the future and current action
        # Construct Chat-based Prompt =>> Input is default query + language instruction, output are the action tokens
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {lang}?"},
            {"from": "gpt", "value": 'Robot should take following actions: '},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = [self.image_transform(img) for img in imgs] # it will automatically genearte batch dimension
        pixel_values = torch.cat(pixel_values, dim=0) # (1, c, h, w)

        # to numpy        
        # [Diffusion Policy] Convert gripper action to [-1, 1] range for diffusion policy
        action = torch.from_numpy(action).to(pixel_values)
        action[..., -1] = action[..., -1] * 2 - 1
        
        # to numpy
        proprio = torch.from_numpy(proprio).to(pixel_values)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(10 + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, 
                    input_ids=input_ids, 
                    labels=labels, 
                    actions=action,
                    proprio=proprio,
                    instructions=lang,
                    dataset_name=dataset_name)

@dataclass
class RLDSBatchTransform4GalaxeaMultiObsMultiCams:
    action_tokenizer: ActionTokenizer
    base_tokenizer: PreTrainedTokenizerBase
    image_transform: ImageTransform
    prompt_builder_fn: Type[PromptBuilder]
    input_id_max_length: int = 512
    predict_stop_token: bool = True
    window_size: int = 1
    future_action_window_size: int = 7
    camera_views: Sequence[str] = ("primary", )
    aug_instruction: bool = False
    aug_instruction_kwargs: Optional[Dict[str, Any]] = None

    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Converts a RLDS sample (not batch!) to the format expected by the OpenVLA collator/models.
        """
        dataset_name = rlds_batch["dataset_name"]
        imgs = []
        intrinsics = []

        for cam in self.camera_views:
            single_views = [Image.fromarray(i) for i in rlds_batch["observation"][f"image_{cam}"]]
            assert len(single_views) == self.window_size, f"# Images should be equal to window_size for {cam}"
            imgs.append(single_views)

        
        # [[c0_t0, c0_t1, c0_t2], ..., [cn_t0, cn_t1, cn_t2], ...] -> [[c0_t0, c1_t0, ..., cn_t0], ..., [c0_to, ..., cn_to]]
        imgs = list(zip(*imgs))
        # [[c0_t0, c1_t0, ..., cn_t0], ..., [c0_to, ..., cn_to]] -> [c0_t0, c1_t0, ..., cn_t0, ..., c0_to, ..., cn_to]
        imgs = [i for sublist in imgs for i in sublist] # flatten

        # lang = rlds_batch["task"]["language_instruction"].decode().lower()
        lang = augment_instruction(rlds_batch, self.aug_instruction, **(self.aug_instruction_kwargs if self.aug_instruction_kwargs else {}))
        proprio = rlds_batch["observation"]["proprio"] # (t, c)
        assert len(proprio) == self.window_size, "Proprio length should be equal to window_size"
        
        action = rlds_batch["action"][self.window_size-1:] # only take the future and current action
        assert len(action) == self.future_action_window_size + 1, \
            f"Action length should be {self.future_action_window_size+1} but got {len(action)}"
        action_pad_mask = rlds_batch["action_pad_mask"][self.window_size-1:]
        assert len(action_pad_mask) == self.future_action_window_size + 1

        # match the prompt format of the openpi
        # Tokenize (w/ `base_tokenizer`)
        # we don't need to add special tokens here, since we will use the prompt as input_ids
        input_ids = self.base_tokenizer(
            f'{self.base_tokenizer.bos_token}Task: {lang}, ', add_special_tokens=False).input_ids
        input_ids = input_ids[:-1] # remove eos token
        
        labels = list(input_ids)
        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels = [IGNORE_INDEX for _ in labels]
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF LLM.forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = [self.image_transform(img) for img in imgs] # it will automatically genearte batch dimension
        pixel_values = torch.cat(pixel_values, dim=0) # (n, c, h, w), n = window_size * len(camera_views)

        if len(input_ids) > self.input_id_max_length:
            raise InstructionOverLengthError(f"Instruction: {lang} with Input ID length {len(input_ids)} exceeds max length {self.input_id_max_length}")
        
        action = torch.tensor(action).to(pixel_values) # use from_numpy() throws warnings
        proprio = torch.tensor(proprio).to(pixel_values)
        action_pad_mask = torch.tensor(action_pad_mask, dtype=torch.bool, device=pixel_values.device)

        
        sample = dict(pixel_values=pixel_values, 
                    input_ids=input_ids, 
                    labels=labels, 
                    actions=action,
                    action_pad_mask=action_pad_mask,
                    proprio=proprio,
                    instructions=lang,
                    dataset_name=dataset_name)
        if "is_first" in rlds_batch:
            sample["is_first"] = rlds_batch["is_first"]
        if "is_last" in rlds_batch:
            sample["is_last"] = rlds_batch["is_last"]
        
        return sample

class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        resize_resolution: Tuple[int, int],
        shuffle_buffer_size: int = 256_000,
        balance_weights=True,
        train: bool = True,
        image_aug: bool = False,
        image_augment_kwargs=None,
        window_size: int = 1,
        future_action_window_size: int = 7,
        goal_relabeling_strategy: str = "identity",
        repeat_last_timestep_ratio: float = 0.0,
        load_camera_views=("primary",),
        load_depth=False,
        use_last_action=False,
        last_action_drop_prob=0.0,
        last_action_dim: Optional[int] = None,
        proprio_noise_std: float = 0.0,
        action_proprio_normalization_type="normal",
        preset_datasets_statistics: bool = False,
        max_trajectories_per_dataset: Optional[Union[int, float]] = None,
        use_relative_joint_action: bool = False,
        **kwargs,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir, self.data_mix, self.batch_transform = data_root_dir, data_mix, batch_transform

        # Configure RLDS Dataset(s)
        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            # Assume that passed "mixture" name is actually a single dataset -- create single-dataset "mix"
            mixture_spec = [(self.data_mix, 1.0)]

        action_proprio_normalization_map = {
            "normal": NormalizationType.NORMAL,
            "q99": NormalizationType.BOUNDS_Q99,
            "max": NormalizationType.BOUNDS
        }
        if not use_last_action:
            assert last_action_drop_prob == 0.0 and last_action_dim is None, "Cannot set `last_action_drop_prob` or `last_action_dim` if `use_last_action` is False"

        # fmt: off
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=load_camera_views,
            load_depth=load_depth,
            use_last_action=use_last_action,
            load_proprio=True,
            load_language=True,
            action_proprio_normalization_type=action_proprio_normalization_map[
                action_proprio_normalization_type],
            max_trajectories=max_trajectories_per_dataset,
            use_relative_joint_action=use_relative_joint_action,
        )
        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=window_size,                                      # If we wanted to feed / predict more than one step
                future_action_window_size=future_action_window_size,          # For action chunking
                skip_unlabeled=True,                                # Skip trajectories without language labels
                goal_relabeling_strategy=goal_relabeling_strategy,                 # Use the whole trajectory
                repeat_last_timestep_ratio=repeat_last_timestep_ratio,  # Repeat last timesteps multiple times
            ),
            frame_transform_kwargs=dict(
                resize_size=resize_resolution,
                depth_resize_size=resize_resolution,
                last_action_drop_prob=last_action_drop_prob,
                last_action_dim=last_action_dim,
                proprio_noise_std=proprio_noise_std,
                num_parallel_calls=4,                          # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=balance_weights,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
            preset_datasets_statistics=preset_datasets_statistics,
        )

        # If applicable, enable image augmentations
        if image_aug:
            if image_augment_kwargs is None:
                image_augment_kwargs = dict(
                    random_resized_crop=dict(scale=[0.9, 0.9], ratio=[1.0, 1.0]),
                    random_brightness=[0.2],
                    random_contrast=[0.8, 1.2],
                    random_saturation=[0.8, 1.2],
                    random_hue=[0.05],
                    augment_order=[
                        "random_resized_crop",
                        "random_brightness",
                        "random_contrast",
                        "random_saturation",
                        "random_hue",
                    ],
                )

            rlds_config["frame_transform_kwargs"].update({"image_augment_kwargs": image_augment_kwargs}),
        # fmt: on

        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            try:
                sample = self.batch_transform(rlds_batch)
            except InstructionOverLengthError as e:
                print(f"Skipping sample due to over-length instruction: {e}")
                continue
            # sample = self.batch_transform(rlds_batch)
            yield sample

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")


class EpisodicRLDSDataset(RLDSDataset):
    """Returns full episodes as list of steps instead of individual transitions (useful for visualizations)."""

    def make_dataset(self, rlds_config):
        per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
        assert len(per_dataset_kwargs) == 1, "Only support single-dataset `mixes` for episodic datasets."

        return make_single_dataset(
            per_dataset_kwargs[0],
            train=rlds_config["train"],
            traj_transform_kwargs=rlds_config["traj_transform_kwargs"],
            frame_transform_kwargs=rlds_config["frame_transform_kwargs"],
        )

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            out = [
                self.batch_transform(tree_map(lambda x: x[i], rlds_batch))  # noqa: B023
                for i in range(rlds_batch["action"].shape[0])
            ]
            yield out


class DummyDataset(Dataset):
    def __init__(
        self,
        action_tokenizer: ActionTokenizer,
        base_tokenizer: PreTrainedTokenizerBase,
        image_transform: ImageTransform,
        prompt_builder_fn: Type[PromptBuilder],
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.
        self.dataset_statistics = {
            "dummy_dataset": {
                "action": {"q01": np.zeros((7,), dtype=np.float32), "q99": np.ones((7,), dtype=np.float32)}
            }
        }

    def __len__(self):
        # Replace with number of elements in your dataset!
        return 10000

    def __getitem__(self, idx):
        # Load image, action and instruction from disk -- we use dummy values
        image = Image.fromarray(np.asarray(np.random.rand(224, 224, 3) * 255.0, dtype=np.uint8))
        action = np.asarray(np.random.rand(7), dtype=np.float32)
        instruction = "do something spectacular"

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {instruction}?"},
            {"from": "gpt", "value": self.action_tokenizer(action)},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)
        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[: -(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
