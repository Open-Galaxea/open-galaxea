"""
dataset.py

Core interface script for configuring and initializing RLDS datasets.
"""

import copy
import inspect
import json
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Union, Sequence
import torch

import dlimp as dl
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from prismatic.overwatch import initialize_overwatch
from prismatic.vla.datasets.rlds import obs_transforms, traj_transforms
from prismatic.vla.datasets.rlds.utils import goal_relabeling, task_augmentation
from prismatic.vla.datasets.rlds.utils.data_utils import (
    NormalizationType,
    allocate_threads,
    get_dataset_statistics,
    normalize_action_and_proprio,
    pprint_data_mixture,
    tree_map,
)
from prismatic.vla.datasets.rlds.utils.encoding import ActionEncoding
from accelerate import PartialState

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# Configure Tensorflow with *no GPU devices* (to prevent clobber with PyTorch)
tf.config.set_visible_devices([], "GPU")

DEFAULT_TRAIN_RATIO = 99

# ruff: noqa: B006
def make_dataset_from_rlds(
    name: str,
    data_dir: str,
    *,
    train: bool,
    standardize_fn: Optional[Callable[[dict], dict]] = None,
    shuffle: bool = True,
    image_obs_keys: Dict[str, Optional[str]] = {},
    depth_obs_keys: Dict[str, Optional[str]] = {},
    state_obs_keys: List[Optional[str]] = (),
    language_key: Optional[str] = None,
    action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL,
    dataset_statistics: Optional[Union[dict, str]] = None,
    absolute_action_mask: Optional[List[bool]] = None,
    action_normalization_mask: Optional[List[bool]] = None,
    num_parallel_reads: int = tf.data.AUTOTUNE,
    num_parallel_calls: int = tf.data.AUTOTUNE,
    state_encoding: Optional[Sequence] = None,
    action_encoding: Optional[Sequence] = None,
    max_trajectories: Optional[Union[int, float]] = None,
) -> Tuple[dl.DLataset, dict]:
    """
    This function is responsible for loading a specific RLDS dataset from storage and getting it into a standardized
    format. Yields a dataset of trajectories. Does not include CPU-intensive operations.

    If `standardize_fn` is provided, it will be applied to each trajectory. This function should get the trajectory
    into a standard format, which includes the keys "observation" and "action". Entry "observation" should be a
    dictionary containing some number of additional keys, which will be extracted into an even more standardized format
    according to the "*_obs_keys" arguments.

    The `image_obs_keys` and `depth_obs_keys` arguments are mappings from new names to old names, or None in place of an
    old name to insert padding. For example, if after `standardize_fn`, your "observation" dict has RGB images called
    "workspace" and "wrist", and `image_obs_keys={"primary": "workspace", "secondary": None, "wrist": "wrist"}`, then
    the resulting dataset will have an "observation" dict containing the keys "image_primary", "image_secondary", and
    "image_wrist", where "image_primary" corresponds to "workspace", "image_secondary" is a padding image, and
    "image_wrist" corresponds to "wrist".

    Entry `state_obs_keys` is a list of 1-dimensional proprioceptive keys to concatenate into a single array, which will
    be placed in the "proprio" key of the "observation" dict. A single padding element (zero) will be inserted for each
    None entry.

    The dataset will also include a "task" dict. If `language_key` is provided, then the "task" dict will contain the
    key "language_instruction", extracted from `traj[language_key]`.

    Args:
        name (str): The name of the RLDS dataset (usually "name" or "name:version").
        data_dir (str): The path to the data directory.
        train (bool): Whether to use the training or validation split.
        shuffle (bool, optional): Whether to shuffle the file read order (does NOT fully shuffle the dataset, since one
            file usually contains many trajectories)!
        standardize_fn (Callable[[dict], dict], optional): A function that, if provided, will be the first
            thing applied to each trajectory.
        image_obs_keys (Mapping[str, str|None]): Mapping from {new: old} indicating which RGB images to extract from the
            "observation" dict. `new_obs = {f"image_{new}": old_obs[old] for new, old in image_obs_keys.items()}`.
            If a value of `old` is None, inserts a padding image instead (empty string).
        depth_obs_keys (Mapping[str, str|None]): Same as `image_obs_keys`, but for depth images. Keys will be
            prefixed with "depth_" instead of "image_".
        state_obs_keys (Sequence[str|None]): List of 1-dimensional proprioception keys to be extracted from the
            "observation" dict, concatenated, and mapped to "proprio". Inserts 1 element of padding for each None entry.
        language_key (str, optional): If provided, the "task" dict will contain the key "language_instruction",
            extracted from `traj[language_key]`.
        action_proprio_normalization_type (str, optional): The type of normalization to perform on the action,
            proprio, or both. Can be "normal" (mean 0, std 1) or "bounds" (normalized to [-1, 1]).
        dataset_statistics: (dict|str, optional): dict (or path to JSON file) that contains dataset statistics
            for normalization. If `action_proprio_normalization_type` is "normal", this should contain "mean" and
            "std" keys. If `action_proprio_normalization_type` is "bounds", this should contain "min" and "max"
            keys. May also provide "num_transitions" and "num_trajectories" keys for downstream usage (e.g., for
            `make_interleaved_dataset`). If not provided, the statistics will be computed on the fly.
        absolute_action_mask (Sequence[bool], optional): By default, all action dimensions are assumed to be
            relative. This is important for when `future_action_window_size > 0`: actions that are taken
            from beyond the end of the trajectory (or beyond the goal timestep when goal relabeling is used)
            need to be made "neutral" to indicate that the task has been completed. For relative actions,
            "neutral" means zero, but for absolute actions, "neutral" means repeating the last valid action.
            This mask, if provided, indicates which action dimensions are absolute.
        action_normalization_mask (Sequence[bool], optional): If provided, indicates which action dimensions
            should be normalized. For example, you might not want to normalize the gripper action dimension if
            it's always exactly 0 or 1. By default, all action dimensions are normalized.
        num_parallel_reads (int): number of parallel read workers. Default to AUTOTUNE.
        num_parallel_calls (int): number of parallel calls for traj_map operations. Default to AUTOTUNE.
        max_trajectories: (int|float, optional): int for maximum number of trajectories, or 
            float for percentage of trajectories to keep
        
    Returns:
        Dataset of trajectories where each step has the following fields:
        - observation:
            - image_{name1, name2, ...} # RGB image observations
            - depth_{name1, name2, ...} # depth image observations
            - proprio                   # 1-dimensional array of proprioceptive observations
            - timestep                  # timestep of each frame
        - task:
            - language_instruction      # language instruction, present if `language_key` is provided
        - action                        # action vector
        - dataset_name                  # name of the dataset
    """
    REQUIRED_KEYS = {"observation", "action", "is_first", "is_last"}
    if language_key is not None:
        REQUIRED_KEYS.add(language_key)
    assert action_encoding is not None, "Action encoding must be provided."

    def restructure(traj):
        # apply a standardization function, if provided
        if standardize_fn is not None:
            traj = standardize_fn(traj)

        if not all(k in traj for k in REQUIRED_KEYS):
            raise ValueError(
                f"Trajectory is missing keys: {REQUIRED_KEYS - set(traj.keys())}. " "Did you write a `standardize_fn`?"
            )

        # extracts images, depth images and proprio from the "observation" dict
        traj_len = tf.shape(traj["action"])[0]
        old_obs = traj["observation"]
        new_obs = {}
        for new, old in image_obs_keys.items():
            if old is None:
                new_obs[f"image_{new}"] = tf.repeat("", traj_len)  # padding
            else:
                new_obs[f"image_{new}"] = old_obs[old]

        for new, old in depth_obs_keys.items():
            if old is None:
                new_obs[f"depth_{new}"] = tf.repeat("", traj_len)  # padding
            else:
                new_obs[f"depth_{new}"] = old_obs[old]
                if old.replace("depth", "intrinsics") in old_obs:
                    new_obs[f"intrinsics_{new}"] = old_obs[old.replace("depth", "intrinsics")]
            
        if state_obs_keys:
            new_obs["proprio"] = tf.concat(
                [
                    (
                        tf.zeros((traj_len, 1), dtype=tf.float32)  # padding
                        if key is None
                        else tf.cast(old_obs[key], tf.float32)
                    )
                    for key in state_obs_keys
                ],
                axis=1,
            )

        # add timestep info
        new_obs["timestep"] = tf.range(traj_len)

        # extracts `language_key` into the "task" dict
        task = {}
        if language_key is not None:
            if traj[language_key].dtype != tf.string:
                raise ValueError(
                    f"Language key {language_key} has dtype {traj[language_key].dtype}, " "but it must be tf.string."
                )
            task["language_instruction"] = traj.pop(language_key)
        
        if "variant_idx" not in traj:
            if name == 'galaxea_desktop_organizing_v0':
                variant_idx = traj["bottle_order_idx"] * 2 + traj["bowl_order_idx"]
            else:
                variant_idx = tf.repeat(0, traj_len)
        else:
            variant_idx = traj["variant_idx"]
                
        # stage_idx to tf.int32
        variant_idx = tf.cast(variant_idx, tf.int32)
        
        if "segment_idx" not in traj:
            segment_idx = tf.repeat(0, traj_len)
        else:
            segment_idx = traj["segment_idx"]
            
        # segment_idx to tf.int32
        segment_idx = tf.cast(segment_idx, tf.int32)

        if action_encoding == ActionEncoding.R1_LITE_RELJOINT:
            print("testing relative action encoding")
            # convert joint actions to relative
            action_abs = tf.cast(traj["action"], tf.float32) # [T, D_a]
            proprio = tf.cast(new_obs["proprio"], tf.float32) # [T, D_p]
            action_left_arm, action_left_gripper = action_abs[:, :6], action_abs[:, 6:7]
            action_right_arm, action_right_gripper = action_abs[:, 7:13], action_abs[:, 13:14]
            proprio_left_arm, proprio_left_gripper = proprio[:, :6], proprio[:, 6:7]
            proprio_right_arm, proprio_right_gripper = proprio[:, 7:13], proprio[:, 13:14]
            
            action_rel = tf.concat([
                action_left_arm - proprio_left_arm,
                action_left_gripper,
                action_right_arm - proprio_right_arm,
                action_right_gripper,
                action_abs[:, 14:],
            ], axis=1)
            traj["action"] = action_rel
        
        traj = {
            "observation": new_obs,
            "task": task,
            "action": tf.cast(traj["action"], tf.float32),
            "dataset_name": tf.repeat(name, traj_len),
            "is_first": traj["is_first"],
            "is_last": traj["is_last"],
            "variant_idx": variant_idx,
            "segment_idx": segment_idx,
        }
        
        if absolute_action_mask is not None:
            if len(absolute_action_mask) != traj["action"].shape[-1]:
                raise ValueError(
                    f"Length of absolute_action_mask ({len(absolute_action_mask)}) "
                    f"does not match action dimension ({traj['action'].shape[-1]})."
                )
            traj["absolute_action_mask"] = tf.tile(
                tf.convert_to_tensor(absolute_action_mask, dtype=tf.bool)[None],
                [traj_len, 1],
            )

        return traj

    builder = tfds.builder(name, data_dir=data_dir)

    # load or compute dataset statistics
    if isinstance(dataset_statistics, str):
        with tf.io.gfile.GFile(dataset_statistics, "r") as f:
            dataset_statistics = json.load(f)
    elif dataset_statistics is None:
        full_dataset = dl.DLataset.from_rlds(
            builder, split="all", shuffle=False, num_parallel_reads=num_parallel_reads
        ).traj_map(restructure, num_parallel_calls)
        # go through train or val again to get the number of train or val splits
        train_dataset = dl.DLataset.from_rlds(
            builder, split=f"train[:{DEFAULT_TRAIN_RATIO}%]" if "val" not in builder.info.splits else "train", 
            shuffle=False, num_parallel_reads=num_parallel_reads
        ).traj_map(restructure, num_parallel_calls)
        val_dataset = dl.DLataset.from_rlds(
            builder, split=f"train[{DEFAULT_TRAIN_RATIO}%:]" if "val" not in builder.info.splits else "val",
            shuffle=False, num_parallel_reads=num_parallel_reads
            ).traj_map(restructure, num_parallel_calls)
        
        # tries to load from cache, otherwise computes on the fly
        dataset_statistics = get_dataset_statistics(
            full_dataset,
            train_dataset,
            val_dataset,
            hash_dependencies=(
                str(builder.info),
                str(state_obs_keys),
                inspect.getsource(standardize_fn) if standardize_fn is not None else "",
                str(action_encoding),
            ),
            save_dir=builder.data_dir,
        )
    
    dataset_statistics = tree_map(np.array, dataset_statistics)

    # skip normalization for certain action dimensions
    if action_normalization_mask is not None:
        if len(action_normalization_mask) != dataset_statistics["action"]["mean"].shape[-1]:
            raise ValueError(
                f"Length of skip_normalization_mask ({len(action_normalization_mask)}) "
                f"does not match action dimension ({dataset_statistics['action']['mean'].shape[-1]})."
            )
        dataset_statistics["action"]["mask"] = np.array(action_normalization_mask)

    # get the split, hardcode here
    if "val" not in builder.info.splits:
        split = f"train[:{DEFAULT_TRAIN_RATIO}%]" if train else f"train[{DEFAULT_TRAIN_RATIO}%:]"
    else:
        split = "train" if train else "val"

    dataset = dl.DLataset.from_rlds(builder, split=split, shuffle=shuffle, num_parallel_reads=num_parallel_reads)
    true_length = dataset_statistics["num_train_transitions"] if train else dataset_statistics["num_val_transitions"]
    
    dataset = dataset.traj_map(restructure, num_parallel_calls)

    if max_trajectories is not None:
        # Calculate the total number of trajectories in the dataset
        total_trajectories = dataset.cardinality().numpy()
        
        # If max_trajectories is a float between 0 and 1, treat it as a percentage
        if isinstance(max_trajectories, float) and 0.0 < max_trajectories <= 1.0:
            max_trajectories = int(total_trajectories * max_trajectories)
            overwatch.info(f"Using {max_trajectories} trajectories ({max_trajectories/total_trajectories:.2%} of total)")
        
        if total_trajectories > max_trajectories and total_trajectories != tf.data.UNKNOWN_CARDINALITY:
            indices = tf.random.shuffle(tf.range(total_trajectories))[:max_trajectories]
            
            def filter_by_indices(idx, traj):
                return tf.reduce_any(tf.equal(idx, indices))
                
            dataset = dataset.enumerate()
            dataset = dataset.filter(filter_by_indices)
            dataset = dataset.map(lambda idx, traj: traj, 
                                num_parallel_calls=num_parallel_calls)
            
            # update dataset_statistics accordingly
            # calculate average transitions per trajectory
            avg_transitions_per_traj = dataset_statistics["num_transitions"] / dataset_statistics["num_trajectories"]
            dataset_statistics["num_trajectories"] = max_trajectories
            dataset_statistics["num_transitions"] = int(max_trajectories * avg_transitions_per_traj)
            
            if train:
                avg_train_transitions_per_traj = dataset_statistics["num_train_transitions"] / dataset_statistics["num_train_trajectories"]
                dataset_statistics["num_train_trajectories"] = max_trajectories
                dataset_statistics["num_train_transitions"] = int(max_trajectories * avg_train_transitions_per_traj)
                true_length = dataset_statistics["num_train_transitions"]
            else:
                avg_val_transitions_per_traj = dataset_statistics["num_val_transitions"] / dataset_statistics["num_val_trajectories"]
                dataset_statistics["num_val_trajectories"] = max_trajectories
                dataset_statistics["num_val_transitions"] = int(max_trajectories * avg_val_transitions_per_traj)
                true_length = dataset_statistics["num_val_transitions"]

    return dataset, dataset_statistics, true_length


def apply_trajectory_transforms(
    dataset: dl.DLataset,
    *,
    train: bool,
    goal_relabeling_strategy: Optional[str] = None,
    goal_relabeling_kwargs: dict = {},
    repeat_last_timestep_ratio: float = 0.0,
    window_size: int = 1,
    future_action_window_size: int = 0,
    subsample_length: Optional[int] = None,
    skip_unlabeled: bool = False,
    max_action: Optional[float] = None,
    max_proprio: Optional[float] = None,
    task_augment_strategy: Optional[str] = None,
    task_augment_kwargs: dict = {},
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> dl.DLataset:
    """
    Applies common transforms that happen at a trajectory level. Such transforms are usually some sort of "relabeling"
    (e.g., filtering, chunking, adding goals, dropping keys).

    Transforms in this function should have the following properties:
        - They require access to an entire trajectory (i.e., they cannot be applied frame-wise).
        - They are generally not CPU-intensive, mostly involving moving and copying data.
        - They do not require decoded images.

    Args:
        dataset (dl.DLataset): The dataset to transform.
        train (bool): Whether the dataset is for training (affects subsampling).
        goal_relabeling_strategy (str, optional): The goal relabeling strategy to use, or None for
            no goal relabeling. See `goal_relabeling.py`.
        goal_relabeling_kwargs (dict, optional): Additional keyword arguments to pass to the goal relabeling function.
        window_size (int, optional): The length of the snippets that trajectories are chunked into.
        future_action_window_size (int, optional): The number of future actions beyond window_size to include
            in the chunked actions.
        subsample_length (int, optional): If provided, trajectories longer than this will be subsampled to
            this length (after goal relabeling and chunking).
        skip_unlabeled (bool, optional): Whether to skip trajectories with no language labels.
        max_action: (float, optional): If provided, trajectories in which *any* action dimension
            of *any* transition has an absolute value larger than this will be skipped.
        max_proprio: (float, optional): If provided, trajectories in which *any* proprio dimension
            of *any* transition has an absolute value larger than this will be skipped.
        task_augment_strategy (str, optional): The task augmentation strategy to use, or None for no task
            augmentation. See `task_augmentation.py`.
        task_augment_kwargs (dict, optional): Additional keyword arguments to pass to the task augmentation
            function.
        num_parallel_calls (int, optional): number of parallel calls for map operations. Default to AUTOTUNE.
    """
    if skip_unlabeled:
        if "language_instruction" not in dataset.element_spec["task"]:
            raise ValueError("skip_unlabeled=True but dataset does not have language labels.")

        dataset = dataset.filter(lambda x: tf.math.reduce_any(x["task"]["language_instruction"] != ""))

    if max_action is not None:
        dataset = dataset.filter(lambda x: tf.math.reduce_all(tf.math.abs(x["action"]) <= max_action))

    if max_proprio is not None and "proprio" in dataset.element_spec["observation"]:
        dataset = dataset.filter(lambda x: tf.math.reduce_all(tf.math.abs(x["observation"]["proprio"]) <= max_proprio))

    # marks which entires of the observation and task dicts are padding
    dataset = dataset.traj_map(traj_transforms.add_pad_mask_dict, num_parallel_calls)

    # updates the "task" dict
    if goal_relabeling_strategy is not None:
        dataset = dataset.traj_map(
            partial(getattr(goal_relabeling, goal_relabeling_strategy), **goal_relabeling_kwargs),
            num_parallel_calls,
        )

    # must run task augmentation before chunking, in case it changes goal timesteps
    if train and task_augment_strategy is not None:
        # perform task augmentation (e.g., dropping keys)
        dataset = dataset.traj_map(
            partial(
                getattr(task_augmentation, task_augment_strategy),
                **task_augment_kwargs,
            ),
            num_parallel_calls,
        )

    # chunks observations and actions, giving them a new axis at index 1 of size `window_size` and
    # `window_size + future_action_window_size`, respectively
    dataset = dataset.traj_map(
        partial(
            traj_transforms.chunk_act_obs,
            window_size=window_size,
            future_action_window_size=future_action_window_size,
        ),
        num_parallel_calls,
    )

    if train and subsample_length is not None:
        dataset = dataset.traj_map(
            partial(traj_transforms.subsample, subsample_length=subsample_length),
            num_parallel_calls,
        )
    
    if train and repeat_last_timestep_ratio > 0.0:
        dataset = dataset.traj_map(
            partial(
                traj_transforms.repeat_last_timestep,
                repeat_ratio=repeat_last_timestep_ratio,
            )
        )

    return dataset


def apply_per_dataset_frame_transforms(
    dataset: dl.DLataset,
    chunk_filter_fn: Optional[Callable] = None,
):
    """
    Optionally applied *per-dataset* transforms that happen at a frame level.

    Args:
        chunk_filter_fn (callable, optional): Filter function for chunks.
    """
    if chunk_filter_fn:
        dataset = dataset.filter(chunk_filter_fn)
    return dataset


def apply_frame_transforms(
    dataset: dl.DLataset,
    *,
    train: bool,
    image_augment_kwargs: Union[Dict, Dict[str, Dict]] = {},
    resize_size: Union[Tuple[int, int], Dict[str, Tuple[int, int]]] = {},
    depth_resize_size: Union[Tuple[int, int], Dict[str, Tuple[int, int]]] = {},
    last_action_drop_prob: float = 0.0,
    last_action_dim: Optional[int] = None,
    proprio_noise_std: float = 0.0,
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> dl.DLataset:
    """
    Applies common transforms that happen at a frame level. These transforms are usually more CPU-intensive, (e.g.,
    decoding or resizing images).

    Args:
        train (bool): Whether the dataset is for training (affects image augmentation).
        dataset (dl.DLataset): The dataset to transform.
        image_augment_kwargs (dict|Mapping[str, dict]): Keyword arguments to pass to the image augmentation
            function. See `dlimp.transforms.augment_image` for documentation of these kwargs. If a dict of
            dicts is provided, then key "k" will be used for "image_{k}" (names determined by `image_obs_keys`
            in `make_dataset_from_rlds`). Augmentation will be skipped for missing keys (so pass an empty dict
            to skip augmentation for all images).
        resize_size (Tuple[int, int]|Mapping[str, Tuple[int, int]]): If provided, images will be resized to
            this size. If a dict of tuples is provided, then key "k" will be used for "image_{k}" (names
            determined by `image_obs_keys` in `make_dataset_from_rlds`). Resizing will be skipped for missing
            keys (so pass an empty dict to skip resizing for all images).
        depth_resize_size (Tuple[int, int]|Mapping[str, Tuple[int, int]]): Same as resize_size, but for depth
            images.
        num_parallel_calls (int): number of parallel calls for frame_map operations. Default to AUTOTUNE.
    """

    # Convenience wrapper that takes a function that operates on a non-chunked "observation" dict and applies
    # it to the chunked "observation" dict as well as the non-chunked "task" dict
    def apply_obs_transform(fn: Callable[[Dict], Dict], frame: Dict) -> Dict:
        frame["task"] = fn(frame["task"])
        frame["observation"] = dl.vmap(fn)(frame["observation"])
        return frame

    # Decode + resize images (and depth images)
    dataset = dataset.frame_map(
        partial(
            apply_obs_transform,
            partial(obs_transforms.decode_and_resize, resize_size=resize_size, depth_resize_size=depth_resize_size),
        ),
        num_parallel_calls,
    )

    if train:
        # Augment all images with the same seed, skipping padding images
        def aug(frame: dict):
            seed = tf.random.uniform([2], maxval=tf.dtypes.int32.max, dtype=tf.int32)
            aug_fn = partial(obs_transforms.augment, seed=seed, augment_kwargs=image_augment_kwargs)
            return apply_obs_transform(aug_fn, frame)

        dataset = dataset.frame_map(aug, num_parallel_calls)

        def random_drop_last_action(frame: dict, last_action_dim: Optional[int] = None, drop_prob: float = 0.0):
            flag = tf.random.uniform([], minval=0, maxval=1, dtype=tf.float32) < drop_prob
            proprio = frame["proprio"]
            proprio_dim = proprio.shape[-1]
            if last_action_dim is None:
                last_action_dim = proprio_dim // 2
            last_action = proprio[..., last_action_dim:]
            if flag:
                last_action = tf.zeros_like(last_action)
            frame["proprio"] = tf.concat([proprio[..., :last_action_dim], last_action], axis=-1)
            return frame
        
        def random_noise_on_proprio(frame: dict, noise_std: float = 0.0):
            proprio = frame["proprio"]
            noise = tf.random.normal(proprio.shape, stddev=noise_std)
            proprio = proprio + noise
            frame["proprio"] = proprio
            return frame

        dataset = dataset.frame_map(
            partial(
                apply_obs_transform,
                partial(random_drop_last_action, last_action_dim=last_action_dim, drop_prob=last_action_drop_prob),
            ),
            num_parallel_calls,
        )

        dataset = dataset.frame_map(
            partial(
                apply_obs_transform,
                partial(random_noise_on_proprio, noise_std=proprio_noise_std),
            ),
            num_parallel_calls,
        )

    return dataset


def make_single_dataset(
    dataset_kwargs: dict,
    *,
    train: bool,
    traj_transform_kwargs: dict = {},
    frame_transform_kwargs: dict = {},
) -> dl.DLataset:
    """Creates a single dataset from kwargs. Returns a dataset of trajectories.

    Args:
        dataset_kwargs: kwargs passed to `make_dataset_from_rlds` that are dataset-specific.
        train: whether this is a training or validation dataset.
        traj_transform_kwargs: kwargs passed to 'apply_trajectory_transforms'.
        frame_transform_kwargs: kwargs passed to 'get_frame_transforms'.
    """
    dataset, dataset_statistics = make_dataset_from_rlds(
        **dataset_kwargs,
        train=train,
    )
    dataset = apply_trajectory_transforms(dataset, **traj_transform_kwargs, train=train)
    dataset = apply_frame_transforms(dataset, **frame_transform_kwargs, train=train)

    # this seems to reduce memory usage without affecting speed
    dataset = dataset.with_ram_budget(1)

    # save for later
    return dataset, dataset_statistics["num_trajectories"], dataset_statistics


# === Core Initializer ===
def make_interleaved_dataset(
    dataset_kwargs_list: List[Dict],
    sample_weights: Optional[List[float]] = None,
    preset_datasets_statistics: Optional[Dict] = None,
    *,
    train: bool,
    shuffle_buffer_size: int,
    traj_transform_kwargs: Optional[Dict] = None,
    frame_transform_kwargs: Optional[Dict] = None,
    batch_size: Optional[int] = None,
    balance_weights: bool = False,
    traj_transform_threads: Optional[int] = None,
    traj_read_threads: Optional[int] = None,
) -> dl.DLataset:
    """
    Creates an interleaved dataset from list of dataset configs (kwargs). Returns a dataset of batched frames.

    Args:
        dataset_kwargs_list: list of kwargs, each element of which is passed to `make_dataset_from_rlds`.
            "num_parallel_calls" and "num_parallel_reads" are overridden using `traj_transform_threads` and
            `traj_read_threads`, respectively.
        sample_weights: sampling weights for each dataset in list. If None, defaults to uniform.
        train: whether this is a training or validation dataset.
        shuffle_buffer_size: size of the dataset shuffle buffer (in number of frames).
        traj_transform_kwargs: kwargs passed to `apply_trajectory_transforms`. "num_parallel_calls" is
            overridden using `traj_transform_threads`.
        frame_transform_kwargs: kwargs passed to `apply_frame_transforms`.
        batch_size: batch size, if not provided output is not batched.
        balance_weights: if True, the sample weights are multiplied by the number of frames in each dataset.
            This makes it so that, if all the sample weights are equal, one full iteration through the interleaved
            dataset will correspond to one full iteration through each individual dataset (only in expectation,
            since in practice the sampling is random).
        traj_transform_threads: total number of parallel calls for trajectory transforms, distributed across
            datasets according to their sampling weights. If None, defaults to AUTOTUNE for every dataset.
        traj_read_threads: total number of parallel read workers for trajectory transforms, distributed across
            datasets according to their sampling weights. If None, defaults to AUTOTUNE for every dataset.
    """
    # Default to uniform sampling (if `sample_weights` is not specified)
    if not sample_weights:
        sample_weights = [1.0] * len(dataset_kwargs_list)

    if len(sample_weights) != len(dataset_kwargs_list):
        raise ValueError(f"sample_weights must be None or have length {len(dataset_kwargs_list)}.")

    # Check valid `traj_transform_kwargs` and `frame_transform_kwargs`
    if (traj_transform_kwargs is None) or (frame_transform_kwargs is None):
        raise ValueError("Missing `traj_transform_kwargs` and `frame_transform_kwargs`!")

    # Get Dataset Sizes
    dataset_sizes, all_dataset_statistics = [], {}
    for dataset_kwargs in dataset_kwargs_list:
        data_kwargs = copy.deepcopy(dataset_kwargs)
        if "dataset_frame_transform_kwargs" in data_kwargs:
            data_kwargs.pop("dataset_frame_transform_kwargs")
        if 'max_trajectories' in data_kwargs:
            data_kwargs.pop('max_trajectories')
        _, dataset_statistics, true_dataset_length = make_dataset_from_rlds(**data_kwargs, train=train)
        # BUG: dataset_sizes is not correct since it's on "all" split, instead of the actual split we are using
        # dataset_sizes.append(dataset_statistics["num_transitions"])
        # FIXED: adding the actual split size to the dataset_statistics
        dataset_sizes.append(true_dataset_length)
        all_dataset_statistics[dataset_kwargs["name"]] = dataset_statistics

    # Get the indices of the "primary" datasets (i.e., datasets with sample_weight == 1.0)
    # XXX not sure if it is used, why only sample_weight == 1.0?
    primary_dataset_indices = np.array([idx for idx in range(len(sample_weights)) if sample_weights[idx] == 1.0])

    # Balance and Normalize Weights
    if balance_weights:
        sample_weights = np.array(sample_weights) * np.array(dataset_sizes)
    sample_weights = np.array(sample_weights) / np.sum(sample_weights)
    pprint_data_mixture(dataset_kwargs_list, sample_weights)

    # Effective Dataset Length = Number of samples until each dataset has completed at least one epoch
    #   =>> Note :: Only counting the "primary" datasets (i.e., datasets with sample_weight == 1.0)
    # XXX not sure if it is used, why only sample_weight == 1.0?
    dataset_len = int((np.array(dataset_sizes) / sample_weights).max())

    # Allocate Threads based on Weights
    threads_per_dataset = allocate_threads(traj_transform_threads, sample_weights)
    reads_per_dataset = allocate_threads(traj_read_threads, sample_weights)

    overwatch.info("Threads per Dataset: %s", threads_per_dataset)
    overwatch.info("Reads per Dataset: %s", reads_per_dataset)

    # Construct Datasets
    overwatch.info("Constructing datasets...")
    datasets = []
    true_datasets_lengths = []  # do not use dataset statistics as it assumes full split
    for dataset_kwargs, threads, reads in zip(
        dataset_kwargs_list,
        threads_per_dataset,
        reads_per_dataset,
    ):
        dataset_frame_transform_kwargs = (
            dataset_kwargs.pop("dataset_frame_transform_kwargs")
            if "dataset_frame_transform_kwargs" in dataset_kwargs
            else {}
        )
        dataset, _, true_dataset_length = make_dataset_from_rlds(
            **dataset_kwargs,
            train=train,
            num_parallel_calls=threads,
            num_parallel_reads=reads,
            dataset_statistics=all_dataset_statistics[dataset_kwargs["name"]],
        )

        # use calculated total statistics
        if preset_datasets_statistics is not None:
            preset_datasets_statistics = preset_datasets_statistics.copy()
            # calculate training total number of transitions and trajectories
            for k in ["num_transitions", "num_trajectories", "num_train_transitions", 
                        "num_train_trajectories", "num_val_transitions", "num_val_trajectories"]:
                preset_datasets_statistics["__total__"][k] = 0
            for name, stat in all_dataset_statistics.items():
                for k in ["num_transitions", "num_trajectories", "num_train_transitions", 
                        "num_train_trajectories", "num_val_transitions", "num_val_trajectories"]:
                    preset_datasets_statistics["__total__"][k] += stat[k]
            for d in all_dataset_statistics:
                if d != "__total__":
                    preset_datasets_statistics[d] = all_dataset_statistics[d]
            all_dataset_statistics = preset_datasets_statistics.copy()
            dataset = dataset.traj_map(
                partial(
                    normalize_action_and_proprio,
                    metadata=all_dataset_statistics["__total__"],
                    normalization_type=dataset_kwargs["action_proprio_normalization_type"],
                    action_encoding=dataset_kwargs["action_encoding"],
                ),
                num_parallel_calls=threads,
            )
        else:
            dataset = dataset.traj_map(
                partial(
                    normalize_action_and_proprio,
                    metadata=all_dataset_statistics[dataset_kwargs["name"]],
                    normalization_type=dataset_kwargs["action_proprio_normalization_type"],
                    action_encoding=dataset_kwargs["action_encoding"],
                ),
                num_parallel_calls=threads,
            )

        true_datasets_lengths.append(true_dataset_length)
        dataset = apply_trajectory_transforms(
            dataset.repeat(),
            **traj_transform_kwargs,
            num_parallel_calls=threads,
            train=train,
        )

        dataset = dataset.flatten(num_parallel_calls=threads)
        dataset = apply_per_dataset_frame_transforms(dataset, **dataset_frame_transform_kwargs)
        datasets.append(dataset)

    # Interleave at the Frame Level
    dataset: dl.DLataset = dl.DLataset.sample_from_datasets(datasets, sample_weights)

    # Validation =>> fix a single shuffle buffer of data and cache it in RAM; prevents gradual memory increase!
    if not train or shuffle_buffer_size == 0:
        # dataset = dataset.take(shuffle_buffer_size).cache()
        pass
    else:
        # Shuffle the Dataset
        #   =>> IMPORTANT :: Shuffle AFTER .cache(), or else memory will still leak!
        dataset = dataset.shuffle(shuffle_buffer_size)

    # Apply Frame Transforms
    overwatch.info("Applying frame transforms on dataset...")
    dataset = apply_frame_transforms(dataset, **frame_transform_kwargs, train=train)

    # [Contract] When training VLA Policies, we let the Collator handle Batching!
    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    # Note =>> Seems to reduce memory usage without affecting speed?
    dataset = dataset.with_ram_budget(1)

    # Save for Later
    dataset.sample_weights = sample_weights

    dataset.dataset_statistics = all_dataset_statistics
    dataset.true_lengths = true_datasets_lengths
    dataset.true_total_length = sum(true_datasets_lengths)

    return dataset, dataset_len, all_dataset_statistics
