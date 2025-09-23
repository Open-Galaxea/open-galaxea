import torch
import numpy as np
from typing import List, Literal, Dict
from tqdm import tqdm
from copy import deepcopy
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.models.policy.signal_transform import SignalTransform
from src.utils.diffusion_policy import LinearNormalizer
from src.utils.normalize_utils import get_range_normalizer_from_stat
from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class GalaxeaLerobotDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        dataset_dirs: List[str], 
        shape_meta: Dict[str, Dict], 
        action_keys: List[str],
        qpos_keys: List[str],
        chunk_size: int, 
        vision_obs_size: int=1,
        qpos_obs_size: int=1, 
        rotation_type: Literal["quaternion", "rotation_6d", "rotation_9d"]="quaternion", 
        use_relative_control: bool=False, 
        val_set_proportion: float=0.0, 
        is_training: bool=True, 
        use_cache: bool=False, # make sure num_workers=1
    ):
        meta = LeRobotDatasetMetadata(repo_id="", root=dataset_dirs[0])
        fps = meta.fps
        self.image_keys = ["head_rgb", "left_wrist_rgb", "right_wrist_rgb"]
        delta_timestamps = {f"observation.images.{key}":[t / fps for t in range(0, -vision_obs_size, -1)] for key in self.image_keys}

        for key in qpos_keys:
            delta_timestamps.update({f"observation.state.{key}": [t / fps for t in range(0, -qpos_obs_size, -1)],})
        for key in action_keys:
            delta_timestamps.update({f"action.{key}": [t / fps for t in range(0, chunk_size, 1)],})

        self.datasets = []
        for root in dataset_dirs:
            meta = LeRobotDatasetMetadata(repo_id="", root=root)
            ratio = np.round((1 - val_set_proportion) / val_set_proportion).astype(int)
            if is_training:
                episodes = [0] + [i for i in range(meta.total_episodes) if i % ratio != 0]
            else:
                episodes = list(range(ratio, meta.total_episodes, ratio))

            self.datasets.append(
                LeRobotDataset(
                    repo_id="",
                    root=root,
                    episodes=episodes,
                    delta_timestamps=delta_timestamps,
                    # video_backend="pyav",
                )
            )
        self.during_training = True
        self.action_keys = action_keys
        self.qpos_keys = qpos_keys
        self.rotation_type = rotation_type
        self.use_relative_control = use_relative_control

        episode_data_index_from = []
        episode_data_index_to = []
        for dataset_idx in range(len(dataset_dirs)):
            episode_data_index_from.extend(self.datasets[dataset_idx].episode_data_index["from"].numpy().tolist())
            episode_data_index_to.extend(self.datasets[dataset_idx].episode_data_index["to"].numpy().tolist())
        self.episode_data_index = {
            "from": episode_data_index_from,
            "to": episode_data_index_to,
        }
        episode_sizes = [0]
        for i in range(len(episode_data_index_from)):
            episode_sizes.append(episode_data_index_to[i] - episode_data_index_from[i])
        self.episode_divs = np.cumsum(episode_sizes)[:-1]
        self.num_episodes = len(episode_data_index_from)
        log.info(f"Dataset_dir: {dataset_dirs} episode_num: {self.num_episodes}, timesteps: {len(self)}")

        if use_relative_control:
            self.episode_start_ee_poses = []
            ee_pose_keys = [key for key in self.qpos_keys if "ee_pose" in key]

            for dataset_idx in range(len(dataset_dirs)):
                dataset_episode_start_ee_poses = []
                ep_index_np = self.datasets[dataset_idx].episode_data_index["from"].numpy().tolist()
                for start in ep_index_np:
                    episodes_start_ee_pose = {f"episode_start_{key}": self.datasets[dataset_idx][start][f"observation.state.{key}"] for key in ee_pose_keys}
                    dataset_episode_start_ee_poses.append(episodes_start_ee_pose)
                self.episode_start_ee_poses.append(dataset_episode_start_ee_poses)
        
        self.use_cache = use_cache
        if use_cache:
            self.cache = {}

    def __len__(self):
        return sum(d.num_frames for d in self.datasets)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        if self.use_cache and idx in self.cache:
            return self.cache[idx]
        # Determine which dataset to get an item from based on the index.
        start_idx = 0
        dataset_idx = 0
        for dataset in self.datasets:
            if idx >= start_idx + dataset.num_frames:
                start_idx += dataset.num_frames
                dataset_idx += 1
                continue
            break
        else:
            raise AssertionError("We expect the loop to break out as long as the index is within bounds.")
        lerobot_sample = self.datasets[dataset_idx][idx - start_idx]
        sample = {"action": {}, "obs": {}, "mcap": lerobot_sample["task"], "step": idx - start_idx}

        for key in self.qpos_keys:
            action = lerobot_sample[f"observation.state.{key}"]
            action = action.unsqueeze(-1) if len(action.shape) == 1 else action # gripper
            sample["obs"][key] = action

        for key in self.action_keys:
            action = lerobot_sample[f"action.{key}"]
            action = action.unsqueeze(-1) if len(action.shape) == 1 else action # gripper
            sample["action"][key] = action

        sample["gt_action"] = deepcopy(sample["action"])

        if self.use_relative_control:
            ep_idx = self.datasets[dataset_idx].episodes.index(lerobot_sample["episode_index"].item())
            sample["obs"].update(self.episode_start_ee_poses[dataset_idx][ep_idx])

        if self.during_training:
            for img_key in self.image_keys:
                img = lerobot_sample[f"observation.images.{img_key}"]
                if len(img.shape) == 3: # when using video, the time dim will lost
                    img = img.unsqueeze(0)
                sample["obs"][img_key] = (img * 255).to(torch.uint8) # (1, 3, H, W)

        if self.use_cache:
            self.cache[idx]=sample
        return sample

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        """Will be call by DiffusionPolicyBCModule.setup() to set DiffusionUnetImagePolicy.normalizer"""
        stats = self.get_norm_stats()
        normalizer = LinearNormalizer()
        normalizer["action"] = get_range_normalizer_from_stat(stats["action"], **kwargs)
        normalizer["qpos"] = get_range_normalizer_from_stat(stats["qpos"], **kwargs)
        return normalizer

    def get_norm_stats(self):
        self.during_training = False

        # policy will init its own signal_transform with same parameters.
        signal_transform = SignalTransform(
            action_keys=self.action_keys, 
            qpos_keys=self.qpos_keys, 
            rotation_type=self.rotation_type, 
            use_relative_control=self.use_relative_control, 
        )
        dataloader = torch.utils.data.DataLoader(
            dataset=self, 
            batch_size=32, 
            num_workers=32, 
            shuffle=False, 
        )
        qpos_min, qpos_max, action_min, action_max = [], [], [], []
        for batch in tqdm(dataloader, desc='Iterating dataset to get normalization'):
            batch = signal_transform.forward(batch)
            qpos = batch["obs"]["qpos"][:, 0, :]
            qpos_min.append(qpos.amin(0))
            qpos_max.append(qpos.amax(0))
            action = batch["action"][:, 0, :]
            action_min.append(action.amin(0))
            action_max.append(action.amax(0))

        qpos_min = torch.stack(qpos_min).amin(0)
        qpos_max = torch.stack(qpos_max).amax(0)
        action_min = torch.stack(action_min).amin(0)
        action_max = torch.stack(action_max).amax(0)

        norm_stats = dict(
            qpos = dict(min=qpos_min, max=qpos_max), 
            action = dict(min=action_min, max=action_max), 
        )

        self.during_training = True
        del signal_transform
        del dataloader

        return norm_stats


