from collections import deque
import copy
from omegaconf import OmegaConf
import random
import os
import math

from typing import Callable, Dict, List, Optional, Tuple, Union, Sequence

from PIL import Image

import torch
import numpy as np
import tensorflow as tf
from vla.load import load_from_checkpoint
from vla.helper import log_execution_time

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)

CAMERA_VIEW_NAME_LUT = {
    "head": "head_rgb",
    "wrist_left": "left_hand_rgb",
    "wrist_right": "right_hand_rgb",
}

import time

class GalaxeaZeroPolicy:
    def __init__(self, cfg_file, checkpoint_path, seed=42, device="cuda", dtype="fp32", use_torch_compile=False, use_remote=False, port=8000):
        cfg = OmegaConf.load(cfg_file)
        OmegaConf.resolve(cfg)
        self.cfg = cfg
        self.seed = seed
        if dtype == "fp16":
            self.dtype = torch.float16
        elif dtype == "bf16":
            self.dtype = torch.bfloat16
        elif dtype == "fp32":
            self.dtype = torch.float32
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        set_seed_everywhere(seed)

        self.device = device

        self.To = cfg.MODEL.cond_steps
        self.Ta = cfg.MODEL.horizon_steps
        self.camera_views = [CAMERA_VIEW_NAME_LUT[k] for k in cfg.DATASET.camera_views]
        self.action_dim = cfg.MODEL.action_dim
        self.proprio_dim = cfg.MODEL.proprio_dim

        assert cfg.model_family == "galaxea_zero"

        model = load_from_checkpoint(checkpoint_path,
                                    load_for_training=False,
                                    action_expert_only=cfg.MODEL.action_expert_only,
                                    model_cfg=cfg.MODEL)
        self.client = None
        model = model.to(self.dtype)
        model.eval()
        if use_torch_compile:
            model = torch.compile(
                model, mode="default"
            )
        self.model = model.to(device)

        # hard-coded for now
        self.img_resize_size = (224, 224)

        self.unnorm_key = "__total__" if cfg.DATASET.get("use_pretrained_data_stats", False) else cfg.dataset_name
        self.unnorm_type = cfg.DATASET.get("action_proprio_normalization_type", "q99")

    @log_execution_time()
    @torch.no_grad()
    def infer(
        self, 
        obs: Union[Dict, List[Dict]], 
        instruction: str="place the gray block in the middle.", 
        binarize_gripper: bool=False,
        center_crop: bool=False,
    ):
        """
        Args:
            obs: dictionary or queue of dictionaries of observations, including 
                "head_rgb": (h, w, 3),
                "left_hand_rgb": (h, w, 3), optional
                "right_hand_rgb": (h, w, 3), optional
                "/motion_control/pose_ee_arm_left": (7,),
                "/motion_control/pose_ee_arm_right": (7,),
                "/hdas/feedback_gripper_left": scalar,
                "/hdas/feedback_gripper_right": scalar,
            instruction: string instruction, for task053 should be "place the gray block in the middle."
            binarize_gripper: bool, whether to binarize gripper action to 0 or 1, default False.
        
        Returns:
            action: (Ta, action_dim)
        """

        # some preprocessing
        if isinstance(obs, dict):
            obs = [obs]
        else:
            assert isinstance(obs, list)
        assert len(obs) == self.To, f"Expected {self.To} frame of observations, got {len(obs)}"

        new_obs = []
        for ob in obs:
            for k, v in ob.items():
                if isinstance(v, torch.Tensor):
                    v = v.squeeze()
                    if str(v.device) != "cpu":
                        v = v.cpu()
                    ob[k] = v.numpy()
                elif isinstance(v, list):
                    ob[k] = np.array(v)
                else:
                    assert isinstance(v, np.ndarray)
            new_obs.append(ob)
        obs = new_obs

        ################# Images #################
        # we only need to resize the images, center crop is done in the model
        imgs = {}
        for cam in self.camera_views:
            imgs[cam] = []
            for ob in obs:
                image = ob[cam]
                if image.shape[-1] != 3:
                    image = np.transpose(image, (1, 2, 0)) # from (c, h, w) to (h, w, c)

                image = tf.image.convert_image_dtype(image, tf.uint8)
                image = tf.image.resize(image, self.img_resize_size, method="lanczos3", antialias=True)
                image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
                imgs[cam].append(image.numpy())

        ################# Proprio #################
        # 1. construct proprio
        proprios = []
        for i in range(self.To):
            joint_position_arm_left = obs[i]["/hdas/feedback_arm_left/position"]
            joint_position_arm_right = obs[i]["/hdas/feedback_arm_right/position"]

            joint_velocity_arm_left = obs[i]["/hdas/feedback_arm_left/velocity"]
            joint_velocity_arm_right = obs[i]["/hdas/feedback_arm_right/velocity"]

            gripper_state_left = obs[i]["/hdas/feedback_gripper_left"]
            gripper_state_left = gripper_state_left.item() if isinstance(gripper_state_left, np.ndarray) else float(gripper_state_left)
            gripper_state_right = obs[i]["/hdas/feedback_gripper_right"]
            gripper_state_right = gripper_state_right.item() if isinstance(gripper_state_right, np.ndarray) else float(gripper_state_right)

            joint_position_torso = obs[i]["/hdas/feedback_torso"]
            base_velocity = obs[i]["/hdas/feedback_chassis"]

            last_action = obs[i]["last_action"]

            p = np.concatenate([
                joint_position_arm_left,
                # joint_velocity_arm_left,
                [gripper_state_left],
                joint_position_arm_right,
                # joint_velocity_arm_right,
                [gripper_state_right],
                joint_position_torso,
                base_velocity,
            ])
            proprios.append(p)

        ################# Forward #################
        if self.client is not None:

            while True:
                try:
                    actions = self.client.infer(
                        dict(
                            images=imgs,
                            instruction=instruction,
                            proprio=proprios,
                            unnorm_key=self.unnorm_key,
                            unnorm_type=self.unnorm_type,
                            select_keys=self.camera_views,
                            center_crop=center_crop,
                            autocast_dtype=str(self.dtype).strip("torch.")
                        )
                    )
                    if isinstance(actions, dict):
                        actions = actions["actions"]
                    break
                except:
                    print("Error in inference, retrying...")
                    time.sleep(5)
                    self.client = _websocket_client_policy.WebsocketClientPolicy(self.host, self.port)
                    print("Retrying...")
                    continue

        else:
            proprios = torch.tensor(np.array(proprios), dtype=torch.float32) # (To, 14)
            actions = self.model.predict_action(
                images=imgs,
                instruction=instruction,
                proprio=proprios,
                unnorm_key=self.unnorm_key,
                unnorm_type=self.unnorm_type,
                select_keys=self.camera_views,
                center_crop=center_crop,
                autocast_dtype=self.dtype
            )

        # 4. denormalize grippers
        action_arm_left = actions[:, :6]
        action_gripper_left = actions[:, 6:7]
        action_arm_right = actions[:, 7:13]
        action_gripper_right = actions[:, 13:14]
        action_torso = actions[:, 14:20]
        action_chassis = actions[:, 20:26]

        action_gripper_left = np.clip(action_gripper_left, 0.0, 1.0)
        action_gripper_right = np.clip(action_gripper_right, 0.0, 1.0)
        if binarize_gripper:
            action_gripper_left = (action_gripper_left > 0.5).astype(np.float32)
            action_gripper_right = (action_gripper_right > 0.5).astype(np.float32)

        # see prismatic.vla.datasets.rlds.oxe.transforms.galaxea_dataset_transform
        action_gripper_left = action_gripper_left * (100.0 - 0.0) + 0.0 
        action_gripper_right = action_gripper_right * (100.0 - 0.0) + 0.0

        actions = np.concatenate([
            action_arm_left, action_gripper_left,
            action_arm_right, action_gripper_right,
            action_torso, action_chassis
        ], axis=-1) # (Ta, 26)

        return actions


def quat_to_rpy(quaternion):
    """
    Convert a quaternion [x, y, z, w] to roll, pitch, yaw (RPY).
    
    Parameters
    ----------
    quaternion : list or ndarray
        Quaternion as [x, y, z, w].
    
    Returns
    -------
    rpy : list or ndarray
        Rotation as [roll, pitch, yaw] in radians.
    """
    x, y, z, w = quaternion

    # Compute roll (X-axis rotation)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x**2 + y**2)
    roll = np.arctan2(t0, t1)

    # Compute pitch (Y-axis rotation)
    t2 = 2.0 * (w * y - z * x)
    t2 = np.clip(t2, -1.0, 1.0)  # Clamp to avoid numerical issues
    pitch = np.arcsin(t2)

    # Compute yaw (Z-axis rotation)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y**2 + z**2)
    yaw = np.arctan2(t3, t4)

    return np.array([roll, pitch, yaw])

def rpy_to_quat(rpy):
    """
    Convert roll, pitch, yaw (RPY) to a quaternion [x, y, z, w].
    
    Parameters
    ----------
    rpy : list or ndarray
        Rotation as [roll, pitch, yaw] in radians.
    
    Returns
    -------
    quaternion : list or ndarray
        Quaternion as [x, y, z, w].
    """

    roll, pitch, yaw = rpy

    # Quaternion from RPY
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr
    w = cy * cp * cr + sy * sp * sr

    return np.array([x, y, z, w])

def pose_quat_to_euler(pose):
    pos = pose[:3]
    quat = pose[3:]
    euler = quat_to_rpy(quat)
    return np.concatenate([pos, euler])

def set_seed_everywhere(seed: int):
    """Sets the random seed for Python, NumPy, and PyTorch functions."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def visualize_one_episode(pred_actions, gt_actions, save_name="vis.png"):
    """
    Visualize one episode of predicted and ground-truth actions.
    Params:
        pred_actions (Tensor[seq_len, chunk_size, d])
        gt_actions (Tensor[seq_len, d])
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import numpy as np

    seq_len, chunk_size, d = pred_actions.shape

    fig, axes = plt.subplots(nrows=d, ncols=1, figsize=(10, 3 * 8))
    for dim in range(pred_actions.shape[-1]):
        chunk_collection = [np.column_stack([np.arange(i, i + chunk_size), pred_actions[i, :, dim]]) for i in range(seq_len)]
        chunk_collection = LineCollection(chunk_collection, color="b", linewidth=1)
        axes[dim].add_collection(chunk_collection)

        if gt_actions is not None:
            axes[dim].plot(np.arange(gt_actions.shape[0]), gt_actions[:, dim], marker='o', linestyle='-', color='r', markersize=2, label='action')
        axes[dim].legend(loc="upper right")
        axes[dim].set_title(dim)
        axes[dim].set_xlim([0, seq_len*1.05])

        y_min = pred_actions[:,:,dim].min()
        y_max = pred_actions[:,:,dim].max()
        if gt_actions is not None:
            y_min = min(y_min, gt_actions[:, dim].min())
            y_max = max(y_max, gt_actions[:, dim].max())
        axes[dim].set_ylim([y_min, y_max])
    
    plt.tight_layout()
    plt.savefig(save_name, format='png', dpi=300)
    print("save to ", save_name)

if __name__ == "__main__":
    pass


