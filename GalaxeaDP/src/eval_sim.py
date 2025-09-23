from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import sys
import os
import cv2
import json
import hydra
import lightning as L
import rootutils
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import plotly.graph_objects as go

from pathlib import Path
import gymnasium as gym
import imageio
import numpy as np
import pickle
import torch
import tyro
import cv2

from galaxea_sim.utils.data_utils import save_dict_list_to_json

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("max", lambda x: max(x))
OmegaConf.register_new_resolver("split", lambda s, idx: s.split('/')[int(idx)])

# Add the project root directory to the Python path
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from src.data.galaxea_lerobot_dataset import GalaxeaLerobotDataset
from src.utils.pytorch_utils import dict_apply
from src.utils.visualize import plot_result


def make_single_frame_batch_eef(env_obs, device):
    # print(env_obs.keys())
    obs_map = {
        "head_rgb":        env_obs["rgb_head"],
        "left_wrist_rgb":  env_obs["rgb_left_hand"],
        "right_wrist_rgb": env_obs["rgb_right_hand"],
        "head_depth":      env_obs["depth_head"],
        "left_ee_pose":    env_obs["left_arm_ee_pose"],       # (x,y,z,qx,qy,qz,qw)
        "left_gripper":    env_obs["left_arm_gripper_position"][None],  # (1,)
        "right_ee_pose":   env_obs["right_arm_ee_pose"],
        "right_gripper":   env_obs["right_arm_gripper_position"][None],
    }

    obs = {}
    # --- images ---
    for k in ["head_rgb","left_wrist_rgb","right_wrist_rgb"]:
        img = cv2.resize(obs_map[k], (224,224))
        t = torch.from_numpy(img).float().permute(2,0,1)
        # t = torch.from_numpy(img).permute(2,0,1)
        obs[k] = t.unsqueeze(0).unsqueeze(0).to(device)                     # (1,1,3,224,224)

    # --- states ---
    for k in ["left_ee_pose","left_gripper","right_ee_pose","right_gripper"]:
        arr = torch.from_numpy(obs_map[k]).float()
        obs[k] = arr.reshape(1, 1, -1).to(device) 

    batch = {"obs": obs}

    return batch

def make_single_frame_batch_joints(env_obs, device):
    # print(env_obs.keys())
    obs_map = {
        "head_rgb":        env_obs["rgb_head"],
        "left_wrist_rgb":  env_obs["rgb_left_hand"],
        "right_wrist_rgb": env_obs["rgb_right_hand"],
        "head_depth":      env_obs["depth_head"],
        "left_arm_joints":    env_obs["left_arm_joint_position"],       # (x,y,z,qx,qy,qz,qw)
        "left_gripper":    env_obs["left_arm_gripper_position"][None],  # (1,)
        "right_arm_joints":   env_obs["right_arm_joint_position"],
        "right_gripper":   env_obs["right_arm_gripper_position"][None],
    }

    obs = {}
    # --- images ---
    for k in ["head_rgb","left_wrist_rgb","right_wrist_rgb"]:
        img = cv2.resize(obs_map[k], (224,224))
        t = torch.from_numpy(img).float().permute(2,0,1)
        # t = torch.from_numpy(img).permute(2,0,1)
        obs[k] = t.unsqueeze(0).unsqueeze(0).to(device)                     # (1,1,3,224,224)

    # --- states ---
    for k in ["left_arm_joints","left_gripper","right_arm_joints","right_gripper"]:
        arr = torch.from_numpy(obs_map[k]).float()
        obs[k] = arr.reshape(1, 1, -1).to(device) 

    batch = {"obs": obs}

    return batch

def save_video_ffmpeg(video_path, frames, fps):
    frames_np = np.stack(frames).astype(np.uint8)
    if frames_np.shape[-1] == 4:
        frames_np = frames_np[..., :3] 
    fps = int(fps)

    h, w = frames_np.shape[1:3]
    with imageio.get_writer(
        str(video_path),
        fps=fps,
        format="ffmpeg",
        codec="libx264",
        pixelformat="yuv420p"
    ) as writer:
        for f in frames_np:
            writer.append_data(f)



@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig,
    device: str = "cuda",
    headless: bool = False,
    num_evaluations: int = 5,
    num_action_steps: int = 16,
    save_video: bool = True,
):
    """Evaluate a pretrained policy in a simulated environment multiple times."""
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
    output_dir = Path(os.path.abspath(os.path.expanduser(cfg.paths.output_dir)))
    output_dir.mkdir(exist_ok=True)
    print(f"Output dir: {output_dir}")

    env = cfg.env
    target_controller_type = cfg.target_controller_type
    # load model
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    model.load_state_dict(torch.load(cfg.ckpt_path, map_location="cuda:0", weights_only=False)["state_dict"])
    policy: DiffusionUnetImagePolicy = model.policy.cuda().eval()

    num_success = 0 # Initialize success count
    infos = []

    env = gym.make(env, control_freq=30, headless=headless, max_episode_steps=600, controller_type = target_controller_type)
    for eval_idx in range(num_evaluations):
        print(f"Starting evaluation {eval_idx + 1}/{num_evaluations}")

        # policy.reset() # how to reset
        numpy_observation, info = env.reset(seed=42)
        if save_video: env.render()

        rewards = []
        frames = []
        actions_log = []
        if save_video: frames.append(env.render())

        step = 0
        done = False
        batch = []

        # Initialize action sequence and index
        action_seq: Optional[np.ndarray] = None   # shape = (num_action_steps, action_dim)
        seq_idx: int = 0

        while not done:
            # New Inference Loop
            if action_seq is None or seq_idx >= len(action_seq):
                if target_controller_type == 'bimanual_joint_position':
                    # Create batch for joint position policy
                    batch = make_single_frame_batch_joints(
                        numpy_observation['upper_body_observations'],
                        device
                    )
                elif target_controller_type == 'bimanual_relaxed_ik':
                    batch = make_single_frame_batch_eef(
                        numpy_observation['upper_body_observations'],
                        device
                    )

                with torch.inference_mode():
                    pred = policy.predict_action(batch)          # (1, horizon, action_dim)
                action_seq = pred[:,:num_action_steps,:].squeeze(0).cpu().numpy()       # (horizon, action_dim)
                seq_idx = 0

            # Select action from the sequence
            action = action_seq[seq_idx]
            seq_idx += 1

            # Update environment with the selected action
            numpy_observation, reward, terminated, truncated, info = env.step(action)
            
            actions_log.append(action[:8])
            rewards.append(reward)
            if save_video: 
                frames.append(env.render())
                assert frames[-1] is not None, "Rendering failed, check the environment setup."
            done = terminated or truncated or done
            step += 1


        if terminated:
            print("Success!")
            num_success += 1
        else:
            print("Failure!")
        infos.append(info)
        save_dict_list_to_json(infos, output_dir / "info.json")
        if save_video:
            fps = env.unwrapped.control_freq
            video_path = output_dir / f"rollout_{eval_idx + 1}.mp4"
            # imageio.mimsave(str(video_path), np.stack(frames), fps=fps)
            save_video_ffmpeg(video_path, frames, env.unwrapped.control_freq)
            print(f"Video saved at {video_path}")

        # Plot actions
        actions_array = np.stack(actions_log)  # [T, 8]
        fig, axes = plt.subplots(8, 1, figsize=(10, 14), sharex=True)
        time_steps = np.arange(actions_array.shape[0])
        for i in range(8):
            ax = axes[i]
            ax.plot(time_steps, actions_array[:, i], label=f'action[{i}]', color='tab:blue')
            ax.set_ylabel(f'{"xyzwxyzG"[i]}')
            ax.grid(True)
            if i == 0:
                ax.legend(loc='upper right', fontsize='small')
        
        axes[-1].set_xlabel("Time Step")
        fig.suptitle(f"Joint Action Trajectories with Limits (Eval {eval_idx + 1})", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        plot_path = output_dir / f"rollout_{eval_idx + 1}.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Action plot saved at {plot_path}")

    print(f"Success rate: {num_success / num_evaluations * 100:.2f}% ({num_success}/{num_evaluations})")

if __name__ == "__main__":
    main()