from pathlib import Path
import gymnasium as gym
import imageio
import numpy as np
import pickle
import torch
import tyro
import cv2

from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

from galaxea_sim.utils.data_utils import save_dict_list_to_json

import numpy as np

def evaluate(
    task: str,
    pretrained_policy_path: str,
    target_controller_type: str = "bimanual_relaxed_ik",
    device: str = "cuda",
    headless: bool = True,
    num_evaluations: int = 100,
    num_action_steps: int = 16,
    save_video: bool = False,
):
    """Evaluate a pretrained policy in a simulated environment multiple times."""
    output_directory = Path(pretrained_policy_path) / "evaluations" / str(num_action_steps) 
    output_directory.mkdir(parents=True, exist_ok=True)
    with open(Path(pretrained_policy_path) / "dataset_metadata.pkl", "rb") as f:
        dataset_metadata: LeRobotDatasetMetadata = pickle.load(f)
    policy = DiffusionPolicy.from_pretrained(pretrained_policy_path, map_location=device, dataset_stats=dataset_metadata.stats)
    policy.config.n_action_steps = num_action_steps
    infos = []
    env = gym.make(task, control_freq=15, headless=headless, max_episode_steps=500, controller_type = target_controller_type)
    for eval_idx in range(num_evaluations):
        print(f"Starting evaluation {eval_idx + 1}/{num_evaluations}")

        policy.reset()
        numpy_observation, info = env.reset(seed=42)
        if save_video: env.render()

        rewards = []
        frames = []
        if save_video: frames.append(env.render())

        step = 0
        done = False
        
        while not done:
            obs = numpy_observation['upper_body_observations']
            if target_controller_type == "bimanual_joint_position":
                state = torch.cat([
                    torch.from_numpy(obs['left_arm_joint_position']),
                    torch.from_numpy(obs['left_arm_gripper_position']),
                    torch.from_numpy(obs['right_arm_joint_position']),
                    torch.from_numpy(obs['right_arm_gripper_position']),
                ], dim=-1).to(torch.float32).to(device, non_blocking=True).unsqueeze(0)
            else:
                state = torch.cat([
                    torch.from_numpy(obs['left_arm_ee_pose']),
                    torch.from_numpy(obs['left_arm_gripper_position']),
                    torch.from_numpy(obs['right_arm_ee_pose']),
                    torch.from_numpy(obs['right_arm_gripper_position']),
                ], dim=-1).to(torch.float32).to(device, non_blocking=True).unsqueeze(0)
            resized_image = {
                k: cv2.resize(obs[k], (224, 224)) for k in ['rgb_head', 'rgb_left_hand', 'rgb_right_hand']
            }
            images = {
                k: torch.from_numpy(resized_image[k]).to(torch.float32).permute(2, 0, 1) / 255.0
                for k in ['rgb_head', 'rgb_left_hand', 'rgb_right_hand']
            }
            images = {k: v.to(device, non_blocking=True).unsqueeze(0) for k, v in images.items()}

            observation = {"observation.state": state, **{f"observation.images.{k}": v for k, v in images.items()}}
            
            with torch.inference_mode():
                action = policy.select_action(observation)

            numpy_observation, reward, terminated, truncated, info = env.step(action.squeeze(0).cpu().numpy())
            rewards.append(reward)
            if save_video: frames.append(env.render())
            done = terminated or truncated or done
            step += 1

        print("Success!" if terminated else "Failure!")
        infos.append(info)
        save_dict_list_to_json(infos, output_directory / "info.json")
        if save_video:
            fps = env.unwrapped.control_freq
            video_path = output_directory / f"rollout_{eval_idx + 1}.mp4"
            imageio.mimsave(str(video_path), np.stack(frames), fps=fps)
            print(f"Video saved at {video_path}")

if __name__ == "__main__":
    tyro.cli(evaluate)
