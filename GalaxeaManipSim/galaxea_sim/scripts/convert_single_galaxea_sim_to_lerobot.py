import shutil

from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from typing import Literal
import tyro
import glob
import h5py
import numpy as np
import cv2

REPO_PREFIX = "galaxea"

def main(task: str, data_dir: str = "datasets", tag: str | None = None, robot: Literal['r1', 'r1_pro'] = 'r1', use_eef: bool = False, use_video: bool = False, push_to_hub: bool = False):
    output_path = HF_LEROBOT_HOME / REPO_PREFIX / task
    if output_path.exists():
        shutil.rmtree(output_path)
    shape = (224, 224, 3)  # Resize images to 224x224
    arm_dof = 7 if (robot == 'r1_pro' or use_eef) else 6
    
    dataset = LeRobotDataset.create(
        repo_id=f"{REPO_PREFIX}/{output_path.name}",
        robot_type=robot,
        fps=15,
        features={
            "observation.images.rgb_head": {
                "dtype": "image" if not use_video else "video",
                "shape": shape,
                "names": ["width", "height", "channel"],
            },
            "observation.images.rgb_left_hand": {
                "dtype": "image" if not use_video else "video",
                "shape": shape,
                "names": ["width", "height", "channel"],
            },
            "observation.images.rgb_right_hand": {
                "dtype": "image" if not use_video else "video",
                "shape": shape,
                "names": ["width", "height", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (arm_dof * 2 + 2,),
                "names": None,
            },
            "action": {
                "dtype": "float32",
                "shape": (arm_dof * 2 + 2,),
                "names": None,
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )
    for raw_dataset_name in [task]:
        if tag:
            h5_paths = glob.glob(f"{data_dir}/{raw_dataset_name}/{tag}/*.h5", recursive=True)
        else:
            h5_paths = glob.glob(f"{data_dir}/{raw_dataset_name}/**/*.h5", recursive=True)
        for h5_path in h5_paths:
            with h5py.File(h5_path, 'r') as f:
                rgb_head = f['upper_body_observations']['rgb_head'][()]
                rgb_left_hand = f['upper_body_observations']['rgb_left_hand'][()]
                rgb_right_hand = f['upper_body_observations']['rgb_right_hand'][()]
                
                # resize images to shape
                rgb_head_resized = np.array([cv2.resize(img, (shape[1], shape[0])) for img in rgb_head])
                rgb_left_hand_resized = np.array([cv2.resize(img, (shape[1], shape[0])) for img in rgb_left_hand])
                rgb_right_hand_resized = np.array([cv2.resize(img, (shape[1], shape[0])) for img in rgb_right_hand])
                
                left_arm_joint_position = f['upper_body_observations']['left_arm_joint_position'][()]
                left_arm_gripper_position = f['upper_body_observations']['left_arm_gripper_position'][()]
                right_arm_joint_position = f['upper_body_observations']['right_arm_joint_position'][()]
                right_arm_gripper_position = f['upper_body_observations']['right_arm_gripper_position'][()]

                if use_eef:
                    left_arm_ee_pose = f['upper_body_observations']['left_arm_ee_pose'][()]
                    right_arm_ee_pose = f['upper_body_observations']['right_arm_ee_pose'][()]
                
                left_arm_state, right_arm_state = (left_arm_ee_pose, right_arm_ee_pose) if use_eef else (left_arm_joint_position, right_arm_joint_position)

                state = np.concatenate(
                    [left_arm_state, left_arm_gripper_position, right_arm_state, right_arm_gripper_position],  # type: ignore
                    axis=-1
                )
                
                left_arm_joint_position_cmd = f['upper_body_action_dict']['left_arm_joint_position_cmd'][()]
                left_arm_gripper_position_cmd = f['upper_body_action_dict']['left_arm_gripper_position_cmd'][()]
                right_arm_joint_position_cmd = f['upper_body_action_dict']['right_arm_joint_position_cmd'][()]
                right_arm_gripper_position_cmd = f['upper_body_action_dict']['right_arm_gripper_position_cmd'][()]

                if use_eef:
                    left_arm_ee_pose_cmd = f['upper_body_action_dict']['left_arm_ee_pose_cmd'][()]
                    right_arm_ee_pose_cmd = f['upper_body_action_dict']['right_arm_ee_pose_cmd'][()]

                left_arm_action, right_arm_action = (left_arm_ee_pose_cmd, right_arm_ee_pose_cmd) if use_eef else (left_arm_joint_position_cmd, right_arm_joint_position_cmd)
                
                action = np.concatenate(
                    [left_arm_action, left_arm_gripper_position_cmd, right_arm_action, right_arm_gripper_position_cmd],  # type: ignore
                    axis=-1
                )
                
                episode_length = rgb_head.shape[0]
                for i in range(episode_length):
                    dataset.add_frame(
                        {
                            "observation.images.rgb_head": rgb_head_resized[i],
                            "observation.images.rgb_left_hand": rgb_left_hand_resized[i],
                            "observation.images.rgb_right_hand": rgb_right_hand_resized[i],
                            "observation.state": state[i].astype(np.float32),
                            "action": action[i].astype(np.float32),
                            
                        },
                        task=raw_dataset_name,
                    )
            dataset.save_episode()            
    print(f"Dataset {output_path.name} created successfully with {len(dataset)} frames.")
    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        raise NotImplementedError("Pushing to the Hugging Face Hub is not supported yet.")


if __name__ == "__main__":
    tyro.cli(main)