import datetime
import uuid

import gymnasium as gym
import h5py
import numpy as np
import tyro
import tqdm
from loguru import logger
from pathlib import Path

import galaxea_sim.envs
from galaxea_sim.envs.base.bimanual_manipulation import BimanualManipulationEnv
from galaxea_sim.utils.data_utils import save_dict_list_to_hdf5, save_dict_list_to_json 
import json

def main(
    env_name: str, 
    num_demos: int = 100, 
    dataset_dir: str = 'datasets', 
    target_controller_type: str = 'bimanual_relaxed_ik', 
    control_freq: int = 15, 
    headless: bool = True, 
    ray_tracing: bool = False
):
    env = gym.make(
        env_name,
        control_freq=control_freq,
        headless=headless,
        controller_type=target_controller_type,
        ray_tracing=ray_tracing,
    )
    assert isinstance(env.unwrapped, BimanualManipulationEnv)
    save_dir = Path(dataset_dir) / env_name / "replayed"
    source_dir = Path(dataset_dir) / env_name
    meta_info_path = Path(source_dir) / "meta_info.json"
    h5_paths = list(Path(source_dir).glob("*/*.h5")) # except for final in the source_dir
    h5_paths = [
        h5_path for h5_path in h5_paths if "final" not in str(h5_path)
    ]
    num_collected = 0
    num_tries = 0
    meta_info_list = []
    logger.info(f"Collecting {num_demos} demos from {len(h5_paths)} h5 files.")
    # for h5_path in h5_paths:
    # set pbar for num_collected
    pbar = tqdm.tqdm(total=num_demos, desc="Collecting demos")


    # get existing demo h5
    existing = sorted(save_dir.glob("demo_*.h5"))
    num_collected = len(existing)
    pbar = tqdm.tqdm(total=num_demos, initial=num_collected,
                    desc="Collecting demos")

    # skip h5 that has already been processed
    processed = {int(p.stem.split("_")[-1]) for p in existing}
    h5_paths = [p for p in h5_paths
                if int(p.stem.split("_")[-1]) not in processed]
    # ----------------------------------------------------

    for h5_path in h5_paths:
        # TODO: add a check to ensure the h5 file is not already processed
        h5_file = h5py.File(h5_path, "r")
        demo_idx = int(h5_path.stem.split("_")[-1])

        meta_info_path = h5_path.parent / "meta_info.json"
        if meta_info_path.exists():
            with open(meta_info_path, "r") as f:
                source_meta_info_list = json.load(f)
        else:
            source_meta_info_list = None
        
        traj = []
        info = {}
        reset_info = source_meta_info_list[demo_idx]["reset_info"] if source_meta_info_list is not None else {}
        env.reset()
        env.reset_world(reset_info)
        # env.render()
        left_ee_pose = h5_file['upper_body_observations']['left_arm_ee_pose'][()]
        right_ee_pose = h5_file['upper_body_observations']['right_arm_ee_pose'][()]
        
        left_arm_joint_position_cmd = h5_file['upper_body_action_dict']['left_arm_joint_position_cmd'][()]
        right_arm_joint_position_cmd = h5_file['upper_body_action_dict']['right_arm_joint_position_cmd'][()]
        left_arm_gripper_position_cmd = h5_file['upper_body_action_dict']['left_arm_gripper_position_cmd'][()]
        right_arm_gripper_position_cmd = h5_file['upper_body_action_dict']['right_arm_gripper_position_cmd'][()]
        episode_length = left_ee_pose.shape[0]
        
        if target_controller_type == 'bimanual_joint_position':
            actions = np.concatenate(
                [left_arm_joint_position_cmd, left_arm_gripper_position_cmd, right_arm_joint_position_cmd, right_arm_gripper_position_cmd],
                axis=-1
            ) # type: ignore
        elif target_controller_type == 'bimanual_ee_pose' or target_controller_type == 'bimanual_relaxed_ik':
            actions = np.concatenate(
                [left_ee_pose, left_arm_gripper_position_cmd, right_ee_pose, right_arm_gripper_position_cmd],
                axis=-1 
            ) # type: ignore
        else:
            raise ValueError(f"Unknown target controller type: {target_controller_type}")
        
        for i in range(episode_length):
            obs, _, _, _, info = env.step(actions[i])
            traj.append(obs)
            if not headless:
                env.render()
        for i in range(5):
            obs, _, _, _, info = env.step(actions[-1])
            traj.append(obs)
            if not headless:
                env.render()
        num_tries += 1
        if info["success"]:
            save_dict_list_to_hdf5(
                traj, save_dir / f"demo_{num_collected}.h5"
            )
            num_collected += 1
            meta_info = dict(reset_info=reset_info, success=info["success"], total_steps=len(traj))
            meta_info_list.append(meta_info)
            save_dict_list_to_json(meta_info_list, save_dir / "meta_info.json")
            # logger.info(f"Collected {num_collected} demos in {num_tries} tries. Success rate: {int(num_collected/num_tries*100)}%")  
            # tqdm.tqdm.write(f"Collected {num_collected} demos in {num_tries} tries. Success rate: {int(num_collected/num_tries*100)}%")
            pbar.update(1)
            pbar.set_postfix_str(f"Collected {num_collected} demos in {num_tries} tries. Success rate: {int(num_collected/num_tries*100)}%")
            if num_collected >= num_demos:
                break

    if num_collected < num_demos:
        logger.warning(f"Collected {num_collected} demos in {num_tries} tries. Success rate: {int(num_collected/num_tries*100)}%")
        logger.warning(f"Failed to collect {num_demos - num_collected} demos.")      

if __name__ == "__main__":
    tyro.cli(main)