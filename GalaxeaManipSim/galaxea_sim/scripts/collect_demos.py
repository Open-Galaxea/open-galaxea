import datetime
import uuid

from typing import Literal, Optional

import gymnasium as gym
import tyro
from loguru import logger
from pathlib import Path

import galaxea_sim.envs
from galaxea_sim.envs.base.bimanual_manipulation import BimanualManipulationEnv
from galaxea_sim.planners.bimanual import BimanualPlanner
from galaxea_sim.utils.data_utils import save_dict_list_to_hdf5, save_dict_list_to_json 

def main(
    env_name: str, 
    num_demos: int = 100, 
    dataset_dir: str = 'datasets', 
    control_freq: int = 15, 
    headless: bool = True, 
    obs_mode: Literal['state', 'image'] = 'state', 
    tag: Optional[str] = 'collected', 
    ray_tracing: bool = False
):
    env = gym.make(
        env_name,
        control_freq=control_freq,
        headless=headless,
        obs_mode=obs_mode,
        ray_tracing=ray_tracing,
    )
    assert isinstance(env.unwrapped, BimanualManipulationEnv)
    planner = BimanualPlanner(
        urdf_path=f"{env.unwrapped.robot.name}/robot.urdf",
        srdf_path=None,
        left_arm_move_group=env.unwrapped.left_ee_link_name,
        right_arm_move_group=env.unwrapped.right_ee_link_name,
        active_joint_names=env.unwrapped.active_joint_names,
        control_freq=env.unwrapped.control_freq,
    )

    save_dir = Path(dataset_dir) / env_name / tag
    num_collected = 0
    num_tries = 0
    meta_info_list = []
    while num_collected < num_demos:
        num_steps = 0
        traj = []
        info = {}
        _, rest_info = env.reset()
        if not headless: env.render()
        for substep in env.unwrapped.solution():
            actions = planner.solve(
                substep, env.unwrapped.robot.get_qpos(), env.unwrapped.last_gripper_cmd, 
                verbose=False
            )
            if actions is not None:
                for action in actions:
                    num_steps += 1
                    obs, _, _, _, info = env.step(action)
                    traj.append(obs)
                    if not headless: env.render()
        num_tries += 1
        if info["success"]:
            save_dict_list_to_hdf5(
                traj, save_dir / f"demo_{num_collected}.h5"
            )
            num_collected += 1
            meta_info = dict(reset_info=rest_info, success=info["success"], total_steps=num_steps, num_collected=num_collected, num_tries=num_tries)
            meta_info_list.append(meta_info)
            save_dict_list_to_json(meta_info_list, save_dir / "meta_info.json")
            logger.info(f"Collected {num_collected} demos in {num_tries} tries. Success rate: {int(num_collected/num_tries*100)}%")        

if __name__ == "__main__":
    tyro.cli(main)