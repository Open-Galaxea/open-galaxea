from pathlib import Path

import numpy as np
import time
import torch
import tyro

from robot_interface import GalaxeaInferfaceConfig
from galaxea_real_utils import get_wrapped_env

from experiments.policy_r1_lite import GalaxeaZeroPolicy

INSTRUCTION_PATH = Path(__file__).resolve().parent / "instruction.txt"

def main(interface_config: GalaxeaInferfaceConfig, run_dir: Path, ckpt_id: int, num_action_steps: int = 16, dtype: str = 'fp32'):
    env = get_wrapped_env(interface_config)
    policy = GalaxeaZeroPolicy(
        cfg_file=str(run_dir / "config.yaml"),
        checkpoint_path=str(run_dir / f"model_{ckpt_id}.pt"),
        dtype=dtype,
    )
    input("Press Enter to start the robot...")
    obs = env.get_observations()
    while obs is None:
        time.sleep(0.1)
        obs = env.get_observations()
    last_action = np.concatenate(
        [
            obs["/hdas/feedback_arm_left/position"],
            obs["/hdas/feedback_gripper_left"],
            obs["/hdas/feedback_arm_right/position"],
            obs["/hdas/feedback_gripper_right"],
            np.zeros(12, dtype=np.float32)
        ]
    )
    while not env.is_close():
        if obs is None:
            time.sleep(0.1)
            obs = env.get_observations()
            continue

        instruction = INSTRUCTION_PATH.read_text()

        obs["last_action"] = last_action
        if instruction in ['', 'nothing']:
            obs = None
            continue
        else:
            with torch.inference_mode():
                action = policy.infer(
                    obs=obs,
                    instruction=instruction,
                )
            for i in range(num_action_steps):
                obs = env.step(action[i])
                last_action = action[i]


if __name__ == '__main__':
    tyro.cli(main)
