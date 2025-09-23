from pathlib import Path
import tyro
import numpy as np
import tqdm

from robot_interface import GalaxeaInferfaceConfig
from galaxea_real_utils import get_wrapped_env

INIT_LEFT_POSITION = [-0.01680851, -0.00468085,  0.02680851,  0.08446808,  0.06914894,  0.15425532]
INIT_RIGHT_POSITION = [-0.09851064, -0.00468085,  0.00851064,  0.03829787, -0.10914893,  0.02297872]

def main(interface_config: GalaxeaInferfaceConfig, replay_freq: int = 100, replay_traj_path: Path = Path(__file__).parent / 'tmp' / 'traj_0.npy'):
    env = get_wrapped_env(interface_config)
    replay_traj = np.load(replay_traj_path, allow_pickle=True)
    replay_length = replay_traj.shape[0]
    obs = env.get_observations()
    cur_left_position = obs['/hdas/feedback_arm_left/position']
    cur_right_position = obs['/hdas/feedback_arm_right/position']
    target_left_position = replay_traj[0]['action'][:6]
    target_right_position = replay_traj[0]['action'][7:13]
    left_offset = target_left_position - cur_left_position
    right_offset = target_right_position - cur_right_position
    for i in range(replay_freq):
       action = np.zeros_like(replay_traj[0]['action'])
       action[:6] = cur_left_position + left_offset * i / replay_freq
       action[7:13] =  cur_right_position + right_offset * i / replay_freq
       env.step(action)

    for i in range(replay_length):
        action = replay_traj[i]['action']
        env.step(action)
        

if __name__ == '__main__':
    tyro.cli(main)