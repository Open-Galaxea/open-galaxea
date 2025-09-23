import gymnasium as gym
import numpy as np
import torch
from sapien.pysapien import Pose

from galaxea_sim import ASSETS_DIR
from galaxea_sim.controllers.bimanual_joint_position import BimanualJointPositionController
from galaxea_sim.robots.bimanual import BimanualRobot
from galaxea_sim.controllers.utils.relaxed_ik_solver import RelaxedIKSolver

class BimanualRelaxedIKController(BimanualJointPositionController):
    def __init__(self, robot: BimanualRobot, **kwargs):
        super().__init__(robot)   
        self.init_qpos = robot.init_qpos
        self.articulation = robot.articulation
        assert robot.left_relaxed_ik_setting_path is not None and robot.right_relaxed_ik_setting_path is not None
        self.ee_tracker = RelaxedIKSolver(
            left_setting_file_path=(ASSETS_DIR / robot.left_relaxed_ik_setting_path).as_posix(),
            right_setting_file_path=(ASSETS_DIR / robot.right_relaxed_ik_setting_path).as_posix()
        )
        self._left_arm_target_pose_cmd = Pose()
        self._right_arm_target_pose_cmd = Pose()
        
    @property
    def left_arm_cmd(self):
        return np.concatenate([self._left_arm_target_pose_cmd.p, self._left_arm_target_pose_cmd.q])
    
    @property
    def right_arm_cmd(self):
        return np.concatenate([self._right_arm_target_pose_cmd.p, self._right_arm_target_pose_cmd.q])
        
    @property
    def tolerance(self):
        return [0.001] * 6
        
    def get_control_signal(self, action):
        left_pos,  left_quat  = action[0:3],  action[3:7]   # wxyz
        left_grip             = action[7]
        right_pos, right_quat = action[8:11], action[11:15]
        right_grip            = action[15]

        self._left_arm_target_pose_cmd.set_p(left_pos)
        self._left_arm_target_pose_cmd.set_q(left_quat)
        self._right_arm_target_pose_cmd.set_p(right_pos)
        self._right_arm_target_pose_cmd.set_q(right_quat)

        left_joints  = np.asarray(self.ee_tracker.solve_position_left(left_pos,  left_quat))
        right_joints = np.asarray(self.ee_tracker.solve_position_right(right_pos, right_quat))

        joint_position_action = np.concatenate([
            left_joints,        [left_grip],
            right_joints,       [right_grip]
        ])
        return super().get_control_signal(joint_position_action)
    
    @property
    def action_dim(self):
        return 16
    
    @property
    def action_space(self):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.action_dim,), dtype=np.float32)
    
    def reset(self):
        self._left_arm_target_pose_cmd = Pose()
        self._right_arm_target_pose_cmd = Pose()