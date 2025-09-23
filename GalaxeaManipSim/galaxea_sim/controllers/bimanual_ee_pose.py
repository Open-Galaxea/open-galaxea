import gymnasium as gym
import numpy as np
import torch
from sapien.pysapien import Pose

from galaxea_sim.controllers.utils.kinematics import Kinematics
from galaxea_sim.controllers.bimanual_joint_position import BimanualJointPositionController
from galaxea_sim.robots.bimanual import BimanualRobot

class BimanualEEPoseController(BimanualJointPositionController):
    def __init__(self, robot: BimanualRobot, **kwargs):
        super().__init__(robot)   
        self.articulation = robot.articulation
        self.left_kinematics = Kinematics(
            urdf_path=robot.urdf_path,
            end_link=robot.left_ee_link,
            articulation=robot.articulation,
            controlled_active_joint_indices=self.left_arm_indices
        )
        self.right_kinematics = Kinematics(
            urdf_path=robot.urdf_path,
            end_link=robot.right_ee_link,
            articulation=robot.articulation,
            controlled_active_joint_indices=self.right_arm_indices
        )
        self.left_control_frame_link = robot.left_control_frame_link
        self.right_control_frame_link = robot.right_control_frame_link
        self._left_arm_target_pose_cmd = Pose()
        self._right_arm_target_pose_cmd = Pose()
        
    @property
    def left_arm_cmd(self):
        return np.concatenate([self._left_arm_target_pose_cmd.p, self._left_arm_target_pose_cmd.q])
    
    @property
    def right_arm_cmd(self):
        return np.concatenate([self._right_arm_target_pose_cmd.p, self._right_arm_target_pose_cmd.q])
        
    def get_control_signal(self, action):
        self._left_arm_target_pose_cmd = Pose(action[:3], action[3:7])
        self._right_arm_target_pose_cmd = Pose(action[8:11], action[11:15])
        # pose from frame link frame to world frame
        left_control_frame_pose = self.left_control_frame_link.get_entity_pose()
        right_control_frame_pose = self.right_control_frame_link.get_entity_pose()
        left_arm_target_pose = left_control_frame_pose * self._left_arm_target_pose_cmd
        right_arm_target_pose = right_control_frame_pose * self._right_arm_target_pose_cmd
        left_arm_target_qpos_torch = self.left_kinematics.compute_ik(
            target_pose=left_arm_target_pose,
            q0=torch.tensor(self.articulation.get_qpos(), device=self.left_kinematics.device).unsqueeze(0),
        )
        right_arm_target_qpos_torch = self.right_kinematics.compute_ik(
            target_pose=right_arm_target_pose,
            q0=torch.tensor(self.articulation.get_qpos(), device=self.right_kinematics.device).unsqueeze(0),
        )
        if left_arm_target_qpos_torch is None:
            left_arm_target_qpos_torch = torch.tensor(
                self.articulation.get_qpos(), device=self.left_kinematics.device
            )[self.left_kinematics.active_ancestor_joint_idxs][None]
        if right_arm_target_qpos_torch is None:
            right_arm_target_qpos_torch = torch.tensor(
                self.articulation.get_qpos(), device=self.right_kinematics.device
            )[self.right_kinematics.active_ancestor_joint_idxs][None]
        assert isinstance(left_arm_target_qpos_torch, torch.Tensor)
        assert isinstance(right_arm_target_qpos_torch, torch.Tensor)
        
        left_arm_action = left_arm_target_qpos_torch.cpu().numpy()[0][-self.num_left_arm_joints:]
        left_gripper_action = action[7]
        right_arm_action = right_arm_target_qpos_torch.cpu().numpy()[0][-self.num_left_arm_joints:]
        right_gripper_action = action[15]
        
        joint_position_action = np.concatenate(
            [left_arm_action, [left_gripper_action], right_arm_action, [right_gripper_action]]
        )
        return super().get_control_signal(joint_position_action)
    
    @property
    def action_dim(self):
        return 16
    
    @property
    def action_space(self):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.action_dim,), dtype=np.float32)