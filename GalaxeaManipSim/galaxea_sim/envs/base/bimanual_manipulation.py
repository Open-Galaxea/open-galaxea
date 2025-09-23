from typing import Literal

import numpy as np
import sapien
from loguru import logger
from sapien.utils.viewer import Viewer

from galaxea_sim.controllers import *
from galaxea_sim.envs.base.base import SapienEnv
from galaxea_sim.utils.gym_utils import get_observation_space_from_example
from galaxea_sim.robots.bimanual import BimanualRobot

class BimanualManipulationEnv(SapienEnv):
    def __init__(
        self,
        robot_class: type[BimanualRobot],
        robot_kwargs: dict = {},
        controller_type: str = "bimanual_joint_position",
        control_freq: int = 15,
        timestep: float = 0.01,
        headless: bool = True,
        obs_mode: Literal["state", "image"] = "image",  
        ray_tracing: bool = False,
    ):
        self.eval_mode = False
        self.robot_name = robot_class.name
        super().__init__(control_freq, timestep, headless, ray_tracing)
        self.robot: BimanualRobot = robot_class(self._scene, **robot_kwargs)
        self._init_controller(controller_type)
        self._init_buffers()
        self.action_space = self.controller.action_space
        self.obs_mode = obs_mode
        self.observation_space = get_observation_space_from_example(self._get_obs())
        
    def eval(self):
        self.eval_mode = True
        
    def _build_world(self):
        self._scene.add_ground(0)
        
    @property
    def left_arm_joint_indices(self):
        return self.robot.left_arm_joint_indices
    
    @property
    def right_arm_joint_indices(self):
        return self.robot.right_arm_joint_indices
    
    @property
    def left_gripper_joint_indices(self):
        return self.robot.left_gripper_joint_indices
    
    @property
    def right_gripper_joint_indices(self):
        return self.robot.right_gripper_joint_indices
    
    @property
    def torso_joint_indices(self):
        return self.robot.torso_joint_indices
    
    @property
    def active_joints(self):
        return self.robot.active_joints
    
    @property
    def num_dofs(self):
        return self.robot.num_dofs
    
    @property
    def init_qpos(self):
        return self.robot.init_qpos
    
    @property
    def active_joint_names(self):
        return self.robot.active_joint_names
    
    @property
    def left_ee_link_name(self):
        return self.robot.left_ee_link_name
    
    @property
    def right_ee_link_name(self):
        return self.robot.right_ee_link_name

    def _init_buffers(self):
        self.last_gripper_cmd = [0, 0]
        self.left_arm_joint_position_cmd = np.zeros(len(self.left_arm_joint_indices))
        self.right_arm_joint_position_cmd = np.zeros(len(self.right_arm_joint_indices))
        self.left_arm_gripper_position_cmd = 0.
        self.right_arm_gripper_position_cmd = 0.
            
    def _init_controller(self, controller_type):
        self.controller_type = controller_type
        if controller_type == "bimanual_joint_position":
            self.controller = BimanualJointPositionController(self.robot)
        elif controller_type == "bimanual_ee_pose":
            self.controller = BimanualEEPoseController(self.robot)
        elif controller_type == "bimanual_relaxed_ik":
            self.controller = BimanualRelaxedIKController(self.robot)
        else:
            raise ValueError(f"Invalid controller type. Got: {controller_type}")    
        
    def _setup_viewer(self):
        self.engine = sapien.Engine()
        self.renderer = sapien.SapienRenderer()
        self.engine.set_renderer(self.renderer)
        self.viewer = Viewer(self.renderer)
        self.viewer.plugins[5].show_camera_linesets = False
        self.viewer.set_scene(self._scene)
        self.viewer.set_camera_xyz(x=1.2, y=0.25, z=1.5,)
        self.viewer.set_camera_rpy(r=0, p=-0.4, y=2.7)

    def step(self, action):
        left_arm_action, left_gripper_action, right_arm_action, right_gripper_action = self.controller.get_control_signal(action)
        self.left_arm_joint_position_cmd = left_arm_action
        self.right_arm_joint_position_cmd = right_arm_action
        self.left_arm_gripper_position_cmd = left_gripper_action
        self.right_arm_gripper_position_cmd = right_gripper_action
        for i in range(self.num_dofs):
            self.active_joints[i].set_drive_target(self.init_qpos[i])
        for i, joint_index in enumerate(self.left_arm_joint_indices):
            self.active_joints[joint_index].set_drive_target(left_arm_action[i])
        for i, joint_index in enumerate(self.right_arm_joint_indices):
            self.active_joints[joint_index].set_drive_target(right_arm_action[i])
        for joint_index, joint_sign in zip(self.left_gripper_joint_indices, self.robot.gripper_finger_sign):
            self.active_joints[joint_index].set_drive_target(left_gripper_action * joint_sign)
        for joint_index, joint_sign in zip(self.right_gripper_joint_indices, self.robot.gripper_finger_sign):
            self.active_joints[joint_index].set_drive_target(right_gripper_action * joint_sign)
        self.last_gripper_cmd = [left_gripper_action, right_gripper_action]
        for i in range(self.decimation):
            qf = self.robot.compute_passive_force(
                gravity=True, coriolis_and_centrifugal=True
            )
            self.robot.set_qf(qf)
            self._scene.step()

        obs = self._get_obs()
        reward = self._get_reward()
        terminated = self._check_termination()
        truncated = self._check_truncation()
        info = self._get_info()
        
        self._scene.update_render()

        return obs, reward, terminated, truncated, info
    
    def _get_info(self):
        return {}
    
    def _get_reset_info(self):
        return {}
    
    def _check_truncation(self):
        return False
    
    def _check_termination(self) -> bool:
        return False

    def reset(self, *, seed=None, options=None):
        self.robot.set_qpos(self.init_qpos)
        self.controller.reset()
        self.reset_world()
        self._scene.update_render()
        return self._get_obs(), self._get_reset_info()

    def reset_world(self, reset_info=None):
        pass

    def _get_obs(self):
        qpos = self.robot.get_qpos()
        qvel = self.robot.get_qvel()
        left_arm_ee_pose = self.robot.left_ee_pose_wrt_control_frame
        right_arm_ee_pose = self.robot.right_ee_pose_wrt_control_frame
        left_arm_joint_position = qpos[self.left_arm_joint_indices]
        right_arm_joint_position = qpos[self.right_arm_joint_indices]
        left_arm_joint_velocity = qvel[self.left_arm_joint_indices]
        right_arm_joint_velocity = qvel[self.right_arm_joint_indices]
        left_arm_gripper_position = qpos[self.left_gripper_joint_indices][0:1]
        right_arm_gripper_position = qpos[self.right_gripper_joint_indices][0:1]
        torso_joint_position = qpos[self.torso_joint_indices]
        
        upper_body_observations = dict(
            left_arm_joint_position=left_arm_joint_position,
            right_arm_joint_position=right_arm_joint_position,
            left_arm_gripper_position=left_arm_gripper_position,
            right_arm_gripper_position=right_arm_gripper_position,          
            left_arm_joint_velocity=left_arm_joint_velocity,
            right_arm_joint_velocity=right_arm_joint_velocity,  
            left_arm_ee_pose=np.concatenate([left_arm_ee_pose.p, left_arm_ee_pose.q]),
            right_arm_ee_pose=np.concatenate([right_arm_ee_pose.p, right_arm_ee_pose.q]),
        )
        upper_body_action_dict = dict(
            left_arm_joint_position_cmd=self.left_arm_joint_position_cmd,
            right_arm_joint_position_cmd=self.right_arm_joint_position_cmd,
            left_arm_gripper_position_cmd=np.array([self.left_arm_gripper_position_cmd]),
            right_arm_gripper_position_cmd=np.array([self.right_arm_gripper_position_cmd]),
        )
        if self.controller_type == "bimanual_ee_pose" or self.controller_type == "bimanual_relaxed_ik":
            upper_body_action_dict.update(
                left_arm_ee_pose_cmd=self.controller.left_arm_cmd,
                right_arm_ee_pose_cmd=self.controller.right_arm_cmd,
            )
        lower_body_observations = dict(
            chassis_joint_position=np.zeros(3),
            torso_joint_position=torso_joint_position,
        )
        lower_body_action_dict = dict(
            chassis_target_speed_cmd=np.zeros(3),
        )
        if self.obs_mode == "image":
            image_dict = self.get_image_dict()
            for key, value in image_dict.items():
                upper_body_observations[key] = value
        object_dict = self.get_object_dict()       
        
        return dict(
            upper_body_observations=upper_body_observations,
            upper_body_action_dict=upper_body_action_dict,
            lower_body_observations=lower_body_observations,
            lower_body_action_dict=lower_body_action_dict,
            language_instruction=self.language_instruction,
            object_dict=object_dict,
        )
        
    def get_object_dict(self):
        return {}
        
    @property
    def language_instruction(self) -> str:
        return ""

    def _get_reward(self):
        return 0.
    
    def solution(self):
        substeps = []
        for substep in substeps:
            yield substep
    
    @property
    def cameras(self):
        return self.robot.cameras