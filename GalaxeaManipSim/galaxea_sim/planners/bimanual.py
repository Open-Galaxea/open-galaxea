import os

import mplib
import numpy as np
from mplib.pymp import Pose
from mplib.sapien_utils.conversion import SapienPlanner, SapienPlanningWorld
from loguru import logger

from galaxea_sim import ASSETS_DIR
from galaxea_sim.planners.base import BasePlanner
from galaxea_sim.utils.mplib_utils import get_planner_mask, get_sim2mplib_mapping, planner_attach_obj, planner_detach_obj, disable_table_collision

GRIPPER_STEPS = 10
class BimanualPlanner(BasePlanner):
    def __init__(self, urdf_path, srdf_path, left_arm_move_group, right_arm_move_group, active_joint_names, control_freq):
        self.left_arm_planner = mplib.planner.Planner(
            urdf=ASSETS_DIR / urdf_path,
            srdf=ASSETS_DIR / srdf_path if srdf_path else None,
            move_group=left_arm_move_group,
        )
        self.right_arm_planner = mplib.planner.Planner(
            urdf=ASSETS_DIR / urdf_path,
            srdf=ASSETS_DIR / srdf_path if srdf_path else None,
            move_group=right_arm_move_group,
        )
        self.time_step = 1 / control_freq
        num_dofs = len(active_joint_names)
        self.left_arm_planner_mask = get_planner_mask(num_dofs, self.left_arm_planner.move_group_joint_indices, self.right_arm_planner.move_group_joint_indices)
        self.right_arm_planner_mask = get_planner_mask(num_dofs, self.right_arm_planner.move_group_joint_indices, self.left_arm_planner.move_group_joint_indices)
        self.sim2mplib_mapping = get_sim2mplib_mapping(
            active_joint_names,
            self.left_arm_planner.user_joint_names,
        )
        self.action_dim = self.left_arm_action_dim + self.right_arm_action_dim + 2
        logger.debug(f"Left arm move group joint indices: {self.left_arm_planner.move_group_joint_indices}")
        logger.debug(f"Right arm move group joint indices: {self.right_arm_planner.move_group_joint_indices}")
        logger.debug(f"Left arm planner mask: {self.left_arm_planner_mask}")
        logger.debug(f"Right arm planner mask: {self.right_arm_planner_mask}")
        logger.debug(f"action_dim: {self.action_dim}")
        
    @property
    def left_arm_action_dim(self):
        return  np.sum((~self.left_arm_planner_mask).astype(np.int32))
    
    @property
    def right_arm_action_dim(self):
        return np.sum((~self.right_arm_planner_mask).astype(np.int32))
    
    def plan_pose(self, left_pose, right_pose, robot_qpos, verbose=False):
        left_result = None
        right_result = None
        if left_pose is not None:
            left_result = self.left_arm_planner.plan_pose(
                left_pose, robot_qpos, time_step=self.time_step, mask=self.left_arm_planner_mask, verbose=verbose
            )
            if left_result["status"] != "Success":
                if verbose: logger.error(left_result["status"])
                return None
        
        if right_pose is not None:
            right_result = self.right_arm_planner.plan_pose(
                right_pose, robot_qpos, time_step=self.time_step, mask=self.right_arm_planner_mask, verbose=verbose
            )
            if right_result["status"] != "Success":
                if verbose: logger.error(right_result["status"])
                return None
        
        return left_result, right_result
    
    def plan_screw(self, left_pose, right_pose, robot_qpos):
        left_result = self.left_arm_planner.plan_screw(left_pose, robot_qpos, time_step=self.time_step) if left_pose else None
        right_result = self.right_arm_planner.plan_screw(right_pose, robot_qpos, time_step=self.time_step) if right_pose else None
        
        if (left_result and left_result["status"] != "Success") or (right_result and right_result["status"] != "Success"):
            return None
        return left_result, right_result
    
    def move_to_pose(self, left_pose=None, right_pose=None, robot_qpos=None, with_screw=False, verbose=False):
        """API to multiplex between the two planning methods"""
        if isinstance(left_pose, (list, np.ndarray)):
            left_pose = Pose(p=left_pose[:3], q=left_pose[3:])
        if isinstance(right_pose, (list, np.ndarray)):
            right_pose = Pose(p=right_pose[:3], q=right_pose[3:])
        if with_screw:
            return self.plan_screw(left_pose, right_pose, robot_qpos)
        else:
            return self.plan_pose(left_pose, right_pose, robot_qpos, verbose=verbose)
        
    def get_move_trajectory(self, init_pos, left_result=None, right_result=None):
        n_step_left = left_result["position"].shape[0] if left_result else 0
        n_step_right = right_result["position"].shape[0] if right_result else 0
        n_step = max(n_step_left, n_step_right)
        trajectory = np.stack([init_pos] * n_step) if n_step > 0 else np.zeros((1, self.action_dim))
        left_arm_action_dim = self.left_arm_action_dim
        right_arm_action_dim = self.right_arm_action_dim
        if n_step_left > 0:
            trajectory[:n_step_left, :left_arm_action_dim] = left_result["position"][..., -left_arm_action_dim:]
            trajectory[n_step_left:, :left_arm_action_dim] = left_result["position"][-1][-left_arm_action_dim:]
        if n_step_right > 0:
            trajectory[:n_step_right, -right_arm_action_dim-1:-1] = right_result["position"][..., -right_arm_action_dim:]
            trajectory[n_step_right:, -right_arm_action_dim-1:-1] = right_result["position"][-1][-right_arm_action_dim:]
        return trajectory

    def get_gripper_trajectory(self, init_pos, method, kwargs):
        trajectory = np.stack([init_pos] * kwargs.get("steps", GRIPPER_STEPS))
        if "gripper_target_state" in kwargs.keys():
            gripper_target_state = kwargs["gripper_target_state"]
        else:
            gripper_target_state = 0.05 if method == "open_gripper" else 0.0
        gripper_indices = []
        if kwargs["action_mode"] == "left":
            gripper_indices = [self.left_arm_action_dim]
        elif kwargs["action_mode"] == "right":
            gripper_indices = [-1]
        elif kwargs["action_mode"] == "both":
            gripper_indices = [self.left_arm_action_dim, -1]
        trajectory[:GRIPPER_STEPS, gripper_indices] = np.linspace(init_pos[gripper_indices], gripper_target_state, GRIPPER_STEPS)
        trajectory[GRIPPER_STEPS:, gripper_indices] = gripper_target_state
        return trajectory
        
    def solve(self, substep, robot_qpos_in_sim, last_gripper_cmd, verbose=False):
        method, kwargs = substep
        robot_qpos = self.sim2mplib_mapping(robot_qpos_in_sim)
        init_pos = np.zeros(self.action_dim)
        init_pos[:self.left_arm_action_dim] = robot_qpos[~self.left_arm_planner_mask]
        init_pos[self.left_arm_action_dim] = last_gripper_cmd[0]
        init_pos[-self.right_arm_action_dim-1:-1] = robot_qpos[~self.right_arm_planner_mask]
        init_pos[-1] = last_gripper_cmd[1]
        
        if method == "move_to_pose":
            result = self.move_to_pose(**kwargs, robot_qpos=robot_qpos, verbose=verbose)
            if result is None:
                return None
            return self.get_move_trajectory(init_pos, *result)
        elif method in ["open_gripper", "close_gripper"]:
            return self.get_gripper_trajectory(init_pos, method, kwargs)
        else:
            raise NotImplementedError(f"Method {method} not implemented in BimanualPlanner")
        
class SapienBimanualPlanner(BimanualPlanner):
    def __init__(self, scene, robot, left_arm_move_group, right_arm_move_group, active_joint_names, control_freq):
        planning_world = SapienPlanningWorld(scene, [robot])
        self.left_arm_planner = SapienPlanner(planning_world, left_arm_move_group)
        self.right_arm_planner = SapienPlanner(planning_world, right_arm_move_group)
        self.num_dofs = len(active_joint_names)
        num_dofs = len(active_joint_names)
        self.left_arm_planner_mask = get_planner_mask(num_dofs, self.left_arm_planner.move_group_joint_indices, self.right_arm_planner.move_group_joint_indices)
        self.right_arm_planner_mask = get_planner_mask(num_dofs, self.right_arm_planner.move_group_joint_indices, self.left_arm_planner.move_group_joint_indices)
        self.sim2mplib_mapping = get_sim2mplib_mapping(
            active_joint_names,
            self.left_arm_planner.user_joint_names,
        )
        self.action_dim = self.left_arm_action_dim + self.right_arm_action_dim + 2
        logger.debug(f"Left arm move group joint indices: {self.left_arm_planner.move_group_joint_indices}")
        logger.debug(f"Right arm move group joint indices: {self.right_arm_planner.move_group_joint_indices}")
        logger.debug(f"Left arm planner mask: {self.left_arm_planner_mask}")
        logger.debug(f"Right arm planner mask: {self.right_arm_planner_mask}")
        logger.debug(f"action_dim: {self.action_dim}")
        self.time_step = 1 / control_freq

        disable_table_collision(self.left_arm_planner)
        disable_table_collision(self.right_arm_planner)
        
    def get_gripper_trajectory(self, init_pos, method, kwargs):
        trajectory = np.stack([init_pos] * GRIPPER_STEPS)
        gripper_target_state = 0.04 if method == "open_gripper" else 0.0
        gripper_indices = []
        use_left, use_right = False, False
        if kwargs["action_mode"] == "left":
            gripper_indices = [self.left_arm_action_dim]
            use_left = True
        elif kwargs["action_mode"] == "right":
            gripper_indices = [-1]
            use_right = True
        elif kwargs["action_mode"] == "both":
            gripper_indices = [self.left_arm_action_dim, -1]
            use_left, use_right = True, True
        if use_left:
            if method == "close_gripper":
                planner_attach_obj(self.left_arm_planner, kwargs["cube_actor"], touch_links=["left_gripper_finger_link1", "left_gripper_finger_link2"])
            elif "cube_actor" in kwargs:
                planner_detach_obj(self.left_arm_planner, kwargs["cube_actor"])
        if use_right:
            if method == "close_gripper":
                planner_attach_obj(self.right_arm_planner, kwargs["cube_actor"], touch_links=["right_gripper_finger_link1", "right_gripper_finger_link2"])
            elif "cube_actor" in kwargs:
                planner_detach_obj(self.right_arm_planner, kwargs["cube_actor"])
        trajectory[:, gripper_indices] = gripper_target_state
        return trajectory