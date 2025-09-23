from copy import deepcopy, copy

import sapien
import numpy as np

from sapien import Pose

from galaxea_sim.envs.base.bimanual_manipulation import BimanualManipulationEnv
from galaxea_sim.utils.robotwin_utils import create_table, create_box, rand_create_glb
from galaxea_sim.utils.rand_utils import rand_pose
from .robotwin_base import RoboTwinBaseEnv

class DualBottlesPickEasyEnv(RoboTwinBaseEnv):
    @property
    def table_height(self):
        return 0.9
    
    @property
    def tabletop_center_x(self):
        return 0.7
    
    def _setup_red_bottle(self):
        self.red_bottle_xlim = [-0.2, 0.]
        self.red_bottle_ylim = [-0.25, -0.05]
        self.red_bottle_zlim = [0.125]
        self.red_bottle_qpos = [0.707, 0.707, 0, 0]
        self.red_bottle, _ = rand_create_glb(
            scene=self._scene,
            modelname="001_bottles",
            xlim=self.red_bottle_xlim,
            ylim=self.red_bottle_ylim,
            zlim=self.red_bottle_zlim,
            rotate_rand=False,
            qpos=self.red_bottle_qpos,
            scale=(0.132, 0.132, 0.132),
            model_id=13,
            tabeltop_center_in_world=self.tabletop_center_in_world,
            convex=True
        )     
        
    def _setup_green_bottle(self):
        self.green_bottle_xlim = [-0.2, 0.]
        self.green_bottle_ylim = [0.05, 0.25]
        self.green_bottle_zlim = [0.125]
        self.green_bottle_qpos = [0.707, 0.707, 0, 0]
        self.green_bottle, _ = rand_create_glb(
            scene=self._scene,
            modelname="001_bottles",
            xlim=self.green_bottle_xlim,
            ylim=self.green_bottle_ylim,
            zlim=self.green_bottle_zlim,
            rotate_rand=False,
            qpos=self.green_bottle_qpos,
            scale=(0.132, 0.132, 0.132),
            model_id=16,
            tabeltop_center_in_world=self.tabletop_center_in_world,
            convex=True
        )
        self.green_bottle_initial_pose_on_table = self.green_bottle.get_pose()
        self.red_bottle_initial_pose_on_table = self.red_bottle.get_pose()
        
    def reset_world(self, reset_info=None):
        if hasattr(self, "red_bottle"):
            self._scene.remove_actor(self.red_bottle)
        if hasattr(self, "green_bottle"):
            self._scene.remove_actor(self.green_bottle)
        self._setup_red_bottle()
        self._setup_green_bottle()
        if reset_info is not None:
            self.red_bottle.set_pose(sapien.Pose(reset_info["red_bottle_initial_pos_on_table"], self.red_bottle_qpos))
            self.green_bottle.set_pose(sapien.Pose(reset_info["green_bottle_initial_pos_on_table"], self.green_bottle_qpos))
            self.green_bottle_initial_pose_on_table = self.green_bottle.get_pose()
            self.red_bottle_initial_pose_on_table = self.red_bottle.get_pose()
            
    @property
    def left_target_pose(self):
        if self.robot_name == "r1_pro":
            p = [0.5, 0.2, 1.2]
        elif self.robot_name == "r1_lite":
            p = [0.3, 0.13, 1.2]
        else:
            p = [0.6, 0.13, 1.2]
        (target_pose := Pose(p=p)).set_rpy(
            rpy=np.array([np.pi, np.pi, np.pi / 2], dtype=np.float32) + self.robot.left_ee_rpy_offset
        )
        return target_pose
    
    @property
    def right_target_pose(self):
        if self.robot_name == "r1_pro":
            p = [0.5, -0.2, 1.2]
        elif self.robot_name == "r1_lite":
            p = [0.3, -0.13, 1.2]
        else:
            p = [0.6, -0.13, 1.2]
        (target_pose := Pose(p=p)).set_rpy(
            rpy=np.array([0, 0, np.pi / 2], dtype=np.float32)+ self.robot.right_ee_rpy_offset
        )
        return target_pose
        
    def solution(self):
        (left_grasp_ori := Pose()).set_rpy(rpy=(np.array([np.pi, -np.pi, 2.26], dtype=np.float32) + self.robot.left_ee_rpy_offset))
        (right_grasp_ori := Pose()).set_rpy(rpy=(np.array([0, 0, 0.88], dtype=np.float32) + self.robot.right_ee_rpy_offset))
        left_pose0 = Pose(p=self.green_bottle.get_pose().p+[-0.1096, 0.1164, 0.], q=left_grasp_ori.q)
        right_pose0 = Pose(p=self.red_bottle.get_pose().p+[-0.1096, -0.1164, 0.], q=right_grasp_ori.q)
        left_pose1 = Pose(p=self.green_bottle.get_pose().p+[-0.0196, 0.0164, 0.], q=left_grasp_ori.q)
        right_pose1 = Pose(p=self.red_bottle.get_pose().p+[-0.0196, -0.0164, 0.], q=right_grasp_ori.q)
        substeps = [
            ("move_to_pose", {"left_pose": deepcopy(left_pose0), "right_pose": deepcopy(right_pose0)}),
            ("open_gripper", {"action_mode": "both"}),
            ("move_to_pose", {"left_pose": deepcopy(left_pose1), "right_pose": deepcopy(right_pose1)}),
            ("close_gripper", {"action_mode": "both"}),
            ("move_to_pose", {"left_pose": deepcopy(self.left_target_pose), "right_pose": deepcopy(self.right_target_pose)}),
            ("move_to_pose", {"left_pose": deepcopy(self.left_target_pose), "right_pose": deepcopy(self.right_target_pose)}),
        ]
        for substep in substeps:
            yield substep
    
    def _get_info(self):
        left_distance = np.linalg.norm(self.green_bottle.get_pose().p - self.left_target_pose.p)
        right_distance = np.linalg.norm(self.red_bottle.get_pose().p - self.right_target_pose.p)
        left_height = self.green_bottle.get_pose().p[2].item()
        right_height = self.red_bottle.get_pose().p[2].item()
        left_success = left_distance < 0.15 and left_height >= self.left_target_pose.p[2].item() - 0.07
        right_success = right_distance < 0.15 and right_height >= self.right_target_pose.p[2].item() - 0.07
        
        return dict(
            left_success=left_success,
            right_success=right_success,
            success=left_success and right_success,
            left_distance=left_distance,
            right_distance=right_distance,
            left_height=left_height,
            right_height=right_height,
        )
    
    def _get_reset_info(self):
        return dict(
            red_bottle_initial_pos_on_table=copy(self.red_bottle_initial_pose_on_table.p),
            green_bottle_initial_pos_on_table=copy(self.green_bottle_initial_pose_on_table.p),
        )
        
    def _check_termination(self) -> bool:
        return bool(self._get_info()["success"])
    
    @property
    def language_instruction(self):
        return "pick up the two bottles simultaneously"
    
    def get_object_dict(self):
        return dict(
            red_bottle=np.concatenate([self.red_bottle.get_pose().p, self.red_bottle.get_pose().q]),
            green_bottle=np.concatenate([self.green_bottle.get_pose().p, self.green_bottle.get_pose().q]),
        )
        
    def _get_reward(self):
        return 1.0 if self._get_info()["success"] else 0.0