from copy import deepcopy, copy

import sapien
import numpy as np

from galaxea_sim.envs.base.bimanual_manipulation import BimanualManipulationEnv
from galaxea_sim.utils.robotwin_utils import create_table, create_box, rand_create_glb, get_grasp_pose_w_labeled_direction
from galaxea_sim.utils.rand_utils import rand_pose
from .robotwin_base import RoboTwinBaseEnv

class BottleAdjustEnv(RoboTwinBaseEnv):
    def _setup_bottle(self):
        self.qpose_tag = np.random.randint(0, 2)
        qposes = [[0, 0, 0, -1], [1, 0, 0, 0]]
        self.bottle_xlim = [-0.2, -0.1]
        self.bottle_ylim = [0, 0]
        self.bottle_zlim = [0.04]
        self.bottle_qpos = qposes[self.qpose_tag]
        self.bottle, self.bottle_data = rand_create_glb(
            scene=self._scene,
            modelname="001_bottles",
            xlim=self.bottle_xlim,
            ylim=self.bottle_ylim,
            zlim=self.bottle_zlim,
            rotate_rand=False,
            qpos=self.bottle_qpos,
            model_id=13,
            tabeltop_center_in_world=self.tabletop_center_in_world,
            convex=True
        )     

    def reset_world(self, reset_info=None):
        if hasattr(self, "bottle"):
            self._scene.remove_actor(self.bottle)
        self._setup_bottle()
        if reset_info is not None:
            self.bottle.set_pose(sapien.Pose(p=reset_info["init_bottle_pose"][:3], q=reset_info["init_bottle_pose"][3:]))
        
    def solution(self):
        grasp_pose = get_grasp_pose_w_labeled_direction(
            self.bottle, self.bottle_data, grasp_matrix=np.array([[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,0,1]]), pre_dis=-0.1
        )
        pre_pose = get_grasp_pose_w_labeled_direction(
            self.bottle, self.bottle_data, grasp_matrix=np.array([[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,0,1]]), pre_dis=0.08
        )

        if self.qpose_tag == 1:
            target_pose = [0.7, -0.25, self.table_height + 0.18, 1, 0, 0, 0]
            arm = "right"
        else:
            target_pose = [0.7, 0.25, self.table_height + 0.18, 1, 0, 0, 0]
            arm = "left"
            
        yield ("move_to_pose", {f"{arm}_pose": pre_pose})
        yield ("open_gripper", {"action_mode": arm})
        yield ("move_to_pose", {f"{arm}_pose": grasp_pose})
        yield ("close_gripper", {"action_mode": arm})
        yield ("move_to_pose", {f"{arm}_pose": pre_pose})
        yield ("move_to_pose", {f"{arm}_pose": deepcopy(target_pose)})
        target_pose[2] -= 0.05
        yield ("move_to_pose", {f"{arm}_pose": deepcopy(target_pose)})
        yield ("open_gripper", {"action_mode": arm})
        target_pose[0] -= 0.2
        yield ("move_to_pose", {f"{arm}_pose": deepcopy(target_pose)})
        yield ("move_to_pose", {f"{arm}_pose": self.robot.right_init_ee_pose if arm == "right" else self.robot.left_init_ee_pose})
            
    def _get_info(self):
        target_height = 0.83 - 0.74 + self.table_height
        success = self.bottle.get_pose().p[2] > target_height
        return dict(
            success=success
        )
    
    def _get_reset_info(self):
        return dict(
            init_bottle_pose=np.concatenate([self.bottle.get_pose().p, self.bottle.get_pose().q])
        )
        
    def _check_termination(self) -> bool:
        return bool(self._get_info()["success"])
    
    @property
    def language_instruction(self):
        return "adjust the bottle pose to make it stand on the table" 
    
    def get_object_dict(self):
        return dict(
            bottle=np.concatenate([self.bottle.get_pose().p, self.bottle.get_pose().q]),
        )
        
    def _get_reward(self):
        return 1.0 if self._get_info()["success"] else 0.0