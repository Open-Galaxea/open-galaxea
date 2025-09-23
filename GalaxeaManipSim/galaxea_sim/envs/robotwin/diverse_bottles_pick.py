from copy import deepcopy, copy

import sapien
import numpy as np

from galaxea_sim.utils.robotwin_utils import rand_create_glb, get_grasp_pose_w_given_direction
from .dual_bottles_pick_easy import DualBottlesPickEasyEnv

class DiverseBottlesPickEnv(DualBottlesPickEasyEnv):
    @property
    def id_list(self):
        return list(range(11))[::2]
        
    def _setup_bottle1(self, bottle_id=None):
        if bottle_id is None:
            bottle_id = np.random.choice(self.id_list)
        self._bottle1_id = int(bottle_id)
        self.bottle1, self.bottle1_data = rand_create_glb(
            self._scene,
            xlim=[-0.2, 0],
            ylim=[0.05, 0.25],
            zlim=[0.125],
            rotate_rand=False,
            qpos=(sapien.Pose(q=[0.92387, 0, 0, 0.38268]) * sapien.Pose(q=[0.707, 0.707, 0, 0])).q,
            modelname="001_bottles",
            model_id=self._bottle1_id,
            tabeltop_center_in_world=self.tabletop_center_in_world,
            model_z_val=True,
            convex=True,
        )
        
    def _setup_bottle2(self, bottle_id=None):
        if bottle_id is None:
            bottle_id = np.random.choice(self.id_list)
        self._bottle2_id = int(bottle_id)
        self.bottle2, self.bottle2_data = rand_create_glb(
            self._scene,
            xlim=[-0.2, 0],
            ylim=[-0.25, -0.05],
            zlim=[0.125],
            rotate_rand=False,
            qpos=(sapien.Pose(q=[0.92387, 0, 0, 0.38268]) * sapien.Pose(q=[0.707, 0.707, 0, 0])).q,
            modelname="001_bottles",
            model_id=self._bottle2_id,
            tabeltop_center_in_world=self.tabletop_center_in_world,
            model_z_val=True,
            convex=True
        )

    def reset_world(self, reset_info=None):
        if hasattr(self, "bottle1"):
            self._scene.remove_actor(self.bottle1)
        if hasattr(self, "bottle2"):
            self._scene.remove_actor(self.bottle2)
        self._setup_bottle1()
        self._setup_bottle2()
        if reset_info is not None:
            self.bottle1.set_pose(sapien.Pose(p=reset_info["init_bottle1_pose"][:3], q=reset_info["init_bottle1_pose"][3:]))
            self.bottle2.set_pose(sapien.Pose(p=reset_info["init_bottle2_pose"][:3], q=reset_info["init_bottle2_pose"][3:]))

    def solution(self):
        left_pose0 = get_grasp_pose_w_given_direction(self.bottle1, self.bottle1_data, grasp_qpos=[-0.906, 0, 0, 0.424], pre_dis=0.1)
        right_pose0 = get_grasp_pose_w_given_direction(self.bottle2, self.bottle2_data, grasp_qpos=[0.906, 0, 0, 0.424], pre_dis=0.1)
        left_pose1 = get_grasp_pose_w_given_direction(self.bottle1, self.bottle1_data, grasp_qpos=[-0.906, 0, 0, 0.424], pre_dis=-0.075)
        right_pose1 = get_grasp_pose_w_given_direction(self.bottle2, self.bottle2_data, grasp_qpos=[0.906, 0, 0, 0.424], pre_dis=-0.075)
        yield ("move_to_pose", {"left_pose": deepcopy(left_pose0), "right_pose": deepcopy(right_pose0)})
        yield ("open_gripper", {"action_mode": "both"})
        yield ("open_gripper", {"action_mode": "both"})
        yield ("move_to_pose", {"left_pose": deepcopy(left_pose1), "right_pose": deepcopy(right_pose1)})
        yield ("move_to_pose", {"left_pose": deepcopy(left_pose1), "right_pose": deepcopy(right_pose1)})
        yield ("close_gripper", {"action_mode": "both"})
        yield ("move_to_pose", {"left_pose": deepcopy(self.left_target_pose), "right_pose": deepcopy(self.right_target_pose)})
    
    def _get_info(self):
        left_distance = np.linalg.norm(self.bottle1.get_pose().p - self.left_target_pose.p)
        right_distance = np.linalg.norm(self.bottle2.get_pose().p - self.right_target_pose.p)
        left_height = self.bottle1.get_pose().p[2].item()
        right_height = self.bottle2.get_pose().p[2].item()
        left_success = left_distance < 0.1 and left_height >= 1.10
        right_success = right_distance < 0.1 and right_height >= 1.10
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
            init_bottle1_pose=np.concatenate([self.bottle1.get_pose().p, self.bottle1.get_pose().q]),
            init_bottle2_pose=np.concatenate([self.bottle2.get_pose().p, self.bottle2.get_pose().q]),
            bottle1_id=self._bottle1_id,
            bottle2_id=self._bottle2_id,
        )
        
    def _check_termination(self) -> bool:
        return bool(self._get_info()["success"])
    
    def get_object_dict(self):
        return dict(
            bottle1=np.concatenate([self.bottle1.get_pose().p, self.bottle1.get_pose().q]),
            bottle2=np.concatenate([self.bottle2.get_pose().p, self.bottle2.get_pose().q]),
            bottle1_id=np.array([id == self._bottle1_id for id in self.id_list], dtype=np.float32),
            bottle2_id=np.array([id == self._bottle2_id for id in self.id_list], dtype=np.float32),
        )
        
    def _get_reward(self):
        return 1.0 if self._get_info()["success"] else 0.0