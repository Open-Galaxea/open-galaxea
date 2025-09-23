from copy import deepcopy

import sapien
import numpy as np

from galaxea_sim.utils.robotwin_utils import rand_create_glb, get_grasp_pose_w_given_direction, get_grasp_pose_w_labeled_direction, create_obj, create_glb
from galaxea_sim.utils.rand_utils import rand_pose
from .robotwin_base import RoboTwinBaseEnv

class BottleCupEnv(RoboTwinBaseEnv):        
    def _setup_bottle1(self):
        tag = np.random.randint(0, 2)
        qpose = [[0.707, 0, 0, -0.707], [0.707, 0.707, 0, 0]]
        xlim = [[-0.2, 0.05], [0.03, 0.023]]
        zlim = [[0.045], [0.125]]
        rotate_lim = [(0, 0, 1.4), (0, 0, 0)]
        self.bottle1, self.bottle1_data = rand_create_glb(
            self._scene,
            xlim=xlim[tag],
            ylim=[0.15, 0.3],
            zlim=zlim[tag],
            rotate_rand=True,
            qpos=qpose[tag],
            rotate_lim=rotate_lim[tag],
            modelname="001_bottles",
            model_id=13,
            tabeltop_center_in_world=self.tabletop_center_in_world,
            model_z_val=True,
            convex=True,
        )
        
    def _setup_bottle2(self):
        tag = np.random.randint(0, 2)
        qpose = [[0.707, 0, 0, -0.707], [0.707, 0.707, 0, 0]]
        xlim = [[-0.2, 0.05], [0.03, 0.023]]
        zlim = [[0.045], [0.125]]
        rotate_lim = [(0, 0, 1.4), (0, 0, 0)]
        self.bottle2, self.bottle2_data = rand_create_glb(
            self._scene,
            xlim=xlim[tag],
            ylim=[-0.3, -0.15],
            zlim=zlim[tag],
            rotate_rand=True,
            qpos=qpose[tag],
            rotate_lim=rotate_lim[tag],
            modelname="001_bottles",
            model_id=16,
            tabeltop_center_in_world=self.tabletop_center_in_world,
            model_z_val=True,
            convex=True,
        )
        
    def _setup_coaster(self):
        coaster_pose = rand_pose(
            xlim=[-0.2, 0],
            ylim=[-0.05, 0.05],
            zlim=[0.02],
            rotate_rand=False,
            qpos=[0.707, 0.707, 0, 0],
        )
        self.coaster, _ = create_obj(
            self._scene,
            pose=coaster_pose,
            modelname="019_coaster",
            convex=True,
            tabeltop_center_in_world=self.tabletop_center_in_world,
        )
        
    def _setup_cup(self):
        tag = np.random.randint(0, 2)
        cup_pose = None
        while cup_pose is None or np.linalg.norm(cup_pose.p - self.coaster.get_pose().p) < 0.1 \
                or np.linalg.norm(cup_pose.p - self.bottle1.get_pose().p + self.tabletop_center_in_world) < 0.1 \
                or np.linalg.norm(cup_pose.p - self.bottle2.get_pose().p + self.tabletop_center_in_world) < 0.1:
            cup_pose = rand_pose(
                xlim=[-0.2, 0.05],
                ylim=[-0.3, -0.15] if tag == 0 else [0.15, 0.3],
                zlim=[0.06],
                rotate_rand=False,
                qpos=[0.707, 0.707, 0, 0],
            )
        self.cup, self.cup_data = create_glb(
            self._scene,
            pose=cup_pose,
            modelname="022_cup",
            convex=True,
            tabeltop_center_in_world=self.tabletop_center_in_world,
        )

    def reset_world(self, reset_info=None):
        if hasattr(self, "bottle1"):
            self._scene.remove_actor(self.bottle1)
        if hasattr(self, "bottle2"):
            self._scene.remove_actor(self.bottle2)
        if hasattr(self, "coaster"):
            self._scene.remove_actor(self.coaster)
        if hasattr(self, "cup"):
            self._scene.remove_actor(self.cup)
        self._setup_bottle1()
        self._setup_bottle2()
        self._setup_coaster()
        self._setup_cup()
        if reset_info is not None:
            self.bottle1.set_pose(sapien.Pose(p=reset_info["init_bottle1_pose"][:3], q=reset_info["init_bottle1_pose"][3:]))
            self.bottle2.set_pose(sapien.Pose(p=reset_info["init_bottle2_pose"][:3], q=reset_info["init_bottle2_pose"][3:]))
            self.coaster.set_pose(sapien.Pose(p=reset_info["init_coaster_pose"][:3], q=reset_info["init_coaster_pose"][3:]))
            self.cup.set_pose(sapien.Pose(p=reset_info["init_container_pose"][:3], q=reset_info["init_container_pose"][3:]))

    @property
    def left_target_pose(self):
        return np.array([0.6, 0.13, 1.2] + [1, 0, 0, -1])
    
    @property
    def right_target_pose(self):
        return np.array([0.6, -0.13, 1.2] + [1, 0, 0, 1])

    def solution(self):
        if self.variant_idx == 0:
            if self.bottle1.get_pose().p[2] > 0.06 + self.table_height:
                left_pose0 = get_grasp_pose_w_given_direction(self.bottle1, self.bottle1_data, grasp_qpos=[-0.906, 0, 0, 0.424], pre_dis=0.1)
                left_pose1 = get_grasp_pose_w_given_direction(self.bottle1, self.bottle1_data, grasp_qpos=[-0.906, 0, 0, 0.424], pre_dis=-0.075)
            else:
                left_pose0 = get_grasp_pose_w_labeled_direction(self.bottle1, self.bottle1_data, grasp_matrix=np.array([[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,0,1]]), pre_dis=0)
                left_pose1 = get_grasp_pose_w_labeled_direction(self.bottle1, self.bottle1_data, grasp_matrix=np.array([[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,0,1]]), pre_dis=-0.1)
                
            if self.bottle2.get_pose().p[2] > 0.06 + self.table_height:
                right_pose0 = get_grasp_pose_w_given_direction(self.bottle2, self.bottle2_data, grasp_qpos=[0.906, 0, 0, 0.424], pre_dis=0.1)
                right_pose1 = get_grasp_pose_w_given_direction(self.bottle2, self.bottle2_data, grasp_qpos=[0.906, 0, 0, 0.424], pre_dis=-0.075)
            else:
                right_pose0 = get_grasp_pose_w_labeled_direction(self.bottle2, self.bottle2_data, grasp_matrix=np.array([[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,0,1]]), pre_dis=0)
                right_pose1 = get_grasp_pose_w_labeled_direction(self.bottle2, self.bottle2_data, grasp_matrix=np.array([[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,0,1]]), pre_dis=-0.1)
            
            yield ("move_to_pose", {"left_pose": deepcopy(left_pose0), "right_pose": deepcopy(right_pose0)})
            yield ("open_gripper", {"action_mode": "both"})
            yield ("open_gripper", {"action_mode": "both"})
            yield ("move_to_pose", {"left_pose": deepcopy(left_pose1), "right_pose": deepcopy(right_pose1)})
            yield ("move_to_pose", {"left_pose": deepcopy(left_pose1), "right_pose": deepcopy(right_pose1)})
            yield ("close_gripper", {"action_mode": "both"})
            yield ("move_to_pose", {"left_pose": deepcopy(self.left_target_pose), "right_pose": deepcopy(self.right_target_pose)})
        elif self.variant_idx == 1:
            arm = 'left' if self.cup.pose.p[1] > 0 else 'right'
            sign = 1 if arm == 'left' else -1
            pose0 = list(self.cup.get_pose().p + [0, sign*0.048, 0.15]) + [1, 0, 1, 0]
            yield ("move_to_pose", {f"{arm}_pose": deepcopy(pose0)})
            yield ("open_gripper", {"action_mode": arm, "gripper_target_state": 0.03})  
            pose0[2] -= 0.12
            yield ("move_to_pose", {f"{arm}_pose": deepcopy(pose0)})
            yield ("close_gripper", {"action_mode": arm})
            pose0[2] += 0.12
            yield ("move_to_pose", {f"{arm}_pose": deepcopy(pose0)})
            pose1 = list(self.coaster.get_pose().p + [0, sign*0.024, 0.15]) + [1, 0, 1, 0]
            yield ("move_to_pose", {f"{arm}_pose": deepcopy(pose1)})   
            pose1[2] -= 0.05    
            yield ("move_to_pose", {f"{arm}_pose": deepcopy(pose1)})
            yield ("open_gripper", {"action_mode": arm, "gripper_target_state": 0.03})
            pose1[2] += 0.1
            yield ("move_to_pose", {f"{arm}_pose": deepcopy(pose1)})
            yield ("move_to_pose", {f"{arm}_pose": self.robot.left_init_ee_pose if arm == 'left' else self.robot.right_init_ee_pose})
    
    def _get_info(self):
        left_distance = np.linalg.norm(self.bottle1.get_pose().p - self.left_target_pose[:3])
        right_distance = np.linalg.norm(self.bottle2.get_pose().p - self.right_target_pose[:3])
        left_height = self.bottle1.get_pose().p[2].item()
        right_height = self.bottle2.get_pose().p[2].item()
        left_success = left_distance < 0.1 and left_height >= 1.10
        right_success = right_distance < 0.1 and right_height >= 1.10
        bottle_success = left_success and right_success
        
        eps = 0.025
        coaster_pose = self.coaster.get_pose().p
        cup_pose = self.cup.get_pose().p
        cup_success = abs(cup_pose[0] - coaster_pose[0]) < eps and abs(cup_pose[1] - coaster_pose[1]) < eps and (cup_pose[2] - 0.052 - self.table_height) < 0.005
        
        if self.variant_idx == 0:
            success = bottle_success
        else:
            success = cup_success
        
        return dict(
            left_success=left_success,
            right_success=right_success,
            bottle_success=bottle_success,
            left_distance=left_distance,
            right_distance=right_distance,
            left_height=left_height,
            right_height=right_height,
            success=success,
            cup_success=cup_success,
        )
        
    def _get_reset_info(self):
        return dict(
            init_bottle1_pose=np.concatenate([self.bottle1.get_pose().p, self.bottle1.get_pose().q]),
            init_bottle2_pose=np.concatenate([self.bottle2.get_pose().p, self.bottle2.get_pose().q]),
            init_coaster_pose=np.concatenate([self.coaster.get_pose().p, self.coaster.get_pose().q]),
            init_container_pose=np.concatenate([self.cup.get_pose().p, self.cup.get_pose().q]),
        )
        
    def _check_termination(self) -> bool:
        return bool(self._get_info()["success"])
    
    @property
    def language_instruction(self):
        if self.variant_idx == 0:
            return "pick up the two bottles simultaneously"
        elif self.variant_idx == 1:
            return "place the cup on the coaster"
        else:
            return ""
    
    def get_object_dict(self):
        return dict(
            bottle1=np.concatenate([self.bottle1.get_pose().p, self.bottle1.get_pose().q]),
            bottle2=np.concatenate([self.bottle2.get_pose().p, self.bottle2.get_pose().q]),
        )
        
    def _get_reward(self):
        return 1.0 if self._get_info()["success"] else 0.0