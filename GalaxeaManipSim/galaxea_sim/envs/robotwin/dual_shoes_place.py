from copy import deepcopy

import sapien
import numpy as np

from galaxea_sim.utils.robotwin_utils import create_visual_box, create_glb, get_grasp_pose_w_labeled_direction
from galaxea_sim.utils.rand_utils import rand_pose
from .robotwin_base import RoboTwinBaseEnv

class DualShoesPlaceEnv(RoboTwinBaseEnv):        
    def _setup_target(self):
        self.target = create_visual_box(
            self._scene,
            pose=sapien.Pose(p=np.array([-0.13, 0, 0]) + self.tabletop_center_in_world),
            half_size=(0.13, 0.1, 0.0005),
            color=(0, 0, 1),
            name="box",
        )
        
    @property
    def id_list(self):
        return list(range(5))[::2]
    
    def _setup_shoes(self, shoe_id=None):
        if shoe_id is None:
            shoe_id = np.random.choice(self.id_list)
        self._shoe_id = int(shoe_id)
        shoe_pose = None
        while shoe_pose is None or np.linalg.norm(shoe_pose.p) < 0.15:
            shoe_pose = rand_pose(
                xlim=[-0.25, -0.2],
                ylim=[0.2, 0.3],
                zlim=[0.06],
                rotate_rand=True,
                rotate_lim=[0, 3.14, 0],
                qpos=[0.5, 0.5, -0.5, -0.5],
            )
        self.shoe1, self.shoe1_data = create_glb(
            scene=self._scene,
            modelname="041_shoes",
            pose=shoe_pose,
            model_id=shoe_id,
            model_z_val=True,
            tabeltop_center_in_world=self.tabletop_center_in_world,
            convex=True,
        )
        
        shoe_pose = None
        while shoe_pose is None or np.linalg.norm(shoe_pose.p) < 0.15:
            shoe_pose = rand_pose(
                xlim=[-0.25, -0.2],
                ylim=[-0.3, -0.2],
                zlim=[0.06],
                rotate_rand=True,
                rotate_lim=[0, 3.14, 0],
                qpos=[0.5, 0.5, -0.5, -0.5],
            )
        self.shoe2, self.shoe2_data = create_glb(
            scene=self._scene,
            modelname="041_shoes",
            pose=shoe_pose,
            model_id=shoe_id,
            model_z_val=True,
            tabeltop_center_in_world=self.tabletop_center_in_world,
            convex=True,
        )

    def reset_world(self, reset_info=None):
        if hasattr(self, "target"):
            self._scene.remove_actor(self.target)
        if hasattr(self, "shoe1"):
            self._scene.remove_actor(self.shoe1)
        if hasattr(self, "shoe2"):
            self._scene.remove_actor(self.shoe2)
        self._setup_shoes(reset_info["shoe_id"] if reset_info is not None else None)
        self._setup_target()
        if reset_info is not None:
            self.shoe1.set_pose(sapien.Pose(p=reset_info["init_shoe1_pose"][:3], q=reset_info["init_shoe1_pose"][3:]))
            self.shoe2.set_pose(sapien.Pose(p=reset_info["init_shoe2_pose"][:3], q=reset_info["init_shoe2_pose"][3:]))

    def get_target_grap_pose(self,shoe_rpy):
        if np.fmod(np.fmod(shoe_rpy[2]+shoe_rpy[0], 2*np.pi)+2*np.pi, 2*np.pi) < np.pi:
            grasp_matrix = np.array([[-1, 0, 0, 0],[0, 1, 0, 0], [0 ,0, -1, 0], [0, 0, 0, 1]])
            # target_quat = [0.5, 0.5, 0.5, -0.5]
            target_quat = [0, 0.707, 0, -0.707]
        else:
            grasp_matrix = np.eye(4)
            # target_quat = [0.5, -0.5, 0.5, 0.5]
            target_quat = [-0.707, 0, -0.707, 0]
        return grasp_matrix, target_quat

    def solution(self):
        left_shoe_rpy = self.shoe1.get_pose().get_rpy()
        right_shoe_rpy = self.shoe2.get_pose().get_rpy()

        left_grasp_matrix, left_target_quat = self.get_target_grap_pose(left_shoe_rpy)
        right_grasp_matrix, right_target_quat = self.get_target_grap_pose(right_shoe_rpy)
        left_pose1 = get_grasp_pose_w_labeled_direction(self.shoe1, self.shoe1_data, grasp_matrix=left_grasp_matrix, pre_dis=0.03)
        right_pose1 = get_grasp_pose_w_labeled_direction(self.shoe2, self.shoe2_data, grasp_matrix=right_grasp_matrix, pre_dis=0.03)
        yield ("move_to_pose", {"left_pose": deepcopy(left_pose1), "right_pose": deepcopy(right_pose1)})
        yield ("open_gripper", {"action_mode": "both"})
        left_pose2 = get_grasp_pose_w_labeled_direction(self.shoe1, self.shoe1_data, grasp_matrix=left_grasp_matrix, pre_dis=-0.05)
        right_pose2 = get_grasp_pose_w_labeled_direction(self.shoe2, self.shoe2_data, grasp_matrix=right_grasp_matrix, pre_dis=-0.05)
        yield ("move_to_pose", {"left_pose": deepcopy(left_pose2), "right_pose": deepcopy(right_pose2)})
        yield ("close_gripper", {"action_mode": "both"})
        left_pose2[2] += 0.1
        right_pose2[2] += 0.1
        yield ("move_to_pose", {"left_pose": deepcopy(left_pose2), "right_pose": deepcopy(right_pose2)})
        
        right_temp_pose = [self.tabletop_center_in_world[0], self.tabletop_center_in_world[1]-0.25, right_pose2[2]] + right_target_quat
        left_target_pose = [self.tabletop_center_in_world[0]-0.1, self.tabletop_center_in_world[1]+0.05, left_pose2[2]] + left_target_quat
        right_target_pose = [self.tabletop_center_in_world[0]-0.1, self.tabletop_center_in_world[1]-0.05, right_pose2[2]] + right_target_quat
                
        yield ("move_to_pose", {"left_pose": deepcopy(left_target_pose), "right_pose": deepcopy(right_temp_pose)})
        left_target_pose[2] -= 0.06
        yield ("move_to_pose", {"left_pose": deepcopy(left_target_pose)})
        yield ("open_gripper", {"action_mode": "left"})
        left_target_pose[2] += 0.06
        yield ("move_to_pose", {"left_pose": deepcopy(left_target_pose)})
        yield ("move_to_pose", {"left_pose": deepcopy(self.robot.left_init_ee_pose), "right_pose": deepcopy(right_target_pose)})
        right_target_pose[2] -= 0.06
        yield ("move_to_pose", {"right_pose": deepcopy(right_target_pose), "left_pose": deepcopy(self.robot.left_init_ee_pose)})
        yield ("open_gripper", {"action_mode": "right"})
        yield ("move_to_pose", {"right_pose": deepcopy(self.robot.right_init_ee_pose), "left_pose": deepcopy(self.robot.left_init_ee_pose)})
        
    def _get_reset_info(self):
        return dict(
            init_shoe1_pose=np.concatenate([self.shoe1.get_pose().p, self.shoe1.get_pose().q]),
            init_shoe2_pose=np.concatenate([self.shoe2.get_pose().p, self.shoe2.get_pose().q]),
            shoe_id=self._shoe_id,
        )
    
    def _get_info(self):
        left_shoe_pose_p = np.array(self.shoe1.get_pose().p)
        left_shoe_pose_q = np.array(self.shoe1.get_pose().q)
        right_shoe_pose_p = np.array(self.shoe2.get_pose().p)
        right_shoe_pose_q = np.array(self.shoe2.get_pose().q)
        if left_shoe_pose_q[0] < 0:
            left_shoe_pose_q *= -1
        if right_shoe_pose_q[0] < 0:
            right_shoe_pose_q *= -1
        target_pose_p = np.array([self.tabletop_center_in_world[0]-0.1, self.tabletop_center_in_world[1]])
        target_pose_q = np.array([0.5, 0.5, 0.5, 0.5])
        eps = np.array([0.05, 0.05, 0.075, 0.075, 0.075, 0.075])
        success = np.all(abs(left_shoe_pose_p[:2] - (target_pose_p + [0, 0.05])) < eps[:2]) and np.all(abs(left_shoe_pose_q - target_pose_q) < eps[-4:]) and \
                  np.all(abs(right_shoe_pose_p[:2] - (target_pose_p - [0, 0.05])) < eps[:2]) and np.all(abs(right_shoe_pose_q - target_pose_q) < eps[-4:]) and \
                  left_shoe_pose_p[2] < 1 and right_shoe_pose_p[2] < 1
        return dict(
            success=success
        )
        
    def _check_termination(self) -> bool:
        return bool(self._get_info()["success"])
    
    @property
    def language_instruction(self):
        return "place the shoes on the target"
    
    def get_object_dict(self):
        return dict(
            shoe1=np.concatenate([self.shoe1.get_pose().p, self.shoe1.get_pose().q]),
            shoe2=np.concatenate([self.shoe2.get_pose().p, self.shoe2.get_pose().q]),
            shoe_id=np.array([id == self._shoe_id for id in self.id_list], dtype=np.float32),
        )
        
    def _get_reward(self):
        return 1.0 if self._get_info()["success"] else 0.0