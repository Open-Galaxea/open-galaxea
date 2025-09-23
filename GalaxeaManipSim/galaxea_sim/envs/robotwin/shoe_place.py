from copy import deepcopy

import sapien
import numpy as np

from galaxea_sim.utils.robotwin_utils import create_visual_box, create_glb, get_grasp_pose_w_labeled_direction
from galaxea_sim.utils.rand_utils import rand_pose
from .robotwin_base import RoboTwinBaseEnv

class ShoePlaceEnv(RoboTwinBaseEnv):        
    def _setup_target(self):
        self.target = create_visual_box(
            self._scene,
            pose=sapien.Pose(p=np.array([-0.13, 0, 0]) + self.tabletop_center_in_world),
            half_size=(0.13, 0.05, 0.0005),
            color=(0, 0, 1),
            name="box",
        )
        
    @property
    def id_list(self):
        if self.eval_mode:
            return list(range(5))[1::2]
        else:
            return list(range(5))[::2]
    
    def _setup_shoes(self, shoe_id=None):
        if shoe_id is None:
            shoe_id = np.random.choice(self.id_list)
        self._shoe_id = int(shoe_id)
        shoe_pose = None
        while shoe_pose is None or np.linalg.norm(shoe_pose.p) < 0.15:
            shoe_pose = rand_pose(
                xlim=[-0.25, -0.2],
                ylim=[-0.25, 0.25],
                zlim=[0.06],
                rotate_rand=True,
                rotate_lim=[0, 3.14, 0],
                qpos=[0.5, 0.5, -0.5, -0.5],
            )
        self.shoe, self.shoe_data = create_glb(
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
        if hasattr(self, "shoe"):
            self._scene.remove_actor(self.shoe)
        self._setup_shoes(reset_info["shoe_id"] if reset_info is not None else None)
        self._setup_target()
        if reset_info is not None:
            self.shoe.set_pose(sapien.Pose(p=reset_info["init_shoe_pose"][:3], q=reset_info["init_shoe_pose"][3:]))

    def get_target_grap_pose(self,shoe_rpy):
        if np.fmod(np.fmod(shoe_rpy[2]+shoe_rpy[0], 2*np.pi)+2*np.pi, 2*np.pi) < np.pi:
            grasp_matrix = np.array([[-1, 0, 0, 0],[0, 1, 0, 0], [0 ,0, -1, 0], [0, 0, 0, 1]])
            target_quat = [0, 0.707, 0, -0.707]
        else:
            grasp_matrix = np.eye(4)
            target_quat = [-0.707, 0, -0.707, 0]
        return grasp_matrix, target_quat

    def solution(self):
        arm = 'left' if self.shoe.get_pose().p[1] > 0 else 'right'
        init_ee_pose = self.robot.left_ee_link.get_entity_pose() if arm == 'left' else self.robot.right_ee_link.get_entity_pose()
        shoe_rpy = self.shoe.get_pose().get_rpy()

        grasp_matrix, target_quat = self.get_target_grap_pose(shoe_rpy)
        pose1 = get_grasp_pose_w_labeled_direction(self.shoe, self.shoe_data, grasp_matrix=grasp_matrix, pre_dis=0.03)
        yield ("move_to_pose", {f"{arm}_pose": deepcopy(pose1)})
        yield ("open_gripper", {"action_mode": arm})
        pose2 = get_grasp_pose_w_labeled_direction(self.shoe, self.shoe_data, grasp_matrix=grasp_matrix, pre_dis=-0.05)
        yield ("move_to_pose", {f"{arm}_pose": deepcopy(pose2)})
        yield ("close_gripper", {"action_mode": arm})
        pose2[2] += 0.1
        yield ("move_to_pose", {f"{arm}_pose": deepcopy(pose2)})
        
        target_pose = [self.tabletop_center_in_world[0]-0.1, self.tabletop_center_in_world[1], pose2[2]] + target_quat
               
        yield ("move_to_pose", {f"{arm}_pose": deepcopy(target_pose)})
        target_pose[2] -= 0.06
        yield ("move_to_pose", {f"{arm}_pose": deepcopy(target_pose)})
        yield ("open_gripper", {"action_mode": arm})
        target_pose[2] += 0.06
        yield ("move_to_pose", {f"{arm}_pose": deepcopy(target_pose)})
        yield ("move_to_pose", {f"{arm}_pose": deepcopy(init_ee_pose)})
        
    def _get_reset_info(self):
        return dict(
            init_shoe_pose=np.concatenate([self.shoe.get_pose().p, self.shoe.get_pose().q]),
            shoe_id=self._shoe_id,
        )
    
    def _get_info(self):
        shoe_pose_p = np.array(self.shoe.get_pose().p)
        shoe_pose_q = np.array(self.shoe.get_pose().q)
        if shoe_pose_q[0] < 0:
            shoe_pose_q *= -1
        target_pose_p = np.array([self.tabletop_center_in_world[0]-0.1, self.tabletop_center_in_world[1]])
        target_pose_q = np.array([0.5, 0.5, 0.5, 0.5])
        eps = np.array([0.05, 0.05, 0.075, 0.075, 0.075, 0.075])
        success = np.all(abs(shoe_pose_p[:2] - target_pose_p) < eps[:2]) and np.all(abs(shoe_pose_q - target_pose_q) < eps[-4:]) and \
                  shoe_pose_p[2] < 1
        return dict(
            success=success
        )
        
    def _check_termination(self) -> bool:
        return bool(self._get_info()["success"])
    
    @property
    def language_instruction(self):
        return "place the shoe on the target"
    
    def get_object_dict(self):
        return dict(
            shoe=np.concatenate([self.shoe.get_pose().p, self.shoe.get_pose().q]),
            shoe_id=np.array([id == self._shoe_id for id in self.id_list], dtype=np.float32),
        )
        
    def _get_reward(self):
        return 1.0 if self._get_info()["success"] else 0.0