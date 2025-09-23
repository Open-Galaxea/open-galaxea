from copy import deepcopy

import sapien
import numpy as np

from galaxea_sim.utils.robotwin_utils import create_obj, rand_create_urdf_obj
from galaxea_sim.utils.rand_utils import rand_pose
from .robotwin_base import RoboTwinBaseEnv

class PutAppleCabinetEnv(RoboTwinBaseEnv):
    def _setup_apple(self):
        apple_pose = rand_pose(
            xlim=[-0.2, -0.1],
            ylim=[-0.32, -0.2],
            zlim=[0.04],
            rotate_rand=False,
        )
        self.apple, self.apple_data = create_obj(
            self._scene,
            pose=apple_pose,
            modelname="035_apple",
            convex=True,
            tabeltop_center_in_world=self.tabletop_center_in_world,
        )   
        
    def _setup_cabinet(self):
        self.cabinet, _ = rand_create_urdf_obj(
            self._scene,
            modelname="036_cabine",
            xlim=[0.2, 0.2],
            ylim=[0.1, 0.1],
            zlim=[0.22],
            rotate_rand=False,
            qpos=[1, 0, 0, 0],
            scale=0.27,
            tabletop_center_in_world=self.tabletop_center_in_world,
        )
        self.cabinet_active_joints = self.cabinet.get_active_joints()
        for joint in self.cabinet_active_joints:
            joint.set_drive_property(stiffness=20, damping=5, force_limit=1000, mode="force")
        self.cabinet_all_joints = self.cabinet.get_joints()

    def reset_world(self, reset_info=None):
        # chek if red bottle on the scene, then remove it
        if hasattr(self, "apple"):
            self._scene.remove_actor(self.apple)
        if hasattr(self, "cabinet"):
            self.cabinet.set_qpos([0, 0, 0])
        else:
            self._setup_cabinet()
        self._setup_apple()
        if reset_info is not None:
            self.apple.set_pose(sapien.Pose(p=reset_info["init_apple_pose"][:3], q=reset_info["init_apple_pose"][3:]))
        
    def solution(self):
        pre_pose0 = list(self.cabinet.get_pose().p + [-0.25, 0.07, 0]) + [0.707, 0.707, 0, 0]
        pose0 = list(self.cabinet.get_pose().p + [-0.25, 0.07, -0.09]) + [0.707, 0.707, 0, 0]
        pose1 = list(self.apple.get_pose().p + [0, 0, 0.17]) + [0.707, 0, 0.707, 0]
        yield ("move_to_pose", {"left_pose": deepcopy(pre_pose0), "right_pose": deepcopy(pose1)})
        yield ("move_to_pose", {"left_pose": deepcopy(pose0)})
        yield ("open_gripper", {"action_mode": "both"})
        pose0[0] += 0.075
        pose1[2] -= 0.15
        yield ("move_to_pose", {"left_pose": deepcopy(pose0), "right_pose": deepcopy(pose1)})
        yield ("close_gripper", {"action_mode": "both"})
        pose0[0] -= 0.15
        pose1[2] += 0.15
        yield ("move_to_pose", {"left_pose": deepcopy(pose0), "right_pose": deepcopy(pose1)})
        
        pose2 = list(self.cabinet.get_pose().p + [-0.18, -0.12, 0.02]) + [0.707, 0, 0, 0.707]
        yield ("move_to_pose", {"right_pose": deepcopy(pose2)})
        yield ("open_gripper", {"action_mode": "right", "steps": 30})
        pose0[0] += 0.15
        pose2[1] -= 0.15
        yield ("move_to_pose", {"left_pose": deepcopy(pose0), "right_pose": deepcopy(pose2)})
        yield ("open_gripper", {"action_mode": "left"})
        pose0[0] -= 0.05
        yield ("move_to_pose", {"left_pose": deepcopy(pose0)})
        yield ("move_to_pose", {"left_pose": self.robot.left_init_ee_pose, "right_pose": self.robot.right_init_ee_pose})
        
    def _get_info(self):
        success = self.apple.get_pose().p[0] - self.cabinet.get_pose().p[0] < 0.2 and self.apple.get_pose().p[0] > self.cabinet.get_pose().p[0] - 0.2
        return dict(
            success=success
        )
    
    def _get_reset_info(self):
        return dict(
            init_apple_pose=np.concatenate([self.apple.get_pose().p, self.apple.get_pose().q]),
        )
        
    def _check_termination(self) -> bool:
        return bool(self._get_info()["success"])
    
    @property
    def language_instruction(self):
        return "put the apple in the cabinet" 
    
    def get_object_dict(self):
        return dict(
            apple=np.concatenate([self.apple.get_pose().p, self.apple.get_pose().q]),
            cabinet_qpos=np.array(self.cabinet.get_qpos()),
        )
        
    def _get_reward(self):
        return 1.0 if self._get_info()["success"] else 0.0