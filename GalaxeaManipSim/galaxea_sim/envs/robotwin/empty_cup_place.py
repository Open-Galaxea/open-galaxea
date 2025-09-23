from copy import deepcopy, copy

import sapien
import numpy as np

from galaxea_sim.utils.robotwin_utils import create_glb, create_obj
from galaxea_sim.utils.rand_utils import rand_pose
from .robotwin_base import RoboTwinBaseEnv

class EmptyCupPlaceEnv(RoboTwinBaseEnv):
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
        while cup_pose is None or np.linalg.norm(cup_pose.p - self.coaster.get_pose().p) < 0.1:
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
        # chek if red bottle on the scene, then remove it
        if hasattr(self, "coaster"):
            self._scene.remove_actor(self.coaster)
        if hasattr(self, "cup"):
            self._scene.remove_actor(self.cup)
        self._setup_coaster()
        self._setup_cup()
        if reset_info is not None:
            self.coaster.set_pose(sapien.Pose(p=reset_info["init_coaster_pose"][:3], q=reset_info["init_coaster_pose"][3:]))
            self.cup.set_pose(sapien.Pose(p=reset_info["init_container_pose"][:3], q=reset_info["init_container_pose"][3:]))
        
    def solution(self):
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
        eps = 0.025
        coaster_pose = self.coaster.get_pose().p
        cup_pose = self.cup.get_pose().p
        success = abs(cup_pose[0] - coaster_pose[0]) < eps and abs(cup_pose[1] - coaster_pose[1]) < eps and (cup_pose[2] - 0.052 - self.table_height) < 0.005
        return dict(
            success=success
        )
    
    def _get_reset_info(self):
        return dict(
            init_coaster_pose=np.concatenate([self.coaster.get_pose().p, self.coaster.get_pose().q]),
            init_container_pose=np.concatenate([self.cup.get_pose().p, self.cup.get_pose().q]),
        )
        
    def _check_termination(self) -> bool:
        return bool(self._get_info()["success"])
    
    @property
    def language_instruction(self):
        return "place the cup on the coaster" 
    
    def get_object_dict(self):
        return dict(
            coaster=np.concatenate([self.coaster.get_pose().p, self.coaster.get_pose().q]),
            cup=np.concatenate([self.cup.get_pose().p, self.cup.get_pose().q]),
        )
        
    def _get_reward(self):
        return 1.0 if self._get_info()["success"] else 0.0