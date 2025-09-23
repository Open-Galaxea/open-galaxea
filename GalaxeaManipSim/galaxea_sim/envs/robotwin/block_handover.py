import sapien
import numpy as np

from galaxea_sim.utils.robotwin_utils import create_box, rand_pose
from galaxea_sim.utils.rand_utils import rand_pose
from .robotwin_base import RoboTwinBaseEnv

class BlockHandoverEnv(RoboTwinBaseEnv):
    @property
    def table_height(self):
        if self.robot_name == "r1_lite":
            return 0.8
        else:
            return super().table_height
        
    @property
    def tabletop_center_x(self):
        if self.robot_name == "r1_lite":
            return 0.6
        else:
            return super().tabletop_center_x
    
    def _setup_box(self):
        rand_pos = rand_pose(
            xlim=[-0.125, 0.125],
            ylim=[0.05, 0.25],
            zlim=[0.842 - 0.74],
            qpos=[-0.906, 0, 0, 0.424]
        )
        rand_pos.set_p(rand_pos.p + self.tabletop_center_in_world)
        self.box = create_box(
            scene=self._scene,
            pose=rand_pos,
            half_size=(0.0225,0.0225,0.1),
            color=(1, 0, 0),
            name="box",
        )
        
    def _setup_target(self): 
        rand_pos = rand_pose(
            xlim=[0.09,0.09],
            ylim=[-0.23,-0.23],
            zlim=[0],
        )
        rand_pos.set_p(rand_pos.p + self.tabletop_center_in_world)
        self.target = create_box(
            scene=self._scene,
            pose=rand_pos,
            half_size=(0.05,0.05,0.005),
            color=(0,0,1),
            name="box"
        )
        
        
    def reset_world(self, reset_info=None):
        if hasattr(self, "box"):
            self._scene.remove_actor(self.box)
        if hasattr(self, "target"):
            self._scene.remove_actor(self.target)
        self._setup_box()
        self._setup_target()
        if reset_info is not None:
            self.box.set_pose(sapien.Pose(reset_info["box_pose"][:3], reset_info["box_pose"][3:]))
            self.target.set_pose(sapien.Pose(reset_info["target_pose"][:3], reset_info["target_pose"][3:]))
        
    def solution(self):
        init_left_pose = self.robot.left_ee_link.get_entity_pose()
        init_right_pose = self.robot.right_ee_link.get_entity_pose()
        left_pose0 = list(self.box.get_pose().p+[-0.14,0.18,0.07])+[-0.906,0,0,0.424]
        left_pose1 = list(self.box.get_pose().p+[-0.08/3,0.11/3,0.07])+[-0.906,0,0,0.424]
        left_target_pose = [0.6, 0.1, self.table_height + 0.3, 1, 0, 0, -1]
        right_pick_pre_pose = [0.6, -0.1, self.table_height + 0.2, 1, 0, 0, 1]
        right_pick_pose = [0.6, 0.025, self.table_height + 0.2, 1, 0, 0, 1]
        yield ("move_to_pose", {"left_pose": left_pose0})
        yield ("open_gripper", {"action_mode": "left"})
        yield ("move_to_pose", {"left_pose": left_pose1})
        yield ("close_gripper", {"action_mode": "left"})
        left_pose1[2] += 0.06
        yield ("move_to_pose", {"left_pose": left_pose1})
        yield ("move_to_pose", {"left_pose": left_target_pose, "right_pose": right_pick_pre_pose})
        yield ("open_gripper", {"action_mode": "right", "step": 20})
        yield ("move_to_pose", {"right_pose": right_pick_pose})
        yield ("close_gripper", {"action_mode": "right"})
        yield ("close_gripper", {"action_mode": "right"})
        yield ("open_gripper", {"action_mode": "left"})
        
        right_pick_pose[1] -= 0.1
        left_target_pose[1] += 0.1
        yield ("move_to_pose", {"left_pose": left_target_pose, "right_pose": right_pick_pose})
        
        right_target_pose = list(self.target.get_pose().p + [-0.05, 0, 0.3]) + [1, 0, 0, 0]
        
        yield ("move_to_pose", {"right_pose": right_target_pose, "left_pose": init_left_pose})
        
        right_target_pose[2] -= 0.25
        yield ("move_to_pose", {"right_pose": right_target_pose})
        yield ("open_gripper", {"action_mode": "right"})
        right_target_pose[0] -= 0.2
        yield ("move_to_pose", {"right_pose": right_target_pose})
        yield ("move_to_pose", {"right_pose": init_right_pose})
        
    
    def _get_info(self):
        box_pos = self.box.get_pose().p
        target_pos = self.target.get_pose().p
        eps = 0.04
        success = abs(box_pos[0] - target_pos[0]) < eps and abs(box_pos[1] - target_pos[1]) < eps and abs(box_pos[2] - 0.11 - self.table_height) < 0.0015
        return dict(
            success=success,
        )
    
    def _get_reset_info(self):
        return dict(
            box_pose=np.concatenate([self.box.get_pose().p, self.box.get_pose().q], axis=0),
            target_pose=np.concatenate([self.target.get_pose().p, self.target.get_pose().q], axis=0),
        )
        
    def _check_termination(self) -> bool:
        return bool(self._get_info()["success"])
    
    @property
    def language_instruction(self):
        return "handover the block to the target position."
    
    def get_object_dict(self):
        return dict(
            box=np.concatenate([self.box.get_pose().p, self.box.get_pose().q], axis=0),
            target=np.concatenate([self.target.get_pose().p, self.target.get_pose().q], axis=0),
        )
        
    def _get_reward(self):
        return 1.0 if self._get_info()["success"] else 0.0