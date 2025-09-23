from copy import deepcopy

import sapien
import numpy as np

from galaxea_sim.utils.robotwin_utils import create_box, create_glb, get_grasp_pose_w_labeled_direction, rand_pose, get_target_pose_from_goal_point_and_direction, get_actor_goal_pose
from galaxea_sim.utils.rand_utils import rand_pose
from .robotwin_base import RoboTwinBaseEnv

class BlockHammerBeatEnv(RoboTwinBaseEnv):
    def _setup_hammer(self):
        self.hammer, self.hammer_data = create_glb(
            self._scene,
            pose=sapien.Pose([-0.06, 0, 0.043], [0.072, 0.703, 0.703, 0.075]),
            modelname="020_hammer_2",
            tabeltop_center_in_world=self.tabletop_center_in_world,
            convex=True,
        )
        
    def _setup_block(self): 
        self.block_xlim = [-0.15, 0.05]
        self.block_ylim = [-0.25, 0.25]
        self.block_zlim = [0.02]
        self.block_qpos = [0.5, 0.5, 0.5, 0.5]
        
        self.block_pose_wrt_table = rand_pose(
            xlim=self.block_xlim,
            ylim=self.block_ylim,
            zlim=self.block_zlim,
            qpos=self.block_qpos,
            rotate_rand=True,
            rotate_lim=[0, 1, 0],
        )
        while abs(self.block_pose_wrt_table.p[1]) < 0.1:
            self.block_pose_wrt_table = rand_pose(
                xlim=self.block_xlim,
                ylim=self.block_ylim,
                zlim=self.block_zlim,
                qpos=self.block_qpos,
                rotate_rand=True,
                rotate_lim=[0, 1, 0],
            )
        self.block_pose = deepcopy(self.block_pose_wrt_table)
        self.block_pose.set_p(self.block_pose.get_p() + self.tabletop_center_in_world)
        
        self.block = create_box(
            scene=self._scene,
            pose=self.block_pose,
            half_size=(0.025, 0.025, 0.025),
            color=(1, 0, 0),
            name="box"
        )
        
        
    def reset_world(self, reset_info=None):
        if hasattr(self, "hammer"):
            self._scene.remove_actor(self.hammer)
        if hasattr(self, "block"):
            self._scene.remove_actor(self.block)
        self._setup_hammer()
        self._setup_block()
        if reset_info is not None:
            self.hammer.set_pose(sapien.Pose(reset_info["hammer_pose"][:3], reset_info["hammer_pose"][3:]))
            self.block.set_pose(sapien.Pose(reset_info["block_pose"][:3], reset_info["block_pose"][3:]))
        
    def solution(self):
        action_mode = "left" if self.block_pose_wrt_table.p[1] > 0 else "right"
        pose1 = get_grasp_pose_w_labeled_direction(self.hammer, self.hammer_data, pre_dis=0.1) # pre grasp pose
        yield ("open_gripper", {"action_mode": action_mode})
        yield ("move_to_pose", {f"{action_mode}_pose": deepcopy(pose1)})
        pose2 = get_grasp_pose_w_labeled_direction(self.hammer, self.hammer_data, pre_dis=-0.075) # grap pose
        yield ("move_to_pose", {f"{action_mode}_pose": deepcopy(pose2)})
        yield ("close_gripper", {"action_mode": action_mode})
        pose3 = pose2 + np.array([0, 0, 0.1, 0, 0, 0, 0]) # lift pose
        yield ("move_to_pose", {f"{action_mode}_pose": deepcopy(pose3)})
        pose4 = get_target_pose_from_goal_point_and_direction(
            self.hammer,self.hammer_data, self.robot.left_ee_link if action_mode == "left" else self.robot.right_ee_link, 
            target_pose=self.block.get_pose().p+[-0.125,0,0.08], 
            target_grasp_qpose=[0.775443, -0.0042941, 0.631382, 0.00519282]
        ) 
        yield ("move_to_pose", {f"{action_mode}_pose": deepcopy(pose4)})
        pose5 = pose4 + np.array([0, 0, -0.1, 0, 0, 0, 0]) # drop pose
        yield ("move_to_pose", {f"{action_mode}_pose": deepcopy(pose5)})
    
    def _get_info(self):
        hammer_target_pose = get_actor_goal_pose(self.hammer, self.hammer_data)
        block_pose = self.block.get_pose().p
        eps = np.array([0.02, 0.02])
        success = np.all(np.abs(hammer_target_pose[:2] - block_pose[:2]) < eps) and hammer_target_pose[2] < self.table_height + 0.07 and hammer_target_pose[2] > self.table_height + 0.04
        return dict(
            hammer_target_pos=hammer_target_pose,
            block_pos=block_pose,
            success=success,
        )
    
    def _get_reset_info(self):
        return dict(
            hammer_pose=np.concatenate([self.hammer.get_pose().p, self.hammer.get_pose().q], axis=0),
            block_pose=np.concatenate([self.block.get_pose().p, self.block.get_pose().q], axis=0),
        )
        
    def _check_termination(self) -> bool:
        return bool(self._get_info()["success"])
    
    @property
    def language_instruction(self):
        return "use the hammer to hit the block on the table."
    
    def get_object_dict(self):
        return dict(
            hammer=np.concatenate([self.hammer.get_pose().p, self.hammer.get_pose().q], axis=0),
            block=np.concatenate([self.block.get_pose().p, self.block.get_pose().q], axis=0),
        )
        
    def _get_reward(self):
        return 1.0 if self._get_info()["success"] else 0.0