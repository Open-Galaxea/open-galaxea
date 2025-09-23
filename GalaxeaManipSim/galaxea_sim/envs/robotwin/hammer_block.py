import math
from copy import deepcopy

import sapien
import numpy as np
import transforms3d as t3d

from galaxea_sim.utils.robotwin_utils import create_box, create_glb, get_grasp_pose_w_labeled_direction, rand_pose, get_target_pose_from_goal_point_and_direction, get_actor_goal_pose
from galaxea_sim.utils.rand_utils import rand_pose
from .robotwin_base import RoboTwinBaseEnv

class HammerBlockEnv(RoboTwinBaseEnv):
    def _setup_hammer(self):
        self.hammer, self.hammer_data = create_glb(
            self._scene,
            pose=sapien.Pose([-0.06, 0, 0.043], [0.072, 0.703, 0.703, 0.075]),
            modelname="020_hammer_2",
            tabeltop_center_in_world=self.tabletop_center_in_world,
            convex=True,
        )
        
    def _setup_block(self): 
        self.block_half_size = 0.02
        self.block_xlim = [-0.15, 0.05]
        self.block_ylim = [-0.25, 0.25]
        self.block_zlim = [self.block_half_size]
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
            half_size=(self.block_half_size, self.block_half_size, self.block_half_size),
            color=(1, 0, 0),
            name="box"
        )
        
    def _setup_block2(self): 
        rand_pos = rand_pose(
            xlim=[-0.2, 0.0],
            ylim=[-0.25, 0.25],
            zlim=[self.block_half_size],
            qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=True,
            rotate_lim=[0, 1.57, 0],
        )
        while abs(rand_pos.p[1]) < 0.05 or np.sum(pow(rand_pos.p[:2] - np.array([-0.1, 0]), 2)) < 0.0225 \
            or np.linalg.norm(rand_pos.p[:2] - self.block.get_pose().p[:2] + self.tabletop_center_in_world[:2]) < 0.06:
            rand_pos = rand_pose(
                xlim=[-0.2, 0.0],
                ylim=[-0.25, 0.25],
                zlim=[self.block_half_size],
                qpos=[0.5, 0.5, 0.5, 0.5],
                rotate_rand=True,
                rotate_lim=[0, 1.57, 0],
            )
        rand_pos.set_p(rand_pos.p + self.tabletop_center_in_world)
        self.block2 = create_box(
            scene=self._scene,
            pose=rand_pos,
            half_size=(self.block_half_size, self.block_half_size, self.block_half_size),
            color=(0,1,0),
            name="box"
        )
        
        
    def reset_world(self, reset_info=None):
        if hasattr(self, "hammer"):
            self._scene.remove_actor(self.hammer)
        if hasattr(self, "block"):
            self._scene.remove_actor(self.block)
        if hasattr(self, "block2"):
            self._scene.remove_actor(self.block2)
        self._setup_hammer()
        self._setup_block()
        self._setup_block2()
        if reset_info is not None:
            self.hammer.set_pose(sapien.Pose(reset_info["hammer_pose"][:3], reset_info["hammer_pose"][3:]))
            self.block.set_pose(sapien.Pose(reset_info["block_pose"][:3], reset_info["block_pose"][3:]))
            self.block2.set_pose(sapien.Pose(reset_info["block2_pose"][:3], reset_info["block2_pose"][3:]))

    def move_block(self, actor: sapien.Entity, id, last_arm=None):
        actor_rpy = actor.get_pose().get_rpy()
        actor_pos = actor.get_pose().p
        actor_euler = math.fmod(actor_rpy[2], math.pi / 2)
        grasp_euler = actor_euler - math.pi/2  if actor_euler > math.pi/4 else actor_euler
        grasp_trans_quat = t3d.euler.euler2quat(0, 0, grasp_euler)
        if actor_pos[1] < 0:
            grasp_qpose = t3d.quaternions.qmult(grasp_trans_quat, [-0.5, 0.5, -0.5, -0.5]).tolist()
            target_pose = [self.tabletop_center_in_world[0] - 0.1, 0.005, self.table_height + id * self.block_half_size * 2 + 0.1, -0.5, 0.5, -0.5, -0.5]
        else:
            grasp_qpose = t3d.quaternions.qmult(grasp_trans_quat, [0.5, 0.5, 0.5, -0.5]).tolist()
            target_pose = [self.tabletop_center_in_world[0] - 0.1, -0.005, self.table_height + id * self.block_half_size * 2 + 0.1, 0.5, 0.5, 0.5, -0.5]
        
        substeps = []
        pose1 = list(actor_pos + [0, 0, 0.2]) + grasp_qpose
        if actor_pos[1] < 0:
            now_arm = 'right'
            if now_arm == last_arm or last_arm is None:
                if now_arm == last_arm:
                    pose0 = list(self.robot.right_ee_link.get_entity_pose().p + [0, 0, 0.05]) + [-0.5, 0.5, -0.5, -0.5]
                    substeps.append(("move_to_pose", {"right_pose": pose0}))
                substeps.append(("move_to_pose", {"right_pose": deepcopy(pose1)}))
            else:
                substeps.append(("move_to_pose", {"right_pose": pose1, "left_pose": self.robot.left_init_ee_pose}))
        else:
            now_arm = 'left'
            if now_arm == last_arm or last_arm is None:
                if now_arm == last_arm:
                    pose0 = list(self.robot.left_ee_link.get_entity_pose().p + [0, 0, 0.05]) + [-0.5, 0.5, -0.5, -0.5]
                    substeps.append(("move_to_pose", {"left_pose": pose0}))
                substeps.append(("move_to_pose", {"left_pose": deepcopy(pose1)}))
            else:
                substeps.append(("move_to_pose", {"left_pose": pose1, "right_pose": self.robot.right_init_ee_pose}))
                
        substeps.append(("open_gripper", {"action_mode": now_arm}))
        pose1[2] -= 0.15
        substeps.append(("move_to_pose", {f"{now_arm}_pose": deepcopy(pose1)}))
        substeps.append(("close_gripper", {"action_mode": now_arm}))
        pose1[2] += 0.15
        substeps.append(("move_to_pose", {f"{now_arm}_pose": deepcopy(pose1)}))
        substeps.append(("move_to_pose", {f"{now_arm}_pose": deepcopy(target_pose)}))
        target_pose[2] -= 0.05
        substeps.append(("move_to_pose", {f"{now_arm}_pose": deepcopy(target_pose)}))
        substeps.append(("open_gripper", {"action_mode": now_arm}))
        target_pose[2] += 0.1
        substeps.append(("move_to_pose", {f"{now_arm}_pose": deepcopy(target_pose)}))
            
        return substeps, now_arm
    
    def solution(self):
        if self.variant_idx == 0:
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
            ) # target pose
            yield ("move_to_pose", {f"{action_mode}_pose": deepcopy(pose4)})
            pose5 = pose4 + np.array([0, 0, -0.1, 0, 0, 0, 0]) # drop pose
            yield ("move_to_pose", {f"{action_mode}_pose": deepcopy(pose5)})
        elif self.variant_idx == 1:
            substeps, last_arm = self.move_block(self.block, 1)
            for substep in substeps:
                yield substep
            substeps, last_arm = self.move_block(self.block2, 2, last_arm)
            for substep in substeps:
                yield substep
    
    def _get_info(self):
        hammer_target_pose = get_actor_goal_pose(self.hammer, self.hammer_data)
        block_pose = self.block.get_pose().p
        eps = np.array([0.02, 0.02])
        success_hammer = np.all(np.abs(hammer_target_pose[:2] - block_pose[:2]) < eps) and hammer_target_pose[2] < self.table_height + 0.07 and hammer_target_pose[2] > self.table_height + 0.04
        
        block1_pose = self.block.get_pose().p
        block2_pose = self.block2.get_pose().p
        target_pose = [-0.1 + self.tabletop_center_in_world[0], 0]
        eps = [0.025, 0.025, 0.01]
        success_block = np.all(np.abs(block1_pose - np.array(target_pose + [self.table_height + self.block_half_size])) < eps) and \
               np.all(np.abs(block2_pose - np.array(target_pose + [self.table_height + self.block_half_size * 3])) < eps)      
        if self.variant_idx == 0:
            success = success_hammer
        elif self.variant_idx == 1:
            success = success_block
        else:
            success = False
        return dict(
            hammer_target_pos=hammer_target_pose,
            block_pos=block_pose,
            success_hammer=success_hammer,
            success_block=success_block,
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
        if self.variant_idx == 0:
            return "use the hammer to hit the block on the table."
        elif self.variant_idx == 1:
            return "stack the blocks on the table."
        else:
            return ""
    
    def get_object_dict(self):
        return dict(
            hammer=np.concatenate([self.hammer.get_pose().p, self.hammer.get_pose().q], axis=0),
            block=np.concatenate([self.block.get_pose().p, self.block.get_pose().q], axis=0),
        )
        
    def _get_reward(self):
        return 1.0 if self._get_info()["success"] else 0.0