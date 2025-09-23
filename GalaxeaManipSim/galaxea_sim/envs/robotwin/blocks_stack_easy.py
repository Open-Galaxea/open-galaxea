import math
from copy import deepcopy

import sapien
import numpy as np 

from galaxea_sim.utils.robotwin_utils import create_box
from galaxea_sim.utils.rand_utils import rand_pose
from .robotwin_base import RoboTwinBaseEnv

class BlocksStackEasyEnv(RoboTwinBaseEnv):
    place_pos_x_offset = -0.2
    block_half_size = 0.02

    def _rand_pose(self):
        return rand_pose(
            xlim=[-0.12, -0.01],
            ylim=[-0.25, 0.25],
            zlim=[self.block_half_size],
            qpos=[0.27, 0.27, 0.65, 0.65],
            rotate_rand=True,
            rotate_lim=[0, 0.1, 0],
        )

    def _setup_block1(self):
        rand_pos = self._rand_pose()
        while abs(rand_pos.p[1]) < 0.05 \
            or np.sum(pow(rand_pos.p[:2] - np.array([self.place_pos_x_offset, 0]), 2)) < 0.0225:
            rand_pos = self._rand_pose()
        rand_pos.set_p(rand_pos.p + self.tabletop_center_in_world)
        self.block1 = create_box(
            scene=self._scene,
            pose=rand_pos,
            half_size=(self.block_half_size, self.block_half_size, self.block_half_size),
            color=(1,0,0),
            name="box"
        )
        
    def _setup_block2(self): 
        rand_pos = self._rand_pose()
        while abs(rand_pos.p[1]) < 0.05 \
            or np.sum(pow(rand_pos.p[:2] - np.array([self.place_pos_x_offset, 0]), 2)) < 0.0225 \
            or np.linalg.norm((rand_pos.p[:2] + self.tabletop_center_in_world[:2]) - self.block1.get_pose().p[:2]) < 0.1:
            rand_pos = self._rand_pose()
        rand_pos.set_p(rand_pos.p + self.tabletop_center_in_world)
        self.block2 = create_box(
            scene=self._scene,
            pose=rand_pos,
            half_size=(self.block_half_size, self.block_half_size, self.block_half_size),
            color=(0,1,0),
            name="box"
        )

    def reset_world(self, reset_info=None):
        if hasattr(self, "block1"):
            self._scene.remove_actor(self.block1)
        if hasattr(self, "block2"):
            self._scene.remove_actor(self.block2)
        self._setup_block1()
        self._setup_block2()
        if reset_info is not None:
            self.block1.set_pose(sapien.Pose(reset_info["block1_pose"][:3], reset_info["block1_pose"][3:]))
            self.block2.set_pose(sapien.Pose(reset_info["block2_pose"][:3], reset_info["block2_pose"][3:]))

    def rot_down_grip_pose(self, pose: sapien.Pose):
        angle = math.pi/4 if self.robot_name == "r1_pro" else math.pi/2
        pose_mat = pose.to_transformation_matrix()
        (lower_trans_quat := sapien.Pose()).set_rpy(rpy=(np.array([0, angle, 0])))
        lower_trans_mat = lower_trans_quat.to_transformation_matrix()

        new_pos = np.dot(pose_mat, lower_trans_mat)
        new_pose = sapien.Pose(matrix=new_pos)
        return new_pose
    
    def tf_to_grasp(self, pose: list):
        if self.robot_name != "r1_pro":
            return pose
        origin_pose=sapien.Pose(p=pose[:3],q=pose[3:])
        pose_mat = origin_pose.to_transformation_matrix()
        tf_mat = np.array([
            [1,0,0,-0.05],
            [0,1,0,0],
            [0,0,1,0.02],
            [0,0,0,1]
        ])
        new_pos = np.dot(pose_mat, tf_mat)
        new_pose = sapien.Pose(matrix=new_pos)
        return list(new_pose.p) + list(new_pose.q)

    def move_block(self, actor: sapien.Entity, id, last_arm=None):
        actor_rpy = actor.get_pose().get_rpy()
        actor_pos = actor.get_pose().p
        actor_euler = math.fmod(actor_rpy[2], math.pi / 2)
        if actor_pos[1] < 0:
            # closer to right arm
            grasp_euler = actor_euler
            (grasp_qpose := sapien.Pose()).set_rpy(rpy=(np.array([0, 0, grasp_euler]) + self.robot.right_ee_rpy_offset))
            grasp_qpose = self.rot_down_grip_pose(grasp_qpose).q.tolist()

            (target_qpose := sapien.Pose()).set_rpy(rpy=(np.array([0, 0, math.pi/2]) + self.robot.right_ee_rpy_offset))
            target_qpose = self.rot_down_grip_pose(target_qpose).q.tolist()
            target_pose = [self.tabletop_center_in_world[0] + self.place_pos_x_offset, 0.01, self.table_height + id * self.block_half_size * 2 + 0.1] + target_qpose
        else:
            grasp_euler = actor_euler - math.pi/2
            (grasp_qpose := sapien.Pose()).set_rpy(rpy=(np.array([0, 0, grasp_euler]) + self.robot.right_ee_rpy_offset))
            grasp_qpose = self.rot_down_grip_pose(grasp_qpose).q.tolist()

            (target_qpose := sapien.Pose()).set_rpy(rpy=(np.array([0, 0, -math.pi/2]) + self.robot.right_ee_rpy_offset))
            target_qpose = self.rot_down_grip_pose(target_qpose).q.tolist()
            target_pose = [self.tabletop_center_in_world[0] + self.place_pos_x_offset, -0.01, self.table_height + id * self.block_half_size * 2 + 0.1] + target_qpose
        
        self.grasp_euler = grasp_euler
        target_pose = self.tf_to_grasp(target_pose)
        substeps = []
        pre_grasp_pose = list(actor_pos + [0, 0, 0.2]) + grasp_qpose
        pre_grasp_pose = self.tf_to_grasp(pre_grasp_pose)
        if actor_pos[1] < 0:
            now_arm = 'right'
            if now_arm == last_arm or last_arm is None:
                if now_arm == last_arm:
                    pose0 = list(self.robot.right_ee_link.get_entity_pose().p + [0, 0, 0.05]) + list(self.robot.right_ee_link.get_entity_pose().q)
                    substeps.append(("move_to_pose", {"right_pose": pose0}))
                substeps.append(("move_to_pose", {"right_pose": deepcopy(pre_grasp_pose)}))
            else:
                substeps.append(("move_to_pose", {"right_pose": pre_grasp_pose, "left_pose": self.robot.left_init_ee_pose}))
        else:
            now_arm = 'left'
            if now_arm == last_arm or last_arm is None:
                if now_arm == last_arm:
                    pose0 = list(self.robot.left_ee_link.get_entity_pose().p + [0, 0, 0.05]) + list(self.robot.left_ee_link.get_entity_pose().q)
                    substeps.append(("move_to_pose", {"left_pose": pose0}))
                substeps.append(("move_to_pose", {"left_pose": deepcopy(pre_grasp_pose)}))
            else:
                substeps.append(("move_to_pose", {"left_pose": pre_grasp_pose, "right_pose": self.robot.right_init_ee_pose}))
                
        substeps.append(("open_gripper", {"action_mode": now_arm}))
        pre_grasp_pose[2] -= 0.15
        substeps.append(("move_to_pose", {f"{now_arm}_pose": deepcopy(pre_grasp_pose)}))
        substeps.append(("close_gripper", {"action_mode": now_arm}))
        pre_grasp_pose[2] += 0.15
        substeps.append(("move_to_pose", {f"{now_arm}_pose": deepcopy(pre_grasp_pose)}))
        substeps.append(("move_to_pose", {f"{now_arm}_pose": deepcopy(target_pose)}))
        target_pose[2] -= 0.05
        substeps.append(("move_to_pose", {f"{now_arm}_pose": deepcopy(target_pose)}))
        substeps.append(("open_gripper", {"action_mode": now_arm}))
        target_pose[2] += 0.1
        substeps.append(("move_to_pose", {f"{now_arm}_pose": deepcopy(target_pose)}))
            
        return substeps, now_arm

    def solution(self):
        substeps, last_arm = self.move_block(self.block1, 1)
        self.info = f"move block 1,{self.block1.get_pose().p}"
        for substep in substeps:
            yield substep
        substeps, last_arm = self.move_block(self.block2, 2, last_arm)
        self.info = f"move block 2,{self.block2.get_pose().p}"
        for substep in substeps:
            yield substep
    
    def _get_info(self):
        block1_pose = self.block1.get_pose().p
        block2_pose = self.block2.get_pose().p
        target_pose = [self.tabletop_center_in_world[0] + self.place_pos_x_offset, 0]
        eps = [0.025, 0.025, 0.01]
        success = np.all(np.abs(block1_pose - np.array(target_pose + [self.table_height + self.block_half_size])) < eps) and \
               np.all(np.abs(block2_pose - np.array(target_pose + [self.table_height + self.block_half_size * 3])) < eps)
        return dict(
            success=success,
        )
    
    def _get_reset_info(self):
        return dict(
            block1_pose=np.concatenate([self.block1.get_pose().p, self.block1.get_pose().q]),
            block2_pose=np.concatenate([self.block2.get_pose().p, self.block2.get_pose().q]),
        )
        
    def _check_termination(self) -> bool:
        return bool(self._get_info()["success"])
    
    @property
    def language_instruction(self):
        return "stack the blocks on the table."
    
    def get_object_dict(self):
        return dict(
            block1=np.concatenate([self.block1.get_pose().p, self.block1.get_pose().q]),
            block2=np.concatenate([self.block2.get_pose().p, self.block2.get_pose().q]),
        )
        
    def _get_reward(self):
        return 1.0 if self._get_info()["success"] else 0.0