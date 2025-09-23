from copy import deepcopy

import sapien
import numpy as np

from galaxea_sim.utils.robotwin_utils import rand_create_glb, get_grasp_pose_w_given_direction, get_grasp_pose_w_labeled_direction
from .dual_bottles_pick_easy import DualBottlesPickEasyEnv

class DualBottlesPickHardEnv(DualBottlesPickEasyEnv):        
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
    
    def _get_info(self):
        # check distance between bottle and target
        left_distance = np.linalg.norm(self.bottle1.get_pose().p - self.left_target_pose.p)
        right_distance = np.linalg.norm(self.bottle2.get_pose().p - self.right_target_pose.p)
        left_height = self.bottle1.get_pose().p[2].item()
        right_height = self.bottle2.get_pose().p[2].item()
        left_success = left_distance < 0.1 and left_height >= 1.10
        right_success = right_distance < 0.1 and right_height >= 1.10
        # print(left_distance, right_distance, left_height, right_height)
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
        )
        
    def _check_termination(self) -> bool:
        return bool(self._get_info()["success"])
    
    @property
    def language_instruction(self):
        return "pick up the two bottles simultaneously"
    
    def get_object_dict(self):
        return dict(
            bottle1=np.concatenate([self.bottle1.get_pose().p, self.bottle1.get_pose().q]),
            bottle2=np.concatenate([self.bottle2.get_pose().p, self.bottle2.get_pose().q]),
        )
        
    def _get_reward(self):
        return 1.0 if self._get_info()["success"] else 0.0

if __name__ == "__main__":
    from galaxea_sim.planners.bimanual import BimanualPlanner
    from galaxea_sim.robots.r1 import R1Robot
    
    env = DualBottlesPickHardEnv(
        robot_class=R1Robot, 
        robot_kwargs=dict(
            init_qpos=[
                0.70050001, -1.40279996, -0.99959999, 0.0, 
                0, 0, 
                1.57, 1.57, 
                -0.96, -0.96,
                0, 0,
                0, 0, 
                0, 0, 
                0, 0,
                0, 0,
            ]
        ), 
        headless=False
    )
    
    planner = BimanualPlanner(
        urdf_path="r1/robot.urdf",
        srdf_path="r1/robot.srdf",
        left_arm_move_group=env.left_ee_link_name,
        right_arm_move_group=env.right_ee_link_name,
        active_joint_names=env.active_joint_names,
        control_freq=env.control_freq,
    )
    
    for substep in env.solution():
        actions = planner.solve(
            substep, env.robot.get_qpos(), env.last_gripper_cmd, 
            verbose=False
        )
        if actions is not None:
            for action in actions:
                obs, _, _, _, info = env.step(action)
                env.render()
    
    while True:
        env.render()