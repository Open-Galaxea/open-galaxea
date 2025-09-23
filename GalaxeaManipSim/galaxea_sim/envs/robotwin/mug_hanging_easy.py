from copy import deepcopy

import sapien
import numpy as np

from galaxea_sim.utils.robotwin_utils import rand_create_glb, create_obj, get_actor_goal_pose, get_grasp_pose_w_labeled_direction, get_target_pose_from_goal_point_and_direction
from galaxea_sim.utils.rand_utils import rand_pose
from .robotwin_base import RoboTwinBaseEnv

class MugHangingEasyEnv(RoboTwinBaseEnv):
    def _setup_rack(self):
        rack_pose = rand_pose(
            xlim=[0.1, 0.1], 
            ylim=[-0.2, -0.2],
            zlim=[0.005],
            rotate_rand=False,
            qpos=[0.31906068, 0.31906068, 0.63103113, 0.63103113]
        )
        self.rack, self.rack_data = create_obj(
            self._scene,
            pose=rack_pose,
            modelname="040_rack",
            is_static=True,
            convex=True,
            tabeltop_center_in_world=self.tabletop_center_in_world,
        )
        
    @property
    def table_height(self):
        if self.robot_name == "r1_lite":
            return 0.65
        else:
            return 0.88
        
    @property
    def id_list(self):
        return [2]

    def _setup_mug(self, mug_id=None):
        if mug_id is None:
            mug_id = np.random.choice(self.id_list)
        self._mug_id = int(mug_id)
        self.mug, self.mug_data = rand_create_glb(
            self._scene,
            xlim=[-0.05, -0.1],
            ylim=[0.15, 0.25],
            zlim=[0.05],
            rotate_rand=True,
            rotate_lim=[0, 1.57, 0],
            qpos=[0.5, 0.5, -0.5, -0.5],
            modelname="039_mug",
            tabeltop_center_in_world=self.tabletop_center_in_world,
            model_id=self._mug_id,
            convex=True,
        )

        coaster_pose = rand_pose(
            xlim=[-0.125, -0.125],
            ylim=[0, 0],
            zlim=[0.01],
            rotate_rand=False,
            qpos=[0.707, 0.707, 0, 0],
        )
        self.coaster, _ = create_obj(
            self._scene,
            pose=coaster_pose,
            modelname="019_coaster",
            convex=True,
            scale=[0.055 * 1.5, 0.055 / 2, 0.055 * 1.5],
            is_static=True,
            tabeltop_center_in_world=self.tabletop_center_in_world,
        )

    def reset_world(self, reset_info=None):
        if hasattr(self, "rack"):
            self._scene.remove_actor(self.rack)
        if hasattr(self, "mug"):
            self._scene.remove_actor(self.mug)
        if hasattr(self, "coaster"):
            self._scene.remove_actor(self.coaster)
        self._setup_rack()
        self._setup_mug(reset_info["mug_id"] if reset_info is not None else None)
        if reset_info is not None:
            self.mug.set_pose(sapien.Pose(p=reset_info["init_mug_pose"][:3], q=reset_info["init_mug_pose"][3:]))
            self.rack.set_pose(sapien.Pose(p=reset_info["init_rack_pose"][:3], q=reset_info["init_rack_pose"][3:]))

    def solution(self):
        left_pose1 = get_grasp_pose_w_labeled_direction(self.mug, self.mug_data, pre_dis=0.01)
        yield ("move_to_pose", {"left_pose": deepcopy(left_pose1)})
        yield ("open_gripper", {"action_mode": 'left', "gripper_target_state": 0.03})  
        left_pose1 = get_grasp_pose_w_labeled_direction(self.mug, self.mug_data, pre_dis=-0.1)
        yield ("move_to_pose", {"left_pose": deepcopy(left_pose1)})
        yield ("close_gripper", {"action_mode": 'left'})
        left_pose1[2] += 0.1
        yield ("move_to_pose", {"left_pose": deepcopy(left_pose1)})

        left_pose2 = [
            self.tabletop_center_in_world[0]-0.15, self.tabletop_center_in_world[1]-0.05, left_pose1[2], 
            -0.686497, -0.159075, -0.692153,  0.154833
        ]
        yield ("move_to_pose", {"left_pose": deepcopy(left_pose2)})
        left_pose2[2] -= 0.1
        yield ("move_to_pose", {"left_pose": deepcopy(left_pose2)})
        yield ("open_gripper", {"action_mode": 'left', "gripper_target_state": 0.03})
        left_pose2[2] += 0.1
        yield ("move_to_pose", {"left_pose": deepcopy(left_pose2)})
        left_pose2[1] += 0.2
        yield ("move_to_pose", {"left_pose": deepcopy(left_pose2)})

        right_pose1 = get_grasp_pose_w_labeled_direction(
            self.mug, self.mug_data, 
            grasp_matrix=np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]]), 
            pre_dis=0.02,
            id=1
        )
        yield ("move_to_pose", {"right_pose": deepcopy(right_pose1), "left_pose": deepcopy(self.robot.left_init_ee_pose)})
        yield ("open_gripper", {"action_mode": 'right', "gripper_target_state": 0.03})
        right_pose1 = get_grasp_pose_w_labeled_direction(
            self.mug, self.mug_data, 
            grasp_matrix=np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]]), 
            pre_dis=-0.1,
            id=1
        )
        yield ("move_to_pose", {"right_pose": deepcopy(right_pose1)})
        yield ("close_gripper", {"action_mode": 'right'})
        right_pose1[2] += 0.1
        yield ("move_to_pose", {"right_pose": deepcopy(right_pose1)})

        target_pose_p = get_actor_goal_pose(self.rack, self.rack_data) + np.array([-0.03, 0, -0.13])
        target_pose_q = [-0.84473563, -0.40150601, -0.15154333, -0.31929181]
        right_target_pose = get_target_pose_from_goal_point_and_direction(
            self.mug, self.mug_data, self.robot.right_ee_link, target_pose_p, target_pose_q
        )
        yield ("move_to_pose", {"right_pose": deepcopy(right_target_pose)})
        right_target_pose[0] += 0.03
        right_target_pose[1] -= 0.03
        right_target_pose[2] -= 0.06
        yield ("move_to_pose", {"right_pose": deepcopy(right_target_pose)})
        yield ("open_gripper", {"action_mode": 'right'})
        yield ("move_to_pose", {"right_pose": deepcopy(self.robot.right_init_ee_pose)})

    def _get_info(self):
        mug_target_pose = get_actor_goal_pose(self.mug, self.mug_data)
        eps = np.array([0.03, 0.03, 0.03])
        success = np.all(abs(mug_target_pose - self.rack.get_pose().p + [0.02, -0.02, -0.1]) < eps)
        return dict(
            success=success
        )

    def _get_reset_info(self):
        return dict(
            init_mug_pose=np.concatenate([self.mug.get_pose().p, self.mug.get_pose().q]),
            init_rack_pose=np.concatenate([self.rack.get_pose().p, self.rack.get_pose().q]),
            mug_id=self._mug_id,
        )

    def _check_termination(self) -> bool:
        return bool(self._get_info()["success"])

    def get_object_dict(self):
        return dict(
            mug=np.concatenate([self.mug.get_pose().p, self.mug.get_pose().q]),
            rack=np.concatenate([self.rack.get_pose().p, self.rack.get_pose().q]),
            mug_id=np.array([id == self._mug_id for id in self.id_list], dtype=np.float32),
        )

    def _get_reward(self):
        return 1.0 if self._get_info()["success"] else 0.0

    @property
    def language_instruction(self):
        return "hang the mug on the rack" 

if __name__ == "__main__":
    from galaxea_sim.planners.bimanual import BimanualPlanner
    from galaxea_sim.robots.r1 import R1Robot

    env = MugHangingEasyEnv(
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