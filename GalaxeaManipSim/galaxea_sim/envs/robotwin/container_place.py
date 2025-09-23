from copy import deepcopy

import sapien
import numpy as np

from galaxea_sim.utils.robotwin_utils import create_glb, get_target_pose_from_goal_point_and_direction, get_actor_goal_pose
from galaxea_sim.utils.rand_utils import rand_pose
from .robotwin_base import RoboTwinBaseEnv

class ContainerPlaceEnv(RoboTwinBaseEnv):
    def _setup_plate(self):
        self.plate, _ = create_glb(
            self._scene,
            pose=sapien.Pose(
                [self.tabletop_center_in_world[0]-0.05, self.tabletop_center_in_world[1], self.table_height + 0.013], 
                [0.5, 0.5, 0.5, 0.5]
            ),
            modelname="003_plate",
            scale=[0.025, 0.025, 0.025],
            is_static=True,
            convex=True
        )
    
    @property
    def id_list(self):
        return [8, 9]
        
    def _setup_container(self, container_id=None):
        container_pose = None
        while container_pose is None or abs(container_pose.p[1]) < 0.15:
            container_pose = rand_pose(
                ylim=[-0.3, 0.3],
                xlim=[-0.1, 0.05],
                zlim=[0.06],
                rotate_rand=False,
                qpos=list((sapien.Pose(q=[0.707, 0, 0, 0.707]) * sapien.Pose(q=[0.707, 0.707, 0, 0])).q)
            )
        if container_id is None:
            container_id = np.random.choice(self.id_list)
        self._id = int(container_id)
        self.container, self.container_data = create_glb(
            self._scene,
            pose=container_pose,
            modelname="002_container",
            model_id=self._id,
            tabeltop_center_in_world=self.tabletop_center_in_world,
            model_z_val=True,
            convex=True
        )
        self.container.find_component_by_type(sapien.physx.PhysxRigidDynamicComponent).mass = 0.1

    def reset_world(self, reset_info=None):
        # chek if red bottle on the scene, then remove it
        if hasattr(self, "plate"):
            self._scene.remove_actor(self.plate)
        if hasattr(self, "container"):
            self._scene.remove_actor(self.container)
        self._setup_plate()
        self._setup_container(reset_info["container_id"] if reset_info is not None else None)
        if reset_info is not None:
            self.plate.set_pose(sapien.Pose(p=reset_info["init_plate_pose"][:3], q=reset_info["init_plate_pose"][3:]))
            self.container.set_pose(sapien.Pose(p=reset_info["init_container_pose"][:3], q=reset_info["init_container_pose"][3:]))
        
    def solution(self):
        arm = 'right' if self.container.pose.p[1] < 0 else 'left'
        init_pose = self.robot.left_init_ee_pose if arm == 'left' else self.robot.right_init_ee_pose
        ee_link = self.robot.left_ee_link if arm == 'left' else self.robot.right_ee_link
        
        container_pose = self.container.get_pose().p
        container_edge_dis = np.array(self.container_data['extents']) * np.array(self.container_data['scale'])
        container_edge_dis = [0, -container_edge_dis[0]/2, container_edge_dis[1]/2 + 0.12]
        if arm == 'left':
            container_edge_dis[1] *= -1
        
        pose1 = (container_pose + container_edge_dis).tolist() + [1, 0, 1, 0]
        yield ("move_to_pose", {f"{arm}_pose": deepcopy(pose1)})
        yield ("open_gripper", {"action_mode": arm, "gripper_target_state": 0.03})
        pose1[2] -= 0.1
        yield ("move_to_pose", {f"{arm}_pose": deepcopy(pose1)})
        yield ("close_gripper", {"action_mode": arm})
        pose1[2] += 0.1
        yield ("move_to_pose", {f"{arm}_pose": deepcopy(pose1)})
        
        target_pose = get_target_pose_from_goal_point_and_direction(
            self.container, self.container_data, ee_link, np.array([-0.05, 0, 0.09]) + self.tabletop_center_in_world, [1, 0, 1, 0]
        )
        target_pose[1] *= -1
        yield ("move_to_pose", {f"{arm}_pose": deepcopy(target_pose)})
        target_pose[2] -= 0.1
        yield ("move_to_pose", {f"{arm}_pose": deepcopy(target_pose)})
        yield ("open_gripper", {"action_mode": arm})
        target_pose[2] += 0.075
        yield ("move_to_pose", {f"{arm}_pose": deepcopy(target_pose)})
        yield ("move_to_pose", {f"{arm}_pose": deepcopy(init_pose)})
        
    def _get_info(self):
        container_pose = get_actor_goal_pose(self.container, self.container_data)
        target_pose = np.array([-0.05, 0, 0]) + self.tabletop_center_in_world
        eps = np.array([0.02, 0.02, 0.01])
        success = np.all(np.abs(container_pose - target_pose) < eps)
        return dict(
            success=success
        )
    
    def _get_reset_info(self):
        return dict(
            init_plate_pose=np.concatenate([self.plate.get_pose().p, self.plate.get_pose().q]),
            init_container_pose=np.concatenate([self.container.get_pose().p, self.container.get_pose().q]),
            container_id=self._id
        )
        
    def _check_termination(self) -> bool:
        return bool(self._get_info()["success"])
    
    @property
    def language_instruction(self):
        return "place the container on the plate" 
    
    def get_object_dict(self):
        return dict(
            plate=np.concatenate([self.plate.get_pose().p, self.plate.get_pose().q]),
            container=np.concatenate([self.container.get_pose().p, self.container.get_pose().q]),
            container_id=np.array([id == self._id for id in range(len(self.id_list))], dtype=np.float32),
        )
        
    def _get_reward(self):
        return 1.0 if self._get_info()["success"] else 0.0