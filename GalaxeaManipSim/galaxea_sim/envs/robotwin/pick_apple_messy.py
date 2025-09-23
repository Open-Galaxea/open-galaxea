from copy import deepcopy

import sapien
import numpy as np

from galaxea_sim.utils.robotwin_utils import create_obj, create_glb, get_grasp_pose_w_given_direction
from galaxea_sim.utils.rand_utils import rand_pose
from .robotwin_base import RoboTwinBaseEnv

class PickAppleMessyEnv(RoboTwinBaseEnv):
    def _setup_apple(self):
        apple_pose = rand_pose(
            xlim=[-0.2, 0],
            ylim=[-0.25, 0.25],
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
        
    @property
    def obj_num(self):
        return 4
        
    def _setup_messy(self, obj_list=None):
        self.actor_list = []
        self.actor_data_list = []
        
        if obj_list is None:
            obj_list = np.random.choice(self.rand_model_data["number"], size=self.obj_num, replace=False)
        self._obj_list = obj_list
        pose_list = [self.apple.get_pose()]
        for i in obj_list:
            model_index = f"model{i}"
            actor_pose = None
            while actor_pose is None or any([np.linalg.norm(actor_pose.p[:2] - pose.p[:2] + self.tabletop_center_in_world[:2]) < 0.15 for pose in pose_list]):
                actor_pose = rand_pose(
                    xlim=self.rand_model_data[model_index]["xlim"],
                    ylim=self.rand_model_data[model_index]["ylim"],
                    zlim=self.rand_model_data[model_index]["zlim"],
                    rotate_rand=True,
                    rotate_lim=self.rand_model_data[model_index]["rotate_lim"],
                    qpos=self.rand_model_data[model_index]["init_qpos"]
                )
            pose_list.append(actor_pose)
            try:
                model, model_data = create_glb(
                    self._scene,
                    pose=actor_pose,
                    modelname=self.rand_model_data[model_index]["name"],
                    convex=True,
                    model_z_val=self.rand_model_data[model_index]["model_z_val"],
                    tabeltop_center_in_world=self.tabletop_center_in_world,
                )
            except:
                model, model_data = create_obj(
                    self._scene,
                    pose=actor_pose,
                    modelname=self.rand_model_data[model_index]["name"],
                    convex=True,
                    model_z_val=self.rand_model_data[model_index]["model_z_val"],
                    tabeltop_center_in_world=self.tabletop_center_in_world,
                )
            self.actor_list.append(model)
            self.actor_data_list.append(model_data)

    def reset_world(self, reset_info=None):
        if hasattr(self, "apple"):
            self._scene.remove_actor(self.apple)
        if hasattr(self, "actor_list"):
            for model in self.actor_list:
                self._scene.remove_actor(model)
        if reset_info is not None:
            obj_list = reset_info["obj_list"]
        else:
            obj_list = None
        self._setup_apple()
        self._setup_messy(obj_list)
        if reset_info is not None:
            self.apple.set_pose(sapien.Pose(p=reset_info["init_apple_pose"][:3], q=reset_info["init_apple_pose"][3:]))
            for i, model in enumerate(self.actor_list):
                model.set_pose(sapien.Pose(p=reset_info["init_actors_pose"][i][:3], q=reset_info["init_actors_pose"][i][3:]))
        
    def solution(self):
        arm = "left" if self.apple.get_pose().p[1] > 0 else "right"
        pose1 = get_grasp_pose_w_given_direction(self.apple, self.apple_data, grasp_qpos=[0.707, 0, 0.707, 0], pre_dis=0)
        pose2 = get_grasp_pose_w_given_direction(self.apple, self.apple_data, grasp_qpos=[0.707, 0, 0.707, 0], pre_dis=-0.1)
        yield ("move_to_pose", {f"{arm}_pose": deepcopy(pose1)})
        yield ("open_gripper", {"action_mode": arm})
        yield ("move_to_pose", {f"{arm}_pose": deepcopy(pose2)})
        yield ("close_gripper", {"action_mode": arm})
        pose2[2] += 0.15
        yield ("move_to_pose", {f"{arm}_pose": deepcopy(pose2)})
        
    def _get_info(self):
        success = self.apple.get_pose().p[2] > 0.07 + self.tabletop_center_in_world[2]
        return dict(
            success=success
        )
    
    def _get_reset_info(self):
        return dict(
            init_apple_pose=np.concatenate([self.apple.get_pose().p, self.apple.get_pose().q]),
            obj_list=self._obj_list,
            init_actors_pose=[np.concatenate([model.get_pose().p, model.get_pose().q]) for model in self.actor_list]
        )
        
    def _check_termination(self) -> bool:
        return bool(self._get_info()["success"])
    
    @property
    def language_instruction(self):
        return "pick the apple" 
    
    def get_object_dict(self):
        return dict(
            apple=np.concatenate([self.apple.get_pose().p, self.apple.get_pose().q]),
        )
        
    def _get_reward(self):
        return 1.0 if self._get_info()["success"] else 0.0
    
    @property
    def rand_model_data(self):
        return {
            'number': 6,
            'model0': {
                'init_qpos': np.array([-0.123725, 0.612969, 0.060095, 0.707707]),
                'model_z_val': 1,
                'name': '024_brush',
                'rotate_lim': [1.57, 0, 0],
                'rotate_rand': 1,
                'xlim': [-0.15, 0.3],
                'ylim': [-0.35, 0.35],
                'zlim': [0.06000000000000005]
            },
            'model1': {
                'init_qpos': np.array([0.499849, 0.499849, -0.499849, -0.499849]),
                'model_z_val': 1,
                'name': '022_cup',
                'rotate_lim': [0, 0, 0],
                'rotate_rand': 0,
                'xlim': [0, 0.3],
                'ylim': [-0.35, 0.35],
                'zlim': [0.06000000000000005]
            },
            'model2': {
                'init_qpos': np.array([0.499849, 0.499849, -0.499849, -0.499849]),
                'model_z_val': 1,
                'name': '019_coaster',
                'rotate_lim': [0, 0, 0],
                'rotate_rand': 0,
                'xlim': [-0.15, 0.25],
                'ylim': [-0.35, 0.35],
                'zlim': [0.030000000000000027]
            },
            'model3': {
                'init_qpos': np.array([0.31815, 0.31815, 0.62923, 0.62923]),
                'model_z_val': 0,
                'name': '040_rack',
                'rotate_lim': [0, 0, 0],
                'rotate_rand': 1,
                'xlim': [0, 0.25],
                'ylim': [-0.35, 0.35],
                'zlim': [0.06000000000000005]
            },
            'model4': {
                'init_qpos': np.array([0.695688, 0.695688, 0.124432, 0.124432]),
                'model_z_val': 1,
                'name': '028_dustpan',
                'rotate_lim': [0, 1.57, 0],
                'rotate_rand': 1,
                'xlim': [-0.125, 0.125],
                'ylim': [-0.35, 0.35],
                'zlim': [0.040000000000000036]
            },
            'model5': {
                'init_qpos': np.array([0.429149, -0.550046, -0.469448, -0.540855]),
                'model_z_val': 1,
                'name': '020_hammer_2',
                'rotate_lim': [1.57, 0, 0],
                'rotate_rand': 1,
                'xlim': [-0.15, 0.3],
                'ylim': [-0.35, 0.35],
                'zlim': [0.040000000000000036]
            }
        }