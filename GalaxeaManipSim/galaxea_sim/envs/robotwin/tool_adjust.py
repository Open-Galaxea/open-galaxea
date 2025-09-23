import sapien
import numpy as np

from galaxea_sim.utils.robotwin_utils import rand_create_glb, get_grasp_pose_w_given_direction
from .robotwin_base import RoboTwinBaseEnv

class ToolAdjustEnv(RoboTwinBaseEnv):
    @property
    def table_height(self):
        if self.robot_name == "r1_lite":
            return 0.6
        else:
            return super().table_height
    
    def _setup_object(self, object_id=None):
        if object_id is None:
            object_id = np.random.choice(self.id_list)
        self._object_id = int(object_id)
        self.hand_tag = np.random.randint(0,2) 
        qpos_list = [[[0.077,0.703,-0.703,0.077],[0.707,0,0,0.707],[-0.465,-0.533,-0.471,0.527],[0,0.707,-0.707,0],
                      [-0.5,-0.5,0.5,0.5],[0.5,-0.5,0.5,0.5],[0.5,0.5,0.5,0.5],[0.5,0.5,-0.5,0.5],[0.707,0,0,0.707], [0.707,0,0,0.707]],
                     [[0.077,0.703,0.703,0.077],[0.707,0,0,-0.707],[-0.465,0.533,-0.471,-0.527],[0,0.707,0.707,0],
                      [0.5,0.5,0.5,0.5],[0.5,0.5,0.5,-0.5],[-0.5,-0.5,0.5,0.5],[-0.5,0.5,0.5,0.5],[0.707,0,0,-0.707], [0.707,0,0,-0.707]]]
        zlim_list = [0.785, 0.78, 0.79, 0.79, 0.79, 0.79,0.79,0.78, 0.769, 0.77]
        qpos = qpos_list[self.hand_tag][self._object_id]
        now_qpos = (sapien.Pose(q=[0.707, 0, 0, -0.707]) * sapien.Pose(q=qpos)).q.tolist()
        zlim = zlim_list[self._object_id] - 0.74
        self.object, self.object_data = rand_create_glb(
            self._scene,
            xlim=[-0.2, -0.1],
            ylim=[0., 0.],
            zlim=[zlim],
            rotate_rand=False,
            qpos=now_qpos,
            modelname="tools",
            convex=True,
            model_id=self._object_id,
            tabeltop_center_in_world=self.tabletop_center_in_world,
        )
    
    @property
    def id_list(self):
        if self.eval_mode:
            return [1, 4, 7, 8]
        else:
            return [3, 5]
        
    def reset_world(self, reset_info=None):
        if hasattr(self, "object"):
            self._scene.remove_actor(self.object)
        self._setup_object(reset_info["object_id"] if reset_info is not None else None)
        if reset_info is not None:
            self.object.set_pose(sapien.Pose(reset_info["object_pose"][:3], reset_info["object_pose"][3:]))
        
    def solution(self):
        arm = "left" if self.hand_tag == 0 else "right"
        grasp_qpos = [0.5, -0.5, 0.5, 0.5] if arm == "right" else [0.5, 0.5, 0.5, -0.5]
        pre_pose = get_grasp_pose_w_given_direction(self.object, self.object_data, grasp_qpos=grasp_qpos, pre_dis=0.05)
        yield ("move_to_pose", {f"{arm}_pose": pre_pose})
        yield ("open_gripper", {"action_mode": arm, "steps": 30})
        grasp_pose = get_grasp_pose_w_given_direction(self.object, self.object_data, grasp_qpos=grasp_qpos, pre_dis=-0.09)
        yield ("move_to_pose", {f"{arm}_pose": grasp_pose})
        yield ("close_gripper", {"action_mode": arm})
        grasp_pose[2] += 0.2
        yield ("move_to_pose", {f"{arm}_pose": grasp_pose})
        
    def _get_info(self):
        is_pick = self.object.get_pose().p[2] > self.table_height + 0.1
        if self.hand_tag == 0:
            is_correct_hand = np.all(np.abs(self.robot.left_ee_link.get_entity_pose().p[:2] - self.object.get_pose().p[:2]) < 0.05)
        else:
            is_correct_hand = np.all(np.abs(self.robot.right_ee_link.get_entity_pose().p[:2] - self.object.get_pose().p[:2]) < 0.05)
        return dict(
            success=is_pick and is_correct_hand,
        )
    
    def _get_reset_info(self):
        return dict(
            object_pose=np.concatenate([self.object.get_pose().p, self.object.get_pose().q], axis=0),
            object_id=self._object_id,
        )
        
    def _check_termination(self) -> bool:
        return bool(self._get_info()["success"])
    
    @property
    def language_instruction(self):
        return "pick up the tool"
    
    def get_object_dict(self):
        return dict(
            object=np.concatenate([self.object.get_pose().p, self.object.get_pose().q], axis=0),
            object_id=np.array([id == self._object_id for id in self.id_list], dtype=np.float32),
        )
        
    def _get_reward(self):
        return 1.0 if self._get_info()["success"] else 0.0