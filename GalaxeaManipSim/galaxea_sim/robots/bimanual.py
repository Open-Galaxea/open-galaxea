import numpy as np
import sapien
import yaml
from loguru import logger
from typing import Optional, List

from galaxea_sim import ASSETS_DIR
from galaxea_sim.utils.sapien_utils import get_joint_indices_by_key, get_link_by_name

from .base import BaseRobot

class BimanualRobot(BaseRobot):
    name: str = "bimanual_robot"
    def __init__(
        self, 
        scene: sapien.Scene,
        urdf_path: str,
        robot_origin_xyz: list,
        robot_origin_quat: list,
        joint_stiffness: float,
        joint_damping: float, 
        init_qpos: list,
        touch_link_names: list, 
        left_arm_joint_key: str,
        right_arm_joint_key: str,
        torso_joint_key: str,
        left_gripper_joint_key: str,
        right_gripper_joint_key: str,
        left_ee_link_name: Optional[str] = None,
        right_ee_link_name: Optional[str] = None,
        left_control_frame_name: Optional[str] = None,
        right_control_frame_name: Optional[str] = None,
        left_relaxed_ik_setting_path: Optional[str] = None,
        right_relaxed_ik_setting_path: Optional[str] = None,
    ):
        super().__init__(
            scene,
            urdf_path,
            robot_origin_xyz,
            robot_origin_quat,
            joint_stiffness,
            joint_damping,
            init_qpos,
            touch_link_names,
        )
        self.left_arm_joint_key = left_arm_joint_key
        self.right_arm_joint_key = right_arm_joint_key
        self.torso_joint_key = torso_joint_key
        self.left_gripper_joint_key = left_gripper_joint_key
        self.right_gripper_joint_key = right_gripper_joint_key
        self.left_ee_link_name = left_ee_link_name
        self.right_ee_link_name = right_ee_link_name
        self.left_control_frame_name = left_control_frame_name
        self.right_control_frame_name = right_control_frame_name
        self.left_relaxed_ik_setting_path = left_relaxed_ik_setting_path
        self.right_relaxed_ik_setting_path = right_relaxed_ik_setting_path
        self._sync_relaxed_ik_config()
        self._load_robot()
        
    def _sync_relaxed_ik_config(self):
        if self.left_relaxed_ik_setting_path is not None:
            with open(ASSETS_DIR / self.left_relaxed_ik_setting_path, "r") as f:
                self.left_relaxed_ik_config = yaml.safe_load(f)
            self.left_ee_link_name = self.left_relaxed_ik_config["ee_links"][0]   
            self.left_control_frame_name = self.left_relaxed_ik_config["base_links"][0]
        if self.right_relaxed_ik_setting_path is not None:
            with open(ASSETS_DIR / self.right_relaxed_ik_setting_path, "r") as f:
                self.right_relaxed_ik_config = yaml.safe_load(f)
            self.right_ee_link_name = self.right_relaxed_ik_config["ee_links"][0]
            self.right_control_frame_name = self.right_relaxed_ik_config["base_links"][0]
        logger.debug(f"Left EE link name: {self.left_ee_link_name}")
        logger.debug(f"Right EE link name: {self.right_ee_link_name}")
        logger.debug(f"Left control frame name: {self.left_control_frame_name}")
        logger.debug(f"Right control frame name: {self.right_control_frame_name}")
    
    def _load_robot(self):
        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = True
        self.robot: sapien.pysapien.physx.PhysxArticulation = loader.load(
            (ASSETS_DIR / self.urdf_path).as_posix()
        )
        self.robot.set_root_pose(
            sapien.Pose(
                self.robot_origin_xyz,
                self.robot_origin_quat,
            )
        )
        if self.init_qpos is None:
            self.init_qpos = [0] * self.robot.dof
        else:
            self.init_qpos = self.init_qpos
        logger.debug(f"Initial qpos: {self.init_qpos}")
        self.robot.set_qpos(self.init_qpos)
        self.active_joints = self.robot.get_active_joints()
        self.active_joint_names = [joint.get_name() for joint in self.active_joints]
        self.links = self.robot.get_links()
        self.left_arm_joint_indices = get_joint_indices_by_key(self.active_joints, self.left_arm_joint_key)
        self.right_arm_joint_indices = get_joint_indices_by_key(self.active_joints, self.right_arm_joint_key)
        self.left_gripper_joint_indices = get_joint_indices_by_key(self.active_joints, self.left_gripper_joint_key)
        self.right_gripper_joint_indices = get_joint_indices_by_key(self.active_joints, self.right_gripper_joint_key)
        self.torso_joint_indices = get_joint_indices_by_key(self.active_joints, self.torso_joint_key)
        self.left_ee_link = get_link_by_name(self.links, self.left_ee_link_name)
        self.right_ee_link = get_link_by_name(self.links, self.right_ee_link_name)
        self.left_init_ee_pose = self.left_ee_link.get_entity_pose()
        self.right_init_ee_pose = self.right_ee_link.get_entity_pose()
        self.touch_links = [get_link_by_name(self.links, name) for name in self.touch_link_names]
        self.left_control_frame_link = get_link_by_name(self.links, self.left_control_frame_name)
        self.right_control_frame_link = get_link_by_name(self.links, self.right_control_frame_name)
        
        logger.debug(f"Active joints: {len(self.active_joints)}")
        logger.debug(f"Left arm joint indices: {self.left_arm_joint_indices}")
        logger.debug(f"Right arm joint indices: {self.right_arm_joint_indices}")
        logger.debug(f"Left gripper joint indices: {self.left_gripper_joint_indices}")
        logger.debug(f"Right gripper joint indices: {self.right_gripper_joint_indices}")
        
        self.num_dofs = len(self.active_joints)
        for joint in self.active_joints:
            joint.set_drive_property(
                stiffness=self.joint_stiffness,
                damping=self.joint_damping,
            )
            
        self.left_arm_joint_position_cmd = np.zeros(len(self.left_arm_joint_indices))
        self.right_arm_joint_position_cmd = np.zeros(len(self.right_arm_joint_indices))
        self.left_arm_gripper_position_cmd = 0.
        self.right_arm_gripper_position_cmd = 0.
        self.last_gripper_cmd = [0, 0]
        
        self._set_touch_links_material()
        self._add_sensors()

    def _set_touch_links_material(self):
        for link in self.touch_links:
            for component in link.entity.components:
                if isinstance(component, sapien.pysapien.physx.PhysxArticulationLinkComponent):
                    for collision_shape in component.get_collision_shapes():
                        collision_shape.set_physical_material(
                            self._scene.create_physical_material(1.0, 1.0, 0.6)
                        )
                        
    def _add_sensors(self):
        pass
    
    def get_qpos(self):
        return self.robot.get_qpos()
    
    def set_qpos(self, qpos):
        self.robot.set_qpos(qpos)
        
    def get_pose(self):
        return self.robot.get_root_pose()
    
    def get_qvel(self):
        return self.robot.get_qvel()
    
    def compute_passive_force(self, gravity: bool = True, coriolis_and_centrifugal: bool = True):
        return self.robot.compute_passive_force(gravity, coriolis_and_centrifugal)
        
    def set_qf(self, qf):
        self.robot.set_qf(qf)
        
    @property
    def left_ee_pose_wrt_control_frame(self):
        return self.left_control_frame_link.get_entity_pose().inv() * self.left_ee_link.get_entity_pose()
    
    @property
    def right_ee_pose_wrt_control_frame(self):
        return self.right_control_frame_link.get_entity_pose().inv() * self.right_ee_link.get_entity_pose()
    
    
    @property
    def cameras(self):
        return []
    
    @property
    def articulation(self):
        return self.robot
    
    @property
    def left_ee_rpy_offset(self):
        return np.array([0, 0, 0])
    
    @property
    def right_ee_rpy_offset(self):
        return np.array([0, 0, 0])
    
    @property
    def gripper_finger_sign(self):
        return [1, 1]