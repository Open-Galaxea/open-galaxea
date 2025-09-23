import sapien.core as sapien

class BaseRobot:
    name: str = "base_robot"
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
    ):
        self._scene = scene
        self.urdf_path = urdf_path
        self.robot_origin_xyz = robot_origin_xyz
        self.robot_origin_quat = robot_origin_quat
        self.joint_stiffness = joint_stiffness
        self.joint_damping = joint_damping
        self.init_qpos = init_qpos
        self.touch_link_names = touch_link_names
        