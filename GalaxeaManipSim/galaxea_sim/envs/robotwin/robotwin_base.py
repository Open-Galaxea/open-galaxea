import sapien
import numpy as np

from typing import Literal

from galaxea_sim.envs.base.bimanual_manipulation import BimanualManipulationEnv
from galaxea_sim.utils.robotwin_utils import create_table, create_box

class RoboTwinBaseEnv(BimanualManipulationEnv):
    def __init__(
        self,
        robot_class,
        robot_kwargs: dict = {},
        controller_type: str = "bimanual_joint_position",
        control_freq: int = 15,
        timestep: float = 0.01,
        headless: bool = True,
        obs_mode: Literal["state", "image"] = "image",  
        variant_idx: int = 0,
        ray_tracing: bool = False,
    ):
        self.variant_idx = variant_idx
        super().__init__(
            robot_class,
            robot_kwargs,
            controller_type,
            control_freq,
            timestep,
            headless,
            obs_mode,
            ray_tracing
        )
    
    def _build_world(self):
        super()._build_world()
        self._setup_table()
        self._setup_wall()
        self.reset_world()

    @property
    def table_length(self):
        return 1.2
    
    @property
    def table_width(self):
        return 0.7
    
    @property
    def table_height(self):
        if self.robot_name == "r1_lite":
            return 0.7
        else:
            return 0.9
        
    @property
    def tabletop_center_x(self):
        if self.robot_name == "r1_lite":
            return 0.5
        else:
            return 0.7

    def _setup_table(self):
        self.table_static = True
        self.tabletop_center_in_world = np.array([self.tabletop_center_x, 0, self.table_height])
        self.table = create_table(
            self._scene,
            sapien.Pose(p=self.tabletop_center_in_world),
            length=self.table_length,
            width=self.table_width,
            height=self.table_height,
            thickness=0.05,
            is_static=self.table_static
        )

    def _setup_wall(self):
        self.wall = create_box(
            self._scene,
            sapien.Pose(p=[1.0 + self.tabletop_center_in_world[0], 0, 1.5]),
            half_size=[0.6, 3, 1.5],
            color=(1, 0.9, 0.9), 
            name='wall',
        )
        
    def _add_light(self):
        self._scene.set_ambient_light([0.5, 0.5, 0.5])
        shadow = True
        direction_lights = [[[0.5 + 0.7, 0, -1], [0.5, 0.5, 0.5]]]
        for direction_light in direction_lights:
            self._scene.add_directional_light(
                direction_light[0], direction_light[1], shadow=shadow
            )
        point_lights = [[[0.7, 1, 1.8], [1, 1, 1]], [[0.7, -1, 1.8], [1, 1, 1]]]
        for point_light in point_lights:
            self._scene.add_point_light(point_light[0], point_light[1], shadow=shadow)