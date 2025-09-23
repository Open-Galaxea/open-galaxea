import sapien.core as sapien
import numpy as np
import gymnasium as gym
from gymnasium.utils import seeding

from galaxea_sim.utils.sapien_utils import add_camera_to_scene


class SapienEnv(gym.Env):
    def __init__(self, control_freq, timestep, headless, ray_tracing):
        if ray_tracing:
            sapien.render.set_camera_shader_dir("rt")
            sapien.render.set_viewer_shader_dir("rt")
            sapien.render.set_ray_tracing_samples_per_pixel(4)  # change to 256 for less noise
            sapien.render.set_ray_tracing_denoiser("oidn") # change to "optix" or "oidn"

        self.control_freq = control_freq  # alias: frame_skip in mujoco_py
        self.timestep = timestep
        self.headless = headless

        self._scene = sapien.Scene()
        self._scene.set_timestep(timestep)

        self._build_world()
        self.viewer = None
        self.seed()
        self._add_light()
        self._add_scene_camera()

    @property
    def cameras(self) -> list[sapien.pysapien.render.RenderCameraComponent]:
        return []

    def _add_light(self):
        self._scene.set_ambient_light([0.5, 0.5, 0.5])
        shadow = True
        direction_lights = [[[0, 1, -1], [0.5, 0.5, 0.5]]]
        for direction_light in direction_lights:
            self._scene.add_directional_light(
                direction_light[0], direction_light[1], shadow=shadow
            )
            
        point_lights = [[[1, 2, 2], [1, 1, 1]], [[1, -2, 2], [1, 1, 1]], [[-1, 0, 1], [1, 1, 1]]]
        for point_light in point_lights:
            self._scene.add_point_light(point_light[0], point_light[1], shadow=shadow)

    def _build_world(self):
        raise NotImplementedError()

    def _setup_viewer(self):
        raise NotImplementedError()

    # ---------------------------------------------------------------------------- #
    # Override gym functions
    # ---------------------------------------------------------------------------- #
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        if self.viewer is not None:
            pass  # release viewer
        
    def _add_scene_camera(self):
        pose = sapien.Pose()
        pose.set_p([1.8, -0.9, 1.])
        pose.set_rpy([0, 0, 2.64])
        self.default_camera = add_camera_to_scene(self._scene, "default_camera", pose=pose)

    def render(self):
        self.default_camera.take_picture()
        rgba = self.default_camera.get_picture("Color")  # [H, W, 4]
        rgb_img = (rgba * 255).clip(0, 255).astype("uint8")[..., :3]       
        if not self.headless:
            if self.viewer is None:
                self._setup_viewer()
            else:
                if not self.viewer.closed:
                    self.viewer.render()
        return rgb_img

    # ---------------------------------------------------------------------------- #
    # Utilities
    # ---------------------------------------------------------------------------- #
    def get_image_dict(self):
        image_dict = {}
        for camera in self.cameras:
            camera.take_picture()
            rgba = camera.get_picture("Color")  # [H, W, 4]
            rgb_img = (rgba * 255).clip(0, 255).astype("uint8")[..., :3]
            image_dict[f"rgb_{camera.name}"] = rgb_img
            
            position = camera.get_picture('Position')  # [H, W, 4]
            depth = -position[..., 2]
            depth_image = depth * 1000.0
            depth_image = depth_image.clip(0, 256 * 256 - 1).astype("uint16")
            
            image_dict[f"depth_{camera.name}"] = depth_image
        return image_dict
    
    def get_actor(self, name):
        all_actors = self._scene.get_all_actors()
        actor = [x for x in all_actors if x.name == name]
        if len(actor) > 1:
            raise RuntimeError(f'Not a unique name for actor: {name}')
        elif len(actor) == 0:
            raise RuntimeError(f'Actor not found: {name}')
        return actor[0]

    def get_articulation(self, name):
        all_articulations = self._scene.get_all_articulations()
        articulation = [x for x in all_articulations if x.name == name]
        if len(articulation) > 1:
            raise RuntimeError(f'Not a unique name for articulation: {name}')
        elif len(articulation) == 0:
            raise RuntimeError(f'Articulation not found: {name}')
        return articulation[0]

    @property
    def dt(self):
        return 1 / self.control_freq
    
    @property
    def decimation(self):
        return int(self.dt / self.timestep)