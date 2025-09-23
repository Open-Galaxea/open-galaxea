import numpy as np
from robot_interface import GalaxeaInterface

class Wrapper:
    def __init__(self, interface: GalaxeaInterface):
        self.interface = interface

    def step(self, action):
        action_dict = {}
        if action.shape[0] == (6 + 1) * 2 + 6 + 6:
            action_dict = dict(
                left_arm=action[:6],
                left_gripper=action[6:7],
                right_arm=action[7:13],
                right_gripper=action[13:14],
                torso=action[14:20],
                chassis=action[20:],
            )
        else:
            raise NotImplementedError
        obs = self.interface.step(action_dict)
        return self.wrap_obs(obs)

    def wrap_obs(self, obs):
        if obs is None: return obs
        if 'torso' not in obs:
            obs['torso'] = dict(position=np.zeros(4, dtype=np.float32))
        if 'chassis' not in obs:
            obs['chassis'] = dict(position=np.zeros(3, dtype=np.float32))
        4 + 3 + 7 * 2 + 26
        obs_dict = {}
        for s, t in zip(
            ["head_rgb", "left_hand_rgb", "right_hand_rgb"],
            ["head_rgb", "left_rgb", "right_rgb"],
        ):
            obs_dict[s] = obs[t]["data"]
        
        for s, t in zip(
            [
                "/hdas/feedback_gripper_left", "/hdas/feedback_gripper_right",
                "/hdas/feedback_torso", "/hdas/feedback_chassis"
            ],
            ["left_gripper", "right_gripper", "torso", "chassis"]
        ):
            obs_dict[s] = obs[t]["position"]    

        for s, t in zip(
            [
                "/hdas/feedback_arm_left",
                "/hdas/feedback_arm_right", 
            ],
            ["left_arm", "right_arm"]
        ):
            obs_dict[s + "/position"] = obs[t]["position"][:-1]
            obs_dict[s + "/velocity"] = obs[t]["velocity"][:-1]
        return obs_dict
    
    def get_observations(self):
        obs = self.interface.get_observations()
        return self.wrap_obs(obs)
    
    def is_close(self):
        return self.interface.is_close()

    def get_latest_instruction(self):
        return self.interface.get_latest_instruction()
    
if __name__ == "__main__":
    import time
    from robot_interface import GalaxeaInterface, GalaxeaInferfaceConfig
    interface = GalaxeaInterface(GalaxeaInferfaceConfig())
    interface = Wrapper(interface)
    time.sleep(1)
    for i in range(10):
        time.sleep(0.1)
        obs = interface.get_observations()
        for k, v in obs.items():
            print(k, v.shape)

