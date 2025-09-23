import gymnasium as gym
import numpy as np

def recursive_get_observation_space(obs, path):
    """
    This function takes in an observation and returns the observation space.
    """
    if isinstance(obs, dict):
        observation_space = gym.spaces.Dict()
        for key, value in obs.items():
            observation_space[key] = recursive_get_observation_space(value, path + "/" + key)
        return observation_space
    elif isinstance(obs, np.ndarray):
        # check if the array is unsigned
        if np.issubdtype(obs.dtype, np.unsignedinteger):
            return gym.spaces.Box(low=0, high=np.iinfo(obs.dtype).max, shape=obs.shape, dtype=obs.dtype)
        else:
            return gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape, dtype=obs.dtype)
    elif isinstance(obs, str):
        return gym.spaces.Text(max_length=max(len(obs), 1))
    else:
        raise ValueError("Unsupported observation type: {}".format(type(obs)))

def get_observation_space_from_example(obs):
    return recursive_get_observation_space(obs, "")