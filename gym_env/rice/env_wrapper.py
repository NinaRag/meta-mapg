import os
import sys
import numpy as np
from gym.spaces import Box, Dict
from ray.rllib import MultiAgentEnv
from fixed_paths import PUBLIC_REPO_DIR
from run_unittests import import_class_from_path

sys.path.append(PUBLIC_REPO_DIR)


class EnvWrapper(MultiAgentEnv):
    """
    The environment wrapper class.
    """

    def __init__(self, env_config=None):
        env_config_copy = env_config.copy()
        if env_config_copy is None:
            env_config_copy = {}
        source_dir = env_config_copy.get("source_dir", None)
        # Remove source_dir key in env_config if it exists
        if "source_dir" in env_config_copy:
            del env_config_copy["source_dir"]
        if source_dir is None:
            source_dir = PUBLIC_REPO_DIR
        assert isinstance(env_config_copy, dict)
        env_file = "rice.py"
        if env_config["simple_game_instead_of_RICE"] is True:
            env_file = "simple_game.py"

        self.env = import_class_from_path("Rice", os.path.join(source_dir, env_file))(
            **env_config_copy
        )

        self.action_space = self.env.action_space

        self.observation_space = recursive_obs_dict_to_spaces_dict(self.env.reset())

    def get_env(self):
        return self.env

    def reset(self):
        """Reset the env."""
        obs = self.env.reset()
        return recursive_list_to_np_array(obs)

    def step(self, actions=None):
        """Step through the env."""
        assert actions is not None
        assert isinstance(actions, dict)
        obs, rew, done, info = self.env.step(actions)
        return recursive_list_to_np_array(obs), rew, done, info


def recursive_list_to_np_array(dictionary):
    """
    Numpy-ify dictionary object to be used with RLlib.
    """
    if isinstance(dictionary, dict):
        new_d = {}
        for key, val in dictionary.items():
            if isinstance(val, list):
                new_d[key] = np.array(val)
            elif isinstance(val, dict):
                new_d[key] = recursive_list_to_np_array(val)
            elif isinstance(val, (int, np.integer, float, np.floating)):
                new_d[key] = np.array([val])
            elif isinstance(val, np.ndarray):
                new_d[key] = val
            else:
                raise AssertionError
        return new_d
    raise AssertionError


_BIG_NUMBER = 1e20


def recursive_obs_dict_to_spaces_dict(obs):
    """Recursively return the observation space dictionary
    for a dictionary of observations
    Args:
        obs (dict): A dictionary of observations keyed by agent index
        for a multi-agent environment
    Returns:
        spaces.Dict: A dictionary of observation spaces
    """
    assert isinstance(obs, dict)
    dict_of_spaces = {}
    for key, val in obs.items():

        # list of lists are 'listified' np arrays
        _val = val
        if isinstance(val, list):
            _val = np.array(val)
        elif isinstance(val, (int, np.integer, float, np.floating)):
            _val = np.array([val])

        # assign Space
        if isinstance(_val, np.ndarray):
            large_num = float(_BIG_NUMBER)
            box = Box(
                low=-large_num, high=large_num, shape=_val.shape, dtype=_val.dtype
            )
            low_high_valid = (box.low < 0).all() and (box.high > 0).all()

            # This loop avoids issues with overflow to make sure low/high are good.
            while not low_high_valid:
                large_num = large_num // 2
                box = Box(
                    low=-large_num, high=large_num, shape=_val.shape, dtype=_val.dtype
                )
                low_high_valid = (box.low < 0).all() and (box.high > 0).all()

            dict_of_spaces[key] = box

        elif isinstance(_val, dict):
            dict_of_spaces[key] = recursive_obs_dict_to_spaces_dict(_val)
        else:
            raise TypeError
    return Dict(dict_of_spaces)