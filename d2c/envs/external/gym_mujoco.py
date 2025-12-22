"""An implementation of the Env for D4RL benchmark."""

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Space
from typing import Tuple, Any, Union
from d2c.envs import BaseEnv
from d2c.utils.wrappers import wrapped_norm_obs_env
from d2c.networks_and_utils_for_agent.ppo_nets_utils import ppo_make_env
from d2c.networks_and_utils_for_agent.sac_nets_utils import sac_make_env

def make_env_bridge(env_id, model_name='sac', seed=0, idx=0, capture_video=False, run_name="", gamma=0.99):
    if model_name == 'sac':
        return sac_make_env(env_id, seed, idx, capture_video, run_name)
    if model_name == 'ppo':
        return ppo_make_env(env_id, idx, capture_video, run_name, gamma)
    return sac_make_env(env_id, seed, idx, capture_video, run_name)

class GymEnv(BaseEnv):
    """The Env for Gym benchmark.

    :param str env_name: the name of env.
    """

    def __init__(
            self,
            env_name: str,
            model_name: str,
    ) -> None:
        self._env_name = env_name
        self._model_name = model_name
        self._load_model()
        super(GymEnv, self).__init__()

    def _load_model(self):
        gym_envs = gym.vector.SyncVectorEnv([make_env_bridge(self._env_name, self._model_name, 0, 0, False, "", 0.99)])
        self._env = gym_envs

    def _set_action_space(self) -> Space:
        self.action_space = self._env.action_space
        return self.action_space

    def _set_observation_space(self) -> Space:
        self.observation_space = self._env.observation_space
        return self.observation_space

    def step(self, a: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        return self._env.step(a)

    def reset(self, **kwargs: Any) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
        return self._env.reset(**kwargs)

    @staticmethod
    def make_env_space(data_source=None, env_name=None):
        """Return an object with `.observation` and `.action` spaces for given env_name.

        This is used by `ConfigBuilder._get_env_space` which expects
        `environment_space.observation` and `environment_space.action`.
        """
        env = gym.make(env_name)
        try:
            observation_space = env.observation_space
            action_space = env.action_space
        finally:
            try:
                env.close()
            except Exception:
                pass

        # lightweight container for the two spaces
        from types import SimpleNamespace
        return SimpleNamespace(observation=observation_space, action=action_space)