"""Dummy gym.spaces.Box pixel environment for testing purpose."""
import gym
import numpy as np

from tests.fixtures.envs.dummy import DummyEnv


class DummyBoxPixelEnv(DummyEnv):
    """A dummy gym.spaces.Box environment with pixel values.

    This environment is identical to DummyBoxEnv except that
    observation space has low, high values of 0, 255 and dtype np.uint8
    to enable conversion an akro.Image space.

    Args:
        random (bool): If observations are randomly generated or not.
        obs_dim (tuple[int]): Observation space dimension.
        action_dim (tuple[int]): Action space dimension.

    """

    def __init__(self, random=True, obs_dim=(10, 10, 3), action_dim=(2, )):
        super().__init__(random, obs_dim, action_dim)

        self._observation_space = gym.spaces.Box(low=0,
                                                 high=255,
                                                 shape=self._obs_dim,
                                                 dtype=np.uint8)

        self._action_space = gym.spaces.Box(low=-5.0,
                                            high=5.0,
                                            shape=self._action_dim,
                                            dtype=np.float32)

    @property
    def observation_space(self):
        """Return an observation space.

        Returns:
            gym.space.Box: The observation space of the environment.

        """
        return self._observation_space

    @property
    def action_space(self):
        """Return an action space.

        Returns:
            gym.spaces.Box: The action space of the environment.

        """
        return self._action_space

    def reset(self):
        """Reset the environment.

        Returns:
            np.ndarray: The observation obtained after reset.

        """
        return np.ones(self._obs_dim, dtype=np.uint8)

    def step(self, action):
        """Step the environment.

        Args:
            action (int): Action input of shape :math:`(A*, )`.

        Returns:
            np.ndarray: Observation of shape :math:`(O*, )`.
            float: Reward.
            bool: If the environment is terminated.
            dict: Environment information.

        """
        return self.observation_space.sample(), 0, False, dict(dummy='dummy')
