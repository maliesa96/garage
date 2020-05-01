"""Pixel observation wrapper for gym.Env."""
import gym
from gym.wrappers.pixel_observation import PixelObservationWrapper


class PixelObservation(gym.Wrapper):
    """Pixel observation wrapper for obtaining pixel observations.

    This behaves like gym.wrappers.PixelObservationWrapper but returns a
    gym.spaces.Box observation space and observation instead of
    a gym.spaces.Dict.

    Args:
        env (gym.Env): The enivornment to wrap. This environment must produce
            non-pixel observations and have a Box observation space.

    """

    def __init__(self, env):
        env = PixelObservationWrapper(env)

        if not isinstance(env.observation_space['pixels'], gym.spaces.Box):
            raise ValueError(
                'PixelObservation wrapper can only be used with gym.spaces.Box'
                ' observation spaces, but received {} instead.'.format(
                    type(self._observation_space).__name__))
        super().__init__(env)
        self._observation_space = env.observation_space['pixels']

    @property
    def observation_space(self):
        """gym.spaces.Box: Environment observation space."""
        return self._observation_space

    @observation_space.setter
    def observation_space(self, observation_space):
        self._observation_space = observation_space

    def reset(self, **kwargs):
        """gym.Env reset function.

        Args:
            kwargs (dict): Keyword arguments to be passed to gym.Env.reset.

        Returns:
            np.ndarray: Pixel observation of shape :math:`(O*, )`
                from the wrapped environment.
        """
        return self.env.reset(**kwargs)['pixels']

    def step(self, action):
        """gym.Env step function.

        Performs one action step in the enviornment.

        Args:
            action (np.ndarray): Action of shape :math:`(A*, ) `
                to pass to the environment.

        Returns:
            np.ndarray: Pixel observation of shape :math:`(O*, )`
                from the wrapped environment.
            float : Amount of reward returned after previous action.
            bool : Whether the episode has ended, in which case further step()
                calls will return undefined results.
            dict: Contains auxiliary diagnostic information (helpful for
                debugging, and sometimes learning).
        """
        obs, reward, done, info = self.env.step(action)
        return obs['pixels'], reward, done, info
