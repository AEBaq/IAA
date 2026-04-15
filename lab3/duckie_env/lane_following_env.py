import gym

from reward import basic_reward_function
from .duckiebot_wrapper import DuckiebotWrapper


class LaneFollowingEnv(gym.Wrapper):
    """Gym wrapper that replaces the simulator's built-in reward with a
    custom reward function and forwards all other calls to the inner env.
    """

    def __init__(self, env: DuckiebotWrapper, reward_function):
        """Wrap an existing :class:`~duckie_env.DuckiebotWrapper` with a custom reward.

        Parameters
        ----------
        env:
            A :class:`~duckie_env.DuckiebotWrapper` instance (or any
            ``gym.Wrapper`` that wraps one).
        reward_function:
            Callable with the signature::

                reward_function(observation, action, done, info, **kwargs) -> float

            ``prev_action`` is passed as a keyword argument so that
            reward functions can penalise abrupt steering changes.
        """
        super(LaneFollowingEnv, self).__init__(env)
        self.reward_function = reward_function
        self.prev_action = None

    def step(self, action):
        """Step the environment and compute the custom reward.

        The simulator's built-in reward is discarded; ``self.reward_function``
        is called instead with the new observation, the action, and the
        previous action.

        Parameters
        ----------
        action : array-like, shape (2,)
            ``[velocity, steering]`` as produced by the agent.

        Returns
        -------
        observation : dict
            Observation dict (``"image"`` key).
        reward : float
            Reward computed by ``self.reward_function``.
        done : bool
            Whether the episode has ended.
        info : dict
            Auxiliary simulator info.
        """
        observation, _, done, info = self.env.step(action)

        # Calculate custom reward
        reward = self.reward_function(
            observation,
            action,
            done,
            info,
            prev_action=self.prev_action,
        )

        # Update previous action
        self.prev_action = action

        return observation, reward, done, info

    def reset(self):
        """Reset the environment and clear the previous-action buffer.

        Returns
        -------
        obs : dict
            Initial observation from the inner :class:`~duckie_env.DuckiebotWrapper`.
        """
        self.prev_action = None
        return self.env.reset()
