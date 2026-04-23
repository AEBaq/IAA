import numpy as np


def reward_function(observation, action, done, info, **kwargs):
    """Compute the per-step reward for the Duckiebot agent.

    This is a **stub** – replace the body with your own shaping logic.

    Parameters
    ----------
    observation : dict
        Current environment observation.  Contains at least the key
        ``"image"`` (H×W×3 uint8 NumPy array from the forward camera).
    action : array-like, shape (2,)
        Action taken by the agent: ``[velocity ∈ [0,1], steering ∈ [-1,1]]``.
    done : bool
        ``True`` if the episode ended (collision, timeout, or goal reached).
    info : dict
        Auxiliary simulator diagnostics.

    **kwargs
        Optional keyword arguments forwarded by :class:`~duckie_env.LaneFollowingEnv`.
        Currently passes ``prev_action`` (the action from the previous step,
        or ``None`` at the first step) so smooth-control penalties are possible.

    Returns
    -------
    reward : float
        Scalar reward signal for this transition.

    Implementation hints
    --------------------
    * **Lane-following bonus** – use ``info['Simulator']['lane_position']['dist']``
      (lateral distance from the lane centre) as a negative penalty.
    * **Heading alignment** – use ``info['Simulator']['lane_position']['dot_dir']``
      (dot product of robot heading and lane direction) as a positive bonus.
    * **Collision / out-of-road penalty** – apply a large negative reward when
      ``done`` is ``True`` and the agent did *not* reach the goal.
    * **Goal reward** – if you implement goal detection (finish node reached),
      give a large positive bonus.
    * **Smooth steering** – penalise abrupt changes between ``action[1]`` and
      ``kwargs.get('prev_action', action)[1]``.
    """
    reward = 0.0  # TODO: implement reward
    return reward