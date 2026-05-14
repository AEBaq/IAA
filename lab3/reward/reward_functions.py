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
      (lateral distance from the lane centre) as a negative penalty. : DONE
    * **Heading alignment** – use ``info['Simulator']['lane_position']['dot_dir']``
      (dot product of robot heading and lane direction) as a positive bonus. : DONE
    * **Collision / out-of-road penalty** – apply a large negative reward when
      ``done`` is ``True`` and the agent did *not* reach the goal. : DONE
    * **Goal reward** – if you implement goal detection (finish node reached),
      give a large positive bonus. : DONE
    * **Smooth steering** – penalise abrupt changes between ``action[1]`` and
      ``kwargs.get('prev_action', action)[1]``. : DONE
    """
    reward = 0.0  # TODO: implement reward

    velocity = action[0]
    steering = action[1]

    reward += velocity  # encourage forward motion

    simulator_info = info.get('Simulator', {})
    lane_position = simulator_info.get('lane_position', {})

    if lane_position:
        # Lane-following bonus
        dist_from_center = lane_position.get('dist', 0.0)
        reward -= dist_from_center  * 1 # penalize distance from lane center, can be tuned with a weight (currently set to 1)

        # Heading alignment bonus
        dot_dir = lane_position.get('dot_dir', 0.0)
        reward += dot_dir * 1.2 # encourage alignment with lane direction, can be tuned with weight (currently set to 1.2)
    
    # Smooth steering
    prev_action = kwargs.get('prev_action') # get the previous action from kwargs, default to None if not provided
    if prev_action is not None:
        steering_change = abs(steering - prev_action[1])
        reward -= steering_change * 0.5 # penalize abrupt steering changes, can be tuned with weight (currently set to 0.5)

    if done:
        # get the message from simulator info to determine
        # if the episode ended due to reaching the goal or a collision
        msg = simulator_info.get('msg', '') 
        if 'goal' in msg: # Goal reward
            reward += 100.0  # large positive reward for reaching the goal
        else: # Collision / out-of-road penalty
            reward -= 100.0  # large negative penalty for collision or going out of bounds

    return reward