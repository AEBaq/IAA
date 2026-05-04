"""Evaluation loop for a trained Duckiebot lane-following agent.

The public entry point is :func:`evaluate_agent`.  It loads a checkpoint,
runs a fixed number of episodes, records per-episode metrics, saves a GIF for
every episode, and prints an aggregated summary.

To adapt this file to your own agent
--------------------------------------
1. Import your agent class (uncomment / edit the ``# TODO: Import your agent``
   line at the top).
2. Instantiate the agent inside :func:`evaluate_agent` (replace the
   ``# TODO: Initialize your agent`` block).
3. Wire up the action selection call marked with a TODO comment::

       action = agent.choose_action(obs)

4. Add any additional metrics you want to track to the ``metrics`` dict.
"""

import os

import imageio
import numpy as np

# TODO: Import your agent
# from agent import YourAgent
from agent import PPOAgent

from duckie_env import create_env, DuckiebotWrapper, LaneFollowingEnv
from reward import reward_function


def evaluate_agent(map_name: str, map_graph: str, num_episodes: int = 10,
                   render: bool = True, checkpoint_path: str = None) -> dict:
    """Evaluate a trained agent over multiple episodes and report metrics.

    Loads a checkpoint, runs ``num_episodes`` full episodes without any
    learning updates, and saves one GIF per episode.

    Parameters
    ----------
    map_name:
        Name of the gym-duckietown map (e.g. ``"iaa26_lab3"``).
    map_graph:
        Path to the YAML road-network graph file
        (e.g. ``"map/iaa26_lab3_graph.yaml"``).
    num_episodes:
        Number of evaluation episodes to run.
    render:
        If ``True``, frames are captured via ``env.render(mode='rgb_array')``
        and saved as GIF files (``episode_<N>_eval.gif``).
    checkpoint_path:
        Path to the agent checkpoint.  Falls back to
        ``checkpoints/agent_best.pth`` or ``agent.pth`` when ``None``.

    Returns
    -------
    metrics : dict
        Dictionary of aggregated evaluation metrics, currently containing::

            {"Average Reward": float}

        Extend this dict with additional metrics (success rate, average steps,
        etc.) as needed.

    Notes
    -----
    Several lines are left as **TODO** stubs that students must fill in:

    * Agent instantiation (``# TODO: Initialize your agent``).
    * Action selection (``agent.choose_action(obs)``).
    """
    # TODO: Initialize your agent
    # agent = YourAgent(
    #     input_dims=(120, 160, 3),
    #     n_actions=2,
    # )
    agent = PPOAgent()

    if checkpoint_path is None:
        checkpoint_path = 'checkpoints/agent_best.pth' if os.path.exists('checkpoints/agent_best.pth') else 'agent.pth'

    print(f"[Evaluator] Loading checkpoint from {checkpoint_path}")
    # TODO: adapt to your agent's checkpoint loading mechanism
    agent.load(checkpoint_path)

    # Set up the environment
    env = create_env(map_name)
    env = DuckiebotWrapper(env, map_graph)
    env = LaneFollowingEnv(env, reward_function=reward_function)

    # Per-episode metrics storage
    episode_rewards = []

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        # Video frames for GIF recording
        video_frames = []

        while not done:
            # TODO: adapt to your agent's action selection method
            action = agent.choose_action(obs)

            obs, reward, done, info = env.step(action)

            # Record rendered frame
            if render:
                frame = env.render(mode='rgb_array')
                video_frames.append(frame)

            episode_reward += reward
            steps += 1

        # Save GIF
        imageio.mimsave(f'episode_{episode}_eval.gif', video_frames, duration=100)  # 100ms = 10 fps
        
        # Accumulate per-episode metrics
        episode_rewards.append(episode_reward)

    # Aggregate metrics across all episodes
    metrics = {
        "Average Reward": np.mean(episode_rewards),
    }

    # Print metrics
    print("\nEvaluation Metrics:")
    print("-" * 50)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")

    return metrics
