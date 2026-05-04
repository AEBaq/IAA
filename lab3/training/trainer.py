"""Training loop for the Duckiebot lane-following agent.

The main public entry point is :func:`train_agent`.  The private helpers
(``_checkpoint_state_path``, ``_load_checkpoint``, ``_save_training_state``,
``_start_render_toggle_listener``) are intentionally small and focused so that
students can drop in any RL algorithm without touching the scaffolding.

To adapt this file to your own agent
--------------------------------------
1. Import your agent class (uncomment / edit the ``# TODO: Import your agent``
   line at the top).
2. Instantiate the agent inside :func:`train_agent` (replace the
   ``# TODO: Initialize your agent`` block).
3. Wire up the three agent API calls marked with TODO comments::

       action = agent.choose_action(obs)
       agent.store_transition(obs, action, reward, done)
       agent.learn()

4. Uncomment the ``agent.save(...)`` calls in the checkpoint blocks.
"""

import json
import os
import sys
import threading

import matplotlib.pyplot as plt
import numpy as np

# TODO: Import your agent
# from agent import YourAgent
from agent import PPOAgent

from duckie_env import create_env, DuckiebotWrapper, LaneFollowingEnv
from reward import advanced_reward_function


def _checkpoint_state_path(checkpoint_path: str) -> str:
    """Derive training-state JSON path from a checkpoint path.

    Convention:
        checkpoints/agent_best.pth   →  checkpoints/training_state_best.json
        checkpoints/agent_ep500.pth  →  checkpoints/training_state_ep500.json
    """
    return checkpoint_path.replace("agent_", "training_state_").replace(".pth", ".json")


def _load_checkpoint(agent, checkpoint_path: str) -> dict:
    """Load agent weights and return the saved training state dict.

    Returns an empty dict if no training-state file is found, so the caller
    can start fresh counters while still benefitting from the pre-trained weights.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[Resume] Loading checkpoint from {checkpoint_path}")
    # TODO: adapt this to your agent's checkpoint loading mechanism
    agent.load(checkpoint_path)

    state_path = _checkpoint_state_path(checkpoint_path)
    training_state = {}
    if os.path.exists(state_path):
        with open(state_path) as f:
            training_state = json.load(f)
        print(f"[Resume] Loaded training state from {state_path} "
              f"(episode {training_state.get('episode', 0)}, "
              f"best_avg {training_state.get('best_avg_score', 'n/a'):.2f})")
    else:
        print(f"[Resume] No training state found at {state_path} — starting counters from zero.")

    return training_state


def _save_training_state(checkpoint_path: str, episode: int, update_count: int,
                         scores: list, best_avg_score: float):
    """Write training state alongside the checkpoint so it can be restored later."""
    state_path = _checkpoint_state_path(checkpoint_path)
    state = {
        "episode":        episode,
        "update_count":   update_count,
        "best_avg_score": best_avg_score,
        # Keep last 200 scores so the running average is accurate on resume
        "scores":         scores[-200:],
    }
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)


def _start_render_toggle_listener(state: dict) -> None:
    """Daemon thread: press Enter at any time to toggle rendering on/off."""
    print("[Render toggle] Press Enter at any time to toggle rendering on/off.")
    while True:
        try:
            sys.stdin.readline()
            state['render'] = not state['render']
            print(f"\n[Render {'ON' if state['render'] else 'OFF'}]")
        except Exception:
            break


def train_agent(map_name: str, map_graph: str, render: bool = False,
                resume_from: str = None):
    """Run the main training loop.

    The loop runs for ``n_episodes`` episodes.  At every episode the agent
    interacts with the environment, accumulates experience, and triggers a
    learning update.  Checkpoints are saved every 500 episodes and whenever
    a new best 100-episode average is reached.

    Parameters
    ----------
    map_name:
        Name of the gym-duckietown map to load (e.g. ``"iaa26_lab3"``).
        Must be registered in the gym-duckietown map registry.
    map_graph:
        Path to the YAML road-network graph file
        (e.g. ``"map/iaa26_lab3_graph.yaml"``).
    render:
        If ``True``, open a live rendering window from the first episode.
        Can be toggled at any time by pressing **Enter** in the terminal.
    resume_from:
        Path to a ``.pth`` checkpoint file.  When provided, agent weights
        *and* training counters (episode, scores, best average) are restored
        so training continues seamlessly.  Pass ``None`` to start fresh.

    Side-effects
    ------------
    * Writes periodic checkpoints to ``checkpoints/`` (creates the directory
      if it does not exist).
    * Saves ``training_metrics.png`` with score / average-score curves.

    Notes
    -----
    Several lines are left as **TODO** stubs that students must fill in:

    * Agent instantiation (``# TODO: Initialize your agent``).
    * Action selection (``agent.choose_action(obs)``).
    * Transition storage (``agent.store_transition(...)``).
    * Learning update (``agent.learn()``).
    * Checkpoint saving (``agent.save(...)`` calls).
    """
    # Shared mutable render state — can be flipped live from the listener thread
    render_state = {'render': render}
    listener = threading.Thread(target=_start_render_toggle_listener, args=(render_state,), daemon=True)
    listener.start()

    # Create and wrap the environment
    env = create_env(map_name)
    env = DuckiebotWrapper(env, map_graph)
    env = LaneFollowingEnv(env, reward_function=advanced_reward_function)

    # Hyperparameters
    n_episodes = 5000
    # TODO: add your hyperparameters here

    # TODO: Initialize your agent
    # agent = YourAgent(
    #     input_dims=(120, 160, 3),
    #     n_actions=2,
    # )
    agent = PPOAgent()

    # ---- Restore checkpoint (if requested) ----
    start_episode   = 0
    update_count    = 0
    scores          = []
    best_avg_score  = -float('inf')
    if resume_from is not None:
        state = _load_checkpoint(agent, resume_from)
        start_episode  = state.get('episode',        0)
        update_count   = state.get('update_count',   0)
        scores         = state.get('scores',         [])
        best_avg_score = state.get('best_avg_score', -float('inf'))
        print(f"[Resume] Continuing from episode {start_episode} "
              f"(update_count={update_count}, best_avg={best_avg_score:.2f})")

    # Training metrics (pre-populated if resuming)
    avg_scores = [np.mean(scores[-100:]) if scores else 0.0]
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    for episode in range(start_episode, start_episode + n_episodes):
        obs = env.reset()
        if render_state['render']:
            # Recreate the window at every episode start to flush accumulated
            # OpenGL blend state that causes progressive brightness drift
            base_env = env.unwrapped
            if hasattr(base_env, 'window') and base_env.window is not None:
                base_env.window.close()
                base_env.window = None
            env.render()
        done = False
        score = 0
        episode_steps = 0

        while not done:
            # TODO: adapt to your agent's action selection method
            action = agent.choose_action(obs)

            next_obs, reward, done, info = env.step(action)
            score += reward
            episode_steps += 1

            # TODO: adapt to your agent's transition storage method
            agent.store_transition(obs, action, reward, done)
            obs = next_obs

            if render_state['render']:
                env.render()
            else:
                # Close the window whenever rendering is off
                base_env = env.unwrapped
                if hasattr(base_env, 'window') and base_env.window is not None:
                    base_env.window.close()
                    base_env.window = None

        # TODO: trigger your agent's learning/update step
        agent.learn()
        update_count += 1

        scores.append(score)
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)

        # --- Periodic checkpoint (every 500 episodes) ---
        if (episode + 1) % 500 == 0:
            ckpt_path = os.path.join(checkpoint_dir, f'agent_ep{episode+1}.pth')
            # TODO: adapt to your agent's checkpoint saving mechanism
            # agent.save(ckpt_path)
            agent.save(ckpt_path)
            _save_training_state(ckpt_path, episode + 1, update_count, scores, best_avg_score)
            print(f'  [Checkpoint] Saved at episode {episode+1}')

        # --- Save best model ---
        if avg_score > best_avg_score:
            best_avg_score = avg_score
            best_ckpt_path = os.path.join(checkpoint_dir, 'agent_best.pth')
            # TODO: adapt to your agent's checkpoint saving mechanism
            # agent.save(best_ckpt_path)
            agent.save(best_ckpt_path)
            _save_training_state(best_ckpt_path, episode + 1, update_count, scores, best_avg_score)

    # Plot training metrics
    plt.figure(figsize=(12, 4))
    plt.plot(scores, label='Score')
    plt.plot(avg_scores, label='Average Score')
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig('training_metrics.png')
    plt.close()

    # TODO: save your final model
    # agent.save('agent.pth')
    agent.save('agent.pth')
