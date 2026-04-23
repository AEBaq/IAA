# Lab 3 – Reinforcement Learning for Duckiebot Lane Following

In this lab you will train a **Reinforcement Learning agent** to drive a Duckiebot
autonomously through a road network.  The robot must follow the lane and navigate
from a randomly sampled start tile to a goal tile.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [What You Have to Implement](#what-you-have-to-implement)
5. [Running the Code](#running-the-code)
6. [Environment Reference](#environment-reference)
7. [Tips & Common Pitfalls](#tips--common-pitfalls)

---

## Overview

The simulator is [gym-duckietown](https://github.com/duckietown/gym-duckietown)
([official documentation](https://docs.duckietown.com/daffy/devmanual-software/intermediate/simulation/gym-simulation-in-duckietown.html#)).
The code base wraps it in two Gym layers and exposes a clean RL interface:

```
Simulator  (gym-duckietown)
    └─ DuckiebotWrapper     converts [velocity, steering] → wheel commands,
    │                       handles map-graph path planning & episode resets
    └─ LaneFollowingEnv     replaces the built-in reward with your custom one
```

At each episode reset the environment:
- Samples a random **start** tile and a non-adjacent **finish** tile (both on
  straight road sections).
- Computes the shortest path between them using the road-network graph.
- Places the robot at the start tile facing the first waypoint.

Your agent receives an **RGB camera image** and must output a
**[velocity, steering]** action at every step.

---

## Project Structure

```
lab3/
├── main.py                  # Entry point — training + evaluation CLI
├── manual_control.py        # Drive the robot manually with the keyboard
│
├── agent/                   # ← YOUR AGENT GOES HERE
│   └── __init__.py          #   Export your class from this file
│
├── duckie_env/
│   ├── factory.py           # create_env() — builds the raw Simulator
│   ├── duckiebot_wrapper.py # Action-space conversion + path planning
│   └── lane_following_env.py# Custom reward injection
│
├── map/
│   ├── MapGraph.py          # Road-network graph utilities (NetworkX)
│   ├── iaa26_lab3.yaml      # Map layout (copy to duckietown_world, see below)
│   └── iaa26_lab3_graph.yaml# Graph edges used for path planning
│
├── reward/
│   └── reward_functions.py  # ← YOUR REWARD FUNCTION GOES HERE
│
└── training/
    ├── trainer.py           # train_agent() — main training loop
    └── evaluator.py         # evaluate_agent() — evaluation loop + GIF export
```

---

## Installation

> **Remote machine?**  Connect with `ssh -X` so the simulator can open a window.
> Without it you will get:
> `pyglet.canvas.xlib.NoSuchDisplayException: Cannot connect to "None"`

### 1 — System dependencies

System dependencies you may already have.

```bash
sudo apt-get install curl gcc build-essential zlib1g zlib1g-dev libssl-dev libbz2-dev libsqlite3-dev \
   libssl-dev libffi-dev freeglut3-dev python3-gi python3-gi-cairo libgdk-pixbuf2.0-0 \
   gir1.2-gdkpixbuf-2.0 libgirepository1.0-dev libcairo2-dev libjpeg-dev libgif-dev libgtk2.0-0
```

Install UV, PyEnv, and Python 3.7.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
curl -fsSL https://pyenv.run | bash

cat >> ~/.bashrc << 'EOF'

export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"
eval "$(pyenv virtualenv-init -)"
EOF

exec "$SHELL"

pyenv install 3.7
```


### 2 — Python environment

Select the appropriate PyTorch build for your GPU architecture.

```bash
cd lab3/
uv init --python 3.7
uv add "duckietown-gym-daffy @ https://github.com/duckietown/gym-duckietown.git"
uv add "pyglet<=1.5.0,>=1.4.0"
uv add typing-extensions
uv add "numpy==1.19.0"
uv add "gym>=0.9.0"
uv add PyGObject
uv pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117
uv sync
```

### 3 — Register the custom map

The map YAML must be placed in the `duckietown_world` data directory so the
simulator can find it:

```bash
cp map/iaa26_lab3.yaml \
   .venv/lib/python3.7/site-packages/duckietown_world/data/gd1/maps/
```

---

## What You Have to Implement

There are **four independent tasks**.  Each is marked with a `# TODO` comment in
the relevant file.

### Task 1 — Path planning (`duckie_env/duckiebot_wrapper.py`)

Inside `DuckiebotWrapper.reset()`, two variables must be populated:

| Variable | Type | Description |
|---|---|---|
| `self.path` | `list[str]` | Ordered node names from start to finish, e.g. `["T_2_3", "T_2_4", …]` |
| `self.next_node` | `str` | The first node after `self.start_node` (the immediate waypoint) |

Use `self.map_graph` (a `MapGraph` instance) and its underlying NetworkX graph
`self.map_graph.G`.

### Task 2 — Reward function (`reward/reward_functions.py`)

Implement `reward_function(observation, action, done, info, **kwargs) -> float`.

Key signals available in `info["Simulator"]`:

| Key | Type | Meaning |
|---|---|---|
| `lane_position["dist"]` | `float` | Signed lateral distance from lane centre (metres) |
| `lane_position["dot_dir"]` | `float` | Dot product of robot heading and lane direction ∈ [-1, 1] |
| `lane_position["angle_deg"]` | `float` | Heading error in degrees |
| `robot_speed` | `float` | Current speed (m/s) |
| `proximity_penalty` | `float` | Cost from being close to obstacles |

The `prev_action` keyword argument holds the previous `[velocity, steering]` so
you can penalise abrupt steering changes.

> Note that even though these features are made available to you by the simulator, they may not be available outside of GYM.

### Task 3 — RL agent (`agent/`)

Implement your RL algorithm in the `agent/` package and export your class from
`agent/__init__.py`.  Your class must expose at minimum:

```python
class YourAgent:
    def choose_action(self, observation) -> np.ndarray:
        """Return [velocity, steering] given the current observation."""

    def store_transition(self, obs, action, reward, done):
        """Store a transition in the replay buffer / rollout storage."""

    def learn(self):
        """Perform one learning/update step (called once per episode)."""

    def save(self, path: str):
        """Save model weights to disk."""

    def load(self, path: str):
        """Load model weights from disk."""
```

The observation passed to `choose_action` is a dict with key `"image"`
(a `(120, 160, 3)` uint8 NumPy array from the forward-facing camera).

### Task 4 — Wire up trainer & evaluator

Once your agent class exists:

1. **`training/trainer.py`** — uncomment / edit the `# TODO` blocks:
   - Import and instantiate your agent.
   - Call `agent.choose_action`, `agent.store_transition`, `agent.learn`.
   - Uncomment `agent.save(...)` in the checkpoint blocks.

2. **`training/evaluator.py`** — same import + instantiation pattern, plus
   `agent.choose_action`.

---

## Running the Code

### Train from scratch

```bash
uv run main.py
```

### Train with live rendering (slower)

```bash
uv run main.py --render
# Press Enter at any time to toggle rendering on/off
```

### Resume training from a checkpoint

```bash
# Resume from the automatically saved best checkpoint
uv run main.py --resume

# Resume from a specific checkpoint
uv run main.py --resume checkpoints/agent_ep1000.pth
```

### Evaluate only (no training)

```bash
# Uses checkpoints/agent_best.pth by default
uv run main.py --eval-only

# Evaluate a specific checkpoint
uv run main.py --eval-only --checkpoint checkpoints/agent_ep2000.pth
```

### Manual control

Drive the robot yourself to understand the simulator before coding your agent:

```bash
uv run manual_control.py
```

---

## Environment Reference

### Action space

| Index | Name | Range | Effect |
|---|---|---|---|
| 0 | velocity | [0.0, 1.0] | Forward speed (0 = stopped, 1 = full speed) |
| 1 | steering | [-1.0, 1.0] | Turn rate (+ = left, − = right) |

The wrapper converts this to differential-drive `[left_motor, right_motor]`
commands internally.

### Observation space

```python
{"image": np.ndarray}   # shape (120, 160, 3), dtype uint8, RGB
```

### Episode lifecycle

1. `env.reset()` — samples start/finish nodes, places the robot, returns first obs.
2. `env.step(action)` — advances the simulation; returns `(obs, reward, done, info)`.
3. `done = True` when the robot collides, leaves the road, or `max_steps` (600) is reached.

### Map graph

Node names follow the pattern `T_<row>_<col>` (zero-indexed, top-left origin).
Tile coordinates include a `+1` offset in both axes (see `MapGraph.node_name_offset`).
Use `map_graph.G` to access the raw NetworkX graph for path-planning algorithms.

### Checkpoints

The training loop saves two types of files per checkpoint into `checkpoints/`:

| File | Contents |
|---|---|
| `agent_best.pth` / `agent_ep<N>.pth` | Model weights (format: your agent's `save`) |
| `training_state_best.json` / `training_state_ep<N>.json` | Episode counter, scores, best average |

---