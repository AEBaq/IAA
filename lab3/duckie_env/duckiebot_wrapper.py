import logging
import math

import cv2
import gym
import numpy as np

from gym_duckietown.simulator import Simulator, logger

logger.setLevel(logging.WARNING)

from map import MapGraph


class DuckiebotWrapper(gym.Wrapper):
    """Gym wrapper that adapts the raw ``gym-duckietown`` :class:`Simulator` for
    the IAA-26 lab 3 lane-following task.

    Responsibilities
    ----------------
    * Converts the agent's ``[velocity, steering]`` action space into the
      differential-drive ``[left_motor, right_motor]`` format expected by the
      simulator (see :meth:`_vel_steer_to_wheels`).
    * Exposes a ``Dict`` observation space with an ``"image"`` key.
    * Integrates with :class:`~map.MapGraph` to:

        - Sample a random start/finish node pair at each episode reset.
        - Find a path between them (to be implemented with ``self.path``).
        - Orient the robot at the start tile so it faces the first waypoint.

    Attributes
    ----------
    map_graph : MapGraph
        Loaded road-network graph used for path planning.
    start_node : str or None
        Node name of the current episode's start tile (e.g. ``'T_2_3'``).
    finish_node : str or None
        Node name of the current episode's goal tile.
    path : list[str] or None
        Ordered list of node names from start to finish.
        **Must be populated in** :meth:`reset` (currently a TODO).
    next_node : str or None
        The immediate next waypoint on ``path``.
        **Must be populated in** :meth:`reset` (currently a TODO).
    """

    def __init__(self, env: Simulator, map_graph: str):
        """Initialise the wrapper.

        Parameters
        ----------
        env:
            A :class:`~gym_duckietown.simulator.Simulator` instance, typically
            created via :func:`~duckie_env.factory.create_env`.
        map_graph:
            Path to the YAML file that describes the road-network graph
            (e.g. ``"map/iaa26_lab3_graph.yaml"``).
        """
        super(DuckiebotWrapper, self).__init__(env)

        self.map_graph = MapGraph(map_graph)
        logger.debug(f"Graph loaded with nodes: {self.map_graph.nodes()}")

        self.start_node = None
        self.finish_node = None
        self.path = None
        self.next_node = None

        # Define action and observation spaces
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, -1.0], dtype=np.float32),  # [velocity, steering]
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict({
            "image": gym.spaces.Box(
                low=0, high=255,
                shape=(120, 160, 3),
                dtype=np.uint8
            ),
        })

    @staticmethod
    def _vel_steer_to_wheels(action):
        """Convert [velocity, steering] to differential-drive [left, right].

        The gym-duckietown simulator expects ``action = [left_motor, right_motor]``
        (see ``DynamicsInfo(motor_left=action[0], motor_right=action[1])``).
        Our agent outputs ``[velocity ∈ [0,1], steering ∈ [-1,1]]``.

        Convention:  steering > 0 → turn left  → right wheel faster.
                     steering < 0 → turn right → left wheel faster.

        Mapping:
            left  = velocity - steering
            right = velocity + steering

        Both wheels are clipped to [-1, 1] to satisfy the simulator's limits.
        """
        velocity  = float(action[0])
        steering  = float(action[1])
        left  = velocity - steering
        right = velocity + steering
        return np.clip(np.array([left, right]), -1.0, 1.0)

    def step(self, action):
        """Advance the simulation by one step.

        Converts the agent's ``[velocity, steering]`` action to wheel commands
        before forwarding to the underlying simulator.

        Parameters
        ----------
        action : array-like, shape (2,)
            ``[velocity ∈ [0, 1], steering ∈ [-1, 1]]``.

        Returns
        -------
        obs : dict
            Observation dict with key ``"image"`` (H×W×3 uint8 array).
        reward : float
            Raw simulator reward (replaced by a custom reward in
            :class:`~duckie_env.LaneFollowingEnv`).
        done : bool
            Whether the episode has ended.
        info : dict
            Auxiliary diagnostic information from the simulator.
        """
        wheel_action = self._vel_steer_to_wheels(action)
        obs, reward, done, info = self.env.step(wheel_action)
        return obs, reward, done, info

    def reset(self):
        """Reset the environment and place the robot at a new random start tile.

        At each call this method:

        1. Samples a random ``(start_node, finish_node)`` pair from the map
           graph (both on straight tiles, finish not adjacent to start).
        2. **TODO** – compute ``self.path`` (list of node names from start to
           finish) and set ``self.next_node`` to the first waypoint after start.
        3. Positions the robot at the centre of the start tile, in the lane
           appropriate for the direction toward ``self.next_node``.

        Returns
        -------
        obs : dict
            Initial observation dict with key ``"image"``.

        Notes
        -----
        The angle / lane positioning follows the gym-duckietown convention::

            angle 0      → East  (+X, increasing col)
            angle π/2    → North (−Z, decreasing row)
            angle π      → West  (−X, decreasing col)
            angle 3π/2   → South (+Z, increasing row)

        The robot is placed in the *right lane* for the direction of travel so
        that the standard lane-following reward fires immediately.
        """
        # sample a new random start and finish node for each episode
        self.start_node, self.finish_node = self.map_graph.sample_random_start_finish_nodes(self.env._get_tile)
        # find the path between start and finish nodes using the graph
        self.path = None # TODO
        self.next_node = None # TODO
        
        # set the duckiebot's position to the start node
        start_tile_coords = self.map_graph.node_name_to_tile_coords(self.start_node)
        self.env.user_tile_start = (start_tile_coords['col'], start_tile_coords['row'])

        # Place the duckiebot on the start tile facing TOWARD the next node on the path
        #
        # Gym-duckietown angle convention (get_dir_vec):
        #   angle 0     → +X  (East,  increasing col)
        #   angle π/2   → −Z  (North, decreasing row)
        #   angle π     → −X  (West,  decreasing col)
        #   angle 3π/2  → +Z  (South, increasing row)
        next_tile = self.map_graph.node_name_to_tile_coords(self.next_node)
        delta_col = next_tile['col'] - start_tile_coords['col']
        delta_row = next_tile['row'] - start_tile_coords['row']

        if delta_col > 0:        # next node is East  → face +X
            starting_angle = 0.
        elif delta_col < 0:      # next node is West  → face −X
            starting_angle = math.pi
        elif delta_row < 0:      # next node is North → face −Z
            starting_angle = math.pi / 2
        elif delta_row > 0:      # next node is South → face +Z
            starting_angle = 3 * math.pi / 2
        else:
            raise ValueError(
                f"Start and next node map to the same tile: "
                f"{self.start_node} → {self.next_node}"
            )

        if starting_angle == 0.: # horizontal straight tile, going from left to right
            self.env.start_pose = [[self.road_tile_size / 2, 0, 3 * self.road_tile_size / 4],
                                    starting_angle]
        elif starting_angle == math.pi: # horizontal straight tile, going from right to left
            self.env.start_pose = [[self.road_tile_size / 2, 0, 1 * self.road_tile_size / 4],
                                    starting_angle]
        elif starting_angle == math.pi / 2: # vertical straight tile, going from down to up
            self.env.start_pose = [[3 * self.road_tile_size / 4, 0, self.road_tile_size / 2],
                                    starting_angle]
        elif starting_angle == 3 * math.pi / 2: # vertical straight tile, going from up to down
            self.env.start_pose = [[1 * self.road_tile_size / 4, 0, self.road_tile_size / 2],
                                    starting_angle]
        else:
            raise ValueError(f"Invalid random angle {starting_angle}")

        obs = self.env.reset()

        logger.debug(
            f"Resetting environment.\t Start node: {self.start_node},"
            f"\n\t\t\t Next node: {self.next_node}, Path: {self.path}"
        )

        return obs
