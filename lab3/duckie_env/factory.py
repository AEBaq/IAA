from gym_duckietown.simulator import Simulator


def create_env(map_name: str, domain_rand: bool = False, max_steps: int = 600) -> Simulator:
    """Create a Duckietown Simulator instance for the given map.

    Args:
        map_name:    Name of the map (must exist in the gym-duckietown map registry).
        domain_rand: Enable domain randomisation (textures, lighting, etc.).
        max_steps:   Maximum number of steps per episode before forced termination.

    Returns:
        A configured :class:`Simulator` instance.
    """
    return Simulator(
        map_name=map_name,
        domain_rand=domain_rand,
        accept_start_angle_deg=4,
        full_transparency=True,
        distortion=True,
        max_steps=max_steps,
    )
