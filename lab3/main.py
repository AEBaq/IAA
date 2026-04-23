"""Entry point for training and evaluating the Duckiebot lane-following agent.

Usage examples::

    # Train from scratch (no rendering)
    python main.py

    # Train with live rendering (slower)
    python main.py --render

    # Resume training from the default best checkpoint
    python main.py --resume

    # Resume training from a specific checkpoint
    python main.py --resume checkpoints/agent_ep1000.pth

    # Skip training and evaluate the best saved checkpoint
    python main.py --eval-only

    # Evaluate a specific checkpoint
    python main.py --eval-only --checkpoint checkpoints/agent_ep1000.pth

Map files are resolved relative to the working directory (``map/``).
"""

import argparse

from training import train_agent, evaluate_agent


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train / evaluate the Duckiebot agent.")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="checkpoints/agent_best.pth",
        default=None,
        metavar="CHECKPOINT",
        help=(
            "Resume training from a checkpoint.  "
            "Optionally supply the path to the checkpoint file; "
            "defaults to checkpoints/agent_best.pth when the flag is "
            "given without a value."
        ),
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=False,
        help="Render training (slow — toggle live with Enter).",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        default=False,
        help="Skip training and only run evaluation (loads best checkpoint).",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        metavar="CHECKPOINT",
        help=(
            "Path to the checkpoint to use for evaluation.  "
            "Only relevant when --eval-only is set.  "
            "Defaults to checkpoints/agent_best.pth."
        ),
    )
    args = parser.parse_args()

    map_name  = "iaa26_lab3"
    map_graph = "map/iaa26_lab3_graph.yaml"

    if not args.eval_only:
        train_agent(map_name, map_graph, render=args.render, resume_from=args.resume)

    # Evaluate the trained agent
    print("\nEvaluating trained agent...")
    metrics = evaluate_agent(map_name, map_graph, num_episodes=10, render=True, checkpoint_path=args.checkpoint)
