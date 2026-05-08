from __future__ import annotations

import argparse

# importing the workflows from the src folder
from src.evaluate import evaluate 
from src.train import train


# defines the CLI interface engagement 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a malaria CNN.")
    parser.add_argument(
        "command",
        choices=["train", "evaluate", "all"], # possible choices you can run i.e python main.py train/evaluate/all
        help="Pipeline step to run.",
    )
    parser.add_argument( #managing the config files, you can change as desired
        "--config",
        default="config/config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument( # optional, allows you to evaluate any other model
        "--model",
        default=None,
        help="Optional model path for evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command in {"train", "all"}:
        model_path = train(args.config)
        print(f"Saved model to {model_path}")

    if args.command in {"evaluate", "all"}:
        metrics_path = evaluate(args.config, args.model)
        print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
