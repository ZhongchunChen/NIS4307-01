from __future__ import annotations

import argparse

from src.cli.arguments import positive_int


def run(
    config_path: str,
    device: str | None = None,
    epochs: int | None = None,
    output_dir: str | None = None,
    checkpoint_dir: str | None = None,
) -> None:
    from src.config import load_config
    from src.model import train

    config = load_config(config_path)
    overrides = {
        "device": device,
        "epochs": epochs,
        "output_dir": output_dir,
        "checkpoint_dir": checkpoint_dir,
    }
    for name, value in overrides.items():
        if value is not None:
            config["training"][name] = value
    train(config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune BERTweet for binary classification.")
    parser.add_argument("--config", default="configs/bertweet.yaml")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=positive_int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        args.config,
        args.device,
        args.epochs,
        args.output_dir,
        args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
