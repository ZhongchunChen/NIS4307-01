from __future__ import annotations

import argparse

from src.config import load_config
from src.model import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune BERTweet for binary classification.")
    parser.add_argument("--config", default="configs/bertweet.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
