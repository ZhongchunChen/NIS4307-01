"""Rumor Detection System — entry point.

Usage:
    python main.py                  Start the web server (default)
    python main.py --display        Start the web server
    python main.py --train          Train the BERTweet model
    python main.py --help           Show help
"""

import argparse

import uvicorn

from src.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rumor Detection System",
        usage="python main.py [--display | --train | --help]",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--display", action="store_true",
        help="Start the web server (default)",
    )
    group.add_argument(
        "--train", action="store_true",
        help="Train the BERTweet model",
    )

    args = parser.parse_args()

    if args.train:
        from src.train import train
        from src.config import load_config
        train(load_config("configs/bertweet.yaml"))
    else:
        app = create_app()
        uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
