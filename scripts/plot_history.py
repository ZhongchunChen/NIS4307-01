from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import load_config
from src.training import plot_history


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training curves from history.json.")
    parser.add_argument("--config", default="configs/bertweet.yaml")
    parser.add_argument("--history", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config["training"]["output_dir"])
    history_path = Path(args.history or output_dir / "history.json")
    with history_path.open("r", encoding="utf-8") as file:
        history = json.load(file)

    training = config["training"]
    plotted = plot_history(
        history,
        output_dir,
        training["early_stopping_metric"],
        training.get("early_stopping_mode", "max"),
    )
    if plotted:
        print(f"Saved plots to {output_dir / 'plots'}")


if __name__ == "__main__":
    main()
