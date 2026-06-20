from __future__ import annotations

import argparse
import json
from pathlib import Path


def run(
    config_path: str,
    history_path: str | None = None,
    output_path: str | None = None,
) -> None:
    from src.config import load_config
    from src.model import plot_history

    config = load_config(config_path)
    configured_output_dir = Path(config["training"]["output_dir"])
    history_file = (
        Path(history_path)
        if history_path is not None
        else configured_output_dir / "history.json"
    )
    output_dir = Path(output_path) if output_path is not None else configured_output_dir
    with history_file.open("r", encoding="utf-8") as file:
        history = json.load(file)

    plotted = plot_history(history, output_dir)
    if plotted:
        print(f"Saved plots to {output_dir / 'plots'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training curves from history.json.")
    parser.add_argument("--config", default="configs/bertweet.yaml")
    parser.add_argument("--history", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    run(args.config, args.history, args.output_dir)


if __name__ == "__main__":
    main()
