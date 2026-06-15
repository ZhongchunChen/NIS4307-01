from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    project_root = path.parent.parent
    config["_config_path"] = str(path)
    config["_project_root"] = str(project_root)

    for key in ("train_path", "val_path"):
        value = Path(config["data"][key])
        config["data"][key] = str(value if value.is_absolute() else project_root / value)

    extra_datasets = config["data"].get("extra_datasets", {})
    extra_paths = extra_datasets.get("paths", [])
    config["data"].setdefault("extra_datasets", {})
    config["data"]["extra_datasets"]["paths"] = [
        str(path if (path := Path(extra_path)).is_absolute() else project_root / path)
        for extra_path in extra_paths
    ]

    output_dir = Path(config["training"]["output_dir"])
    config["training"]["output_dir"] = str(
        output_dir if output_dir.is_absolute() else project_root / output_dir
    )

    checkpoint_dir = config["training"].get("checkpoint_dir")
    if checkpoint_dir:
        value = Path(checkpoint_dir)
        config["training"]["checkpoint_dir"] = str(
            value if value.is_absolute() else project_root / value
        )
    else:
        config["training"]["checkpoint_dir"] = str(project_root / "checkpoints" / "bertweet")

    base_checkpoint_dir = config.get("model", {}).get("base_checkpoint_dir")
    if base_checkpoint_dir:
        value = Path(base_checkpoint_dir)
        config["model"]["base_checkpoint_dir"] = str(
            value if value.is_absolute() else project_root / value
        )
    return config
