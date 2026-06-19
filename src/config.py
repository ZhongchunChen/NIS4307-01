from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
import os
from dotenv import load_dotenv

load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _is_complete_checkpoint(path: str | Path) -> bool:
    checkpoint = Path(path)
    model_exists = any(
        (checkpoint / filename).exists()
        for filename in (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
    )
    return model_exists and all(
        (checkpoint / filename).exists()
        for filename in ("config.json", "tokenizer_config.json")
    )


def resolve_checkpoint_path(
    config: dict[str, Any],
    checkpoint_path: str | Path | None = None,
) -> str:
    """Prefer an explicit or trained local checkpoint, then use the HF fallback."""
    if checkpoint_path is not None:
        explicit_path = Path(checkpoint_path).expanduser()
        if not explicit_path.is_absolute():
            explicit_path = PROJECT_ROOT / explicit_path
        if not _is_complete_checkpoint(explicit_path):
            raise FileNotFoundError(
                f"Model checkpoint is missing or incomplete at {explicit_path}"
            )
        return str(explicit_path)

    local_path = Path(config["training"]["checkpoint_dir"]) / "best_model"
    if _is_complete_checkpoint(local_path):
        return str(local_path)

    huggingface_config = config["model"].get("huggingface_checkpoint", {})
    repo_id = huggingface_config.get("repo_id")
    if not repo_id:
        raise FileNotFoundError(
            f"Model checkpoint not found at {local_path} and no Hugging Face fallback is configured"
        )

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    download_args = {
        "repo_id": repo_id,
        "repo_type": "model",
        "revision": huggingface_config.get("revision", "main"),
    }
    try:
        try:
            cached_path = snapshot_download(**download_args, local_files_only=True)
            if _is_complete_checkpoint(cached_path):
                return cached_path
        except LocalEntryNotFoundError:
            pass
        downloaded_path = snapshot_download(**download_args)
    except Exception as exc:
        raise FileNotFoundError(
            f"Unable to resolve the Hugging Face checkpoint {repo_id!r}"
        ) from exc
    if not _is_complete_checkpoint(downloaded_path):
        raise FileNotFoundError(
            f"Downloaded Hugging Face checkpoint is incomplete at {downloaded_path}"
        )
    return downloaded_path


def resolve_data_path(
    value: str | Path,
    huggingface_config: dict[str, Any] | None = None,
) -> str:
    """Resolve a local data path, downloading it from Hugging Face if absent."""
    path = Path(value).expanduser()
    local_path = path if path.is_absolute() else PROJECT_ROOT / path
    if local_path.exists() or not huggingface_config:
        return str(local_path)

    repo_id = huggingface_config.get("repo_id")
    if not repo_id:
        return str(local_path)

    local_dir = _project_path(huggingface_config.get("local_dir", "datasets"))
    try:
        filename = local_path.relative_to(local_dir).as_posix()
    except ValueError:
        try:
            filename = local_path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            filename = path.as_posix()

    try:
        project_path = local_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        project_path = path.as_posix()
    file_map = huggingface_config.get("files", {})
    filename = file_map.get(project_path, file_map.get(path.as_posix(), filename))

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    download_args = {
        "repo_id": repo_id,
        "filename": filename,
        "repo_type": "dataset",
        "revision": huggingface_config.get("revision", "main"),
    }
    try:
        return hf_hub_download(**download_args, local_files_only=True)
    except LocalEntryNotFoundError:
        pass
    return hf_hub_download(
        **download_args,
    )


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    project_root = PROJECT_ROOT
    config["_config_path"] = str(path)
    config["_project_root"] = str(project_root)

    for key in ("train_path", "val_path"):
        value = Path(config["data"][key]).expanduser()
        config["data"][key] = str(
            value if value.is_absolute() else project_root / value
        )

    extra_datasets = config["data"].get("extra_datasets", {})
    extra_paths = extra_datasets.get("paths", [])
    config["data"].setdefault("extra_datasets", {})
    config["data"]["extra_datasets"]["paths"] = [
        str(
            path
            if (path := Path(extra_path).expanduser()).is_absolute()
            else project_root / path
        )
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

API_BASE_URL = os.getenv("API", "https://models.sjtu.edu.cn/api/v1")
API_SECRET = os.getenv("API_SECRET", "")
API_MODEL = os.getenv("API_MODEL", "deepseek-reasoner")
