"""Regression tests for CLI and configuration parameter behavior."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

import main
from src.cli import plot_history as plot_history_cli
from src.cli import train as train_cli
from src.rag import service as rag_service


@pytest.mark.parametrize(
    "arguments",
    [
        ["serve", "--port", "0"],
        ["serve", "--port", "65536"],
        ["train", "--epochs", "0"],
        ["rag", "query", "statement", "--top-k", "-1"],
        ["--log-level", "verbose", "predict", "statement"],
    ],
)
def test_invalid_parameters_are_rejected(arguments: list[str]) -> None:
    with pytest.raises(SystemExit):
        main.build_parser().parse_args(arguments)


def test_log_level_is_case_insensitive() -> None:
    args = main.build_parser().parse_args(["--log-level", "debug", "predict", "text"])
    assert args.log_level == "DEBUG"


def test_reload_uses_import_string_and_preserves_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reload_env_names = (
        "RUMOR_CONFIG_PATH",
        "RUMOR_CHECKPOINT_PATH",
        "RUMOR_DEVICE",
        "RUMOR_FORCE_MOCK",
        "RUMOR_ENABLE_RAG",
        "RUMOR_ENABLE_LLM",
    )
    for name in reload_env_names:
        monkeypatch.delenv(name, raising=False)

    uvicorn = ModuleType("uvicorn")
    uvicorn.run = Mock()  # type: ignore[attr-defined]
    model = ModuleType("src.model")
    model.configure_inference = Mock()  # type: ignore[attr-defined]
    web_app = ModuleType("src.web.app")
    web_app.create_app = Mock()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn)
    monkeypatch.setitem(sys.modules, "src.model", model)
    monkeypatch.setitem(sys.modules, "src.web.app", web_app)

    args = main.build_parser().parse_args(
        [
            "serve",
            "--config",
            "custom.yaml",
            "--host",
            "127.0.0.1",
            "--port",
            "9000",
            "--reload",
            "--checkpoint",
            "model",
            "--device",
            "cpu",
            "--no-rag",
            "--no-llm",
            "--mock-model",
        ]
    )
    main.serve(args)

    uvicorn.run.assert_called_once_with(  # type: ignore[attr-defined]
        "src.web.app:create_runtime_app",
        factory=True,
        host="127.0.0.1",
        port=9000,
        reload=True,
    )
    assert main.os.environ["RUMOR_CONFIG_PATH"] == "custom.yaml"
    assert main.os.environ["RUMOR_CHECKPOINT_PATH"] == "model"
    assert main.os.environ["RUMOR_DEVICE"] == "cpu"
    assert main.os.environ["RUMOR_FORCE_MOCK"] == "1"
    assert main.os.environ["RUMOR_ENABLE_RAG"] == "0"
    assert main.os.environ["RUMOR_ENABLE_LLM"] == "0"


def test_installed_train_command_applies_all_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = {"training": {}}
    config_module = ModuleType("src.config")
    config_module.load_config = Mock(return_value=config)  # type: ignore[attr-defined]
    model_module = ModuleType("src.model")
    model_module.train = Mock()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src.config", config_module)
    monkeypatch.setitem(sys.modules, "src.model", model_module)

    train_cli.run("custom.yaml", "cpu", 4, "output", "checkpoints")

    assert config["training"] == {
        "device": "cpu",
        "epochs": 4,
        "output_dir": "output",
        "checkpoint_dir": "checkpoints",
    }
    model_module.train.assert_called_once_with(config)  # type: ignore[attr-defined]


def test_plot_output_override_does_not_change_default_history_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    configured_output = tmp_path / "configured-output"
    configured_output.mkdir()
    (configured_output / "history.json").write_text(
        '[{"epoch": 1, "macro_f1": 0.5}]',
        encoding="utf-8",
    )
    requested_output = tmp_path / "requested-output"

    config_module = ModuleType("src.config")
    config_module.load_config = Mock(  # type: ignore[attr-defined]
        return_value={
            "training": {
                "output_dir": str(configured_output),
            }
        }
    )
    model_module = ModuleType("src.model")
    model_module.plot_history = Mock(return_value=True)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src.config", config_module)
    monkeypatch.setitem(sys.modules, "src.model", model_module)

    plot_history_cli.run("config.yaml", output_path=str(requested_output))

    model_module.plot_history.assert_called_once_with(  # type: ignore[attr-defined]
        [{"epoch": 1, "macro_f1": 0.5}],
        requested_output,
    )


def test_frontend_rag_uses_shared_top_k(monkeypatch: pytest.MonkeyPatch) -> None:
    retriever = Mock()
    retriever.retrieve.return_value = []
    retriever_module = ModuleType("src.rag.retriever")
    retriever_module.ChromaRetriever = Mock(return_value=retriever)  # type: ignore[attr-defined]
    config_module = SimpleNamespace(TOP_K=9)
    monkeypatch.setitem(sys.modules, "src.rag.retriever", retriever_module)
    monkeypatch.setitem(sys.modules, "src.rag.config", config_module)

    assert rag_service.retrieve_rag_evidence("statement") == []
    retriever.retrieve.assert_called_once_with("statement", top_k=9)


def test_custom_config_paths_resolve_from_project_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dotenv = ModuleType("dotenv")
    dotenv.load_dotenv = Mock()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "dotenv", dotenv)
    config_module = importlib.import_module("src.config")
    config_module = importlib.reload(config_module)
    config_path = tmp_path / "custom.yaml"
    config_path.write_text(
        """
data:
  train_path: datasets/train.csv
  val_path: datasets/val.csv
training:
  output_dir: outputs/model
  checkpoint_dir: checkpoints/model
model:
  base_checkpoint_dir: checkpoints/base
""".strip(),
        encoding="utf-8",
    )

    loaded = config_module.load_config(config_path)

    root = config_module.PROJECT_ROOT
    assert loaded["data"]["train_path"] == str(root / "datasets/train.csv")
    assert loaded["training"]["output_dir"] == str(root / "outputs/model")


def test_loading_config_does_not_download_datasets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src import config as config_module

    resolve_data_path = Mock()
    monkeypatch.setattr(config_module, "resolve_data_path", resolve_data_path)
    config_path = tmp_path / "model.yaml"
    config_path.write_text(
        """
data:
  huggingface:
    repo_id: owner/dataset
  train_path: datasets/missing-train.csv
  val_path: datasets/missing-val.csv
  extra_datasets:
    paths:
      - datasets/missing-extra.csv
training:
  output_dir: outputs/model
  checkpoint_dir: checkpoints/model
model: {}
""".strip(),
        encoding="utf-8",
    )

    loaded = config_module.load_config(config_path)

    resolve_data_path.assert_not_called()
    assert loaded["data"]["train_path"] == str(
        config_module.PROJECT_ROOT / "datasets/missing-train.csv"
    )


def test_missing_data_path_is_downloaded_from_huggingface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_module = importlib.import_module("src.config")
    download = Mock(return_value="/cache/not-present.csv")
    hub_module = ModuleType("huggingface_hub")
    hub_module.hf_hub_download = download  # type: ignore[attr-defined]
    errors_module = ModuleType("huggingface_hub.errors")
    errors_module.LocalEntryNotFoundError = RuntimeError  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_module)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors_module)

    resolved = config_module.resolve_data_path(
        "datasets/not-present.csv",
        {
            "repo_id": "owner/dataset",
            "revision": "v1",
            "local_dir": "datasets",
        },
    )

    assert resolved == "/cache/not-present.csv"
    download.assert_called_once_with(
        repo_id="owner/dataset",
        filename="not-present.csv",
        repo_type="dataset",
        revision="v1",
        local_files_only=True,
    )


def test_huggingface_file_map_uses_remote_parquet_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_module = importlib.import_module("src.config")
    download = Mock(return_value="/cache/train.parquet")
    hub_module = ModuleType("huggingface_hub")
    hub_module.hf_hub_download = download  # type: ignore[attr-defined]
    errors_module = ModuleType("huggingface_hub.errors")
    errors_module.LocalEntryNotFoundError = RuntimeError  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_module)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors_module)

    resolved = config_module.resolve_data_path(
        "datasets/not-present.csv",
        {
            "repo_id": "owner/dataset",
            "local_dir": "datasets",
            "files": {
                "datasets/not-present.csv": "data/train-00000-of-00001.parquet"
            },
        },
    )

    assert resolved == "/cache/train.parquet"
    assert download.call_args.kwargs["filename"] == "data/train-00000-of-00001.parquet"


def test_huggingface_file_map_accepts_normalized_absolute_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_module = importlib.import_module("src.config")
    download = Mock(return_value="/cache/train.parquet")
    hub_module = ModuleType("huggingface_hub")
    hub_module.hf_hub_download = download  # type: ignore[attr-defined]
    errors_module = ModuleType("huggingface_hub.errors")
    errors_module.LocalEntryNotFoundError = RuntimeError  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_module)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors_module)

    config_module.resolve_data_path(
        config_module.PROJECT_ROOT / "datasets/not-present.csv",
        {
            "repo_id": "owner/dataset",
            "local_dir": "datasets",
            "files": {
                "datasets/not-present.csv": "data/train-00000-of-00001.parquet"
            },
        },
    )

    assert download.call_args.kwargs["filename"] == "data/train-00000-of-00001.parquet"


def test_existing_local_data_path_takes_precedence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_module = importlib.import_module("src.config")
    local_file = tmp_path / "test.csv"
    local_file.write_text("text,label\nclaim,0\n", encoding="utf-8")
    hub_module = ModuleType("huggingface_hub")
    hub_module.hf_hub_download = Mock()  # type: ignore[attr-defined]
    errors_module = ModuleType("huggingface_hub.errors")
    errors_module.LocalEntryNotFoundError = RuntimeError  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_module)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors_module)

    resolved = config_module.resolve_data_path(
        local_file,
        {"repo_id": "owner/dataset"},
    )

    assert resolved == str(local_file)
    hub_module.hf_hub_download.assert_not_called()  # type: ignore[attr-defined]


def test_local_trained_checkpoint_takes_precedence(tmp_path: Path) -> None:
    from src.config import resolve_checkpoint_path

    checkpoint_dir = tmp_path / "checkpoints"
    best_model = checkpoint_dir / "best_model"
    best_model.mkdir(parents=True)
    for filename in ("config.json", "tokenizer_config.json", "model.safetensors"):
        (best_model / filename).touch()
    config = {
        "training": {"checkpoint_dir": str(checkpoint_dir)},
        "model": {
            "huggingface_checkpoint": {"repo_id": "owner/remote-checkpoint"}
        },
    }

    assert resolve_checkpoint_path(config) == str(best_model)


def test_missing_local_checkpoint_uses_huggingface_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src.config import resolve_checkpoint_path

    cached_model = tmp_path / "cached-model"
    cached_model.mkdir()
    for filename in ("config.json", "tokenizer_config.json", "model.safetensors"):
        (cached_model / filename).touch()
    snapshot_download = Mock(return_value=str(cached_model))
    hub_module = ModuleType("huggingface_hub")
    hub_module.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    errors_module = ModuleType("huggingface_hub.errors")
    errors_module.LocalEntryNotFoundError = RuntimeError  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_module)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors_module)
    config = {
        "training": {"checkpoint_dir": str(tmp_path / "missing")},
        "model": {
            "huggingface_checkpoint": {
                "repo_id": "owner/remote-checkpoint",
                "revision": "v1",
            }
        },
    }

    assert resolve_checkpoint_path(config) == str(cached_model)
    snapshot_download.assert_called_once_with(
        repo_id="owner/remote-checkpoint",
        repo_type="model",
        revision="v1",
        local_files_only=True,
    )
