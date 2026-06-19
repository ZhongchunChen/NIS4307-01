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
