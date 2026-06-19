"""Tests for local-first ChromaDB archive resolution."""

from __future__ import annotations

import importlib
import sys
import zipfile
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest

from src.rag import config


def test_importing_retriever_does_not_resolve_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ensure_database = Mock()
    monkeypatch.setattr(config, "ensure_chroma_database", ensure_database)
    monkeypatch.delitem(sys.modules, "src.rag.retriever", raising=False)

    importlib.import_module("src.rag.retriever")

    ensure_database.assert_not_called()


def test_pheme_numeric_labels_match_classifier_semantics() -> None:
    from src.rag.retriever import _normalize_label

    assert config.PHEME_LABELS == {0: "non-rumor", 1: "rumor"}
    assert _normalize_label(0) == "real"
    assert _normalize_label(1) == "fake"
    assert _normalize_label(config.PHEME_LABELS[0]) == "real"
    assert _normalize_label(config.PHEME_LABELS[1]) == "fake"


def test_rag_cli_help_does_not_resolve_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.rag import cli

    run_once = Mock()
    monkeypatch.setattr(cli, "run_once", run_once)

    with pytest.raises(SystemExit) as error:
        cli.main(["rag-demo", "--help"])

    assert error.value.code == 0
    run_once.assert_not_called()


def test_existing_chroma_database_takes_precedence(tmp_path: Path) -> None:
    database = tmp_path / "data"
    database.mkdir()
    (database / "chroma.sqlite3").touch()

    assert config.ensure_chroma_database(database) == str(database)


def test_missing_chroma_database_is_downloaded_and_extracted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "ChromaDB_data_populate.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "ChromaDB_data_populate/DataBase/data/chroma.sqlite3",
            "database",
        )
        archive.writestr(
            "ChromaDB_data_populate/DataBase/data/index/segment.bin",
            "segment",
        )

    download = Mock(return_value=str(archive_path))
    hub_module = ModuleType("huggingface_hub")
    hub_module.hf_hub_download = download  # type: ignore[attr-defined]
    errors_module = ModuleType("huggingface_hub.errors")
    errors_module.LocalEntryNotFoundError = RuntimeError  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_module)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors_module)
    monkeypatch.setattr(config, "CHROMA_HF_REPO_ID", "owner/chromadb")
    monkeypatch.setattr(config, "CHROMA_HF_REPO_TYPE", "dataset")
    monkeypatch.setattr(config, "CHROMA_HF_REVISION", "v1")
    monkeypatch.setattr(config, "CHROMA_HF_ARCHIVE", archive_path.name)
    database = tmp_path / "installed" / "DataBase" / "data"

    assert config.ensure_chroma_database(database) == str(database)
    assert (database / "chroma.sqlite3").read_text(encoding="utf-8") == "database"
    assert (database / "index" / "segment.bin").read_text(encoding="utf-8") == "segment"
    download.assert_called_once_with(
        repo_id="owner/chromadb",
        filename="ChromaDB_data_populate.zip",
        repo_type="dataset",
        revision="v1",
        local_files_only=True,
    )


def test_rag_cli_reports_database_failure_as_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.rag import cli

    monkeypatch.setattr(
        cli,
        "ChromaRetriever",
        Mock(side_effect=RuntimeError("database unavailable")),
    )

    result = cli.run_once("claim")

    assert result["label"] == "unavailable"
    assert result["confidence"] == 0.0
    assert "database unavailable" in result["reason"]


def test_rag_cli_returns_failure_status_for_unavailable_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.rag import cli

    monkeypatch.setattr(
        cli,
        "run_once",
        Mock(
            return_value={
                "label": "unavailable",
                "confidence": 0.0,
                "reason": "database unavailable",
                "evidence": [],
            }
        ),
    )

    assert cli.main(["rag-demo", "claim"]) == 1


def test_rag_judge_without_llm_or_fallback_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.rag import llm_judge

    monkeypatch.setattr(config, "OPENAI_API_KEY", "")
    monkeypatch.setattr(config, "FALLBACK_MAJORITY_VOTE", False)

    result = llm_judge.judge("claim", [])

    assert result["label"] == "unavailable"
    assert result["confidence"] == 0.0
