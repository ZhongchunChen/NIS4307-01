"""Tests for local-first ChromaDB archive resolution."""

from __future__ import annotations

import importlib
import os
import sys
import zipfile
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, call

import pytest

from src.rag import config


def test_importing_retriever_does_not_resolve_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ensure_database = Mock()
    monkeypatch.setattr(config, "ensure_chroma_database", ensure_database)
    monkeypatch.setattr(config, "HF_HUB_OFFLINE", False)
    monkeypatch.setattr(config, "TRANSFORMERS_OFFLINE", False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.delitem(sys.modules, "src.rag.retriever", raising=False)

    importlib.import_module("src.rag.retriever")

    ensure_database.assert_not_called()
    assert os.getenv("HF_HUB_OFFLINE") is None
    assert os.getenv("TRANSFORMERS_OFFLINE") is None


def test_importing_build_index_does_not_open_chromadb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persistent_client = Mock(side_effect=AssertionError("must not open ChromaDB"))
    chromadb_module = ModuleType("chromadb")
    chromadb_module.PersistentClient = persistent_client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "chromadb", chromadb_module)
    monkeypatch.delitem(sys.modules, "src.rag.build_index", raising=False)

    importlib.import_module("src.rag.build_index")

    persistent_client.assert_not_called()


def test_build_index_keeps_existing_database_when_sources_are_missing(
    tmp_path: Path,
) -> None:
    from src.rag.build_index import build_index

    database = tmp_path / "database"
    database.mkdir()
    marker = database / "existing-index"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Missing RAG index source files"):
        build_index(
            database_path=database,
            fever_claim_path=tmp_path / "missing-fever.jsonl",
            train_csv_path=tmp_path / "missing-train.csv",
        )

    assert marker.read_text(encoding="utf-8") == "keep"


def test_build_index_replaces_database_only_after_staging_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src.rag import build_index as build_index_module

    database = tmp_path / "database"
    database.mkdir()
    (database / "old-index").write_text("old", encoding="utf-8")
    monkeypatch.setattr(
        build_index_module,
        "_load_source_data",
        Mock(return_value=(object(), [], object())),
    )

    def build_staged(staged_path, *_args):
        staged_path.mkdir()
        (staged_path / "chroma.sqlite3").write_text("new", encoding="utf-8")

    monkeypatch.setattr(build_index_module, "_build_staged_database", build_staged)

    result = build_index_module.build_index(database_path=database)

    assert result == str(database)
    assert not (database / "old-index").exists()
    assert (database / "chroma.sqlite3").read_text(encoding="utf-8") == "new"


def test_build_index_keeps_existing_database_when_staging_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from src.rag import build_index as build_index_module

    database = tmp_path / "database"
    database.mkdir()
    marker = database / "old-index"
    marker.write_text("old", encoding="utf-8")
    monkeypatch.setattr(
        build_index_module,
        "_load_source_data",
        Mock(return_value=(object(), [], object())),
    )

    def fail_staging(staged_path, *_args):
        staged_path.mkdir()
        (staged_path / "partial").touch()
        raise RuntimeError("embedding failed")

    monkeypatch.setattr(build_index_module, "_build_staged_database", fail_staging)

    with pytest.raises(RuntimeError, match="embedding failed"):
        build_index_module.build_index(database_path=database)

    assert marker.read_text(encoding="utf-8") == "old"


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


def test_chroma_database_cache_miss_falls_back_to_online_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class LocalCacheMiss(Exception):
        pass

    archive_path = tmp_path / "database.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("archive/data/chroma.sqlite3", "database")

    download = Mock(side_effect=[LocalCacheMiss("not cached"), str(archive_path)])
    hub_module = ModuleType("huggingface_hub")
    hub_module.hf_hub_download = download  # type: ignore[attr-defined]
    errors_module = ModuleType("huggingface_hub.errors")
    errors_module.LocalEntryNotFoundError = LocalCacheMiss  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_module)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", errors_module)
    database = tmp_path / "installed" / "data"
    expected_args = {
        "repo_id": config.CHROMA_HF_REPO_ID,
        "filename": config.CHROMA_HF_ARCHIVE,
        "repo_type": config.CHROMA_HF_REPO_TYPE,
        "revision": config.CHROMA_HF_REVISION,
    }

    assert config.ensure_chroma_database(database) == str(database)
    assert download.call_args_list == [
        call(**expected_args, local_files_only=True),
        call(**expected_args),
    ]


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
