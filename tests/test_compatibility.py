"""Tests for temporary import and command compatibility wrappers."""

from __future__ import annotations

import importlib
import sys

import pytest


@pytest.mark.parametrize(
    ("legacy_module", "canonical_module", "symbol"),
    [
        ("src.app", "src.web.app", "create_app"),
        ("src.api_service", "src.web.llm_service", "judge_statement"),
        ("src.rag_service", "src.rag.service", "retrieve_rag_evidence"),
        ("src.training", "src.model.pipeline", "train"),
        ("src.data", "src.model.data", "TweetDataset"),
        ("src.metrics", "src.model.metrics", "classification_metrics"),
        ("src.utils", "src.model.utils", "get_device"),
        ("scripts.train", "src.cli.train", "main"),
        ("scripts.evaluate", "src.cli.evaluate", "main"),
        ("scripts.predict", "src.cli.predict", "main"),
        ("scripts.plot_history", "src.cli.plot_history", "main"),
        (
            "scripts.prepare_extra_datasets",
            "src.cli.prepare_extra_datasets",
            "main",
        ),
    ],
)
def test_legacy_module_forwards_public_symbol(
    legacy_module: str,
    canonical_module: str,
    symbol: str,
) -> None:
    canonical = importlib.import_module(canonical_module)
    sys.modules.pop(legacy_module, None)

    with pytest.warns(DeprecationWarning, match=f"{legacy_module} is deprecated"):
        legacy = importlib.import_module(legacy_module)

    assert getattr(legacy, symbol) is getattr(canonical, symbol)
