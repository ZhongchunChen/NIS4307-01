"""Tests for temporary import and command compatibility wrappers."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

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


def test_legacy_training_plot_api_accepts_metric_and_mode(tmp_path: Path) -> None:
    sys.modules.pop("src.training", None)
    with pytest.warns(DeprecationWarning, match="src.training is deprecated"):
        legacy_training = importlib.import_module("src.training")

    history = [
        {"epoch": 1, "loss": 0.8, "macro_f1": 0.7},
        {"epoch": 2, "loss": 0.4, "macro_f1": 0.6},
    ]

    assert legacy_training.find_best_epoch(history, "loss", "min") == 2
    assert legacy_training.find_best_epoch(history) == 1
    assert legacy_training.plot_history(history, tmp_path, "loss", "min") is True
    assert (tmp_path / "plots" / "loss_curve.png").is_file()
