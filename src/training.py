"""Compatibility wrapper for :mod:`src.model.pipeline`."""

from src._compat import warn_legacy_module
from src.model.pipeline import (
    build_loader,
    ensure_base_checkpoint,
    evaluate,
    find_best_epoch,
    plot_history,
    train,
)

warn_legacy_module("src.training", "src.model.pipeline")

__all__ = [
    "build_loader",
    "ensure_base_checkpoint",
    "evaluate",
    "find_best_epoch",
    "plot_history",
    "train",
]
