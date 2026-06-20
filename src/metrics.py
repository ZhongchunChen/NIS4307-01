"""Compatibility wrapper for :mod:`src.model.metrics`."""

from src._compat import warn_legacy_module
from src.model.metrics import classification_metrics

warn_legacy_module("src.metrics", "src.model.metrics")

__all__ = ["classification_metrics"]
