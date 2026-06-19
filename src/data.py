"""Compatibility wrapper for :mod:`src.model.data`."""

from src._compat import warn_legacy_module
from src.model.data import (
    CleaningReport,
    TweetDataset,
    load_datasets,
    normalize_text,
    read_dataframe,
)

warn_legacy_module("src.data", "src.model.data")

__all__ = [
    "CleaningReport",
    "TweetDataset",
    "load_datasets",
    "normalize_text",
    "read_dataframe",
]
