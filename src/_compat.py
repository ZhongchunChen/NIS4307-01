"""Helpers for temporary backwards-compatible module aliases."""

from __future__ import annotations

import warnings


def warn_legacy_module(legacy: str, replacement: str) -> None:
    warnings.warn(
        f"{legacy} is deprecated; import from {replacement} instead.",
        DeprecationWarning,
        stacklevel=3,
    )
