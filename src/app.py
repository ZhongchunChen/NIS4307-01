"""Compatibility wrapper for :mod:`src.web.app`."""

from src._compat import warn_legacy_module
from src.web.app import create_app, create_runtime_app

warn_legacy_module("src.app", "src.web.app")

__all__ = ["create_app", "create_runtime_app"]
