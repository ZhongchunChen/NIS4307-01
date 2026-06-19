"""Compatibility wrapper for :mod:`src.model.utils`."""

from src._compat import warn_legacy_module
from src.model.utils import get_device, save_json, set_seed

warn_legacy_module("src.utils", "src.model.utils")

__all__ = ["get_device", "save_json", "set_seed"]
