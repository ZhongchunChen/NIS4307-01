"""Compatibility wrapper for :mod:`src.cli.evaluate`."""

from src._compat import warn_legacy_module
from src.cli.evaluate import main, run

warn_legacy_module("scripts.evaluate", "src.cli.evaluate")

__all__ = ["main", "run"]


if __name__ == "__main__":
    main()
