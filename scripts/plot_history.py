"""Compatibility wrapper for :mod:`src.cli.plot_history`."""

from src._compat import warn_legacy_module
from src.cli.plot_history import main, run

warn_legacy_module("scripts.plot_history", "src.cli.plot_history")

__all__ = ["main", "run"]


if __name__ == "__main__":
    main()
