"""Compatibility wrapper for :mod:`src.cli.predict`."""

from src._compat import warn_legacy_module
from src.cli.predict import main, run

warn_legacy_module("scripts.predict", "src.cli.predict")

__all__ = ["main", "run"]


if __name__ == "__main__":
    main()
