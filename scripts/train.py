"""Compatibility wrapper for :mod:`src.cli.train`."""

from src._compat import warn_legacy_module
from src.cli.train import main, parse_args, run

warn_legacy_module("scripts.train", "src.cli.train")

__all__ = ["main", "parse_args", "run"]


if __name__ == "__main__":
    main()
