"""Compatibility wrapper for :mod:`src.cli.prepare_extra_datasets`."""

from src._compat import warn_legacy_module
from src.cli.prepare_extra_datasets import (
    FEVER_LABEL,
    GOSSIPCOP_LABEL,
    GOSSIPCOP_NAME_PATTERN,
    GOSSIPCOP_SOURCE,
    clean_binary_dataframe,
    combine_text_fields,
    convert_gossipcop,
    convert_shared_task,
    main,
    parse_args,
    run,
)

warn_legacy_module(
    "scripts.prepare_extra_datasets",
    "src.cli.prepare_extra_datasets",
)

__all__ = [
    "FEVER_LABEL",
    "GOSSIPCOP_LABEL",
    "GOSSIPCOP_NAME_PATTERN",
    "GOSSIPCOP_SOURCE",
    "clean_binary_dataframe",
    "combine_text_fields",
    "convert_gossipcop",
    "convert_shared_task",
    "main",
    "parse_args",
    "run",
]


if __name__ == "__main__":
    main()
