"""Compatibility wrapper for :mod:`src.web.llm_service`."""

from src._compat import warn_legacy_module
from src.web.llm_service import (
    STAGE1_PROMPT,
    STAGE2_PROMPT,
    STAGE3_CONVERGE_PROMPT,
    STAGE3_DIVERGE_PROMPT,
    analyze_root_cause,
    client,
    compare_results,
    format_rag_evidence,
    judge_statement,
)

warn_legacy_module("src.api_service", "src.web.llm_service")

__all__ = [
    "STAGE1_PROMPT",
    "STAGE2_PROMPT",
    "STAGE3_CONVERGE_PROMPT",
    "STAGE3_DIVERGE_PROMPT",
    "analyze_root_cause",
    "client",
    "compare_results",
    "format_rag_evidence",
    "judge_statement",
]
