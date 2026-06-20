"""Compatibility wrapper for :mod:`src.rag.service`."""

from src._compat import warn_legacy_module
from src.rag.service import clear_rag_retriever_cache, retrieve_rag_evidence

warn_legacy_module("src.rag_service", "src.rag.service")

__all__ = ["clear_rag_retriever_cache", "retrieve_rag_evidence"]
