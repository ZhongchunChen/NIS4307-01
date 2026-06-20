from __future__ import annotations

import threading
from typing import Any


_retriever: Any | None = None
_retriever_lock = threading.Lock()


def _get_retriever() -> Any:
    global _retriever

    if _retriever is None:
        with _retriever_lock:
            if _retriever is None:
                from src.rag.retriever import ChromaRetriever

                _retriever = ChromaRetriever()
    return _retriever


def clear_rag_retriever_cache() -> None:
    """Clear the process-local retriever cache, primarily for reconfiguration/tests."""
    global _retriever

    with _retriever_lock:
        _retriever = None


def retrieve_rag_evidence(statement: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """
    Optional RAG evidence retrieval.

    If the RAG database or dependencies are unavailable, return an empty list
    so the main system can still run.
    """
    try:
        from src.rag import config

        retriever = _get_retriever()
        evidence = retriever.retrieve(
            statement,
            top_k=config.TOP_K if top_k is None else top_k,
        )
        print(f"[RAG] retrieved {len(evidence)} evidence items")
        return evidence

    except Exception as e:
        print(f"[RAG disabled] {e}")
        return []
