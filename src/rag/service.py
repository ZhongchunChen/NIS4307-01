from __future__ import annotations

from typing import Any


def retrieve_rag_evidence(statement: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """
    Optional RAG evidence retrieval.

    If the RAG database or dependencies are unavailable, return an empty list
    so the main system can still run.
    """
    try:
        from src.rag import config
        from src.rag.retriever import ChromaRetriever

        retriever = ChromaRetriever()
        evidence = retriever.retrieve(
            statement,
            top_k=config.TOP_K if top_k is None else top_k,
        )
        print(f"[RAG] retrieved {len(evidence)} evidence items")
        return evidence

    except Exception as e:
        print(f"[RAG disabled] {e}")
        return []
