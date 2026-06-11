from __future__ import annotations

from typing import Any


def retrieve_rag_evidence(statement: str, top_k: int = 3) -> list[dict[str, Any]]:
    """
    Optional RAG evidence retrieval.

    If the RAG database or dependencies are unavailable, return an empty list
    so the main system can still run.
    """
    try:
        import sys
        from pathlib import Path

        project_root = Path(__file__).resolve().parents[1]
        rag_dir = project_root / "RAG"

        if str(rag_dir) not in sys.path:
            sys.path.insert(0, str(rag_dir))

        from chroma_retriever import ChromaRetriever

        retriever = ChromaRetriever()
        evidence = retriever.retrieve(statement, top_k=top_k)
        print(f"[RAG] retrieved {len(evidence)} evidence items")
        return evidence

    except Exception as e:
        print(f"[RAG disabled] {e}")
        return []