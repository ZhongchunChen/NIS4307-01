"""Build the optional ChromaDB index without modifying a live index in place."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from src.rag import config


GOSSIPCOP_SPLITS = {
    "HF": "data/HF-00000-of-00001-b7ad0013efd98ff4.parquet",
    "HR": "data/HR-00000-of-00001-043a35ac2a425b62.parquet",
}
GOSSIPCOP_LABELS = {"HF": "fake", "HR": "real"}
GOSSIPCOP_HF_REPO = "hf://datasets/Jinyan1/GossipCop/"
FEVER_CLAIM_JSONL = config.RAG_DATA_ROOT / "shared_task_dev.jsonl"
TRAIN_CSV = config.RAG_DATA_ROOT / "train.csv"
CHROMA_DB_DIR = Path(config.CHROMA_DB_PATH)
CHROMA_BATCH_SIZE = 5000


def _configure_huggingface_environment() -> None:
    if config.HF_ENDPOINT:
        os.environ.setdefault("HF_ENDPOINT", config.HF_ENDPOINT)
    if config.HF_HUB_OFFLINE:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if config.TRANSFORMERS_OFFLINE:
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _validate_local_sources(fever_claim_path: Path, train_csv_path: Path) -> None:
    missing = [
        str(path)
        for path in (fever_claim_path, train_csv_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing RAG index source files: {', '.join(missing)}")


def _load_source_data(
    fever_claim_path: Path,
    train_csv_path: Path,
    gossipcop_repo: str,
) -> tuple[Any, list[dict[str, str]], Any]:
    """Load and validate every source before a staged database is created."""
    _validate_local_sources(fever_claim_path, train_csv_path)

    import pandas as pd

    print("Loading GossipCop datasets...")
    gossip_frames = []
    for split_name, split_path in GOSSIPCOP_SPLITS.items():
        frame = pd.read_parquet(gossipcop_repo + split_path)
        missing_columns = {"title", "text"} - set(frame.columns)
        if missing_columns:
            raise ValueError(
                f"GossipCop {split_name} is missing columns: {sorted(missing_columns)}"
            )
        frame = frame.copy()
        frame["content"] = (
            frame["title"].fillna("").astype(str)
            + ". "
            + frame["text"].fillna("").astype(str)
        ).str.strip(" .")
        frame["label"] = GOSSIPCOP_LABELS[split_name]
        gossip_frames.append(frame.loc[frame["content"].ne(""), ["content", "label"]])
    gossipcop = pd.concat(gossip_frames, ignore_index=True)

    print("Loading FEVER claims...")
    fever_docs: list[dict[str, str]] = []
    with fever_claim_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {fever_claim_path} at line {line_number}"
                ) from exc
            claim = str(item.get("claim") or "").strip()
            if not claim:
                continue
            raw_label = str(item.get("label") or "").strip().upper()
            if raw_label in {"SUPPORTED", "SUPPORTS"}:
                label = "real"
            elif raw_label in {"REFUTED", "REFUTES"}:
                label = "fake"
            else:
                label = "nei"
            fever_docs.append({"content": claim, "label": label})

    print("Loading classifier training data...")
    train = pd.read_csv(train_csv_path)
    missing_columns = {"text", "label"} - set(train.columns)
    if missing_columns:
        raise ValueError(
            f"{train_csv_path} is missing columns: {sorted(missing_columns)}"
        )
    train = train.copy()
    train["content"] = train["text"].fillna("").astype(str).str.strip()
    numeric_labels = pd.to_numeric(train["label"], errors="coerce")
    invalid_labels = ~numeric_labels.isin(config.PHEME_LABELS)
    if invalid_labels.any():
        values = sorted({str(value) for value in train.loc[invalid_labels, "label"]})
        raise ValueError(f"{train_csv_path} contains invalid labels: {values}")
    train["label_str"] = numeric_labels.astype(int).map(config.PHEME_LABELS)
    train["event_str"] = (
        train["event"].fillna("").astype(str) if "event" in train.columns else ""
    )
    train = train.loc[train["content"].ne("")].reset_index(drop=True)
    return gossipcop, fever_docs, train


def _add_batches(
    collection: Any,
    documents: list[str],
    metadatas: list[dict[str, str]],
    id_prefix: str,
    description: str,
    batch_size: int,
) -> None:
    from tqdm import tqdm

    for start in tqdm(range(0, len(documents), batch_size), desc=description):
        stop = min(start + batch_size, len(documents))
        collection.add(
            documents=documents[start:stop],
            metadatas=metadatas[start:stop],
            ids=[f"{id_prefix}_{index}" for index in range(start, stop)],
        )


def _close_client(client: Any) -> None:
    """Release Chroma resources before the staged directory is renamed."""
    system = getattr(client, "_system", None)
    stop = getattr(system, "stop", None)
    if callable(stop):
        try:
            stop()
        except Exception:
            pass
    del client
    gc.collect()


def _build_staged_database(
    staged_path: Path,
    gossipcop: Any,
    fever_docs: list[dict[str, str]],
    train: Any,
    batch_size: int,
) -> None:
    import chromadb
    from chromadb.utils import embedding_functions

    embedding_fn = embedding_functions.DefaultEmbeddingFunction()
    client = chromadb.PersistentClient(path=str(staged_path))
    try:
        gossipcop_collection = client.create_collection(
            name="gossipcop",
            embedding_function=embedding_fn,
        )
        fever_collection = client.create_collection(
            name="fever_claims",
            embedding_function=embedding_fn,
        )
        train_collection = client.create_collection(
            name="train",
            embedding_function=embedding_fn,
        )

        _add_batches(
            gossipcop_collection,
            gossipcop["content"].tolist(),
            [{"label": label} for label in gossipcop["label"].tolist()],
            "gossipcop",
            "Adding GossipCop to ChromaDB",
            batch_size,
        )
        _add_batches(
            fever_collection,
            [document["content"] for document in fever_docs],
            [{"label": document["label"]} for document in fever_docs],
            "fever",
            "Adding FEVER claims to ChromaDB",
            batch_size,
        )
        _add_batches(
            train_collection,
            train["content"].tolist(),
            [
                {"label": label, "event": event}
                for label, event in zip(train["label_str"], train["event_str"])
            ],
            "train",
            "Adding training data to ChromaDB",
            batch_size,
        )
    finally:
        _close_client(client)

    if not config.is_chroma_database(staged_path):
        raise RuntimeError(f"Staged ChromaDB is incomplete at {staged_path}")


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _replace_database(staged_path: Path, destination: Path) -> None:
    """Install a staged database and restore the previous one on failure."""
    backup_path = destination.parent / (
        f".{destination.name}.backup-{uuid.uuid4().hex}"
    )
    had_existing_database = destination.exists() or destination.is_symlink()

    try:
        if had_existing_database:
            os.replace(destination, backup_path)
        os.replace(staged_path, destination)
    except Exception:
        if backup_path.exists() and not destination.exists():
            os.replace(backup_path, destination)
        raise

    if backup_path.exists() or backup_path.is_symlink():
        try:
            _remove_path(backup_path)
        except OSError as exc:
            print(f"Warning: unable to remove old ChromaDB backup {backup_path}: {exc}")


def build_index(
    database_path: str | Path = CHROMA_DB_DIR,
    fever_claim_path: str | Path = FEVER_CLAIM_JSONL,
    train_csv_path: str | Path = TRAIN_CSV,
    gossipcop_repo: str = GOSSIPCOP_HF_REPO,
    batch_size: int = CHROMA_BATCH_SIZE,
) -> str:
    """Build a complete index off to the side, then replace the live database."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    _configure_huggingface_environment()
    destination = Path(database_path).expanduser()
    fever_path = Path(fever_claim_path).expanduser()
    train_path = Path(train_csv_path).expanduser()
    gossipcop, fever_docs, train = _load_source_data(
        fever_path,
        train_path,
        gossipcop_repo,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.staging-",
        dir=destination.parent,
    ) as temporary_dir:
        staged_path = Path(temporary_dir) / "database"
        _build_staged_database(
            staged_path,
            gossipcop,
            fever_docs,
            train,
            batch_size,
        )
        _replace_database(staged_path, destination)

    return str(destination)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the optional ChromaDB RAG index."
    )
    parser.add_argument("--database", default=str(CHROMA_DB_DIR))
    parser.add_argument("--fever-claims", default=str(FEVER_CLAIM_JSONL))
    parser.add_argument("--train-csv", default=str(TRAIN_CSV))
    parser.add_argument("--gossipcop-repo", default=GOSSIPCOP_HF_REPO)
    parser.add_argument("--batch-size", type=int, default=CHROMA_BATCH_SIZE)
    args = parser.parse_args()

    installed_path = build_index(
        database_path=args.database,
        fever_claim_path=args.fever_claims,
        train_csv_path=args.train_csv,
        gossipcop_repo=args.gossipcop_repo,
        batch_size=args.batch_size,
    )
    print(f"ChromaDB database built successfully at: {installed_path}")


if __name__ == "__main__":
    main()
