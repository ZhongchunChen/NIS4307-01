"""BERTweet model package for training, evaluation, and inference."""

from src.model.data import CleaningReport, TweetDataset, load_datasets, normalize_text
from src.model.inference import classify_statement, configure_inference
from src.model.pipeline import (
    build_loader,
    ensure_base_checkpoint,
    evaluate,
    find_best_epoch,
    plot_history,
    train,
)
from src.model.utils import get_device, save_json, set_seed

__all__ = [
    "CleaningReport",
    "TweetDataset",
    "build_loader",
    "classify_statement",
    "configure_inference",
    "ensure_base_checkpoint",
    "evaluate",
    "find_best_epoch",
    "get_device",
    "load_datasets",
    "normalize_text",
    "plot_history",
    "save_json",
    "set_seed",
    "train",
]
