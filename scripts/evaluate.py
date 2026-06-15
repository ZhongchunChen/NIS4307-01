from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import load_config
from src.data import TweetDataset, load_datasets
from src.training import build_loader, evaluate
from src.utils import get_device, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained BERTweet model.")
    parser.add_argument("--config", default="configs/bertweet.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    training = config["training"]
    checkpoint = Path(args.checkpoint or training["checkpoint_dir"]) / (
        "" if args.checkpoint else "best_model"
    )
    device = get_device(training["device"])

    _, val_df, _ = load_datasets(config)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)
    dataset = TweetDataset(
        val_df,
        tokenizer,
        config["data"]["text_column"],
        config["data"]["label_column"],
        config["model"]["max_length"],
    )
    loader = build_loader(
        dataset,
        training["eval_batch_size"],
        False,
        training["num_workers"],
        device.type == "cuda",
    )
    metrics = evaluate(model, loader, device)
    save_json(metrics, Path(training["output_dir"]) / "evaluation_metrics.json")
    print(f"macro_f1={metrics['macro_f1']:.4f}; accuracy={metrics['accuracy']:.4f}")
    print(f"confusion_matrix={metrics['confusion_matrix']}")


if __name__ == "__main__":
    main()
