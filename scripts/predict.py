from __future__ import annotations

import argparse
import html
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import load_config
from src.utils import get_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict the label of one text.")
    parser.add_argument("--text", required=True)
    parser.add_argument("--config", default="configs/bertweet.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    training = config["training"]
    checkpoint = Path(args.checkpoint or training["checkpoint_dir"]) / (
        "" if args.checkpoint else "best_model"
    )
    device = get_device(training["device"])
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)
    model.eval()

    encoded = tokenizer(
        " ".join(html.unescape(args.text).split()),
        max_length=config["model"]["max_length"],
        truncation=True,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        probabilities = model(**encoded).logits.softmax(dim=-1).squeeze(0).cpu()

    label = int(probabilities.argmax())
    print(
        {
            "label": label,
            "probability": float(probabilities[label]),
            "probabilities": {"0": float(probabilities[0]), "1": float(probabilities[1])},
        }
    )


if __name__ == "__main__":
    main()
