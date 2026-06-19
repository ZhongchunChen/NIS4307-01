from __future__ import annotations

import argparse
from typing import Any


def run(
    text: str,
    config_path: str,
    checkpoint_path: str | None = None,
    device_name: str | None = None,
    json_output: bool = False,
) -> dict[str, Any]:
    import html
    import json
    from pathlib import Path

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from src.config import load_config
    from src.model import get_device

    config = load_config(config_path)
    training = config["training"]
    checkpoint = Path(checkpoint_path or training["checkpoint_dir"]) / (
        "" if checkpoint_path else "best_model"
    )
    device = get_device(device_name or training["device"])
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)
    model.eval()

    encoded = tokenizer(
        " ".join(html.unescape(text).split()),
        max_length=config["model"]["max_length"],
        truncation=True,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        probabilities = model(**encoded).logits.softmax(dim=-1).squeeze(0).cpu()

    label = int(probabilities.argmax())
    result = {
        "label": label,
        "probability": float(probabilities[label]),
        "probabilities": {"0": float(probabilities[0]), "1": float(probabilities[1])},
    }
    print(json.dumps(result, ensure_ascii=False) if json_output else result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict the label of one text.")
    parser.add_argument("--text", required=True)
    parser.add_argument("--config", default="configs/bertweet.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    run(args.text, args.config, args.checkpoint, args.device, args.json)


if __name__ == "__main__":
    main()
