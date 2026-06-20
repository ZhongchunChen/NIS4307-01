from __future__ import annotations

import argparse
from pathlib import Path


def run(
    config_path: str,
    test_path: str | None = None,
    checkpoint_path: str | None = None,
    output_path: str | None = None,
    device_name: str | None = None,
) -> None:
    import pandas as pd
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from src.config import load_config, resolve_checkpoint_path, resolve_data_path
    from src.model import (
        TweetDataset,
        build_loader,
        evaluate,
        get_device,
        load_datasets,
        normalize_text,
        read_dataframe,
        save_json,
    )

    config = load_config(config_path)
    training = config["training"]
    checkpoint = resolve_checkpoint_path(config, checkpoint_path)
    device = get_device(device_name or training["device"])

    text_column = config["data"]["text_column"]
    label_column = config["data"]["label_column"]
    if test_path:
        resolved_test_path = resolve_data_path(
            test_path,
            config["data"].get("huggingface"),
        )
        eval_df = read_dataframe(resolved_test_path)
        missing = {text_column, label_column} - set(eval_df.columns)
        if missing:
            raise ValueError(
                f"{resolved_test_path} is missing required columns: {sorted(missing)}"
            )
        eval_df = eval_df.dropna(subset=[text_column, label_column]).copy()
        eval_df[text_column] = eval_df[text_column].astype(str).map(normalize_text)
        eval_df[label_column] = pd.to_numeric(eval_df[label_column], errors="coerce")
        eval_df = eval_df[eval_df[text_column].ne("") & eval_df[label_column].isin([0, 1])]
        eval_df[label_column] = eval_df[label_column].astype(int)
        default_output = Path(training["output_dir"]) / "test_metrics.json"
    else:
        _, eval_df, _ = load_datasets(config)
        default_output = Path(training["output_dir"]) / "evaluation_metrics.json"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)
    dataset = TweetDataset(
        eval_df,
        tokenizer,
        text_column,
        label_column,
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
    metrics_path = Path(output_path or default_output)
    save_json(metrics, metrics_path)
    print(f"macro_f1={metrics['macro_f1']:.4f}; accuracy={metrics['accuracy']:.4f}")
    print(f"confusion_matrix={metrics['confusion_matrix']}")
    print(f"Saved metrics to {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained BERTweet model.")
    parser.add_argument("--config", default="configs/bertweet.yaml")
    parser.add_argument("--test", default=None, help="Evaluate a labeled CSV file.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    run(args.config, args.test, args.checkpoint, args.output, args.device)


if __name__ == "__main__":
    main()
