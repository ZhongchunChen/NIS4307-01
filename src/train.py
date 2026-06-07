from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

from src.config import load_config
from src.data import TweetDataset, load_datasets
from src.metrics import classification_metrics
from src.utils import get_device, save_json, set_seed


def build_loader(
    dataset: TweetDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    losses: list[float] = []
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            losses.append(outputs.loss.item())
            all_logits.append(outputs.logits.detach().cpu().numpy())
            all_labels.append(batch["labels"].detach().cpu().numpy())

    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels)
    predictions = logits.argmax(axis=-1)
    metrics = classification_metrics(labels, predictions)
    metrics["loss"] = float(np.mean(losses))
    return metrics


def _has_local_base_checkpoint(checkpoint_dir: Path) -> bool:
    model_file_exists = any(
        (checkpoint_dir / filename).exists()
        for filename in ("model.safetensors", "pytorch_model.bin")
    )
    required_files = ("config.json", "tokenizer_config.json", "vocab.txt", "bpe.codes")
    return model_file_exists and all(
        (checkpoint_dir / filename).exists() for filename in required_files
    )


def ensure_base_checkpoint(model_config: dict[str, Any]) -> str:
    pretrained_name = model_config["pretrained_name"]
    base_checkpoint_value = model_config.get("base_checkpoint_dir")
    if not base_checkpoint_value:
        return pretrained_name
    base_checkpoint_dir = Path(base_checkpoint_value).expanduser()

    if _has_local_base_checkpoint(base_checkpoint_dir):
        print(f"Loading pretrained base from local checkpoint: {base_checkpoint_dir}")
        return str(base_checkpoint_dir)

    print(
        f"Local base checkpoint not found or incomplete: {base_checkpoint_dir}\n"
        f"Downloading pretrained base from Hugging Face: {pretrained_name}"
    )
    base_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_name,
        use_fast=model_config["use_fast_tokenizer"],
        normalization=model_config["tokenizer_normalization"],
    )
    base_model = AutoModel.from_pretrained(pretrained_name)
    tokenizer.save_pretrained(base_checkpoint_dir)
    base_model.save_pretrained(base_checkpoint_dir)
    print(f"Saved pretrained base checkpoint to: {base_checkpoint_dir}")
    return str(base_checkpoint_dir)


def _plot_metric(
    history: list[dict[str, Any]],
    output_path: Path,
    metric_keys: list[str],
    title: str,
    ylabel: str,
    plt: Any,
) -> None:
    epochs = [record["epoch"] for record in history]
    plt.figure(figsize=(8, 5))
    plotted = False
    for key in metric_keys:
        values = [record[key] for record in history if key in record]
        if len(values) == len(epochs):
            plt.plot(epochs, values, marker="o", label=key)
            plotted = True
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    if plotted:
        plt.legend()
    else:
        plt.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_history(history: list[dict[str, Any]], output_dir: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skip training curve plots.")
        return False

    plots_dir = output_dir / "plots"
    _plot_metric(
        history,
        plots_dir / "loss_curve.png",
        ["train_loss", "loss"],
        "Training and Validation Loss",
        "loss",
        plt,
    )
    _plot_metric(
        history,
        plots_dir / "accuracy_curve.png",
        ["accuracy"],
        "Validation Accuracy",
        "accuracy",
        plt,
    )
    _plot_metric(
        history,
        plots_dir / "macro_f1_curve.png",
        ["macro_f1"],
        "Validation Macro-F1",
        "macro_f1",
        plt,
    )
    _plot_metric(
        history,
        plots_dir / "grad_norm_curve.png",
        ["avg_grad_norm", "max_grad_norm"],
        "Gradient Norm",
        "grad_norm",
        plt,
    )
    return True


def train(config: dict[str, Any]) -> None:
    training = config["training"]
    model_config = config["model"]
    output_dir = Path(training["output_dir"])
    checkpoint_dir = Path(training["checkpoint_dir"])
    best_model_dir = checkpoint_dir / "best_model"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config["seed"])
    device = get_device(training["device"])
    use_amp = bool(training["fp16"] and device.type == "cuda")
    print(f"Device: {device}; mixed precision: {use_amp}")

    train_df, val_df, cleaning_report = load_datasets(config)
    print(yaml.safe_dump(cleaning_report, allow_unicode=True, sort_keys=False))
    save_json(cleaning_report, output_dir / "cleaning_report.json")

    pretrained_source = ensure_base_checkpoint(model_config)
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_source,
        use_fast=model_config["use_fast_tokenizer"],
        normalization=model_config["tokenizer_normalization"],
    )
    train_dataset = TweetDataset(
        train_df,
        tokenizer,
        config["data"]["text_column"],
        config["data"]["label_column"],
        model_config["max_length"],
    )
    val_dataset = TweetDataset(
        val_df,
        tokenizer,
        config["data"]["text_column"],
        config["data"]["label_column"],
        model_config["max_length"],
    )
    train_loader = build_loader(
        train_dataset,
        training["train_batch_size"],
        True,
        training["num_workers"],
        device.type == "cuda",
    )
    val_loader = build_loader(
        val_dataset,
        training["eval_batch_size"],
        False,
        training["num_workers"],
        device.type == "cuda",
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_source,
        num_labels=2,
        id2label={0: "LABEL_0", 1: "LABEL_1"},
        label2id={"LABEL_0": 0, "LABEL_1": 1},
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )

    updates_per_epoch = math.ceil(
        len(train_loader) / training["gradient_accumulation_steps"]
    )
    total_steps = updates_per_epoch * training["epochs"]
    warmup_steps = int(total_steps * training["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: list[dict[str, Any]] = []
    best_score = -float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, training["epochs"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        grad_norms: list[float] = []
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{training['epochs']}")

        for step, batch in enumerate(progress, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(**batch)
                loss = outputs.loss / training["gradient_accumulation_steps"]

            scaler.scale(loss).backward()
            running_loss += loss.item() * training["gradient_accumulation_steps"]

            should_update = (
                step % training["gradient_accumulation_steps"] == 0
                or step == len(train_loader)
            )
            if should_update:
                scaler.unscale_(optimizer)
                grad_norm = clip_grad_norm_(model.parameters(), training["max_grad_norm"])
                grad_norms.append(float(grad_norm.detach().cpu()))
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            progress.set_postfix(loss=f"{running_loss / step:.4f}")

        metrics = evaluate(model, val_loader, device)
        epoch_record = {
            "epoch": epoch,
            "train_loss": running_loss / len(train_loader),
            "avg_grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
            "max_grad_norm": float(np.max(grad_norms)) if grad_norms else 0.0,
            **metrics,
        }
        history.append(epoch_record)
        save_json(history, output_dir / "history.json")
        plot_history(history, output_dir)
        print(
            f"Epoch {epoch}: train_loss={epoch_record['train_loss']:.4f}, "
            f"val_loss={metrics['loss']:.4f}, macro_f1={metrics['macro_f1']:.4f}, "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"avg_grad_norm={epoch_record['avg_grad_norm']:.4f}"
        )

        score = float(metrics[training["early_stopping_metric"]])
        if score > best_score:
            best_score = score
            epochs_without_improvement = 0
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            save_json(metrics, output_dir / "best_metrics.json")
            shutil.copy2(config["_config_path"], output_dir / "config.yaml")
            shutil.copy2(config["_config_path"], checkpoint_dir / "config.yaml")
            print(f"Saved best model with {training['early_stopping_metric']}={score:.4f}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= training["early_stopping_patience"]:
                print("Early stopping.")
                break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune BERTweet for binary classification.")
    parser.add_argument("--config", default="configs/bertweet.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
