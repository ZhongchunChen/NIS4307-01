from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class CleaningReport:
    source_rows: int
    invalid_rows_removed: int
    conflicting_rows_removed: int
    duplicate_rows_removed: int
    overlap_rows_removed: int
    final_rows: int


class TweetDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: Any,
        text_column: str,
        label_column: str,
        max_length: int,
    ) -> None:
        self.texts = dataframe[text_column].tolist()
        self.labels = dataframe[label_column].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[index],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def normalize_text(text: str) -> str:
    return " ".join(html.unescape(text).split())


def _clean_dataframe(
    dataframe: pd.DataFrame,
    source_rows: int,
    text_column: str,
    label_column: str,
    drop_duplicate_texts: bool,
    remove_conflicting_texts: bool,
) -> tuple[pd.DataFrame, CleaningReport]:
    dataframe = dataframe.dropna(subset=[text_column, label_column]).copy()
    dataframe[text_column] = dataframe[text_column].astype(str).map(normalize_text)
    dataframe[label_column] = pd.to_numeric(dataframe[label_column], errors="coerce")
    dataframe = dataframe[
        dataframe[text_column].ne("") & dataframe[label_column].isin([0, 1])
    ].copy()
    dataframe[label_column] = dataframe[label_column].astype(int)
    invalid_rows_removed = source_rows - len(dataframe)

    conflicting_rows_removed = 0
    if remove_conflicting_texts:
        label_counts = dataframe.groupby(text_column)[label_column].nunique()
        conflicting_texts = set(label_counts[label_counts > 1].index)
        conflicting_rows_removed = dataframe[text_column].isin(conflicting_texts).sum()
        dataframe = dataframe[~dataframe[text_column].isin(conflicting_texts)].copy()

    duplicate_rows_removed = 0
    if drop_duplicate_texts:
        before = len(dataframe)
        dataframe = dataframe.drop_duplicates(subset=[text_column], keep="first").copy()
        duplicate_rows_removed = before - len(dataframe)

    dataframe = dataframe.reset_index(drop=True)
    report = CleaningReport(
        source_rows=source_rows,
        invalid_rows_removed=int(invalid_rows_removed),
        conflicting_rows_removed=int(conflicting_rows_removed),
        duplicate_rows_removed=int(duplicate_rows_removed),
        overlap_rows_removed=0,
        final_rows=len(dataframe),
    )
    return dataframe, report


def _clean_split(
    path: str | Path,
    text_column: str,
    label_column: str,
    drop_duplicate_texts: bool,
    remove_conflicting_texts: bool,
) -> tuple[pd.DataFrame, CleaningReport]:
    dataframe = pd.read_csv(path)
    required_columns = {text_column, label_column}
    missing = required_columns - set(dataframe.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    return _clean_dataframe(
        dataframe,
        len(dataframe),
        text_column,
        label_column,
        drop_duplicate_texts,
        remove_conflicting_texts,
    )


def _load_extra_datasets(
    data_config: dict[str, Any],
    cleaning: dict[str, Any],
) -> tuple[list[pd.DataFrame], dict[str, Any]]:
    extra_config = data_config.get("extra_datasets", {})
    if not extra_config.get("enabled", False):
        return [], {"enabled": False, "datasets": {}}

    text_column = data_config["text_column"]
    label_column = data_config["label_column"]
    frames = []
    reports = {"enabled": True, "datasets": {}}
    for path in extra_config.get("paths", []):
        extra_path = Path(path)
        if not extra_path.exists():
            raise FileNotFoundError(f"Extra dataset does not exist: {extra_path}")
        dataframe, report = _clean_split(
            extra_path,
            text_column,
            label_column,
            cleaning["drop_duplicate_texts"],
            cleaning["remove_conflicting_texts"],
        )
        dataframe = dataframe.copy()
        dataframe["extra_dataset_path"] = str(extra_path)
        frames.append(dataframe)
        reports["datasets"][str(extra_path)] = vars(report)
    return frames, reports


def load_datasets(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    data_config = config["data"]
    text_column = data_config["text_column"]
    label_column = data_config["label_column"]
    cleaning = data_config["cleaning"]

    train_df, train_report = _clean_split(
        data_config["train_path"],
        text_column,
        label_column,
        cleaning["drop_duplicate_texts"],
        cleaning["remove_conflicting_texts"],
    )
    val_df, val_report = _clean_split(
        data_config["val_path"],
        text_column,
        label_column,
        cleaning["drop_duplicate_texts"],
        cleaning["remove_conflicting_texts"],
    )
    extra_frames, extra_report = _load_extra_datasets(data_config, cleaning)
    if extra_frames:
        train_df = pd.concat([train_df, *extra_frames], ignore_index=True)
        train_df, train_combined_report = _clean_dataframe(
            train_df,
            len(train_df),
            text_column,
            label_column,
            cleaning["drop_duplicate_texts"],
            cleaning["remove_conflicting_texts"],
        )
    else:
        train_combined_report = None

    if cleaning["remove_train_val_overlap"]:
        overlap_mask = val_df[text_column].isin(set(train_df[text_column]))
        val_report.overlap_rows_removed = int(overlap_mask.sum())

    reports = {
        "train": vars(train_report),
        "extra_datasets": extra_report,
        "train_after_extra_merge": (
            vars(train_combined_report) if train_combined_report is not None else None
        ),
        "val": vars(val_report),
        "train_label_counts": train_df[label_column].value_counts().sort_index().to_dict(),
        "val_label_counts": val_df[label_column].value_counts().sort_index().to_dict(),
    }
    return train_df, val_df, reports
