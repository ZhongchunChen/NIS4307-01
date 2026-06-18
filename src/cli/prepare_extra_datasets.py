from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.model import normalize_text, save_json


GOSSIPCOP_NAME_PATTERN = re.compile(r"gossipcop_([hm])([rf])\.parquet$", re.IGNORECASE)
GOSSIPCOP_SOURCE = {"h": "human", "m": "machine"}
GOSSIPCOP_LABEL = {"r": 0, "f": 1}
FEVER_LABEL = {"SUPPORTS": 0, "REFUTES": 1}


def clean_binary_dataframe(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_rows = len(dataframe)
    dataframe = dataframe.dropna(subset=["text", "label"]).copy()
    dataframe["text"] = dataframe["text"].astype(str).map(normalize_text)
    dataframe["label"] = pd.to_numeric(dataframe["label"], errors="coerce")
    dataframe = dataframe[dataframe["text"].ne("") & dataframe["label"].isin([0, 1])].copy()
    dataframe["label"] = dataframe["label"].astype(int)
    invalid_rows_removed = source_rows - len(dataframe)

    label_counts = dataframe.groupby("text")["label"].nunique()
    conflicting_texts = set(label_counts[label_counts > 1].index)
    conflicting_rows_removed = int(dataframe["text"].isin(conflicting_texts).sum())
    dataframe = dataframe[~dataframe["text"].isin(conflicting_texts)].copy()

    before_dedup = len(dataframe)
    dataframe = dataframe.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    duplicate_rows_removed = before_dedup - len(dataframe)

    report = {
        "source_rows": int(source_rows),
        "invalid_rows_removed": int(invalid_rows_removed),
        "conflicting_rows_removed": int(conflicting_rows_removed),
        "duplicate_rows_removed": int(duplicate_rows_removed),
        "final_rows": int(len(dataframe)),
        "label_counts": dataframe["label"].value_counts().sort_index().to_dict(),
    }
    return dataframe, report


def combine_text_fields(row: pd.Series, fields: list[str]) -> str:
    parts = []
    for field in fields:
        value = row.get(field)
        if pd.notna(value):
            text = normalize_text(str(value))
            if text:
                parts.append(text)
    return " ".join(parts)


def convert_gossipcop(
    input_dir: Path,
    output_path: Path,
    text_fields: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames = []
    file_reports: dict[str, Any] = {}
    for path in sorted(input_dir.glob("gossipcop_*.parquet")):
        match = GOSSIPCOP_NAME_PATTERN.match(path.name)
        if not match:
            continue

        source_code, label_code = match.groups()
        source = GOSSIPCOP_SOURCE[source_code.lower()]
        label = GOSSIPCOP_LABEL[label_code.lower()]
        raw = pd.read_parquet(path)
        missing_fields = [field for field in text_fields if field not in raw.columns]
        if missing_fields:
            raise ValueError(f"{path} is missing text fields: {missing_fields}")

        converted = pd.DataFrame(
            {
                "id": raw["id"].astype(str) if "id" in raw.columns else [f"{path.stem}-{i}" for i in range(len(raw))],
                "text": raw.apply(lambda row: combine_text_fields(row, text_fields), axis=1),
                "label": label,
                "source": f"gossipcop_{source}",
                "subset": path.stem,
                "original_label": label_code.upper(),
            }
        )
        cleaned, report = clean_binary_dataframe(converted)
        frames.append(cleaned)
        file_reports[path.name] = report

    if frames:
        output = pd.concat(frames, ignore_index=True)
        output, combined_report = clean_binary_dataframe(output)
    else:
        output = pd.DataFrame(columns=["id", "text", "label", "source", "subset", "original_label"])
        combined_report = {
            "source_rows": 0,
            "invalid_rows_removed": 0,
            "conflicting_rows_removed": 0,
            "duplicate_rows_removed": 0,
            "final_rows": 0,
            "label_counts": {},
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output, {"files": file_reports, "combined": combined_report}


def convert_shared_task(input_path: Path, output_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    skipped_labels: dict[str, int] = {}
    with input_path.open("r", encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            original_label = record.get("label")
            if original_label not in FEVER_LABEL:
                skipped_labels[original_label] = skipped_labels.get(original_label, 0) + 1
                continue
            rows.append(
                {
                    "id": f"shared_task-{record.get('id')}",
                    "text": record.get("claim", ""),
                    "label": FEVER_LABEL[original_label],
                    "source": "shared_task",
                    "subset": "shared_task_dev",
                    "original_label": original_label,
                    "verifiable": record.get("verifiable"),
                }
            )

    converted = pd.DataFrame(rows)
    cleaned, report = clean_binary_dataframe(converted)
    report["skipped_labels"] = skipped_labels
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    return cleaned, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert public extra datasets to trainable CSV files.")
    parser.add_argument("--input-dir", default="datasets/raw_data")
    parser.add_argument("--output-dir", default="datasets")
    parser.add_argument("--shared-task", default="datasets/raw_data/shared_task_dev.jsonl")
    parser.add_argument("--gossipcop-output", default="gossipcop_extra.csv")
    parser.add_argument("--shared-task-output", default="shared_task_extra.csv")
    parser.add_argument("--combined-output", default="all_extra.csv")
    parser.add_argument(
        "--gossipcop-text-fields",
        nargs="+",
        default=["title", "description", "text"],
        help="Text fields to concatenate, in order.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    gossip_df, gossip_report = convert_gossipcop(
        input_dir,
        output_dir / args.gossipcop_output,
        args.gossipcop_text_fields,
    )
    shared_df, shared_report = convert_shared_task(
        Path(args.shared_task),
        output_dir / args.shared_task_output,
    )

    combined = pd.concat([gossip_df, shared_df], ignore_index=True)
    combined, combined_report = clean_binary_dataframe(combined)
    combined_path = output_dir / args.combined_output
    combined.to_csv(combined_path, index=False)

    report = {
        "gossipcop": gossip_report,
        "shared_task": shared_report,
        "combined": combined_report,
        "outputs": {
            "gossipcop": str(output_dir / args.gossipcop_output),
            "shared_task": str(output_dir / args.shared_task_output),
            "combined": str(combined_path),
        },
        "label_mapping": {
            "gossipcop": {"R": 0, "F": 1, "H": "human", "M": "machine"},
            "shared_task": FEVER_LABEL,
        },
    }
    save_json(report, output_dir / "extra_datasets_report.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
