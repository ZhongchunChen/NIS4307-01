"""Regression tests for optional dataset conversion."""

from __future__ import annotations

import json
from pathlib import Path

from src.cli.prepare_extra_datasets import convert_shared_task


def test_shared_task_with_no_binary_rows_writes_empty_dataset(tmp_path: Path) -> None:
    input_path = tmp_path / "shared_task.jsonl"
    input_path.write_text(
        json.dumps(
            {
                "id": 1,
                "claim": "Insufficient evidence",
                "label": "NOT ENOUGH INFO",
                "verifiable": "NOT VERIFIABLE",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "shared_task.csv"

    dataframe, report = convert_shared_task(input_path, output_path)

    assert dataframe.empty
    assert list(dataframe.columns) == [
        "id",
        "text",
        "label",
        "source",
        "subset",
        "original_label",
        "verifiable",
    ]
    assert report["final_rows"] == 0
    assert report["skipped_labels"] == {"NOT ENOUGH INFO": 1}
    assert output_path.read_text(encoding="utf-8").startswith("id,text,label,")
