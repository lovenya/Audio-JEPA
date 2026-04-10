#!/usr/bin/env python3
"""Prepare AudioSet JSONL manifests from official CSV metadata.

Each output line contains:
  - audio_name: stable clip id used for filename and HDF5 key
  - ytid: YouTube id
  - start: segment start in seconds
  - end: segment end in seconds
  - label_indices: list of class indices (0..526)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_mid_to_index(class_csv: Path) -> dict[str, int]:
    mapping: dict[str, int] = {}
    with class_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row["mid"].strip()] = int(row["index"])
    if not mapping:
        raise ValueError(f"No class mappings found in {class_csv}")
    return mapping


def rows_without_comments(csv_path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if line.lstrip().startswith("#"):
                continue
            rows.append(next(csv.reader([line], skipinitialspace=True)))
    return rows


def parse_labels(raw: str, mid_to_index: dict[str, int]) -> list[int]:
    mids = [x.strip() for x in raw.split(",") if x.strip()]
    labels = sorted({mid_to_index[mid] for mid in mids if mid in mid_to_index})
    return labels


def make_audio_name(ytid: str, start: float, end: float) -> str:
    # millisecond resolution so names remain stable and unique.
    return f"{ytid}_{int(round(start * 1000)):010d}_{int(round(end * 1000)):010d}"


def convert_split(
    input_csv: Path,
    class_csv: Path,
    output_jsonl: Path,
    max_items: int | None = None,
) -> tuple[int, int]:
    mid_to_index = load_mid_to_index(class_csv)
    rows = rows_without_comments(input_csv)

    n_total = 0
    n_written = 0
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with output_jsonl.open("w", encoding="utf-8") as out_f:
        for row in rows:
            n_total += 1
            if len(row) < 4:
                continue
            ytid = row[0].strip()
            start = float(row[1].strip())
            end = float(row[2].strip())
            labels = parse_labels(row[3].strip().strip('"'), mid_to_index)

            item = {
                "audio_name": make_audio_name(ytid, start, end),
                "ytid": ytid,
                "start": start,
                "end": end,
                "label_indices": labels,
            }
            out_f.write(json.dumps(item, ensure_ascii=True) + "\n")
            n_written += 1

            if max_items is not None and n_written >= max_items:
                break

    return n_total, n_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare AudioSet manifests.")
    parser.add_argument("--class-csv", type=Path, required=True)
    parser.add_argument("--balanced-csv", type=Path, required=True)
    parser.add_argument("--eval-csv", type=Path, required=True)
    parser.add_argument("--unbalanced-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--max-items-per-split",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    splits = [
        ("balanced_train", args.balanced_csv),
        ("eval", args.eval_csv),
        ("unbalanced_train", args.unbalanced_csv),
    ]

    for split_name, split_csv in splits:
        out_path = args.out_dir / f"{split_name}.jsonl"
        total, written = convert_split(
            input_csv=split_csv,
            class_csv=args.class_csv,
            output_jsonl=out_path,
            max_items=args.max_items_per_split,
        )
        print(
            f"[manifest] {split_name}: source_rows={total} written={written} -> {out_path}"
        )


if __name__ == "__main__":
    main()

