#!/usr/bin/env python3
"""Build AudioSet HDF5 files expected by Audio-JEPA dataloader."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import soundfile as sf


def load_manifest(path: Path, max_items: int | None = None) -> list[dict]:
    items: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            items.append(json.loads(line))
            if max_items is not None and i + 1 >= max_items:
                break
    return items


def to_int16_clip(wav: np.ndarray, target_len: int) -> np.ndarray:
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32, copy=False)
    if wav.shape[0] < target_len:
        padded = np.zeros(target_len, dtype=np.float32)
        padded[: wav.shape[0]] = wav
        wav = padded
    elif wav.shape[0] > target_len:
        wav = wav[:target_len]
    wav = np.clip(wav, -1.0, 1.0)
    return (wav * 32767.0).astype(np.int16)


def write_silent_csv(indices: list[int], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Index"])
        for idx in indices:
            writer.writerow([idx])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build AudioSet HDF5 from WAV files.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output-h5", type=Path, required=True)
    parser.add_argument("--silent-csv", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, default=527)
    parser.add_argument("--target-sr", type=int, default=32000)
    parser.add_argument("--clip-seconds", type=float, default=10.0)
    parser.add_argument("--max-items", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_len = int(args.target_sr * args.clip_seconds)
    manifest = load_manifest(args.manifest, args.max_items)

    valid: list[tuple[dict, Path]] = []
    missing = 0
    for item in manifest:
        wav_path = args.audio_dir / f"{item['audio_name']}.wav"
        if wav_path.exists() and wav_path.stat().st_size > 0:
            valid.append((item, wav_path))
        else:
            missing += 1

    n = len(valid)
    if n == 0:
        raise RuntimeError(f"No valid wav files found in {args.audio_dir}")

    args.output_h5.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")

    print(
        f"[hdf5] building {args.output_h5} with {n} clips (missing from manifest={missing})"
    )

    silent_indices: list[int] = []
    with h5py.File(args.output_h5, "w") as h5f:
        ds_name = h5f.create_dataset("audio_name", shape=(n,), dtype=string_dtype)
        ds_wave = h5f.create_dataset(
            "waveform",
            shape=(n, target_len),
            dtype=np.int16,
            chunks=(1, target_len),
            compression="gzip",
            compression_opts=3,
        )
        target_chunk_rows = min(n, 256)
        ds_target = h5f.create_dataset(
            "target",
            shape=(n, args.num_classes),
            dtype=np.float32,
            chunks=(target_chunk_rows, args.num_classes),
            compression="gzip",
            compression_opts=3,
        )

        for i, (item, wav_path) in enumerate(valid):
            wav, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
            if sr != args.target_sr:
                raise ValueError(
                    f"Unexpected sample rate for {wav_path}: {sr} != {args.target_sr}"
                )
            clip = to_int16_clip(wav, target_len=target_len)
            target = np.zeros(args.num_classes, dtype=np.float32)
            target[np.asarray(item["label_indices"], dtype=np.int32)] = 1.0

            if np.max(np.abs(clip)) < 8:
                silent_indices.append(i)

            ds_name[i] = item["audio_name"]
            ds_wave[i] = clip
            ds_target[i] = target

            if (i + 1) % 200 == 0 or (i + 1) == n:
                print(f"[hdf5] wrote {i + 1}/{n}")

    write_silent_csv(silent_indices, args.silent_csv)
    print(f"[hdf5] done: {args.output_h5}")
    print(f"[hdf5] silent csv: {args.silent_csv} (rows={len(silent_indices)})")


if __name__ == "__main__":
    main()
