#!/usr/bin/env python3
"""Download AudioSet segments with yt-dlp into per-split WAV files."""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def load_manifest(
    path: Path,
    max_items: int | None = None,
    skip_items: int = 0,
) -> list[dict]:
    items: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            if i < skip_items:
                continue
            items.append(json.loads(line))
            if max_items is not None and len(items) >= max_items:
                break
    return items


def run_cmd(cmd: list[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-8:])
        return proc.returncode, stderr_tail
    except subprocess.TimeoutExpired as ex:
        return 124, f"timeout: {ex}"


def download_one(
    item: dict,
    out_dir: Path,
    yt_dlp_bin: str,
    retries: int,
    extractor_args: str,
    per_clip_timeout: int,
) -> tuple[str, str]:
    audio_name = item["audio_name"]
    ytid = item["ytid"]
    start = float(item["start"])
    end = float(item["end"])

    out_path = out_dir / f"{audio_name}.wav"
    if out_path.exists() and out_path.stat().st_size > 0:
        return audio_name, "exists"

    # yt-dlp uses ffmpeg to clip the requested segment.
    cmd = [
        yt_dlp_bin,
        "--quiet",
        "--no-warnings",
        "--ignore-errors",
        "--no-playlist",
        "--extractor-args",
        extractor_args,
        "--extract-audio",
        "--audio-format",
        "wav",
        "--audio-quality",
        "0",
        "--postprocessor-args",
        "ffmpeg:-ac 1 -ar 32000",
        "--download-sections",
        f"*{start}-{end}",
        "-o",
        str(out_dir / f"{audio_name}.%(ext)s"),
        f"https://www.youtube.com/watch?v={ytid}",
    ]

    last_err = ""
    for _ in range(retries + 1):
        code, err = run_cmd(cmd + ["--socket-timeout", str(per_clip_timeout)])
        if code == 0 and out_path.exists() and out_path.stat().st_size > 0:
            return audio_name, "ok"
        last_err = err
    return audio_name, f"failed: {last_err or 'unknown error'}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download AudioSet clips.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--yt-dlp-bin", type=str, default="yt-dlp")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--skip-items", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument(
        "--extractor-args",
        type=str,
        default="youtube:player_client=android",
        help="yt-dlp extractor args for YouTube (android client avoids common bot checks).",
    )
    parser.add_argument(
        "--per-clip-timeout",
        type=int,
        default=120,
        help="yt-dlp network timeout per clip request in seconds.",
    )
    parser.add_argument("--status-log", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.status_log.parent.mkdir(parents=True, exist_ok=True)

    items = load_manifest(args.manifest, args.max_items, args.skip_items)
    total = len(items)
    print(f"[download] queued={total} manifest={args.manifest}")

    ok = 0
    exists = 0
    failed = 0

    with args.status_log.open("w", encoding="utf-8", buffering=1) as log_f:
        log_f.write("audio_name,status\n")
        log_f.flush()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(
                    download_one,
                    item=item,
                    out_dir=args.out_dir,
                    yt_dlp_bin=args.yt_dlp_bin,
                    retries=args.retries,
                    extractor_args=args.extractor_args,
                    per_clip_timeout=args.per_clip_timeout,
                ): item["audio_name"]
                for item in items
            }

            done = 0
            for fut in as_completed(futures):
                audio_name, status = fut.result()
                done += 1
                if status == "ok":
                    ok += 1
                elif status == "exists":
                    exists += 1
                else:
                    failed += 1
                log_f.write(f"{audio_name},{status}\n")
                log_f.flush()
                if done % 50 == 0 or done == total:
                    print(
                        f"[download] done={done}/{total} ok={ok} exists={exists} failed={failed}"
                    )

    print(f"[download] finished: ok={ok} exists={exists} failed={failed}")
    print(f"[download] status log: {args.status_log}")


if __name__ == "__main__":
    main()
