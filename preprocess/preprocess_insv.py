#!/usr/bin/env python3
"""Split an INSV file into front/back MP4 streams and export lossless frames."""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return ivalue


def require_binaries() -> None:
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(f"Missing required tool(s): {joined}. Please install ffmpeg.")


def run_command(cmd: list[str]) -> None:
    pretty = " ".join(shlex.quote(str(part)) for part in cmd)
    print(f"  Running: {pretty}")
    subprocess.run(cmd, check=True)


def video_stream_exists(insv_path: Path, stream_index: int) -> bool:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        f"v:{stream_index}",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(insv_path),
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    return bool(result.stdout.strip())


def extract_stream(
    insv_path: Path, stream_index: int, output_path: Path, overwrite: bool
) -> bool:
    if output_path.exists() and not overwrite:
        print(f"  {output_path.name} already exists; skipping extraction.")
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(insv_path),
        "-map",
        f"0:v:{stream_index}",
        "-c:v",
        "copy",
        str(output_path),
    ]
    try:
        run_command(cmd)
    except subprocess.CalledProcessError as exc:
        if output_path.exists() and not overwrite:
            return True
        raise SystemExit(f"Failed to extract stream v:{stream_index}: {exc}") from exc

    return True


def extract_frames(
    video_path: Path,
    frame_dir: Path,
    frame_interval: int,
    image_format: str,
    overwrite: bool,
    max_frames: int | None,
) -> None:
    if not video_path.exists():
        print(f"  Skipping frames; source missing: {video_path}")
        return

    if frame_dir.exists() and overwrite:
        # Remove stale frames so new interval output isn't polluted by older exports.
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    pattern = frame_dir / f"%06d.{image_format}"

    filter_expr = f"select=not(mod(n\\,{frame_interval}))"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(video_path),
        "-vf",
        filter_expr,
        "-vsync",
        "vfr",
        "-frame_pts",
        "1",
    ]
    if max_frames is not None:
        cmd.extend(["-frames:v", str(max_frames)])
    cmd.append(str(pattern))
    try:
        run_command(cmd)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Failed to extract frames from {video_path}: {exc}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split an INSV file into front/back MP4 files and export frames."
    )
    parser.add_argument("insv_path", type=Path, help="Path to the source .insv file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated videos and frames (default: alongside the INSV).",
    )
    parser.add_argument(
        "--frame-interval",
        type=positive_int,
        default=1,
        help="Only keep every Nth frame when exporting images (default: 1).",
    )
    parser.add_argument(
        "--max-frames",
        type=positive_int,
        default=None,
        help="Limit the number of frames exported per stream (after interval filtering).",
    )
    parser.add_argument(
        "--image-format",
        default="png",
        help="Image format/extension for frames (lossless formats such as png/tiff).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing MP4 and frame files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_binaries()

    insv_path = args.insv_path.expanduser().resolve()
    if not insv_path.is_file():
        raise SystemExit(f"Input file not found: {insv_path}")

    output_root = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else insv_path.parent
    )
    output_root.mkdir(parents=True, exist_ok=True)

    base_name = insv_path.stem
    stream_configs = [
        ("front", 0, True),
        ("back", 1, False),
    ]

    print(f"Processing {insv_path}")
    print(f"  Output directory: {output_root}")

    for label, stream_index, required in stream_configs:
        mp4_path = output_root / f"{base_name}_{label}.mp4"
        frames_dir = output_root / f"{base_name}_{label}_frames"

        if not video_stream_exists(insv_path, stream_index):
            if required:
                raise SystemExit(f"No {label} video stream (v:{stream_index}) found; aborting.")
            print(f"  No {label} video stream detected; skipping {label} export.")
            continue

        extract_stream(insv_path, stream_index, mp4_path, args.overwrite)
        extract_frames(
            mp4_path,
            frames_dir,
            args.frame_interval,
            args.image_format,
            args.overwrite,
            args.max_frames,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
