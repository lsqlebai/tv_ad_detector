#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

import detect_ads


def sample_template(video_path: Path, start: float, end: float, sample_rate: float) -> np.ndarray:
    if end <= start:
        raise ValueError("end must be after start")
    reader = imageio_ffmpeg.read_frames(
        str(video_path),
        pix_fmt="rgb24",
        input_params=["-ss", f"{start:.3f}"],
        output_params=["-t", f"{end - start:.3f}", "-vf", f"fps={sample_rate},scale=320:-2"],
    )
    try:
        meta = next(reader)
    except StopIteration as exc:
        raise RuntimeError(f"Cannot open video: {video_path}") from exc

    width, height = meta["size"]
    features = []
    for frame_bytes in reader:
        rgb = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((height, width, 3))
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        features.append(detect_ads.frame_feature(frame))
    if len(features) < 3:
        raise RuntimeError("Template segment is too short; need at least 3 sampled frames")
    return np.vstack(features)


def unique_template_path(template_dir: Path, video_path: Path, start: float, end: float) -> Path:
    base = f"{video_path.stem}_review_{detect_ads.safe_time_name(start)}_{detect_ads.safe_time_name(end)}"
    path = template_dir / f"{base}.npz"
    suffix = 2
    while path.exists():
        path = template_dir / f"{base}_{suffix}.npz"
        suffix += 1
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Create an ad template from a reviewed segment.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--template-dir", type=Path, default=Path("ad_templates"))
    parser.add_argument("--sample-rate", type=float, default=1.0)
    args = parser.parse_args()

    start = detect_ads.parse_time(args.start)
    end = detect_ads.parse_time(args.end)
    if end <= start:
        raise SystemExit("end must be after start")
    if not args.video.exists() or not args.video.is_file():
        raise SystemExit(f"Video not found: {args.video}")

    features = sample_template(args.video, start, end, args.sample_rate)
    args.template_dir.mkdir(parents=True, exist_ok=True)
    path = unique_template_path(args.template_dir, args.video, start, end)
    source = f"{detect_ads.format_time_precise(start)}-{detect_ads.format_time_precise(end)}"
    np.savez_compressed(
        path,
        features=features,
        duration=np.asarray([end - start], dtype=np.float32),
        sample_rate=np.asarray([args.sample_rate], dtype=np.float32),
        source=np.asarray([source]),
        video=np.asarray([args.video.name]),
    )
    print(path.as_posix(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
