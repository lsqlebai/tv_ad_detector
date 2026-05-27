from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import imageio_ffmpeg
import numpy as np

from .features import frame_feature, frame_metrics


def cv_imread(path: Path, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)

def cv_imwrite(path: Path, image: np.ndarray) -> bool:
    success, encoded = cv2.imencode(path.suffix, image)
    if not success:
        return False
    encoded.tofile(str(path))
    return True

def read_scaled_frame(video_path: Path, time_value: float) -> Optional[np.ndarray]:
    reader = imageio_ffmpeg.read_frames(
        str(video_path),
        pix_fmt="rgb24",
        input_params=["-ss", f"{max(0.0, time_value):.3f}"],
        output_params=["-frames:v", "1", "-vf", "scale=320:-2"],
    )
    try:
        meta = next(reader)
        width, height = meta["size"]
        frame_bytes = next(reader)
    except (OSError, StopIteration):
        return None
    rgb = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((height, width, 3))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

def sample_video(video_path: Path, sample_rate: float, skip_frame: str = "noref") -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    input_params = [] if skip_frame == "none" else ["-skip_frame", skip_frame]
    reader = imageio_ffmpeg.read_frames(
        str(video_path),
        pix_fmt="rgb24",
        input_params=input_params,
        output_params=["-vf", f"fps={sample_rate},scale=320:-2"],
    )
    try:
        meta = next(reader)
    except StopIteration as exc:
        raise RuntimeError(f"Cannot open video: {video_path}") from exc

    width, height = meta["size"]
    duration = float(meta.get("duration") or 0)
    times: list[float] = []
    features: list[np.ndarray] = []
    metrics: list[np.ndarray] = []
    previous_gray: Optional[np.ndarray] = None

    for frame_index, frame_bytes in enumerate(reader):
        rgb = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((height, width, 3))
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        times.append(frame_index / sample_rate)
        features.append(frame_feature(frame))
        item_metrics, previous_gray = frame_metrics(frame, previous_gray)
        metrics.append(item_metrics)

    if not features:
        raise RuntimeError(f"No frames sampled from {video_path}")
    return np.asarray(times, dtype=np.float32), np.vstack(features), np.vstack(metrics), duration

def boundary_frames(
    video_path: Path,
    start: float,
    end: float,
    sample_rate: float,
    duration: float,
) -> list[tuple[float, float, float, float]]:
    start = max(0.0, min(duration, start))
    end = max(start, min(duration, end))
    if end - start < 1.0 / sample_rate:
        return []
    return list(
        _boundary_frames_cached(
            str(video_path),
            round(start, 3),
            round(end, 3),
            round(sample_rate, 3),
        )
    )


@lru_cache(maxsize=512)
def _boundary_frames_cached(
    video_path_text: str,
    start: float,
    end: float,
    sample_rate: float,
) -> tuple[tuple[float, float, float, float], ...]:
    reader = imageio_ffmpeg.read_frames(
        video_path_text,
        pix_fmt="rgb24",
        input_params=["-ss", f"{start:.3f}"],
        output_params=["-t", f"{end - start:.3f}", "-vf", f"fps={sample_rate},scale=320:-2"],
    )
    try:
        meta = next(reader)
    except (OSError, StopIteration):
        return []

    width, height = meta["size"]
    rows: list[tuple[float, float, float, float]] = []
    previous_gray: Optional[np.ndarray] = None
    for frame_index, frame_bytes in enumerate(reader):
        time_value = start + frame_index / sample_rate
        rgb = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((height, width, 3))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        brightness = float(gray.mean() / 255.0)
        dark_ratio = float(np.mean(gray < 24))
        diff = 0.0
        if previous_gray is not None:
            diff = float(np.mean(cv2.absdiff(gray, previous_gray)) / 255.0)
        rows.append((time_value, brightness, dark_ratio, diff))
        previous_gray = gray
    return tuple(rows)

def extract_raw_snapshots(
    ffmpeg: str,
    video_path: Path,
    items: list[tuple[float, Path]],
) -> None:
    if not items:
        return

    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin"]
    for target_time, _ in items:
        command.extend(["-ss", f"{target_time:.3f}", "-i", str(video_path)])
    for index, (_, raw_path) in enumerate(items):
        command.extend(["-map", f"{index}:v:0", "-frames:v", "1", "-q:v", "2", "-y", str(raw_path)])

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError:
        for target_time, raw_path in items:
            subprocess.run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostdin",
                    "-ss",
                    f"{target_time:.3f}",
                    "-i",
                    str(video_path),
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    "-y",
                    str(raw_path),
                ],
                check=True,
            )
