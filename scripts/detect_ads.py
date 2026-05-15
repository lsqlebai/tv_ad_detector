#!/usr/bin/env python3
import argparse
import csv
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import imageio_ffmpeg
import numpy as np


@dataclass(frozen=True)
class Keypoint:
    start: float
    end: Optional[float]
    raw: str


METRIC_COLUMNS = [
    "brightness",
    "contrast",
    "saturation",
    "edge_density",
    "colorfulness",
    "motion",
]


def parse_time(value: str) -> float:
    parts = [float(part) for part in value.strip().split(":")]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    raise ValueError(f"Bad timestamp: {value!r}")


def format_time(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{sec:02d}"
    return f"{minutes}:{sec:02d}"


def safe_time_name(seconds: float) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "-", format_time(seconds)).strip("-")


def parse_keypoints(path: Path) -> list[Keypoint]:
    if not path.exists() or path.is_dir():
        return []

    keypoints: list[Keypoint] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        if "-" in raw:
            start_raw, end_raw = raw.split("-", 1)
            end = parse_time(end_raw) if end_raw.strip() else None
        else:
            start_raw, end = raw, None
        keypoints.append(Keypoint(parse_time(start_raw), end, raw))
    return keypoints


def frame_feature(frame: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame, (32, 18), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).flatten()
    hist = hist.astype(np.float32)
    hist /= np.linalg.norm(hist) + 1e-8

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    mean = np.array(
        [
            rgb[:, :, 0].mean(),
            rgb[:, :, 1].mean(),
            rgb[:, :, 2].mean(),
            gray.mean(),
            gray.std(),
        ],
        dtype=np.float32,
    )
    coarse = cv2.resize(rgb, (8, 5), interpolation=cv2.INTER_AREA).flatten()
    feat = np.concatenate([coarse, hist, mean])
    feat /= np.linalg.norm(feat) + 1e-8
    return feat


def frame_metrics(frame: np.ndarray, previous_gray: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    edges = cv2.Canny(gray, 80, 160)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32)
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    rg = np.abs(red - green)
    yb = np.abs(0.5 * (red + green) - blue)

    motion = 0.0
    if previous_gray is not None:
        motion = float(np.mean(cv2.absdiff(gray, previous_gray)) / 255.0)

    metrics = np.array(
        [
            float(gray.mean() / 255.0),
            float(gray.std() / 255.0),
            float(hsv[:, :, 1].mean() / 255.0),
            float(np.count_nonzero(edges) / edges.size),
            float((rg.std() + yb.std()) / 255.0),
            motion,
        ],
        dtype=np.float32,
    )
    return metrics, gray


def sample_video(video_path: Path, sample_rate: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    reader = imageio_ffmpeg.read_frames(
        str(video_path),
        pix_fmt="rgb24",
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


def cosine_window_scores(features: np.ndarray, template: np.ndarray) -> np.ndarray:
    window = len(template)
    if window > len(features):
        return np.asarray([], dtype=np.float32)
    scores = np.empty(len(features) - window + 1, dtype=np.float32)
    for index in range(len(scores)):
        scores[index] = float(np.mean(np.sum(features[index : index + window] * template, axis=1)))
    return scores


def merge_candidates(candidates: list[dict], merge_gap: float) -> list[dict]:
    candidates = sorted(candidates, key=lambda item: (item["start"], -item["score"]))
    merged: list[dict] = []
    for item in candidates:
        if not merged or item["start"] > merged[-1]["end"] + merge_gap:
            merged.append(item.copy())
            continue
        merged[-1]["end"] = max(merged[-1]["end"], item["end"])
        merged[-1]["score"] = max(merged[-1]["score"], item["score"])
        merged[-1]["sources"] = sorted(set(merged[-1]["sources"]) | set(item["sources"]))
    return merged


def overlaps(start: float, end: float, other_start: float, other_end: float) -> bool:
    return max(start, other_start) < min(end, other_end)


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def robust_norm(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    low = float(np.percentile(values, 10))
    high = float(np.percentile(values, 90))
    if high - low < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def duration_score(length: float) -> float:
    if length < 4:
        return 0.0
    if length <= 8:
        return (length - 4) / 4
    if length <= 35:
        return 1.0
    if length <= 70:
        return 1.0 - (length - 35) / 35
    return 0.0


def segment_bounds_from_cuts(
    times: np.ndarray,
    features: np.ndarray,
    duration: float,
    min_gap: float,
) -> tuple[list[float], np.ndarray]:
    if len(times) < 3:
        return [0.0, duration], np.asarray([], dtype=np.float32)

    deltas = 1.0 - np.sum(features[1:] * features[:-1], axis=1)
    threshold = max(float(np.percentile(deltas, 92)), float(deltas.mean() + deltas.std() * 1.15))
    cut_indices = np.flatnonzero(deltas >= threshold) + 1

    boundaries = [0.0]
    last = 0.0
    for index in cut_indices:
        cut_time = float(times[index])
        if cut_time - last < min_gap or duration - cut_time < min_gap:
            continue
        boundaries.append(cut_time)
        last = cut_time
    if duration - boundaries[-1] >= 1.0:
        boundaries.append(duration)
    else:
        boundaries[-1] = duration
    return boundaries, deltas


def boundary_strength(time_value: float, times: np.ndarray, deltas: np.ndarray) -> float:
    if len(deltas) == 0:
        return 0.0
    index = int(np.searchsorted(times, time_value))
    lower = max(0, index - 3)
    upper = min(len(deltas), index + 3)
    if lower >= upper:
        return 0.0
    strongest = float(np.max(deltas[lower:upper]))
    high = float(np.percentile(deltas, 95)) + 1e-8
    return clamp(strongest / high)


def discover_ad_candidates(
    times: np.ndarray,
    features: np.ndarray,
    metrics: np.ndarray,
    duration: float,
    threshold: float,
    min_gap: float,
    max_candidates: int,
) -> list[dict]:
    boundaries, deltas = segment_bounds_from_cuts(times, features, duration, min_gap=max(2.0, min_gap / 2))
    if len(boundaries) < 2:
        return []

    normalized = np.column_stack([robust_norm(metrics[:, index]) for index in range(metrics.shape[1])])
    metric_index = {name: index for index, name in enumerate(METRIC_COLUMNS)}

    candidates: list[dict] = []
    max_auto_duration = min(45.0, duration * 0.35)
    for start_index in range(len(boundaries) - 1):
        for end_index in range(start_index + 1, len(boundaries)):
            start = boundaries[start_index]
            end = boundaries[end_index]
            length = end - start
            if length > max_auto_duration:
                break
            if length < 4:
                continue

            mask = (times >= start) & (times < end)
            if np.count_nonzero(mask) < 2:
                continue

            segment_metrics = normalized[mask]
            raw_segment_metrics = metrics[mask]
            saturation = float(segment_metrics[:, metric_index["saturation"]].mean())
            edge_density = float(segment_metrics[:, metric_index["edge_density"]].mean())
            contrast = float(segment_metrics[:, metric_index["contrast"]].mean())
            colorfulness = float(segment_metrics[:, metric_index["colorfulness"]].mean())
            motion = float(raw_segment_metrics[:, metric_index["motion"]].mean())

            visual_score = clamp(
                0.30 * saturation
                + 0.30 * edge_density
                + 0.20 * contrast
                + 0.20 * colorfulness
            )
            time_position = 0.0
            if start <= 75:
                time_position = max(time_position, 1.0 - start / 75)
            if duration - end <= 90:
                time_position = max(time_position, 1.0 - max(0.0, duration - end) / 90)

            cut_score = max(boundary_strength(start, times, deltas), boundary_strength(end, times, deltas))
            scene_count = end_index - start_index
            scene_bonus = 0.08 if 2 <= scene_count <= 5 and 10 <= length <= 35 else 0.0
            still_card_bonus = 0.12 if motion < 0.025 and visual_score > 0.45 else 0.0
            score = clamp(
                0.34 * visual_score
                + 0.26 * duration_score(length)
                + 0.22 * time_position
                + 0.18 * cut_score
                + scene_bonus
                + still_card_bonus
            )

            if score < threshold:
                continue

            reasons = [
                f"visual={visual_score:.2f}",
                f"duration={duration_score(length):.2f}",
                f"position={time_position:.2f}",
                f"cut={cut_score:.2f}",
                f"scenes={scene_count}",
            ]
            if still_card_bonus:
                reasons.append("still_card")

            candidates.append(
                {
                    "start": start,
                    "end": end,
                    "score": score,
                    "sources": reasons,
                    "kind": "auto_discovery",
                }
            )

    picked: list[dict] = []
    for item in sorted(candidates, key=lambda value: value["score"], reverse=True):
        if any(overlaps(float(item["start"]), float(item["end"]), float(old["start"]) - min_gap, float(old["end"]) + min_gap) for old in picked):
            continue
        picked.append(item)
        if len(picked) >= max_candidates:
            break
    return sorted(picked, key=lambda item: item["start"])


def detect_from_templates(
    times: np.ndarray,
    features: np.ndarray,
    keypoints: list[Keypoint],
    threshold: float,
    min_gap: float,
    duration: float,
) -> list[dict]:
    candidates: list[dict] = []
    sample_step = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    confirmed_ranges: list[tuple[float, float]] = []

    for keypoint in keypoints:
        end = keypoint.end if keypoint.end is not None else duration
        if end <= keypoint.start:
            continue
        confirmed_ranges.append((keypoint.start, end))
        candidates.append(
            {
                "start": keypoint.start,
                "end": end,
                "score": 1.0,
                "sources": [keypoint.raw],
                "kind": "manual_confirmed" if keypoint.end is not None else "manual_open_to_end",
            }
        )

    for keypoint in keypoints:
        if keypoint.end is None or keypoint.end <= keypoint.start:
            continue
        mask = (times >= keypoint.start) & (times < keypoint.end)
        template = features[mask]
        if len(template) < 3:
            continue

        scores = cosine_window_scores(features, template)
        if len(scores) == 0:
            continue
        window_duration = max(keypoint.end - keypoint.start, len(template) * sample_step)

        order = np.argsort(scores)[::-1]
        picked: list[tuple[float, float, float]] = []
        for score_index in order:
            score = float(scores[score_index])
            if score < threshold:
                break
            start = float(times[score_index])
            end = start + window_duration
            if any(overlaps(start, end, known_start, known_end) for known_start, known_end in confirmed_ranges):
                continue
            if any(abs(start - old_start) < min_gap for old_start, _, _ in picked):
                continue
            picked.append((start, end, score))
            candidates.append(
                {
                    "start": start,
                    "end": end,
                    "score": score,
                    "sources": [keypoint.raw],
                    "kind": "template_match",
                }
            )

    return merge_candidates(candidates, merge_gap=min_gap)


def build_templates_from_keypoints(
    times: np.ndarray,
    features: np.ndarray,
    keypoints: list[Keypoint],
    duration: float,
) -> list[dict]:
    templates: list[dict] = []
    for keypoint in keypoints:
        end = keypoint.end if keypoint.end is not None else duration
        if end <= keypoint.start:
            continue
        mask = (times >= keypoint.start) & (times < end)
        template = features[mask]
        if len(template) < 3:
            continue
        templates.append(
            {
                "features": template,
                "duration": end - keypoint.start,
                "source": keypoint.raw,
            }
        )
    return templates


def save_templates(template_dir: Path, video_path: Path, templates: list[dict]) -> None:
    if not templates:
        return
    template_dir.mkdir(parents=True, exist_ok=True)
    for index, item in enumerate(templates, start=1):
        source = safe_time_name(parse_time(item["source"].split("-", 1)[0]))
        path = template_dir / f"{video_path.stem}_{index:02d}_{source}.npz"
        np.savez_compressed(
            path,
            features=item["features"],
            duration=np.asarray([item["duration"]], dtype=np.float32),
            source=np.asarray([item["source"]]),
            video=np.asarray([video_path.name]),
        )


def load_templates(template_dir: Path) -> list[dict]:
    if not template_dir.exists():
        return []
    templates: list[dict] = []
    for path in sorted(template_dir.glob("*.npz")):
        data = np.load(path)
        templates.append(
            {
                "features": data["features"],
                "duration": float(data["duration"][0]),
                "source": str(data["source"][0]),
                "path": str(path),
            }
        )
    return templates


def detect_from_template_library(
    times: np.ndarray,
    features: np.ndarray,
    templates: list[dict],
    threshold: float,
    min_gap: float,
) -> list[dict]:
    candidates: list[dict] = []
    for template_item in templates:
        template = template_item["features"]
        scores = cosine_window_scores(features, template)
        if len(scores) == 0:
            continue

        picked: list[tuple[float, float, float]] = []
        for score_index in np.argsort(scores)[::-1]:
            score = float(scores[score_index])
            if score < threshold:
                break
            start = float(times[score_index])
            end = start + float(template_item["duration"])
            if any(abs(start - old_start) < min_gap for old_start, _, _ in picked):
                continue
            picked.append((start, end, score))
            candidates.append(
                {
                    "start": start,
                    "end": end,
                    "score": score,
                    "sources": [f"template={template_item['source']}"],
                    "kind": "template_library",
                }
            )
    return merge_candidates(candidates, merge_gap=min_gap)


def combine_detections(detections: list[dict], merge_gap: float) -> list[dict]:
    priority = {
        "manual_confirmed": 0,
        "manual_open_to_end": 0,
        "template_match": 1,
        "template_library": 1,
        "auto_discovery": 2,
    }
    combined: list[dict] = []
    for item in sorted(detections, key=lambda value: priority.get(value["kind"], 9)):
        start = float(item["start"])
        end = float(item["end"])
        merged = False
        for existing in combined:
            if not overlaps(start, end, float(existing["start"]) - merge_gap, float(existing["end"]) + merge_gap):
                continue
            existing_priority = priority.get(existing["kind"], 9)
            item_priority = priority.get(item["kind"], 9)
            if item_priority < existing_priority:
                existing.update(item)
            elif item_priority > existing_priority:
                existing["score"] = max(float(existing["score"]), float(item["score"]))
                existing["sources"] = sorted(set(existing["sources"]) | set(item["sources"]))
            else:
                existing["start"] = min(float(existing["start"]), start)
                existing["end"] = max(float(existing["end"]), end)
                existing["score"] = max(float(existing["score"]), float(item["score"]))
                existing["sources"] = sorted(set(existing["sources"]) | set(item["sources"]))
            merged = True
            break
        if not merged:
            combined.append(item.copy())
    return sorted(combined, key=lambda value: value["start"])


def snapshot_targets(detections: list[dict], duration: float) -> list[dict]:
    targets: list[dict] = []
    for index, item in enumerate(detections, start=1):
        start = float(item["start"])
        end = float(item["end"])
        if end <= start:
            targets.append({"detection": index, "label": "start", "time": start})
            continue

        span = end - start
        inside_offset = min(0.5, max(0.0, span / 4))
        end_inside = min(duration, max(start, end - inside_offset))
        mid = start + span / 2
        points = [
            ("start", min(duration, start + inside_offset)),
            ("middle", min(duration, mid)),
            ("end", end_inside),
        ]
        for label, point_time in points:
            targets.append({"detection": index, "label": label, "time": point_time})
    return sorted(targets, key=lambda item: item["time"])


def overlay_label(frame: np.ndarray, text: str) -> np.ndarray:
    image = frame.copy()
    cv2.rectangle(image, (0, 0), (image.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(image, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def extract_snapshots(
    video_path: Path,
    detections: list[dict],
    duration: float,
    output_dir: Path,
    write_contact_sheet: bool,
) -> dict[int, dict[str, str]]:
    targets = snapshot_targets(detections, duration)
    if not targets:
        return {}

    snapshot_dir = output_dir / f"{video_path.stem}.ad_frames"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for old_file in snapshot_dir.glob("*.jpg"):
        old_file.unlink()

    snapshots: dict[int, dict[str, str]] = {}
    contact_frames: list[np.ndarray] = []
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    for target in targets:
        detection_index = target["detection"]
        label = target["label"]
        target_time = float(target["time"])
        filename = f"{detection_index:02d}_{label}_{safe_time_name(target_time)}.jpg"
        path = snapshot_dir / filename
        raw_path = snapshot_dir / f".{filename}.raw.jpg"

        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-ss",
            f"{target_time:.3f}",
            "-frames:v",
            "1",
            "-q:v",
            "2",
            "-y",
            str(raw_path),
        ]
        subprocess.run(command, check=True)
        frame = cv2.imread(str(raw_path))
        raw_path.unlink(missing_ok=True)
        if frame is None:
            continue

        display = overlay_label(frame, f"ad {detection_index} {label} {format_time(target_time)}")
        cv2.imwrite(str(path), display)
        report_path = path.relative_to(output_dir.parent)
        snapshots.setdefault(detection_index, {})[label] = report_path.as_posix()
        contact_frames.append(cv2.resize(display, (320, 180), interpolation=cv2.INTER_AREA))

    if write_contact_sheet and contact_frames:
        columns = 3
        rows = []
        blank = np.zeros_like(contact_frames[0])
        for index in range(0, len(contact_frames), columns):
            row = contact_frames[index : index + columns]
            while len(row) < columns:
                row.append(blank)
            rows.append(np.hstack(row))
        cv2.imwrite(str(output_dir / f"{video_path.stem}.ads.keyframes.jpg"), np.vstack(rows))

    return snapshots


def write_outputs(
    output_dir: Path,
    video_path: Path,
    detections: list[dict],
    duration: float,
    write_debug_files: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    snapshots = extract_snapshots(video_path, detections, duration, output_dir, write_debug_files)

    with (output_dir / f"{stem}.ads.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "start",
                "end",
                "start_seconds",
                "end_seconds",
                "score",
                "kind",
                "sources",
                "review_required",
                "start_frame",
                "middle_frame",
                "end_frame",
            ],
        )
        writer.writeheader()
        for index, item in enumerate(detections, start=1):
            item_snapshots = snapshots.get(index, {})
            writer.writerow(
                {
                    "start": format_time(item["start"]),
                    "end": format_time(item["end"]) if item["end"] > item["start"] else "",
                    "start_seconds": round(item["start"], 3),
                    "end_seconds": round(item["end"], 3) if item["end"] > item["start"] else "",
                    "score": round(item["score"], 4),
                    "kind": item["kind"],
                    "sources": ";".join(item["sources"]),
                    "review_required": "yes" if item["kind"] == "auto_discovery" else "no",
                    "start_frame": item_snapshots.get("start", ""),
                    "middle_frame": item_snapshots.get("middle", ""),
                    "end_frame": item_snapshots.get("end", ""),
                }
            )

    for index, item in enumerate(detections, start=1):
        item["snapshots"] = snapshots.get(index, {})

    if write_debug_files:
        trusted_lines = []
        candidate_lines = []
        for item in detections:
            if item["end"] <= item["start"]:
                line = f"{format_time(item['start'])}-"
            else:
                line = f"{format_time(item['start'])}-{format_time(item['end'])}"
            candidate_lines.append(line)
            if item["kind"] != "auto_discovery":
                trusted_lines.append(line)
        (output_dir / f"{stem}.ads.txt").write_text(
            "\n".join(trusted_lines) + ("\n" if trusted_lines else ""),
            encoding="utf-8",
        )
        (output_dir / f"{stem}.candidates.txt").write_text(
            "\n".join(candidate_lines) + ("\n" if candidate_lines else ""),
            encoding="utf-8",
        )

        payload = {
            "video": str(video_path),
            "duration_seconds": duration,
            "detections": detections,
        }
        (output_dir / f"{stem}.ads.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect repeated ad segments from confirmed keypoints.")
    parser.add_argument("--input-dir", type=Path, default=Path("input"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/detect"))
    parser.add_argument("--keypoints", type=Path, default=Path("keypoint.txt"))
    parser.add_argument("--ignore-keypoints", action="store_true", help="Ignore keypoint file and rely on template/auto discovery.")
    parser.add_argument("--template-dir", type=Path, default=Path("ad_templates"), help="Directory used to save/load confirmed ad templates.")
    parser.add_argument("--sample-rate", type=float, default=2.0, help="Frames per second to sample.")
    parser.add_argument("--threshold", type=float, default=0.94, help="Template similarity threshold.")
    parser.add_argument("--auto-threshold", type=float, default=0.80, help="Minimum score for automatic ad candidates.")
    parser.add_argument("--max-auto-candidates", type=int, default=8, help="Maximum automatic candidates per video.")
    parser.add_argument("--min-gap", type=float, default=8.0, help="Minimum seconds between detections.")
    parser.add_argument("--no-auto-discover", action="store_true", help="Disable automatic ad discovery.")
    parser.add_argument("--write-debug-files", action="store_true", help="Also write *.ads.txt, *.candidates.txt, *.ads.json, and keyframe contact sheets.")
    args = parser.parse_args()
    template_dir = args.template_dir

    keypoints = [] if args.ignore_keypoints else parse_keypoints(args.keypoints)
    if args.ignore_keypoints:
        print("Ignoring keypoints; using template library plus automatic discovery.")
    elif not keypoints:
        print(f"No keypoints found in {args.keypoints}; using template library plus automatic discovery.")

    videos = sorted(args.input_dir.glob("*.mp4"))
    if not videos:
        raise SystemExit(f"No .mp4 files found in {args.input_dir}")

    for video_path in videos:
        print(f"Sampling {video_path.name}...")
        times, features, metrics, duration = sample_video(video_path, args.sample_rate)
        keypoint_templates = build_templates_from_keypoints(times, features, keypoints, duration)
        save_templates(template_dir, video_path, keypoint_templates)

        detections = detect_from_templates(times, features, keypoints, args.threshold, args.min_gap, duration)
        library_templates = load_templates(template_dir)
        if library_templates:
            detections.extend(
                detect_from_template_library(
                    times,
                    features,
                    library_templates,
                    args.threshold,
                    args.min_gap,
                )
            )
        if not args.no_auto_discover:
            detections.extend(
                discover_ad_candidates(
                    times,
                    features,
                    metrics,
                    duration,
                    args.auto_threshold,
                    args.min_gap,
                    args.max_auto_candidates,
                )
            )
        detections = combine_detections(detections, merge_gap=args.min_gap)
        write_outputs(args.output_dir, video_path, detections, duration, args.write_debug_files)
        print(f"{video_path.name}:")
        for item in detections:
            if item["end"] <= item["start"]:
                print(f"  {format_time(item['start'])}-  {item['kind']} score={item['score']:.3f}")
            else:
                print(f"  {format_time(item['start'])}-{format_time(item['end'])}  {item['kind']} score={item['score']:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
