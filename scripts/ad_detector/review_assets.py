from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from .features import frame_diff_score
from .models import AUTO_TRUST_THRESHOLD, MIN_BOUNDARY_FRAME_DIFF, SNAPSHOT_LABELS
from .time_utils import format_time, format_time_precise, safe_time_name
from .video_io import cv_imread, cv_imwrite, extract_raw_snapshots

BOUNDARY_ERROR_PAIRS = [
    ("start", "start_before", "start"),
    ("end", "end", "end_after"),
    ("start", "start", "end_after"),
    ("end", "start_before", "end"),
]
def boundary_debug_json(item: dict, boundary_review_notes: list[str]) -> str:
    debug = dict(item.get("boundary_debug", {}))
    if boundary_review_notes:
        debug["review_notes"] = boundary_review_notes
        debug["review_note_count"] = len(boundary_review_notes)
    if not debug:
        return ""
    return json.dumps(debug, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def snapshot_targets(detections: list[dict], duration: float) -> list[dict]:
    targets: list[dict] = []
    adjacent_frame_step = 1.0 / 24.0
    max_snapshot_time = max(0.0, duration - 0.5)
    for index, item in enumerate(detections, start=1):
        start = float(item["start"])
        end = float(item["end"])
        if end <= start:
            targets.append({"detection": index, "label": "start", "time": start})
            continue

        span = end - start
        mid = start + span / 2
        points = [
            ("start_before", max(0.0, start - adjacent_frame_step), False),
            ("start", min(max_snapshot_time, start), False),
            ("middle", min(max_snapshot_time, mid), False),
            ("end", min(max_snapshot_time, max(start, end - adjacent_frame_step)), False),
            ("end_after", min(max_snapshot_time, end + adjacent_frame_step), end >= duration - 0.5),
        ]
        for label, point_time, is_placeholder in points:
            targets.append({"detection": index, "label": label, "time": point_time, "placeholder": is_placeholder})
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
) -> tuple[dict[int, dict[str, str]], dict[int, list[str]]]:
    targets = snapshot_targets(detections, duration)
    if not targets:
        return {}, {}

    snapshot_dir = output_dir / f"{video_path.stem}.ad_frames"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    for old_file in snapshot_dir.glob("*.jpg"):
        old_file.unlink()

    snapshots: dict[int, dict[str, str]] = {}
    raw_frames: dict[int, dict[str, np.ndarray]] = {}
    raw_times: dict[int, dict[str, float]] = {}
    contact_frames: list[np.ndarray] = []
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    pending: list[tuple[dict, float, Path, Path]] = []

    for target in targets:
        detection_index = target["detection"]
        label = target["label"]
        if target.get("placeholder"):
            snapshots.setdefault(detection_index, {})[label] = "__VIDEO_END__"
            continue
        target_time = float(target["time"])
        filename = f"{detection_index:02d}_{label}_{safe_time_name(target_time)}.jpg"
        path = snapshot_dir / filename
        raw_path = snapshot_dir / f".{filename}.raw.jpg"
        pending.append((target, target_time, path, raw_path))

    extract_raw_snapshots(ffmpeg, video_path, [(target_time, raw_path) for _, target_time, _, raw_path in pending])

    for target, target_time, path, raw_path in pending:
        detection_index = target["detection"]
        label = target["label"]
        if not raw_path.exists():
            continue
        frame = cv_imread(raw_path)
        raw_path.unlink(missing_ok=True)
        if frame is None:
            continue
        raw_frames.setdefault(detection_index, {})[label] = frame
        raw_times.setdefault(detection_index, {})[label] = target_time

        display = overlay_label(frame, f"ad {detection_index} {label} {format_time_precise(target_time)}")
        cv_imwrite(path, display)
        report_path = path.relative_to(output_dir.parent)
        snapshots.setdefault(detection_index, {})[label] = report_path.as_posix()
        contact_frames.append(cv2.resize(display, (320, 180), interpolation=cv2.INTER_AREA))

    if write_contact_sheet and contact_frames:
        columns = len(SNAPSHOT_LABELS)
        rows = []
        blank = np.zeros_like(contact_frames[0])
        for index in range(0, len(contact_frames), columns):
            row = contact_frames[index : index + columns]
            while len(row) < columns:
                row.append(blank)
            rows.append(np.hstack(row))
        cv_imwrite(output_dir / f"{video_path.stem}.ads.keyframes.jpg", np.vstack(rows))

    boundary_notes: dict[int, list[str]] = {}
    for detection_index, frames in raw_frames.items():
        times = raw_times.get(detection_index, {})
        for side, left_label, right_label in BOUNDARY_ERROR_PAIRS:
            left_frame = frames.get(left_label)
            right_frame = frames.get(right_label)
            if left_frame is None or right_frame is None:
                continue
            if abs(float(times.get(left_label, 0.0)) - float(times.get(right_label, 0.0))) < 1e-4:
                continue
            score = frame_diff_score(left_frame, right_frame)
            if score < MIN_BOUNDARY_FRAME_DIFF:
                boundary_notes.setdefault(detection_index, []).append(
                    f"{side}_boundary_error={left_label}~{right_label}:{score:.3f}"
                )

    return snapshots, boundary_notes

def write_outputs(
    output_dir: Path,
    video_path: Path,
    detections: list[dict],
    duration: float,
    write_debug_files: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    snapshots, boundary_notes = extract_snapshots(video_path, detections, duration, output_dir, write_debug_files)

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
                "boundary_debug",
                "start_before_frame",
                "start_frame",
                "middle_frame",
                "end_frame",
                "end_after_frame",
            ],
        )
        writer.writeheader()
        for index, item in enumerate(detections, start=1):
            item_snapshots = snapshots.get(index, {})
            boundary_review_notes = boundary_notes.get(index, [])
            review_required = (item["kind"] == "auto_discovery" and float(item["score"]) < AUTO_TRUST_THRESHOLD) or bool(boundary_review_notes)
            sources = sorted(set(item["sources"]) | set(boundary_review_notes))
            writer.writerow(
                {
                    "start": format_time(item["start"]),
                    "end": format_time(item["end"]) if item["end"] > item["start"] else "",
                    "start_seconds": round(item["start"], 3),
                    "end_seconds": round(item["end"], 3) if item["end"] > item["start"] else "",
                    "score": round(item["score"], 4),
                    "kind": item["kind"],
                    "sources": ";".join(sources),
                    "review_required": "yes" if review_required else "no",
                    "boundary_debug": boundary_debug_json(item, boundary_review_notes),
                    "start_before_frame": item_snapshots.get("start_before", ""),
                    "start_frame": item_snapshots.get("start", ""),
                    "middle_frame": item_snapshots.get("middle", ""),
                    "end_frame": item_snapshots.get("end", ""),
                    "end_after_frame": item_snapshots.get("end_after", ""),
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


class ReviewAssetWriter:
    def __init__(self, output_dir: Path, write_debug_files: bool) -> None:
        self.output_dir = output_dir
        self.write_debug_files = write_debug_files

    def write(self, video_path: Path, detections: list[dict], duration: float) -> None:
        write_outputs(self.output_dir, video_path, detections, duration, self.write_debug_files)
