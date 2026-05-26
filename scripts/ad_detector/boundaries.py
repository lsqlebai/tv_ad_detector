from __future__ import annotations

from pathlib import Path
from typing import Optional

from .features import frame_diff_score
from .models import MIN_BOUNDARY_FRAME_DIFF
from .time_utils import parse_time
from .video_io import boundary_frames, read_scaled_frame


def strongest_cut_time(rows: list[tuple[float, float, float, float]], center: float, radius: float) -> Optional[float]:
    candidates = [row for row in rows if abs(row[0] - center) <= radius]
    if not candidates:
        return None
    time_value, _, _, diff = max(candidates, key=lambda row: row[3])
    if diff < 0.14:
        return None
    return time_value

def strongest_cut_time_after(rows: list[tuple[float, float, float, float]], center: float, radius: float) -> Optional[float]:
    candidates = [row for row in rows if center <= row[0] <= center + radius]
    if not candidates:
        return None
    time_value, _, _, diff = max(candidates, key=lambda row: row[3])
    if diff < 0.14:
        return None
    return time_value

def strongest_cut_time_between(
    rows: list[tuple[float, float, float, float]],
    lower: float,
    upper: float,
) -> Optional[float]:
    candidates = [row for row in rows if lower <= row[0] <= upper]
    if not candidates:
        return None
    time_value, _, _, diff = max(candidates, key=lambda row: row[3])
    if diff < 0.14:
        return None
    return time_value

def leading_cut_time(
    rows: list[tuple[float, float, float, float]],
    center: float,
    radius: float,
) -> Optional[float]:
    candidates = [row for row in rows if center - radius <= row[0] <= center + radius and row[3] >= 0.14]
    if not candidates:
        return None
    max_diff = max(row[3] for row in candidates)
    strong = [row for row in candidates if row[3] >= max(0.14, max_diff * 0.72)]
    return min(row[0] for row in strong)

def dark_transition_end(rows: list[tuple[float, float, float, float]], center: float, sample_rate: float) -> Optional[float]:
    dark_times = [time_value for time_value, brightness, dark_ratio, _ in rows if time_value >= center - 1.0 and brightness < 0.16 and dark_ratio > 0.65]
    if not dark_times:
        return None
    return max(dark_times) + 1.0 / sample_rate

def any_dark_transition_end(
    rows: list[tuple[float, float, float, float]],
    lower: float,
    upper: float,
    sample_rate: float,
) -> Optional[float]:
    dark_times = [time_value for time_value, brightness, dark_ratio, _ in rows if lower <= time_value <= upper and brightness < 0.08 and dark_ratio > 0.9]
    if not dark_times:
        return None
    return max(dark_times) + 1.0 / sample_rate

def trailing_cut_time(
    rows: list[tuple[float, float, float, float]],
    center: float,
    radius: float,
    sample_rate: float,
) -> Optional[float]:
    candidates = [row for row in rows if center - 0.25 <= row[0] <= center + radius and row[3] >= 0.14]
    if not candidates:
        return None
    max_diff = max(row[3] for row in candidates)
    strong = [row for row in candidates if row[3] >= max(0.14, max_diff * 0.72)]
    return max(row[0] for row in strong) + 1.0 / sample_rate

def boundary_has_cut(rows: list[tuple[float, float, float, float]], center: float, radius: float = 0.2) -> bool:
    return any(abs(time_value - center) <= radius and diff >= 0.14 for time_value, _, _, diff in rows)

def strong_cut_candidates(
    rows: list[tuple[float, float, float, float]],
    lower: float,
    upper: float,
) -> list[tuple[float, float, float, float]]:
    candidates = [row for row in rows if lower <= row[0] <= upper and row[3] >= 0.14]
    if not candidates:
        return []
    max_diff = max(row[3] for row in candidates)
    threshold = max(0.14, max_diff * 0.72)
    return [row for row in candidates if row[3] >= threshold]

def nearest_cut_time(
    rows: list[tuple[float, float, float, float]],
    center: float,
    lower: float,
    upper: float,
) -> Optional[float]:
    strong = strong_cut_candidates(rows, lower, upper)
    if not strong:
        return None
    time_value, _, _, _ = min(strong, key=lambda row: abs(row[0] - center))
    return time_value

def nearest_boundary_cut(
    rows: list[tuple[float, float, float, float]],
    center: float,
    lower: float,
    upper: float,
    min_diff: float = 0.14,
) -> Optional[float]:
    candidates = [row for row in rows if lower <= row[0] <= upper and row[3] >= min_diff]
    if not candidates:
        return None
    time_value, _, _, _ = min(candidates, key=lambda row: abs(row[0] - center))
    return time_value

def find_boundary_repair_cut(
    video_path: Path,
    side: str,
    boundary_time: float,
    opposite_time: float,
    duration: float,
    sample_rate: float,
    search_radius: float,
) -> Optional[float]:
    lower = max(0.0, boundary_time - search_radius)
    upper = min(duration, boundary_time + search_radius)
    if side == "start":
        upper = min(upper, opposite_time - 0.5)
    elif side == "end":
        lower = max(lower, opposite_time + 0.5)
    else:
        raise ValueError(f"Unknown boundary side: {side}")
    rows = boundary_frames(video_path, lower, upper, sample_rate, duration)
    cut_time = nearest_boundary_cut(rows, boundary_time, lower, upper)
    if cut_time is not None and abs(cut_time - boundary_time) <= 1.5 / sample_rate:
        if side == "start":
            cut_time = normalize_template_start_cut(video_path, cut_time, sample_rate)
        else:
            cut_time = max(0.0, cut_time - 1.0 / sample_rate)
    return cut_time

def template_source_start(item: dict) -> Optional[float]:
    for source in item.get("sources", []):
        text = str(source)
        if not text.startswith("template=") or "-" not in text:
            continue
        start_text = text.removeprefix("template=").split("-", 1)[0]
        try:
            return parse_time(start_text)
        except ValueError:
            return None
    return None

def normalize_template_start_cut(video_path: Path, cut_time: Optional[float], sample_rate: float) -> Optional[float]:
    if cut_time is None or cut_time <= 1.0 / sample_rate:
        return cut_time
    previous_frame = read_scaled_frame(video_path, cut_time - 1.0 / sample_rate)
    current_frame = read_scaled_frame(video_path, cut_time)
    if previous_frame is None or current_frame is None:
        return cut_time
    if frame_diff_score(previous_frame, current_frame) < 0.10:
        return max(0.0, cut_time - 1.0 / sample_rate)
    return cut_time

def refine_detection_boundaries(
    video_path: Path,
    detections: list[dict],
    duration: float,
    sample_rate: float = 24.0,
    radius: float = 2.5,
) -> list[dict]:
    refined: list[dict] = []
    for item in detections:
        result = item.copy()
        if item["kind"] not in {"template_match", "template_library", "auto_discovery"} or float(item["end"]) <= float(item["start"]):
            refined.append(result)
            continue

        start = float(item["start"])
        end = float(item["end"])
        start_radius = 4.0 if item["kind"] == "auto_discovery" else radius
        end_radius = 6.0 if item["kind"] == "auto_discovery" else radius
        start_rows = boundary_frames(video_path, start - start_radius, start + start_radius, sample_rate, duration)
        end_rows = boundary_frames(video_path, end - end_radius, end + radius, sample_rate, duration)

        if item["kind"] == "auto_discovery":
            if start <= 1.0:
                refined_start = start
            else:
                refined_start = leading_cut_time(start_rows, start, radius=start_radius) or strongest_cut_time(start_rows, start, radius=start_radius)
            refined_end = (
                any_dark_transition_end(end_rows, end - end_radius, end + radius, sample_rate)
                or trailing_cut_time(end_rows, end, radius=end_radius, sample_rate=sample_rate)
                or strongest_cut_time(end_rows, end, radius=end_radius)
            )
            if duration - end <= 8.0:
                refined_end = duration
        else:
            refined_start = normalize_template_start_cut(
                video_path,
                strongest_cut_time_between(start_rows, max(0.0, start - 0.5), min(duration, start + 1.5)),
                sample_rate,
            )
            refined_end = dark_transition_end(end_rows, end, sample_rate) or trailing_cut_time(end_rows, end, radius=1.5, sample_rate=sample_rate) or strongest_cut_time(end_rows, end, radius=1.5)

        notes = []
        if refined_start is not None and abs(refined_start - start) <= start_radius:
            result["start"] = max(0.0, min(duration, refined_start))
            notes.append(f"start_refined={start:.3f}->{result['start']:.3f}")
        if refined_end is not None and (abs(refined_end - end) <= end_radius or abs(refined_end - duration) < 1e-6):
            result["end"] = max(float(result["start"]), min(duration, refined_end))
            notes.append(f"end_refined={end:.3f}->{result['end']:.3f}")
        if notes:
            result["sources"] = sorted(set(result["sources"]) | set(notes))
        refined.append(result)
    return refined

def repair_unclear_boundaries(
    video_path: Path,
    detections: list[dict],
    duration: float,
    sample_rate: float = 24.0,
    search_radius: float = 8.0,
) -> list[dict]:
    repaired: list[dict] = []
    for item in detections:
        result = item.copy()
        if item["kind"] not in {"template_match", "template_library", "auto_discovery"} or float(item["end"]) <= float(item["start"]):
            repaired.append(result)
            continue

        start = float(item["start"])
        end = float(item["end"])
        notes: list[str] = []

        if start > 0.5:
            start_rows = boundary_frames(video_path, start - search_radius, start + search_radius, sample_rate, duration)
            if not boundary_has_cut(start_rows, start):
                repaired_start = nearest_cut_time(start_rows, start, max(0.0, start - search_radius), min(duration, start + search_radius))
                if repaired_start is not None and abs(repaired_start - start) > 1.0 / sample_rate and repaired_start < end:
                    result["start"] = max(0.0, min(duration, repaired_start))
                    notes.append(f"start_boundary_repaired={start:.3f}->{result['start']:.3f}")

        current_start = float(result["start"])
        if duration - end > 0.5:
            end_rows = boundary_frames(video_path, end - search_radius, end + search_radius, sample_rate, duration)
            if not boundary_has_cut(end_rows, end):
                repaired_end = nearest_cut_time(end_rows, end, max(0.0, end - search_radius), min(duration, end + search_radius))
                if repaired_end is not None:
                    repaired_end = min(duration, repaired_end + 1.0 / sample_rate)
                if repaired_end is not None and abs(repaired_end - end) > 1.0 / sample_rate and repaired_end > current_start:
                    result["end"] = max(current_start, min(duration, repaired_end))
                    notes.append(f"end_boundary_repaired={end:.3f}->{result['end']:.3f}")

        if notes:
            result["sources"] = sorted(set(result["sources"]) | set(notes))
        repaired.append(result)
    return repaired

def repair_snapshot_similar_boundaries(
    video_path: Path,
    detections: list[dict],
    duration: float,
    output_dir: Path,
    write_debug_files: bool,
    sample_rate: float = 24.0,
    search_radius: float = 10.0,
) -> list[dict]:
    from .review_assets import extract_snapshots

    def has_boundary_error(notes: set[str], side: str) -> bool:
        return any(
            note.startswith(f"{side}_boundary_error=")
            or note.startswith(f"{side}_boundary_similar=")
            for note in notes
        )

    repaired = [item.copy() for item in detections]
    for _ in range(3):
        _, boundary_notes = extract_snapshots(video_path, repaired, duration, output_dir, write_debug_files)
        template_start_indexes = {
            index + 1
            for index, item in enumerate(repaired)
            if item["kind"] == "template_library"
            and float(item["end"]) > float(item["start"])
            and (
                (source_start := template_source_start(item)) is not None
                and abs(float(item["start"]) - source_start) > 0.5
            )
            and not any(str(note).startswith("start_boundary_repaired_by_template_cut=") for note in item.get("sources", []))
        }
        candidate_indexes = sorted(set(boundary_notes) | template_start_indexes)
        if not candidate_indexes:
            break

        changed = False
        for index in candidate_indexes:
            notes = boundary_notes.get(index, [])
            item_index = index - 1
            if item_index < 0 or item_index >= len(repaired):
                continue
            item = repaired[item_index]
            if item["kind"] not in {"template_match", "template_library", "auto_discovery"} or float(item["end"]) <= float(item["start"]):
                continue

            note_set = set(notes)
            item_notes: list[str] = []
            start = float(item["start"])
            end = float(item["end"])
            start_has_boundary_error = has_boundary_error(note_set, "start")
            should_check_template_start = item["kind"] == "template_library" and index in template_start_indexes

            if start > 0.5 and (start_has_boundary_error or should_check_template_start):
                repaired_start = find_boundary_repair_cut(
                    video_path,
                    "start",
                    start,
                    end,
                    duration,
                    sample_rate,
                    search_radius,
                )
                if repaired_start is not None and repaired_start < end and abs(repaired_start - start) > 0.5 / sample_rate:
                    item["start"] = max(0.0, min(duration, repaired_start))
                    repair_reason = "boundary_error" if start_has_boundary_error else "template_cut"
                    item_notes.append(f"start_boundary_repaired_by_{repair_reason}={start:.3f}->{item['start']:.3f}")
                    changed = True

            current_start = float(item["start"])
            current_end = float(item["end"])
            if duration - current_end > 0.5 and has_boundary_error(note_set, "end"):
                repaired_end = find_boundary_repair_cut(
                    video_path,
                    "end",
                    current_end,
                    current_start,
                    duration,
                    sample_rate,
                    search_radius,
                )
                if repaired_end is not None and repaired_end > current_start and abs(repaired_end - current_end) > 0.5 / sample_rate:
                    item["end"] = max(current_start, min(duration, repaired_end))
                    item_notes.append(f"end_boundary_repaired_by_boundary_error={current_end:.3f}->{item['end']:.3f}")
                    changed = True

            if item_notes:
                item["sources"] = sorted(set(item["sources"]) | set(item_notes))

        if not changed:
            break

    return repaired



class BoundaryRefiner:
    def __init__(self, output_dir: Path, write_debug_files: bool, sample_rate: float = 24.0) -> None:
        self.output_dir = output_dir
        self.write_debug_files = write_debug_files
        self.sample_rate = sample_rate

    def refine(self, video_path: Path, detections: list[dict], duration: float) -> list[dict]:
        detections = refine_detection_boundaries(video_path, detections, duration, sample_rate=self.sample_rate)
        detections = repair_unclear_boundaries(video_path, detections, duration, sample_rate=self.sample_rate)
        return repair_snapshot_similar_boundaries(
            video_path,
            detections,
            duration,
            self.output_dir,
            self.write_debug_files,
            sample_rate=self.sample_rate,
        )
