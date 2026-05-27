from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..time_utils import parse_time
from ..video_io import boundary_frames
from .alignment import EdgeResolution, normalize_template_start_cut, resolve_ad_edge
from .config import BOUNDARY_CONFIDENCE_THRESHOLD, BoundaryContext
from .cuts import boundary_has_cut, nearest_boundary_cut, nearest_cut_time


def find_boundary_repair_cut(
    video_path: Path,
    side: str,
    boundary_time: float,
    opposite_time: float,
    duration: float,
    sample_rate: float,
    search_radius: float,
) -> Optional[EdgeResolution]:
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
        return resolve_ad_edge(video_path, cut_time, sample_rate, side)
    return None


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


def has_confident_refined_boundary(item: dict, side: str) -> bool:
    debug = item.get("boundary_debug")
    if not isinstance(debug, dict):
        return False
    score = debug.get(f"{side}_score")
    try:
        return score is not None and float(score) >= BOUNDARY_CONFIDENCE_THRESHOLD
    except (TypeError, ValueError):
        return False


def mark_repair_skip(result: dict, side: str) -> None:
    debug = dict(result.get("boundary_debug", {}))
    debug[f"{side}_unclear_repair_skipped"] = True
    result["boundary_debug"] = debug


def high_confidence_template_fast_path(item: dict) -> bool:
    if item.get("kind") != "template_library":
        return False
    debug = item.get("boundary_debug")
    if not isinstance(debug, dict):
        return False
    try:
        start_score = float(debug.get("start_score"))
        end_score = float(debug.get("end_score"))
        start_visual = float(debug.get("start_visual_score"))
        end_visual = float(debug.get("end_visual_score"))
    except (TypeError, ValueError):
        return False
    return (
        start_score >= 0.80
        and end_score >= 0.80
        and start_visual >= 0.65
        and end_visual >= 0.65
        and not debug.get("start_expanded_used")
        and not debug.get("end_expanded_used")
    )


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

        if start > 0.5 and has_confident_refined_boundary(item, "start"):
            mark_repair_skip(result, "start")
        elif start > 0.5:
            start_rows = boundary_frames(video_path, start - search_radius, start + search_radius, sample_rate, duration)
            if not boundary_has_cut(start_rows, start):
                repaired_start = nearest_cut_time(start_rows, start, max(0.0, start - search_radius), min(duration, start + search_radius))
                if repaired_start is not None:
                    repaired_start = normalize_template_start_cut(video_path, repaired_start, sample_rate)
                if repaired_start is not None and abs(repaired_start - start) > 1.0 / sample_rate and repaired_start < end:
                    result["start"] = max(0.0, min(duration, repaired_start))
                    notes.append(f"start_boundary_repaired={start:.3f}->{result['start']:.3f}")

        current_start = float(result["start"])
        if duration - end > 0.5 and has_confident_refined_boundary(item, "end"):
            mark_repair_skip(result, "end")
        elif duration - end > 0.5:
            end_rows = boundary_frames(video_path, end - search_radius, end + search_radius, sample_rate, duration)
            if not boundary_has_cut(end_rows, end):
                repaired_end = nearest_cut_time(end_rows, end, max(0.0, end - search_radius), min(duration, end + search_radius))
                if repaired_end is not None:
                    repaired_end_edge = resolve_ad_edge(video_path, repaired_end, sample_rate, "end")
                    repaired_end = repaired_end_edge.ad_frame if repaired_end_edge is not None else None
                    repaired_end_after = repaired_end_edge.content_frame if repaired_end_edge is not None else None
                else:
                    repaired_end_after = None
                if repaired_end is not None and abs(repaired_end - end) > 1.0 / sample_rate and repaired_end > current_start:
                    result["end"] = max(current_start, min(duration, repaired_end))
                    if repaired_end_after is not None and repaired_end_after > result["end"]:
                        result["end_after"] = min(duration, repaired_end_after)
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
    from ..review_assets import extract_snapshots

    def has_boundary_error(notes: set[str], side: str) -> bool:
        return any(
            note.startswith(f"{side}_boundary_error=")
            or note.startswith(f"{side}_boundary_similar=")
            for note in notes
        )

    repaired = [item.copy() for item in detections]
    for _ in range(3):
        fast_path_indexes = {
            index
            for index, item in enumerate(repaired, start=1)
            if high_confidence_template_fast_path(item)
        }
        validation_items = [
            item
            for index, item in enumerate(repaired, start=1)
            if index not in fast_path_indexes
        ]
        validation_index_map = [
            index
            for index, _ in enumerate(repaired, start=1)
            if index not in fast_path_indexes
        ]
        if validation_items:
            _, subset_boundary_notes = extract_snapshots(video_path, validation_items, duration, output_dir, write_debug_files)
            boundary_notes = {
                validation_index_map[subset_index - 1]: notes
                for subset_index, notes in subset_boundary_notes.items()
                if 0 < subset_index <= len(validation_index_map)
            }
        else:
            boundary_notes = {}
        template_start_indexes = {
            index + 1
            for index, item in enumerate(repaired)
            if item["kind"] == "template_library"
            and index + 1 not in fast_path_indexes
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
                repaired_start_edge = find_boundary_repair_cut(
                    video_path,
                    "start",
                    start,
                    end,
                    duration,
                    sample_rate,
                    search_radius,
                )
                repaired_start = repaired_start_edge.ad_frame if repaired_start_edge is not None else None
                if repaired_start is not None and repaired_start < end and abs(repaired_start - start) > 0.5 / sample_rate:
                    item["start"] = max(0.0, min(duration, repaired_start))
                    repair_reason = "boundary_error" if start_has_boundary_error else "template_cut"
                    item_notes.append(f"start_boundary_repaired_by_{repair_reason}={start:.3f}->{item['start']:.3f}")
                    changed = True

            current_start = float(item["start"])
            current_end = float(item["end"])
            if duration - current_end > 0.5 and has_boundary_error(note_set, "end"):
                repaired_end_edge = find_boundary_repair_cut(
                    video_path,
                    "end",
                    current_end,
                    current_start,
                    duration,
                    sample_rate,
                    search_radius,
                )
                repaired_end = repaired_end_edge.ad_frame if repaired_end_edge is not None else None
                repaired_end_after = repaired_end_edge.content_frame if repaired_end_edge is not None else None
                if repaired_end is not None and repaired_end > current_start and abs(repaired_end - current_end) > 0.5 / sample_rate:
                    item["end"] = max(current_start, min(duration, repaired_end))
                    if repaired_end_after is not None and repaired_end_after > item["end"]:
                        item["end_after"] = min(duration, repaired_end_after)
                    item_notes.append(f"end_boundary_repaired_by_boundary_error={current_end:.3f}->{item['end']:.3f}")
                    changed = True

            if item_notes:
                item["sources"] = sorted(set(item["sources"]) | set(item_notes))
                debug = dict(item.get("boundary_debug", {}))
                debug["snapshot_repair_notes"] = item_notes
                debug["snapshot_repair_count"] = len(item_notes)
                item["boundary_debug"] = debug

        if not changed:
            break

    return repaired


def validate_detection_boundaries(
    context: BoundaryContext,
    detections: list[dict],
) -> list[dict]:
    if context.output_dir is None:
        raise ValueError("BoundaryContext.output_dir is required for snapshot validation")
    detections = repair_unclear_boundaries(
        context.video_path,
        detections,
        context.duration,
        sample_rate=context.sample_rate,
    )
    return repair_snapshot_similar_boundaries(
        context.video_path,
        detections,
        context.duration,
        context.output_dir,
        context.write_debug_files,
        sample_rate=context.sample_rate,
    )
