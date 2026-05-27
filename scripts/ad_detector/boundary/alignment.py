from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..features import frame_diff_score
from ..video_io import boundary_frames, read_scaled_frame
from .config import BOUNDARY_CONFIDENCE_THRESHOLD, BoundaryContext, BoundaryRefineSettings, boundary_refine_settings, expanded_boundary_refine_settings
from .cuts import any_dark_transition_end, dark_transition_end, leading_cut_time, strongest_cut_time, strongest_cut_time_between, trailing_cut_time
from .scoring import cut_confidence


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


def refine_start_cut(
    video_path: Path,
    rows: list[tuple[float, float, float, float]],
    center: float,
    radius: float,
    sample_rate: float,
    duration: float,
) -> Optional[float]:
    if center <= 1.0:
        return center
    cut_time = (
        strongest_cut_time_between(rows, max(0.0, center - 0.5), min(duration, center + 1.5))
        or leading_cut_time(rows, center, radius=radius)
        or strongest_cut_time(rows, center, radius=radius)
    )
    return normalize_template_start_cut(video_path, cut_time, sample_rate)


def refine_end_cut(
    rows: list[tuple[float, float, float, float]],
    center: float,
    sample_rate: float,
    duration: float,
    settings: BoundaryRefineSettings,
) -> Optional[float]:
    refined_end = (
        dark_transition_end(rows, center, sample_rate)
        or any_dark_transition_end(rows, center - settings.end_scan_before, center + settings.end_scan_after, sample_rate)
        or trailing_cut_time(rows, center, radius=settings.end_pick_radius, sample_rate=sample_rate)
        or strongest_cut_time(rows, center, radius=settings.end_pick_radius)
    )
    return refined_end


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
        settings = boundary_refine_settings(str(item["kind"]), radius)
        start_rows = boundary_frames(video_path, start - settings.start_radius, start + settings.start_radius, sample_rate, duration)
        end_rows = boundary_frames(video_path, end - settings.end_scan_before, end + settings.end_scan_after, sample_rate, duration)

        refined_start = refine_start_cut(
            video_path,
            start_rows,
            start,
            settings.start_radius,
            sample_rate,
            duration,
        )
        start_score = 1.0 if start <= 1.0 and refined_start is not None else cut_confidence(
            start_rows,
            refined_start,
            start,
            settings.start_radius,
            sample_rate,
        )
        refined_end = refine_end_cut(end_rows, end, sample_rate, duration, settings)
        end_score = cut_confidence(end_rows, refined_end, end, settings.end_pick_radius, sample_rate)

        expanded_settings = expanded_boundary_refine_settings()
        start_expanded_attempt = start_score < BOUNDARY_CONFIDENCE_THRESHOLD
        start_expanded_used = False
        end_expanded_attempt = end_score < BOUNDARY_CONFIDENCE_THRESHOLD
        end_expanded_used = False
        if start_expanded_attempt:
            expanded_start_rows = boundary_frames(
                video_path,
                start - expanded_settings.start_radius,
                start + expanded_settings.start_radius,
                sample_rate,
                duration,
            )
            expanded_start = refine_start_cut(
                video_path,
                expanded_start_rows,
                start,
                expanded_settings.start_radius,
                sample_rate,
                duration,
            )
            expanded_start_score = cut_confidence(
                expanded_start_rows,
                expanded_start,
                start,
                expanded_settings.start_radius,
                sample_rate,
            )
            if expanded_start is not None and (refined_start is None or expanded_start_score > start_score):
                refined_start = expanded_start
                start_score = expanded_start_score
                start_expanded_used = True

        if end_expanded_attempt:
            expanded_end_rows = boundary_frames(
                video_path,
                end - expanded_settings.end_scan_before,
                end + expanded_settings.end_scan_after,
                sample_rate,
                duration,
            )
            expanded_end = refine_end_cut(expanded_end_rows, end, sample_rate, duration, expanded_settings)
            expanded_end_score = cut_confidence(
                expanded_end_rows,
                expanded_end,
                end,
                expanded_settings.end_pick_radius,
                sample_rate,
            )
            if expanded_end is not None and (refined_end is None or expanded_end_score > end_score):
                refined_end = expanded_end
                end_score = expanded_end_score
                end_expanded_used = True

        notes = []
        if refined_start is not None and abs(refined_start - start) <= expanded_settings.start_radius:
            result["start"] = max(0.0, min(duration, refined_start))
            notes.append(f"start_refined={start:.3f}->{result['start']:.3f}")
        if refined_end is not None and (
            abs(refined_end - end) <= expanded_settings.end_scan_before
            or abs(refined_end - duration) < 1e-6
        ):
            result["end"] = max(float(result["start"]), min(duration, refined_end))
            notes.append(f"end_refined={end:.3f}->{result['end']:.3f}")
        if notes:
            result["sources"] = sorted(set(result["sources"]) | set(notes))
        debug = dict(result.get("boundary_debug", {}))
        debug.update(
            {
                "start_score": round(start_score, 3),
                "end_score": round(end_score, 3),
                "start_expanded_attempt": start_expanded_attempt,
                "start_expanded_used": start_expanded_used,
                "end_expanded_attempt": end_expanded_attempt,
                "end_expanded_used": end_expanded_used,
                "start_seconds_before": round(start, 3),
                "end_seconds_before": round(end, 3),
            }
        )
        result["boundary_debug"] = debug
        refined.append(result)
    return refined


def align_detection_boundaries(context: BoundaryContext, detections: list[dict]) -> list[dict]:
    return refine_detection_boundaries(
        context.video_path,
        detections,
        context.duration,
        sample_rate=context.sample_rate,
    )
