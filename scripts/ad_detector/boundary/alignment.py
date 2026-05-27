from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..features import frame_diff_score, robust_norm
from ..models import METRIC_COLUMNS
from ..video_io import boundary_frames, read_scaled_frame
from .config import BOUNDARY_CONFIDENCE_THRESHOLD, BoundaryContext, BoundaryRefineSettings, boundary_refine_settings, expanded_boundary_refine_settings
from .cuts import any_dark_transition_end, dark_transition_end, leading_cut_time, strong_cut_candidates, strongest_cut_time, strongest_cut_time_between, trailing_cut_time
from .scoring import cut_confidence


@dataclass(frozen=True)
class EdgeResolution:
    ad_frame: float
    content_frame: Optional[float] = None


@dataclass(frozen=True)
class ScoredEdge:
    resolution: EdgeResolution
    score: float
    cut_score: float
    visual_score: Optional[float] = None
    ad_visual_score: Optional[float] = None
    content_visual_score: Optional[float] = None


def normalize_template_start_cut(video_path: Path, cut_time: Optional[float], sample_rate: float) -> Optional[float]:
    resolution = resolve_ad_edge(video_path, cut_time, sample_rate, "start")
    return resolution.ad_frame if resolution is not None else None


def ad_frame_time_for_edge(side: str, left_time: float, right_time: float) -> float:
    if side == "start":
        return right_time
    if side == "end":
        return left_time
    raise ValueError(f"Unknown boundary side: {side}")


def edge_resolution_for_cut(side: str, left_time: float, right_time: float) -> EdgeResolution:
    if side == "start":
        return EdgeResolution(ad_frame=right_time, content_frame=left_time)
    if side == "end":
        return EdgeResolution(ad_frame=left_time, content_frame=right_time)
    raise ValueError(f"Unknown boundary side: {side}")


def approximate_edge_resolution(side: str, cut_time: float, sample_rate: float, duration: float) -> EdgeResolution:
    frame_step = 1.0 / sample_rate
    if side == "start":
        return EdgeResolution(ad_frame=max(0.0, min(duration, cut_time)), content_frame=max(0.0, cut_time - frame_step))
    if side == "end":
        return EdgeResolution(ad_frame=max(0.0, cut_time - frame_step), content_frame=min(duration, cut_time))
    raise ValueError(f"Unknown boundary side: {side}")


def resolve_ad_edge_frame(
    video_path: Path,
    cut_time: Optional[float],
    sample_rate: float,
    side: str,
    search_frames: int = 4,
) -> Optional[float]:
    resolution = resolve_ad_edge(video_path, cut_time, sample_rate, side, search_frames)
    return resolution.ad_frame if resolution is not None else None


def resolve_ad_edge(
    video_path: Path,
    cut_time: Optional[float],
    sample_rate: float,
    side: str,
    search_frames: int = 4,
) -> Optional[EdgeResolution]:
    if cut_time is None:
        return None
    frame_step = 1.0 / sample_rate
    if cut_time <= frame_step:
        return EdgeResolution(ad_frame=max(0.0, cut_time))

    offsets = range(-search_frames, search_frames + 2)
    frames = {offset: read_scaled_frame(video_path, cut_time + offset * frame_step) for offset in offsets}
    if any(frame is None for frame in frames.values()):
        return EdgeResolution(ad_frame=max(0.0, cut_time))

    candidates: list[tuple[float, float, EdgeResolution]] = []
    for offset in range(-search_frames + 1, search_frames):
        previous_frame = frames[offset - 1]
        left_frame = frames[offset]
        right_frame = frames[offset + 1]
        next_frame = frames[offset + 2]
        if previous_frame is None or left_frame is None or right_frame is None or next_frame is None:
            continue
        left_stability = frame_diff_score(previous_frame, left_frame)
        boundary_diff = frame_diff_score(left_frame, right_frame)
        right_stability = frame_diff_score(right_frame, next_frame)
        if boundary_diff < 0.10:
            continue
        stability_penalty = min(left_stability, 0.12) + min(right_stability, 0.12)
        distance_penalty = abs(offset) * 0.01
        score = boundary_diff - 0.45 * stability_penalty - distance_penalty
        if score <= 0:
            continue
        left_time = cut_time + offset * frame_step
        right_time = left_time + frame_step
        candidates.append((score, -abs(offset), edge_resolution_for_cut(side, left_time, right_time)))
    if candidates:
        _, _, resolution = max(candidates, key=lambda item: (item[0], item[1]))
        return EdgeResolution(
            ad_frame=max(0.0, resolution.ad_frame),
            content_frame=max(0.0, resolution.content_frame) if resolution.content_frame is not None else None,
        )
    return EdgeResolution(ad_frame=max(0.0, cut_time))


def visual_score_series(sample_metrics: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if sample_metrics is None or len(sample_metrics) == 0:
        return None
    metric_index = {name: index for index, name in enumerate(METRIC_COLUMNS)}
    normalized = np.column_stack([robust_norm(sample_metrics[:, index]) for index in range(sample_metrics.shape[1])])
    return (
        0.30 * normalized[:, metric_index["saturation"]]
        + 0.30 * normalized[:, metric_index["edge_density"]]
        + 0.20 * normalized[:, metric_index["contrast"]]
        + 0.20 * normalized[:, metric_index["colorfulness"]]
    ).astype(np.float32)


def window_visual_score(
    sample_times: Optional[np.ndarray],
    visual_scores: Optional[np.ndarray],
    start: float,
    end: float,
) -> Optional[float]:
    if sample_times is None or visual_scores is None or end <= start:
        return None
    mask = (sample_times >= start) & (sample_times < end)
    if np.count_nonzero(mask) == 0:
        return None
    return float(np.mean(visual_scores[mask]))


def edge_ad_visual_score(
    side: str,
    resolution: EdgeResolution,
    sample_times: Optional[np.ndarray],
    visual_scores: Optional[np.ndarray],
    duration: float,
    window: float = 3.0,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    if side == "start":
        ad_visual = window_visual_score(
            sample_times,
            visual_scores,
            resolution.ad_frame,
            min(duration, resolution.ad_frame + window),
        )
        return ad_visual, ad_visual, None
    if side == "end":
        ad_visual = window_visual_score(
            sample_times,
            visual_scores,
            max(0.0, resolution.ad_frame - window),
            resolution.ad_frame,
        )
        content_start = resolution.content_frame if resolution.content_frame is not None else resolution.ad_frame
        content_visual = window_visual_score(
            sample_times,
            visual_scores,
            content_start,
            min(duration, content_start + window),
        )
        if ad_visual is None and content_visual is None:
            return None, None, None
        if content_visual is None:
            return ad_visual, ad_visual, content_visual
        if ad_visual is None:
            return max(0.0, 1.0 - content_visual), ad_visual, content_visual
        return max(0.0, 0.75 * ad_visual + 0.25 * (1.0 - content_visual)), ad_visual, content_visual
    raise ValueError(f"Unknown boundary side: {side}")


def candidate_cut_times(
    side: str,
    rows: list[tuple[float, float, float, float]],
    center: float,
    sample_rate: float,
    duration: float,
    settings: BoundaryRefineSettings,
) -> list[float]:
    if side == "start":
        lower = max(0.0, center - settings.start_radius)
        upper = min(duration, center + settings.start_radius)
    elif side == "end":
        lower = max(0.0, center - settings.end_scan_before)
        upper = min(duration, center + settings.end_scan_after)
    else:
        raise ValueError(f"Unknown boundary side: {side}")

    candidates = [time_value for time_value, _, _, _ in strong_cut_candidates(rows, lower, upper)]
    if side == "end":
        for cut_time in (
            dark_transition_end(rows, center, sample_rate),
            any_dark_transition_end(rows, center - settings.end_scan_before, center + settings.end_scan_after, sample_rate),
        ):
            if cut_time is not None:
                candidates.append(cut_time)
    seen: set[float] = set()
    unique: list[float] = []
    for cut_time in sorted(candidates, key=lambda value: abs(value - center)):
        key = round(cut_time, 3)
        if key in seen:
            continue
        seen.add(key)
        unique.append(cut_time)
    return unique


def boundary_candidate_time(
    side: str,
    rows: list[tuple[float, float, float, float]],
    center: float,
    sample_rate: float,
    duration: float,
    settings: BoundaryRefineSettings,
) -> Optional[float]:
    if side == "start":
        return (
            strongest_cut_time_between(rows, max(0.0, center - 0.5), min(duration, center + 1.5))
            or leading_cut_time(rows, center, radius=settings.start_radius)
            or strongest_cut_time(rows, center, radius=settings.start_radius)
        )
    if side == "end":
        return (
            dark_transition_end(rows, center, sample_rate)
            or any_dark_transition_end(rows, center - settings.end_scan_before, center + settings.end_scan_after, sample_rate)
            or trailing_cut_time(rows, center, radius=settings.end_pick_radius, sample_rate=sample_rate)
            or strongest_cut_time(rows, center, radius=settings.end_pick_radius)
        )
    raise ValueError(f"Unknown boundary side: {side}")


def refined_time_from_ad_frame(side: str, ad_frame_time: float, sample_rate: float, duration: float) -> float:
    if side == "start":
        return max(0.0, ad_frame_time)
    if side == "end":
        return min(duration, ad_frame_time)
    raise ValueError(f"Unknown boundary side: {side}")


def refine_cut(
    side: str,
    video_path: Path,
    rows: list[tuple[float, float, float, float]],
    center: float,
    sample_rate: float,
    duration: float,
    settings: BoundaryRefineSettings,
) -> Optional[float]:
    resolution = refine_edge(side, video_path, rows, center, sample_rate, duration, settings)
    if resolution is None:
        return None
    return refined_time_from_ad_frame(side, resolution.ad_frame, sample_rate, duration)


def refine_edge_candidate(
    side: str,
    video_path: Path,
    rows: list[tuple[float, float, float, float]],
    center: float,
    sample_rate: float,
    duration: float,
    settings: BoundaryRefineSettings,
    sample_times: Optional[np.ndarray] = None,
    visual_scores: Optional[np.ndarray] = None,
) -> Optional[ScoredEdge]:
    if side == "start" and center <= 1.0:
        resolution = EdgeResolution(ad_frame=center)
        return ScoredEdge(resolution=resolution, score=1.0, cut_score=1.0, visual_score=1.0, ad_visual_score=1.0)

    candidates = candidate_cut_times(side, rows, center, sample_rate, duration, settings)
    fallback_cut_time = boundary_candidate_time(side, rows, center, sample_rate, duration, settings)
    if fallback_cut_time is not None:
        candidates.append(fallback_cut_time)
    if not candidates:
        return None

    rough_scored: list[tuple[float, float, float]] = []
    seen: set[float] = set()
    for cut_time in candidates:
        key = round(cut_time, 3)
        if key in seen:
            continue
        seen.add(key)
        resolution = approximate_edge_resolution(side, cut_time, sample_rate, duration)
        cut_score = cut_confidence(
            rows,
            resolution.content_frame if side == "end" and resolution.content_frame is not None else resolution.ad_frame,
            center,
            settings.end_pick_radius if side == "end" else settings.start_radius,
            sample_rate,
        )
        visual_score, ad_visual, content_visual = edge_ad_visual_score(side, resolution, sample_times, visual_scores, duration)
        score = cut_score if visual_score is None else 0.45 * cut_score + 0.55 * visual_score
        rough_scored.append((score, cut_score, cut_time))
    if not rough_scored:
        return None
    _, _, best_cut_time = max(rough_scored, key=lambda item: item[0])

    resolution = resolve_ad_edge(video_path, best_cut_time, sample_rate, side)
    if resolution is None:
        return None
    resolution = EdgeResolution(
        ad_frame=refined_time_from_ad_frame(side, resolution.ad_frame, sample_rate, duration),
        content_frame=min(duration, resolution.content_frame) if resolution.content_frame is not None else None,
    )
    cut_score = cut_confidence(
        rows,
        resolution.content_frame if side == "end" and resolution.content_frame is not None else resolution.ad_frame,
        center,
        settings.end_pick_radius if side == "end" else settings.start_radius,
        sample_rate,
    )
    visual_score, ad_visual, content_visual = edge_ad_visual_score(side, resolution, sample_times, visual_scores, duration)
    score = cut_score if visual_score is None else 0.45 * cut_score + 0.55 * visual_score
    return ScoredEdge(
        resolution=resolution,
        score=score,
        cut_score=cut_score,
        visual_score=visual_score,
        ad_visual_score=ad_visual,
        content_visual_score=content_visual,
    )


def refine_edge(
    side: str,
    video_path: Path,
    rows: list[tuple[float, float, float, float]],
    center: float,
    sample_rate: float,
    duration: float,
    settings: BoundaryRefineSettings,
    sample_times: Optional[np.ndarray] = None,
    visual_scores: Optional[np.ndarray] = None,
) -> Optional[EdgeResolution]:
    scored = refine_edge_candidate(side, video_path, rows, center, sample_rate, duration, settings, sample_times, visual_scores)
    return scored.resolution if scored is not None else None


def refine_detection_boundaries(
    video_path: Path,
    detections: list[dict],
    duration: float,
    sample_rate: float = 24.0,
    radius: float = 2.5,
    sample_times: Optional[np.ndarray] = None,
    sample_metrics: Optional[np.ndarray] = None,
) -> list[dict]:
    refined: list[dict] = []
    visual_scores = visual_score_series(sample_metrics)
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

        refined_start_candidate = refine_edge_candidate(
            "start",
            video_path,
            start_rows,
            start,
            sample_rate,
            duration,
            settings,
            sample_times,
            visual_scores,
        )
        refined_start = refined_start_candidate.resolution.ad_frame if refined_start_candidate is not None else None
        start_score = refined_start_candidate.score if refined_start_candidate is not None else 0.0
        start_visual_score = refined_start_candidate.visual_score if refined_start_candidate is not None else None
        refined_end_candidate = refine_edge_candidate("end", video_path, end_rows, end, sample_rate, duration, settings, sample_times, visual_scores)
        refined_end = refined_end_candidate.resolution.ad_frame if refined_end_candidate is not None else None
        refined_end_after = refined_end_candidate.resolution.content_frame if refined_end_candidate is not None else None
        end_score = refined_end_candidate.score if refined_end_candidate is not None else 0.0
        end_visual_score = refined_end_candidate.visual_score if refined_end_candidate is not None else None

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
            expanded_start_candidate = refine_edge_candidate(
                "start",
                video_path,
                expanded_start_rows,
                start,
                sample_rate,
                duration,
                expanded_settings,
                sample_times,
                visual_scores,
            )
            expanded_start = expanded_start_candidate.resolution.ad_frame if expanded_start_candidate is not None else None
            expanded_start_score = expanded_start_candidate.score if expanded_start_candidate is not None else 0.0
            if expanded_start is not None and (refined_start is None or expanded_start_score > start_score):
                refined_start = expanded_start
                start_score = expanded_start_score
                start_visual_score = expanded_start_candidate.visual_score if expanded_start_candidate is not None else None
                start_expanded_used = True

        if end_expanded_attempt:
            expanded_end_rows = boundary_frames(
                video_path,
                end - expanded_settings.end_scan_before,
                end + expanded_settings.end_scan_after,
                sample_rate,
                duration,
            )
            expanded_end_candidate = refine_edge_candidate(
                "end",
                video_path,
                expanded_end_rows,
                end,
                sample_rate,
                duration,
                expanded_settings,
                sample_times,
                visual_scores,
            )
            expanded_end = expanded_end_candidate.resolution.ad_frame if expanded_end_candidate is not None else None
            expanded_end_after = expanded_end_candidate.resolution.content_frame if expanded_end_candidate is not None else None
            expanded_end_score = expanded_end_candidate.score if expanded_end_candidate is not None else 0.0
            if expanded_end is not None and (refined_end is None or expanded_end_score > end_score):
                refined_end = expanded_end
                refined_end_after = expanded_end_after
                end_score = expanded_end_score
                end_visual_score = expanded_end_candidate.visual_score if expanded_end_candidate is not None else None
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
            if refined_end_after is not None and refined_end_after > result["end"]:
                result["end_after"] = min(duration, refined_end_after)
            else:
                result["end_after"] = result["end"]
            notes.append(f"end_refined={end:.3f}->{result['end']:.3f}")
        if notes:
            result["sources"] = sorted(set(result["sources"]) | set(notes))
        debug = dict(result.get("boundary_debug", {}))
        debug.update(
            {
                "start_score": round(start_score, 3),
                "end_score": round(end_score, 3),
                "start_visual_score": round(start_visual_score, 3) if start_visual_score is not None else None,
                "end_visual_score": round(end_visual_score, 3) if end_visual_score is not None else None,
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
        sample_times=context.sample_times,
        sample_metrics=context.sample_metrics,
    )
