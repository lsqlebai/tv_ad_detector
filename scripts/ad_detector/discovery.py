from __future__ import annotations

import numpy as np

from .features import clamp, duration_score, robust_norm
from .matching import overlaps
from .models import METRIC_COLUMNS


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
