from __future__ import annotations

from pathlib import Path

import numpy as np

from .features import cosine_window_scores, resample_feature_sequence
from .matching import merge_candidates, overlaps
from .models import Keypoint, Template
from .time_utils import parse_time, safe_time_name


def sample_step_from_times(times: np.ndarray) -> float:
    return float(np.median(np.diff(times))) if len(times) > 1 else 1.0

def sample_rate_from_times(times: np.ndarray) -> float:
    step = sample_step_from_times(times)
    return 1.0 / step if step > 0 else 1.0

def template_sample_rate(item: dict) -> float:
    sample_rate = item.get("sample_rate")
    if sample_rate not in (None, ""):
        return float(sample_rate)
    duration = float(item.get("duration") or 0)
    features = item.get("features")
    if duration > 0 and features is not None and len(features) > 0:
        return len(features) / duration
    return 1.0


def detect_from_templates(
    times: np.ndarray,
    features: np.ndarray,
    keypoints: list[Keypoint],
    threshold: float,
    min_gap: float,
    duration: float,
) -> list[dict]:
    candidates: list[dict] = []
    sample_step = sample_step_from_times(times)
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
    sample_rate = sample_rate_from_times(times)
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
                "sample_rate": sample_rate,
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
            sample_rate=np.asarray([template_sample_rate(item)], dtype=np.float32),
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
                "sample_rate": float(data["sample_rate"][0]) if "sample_rate" in data else None,
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
    sample_step = sample_step_from_times(times)
    for template_item in templates:
        template_duration = float(template_item["duration"])
        target_length = max(3, int(round(template_duration / sample_step)))
        template = resample_feature_sequence(template_item["features"], target_length)
        scores = cosine_window_scores(features, template)
        if len(scores) == 0:
            continue

        picked: list[tuple[float, float, float]] = []
        for score_index in np.argsort(scores)[::-1]:
            score = float(scores[score_index])
            if score < threshold:
                break
            start = float(times[score_index])
            end = start + template_duration
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


class TemplateStore:
    def __init__(self, template_dir: Path) -> None:
        self.template_dir = template_dir

    def load(self) -> list[dict]:
        return load_templates(self.template_dir)

    def save(self, video_path: Path, templates: list[dict]) -> None:
        save_templates(self.template_dir, video_path, templates)
