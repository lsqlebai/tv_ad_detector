from __future__ import annotations

from typing import Optional


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
    return max(dark_times)


def any_dark_transition_end(
    rows: list[tuple[float, float, float, float]],
    lower: float,
    upper: float,
    sample_rate: float,
) -> Optional[float]:
    dark_times = [time_value for time_value, brightness, dark_ratio, _ in rows if lower <= time_value <= upper and brightness < 0.08 and dark_ratio > 0.9]
    if not dark_times:
        return None
    return max(dark_times)


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
    return min(row[0] for row in strong)


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
