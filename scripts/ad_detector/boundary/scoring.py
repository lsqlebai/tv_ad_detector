from __future__ import annotations

from typing import Optional


def cut_confidence(
    rows: list[tuple[float, float, float, float]],
    cut_time: Optional[float],
    center: float,
    radius: float,
    sample_rate: float,
) -> float:
    if cut_time is None or not rows:
        return 0.0

    frame_step = 1.0 / sample_rate
    nearby = [
        row
        for row in rows
        if abs(row[0] - cut_time) <= frame_step * 1.5
        or abs(row[0] - max(0.0, cut_time - frame_step)) <= frame_step * 1.5
    ]
    if not nearby:
        nearby = [min(rows, key=lambda row: abs(row[0] - cut_time))]

    _, _, _, diff = max(nearby, key=lambda row: row[3])
    local_diffs = [row[3] for row in rows if abs(row[0] - center) <= radius]
    if not local_diffs:
        local_diffs = [row[3] for row in rows]
    local_max = max(local_diffs) + 1e-8
    relative_strength = min(1.0, diff / local_max)
    absolute_strength = min(1.0, diff / 0.20)
    proximity = max(0.0, 1.0 - abs(cut_time - center) / max(radius, 1e-6))
    return 0.65 * relative_strength + 0.25 * absolute_strength + 0.10 * proximity
