from __future__ import annotations

import numpy as np


def overlaps(start: float, end: float, other_start: float, other_end: float) -> bool:
    return max(start, other_start) < min(end, other_end)

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
                if item["kind"] in {"template_match", "template_library"}:
                    # Overlapping template windows are competing estimates of
                    # the same occurrence. Preserve the better window instead
                    # of expanding its boundaries to the union of both.
                    if float(item["score"]) > float(existing["score"]):
                        old_sources = existing["sources"]
                        existing.update(item)
                        existing["sources"] = sorted(set(old_sources) | set(item["sources"]))
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
