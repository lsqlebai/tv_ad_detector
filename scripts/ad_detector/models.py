from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Keypoint:
    start: float
    end: Optional[float]
    raw: str


@dataclass
class Detection:
    start: float
    end: float
    score: float
    kind: str
    end_after: Optional[float] = None
    sources: list[str] = field(default_factory=list)
    snapshots: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, item: dict) -> "Detection":
        return cls(
            start=float(item["start"]),
            end=float(item["end"]),
            score=float(item["score"]),
            kind=str(item["kind"]),
            end_after=float(item["end_after"]) if item.get("end_after") not in (None, "") else None,
            sources=[str(source) for source in item.get("sources", [])],
            snapshots={str(key): str(value) for key, value in item.get("snapshots", {}).items()},
        )

    def to_dict(self) -> dict:
        payload = {
            "start": self.start,
            "end": self.end,
            "score": self.score,
            "kind": self.kind,
            "sources": list(self.sources),
        }
        if self.end_after is not None:
            payload["end_after"] = self.end_after
        if self.snapshots:
            payload["snapshots"] = dict(self.snapshots)
        return payload


@dataclass(frozen=True)
class Template:
    features: np.ndarray
    duration: float
    source: str
    path: Optional[str] = None


@dataclass(frozen=True)
class VideoSample:
    times: np.ndarray
    features: np.ndarray
    metrics: np.ndarray
    duration: float


@dataclass(frozen=True)
class BoundaryFrame:
    time: float
    brightness: float
    dark_ratio: float
    diff: float


@dataclass
class DetectionResult:
    video_path: Path
    duration: float
    detections: list[Detection]


METRIC_COLUMNS = [
    "brightness",
    "contrast",
    "saturation",
    "edge_density",
    "colorfulness",
    "motion",
]
AUTO_TRUST_THRESHOLD = 0.98
MIN_BOUNDARY_FRAME_DIFF = 0.025
SNAPSHOT_LABELS = [
    "start_before",
    "start",
    "middle",
    "end",
    "end_after",
]
