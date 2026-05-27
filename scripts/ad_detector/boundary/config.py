from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class BoundaryRefineSettings:
    start_radius: float
    end_scan_before: float
    end_scan_after: float
    end_pick_radius: float


@dataclass(frozen=True)
class BoundaryContext:
    video_path: Path
    duration: float
    sample_rate: float
    sample_times: Optional[np.ndarray] = None
    sample_metrics: Optional[np.ndarray] = None
    output_dir: Optional[Path] = None
    write_debug_files: bool = False


BOUNDARY_CONFIDENCE_THRESHOLD = 0.70


def boundary_refine_settings(kind: str, default_radius: float) -> BoundaryRefineSettings:
    return BoundaryRefineSettings(
        start_radius=default_radius,
        end_scan_before=default_radius,
        end_scan_after=default_radius,
        end_pick_radius=1.5,
    )


def expanded_boundary_refine_settings() -> BoundaryRefineSettings:
    return BoundaryRefineSettings(
        start_radius=8.0,
        end_scan_before=6.0,
        end_scan_after=6.0,
        end_pick_radius=6.0,
    )
