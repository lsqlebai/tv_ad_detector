from __future__ import annotations

from pathlib import Path

from .alignment import align_detection_boundaries
from .config import BoundaryContext
from .validation import validate_detection_boundaries


class BoundaryRefiner:
    def __init__(self, output_dir: Path, write_debug_files: bool, sample_rate: float = 24.0) -> None:
        self.output_dir = output_dir
        self.write_debug_files = write_debug_files
        self.sample_rate = sample_rate

    def refine(self, video_path: Path, detections: list[dict], duration: float) -> list[dict]:
        context = BoundaryContext(
            video_path,
            duration,
            self.sample_rate,
            output_dir=self.output_dir,
            write_debug_files=self.write_debug_files,
        )
        detections = align_detection_boundaries(context, detections)
        return validate_detection_boundaries(context, detections)
