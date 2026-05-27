from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from .boundaries import BoundaryRefiner
from .discovery import discover_ad_candidates
from .keypoints import parse_keypoints
from .matching import combine_detections
from .models import Detection, DetectionResult
from .review_assets import ReviewAssetWriter
from .templates import (
    TemplateStore,
    build_templates_from_keypoints,
    detect_from_template_library,
    detect_from_templates,
    save_templates,
)
from .time_utils import format_time
from .video_io import sample_video


@dataclass(frozen=True)
class DetectionConfig:
    input_dir: Path
    output_dir: Path
    keypoints: Path
    template_dir: Path
    ignore_keypoints: bool = False
    sample_rate: float = 1.0
    sample_skip_frame: str = "noref"
    threshold: float = 0.94
    auto_threshold: float = 0.80
    max_auto_candidates: int = 8
    min_gap: float = 8.0
    no_auto_discover: bool = False
    no_refine_boundaries: bool = False
    write_debug_files: bool = False


class AdDetectionPipeline:
    def __init__(self, config: DetectionConfig) -> None:
        self.config = config
        self.template_store = TemplateStore(config.template_dir)
        self.review_asset_writer = ReviewAssetWriter(config.output_dir, config.write_debug_files)

    def keypoints(self):
        return [] if self.config.ignore_keypoints else parse_keypoints(self.config.keypoints)

    def run(self, video_path: Path, keypoints: list | None = None) -> DetectionResult:
        config = self.config
        if keypoints is None:
            keypoints = self.keypoints()

        video_started = time.perf_counter()
        print(f"Sampling {video_path.name} at {config.sample_rate:g} fps (skip_frame={config.sample_skip_frame})...")
        stage_started = time.perf_counter()
        times, features, metrics, duration = sample_video(video_path, config.sample_rate, config.sample_skip_frame)
        print(f"  sampled {len(times)} frames in {time.perf_counter() - stage_started:.1f}s")

        keypoint_templates = build_templates_from_keypoints(times, features, keypoints, duration)
        save_templates(config.template_dir, video_path, keypoint_templates)

        stage_started = time.perf_counter()
        detections = detect_from_templates(times, features, keypoints, config.threshold, config.min_gap, duration)
        library_templates = self.template_store.load()
        if library_templates:
            detections.extend(
                detect_from_template_library(
                    times,
                    features,
                    library_templates,
                    config.threshold,
                    config.min_gap,
                )
            )
        print(f"  matched templates in {time.perf_counter() - stage_started:.1f}s")

        if not config.no_auto_discover:
            stage_started = time.perf_counter()
            detections.extend(
                discover_ad_candidates(
                    times,
                    features,
                    metrics,
                    duration,
                    config.auto_threshold,
                    config.min_gap,
                    config.max_auto_candidates,
                )
            )
            print(f"  discovered candidates in {time.perf_counter() - stage_started:.1f}s")

        detections = combine_detections(detections, merge_gap=config.min_gap)
        if not config.no_refine_boundaries:
            stage_started = time.perf_counter()
            detections = BoundaryRefiner(config.output_dir, config.write_debug_files).refine(
                video_path,
                detections,
                duration,
                sample_times=times,
                sample_metrics=metrics,
            )
            print(f"  refined boundaries in {time.perf_counter() - stage_started:.1f}s")

        stage_started = time.perf_counter()
        self.review_asset_writer.write(video_path, detections, duration)
        print(f"  wrote review assets in {time.perf_counter() - stage_started:.1f}s")

        print(f"{video_path.name}:")
        for item in detections:
            if item["end"] <= item["start"]:
                print(f"  {format_time(item['start'])}-  {item['kind']} score={item['score']:.3f}")
            else:
                print(f"  {format_time(item['start'])}-{format_time(item['end'])}  {item['kind']} score={item['score']:.3f}")
        print(f"  total {time.perf_counter() - video_started:.1f}s")

        return DetectionResult(
            video_path=video_path,
            duration=duration,
            detections=[Detection.from_dict(item) for item in detections],
        )
