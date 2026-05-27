from __future__ import annotations

import argparse
from pathlib import Path

from .keypoints import parse_keypoints
from .pipeline import AdDetectionPipeline, DetectionConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect repeated ad segments from confirmed keypoints.")
    parser.add_argument("--input-dir", type=Path, default=Path("input"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/detect"))
    parser.add_argument("--keypoints", type=Path, default=Path("keypoint.txt"))
    parser.add_argument("--ignore-keypoints", action="store_true", help="Ignore keypoint file and rely on template/auto discovery.")
    parser.add_argument("--template-dir", type=Path, default=Path("ad_templates"), help="Directory used to save/load confirmed ad templates.")
    parser.add_argument("--sample-rate", type=float, default=1.0, help="Frames per second to sample.")
    parser.add_argument(
        "--sample-skip-frame",
        choices=("none", "noref", "nokey"),
        default="noref",
        help="FFmpeg frame skipping used during low-rate sampling. Use none for full decode, noref to skip non-reference frames, or nokey for keyframes only.",
    )
    parser.add_argument("--threshold", type=float, default=0.94, help="Template similarity threshold.")
    parser.add_argument("--auto-threshold", type=float, default=0.80, help="Minimum score for automatic ad candidates.")
    parser.add_argument("--max-auto-candidates", type=int, default=8, help="Maximum automatic candidates per video.")
    parser.add_argument("--min-gap", type=float, default=8.0, help="Minimum seconds between detections.")
    parser.add_argument("--files", nargs="*", default=[], help="Optional input video filenames to process.")
    parser.add_argument("--no-auto-discover", action="store_true", help="Disable automatic ad discovery.")
    parser.add_argument("--no-refine-boundaries", action="store_true", help="Disable high-rate scene-cut and dark-frame boundary refinement.")
    parser.add_argument("--write-debug-files", action="store_true", help="Also write *.ads.txt, *.candidates.txt, *.ads.json, and keyframe contact sheets.")
    args = parser.parse_args()

    keypoints = [] if args.ignore_keypoints else parse_keypoints(args.keypoints)
    if args.ignore_keypoints:
        print("Ignoring keypoints; using template library plus automatic discovery.")
    elif not keypoints:
        print(f"No keypoints found in {args.keypoints}; using template library plus automatic discovery.")

    videos = sorted(args.input_dir.glob("*.mp4"))
    if args.files:
        wanted = set(args.files)
        videos = [path for path in videos if path.name in wanted]
    if not videos:
        raise SystemExit(f"No .mp4 files found in {args.input_dir}")

    config = DetectionConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        keypoints=args.keypoints,
        template_dir=args.template_dir,
        ignore_keypoints=args.ignore_keypoints,
        sample_rate=args.sample_rate,
        sample_skip_frame=args.sample_skip_frame,
        threshold=args.threshold,
        auto_threshold=args.auto_threshold,
        max_auto_candidates=args.max_auto_candidates,
        min_gap=args.min_gap,
        no_auto_discover=args.no_auto_discover,
        no_refine_boundaries=args.no_refine_boundaries,
        write_debug_files=args.write_debug_files,
    )
    pipeline = AdDetectionPipeline(config)
    for video_path in videos:
        pipeline.run(video_path, keypoints=keypoints)

    return 0
