#!/usr/bin/env python3
import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect ad candidates and build the Excel review workbook.")
    parser.add_argument("--input-dir", type=Path, default=Path("input"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/detect"))
    parser.add_argument("--xlsx", type=Path, default=Path("output/ad_review.xlsx"))
    parser.add_argument("--keypoints", type=Path, default=Path("keypoint.txt"))
    parser.add_argument("--ignore-keypoints", action="store_true", default=True)
    parser.add_argument("--use-keypoints", action="store_false", dest="ignore_keypoints")
    parser.add_argument("--template-dir", type=Path, default=Path("ad_templates"))
    parser.add_argument("--clean-detect", action="store_true", default=True)
    parser.add_argument("--keep-detect", action="store_false", dest="clean_detect")
    parser.add_argument("--detect-extra", default="", help="Extra arguments passed to detect_ads.py.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    if args.clean_detect and args.output_dir.exists():
        shutil.rmtree(args.output_dir)

    detect_command = [
        sys.executable,
        str(root / "detect_ads.py"),
        "--input-dir",
        str(args.input_dir),
        "--output-dir",
        str(args.output_dir),
        "--keypoints",
        str(args.keypoints),
        "--template-dir",
        str(args.template_dir),
    ]
    if args.ignore_keypoints:
        detect_command.append("--ignore-keypoints")
    if args.detect_extra:
        detect_command.extend(args.detect_extra.split())

    review_command = [
        sys.executable,
        str(root / "review_ads.py"),
        "--output-dir",
        str(args.output_dir),
        "--xlsx",
        str(args.xlsx),
    ]

    run(detect_command)
    run(review_command)
    print(f"Wrote review workbook: {args.xlsx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
