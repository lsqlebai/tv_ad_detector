#!/usr/bin/env python3
import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def input_files(input_dir: Path, requested: list[str]) -> list[str]:
    videos = sorted(input_dir.glob("*.mp4"))
    if requested:
        wanted = set(requested)
        videos = [path for path in videos if path.name in wanted]
    if not videos:
        raise SystemExit(f"No .mp4 files found in {input_dir}")
    return [path.name for path in videos]


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
    parser.add_argument("--files", nargs="*", default=[], help="Optional input video filenames to process.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    if args.clean_detect and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    if args.clean_detect:
        args.xlsx.unlink(missing_ok=True)
        args.xlsx.with_suffix(".tmp.xlsx").unlink(missing_ok=True)

    detect_base_command = [
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
        detect_base_command.append("--ignore-keypoints")
    if args.detect_extra:
        detect_base_command.extend(shlex.split(args.detect_extra))

    review_command = [
        sys.executable,
        str(root / "review_ads.py"),
        "--output-dir",
        str(args.output_dir),
        "--xlsx",
        str(args.xlsx),
        "--append",
    ]

    files = input_files(args.input_dir, args.files)
    for index, file_name in enumerate(files, start=1):
        print(f"Processing {index}/{len(files)}: {file_name}", flush=True)
        run([*detect_base_command, "--files", file_name])
        run([*review_command, "--files", file_name])
    print(f"Wrote review workbook: {args.xlsx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
