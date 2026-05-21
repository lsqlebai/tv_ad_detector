#!/usr/bin/env python3
import argparse
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "dist" / "tv_ad_detector_docker_bundle.zip"

FILES = [
    ".dockerignore",
    "Dockerfile",
    "README.md",
    "docker-compose.yml",
    "requirements.txt",
]

DIRECTORIES = [
    "scripts",
    "ad_templates",
]

PLACEHOLDERS = [
    "input/.gitkeep",
    "output/.gitkeep",
]


def add_file(zip_file: zipfile.ZipFile, path: Path, arcname: str) -> None:
    if path.exists() and path.is_file():
        zip_file.write(path, arcname)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a NAS-ready Docker deployment zip.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output_path = args.output
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for item in FILES:
            add_file(zip_file, ROOT / item, item)

        for directory in DIRECTORIES:
            for path in sorted((ROOT / directory).rglob("*")):
                if path.is_file() and "__pycache__" not in path.parts:
                    zip_file.write(path, path.relative_to(ROOT).as_posix())

        for placeholder in PLACEHOLDERS:
            zip_file.writestr(placeholder, "")

    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
