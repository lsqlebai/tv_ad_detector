from pathlib import Path

from .models import Keypoint
from .time_utils import parse_time


def parse_keypoints(path: Path) -> list[Keypoint]:
    if not path.exists() or path.is_dir():
        return []

    keypoints: list[Keypoint] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        if "-" in raw:
            start_raw, end_raw = raw.split("-", 1)
            end = parse_time(end_raw) if end_raw.strip() else None
        else:
            start_raw, end = raw, None
        keypoints.append(Keypoint(parse_time(start_raw), end, raw))
    return keypoints
