#!/usr/bin/env python3
from ad_detector.cli import main
from ad_detector.features import frame_feature
from ad_detector.time_utils import format_time_precise, parse_time, safe_time_name


if __name__ == "__main__":
    raise SystemExit(main())
