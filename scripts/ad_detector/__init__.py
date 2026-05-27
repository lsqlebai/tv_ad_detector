from __future__ import annotations

import logging

from .models import Detection, DetectionResult, Keypoint, Template, VideoSample
from .time_utils import format_time, format_time_precise, parse_time, safe_time_name


class _ImageioFrameSizeFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not str(record.getMessage()).startswith("The frame size for reading ")


_imageio_logger = logging.getLogger("imageio_ffmpeg")
if not any(isinstance(item, _ImageioFrameSizeFilter) for item in _imageio_logger.filters):
    _imageio_logger.addFilter(_ImageioFrameSizeFilter())
