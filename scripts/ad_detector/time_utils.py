import re


def parse_time(value: str) -> float:
    parts = [float(part) for part in value.strip().split(":")]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return hours * 3600 + minutes * 60 + seconds
    raise ValueError(f"Bad timestamp: {value!r}")

def format_time(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{sec:02d}"
    return f"{minutes}:{sec:02d}"

def format_time_precise(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining = seconds - hours * 3600 - minutes * 60
    if hours:
        return f"{hours}:{minutes:02d}:{remaining:06.3f}"
    return f"{minutes}:{remaining:06.3f}"

def safe_time_name(seconds: float) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "-", format_time_precise(seconds)).strip("-")
