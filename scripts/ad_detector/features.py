from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def frame_feature(frame: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame, (32, 18), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).flatten()
    hist = hist.astype(np.float32)
    hist /= np.linalg.norm(hist) + 1e-8

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    mean = np.array(
        [
            rgb[:, :, 0].mean(),
            rgb[:, :, 1].mean(),
            rgb[:, :, 2].mean(),
            gray.mean(),
            gray.std(),
        ],
        dtype=np.float32,
    )
    coarse = cv2.resize(rgb, (8, 5), interpolation=cv2.INTER_AREA).flatten()
    feat = np.concatenate([coarse, hist, mean])
    feat /= np.linalg.norm(feat) + 1e-8
    return feat

def frame_metrics(frame: np.ndarray, previous_gray: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    edges = cv2.Canny(gray, 80, 160)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32)
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    rg = np.abs(red - green)
    yb = np.abs(0.5 * (red + green) - blue)

    motion = 0.0
    if previous_gray is not None:
        motion = float(np.mean(cv2.absdiff(gray, previous_gray)) / 255.0)

    metrics = np.array(
        [
            float(gray.mean() / 255.0),
            float(gray.std() / 255.0),
            float(hsv[:, :, 1].mean() / 255.0),
            float(np.count_nonzero(edges) / edges.size),
            float((rg.std() + yb.std()) / 255.0),
            motion,
        ],
        dtype=np.float32,
    )
    return metrics, gray

def cosine_window_scores(features: np.ndarray, template: np.ndarray) -> np.ndarray:
    window = len(template)
    if window > len(features):
        return np.asarray([], dtype=np.float32)
    scores = np.empty(len(features) - window + 1, dtype=np.float32)
    for index in range(len(scores)):
        scores[index] = float(np.mean(np.sum(features[index : index + window] * template, axis=1)))
    return scores

def resample_feature_sequence(features: np.ndarray, target_length: int) -> np.ndarray:
    if target_length <= 0 or len(features) == 0:
        return np.asarray([], dtype=np.float32)
    if target_length == len(features):
        return features
    if len(features) == 1:
        return np.repeat(features, target_length, axis=0)

    source_x = np.linspace(0.0, 1.0, len(features), dtype=np.float32)
    target_x = np.linspace(0.0, 1.0, target_length, dtype=np.float32)
    resampled = np.vstack(
        [np.interp(target_x, source_x, features[:, column]) for column in range(features.shape[1])]
    ).T.astype(np.float32)
    norms = np.linalg.norm(resampled, axis=1, keepdims=True)
    return resampled / (norms + 1e-8)

def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))

def robust_norm(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    low = float(np.percentile(values, 10))
    high = float(np.percentile(values, 90))
    if high - low < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)

def duration_score(length: float) -> float:
    if length < 4:
        return 0.0
    if length <= 8:
        return (length - 4) / 4
    if length <= 35:
        return 1.0
    if length <= 70:
        return 1.0 - (length - 35) / 35
    return 0.0

def frame_diff_score(left: np.ndarray, right: np.ndarray) -> float:
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    if left_gray.shape != right_gray.shape:
        right_gray = cv2.resize(right_gray, (left_gray.shape[1], left_gray.shape[0]), interpolation=cv2.INTER_AREA)
    left_small = cv2.resize(left_gray, (160, 90), interpolation=cv2.INTER_AREA)
    right_small = cv2.resize(right_gray, (160, 90), interpolation=cv2.INTER_AREA)
    return float(np.mean(cv2.absdiff(left_small, right_small)) / 255.0)
