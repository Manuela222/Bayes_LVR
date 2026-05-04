from __future__ import annotations

import math
from typing import Iterable

import numpy as np


EPS = 1e-12


def safe_clip(value: float | np.ndarray, low: float, high: float) -> float | np.ndarray:
    return np.clip(value, low, high)


def safe_sqrt(value: float | np.ndarray, eps: float = EPS) -> float | np.ndarray:
    return np.sqrt(np.maximum(value, eps))


def ensure_finite_array(values: Iterable[float], fill_value: float = 0.0) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float32)
    return np.nan_to_num(arr, nan=fill_value, posinf=fill_value, neginf=fill_value)


def safe_log_return(current: float, previous: float, eps: float = EPS) -> float:
    return float(math.log(max(current, eps) / max(previous, eps)))


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or len(y) < 2:
        return 0.0
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return 0.0
    if float(np.std(x)) < EPS or float(np.std(y)) < EPS:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def safe_rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return 0.0
    xr = np.argsort(np.argsort(x))
    yr = np.argsort(np.argsort(y))
    return safe_corr(xr.astype(float), yr.astype(float))
