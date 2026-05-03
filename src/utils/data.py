from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class DataSplits:
    full: pd.DataFrame
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def load_price_data(csv_path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(csv_path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame


def split_by_ratio(frame: pd.DataFrame, train_ratio: float, validation_ratio: float) -> DataSplits:
    n = len(frame)
    train_end = int(n * train_ratio)
    validation_end = int(n * (train_ratio + validation_ratio))
    return DataSplits(
        full=frame,
        train=frame.iloc[:train_end].reset_index(drop=True),
        validation=frame.iloc[train_end:validation_end].reset_index(drop=True),
        test=frame.iloc[validation_end:].reset_index(drop=True),
    )


def build_rolling_windows(frame: pd.DataFrame, train_size: int, validation_size: int, test_size: int, step_size: int) -> list[dict]:
    windows = []
    start = 0
    idx = 0
    total = len(frame)
    while start + train_size + validation_size + test_size <= total:
        train = frame.iloc[start : start + train_size].reset_index(drop=True)
        validation = frame.iloc[start + train_size : start + train_size + validation_size].reset_index(drop=True)
        test = frame.iloc[start + train_size + validation_size : start + train_size + validation_size + test_size].reset_index(drop=True)
        windows.append({"window_id": idx, "train": train, "validation": validation, "test": test})
        start += step_size
        idx += 1
    return windows
