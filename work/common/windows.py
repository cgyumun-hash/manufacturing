"""window 생성과 segment 단위 Group CV 분할 (2.6 · 2.7 규칙).

- 모든 입력 구성은 2초(20시점) 입력이 가능한 같은 segment만 쓴다 (정상 452 · 이상 13).
- window는 segment 안에서만 만들고, 이동 간격 = max(1, 시점 수 // 2).
- 학습·평가용(capped): segment 끝의 window를 추가하고, segment당 최대 10개를 고르게 고른다.
- 실시간 재생용(full, 5장): 일정 간격 window를 모두 쓰고 끝 window를 따로 추가하지 않는다.
- 같은 segment의 window는 train / valid / test 중 한 곳에만 들어간다.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

SEED = 20261007
N_FOLDS = 5
MIN_SAMPLES = 20
MAX_WINDOWS_PER_GROUP = 10

INPUTS = [
    {"key": "raw_single", "label": "단일시점", "kind": "원변수 단일시점", "representation": "single", "length": 1},
    {"key": "raw_window_5", "label": "원데이터 0.5초", "kind": "원데이터 window", "representation": "raw_window", "length": 5},
    {"key": "raw_window_10", "label": "원데이터 1.0초", "kind": "원데이터 window", "representation": "raw_window", "length": 10},
    {"key": "raw_window_15", "label": "원데이터 1.5초", "kind": "원데이터 window", "representation": "raw_window", "length": 15},
    {"key": "raw_window_20", "label": "원데이터 2.0초", "kind": "원데이터 window", "representation": "raw_window", "length": 20},
    {"key": "derived_5", "label": "파생 0.5초", "kind": "파생변수/파생채널", "representation": "derived", "length": 5},
    {"key": "derived_10", "label": "파생 1.0초", "kind": "파생변수/파생채널", "representation": "derived", "length": 10},
    {"key": "derived_15", "label": "파생 1.5초", "kind": "파생변수/파생채널", "representation": "derived", "length": 15},
    {"key": "derived_20", "label": "파생 2.0초", "kind": "파생변수/파생채널", "representation": "derived", "length": 20},
]
INPUT_BY_KEY = {item["key"]: item for item in INPUTS}
INPUT_ORDER = [item["label"] for item in INPUTS]


def stride_of(length: int) -> int:
    return max(1, length // 2)


def eligible_segments(segments: list[dict]) -> list[dict]:
    return [segment for segment in segments if segment["samples"] >= MIN_SAMPLES]


def choose_evenly(items: list[int], maximum: int) -> list[int]:
    if len(items) <= maximum:
        return items
    positions = np.linspace(0, len(items) - 1, maximum).round().astype(int)
    return [items[index] for index in np.unique(positions)]


def window_starts(samples: int, length: int, capped: bool = True) -> list[int]:
    last_start = samples - length
    if last_start < 0:
        return []
    starts = list(range(0, last_start + 1, stride_of(length)))
    if capped:
        if starts[-1] != last_start:
            starts.append(last_start)
        starts = choose_evenly(starts, MAX_WINDOWS_PER_GROUP)
    return starts


def make_windows(segments: list[dict], length: int, capped: bool = True) -> dict[str, np.ndarray]:
    """segment 안에서만 window를 만든다. raw 모양 = (window 수, 시점 수, 센서 3)."""
    raw, labels, groups, starts = [], [], [], []
    for segment in segments:
        for start in window_starts(segment["samples"], length, capped):
            raw.append(segment["values"][start:start + length])
            labels.append(segment["label"])
            groups.append(segment["group"])
            starts.append(start)
    return {
        "raw": np.stack(raw).astype(np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
        "groups": np.asarray(groups),
        "starts": np.asarray(starts, dtype=np.int64),
    }


def make_splits(group_names: np.ndarray, group_labels: np.ndarray) -> list[dict]:
    """5-fold Group CV. 각 fold에서 test를 뺀 나머지를 train 75% : valid 25%로 나눈다."""
    outer = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = []
    for fold, (remain_index, test_index) in enumerate(outer.split(group_names, group_labels), start=1):
        train_names, valid_names = train_test_split(
            group_names[remain_index], test_size=0.25, stratify=group_labels[remain_index], random_state=SEED + fold
        )
        splits.append({"fold": fold, "train": np.asarray(train_names), "valid": np.asarray(valid_names),
                       "test": np.asarray(group_names[test_index])})
    return splits


def select(windows: dict[str, np.ndarray], names: np.ndarray) -> dict[str, np.ndarray]:
    mask = np.isin(windows["groups"], names)
    return {key: value[mask] for key, value in windows.items()}


def prepare():
    """segment, 그룹 목록, fold 분할을 한 번에 만든다."""
    from .data import load_segments

    eligible = eligible_segments(load_segments())
    names = np.asarray([segment["group"] for segment in eligible])
    labels = np.asarray([segment["label"] for segment in eligible], dtype=np.int64)
    return eligible, names, labels, make_splits(names, labels)
