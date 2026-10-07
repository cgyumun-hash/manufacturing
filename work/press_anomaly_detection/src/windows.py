"""window 생성과 segment 단위 5-fold Group CV 분할.

- window는 segment 안에서만 만든다 (공백을 넘지 않음).
- capped=True (학습·평가): segment 끝 window를 추가하고 segment당 최대 10개를 고르게 고른다.
- capped=False (실시간 재생, 마르코프): 0.5초 간격 window를 모두 쓴다.
- 같은 segment의 window는 train / valid / test 중 한 곳에만 들어간다.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import MAX_WINDOWS_PER_SEGMENT, N_FOLDS, SEED, STRIDE, VALID_SIZE, WINDOW


def _choose_evenly(items, maximum):
    if len(items) <= maximum:
        return items
    positions = np.linspace(0, len(items) - 1, maximum).round().astype(int)
    return [items[i] for i in np.unique(positions)]


def make_windows(segments, capped=True):
    raw, labels, groups, starts = [], [], [], []
    for seg in segments:
        last = seg["samples"] - WINDOW
        if last < 0:
            continue
        index = list(range(0, last + 1, STRIDE))
        if capped:
            if index[-1] != last:
                index.append(last)
            index = _choose_evenly(index, MAX_WINDOWS_PER_SEGMENT)
        for start in index:
            raw.append(seg["values"][start:start + WINDOW])
            labels.append(seg["label"])
            groups.append(seg["group"])
            starts.append(start)
    return {"raw": np.stack(raw).astype(np.float32), "labels": np.asarray(labels, dtype=np.int64),
            "groups": np.asarray(groups), "starts": np.asarray(starts, dtype=np.int64)}


def make_splits(segments):
    names = np.asarray([s["group"] for s in segments])
    labels = np.asarray([s["label"] for s in segments], dtype=np.int64)
    outer = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = []
    for fold, (remain, test) in enumerate(outer.split(names, labels), start=1):
        train, valid = train_test_split(names[remain], test_size=VALID_SIZE, stratify=labels[remain],
                                        random_state=SEED + fold)
        splits.append({"fold": fold, "train": np.asarray(train), "valid": np.asarray(valid), "test": names[test]})
    return splits


def select(windows, names):
    mask = np.isin(windows["groups"], names)
    return {key: value[mask] for key, value in windows.items()}
