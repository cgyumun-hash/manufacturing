"""SHAP · ablation 실험 설정 (4.3).

이 실험은 3장 8개 모델 비교보다 **먼저 했던 파생변수 탐색 실험**이다.
- 분할 시드가 다르다 (20261005). 그래서 3장 OOF 성능표와 숫자를 직접 합치지 않는다.
- 파생변수(Tabular 19개 · Sequence 12채널)만 입력으로 쓰고, 그룹을 하나씩 빼며 F1이 얼마나 떨어지는지 본다.
- SHAP은 Tabular · Sequence 각각에서 fold 평균 F1이 가장 높았던 모델 하나(winner)에 대해 계산한다
  → One-Class SVM 1.5초, 1D-CNN 1.5초.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.data import load_segments  # noqa: E402
from common.windows import make_windows  # noqa: E402

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
SEED = 20261005
N_FOLDS = 5
LENGTHS = [5, 10, 15, 20]
TABULAR_MODELS = ["LightGBM", "XGBoost", "Logistic Regression", "One-Class SVM", "Mahalanobis distance",
                  "Isolation Forest", "PCA reconstruction"]
TEMPORAL_MODELS = ["LSTM-Autoencoder", "1D-CNN"]


def configuration_seed(fold: int, length: int, family: str, model_name: str) -> int:
    models = TABULAR_MODELS if family == "tabular" else TEMPORAL_MODELS
    return SEED + fold * 10000 + length * 100 + (0 if family == "tabular" else 50) + models.index(model_name)


def prepare_windows():
    eligible = [s for s in load_segments() if s["samples"] >= max(LENGTHS)]
    by_length = {}
    for length in LENGTHS:
        w = make_windows(eligible, length, capped=True)
        by_length[length] = (w["raw"], w["labels"], w["groups"])
    frame = pd.DataFrame({"group": by_length[20][2], "label": by_length[20][1]}).drop_duplicates("group")
    frame = frame.sort_values("group", kind="stable")
    return by_length, frame.group.to_numpy(), frame.label.to_numpy()


def make_splits(group_names, group_labels):
    outer = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    result = []
    for fold, (remain, test) in enumerate(outer.split(group_names, group_labels), start=1):
        train, valid = train_test_split(group_names[remain], test_size=0.25, stratify=group_labels[remain],
                                        random_state=SEED + fold)
        result.append({"fold": fold, "train": np.asarray(train), "valid": np.asarray(valid), "test": group_names[test]})
    return result


def split_raw(raw, y, groups, split):
    masks = {key: np.isin(groups, split[key]) for key in ("train", "valid", "test")}
    return ({k: raw[m] for k, m in masks.items()}, {k: y[m] for k, m in masks.items()}, {k: groups[m] for k, m in masks.items()})


def scale_features(values: dict, labels: dict, temporal: bool) -> dict:
    normal = values["train"][labels["train"] == 0]
    if temporal:
        scaler = StandardScaler().fit(normal.reshape(-1, normal.shape[-1]))
        return {k: scaler.transform(v.reshape(-1, v.shape[-1])).reshape(v.shape).astype(np.float32) for k, v in values.items()}
    scaler = StandardScaler().fit(normal)
    return {k: scaler.transform(v).astype(np.float32) for k, v in values.items()}


def select_columns(values: dict, names: list, selected: list, temporal: bool) -> dict:
    index = [names.index(n) for n in selected]
    return {k: (v[:, :, index] if temporal else v[:, index]) for k, v in values.items()}
