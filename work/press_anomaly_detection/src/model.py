"""LightGBM 모델과 입력 전처리.

입력: window 1개 = 10시점 × 3센서(AI0 상부 진동, AI1 하부 진동, AI2 전류)를 시간 순서대로 펼친 30개 값.
전처리: train 정상 window의 평균·표준편차로 표준화 (valid · test에는 같은 값을 적용만 함).
출력: window 이상 점수 (0 ~ 1, 이상일 확률).
"""
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.preprocessing import StandardScaler

from .config import LGBM_PARAMS, MODEL_DIR


def flatten(raw: np.ndarray) -> np.ndarray:
    return raw.reshape(len(raw), -1)


def fit(train_raw: np.ndarray, train_labels: np.ndarray, seed: int):
    x = flatten(train_raw)
    scaler = StandardScaler().fit(x[train_labels == 0])
    model = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=seed).fit(
        scaler.transform(x).astype(np.float32), train_labels)
    return scaler, model


def score(scaler, model, raw: np.ndarray) -> np.ndarray:
    return model.predict_proba(scaler.transform(flatten(raw)).astype(np.float32))[:, 1]


def fold_dir(fold: int) -> Path:
    return MODEL_DIR / f"fold{fold}"


def save(scaler, model, fold: int) -> None:
    folder = fold_dir(fold)
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, folder / "scaler.joblib")
    joblib.dump(model, folder / "lightgbm.joblib")
    model.booster_.save_model(str(folder / "lightgbm.txt"))  # 버전과 무관하게 읽을 수 있는 텍스트 형식


def load(fold: int):
    folder = fold_dir(fold)
    return joblib.load(folder / "scaler.joblib"), joblib.load(folder / "lightgbm.joblib")
