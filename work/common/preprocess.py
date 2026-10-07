"""입력 형식(단일시점 / 원데이터 window / 파생)을 모델 입력으로 바꾸는 전처리기.

fit은 train window로만 한다:
- 파생변수의 전류-진동 회귀식
- 표준화(StandardScaler): train **정상** window의 평균·표준편차
valid · test에는 transform만 적용한다.

Tabular 모델: 2차원 (window 수, 변수 수)
  - 단일시점 3개 / 원데이터 시점 수 × 3개를 한 줄로 펼침 / 파생 19개
Sequence 모델: 3차원 (window 수, 시점 수, 채널 수)
  - 단일시점 1 × 3 / 원데이터 시점 수 × 3 / 파생 시점 수 × 12
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from .data import SENSORS
from .features import TabularFeatures, TemporalFeatures


class InputPreprocessor:
    def __init__(self, representation: str, family: str, length: int):
        self.representation = representation
        self.family = family
        self.length = length
        self.maker = None

    @property
    def names(self) -> list[str]:
        if self.representation == "derived":
            return list(self.maker.names)
        if self.family == "tabular" and self.representation == "raw_window":
            return [f"t{point + 1:02d}_{sensor}" for point in range(self.length) for sensor in SENSORS]
        return list(SENSORS)

    def _unscaled(self, raw: np.ndarray) -> np.ndarray:
        if self.representation == "single":
            return raw[:, 0, :] if self.family == "tabular" else raw
        if self.representation == "raw_window":
            return raw.reshape(len(raw), -1) if self.family == "tabular" else raw
        return self.maker.transform(raw)

    def fit(self, raw_train: np.ndarray, y_train: np.ndarray) -> "InputPreprocessor":
        if self.representation == "derived":
            maker = TabularFeatures() if self.family == "tabular" else TemporalFeatures()
            self.maker = maker.fit(raw_train, y_train)
        normal = self._unscaled(raw_train)[y_train == 0]
        if self.family == "tabular":
            self.scaler = StandardScaler().fit(normal)
        else:
            self.scaler = StandardScaler().fit(normal.reshape(-1, normal.shape[-1]))
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        values = self._unscaled(raw)
        if self.family == "tabular":
            return self.scaler.transform(values).astype(np.float32)
        return self.scaler.transform(values.reshape(-1, values.shape[-1])).reshape(values.shape).astype(np.float32)
