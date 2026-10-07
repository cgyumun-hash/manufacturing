"""파생변수 계산 (3.4 Tabular 19개 · 3.5 Sequence 12채널).

전류로 진동을 설명하는 회귀식(residual)은 **train 정상 window에서만** 맞춘다(fit).
valid · test에는 그 식을 그대로 적용한다(transform). 그래서 평가 데이터 정보가 새지 않는다.
"""
from __future__ import annotations

import numpy as np

SHORT_RMS_SAMPLES = 3

TABULAR_GROUPS = {
    "G1": ["AI0_RMS", "AI1_RMS"],
    "G2": ["AI0_AI1_RMS_diff", "AI0_AI1_RMS_ratio", "AI0_AI1_corr"],
    "G3": ["Current_RMS", "Current_RMS_to_AI0_RMS_ratio", "Current_RMS_to_AI1_RMS_ratio",
           "AI0_current_residual", "AI1_current_residual"],
    "G4": ["AI0_RMS_slope", "AI1_RMS_slope", "Current_RMS_slope", "dCurrent_RMS_x_dAI0_RMS", "dCurrent_RMS_x_dAI1_RMS"],
    "G5": ["AI0_kurtosis", "AI1_kurtosis", "AI0_crest_factor", "AI1_crest_factor"],
}
TEMPORAL_GROUPS = {
    "T0": ["AI0_Vibration", "AI1_Vibration", "AI2_Current"],
    "T1": ["AI0_short_RMS", "AI1_short_RMS", "Current_short_RMS"],
    "T2": ["AI0_AI1_RMS_diff", "AI0_current_residual", "AI1_current_residual"],
    "T3": ["AI0_RMS_slope", "AI1_RMS_slope", "Current_RMS_slope"],
}
TABULAR_FEATURES = sum(TABULAR_GROUPS.values(), [])
TEMPORAL_FEATURES = sum(TEMPORAL_GROUPS.values(), [])


def trailing_rms(raw: np.ndarray, width: int = SHORT_RMS_SAMPLES) -> np.ndarray:
    """각 시점에서 최근 width시점(그 시점 포함)의 RMS."""
    output = np.empty_like(raw, dtype=np.float64)
    for point in range(raw.shape[1]):
        start = max(0, point - width + 1)
        output[:, point, :] = np.sqrt(np.mean(raw[:, start:point + 1, :] ** 2, axis=1))
    return output


def rolling_slope(values: np.ndarray, width: int = SHORT_RMS_SAMPLES) -> np.ndarray:
    """각 시점에서 최근 width시점 값의 직선 기울기 (초당 변화)."""
    output = np.zeros_like(values, dtype=np.float64)
    for point in range(1, values.shape[1]):
        start = max(0, point - width + 1)
        t = np.arange(point - start + 1, dtype=np.float64) * 0.1
        centered_t = t - t.mean()
        part = values[:, start:point + 1, :]
        centered_x = part - part.mean(axis=1, keepdims=True)
        output[:, point, :] = np.sum(centered_x * centered_t[None, :, None], axis=1) / max(np.sum(centered_t ** 2), 1e-12)
    return output


def window_slope(values: np.ndarray) -> np.ndarray:
    t = np.arange(values.shape[1], dtype=np.float64) * 0.1
    centered_t = t - t.mean()
    centered_x = values - values.mean(axis=1, keepdims=True)
    return np.sum(centered_x * centered_t[None, :, None], axis=1) / max(np.sum(centered_t ** 2), 1e-12)


def safe_corr(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_c = left - left.mean(axis=1, keepdims=True)
    right_c = right - right.mean(axis=1, keepdims=True)
    numerator = np.sum(left_c * right_c, axis=1)
    denominator = np.sqrt(np.sum(left_c ** 2, axis=1) * np.sum(right_c ** 2, axis=1))
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)


def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """y = 절편 + 기울기 × x 를 최소제곱으로 맞춘다."""
    design = np.column_stack([np.ones(len(x)), x])
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(coefficients[0]), float(coefficients[1])


def _clean(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(values, nan=0.0, posinf=1e9, neginf=-1e9).astype(np.float32)


class TabularFeatures:
    """window 하나를 19개 숫자로 요약한다 (크기·관계·부하-진동·추세·충격)."""

    names = TABULAR_FEATURES

    def __init__(self, short_rms_samples: int = SHORT_RMS_SAMPLES):
        self.short_rms_samples = short_rms_samples

    def _base(self, raw: np.ndarray) -> dict:
        rms = np.sqrt(np.mean(raw.astype(np.float64) ** 2, axis=1))
        half = max(2, raw.shape[1] // 2)
        early = np.sqrt(np.mean(raw[:, :half, :].astype(np.float64) ** 2, axis=1))
        late = np.sqrt(np.mean(raw[:, -half:, :].astype(np.float64) ** 2, axis=1))
        mean = raw.mean(axis=1, dtype=np.float64)
        std = raw.std(axis=1, dtype=np.float64)
        centered = raw - mean[:, None, :]
        return {
            "rms": rms,
            "slope": window_slope(trailing_rms(raw, width=self.short_rms_samples)),
            "delta": late - early,
            "kurtosis": np.mean(centered ** 4, axis=1) / np.maximum(std ** 4, 1e-12) - 3.0,
            "crest": np.max(np.abs(raw), axis=1) / np.maximum(rms, 1e-12),
            "corr": safe_corr(raw[:, :, 0], raw[:, :, 1]),
        }

    def fit(self, raw_train: np.ndarray, y_train: np.ndarray) -> "TabularFeatures":
        rms = self._base(raw_train)["rms"][y_train == 0]
        self.line0 = fit_line(rms[:, 2], rms[:, 0])
        self.line1 = fit_line(rms[:, 2], rms[:, 1])
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        values = self._base(raw)
        rms, slope, delta = values["rms"], values["slope"], values["delta"]
        columns = {
            "AI0_RMS": rms[:, 0],
            "AI1_RMS": rms[:, 1],
            "AI0_AI1_RMS_diff": rms[:, 0] - rms[:, 1],
            "AI0_AI1_RMS_ratio": rms[:, 0] / np.maximum(rms[:, 1], 1e-12),
            "AI0_AI1_corr": values["corr"],
            "Current_RMS": rms[:, 2],
            "Current_RMS_to_AI0_RMS_ratio": rms[:, 2] / np.maximum(rms[:, 0], 1e-12),
            "Current_RMS_to_AI1_RMS_ratio": rms[:, 2] / np.maximum(rms[:, 1], 1e-12),
            "AI0_current_residual": rms[:, 0] - (self.line0[0] + self.line0[1] * rms[:, 2]),
            "AI1_current_residual": rms[:, 1] - (self.line1[0] + self.line1[1] * rms[:, 2]),
            "AI0_RMS_slope": slope[:, 0],
            "AI1_RMS_slope": slope[:, 1],
            "Current_RMS_slope": slope[:, 2],
            "dCurrent_RMS_x_dAI0_RMS": delta[:, 2] * delta[:, 0],
            "dCurrent_RMS_x_dAI1_RMS": delta[:, 2] * delta[:, 1],
            "AI0_kurtosis": values["kurtosis"][:, 0],
            "AI1_kurtosis": values["kurtosis"][:, 1],
            "AI0_crest_factor": values["crest"][:, 0],
            "AI1_crest_factor": values["crest"][:, 1],
        }
        return _clean(np.column_stack([columns[name] for name in TABULAR_FEATURES]))


class TemporalFeatures:
    """window의 시점 순서를 유지한 채 12채널(원값 3 + 국소 RMS 3 + 관계 3 + 추세 3)로 만든다."""

    names = TEMPORAL_FEATURES

    def fit(self, raw_train: np.ndarray, y_train: np.ndarray) -> "TemporalFeatures":
        local = trailing_rms(raw_train)[y_train == 0]
        current = local[:, :, 2].reshape(-1)
        self.line0 = fit_line(current, local[:, :, 0].reshape(-1))
        self.line1 = fit_line(current, local[:, :, 1].reshape(-1))
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        rms = trailing_rms(raw)
        slopes = rolling_slope(rms)
        residual0 = rms[:, :, 0] - (self.line0[0] + self.line0[1] * rms[:, :, 2])
        residual1 = rms[:, :, 1] - (self.line1[0] + self.line1[1] * rms[:, :, 2])
        return _clean(np.stack([
            raw[:, :, 0], raw[:, :, 1], raw[:, :, 2],
            rms[:, :, 0], rms[:, :, 1], rms[:, :, 2],
            rms[:, :, 0] - rms[:, :, 1], residual0, residual1,
            slopes[:, :, 0], slopes[:, :, 1], slopes[:, :, 2],
        ], axis=2))


def tabular_features(raw_split: dict, y_split: dict) -> dict:
    maker = TabularFeatures().fit(raw_split["train"], y_split["train"])
    return {part: maker.transform(raw) for part, raw in raw_split.items()}


def temporal_features(raw_split: dict, y_split: dict) -> dict:
    maker = TemporalFeatures().fit(raw_split["train"], y_split["train"])
    return {part: maker.transform(raw) for part, raw in raw_split.items()}
