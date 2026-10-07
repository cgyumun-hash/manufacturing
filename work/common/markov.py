"""5장: segment 평균 판정 OR 보완된 마르코프 연속경고.

    ŷ = I{ segment 평균 점수 ≥ T*  ∨  Run_K(window 점수 ≥ T_w) }

- T*  : 3장 validation에서 고른 segment 평균 임곗값 (기존 판정, 안전망)
- T_w : validation **정상 window** 점수의 99% 분위수 (경고 기준)
- r   : 정상에서 '정상 → 경고'로 바뀔 확률 (경고 시작)
- q   : 정상에서 '경고 → 경고'로 이어질 확률 (경고 지속)
- K_M : N·π0·r·q^(K−1) ≤ 0.01 (시간당 기대 헛경보 연속)을 만족하는 가장 작은 K
- 95% 상한 K : r, q, π0의 Beta 사후분포 95% 상한으로 계산한 보수적 K
r, q는 정상 validation segment **안의** 전이만 센다 (segment 경계는 넘지 않는다).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_distribution

NORMAL_QUANTILE = 0.99
TARGET_FALSE_RUNS_PER_HOUR = 0.01
JEFFREYS = 0.5
UPPER_PROBABILITY = 0.95
MAX_K = 1000


def warning_threshold(normal_valid_scores: np.ndarray) -> float:
    return float(np.quantile(normal_valid_scores, NORMAL_QUANTILE, method="higher"))


def transition_statistics(windows: pd.DataFrame, threshold: float) -> dict:
    counts = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    zeros = ones = segments = 0
    for _, frame in windows[windows.label == 0].groupby("group", sort=False):
        segments += 1
        states = (frame.sort_values("start").score.to_numpy() >= threshold).astype(np.int64)
        zeros += int((states == 0).sum())
        ones += int((states == 1).sum())
        for current, following in zip(states[:-1], states[1:]):
            counts[(int(current), int(following))] += 1
    n00, n01, n10, n11 = counts[(0, 0)], counts[(0, 1)], counts[(1, 0)], counts[(1, 1)]
    return {
        "normal_validation_segments": segments,
        "normal_validation_states": zeros + ones,
        "n00": n00, "n01": n01, "n10": n10, "n11": n11,
        "pi0": (zeros + JEFFREYS) / (zeros + ones + 2 * JEFFREYS),
        "start_probability_r": (n01 + JEFFREYS) / (n00 + n01 + 2 * JEFFREYS),
        "continuation_probability_q": (n11 + JEFFREYS) / (n10 + n11 + 2 * JEFFREYS),
        "pi0_upper_95": float(beta_distribution.ppf(UPPER_PROBABILITY, zeros + JEFFREYS, ones + JEFFREYS)),
        "r_upper_95": float(beta_distribution.ppf(UPPER_PROBABILITY, n01 + JEFFREYS, n00 + JEFFREYS)),
        "q_upper_95": float(beta_distribution.ppf(UPPER_PROBABILITY, n11 + JEFFREYS, n10 + JEFFREYS)),
    }


def markov_k(decisions_per_hour: float, pi0: float, r: float, q: float) -> tuple[int, float]:
    for k in range(1, MAX_K + 1):
        expected = decisions_per_hour * pi0 * r * q ** (k - 1)
        if expected <= TARGET_FALSE_RUNS_PER_HOUR:
            return k, float(expected)
    return MAX_K, float(decisions_per_hour * pi0 * r * q ** (MAX_K - 1))


def independent_k(probability: float, decisions_per_hour: float) -> int:
    """경고가 서로 독립이라고 볼 때의 K (비교용)."""
    if probability <= 0:
        return 1
    if probability >= 1:
        return 100
    value = math.log(TARGET_FALSE_RUNS_PER_HOUR / decisions_per_hour) / math.log(probability)
    return max(1, min(100, int(math.ceil(value))))


def run_alarm(windows: pd.DataFrame, warning: float, k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """segment 안에서 경고가 K번 연속되면 그 순간 경보. segment가 바뀌면 연속 횟수는 0부터."""
    segment_rows, window_frames = [], []
    for group, frame in windows.groupby("group", sort=False):
        frame = frame.sort_values("start").copy()
        is_warning = frame.score.to_numpy() >= warning
        run, first, trigger, runs = 0, None, np.zeros(len(frame), dtype=np.int64), []
        for index, flag in enumerate(is_warning):
            run = run + 1 if flag else 0
            runs.append(run)
            if run >= k:
                first = index if first is None else first
                trigger[index] = 1
        frame["warning"] = is_warning.astype(np.int64)
        frame["run_count"] = runs
        frame["run_alarm_trigger"] = trigger
        window_frames.append(frame)
        segment_rows.append({
            "group": group, "label": int(frame.label.iloc[0]),
            "run_pred": int(first is not None),
            "first_alarm_window": None if first is None else int(first) + 1,
            "window_count": len(frame),
            "max_run": int(max(runs)) if runs else 0,
        })
    return pd.DataFrame(segment_rows), pd.concat(window_frames, ignore_index=True)
