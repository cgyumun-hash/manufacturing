"""마르코프 연속경고 (조기 경보).

  최종 판정 = segment 평균 ≥ T*  OR  window 점수 ≥ T_w 가 K번 연속

- T_w : validation 정상 window 점수의 99% 분위수
- r   : 정상에서 경고가 새로 시작될 확률,  q : 경고가 다음 판정에도 이어질 확률
        (validation 정상 segment 안의 전이만 세고, Jeffreys 보정 0.5)
- K   : N·π0·r·q^(K−1) ≤ 0.01 (시간당 기대 헛경보 연속)을 만족하는 가장 작은 K, N = 시간당 판정 수
"""
import numpy as np
import pandas as pd
from scipy.stats import beta

from .config import DT, JEFFREYS, NORMAL_QUANTILE, TARGET_FALSE_RUNS_PER_HOUR, UPPER_PROBABILITY, WINDOW


def warning_threshold(normal_scores) -> float:
    return float(np.quantile(normal_scores, NORMAL_QUANTILE, method="higher"))


def transitions(windows: pd.DataFrame, t_w: float) -> dict:
    n = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    zeros = ones = 0
    for _, frame in windows[windows.label == 0].groupby("group", sort=False):
        states = (frame.sort_values("start").score.to_numpy() >= t_w).astype(int)
        zeros += int((states == 0).sum())
        ones += int((states == 1).sum())
        for a, b in zip(states[:-1], states[1:]):
            n[(int(a), int(b))] += 1
    n00, n01, n10, n11 = n[(0, 0)], n[(0, 1)], n[(1, 0)], n[(1, 1)]
    return {"n00": n00, "n01": n01, "n10": n10, "n11": n11,
            "pi0": (zeros + JEFFREYS) / (zeros + ones + 2 * JEFFREYS),
            "r": (n01 + JEFFREYS) / (n00 + n01 + 2 * JEFFREYS),
            "q": (n11 + JEFFREYS) / (n10 + n11 + 2 * JEFFREYS),
            "pi0_upper95": float(beta.ppf(UPPER_PROBABILITY, zeros + JEFFREYS, ones + JEFFREYS)),
            "r_upper95": float(beta.ppf(UPPER_PROBABILITY, n01 + JEFFREYS, n00 + JEFFREYS)),
            "q_upper95": float(beta.ppf(UPPER_PROBABILITY, n11 + JEFFREYS, n10 + JEFFREYS))}


def markov_k(decisions_per_hour, pi0, r, q, max_k=1000) -> int:
    for k in range(1, max_k + 1):
        if decisions_per_hour * pi0 * r * q ** (k - 1) <= TARGET_FALSE_RUNS_PER_HOUR:
            return k
    return max_k


def run_alarm(windows: pd.DataFrame, t_w: float, k: int):
    """segment 안에서 경고가 K번 연속되면 경보. segment가 바뀌면 연속 횟수는 0부터."""
    seg_rows, win_frames = [], []
    for group, frame in windows.groupby("group", sort=False):
        frame = frame.sort_values("start").copy()
        warning = frame.score.to_numpy() >= t_w
        run, first, runs = 0, None, []
        for i, flag in enumerate(warning):
            run = run + 1 if flag else 0
            runs.append(run)
            if run >= k and first is None:
                first = i
        frame["warning"] = warning.astype(int)
        frame["run_count"] = runs
        win_frames.append(frame)
        seg_rows.append({"group": group, "run_pred": int(first is not None),
                         # 경보 시각 = 경보가 난 window의 마지막 시점 (segment 시작 후 초)
                         "first_alarm_seconds": None if first is None else round((int(frame.start.iloc[first]) + WINDOW - 1) * DT, 1)})
    return pd.DataFrame(seg_rows), pd.concat(win_frames, ignore_index=True)
