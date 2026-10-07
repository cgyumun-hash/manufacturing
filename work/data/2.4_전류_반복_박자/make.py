"""2.4 핵심 시각화 — 전류에 반복되는 프레스 박자(약 1.7초)가 있는가.

① ② 가장 긴 segment의 전류와 봉우리(peak) 간격 (샘플 1개씩)
③ 자기상관: 몇 초 뒤에 같은 모양이 돌아오나 (20시점 이상 segment 전체)
④ 이웃 봉우리 간격 분포 (20시점 이상 segment 전체)
봉우리 기준: 그 구간 표준편차의 0.5배 이상 솟아오르고, 이웃 봉우리와 0.3초 이상 떨어진 점.

출력: 그림_전류_반복_박자.png, 표_전류_반복_박자.csv, 표_자기상관_중앙값.csv
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # work 폴더

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402

from common.data import DT, load_frames  # noqa: E402
from common.paths import save_table  # noqa: E402
from common.plot_style import C, INK, INK2, LABEL, plt, save  # noqa: E402

fr = load_frames()


def longest(frame):
    sid = frame.groupby("segment_id").size().idxmax()
    return frame[frame.segment_id == sid].reset_index(drop=True)


def seg_acf(v, max_lag=15):
    out = np.full(max_lag, np.nan)
    for lag in range(1, min(max_lag, len(v) - 2) + 1):
        a, b = v[:-lag], v[lag:]
        if a.std() > 0 and b.std() > 0:
            out[lag - 1] = np.corrcoef(a, b)[0, 1]
    return out


def peaks(v):
    p, _ = find_peaks(v, prominence=max(0.5 * v.std(), 1e-12), distance=3)
    return p


acfs, intervals = {}, {}
for k in ["normal", "anomaly"]:
    A, I = [], []
    for _, g in fr[k].groupby("segment_id"):
        v = g.AI2_Current.to_numpy(float)
        if len(v) < 20:
            continue
        A.append(seg_acf(v))
        I.extend(np.diff(peaks(v)) * DT)
    acfs[k], intervals[k] = np.array(A), np.round(np.array(I), 1)

fig = plt.figure(figsize=(10, 6.4))
gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.25)
for j, k in enumerate(["normal", "anomaly"]):
    ax = fig.add_subplot(gs[0, j])
    v = longest(fr[k]).AI2_Current.to_numpy(float)
    t = np.arange(len(v)) * DT
    p = peaks(v)
    ax.plot(t, v, color=C[k], lw=1.5)
    ax.scatter(t[p], v[p], s=30, color=INK, zorder=3, label="찾은 peak")
    for a, b in zip(p[:-1], p[1:]):
        ax.annotate("", xy=(t[b], v.max() * 1.18), xytext=(t[a], v.max() * 1.18),
                    arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.7))
        ax.text((t[a] + t[b]) / 2, v.max() * 1.24, f"{t[b] - t[a]:.1f}", ha="center", fontsize=7.5, color=INK2)
    ax.set_ylim(min(v.min(), -abs(v.max())) * 1.1, v.max() * 1.42)
    ax.set_title(f"{'①②'[j]} {LABEL[k]} 가장 긴 segment 전류와 peak 간격(초)")
    ax.set_xlabel("시간 (초)")
    ax.set_ylabel("전류 값")
    ax.legend(loc="lower right")
ax = fig.add_subplot(gs[1, 0])
lags = np.arange(1, 16) * DT
for k in ["normal", "anomaly"]:
    q1, q3 = np.nanquantile(acfs[k], [.25, .75], axis=0)
    ax.fill_between(lags, q1, q3, color=C[k], alpha=0.15, linewidth=0)
    ax.plot(lags, np.nanmedian(acfs[k], axis=0), color=C[k], lw=2, marker="o", ms=4,
            label=f"{LABEL[k]} 중앙값 ({len(acfs[k])}개 segment)")
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xlabel("몇 초 뒤의 자기 자신과 비교했나 (초)")
ax.set_ylabel("닮은 정도 (자기상관)")
ax.set_title("③ 자기상관: 몇 초 뒤에 같은 모양이 돌아오나")
ax.set_ylim(-1, 1.05)
ax.legend(loc="upper center", fontsize=8)
ax = fig.add_subplot(gs[1, 1])
for k in ["normal", "anomaly"]:
    w = np.ones(len(intervals[k])) / len(intervals[k]) * 100
    ax.hist(intervals[k], bins=np.arange(0.25, 2.6, 0.1), weights=w, color=C[k], alpha=0.75, edgecolor="white",
            linewidth=1, label=f"{LABEL[k]} ({len(intervals[k])}개 간격)")
ax.set_xlabel("이웃한 peak 사이 시간 (초)")
ax.set_ylabel("비율 (%)")
ax.set_title("④ peak 간격 분포")
ax.legend(loc="upper left")
save(fig, HERE / "그림_전류_반복_박자.png")

med = {k: np.nanmedian(acfs[k], axis=0) for k in acfs}
table = pd.DataFrame([{"구분": LABEL[k], "2초 이상 segment": len(acfs[k]), "봉우리 간격 수": len(intervals[k]),
                       "봉우리 간격 중앙값(초)": float(np.median(intervals[k])),
                       "25%(초)": float(np.quantile(intervals[k], .25)), "75%(초)": float(np.quantile(intervals[k], .75)),
                       "0.8초 뒤 자기상관": round(float(med[k][7]), 2), "1.5초 뒤 자기상관": round(float(med[k][14]), 2)}
                      for k in ["normal", "anomaly"]])
save_table(table, HERE / "표_전류_반복_박자.csv")
save_table(pd.DataFrame({"lag(초)": np.round(lags, 1), "정상 중앙값": med["normal"].round(3), "이상 중앙값": med["anomaly"].round(3)}),
           HERE / "표_자기상관_중앙값.csv")
print(table.to_string(index=False))
