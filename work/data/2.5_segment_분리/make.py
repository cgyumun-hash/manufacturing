"""2.5 불연속 구간 처리 — 이웃 기록 간격이 0.1초가 아니면 그 자리에서 segment를 나눈다 (보간하지 않음).

① 정상 처음 40초 예시: 색 막대 = segment, 화살표 = 공백
② segment 길이 분포 (전체 segment, 정상 599 · 이상 21)

출력: 그림_segment_분리.png, 표_segment_길이.csv, 표_segment_목록.csv
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # work 폴더

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common.data import load_frames  # noqa: E402
from common.paths import save_table  # noqa: E402
from common.plot_style import AQUA, C, INK, INK2, LABEL, plt, save  # noqa: E402

fr = load_frames()
fig = plt.figure(figsize=(10, 4.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.5], hspace=0.65, wspace=0.25)
ax = fig.add_subplot(gs[0, :])
f = fr["normal"]
t0 = f.TimeStamp.iloc[0]
sub = f[(f.TimeStamp - t0).dt.total_seconds() <= 40]
palette = [C["normal"], AQUA]
for i, (sid, g) in enumerate(sub.groupby("segment_id")):
    tt = (g.TimeStamp - t0).dt.total_seconds()
    ax.plot([tt.min(), tt.max()], [0, 0], color=palette[i % 2], lw=7, solid_capstyle="butt")
    ax.text((tt.min() + tt.max()) / 2, 0.55, f"segment {sid}\n{len(g)}시점", ha="center", va="bottom", fontsize=8, color=INK)
ends = sub.groupby("segment_id").TimeStamp.agg(["min", "max"])
for a, b in zip(ends["max"].iloc[:-1], ends["min"].iloc[1:]):
    x0, x1 = (a - t0).total_seconds(), (b - t0).total_seconds()
    ax.annotate("", xy=(x1, -0.45), xytext=(x0, -0.45), arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.8))
    ax.text((x0 + x1) / 2, -0.75, f"공백 {x1 - x0:.1f}초", ha="center", va="top", fontsize=8, color=INK2)
ax.set_ylim(-1.6, 1.8)
ax.set_yticks([])
ax.spines["left"].set_visible(False)
ax.set_xlim(-0.5, (sub.TimeStamp - t0).dt.total_seconds().max() + 0.5)
ax.set_xlabel("정상 데이터 처음 40초 (초)")
ax.set_title("① 시간이 끊긴 곳에서 자르면 연속 구간(segment)이 된다")
stats = []
for j, k in enumerate(["normal", "anomaly"]):
    a = fig.add_subplot(gs[1, j])
    lens = fr[k].groupby("segment_id").size()
    a.hist(lens, bins=np.arange(0, 55, 5), color=C[k], edgecolor="white", linewidth=1.5)
    a.axvline(lens.median(), color=INK, lw=1, ls=(0, (3, 2)))
    a.text(lens.median() + 0.8, a.get_ylim()[1] * 0.92, f"중앙값 {lens.median():.0f}시점", fontsize=8, color=INK)
    a.set_xlabel("segment 길이 (시점, 1시점 = 0.1초)")
    a.set_ylabel("segment 개수")
    a.set_title(f"② {LABEL[k]} segment 길이 분포 (총 {len(lens)}개)")
    a.set_xticks(range(0, 55, 10))
    stats.append({"구분": LABEL[k], "segment 수": len(lens), "평균": round(lens.mean(), 1), "표준편차": round(lens.std(), 1),
                  "최소": int(lens.min()), "25%": lens.quantile(.25), "중앙값": lens.median(), "75%": lens.quantile(.75),
                  "최대": int(lens.max()), "50시점(최대 길이)인 segment": int((lens == 50).sum())})
save(fig, HERE / "그림_segment_분리.png")

listing = []
for k, f in fr.items():
    for sid, g in f.groupby("segment_id"):
        listing.append({"class": k, "group": f"{k}_{sid}", "segment_id": sid, "시작 시각": g.TimeStamp.iloc[0],
                        "끝 시각": g.TimeStamp.iloc[-1], "시점 수": len(g), "길이(초)": round((len(g) - 1) * 0.1, 1),
                        "앞 segment와의 공백(초)": None if sid == 1 else round(float(g.interval.iloc[0]), 3)})
save_table(pd.DataFrame(stats), HERE / "표_segment_길이.csv")
save_table(pd.DataFrame(listing), HERE / "표_segment_목록.csv")
print(pd.DataFrame(stats).to_string(index=False))
