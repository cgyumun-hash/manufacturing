"""2.3 기초 통계 — 센서 값의 분포 모양과 분위수 범위 (전체 행).

출력: 그림_값_분포.png, 그림_분위수_범위.png, 표_기술통계.csv
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # work 폴더

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from common.data import SENSOR_KO, SENSORS, load_frames  # noqa: E402
from common.paths import save_table  # noqa: E402
from common.plot_style import C, INK, INK2, LABEL, plt, save  # noqa: E402

fr = load_frames()
unit = lambda s: " (전류 단위)" if "Current" in s else " (진동 단위)"  # noqa: E731

# ① 분포 (면적 1로 맞춘 계단 히스토그램, 양 끝 0.5%는 그림에서만 자름)
fig, axes = plt.subplots(2, 2, figsize=(10, 6.4))
axes = axes.ravel()
for n, (ax, s) in enumerate(zip(axes, SENSORS)):
    lo = min(np.quantile(fr[k][s], 0.005) for k in fr)
    hi = max(np.quantile(fr[k][s], 0.995) for k in fr)
    bins = np.linspace(lo, hi, 45)
    for k in ["normal", "anomaly"]:
        v = fr[k][s]
        ax.hist(v[(v >= lo) & (v <= hi)], bins=bins, density=True, histtype="step", lw=1.8, color=C[k],
                label=f"{LABEL[k]} ({len(fr[k]):,}행)")
    ax.set_title(f"{'①②③'[n]} {SENSOR_KO[s]}")
    ax.set_xlabel("센서 값" + unit(s))
    ax.set_ylabel("밀도")
h, l = axes[0].get_legend_handles_labels()
axes[3].axis("off")
axes[3].legend(h, l, loc="center", title="선 색 = 데이터 (전체 행)")
fig.tight_layout(w_pad=2, h_pad=1.5)
save(fig, HERE / "그림_값_분포.png")

# ② 분위수 범위
fig, axes = plt.subplots(2, 2, figsize=(10, 5.2))
axes = axes.ravel()
rows = []
for n, (ax, s) in enumerate(zip(axes, SENSORS)):
    for i, k in enumerate(["normal", "anomaly"]):
        v = fr[k][s]
        q5, q25, q50, q75, q95 = np.quantile(v, [.05, .25, .5, .75, .95])
        y = 1 - i
        ax.plot([q5, q95], [y, y], color=C[k], lw=1.4)
        ax.plot([q25, q75], [y, y], color=C[k], lw=9, solid_capstyle="butt", alpha=0.9)
        ax.plot([q50], [y], marker="o", ms=7, mfc="white", mec=INK, mew=1.2)
        rows.append({"구분": LABEL[k], "센서": SENSOR_KO[s], "행 수": len(v), "평균": v.mean(), "표준편차": v.std(),
                     "중앙값": q50, "IQR": q75 - q25, "5%": q5, "25%": q25, "75%": q75, "95%": q95,
                     "최솟값": v.min(), "최댓값": v.max()})
    ax.set_yticks([1, 0])
    ax.set_yticklabels(["정상", "이상"])
    ax.set_ylim(-0.6, 1.6)
    ax.set_title(f"{'①②③'[n]} {SENSOR_KO[s]}")
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("센서 값" + unit(s))
axes[3].axis("off")
axes[3].legend(handles=[Line2D([], [], color=INK2, lw=1.4, label="가는 선 = 5% ~ 95% 범위"),
                        Line2D([], [], color=INK2, lw=9, label="굵은 막대 = 25% ~ 75% (가운데 절반)"),
                        Line2D([], [], color="none", marker="o", ms=7, mfc="white", mec=INK, label="흰 동그라미 = 중앙값")],
               loc="center", title="읽는 법 (파랑 = 정상, 빨강 = 이상)")
fig.tight_layout(w_pad=2, h_pad=1.5)
save(fig, HERE / "그림_분위수_범위.png")

table = pd.DataFrame(rows).round(4)
save_table(table, HERE / "표_기술통계.csv")
print(table.to_string(index=False))
