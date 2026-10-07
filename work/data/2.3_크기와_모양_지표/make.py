"""2.3 기초 통계 — 신호의 크기(RMS)와 모양(왜도 · 첨도 · crest factor). 클래스별 전체 행을 한 덩어리로 계산.

출력: 그림_RMS_왜도_첨도_crest.png, 표_RMS_왜도_첨도_crest.csv
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # work 폴더

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kurtosis, skew  # noqa: E402

from common.data import SENSOR_KO, SENSORS, load_frames  # noqa: E402
from common.paths import save_table  # noqa: E402
from common.plot_style import C, INK, INK2, LABEL, plt, save  # noqa: E402

fr = load_frames()
rows = []
for k in ["normal", "anomaly"]:
    for s in SENSORS:
        v = fr[k][s].to_numpy(float)
        rms = float(np.sqrt(np.mean(v ** 2)))
        rows.append({"cls": k, "sensor": s, "rms": rms, "skew": skew(v, bias=False), "kurt": kurtosis(v, bias=False), "crest": np.abs(v).max() / rms})
desc = pd.DataFrame(rows)
n = desc[desc.cls == "normal"].set_index("sensor")
a = desc[desc.cls == "anomaly"].set_index("sensor")
ratio = (a.rms / n.rms).reindex(SENSORS)

short = {"AI0_Vibration": "AI0", "AI1_Vibration": "AI1", "AI2_Current": "전류"}
x = np.arange(3)
fig, axes = plt.subplots(2, 2, figsize=(10, 6.4))
axes = axes.ravel()
ax = axes[0]
ax.bar(x, ratio, color=C["anomaly"], width=0.55)
ax.axhline(1, color=INK2, lw=0.8, ls=(0, (3, 3)))
for i, v in enumerate(ratio):
    ax.text(i, v + 0.15, f"{v:.1f}배", ha="center", fontsize=9, color=INK)
ax.set_title("① RMS: 이상 ÷ 정상")
ax.set_ylim(0, ratio.max() * 1.2)
ax.set_ylabel("배율 (1 = 정상과 같음)")
for ax, col, title in zip(axes[1:], ["skew", "kurt", "crest"], ["② 왜도", "③ 첨도(excess)", "④ Crest factor"]):
    w = 0.36
    for j, k in enumerate(["normal", "anomaly"]):
        d = desc[desc.cls == k].set_index("sensor").reindex(SENSORS)
        ax.bar(x + (j - 0.5) * (w + 0.02), d[col], width=w, color=C[k], label=LABEL[k])
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_title(title)
for ax in axes:
    ax.set_xticks(x)
    ax.set_xticklabels([short[s] for s in SENSORS])
axes[1].legend(loc="lower left")
fig.tight_layout(w_pad=2, h_pad=1.5)
save(fig, HERE / "그림_RMS_왜도_첨도_crest.png")

table = pd.DataFrame([{"구분": LABEL[r.cls], "센서": SENSOR_KO[r.sensor], "RMS": r.rms,
                       "RMS 배율(이상÷정상)": ratio[r.sensor] if r.cls == "anomaly" else 1.0,
                       "왜도": r.skew, "첨도(excess)": r.kurt, "Crest factor": r.crest} for r in desc.itertuples()]).round(4)
save_table(table, HERE / "표_RMS_왜도_첨도_crest.csv")
print(table.to_string(index=False))
