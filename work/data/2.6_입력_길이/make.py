"""2.6 입력 길이 선정 — 입력을 길게 잡을수록 그 길이만큼 이어진 segment만 쓸 수 있어 사례가 줄어든다.

출력: 그림_입력_길이.png, 표_입력_길이별_segment.csv, 표_시점수와_입력길이.csv
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
secs = [0.3] + list(np.arange(0.5, 5.01, 0.5))
pts = [round(x * 10) for x in secs]
lens = {k: f.groupby("segment_id").size() for k, f in fr.items()}
cnt = {k: [int((lens[k] >= p).sum()) for p in pts] for k in lens}

fig, axes = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True)
x = np.arange(len(secs))
for n, (ax, k) in enumerate(zip(axes, ["normal", "anomaly"])):
    ax.axvspan(0.5, 4.5, color=AQUA, alpha=0.08, lw=0)
    ax.axvspan(4.5, len(secs) - 0.5, color="#eda100", alpha=0.08, lw=0)
    ax.bar(x, cnt[k], color=C[k], width=0.62)
    for i, v in enumerate(cnt[k]):
        lab = f"{v}" if k == "normal" else f"{v}개\n({v / 21 * 100:.0f}%)"
        ax.text(i, v + max(cnt[k]) * 0.03, lab, ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_ylim(0, max(cnt[k]) * 1.35)
    ax.set_ylabel("쓸 수 있는 segment 수")
    ax.set_title(f"{'①②'[n]} {LABEL[k]}: 이 길이 이상으로 이어진 segment 수")
axes[0].text(2.5, max(cnt["normal"]) * 1.22, "주요 비교 범위 0.5~2.0초", ha="center", fontsize=8.5, color="#0d7a55")
axes[0].text(7.5, max(cnt["normal"]) * 1.22, "표본 감소 확인용 2.5~5.0초", ha="center", fontsize=8.5, color="#8a5d00")
axes[1].axhline(21 / 2, color=INK2, lw=0.8, ls=(0, (3, 3)))
axes[1].text(len(secs) - 0.45, 21 / 2 + 0.3, "이상 전체의 절반", ha="right", va="bottom", fontsize=8, color=INK2)
axes[1].set_xticks(x)
axes[1].set_xticklabels([f"{s:.1f}초\n({p}시점)" for s, p in zip(secs, pts)])
axes[1].set_xlabel("모델 입력 길이")
fig.tight_layout(h_pad=1)
save(fig, HERE / "그림_입력_길이.png")

table = pd.DataFrame({"입력": [f"{s:.1f}초" for s in secs], "시점": pts, "정상 segment": cnt["normal"],
                      "이상 segment": cnt["anomaly"], "이상 잔존율(%)": [round(v / 21 * 100, 1) for v in cnt["anomaly"]]})
save_table(table, HERE / "표_입력_길이별_segment.csv")
save_table(pd.DataFrame({"입력 표현": ["3시점", "0.5초 window", "1.0초 window", "1.5초 window", "2.0초 window"],
                         "시점 수": [3, 5, 10, 15, 20], "입력 길이(관례)": ["0.3초", "0.5초", "1.0초", "1.5초", "2.0초"],
                         "첫 시점 ~ 마지막 시점 실제 간격": ["0.2초", "0.4초", "0.9초", "1.4초", "1.9초"]}),
           HERE / "표_시점수와_입력길이.csv")
print(table.to_string(index=False))
