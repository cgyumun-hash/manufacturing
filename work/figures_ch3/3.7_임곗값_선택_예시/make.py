"""3.7 임곗값 선택 방법 — validation segment 점수에서 F1이 가장 높은 임곗값을 고르는 과정 (가상 예시).

정상 6개, 이상 3개 segment의 가상 점수로 설명한다 (실제 모델 결과 아님).
실제 선택 코드는 common/evaluation.py의 best_threshold, 각 모델의 valid.py.

출력: 그림_임곗값_선택_예시.png, 표_후보_임곗값별_판정.csv
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # work 폴더

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common.paths import save_table  # noqa: E402
from common.plot_style import C, plt, save  # noqa: E402

NORMAL = np.array([0.05, 0.12, 0.20, 0.31, 0.45, 0.58])
ANOMALY = np.array([0.40, 0.66, 0.90])


def f1_at(t):
    tp, fp = int((ANOMALY >= t).sum()), int((NORMAL >= t).sum())
    fn = len(ANOMALY) - tp
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn)
    return (2 * p * r / (p + r) if p + r else 0.0), tp, fp, fn, p, r


ts = np.linspace(0, 1, 1001)
f1 = np.array([f1_at(t)[0] for t in ts])
fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True, gridspec_kw={"height_ratios": [1, 1.4]})
ax = axes[0]
ax.scatter(NORMAL, np.zeros_like(NORMAL), s=70, color=C["normal"], label="정상 segment (6개)", zorder=3)
ax.scatter(ANOMALY, np.zeros_like(ANOMALY), s=90, color=C["anomaly"], edgecolors="black", linewidths=0.6,
           label="이상 segment (3개)", zorder=3)
for v in np.r_[NORMAL, ANOMALY]:
    ax.text(v, 0.35, f"{v:.2f}", ha="center", fontsize=8.5)
ax.axvline(0.62, color="black", linestyle="--", linewidth=1.2)
ax.text(0.63, -0.45, "선택된 임곗값 T* = 0.62", fontsize=9)
ax.set_ylim(-0.7, 0.7)
ax.set_yticks([])
ax.set_title("① validation segment의 이상 점수 (가상 예시)")
ax.legend(loc="lower left", ncol=2)
ax = axes[1]
ax.plot(ts, f1, color="black", linewidth=1.5)
for t in (0.35, 0.50, 0.62, 0.70):
    v = f1_at(t)[0]
    ax.scatter([t], [v], s=40, color=C["anomaly"] if t == 0.62 else "#777777", zorder=3)
    ax.text(t, v + 0.05, f"T={t:.2f}\nF1 {v:.2f}", ha="center", fontsize=8)
ax.set_ylim(0, 1.05)
ax.set_xlim(0, 1)
ax.set_xlabel("임곗값 T (점수가 T 이상이면 '이상'으로 판정)")
ax.set_ylabel("validation F1")
ax.set_title("② 임곗값을 바꿔가며 계산한 validation F1 → 가장 높은 곳을 선택")
fig.tight_layout()
save(fig, HERE / "그림_임곗값_선택_예시.png")

rows = []
for t in (0.35, 0.42, 0.50, 0.62, 0.70):
    f, tp, fp, fn, p, r = f1_at(t)
    rows.append({"후보 임곗값 T": t, "TP": tp, "FP": fp, "FN": fn, "Precision": round(p, 3), "Recall": round(r, 3), "F1": round(f, 3)})
save_table(pd.DataFrame(rows), HERE / "표_후보_임곗값별_판정.csv")
print(pd.DataFrame(rows).to_string(index=False))
