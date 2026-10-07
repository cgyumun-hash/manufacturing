"""3.8 주요 성능 결과 — 모델별 최고 입력의 Precision · Recall · F1 (OOF).

입력: models/<모델>/results/performance.csv
출력: 그림_모델별_최고_입력.png, 표_모델별_최고_입력.csv
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # work 폴더

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common.paths import save_table  # noqa: E402
from common.plot_style import RAW_C, plt, save  # noqa: E402
from common.results import SHORT, best_rows, performance  # noqa: E402

b = best_rows(performance())
fig, ax = plt.subplots(figsize=(10, 4.4))
x = np.arange(len(b))
w = 0.26
for k, (col, lab, color) in enumerate([("oof_precision", "Precision", "#9BB8D3"), ("oof_recall", "Recall", "#D9534F"),
                                       ("oof_f1", "F1", RAW_C)]):
    xs = x + (k - 1) * w
    ax.bar(xs, b[col], width=w, color=color, label=lab, edgecolor="white", linewidth=1)
    if col == "oof_f1":
        for xi, v in zip(xs, b[col]):
            ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels([f"{SHORT.get(m, m)}\n({i})" for m, i in zip(b.model, b.input_label)])
ax.set_ylim(0, 1.12)
ax.set_ylabel("OOF 지표 (0 ~ 1)")
ax.set_title("모델별 최고 입력의 Precision · Recall · F1")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False)
fig.tight_layout()
save(fig, HERE / "그림_모델별_최고_입력.png")
table = pd.DataFrame({"순위": np.arange(1, len(b) + 1), "모델": b.model, "최고 입력": b.input_label,
                      "Precision": b.oof_precision.round(3), "Recall": b.oof_recall.round(3), "F1": b.oof_f1.round(3),
                      "PR-AUC": b.oof_pr_auc.round(3), "FP (헛경보)": b.fp, "FN (놓침)": b.fn, "TP": b.tp, "TN": b.tn})
save_table(table, HERE / "표_모델별_최고_입력.csv")
print(table.to_string(index=False))
