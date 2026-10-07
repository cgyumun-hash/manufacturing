"""4.2 파생변수가 항상 성능을 높이지 않은 이유 — 모델별로 단일시점에서 최고 입력으로 바꿨을 때의 OOF F1 변화.

입력: models/<모델>/results/performance.csv
출력: 그림_단일시점_대비_최고.png, 표_단일시점_대비_최고.csv
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # work 폴더

import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from common.paths import save_table  # noqa: E402
from common.plot_style import DER_C, RAW_C, SINGLE_C, plt, save  # noqa: E402
from common.results import MODELS, SHORT, performance  # noqa: E402

d = performance()
rows = []
for m in MODELS:
    g = d[d.model == m]
    single = float(g[g.input_label == "단일시점"].oof_f1.iloc[0])
    b = g[g.input_label != "단일시점"].sort_values(["oof_f1", "oof_pr_auc"], ascending=False).iloc[0]
    rows.append((m, single, float(b.oof_f1), b.input_label))
rows.sort(key=lambda r: r[2] - r[1])
fig, ax = plt.subplots(figsize=(10, 4.6))
for i, (m, s_, b, lab) in enumerate(rows):
    color = RAW_C if lab.startswith("원데이터") else DER_C
    ax.plot([s_, b], [i, i], color="#BBBBBB", linewidth=2, zorder=1)
    ax.scatter([s_], [i], color=SINGLE_C, s=45, zorder=2)
    ax.scatter([b], [i], color=color, s=60, zorder=3, marker="o" if color == RAW_C else "s")
    ax.text(b + 0.012, i, f"{b:.3f}  ({lab})", va="center", fontsize=8.5)
    ax.text(s_ - 0.012, i, f"{s_:.3f}", va="center", ha="right", fontsize=8.5, color=SINGLE_C)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([SHORT.get(r[0], r[0]) for r in rows])
ax.set_xlim(0.25, 1.15)
ax.set_xlabel("OOF F1")
ax.set_title("단일시점 → 최고 입력으로 바꿨을 때의 OOF F1")
ax.legend(handles=[Line2D([], [], marker="o", color=SINGLE_C, linestyle="", label="단일시점"),
                   Line2D([], [], marker="o", color=RAW_C, linestyle="", label="최고 = 원데이터 window"),
                   Line2D([], [], marker="s", color=DER_C, linestyle="", label="최고 = 파생변수 / 파생채널")],
          loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
fig.tight_layout()
save(fig, HERE / "그림_단일시점_대비_최고.png")
table = pd.DataFrame([{"모델": m, "단일시점 F1": round(s_, 3), "최고 입력": lab, "최고 F1": round(b, 3), "상승폭": round(round(b, 3) - round(s_, 3), 3)}
                      for m, s_, b, lab in rows[::-1]])
save_table(table, HERE / "표_단일시점_대비_최고.csv")
print(table.to_string(index=False))
