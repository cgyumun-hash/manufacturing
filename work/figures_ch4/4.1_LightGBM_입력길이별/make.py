"""4.1 왜 LightGBM 1초 원데이터가 가장 높았나 — 입력 길이별 OOF F1과 학습에 쓰인 이상 window 수.

입력: models/LightGBM/results/performance.csv, common/windows.py (window 수 계산)
출력: 그림_LightGBM_입력길이별.png, 표_LightGBM_입력길이별.csv
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # work 폴더

import pandas as pd  # noqa: E402

from common.paths import save_table  # noqa: E402
from common.plot_style import DER_C, RAW_C, SINGLE_C, plt, save  # noqa: E402
from common.results import performance  # noqa: E402
from common.windows import make_windows, prepare  # noqa: E402

g = performance().query("model == 'LightGBM'").set_index("input_label")
eligible = prepare()[0]
secs = [0.5, 1.0, 1.5, 2.0]
counts = {s: make_windows(eligible, round(s * 10))["labels"] for s in secs}
raw = [g.oof_f1[f"원데이터 {s:.1f}초"] for s in secs]
der = [g.oof_f1[f"파생 {s:.1f}초"] for s in secs]

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
ax = axes[0]
ax.axhline(g.oof_f1["단일시점"], color=SINGLE_C, linestyle="--", linewidth=1.2, label="단일시점")
ax.plot(secs, raw, color=RAW_C, marker="o", linewidth=1.8, label="원데이터 window")
ax.plot(secs, der, color=DER_C, marker="s", linewidth=1.8, label="파생변수 window")
for x, v in zip(secs, raw):
    ax.text(x + (0.07 if x == 2.0 else 0), v + (-0.035 if x == 2.0 else 0.02), f"{v:.3f}", ha="center", fontsize=8.5, color=RAW_C)
ax.set_xticks(secs)
ax.set_ylim(0.6, 1.02)
ax.set_xlabel("입력 길이 (초)")
ax.set_ylabel("OOF F1")
ax.set_title("① LightGBM 입력 길이별 OOF F1")
ax.legend(loc="lower left")
ax = axes[1]
anomaly = [int((counts[s] == 1).sum()) for s in secs]
ax.bar(secs, anomaly, width=0.3, color="#D9534F")
for x, v in zip(secs, anomaly):
    ax.text(x, v + 2, f"{v}", ha="center", fontsize=9)
ax.set_xticks(secs)
ax.set_ylim(0, 150)
ax.set_xlabel("입력 길이 (초)")
ax.set_ylabel("이상 window 수 (개)")
ax.set_title("② 입력 길이별 이상 window 수 (이상 13개 segment)")
fig.tight_layout(w_pad=2.5)
save(fig, HERE / "그림_LightGBM_입력길이별.png")
table = pd.DataFrame({"입력 길이(초)": secs, "시점 수": [round(s * 10) for s in secs], "원데이터 F1": [round(v, 3) for v in raw],
                      "파생변수 F1": [round(v, 3) for v in der], "정상 window": [int((counts[s] == 0).sum()) for s in secs],
                      "이상 window": anomaly, "원데이터 PR-AUC": [round(g.oof_pr_auc[f"원데이터 {s:.1f}초"], 3) for s in secs]})
save_table(table, HERE / "표_LightGBM_입력길이별.csv")
print(table.to_string(index=False), "단일시점", round(g.oof_f1["단일시점"], 3))
