"""3.9 입력 형식에 따른 차이 — 모델별로 본 입력 형식(원데이터 / 파생) · 길이(0.5~2.0초)에 따른 OOF F1.

입력: models/<모델>/results/performance.csv
출력: 그림_입력_형식별_F1.png, 표_형식별_최고_F1.csv
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # work 폴더

import pandas as pd  # noqa: E402

from common.paths import save_table  # noqa: E402
from common.plot_style import DER_C, RAW_C, SINGLE_C, plt, save  # noqa: E402
from common.results import MODELS, SHORT, performance  # noqa: E402

d = performance()
secs = [0.5, 1.0, 1.5, 2.0]
fig, axes = plt.subplots(4, 2, figsize=(10, 10), sharex=True, sharey=True)
rows = []
for n, (ax, m) in enumerate(zip(axes.flat, MODELS)):
    g = d[d.model == m].set_index("input_label")
    raw = [g.oof_f1[f"원데이터 {s:.1f}초"] for s in secs]
    der = [g.oof_f1[f"파생 {s:.1f}초"] for s in secs]
    single = g.oof_f1["단일시점"]
    ax.axhline(single, color=SINGLE_C, linestyle="--", linewidth=1.2, label="단일시점")
    ax.plot(secs, raw, color=RAW_C, marker="o", linewidth=1.6, label="원데이터 window")
    ax.plot(secs, der, color=DER_C, marker="s", linewidth=1.6, label="파생변수 / 파생채널")
    ax.set_title(f"{'①②③④⑤⑥⑦⑧'[n]} {SHORT.get(m, m)}")
    ax.set_ylim(0.3, 1.02)
    ax.set_xticks(secs)
    best_raw, best_der = max(zip(raw, secs)), max(zip(der, secs))
    rows.append({"모델": m, "단일시점": round(single, 3), "원데이터 최고": round(best_raw[0], 3), "원데이터 최고 길이(초)": best_raw[1],
                 "파생 최고": round(best_der[0], 3), "파생 최고 길이(초)": best_der[1],
                 "더 좋은 형식": "원데이터" if best_raw[0] > best_der[0] else "파생"})
for ax in axes[-1]:
    ax.set_xlabel("입력 길이 (초)")
for ax in axes[:, 0]:
    ax.set_ylabel("OOF F1")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
fig.suptitle("입력 형식·길이에 따른 OOF F1 (모델별)")
fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=1.5, w_pad=2)
save(fig, HERE / "그림_입력_형식별_F1.png")
save_table(pd.DataFrame(rows), HERE / "표_형식별_최고_F1.csv")
print(pd.DataFrame(rows).to_string(index=False))
