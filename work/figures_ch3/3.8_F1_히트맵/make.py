"""3.8 주요 성능 결과 — 8개 모델 × 9개 입력의 OOF F1 (빨간 테두리 = 모델별 최고, 동점 포함).

입력: models/<모델>/results/performance.csv (각 모델의 test.py 결과)
출력: 그림_F1_히트맵.png, 표_F1_모델x입력.csv
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # work 폴더

import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from common.paths import save_table  # noqa: E402
from common.plot_style import plt, save  # noqa: E402
from common.results import MODELS, performance  # noqa: E402
from common.windows import INPUT_ORDER  # noqa: E402

perf = performance()
p = perf.pivot(index="model", columns="input_label", values="oof_f1").reindex(index=MODELS, columns=INPUT_ORDER)
fig, ax = plt.subplots(figsize=(10, 5.4))
im = ax.imshow(p.values, cmap="Blues", vmin=0.3, vmax=1.0, aspect="auto")
for i in range(p.shape[0]):
    top = np.nanmax(p.values[i])
    for j in range(p.shape[1]):
        v = p.values[i, j]
        best = np.isclose(v, top)
        ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8.5, color="white" if v >= 0.8 else "black",
                weight="bold" if best else "normal")
        if best:
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#D9534F", linewidth=2.2))
for x in (0.5, 4.5):
    ax.axvline(x, color="white", linewidth=3)
ax.axhline(4.5, color="white", linewidth=3)
ax.set_xticks(range(len(INPUT_ORDER)))
ax.set_xticklabels([x.replace(" ", "\n") for x in INPUT_ORDER])
ax.set_yticks(range(len(MODELS)))
ax.set_yticklabels(MODELS)
ax.set_title("모델 × 입력별 OOF F1 (빨간 테두리 = 모델별 최고, 동점 포함)")
fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02).set_label("OOF F1")
fig.tight_layout()
save(fig, HERE / "그림_F1_히트맵.png")
save_table(p.round(3).reset_index(), HERE / "표_F1_모델x입력.csv")
print(p.round(3).to_string())
