"""2.2 데이터 품질 — 이웃한 두 기록 사이 시간 간격 분포 (0.1초가 아니면 '끊김').

출력: 그림_타임스탬프_간격.png, 표_시간_간격_통계.csv
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # work 폴더

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common.data import DT, load_frames  # noqa: E402
from common.paths import save_table  # noqa: E402
from common.plot_style import C, LABEL, plt, save  # noqa: E402

fr = load_frames()
fig, axes = plt.subplots(2, 1, figsize=(10, 6))
for n, (ax, k) in enumerate(zip(axes, ["normal", "anomaly"])):
    dt = fr[k].interval.dropna().to_numpy(float)
    ax.hist(dt, bins=80, color=C[k], alpha=0.85)
    ax.axvline(DT, color="black", linestyle="--", label="기대 간격 0.1초")
    ax.set_yscale("log")
    ax.set_title(f"{'①②'[n]} {LABEL[k]} 타임스탬프 간격 ({len(dt):,}개)")
    ax.set_xlabel("이웃한 두 기록 사이 시간 (초)")
    ax.set_ylabel("개수 (로그축)")
    ax.legend()
fig.tight_layout()
save(fig, HERE / "그림_타임스탬프_간격.png")

rows = []
for k, f in fr.items():
    iv = f.interval.iloc[1:]
    gap = ~np.isclose(iv, DT, atol=1e-9)
    rows.append({"구분": LABEL[k], "간격 수": len(iv), "0.1초 (정상 연결)": int((~gap).sum()), "0.1초 초과 (끊김)": int(gap.sum()),
                 "끊김의 중앙값(초)": round(float(iv[gap].median()), 1), "가장 긴 공백(초)": round(float(iv.max()), 1),
                 "평균(초)": round(float(iv.mean()), 3), "표준편차(초)": round(float(iv.std()), 3)})
table = pd.DataFrame(rows)
save_table(table, HERE / "표_시간_간격_통계.csv")
print(table.to_string(index=False))
