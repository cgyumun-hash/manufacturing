"""2.1 데이터의 주요 특징 — 정상과 이상의 행 수 · 연속 segment 수 · 수집 시간 비교.

출력: 그림_데이터_규모.png, 표_데이터_구성.csv
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # work 폴더

from common.data import load_frames  # noqa: E402
from common.paths import save_table  # noqa: E402
from common.plot_style import C, INK, LABEL, plt, save  # noqa: E402

import pandas as pd  # noqa: E402

fr = load_frames()
rows = {k: len(v) for k, v in fr.items()}
segs = {k: v.segment_id.nunique() for k, v in fr.items()}
dur = {k: (v.TimeStamp.iloc[-1] - v.TimeStamp.iloc[0]).total_seconds() / 60 for k, v in fr.items()}

fig, axes = plt.subplots(1, 3, figsize=(10, 3.0))
for ax, (title, data, fmt) in zip(axes, [("① 행 개수 (전체)", rows, "{:,.0f}행"), ("② 연속 segment 개수", segs, "{:,.0f}개"),
                                         ("③ 수집 시간 길이 (분)", dur, "{:.1f}분")]):
    ks = ["normal", "anomaly"]
    vals = [data[k] for k in ks]
    ax.barh([LABEL[k] for k in ks], vals, color=[C[k] for k in ks], height=0.55)
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.02, i, fmt.format(v), va="center", fontsize=9, color=INK)
    ax.set_xlim(0, max(vals) * 1.32)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.tick_params(axis="y", length=0)
fig.tight_layout(w_pad=2.5)
save(fig, HERE / "그림_데이터_규모.png")

table = pd.DataFrame([{
    "구분": LABEL[k], "원본 행": fr[k].attrs["raw_rows"], "정제 후 행": rows[k], "연속 segment": segs[k],
    "수집 날짜": f"{fr[k].TimeStamp.iloc[0]:%Y-%m-%d}",
    "첫 기록": f"{fr[k].TimeStamp.iloc[0]:%H:%M:%S}", "마지막 기록": f"{fr[k].TimeStamp.iloc[-1]:%H:%M:%S}",
    "수집 시간(분)": round(dur[k], 1), "라벨": 0 if k == "normal" else 1} for k in ["normal", "anomaly"]])
save_table(table, HERE / "표_데이터_구성.csv")
print(table.to_string(index=False))
