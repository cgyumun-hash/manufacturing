"""2.4 핵심 시각화 — 정상·이상 각각 가장 긴 segment(50시점 = 4.9초)의 세 센서 파형 (샘플 1개씩).

출력: 그림_대표_segment_파형.png, 표_대표_segment.csv, 표_대표_segment_원값.csv
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # work 폴더

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common.data import DT, SENSOR_KO, SENSORS, load_frames  # noqa: E402
from common.paths import save_table  # noqa: E402
from common.plot_style import C, LABEL, plt, save  # noqa: E402

fr = load_frames()


def longest(frame):
    sid = frame.groupby("segment_id").size().idxmax()  # 가장 긴 길이 중 시간상 첫 번째
    return frame[frame.segment_id == sid].reset_index(drop=True)


fig, axes = plt.subplots(3, 2, figsize=(10, 7.2), sharex="col")
rows, values = [], []
for col, k in enumerate(["normal", "anomaly"]):
    g = longest(fr[k])
    t = np.arange(len(g)) * DT
    for row, s in enumerate(SENSORS):
        axes[row, col].plot(t, g[s], color=C[k], linewidth=1.1)
        axes[row, col].set_ylabel(SENSOR_KO[s])
    axes[0, col].set_title(f"{'①②'[col]} {LABEL[k]} — 가장 긴 segment ({len(g)}시점)")
    axes[-1, col].set_xlabel("segment 시작 후 시간 (초)")
    row = {"구분": LABEL[k], "segment 번호": int(g.segment_id.iloc[0]), "시작 시각": f"{g.TimeStamp.iloc[0]:%Y-%m-%d %H:%M:%S.%f}"[:-3],
           "시점 수": len(g)}
    for s in SENSORS:
        v = g[s].to_numpy(float)
        row.update({f"{SENSOR_KO[s]} RMS": round(float(np.sqrt(np.mean(v ** 2))), 4),
                    f"{SENSOR_KO[s]} 최솟값": round(v.min(), 4), f"{SENSOR_KO[s]} 최댓값": round(v.max(), 4)})
    rows.append(row)
    values.append(g.assign(구분=LABEL[k], 경과초=np.round(t, 1))[["구분", "segment_id", "TimeStamp", "경과초", *SENSORS]])
fig.tight_layout()
save(fig, HERE / "그림_대표_segment_파형.png")
save_table(pd.DataFrame(rows), HERE / "표_대표_segment.csv")
save_table(pd.concat(values), HERE / "표_대표_segment_원값.csv")
print(pd.DataFrame(rows).T.to_string())
