"""2.4 핵심 시각화 — 여러 특징을 2차원으로 줄여 본 전체 분포 (PCA, 탐색용).

20시점 이상 segment마다 가운데 2초(20시점)를 1개씩 잘라 10개 특징을 계산한다 (정상 452 · 이상 13).
특징: 세 센서 각각의 RMS · 첨도 · crest factor (9개) + AI0·AI1 상관 (1개). 표준화 후 PCA.

출력: 그림_PCA.png, 표_PCA_loading.csv, 표_PCA_설명분산.csv, 표_PCA_점수.csv, 표_2초_특징.csv
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # work 폴더

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kurtosis  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from common.data import SENSORS, load_frames  # noqa: E402
from common.paths import save_table  # noqa: E402
from common.plot_style import C, GRID, LABEL, plt, save  # noqa: E402

fr = load_frames()
rows = []
for k, frame in fr.items():
    for sid, segment in frame.groupby("segment_id", sort=False):
        if len(segment) < 20:
            continue
        start = (len(segment) - 20) // 2
        window = segment.iloc[start:start + 20]
        v = window[SENSORS].to_numpy(dtype=float)
        rms = np.sqrt(np.mean(v ** 2, axis=0))
        crest = np.max(np.abs(v), axis=0) / rms
        excess = kurtosis(v, axis=0, fisher=True, bias=False)
        rows.append({"class": k, "date": window.TimeStamp.iloc[0].date().isoformat(), "segment_id": int(sid),
                     "window_start": window.TimeStamp.iloc[0],
                     "AI0_RMS": rms[0], "AI1_RMS": rms[1], "Current_RMS": rms[2],
                     "AI0_kurtosis": excess[0], "AI1_kurtosis": excess[1], "Current_kurtosis": excess[2],
                     "AI0_crest_factor": crest[0], "AI1_crest_factor": crest[1], "Current_crest_factor": crest[2],
                     "AI0_AI1_signed_corr": np.corrcoef(v[:, 0], v[:, 1])[0, 1]})
features = pd.DataFrame(rows)
cols = ["AI0_RMS", "AI1_RMS", "Current_RMS", "AI0_kurtosis", "AI1_kurtosis", "Current_kurtosis",
        "AI0_crest_factor", "AI1_crest_factor", "Current_crest_factor", "AI0_AI1_signed_corr"]
model = PCA(n_components=2)
scores = model.fit_transform(StandardScaler().fit_transform(features[cols]))
features["PC1"], features["PC2"] = scores[:, 0], scores[:, 1]
ev = model.explained_variance_ratio_

fig, ax = plt.subplots(figsize=(10, 4.4))
for k in ["normal", "anomaly"]:
    g = features[features["class"] == k]
    ax.scatter(g.PC1, g.PC2, s=12 if k == "normal" else 34, color=C[k], alpha=0.5 if k == "normal" else 0.95,
               edgecolors="white", linewidths=0 if k == "normal" else 0.8, label=f"{LABEL[k]} = {g.date.iloc[0]} ({len(g)}개 구간)")
ax.axhline(0, color=GRID, lw=1)
ax.axvline(0, color=GRID, lw=1)
ax.set_xlabel(f"PC1 (전체 정보의 {ev[0] * 100:.1f}%)")
ax.set_ylabel(f"PC2 ({ev[1] * 100:.1f}%)")
ax.legend(loc="upper right")
ax.set_title("2초 구간 10개 특징을 2차원으로 요약")
fig.tight_layout()
save(fig, HERE / "그림_PCA.png")

save_table(pd.DataFrame(model.components_.T, index=cols, columns=["PC1", "PC2"]).round(3).reset_index(names="특징"),
           HERE / "표_PCA_loading.csv")
save_table(pd.DataFrame({"성분": ["PC1", "PC2"], "설명 분산 비율": ev.round(4)}), HERE / "표_PCA_설명분산.csv")
save_table(features[["class", "date", "segment_id", "window_start", "PC1", "PC2"]].round(4), HERE / "표_PCA_점수.csv")
save_table(features[["class", "date", "segment_id", "window_start", *cols]].round(5), HERE / "표_2초_특징.csv")
print(pd.DataFrame(model.components_.T, index=cols, columns=["PC1", "PC2"]).round(2), ev)
