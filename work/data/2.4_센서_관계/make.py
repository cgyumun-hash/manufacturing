"""2.4 핵심 시각화 — 센서끼리의 관계: 정상의 '짝 관계'가 이상에서 깨진다.

5시점(0.5초) 이상인 모든 segment를 한 번씩 사용 (segment 전체 길이). 점 1개 = segment 1개.
① segment 안 AI0·AI1 상관   ② AI0 RMS ↔ AI1 RMS   ③ 전류 RMS ↔ AI0 RMS   ④ 전류 RMS ↔ AI1 RMS

출력: 그림_센서_관계.png, 표_센서_관계.csv, 표_segment별_RMS.csv
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # work 폴더

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common.data import SENSORS, load_frames  # noqa: E402
from common.paths import save_table  # noqa: E402
from common.plot_style import C, INK, INK2, LABEL, plt, save  # noqa: E402

fr = load_frames()
rms = lambda x: float(np.sqrt(np.mean(np.square(x))))  # noqa: E731
rows = []
for k, f in fr.items():
    for sid, g in f.groupby("segment_id"):
        if len(g) < 5:
            continue
        a0, a1, cu = (g[s].to_numpy(float) for s in SENSORS)
        r = np.corrcoef(a0, a1)[0, 1] if a0.std() > 0 and a1.std() > 0 else np.nan
        rows.append(dict(cls=k, sid=sid, n=len(g), AI0=rms(a0), AI1=rms(a1), CUR=rms(cu), r01=r))
d = pd.DataFrame(rows)

fig, axes = plt.subplots(2, 2, figsize=(10, 6.4))
ax = axes[0, 0]
rng = np.random.default_rng(0)
for i, k in enumerate(["normal", "anomaly"]):
    v = d[d.cls == k].r01.dropna()
    ax.boxplot(v, positions=[i], widths=0.45, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="white", edgecolor=INK2), medianprops=dict(color=INK, lw=1.5),
               whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
    ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=9 if k == "normal" else 22, color=C[k],
               alpha=0.45 if k == "normal" else 0.9, linewidths=0, zorder=3)
    ax.text(i + 0.3, v.median(), f"중앙값 {v.median():+.2f}", va="center", fontsize=8.5, color=INK)
ax.axhline(0, color=INK2, lw=0.8, ls=(0, (3, 3)))
ax.set_xticks([0, 1])
ax.set_xticklabels([f"정상 ({(d.cls == 'normal').sum()}개)", f"이상 ({(d.cls == 'anomaly').sum()}개)"])
ax.set_ylabel("segment 안 AI0·AI1 상관 (r)")
ax.set_ylim(-1.05, 1.05)
ax.set_title("① 상·하부 진동이 같은 방향으로 움직이나")
pairs = [(axes[0, 1], "AI0", "AI1", "② AI0 RMS ↔ AI1 RMS", "AI0 상부 진동 RMS", "AI1 하부 진동 RMS"),
         (axes[1, 0], "CUR", "AI0", "③ 전류 RMS ↔ AI0 RMS", "전류 RMS", "AI0 상부 진동 RMS"),
         (axes[1, 1], "CUR", "AI1", "④ 전류 RMS ↔ AI1 RMS", "전류 RMS", "AI1 하부 진동 RMS")]
corr = {}
for ax, xc, yc, title, xl, yl in pairs:
    txt = []
    for k in ["normal", "anomaly"]:
        s = d[d.cls == k]
        ax.scatter(s[xc], s[yc], s=10 if k == "normal" else 30, color=C[k], alpha=0.5 if k == "normal" else 0.95,
                   linewidths=0 if k == "normal" else 0.8, edgecolors="white", label=LABEL[k], zorder=3 if k == "anomaly" else 2)
        b = np.polyfit(s[xc], s[yc], 1)
        xs = np.linspace(s[xc].min(), s[xc].max(), 10)
        ax.plot(xs, np.polyval(b, xs), color=C[k], lw=1.8)
        corr[(title, k)] = np.corrcoef(s[xc], s[yc])[0, 1]
        txt.append(f"{LABEL[k]} r = {corr[(title, k)]:+.2f}")
    ax.text(0.02, 0.97, "\n".join(txt), transform=ax.transAxes, va="top", fontsize=8.5, color=INK,
            bbox=dict(fc="white", ec="#c9c8c2", lw=0.6, pad=3))
    ax.set_title(title)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
axes[0, 1].legend(loc="lower right")
fig.tight_layout(h_pad=1.6, w_pad=2)
save(fig, HERE / "그림_센서_관계.png")

med = d.groupby("cls").r01.median()
table = pd.DataFrame([
    {"관계": "AI0 ↔ AI1 (segment 안 상관의 중앙값)", "정상": round(med["normal"], 2), "이상": round(med["anomaly"], 2),
     "무엇을 보나": "상·하부가 같은 방향으로 흔들리나"},
    *[{"관계": t.split(" ", 1)[1], "정상": round(corr[(t, "normal")], 2), "이상": round(corr[(t, "anomaly")], 2), "무엇을 보나": w}
      for (_, _, _, t, _, _), w in zip(pairs, ["상·하부 흔들림 크기가 같이 커지나", "부하가 크면 상부도 더 흔들리나",
                                               "부하가 크면 하부도 더 흔들리나"])]])
save_table(table, HERE / "표_센서_관계.csv")
save_table(d.rename(columns={"cls": "class", "sid": "segment_id", "n": "시점 수", "AI0": "AI0_RMS", "AI1": "AI1_RMS",
                             "CUR": "Current_RMS", "r01": "AI0_AI1_상관"}).round(5), HERE / "표_segment별_RMS.csv")
print(table.to_string(index=False))
