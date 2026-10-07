"""2.7 window 생성과 누수 방지.

① 입력 길이별 window 중첩률: 이동 간격 = max(1, 시점 수 // 2), 중첩률 = (시점 수 − 이동) ÷ 시점 수
② segment 단위 5-fold Group CV 분할 (2초 입력 가능 segment 전체: 정상 452 · 이상 13).
   같은 segment의 window는 train / valid / test 중 한 곳에만 들어간다.
분할 규칙은 models/의 학습과 똑같은 common/windows.py를 쓴다.

출력: 그림_window_GroupCV.png, 표_fold별_segment_배정.csv, 표_중첩률.csv, 표_입력별_window_수.csv, 표_fold별_segment_목록.csv
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # work 폴더

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common.paths import save_table  # noqa: E402
from common.plot_style import C, LABEL, plt, save  # noqa: E402
from common.windows import INPUTS, make_windows, prepare, stride_of  # noqa: E402

eligible, names, labels, splits = prepare()
label_of = dict(zip(names, labels))
split_rows, member_rows = [], []
for sp in splits:
    row = {"fold": sp["fold"]}
    for part in ("train", "valid", "test"):
        lab = np.asarray([label_of[g] for g in sp[part]])
        row[f"{part}_정상"] = int((lab == 0).sum())
        row[f"{part}_이상"] = int((lab == 1).sum())
        member_rows += [{"fold": sp["fold"], "part": part, "group": g, "label": label_of[g]} for g in sp[part]]
    split_rows.append(row)
split_table = pd.DataFrame(split_rows)

secs, pts = [0.3, 0.5, 1.0, 1.5, 2.0], [3, 5, 10, 15, 20]
rates = [(p - stride_of(p)) / p * 100 for p in pts]
fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
ax = axes[0]
x = np.arange(len(secs))
ax.bar(x, rates, color=C["normal"], width=0.9, edgecolor="white", linewidth=1.5)
ax.axhline(50, color="black", linestyle="--", linewidth=1.2, label="정확히 절반 (50%)")
ax.legend(loc="upper right")
for xi, r in zip(x, rates):
    ax.text(xi, r / 2, f"{r:.1f}%", ha="center", va="center", color="white", fontsize=9, weight="bold")
ax.set_xticks(x)
ax.set_xticklabels([f"{s:.1f}초\n({p}시점)" for s, p in zip(secs, pts)])
ax.set_ylim(0, 80)
ax.set_xlabel("모델 입력 길이")
ax.set_ylabel("이웃 window와 겹치는 비율 (%)")
ax.set_title("① 입력 길이별 window 중첩률")
parts = [("train", "Train\n학습"), ("valid", "Validation\n기준값 결정"), ("test", "Test\n최종 평가")]
ax = axes[1]
w = 0.42
for j, (k, ko) in enumerate([("normal", "정상"), ("anomaly", "이상")]):
    cnt = np.array([split_table[f"{a}_{ko}"].mean() for a, _ in parts])
    pct = cnt / cnt.sum() * 100
    xs = np.arange(3) + (j - 0.5) * w
    ax.bar(xs, pct, width=w, color=C[k], edgecolor="white", linewidth=1.5, label=f"{LABEL[k]} segment (총 {cnt.sum():.0f}개)")
    for xi, pc, c in zip(xs, pct, cnt):
        ax.text(xi, pc + 1.5, f"{c:.1f}개".replace(".0개", "개"), ha="center", va="bottom", fontsize=8.5)
ax.set_xticks(np.arange(3))
ax.set_xticklabels([b for _, b in parts])
ax.set_ylim(0, 80)
ax.set_ylabel("segment 비율 (%)")
ax.set_title("② Group CV 분할 (5개 fold 평균)")
ax.legend()
fig.tight_layout(w_pad=2.5)
save(fig, HERE / "그림_window_GroupCV.png")

overlap = pd.DataFrame({"입력": [f"{s:.1f}초" for s in secs], "window 시점": pts, "한 번에 이동": [stride_of(p) for p in pts],
                        "이웃과 겹치는 시점": [p - stride_of(p) for p in pts], "실제 중첩률(%)": np.round(rates, 1)})
counts = []
for item in INPUTS:
    wv = make_windows(eligible, item["length"])
    counts.append({"입력": item["label"], "window 시점": item["length"], "이동 간격": stride_of(item["length"]),
                   "정상 window": int((wv["labels"] == 0).sum()), "이상 window": int((wv["labels"] == 1).sum()),
                   "정상 segment": int((labels == 0).sum()), "이상 segment": int((labels == 1).sum())})
save_table(split_table, HERE / "표_fold별_segment_배정.csv")
save_table(overlap, HERE / "표_중첩률.csv")
save_table(pd.DataFrame(counts), HERE / "표_입력별_window_수.csv")
save_table(pd.DataFrame(member_rows), HERE / "표_fold별_segment_목록.csv")
print(split_table.to_string(index=False)); print(pd.DataFrame(counts).to_string(index=False))
