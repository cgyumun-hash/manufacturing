"""4.3 SHAP과 ablation의 보조 해석 — 어떤 변수 · 그룹이 판단에 많이 쓰였나.

① One-Class SVM 1.5초 SHAP 상위 10개  ② One-Class SVM 그룹 제거 시 F1 감소  ③ 1D-CNN 그룹 제거 시 F1 감소
입력: shap_ablation/results/shap_summary.csv, group_drops.csv (shap_ablation 폴더의 실험 결과)
출력: 그림_SHAP_ablation.png, 표_SHAP_상위10.csv, 표_그룹_제거_F1감소.csv
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # work 폴더

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common.paths import SHAP_ABLATION, save_table  # noqa: E402
from common.plot_style import DER_C, RAW_C, plt, save  # noqa: E402

shap = pd.read_csv(SHAP_ABLATION / "results" / "shap_summary.csv")
sh = shap[(shap.model == "One-Class SVM") & (shap.window_seconds == 1.5)].sort_values("rank").head(10)
drops = pd.read_csv(SHAP_ABLATION / "results" / "group_drops.csv")
pick = lambda m: drops[(drops.model == m) & (drops.window_seconds == 1.5)]  # noqa: E731
oc, cn = pick("One-Class SVM"), pick("1D-CNN")
names = {"G1": "G1 진동 크기", "G2": "G2 상·하부 관계", "G3": "G3 부하-진동", "G4": "G4 시간 변화", "G5": "G5 충격성",
         "T0": "T0 원센서", "T1": "T1 국소 에너지", "T2": "T2 관계", "T3": "T3 추세"}

fig = plt.figure(figsize=(10, 6.4))
gsp = fig.add_gridspec(2, 2, height_ratios=[1.25, 1])
axes = [fig.add_subplot(gsp[0, :]), fig.add_subplot(gsp[1, 0]), fig.add_subplot(gsp[1, 1])]
ax = axes[0]
y = np.arange(len(sh))[::-1]
ax.barh(y, sh.balanced_mean_abs_shap, color=DER_C)
for yi, v in zip(y, sh.balanced_mean_abs_shap):
    ax.text(v + 0.005, yi, f"{v:.3f}", va="center", fontsize=9)
ax.set_yticks(y)
ax.set_yticklabels(sh.feature)
ax.set_xlim(0, 0.58)
ax.set_xlabel("평균 |SHAP| (정상·이상 균형)")
ax.set_title("① One-Class SVM 1.5초 SHAP 상위 10개")
for ax, g, title, color in [(axes[1], oc, "② One-Class SVM 그룹 제거", DER_C), (axes[2], cn, "③ 1D-CNN 그룹 제거", RAW_C)]:
    g = g.sort_values("f1_drop")
    yy = np.arange(len(g))
    ax.barh(yy, g.f1_drop, color=color)
    for yi, v in zip(yy, g.f1_drop):
        ax.text(v + 0.005, yi, f"{v:.3f}", va="center", fontsize=8.5)
    ax.set_yticks(yy)
    ax.set_yticklabels([names[r] for r in g.removed])
    ax.set_xlim(0, 0.3)
    ax.set_xlabel("제거 시 F1 감소 (fold 평균)")
    ax.set_title(title)
fig.tight_layout(w_pad=2, h_pad=1.5)
save(fig, HERE / "그림_SHAP_ablation.png")

save_table(sh[["rank", "feature", "balanced_mean_abs_shap", "mean_abs_shap_normal", "mean_abs_shap_anomaly",
               "mean_signed_shap_anomaly"]].round(4), HERE / "표_SHAP_상위10.csv")
table = pd.concat([oc, cn]).assign(그룹=lambda x: x.removed.map(names))[
    ["model", "window_seconds", "그룹", "full_f1", "f1_mean", "f1_drop"]].rename(
    columns={"full_f1": "전체 F1", "f1_mean": "제거 후 F1", "f1_drop": "F1 감소"}).round(3)
save_table(table.sort_values(["model", "F1 감소"], ascending=[False, False]), HERE / "표_그룹_제거_F1감소.csv")
print(table.to_string(index=False))
