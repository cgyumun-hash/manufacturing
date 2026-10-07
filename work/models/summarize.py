"""8개 모델의 성능표와 5장 마르코프 결과를 한곳(models/summary/)에 모은다.

실행 순서: 각 모델의 train → valid → test → markov 를 끝낸 뒤
    python work/models/summarize.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import pandas as pd  # noqa: E402

from common.paths import SUMMARY, save_table  # noqa: E402
from common.results import best_rows  # noqa: E402
from common.windows import INPUT_ORDER  # noqa: E402

MODEL_DIRS = ["LogisticRegression", "LightGBM", "XGBoost", "Mahalanobis", "OneClassSVM", "CNN1D", "GRU", "LSTM_AE"]


def main():
    perf = pd.concat([pd.read_csv(HERE / d / "results" / "performance.csv") for d in MODEL_DIRS], ignore_index=True)
    perf["input_label"] = pd.Categorical(perf.input_label, INPUT_ORDER, ordered=True)
    perf = perf.sort_values(["model", "input_label"], key=lambda s: s.map({m: i for i, m in enumerate(
        pd.unique(perf.model))}) if s.name == "model" else s).reset_index(drop=True)
    best = best_rows(perf)
    markov = pd.concat([pd.read_csv(HERE / d / "markov" / "markov_summary.csv") for d in MODEL_DIRS], ignore_index=True)
    markov = markov.sort_values(["markov_f1", "markov_recall"], ascending=False)
    save_table(perf, SUMMARY / "all_performance.csv")
    save_table(best, SUMMARY / "best_by_model.csv")
    save_table(markov, SUMMARY / "markov_summary_all.csv")

    lines = ["# 8개 모델 요약", "", "## 모델별 최고 입력 (3장)", "",
             "| 순위 | 모델 | 최고 입력 | Precision | Recall | F1 | PR-AUC | FP | FN |", "|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for i, r in best.iterrows():
        lines.append(f"| {i + 1} | {r.model} | {r.input_label} | {r.oof_precision:.3f} | {r.oof_recall:.3f} | "
                     f"**{r.oof_f1:.3f}** | {r.oof_pr_auc:.3f} | {r.fp} | {r.fn} |")
    table = perf.pivot(index="model", columns="input_label", values="oof_f1")
    lines += ["", "## 모델 × 입력 OOF F1", "", "| 모델 | " + " | ".join(INPUT_ORDER) + " |",
              "|---|" + "---:|" * len(INPUT_ORDER)]
    for model in pd.unique(perf.model):
        lines.append(f"| {model} | " + " | ".join(f"{table.loc[model, c]:.3f}" for c in INPUT_ORDER) + " |")
    lines += ["", "## 최고 입력 + 마르코프 연속경고 (5장)", "",
              "| 모델 | 입력 | 기존 F1 | 마르코프 F1 | Precision | Recall | 95% 상한 F1 | K | 상한 K |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in markov.iterrows():
        k = f"{r.markov_k_min}~{r.markov_k_max}" if r.markov_k_min != r.markov_k_max else str(r.markov_k_min)
        ku = f"{r.upper_k_min}~{r.upper_k_max}" if r.upper_k_min != r.upper_k_max else str(r.upper_k_min)
        lines.append(f"| {r.model} | {r.input_label} | {r.baseline_f1:.3f} | **{r.markov_f1:.3f}** | "
                     f"{r.markov_precision:.3f} | {r.markov_recall:.3f} | {r.upper_f1:.3f} | {k} | {ku} |")
    (SUMMARY / "전체_요약.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(best[["model", "input_label", "oof_f1"]].to_string(index=False))


if __name__ == "__main__":
    main()
