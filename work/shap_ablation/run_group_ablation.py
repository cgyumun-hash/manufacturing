"""4.3 그룹 ablation — 파생변수 그룹을 하나씩 통째로 빼고 다시 학습해 F1이 얼마나 떨어지는지 본다.

Tabular: G1 진동 크기 · G2 상·하부 관계 · G3 부하-진동 · G4 시간 변화 · G5 충격성 (19개 변수)
Sequence: T0 원센서 · T1 국소 에너지 · T2 관계 · T3 추세 (12채널)
평가: fold별 validation segment 평균 점수로 임곗값을 고르고 test에 적용, 5개 fold의 F1 평균.

출력: results/cv_folds.csv, results/cv_summary.csv, results/group_drops.csv
실행: python work/shap_ablation/run_group_ablation.py
"""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import ablation_setup as S
import ablation_models as AM
from common.evaluation import best_threshold, classification
from common.features import (TABULAR_FEATURES, TABULAR_GROUPS, TEMPORAL_FEATURES, TEMPORAL_GROUPS,
                             tabular_features, temporal_features)
from common.paths import save_table
from common.pipeline import seed_all


def evaluate(groups, labels, scores):
    def grouped(part):
        return (pd.DataFrame({"group": groups[part], "label": labels[part], "score": scores[part]})
                .groupby("group", sort=False).agg(label=("label", "first"), score=("score", "mean")).reset_index())
    valid = grouped("valid")
    threshold = best_threshold(valid.label.to_numpy(), valid.score.to_numpy())
    test = grouped("test")
    pred = (test.score >= threshold).astype(int)
    margin = (test.score - threshold) / (abs(threshold) + 1e-9)
    return {**classification(test.label, pred), "pr_auc": average_precision_score(test.label, margin)}


def summarize(folds):
    keys = ["family", "model", "window_samples", "window_seconds", "experiment", "removed", "feature_count", "features"]
    return folds.groupby(keys, as_index=False).agg(
        precision_mean=("precision", "mean"), precision_std=("precision", "std"),
        recall_mean=("recall", "mean"), recall_std=("recall", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"), pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
        tn=("tn", "sum"), fp=("fp", "sum"), fn=("fn", "sum"), tp=("tp", "sum"), fit_seconds=("fit_seconds", "sum"))


def group_drops(summary):
    full = summary[summary.experiment == "full"][["family", "model", "window_samples", "f1_mean", "pr_auc_mean"]].rename(
        columns={"f1_mean": "full_f1", "pr_auc_mean": "full_pr_auc"})
    result = summary[summary.experiment == "group_ablation"].merge(full, on=["family", "model", "window_samples"], how="left")
    result["f1_drop"] = result.full_f1 - result.f1_mean
    result["pr_auc_drop"] = result.full_pr_auc - result.pr_auc_mean
    return result


def main():
    seed_all(S.SEED)
    by_length, names, labels = S.prepare_windows()
    rows = []
    for split in S.make_splits(names, labels):
        fold = split["fold"]
        for length in S.LENGTHS:
            raw_split, y_split, g_split = S.split_raw(*by_length[length], split)
            tab = S.scale_features(tabular_features(raw_split, y_split), y_split, temporal=False)
            tem = S.scale_features(temporal_features(raw_split, y_split), y_split, temporal=True)
            for family, models, full, feat, groups, temporal in (
                    ("tabular", S.TABULAR_MODELS, tab, TABULAR_FEATURES, TABULAR_GROUPS, False),
                    ("temporal", S.TEMPORAL_MODELS, tem, TEMPORAL_FEATURES, TEMPORAL_GROUPS, True)):
                variants = [("full", "none", list(feat))] + [
                    ("group_ablation", g, [n for n in feat if n not in members]) for g, members in groups.items()]
                for experiment, removed, selected in variants:
                    values = S.select_columns(full, feat, selected, temporal)
                    for model_name in models:
                        started = time.perf_counter()
                        model = AM.fit(model_name, values, y_split, S.configuration_seed(fold, length, family, model_name))
                        scores = {k: AM.score(model_name, model, v) for k, v in values.items()}
                        elapsed = time.perf_counter() - started
                        rows.append({"family": family, "model": model_name, "window_samples": length,
                                     "window_seconds": length / 10, "fold": fold, "experiment": experiment,
                                     "removed": removed, "feature_count": len(selected), "features": ",".join(selected),
                                     **evaluate(g_split, y_split, scores), "fit_seconds": elapsed})
            print(f"group ablation: fold {fold}/5, {length / 10:.1f}초 완료", flush=True)
    folds = pd.DataFrame(rows)
    summary = summarize(folds)
    save_table(folds, S.RESULTS / "cv_folds.csv")
    save_table(summary, S.RESULTS / "cv_summary.csv")
    save_table(group_drops(summary), S.RESULTS / "group_drops.csv")
    for family in ("tabular", "temporal"):
        w = summary[(summary.family == family) & (summary.experiment == "full")].sort_values(
            ["f1_mean", "pr_auc_mean", "recall_mean"], ascending=False).iloc[0]
        print(f"{family} winner: {w.model} {w.window_seconds}초 F1 {w.f1_mean:.3f}")


if __name__ == "__main__":
    main()
