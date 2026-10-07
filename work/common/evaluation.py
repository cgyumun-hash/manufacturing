"""segment 단위 평가 (3.6 지표 · 3.7 임곗값).

- window 점수를 segment마다 평균 → segment 점수
- validation segment 점수에서 F1이 가장 높은 값을 임곗값 T*로 고른다.
- test에는 T*를 그대로 적용한다 (test로 임곗값을 고치지 않는다).
- 5개 test fold의 예측을 모두 합쳐 OOF Precision · Recall · F1을 계산한다.
- PR-AUC는 fold마다 (점수 − T*) ÷ validation 점수 표준편차로 맞춘 margin을 합쳐 계산한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score)


def segment_scores(groups, labels, scores, threshold=None, scale=None) -> pd.DataFrame:
    frame = (pd.DataFrame({"group": groups, "label": labels, "score": scores})
             .groupby("group", sort=False).agg(label=("label", "first"), score=("score", "mean")).reset_index())
    if threshold is not None:
        frame["pred"] = (frame["score"] >= threshold).astype(int)
        frame["margin"] = (frame["score"] - threshold) / max(float(scale or 0.0), 1e-9)
    return frame


def best_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """F1이 가장 높아지는 임곗값 (점수 ≥ 임곗값이면 이상)."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if len(thresholds) == 0:
        return float(np.median(scores))
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def classification(labels, preds) -> dict:
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {"precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def fold_metrics(frame: pd.DataFrame) -> dict:
    return {**classification(frame.label, frame.pred), "pr_auc": average_precision_score(frame.label, frame.margin)}


def summarize(folds: pd.DataFrame, predictions: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """fold 평균 ± 표준편차와 OOF(5개 test fold를 합친) 지표를 한 표로."""
    fold_summary = folds.groupby(keys, as_index=False, sort=False).agg(
        precision_mean=("precision", "mean"), precision_std=("precision", "std"),
        recall_mean=("recall", "mean"), recall_std=("recall", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
    )
    rows = []
    for values, frame in predictions.groupby(keys, sort=False):
        values = values if isinstance(values, tuple) else (values,)
        metrics = classification(frame.label, frame.pred)
        rows.append({**dict(zip(keys, values)),
                     "normal_segments": int((frame.label == 0).sum()), "anomaly_segments": int((frame.label == 1).sum()),
                     "tp": metrics["tp"], "fn": metrics["fn"], "fp": metrics["fp"], "tn": metrics["tn"],
                     "oof_precision": metrics["precision"], "oof_recall": metrics["recall"], "oof_f1": metrics["f1"],
                     "oof_pr_auc": average_precision_score(frame.label, frame.margin)})
    return fold_summary.merge(pd.DataFrame(rows), on=keys, how="left")


def pick_best(summary: pd.DataFrame) -> pd.Series:
    """OOF F1 → PR-AUC → Recall 순으로 가장 좋은 입력."""
    return summary.sort_values(["oof_f1", "oof_pr_auc", "oof_recall"], ascending=False).iloc[0]
