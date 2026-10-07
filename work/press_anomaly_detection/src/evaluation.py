"""segment 단위 판정과 지표.

segment 점수 = segment 안 window 점수의 평균. 점수 ≥ T* 이면 이상.
T* = validation segment 점수에서 F1이 가장 높은 값 (test에는 그대로 적용).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score)


def segment_scores(groups, labels, scores) -> pd.DataFrame:
    return (pd.DataFrame({"group": groups, "label": labels, "score": scores})
            .groupby("group", sort=False).agg(label=("label", "first"), score=("score", "mean")).reset_index())


def best_threshold(labels, scores) -> float:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    if len(thresholds) == 0:
        return float(np.median(scores))
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def metrics(labels, preds) -> dict:
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {"precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0),
            "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn)}


def pr_auc(labels, margins) -> float:
    return float(average_precision_score(labels, margins))
