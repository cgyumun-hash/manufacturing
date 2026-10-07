"""4.3 SHAP — 그룹 ablation에서 fold 평균 F1이 가장 높았던 모델(Tabular · Sequence 각 1개)의 변수 기여도.

- One-Class SVM · 1D-CNN 등 트리가 아닌 모델: Monte Carlo permutation SHAP (32번 순열)
  기준 입력 = train 정상 평균(표준화 후 0). 시계열은 한 채널의 모든 시점을 한 변수로 묶는다.
- 각 fold test에서 정상 · 이상 최대 24개씩 뽑아 계산하고, 정상 평균 |SHAP|과 이상 평균 |SHAP|을 다시 평균한다
  (개수 불균형 보정 = balanced_mean_abs_shap).
SHAP은 모델이 '무엇을 썼는지'를 보여줄 뿐 고장 원인을 증명하지 않는다.

입력: results/cv_summary.csv (run_group_ablation.py 결과)
출력: results/shap_folds.csv, results/shap_summary.csv
실행: python work/shap_ablation/run_shap.py
"""
import numpy as np
import pandas as pd
import xgboost as xgb

import ablation_setup as S
import ablation_models as AM
from common.features import TABULAR_FEATURES, TEMPORAL_FEATURES, tabular_features, temporal_features
from common.paths import save_table
from common.pipeline import seed_all


def balanced_sample(y, maximum, seed):
    rng = np.random.default_rng(seed)
    out = []
    for label in (0, 1):
        index = np.flatnonzero(y == label)
        if len(index) > maximum:
            index = rng.choice(index, maximum, replace=False)
        out.extend(index.tolist())
    return np.asarray(out, dtype=int)


def sampled_shapley(name, model, values, temporal, seed, permutations=32):
    rng = np.random.default_rng(seed)
    count = values.shape[-1]
    baseline = np.zeros_like(values)
    shap = np.zeros((len(values), count), dtype=np.float64)
    for _ in range(permutations):
        current = baseline.copy()
        previous = AM.score(name, model, current)
        for feature in rng.permutation(count):
            if temporal:
                current[:, :, feature] = values[:, :, feature]
            else:
                current[:, feature] = values[:, feature]
            now = AM.score(name, model, current)
            shap[:, feature] += now - previous
            previous = now
    return shap / permutations


def tree_shap(name, model, values):
    if name == "LightGBM":
        return np.asarray(model.predict(values, pred_contrib=True))[:, :-1]
    return model.get_booster().predict(xgb.DMatrix(values, feature_names=TABULAR_FEATURES), pred_contribs=True)[:, :-1]


def summarize(folds):
    cls = folds.groupby(["family", "model", "window_seconds", "method", "class", "feature"], as_index=False).agg(
        mean_abs_shap=("mean_abs_shap", "mean"), mean_signed_shap=("mean_signed_shap", "mean"))
    pivot = cls.pivot_table(index=["family", "model", "window_seconds", "method", "feature"], columns="class",
                            values=["mean_abs_shap", "mean_signed_shap"]).reset_index()
    pivot.columns = ["_".join(str(i) for i in c if str(i)) if isinstance(c, tuple) else str(c) for c in pivot.columns]
    pivot["balanced_mean_abs_shap"] = (pivot.mean_abs_shap_normal + pivot.mean_abs_shap_anomaly) / 2
    pivot["rank"] = pivot.groupby("family").balanced_mean_abs_shap.rank(method="first", ascending=False).astype(int)
    return pivot.sort_values(["family", "rank"])


def main():
    seed_all(S.SEED)
    summary = pd.read_csv(S.RESULTS / "cv_summary.csv")
    winners = {f: summary[(summary.family == f) & (summary.experiment == "full")].sort_values(
        ["f1_mean", "pr_auc_mean", "recall_mean"], ascending=False).iloc[0] for f in ("tabular", "temporal")}
    by_length, names, labels = S.prepare_windows()
    splits = S.make_splits(names, labels)
    rows = []
    for family, w in winners.items():
        name, length, temporal = str(w.model), int(w.window_samples), family == "temporal"
        feat = TEMPORAL_FEATURES if temporal else TABULAR_FEATURES
        for split in splits:
            fold = split["fold"]
            raw_split, y_split, _ = S.split_raw(*by_length[length], split)
            unscaled = temporal_features(raw_split, y_split) if temporal else tabular_features(raw_split, y_split)
            values = S.scale_features(unscaled, y_split, temporal)
            model = AM.fit(name, values, y_split, S.configuration_seed(fold, length, family, name))
            index = balanced_sample(y_split["test"], 24, S.SEED + fold)
            sample, sample_y = values["test"][index], y_split["test"][index]
            if not temporal and name in {"LightGBM", "XGBoost"}:
                attribution, method = tree_shap(name, model, sample), "exact TreeSHAP"
            else:
                attribution = sampled_shapley(name, model, sample, temporal, S.SEED + 900000 + fold)
                method = "Monte Carlo permutation SHAP (32 permutations)"
            for i, feature in enumerate(feat):
                for label, cls in ((0, "normal"), (1, "anomaly")):
                    mask = sample_y == label
                    rows.append({"family": family, "model": name, "window_seconds": length / 10, "fold": fold,
                                 "method": method, "class": cls, "feature": feature,
                                 "mean_abs_shap": float(np.mean(np.abs(attribution[mask, i]))),
                                 "mean_signed_shap": float(np.mean(attribution[mask, i])), "samples": int(mask.sum())})
        print(f"SHAP 완료: {family} / {name} / {length / 10:.1f}초", flush=True)
    folds = pd.DataFrame(rows)
    save_table(folds, S.RESULTS / "shap_folds.csv")
    result = summarize(folds)
    save_table(result, S.RESULTS / "shap_summary.csv")
    print(result[result["rank"] <= 5][["family", "model", "feature", "balanced_mean_abs_shap", "rank"]].to_string(index=False))


if __name__ == "__main__":
    main()
