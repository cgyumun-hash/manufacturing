"""모델 폴더의 train.py · valid.py · test.py · markov.py가 부르는 공통 흐름.

모델 폴더 구조
    models/<모델>/model.py      모델 정의 · 하이퍼파라미터 (fit / score / save / load)
    models/<모델>/train.py      9개 입력 × 5 fold 학습 → artifacts/<입력>/fold<k>/
    models/<모델>/valid.py      validation 점수 → 임곗값 T* 선택 → results/valid_thresholds.csv
    models/<모델>/test.py       T*를 test에 그대로 적용 → results/performance.csv (성능표)
    models/<모델>/markov.py     5장: 최고 입력 하나에 보완 마르코프 연속경고 → markov/
"""
from __future__ import annotations

import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import evaluation as ev
from . import markov as mk
from .paths import save_table
from .preprocess import InputPreprocessor
from .windows import INPUT_BY_KEY, INPUTS, SEED, make_windows, prepare, select, stride_of

PARTS = ("train", "valid", "test")


def seed_all(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))


def model_seed(M, fold: int, length: int) -> int:
    return SEED + fold * 1000 + length * 20 + M.MODEL_INDEX


def artifact_dir(model_dir: Path, key: str, fold: int) -> Path:
    return Path(model_dir) / "artifacts" / key / f"fold{fold}"


def load_fold(M, model_dir: Path, key: str, fold: int):
    folder = artifact_dir(model_dir, key, fold)
    return joblib.load(folder / "preprocessor.joblib"), M.load(folder)


def _windows_by_length(eligible, capped=True):
    return {length: make_windows(eligible, length, capped) for length in sorted({i["length"] for i in INPUTS})}


# ------------------------------------------------------------------ train
def train(M, model_dir: Path) -> pd.DataFrame:
    """fold마다 train window로 전처리기와 모델을 학습해 저장한다.
    (Sequence 지도학습·LSTM-AE는 validation 손실로 조기 종료만 판단한다.)"""
    model_dir = Path(model_dir)
    seed_all(SEED)
    eligible, _, _, splits = prepare()
    windows = _windows_by_length(eligible)
    rows = []
    for split in splits:
        fold = split["fold"]
        for item in INPUTS:
            data = {part: select(windows[item["length"]], split[part]) for part in ("train", "valid")}
            pre = InputPreprocessor(item["representation"], M.FAMILY, item["length"]).fit(
                data["train"]["raw"], data["train"]["labels"])
            values = {part: pre.transform(data[part]["raw"]) for part in data}
            labels = {part: data[part]["labels"] for part in data}
            started = time.perf_counter()
            model = M.fit(values, labels, model_seed(M, fold, item["length"]))
            elapsed = time.perf_counter() - started
            folder = artifact_dir(model_dir, item["key"], fold)
            folder.mkdir(parents=True, exist_ok=True)
            joblib.dump(pre, folder / "preprocessor.joblib")
            M.save(model, folder)
            rows.append({"fold": fold, "configuration": item["key"], "input_label": item["label"],
                         "train_normal_windows": int((labels["train"] == 0).sum()),
                         "train_anomaly_windows": int((labels["train"] == 1).sum()),
                         "feature_count": len(pre.names), "features": ",".join(pre.names),
                         "fit_seconds": round(elapsed, 3)})
        print(f"[{M.NAME}] train fold {fold}/5 완료", flush=True)
    log = pd.DataFrame(rows)
    save_table(log, model_dir / "results" / "train_log.csv")
    return log


# ------------------------------------------------------------------ valid
def valid(M, model_dir: Path) -> pd.DataFrame:
    """validation segment 평균 점수에서 F1이 최대인 임곗값 T*를 고른다."""
    model_dir = Path(model_dir)
    eligible, _, _, splits = prepare()
    windows = _windows_by_length(eligible)
    rows, segment_rows = [], []
    for split in splits:
        fold = split["fold"]
        for item in INPUTS:
            data = select(windows[item["length"]], split["valid"])
            pre, model = load_fold(M, model_dir, item["key"], fold)
            scores = M.score(model, pre.transform(data["raw"]))
            segments = ev.segment_scores(data["groups"], data["labels"], scores)
            threshold = ev.best_threshold(segments.label.to_numpy(), segments.score.to_numpy())
            scale = float(segments.score.std(ddof=0))
            pred = (segments.score >= threshold).astype(int)
            metrics = ev.classification(segments.label, pred)
            rows.append({"fold": fold, "configuration": item["key"], "input_label": item["label"],
                         "threshold": threshold, "validation_score_std": scale,
                         "valid_normal_segments": int((segments.label == 0).sum()),
                         "valid_anomaly_segments": int((segments.label == 1).sum()),
                         "valid_f1_at_threshold": metrics["f1"]})
            segment_rows.append(segments.assign(fold=fold, configuration=item["key"], input_label=item["label"]))
    table = pd.DataFrame(rows)
    save_table(table, model_dir / "results" / "valid_thresholds.csv", float_format="%.17g")
    save_table(pd.concat(segment_rows, ignore_index=True), model_dir / "results" / "valid_segment_scores.csv")
    print(f"[{M.NAME}] valid 완료 — 임곗값 {len(table)}개", flush=True)
    return table


# ------------------------------------------------------------------ test
def test(M, model_dir: Path) -> pd.DataFrame:
    """valid에서 고른 T*를 test에 그대로 적용하고, 5개 fold를 합쳐 성능표를 만든다."""
    model_dir = Path(model_dir)
    eligible, _, _, splits = prepare()
    windows = _windows_by_length(eligible)
    thresholds = pd.read_csv(model_dir / "results" / "valid_thresholds.csv").set_index(["fold", "configuration"])
    fold_rows, prediction_rows = [], []
    for split in splits:
        fold = split["fold"]
        for item in INPUTS:
            data = select(windows[item["length"]], split["test"])
            pre, model = load_fold(M, model_dir, item["key"], fold)
            scores = M.score(model, pre.transform(data["raw"]))
            row = thresholds.loc[(fold, item["key"])]
            segments = ev.segment_scores(data["groups"], data["labels"], scores,
                                         float(row.threshold), float(row.validation_score_std))
            keys = {"configuration": item["key"], "input_label": item["label"], "input_kind": item["kind"],
                    "window_samples": item["length"]}
            fold_rows.append({**keys, "fold": fold, "threshold": float(row.threshold), **ev.fold_metrics(segments)})
            prediction_rows.append(segments.assign(**keys, fold=fold, threshold=float(row.threshold)))
    folds = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    summary = ev.summarize(folds, predictions, ["configuration", "input_label", "input_kind", "window_samples"])
    summary.insert(0, "model", M.NAME)
    results = model_dir / "results"
    save_table(folds, results / "fold_metrics.csv")
    save_table(predictions, results / "test_predictions.csv")
    save_table(summary, results / "performance.csv")
    write_performance_md(M, summary, results / "성능표.md")
    best = ev.pick_best(summary)
    print(f"[{M.NAME}] test 완료 — 최고 입력 {best.input_label}, OOF F1 {best.oof_f1:.3f}", flush=True)
    return summary


def write_performance_md(M, summary: pd.DataFrame, path: Path) -> None:
    best = ev.pick_best(summary)
    lines = [f"# {M.NAME} 성능표", "",
             f"- 모델 구분: {M.FAMILY_KO} · {M.LEARNING_KO}",
             "- 평가: 5-fold Group CV, segment 평균 점수, 임곗값은 validation에서 선택",
             "- 사용 segment: 2초 입력이 가능한 정상 452개 · 이상 13개 (모든 입력 공통)",
             f"- **최고 입력: {best.input_label} (OOF F1 {best.oof_f1:.3f}, PR-AUC {best.oof_pr_auc:.3f})**", "",
             "| 입력 | OOF Precision | OOF Recall | OOF F1 | PR-AUC | fold F1 평균 ± 표준편차 | TP | FN | FP | TN |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, row in summary.iterrows():
        mark = "**" if row.configuration == best.configuration else ""
        lines.append(f"| {mark}{row.input_label}{mark} | {row.oof_precision:.3f} | {row.oof_recall:.3f} | "
                     f"{mark}{row.oof_f1:.3f}{mark} | {row.oof_pr_auc:.3f} | {row.f1_mean:.3f} ± {row.f1_std:.3f} | "
                     f"{row.tp} | {row.fn} | {row.fp} | {row.tn} |")
    lines += ["", "OOF = 5개 test fold의 segment 예측을 한 번씩 모두 합쳐 계산한 값."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ markov (5장)
def best_configuration(model_dir: Path) -> dict:
    summary = pd.read_csv(Path(model_dir) / "results" / "performance.csv")
    return INPUT_BY_KEY[ev.pick_best(summary).configuration]


def markov(M, model_dir: Path) -> pd.DataFrame:
    """3장에서 가장 좋았던 입력 하나만 골라, segment 평균 OR 마르코프 연속경고로 다시 판정한다.
    학습된 모델과 T*는 train.py · valid.py 결과를 그대로 쓴다 (재학습 없음)."""
    model_dir = Path(model_dir)
    item = best_configuration(model_dir)
    length = item["length"]
    stride = stride_of(length)
    decisions_per_hour = 3600.0 / (stride * 0.1)
    eligible, _, _, splits = prepare()
    capped = make_windows(eligible, length, capped=True)
    full = make_windows(eligible, length, capped=False)
    thresholds = pd.read_csv(model_dir / "results" / "valid_thresholds.csv").set_index(["fold", "configuration"])
    fold_rows, segment_rows, window_rows = [], [], []
    for split in splits:
        fold = split["fold"]
        pre, model = load_fold(M, model_dir, item["key"], fold)
        t_star = float(thresholds.loc[(fold, item["key"])].threshold)

        def scored(windows, names):
            data = select(windows, names)
            return pd.DataFrame({"group": data["groups"], "label": data["labels"], "start": data["starts"],
                                 "score": M.score(model, pre.transform(data["raw"]))})

        capped_test = scored(capped, split["test"])
        full_valid, full_test = scored(full, split["valid"]), scored(full, split["test"])

        baseline = ev.segment_scores(capped_test.group, capped_test.label, capped_test.score, t_star, 1.0)
        baseline = baseline.rename(columns={"score": "baseline_score", "pred": "baseline_pred"}).drop(columns="margin")

        t_w = mk.warning_threshold(full_valid.loc[full_valid.label == 0, "score"].to_numpy())
        warn_prob = float((full_valid.loc[full_valid.label == 0, "score"] >= t_w).mean())
        trans = mk.transition_statistics(full_valid, t_w)
        k_ind = mk.independent_k(warn_prob, decisions_per_hour)
        k_m, expected = mk.markov_k(decisions_per_hour, trans["pi0"], trans["start_probability_r"],
                                    trans["continuation_probability_q"])
        k_up, expected_up = mk.markov_k(decisions_per_hour, trans["pi0_upper_95"], trans["r_upper_95"],
                                        trans["q_upper_95"])

        result = baseline.copy()
        for method, k in (("independent", k_ind), ("markov", k_m), ("upper", k_up)):
            seg, win = mk.run_alarm(full_test, t_w, k)
            lookup = seg.set_index("group")
            result[f"{method}_run_pred"] = result.group.map(lookup.run_pred).astype(int)
            result[f"{method}_or_pred"] = ((result.baseline_pred == 1) | (result[f"{method}_run_pred"] == 1)).astype(int)
            if method == "markov":
                result["markov_first_alarm_window"] = result.group.map(lookup.first_alarm_window)
                window_rows.append(win.assign(fold=fold, warning_threshold=t_w, markov_k=k_m, upper_k=k_up))
        result = result.assign(fold=fold, baseline_threshold=t_star, warning_threshold=t_w,
                               independent_k=k_ind, markov_k=k_m, upper_k=k_up)
        segment_rows.append(result)
        row = {"fold": fold, "configuration": item["key"], "input_label": item["label"], "window_samples": length,
               "stride_samples": stride, "decision_interval_seconds": stride * 0.1,
               "decisions_per_hour": decisions_per_hour, "baseline_threshold": t_star, "warning_threshold": t_w,
               "warning_probability": warn_prob, "independent_k": k_ind, "markov_k": k_m, "upper_k": k_up,
               "markov_expected_false_runs_per_hour": expected, "upper_expected_false_runs_per_hour": expected_up,
               **trans}
        for name, col in (("baseline", "baseline_pred"), ("independent", "independent_or_pred"),
                          ("markov", "markov_or_pred"), ("upper", "upper_or_pred")):
            row.update({f"{name}_{key}": value for key, value in ev.classification(result.label, result[col]).items()})
        fold_rows.append(row)
        print(f"[{M.NAME}] markov fold {fold}: T_w={t_w:.5f}, r={trans['start_probability_r']:.4f}, "
              f"q={trans['continuation_probability_q']:.4f}, K={k_m} (상한 {k_up})", flush=True)

    folds = pd.DataFrame(fold_rows)
    segments = pd.concat(segment_rows, ignore_index=True)
    windows_out = pd.concat(window_rows, ignore_index=True)
    summary = {"model": M.NAME, "configuration": item["key"], "input_label": item["label"]}
    for name, col in (("baseline", "baseline_pred"), ("independent", "independent_or_pred"),
                      ("markov", "markov_or_pred"), ("upper", "upper_or_pred")):
        summary.update({f"{name}_{key}": value for key, value in ev.classification(segments.label, segments[col]).items()})
    summary.update({
        "markov_k_min": int(folds.markov_k.min()), "markov_k_max": int(folds.markov_k.max()),
        "upper_k_min": int(folds.upper_k.min()), "upper_k_max": int(folds.upper_k.max()),
        "recovered_fn": int(((segments.label == 1) & (segments.baseline_pred == 0) & (segments.markov_or_pred == 1)).sum()),
        "new_fp": int(((segments.label == 0) & (segments.baseline_pred == 0) & (segments.markov_or_pred == 1)).sum()),
    })
    summary = pd.DataFrame([summary])
    out = model_dir / "markov"
    save_table(folds, out / "markov_folds.csv")
    save_table(segments, out / "markov_segments.csv")
    save_table(windows_out, out / "markov_windows.csv")
    save_table(summary, out / "markov_summary.csv")
    write_markov_md(M, item, folds, summary.iloc[0], out / "마르코프_결과.md")
    s = summary.iloc[0]
    print(f"[{M.NAME}] markov 완료 — {item['label']}: F1 {s.baseline_f1:.3f} → {s.markov_f1:.3f} (상한 K {s.upper_f1:.3f})",
          flush=True)
    return summary


def write_markov_md(M, item, folds: pd.DataFrame, s: pd.Series, path: Path) -> None:
    lines = [f"# {M.NAME} — segment 평균 OR 마르코프 연속경고 (5장)", "",
             f"- 적용 입력: **{item['label']}** (3장 성능표에서 OOF F1이 가장 높은 입력)",
             f"- 판정 간격: {stride_of(item['length'])}시점 = {stride_of(item['length']) * 0.1:.1f}초마다 window 점수 계산",
             "- 판정: segment 평균 ≥ T* **또는** window 점수 ≥ T_w 가 K번 연속", "",
             "| 판정 방식 | Precision | Recall | F1 | TP | FN | FP | TN |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, label in (("baseline", "기존 (segment 평균만)"), ("independent", "OR 독립 K"),
                        ("markov", "**OR 마르코프 K (제안)**"), ("upper", "OR 마르코프 95% 상한 K")):
        lines.append(f"| {label} | {s[f'{name}_precision']:.3f} | {s[f'{name}_recall']:.3f} | {s[f'{name}_f1']:.3f} | "
                     f"{s[f'{name}_tp']} | {s[f'{name}_fn']} | {s[f'{name}_fp']} | {s[f'{name}_tn']} |")
    lines += ["", f"- 기존에 놓친 이상을 새로 찾음: {s.recovered_fn}개 · 새로 생긴 헛경보: {s.new_fp}개", "",
              "## fold별 기준값", "",
              "| fold | T* | T_w | n00 | n01 | n10 | n11 | r | q | K (마르코프) | K (95% 상한) | K (독립) |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in folds.iterrows():
        lines.append(f"| {r.fold} | {r.baseline_threshold:.4f} | {r.warning_threshold:.5f} | {r.n00} | {r.n01} | {r.n10} | "
                     f"{r.n11} | {r.start_probability_r:.4f} | {r.continuation_probability_q:.4f} | {r.markov_k} | "
                     f"{r.upper_k} | {r.independent_k} |")
    lines += ["", "주의: 정상 validation의 '경고 → 경고' 전이가 매우 적어 K는 fold마다 크게 달라질 수 있다. "
              "새 날짜의 정상 데이터에서 다시 계산해야 한다."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
