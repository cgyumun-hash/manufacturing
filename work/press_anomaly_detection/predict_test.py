"""3단계 — test 데이터 예측과 성능 평가.

각 fold의 test segment(학습·기준값 결정에 쓰지 않은 segment)를 예측한다. 5개 fold를 합치면
정상 452 · 이상 13 segment가 모두 정확히 한 번씩 test 예측을 받는다 (OOF).

  기존 판정    : segment 평균 점수 ≥ T*
  최종 판정    : 기존 판정 OR (window 점수 ≥ T_w 가 K번 연속)  ← 제출 모델

결과
  results/test_predictions.csv   segment별 최종 예측 (제출용 예측 결과)
  results/test_window_scores.csv window별 점수 · 경고 · 연속 횟수
  results/performance.csv, results/performance.md   성능

실행: python predict_test.py
"""
import pandas as pd

from src import markov, model
from src.config import RESULT_DIR, setup_console
from src.data import eligible_segments
from src.evaluation import metrics, pr_auc, segment_scores
from valid import scored
from src.windows import make_splits, make_windows


def main():
    setup_console()
    segments = eligible_segments()
    info = {s["group"]: s for s in segments}
    capped, full = make_windows(segments, capped=True), make_windows(segments, capped=False)
    th = pd.read_csv(RESULT_DIR / "valid_thresholds.csv").set_index("fold")
    seg_frames, win_frames, fold_rows = [], [], []
    for split in make_splits(segments):
        fold = split["fold"]
        p = th.loc[fold]
        scaler, lgbm = model.load(fold)
        test = scored(capped, split["test"], scaler, lgbm)
        seg = segment_scores(test.group, test.label, test.score)
        seg["pred_segment_mean"] = (seg.score >= p.T_star).astype(int)
        seg["margin"] = (seg.score - p.T_star) / max(float(p.valid_score_std), 1e-9)
        run_seg, run_win = markov.run_alarm(scored(full, split["test"], scaler, lgbm), p.T_w, int(p.K))
        seg = seg.merge(run_seg, on="group")
        seg["pred_final"] = ((seg.pred_segment_mean == 1) | (seg.run_pred == 1)).astype(int)
        seg = seg.assign(fold=fold, T_star=p.T_star, T_w=p.T_w, K=int(p.K))
        seg_frames.append(seg)
        win_frames.append(run_win.assign(fold=fold, T_w=p.T_w, K=int(p.K)))
        fold_rows.append({"fold": fold, **{f"segment_mean_{k}": v for k, v in metrics(seg.label, seg.pred_segment_mean).items()},
                          **{f"final_{k}": v for k, v in metrics(seg.label, seg.pred_final).items()}})

    seg = pd.concat(seg_frames, ignore_index=True)
    out = pd.DataFrame({
        "fold": seg.fold, "segment": seg.group,
        "start_time": seg.group.map(lambda g: info[g]["start_time"]), "end_time": seg.group.map(lambda g: info[g]["end_time"]),
        "samples": seg.group.map(lambda g: info[g]["samples"]),
        "label": seg.label, "segment_mean_score": seg.score.round(6), "T_star": seg.T_star.round(6),
        "pred_segment_mean": seg.pred_segment_mean, "T_w": seg.T_w.round(6), "K": seg.K, "pred_consecutive": seg.run_pred,
        "first_alarm_seconds": seg.first_alarm_seconds, "prediction": seg.pred_final,
        "correct": (seg.pred_final == seg.label).astype(int)})
    out.to_csv(RESULT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")
    windows = pd.concat(win_frames, ignore_index=True)
    windows["window_end_seconds"] = ((windows.start + 9) * 0.1).round(1)
    windows[["fold", "group", "label", "start", "window_end_seconds", "score", "T_w", "warning", "run_count", "K"]].rename(
        columns={"group": "segment"}).to_csv(RESULT_DIR / "test_window_scores.csv", index=False, encoding="utf-8-sig")

    base, final = metrics(seg.label, seg.pred_segment_mean), metrics(seg.label, seg.pred_final)
    perf = pd.DataFrame([{"method": "segment_mean (T*)", **base, "pr_auc": pr_auc(seg.label, seg.margin)},
                         {"method": "segment_mean OR Markov consecutive (final)", **final, "pr_auc": None}])
    perf.to_csv(RESULT_DIR / "performance.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(RESULT_DIR / "performance_by_fold.csv", index=False, encoding="utf-8-sig")
    lines = ["# 성능 (5-fold Group CV, test segment 예측을 모두 합친 OOF)", "",
             f"- test segment: 정상 {(seg.label == 0).sum()} · 이상 {(seg.label == 1).sum()}", "",
             "| 판정 방식 | Precision | Recall | F1 | TP | FN | FP | TN |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, m in (("segment 평균 ≥ T*", base), ("**최종: segment 평균 OR 마르코프 연속경고**", final)):
        lines.append(f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['tp']} | {m['fn']} | {m['fp']} | {m['tn']} |")
    lines += ["", f"segment 평균 판정의 PR-AUC: {perf.pr_auc.iloc[0]:.3f}"]
    (RESULT_DIR / "performance.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"segment 평균만      : Precision {base['precision']:.3f}  Recall {base['recall']:.3f}  F1 {base['f1']:.3f}")
    print(f"최종 (OR 연속경고)  : Precision {final['precision']:.3f}  Recall {final['recall']:.3f}  F1 {final['f1']:.3f}")
    print("예측 결과 → results/test_predictions.csv", flush=True)


if __name__ == "__main__":
    main()
