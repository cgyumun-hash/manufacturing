"""2단계 — validation으로 기준값 결정 (test 데이터는 쓰지 않음).

fold마다
  - T*  : validation segment 평균 점수에서 F1이 가장 높은 임곗값
  - T_w : validation 정상 window 점수의 99% 분위수 (경고 기준)
  - r, q, K : validation 정상 segment 안의 경고 전이로 계산한 마르코프 연속 횟수
결과: results/valid_thresholds.csv

실행: python valid.py
"""
import pandas as pd

from src import markov, model
from src.config import DT, RESULT_DIR, STRIDE, setup_console
from src.data import eligible_segments
from src.evaluation import best_threshold, segment_scores
from src.windows import make_splits, make_windows, select

DECISIONS_PER_HOUR = 3600.0 / (STRIDE * DT)  # 0.5초마다 판정 → 시간당 7,200번


def scored(windows, names, scaler, lgbm):
    data = select(windows, names)
    return pd.DataFrame({"group": data["groups"], "label": data["labels"], "start": data["starts"],
                         "score": model.score(scaler, lgbm, data["raw"])})


def main():
    setup_console()
    segments = eligible_segments()
    capped, full = make_windows(segments, capped=True), make_windows(segments, capped=False)
    rows = []
    for split in make_splits(segments):
        fold = split["fold"]
        scaler, lgbm = model.load(fold)
        valid = scored(capped, split["valid"], scaler, lgbm)
        seg = segment_scores(valid.group, valid.label, valid.score)
        t_star = best_threshold(seg.label.to_numpy(), seg.score.to_numpy())
        full_valid = scored(full, split["valid"], scaler, lgbm)
        t_w = markov.warning_threshold(full_valid.loc[full_valid.label == 0, "score"].to_numpy())
        tr = markov.transitions(full_valid, t_w)
        k = markov.markov_k(DECISIONS_PER_HOUR, tr["pi0"], tr["r"], tr["q"])
        k_upper = markov.markov_k(DECISIONS_PER_HOUR, tr["pi0_upper95"], tr["r_upper95"], tr["q_upper95"])
        rows.append({"fold": fold, "T_star": t_star, "valid_score_std": float(seg.score.std(ddof=0)), "T_w": t_w,
                     "decisions_per_hour": DECISIONS_PER_HOUR, **tr, "K": k, "K_upper95": k_upper})
        print(f"fold {fold}: T*={t_star:.4f}, T_w={t_w:.5f}, r={tr['r']:.4f}, q={tr['q']:.4f}, K={k}", flush=True)
    pd.DataFrame(rows).to_csv(RESULT_DIR / "valid_thresholds.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
    print("기준값 저장 → results/valid_thresholds.csv", flush=True)


if __name__ == "__main__":
    main()
