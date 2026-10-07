"""1단계 — 학습.

5-fold Group CV의 각 fold에서 train segment의 window로 표준화 기준과 LightGBM을 학습해 models/fold<k>/에 저장한다.
segment 분할표(어느 segment가 어느 fold의 train / valid / test인지)는 results/fold_splits.csv로 저장한다.

실행: python train.py
"""
import time

import pandas as pd

from src import model
from src.config import RESULT_DIR, fold_seed, setup_console
from src.data import eligible_segments
from src.windows import make_splits, make_windows, select


def main():
    setup_console()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    segments = eligible_segments()
    info = {s["group"]: s for s in segments}
    windows = make_windows(segments, capped=True)
    rows = []
    for split in make_splits(segments):
        fold = split["fold"]
        train = select(windows, split["train"])
        started = time.perf_counter()
        scaler, lgbm = model.fit(train["raw"], train["labels"], fold_seed(fold))
        model.save(scaler, lgbm, fold)
        print(f"fold {fold}: train window 정상 {(train['labels'] == 0).sum()} · 이상 {(train['labels'] == 1).sum()}, "
              f"{time.perf_counter() - started:.1f}초", flush=True)
        for part in ("train", "valid", "test"):
            for g in split[part]:
                rows.append({"fold": fold, "part": part, "segment": g, "label": info[g]["label"],
                             "start_time": info[g]["start_time"], "end_time": info[g]["end_time"],
                             "samples": info[g]["samples"]})
    pd.DataFrame(rows).to_csv(RESULT_DIR / "fold_splits.csv", index=False, encoding="utf-8-sig")
    print("학습 완료 → models/", flush=True)


if __name__ == "__main__":
    main()
