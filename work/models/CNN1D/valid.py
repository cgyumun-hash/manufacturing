"""validation segment 평균 점수에서 F1이 가장 높은 임곗값 T*를 골라 results/valid_thresholds.csv에 저장한다.

실행: python work/models/CNN1D/valid.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]

import model as M  # noqa: E402
from common import pipeline  # noqa: E402

if __name__ == "__main__":
    pipeline.valid(M, HERE)
