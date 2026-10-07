"""9개 입력(단일시점 · 원데이터 0.5~2.0초 · 파생 0.5~2.0초) × 5 fold를 학습해 artifacts/에 저장한다.

실행: python work/models/CNN1D/train.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]

import model as M  # noqa: E402
from common import pipeline  # noqa: E402

if __name__ == "__main__":
    pipeline.train(M, HERE)
