"""T*를 test segment에 그대로 적용하고 5개 fold를 합쳐 results/performance.csv · 성능표.md를 만든다.

실행: python work/models/LightGBM/test.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]

import model as M  # noqa: E402
from common import pipeline  # noqa: E402

if __name__ == "__main__":
    pipeline.test(M, HERE)
