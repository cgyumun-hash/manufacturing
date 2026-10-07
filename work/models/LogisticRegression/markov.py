"""5장: 성능표에서 가장 좋은 입력 하나에 'segment 평균 OR 마르코프 연속경고'를 적용해 markov/에 저장한다.

실행: python work/models/LogisticRegression/markov.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]

import model as M  # noqa: E402
from common import pipeline  # noqa: E402

if __name__ == "__main__":
    pipeline.markov(M, HERE)
