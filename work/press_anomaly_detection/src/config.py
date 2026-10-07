"""설정값: 경로, 시드, 입력 길이, LightGBM 하이퍼파라미터, 마르코프 기준."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"
NORMAL_CSV = DATA_DIR / "press_data_normal.csv"
ANOMALY_CSV = DATA_DIR / "outlier_data.csv"

SENSORS = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]
DT = 0.1                      # 기록 간격 (초)

# 데이터 분할 · window
SEED = 20261007
N_FOLDS = 5
VALID_SIZE = 0.25             # 각 fold에서 test를 뺀 나머지 중 validation 비율
MIN_SAMPLES = 20              # 2초(20시점) 이상 이어진 segment만 사용 (정상 452 · 이상 13)
WINDOW = 10                   # 입력 길이 1.0초 = 10시점 × 3센서 = 30개 값
STRIDE = WINDOW // 2          # 0.5초씩 이동 (50% 중첩)
MAX_WINDOWS_PER_SEGMENT = 10  # 학습·평가용 window는 segment당 최대 10개를 고르게 선택

# LightGBM (n_jobs=1 · deterministic: CPU 코어 수와 관계없이 같은 결과가 나오도록 고정)
MODEL_INDEX = 1               # 원 실험(8개 모델 비교)에서 LightGBM의 순번 — 시드 재현용
LGBM_PARAMS = dict(n_estimators=180, learning_rate=0.05, num_leaves=15, max_depth=5, min_child_samples=10,
                   class_weight="balanced", n_jobs=1, verbosity=-1, deterministic=True, force_col_wise=True)

# 마르코프 연속경고
NORMAL_QUANTILE = 0.99            # T_w = validation 정상 window 점수의 99% 분위수
TARGET_FALSE_RUNS_PER_HOUR = 0.01  # 시간당 기대 헛경보 연속 목표
JEFFREYS = 0.5
UPPER_PROBABILITY = 0.95


def fold_seed(fold: int) -> int:
    return SEED + fold * 1000 + WINDOW * 20 + MODEL_INDEX


def setup_console() -> None:
    """Windows 콘솔(cp949 등)에서도 한글 출력이 깨지거나 멈추지 않게 한다."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
