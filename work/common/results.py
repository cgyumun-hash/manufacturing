"""models/ 폴더의 결과를 읽어 오는 도우미 (3장 · 4장 그림용)."""
import pandas as pd

from .paths import MODELS as MODELS_DIR

MODEL_DIRS = {"Logistic Regression": "LogisticRegression", "LightGBM": "LightGBM", "XGBoost": "XGBoost",
              "Mahalanobis distance": "Mahalanobis", "One-Class SVM": "OneClassSVM", "1D-CNN": "CNN1D",
              "GRU classifier": "GRU", "LSTM-Autoencoder": "LSTM_AE"}
MODELS = list(MODEL_DIRS)  # Tabular 5개 → Sequence 3개 순서
SHORT = {"Mahalanobis distance": "Mahalanobis", "GRU classifier": "GRU", "LSTM-Autoencoder": "LSTM-AE"}


def performance() -> pd.DataFrame:
    """8개 모델 × 9개 입력 = 72행 성능표."""
    return pd.concat([pd.read_csv(MODELS_DIR / d / "results" / "performance.csv") for d in MODEL_DIRS.values()],
                     ignore_index=True)


def best_rows(perf: pd.DataFrame) -> pd.DataFrame:
    """모델마다 최고 입력 1개. 순위는 OOF F1 → PR-AUC → Recall, 완전 동점이면 모델 이름순."""
    return (perf.sort_values(["oof_f1", "oof_pr_auc", "oof_recall", "model"], ascending=[False, False, False, True])
            .groupby("model", sort=False).head(1).reset_index(drop=True))


def f1(perf: pd.DataFrame, model: str, label: str) -> float:
    return float(perf[(perf.model == model) & (perf.input_label == label)].oof_f1.iloc[0])
