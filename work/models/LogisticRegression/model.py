"""Logistic Regression — Tabular · 정상·이상 지도학습.

입력 변수들의 가중합으로 '이상일 확률'을 계산하는 직선 경계 모델.
어떤 변수가 이상 쪽으로 점수를 올리는지 계수로 바로 읽을 수 있다.
"""
import joblib
from sklearn.linear_model import LogisticRegression

NAME = "Logistic Regression"
FAMILY, LEARNING = "tabular", "supervised"
FAMILY_KO, LEARNING_KO = "Tabular(표형)", "정상·이상 지도학습"
MODEL_INDEX = 0  # 시드 계산용 (8개 모델 공통 순서)


def fit(values, labels, seed):
    return LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed).fit(
        values["train"], labels["train"])


def score(model, x):
    return model.predict_proba(x)[:, 1]


def save(model, folder):
    joblib.dump(model, folder / "model.joblib")


def load(folder):
    return joblib.load(folder / "model.joblib")
