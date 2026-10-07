"""Mahalanobis distance — Tabular · 정상 전용.

train 정상 window의 평균과 공분산(LedoitWolf 수축 추정)을 구하고,
새 window가 그 '정상 구름'에서 얼마나 멀리 떨어졌는지(변수 간 상관까지 고려한 거리)를 이상 점수로 쓴다.
"""
import joblib
from sklearn.covariance import LedoitWolf

NAME = "Mahalanobis distance"
FAMILY, LEARNING = "tabular", "normal-only"
FAMILY_KO, LEARNING_KO = "Tabular(표형)", "정상 전용 학습"
MODEL_INDEX = 2


def fit(values, labels, seed):
    return LedoitWolf().fit(values["train"][labels["train"] == 0])


def score(model, x):
    return model.mahalanobis(x)


def save(model, folder):
    joblib.dump(model, folder / "model.joblib")


def load(folder):
    return joblib.load(folder / "model.joblib")
