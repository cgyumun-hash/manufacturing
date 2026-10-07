"""One-Class SVM — Tabular · 정상 전용.

train 정상 window만으로 '정상이 사는 영역'의 경계를 그린다 (RBF 커널, nu = 0.05).
경계 밖으로 멀수록 이상 점수가 크다. 이상 데이터가 거의 없어도 쓸 수 있다.
"""
import joblib
from sklearn.svm import OneClassSVM

NAME = "One-Class SVM"
FAMILY, LEARNING = "tabular", "normal-only"
FAMILY_KO, LEARNING_KO = "Tabular(표형)", "정상 전용 학습"
MODEL_INDEX = 3


def fit(values, labels, seed):
    return OneClassSVM(kernel="rbf", gamma="scale", nu=0.05, cache_size=1024).fit(
        values["train"][labels["train"] == 0])


def score(model, x):
    return -model.decision_function(x)


def save(model, folder):
    joblib.dump(model, folder / "model.joblib")


def load(folder):
    return joblib.load(folder / "model.joblib")
