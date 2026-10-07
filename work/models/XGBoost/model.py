"""XGBoost — Tabular · 정상·이상 지도학습.

LightGBM과 같은 부스팅 나무 계열. 얕은 나무(깊이 3)를 많이 쌓고, 이상에 (정상 수 ÷ 이상 수)배 가중치를 준다.
"""
import joblib
from xgboost import XGBClassifier

NAME = "XGBoost"
FAMILY, LEARNING = "tabular", "supervised"
FAMILY_KO, LEARNING_KO = "Tabular(표형)", "정상·이상 지도학습"
MODEL_INDEX = 7

PARAMS = dict(n_estimators=180, max_depth=3, learning_rate=0.05, subsample=0.85, colsample_bytree=0.9,
              min_child_weight=2, reg_lambda=1.0, objective="binary:logistic", eval_metric="aucpr", n_jobs=8)


def fit(values, labels, seed):
    y = labels["train"]
    ratio = float((y == 0).sum() / max((y == 1).sum(), 1))
    return XGBClassifier(**PARAMS, scale_pos_weight=ratio, random_state=seed).fit(values["train"], y)


def score(model, x):
    return model.predict_proba(x)[:, 1]


def save(model, folder):
    joblib.dump(model, folder / "model.joblib")


def load(folder):
    return joblib.load(folder / "model.joblib")
