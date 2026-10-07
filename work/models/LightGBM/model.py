"""LightGBM — Tabular · 정상·이상 지도학습.

'몇 번째 시점의 어느 센서 값이 얼마 이상인가' 같은 질문을 나무 여러 그루로 이어 붙여(부스팅) 판단한다.
원데이터 window를 펼친 값(예: 1.0초 = 10시점 × 3센서 = 30개)을 그대로 쓸 수 있다.
"""
import joblib
import lightgbm as lgb

NAME = "LightGBM"
FAMILY, LEARNING = "tabular", "supervised"
FAMILY_KO, LEARNING_KO = "Tabular(표형)", "정상·이상 지도학습"
MODEL_INDEX = 1

PARAMS = dict(n_estimators=180, learning_rate=0.05, num_leaves=15, max_depth=5, min_child_samples=10,
              class_weight="balanced", n_jobs=8, verbosity=-1)


def fit(values, labels, seed):
    return lgb.LGBMClassifier(**PARAMS, random_state=seed).fit(values["train"], labels["train"])


def score(model, x):
    return model.predict_proba(x)[:, 1]


def save(model, folder):
    joblib.dump(model, folder / "model.joblib")


def load(folder):
    return joblib.load(folder / "model.joblib")
