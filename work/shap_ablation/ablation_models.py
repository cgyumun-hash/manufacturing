"""SHAP · ablation 실험에서 쓴 모델 (Tabular 7개 · Sequence 2개). 3장 모델과 하이퍼파라미터가 조금 다르다."""
import copy

import lightgbm as lgb
import numpy as np
import torch
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

from common.pipeline import seed_all

TEMPORAL = {"LSTM-Autoencoder", "1D-CNN"}


class TemporalCNN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(channels, 24, kernel_size=3, padding=1), nn.ReLU(),
                                 nn.Conv1d(24, 16, kernel_size=3, padding=1), nn.ReLU(),
                                 nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(16, 1))

    def forward(self, x):
        return self.net(x.transpose(1, 2)).squeeze(1)


class TemporalAE(nn.Module):
    def __init__(self, channels, hidden=24, latent=10):
        super().__init__()
        self.encoder = nn.LSTM(channels, hidden, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent)
        self.from_latent = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.output = nn.Linear(hidden, channels)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        latent = torch.relu(self.to_latent(hidden[-1]))
        repeated = torch.relu(self.from_latent(latent)).unsqueeze(1).repeat(1, x.shape[1], 1)
        decoded, _ = self.decoder(repeated)
        return self.output(decoded)


def fit_temporal(name, values, labels, seed):
    seed_all(seed)
    channels = values["train"].shape[-1]
    if name == "1D-CNN":
        model = TemporalCNN(channels)
        ratio = float((labels["train"] == 0).sum() / max((labels["train"] == 1).sum(), 1))
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(ratio, dtype=torch.float32))
        train_x = torch.tensor(values["train"], dtype=torch.float32)
        train_y = torch.tensor(labels["train"], dtype=torch.float32)
        valid_x = torch.tensor(values["valid"], dtype=torch.float32)
        valid_y = torch.tensor(labels["valid"], dtype=torch.float32)
    else:
        model = TemporalAE(channels)
        loss_fn = nn.MSELoss()
        train_x = train_y = torch.tensor(values["train"][labels["train"] == 0], dtype=torch.float32)
        valid_x = valid_y = torch.tensor(values["valid"][labels["valid"] == 0], dtype=torch.float32)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=256, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    best_state, best_loss, stale = None, float("inf"), 0
    for _ in range(20):
        model.train()
        for batch, target in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch), target)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_loss = float(loss_fn(model(valid_x), valid_y))
        if valid_loss < best_loss - 1e-6:
            best_loss, best_state, stale = valid_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= 4:
                break
    model.load_state_dict(best_state)
    return model


def fit_tabular(name, values, labels, seed):
    x, y = values["train"], labels["train"]
    normal = x[y == 0]
    if name == "LightGBM":
        return lgb.LGBMClassifier(n_estimators=160, learning_rate=0.05, num_leaves=15, max_depth=5, min_child_samples=10,
                                  class_weight="balanced", random_state=seed, n_jobs=8, verbosity=-1).fit(x, y)
    if name == "XGBoost":
        ratio = float((y == 0).sum() / max((y == 1).sum(), 1))
        return XGBClassifier(n_estimators=160, max_depth=3, learning_rate=0.05, subsample=0.85, colsample_bytree=0.9,
                             min_child_weight=2, reg_lambda=1.0, scale_pos_weight=ratio, objective="binary:logistic",
                             eval_metric="aucpr", random_state=seed, n_jobs=8).fit(x, y)
    if name == "Logistic Regression":
        return LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed).fit(x, y)
    if name == "One-Class SVM":
        return OneClassSVM(kernel="rbf", gamma="scale", nu=0.05, cache_size=1024).fit(normal)
    if name == "Mahalanobis distance":
        return LedoitWolf().fit(normal)
    if name == "Isolation Forest":
        return IsolationForest(n_estimators=240, contamination="auto", random_state=seed, n_jobs=8).fit(normal)
    if name == "PCA reconstruction":
        return PCA(n_components=0.95, svd_solver="full").fit(normal)
    raise ValueError(name)


def fit(name, values, labels, seed):
    return fit_temporal(name, values, labels, seed) if name in TEMPORAL else fit_tabular(name, values, labels, seed)


def score(name, model, x):
    if name in {"LightGBM", "XGBoost", "Logistic Regression"}:
        return model.predict_proba(x)[:, 1]
    if name in {"One-Class SVM", "Isolation Forest"}:
        return -model.decision_function(x)
    if name == "Mahalanobis distance":
        return model.mahalanobis(x)
    if name == "PCA reconstruction":
        return np.mean((x - model.inverse_transform(model.transform(x))) ** 2, axis=1)
    tensor = torch.tensor(x, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        if name == "1D-CNN":
            return torch.sigmoid(model(tensor)).numpy()
        return torch.mean((model(tensor) - tensor) ** 2, dim=(1, 2)).numpy()
