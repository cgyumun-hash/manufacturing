"""Sequence 모델 학습 공통 루프 (1D-CNN · GRU · LSTM-Autoencoder).

최대 30 epoch, validation 손실이 5번 연속 좋아지지 않으면 멈추고 가장 좋았던 가중치를 쓴다.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .pipeline import seed_all

EPOCHS, PATIENCE, BATCH, LR, WEIGHT_DECAY = 30, 5, 256, 2e-3, 1e-4


def train_loop(model: nn.Module, loss_fn, train_x, train_y, valid_x, valid_y, seed: int) -> nn.Module:
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=BATCH, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_state, best_loss, stale = None, float("inf"), 0
    for _ in range(EPOCHS):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_loss = float(loss_fn(model(valid_x), valid_y))
        if valid_loss < best_loss - 1e-6:
            best_loss, best_state, stale = valid_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def fit_classifier(build, values: dict, labels: dict, seed: int) -> nn.Module:
    """정상·이상 지도학습. 이상이 적으므로 손실에서 이상에 (정상 수 ÷ 이상 수)배 가중치."""
    seed_all(seed)
    model = build(values["train"].shape[-1])
    ratio = float((labels["train"] == 0).sum() / max((labels["train"] == 1).sum(), 1))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(ratio, dtype=torch.float32))
    tensor = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
    return train_loop(model, loss_fn, tensor(values["train"]), tensor(labels["train"]),
                      tensor(values["valid"]), tensor(labels["valid"]), seed)


def fit_autoencoder(build, values: dict, labels: dict, seed: int) -> nn.Module:
    """정상 전용. 정상 window만 복원하도록 학습한다."""
    seed_all(seed)
    model = build(values["train"].shape[-1])
    train = torch.tensor(values["train"][labels["train"] == 0], dtype=torch.float32)
    valid = torch.tensor(values["valid"][labels["valid"] == 0], dtype=torch.float32)
    return train_loop(model, nn.MSELoss(), train, train, valid, valid, seed)


def save(model: nn.Module, folder) -> None:
    torch.save({"channels": model.channels, "state": model.state_dict()}, folder / "model.pt")


def load(build, folder) -> nn.Module:
    saved = torch.load(folder / "model.pt", weights_only=True)
    model = build(saved["channels"])
    model.load_state_dict(saved["state"])
    model.eval()
    return model


def predict(model: nn.Module, x: np.ndarray, kind: str) -> np.ndarray:
    tensor = torch.tensor(x, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        if kind == "classifier":
            return torch.sigmoid(model(tensor)).numpy()
        return torch.mean((model(tensor) - tensor) ** 2, dim=(1, 2)).numpy()
