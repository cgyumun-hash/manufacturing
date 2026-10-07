"""LSTM-Autoencoder — Sequence · 정상 전용.

정상 window를 작은 요약(latent 10)으로 압축했다가 다시 복원하도록 학습한다.
정상과 다른 모양은 잘 복원되지 않으므로, 복원 오차(평균 제곱 오차)를 이상 점수로 쓴다.
"""
import torch
from torch import nn

from common import torch_models as tm

NAME = "LSTM-Autoencoder"
FAMILY, LEARNING = "temporal", "normal-only"
FAMILY_KO, LEARNING_KO = "Sequence(시계열)", "정상 전용 학습"
MODEL_INDEX = 6


class LSTMAutoencoder(nn.Module):
    def __init__(self, channels: int, hidden: int = 24, latent: int = 10):
        super().__init__()
        self.channels = channels
        self.encoder = nn.LSTM(channels, hidden, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent)
        self.from_latent = nn.Linear(latent, hidden)
        self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.output = nn.Linear(hidden, channels)

    def forward(self, values):
        _, (hidden, _) = self.encoder(values)
        latent = torch.relu(self.to_latent(hidden[-1]))
        repeated = torch.relu(self.from_latent(latent)).unsqueeze(1).repeat(1, values.shape[1], 1)
        decoded, _ = self.decoder(repeated)
        return self.output(decoded)


def fit(values, labels, seed):
    return tm.fit_autoencoder(LSTMAutoencoder, values, labels, seed)


def score(model, x):
    return tm.predict(model, x, "reconstruction")


def save(model, folder):
    tm.save(model, folder)


def load(folder):
    return tm.load(LSTMAutoencoder, folder)
