"""GRU classifier — Sequence · 정상·이상 지도학습.

시점을 순서대로 읽으며 기억(hidden 24)을 갱신하고, 마지막 시점의 기억으로 이상 확률을 낸다.
"""
from torch import nn

from common import torch_models as tm

NAME = "GRU classifier"
FAMILY, LEARNING = "temporal", "supervised"
FAMILY_KO, LEARNING_KO = "Sequence(시계열)", "정상·이상 지도학습"
MODEL_INDEX = 5


class GRUClassifier(nn.Module):
    def __init__(self, channels: int, hidden: int = 24):
        super().__init__()
        self.channels = channels
        self.gru = nn.GRU(channels, hidden, batch_first=True)
        self.output = nn.Linear(hidden, 1)

    def forward(self, values):
        sequence, _ = self.gru(values)
        return self.output(sequence[:, -1, :]).squeeze(1)


def fit(values, labels, seed):
    return tm.fit_classifier(GRUClassifier, values, labels, seed)


def score(model, x):
    return tm.predict(model, x, "classifier")


def save(model, folder):
    tm.save(model, folder)


def load(folder):
    return tm.load(GRUClassifier, folder)
