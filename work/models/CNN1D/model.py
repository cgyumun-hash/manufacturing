"""1D-CNN — Sequence · 정상·이상 지도학습.

시간축을 따라 3시점 길이 필터를 밀며 짧은 파형(튀는 모양·꺾임)을 찾고, 구간 전체에서 평균해 이상 확률을 낸다.
입력 모양 = (시점 수, 채널 수). 원데이터 3채널 또는 파생 12채널.
"""
from torch import nn

from common import torch_models as tm

NAME = "1D-CNN"
FAMILY, LEARNING = "temporal", "supervised"
FAMILY_KO, LEARNING_KO = "Sequence(시계열)", "정상·이상 지도학습"
MODEL_INDEX = 4


class TemporalCNN(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.network = nn.Sequential(
            nn.Conv1d(channels, 24, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(24, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(16, 1),
        )

    def forward(self, values):
        return self.network(values.transpose(1, 2)).squeeze(1)


def fit(values, labels, seed):
    return tm.fit_classifier(TemporalCNN, values, labels, seed)


def score(model, x):
    return tm.predict(model, x, "classifier")


def save(model, folder):
    tm.save(model, folder)


def load(folder):
    return tm.load(TemporalCNN, folder)
