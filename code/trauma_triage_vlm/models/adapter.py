from __future__ import annotations


from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class AdapterOutput:
    logits: torch.Tensor
    confidence: torch.Tensor


class CalibrationHead(nn.Module):
    def __init__(self, input_dim: int, classes: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.linear = nn.Linear(input_dim, classes)
        self.temperature = nn.Parameter(torch.ones(()))

    def forward(self, features: torch.Tensor) -> AdapterOutput:
        logits = self.linear(self.norm(features)) / self.temperature.clamp_min(0.05)
        confidence = torch.softmax(logits, dim=-1).amax(dim=-1)
        return AdapterOutput(logits=logits, confidence=confidence)


class EdgeAcuityRegressor(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(features)).squeeze(-1)
