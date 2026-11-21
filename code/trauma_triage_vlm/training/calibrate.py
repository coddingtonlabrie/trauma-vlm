from __future__ import annotations


from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from trauma_triage_vlm.models.adapter import CalibrationHead


@dataclass(frozen=True)
class CalibrationResult:
    losses: tuple[float, ...]


def fit_calibration(features: torch.Tensor, labels: torch.Tensor, epochs: int, batch_size: int, lr: float) -> tuple[CalibrationHead, CalibrationResult]:
    model = CalibrationHead(features.shape[1], int(labels.max().item()) + 1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(features, labels), batch_size=batch_size, shuffle=True)
    losses: list[float] = []
    for _ in range(epochs):
        total = 0.0
        seen = 0
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = loss_fn(out.logits, yb)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * xb.shape[0]
            seen += xb.shape[0]
        losses.append(total / max(seen, 1))
    return model, CalibrationResult(losses=tuple(losses))
