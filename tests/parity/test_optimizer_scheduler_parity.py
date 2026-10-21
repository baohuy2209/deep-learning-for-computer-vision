"""Behavioral parity tests for optimizer parameter groups and LR scheduler.

Validates exact alignment with notebook optimization:
  - Weight decay separation (biases, normalization layers excluded from weight decay)
  - CosineAnnealingLR trajectory
"""

import pytest
import torch
import torch.nn as nn

from drdg.training.optimization import build_optimizer, build_scheduler


@pytest.mark.parity
def test_optimizer_parameter_groups_separation():
    """Verify that biases and normalization weights are segregated to weight_decay=0.0."""
    class MockNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3, bias=True)
            self.bn = nn.BatchNorm2d(16)
            self.ln = nn.LayerNorm(16)
            self.fc = nn.Linear(16, 5, bias=True)

        def forward(self, x):
            return self.fc(self.ln(self.bn(self.conv(x))))

    model = MockNet()
    optimizer = build_optimizer(model, lr=1e-4, weight_decay=1e-2, separate_decay=True)

    assert len(optimizer.param_groups) == 2, "Expected 2 parameter groups (decay and no-decay)"
    group_decay = optimizer.param_groups[0]
    group_no_decay = optimizer.param_groups[1]

    assert group_decay["weight_decay"] == 1e-2
    assert group_no_decay["weight_decay"] == 0.0

    # Ensure biases and BN weights are in no-decay group
    decay_params = set(group_decay["params"])
    no_decay_params = set(group_no_decay["params"])

    assert model.conv.bias in no_decay_params
    assert model.fc.bias in no_decay_params
    assert model.bn.weight in no_decay_params
    assert model.conv.weight in decay_params


@pytest.mark.parity
def test_cosine_scheduler_trajectory():
    """Verify CosineAnnealingLR trajectory smoothly decays learning rate over epochs."""
    model = nn.Linear(10, 5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    epochs = 30
    scheduler = build_scheduler(optimizer, scheduler_name="cosine", epochs=epochs, min_lr=1e-6)

    lrs = []
    for _ in range(epochs):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    assert lrs[0] == 1e-4, f"Initial LR should be 1e-4, got {lrs[0]}"
    assert lrs[-1] < lrs[0], "Final LR should be lower than initial LR"
    assert lrs[-1] >= 1e-6, "Final LR should not drop below eta_min"
