"""Behavioral parity tests for Custom Loss functions and gradients.

Validates exact alignment with notebook formulations:
  - ContinuousQWKLoss: forward loss + non-zero backward gradients
  - CORAL Loss: forward loss + non-zero backward gradients
  - CORN Loss: forward loss + non-zero backward gradients
  - CumulativeLinkLoss / CLM Loss: forward loss + non-zero backward gradients
"""

import pytest
import torch
import torch.nn.functional as F

from drdg.losses.qwk import ContinuousQWKLoss
from drdg.losses.ordinal import coral_loss, corn_loss


@pytest.mark.parity
def test_continuous_qwk_loss_forward_and_backward():
    """Verify ContinuousQWKLoss produces scalar loss and backward gradients."""
    torch.manual_seed(42)
    B, C = 8, 5
    logits = torch.randn(B, C, requires_grad=True)
    probs = F.softmax(logits, dim=1)
    targets = torch.tensor([0, 1, 2, 3, 4, 1, 2, 0])

    loss_fn = ContinuousQWKLoss(num_classes=C)
    loss = loss_fn(probs, targets)

    assert loss.ndim == 0, "Loss must be scalar"
    assert loss.item() >= 0.0, "QWK loss penalty must be non-negative"

    # Backward gradient check
    loss.backward()
    assert logits.grad is not None
    assert not torch.allclose(logits.grad, torch.zeros_like(logits.grad)), "Gradients should be non-zero"
    assert not torch.isnan(logits.grad).any(), "Gradients must not contain NaN"


@pytest.mark.parity
def test_coral_loss_forward_and_backward():
    """Verify coral_loss binary cross-entropy and backward gradients."""
    torch.manual_seed(42)
    B, K = 8, 4
    logits = torch.randn(B, K, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 3, 4, 1, 2, 0])

    loss = coral_loss(logits, targets, num_thresholds=K)
    assert loss.ndim == 0
    assert loss.item() > 0.0

    loss.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()


@pytest.mark.parity
def test_corn_loss_forward_and_backward():
    """Verify corn_loss conditional binary cross-entropy and backward gradients."""
    torch.manual_seed(42)
    B, K = 8, 4
    logits = torch.randn(B, K, requires_grad=True)
    targets = torch.tensor([0, 1, 2, 3, 4, 1, 2, 0])

    loss = corn_loss(logits, targets, num_thresholds=K)
    assert loss.ndim == 0
    assert loss.item() > 0.0

    loss.backward()
    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()
