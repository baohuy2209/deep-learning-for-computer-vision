"""Unit tests for ordinal and classification heads in drdg.models.heads."""

import pytest
import torch

from drdg.losses.qwk import ContinuousQWKLoss
from drdg.models.heads.clm import CumulativeLinkModelQWKHead
from drdg.models.heads.coral import CORALHead
from drdg.models.heads.corn import CORNHead
from drdg.models.heads.softmax import SoftmaxCEHead, SoftmaxQWKHead


def test_continuous_qwk_loss():
    loss_fn = ContinuousQWKLoss(num_classes=5)
    # Perfect prediction should yield zero or near-zero loss
    probs = torch.eye(5)
    targets = torch.arange(5)
    loss = loss_fn(probs, targets)
    assert loss.item() >= 0.0
    assert torch.is_tensor(loss)


def test_softmax_ce_head():
    head = SoftmaxCEHead(in_features=64, num_classes=5)
    x = torch.randn(4, 64)
    targets = torch.tensor([0, 1, 2, 3])

    out = head(x)
    assert "logits" in out
    assert "probs" in out
    assert "preds" in out
    assert out["probs"].shape == (4, 5)
    assert out["preds"].shape == (4,)
    assert torch.allclose(out["probs"].sum(dim=1), torch.ones(4), atol=1e-5)

    loss = head.loss_fn(out, targets)
    loss.backward()
    assert head.fc.weight.grad is not None


def test_softmax_qwk_head():
    head = SoftmaxQWKHead(in_features=64, num_classes=5, qwk_weight=1.0)
    x = torch.randn(4, 64)
    targets = torch.tensor([0, 1, 4, 2])

    out = head(x)
    assert "continuous_preds" in out
    loss = head.loss_fn(out, targets)
    loss.backward()
    assert head.fc.weight.grad is not None


def test_coral_head():
    head = CORALHead(in_features=64, num_classes=5)
    x = torch.randn(4, 64)
    targets = torch.tensor([0, 2, 4, 1])

    out = head(x)
    assert "rank_logits" in out
    assert "rank_probs" in out
    assert "probs" in out
    assert "preds" in out
    assert out["rank_probs"].shape == (4, 4)
    assert out["probs"].shape == (4, 5)

    loss = head.loss_fn(out, targets)
    loss.backward()
    assert head.biases.grad is not None


def test_corn_head():
    head = CORNHead(in_features=64, num_classes=5)
    x = torch.randn(4, 64)
    targets = torch.tensor([0, 1, 2, 3])

    out = head(x)
    assert "cond_logits" in out
    assert "cond_probs" in out
    assert "cum_probs" in out
    assert "probs" in out

    # Cum probs must be monotonically non-increasing
    cum = out["cum_probs"].detach()
    diffs = cum[:, 1:] - cum[:, :-1]
    assert (diffs <= 1e-5).all()

    loss = head.loss_fn(out, targets)
    loss.backward()
    assert head.fc.weight.grad is not None


def test_clm_head():
    head = CumulativeLinkModelQWKHead(in_features=64, num_classes=5, qwk_weight=0.4)
    head.set_qwk_weight(0.4)
    x = torch.randn(4, 64)
    targets = torch.tensor([0, 1, 2, 4])

    out = head(x)
    assert "severity_score" in out
    assert "cutpoints" in out
    assert "cum_probs" in out
    assert "probs" in out
    assert "expected_grade" in out
    assert "preds" in out

    cutpoints = out["cutpoints"].detach()
    assert len(cutpoints) == 4
    # Cutpoints must be strictly monotonic
    for i in range(len(cutpoints) - 1):
        assert cutpoints[i] < cutpoints[i + 1]

    loss = head.loss_fn(out, targets)
    loss.backward()
    assert head.first_cutpoint.grad is not None
    assert head.cutpoint_increments.grad is not None
