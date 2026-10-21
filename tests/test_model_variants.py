"""Unit tests for model variants V0-V3 and ablation models."""

import pytest
import torch

from drdg.models.factory import build_ablation_model, build_variant


def test_v0_variant():
    model = build_variant("v0", pretrained=False, fusion_dim=64)
    model.eval()
    x = torch.randn(2, 3, 128, 128)
    targets = torch.tensor([0, 1])

    out = model(x, targets=targets, return_loss=True)
    assert "probs" in out
    assert "preds" in out
    assert "latent" in out
    assert "loss" in out
    assert out["probs"].shape == (2, 5)
    assert out["latent"].shape == (2, 64)


def test_v1_variant():
    model = build_variant("v1", pretrained=False, fusion_dim=64)
    model.eval()
    x = torch.randn(2, 3, 128, 128)
    patches = torch.randn(2, 4, 3, 64, 64)
    targets = torch.tensor([1, 2])

    out = model(x, local_patches=patches, targets=targets, return_loss=True)
    assert "probs" in out
    assert "preds" in out
    assert out["probs"].shape == (2, 5)


def test_v2_variant():
    model = build_variant("v2", pretrained=False, fusion_dim=64)
    model.eval()
    x = torch.randn(2, 3, 128, 128)
    patches = torch.randn(2, 4, 3, 64, 64)
    targets = torch.tensor([2, 3])

    out = model(x, local_patches=patches, targets=targets, return_loss=True)
    assert "probs" in out
    assert "preds" in out
    assert out["probs"].shape == (2, 5)


def test_v3_variant():
    model = build_variant("v3", pretrained=False, fusion_dim=64)
    model.train()
    head = getattr(model, "head", None)
    if head is not None and hasattr(head, "set_qwk_weight"):
        head.set_qwk_weight(0.4)

    x = torch.randn(2, 3, 128, 128)
    patches = torch.randn(2, 4, 3, 64, 64)
    patch_mask = torch.ones(2, 4)
    targets = torch.tensor([0, 4])

    out = model(x, local_patches=patches, patch_mask=patch_mask, targets=targets, return_loss=True)
    assert "probs" in out
    assert "preds" in out
    assert "cutpoints" in out
    assert "severity_score" in out
    assert "loss" in out

    out["loss"].backward()


def test_resnet50_ablation_model():
    model = build_ablation_model(head_type="CLM_QWK", pretrained=False)
    x = torch.randn(2, 3, 128, 128)
    targets = torch.tensor([1, 3])

    out = model(x, targets=targets, return_loss=True)
    assert "probs" in out
    assert "preds" in out
    assert "loss" in out
    assert out["latent"].shape == (2, 512)
