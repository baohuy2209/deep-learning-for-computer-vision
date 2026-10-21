"""Behavioral parity tests for EfficientNet-B0 LocalMILBranch with Dual Pooling.

Validates exact alignment with main_model (6).ipynb Cell 14:
  - EfficientNet-B0 feature extraction per patch
  - Gated Attention Pooling + 1D Max Pooling -> 2560-d output
  - Patch mask gating for padded patches
"""

import pytest
import torch

from drdg.models.streams.local_mil import GatedAttentionPool, LocalMILBranch


@pytest.mark.parity
def test_gated_attention_pool_masking():
    """Verify GatedAttentionPool ignores masked-out padded patches."""
    torch.manual_seed(42)
    B, K, D = 2, 4, 128
    features = torch.randn(B, K, D)

    # Sample 0 has 4 valid patches; Sample 1 has only 2 valid patches
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)

    pool = GatedAttentionPool(in_features=D, hidden_dim=64)
    pool.eval()

    with torch.no_grad():
        out = pool(features, patch_mask=mask)

    assert out.shape == (B, D)
    # Check that attention weights for masked patches are ~0
    weights = pool.last_attention_weights.squeeze(-1)  # [B, K]
    assert weights[1, 2].item() < 1e-4
    assert weights[1, 3].item() < 1e-4
    assert torch.isclose(weights[1].sum(), torch.tensor(1.0), atol=1e-4)


@pytest.mark.parity
def test_local_mil_branch_shapes():
    """Verify LocalMILBranch output shapes with dual pooling (2560) vs single pooling (1280)."""
    torch.manual_seed(42)
    B, K = 2, 3
    # Small test patch resolution
    patches = torch.randn(B, K, 3, 112, 112)
    mask = torch.ones(B, K, dtype=torch.bool)

    # 1. Dual pooling (Max + Gated Attention) -> 2560-d
    mil_dual = LocalMILBranch(pretrained=False, use_dual_pooling=True)
    mil_dual.eval()

    with torch.no_grad():
        out_dual = mil_dual(patches, patch_mask=mask)

    assert out_dual.shape == (B, 2560), f"Expected (B, 2560), got {out_dual.shape}"

    # 2. Single pooling -> 1280-d
    mil_single = LocalMILBranch(pretrained=False, use_dual_pooling=False)
    mil_single.eval()

    with torch.no_grad():
        out_single = mil_single(patches, patch_mask=mask)

    assert out_single.shape == (B, 1280), f"Expected (B, 1280), got {out_single.shape}"
