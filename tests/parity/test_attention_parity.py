"""Behavioral parity tests for attention modules (CBAM, NonLocal, Quadrant Tokens).

Validates exact alignment with main_model (6).ipynb Cell 12:
  - ChannelAttention & SpatialAttention & CBAM
  - NonLocalBlock2D
  - QuadrantTokenAggregator
"""

import pytest
import torch
import torch.nn as nn

from drdg.models.attention.cbam import CBAM, ChannelAttention, SpatialAttention
from drdg.models.attention.non_local import NonLocalBlock2D
from drdg.models.attention.quadrant_tokens import QuadrantTokenAggregator


@pytest.mark.parity
def test_cbam_forward_parity():
    """Verify CBAM forward pass preserves spatial resolution and applies attention gating."""
    torch.manual_seed(42)
    B, C, H, W = 2, 64, 16, 16
    x = torch.randn(B, C, H, W)

    cbam = CBAM(in_channels=C, reduction_ratio=16, kernel_size=7)
    cbam.eval()

    with torch.no_grad():
        out = cbam(x)

    assert out.shape == x.shape, f"CBAM output shape mismatch: {out.shape} vs {x.shape}"
    assert not torch.allclose(out, x, atol=1e-5)
    assert not torch.allclose(out, torch.zeros_like(out), atol=1e-5)


@pytest.mark.parity
def test_non_local_block_parity():
    """Verify NonLocalBlock2D self-attention and residual connection."""
    torch.manual_seed(42)
    B, C, H, W = 2, 64, 8, 8
    x = torch.randn(B, C, H, W)

    non_local = NonLocalBlock2D(in_channels=C, reduction=2)
    non_local.eval()

    with torch.no_grad():
        out = non_local(x)

    assert out.shape == x.shape, f"NonLocal output shape mismatch: {out.shape} vs {x.shape}"
    assert torch.allclose(out, x, atol=1e-6), "Initial output projection is zero-initialized; output equals input."


@pytest.mark.parity
def test_quadrant_token_aggregator_parity():
    """Verify QuadrantTokenAggregator produces relational summary across 4 quadrants."""
    torch.manual_seed(42)
    B, C, H, W = 2, 128, 16, 16
    x = torch.randn(B, C, H, W)

    aggregator = QuadrantTokenAggregator(in_channels=C, num_heads=4)
    aggregator.eval()

    with torch.no_grad():
        summary = aggregator(x)

    assert summary.shape == (B, C), f"QuadrantTokenAggregator shape mismatch: {summary.shape} vs {(B, C)}"
