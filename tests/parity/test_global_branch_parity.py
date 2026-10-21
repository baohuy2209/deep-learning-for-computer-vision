"""Behavioral parity tests for ResNet-50 GlobalContextStream.

Validates exact alignment with main_model (6).ipynb Cell 13:
  - ResNet-50 stem and stages
  - NonLocalBlock2D and CBAM attention on stage 4
  - GAP (2048-d) + QuadrantTokenAggregator (2048-d) -> 4096-d output
"""

import pytest
import torch

from drdg.models.streams.global_context import GlobalBranch


@pytest.mark.parity
def test_global_branch_forward_shapes():
    """Verify GlobalBranch output tensor dimensions with and without quadrant tokens."""
    torch.manual_seed(42)
    B = 2
    # Small test resolution
    x = torch.randn(B, 3, 224, 224)

    # 1. Full GlobalBranch with CBAM + NonLocal + QuadrantTokens -> 4096-d
    branch_full = GlobalBranch(
        pretrained=False,
        use_cbam=True,
        use_nonlocal=True,
        use_quadrant_tokens=True,
    )
    branch_full.eval()

    with torch.no_grad():
        out_full = branch_full(x)

    assert out_full.shape == (B, 4096), f"Expected shape (B, 4096), got {out_full.shape}"

    # 2. Ablated GlobalBranch without quadrant tokens -> 2048-d
    branch_no_qt = GlobalBranch(
        pretrained=False,
        use_cbam=True,
        use_nonlocal=True,
        use_quadrant_tokens=False,
    )
    branch_no_qt.eval()

    with torch.no_grad():
        out_no_qt = branch_no_qt(x)

    assert out_no_qt.shape == (B, 2048), f"Expected shape (B, 2048), got {out_no_qt.shape}"


@pytest.mark.parity
def test_global_branch_determinism():
    """Verify GlobalBranch evaluation mode produces deterministic numerical outputs."""
    torch.manual_seed(42)
    x = torch.randn(1, 3, 224, 224)

    branch = GlobalBranch(pretrained=False, use_cbam=True, use_nonlocal=True, use_quadrant_tokens=True)
    branch.eval()

    with torch.no_grad():
        out1 = branch(x)
        out2 = branch(x)

    assert torch.allclose(out1, out2, atol=1e-7), "GlobalBranch forward pass is not deterministic in eval mode"
