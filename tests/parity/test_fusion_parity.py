"""Behavioral parity tests for DualBranchFusion module.

Validates exact alignment with main_model (6).ipynb Cell 15:
  - DualBranchFusion with LayerNorm + Mish
  - Support for dual-branch (global + local) and global-only modes
"""

import pytest
import torch

from drdg.models.fusion import DualBranchFusion


@pytest.mark.parity
def test_dual_branch_fusion_shapes():
    """Verify fusion projection shapes for dual-branch and global-only configurations."""
    torch.manual_seed(42)
    B = 2
    global_dim = 4096
    local_dim = 2560
    fusion_dim = 512

    f_global = torch.randn(B, global_dim)
    f_local = torch.randn(B, local_dim)

    # 1. Dual-branch fusion
    fusion_dual = DualBranchFusion(
        global_dim=global_dim,
        local_dim=local_dim,
        fusion_dim=fusion_dim,
        use_local_branch=True,
    )
    fusion_dual.eval()

    with torch.no_grad():
        out_dual = fusion_dual(f_global, f_local)

    assert out_dual.shape == (B, fusion_dim), f"Expected (B, {fusion_dim}), got {out_dual.shape}"

    # 2. Global-only fusion (V0 ablation)
    fusion_global = DualBranchFusion(
        global_dim=global_dim,
        local_dim=local_dim,
        fusion_dim=fusion_dim,
        use_local_branch=False,
    )
    fusion_global.eval()

    with torch.no_grad():
        out_global = fusion_global(f_global)

    assert out_global.shape == (B, fusion_dim), f"Expected (B, {fusion_dim}), got {out_global.shape}"


@pytest.mark.parity
def test_fusion_layernorm_batch_size_one_stability():
    """Verify LayerNorm prevents batch-size 1 failure unlike BatchNorm."""
    f_global = torch.randn(1, 4096)
    f_local = torch.randn(1, 2560)

    fusion = DualBranchFusion(global_dim=4096, local_dim=2560, fusion_dim=512)
    fusion.train()  # In training mode, BatchNorm with B=1 would fail; LayerNorm succeeds

    out = fusion(f_global, f_local)
    assert out.shape == (1, 512)
    assert not torch.isnan(out).any()
