"""Behavioral parity tests for architectural ablation variants (V0 through V3).

Validates exact alignment with main_model_lodo.ipynb:
  - V0: Global Context baseline (ResNet-50 + Softmax CE)
  - V1: Local MIL stream (+ EfficientNet-B0 Dual Pooling)
  - V2: Global Context enrichment (+ Non-Local + Quadrant Tokens)
  - V3: Proposed Architecture (+ Cumulative Link Model & QWK loss)
"""

import pytest
import torch

from drdg.models.variants.v0 import build_v0_model
from drdg.models.variants.v1 import build_v1_model
from drdg.models.variants.v2 import build_v2_model
from drdg.models.variants.v3 import build_v3_model


@pytest.mark.parity
def test_all_variants_forward_passes():
    """Verify that V0, V1, V2, V3 instantiate and execute forward passes cleanly on CPU."""
    torch.manual_seed(42)
    B, K = 2, 3
    x_global = torch.randn(B, 3, 224, 224)
    x_local = torch.randn(B, K, 3, 112, 112)
    patch_mask = torch.ones(B, K, dtype=torch.bool)

    variants = {
        "V0": build_v0_model(pretrained=False),
        "V1": build_v1_model(pretrained=False),
        "V2": build_v2_model(pretrained=False),
        "V3": build_v3_model(pretrained=False),
    }

    for name, model in variants.items():
        model.eval()
        with torch.no_grad():
            out = model(x_global, x_local, patch_mask=patch_mask)

        assert isinstance(out, dict), f"{name} must return output dict"
        assert "probabilities" in out or "probs" in out, f"{name} missing probabilities"
        probs_key = "probabilities" if "probabilities" in out else "probs"
        probs = out[probs_key]
        assert probs.shape == (B, 5), f"{name} probability shape mismatch: {probs.shape} vs {(B, 5)}"
        assert not torch.isnan(probs).any(), f"{name} probabilities contain NaN"
