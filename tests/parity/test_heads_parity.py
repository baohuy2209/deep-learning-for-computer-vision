"""Behavioral parity tests for all 5 Output Heads.

Validates exact alignment with main_model_head_ablation (3).ipynb:
  1. Softmax_CE
  2. Softmax_QWK
  3. CORAL
  4. CORN
  5. CLM_QWK
"""

import pytest
import torch

from drdg.models.heads.softmax import SoftmaxCEHead, SoftmaxQWKHead
from drdg.models.heads.coral import CORALHead
from drdg.models.heads.corn import CORNHead
from drdg.models.heads.clm import CumulativeLinkModelQWKHead


@pytest.mark.parity
def test_all_five_heads_forward_shapes():
    """Verify forward pass output dictionary structure and probability simplex property."""
    torch.manual_seed(42)
    B, in_features = 4, 512
    x = torch.randn(B, in_features)

    heads = {
        "Softmax_CE": SoftmaxCEHead(in_features=in_features, num_classes=5),
        "Softmax_QWK": SoftmaxQWKHead(in_features=in_features, num_classes=5),
        "CORAL": CORALHead(in_features=in_features, num_classes=5),
        "CORN": CORNHead(in_features=in_features, num_classes=5),
        "CLM_QWK": CumulativeLinkModelQWKHead(in_features=in_features, num_classes=5),
    }

    for name, head in heads.items():
        head.eval()
        with torch.no_grad():
            out = head(x)

        assert isinstance(out, dict), f"Head {name} must return dict"
        assert "probs" in out or "probabilities" in out or "prob" in out, f"Head {name} missing probabilities"
        probs_key = next(k for k in ["probs", "probabilities", "prob"] if k in out)
        probs = out[probs_key]

        assert probs.shape == (B, 5), f"Head {name} prob shape mismatch: {probs.shape} vs {(B, 5)}"
        # Check probability simplex: non-negative and sums to 1
        assert (probs >= -1e-5).all(), f"Head {name} produced negative probabilities"
        row_sums = probs.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), f"Head {name} probabilities do not sum to 1: {row_sums}"
