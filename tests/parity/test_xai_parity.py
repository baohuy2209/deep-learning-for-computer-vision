import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from drdg.explainability.gradcam import GradCAM
from drdg.explainability.mil_saliency import MILAttentionSaliency


class DummyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 5)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        pooled = self.pool(h).flatten(1)
        logits = self.fc(pooled)
        return {"logits": logits}


def test_gradcam_parity_and_cleanup():
    torch.manual_seed(42)
    model = DummyCNN()
    gradcam = GradCAM(model, model.conv2)

    img = torch.randn(1, 3, 64, 64)
    cam = gradcam.generate_saliency_map(img, target_class=2, target_size=(64, 64))

    assert cam.shape == (64, 64)
    assert cam.dtype == np.float32
    assert 0.0 <= cam.min() <= cam.max() <= 1.0

    # Ensure hooks can be safely removed
    gradcam.remove_hooks()
    assert len(gradcam._hook_handles) == 0


def test_mil_attention_saliency_exact_parity():
    # Synthetic attention weights and anchors
    att_weights = np.array([0.1, 0.4, 0.3, 0.2], dtype=np.float32)
    anchors = np.array([[0, 0], [0, 32], [32, 0], [32, 32]], dtype=np.int32)
    canvas_size = (64, 64)
    patch_size = 32

    # 1. Golden Reference Algorithm (from notebook Cell 25 / 5.3)
    canvas = np.zeros(canvas_size, dtype=np.float32)
    counts = np.zeros(canvas_size, dtype=np.float32)
    H_can, W_can = canvas_size

    for k, (y0, x0) in enumerate(anchors):
        alpha_k = float(att_weights[k])
        y1 = min(int(y0 + patch_size), H_can)
        x1 = min(int(x0 + patch_size), W_can)
        canvas[y0:y1, x0:x1] += alpha_k
        counts[y0:y1, x0:x1] += 1.0

    mil_map_ref = np.divide(canvas, counts, out=np.zeros_like(canvas), where=(counts > 0))
    denom = mil_map_ref.max() - mil_map_ref.min()
    if denom > 1e-7:
        mil_map_ref = (mil_map_ref - mil_map_ref.min()) / denom

    # 2. Package Implementation
    saliency_tool = MILAttentionSaliency(patch_size=patch_size, sigma_smooth=0.0)
    mil_map_pkg = saliency_tool.generate_saliency_map(
        attention_weights=att_weights,
        patch_anchors=anchors,
        img_shape=canvas_size,
        normalize=True,
    )

    # 3. Exact Parity Check
    np.testing.assert_allclose(mil_map_pkg, mil_map_ref, rtol=1e-5, atol=1e-6)
