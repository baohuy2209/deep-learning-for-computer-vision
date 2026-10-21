"""Unit tests for sliding-window patch extraction and foreground filtering."""

import numpy as np
import pytest
import torch

from drdg.data.patches import compute_valid_anchors, extract_mil_patches_sliding


def test_compute_valid_anchors():
    # Synthetic image of shape (400, 400)
    fov_mask = np.zeros((400, 400), dtype=bool)
    fov_mask[50:350, 50:350] = True  # Central 300x300 foreground square

    anchors = compute_valid_anchors(
        img_shape=(400, 400),
        fov_mask=fov_mask,
        patch_size=128,
        stride=64,
        fov_thresh=0.5,
    )

    assert isinstance(anchors, np.ndarray)
    assert anchors.ndim == 2
    assert anchors.shape[1] == 2
    assert len(anchors) > 0

    # Ensure all anchors are within bounds
    for y0, x0 in anchors:
        assert 0 <= y0 <= 400 - 128
        assert 0 <= x0 <= 400 - 128


def test_extract_mil_patches_sliding():
    img_np = np.full((300, 300, 3), 128, dtype=np.uint8)
    fov_mask = np.ones((300, 300), dtype=bool)

    # Test train mode (fixed number of sampled patches)
    patches, coords = extract_mil_patches_sliding(
        img_np=img_np,
        fov_mask=fov_mask,
        patch_size=64,
        stride=32,
        mode="train",
        num_sample=8,
        return_anchors=True,
    )

    assert isinstance(patches, torch.Tensor)
    assert patches.shape == (8, 3, 64, 64)
    assert coords.shape == (8, 2)

    # Test eval mode (deterministic sampling)
    patches_eval = extract_mil_patches_sliding(
        img_np=img_np,
        fov_mask=fov_mask,
        patch_size=64,
        stride=32,
        mode="eval",
        max_patches_eval=16,
    )
    assert patches_eval.shape[0] <= 16
    assert patches_eval.shape[1:] == (3, 64, 64)
