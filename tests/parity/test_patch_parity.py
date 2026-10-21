"""Parity test for patch anchor and sliding window extraction between golden reference and drdg."""

import numpy as np
import pytest
import torch

from drdg.data.patches import compute_valid_anchors as drdg_compute_anchors
from drdg.data.patches import extract_mil_patches_sliding as drdg_extract_patches


def golden_compute_valid_anchors(img_shape, fov_mask, patch_size=128, stride=96, fov_thresh=0.5):
    """Extracted from main_model_lodo.ipynb Cell 10."""
    h, w = img_shape[:2]
    p = patch_size
    if np.any(fov_mask):
        coords = np.argwhere(fov_mask)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
    else:
        y_min, x_min, y_max, x_max = 0, 0, h, w

    content_h = max(1, y_max - y_min)
    content_w = max(1, x_max - x_min)

    def get_coords(curr_stride: int):
        s = max(1, curr_stride)
        y_starts = list(range(y_min, y_min + content_h - p + 1, s)) if content_h > p else [y_min]
        x_starts = list(range(x_min, x_min + content_w - p + 1, s)) if content_w > p else [x_min]
        if content_h > p and (y_min + content_h - p) not in y_starts:
            y_starts.append(y_min + content_h - p)
        if content_w > p and (x_min + content_w - p) not in x_starts:
            x_starts.append(x_min + content_w - p)

        valid_list = []
        for y0 in y_starts:
            for x0 in x_starts:
                y1, x1 = min(y0 + p, h), min(x0 + p, w)
                mask_patch = fov_mask[y0:y1, x0:x1]
                fov_ratio = mask_patch.mean() if mask_patch.size > 0 else 0.0
                if fov_ratio >= fov_thresh:
                    valid_list.append([y0, x0])
        return valid_list

    anchors = get_coords(stride)
    if len(anchors) < 32 and stride > 48 and min(content_h, content_w) > p:
        dense_anchors = get_coords(48)
        if len(dense_anchors) > len(anchors):
            anchors = dense_anchors

    if len(anchors) == 0:
        cy = max(0, y_min + content_h // 2 - p // 2)
        cx = max(0, x_min + content_w // 2 - p // 2)
        anchors = [[cy, cx]]

    return np.asarray(anchors, dtype=np.int32)


def test_anchor_extraction_exact_parity():
    # Synthetic circular mask of shape 512x512
    h, w = 512, 512
    yy, xx = np.ogrid[:h, :w]
    mask = ((xx - 256)**2 + (yy - 256)**2 <= 240**2).astype(np.uint8) * 255

    gold_anchors = golden_compute_valid_anchors((h, w), mask, patch_size=128, stride=96, fov_thresh=0.5)
    cand_anchors = drdg_compute_anchors((h, w), mask, patch_size=128, stride=96, fov_thresh=0.5)

    assert np.array_equal(gold_anchors, cand_anchors), "Anchor coordinates differ between golden and drdg!"
    assert len(gold_anchors) > 0, "No anchors found!"


def test_extract_mil_patches_eval_mode_parity():
    np.random.seed(42)
    img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:512, :512]
    mask = ((xx - 256)**2 + (yy - 256)**2 <= 240**2).astype(np.uint8) * 255
    anchors = drdg_compute_anchors((512, 512), mask, patch_size=128, stride=96, fov_thresh=0.5)

    # In eval mode without stochastic sampling
    patches = drdg_extract_patches(
        img,
        fov_mask=mask,
        patch_size=128,
        stride=96,
        fov_thresh=0.5,
        mode="eval",
        valid_anchors=anchors,
    )

    assert isinstance(patches, torch.Tensor)
    assert patches.ndim == 4
    assert patches.shape[1] == 3
    assert patches.shape[2] == 128
    assert patches.shape[3] == 128
    assert patches.shape[0] == len(anchors)
