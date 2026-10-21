"""Unit tests for morphological FOV mask computation."""

import numpy as np
import pytest

from drdg.data.fov import apply_fov_mask, compute_fov_mask
from drdg.data.preprocessing import crop_fundus_roi, extract_green_channel


def test_compute_fov_mask_on_circular_disk():
    # Construct a synthetic 200x200 image with a bright circle in the center
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    y, x = np.ogrid[:200, :200]
    center = (100, 100)
    radius = 80
    circle_mask = ((x - center[0]) ** 2 + (y - center[1]) ** 2) <= radius ** 2
    img[circle_mask] = 180

    # Add artificial dark spot (simulating fovea or lesion)
    spot_mask = ((x - 100) ** 2 + (y - 100) ** 2) <= 10 ** 2
    img[spot_mask] = 0

    fov_mask = compute_fov_mask(img, threshold=10, kernel_size=25)

    assert fov_mask.shape == (200, 200)
    assert fov_mask.dtype == bool
    # Center spot should be filled by morphological closing
    assert fov_mask[100, 100] is True or fov_mask[100, 100] == 1
    # Outer corners must remain background
    assert fov_mask[5, 5] == False
    assert fov_mask[195, 195] == False


def test_crop_fundus_roi():
    # Construct image with black borders around 100x100 tissue in a 300x300 image
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[100:200, 80:220] = 150

    cropped = crop_fundus_roi(img, tolerance=10)
    assert cropped.shape[0] == 100
    assert cropped.shape[1] == 140


def test_extract_green_channel():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    img[:, :, 1] = 200  # Green channel only

    g_single = extract_green_channel(img, as_three_channel=False)
    assert g_single.shape == (50, 50)
    assert (g_single == 200).all()

    g_triple = extract_green_channel(img, as_three_channel=True)
    assert g_triple.shape == (50, 50, 3)
    assert (g_triple[:, :, 0] == 200).all()
