"""Parity test for fundus image preprocessing between golden notebook and drdg."""

import cv2
import numpy as np
import pytest

from drdg.data.preprocessing import (
    apply_green_channel_ablation,
    crop_fundus_roi,
    extract_green_channel,
)


def golden_crop_fundus_roi(img, tolerance=7):
    """Extracted from main_model_lodo.ipynb / resize_image.py."""
    if img is None or img.size == 0:
        return img
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    mask = gray > tolerance
    if not np.any(mask):
        return img
    row_mask = np.any(mask, axis=1)
    col_mask = np.any(mask, axis=0)
    ymin, ymax = np.where(row_mask)[0][[0, -1]]
    xmin, xmax = np.where(col_mask)[0][[0, -1]]
    cropped = img[ymin : ymax + 1, xmin : xmax + 1]
    return cropped if cropped.size > 0 else img


def golden_apply_green_channel_ablation(img_rgb, ablation_mode="ben_graham_green"):
    """Extracted verbatim from main_model_lodo.ipynb lines 472-495."""
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    denoised = cv2.bilateralFilter(bgr, d=9, sigmaColor=75, sigmaSpace=75)
    b, g, r = cv2.split(denoised)

    if ablation_mode == "ben_graham_green":
        blurred = cv2.GaussianBlur(g, (0, 0), 30)
        adjusted = cv2.addWeighted(g, 4, blurred, -4, 128)
        processed = cv2.merge([adjusted, adjusted, adjusted])
    elif ablation_mode == "clahe_green":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_g = clahe.apply(g)
        processed = cv2.merge([enhanced_g, enhanced_g, enhanced_g])
    else:
        processed = cv2.merge([g, g, g])

    return cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)


def test_crop_fundus_roi_exact_parity():
    """Verify ROI cropping produces identical bounded box."""
    # Create synthetic image with circular aperture and black border
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.circle(img, (150, 150), 100, (80, 150, 50), -1)

    gold_crop = golden_crop_fundus_roi(img, tolerance=7)
    cand_crop = crop_fundus_roi(img, tolerance=7)

    assert np.array_equal(gold_crop, cand_crop)
    assert gold_crop.shape == cand_crop.shape


@pytest.mark.parametrize("mode", ["ben_graham_green", "clahe_green", "raw_green"])
def test_green_channel_ablation_exact_parity(mode):
    """Verify green channel enhancement modes are byte-for-byte identical."""
    np.random.seed(123)
    # Generate synthetic color fundus image (RGB)
    img_rgb = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)

    gold_res = golden_apply_green_channel_ablation(img_rgb, ablation_mode=mode)
    cand_res = apply_green_channel_ablation(img_rgb, ablation_mode=mode)

    assert np.array_equal(gold_res, cand_res), f"Mismatch in preprocessing mode: {mode}"
