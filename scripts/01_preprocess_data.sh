#!/usr/bin/env bash
# ==============================================================================
# Script 01: Preprocess Fundus Images
# Auto-crops circular fundus ROI across all 6 raw cohorts to native_cropped/
# ==============================================================================

set -e

echo "[INFO] Running Fundus Circular ROI Cropping across all 6 cohorts..."
python -m dr_multiscale_mil.data.preprocess \
    --config configs/data/datasets.yaml \
    --tolerance 7 \
    --num-workers 8 \
    --jpeg-quality 95

echo "[SUCCESS] Preprocessing completed."
