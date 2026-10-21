#!/usr/bin/env bash
# ==============================================================================
# Script 02: Generate In-Domain and 6-Fold LODO Splits
# Produces zero-leakage patient-level splits and multi-source quota-balanced pools
# ==============================================================================

set -e

echo "[INFO] Generating In-Domain 70/15/15 and 6-Fold LODO Splits..."
python -m dr_multiscale_mil.data.make_splits \
    --config configs/data/datasets.yaml \
    --lodo-config configs/data/lodo_folds.yaml \
    --mode both \
    --seed 42 \
    --target-min 1500 \
    --target-max 2500

echo "[SUCCESS] Splits generated successfully."
