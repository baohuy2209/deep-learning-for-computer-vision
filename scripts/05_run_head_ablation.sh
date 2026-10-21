#!/usr/bin/env bash
# ==============================================================================
# Script 05: Run Head Ablation Study
# Compares Softmax CE, CORAL, and Consistent Ordinal Regression Heads
# ==============================================================================

set -e

echo "[INFO] Running Prediction Head Ablation Study..."
python -m dr_multiscale_mil.train_ablation \
    --config configs/ablations/head_ablation.yaml \
    --device cuda:0

echo "[SUCCESS] Head ablation study completed."
