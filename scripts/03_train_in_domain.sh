#!/usr/bin/env bash
# ==============================================================================
# Script 03: Train In-Domain Multi-Scale MIL Model
# Trains champion model with Consistent Ordinal Regression
# ==============================================================================

set -e

echo "[INFO] Training In-Domain Multi-Scale MIL Model..."
python -m dr_multiscale_mil.train \
    --config configs/train/in_domain.yaml \
    --checkpoint-dir checkpoints/checkpoints_main \
    --device cuda:0

echo "[SUCCESS] In-domain model training completed."
