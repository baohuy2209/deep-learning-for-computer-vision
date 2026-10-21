#!/usr/bin/env bash
# ==============================================================================
# Script 04: Run 6-Fold Leave-One-Dataset-Out (LODO) Generalization Benchmark
# Evaluates clinical domain shift across all 6 held-out cohorts
# ==============================================================================

set -e

echo "[INFO] Running 6-Fold LODO Generalization Benchmark..."
python -m dr_multiscale_mil.train_lodo \
    --config configs/train/lodo_benchmark.yaml \
    --all-folds \
    --device cuda:0

echo "[SUCCESS] 6-Fold LODO Benchmark completed."
