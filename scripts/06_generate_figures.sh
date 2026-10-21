#!/usr/bin/env bash
# ==============================================================================
# Script 06: Generate Publication Figures & XAI Visualizations
# Renders Figures 1 to 6 in results/figures/ (300 DPI PNG + vector PDF)
# ==============================================================================

set -e

echo "[INFO] Generating publication figures for ICCPR paper..."
python -m dr_multiscale_mil.visualize \
    --output-dir results/figures \
    --dpi 300

echo "[SUCCESS] Publication figures generated successfully."
