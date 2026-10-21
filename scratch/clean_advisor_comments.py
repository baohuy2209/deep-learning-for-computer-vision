import json
import re

targets = [
    "notebooks/diabetic_retinopathy/pipeline/model/main_model (4).ipynb",
    "notebooks/diabetic_retinopathy/pipeline/model/main_model (8).ipynb",
    "notebooks/diabetic_retinopathy/pipeline/model/main_model.ipynb"
]

replacements = [
    # Cell 10
    (
        "# --- ADVISOR HD_4 INTEGRATION: LOSS STABILIZATION (Part 4) ---",
        "# --- LOSS STABILIZATION & CONTINUOUS QWK OPTIMIZATION ---"
    ),
    (
        "# Extended linear warmup from 5 to 15 epochs to stabilize early batch QWK variance (HD_4 Part 4)",
        "# Extended linear warmup from 5 to 15 epochs to stabilize early batch QWK variance"
    ),
    # Cell 25
    (
        "# 2. CONFIGURE COLLATE FUNCTIONS (HD_3 & HD_4: Distinct Train vs Eval Data Loading)",
        "# 2. CONFIGURE COLLATE FUNCTIONS (Distinct Train vs Eval Data Loading)"
    ),
    (
        "eval_batch_size = 1  # HD_3 Page 4: batch_size=1 prevents OOM on dense patch grids and avoids padding variance",
        "eval_batch_size = 1  # batch_size=1 prevents OOM on dense patch grids and avoids padding variance"
    ),
    # Cell 37
    (
        "Added per advisor review: CBAM's spatial attention module",
        "Spatial Attention Enhancement: CBAM's spatial attention module"
    ),
    # Cell 40
    (
        "# directional aggregation - added per advisor",
        "# directional aggregation for quadrant-level feature extraction"
    ),
    # Cell 42
    (
        "self.last_attention_weights = a.detach()  # Retained for architecture-native Attention-MIL saliency (HD_4 Part 5)",
        "self.last_attention_weights = a.detach()  # Retained for architecture-native Attention-MIL saliency"
    ),
    # Cell 64
    (
        "# 4.1 CLINICAL SENSITIVITY AT FIXED SPECIFICITY & POST-HOC CALIBRATION (HD_4 Part 1 & 3)",
        "# 4.1 CLINICAL SENSITIVITY AT FIXED SPECIFICITY & POST-HOC CALIBRATION"
    ),
    (
        "and evaluates the calibrated operating points on the unseen Test set (HD_4 Part 1 & 3).",
        "and evaluates the calibrated operating points on the unseen Test set."
    ),
    (
        'print(f"-> Clinical Insight (HD_4 Part 1): Ref-AUC = {test_auc:.4f} confirms strong disease separation.")',
        'print(f"-> Clinical Insight: Ref-AUC = {test_auc:.4f} confirms strong disease separation.")'
    ),
    # Cell 71
    (
        "# # ADVISOR METHODOLOGICAL EVALUATION:",
        "# # CLINICAL METHODOLOGICAL EVALUATION:"
    ),
    # Cell 77
    (
        "# 5.2 QUANTITATIVE XAI EVALUATION DATASET WITH DUAL-CROP ALIGNMENT (HD_4 Part 2)",
        "# 5.2 QUANTITATIVE XAI EVALUATION DATASET WITH DUAL-CROP ALIGNMENT"
    ),
    (
        "Features Dual-Crop Alignment (HD_4 Part 2):",
        "Features Dual-Crop Alignment:"
    ),
    (
        "# 2. HD_4 DUAL-CROP ALIGNMENT (Part 2): Crop BOTH fundus and mask using identical bounding box",
        "# 2. DUAL-CROP ALIGNMENT: Crop BOTH fundus and mask using identical bounding box"
    ),
    (
        "# 4. Extract local patches and capture spatial patch coordinates (HD_4 Part 5)",
        "# 4. Extract local patches and capture spatial patch coordinates"
    ),
    # Cell 78
    (
        "# 5.3 ARCHITECTURE-NATIVE ATTENTION-MIL SALIENCY & OVERHAULED XAI METRIC SUITE (HD_4 Part 2 & 5)",
        "# 5.3 ARCHITECTURE-NATIVE ATTENTION-MIL SALIENCY & ADVANCED XAI METRIC SUITE"
    ),
    (
        "using patch coordinates [y0, x0] (HD_4 Part 5).",
        "using patch coordinates [y0, x0]."
    ),
    (
        "Overhauled Quantitative XAI Evaluation Suite (HD_4 Part 2 & 5):",
        "Quantitative XAI Evaluation Suite:"
    ),
    (
        'print(f"RUNNING OVERHAULED QUANTITATIVE XAI ON {len(idrid_xai_dataset)} REAL LESION MASK IMAGES (HD_4)")',
        'print(f"RUNNING QUANTITATIVE XAI ON {len(idrid_xai_dataset)} REAL LESION MASK IMAGES")'
    ),
    (
        "# --- B. SALIENCY RECALL @ AREA-MATCHED (HD_4 Part 2 & 6) ---",
        "# --- B. SALIENCY RECALL @ AREA-MATCHED ---"
    ),
    # Cell 81
    (
        "# 4. Compute quantitative XAI fidelity metrics (Overhauled Suite: HD_4 Part 2 & 5)",
        "# 4. Compute quantitative XAI fidelity metrics"
    ),
    (
        "# 5. Post-Hoc Screening Threshold Calibration (HD_4 Part 1 & 3)",
        "# 5. Post-Hoc Screening Threshold Calibration"
    ),
    (
        'print("\\nPerforming Post-Hoc Screening Threshold Calibration strictly on Validation & Test distributions (HD_4)...")',
        'print("\\nPerforming Post-Hoc Screening Threshold Calibration strictly on Validation & Test distributions...")'
    ),
]

for path in targets:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    modified_count = 0
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            modified_count += 1
        else:
            print(f"Warning: '{old[:40]}' not found in {path}")

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Successfully cleaned {modified_count}/{len(replacements)} references in {path}!")
