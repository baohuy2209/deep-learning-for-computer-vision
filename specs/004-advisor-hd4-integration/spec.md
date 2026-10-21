# Feature Specification: Proposed Model Experimental Upgrade (Advisor HD_4 Integration)

**Feature Directory**: `specs/004-advisor-hd4-integration`  
**Created**: 2026-09-03  
**Status**: Ready for Planning  
**Input**: Advisor review notes (`HD_4.docx` / `HD_4.pdf`) and thesis experimental requirements across `HD_1.pdf`, `HD_2.pdf`, `HD_3.pdf`, `HD_4.pdf`.  
**Target Codebase Artifacts**:
- [`notebooks/diabetic_retinopathy/pipeline/model/main_model (4).ipynb`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/model/main_model%20(4).ipynb)
- [`notebooks/diabetic_retinopathy/pipeline/model/main_model.ipynb`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/model/main_model.ipynb)

---

## Executive Summary & Alignment with Advisor Directives

This specification translates the methodological critique, bug fixes, and experimental roadmap provided by the thesis advisor in `HD_4` into verifiable functional requirements. Under Constitution v1.1.0 (Strict Preservation & Delta-Only Directives), all modifications are confined strictly to targeted functions without altering stable data loaders, feature extractors, or loss formulations outside the explicit scope.

---

## User Scenarios & Prioritized User Stories

### User Story 1 - Dual-Crop Alignment & XAI Metric Overhaul (Priority: P1)
As a clinical researcher evaluating Explainable AI (XAI) localization fidelity on external benchmarks, I need the IDRiD fundus image and its expert lesion mask to be cropped using the identical retinal Field-of-View (FoV) bounding box, and evaluated against appropriate small-lesion metrics, so that scale mismatch is eliminated and localization performance is scientifically sound.

**Why this priority**:
- `HD_4 Part 2 & 6` proves that raw IDRiD images ($4288 \times 2848$) contain substantial black lateral margins, whereas training images were `native_cropped`. Feeding raw IDRiD images into Grad-CAM introduced an artificial scale distortion.
- Raw Mean IoU on microaneurysms has a mathematical upper bound of $\approx 0.050$ (when CAM covers ~30% and mask covers ~1.5%), making raw IoU an invalid metric for tiny lesions.

**Independent Test**:
- Execute `compute_quantitative_xai` on `IDRiDXAIEvalDataset`. Verify that both image and lesion mask share exact FoV bounding box crop, Pointing Game is reported alongside the random baseline ($\approx 1.5\%$), and `Saliency Recall @ Area-Matched` plus `Pixel-level AUROC` are computed without exceptions.

**Acceptance Scenarios**:
1. **Given** a raw IDRiD image and its combined binary lesion mask, **When** loaded by `IDRiDXAIEvalDataset`, **Then** the retinal bounding box $[y_0:y_1, x_0:x_1]$ is derived via `compute_fov_mask(img_np, erode_px=0)` and applied simultaneously to both the fundus image and lesion mask before resizing to $512 \times 512$.
2. **Given** visual saliency heatmaps and binary lesion masks, **When** XAI metrics are evaluated, **Then** the report outputs:
   - Pointing Game Accuracy with explicit random baseline (e.g., `11.11% vs. 1.5% random baseline, 7.4x above chance`).
   - Saliency Recall @ Area-Matched (thresholding top-$q\%$ heatmap pixels where $q$ equals ground-truth lesion mask area ratio).
   - Pixel-level AUROC (ROC-AUC computed directly over continuous heatmap intensities against binary lesion pixels).
   - Raw Mean IoU is deprecated or contextualized against its theoretical ceiling of 0.05.

---

### User Story 2 - Post-Hoc Threshold Calibration for Referable DR Screening (Priority: P1)
As an ophthalmological AI evaluator, I need the binary screening decision boundary ($y \ge 2$) to be calibrated on the Validation set continuous probability predictions $P(y \ge 2)$ rather than taking the nominal 5-class `argmax`, so that clinical screening sensitivities can be measured at standardized operating specificities.

**Why this priority**:
- `HD_4 Part 1 & 3` demonstrates that the proposed model already achieved a strong Test Ref-AUC of $0.9063$. The observed sub-80% sensitivity at `argmax` was purely due to operating point placement (the `argmax` boundary sits at $\approx 95\%$ specificity).
- Calibrating the operating threshold on Validation probabilities requires 0 GPU training hours and provides precise clinical control without degrading representation quality.

**Independent Test**:
- Fit decision thresholds on `val_results` predicted probabilities $P(y \ge 2)$ for target specificities $[95\%, 90\%, 85\%, 80\%]$, apply these thresholds to `test_results`, and verify that sensitivity and specificity curves match the ROC trajectory.

**Acceptance Scenarios**:
1. **Given** model probability outputs on the Validation set, **When** screening calibration is performed, **Then** operating thresholds are identified to achieve specificities of $95\%$, $90\%$, $85\%$, and $80\%$.
2. **Given** calibrated thresholds, **When** evaluated on the unseen Test set, **Then** the pipeline outputs:
   - Full Referable DR ROC curve with Area Under Curve (Ref-AUC).
   - A standardized clinical performance table reporting Test Sensitivity and Test Specificity at each calibrated threshold.
   - Clear documentation highlighting UK NSC standard alignment (Sensitivity $\ge 80\%$ at Specificity $\ge 95\%$) and limitations.

---

### User Story 3 - Architecture-Native Attention-MIL Saliency Map Engine (Priority: P2)
As a machine learning researcher validating Hypothesis $H_1$, I need spatial patch coordinates `[y0, x0]` forwarded through the Local Branch and attention weights $\alpha_k$ projected onto a 2D spatial canvas, so that the model generates an architecture-native fine-grained saliency map (~128 px resolution) to compare against global Grad-CAM (~268 px resolution).

**Why this priority**:
- `HD_4 Part 5` emphasizes that `GatedAttentionPool` inherently assigns attention weights $\alpha_k$ to each local patch. Re-projecting $\alpha_k$ back to the image plane provides intrinsic explainability directly from the model's design, providing independent evidence for $H_1$.

**Independent Test**:
- Run a forward pass through `DualBranchMILModel` with patch coordinates enabled. Extract attention vector $\boldsymbol{\alpha}$, project onto a $512 \times 512$ canvas, and compute localization fidelity against IDRiD lesion masks.

**Acceptance Scenarios**:
1. **Given** a fundus image processed by `FundusDualBranchDataset`, **When** patches are extracted, **Then** normalized or pixel-space patch anchors $[y_0, x_0]$ are bundled into the batch dictionary.
2. **Given** patch attention weights $\alpha_k$ from `GatedAttentionPool`, **When** visual saliency is requested, **Then** a 2D Attention-MIL heatmap is rendered by accumulative splatting/splining onto the canvas.
3. **Given** both Global Grad-CAM and Local Attention-MIL heatmaps, **When** evaluated on the IDRiD mask set, **Then** comparative localization metrics (Pointing Game, Area-Matched Recall, AUROC) are displayed side-by-side.

---

### User Story 4 - Training Stability & Convergence Configuration (Priority: P2)
As a researcher training or fine-tuning the proposed dual-branch model, I need loss warmup schedules extended to 15 epochs, loss weight $\lambda_{\text{QWK}}$ adjusted to 0.4, and stopping criteria unified on Validation QWK with patience 15–20, so that batch-level QWK gradient variance is suppressed and the model converges stably.

**Why this priority**:
- `HD_4 Part 4` identifies that Validation QWK oscillations began at Epoch 8 (immediately after the 5-epoch warmup ended and full weight $\lambda = 1.2$ engaged).
- The existing run halted at Epoch 16 with a 0.20 generalization gap because `EarlyStopping` was tracking `val_loss` while `ModelCheckpoint` was tracking `val_QWK`.

**Independent Test**:
- Inspect centralized configuration parameters in Section 1.2/1.3 of the notebook. Verify `QWK_WARMUP_EPOCHS = 15`, `TARGET_QWK_WEIGHT = 0.4`, `PATIENCE_EARLY_STOPPING = 15`, and that early stopping is strictly bound to `val_QWK`.

**Acceptance Scenarios**:
1. **Given** model training initialization, **When** hyperparameters are parsed, **Then** `QWK_WARMUP_EPOCHS` is set to 15, `TARGET_QWK_WEIGHT` is set to 0.4, and `PATIENCE_EARLY_STOPPING` is set to 15–20.
2. **Given** epoch evaluation in `DRTrainer`, **When** early stopping is checked, **Then** `val_QWK` is the sole governing metric for both checkpoint saving and early stopping decisions.

---

## Edge Cases & Mitigation Strategies

| Edge Case | Impact | Mitigation Strategy |
| :--- | :--- | :--- |
| **Empty / Degenerate FoV Mask in IDRiD** | Bounding box coordinates $[y_0, y_1, x_0, x_1]$ become invalid or inverted. | Fallback to full image dimensions $[0:H, 0:W]$ if `np.where` detects zero foreground pixels. |
| **IDRiD Image Without Lesion Mask** | Missing ground-truth file causes crash during XAI evaluation. | Filter sample list during dataset initialization (`len(samples)` check); skip unpaired images gracefully. |
| **Zero Ground-Truth Lesion Area** | Area ratio $q = 0$, causing division by zero in Area-Matched Recall. | Impose minimum threshold clamp $q = \max(q, 10^{-4})$ and return 0.0 recall if mask has zero positive pixels. |
| **Uniform Attention Weights in MIL** | $\alpha_k = 1/M$ produces flat heatmap without distinct peaks. | Normalize canvas using dynamic epsilon: $(\text{canvas} - \min) / (\max - \min + 10^{-7})$; detect flat gradients in Pointing Game. |
| **Validation Threshold Degeneracy** | Specificity cannot hit exact target (e.g., 95.00%). | Select the closest operating point $\le \text{target\_spec}$ with optimal sensitivity via monotonic ROC interpolation. |

---

## Requirements

### Functional Requirements

- **FR-001**: `IDRiDXAIEvalDataset` MUST compute the retinal Field-of-View bounding box on the raw RGB image using `compute_fov_mask(img_np, erode_px=0)` and apply the identical box slice $[y_0:y_1, x_0:x_1]$ to both the fundus image and its corresponding binary lesion mask before any resizing.
- **FR-002**: Saliency evaluation MUST calculate and display Pointing Game Accuracy alongside the empirical random baseline ($\text{Mask Area} / \text{Total FoV Area}$).
- **FR-003**: Saliency evaluation MUST implement `Saliency Recall @ Area-Matched`, where the continuous saliency map is binarized at the top-$q\%$ quantile ($q$ matching ground-truth mask area ratio) and the proportion of recovered lesion pixels is computed.
- **FR-004**: Saliency evaluation MUST implement `Pixel-level AUROC`, treating continuous heatmap pixel values as prediction scores and binary lesion mask pixels as ground-truth labels.
- **FR-005**: Raw Mean IoU MUST NOT be presented as a standalone failure metric; if reported, it MUST include explanatory documentation regarding the $\approx 0.05$ theoretical ceiling for micro-lesions.
- **FR-006**: `FundusDualBranchDataset` and its collate routines (`train_collate_fn`, `eval_collate_fn`) MUST forward spatial patch anchor coordinates $[y_0, x_0]$ in the batch output dictionary.
- **FR-007**: `GatedAttentionPool` MUST expose or return its patch-level attention weights $\boldsymbol{\alpha} \in \mathbb{R}^{B \times M \times 1}$.
- **FR-008**: The system MUST provide an `AttentionMILSaliencyGenerator` that projects patch attention weights $\alpha_k$ onto a $512 \times 512$ canvas using patch coordinates $[y_0, x_0]$ and patch size $128 \times 128$.
- **FR-009**: The evaluation suite MUST provide a comparative analysis between Global Grad-CAM (ResNet-50 Layer 4) and Local Attention-MIL across the IDRiD lesion mask benchmark.
- **FR-010**: Binary Referable DR classification ($y \ge 2$) MUST NOT be derived from nominal class `argmax`.
- **FR-011**: Decision thresholds for Referable DR screening MUST be fitted strictly on Validation set predicted continuous probabilities $P(y \ge 2)$.
- **FR-012**: Calibrated operating thresholds MUST be evaluated on the held-out Test set, reporting exact Test Sensitivities at target specificities of $95\%$, $90\%$, $85\%$, and $80\%$.
- **FR-013**: The notebook configuration MUST update default training parameters: `QWK_WARMUP_EPOCHS = 15`, `TARGET_QWK_WEIGHT = 0.4`, `PATIENCE_EARLY_STOPPING = 15`.
- **FR-014**: Both `EarlyStopping` and `ModelCheckpoint` MUST monitor the identical validation metric (`Validation QWK`).
- **FR-015**: All updates MUST be applied using minimal-scope deltas, preserving existing model architectures, function signatures, and directory paths.

---

## Key Entities & Data Contracts

- **IDRiDSample**:
  - `image`: Tensor $[3, 512, 512]$, uint8 normalized on GPU, cropped with retinal FoV bounding box.
  - `mask`: Tensor $[1, 512, 512]$, binary float $\{0.0, 1.0\}$, cropped with identical retinal FoV bounding box.
  - `patient_id`: String identifier matching IDRiD record.
- **PatchAnchorBatch**:
  - `patches`: Tensor $[B, M, 3, 128, 128]$.
  - `patch_mask`: Tensor $[B, M]$, boolean indicator of valid non-padded patches.
  - `anchors`: Tensor $[B, M, 2]$, spatial coordinates $[y_0, x_0]$ in image space.
- **ScreeningCalibrationTable**:
  - `target_specificity`: Float $\in \{0.95, 0.90, 0.85, 0.80\}$.
  - `val_calibrated_threshold`: Float $\tau \in [0, 1]$.
  - `test_sensitivity`: Float $\in [0, 1]$.
  - `test_specificity`: Float $\in [0, 1]$.
  - `test_ref_auc`: Float (overall ROC-AUC).

---

## Measurable Success Criteria

- **SC-001**: $100\%$ of IDRiD image-mask evaluation pairs share identical bounding box crops, eliminating scale shift between training and XAI evaluation.
- **SC-002**: XAI evaluation outputs Pointing Game Accuracy with random baseline, Area-Matched Saliency Recall, and Pixel-level AUROC for all evaluated samples without runtime exceptions.
- **SC-003**: Local Attention-MIL produces visual saliency heatmaps with effective spatial resolution of $\approx 128\text{ px}$ (compared to $\approx 268\text{ px}$ for layer4 Grad-CAM).
- **SC-004**: Referable DR screening metrics are decoupled from `argmax` and reported at operating specificities $95\%$, $90\%$, $85\%$, and $80\%$ on the test set.
- **SC-005**: Notebook AST syntax validation passes with $100\%$ zero errors.
- **SC-006**: Existing model architecture definitions, 6656-d multimodal fusion layer, and training loops remain functionally non-regressive under Constitution v1.1.0.

---

## Assumptions & Scope Boundaries

- **Validation / Test Decoupling**: Threshold calibration is strictly fitted on Validation data; under no circumstances are thresholds tuned on the Test set.
- **Grad-CAM Layer Target**: Layer 4 of ResNet-50 remains the primary global Grad-CAM target layer; Attention-MIL serves as the complementary high-resolution local XAI stream.
- **Backward Compatibility**: Both `main_model (4).ipynb` and `main_model.ipynb` will be maintained in synchronized harmony with these enhancements.
