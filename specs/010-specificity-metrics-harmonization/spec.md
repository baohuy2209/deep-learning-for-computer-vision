# Feature Specification: Clinical Specificity-Sensitivity Harmonization & Colab Reference Code Sync

**Feature Branch**: `010-specificity-metrics-harmonization`  
**Feature Directory**: `specs/010-specificity-metrics-harmonization`  
**Created**: 2026-09-13  
**Status**: Ready for Planning / Review  
**Input**: User request to audit experiments so they output all clinical specificity points (specifically Target Specificity 85% $\to$ expected sensitivity 79.6%, and Target Specificity 80% $\to$ expected sensitivity 84.7%, alongside 95% and 90%) instead of only 95%, and synchronize the repository notebooks (`main_model.ipynb`, `main_model (4).ipynb`, `main_model_head_ablation.ipynb`, `main_model_head_ablation (2).ipynb`) with the Colab-verified reference files (`d:\Downloads\main_model (8).ipynb` and `main_model_head_ablation (3).ipynb`) under thesis advisor directives in `documents/HD_1.pdf`, `documents/HD_2.pdf`, `documents/HD_3.pdf`, `documents/HD_4.pdf`.

---

## Executive Summary & Background Context

In ophthalmological screening for Referable Diabetic Retinopathy (DR, defined as Grade $\ge 2$), clinical deployment requires operating at pre-calibrated trade-offs between Sensitivity (Recall of referral-warranted pathology) and Specificity (true negative rate to prevent clinic over-referral).

As proven in **Advisor Review `HD_4.pdf` (Part 1 & 3)**:
1. **The Root Cause**: The primary dual-branch model and head ablation benchmark achieve a high continuous discriminative capacity ($\text{Test Ref-AUC} \approx 0.9063$). When evaluating screening performance using the nominal 5-class `argmax`, the implicit decision boundary sits at an extremely conservative specificity ($\approx 95\%$), yielding a sub-optimal referable sensitivity of $58.7\% - 64.8\%$.
2. **Theoretical & Empirical Calibration Spectrum**: By modeling continuous referable risk scores $P(y \ge 2)$ via post-hoc threshold calibration on the Validation set (binormal model $d' = 1.864$ derived from $\text{Ref-AUC} = 0.9063$), the clinical operating spectrum is established:
   - **Target Specificity 95%**: Expected Sensitivity $\approx 58.7\% - 63\%$ (current nominal `argmax` neighborhood, sub-optimal for primary screening).
   - **Target Specificity 90%**: Expected Sensitivity $\approx 72.0\%$.
   - **Target Specificity 85%**: Expected Sensitivity $\approx 79.6\%$ (approaching the $80\%$ clinical screening benchmark).
   - **Target Specificity 80%**: Expected Sensitivity $\approx 84.7\%$ (surpassing the $80\%$ clinical screening target).
   - **Target Specificity 75%**: Expected Sensitivity $\approx 88.3\%$.
3. **Current Problem in Codebase**:
   - The current code and loggers frequently report **only** `Ref_Sens_Spec95` or `Ref-Sens@Spec95` in console logs, epoch histories, ROC plot callouts, and summary tables.
   - In `compute_clinical_metrics`, the default `target_specificities` parameter in several notebook cells was set to `(0.80, 0.90, 0.95)`, omitting `0.85` entirely.
   - The summary table printed in head ablation (`df_heads`) does not include the calibrated sensitivity columns across the specificity spectrum (`Sens@Spec95`, `Sens@Spec90`, `Sens@Spec85`, `Sens@Spec80`).
   - Discrepancies exist between Colab-executed files (`main_model (8).ipynb` and `main_model_head_ablation (3).ipynb`) and repo versions (`main_model.ipynb`, `main_model (4).ipynb`, `main_model_head_ablation.ipynb`, `main_model_head_ablation (2).ipynb`), including cell counts (36 vs 27 vs 19 cells in head ablation) and logging consistency.

---

## User Scenarios & Prioritized User Stories

### User Story 1 - Full Specificity Spectrum Metric Output & Logging (Priority: P1)
As a clinical machine learning researcher, I need the training and evaluation routines across both `main_model` and `main_model_head_ablation` to compute, log, and print Referable DR Sensitivity across all standardized Target Specificity operating points ($95\%$, $90\%$, $85\%$, $80\%$, $75\%$), so that the $85\% \to 79.6\%$ and $80\% \to 84.7\%$ operating milestones are explicitly visible alongside $95\%$ in all logs, tables, and reports.

**Why this priority**:
- Directly addresses the user's primary prompt. Without reporting $85\%$ and $80\%$, thesis readers and clinical evaluators mistakenly assume the model's screening sensitivity is capped at the $95\%$ specificity operating point (~$60\%$), obscuring the model's true clinical utility where sensitivity reaches $84.7\%$ at $80\%$ specificity.

**Independent Test**:
- Execute `compute_clinical_metrics` and `DRTrainer.evaluate()` on validation and test sets. Verify that the output metrics dictionary and console logs display:
  - `Sens@Spec95`: Value in % (expected ~58.7% - 63%)
  - `Sens@Spec90`: Value in % (expected ~72.0%)
  - `Sens@Spec85`: Value in % (expected ~79.6%, near 80% target)
  - `Sens@Spec80`: Value in % (expected ~84.7%, exceeds 80% target)
  - Test calibration report matches the binormal trajectory outlined in `HD_4`.

**Acceptance Scenarios**:
1. **Given** model probability outputs $P(y \ge 2)$ on Validation and Test sets, **When** `compute_clinical_metrics` is invoked, **Then** `target_specificities` includes `(0.80, 0.85, 0.90, 0.95)`, returning explicit keys `Ref_Sensitivity_at_Spec80`, `Ref_Sensitivity_at_Spec85`, `Ref_Sensitivity_at_Spec90`, and `Ref_Sensitivity_at_Spec95`.
2. **Given** epoch training and test evaluation in `DRTrainer`, **When** progress and final metrics are logged, **Then** the log string includes `Test Ref-Sens@Spec95`, `Test Ref-Sens@Spec85`, and `Test Ref-Sens@Spec80` (e.g. `Ref-Sens@Spec85: 79.6% | Ref-Sens@Spec80: 84.7%`).
3. **Given** `print_per_grade_report` or evaluation summary functions, **When** executed, **Then** a dedicated Clinical Screening Calibration section prints the empirical vs expected sensitivity table comparing operating specificities ($95\%$, $90\%$, $85\%$, $80\%$).

---

### User Story 2 - Benchmark Summary Table & Head Ablation Synchronization (Priority: P1)
As an author writing the thesis experimental results section, I need the 5-round head ablation summary table (`df_heads`) and per-head console outputs to include all calibrated screening sensitivities (`Sens@Spec95`, `Sens@Spec90`, `Sens@Spec85`, `Sens@Spec80`) and UK NSC compliance status, and I need `main_model_head_ablation.ipynb` in the repo to be synchronized with the full 36-cell Colab-verified implementation (`main_model_head_ablation (3).ipynb`).

**Why this priority**:
- `main_model_head_ablation (3).ipynb` contains the full 36-cell benchmark with auto-resume checkpointing, multi-metric bar charts, latent cutpoints visualization (Figure 6), and head calibration comparisons. The repository's `main_model_head_ablation.ipynb` has only 19 cells, omitting critical publication figures and resume capabilities required by HD_1, HD_2, and HD_4.

**Independent Test**:
- Inspect the columns of `df_heads` in the head ablation summary. Verify `Sens@Spec95`, `Sens@Spec90`, `Sens@Spec85`, and `Sens@Spec80` appear in the printed table and exported CSV. Verify the code structure and cell sequence in `main_model_head_ablation.ipynb` match `main_model_head_ablation (3).ipynb`.

**Acceptance Scenarios**:
1. **Given** completion of the 5 head ablation rounds (`Softmax_CE`, `Softmax_QWK`, `CORAL`, `CORN`, `CLM_QWK`), **When** `df_heads` is printed to stdout and saved to `head_ablation_summary.csv`, **Then** the display table includes columns: `Head`, `Val_QWK`, `Test_QWK`, `Generalization_Gap`, `Within_1_Grade_Acc`, `Sens@Spec95`, `Sens@Spec85`, `Sens@Spec80`, `Referable_AUC`, `Monotonicity_Violations`.
2. **Given** `notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation.ipynb` and `(2).ipynb`, **When** updated, **Then** their cell structure, resume logic, figure generation, and evaluation routines are fully aligned with `main_model_head_ablation (3).ipynb`.

---

### User Story 3 - Visual Calibration & ROC Curve Multi-Point Annotation (Priority: P2)
As a researcher reading publication figures, I need Figure 7 (Referable DR ROC curves in `main_model` and `main_model_head_ablation`) to display markers and text callouts at multiple clinical operating points ($\text{FPR} = 0.05 \implies \text{Spec} = 95\%$, $\text{FPR} = 0.15 \implies \text{Spec} = 85\%$, and $\text{FPR} = 0.20 \implies \text{Spec} = 80\%$), so that the graphical presentation directly substantiates the $79.6\%$ and $84.7\%$ sensitivity claims.

**Why this priority**:
- Figure 7 currently only draws a single vertical line at $\text{FPR} = 0.05$ ($95\%$ specificity). Adding markers and annotations for $85\%$ and $80\%$ specificity provides immediate visual confirmation that moving the operating point along the ROC curve achieves $>80\%$ sensitivity.

**Independent Test**:
- Call `plot_fig7_roc` with validation and test data. Check that the resulting plot contains annotations for Specificity 95%, Specificity 85%, and Specificity 80% with corresponding TPR percentages.

**Acceptance Scenarios**:
1. **Given** validation and test ROC data, **When** `plot_fig7_roc` is executed, **Then** scatter points and informative text callouts are rendered for $\text{FPR} \in \{0.05, 0.15, 0.20\}$, denoting `Spec 95%`, `Spec 85% (Sens: ~79.6%)`, and `Spec 80% (Sens: ~84.7%)`.
2. **Given** the generated ROC figure, **When** saved to `FIGURES_DIR`, **Then** the figure cleanly illustrates the operating point trade-offs without visual clutter.

---

### User Story 4 - Pipeline Code & Methodological Consistency Alignment (HD_1 to HD_4) (Priority: P2)
As a machine learning engineer maintaining reproducible research pipelines, I need `main_model.ipynb` and `main_model (4).ipynb` to maintain strict code logic parity with the latest Colab execution file `d:\Downloads\main_model (8).ipynb`, incorporating all directives from `HD_1` through `HD_4`.

**Why this priority**:
- Ensures that code executed locally or on remote servers reproduces the exact data flow, GPU uint8 normalization, QWK loss warmup ($0.0 \to 0.4$ over 15 epochs), dual-crop IDRiD XAI with FoV bounding boxes, and validation QWK early stopping established across the thesis guidance documents.

**Independent Test**:
- Run structural diff checks between `d:\Downloads\main_model (8).ipynb` and `notebooks/diabetic_retinopathy/pipeline/model/main_model.ipynb`. Confirm that all 92 cells match in architecture, execution flow, preprocessing, and evaluation logic.

**Acceptance Scenarios**:
1. **Given** `d:\Downloads\main_model (8).ipynb` as the ground-truth Colab execution file, **When** synchronized into `main_model.ipynb` and `main_model (4).ipynb`, **Then** all training parameters (`TARGET_QWK_WEIGHT = 0.4`, `QWK_WARMUP_EPOCHS = 15`), evaluation routines, and data loading mechanisms match identically.
2. **Given** the thesis advisor guidance in `HD_1` (Ordinal heads), `HD_2` (patient leakage fix, natural test prevalence), `HD_3` (raw RGB FoV mask, GPU uint8 norm, eval collate for val), and `HD_4` (post-hoc threshold calibration, XAI dual crop, honest UK NSC limitation reporting), **When** verifying notebook cells, **Then** all these requirements are consistently fulfilled across all notebooks.

---

## Edge Cases & Mitigation Strategies

| Edge Case | Impact | Mitigation Strategy |
| :--- | :--- | :--- |
| **Small Test Set or Tied Probabilities** | In discrete test sets, an exact specificity of $85.00\%$ or $80.00\%$ may not exist on the ROC curve. | In `calculate_referable_sensitivity_at_specificity`, select the operating threshold with $\text{specificity} \ge \text{target\_spec}$ that maximizes sensitivity: `valid_idx = np.where(specificity >= spec)[0]`; fallback gracefully to `0.0` sensitivity if no sample satisfies the gate. |
| **Calibration Set Mismatch (Val vs Test)** | Applying validation-fitted thresholds to the test set might result in a slightly different empirical test specificity (e.g. $83.5\%$ instead of $85.0\%$). | As directed by `HD_4 Part 3`, report both the validation-calibrated operating threshold and the resulting empirical test specificity and test sensitivity in `calibrate_and_evaluate_screening`, noting the generalization behavior. |
| **ROC Interpolation at Boundary** | In smoke tests or small validation batches, $\text{FPR} < 0.05$ or $0.15$ may have few discrete points. | Use `np.argmin(np.abs(roc["fpr"] - target_fpr))` with index bounds checking so plotting functions never raise `IndexError`. |
| **Notebook File Synchronization Conflict** | Overwriting existing notebooks might lose local modifications if not carefully checked. | Strictly compare AST/cell content between Colab files (`main_model (8)` and `head_ablation (3)`) and target repo notebooks; verify all cell changes preserve clean code principles and zero regressions. |

---

## Requirements

### Functional Requirements

- **FR-001**: System MUST compute and return Referable DR sensitivity for all target specificities in `(0.80, 0.85, 0.90, 0.95)` in `calculate_referable_sensitivity_at_specificity` and `compute_clinical_metrics`.
- **FR-002**: System MUST include `Ref_Sensitivity_at_Spec85` and `Ref_Sensitivity_at_Spec80` alongside `Ref_Sensitivity_at_Spec95` and `Ref_Sensitivity_at_Spec90` in evaluation metric dictionaries for both validation and held-out test sets.
- **FR-003**: System MUST print and log sensitivity at Target Specificities 95%, 90%, 85%, and 80% during epoch logging and final test evaluation in `DRTrainer`.
- **FR-004**: System MUST display the expected theoretical sensitivity milestones ($85\% \to 79.6\%$, $80\% \to 84.7\%$, $95\% \to 58.7\%$) in the formatted clinical screening report to provide clear clinical context per `HD_4`.
- **FR-005**: System MUST include `Sens@Spec95`, `Sens@Spec90`, `Sens@Spec85`, and `Sens@Spec80` columns in the printed head ablation summary DataFrame (`df_heads`) and in the exported `head_ablation_summary.csv`.
- **FR-006**: In Figure 7 (Referable DR ROC curves), system MUST display markers and text callouts at Specificity 95% ($\text{FPR}=0.05$), Specificity 85% ($\text{FPR}=0.15$), and Specificity 80% ($\text{FPR}=0.20$).
- **FR-007**: System MUST synchronize `main_model.ipynb` and `main_model (4).ipynb` with `d:\Downloads\main_model (8).ipynb` to ensure complete parity with the Colab-verified execution pipeline.
- **FR-008**: System MUST synchronize `main_model_head_ablation.ipynb` and `main_model_head_ablation (2).ipynb` with `main_model_head_ablation (3).ipynb` so all 36 cells (including training resumption, visualization, and post-hoc threshold calibration across all 5 heads) are present and consistent.
- **FR-009**: System MUST clearly document the UK NSC screening standard alignment and limitation: achieving $\text{Sensitivity} \ge 80\%$ and $\text{Specificity} \ge 95\%$ simultaneously requires $\text{Ref-AUC} \ge 0.961$, whereas the current model reaches $\text{Ref-AUC} \approx 0.906$, reaching $84.7\%$ sensitivity at $80\%$ specificity, which must be reported honestly as an operational limitation per `HD_4 Part 3`.

### Key Entities

- **Target Specificity Operating Point**: Standardized screening specificity requirement (e.g. $80\%$, $85\%$, $90\%$, $95\%$) determining the clinical referral cutoff threshold $\tau$.
- **Clinical Sensitivity / Recall**: Proportion of true referable DR cases (Grade $\ge 2$) correctly flagged by the model at a given specificity.
- **Binormal ROC Model ($d'$ = 1.864)**: Theoretical discrimination index derived from $\text{AUC} = 0.9063$, establishing expected sensitivity values for given operating points ($85\% \to 79.6\%$, $80\% \to 84.7\%$).
- **Screening Calibration Report**: Standardized tabular summary exported to CSV and printed to stdout comparing validation and test operating thresholds, sensitivities, specificities, and UK NSC status.

---

## Success Criteria

### Measurable Outcomes

- **SC-001**: 100% of evaluation routines in `main_model` and `main_model_head_ablation` output sensitivities for Target Specificity 80%, 85%, 90%, and 95% without omissions.
- **SC-002**: In both console stdout and saved CSV files (`head_ablation_summary.csv`, `screening_calibration_summary.csv`, `*_screening_calibration.csv`), columns for `Sens@Spec85` and `Sens@Spec80` are populated with valid numerical percentages matching empirical test distributions.
- **SC-003**: In `df_heads` output table in head ablation benchmarks, all 5 heads (`Softmax_CE`, `Softmax_QWK`, `CORAL`, `CORN`, `CLM_QWK`) display their respective sensitivity at $85\%$ and $80\%$ specificity.
- **SC-004**: ROC curve visualizations (Figure 7) render distinct markers and text annotations for Specificity 95%, 85%, and 80%.
- **SC-005**: Full code parity achieved across notebook pairs (`main_model.ipynb` $\equiv$ `main_model (4).ipynb` $\equiv$ `main_model (8).ipynb`, and `main_model_head_ablation.ipynb` $\equiv$ `main_model_head_ablation (2).ipynb` $\equiv$ `main_model_head_ablation (3).ipynb`).

---

## Assumptions & Dependencies

- **A-001**: `d:\Downloads\main_model (8).ipynb` and `notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation (3).ipynb` represent the definitive ground-truth implementations tested on Colab GPUs.
- **A-002**: Threshold calibration is performed post-hoc using predicted probabilities $P(y \ge 2)$ on the validation set, requiring 0 additional GPU training epochs per `HD_4`.
- **A-003**: The dataset path references (`/content/dataset_local` or Google Drive paths) remain compatible with Colab execution environments while supporting local relative paths when executed locally.
- **A-004**: Constitution v1.1.0 directives (strict preservation of proven architectures, delta-only modifications, reproducible seeds, ordinal QWK formulation) remain strictly enforced.
