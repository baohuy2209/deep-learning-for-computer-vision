# Feature Specification: 6-Fold Leave-One-Dataset-Out (LODO) Cross-Domain Generalization Benchmark

**Feature Branch**: `011-dr-lodo-benchmark`  
**Feature Directory**: `specs/011-dr-lodo-benchmark`  
**Created**: 2026-09-15  
**Status**: Ready for Planning / Review  
**Input**: User request: "/speckit-specify notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb - Validate and complete the 6-fold LODO benchmark before execution: exact dataset unzipping from Google Drive without fallback (as in main_model (6).ipynb), LODO protocol with patient-level splitting and per-fold metadata CSV exports (following merge-dataset.ipynb), full alignment with thesis advisor directives (HD_1, HD_2, HD_3, HD_4), 3-way preprocessing caching (train/val fast NPZ cache, test cached from raw and zipped to Google Drive), and intelligent checkpoint resumption with completed-fold skipping to prevent redundant training and save Colab Compute Units (CU)."

---

## Executive Summary & Alignment with Research Advisor Directives

The central research question formulated in **`HD_1.pdf`** establishes:
> *"Thành phần nào của mô hình chấm điểm DR thực sự SỐNG SÓT qua domain shift?"*  
> *(Which architectural components of the DR grading model truly survive domain shift?)*

Standard internal validation benchmarks in Diabetic Retinopathy typically pool multiple datasets together and perform a random $70/15/15\%$ split. While effective for internal testing, this severely masks real-world domain collapse (as demonstrated by Hashmi et al., where QWK drops precipitously from $0.88 \to 0.68$ on external cohorts).

To provide definitive, publication-worthy empirical proof for the three core research hypotheses:
- **$H_1$ (Local MIL Branch rescues Grade 1 under resolution collapse)**: Investigated by decomposing per-grade recall on unseen external datasets, specifically verifying if Grade 1 sensitivity is preserved through native-resolution sliding-window patch extraction.
- **$H_2$ (Global Non-Local & Quadrant Tokens rescue Grade 3 under context loss)**: Investigated by quantifying Grade 3 sensitivity across varied clinical camera distributions, verifying that 4-quadrant Transformer relational encoding (ICDR 4-2-1 rule) prevents misclassification.
- **$H_3$ (Cumulative Link Model / CLM with Continuous QWK Loss suppresses rank monotonicity inversions and closes the generalization gap)**: Quantified by auditing rank probability monotonicity violation rates and measuring the headline generalization gap metric:
  $$\Delta_{\text{LODO}} = \text{QWK}_{\text{In-Domain Val}} - \text{QWK}_{\text{OOD Test}}$$

This feature updates and perfects `main_model_lodo.ipynb` into a production-grade, Google Colab-optimized, resilient execution pipeline adhering strictly to the user's constraints and advisor guidelines (`HD_1`, `HD_2`, `HD_3`, `HD_4`).

---

## User Scenarios & Prioritized User Stories

### User Story 1 - Exact Drive Unzipping & Zero-Leakage 6-Fold Metadata Export (Priority: P1) 🎯 MVP
As a computer vision researcher running experiments on Google Colab, I need the notebook to automatically mount Google Drive and decompress the preprocessed dataset archive (`/content/drive/MyDrive/native_cropped.zip`) directly to local fast SSD (`/content/dataset_local`) using the exact path from `main_model (6).ipynb` without any dummy or synthetic data fallbacks, and execute the 6-fold LODO protocol where each cohort is held out sequentially as a 100% unseen test set with natural clinical prevalence while the remaining 5 cohorts are split at patient level (85% train, 15% val) with quota balancing applied strictly to train, exporting dedicated metadata CSV files for each fold (`folds/fold_{k}_{cohort}_train.csv`, `val.csv`, `test.csv`) following the logic of `merge-dataset.ipynb`.

**Why this priority**:
- `HD_2 Part 1 & 2` uncovered critical methodological flaws in standard merging:
  1. Entire cohorts (e.g. all 516 IDRiD images) were grouped into a single patient ID.
  2. Test sets were artificially balanced, distorting natural clinical prevalence.
  3. Preprocessing was coupled to splits, requiring 1.5 hours of re-cropping per fold.
- The user explicitly mandated: "lấy path chuẩn luôn không fallback gì hết á" (use exact standard paths, no synthetic fallbacks).
- Saving per-fold metadata CSVs provides full provenance, reproducibility, and auditability for peer review.

**Independent Test**:
- Run dataset path verification: Assert `/content/drive/MyDrive/native_cropped.zip` is decompressed to `/content/dataset_local`.
- Run patient overlap verification on exported fold CSVs:
  $$\text{Patients}(\text{Train}) \cap \text{Patients}(\text{Val}) = \emptyset$$
  $$\text{Patients}(\text{Train} \cup \text{Val}) \cap \text{Patients}(\text{Held-Out OOD Test}) = \emptyset$$
- Confirm that `test.csv` has unaltered natural prevalence and `train.csv` has balanced class quotas across sources.

**Acceptance Scenarios**:
1. **Given** Google Drive mounted at `/content/drive`, **When** Section 1 executes, **Then** `/content/drive/MyDrive/native_cropped.zip` is decompressed to `/content/dataset_local` if not already present, with exact status logging and zero synthetic fallback.
2. **Given** 6 clinical fundus cohorts loaded into the master dataframe, **When** fold $k$ is initialized, **Then** cohort $k$ is designated as the OOD Test set, and the remaining 5 cohorts are grouped by `patient_id` using `GroupShuffleSplit(train_size=0.85)` into `train_raw` and `val_raw`.
3. **Given** `train_raw` containing samples from 5 sources, **When** training balancing is applied, **Then** `create_balanced_training_pool` applies multi-source quota balancing *only* on `train_raw`, leaving `val_raw` and `ood_test_df` with their natural clinical prevalence intact.
4. **Given** each fold $k$, **When** splitting finishes, **Then** `folds/fold_{k}_{cohort}_train.csv`, `folds/fold_{k}_{cohort}_val.csv`, and `folds/fold_{k}_{cohort}_test.csv` are written to disk.

---

### User Story 2 - Intelligent Checkpoint Resumption & Compute Unit (CU) Protection (Priority: P1) 🎯 MVP
As a deep learning researcher with finite Google Colab Compute Units (CU), I need each LODO fold to track its full completion status (`is_completed = True` in `_last.pth` and `lodo_summary.json`), so that if a fold has already finished training and evaluation, re-executing or reconnecting the notebook will immediately recognize the completed fold, restore its evaluation metrics, and advance directly to the next fold without re-training from the best model epoch, while still allowing interrupted folds to resume from their exact last epoch.

**Why this priority**:
- In `main_model (6).ipynb`, loading `_best.pth` restores `epoch = best_epoch + 1`. If the best model was saved at epoch 8 of 15 (or 25 of 100), reloading causes the trainer to train for the remaining epochs even if the fold had already reached full convergence.
- In a 6-fold LODO benchmark, retraining completed folds wastes massive amounts of GPU compute and can cause timeout disconnects.

**Independent Test**:
- Complete Fold 1 training and evaluation. Stop execution. Re-run Section 6.
- Verify that the system logs `[SKIP] Fold 1 (APTOS) already completed. Restoring cached results and proceeding to Fold 2.` without invoking `train_epoch`.

**Acceptance Scenarios**:
1. **Given** an interrupted fold at epoch $e < E$, **When** resumed, **Then** training resumes from epoch $e + 1$ using `_last.pth` weights, optimizer, scaler, and scheduler states.
2. **Given** a fold that reached $E$ epochs or triggered early stopping, **When** completed, **Then** `checkpoint_state["is_completed"] = True` is saved in `_last.pth` and `_best.pth`, and fold metrics are recorded in `lodo_summary.json`.
3. **Given** a completed fold on disk, **When** the LODO loop runs, **Then** the training loop is bypassed entirely, test metrics are validated from disk, and execution seamlessly proceeds to the next incomplete fold.

---

### User Story 3 - 3-Way Preprocessing Caching with Google Drive Archival (Priority: P2)
As a researcher running multiple folds across sessions, I need a high-performance 3-way data caching pipeline where Train and In-Domain Validation sets utilize local fast `.npz` caching (uint8 Ben Graham image + valid anchor coordinates), while the Held-Out Test cohort is precomputed from raw data, compressed into a zip archive, and automatically persisted to Google Drive (`/content/drive/MyDrive/.../test_cache_{cohort}.zip`), so that future test evaluations load instantaneously without re-computing FOV masks or Ben Graham filtering.

**Why this priority**:
- CPU processing (Otsu FOV mask + bilateral filter + Ben Graham + sliding window anchors) takes $\approx 0.81\text{s}$ per image on single-thread CPU (`HD_3 Page 3`).
- Over thousands of images, calculating this on-the-fly consumes CPU time and causes GPU starvation.
- The held-out test cohort never changes between experiments; saving its precomputed `.npz` cache to Google Drive enables instant re-evaluation across different runs or model heads.

**Independent Test**:
- Run the test cache routine on the held-out cohort. Verify that `.npz` files are generated, zipped, and saved to Google Drive.
- Delete the local test cache and verify that the pipeline automatically restores the archive from Google Drive.

**Acceptance Scenarios**:
1. **Given** training and validation datasets, **When** batches are loaded, **Then** `FundusDualBranchDataset` checks for `.npz` files in `CACHE_DIR_LODO`, loading `img` (uint8) and `anchors` in $< 20\text{ms}$ per image.
2. **Given** a held-out test cohort, **When** initialized for fold $k$, **Then** the system checks if `test_cache_{cohort}.zip` exists on Google Drive. If found, it unzips into local cache; if not found, it generates the cache from raw fundus images, zips the folder, and uploads it to Google Drive.

---

### User Story 4 - Full Specificity Spectrum & Post-Hoc Threshold Calibration in LODO (Priority: P2)
As an ophthalmologist evaluating screening safety across diverse clinical populations, I need each LODO fold to evaluate Referable DR screening sensitivity across the standardized specificity spectrum ($95\%$, $90\%$, $85\%$ [expected $79.6\%$], $80\%$ [expected $84.7\%$]), and calibrate decision thresholds post-hoc on the in-domain validation set before testing on the unseen external cohort, reporting UK NSC status and operational trade-offs per `HD_4 Part 1 & 3`.

**Why this priority**:
- `HD_4 Part 1 & 3` emphasizes: screening evaluation must never rely on nominal `argmax`.
- Evaluating calibrated thresholds on external cohorts reveals whether decision boundaries generalize across different camera optics and patient demographics.

**Independent Test**:
- For each fold, inspect the generated `lodo_results_summary.csv` and console logs. Confirm that `Ref-Sens@Spec95`, `Ref-Sens@Spec90`, `Ref-Sens@Spec85`, and `Ref-Sens@Spec80` are reported for both in-domain validation and out-of-domain test sets.

**Acceptance Scenarios**:
1. **Given** trained dual-branch model weights for fold $k$, **When** evaluated on the in-domain validation set, **Then** optimal thresholds $\tau$ are identified meeting target specificities of $95\%$, $90\%$, $85\%$, and $80\%$.
2. **Given** validation-calibrated thresholds $\tau$, **When** applied to the held-out unseen cohort, **Then** generalization sensitivity and specificity are measured and recorded in `df_lodo`.

---

### User Story 5 - Hypothesis Decomposition & Domain Collapse Quantification (Priority: P2)
As a lead author drafting the paper discussion, I need the LODO benchmark to compute the headline domain gap $\Delta_{\text{LODO}} = \text{QWK}_{\text{In-Domain}} - \text{QWK}_{\text{OOD}}$, calculate per-grade sensitivity drops ($\Delta \text{Recall}_0 \dots \Delta \text{Recall}_4$), and audit rank probability monotonicity violations across all 6 folds, so that Hypotheses $H_1$, $H_2$, and $H_3$ can be rigorously evaluated with quantitative empirical data.

**Why this priority**:
- `HD_1.pdf` notes: *"Metric chính không phải QWK nội bộ, mà là $\Delta = (\text{nội bộ} - \text{ngoại miền})$. Toàn bộ luận điểm H1/H2 nằm ở grade 1 và grade 3. Chỉ báo QWK tổng thì không chứng minh được gì."*
- Decomposing performance drops by grade proves whether domain collapse is concentrated in mild microaneurysms (Grade 1 / $H_1$) or severe hemorrhages/exudates (Grade 3 / $H_2$).

**Independent Test**:
- Inspect the output summary tables and generated figures. Confirm that $\Delta_{\text{LODO}}$, per-grade recall breakdown, and monotonicity violations are computed for each fold and summarized across all 6 folds (Mean $\pm$ Std).

**Acceptance Scenarios**:
1. **Given** evaluation results on in-domain validation and held-out test splits, **When** summarizing fold metrics, **Then** $\Delta_{\text{LODO}} = \text{Val\_QWK} - \text{Test\_QWK}$ is recorded.
2. **Given** confusion matrices for each fold, **When** per-grade recalls are extracted, **Then** individual recall drops for Grade 1 and Grade 3 are explicitly logged to substantiate Hypotheses $H_1$ and $H_2$.
3. **Given** predicted ordinal cumulative link outputs, **When** monotonicity is audited, **Then** the violation rate is verified on out-of-domain test sets (confirming $H_3$).

---

### User Story 6 - Publication Figure Suite Generation (Priority: P3)
As a researcher finalizing the manuscript for submission, I need the benchmark to render and save the complete 8-figure publication visualization suite ($300\text{ DPI}$) to `FIGURES_DIR` upon benchmark completion.

**Why this priority**:
- Visual evidence (Generalization Radar, $\Delta_{\text{QWK}}$ Waterfall, ROC Curves, Latent Geometry, Grad-CAM Saliency) provides the empirical core for the thesis chapter and journal submission.

**Independent Test**:
- Verify that Figures 1 through 8 are exported to `FIGURES_DIR` as high-resolution PNGs.

---

## Edge Cases & Mitigation Strategies

| Edge Case | Impact | Mitigation Strategy |
| :--- | :--- | :--- |
| **Missing Google Drive Zip on Colab** | Pipeline cannot extract images if `/content/drive/MyDrive/native_cropped.zip` is absent. | Verify `os.path.exists(zip_source)` explicitly; if absent, prompt user with the exact missing path and halt immediately without falling back to synthetic dummy data. |
| **Colab Disconnect Mid-Epoch** | Training interrupted before epoch finishes. | `_last.pth` is saved at the end of every completed epoch; on restart, training resumes from the last fully completed epoch. |
| **Colab Disconnect Between Folds** | Fold $k$ is finished, but notebook disconnected before starting Fold $k+1$. | Checkpoint state records `is_completed = True` and summary JSON records fold results; on restart, finished folds are skipped in 0 seconds. |
| **Grade Imbalance in Small Left-Out Cohorts (e.g. IDRiD)** | IDRiD has only 516 images with sparse Grade 1 or Grade 4 cases; discrete specificity targets may lack exact operating points. | Fallback gracefully using `np.where(specificity >= target_sp)[0]`; if no sample meets the gate, clamp sensitivity to 0.0 and threshold to max value. |
| **Test Set Cache Size on Google Drive** | Zipping large cohorts could consume excessive Drive storage. | Zip only the `.npz` files (which store uint8 images and integer coordinates, taking $\approx 50\text{--}150\text{ MB}$ per cohort) rather than uncompressed raw files. |
| **Single-Patient Inter-Dataset ID Collision** | Patient "001" exists in both IDRiD and EyePACS. | Prepend cohort key in `parse_patient_and_side`: e.g. `f"{dataset_key}::{pid}"`. |
| **OpenCV Multi-Threading Contention** | Default OpenCV worker threads cause thread thrashing on Colab vCPUs. | Call `cv2.setNumThreads(0)` immediately after import cv2 (`HD_3`). |

---

## Requirements

### Functional Requirements

- **FR-001**: System MUST mount Google Drive and decompress `/content/drive/MyDrive/native_cropped.zip` directly to `/content/dataset_local` using the exact path and logic from `main_model (6).ipynb` without fallback to synthetic or dummy data.
- **FR-002**: System MUST support 6-fold Leave-One-Dataset-Out cross-domain evaluation iterating over the 6 clinical cohorts: `aptos2019` (or `APTOS`), `ddr` (or `DDR`), `deepdrid` (or `DeepDRiD`), `idrid` (or `IDRiD`), `messidor2` (or `Messidor-2`), and `eyepacs` (or `EyePACS`).
- **FR-003**: In each fold, system MUST isolate the left-out cohort as the 100% unseen test set with its natural clinical prevalence preserved (`test_raw` without balancing).
- **FR-004**: In each fold, system MUST split the remaining 5 cohorts at the patient level (`GroupShuffleSplit` on `patient_id` with 85% train, 15% val), and apply source-aware balancing (`create_balanced_training_pool`) strictly to the training split.
- **FR-005**: System MUST parse patient IDs and eye sides without leakage across all cohorts (`idrid`, `eyepacs`, `deepdrid`, `ddr`, `messidor`, `aptos`), prefixing IDs with cohort namespaces (`{dataset_key}::{pid}`).
- **FR-006**: In each fold, system MUST save dedicated metadata CSV files (`folds/fold_{k}_{cohort}_train.csv`, `folds/fold_{k}_{cohort}_val.csv`, `folds/fold_{k}_{cohort}_test.csv`) documenting exact image paths, patient IDs, cohort sources, and clinical severity labels for complete auditability and prevalence verification.
- **FR-007**: System MUST implement a 3-way preprocessing cache mechanism:
  - Train and Val sets: Cached locally as `.npz` files (uint8 Ben Graham image + valid anchor coordinates `(N, 2)`).
  - Held-out Test set: Cached from raw fundus data, compressed into a `.zip` file, and persisted to Google Drive (`test_cache_{cohort}.zip`). If already present in Drive, it is loaded/unzipped directly to skip re-computation.
- **FR-008**: System MUST implement an intelligent checkpoint resumption mechanism:
  - Save `_last.pth` at every epoch and `_best.pth` when validation QWK improves.
  - When training completes for a fold, mark `is_completed = True` in the checkpoint and write fold summary to `lodo_summary.json`.
  - When starting a fold, if `is_completed == True`, restore best model weights and evaluation metrics, log a skip notice, and proceed immediately to the next fold without re-training from best epoch or consuming unnecessary Colab CUs.
  - If a fold was interrupted mid-training, resume from `_last.pth` and continue training only the remaining epochs.
- **FR-009**: System MUST compute and output Referable DR Sensitivity for Target Specificities 95%, 90%, 85% (expected 79.6%), and 80% (expected 84.7%) on both in-domain validation and OOD test sets in `calculate_referable_sensitivity_at_specificity` and `DRTrainer`.
- **FR-010**: System MUST perform post-hoc threshold calibration on in-domain validation continuous probabilities $P(y \ge 2)$ and apply calibrated cutoffs $\tau$ to the held-out test cohort, reporting UK NSC status and operational trade-offs per `HD_4 Part 1 & 3`.
- **FR-011**: System MUST record and export per-fold and aggregated cross-domain metrics to `lodo_results_summary.csv` and `lodo_summary.json`, including:
  - `In_Domain_Val_QWK`, `OOD_Test_QWK`, `Delta_LODO`
  - Per-grade recall ($G_0 \dots G_4$) for in-domain and OOD
  - Within-1-Grade accuracy
  - Sensitivities at Specificities 95%, 90%, 85%, 80%
  - Rank monotonicity violation rate
- **FR-012**: System MUST render and save the complete 8-figure publication visualization suite to `FIGURES_DIR`.
- **FR-013**: System MUST incorporate all technical bug fixes from `HD_2` (patient leakage, natural prevalence), `HD_3` (raw RGB FOV mask, GPU uint8 norm, eval collate for val/test, `cv2.setNumThreads(0)`), and `HD_4` (post-hoc calibration, honest UK NSC limitation reporting).
- **FR-014**: Complete Dual-Branch Architecture Ingestion: `main_model_lodo.ipynb` MUST include full class definitions for `ChannelAttention`, `SpatialAttention`, `CBAM`, `NonLocalBlock2D`, `QuadrantTokenAggregator`, `GlobalBranch`, `GatedAttentionPool`, `DualBranchMILModel`, ordinal heads (`CLM_QWK`, `ContinuousQWKLoss`), `DRFullModel`, and the factory function `build_model`.
- **FR-015**: Colab-Optimized Production Execution: The notebook MUST default to `SMOKE_TEST = False` with full multi-epoch training, automatic checkpoint resumption, early stopping on validation QWK, decoupled LR scheduling, and precomputed/on-the-fly `.npz` feature caching.

### Key Entities

- **LODO Fold**: Independent experimental iteration where 1 clinical dataset is held out as the test set and the remaining 5 are used for training and in-domain validation.
- **Domain Generalization Gap ($\Delta_{\text{LODO}}$)**: Quantitative drop $\text{Val\_QWK} - \text{OOD\_Test\_QWK}$ measuring model resilience against external camera optics and population shift.
- **Source-Aware Training Pool**: Balanced subset of the 5 training cohorts with equalized grade quotas to prevent majority cohorts from dominating.
- **Natural Prevalence Evaluation Split**: Validation and test sets reflecting unaltered epidemiological disease distributions.
- **Completed Fold Checkpoint**: Checkpoint state dictionary containing `"is_completed": True`, preventing redundant re-training.
- **Test Cache Zip**: Compressed archive of preprocessed test image `.npz` files stored in Google Drive for instant cross-session reuse.

---

## Success Criteria

### Measurable Outcomes

- **SC-001**: 100% of patient splits across all 6 folds verify zero inter-split patient overlap ($\text{Train} \cap \text{Val} = \emptyset$ and $\text{Train} \cap \text{Test} = \emptyset$).
- **SC-002**: Unzipping from `/content/drive/MyDrive/native_cropped.zip` executes cleanly to `/content/dataset_local` without path failures or synthetic dummy fallbacks.
- **SC-003**: 100% of finished folds are detected upon notebook restart and skipped in $< 5\text{ seconds}$, consuming zero extra training Compute Units.
- **SC-004**: Output CSV (`lodo_results_summary.csv`), JSON summaries (`lodo_summary.json`), and fold split metadata CSVs (`folds/fold_{k}_{cohort}_*.csv`) are fully populated and exported to Google Drive / results directory.
- **SC-005**: The 8 publication-quality figures are exported to `FIGURES_DIR` at $300\text{ DPI}$ without rendering errors.
- **SC-006**: All code cells in `main_model_lodo.ipynb` execute cleanly from top to bottom on Colab GPU with mixed precision (AMP) without CUDA OOM or memory leaks.

---

## Assumptions & Dependencies

- **A-001**: The preprocessed native cropped dataset archive is available in Google Drive at `/content/drive/MyDrive/native_cropped.zip` (or local path `data/processed/native_cropped/`).
- **A-002**: Google Colab environment has GPU acceleration (T4, V100, or A100) and Google Drive mounted at `/content/drive`.
- **A-003**: Master metadata CSV files or raw cohort CSVs are located under `/content/dataset_local` or repository relative paths.
- **A-004**: The dual-branch architecture, CBAM, sliding window MIL extractor, and CLM QWK loss implementation match `main_model (6).ipynb` for full scientific consistency.
- **A-005**: Research Constitution v1.1.0 directives (strict file integrity, delta-only updates, reproducible seeds) remain strictly enforced.
