# Data Model & Entity Specifications: Public Academic Repository Refactor

**Feature Branch**: `017-public-repo-refactor`
**Date**: 2026-09-23
**Status**: Completed

---

## 1. Core Entity Relationship Diagram

```
+------------------------------------------------------------------------------------+
|                                MasterFundusRecord                                  |
+------------------------------------------------------------------------------------+
| image_id: str                   [PK] Unique clean image identifier                 |
| diagnosis: int                  ICDRS ground truth severity in {0, 1, 2, 3, 4}     |
| dataset_name: str               Display name: 'APTOS 2019', 'DDR', 'DeepDRiD', etc.|
| dataset_key: str                Normalized slug: 'aptos', 'ddr', 'deepdrid', etc.  |
| raw_full_path: Path             Absolute/relative path to raw unprocessed image    |
| native_path: Path               Path to cropped circular FOV image in native_cropped
| patient_id: str                 Namespaced identifier: '<dataset_key>::<orig_id>'   |
| side: str                       Eye lateral position: 'left', 'right', 'unknown'   |
| original_image_id: Optional[str] Original cohort identifier prior to harmonization  |
+------------------------------------------------------------------------------------+
                                      |
                                      | 1..* partitioned into
                                      v
+------------------------------------------------------------------------------------+
|                                  PatientSplit                                      |
+------------------------------------------------------------------------------------+
| split_id: str                   [PK] Unique split configuration name               |
| split_type: str                 'stratified_70_15_15' | 'lodo_6fold'               |
| fold_idx: Optional[int]         Fold index in {0, 1, 2, 3, 4, 5}                   |
| held_out_target: Optional[str]  Target cohort key for LODO fold                    |
| train_patient_ids: Set[str]     Disjoint set of training patient IDs               |
| val_patient_ids: Set[str]       Disjoint set of validation patient IDs             |
| test_patient_ids: Set[str]      Disjoint set of evaluation patient IDs             |
+------------------------------------------------------------------------------------+
                                      |
                                      | produces inputs for
                                      v
+------------------------------------------------------------------------------------+
|                               DualBranchFundusBag                                  |
+------------------------------------------------------------------------------------+
| global_image: Tensor            Float32 tensor [3, 512, 512], ImageNet normalized  |
| patches: Tensor                 Float32 tensor [N_patches, 3, 224, 224]            |
| patch_coords: Tensor            Int64 tensor [N_patches, 4] (ymin, xmin, ymax, xmax|
| patch_mask: Tensor              Bool tensor [N_patches] active bag element mask    |
| diagnosis: int                  Ground truth grade in {0, 1, 2, 3, 4}              |
| patient_id: str                 Namespaced patient ID                              |
| image_id: str                   Master image ID                                    |
+------------------------------------------------------------------------------------+
                                      |
                                      | processed by model yielding
                                      v
+------------------------------------------------------------------------------------+
|                                ModelPrediction                                     |
+------------------------------------------------------------------------------------+
| logits: Tensor                  Float32 tensor [B, 5] (or rank logits [B, 4])      |
| probabilities: Tensor           Float32 tensor [B, 5] normalized (sum to 1.0)      |
| continuous_severity: Tensor     Float32 tensor [B] in range [0.0, 4.0]             |
| predicted_grade: Tensor         Int64 tensor [B] argmax / thresholded class        |
| rdr_score: Tensor               Float32 tensor [B] P(Grade >= 2) in [0.0, 1.0]     |
| global_attention_map: Tensor    Float32 tensor [B, 1, H, W] Grad-CAM activation    |
| mil_patch_weights: Tensor       Float32 tensor [B, N_patches] Gated Attention alpha|
+------------------------------------------------------------------------------------+
                                      |
                                      | evaluated and serialized as
                                      v
+------------------------------------------------------------------------------------+
|                                  LODOFoldResult                                    |
+------------------------------------------------------------------------------------+
| variant: str                    Architecture variant: 'v0', 'v1', 'v2', 'v3'       |
| held_out_dataset: str           Target cohort name                                 |
| fold: int                       Fold index 0..5                                    |
| source_validation: MetricBundle Validation set metrics used strictly for checkpoint|
| source_test: MetricBundle       Source domain test set performance                 |
| target_test: MetricBundle       Zero-shot out-of-distribution target performance   |
| threshold_rdr_spec95: float     Frozen threshold calibrated on source_validation   |
+------------------------------------------------------------------------------------+
```

---

## 2. Entity Specifications & Field Constraints

### 2.1 `MasterFundusRecord`
- **Location**: Defined in `src/drdg/data/metadata.py`.
- **Validation Rules**:
  - `image_id`: Must be non-empty, alphanumeric with underscores/hyphens.
  - `diagnosis`: Strict integer validation: `diagnosis in {0, 1, 2, 3, 4}`. Missing, float, or out-of-bounds grades raise `ValueError`.
  - `dataset_key`: Must belong to valid registered cohorts: `{'aptos', 'ddr', 'deepdrid', 'idrid', 'messidor2', 'eyepacs'}`.
  - `patient_id`: Must contain the namespace delimiter `::` (e.g. `ddr::00124`).
  - `side`: Normalized to lowercase string in `{'left', 'right', 'unknown'}`.
  - `native_path`: Must point to an existing image file on disk.

### 2.2 `PatientSplit`
- **Location**: Defined in `src/drdg/data/splits.py` and `src/drdg/data/lodo.py`.
- **Invariants**:
  - Mutual Exclusivity:
    $$\text{train\_patient\_ids} \cap \text{val\_patient\_ids} = \emptyset$$
    $$\text{train\_patient\_ids} \cap \text{test\_patient\_ids} = \emptyset$$
    $$\text{val\_patient\_ids} \cap \text{test\_patient\_ids} = \emptyset$$
  - LODO Target Isolation: For any LODO fold with `held_out_target = T`, all images where `dataset_key == T` MUST belong exclusively to `test_patient_ids`. No patient with `dataset_key == T` may appear in `train_patient_ids` or `val_patient_ids`.

### 2.3 `MetricBundle`
- **Location**: Defined in `src/drdg/evaluation/result_schema.py`.
- **Fields**:
  - `qwk`: Quadratic Weighted Kappa, float in $[-1.0, 1.0]$.
  - `accuracy`: Overall multi-class accuracy, float in $[0.0, 1.0]$.
  - `within_1_accuracy`: Accuracy within $\pm 1$ grade, float in $[0.0, 1.0]$.
  - `per_grade_sensitivity`: Dictionary mapping grade `'0'` through `'4'` to float in $[0.0, 1.0]$.
  - `rdr_auc`: Area Under the ROC Curve for Referable DR ($\text{Grade} \ge 2$), float in $[0.0, 1.0]$.
  - `rdr_sensitivity_at_95_specificity`: Sensitivity on binary RDR task evaluated at the frozen source-calibrated 95% specificity threshold, float in $[0.0, 1.0]$.
  - `confusion_matrix`: $5 \times 5$ integer matrix.

### 2.4 `CutpointParameters` (Cumulative Link Model)
- **Location**: Defined in `src/drdg/models/heads/clm.py`.
- **Mathematical Invariant**:
  - Cutpoints must be strictly ordered:
    $$-\infty = b_0 < b_1 < b_2 < b_3 < b_4 < b_5 = +\infty$$
  - Internal representation:
    $$b_1 \in \mathbb{R}, \quad \Delta_k \in \mathbb{R}^+ \ (k = 2, 3, 4)$$
    $$b_k = b_1 + \sum_{j=2}^k \text{softplus}(\Delta_j)$$
  - Guarantees $b_k > b_{k-1}$ for all iterations and gradient steps.
