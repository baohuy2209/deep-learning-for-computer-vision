# Data Model: Baseline Oracle Training & Evaluation

**Feature**: `016-baseline-oracle-training`  
**Date**: 2026-09-22  
**Purpose**: Define the data structures, entities, schema representations, and artifact formats across all 6 Oracle pipelines.

---

## 1. Primary Entities

### 1.1 In-Domain Dataset Record (`FundusRecord`)
Represents an individual retinal image in a cohort partition.

| Attribute | Type | Description | Validation Rule |
| :--- | :--- | :--- | :--- |
| `image_id` | `str` | Unique image identifier (e.g. `1ae8c165fd53`) | Non-empty string |
| `patient_id` | `str` | Globally unique patient identifier (`{dataset_key}::{pid}`) | Must prevent bilateral cross-eye leakage |
| `dataset_key` | `str` | Cohort slug (`APTOS`, `DDR`, `DeepDRiD`, `IDRiD`, `Messidor-2`, `EyePACS`) | Must belong to the 6 supported keys |
| `diagnosis` | `int` | ICDRS DR Severity Grade | Integer in $\{0, 1, 2, 3, 4\}$ |
| `image_path` | `str` | Absolute / relative path to native cropped image | File must exist on disk or drive |
| `split` | `str` | Partition assignment | Must be one of `["train", "val", "test"]` |

### 1.2 Training & Evaluation Batch (`DualBranchBatch`)
Structure emitted by `FundusDualBranchDataset` and `trainer_compatible_collate_fn`.

| Attribute | Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `global_img` | `[B, 3, 512, 512]` | `torch.float32` | Ben Graham contrast-enhanced whole-fundus context |
| `local_patches` | `[B, N, 3, 128, 128]` | `torch.float32` | $N$ sliding-window native patches ($N \in [32, 64]$) |
| `patch_mask` | `[B, N]` | `torch.bool` | Valid FOV mask per patch |
| `targets` | `[B]` | `torch.int64` | Ground truth ICDRS grade |
| `image_ids` | `List[str]` | `str` | Batch tracking identifiers |

### 1.3 Oracle Checkpoint Metadata (`OracleCheckpoint`)
Persisted state dictionary saved at `/content/drive/MyDrive/checkpoints_oracle/oracle_{cohort}_best.pth`.

```json
{
  "epoch": 28,
  "model_state_dict": "<OrderedDict>",
  "optimizer_state_dict": "<dict>",
  "scheduler_state_dict": "<dict>",
  "scaler_state_dict": "<dict>",
  "best_val_qwk": 0.8654,
  "cohort": "APTOS 2019",
  "metrics": {
    "val_qwk": 0.8654,
    "val_accuracy": 0.8245,
    "val_within_1": 0.9650,
    "val_ref_auc": 0.9412
  }
}
```

### 1.4 Benchmark Comparison Output (`OracleEvaluationResult`)
Saved as `oracle_{cohort}_results.json` and consumed for updating manuscript Table 3.

```json
{
  "cohort": "APTOS 2019",
  "test_sample_size": 550,
  "test_class_distribution": {
    "0": 271,
    "1": 56,
    "2": 150,
    "3": 29,
    "4": 44
  },
  "oracle_in_domain": {
    "qwk": 0.8842,
    "within_1_accuracy": 0.9727,
    "referable_auc": 0.9580,
    "sens_at_spec95": 0.8214
  },
  "lodo_cross_domain": {
    "qwk": 0.8125,
    "within_1_accuracy": 0.9418,
    "referable_auc": 0.9150,
    "sens_at_spec95": 0.7420
  },
  "delta_transfer": {
    "delta_qwk": 0.0717,
    "delta_within_1": 0.0309,
    "delta_ref_auc": 0.0430,
    "interpretation": "Positive degradation verifying authentic domain shift penalty without prior shift contamination."
  }
}
```

---

## 2. Partition State Lifecycle

```mermaid
stateDiagram-v2
    [*] --> RawCohortCSV: Read cohort metadata
    RawCohortCSV --> StandardizedDF: Validate schema (image_id, diagnosis in 0..4)
    StandardizedDF --> PatientGrouping: Prefix dataset_key::patient_id
    PatientGrouping --> TrainSplit: 70% GroupShuffleSplit (Train)
    PatientGrouping --> TempSplit: 30% GroupShuffleSplit (Holdout)
    TempSplit --> ValSplit: 15% GroupShuffleSplit (Val)
    TempSplit --> TestSplit: 15% GroupShuffleSplit (Test)
    
    TrainSplit --> OracleModel: In-domain Training (40 ep, CLM+QWK)
    ValSplit --> OracleModel: Checkpoint selection (val_qwk)
    
    OracleModel --> TestEvaluation: Oracle QWK(target)
    LODOChampion --> TestEvaluation: OOD QWK(target)
    TestEvaluation --> DeltaTransfer: Delta = QWK_oracle - QWK_OOD
```
