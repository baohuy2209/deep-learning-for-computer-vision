# Experimental Results Serialization Schema

This document defines the machine-readable JSON and CSV serialization schemas utilized across all Leave-One-Dataset-Out (LODO) benchmark experiments in `drdg`.

---

## 1. Bipartite Partition Isolation

A prevalent source of scientific bias in clinical machine learning is ambiguous evaluation reporting, where source validation metrics are accidentally reported as target generalization results, or thresholds are tuned post-hoc on the target test cohort.

To eliminate this ambiguity and faithfully reflect the research design in `main_model_lodo.ipynb` (Cell 17 Section 7), `drdg` enforces strict **bipartite results serialization** across all evaluations (`drdg.evaluation.result_schema.LODOFoldResult`):

```
                        LODO Experimental Evaluation
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
             [in_domain_val]                      [ood_test]
           Used strictly for:                Measures true out-of-domain
           - Model selection                 generalization on 100% of
           - Early stopping                  the held-out clinical
           - RDR threshold (tau*)            target cohort, evaluated
             calibration                     with frozen tau* threshold
```

> **Parity Note**: There is NO artificial third "source_test" partition in the research methodology. The tripartite schema (`source_validation`, `source_test`, `target_test`) was a prior refactoring fabrication that has been eradicated in favor of the authentic bipartite protocol.

---

## 2. Complete LODO Fold JSON Schema

Below is the formal JSON schema generated per fold in `results/lodo/<variant>/<held_out>/metrics.json`:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "LODOFoldResult",
  "type": "object",
  "required": [
    "fold",
    "variant",
    "model_name",
    "calibrated_threshold",
    "in_domain_val",
    "ood_test",
    "delta_lodo"
  ],
  "properties": {
    "fold": {
      "type": "string",
      "description": "Identifier of the held-out target cohort (e.g., 'aptos', 'ddr', 'deepdrid', 'idrid', 'messidor2', 'eyepacs')."
    },
    "variant": {
      "type": "string",
      "enum": ["v0", "v1", "v2", "v3"],
      "description": "Architectural variant evaluated."
    },
    "model_name": {
      "type": "string"
    },
    "checkpoint": {
      "type": ["string", "null"],
      "description": "Path to serialized weights file."
    },
    "calibrated_threshold": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0,
      "description": "Referable DR decision threshold (tau*) calibrated on in_domain_val at target specificity."
    },
    "delta_lodo": {
      "type": "number",
      "description": "Generalization gap: in_domain_val.qwk - ood_test.qwk."
    },
    "in_domain_val": {
      "$ref": "#/definitions/MetricBundle",
      "description": "Metrics evaluated on pooled source validation patients."
    },
    "ood_test": {
      "$ref": "#/definitions/MetricBundle",
      "description": "Metrics evaluated on 100% of held-out target cohort images using frozen calibrated_threshold."
    }
  },
  "definitions": {
    "MetricBundle": {
      "type": "object",
      "required": [
        "qwk",
        "accuracy",
        "within_1_accuracy",
        "per_grade_sensitivity",
        "rdr_auc",
        "rdr_sensitivity_at_95_specificity",
        "total_samples"
      ],
      "properties": {
        "qwk": {
          "type": "number",
          "description": "Quadratic Weighted Kappa."
        },
        "accuracy": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0
        },
        "within_1_accuracy": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0
        },
        "per_grade_sensitivity": {
          "type": "object",
          "properties": {
            "0": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
            "1": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
            "2": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
            "3": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
            "4": { "type": "number", "minimum": 0.0, "maximum": 1.0 }
          },
          "required": ["0", "1", "2", "3", "4"]
        },
        "rdr_auc": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0
        },
        "rdr_sensitivity_at_95_specificity": {
          "type": "number",
          "minimum": 0.0,
          "maximum": 1.0
        },
        "total_samples": {
          "type": "integer",
          "minimum": 1
        },
        "confusion_matrix": {
          "type": "array",
          "items": {
            "type": "array",
            "items": { "type": "integer" }
          }
        }
      }
    }
  }
}
```

---

## 3. LODO Benchmark Summary CSV Schema

Aggregated across all 6 folds, results are saved to `results/lodo/<variant>/summary.csv`:

| Column | Type | Description |
|---|---|---|
| `held_out_cohort` | `string` | Target cohort name (`aptos`, `ddr`, `deepdrid`, `idrid`, `messidor2`, `eyepacs`) |
| `val_qwk` | `float` | In-domain validation QWK |
| `val_acc` | `float` | In-domain validation accuracy |
| `val_within_1` | `float` | In-domain validation within-1 accuracy |
| `val_rdr_auc` | `float` | In-domain validation RDR AUC |
| `tau_star` | `float` | Calibrated RDR threshold operating point |
| `test_qwk` | `float` | Out-of-domain target test QWK |
| `test_acc` | `float` | Out-of-domain target test accuracy |
| `test_within_1` | `float` | Out-of-domain target test within-1 accuracy |
| `test_rdr_auc` | `float` | Out-of-domain target test RDR AUC |
| `test_rdr_sens` | `float` | Target sensitivity evaluated at frozen `tau_star` |
| `test_rdr_spec` | `float` | Target specificity evaluated at frozen `tau_star` |
| `delta_qwk` | `float` | Generalization drop: `val_qwk - test_qwk` |
| `n_val_samples` | `int` | Number of images in source validation partition |
| `n_test_samples` | `int` | Number of images in target test cohort |

---

## 4. Controlled Head Ablation CSV Schema

Saved to `results/head_ablation/summary.csv`:

| Column | Type | Description |
|---|---|---|
| `head` | `string` | Output head: `softmax_ce`, `softmax_qwk`, `coral`, `corn`, `clm_qwk` |
| `val_qwk` | `float` | Validation QWK |
| `val_within_1` | `float` | Validation within-1 accuracy |
| `val_off_by_2_plus` | `float` | Severe error rate: $|y - \hat{y}| \ge 2$ |
| `val_mae` | `float` | Mean Absolute Error across integer grades |
| `test_qwk` | `float` | Target test QWK |
| `test_within_1` | `float` | Target test within-1 accuracy |
| `test_off_by_2_plus`| `float` | Target severe error rate |
| `test_mae` | `float` | Target Mean Absolute Error |

---

## 5. Verification

Automated schema validation is enforced in `tests/test_result_schema.py`:
- Checks strict presence of `in_domain_val`, `ood_test`, `delta_lodo`, `calibrated_threshold`.
- Rejects non-existent keys (e.g. `source_test`).
- Asserts boundary invariants ($0.0 \le \tau^* \le 1.0$, $N \ge 1$).
