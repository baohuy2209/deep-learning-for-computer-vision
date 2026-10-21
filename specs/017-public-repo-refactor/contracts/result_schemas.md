# Result Serialization Schemas: Public Academic Repository Refactor

**Feature Branch**: `017-public-repo-refactor`
**Date**: 2026-09-23
**Status**: Completed

---

## 1. Six-Fold LODO Fold Result Schema (`results/lodo/<variant>_fold<idx>_<held_out>.json`)

To resolve critical ambiguities from legacy notebooks, result schemas strictly separate `source_validation`, `source_test`, and `target_test`:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "LODOFoldResult",
  "type": "object",
  "required": [
    "variant",
    "held_out_dataset",
    "fold",
    "seed",
    "threshold_rdr_spec95",
    "source_validation",
    "source_test",
    "target_test"
  ],
  "properties": {
    "variant": { "type": "string", "enum": ["v0", "v1", "v2", "v3"] },
    "held_out_dataset": { "type": "string" },
    "fold": { "type": "integer", "minimum": 0, "maximum": 5 },
    "seed": { "type": "integer" },
    "threshold_rdr_spec95": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
    "source_validation": { "$ref": "#/definitions/MetricBundle" },
    "source_test": { "$ref": "#/definitions/MetricBundle" },
    "target_test": { "$ref": "#/definitions/MetricBundle" }
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
        "qwk": { "type": "number" },
        "accuracy": { "type": "number" },
        "within_1_accuracy": { "type": "number" },
        "per_grade_sensitivity": {
          "type": "object",
          "properties": {
            "0": { "type": "number" },
            "1": { "type": "number" },
            "2": { "type": "number" },
            "3": { "type": "number" },
            "4": { "type": "number" }
          },
          "required": ["0", "1", "2", "3", "4"]
        },
        "rdr_auc": { "type": "number" },
        "rdr_sensitivity_at_95_specificity": { "type": "number" },
        "total_samples": { "type": "integer" },
        "confusion_matrix": {
          "type": "array",
          "items": { "type": "array", "items": { "type": "integer" } }
        }
      }
    }
  }
}
```

---

## 2. Output-Head Ablation Summary Schema (`results/head_ablation/summary.json`)

```json
{
  "experiment": "controlled_head_ablation",
  "backbone": "resnet50_standardized",
  "dataset_split": "in_domain_70_15_15",
  "timestamp": "2026-09-23T20:00:00Z",
  "heads": {
    "softmax_ce": {
      "test_qwk": 0.8124,
      "test_accuracy": 0.7450,
      "within_1_accuracy": 0.9410,
      "rdr_auc": 0.9230,
      "per_grade_sensitivity": { "0": 0.88, "1": 0.42, "2": 0.71, "3": 0.65, "4": 0.82 }
    },
    "softmax_qwk": { ... },
    "coral": { ... },
    "corn": { ... },
    "clm_qwk": { ... }
  }
}
```

---

## 3. Quantitative XAI Summary Schema (`results/xai/metrics_summary.json`)

```json
{
  "checkpoint": "checkpoints/v3_clm_lodo_aptos_best.pt",
  "cohort": "idrid_segmentation",
  "total_images_evaluated": 81,
  "metrics": {
    "global_gradcam": {
      "pointing_game_accuracy": 0.7654,
      "area_matched_saliency_recall": 0.3820,
      "pixel_auroc": 0.8140
    },
    "local_mil_attention": {
      "pointing_game_accuracy": 0.8519,
      "area_matched_saliency_recall": 0.4912,
      "pixel_auroc": 0.8765
    }
  }
}
```
