# Data Model Specification: Clinical Evaluation & Calibration Metrics

**Feature**: Clinical Specificity-Sensitivity Harmonization & Colab Reference Code Sync  
**Branch**: `010-specificity-metrics-harmonization`  
**Date**: 2026-09-13  
**Spec**: [spec.md](file:///d:/Github/deep-learning-for-computer-vision/specs/010-specificity-metrics-harmonization/spec.md)

---

## 1. Entity: `ClinicalMetricsDict`

Standard dictionary returned by `compute_clinical_metrics` and `DRTrainer.evaluate()`.

| Field Name | Type | Range / Constraints | Description |
| :--- | :--- | :--- | :--- |
| `Loss` | `float` | $[0, +\infty)$ | Average batch loss over the evaluation split. |
| `QWK` | `float` | $[-1.0, 1.0]$ | Quadratic Weighted Kappa against ground truth ordinal grades. |
| `Accuracy` | `float` | $[0.0, 1.0]$ | Standard exact-match multiclass accuracy. |
| `Within_1_Grade_Acc` | `float` | $[0.0, 1.0]$ | Proportion of predictions where $\|\hat{y} - y\| \le 1$. |
| `Recall_Grade_{c}` | `float` | $[0.0, 1.0]$ | Class-conditional sensitivity for each grade $c \in \{0, 1, 2, 3, 4\}$. |
| `Referable_Sensitivity` | `float` | $[0.0, 1.0]$ | Screening sensitivity evaluated at the default nominal `argmax >= 2` cutoff. |
| `Referable_Specificity` | `float` | $[0.0, 1.0]$ | Screening specificity evaluated at the default nominal `argmax >= 2` cutoff. |
| `Referable_AUC` | `float` | $[0.0, 1.0]$ | Area Under the ROC Curve for continuous screening scores $P(y \ge 2)$. |
| `Ref_Sensitivity_at_Spec95` | `float` | $[0.0, 1.0]$ | Referable sensitivity at target specificity $\ge 95\%$. |
| `Ref_Sensitivity_at_Spec90` | `float` | $[0.0, 1.0]$ | Referable sensitivity at target specificity $\ge 90\%$. |
| `Ref_Sensitivity_at_Spec85` | `float` | $[0.0, 1.0]$ | **MANDATORY**: Referable sensitivity at target specificity $\ge 85\%$ (expected $\approx 79.6\%$). |
| `Ref_Sensitivity_at_Spec80` | `float` | $[0.0, 1.0]$ | **MANDATORY**: Referable sensitivity at target specificity $\ge 80\%$ (expected $\approx 84.7\%$). |
| `Ref_Threshold_at_Spec{XX}` | `float` | $[0.0, 1.0]$ | Optimal decision cutoff $\tau$ meeting target specificity on the operating curve. |
| `Monotonicity_Violations` | `float` | $[0.0, 1.0]$ | Proportion of ordinal samples with rank probability inversions (audit metric for $H_3$). |

---

## 2. Entity: `ScreeningCalibrationRecord`

Tabular record produced by `calibrate_and_evaluate_screening()` comparing Validation calibration with Held-Out Test performance.

| Field Name | Type | Format / Example | Description |
| :--- | :--- | :--- | :--- |
| `Target_Specificity` | `str` | `"95%"`, `"90%"`, `"85%"`, `"80%"` | Clinical specificity operating point. |
| `Val_Threshold` | `float` | `0.4512` | Cutoff $\tau$ determined strictly on the Validation set ROC. |
| `Val_Sensitivity` | `float` | `0.7960` | Empirical true positive rate on the Validation set at threshold $\tau$. |
| `Val_Specificity` | `float` | `0.8510` | Empirical true negative rate on the Validation set at threshold $\tau$. |
| `Test_Sensitivity` | `float` | `0.7952` | Generalization sensitivity evaluated on the unseen Test set using $\tau$. |
| `Test_Specificity` | `float` | `0.8488` | Generalization specificity evaluated on the unseen Test set using $\tau$. |
| `Expected_Sensitivity` | `str` | `"79.6% (Near 80% Target)"` | Theoretical sensitivity derived from binormal model ($d' = 1.864$) in `HD_4`. |
| `UK_NSC_Status` | `str` | `"PASSED"` or `"SUB-OPTIMAL"` | Strict check: `PASSED` iff Test Sens $\ge 80\%$ AND Test Spec $\ge 95\%$. |

---

## 3. Entity: `HeadAblationBenchmarkRecord`

Row representation for `df_heads` in `main_model_head_ablation.ipynb` and `head_ablation_summary.csv`.

```text
Columns:
[
  "Head",                    # e.g., "Softmax_CE", "Softmax_QWK", "CORAL", "CORN", "CLM_QWK"
  "Val_QWK",                 # Best validation QWK achieved during training
  "Test_QWK",                # Held-out test QWK evaluated using best checkpoint
  "Generalization_Gap",      # Val_QWK - Test_QWK (Delta metric)
  "Within_1_Grade_Acc",      # Tolerance accuracy (%)
  "Sens@Spec95",             # Test referable sensitivity at 95% specificity (%)
  "Sens@Spec90",             # Test referable sensitivity at 90% specificity (%)
  "Sens@Spec85",             # Test referable sensitivity at 85% specificity (%) [Target: ~79.6%]
  "Sens@Spec80",             # Test referable sensitivity at 80% specificity (%) [Target: ~84.7%]
  "Referable_AUC",           # Test ROC-AUC for referable DR
  "Monotonicity_Violations"  # Violations rate (%)
]
```
