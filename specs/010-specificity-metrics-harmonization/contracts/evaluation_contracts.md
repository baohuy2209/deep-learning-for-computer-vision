# Python Function Interface Contract: Clinical Evaluation & Calibration

**Feature**: Clinical Specificity-Sensitivity Harmonization & Colab Reference Code Sync  
**Branch**: `010-specificity-metrics-harmonization`  
**Date**: 2026-09-13  
**Spec**: [spec.md](file:///d:/Github/deep-learning-for-computer-vision/specs/010-specificity-metrics-harmonization/spec.md)

---

## 1. Function: `calculate_referable_sensitivity_at_specificity`

Location: Section 4.1 in both `main_model.ipynb` and `main_model_head_ablation.ipynb`

```python
def calculate_referable_sensitivity_at_specificity(
    y_true_grades: np.ndarray,
    referable_probs: np.ndarray,
    target_specificities: Tuple[float, ...] = (0.80, 0.85, 0.90, 0.95),
) -> Dict[str, float]:
    """
    Computes clinical Sensitivity at fixed Specificity operating points for Referable DR screening
    (binary classification threshold: Grade >= 2, Moderate/Severe/PDR vs. No DR/Mild).

    Args:
        y_true_grades: Ground-truth ordinal labels (integers 0 to 4).
        referable_probs: Continuous predicted referable risk score P(y >= 2).
        target_specificities: Tuple of target specificities to evaluate (default: 80%, 85%, 90%, 95%).

    Returns:
        Dict containing:
          - "Referable_AUC": float
          - "Ref_Sensitivity_at_Spec80": float (expected ~84.7%)
          - "Ref_Sensitivity_at_Spec85": float (expected ~79.6%)
          - "Ref_Sensitivity_at_Spec90": float
          - "Ref_Sensitivity_at_Spec95": float
          - "Ref_Threshold_at_Spec80": float
          - "Ref_Threshold_at_Spec85": float
          - "Ref_Threshold_at_Spec90": float
          - "Ref_Threshold_at_Spec95": float
    """
```

---

## 2. Function: `compute_clinical_metrics`

Location: Section 4.1 in both notebooks

```python
def compute_clinical_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
    target_specificities: Tuple[float, ...] = (0.80, 0.85, 0.90, 0.95),
) -> Dict[str, Any]:
    """
    Computes multi-dimensional clinical metrics across discrete classifications and continuous probabilities.

    Guarantees:
      - Class recall computed for all 5 stages (Grade 0..4).
      - Within-1-Grade accuracy computed.
      - Referable DR metrics computed for all specified target_specificities (80%, 85%, 90%, 95%).
    """
```

---

## 3. Function: `calibrate_and_evaluate_screening`

Location: Section 4.1 in `main_model.ipynb` and Section 4.2 in `main_model_head_ablation.ipynb`

```python
def calibrate_and_evaluate_screening(
    val_probs: np.ndarray,
    val_targets: np.ndarray,
    test_probs: np.ndarray,
    test_targets: np.ndarray,
    target_specificities: Tuple[float, ...] = (0.80, 0.85, 0.90, 0.95),
) -> pd.DataFrame:
    """
    Executes rigorous post-hoc threshold calibration strictly on Validation continuous predictions,
    and applies calibrated cutoffs tau to the Held-Out Test set.

    Guarantees:
      - Fits tau on val_probs where val_spec >= target_sp.
      - Evaluates test_sens and test_spec on test_targets using tau.
      - Includes Expected_Sensitivity column documenting binormal theoretical benchmarks:
          * Spec 95%: 58.7%
          * Spec 90%: 72.0%
          * Spec 85%: 79.6% (Near 80% Target)
          * Spec 80%: 84.7% (Exceeds 80% Target)
      - Returns formatted DataFrame and prints full clinical screening report to stdout.
    """
```

---

## 4. Function: `plot_fig7_roc`

Location: Section 5 in `main_model.ipynb` and Section 5 in `main_model_head_ablation.ipynb`

```python
def plot_fig7_roc(
    val_data: Union[Dict[str, Any], Any],
    test_data: Union[Dict[str, Any], Any],
    save_path: Optional[str] = None,
) -> None:
    """
    Renders publication-quality ROC curves for Referable DR screening (Grade >= 2).

    Visual Specifications:
      - Plots Validation ROC (navy solid line) and Held-Out Test ROC (amber dashed line).
      - Renders scatter markers and explicit text callouts for:
          1. Spec 95% (FPR = 0.05)
          2. Spec 85% (FPR = 0.15, Sens ~79.6%)
          3. Spec 80% (FPR = 0.20, Sens ~84.7%)
      - Exports high-resolution PDF/PNG to save_path.
    """
```
