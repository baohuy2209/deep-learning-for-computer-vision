# Interface Contract: Clinical Metrics & Screening Calibration Schema

**Feature**: `019-sync-notebook-training-pipelines`  
**Contract Version**: `1.0.0`  
**Applicability**: `calculate_dr_metrics` and `calibrate_and_evaluate_screening` across all notebooks.

---

## 1. `calculate_dr_metrics` Return Payload

```python
{
    "QWK": float,                       # Quadratic Weighted Kappa on 5 ordinal grades [0, 4]
    "Accuracy": float,                  # Multi-class exact match accuracy [0.0, 1.0]
    "Within_1_Grade_Acc": float,        # Clinical viability tolerance |y_true - y_pred| <= 1 [0.0, 1.0]
    "Referable_Sensitivity": float,     # Sensitivity for Referable DR (grades 2-4 vs 0-1) at default tau=0.5
    "Referable_Specificity": float,     # Specificity for Referable DR (grades 2-4 vs 0-1) at default tau=0.5
    "Referable_AUC": float,             # Area Under ROC Curve for Referable DR (requires y_probs)
    "Ref_Sensitivity_at_Spec95": float, # Sensitivity at operating point where Specificity >= 95%
    "Ref_Sensitivity_at_Spec90": float, # Sensitivity at operating point where Specificity >= 90%
    "Ref_Sensitivity_at_Spec85": float, # Sensitivity at operating point where Specificity >= 85%
    "Ref_Sensitivity_at_Spec80": float, # Sensitivity at operating point where Specificity >= 80%
    "Recall_Grade_0": float,            # Sensitivity on No DR
    "Recall_Grade_1": float,            # Sensitivity on Mild DR
    "Recall_Grade_2": float,            # Sensitivity on Moderate DR
    "Recall_Grade_3": float,            # Sensitivity on Severe DR
    "Recall_Grade_4": float,            # Sensitivity on Proliferative DR
    "Monotonicity_Violations": float,   # Fraction of samples violating P(y>0) >= P(y>1) >= P(y>2) >= P(y>3)
    "Confusion_Matrix": List[List[int]],# 5x5 integer confusion matrix
    "Loss": float,                      # Mean batch cross-entropy / ordinal loss
    "probs": Optional[np.ndarray]       # Array of shape (N, 5) with class probabilities
}
```

---

## 2. `calibrate_and_evaluate_screening` Output DataFrame

| Column | Type | Description |
| :--- | :--- | :--- |
| `Target_Specificity` | `float` | Target clinical threshold (e.g. 0.80, 0.85, 0.90, 0.95) |
| `Optimal_Threshold` | `float` | $\tau^*$ chosen strictly on Validation set to achieve $\text{Spec} \ge \text{Target}$ |
| `Val_Sensitivity` | `float` | In-domain validation sensitivity achieved at $\tau^*$ |
| `Val_Specificity` | `float` | In-domain validation specificity achieved at $\tau^*$ |
| `Test_Sensitivity` | `float` | Unseen Held-out test sensitivity evaluated with frozen $\tau^*$ |
| `Test_Specificity` | `float` | Unseen Held-out test specificity evaluated with frozen $\tau^*$ |
