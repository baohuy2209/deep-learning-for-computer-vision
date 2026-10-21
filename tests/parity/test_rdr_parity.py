"""Parity test for Referable Diabetic Retinopathy (RDR) screening calibration.

Compares golden reference calibration logic from visualize_main_model.py (lines 1920-1954)
against src/drdg/evaluation/rdr.py (calibrate_rdr_threshold and evaluate_rdr_with_frozen_threshold).
"""

import numpy as np
import pytest
from sklearn.metrics import auc, roc_curve

from drdg.evaluation.rdr import calibrate_rdr_threshold, evaluate_rdr_with_frozen_threshold


def golden_compute_referable_dr_metrics(
    y_true,
    referable_probs,
    target_specificities=(0.80, 0.85, 0.90, 0.95),
):
    """Extracted verbatim from visualize_main_model.py lines 1925-1954."""
    y_true_ref = (np.asarray(y_true, dtype=int) >= 2).astype(int)
    referable_probs = np.asarray(referable_probs, dtype=float)

    results = {}
    n_pos = int(y_true_ref.sum())
    n_neg = int((1 - y_true_ref).sum())
    if n_pos == 0 or n_neg == 0:
        results["Referable_AUC"] = float("nan")
        for spec in target_specificities:
            results[f"Ref_Sensitivity_at_Spec{int(round(spec * 100))}"] = float("nan")
            results[f"Ref_Threshold_at_Spec{int(round(spec * 100))}"] = float("nan")
        return results

    fpr, tpr, thresholds = roc_curve(y_true_ref, referable_probs)
    results["Referable_AUC"] = float(auc(fpr, tpr))
    specificity = 1.0 - fpr

    for spec in target_specificities:
        valid_idx = np.where(specificity >= spec)[0]
        if len(valid_idx) == 0:
            sens_at_spec = 0.0
            best_thresh = float(thresholds[-1])
        else:
            best_sub_idx = valid_idx[np.argmax(tpr[valid_idx])]
            sens_at_spec = float(tpr[best_sub_idx])
            best_thresh = float(thresholds[best_sub_idx])
        results[f"Ref_Sensitivity_at_Spec{int(round(spec * 100))}"] = sens_at_spec
        results[f"Ref_Threshold_at_Spec{int(round(spec * 100))}"] = best_thresh

    return results


def test_rdr_calibration_numerical_parity():
    """Verify calibrated threshold and sensitivity match between golden and candidate."""
    np.random.seed(42)
    # Generate 100 deterministic predictions (binary DR >= 2)
    y_true = np.random.choice([0, 1, 2, 3, 4], size=100, p=[0.4, 0.2, 0.2, 0.1, 0.1])
    # Scores correlated with grade
    probs = np.clip(y_true / 4.0 + np.random.normal(0, 0.15, size=100), 0.0, 1.0)

    for target_spec in [0.80, 0.85, 0.90, 0.95]:
        gold_res = golden_compute_referable_dr_metrics(
            y_true, probs, target_specificities=(target_spec,)
        )
        cand_res = calibrate_rdr_threshold(
            y_true, probs, target_specificity=target_spec
        )

        gold_thresh = gold_res[f"Ref_Threshold_at_Spec{int(round(target_spec * 100))}"]
        gold_sens = gold_res[f"Ref_Sensitivity_at_Spec{int(round(target_spec * 100))}"]

        # When threshold is clamped to <= 1.0 in candidate:
        expected_cand_thresh = min(1.0, max(0.0, gold_thresh))

        assert np.isclose(cand_res.calibrated_threshold, expected_cand_thresh, atol=1e-5), (
            f"Spec {target_spec}: Threshold mismatch: cand {cand_res.calibrated_threshold} vs gold {gold_thresh}"
        )
        assert np.isclose(cand_res.validation_auc, gold_res["Referable_AUC"], atol=1e-5), (
            f"Spec {target_spec}: AUC mismatch"
        )
        assert np.isclose(cand_res.validation_sensitivity, gold_sens, atol=1e-5), (
            f"Spec {target_spec}: Sensitivity mismatch: cand {cand_res.validation_sensitivity} vs gold {gold_sens}"
        )


def test_frozen_threshold_evaluation_zero_leakage():
    """Verify test evaluation strictly applies frozen threshold without re-fitting."""
    y_val = [0, 0, 1, 2, 3, 4]
    probs_val = [0.1, 0.2, 0.3, 0.7, 0.8, 0.95]

    calib = calibrate_rdr_threshold(y_val, probs_val, target_specificity=0.90)
    frozen_tau = calib.calibrated_threshold

    y_test = [0, 1, 2, 4]
    probs_test = [0.15, 0.25, 0.75, 0.9]

    eval_res = evaluate_rdr_with_frozen_threshold(y_test, probs_test, frozen_threshold=frozen_tau)
    assert eval_res.frozen_threshold == frozen_tau
    assert eval_res.test_sensitivity == 1.0  # both >=2 cases (0.75, 0.9) exceed tau ~0.7
    assert eval_res.test_specificity == 1.0
