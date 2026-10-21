"""Unit tests for Referable DR screening calibration and frozen threshold isolation."""

import numpy as np
import pytest

from drdg.evaluation.rdr import calibrate_rdr_threshold, evaluate_rdr_with_frozen_threshold


@pytest.fixture
def synthetic_val_data():
    """Generates synthetic validation probabilities and binary referable labels."""
    np.random.seed(42)
    # 100 negatives (No DR, Mild DR -> 0)
    neg_probs = np.random.beta(1.5, 5.0, size=100)
    # 50 positives (Moderate, Severe, PDR -> 1)
    pos_probs = np.random.beta(5.0, 1.5, size=50)

    y_val = np.concatenate([np.zeros(100, dtype=int), np.ones(50, dtype=int)])
    probs_val = np.concatenate([neg_probs, pos_probs])
    return y_val, probs_val


def test_calibrate_rdr_threshold_meets_specificity(synthetic_val_data):
    y_val, probs_val = synthetic_val_data
    calib = calibrate_rdr_threshold(y_val, probs_val, target_specificity=0.95)

    assert 0.0 <= calib.calibrated_threshold <= 1.0
    assert calib.validation_specificity >= 0.94  # Tolerance for finite sample discretization
    assert calib.validation_auc > 0.80
    assert calib.total_val_samples == 150


def test_frozen_threshold_isolation(synthetic_val_data):
    """Verifies that changing target test data has zero effect on the calibrated threshold."""
    y_val, probs_val = synthetic_val_data
    calib = calibrate_rdr_threshold(y_val, probs_val, target_specificity=0.95)
    frozen_tau = calib.calibrated_threshold

    # Test distribution 1: Easy target domain
    y_test1 = np.array([0, 0, 0, 1, 1])
    probs_test1 = np.array([0.1, 0.2, 0.15, 0.9, 0.85])
    eval1 = evaluate_rdr_with_frozen_threshold(y_test1, probs_test1, frozen_threshold=frozen_tau)

    # Test distribution 2: Heavily shifted difficult target domain
    y_test2 = np.array([1, 1, 1, 0, 0])
    probs_test2 = np.array([0.4, 0.5, 0.45, 0.35, 0.6])
    eval2 = evaluate_rdr_with_frozen_threshold(y_test2, probs_test2, frozen_threshold=frozen_tau)

    # The threshold used in both evaluations must be strictly equal to frozen_tau
    assert eval1.frozen_threshold == frozen_tau
    assert eval2.frozen_threshold == frozen_tau


def test_evaluate_rdr_edge_cases():
    """Verifies robust handling of empty or single-class distributions."""
    empty_eval = evaluate_rdr_with_frozen_threshold([], [], frozen_threshold=0.5)
    assert empty_eval.total_test_samples == 0
    assert np.isnan(empty_eval.test_auc)

    # Single-class test data
    single_class_eval = evaluate_rdr_with_frozen_threshold([1, 1, 1], [0.8, 0.9, 0.7], frozen_threshold=0.5)
    assert single_class_eval.total_test_samples == 3
    assert single_class_eval.test_sensitivity == 1.0
