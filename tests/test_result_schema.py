"""Unit tests for bipartite metric schema validation (LODO).

Tests the LODOFoldResult schema with in_domain_val + ood_test partitions,
matching main_model_lodo.ipynb Cell 17 Section 7.
"""

import json
from pathlib import Path

import pytest

from drdg.evaluation.result_schema import LODOFoldResult, MetricBundle, validate_lodo_result_dict


def sample_metric_bundle():
    return MetricBundle(
        qwk=0.8421,
        accuracy=0.7650,
        within_1_accuracy=0.9520,
        per_grade_sensitivity={"0": 0.91, "1": 0.45, "2": 0.78, "3": 0.69, "4": 0.88},
        rdr_auc=0.9412,
        rdr_sensitivity_at_95_specificity=0.8120,
        total_samples=350,
        confusion_matrix=[[10, 1, 0, 0, 0], [2, 8, 1, 0, 0], [0, 1, 12, 1, 0], [0, 0, 1, 9, 1], [0, 0, 0, 1, 15]],
    )


def test_valid_lodo_fold_result(tmp_path):
    val_bundle = sample_metric_bundle()
    ood_bundle = MetricBundle(
        qwk=0.7800,
        accuracy=0.7200,
        within_1_accuracy=0.9300,
        per_grade_sensitivity={"0": 0.88, "1": 0.40, "2": 0.72, "3": 0.60, "4": 0.82},
        rdr_auc=0.9100,
        rdr_sensitivity_at_95_specificity=0.7800,
        total_samples=200,
    )
    result = LODOFoldResult(
        variant="v3",
        held_out_dataset="aptos",
        fold=0,
        seed=42,
        threshold_rdr_spec95=0.485,
        in_domain_val=val_bundle,
        ood_test=ood_bundle,
    )

    d = result.to_dict()
    assert d["variant"] == "v3"
    assert d["held_out_dataset"] == "aptos"
    assert "in_domain_val" in d
    assert "ood_test" in d
    assert d["in_domain_val"]["qwk"] == 0.8421
    assert d["ood_test"]["qwk"] == 0.7800

    # Verify delta_lodo
    assert abs(d["delta_lodo"] - (0.8421 - 0.7800)) < 1e-6

    # Roundtrip to JSON
    json_path = tmp_path / "v3_fold0_aptos.json"
    result.save_json(json_path)
    assert json_path.exists()

    loaded = LODOFoldResult.load_json(json_path)
    assert loaded.variant == "v3"
    assert loaded.fold == 0
    assert loaded.threshold_rdr_spec95 == 0.485
    assert loaded.ood_test.accuracy == 0.7200
    assert loaded.in_domain_val.qwk == 0.8421
    assert abs(loaded.delta_lodo - (0.8421 - 0.7800)) < 1e-6


def test_invalid_variant():
    bundle = sample_metric_bundle()
    with pytest.raises(ValueError, match="variant must be one of"):
        LODOFoldResult(
            variant="v99",
            held_out_dataset="aptos",
            fold=0,
            seed=42,
            threshold_rdr_spec95=0.5,
            in_domain_val=bundle,
            ood_test=bundle,
        )


def test_invalid_fold():
    bundle = sample_metric_bundle()
    with pytest.raises(ValueError, match="fold must be in"):
        LODOFoldResult(
            variant="v2",
            held_out_dataset="ddr",
            fold=10,  # Invalid, only 0-5
            seed=42,
            threshold_rdr_spec95=0.5,
            in_domain_val=bundle,
            ood_test=bundle,
        )


def test_invalid_threshold():
    bundle = sample_metric_bundle()
    with pytest.raises(ValueError, match="threshold_rdr_spec95 must be in"):
        LODOFoldResult(
            variant="v1",
            held_out_dataset="idrid",
            fold=2,
            seed=42,
            threshold_rdr_spec95=1.5,  # Out of [0, 1]
            in_domain_val=bundle,
            ood_test=bundle,
        )


def test_validate_lodo_result_dict():
    bundle_d = sample_metric_bundle().to_dict()
    valid_dict = {
        "variant": "v0",
        "held_out_dataset": "messidor2",
        "fold": 4,
        "seed": 42,
        "threshold_rdr_spec95": 0.42,
        "in_domain_val": bundle_d,
        "ood_test": bundle_d,
    }
    assert validate_lodo_result_dict(valid_dict) is True
