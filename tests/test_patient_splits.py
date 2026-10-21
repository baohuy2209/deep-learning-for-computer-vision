"""Unit tests for patient-level stratified splitting and leakage auditing."""

import pandas as pd
import pytest

from drdg.data.splits import PatientSplit, audit_patient_leakage, split_patient_stratified


@pytest.fixture
def sample_cohort_df():
    """Generates synthetic cohort DataFrame with multi-image patients."""
    records = []
    # 50 patients, each having 2 images (e.g. left and right eye)
    for p_id in range(50):
        patient_str = f"test_cohort_p{p_id:03d}"
        records.append({
            "image_id": f"img_{p_id}_left",
            "patient_id": patient_str,
            "cohort": "test_cohort",
            "dr_grade": p_id % 5,
            "eye": "left",
        })
        records.append({
            "image_id": f"img_{p_id}_right",
            "patient_id": patient_str,
            "cohort": "test_cohort",
            "dr_grade": p_id % 5,
            "eye": "right",
        })
    return pd.DataFrame(records)


def test_split_patient_stratified_disjoint(sample_cohort_df):
    """Verifies that patient partitions have zero intersection."""
    split = split_patient_stratified(
        sample_cohort_df,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    # Mathematical disjointness
    assert len(split.train_patients.intersection(split.val_patients)) == 0
    assert len(split.train_patients.intersection(split.test_patients)) == 0
    assert len(split.val_patients.intersection(split.test_patients)) == 0

    # Total unique patients preserved
    total_patients = set(sample_cohort_df["patient_id"].unique())
    assert split.train_patients.union(split.val_patients).union(split.test_patients) == total_patients


def test_multi_image_patient_isolation(sample_cohort_df):
    """Verifies that all images from the same patient are kept in the same split."""
    split = split_patient_stratified(sample_cohort_df, seed=42)

    for p in split.train_patients:
        p_imgs = sample_cohort_df[sample_cohort_df["patient_id"] == p]["image_id"].tolist()
        for img in p_imgs:
            assert img in split.train_df["image_id"].values
            assert img not in split.val_df["image_id"].values
            assert img not in split.test_df["image_id"].values


def test_verify_zero_leakage_exception():
    """Verifies that PatientSplit.verify_zero_leakage raises ValueError when leakage occurs."""
    dummy_df = pd.DataFrame({"patient_id": ["p1", "p2"]})
    split_with_leakage = PatientSplit(
        train_df=dummy_df,
        val_df=dummy_df,
        test_df=dummy_df,
        train_patients={"p1", "p2"},
        val_patients={"p2", "p3"},  # p2 is leaked!
        test_patients={"p4"},
    )
    with pytest.raises(ValueError, match="Patient leakage detected between train and val"):
        split_with_leakage.verify_zero_leakage()


def test_audit_patient_leakage_reporting(sample_cohort_df):
    """Verifies audit_patient_leakage detects and reports leaks correctly."""
    split = split_patient_stratified(sample_cohort_df, seed=42)
    clean_audit = audit_patient_leakage(split.train_df, split.val_df, split.test_df)
    assert not clean_audit["has_leakage"]
    assert clean_audit["leakage_train_val"] == []
    assert clean_audit["leakage_train_test"] == []
    assert clean_audit["leakage_val_test"] == []

    # Inject deliberate leakage
    leaked_val_df = pd.concat([split.val_df, split.train_df.iloc[:2]], ignore_index=True)
    dirty_audit = audit_patient_leakage(split.train_df, leaked_val_df, split.test_df)
    assert dirty_audit["has_leakage"]
    assert len(dirty_audit["leakage_train_val"]) > 0


def test_split_determinism(sample_cohort_df):
    """Verifies that same seed yields identical patient partitions."""
    split1 = split_patient_stratified(sample_cohort_df, seed=123)
    split2 = split_patient_stratified(sample_cohort_df, seed=123)

    assert split1.train_patients == split2.train_patients
    assert split1.val_patients == split2.val_patients
    assert split1.test_patients == split2.test_patients
