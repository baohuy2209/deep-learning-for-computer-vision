"""Behavioral parity test for 6-Fold Leave-One-Dataset-Out (LODO) split generation.

Validates exact alignment with merge_dataset_lodo.ipynb:
  - 18 CSV files in data/processed/lodo_folds/ (6 folds × 3 splits: train, val, test)
  - Zero patient leakage: Train ∩ Val = ∅, Train ∩ Test = ∅, Val ∩ Test = ∅
  - Zero presence of legacy 4-way split files (source_test, target_test)
"""

from pathlib import Path
import pandas as pd
import pytest

from drdg.paths import LODO_FOLDS_DIR

CANONICAL_COHORTS = ["aptos", "idrid", "messidor2", "ddr", "eyepacs", "deepdrid"]
SPLITS = ["train", "val", "test"]


@pytest.mark.parity
def test_lodo_18_csv_files_exist():
    """Verify that all 18 authentic fold CSVs exist in data/processed/lodo_folds/."""
    assert LODO_FOLDS_DIR.exists(), f"LODO_FOLDS_DIR missing: {LODO_FOLDS_DIR}"

    missing_files = []
    for idx, cohort in enumerate(CANONICAL_COHORTS):
        for split in SPLITS:
            expected_csv = LODO_FOLDS_DIR / f"fold_{idx}_{cohort}_{split}.csv"
            if not expected_csv.exists():
                missing_files.append(expected_csv.name)

    assert not missing_files, f"Missing LODO fold CSVs in {LODO_FOLDS_DIR}: {missing_files}"


@pytest.mark.parity
def test_no_legacy_4way_split_files():
    """Assert that no legacy 4-way split files (source_test, target_test) exist."""
    invalid_files = list(LODO_FOLDS_DIR.glob("*source_test*.csv")) + list(
        LODO_FOLDS_DIR.glob("*target_test*.csv")
    )
    assert not invalid_files, f"Found forbidden legacy 4-way split CSVs: {invalid_files}"


@pytest.mark.parity
def test_lodo_summary_csv_schema():
    """Verify lodo_splits_summary.csv exists and has 6 fold rows."""
    summary_path = LODO_FOLDS_DIR / "lodo_splits_summary.csv"
    assert summary_path.exists(), f"Summary CSV missing: {summary_path}"

    df = pd.read_csv(summary_path)
    assert len(df) == 6, f"Expected 6 fold summary records, got {len(df)}"
    required_cols = {"fold", "held_out_dataset", "train_images", "val_images", "test_images"}
    assert required_cols.issubset(df.columns), f"Summary missing columns: {required_cols - set(df.columns)}"


@pytest.mark.parity
def test_zero_patient_leakage_across_all_18_csvs():
    """Verify strictly zero patient leakage for every fold in the physical repository."""
    for idx, cohort in enumerate(CANONICAL_COHORTS):
        p_train = LODO_FOLDS_DIR / f"fold_{idx}_{cohort}_train.csv"
        p_val = LODO_FOLDS_DIR / f"fold_{idx}_{cohort}_val.csv"
        p_test = LODO_FOLDS_DIR / f"fold_{idx}_{cohort}_test.csv"

        df_train = pd.read_csv(p_train)
        df_val = pd.read_csv(p_val)
        df_test = pd.read_csv(p_test)

        train_patients = set(df_train["patient_id"].unique())
        val_patients = set(df_val["patient_id"].unique())
        test_patients = set(df_test["patient_id"].unique())

        inter_train_val = train_patients.intersection(val_patients)
        inter_train_test = train_patients.intersection(test_patients)
        inter_val_test = val_patients.intersection(test_patients)

        assert not inter_train_val, f"Fold {idx} ({cohort}): Train vs Val patient leakage: {len(inter_train_val)}"
        assert not inter_train_test, f"Fold {idx} ({cohort}): Train vs Test patient leakage: {len(inter_train_test)}"
        assert not inter_val_test, f"Fold {idx} ({cohort}): Val vs Test patient leakage: {len(inter_val_test)}"


@pytest.mark.parity
def test_held_out_cohort_isolation():
    """Verify that test split contains ONLY held-out cohort data, and train/val contain NONE."""
    for idx, cohort in enumerate(CANONICAL_COHORTS):
        p_train = LODO_FOLDS_DIR / f"fold_{idx}_{cohort}_train.csv"
        p_val = LODO_FOLDS_DIR / f"fold_{idx}_{cohort}_val.csv"
        p_test = LODO_FOLDS_DIR / f"fold_{idx}_{cohort}_test.csv"

        df_train = pd.read_csv(p_train)
        df_val = pd.read_csv(p_val)
        df_test = pd.read_csv(p_test)

        cohort_col = "fold_key" if "fold_key" in df_test.columns else ("dataset_key" if "dataset_key" in df_test.columns else "cohort")

        test_cohorts = {str(c).lower() for c in df_test[cohort_col].unique()}
        train_cohorts = {str(c).lower() for c in df_train[cohort_col].unique()}
        val_cohorts = {str(c).lower() for c in df_val[cohort_col].unique()}

        assert cohort.lower() in test_cohorts, f"Fold {idx}: Target cohort '{cohort}' not in test split"
        assert cohort.lower() not in train_cohorts, f"Fold {idx}: Target cohort '{cohort}' leaked into train split"
        assert cohort.lower() not in val_cohorts, f"Fold {idx}: Target cohort '{cohort}' leaked into val split"
