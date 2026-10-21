"""Unit tests for Leave-One-Dataset-Out (LODO) split generation and zero-leakage assertions.

Tests the bipartite fold structure matching merge_dataset_lodo.ipynb:
  - 3 splits per fold: train, val, test
  - CSV naming: fold_{fold_idx}_{fold_slug}_{split}.csv
  - Zero patient leakage across all partitions
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from drdg.data.lodo import LODOFold, generate_and_save_lodo_splits, generate_lodo_folds


@pytest.fixture
def multi_cohort_mock_df():
    """Generates synthetic multi-cohort DataFrame simulating all 6 cohorts."""
    cohorts = ["aptos", "ddr", "deepdrid", "idrid", "messidor2", "eyepacs"]
    records = []
    for c_idx, cohort in enumerate(cohorts):
        # 10 patients per cohort, 2 eyes per patient = 20 images per cohort
        for p_idx in range(10):
            patient_id = f"{cohort}_p{p_idx:03d}"
            for eye in ["left", "right"]:
                records.append({
                    "image_id": f"{cohort}_{p_idx}_{eye}",
                    "patient_id": patient_id,
                    "cohort": cohort,
                    "dr_grade": (p_idx + c_idx) % 5,
                    "eye": eye,
                })
    return pd.DataFrame(records)


def test_generate_lodo_folds_structure(multi_cohort_mock_df):
    """Verifies that 6 folds are created and target cohorts are isolated."""
    folds = generate_lodo_folds(multi_cohort_mock_df, seed=42)

    assert len(folds) == 6
    expected_targets = {"aptos", "ddr", "deepdrid", "idrid", "messidor2", "eyepacs"}
    observed_targets = {f.target_cohort for f in folds}
    assert observed_targets == expected_targets

    for fold in folds:
        # Target test set must only contain the target cohort
        assert set(fold.target_test_df["cohort"].unique()) == {fold.target_cohort}

        # Source sets must NOT contain the target cohort
        assert fold.target_cohort not in fold.source_train_df["cohort"].unique()
        assert fold.target_cohort not in fold.source_val_df["cohort"].unique()

        # Target test set must contain all images of that target cohort
        cohort_img_count = len(multi_cohort_mock_df[multi_cohort_mock_df["cohort"] == fold.target_cohort])
        assert len(fold.target_test_df) == cohort_img_count


def test_lodo_zero_leakage_verification(multi_cohort_mock_df):
    """Verifies that each fold passes zero-leakage assertions.

    Matches merge_dataset_lodo.ipynb Cell 7 assertions:
      assert len(train_pids.intersection(val_pids)) == 0
      assert len(train_pids.intersection(test_pids)) == 0
      assert len(val_pids.intersection(test_pids)) == 0
    """
    folds = generate_lodo_folds(multi_cohort_mock_df, seed=42)

    for fold in folds:
        # Should execute without raising ValueError
        fold.verify_zero_leakage()

        target_pats = set(fold.target_test_df["patient_id"].unique())
        source_train_pats = set(fold.source_train_df["patient_id"].unique())
        source_val_pats = set(fold.source_val_df["patient_id"].unique())

        # Target isolation from all source partitions
        assert len(target_pats.intersection(source_train_pats)) == 0
        assert len(target_pats.intersection(source_val_pats)) == 0

        # Source train vs val disjointness
        assert len(source_train_pats.intersection(source_val_pats)) == 0


def test_lodo_verify_zero_leakage_catches_overlap(multi_cohort_mock_df):
    """Verifies that injected patient overlap triggers ValueError."""
    folds = generate_lodo_folds(multi_cohort_mock_df, seed=42)
    fold0 = folds[0]

    # Create artificial leakage by adding target row to source_train
    leaked_source_train = pd.concat([fold0.source_train_df, fold0.target_test_df.iloc[:1]], ignore_index=True)
    corrupted_fold = LODOFold(
        fold_idx=fold0.fold_idx,
        target_cohort=fold0.target_cohort,
        fold_slug=fold0.fold_slug,
        source_cohorts=fold0.source_cohorts,
        source_train_df=leaked_source_train,
        source_val_df=fold0.source_val_df,
        target_test_df=fold0.target_test_df,
        leakage_audit={"has_leakage": True},
    )

    with pytest.raises(ValueError, match="PATIENT LEAKAGE"):
        corrupted_fold.verify_zero_leakage()


def test_generate_and_save_lodo_splits(multi_cohort_mock_df, tmp_path):
    """Verifies serialization of LODO fold CSVs and summary files.

    Checks that 18 CSV files (6 folds × 3 splits) + summary files are created,
    matching merge_dataset_lodo.ipynb output structure.
    """
    summary = generate_and_save_lodo_splits(multi_cohort_mock_df, output_dir=tmp_path, seed=42)

    assert summary["all_zero_leakage_verified"] is True
    assert summary["n_folds"] == 6

    # Verify summary JSON
    summary_file = tmp_path / "lodo_summary.json"
    assert summary_file.exists()
    with open(summary_file, "r", encoding="utf-8") as f:
        loaded_summary = json.load(f)
    assert loaded_summary["n_folds"] == 6

    # Verify summary CSV (matching notebook's lodo_splits_summary.csv)
    summary_csv = tmp_path / "lodo_splits_summary.csv"
    assert summary_csv.exists()
    summary_df = pd.read_csv(summary_csv)
    assert len(summary_df) == 6

    # Verify CSV files exist for each fold with correct naming
    # Naming: fold_{fold_idx}_{fold_slug}_{split}.csv
    cohorts_sorted = sorted(["aptos", "ddr", "deepdrid", "idrid", "messidor2", "eyepacs"])
    for fold_idx, cohort in enumerate(cohorts_sorted):
        assert (tmp_path / f"fold_{fold_idx}_{cohort}_train.csv").exists(), \
            f"Missing fold_{fold_idx}_{cohort}_train.csv"
        assert (tmp_path / f"fold_{fold_idx}_{cohort}_val.csv").exists(), \
            f"Missing fold_{fold_idx}_{cohort}_val.csv"
        assert (tmp_path / f"fold_{fold_idx}_{cohort}_test.csv").exists(), \
            f"Missing fold_{fold_idx}_{cohort}_test.csv"

    # Verify total file count: 18 CSVs + 2 summary files
    csv_files = list(tmp_path.glob("fold_*.csv"))
    assert len(csv_files) == 18, f"Expected 18 fold CSVs, got {len(csv_files)}: {[f.name for f in csv_files]}"
