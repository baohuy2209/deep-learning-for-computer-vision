#!/usr/bin/env python3
"""CLI utility to generate 6 deterministic Leave-One-Dataset-Out (LODO) fold CSVs and summary audit statistics."""

import argparse
import sys
from pathlib import Path

import pandas as pd

from drdg.data.lodo import generate_and_save_lodo_splits
from drdg.paths import LODO_FOLDS_DIR, MASTER_METADATA_CSV
from drdg.utils.logging import get_logger

logger = get_logger("generate_lodo_splits")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 6 deterministic Leave-One-Dataset-Out (LODO) fold CSVs and verify zero leakage."
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=MASTER_METADATA_CSV,
        help=f"Path to master harmonized metadata CSV (default: {MASTER_METADATA_CSV}).",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=LODO_FOLDS_DIR,
        help=f"Output directory for LODO partition CSVs and summary (default: {LODO_FOLDS_DIR}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for source-domain stratified patient partitioning (default: 42).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger.info(f"Loading master metadata from: {args.metadata_csv}")

    if not args.metadata_csv.exists():
        logger.error(
            f"Master metadata file not found at: {args.metadata_csv}\n"
            f"Please run 'python scripts/build_master_metadata.py' or provide a valid CSV path via --metadata-csv."
        )
        return 1

    df = pd.read_csv(args.metadata_csv)
    if "cohort" not in df.columns:
        if "dataset_key" in df.columns:
            df["cohort"] = df["dataset_key"]
        else:
            logger.error("Required column 'cohort' or 'dataset_key' not found in master metadata.")
            return 1

    if "patient_id" not in df.columns or "image_id" not in df.columns:
        logger.error("Required columns 'patient_id' and 'image_id' must be present in master metadata.")
        return 1

    logger.info(
        f"Master metadata loaded: {len(df)} images, {df['patient_id'].nunique()} patients across cohorts: "
        f"{sorted(df['cohort'].unique().tolist())}"
    )

    try:
        summary = generate_and_save_lodo_splits(
            data=df,
            output_dir=args.splits_dir,
            seed=args.seed,
            patient_col="patient_id",
            cohort_col="cohort",
        )

        logger.info("LODO fold generation complete:")
        logger.info(f"Total folds generated: {summary['n_folds']}")
        for cohort, f_info in summary["folds"].items():
            logger.info(
                f"  - Target: {cohort:10s} | Train: {f_info['n_train_images']} | "
                f"Val: {f_info['n_val_images']} | Test: {f_info['n_test_images']} | "
                f"Leakage-free: {f_info['zero_leakage_verified']}"
            )

        logger.info(f"All partition CSVs and lodo_summary.json saved to: {args.splits_dir}")
        return 0

    except Exception as e:
        logger.error(f"Failed to generate LODO splits: {e}", exc_info=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
