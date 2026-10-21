#!/usr/bin/env python3
"""CLI utility to harmonize individual cohort manifests into a unified master fundus metadata table."""

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

from drdg.data.metadata import MasterFundusRecord, records_to_dataframe, validate_records
from drdg.paths import MASTER_METADATA_CSV, PROCESSED_DATA_DIR
from drdg.utils.logging import get_logger

logger = get_logger("build_master_metadata")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Harmonize individual cohort manifests into canonical MasterFundusRecord CSV."
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help=f"Directory containing individual prepared cohort CSVs (default: {PROCESSED_DATA_DIR}).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=MASTER_METADATA_CSV,
        help=f"Path to output canonical metadata CSV (default: {MASTER_METADATA_CSV}).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger.info(f"Scanning for cohort manifests in: {args.processed_root}")

    # Discover cohort CSVs excluding master_fundus_metadata itself
    csv_files = [
        p
        for p in args.processed_root.glob("*_metadata.csv")
        if p.name != args.output_csv.name and not p.name.startswith("master_")
    ]

    # Also check subdirectories (e.g. data/processed/aptos/aptos_metadata.csv)
    sub_csvs = [
        p
        for p in args.processed_root.glob("*/*_metadata.csv")
        if p.name != args.output_csv.name
    ]
    all_csvs = sorted(list(set(csv_files + sub_csvs)))

    if not all_csvs:
        logger.warning(
            f"No individual cohort CSVs (*_metadata.csv) found in {args.processed_root}.\n"
            f"Please run 'python scripts/prepare_dataset.py --dataset all' first or place cohort CSVs in {args.processed_root}."
        )
        return 0

    all_records: List[MasterFundusRecord] = []
    cohort_counts = {}

    for csv_p in all_csvs:
        logger.info(f"Reading cohort manifest: {csv_p}")
        df = pd.read_csv(csv_p)
        cohort_name = csv_p.stem.replace("_metadata", "")
        count = 0

        for _, row in df.iterrows():
            try:
                rec = MasterFundusRecord.from_dict(row.to_dict())
                all_records.append(rec)
                count += 1
            except Exception as e:
                logger.debug(f"Skipping invalid record in {csv_p.name}: {e}")

        cohort_counts[cohort_name] = count
        logger.info(f"Loaded {count} validated records from {csv_p.name}")

    if not all_records:
        logger.error("No valid MasterFundusRecord records could be assembled.")
        return 1

    master_df = records_to_dataframe(all_records)
    validate_records(all_records)

    # Check for duplicate image_id entries
    duplicate_mask = master_df.duplicated(subset=["image_id"], keep=False)
    if duplicate_mask.any():
        num_dupes = duplicate_mask.sum()
        logger.warning(f"Found {num_dupes} duplicate image_ids in master metadata. Deduplicating by image_id.")
        master_df = master_df.drop_duplicates(subset=["image_id"]).reset_index(drop=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    master_df.to_csv(args.output_csv, index=False)

    logger.info("Master metadata successfully assembled and verified:")
    logger.info(f"Total images: {len(master_df)}")
    logger.info(f"Unique patients: {master_df['patient_id'].nunique()}")
    for cohort, count in cohort_counts.items():
        logger.info(f"  - {cohort}: {count} images")
    logger.info(f"Saved master metadata table to: {args.output_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
