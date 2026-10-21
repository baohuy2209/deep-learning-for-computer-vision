#!/usr/bin/env python3
"""CLI utility to standardize raw cohorts into native circular cropped ROIs and produce cohort CSVs."""

import argparse
import sys
from pathlib import Path

import pandas as pd

from drdg.data.metadata import records_to_dataframe
from drdg.data.preparation import PREPARATION_MAP, prepare_dataset
from drdg.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR
from drdg.utils.logging import get_logger

logger = get_logger("prepare_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standardize raw cohort images and generate intermediate cohort CSVs."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Cohort key: aptos, ddr, deepdrid, idrid, idrid-segmentation, messidor2, eyepacs, or all.",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=RAW_DATA_DIR,
        help=f"Input directory containing raw dataset folders (default: {RAW_DATA_DIR}).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help=f"Destination directory for processed assets and CSVs (default: {PROCESSED_DATA_DIR}).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    key = args.dataset.strip().lower().replace("-", "_")

    logger.info(f"Starting dataset preparation for '{key}'...")
    logger.info(f"Raw root: {args.raw_root}, Output root: {args.output_root}")

    try:
        records = prepare_dataset(
            dataset_key=key,
            raw_dir=args.raw_root if key != "all" else args.raw_root,
            output_dir=args.output_root if key != "all" else args.output_root,
        )

        if not records:
            logger.warning(f"No records produced for dataset '{key}'. Check that raw images/labels exist.")
            return 0

        df = records_to_dataframe(records)
        out_csv = args.output_root / f"{key}_metadata.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        logger.info(f"Successfully prepared {len(records)} records for '{key}'. Saved manifest to {out_csv}")
        return 0

    except Exception as e:
        logger.error(f"Error during dataset preparation for '{key}': {e}", exc_info=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
