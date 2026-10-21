#!/usr/bin/env python3
"""CLI utility to download official public fundus datasets."""

import argparse
import sys
from pathlib import Path

from drdg.data.downloaders import DOWNLOADER_MAP, download_dataset
from drdg.utils.logging import get_logger

logger = get_logger("download_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download public diabetic retinopathy datasets into data/raw/<dataset_key>."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset key to download. Options: ddr, deepdrid, idrid, idrid-segmentation, messidor2.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("./data/raw"),
        help="Destination root directory for raw dataset archives (default: ./data/raw).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    key = args.dataset.strip().lower().replace("-", "_")

    if key in ("eyepacs", "aptos", "aptos2019"):
        logger.error(
            f"Dataset '{args.dataset}' requires Kaggle competition rules agreement or manual acquisition.\n"
            f"Please refer to docs/DATA.md for full acquisition instructions and terms of use.\n"
            f"Once acquired, place raw files into: {args.data_root / key}"
        )
        return 1

    normalized_key = "idrid_segmentation" if key in ("idrid_segmentation", "idrid_seg") else key
    if normalized_key not in DOWNLOADER_MAP:
        logger.error(
            f"Unknown dataset '{args.dataset}'. Supported automated keys: "
            f"{list(DOWNLOADER_MAP.keys())} or 'idrid-segmentation'."
        )
        return 1

    target_dir = args.data_root / normalized_key
    target_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Initiating download for '{normalized_key}' into {target_dir}...")
    try:
        result_path = download_dataset(normalized_key, target_dir=target_dir)
        logger.info(f"Successfully downloaded and prepared {normalized_key} at {result_path}")
        return 0
    except Exception as e:
        logger.error(f"Download failed for '{normalized_key}': {e}", exc_info=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
