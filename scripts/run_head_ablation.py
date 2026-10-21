#!/usr/bin/env python3
"""CLI utility to execute controlled 5-head benchmark on standardized ResNet-50 features."""

import argparse
import sys
from pathlib import Path

from drdg.experiments.head_ablation import CANDIDATE_HEADS, HeadAblationRunner
from drdg.paths import CHECKPOINTS_DIR, HEAD_ABLATION_RESULTS_DIR, MASTER_METADATA_CSV
from drdg.utils.logging import get_logger

logger = get_logger("run_head_ablation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute controlled output-head ablation benchmark on standardized ResNet-50 backbones."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--head",
        type=str,
        default=None,
        choices=CANDIDATE_HEADS,
        help=f"Single head to train and benchmark: {CANDIDATE_HEADS}.",
    )
    group.add_argument(
        "--all-heads",
        action="store_true",
        help="Sequentially train and benchmark all 5 candidate heads.",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=MASTER_METADATA_CSV,
        help=f"Path to master metadata CSV (default: {MASTER_METADATA_CSV}).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=HEAD_ABLATION_RESULTS_DIR,
        help=f"Directory to serialize ablation summary metrics (default: {HEAD_ABLATION_RESULTS_DIR}).",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=CHECKPOINTS_DIR / "head_ablation",
        help="Directory to save head checkpoints (default: ./checkpoints/head_ablation).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute 1-epoch dry run using synthetic tensors without loading images.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for training and evaluation (default: 4).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Maximum training epochs (default: 30).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for head and backbone (default: 1e-4).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Execution device: auto, cuda, cpu (default: auto).",
    )
    parser.add_argument(
        "--amp",
        type=str,
        default="auto",
        help="Mixed precision mode: auto, fp16, bf16, none (default: auto).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.all_heads and args.head is None:
        logger.info("Neither --head nor --all-heads specified. Defaulting to 'softmax_ce'.")
        args.head = "softmax_ce"

    logger.info(f"Initializing Head Ablation runner for: {args.head if args.head else 'ALL 5 HEADS'}")
    runner = HeadAblationRunner(
        head=args.head,
        all_heads=args.all_heads,
        results_dir=args.results_dir,
        checkpoints_dir=args.checkpoints_dir,
        metadata_csv=args.metadata_csv,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        amp=args.amp,
    )

    try:
        summary = runner.run()
        logger.info("Head ablation benchmark completed successfully.")
        return 0
    except Exception as e:
        logger.error(f"Head ablation encountered an error: {e}", exc_info=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
