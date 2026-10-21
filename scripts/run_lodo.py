#!/usr/bin/env python3
"""CLI utility to execute 6-fold Leave-One-Dataset-Out (LODO) cross-domain evaluation."""

import argparse
import sys
from pathlib import Path

from drdg.experiments.lodo import LODOExperimentRunner
from drdg.paths import CHECKPOINTS_DIR, LODO_FOLDS_DIR, LODO_RESULTS_DIR
from drdg.utils.logging import get_logger

logger = get_logger("run_lodo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute 6-fold Leave-One-Dataset-Out (LODO) generalization benchmark."
    )
    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        choices=["v0", "v1", "v2", "v3"],
        help="Model variant to benchmark: v0 (Global), v1 (Local MIL), v2 (Context), v3 (Ordinal CLM).",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--held-out",
        type=str,
        default=None,
        choices=["aptos", "ddr", "deepdrid", "idrid", "messidor2", "eyepacs"],
        help="Single target cohort to evaluate out-of-distribution.",
    )
    group.add_argument(
        "--all-folds",
        action="store_true",
        help="Sequentially execute all 6 LODO evaluation folds.",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=LODO_FOLDS_DIR,
        help=f"Directory containing pre-generated LODO fold CSVs (default: {LODO_FOLDS_DIR}).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=LODO_RESULTS_DIR,
        help=f"Directory for serialized result JSON/CSV outputs (default: {LODO_RESULTS_DIR}).",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=CHECKPOINTS_DIR,
        help=f"Directory to save/load model checkpoints (default: {CHECKPOINTS_DIR}).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Automatically resume from existing fold checkpoints or JSON results.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run lightweight 1-epoch dry-run on synthetic tensors without loading datasets.",
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
        help="Learning rate for backbone and attention modules (default: 1e-4).",
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
    logger.info(f"Initializing LODO runner for variant: {args.variant}")

    if not args.all_folds and args.held_out is None:
        logger.info("Neither --held-out nor --all-folds specified. Defaulting to fold 0 (held-out: aptos).")
        args.held_out = "aptos"

    runner = LODOExperimentRunner(
        variant=args.variant,
        held_out=args.held_out,
        all_folds=args.all_folds,
        splits_dir=args.splits_dir,
        results_dir=args.results_dir,
        checkpoints_dir=args.checkpoints_dir,
        resume=args.resume,
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
        logger.info("LODO execution completed successfully.")
        return 0
    except Exception as e:
        logger.error(f"LODO execution encountered an error: {e}", exc_info=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
