#!/usr/bin/env python3
"""CLI utility to execute quantitative XAI evaluation against IDRiD pixel lesion ground truth."""

import argparse
import sys
from pathlib import Path

from drdg.experiments.xai import QuantitativeXAIRunner
from drdg.paths import RESULTS_DIR
from drdg.utils.logging import get_logger

logger = get_logger("run_xai")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute quantitative XAI evaluation (Pointing Game, Saliency Recall, Pixel AUROC) against IDRiD masks."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to trained model checkpoint (.pt / .pth). If omitted, an initialized model is used.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="v3",
        choices=["v0", "v1", "v2", "v3"],
        help="Model variant to evaluate (default: v3).",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Path to IDRiD lesion segmentation dataset directory.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR / "xai",
        help=f"Directory to serialize evaluation metrics and maps (default: {RESULTS_DIR / 'xai'}).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run inference on ('cpu', 'cuda', 'mps', or 'auto').",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50,
        help="Maximum number of IDRiD test samples to evaluate (default: 50).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute synthetic dry run without requiring full IDRiD image downloads.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Initializing Quantitative XAI Evaluation Runner...")

    runner = QuantitativeXAIRunner(
        checkpoint_path=args.checkpoint,
        variant=args.variant,
        dataset_dir=args.dataset_dir,
        results_dir=args.results_dir,
        device=args.device,
        max_samples=args.max_samples,
        dry_run=args.dry_run,
    )

    try:
        summary = runner.run()
        logger.info("Evaluation Summary:")
        import json
        logger.info("\n" + json.dumps(summary, indent=2))
        sys.exit(0)
    except Exception as e:
        logger.error(f"Quantitative XAI evaluation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
