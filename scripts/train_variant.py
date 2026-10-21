"""CLI single-variant training entry point for diabetic retinopathy models.

Usage:
    python scripts/train_variant.py --variant v3 --epochs 50 --batch_size 16
    python scripts/train_variant.py --variant v0 --dry_run
"""

import argparse
import sys
from typing import List, Optional

import torch

from drdg.config import ModelConfig, TrainingConfig
from drdg.models.factory import build_variant
from drdg.training.optimization import build_optimizer, build_scheduler
from drdg.training.trainer import DRTrainer
from drdg.utils.device import get_device_and_amp
from drdg.utils.logging import get_logger
from drdg.utils.seed import set_seed

logger = get_logger("train_variant")


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an architectural variant (V0-V3) for diabetic retinopathy grading."
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="v3",
        choices=["v0", "v1", "v2", "v3"],
        help="Ablation variant: v0 (baseline), v1 (local MIL), v2 (global context), v3 (ordinal CLM)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Training batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Base learning rate.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay for regularization.",
    )
    parser.add_argument(
        "--qwk_weight",
        type=float,
        default=0.4,
        help="Target lambda weight for continuous QWK loss.",
    )
    parser.add_argument(
        "--qwk_warmup_epochs",
        type=int,
        default=5,
        help="Epochs for linear QWK weight warmup.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="Directory to persist checkpoints.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Run 2 synthetic mini-batches on CPU to verify model instantiation and backward pass.",
    )
    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> int:
    parsed = parse_args(args)
    set_seed(parsed.seed)
    device, use_amp = get_device_and_amp()

    logger.info(f"Initializing variant: {parsed.variant.upper()} on device: {device} (AMP: {use_amp})")

    model = build_variant(
        variant=parsed.variant,
        pretrained=not parsed.dry_run,
        qwk_weight=parsed.qwk_weight,
    )
    model.to(device)

    if parsed.dry_run:
        logger.info("Executing dry-run forward and backward pass with synthetic tensors...")
        batch_size = 2
        global_img = torch.randn(batch_size, 3, 512, 512, device=device)
        targets = torch.tensor([0, 3], device=device)

        if parsed.variant in ("v1", "v2", "v3"):
            local_patches = torch.randn(batch_size, 4, 3, 224, 224, device=device)
            patch_mask = torch.ones(batch_size, 4, device=device)
            outputs = model(global_img, local_patches=local_patches, patch_mask=patch_mask, targets=targets, return_loss=True)
        else:
            outputs = model(global_img, targets=targets, return_loss=True)

        loss = outputs["loss"]
        loss.backward()
        logger.info(f"Dry run successful! Loss: {loss.item():.4f}, Preds: {outputs['preds'].tolist()}")
        return 0

    logger.info(f"Model {parsed.variant.upper()} ready for full training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
