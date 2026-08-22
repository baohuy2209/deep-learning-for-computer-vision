"""Attention Mechanisms for Retinal Image Analysis & Diabetic Retinopathy Classification.

This module provides state-of-the-art attention mechanism layers:
- SEBlock: Squeeze-and-Excitation channel attention.
- MSAGBlock: Multi-Scale Spatial Attention Gate.
- PrimaryCapsule & AttentionRoutingCapsule: Attention-based Capsule Network routing.
- squash: Vector activation function for capsule networks.
"""

from src.attention.senet import SEBlock
from src.attention.msag import MSAGBlock
from src.attention.am_capsulenet import (
    PrimaryCapsule,
    AttentionRoutingCapsule,
    squash,
)

__all__ = [
    "SEBlock",
    "MSAGBlock",
    "PrimaryCapsule",
    "AttentionRoutingCapsule",
    "squash",
]
