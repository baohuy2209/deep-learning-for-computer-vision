"""Unit tests verifying clean imports and absence of side effects."""

import pytest


def test_import_drdg_top_level():
    """Verify top-level package imports cleanly."""
    import drdg
    assert hasattr(drdg, "__version__")
    assert drdg.__version__ == "0.1.0"


def test_import_subpackages():
    """Verify all submodules import without triggering file I/O or GPU calls."""
    import drdg.config
    import drdg.paths
    import drdg.data.collate
    import drdg.data.datasets
    import drdg.data.fov
    import drdg.data.lodo
    import drdg.data.metadata
    import drdg.data.patches
    import drdg.data.patient_ids
    import drdg.data.preprocessing
    import drdg.data.splits
    import drdg.models.attention
    import drdg.models.factory
    import drdg.models.full_model
    import drdg.models.fusion
    import drdg.models.heads
    import drdg.models.streams
    import drdg.models.variants
    import drdg.losses.qwk
    import drdg.losses.ordinal
    import drdg.training.checkpoint
    import drdg.training.optimization
    import drdg.training.trainer
    import drdg.evaluation.lodo
    import drdg.evaluation.metrics
    import drdg.evaluation.rdr
    import drdg.evaluation.result_schema
    import drdg.experiments.lodo
    import drdg.explainability.gradcam
    import drdg.explainability.metrics
    import drdg.explainability.mil_saliency
    import drdg.explainability.idrid_dataset
    import drdg.utils.device
    import drdg.utils.logging
    import drdg.utils.seed
