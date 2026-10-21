# Quickstart & Verification Guide: Restore Exact Notebook-to-Python Parity

**Feature**: `018-restore-notebook-parity` | **Date**: 2026-09-24

This guide outlines runnable scenarios to prove that the modular pure-Python package `drdg` achieves complete algorithmic, numerical, and workflow parity with the canonical research notebooks.

---

## 1. Prerequisites & Installation

Activate the Python virtual environment and install `drdg` in editable development mode:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -e .

# Verify clean package import
python -c "import drdg; print(f'DRDG version: {drdg.__version__}')"
```

---

## 2. Validate LODO Split File Structure (18 CSVs)

Verify that the 18 pre-generated LODO fold CSVs exist in `data/processed/lodo_folds/` adhering to the authentic 3-split schema (`train`, `val`, `test`):

```bash
python -c "
from pathlib import Path

lodo_dir = Path('data/processed/lodo_folds')
cohorts = ['aptos', 'idrid', 'messidor2', 'ddr', 'eyepacs', 'deepdrid']
splits = ['train', 'val', 'test']

missing = []
for idx, cohort in enumerate(cohorts):
    for split in splits:
        p = lodo_dir / f'fold_{idx}_{cohort}_{split}.csv'
        if not p.exists():
            missing.append(str(p))

if missing:
    print(f'[FAIL] Missing {len(missing)} fold CSVs:\n' + '\n'.join(missing))
else:
    print(f'[PASS] All 18 authentic LODO fold CSVs exist with zero missing partitions.')
"
```

---

## 3. Run Behavioral Parity Tests

Execute the automated parity test suite verifying numerical, gradient, and architectural equivalence against notebook references:

```bash
# Run all behavioral parity tests on CPU
pytest tests/parity/ -v --tb=short

# Run specific sub-modules
pytest tests/parity/test_lodo_split_parity.py -v
pytest tests/parity/test_heads_parity.py -v
pytest tests/parity/test_losses_parity.py -v
pytest tests/parity/test_training_step_parity.py -v
```

**Expected Outcome**: All tests pass with exit code 0, confirming FP32 maximum absolute error $< 1\times 10^{-6}$ for forward passes and $< 1\times 10^{-5}$ for backward gradients.

---

## 4. Dry-Run Single LODO Fold Execution

Test the LODO training and evaluation runner (`scripts/run_lodo.py`) in dry-run mode (1 fast epoch on synthetic batch data) without requiring GPU resources:

```bash
python scripts/run_lodo.py --variant v3 --held-out aptos --dry-run
```

**Expected Outcome**:
- Clean initialization of V3 Dual-Branch architecture (ResNet-50 + CBAM + NonLocal + Quadrant Tokens + EfficientNet-B0 Local MIL + CLM QWK Head).
- Successful execution of 1 training epoch and validation pass.
- Successful threshold calibration $\tau^*$ on `in_domain_val`.
- Evaluation of `ood_test` using frozen $\tau^*$.
- JSON result serialized to `results/lodo/v3_fold0_aptos.json` following the bipartite schema.

---

## 5. Dry-Run Controlled Head Ablation

Execute the controlled 5-head ablation runner (`scripts/run_head_ablation.py`) in dry-run mode:

```bash
python scripts/run_head_ablation.py --head clm_qwk --dry-run
```

**Expected Outcome**:
- Successful instantiation of `CumulativeLinkModelQWKHead`.
- Differentiable cumulative link loss + QWK loss computation.
- JSON metrics saved to `results/head_ablation/clm_qwk_results.json`.

---

## 6. Verify Absolute Notebook Preservation

Verify that the 5 canonical Jupyter research notebooks have not been modified or turned into wrappers:

```bash
git status -- notebooks/diabetic_retinopathy/
```

**Expected Outcome**: Git reports zero modifications to the canonical `.ipynb` files, certifying 100% preservation of historical research provenance.
