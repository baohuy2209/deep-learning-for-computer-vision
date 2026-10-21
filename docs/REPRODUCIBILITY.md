# Reproducibility, Hardware & Determinism Guide

This document details environment setup, deterministic seed control, floating-point precision, hardware configurations, and verification procedures to ensure end-to-end reproducibility of `drdg`.

---

## 1. Environment Setup & Installation

The package requires Python 3.10+ and PyTorch 2.2+.

### Quick Installation:
```bash
# Clone the repository
git clone https://github.com/baohuy2209/deep-learning-for-computer-vision.git
cd deep-learning-for-computer-vision

# Create and activate virtual environment
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
.\venv\Scripts\Activate.ps1

# Install package in editable mode with dependencies
pip install -e .

# Install development & test dependencies
pip install -r requirements.txt
```

---

## 2. Seed Control & Determinism

Stochasticity in neural network training stems from pseudo-random number generators (PRNGs), data loader shuffling, non-deterministic CUDA kernel schedules, and cuDNN algorithm heuristics.

`drdg` centralizes seed control through `drdg.utils.seed.seed_everything`:

```python
from drdg.utils.seed import seed_everything

# Centralize seed across random, numpy, PyTorch CPU, and CUDA
seed_everything(seed=42, deterministic=True)
```

Implementation details:
- `random.seed(seed)`
- `np.random.seed(seed)`
- `torch.manual_seed(seed)`
- `torch.cuda.manual_seed_all(seed)`
- When `deterministic=True`:
  - `torch.backends.cudnn.deterministic = True`
  - `torch.backends.cudnn.benchmark = False`
  - `torch.use_deterministic_algorithms(False)` (defaulted to False to maintain performance while preserving seed-level consistency across runs).

---

## 3. Hardware Requirements & Performance

The codebase supports execution across CPU, Apple Silicon (MPS), and CUDA devices.

| Hardware Tier | Memory | Batch Size | Recommended AMP Mode | Expected Epoch Time (5 Folds Train) |
|---|---|---|---|---|
| **High-End GPU** (A100 / RTX 4090 / 3090) | $\ge 24$ GB | 16--32 | `bfloat16` or `float16` | $\approx 4$--$8$ minutes |
| **Mid-Range GPU** (RTX 3070 / 4060) | 8--12 GB | 4--8 | `float16` | $\approx 15$--$25$ minutes |
| **Local CPU** (Intel i7 / Apple Silicon M-series) | $\ge 16$ GB RAM | 2--4 | `none` (`float32`) | Used for testing & verification |

### Device Selection:
Pass `--device auto`, `--device cuda`, or `--device cpu` to any CLI script:
```bash
python scripts/run_lodo.py --variant v3 --held-out aptos --device cuda
```

---

## 4. PyTorch 2.6+ Checkpoint Serialization Note

In PyTorch 2.6+, `torch.load` defaults to `weights_only=True`, which raises a `pickle.UnpicklingError` when loading checkpoints containing Python dictionaries with numpy scalars or training histories.

`drdg` resolves this systematically in `drdg.training.checkpoint.load_checkpoint` by safely configuring `weights_only=False` for local, trusted checkpoint files.

---

## 5. End-to-End Verification Checklist

To verify that your installation is reproducible, faithful to the research notebooks, and functioning properly on your hardware:

1. **Verify Notebook Preservation**:
   ```bash
   python scripts/verify_notebook_integrity.py
   ```
2. **Execute Full Parity & Unit Test Suite**:
   ```bash
   pytest tests/ tests/parity/ -v
   ```
3. **Execute Model Variant Dry-Runs**:
   ```bash
   python scripts/run_lodo.py --variant v3 --held-out aptos --dry-run --device cpu
   ```
4. **Execute Head Ablation Dry-Run**:
   ```bash
   python scripts/run_head_ablation.py --head clm_qwk --dry-run --device cpu
   ```
5. **Execute XAI Dry-Run**:
   ```bash
   python scripts/run_xai.py --dry-run --device cpu
   ```
6. **Verify Paper Artifact Export**:
   ```bash
   python scripts/export_paper_artifacts.py
   ```

