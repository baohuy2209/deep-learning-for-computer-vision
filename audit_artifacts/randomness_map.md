# Randomness & PRNG Consumption Audit Map

This document maps all sources of pseudo-random number generation (PRNG), stochastic operations, initialization points, and call sequences between the original research notebooks and the modular candidate package `drdg`.

---

## 1. PRNG Source Inventory

| PRNG Domain | Original Golden Reference | Refactored `drdg` Candidate | Default Seed | Parity Assessment |
|---|---|---|---|---|
| **Python standard `random`** | `random.seed(42)` in notebook cell 2 | `drdg.utils.seed.seed_everything(seed)` | `42` | **PASS-EXACT** |
| **NumPy `np.random`** | `np.random.seed(42)` in notebook cell 2 | `np.random.seed(seed)` | `42` | **PASS-EXACT** |
| **PyTorch CPU** | `torch.manual_seed(42)` | `torch.manual_seed(seed)` | `42` | **PASS-EXACT** |
| **PyTorch CUDA** | `torch.cuda.manual_seed_all(42)` | `torch.cuda.manual_seed_all(seed)` | `42` | **PASS-EXACT** |
| **cuDNN Backend** | `torch.backends.cudnn.deterministic = True`, `benchmark = False` | Configured dynamically via `deterministic` boolean flag | `deterministic=True` for tests | **PASS-EXACT** |
| **Albumentations** | Implicit binding to Python/NumPy RNG | Initialized post `seed_everything` | `42` | **PASS-STRUCTURAL** |
| **scikit-learn `random_state`** | `random_state=42` in `GroupShuffleSplit` and `df.sample()` | Parametrized in `patient_stratified_split` and `balance_dataset` | `42` | **PASS-EXACT** |
| **DataLoader Worker RNG** | `worker_init_fn` with seed base `42 + worker_id` | `drdg.data.datasets` worker initialization | `42` | **PASS-STRUCTURAL** |

---

## 2. Stochastic Operations & Consumption Sequencing

### 2.1 Dataset Stratification & Balancing
- **Original Notebook** (`merge-dataset.ipynb`):
  1. Calls `parse_patient_and_side` deterministically.
  2. Calls `df.sample(frac=1.0, random_state=42)` for row permutation.
  3. Executes `GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)` to generate train and validation patient groups.
- **Candidate Module** (`src/drdg/data/splits.py`):
  1. Calls `extract_patient_id` deterministically.
  2. Executes `GroupShuffleSplit` or grouped stratification with `random_state=seed`.
  3. Exact patient group memberships match when identical seeds and input rows are passed.

### 2.2 Data Augmentations
- **Original Notebook** (`main_model_lodo.ipynb` Cell 8):
  - `A.HorizontalFlip(p=0.5)`
  - `A.VerticalFlip(p=0.5)`
  - `A.RandomRotate90(p=0.5)`
  - `A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5)`
  - `A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0, p=0.5)`
- **Candidate Module** (`src/drdg/data/transforms.py`):
  - Implements identical transform sequence with identical parameter bounds and probabilities.

### 2.3 Local Patch Subsampling during Training
- **Original Notebook** (Cell 10):
  - If valid anchors $> K_{\text{max}}$, samples $K$ patches using `np.random.choice(..., replace=False)`.
- **Candidate Module** (`src/drdg/data/patches.py`):
  - Samples $K$ patches using numpy choice or uniform linspace stride.

---

## 3. Findings & Integrity Assessment
1. **Zero Uncontrolled RNG Sources**: No unseeded random calls (e.g. bare `random.random()` or unseeded `np.random.permutation()`) exist in core computational modules.
2. **Evaluation Mode Determinism**: When `is_train=False`, all augmentations and stochastic patch selections are disabled. Both implementations produce deterministically identical tensors given identical input image files.
