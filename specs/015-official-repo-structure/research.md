# Research & Technical Decisions: Official Repository Architecture

**Feature**: [spec.md](file:///d:/Github/deep-learning-for-computer-vision/specs/015-official-repo-structure/spec.md) | **Date**: 2026-09-19  
**Paper**: *Deconstructing Domain Shift in Diabetic Retinopathy Grading: Multi-Scale MIL with Consistent Ordinal Regression* (ICCPR Conference)

---

## 1. Package Architecture & Code Reuse Strategy

### Decision: Modular Python Package `dr_multiscale_mil` under `src/`
- **Chosen Approach**: Create a clean, production-grade research package `src/dr_multiscale_mil/` that directly imports and modularizes existing, verified code from `notebooks/diabetic_retinopathy/pipeline/full/` and `scratch/extracted_defs_from_main_model.py`.
- **Rationale**: 
  1. Adheres to Principle V (Strict Preservation & Zero-Regression) of the Research Constitution: absolutely no changes to math, losses, network layers, or preprocessing formulas.
  2. Eliminates duplicated code across multiple notebooks (`main_model (6).ipynb`, `main_model_head_ablation.ipynb`, `main_model_lodo.ipynb`).
  3. Enables clean imports for both command-line scripts (`src/dr_multiscale_mil/...`) and interactive Google Colab notebooks (`from dr_multiscale_mil.models import MultiScaleMILNet`).
- **Alternatives Considered**:
  - *Keeping code in monolithic notebooks*: Rejected because notebooks are difficult to version-control, cannot easily be run via CI/CD or batch cluster jobs, and make code sharing/review for ICCPR difficult.
  - *Rewriting modules from scratch*: Strictly prohibited by the user and Constitution Principle V to avoid introducing mathematical or algorithmic regressions.

---

## 2. Configuration Management

### Decision: Structured YAML Configurations in `configs/`
- **Chosen Approach**: Use modular YAML files grouped by concern:
  - `configs/data/`: Dataset paths, ROI crop tolerance, and LODO fold definitions.
  - `configs/model/`: Dual-branch parameters (backbone, patch size 224/300, patch count 4/8, attention type CBAM/MSAG, prediction head).
  - `configs/train/`: Optimizer (AdamW), learning rate schedule (Cosine Annealing), epochs, early stopping on QWK.
  - `configs/ablations/`: Specific overrides for head ablations (Nominal, CORAL, Consistent Ordinal, Regression) and preprocessing ablations.
- **Rationale**: 
  - Allows researchers to reproduce specific paper experiments with a single `--config` flag without hardcoded script variables.
  - Decouples hyperparameter tuning from model code.
- **Alternatives Considered**:
  - *Pure CLI flags (`argparse` only)*: Rejected because passing 20+ arguments via CLI is error-prone and hard to document in the paper's appendix.
  - *Python config files (`config.py`)*: Rejected because YAML is language-agnostic and easier for non-Python practitioners to inspect.

---

## 3. Data Harmonization & Zero-Leakage Splitting

### Decision: Two-Tier Storage with Namespaced Patient Splitting
- **Chosen Approach**:
  1. **Tier 1 (Native Cropped Storage)**: Raw images from 6 cohorts are cropped to fundus circular ROI (removing uninformative black borders) and saved to `data/raw/hybrid/native_cropped/{dataset_key}/{image_id}.jpg` at full native resolution with JPEG quality 95. No premature downsampling occurs at this stage.
  2. **Tier 2 (Frozen Metadata Splits)**: Patient IDs are explicitly namespaced (`{dataset_key}::{patient_id}`) to prevent ID collisions across cohorts. All splits are generated strictly at patient level using `GroupShuffleSplit` with fixed seeds (`SEED=42`), ensuring $Train \cap Val = \emptyset$, $Train \cap Test = \emptyset$, and $Val \cap Test = \emptyset$.
  3. **Multi-Source Balanced Sampling**: Training sets are quota-balanced (1,500 to 2,500 images per DR grade 0-4) using proportional multi-source allocation to prevent dominant datasets (e.g., EyePACS) from drowning out smaller datasets (e.g., IDRiD, DeepDRiD). Validation and test sets remain strictly unperturbed at natural clinical prevalence.
- **Rationale**:
  - Exactly preserves the verified data preparation logic in `merge-dataset.ipynb`, `merge_dataset_lodo.ipynb`, and `scratch/generate_lodo_splits.py`.
- **Alternatives Considered**:
  - *Image-level random splitting*: Strongly rejected due to patient leakage (left and right eyes of the same patient appearing in both train and test sets, artificially inflating performance).

---

## 4. Model Architecture & Ordinal Formulation

### Decision: Multi-Scale MIL with Dual-Stream & Consistent Ordinal Head
- **Chosen Approach**:
  1. **Global Context Stream**: Whole-fundus image resized to $512 \times 512$, processed with Green-channel contrast enhancement (Ben Graham / CLAHE), passed through a CNN backbone (EfficientNet-B0 default) to capture hemispheric vascular topology and 4-quadrant lesion patterns.
  2. **Local Patch Stream**: Multiple high-resolution patches extracted at native resolution (e.g., 4 or 8 patches of size $224 \times 224$ or $300 \times 300$) using saliency/gradient attention or dense multi-scale sampling to capture microaneurysms, hemorrhages, and exudates without downsampling blur.
  3. **Multi-Scale Attention MIL Pooling**: Aggregates patch embeddings via learnable attention gating (MSAG / CBAM) into a compact representation.
  4. **Consistent Ordinal Prediction Head**: Formulates DR grading (0 to 4) as an ordinal classification task, outputting 4 binary cumulative rank probabilities ($P(y > k)$ for $k \in \{0, 1, 2, 3\}$) constrained to be monotonically non-increasing, optimized directly for Quadratic Weighted Kappa (QWK).
- **Rationale**:
  - Matches the exact core novelty presented in the ICCPR paper title and text.

---

## 5. Google Colab & Local Workstation Interoperability

### Decision: Dynamic Path Resolution & Hardware Fallback
- **Chosen Approach**:
  - Detect runtime environment via `'google.colab' in sys.modules or os.path.exists('/content')`.
  - On Colab: Automate Google Drive mounting (`/content/drive/MyDrive/...`), unzip pre-generated splits or datasets to `/content/` for high-speed local NVMe I/O, and optimize DataLoader with `cv2.setNumThreads(0)` and `pin_memory=True`.
  - On Local: Resolve paths relative to workspace root (`data/`, `checkpoints/`, `results/`).
  - Seamlessly support CUDA GPU, MPS (Apple Silicon), and CPU.
- **Rationale**:
  - Reviewers often evaluate code on Google Colab free/Pro tiers. Seamless execution without manual path edits is critical for review score.
