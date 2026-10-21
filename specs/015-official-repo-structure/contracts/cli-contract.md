# CLI Interface Contract: Official ICCPR Repository

**Feature**: [spec.md](file:///d:/Github/deep-learning-for-computer-vision/specs/015-official-repo-structure/spec.md) | **Date**: 2026-09-19  
**Paper**: *Deconstructing Domain Shift in Diabetic Retinopathy Grading: Multi-Scale MIL with Consistent Ordinal Regression* (ICCPR Conference)

---

## 1. Data Preprocessing & Split Generation

### 1.1 `dr_multiscale_mil.data.preprocess` (`scripts/01_preprocess_data.sh`)
Executes fundus circular ROI cropping across all 6 raw cohorts to produce the `native_cropped` storage tier.

```bash
python -m dr_multiscale_mil.data.preprocess \
    --raw-dir data/raw \
    --output-dir data/raw/hybrid/native_cropped \
    --tolerance 7 \
    --num-workers 8 \
    --jpeg-quality 95
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--raw-dir` | `str` | `data/raw` | Root directory containing raw dataset folders |
| `--output-dir` | `str` | `data/raw/hybrid/native_cropped` | Destination directory for cropped images |
| `--tolerance` | `int` | `7` | Black border pixel intensity threshold |
| `--num-workers` | `int` | `8` | Parallel worker threads for image processing |
| `--jpeg-quality`| `int` | `95` | Quality factor for saved JPEG files |

---

### 1.2 `dr_multiscale_mil.data.make_splits` (`scripts/02_make_splits.sh`)
Generates patient-level zero-leakage splits for in-domain 70/15/15 or 6-fold LODO benchmarks.

```bash
# Generate In-Domain 70/15/15 Split
python -m dr_multiscale_mil.data.make_splits \
    --mode in_domain \
    --output-dir data/raw/hybrid \
    --seed 42

# Generate 6-Fold LODO Splits
python -m dr_multiscale_mil.data.make_splits \
    --mode lodo \
    --output-dir data/processed/lodo_folds \
    --zip-archive \
    --seed 42
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--mode` | `str` | `lodo` | Split mode: `'in_domain'` or `'lodo'` |
| `--output-dir` | `str` | `data/processed/lodo_folds` | Directory to save CSV split files |
| `--target-min` | `int` | `1500` | Minimum samples per grade in quota balancing |
| `--target-max` | `int` | `2500` | Maximum samples per grade in quota balancing |
| `--zip-archive`| `flag` | `True` | Whether to compress generated CSVs to `lodo_folds.zip` |
| `--seed` | `int` | `42` | Random seed for deterministic patient splitting |

---

## 2. Model Training

### 2.1 `dr_multiscale_mil.train` (`scripts/03_train_in_domain.sh`)
Trains the Multi-Scale MIL network on the in-domain balanced training pool.

```bash
python -m dr_multiscale_mil.train \
    --config configs/train/in_domain.yaml \
    --checkpoint-dir checkpoints/checkpoints_main \
    --device cuda:0
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--config` | `str` | Required | Path to YAML experiment configuration file |
| `--checkpoint-dir`| `str` | `checkpoints/checkpoints_main` | Directory to save model checkpoints |
| `--device` | `str` | `cuda:0` | PyTorch device (`cuda:0`, `cpu`, `mps`) |
| `--resume` | `str` | `None` | Path to checkpoint to resume training from |

---

### 2.2 `dr_multiscale_mil.train_lodo` (`scripts/04_run_lodo_benchmark.sh`)
Trains and evaluates one or all 6 folds of the Leave-One-Dataset-Out (LODO) benchmark.

```bash
# Train single fold
python -m dr_multiscale_mil.train_lodo \
    --config configs/train/lodo_benchmark.yaml \
    --fold 0 \
    --checkpoint-dir checkpoints/checkpoints_lodo

# Train all 6 folds sequentially
python -m dr_multiscale_mil.train_lodo \
    --config configs/train/lodo_benchmark.yaml \
    --all-folds \
    --checkpoint-dir checkpoints/checkpoints_lodo
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--config` | `str` | Required | Path to LODO benchmark YAML config |
| `--fold` | `int` | `None` | Specific fold to train (0 to 5) |
| `--all-folds` | `flag` | `False` | Run all 6 folds sequentially |
| `--checkpoint-dir`| `str` | `checkpoints/checkpoints_lodo` | Directory to save fold checkpoints |

---

## 3. Evaluation & Publication Figures

### 3.1 `dr_multiscale_mil.evaluate`
Evaluates trained checkpoints on in-domain or OOD test sets and computes clinical metrics.

```bash
python -m dr_multiscale_mil.evaluate \
    --checkpoint checkpoints/checkpoints_main/best_model_qwk.pt \
    --test-csv data/raw/hybrid/test_raw_metadata.csv \
    --output-json results/tables/in_domain_eval.json
```

---

### 3.2 `dr_multiscale_mil.visualize` (`scripts/06_generate_figures.sh`)
Renders publication-ready figures (Figures 1 to 6) at 300 DPI and vector PDF.

```bash
python -m dr_multiscale_mil.visualize \
    --checkpoint checkpoints/checkpoints_main/best_model_qwk.pt \
    --output-dir results/figures \
    --figures all \
    --dpi 300
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `--checkpoint` | `str` | Required | Checkpoint file path |
| `--output-dir` | `str` | `results/figures` | Output directory for figures |
| `--figures` | `str` | `all` | Specific figure(s) to render (`all`, or `1,2,3,4,5,6`) |
| `--dpi` | `int` | `300` | Output raster resolution |
| `--format` | `str` | `png,pdf` | Comma-separated output formats |
