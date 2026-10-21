# Implementation Plan: Robust Image Preprocessing & Harmonization Pipeline for 6-Dataset DR Hybrid Pool

**Branch**: `002-dr-preprocessing-harmonization` | **Date**: 2026-08-09 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/002-dr-preprocessing-harmonization/spec.md`

## Summary

Build a robust, leak-free image preprocessing, harmonization, and class balancing pipeline for a 6-dataset Diabetic Retinopathy hybrid pool (53,977 images). The pipeline performs schema harmonization, 70/15/15 stratified splitting before augmentation, ROI auto-cropping, Green channel CLAHE contrast enhancement, Gaussian blur noise reduction, Albumentations minority augmentation (strictly on Train set to reach 2,000 images/class), pixel normalization, and dual-resolution image export (`300x300` & `512x512`).

## Technical Context

**Language/Version**: Python 3.10+

**Primary Dependencies**: `opencv-python` (`cv2`), `albumentations`, `scikit-learn`, `pandas`, `numpy`, `pillow`, `matplotlib`, `tqdm`

**Storage**: Input raw datasets in `data/raw/`, preprocessed outputs saved in `data/processed/hybrid/300x300/` and `data/processed/hybrid/512x512/` (and reference `data/raw/hybrid/`).

**Testing**: Inline verification assertions and quickstart execution checks

**Target Platform**: Windows / Linux (Python 3.10+ GPU/CPU execution environment)

**Project Type**: Data Science / Deep Learning Computer Vision Preprocessing Pipeline

**Performance Goals**: Process 53,977 images with ROI auto-cropping and dual-resolution saving in efficient batch iterations.

**Constraints**: Strict 70/15/15 stratified split before augmentation to prevent data leakage, 2,000 target images per class in Train set (10,000 total), unaugmented Val/Test sets.

**Scale/Scope**: 6 datasets, 53,977 input images, 10,000 balanced training images, dual spatial resolutions (`300x300` and `512x512`).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Rule 1 (Research Context & Methodology)**: PASS — Preprocessing and leak-free stratified balancing directly serve dual-branch architecture and QWK loss optimization.
- **Rule 2 (Strict File Integrity)**: PASS — All preprocessed outputs will be created as new files under `data/processed/hybrid/` and `notebooks/diabetic_retinopathy/pipeline/preprocessing/preprocessing.ipynb` without overwriting raw source data.
- **Rule 3 (Strict Directory Path Adherence)**: PASS — Uses exact project structure paths (`data/raw/`, `data/processed/`, `notebooks/diabetic_retinopathy/`).
- **Rule 4 (Code Generation Rules)**: PASS — Functional code with complete logic, random seed reproducibility (`SEED=42`), and explicit dependency handling.

## Project Structure

### Documentation (this feature)

```text
specs/002-dr-preprocessing-harmonization/
├── plan.md              # Implementation plan
├── research.md          # Technical research & decisions (Phase 0)
├── data-model.md        # Data model & entity schema (Phase 1)
├── quickstart.md        # End-to-end validation guide (Phase 1)
├── contracts/           # Pipeline function contracts (Phase 1)
│   └── preprocessing_pipeline_contract.md
├── checklists/
│   └── requirements.md  # Quality checklist
└── spec.md              # Feature specification
```

### Source Code (repository root)

```text
notebooks/
└── diabetic_retinopathy/
    └── pipeline/
        └── preprocessing/
            ├── preprocessing.ipynb   # Executable Jupyter Notebook
            └── preprocessing_script.py # VS Code `# %%` executable script
```

**Structure Decision**: Preprocessing pipeline notebook created at `notebooks/diabetic_retinopathy/pipeline/preprocessing/preprocessing.ipynb` as requested.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
| :--- | :--- | :--- |
| Dual resolution export | Supports both rapid baseline context branch (300x300) and high-resolution local patch branch (512x512) | Single resolution would restrict dual-branch architectural experiments |
