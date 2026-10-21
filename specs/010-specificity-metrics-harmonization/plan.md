# Implementation Plan: Clinical Specificity-Sensitivity Harmonization & Colab Reference Code Sync

**Branch**: `010-specificity-metrics-harmonization` | **Date**: 2026-09-13 | **Spec**: [spec.md](file:///d:/Github/deep-learning-for-computer-vision/specs/010-specificity-metrics-harmonization/spec.md)

**Input**: Feature specification from `/specs/010-specificity-metrics-harmonization/spec.md`

---

## Summary

This plan synchronizes all repository training and evaluation notebooks (`main_model.ipynb`, `main_model (4).ipynb`, `main_model_head_ablation.ipynb`, `main_model_head_ablation (2).ipynb`) with the ground-truth Google Colab run files (`d:\Downloads\main_model (8).ipynb` and `main_model_head_ablation (3).ipynb`). It expands clinical evaluation routines across the entire specificity spectrum:
- Explicitly outputting and logging Referable DR sensitivity at Target Specificities **95%** ($\approx 58.7\%$), **90%** ($\approx 72.0\%$), **85%** ($\approx 79.6\%$, near 80% target), and **80%** ($\approx 84.7\%$, exceeding 80% target).
- Integrating full specificity columns (`Sens@Spec95`, `Sens@Spec90`, `Sens@Spec85`, `Sens@Spec80`) into head ablation summary tables (`df_heads`) and exported CSV files.
- Updating Referable DR ROC visualization (Fig. 7) with scatter markers and text annotations for Specificity 95%, 85%, and 80%.
- Unifying all head ablation notebooks under the complete 36-cell Colab implementation (with automatic checkpoint resumption, publication figures, and 5-head post-hoc calibration comparisons) in accordance with advisor directives in `HD_1.pdf`, `HD_2.pdf`, `HD_3.pdf`, and `HD_4.pdf`.

---

## Technical Context

**Language/Version**: Python 3.12, PyTorch $\ge 2.0.0$, `torchvision` $\ge 0.15.0$

**Primary Dependencies**: `torch`, `torch.nn`, `torchvision.models`, `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`

**Storage**: Local and cloud filesystem (`notebooks/diabetic_retinopathy/pipeline/model/`, `checkpoints_main/`, `checkpoints_head_ablation/`, `results/`)

**Testing**: Offline unit verification of calibration calculations and JSON notebook structural validation via Python scripts

**Target Platform**: Google Colab (Linux / NVIDIA A100 / V100 GPU) and Windows 11 development workstation

**Project Type**: Deep Learning Medical Imaging Research & Evaluation Pipeline (Jupyter Notebooks)

**Performance Goals**:
- Post-hoc threshold calibration completes in $<1$ second on validation predictions.
- Full parity between Colab and repository notebook code logic.

**Constraints**:
- Strict adherence to Constitution v1.1.0: Zero regression on established data loading, feature extractors, and loss functions.
- Minimal scope of modifications (Delta-Only): Confined to evaluation metrics calculation, logging, reporting, visualization, and cell synchronization.

**Scale/Scope**: 5 Jupyter notebooks covering single-model training and 5-head ablation benchmarks across multi-dataset fundus images.

---

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Principle I (Research Context & Methodology)**: PASS. Ordinal classification framework is maintained. Direct QWK evaluation and binormal post-hoc calibration ($d' = 1.864$) strictly address the referable screening problem.
- **Principle II (Strict File & Code Integrity)**: PASS. Changes are performed within the explicitly commanded scope to harmonize repository notebooks with Colab run files requested by the user.
- **Principle III (Strict Directory Path Adherence)**: PASS. All files reside under `notebooks/diabetic_retinopathy/pipeline/model/` and `specs/010-specificity-metrics-harmonization/`.
- **Principle IV (Code Generation & Reproducibility Rules)**: PASS. Fully functional Python code without placeholders; reproducible seeds; verification scripts provided.
- **Principle V (Strict Preservation & Delta-Only Directives)**: PASS. Backbones, feature dimensions, CBAM, MIL patch extraction, and QWK warmup hyperparameters ($0.0 \to 0.4$ over 15 epochs) are strictly preserved.

---

## Project Structure

### Documentation (this feature)

```text
specs/010-specificity-metrics-harmonization/
├── spec.md              # Feature specification
├── plan.md              # Implementation plan (this file)
├── research.md          # Methodological decisions and mathematical derivation
├── data-model.md        # Metric schemas and tabular record definitions
├── quickstart.md        # Verification scripts and run scenarios
├── contracts/
│   └── evaluation_contracts.md  # Python function signatures and contracts
└── checklists/
    └── requirements.md  # Quality checklist
```

### Source Code (repository root)

```text
notebooks/diabetic_retinopathy/pipeline/model/
├── main_model.ipynb                       # Proposed dual-branch model (harmonized with Colab run 8)
├── main_model (4).ipynb                   # Proposed model backup/variant (harmonized)
├── main_model_head_ablation.ipynb        # 5-head ablation benchmark (harmonized with Colab run 3, 36 cells)
├── main_model_head_ablation (2).ipynb    # Head ablation variant (harmonized)
└── main_model_head_ablation (3).ipynb    # Colab-verified reference notebook (36 cells)
```

**Structure Decision**: Direct delta-only updates to Jupyter notebook JSON cells to synchronize code logic, metric outputs, summary tables, and visualizations.

---

## Complexity Tracking

*No Constitution violations. Table empty.*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None | N/A | N/A |
