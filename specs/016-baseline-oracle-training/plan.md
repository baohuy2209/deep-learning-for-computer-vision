# Implementation Plan: Baseline Oracle Training for 6 Fundus Cohorts

**Branch**: `016-baseline-oracle-training` | **Date**: 2026-09-22 | **Spec**: [spec.md](file:///d:/Github/deep-learning-for-computer-vision/specs/016-baseline-oracle-training/spec.md)

**Input**: Feature specification from `specs/016-baseline-oracle-training/spec.md`

---

## Summary

This feature implements 6 standalone in-domain training pipelines (Jupyter Notebooks) for the Champion Architecture (`main_model`) across 6 fundus photography cohorts (APTOS 2019, DDR, DeepDRiD, IDRiD, Messidor-2, EyePACS). By establishing in-domain upper bounds ($QWK_{\text{oracle}}$) on the identical 15% held-out test split as the cross-domain LODO benchmark ($QWK_{\text{OOD}}$), this implementation eliminates class marginal prior confounding and computes the authentic domain-shift penalty $\Delta_{\text{transfer}} = QWK_{\text{oracle}} - QWK_{\text{OOD}}$, directly resolving Section A1 of the academic review in `documents/R_v0.pdf`.

---

## Technical Context

**Language/Version**: Python 3.10+ / 3.11  
**Primary Dependencies**: PyTorch 2.0+, Torchvision, Albumentations, OpenCV, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn  
**Storage**: Checkpoints saved to `/content/drive/MyDrive/checkpoints_oracle/` (`.pth` weights + optimizer + scaler state)  
**Testing**: Offline validation script verifying zero patient leakage (`GroupShuffleSplit`), batch output shapes, loss convergence, and metric calculation  
**Target Platform**: Google Colab (NVIDIA A100 / V100 / T4 GPU) with Automatic Mixed Precision (`torch.amp`)  
**Project Type**: Deep Learning Medical Imaging Research Pipelines (Jupyter Notebooks)  
**Performance Goals**: Model training completes within 40 epochs; inference latency $<15\text{ms}$ per retinal image  
**Constraints**: Zero bilateral patient-level data contamination; exact reproduction of Champion Architecture ($V_3$)  
**Scale/Scope**: 6 fundus cohorts comprising 55,570 total standardized retinal images across 35,474 patients  

---

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Compliance Status | Analysis & Verification |
| :--- | :---: | :--- |
| **I. Research Context & Methodology** | **PASS** | Formulated as Ordinal Classification optimizing Quadratic Weighted Kappa (QWK) via Cumulative Link Model (CLM) + continuous QWK loss; implements Dual-Branch framework (512x512 global + 128x128 local MIL). |
| **II. Strict File Integrity** | **PASS** | Creates 6 new dedicated standalone notebooks in `notebooks/diabetic_retinopathy/pipeline/oracle/`. ZERO existing files overwritten or deleted. |
| **III. Directory Path Adherence** | **PASS** | Adheres strictly to canonical paths: `notebooks/diabetic_retinopathy/pipeline/oracle/`, `data/raw/`, `data/processed/`, `/content/drive/MyDrive/checkpoints_oracle/`. |
| **IV. Reproducibility Rules** | **PASS** | Seeds fixed explicitly across NumPy, PyTorch, and Python random (`seed=42`); no placeholder boilerplate or mock data. |
| **V. Zero-Regression & Minimal Scope** | **PASS** | Leaves existing LODO and core model notebooks intact; adds standalone Oracle experiments. |

---

## Project Structure

### Documentation (this feature)

```text
specs/016-baseline-oracle-training/
├── spec.md              # Feature specification with 5 clarified decisions
├── plan.md              # This file (implementation plan)
├── research.md          # Academic literature & architectural decisions
├── data-model.md        # Entities, schemas & lifecycle
├── quickstart.md        # Google Colab execution guide
├── contracts/
│   └── oracle_evaluation_schema.json  # Output JSON contract for Table 3
└── checklists/
    └── requirements.md  # Quality verification checklist
```

### Source Code (repository root)

```text
notebooks/diabetic_retinopathy/pipeline/oracle/
├── oracle_aptos.ipynb         # APTOS 2019 Oracle training (3,662 images)
├── oracle_ddr.ipynb           # DDR Oracle training (12,522 images)
├── oracle_deepdrid.ipynb      # DeepDRiD Oracle training (2,000 images)
├── oracle_idrid.ipynb         # IDRiD Oracle training (516 images)
├── oracle_messidor2.ipynb     # Messidor-2 Oracle training (1,744 images)
└── oracle_eyepacs.ipynb       # EyePACS Oracle training (35,126 images)
```

**Structure Decision**: 6 standalone Jupyter Notebooks inside `notebooks/diabetic_retinopathy/pipeline/oracle/` ensuring each cohort can be launched independently in parallel Google Colab sessions without cross-session coupling or interference.

---

## Complexity Tracking

> No constitution violations detected. Design is fully compliant.
