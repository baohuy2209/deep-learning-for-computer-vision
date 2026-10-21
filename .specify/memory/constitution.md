<!--
### Sync Impact Report
- Version change: 1.0.0 -> 1.1.0
- List of modified principles:
  - Principle II: Refined to harmonize explicit authorization with targeted task scopes
  - Added Principle V: Strict Preservation, Zero-Regression & Minimal-Scope (Delta-Only) Directives
- Added sections:
  - Principle V (Advisor Review & Zero-Regression Guardrails)
- Removed sections: None
- Follow-up TODOs: None. All core principles fully specified.
-->

# Deep Learning for Computer Vision Research Constitution

## Core Principles

### I. Research Context & Methodology
This project focuses on diagnosing Diabetic Retinopathy (DR) using advanced deep learning architectures.
- **Problem Type:** This is an **Ordinal Classification** problem (grading DR from 0 to 4), NOT a standard nominal multi-class problem. All model evaluations and loss formulations MUST optimize for **Quadratic Weighted Kappa (QWK)**, rather than standard classification accuracy alone.
- **Architecture:** Primary model architectural exploration MUST focus on a **Dual-branch framework** combining high-resolution local patches with global image context.
- **Advanced Data & Explainability Integrations:** Development MUST account for **Multimodal data fusion** (combining fundus images with clinical metadata such as HbA1c and diabetes duration) and **Explainable AI (XAI)** techniques (Grad-CAM for spatial localization, SHAP for tabular data).

### II. Strict File & Code Integrity (DO NOT MODIFY RULE)
- **NEVER overwrite, modify, or delete any existing files** unless explicitly commanded by the user with the exact authorization phrase or within an agreed, explicit scope of task execution.
- When suggesting experimental changes outside the primary pipeline, **always generate a NEW file** (e.g., `model_v2.py`, `notebook_experiment_1.ipynb`) or provide code in a markdown block for review.
- Treat all verified existing scripts and notebooks as stable reference materials.

### III. Strict Directory Path Adherence
All code and scripts MUST strictly adhere to the project directory structure below. AI assistants MUST NEVER invent, hallucinate, or use generic fallback paths (e.g., `/path/to/data` or `C:/User/...`). If an unlisted path is required, execution MUST stop and request the exact path from the user.

**Project Root:** `deep-learning-for-computer-vision/`
- `data/raw/`: Read-only access for raw datasets. Writing to this directory is strictly prohibited.
- `data/processed/`: Read and save all preprocessed datasets (e.g., CLAHE enhanced images, cropped patches).
- `data/models/`: Storage location for all trained weights, model checkpoints (`.keras`, `.h5`, `.pt`), and logs.
- `notebooks/diabetic_retinopathy/`: Storage location for exploratory data analysis (EDA), training experiments, and evaluation notebooks for the DR project.
- `src/`: Location for reusable Python scripts, custom loss functions (e.g., QWK loss), data loaders, and model definitions.
- `paper-reference/`: Reference directory for literature and academic papers provided by the user.

### IV. Code Generation & Reproducibility Rules
- **No Fallbacks / Placeholders:** Code generation MUST be complete and functional. Incomplete boilerplate containing `pass` or `TODO` for critical components (such as data loading or loss calculation) is strictly prohibited.
- **Reproducibility:** All scripts MUST explicitly set random seeds for NumPy, PyTorch / TensorFlow, and Python standard random modules at initialization to ensure reproducible research.
- **Environment & Dependency Integrity:** Missing required libraries (defined in `requirement.txt` or `requirement-win.txt`) MUST be flagged to the user for installation, rather than altering code to bypass missing dependencies.

### V. Strict Preservation, Zero-Regression & Minimal-Scope (Delta-Only) Directives
When updating or enhancing existing validated codebase components based on thesis advisor/professor feedback:
- **Strict Preservation Directive:** The existing verified pipeline and logic are stable. Under NO circumstances shall AI assistants refactor, rewrite, rename variables/functions, restructure architectures, or apply unsolicited "optimizations" outside the explicit scope of changes. Existing design patterns, execution flows, function signatures, and dependency chains MUST be preserved exactly as they are.
- **Minimal Scope of Modification (Delta-Only):** Modifications MUST be strictly confined to targeted functions or lines specified by the review notes. If an enhancement requires adding logic inside an existing function, all remaining lines and original behavior MUST remain untouched.
- **Pre-Modification Verification:** Every change must be validated against downstream dependencies to guarantee zero regression before finalizing edits.

## Governance

- **Supremacy & Compliance:** This Constitution defines non-negotiable rules for AI assistance and research code development. All generated code, pull requests, and experiment scripts MUST strictly comply with these principles.
- **Amendments:** Any modification to this Constitution requires explicit user approval and a semantic version bump (`MAJOR.MINOR.PATCH`).
- **Versioning Policy:**
  - **MAJOR**: Removals or redefinitions of core principles or file integrity constraints.
  - **MINOR**: Additions of new research methodologies, governance guardrails, or structural path rules.
  - **PATCH**: Wording improvements, clarifications, and non-semantic formatting updates.

**Version**: 1.1.0 | **Ratified**: 2026-08-09 | **Last Amended**: 2026-09-02
