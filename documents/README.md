# Technical Documentation: Multi-Scale Ordinal Multiple Instance Learning Pipeline for Diabetic Retinopathy Grading

> **Project:** Deconstructing Domain Shift in Diabetic Retinopathy Grading: A Multi-Scale Multiple Instance Learning Approach with Ordinal Regression  
> **Source Code Directory:** [`notebooks/diabetic_retinopathy/pipeline/`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline)  
> **Documentation Language:** English (Academic & Technical Specification)  

---

## Executive Overview

This documentation repository provides a synchronized technical reference for the deep learning pipeline, experimental protocols, patient-level sterile benchmarks, and multi-center Leave-One-Dataset-Out (LODO) evaluations located in [`notebooks/diabetic_retinopathy/pipeline`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline).

The system is designed to resolve two fundamental bottlenecks in automated medical retinal screening:
1. **Imaging Domain Shift & Generalization Collapse:** Mitigating performance degradation when deploying AI across heterogeneous fundus cameras, illuminations, and patient populations.
2. **Resolution Collapse & Ordinal Severity Preservation:** Eliminating focal microaneurysm destruction caused by image downsampling and preserving strictly monotonic disease progression across the 5 international clinical grades (ICDRS 0–4).

---

## Documentation Architecture

The documentation suite in [`documents/`](file:///d:/Github/deep-learning-for-computer-vision/documents) is structured into specialized monographs:

```plaintext
documents/
├── README.md                                                 # (This File) Master Documentation Hub & Repository Map
├── 01_datasets_analysis.md                                   # Comprehensive 6-Dataset Analysis & Bilateral Leakage Elimination
├── 02_pipeline_architecture.md                               # End-to-End Pipeline Architecture & Component Formulations
├── 03_paper_outline_and_methodology.md                       # Complete Research Paper Outline, Hypotheses & Experimental Design
├── 04_factorial_baseline_and_domain_shift_framework.md      # Factorial Baseline Matrix & Multi-Center LODO Evaluation Framework
├── 05_manuscript_draft_sections.md                           # Research Manuscript Text: Related Work, Method, and Experimental Setup
├── 06_citation_audit_and_literature_mapping.md              # Citation Database Audit, 5 Ordinal Groups & Literature Matrix
└── 07_technical_report_and_experimental_appendix.md         # Comprehensive Technical Report & Experimental Appendix (Code-Synchronized)
```

---

## Source Code Directory Map

The engineering codebase is organized into four complementary subsystems:

```plaintext
notebooks/diabetic_retinopathy/pipeline/
├── eda/
│   └── eda_6_datasets.ipynb               # Exploratory Data Analysis across 6 multi-center cohorts (55,570 images)
├── merge-dataset/
│   └── merge-dataset.ipynb                # Harmonization schema, patient-ID parsing, quota balancing, 4-step FOV auto-cropping
├── full/                                  # Modular, Production-Grade Python Subsystem
│   ├── preprocessing.py                   # 4-step FOV mask, 3 preprocessing streams (Ben Graham, Green CLAHE, Minimal)
│   ├── data_loader.py                     # Data ingestion, GroupShuffleSplit, LODO generator, Dual-Branch PyTorch Dataset
│   ├── models.py                          # ResNet-50+NonLocal+CBAM+QuadrantTokens (4096-d), EffNet-B0 Dual Pool (2560-d), Fusion MLP (6656->512)
│   ├── losses_and_heads.py                # 5 Output Heads (Softmax CE/QWK, CORAL, CORN, CLM_QWK), Continuous QWK Loss
│   ├── trainer.py                         # DRTrainer (AMP, QWK Warmup), FastHeadTrainer, Clinical Metrics, FundusGradCAM
│   └── visualization.py                   # Unified Publication Visualization Module (Figs 1 to 8, 300 DPI)
└── model/                                 # Self-Contained Experimental Jupyter Notebooks
    ├── main_model (6).ipynb               # Proposed Full Architecture Champion (Dual-Branch MIL + CLM_QWK + Synchronized DRTrainer)
    ├── main_model_head_ablation (3).ipynb # Champion Head Ablation Benchmark (5-Round Synchronized DRHeadTrainer)
    ├── main_model_lodo.ipynb              # 6-Fold Leave-One-Dataset-Out (LODO) Generalization Benchmark Suite
    ├── main_model.ipynb                   # Full Architecture Prototype & Qualitative XAI
    ├── main_model (4).ipynb               # Full Architecture with HD_4 Dual-Crop XAI & Threshold Calibration
    ├── main_model_head_ablation.ipynb     # 5-Round End-to-End Head Ablation Benchmark & Monotonicity Audit
    ├── main_model_head_ablation (2).ipynb # Synchronized Head Ablation Suite with Post-Hoc Screening Calibration
    └── main_model_preprocessing_ablation.ipynb # Preprocessing Stream Comparison (Baseline vs. Ben Graham vs. Green CLAHE)
```

---

## Executive Summaries of Monograph Documents

### 1. [01. Comprehensive Dataset Analysis & Patient-Level Leakage Audit](file:///d:/Github/deep-learning-for-computer-vision/documents/01_datasets_analysis.md)
* **Multi-Center Cohort Ingestion (55,570 Images):** Harmonization of APTOS 2019, EyePACS, Messidor-2, IDRiD, DeepDRiD, and DDR.
* **Elimination of Bilateral Data Leakage:** Implementation of `GroupShuffleSplit` on `patient_id` with dataset namespace prefixes (`dataset_key::patient_id`) ensuring zero cross-eye contamination.
* **2-Tier Data Partitioning:** Splitting the raw pool into 70% Train, 15% Validation, and 15% Test before applying multi-source quota balancing to the training pool.

### 2. [02. End-to-End Pipeline Architecture & Mathematical Formulations](file:///d:/Github/deep-learning-for-computer-vision/documents/02_pipeline_architecture.md)
* **Stage 1 (Data, Sterile Split & Caching Engine):** Patient-level grouping, multi-source quota allocation, low-overhead `uint8` IPC transmission (reducing memory bandwidth from $755\text{ MB}$ to $189\text{ MB/batch}$), and **Offline 2-Tier NPZ Coordinate Pre-Caching** eliminating $158\text{ CPU hours}$ of redundant bilateral filtering, Gaussian subtraction, and morphological FOV grid searching across 100 epochs.
* **Stage 2 (4-Step FOV & Preprocessing Ablation):** Otsu's adaptive thresholding, morphological Close+Open (ellipse 25x25), largest component filtering, and $15\text{px}$ boundary erosion; selective Green-channel CLAHE contrast enhancement without terminal Gaussian smoothing.
* **Stage 3 (Dual-Branch Multi-Scale MIL Backbone):**
  * *Global Context Stream ($512 \times 512$):* ResNet-50 + `NonLocalBlock2D` + `CBAM` Attention + `QuadrantTokenAggregator` (4-quadrant Transformer relational encoding) capturing vascular geometry and verifying the clinical 4-2-1 rule for **Grade 3 ($H_2$)** $\to F_{\text{Global}} \in \mathbb{R}^{4096}$.
  * *Local Sliding-Window MIL Stream (v2):* EfficientNet-B0 scanning native-resolution overlapping $128 \times 128$ patches ($s=96$, FOV filter $\ge 0.5$, dynamic $M \in [32, 64]$ / eval dense sweep) with Dual Pooling (Max-Pooling + Masked Gated Attention) $\to F_{\text{Local}} \in \mathbb{R}^{2560}$ for **Grade 1 ($H_1$)**.
  * *Multimodal Fusion:* $6656\text{-d} \to \text{LayerNorm MLP} \to 512\text{-d}$ Latent Space $\mathbf{z}$.
* **Stage 4 (Ordinal Classification & 5 Heads):** Champion CLM_QWK Head ($\text{clog-log}$ link, strictly monotonic cutpoints, Continuous Differentiable QWK Loss with linear warmup). Fully synchronized across 3 head ablation benchmark notebooks (`main_model_head_ablation.ipynb`, `(2)`, `(3)`) with post-hoc threshold calibration ($95\%, 90\%, 85\%, 80\%$ specificity).
* **Stage 5 (Explainable AI & Saliency):** FundusGradCAM with memory-safe hook cleanup and quantitative evaluation against IDRiD pixel lesion masks. Overhauled with Dual-Crop registration, empirical random baseline ($1.5\%$), Area-Matched Saliency Recall, and native Attention-MIL saliency ($18.52\%$ Pointing Game, $12.3\times$ chance).

### 3. [03. Research Paper Outline, Hypotheses & Methodology](file:///d:/Github/deep-learning-for-computer-vision/documents/03_paper_outline_and_methodology.md)
* **Standard 6-Chapter Paper Structure:** Introduction, Related Work, Methodology, Experimental Setup, Results & Analysis (verifying Hypotheses $H_1$ to $H_5$), Discussion & Conclusion.
* **Complete Figure Suite (Figs 1 to 8):** Slope charts, waterfall decompositions, LODO heatmaps, and latent score distributions.

### 4. [04. Factorial Baseline Matrix & Deconstructive Domain Shift Framework](file:///d:/Github/deep-learning-for-computer-vision/documents/04_factorial_baseline_and_domain_shift_framework.md)
* **Factorial Matrix ($3 \times 3 \times 5$):** Spatial representations $\times$ Preprocessing pipelines $\times$ Output head formulations.
* **5-Round End-to-End Head Ablation:** Controlled comparison on ResNet-50 backbones.
* **Rank Monotonicity Metric:** Auditing cumulative exceedance probabilities $P(y > k)$ before non-negative projections.

### 5. [06. Citation Audit & Literature Mapping](file:///d:/Github/deep-learning-for-computer-vision/documents/06_citation_audit_and_literature_mapping.md)
* **Citation Database Audit:** Verification of literature references across 5 ordinal modeling taxonomy groups.
* **Domain Adaptation Matrix:** Prior work comparison and theoretical placement.

### 6. [07. Technical Report & Experimental Appendix](file:///d:/Github/deep-learning-for-computer-vision/documents/07_technical_report_and_experimental_appendix.md)
* **Code-Synchronized Architecture Specifications:** Dual-Branch NonLocal+CBAM+QuadrantTokens, EfficientNet-B0 Dual Pooling MIL, and LayerNorm MLP Fusion.
* **Mathematical Formulations:** Monotonic cutpoint Softplus parameterization, clog-log link function, and Continuous QWK loss.
* **Training Dynamics & Hyperparameter Atlas:** Optimization parameters, dynamic warm-up scheduling, and decoupled dual monitoring.
* **Preprocessing & 2-Tier Caching Engine:** 4-step raw RGB FOV segmentation, .npz coordinate caching, and zero data-leakage LODO protocol.

---

---

## Advisor Review Sync & Methodological Directives (HD_1, HD_2, HD_3, HD_4)

The codebase and technical monographs strictly incorporate all directives from the advisor review series:
1. **HD_1 (Thesis Framing & Mandatory Metrics):** Framing around Domain Shift survival ($H_1, H_2, H_3$), 5-Head ablation suite, 6-Fold LODO protocol, and mandatory reporting of per-grade recall on Grade 1/Grade 3, within-1 accuracy, and rank monotonicity violations.
2. **HD_2 (Methodological Bugs & Structural Refinements):** Resolving patient ID parsing collisions (`f"{dataset_key}::{pid}"`), enforcing sterile split before balancing (preserving natural prevalence on Val/Test), replacing self-comparison XAI with real 81 IDRiD masks, auditing monotonicity on raw pre-activation rank logits, and upgrading Global/Local branches with NonLocal+QuadrantTokens and Dual-Pooling.
3. **HD_3 (Root Cause of Patch Defect & I/O Speedups):** Computing FOV mask on raw RGB before Ben Graham (preventing Otsu background gray 128 collapse), setting sliding stride to $s=96$ (25% overlap), switching IPC to `uint8` tensors (cutting memory bandwidth $4	imes$), disabling OpenCV thread thrashing (`cv2.setNumThreads(0)`), and caching preprocessed anchors.
4. **HD_4 (Post-Hoc Screening, XAI Overhaul & Loss Stabilization):**
   * *Part 1 & 3:* Decoupling referable DR ($y \ge 2$) from nominal `argmax`, establishing Ref-AUC $= 0.9063$, calibrating thresholds post-hoc on Validation to hit target specificities ($95\%, 90\%, 85\%, 80\%$), and benchmarking against UK NSC criteria ($	ext{Sensitivity} \ge 80\%$).
   * *Part 2 & 6:* Dual-Crop FoV bounding box alignment on IDRiD, reporting Pointing Game with random baseline ($11.11\%$ vs $1.5\%$), implementing Area-Matched Saliency Recall and Pixel-level AUROC, and documenting the $pprox 0.050$ theoretical IoU ceiling.
   * *Part 4:* Extending QWK loss warmup from 5 to 15 epochs, scaling target $\lambda_{	ext{QWK}} = 0.4$, and aligning EarlyStopping and ModelCheckpoint on `val_QWK` with patience 15-20.
   * *Part 5:* Native Attention-MIL Saliency Map Engine generating fine-grained $pprox 128	ext{ px}$ maps to provide visual and quantitative proof for Hypothesis $H_1$.

## Quick Access Codebase Index

| Pipeline Component | Source File Link | Functionality & Key Implementation |
| :--- | :--- | :--- |
| **Data Ingestion & LODO** | [`pipeline/full/data_loader.py`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/full/data_loader.py) | Ingests 6 cohorts, GroupShuffleSplit by patient, LODO generator, DualBranch Dataset |
| **Image Preprocessing** | [`pipeline/full/preprocessing.py`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/full/preprocessing.py) | 4-step FOV segmentation (`compute_fov_mask`), 3 preprocessing streams, sliding-window patch extractor |
| **Backbone Models** | [`pipeline/full/models.py`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/full/models.py) | Non-Local Block, CBAM Attention, QuadrantTokenAggregator, GlobalBranch (4096-d), LocalMILBranch (2560-d), Fusion MLP (6656 -> 512-d) |
| **Losses & Heads** | [`pipeline/full/losses_and_heads.py`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/full/losses_and_heads.py) | Softmax CE/QWK, CORAL, CORN, Cumulative Link Model (CLM), Continuous QWK Loss |
| **Trainer & Evaluation** | [`pipeline/full/trainer.py`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/full/trainer.py) | DRTrainer (AMP, QWK Warmup), FundusGradCAM with hook cleanup, Clinical Metrics |
| **Visualization Suite** | [`pipeline/full/visualization.py`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/full/visualization.py) | Publication-ready renderers for Figs 1 to 8 (300 DPI, academic aesthetics) |
