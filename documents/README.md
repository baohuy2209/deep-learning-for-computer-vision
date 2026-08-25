# Technical Documentation: Multi-Scale Ordinal Multiple Instance Learning Pipeline for Diabetic Retinopathy Grading

> **Project:** Deconstructing Domain Shift in Diabetic Retinopathy Grading: A Multi-Scale Multiple Instance Learning Approach with Ordinal Regression  
> **Source Code Directory:** [`notebooks/diabetic_retinopathy/pipeline/`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline)  
> **Documentation Language:** English (Academic & Technical Specification)  

---

## 📌 Executive Overview

This documentation repository provides a comprehensive and synchronized technical reference for the deep learning pipeline, experimental protocols, patient-level sterile benchmarks, and multi-center Leave-One-Dataset-Out (LODO) evaluations located in [`notebooks/diabetic_retinopathy/pipeline`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline).

The system is designed to resolve two fundamental bottlenecks in automated medical retinal screening:
1. **Imaging Domain Shift & Generalization Collapse:** Mitigating performance degradation when deploying AI across heterogeneous fundus cameras, illuminations, and patient populations.
2. **Resolution Collapse & Ordinal Severity Preservation:** Eliminating focal microaneurysm destruction caused by image downsampling and preserving strictly monotonic disease progression across the 5 international clinical grades (ICDRS 0–4).

---

## 🗂️ Documentation Architecture

The documentation suite in [`documents/`](file:///d:/Github/deep-learning-for-computer-vision/documents) is structured into four specialized monographs:

```plaintext
documents/
├── README.md                                                 # (This File) Master Documentation Hub & Repository Map
├── 01_datasets_analysis.md                                   # Comprehensive 6-Dataset Analysis & Bilateral Leakage Elimination
├── 02_pipeline_architecture.md                               # End-to-End Pipeline Architecture & Component Formulations
├── 03_paper_outline_and_methodology.md                       # Complete Research Paper Outline, Hypotheses & Experimental Design
└── 04_factorial_baseline_and_domain_shift_framework.md      # Factorial Baseline Matrix & Multi-Center LODO Evaluation Framework
```

---

## 🗺️ Source Code Directory Map

The engineering codebase is organized into four complementary subsystems:

```plaintext
notebooks/diabetic_retinopathy/pipeline/
├── eda/
│   └── eda_6_datasets.ipynb               # Exploratory Data Analysis across 6 multi-center cohorts (55,509 images)
├── merge-dataset/
│   └── merge-dataset.ipynb                # Harmonization schema, patient-ID parsing, quota balancing, ROI auto-cropping
├── full/                                  # Modular, Production-Grade Python Subsystem
│   ├── preprocessing.py                   # Auto-crop ROI, 3 preprocessing streams (Ben Graham, Green CLAHE, Minimal)
│   ├── data_loader.py                     # Data ingestion, GroupShuffleSplit, LODO generator, Dual-Branch PyTorch Dataset
│   ├── models.py                          # ResNet-50+CBAM (Global), EfficientNet-B0+MIL (Local), Multimodal Fusion MLP
│   ├── losses_and_heads.py                # 5 Output Heads (Softmax CE/QWK, CORAL, CORN, CLM_QWK), Continuous QWK Loss
│   ├── trainer.py                         # DRTrainer (Mixed Precision AMP, Callbacks), FastHeadTrainer, Clinical Metrics
│   └── baselines.py                       # 5 Baseline model classes (ResNet50, EffNetB0, DenseNet121, MSAG, ViT)
└── model/                                 # Self-Contained Experimental Jupyter Notebooks
    ├── main_model.ipynb                   # Proposed Full Architecture (Dual-Branch MIL + CLM_QWK) & 4-Panel Dashboard
    ├── main_model_ablation.ipynb          # Fast-Probing Ablation on 5 Heads (Cached Latents) & 5 Publication Figures
    ├── main_model_lodo.ipynb              # 6-Fold Leave-One-Dataset-Out (LODO) Generalization Benchmark Suite
    ├── resnet50_baseline.ipynb            # Baseline 1: Vanilla ResNet-50 (Global Stream Anchor)
    ├── efficientnetb0_baseline.ipynb      # Baseline 2: Vanilla EfficientNet-B0 (Local Stream Anchor)
    ├── densenet121_baseline.ipynb         # Baseline 3: DenseNet-121 (Dense Feature Reuse Baseline)
    ├── efficientnetb0_msag.ipynb          # Baseline 4: EfficientNet-B0 + Multi-Scale Spatial Attention Gating (MSAG)
    └── efficientnetb0_vit.ipynb           # Baseline 5: Hybrid EfficientNetB0-Vision Transformer (ViT)
```

---

## 📑 Executive Summaries of Monograph Documents

### 1. [01. Comprehensive Dataset Analysis & Patient-Level Leakage Audit](file:///d:/Github/deep-learning-for-computer-vision/documents/01_datasets_analysis.md)
* **Multi-Center Cohort Ingestion (55,509 Images):** Harmonization of APTOS 2019, EyePACS, Messidor-2, IDRiD, DeepDRiD, and DDR.
* **Elimination of Bilateral Data Leakage:** Mathematical proof and implementation of `GroupShuffleSplit` on `patient_id` ensuring zero cross-eye contamination between Train, Validation, and Test sets.
* **Multi-Dimensional Domain Shift Analysis:** Quantifying Label Shift ($6.77\text{x}$ to $36.00\text{x}$ class imbalance), Covariate Shift ($512\times 512$ to $4288\times 2848$), and Photometric / Illumination Shift ($\mu_B$ channel drift).

### 2. [02. End-to-End Pipeline Architecture & Mathematical Formulations](file:///d:/Github/deep-learning-for-computer-vision/documents/02_pipeline_architecture.md)
* **Stage 1 (Data & Multi-Source Quota Allocation):** Two-tier quota balancing algorithm preventing large datasets (EyePACS) from overwhelming specialized clinical cohorts (IDRiD, Messidor-2).
* **Stage 2 (Preprocessing Ablation):** Selective Green-channel CLAHE contrast enhancement without Gaussian smoothing to protect focal microaneurysms.
* **Stage 3 (Dual-Branch Multi-Scale MIL Backbone):**
  * *Global Context Stream ($300 \times 300$):* ResNet-50 + CBAM Attention capturing 4-quadrant vascular geometry for **Grade 3**.
  * *Local MIL Stream ($K=16, 128 \times 128$):* EfficientNet-B0 scanning native-resolution patches with optic disc masking for **Grade 1**.
  * *Multimodal Fusion:* $3328\text{-d} \to \text{Mish MLP} \to 512\text{-d}$ Latent Space.
* **Stage 4 (Ordinal Classification & Feature Cache):** Mathematical formulation of the **Champion CLM_QWK Head** (clog-log link, strictly monotonic cutpoints, Continuous Differentiable QWK Loss).
* **Stage 5 (Quantitative XAI):** Grad-CAM saliency verification against IDRiD ophthalmologist ground-truth masks (IoU and Pointing Game).

### 3. [03. Research Paper Outline, Hypotheses & Methodology](file:///d:/Github/deep-learning-for-computer-vision/documents/03_paper_outline_and_methodology.md)
* **Standard 6-Chapter Paper Structure:** Introduction, Related Work, Methodology, Experimental Setup, Results & Analysis (verifying Hypotheses $H_1$ to $H_5$), Discussion & Conclusion.
* **Master Comparative Benchmarking:** Comprehensive comparative tables pitting the proposed model against all 5 baseline architectures across in-domain and out-of-domain metrics.

### 4. [04. Factorial Baseline Matrix & Deconstructive Domain Shift Framework](file:///d:/Github/deep-learning-for-computer-vision/documents/04_factorial_baseline_and_domain_shift_framework.md)
* **Methodological Critique of Prior Art:** Analysis of confounding variables, in-domain over-optimization traps, and the clinical fallacy of binary classification.
* **The 3-Axis Factorial Baseline Matrix:** Factorizing models along (1) Spatial Representation, (2) Output Loss Formulations, and (3) Preprocessing Pipelines.
* **Empirical Verification of Core Hypotheses:** Quantified results for $H_1$ (Mild DR pixel-loss recovery), $H_2$ (Severe DR 4-2-1 context aggregation), and $H_3$ (Softmax boundary collapse elimination).

---

## 🔗 Quick Access Codebase Index

| Pipeline Component | Source File Link | Functionality & Key Implementation |
| :--- | :--- | :--- |
| **Data Ingestion & LODO** | [`pipeline/full/data_loader.py`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/full/data_loader.py) | Ingests 6 cohorts, GroupShuffleSplit by patient, LODO generator, DualBranch Dataset |
| **Image Preprocessing** | [`pipeline/full/preprocessing.py`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/full/preprocessing.py) | Auto-crop ROI, 3 preprocessing streams, native patch sliding window generator |
| **Backbone Models** | [`pipeline/full/models.py`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/full/models.py) | CBAM Attention, GlobalBranch, LocalMILBranch with 1D Max-Pooling, Fusion MLP |
| **Losses & Heads** | [`pipeline/full/losses_and_heads.py`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/full/losses_and_heads.py) | Softmax CE/QWK, CORAL, CORN, Cumulative Link Model (CLM), Continuous QWK Loss |
| **Baseline Models** | [`pipeline/full/baselines.py`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/full/baselines.py) | Standardized implementations of ResNet-50, EffNet-B0, DenseNet-121, MSAG, ViT |
| **Training Engine** | [`pipeline/full/trainer.py`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/full/trainer.py) | Mixed-precision AMP DRTrainer, FastHeadTrainer, metric suite ($\Delta_{\text{QWK}}$, RMV, XAI) |
| **Proposed Full Notebook** | [`pipeline/model/main_model.ipynb`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/model/main_model.ipynb) | End-to-end training of proposed Dual-Branch CLM model & 4-panel clinical dashboard |
| **Head Ablation Notebook** | [`pipeline/model/main_model_ablation.ipynb`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/model/main_model_ablation.ipynb) | Fast latent probing of 5 output heads & auto-generation of Figures 1, 3, 4, 5 |
| **6-Fold LODO Notebook** | [`pipeline/model/main_model_lodo.ipynb`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb) | Full execution of 6-Fold Leave-One-Dataset-Out cross-domain evaluation suite |
| **ResNet-50 Baseline** | [`pipeline/model/resnet50_baseline.ipynb`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/model/resnet50_baseline.ipynb) | Global stream anchor baseline (ResNet-50 GAP) verifying Hypothesis $H_2$ |
| **EffNet-B0 Baseline** | [`pipeline/model/efficientnetb0_baseline.ipynb`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/model/efficientnetb0_baseline.ipynb) | Local stream anchor baseline on $300\times 300$ input verifying Hypothesis $H_1$ |
| **DenseNet-121 Baseline** | [`pipeline/model/densenet121_baseline.ipynb`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/model/densenet121_baseline.ipynb) | Dense connectivity baseline evaluating feature accumulation under nominal CE |
| **EffNet-B0 + MSAG** | [`pipeline/model/efficientnetb0_msag.ipynb`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/model/efficientnetb0_msag.ipynb) | Multi-scale feature attention baseline proving the downsampling pixel-loss flaw |
| **Hybrid EffNet-ViT** | [`pipeline/model/efficientnetb0_vit.ipynb`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/model/efficientnetb0_vit.ipynb) | Hybrid CNN-Transformer baseline assessing domain overfitting and convergence |
| **EDA Notebook** | [`pipeline/eda/eda_6_datasets.ipynb`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/eda/eda_6_datasets.ipynb) | 55,509-image distribution, resolution, color channels, and Laplacian sharpness |
| **Dataset Merging** | [`pipeline/merge-dataset/merge-dataset.ipynb`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/merge-dataset/merge-dataset.ipynb) | Patient ID parsing, multi-source quota allocation, metadata generation |
