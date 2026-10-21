# Research & Methodological Decisions: Clinical Specificity Harmonization & Colab Sync

**Feature**: Clinical Specificity-Sensitivity Harmonization & Colab Reference Code Sync  
**Branch**: `010-specificity-metrics-harmonization`  
**Date**: 2026-09-13  
**Spec**: [spec.md](file:///d:/Github/deep-learning-for-computer-vision/specs/010-specificity-metrics-harmonization/spec.md)

---

## 1. Research Questions & Methodological Foundations

### R1: Why did the model report sub-80% screening sensitivity at default classification, and why does threshold calibration resolve this without retraining?
- **Finding (HD_4 Part 1 & 3)**: The nominal 5-class classification uses `argmax(probs)` to assign discrete grades $0 \dots 4$. In medical screening for Referable DR (defined as Grade $\ge 2$), predicting referable status via `argmax >= 2` corresponds to an operating threshold on the cumulative posterior $P(y \ge 2)$ that happens to sit at approximately $95\%$ specificity on the ROC curve.
- **Decision**: Decouple the nominal 5-class grade prediction from the binary screening decision boundary. Perform post-hoc threshold calibration on the continuous predicted risk score $P(y \ge 2) = \sum_{k=2}^4 P(y=k)$ fitted on the Validation set.
- **Rationale**: The model achieves $\text{Test Ref-AUC} \approx 0.9063$, proving strong underlying discriminative separation. In a binormal detection theoretic model with discrimination index $d' = \sqrt{2} \cdot \Phi^{-1}(\text{AUC}) = 1.864$:
  $$\text{TPR} = \Phi\left(d' + \Phi^{-1}(\text{FPR})\right)$$
  - For $\text{FPR} = 0.05$ ($\text{Spec} = 95\%$): $\text{TPR} = \Phi(1.864 - 1.645) = \Phi(0.219) \approx 58.7\%$ (matches empirical argmax observation).
  - For $\text{FPR} = 0.15$ ($\text{Spec} = 85\%$): $\text{TPR} = \Phi(1.864 - 1.036) = \Phi(0.828) \approx 79.6\%$ (sát mục tiêu $80\%$).
  - For $\text{FPR} = 0.20$ ($\text{Spec} = 80\%$): $\text{TPR} = \Phi(1.864 - 0.842) = \Phi(1.022) \approx 84.7\%$ (vượt mục tiêu $80\%$).
- **Alternatives Considered**:
  - *Class re-weighting during training*: Rejected. As proven in `HD_4 Part 1`, Grade 1 (2,336) and Grade 2 (2,288) are the two largest classes in the dataset. Artificially weighting Grade 2+ distorts the well-calibrated posterior probabilities without improving representation quality, and requires expensive retraining.

---

### R2: What is the complete operating specificity set that must be reported across all notebooks?
- **Decision**: Standardize `target_specificities = (0.80, 0.85, 0.90, 0.95)` across all evaluation functions (`calculate_referable_sensitivity_at_specificity`, `compute_clinical_metrics`, `calibrate_and_evaluate_screening`), with optional reporting of `0.75` where full curves are tabulated.
- **Rationale**:
  - $95\%$ represents the stringent UK NSC specificity threshold.
  - $85\%$ demonstrates the transition point where sensitivity approaches the clinical $80\%$ threshold ($79.6\%$).
  - $80\%$ demonstrates the operating point where sensitivity cleanly surpasses the clinical $80\%$ threshold ($84.7\%$).
  - $90\%$ provides an intermediate clinical benchmark ($72.0\%$).
- **Alternatives Considered**:
  - *Reporting only 95% and 80%*: Rejected. Omitting 85% loses the critical milestone highlighted in `HD_4.pdf`.
  - *Reporting only argmax*: Strictly forbidden by `HD_4` as scientifically invalid for binary screening evaluation.

---

### R3: How to harmonize `main_model.ipynb` and `main_model_head_ablation.ipynb` with their Colab-downloaded counterparts?
- **Finding**:
  - `d:\Downloads\main_model (8).ipynb` contains the latest verified Colab execution logic for the unified dual-branch model.
  - `notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation (3).ipynb` contains the complete 36-cell implementation of the 5-round head ablation benchmark, including:
    - Auto-resume logic (`resume_path` detection to avoid restarting from scratch if Colab disconnects).
    - Multi-metric publication-ready visualization (Figures 1, 3, 4, 6, 7).
    - Post-hoc screening calibration comparison across all 5 heads.
  - In contrast, repo files `main_model_head_ablation.ipynb` (19 cells) and `(2).ipynb` (27 cells) are older partial versions.
- **Decision**:
  - Synchronize `main_model.ipynb` and `main_model (4).ipynb` to inherit the structure and logic of `main_model (8).ipynb`, updated with the full specificity outputs (80%, 85%, 90%, 95%) and Figure 7 annotations.
  - Synchronize `main_model_head_ablation.ipynb` and `(2).ipynb` to inherit the complete 36-cell architecture of `main_model_head_ablation (3).ipynb`, updated with the complete specificity metrics in `df_heads` and per-head logs.
- **Rationale**: Ensures complete code parity between local git repositories and remote Google Colab runtimes, preventing divergence in experiments and publications.

---

### R4: How should the UK National Screening Committee (UK NSC) standard and limitations be framed?
- **Finding (HD_4 Part 3)**:
  - The UK NSC standard requires $\text{Sensitivity} \ge 80\%$ AND $\text{Specificity} \ge 95\%$ *simultaneously*.
  - To achieve both simultaneously on a binormal ROC curve requires $d' \ge 1.645 + 0.842 = 2.487$, which corresponds to $\text{Ref-AUC} \ge 0.961$.
  - The current model achieves $\text{Ref-AUC} \approx 0.9063$. Thus, at $95\%$ specificity, sensitivity is $\approx 59\% - 63\%$; to achieve $\ge 80\%$ sensitivity ($84.7\%$), specificity must be set to $80\%$.
- **Decision**: In both notebook markdown commentary and exported calibration reports, explicitly label UK NSC compliance as `PASSED` only if $\text{Sens} \ge 80\%$ and $\text{Spec} \ge 95\%$ simultaneously; otherwise label as `SUB-OPTIMAL (OPERATIONAL TRADEOFF)`. State clearly in the text that reaching $84.7\%$ sensitivity requires operating at $80\%$ specificity, which is an explicit clinical trade-off.
- **Rationale**: Academic and clinical honesty. Prevents reviewers from accusing the thesis of claiming compliance when operating at $80\%$ specificity.
