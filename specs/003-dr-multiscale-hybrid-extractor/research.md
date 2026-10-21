# Phase 0 Research: Multi-Scale All-CNN Hybrid Extractor

**Feature**: Multi-Scale All-CNN Hybrid Extractor for Diabetic Retinopathy Grading
**Branch**: `003-dr-multiscale-hybrid-extractor`
**Date**: 2026-08-11

---

## 1. Global Context Branch Architecture

- **Decision**: Pre-trained `ResNet50` backbone combined with Convolutional Block Attention Module (`CBAM`) after layer4 feature maps.
- **Rationale**: ResNet50 provides robust macroscopic spatial feature representation for full fundus images ($224 \times 224$). Adding CBAM attention allows the network to dynamically weight informative spatial channels (optical disc, vascular arch) before adaptive global pooling, producing a compact $2048$-dimensional feature vector $F_{\text{global}}$.
- **Alternatives Considered**:
  - *Standard ResNet50 without attention*: Lacks explicit channel/spatial re-weighting for global retinal orientation.
  - *ViT / Swin Transformer*: Higher memory footprint and requires larger pre-training datasets compared to ResNet50 for global context.

---

## 2. Local Patch Branch & Aggregation Mechanism

- **Decision**: Pre-trained `EfficientNet-B0` backbone with batch flattening ($[B \times K, 3, 128, 128]$) and Max-Pooling feature aggregation across the $K$ patch dimension.
- **Rationale**: EfficientNet-B0 offers high parameter efficiency ($1280$-dim output) for fine-grained patch processing. Reshaping $B$ and $K$ into a single dimension allows standard 2D CNN forward execution. Max-pooling across $K$ patches ($\text{max}_{k=1..K}$) selects the strongest lesion activation (e.g., microaneurysms, hemorrhages, exudates) for each feature channel.
- **Alternatives Considered**:
  - *Average Pooling across patches*: Smoothes out localized micro-lesions, diluting strong focal diagnostic signals.
  - *Patch Attention / Transformer Aggregation*: Introduces extra parameter complexity without clear benefit over channel-wise max activation for focal lesions.

---

## 3. Ordinal Regression Framework (CORAL)

- **Decision**: Consistent Rank Logits (CORAL) framework outputting 4 binary logits representing cumulative threshold probabilities $P(y > 0), P(y > 1), P(y > 2), P(y > 3)$.
- **Rationale**: Diabetic Retinopathy severity grading is inherently ordinal ($0 < 1 < 2 < 3 < 4$). Nominal Softmax multi-class loss treats adjacent grade errors (e.g., Grade 0 vs. Grade 1) identically to extreme errors (e.g., Grade 0 vs. Grade 4). CORAL enforces ordinal rank consistency and directly aligns with Quadratic Weighted Kappa (QWK) evaluation metrics.
- **Alternatives Considered**:
  - *Standard 5-class Softmax*: Fails to model disease progression continuity; unstable under severe class imbalance.
  - *Unconstrained Regression (MSE)*: Outputs unbounded continuous values requiring arbitrary threshold tuning.

---

## 4. Hybrid Loss Formulation (`HybridOrdinalQWKLoss`)

- **Decision**: Composite loss function: $\text{Loss} = \alpha L_{\text{WCE}} + \beta L_{\text{Ordinal}} + \gamma L_{\text{QWK}}$.
  - $L_{\text{Ordinal}}$: Binary Cross Entropy over 4 CORAL rank logits.
  - $L_{\text{WCE}}$: Weighted Cross-Entropy using inverse class frequencies on derived 5-class probabilities.
  - $L_{\text{QWK}}$: Differentiable Soft Quadratic Weighted Kappa loss based on expected rank MSE.
- **Rationale**: Multi-objective loss optimization provides balanced gradient signals: $L_{\text{Ordinal}}$ stabilizes rank ordering, $L_{\text{WCE}}$ penalizes underrepresented severe DR classes, and $L_{\text{QWK}}$ directly optimizes the primary clinical benchmark metric.
- **Alternatives Considered**:
  - *Pure QWK Loss*: Highly non-convex with flat gradients in early training stages.
  - *BCE-only Loss*: Ignores total class distribution imbalance in multi-dataset training.

---

## 5. High-Contrast Smart Crop Patch Selection

- **Decision**: Sliding window grid search scanning high-resolution fundus images to rank patches by local intensity variance, followed by Non-Maximum Suppression (NMS) spatial overlap filtering.
- **Rationale**: Microaneurysms and exudates present high local intensity variance against background retina. Variance-based patch ranking automatically captures pathology regions without requiring manual bounding-box annotations.
- **Alternatives Considered**:
  - *Random Cropping*: Frequently crops uniform black or empty retinal background.
  - *Entropy-based Cropping*: Slightly more computationally expensive than variance with near-identical patch selection quality.
