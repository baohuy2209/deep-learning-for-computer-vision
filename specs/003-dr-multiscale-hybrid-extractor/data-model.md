# Data Model & Tensor Architecture: Multi-Scale All-CNN Hybrid Extractor

**Feature**: Multi-Scale All-CNN Hybrid Extractor for Diabetic Retinopathy Grading
**Branch**: `003-dr-multiscale-hybrid-extractor`
**Date**: 2026-08-11

---

## 1. Primary Tensors & Shapes

### Input Tensors

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| `img_global` | `torch.FloatTensor` | `[Batch, 3, 224, 224]` | Resized whole-fundus image capturing global macroscopic ocular structure. |
| `patches_local` | `torch.FloatTensor` | `[Batch, K, 3, 128, 128]` | $K$ high-contrast cropped patches (default $K=4$) capturing micro-lesion details. |
| `targets` | `torch.LongTensor` | `[Batch]` | Discrete ground-truth DR severity grades ($y \in \{0, 1, 2, 3, 4\}$). |

### Intermediate Representations

| Module | Tensor Name | Shape | Transformation / Operation |
|--------|-------------|-------|----------------------------|
| `GlobalBranch` | `F_global` | `[Batch, 2048]` | ResNet50 layer4 $\rightarrow$ CBAM attention $\rightarrow$ AdaptiveAvgPool2d(1) $\rightarrow$ Flatten. |
| `LocalBranch` | `patches_flat` | `[Batch * K, 3, 128, 128]` | Reshape trick combining batch and patch dimensions. |
| `LocalBranch` | `feat_patches` | `[Batch * K, 1280]` | EfficientNet-B0 features $\rightarrow$ AvgPool $\rightarrow$ Flatten. |
| `LocalBranch` | `F_local` | `[Batch, 1280]` | Unflatten to `[Batch, K, 1280]` $\rightarrow$ `torch.max(dim=1)` patch max-pooling. |
| `Fusion MLP` | `F_fused` | `[Batch, 3328]` | Channel concatenation `torch.cat([F_global, F_local], dim=1)`. |
| `Fusion MLP` | `F_latent` | `[Batch, 512]` | 2-layer MLP (`Linear(3328, 1024)` $\rightarrow$ `BN` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.5)` $\rightarrow$ `Linear(1024, 512)` $\rightarrow$ `BN` $\rightarrow$ `ReLU`). |

### Output Tensors

| Name | Type | Shape | Description |
|------|------|-------|-------------|
| `logits` | `torch.FloatTensor` | `[Batch, 4]` | CORAL binary logits corresponding to $P(y > 0), P(y > 1), P(y > 2), P(y > 3)$. |
| `class_probs` | `torch.FloatTensor` | `[Batch, 5]` | Derived discrete nominal class probabilities $\sum_{k=0}^4 P(y = k) = 1.0$. |

---

## 2. Ordinal Rank & Probability Mapping Rules

Given CORAL logits $Z = [z_0, z_1, z_2, z_3]$:

$$\text{Cumulative Probabilities: } P(y > k) = \sigma(z_k) \quad \text{for } k \in \{0, 1, 2, 3\}$$

$$\text{Derived Class Probabilities:}$$
- $P(y = 0) = 1 - P(y > 0)$
- $P(y = 1) = P(y > 0) - P(y > 1)$
- $P(y = 2) = P(y > 1) - P(y > 2)$
- $P(y = 3) = P(y > 2) - P(y > 3)$
- $P(y = 4) = P(y > 3)$

$$\text{Predicted Grade: } \hat{y} = \sum_{k=0}^3 \mathbb{I}(\sigma(z_k) > 0.5)$$

---

## 3. Preprocessing Entity: Smart Crop Patch Selection

- **Input**: High-resolution image (PIL Image or NumPy array $H \times W \times 3$)
- **Algorithm**:
  1. Convert image to grayscale intensity map $I(x, y)$.
  2. Sliding window grid search with step size $\text{step} = \text{patch\_size} / 2$.
  3. Calculate window variance $\text{Var}(I_{\text{win}})$.
  4. Non-Maximum Suppression (NMS) to reject overlapping windows.
  5. Select top $K$ non-overlapping patches of shape $[128, 128, 3]$.
