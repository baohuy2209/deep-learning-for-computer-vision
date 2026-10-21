# Model Architecture & Mathematical Foundations

This document provides a detailed specification of the network architecture, tensor dimensions, attention mechanisms, cross-scale fusion, and the Cumulative Link Model (CLM) ordinal classification formulation in `drdg`.

---

## 1. Architectural Overview

Diabetic retinopathy (ICDRS Grades 0 to 4) exhibits multi-scale pathological manifestations:
- **Global Context**: Diffuse vascular tortuosity, quadrant-wide microvascular abnormalities (IRMA), optic disc neovascularization, and overall retinal symmetry.
- **Local Lesions**: Highly localized focal microaneurysms ($<50\,\mu\text{m}$), punctate dot-and-blot haemorrhages, and hard lipid exudates that become lost during naive downsampling.

To jointly address both regimes, the proposed architecture combines a **Global Context Stream** and a **Local High-Resolution Patch Multiple Instance Learning (MIL) Stream**, fused through an ordinal cumulative link mechanism:

```
[Global Fundus Image: 3 x 512 x 512]
         │
         ▼
[ResNet-50 Feature Extractor (Layer 4: 2048 x 16 x 16)]
         │
         ▼
[Channel Compression: 2048 -> 512]
         │
         ├────────────────────────────────────────┐
         ▼                                        ▼
[CBAM (Channel + Spatial)]               [Non-Local Self-Attention]
         │                                        │
         └───────────────────┬────────────────────┘
                             ▼
                 [Quadrant Token Pooling]
                 - 4 Quadrants: 4 x 512 = 2048
                 - Global GAP: 512
                 - Concatenated -> Project to 512 (h_global)
                             │
                             ├──────────────────────────────────────┐
                             │                                      │
[Local Retinal Patches: K x 3 x 128 x 128]                          │
         │                                                          │
         ▼                                                          │
[EfficientNet-B0 Patch Backbone: K x 1280]                         │
         │                                                          │
         ▼                                                          │
[Gated Attention Dual Pooling (Mean + Top-k Max)]                   │
         │                                                          │
         ▼                                                          │
[Local Representation Projection: 512 (h_local)]                    │
         │                                                          │
         └───────────────────┬──────────────────────────────────────┘
                             ▼
              [Residual Multimodal Fusion: 1024 -> 512]
                             │
                             ▼
              [Cumulative Link Model (CLM) Head]
              - Latent Linear Score: z = w^T h_fused
              - Softplus Monotonic Cutpoints: b_0 < b_1 < b_2 < b_3
              - Complementary Log-Log (clog-log) Link Function
                             │
                             ▼
              [Ordinal Probability Distribution: p_0, ..., p_4]
```

---

## 2. Feature Representation & Tensor Dimensions

### 2.1 Global Context Stream (`drdg.models.global_stream.GlobalContextStream`)
- **Input**: $\mathbf{x}_{\text{global}} \in \mathbb{R}^{B \times 3 \times 512 \times 512}$.
- **Backbone**: ResNet-50 truncated at `layer4`, producing feature tensor $\mathbf{F} \in \mathbb{R}^{B \times 2048 \times 16 \times 16}$.
- **Channel Projection**: $1 \times 1$ convolution reducing channels: $2048 \rightarrow 512$ ($\mathbf{F}' \in \mathbb{R}^{B \times 512 \times 16 \times 16}$).
- **CBAM Attention**: Computes channel attention $\mathbf{M}_c(\mathbf{F}')$ and spatial attention $\mathbf{M}_s(\mathbf{F}')$ sequentially (`drdg.models.attention.CBAM`).
- **Non-Local Attention**: Captures long-range bilateral vascular relationships across hemispheres (`drdg.models.attention.NonLocalBlock2D`).
- **Quadrant Token Pooling**: Subdivides the $16 \times 16$ spatial grid into 4 anatomic quadrants ($8 \times 8$ each), computing average and max representations per quadrant ($4 \times 512 = 2048$). Concatenated with global average pooling ($512$) and projected (`drdg.models.attention.QuadrantTokenAggregator`):
  $$\mathbf{h}_{\text{global}} = \text{MLP}(\text{Concat}[\mathbf{q}_1, \mathbf{q}_2, \mathbf{q}_3, \mathbf{q}_4, \mathbf{g}]) \in \mathbb{R}^{B \times 512}$$

### 2.2 Local Multiple Instance Learning (MIL) Stream (`drdg.models.local_mil.LocalMILBranch`)
- **Input**: Bag of $K$ high-resolution patches $\mathbf{X}_{\text{local}} \in \mathbb{R}^{B \times K \times 3 \times 128 \times 128}$ (default $K=16$).
- **Instance Encoder**: EfficientNet-B0 backbone yielding instance embeddings $\mathbf{h}_k \in \mathbb{R}^{1280}$ for each patch $k \in \{1, \dots, K\}$.
- **Gated Attention Mechanism** (`drdg.models.local_mil.GatedAttentionPool`):
  $$\mathbf{a}_k = \frac{\exp\left(\mathbf{w}^T \left(\tanh(\mathbf{V} \mathbf{h}_k) \odot \sigma(\mathbf{U} \mathbf{h}_k)\right)\right)}{\sum_{j=1}^K \exp\left(\mathbf{w}^T \left(\tanh(\mathbf{V} \mathbf{h}_j) \odot \sigma(\mathbf{U} \mathbf{h}_j)\right)\right)}$$
  where $\mathbf{V}, \mathbf{U} \in \mathbb{R}^{D_a \times 1280}$ and $\mathbf{w} \in \mathbb{R}^{D_a}$ are learnable parameters.
- **Dual Aggregation**: Computes both attention-weighted average $\sum_k a_k \mathbf{h}_k$ and top-$k$ instance max pooling to ensure sparse, solitary microaneurysms trigger detection. Projected to:
  $$\mathbf{h}_{\text{local}} \in \mathbb{R}^{B \times 512}$$

### 2.3 Cross-Scale Fusion (`drdg.models.fusion.DualBranchFusion`)
- Concatenates global and local vectors:
  $$\mathbf{h}_{\text{concat}} = [\mathbf{h}_{\text{global}} \,\|\, \mathbf{h}_{\text{local}}] \in \mathbb{R}^{B \times 1024}$$
- Passes through LayerNorm, residual MLP block, and dropout ($p=0.2$):
  $$\mathbf{h}_{\text{fused}} = \text{GELU}(\mathbf{W}_2 \, \text{Dropout}(\text{GELU}(\mathbf{W}_1 \mathbf{h}_{\text{concat}}))) + \mathbf{W}_{\text{res}} \mathbf{h}_{\text{concat}} \in \mathbb{R}^{B \times 512}$$

---

## 3. Cumulative Link Model (CLM) Formulation

Unlike standard categorical cross-entropy which treats severity grades ($0 < 1 < 2 < 3 < 4$) as unordered classes, the CLM respects strict ordinal geometry.

### 3.1 Latent Linear Score
The network maps the fused representation $\mathbf{h}_{\text{fused}}$ to a scalar disease progression score:
$$z = \mathbf{w}_{\text{clm}}^T \mathbf{h}_{\text{fused}} \in \mathbb{R}$$

### 3.2 Monotonic Cutpoints via Softplus
To prevent threshold inversion ($b_j \ge b_{j+1}$), thresholds are parameterized via unconstrained real parameters $\theta_0, \Delta_1, \Delta_2, \Delta_3 \in \mathbb{R}$:
$$b_0 = \theta_0$$
$$b_k = b_{k-1} + \text{softplus}(\Delta_k) \quad \text{for } k \in \{1, 2, 3\}$$
This guarantees the strict monotonic ordering $b_0 < b_1 < b_2 < b_3$.

### 3.3 Complementary Log-Log (clog-log) Link
For skewed ordinal distributions where extreme stages (Severe NPDR, PDR) occur at lower frequencies, the asymmetric clog-log link is employed:
$$F(u) = 1 - \exp(-\exp(u))$$
The cumulative probability of class $Y \le k$ is:
$$P(Y \le k \mid z) = F(b_k - z) = 1 - \exp(-\exp(b_k - z))$$

Class probabilities are computed by differencing cumulative bounds:
$$P(Y = 0) = P(Y \le 0)$$
$$P(Y = k) = P(Y \le k) - P(Y \le k-1) \quad \text{for } k \in \{1, 2, 3\}$$
$$P(Y = 4) = 1 - P(Y \le 3)$$

### 3.4 Combined Loss Function
The model is optimized via negative log-likelihood combined with Continuous Quadratic Weighted Kappa (QWK) loss:
$$\mathcal{L}_{\text{total}} = -\sum_{i=1}^B \log P(Y = y_i \mid z_i) + \lambda_{\text{qwk}} \mathcal{L}_{\text{QWK}}(\mathbf{p}, \mathbf{y})$$
where $\lambda_{\text{qwk}} = 1.0$ and $\mathcal{L}_{\text{QWK}}$ computes the continuous quadratic penalty matrix $W_{ij} = \frac{(i - j)^2}{(C - 1)^2}$.

---

## 4. Model Variant Progression

| Variant | Global Branch | Local Branch | Output Head | Primary Hypothesis |
|---|---|---|---|---|
| **`v0`** | Standard ResNet-50 | None | Softmax Cross-Entropy | Standard global image classification baseline |
| **`v1`** | Standard ResNet-50 | EfficientNet-B0 + Gated Attention | Softmax Cross-Entropy | Adding high-resolution patch MIL detects small focal lesions |
| **`v2`** | ResNet-50 + CBAM + NonLocal + Quadrants | EfficientNet-B0 + Gated Attention | Softmax Cross-Entropy | Global context attention captures structural symmetry & disc changes |
| **`v3`** (Proposed) | ResNet-50 + CBAM + NonLocal + Quadrants | EfficientNet-B0 + Gated Attention | Cumulative Link Model (CLM) + QWK | Ordinal formulation aligns predictions with clinical disease progression |
