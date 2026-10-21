# Research & Architectural Decisions: Baseline Oracle Training

**Feature**: `016-baseline-oracle-training`  
**Date**: 2026-09-22  
**Purpose**: Synthesize research findings, architectural decisions, and technical specifications for the 6 in-domain Oracle training pipelines.

---

## 1. Academic Rationale & Literature Alignment

### Decision 1: Establishing In-Domain Oracles to Resolve QWK Marginal Shift
- **Decision**: Train an independent in-domain model for each target cohort and evaluate both the Oracle model and the LODO OOD model on the identical 15% patient-level test split.
- **Rationale**: 
  - Section A1 in `documents/R_v0.pdf` demonstrated that calculating $\Delta = QWK_{\text{ID}} - QWK_{\text{OOD}}$ when ID is a 5-cohort pool and OOD is a single cohort with an extreme prior (e.g., EyePACS with 73.5% Grade 0) is mathematically confounded. QWK depends heavily on class marginal distributions $P(Y)$.
  - Literature in domain generalization (e.g., Gulrajani & Lopez-Paz, ICLR 2021; Koh et al., WILDS, ICML 2021) establishes the in-domain oracle as the gold standard for measuring the transfer gap:
    $$\Delta_{\text{transfer}}(\text{target}) = QWK_{\text{oracle}}(\text{target}) - QWK_{\text{OOD}}(\text{target})$$
  - Since both models are evaluated on the exact same test subset with the exact same class prior, $\Delta_{\text{transfer}}$ is an unconfounded, pure measurement of domain degradation caused by camera sensor and illumination differences.
- **Alternatives Considered**:
  - *Metric Prior Reweighting (Option 2 in R_v0.pdf)*: Reweighting pool ID to match the target OOD prior. Rejected because post-hoc reweighting QWK is statistically complex, non-standard, and difficult to justify to clinical reviewers.
  - *Discarding QWK for Per-Grade Sensitivity (Option 1 in R_v0.pdf)*: Rejected because QWK is the universal clinical benchmark metric in Diabetic Retinopathy screening competitions and literature.

---

## 2. Model Architecture & Component Reuse

### Decision 2: Strict Champion Architecture Parity (`main_model`)
- **Decision**: Re-implement the exact Champion ($V_3$) architecture from `notebooks/diabetic_retinopathy/pipeline/model/main_model (6).ipynb` and `main_model_lodo.ipynb`.
- **Key Modules**:
  1. **DualBranchMILModel**:
     - *Global Branch*: Pre-trained ResNet-50 taking $512 \times 512$ image with Ben Graham Green-channel contrast enhancement. Passes through Non-Local Block, CBAM spatial/channel attention, and 4 Quadrant Tokens to capture intra-retinal lesion distribution (4-2-1 rule).
     - *Local MIL Branch*: Pre-trained EfficientNet-B0 processing sliding-window FOV patches ($128 \times 128$). Gated Attention MIL Pooling weights patches, combined with Global Max Pooling.
     - *Latent Fusion*: Concatenation of global feature and MIL local feature, projected via a 2-layer MLP (LayerNorm, Mish, Dropout 0.3) into a 512-dimensional latent vector $z \in \mathbb{R}^{512}$.
  2. **CumulativeLinkModelQWKHead (Head 5)**:
     - 1D linear projection $s(x) = w^T z + b$.
     - Learnable strictly monotonic cutpoints: $\theta_1 \in \mathbb{R}$, $\theta_{k+1} = \theta_k + \text{softplus}(\alpha_k) + 1e-4$ for $k=1, 2, 3$.
     - Ordinal CDF formulation: $P(Y \le k \mid x) = \sigma(\theta_{k+1} - s(x))$ (using 0-indexed $k \in \{0, 1, 2, 3\}$).
     - Loss: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CLM}} + \lambda(t) \mathcal{L}_{\text{QWK}}$, where $\lambda(t) = 0.4 \times \min(1.0, t / 15)$.

---

## 3. Cohort Specifications & Dynamic Paths

### Decision 3: Cohort Ingestion & Patient-Level Grouping
- **Decision**: Map each of the 6 cohorts to its verified raw and native-cropped paths from `notebooks/diabetic_retinopathy/pipeline/merge-dataset/merge-dataset.ipynb`.

| Cohort Key | Dataset Name | Total Images | Image ID Column | Label Column | Suffix | Patient Grouping Rule |
| :--- | :--- | :---: | :--- | :--- | :---: | :--- |
| `APTOS` | APTOS 2019 | 3,662 | `id_code` | `diagnosis` | `.png` | 1 image = 1 patient (`APTOS::{id_code}`) |
| `DDR` | DDR Dataset | 12,522 | `id_code` | `diagnosis` | `.jpg` | First 2 dash tokens (`DDR::{prefix}`) |
| `DeepDRiD` | DeepDRiD | 2,000 | `image_id` | `patient_DR_Level` | `.jpg` | Underscore prefix (`DeepDRiD::{pid}`) |
| `IDRiD` | IDRiD | 516 | `image_id` | `diagnosis` | `.jpg` | Trailing ID token (`IDRiD::{number}`) |
| `Messidor-2`| Messidor-2 | 1,744 | `id_code` | `diagnosis` | `.png` | Examination date prefix (`Messidor2::{date}`) |
| `EyePACS` | EyePACS | 35,126 | `image` | `level` | `.jpeg` | Underscore prefix (`EyePACS::{pid}`) |

- **Image Path Resolution**:
  - Prefer `data/raw/hybrid/native_cropped/{dataset_key}/{image_id}.jpg` (pre-cropped FOV).
  - Fallback dynamically to raw directory if running in fresh environments.

---

## 4. Hyperparameter Schedule & Class Balancing

### Decision 4: Training & Optimization Regimes
- **Epoch Budget**: Maximum 40 epochs.
- **Batch Size**: 16 with dynamic collate (32 to 64 patches sampled during training; all patches sampled during evaluation).
- **Early Stopping**: 15 epochs without improvement on Validation QWK (`val_qwk`).
- **Learning Rate**: $1 \times 10^{-4}$ for backbones, $5 \times 10^{-4}$ for fusion MLP and CLM head.
- **LR Scheduler**: `CosineAnnealingLR(T_max=40, eta_min=1e-6)`.
- **Mixed Precision**: Automatic Mixed Precision (`torch.amp.autocast`) with `GradScaler` for bfloat16/float16.
- **Class Weights**: Dynamically computed on the 70% training split:
  $$w_c = \frac{N_{\text{train}}}{5 \cdot N_c}$$
  clipped to range $[0.5, 5.0]$ and normalized to preserve gradient magnitude.
