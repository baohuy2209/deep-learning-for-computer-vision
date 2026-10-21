# Feature Specification: Multi-Scale All-CNN Hybrid Extractor for Diabetic Retinopathy Grading

**Feature Branch**: `003-dr-multiscale-hybrid-extractor`

**Created**: 2026-08-11

**Status**: Draft

**Input**: User description: "/speckit-specify Act as a Senior AI/Deep Learning Researcher specializing in Medical Image Analysis and PyTorch. I need you to write a complete, modular, and well-commented PyTorch implementation for a custom neural network called 'Multi-Scale All-CNN Hybrid Extractor'. This model is designed for 5-class Diabetic Retinopathy (DR) grading using a Unilateral (single-eye) approach."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Multi-Scale Feature Extraction and Representation (Priority: P1)

As a deep learning researcher, I want to pass both a whole-fundus global image and multiple high-resolution local patches into a hybrid neural network architecture, so that macroscopic structural context (e.g., optical disc alignment) and localized lesion details (microaneurysms, hemorrhages) are jointly extracted and fused into a unified embedding vector.

**Why this priority**: DR severity grading relies on both global ocular features and micro-lesions. Combining zoom-out global features with zoom-in local patches addresses resolution bottlenecks inherent in single-scale CNN architectures.

**Independent Test**: Can be independently verified by feeding a batch of global images `[Batch, 3, 224, 224]` and local patch tensors `[Batch, K, 3, 128, 128]` into the feature extraction branches to verify output feature dimensions `F_global` `[Batch, 2048]` and `F_local` `[Batch, 1280]`.

**Acceptance Scenarios**:

1. **Given** a global image tensor of shape `[Batch, 3, 224, 224]`, **When** processed through the global ResNet50 backbone enhanced with CBAM attention, **Then** a 2048-dimensional global feature embedding `F_global` is extracted.
2. **Given** a tensor of $K=4$ cropped local patches of shape `[Batch, 4, 3, 128, 128]`, **When** processed through the EfficientNet-B0 backbone using batch unflattening and max-pooling aggregation across patches, **Then** a 1280-dimensional local feature embedding `F_local` is produced.

---

### User Story 2 - Consistent Rank Logits (CORAL) Ordinal Classification (Priority: P2)

As a clinical AI specialist, I want the classification head to output 4 binary logits representing cumulative disease progression probabilities $P(y > 0), P(y > 1), P(y > 2), P(y > 3)$ rather than standard 5-class nominal Softmax probabilities, so that ordinal distance between DR grades is explicitly preserved.

**Why this priority**: Diabetic Retinopathy is inherently ordinal (No DR < Mild < Moderate < Severe < Proliferative). Nominal multi-class loss functions penalize misclassifications equally regardless of grade distance, whereas CORAL ordinal regression respects disease progression ranking.

**Independent Test**: Can be tested independently by forwarding a fused feature tensor `[Batch, 3328]` through the MLP and Ordinal Head to obtain 4 cumulative binary logits and verifying that derived class probabilities sum to 1.0.

**Acceptance Scenarios**:

1. **Given** concatenated global and local embeddings `[Batch, 3328]`, **When** passed through the 2-layer MLP and CORAL Ordinal Head, **Then** the model outputs 4 binary logits corresponding to $P(y > k)$ for $k \in \{0, 1, 2, 3\}$.
2. **Given** 4 ordinal logits, **When** probabilities are accumulated and subtracted sequentially, **Then** valid 5-class discrete probability distributions $P(y = k)$ summing to 1.0 are reconstructed for loss computation and evaluation.

---

### User Story 3 - Hybrid Ordinal & Quadratic Weighted Kappa Loss Optimization (Priority: P3)

As a model training engineer, I want a custom composite loss module `HybridOrdinalQWKLoss` combining Weighted Cross-Entropy ($L_{WCE}$), Binary Cross-Entropy over CORAL logits ($L_{Ordinal}$), and Soft Quadratic Weighted Kappa ($L_{QWK}$), so that model optimization directly maximizes the primary clinical evaluation metric (QWK) while maintaining stable gradient flow.

**Why this priority**: Standard cross-entropy loss does not directly optimize for QWK. Combining ordinal loss, class-weighted cross-entropy, and soft QWK loss balances class imbalance penalty, ordinal ranking stability, and QWK score direct alignment.

**Independent Test**: Can be tested by computing loss on dummy logits and targets, verifying non-zero scalar loss return, and checking backward gradient propagation through all network parameters.

**Acceptance Scenarios**:

1. **Given** predicted CORAL logits `[Batch, 4]` and target DR grades `[Batch]`, **When** evaluated via `HybridOrdinalQWKLoss`, **Then** total loss is computed as $\alpha L_{WCE} + \beta L_{Ordinal} + \gamma L_{QWK}$ with backpropagation enabled.
2. **Given** severe class imbalance in DR datasets, **When** $L_{WCE}$ is computed using inverse class frequency weights, **Then** underrepresented severe DR classes receive higher loss weighting.

---

### User Story 4 - High-Contrast Smart Crop Patch Selection (Priority: P4)

As a data preprocessing developer, I want a utility function `smart_crop_patches(image, K=4, patch_size=128)` that scans high-resolution fundus images using sliding windows to detect regions with highest visual variance or entropy (microaneurysms, exudates, hemorrhages), so that local patches capture clinically relevant lesions automatically.

**Why this priority**: Random cropping often selects uniform dark background regions of fundus images. Smart cropping based on variance/entropy ensures high-information pathology patches are extracted.

**Independent Test**: Can be tested by running `smart_crop_patches` on sample fundus images and verifying that $K$ patch tensors of shape `[K, 3, 128, 128]` are returned corresponding to high-contrast regions.

**Acceptance Scenarios**:

1. **Given** a high-resolution fundus image, **When** `smart_crop_patches` is called with $K=4$ and `patch_size=128`, **Then** the function returns 4 non-overlapping patches containing maximum local intensity variance/entropy.

---

### Edge Cases

- **Handling batch size of 1**: BatchNorm layers in MLP and backbone must handle `Batch=1` gracefully without throwing runtime errors (e.g., during validation/inference).
- **Zero variance in fundus background**: When an image has large flat background regions, patch selection must fallback safely to uniform grid sampling without indexing errors or NaN variance divisions.
- **Extreme Class Imbalance**: Missing classes in a mini-batch during QWK loss computation must be handled using epsilon smoothing to prevent zero-division errors.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST define a modular `CBAM` PyTorch module implementing sequential Channel Attention (AvgPool + MaxPool + Shared MLP) and Spatial Attention (Channel AvgPool/MaxPool + 7x7 Conv) with sigmoid activation.
- **FR-002**: System MUST define a `GlobalBranch` PyTorch module utilizing a pre-trained ResNet50 backbone (without final FC layer) followed by CBAM attention and adaptive pooling, producing a 2048-dimensional feature tensor `F_global`.
- **FR-003**: System MUST define a `LocalBranch` PyTorch module accepting a batch of $K$ local patches `[Batch, K, 3, 128, 128]`, reshaping to `[Batch * K, 3, 128, 128]`, extracting features via EfficientNet-B0 to `[Batch * K, 1280]`, unflattening to `[Batch, K, 1280]`, and aggregating via max-pooling across $K$ patches to produce `F_local` `[Batch, 1280]`.
- **FR-004**: System MUST define a feature fusion MLP combining `F_global` and `F_local` (concatenated shape `[Batch, 3328]`) through a 2-layer MLP (Linear -> BatchNorm -> ReLU -> Dropout(0.5) -> Linear) reducing dimensionality to 512.
- **FR-005**: System MUST define an `OrdinalHead` module outputting 4 binary logits for CORAL ordinal classification representing $P(y > k)$ for $k \in \{0, 1, 2, 3\}$.
- **FR-006**: System MUST combine `GlobalBranch`, `LocalBranch`, Fusion MLP, and `OrdinalHead` into an end-to-end `MultiScaleHybridExtractor` PyTorch `nn.Module`.
- **FR-007**: System MUST implement `HybridOrdinalQWKLoss` combining $L_{Ordinal}$ (BCE with logits), $L_{WCE}$ (Weighted Cross-Entropy over derived 5-class probabilities), and $L_{QWK}$ (differentiable Soft Quadratic Weighted Kappa) weighted by hyperparameters $\alpha, \beta, \gamma$.
- **FR-008**: System MUST provide a `smart_crop_patches(image, K=4, patch_size=128)` utility function evaluating sliding window variance/entropy to extract $K$ informative patches from high-resolution fundus images.
- **FR-009**: System MUST include explicit tensor shape annotations in code comments for every intermediate transformation in the model forward pass.
- **FR-010**: System MUST include a self-contained execution test block (`if __name__ == '__main__':`) verifying dummy batch forward pass, loss calculation, backward pass, and smart cropping without Out-Of-Memory or shape mismatch errors.

### Key Entities

- **Global Context Tensor**: Input tensor `[Batch, 3, 224, 224]` representing resized full ocular view.
- **Local Patch Tensor**: Input tensor `[Batch, K, 3, 128, 128]` representing $K$ high-resolution localized crops.
- **Fused Embedding Vector**: 512-dimensional latent representation combining global spatial relations and local lesion features.
- **CORAL Logit Vector**: 4-element output tensor representing cumulative ordinal thresholds $P(y > 0), P(y > 1), P(y > 2), P(y > 3)$.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Model forward pass executes on standard GPU/CPU memory with batch size of 4 without memory allocation or tensor dimension mismatch errors.
- **SC-002**: Reconstruction of 5 nominal class probabilities from 4 CORAL ordinal logits satisfies $\sum_{k=0}^4 P(y = k) = 1.0 \pm 1e-6$ across all predictions.
- **SC-003**: `HybridOrdinalQWKLoss` returns non-zero differentiable scalar loss value and generates non-null gradients across all trainable parameters during backpropagation.
- **SC-004**: `smart_crop_patches` successfully selects $K=4$ non-overlapping 128x128 patches from 1024x1024 fundus images in under 100ms per image.

## Assumptions

- Input fundus images are preprocessed and RGB normalized to standard ImageNet mean and standard deviation.
- PyTorch version $\ge 2.0.0$ and `torchvision` $\ge 0.15.0$ are available in the execution environment.
- Single-eye (unilateral) diagnosis mode is assumed where each input pair (`img_global`, `patches_local`) corresponds to one fundus image.
