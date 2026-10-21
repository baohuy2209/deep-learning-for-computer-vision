# Implementation Plan: Multi-Scale All-CNN Hybrid Extractor for Diabetic Retinopathy Grading

**Branch**: `003-dr-multiscale-hybrid-extractor` | **Date**: 2026-08-11 | **Spec**: [spec.md](file:///d:/Github/deep-learning-for-computer-vision/specs/003-dr-multiscale-hybrid-extractor/spec.md)

**Input**: Feature specification from `/specs/003-dr-multiscale-hybrid-extractor/spec.md`

## Summary

This feature implements a dual-branch deep neural network combining global fundus context ($224 \times 224$ image via ResNet50 + CBAM attention) and local patch details ($K=4$ patches $128 \times 128$ via EfficientNet-B0 + max pooling). Feature maps are fused via a 2-layer MLP into a 512-dim embedding, followed by a CORAL ordinal classification head (4 binary logits $P(y > k)$) and optimized using a composite `HybridOrdinalQWKLoss` ($\alpha L_{\text{WCE}} + \beta L_{\text{Ordinal}} + \gamma L_{\text{QWK}}$). Data preprocessing is supported via a high-contrast `smart_crop_patches` sliding window variance extraction utility.

## Technical Context

**Language/Version**: Python 3.12, PyTorch $\ge 2.0.0$, `torchvision` $\ge 0.15.0$

**Primary Dependencies**: `torch`, `torch.nn`, `torchvision.models`, `numpy`, `Pillow`

**Storage**: Local file system (`data/processed/`, `src/models/multiscale_hybrid_extractor.py`)

**Testing**: PyTorch execution self-test block (`if __name__ == '__main__':`) in `src/models/multiscale_hybrid_extractor.py`

**Target Platform**: GPU (CUDA) / CPU execution environment

**Project Type**: Deep Learning Model & Preprocessing Library (`src/models/`)

**Performance Goals**: Forward pass inference time $<50$ms per image batch on GPU; `smart_crop_patches` $<100$ms per 1024x1024 fundus image.

**Constraints**: Memory allocation within standard GPU limits (Batch size 4, $K=4$ patches requiring $<4$GB VRAM); strict zero NaN/Inf gradient stability.

**Scale/Scope**: Unilateral (single-eye) 5-class DR severity grading ($0 \dots 4$) across multi-source datasets (APTOS, EyePACS, Messidor-2, DDR, DeepDRiD, IDRiD).

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Principle I (Research Context & Methodology)**: PASS. Problem is framed as Ordinal Classification using CORAL framework; primary loss incorporates direct QWK optimization ($L_{\text{QWK}}$); architecture uses a Dual-branch global/local patch fusion model.
- **Principle II (Strict File & Code Integrity)**: PASS. Code generated in a NEW dedicated module `src/models/multiscale_hybrid_extractor.py` without modifying or overwriting existing baseline files.
- **Principle III (Strict Directory Path Adherence)**: PASS. Adheres strictly to project paths: specs under `specs/003-dr-multiscale-hybrid-extractor/` and PyTorch model under `src/models/`.
- **Principle IV (Code Generation & Reproducibility Rules)**: PASS. Code is fully functional without `pass` or `TODO` placeholders; random seeds are set; self-contained execution test block included.

## Project Structure

### Documentation (this feature)

```text
specs/003-dr-multiscale-hybrid-extractor/
├── plan.md              # This file (/speckit-plan command output)
├── research.md          # Phase 0 output (/speckit-plan command)
├── data-model.md        # Phase 1 output (/speckit-plan command)
├── quickstart.md        # Phase 1 output (/speckit-plan command)
├── contracts/           # Phase 1 output (/speckit-plan command)
│   └── model_interface.md
└── checklists/
    └── requirements.md  # Spec quality checklist
```

### Source Code (repository root)

```text
src/
├── models/
│   └── multiscale_hybrid_extractor.py   # MultiScaleHybridExtractor, CBAM, Loss, Smart Crop
```

**Structure Decision**: Single project model directory under `src/models/multiscale_hybrid_extractor.py` housing all modular components (`GlobalBranch`, `LocalBranch`, `CBAM`, `OrdinalHead`, `MultiScaleHybridExtractor`, `HybridOrdinalQWKLoss`, `smart_crop_patches`).

## Complexity Tracking

*No Constitution violations. Table empty.*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| None | N/A | N/A |
