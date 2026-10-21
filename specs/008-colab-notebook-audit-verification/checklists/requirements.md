# Specification Quality Checklist: Colab Dual-Branch Notebook Audit & Verification

**Purpose**: Validate specification completeness and audit verification before proceeding to Google Colab execution  
**Created**: 2026-09-11  
**Feature**: [spec.md](../spec.md)

## Content Quality

- [x] Clear comparison between candidate notebooks: `main_model (4).ipynb` vs `main_model (8).ipynb`
- [x] Complete mapping against all advisor guidelines: `HD_1.pdf`, `HD_2.pdf`, `HD_3.pdf`, `HD_4.pdf`
- [x] Exact cell numbers and code snippets identified for all critical bugs and fixes
- [x] Clear explanation of theoretical rationale (loss stabilization, calibration, XAI scale alignment)
- [x] Practical pre-flight checklist tailored specifically for Google Colab environments

## Requirement Completeness

- [x] Output heads compared and verified (Softmax_CE, Softmax_QWK, CORAL, CORN, CLM_QWK)
- [x] Hypotheses H1, H2, H3 architectural alignment verified
- [x] Monotonicity violation measurement verified on rank probabilities
- [x] Collate and DataLoader IPC bandwidth optimizations verified (uint8 tensors, GPU normalization)
- [x] EarlyStopping and ModelCheckpoint criteria unified on Val QWK (patience 15)
- [x] QWK continuous loss stabilized (target lambda 0.4, 15-epoch warmup)
- [x] Dual-crop bounding box alignment on IDRiD images and lesion masks verified
- [x] Attention-MIL architecture-native saliency map verified
- [x] Post-hoc screening threshold calibration verified
- [x] Colab runtime pitfalls identified (extraction, Drive paths, GPU VRAM, unassign)

## Feature Readiness

- [x] `main_model (4).ipynb` confirmed as 100% compliant with HD_1 - HD_4 directives
- [x] `main_model (8).ipynb` flagged with 4 fatal runtime/OOM bugs and advisor non-compliances
- [x] Colab execution instructions prepared for immediate user action
