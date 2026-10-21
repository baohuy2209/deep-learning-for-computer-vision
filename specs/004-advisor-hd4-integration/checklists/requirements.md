# Specification Quality Checklist: Advisor HD_4 Proposed Model Experimental Upgrade

**Purpose**: Validate specification completeness, testability, and non-regression guardrails before proceeding to implementation planning.  
**Created**: 2026-09-03  
**Feature Spec**: [spec.md](../spec.md)  
**Governing Constitution**: [.specify/memory/constitution.md](file:///d:/Github/deep-learning-for-computer-vision/.specify/memory/constitution.md) (v1.1.0)  

---

## Content Quality

- [x] No unauthorized implementation details or generic placeholders
- [x] Focused on clinical research rigor, academic validation, and advisor directives
- [x] Written clearly for academic review and clinical stakeholders
- [x] All mandatory sections completed with concrete specifications

## Requirement Completeness

- [x] No `[NEEDS CLARIFICATION]` markers remain (all defaults derived from `HD_4.pdf`)
- [x] Requirements are testable, unambiguous, and mapped to specific components (FR-001 to FR-015)
- [x] Success criteria are measurable and verifiable (SC-001 to SC-006)
- [x] Acceptance scenarios are defined with clear Given-When-Then criteria
- [x] Clinical edge cases are identified with explicit mitigation strategies
- [x] Scope is clearly bounded (decoupling screening calibration, FoV dual-crop, Attention-MIL saliency)
- [x] Dependencies and assumptions strictly documented

## Feature Readiness & Constitution Adherence

- [x] Strict Preservation Directive: Existing validated pipelines, models, and paths are preserved
- [x] Minimal Scope of Modification (Delta-Only): Edits confined strictly to targeted functions in `HD_4`
- [x] Pre-Modification Verification: Dependencies pre-screened to guarantee zero code regression
- [x] Ready for implementation planning and surgical differential updates

---

## Validation Notes

- **Dual-Crop Fix**: `IDRiDXAIEvalDataset` applies identical bounding box from `compute_fov_mask(img_np, erode_px=0)` to both image and mask.
- **XAI Overhaul**: Replaces raw Mean IoU failure narrative with Pointing Game (vs 1.5% random baseline), Area-Matched Saliency Recall, and Pixel AUROC.
- **Attention-MIL Saliency**: Projects $\alpha_k$ via `[y0, x0]` to achieve native ~128 px resolution maps.
- **Screening Calibration**: Fit operating points on Val predicted probabilities $P(y \ge 2)$ and test on held-out Test set at spec 95%, 90%, 85%, 80%.
- **All checklist items passed**: Ready for `/speckit-plan` or execution phase.
