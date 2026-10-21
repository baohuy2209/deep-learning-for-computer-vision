# Feature Specification: Head Ablation Benchmark HD_4 Integration

**Feature Branch / Directory**: `specs/005-head-ablation-hd4-upgrade`  
**Target System**: 5-Round End-to-End Output Head Ablation Benchmark on ResNet-50 (`main_model_head_ablation (3).ipynb`)  
**Reference Document**: Advisor Review Notes (`documents/HD_4.pdf`)  
**Core Objective**: Align the 5-round head ablation benchmark with the advisor's technical directives: loss stabilization, post-hoc screening threshold calibration, and convergence synchronization.

---

## 1. Context & Problem Statement

In `documents/HD_4.pdf`, the advisor evaluated the training dynamics and clinical metrics of the diabetic retinopathy grading pipeline, identifying critical methodological insights:
1. **Optimization Stagnation & Gradient Variance (Part 4)**: Early in training, soft confusion matrices on mini-batches ($B=48$) generate high-variance gradients. If the continuous QWK loss penalty $\lambda$ ramps up too rapidly (5-epoch warmup to $\lambda=1.2$), it destabilizes the cross-entropy / NLL objective. The warmup must be extended to 15 epochs, and the target weight calibrated to $\lambda=0.4$ (matching the $0.3 - 0.5$ recommendation).
2. **Stopping Criteria Synchronization (Part 4)**: Validation cross-entropy loss frequently plateaus while validation QWK continues to ascend. Both `EarlyStopping` and `ModelCheckpoint` must monitor `Validation QWK` with patience $\ge 15$ epochs.
3. **Clinical Screening Decoupling & Post-Hoc Calibration (Part 1 & Part 3)**: Taking multi-class `argmax` for binary referable DR screening ($y \ge 2$) operates at an arbitrary high-specificity threshold (~95%), artificially depressing sensitivity to ~64.8% despite a strong underlying Ref-AUC of $0.9063$. Class weights should not be used to distort probability calibration; instead, operating thresholds $\tau$ must be calibrated post-hoc strictly on Validation set continuous probabilities $P(y \ge 2)$ at fixed specificities (95%, 90%, 85%, 80%), and evaluated on the held-out Test set.

---

## 2. User Stories & Acceptance Criteria

### User Story 1: Loss Stabilization & Smooth Convergence (Priority: P1)
As a researcher, I need the QWK loss warmup in `Softmax_QWK` and `CumulativeLinkModelQWKHead` to smoothly transition across 15 epochs to $\lambda=0.4$, so that gradient updates remain stable and do not cause mid-training validation drops.
- **AC 1.1**: `TARGET_QWK_WEIGHT` and `SOFTMAX_QWK_TARGET` are set to `0.4` in Section 1.1.
- **AC 1.2**: `QWK_WARMUP_EPOCHS` is set to `15` epochs for full runs.
- **AC 1.3**: `PATIENCE_EARLY_STOP` is set to `15` epochs.

### User Story 2: Post-Hoc Screening Threshold Calibration (Priority: P1)
As a clinical researcher, I need each of the 5 output heads to be evaluated for referable DR screening via post-hoc threshold calibration on the Validation set at target specificities (95%, 90%, 85%, 80%), reporting true Test sensitivities and specificities against the UK NSC standard.
- **AC 2.1**: `calculate_referable_sensitivity_at_specificity` supports target specificities `(0.80, 0.85, 0.90, 0.95)` and returns operating thresholds $\tau_s$.
- **AC 2.2**: `calibrate_and_evaluate_screening` is implemented and executed for each head on the restored best checkpoint.
- **AC 2.3**: Calibrated screening metrics are stored in `head_ablation_results` and exported to summary CSV/JSON.

### User Story 3: Visualization & Audit Alignment (Priority: P2)
As an author, I need the publication visualization suite to display the calibrated screening operating points and ROC curves across all 5 heads.
- **AC 3.1**: `plot_fig7_referable_roc_curves` includes visual reference lines for UK NSC operating bounds.

---

## 3. Functional Requirements

- **FR-001**: Update Section 1.1 hyperparameters in `main_model_head_ablation (3).ipynb`: `TARGET_QWK_WEIGHT = 0.4`, `SOFTMAX_QWK_TARGET = 0.4`, `QWK_WARMUP_EPOCHS = 15 if not SMOKE_TEST else 1`, `PATIENCE_EARLY_STOP = 15`.
- **FR-002**: Update `calculate_referable_sensitivity_at_specificity` in Section 4.1 to evaluate at `target_specificities = (0.80, 0.85, 0.90, 0.95)` and record operating thresholds $\tau$.
- **FR-003**: Implement `calibrate_and_evaluate_screening(val_probs, val_targets, test_probs, test_targets, target_specificities=(0.80, 0.85, 0.90, 0.95))` in Section 4.1.
- **FR-004**: Integrate post-hoc calibration into the Section 5 execution loop (`DRHeadTrainer.fit` / round evaluation) and store results in `head_ablation_results`.
- **FR-005**: Maintain 100% backward compatibility, strict English documentation, and zero regression under Constitution v1.1.0.

---

## 4. Verification Plan

- **AST Syntax Validation**: Ensure `main_model_head_ablation (3).ipynb` compiles cleanly with zero syntax errors using `ast.parse`.
- **Parity Check**: Confirm that all 5 output heads run under identical ResNet-50 configurations with standardized loss warmup and calibration metrics.
