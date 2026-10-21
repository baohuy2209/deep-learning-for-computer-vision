# Feature Specification: Deconstructing Domain Shift in Diabetic Retinopathy Grading — Scientific & Experimental Synthesis

**Feature Branch / Directory**: `specs/014-synthesize-experiment-docs`  
**Created**: 2026-09-16  
**Status**: Ready for Planning & Execution  
**Target Manuscript File**: [`documents/05_manuscript_draft_sections.md`](file:///d:/Github/deep-learning-for-computer-vision/documents/05_manuscript_draft_sections.md)  
**Target Reference Notebooks**:
- [`notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb)
- [`notebooks/diabetic_retinopathy/pipeline/model/main_model (6).ipynb`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/model/main_model%20(6).ipynb)
- [`notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation (3).ipynb`](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation%20(3).ipynb)
**Advisor Guidance Documents**:
- [`documents/HD_1.pdf`](file:///d:/Github/deep-learning-for-computer-vision/documents/HD_1.pdf), [`HD_2.pdf`](file:///d:/Github/deep-learning-for-computer-vision/documents/HD_2.pdf), [`HD_3.pdf`](file:///d:/Github/deep-learning-for-computer-vision/documents/HD_3.pdf), [`HD_4.pdf`](file:///d:/Github/deep-learning-for-computer-vision/documents/HD_4.pdf)

---

## 🎯 Overarching Research Vision & Epic User Story

> **Epic User Story (The Core Scientific Narrative):**  
> *"As an ophthalmic AI researcher and clinician, I want to decompose cross-domain performance degradation in automated 5-class diabetic retinopathy grading across six international clinical cohorts ($N=55{,}570$ images from 35{,}474 patients), so that instead of treating domain shift as an opaque aggregate penalty, I can pinpoint which specific clinical disease grades collapse (Grade 1 vs Grade 3) and attribute their survival to distinct clinical-architectural mechanisms (Native-Resolution Local MIL, Global Relational Quadrant Tokens, and Ordinal Cumulative Link Modeling)."*

---

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Cross-Cohort Domain Shift Deconstruction & 6-Fold LODO Benchmark (Priority: P1)

**User Journey**:  
As a clinical screening program lead, I want to evaluate model generalizability using a sterile patient-level 6-Fold Leave-One-Dataset-Out (LODO) benchmark across APTOS 2019, DDR, DeepDRiD, IDRiD, Messidor-2, and EyePACS, so that out-of-domain transfer penalties ($\Delta_{\mathrm{QWK}}$, $\Delta_{\mathrm{Sens}}$) are quantified across disparate camera optics, sensor resolutions, and patient populations without target-domain fine-tuning.

**Why this priority**: Cross-domain generalization collapse across heterogeneous clinical sites is the central clinical bottleneck of automated screening and the primary research question of this work.

**Independent Test**: Can be verified by auditing the 6-fold LODO evaluation matrix where each dataset serves as an out-of-domain test cohort while the remaining 5 form the source training pool, confirming sterile patient isolation via `dataset_key::patient_id` composite keys.

**Acceptance Scenarios**:
1. **Given** 6 multi-center cohorts ($55,570$ images), **When** performing LODO evaluation, **Then** the generalization gap $\Delta_{\mathrm{QWK}} = \mathrm{QWK}_{\mathrm{In\text{-}Domain}} - \mathrm{QWK}_{\mathrm{OOD}}$ is reported for each fold alongside macro-averaged transfer degradation.
2. **Given** four architecture variants ($V_0 \to V_1 \to V_2 \to V_3$), **When** comparing baseline versus proposed systems, **Then** the ablation decouples the independent contributions of the input representation (backbone) versus the decision layer (head).

---

### User Story 2 - Micro-Lesion Resolution Survival via Native-Resolution Local MIL ($H_1$) (Priority: P1)

**User Journey**:  
As an ophthalmologist reviewing Mild NPDR (Grade 1), I want the system to preserve native optical resolution using overlapping sliding-window Multiple Instance Learning ($128 \times 128$ patches, stride 96) and dual pooling (Max-Pooling + Gated Attention), so that sub-$15\mathrm{px}$ microaneurysms are protected from downsampling pixel-loss and Grade 1 sensitivity is preserved out-of-domain.

**Why this priority**: Standard global downsampling physically annihilates microaneurysms, causing severe sensitivity collapse on Grade 1 (the sole defining lesion of early disease).

**Independent Test**: Can be verified by comparing Grade 1 sensitivity (recall) between the global-only baseline ($V_0$) and the dual-branch MIL system ($V_2$ and $V_3$), confirming that the local stream rescues Grade 1 sensitivity under external domain shift.

**Acceptance Scenarios**:
1. **Given** a high-resolution fundus photograph ($3000 \times 2000$), **When** patches are extracted with stride 96 and FOV filtering $\ge 0.50$, **Then** dual pooling isolates focal microaneurysms via 1D Max-Pooling while aggregating spatial lesion density via Gated Attention.
2. **Given** external out-of-domain test sets, **When** evaluating Grade 1 recall, **Then** the dual-branch model demonstrates a statistically significant preservation in Grade 1 sensitivity compared to global downsampled baselines.

---

### User Story 3 - Multi-Quadrant Relational Context for Severe NPDR 4-2-1 Rule ($H_2$) (Priority: P1)

**User Journey**:  
As a retinal specialist diagnosing Severe NPDR (Grade 3), I want the system to encode spatial relational dependencies across anatomical quadrants using Non-Local self-attention and a 4-Quadrant Transformer Token Aggregator, so that the clinical 4-2-1 diagnostic rule is reliably verified across heterogeneous field-of-view geometries.

**Why this priority**: Grade 3 diagnosis requires verifying $>20$ intraretinal hemorrhages in each of 4 quadrants, venous beading in $\ge 2$ quadrants, or IRMA in $\ge 1$ quadrant. Standard convolutional receptive fields fail to capture these cross-quadrant dependencies.

**Independent Test**: Can be verified by comparing Grade 3 sensitivity (recall) with and without the Quadrant Token Aggregator and Non-Local block, confirming that cross-quadrant evidence aggregation stabilizes Grade 3 detection under external test shifts.

**Acceptance Scenarios**:
1. **Given** global spatial feature maps ($16 \times 16 \times 2048$), **When** partitioned into Superior-Temporal, Superior-Nasal, Inferior-Temporal, and Inferior-Nasal quadrants, **Then** a 1-layer Transformer encoder models inter-quadrant lesion relationships.
2. **Given** out-of-domain evaluation on unseen cohorts, **When** evaluating Grade 3 recall, **Then** the quadrant relational mechanism prevents sensitivity drop on Severe NPDR cases.

---

### User Story 4 - Ordinal Geometry Preservation & Monotonic Rank Consistency ($H_3$) (Priority: P1)

**User Journey**:  
As a clinical safety auditor, I want the system to predict disease severity along an ordered latent continuum partitioned by strictly monotonic cutpoints using an asymmetric Complementary Log-Log ($\text{clog-log}$) Cumulative Link Model, so that catastrophic cross-domain rank-inversion violations ($P(y > k) < P(y > k+1)$) are strictly eliminated ($0.0\%$) and out-of-domain decision boundary fragmentation is suppressed.

**Why this priority**: Treating ordinal stages as unordered nominal classes allows decision hyperplanes to cross, producing clinically dangerous predictions and accelerating out-of-domain degradation.

**Independent Test**: Can be verified by auditing pre-activation rank exceedance probabilities $\Delta_{i,k} = P(y_i > k+1) - P(y_i > k)$, confirming that CLM achieves strictly $0.0\%$ monotonicity violations while Softmax and CORAL exhibit substantial violations.

**Acceptance Scenarios**:
1. **Given** the 5-head ablation benchmark on a standardized ResNet-50 backbone, **When** evaluating rank violations on raw logits, **Then** CLM guarantees $0.0\%$ violations by parameterizing cutpoints with positive softplus increments.
2. **Given** external LODO evaluation, **When** measuring the out-of-domain transfer advantage of CLM over Softmax, **Then** the performance preservation gap is greater out-of-domain than in-domain, proving that ordinal inductive bias acts as an out-of-domain regularizer.

---

### User Story 5 - Post-Hoc Screening Triage Calibration at 95% Specificity (Priority: P2)

**User Journey**:  
As a telemedicine screening operator, I want to calibrate the binary referable DR decision threshold on validation ROC continuous probabilities $P(Y \ge 2)$ rather than relying on nominal 5-class $\text{argmax}$, so that the system operates at fixed clinical specificity targets ($85\%$--$95\%$) while achieving $\ge 80\%$ sensitivity in conformance with international screening standards (UK NSC).

**Why this priority**: The $\text{argmax}$ of a multi-class model produces an arbitrary operating threshold clustering at high specificity but depressed sensitivity ($64.8\%$), whereas the underlying representations possess high discrimination ($\text{Ref-AUC} = 0.9063$).

**Independent Test**: Can be verified by checking that operating thresholds $\tau$ are fitted strictly on the validation set ROC curve and evaluated zero-shot on the held-out test set at $95\%$, $90\%$, $85\%$, and $80\%$ target specificity points.

**Acceptance Scenarios**:
1. **Given** continuous referable risk scores $P(Y \ge 2)$, **When** calibrated on validation ROC to achieve $85\%$ specificity, **Then** test sensitivity achieves $\ge 80\%$ without model retraining.
2. **Given** the UK NSC benchmark criteria, **When** reporting screening performance, **Then** threshold calibration is explicitly documented as a decoupled post-hoc clinical tuning stage.

---

### User Story 6 - Quantitative Explainability & Anatomic Lesion Alignment (Priority: P2)

**User Journey**:  
As a regulatory compliance reviewer, I want to quantitatively validate model attention heatmaps against 81 gold-standard IDRiD expert lesion segmentation masks using Dual-Crop FOV registration and Pointing Game accuracy, so that the model's predictions are proven to localize true microvascular pathology rather than optical artifacts or camera glare.

**Why this priority**: Grounding AI decisions on verified pathology (microaneurysms, hemorrhages, hard/soft exudates) is required for clinical interpretability and trust.

**Independent Test**: Can be verified by evaluating Pointing Game accuracy and area-matched saliency recall on IDRiD masks, comparing Attention-MIL saliency against global Layer 4 Grad-CAM and an empirical random chance baseline ($1.5\%$).

**Acceptance Scenarios**:
1. **Given** the 81 IDRiD lesion masks, **When** performing Dual-Crop bounding box registration, **Then** both fundus image and binary masks are registered with identical optical coordinate framing.
2. **Given** fine-grained Attention-MIL saliency maps, **When** evaluating Pointing Game accuracy, **Then** Attention-MIL achieves $>12\times$ empirical random chance, significantly outperforming coarse global Grad-CAM.

---

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST implement a sterile patient-level 6-Fold Leave-One-Dataset-Out (LODO) benchmark across APTOS 2019, DDR, DeepDRiD, IDRiD, Messidor-2, and EyePACS ($N=55{,}570$).
- **FR-002**: The pipeline MUST extract native-resolution local patches ($128 \times 128$, stride 96) and aggregate them via Dual Pooling (Max-Pooling + Gated Attention) to test Hypothesis $H_1$.
- **FR-003**: The global stream MUST incorporate 2D Non-Local self-attention and a 4-Quadrant Transformer Token Aggregator to test Hypothesis $H_2$.
- **FR-004**: The output decision layer MUST implement a Cumulative Link Model (CLM) with asymmetric Complementary Log-Log ($\text{clog-log}$) link and softplus-constrained monotonic cutpoints to test Hypothesis $H_3$.
- **FR-005**: The training loss MUST combine interval negative log-likelihood with continuous differentiable QWK loss under a 15-epoch linear warmup schedule ($\lambda = 0.4$).
- **FR-006**: The evaluation suite MUST report per-grade sensitivity (specifically Grade 1 and Grade 3), within-1 accuracy, and rank monotonicity violations on raw logits.
- **FR-007**: The screening triage evaluation MUST decouple binary referable DR decisions from multi-class argmax via post-hoc validation ROC calibration at $95\%$, $90\%$, $85\%$, and $80\%$ target specificity.
- **FR-008**: The XAI module MUST benchmark Attention-MIL saliency and Grad-CAM against 81 real IDRiD lesion masks via Dual-Crop FOV registration.
- **FR-009**: The research manuscript in `documents/05_manuscript_draft_sections.md` MUST contain full academic LaTeX text for Abstract, Introduction, Related Work, Methodology, Experimental Setup, Results with empirical placeholders $[X], [Y], [Z]$, and Conclusion.
- **FR-010**: All citations in the manuscript MUST link to verified sources from `documents/nguồn trích dẫn.xlsx` and advisor directives (HD_1 to HD_4).

---

## Success Criteria *(mandatory)*

- **SC-001**: Complete alignment between the clinical motivation (ICDRS 4-2-1 rule, microaneurysms, ordinal staging) and the three architectural mechanisms ($H_1, H_2, H_3$).
- **SC-002**: Zero broken cross-references between the manuscript draft (`05_manuscript_draft_sections.md`), technical monographs (`01` to `04`), and experimental notebooks.
- **SC-003**: Strict mathematical parameterization of CLM_QWK ensuring $0.0\%$ rank monotonicity violations by construction.
- **SC-004**: Results tables in the manuscript provide standardized placeholders $[X], [Y], [Z]$ structured for seamless insertion of official GPU cluster numbers post-execution.
