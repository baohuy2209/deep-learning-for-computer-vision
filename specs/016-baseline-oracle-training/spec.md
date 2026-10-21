# Feature Specification: Baseline Oracle Training for 6 Fundus Cohorts

**Feature Branch**: `016-baseline-oracle-training`  
**Created**: 2026-09-22  
**Status**: Draft / Brainstorming  
**Input**: User description: "/speckit-specify @notebooks/diabetic_retinopathy/pipeline/model/main_model (6).ipynb @notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb @notebooks/diabetic_retinopathy/pipeline/merge-dataset/merge-dataset.ipynb @documents/R_v0.pdf baseline oracle 6 datasets /clean-code /brainstorm"

---

## 1. Context & Academic Rationale (Addressing Advisor Feedback R_v0.pdf)

### 1.1 Problem Statement
In the multi-center 6-fold Leave-One-Dataset-Out (LODO) benchmark, Table 3 in the manuscript reported $\Delta = QWK_{\text{ID}} - QWK_{\text{OOD}}$, where in 4 out of 6 folds, $QWK_{\text{OOD}} > QWK_{\text{ID}}$ (e.g., EyePACS exhibited a counter-intuitive gap of $+0.30$).

As demonstrated in `documents/R_v0.pdf` (Section A1):
- $QWK_{\text{ID}}$ was computed on a multi-center pool of 5 training cohorts with a blended class marginal prior $P_{\text{pool}}(Y)$.
- $QWK_{\text{OOD}}$ was computed on the held-out single cohort with a vastly distinct clinical prior $P_{\text{target}}(Y)$ (e.g., EyePACS has $73.5\%$ Grade 0, whereas IDRiD has $30.0\%$ Grade 3+4).
- Because Quadratic Weighted Kappa (QWK) is inherently sensitive to marginal class distributions, the reported $\Delta$ conflated **domain shift** with **class prior shift** and **cohort intrinsic grading difficulty**.

### 1.2 Proposed Solution: In-Domain Baseline "Oracle"
To measure pure domain shift, the advisor recommends implementing Option 3:
$$\Delta_{\text{transfer}}(\text{target}) = QWK_{\text{oracle}}(\text{target}) - QWK_{\text{OOD}}(\text{target})$$
Where:
- $QWK_{\text{oracle}}(\text{target})$ is achieved by training the exact **Champion Architecture (`main_model`)** exclusively in-domain on the target cohort itself.
- Both $QWK_{\text{oracle}}$ and $QWK_{\text{OOD}}$ are evaluated on the **EXACT SAME HELD-OUT TEST SPLIT** with the **EXACT SAME CLASS PRIOR** $P_{\text{target}}(Y)$.
- $\Delta_{\text{transfer}} \ge 0$ measures the authentic cross-domain performance gap attributable strictly to covariate / sensor shift.

---

## Clarifications

### Session 2026-09-22
- Q: Bạn muốn 6 file huấn luyện Baseline Oracle được tạo dưới định dạng tệp và môi trường thực thi nào? → A: Option A - 6 tệp Jupyter Notebook (`.ipynb`) độc lập đặt tại `notebooks/diabetic_retinopathy/pipeline/oracle/`, cấu hình sẵn cho môi trường Google Colab A100 (tích hợp cell mount Drive, A100 AMP, lưu checkpoint và vẽ biểu đồ QWK/Loss).
- Q: Bạn muốn thiết lập chiến lược phân chia tập dữ liệu (train / val / test) cho từng mô hình Oracle theo phương án nào? → A: Option A - Phân chia nội bộ 70% train / 15% val / 15% test theo bệnh nhân (`GroupShuffleSplit` trên `patient_id`, cố định `seed=42`) cho từng cohort. Cả mô hình Oracle và mô hình LODO OOD tương ứng đều được đánh giá trên cùng tập 15% test này để tính $\Delta_{\text{transfer}} = QWK_{\text{oracle}} - QWK_{\text{OOD}}$ công bằng trên cùng một phân bố lớp.
- Q: Bạn muốn cấu hình số Epoch, Batch Size và Early Stopping cho 6 mô hình Oracle theo phương án nào để thích ứng với chênh lệch quy mô mẫu (từ 516 ảnh của IDRiD đến 35,126 ảnh của EyePACS)? → A: Option B (tùy chỉnh) - Cố định đồng nhất cho cả 6 cohort: Tối đa 40 epochs, batch size 16, và Early Stopping patience là 15 epochs dựa trên Validation QWK.
- Q: Bạn muốn xử lý tình trạng mất cân bằng lớp trầm trọng của từng cohort (như EyePACS có 73.5% Grade 0, hay IDRiD có ít ca Grade 1) theo phương án nào trong 6 tệp Oracle? → A: Option A - Sử dụng Dynamic Class Weighting (nghịch đảo tần suất lớp) trong hàm mất mát kết hợp $\mathcal{L}_{\text{CLM}} + \lambda \mathcal{L}_{\text{QWK}}$, giữ nguyên phân bố dữ liệu tự nhiên của cohort mà không oversampling nhân tạo.
- Q: Bạn muốn thiết lập cơ chế lưu Checkpoint trên Google Drive và xuất số liệu đánh giá $\Delta_{\text{transfer}}$ trong từng Notebook theo phương án nào? → A: Option A - Tự động lưu checkpoint `oracle_{cohort}_best.pth` lên Google Drive; tự động nạp checkpoint LODO tương ứng để đánh giá đối đầu trên cùng tập test 15%; xuất tệp `oracle_{cohort}_results.json` và biểu đồ kiểm định chất lượng cao (300 DPI) phục vụ cập nhật Bảng 3 của bài báo.

---

## 2. User Scenarios & Testing *(mandatory)*

### User Story 1 - Train Individual Oracle Models for 6 Cohorts (Priority: P1)
As a researcher, I need 6 distinct, reproducible training entry points corresponding to each fundus cohort (APTOS 2019, DDR, DeepDRiD, IDRiD, Messidor-2, EyePACS) executing the exact champion `main_model` architecture, so that in-domain upper bounds ($QWK_{\text{oracle}}$) are established.

- **Why this priority**: Required to resolve the primary blocker in Section A1 of the paper review.
- **Independent Test**: Run training on a small cohort (e.g. IDRiD or Messidor-2), verify convergence of loss and output of checkpoint and test metrics.
- **Acceptance Criteria**:
  1. Given cohort metadata, when training executes, the model must converge and save `_best.pth` based on validation QWK.
  2. Zero patient-level data leakage between train, val, and test splits (`GroupShuffleSplit` on `patient_id`).

### User Story 2 - Compute Clean Transfer Gap Metric $\Delta_{\text{transfer}}$ (Priority: P2)
As an author, I need each oracle training run to evaluate on the exact designated held-out test split of that cohort and compute $\Delta_{\text{transfer}} = QWK_{\text{oracle}} - QWK_{\text{OOD}}$, producing verified data for Table 3.

- **Why this priority**: Replaces flawed OOD-ID comparisons with clean domain-shift metrics.
- **Independent Test**: Pass predictions from the OOD LODO model and Oracle model on the same test set and verify $\Delta_{\text{transfer}}$ calculation.
- **Acceptance Criteria**:
  1. Test set evaluated by Oracle model matches the test partition used for OOD evaluation.
  2. Metrics calculated include: QWK, Within-1 Accuracy, Quadratic ECE, Referable AUC, and Referable Sensitivity @ 80/85/90/95% Specificity.

---

## 3. Architecture & Functional Requirements *(mandatory)*

### 3.1 Model Architecture Fidelity
- **FR-001**: Model MUST strictly implement the Champion Architecture from `main_model (6).ipynb` / `main_model_lodo.ipynb`:
  - **Global Branch**: ResNet-50 backbone with Green-Channel Ben Graham contrast enhancement ($512 \times 512$), Non-Local Block, CBAM attention, and 4 Quadrant Tokens.
  - **Local MIL Branch**: EfficientNet-B0 backbone extracting high-resolution patches ($128 \times 128$) with Attention MIL pooling and Max pooling.
  - **Fusion**: Multimodal 2-layer MLP with LayerNorm, Mish activation, and Dropout (0.3).
  - **Ordinal Head**: Cumulative Link Model (CLM) with continuous QWK loss ($\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CLM}} + \lambda \mathcal{L}_{\text{QWK}}$, with $\lambda$ warming up linearly to $0.4$).

### 3.2 Dataset & Split Handling
- **FR-002**: Splitting MUST strictly enforce patient-level grouping (`GroupShuffleSplit` on `patient_id` with dataset prefix, `seed=42`) partitioned as 70% Train, 15% Validation, and 15% Test. Both the in-domain Oracle model and the cross-domain LODO champion model MUST be evaluated on this exact identical 15% held-out test split.
- **FR-003**: Class weighting MUST be computed dynamically per cohort as inverse class frequency on the 70% training partition and supplied to the CLM + Continuous QWK loss function without synthetic oversampling, maintaining natural empirical prevalence.
- **FR-004**: Data pipelines must support mixed precision (`torch.amp` with `bfloat16`/`float16`) and robust checkpoint resumption (`_best.pth` and `_last.pth`). All 6 oracle pipelines MUST execute with a uniform training budget of 40 maximum epochs, batch size of 16, and an EarlyStopping patience of exactly 15 epochs monitoring Validation QWK.
- **FR-005**: Delivery artifacts MUST be 6 standalone, self-contained Jupyter Notebooks (`.ipynb`) located at `notebooks/diabetic_retinopathy/pipeline/oracle/` targeting Google Colab with GPU acceleration, Google Drive mounting, and inline metric/checkpoint tracking.
- **FR-006**: Each notebook MUST automate Google Drive checkpoint persistence (`/content/drive/MyDrive/checkpoints_oracle/oracle_{cohort}_best.pth`), automatically restore the corresponding cross-domain LODO checkpoint to evaluate head-to-head performance on the identical 15% test partition, export structured results to `oracle_{cohort}_results.json`, and render publication-grade comparison figures (300 DPI) for Table 3.

---

## 4. Success Criteria *(mandatory)*

- **SC-001**: 6 functional training pipelines generated for APTOS 2019, DDR, DeepDRiD, IDRiD, Messidor-2, and EyePACS.
- **SC-002**: $100\%$ zero patient-leakage verified across all 6 cohorts.
- **SC-003**: In-domain Oracle QWK ($QWK_{\text{oracle}}$) exceeds or strictly bounds the cross-domain $QWK_{\text{OOD}}$ on identical test partitions, yielding consistent non-negative or statistically sound $\Delta_{\text{transfer}}$.
- **SC-004**: Code adheres strictly to `/clean-code` standards: modular, self-documenting, no hardcoded magic paths, type-annotated.
