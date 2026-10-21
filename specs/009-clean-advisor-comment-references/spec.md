# Feature Specification: Clean Advisor Comment References

**Feature Branch**: `009-clean-advisor-comment-references`

**Created**: 2026-09-11

**Status**: Draft

**Input**: User description: "Rồi bây giờ nó có một vấn đề là bạn dò lại toàn bộ comment để bỏ mấy chỗ comment mà có HD_3 HD_4 đi trong notebooks/diabetic_retinopathy/pipeline/model/main_model.ipynb notebooks/diabetic_retinopathy/pipeline/model/main_model (4).ipynb notebooks/diabetic_retinopathy/pipeline/model/main_model (8).ipynb mà vẫn giữ nguyên logic có sẵn nhé (không được tự ý thay đổi logic )"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Sanitization of Advisor References from Code Comments (Priority: P1)

As a researcher preparing the experimental pipeline for submission and sharing, I want all external guidance markers (such as "HD_3", "HD_4", "HD_1", "HD_2", "Advisor") removed from code comments and markdown cells across all experiment notebooks so that the codebase appears clean, professional, self-contained, and publication-ready.

**Why this priority**: Directly requested by the user to sanitize the notebooks before running and presenting the work, while eliminating any internal meta-commentary.

**Independent Test**: Scan the three notebooks (`main_model.ipynb`, `main_model (4).ipynb`, `main_model (8).ipynb`) with a regex search for `HD_[0-9]` and advisor tokens; verify zero matches remain while all 92 cells parse without syntax error.

**Acceptance Scenarios**:

1. **Given** notebooks containing comments referencing `HD_3`, `HD_4`, or advisor notes, **When** comment sanitization is applied, **Then** all such references are reworded into objective technical descriptions or cleanly removed without altering line semantics.
2. **Given** sanitized notebooks, **When** validating execution logic, **Then** all tensor operations, model architectures, data loaders, training loops, and evaluation metrics remain 100% byte-for-byte functionally identical (Principle V: Zero-Regression).

---

### User Story 2 - Uniform Synchronization Across All 3 Notebooks (Priority: P2)

As a developer maintaining multiple execution environments, I want all three notebook files (`main_model.ipynb`, `main_model (4).ipynb`, `main_model (8).ipynb`) to remain perfectly synchronized in terms of both sanitized comments and execution code.

**Why this priority**: Avoids version drift or inconsistencies between different notebook copies.

**Independent Test**: Compare the code and comment diffs across the three notebooks and confirm identical AST structure and clean comments.

**Acceptance Scenarios**:

1. **Given** the three notebook files, **When** checked after sanitization, **Then** all three have identical cell counts (92 cells) and zero residual `HD_` comment artifacts.

---

### Edge Cases

- **Docstrings and print statements**: Any print statements printing `(HD_4)` or similar to stdout should have the string literal cleaned to standard objective scientific text (e.g., `(Screening Calibration)` instead of `(HD_4 Part 1)`).
- **Preservation of Code Logic**: No variable names, function signatures, default arguments, or tensor operations shall be touched.
- **Markdown Headers**: Section titles mentioning `(HD_4 ...)` should be renamed to clean technical headings (e.g., `## 4.1 Clinical Sensitivity at Fixed Specificity & Post-Hoc Calibration`).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST scan and identify all occurrences of `HD_1`, `HD_2`, `HD_3`, `HD_4`, and "Advisor" across code comments, docstrings, print statements, and markdown cells in `main_model.ipynb`, `main_model (4).ipynb`, and `main_model (8).ipynb`.
- **FR-002**: System MUST replace or reword each occurrence with professional, objective, publication-grade academic language without changing code structure.
- **FR-003**: System MUST NOT alter any algorithmic logic, variable names, hyperparameter values, model weights, or loss definitions.
- **FR-004**: System MUST ensure that all code cells continue to pass Python Abstract Syntax Tree (`ast.parse`) validation with zero syntax errors.
- **FR-005**: System MUST maintain full synchronization across all three target notebook files.

### Key Entities

- **Target Notebooks**:
  - `notebooks/diabetic_retinopathy/pipeline/model/main_model (4).ipynb`
  - `notebooks/diabetic_retinopathy/pipeline/model/main_model (8).ipynb`
  - `notebooks/diabetic_retinopathy/pipeline/model/main_model.ipynb`
- **Sanitized Locations**:
  - Cell 10: Loss stabilization and linear warmup comments.
  - Cell 25: DataLoader collate configuration and batch size comments.
  - Cell 37 & 40: Attention comments.
  - Cell 42: Attention-MIL saliency weight retention comment.
  - Cell 64: Sensitivity at fixed specificity and calibration comments / print statements.
  - Cell 71: Methodological evaluation comment.
  - Cell 77: Dual-crop alignment comments and docstrings.
  - Cell 78: Saliency and overhauled XAI metric suite comments, docstrings, and print statements.
  - Cell 81: Screening threshold calibration comments and print statements.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 0 occurrences of `HD_[0-9]` or advisor markers remain in any of the three notebook files.
- **SC-002**: 100% of the 53 code cells pass Python AST validation across all 3 notebooks.
- **SC-003**: 100% of the 16 core pipeline compliance checks remain passing (zero regression).
- **SC-004**: Execution logic, model weights, hyperparameter values, and data flows are completely unchanged.

## Assumptions

- Removing advisor/HD references is purely cosmetic and text-sanitization; no functional code behavior depends on the content of comments or user-facing print messages.
- The user intends to keep the underlying methodology (such as dual-crop FOV alignment, 15-epoch warmup, post-hoc calibration) intact.
