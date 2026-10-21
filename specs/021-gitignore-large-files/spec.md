# Feature Specification: 021-gitignore-large-files

**Feature Branch**: `021-gitignore-large-files`  
**Created**: 2026-10-21  
**Status**: Draft  
**Input**: User description: `"/speckit-specify giúp tôi tìm tất cả các file, dạng file mà nó đang có dung lương hơn 100mb để cho vào gitignore đi"`

---

## 1. Executive Summary & Objective

The objective of this specification is to establish a comprehensive repository hygiene standard preventing files exceeding GitHub's 100MB hard limit (and optionally 50MB warning threshold) from entering Git tracking. 

During the automated audit of the repository, **40 files exceeding 100MB** (ranging from 104MB to 33.7GB) were identified, spanning 5 file types: `.zip`, `.pkl`, `.pth`, `.keras`, and `.h5`. Crucially, **26 of these files (>100MB) are currently tracked in Git's index** under `data/models/` and `data/processed/`, which would cause GitHub remote push rejections (`GH001: Large files detected`).

This feature defines the rules for `.gitignore`, untracking existing indexed large artifacts while preserving local files on disk, and establishing verification checks.

---

## 2. User Scenarios & Testing *(mandatory)*

### User Story 1 - Comprehensive .gitignore Pattern Coverage for Large Binary Artifacts (Priority: P1)

As a researcher or developer committing code to this repository, I need `.gitignore` to explicitly ignore all binary artifact types that regularly exceed 100MB (model checkpoints, pickled dataset caches, raw compressed archives), so that `git add .` or automated staging scripts never stage large binaries accidentally.

**Why this priority**: GitHub enforces an absolute file size limit of 100MB. Staging any file >100MB blocks `git push` to remote origins.

**Independent Test**: Staging test dummy files (e.g., dummy `.pkl`, `.keras`, `.pth`, `.zip`, `.h5`) verifies they are ignored by Git.

**Acceptance Scenarios**:
1. **Given** any file with extensions `.pkl`, `.pickle`, `.pth`, `.pt`, `.ckpt`, `.keras`, `.h5`, `.zip`, `.tar.gz`, `*.bin`, **When** `git status` or `git add` is executed, **Then** Git ignores them by default.
2. **Given** `.gitignore`, **When** inspected, **Then** `*.pkl` and `*.pickle` are explicitly declared in the large artifacts section (previously omitted).

---

### User Story 2 - Safe Untracking of Currently Tracked Files Exceeding 100MB (Priority: P1)

As a repository maintainer, I need to remove currently tracked large files (>100MB) from Git's staging index (`git rm --cached`) without deleting the actual weights, datasets, and archives from the local filesystem, so that local training and inference pipelines remain completely uninterrupted while the Git index is cleaned.

**Why this priority**: Adding patterns to `.gitignore` does not untrack files that were already committed or staged in Git. The 26 tracked files must be safely detached from Git tracking.

**Independent Test**: Run `git ls-files` and verify zero tracked files have size >100MB, while `Test-Path` confirms all 40 files remain physically present on disk.

**Acceptance Scenarios**:
1. **Given** 26 tracked files >100MB under `data/models/` and `data/processed/`, **When** untracking is executed via `git rm --cached`, **Then** the files are staged for removal from Git tracking.
2. **Given** the local disk, **When** untracking finishes, **Then** all physical files remain intact at their original paths with zero data loss.

---

### User Story 3 - Repository Pre-Commit / CI Verification of File Sizes (Priority: P2)

As a collaborative research team member, I need an automated verification command or pre-commit check that scans the staging area for any file exceeding 100MB (or 50MB warning), so that accidental pushes of large files are caught before reaching CI/remote.

**Why this priority**: Prevents future regressions and keeps repository clone times fast.

**Independent Test**: Run verification script on repository; it reports pass/fail with exact sizes.

**Acceptance Scenarios**:
1. **Given** a staged file >100MB, **When** running verification, **Then** it halts with a clear error listing the file and its size.
2. **Given** all files comply with `<100MB`, **When** running verification, **Then** it passes with green exit code 0.

---

## 3. Edge Cases

- **Existing Git History**: Files >100MB already committed in past historical commits will remain in `.git/` packfiles unless a history rewrite (`git-filter-repo` / BFG) is performed. (Specification addresses whether index untracking is sufficient or history purge is requested).
- **Files between 50MB and 100MB**: GitHub issues a warning for files >50MB. There are currently 34 files between 50MB and 100MB (also `.pth`, `.pkl`, `.keras`, `.h5`).
- **Preservation of .gitkeep**: Empty directory keepers (like `data/raw/.gitkeep` and `data/processed/.gitkeep`) must NOT be ignored or untracked so repository folder skeleton remains reproducible for fresh clones.

---

## 4. Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: `.gitignore` MUST include explicit global wildcard patterns for all discovered large file formats: `*.pkl`, `*.pickle`, `*.pth`, `*.pt`, `*.ckpt`, `*.h5`, `*.keras`, `*.zip`, `*.tar.gz`, `*.tgz`, `*.7z`, `*.rar`, `*.bin`, `*.onnx`.
- **FR-002**: `.gitignore` MUST maintain directory-level exclusions for heavy runtime data folders (`checkpoints/`, `data/models/`, `data/raw/*`, `data/processed/*`, `cache/`, `outputs/`, `wandb/`), while preserving `.gitkeep` files (`!data/raw/.gitkeep`, `!data/processed/.gitkeep`).
- **FR-003**: The execution plan MUST safely untrack all 26 currently indexed files >100MB via `git rm --cached` without touching physical filesystem contents.
- **FR-004**: System MUST verify that zero physical files are deleted or modified on disk during the untracking process.
- **FR-005**: [NEEDS CLARIFICATION: Should files between 50MB and 100MB also be untracked from Git index to avoid GitHub's 50MB warnings?]
- **FR-006**: [NEEDS CLARIFICATION: Should Git history rewrite (e.g. `git filter-repo`) be applied to purge large files from historical commits, or is index untracking (`git rm --cached`) sufficient for future commits?]
- **FR-007**: [NEEDS CLARIFICATION: Does the team plan to use Git LFS (Large File Storage) for model checkpoints and sample datasets, or keep all model weights strictly local / on external cloud storage (Drive, HuggingFace)?]

### Key Entities *(Repository Hygiene)*

- **Large File Inventory**: List of 40 local files >100MB (2 `.zip`, 6 `.pkl`, 12 `.pth`, 19 `.keras`, 1 `.h5`).
- **Tracked Large File Set**: Subset of 26 files currently registered in Git index (`100644`).
- **Ignore Ruleset**: Standardized patterns in `.gitignore`.

---

## 5. Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of discovered file extensions (`.zip`, `.pkl`, `.pth`, `.keras`, `.h5`) are covered by `.gitignore`.
- **SC-002**: `git ls-files` check confirms 0 tracked files exceeding 100MB in the current Git index.
- **SC-003**: Zero byte loss on disk — 100% of existing model weights and datasets remain at their exact filesystem paths.
- **SC-004**: Staging any file >100MB is blocked or ignored by default.

---

## 6. Assumptions

- Local training scripts and Jupyter notebooks reference local paths (`data/models/`, `checkpoints/`) which will remain unchanged.
- Team members download raw datasets (`data/raw/`) from external sources (e.g., Kaggle, Drive) rather than Git.
- Operating system is Windows PowerShell.
