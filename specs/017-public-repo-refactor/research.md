# Technical Research & Decisions: Public Academic Repository Refactor

**Feature Branch**: `017-public-repo-refactor`
**Date**: 2026-09-23
**Status**: Completed

---

## 1. Configuration Architecture: Dataclasses vs. YAML

### Decision
Implement configuration exclusively using standard library Python `@dataclass` objects in `src/drdg/config.py`, with an extensible `argparse` CLI binding layer. Prohibit all YAML, JSON, or Hydra configuration files.

### Rationale
- **Constraint Compliance**: Section 5 of the project specification explicitly forbids YAML/Hydra configuration.
- **Type Safety & IDE Support**: Python dataclasses provide static type hints, default value checking, immutability options (`frozen=False` for overrides), and auto-completion in standard IDEs without third-party schema plugins.
- **Simplicity & Dependency Minimization**: Avoids external parsing libraries (`pyyaml`, `hydra-core`, `omegaconf`), keeping the environment lightweight and reproducible across local workstations, clusters, and Google Colab.
- **Hierarchical Typing**:
  - `DataConfig`: Raw and processed paths, cohort keys, image resolution, batch size, workers.
  - `ModelConfig`: Variant selector (`v0`..`v3`), head type, pretrained backbone flag, feature dimensions.
  - `LocalMILConfig`: Patch size, stride, FOV threshold, max bag size (train vs eval).
  - `TrainingConfig`: Epochs, optimizer, learning rate, weight decay, scheduler, warmup epochs, loss weights, mixed precision (`amp_mode`).
  - `LODOConfig`: Held-out target cohort, all-folds flag, checkpoint resume path.
  - `XAIConfig`: Saliency method, target layer, IDRiD mask paths, evaluation metrics.

### Alternatives Considered
- **Hydra / OmegaConf**: Rejected due to explicit prohibition, high learning curve for reviewers, and complex composition overrides.
- **Pydantic**: Rejected because runtime validation overhead is unnecessary when standard library dataclasses with CLI parsing provide full type-safety without external dependencies.
- **Single monolithic dict**: Rejected due to lack of type checking, prone to silent key errors and unversioned defaults.

---

## 2. Package Architecture & Import Namespace

### Decision
Use a standard `src`-layout packaging structure (`src/drdg/`) with `pyproject.toml` supporting editable installation (`pip install -e .`).

### Rationale
- Eliminates all `sys.path.append(...)` and `sys.path.insert(0, ...)` hacks currently present in the notebooks.
- Allows both automated CLI scripts (`scripts/run_lodo.py`) and demonstration notebooks (`notebooks/10_v0_global_baseline.ipynb`) to import cleanly:
  ```python
  from drdg.config import TrainingConfig
  from drdg.models.factory import build_variant
  from drdg.training.trainer import DRTrainer
  ```
- Ensures non-interactive execution: no code executes at import time across all modules.

### Alternatives Considered
- **Flat repository layout (`drdg/` in root)**: Rejected because running pytest or python from different working directories can import uninstalled or stale local directories rather than the installed package.
- **Sub-packages per experiment**: Rejected as over-engineering; a single cohesive `drdg` package with submodules (`data`, `models`, `training`, `evaluation`, `explainability`, `visualization`, `utils`) matches standard academic libraries (e.g., torchvision, mmcv).

---

## 3. Data Ingestion, Harmonization & Leakage Prevention

### Decision
Standardize the ingestion of all six cohorts into a single master metadata schema, enforce dataset-namespaced patient IDs (`<dataset_key>::<orig_patient_id>`), and generate deterministic 6-fold LODO splits stored as immutable CSV files.

### Rationale
- **Patient Identifier Namespaces**: Different cohorts use overlapping numeric IDs (e.g., patient `001` exists in both IDRiD and DeepDRiD). Namespacing (`idrid::001`, `deepdrid::001`) eliminates cross-cohort patient collision.
- **Zero-Leakage Invariants**: Fundus photographs often include bilateral images (left and right eyes) or longitudinal visits. All splits (70/15/15 in-domain and 6-fold LODO) must partition at the patient level:
  $$\text{Train Patients} \cap \text{Val Patients} = \emptyset, \quad \text{Train Patients} \cap \text{Test Patients} = \emptyset, \quad \text{Val Patients} \cap \text{Test Patients} = \emptyset$$
- **Prevalence Preservation**: Target evaluation sets in LODO must retain 100% natural clinical prevalence (no artificial balancing or downsampling) to evaluate authentic clinical screening utility. Training pools use quota-based class balancing strictly on the training partition.
- **Download Automation vs. Third-Party Constraints**:
  - Download scripts provided: DDR, DeepDRiD, IDRiD, IDRiD segmentation, Messidor-2.
  - Manual acquisition documented: EyePACS (Kaggle 2015 Diabetic Retinopathy Detection requires third-party Kaggle competition agreement; automated re-distribution is legally restricted).

### Alternatives Considered
- **Image-level random splitting**: Strictly rejected due to severe patient leakage across bilateral fundus pairs.
- **On-the-fly random LODO partitioning**: Rejected because reproducibility across research groups requires bitwise identical, pre-computed split files.

---

## 4. Model Architecture & Component Decoupling

### Decision
Deconstruct the monolithic dual-branch implementation into isolated, composable modules:
1. `attention/`: Self-contained modules for CBAM, Non-Local 2D attention, and Quadrant Token Aggregation.
2. `streams/`:
   - `GlobalContextStream`: ResNet-50 backbone, stage-4 attention blocks, global pooling, and projection to $D=512$.
   - `LocalMILBranch`: ResNet-50 patch backbone, patch projection, and Gated Attention Pooling (`GatedAttentionPool`) producing bag representation $D=512$.
3. `fusion/`: DualBranchFusion layer combining global and local features into unified latent representation $D=1024$.
4. `heads/`: Standardized head interface exposing `forward(x) -> dict` returning `logits`, `probabilities`, and `severity`.
5. `variants/`: Factory builders composing the exact architectural stages:
   - `V0`: Global Context ResNet-50 baseline (no local MIL).
   - `V1`: V0 + Local MIL stream with dynamic bag collation.
   - `V2`: V1 + Global relational context package (CBAM, Non-Local, Quadrant Tokens).
   - `V3`: V2 + Cumulative Link Model (CLM) ordinal head.

### Rationale
- Guarantees 100% code reuse between standalone variant experiments, full LODO benchmarks, and the 5-head controlled ablation study.
- Preserves exact tensor shapes and layer names for compatibility with existing PyTorch checkpoint state dicts.

---

## 5. Output Heads & Ordinal Formulations

### Decision
Implement the five required output heads under a unified contract on standardized ResNet-50 representations:
1. `Softmax_CE`: Standard linear projection to 5 logits trained with categorical Cross-Entropy.
2. `Softmax_QWK`: Linear projection to 5 logits trained with Continuous Quadratic Weighted Kappa Loss (`ContinuousQWKLoss`).
3. `CORAL` (Consistent Rank Logits): Single linear projection scalar $\theta^T x$ with 4 independent rank thresholds:
   $$P(Y > k \mid x) = \sigma(\theta^T x + b_k), \quad k \in \{0, 1, 2, 3\}$$
4. `CORN` (Conditional Ordinal Neural Network): 4 binary classifiers predicting conditional progression probabilities:
   $$P(Y > k \mid Y \ge k, x)$$
5. `CLM_QWK` (Cumulative Link Model with QWK Loss): Single latent linear score $s(x) = w^T x$ with strictly monotonic cutpoints:
   $$-\infty = b_0 < b_1 < b_2 < b_3 < b_4 < b_5 = +\infty$$
   Cumulative probabilities:
   $$P(Y \le k \mid x) = \sigma(b_k - s(x))$$
   Class probabilities:
   $$P(Y = k \mid x) = P(Y \le k \mid x) - P(Y \le k-1 \mid x)$$
   Cutpoint monotonicity enforced via positive parameter reparameterization:
   $$b_k = b_1 + \sum_{j=2}^k \exp(d_j) \quad \text{or} \quad \text{softplus}(d_j)$$

### Rationale
- Standardizes feature extraction ($D=2048$ from ResNet-50 backbone) so that head performance variations isolate the loss/formulation effect from representation capacity.
- Mathematically guarantees valid probability distributions ($\sum_{k=0}^4 P(Y=k) = 1$, $P(Y=k) \ge 0$) without post-hoc clipping.

---

## 6. Metric Separation & Evaluation Protocol

### Decision
Enforce a non-negotiable tripartite metric hierarchy in all result schemas and logging:
1. `source_validation`: Metrics computed on the held-in validation partition during training, used solely for model selection (checkpoint saving) and freezing decision thresholds.
2. `source_test`: Benchmark metrics evaluated on held-in test partitions (when running standard in-domain experiments).
3. `target_test`: Zero-shot out-of-distribution evaluation on 100% of the held-out target cohort in LODO.

### Rationale
- Fixes the critical bug in earlier research notebooks where validation metrics (`in_domain_val_qwk`) were labeled or reported as in-domain test results.
- Guarantees that target labels NEVER influence model selection or calibration.

---

## 7. Referable DR (RDR) Screening Calibration

### Decision
Referable Diabetic Retinopathy is defined as $\text{Grade} \ge 2$ ($G_2, G_3, G_4$). The screening threshold $\tau^*$ is selected strictly on `source_validation` probabilities:
$$\tau^* = \arg\min_{\tau} \left| \text{Specificity}_{\text{val}}(\tau) - 0.95 \right|$$
This threshold $\tau^*$ is frozen and applied to `target_test`:
$$\hat{y}_{\text{RDR}} = \mathbb{I}\left(P(Y \ge 2 \mid x) \ge \tau^*\right)$$
Target metrics report:
- RDR AUROC (threshold-independent).
- RDR Sensitivity at the fixed $\tau^*$ ($\ge 95\%$ source-calibrated specificity).

### Rationale
- Adheres to clinical tele-screening standards (e.g., British Diabetic Association, FDA screening guidelines) where $\ge 95\%$ specificity is mandated to prevent overwhelming secondary eye care clinics.
- Eliminates target leakage and removes all hardcoded "expected sensitivity" narrative defaults.

---

## 8. Quantitative Explainability (XAI) Protocol

### Decision
Implement spatial localization evaluation using:
1. **Global Branch Grad-CAM**: Activations computed from the final residual block (`layer4`) backpropagated from the predicted grade score.
2. **Local MIL Patch Attention Saliency**: Attention weights $\alpha_i$ from `GatedAttentionPool` mapped back to native fundus pixel coordinates and smoothed with Gaussian kernels.
3. **Quantitative Metrics against IDRiD Pixel Lesion Masks**:
   - **Pointing Game Accuracy**: Hit if the maximum saliency pixel falls within any ground-truth lesion mask.
   - **Area-Matched Saliency Recall**: Fraction of ground-truth lesion pixels overlapping top-$K\%$ salient pixels (where $K$ matches the ground-truth lesion area).
   - **Pixel-level AUROC**: Discrimination between lesion and non-lesion retinal pixels using continuous saliency values.

### Rationale
- Translates qualitative heatmap visual inspection into rigorous, numerical clinical localization benchmarks.
- Adheres to scientific framing: Saliency indicates spatial attention alignment, not causal medical explanations.
