# Experimental Protocols & Benchmarking Guide

This document specifies the protocols, commands, metric definitions, and statistical significance testing procedures for reproducing all experimental results reported in the paper.

---

## 1. Overview of Benchmark Experiments

The empirical evaluation comprises four rigorously controlled experiments:

1. **Leave-One-Dataset-Out (LODO) Cross-Domain Generalization**:
   Evaluates domain generalization by training on 5 source cohorts and testing exclusively on a 6th unseen target cohort across 6 distinct rotations.
2. **Controlled Output-Head & Loss Ablation Suite**:
   Compares five prediction heads (`Softmax_CE`, `Softmax_QWK`, `CORAL`, `CORN`, `CLM_QWK`) on fixed, standardized ResNet-50 feature representations ($D=512$).
3. **Referable Diabetic Retinopathy (RDR) Screening Calibration**:
   Calibrates operating decision thresholds ($\tau^*$) at $\ge 95\%$ clinical specificity strictly on source validation data, evaluating performance on held-out target cohorts.
4. **Quantitative Explainability against Pixel Ground Truth**:
   Evaluates spatial localization fidelity of Global Grad-CAM and Local MIL attention saliency against expert-annotated lesion segmentation masks from IDRiD.

---

## 2. Reproduction Commands

### 2.1 6-Fold LODO Generalization Benchmark
Execute individual folds or full sweeps using `scripts/run_lodo.py`:

```bash
# Run single fold (e.g. held-out APTOS 2019) with proposed variant v3
python scripts/run_lodo.py --variant v3 --held-out aptos --device cuda

# Execute dry-run to verify pipeline integrity without downloading data
python scripts/run_lodo.py --variant v3 --held-out aptos --dry-run --device cpu

# Available held-out options: aptos, ddr, deepdrid, idrid, messidor2, eyepacs
# Available model variants: v0, v1, v2, v3
```

Results are serialized to:
- `results/lodo/<variant>/<held_out>/metrics.json`
- `results/lodo/<variant>/summary.json`

### 2.2 Controlled Output-Head Ablation
Execute 5-head benchmark on identical pre-extracted representations using `scripts/run_head_ablation.py`:

```bash
# Benchmark all 5 heads sequentially
python scripts/run_head_ablation.py --all-heads --device cuda

# Benchmark single head
python scripts/run_head_ablation.py --head clm_qwk --device cuda

# Dry-run validation
python scripts/run_head_ablation.py --head clm_qwk --dry-run --device cpu
```

Outputs are serialized to `results/head_ablation/summary.json` and `results/head_ablation/summary.csv`.

### 2.3 Quantitative XAI Evaluation
Benchmark saliency alignment against IDRiD lesion masks using `scripts/run_xai.py`:

```bash
# Run quantitative evaluation against IDRiD masks
python scripts/run_xai.py --checkpoint checkpoints/v3_best.pt --variant v3 --device cuda

# Dry-run validation
python scripts/run_xai.py --variant v3 --dry-run --device cpu
```

Outputs are serialized to `results/xai/metrics_summary.json`.

### 2.4 Export Paper Figures & Tables
Compile all serialized experimental results into 300 DPI figures and LaTeX tables:

```bash
python scripts/export_paper_artifacts.py --results-dir results/ --output-dir results/paper_artifacts/ --dpi 300
```

---

## 3. Metric Formulations

### 3.1 Quadratic Weighted Kappa (QWK)
Measures agreement between true grades $y$ and predictions $\hat{y} \in \{0, 1, 2, 3, 4\}$, penalizing larger grade discrepancies quadratically:
$$\kappa = 1 - \frac{\sum_{i,j} W_{ij} O_{ij}}{\sum_{i,j} W_{ij} E_{ij}}, \quad W_{ij} = \frac{(i - j)^2}{(C - 1)^2}$$
where $O$ is the observed confusion matrix and $E$ is the expected matrix under marginal independence.

### 3.2 Within-1 Grade Accuracy
Quantifies the proportion of predictions within $\pm 1$ grade of clinical truth:
$$\text{Acc}_{\pm 1} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(|y_i - \hat{y}_i| \le 1)$$
Clinical rationale: Misclassifying Grade 1 (Mild) as Grade 2 (Moderate) is a minor boundary variation, whereas misclassifying Grade 0 (No DR) as Grade 4 (PDR) is a catastrophic off-by-2+ error.

### 3.3 Referable DR Screening Metrics
- **RDR Definition**: Grade $\ge 2$ ($G \in \{2, 3, 4\}$).
- **RDR Risk Score**: $p_{\text{RDR}} = \sum_{k=2}^4 p_k$.
- **Operating Threshold**: Calibrated on source validation data such that $\text{Specificity}(\tau^*) \ge 0.95$.
- Evaluated metrics: Area Under the ROC Curve (AUC), Sensitivity at $\tau^*$, Specificity at $\tau^*$, Positive Predictive Value (PPV), and Negative Predictive Value (NPV).

### 3.4 Quantitative Explainability Metrics
- **Pointing Game Accuracy**: Binary hit/miss checking whether the pixel with global maximum saliency $\arg\max_{(x,y)} S(x, y)$ resides within any ground-truth lesion mask.
- **Area-Matched Saliency Recall**: Fraction of ground-truth lesion pixels recalled when selecting the top-$A$ highest saliency pixels, where $A$ equals the exact pixel count of true lesions:
  $$\text{Recall}_{\text{matched}} = \frac{|\{(x,y) \in \text{Top}_A(S) \cap \text{Lesion Mask}\}|}{|\text{Lesion Mask}|}$$
- **Pixel AUROC**: Treats continuous saliency intensity across all pixels as a binary lesion detector evaluated against the binary lesion mask.

---

## 4. Statistical Significance Testing

To verify whether observed performance gains are statistically significant and not artifacts of cross-cohort variance, pairwise Wilcoxon signed-rank tests are computed across the 6 LODO folds using `drdg.evaluation.statistics.compute_significance`:

- Null hypothesis ($H_0$): Difference between median QWK of model $A$ and model $B$ is zero.
- Alternative hypothesis ($H_1$): Model $A$ produces higher median target QWK.
- Multiplicity adjustment: Bonferroni correction across pairwise variant comparisons ($\alpha = 0.05 / 3 = 0.0167$).
