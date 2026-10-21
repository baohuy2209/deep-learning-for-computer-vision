# Explainability Methodology & Quantitative Validation

This document describes the explainability (XAI) framework in `drdg`, covering Global Branch Grad-CAM, Local MIL Attention Saliency mapping, and objective validation against pixel-level lesion ground truth from IDRiD.

---

## 1. Scientific & Clinical Framing

A major critique of visual saliency in medical imaging is the subjective, qualitative cherry-picking of attention heatmaps without objective validation. Furthermore, saliency heatmaps represent associative model sensitivity rather than causal pathophysiological explanations.

In `drdg`, visual explainability is evaluated under two strict scientific commitments:
1. **Dual-Stream Interpretability**: Separating diffuse global context from localized focal lesion attention.
2. **Objective Quantitative Benchmarking**: Measuring spatial overlap against human ophthalmologist lesion segmentations (microaneurysms, haemorrhages, hard exudates, soft exudates) rather than visual inspection alone.

---

## 2. Explainability Modalities

### 2.1 Global Stream Grad-CAM (`drdg.explainability.gradcam.GradCAM`)
- **Target Layer**: Final residual block of ResNet-50 (`layer4[-1]`), producing spatial feature activations $\mathbf{A} \in \mathbb{R}^{2048 \times 16 \times 16}$.
- **Weight Calculation**: For target ordinal class $c$, gradients of the score $y^c$ with respect to feature maps $\mathbf{A}^k$ are globally pooled:
  $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i,j}^k}$$
- **Saliency Map Generation**:
  $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c \mathbf{A}^k\right)$$
  The resulting $16 \times 16$ heatmap is bilinearly upsampled to the full fundus resolution ($512 \times 512$) and min-max normalized to $[0, 1]$.

### 2.2 Local MIL Gated Attention Saliency (`drdg.explainability.mil_saliency.MILAttentionSaliency`)
- **Instance Attention Weights**: For each patch $k \in \{1, \dots, K\}$ extracted from spatial anchor coordinates $(y_k, x_k)$, the Gated Attention mechanism assigns scalar weight $a_k \in [0, 1]$ ($\sum_k a_k = 1$).
- **Continuous Density Mapping**: Rather than displaying discrete patch boxes, discrete attention weights are projected onto the continuous image canvas through Gaussian smoothing kernels:
  $$S_{\text{MIL}}(y, x) = \sum_{k=1}^K a_k \exp\left(-\frac{(y - c_{y,k})^2 + (x - c_{x,k})^2}{2 \sigma^2}\right)$$
  where $(c_{y,k}, c_{x,k})$ is the geometric center of patch $k$, and $\sigma = 15.0$ pixels controls spatial dispersion.

---

## 3. IDRiD Pixel Lesion Ground Truth

Validation is conducted using the Indian Diabetic Retinopathy Image Dataset (IDRiD) segmentation partition ($N=81$ test fundus photographs), which contains pixel-level annotations for four hallmark clinical lesions:

1. **Microaneurysms (MA)**: Earliest visible sign, tiny circular red vascular out-pouchings ($<50\,\mu\text{m}$).
2. **Haemorrhages (HE)**: Flame-shaped or dot-and-blot intraretinal extravasations.
3. **Hard Exudates (EX)**: Waxy yellow lipoprotein lipid deposits precipitated from permeable vessels.
4. **Soft Exudates / Cotton Wool Spots (SE)**: Ischemic infarcts of retinal nerve fiber layers.

The unified binary lesion ground truth mask is formed via boolean union:
$$\mathbf{M}_{\text{composite}}(y, x) = \mathbf{M}_{\text{MA}}(y, x) \lor \mathbf{M}_{\text{HE}}(y, x) \lor \mathbf{M}_{\text{EX}}(y, x) \lor \mathbf{M}_{\text{SE}}(y, x)$$

---

## 4. Quantitative Localization Metrics

The continuous saliency map $S(y, x) \in [0, 1]$ is benchmarked against $\mathbf{M}_{\text{composite}}$ via three objective metrics (`drdg.explainability.metrics.evaluate_xai_sample`):

### 4.1 Pointing Game Accuracy
Determines whether the model's most attended single pixel falls inside a genuine lesion:
$$\text{Hit} = \mathbb{I}\left( \mathbf{M}_{\text{composite}}\left(\arg\max_{(y,x)} S(y, x)\right) = 1 \right)$$
Overall accuracy is the percentage of hits across all evaluated test images.

### 4.2 Area-Matched Saliency Recall
To evaluate whether attention accurately covers lesion extents without penalizing models for diffuse boundary gradients, the saliency map is thresholded at the top-$A$ highest-intensity pixels, where $A = \sum_{y,x} \mathbf{M}_{\text{composite}}(y, x)$ is the exact area of ground-truth lesions:
$$\text{Recall}_{\text{matched}} = \frac{\sum_{(y,x) \in \text{Top}_A(S)} \mathbf{M}_{\text{composite}}(y, x)}{\sum_{y,x} \mathbf{M}_{\text{composite}}(y, x)}$$

### 4.3 Pixel AUROC
Treats each pixel's saliency value as a continuous prediction score and calculates the Area Under the Receiver Operating Characteristic curve across all retinal pixels inside the valid FOV.

---

## 5. Running the XAI Benchmark

```bash
# Execute evaluation on IDRiD segmentation set
python scripts/run_xai.py --checkpoint checkpoints/v3_best.pt --variant v3 --device cuda

# Interactive inspection notebook
jupyter notebook notebooks/24_xai_idrid.ipynb
```
