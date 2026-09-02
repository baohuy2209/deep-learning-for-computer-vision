# Core Manuscript Sections: Related Work, Method, and Experimental Setup

> **Paper Working Title:**  
> **Deconstructing Domain Shift in Diabetic Retinopathy Grading: A Multi-Scale Multiple Instance Learning Approach with Cumulative Link Ordinal Modeling**  
>
> **Status:** Full pre-experimental manuscript text ($\approx 60\%$ of total paper volume).  
> **Target Venues:** *IEEE Transactions on Medical Imaging (TMI)* / *Medical Image Analysis (MedIA)* / *MICCAI*.

---

## 1. Related Work

### 1.1. Deep Learning in Diabetic Retinopathy Grading & The Domain Shift Challenge
Automated grading of Diabetic Retinopathy (DR) from digital fundus photography has made tremendous strides since early convolutional neural networks demonstrated specialist-level diagnostic sensitivity on homogeneous benchmark cohorts (Gulshan et al., 2016; Ting et al., 2017). However, standard clinical deployment across disparate healthcare systems remains bottlenecked by the *domain shift paradox*: models achieving exceptional Quadratic Weighted Kappa ($\text{QWK} > 0.90$) internally often experience severe performance degradation ($\Delta_{\text{QWK}} \approx 0.15 - 0.35$) when evaluated on unseen clinical populations, sensor optics, and illumination profiles (AlBadawy et al., 2018; Quellec et al., 2021). 

Two primary physical and structural mechanisms underpin this transfer failure:
1. **Downsampling Pixel Loss:** Standard deep architectures require downsampling high-resolution fundus photographs ($3000 \times 2000\text{ px}$) to compact inputs ($224 \times 224$ or $512 \times 512\text{ px}$). This downsampling physically annihilates minute, focal microaneurysms ($\le 120\,\mu\text{m}$, representing sub-$15\text{px}$ lesions), causing severe sensitivity collapse on Grade 1 (Mild DR).
2. **Receptive Field Limitations on Global Context:** Standard convolutional receptive fields struggle to model long-range spatial dependencies across distinct anatomical quadrants, which is mandatory to clinically verify the International Clinical Diabetic Retinopathy (ICDR) **4-2-1 rule** for Grade 3 (Severe NPDR) diagnosis.

---

### 1.2. Taxonomy of Ordinal Classification in Biomedical Staging
DR severity grading follows an intrinsically ordered ordinal scale according to the International Clinical Diabetic Retinopathy Disease Severity Scale:
$$\mathcal{Y} = \{0: \text{No DR}, \; 1: \text{Mild NPDR}, \; 2: \text{Moderate NPDR}, \; 3: \text{Severe NPDR}, \; 4: \text{Proliferative DR}\}$$

Following the mathematical taxonomy established by **Vargas et al. (2020)** in their comprehensive review of ordinal classification, existing deep learning approaches for progressive disease staging can be systematically grouped into five distinct methodologies:

```
                            [Deep Ordinal Staging Taxonomy]
                                           │
         ┌───────────────┬─────────────────┼─────────────────┬───────────────┐
         ▼               ▼                 ▼                 ▼               ▼
     Group 1          Group 2           Group 3           Group 4         Group 5 (Proposed)
  Nominal Softmax   Continuous Metric  Consistent Rank   Conditional     Cumulative Link
     + CE Loss      Regression / QWK    Logits (CORAL)    Partitions      Model (CLM_QWK)
  (Unordered Space) (Distance Penalty) (Shared W, Biases)   (CORN)       (Asymmetric clog-log)
```

#### Group 1: Standard Nominal Multi-Class Classification (Softmax + Cross-Entropy)
The dominant baseline in computer vision treats the $K=5$ severity stages as unordered categorical classes:
$$\hat{p}_k = \frac{\exp(z_k)}{\sum_{j=0}^{K-1} \exp(z_j)}, \quad \mathcal{L}_{\text{CE}} = -\sum_{k=0}^{K-1} y_k \log(\hat{p}_k)$$
* **Fundamental Limitation in Medical Staging:** Cross-entropy penalizes all misclassifications uniformly. Misdiagnosing a Grade 0 (Normal) patient as Grade 1 incurs the exact same loss penalty as misdiagnosing them as Grade 4 (Proliferative DR requiring emergency panretinal photocoagulation). Furthermore, Softmax parameterizes $K$ independent hyperplanes without geometric ordering constraints, resulting in severe decision boundary fragmentation and rank monotonicity violations under domain shift.

#### Group 2: Continuous Metric & Distance Regression Formulations
To penalize severe grading errors proportionally to the clinical distance $|y - \hat{y}|$, regression models map features to a continuous scalar $s \in \mathbb{R}$:
$$\mathcal{L}_{\text{MSE}} = (y - s)^2, \quad \text{or} \quad \mathcal{L}_{\text{Smooth\_L1}} = \text{Smooth}_{L_1}(y - s)$$
Recent works directly optimize differentiable approximations of the Quadratic Weighted Kappa ($\mathcal{L}_{\text{Continuous\_QWK}}$) (de la Rosa et al., 2020). While continuous regression imposes a global distance metric, it imposes a rigid Euclidean equidistant assumption between discrete clinical grades (e.g., assuming the pathological gap between Grade 0 and 1 is mathematically identical to the gap between Grade 3 and 4), which contradicts clinical reality where transitions between stages reflect non-linear biological thresholds.

#### Group 3: Consistent Rank Logits (CORAL / Ordered Hyperplanes)
Proposed by Cao et al. (2020), Consistent Rank Logits (CORAL) transforms the $K$-class ordinal task into $K-1$ binary classification subtasks sharing a single feature projection weight $\mathbf{w}$ with monotonic independent bias intercepts $\{b_0, b_1, \dots, b_{K-2}\}$:
$$P(Y > k \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{z} + b_k), \quad \mathcal{L}_{\text{CORAL}} = -\sum_{k=0}^{K-2} \Big[ \mathbb{I}(y > k) \log(\hat{P}_k) + \mathbb{I}(y \le k) \log(1 - \hat{P}_k) \Big]$$
CORAL guarantees parallel decision hyperplanes with identical orientations, ensuring that the predicted rank probability sequence remains consistent without hyperplane intersections.

#### Group 4: Conditional Ordinal Neural Networks (CORN / Sequential Binary Classifiers)
Proposed by Shi et al. (2021), Conditional Ordinal Neural Networks (CORN) formulate ordinal classification as a series of conditional probability events:
$$P(Y > k \mid Y \ge k, \mathbf{x}) = \sigma(f_k(\mathbf{x})), \quad P(Y > k \mid \mathbf{x}) = \prod_{j=0}^k P(Y > j \mid Y \ge j, \mathbf{x})$$
CORN enforces unconditional rank consistency by construction via the chain rule of conditional probability, preventing label ranking violations.

#### Group 5: Cumulative Link Models with Asymmetric Links (CLM_QWK — Proposed)
Cumulative Link Models (Agresti, 2010) represent the principled statistical formulation for ordered categorical data. Latent disease severity is projected onto a continuous latent axis $s(\mathbf{x}) = \mathbf{w}^T \mathbf{z}$ partitioned by strictly ordered monotonic cutpoints $\theta_1 < \theta_2 < \theta_3 < \theta_4$. By substituting standard symmetric logistic/probit links with the **asymmetric Complementary Log-Log ($\text{clog-log}$)** link function:
$$P(Y \le k \mid \mathbf{x}) = F(\theta_k - s(\mathbf{x})) = 1 - \exp\big(-\exp(\theta_k - s(\mathbf{x}))\big)$$
the model mathematically mirrors the clinical progression of diabetic microvasculopathy, where early microaneurysms progress slowly before triggering accelerating exponential capillary non-perfusion and neovascularization.

---

## 2. Methodology

The complete end-to-end framework consists of five integrated stages:

```
                            [RAW HIGH-RESOLUTION FUNDUS IMAGE]
                                            │
               Stage 2: 4-Step Robust Retinal FOV Segmentation (compute_fov_mask)
               Strictly on RAW RGB (prevents Ben Graham 128-intensity collapse)
               1. Otsu's Adaptive Thresholding (vs. Fixed 'gray > 15')
               2. Morphological Close + Open (Ellipse 25x25)
               3. Largest Connected Component Extraction (Removes watermarks/glare)
               4. Inward Morphological Erosion (erode_px = 15)
                                            │
                  ┌─────────────────────────┴─────────────────────────┐
                  ▼                                                   ▼
      [Global Context Branch (512x512)]           [Local Sliding-Window MIL Branch v2]
                  │                               - Overlapping Patches: 128x128, Stride: s=96 px
              ResNet-50                           - Raw FOV Filter Ratio >= 0.5
                  │                               - Train: Stochastic M in [32, 64] / epoch
          NonLocalBlock2D (H*W Relational)        - Eval: Dense Full Grid (up to 256 patches)
                  │                                                   │
            CBAM Attention                                     EfficientNet-B0
                  │                                                   │
       ┌──────────┴──────────┐                              Dual Aggregation Pooling:
       ▼                     ▼                              1. 1D Max-Pool (Grade 1 MA, H1)
    Global GAP       QuadrantTokenAggregator                2. Masked Gated Attention (H2)
    (2048-d)         (4-Quadrant Transformer, 2048-d)                 │
       └──────────┬──────────┘                                F_Local (2560-d)
                  ▼                                                   │
          F_Global (4096-d)                                           │
                  │                                                   │
                  └─────────────────────────┬─────────────────────────┘
                                            ▼
                               [Multimodal Fusion (6656-d)]
                                            │
                                LayerNorm Projection MLP
                                            │
                                Latent Embedding (512-d)
                                            │
                        [Champion CLM_QWK Ordinal Regression Head]
                        - Latent Severity Score: s(x) = W^T z
                        - Monotonic Cutpoints: theta_1 < theta_2 < theta_3 < theta_4
                        - clog-log Link: P(Y <= k) = 1 - exp(-exp(theta_k - s(x)))
                        - Objective: L = Interval NLL Loss + lambda(t) * Continuous QWK Loss
```

---

### 2.1. 4-Step Robust FOV Segmentation & Optical Preprocessing
To resolve sensor exposure variances and boundary artifacts across different fundus camera manufacturers (Canon, Topcon, Zeiss, Kowa), we employ a 4-step optical segmentation pipeline executed **strictly on the raw RGB fundus image**:
1. **Otsu's Adaptive Thresholding:** Computes the optimal bimodal intensity threshold per image dynamically on the green/luminance channel. *(Crucial: Applying Otsu to Ben Graham preprocessed images maps the dark background to gray 128, destroying bimodal contrast and collapsing detection to a thin 6% boundary halo).*
2. **Morphological Close followed by Open:** A $25 \times 25$ elliptical structuring element seals internal holes from dark vessel trunks and foveal darkness, then cleans peripheral isolated sensor noise.
3. **Largest Connected Component Extraction:** Strictly isolates the primary retinal disk, removing camera watermarks, peripheral artifacts, and calibration text.
4. **Inward Boundary Erosion (`erode_px = 15`):** Strips away untrusted peripheral illumination halos and edge vignetting.

---

### 2.2. Global Context Branch with Relational Attention (Resolving $H_2$ — 4-2-1 Rule)
The global stream processes the whole fundus resized to $512 \times 512 \times 3$ through an ImageNet-pretrained ResNet-50 backbone:
1. **NonLocalBlock2D (Wang et al., 2018):** Inserted after Layer 4 ($2048$ channels), computing pairwise spatial self-attention across the full feature map:
   $$\mathbf{y}_i = \frac{1}{\mathcal{C}(\mathbf{x})} \sum_{\forall j} f(\mathbf{x}_i, \mathbf{x}_j) g(\mathbf{x}_j) + \mathbf{x}_i$$
   This establishes direct long-range connections between distant anatomical quadrants.
2. **CBAM Attention:** Sequential Channel and $7 \times 7$ Spatial Attention modules refine feature activations.
3. **QuadrantTokenAggregator:** Partitioning the spatial feature map into the four clinical quadrants ($\text{Superior-Temporal}, \text{Superior-Nasal}, \text{Inferior-Temporal}, \text{Inferior-Nasal}$), each quadrant is average-pooled into a 2048-d token. The 4 quadrant tokens are enriched with learned positional embeddings and processed through a 1-layer Transformer encoder, explicitly encoding the **ICDR 4-2-1 diagnostic rule** for Severe NPDR (Grade 3):
   $$F_{\text{Quadrant}} = \text{TransformerEncoder}\Big([\mathbf{q}_1 + \mathbf{p}_1, \; \mathbf{q}_2 + \mathbf{p}_2, \; \mathbf{q}_3 + \mathbf{p}_3, \; \mathbf{q}_4 + \mathbf{p}_4]\Big)$$
4. **Global Representation:** Concatenating Global Average Pooling ($2048\text{-d}$) and the Quadrant Transformer Summary ($2048\text{-d}$):
   $$F_{\text{Global}} = [F_{\text{GAP}} \,\|\, F_{\text{Quadrant}}] \in \mathbb{R}^{4096}$$

---

### 2.3. Local Sliding-Window MIL Branch (Resolving $H_1$ — Focal Pixel-Loss)
To capture focal microaneurysms without downsampling pixel-loss, the local branch extracts native-resolution patches ($128 \times 128\text{ px}$) using an overlapping sliding window:
* **Overlapping Scanning Geometry:** Window size $p = 128\text{ px}$, stride $s = 96\text{ px}$ ($25\%$ overlap). This eliminates redundant adjacent computation while providing comprehensive retinal coverage without candidate depletion.
* **Raw FOV Mask Filtering:** Candidate patches are retained only if active retinal tissue $\text{mean}(\text{mask}_{\text{patch}}) \ge 0.50$, discarding uninformative black corners.
* **Dual Operation Modes:**
  - *Training:* Stochastic sub-sampling of $M \in [32, 64]$ valid candidate patches per image per epoch (acts as spatial regularization and caps VRAM).
  - *Evaluation:* Full dense grid sweep (up to 256 patches) to guarantee zero missed focal microaneurysms.
* **Dual Aggregation Pooling:**
  1. *1D Max-Pooling:* Captures extreme localized lesions characteristic of Grade 1 Mild DR ($H_1$):
     $$F_{\text{Max}} = \max_{k \in \{1 \dots N\}} f_{\text{patch}}^{(k)} \in \mathbb{R}^{1280}$$
  2. *Masked Gated Attention Pooling:* Aggregates lesion density across valid patches while ignoring zero-padding tokens via boolean `patch_mask`:
     $$a_k = \frac{\exp\big(\mathbf{w}^T (\tanh(V h_k) \odot \sigma(U h_k)) + \text{mask\_bias}_k\big)}{\sum_{j=1}^N \exp\big(\mathbf{w}^T (\tanh(V h_j) \odot \sigma(U h_j)) + \text{mask\_bias}_j\big)}, \quad F_{\text{Att}} = \sum_{k=1}^N a_k h_k \in \mathbb{R}^{1280}$$
* **Local Representation:**
  $$F_{\text{Local}} = [F_{\text{Max}} \,\|\, F_{\text{Att}}] \in \mathbb{R}^{2560}$$

---

### 2.4. Multimodal Latent Fusion & Champion CLM_QWK Ordinal Head
The global and local feature representations are concatenated and projected into a compact latent severity embedding:
$$F_{\text{Fused}} = [F_{\text{Global}} \,\|\, F_{\text{Local}}] \in \mathbb{R}^{6656}$$
$$\mathbf{z} = \text{LayerNorm}\Big(W_2 \cdot \text{Dropout}\big(\text{Mish}(W_1 F_{\text{Fused}} + b_1)\big) + b_2\Big) \in \mathbb{R}^{512}$$

#### Mathematical Derivation of CLM_QWK
1. **Scalar Disease Severity:** $\mathbf{z}$ is linearly projected to a 1D scalar score:
   $$s(\mathbf{x}) = \mathbf{w}^T \mathbf{z} \in \mathbb{R}$$
2. **Strict Monotonic Cutpoints:** Parameterized via positive softplus increments to mathematically prevent threshold inversions:
   $$\theta_1 \in \mathbb{R}, \quad \theta_{k+1} = \theta_k + \text{softplus}(\alpha_k) + 10^{-4}, \quad k \in \{1, 2, 3\}$$
3. **Asymmetric Complementary Log-Log Link Function:**
   $$P(Y \le k \mid \mathbf{x}) = 1 - \exp\big(-\exp(\theta_k - s(\mathbf{x}))\big)$$
   Discrete category probabilities are obtained via interval differences:
   $$p_0 = P(Y \le 0), \quad p_k = P(Y \le k) - P(Y \le k-1) \; (k=1,2,3), \quad p_4 = 1 - P(Y \le 3)$$
4. **Hybrid Objective Function with Linear Warmup:**
   $$\mathcal{L}_{\text{CLM\_QWK}} = \mathcal{L}_{\text{Interval\_NLL}}(\mathbf{p}, \mathbf{y}) + \lambda(t) \cdot \mathcal{L}_{\text{Continuous\_QWK}}(\mathbf{p}, \mathbf{y})$$
   where the quadratic distance penalty weight $\lambda(t)$ is smoothly warmed up:
   $$\lambda(t) = \min\left(1.0, \; \frac{t}{T_{\text{warmup}}}\right) \cdot \lambda_{\text{target}}$$

---

## 3. Experimental Setup

### 3.1. Multi-Center Cohort Composition & Label Standardization
To evaluate cross-domain generalization under clinical domain shift, we harmonize six international diabetic retinopathy cohorts comprising **55,570 fundus images**:

| Dataset | Origin / Country | Sensor / Camera Hardware | Raw Images | Field of View | Primary Optical Characteristics |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **EyePACS** | USA (Multi-clinic) | Centervue, Canon, Topcon | 35,126 | $45^\circ$ | High clinical heterogeneity, varied exposures |
| **APTOS-2019** | India (Aravind Eye) | Zeiss Visucam, Topcon | 3,662 | $45^\circ$ | High illumination contrast, rural screening |
| **Messidor-2** | France (Multi-center) | Topcon TRC NW6 | 1,748 | $45^\circ$ | High optical resolution, uniform illumination |
| **DDR** | China (Multi-hospital) | Canon CR-2 / Topcon | 13,673 | $45^\circ$ | Extensive lesion annotations (MA, HE, EX, SE) |
| **DeepDRiD** | China (Multi-center) | Topcon TRC-NW400 | 2,000 | $45^\circ$ | Bilateral paired fundus (Macula + Disc) |
| **IDRiD** | India (Specialist) | Kowa VX-10$\alpha$ | 516 | $50^\circ$ | Gold-standard expert lesion segmentation |
| **Total** | **6 International Cohorts** | **8 Camera Models** | **55,570** | **$45^\circ - 50^\circ$** | **Full International Clinical Spectrum** |

All datasets are harmonized to the standardized 5-point International Clinical Diabetic Retinopathy Disease Severity (ICDRS) scale:
* **Grade 0 (No DR):** No microaneurysms or hemorrhages.
* **Grade 1 (Mild NPDR):** Microaneurysms only.
* **Grade 2 (Moderate NPDR):** More than microaneurysms but less than Severe NPDR.
* **Grade 3 (Severe NPDR):** Any of the 4-2-1 rule criteria (hemorrhages in 4 quadrants, venous beading in $\ge 2$ quadrants, or IRMA in $\ge 1$ quadrant).
* **Grade 4 (Proliferative DR):** Neovascularization and/or vitreous/preretinal hemorrhage.

---

### 3.2. Sterile Partitioning Protocol (Zero Patient Leakage)
1. **Patient-Level Grouping:** All images are indexed by unique composite IDs `dataset_key::patient_id`. Data is split strictly at the patient level using `GroupShuffleSplit` ($\mathcal{P}_{\text{Train}} \cap \mathcal{P}_{\text{Val}} = \emptyset, \; \mathcal{P}_{\text{Train}} \cap \mathcal{P}_{\text{Test}} = \emptyset$). Paired left/right eye fundus images from the same patient are strictly kept within the same partition.
2. **Prevalence-Preserving Split:** Splitting into 70% Train, 15% Validation, and 15% Test is executed **before** balancing. Multi-source quota balancing is applied exclusively to the training pool (`train_metadata.csv`), ensuring that Validation and Test sets reflect natural clinical prevalence.

---

### 3.3. 6-Fold Leave-One-Dataset-Out (LODO) Generalization Protocol
To measure transfer resilience without external fine-tuning, we execute a rigorous 6-Fold Leave-One-Dataset-Out benchmark:
$$\text{For Fold } k \in \{1 \dots 6\}: \quad \mathcal{D}_{\text{Train\_Pool}} = \bigcup_{j \neq k} \mathcal{D}_j, \quad \mathcal{D}_{\text{Val}} = \text{Val}(\mathcal{D}_{\text{Train\_Pool}}), \quad \mathcal{D}_{\text{OOD\_Test}} = \mathcal{D}_k$$
For each fold, the model is trained strictly on the 5 pooled source domains and evaluated zero-shot on the entirely held-out 6th dataset.
The domain shift transfer gap is quantified by:
$$\Delta_{\text{QWK}}^{(k)} = \text{QWK}_{\text{In-Domain Val}}^{(k)} - \text{QWK}_{\text{OOD Test}}^{(k)}$$

---

### 3.4. Evaluation Metrics & Statistical Standards
Performance is evaluated across six complementary clinical and statistical dimensions:
1. **Quadratic Weighted Kappa ($\text{QWK}$):** Primary ordinal ranking agreement metric penalizing errors quadratically:
   $$\text{QWK} = 1 - \frac{\sum_{i,j} W_{ij} O_{ij}}{\sum_{i,j} W_{ij} E_{ij}}, \quad W_{ij} = \frac{(i - j)^2}{(K - 1)^2}$$
2. **Within-1-Grade Accuracy:** Proportion of predictions within one clinical stage of the ground truth ($|y - \hat{y}| \le 1$).
3. **Referable DR Sensitivity & Specificity:** Diagnostic sensitivity for referable disease ($\text{Grade} \ge 2$) measured at clinical operating points (e.g., Sensitivity at fixed 95% Specificity screening gate).
4. **Referable Area Under ROC Curve (AUC):** Macro-discriminative diagnostic capacity for referable triage.
5. **Per-Grade Sensitivity Breakdown:** Individual sensitivity metrics for Grades 0 through 4, specifically tracking the restoration of Grade 1 ($H_1$) and Grade 3 ($H_2$).
6. **Rank Monotonicity Violation Percentage:** Percentage of evaluation samples where cumulative predicted probabilities violate monotonic non-increasing ordering constraints ($P(Y > k) > P(Y > k-1)$).

---

### 3.5. Computational Implementation, Pipeline Caching & Hardware Infrastructure
All models are implemented in PyTorch 2.0+ and trained on NVIDIA A100 Tensor Core GPUs (40GB/80GB VRAM). 
* **Mixed Precision:** Native `bfloat16` / `float16` with `torch.amp.autocast`.
* **Optimizer:** AdamW ($\text{lr} = 10^{-4}$, weight decay $= 10^{-4}$) with `CosineAnnealingLR` ($\eta_{\min} = 10^{-6}$).
* **Offline 2-Tier Coordinate & Preprocessing Pre-Caching:**
  * To eliminate the CPU computational bottleneck ($810\text{ ms/image}$, comprising $187\text{ ms}$ bilateral filtering, $354\text{ ms}$ Gaussian subtraction, $242\text{ ms}$ morphological FOV segmentation, and $23\text{ ms}$ grid search $\implies 158\text{ CPU hours}$ across 100 epochs), preprocessed images and $(N, 2)$ valid coordinate anchors $[y_0, x_0]$ are serialized into compressed `.npz` archives.
  * In `__getitem__`, `.npz` loads take $20\text{--}30\text{ ms}$, and uniform batch patch slicing in `trainer_compatible_collate_fn` requires $<1\text{ ms}$, achieving a $>25\times$ data pipeline speedup without memory bloat ($\approx 50\text{--}100\text{ KB/sample}$ vs. $>50\text{ MB/sample}$ for sliced patches).
* **IPC Shared Memory Optimization:** DataLoader multi-workers communicate raw `uint8` tensors across shared memory, cutting batch transfer footprint from $755\text{ MB}$ to $189\text{ MB}$ ($4\times$ reduction). ImageNet Z-score standardization (`div(255).sub(mean).div(std)`) is executed on GPU Tensor Cores.
* **Thread Safety:** `cv2.setNumThreads(0)` is strictly enforced to prevent CPU core over-subscription and thread thrashing across DataLoader worker processes.
