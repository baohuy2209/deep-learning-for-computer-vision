# Exploratory Data Analysis (EDA) Report
## Multi-Center Evaluation Across 6 Diabetic Retinopathy Fundus Datasets (55,570 Fundus Photographs)

**Author:** Medical Computer Vision Research Group  
**Target Datasets:** APTOS 2019, DDR, DeepDRiD, IDRiD, Messidor-2, EyePACS (Kaggle)  
**Reference Documents:** [eda_6_datasets.ipynb](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/eda/eda_6_datasets.ipynb) | [01_datasets_analysis.md](file:///d:/Github/deep-learning-for-computer-vision/documents/01_datasets_analysis.md) | [02_pipeline_architecture.md](file:///d:/Github/deep-learning-for-computer-vision/documents/02_pipeline_architecture.md)

---

## Executive Summary

In Diabetic Retinopathy (DR) grading from Color Fundus Photography (CFP), the primary obstacle preventing successful real-world clinical deployment is not model parameter capacity, but severe **Domain Shift** and **Ordinal Class Imbalance** across different healthcare centers.

This report synthesizes the statistical and empirical findings from **55,570 standardized fundus photographs** across 6 international cohorts following complete harmonization of the IDRiD dataset (516 images across combined training and testing splits). The data audit highlights three primary phenomena:
1. **Patient-Level Data Fragmentation & Leakage Risk:** Many patients possess bilateral examinations (Left/Right eyes) or multi-field views. Standard naive random splitting (Random Stratified Split) causes patient-level anatomical vascular features to leak between training and testing sets, producing deceptive validation performance.
2. **Label Shift / Prior Probability Shift:** Disease prevalence varies dramatically, ranging from 73.5% normal images (Grade 0) in community screening programs (EyePACS) to 67.4% disease-positive cases in tertiary referral ophthalmology clinics (IDRiD). The maximum class imbalance reaches **36.45x**, rendering standard Cross-Entropy loss ineffective for severe and proliferative stages (Grades 3 & 4).
3. **Covariate / Sensor Shift:** Sensor dimensions vary from standardized square crops ($512 \times 512$) to ultra-high-resolution SLR fundus cameras ($4288 \times 2848$). Three-channel color analysis reveals significant Blue channel intensity shifts ($\mu_{Blue}$) from $17.55$ to $37.78$ across camera manufacturers (Topcon, Canon, Zeiss, Kowa, Centervue), while focus sharpness scores degrade substantially in community cohorts due to real-world cataracts and corneal reflections.

Based on quantitative evidence, this report establishes the rationale supporting the **Dual-Branch Multi-Scale Multiple Instance Learning (MIL)** architecture, color constancy preprocessing (**Ben Graham & Green-Channel CLAHE**), ordinal regression heads (**Cumulative Link Models - CLM/CORAL/CORN**) with continuous **Continuous QWK Loss**, and the **Leave-One-Dataset-Out (LODO)** evaluation protocol.

```
+----------------------------------------------------------------------------------------------------+
|                                  MULTI-CENTER DATASET OVERVIEW                                     |
|                                    55,570 Fundus Images (CFP)                                      |
+----------------------------------------------------------------------------------------------------+
       |                                      |                                      |
       v                                      v                                      v
  [APTOS 2019]                            [EyePACS]                              [IDRiD]
  3,662 images                            35,126 images                          516 images (Train+Test)
  India (Aravind Eye Hospital)            USA (Telehealth Screening)             India (Eye Clinic)
  Sensor: 1886x1458 (1.25x)               Sensor: 3651x2490 (1.46x)              Sensor: 4288x2848 (1.51x)
       |                                      |                                      |
       v                                      v                                      v
    [DDR]                                [DeepDRiD]                            [Messidor-2]
  12,522 images                           2,000 images (500 pts, 4 views)        1,744 images
  China (Multi-Hospital)                  China (Dual-Eye, Two-Field)            France (3 Univ. Hospitals)
  Sensor: 816x734 (1.05x)                 Sensor: 1820x1872 (0.97x)              Sensor: 512x512 (1.00x)
```

---

## 1. Data Integrity & Anti-Leakage Protocol

### 1.1 Risk of Patient-Level Data Leakage
In clinical ophthalmology, paired eyes from the same individual share systemic physiological factors (blood pressure, glycemic control via HbA1c, disease duration, retinal microvascular geometry).

Applying standard image-level random partitioning creates silent data leakage:
* The left eye ($I_L$) is placed in the Training set while the right eye ($I_R$) of the same patient resides in the Validation/Test set.
* Convolutional neural networks memorize patient-specific vascular morphology and fundus pigmentation rather than generalizable pathological micro-lesions (Microaneurysms, Hemorrhages, Hard Exudates, Neovascularization).
* This creates **over-optimistic performance** during development that fails upon clinical deployment to unseen patients.

### 1.2 Strict Patient-Level Grouping Protocol
The data loader extracts unique patient identifiers (`patient_id`) and ocular laterality (`side` in `{left, right}`) via deterministic parsing rules:
* **DeepDRiD:** 2,000 images from 500 patients (4 views per patient: central and disc-centered). The parser isolates `image_id.split('_')[0]`, grouping all 4 images of a patient into a single partition unit.
* **EyePACS:** Parses the `<patient_id>_<left|right>.jpeg` structure, ensuring all 35,126 images are strictly grouped by patient identity.
* **IDRiD:** Merged 413 training and 103 test images into 516 standardized records in `idrid_total.csv`, standardizing names as `train_IDRiD_xxx` and `test_IDRiD_xxx`.
* **Partitioning Rule:** Enforces `GroupShuffleSplit` on `patient_id` such that:
  $$\mathcal{P}_{\text{Train}} \cap \mathcal{P}_{\text{Val}} = \emptyset, \quad \mathcal{P}_{\text{Train}} \cap \mathcal{P}_{\text{Test}} = \emptyset$$

### 1.3 Consolidated Data Audit Table

Audit results extracted via `load_and_standardize_dataset()` in `eda_6_datasets.ipynb`:

| Dataset Key | Dataset Name | Total Records | Valid Records | Duplicates | Null Labels | Invalid Labels | Unique Patients | Left Eyes | Right Eyes | Ingestion Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `APTOS` | **APTOS 2019** | 3,662 | 3,662 | 0 | 0 | 0 | 3,662 | 0* | 0* | PASSED (100%) |
| `DDR` | **DDR Dataset** | 12,522 | 12,522 | 0 | 0 | 0 | 12,522 | 0* | 0* | PASSED (100%) |
| `DeepDRiD` | **DeepDRiD** | 2,000 | 2,000 | 0 | 0 | 0 | 500 | 999 | 1,001 | PASSED (100%) |
| `IDRiD` | **IDRiD** | 516 | 516 | 0 | 0 | 0 | 413 | 0* | 0* | PASSED (100%) |
| `Messidor-2`| **Messidor-2** | 1,744 | 1,744 | 0 | 0 | 0 | 872 | 0* | 0* | PASSED (100%) |
| `EyePACS` | **EyePACS (Kaggle)** | 35,126 | 35,126 | 0 | 0 | 0 | 17,563 | 17,563 | 17,563 | PASSED (100%) |
| **TOTAL** | **Combined Cohort** | **55,570** | **55,570** | **0** | **0** | **0** | **35,532** | **18,562** | **18,564** | **STERILE (100%)** |

*\*Note:* Patient IDs are prefixed as `{dataset_key}::{patient_id}` to prevent cross-dataset namespace collisions.

---

## 2. In-Depth Domain Shift Analysis

### 2.1 Label Shift / Prior Probability Shift $P(Y)$

The International Clinical Diabetic Retinopathy Severity (ICDRS) scale defines 5 disease grades:
* **Grade 0 (No DR):** No microaneurysms or retinal hemorrhages.
* **Grade 1 (Mild NPDR):** Microaneurysms (MA) only.
* **Grade 2 (Moderate NPDR):** More than microaneurysms, but less than Severe NPDR (scattered hemorrhages, hard/soft exudates).
* **Grade 3 (Severe NPDR):** 4-2-1 rule (>20 intraretinal hemorrhages in all 4 quadrants, venous beading in $\ge 2$ quadrants, or prominent IRMA in $\ge 1$ quadrant).
* **Grade 4 (Proliferative DR - PDR):** Neovascularization, vitreous/preretinal hemorrhage, fibrous proliferation.

#### Detailed Severity Breakdown Across 6 Cohorts:

| Dataset | Total Images | Grade 0 (No DR) | Grade 1 (Mild) | Grade 2 (Moderate) | Grade 3 (Severe) | Grade 4 (PDR) | Imbalance Ratio (Max / Min) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **APTOS 2019** | 3,662 | 1,805 (49.3%) | 370 (10.1%) | 999 (27.3%) | 193 (5.3%) | 295 (8.1%) | **9.35x** |
| **DDR Dataset** | 12,522 | 6,266 (50.0%) | 630 (5.0%) | 4,477 (35.8%) | 236 (1.9%) | 913 (7.3%) | **26.55x** |
| **DeepDRiD** | 2,000 | 680 (34.0%) | 356 (17.8%) | 392 (19.6%) | 392 (19.6%) | 180 (9.0%) | **3.78x** |
| **IDRiD** | 516 | 168 (32.6%) | 25 (4.8%) | 168 (32.6%) | 93 (18.0%) | 62 (12.0%) | **6.72x** |
| **Messidor-2** | 1,744 | 1,017 (58.3%) | 270 (15.5%) | 347 (19.9%) | 75 (4.3%) | 35 (2.0%) | **29.06x** |
| **EyePACS** | 35,126 | 25,810 (73.5%) | 2,443 (7.0%) | 5,292 (15.1%) | 873 (2.5%) | 708 (2.0%) | **36.45x** |
| **COMBINED POOL** | **55,570** | **35,746 (64.33%)**| **4,094 (7.37%)** | **11,675 (21.01%)**| **1,862 (3.35%)** | **2,193 (3.95%)** | **19.20x** |

#### Clinical Analysis of Label Shift:
1. **Clinical Context Divergence (Screening vs. Tertiary Referral):**
   * **EyePACS (Kaggle)** reflects a wide-scale community screening initiative in the United States, resulting in Grade 0 dominating at **73.5%**, while vision-threatening Grades 3 and 4 account for only **2.5%** and **2.0%**, yielding an extreme imbalance ratio of **36.45x**.
   * In contrast, **IDRiD** was gathered at a specialized tertiary eye clinic (Nanded, India). Grades 3 and 4 reach **30.0%** (Grade 3: 18.0%, Grade 4: 12.0%), over 7 times higher than in EyePACS.
2. **Failure of Standard Cross-Entropy Optimization:**
   * When training standard classification networks (Softmax + Cross-Entropy) on the combined dataset (where Grade 0 is 64.33% and Grade 3 is only 3.35%), Grade 0 gradients dominate parameter updates. The model rapidly converges to a trivial local minimum: predicting majority classes (Grade 0 / Grade 2) to minimize loss, entirely failing on severe proliferative cases.

---

### 2.2 Covariate Shift / Feature Shift $P(X)$

Each dataset was acquired with differing optical systems, sensor resolutions, fields of view ($45^\circ$ vs $50^\circ$), and pupil conditions (Mydriatic vs. Non-mydriatic).

#### A. Spatial Resolution & Aspect Ratio Statistics:

| Dataset Name | Sampled N | Width: Min - Max | Width: Mean $\pm$ Std | Height: Min - Max | Height: Mean $\pm$ Std | Aspect Ratio (W/H) | Sensor Geometry Type |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **APTOS 2019** | 100 | 640 - 4288 | $1885.85 \pm 891.27$ | 480 - 2848 | $1457.92 \pm 511.53$ | $1.25 \pm 0.20$ | Optical sensor variance |
| **DDR Dataset**| 100 | 512 - 3264 | $816.35 \pm 682.29$ | 512 - 2448 | $733.67 \pm 493.25$ | $1.05 \pm 0.13$ | Tight square crop |
| **DeepDRiD** | 100 | 1734 - 2232 | $1820.26 \pm 125.76$| 1725 - 1984 | $1871.58 \pm 76.97$ | $0.97 \pm 0.05$ | Standardized square ($1:1$) |
| **IDRiD** | 100 | 4288 - 4288 | **$4288.00 \pm 0.00$**| 2848 - 2848 | **$2848.00 \pm 0.00$**| **$1.51 \pm 0.00$** | High-res Kowa DSLR ($3:2$) |
| **Messidor-2** | 100 | 512 - 512 | **$512.00 \pm 0.00$** | 512 - 512 | **$512.00 \pm 0.00$** | **$1.00 \pm 0.00$** | Fixed square ($1:1$) |
| **EyePACS** | 100 | 1792 - 4928 | $3651.20 \pm 872.23$| 1184 - 3264 | $2490.48 \pm 511.67$| $1.46 \pm 0.07$ | Wide-field sensor |

**Architectural Consequence:**
Downsampling a $4288 \times 2848$ image down to $224 \times 224$ or $512 \times 512$ physically obliterates subtle micro-lesions (microaneurysms spanning $15 - 50\,\mu\text{m}$, sub-$15\text{px}$ on raw images).

---

#### B. Illumination & Color Channel Distributions:

Extracted Grayscale brightness $\mu_{Gray}$, individual color channels $\mu_R, \mu_G, \mu_B \in [0, 255]$, and optical focus sharpness $\sigma^2(\nabla^2 I)$ (Laplacian focus score):

| Dataset Name | Sampled N | Grayscale Brightness ($\mu_{Gray}$) | Red Channel ($\mu_R$) | Green Channel ($\mu_G$) | Blue Channel ($\mu_B$) | Focus Sharpness Score ($\sigma^2$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **APTOS 2019** | 100 | $68.75 \pm 17.82$ | $109.75 \pm 29.47$ | $57.78 \pm 15.78$ | **$17.55 \pm 14.03$** | $26.25 \pm 17.97$ |
| **DDR Dataset**| 100 | $71.91 \pm 24.54$ | $103.03 \pm 30.04$ | $63.66 \pm 24.22$ | $32.66 \pm 21.35$ | **$133.43 \pm 127.84$**|
| **DeepDRiD** | 100 | $67.88 \pm 25.14$ | $95.66 \pm 32.74$ | $59.84 \pm 23.28$ | **$36.40 \pm 21.51$** | $43.00 \pm 28.29$ |
| **IDRiD** | 100 | $67.86 \pm 12.98$ | $112.33 \pm 18.12$ | $54.87 \pm 12.91$ | **$18.29 \pm 13.76$** | $22.04 \pm 6.18$ |
| **Messidor-2** | 100 | $70.18 \pm 18.61$ | $117.35 \pm 30.09$ | $55.86 \pm 16.67$ | $20.27 \pm 8.59$ | $74.80 \pm 34.77$ |
| **EyePACS** | 100 | $59.97 \pm 27.47$ | $78.67 \pm 33.87$ | $54.75 \pm 26.38$ | **$37.78 \pm 23.77$** | **$14.70 \pm 14.81$** |

```
                              OPTICAL SPECTRAL ABSORPTION PROFILE
    Intensity (0-255)
    140 +--------------------------------------------------------------------+
        |   Red Channel (Red ~ 78-117): Strong Choroidal Reflection          |
    110 |   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ |
        |                                                                    |
     80 |   Green Channel (Green ~ 54-64): OPTIMAL HEMOGLOBIN ABSORPTION PEAK|
        |   ---------------------------------------------------------------- |
     50 |                                                                    |
        |   Blue Channel (Blue ~ 17-38): SEVERE SENSOR/FLASH DOMAIN SHIFT    |
     20 |   ................................................................ |
        +--------------------------------------------------------------------+
            APTOS        IDRiD      Messidor-2      DDR        EyePACS
          (Blue=17.5) (Blue=18.3)  (Blue=20.3)  (Blue=32.7)  (Blue=37.8)
```

#### Optical Analysis:
1. **Blue Channel Shift:** The Blue channel displays dramatic shifts: APTOS and IDRiD have $\mu_B \approx 17.5 - 18.3$, while EyePACS and DeepDRiD reach $\mu_B \approx 36.4 - 37.8$ (>210% variance). This stems from differing flash color temperatures (Xenon vs. LED) and sensor response curves.
2. **Green Channel (Pathological Signal Carrier):** Hemoglobin exhibits maximum optical absorption at $540 - 577\,\text{nm}$ (within the Green spectrum). Retinal hemorrhages and microaneurysms appear with highest contrast in the Green channel, whereas the Red channel is saturated by the retinal pigment epithelium (RPE).
3. **Sharpness & Real-World Artifacts:** Focus sharpness in EyePACS is very low ($14.70 \pm 14.81$) compared to DDR ($133.43 \pm 127.84$), reflecting community screening challenges: cataracts, pupil constriction, and corneal reflex artifacts.

---

## 3. Methodological Defense & System Architecture

Quantitative findings from the 55,570-image cohort establish the foundation for four core system design decisions:

```
+----------------------------------------------------------------------------------------------------+
|                                 FOUR EDA-BACKED DESIGN PILLARS                                     |
+----------------------------------------------------------------------------------------------------+
| 1. Preprocessing Standardization: Ben Graham Color Constancy + Green-Channel CLAHE                 |
| 2. Dual-Branch Multi-Scale Architecture: Global Context Branch + High-Res Local MIL Patches        |
| 3. Ordinal Regression Heads: Cumulative Link Model (CLM) / CORAL / CORN + Continuous QWK Loss      |
| 4. Out-of-Distribution Benchmark: Leave-One-Dataset-Out (LODO) Hold-out Protocol                  |
+----------------------------------------------------------------------------------------------------+
```

---

### 3.1 Preprocessing: Ben Graham Color Constancy & Green-Channel CLAHE

#### Problem:
Variations in $\mu_B$ ($17.5$ to $37.8$) and non-uniform illumination gradients (bright center, dark periphery) cause CNNs to latch onto camera color profiles rather than clinical pathology.

#### Formulation:
1. **Ben Graham Color Normalization:**
   $$\hat{I}(x, y) = \alpha \cdot I(x, y) - \beta \cdot \text{Gaussian}(I(x, y); \sigma) + \gamma$$
   Where $\alpha = 4$, $\beta = 4$, $\gamma = 128$, blur radius $\sigma = \text{scale}/30$.
   *Subtracting the low-pass Gaussian estimate removes non-uniform illumination gradients while preserving high-frequency microaneurysms and vessel edges.*
2. **Green-Channel Contrast Enhancement (CLAHE):**
   Applying CLAHE (`clipLimit=2.0`, `tileGridSize=(8,8)`) directly to the Green channel maximizes vascular contrast without amplifying background camera noise.

---

### 3.2 Architectural Defense: Dual-Branch Multi-Scale MIL

#### Problem:
Downsampling whole fundus images to $224 \times 224$ obliterates focal microaneurysms ($\le 0.01\%$ image area). Conversely, training whole images at $4288 \times 2848$ causes GPU memory exhaustion (OOM) and loses macro-retinal 4-quadrant spatial context.

#### Dual-Branch Solution:
```
                      +------------------------------------------+
                      |         Raw Fundus Image (CFP)           |
                      +------------------------------------------+
                                    /              \
     [Global Context Branch]       /                \    [Local MIL Branch]
                                  v                  v
                      +--------------------+   +------------------------------------+
                      | Downsample 512x512 |   | Sliding-Window MIL Patches (p=128, s=48) |
                      +--------------------+   +------------------------------------+
                                  |                              |
                                  v                              v
                      +--------------------+   +------------------------------------+
                      | ResNet50 + CBAM    |   | EfficientNet-B0 Patch Extractor    |
                      +--------------------+   +------------------------------------+
                                  |                              |
                                  |                      +---------------+
                                  |                      | Gated-Attention|
                                  |                      | MIL Aggregator|
                                  |                      +---------------+
                                  |                              |
                                  v                              v
                              [f_global]                     [f_local]
                                  \                              /
                                   \                            /
                                    v                          v
                                 +--------------------------------+
                                 | Concatenation: [f_g, f_l]      |
                                 +--------------------------------+
                                                 |
                                                 v
                                 +--------------------------------+
                                 | Ordinal Head (CLM_QWK / CORAL) |
                                 +--------------------------------+
```
* **Global Context Branch:** `ResNet-50 + CBAM` operates on $512 \times 512$ inputs to capture macular context and 4-quadrant hemifield lesion distribution ($F_{\text{Global}} \in \mathbb{R}^{2048}$).
* **Local MIL Branch (Sliding-Window v2):** Extracts overlapping $128 \times 128$ patches at native resolution (stride $s=96$, retinal FOV filter $\ge 0.5$). `EfficientNet-B0` extracts patch embeddings, aggregated via **Dual Pooling** (1D Max-Pooling + Masked Gated Attention Pooling) ($F_{\text{Local}} \in \mathbb{R}^{2560}$):
  $$F_{\text{Max}} = \max_{k \in \{1 \dots N\}} f_{\text{patch}}^{(k)}, \quad F_{\text{Att}} = \sum_{k=1}^N a_k h_k, \quad F_{\text{Local}} = [F_{\text{Max}} \,\|\, F_{\text{Att}}]$$
* Feature concatenation $F_{\text{Fused}} = [F_{\text{Global}} \,\|\, F_{\text{Local}}] \in \mathbb{R}^{4608}$ projected through a LayerNorm MLP to $512\text{-d}$ harmonizes macro-structural context and micro-lesion visibility.

---

### 3.3 Ordinal Regression Heads & Continuous QWK Loss

#### Problem:
Diabetic retinopathy grades possess a **Natural Clinical Hierarchy**:
$$\text{Grade } 0 < \text{Grade } 1 < \text{Grade } 2 < \text{Grade } 3 < \text{Grade } 4$$
Standard multi-class Cross-Entropy treats classes as unordered and independent:
$$\mathcal{L}_{\text{CE}}(\hat{y}=1, y=0) = \mathcal{L}_{\text{CE}}(\hat{y}=4, y=0)$$
Mispredicting Normal (Grade 0) as Mild (Grade 1) is a tolerable screening discrepancy, but mispredicting Grade 0 as Proliferative (Grade 4) is a critical clinical error.

#### Mathematical Solutions:
1. **Cumulative Link Models (CLM):**
   Projects features onto a scalar severity score $z(x) = w^T f(x)$ with $K-1 = 4$ strictly ordered monotonic cutpoints $\theta_1 < \theta_2 < \theta_3 < \theta_4$:
   $$P(Y \le k \mid x) = \sigma(\theta_k - z(x)) = \frac{1}{1 + e^{-(\theta_k - z(x))}}$$
   $$P(Y = k \mid x) = P(Y \le k \mid x) - P(Y \le k-1 \mid x)$$
2. **Continuous Differentiable QWK Loss:**
   Approximates Cohen's Kappa via soft observed ($\tilde{O}_{ij}$) and expected ($E_{ij}$) confusion matrices:
   $$\tilde{O}_{ij} = \sum_{n=1}^N y_{n, i} \cdot \hat{y}_{n, j}$$
   $$\mathcal{L}_{\text{QWK}} = \frac{\sum_{i,j} W_{ij} \tilde{O}_{ij}}{\sum_{i,j} W_{ij} E_{ij}}, \quad W_{ij} = \frac{(i - j)^2}{(C - 1)^2}$$
   Direct minimization of $\mathcal{L}_{\text{QWK}}$ penalizes off-diagonal predictions quadratically, counteracting class imbalance.

---

### 3.4 Out-of-Distribution Evaluation: Leave-One-Dataset-Out (LODO)

#### Problem:
Standard k-Fold cross-validation over pooled multi-source data mixes images from the same camera/clinic across splits, encouraging in-domain shortcut learning.

#### LODO Protocol:
To prove true cross-center generalization (Out-of-Distribution Generalizability):
* **Fold 1:** Train on {APTOS, DDR, DeepDRiD, Messidor-2, EyePACS} $\to$ Evaluate zero-shot on **IDRiD**.
* **Fold 2:** Train on {APTOS, DDR, DeepDRiD, IDRiD, EyePACS} $\to$ Evaluate zero-shot on **Messidor-2**.
* **Fold 3:** Train on {APTOS, DDR, DeepDRiD, IDRiD, Messidor-2} $\to$ Evaluate zero-shot on **APTOS**.

A model achieving $\text{QWK} \ge 0.85$ on completely unseen external cohorts demonstrates genuine clinical deployment viability.

---

## 4. Master Dataset Profile & Specifications

| Metric / Specification | **APTOS 2019** | **DDR Dataset** | **DeepDRiD** | **IDRiD** | **Messidor-2** | **EyePACS (Kaggle)** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Origin / Country** | India | China | China | India | France | United States |
| **Clinical Setting** | Aravind Eye Hospital | Multi-hospital network | Shanghai Univ. Hospital | Tertiary Eye Clinic | 3 French Univ. Hospitals | Telehealth Screening |
| **Clinical Context** | Screening & Outpatient | Multi-tier clinic | Specialized Clinic | Tertiary Referral Center | Academic Multi-Center | Community Screening |
| **Total Valid Images** | **3,662** | **12,522** | **2,000** | **516** (Combined) | **1,744** | **35,126** |
| **Unique Patients** | 3,662 individuals | 12,522 individuals | 500 individuals (4 views)| 516 individuals | 1,744 individuals | 35,126 individuals (Paired) |
| **Native Resolution** | $1886 \times 1458$ (Var) | $816 \times 734$ (Crop) | $1820 \times 1872$ (Square)| **$4288 \times 2848$ (SLR $3:2$)**| **$512 \times 512$ (Square $1:1$)**| $3651 \times 2490$ (Wide-field) |
| **Aspect Ratio (W/H)**| $1.25 \pm 0.20$ | $1.05 \pm 0.13$ | $0.97 \pm 0.05$ | $1.51 \pm 0.00$ | $1.00 \pm 0.00$ | $1.46 \pm 0.07$ |
| **Camera Models** | Diverse clinical | Diverse clinical | Topcon TRC-NW400 | Kowa VX-10 $\alpha$ | Topcon TRC-NW6 | Diverse field screening |
| **Field of View (FOV)**| $45^\circ$ | $45^\circ$ | $45^\circ$ (Central/Macular) | $50^\circ$ | $45^\circ$ | $45^\circ$ |
| **Laterality Labels** | Unspecified | Unspecified | Available (999 L / 1,001 R)| Unspecified | Available via Pairing | Available (17,563 L / R) |
| **Grayscale Mean ($\mu$)**| $68.75 \pm 17.82$ | $71.91 \pm 24.54$ | $67.88 \pm 25.14$ | $67.86 \pm 12.98$ | $70.18 \pm 18.61$ | $59.97 \pm 27.47$ |
| **Blue Mean ($\mu_B$)** | **$17.55 \pm 14.03$ (Low)**| $32.66 \pm 21.35$ | **$36.40 \pm 21.51$ (High)**| **$18.29 \pm 13.76$ (Low)** | $20.27 \pm 8.59$ | **$37.78 \pm 23.77$ (High)** |
| **Focus Score ($\sigma^2$)**| $26.25 \pm 17.97$ | **$133.43 \pm 127.84$** | $43.00 \pm 28.29$ | $22.04 \pm 6.18$ | $74.80 \pm 34.77$ | **$14.70 \pm 14.81$ (Low/Noise)**|
| **Imbalance Characteristics**| Grade 0 (49.3%), G2 (27.3%)| Grade 0 (50.0%), G2 (35.8%)| Grade 0 (34.0%), G1-3 (~19%)| **Grade 3+4 Highest (30.0%)** | Grade 0 Dominant (58.3%) | **Grade 0 Extreme (73.5%)** |
| **Imbalance Ratio** | **9.35x** | **26.55x** | **3.78x** | **6.72x** | **29.06x** | **36.45x** |
| **Pipeline Role** | Training & LODO Fold | Large-scale Training Pool | Paired View Evaluation | **External Hold-out & XAI** | **External Hold-out** | Large-scale Training Pool |

---

## 5. Conclusion & Implementation Roadmap

The exploratory data audit establishes the empirical and biomedical foundations for pipeline implementation:

1. **Preprocessing Pipeline ([preprocessing.py](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/full/preprocessing.py)):**
   * Encapsulates automated circular retinal ROI detection, Ben Graham illumination normalization, and Green-Channel CLAHE.
2. **Data Loader & Sterile Splitting ([data_loader.py](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/full/data_loader.py)):**
   * Implements multi-modal dual-stream dataset handling synchronized global context and local native patch sampling, with patient-level sterile grouping.
3. **Dual-Branch Backbone Architecture ([models.py](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/full/models.py)):**
   * Integrates `ResNet-50 + CBAM` global context extraction with `EfficientNet-B0 + Gated Attention MIL` micro-lesion analysis.
4. **Ordinal Classification Heads ([losses_and_heads.py](file:///d:/Github/deep-learning-for-computer-vision/notebooks/diabetic_retinopathy/pipeline/full/losses_and_heads.py)):**
   * Implements and evaluates `Softmax_CE`, `Softmax_QWK`, `CORAL`, `CORN`, and the champion `CLM_QWK` head under standardized optimization.
5. **XAI Quantitative Validation:**
   * Evaluates Grad-CAM visual attention maps against ground-truth expert lesion masks on IDRiD via Mean IoU and Pointing Game metrics.
