# Dataset Setup & Preparation Guide

This guide provides step-by-step instructions for downloading and setting up the **6 multi-center retinal fundus datasets** used in the paper:
*Deconstructing Domain Shift in Diabetic Retinopathy Grading: Multi-Scale MIL with Consistent Ordinal Regression* (ICCPR).

---

## 1. Directory Structure

Place raw datasets into `data/raw/` adhering to the structure below:

```text
data/raw/
├── aptos/
│   ├── aptos_total.csv
│   └── aptos_total_images/
├── ddr/
│   ├── DR_grading.csv
│   └── DR_grading/DR_grading/
├── deepdrid/
│   ├── deepdrid_total.csv
│   └── deepdrid_total_images/
├── idrid/
│   ├── idrid_total.csv
│   └── idrid_total_images/
├── messidor2preprocess/
│   ├── messidor_data.csv
│   └── messidor-2/messidor-2/preprocess/
└── zipEyepacs/
    ├── trainLabels.csv/trainLabels.csv
    └── train/
```

---

## 2. Public Dataset Sources

| Dataset | Sample Count | Primary Camera | Source URL |
|---|---|---|---|
| **APTOS 2019** | 3,662 | Various (India) | [Kaggle APTOS 2019](https://www.kaggle.com/c/aptos2019-blindness-detection) |
| **DDR** | 12,522 | Topcon TRC-NW400 | [DDR Dataset (GitHub)](https://github.com/nkicsl/DDR-dataset) |
| **DeepDRiD** | 2,000 | Topcon TRC-NW8 | [ISBI 2020 DeepDRiD](https://isbi.deepdr.org/) |
| **IDRiD** | 516 | Kowa VX-10alpha | [IEEE Dataport IDRiD](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) |
| **Messidor-2** | 1,744 | Topcon TRC-NW6 | [Messidor-2 (LaTIM)](http://www.adcis.net/en/third-party/messidor2/) |
| **EyePACS** | 35,126 | Centervue, Canon, etc. | [Kaggle Diabetic Retinopathy](https://www.kaggle.com/c/diabetic-retinopathy-detection) |

---

## 3. Automated Preprocessing

Run the automated ROI crop to remove uninformative black borders:

```bash
bash scripts/01_preprocess_data.sh
```

Images will be saved to `data/raw/hybrid/native_cropped/` at full native resolution with JPEG quality 95.
