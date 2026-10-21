# Quickstart: Executing Baseline Oracle Notebooks

**Feature**: `016-baseline-oracle-training`  
**Date**: 2026-09-22  
**Target Platform**: Google Colab (A100 / V100 GPU) & Local PyTorch Environment

---

## 1. Directory Structure

The 6 Oracle training notebooks will be generated in:
```text
notebooks/diabetic_retinopathy/pipeline/oracle/
├── oracle_aptos.ipynb         # APTOS 2019 in-domain baseline (3,662 images)
├── oracle_ddr.ipynb           # DDR in-domain baseline (12,522 images)
├── oracle_deepdrid.ipynb      # DeepDRiD in-domain baseline (2,000 images)
├── oracle_idrid.ipynb         # IDRiD in-domain baseline (516 images)
├── oracle_messidor2.ipynb     # Messidor-2 in-domain baseline (1,744 images)
└── oracle_eyepacs.ipynb       # EyePACS in-domain baseline (35,126 images)
```

---

## 2. Notebook Execution Workflow on Google Colab

Each notebook is completely self-contained and follows a standardized 8-step pipeline:

```text
[1. Colab Setup & Drive Mount] 
       ↓
[2. GPU & AMP Verification (A100 bfloat16)] 
       ↓
[3. Ingest Cohort Metadata & 70/15/15 Patient Split] 
       ↓
[4. Dual-Branch Dataset & Collate Fn (512x512 Global + 128x128 MIL)] 
       ↓
[5. Build Champion Model (ResNet50 + EfficientNet-B0 + CLM QWK Head)] 
       ↓
[6. DRTrainer Loop (40 Epochs, Batch 16, Patience 15)] 
       ↓
[7. Restore Best Checkpoint & Evaluate In-Domain QWK_oracle] 
       ↓
[8. Restore LODO Checkpoint & Compute Delta_transfer = QWK_oracle - QWK_OOD]
```

### Step 1: Open in Colab & Verify GPU
1. Open any of the 6 notebooks (e.g. `oracle_idrid.ipynb` for a fast initial benchmark).
2. Ensure Runtime is set to **GPU (A100 or T4)**:
   ```python
   !nvidia-smi
   ```

### Step 2: Google Drive Mounting & Image Root Setup
The notebook mounts Drive and verifies that image files are decompressed:
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 3: Train & Save Checkpoint
The notebook automatically trains the `DRFullModel` with:
- `epochs = 40`
- `batch_size = 16`
- `patience = 15` (monitored metric: `val_qwk`)
Checkpoints are automatically saved to:
`/content/drive/MyDrive/checkpoints_oracle/oracle_{cohort}_best.pth`

### Step 4: Automated Transfer Gap Computation
Once Oracle training finishes, the notebook loads the corresponding LODO fold checkpoint:
```python
delta_qwk = oracle_test_qwk - lodo_test_qwk
print(f"[RESULT] {cohort} In-Domain QWK: {oracle_test_qwk:.4f}")
print(f"[RESULT] {cohort} LODO OOD QWK  : {lodo_test_qwk:.4f}")
print(f"[TRANSFER GAP] Delta_transfer   : {delta_qwk:+.4f}")
```
This directly produces the verified numbers for Section 5 and Table 3 of the paper.
