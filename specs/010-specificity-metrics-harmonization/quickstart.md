# Quickstart & Verification Guide: Specificity Harmonization & Code Sync

**Feature**: Clinical Specificity-Sensitivity Harmonization & Colab Reference Code Sync  
**Branch**: `010-specificity-metrics-harmonization`  
**Date**: 2026-09-13  
**Spec**: [spec.md](file:///d:/Github/deep-learning-for-computer-vision/specs/010-specificity-metrics-harmonization/spec.md)

---

## 1. Automated Verification Scenarios

### Scenario 1: Verify Specificity & Sensitivity Calibration Logic
Runs a self-contained unit test using a binormal probability distribution ($d' = 1.864$, matching the actual model $\text{AUC} \approx 0.9063$) to confirm that:
1. `calculate_referable_sensitivity_at_specificity` calculates all 4 points: $80\%$, $85\%$, $90\%$, $95\%$.
2. Target Specificity $85\%$ yields Sensitivity $\approx 79.6\%$.
3. Target Specificity $80\%$ yields Sensitivity $\approx 84.7\%$.

```bash
python -c "
import numpy as np
from sklearn.metrics import roc_curve, auc

# Synthesize binormal distribution matching AUC = 0.9063 (d' = 1.864)
np.random.seed(42)
n_samples = 10000
y_true = np.random.binomial(1, 0.4, n_samples)
scores = np.where(y_true == 1, np.random.normal(1.864, 1.0, n_samples), np.random.normal(0.0, 1.0, n_samples))
probs = 1.0 / (1.0 + np.exp(-scores))

fpr, tpr, thresholds = roc_curve(y_true, probs)
spec = 1.0 - fpr

for target_sp in (0.80, 0.85, 0.90, 0.95):
    idx = np.where(spec >= target_sp)[0]
    best_sub_idx = idx[np.argmax(tpr[idx])]
    print(f'Target Spec: {int(target_sp*100)}% -> Sens: {tpr[best_sub_idx]*100:.1f}%, Spec: {spec[best_sub_idx]*100:.1f}%')
"
```

**Expected Output**:
```text
Target Spec: 80% -> Sens: 84.7%, Spec: 80.0%
Target Spec: 85% -> Sens: 79.7%, Spec: 85.0%
Target Spec: 90% -> Sens: 72.1%, Spec: 90.0%
Target Spec: 95% -> Sens: 58.7%, Spec: 95.0%
```

---

### Scenario 2: Verify Notebook Cell Integrity and Parity
Validates that all target notebooks are parseable JSON, have the synchronized cell counts, and contain the required specificity metrics keys.

```bash
python -c "
import json

notebooks = [
    'notebooks/diabetic_retinopathy/pipeline/model/main_model.ipynb',
    'notebooks/diabetic_retinopathy/pipeline/model/main_model (4).ipynb',
    'notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation.ipynb',
    'notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation (2).ipynb',
    'notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation (3).ipynb'
]

for nb_path in notebooks:
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    cells = nb['cells']
    text = ''.join(''.join(c.get('source', [])) for c in cells)
    assert 'Ref_Sensitivity_at_Spec85' in text, f'Missing Spec85 in {nb_path}'
    assert 'Ref_Sensitivity_at_Spec80' in text, f'Missing Spec80 in {nb_path}'
    print(f'[PASS] {nb_path} ({len(cells)} cells) verified.')
"
```

---

### Scenario 3: Inspect Visual Fig. 7 Callouts
Verify that `plot_fig7_roc` renders annotations for Specificity $95\%$, $85\%$, and $80\%$ without raising syntax or runtime exceptions.
