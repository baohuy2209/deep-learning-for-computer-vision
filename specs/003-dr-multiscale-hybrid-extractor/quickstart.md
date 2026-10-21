# Quickstart Validation Guide: Multi-Scale All-CNN Hybrid Extractor

**Feature**: Multi-Scale All-CNN Hybrid Extractor for Diabetic Retinopathy Grading
**Branch**: `003-dr-multiscale-hybrid-extractor`
**Date**: 2026-08-11

---

## 1. Overview & Objectives

This quickstart guide provides step-by-step instructions to validate the end-to-end functionality of the Multi-Scale All-CNN Hybrid Extractor model, custom loss function, and smart crop preprocessing utility.

### Key Verification Scenarios

1. **Model Instantiation & Shape Validation**: Verify global image `[B, 3, 224, 224]` and local patches `[B, K, 3, 128, 128]` produce `[B, 4]` CORAL logits.
2. **CORAL Logit to Class Probability Reconstruction**: Verify 4 cumulative logits map to 5 discrete probabilities summing to $1.0 \pm 1e-6$.
3. **Loss Computation & Gradient Backpropagation**: Verify composite loss $\alpha L_{\text{WCE}} + \beta L_{\text{Ordinal}} + \gamma L_{\text{QWK}}$ computes non-zero loss and successfully propagates backward gradients.
4. **Smart Crop Patch Extraction**: Verify $K=4$ non-overlapping $128 \times 128$ high-contrast patches are extracted from fundus images.

---

## 2. Prerequisites & Environment Setup

Ensure Python $\ge 3.10$ and PyTorch are available:

```bash
# Run model verification script directly
python src/models/multiscale_hybrid_extractor.py
```

---

## 3. Step-by-Step Validation Procedure

### Step A: Instantiate Model and Forward Pass

```python
import torch
from src.models.multiscale_hybrid_extractor import MultiScaleHybridExtractor, OrdinalHead

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiScaleHybridExtractor(pretrained=False, num_classes=5).to(device)

# Prepare dummy inputs
img_global = torch.randn(2, 3, 224, 224, device=device)
patches_local = torch.randn(2, 4, 3, 128, 128, device=device)

# Forward pass
logits = model(img_global, patches_local)
class_probs = OrdinalHead.logits_to_class_probs(logits)

assert logits.shape == (2, 4), f"Expected shape (2, 4), got {logits.shape}"
assert class_probs.shape == (2, 5), f"Expected shape (2, 5), got {class_probs.shape}"
print("✓ Model forward pass verified!")
```

### Step B: Verify Loss Function & Backpropagation

```python
from src.models.multiscale_hybrid_extractor import HybridOrdinalQWKLoss

targets = torch.tensor([1, 4], device=device, dtype=torch.long)
weights = torch.tensor([1.0, 2.0, 1.5, 3.0, 4.0], device=device)
criterion = HybridOrdinalQWKLoss(alpha=1.0, beta=1.0, gamma=1.0, class_weights=weights).to(device)

loss, l_wce, l_ordinal, l_qwk = criterion(logits, targets)

# Backpropagation check
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
optimizer.zero_grad()
loss.backward()
optimizer.step()

assert loss.item() > 0, "Loss value must be positive"
print(f"✓ Loss computation & backward pass verified! Total Loss: {loss.item():.4f}")
```

### Step C: Verify Smart Crop Utility

```python
import numpy as np
from PIL import Image
from src.models.multiscale_hybrid_extractor import smart_crop_patches

sample_img = Image.fromarray(np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8))
patches = smart_crop_patches(sample_img, K=4, patch_size=128)

assert len(patches) == 4, f"Expected 4 patches, got {len(patches)}"
assert patches[0].shape == (128, 128, 3), f"Expected shape (128, 128, 3), got {patches[0].shape}"
print("✓ Smart crop patch extraction verified!")
```

---

## 4. Expected Command Line Output

When executing `python src/models/multiscale_hybrid_extractor.py`, the terminal should output:

```text
======================================================================
Testing Multi-Scale All-CNN Hybrid Extractor Implementation...
======================================================================
Running test on device: cpu (or cuda)

1. Instantiating MultiScaleHybridExtractor...
   Input Global Image Shape:   torch.Size([2, 3, 224, 224])
   Input Local Patches Shape:  torch.Size([2, 4, 3, 128, 128])
   Input DR Target Grades:     [1, 4]

2. Executing Model Forward Pass...
   Output CORAL Logits Shape:  torch.Size([2, 4])
   Derived Class Probs Shape:  torch.Size([2, 5])
   Derived Class Probs Sums:   [1. 1.] (Should be 1.0)

3. Testing HybridOrdinalQWKLoss...
   Total Hybrid Loss:          7.0995
   - L_WCE Component:          6.1770
   - L_Ordinal Component:      0.7515
   - L_QWK Component:          0.1710

4. Verifying Backpropagation Gradient Flow...
   Backward pass successful! Parameter gradients computed without errors.

5. Testing Smart Crop Utility...
   Extracted Patches Count:    4
   Individual Patch Shape:     (128, 128, 3)

======================================================================
ALL TESTS PASSED SUCCESSFULLY! Model is fully functional and ready.
======================================================================
```
