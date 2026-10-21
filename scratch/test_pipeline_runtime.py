import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# Extract code from the notebook to ensure the notebook code itself runs in Python runtime
import json

with open("notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Combine cells 2, 3, 4, 6, 7, 8, 10, 11, 13, 14, 15, 16, 18
# This tests every single component of the pipeline
test_cells = [2, 3, 4, 6, 7, 8, 10, 11, 13, 14, 15, 16, 18]
combined_code = ""
for idx in test_cells:
    combined_code += f"\n# --- CELL {idx} ---\n"
    combined_code += "".join(nb["cells"][idx]["source"])

exec_namespace = {}
exec(combined_code, exec_namespace)

print("All classes and functions imported and compiled successfully!")

# Test 1: Model instantiation via build_model
build_model = exec_namespace["build_model"]
model = build_model(
    global_backbone_name="resnet34",
    local_backbone_name="resnet18",
    num_classes=5,
    freeze_backbone=False,
    dropout=0.3
)
model.eval()
print("Model created successfully:", type(model).__name__)

# Test 2: Dummy forward pass
bs = 2
n_patches = 8
dummy_global = torch.randn(bs, 3, 512, 512)
dummy_patches = torch.randn(bs, n_patches, 3, 128, 128)
dummy_mask = torch.ones(bs, n_patches, dtype=torch.bool)

with torch.no_grad():
    outputs = model(dummy_global, dummy_patches, dummy_mask)

print("Forward pass successful!")
print("Output keys:", outputs.keys())
print("Global logits shape:", outputs["global_logits"].shape)
print("Local logits shape:", outputs["local_logits"].shape)
print("Fusion logits shape:", outputs["fusion_logits"].shape)
print("Cumulative probs shape:", outputs["cumulative_probs"].shape)
print("Ordinal preds shape:", outputs["ordinal_preds"].shape)

# Test 3: Multi-task loss computation
MultiTaskRetinopathyLoss = exec_namespace["MultiTaskRetinopathyLoss"]
criterion = MultiTaskRetinopathyLoss(alpha=0.5, beta=0.5, ordinal_weight=0.7, task_weights=(0.5, 0.5))

dummy_labels = torch.tensor([0, 3], dtype=torch.long)
loss, loss_dict = criterion(outputs, dummy_labels)
print(f"Loss computation successful! Total loss: {loss.item():.4f}")
for k, v in loss_dict.items():
    print(f"  {k}: {v:.4f}")

# Test 4: Threshold calibration
calibrate_thresholds = exec_namespace["calibrate_thresholds"]
calc_qwk = exec_namespace["calc_qwk"]

y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4] * 5)
continuous_preds = y_true + np.random.normal(0, 0.3, len(y_true))
best_thresh, opt_qwk = calibrate_thresholds(y_true, continuous_preds)
print("Calibrate thresholds successful!")
print(f"Optimal thresholds: {best_thresh}")
print(f"Optimized QWK: {opt_qwk:.4f}")

# Test 5: Check Collate MIL Patches
collate_mil_patches = exec_namespace["collate_mil_patches"]
mock_batch = [
    {
        "global_image": torch.randn(3, 512, 512),
        "local_native_image": np.zeros((1024, 1024, 3), dtype=np.uint8),
        "valid_anchors": np.array([[0, 0], [96, 96], [192, 192]]),
        "native_path": "test_img_1.jpg",
        "label": torch.tensor(2, dtype=torch.long)
    },
    {
        "global_image": torch.randn(3, 512, 512),
        "local_native_image": np.zeros((800, 800, 3), dtype=np.uint8),
        "valid_anchors": np.array([[0, 0], [96, 96]]),
        "native_path": "test_img_2.jpg",
        "label": torch.tensor(0, dtype=torch.long)
    }
]

collated = collate_mil_patches(mock_batch, max_patches=16, patch_size=128)
print("Collate MIL patches successful!")
print("Collated global_img:", collated["global_img"].shape)
print("Collated local_patches:", collated["local_patches"].shape)
print("Collated patch_mask:", collated["patch_mask"].shape)
print("Collated labels:", collated["label"])

print("\nALL PIPELINE DRY-RUN TESTS PASSED WITH 100% SUCCESS!")
