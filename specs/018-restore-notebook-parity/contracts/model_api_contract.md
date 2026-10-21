# Python API Contract: `drdg` Model & Loss Interface

**Feature**: `018-restore-notebook-parity` | **Date**: 2026-09-24

This document defines the public Python API contracts exposed by the modular `drdg` package.

---

## 1. Model Builders (`drdg.models.variants`)

```python
from drdg.models.variants import build_variant

# Variants: 'v0', 'v1', 'v2', 'v3'
model = build_variant("v3", pretrained=False)
```

### Signature & Specs
```python
def build_variant(
    variant: str = "v3",
    pretrained: bool = True,
    num_classes: int = 5,
    global_backbone: str = "resnet50",
    local_backbone: str = "efficientnet_b0",
    **kwargs
) -> nn.Module
```

### Tensor Contracts
- **Global Input**: `x_global: torch.Tensor` with shape `(B, 3, 256, 256)` (or 512×512)
- **Local Input**: `x_local: torch.Tensor` with shape `(B, K_max, 3, 224, 224)`
- **Patch Mask**: `patch_mask: torch.Tensor` with shape `(B, K_max)` (boolean or float 0/1)
- **Forward Call**: `outputs = model(x_global, x_local, patch_mask)`
- **Forward Output**: Dictionary containing:
  - `logits`: `torch.Tensor` shape `(B, num_classes)` or `(B, num_classes - 1)` for ordinal heads
  - `probabilities`: `torch.Tensor` shape `(B, num_classes)` (summing to 1.0)
  - `global_features`: `torch.Tensor` shape `(B, D_global)`
  - `local_features`: `torch.Tensor` shape `(B, D_local)`
  - `attention_weights`: Optional local patch attention distribution `(B, K_max)`

---

## 2. Loss Functions (`drdg.training.losses`)

### Continuous QWK Loss
```python
from drdg.training.losses import ContinuousQWKLoss

qwk_loss_fn = ContinuousQWKLoss(num_classes=5, epsilon=1e-8)
loss = qwk_loss_fn(probabilities, targets)
```
- `probabilities`: Softmax/ordinal probabilities of shape `(B, 5)`, float32
- `targets`: Ground-truth integer labels `(B,)` in $\{0, 1, 2, 3, 4\}$
- `output`: Scalar differentiable loss tensor

### Cumulative Link Model (CLM) + QWK Loss
```python
from drdg.training.losses import CumulativeLinkLoss

clm_loss_fn = CumulativeLinkLoss(qwk_weight=0.5, link="probit")
loss = clm_loss_fn(model_outputs, targets)
```

---

## 3. Dataset & Collation (`drdg.data`)

```python
from drdg.data.dataset import FundusDualBranchDataset
from drdg.data.collation import collate_dual_branch

dataset = FundusDualBranchDataset(
    csv_path="data/processed/lodo_folds/fold_0_aptos_train.csv",
    img_root="data/raw/hybrid/native_cropped",
    img_size=256,
    patch_size=224,
    is_training=True,
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=collate_dual_branch,
)
```

Each batch yields:
```python
{
    "global_image": torch.Tensor,       # (B, 3, 256, 256)
    "patches": torch.Tensor,            # (B, K_max, 3, 224, 224)
    "patch_mask": torch.Tensor,         # (B, K_max)
    "diagnosis": torch.Tensor,          # (B,) int64
    "patient_id": List[str],            # Length B
    "image_id": List[str],              # Length B
    "dataset_key": List[str],           # Length B
}
```
