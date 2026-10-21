# Interface Contract: DRTrainer Operational API

**Feature**: `019-sync-notebook-training-pipelines`  
**Contract Version**: `1.0.0`  
**Applicability**: `DRTrainer` in `main_model (6).ipynb`, `main_model_lodo.ipynb`, and `DRHeadTrainer` in `main_model_head_ablation (3).ipynb`.

---

## 1. Class Instantiation: `__init__`

```python
def __init__(
    self,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Optional[Any] = None,
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    use_amp: bool = True,
    class_weights: Optional[torch.Tensor] = None,
    grad_clip: float = 1.0,
    checkpoint_dir: str = "./checkpoints",
    qwk_warmup_epochs: int = 15,
) -> None:
```

### Preconditions:
- `model` must be an initialized PyTorch `nn.Module`.
- `optimizer` must be configured with `model.parameters()`.
- `device` must be resolvable via `torch.device(device)`.

### Postconditions:
- `self.model` is moved to `self.device`.
- Hardware AMP precision is auto-detected:
  - If CUDA supports `bfloat16` (`torch.cuda.is_bf16_supported()`): `amp_dtype = torch.bfloat16`, `use_scaler = False`.
  - Else: `amp_dtype = torch.float16`, `use_scaler = self.use_amp`.
- Normalization constants `mean_global`, `std_global`, `mean_local`, `std_local` are registered on device.
- `checkpoint_dir` directory is guaranteed to exist (`os.makedirs(checkpoint_dir, exist_ok=True)`).

---

## 2. Training Epoch: `train_epoch`

```python
def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
```

### Input:
- `dataloader`: PyTorch `DataLoader` yielding dictionary batches with:
  - Dual-Branch: `global_img` (Tensor), `local_patches` (Tensor), `label` (LongTensor), and optional `patch_mask`.
  - Single-Branch (Head Ablation): `image` (Tensor), `label` (LongTensor).

### Behavior:
1. Puts `self.model.train()`.
2. Converts `uint8` image tensors to `float32` on-GPU and normalizes with ImageNet constants.
3. Performs forward pass under `torch.amp.autocast(self.device_type, dtype=self.amp_dtype, enabled=self.use_amp)`.
4. Computes multi-task loss ($\mathcal{L}_{\text{NLL}} + \lambda(t) \cdot \mathcal{L}_{\text{Continuous\_QWK}}$).
5. Executes backward pass with GradScaler or direct `backward()`, followed by gradient clipping (`clip_grad_norm_`).
6. Computes training metrics via `calculate_dr_metrics`.

### Output:
- Returns a dictionary containing `"Loss"` and `"QWK"`.

---

## 3. Evaluation: `evaluate`

```python
@torch.no_grad()
def evaluate(self, dataloader: DataLoader) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
```

### Behavior:
1. Puts `self.model.eval()`.
2. Computes evaluation loss and collects all targets, predictions, probability distributions, and rank probabilities.
3. Computes clinical metrics: QWK, accuracy, within-1 accuracy, referable sensitivity/specificity, per-grade recall, and rank monotonicity violations.

### Output Tuple:
1. `metrics`: `Dict[str, Any]` (containing all clinical metrics, `Loss`, `probs`, and `Monotonicity_Violations`).
2. `y_true`: `np.ndarray` of ground-truth integer labels $[0, 4]$.
3. `y_pred`: `np.ndarray` of predicted integer classes $[0, 4]$.

---

## 4. Main Training Orchestration: `fit`

```python
def fit(
    self,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: Optional[DataLoader] = None,
    epochs: int = 100,
    model_name: str = "dr_model",
    patience_early_stopping: int = 15,
    patience_reduce_lr: int = 4,
    reduce_lr_factor: float = 0.5,
    min_lr: float = 1e-6,
    resume_checkpoint_path: Optional[str] = None
) -> Dict[str, Any]:
```

### Invariant Rules:
1. **Dynamic QWK Warmup**: At each epoch $t \in [1, \text{epochs}]$, set `head.set_qwk_weight(min(1.0, t / qwk_warmup_epochs) * head.target_qwk_weight)`.
2. **Decoupled Dual Monitoring**:
   - Save `_best.pth` and reset `epochs_no_improve_es = 0` **if and only if** `val_qwk > best_val_qwk`.
   - Decay LR by `reduce_lr_factor` **if and only if** `val_loss >= best_val_loss - 1e-4` for `patience_reduce_lr` epochs.
3. **Continuous Recovery**: Save `_last.pth` on every single epoch with full optimizer, scaler, scheduler, and decoupled counters.
4. **Early Termination**: Break loop when `epochs_no_improve_es >= patience_early_stopping`.
5. **Restoration on Held-out Test**: If `test_loader` is provided, restore best weights from `_best.pth` before evaluating test metrics.

### Return Dictionary:
```python
{
    "best_val_qwk": float,
    "best_checkpoint": str,
    "history": Dict[str, List[float]],
    "test_metrics": Dict[str, Any]
}
```
