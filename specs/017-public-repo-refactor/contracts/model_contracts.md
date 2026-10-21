# Model & Component Interface Contracts: Public Academic Repository Refactor

**Feature Branch**: `017-public-repo-refactor`
**Date**: 2026-09-23
**Status**: Completed

---

## 1. Unified Model Forward Pass Contract

All models (`DRFullModel`, `ResNet50AblationModel`, and variants V0 through V3) MUST return a standard Python dictionary with the following tensor keys:

```python
output = model(global_images, patches=None, patch_mask=None)
```

| Key | Tensor Shape | Type | Description |
| :--- | :--- | :--- | :--- |
| `logits` | `[B, 5]` (or `[B, 4]`) | `torch.FloatTensor` | Raw unnormalized outputs from the classification/rank head |
| `probabilities` | `[B, 5]` | `torch.FloatTensor` | Valid probability distribution across the 5 ICDRS grades ($\sum_k P_k = 1.0$) |
| `severity` | `[B]` | `torch.FloatTensor` | Continuous ordinal severity prediction ($E[Y] = \sum_k k \cdot P_k$ or latent score) |
| `predicted_grade` | `[B]` | `torch.LongTensor` | Discrete prediction in $\{0, 1, 2, 3, 4\}$ |
| `rdr_score` | `[B]` | `torch.FloatTensor` | Referable DR binary probability: $P(Y \ge 2) = \sum_{k=2}^4 P_k$ |
| `global_features` | `[B, 512]` | `torch.FloatTensor` | Latent representation extracted from the Global Context stream |
| `local_features` | `[B, 512]` (optional) | `torch.FloatTensor` | Latent representation extracted from the Local MIL stream |
| `mil_attention` | `[B, N_patches]` (opt) | `torch.FloatTensor` | Normalized attention weights $\alpha_i$ from GatedAttentionPool |

---

## 2. Stream Interfaces

### 2.1 `GlobalContextStream` (`src/drdg/models/streams/global_context.py`)
```python
class GlobalContextStream(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        use_cbam: bool = True,
        use_non_local: bool = True,
        use_quadrant_tokens: bool = True,
        feature_dim: int = 512,
    ): ...

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Global fundus images [B, 3, 512, 512]
        Returns:
            global_embedding: [B, feature_dim]
            spatial_feature_map: [B, 2048, H', W'] for Grad-CAM
        """
```

### 2.2 `LocalMILBranch` (`src/drdg/models/streams/local_mil.py`)
```python
class LocalMILBranch(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        feature_dim: int = 512,
    ): ...

    def forward(
        self,
        patches: torch.Tensor,
        patch_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            patches: High-res patch tensors [B, N_patches, 3, 224, 224]
            patch_mask: Boolean active patch mask [B, N_patches]
        Returns:
            bag_embedding: Aggregated representation [B, feature_dim]
            attention_weights: Normalized attention alpha [B, N_patches]
        """
```

---

## 3. Output Head Contract (`src/drdg/models/heads/`)

All five heads (`SoftmaxCEHead`, `SoftmaxQWKHead`, `CORALHead`, `CORNHead`, `CumulativeLinkModelQWKHead`) inherit from `BaseDRHead` and implement:

```python
class BaseDRHead(nn.Module, ABC):
    @abstractmethod
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: Fused feature representation [B, D]
        Returns:
            Dict containing 'logits', 'probabilities', 'severity', 'predicted_grade', 'rdr_score'
        """

    @abstractmethod
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predictions: Output dictionary from forward()
            targets: Integer ground-truth grades [B] in {0, 1, 2, 3, 4}
        Returns:
            Scalar loss tensor
        """
```
