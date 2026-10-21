# Python API Interface Contract: Multi-Scale All-CNN Hybrid Extractor

**Feature**: Multi-Scale All-CNN Hybrid Extractor for Diabetic Retinopathy Grading
**Branch**: `003-dr-multiscale-hybrid-extractor`
**Date**: 2026-08-11

---

## 1. Class: `MultiScaleHybridExtractor`

Module path: `src.models.multiscale_hybrid_extractor.MultiScaleHybridExtractor`

```python
class MultiScaleHybridExtractor(nn.Module):
    def __init__(self, pretrained: bool = True, num_classes: int = 5):
        """
        Initializes the MultiScaleHybridExtractor neural network.

        Args:
            pretrained (bool): Whether to use ImageNet pre-trained weights for backbones. Default: True.
            num_classes (int): Number of ordinal classes (5 for DR grades 0..4). Default: 5.
        """
        ...

    def forward(self, img_global: torch.Tensor, patches_local: torch.Tensor) -> torch.Tensor:
        """
        Executes multi-scale feature extraction and ordinal classification.

        Args:
            img_global (torch.Tensor): Tensor of shape [Batch, 3, 224, 224].
            patches_local (torch.Tensor): Tensor of shape [Batch, K, 3, 128, 128].

        Returns:
            torch.Tensor: CORAL binary logits of shape [Batch, 4].
        """
        ...
```

---

## 2. Class: `OrdinalHead`

Module path: `src.models.multiscale_hybrid_extractor.OrdinalHead`

```python
class OrdinalHead(nn.Module):
    def __init__(self, in_features: int = 512, num_classes: int = 5):
        ...

    @staticmethod
    def logits_to_class_probs(logits: torch.Tensor) -> torch.Tensor:
        """
        Converts 4 CORAL binary logits into 5 discrete nominal class probabilities.

        Args:
            logits (torch.Tensor): Tensor of shape [Batch, 4].

        Returns:
            torch.Tensor: Class probabilities of shape [Batch, 5] summing to 1.0.
        """
        ...
```

---

## 3. Class: `HybridOrdinalQWKLoss`

Module path: `src.models.multiscale_hybrid_extractor.HybridOrdinalQWKLoss`

```python
class HybridOrdinalQWKLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
        num_classes: int = 5
    ):
        """
        Composite loss module combining Weighted Cross-Entropy, BCE over rank logits, and Soft QWK.

        Args:
            alpha (float): Weight multiplier for L_WCE. Default: 1.0.
            beta (float): Weight multiplier for L_Ordinal. Default: 1.0.
            gamma (float): Weight multiplier for L_QWK. Default: 1.0.
            class_weights (Optional[torch.Tensor]): Class weights tensor of shape [5].
            num_classes (int): Number of ordinal classes (5).
        """
        ...

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates loss values.

        Args:
            logits (torch.Tensor): CORAL binary logits of shape [Batch, 4].
            targets (torch.Tensor): Ground truth DR grades of shape [Batch] (values 0..4).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                (total_loss, l_wce, l_ordinal, l_qwk)
        """
        ...
```

---

## 4. Function: `smart_crop_patches`

Module path: `src.models.multiscale_hybrid_extractor.smart_crop_patches`

```python
def smart_crop_patches(
    image: Union[Image.Image, np.ndarray],
    K: int = 4,
    patch_size: int = 128
) -> List[np.ndarray]:
    """
    Extracts K high-contrast patches based on local intensity variance.

    Args:
        image (Union[Image.Image, np.ndarray]): Input fundus image.
        K (int): Number of patches to extract. Default: 4.
        patch_size (int): Square patch dimensions (height and width). Default: 128.

    Returns:
        List[np.ndarray]: List of K cropped NumPy array patches [128, 128, 3].
    """
    ...
```
