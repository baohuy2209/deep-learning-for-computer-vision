"""
================================================================================
DIABETIC RETINOPATHY (DR) PIPELINE - LOSSES & ORDINAL OUTPUT HEADS
================================================================================
Core Research Focus: Ordinal Regression Output Head Ablation & Differentiable QWK Loss.
Supported Heads & Loss Formulations:
  1. Softmax_CE: Linear(5) + Standard/Weighted Cross-Entropy Loss.
  2. Softmax_QWK: Linear(5) + Cross-Entropy + Continuous Differentiable QWK Loss.
  3. CORAL: Consistent Rank Logits (Shared weight, independent threshold biases) + Ordinal BCE.
  4. CORN: Conditional Ordinal Regression Neural Network + Conditional Cross-Entropy.
  5. CLM_QWK (Champion): Cumulative Link Model (Complementary log-log link with monotonic
     thresholds) + Interval NLL Loss + Continuous QWK Loss.
  6. End-to-End Pipeline Wrapper: DRFullModel integrating DualBranchMILModel with any Head.
================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, Union


# ==============================================================================
# 1. CONTINUOUS DIFFERENTIABLE QUADRATIC WEIGHTED KAPPA (QWK) LOSS
# ==============================================================================

class ContinuousQWKLoss(nn.Module):
    """
    Differentiable approximation of Quadratic Weighted Kappa (QWK) Loss for PyTorch.
    Computes soft confusion matrix and soft quadratic penalty weighting:
      Loss = sum(W * O) / sum(W * E)
    """
    def __init__(self, num_classes: int = 5, eps: float = 1e-7):
        super(ContinuousQWKLoss, self).__init__()
        self.num_classes = num_classes
        self.eps = eps
        
        # Quadratic weight matrix W_ij = (i - j)^2 / (C - 1)^2
        w = torch.zeros((num_classes, num_classes), dtype=torch.float32)
        for i in range(num_classes):
            for j in range(num_classes):
                w[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)
        self.register_buffer("W", w)

    def forward(self, pred_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_probs: Softmax or cumulative probability distribution (B, num_classes).
            targets: Ground truth integer class labels (B,) or one-hot targets (B, num_classes).
        """
        device = pred_probs.device
        N = pred_probs.size(0)
        if N == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
            
        if targets.ndim == 1:
            y_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        else:
            y_one_hot = targets.float()
            
        # Ensure probabilities sum to 1
        pred_probs = pred_probs / (pred_probs.sum(dim=1, keepdim=True) + self.eps)
        
        # Soft observed confusion matrix: O = P^T * Y
        O = torch.matmul(pred_probs.t(), y_one_hot)  # (C, C)
        O = O / (N + self.eps)
        
        # Expected confusion matrix under independence: E = hist_pred * hist_true^T
        hist_pred = pred_probs.sum(dim=0, keepdim=True) / (N + self.eps)  # (1, C)
        hist_true = y_one_hot.sum(dim=0, keepdim=True) / (N + self.eps)   # (1, C)
        E = torch.matmul(hist_pred.t(), hist_true)                        # (C, C)
        
        # Quadratic weighted numerator and denominator
        W = self.W.to(device)
        num = torch.sum(W * O)
        den = torch.sum(W * E) + self.eps
        
        # QWK Loss = num / den (Minimizing this maximizes Cohen's Kappa)
        qwk_loss = num / den
        return qwk_loss


# ==============================================================================
# 2. OUTPUT HEAD 1: SOFTMAX CLASSIFICATION (CE)
# ==============================================================================

class SoftmaxCEHead(nn.Module):
    """
    Standard Categorical Classification Head:
      Linear(latent_dim, 5) -> Cross-Entropy Loss.
    """
    def __init__(self, in_features: int = 512, num_classes: int = 5):
        super(SoftmaxCEHead, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.fc(x)  # (B, 5)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return {"logits": logits, "probs": probs, "preds": preds}

    def loss_fn(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        logits = outputs["logits"]
        return F.cross_entropy(logits, targets, weight=class_weights)


# ==============================================================================
# 3. OUTPUT HEAD 2: SOFTMAX + CONTINUOUS QWK LOSS
# ==============================================================================

class SoftmaxQWKHead(nn.Module):
    """
    Softmax Classification Head trained with Hybrid Multitask Loss:
      Loss = Cross-Entropy + lambda * Continuous_QWK_Loss
    """
    def __init__(
        self,
        in_features: int = 512,
        num_classes: int = 5,
        qwk_weight: float = 1.0
    ):
        super(SoftmaxQWKHead, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)
        self.num_classes = num_classes
        self.qwk_weight = qwk_weight
        self.qwk_loss = ContinuousQWKLoss(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.fc(x)
        probs = F.softmax(logits, dim=1)
        # Expected continuous value for ordinal predictions
        continuous_pred = torch.sum(probs * torch.arange(self.num_classes, device=x.device), dim=1)
        preds = torch.round(continuous_pred).clamp(0, self.num_classes - 1).long()
        return {"logits": logits, "probs": probs, "preds": preds, "continuous_preds": continuous_pred}

    def loss_fn(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        logits = outputs["logits"]
        probs = outputs["probs"]
        
        ce_loss = F.cross_entropy(logits, targets, weight=class_weights)
        qwk = self.qwk_loss(probs, targets)
        
        total_loss = ce_loss + self.qwk_weight * qwk
        return total_loss


# ==============================================================================
# 4. OUTPUT HEAD 3: CORAL (CONSISTENT RANK LOGITS)
# ==============================================================================

class CORALHead(nn.Module):
    """
    Consistent Rank Logits (CORAL) for Ordinal Regression.
    Features:
      - Shared weight vector W across all rank thresholds.
      - Independent rank threshold biases b_1, b_2, ..., b_{K-1}.
      - Probability of exceeding rank k: P(y > k) = sigmoid(W^T x + b_k).
    """
    def __init__(self, in_features: int = 512, num_classes: int = 5):
        super(CORALHead, self).__init__()
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1  # 4 thresholds for 5 classes
        
        # Shared linear weight (without bias)
        self.linear = nn.Linear(in_features, 1, bias=False)
        # Independent threshold biases
        self.biases = nn.Parameter(torch.zeros(self.num_thresholds))
        # Initialize biases to strictly increasing values
        with torch.no_grad():
            self.biases.copy_(torch.linspace(-2.0, 2.0, self.num_thresholds))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        base_logit = self.linear(x)  # (B, 1)
        # Binary rank logits for each threshold: z_k = W^T x + b_k
        rank_logits = base_logit + self.biases  # (B, 4)
        rank_probs = torch.sigmoid(rank_logits)  # P(y > k), shape (B, 4)
        
        # Predict class by counting number of satisfied thresholds (threshold > 0.5 -> logit > 0)
        preds = torch.sum(rank_probs > 0.5, dim=1).long()  # [0..4]
        
        # Reconstruct class probabilities: P(y = k) = P(y > k-1) - P(y > k)
        B = x.size(0)
        probs = torch.zeros(B, self.num_classes, device=x.device)
        probs[:, 0] = 1.0 - rank_probs[:, 0]
        for k in range(1, self.num_thresholds):
            probs[:, k] = F.relu(rank_probs[:, k - 1] - rank_probs[:, k])
        probs[:, -1] = rank_probs[:, -1]
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-7)
        
        return {"rank_logits": rank_logits, "rank_probs": rank_probs, "probs": probs, "preds": preds}

    def loss_fn(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        rank_logits = outputs["rank_logits"]  # (B, 4)
        device = targets.device
        B = targets.size(0)
        
        # Encode targets into binary rank indicators: binary_targets[i, k] = 1 if target > k else 0
        binary_targets = torch.zeros(B, self.num_thresholds, device=device)
        for k in range(self.num_thresholds):
            binary_targets[:, k] = (targets > k).float()
            
        # Standard Ordinal Binary Cross-Entropy across all rank thresholds
        bce_loss = F.binary_cross_entropy_with_logits(rank_logits, binary_targets, reduction="mean")
        return bce_loss


# ==============================================================================
# 5. OUTPUT HEAD 4: CORN (CONDITIONAL ORDINAL REGRESSION NN)
# ==============================================================================

class CORNHead(nn.Module):
    """
    Conditional Ordinal Regression Neural Network (CORN).
    Models conditional probabilities: P(y > k | y >= k) = sigmoid(g(x)_k).
    Guarantees consistent probability chains without threshold crossing violations.
    """
    def __init__(self, in_features: int = 512, num_classes: int = 5):
        super(CORNHead, self).__init__()
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1  # 4 conditional tasks
        self.fc = nn.Linear(in_features, self.num_thresholds)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        cond_logits = self.fc(x)  # (B, 4)
        cond_probs = torch.sigmoid(cond_logits)  # P(y > k | y >= k)
        
        # Compute cumulative probabilities: P(y > k) = prod_{j=0}^k P(y > j | y >= j)
        cum_probs = torch.cumprod(cond_probs, dim=1)  # (B, 4)
        
        # Predict class via thresholding cumulative probabilities
        preds = torch.sum(cum_probs > 0.5, dim=1).long()  # [0..4]
        
        # Reconstruct interval class probabilities
        B = x.size(0)
        probs = torch.zeros(B, self.num_classes, device=x.device)
        probs[:, 0] = 1.0 - cum_probs[:, 0]
        for k in range(1, self.num_thresholds):
            probs[:, k] = F.relu(cum_probs[:, k - 1] - cum_probs[:, k])
        probs[:, -1] = cum_probs[:, -1]
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-7)
        
        return {
            "cond_logits": cond_logits,
            "cond_probs": cond_probs,
            "cum_probs": cum_probs,
            "probs": probs,
            "preds": preds
        }

    def loss_fn(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        cond_logits = outputs["cond_logits"]  # (B, 4)
        B = targets.size(0)
        device = targets.device
        
        loss = 0.0
        valid_tasks = 0
        
        for k in range(self.num_thresholds):
            # Conditional subset: only samples where y >= k contribute to training the k-th head
            mask = (targets >= k)
            if mask.sum() > 0:
                subset_logits = cond_logits[mask, k]
                subset_targets = (targets[mask] > k).float()
                task_loss = F.binary_cross_entropy_with_logits(subset_logits, subset_targets)
                loss += task_loss
                valid_tasks += 1
                
        return loss / max(1, valid_tasks)


# ==============================================================================
# 6. OUTPUT HEAD 5: CLM_QWK (CHAMPION - CUMULATIVE LINK MODEL + QWK)
# ==============================================================================

class CumulativeLinkModelQWKHead(nn.Module):
    """
    Champion Output Architecture: Cumulative Link Model (CLM) with Complementary Log-Log (clog-log)
    Link Function & Hybrid Continuous QWK Loss.
    
    Mathematical Formulation:
      - Latent Severity Score: s(x) = W^T x (scalar continuous representation)
      - Strictly Monotonic Cutpoints: theta_1 < theta_2 < theta_3 < theta_4
      - Complementary Log-Log (clog-log) link: F(z) = 1 - exp(-exp(z))
      - Cumulative Probability: P(Y <= k) = 1 - exp(-exp(theta_k - s(x)))
      - Interval Probability: P(Y = k) = P(Y <= k) - P(Y <= k-1)
      - Loss: Negative Log-Likelihood (NLL) + lambda * Continuous_QWK_Loss
    """
    def __init__(
        self,
        in_features: int = 512,
        num_classes: int = 5,
        qwk_weight: float = 1.2
    ):
        super(CumulativeLinkModelQWKHead, self).__init__()
        self.num_classes = num_classes
        self.num_cutpoints = num_classes - 1  # 4 cutpoints for 5 grades
        self.qwk_weight = qwk_weight
        
        # Linear projection to single latent severity score
        self.score_fc = nn.Linear(in_features, 1, bias=False)
        
        # Parameterize monotonic cutpoints using exponential increments:
        # theta_1 = c_1, theta_k = theta_{k-1} + exp(delta_k)
        self.first_cutpoint = nn.Parameter(torch.tensor(-1.5))
        self.cutpoint_increments = nn.Parameter(torch.ones(self.num_cutpoints - 1) * -0.5)
        
        self.qwk_loss = ContinuousQWKLoss(num_classes=num_classes)

    def _get_monotonic_cutpoints(self) -> torch.Tensor:
        """Computes strictly ordered cutpoints theta_1 < theta_2 < ... < theta_4."""
        increments = F.softplus(self.cutpoint_increments) + 1e-4  # Strictly positive
        cutpoints = [self.first_cutpoint]
        for i in range(len(increments)):
            cutpoints.append(cutpoints[-1] + increments[i])
        return torch.stack(cutpoints)  # (4,)

    def _cloglog_cdf(self, z: torch.Tensor) -> torch.Tensor:
        """Complementary log-log link function: F(z) = 1 - exp(-exp(z))."""
        # Clamp to avoid numerical overflow in double exponential
        z_clamped = torch.clamp(z, min=-10.0, max=4.0)
        return 1.0 - torch.exp(-torch.exp(z_clamped))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = x.size(0)
        severity_score = self.score_fc(x).squeeze(1)  # (B,)
        cutpoints = self._get_monotonic_cutpoints()    # (4,)
        
        # Compute latent differences: (theta_k - s(x)) of shape (B, 4)
        z = cutpoints.unsqueeze(0) - severity_score.unsqueeze(1)
        
        # Cumulative probabilities P(Y <= k) for k = 0, 1, 2, 3
        cum_probs = self._cloglog_cdf(z)  # (B, 4)
        
        # Compute discrete interval probabilities P(Y = k)
        probs = torch.zeros(B, self.num_classes, device=x.device)
        probs[:, 0] = cum_probs[:, 0]
        for k in range(1, self.num_cutpoints):
            probs[:, k] = F.relu(cum_probs[:, k] - cum_probs[:, k - 1])
        probs[:, -1] = F.relu(1.0 - cum_probs[:, -1])
        
        # Normalize to valid probability simplex
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-7)
        
        # Expected continuous grade prediction
        class_indices = torch.arange(self.num_classes, device=x.device, dtype=torch.float32)
        expected_grade = torch.sum(probs * class_indices, dim=1)  # (B,)
        preds = torch.round(expected_grade).clamp(0, self.num_classes - 1).long()
        
        return {
            "severity_score": severity_score,
            "cutpoints": cutpoints,
            "cum_probs": cum_probs,
            "probs": probs,
            "expected_grade": expected_grade,
            "preds": preds
        }

    def loss_fn(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        probs = outputs["probs"]  # (B, 5)
        B = targets.size(0)
        
        # 1. Negative Log-Likelihood of observed ordinal grade intervals
        eps = 1e-7
        target_probs = probs[torch.arange(B, device=targets.device), targets]
        nll_loss = -torch.log(torch.clamp(target_probs, min=eps))
        
        if class_weights is not None:
            weights_per_sample = class_weights[targets]
            nll_loss = (nll_loss * weights_per_sample).mean()
        else:
            nll_loss = nll_loss.mean()
            
        # 2. Continuous Quadratic Weighted Kappa Loss
        qwk = self.qwk_loss(probs, targets)
        
        total_loss = nll_loss + self.qwk_weight * qwk
        return total_loss


# ==============================================================================
# 7. INTEGRATED PIPELINE WRAPPER MODEL FACTORY
# ==============================================================================

class DRFullModel(nn.Module):
    """
    End-to-End Deep Learning Architecture integrating DualBranchMILModel backbone
    with any of the 5 selectable Ordinal Regression Heads.
    """
    def __init__(
        self,
        backbone: nn.Module,
        head_type: str = "CLM_QWK",
        num_classes: int = 5,
        qwk_weight: float = 1.2
    ):
        super(DRFullModel, self).__init__()
        self.backbone = backbone
        self.head_type = head_type
        self.num_classes = num_classes
        latent_dim = getattr(backbone, "latent_dim", 512)
        
        # Instantiate requested head
        head_key = head_type.upper()
        if head_key in ["SOFTMAX_CE", "CE"]:
            self.head = SoftmaxCEHead(in_features=latent_dim, num_classes=num_classes)
        elif head_key in ["SOFTMAX_QWK", "QWK"]:
            self.head = SoftmaxQWKHead(in_features=latent_dim, num_classes=num_classes, qwk_weight=qwk_weight)
        elif head_key in ["CORAL"]:
            self.head = CORALHead(in_features=latent_dim, num_classes=num_classes)
        elif head_key in ["CORN"]:
            self.head = CORNHead(in_features=latent_dim, num_classes=num_classes)
        elif head_key in ["CLM_QWK", "CLM", "CHAMPION"]:
            self.head = CumulativeLinkModelQWKHead(in_features=latent_dim, num_classes=num_classes, qwk_weight=qwk_weight)
        else:
            raise ValueError(f"Unknown head_type: '{head_type}'. Choose from Softmax_CE, Softmax_QWK, CORAL, CORN, CLM_QWK.")

    def forward(self, global_img: torch.Tensor, local_patches: torch.Tensor) -> Dict[str, torch.Tensor]:
        latent = self.backbone(global_img, local_patches)
        outputs = self.head(latent)
        outputs["latent"] = latent
        return outputs

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.head.loss_fn(outputs, targets, class_weights=class_weights)


def build_model(
    head_type: str = "CLM_QWK",
    pretrained: bool = True,
    use_cbam: bool = True,
    fusion_dim: int = 512,
    num_classes: int = 5,
    qwk_weight: float = 1.2
) -> DRFullModel:
    """
    Factory function to construct complete DR Multi-Scale Pipeline model with designated output head.
    
    Args:
        head_type: 'Softmax_CE', 'Softmax_QWK', 'CORAL', 'CORN', or 'CLM_QWK'.
        pretrained: Whether backbones use ImageNet pretrained weights.
        use_cbam: Whether to enable CBAM attention on ResNet-50.
        fusion_dim: Dimension of latent multimodal embedding.
        num_classes: 5 ordinal severity grades (0 to 4).
        qwk_weight: Weight coefficient for continuous QWK loss term.
        
    Returns:
        DRFullModel: Assembled PyTorch neural network.
    """
    try:
        from .models import DualBranchMILModel
    except (ImportError, ValueError):
        from models import DualBranchMILModel
        
    backbone = DualBranchMILModel(
        pretrained=pretrained,
        use_cbam=use_cbam,
        fusion_dim=fusion_dim
    )
    
    model = DRFullModel(
        backbone=backbone,
        head_type=head_type,
        num_classes=num_classes,
        qwk_weight=qwk_weight
    )
    return model
