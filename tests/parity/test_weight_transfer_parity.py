"""Weight-transfer numerical parity test between golden reference and drdg modules."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from drdg.losses.qwk import ContinuousQWKLoss
from drdg.models.attention.cbam import CBAM
from drdg.models.heads.clm import CumulativeLinkModelQWKHead as CLMHeadDRDG
from drdg.models.heads.coral import CORALHead as CORALHeadDRDG
from drdg.models.heads.corn import CORNHead as CORNHeadDRDG
from drdg.models.heads.softmax import SoftmaxCEHead as SoftmaxCEDRDG
from drdg.models.heads.softmax import SoftmaxQWKHead as SoftmaxQWKDRDG


# ---------------------------------------------------------------------------
# Golden Reference Implementations (Extracted verbatim from notebook)
# ---------------------------------------------------------------------------
class GoldenContinuousQWKLoss(nn.Module):
    def __init__(self, num_classes: int = 5, eps: float = 1e-7):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps
        w = torch.zeros((num_classes, num_classes), dtype=torch.float32)
        for i in range(num_classes):
            for j in range(num_classes):
                w[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)
        self.register_buffer("W", w)

    def forward(self, pred_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        device = pred_probs.device
        n = pred_probs.size(0)
        if n == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        if targets.ndim == 1:
            y_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        else:
            y_one_hot = targets.float()
        pred_probs = pred_probs / (pred_probs.sum(dim=1, keepdim=True) + self.eps)
        o = torch.matmul(pred_probs.t(), y_one_hot) / (n + self.eps)
        hist_pred = pred_probs.sum(dim=0, keepdim=True) / (n + self.eps)
        hist_true = y_one_hot.sum(dim=0, keepdim=True) / (n + self.eps)
        e = torch.matmul(hist_pred.t(), hist_true)
        w = self.W.to(device)
        num = torch.sum(w * o)
        den = torch.sum(w * e) + self.eps
        return num / den


class GoldenSoftmaxCEHead(nn.Module):
    def __init__(self, in_features: int = 512, num_classes: int = 5):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor):
        logits = self.fc(x)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return {"logits": logits, "probs": probs, "preds": preds}


class GoldenCLMHead(nn.Module):
    def __init__(self, in_features: int = 512, num_classes: int = 5, qwk_weight: float = 0.4):
        super().__init__()
        self.num_classes = num_classes
        self.num_cutpoints = num_classes - 1
        self.target_qwk_weight = qwk_weight
        self.current_qwk_weight = 0.0
        self.score_fc = nn.Linear(in_features, 1, bias=False)
        self.first_cutpoint = nn.Parameter(torch.tensor(-1.5))
        self.cutpoint_increments = nn.Parameter(torch.ones(self.num_cutpoints - 1) * -0.5)
        self.qwk_loss = GoldenContinuousQWKLoss(num_classes=num_classes)

    def _get_monotonic_cutpoints(self) -> torch.Tensor:
        increments = F.softplus(self.cutpoint_increments) + 1e-4
        cutpoints = [self.first_cutpoint]
        for i in range(len(increments)):
            cutpoints.append(cutpoints[-1] + increments[i])
        return torch.stack(cutpoints)

    def _cloglog_cdf(self, z: torch.Tensor) -> torch.Tensor:
        z_clamped = torch.clamp(z, min=-10.0, max=4.0)
        return 1.0 - torch.exp(-torch.exp(z_clamped))

    def forward(self, x: torch.Tensor):
        b = x.size(0)
        severity_score = self.score_fc(x).squeeze(1)
        cutpoints = self._get_monotonic_cutpoints()
        z = cutpoints.unsqueeze(0) - severity_score.unsqueeze(1)
        cum_probs = self._cloglog_cdf(z)
        rank_probs = 1.0 - cum_probs
        probs = torch.zeros(b, self.num_classes, device=x.device)
        probs[:, 0] = cum_probs[:, 0]
        for k in range(1, self.num_cutpoints):
            probs[:, k] = F.relu(cum_probs[:, k] - cum_probs[:, k - 1])
        probs[:, -1] = F.relu(1.0 - cum_probs[:, -1])
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-7)
        class_indices = torch.arange(self.num_classes, device=x.device, dtype=torch.float32)
        expected_grade = torch.sum(probs * class_indices, dim=1)
        preds = torch.round(expected_grade).clamp(0, self.num_classes - 1).long()
        return {
            "severity_score": severity_score,
            "cutpoints": cutpoints,
            "cum_probs": cum_probs,
            "rank_probs": rank_probs,
            "probs": probs,
            "expected_grade": expected_grade,
            "preds": preds,
        }


# ---------------------------------------------------------------------------
# Parity Tests
# ---------------------------------------------------------------------------
def test_continuous_qwk_loss_exact_parity():
    torch.manual_seed(42)
    pred_probs = F.softmax(torch.randn(16, 5), dim=-1)
    targets = torch.randint(0, 5, (16,))

    gold_loss_fn = GoldenContinuousQWKLoss(num_classes=5)
    cand_loss_fn = ContinuousQWKLoss(num_classes=5)

    gold_loss = gold_loss_fn(pred_probs, targets)
    cand_loss = cand_loss_fn(pred_probs, targets)

    assert torch.allclose(gold_loss, cand_loss, atol=1e-7, rtol=1e-6), f"QWK loss mismatch: {gold_loss} vs {cand_loss}"


def test_softmax_ce_head_weight_transfer_parity():
    torch.manual_seed(42)
    x = torch.randn(8, 512)

    gold_head = GoldenSoftmaxCEHead(in_features=512, num_classes=5)
    cand_head = SoftmaxCEDRDG(in_features=512, num_classes=5)

    # Transfer state_dict
    cand_head.load_state_dict(gold_head.state_dict())

    gold_head.eval()
    cand_head.eval()

    with torch.no_grad():
        gold_out = gold_head(x)
        cand_out = cand_head(x)

    assert torch.allclose(gold_out["logits"], cand_out["logits"], atol=1e-7), "Logits mismatch in SoftmaxCEHead!"
    assert torch.allclose(gold_out["probs"], cand_out["probs"], atol=1e-7), "Probs mismatch in SoftmaxCEHead!"
    assert torch.equal(gold_out["preds"], cand_out["preds"]), "Preds mismatch in SoftmaxCEHead!"


def test_clm_head_weight_transfer_parity():
    torch.manual_seed(42)
    x = torch.randn(8, 512)

    gold_head = GoldenCLMHead(in_features=512, num_classes=5)
    cand_head = CLMHeadDRDG(in_features=512, num_classes=5)

    # Transfer weights
    cand_head.load_state_dict(gold_head.state_dict())

    gold_head.eval()
    cand_head.eval()

    with torch.no_grad():
        gold_out = gold_head(x)
        cand_out = cand_head(x)

    assert torch.allclose(gold_out["severity_score"], cand_out["severity_score"], atol=1e-6), "Severity score mismatch!"
    assert torch.allclose(gold_out["cutpoints"], cand_out["cutpoints"], atol=1e-6), "Cutpoints mismatch!"
    assert torch.allclose(gold_out["cum_probs"], cand_out["cum_probs"], atol=1e-6), "Cum probs mismatch!"
    assert torch.allclose(gold_out["probs"], cand_out["probs"], atol=1e-6), "Probs mismatch!"
    assert torch.equal(gold_out["preds"], cand_out["preds"]), "Preds mismatch!"
