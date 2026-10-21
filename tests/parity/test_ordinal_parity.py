"""Parity test for CORAL and CORN heads and losses."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from drdg.losses.ordinal import coral_loss, corn_loss
from drdg.models.heads.coral import CORALHead
from drdg.models.heads.corn import CORNHead


# Golden reference extracted from visualize_main_model.py
class GoldenCORALHead(nn.Module):
    def __init__(self, in_features: int = 512, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1
        self.linear = nn.Linear(in_features, 1, bias=False)
        self.biases = nn.Parameter(torch.zeros(self.num_thresholds))
        with torch.no_grad():
            self.biases.copy_(torch.linspace(2.0, -2.0, self.num_thresholds))

    def forward(self, x: torch.Tensor):
        base_logit = self.linear(x)
        rank_logits = base_logit + self.biases
        rank_probs = torch.sigmoid(rank_logits)
        preds = torch.sum(rank_probs > 0.5, dim=1).long()
        B = x.size(0)
        probs = torch.zeros(B, self.num_classes, device=x.device)
        probs[:, 0] = 1.0 - rank_probs[:, 0]
        for k in range(1, self.num_thresholds):
            probs[:, k] = F.relu(rank_probs[:, k - 1] - rank_probs[:, k])
        probs[:, -1] = rank_probs[:, -1]
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-7)
        return {
            "rank_logits": rank_logits,
            "rank_probs": rank_probs,
            "probs": probs,
            "preds": preds,
        }

    def loss_fn(self, outputs, targets):
        rank_logits = outputs["rank_logits"]
        B = targets.size(0)
        binary_targets = torch.zeros(B, self.num_thresholds, device=targets.device)
        for k in range(self.num_thresholds):
            binary_targets[:, k] = (targets > k).float()
        return F.binary_cross_entropy_with_logits(rank_logits, binary_targets, reduction="mean")


class GoldenCORNHead(nn.Module):
    def __init__(self, in_features: int = 512, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1
        self.fc = nn.Linear(in_features, self.num_thresholds)

    def forward(self, x: torch.Tensor):
        cond_logits = self.fc(x)
        cond_probs = torch.sigmoid(cond_logits)
        cum_probs = torch.cumprod(cond_probs, dim=1)
        preds = torch.sum(cum_probs > 0.5, dim=1).long()
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
            "preds": preds,
        }

    def loss_fn(self, outputs, targets):
        cond_logits = outputs["cond_logits"]
        loss = 0.0
        valid_tasks = 0
        for k in range(self.num_thresholds):
            mask = targets >= k
            if mask.sum() > 0:
                subset_logits = cond_logits[mask, k]
                subset_targets = (targets[mask] > k).float()
                loss += F.binary_cross_entropy_with_logits(subset_logits, subset_targets)
                valid_tasks += 1
        return loss / max(1, valid_tasks)


def test_coral_weight_transfer_and_loss_parity():
    torch.manual_seed(42)
    gold = GoldenCORALHead(in_features=64, num_classes=5)
    cand = CORALHead(in_features=64, num_classes=5)

    cand.load_state_dict(gold.state_dict())
    gold.eval()
    cand.eval()

    x = torch.randn(8, 64)
    targets = torch.tensor([0, 1, 2, 3, 4, 1, 2, 0])

    with torch.no_grad():
        out_gold = gold(x)
        out_cand = cand(x)

    assert torch.allclose(out_gold["rank_logits"], out_cand["rank_logits"], atol=1e-6)
    assert torch.allclose(out_gold["rank_probs"], out_cand["rank_probs"], atol=1e-6)
    assert torch.allclose(out_gold["probs"], out_cand["probs"], atol=1e-6)
    assert torch.equal(out_gold["preds"], out_cand["preds"])

    loss_gold = gold.loss_fn(out_gold, targets)
    loss_cand = cand.loss_fn(out_cand, targets)
    assert torch.allclose(loss_gold, loss_cand, atol=1e-6)


def test_corn_weight_transfer_and_loss_parity():
    torch.manual_seed(42)
    gold = GoldenCORNHead(in_features=64, num_classes=5)
    cand = CORNHead(in_features=64, num_classes=5)

    cand.load_state_dict(gold.state_dict())
    gold.eval()
    cand.eval()

    x = torch.randn(8, 64)
    targets = torch.tensor([0, 1, 2, 3, 4, 1, 2, 0])

    with torch.no_grad():
        out_gold = gold(x)
        out_cand = cand(x)

    assert torch.allclose(out_gold["cond_logits"], out_cand["cond_logits"], atol=1e-6)
    assert torch.allclose(out_gold["cum_probs"], out_cand["cum_probs"], atol=1e-6)
    assert torch.allclose(out_gold["probs"], out_cand["probs"], atol=1e-6)
    assert torch.equal(out_gold["preds"], out_cand["preds"])

    loss_gold = gold.loss_fn(out_gold, targets)
    loss_cand = cand.loss_fn(out_cand, targets)
    assert torch.allclose(loss_gold, loss_cand, atol=1e-6)
