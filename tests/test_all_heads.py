import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class ContinuousQWKLoss(nn.Module):
    def __init__(self, num_classes: int = 5, eps: float = 1e-7):
        super(ContinuousQWKLoss, self).__init__()
        self.num_classes = num_classes
        self.eps = eps
        w = torch.zeros((num_classes, num_classes), dtype=torch.float32)
        for i in range(num_classes):
            for j in range(num_classes):
                w[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)
        self.register_buffer("W", w)

    def forward(self, pred_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        device = pred_probs.device
        N = pred_probs.size(0)
        if N == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        if targets.ndim == 1:
            y_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        else:
            y_one_hot = targets.float()
        pred_probs = pred_probs / (pred_probs.sum(dim=1, keepdim=True) + self.eps)
        O = torch.matmul(pred_probs.t(), y_one_hot) / (N + self.eps)
        hist_pred = pred_probs.sum(dim=0, keepdim=True) / (N + self.eps)
        hist_true = y_one_hot.sum(dim=0, keepdim=True) / (N + self.eps)
        E = torch.matmul(hist_pred.t(), hist_true)
        W = self.W.to(device)
        num = torch.sum(W * O)
        den = torch.sum(W * E) + self.eps
        return num / den

class SoftmaxCEHead(nn.Module):
    def __init__(self, in_features: int = 512, num_classes: int = 5):
        super(SoftmaxCEHead, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.fc(x)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return {"logits": logits, "probs": probs, "preds": preds}

    def loss_fn(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor, class_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        return F.cross_entropy(outputs["logits"], targets, weight=class_weights)

class SoftmaxQWKHead(nn.Module):
    def __init__(self, in_features: int = 512, num_classes: int = 5, qwk_weight: float = 1.0):
        super(SoftmaxQWKHead, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)
        self.num_classes = num_classes
        self.qwk_weight = qwk_weight
        self.qwk_loss = ContinuousQWKLoss(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.fc(x)
        probs = F.softmax(logits, dim=1)
        continuous_pred = torch.sum(probs * torch.arange(self.num_classes, device=x.device), dim=1)
        preds = torch.round(continuous_pred).clamp(0, self.num_classes - 1).long()
        return {"logits": logits, "probs": probs, "preds": preds, "continuous_preds": continuous_pred}

    def loss_fn(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor, class_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        ce = F.cross_entropy(outputs["logits"], targets, weight=class_weights)
        qwk = self.qwk_loss(outputs["probs"], targets)
        return ce + self.qwk_weight * qwk

class CORALHeadCurrent(nn.Module):
    def __init__(self, in_features: int = 512, num_classes: int = 5):
        super(CORALHeadCurrent, self).__init__()
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1
        self.linear = nn.Linear(in_features, 1, bias=False)
        self.biases = nn.Parameter(torch.zeros(self.num_thresholds))
        with torch.no_grad():
            self.biases.copy_(torch.linspace(-2.0, 2.0, self.num_thresholds))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
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
        return {"rank_logits": rank_logits, "rank_probs": rank_probs, "probs": probs, "preds": preds}

    def loss_fn(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor, class_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        rank_logits = outputs["rank_logits"]
        device = targets.device
        B = targets.size(0)
        binary_targets = torch.zeros(B, self.num_thresholds, device=device)
        for k in range(self.num_thresholds):
            binary_targets[:, k] = (targets > k).float()
        return F.binary_cross_entropy_with_logits(rank_logits, binary_targets, reduction="mean")

class CORALHeadFixed(nn.Module):
    def __init__(self, in_features: int = 512, num_classes: int = 5):
        super(CORALHeadFixed, self).__init__()
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1
        self.linear = nn.Linear(in_features, 1, bias=False)
        self.biases = nn.Parameter(torch.zeros(self.num_thresholds))
        with torch.no_grad():
            self.biases.copy_(torch.linspace(2.0, -2.0, self.num_thresholds))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
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
        return {"rank_logits": rank_logits, "rank_probs": rank_probs, "probs": probs, "preds": preds}

    def loss_fn(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor, class_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        rank_logits = outputs["rank_logits"]
        device = targets.device
        B = targets.size(0)
        binary_targets = torch.zeros(B, self.num_thresholds, device=device)
        for k in range(self.num_thresholds):
            binary_targets[:, k] = (targets > k).float()
        return F.binary_cross_entropy_with_logits(rank_logits, binary_targets, reduction="mean")

class CORNHead(nn.Module):
    def __init__(self, in_features: int = 512, num_classes: int = 5):
        super(CORNHead, self).__init__()
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1
        self.fc = nn.Linear(in_features, self.num_thresholds)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
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
        return {"cond_logits": cond_logits, "cond_probs": cond_probs, "cum_probs": cum_probs, "probs": probs, "preds": preds}

    def loss_fn(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor, class_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        cond_logits = outputs["cond_logits"]
        loss = 0.0
        valid_tasks = 0
        for k in range(self.num_thresholds):
            mask = (targets >= k)
            if mask.sum() > 0:
                subset_logits = cond_logits[mask, k]
                subset_targets = (targets[mask] > k).float()
                task_loss = F.binary_cross_entropy_with_logits(subset_logits, subset_targets)
                loss += task_loss
                valid_tasks += 1
        return loss / max(1, valid_tasks)

class CumulativeLinkModelQWKHead(nn.Module):
    def __init__(self, in_features: int = 512, num_classes: int = 5, qwk_weight: float = 1.2):
        super(CumulativeLinkModelQWKHead, self).__init__()
        self.num_classes = num_classes
        self.num_cutpoints = num_classes - 1
        self.target_qwk_weight = qwk_weight
        self.current_qwk_weight = 0.4
        self.score_fc = nn.Linear(in_features, 1, bias=False)
        self.first_cutpoint = nn.Parameter(torch.tensor(-1.5))
        self.cutpoint_increments = nn.Parameter(torch.ones(self.num_cutpoints - 1) * -0.5)
        self.qwk_loss = ContinuousQWKLoss(num_classes=num_classes)

    def _get_monotonic_cutpoints(self) -> torch.Tensor:
        increments = F.softplus(self.cutpoint_increments) + 1e-4
        cutpoints = [self.first_cutpoint]
        for i in range(len(increments)):
            cutpoints.append(cutpoints[-1] + increments[i])
        return torch.stack(cutpoints)

    def _cloglog_cdf(self, z: torch.Tensor) -> torch.Tensor:
        z_clamped = torch.clamp(z, min=-10.0, max=4.0)
        return 1.0 - torch.exp(-torch.exp(z_clamped))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = x.size(0)
        severity_score = self.score_fc(x).squeeze(1)
        cutpoints = self._get_monotonic_cutpoints()
        z = cutpoints.unsqueeze(0) - severity_score.unsqueeze(1)
        cum_probs = self._cloglog_cdf(z)
        rank_probs = 1.0 - cum_probs
        probs = torch.zeros(B, self.num_classes, device=x.device)
        probs[:, 0] = cum_probs[:, 0]
        for k in range(1, self.num_cutpoints):
            probs[:, k] = F.relu(cum_probs[:, k] - cum_probs[:, k - 1])
        probs[:, -1] = F.relu(1.0 - cum_probs[:, -1])
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-7)
        class_indices = torch.arange(self.num_classes, device=x.device, dtype=torch.float32)
        expected_grade = torch.sum(probs * class_indices, dim=1)
        preds = torch.round(expected_grade).clamp(0, self.num_classes - 1).long()
        return {"severity_score": severity_score, "cutpoints": cutpoints, "cum_probs": cum_probs, "rank_probs": rank_probs, "probs": probs, "expected_grade": expected_grade, "preds": preds}

    def loss_fn(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor, class_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        probs = outputs["probs"]
        B = targets.size(0)
        eps = 1e-7
        target_probs = probs[torch.arange(B, device=targets.device), targets]
        nll_loss = -torch.log(torch.clamp(target_probs, min=eps))
        if class_weights is not None:
            nll_loss = (nll_loss * class_weights[targets]).mean()
        else:
            nll_loss = nll_loss.mean()
        qwk = self.qwk_loss(probs, targets)
        return nll_loss + self.current_qwk_weight * qwk

def test_all_candidate_heads():
    torch.manual_seed(42)
    x = torch.randn(16, 512)
    targets = torch.randint(0, 5, (16,))
    class_weights = torch.tensor([1.0, 2.0, 1.5, 3.0, 2.5])

    models = [
        ("SoftmaxCEHead", SoftmaxCEHead()),
        ("SoftmaxQWKHead", SoftmaxQWKHead()),
        ("CORALHead (Current: linspace -2 to 2)", CORALHeadCurrent()),
        ("CORALHead (Fixed: linspace 2 to -2)", CORALHeadFixed()),
        ("CORNHead", CORNHead()),
        ("CumulativeLinkModelQWKHead (CLM)", CumulativeLinkModelQWKHead()),
    ]

    for name, model in models:
        out = model(x)
        assert "probs" in out
        assert out["probs"].shape == (16, 5)
        loss = model.loss_fn(out, targets, class_weights=class_weights)
        loss.backward()
        total_grad = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        assert total_grad > 0.0
        prob_sum = out["probs"].sum(dim=1).mean().item()
        assert abs(prob_sum - 1.0) < 1e-4

if __name__ == "__main__":
    test_all_candidate_heads()
    print("All candidate heads passed unit testing!")

    print("\n--- Detailed Investigation on CORAL Rank Probs & Monotonicity ---")
    c_cur = CORALHeadCurrent()
    out_cur = c_cur(x)
    print("Current CORAL biases:", c_cur.biases.data.tolist())
    print("Current CORAL rank_probs[0]:", [round(v, 4) for v in out_cur["rank_probs"][0].tolist()])
    print("Current CORAL probs[0]:", [round(v, 4) for v in out_cur["probs"][0].tolist()])

    c_fix = CORALHeadFixed()
    out_fix = c_fix(x)
    print("Fixed CORAL biases:", c_fix.biases.data.tolist())
    print("Fixed CORAL rank_probs[0]:", [round(v, 4) for v in out_fix["rank_probs"][0].tolist()])
    print("Fixed CORAL probs[0]:", [round(v, 4) for v in out_fix["probs"][0].tolist()])

    print("\n--- Detailed Investigation on CLM Cutpoints & Probs ---")
    clm = CumulativeLinkModelQWKHead()
    out_clm = clm(x)
    print("CLM monotonic cutpoints:", [round(v, 4) for v in out_clm["cutpoints"].tolist()])
    print("CLM cum_probs[0]:", [round(v, 4) for v in out_clm["cum_probs"][0].tolist()])
    print("CLM probs[0]:", [round(v, 4) for v in out_clm["probs"][0].tolist()])
