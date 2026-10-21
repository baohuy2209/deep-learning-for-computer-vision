import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import sys
from typing import Optional, Dict, Tuple, List, Union

sys.stdout.reconfigure(encoding='utf-8')

# Load main_model (4).ipynb
with open("notebooks/diabetic_retinopathy/pipeline/model/main_model (4).ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Extract code cells containing ContinuousQWKLoss and all 5 heads
cells_to_exec = [47, 49, 51, 53, 55, 57]
code_blocks = []
for c_idx in cells_to_exec:
    code_blocks.append("".join(nb["cells"][c_idx]["source"]))

full_code = "\n".join(code_blocks)

# Setup namespace
ns = {
    'torch': torch,
    'nn': nn,
    'F': F,
    'Dict': Dict,
    'Optional': Optional,
    'Tuple': Tuple,
    'List': List,
    'Union': Union,
}

exec(full_code, ns)

B = 8
in_features = 512
num_classes = 5

x = torch.randn(B, in_features)
targets = torch.randint(0, num_classes, (B,))

print(f"Testing all 5 Heads with input shape: {x.shape}, targets shape: {targets.shape}\n")

heads = {
    "1. SoftmaxCEHead": ns["SoftmaxCEHead"](in_features, num_classes),
    "2. SoftmaxQWKHead": ns["SoftmaxQWKHead"](in_features, num_classes, qwk_weight=0.4),
    "3. CORALHead": ns["CORALHead"](in_features, num_classes),
    "4. CORNHead": ns["CORNHead"](in_features, num_classes),
    "5. CumulativeLinkModelQWKHead": ns["CumulativeLinkModelQWKHead"](in_features, num_classes, qwk_weight=0.4),
}

for name, head in heads.items():
    print(f"--- Testing {name} ---")
    out = head(x)
    loss = head.loss_fn(out, targets)
    
    # Check outputs
    probs = out.get("probs")
    preds = out.get("preds")
    print(f"  * Loss: {loss.item():.4f}")
    print(f"  * Probs shape: {probs.shape}, sum across classes: {probs.sum(dim=-1).detach().numpy().round(3)}")
    print(f"  * Preds shape: {preds.shape}, range: [{preds.min().item()}, {preds.max().item()}]")
    print(f"  * Output keys: {list(out.keys())}")
    
    # Check monotonicity properties if ordinal
    if "rank_probs" in out:
        rank_p = out["rank_probs"].detach().numpy()
        diffs = rank_p[:, 1:] - rank_p[:, :-1]
        violations = (diffs > 1e-4).mean()
        print(f"  * Rank probs P(y > k) violation rate: {violations * 100:.1f}%")
    elif "cum_probs" in out:
        cum_p = out["cum_probs"].detach().numpy()
        # For CORN, cum_probs is P(y > k)
        if "cond_logits" in out: # CORN
            diffs = cum_p[:, 1:] - cum_p[:, :-1]
            violations = (diffs > 1e-4).mean()
            print(f"  * CORN cum_probs P(y > k) violation rate: {violations * 100:.1f}%")
        else: # CLM, cum_probs is P(Y <= k)
            diffs = cum_p[:, 1:] - cum_p[:, :-1]
            non_increasing = (diffs < -1e-4).mean()
            print(f"  * CLM P(Y <= k) non-decreasing check (violations): {non_increasing * 100:.1f}%")
    print()
