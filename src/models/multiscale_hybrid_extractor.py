"""
Multi-Scale All-CNN Hybrid Extractor for Diabetic Retinopathy (DR) Grading.

This module implements a dual-branch hybrid neural network combining:
1. Global Context Branch: ResNet50 + CBAM attention on resized full fundus images (224x224).
2. Local Patch Branch: EfficientNet-B0 on K high-resolution cropped patches (128x128).
3. Intra-Fusion & MLP: Feature concatenation (3328-dim) -> 2-layer MLP -> 512-dim embedding.
4. Ordinal Head: CORAL framework outputting 4 binary cumulative logits P(y > k).
5. Hybrid Loss: HybridOrdinalQWKLoss combining WCE, Ordinal BCE, and Soft QWK losses.
6. Smart Crop Utility: High-contrast patch extraction based on local intensity variance/entropy.
"""

from typing import List, Tuple, Union
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ============================================================================
# 1. ATTENTION MODULE: CBAM (Convolutional Block Attention Module)
# ============================================================================

class ChannelAttention(nn.Module):
    """
    Channel Attention Module for CBAM.
    Aggregates spatial information via Average Pooling and Max Pooling,
    passing features through a shared MLP to produce channel weights.
    """
    def __init__(self, in_planes: int, reduction_ratio: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        hidden_planes = max(in_planes // reduction_ratio, 8)
        self.mlp = nn.Sequential(
            nn.Linear(in_planes, hidden_planes, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_planes, in_planes, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, Channels, H, W]
        b, c, _, _ = x.size()
        avg_out = self.mlp(self.avg_pool(x).view(b, c))  # [Batch, Channels]
        max_out = self.mlp(self.max_pool(x).view(b, c))  # [Batch, Channels]
        out = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)  # [Batch, Channels, 1, 1]
        return out


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module for CBAM.
    Aggregates channel information via Average Pooling and Max Pooling along channels,
    followed by a 7x7 convolution to generate spatial attention maps.
    """
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, Channels, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [Batch, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True) # [Batch, 1, H, W]
        concat = torch.cat([avg_out, max_out], dim=1)  # [Batch, 2, H, W]
        out = self.sigmoid(self.conv(concat))         # [Batch, 1, H, W]
        return out


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    Sequentially applies Channel Attention and Spatial Attention.
    """
    def __init__(self, gate_channels: int, reduction_ratio: int = 16, spatial_kernel_size: int = 7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(gate_channels, reduction_ratio=reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size=spatial_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, Channels, H, W]
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x


# ============================================================================
# 2. GLOBAL CONTEXT BRANCH (Zoom-out)
# ============================================================================

class GlobalBranch(nn.Module):
    """
    Global Context Branch.
    Extracts macroscopic feature representation from whole-fundus images [Batch, 3, 224, 224]
    using a pre-trained ResNet50 backbone enhanced with CBAM attention.
    Output: F_global [Batch, 2048]
    """
    def __init__(self, pretrained: bool = True):
        super(GlobalBranch, self).__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        
        # Remove average pool and fc layer (keep conv stages up to layer4)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Output: [B, 2048, 7, 7]
        self.cbam = CBAM(gate_channels=2048, reduction_ratio=16)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, img_global: torch.Tensor) -> torch.Tensor:
        # img_global shape: [Batch, 3, 224, 224]
        feat = self.backbone(img_global)    # Shape: [Batch, 2048, 7, 7]
        feat = self.cbam(feat)              # Shape: [Batch, 2048, 7, 7]
        feat = self.global_pool(feat)       # Shape: [Batch, 2048, 1, 1]
        F_global = torch.flatten(feat, 1)   # Shape: [Batch, 2048]
        return F_global


# ============================================================================
# 3. LOCAL PATCH BRANCH (Zoom-in)
# ============================================================================

class LocalBranch(nn.Module):
    """
    Local Patch Branch.
    Processes K cropped high-resolution patches [Batch, K, 3, 128, 128]
    using a pre-trained EfficientNet-B0 backbone.
    Applies reshaping trick to process [Batch * K, 3, 128, 128], unflattens,
    and performs Max-Pooling across the K patches to aggregate strongest local features.
    Output: F_local [Batch, 1280]
    """
    def __init__(self, pretrained: bool = True):
        super(LocalBranch, self).__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        effnet = models.efficientnet_b0(weights=weights)
        
        self.features = effnet.features      # Output for 128x128: [B*K, 1280, 4, 4]
        self.avgpool = effnet.avgpool        # Output: [B*K, 1280, 1, 1]

    def forward(self, patches_local: torch.Tensor) -> torch.Tensor:
        # patches_local shape: [Batch, K, 3, 128, 128]
        batch_size, K, C, H, W = patches_local.shape
        
        # Reshape trick: merge Batch and K dimensions -> [Batch * K, C, H, W]
        x = patches_local.view(batch_size * K, C, H, W)  # Shape: [Batch * K, 3, 128, 128]
        
        # Pass through EfficientNet-B0 feature extractor
        feat = self.features(x)              # Shape: [Batch * K, 1280, 4, 4]
        feat = self.avgpool(feat)            # Shape: [Batch * K, 1280, 1, 1]
        feat = torch.flatten(feat, 1)        # Shape: [Batch * K, 1280]
        
        # Unflatten back to [Batch, K, 1280]
        feat = feat.view(batch_size, K, -1)  # Shape: [Batch, K, 1280]
        
        # Max aggregation across K dimension (strongest patch feature per channel)
        F_local, _ = torch.max(feat, dim=1)  # Shape: [Batch, 1280]
        return F_local


# ============================================================================
# 4. ORDINAL CLASSIFICATION HEAD (CORAL Framework)
# ============================================================================

class OrdinalHead(nn.Module):
    """
    Consistent Rank Logits (CORAL) Ordinal Classification Head.
    Outputs 4 binary logits corresponding to cumulative rank probabilities:
    P(y > 0), P(y > 1), P(y > 2), P(y > 3).
    """
    def __init__(self, in_features: int = 512, num_classes: int = 5):
        super(OrdinalHead, self).__init__()
        self.num_classes = num_classes
        self.num_logits = num_classes - 1  # 4 binary logits for 5 ordinal classes
        self.fc = nn.Linear(in_features, self.num_logits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [Batch, 512]
        logits = self.fc(x)  # Shape: [Batch, 4]
        return logits

    @staticmethod
    def logits_to_class_probs(logits: torch.Tensor) -> torch.Tensor:
        """
        Derives discrete 5-class nominal probabilities P(y = k) from 4 CORAL logits.
        P(y > k) = sigmoid(logit_k)
        P(y = 0) = 1 - P(y > 0)
        P(y = 1) = P(y > 0) - P(y > 1)
        P(y = 2) = P(y > 1) - P(y > 2)
        P(y = 3) = P(y > 2) - P(y > 3)
        P(y = 4) = P(y > 3)
        """
        cum_probs = torch.sigmoid(logits)  # Shape: [Batch, 4]
        p_gt0 = cum_probs[:, 0:1]
        p_gt1 = cum_probs[:, 1:2]
        p_gt2 = cum_probs[:, 2:3]
        p_gt3 = cum_probs[:, 3:4]
        
        p0 = 1.0 - p_gt0
        p1 = p_gt0 - p_gt1
        p2 = p_gt1 - p_gt2
        p3 = p_gt2 - p_gt3
        p4 = p_gt3
        
        probs = torch.cat([p0, p1, p2, p3, p4], dim=1)  # Shape: [Batch, 5]
        probs = torch.clamp(probs, min=1e-7, max=1.0)
        probs = probs / probs.sum(dim=1, keepdim=True)  # Renormalize for numerical stability
        return probs


# ============================================================================
# 5. MAIN MODEL ARCHITECTURE: MultiScaleHybridExtractor
# ============================================================================

class MultiScaleHybridExtractor(nn.Module):
    """
    Multi-Scale All-CNN Hybrid Extractor for 5-class Diabetic Retinopathy Grading.
    Integrates Global Context Branch (ResNet50 + CBAM) and Local Patch Branch (EfficientNet-B0 + MaxPool)
    followed by Intra-Fusion MLP and CORAL Ordinal Head.
    """
    def __init__(self, pretrained: bool = True, num_classes: int = 5):
        super(MultiScaleHybridExtractor, self).__init__()
        self.global_branch = GlobalBranch(pretrained=pretrained)  # Outputs 2048-dim
        self.local_branch = LocalBranch(pretrained=pretrained)    # Outputs 1280-dim
        
        fused_dim = 2048 + 1280  # 3328-dim
        
        # 2-Layer MLP Fusion Network
        self.mlp = nn.Sequential(
            nn.Linear(fused_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )
        
        self.ordinal_head = OrdinalHead(in_features=512, num_classes=num_classes)

    def forward(self, img_global: torch.Tensor, patches_local: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        :param img_global: Tensor of shape [Batch, 3, 224, 224]
        :param patches_local: Tensor of shape [Batch, K, 3, 128, 128]
        :return: CORAL logits of shape [Batch, 4]
        """
        # Step A: Global Branch
        F_global = self.global_branch(img_global)          # Shape: [Batch, 2048]
        
        # Step B: Local Patch Branch
        F_local = self.local_branch(patches_local)          # Shape: [Batch, 1280]
        
        # Step C: Intra-Fusion
        F_fused = torch.cat([F_global, F_local], dim=1)     # Shape: [Batch, 3328]
        F_latent = self.mlp(F_fused)                        # Shape: [Batch, 512]
        
        # Step D: Ordinal Head
        logits = self.ordinal_head(F_latent)                # Shape: [Batch, 4]
        return logits


# ============================================================================
# 6. CUSTOM HYBRID LOSS MODULE: HybridOrdinalQWKLoss
# ============================================================================

class HybridOrdinalQWKLoss(nn.Module):
    """
    Custom Hybrid Loss Function for DR Ordinal Classification:
    Loss = alpha * L_WCE + beta * L_Ordinal + gamma * L_QWK

    - L_Ordinal: Binary Cross Entropy across 4 CORAL rank logits.
    - L_WCE: Weighted Cross-Entropy over derived 5-class nominal probabilities.
    - L_QWK: Soft Quadratic Weighted Kappa Loss (differentiable expected rank MSE formulation).
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0,
                 class_weights: Union[torch.Tensor, None] = None, num_classes: int = 5):
        super(HybridOrdinalQWKLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.num_classes = num_classes
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

    def _ordinal_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Converts ground truth DR grades [Batch] (values 0..4) to binary rank targets [Batch, 4].
        Target grade y maps to rank vector where entry i is 1.0 if y > i else 0.0.
        Example: y=2 -> [1, 1, 0, 0]
        """
        batch_size = targets.size(0)
        num_logits = self.num_classes - 1
        ranks = torch.arange(num_logits, device=targets.device).unsqueeze(0).expand(batch_size, num_logits)
        targets_expanded = targets.unsqueeze(1).expand(batch_size, num_logits)
        ordinal_targets = (targets_expanded > ranks).float()
        return ordinal_targets

    def _soft_qwk_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Differentiable approximation of Quadratic Weighted Kappa Loss.
        Uses expected continuous DR grade prediction: hat_y = sum(sigmoid(logits_k))
        and computes normalized Mean Squared Error weighted quadratic penalty.
        """
        cum_probs = torch.sigmoid(logits)           # Shape: [Batch, 4]
        pred_rank = torch.sum(cum_probs, dim=1)     # Expected continuous rank in [0, 4]
        target_rank = targets.float()              # Ground truth in [0, 4]
        
        # Quadratic error normalized by max possible squared distance ((num_classes-1)^2 = 16)
        max_dist_sq = (self.num_classes - 1) ** 2
        mse = torch.mean((pred_rank - target_rank) ** 2)
        qwk_loss = mse / max_dist_sq
        return qwk_loss

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param logits: CORAL binary logits [Batch, 4]
        :param targets: Ground truth DR severity labels [Batch] (int 0..4)
        :return: (total_loss, l_wce, l_ordinal, l_qwk)
        """
        # 1. L_Ordinal (BCE over CORAL rank logits)
        ord_targets = self._ordinal_targets(targets)        # Shape: [Batch, 4]
        l_ordinal = self.bce_loss(logits, ord_targets)
        
        # 2. L_WCE (Weighted Cross Entropy over derived 5-class probabilities)
        class_probs = OrdinalHead.logits_to_class_probs(logits)  # Shape: [Batch, 5]
        log_probs = torch.log(class_probs)
        
        if self.class_weights is not None:
            l_wce = F.nll_loss(log_probs, targets, weight=self.class_weights)
        else:
            l_wce = F.nll_loss(log_probs, targets)
            
        # 3. L_QWK (Soft Quadratic Weighted Kappa Loss)
        l_qwk = self._soft_qwk_loss(logits, targets)
        
        # Composite Loss
        total_loss = (self.alpha * l_wce) + (self.beta * l_ordinal) + (self.gamma * l_qwk)
        return total_loss, l_wce, l_ordinal, l_qwk


# ============================================================================
# 7. SMART CROP UTILITY (Data Preprocessing)
# ============================================================================

def smart_crop_patches(
    image: Union[Image.Image, np.ndarray],
    K: int = 4,
    patch_size: int = 128
) -> List[np.ndarray]:
    """
    Smart Crop utility function for PyTorch Dataset preprocessing.
    Scans a high-resolution fundus image using a sliding window and calculates
    local intensity variance/entropy to extract K informative high-contrast patches
    (e.g., regions containing blood vessels, lesions, exudates, microaneurysms).

    :param image: Input fundus image as PIL Image or NumPy array (RGB [H, W, 3]).
    :param K: Number of patches to extract (default: 4).
    :param patch_size: Size of square patch (default: 128).
    :return: List of K cropped NumPy array patches of shape [patch_size, patch_size, 3].
    """
    if isinstance(image, Image.Image):
        img_np = np.array(image)
    else:
        img_np = image.copy()
        
    H, W, C = img_np.shape
    assert H >= patch_size and W >= patch_size, f"Image dimensions ({H}x{W}) must be >= patch_size ({patch_size})"

    # Convert to grayscale for contrast/variance calculation
    if C == 3:
        gray = np.dot(img_np[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        gray = img_np.squeeze()

    step = patch_size // 2  # 50% overlap sliding window grid
    candidate_windows = []

    for y in range(0, H - patch_size + 1, step):
        for x in range(0, W - patch_size + 1, step):
            win_gray = gray[y : y + patch_size, x : x + patch_size]
            # Compute local variance as proxy for high contrast/lesion density
            var_score = np.var(win_gray)
            candidate_windows.append((var_score, y, x))

    # Sort candidates by variance in descending order
    candidate_windows.sort(key=lambda item: item[0], reverse=True)

    selected_patches = []
    selected_boxes = []

    # Non-maximum suppression to avoid heavily overlapping patches
    for score, y, x in candidate_windows:
        if len(selected_patches) >= K:
            break
            
        # Check overlap with already selected boxes
        overlap = False
        for (sy, sx) in selected_boxes:
            # Overlap threshold: centers within patch_size / 2
            if abs(y - sy) < step and abs(x - sx) < step:
                overlap = True
                break
                
        if not overlap:
            patch = img_np[y : y + patch_size, x : x + patch_size, :]
            selected_patches.append(patch)
            selected_boxes.append((y, x))

    # Fallback if image has very low contrast or non-overlapping count < K
    while len(selected_patches) < K:
        y = np.random.randint(0, H - patch_size + 1)
        x = np.random.randint(0, W - patch_size + 1)
        patch = img_np[y : y + patch_size, x : x + patch_size, :]
        selected_patches.append(patch)

    return selected_patches


# ============================================================================
# 8. DUMMY TESTING BLOCK (Verification)
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Testing Multi-Scale All-CNN Hybrid Extractor Implementation...")
    print("=" * 70)

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running test on device: {device}")

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Dummy tensor dimensions
    Batch_Size = 2
    K_patches = 4
    
    # Instantiate Model
    print("\n1. Instantiating MultiScaleHybridExtractor...")
    model = MultiScaleHybridExtractor(pretrained=False, num_classes=5).to(device)
    model.eval()

    # Create dummy input tensors
    dummy_img_global = torch.randn(Batch_Size, 3, 224, 224, device=device)
    dummy_patches_local = torch.randn(Batch_Size, K_patches, 3, 128, 128, device=device)
    dummy_targets = torch.tensor([1, 4], device=device, dtype=torch.long)  # Example DR targets

    print(f"   Input Global Image Shape:   {dummy_img_global.shape}")
    print(f"   Input Local Patches Shape:  {dummy_patches_local.shape}")
    print(f"   Input DR Target Grades:     {dummy_targets.tolist()}")

    # Forward Pass Test
    print("\n2. Executing Model Forward Pass...")
    with torch.no_grad():
        logits = model(dummy_img_global, dummy_patches_local)
        class_probs = OrdinalHead.logits_to_class_probs(logits)

    print(f"   Output CORAL Logits Shape:  {logits.shape}")
    print(f"   Derived Class Probs Shape:  {class_probs.shape}")
    print(f"   Derived Class Probs Sums:   {class_probs.sum(dim=1).cpu().numpy()} (Should be 1.0)")
    print(f"   Sample Logits:\n{logits.cpu().numpy()}")
    print(f"   Sample Class Probs:\n{class_probs.cpu().numpy()}")

    # Loss Function Test
    print("\n3. Testing HybridOrdinalQWKLoss...")
    weights = torch.tensor([1.0, 2.0, 1.5, 3.0, 4.0], device=device)
    criterion = HybridOrdinalQWKLoss(alpha=1.0, beta=1.0, gamma=1.0, class_weights=weights).to(device)

    model.train()
    logits_train = model(dummy_img_global, dummy_patches_local)
    loss, l_wce, l_ordinal, l_qwk = criterion(logits_train, dummy_targets)

    print(f"   Total Hybrid Loss:          {loss.item():.4f}")
    print(f"   - L_WCE Component:          {l_wce.item():.4f}")
    print(f"   - L_Ordinal Component:      {l_ordinal.item():.4f}")
    print(f"   - L_QWK Component:          {l_qwk.item():.4f}")

    # Backward Pass Verification
    print("\n4. Verifying Backpropagation Gradient Flow...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("   Backward pass successful! Parameter gradients computed without errors.")

    # Smart Crop Utility Test
    print("\n5. Testing Smart Crop Utility...")
    dummy_pil_img = Image.fromarray(np.random.randint(0, 256, (1024, 1024, 3), dtype=np.uint8))
    extracted_patches = smart_crop_patches(dummy_pil_img, K=K_patches, patch_size=128)

    print(f"   Extracted Patches Count:    {len(extracted_patches)}")
    print(f"   Individual Patch Shape:     {extracted_patches[0].shape}")
    assert len(extracted_patches) == K_patches
    assert extracted_patches[0].shape == (128, 128, 3)

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED SUCCESSFULLY! Model is fully functional and ready.")
    print("=" * 70)
