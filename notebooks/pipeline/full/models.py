"""
================================================================================
DIABETIC RETINOPATHY (DR) PIPELINE - DUAL-BRANCH MIL MODEL ARCHITECTURE
================================================================================
Core Research Focus: Dual-Branch Multi-Scale Representation (Global Context + Local MIL).
Architecture Overview:
  1. Global Branch: ResNet-50 + CBAM (Channel & Spatial Attention) -> GAP -> F_Global (2048-d).
  2. Local MIL Branch: EfficientNet-B0 applied per K=16 patches -> GAP -> Max-Pooling 1D -> F_Local (1280-d).
  3. Multi-Scale Fusion: Concatenation [F_Global, F_Local] -> Projection MLP -> Shared Latent Representation.
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Any, Optional, Tuple


# ==============================================================================
# 1. CBAM ATTENTION MODULE (CHANNEL & SPATIAL ATTENTION)
# ==============================================================================

class ChannelAttention(nn.Module):
    """
    Channel Attention Module for CBAM.
    Aggregates spatial information using AvgPool & MaxPool, feeding through a shared MLP.
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        hidden_dim = max(8, in_channels // reduction_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module for CBAM.
    Aggregates channel information via AvgPool & MaxPool along the channel axis,
    followed by a large 7x7 spatial convolution.
    """
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(combined)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM) sequentially applying
    Channel Attention and Spatial Attention.
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 16, kernel_size: int = 7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio=reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel-refined feature map
        x_channel = x * self.channel_att(x)
        # Spatial-refined feature map
        x_out = x_channel * self.spatial_att(x_channel)
        return x_out


# ==============================================================================
# 2. GLOBAL CONTEXT BRANCH (RESNET-50 + CBAM)
# ==============================================================================

class GlobalBranch(nn.Module):
    """
    Global Context Branch:
      Input: (B, 3, 300, 300)
      Backbone: ResNet-50 (Pretrained on ImageNet)
      Attention: CBAM applied to final conv layer feature map (2048 channels)
      Pooling: Global Average Pooling (GAP)
      Output: F_Global vector of dimension 2048
    """
    def __init__(self, pretrained: bool = True, use_cbam: bool = True):
        super(GlobalBranch, self).__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        base_resnet = models.resnet50(weights=weights)
        
        # Stem and ResNet layers 1 to 4
        self.conv1 = base_resnet.conv1
        self.bn1 = base_resnet.bn1
        self.relu = base_resnet.relu
        self.maxpool = base_resnet.maxpool
        self.layer1 = base_resnet.layer1
        self.layer2 = base_resnet.layer2
        self.layer3 = base_resnet.layer3
        self.layer4 = base_resnet.layer4  # Output: (B, 2048, H/32, W/32)
        
        self.use_cbam = use_cbam
        if use_cbam:
            self.cbam = CBAM(in_channels=2048, reduction_ratio=16)
            
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = 2048

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 300, 300)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # (B, 2048, H', W')
        
        if self.use_cbam:
            x = self.cbam(x)
            
        feat = self.gap(x)  # (B, 2048, 1, 1)
        feat = torch.flatten(feat, 1)  # (B, 2048)
        return feat


# ==============================================================================
# 3. LOCAL MIL BRANCH (EFFICIENTNET-B0 + 1D MAX-POOLING)
# ==============================================================================

class LocalMILBranch(nn.Module):
    """
    Local Multi-Instance Learning (MIL) Branch:
      Input: (B, K=16, 3, 128, 128)
      Backbone: EfficientNet-B0 (Pretrained on ImageNet) applied per patch
      Pooling: GAP per patch -> (B, K, 1280)
      Aggregation: 1D Max-Pooling across the K=16 patch dimension -> F_Local (1280)
    """
    def __init__(self, pretrained: bool = True):
        super(LocalMILBranch, self).__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base_effnet = models.efficientnet_b0(weights=weights)
        
        # Feature extractor layers
        self.features = base_effnet.features  # Output: (N, 1280, H', W')
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = 1280

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patches: Tensor of shape (B, K, 3, H_p, W_p)
            
        Returns:
            torch.Tensor: F_Local feature vector of shape (B, 1280)
        """
        B, K, C, H, W = patches.shape
        # Flatten batch and patch dimensions for parallel GPU feature extraction
        flat_patches = patches.view(B * K, C, H, W)  # (B*K, 3, 128, 128)
        
        patch_feats = self.features(flat_patches)    # (B*K, 1280, H', W')
        patch_feats = self.gap(patch_feats)          # (B*K, 1280, 1, 1)
        patch_feats = torch.flatten(patch_feats, 1)  # (B*K, 1280)
        
        # Reshape back to (B, K, 1280)
        patch_feats = patch_feats.view(B, K, self.out_dim)
        
        # Multi-Instance Max-Pooling across K patches: Selects salient lesion signatures
        # (e.g. maximum hemorrhage or microaneurysm confidence across any patch)
        f_local, _ = torch.max(patch_feats, dim=1)  # (B, 1280)
        return f_local


# ==============================================================================
# 4. MULTI-SCALE DUAL-BRANCH FUSED ARCHITECTURE
# ==============================================================================

class DualBranchMILModel(nn.Module):
    """
    Unified Dual-Branch Multi-Scale Deep Learning Network:
      Combines Global Context (ResNet-50 + CBAM) and Local Micro-Lesions (MIL EfficientNet-B0).
      Outputs a compact, fused latent embedding D_latent = 512 for downstream classification/regression heads.
    """
    def __init__(
        self,
        pretrained: bool = True,
        use_cbam: bool = True,
        fusion_dim: int = 512,
        dropout_rate: float = 0.3
    ):
        super(DualBranchMILModel, self).__init__()
        
        self.global_branch = GlobalBranch(pretrained=pretrained, use_cbam=use_cbam)
        self.local_mil_branch = LocalMILBranch(pretrained=pretrained)
        
        combined_dim = self.global_branch.out_dim + self.local_mil_branch.out_dim  # 2048 + 1280 = 3328
        
        # Fusion projection module
        self.fusion_layer = nn.Sequential(
            nn.Linear(combined_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.Mish(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        self.latent_dim = fusion_dim

    def forward(self, global_img: torch.Tensor, local_patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            global_img: (B, 3, 300, 300) global full fundus image.
            local_patches: (B, K=16, 3, 128, 128) high-res local patches.
            
        Returns:
            torch.Tensor: Fused latent feature embedding of shape (B, fusion_dim=512).
        """
        f_global = self.global_branch(global_img)        # (B, 2048)
        f_local = self.local_mil_branch(local_patches)   # (B, 1280)
        
        # Multimodal feature concatenation
        f_fused = torch.cat([f_global, f_local], dim=1)  # (B, 3328)
        
        # Latent projection
        latent = self.fusion_layer(f_fused)              # (B, 512)
        return latent
