"""
================================================================================
DIABETIC RETINOPATHY (DR) PIPELINE - BENCHMARK BASELINES MODULE
================================================================================
Core Research Focus: Two-Tier Baseline Strategy for Cross-Domain Benchmarking.

Tier 1: Single Anchor Baselines (Vanilla Architectures)
  1. ResNet50_Baseline: Pretrained ResNet-50 (Global branch anchor).
  2. EfficientNetB0_Baseline: Pretrained EfficientNet-B0 (Local branch anchor).
  3. DenseNet121_Baseline: Pretrained DenseNet-121 (Dense connectivity anchor).

Tier 2: Complex SOTA Baselines
  4. EfficientNetB0_MSAG: EfficientNet-B0 + Multi-Scale Attention Gate (MSAG).
  5. Hybrid_EffB0_ViT: EfficientNet-B0 CNN feature extractor + Vision Transformer (ViT) Self-Attention.
  6. EfficientNetB3_Baseline: Pretrained EfficientNet-B3 high-capacity anchor.
  7. DFD_Net_Nominal: Dual-Branch Network without MIL for standard Nominal Softmax classification.

All models can output 5-class logits directly OR return latent feature vectors
for dynamic integration with `losses_and_heads.py` (CORAL, CORN, CLM_QWK).
================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Any, Optional, Union, Tuple


# ==============================================================================
# TIER 1: SINGLE ANCHOR BASELINES (VANILLA ARCHITECTURES)
# ==============================================================================

class ResNet50_Baseline(nn.Module):
    """
    Tier 1 Baseline: Standard ResNet-50 Architecture.
    Purpose: Serves as the Global Context single-branch anchor.
    """
    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        return_features: bool = False
    ):
        super(ResNet50_Baseline, self).__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        base_model = models.resnet50(weights=weights)
        
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        self.latent_dim = 2048
        self.return_features = return_features
        
        if not return_features:
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(self.latent_dim, num_classes)
            )
        else:
            self.classifier = nn.Identity()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # Handles dual-input signature if called within multi-scale pipeline
        if isinstance(x, (tuple, list)):
            x = x[0]
            
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        feat = self.gap(x)
        feat = torch.flatten(feat, 1)
        
        if self.return_features:
            return feat
        return self.classifier(feat)


class EfficientNetB0_Baseline(nn.Module):
    """
    Tier 1 Baseline: Standard EfficientNet-B0 Architecture.
    Purpose: Serves as the Local single-branch anchor to prove the superiority of MIL patch aggregation.
    """
    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        return_features: bool = False
    ):
        super(EfficientNetB0_Baseline, self).__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base_model = models.efficientnet_b0(weights=weights)
        
        self.features = base_model.features
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.latent_dim = 1280
        self.return_features = return_features
        
        if not return_features:
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(self.latent_dim, num_classes)
            )
        else:
            self.classifier = nn.Identity()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if isinstance(x, (tuple, list)):
            x = x[0]
            
        feat = self.features(x)
        feat = self.gap(feat)
        feat = torch.flatten(feat, 1)
        
        if self.return_features:
            return feat
        return self.classifier(feat)


class DenseNet121_Baseline(nn.Module):
    """
    Tier 1 Baseline: Standard DenseNet-121 Architecture.
    Purpose: Evaluates dense connectivity feature reuse for diabetic retinopathy classification.
    """
    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        return_features: bool = False
    ):
        super(DenseNet121_Baseline, self).__init__()
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        base_model = models.densenet121(weights=weights)
        
        self.features = base_model.features
        self.relu = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.latent_dim = 1024
        self.return_features = return_features
        
        if not return_features:
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(self.latent_dim, num_classes)
            )
        else:
            self.classifier = nn.Identity()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if isinstance(x, (tuple, list)):
            x = x[0]
            
        features = self.features(x)
        out = self.relu(features)
        out = self.gap(out)
        feat = torch.flatten(out, 1)
        
        if self.return_features:
            return feat
        return self.classifier(feat)


# ==============================================================================
# TIER 2: COMPLEX SOTA BASELINES
# ==============================================================================

# ------------------------------------------------------------------------------
# 4. Multi-Scale Attention Gate (MSAG) Module & Model
# ------------------------------------------------------------------------------

class MultiScaleAttentionGate(nn.Module):
    """
    Multi-Scale Attention Gate (MSAG):
    Fuses intermediate multi-scale feature maps using parallel dilated convolutions and channel gates.
    """
    def __init__(self, in_channels: int, gate_channels: int = 256):
        super(MultiScaleAttentionGate, self).__init__()
        
        # Multi-scale receptive field convs with varying dilation rates
        self.scale1 = nn.Conv2d(in_channels, gate_channels, kernel_size=3, padding=1, dilation=1, bias=False)
        self.scale2 = nn.Conv2d(in_channels, gate_channels, kernel_size=3, padding=2, dilation=2, bias=False)
        self.scale3 = nn.Conv2d(in_channels, gate_channels, kernel_size=3, padding=4, dilation=4, bias=False)
        
        self.bn = nn.BatchNorm2d(gate_channels * 3)
        self.relu = nn.ReLU(inplace=True)
        
        # Spatial-channel attention aggregation
        self.att_conv = nn.Sequential(
            nn.Conv2d(gate_channels * 3, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.scale1(x)
        s2 = self.scale2(x)
        s3 = self.scale3(x)
        
        multi_scale = torch.cat([s1, s2, s3], dim=1)
        multi_scale = self.relu(self.bn(multi_scale))
        
        att_weights = self.att_conv(multi_scale)
        # Residual attention gating
        return x * att_weights + x


class EfficientNetB0_MSAG(nn.Module):
    """
    Tier 2 SOTA Baseline: EfficientNet-B0 + Multi-Scale Attention Gate (MSAG).
    Purpose: Highlights multi-scale feature extraction without full dual-branch patch MIL.
    """
    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        return_features: bool = False
    ):
        super(EfficientNetB0_MSAG, self).__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base_model = models.efficientnet_b0(weights=weights)
        
        self.features = base_model.features  # Output: (B, 1280, H/32, W/32)
        self.msag = MultiScaleAttentionGate(in_channels=1280, gate_channels=256)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        self.latent_dim = 1280
        self.return_features = return_features
        
        if not return_features:
            self.classifier = nn.Sequential(
                nn.BatchNorm1d(self.latent_dim),
                nn.Dropout(p=dropout_rate),
                nn.Linear(self.latent_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate / 2),
                nn.Linear(512, num_classes)
            )
        else:
            self.classifier = nn.Identity()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if isinstance(x, (tuple, list)):
            x = x[0]
            
        feat_map = self.features(x)         # (B, 1280, H', W')
        gated_map = self.msag(feat_map)     # Multi-scale attended features
        
        feat = self.gap(gated_map)
        feat = torch.flatten(feat, 1)       # (B, 1280)
        
        if self.return_features:
            return feat
        return self.classifier(feat)


# ------------------------------------------------------------------------------
# 5. Hybrid CNN + Vision Transformer (Hybrid_EffB0_ViT)
# ------------------------------------------------------------------------------

class PositionalEncoding2D(nn.Module):
    """Learnable 2D positional embeddings for visual tokens."""
    def __init__(self, d_model: int, max_h: int = 16, max_w: int = 16):
        super(PositionalEncoding2D, self).__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, max_h * max_w, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        return x + self.pos_emb[:, :N, :]


class Hybrid_EffB0_ViT(nn.Module):
    """
    Tier 2 SOTA Baseline: Hybrid CNN (EfficientNet-B0) + Vision Transformer (ViT) Architecture.
    Purpose: Evaluates tokenized global self-attention across fundus feature maps.
    """
    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout_rate: float = 0.2,
        return_features: bool = False
    ):
        super(Hybrid_EffB0_ViT, self).__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base_model = models.efficientnet_b0(weights=weights)
        
        self.cnn_features = base_model.features  # (B, 1280, H/32, W/32)
        
        # Feature map projection to transformer embedding dimension
        self.proj = nn.Conv2d(1280, embed_dim, kernel_size=1)
        self.pos_encoding = PositionalEncoding2D(d_model=embed_dim, max_h=20, max_w=20)
        
        # ViT Transformer Encoder Blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Learnable Class Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        self.latent_dim = embed_dim
        self.return_features = return_features
        
        if not return_features:
            self.classifier = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Dropout(p=dropout_rate),
                nn.Linear(embed_dim, num_classes)
            )
        else:
            self.classifier = nn.Identity()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if isinstance(x, (tuple, list)):
            x = x[0]
            
        # 1. Extract CNN feature map
        feat_map = self.cnn_features(x)                 # (B, 1280, H, W)
        proj_map = self.proj(feat_map)                  # (B, embed_dim, H, W)
        
        B, C, H, W = proj_map.shape
        # Flatten spatial grid into sequence of visual tokens
        tokens = proj_map.flatten(2).transpose(1, 2)    # (B, H*W, embed_dim)
        tokens = self.pos_encoding(tokens)
        
        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)   # (B, 1, embed_dim)
        tokens = torch.cat([cls_tokens, tokens], dim=1) # (B, 1 + H*W, embed_dim)
        
        # 2. Process via Transformer Self-Attention
        trans_out = self.transformer(tokens)            # (B, 1 + H*W, embed_dim)
        
        # Extract output corresponding to [CLS] token
        latent = trans_out[:, 0, :]                     # (B, embed_dim)
        
        if self.return_features:
            return latent
        return self.classifier(latent)


# ------------------------------------------------------------------------------
# 6. EfficientNet-B3 Baseline
# ------------------------------------------------------------------------------

class EfficientNetB3_Baseline(nn.Module):
    """
    Tier 2 Baseline: High-Capacity EfficientNet-B3 Architecture.
    Purpose: Provides higher resolution feature capacity for baseline comparisons.
    """
    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        return_features: bool = False
    ):
        super(EfficientNetB3_Baseline, self).__init__()
        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        base_model = models.efficientnet_b3(weights=weights)
        
        self.features = base_model.features
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.latent_dim = 1536
        self.return_features = return_features
        
        if not return_features:
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout_rate),
                nn.Linear(self.latent_dim, num_classes)
            )
        else:
            self.classifier = nn.Identity()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if isinstance(x, (tuple, list)):
            x = x[0]
            
        feat = self.features(x)
        feat = self.gap(feat)
        feat = torch.flatten(feat, 1)
        
        if self.return_features:
            return feat
        return self.classifier(feat)


# ------------------------------------------------------------------------------
# 7. DFD_Net_Nominal (Dual-Branch Dual-Scale Nominal Network)
# ------------------------------------------------------------------------------

class DFD_Net_Nominal(nn.Module):
    """
    Tier 2 SOTA Baseline: Dual-Branch Fundus Network (DFD-Net) with Nominal Output.
    Features:
      - Takes two input representations (Global Image + Center/Local High-Res Image).
      - Processes branches with parallel standard CNNs (ResNet-34 + EfficientNet-B0) WITHOUT MIL.
      - Concatenates features directly into Nominal Softmax classification.
      - Purpose: Explicitly contrasts dual-branch nominal CE against our Ordinal CORAL/CLM framework.
    """
    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        return_features: bool = False
    ):
        super(DFD_Net_Nominal, self).__init__()
        
        # Branch 1: Global Context (ResNet-34)
        r34_weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        r34 = models.resnet34(weights=r34_weights)
        self.global_branch = nn.Sequential(
            r34.conv1, r34.bn1, r34.relu, r34.maxpool,
            r34.layer1, r34.layer2, r34.layer3, r34.layer4,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.dim_global = 512
        
        # Branch 2: High-Resolution Context (EfficientNet-B0)
        eff_weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        eff = models.efficientnet_b0(weights=eff_weights)
        self.local_branch = nn.Sequential(
            eff.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.dim_local = 1280
        
        self.latent_dim = self.dim_global + self.dim_local  # 512 + 1280 = 1792
        self.return_features = return_features
        
        if not return_features:
            self.classifier = nn.Sequential(
                nn.BatchNorm1d(self.latent_dim),
                nn.Dropout(p=dropout_rate),
                nn.Linear(self.latent_dim, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate / 2),
                nn.Linear(512, num_classes)
            )
        else:
            self.classifier = nn.Identity()

    def forward(
        self,
        global_img: torch.Tensor,
        local_img_or_patches: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Supports dual inputs:
          global_img: (B, 3, H, W)
          local_img_or_patches: (B, 3, H_loc, W_loc) or (B, K, 3, H_p, W_p).
        """
        f_global = self.global_branch(global_img)  # (B, 512)
        
        # If local input is patches (B, K, 3, H, W), take average over patches to simulate basic CNN
        if local_img_or_patches is not None:
            if local_img_or_patches.ndim == 5:
                # Average pooling over K patches instead of MIL max-pooling
                B, K, C, H, W = local_img_or_patches.shape
                flat = local_img_or_patches.view(B * K, C, H, W)
                feat = self.local_branch(flat).view(B, K, self.dim_local)
                f_local = feat.mean(dim=1)  # (B, 1280)
            else:
                f_local = self.local_branch(local_img_or_patches)
        else:
            # Fallback: process global image through local branch as well
            f_local = self.local_branch(global_img)
            
        fused = torch.cat([f_global, f_local], dim=1)  # (B, 1792)
        
        if self.return_features:
            return fused
        return self.classifier(fused)


# ==============================================================================
# BASELINE MODEL FACTORY
# ==============================================================================

BASELINE_REGISTRY = {
    "resnet50": ResNet50_Baseline,
    "efficientnet_b0": EfficientNetB0_Baseline,
    "densenet121": DenseNet121_Baseline,
    "efficientnet_b0_msag": EfficientNetB0_MSAG,
    "hybrid_effb0_vit": Hybrid_EffB0_ViT,
    "efficientnet_b3": EfficientNetB3_Baseline,
    "dfd_net_nominal": DFD_Net_Nominal
}

def get_baseline_model(
    model_name: str,
    num_classes: int = 5,
    pretrained: bool = True,
    return_features: bool = False,
    **kwargs
) -> nn.Module:
    """
    Factory function to instantiate any baseline model.
    
    Args:
        model_name: Name key from BASELINE_REGISTRY (case-insensitive).
        num_classes: Number of output severity classes (default 5).
        pretrained: Whether to load ImageNet pretrained weights.
        return_features: If True, returns latent feature vector instead of logits.
        
    Returns:
        nn.Module: Instantiated PyTorch baseline model.
    """
    key = model_name.lower().strip().replace("-", "_")
    if key not in BASELINE_REGISTRY:
        raise ValueError(
            f"Unknown baseline model '{model_name}'. Available baselines: {list(BASELINE_REGISTRY.keys())}"
        )
        
    model_cls = BASELINE_REGISTRY[key]
    return model_cls(
        num_classes=num_classes,
        pretrained=pretrained,
        return_features=return_features,
        **kwargs
    )
