
# ============================== CELL 16 ==============================
def apply_green_channel_ablation(
    img_rgb: np.ndarray, ablation_mode: str = "ben_graham_green"
) -> np.ndarray:
    """
    Applies fundus image preprocessing ablations centered on the Green channel.
    The Green channel provides the highest contrast for retinal vascular micro-lesions
    (microaneurysms, hemorrhages, and exudates).

    Args:
        img_rgb: Input fundus image array in RGB format (H, W, 3).
        ablation_mode: Preprocessing strategy:
            - 'ben_graham_green': Gaussian background subtraction on Green channel.
            - 'clahe_green': Contrast Limited Adaptive Histogram Equalization.
            - 'raw_green' (default/fallback): Unmodified Green channel.

    Returns:
        Processed 3-channel pseudo-RGB image array (H, W, 3).
    """
    # Convert RGB to BGR for OpenCV operations and apply edge-preserving bilateral denoising
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    denoised = cv2.bilateralFilter(bgr, d=9, sigmaColor=75, sigmaSpace=75)
    b, g, r = cv2.split(denoised)

    if ablation_mode == "ben_graham_green":
        # Ben Graham method: subtract local illumination background to normalize color variance
        blurred = cv2.GaussianBlur(g, (0, 0), 30)
        adjusted = cv2.addWeighted(g, 4, blurred, -4, 128)
        processed = cv2.merge([adjusted, adjusted, adjusted])

    elif ablation_mode == "clahe_green":
        # CLAHE: enhances local micro-contrast without over-amplifying background camera noise
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_g = clahe.apply(g)
        processed = cv2.merge([enhanced_g, enhanced_g, enhanced_g])

    else:
        # Fallback: replicate raw green channel across all 3 channels
        processed = cv2.merge([g, g, g])

    # Convert back to standard RGB format for downstream PyTorch transforms
    return cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

# ============================== CELL 18 ==============================
def get_global_augmentations(img_size: int = IMG_SIZE, is_train: bool = True) -> A.Compose:
    """
    Constructs the Albumentations transformation pipeline for the Global Context branch.
    Resizes whole-fundus images to fixed dimensions (IMG_SIZE x IMG_SIZE) and applies spatial/photometric
    augmentations during training while preserving anatomical integrity.
    Args:
        img_size: Target spatial resolution (height and width) in pixels.
        is_train: If True, applies randomized augmentations; otherwise, applies deterministic resize.
    Returns:
        Albumentations Compose transform pipeline.
    """
    if is_train:
        return A.Compose([
            # Deterministic bilinear interpolation resize
            A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_LINEAR),

            # Spatial invariances (rotational symmetry of retinal fundus morphology)
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.10,
                rotate_limit=30,
                p=0.5,
                border_mode=cv2.BORDER_CONSTANT
            ),

            # Photometric invariances (subtle illumination & camera exposure variations)
            A.RandomBrightnessContrast(
                brightness_limit=0.10,
                contrast_limit=0.10,
                p=0.5
            ),
        ])
    else:
        # Validation / Testing: deterministic resize only
        return A.Compose([
            A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_LINEAR)
        ])
# ============================== CELL 34 ==============================
class ChannelAttention(nn.Module):
    """
    Channel Attention Module for CBAM (Convolutional Block Attention Module).
    Exploits inter-channel relationships of features by combining spatial Average Pooling
    and Max Pooling descriptors through a shared bottleneck MLP.
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(ChannelAttention, self).__init__()
        # Spatial pooling descriptors: global context (AvgPool) & most salient features (MaxPool)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared bottleneck MLP using 1x1 convolutions (dimension reduction to preserve parameters)
        hidden_dim = max(8, in_channels // reduction_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=1, bias=False)
        )
        # Gating function producing channel-wise modulation weights in [0, 1]
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass: aggregate spatial information and compute channel attention map Mc(F)
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) # Output shape: [Batch, in_channels, 1, 1]
# ============================== CELL 35 ==============================
class SpatialAttention(nn.Module):
    """
    Spatial Attention Module for CBAM (Convolutional Block Attention Module).
    Captures inter-spatial relationships of features by computing channel-wise Average
    and Max statistics, followed by a large receptive field 7x7 convolution.
    """
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        # Same-padding ensures output spatial dimensions match input (H, W)
        padding = (kernel_size - 1) // 2
        # 2D Convolution mapping 2 aggregated channel descriptors -> 1 spatial attention map
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel-wise pooling along dimension 1 (channel axis) -> [Batch, 1, H, W] each
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate 2 descriptors along channel dimension -> [Batch, 2, H, W]
        combined = torch.cat([avg_out, max_out], dim=1)

        # Convolve and apply Sigmoid to generate 2D spatial attention weights Ms(F) in [0, 1]
        out = self.conv(combined)
        return self.sigmoid(out) # Output shape: [Batch, 1, H, W]
# ============================== CELL 38 ==============================
# ==============================================================================
# NON-LOCAL BLOCK — long-range spatial relational reasoning (addresses H2)
# ==============================================================================
class NonLocalBlock2D(nn.Module):
    """
    Embedded Gaussian Non-local block (Wang et al., 2018).

    Why this is needed for H2: CBAM's spatial attention is a single 7x7 conv
    applied to the ALREADY-pooled feature map - its receptive field only spans
    a fixed local neighborhood, so it can re-weight "this region matters more"
    but cannot let one spatial location directly relate to another far-away
    location. Grade 3's 4-2-1 rule needs exactly that: to know whether lesions
    appear in >= 2 or all 4 retinal quadrants, the network must compare
    evidence across quadrants that CBAM's local conv structurally cannot reach.

    Non-local computes attention between EVERY pair of spatial positions
    (full self-attention over the H*W feature map), so a lesion signal in the
    top-left quadrant can directly inform the representation at the
    bottom-right quadrant - the counting/relational operation CBAM lacks.
    """
    def __init__(self, in_channels: int, reduction: int = 2):
        super().__init__()
        self.inter_channels = max(1, in_channels // reduction)

        self.theta = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.phi   = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.g     = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.out_conv = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1)

        # Zero-init the output projection: the block starts as an identity
        # mapping (x + 0), so inserting it never destabilizes the pretrained
        # ResNet-50 backbone at the start of fine-tuning.
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        theta = self.theta(x).view(B, self.inter_channels, -1).permute(0, 2, 1)  # (B, HW, C')
        phi   = self.phi(x).view(B, self.inter_channels, -1)                     # (B, C', HW)
        g     = self.g(x).view(B, self.inter_channels, -1).permute(0, 2, 1)      # (B, HW, C')

        attn = torch.softmax(torch.bmm(theta, phi) / (self.inter_channels ** 0.5), dim=-1)  # (B, HW, HW)
        y = torch.bmm(attn, g)                       # (B, HW, C')
        y = y.permute(0, 2, 1).view(B, self.inter_channels, H, W)
        y = self.out_conv(y)
        return x + y  # residual connection


# ==============================================================================
# QUADRANT TOKENS — explicit 4-quadrant relational aggregation (addresses H2)
# ==============================================================================
class QuadrantTokenAggregator(nn.Module):
    """
    Splits the final feature map into the 4 spatial quadrants that match the
    clinical retinal quadrant convention used by the ICDR 4-2-1 rule for
    Grade 3 (hemorrhages in 4 quadrants / venous beading in >=2 / IRMA in
    >=1). Each quadrant is pooled into one token, then a small self-attention
    encoder (1-layer Transformer) lets the 4 quadrant tokens directly compare
    against each other before being aggregated - the explicit "how many
    quadrants show signal" reasoning that neither CBAM nor plain GAP performs
    (GAP just averages everything into one vector and destroys which quadrant
    contributed what).
    """
    def __init__(self, in_channels: int, num_heads: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_channels,
            nhead=num_heads,
            dim_feedforward=in_channels * 2,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Learned positional embedding so the encoder knows WHICH quadrant
        # each token came from (top-left/top-right/bottom-left/bottom-right),
        # not just an unordered set of 4 vectors.
        self.quadrant_pos_embed = nn.Parameter(torch.zeros(1, 4, in_channels))
        nn.init.trunc_normal_(self.quadrant_pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h_mid, w_mid = H // 2, W // 2
        quadrants = [
            x[:, :, :h_mid, :w_mid],   # top-left
            x[:, :, :h_mid, w_mid:],   # top-right
            x[:, :, h_mid:, :w_mid],   # bottom-left
            x[:, :, h_mid:, w_mid:],   # bottom-right
        ]
        tokens = torch.stack([self.pool(q).flatten(1) for q in quadrants], dim=1)  # (B, 4, C)
        tokens = tokens + self.quadrant_pos_embed
        tokens = self.encoder(tokens)          # self-attention across the 4 quadrant tokens
        quadrant_summary = tokens.mean(dim=1)  # (B, C) - aggregate relational signal
        return quadrant_summary
# ============================== CELL 42 ==============================
class GatedAttentionPool(nn.Module):
    """
    Gated Attention Mechanism for Multiple Instance Learning (Ilse et al., 2018).
    Learns instance-level attention weights a_k = softmax(w^T (tanh(V h_k) * sigm(U h_k))).
    Enables lesion counting and spatial density aggregation for Severe DR (Grade 3 - H2).
    """
    def __init__(self, in_features: int = 1280, hidden_dim: int = 128):
        super(GatedAttentionPool, self).__init__()
        self.attention_v = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.Tanh())
        self.attention_u = nn.Sequential(nn.Linear(in_features, hidden_dim), nn.Sigmoid())
        self.attention_weights = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, patch_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, K, in_features]
        patch_mask: Optional boolean tensor of shape [B, K] (True = real patch, False = padding patch)
        """
        a_v = self.attention_v(x)
        a_u = self.attention_u(x)
        a = self.attention_weights(a_v * a_u)  # [B, K, 1]

        # If patch_mask is provided from DataLoader (eval mode), mask out padding patch logits with -1e4
        if patch_mask is not None:
            mask_3d = patch_mask.unsqueeze(-1)  # [B, K, 1]
            a = a.masked_fill(~mask_3d, -1e4)

        a = torch.softmax(a, dim=1)  # [B, K, 1]
        self.last_attention_weights = a.detach()  # Retained for architecture-native Attention-MIL saliency
        weighted_out = torch.sum(x * a, dim=1)  # [B, in_features]
        return weighted_out


class LocalMILBranch(nn.Module):
    """
    Local Multi-Instance Learning (MIL) Branch with Dual Pooling:
      Input: (B, K, 3, 128, 128)
      Backbone: EfficientNet-B0 applied per patch
      Pooling: GAP per patch -> (B, K, 1280)
      Dual Aggregation:
        - 1D Max-Pooling (Captures extreme focal microaneurysms for Grade 1 - H1)
        - Gated Attention Pooling (Aggregates multi-lesion density for Grade 3 - H2)
      Output: F_Local feature vector of dimension 2560 (1280 + 1280) if use_dual_pooling else 1280
    """
    def __init__(self, pretrained: bool = True, use_dual_pooling: bool = True):
        super(LocalMILBranch, self).__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base_effnet = models.efficientnet_b0(weights=weights)

        self.features = base_effnet.features
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.patch_dim = 1280
        self.use_dual_pooling = use_dual_pooling

        if use_dual_pooling:
            self.gated_pool = GatedAttentionPool(in_features=self.patch_dim, hidden_dim=128)
            self.out_dim = self.patch_dim * 2  # 2560
        else:
            self.out_dim = self.patch_dim      # 1280

    def forward(self, patches: torch.Tensor, patch_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, K, C, H, W = patches.shape
        flat_patches = patches.view(B * K, C, H, W)
        patch_feats = self.features(flat_patches)
        patch_feats = self.gap(patch_feats)
        patch_feats = torch.flatten(patch_feats, 1)
        patch_feats = patch_feats.view(B, K, self.patch_dim)

        # 1. 1D Max-Pooling (Focal micro-lesions for H1)
        if patch_mask is not None:
            masked_feats = patch_feats.masked_fill(~patch_mask.unsqueeze(-1), -1e4)
            max_pooled, _ = torch.max(masked_feats, dim=1)
        else:
            max_pooled, _ = torch.max(patch_feats, dim=1)

        if self.use_dual_pooling:
            # 2. Gated Attention Pooling (Lesion counting for H2)
            att_pooled = self.gated_pool(patch_feats, patch_mask=patch_mask)
            f_local = torch.cat([max_pooled, att_pooled], dim=1)
        else:
            f_local = max_pooled

        return f_local


# ============================== CELL 44 ==============================
class DualBranchMILModel(nn.Module):
    """
    Unified Dual-Branch Multi-Scale Deep Learning Network:
      Combines Global Context (ResNet-50 + Non-Local + CBAM + Quadrant Tokens, 4096-d)
      and Local Micro-Lesions (MIL EfficientNet-B0 Dual Pooling, 2560-d).
      Outputs a compact, fused latent embedding D_latent = 512 for downstream classification/regression heads.
    """
    def __init__(
        self,
        pretrained: bool = True,
        use_cbam: bool = True,
        use_dual_pooling: bool = True,
        fusion_dim: int = 512,
        dropout_rate: float = 0.3
    ):
        super(DualBranchMILModel, self).__init__()

        self.global_branch = GlobalBranch(pretrained=pretrained, use_cbam=use_cbam)
        self.local_mil_branch = LocalMILBranch(pretrained=pretrained, use_dual_pooling=use_dual_pooling)

        combined_dim = self.global_branch.out_dim + self.local_mil_branch.out_dim  # 4096 + 2560 = 6656 (or 4608)

        # Robust Fusion Projection Module (using LayerNorm to eliminate batch size 1 crash)
        self.fusion_layer = nn.Sequential(
            nn.Linear(combined_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.Mish(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim)
        )
        self.latent_dim = fusion_dim

    def forward(
        self,
        global_img: torch.Tensor,
        local_patches: torch.Tensor,
        patch_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        f_global = self.global_branch(global_img)
        f_local = self.local_mil_branch(local_patches, patch_mask=patch_mask)
        f_fused = torch.cat([f_global, f_local], dim=1)
        latent = self.fusion_layer(f_fused)
        return latent

# ============================== CELL 57 ==============================
# ==============================================================================
# SECTION 3.5: CUMULATIVE LINK MODEL (CLM_QWK) OUTPUT HEAD
# ==============================================================================
class CumulativeLinkModelQWKHead(nn.Module):
    """
    Champion Output Architecture: Cumulative Link Model (CLM) with Complementary Log-Log (clog-log)
    Link Function & Hybrid Continuous QWK Loss with Linear Warmup.

    Mathematical Formulation:
      - Latent Severity Score: s(x) = W^T x (scalar continuous representation)
      - Strictly Monotonic Cutpoints: theta_1 < theta_2 < theta_3 < theta_4 parameterized
        via softplus increments: theta_k = theta_{k-1} + softplus(alpha_k) + 1e-4
      - Complementary Log-Log (clog-log) link: F(z) = 1 - exp(-exp(z))
      - Cumulative Probability: P(Y <= k) = 1 - exp(-exp(theta_k - s(x)))
      - Rank Probability: P(Y > k) = 1 - P(Y <= k)
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
        self.target_qwk_weight = qwk_weight
        self.current_qwk_weight = 0.0  # Warmup starts at 0.0

        self.score_fc = nn.Linear(in_features, 1, bias=False)
        self.first_cutpoint = nn.Parameter(torch.tensor(-1.5))
        self.cutpoint_increments = nn.Parameter(torch.ones(self.num_cutpoints - 1) * -0.5)

        self.qwk_loss = ContinuousQWKLoss(num_classes=num_classes)

    def set_qwk_weight(self, weight: float):
        """Dynamically updates the lambda weight for continuous QWK loss during warmup."""
        self.current_qwk_weight = weight

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

        return {
            "severity_score": severity_score,
            "latent_score": severity_score,  # Retained for compatibility across extraction utilities
            "cutpoints": cutpoints,
            "thetas": cutpoints,             # Alias for Section 6.2 visualization functions
            "cum_probs": cum_probs,
            "rank_probs": rank_probs,
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
        probs = outputs["probs"]
        B = targets.size(0)

        eps = 1e-7
        target_probs = probs[torch.arange(B, device=targets.device), targets]
        nll_loss = -torch.log(torch.clamp(target_probs, min=eps))

        if class_weights is not None:
            weights_per_sample = class_weights[targets]
            nll_loss = (nll_loss * weights_per_sample).mean()
        else:
            nll_loss = nll_loss.mean()

        qwk = self.qwk_loss(probs, targets)
        total_loss = nll_loss + self.current_qwk_weight * qwk
        return total_loss
# ============================== CELL 59 ==============================
class DRFullModel(nn.Module):
    """
    End-to-End Deep Learning Architecture integrating the DualBranchMILModel backbone
    with any of the 5 selectable Ordinal/Nominal Classification Heads for ablation benchmarking.
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
        # Dynamically retrieve latent embedding dimension (default: 512)
        latent_dim = getattr(backbone, "latent_dim", 512)

        # Factory dispatch to instantiate the designated classification/ordinal head
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

    def forward(self, global_img: torch.Tensor, local_patches: torch.Tensor, patch_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Unified forward pass: passes multi-scale inputs through backbone, then through active head.
        """
        latent = self.backbone(global_img, local_patches, patch_mask=patch_mask)  # (Batch, 512)
        outputs = self.head(latent)                        # Compute predictions from selected head
        outputs["latent"] = latent                         # Preserve latent representation for feature analysis/XAI
        return outputs

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Delegates loss calculation directly to the active head's specialized loss function.
        """
        return self.head.loss_fn(outputs, targets, class_weights=class_weights)

# ============================== CELL 61 ==============================
def build_model(
    head_type: str = "CLM_QWK",
    pretrained: bool = True,
    use_cbam: bool = True,
    fusion_dim: int = 512,
    num_classes: int = 5,
    qwk_weight: float = 1.2
) -> DRFullModel:
    """
    Factory function to construct the complete multi-scale Diabetic Retinopathy pipeline model
    configured with the designated classification or ordinal regression output head.

    Args:
        head_type: Selected output head architecture:
                   ['Softmax_CE', 'Softmax_QWK', 'CORAL', 'CORN', 'CLM_QWK'].
        pretrained: Whether convolutional backbones load pre-trained ImageNet weights.
        use_cbam: Whether to apply CBAM Channel & Spatial Attention on ResNet-50.
        fusion_dim: Dimension of the latent multi-scale fused embedding (default: 512).
        num_classes: Number of ordinal severity stages (default: 5 grades, 0 to 4).
        qwk_weight: Loss weight multiplier lambda for the differentiable continuous QWK term.

    Returns:
        DRFullModel: Fully assembled and end-to-end differentiable PyTorch model.
    """
    # 1. Instantiate multi-scale dual-branch backbone (Global ResNet50-CBAM + Local MIL EfficientNet-B0)
    backbone = DualBranchMILModel(
        pretrained=pretrained,
        use_cbam=use_cbam,
        fusion_dim=fusion_dim
    )

    # 2. Wrap backbone with the designated ordinal/nominal classification head
    model = DRFullModel(
        backbone=backbone,
        head_type=head_type,
        num_classes=num_classes,
        qwk_weight=qwk_weight
    )
    return model