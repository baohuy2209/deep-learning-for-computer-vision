import json, ast, os

# Read the current main_model_lodo.ipynb
with open("notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb", "r", encoding="utf-8") as f:
    orig_nb = json.load(f)

print(f"Original cell count: {len(orig_nb['cells'])}")

# We will construct a clean list of cells
clean_cells = []

# Cell 0: Title markdown
clean_cells.append(orig_nb["cells"][0])
# Cell 1: Environment Setup markdown
clean_cells.append(orig_nb["cells"][1])
# Cell 2: Imports
clean_cells.append(orig_nb["cells"][2])
# Cell 3: Plotting style
clean_cells.append(orig_nb["cells"][3])

# Cell 4: Paths, Drive Mounting, Hyperparameters, ARCHITECTURE_VARIANTS
c4_code = """# ==============================================================================
# 1.1 AUTOMATIC GOOGLE DRIVE MOUNTING (COLAB OR LOCAL)
# ==============================================================================
import os
import time

try:
    from google.colab import drive
    if not os.path.exists('/content/drive/MyDrive'):
        print('Connecting to Google Drive...')
        drive.mount('/content/drive')
    else:
        print('Google Drive already mounted at /content/drive.')
except ImportError:
    print('Running in local environment (google.colab not available). Drive mount skipped.')

# ==============================================================================
# 1.2 DATASET & IMAGE ROOT PATHS (MATCHING main_model (6).ipynb)
# ==============================================================================
# 1. Unpack Image Archive: native_cropped.zip -> /content/dataset_local
zip_source = '/content/drive/MyDrive/native_cropped.zip'
extract_dest = '/content/dataset_local'

if os.path.exists(zip_source):
    if not os.path.exists(extract_dest) or len(os.listdir(extract_dest)) == 0:
        os.makedirs(extract_dest, exist_ok=True)
        print(f"Extracting image archive '{zip_source}' to '{extract_dest}'...")
        t0 = time.time()
        get_ipython().system('unzip -q "$zip_source" -d "$extract_dest"')
        print(f"-> Extracted image archive in {time.time() - t0:.1f}s!")
    else:
        print(f"-> Local image directory '{extract_dest}' already populated ({len(os.listdir(extract_dest))} cohorts).")

DATA_ROOT = '/content/dataset_local' if os.path.exists('/content/dataset_local') else os.path.abspath('data/raw/hybrid')
image_root_dir = os.path.join(DATA_ROOT, "native_cropped") if os.path.exists(os.path.join(DATA_ROOT, "native_cropped")) else DATA_ROOT
print(f"[PATHS] Image Root Directory: '{image_root_dir}' (Exists: {os.path.exists(image_root_dir)})")

# 2. Unpack Pre-Generated 6-Fold LODO Splits: lodo_folds.zip -> /content/lodo_folds
lodo_zip_source = '/content/drive/MyDrive/lodo_folds.zip'
lodo_extract_dest = '/content/lodo_folds'

if os.path.exists(lodo_zip_source):
    if not os.path.exists(lodo_extract_dest) or len(os.listdir(lodo_extract_dest)) == 0:
        os.makedirs(lodo_extract_dest, exist_ok=True)
        print(f"Extracting LODO splits archive '{lodo_zip_source}' to '{lodo_extract_dest}'...")
        t0 = time.time()
        get_ipython().system('unzip -q "$lodo_zip_source" -d "$lodo_extract_dest"')
        print(f"-> Extracted LODO splits in {time.time() - t0:.1f}s!")
    LODO_FOLDS_DIR = lodo_extract_dest
elif os.path.exists('/content/drive/MyDrive/lodo_folds'):
    LODO_FOLDS_DIR = '/content/drive/MyDrive/lodo_folds'
elif os.path.exists('data/processed/lodo_folds'):
    LODO_FOLDS_DIR = os.path.abspath('data/processed/lodo_folds')
else:
    LODO_FOLDS_DIR = lodo_extract_dest

print(f"[PATHS] LODO Folds Directory : '{LODO_FOLDS_DIR}' (Exists: {os.path.exists(LODO_FOLDS_DIR)})")

# ==============================================================================
# 1.3 EXPERIMENTAL SETUP & HYPERPARAMETERS
# ==============================================================================
DRIVE_BASE = '/content/drive/MyDrive/Luận Văn Tốt Nghiệp 2026 - 2027/code' if os.path.exists('/content/drive/MyDrive') else '.'
RESULT_ROOT = os.path.join(DRIVE_BASE, 'results')
RESULTS_DIR_LODO    = os.path.join(RESULT_ROOT, "lodo_benchmark")
CHECKPOINT_DIR_LODO = os.path.join(RESULTS_DIR_LODO, "checkpoints")
CACHE_DIR_LODO      = "/content/cache_fundus_dual_branch"
TEST_CACHE_DRIVE_DIR= os.path.join(RESULTS_DIR_LODO, "test_cache_zips")
FIGURES_DIR         = os.path.join(RESULTS_DIR_LODO, "figures")
FOLDS_DIR           = LODO_FOLDS_DIR

os.makedirs(CHECKPOINT_DIR_LODO, exist_ok=True)
os.makedirs(CACHE_DIR_LODO, exist_ok=True)
os.makedirs(TEST_CACHE_DRIVE_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

SMOKE_TEST = False
NUM_EPOCHS = 100
LODO_EPOCHS = 3 if SMOKE_TEST else NUM_EPOCHS
BATCH_SIZE = 16
IMG_SIZE = 512
LEARNING_RATE = 1e-4
MIN_LR = 1e-6
WEIGHT_DECAY = 1e-4
COSINE_T_MAX = 30
EARLY_STOPPING_PATIENCE = 15
TARGET_QWK_WEIGHT = 0.5
QWK_WARMUP_EPOCHS = 15
NUM_WORKERS = 4

SUMMARY_JSON_PATH = os.path.join(RESULTS_DIR_LODO, "lodo_summary.json")
SUMMARY_CSV_PATH  = os.path.join(RESULTS_DIR_LODO, "lodo_summary.csv")

# 6 Clinical Cohorts in Benchmark
DATASET_CONFIG = {
    "aptos": {"name": "APTOS 2019", "folder": "APTOS", "num_classes": 5},
    "idrid": {"name": "IDRiD", "folder": "IDRiD", "num_classes": 5},
    "messidor2": {"name": "Messidor-2", "folder": "Messidor-2", "num_classes": 5},
    "ddr": {"name": "DDR", "folder": "DDR", "num_classes": 5},
    "eyepacs": {"name": "EyePACS", "folder": "EyePACS", "num_classes": 5},
    "deepdrid": {"name": "DeepDRiD", "folder": "DeepDRiD", "num_classes": 5},
}
DATASET_KEYS = list(DATASET_CONFIG.keys())

# ==============================================================================
# 1.4 ARCHITECTURE ABLATION WATERFALL CONFIGURATION (V0 -> V1 -> V2 -> V3)
# ==============================================================================
ARCHITECTURE_VARIANTS = [
    {
        "name": "V0_Baseline_Softmax_GlobalOnly",
        "head_type": "Softmax_CE",
        "use_local_branch": False,
        "use_nonlocal": False,
        "use_quadrant_tokens": False,
        "use_cbam": True,
        "run": True,
        "hypothesis": "Baseline: ResNet-50 + CBAM Global Only + Softmax CE",
    },
    {
        "name": "V1_PlusLocalMIL_H1",
        "head_type": "Softmax_CE",
        "use_local_branch": True,
        "use_nonlocal": False,
        "use_quadrant_tokens": False,
        "use_cbam": True,
        "run": True,
        "hypothesis": "H1: Adds Patch-based Local MIL Branch",
    },
    {
        "name": "V2_PlusGlobalContext_H2",
        "head_type": "Softmax_CE",
        "use_local_branch": True,
        "use_nonlocal": True,
        "use_quadrant_tokens": True,
        "use_cbam": True,
        "run": True,
        "hypothesis": "H2: Adds Non-local Attention & Quadrant Tokens to Global View",
    },
    {
        "name": "V3_PlusOrdinalHead_H3_Champion",
        "head_type": "CLM_QWK",
        "use_local_branch": True,
        "use_nonlocal": True,
        "use_quadrant_tokens": True,
        "use_cbam": True,
        "run": False,  # Merged from existing lodo_summary.json without retraining
        "hypothesis": "H3: Champion Dual-Branch MIL with Ordinal CLM Head",
    },
]
"""
clean_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [l + "\n" for l in c4_code.strip().split("\n")]
})

# Section 2 Markdown: Multi-Scale Retinal Data Pipeline (Matching main_model (6).ipynb)
clean_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 2. Multi-Scale Retinal Data Pipeline (Matching main_model (6).ipynb)\n"]
})

# Cell 10 in orig_nb is the preprocessing & caching functions
c10_cell = orig_nb["cells"][10]
c10_src = "".join(c10_cell["source"])

# Update path resolution in precompute_dataset_cache to exact main_model (6) format
old_c10_path = """        if img_root and not os.path.isabs(native_path_str) and not os.path.exists(native_path_str):
            img_path = os.path.join(img_root, dataset_key, PureWindowsPath(native_path_str).name)
        else:
            img_path = native_path_str"""

new_c10_path = """        if img_root and not os.path.isabs(native_path_str) and not os.path.exists(native_path_str):
            img_path = f"{img_root}/{dataset_key}/{PureWindowsPath(native_path_str).name}"
        else:
            img_path = native_path_str"""

if old_c10_path in c10_src:
    c10_src = c10_src.replace(old_c10_path, new_c10_path, 1)

clean_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [l + "\n" for l in c10_src.split("\n")]
})

# Cell 11 in orig_nb is FundusDualBranchDataset
c11_src = """# ==============================================================================
# 2.2 FUNDUS DUAL-BRANCH PYTORCH DATASET (MATCHING main_model (6).ipynb)
# ==============================================================================
class FundusDualBranchDataset(Dataset):
    def __init__(
        self,
        metadata: Union[pd.DataFrame, str],
        img_size: int = IMG_SIZE,
        img_root: Optional[str] = None,
        split: str = "train",
        ablation_mode: str = "ben_graham_green",
        cache_dir: Optional[str] = None,
        patch_size: int = 128,
        stride: int = 96
    ):
        \"\"\"
        Dataset serving the proposed Dual-Branch model (Global + Local MIL) on Google Colab.
        Matches main_model (6).ipynb path resolution:
          img_path = f"{img_root}/{dataset_key}/{PureWindowsPath(native_path_str).name}"
        \"\"\"
        if isinstance(metadata, str):
            self.df = pd.read_csv(metadata)
        else:
            self.df = metadata.reset_index(drop=True)

        self.records = self.df.to_dict("records")
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride
        self.split = split
        self.img_root = img_root
        self.ablation_mode = ablation_mode
        self.is_train = (split == "train")
        self.cache_dir = cache_dir

        self.global_transform = get_global_augmentations(
            img_size, is_train=self.is_train
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records[idx]
        label = int(row.get("diagnosis", 0))

        # 1. Resolve image path (exact logic from main_model (6).ipynb)
        native_path_str = str(row.get("native_path", row.get("raw_full_path", "")))
        dataset_key = row.get("dataset_key", "default")
        if self.img_root and not os.path.isabs(native_path_str) and not os.path.exists(native_path_str):
            img_path = f"{self.img_root}/{dataset_key}/{PureWindowsPath(native_path_str).name}"
        else:
            img_path = native_path_str

        # 2. Check if precomputed NPZ cache is available
        img_id = PureWindowsPath(native_path_str).stem if native_path_str else str(row.get("image_id", idx))
        cache_file = os.path.join(self.cache_dir, dataset_key, f"{img_id}.npz") if self.cache_dir else None
        processed_np = None
        valid_anchors = None

        if cache_file and os.path.exists(cache_file):
            try:
                with np.load(cache_file) as data:
                    processed_np = data["img"]
                    valid_anchors = data["anchors"]
            except Exception:
                processed_np, valid_anchors = None, None

        if processed_np is None:
            bgr = cv2.imread(img_path)
            if bgr is not None:
                img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            else:
                try:
                    pil_img = Image.open(img_path).convert("RGB")
                    img_rgb = np.array(pil_img)
                except Exception as e:
                    logger.warning(f"Unable to read image at '{img_path}': {e}. Using synthetic fallback.")
                    img_rgb = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

            fov_mask = compute_fov_mask(img_rgb, erode_px=15)
            processed_np = apply_green_channel_ablation(img_rgb, ablation_mode=self.ablation_mode)
            valid_anchors = compute_valid_anchors(
                img_rgb.shape[:2], fov_mask, patch_size=self.patch_size, stride=self.stride, fov_thresh=0.5
            )

        # Global stream (uint8)
        global_augmented = self.global_transform(image=processed_np)
        global_img = global_augmented["image"]
        global_tensor = torch.from_numpy(global_img).permute(2, 0, 1).contiguous()

        return {
            "global_image": global_tensor,
            "local_native_image": processed_np,
            "valid_anchors": valid_anchors,
            "native_path": img_path,
            "label": torch.tensor(label, dtype=torch.long),
            "dataset_key": dataset_key,
            "patient_id": str(row.get("patient_id", f"unknown::{idx}")),
        }
"""
clean_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [l + "\n" for l in c11_src.strip().split("\n")]
})

# Section 3 Markdown: Architecture
clean_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 3. Deep Neural Network Architecture & Ordinal Heads\n"]
})

# CBAM (cell 13 in orig)
clean_cells.append(orig_nb["cells"][13])
# GlobalBranch (cell 14 in orig)
clean_cells.append(orig_nb["cells"][14])

# DualBranchMILModel with ablation support
c15_code = """# ==============================================================================
# 3.3 MULTIMODAL DUAL-BRANCH LATENT FUSION (WITH ABLATION WATERFALL SUPPORT)
# ==============================================================================
class LocalMILBranch(nn.Module):
    def __init__(self, pretrained: bool = True, feature_dim: int = 512, mil_dim: int = 256):
        super().__init__()
        import torchvision.models as models
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
        self.out_dim = mil_dim
        self.mil_pool = GatedAttentionPool(in_features=feature_dim, hidden_dim=mil_dim, out_dim=mil_dim)

    def forward(self, local_patches: torch.Tensor, patch_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C, H, W = local_patches.shape
        patches_flat = local_patches.view(B * N, C, H, W)
        features_flat = self.feature_extractor(patches_flat)
        features_pooled = F.adaptive_avg_pool2d(features_flat, (1, 1)).view(B * N, -1)
        patch_features = features_pooled.view(B, N, -1)
        f_local, attn_weights = self.mil_pool(patch_features, mask=patch_mask)
        return f_local, attn_weights


class DualBranchMILModel(nn.Module):
    \"\"\"
    Dual-Branch Architecture supporting full ablation waterfall:
    - Global Branch: ResNet50 + CBAM + Optional Non-local attention + Optional Quadrant tokens
    - Local MIL Branch: Patch extraction + ResNet-18 + Gated Attention MIL pooling
    - Dynamic Fusion: Projects combined branch representations to fusion_dim
    \"\"\"
    def __init__(
        self,
        pretrained: bool = True,
        use_cbam: bool = True,
        use_local_branch: bool = True,
        use_nonlocal: bool = True,
        use_quadrant_tokens: bool = True,
        fusion_dim: int = 512,
        dropout: float = 0.3
    ):
        super().__init__()
        self.use_local_branch = use_local_branch
        self.use_nonlocal = use_nonlocal
        self.use_quadrant_tokens = use_quadrant_tokens
        self.fusion_dim = fusion_dim

        # 1. Global Branch (forwards use_nonlocal and use_quadrant_tokens)
        self.global_branch = GlobalBranch(
            pretrained=pretrained,
            use_cbam=use_cbam,
            use_nonlocal=use_nonlocal,
            use_quadrant_tokens=use_quadrant_tokens
        )

        # 2. Local MIL Branch (instantiated only when use_local_branch=True)
        if self.use_local_branch:
            self.local_mil_branch = LocalMILBranch(pretrained=pretrained)
            self.combined_dim = self.global_branch.out_dim + self.local_mil_branch.out_dim
        else:
            self.local_mil_branch = None
            self.combined_dim = self.global_branch.out_dim

        # 3. Dynamic Fusion Layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.combined_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        global_images: torch.Tensor,
        local_patches: Optional[torch.Tensor] = None,
        patch_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # 1. Global Branch Forward
        f_global = self.global_branch(global_images)

        # 2. Local Branch Forward (skipped when use_local_branch=False)
        if self.use_local_branch and self.local_mil_branch is not None:
            if local_patches is not None and patch_mask is not None:
                f_local, patch_attn_weights = self.local_mil_branch(local_patches, patch_mask)
                f_combined = torch.cat([f_global, f_local], dim=1)
            else:
                f_combined = f_global
                patch_attn_weights = None
        else:
            f_combined = f_global
            patch_attn_weights = None

        # 3. Dynamic Fusion
        f_fused = self.fusion_layer(f_combined)
        return f_fused, patch_attn_weights
"""
clean_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [l + "\n" for l in c15_code.strip().split("\n")]
})

# Heads & build_model
c16_code = """# ==============================================================================
# 3.4 OUTPUT HEADS (SOFTMAX CE & CLM QWK) AND UNIFIED MODEL FACTORY BUILDER
# ==============================================================================
class SoftmaxCEHead(nn.Module):
    def __init__(self, in_features: int = 512, num_classes: int = 5):
        super().__init__()
        self.classifier = nn.Linear(in_features, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, features: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        logits = self.classifier(features)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        out = {"logits": logits, "probs": probs, "preds": preds}
        if targets is not None:
            out["loss"] = self.loss_fn(logits, targets)
        return out


class CumulativeLinkModelQWKHead(nn.Module):
    def __init__(self, in_features: int = 512, num_classes: int = 5, qwk_weight: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        self.num_cutpoints = num_classes - 1
        self.qwk_weight = qwk_weight

        self.latent_proj = nn.Linear(in_features, 1, bias=False)
        init_cutpoints = torch.tensor([-1.5, -0.5, 0.5, 1.5], dtype=torch.float32)
        self.cutpoint_init = nn.Parameter(init_cutpoints[0].unsqueeze(0))
        self.cutpoint_log_diffs = nn.Parameter(torch.log(torch.diff(init_cutpoints)))
        self.qwk_loss_fn = ContinuousQWKLoss(num_classes=num_classes)

    def get_cutpoints(self) -> torch.Tensor:
        diffs = torch.exp(self.cutpoint_log_diffs)
        subsequent = torch.cumsum(diffs, dim=0)
        cutpoints = torch.cat([self.cutpoint_init, self.cutpoint_init + subsequent])
        return cutpoints

    def forward(self, features: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        eta = self.latent_proj(features).squeeze(-1)
        cutpoints = self.get_cutpoints()
        cutpoints_expanded = cutpoints.unsqueeze(0).expand(eta.size(0), -1)
        eta_expanded = eta.unsqueeze(-1).expand(-1, self.num_cutpoints)

        cum_probs = torch.sigmoid(cutpoints_expanded - eta_expanded)
        pad_0 = torch.zeros(eta.size(0), 1, device=eta.device)
        pad_1 = torch.ones(eta.size(0), 1, device=eta.device)
        extended_cum = torch.cat([pad_0, cum_probs, pad_1], dim=1)
        probs = extended_cum[:, 1:] - extended_cum[:, :-1]
        probs = torch.clamp(probs, min=1e-7, max=1.0)
        probs = probs / probs.sum(dim=1, keepdim=True)

        class_values = torch.arange(self.num_classes, dtype=torch.float32, device=eta.device)
        expected_grade = torch.sum(probs * class_values.unsqueeze(0), dim=1)
        preds = torch.round(expected_grade).long().clamp(0, self.num_classes - 1)

        out = {
            "eta": eta,
            "cutpoints": cutpoints,
            "cum_probs": cum_probs,
            "probs": probs,
            "expected_grade": expected_grade,
            "preds": preds,
        }

        if targets is not None:
            nll_loss = F.nll_loss(torch.log(probs), targets)
            qwk_loss = self.qwk_loss_fn(expected_grade, targets)
            total_loss = (1.0 - self.qwk_weight) * nll_loss + self.qwk_weight * qwk_loss
            out["loss"] = total_loss
            out["nll_loss"] = nll_loss.detach()
            out["qwk_loss"] = qwk_loss.detach()

        return out


class DRFullModel(nn.Module):
    def __init__(self, backbone: DualBranchMILModel, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(
        self,
        global_images: torch.Tensor,
        local_patches: Optional[torch.Tensor] = None,
        patch_mask: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        features, patch_attn_weights = self.backbone(global_images, local_patches, patch_mask)
        outputs = self.head(features, targets=targets)
        outputs["patch_attn_weights"] = patch_attn_weights
        return outputs


def build_model(
    pretrained: bool = True,
    use_cbam: bool = True,
    use_local_branch: bool = True,
    use_nonlocal: bool = True,
    use_quadrant_tokens: bool = True,
    fusion_dim: int = 512,
    num_classes: int = 5,
    head_type: str = "CLM_QWK",
    qwk_weight: float = 0.5,
    dropout: float = 0.3
) -> DRFullModel:
    \"\"\"
    Model factory preserving champion defaults byte-for-byte.
    \"\"\"
    backbone = DualBranchMILModel(
        pretrained=pretrained,
        use_cbam=use_cbam,
        use_local_branch=use_local_branch,
        use_nonlocal=use_nonlocal,
        use_quadrant_tokens=use_quadrant_tokens,
        fusion_dim=fusion_dim,
        dropout=dropout
    )
    if head_type.upper() in ["SOFTMAX_CE", "SOFTMAX", "CE"]:
        head = SoftmaxCEHead(in_features=fusion_dim, num_classes=num_classes)
    elif head_type.upper() in ["CLM_QWK", "CLM", "ORDINAL"]:
        head = CumulativeLinkModelQWKHead(in_features=fusion_dim, num_classes=num_classes, qwk_weight=qwk_weight)
    else:
        raise ValueError(f"Unsupported head_type: {head_type}")

    return DRFullModel(backbone=backbone, head=head)
"""
clean_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [l + "\n" for l in c16_code.strip().split("\n")]
})

# Section 4 Markdown: Clinical Metrics & PyTorch Training Engine
clean_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 4. Clinical Metrics & PyTorch Training Engine\n"]
})

# Metrics (orig cell 17)
clean_cells.append(orig_nb["cells"][17])
# DRTrainer (orig cell 18)
clean_cells.append(orig_nb["cells"][18])

# Section 5 Markdown: LODO Orchestration Loop
clean_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 5. 6-Fold Leave-One-Dataset-Out (LODO) Orchestration Loop\n"]
})

# LODO Fold Loop (orig cell 20) with clean direct CSV reading matching main_model (6).ipynb
c20_src = """# ==============================================================================
# 5.1 MULTI-VARIANT LEAVE-ONE-DATASET-OUT (LODO) ABLATION ORCHESTRATION LOOP
# ==============================================================================
lodo_results = {}
eval_fold_artifacts = {}

# Ingest existing Champion results if summary_json exists
existing_champion_results = {}
if os.path.exists(SUMMARY_JSON_PATH):
    try:
        with open(SUMMARY_JSON_PATH, "r") as f:
            prev_sum = json.load(f)
            if "V3_PlusOrdinalHead_H3_Champion" in prev_sum:
                existing_champion_results = prev_sum["V3_PlusOrdinalHead_H3_Champion"]
            elif any(k in DATASET_KEYS for k in prev_sum.keys()):
                existing_champion_results = prev_sum
        print(f"[INFO] Ingested existing results for {len(existing_champion_results)} folds from '{SUMMARY_JSON_PATH}'!")
    except Exception as e:
        print(f"[WARN] Unable to load summary_json: {e}")

if "V3_PlusOrdinalHead_H3_Champion" not in lodo_results and existing_champion_results:
    if any(k in DATASET_KEYS for k in existing_champion_results.keys()):
        lodo_results["V3_PlusOrdinalHead_H3_Champion"] = existing_champion_results
        print("[INFO] Successfully merged existing Champion results into 'V3_PlusOrdinalHead_H3_Champion'!")

print(f"\\n{'='*75}")
print("STARTING MULTI-VARIANT ARCHITECTURE ABLATION LODO BENCHMARK")
print(f"{'='*75}\\n")

train_collate_fn = partial(trainer_compatible_collate_fn, mode="train", num_sample=(32, 64))
eval_collate_fn = partial(trainer_compatible_collate_fn, mode="eval", max_patches_eval=256)

# ==============================================================================
# OUTER LOOP: ITERATE OVER ARCHITECTURE-ABLATION VARIANTS (V0 -> V1 -> V2 -> V3)
# ==============================================================================
for variant in ARCHITECTURE_VARIANTS:
    variant_name = variant["name"]
    should_run = variant.get("run", True)
    target_folds = variant.get("folds", DATASET_KEYS[:2] if SMOKE_TEST else DATASET_KEYS)

    if variant_name not in lodo_results:
        lodo_results[variant_name] = {}

    print(f"\\n{'#'*80}")
    print(f"### VARIANT: {variant_name} | Run={should_run} | Folds={len(target_folds)} | Head={variant.get('head_type')}")
    print(f"{'#'*80}")

    if not should_run:
        if variant_name in lodo_results and len(lodo_results[variant_name]) > 0:
            print(f"[SKIP] Variant '{variant_name}' already populated with {len(lodo_results[variant_name])} completed folds.")
        elif existing_champion_results and any(k in DATASET_KEYS for k in existing_champion_results.keys()):
            lodo_results[variant_name] = existing_champion_results
            print(f"[MERGED] Ingested {len(existing_champion_results)} folds from existing champion summary.")
        else:
            print(f"[WARN] No existing results found for '{variant_name}'. Set 'run': True to train.")
        continue

    # ==========================================================================
    # INNER LOOP: ITERATE OVER HELD-OUT COHORTS FOR THIS VARIANT
    # ==========================================================================
    for fold_idx, held_out_key in enumerate(target_folds):
        held_out_name = DATASET_CONFIG[held_out_key]["name"]
        print(f"\\n{'='*75}")
        print(f"[{variant_name}] FOLD {fold_idx + 1}/{len(target_folds)}: Holding out [{held_out_name}]")
        print(f"{'='*75}")

        fold_model_name = f"lodo_{variant_name}_fold_{fold_idx}_{held_out_key}"
        ckpt_path = os.path.join(CHECKPOINT_DIR_LODO, f"{fold_model_name}_best.pth")
        last_ckpt_path = os.path.join(CHECKPOINT_DIR_LODO, f"{fold_model_name}_last.pth")

        # Skip already completed fold to save Colab Compute Units
        if held_out_key in lodo_results[variant_name] and os.path.exists(ckpt_path):
            try:
                ckpt_meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                if ckpt_meta.get("is_completed", False):
                    print(f"[SKIP] Fold {fold_idx + 1} ({held_out_name}) for {variant_name} ALREADY COMPLETED!")
                    continue
            except Exception:
                pass

        # 1. Directly read pre-generated LODO fold CSVs (matching main_model (6).ipynb pattern)
        train_csv = os.path.join(LODO_FOLDS_DIR, f"fold_{fold_idx}_{held_out_key}_train.csv")
        val_csv   = os.path.join(LODO_FOLDS_DIR, f"fold_{fold_idx}_{held_out_key}_val.csv")
        test_csv  = os.path.join(LODO_FOLDS_DIR, f"fold_{fold_idx}_{held_out_key}_test.csv")

        train_df = pd.read_csv(train_csv)
        val_df   = pd.read_csv(val_csv)
        ood_test_df = pd.read_csv(test_csv)
        print(f"  [DATA] Fold {fold_idx + 1} ({held_out_name}) Loaded from '{LODO_FOLDS_DIR}':")
        print(f"    Train: {len(train_df):,d} images | Val: {len(val_df):,d} images | Test (OOD): {len(ood_test_df):,d} images")

        # 2. Caching & DataLoaders
        ensure_test_cache_on_drive(
            test_df=ood_test_df,
            held_out_key=held_out_key,
            local_cache_dir=CACHE_DIR_LODO,
            drive_cache_dir=TEST_CACHE_DRIVE_DIR,
            img_root=image_root_dir
        )
        precompute_dataset_cache(val_df, img_root=image_root_dir, cache_dir=CACHE_DIR_LODO)

        train_ds = FundusDualBranchDataset(train_df, img_size=IMG_SIZE, split="train", cache_dir=CACHE_DIR_LODO, img_root=image_root_dir)
        val_ds   = FundusDualBranchDataset(val_df, img_size=IMG_SIZE, split="val", cache_dir=CACHE_DIR_LODO, img_root=image_root_dir)
        test_ds  = FundusDualBranchDataset(ood_test_df, img_size=IMG_SIZE, split="test", cache_dir=CACHE_DIR_LODO, img_root=image_root_dir)

        num_w = NUM_WORKERS
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=num_w, collate_fn=train_collate_fn, pin_memory=torch.cuda.is_available())
        val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_w, collate_fn=eval_collate_fn, pin_memory=torch.cuda.is_available())
        test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=num_w, collate_fn=eval_collate_fn, pin_memory=torch.cuda.is_available())

        # 3. Build Parametric Model Architecture for this Variant
        fold_model = build_model(
            head_type=variant.get("head_type", "CLM_QWK"),
            pretrained=True,
            use_cbam=variant.get("use_cbam", True),
            use_local_branch=variant.get("use_local_branch", True),
            use_nonlocal=variant.get("use_nonlocal", True),
            use_quadrant_tokens=variant.get("use_quadrant_tokens", True),
            fusion_dim=512,
            num_classes=5,
            qwk_weight=TARGET_QWK_WEIGHT
        )
        optimizer = torch.optim.AdamW(fold_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=MIN_LR)

        # 4. Checkpoint Resumption (Mid-Training Recovery)
        start_epoch = 1
        best_val_qwk = -1.0
        epochs_no_improve = 0
        if os.path.exists(last_ckpt_path):
            try:
                ckpt = torch.load(last_ckpt_path, map_location=device, weights_only=False)
                fold_model.load_state_dict(ckpt["model_state_dict"])
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                if "lr_scheduler_state_dict" in ckpt and lr_scheduler is not None:
                    lr_scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])
                start_epoch = ckpt.get("epoch", 0) + 1
                best_val_qwk = ckpt.get("best_val_qwk", -1.0)
                epochs_no_improve = ckpt.get("epochs_no_improve", 0)
                print(f"[RESUME] Resuming Fold {fold_idx + 1} from epoch {start_epoch}/{LODO_EPOCHS} (Best Val QWK so far: {best_val_qwk:.4f})")
            except Exception as e:
                print(f"[WARN] Failed to resume from last checkpoint: {e}. Starting from epoch 1.")

        trainer = DRTrainer(
            model=fold_model,
            optimizer=optimizer,
            device=device,
            checkpoint_dir=CHECKPOINT_DIR_LODO,
            model_name=fold_model_name,
            lr_scheduler=lr_scheduler,
            early_stopping_patience=EARLY_STOPPING_PATIENCE,
            save_best_only=False,
            num_classes=5,
            head_type=variant.get("head_type", "CLM_QWK")
        )
        trainer.start_epoch = start_epoch
        trainer.best_val_qwk = best_val_qwk
        trainer.epochs_no_improve = epochs_no_improve

        # 5. Execute Training Loop
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=LODO_EPOCHS,
            eval_every=1,
            test_loader=test_loader
        )

        # 6. Load Best Checkpoint for Final Comprehensive Evaluation
        if os.path.exists(ckpt_path):
            best_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            fold_model.load_state_dict(best_ckpt["model_state_dict"])
            print(f"[RESTORE] Loaded best weights from '{ckpt_path}' for final LODO validation & OOD testing.")

        fold_model.eval()
        val_eval_metrics = trainer.evaluate(val_loader, split_name=f"InDomain_Val_{held_out_name}")
        ood_eval_metrics = trainer.evaluate(test_loader, split_name=f"OOD_Test_{held_out_name}")

        delta_lodo = val_eval_metrics["qwk"] - ood_eval_metrics["qwk"]
        print(f"\\n>>> [{variant_name} | Fold {fold_idx + 1} ({held_out_name}) Results]:")
        print(f"    In-Domain Val QWK : {val_eval_metrics['qwk']:.4f}")
        print(f"    OOD Test QWK      : {ood_eval_metrics['qwk']:.4f}")
        print(f"    Generalization Drop (Delta LODO) : {delta_lodo:.4f}")

        fold_result_entry = {
            "variant": variant_name,
            "held_out_dataset": held_out_name,
            "held_out_key": held_out_key,
            "fold": fold_idx + 1,
            "in_domain_val_qwk": float(val_eval_metrics["qwk"]),
            "ood_test_qwk": float(ood_eval_metrics["qwk"]),
            "delta_lodo": float(delta_lodo),
            "delta_qwk": float(delta_lodo),
            "val_per_grade_recall": val_eval_metrics.get("per_grade_recall", {}),
            "ood_per_grade_recall": ood_eval_metrics.get("per_grade_recall", {}),
            "val_within_1_grade_acc": float(val_eval_metrics.get("within_1_grade_acc", 0.0)),
            "ood_within_1_grade_acc": float(ood_eval_metrics.get("within_1_grade_acc", 0.0)),
            "val_referable_sensitivity_at_95spec": float(val_eval_metrics.get("referable_sens_at_95spec", 0.0)),
            "ood_referable_sensitivity_at_95spec": float(ood_eval_metrics.get("referable_sens_at_95spec", 0.0)),
            "val_referable_sensitivity_at_85spec": float(val_eval_metrics.get("referable_sens_at_85spec", 0.0)),
            "ood_referable_sensitivity_at_85spec": float(ood_eval_metrics.get("referable_sens_at_85spec", 0.0)),
            "val_monotonicity_violations": "N/A" if "SOFTMAX" in variant.get("head_type", "").upper() else float(val_eval_metrics.get("monotonicity_violations", 0.0)),
            "ood_monotonicity_violations": "N/A" if "SOFTMAX" in variant.get("head_type", "").upper() else float(ood_eval_metrics.get("monotonicity_violations", 0.0)),
        }
        lodo_results[variant_name][held_out_key] = fold_result_entry

        # Save checkpoint completion flag
        if os.path.exists(ckpt_path):
            ckpt_meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            ckpt_meta["is_completed"] = True
            ckpt_meta["metrics"] = fold_result_entry
            torch.save(ckpt_meta, ckpt_path)

        # Save incremental results
        with open(SUMMARY_JSON_PATH, "w") as f:
            json.dump(lodo_results, f, indent=2)

        flat_records = []
        for v_k, v_dict in lodo_results.items():
            for c_k, c_metrics in v_dict.items():
                flat_records.append(c_metrics)
        if flat_records:
            pd.DataFrame(flat_records).to_csv(SUMMARY_CSV_PATH, index=False)

print(f"\\n{'='*75}")
print(f"[SUCCESS] Multi-Variant Architecture-Ablation LODO Orchestration Completed!")
print(f"Results persisted to '{SUMMARY_JSON_PATH}' and '{SUMMARY_CSV_PATH}'")
print(f"{'='*75}\\n")
"""
clean_cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [l + "\n" for l in c20_src.strip().split("\n")]
})

# Section 6 Markdown: Visualization Suite
clean_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 6. Publication-Quality LODO Visualization Suite\n"]
})

# Artifact extraction helpers (orig cell 22)
clean_cells.append(orig_nb["cells"][22])
# Fig 1 & 2 (orig cell 23)
clean_cells.append(orig_nb["cells"][23])
# Fig 3, 4, 5, 7 plotting functions (orig cell 24 & 25)
clean_cells.append(orig_nb["cells"][24])
clean_cells.append(orig_nb["cells"][25])
# Fig execution cell (orig cell 26)
clean_cells.append(orig_nb["cells"][26])

print(f"Cleaned cell count: {len(clean_cells)}")

# Verify AST on all code cells
for i, c in enumerate(clean_cells):
    if c["cell_type"] == "code":
        src = "".join(c["source"])
        ast.parse(src)
        print(f"  [OK] Cell {i:02d} AST validated ({len(src):,d} chars)")

new_nb = {
    "cells": clean_cells,
    "metadata": orig_nb.get("metadata", {}),
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb", "w", encoding="utf-8") as f:
    json.dump(new_nb, f, indent=1, ensure_ascii=False)

print("[SUCCESS] Successfully regenerated notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb matching main_model (6).ipynb!")
