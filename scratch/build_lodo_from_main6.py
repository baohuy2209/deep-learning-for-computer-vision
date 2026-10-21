import json
import os
import re

# 1. Load source notebook main_model (6).ipynb
source_nb_path = 'notebooks/diabetic_retinopathy/pipeline/model/main_model (6).ipynb'
target_nb_path = 'notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb'

with open(source_nb_path, 'r', encoding='utf-8') as f:
    nb6 = json.load(f)

def get_src(idx):
    return ''.join(nb6['cells'][idx]['source'])

def make_code_cell(source_str):
    lines = [l + '\n' for l in source_str.split('\n')]
    if lines and lines[-1] == '\n':
        lines.pop()
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    }

def make_md_cell(source_str):
    lines = [l + '\n' for l in source_str.split('\n')]
    if lines and lines[-1] == '\n':
        lines.pop()
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines
    }

cells = []

# --- Cell 0: Header Markdown ---
cells.append(make_md_cell("""# 6-Fold Leave-One-Dataset-Out (LODO) Cross-Domain Generalization Benchmark
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This notebook implements the definitive **6-Fold Leave-One-Dataset-Out (LODO)** generalization benchmark and **Architecture-Ablation Waterfall ($V_0 \\to V_1 \\to V_2 \\to V_3$)**, inheriting the exact dual-branch ordinal architecture and training engine from `main_model (6).ipynb`.

For each fold, 1 of the 6 clinical cohorts is held out as the unseen out-of-distribution (OOD) test set, while training occurs on the other 5 cohorts using frozen pre-generated splits directly from `/content/lodo_folds/{filename}`."""))

# --- Cell 1: Section 1 Markdown ---
cells.append(make_md_cell("""## 1. Environment Setup & Publication Configuration
Initializes deep learning libraries, logging infrastructure, academic plotting standards, Google Drive mounting, and automated image archive unpacking."""))

# --- Cell 2: Imports (from main_model (6) Cell 4) ---
imports_src = get_src(4)
cells.append(make_code_cell(imports_src))

# --- Cell 3: Plotting configuration (from main_model (6) Cell 12 & 13) ---
plot_config_src = get_src(12) + "\n\n" + get_src(13)
cells.append(make_code_cell(plot_config_src))

# --- Cell 4: Setup Paths, Drive Mount, Unzip, and LODO Directory Setup ---
paths_setup_src = """# ==============================================================================
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
# 1.2 AUTOMATED DATASET UNPACKING TO COLAB LOCAL FAST SSD (/content/dataset_local)
# ==============================================================================
zip_source = '/content/drive/MyDrive/native_cropped.zip'
extract_dest = '/content/dataset_local'

if os.path.exists(zip_source):
    if not os.path.exists(extract_dest) or len(os.listdir(extract_dest)) == 0:
        os.makedirs(extract_dest, exist_ok=True)
        print(f"Extracting image archive '{zip_source}' to '{extract_dest}'...")
        t0 = time.time()
        get_ipython().system('unzip -q "$zip_source" -d "$extract_dest"')
        print(f"-> Extraction completed in {time.time() - t0:.1f} seconds!")
        get_ipython().system('ls -lh "$extract_dest"')
    else:
        print(f"-> Dataset directory '{extract_dest}' already populated ({len(os.listdir(extract_dest))} items).")
else:
    print(f"[INFO] zip_source '{zip_source}' not found or running in local environment.")

DATA_ROOT = '/content/dataset_local' if os.path.exists('/content/dataset_local') else os.path.abspath('data/raw/hybrid')
image_root_dir = f"{DATA_ROOT}/native_cropped" if os.path.exists(f"{DATA_ROOT}/native_cropped") else DATA_ROOT
print(f"[PATHS] Image Root Directory: '{image_root_dir}' (Exists: {os.path.exists(image_root_dir)})")

# ==============================================================================
# 1.3 LODO FOLD SPLITS DIRECTORY SETUP (/content/lodo_folds)
# ==============================================================================
lodo_zip_source = '/content/drive/MyDrive/lodo_folds.zip'
lodo_extract_dest = '/content/lodo_folds'

if os.path.exists(lodo_zip_source):
    if not os.path.exists(lodo_extract_dest) or len(os.listdir(lodo_extract_dest)) == 0:
        os.makedirs(lodo_extract_dest, exist_ok=True)
        print(f"Extracting LODO splits archive '{lodo_zip_source}' to '{lodo_extract_dest}'...")
        t0 = time.time()
        get_ipython().system('unzip -q "$lodo_zip_source" -d "$lodo_extract_dest"')
        print(f"-> Extraction completed in {time.time() - t0:.1f} seconds!")

if os.path.exists('/content/lodo_folds'):
    LODO_FOLDS_DIR = '/content/lodo_folds'
elif os.path.exists('data/processed/lodo_folds'):
    LODO_FOLDS_DIR = os.path.abspath('data/processed/lodo_folds')
elif os.path.exists('/content/drive/MyDrive/lodo_folds'):
    LODO_FOLDS_DIR = '/content/drive/MyDrive/lodo_folds'
else:
    LODO_FOLDS_DIR = '/content/lodo_folds'

print(f"[PATHS] LODO Folds Directory : '{LODO_FOLDS_DIR}' (Exists: {os.path.exists(LODO_FOLDS_DIR)})")

# ==============================================================================
# 1.4 RESULT ROOT & DIRECTORIES CONFIGURATION
# ==============================================================================
DRIVE_BASE = '/content/drive/MyDrive/Luận Văn Tốt Nghiệp 2026 - 2027/code' if os.path.exists('/content/drive/MyDrive') else '.'
RESULT_ROOT = os.path.join(DRIVE_BASE, 'results')
RESULTS_DIR_LODO    = os.path.join(RESULT_ROOT, "lodo_benchmark")
CHECKPOINT_DIR_LODO = os.path.join(RESULTS_DIR_LODO, "checkpoints")
CACHE_DIR_LODO      = "/content/cache_fundus_dual_branch"
FIGURES_DIR         = os.path.join(RESULTS_DIR_LODO, "figures")

os.makedirs(CHECKPOINT_DIR_LODO, exist_ok=True)
os.makedirs(CACHE_DIR_LODO, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
"""
cells.append(make_code_cell(paths_setup_src))

# --- Cell 5: Global Hyperparameters & Architecture Ablation Waterfall Configuration ---
config_src = """# ==============================================================================
# 1.5 GLOBAL EXPERIMENTAL CONFIGURATION & HYPERPARAMETERS
# ==============================================================================
SMOKE_TEST = False

SEED = 42
num_workers = min(8, os.cpu_count() or 4)
NUM_WORKERS = num_workers
use_pin_memory = torch.cuda.is_available()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

IMG_SIZE = 512
NUM_CLASSES = 5
HEAD_TYPE = "CLM_QWK"
FUSION_DIM = 512
PRETRAINED = True
USE_CBAM = True
USE_NONLOCAL = True
USE_QUADRANT_TOKENS = True

TARGET_QWK_WEIGHT = 0.4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MIN_LR = 1e-6

PATIENCE_EARLY_STOPPING = 15
PATIENCE_REDUCE_LR = 4
REDUCE_LR_FACTOR = 0.5
GRAD_CLIP = 1.0

if SMOKE_TEST:
    print("[SMOKE TEST MODE]: Running rapid sanity check on pipeline flow (3 epochs).")
    EPOCHS = 3
    LODO_EPOCHS = 3
    BATCH_SIZE = 8
    QWK_WARMUP_EPOCHS = 1
    T_MAX = 3
else:
    print("[FULL RUN MODE]: Executing official 100-epoch LODO benchmark (CosineAnnealing T_max=30).")
    EPOCHS = 100
    LODO_EPOCHS = 100
    BATCH_SIZE = 16
    QWK_WARMUP_EPOCHS = 15
    T_MAX = 30

SUMMARY_JSON_PATH = os.path.join(RESULTS_DIR_LODO, "lodo_summary.json")
SUMMARY_CSV_PATH  = os.path.join(RESULTS_DIR_LODO, "lodo_summary.csv")

# 6 Clinical Cohorts in Leave-One-Dataset-Out (LODO) Benchmark
DATASET_CONFIG = {
    "aptos": {"name": "APTOS 2019", "folder": "APTOS", "num_classes": 5},
    "idrid": {"name": "IDRiD", "folder": "IDRiD", "num_classes": 5},
    "messidor2": {"name": "Messidor-2", "folder": "Messidor-2", "num_classes": 5},
    "ddr": {"name": "DDR", "folder": "DDR", "num_classes": 5},
    "eyepacs": {"name": "EyePACS", "folder": "EyePACS", "num_classes": 5},
    "deepdrid": {"name": "DeepDRiD", "folder": "DeepDRiD", "num_classes": 5},
}
DATASET_KEYS = list(DATASET_CONFIG.keys())

# Architecture Ablation Waterfall (V0 -> V1 -> V2 -> V3) testing Advisor Hypotheses H1, H2, H3
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
        "run": False,  # Champion model metrics automatically merged from lodo_summary.json
        "hypothesis": "H3: Champion Dual-Branch MIL with Ordinal CLM Head",
    },
]
"""
cells.append(make_code_cell(config_src))

# --- Cell 6: Section 2 Markdown ---
cells.append(make_md_cell("""# 2. Multi-Scale Retinal Data Pipeline (Exact main_model (6).ipynb)
Implements Green-Channel background subtraction, global data augmentations, `FundusDualBranchDataset`, FOV segmentation, sliding-window patch extraction, and `trainer_compatible_collate_fn`."""))

# --- Cell 7: Green-channel preprocessing (main_model (6) Cell 16) ---
cells.append(make_code_cell(get_src(16)))

# --- Cell 8: Global context data augmentations (main_model (6) Cell 18) ---
cells.append(make_code_cell(get_src(18)))

# --- Cell 9: FundusDualBranchDataset (main_model (6) Cell 20) ---
dataset_src = get_src(20)
cells.append(make_code_cell(dataset_src))

# --- Cell 10: FOV Mask, Anchors, Sliding Window, Collate (main_model (6) Cell 23) ---
collate_src = get_src(23)
cells.append(make_code_cell(collate_src))

# --- Cell 11: Section 3 Markdown ---
cells.append(make_md_cell("""# 3. Deep Neural Network Architecture & Ordinal Heads
Implements CBAM, Non-Local Self-Attention, Quadrant Token Aggregation, Global Branch, Local MIL Branch, Dual-Branch Fusion, and the full Head Ablation Suite (`SoftmaxCEHead`, `SoftmaxQWKHead`, `CORALHead`, `CORNHead`, `CumulativeLinkModelQWKHead`)."""))

# --- Cell 12: CBAM (ChannelAttention, SpatialAttention, CBAM - main_model (6) Cells 34, 35, 36) ---
cbam_src = get_src(34) + "\n\n" + get_src(35) + "\n\n" + get_src(36)
cells.append(make_code_cell(cbam_src))

# --- Cell 13: NonLocalBlock2D & QuadrantTokenAggregator (main_model (6) Cell 38) ---
cells.append(make_code_cell(get_src(38)))

# --- Cell 14: GlobalBranch (main_model (6) Cell 40) ---
cells.append(make_code_cell(get_src(40)))

# --- Cell 15: GatedAttentionPool & LocalMILBranch (main_model (6) Cell 42) ---
cells.append(make_code_cell(get_src(42)))

# --- Cell 16: DualBranchMILModel with Parametric Ablation Flags ---
dual_branch_src = """class DualBranchMILModel(nn.Module):
    \"\"\"
    Unified Dual-Branch Multi-Scale Deep Learning Network:
      Combines Global Context (ResNet-50 + Non-Local + CBAM + Quadrant Tokens, 4096-d)
      and Local Micro-Lesions (MIL EfficientNet-B0 Dual Pooling, 2560-d).
      Supports parametric flags (use_local_branch, use_nonlocal, use_quadrant_tokens)
      to execute the full ablation waterfall (V0 -> V1 -> V2 -> V3).
    \"\"\"
    def __init__(
        self,
        pretrained: bool = True,
        use_cbam: bool = True,
        use_dual_pooling: bool = True,
        use_local_branch: bool = True,
        use_nonlocal: bool = True,
        use_quadrant_tokens: bool = True,
        fusion_dim: int = 512,
        dropout_rate: float = 0.3
    ):
        super(DualBranchMILModel, self).__init__()

        self.use_local_branch = use_local_branch
        self.global_branch = GlobalBranch(
            pretrained=pretrained,
            use_cbam=use_cbam,
            use_nonlocal=use_nonlocal,
            use_quadrant_tokens=use_quadrant_tokens
        )

        if use_local_branch:
            self.local_mil_branch = LocalMILBranch(pretrained=pretrained, use_dual_pooling=use_dual_pooling)
            combined_dim = self.global_branch.out_dim + self.local_mil_branch.out_dim
        else:
            self.local_mil_branch = None
            combined_dim = self.global_branch.out_dim

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
        local_patches: Optional[torch.Tensor] = None,
        patch_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        f_global = self.global_branch(global_img)
        if self.use_local_branch and self.local_mil_branch is not None and local_patches is not None:
            f_local = self.local_mil_branch(local_patches, patch_mask=patch_mask)
            f_fused = torch.cat([f_global, f_local], dim=1)
        else:
            f_fused = f_global
        latent = self.fusion_layer(f_fused)
        return latent
"""
cells.append(make_code_cell(dual_branch_src))

# --- Cell 17: Output Heads (main_model (6) Cells 47, 49, 51, 53, 55, 57) ---
heads_src = (
    get_src(47) + "\n\n" +
    get_src(49) + "\n\n" +
    get_src(51) + "\n\n" +
    get_src(53) + "\n\n" +
    get_src(55) + "\n\n" +
    get_src(57)
)
cells.append(make_code_cell(heads_src))

# --- Cell 18: DRFullModel & build_model (main_model (6) Cells 59 & 61 with parametric passthrough) ---
model_builder_src = """# ==============================================================================
# 3.6 END-TO-END MODEL WRAPPER & UNIFIED FACTORY BUILDER
# ==============================================================================
class DRFullModel(nn.Module):
    def __init__(
        self,
        backbone: DualBranchMILModel,
        head_type: str = "CLM_QWK",
        num_classes: int = 5,
        qwk_weight: float = 0.4
    ):
        super(DRFullModel, self).__init__()
        self.backbone = backbone
        self.head_type = head_type
        self.num_classes = num_classes
        in_features = backbone.latent_dim

        if head_type == "Softmax_CE":
            self.head = SoftmaxCEHead(in_features=in_features, num_classes=num_classes)
        elif head_type == "Softmax_QWK":
            self.head = SoftmaxQWKHead(in_features=in_features, num_classes=num_classes, qwk_weight=qwk_weight)
        elif head_type == "CORAL":
            self.head = CORALHead(in_features=in_features, num_classes=num_classes)
        elif head_type == "CORN":
            self.head = CORNHead(in_features=in_features, num_classes=num_classes)
        elif head_type == "CLM_QWK":
            self.head = CumulativeLinkModelQWKHead(in_features=in_features, num_classes=num_classes, qwk_weight=qwk_weight)
        else:
            raise ValueError(f"Unknown head_type: '{head_type}'.")

    def forward(
        self,
        global_img: torch.Tensor,
        local_patches: Optional[torch.Tensor] = None,
        patch_mask: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        return_loss: bool = True
    ) -> Dict[str, Any]:
        latent = self.backbone(global_img, local_patches=local_patches, patch_mask=patch_mask)
        head_outputs = self.head(latent, targets=targets, return_loss=return_loss)
        head_outputs["latent"] = latent
        return head_outputs


def build_model(
    head_type: str = "CLM_QWK",
    pretrained: bool = True,
    use_cbam: bool = True,
    use_local_branch: bool = True,
    use_nonlocal: bool = True,
    use_quadrant_tokens: bool = True,
    fusion_dim: int = 512,
    num_classes: int = 5,
    qwk_weight: float = 0.4
) -> DRFullModel:
    backbone = DualBranchMILModel(
        pretrained=pretrained,
        use_cbam=use_cbam,
        use_local_branch=use_local_branch,
        use_nonlocal=use_nonlocal,
        use_quadrant_tokens=use_quadrant_tokens,
        fusion_dim=fusion_dim
    )
    model = DRFullModel(
        backbone=backbone,
        head_type=head_type,
        num_classes=num_classes,
        qwk_weight=qwk_weight
    )
    return model
"""
cells.append(make_code_cell(model_builder_src))

# --- Cell 19: Section 4 Markdown ---
cells.append(make_md_cell("""# 4. Clinical Evaluation Metrics & DRTrainer Engine
Implements clinical screening metrics (referable sensitivity @ 95% specificity), Quadratic Weighted Kappa (QWK), rank probability extraction, and the full high-performance `DRTrainer` engine with mixed-precision AMP."""))

# --- Cell 20: Metrics & Per-Grade Report (main_model (6) Cells 64, 65, 66, 69) ---
metrics_src = (
    get_src(64) + "\n\n" +
    get_src(65) + "\n\n" +
    get_src(66) + "\n\n" +
    get_src(69)
)
cells.append(make_code_cell(metrics_src))

# --- Cell 21: DRTrainer (main_model (6) Cell 68) ---
trainer_src = get_src(68)
cells.append(make_code_cell(trainer_src))

# --- Cell 22: Section 5 Markdown ---
cells.append(make_md_cell("""# 5. 6-Fold Leave-One-Dataset-Out (LODO) Orchestration Loop
Orchestrates the 6-fold LODO cross-domain benchmark across the 4 architecture variants ($V_0 \\to V_1 \\to V_2 \\to V_3$).
Directly reads pre-generated frozen CSV splits from `/content/lodo_folds/{filename}`:
- `fold_{fold_idx}_{held_out_key}_train.csv`
- `fold_{fold_idx}_{held_out_key}_val.csv`
- `fold_{fold_idx}_{held_out_key}_test.csv`"""))

# --- Cell 23: Complete LODO Orchestration Loop Code ---
lodo_loop_src = """# ==============================================================================
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
eval_collate_fn = partial(trainer_compatible_collate_fn, mode="eval")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for v_idx, variant in enumerate(ARCHITECTURE_VARIANTS):
    variant_name = variant["name"]
    is_run_enabled = variant.get("run", True)
    hypothesis = variant.get("hypothesis", "")

    print(f"\\n{'#'*75}")
    print(f"VARIANT [{v_idx + 1}/{len(ARCHITECTURE_VARIANTS)}]: {variant_name}")
    print(f"Hypothesis : {hypothesis}")
    print(f"Run Status : {'ENABLED (Executing)' if is_run_enabled else 'SKIPPED (Using Saved Results)'}")
    print(f"{'#'*75}\\n")

    if not is_run_enabled:
        print(f"[INFO] Skipping training for {variant_name} (Preserved existing results).")
        continue

    if variant_name not in lodo_results:
        lodo_results[variant_name] = {}
    if variant_name not in eval_fold_artifacts:
        eval_fold_artifacts[variant_name] = {}

    target_folds = variant.get("folds", DATASET_KEYS)

    for fold_idx, held_out_key in enumerate(target_folds):
        held_out_name = DATASET_CONFIG[held_out_key]["name"]
        print(f"\\n{'='*70}")
        print(f"[{variant_name}] FOLD {fold_idx + 1}/{len(target_folds)}: Holding out [{held_out_name}] as OOD Test")
        print(f"{'='*70}")

        fold_model_name = f"lodo_{variant_name}_fold_{fold_idx}_{held_out_key}"
        ckpt_path = os.path.join(CHECKPOINT_DIR_LODO, f"{fold_model_name}_best.pth")
        last_ckpt_path = os.path.join(CHECKPOINT_DIR_LODO, f"{fold_model_name}_last.pth")

        # Skip if fold is already completed
        if os.path.exists(ckpt_path):
            try:
                ckpt_meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                if ckpt_meta.get("is_completed", False):
                    print(f"[SKIP] Fold {fold_idx + 1} ({held_out_name}) for {variant_name} ALREADY COMPLETED!")
                    saved_metrics = ckpt_meta.get("metrics", {})
                    if saved_metrics:
                        lodo_results[variant_name][held_out_key] = saved_metrics
                    continue
            except Exception as e:
                print(f"[WARN] Error reading checkpoint completion flag: {e}")

        # 1. Load Pre-Generated Split CSVs directly from /content/lodo_folds/
        train_csv = os.path.join(LODO_FOLDS_DIR, f"fold_{fold_idx}_{held_out_key}_train.csv")
        val_csv   = os.path.join(LODO_FOLDS_DIR, f"fold_{fold_idx}_{held_out_key}_val.csv")
        test_csv  = os.path.join(LODO_FOLDS_DIR, f"fold_{fold_idx}_{held_out_key}_test.csv")

        if not os.path.exists(train_csv):
            raise FileNotFoundError(
                f"[ERROR] Required LODO split CSV '{train_csv}' not found.\\n"
                f"Please ensure CSV files are uploaded to '{LODO_FOLDS_DIR}/fold_{fold_idx}_{held_out_key}_{{train,val,test}}.csv'."
            )

        print(f"  [DATA] Fold {fold_idx + 1} ({held_out_name}) Loading CSVs from '{LODO_FOLDS_DIR}':")
        print(f"    - Train: {train_csv}")
        print(f"    - Val:   {val_csv}")
        print(f"    - Test:  {test_csv}")

        # 2. Build PyTorch Dual-Branch Datasets & DataLoaders (Exact main_model (6).ipynb)
        train_ds = FundusDualBranchDataset(
            csv_path=train_csv,
            img_root=image_root_dir,
            img_size=IMG_SIZE,
            split="train",
            ablation_mode="ben_graham_green",
            cache_dir=CACHE_DIR_LODO
        )
        val_ds = FundusDualBranchDataset(
            csv_path=val_csv,
            img_root=image_root_dir,
            img_size=IMG_SIZE,
            split="val",
            ablation_mode="ben_graham_green",
            cache_dir=CACHE_DIR_LODO
        )
        test_ds = FundusDualBranchDataset(
            csv_path=test_csv,
            img_root=image_root_dir,
            img_size=IMG_SIZE,
            split="test",
            ablation_mode="ben_graham_green",
            cache_dir=CACHE_DIR_LODO
        )

        print(f"  [DATA] Samples Count: Train={len(train_ds):,d} | Val={len(val_ds):,d} | Test (OOD)={len(test_ds):,d}")

        eval_batch_size = max(1, BATCH_SIZE // 4)
        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
            num_workers=NUM_WORKERS, collate_fn=train_collate_fn,
            pin_memory=use_pin_memory, persistent_workers=(NUM_WORKERS > 0)
        )
        val_loader = DataLoader(
            val_ds, batch_size=eval_batch_size, shuffle=False,
            num_workers=NUM_WORKERS, collate_fn=eval_collate_fn,
            pin_memory=use_pin_memory, persistent_workers=(NUM_WORKERS > 0)
        )
        test_loader = DataLoader(
            test_ds, batch_size=eval_batch_size, shuffle=False,
            num_workers=NUM_WORKERS, collate_fn=eval_collate_fn,
            pin_memory=use_pin_memory, persistent_workers=(NUM_WORKERS > 0)
        )

        # 3. Build Parametric Model Architecture for this Variant
        fold_model = build_model(
            head_type=variant.get("head_type", "CLM_QWK"),
            pretrained=True,
            use_cbam=variant.get("use_cbam", True),
            use_local_branch=variant.get("use_local_branch", True),
            use_nonlocal=variant.get("use_nonlocal", True),
            use_quadrant_tokens=variant.get("use_quadrant_tokens", True),
            fusion_dim=FUSION_DIM,
            num_classes=NUM_CLASSES,
            qwk_weight=TARGET_QWK_WEIGHT
        )
        optimizer = torch.optim.AdamW(fold_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX, eta_min=MIN_LR)

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
            early_stopping_patience=PATIENCE_EARLY_STOPPING,
            save_best_only=False,
            num_classes=NUM_CLASSES,
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

        # 6. Load Best Checkpoint for Final Evaluation
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

        # Save incremental summary JSON and CSV
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
cells.append(make_code_cell(lodo_loop_src))

# --- Cell 24: Section 6 Markdown ---
cells.append(make_md_cell("""# 6. Publication-Quality LODO Visualization Suite
Generates the 8-figure publication suite ($300\\text{ DPI}$) summarizing dataset distribution, architecture ablation waterfall ($V_0 \\to V_3$), per-grade slope charts, LODO generalization heatmap, ROC curves, and attention fidelity. All figures are saved with a `(2)` suffix to prevent overwriting existing Drive files."""))

# --- Cell 25: Visualization Helper Functions (from current main_model_lodo.ipynb Cell 19) ---
with open(target_nb_path, 'r', encoding='utf-8') as f:
    prev_lodo = json.load(f)

for prev_idx in [19, 20, 21, 22, 23]:
    if prev_idx < len(prev_lodo['cells']):
        cells.append(prev_lodo['cells'][prev_idx])

# Assemble final notebook dictionary
new_nb = {
    "cells": cells,
    "metadata": nb6.get("metadata", {}),
    "nbformat": 4,
    "nbformat_minor": 5
}

with open(target_nb_path, 'w', encoding='utf-8') as f:
    json.dump(new_nb, f, indent=1)

print(f"Successfully rebuilt {target_nb_path} with {len(cells)} cells directly from {source_nb_path}!")
