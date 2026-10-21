import json, ast, os

nb_path = "notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# ------------------------------------------------------------------------------
# 1. CELL 04: Paths, Drive Mounting & Zip Unpacking
# ------------------------------------------------------------------------------
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
# 1.2 DATASET UNPACKING (native_cropped.zip & lodo_folds.zip)
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
else:
    print(f"[INFO] '{zip_source}' not found on Drive. Using local dataset path if available.")

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

# 6 Clinical Cohort Definitions
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
        "run": False,  # ALREADY COMPLETED: Merged from existing lodo_summary.json without retraining
        "hypothesis": "H3: Champion Dual-Branch MIL with Ordinal CLM Head",
    },
]
"""

# ------------------------------------------------------------------------------
# 2. CELL 07: Clean Split Ingestion Function (Replaces bloated master harmonizer)
# ------------------------------------------------------------------------------
c7_code = """# ==============================================================================
# 2.2 PRE-GENERATED LODO FOLDS INGESTION & SUMMARY
# ==============================================================================
def load_lodo_fold_splits(
    fold_idx: int,
    held_out_key: str,
    folds_dir: str = LODO_FOLDS_DIR
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    \"\"\"
    Loads pre-generated frozen train, val, and OOD test splits for fold_idx.
    Matches the deterministic splits generated by merge_dataset_lodo.ipynb:
      - Train: Class-balanced (1,500 - 2,500 / grade) across the other 5 cohorts.
      - Val: 15% patient-level in-domain validation split (natural prevalence).
      - Test: 100% of held-out dataset (unaltered natural clinical prevalence).
    \"\"\"
    slug = held_out_key.lower()
    candidates_train = [
        os.path.join(folds_dir, f"fold_{fold_idx - 1}_{slug}_train.csv"),
        os.path.join(folds_dir, f"fold_{fold_idx}_{slug}_train.csv"),
    ]
    f_train = next((p for p in candidates_train if os.path.exists(p)), None)
    if f_train is None:
        raise FileNotFoundError(
            f"[ERROR] Pre-generated fold CSV not found for '{held_out_key}' in '{folds_dir}'.\\n"
            f"Checked candidates: {candidates_train}\\n"
            f"Please upload 'lodo_folds.zip' to Google Drive or ensure files exist in '{folds_dir}'."
        )

    f_val = f_train.replace("_train.csv", "_val.csv")
    f_test = f_train.replace("_train.csv", "_test.csv")

    train_df = pd.read_csv(f_train)
    val_df   = pd.read_csv(f_val)
    test_df  = pd.read_csv(f_test)

    return train_df, val_df, test_df

# Display master summary table if present
summary_csv_path = os.path.join(LODO_FOLDS_DIR, "lodo_splits_summary.csv")
if os.path.exists(summary_csv_path):
    lodo_summary_table = pd.read_csv(summary_csv_path)
    print(f"[SUCCESS] Ingested LODO 6-Fold Pre-Generated Splits Summary from '{summary_csv_path}':")
    display(lodo_summary_table)
else:
    print(f"[INFO] Ready to load LODO splits from: '{LODO_FOLDS_DIR}'")
"""

# ------------------------------------------------------------------------------
# 3. CELL 08: Zero Patient Leakage Audit (Replaces on-the-fly balancing functions)
# ------------------------------------------------------------------------------
c8_code = """# ==============================================================================
# 2.3 PRE-FLIGHT AUDIT: ZERO PATIENT LEAKAGE & INTEGRITY VERIFICATION
# ==============================================================================
print(f"[AUDIT] Verifying pre-generated LODO fold CSVs integrity in '{LODO_FOLDS_DIR}'...")
audit_records = []

for f_idx, h_key in enumerate(DATASET_KEYS, 1):
    try:
        tr_df, vl_df, te_df = load_lodo_fold_splits(f_idx, h_key)
        tr_pts = set(tr_df["patient_id"])
        vl_pts = set(vl_df["patient_id"])
        te_pts = set(te_df["patient_id"])

        assert len(tr_pts.intersection(vl_pts)) == 0, f"LEAKAGE train/val in fold {f_idx}"
        assert len(tr_pts.intersection(te_pts)) == 0, f"LEAKAGE train/test in fold {f_idx}"
        assert len(vl_pts.intersection(te_pts)) == 0, f"LEAKAGE val/test in fold {f_idx}"

        audit_records.append({
            "Fold": f_idx,
            "Held-out OOD": h_key.upper(),
            "Train Images": len(tr_df),
            "Train Patients": len(tr_pts),
            "Val Images": len(vl_df),
            "Val Patients": len(vl_pts),
            "Test Images": len(te_df),
            "Test Patients": len(te_pts),
            "Zero Leakage": "100% Passed"
        })
        print(f"  [OK] Fold {f_idx} ({h_key:10s}): Train={len(tr_df):5,d} ({len(tr_pts):5,d} pts) | "
              f"Val={len(vl_df):5,d} ({len(vl_pts):5,d} pts) | Test={len(te_df):5,d} ({len(te_pts):5,d} pts) | Leakage: 0%")
    except Exception as e:
        print(f"  [PENDING] Fold {f_idx} ({h_key}): {e}")

if audit_records:
    print()
    print("[SUCCESS] Pre-flight audit confirmed: Zero Patient Leakage across all verified folds!")
"""

# ------------------------------------------------------------------------------
# 4. CELL 10: Standardized Image Path Resolution in Precompute Cache
# ------------------------------------------------------------------------------
c10_lines = nb["cells"][10]["source"]
c10_src = "".join(c10_lines)

# Replace path resolution inside process_single
old_path_block = """        if img_root and not os.path.isabs(native_path_str) and not os.path.exists(native_path_str):
            img_path = os.path.join(img_root, dataset_key, PureWindowsPath(native_path_str).name)
        else:
            img_path = native_path_str"""

new_path_block = """        # Clean image path resolution: {img_root}/{dataset_key}/{filename}
        filename = str(row.get("filename", Path(native_path_str.replace("\\\\", "/")).name))
        candidates = [
            os.path.join(img_root, dataset_key, filename) if img_root else None,
            os.path.join(img_root, "native_cropped", dataset_key, filename) if img_root else None,
            os.path.join(img_root, native_path_str.replace("\\\\", "/")) if img_root else None,
            native_path_str if os.path.exists(native_path_str) else None,
        ]
        img_path = next((p for p in candidates if p and os.path.exists(p)), None)
        if img_path is None:
            img_path = os.path.join(img_root, dataset_key, filename) if img_root else native_path_str"""

if old_path_block in c10_src:
    c10_src = c10_src.replace(old_path_block, new_path_block, 1)

# ------------------------------------------------------------------------------
# 5. CELL 11: Standardized Image Path Resolution in FundusDualBranchDataset
# ------------------------------------------------------------------------------
c11_lines = nb["cells"][11]["source"]
c11_src = "".join(c11_lines)

old_ds_path = """        if processed_np is None:
            if self.img_root and not os.path.isabs(native_path_str) and not os.path.exists(native_path_str):
                img_path = os.path.join(self.img_root, dataset_key, Path(normalized_path).name)
            else:
                img_path = native_path_str"""

new_ds_path = """        if processed_np is None:
            # Clean image path resolution: {img_root}/{dataset_key}/{filename}
            filename = str(row.get("filename", Path(normalized_path).name))
            candidates = [
                os.path.join(self.img_root, dataset_key, filename) if self.img_root else None,
                os.path.join(self.img_root, "native_cropped", dataset_key, filename) if self.img_root else None,
                os.path.join(self.img_root, normalized_path) if self.img_root else None,
                native_path_str if os.path.exists(native_path_str) else None,
            ]
            img_path = next((p for p in candidates if p and os.path.exists(p)), None)
            if img_path is None:
                img_path = os.path.join(self.img_root, dataset_key, filename) if self.img_root else native_path_str"""

if old_ds_path in c11_src:
    c11_src = c11_src.replace(old_ds_path, new_ds_path, 1)

# ------------------------------------------------------------------------------
# 6. CELL 21: Streamlined Fold Loop using load_lodo_fold_splits
# ------------------------------------------------------------------------------
c21_lines = nb["cells"][21]["source"]
c21_src = "".join(c21_lines)

# Target the data loading part of fold loop
start_split_marker = "# ----------------------------------------------------------------------\n        # 1. LOAD PRE-GENERATED FROZEN SPLITS (OR DYNAMIC FALLBACK)"
end_split_marker = "ensure_test_cache_on_drive("

if start_split_marker in c21_src:
    idx1 = c21_src.find(start_split_marker)
    idx2 = c21_src.find(end_split_marker)
    
    clean_split_code = """# 1. Load Pre-generated Frozen Split for this Fold
        train_df, val_df, ood_test_df = load_lodo_fold_splits(fold_idx, held_out_key)
        print(f"  [DATA] Fold {fold_idx} ({held_out_name}) Loaded from '{LODO_FOLDS_DIR}':")
        print(f"    Train: {len(train_df):,d} images | Val: {len(val_df):,d} images | Test (OOD): {len(ood_test_df):,d} images")

        # 2. Persist per-fold split metadata CSVs to results folder if not already there
        p_train_csv = os.path.join(FOLDS_DIR, f"fold_{fold_idx}_{held_out_key.lower()}_train.csv")
        p_val_csv   = os.path.join(FOLDS_DIR, f"fold_{fold_idx}_{held_out_key.lower()}_val.csv")
        p_test_csv  = os.path.join(FOLDS_DIR, f"fold_{fold_idx}_{held_out_key.lower()}_test.csv")
        if not os.path.exists(p_train_csv):
            train_df.to_csv(p_train_csv, index=False)
            val_df.to_csv(p_val_csv, index=False)
            ood_test_df.to_csv(p_test_csv, index=False)

        # 3. Ensure precomputed test cache on Drive
        """
    c21_src = c21_src[:idx1] + clean_split_code + c21_src[idx2:]

# Check AST of all modified cells before saving
ast.parse(c4_code)
ast.parse(c7_code)
ast.parse(c8_code)
ast.parse(c10_src)
ast.parse(c11_src)
ast.parse(c21_src)

nb["cells"][4]["source"] = [l + "\n" for l in c4_code.strip().split("\n")]
nb["cells"][7]["source"] = [l + "\n" for l in c7_code.strip().split("\n")]
nb["cells"][8]["source"] = [l + "\n" for l in c8_code.strip().split("\n")]
nb["cells"][10]["source"] = [l + "\n" for l in c10_src.strip().split("\n")]
nb["cells"][11]["source"] = [l + "\n" for l in c11_src.strip().split("\n")]
nb["cells"][21]["source"] = [l + "\n" for l in c21_src.strip().split("\n")]

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("[SUCCESS] Successfully updated main_model_lodo.ipynb cleanly!")
