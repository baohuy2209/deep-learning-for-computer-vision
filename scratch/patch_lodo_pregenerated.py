import json, ast

nb_path = "notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# 1. Update Cell 04 with LODO_FOLDS_DIR
c4_lines = nb["cells"][4]["source"]

# Find where FOLDS_DIR is defined
new_c4 = []
for line in c4_lines:
    new_c4.append(line)
    if 'FOLDS_DIR = os.path.join(RESULT_ROOT, "folds")' in line:
        extra = """
# Pre-generated 6-Fold LODO Dataset Directory (from merge_dataset_lodo.ipynb)
LODO_FOLDS_DRIVE_ZIP = os.path.join(DRIVE_BASE, "lodo_folds.zip")
LODO_FOLDS_DRIVE_DIR = os.path.join(DRIVE_BASE, "lodo_folds")
LODO_FOLDS_LOCAL_DIR = "/content/lodo_folds"

if os.path.exists(LODO_FOLDS_DRIVE_ZIP):
    if not os.path.exists(LODO_FOLDS_LOCAL_DIR):
        os.makedirs(LODO_FOLDS_LOCAL_DIR, exist_ok=True)
        print(f"[UNPACK] Extracting '{LODO_FOLDS_DRIVE_ZIP}' to '{LODO_FOLDS_LOCAL_DIR}'...")
        get_ipython().system('unzip -q "$LODO_FOLDS_DRIVE_ZIP" -d "$LODO_FOLDS_LOCAL_DIR"')
    LODO_FOLDS_DIR = LODO_FOLDS_LOCAL_DIR
elif os.path.exists(LODO_FOLDS_DRIVE_DIR):
    LODO_FOLDS_DIR = LODO_FOLDS_DRIVE_DIR
elif os.path.exists("data/processed/lodo_folds"):
    LODO_FOLDS_DIR = os.path.abspath("data/processed/lodo_folds")
else:
    LODO_FOLDS_DIR = FOLDS_DIR
print(f"[LODO DATA] Frozen Splits Directory: '{LODO_FOLDS_DIR}' (Exists: {os.path.exists(LODO_FOLDS_DIR)})")
"""
        new_c4.append(extra)

nb["cells"][4]["source"] = new_c4
ast.parse(''.join(new_c4))
print("[OK] Cell 04 patched and passed AST parse!")

# 2. Update Cell 21 with pre-generated split loader
c21_src = ''.join(nb["cells"][21]["source"])

target_block = """        # 1. Isolate Held-out OOD Test Set (natural clinical prevalence)
        ood_test_df = master_df[master_df["dataset_key"] == held_out_key].copy().reset_index(drop=True)
        ood_test_df["split"] = "test"

        # 2. Pool remaining 5 datasets
        train_5_df = master_df[master_df["dataset_key"] != held_out_key].copy().reset_index(drop=True)

        # 3. Patient-level split: 85% train, 15% val
        train_raw_df, val_df = split_5_dataset_training_pool(train_5_df, train_ratio=0.85, random_state=42 + fold_idx)

        # 4. Source-aware quota balancing strictly on train
        target_sample_range = (40, 80) if SMOKE_TEST else (1500, 2500)
        train_df = create_balanced_training_pool(train_raw_df, target_range=target_sample_range, seed=42 + fold_idx)

        # 5. Persist per-fold split metadata CSVs
        p_train_csv = os.path.join(FOLDS_DIR, f"fold_{fold_idx}_{held_out_key.lower()}_train.csv")
        p_val_csv   = os.path.join(FOLDS_DIR, f"fold_{fold_idx}_{held_out_key.lower()}_val.csv")
        p_test_csv  = os.path.join(FOLDS_DIR, f"fold_{fold_idx}_{held_out_key.lower()}_test.csv")

        if not os.path.exists(p_train_csv):
            train_df.to_csv(p_train_csv, index=False)
            val_df.to_csv(p_val_csv, index=False)
            ood_test_df.to_csv(p_test_csv, index=False)"""

replacement_block = """        # ----------------------------------------------------------------------
        # 1. LOAD PRE-GENERATED FROZEN SPLITS (OR DYNAMIC FALLBACK)
        # ----------------------------------------------------------------------
        # Check candidate locations (supporting both 0-indexed and 1-indexed patterns)
        slug = held_out_key.lower()
        candidates = [
            os.path.join(LODO_FOLDS_DIR, f"fold_{fold_idx - 1}_{slug}_train.csv"),
            os.path.join(LODO_FOLDS_DIR, f"fold_{fold_idx}_{slug}_train.csv"),
            os.path.join(FOLDS_DIR, f"fold_{fold_idx}_{slug}_train.csv"),
        ]
        matched_train = next((p for p in candidates if os.path.exists(p)), None)

        if matched_train is not None:
            matched_val = matched_train.replace("_train.csv", "_val.csv")
            matched_test = matched_train.replace("_train.csv", "_test.csv")
            train_df = pd.read_csv(matched_train)
            val_df = pd.read_csv(matched_val)
            ood_test_df = pd.read_csv(matched_test)
            print(f"  [FROZEN DATA] Loaded pre-generated split from '{os.path.basename(matched_train)}'")
            print(f"    Train: {len(train_df):,d} images | Val: {len(val_df):,d} images | Test (OOD): {len(ood_test_df):,d} images")
        else:
            print(f"  [DYNAMIC DATA] Pre-generated CSV not found in '{LODO_FOLDS_DIR}'. Generating on-the-fly...")
            ood_test_df = master_df[master_df["dataset_key"] == held_out_key].copy().reset_index(drop=True)
            ood_test_df["split"] = "test"
            train_5_df = master_df[master_df["dataset_key"] != held_out_key].copy().reset_index(drop=True)
            train_raw_df, val_df = split_5_dataset_training_pool(train_5_df, train_ratio=0.85, random_state=42 + fold_idx)
            target_sample_range = (40, 80) if SMOKE_TEST else (1500, 2500)
            train_df = create_balanced_training_pool(train_raw_df, target_range=target_sample_range, seed=42 + fold_idx)

            p_train_csv = os.path.join(FOLDS_DIR, f"fold_{fold_idx}_{slug}_train.csv")
            p_val_csv   = os.path.join(FOLDS_DIR, f"fold_{fold_idx}_{slug}_val.csv")
            p_test_csv  = os.path.join(FOLDS_DIR, f"fold_{fold_idx}_{slug}_test.csv")
            train_df.to_csv(p_train_csv, index=False)
            val_df.to_csv(p_val_csv, index=False)
            ood_test_df.to_csv(p_test_csv, index=False)"""

assert target_block in c21_src, "target_block not found in Cell 21"
c21_src = c21_src.replace(target_block, replacement_block, 1)
ast.parse(c21_src)
nb["cells"][21]["source"] = [line + "\n" for line in c21_src.split("\n")]
print("[OK] Cell 21 patched and passed AST parse!")

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"[SUCCESS] Updated {nb_path} with pre-generated split loader!")
