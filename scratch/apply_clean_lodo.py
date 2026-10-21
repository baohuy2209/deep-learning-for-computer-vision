import json, ast, os

nb_path = "notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# 1. Verify and delete Cells 5, 6, 7, 8
del nb["cells"][5:9]
print(f"Deleted cells 5..8. Remaining cells: {len(nb['cells'])}")

# Update Markdown Section Titles
nb["cells"][5]["source"] = ["## 2. Multi-Scale Retinal Data Pipeline (Matching main_model (6).ipynb)\n"]
nb["cells"][7]["source"] = ["## 3. Deep Neural Network Architecture & Ordinal Heads\n"]
nb["cells"][12]["source"] = ["## 4. Clinical Metrics & PyTorch Training Engine\n"]
nb["cells"][15]["source"] = ["## 5. 6-Fold Leave-One-Dataset-Out (LODO) Orchestration Loop\n"]
nb["cells"][17]["source"] = ["## 6. Publication-Quality LODO Visualization Suite\n"]

# 2. In Cell 06 (previously 10): Ensure PureWindowsPath image_path resolution in precompute_dataset_cache
c6_src = "".join(nb["cells"][6]["source"])
old_c6_path = """        # Clean image path resolution: {img_root}/{dataset_key}/{filename}
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

new_c6_path = """        # Resolve image path (exact logic from main_model (6).ipynb)
        if img_root and not os.path.isabs(native_path_str) and not os.path.exists(native_path_str):
            img_path = f"{img_root}/{dataset_key}/{PureWindowsPath(native_path_str).name}"
        else:
            img_path = native_path_str"""

if old_c6_path in c6_src:
    c6_src = c6_src.replace(old_c6_path, new_c6_path, 1)
nb["cells"][6]["source"] = [l + "\n" for l in c6_src.split("\n")]

# 3. In Cell 07 (previously 11): FundusDualBranchDataset path resolution
c7_src = "".join(nb["cells"][7]["source"])
old_c7_path = """        if processed_np is None:
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

new_c7_path = """        if processed_np is None:
            # 1. Resolve image path (exact logic from main_model (6).ipynb)
            if self.img_root and not os.path.isabs(native_path_str) and not os.path.exists(native_path_str):
                img_path = f"{self.img_root}/{dataset_key}/{PureWindowsPath(native_path_str).name}"
            else:
                img_path = native_path_str"""

if old_c7_path in c7_src:
    c7_src = c7_src.replace(old_c7_path, new_c7_path, 1)
nb["cells"][7]["source"] = [l + "\n" for l in c7_src.split("\n")]

# 4. In Cell 16 (previously 21): Update fold loop
c16_src = "".join(nb["cells"][16]["source"])

# Replace inner fold data loading block cleanly
start_marker = "# 1. Load Pre-generated Frozen Split for this Fold"
end_marker = "# 3. Ensure precomputed test cache on Drive"

idx1 = c16_src.find(start_marker)
idx2 = c16_src.find(end_marker)

if idx1 != -1 and idx2 != -1:
    clean_loader = """# 1. Directly read pre-generated LODO fold CSVs (matching main_model (6).ipynb)
        train_csv = os.path.join(LODO_FOLDS_DIR, f"fold_{fold_idx}_{held_out_key}_train.csv")
        val_csv   = os.path.join(LODO_FOLDS_DIR, f"fold_{fold_idx}_{held_out_key}_val.csv")
        test_csv  = os.path.join(LODO_FOLDS_DIR, f"fold_{fold_idx}_{held_out_key}_test.csv")

        train_df = pd.read_csv(train_csv)
        val_df   = pd.read_csv(val_csv)
        ood_test_df = pd.read_csv(test_csv)
        print(f"  [DATA] Fold {fold_idx + 1} ({held_out_name}) Loaded from '{LODO_FOLDS_DIR}':")
        print(f"    Train: {len(train_df):,d} images | Val: {len(val_df):,d} images | Test (OOD): {len(ood_test_df):,d} images")

        """
    c16_src = c16_src[:idx1] + clean_loader + c16_src[idx2:]

# Ensure fold_idx iteration is 0-indexed in Cell 16 loop:
c16_src = c16_src.replace("for fold_idx, held_out_key in enumerate(target_folds, 1):", "for fold_idx, held_out_key in enumerate(target_folds):")
c16_src = c16_src.replace("FOLD {fold_idx}/{len(target_folds)}", "FOLD {fold_idx + 1}/{len(target_folds)}")
c16_src = c16_src.replace("Fold {fold_idx} ({held_out_name})", "Fold {fold_idx + 1} ({held_out_name})")

nb["cells"][16]["source"] = [l + "\n" for l in c16_src.split("\n")]

# Verify AST of all code cells
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code":
        src = "".join(c["source"])
        ast.parse(src)
        print(f"Cell {i:02d}: AST OK")

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"[SUCCESS] main_model_lodo.ipynb cleanly updated and verified ({len(nb['cells'])} total cells)!")
