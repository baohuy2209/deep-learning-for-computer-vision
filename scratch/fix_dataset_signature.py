import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

target_files = [
    "notebooks/diabetic_retinopathy/pipeline/model/main_model (4).ipynb",
    "notebooks/diabetic_retinopathy/pipeline/model/main_model (8).ipynb",
    "notebooks/diabetic_retinopathy/pipeline/model/main_model.ipynb"
]

for tf in target_files:
    print(f"\n--- Patching {tf} ---")
    with open(tf, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # 1. Update Cell 20 (FundusDualBranchDataset.__init__)
    c20_src = "".join(nb["cells"][20]["source"])
    
    # Check if we need to add aliases
    old_init = """    def __init__(
        self,
        csv_path: Union[str, pd.DataFrame],
        img_size: int = IMG_SIZE,
        img_root: Optional[str] = None,
        split: str = "train",
        ablation_mode: str = "ben_graham_green",
        cache_dir: Optional[str] = None,
        patch_size: int = 128,
        stride: int = 96
    ):"""

    new_init = """    def __init__(
        self,
        csv_path: Optional[Union[str, pd.DataFrame]] = None,
        img_size: int = IMG_SIZE,
        img_root: Optional[str] = None,
        split: str = "train",
        ablation_mode: str = "ben_graham_green",
        cache_dir: Optional[str] = None,
        patch_size: int = 128,
        stride: int = 96,
        transform: Optional[Any] = None,
        metadata: Optional[Union[str, pd.DataFrame]] = None,
        image_root_dir: Optional[str] = None
    ):"""

    if old_init in c20_src:
        c20_src = c20_src.replace(old_init, new_init)
        fallback_code = """        if csv_path is None:
            if metadata is not None:
                csv_path = metadata
            else:
                raise ValueError("FundusDualBranchDataset requires 'csv_path' or 'metadata'.")
        if img_root is None and image_root_dir is not None:
            img_root = image_root_dir

        if isinstance(csv_path, pd.DataFrame):"""
        c20_src = c20_src.replace("        if isinstance(csv_path, pd.DataFrame):", fallback_code)
        nb["cells"][20]["source"] = [line + "\n" for line in c20_src.splitlines()]
        print("  -> Cell 20 updated with backward-compatible aliases (metadata & image_root_dir).")
    else:
        print("  -> Cell 20 pattern check: already updated or slightly different.")

    # 2. Update Cell 25 (Instantiate with canonical parameters)
    c25_src = "".join(nb["cells"][25]["source"])
    old_c25_init = """# Instantiate Dual-Branch Datasets
train_dataset = FundusDualBranchDataset(
    metadata=csv_train_path,
    image_root_dir=image_root_dir,
    img_size=IMG_SIZE,
    split="train",
    cache_dir=CACHE_DIR
)

val_dataset = FundusDualBranchDataset(
    metadata=csv_val_path,
    image_root_dir=image_root_dir,
    img_size=IMG_SIZE,
    split="val",
    cache_dir=CACHE_DIR
)

test_dataset = FundusDualBranchDataset(
    metadata=csv_test_path,
    image_root_dir=image_root_dir,
    img_size=IMG_SIZE,
    split="test",
    cache_dir=CACHE_DIR
)"""

    new_c25_init = """# Instantiate Dual-Branch Datasets
train_dataset = FundusDualBranchDataset(
    csv_path=csv_train_path,
    img_root=image_root_dir,
    img_size=IMG_SIZE,
    split="train",
    cache_dir=CACHE_DIR
)

val_dataset = FundusDualBranchDataset(
    csv_path=csv_val_path,
    img_root=image_root_dir,
    img_size=IMG_SIZE,
    split="val",
    cache_dir=CACHE_DIR
)

test_dataset = FundusDualBranchDataset(
    csv_path=csv_test_path,
    img_root=image_root_dir,
    img_size=IMG_SIZE,
    split="test",
    cache_dir=CACHE_DIR
)"""

    if old_c25_init in c25_src:
        c25_src = c25_src.replace(old_c25_init, new_c25_init)
        nb["cells"][25]["source"] = [line + "\n" for line in c25_src.splitlines()]
        print("  -> Cell 25 updated to use canonical 'csv_path' and 'img_root'.")
    else:
        print("  -> Cell 25 pattern check: already updated or slightly different.")

    with open(tf, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"  -> Successfully saved {tf}.")
