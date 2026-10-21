import json

targets = [
    "notebooks/diabetic_retinopathy/pipeline/model/main_model (4).ipynb",
    "notebooks/diabetic_retinopathy/pipeline/model/main_model (8).ipynb",
    "notebooks/diabetic_retinopathy/pipeline/model/main_model.ipynb"
]

new_cell_25_source = """# ==============================================================================
# 2.5 DATALOADER INITIALIZATION & PRE-WARMING CACHE ROUTINE
# ==============================================================================
from functools import partial

# ------------------------------------------------------------------------------
# 1. INSTANTIATE DUAL-BRANCH DATASETS (Train / Val / Test)
# ------------------------------------------------------------------------------
# Training Dataset: enables spatial/photometric data augmentations on Global stream
train_dataset = FundusDualBranchDataset(
    csv_path=csv_train_path,
    img_root=image_root_dir,
    img_size=IMG_SIZE,
    split="train",
    ablation_mode="ben_graham_green",
    cache_dir=CACHE_DIR
)

# In-Domain Validation Dataset: deterministic resize without random augmentations
val_dataset = FundusDualBranchDataset(
    csv_path=csv_val_path,
    img_root=image_root_dir,
    img_size=IMG_SIZE,
    split="val",
    ablation_mode="ben_graham_green",
    cache_dir=CACHE_DIR
)

# Held-out In-Distribution Test Dataset (15% split): deterministic evaluation
test_dataset = FundusDualBranchDataset(
    csv_path=csv_test_path,
    img_root=image_root_dir,
    img_size=IMG_SIZE,
    split="test",
    ablation_mode="ben_graham_green",
    cache_dir=CACHE_DIR
)

# ------------------------------------------------------------------------------
# 2. CONFIGURE COLLATE FUNCTIONS (HD_3 & HD_4: Distinct Train vs Eval Data Loading)
# ------------------------------------------------------------------------------
# Train mode: Randomly samples M in [32, 64] patches per image for stochastic regularization
train_collate_fn = partial(trainer_compatible_collate_fn, mode="train", num_sample=(32, 64))

# Eval & Test mode: Deterministic dense patch sweep (up to max_patches_eval=256) + boolean patch_mask
eval_collate_fn = partial(trainer_compatible_collate_fn, mode="eval")

# ------------------------------------------------------------------------------
# 3. OPTIONAL PRE-WARMING CACHE ENGINE (Precomputes all .npz files before training)
# ------------------------------------------------------------------------------
def warm_cache_dual_branch(dataset: FundusDualBranchDataset, num_workers: int = num_workers, batch_size: int = 16):
    \"\"\"
    Iterates through the dataset once to precompute and save all .npz archives 
    (FOV masks, patch anchors, preprocessed images) on disk before launching training.
    \"\"\"
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=False,
        collate_fn=lambda b: [item.get("native_path", "") for item in b]
    )
    for _ in tqdm(loader, desc=f"Pre-caching [{dataset.split}]"):
        pass

# ------------------------------------------------------------------------------
# 4. INITIALIZE PYTORCH DATALOADERS
# ------------------------------------------------------------------------------
eval_batch_size = 1  # HD_3 Page 4: batch_size=1 prevents OOM on dense patch grids and avoids padding variance

# Training DataLoader: batch shuffling & parallel workers enabled
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=use_pin_memory,
    collate_fn=train_collate_fn,
    persistent_workers=(num_workers > 0),
    drop_last=True
)

# Validation DataLoader: deterministic sequential evaluation (eval_collate_fn)
val_loader = DataLoader(
    val_dataset,
    batch_size=eval_batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=use_pin_memory,
    collate_fn=eval_collate_fn,
    persistent_workers=(num_workers > 0)
)

# Held-out Test DataLoader: deterministic sequential evaluation (eval_collate_fn)
test_loader = DataLoader(
    test_dataset,
    batch_size=eval_batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=use_pin_memory,
    collate_fn=eval_collate_fn,
    persistent_workers=(num_workers > 0)
)

print(f"[INFO] DataLoaders Initialized Successfully:")
print(f"  - Train: batch_size={BATCH_SIZE}, mode='train' (M=32 - 64 stochastic sub-sampling, shuffle=True)")
print(f"  - Val:   batch_size={eval_batch_size}, mode='eval' (Deterministic full dense sweep + Masked Pooling)")
print(f"  - Test:  batch_size={eval_batch_size}, mode='eval' (Held-out In-Distribution Test, dense sweep)")
"""

lines_25 = [l + "\n" for l in new_cell_25_source.split("\n")]
lines_25[-1] = lines_25[-1].rstrip("\n")

for path in targets:
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # In cell 23, add aliases at end if not already present
    src23 = "".join(nb["cells"][23]["source"])
    if "train_collate_fn = partial(" not in src23:
        nb["cells"][23]["source"].append("\n# Collate function aliases for pipeline convenience\n")
        nb["cells"][23]["source"].append("from functools import partial\n")
        nb["cells"][23]["source"].append("train_collate_fn = partial(trainer_compatible_collate_fn, mode=\"train\", num_sample=(32, 64))\n")
        nb["cells"][23]["source"].append("eval_collate_fn = partial(trainer_compatible_collate_fn, mode=\"eval\")\n")

    # Update cell 25
    nb["cells"][25]["source"] = lines_25

    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"Successfully updated {path}!")
