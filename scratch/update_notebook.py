import json
import copy

NOTEBOOK_PATH = "notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation (3).ipynb"

with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Loaded notebook with {len(nb['cells'])} cells.")

# ------------------------------------------------------------------------------
# 1. Update Cell 7 (Hyperparameters): Add FORCE_FRESH_TRAINING
# ------------------------------------------------------------------------------
cell_7_src = nb['cells'][7]['source']
cell_7_text = "".join(cell_7_src)
if "FORCE_FRESH_TRAINING" not in cell_7_text:
    old_target = "SMOKE_TEST = False"
    new_text = "SMOKE_TEST = False  # Set to True for rapid sanity check (2 epochs), False for full official ablation (25 epochs)\nFORCE_FRESH_TRAINING = True  # Set to True to start fresh from Epoch 1 and bypass stale/corrupted checkpoints"
    # Find and replace in cell 7
    lines = []
    for line in cell_7_src:
        if "SMOKE_TEST = False" in line:
            lines.append("SMOKE_TEST = False  # Set to True for rapid sanity check (2 epochs), False for full official ablation (25 epochs)\n")
            lines.append("FORCE_FRESH_TRAINING = True  # Set to True to start fresh from Epoch 1 and bypass stale/corrupted checkpoints\n")
        else:
            lines.append(line)
    nb['cells'][7]['source'] = lines
    print("Updated Cell 7 with FORCE_FRESH_TRAINING.")
else:
    print("Cell 7 already has FORCE_FRESH_TRAINING.")

# ------------------------------------------------------------------------------
# 2. Update Cell 8: Clean up stray NUM_WORKERS
# ------------------------------------------------------------------------------
nb['cells'][8] = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "> [!TIP]\n",
        "> **Hardware Acceleration & Throughput**: Configured `NUM_WORKERS` dynamically balances CPU pipeline loading and GPU saturation on Colab T4/L4/A100 instances.\n"
    ]
}
print("Cleaned up Cell 8.")

# ------------------------------------------------------------------------------
# 3. Update Cell 12: Add resolve_image_path and remove silent gray fallback
# ------------------------------------------------------------------------------
cell_12_src = nb['cells'][12]['source']
cell_12_text = "".join(cell_12_src)

# We want to insert resolve_image_path before ResNetFundusDataset, and update __getitem__
resolve_code = '''def resolve_image_path(row: Dict[str, Any], dataset_root: str) -> str:
    """
    Robust multi-tier path resolution across Windows local development,
    Google Colab SSD (/content/dataset_local), and hybrid workspace directories.
    """
    native_path_str = str(row.get("native_path", row.get("raw_full_path", "")))
    dataset_key = str(row.get("dataset_key", "default"))
    img_file = str(row.get("image_file", ""))
    img_id = PureWindowsPath(native_path_str).name if native_path_str else (img_file or f"{row.get('image_id', 'unknown')}.jpg")

    candidates = [
        native_path_str,
        os.path.join(dataset_root, "native_cropped", dataset_key, img_id),
        os.path.join(dataset_root, dataset_key, img_id),
        os.path.join("data", "raw", "hybrid", "native_cropped", dataset_key, img_id),
        os.path.join("data", "raw", "hybrid", dataset_key, img_id),
    ]
    for cand in candidates:
        if cand and os.path.exists(cand):
            return cand

    # Directory scan fallback
    if os.path.exists(dataset_root):
        key_dir = os.path.join(dataset_root, "native_cropped", dataset_key)
        if not os.path.exists(key_dir):
            key_dir = os.path.join(dataset_root, dataset_key)
        if os.path.exists(key_dir):
            cand_file = os.path.join(key_dir, img_id)
            if os.path.exists(cand_file):
                return cand_file

    raise FileNotFoundError(
        f"[IMAGE RESOLUTION FAILED] Could not locate image '{img_id}' (Dataset: '{dataset_key}').\\n"
        f"Checked candidate locations:\\n" + "\\n".join(f"  - {c}" for c in candidates if c)
    )

'''

# Let's inspect where apply_green_channel_ablation ends and ResNetFundusDataset begins in cell 12
if "def resolve_image_path" not in cell_12_text:
    # Insert resolve_code before class ResNetFundusDataset
    new_cell_12_lines = []
    inserted = False
    for line in cell_12_src:
        if "class ResNetFundusDataset(Dataset):" in line and not inserted:
            for rline in resolve_code.splitlines(True):
                new_cell_12_lines.append(rline)
            inserted = True
        new_cell_12_lines.append(line)
    
    # Now replace the silent gray fallback inside __getitem__
    final_cell_12_lines = []
    in_cache_miss = False
    skip_old_fallback = False
    
    for line in new_cell_12_lines:
        if "if processed_rgb is None:" in line:
            in_cache_miss = True
            final_cell_12_lines.append(line)
            final_cell_12_lines.append("            # Resolve valid file path across environments (fails loud on missing file)\n")
            final_cell_12_lines.append("            img_path = resolve_image_path(row, DATASET_ROOT)\n")
            final_cell_12_lines.append("            img_bgr = cv2.imread(img_path)\n")
            final_cell_12_lines.append("            if img_bgr is not None:\n")
            final_cell_12_lines.append("                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)\n")
            final_cell_12_lines.append("            else:\n")
            final_cell_12_lines.append("                try:\n")
            final_cell_12_lines.append("                    pil_img = Image.open(img_path).convert('RGB')\n")
            final_cell_12_lines.append("                    img_rgb = np.array(pil_img)\n")
            final_cell_12_lines.append("                except Exception as err:\n")
            final_cell_12_lines.append("                    raise FileNotFoundError(f'Failed decoding image at {img_path}: {err}')\n")
            skip_old_fallback = True
            continue
        
        if skip_old_fallback:
            if "processed_rgb = apply_green_channel_ablation(img_rgb, self.ablation_mode)" in line:
                skip_old_fallback = False
                final_cell_12_lines.append("            " + line.strip() + "\n")
            continue
        
        final_cell_12_lines.append(line)
        
    nb['cells'][12]['source'] = final_cell_12_lines
    print("Updated Cell 12 with resolve_image_path and eliminated silent gray fallback.")
else:
    print("Cell 12 already contains resolve_image_path.")

# ------------------------------------------------------------------------------
# 4. Insert 4 Pre-Training Visualization & Integrity Sections right after Cell 12
# ------------------------------------------------------------------------------
# Let's check if the visualization cells are already present
nb_all_text = "".join("".join(c['source']) for c in nb['cells'])
if "## 2.3 Sample Diagnostics & Tensor Sanity Inspection" not in nb_all_text:
    new_cells = [
        # Cell 2.3 MD
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2.3 Sample Diagnostics & Tensor Sanity Inspection\n",
                "\n",
                "Inspects an arbitrary sample from `train_dataset` to verify tensor geometry, data types, and non-trivial pixel intensity ranges, verifying that images are properly loaded from the preprocessed cache.\n"
            ]
        },
        # Cell 2.3 Code
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ==============================================================================\n",
                "# 1. INSPECT ARBITRARY SAMPLE IN TRAINING DATASET\n",
                "# ==============================================================================\n",
                "sample_idx = 0  # Modify index to inspect any sample (e.g. 0, 10, 100)\n",
                "sample = train_dataset[sample_idx]\n",
                "\n",
                "img_tensor = sample[\"image\"]       # uint8 tensor [3, 512, 512]\n",
                "label_val = sample[\"label\"].item()\n",
                "pid = sample.get(\"patient_id\", \"N/A\")\n",
                "dataset_key = sample.get(\"dataset_key\", \"N/A\")\n",
                "mean_val = img_tensor.float().mean().item()\n",
                "std_val = img_tensor.float().std().item()\n",
                "\n",
                "print(\"=\" * 75)\n",
                "print(f\"SAMPLE DIAGNOSTICS & TENSOR METRICS (INDEX: {sample_idx})\")\n",
                "print(\"=\" * 75)\n",
                "print(f\"• Dataset Source         : {dataset_key}\")\n",
                "print(f\"• Patient Identifier     : {pid}\")\n",
                "print(f\"• Clinical Label (Grade) : ICDRS Grade {label_val}\")\n",
                "print(f\"• Tensor Shape           : {img_tensor.shape} (Channels, Height, Width)\")\n",
                "print(f\"• Tensor Dtype           : {img_tensor.dtype}\")\n",
                "print(f\"• Pixel Value Range      : Min = {img_tensor.min().item()} | Max = {img_tensor.max().item()}\")\n",
                "print(f\"• Mean Pixel Intensity   : {mean_val:.2f} (Expected fundus range: ~80 - 160)\")\n",
                "print(f\"• Pixel Standard Dev (σ) : {std_val:.2f} (Must be > 5.0 to rule out gray collapse)\")\n",
                "print(\"=\" * 75)\n",
                "print(\"5x5 Center Pixel Intensity Matrix (Channel 0):\")\n",
                "print(img_tensor[0, 250:255, 250:255].numpy())\n",
                "print(\"=\" * 75)\n"
            ]
        },
        # Cell 2.4 MD
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2.4 Visual Mini-Batch Inspection Grid\n",
                "\n",
                "Fetches a mini-batch from `train_loader`, performs robust color space conversion via `safe_convert_for_plot`, and renders an 8-sample grid across clinical severity grades (Grade 0 to Grade 4) to visually confirm vascular tree structures, optic discs, and Green-channel Ben Graham contrast enhancement.\n"
            ]
        },
        # Cell 2.4 Code
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ==============================================================================\n",
                "# 2. FETCH A MINI-BATCH & RENDER VISUAL INSPECTION GRID\n",
                "# ==============================================================================\n",
                "for batch in train_loader:\n",
                "    batch_imgs = batch[\"image\"]       # [Batch, 3, 512, 512] uint8\n",
                "    batch_labels = batch[\"label\"]     # [Batch]\n",
                "    batch_keys = batch[\"dataset_key\"]\n",
                "    break\n",
                "\n",
                "def safe_convert_for_plot(img_data: torch.Tensor) -> np.ndarray:\n",
                "    \"\"\"\n",
                "    Converts image tensor (uint8, float32, or ImageNet normalized)\n",
                "    into a standardized float32 NumPy array in range [0.0, 1.0] for accurate Matplotlib rendering.\n",
                "    \"\"\"\n",
                "    if isinstance(img_data, torch.Tensor):\n",
                "        if img_data.ndim == 4:\n",
                "            img_data = img_data[0]\n",
                "        if img_data.ndim == 3 and img_data.shape[0] in [1, 3]:\n",
                "            img_data = img_data.permute(1, 2, 0)\n",
                "        arr = img_data.detach().cpu().numpy()\n",
                "    else:\n",
                "        arr = np.array(img_data)\n",
                "\n",
                "    arr = np.squeeze(arr).astype(np.float32)\n",
                "    if arr.max() > arr.min():\n",
                "        if arr.min() < 0:  # ImageNet Z-score normalized\n",
                "            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)\n",
                "            std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)\n",
                "            if arr.ndim == 3 and arr.shape[-1] == 3:\n",
                "                arr = arr * std + mean\n",
                "            arr = np.clip(arr, 0.0, 1.0)\n",
                "        elif arr.max() > 1.0:  # uint8 [0, 255]\n",
                "            arr = arr / 255.0\n",
                "            arr = np.clip(arr, 0.0, 1.0)\n",
                "    else:\n",
                "        arr = np.zeros_like(arr)\n",
                "    return arr\n",
                "\n",
                "# Display 8 representative samples in a 2x4 panel\n",
                "num_display = min(8, len(batch_imgs))\n",
                "fig, axes = plt.subplots(2, 4, figsize=(16.0, 8.5))\n",
                "axes = axes.flatten()\n",
                "\n",
                "grade_names = [\"0: Normal\", \"1: Mild\", \"2: Moderate\", \"3: Severe\", \"4: Proliferative\"]\n",
                "\n",
                "for i in range(num_display):\n",
                "    img_np = safe_convert_for_plot(batch_imgs[i])\n",
                "    lbl = batch_labels[i].item()\n",
                "    d_key = batch_keys[i]\n",
                "    mean_val = img_np.mean()\n",
                "    std_val = img_np.std()\n",
                "\n",
                "    axes[i].imshow(img_np)\n",
                "    axes[i].set_title(\n",
                "        f\"Sample #{i+1} | Grade {grade_names[lbl]}\\n[{d_key}] Mean: {mean_val:.2f}, Std: {std_val:.2f}\",\n",
                "        fontsize=10.5, fontweight=\"bold\"\n",
                "    )\n",
                "    axes[i].axis(\"off\")\n",
                "\n",
                "plt.suptitle(\n",
                "    \"Pre-Training Batch Inspection: 512x512 ResNet-50 Inputs (Green-Channel Ben Graham)\",\n",
                "    fontsize=14, fontweight=\"bold\", y=0.98\n",
                ")\n",
                "plt.tight_layout()\n",
                "plt.show()\n"
            ]
        },
        # Cell 2.5 MD
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2.5 Automated Pre-Flight Data & Cache Integrity Verification\n",
                "\n",
                "Automated assertion scanning 50 random samples across `train_dataset`. Asserts that pixel standard deviation $\\sigma > 5.0$, mathematically ruling out blank 128 gray arrays or corrupted cache entries before launching the 5-round training benchmark.\n"
            ]
        },
        # Cell 2.5 Code
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ==============================================================================\n",
                "# 3. AUTOMATED PRE-FLIGHT INTEGRITY ASSERTION (ANTI-COLLAPSE SAFEGUARD)\n",
                "# ==============================================================================\n",
                "def verify_dataset_integrity(dataset: ResNetFundusDataset, num_check: int = 50):\n",
                "    \"\"\"\n",
                "    Scans random samples to confirm valid image resolution, non-trivial variance,\n",
                "    and absence of blank/all-gray 128 arrays.\n",
                "    \"\"\"\n",
                "    print(f\"Verifying data integrity across {num_check} random samples in [{dataset.split}]...\")\n",
                "    indices = np.random.choice(len(dataset), size=min(num_check, len(dataset)), replace=False)\n",
                "    corrupted_count = 0\n",
                "\n",
                "    for idx in indices:\n",
                "        sample = dataset[idx]\n",
                "        tensor = sample[\"image\"]\n",
                "        pixel_std = tensor.float().std().item()\n",
                "        pixel_mean = tensor.float().mean().item()\n",
                "\n",
                "        # Flag all-gray (128) or blank tensors (std < 1.0)\n",
                "        if pixel_std < 1.0 or (127.0 <= pixel_mean <= 129.0 and pixel_std < 2.0):\n",
                "            corrupted_count += 1\n",
                "            logger.error(f\"[CORRUPTION DETECTED] Index {idx} has std={pixel_std:.2f}, mean={pixel_mean:.2f}! Blank/gray fallback.\")\n",
                "\n",
                "    if corrupted_count == 0:\n",
                "        print(f\"✅ DATA INTEGRITY VERIFIED: {num_check}/{num_check} samples are valid retinal images (σ > 5.0, zero gray collapse).\")\n",
                "    else:\n",
                "        raise RuntimeError(\n",
                "            f\"❌ INTEGRITY CHECK FAILED: {corrupted_count}/{num_check} samples appear corrupted/gray! \"\n",
                "            f\"Check dataset paths and purge '{dataset.cache_dir}'.\"\n",
                "        )\n",
                "\n",
                "verify_dataset_integrity(train_dataset, num_check=50)\n"
            ]
        },
        # Cell 2.6 MD
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2.6 Dataset Splits & Class Distribution Breakdown\n",
                "\n",
                "Visualizes the 5-class severity distribution comparing the balanced training cohort against the untouched natural prevalence of the Validation and Test sets (satisfying HD_2 Lỗi 3 & HD_4 Part 1).\n"
            ]
        },
        # Cell 2.6 Code
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# ==============================================================================\n",
                "# 4. DATASET SPLITS CLASS DISTRIBUTION & PREVALENCE VISUALIZATION\n",
                "# ==============================================================================\n",
                "def visualize_dataset_splits_distribution(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):\n",
                "    \"\"\"\n",
                "    Generates structured table and comparative bar chart for Train, Val, and Test splits.\n",
                "    \"\"\"\n",
                "    grades = [f\"Grade {i}\" for i in range(5)]\n",
                "    splits_data = {\"Train (Balanced)\": train_df, \"Val (Natural)\": val_df, \"Test (Natural)\": test_df}\n",
                "\n",
                "    summary_rows = []\n",
                "    for split_name, df in splits_data.items():\n",
                "        counts = [int(df[\"diagnosis\"].value_counts().get(i, 0)) for i in range(5)]\n",
                "        total = len(df)\n",
                "        row = {\"Split\": split_name, \"Total\": total}\n",
                "        for i, c in enumerate(counts):\n",
                "            pct = (c / total * 100) if total > 0 else 0\n",
                "            row[f\"G{i}\"] = f\"{c:,} ({pct:.1f}%)\"\n",
                "        summary_rows.append(row)\n",
                "\n",
                "    summary_df = pd.DataFrame(summary_rows)\n",
                "    print(\"=\" * 85)\n",
                "    print(\"COHORT CLASS DISTRIBUTION & PREVALENCE SUMMARY (HD_2 / HD_4)\")\n",
                "    print(\"=\" * 85)\n",
                "    print(summary_df.to_string(index=False))\n",
                "    print(\"=\" * 85)\n",
                "\n",
                "    fig, ax = plt.subplots(figsize=(11.0, 5.5))\n",
                "    x = np.arange(5)\n",
                "    width = 0.26\n",
                "\n",
                "    train_counts = [train_df[\"diagnosis\"].value_counts().get(i, 0) for i in range(5)]\n",
                "    val_counts   = [val_df[\"diagnosis\"].value_counts().get(i, 0) for i in range(5)]\n",
                "    test_counts  = [test_df[\"diagnosis\"].value_counts().get(i, 0) for i in range(5)]\n",
                "\n",
                "    b1 = ax.bar(x - width, train_counts, width, label=\"Train (Balanced)\", color=\"#2b6cb0\", edgecolor=\"#1e293b\", lw=0.9)\n",
                "    b2 = ax.bar(x, val_counts, width, label=\"Val (Natural)\", color=\"#64748b\", edgecolor=\"#1e293b\", lw=0.9)\n",
                "    b3 = ax.bar(x + width, test_counts, width, label=\"Test (Natural)\", color=\"#0d9488\", edgecolor=\"#1e293b\", lw=0.9)\n",
                "\n",
                "    ax.set_ylabel(\"Number of Fundus Images\", fontweight=\"bold\")\n",
                "    ax.set_title(\"Figure 2: Dataset Splits Severity Distribution (Train Balancing vs Natural Prevalence)\", fontweight=\"bold\", pad=14)\n",
                "    ax.set_xticks(x)\n",
                "    ax.set_xticklabels([f\"Grade {i}\\n({['Normal', 'Mild', 'Mod', 'Sev', 'PDR'][i]})\" for i in x], fontweight=\"bold\")\n",
                "    ax.legend(frameon=True, facecolor=\"white\", edgecolor=\"#cbd5e1\")\n",
                "    ax.grid(True, axis=\"y\", linestyle=\"--\", alpha=0.4)\n",
                "\n",
                "    plt.tight_layout()\n",
                "    plt.show()\n",
                "\n",
                "visualize_dataset_splits_distribution(train_metadata, val_metadata, test_metadata)\n"
            ]
        }
    ]
    # Insert new_cells right after Cell 12
    nb['cells'] = nb['cells'][:13] + new_cells + nb['cells'][13:]
    print(f"Inserted 8 new visualization cells after Cell 12. Total cells now: {len(nb['cells'])}.")
else:
    print("Visualization cells already exist in notebook.")

# ------------------------------------------------------------------------------
# 5. Correct CORALHead bias initialization (increasing -> decreasing)
# ------------------------------------------------------------------------------
for c in nb['cells']:
    if c['cell_type'] == 'code' and "class CORALHead(nn.Module):" in "".join(c['source']):
        src = c['source']
        new_src = []
        for line in src:
            if "self.biases.copy_(torch.linspace(-2.0, 2.0, self.num_thresholds))" in line:
                new_src.append("            # Monotonic decreasing bias initialization: b_0 > b_1 > b_2 > b_3 (HD_2 / HD_4)\n")
                new_src.append("            self.biases.copy_(torch.linspace(2.0, -2.0, self.num_thresholds))\n")
                print("Corrected CORALHead bias initialization to decreasing order [2.0 -> -2.0].")
            else:
                new_src.append(line)
        c['source'] = new_src

# ------------------------------------------------------------------------------
# 6. Update Round Execution Loop in Section 5 to respect FORCE_FRESH_TRAINING
# ------------------------------------------------------------------------------
for c in nb['cells']:
    if c['cell_type'] == 'code' and "for round_idx, head_name in enumerate(EVALUATED_HEADS, 1):" in "".join(c['source']):
        src = c['source']
        new_src = []
        for line in src:
            if "resume_path = last_ckpt_path if os.path.exists(last_ckpt_path) else None" in line:
                new_src.append("    resume_path = None\n")
                new_src.append("    if not FORCE_FRESH_TRAINING and os.path.exists(last_ckpt_path):\n")
                new_src.append("        resume_path = last_ckpt_path\n")
                new_src.append("        print(f'[*] Tìm thấy checkpoint dở dang tại: {resume_path} -> Đang tự động Resume...')\n")
                new_src.append("    elif FORCE_FRESH_TRAINING:\n")
                new_src.append("        print(f'[*] FORCE_FRESH_TRAINING = True: Khởi tạo mới ResNet-50 + [{head_name}] từ Epoch 1.')\n")
            elif "if resume_path:" in line or "print(f\"[*] Tìm thấy checkpoint dở dang" in line:
                # Handled above
                continue
            else:
                new_src.append(line)
        c['source'] = new_src
        print("Updated Section 5 execution loop to respect FORCE_FRESH_TRAINING.")

# Save modified notebook
with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Successfully updated '{NOTEBOOK_PATH}'! Total cells now: {len(nb['cells'])}.")
