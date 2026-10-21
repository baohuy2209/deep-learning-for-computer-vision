import json
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

def audit_full(nb_path):
    print(f"\n{'#'*80}")
    print(f"# COMPREHENSIVE AUDIT REPORT: {nb_path}")
    print(f"{'#'*80}")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb["cells"]
    
    # 1. Check EarlyStopping criterion & patience
    print("\n--- [1] EARLY STOPPING & CHECKPOINT CRITERION (HD_4 Part 4) ---")
    trainer_cells = [i for i, c in enumerate(cells) if "class DRTrainer" in "".join(c.get("source", []))]
    for tc in trainer_cells:
        src = "".join(cells[tc].get("source", []))
        es_qwk = "val_qwk" in src and "patience_early_stopping" in src and "improved" in src
        es_loss = "val_loss" in src and "patience_early_stopping" in src and "không cải thiện" in src
        print(f"Cell {tc} DRTrainer: EarlyStopping on val_qwk={'YES' if es_qwk else 'NO'}, EarlyStopping on val_loss={'YES (BUGGY)' if es_loss else 'NO'}")
        
    cfg_cells = [i for i, c in enumerate(cells) if "PATIENCE_EARLY_STOPPING" in "".join(c.get("source", []))]
    for cc in cfg_cells:
        src = "".join(cells[cc].get("source", []))
        for line in src.splitlines():
            if "PATIENCE_EARLY_STOPPING" in line:
                print(f"Cell {cc} Config: {line.strip()}")

    # 2. Check QWK loss warmup & lambda target (HD_4 Part 4)
    print("\n--- [2] QWK LOSS WARMUP & LAMBDA WEIGHT (HD_4 Part 4) ---")
    for i, c in enumerate(cells):
        src = "".join(c.get("source", []))
        if "qwk_weight" in src.lower() or "target_qwk_weight" in src.lower() or "qwk_warmup_epochs" in src.lower():
            for line in src.splitlines():
                if any(k in line.lower() for k in ["target_qwk_weight", "qwk_warmup_epochs", "qwk_weight ="]):
                    print(f"Cell {i}: {line.strip()}")

    # 3. Check IDRiD XAI Dual-Crop Alignment (HD_4 Part 2)
    print("\n--- [3] IDRiD XAI DUAL-CROP ALIGNMENT (HD_4 Part 2) ---")
    xai_data_cells = [i for i, c in enumerate(cells) if "class IDRiDXAIEvalDataset" in "".join(c.get("source", []))]
    for xc in xai_data_cells:
        src = "".join(cells[xc].get("source", []))
        dual_crop = "fov_mask_raw" in src and "ys, xs = np.where(fov_mask_raw)" in src and "lesion_mask[y0:y1, x0:x1]" in src
        print(f"Cell {xc} IDRiDXAIEvalDataset: Dual-Crop Alignment implemented = {'YES (ALIGNED)' if dual_crop else 'NO (SCALE MISMATCH BUGGY)'}")

    # 4. Check Attention-MIL Saliency Map & Overhauled XAI Metrics (HD_4 Part 2 & 5)
    print("\n--- [4] ATTENTION-MIL SALIENCY & OVERHAULED XAI METRICS (HD_4 Part 2 & 5) ---")
    att_cells = [i for i, c in enumerate(cells) if "generate_attention_mil_saliency_map" in "".join(c.get("source", []))]
    print(f"generate_attention_mil_saliency_map present: {'YES (Cell ' + str(att_cells) + ')' if att_cells else 'NO (MISSING)'}")
    
    qxai_cells = [i for i, c in enumerate(cells) if "def compute_quantitative_xai" in "".join(c.get("source", []))]
    for qc in qxai_cells:
        src = "".join(cells[qc].get("source", []))
        has_rand_base = "Random_Pointing_Chance" in src or "random baseline" in src.lower()
        has_area_matched = "Area_Matched_Recall" in src or "area-matched" in src.lower()
        has_pixel_auc = "Pixel_AUROC" in src or "auroc" in src.lower()
        print(f"Cell {qc} compute_quantitative_xai: Random Pointing Chance={'YES' if has_rand_base else 'NO'}, Area-Matched Recall={'YES' if has_area_matched else 'NO'}, Pixel AUROC={'YES' if has_pixel_auc else 'NO'}")

    # 5. Check Threshold Calibration (HD_4 Part 1 & 3)
    print("\n--- [5] THRESHOLD CALIBRATION EXECUTION IN EVALUATION (HD_4 Part 1 & 3) ---")
    eval_cells = [i for i, c in enumerate(cells) if "calibrate_and_evaluate_screening" in "".join(c.get("source", []))]
    print(f"calibrate_and_evaluate_screening called in evaluation: {'YES (Cell ' + str(eval_cells) + ')' if eval_cells else 'NO (MISSING)'}")

    # 6. Check DataLoader batch sizes and collate_fn (HD_3)
    print("\n--- [6] DATALOADER BATCH SIZES & DROP_LAST & COLLATE_FN (HD_2 & HD_3) ---")
    dl_cells = [i for i, c in enumerate(cells) if "train_loader = DataLoader" in "".join(c.get("source", []))]
    for dc in dl_cells:
        src = "".join(cells[dc].get("source", []))
        drop_last = "drop_last=True" in src
        val_b1 = re.search(r'val_loader\s*=\s*DataLoader\s*\([^)]*batch_size\s*=\s*1', src) or "batch_size=1" in src
        eval_b = re.search(r'eval_batch_size\s*=\s*BATCH_SIZE', src)
        print(f"Cell {dc}: drop_last in train_loader={'YES' if drop_last else 'NO (CRASH RISK ON BATCH_SIZE=1)'}")
        print(f"Cell {dc}: val_loader batch_size=1={'YES (SAFE)' if val_b1 else 'NO'}, eval_batch_size=BATCH_SIZE={'YES (RISK OF OOM)' if eval_b else 'NO'}")

    # 7. Check Precomputed NPZ Cache Function (HD_3)
    print("\n--- [7] OFFLINE PRECOMPUTE CACHE ENGINE (HD_3) ---")
    cache_fn_cells = [i for i, c in enumerate(cells) if "precompute_offline_npz_cache" in "".join(c.get("source", []))]
    print(f"precompute_offline_npz_cache function defined: {'YES (Cell ' + str(cache_fn_cells) + ')' if cache_fn_cells else 'NO'}")

    # 8. Check runtime.unassign() at the end
    print("\n--- [8] RUNTIME.UNASSIGN() AT NOTEBOOK END ---")
    unassign_cells = [i for i, c in enumerate(cells) if "runtime.unassign()" in "".join(c.get("source", []))]
    print(f"runtime.unassign() found at Cell {unassign_cells}")

if __name__ == "__main__":
    audit_full("notebooks/diabetic_retinopathy/pipeline/model/main_model (4).ipynb")
    audit_full("notebooks/diabetic_retinopathy/pipeline/model/main_model (8).ipynb")
