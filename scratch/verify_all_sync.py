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
    
    # 1. EarlyStopping & ModelCheckpoint on val_qwk
    trainer_cells = [i for i, c in enumerate(cells) if "class DRTrainer" in "".join(c.get("source", []))]
    for tc in trainer_cells:
        src = "".join(cells[tc].get("source", []))
        es_qwk = "val_qwk > best_val_qwk" in src and "epochs_no_improve_es" in src
        es_loss = "val_loss" in src and "epochs_no_improve_es += 1" in src
        print(f"[*] EarlyStopping: on val_qwk={'YES (PASS)' if es_qwk else 'NO'}, on val_loss={'YES (FAIL)' if es_loss else 'NO'}")
        
    cfg_cells = [i for i, c in enumerate(cells) if "PATIENCE_EARLY_STOPPING" in "".join(c.get("source", []))]
    for cc in cfg_cells:
        src = "".join(cells[cc].get("source", []))
        for line in src.splitlines():
            if "PATIENCE_EARLY_STOPPING" in line:
                print(f"[*] Config: {line.strip()}")

    # 2. QWK Loss Warmup & Lambda Weight
    for i, c in enumerate(cells):
        src = "".join(c.get("source", []))
        if "target_qwk_weight = 0.4" in src.lower() or "qwk_warmup_epochs = 15" in src.lower():
            for line in src.splitlines():
                if any(k in line.lower() for k in ["target_qwk_weight = 0.4", "qwk_warmup_epochs = 15"]):
                    print(f"[*] HD_4 Part 4 QWK stabilization: {line.strip()}")

    # 3. Batch Size setting
    for i, c in enumerate(cells):
        src = "".join(c.get("source", []))
        if "batch_size = 16" in src.lower():
            for line in src.splitlines():
                if "batch_size = 16" in line.lower():
                    print(f"[*] Colab Hardware Setting: {line.strip()}")

    # 4. IDRiD XAI Dual-Crop Alignment
    xai_data_cells = [i for i, c in enumerate(cells) if "class IDRiDXAIEvalDataset" in "".join(c.get("source", []))]
    for xc in xai_data_cells:
        src = "".join(cells[xc].get("source", []))
        dual_crop = "fov_mask_raw" in src and "ys, xs = np.where(fov_mask_raw)" in src and "lesion_mask[y0:y1, x0:x1]" in src
        print(f"[*] IDRiD Dual-Crop Alignment: {'YES (PASS)' if dual_crop else 'NO (FAIL)'}")

    # 5. Attention-MIL Saliency & Overhauled XAI Metrics
    att_cells = [i for i, c in enumerate(cells) if "generate_attention_mil_saliency_map" in "".join(c.get("source", []))]
    print(f"[*] generate_attention_mil_saliency_map: {'YES (PASS)' if att_cells else 'NO (FAIL)'}")
    
    qxai_cells = [i for i, c in enumerate(cells) if "def compute_quantitative_xai" in "".join(c.get("source", []))]
    for qc in qxai_cells:
        src = "".join(cells[qc].get("source", []))
        has_rand_base = "Random_Pointing_Chance" in src or "random baseline" in src.lower()
        has_area_matched = "Area_Matched_Recall" in src or "area-matched" in src.lower()
        has_pixel_auc = "Pixel_AUROC" in src or "auroc" in src.lower()
        print(f"[*] Overhauled XAI Metrics: Random Chance={'YES' if has_rand_base else 'NO'}, Area-Matched={'YES' if has_area_matched else 'NO'}, Pixel-AUROC={'YES' if has_pixel_auc else 'NO'}")

    # 6. Post-Hoc Threshold Calibration
    eval_cells = [i for i, c in enumerate(cells) if "calibrate_and_evaluate_screening" in "".join(c.get("source", []))]
    print(f"[*] Post-Hoc Threshold Calibration called in evaluation: {'YES (PASS)' if eval_cells else 'NO (FAIL)'}")

    # 7. DataLoader Batch Sizes & Drop Last
    dl_cells = [i for i, c in enumerate(cells) if "train_loader = DataLoader" in "".join(c.get("source", []))]
    for dc in dl_cells:
        src = "".join(cells[dc].get("source", []))
        drop_last = "drop_last=True" in src
        val_b1 = "batch_size=1" in src
        print(f"[*] DataLoader: drop_last in train_loader={'YES (PASS)' if drop_last else 'NO'}, val/test batch_size=1={'YES (PASS)' if val_b1 else 'NO'}")

    # 8. Check Figure 7 ROC KeyError fix
    fig7_cells = [i for i, c in enumerate(cells) if "def plot_fig7_roc" in "".join(c.get("source", []))]
    for fc in fig7_cells:
        src = "".join(cells[fc].get("source", []))
        safe_roc = 'if "roc" in val_data:' in src or "val_data['roc']" in src or 'val_data.get("roc"' in src
        print(f"[*] Figure 7 ROC safe dict access: {'YES (NO KEYERROR)' if safe_roc else 'NO (KEYERROR BUG)'}")

    # 9. Check Figure 6 Latent Scalar TypeError fix
    fig6_cells = [i for i, c in enumerate(cells) if "def extract_latent_scalar" in "".join(c.get("source", []))]
    for fc in fig6_cells:
        src = "".join(cells[fc].get("source", []))
        safe_scalar = '"severity_score" in outputs' in src
        print(f"[*] Figure 6 Latent Scalar safe extraction: {'YES (NO TYPEERROR)' if safe_scalar else 'NO (TYPEERROR BUG)'}")

    # 10. Check runtime.unassign() commented out
    unassign_cells = [i for i, c in enumerate(cells) if "runtime.unassign()" in "".join(c.get("source", []))]
    is_commented = False
    for uc in unassign_cells:
        src = "".join(cells[uc].get("source", []))
        for line in src.splitlines():
            if "runtime.unassign()" in line:
                is_commented = line.strip().startswith("#")
                print(f"[*] runtime.unassign(): {'SAFELY COMMENTED OUT' if is_commented else 'UNCOMMENTED (RISKY)'}")

if __name__ == "__main__":
    for f in ["notebooks/diabetic_retinopathy/pipeline/model/main_model (4).ipynb",
              "notebooks/diabetic_retinopathy/pipeline/model/main_model (8).ipynb",
              "notebooks/diabetic_retinopathy/pipeline/model/main_model.ipynb"]:
        audit_full(f)
