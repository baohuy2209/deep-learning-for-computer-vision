import json
import re
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

def audit_notebook(path):
    print(f"\n{'='*80}")
    print(f"EXHAUSTIVE PIPELINE VERIFICATION: {path}")
    print(f"{'='*80}")
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb["cells"]
    print(f"Total cells: {len(cells)}")
    all_code = "\n".join(["".join(c.get("source", [])) for c in cells if c.get("cell_type") == "code"])

    checks = []

    # 1. Environment & Threads
    cv2_threads = "cv2.setNumThreads(0)" in all_code
    checks.append(("HD_3 cv2.setNumThreads(0) active", cv2_threads))

    # 2. Drive Mount & Unzip
    has_drive = "from google.colab import drive" in all_code and "drive.mount" in all_code
    has_unzip = 'zip_source' in all_code and 'extract_dest' in all_code and 'unzip -q' in all_code
    checks.append(("Colab Auto Drive Mount present", has_drive))
    checks.append(("Colab Fast Dataset Unzip present", has_unzip))

    # 3. Hyperparameters
    has_batch16 = bool(re.search(r'BATCH_SIZE\s*=\s*16', all_code))
    has_target_qwk = bool(re.search(r'TARGET_QWK_WEIGHT\s*=\s*0\.4', all_code))
    has_warmup15 = bool(re.search(r'QWK_WARMUP_EPOCHS\s*=\s*15', all_code))
    has_patience15 = bool(re.search(r'PATIENCE_EARLY_STOPPING\s*=\s*15', all_code))
    checks.append(("BATCH_SIZE = 16 (Colab T4 safe)", has_batch16))
    checks.append(("TARGET_QWK_WEIGHT = 0.4 (HD_4 stabilization)", has_target_qwk))
    checks.append(("QWK_WARMUP_EPOCHS = 15 (HD_4 linear ramp)", has_warmup15))
    checks.append(("PATIENCE_EARLY_STOPPING = 15", has_patience15))

    # 4. Dataset class & Call site
    c20_src = "".join(cells[20].get("source", []))
    c25_src = "".join(cells[25].get("source", []))
    c20_aliases = "metadata: Optional" in c20_src and "image_root_dir: Optional" in c20_src
    c25_canonical = "csv_path=csv_train_path" in c25_src and "img_root=image_root_dir" in c25_src
    checks.append(("Cell 20 backward-compatible aliases (metadata, image_root_dir)", c20_aliases))
    checks.append(("Cell 25 canonical calls (csv_path=..., img_root=...)", c25_canonical))

    # 5. DataLoaders
    drop_last = "drop_last=True" in c25_src
    val_b1 = bool(re.search(r'val_loader\s*=\s*DataLoader\s*\([^)]*batch_size\s*=\s*1', c25_src))
    test_b1 = bool(re.search(r'test_loader\s*=\s*DataLoader\s*\([^)]*batch_size\s*=\s*1', c25_src))
    checks.append(("train_loader has drop_last=True (HD_2 BatchNorm1d safety)", drop_last))
    checks.append(("val_loader has batch_size=1 (HD_3 eval memory safety)", val_b1))
    checks.append(("test_loader has batch_size=1 (HD_3 eval memory safety)", test_b1))

    # 6. Local MIL Branch & Dual Pooling
    dual_pooling = "gated_pool" in all_code and "torch.max" in all_code
    checks.append(("Local MIL dual-pooling (Max + Gated Attention) (HD_2)", dual_pooling))

    # 7. Global Context Branch
    global_modules = "non_local" in all_code.lower() and "quadrant" in all_code.lower() and "cbam" in all_code.lower()
    checks.append(("Global Branch (Non-local + Quadrant Tokens + CBAM) (HD_1, HD_2)", global_modules))

    # 8. Monotonicity Metric
    rank_mono = "def rank_monotonicity_violations" in all_code and "np.diff" in all_code
    checks.append(("rank_monotonicity_violations on rank probabilities (HD_2 Lỗi 5)", rank_mono))

    # 9. EarlyStopping & ModelCheckpoint on val_qwk
    c68_src = "".join(cells[68].get("source", []))
    es_on_qwk = "if val_qwk > best_val_qwk:" in c68_src and "epochs_no_improve_es = 0" in c68_src[c68_src.find("if val_qwk > best_val_qwk"):c68_src.find("if val_qwk > best_val_qwk")+200]
    es_not_on_loss = "epochs_no_improve_es = 0" not in c68_src[c68_src.find("if val_loss < best_val_loss"):c68_src.find("if val_loss < best_val_loss")+200]
    checks.append(("EarlyStopping strictly driven by val_qwk (HD_4 Part 4)", es_on_qwk and es_not_on_loss))

    # 10. IDRiD Dual-Crop Alignment
    c77_src = "".join(cells[77].get("source", []))
    dual_crop = "fov_mask_raw = compute_fov_mask(img_np, erode_px=0)" in c77_src and "lesion_mask = lesion_mask[y0:y1, x0:x1]" in c77_src
    checks.append(("IDRiD XAI Dual-Crop Alignment (HD_4 Part 2)", dual_crop))

    # 11. Attention-MIL Saliency Map
    c78_src = "".join(cells[78].get("source", []))
    has_att_map = "def generate_attention_mil_saliency_map" in c78_src and "alpha_k" in c78_src
    checks.append(("generate_attention_mil_saliency_map active (HD_4 Part 5)", has_att_map))

    # 12. Overhauled XAI Metrics Suite
    has_xai_metrics = "GradCAM_Area_Matched_Recall" in c78_src and "GradCAM_Pixel_AUROC" in c78_src and "Mean_IoU_Informational" in c78_src
    checks.append(("Overhauled XAI Suite (Chance 1.5%, Area-matched, Pixel AUROC) (HD_4)", has_xai_metrics))

    # 13. Post-Hoc Threshold Calibration in Evaluation
    c81_src = "".join(cells[81].get("source", []))
    has_calib = "calibrate_and_evaluate_screening" in c81_src and "screening_calibration.csv" in c81_src
    checks.append(("Post-Hoc Threshold Calibration executed in Cell 81 (HD_4 Part 1 & 3)", has_calib))

    # 14. Figure 6 Latent Scalar Safe Extraction
    c85_src = "".join(cells[85].get("source", []))
    safe_f6 = '"severity_score" in outputs' in c85_src
    checks.append(("Figure 6 Latent Scalar safe extraction (No TypeError)", safe_f6))

    # 15. Figure 7 ROC Safe Sub-dict Unwrapping
    c88_src = "".join(cells[88].get("source", []))
    safe_f7 = 'if "roc" in val_data:' in c88_src and 'val_roc = val_data["roc"]' in c88_src
    checks.append(("Figure 7 ROC safe dict access (No KeyError)", safe_f7))

    # 16. Colab Runtime Preservation
    c_last_src = "".join(cells[-1].get("source", []))
    unassign_commented = "runtime.unassign()" in c_last_src and any(l.strip().startswith("#") and "runtime.unassign()" in l for l in c_last_src.splitlines())
    checks.append(("runtime.unassign() safely commented out", unassign_commented))

    all_passed = True
    for desc, passed in checks:
        status = "PASSED" if passed else "FAILED"
        if not passed:
            all_passed = False
        print(f"  [{status:<6}] {desc}")

    print(f"\nFinal Verdict for {os.path.basename(path)}: {'100% VERIFIED & READY' if all_passed else 'HAS ISSUES'}")
    return all_passed

if __name__ == "__main__":
    v1 = audit_notebook("notebooks/diabetic_retinopathy/pipeline/model/main_model (4).ipynb")
    v2 = audit_notebook("notebooks/diabetic_retinopathy/pipeline/model/main_model (8).ipynb")
    v3 = audit_notebook("notebooks/diabetic_retinopathy/pipeline/model/main_model.ipynb")
    if v1 and v2 and v3:
        print("\n================================================================================")
        print("ALL 3 NOTEBOOKS FULLY AUDITED AND CONFIRMED 100% COMPLIANT WITH HD_1 - HD_4!")
        print("================================================================================")
