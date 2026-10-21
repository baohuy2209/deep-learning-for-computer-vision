import json
import os
import copy
import re

def update_head_ablation_3(nb_path):
    print(f"Updating {nb_path}...")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # 1. Update Cell 26: calibrate_and_evaluate_screening to add Expected_Sensitivity_HD4
    src_26 = "".join(nb["cells"][26]["source"])
    old_record = '''        calibration_records.append({
            "Target_Specificity": f"{int(round(target_sp * 100))}%",
            "Calibrated_Threshold_tau": tau,
            "Val_Sensitivity": v_sens,
            "Val_Specificity": v_spec,
            "Test_Sensitivity": t_sens,
            "Test_Specificity": t_spec,
            "UK_NSC_Standard": ">=80% Sens & >=95% Spec",
            "UK_NSC_Status": "PASSED" if (t_sens >= 0.80 and t_spec >= 0.95) else "SUB-OPTIMAL"
        })'''

    new_record = '''        expected_sens_map = {95: "58.7%", 90: "72.0%", 85: "79.6% (Near Target)", 80: "84.7% (Exceeds Target)"}
        sp_int = int(round(target_sp * 100))
        calibration_records.append({
            "Target_Specificity": f"{sp_int}%",
            "Expected_Sens_HD4": expected_sens_map.get(sp_int, "N/A"),
            "Calibrated_Threshold_tau": tau,
            "Val_Sensitivity": v_sens,
            "Val_Specificity": v_spec,
            "Test_Sensitivity": t_sens,
            "Test_Specificity": t_spec,
            "UK_NSC_Standard": ">=80% Sens & >=95% Spec",
            "UK_NSC_Status": "PASSED" if (t_sens >= 0.80 and t_spec >= 0.95) else "SUB-OPTIMAL"
        })'''
    if old_record in src_26:
        src_26 = src_26.replace(old_record, new_record)
        nb["cells"][26]["source"] = [line + "\n" for line in src_26.split("\n")][:-1]
        print("  -> Cell 26 updated with Expected_Sens_HD4.")
    else:
        print("  -> Warning: old_record not found in Cell 26.")

    # 2. Update Cell 27: DRHeadTrainer logging for Spec85 and Spec80
    src_27 = "".join(nb["cells"][27]["source"])
    
    # In history init:
    if '"val_ref_sens_at_spec95": [],' in src_27 and '"val_ref_sens_at_spec85": [],' not in src_27:
        src_27 = src_27.replace(
            '"val_ref_sens_at_spec95": [], "val_ref_auc": []',
            '"val_ref_sens_at_spec95": [], "val_ref_sens_at_spec85": [], "val_ref_sens_at_spec80": [], "val_ref_auc": []'
        )
    
    # In history append:
    if 'history["val_ref_sens_at_spec95"].append(val_metrics.get("Ref_Sensitivity_at_Spec95", float(\'nan\')))' in src_27:
        old_append = 'history["val_ref_sens_at_spec95"].append(val_metrics.get("Ref_Sensitivity_at_Spec95", float(\'nan\')))'
        new_append = (
            'history["val_ref_sens_at_spec95"].append(val_metrics.get("Ref_Sensitivity_at_Spec95", float(\'nan\')))\n'
            '            history["val_ref_sens_at_spec85"].append(val_metrics.get("Ref_Sensitivity_at_Spec85", float(\'nan\')))\n'
            '            history["val_ref_sens_at_spec80"].append(val_metrics.get("Ref_Sensitivity_at_Spec80", float(\'nan\')))'
        )
        src_27 = src_27.replace(old_append, new_append)

    # In epoch log_msg:
    old_log_msg = 'f"Val Ref-Sens@Spec95: {val_metrics.get(\'Ref_Sensitivity_at_Spec95\', float(\'nan\'))*100:.1f}% | "'
    new_log_msg = (
        'f"Val Ref-Sens@Spec95: {val_metrics.get(\'Ref_Sensitivity_at_Spec95\', float(\'nan\'))*100:.1f}% | "\n'
        '                f"Val Ref-Sens@Spec85: {val_metrics.get(\'Ref_Sensitivity_at_Spec85\', float(\'nan\'))*100:.1f}% | "\n'
        '                f"Val Ref-Sens@Spec80: {val_metrics.get(\'Ref_Sensitivity_at_Spec80\', float(\'nan\'))*100:.1f}% | "'
    )
    if old_log_msg in src_27:
        src_27 = src_27.replace(old_log_msg, new_log_msg)

    # In test logger.info:
    old_test_log = 'f"Test Ref-Sens@Spec95: {test_metrics.get(\'Ref_Sensitivity_at_Spec95\', float(\'nan\'))*100:.1f}% | "'
    new_test_log = (
        'f"Test Ref-Sens@Spec95: {test_metrics.get(\'Ref_Sensitivity_at_Spec95\', float(\'nan\'))*100:.1f}% | "\n'
        '                f"Test Ref-Sens@Spec85: {test_metrics.get(\'Ref_Sensitivity_at_Spec85\', float(\'nan\'))*100:.1f}% | "\n'
        '                f"Test Ref-Sens@Spec80: {test_metrics.get(\'Ref_Sensitivity_at_Spec80\', float(\'nan\'))*100:.1f}% | "'
    )
    if old_test_log in src_27:
        src_27 = src_27.replace(old_test_log, new_test_log)

    nb["cells"][27]["source"] = [line + "\n" for line in src_27.split("\n")][:-1]
    print("  -> Cell 27 updated with Spec85 & Spec80 logs.")

    # 3. Update Cell 29: print statements and df_heads summary table
    src_29 = "".join(nb["cells"][29]["source"])
    old_final_print = '''    print(f"  • Within-1-Grade Accuracy      : {head_ablation_results[head_name]['Within_1_Grade_Acc']:.1f}%")
    print(f"  • Rank Monotonicity Violations : {head_ablation_results[head_name]['Monotonicity_Violations']:.2f}%")'''

    new_final_print = '''    print(f"  • Within-1-Grade Accuracy      : {head_ablation_results[head_name]['Within_1_Grade_Acc']:.1f}%")
    print(f"  • Referable AUC                : {head_ablation_results[head_name]['Referable_AUC']:.4f}")
    print(f"  • Sens @ Spec 95%              : {head_ablation_results[head_name]['Ref_Sens_Spec95']:.1f}%")
    print(f"  • Sens @ Spec 90%              : {head_ablation_results[head_name]['Ref_Sens_Spec90']:.1f}%")
    print(f"  • Sens @ Spec 85% (Target ~79.6%): {head_ablation_results[head_name]['Ref_Sens_Spec85']:.1f}%")
    print(f"  • Sens @ Spec 80% (Target ~84.7%): {head_ablation_results[head_name]['Ref_Sens_Spec80']:.1f}%")
    print(f"  • Rank Monotonicity Violations : {head_ablation_results[head_name]['Monotonicity_Violations']:.2f}%")'''
    if old_final_print in src_29:
        src_29 = src_29.replace(old_final_print, new_final_print)

    old_df_heads_print = 'print(df_heads[["Head", "Val_QWK", "Test_QWK", "Generalization_Gap", "Within_1_Grade_Acc", "Monotonicity_Violations"]].to_string())'
    new_df_heads_print = '''summary_cols = [
    "Head", "Val_QWK", "Test_QWK", "Generalization_Gap", "Within_1_Grade_Acc",
    "Ref_Sens_Spec95", "Ref_Sens_Spec90", "Ref_Sens_Spec85", "Ref_Sens_Spec80",
    "Referable_AUC", "Monotonicity_Violations"
]
print(df_heads[summary_cols].to_string())'''
    if old_df_heads_print in src_29:
        src_29 = src_29.replace(old_df_heads_print, new_df_heads_print)

    nb["cells"][29]["source"] = [line + "\n" for line in src_29.split("\n")][:-1]
    print("  -> Cell 29 updated with full specificity columns.")

    # 4. Update Cell 32: plot_fig7_referable_roc_curves multi-point annotations
    src_32 = "".join(nb["cells"][32]["source"])
    old_fig7_gate = '''    ax.plot([0, 1], [0, 1], color="#94a3b8", linestyle=":", lw=1.2)
    ax.axvline(0.05, color="#cbd5e1", linestyle="--", lw=1.2)
    ax.text(0.06, 0.40, "95% Screening Specificity Gate (FPR = 5%)", rotation=90, va="center", color="#475569", fontsize=9.5)'''

    new_fig7_gate = '''    ax.plot([0, 1], [0, 1], color="#94a3b8", linestyle=":", lw=1.2)
    ax.axvline(0.05, color="#cbd5e1", linestyle="--", lw=1.2)
    ax.axvline(0.15, color="#fed7aa", linestyle="--", lw=1.2)
    ax.axvline(0.20, color="#bbf7d0", linestyle="--", lw=1.2)
    ax.text(0.055, 0.35, "Spec 95% Gate (FPR 5%)", rotation=90, va="center", color="#475569", fontsize=8.5, fontweight="bold")
    ax.text(0.155, 0.35, "Spec 85% Gate (Sens ~79.6%)", rotation=90, va="center", color="#c2410c", fontsize=8.5, fontweight="bold")
    ax.text(0.205, 0.35, "Spec 80% Gate (Sens ~84.7%)", rotation=90, va="center", color="#15803d", fontsize=8.5, fontweight="bold")'''
    if old_fig7_gate in src_32:
        src_32 = src_32.replace(old_fig7_gate, new_fig7_gate)
        nb["cells"][32]["source"] = [line + "\n" for line in src_32.split("\n")][:-1]
        print("  -> Cell 32 updated with Fig 7 multi-operating point annotations.")

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Successfully saved updated {nb_path}!\n")

def update_main_model_8(nb_path):
    print(f"Updating {nb_path}...")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # 1. Update Cell 64: calibrate_and_evaluate_screening to add Expected_Sens_HD4
    src_64 = "".join(nb["cells"][64]["source"])
    old_record_64 = '''        calibration_records.append({
            "Target_Specificity": f"{int(round(target_sp * 100))}%",
            "Calibrated_Threshold_tau": tau,
            "Val_Sensitivity": v_sens,
            "Val_Specificity": v_spec,
            "Test_Sensitivity": t_sens,
            "Test_Specificity": t_spec,
            "UK_NSC_Standard": ">=80% Sens & >=95% Spec",
            "UK_NSC_Status": "PASSED" if (t_sens >= 0.80 and t_spec >= 0.95) else "SUB-OPTIMAL"
        })'''
    new_record_64 = '''        expected_sens_map = {95: "58.7%", 90: "72.0%", 85: "79.6% (Near Target)", 80: "84.7% (Exceeds Target)"}
        sp_int = int(round(target_sp * 100))
        calibration_records.append({
            "Target_Specificity": f"{sp_int}%",
            "Expected_Sens_HD4": expected_sens_map.get(sp_int, "N/A"),
            "Calibrated_Threshold_tau": tau,
            "Val_Sensitivity": v_sens,
            "Val_Specificity": v_spec,
            "Test_Sensitivity": t_sens,
            "Test_Specificity": t_spec,
            "UK_NSC_Standard": ">=80% Sens & >=95% Spec",
            "UK_NSC_Status": "PASSED" if (t_sens >= 0.80 and t_spec >= 0.95) else "SUB-OPTIMAL"
        })'''
    if old_record_64 in src_64:
        src_64 = src_64.replace(old_record_64, new_record_64)
        nb["cells"][64]["source"] = [line + "\n" for line in src_64.split("\n")][:-1]
        print("  -> Cell 64 updated with Expected_Sens_HD4.")

    # 2. Update Cell 65: compute_clinical_metrics default target_specificities
    src_65 = "".join(nb["cells"][65]["source"])
    old_ts = "target_specificities: Tuple[float, ...] = (0.80, 0.90, 0.95)"
    new_ts = "target_specificities: Tuple[float, ...] = (0.80, 0.85, 0.90, 0.95)"
    if old_ts in src_65:
        src_65 = src_65.replace(old_ts, new_ts)
        nb["cells"][65]["source"] = [line + "\n" for line in src_65.split("\n")][:-1]
        print("  -> Cell 65 updated with target_specificities (0.80, 0.85, 0.90, 0.95).")

    # 3. Update Cell 68: DRTrainer logging
    src_68 = "".join(nb["cells"][68]["source"])
    if '"val_ref_sens_at_spec85": [],' not in src_68:
        src_68 = src_68.replace(
            '"val_ref_sens_at_spec95": [], "val_ref_auc": []',
            '"val_ref_sens_at_spec95": [], "val_ref_sens_at_spec85": [], "val_ref_sens_at_spec80": [], "val_ref_auc": []'
        )
    if 'history["val_ref_sens_at_spec85"]' not in src_68:
        old_app = 'history["val_ref_sens_at_spec95"].append(val_metrics.get("Ref_Sensitivity_at_Spec95", float(\'nan\')))'
        new_app = (
            'history["val_ref_sens_at_spec95"].append(val_metrics.get("Ref_Sensitivity_at_Spec95", float(\'nan\')))\n'
            '            history["val_ref_sens_at_spec85"].append(val_metrics.get("Ref_Sensitivity_at_Spec85", float(\'nan\')))\n'
            '            history["val_ref_sens_at_spec80"].append(val_metrics.get("Ref_Sensitivity_at_Spec80", float(\'nan\')))'
        )
        src_68 = src_68.replace(old_app, new_app)

    old_log_msg_68 = 'f"Val Ref-Sens@Spec95: {val_metrics.get(\'Ref_Sensitivity_at_Spec95\', float(\'nan\'))*100:.1f}% | "'
    new_log_msg_68 = (
        'f"Val Ref-Sens@Spec95: {val_metrics.get(\'Ref_Sensitivity_at_Spec95\', float(\'nan\'))*100:.1f}% | "\n'
        '                f"Val Ref-Sens@Spec85: {val_metrics.get(\'Ref_Sensitivity_at_Spec85\', float(\'nan\'))*100:.1f}% | "\n'
        '                f"Val Ref-Sens@Spec80: {val_metrics.get(\'Ref_Sensitivity_at_Spec80\', float(\'nan\'))*100:.1f}% | "'
    )
    if old_log_msg_68 in src_68:
        src_68 = src_68.replace(old_log_msg_68, new_log_msg_68)

    old_test_log_68 = 'f"Test Ref-Sens@Spec95: {test_metrics.get(\'Ref_Sensitivity_at_Spec95\', float(\'nan\'))*100:.1f}% | "'
    new_test_log_68 = (
        'f"Test Ref-Sens@Spec95: {test_metrics.get(\'Ref_Sensitivity_at_Spec95\', float(\'nan\'))*100:.1f}% | "\n'
        '                f"Test Ref-Sens@Spec85: {test_metrics.get(\'Ref_Sensitivity_at_Spec85\', float(\'nan\'))*100:.1f}% | "\n'
        '                f"Test Ref-Sens@Spec80: {test_metrics.get(\'Ref_Sensitivity_at_Spec80\', float(\'nan\'))*100:.1f}% | "'
    )
    if old_test_log_68 in src_68:
        src_68 = src_68.replace(old_test_log_68, new_test_log_68)

    nb["cells"][68]["source"] = [line + "\n" for line in src_68.split("\n")][:-1]
    print("  -> Cell 68 updated with Spec85 & Spec80 logs.")

    # 4. Update Cell 69: print_per_grade_report
    src_69 = "".join(nb["cells"][69]["source"])
    old_tail_69 = '''    violations = val_metrics.get('Monotonicity_Violations', 0.0)
    print(f"  • Rank Monotonicity Violation Rate: {violations*100:.2f}%")
    print("="*60)'''
    new_tail_69 = '''    violations = val_metrics.get('Monotonicity_Violations', 0.0)
    print(f"  • Rank Monotonicity Violation Rate: {violations*100:.2f}%")

    # 4. Print referable DR clinical screening sensitivities across target specificities
    print("\n  [REFERABLE DR SCREENING SENSITIVITY SPECTRUM (HD_4 PART 1)]")
    print(f"  • Spec 95%: {val_metrics.get('Ref_Sensitivity_at_Spec95', float('nan'))*100:.1f}% (Expected: ~58.7% near argmax)")
    print(f"  • Spec 90%: {val_metrics.get('Ref_Sensitivity_at_Spec90', float('nan'))*100:.1f}% (Expected: ~72.0%)")
    print(f"  • Spec 85%: {val_metrics.get('Ref_Sensitivity_at_Spec85', float('nan'))*100:.1f}% (HD_4 Expected: 79.6% - Near Target)")
    print(f"  • Spec 80%: {val_metrics.get('Ref_Sensitivity_at_Spec80', float('nan'))*100:.1f}% (HD_4 Expected: 84.7% - Exceeds Target)")
    print(f"  • Referable Continuous Risk AUC: {val_metrics.get('Referable_AUC', float('nan')):.4f}")
    print("="*60)'''
    if old_tail_69 in src_69:
        src_69 = src_69.replace(old_tail_69, new_tail_69)
        nb["cells"][69]["source"] = [line + "\n" for line in src_69.split("\n")][:-1]
        print("  -> Cell 69 updated with detailed clinical screening sensitivity report.")

    # 5. Update Cell 88: plot_fig7_roc multi-point annotations
    src_88 = "".join(nb["cells"][88]["source"])
    old_fig7_code = '''    idx_val = np.argmin(np.abs(val_roc["fpr"] - 0.05))
    idx_test = np.argmin(np.abs(test_roc["fpr"] - 0.05))
    ax.scatter([0.05], [val_roc["tpr"][idx_val]], color=C_NAVY, s=80, zorder=5)
    ax.scatter([0.05], [test_roc["tpr"][idx_test]], color=C_ORANGE, s=80, zorder=5)
    ax.axvline(0.05, color="#cbd5e1", linestyle="--", lw=1.2)

    ax.text(0.08, val_roc["tpr"][idx_val] - 0.02, f"Validation Sensitivity: {val_roc['tpr'][idx_val]*100:.1f}%", fontweight="bold", color=C_NAVY)
    ax.text(0.08, test_roc["tpr"][idx_test] - 0.06, f"Test Sensitivity: {test_roc['tpr'][idx_test]*100:.1f}%", fontweight="bold", color=C_ORANGE)'''

    new_fig7_code = '''    idx_val_95 = np.argmin(np.abs(val_roc["fpr"] - 0.05))
    idx_test_95 = np.argmin(np.abs(test_roc["fpr"] - 0.05))
    idx_test_85 = np.argmin(np.abs(test_roc["fpr"] - 0.15))
    idx_test_80 = np.argmin(np.abs(test_roc["fpr"] - 0.20))

    ax.scatter([0.05], [test_roc["tpr"][idx_test_95]], color=C_ORANGE, s=80, zorder=5)
    ax.scatter([0.15], [test_roc["tpr"][idx_test_85]], color="#ea580c", s=80, zorder=5)
    ax.scatter([0.20], [test_roc["tpr"][idx_test_80]], color="#16a34a", s=80, zorder=5)

    ax.axvline(0.05, color="#cbd5e1", linestyle="--", lw=1.0)
    ax.axvline(0.15, color="#fed7aa", linestyle="--", lw=1.0)
    ax.axvline(0.20, color="#bbf7d0", linestyle="--", lw=1.0)

    ax.text(0.06, test_roc["tpr"][idx_test_95] - 0.04, f"Spec 95% -> Test Sens: {test_roc['tpr'][idx_test_95]*100:.1f}%", fontweight="bold", color=C_ORANGE, fontsize=9)
    ax.text(0.16, test_roc["tpr"][idx_test_85] - 0.04, f"Spec 85% -> Test Sens: {test_roc['tpr'][idx_test_85]*100:.1f}% (HD_4 Target 79.6%)", fontweight="bold", color="#ea580c", fontsize=9)
    ax.text(0.21, test_roc["tpr"][idx_test_80] - 0.04, f"Spec 80% -> Test Sens: {test_roc['tpr'][idx_test_80]*100:.1f}% (HD_4 Target 84.7%)", fontweight="bold", color="#16a34a", fontsize=9)'''
    if old_fig7_code in src_88:
        src_88 = src_88.replace(old_fig7_code, new_fig7_code)
        nb["cells"][88]["source"] = [line + "\n" for line in src_88.split("\n")][:-1]
        print("  -> Cell 88 updated with Fig 7 multi-operating point annotations.")

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Successfully saved updated {nb_path}!\n")

def sync_notebooks():
    print("=== SYNCHRONIZING NOTEBOOKS ===")
    ref_head = "notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation (3).ipynb"
    targets_head = [
        "notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation.ipynb",
        "notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation (2).ipynb"
    ]
    with open(ref_head, "r", encoding="utf-8") as f:
        content_head = json.load(f)

    for target in targets_head:
        with open(target, "w", encoding="utf-8") as f:
            json.dump(content_head, f, indent=1)
        print(f"  [SYNCED] {ref_head} -> {target} ({len(content_head['cells'])} cells)")

    ref_main = r"d:\Downloads\main_model (8).ipynb"
    targets_main = [
        "notebooks/diabetic_retinopathy/pipeline/model/main_model.ipynb",
        "notebooks/diabetic_retinopathy/pipeline/model/main_model (4).ipynb"
    ]
    with open(ref_main, "r", encoding="utf-8") as f:
        content_main = json.load(f)

    for target in targets_main:
        with open(target, "w", encoding="utf-8") as f:
            json.dump(content_main, f, indent=1)
        print(f"  [SYNCED] {ref_main} -> {target} ({len(content_main['cells'])} cells)")

if __name__ == "__main__":
    head_ablation_3_path = "notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation (3).ipynb"
    main_model_8_path = r"d:\Downloads\main_model (8).ipynb"

    update_head_ablation_3(head_ablation_3_path)
    update_main_model_8(main_model_8_path)
    sync_notebooks()
