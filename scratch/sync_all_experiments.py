import json
import re
import ast
import os
import pandas as pd
import numpy as np

def update_main_and_ablation_files():
    target_files = [
        "notebooks/diabetic_retinopathy/pipeline/model/main_model (8).ipynb",
        "d:/Downloads/main_model (8).ipynb",
        "notebooks/diabetic_retinopathy/pipeline/model/main_model.ipynb",
        "notebooks/diabetic_retinopathy/pipeline/model/main_model (4).ipynb",
        "notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation.ipynb",
        "notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation (2).ipynb",
        "notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation (3).ipynb",
        "notebooks/diabetic_retinopathy/pipeline/model/main_model_preprocessing_ablation.ipynb",
    ]

    for file_path in target_files:
        if not os.path.exists(file_path):
            print(f"Skipping non-existent: {file_path}")
            continue
        print(f"\nProcessing: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)

        changed = False
        for idx, cell in enumerate(nb['cells']):
            if cell.get('cell_type') != 'code':
                continue
            src = "".join(cell.get('source', []))
            orig_src = src

            # 1. calculate_referable_sensitivity_at_specificity
            if "def calculate_referable_sensitivity_at_specificity(" in src:
                src = re.sub(
                    r"(def calculate_referable_sensitivity_at_specificity\([^)]*target_specificities:\s*Tuple\[float,\s*\.\.\.\]\s*=\s*)\([^\)]*\)",
                    r"\1(0.80, 0.85, 0.90, 0.95)",
                    src
                )

            # 2. calculate_dr_metrics
            if "def calculate_dr_metrics(" in src:
                src = re.sub(
                    r"(def calculate_dr_metrics\([^)]*target_specificities:\s*Tuple\[float,\s*\.\.\.\]\s*=\s*)\([^\)]*\)",
                    r"\1(0.80, 0.85, 0.90, 0.95)",
                    src
                )

            # 3. DRTrainer or HeadVariantTrainer
            if ("class DRTrainer" in src or "class HeadVariantTrainer" in src) and ("def fit(" in src or "def train(" in src):
                # History dict
                src = re.sub(
                    r'"val_ref_sens_at_spec95":\s*\[\].*?"val_ref_auc":\s*\[\]',
                    '"val_ref_sens_at_spec95": [],\n            "val_ref_sens_at_spec90": [],\n            "val_ref_sens_at_spec85": [],\n            "val_ref_sens_at_spec80": [],\n            "val_ref_auc": []',
                    src,
                    flags=re.DOTALL
                )

                # History appends
                src = re.sub(
                    r'history(?:\.setdefault\(|\[)"val_ref_sens_at_spec95".*?(?=history\["val_ref_auc"\]|history\.setdefault\("val_ref_auc"\))',
                    'history.setdefault("val_ref_sens_at_spec95", []).append(val_metrics.get("Ref_Sensitivity_at_Spec95", float(\'nan\')))\n'
                    '            history.setdefault("val_ref_sens_at_spec90", []).append(val_metrics.get("Ref_Sensitivity_at_Spec90", float(\'nan\')))\n'
                    '            history.setdefault("val_ref_sens_at_spec85", []).append(val_metrics.get("Ref_Sensitivity_at_Spec85", float(\'nan\')))\n'
                    '            history.setdefault("val_ref_sens_at_spec80", []).append(val_metrics.get("Ref_Sensitivity_at_Spec80", float(\'nan\')))\n            ',
                    src,
                    flags=re.DOTALL
                )

                # log_msg:
                # Replace Val Ref-Sens@Spec... lines cleanly
                if "Val Ref-Sens@Spec95" in src:
                    src = re.sub(
                        r'f"Val Ref-Sens@Spec95:[^"]*"\s*\|\s*\\n"(?:\s*f"Val Ref-Sens@Spec(?:90|85|80):[^"]*"\s*\|\s*\\n")*',
                        'f"Val Ref-Sens@Spec95: {val_metrics.get(\'Ref_Sensitivity_at_Spec95\', float(\'nan\'))*100:.1f}% | "\\n'
                        '                f"Val Ref-Sens@Spec90: {val_metrics.get(\'Ref_Sensitivity_at_Spec90\', float(\'nan\'))*100:.1f}% | "\\n'
                        '                f"Val Ref-Sens@Spec85: {val_metrics.get(\'Ref_Sensitivity_at_Spec85\', float(\'nan\'))*100:.1f}% | "\\n'
                        '                f"Val Ref-Sens@Spec80: {val_metrics.get(\'Ref_Sensitivity_at_Spec80\', float(\'nan\'))*100:.1f}% | "\\n                ',
                        src
                    )
                elif "Val Ref-Sens:" in src and "class HeadVariantTrainer" in src:
                    # In head ablation, insert after Val Ref-Sens:
                    src = re.sub(
                        r'(f"Val Ref-Sens:\s*\{val_metrics\[\'Referable_Sensitivity\'\]\*100:\.1f\}%\s*\|\s*\\n")',
                        r'\1'
                        r'                f"Val Ref-Sens@Spec95: {val_metrics.get(\'Ref_Sensitivity_at_Spec95\', float(\'nan\'))*100:.1f}% | "\n'
                        r'                f"Val Ref-Sens@Spec90: {val_metrics.get(\'Ref_Sensitivity_at_Spec90\', float(\'nan\'))*100:.1f}% | "\n'
                        r'                f"Val Ref-Sens@Spec85: {val_metrics.get(\'Ref_Sensitivity_at_Spec85\', float(\'nan\'))*100:.1f}% | "\n'
                        r'                f"Val Ref-Sens@Spec80: {val_metrics.get(\'Ref_Sensitivity_at_Spec80\', float(\'nan\'))*100:.1f}% | "\n',
                        src
                    )

                # test_metrics log:
                if "Test Ref-Sens@Spec95" in src:
                    src = re.sub(
                        r'f"Test Ref-Sens@Spec95:[^"]*"\s*\|\s*\\n"(?:\s*f"Test Ref-Sens@Spec(?:90|85|80):[^"]*"\s*\|\s*\\n")*',
                        'f"Test Ref-Sens@Spec95: {test_metrics.get(\'Ref_Sensitivity_at_Spec95\', float(\'nan\'))*100:.1f}% | "\\n'
                        '                f"Test Ref-Sens@Spec90: {test_metrics.get(\'Ref_Sensitivity_at_Spec90\', float(\'nan\'))*100:.1f}% | "\\n'
                        '                f"Test Ref-Sens@Spec85: {test_metrics.get(\'Ref_Sensitivity_at_Spec85\', float(\'nan\'))*100:.1f}% | "\\n'
                        '                f"Test Ref-Sens@Spec80: {test_metrics.get(\'Ref_Sensitivity_at_Spec80\', float(\'nan\'))*100:.1f}% | "\\n                ',
                        src
                    )

                # Clean up any potential double indentation before Val Ref-AUC or Test Ref-AUC
                src = re.sub(r'\n\s+f"Val Ref-AUC:', r'\n                f"Val Ref-AUC:', src)
                src = re.sub(r'\n\s+f"Test Ref-AUC:', r'\n                f"Test Ref-AUC:', src)

            # 4. print_per_grade_report
            if "def print_per_grade_report(" in src:
                if "Ref_Sensitivity_at_Spec90" not in src:
                    old_tail = """    violations = val_metrics.get('Monotonicity_Violations', 0.0)
    print(f"  • Rank Monotonicity Violation Rate: {violations*100:.2f}%")
    print("="*60)"""
                    new_tail = """    violations = val_metrics.get('Monotonicity_Violations', 0.0)
    print(f"  • Rank Monotonicity Violation Rate: {violations*100:.2f}%")

    # 4. Print referable DR clinical screening sensitivities across target specificities
    print("\\n  [REFERABLE DR SCREENING SENSITIVITY SPECTRUM (HD_4 PART 1)]")
    print(f"  • Spec 95%: {val_metrics.get('Ref_Sensitivity_at_Spec95', float('nan'))*100:.1f}% (Expected: ~58.7% near argmax)")
    print(f"  • Spec 90%: {val_metrics.get('Ref_Sensitivity_at_Spec90', float('nan'))*100:.1f}% (Expected: ~72.0%)")
    print(f"  • Spec 85%: {val_metrics.get('Ref_Sensitivity_at_Spec85', float('nan'))*100:.1f}% (HD_4 Expected: 79.6% - Near Target)")
    print(f"  • Spec 80%: {val_metrics.get('Ref_Sensitivity_at_Spec80', float('nan'))*100:.1f}% (HD_4 Expected: 84.7% - Exceeds Target)")
    print(f"  • Referable Continuous Risk AUC: {val_metrics.get('Referable_AUC', float('nan')):.4f}")
    print("="*60)"""
                    if old_tail in src:
                        src = src.replace(old_tail, new_tail)

            if src != orig_src:
                try:
                    ast.parse(src)
                except SyntaxError as e:
                    print(f"  [ERROR] Syntax error in {file_path} cell {idx}: {e}")
                    continue
                cell['source'] = [line + '\n' for line in src.splitlines()]
                if cell['source']:
                    cell['source'][-1] = cell['source'][-1].rstrip('\n')
                changed = True
                print(f"  Patched cell {idx}")

        if changed:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=1, ensure_ascii=False)
            print(f"Saved: {file_path}")
        else:
            print(f"No changes required for: {file_path}")

if __name__ == "__main__":
    update_main_and_ablation_files()
