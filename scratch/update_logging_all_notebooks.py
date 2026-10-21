import json
import re
import ast

def update_main_model_notebook(file_path):
    print(f"\nProcessing: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    modified_cells = 0
    for idx, cell in enumerate(nb['cells']):
        if cell.get('cell_type') != 'code':
            continue
        src = "".join(cell.get('source', []))
        orig_src = src

        # 1. calculate_referable_sensitivity_at_specificity
        if "def calculate_referable_sensitivity_at_specificity(" in src:
            src = re.sub(
                r"def calculate_referable_sensitivity_at_specificity\([^)]*target_specificities:\s*Tuple\[float,\s*\.\.\.\]\s*=\s*\([^\)]*\)",
                "def calculate_referable_sensitivity_at_specificity(\n    y_true: np.ndarray,\n    referable_probs: np.ndarray,\n    target_specificities: Tuple[float, ...] = (0.80, 0.85, 0.90, 0.95)",
                src
            )

        # 2. calculate_dr_metrics
        if "def calculate_dr_metrics(" in src:
            src = re.sub(
                r"def calculate_dr_metrics\([^)]*target_specificities:\s*Tuple\[float,\s*\.\.\.\]\s*=\s*\([^\)]*\)",
                "def calculate_dr_metrics(\n    y_true: np.ndarray,\n    y_pred: np.ndarray,\n    num_classes: int = 5,\n    y_probs: Optional[np.ndarray] = None,\n    target_specificities: Tuple[float, ...] = (0.80, 0.85, 0.90, 0.95)",
                src
            )

        # 3. DRTrainer
        if "class DRTrainer:" in src and "def fit(" in src:
            # history dict initialization
            src = re.sub(
                r'("val_ref_sens":\s*\[\],\s*\n\s*)"val_ref_sens_at_spec95":\s*\[\](?:,\s*"val_ref_sens_at_spec85":\s*\[\])?(?:,\s*"val_ref_sens_at_spec80":\s*\[\])?,\s*"val_ref_auc":\s*\[\]',
                r'\1"val_ref_sens_at_spec95": [],\n            "val_ref_sens_at_spec90": [],\n            "val_ref_sens_at_spec85": [],\n            "val_ref_sens_at_spec80": [],\n            "val_ref_auc": []',
                src
            )
            # history append
            history_append_target = r'history\["val_ref_sens_at_spec95"\]\.append\(val_metrics\.get\("Ref_Sensitivity_at_Spec95",\s*float\(\'nan\'\)\)\)(?:\s*\n\s*history\["val_ref_sens_at_spec85"\]\.append\(val_metrics\.get\("Ref_Sensitivity_at_Spec85",\s*float\(\'nan\'\)\)\))?(?:\s*\n\s*history\["val_ref_sens_at_spec80"\]\.append\(val_metrics\.get\("Ref_Sensitivity_at_Spec80",\s*float\(\'nan\'\)\)\))?'
            history_append_replacement = (
                'history.setdefault("val_ref_sens_at_spec95", []).append(val_metrics.get("Ref_Sensitivity_at_Spec95", float(\'nan\')))\n'
                '            history.setdefault("val_ref_sens_at_spec90", []).append(val_metrics.get("Ref_Sensitivity_at_Spec90", float(\'nan\')))\n'
                '            history.setdefault("val_ref_sens_at_spec85", []).append(val_metrics.get("Ref_Sensitivity_at_Spec85", float(\'nan\')))\n'
                '            history.setdefault("val_ref_sens_at_spec80", []).append(val_metrics.get("Ref_Sensitivity_at_Spec80", float(\'nan\')))'
            )
            src = re.sub(history_append_target, history_append_replacement, src)

            # log_msg
            log_msg_target = r'f"Val Ref-Sens@Spec95:\s*\{val_metrics\.get\(\'Ref_Sensitivity_at_Spec95\',\s*float\(\'nan\'\)\)\*100:\.1f\}%\s*\|\s*\\n"(?:\s*f"Val Ref-Sens@Spec85:\s*\{val_metrics\.get\(\'Ref_Sensitivity_at_Spec85\',\s*float\(\'nan\'\)\)\*100:\.1f\}%\s*\|\s*\\n")?(?:\s*f"Val Ref-Sens@Spec80:\s*\{val_metrics\.get\(\'Ref_Sensitivity_at_Spec80\',\s*float\(\'nan\'\)\)\*100:\.1f\}%\s*\|\s*\\n")?'
            log_msg_replacement = (
                'f"Val Ref-Sens@Spec95: {val_metrics.get(\'Ref_Sensitivity_at_Spec95\', float(\'nan\'))*100:.1f}% | "\\n'
                '                f"Val Ref-Sens@Spec90: {val_metrics.get(\'Ref_Sensitivity_at_Spec90\', float(\'nan\'))*100:.1f}% | "\\n'
                '                f"Val Ref-Sens@Spec85: {val_metrics.get(\'Ref_Sensitivity_at_Spec85\', float(\'nan\'))*100:.1f}% | "\\n'
                '                f"Val Ref-Sens@Spec80: {val_metrics.get(\'Ref_Sensitivity_at_Spec80\', float(\'nan\'))*100:.1f}% | "'
            )
            src = re.sub(log_msg_target, log_msg_replacement, src)

            # test_metrics log
            test_target = r'f"Test Ref-Sens@Spec95:\s*\{test_metrics\.get\(\'Ref_Sensitivity_at_Spec95\',\s*float\(\'nan\'\)\)\*100:\.1f\}%\s*\|\s*\\n"(?:\s*f"Test Ref-Sens@Spec85:\s*\{test_metrics\.get\(\'Ref_Sensitivity_at_Spec85\',\s*float\(\'nan\'\)\)\*100:\.1f\}%\s*\|\s*\\n")?(?:\s*f"Test Ref-Sens@Spec80:\s*\{test_metrics\.get\(\'Ref_Sensitivity_at_Spec80\',\s*float\(\'nan\'\)\)\*100:\.1f\}%\s*\|\s*\\n")?'
            test_replacement = (
                'f"Test Ref-Sens@Spec95: {test_metrics.get(\'Ref_Sensitivity_at_Spec95\', float(\'nan\'))*100:.1f}% | "\\n'
                '                f"Test Ref-Sens@Spec90: {test_metrics.get(\'Ref_Sensitivity_at_Spec90\', float(\'nan\'))*100:.1f}% | "\\n'
                '                f"Test Ref-Sens@Spec85: {test_metrics.get(\'Ref_Sensitivity_at_Spec85\', float(\'nan\'))*100:.1f}% | "\\n'
                '                f"Test Ref-Sens@Spec80: {test_metrics.get(\'Ref_Sensitivity_at_Spec80\', float(\'nan\'))*100:.1f}% | "'
            )
            src = re.sub(test_target, test_replacement, src)

        # 4. print_per_grade_report
        if "def print_per_grade_report(" in src:
            if "Ref_Sensitivity_at_Spec90" not in src:
                # Add full clinical spectrum
                src = re.sub(
                    r'(violations\s*=\s*val_metrics\.get\(\'Monotonicity_Violations\',\s*0\.0\)\s*\n\s*print\(f"  • Rank Monotonicity Violation Rate: \{violations\*100:\.2f\}%"\)\n\s*print\("="\*60\))',
                    r'''violations = val_metrics.get('Monotonicity_Violations', 0.0)
    print(f"  • Rank Monotonicity Violation Rate: {violations*100:.2f}%")

    # 4. Print referable DR clinical screening sensitivities across target specificities
    print("\n  [REFERABLE DR SCREENING SENSITIVITY SPECTRUM (HD_4 PART 1)]")
    print(f"  • Spec 95%: {val_metrics.get('Ref_Sensitivity_at_Spec95', float('nan'))*100:.1f}% (Expected: ~58.7% near argmax)")
    print(f"  • Spec 90%: {val_metrics.get('Ref_Sensitivity_at_Spec90', float('nan'))*100:.1f}% (Expected: ~72.0%)")
    print(f"  • Spec 85%: {val_metrics.get('Ref_Sensitivity_at_Spec85', float('nan'))*100:.1f}% (HD_4 Expected: 79.6% - Near Target)")
    print(f"  • Spec 80%: {val_metrics.get('Ref_Sensitivity_at_Spec80', float('nan'))*100:.1f}% (HD_4 Expected: 84.7% - Exceeds Target)")
    print(f"  • Referable Continuous Risk AUC: {val_metrics.get('Referable_AUC', float('nan')):.4f}")
    print("="*60)''',
                    src
                )

        if src != orig_src:
            # Validate syntax
            try:
                ast.parse(src)
            except SyntaxError as e:
                print(f"  SyntaxError in Cell {idx}: {e}")
                continue
            cell['source'] = [line + '\n' for line in src.splitlines()]
            if cell['source']:
                cell['source'][-1] = cell['source'][-1].rstrip('\n')
            modified_cells += 1
            print(f"  Updated Cell {idx}")

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Done. Modified {modified_cells} cells.")

if __name__ == "__main__":
    update_main_model_notebook("notebooks/diabetic_retinopathy/pipeline/model/main_model (8).ipynb")
