import json
import re
import ast

def patch_source(src):
    orig = src

    # 1. Target specificities in calculate_referable_sensitivity_at_specificity
    if "def calculate_referable_sensitivity_at_specificity(" in src:
        src = re.sub(
            r"(def calculate_referable_sensitivity_at_specificity\([^)]*target_specificities:\s*Tuple\[float,\s*\.\.\.\]\s*=\s*)\([^\)]*\)",
            r"\1(0.80, 0.85, 0.90, 0.95)",
            src
        )

    # 2. Target specificities in calculate_dr_metrics
    if "def calculate_dr_metrics(" in src:
        src = re.sub(
            r"(def calculate_dr_metrics\([^)]*target_specificities:\s*Tuple\[float,\s*\.\.\.\]\s*=\s*)\([^\)]*\)",
            r"\1(0.80, 0.85, 0.90, 0.95)",
            src
        )

    # 3. DRTrainer or HeadVariantTrainer
    if ("class DRTrainer" in src or "class HeadVariantTrainer" in src) and "def fit(" in src or "def train(" in src:
        # History dict
        # Match anything from "val_ref_sens_at_spec95": [] up to "val_ref_auc": []
        src = re.sub(
            r'"val_ref_sens_at_spec95":\s*\[\].*?"val_ref_auc":\s*\[\]',
            '"val_ref_sens_at_spec95": [],\n            "val_ref_sens_at_spec90": [],\n            "val_ref_sens_at_spec85": [],\n            "val_ref_sens_at_spec80": [],\n            "val_ref_auc": []',
            src,
            flags=re.DOTALL
        )

        # History appends
        # Match history append lines for spec95, spec90, spec85, spec80
        src = re.sub(
            r'history(?:\.setdefault\(|\[)"val_ref_sens_at_spec95".*?(?=history\["val_ref_auc"\]|history\.setdefault\("val_ref_auc"\))',
            'history.setdefault("val_ref_sens_at_spec95", []).append(val_metrics.get("Ref_Sensitivity_at_Spec95", float(\'nan\')))\n'
            '            history.setdefault("val_ref_sens_at_spec90", []).append(val_metrics.get("Ref_Sensitivity_at_Spec90", float(\'nan\')))\n'
            '            history.setdefault("val_ref_sens_at_spec85", []).append(val_metrics.get("Ref_Sensitivity_at_Spec85", float(\'nan\')))\n'
            '            history.setdefault("val_ref_sens_at_spec80", []).append(val_metrics.get("Ref_Sensitivity_at_Spec80", float(\'nan\')))\n            ',
            src,
            flags=re.DOTALL
        )

        # log_msg: replace Val Ref-Sens@Spec... up to Val Ref-AUC
        src = re.sub(
            r'f"Val Ref-Sens@Spec95:[^"]*"\s*\n(?:\s*f"Val Ref-Sens@Spec(?:90|85|80):[^"]*"\s*\n)*',
            'f"Val Ref-Sens@Spec95: {val_metrics.get(\'Ref_Sensitivity_at_Spec95\', float(\'nan\'))*100:.1f}% | "\\n'
            '                f"Val Ref-Sens@Spec90: {val_metrics.get(\'Ref_Sensitivity_at_Spec90\', float(\'nan\'))*100:.1f}% | "\\n'
            '                f"Val Ref-Sens@Spec85: {val_metrics.get(\'Ref_Sensitivity_at_Spec85\', float(\'nan\'))*100:.1f}% | "\\n'
            '                f"Val Ref-Sens@Spec80: {val_metrics.get(\'Ref_Sensitivity_at_Spec80\', float(\'nan\'))*100:.1f}% | "\\n                ',
            src
        )

        # test_metrics log: replace Test Ref-Sens@Spec... up to Test Ref-AUC
        src = re.sub(
            r'f"Test Ref-Sens@Spec95:[^"]*"\s*\n(?:\s*f"Test Ref-Sens@Spec(?:90|85|80):[^"]*"\s*\n)*',
            'f"Test Ref-Sens@Spec95: {test_metrics.get(\'Ref_Sensitivity_at_Spec95\', float(\'nan\'))*100:.1f}% | "\\n'
            '                f"Test Ref-Sens@Spec90: {test_metrics.get(\'Ref_Sensitivity_at_Spec90\', float(\'nan\'))*100:.1f}% | "\\n'
            '                f"Test Ref-Sens@Spec85: {test_metrics.get(\'Ref_Sensitivity_at_Spec85\', float(\'nan\'))*100:.1f}% | "\\n'
            '                f"Test Ref-Sens@Spec80: {test_metrics.get(\'Ref_Sensitivity_at_Spec80\', float(\'nan\'))*100:.1f}% | "\\n                ',
            src
        )

    # 4. print_per_grade_report
    if "def print_per_grade_report(" in src:
        # Check if already has full spectrum
        if "Ref_Sensitivity_at_Spec90" not in src:
            # Replace the end of print_per_grade_report
            old_tail = '''    violations = val_metrics.get('Monotonicity_Violations', 0.0)
    print(f"  • Rank Monotonicity Violation Rate: {violations*100:.2f}%")
    print("="*60)'''
            new_tail = '''    violations = val_metrics.get('Monotonicity_Violations', 0.0)
    print(f"  • Rank Monotonicity Violation Rate: {violations*100:.2f}%")

    # 4. Print referable DR clinical screening sensitivities across target specificities
    print("\\n  [REFERABLE DR SCREENING SENSITIVITY SPECTRUM (HD_4 PART 1)]")
    print(f"  • Spec 95%: {val_metrics.get('Ref_Sensitivity_at_Spec95', float('nan'))*100:.1f}% (Expected: ~58.7% near argmax)")
    print(f"  • Spec 90%: {val_metrics.get('Ref_Sensitivity_at_Spec90', float('nan'))*100:.1f}% (Expected: ~72.0%)")
    print(f"  • Spec 85%: {val_metrics.get('Ref_Sensitivity_at_Spec85', float('nan'))*100:.1f}% (HD_4 Expected: 79.6% - Near Target)")
    print(f"  • Spec 80%: {val_metrics.get('Ref_Sensitivity_at_Spec80', float('nan'))*100:.1f}% (HD_4 Expected: 84.7% - Exceeds Target)")
    print(f"  • Referable Continuous Risk AUC: {val_metrics.get('Referable_AUC', float('nan')):.4f}")
    print("="*60)'''
            if old_tail in src:
                src = src.replace(old_tail, new_tail)

    return src

def process_file(path):
    print(f"Checking {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    changed = False
    for i, cell in enumerate(nb['cells']):
        if cell.get('cell_type') != 'code':
            continue
        src = "".join(cell.get('source', []))
        patched = patch_source(src)
        if patched != src:
            try:
                ast.parse(patched)
            except SyntaxError as e:
                print(f"  SyntaxError in {path} cell {i}: {e}")
                continue
            cell['source'] = [line + '\n' for line in patched.splitlines()]
            if cell['source']:
                cell['source'][-1] = cell['source'][-1].rstrip('\n')
            changed = True
            print(f"  Patched cell {i}")

    if changed:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Saved {path}")
    else:
        print(f"No changes needed in {path}")

if __name__ == '__main__':
    process_file("notebooks/diabetic_retinopathy/pipeline/model/main_model (8).ipynb")
