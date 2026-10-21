import json
import ast
import os

target_files = [
    "notebooks/diabetic_retinopathy/pipeline/model/main_model.ipynb",
    "notebooks/diabetic_retinopathy/pipeline/model/main_model (4).ipynb",
    "d:/Downloads/main_model (8).ipynb",
    "notebooks/diabetic_retinopathy/pipeline/model/main_model (8).ipynb",
]

fixed_code = '''def print_per_grade_report(val_metrics: Dict[str, Any]):
    """
    Prints a formatted clinical per-grade diagnostic evaluation report.
    Directly displays sensitivities for Grade 0 to Grade 4 (verifying Hypotheses H1 & H2),
    Within-1-Grade tolerance accuracy, and rank monotonicity violation rate.

    Args:
        val_metrics: Dictionary containing computed evaluation metrics from evaluate().
    """
    print("\\n" + "="*60)
    print(" DETAILED CLINICAL PER-GRADE RECALL REPORT (VERIFYING H1 & H2)")
    print("="*60)
    # 1. Print individual sensitivities for all 5 Diabetic Retinopathy stages
    for c in range(5):
        recall_val = val_metrics.get(f"Recall_Grade_{c}", 0.0)
        print(f"  • Grade {c} Sensitivity (Recall): {recall_val*100:.2f}%")

    # 2. Print clinical tolerance metric (within ±1 grade)
    within_1 = val_metrics.get('Within_1_Grade_Acc', 0.0)
    print(f"  • Within-1-Grade Diagnostic Tolerance: {within_1*100:.2f}%")

    # 3. Print ordinal probability monotonicity violation rate
    violations = val_metrics.get('Monotonicity_Violations', 0.0)
    print(f"  • Rank Monotonicity Violation Rate: {violations*100:.2f}%")

    # 4. Print referable DR clinical screening sensitivities across target specificities
    print("\\n  [REFERABLE DR SCREENING SENSITIVITY SPECTRUM (HD_4 PART 1)]")
    print(f"  • Spec 95%: {val_metrics.get('Ref_Sensitivity_at_Spec95', float('nan'))*100:.1f}% (Expected: ~58.7% near argmax)")
    print(f"  • Spec 90%: {val_metrics.get('Ref_Sensitivity_at_Spec90', float('nan'))*100:.1f}% (Expected: ~72.0%)")
    print(f"  • Spec 85%: {val_metrics.get('Ref_Sensitivity_at_Spec85', float('nan'))*100:.1f}% (HD_4 Expected: 79.6% - Near Target)")
    print(f"  • Spec 80%: {val_metrics.get('Ref_Sensitivity_at_Spec80', float('nan'))*100:.1f}% (HD_4 Expected: 84.7% - Exceeds Target)")
    print(f"  • Referable Continuous Risk AUC: {val_metrics.get('Referable_AUC', float('nan')):.4f}")
    print("="*60)'''

ast.parse(fixed_code)

for fpath in target_files:
    if not os.path.exists(fpath):
        continue
    with open(fpath, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for i, cell in enumerate(nb['cells']):
        if cell.get('cell_type') != 'code':
            continue
        src = "".join(cell.get('source', []))
        if "def print_per_grade_report(" in src:
            cell['source'] = [line + '\n' for line in fixed_code.splitlines()]
            cell['source'][-1] = cell['source'][-1].rstrip('\n')
            print(f"Fixed Cell {i} in {fpath}")

    with open(fpath, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

print("Done fixing Cell 69.")
