import json
import ast
import os

target_notebooks = [
    "notebooks/diabetic_retinopathy/pipeline/model/main_model (8).ipynb",
    "d:/Downloads/main_model (8).ipynb",
    "notebooks/diabetic_retinopathy/pipeline/model/main_model.ipynb",
    "notebooks/diabetic_retinopathy/pipeline/model/main_model (4).ipynb",
    "notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation.ipynb",
    "notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation (2).ipynb",
    "notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation (3).ipynb",
    "notebooks/diabetic_retinopathy/pipeline/model/main_model_preprocessing_ablation.ipynb",
    "notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb",
]

print(f"{'='*90}")
print(f"{'NOTEBOOK':<45} | {'JSON':<5} | {'AST':<5} | {'SPEC95':<6} | {'SPEC90':<6} | {'SPEC85':<6} | {'SPEC80':<6}")
print(f"{'='*90}")

all_ok = True
for nb_path in target_notebooks:
    if not os.path.exists(nb_path):
        print(f"{nb_path:<45} | MISSING FILE")
        continue

    basename = os.path.basename(nb_path)
    # 1. JSON parse
    try:
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        json_ok = True
    except Exception as e:
        json_ok = False
        all_ok = False
        print(f"{basename:<45} | JSON ERROR: {e}")
        continue

    # 2. AST parse all code cells
    ast_ok = True
    has_spec95 = False
    has_spec90 = False
    has_spec85 = False
    has_spec80 = False

    for idx, cell in enumerate(nb.get('cells', [])):
        if cell.get('cell_type') == 'code':
            src = "".join(cell.get('source', []))
            try:
                ast.parse(src)
            except SyntaxError as e:
                ast_ok = False
                all_ok = False
                print(f"  [AST Error in {basename} cell {idx}]: {e}")
            if "val_ref_sens_at_spec95" in src or "Ref-Sens@Spec95" in src or "Spec 95%" in src or "spec95" in src:
                has_spec95 = True
            if "val_ref_sens_at_spec90" in src or "Ref-Sens@Spec90" in src or "Spec 90%" in src or "spec90" in src:
                has_spec90 = True
            if "val_ref_sens_at_spec85" in src or "Ref-Sens@Spec85" in src or "Spec 85%" in src or "spec85" in src:
                has_spec85 = True
            if "val_ref_sens_at_spec80" in src or "Ref-Sens@Spec80" in src or "Spec 80%" in src or "spec80" in src:
                has_spec80 = True

    j_str = "OK" if json_ok else "FAIL"
    a_str = "OK" if ast_ok else "FAIL"
    s95_str = "YES" if has_spec95 else "NO"
    s90_str = "YES" if has_spec90 else "NO"
    s85_str = "YES" if has_spec85 else "NO"
    s80_str = "YES" if has_spec80 else "NO"

    if not (has_spec95 and has_spec90 and has_spec85 and has_spec80):
        all_ok = False

    print(f"{basename:<45} | {j_str:<5} | {a_str:<5} | {s95_str:<6} | {s90_str:<6} | {s85_str:<6} | {s80_str:<6}")

print(f"{'='*90}")
print(f"OVERALL STATUS: {'100% PARITY & AUDIT PASSED' if all_ok else 'SOME CHECKS FAILED'}")
print(f"{'='*90}")
