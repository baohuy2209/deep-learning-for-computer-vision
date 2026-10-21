import json

with open("notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation.ipynb", 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    src = "".join(cell.get('source', []))
    if "HeadVariantTrainer" in src:
        print(f"Cell {idx}: length {len(src)}")
        print("contains def fit:", "def fit(" in src)
        lines = src.splitlines()
        for i, line in enumerate(lines):
            if "val_ref_sens" in line or "Ref_Sensitivity_at_Spec" in line or "Val Ref-Sens" in line or "Test Ref-Sens" in line:
                print(f"  Line {i}: {line}")
