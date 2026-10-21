import json

with open("notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation.ipynb", 'r', encoding='utf-8') as f:
    nb = json.load(f)

src = "".join(nb['cells'][27]['source'])
for i, line in enumerate(src.splitlines()):
    if "val_ref_sens" in line or "Ref_Sensitivity_at_Spec" in line or "Val Ref-Sens" in line or "Test Ref-Sens" in line:
        print(f"Line {i}: {line}")
