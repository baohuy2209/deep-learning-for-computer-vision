import json

with open("notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation.ipynb", 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    src = "".join(cell.get('source', []))
    for line in src.splitlines():
        if line.startswith("class "):
            print(f"Cell {idx}: {line}")
