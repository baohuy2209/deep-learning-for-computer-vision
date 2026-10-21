import json

with open("notebooks/diabetic_retinopathy/pipeline/model/main_model (6).ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx in [81, 83, 84]:
    cell = nb['cells'][idx]
    print(f"================== CELL {idx} [{cell['cell_type']}] ==================")
    print("".join(cell.get('source', []))[:1500])
