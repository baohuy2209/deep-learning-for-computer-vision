import json

with open("notebooks/diabetic_retinopathy/pipeline/model/main_model (6).ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx in range(85, len(nb['cells'])):
    cell = nb['cells'][idx]
    print(f"================== CELL {idx} [{cell['cell_type']}] ==================")
    src = "".join(cell.get('source', []))
    for line in src.split("\n")[:10]:
        print("  ", line[:100].encode("ascii", "backslashreplace").decode())
