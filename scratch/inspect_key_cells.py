import json

nb_path = "notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Let's inspect cells 4 to 12
for idx in [4, 6, 7, 8, 10, 11, 13, 14, 15, 17, 18, 20]:
    if idx < len(nb["cells"]):
        cell = nb["cells"][idx]
        print(f"\n{'='*25} FULL CONTENT OF CELL {idx} ({cell['cell_type']}) {'='*25}")
        print("".join(cell.get("source", [])))
