import json

nb_path = "notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

with open("scratch/model_cells_dump.txt", "w", encoding="utf-8") as out:
    for idx in range(12, 17):
        cell = nb["cells"][idx]
        out.write(f"\n{'='*25} CELL {idx} ({cell['cell_type']}) {'='*25}\n")
        out.write("".join(cell.get("source", [])))

print("Dumped to scratch/model_cells_dump.txt")
