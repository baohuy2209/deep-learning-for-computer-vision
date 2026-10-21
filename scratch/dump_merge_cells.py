import json

nb_path = "notebooks/diabetic_retinopathy/pipeline/merge-dataset/merge-dataset.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

with open("scratch/merge_cells.txt", "w", encoding="utf-8") as out:
    for idx in [10, 11, 12]:
        out.write(f"\n# {'='*30} CELL {idx} {'='*30}\n")
        out.write("".join(nb["cells"][idx].get("source", [])))

print("Dumped merge cells")
