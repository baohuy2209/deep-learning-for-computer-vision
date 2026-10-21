import json

nb_path = "notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cell11 = "".join(nb["cells"][11].get("source", []))
with open("scratch/cell11_dump.txt", "w", encoding="utf-8") as out:
    out.write(cell11)

print("Cell 11 dumped")
