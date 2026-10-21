import json

nb_path = "notebooks/diabetic_retinopathy/pipeline/model/main_model.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]
with open("scratch/extracted_defs_from_main_model.py", "w", encoding="utf-8") as out:
    for idx in [16, 18, 34, 35, 38, 42, 44, 57, 59, 61]:
        out.write(f"\n# {'='*30} CELL {idx} {'='*30}\n")
        out.write("".join(cells[idx].get("source", [])))

print("Extracted definitions from main_model.ipynb")
