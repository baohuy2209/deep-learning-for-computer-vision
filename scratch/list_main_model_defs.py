import json

nb_path = "notebooks/diabetic_retinopathy/pipeline/model/main_model.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print("Cells in main_model.ipynb:")
for i, c in enumerate(nb["cells"]):
    if c.get("cell_type") == "code":
        src = "".join(c.get("source", []))
        defs = [line.strip() for line in src.splitlines() if line.strip().startswith("class ") or line.strip().startswith("def ")]
        if defs:
            print(f"Cell {i:2d}: {', '.join(defs[:4])}")
