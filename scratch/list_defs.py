import json

nb_path = "notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for idx, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") == "code":
        code = "".join(cell.get("source", []))
        class_defs = [line.strip() for line in code.splitlines() if line.strip().startswith("class ")]
        func_defs = [line.strip() for line in code.splitlines() if line.strip().startswith("def ")]
        if class_defs or func_defs:
            print(f"\nCell {idx}:")
            for c in class_defs:
                print(f"  {c}")
            for f in func_defs:
                print(f"  {f}")
