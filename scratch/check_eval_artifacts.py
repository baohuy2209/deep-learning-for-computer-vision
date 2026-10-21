import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

for name, path, cell_idx in [("(4)", "notebooks/diabetic_retinopathy/pipeline/model/main_model (4).ipynb", 85),
                             ("(8)", "notebooks/diabetic_retinopathy/pipeline/model/main_model (8).ipynb", 86)]:
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    src = "".join(nb["cells"][cell_idx].get("source", []))
    print(f"=== {name} Cell {cell_idx} ===")
    for line in src.splitlines()[:35]:
        print(line)
