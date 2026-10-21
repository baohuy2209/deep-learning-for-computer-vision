import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

for path in ["notebooks/diabetic_retinopathy/pipeline/model/main_model (4).ipynb", 
             "notebooks/diabetic_retinopathy/pipeline/model/main_model (8).ipynb"]:
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    print(f"\nChecking imports and drive mount in {path}:")
    for i in [3, 4, 5, 6, 7, 8]:
        src = "".join(nb["cells"][i].get("source", []))
        print(f"Cell {i}: {src}")
