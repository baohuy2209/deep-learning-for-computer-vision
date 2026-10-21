import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

for name, path in [("(4)", "notebooks/diabetic_retinopathy/pipeline/model/main_model (4).ipynb"),
                   ("(8)", "notebooks/diabetic_retinopathy/pipeline/model/main_model (8).ipynb")]:
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    print(f"\n=================== Cells 82-end in {name} ===================")
    for i in range(82, len(nb["cells"])):
        src = "".join(nb["cells"][i].get("source", []))
        first_line = src.strip().splitlines()[0] if src.strip() else "[EMPTY]"
        print(f"Cell {i} [{nb['cells'][i]['cell_type']}]: {first_line[:70]}")
