import json
from pathlib import Path

nb_path = Path("notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i, cell in enumerate(nb["cells"]):
    cell_type = cell.get("cell_type")
    source = "".join(cell.get("source", []))
    lines = source.split("\n")
    header = lines[0] if lines else ""
    if cell_type == "markdown":
        if header.startswith("#"):
            print(f"Cell {i:2d} [MD]: {header[:80]}")
    else:
        # code cell
        first_meaningful = next((line for line in lines if line.strip() and not line.strip().startswith("#")), "")
        print(f"Cell {i:2d} [CODE]: {lines[0][:50]} | {first_meaningful[:50]}")
