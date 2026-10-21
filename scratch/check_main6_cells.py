import json

with open("notebooks/diabetic_retinopathy/pipeline/model/main_model (6).ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells in main_model (6).ipynb: {len(nb['cells'])}")
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    first_line = src.split("\n")[0] if src else ""
    for kw in ["xai", "cam", "idrid", "mask", "figure", "plot", "visualiz", "iou", "pointing", "5.1", "5.2", "6.1", "6.2"]:
        if kw in src.lower() or kw in first_line.lower():
            print(f"Cell {i} [{c['cell_type']}]: {first_line[:80]}")
            break
