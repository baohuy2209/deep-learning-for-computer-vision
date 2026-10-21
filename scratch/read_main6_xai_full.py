import json

with open("notebooks/diabetic_retinopathy/pipeline/model/main_model (6).ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

with open("scratch/xai_cells.txt", "w", encoding="utf-8") as out:
    for idx in [74, 76, 77, 78]:
        out.write(f"==================== CELL {idx} ====================\n")
        out.write("".join(nb['cells'][idx]['source']) + "\n")
