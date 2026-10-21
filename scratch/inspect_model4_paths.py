import json
import sys

with open('notebooks/diabetic_retinopathy/pipeline/model/main_model (4).ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('scratch/model4_paths.txt', 'w', encoding='utf-8') as out:
    for idx in [7, 8, 9, 10]:
        cell = nb['cells'][idx]
        out.write(f"*** Cell {idx} ({cell['cell_type']}) ***\n")
        out.write(''.join(cell['source']))
        out.write("\n\n")

print("Wrote to scratch/model4_paths.txt")
