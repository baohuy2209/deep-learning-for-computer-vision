import json

with open('notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('scratch/inspected_cells_7_10.txt', 'w', encoding='utf-8') as f:
    for i in [7, 8, 9, 10]:
        f.write(f"\n{'='*70}\n=== CELL {i} ===\n{'='*70}\n")
        f.write("".join(nb['cells'][i]['source']))
print("Wrote scratch/inspected_cells_7_10.txt")
