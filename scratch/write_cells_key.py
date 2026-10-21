import json

with open('notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('scratch/inspected_cells_key.txt', 'w', encoding='utf-8') as f:
    for i in [10, 16, 17, 18, 20, 21, 23, 25, 27, 28, 29]:
        f.write(f"\n{'='*70}\n=== CELL {i} ===\n{'='*70}\n")
        f.write("".join(nb['cells'][i]['source']))
print("Wrote scratch/inspected_cells_key.txt")
