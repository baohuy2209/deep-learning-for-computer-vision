import json

with open('notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('scratch/nb_cells_detail.txt', 'w', encoding='utf-8') as out:
    for i, cell in enumerate(nb['cells']):
        out.write(f"\n{'='*70}\n")
        out.write(f"CELL {i} ({cell['cell_type'].upper()})\n")
        out.write(f"{'='*70}\n")
        src = "".join(cell.get('source', []))
        out.write(src)
        out.write("\n")

print("Dumped all cells to scratch/nb_cells_detail.txt")
