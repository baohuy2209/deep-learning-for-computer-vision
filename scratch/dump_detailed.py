import json

with open('notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation (3).ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('scratch/head_ablation_detailed.txt', 'w', encoding='utf-8') as out:
    for idx, cell in enumerate(nb['cells']):
        out.write(f"\n{'='*70}\n")
        out.write(f"CELL {idx} ({cell['cell_type']})\n")
        out.write(f"{'='*70}\n")
        out.write(''.join(cell['source']))
        out.write("\n")

print("Wrote detailed dump to scratch/head_ablation_detailed.txt")
