import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

lodo_nb = json.load(open('notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb', encoding='utf-8'))
merge_nb = json.load(open('notebooks/diabetic_retinopathy/pipeline/merge-dataset/merge-dataset.ipynb', encoding='utf-8'))
nb6 = json.load(open('notebooks/diabetic_retinopathy/pipeline/model/main_model (6).ipynb', encoding='utf-8'))

print(f"main_model_lodo.ipynb cells: {len(lodo_nb['cells'])}")
print(f"merge-dataset.ipynb cells: {len(merge_nb['cells'])}")
print(f"main_model (6).ipynb cells: {len(nb6['cells'])}")

print("\n--- main_model_lodo.ipynb Table of Contents ---")
for idx, c in enumerate(lodo_nb['cells']):
    ctype = c['cell_type']
    src = ''.join(c.get('source', []))
    lines = [l.strip() for l in src.split('\n') if l.strip()]
    first_line = lines[0] if lines else ''
    if ctype == 'markdown':
        if first_line.startswith('#'):
            print(f"Cell {idx:2d} [MD]: {first_line[:80]}")
    else:
        if any(w in first_line.lower() for w in ['def ', 'class ', 'import ', 'for fold', 'lodo', 'dataset', 'fold']):
            print(f"Cell {idx:2d} [CODE]: {first_line[:80]}")
