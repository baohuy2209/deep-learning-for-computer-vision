import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('notebooks/diabetic_retinopathy/pipeline/merge-dataset/merge-dataset.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])
    first_line = src.splitlines()[0] if src.splitlines() else 'EMPTY'
    print(f"Cell {i:02d} [{c.get('cell_type')}]: {first_line[:80]}")
