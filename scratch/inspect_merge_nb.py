import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

merge_nb = json.load(open('notebooks/diabetic_retinopathy/pipeline/merge-dataset/merge-dataset.ipynb', encoding='utf-8'))

for idx, c in enumerate(merge_nb['cells']):
    ctype = c['cell_type']
    src = ''.join(c.get('source', []))
    lines = [l for l in src.split('\n') if l.strip()]
    first_few = '\n'.join(lines[:4])
    print(f"=== MERGE Cell {idx:2d} ({ctype}) ===")
    print(first_few)
    print("\n" + "-"*40)
