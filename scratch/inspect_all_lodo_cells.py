import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

lodo_nb = json.load(open('notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb', encoding='utf-8'))

for idx, c in enumerate(lodo_nb['cells']):
    ctype = c['cell_type']
    src = ''.join(c.get('source', []))
    lines = [l for l in src.split('\n') if l.strip()]
    first_few = '\n'.join(lines[:5])
    print(f"=== CELL {idx:2d} ({ctype}, {len(src)} chars, {len(lines)} lines) ===")
    print(first_few)
    print("\n" + "-"*50)
