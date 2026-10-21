import json, sys
sys.stdout.reconfigure(encoding='utf-8')

nb = json.load(open('notebooks/diabetic_retinopathy/pipeline/model/main_model (6).ipynb', encoding='utf-8'))
for idx, c in enumerate(nb['cells']):
    first_line = ''.join(c['source']).splitlines()[0] if c['source'] else 'EMPTY'
    print(f"Cell {idx:02d} [{c['cell_type']}]: {first_line[:80]}")
