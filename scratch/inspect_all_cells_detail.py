import json, sys
sys.stdout.reconfigure(encoding='utf-8')

nb = json.load(open('notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb', encoding='utf-8'))
for idx, c in enumerate(nb['cells']):
    first = ''.join(c['source']).splitlines()[0] if c['source'] else 'EMPTY'
    second = ''.join(c['source']).splitlines()[1] if len(c['source']) > 1 else ''
    print(f"Cell {idx:02d} [{c['cell_type']:8s}]: {first[:60]} | {second[:40]}")
