import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for idx in [6, 7, 8]:
    print(f"=== Cell {idx} ({nb['cells'][idx]['cell_type']}) ===")
    src = ''.join(nb['cells'][idx]['source'])
    print(src)
