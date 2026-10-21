import json

with open('notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for idx in [4, 15, 16, 21, 25, 26]:
    src = ''.join(nb['cells'][idx]['source'])
    first_line = src.splitlines()[0] if src.splitlines() else 'EMPTY'
    print(f"Cell {idx:02d}: {len(src)} chars | {first_line}")
