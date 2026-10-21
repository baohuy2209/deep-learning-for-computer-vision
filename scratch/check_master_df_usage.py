import json

with open('notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'master_df' in src:
        print(f"Cell {idx:02d} ({cell['cell_type']}): 'master_df' count = {src.count('master_df')}")
