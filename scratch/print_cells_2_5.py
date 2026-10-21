import json

with open('notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i in [2, 3, 4, 5]:
    print(f"=== CELL {i} ===")
    print("".join(nb['cells'][i]['source']))
