import json

with open('notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation (3).ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx in [6, 11, 12, 18, 26]:
    src = ''.join(nb['cells'][idx]['source'])
    print(f"=== CELL {idx} ===")
    for line in src.split('\n'):
        if 'fallback' in line.lower():
            print('  Line:', line.strip())
