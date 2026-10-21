import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('notebooks/diabetic_retinopathy/pipeline/merge-dataset/merge-dataset.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for idx in [3, 5, 10, 11]:
    print(f"=== Cell {idx} ===")
    src = ''.join(nb['cells'][idx]['source'])
    print(src[:2000])
