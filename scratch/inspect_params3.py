import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('notebooks/diabetic_retinopathy/pipeline/model/main_model (6).ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx in [9, 10]:
    if idx < len(nb['cells']):
        print(f"--- Cell {idx} ---")
        print("".join(nb['cells'][idx].get('source', [])))
