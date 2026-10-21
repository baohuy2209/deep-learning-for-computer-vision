import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation (3).ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx in [8, 9, 10, 11]:
    if idx < len(nb['cells']):
        print(f"--- Cell {idx} ---")
        print("".join(nb['cells'][idx].get('source', [])))
