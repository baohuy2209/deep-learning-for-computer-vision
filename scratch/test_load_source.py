import json
import os
import sys

# Load original notebook to get exact source of critical functions
with open('notebooks/diabetic_retinopathy/pipeline/model/main_model (8).ipynb', 'r', encoding='utf-8') as f:
    nb_src = json.load(f)

cells_src = nb_src['cells']

def get_src(idx):
    return ''.join(cells_src[idx]['source'])

print(f"Loaded source notebook with {len(cells_src)} cells.")
