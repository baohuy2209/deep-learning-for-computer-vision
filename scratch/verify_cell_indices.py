import json
import os

# Load source notebook
nb_src = json.load(open('notebooks/diabetic_retinopathy/pipeline/model/main_model (8).ipynb', encoding='utf-8'))

def get_src(idx):
    return ''.join(nb_src['cells'][idx]['source'])

print(f"Total cells in source: {len(nb_src['cells'])}")
print("Key cells mapped:")
for idx in [4, 8, 10, 12, 13, 16, 18, 20, 23, 25, 34, 35, 36, 40, 42, 44, 47, 49, 51, 53, 55, 57, 59, 61, 64, 65, 68, 74, 77, 78, 83, 85, 86, 87, 88, 89, 90]:
    if idx < len(nb_src['cells']):
        print(f"Cell {idx}: {nb_src['cells'][idx]['cell_type']} | length {len(get_src(idx))}")
