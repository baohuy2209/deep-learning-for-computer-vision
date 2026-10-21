import json

nb = json.load(open('notebooks/diabetic_retinopathy/pipeline/model/main_model (8).ipynb', encoding='utf-8'))

def get_source(idx):
    return ''.join(nb['cells'][idx]['source'])

# Let's inspect cell 20, 23, 64, 65
print("Cell 64:", get_source(64)[:300])
print("Cell 65:", get_source(65)[:300])
print("Cell 83:", get_source(83)[:300])
print("Cell 85:", get_source(85)[:300])
