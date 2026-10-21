import json

with open("notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation.ipynb", 'r', encoding='utf-8') as f:
    nb = json.load(f)

src = "".join(nb['cells'][6].get('source', []))
print("Cell 6 type:", nb['cells'][6]['cell_type'])
print("Cell 6 source:\n", src[:300])
