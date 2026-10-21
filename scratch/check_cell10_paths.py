import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

src = ''.join(nb['cells'][10]['source'])
for line in src.splitlines():
    if any(k in line for k in ['native_path', 'img_path', 'img_root', 'dataset_key']):
        print(line.strip())
