import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('notebooks/diabetic_retinopathy/pipeline/merge-dataset/merge-dataset.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

src = ''.join(nb['cells'][11]['source'])
print(src[2000:])
