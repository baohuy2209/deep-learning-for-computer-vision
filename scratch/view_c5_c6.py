import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('notebooks/diabetic_retinopathy/pipeline/merge-dataset/merge-dataset.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

print("=== Cell 5 ===")
print(''.join(nb['cells'][5]['source']))

print("=== Cell 6 ===")
print(''.join(nb['cells'][6]['source']))
