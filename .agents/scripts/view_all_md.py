import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('notebooks/diabetic_retinopathy/pipeline/model/main_model.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        src = ''.join(cell['source'])
        print(f"=== Markdown Cell {i:02d} ===")
        print(src)
        print("\n" + "-"*40 + "\n")
