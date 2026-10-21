import json
import sys

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

nb = json.load(open('notebooks/diabetic_retinopathy/pipeline/model/main_model (8).ipynb', encoding='utf-8'))

for i in range(7, 14):
    cell = nb['cells'][i]
    print(f"=== CELL {i} ({cell['cell_type']}) ===")
    src = ''.join(cell.get('source', []))
    print(src)
    print("\n" + "-"*40)
