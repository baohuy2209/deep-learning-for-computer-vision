import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

lodo_nb = json.load(open('notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb', encoding='utf-8'))
for i in [12, 13, 14, 15, 16]:
    print(f"=== CELL {i} ({lodo_nb['cells'][i]['cell_type']}) ===")
    print(''.join(lodo_nb['cells'][i]['source']))
    print("\n" + "="*50 + "\n")
