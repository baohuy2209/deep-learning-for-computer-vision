import json

nb = json.load(open('notebooks/diabetic_retinopathy/pipeline/model/main_model (8).ipynb', encoding='utf-8'))

for idx in [70, 71, 74, 75, 76, 77, 78, 80, 81, 85, 90]:
    cell = nb['cells'][idx]
    ctype = cell['cell_type']
    src = ''.join(cell.get('source', []))
    print(f"=== CELL {idx} ({ctype}) ===")
    print(src)
    print("\n" + "="*50 + "\n")
