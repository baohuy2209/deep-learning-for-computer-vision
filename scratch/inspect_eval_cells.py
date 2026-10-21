import json

nb = json.load(open('notebooks/diabetic_retinopathy/pipeline/model/main_model (8).ipynb', encoding='utf-8'))

for idx in range(70, len(nb['cells'])):
    cell = nb['cells'][idx]
    ctype = cell['cell_type']
    src = ''.join(cell.get('source', []))
    print(f"=== CELL {idx} ({ctype}) ===")
    print(src[:500])
    if len(src) > 500:
        print("... [TRUNCATED] ...")
        print(src[-300:])
    print("\n" + "="*40 + "\n")
