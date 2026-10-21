import json, ast

with open('notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total notebook cells: {len(nb['cells'])}")
errors = 0
for i, c in enumerate(nb['cells']):
    if c.get('cell_type') == 'code':
        src = ''.join(c['source'])
        try:
            ast.parse(src)
            print(f"Cell {i:02d}: OK ({len(src)} chars)")
        except Exception as e:
            print(f"Cell {i:02d}: ERROR {e}")
            errors += 1

if errors == 0:
    print("ALL CODE CELLS PASSED SYNTAX & AST PARSING WITH ZERO ERRORS!")
else:
    print(f"Encountered {errors} errors.")
