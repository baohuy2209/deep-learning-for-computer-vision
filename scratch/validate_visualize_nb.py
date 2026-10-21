import json
import ast

with open('notebooks/diabetic_retinopathy/pipeline/model/visualize_main_model.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")

for i, cell in enumerate(nb['cells']):
    ctype = cell['cell_type']
    src = ''.join(cell.get('source', []))
    first_line = src.strip().split('\n')[0] if src.strip() else '<empty>'
    print(f"Cell {i} [{ctype}]: {first_line[:75]}")
    
    if ctype == 'code':
        try:
            ast.parse(src)
        except SyntaxError as e:
            print(f"  [SYNTAX ERROR in Cell {i}]: {e}")
            lines = src.split('\n')
            err_line = e.lineno - 1
            for j in range(max(0, err_line-3), min(len(lines), err_line+4)):
                marker = ">>> " if j == err_line else "    "
                print(f"{marker}{j+1}: {lines[j]}")
            sys.exit(1)

print("\n[ALL CODE CELLS ARE SYNTACTICALLY VALID PYTHON!]")
