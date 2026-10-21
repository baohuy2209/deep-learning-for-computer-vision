import json
import ast

with open('notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation (3).ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
error_count = 0

for idx, cell in enumerate(nb['cells']):
    ctype = cell['cell_type']
    source = ''.join(cell['source'])
    if ctype == 'code':
        # Filter out IPython magic commands
        code_lines = [l for l in source.split('\n') if not l.strip().startswith('!') and not l.strip().startswith('%')]
        clean_code = '\n'.join(code_lines)
        try:
            ast.parse(clean_code)
        except Exception as e:
            print(f"Syntax error in Cell {idx}: {e}")
            error_count += 1

if error_count == 0:
    print("SUCCESS: 100% of code cells in 'main_model_head_ablation (3).ipynb' passed ast.parse validation with 0 errors!")
else:
    print(f"FAILED: Found {error_count} syntax errors!")
