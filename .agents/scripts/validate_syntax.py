import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('notebooks/diabetic_retinopathy/pipeline/model/main_model.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        # ignore lines starting with ! or % or google colab drive mount
        lines = []
        for l in src.splitlines():
            stripped = l.strip()
            if stripped.startswith('!') or stripped.startswith('%'):
                lines.append('# ' + l)
            else:
                lines.append(l)
        code = '\n'.join(lines)
        try:
            compile(code, f'Cell_{i}', 'exec')
        except SyntaxError as e:
            print(f"Syntax error in Cell {i}: {e}")

print("Syntax validation completed for current main_model.ipynb.")
