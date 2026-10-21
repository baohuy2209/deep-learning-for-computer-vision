import sys
import io
import json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
with open('notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i, c in enumerate(nb['cells']):
    ctype = c.get('cell_type')
    src = ''.join(c.get('source', []))
    lines = [l for l in src.split('\n') if l.strip()]
    header = lines[0] if lines else 'EMPTY'
    sec = lines[1] if len(lines) > 1 else ''
    print(f"Cell {i:2d} [{ctype}]: {header[:80]}")
    if sec:
        print(f"         {sec[:80]}")
