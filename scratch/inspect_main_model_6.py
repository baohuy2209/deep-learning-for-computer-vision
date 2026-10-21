import json
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

with open('notebooks/diabetic_retinopathy/pipeline/model/main_model (6).ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells in main_model (6).ipynb: {len(nb['cells'])}")
for i, c in enumerate(nb['cells']):
    ctype = c.get('cell_type')
    src = ''.join(c.get('source', []))
    lines = [l for l in src.split('\n') if l.strip()]
    header = lines[0] if lines else 'EMPTY'
    print(f"Cell {i:2d} [{ctype:8s}] ({len(lines):3d} lines): {header[:80]}")
