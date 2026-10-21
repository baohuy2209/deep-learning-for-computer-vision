import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('notebooks/diabetic_retinopathy/pipeline/model/main_model (6).ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i, c in enumerate(nb['cells']):
    src = "".join(c.get('source', []))
    if c['cell_type'] == 'markdown':
        lines = [l.strip() for l in src.split('\n') if l.strip().startswith('#')]
        if lines:
            print(f"Cell {i:02d} [MD]: {lines}")
    elif c['cell_type'] == 'code':
        lines = [l.strip() for l in src.split('\n') if any(l.strip().startswith(k) for k in [
            'CONFIG =', 'CONFIG=', 'BATCH_SIZE', 'LR =', 'LEARNING_RATE', 'EPOCHS', 'OPTIMIZER', 
            'class ', 'def ', 'PATIENCE', 'TARGET_QWK', 'WARMUP', 'LAMBDA'
        ])]
        if lines:
            print(f"Cell {i:02d} [Code]: {lines[:5]}")
