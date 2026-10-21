import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open('notebooks/diabetic_retinopathy/pipeline/oracle/oracle_aptos.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i in range(min(12, len(nb['cells']))):
    c = nb['cells'][i]
    src = "".join(c.get('source', []))
    if c['cell_type'] == 'markdown':
        print(f"Cell {i} [MD]: {src.split('\n')[0]}")
    else:
        for line in src.split('\n'):
            if any(k in line for k in ['def ', 'class ', 'EPOCHS', 'BATCH_SIZE', 'LEARNING_RATE', 'Oracle']):
                print(f"Cell {i} [Code]: {line[:80]}")
