import json
import os

notebooks = [
    'notebooks/diabetic_retinopathy/pipeline/model/main_model (6).ipynb',
    'notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation (3).ipynb',
    'notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb',
    'notebooks/diabetic_retinopathy/pipeline/merge-dataset/merge-dataset.ipynb',
    'notebooks/diabetic_retinopathy/pipeline/merge-dataset/merge_dataset_lodo.ipynb'
]

for nb_path in notebooks:
    if not os.path.exists(nb_path):
        print(f"NOT FOUND: {nb_path}")
        continue
    print(f"\n==================== {nb_path} ====================")
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    print(f"Total cells: {len(nb.get('cells', []))}")
    for i, c in enumerate(nb.get('cells', [])):
        src = "".join(c.get('source', []))
        if c.get('cell_type') == 'markdown':
            lines = [l.strip() for l in src.split('\n') if l.strip().startswith('#')]
            if lines:
                print(f"Cell {i:02d} [MD]: {lines[:2]}")
        elif c.get('cell_type') == 'code':
            lines = src.split('\n')
            keywords = ('class ', 'def ', 'CONFIG =', 'CONFIG=', 'cfg =', 'cfg=', 'BATCH_SIZE', 'LR', 'EPOCHS', 'optimizer =', 'scheduler =', 'EarlyStopping', 'class NonLocal', 'class CBAM', 'class QuadrantToken', 'class CumulativeLink', 'class ContinuousQWK')
            matches = [l.strip() for l in lines if any(l.strip().startswith(kw) for kw in keywords)]
            if matches:
                print(f"Cell {i:02d} [Code]: {matches[:3]}")
