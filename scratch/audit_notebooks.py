import json
import sys

def audit_nb(path):
    print(f"\n{'='*30} {path} {'='*30}")
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    print(f"Total cells: {len(nb['cells'])}")
    for i, cell in enumerate(nb['cells']):
        lines = [line.strip() for line in cell.get('source', []) if line.strip()]
        first = lines[0] if lines else '(empty)'
        src = "".join(cell.get('source', []))
        # Print if cell mentions training, optimizer, scheduler, loss, eval, config
        keywords = ['DRTrainer', 'DRHeadTrainer', 'train(', 'optimizer', 'scheduler', 'warmup', 'epoch', 'CONFIG', 'Config', 'LODO', 'tau', 'threshold', 'bipartite']
        matched = [k for k in keywords if k.lower() in src.lower()]
        if cell['cell_type'] == 'markdown' and (first.startswith('#') or 'train' in first.lower()):
            print(f"Cell {i:2d} [MD]: {first[:80]}")
        elif any(k in matched for k in ['drtrainer', 'drheadtrainer', 'train(', 'optimizer', 'bipartite']):
            print(f"Cell {i:2d} [CODE]: {first[:80]}  (matches: {matched[:3]})")

if __name__ == '__main__':
    audit_nb('notebooks/diabetic_retinopathy/pipeline/model/main_model_lodo.ipynb')
    audit_nb('notebooks/diabetic_retinopathy/pipeline/model/main_model (6).ipynb')
    audit_nb('notebooks/diabetic_retinopathy/pipeline/model/main_model_head_ablation (3).ipynb')
